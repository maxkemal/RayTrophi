"""Non-build stage/SBT/redispatch contracts. No driver/GLSL execution claimed."""
import itertools
from pathlib import Path
import re
import struct
import unittest
import xml.etree.ElementTree as ET

from volume_shader_source import read_volume_shader, EXTRACTED

ROOT = Path(__file__).resolve().parents[2]
SH = ROOT/'RayTrophiStudio/source/shaders'
SRC = ROOT/'RayTrophiStudio/source'


def stage_source(stage):
    # Only the six new stage guards are evaluated; existing includes stay intact.
    source = read_volume_shader(SH/'volume_closesthit.rchit')
    enabled, stack, output = True, [], []
    for line in source.splitlines():
        guard = re.fullmatch(r'#if (!?)defined\((RT_VOLUME_\w+_STAGE)\)', line)
        if guard:
            stack.append(enabled)
            matches = guard[2] == stage
            enabled = enabled and (not matches if guard[1] else matches)
        elif line == '#endif':
            enabled = stack.pop()
        elif enabled:
            output.append(line)
    if stack:
        raise AssertionError('Unbalanced stage guard')
    return '\n'.join(output)


def shade_body(source):
    return source.split('void shadeVolumeHit() {', 1)[1].split('void main()', 1)[0]


class VolumeStageSplit(unittest.TestCase):
    def test_stage_entry_call_trees_are_disjoint(self):
        fog = shade_body(stage_source('RT_VOLUME_FOG_STAGE'))
        surface_source = stage_source('RT_VOLUME_SURFACE_STAGE')
        surface = shade_body(surface_source)
        legacy = shade_body(stage_source(None))
        for token in ('scatterPrincipled(', 'ISO_THRESH', 'foam_inscatter'):
            self.assertNotIn(token, fog)
            self.assertIn(token, surface)
            self.assertIn(token, legacy)
        for token in ('shadeFogMixture(', 'volumeRandomWalk(', 'volumeLightCacheFetch('):
            self.assertIn(token, fog)
            self.assertNotIn(token, surface)
            self.assertIn(token, legacy)
        self.assertNotIn('#include "volume_mixture.glsl"', surface_source)
        self.assertNotIn('void volumeRandomWalk(', surface_source)
        self.assertNotIn('VOLUME_FOREGROUND_GAS_DISPATCH;', legacy)

    def test_foreground_selection_precedes_signal_and_cloud_is_not_duplicated(self):
        surface = shade_body(stage_source('RT_VOLUME_SURFACE_STAGE'))
        self.assertLess(surface.index('selectForegroundGas('),
                        surface.index('payload.volumeTraversalMask = VOLUME_FOREGROUND_GAS_DISPATCH'))
        self.assertLess(surface.index('VOLUME_FOREGROUND_GAS_DISPATCH'), surface.index('ISO_THRESH'))
        root = (SH/'volume_closesthit.rchit').read_text(encoding='utf-8')
        main = root.split('void main()', 1)[1]
        self.assertLess(main.index('VOLUME_FOREGROUND_GAS_DISPATCH'),
                        main.index('cloudApplyVolumeSegment();'))
        raygen = (SH/'raygen.rgen').read_text(encoding='utf-8')
        self.assertLess(raygen.index('volumeFinishForegroundDispatch(rayOrigin'),
                        raygen.index('seed = payload.seed;'))
        helper = (SH/'volume_foreground_dispatch.glsl').read_text(encoding='utf-8')
        self.assertEqual(helper.count('traceRayEXT('), 1)
        self.assertIn('0x02u, 0, 0, 0,', helper)
        self.assertIn('origin, nearT, direction, farT, 0)', helper)
        code = re.sub(r'//[^\n]*', '', helper)
        for forbidden in ('rnd(', '++', 'cloudApply', 'payload.seed =', 'origin +'):
            self.assertNotIn(forbidden, code)

    def test_marker_cannot_alias_solid_distance_encoding(self):
        marker = 0xFFFFFFFE
        for distance in (.001, .1, 1, 1000, 1e6, 1e20, 3e38):
            bound = struct.unpack('<I', struct.pack('<f', distance))[0] & 0xFFFFFF00
            bound = max(bound - 0x100, 0)
            for mask in (0, 2, 8, 0xF5, 0xFD, 0xFF):
                self.assertNotEqual(marker, bound | mask)

    def test_sbt_appends_surface_and_preserves_existing_offsets(self):
        policy = (SRC/'include/Backend/VulkanVolumeStageSplit.h').read_text(encoding='utf-8')
        self.assertIn('2u + (hair ? 1u : 0u) + (sphere ? 1u : 0u)', policy)
        for hair, sphere in itertools.product((False, True), repeat=2):
            groups = ['triangle', 'fog']
            if hair:
                groups.append('hair')
            if sphere:
                groups.append('sphere')
            original = list(groups)
            groups.append('surface')
            self.assertEqual(groups.index('surface'), 2 + int(hair) + int(sphere))
            for name in original:
                self.assertEqual(groups.index(name), original.index(name))
        pipeline = (SRC/'src/Backend/VulkanRTPipelineBuild.cpp').read_text(encoding='utf-8')
        groups = pipeline.split('job.triHitGroupIdx =', 1)[1].split('job.photonGroupIdx =', 1)[0]
        self.assertLess(groups.index('if (job.hasSphere)'), groups.index('if (job.hasSurfaceVolume)'))
        self.assertIn('mixVec(s.volumeSurfaceClosestHit);', pipeline)
        self.assertIn('shaders.volumeSurfaceClosestHit.clear();', pipeline)
        self.assertIn('(job.hasSurfaceVolume ? 1u : 0u)', pipeline)

    def test_cpu_gpu_routes_and_install_refresh_use_shared_policy(self):
        backend = (SRC/'src/Backend/VulkanBackend.cpp').read_text(encoding='utf-8')
        for call in ('getVolumeRoutedSbtOffset(src.mask, src.sbtRecordOffset)',
                     'getVolumeRoutedSbtOffset(mask, vi.sbtRecordOffset)',
                     'getVolumeRoutedSbtOffset(desiredMask, vi.sbtRecordOffset)'):
            self.assertIn(call, backend)
        install = backend.split('void VulkanBackendAdapter::onRTPipelineInstalled()', 1)[1]
        self.assertIn('refreshVolumeStageRouting();', install.split('kRasterWhile', 1)[0])
        module = (SRC/'src/Backend/VulkanVolumeStageSplit.cpp').read_text(encoding='utf-8')
        self.assertIn('for (auto& source : m_pendingGpuInstanceSources)', module)
        self.assertIn('m_device->updateTLAS(merged)', module)
        self.assertIn('refreshVolumeStageMask(index)', module)
        volume_upload = (SRC/'src/Backend/VulkanBackend_Volumes.cpp').read_text(encoding='utf-8')
        self.assertIn('refreshVolumeStageMask(i)', volume_upload)

    def test_pair_opt_in_and_build_registration(self):
        module = (SRC/'src/Backend/VulkanVolumeStageSplit.cpp').read_text(encoding='utf-8')
        self.assertIn('std::getenv("RT_VK_SPLIT_VOLUME")', module)
        self.assertIn('std::strcmp(value, "1") == 0', module)
        self.assertLess(module.index('if (!requested)'), module.index('auto fogStage'))
        self.assertLess(module.index('fogStage.empty() || surfaceStage.empty()'),
                        module.index('fog = std::move(fogStage)'))
        self.assertLess(module.index('!splitManifestMatches('),
                        module.index('fog = std::move(fogStage)'))
        self.assertIn('volume-split-v1', module)
        self.assertIn('const std::vector<std::uint32_t>& raygen', module)
        driver = (ROOT/'RayTrophiStudio/compile_shaders.ps1').read_text(encoding='utf-8')
        for token in ('-DRT_VOLUME_FOG_STAGE=1', '-DRT_VOLUME_SURFACE_STAGE=1',
                      "'raygen.spv', 'volume_fog_closesthit.spv', 'volume_surface_closesthit.spv'"):
            self.assertIn(token, driver)
        project = ET.parse(ROOT/'RayTrophiStudio/RayTrophiStudio.vcxproj')
        ns = {'ms': 'http://schemas.microsoft.com/developer/msbuild/2003'}
        compiled = [n.get('Include') for n in project.findall('.//ms:ClCompile', ns)]
        self.assertEqual(compiled.count('source\\src\\Backend\\VulkanVolumeStageSplit.cpp'), 1)
        for name in EXTRACTED | {'volume_closesthit.rchit'}:
            self.assertLess(len((SH/name).read_text(encoding='utf-8').splitlines()), 2000)

    def test_foreground_redispatch_preserves_interval_and_work_budget(self):
        # Reference: an SDF box wins before a gas box inside it, but the true
        # surface lies farther away. Both designs integrate the same member set
        # and stop at that boundary, irrespective of gas BVH visit order.
        fields = [(2, 1.0, 5.0, .3), (3, 1.0, 7.0, .7), (4, 2.0, 8.0, .5)]
        for surface in (.5, 1.0, 1.5, 4.0, 9.0):
            for order in itertools.permutations(fields):
                foreground = [f for f in order if f[1] > 0 and surface > f[1] + .003]
                if not foreground:
                    continue  # SDF stage shades directly; no signal is emitted.
                near = min(f[1] for f in foreground)
                legacy_members = sorted(f for f in fields if f[1] <= near < f[2])
                # Gas-only traversal can pick either tie; canonical overlap
                # resolution still gathers every active field at the same nearT.
                winner = min(order, key=lambda f: f[1])
                split_members = sorted(f for f in fields if f[1] <= winner[1] < f[2])
                self.assertEqual(legacy_members, split_members)
                end = min(surface, min(f[2] for f in legacy_members))
                reference_tau = sum(f[3] * (end-near) for f in legacy_members)
                split_tau = sum(f[3] * (end-winner[1]) for f in split_members)
                self.assertEqual(reference_tau, split_tau)
                # No call to RNG, origin shift or outer-loop increment belongs
                # to the routing helper (source contract checked separately).


if __name__ == '__main__':
    unittest.main()
