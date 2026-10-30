"""Non-build equivalence oracle and call-boundary checks for compact responses.

Python math checks do not establish compiled GLSL parity or optimizer success.
"""
import math
from pathlib import Path
import random
import unittest

SH = Path(__file__).resolve().parents[2] / 'RayTrophiStudio/source/shaders'


def clamp(value, lo, hi):
    return min(max(value, lo), hi)


def reference(instance, graph, mask, density, ordinal):
    # Prior implementation: graph mutates a full per-sample instance copy.
    v = dict(instance)
    names = ('density', 'scatter_color', 'scatter_coefficient', 'absorption_color',
             'absorption_coefficient', 'emission_color', 'emission_intensity',
             'scatter_anisotropy', 'scatter_multi')
    v['density'] = density
    for bit, name in enumerate(names):
        if mask & (1 << bit):
            v[name] = graph[name]
    d = max(v['density'], 0)
    front = clamp(v['scatter_anisotropy'], -.99, .99) if mask & 128 else v['scatter_anisotropy']
    multi = clamp(v['scatter_multi'], 0, 1)
    mean = v['scatter_anisotropy_back'] * (1-v['scatter_lobe_mix']) + front*v['scatter_lobe_mix']
    reduced = v['random_walk_enabled'] and mean > .05 and ordinal >= clamp(v['exact'], 1, 512)
    coefficient = max(v['scatter_coefficient'], 0) * (1-mean if reduced else 1)
    rate = d*coefficient
    absorption = [max(c, 0) for c in v['absorption_color']]
    peak = max(absorption)
    weights = [c/peak for c in absorption] if peak > 1e-5 else [1]*3
    extinction = [rate + d*max(v['absorption_coefficient'], 0)*w for w in weights]
    scattering = [rate*clamp(c, 0, 1) for c in v['scatter_color']]
    emission = [c*d*max(v['emission_intensity'], 0)*sigma
                for c, sigma in zip(v['emission_color'], extinction)]
    return extinction, scattering, emission, rate, 0 if reduced else front, multi


def compact(instance, graph, mask, density, ordinal):
    # New representation has no transforms, ramps or acceleration arrays.
    choose = lambda bit, name: graph[name] if mask & bit else instance[name]
    d = max(graph['density'], 0) if mask & 1 else max(density, 0)
    scatter = max(choose(4, 'scatter_coefficient'), 0)
    absorption_coefficient = max(choose(16, 'absorption_coefficient'), 0)
    front = choose(128, 'scatter_anisotropy')
    if mask & 128:
        front = clamp(front, -.99, .99)
    multi = clamp(choose(256, 'scatter_multi'), 0, 1)
    back, mix = instance['scatter_anisotropy_back'], instance['scatter_lobe_mix']
    mean = back*(1-mix) + front*mix
    reduced = (instance['random_walk_enabled'] and mean > .05 and
               ordinal >= clamp(instance['exact'], 1, 512))
    if reduced:
        scatter *= 1-mean
    rate = d*scatter
    color = [max(c, 0) for c in choose(8, 'absorption_color')]
    peak = max(color)
    weights = [c/peak for c in color] if peak > 1e-5 else [1]*3
    extinction = [rate + d*absorption_coefficient*w for w in weights]
    scattering = [rate*clamp(c, 0, 1) for c in choose(2, 'scatter_color')]
    strength = max(choose(64, 'emission_intensity'), 0)
    emission = [c*d*strength*sigma for c, sigma in
                zip(choose(32, 'emission_color'), extinction)]
    return extinction, scattering, emission, rate, 0 if reduced else front, multi


class CompactMixtureResponse(unittest.TestCase):
    def test_all_graph_write_masks_preserve_colored_optics_and_similarity(self):
        rng = random.Random(20261030)
        for mask in range(512):
            for walking in (False, True):
                for ordinal in (0, 512):
                    def values():
                        return dict(
                            density=rng.uniform(-1, 4),
                            scatter_color=[rng.uniform(-1, 2) for _ in range(3)],
                            absorption_color=[rng.uniform(-1, 2) for _ in range(3)],
                            emission_color=[rng.uniform(-1, 2) for _ in range(3)],
                            scatter_coefficient=rng.uniform(-1, 10),
                            absorption_coefficient=rng.uniform(-1, 10),
                            emission_intensity=rng.uniform(-1, 10),
                            scatter_anisotropy=rng.uniform(-.99, .99),
                            scatter_multi=rng.uniform(-1, 2))
                    instance, graph = values(), values()
                    graph['scatter_anisotropy'] = rng.uniform(-2, 2)
                    instance.update(scatter_anisotropy_back=rng.uniform(-.99, .99),
                                    scatter_lobe_mix=rng.random(),
                                    random_walk_enabled=walking, exact=32)
                    density = 0 if mask & 1 else rng.uniform(-1, 4)
                    self.assertEqual(reference(instance, graph, mask, density, ordinal),
                                     compact(instance, graph, mask, density, ordinal))

    def test_absorption_black_fallback_and_zero_input_graph_density(self):
        instance = dict(density=0, scatter_color=[1, .5, 0], absorption_color=[0]*3,
                        emission_color=[1]*3, scatter_coefficient=2,
                        absorption_coefficient=3, emission_intensity=4,
                        scatter_anisotropy=0, scatter_multi=0,
                        scatter_anisotropy_back=0, scatter_lobe_mix=0,
                        random_walk_enabled=False, exact=32)
        graph = dict(instance, density=2)
        result = compact(instance, graph, 1, 0, 0)
        self.assertEqual(result[0], [10]*3)
        self.assertTrue(all(math.isfinite(v) for v in result[0]))

    def test_shader_boundaries_keep_shadow_graphs_without_emission_tree(self):
        sample = (SH/'volume_mixture_material_sample.glsl').read_text(encoding='utf-8')
        optics = (SH/'volume_mixture_optical_terms.glsl').read_text(encoding='utf-8')
        mixture = (SH/'volume_mixture.glsl').read_text(encoding='utf-8')
        for module in (sample, optics):
            self.assertNotIn('VkVolumeInstance', module)
        sampling = sample.split('bool mfSampleMaterial(', 1)[1].split('MfTerms mfEvaluate(', 1)[0]
        self.assertIn('program != MATPROG_NONE', sampling)
        self.assertIn('sampleTemperature(volumes.v[index], position)', sampling)
        self.assertIn('evalMaterialProgram(', sampling)
        self.assertIn('density <= 0.0 && program == MATPROG_NONE', sampling)
        self.assertNotIn('volumeEmissionAt(', sampling)
        self.assertNotIn('volumeEmissionSource(', sampling)
        shadow = sample.split('vec3 mfShadowExtinction(', 1)[1]
        self.assertIn('mfSampleMaterial(', shadow)
        self.assertIn('mfExtinctionTerms(', shadow)
        self.assertNotIn('mfOpticalTerms(', shadow)
        extinction = optics.split('MfTerms mfExtinctionTerms(', 1)[1].split('MfTerms mfOpticalTerms(', 1)[0]
        self.assertNotIn('volumeEmission', extinction)
        self.assertNotIn('sampleTemperature', extinction)
        self.assertIn('source = volumeEmissionAt(volumes.v[index]', optics)
        self.assertIn('emissionStrength, volumes.v[index]._retired_cloud', optics)
        self.assertIn('mfShadowExtinction(index, samplePosition, direction)', mixture)
        self.assertIn('float mfDensity(uint index, vec3 position, bool cameraFast)', mixture)
        self.assertNotIn('v.scatter_color =', sample)


if __name__ == '__main__':
    unittest.main()
