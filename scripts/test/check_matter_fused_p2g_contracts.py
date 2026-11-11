"""Static contract: the MPM lane's fused P2G (sim_matter_p2g_fused, 2026-11-11).

Three dispatches (clear, scatter, normalize) replace 3 x 7 per substep. The FLIP
baseline must stay the transferred velocity WITHOUT the stress impulse.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RayTrophiStudio" / "source"


def read(rel):
    return (SRC / rel).read_text(encoding="utf-8")


def main():
    comp = read("shaders/sim_matter_p2g_fused.glsl")
    assert '#include "sim_matter_p2g_fused.glsl"' in read("shaders/sim_matter_p2g_fused.comp")
    assert '#define P2G_CLUSTERED' in read("shaders/sim_matter_p2g_fused_clustered.comp")
    # Clustered: reduce momentum, weight and stress per node, one leader writes;
    # no early `continue` before the cluster ops (an out-of-grid lane takes a dead base).
    assert "const float impulse = subgroupClusteredAdd(stress, 8u);" in comp
    assert "if (!inside) g = vec3(-100.0);" in comp
    # The plain fused kernel: everything outside the P2G_BLOCK sections.
    plain = re.sub(r"#ifdef P2G_BLOCK.*?#endif", "", comp, flags=re.S)
    assert sorted(int(b) for b in re.findall(r"binding\s*=\s*(\d+)", plain)) == list(range(19))
    push = re.search(r"layout\(push_constant\).*?\{(.*?)\}\s*pc;", plain, re.S).group(1)
    assert len(re.findall(r"\b(?:int|uint|float)\s+\w+", re.sub(r"//[^\n]*", "", push))) * 4 == 48
    # Block variant (S5, 2026-11-11): 21 bindings / 64 B; shared tile, global
    # atomic fallback outside it; a per-substep block sort (permutation only).
    assert sorted(int(b) for b in re.findall(r"binding\s*=\s*(\d+)", comp)) == list(range(21))
    push_all = re.search(r"layout\(push_constant\).*?\{(.*?)\}\s*pc;", comp, re.S).group(1)
    assert len(re.findall(r"\b(?:int|uint|float)\s+\w+", re.sub(r"//[^\n]*", "", push_all))) * 4 == 64
    assert '#define P2G_BLOCK' in read("shaders/sim_matter_p2g_block.comp")
    assert "shared float tile[3 * 3 * kTileNodes];" in comp
    assert "g_tile_lo = bc * kBlock - ivec3(2);" in comp
    # Cell clusters with node ownership were measured SLOWER and removed (2026-11-11).
    assert "subgroupShuffle" not in comp and "supportsClusterShuffle" not in read("src/Device/SimulationComputeVulkan.cpp")
    assert "const int rotate = int(gl_LocalInvocationID.x % 27u);" in comp
    sort = read("shaders/sim_matter_p2g_block_sort.comp")
    assert sorted(int(b) for b in re.findall(r"binding\s*=\s*(\d+)", sort)) == list(range(5))
    # Stress apart from the momentum with FLIP; baseline = momentum / w; velocity = (m + s) / w.
    assert "if ((pc.flags & 1u) == 0u) { addMomentum(c, f, v); return; }" in comp
    assert "float velocity = has ? (m + s) / w : 0.0;" in comp
    assert "float baseline = has ? m / w : 0.0;" in comp
    # Same physics as the per-component kernels it replaces.
    assert "addMomentum(c, f, w * mass * (vp + dot(C_row, dxw)));" in comp
    assert "addStress(c, f, -pc.dt * dry * dot(row, grad));" in comp
    assert "const float kMinNodeWeight = 1.0e-4;" in comp
    vk = read("src/Device/SimulationComputeVulkan.cpp")
    assert '{ "sim_matter_p2g_fused", "sim_matter_p2g_fused.spv", 19, 48 }' in vk
    assert 'kernel == "sim_matter_p2g_fused" ||' in vk, "float-atomic gate"
    assert '{ "sim_matter_p2g_block", "sim_matter_p2g_block.spv", 21, 64 }' in vk
    assert '{ "sim_matter_p2g_block_sort", "sim_matter_p2g_block_sort.spv", 5, 64 }' in vk
    assert 'kernel == "sim_matter_p2g_block" && !m_has_shared_float_atomics' in vk
    assert "atomicFloatFeatures.shaderSharedFloat32AtomicAdd =" in read("src/Backend/VulkanBackend.cpp")
    bat = read("shaders/compile_sim_shaders.bat")
    for name in ("sim_matter_p2g_fused", "sim_matter_p2g_fused_clustered",
                 "sim_matter_p2g_block", "sim_matter_p2g_block_sort",
                 "sim_matter_p2g_clustered", "sim_matter_stress_p2g_clustered"):
        assert " " + name + " " in bat, name
    assert 'std::string(d.name) == "sim_matter_p2g_fused")) {' in vk
    for d in (SRC / "shaders", ROOT / "x64/Release/shaders"):
        assert (d / "sim_matter_p2g_fused.spv").exists(), d
        assert (d / "sim_matter_p2g_fused_clustered.spv").exists(), d
    p2g = read("src/Physics/Fluid/FluidGpuP2G.inl")
    gate = p2g[p2g.index("bool fused_transfer = false;"):p2g.index("bool ok = true;")]
    assert "granular && gpu_buffers.matter_model.enabled" in gate
    assert "!window_normalize && compute->backendType() == ComputeBackendType::VulkanCompute" in gate
    assert "if (!fused_transfer) return false;" in gate
    assert p2g.count("!sparse_transfer && !fused_transfer") == 2
    print("PASS fused MPM P2G: 19 bindings / 48 B, FLIP baseline stress-free, MPM-lane gate, per-component chain skipped")


if __name__ == "__main__":
    main()
