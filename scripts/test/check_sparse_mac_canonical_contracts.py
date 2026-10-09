"""Source contracts for sparse S1: compact pages as the canonical liquid MAC velocity.

docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md. Not a compiler or GPU test: it expands the
GLSL entries with their defines, checks every binding/push ABI against the Vulkan
kernel registry, and runs an independent CPU oracle of the shared address contract
(sim_mac_lane.glsl): dense addresses equal the old index formulas, and compact face
lanes cover every owned face of the resident tiles exactly once.
"""
import random
import re
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "RayTrophiStudio/source").is_dir())
SOURCE = ROOT / "RayTrophiStudio/source"
SHADERS = SOURCE / "shaders"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def expand(name, defines, seen=None):
    """Tiny GLSL preprocessor: #include, #define NAME, #ifdef/#ifndef/#if defined/#elif/#else."""
    seen = set() if seen is None else seen
    text = (SHADERS / name).read_text(encoding="utf-8-sig")
    out = []
    stack = []  # (taking, any_branch_taken)

    def active():
        return all(taking for taking, _ in stack)

    def evaluate(expression):
        expression = expression.strip()
        names = re.findall(r"defined\s*\(?\s*(\w+)\s*\)?", expression)
        if not names:
            return False
        value = all(n in defines for n in names) if "&&" in expression else \
            any(n in defines for n in names)
        return not value if expression.startswith("!") else value

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#ifdef"):
            taking = stripped.split()[1] in defines
            stack.append((taking, taking))
        elif stripped.startswith("#ifndef"):
            taking = stripped.split()[1] not in defines
            stack.append((taking, taking))
        elif stripped.startswith("#if "):
            taking = evaluate(stripped[3:])
            stack.append((taking, taking))
        elif stripped.startswith("#elif"):
            _, taken = stack.pop()
            taking = not taken and evaluate(stripped[5:])
            stack.append((taking, taken or taking))
        elif stripped.startswith("#else"):
            _, taken = stack.pop()
            stack.append((not taken, True))
        elif stripped.startswith("#endif"):
            stack.pop()
        elif not active():
            continue
        elif stripped.startswith("#define"):
            parts = stripped.split()
            if len(parts) >= 2 and "(" not in parts[1]:
                defines = defines | {parts[1]}
            out.append(line)
        elif stripped.startswith("#include"):
            target = re.search(r'"([^"]+)"', stripped).group(1)
            if target in seen and target != "sim_dispatch.glsl":
                continue
            seen.add(target)
            body, defines = expand(target, defines, seen)
            out.append(body)
        else:
            out.append(line)
    assert not stack, name
    return "\n".join(out), defines


def entry(name):
    text = (SHADERS / (name + ".comp")).read_text(encoding="utf-8-sig")
    defines = set(re.findall(r"^#define\s+(\w+)", text, re.M))
    return expand(name + ".comp", defines)[0]


def push_bytes(source):
    block = re.search(r"layout\(push_constant\)\s*uniform\s+\w+\s*\{(.*?)\}\s*pc;", source, re.S)
    body = re.sub(r"//[^\n]*", "", block.group(1))
    return 4 * len(re.findall(r"\b(?:int|uint|float)\s+\w+", body)) + \
        16 * len(re.findall(r"\b(?:uvec4|vec4|ivec4)\s+\w+", body))


def bindings(source):
    found = sorted(int(b) for b in re.findall(r"binding\s*=\s*(\d+)", source))
    assert found == list(range(len(found))), found
    return len(found)


def check_abis():
    registry = read("src/Device/SimulationComputeVulkan.cpp")
    batch = read("shaders/compile_sim_shaders.bat")
    compact = {
        "sim_sparse_mac_zero_faces": (6, 36), "sim_sparse_mac_matter_zero_faces": (6, 36),
        "sim_sparse_mac_divergence": (7, 52), "sim_sparse_mac_divergence_var": (13, 52),
        "sim_sparse_mac_divergence_porous": (13, 52),
        "sim_sparse_mac_subtract_gradient": (7, 52),
        "sim_sparse_mac_subtract_gradient_var": (13, 52),
        "sim_sparse_mac_capture_compact": (8, 36), "sim_sparse_mac_advect": (11, 80),
        "sim_sparse_mac_matter_advect": (13, 84), "sim_sparse_mac_matter_contact": (21, 16),
        "sim_sparse_mac_viscosity_capture": (15, 68), "sim_sparse_mac_viscosity_sweep": (15, 68),
        "sim_sparse_pressure_init_mac_weights": (18, 80),
        "sim_sparse_pressure_spmv_mac_weights": (18, 80),
    }
    # Dense twins: unchanged ABIs over the shared bodies.
    dense = {
        "sim_fluid_zero_solid_faces": (4, 36), "sim_matter_zero_faces": (4, 36),
        "sim_fluid_divergence": (5, 52), "sim_fluid_divergence_var": (11, 52),
        "sim_fluid_divergence_porous": (11, 52), "sim_fluid_subtract_gradient": (5, 52),
        "sim_fluid_subtract_gradient_var": (11, 52), "sim_fluid_advect_tail": (9, 80),
        "sim_matter_advect": (11, 84), "sim_matter_contact": (19, 16),
        "sim_sparse_viscosity_capture": (13, 68), "sim_sparse_viscosity_sweep": (13, 68),
        "sim_fluid_viscosity_rbgs": (11, 52), "sim_sparse_mac_capture": (8, 36),
        "sim_sparse_pressure_init": (16, 80), "sim_sparse_pressure_spmv": (16, 80),
    }
    for table in (compact, dense):
        for kernel, (count, push) in table.items():
            source = entry(kernel)
            assert bindings(source) == count, (kernel, bindings(source), count)
            assert push_bytes(source) == push, (kernel, push_bytes(source), push)
            assert re.search(rf'"{kernel}",\s*"{kernel}\.spv",\s*{count},\s*{push}\s*\}}',
                             registry), kernel
            if kernel in compact:
                # New entries are listed literally; dense twins keep their
                # existing (partly pattern-based) compile lines.
                assert kernel in batch, kernel
            assert source.count("#version 450") == 1, kernel
            if kernel in compact and kernel != "sim_sparse_mac_capture_compact":
                assert "mac_tile_map[" in source, kernel
    lane = (SHADERS / "sim_mac_lane.glsl").read_text(encoding="utf-8")
    assert "return slot == 0u ? MAC_ABSENT" not in lane  # explicit branch below
    assert "if (slot == 0u) {" in lane and "(slot - 1u) * 576u" in lane


def check_host():
    p2g = read("src/Physics/Fluid/FluidGpuP2G.inl")
    assert "gpu_buffers.sparse_mac_transfer.compact_owner &&" in p2g
    assert p2g.index("runSparseMacP2G(") < p2g.index("for (int comp = 0; comp < 3 && !sparse_transfer")
    mac = read("src/Physics/Fluid/SparseMacTransferGpu.cpp")
    assert "(!canonical && !operation(compute, buffers, constants, \"sim_sparse_mac_publish\"" in mac
    assert "storage.canonical ? \"sim_sparse_mac_capture_compact\"" in mac
    assert "storage.compact_blocked = blocked;" in mac
    allocator = read("src/Physics/ParticleSimulation.cpp")
    assert "releaseDenseMacForCompactOwner(compute, buffers)" in allocator
    assert "buffers.pressure.valid() &&\n        compute.getBufferSize(buffers.pressure) == 0" in \
        allocator.replace("\r\n", "\n")
    pressure = read("src/Physics/Fluid/FluidGpuPressure.inl")
    assert "mac.compact && !sparse_pressure" in pressure
    assert "\"sim_sparse_mac_subtract_gradient_var\"" in pressure
    mixed = read("src/Physics/Fluid/MatterGpuStep.inl")
    assert mixed.index("mac_storage.compact_owner = compact_wanted") < mixed.index("if (!ensure(primary))")
    assert "Fluid::publishCompactMacToHost(*compute, primary," in mixed
    assert "\"sim_sparse_mac_matter_contact\"" in mixed
    assert "liquid_mac.compact ? 21 : 19" in mixed
    pure = read("src/Physics/Fluid/FluidDomainStep.inl")
    assert pure.count("gpu_buffers.sparse_mac_transfer.compact_owner = false;") == 2
    view = read("src/Physics/Fluid/MatterGpuModelView.cpp")
    assert "\"sim_sparse_mac_matter_zero_faces\"" in view and "\"sim_sparse_mac_matter_advect\"" in view
    # One core field, three surfaces.
    for path in ("src/Api/RtIpc.cpp", "src/Api/RtPython.cpp", "src/Api/RtApiParticle.cpp"):
        text = read(path)
        assert "transfer_sparse_canonical" in text and "transfer_sparse_blocked" in text, path
    assert "transfer_sparse_blocked" in read("include/Fluid/FluidActiveWindow.h")


def check_weight_ownership():
    """Guard the allocator, descriptor aliases and compact-failure recovery.

    GPU parity remains a live IPC gate. This rejects accidentally retaining a
    dense weight bank or removing its allocation from the dense recovery path.
    """
    transfer = read("src/Physics/Fluid/SparseMacTransferGpu.cpp")
    release = transfer.split("bool releaseDenseMacForCompactOwner(", 1)[1]
    release = release.split("MacVelocityBinding macVelocityBinding(", 1)[0]
    assert release.index("if (!buffers.sparse_mac_transfer.compact_owner)") < \
        release.index("compute.destroyBuffer(*handle)")
    banks = ("vel_x", "vel_y", "vel_z", "scratch_vel_x", "scratch_vel_y",
             "scratch_vel_z", "temperature", "fuel", "scratch_scalar",
             "var_u_weight", "var_v_weight", "var_w_weight")
    assert set(re.findall(r"&buffers\.(\w+)", release)) == set(banks)
    assert "*handle = {};" in release
    allocator = read("src/Physics/ParticleSimulation.cpp")
    for bank in ("temperature", "fuel", "scratch_scalar",
                 "var_u_weight", "var_v_weight", "var_w_weight"):
        assert re.search(
            rf"if \(dense_mac\) \{{[^{{}}]*ensureComputeBuffer\(compute, "
            rf"buffers\.{bank},[^{{}}]*\}}", allocator), bank
    assert re.search(
        r"\(!dense_mac \|\| \(buffers\.temperature\.valid\(\) && "
        r"buffers\.fuel\.valid\(\) &&\s*buffers\.scratch_scalar\.valid\(\)\)\)",
        allocator)
    # Page-only dispatches still bind every descriptor; unused dense slots alias
    # the matching pages. Noncanonical publication validates all dense capacities.
    assert "dense(weights[axis], pool[5 + axis])" in transfer
    assert "if (!canonical && (!dense_velocity[axis].valid()" in transfer
    assert "storage.compact_blocked = storage.compact_blocked || canonical;" in transfer
    mixed = read("src/Physics/Fluid/MatterGpuStep.inl")
    recovery = mixed.split("if (!transferred && lane == 0 &&", 1)[1]
    recovery = recovery.split("if (!transferred ||", 1)[0]
    assert recovery.index("transferred = ensure(primary);") < \
        recovery.index("runGpuFluidP2G(")
    assert "!mac_storage.compact_owner" in recovery


def dense_index(face, comp, cells):
    dims = list(cells)
    dims[comp] += 1
    return face[0] + dims[0] * (face[1] + dims[1] * face[2])


def old_index(face, comp, nx, ny, nz):
    i, j, k = face
    if comp == 0:
        return i + j * (nx + 1) + k * (nx + 1) * ny
    if comp == 1:
        return i + j * nx + k * nx * (ny + 1)
    return i + j * nx + k * nx * ny


def tile_key(face, comp, cells):
    owner = list(face)
    owner[comp] = min(owner[comp], cells[comp] - 1)
    tiles = [(c + 7) // 8 for c in cells]
    t = [o // 8 for o in owner]
    return t[0] + tiles[0] * (t[1] + tiles[1] * t[2])


def lane_face(lane, comp, key, cells):
    tiles = [(c + 7) // 8 for c in cells]
    tile = (key % tiles[0], (key // tiles[0]) % tiles[1], key // (tiles[0] * tiles[1]))
    dims = [8, 8, 8]
    dims[comp] = 9
    local = lane % 576
    face = (tile[0] * 8 + local % dims[0], tile[1] * 8 + (local // dims[0]) % dims[1],
            tile[2] * 8 + local // (dims[0] * dims[1]))
    maximum = [c - 1 for c in cells]
    maximum[comp] += 1
    owned = all(0 <= face[a] <= maximum[a] for a in range(3)) and \
        tile_key(face, comp, cells) == key
    return face, owned


def check_oracle():
    rng = random.Random(20261009)
    for _ in range(40):
        cells = [rng.randint(1, 30) for _ in range(3)]
        for comp in range(3):
            dims = list(cells)
            dims[comp] += 1
            for _ in range(200):
                face = [rng.randrange(d) for d in dims]
                assert dense_index(face, comp, cells) == old_index(face, comp, *cells)
        tiles = [(c + 7) // 8 for c in cells]
        all_keys = list(range(tiles[0] * tiles[1] * tiles[2]))
        resident = rng.sample(all_keys, rng.randint(1, len(all_keys)))
        for comp in range(3):
            covered = {}
            for slot, key in enumerate(resident):
                for local in range(576):
                    face, owned = lane_face(slot * 576 + local, comp, key, cells)
                    if owned:
                        assert face not in covered, (cells, comp, face)
                        covered[face] = slot * 576 + local
            dims = list(cells)
            dims[comp] += 1
            for k in range(dims[2]):
                for j in range(dims[1]):
                    for i in range(dims[0]):
                        face = (i, j, k)
                        key = tile_key(face, comp, cells)
                        # Every face of a resident tile is covered once; others absent.
                        assert (face in covered) == (key in resident), (cells, comp, face)
                        if face in covered:
                            slot = resident.index(key)
                            local = covered[face] - slot * 576
                            # macAddress agrees with the lane that owns the face.
                            owner = list(face)
                            owner[comp] = min(owner[comp], cells[comp] - 1)
                            base = [(o // 8) * 8 for o in owner]
                            d = [8, 8, 8]
                            d[comp] = 9
                            lf = [face[a] - base[a] for a in range(3)]
                            assert lf[0] + d[0] * (lf[1] + d[1] * lf[2]) == local


def main():
    check_abis()
    check_host()
    check_weight_ownership()
    check_oracle()
    print("PASS sparse S1/S1b: compact canonical MAC and solid-weight ABIs, "
          "host ownership/fallback/publication wiring, S1b P2G weight allocation gate, "
          "dense address + compact coverage oracle")


if __name__ == "__main__":
    main()
