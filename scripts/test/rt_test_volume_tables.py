"""Which backend actually HOLDS the volume table?

This test exists because of a failure with no symptom other than absence.

The process can run TWO Vulkan devices at once: the render backend, and a
dedicated raster viewport backend (`g_viewport_backend`, created whenever Vulkan
is available). The volume SSBO is per DEVICE. The volume packet was published to
the render backend only, so the viewport adapter's table stayed empty for the
whole session — and every realtime volume consumer (the gas march, the
SurfaceSDF pass, the material-preview branch) gates on `m_volumeCount > 0` and
refused silently. No warning, no crash, no partial result: an empty frame, on
every scene.

★ `render.volume_stats()` CANNOT see this. It reports what the shader counted,
and a pass that was never recorded counts zero — the same zero a scene with no
volumes produces. That ambiguity is what let this survive. `volume_tables()`
reports what was PUBLISHED, per consumer, which separates the two.

Run with the app open and at least one volume in the scene (VDB, gas domain or
a fluid SurfaceSDF):

    python rt_test_volume_tables.py
"""

import sys

import rt


def main():
    t = rt.render.volume_tables()
    if not t["available"]:
        print("FAIL: no Vulkan backend at all — nothing to report")
        return 1

    rows = t["backends"]
    print(f"{len(rows)} backend row(s):")
    for b in rows:
        print(f"  role={b['role']:<9} vulkan={int(b['is_vulkan'])} "
              f"instances={b['instance_count']:<4} "
              f"buffer={int(b['buffer_allocated'])} "
              f"sim_device_is_mine={int(b['sim_device_is_this_backends'])} "
              f"dense_gas_mirrors={b.get('dense_gas_mirror_buffers', 0)}")

    by_role = {b["role"]: b for b in rows}
    render = by_role.get("render")
    viewport = by_role.get("viewport")

    if viewport is None:
        print()
        print("Only one row: this session has a single Vulkan backend serving "
              "both render and raster viewport. The split this test guards "
              "against cannot occur here — run it again with a dedicated "
              "viewport backend (OptiX or CPU as the render device) to exercise "
              "the real case.")
        return 0

    failed = False
    if render and render["instance_count"] > 0 and viewport["instance_count"] == 0:
        print()
        print("FAIL: the render backend holds "
              f"{render['instance_count']} volume instance(s) and the raster "
              "viewport holds NONE.")
        print("      The realtime viewport cannot draw a volume it was never "
              "given. Expect: no gas, no SurfaceSDF, and the material-preview "
              "branch skipped entirely — with no message anywhere.")
        print("      SceneLog will agree: '[MPVolume] pass gates: ... "
              "volumeCount=0 bound=0 -> SKIPPED'.")
        failed = True

    if not viewport["buffer_allocated"] and viewport["instance_count"] > 0:
        print()
        print("FAIL: viewport reports instances but no SSBO allocation — the "
              "count and the buffer disagree, which is worse than either being "
              "zero.")
        failed = True

    # The cross-device gas path. A backend that does not own the simulation
    # device must hold its OWN copy of every live dense gas grid, or it draws no
    # gas at all — and draws it silently, because a foreign device address reads
    # as zero density, which is pixel-for-pixel identical to empty smoke.
    for b in rows:
        if not (b["is_vulkan"] and b["instance_count"] > 0):
            continue
        if b["sim_device_is_this_backends"]:
            continue
        mirrors = b.get("dense_gas_mirror_buffers", 0)
        print()
        print(f"NOTE: backend '{b['role']}' does not own the simulation compute "
              f"device; it holds {mirrors} dense-gas mirror buffer(s).")
        if mirrors == 0:
            print("      If this scene HAS a live gas domain, that is the "
                  "failure: no mirror means no density source, and the gas will "
                  "not be drawn at all. Check SceneLog for '[VolumePublish] "
                  "... simDeviceIsMine=0'.")
            print("      If the scene has only baked VDB / SurfaceSDF volumes, "
                  "0 is correct — those upload through the NanoVDB path.")
        else:
            print("      The cross-device copy is live. Gas should render here "
                  "as it does in Rendered, one publish behind at most.")

    if failed:
        return 1
    print()
    print("OK: every Vulkan backend that can display volumes holds a table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
