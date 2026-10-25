"""A/B of the Vulkan RT gas march LOD on the open scene (IPC).

    python scripts/test/rt_volume_march_lod_ab_ipc.py --domain "Physics Domain 1"

docs/dev/RT_GAS_SECONDARY_MARCH.md §8. For each phase-route case (splat only,
gas only, splat + gas) measures ms/sample and the camera/secondary volume
counters with the reference march (1/1/1/0) and with the given LOD. Phase
routes and the LOD setting are restored at the end, also on failure.
Counters are reported as totals per frame sample (thousands): per-path values
are not comparable across cases because paths_traced differs.
Needs Vulkan RT on and an accumulating viewport; the build is the user's.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_ipc import RtIpc  # noqa: E402
from rt_domain_layer_cost_ipc import wait_samples  # noqa: E402

KEYS = ("volume_rays", "camera_volume_rays", "secondary_volume_rays",
        "density_samples", "camera_density_samples", "secondary_density_samples",
        "shadow_density_samples", "light_cache_misses",
        "radiance_cache_marches", "radiance_cache_fallbacks")
CASES = (("splat_only", dict(solid="splat", liquid="splat", gas="hidden")),
         ("gas_only", dict(solid="hidden", liquid="hidden", gas="fog")),
         ("splat_gas", dict(solid="splat", liquid="splat", gas="fog")))
REFERENCE = dict(camera_light_stride=1, secondary_light_stride=1,
                 secondary_step_scale=1, secondary_max_steps=0,
                 secondary_radiance_cache=False)


def measure(ipc, frame, label):
    time.sleep(1.0)
    ipc.call("render.volume_counters", enabled=False)
    ipc.call("reset_accumulation")
    wait_samples(ipc, 16, frame)          # light cache refills after a change
    ipc.call("reset_accumulation")
    a, s0 = wait_samples(ipc, 8, frame)
    b, s1 = wait_samples(ipc, s0 + 32, frame)
    ms = (b - a) * 1000.0 / (s1 - s0)
    ipc.call("render.volume_counters", enabled=True)
    _, c0 = wait_samples(ipc, 0, frame)
    _, c1 = wait_samples(ipc, c0 + 8, frame)
    st = ipc.call("render.volume_stats")
    n = max(c1 - c0, 1)
    row = {"case": label, "ms": round(ms, 1)}
    row.update({k: round(st[k] / n / 1000.0, 1) for k in KEYS})
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="Physics Domain 1")
    ap.add_argument("--lod", default="1,4,2,128,1",
                    help="camera_stride,secondary_stride,step_scale,max_steps,"
                         "radiance_cache(0/1)")
    args = ap.parse_args()
    f = [int(x) for x in args.lod.split(",")]
    lod = dict(camera_light_stride=f[0], secondary_light_stride=f[1],
               secondary_step_scale=f[2], secondary_max_steps=f[3])
    if len(f) >= 5:
        lod.update(secondary_radiance_cache=bool(f[4]))

    ipc = RtIpc()
    frame = ipc.call("timeline.get_frame")
    routes = {k: v for k, v in ipc.call("fluid.get_phase_output", domain=args.domain).items()
              if k in ("solid", "liquid", "gas")}
    original = ipc.call("render.volume_march_lod")
    rows = []
    try:
        for case, patch in CASES:
            ipc.call("fluid.set_phase_output", domain=args.domain, **patch)
            for tag, setting in (("ref", REFERENCE), ("lod", lod)):
                ipc.call("render.set_volume_march_lod", **setting)
                rows.append(measure(ipc, frame, case + " " + tag))
    finally:
        ipc.call("render.set_volume_march_lod", **original)
        ipc.call("fluid.set_phase_output", domain=args.domain, **routes)
        print("restored", ipc.call("fluid.get_phase_output", domain=args.domain),
              ipc.call("render.volume_march_lod"))
    by = {r["case"]: r for r in rows}
    for case, _ in CASES:
        ref, new = by[case + " ref"], by[case + " lod"]
        print("%-10s ref %6.1f ms  lod %6.1f ms  (%+.0f%%)" % (
            case, ref["ms"], new["ms"],
            100.0 * (new["ms"] - ref["ms"]) / max(ref["ms"], 1e-6)))


if __name__ == "__main__":
    main()
