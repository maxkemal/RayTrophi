"""Single post-build run: independent emission on the bounce radiance cache.

External process, open pyro scene in Vulkan Rendered, paused frame. Compares the
cache against the full bounce march, which stays the reference. Changes the
material and march LOD, restores both, verifies the restore and never saves.

    python scripts/test/rt_test_pyro_bounce_cache_ipc.py --target "gas:Nuclear Gas" --out <scratch dir>

Images and report.json go to --out, which must not be inside the repository.
"""
import argparse
import base64
import json
import os
import sys
import time

import numpy as np
from PIL import Image

from rt_ipc import RtIpc

SPP = 64
CAP = (slice(85, 480), slice(700, 1140))
GROUND = (slice(700, 900), slice(300, 1400))
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
KEYS = ("independent_emission", "max_steps", "density_multiplier",
        "emission_exposure_ev", "soot_burnout", "soot_burnout_start", "soot_burnout_end")


def wait_samples(client, count, timeout=180.0):
    start = time.time()
    samples = 0
    while time.time() - start < timeout:
        samples = client.call("viewport.status")["samples"]
        if samples >= count:
            break
        time.sleep(0.05)
    return samples


def render(client, out_dir, name):
    client.call("render.set_gpu_timing", enabled=True)
    client.call("reset_accumulation")
    time.sleep(0.05)
    samples = wait_samples(client, SPP)
    shot = client.call("viewport.get_screenshot")
    client.call("render.set_gpu_timing", enabled=False)
    timing = client.call("render.gpu_timing_stats")
    path = os.path.join(out_dir, name + ".png")
    with open(path, "wb") as handle:
        handle.write(base64.b64decode(shot["image_base64"]))
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return rgb, {"samples": samples, "gpu_mean_ms": timing.get("gpu_mean_ms")}


def counters(client):
    client.call("render.volume_counters", enabled=True)
    client.call("reset_accumulation")
    wait_samples(client, 6, 60.0)
    stats = client.call("render.volume_stats")
    client.call("render.volume_counters", enabled=False)
    return {key: stats.get(key) for key in (
        "radiance_cache_marches", "radiance_cache_fallbacks", "secondary_volume_rays",
        "secondary_density_samples", "camera_density_samples")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="gas:Nuclear Gas")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    client = RtIpc()
    target = args.target
    original = client.call("volume.get_material", target=target)
    if "soot_burnout" not in original:
        client.close()
        raise SystemExit("Binary is older than soot burnout: build C++ and shaders first")
    lod = client.call("render.volume_march_lod")
    denoiser = client.call("render.realtime_denoiser_stats").get("mode", "svgf")
    base = {key: original[key] for key in KEYS}
    ev = original["emission_exposure_ev"]
    tuned = dict(base, independent_emission=True, max_steps=128,
                 density_multiplier=3.0, emission_exposure_ev=ev - 5, soot_burnout=0.9,
                 soot_burnout_start=0.2, soot_burnout_end=0.6)
    report = {"cases": {}}
    try:
        client.call("render.set_realtime_denoiser", mode="off")
        client.call("viewport.capture", enabled=True)
        # Second material is 4x brighter: the fire most able to show a ground drift.
        for label, material in (("tuned", tuned),
                                ("bright", dict(tuned, emission_exposure_ev=ev - 3))):
            client.call("volume.set_material", target=target, settings=material)
            client.call("render.set_volume_march_lod", secondary_radiance_cache=False)
            reference, reference_row = render(client, args.out, label + "_full")
            client.call("render.set_volume_march_lod", secondary_radiance_cache=True)
            cached, cached_row = render(client, args.out, label + "_cache")
            if label == "tuned":
                report["counters"] = counters(client)
            luma_ref, luma_cached = reference @ LUMA, cached @ LUMA
            row = {
                "full_ms": reference_row["gpu_mean_ms"], "cache_ms": cached_row["gpu_mean_ms"],
                "cap_full": float(luma_ref[CAP].mean()), "cap_cache": float(luma_cached[CAP].mean()),
                "ground_full": float(luma_ref[GROUND].mean()),
                "ground_cache": float(luma_cached[GROUND].mean()),
                "image_mae": float(np.abs(reference - cached).mean()),
            }
            report["cases"][label] = row
            print(label, json.dumps(row), flush=True)
        print("counters", json.dumps(report.get("counters")), flush=True)
    finally:
        client.call("volume.set_material", target=target, settings=base)
        client.call("render.set_volume_march_lod",
                    secondary_radiance_cache=lod["secondary_radiance_cache"])
        client.call("render.set_realtime_denoiser", mode=denoiser)
        client.call("viewport.capture", enabled=False)
        restored = (client.call("volume.get_material", target=target) == original and
                    client.call("render.volume_march_lod") == lod)
        print("RESTORED", restored, flush=True)
        report["restored"] = restored
        client.close()

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    failures = []
    if not report["restored"]:
        failures.append("restore")
    if not report["counters"].get("radiance_cache_marches"):
        failures.append("no ray reached the radiance cache")
    for label, row in report["cases"].items():
        if abs(row["ground_cache"] - row["ground_full"]) > 0.02:
            failures.append(label + ": ground differs by more than 5/255")
    print("FAIL: " + "; ".join(failures) if failures else "PASS", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
