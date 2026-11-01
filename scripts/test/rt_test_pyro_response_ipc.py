"""Acceptance run for the pyro response batches. Vulkan Rendered, open pyro scene.

Measures the field, the soft limiter, the temperature fit, and random walk
against first-scatter fire lighting. Restores material, never saves.

    python scripts/test/rt_test_pyro_response_ipc.py --domain "Nuclear Gas" --out <scratch dir>

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

from rt_ipc import RtIpc, RtIpcError

CAP = (slice(85, 480), slice(700, 1140))
RING = (slice(600, 700), slice(640, 1170))
STEM = (slice(450, 620), slice(850, 980))
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
KEYS = ("independent_emission", "max_steps", "density_multiplier",
        "emission_exposure_ev", "soot_burnout", "soot_burnout_start", "soot_burnout_end",
        "temperature_power", "temperature_min", "temperature_max", "temperature_gamma",
        "random_walk")


class Run:
    def __init__(self, client, target, out_dir):
        self.c, self.target, self.out = client, target, out_dir

    def wait(self, count, timeout):
        start = time.time()
        samples = 0
        while time.time() - start < timeout:
            samples = self.c.call("viewport.status")["samples"]
            if samples >= count:
                break
            time.sleep(0.05)
        return samples

    def image(self, name):
        shot = self.c.call("viewport.get_screenshot")
        path = os.path.join(self.out, name + ".png")
        with open(path, "wb") as handle:
            handle.write(base64.b64decode(shot["image_base64"]))
        return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

    def render(self, name, material, spp=32, timeout=240.0):
        self.c.call("volume.set_material", target=self.target, settings=material)
        self.c.call("render.set_gpu_timing", enabled=True)
        self.c.call("reset_accumulation")
        time.sleep(0.05)
        self.wait(4, timeout)
        low = self.image(name + "_lo")
        samples = self.wait(spp, timeout)
        high = self.image(name)
        self.c.call("render.set_gpu_timing", enabled=False)
        ms = self.c.call("render.gpu_timing_stats").get("gpu_mean_ms")
        luma = high @ LUMA
        noise = luma - low @ LUMA
        row = {"samples": samples, "gpu_ms": ms, "cap_mean": float(luma[CAP].mean()),
               "cap_p99": float(np.percentile(luma[CAP], 99)),
               "noise_cap": float(noise[CAP].std()), "noise_ring": float(noise[RING].std()),
               "noise_stem": float(noise[STEM].std())}
        print(f"{name:18s} {ms or 0:7.1f} ms mean {row['cap_mean']:.4f} p99 {row['cap_p99']:.4f} "
              f"noise {row['noise_cap']:.4f}/{row['noise_ring']:.4f}/{row['noise_stem']:.4f}",
              flush=True)
        return row, high


def response(before, after, stops):
    a, b = before[CAP], after[CAP]
    la, lb = a @ LUMA, b @ LUMA
    fire = (a[..., 0] - a[..., 2]) > 0.15
    rows = {}
    for low in (0.45, 0.55, 0.65, 0.75, 0.85):
        band = fire & (la >= low) & (la < low + 0.1)
        if band.sum() > 200:
            rows[f"{low:.2f}"] = float((lb - la)[band].mean() / stops)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="Nuclear Gas")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    client = RtIpc()
    target = "gas:" + args.domain
    original = client.call("volume.get_material", target=target)
    if "temperature_gamma" not in original:
        client.close()
        raise SystemExit("Binary is older than this batch")
    denoiser = client.call("render.realtime_denoiser_stats").get("mode", "svgf")
    base = {key: original[key] for key in KEYS}
    run = Run(client, target, args.out)
    report = {"cases": {}}
    cases = report["cases"]
    try:
        stats = client.call("gas.field_stats", domain=args.domain, bins=32)
        report["field_live"] = stats
        print("temperature K", json.dumps(stats["temperature_kelvin"]), flush=True)
        print("density", json.dumps(stats["density"]), flush=True)
        print("ramp", json.dumps(stats["ramp_coordinate"]), flush=True)
        print("emission", json.dumps(stats["emission_luminance"]), flush=True)
        print("tau/voxel", json.dumps(stats["voxel_optical_depth"]), flush=True)
        print("limited", stats["limited_cells"], "of", stats["emitting_cells"], flush=True)
        rejected = False
        try:
            client.call("volume.set_material", target=target, settings={"temperature_gamma": 9.0})
        except RtIpcError:
            rejected = True
        report["gamma_rejected"] = rejected

        client.call("render.set_realtime_denoiser", mode="off")
        client.call("viewport.capture", enabled=True)
        ev = original["emission_exposure_ev"]
        off = dict(base, independent_emission=True, random_walk=False)

        # Soft limiter on the material the hard limit hurt.
        cases["live"], live = run.render("live", dict(off))
        cases["live_ev2"], live2 = run.render("live_ev2", dict(off, emission_exposure_ev=ev + 2))
        report["live_response_per_stop"] = response(live, live2, 2.0)
        print("live dL/stop", report["live_response_per_stop"], flush=True)

        tuned = dict(off, max_steps=128, density_multiplier=3.0, emission_exposure_ev=ev - 2.32,
                     temperature_power=3.0, soot_burnout=0.9, soot_burnout_start=0.2,
                     soot_burnout_end=0.6)
        cases["tuned"], _ = run.render("tuned", tuned)

        # Temperature fit, keeping the tuned look otherwise.
        client.call("volume.set_material", target=target, settings=tuned)
        fit = client.call("gas.fit_temperature_range", domain=args.domain)
        report["fit"] = fit
        print("fit", json.dumps(fit), flush=True)
        fitted = dict(tuned, temperature_min=fit["temperature_min"],
                      temperature_max=fit["temperature_max"],
                      temperature_gamma=fit["temperature_gamma"])
        cases["fitted"], _ = run.render("fitted", fitted)
        report["field_fitted"] = client.call("gas.field_stats", domain=args.domain, bins=32)
        print("ramp fitted", json.dumps(report["field_fitted"]["ramp_coordinate"]), flush=True)

        # Fire to smoke through the real walk, at 64 spp so sparks can settle. The
        # second material gives back the energy the walk adds.
        cases["walk"], _ = run.render("walk", dict(tuned, random_walk=True), spp=64)
        cases["walk_ev1"], _ = run.render("walk_ev1", dict(
            tuned, random_walk=True, emission_exposure_ev=ev - 3.32), spp=64)
        client.call("render.volume_counters", enabled=True)
        client.call("reset_accumulation")
        run.wait(6, 120.0)
        walk = client.call("render.volume_stats")
        client.call("render.volume_counters", enabled=False)
        report["walk_counters"] = {key: walk.get(key) for key in (
            "walk_paths", "walk_events", "walk_event_capped", "walk_shadow_traces")}
        print("walk", json.dumps(report["walk_counters"]), flush=True)
        names = ["tuned", "fitted", "walk", "walk_ev1", "live", "live_ev2"]
        tiles = [Image.open(os.path.join(args.out, n + ".png")).convert("RGB")
                 .crop((640, 70, 1180, 700)) for n in names]
        width, height = tiles[0].size
        sheet = Image.new("RGB", (width * 3, height * 2))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 3) * width, (i // 3) * height))
        sheet.save(os.path.join(args.out, "montage.png"))
    finally:
        client.call("volume.set_material", target=target, settings=base)
        client.call("render.set_realtime_denoiser", mode=denoiser)
        client.call("viewport.capture", enabled=False)
        report["restored"] = client.call("volume.get_material", target=target) == original
        print("RESTORED", report["restored"], flush=True)
        client.close()

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    failures = []
    if not report["restored"]:
        failures.append("restore")
    if not report.get("gamma_rejected"):
        failures.append("gamma range not enforced")
    if not report.get("field_live", {}).get("emission_measured"):
        failures.append("emission chain not measured")
    if not report.get("walk_counters", {}).get("walk_paths"):
        failures.append("random walk did not run on the emitter")
    print("FAIL: " + "; ".join(failures) if failures else "PASS", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
