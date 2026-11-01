"""Acceptance run for the published flame field. Needs the pyro project at frame 0.

Steps the live simulation, and at each frame compares flame off, default and
boosted on one tuned material, with and without burnout. Restores, never saves.

    python scripts/test/rt_test_pyro_flame_ipc.py --domain "Nuclear Gas" --out <scratch dir>

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

FRAMES = (12, 30, 69)
SPP = 32
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
KEYS = ("independent_emission", "max_steps", "density_multiplier", "emission_exposure_ev",
        "soot_burnout", "soot_burnout_start", "soot_burnout_end", "temperature_power",
        "temperature_gamma", "flame_emission", "random_walk")


def render(client, target, out_dir, name, material):
    client.call("volume.set_material", target=target, settings=material)
    client.call("render.set_gpu_timing", enabled=True)
    client.call("reset_accumulation")
    time.sleep(0.05)
    start = time.time()
    while time.time() - start < 120 and client.call("viewport.status")["samples"] < SPP:
        time.sleep(0.05)
    shot = client.call("viewport.get_screenshot")
    client.call("render.set_gpu_timing", enabled=False)
    ms = client.call("render.gpu_timing_stats").get("gpu_mean_ms")
    path = os.path.join(out_dir, name + ".png")
    with open(path, "wb") as handle:
        handle.write(base64.b64decode(shot["image_base64"]))
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    luma = rgb @ LUMA
    fire = (rgb[..., 0] - rgb[..., 2]) > 0.15
    values = luma[fire] if fire.any() else luma
    row = {"gpu_ms": ms, "fire_pixels": int(fire.sum()), "p50": float(np.percentile(values, 50)),
           "p95": float(np.percentile(values, 95)), "contrast": float(
               np.percentile(values, 95) - np.percentile(values, 5))}
    print(f"{name:20s} {ms or 0:7.1f} ms fire {row['fire_pixels']:7d} p50 {row['p50']:.3f} "
          f"p95 {row['p95']:.3f} contrast {row['contrast']:.3f}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="Nuclear Gas")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    client = RtIpc()
    target = "gas:" + args.domain
    original = client.call("volume.get_material", target=target)
    if "flame_emission" not in original:
        client.close()
        raise SystemExit("Binary is older than this batch")
    denoiser = client.call("render.realtime_denoiser_stats").get("mode", "svgf")
    base = {key: original[key] for key in KEYS}
    ev = original["emission_exposure_ev"]
    tuned = dict(base, independent_emission=True, max_steps=128, density_multiplier=3.0,
                 emission_exposure_ev=ev - 2.32, temperature_power=3.0, soot_burnout=0.0,
                 random_walk=False)
    burn = dict(tuned, soot_burnout=0.9, soot_burnout_start=0.2, soot_burnout_end=0.6)
    report = {"frames": {}}
    try:
        client.call("render.set_realtime_denoiser", mode="off")
        client.call("viewport.capture", enabled=True)
        client.call("volume.set_material", target=target, settings=dict(burn, flame_emission=1.0))
        frame = 0
        for stop in FRAMES:
            while frame < stop:
                frame += 1
                client.call("timeline.set_frame", frame=frame)
                time.sleep(0.15)
            time.sleep(1.0)
            stats = client.call("gas.field_stats", domain=args.domain, bins=8)
            row = {"active_cells": stats["active_cells"], "burning_cells": stats["burning_cells"],
                   "flame": stats["flame"], "temperature_kelvin": stats["temperature_kelvin"]}
            print(f"frame {stop}: burning {row['burning_cells']} of {row['active_cells']}, flame "
                  f"p50 {row['flame']['p50']:.3f} p95 {row['flame']['p95']:.3f}, "
                  f"T p95 {row['temperature_kelvin']['p95']:.0f}", flush=True)
            cases = {}
            for name, material in (
                    ("noflame", dict(tuned, flame_emission=0.0)),
                    ("add05", dict(tuned, flame_emission=0.5)),
                    ("add2", dict(tuned, flame_emission=2.0)),
                    ("burn_noflame", dict(burn, flame_emission=0.0)),
                    ("burn_add2", dict(burn, flame_emission=2.0))):
                cases[name] = render(client, target, args.out, f"f{stop}_{name}", material)
            row["cases"] = cases
            report["frames"][str(stop)] = row
        names = [f"f{f}_{n}" for f in FRAMES for n in ("noflame", "add2", "burn_add2")]
        tiles = [Image.open(os.path.join(args.out, n + ".png")).convert("RGB")
                 .crop((560, 40, 1260, 720)).resize((525, 510)) for n in names]
        sheet = Image.new("RGB", (525 * 3, 510 * len(FRAMES)))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 3) * 525, (i // 3) * 510))
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
    first = report["frames"].get(str(FRAMES[0]), {}).get("cases", {})
    if first and not first["add2"]["p95"] > first["noflame"]["p95"]:
        failures.append("flame emission changed nothing on a burning frame")
    last = report["frames"].get(str(FRAMES[-1]), {}).get("cases", {})
    if last and not last["add2"]["fire_pixels"] + last["add2"]["p95"] > \
            last["noflame"]["fire_pixels"] + last["noflame"]["p95"]:
        failures.append("flame emission is still zero on the late frame")
    print("FAIL: " + "; ".join(failures) if failures else "PASS", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
