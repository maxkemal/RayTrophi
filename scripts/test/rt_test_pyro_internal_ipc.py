"""Acceptance run for internal fire light, burnout together, and smoke appearance.

Needs the pyro project at frame 0. Checks the smoke_* view, then steps the live
simulation and compares the legacy look, independent burnout, internal light on
both, and additive flame. Restores, never saves.

    python scripts/test/rt_test_pyro_internal_ipc.py --domain "Nuclear Gas" --out <scratch dir>

Images and report.json go to --out, which must not be inside the repository.
"""
import argparse
import base64
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageFilter

from rt_ipc import RtIpc, RtIpcError

FRAMES = (12, 30, 69)
SPP = 32
# One extra accumulated sample reshuffles the ground sparkle, so images are
# compared above the horizon only.
HORIZON = 625
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
KEYS = ("independent_emission", "max_steps", "density_multiplier", "density_power",
        "emission_exposure_ev", "soot_burnout", "soot_burnout_start", "soot_burnout_end",
        "temperature_power", "temperature_gamma", "flame_emission", "random_walk",
        "internal_light_steps", "internal_light_stride", "scatter_color",
        "scatter_coefficient", "absorption_color", "absorption_coefficient", "multi_scatter")


def close(a, b, tolerance=1e-3):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tolerance)


def check_smoke(client, target, base, failures):
    client.call("volume.set_material", target=target, settings=dict(
        base, scatter_color=[1, 1, 1], scatter_coefficient=2.0,
        absorption_color=[1, 1, 1], absorption_coefficient=1.0))
    look = client.call("volume.get_material", target=target)
    if not close(look["smoke_color"], [2 / 3] * 3) or not close(look["smoke_opacity"], 3.0):
        failures.append("smoke view does not read the coefficients")
    client.call("volume.set_material", target=target, settings={"smoke_opacity": 6.0})
    doubled = client.call("volume.get_material", target=target)
    if not close(doubled["smoke_color"], look["smoke_color"]):
        failures.append("opacity moved color")
    if not close(doubled["scatter_coefficient"] + doubled["absorption_coefficient"], 6.0):
        failures.append("opacity is not total extinction")
    client.call("volume.set_material", target=target, settings={"smoke_color": [0.5, 0.4, 0.3]})
    tinted = client.call("volume.get_material", target=target)
    if not close(tinted["smoke_opacity"], 6.0) or not close(tinted["smoke_color"], [0.5, 0.4, 0.3]):
        failures.append("color moved opacity")
    try:
        client.call("volume.set_material", target=target,
                    settings={"smoke_opacity": 1.0, "scatter_coefficient": 1.0})
        failures.append("smoke and raw keys were accepted together")
    except RtIpcError:
        pass
    print("smoke view checked", flush=True)


def dark_specks(rgb):
    luma = (rgb @ LUMA) * 255.0
    median = np.asarray(Image.fromarray(luma.astype("uint8")).filter(
        ImageFilter.MedianFilter(7))).astype(np.float32)
    fire = (rgb[..., 0] - rgb[..., 2]) > 0.235
    return int(((luma < median * 0.7) & fire & (median > 110)).sum())


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
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)[:HORIZON] / 255.0
    luma = rgb @ LUMA
    fire = (rgb[..., 0] - rgb[..., 2]) > 0.15
    values = luma[fire] if fire.any() else luma
    row = {"gpu_ms": ms, "fire_pixels": int(fire.sum()), "specks": dark_specks(rgb),
           "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
           "contrast": float(np.percentile(values, 95) - np.percentile(values, 5))}
    print(f"{name:24s} {ms or 0:7.1f} ms fire {row['fire_pixels']:7d} p50 {row['p50']:.3f} "
          f"p95 {row['p95']:.3f} contrast {row['contrast']:.3f} specks {row['specks']}", flush=True)
    return rgb, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="Nuclear Gas")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    client = RtIpc()
    target = "gas:" + args.domain
    original = client.call("volume.get_material", target=target)
    if "smoke_opacity" not in original:
        client.close()
        raise SystemExit("Binary is older than this batch")
    denoiser = client.call("render.realtime_denoiser_stats").get("mode", "svgf")
    base = {key: original[key] for key in KEYS}
    ev = original["emission_exposure_ev"]
    # The look the user reported as better than the random walk.
    legacy = dict(base, independent_emission=False, max_steps=121, density_multiplier=8.333,
                  density_power=3.194, temperature_power=0.277, soot_burnout=0.0,
                  flame_emission=0.0, random_walk=False, internal_light_steps=32,
                  internal_light_stride=32, multi_scatter=0.819)
    burn = dict(base, independent_emission=True, max_steps=128, density_multiplier=3.0,
                emission_exposure_ev=ev - 2.32, temperature_power=3.0, soot_burnout=0.9,
                soot_burnout_start=0.2, soot_burnout_end=0.6, flame_emission=0.0,
                random_walk=False, internal_light_steps=0)
    lit = dict(burn, internal_light_steps=32, internal_light_stride=32)
    cases = (("legacy", legacy), ("legacy_off", dict(legacy, internal_light_steps=0)),
             ("burn", burn), ("burn_lit", lit), ("burn_lit8", dict(lit, internal_light_steps=8)),
             ("burn_add", dict(burn, flame_emission=2.0)),
             ("burn_lit_add", dict(lit, flame_emission=2.0)))
    failures = []
    report = {"frames": {}}
    images = {}
    try:
        check_smoke(client, target, base, failures)
        client.call("render.set_realtime_denoiser", mode="off")
        client.call("viewport.capture", enabled=True)
        client.call("volume.set_material", target=target, settings=dict(lit, flame_emission=2.0))
        frame = 0
        for stop in FRAMES:
            while frame < stop:
                frame += 1
                client.call("timeline.set_frame", frame=frame)
                time.sleep(0.15)
            time.sleep(1.0)
            rows = {}
            for name, material in cases:
                images[f"f{stop}_{name}"], rows[name] = render(
                    client, target, args.out, f"f{stop}_{name}", material)
            report["frames"][str(stop)] = rows
            # Steps set reach only, so a smaller budget must not change the energy.
            if stop == FRAMES[-1] and abs(rows["burn_lit"]["p50"] - rows["burn_lit8"]["p50"]) > 0.03:
                failures.append("internal light energy still depends on steps")
            for a, b, what in (("legacy", "legacy_off", "internal light off in legacy mode"),
                               ("burn_lit", "burn", "internal light off in independent mode"),
                               ("burn_add", "burn", "flame emission adds nothing")):
                change = float(np.abs(images[f"f{stop}_{a}"] - images[f"f{stop}_{b}"]).mean())
                rows[f"{a}_vs_{b}"] = change
                print(f"  {a} vs {b}: {change:.5f}", flush=True)
                # Flame at frame 69 lies on the ground, below the compared region.
                if change < 1e-4 and not (a == "burn_add" and stop == FRAMES[-1]):
                    failures.append(f"frame {stop}: {what}")
        names = [f"f{f}_{n}" for f in FRAMES for n in ("legacy", "burn", "burn_lit", "burn_lit_add")]
        tiles = [Image.open(os.path.join(args.out, n + ".png")).convert("RGB")
                 .crop((560, 40, 1260, 720)).resize((394, 383)) for n in names]
        sheet = Image.new("RGB", (394 * 4, 383 * len(FRAMES)))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 4) * 394, (i // 4) * 383))
        sheet.save(os.path.join(args.out, "montage.png"))
    finally:
        client.call("volume.set_material", target=target, settings=base)
        client.call("render.set_realtime_denoiser", mode=denoiser)
        client.call("viewport.capture", enabled=False)
        report["restored"] = close(
            client.call("volume.get_material", target=target)["smoke_opacity"],
            original["smoke_opacity"])
        print("RESTORED", report["restored"], flush=True)
        client.close()

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    if not report["restored"]:
        failures.append("restore")
    print("FAIL: " + "; ".join(failures) if failures else "PASS", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
