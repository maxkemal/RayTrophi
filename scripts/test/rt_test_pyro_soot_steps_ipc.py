"""Single post-build acceptance run for soot burnout and the RT camera step cap.

External process, open pyro scene in Vulkan Rendered, paused frame. It changes
the material and march LOD, restores both, verifies the restore and never saves.

    python scripts/test/rt_test_pyro_soot_steps_ipc.py --target "gas:Nuclear Gas" --out <scratch dir>

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

SPP = 32
CAP = (slice(85, 480), slice(700, 1140))  # 1680x945 cap, same box as the 2026-11-01 baseline
SOOT_KEYS = ("soot_burnout", "soot_burnout_start", "soot_burnout_end")
TUNED_KEYS = SOOT_KEYS + ("density_multiplier",
                          "emission_exposure_ev", "max_steps", "independent_emission")


def wait_samples(client, count, timeout=120.0):
    start = time.time()
    samples = 0
    while time.time() - start < timeout:
        samples = client.call("viewport.status")["samples"]
        if samples >= count:
            break
        time.sleep(0.05)
    return samples


def measure(client, target, out_dir, name, patch, base):
    client.call("volume.set_material", target=target, settings=dict(base, **patch))
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
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    cap = luminance[CAP]
    row = {"samples": samples, "gpu_mean_ms": timing.get("gpu_mean_ms"),
           "cap_mean": float(cap.mean()), "cap_p50": float(np.median(cap)),
           "cap_p99": float(np.percentile(cap, 99)),
           "cap_contrast": float(np.percentile(cap, 95) - np.percentile(cap, 5))}
    print(f"{name:26s} {row['gpu_mean_ms'] or 0:7.1f} ms  mean {row['cap_mean']:.4f}  "
          f"p50 {row['cap_p50']:.4f}  p99 {row['cap_p99']:.4f}  contrast {row['cap_contrast']:.4f}",
          flush=True)
    return row


def rejected(client, method, **params):
    try:
        client.call(method, **params)
    except RtIpcError:
        return True
    return False


def montage(out_dir, names):
    tiles = [Image.open(os.path.join(out_dir, n + ".png")).convert("RGB")
             .crop((680, 70, 1160, 640)) for n in names]
    width, height = tiles[0].size
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, height * rows))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * width, (i // columns) * height))
    sheet.save(os.path.join(out_dir, "montage.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="gas:Nuclear Gas")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    client = RtIpc()
    report = {"contract": {}, "cases": {}}
    original = client.call("volume.get_material", target=args.target)
    if not all(key in original for key in SOOT_KEYS):
        client.close()
        raise SystemExit("Binary does not expose soot_burnout: build C++ and shaders first")
    lod = client.call("render.volume_march_lod")
    denoiser = client.call("render.realtime_denoiser_stats").get("mode", "svgf")
    base = {key: original[key] for key in TUNED_KEYS}
    target = args.target
    try:
        contract = report["contract"]
        contract["default_camera_cap_64"] = lod.get("camera_max_steps") == 64
        contract["cap_3_rejected"] = rejected(client, "render.set_volume_march_lod",
                                              camera_max_steps=3)
        contract["cap_512_rejected"] = rejected(client, "render.set_volume_march_lod",
                                                camera_max_steps=512)
        contract["soot_range_rejected"] = rejected(client, "volume.set_material", target=target,
                                                   settings={"soot_burnout": 1.5})
        contract["soot_order_rejected"] = rejected(client, "volume.set_material", target=target,
            settings={"soot_burnout_start": 0.7, "soot_burnout_end": 0.5})
        contract["unchanged_after_rejects"] = (
            client.call("volume.get_material", target=target) == original)
        print(json.dumps(contract), flush=True)

        client.call("render.set_realtime_denoiser", mode="off")
        client.call("viewport.capture", enabled=True)
        cases = report["cases"]
        off = {"independent_emission": True, "max_steps": 128}
        ev = original["emission_exposure_ev"]
        thin = dict(off, density_multiplier=3.0, emission_exposure_ev=ev - 5)

        # Step cap: same material, authored 128 against the new 64 and the 32 drift.
        client.call("render.set_volume_march_lod", camera_max_steps=0)
        cases["cap0_thin"] = measure(client, target, args.out, "cap0_thin", thin, base)
        for cap in (64, 32):
            client.call("render.set_volume_march_lod", camera_max_steps=cap)
            cases[f"cap{cap}_thin"] = measure(client, target, args.out, f"cap{cap}_thin", thin, base)
        client.call("render.set_volume_march_lod", camera_max_steps=64)

        # Soot burnout. The authored temperature range is wide, so also try a low band.
        grid = [
            ("thick_b0", dict(off), None),
            ("thick_b90", dict(off), (0.9, 0.2, 0.6)),
            ("thick_b90_low", dict(off), (0.9, 0.05, 0.3)),
            ("thin_b0", dict(thin), None),
            ("thin_b70", dict(thin), (0.7, 0.2, 0.6)),
            ("thin_b90", dict(thin), (0.9, 0.2, 0.6)),
            ("thin_b90_low", dict(thin), (0.9, 0.05, 0.3)),
        ]
        for name, patch, soot in grid:
            if soot:
                patch.update(dict(zip(SOOT_KEYS, soot)))
            cases[name] = measure(client, target, args.out, name, patch, base)
        # Must equal thick_b0: burnout is defined only for independent emission.
        cases["legacy_gate"] = measure(client, target, args.out, "legacy_gate",
            dict(off, independent_emission=False,
                 **dict(zip(SOOT_KEYS, (0.9, 0.2, 0.6)))), base)
        off_legacy = dict(off, independent_emission=False)
        cases["legacy_b0"] = measure(client, target, args.out, "legacy_b0", off_legacy, base)
        montage(args.out, ["thick_b0", "thick_b90", "thick_b90_low", "legacy_gate",
                           "thin_b0", "thin_b70", "thin_b90", "thin_b90_low"])
    finally:
        client.call("volume.set_material", target=target, settings=base)
        client.call("render.set_volume_march_lod",
                    camera_max_steps=lod.get("camera_max_steps", 64))
        client.call("render.set_realtime_denoiser", mode=denoiser)
        client.call("viewport.capture", enabled=False)
        restored = client.call("volume.get_material", target=target) == original
        print("RESTORED", restored, flush=True)
        report["restored"] = restored
        client.close()

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    cases = report["cases"]
    failures = [key for key, value in report["contract"].items() if not value]
    if not report["restored"]:
        failures.append("restore")
    if abs(cases["legacy_gate"]["cap_mean"] - cases["legacy_b0"]["cap_mean"]) > 0.003:
        failures.append("legacy_gate differs: burnout leaked into source-function emission")
    if not cases["thin_b90"]["cap_mean"] != cases["thin_b0"]["cap_mean"]:
        failures.append("soot burnout changed nothing")
    print("FAIL: " + "; ".join(failures) if failures else "PASS", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
