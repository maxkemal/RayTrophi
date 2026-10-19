"""Audible, external V2 smoke test using temporary sources and injected activity.

Does not run simulation, touch solver state or stop existing sources. Restores the
audio playback mode, removes only its sources, and lets its private signal expire.
This verifies native PCM, not real solver events or subjective naturalness.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from rt_ipc import RtIpc

MODELS = ("liquid_flow", "gas_flow", "friction", "rolling", "impact",
          "contact_activity", "deformation", "fracture_activity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--resonators", nargs="+", choices=("wood", "metal", "stone", "glass"),
                        default=["wood", "metal", "stone", "glass"])
    parser.add_argument("--character-revision", type=int, choices=(1, 2), default=2)
    parser.add_argument("--surface", choices=("organic", "mineral", "metal", "glass"), default="organic")
    parser.add_argument("--gain", type=float, default=0.35)
    args = parser.parse_args()
    assert math.isfinite(args.gain) and 0 < args.gain <= 1
    client = RtIpc()
    baseline = client.call("audio.list")
    mixer = client.call("audio.mixer_get")
    master = next(row for row in mixer["buses"] if row["name"] == "Master")
    assert not mixer["output_muted"] and not master["muted"] and master["gain"] > 0, (
        "Output/Master is muted; this demo does not change user volume or mute")
    old_mode = baseline["playback_mode"]
    key = "@test/natural_listen/" + str(time.time_ns())
    source = None
    report = {"passed": False, "utc": datetime.now(timezone.utc).isoformat(),
              "scope": "injected controls and native PCM, no measured solver events",
              "listening_acceptance": "awaiting_user_feedback", "cases": [], "cleanup": []}

    def wait(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            if args.stop_file and args.stop_file.exists():
                raise KeyboardInterrupt("Listen demo stopped")
            time.sleep(min(0.1, max(0, until - time.monotonic())))

    def sample():
        runtime = client.call("audio.source_runtime", id=source)
        analysis = client.call("audio.analysis_get", bus="Master")
        peak = runtime["generator_runtime"].get("block_peak", 0)
        assert math.isfinite(peak) and math.isfinite(analysis["peak_dbfs"])
        assert runtime["playing"], runtime
        assert analysis["ready"] and analysis["fresh"]
        return {"block_peak": peak, "master_peak_dbfs": analysis["peak_dbfs"],
                "master_rms_dbfs": analysis["rms_dbfs"], "end_frame": analysis["end_frame"],
                "clip_samples_window": analysis["clip_samples_window"]}

    try:
        client.call("audio.playback_update", mode="continuous")
        for model in args.models:
            print(f"LISTEN V2 R{args.character_revision} {model}", flush=True)
            source = client.call("audio.create", name="Listen V2 / " + model,
                                 gain=args.gain, spatial=False, bus="Master")["id"]
            recipe = {"version": 2, "character_revision": args.character_revision, "model": model, "seed": 17, "brightness": 0.6,
                      "texture": 0.65, "variation": 0.7, "body": 0.55}
            if model == "friction" and args.character_revision == 2:
                recipe["surface"] = args.surface
            params = {"id": source, "generator": recipe}
            if model != "impact":
                params["reaction"] = {"driver": "signal", "signal_key": key,
                                      "automatic": False, "response_seconds": 0.03,
                                      "gain_min": 1, "gain_max": 1}
            client.call("audio.physical_configure", **params)
            rows = []
            if model == "impact":
                for index, material in enumerate(args.resonators):
                    print("  IMPACT " + material, flush=True)
                    client.call("audio.physical_configure", id=source,
                                generator=dict(recipe, resonator=material,
                                               decay_seconds=0.6, size_m=0.1))
                    for hit in range(2):
                        client.call("audio.physical_impact", id=source,
                                    sequence=index * 2 + hit + 1, energy_j=0.1)
                        wait(0.15)
                        rows.append(dict(sample(), material=material))
                        wait(0.45)
            else:
                client.call("audio.signal_set", key=key, value=0, ttl_seconds=2)
                client.call("audio.play", id=source)
                for activity in (0.25, 0.6, 1):
                    client.call("audio.signal_set", key=key, value=activity, ttl_seconds=3)
                    wait(1)
                    rows.append(dict(sample(), activity=activity))
                    wait(1)
                client.call("audio.signal_set", key=key, value=0, ttl_seconds=1)
                wait(0.3)
            assert max(row["block_peak"] for row in rows) > 1e-8, (model, rows)
            assert all(b["end_frame"] > a["end_frame"] for a, b in zip(rows, rows[1:])), rows
            # Master can contain user sources; only require clean full-scale behavior in isolation.
            if not baseline["emitters"] and not baseline["clips"]:
                assert all(row["clip_samples_window"] == 0 for row in rows), rows
            report["cases"].append({"model": model, "character_revision": args.character_revision,
                                    "surface": recipe.get("surface"), "passed": True, "samples": rows})
            client.call("audio.stop", id=source)
            client.call("audio.remove", id=source)
            source = None
            wait(0.35)
        report["passed"] = True
    except KeyboardInterrupt as error:
        report["cancelled"] = str(error)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        if source is not None:
            for method in ("audio.stop", "audio.remove"):
                try:
                    client.call(method, id=source)
                    report["cleanup"].append({"method": method, "ok": True})
                except Exception as error:
                    report["cleanup"].append({"method": method, "error": str(error)})
        try:
            client.call("audio.signal_set", key=key, value=0, ttl_seconds=0.01)
            client.call("audio.playback_update", mode=old_mode)
            after = client.call("audio.list")
            for field in ("emitters", "assets", "clips", "tracks"):
                report["cleanup"].append({"field": field,
                    "preserved_ids": {row["id"] for row in after[field]} ==
                                     {row["id"] for row in baseline[field]}})
            report["cleanup"].append({"playback_mode_restored": after["playback_mode"] == old_mode})
        finally:
            client.close()
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("REPORT " + str(args.report), flush=True)


if __name__ == "__main__":
    main()
