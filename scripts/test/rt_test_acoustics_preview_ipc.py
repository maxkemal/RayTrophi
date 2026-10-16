"""External post-build direct-path and native PCM-meter acceptance.

Owns temporary walls/source/file; restores playback mode and removes test assets.
Does not save the project or clear undo history. A derived test material may remain
pinned by undo history. Output meters measure engine PCM, not speaker/microphone SPL.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import struct
import tempfile
import time
import wave
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    client = RtIpc()
    report = {"measurement": "native mixer PCM callback RMS and peak, before OS/device",
              "cases": [], "cleanup": [], "passed": False}
    initial = client.call("audio.list")
    assert not initial["emitters"] and not initial["clips"], "Use an audio-empty scene"
    assert not client.call("audio.acoustics_preview_status"), "Active preview exists"
    camera = client.call("camera.get")["position"]
    objects = client.call("scene.list_objects")
    render_settings = client.call("render.get_settings")
    mode = initial["playback_mode"]
    walls, source, asset = [], None, None
    material = "Acoustic acceptance " + str(time.time_ns())
    bands = (125, 250, 500, 1000, 2000, 4000)
    baseline = None

    def fields(transmissions, defined=True):
        return {"acoustic_defined": defined,
                **{f"acoustic_absorption_{b}": 0.2 for b in bands},
                **{f"acoustic_transmission_{b}": t for b, t in zip(bands, transmissions)}}

    def place(wall, offset, aside=False):
        client.call("scene.set_transform", name=wall,
                    translation=[camera[0] + offset, camera[1] + (20 if aside else 0.13),
                                 camera[2] + 0.27], scale=[0.3, 3, 3])

    def configure(enabled=True, model="object_barrier"):
        return client.call("audio.acoustics_preview", id=source, enabled=enabled,
                           boundary_mode=model, meters_per_unit=1)

    def sample(label, expected, status=None, hits=None):
        nonlocal baseline
        deadline = time.monotonic() + 12
        live = None
        while status and time.monotonic() < deadline:
            entries = client.call("audio.acoustics_preview_status")
            live = next((e for e in entries if e["id"] == source), None)
            if live and live["status"] == status and abs(live["amplitude_gain"] - expected) < 1e-5:
                break
            time.sleep(0.05)
        if status:
            assert live and live["status"] == status, (label, live)
            assert abs(live["amplitude_gain"] - expected) < 1e-5, (label, live)
            if hits is not None:
                path = live["report"]
                crossings = path.get("crossings", len(path.get("hits", [])))
                assert crossings == hits, (label, live)
                if "solver" in path:
                    expected_state = "failed" if status == "unknown_material" else "ready"
                    assert path["solver"]["state"] == expected_state, (label, live)
                    assert path["solver"]["cpu_bvh"] is False, (label, live)
                    assert path["solver"]["result_bytes_per_source"] == 64, (label, live)
        debug = client.call("audio.acoustics_debug", id=source)
        assert debug["id"] == source and debug["issues_gpu_query"] is False
        assert debug["range_is_hard_cutoff"] is False
        assert debug["hit_positions_available"] is False
        assert len(debug["source_position"]) == len(debug["listener_position"]) == 3
        assert debug["distance_scene_units"] >= 0
        time.sleep(0.4)
        readings = []
        for _ in range(16):
            buses = client.call("audio.mixer_get")["buses"]
            meter = next(b["meter"] for b in buses if b["name"] == "Master")
            assert meter["ready"] and meter["overload_blocks"] == 0, meter
            readings.append(meter)
            time.sleep(0.015)
        rms = math.sqrt(statistics.mean(m["rms"] ** 2 for m in readings))
        peak = max(m["peak"] for m in readings)
        if baseline is None:
            assert rms > 0.005 and peak > 0.01, "No native PCM signal"
            baseline = (rms, peak)
        ratio, peak_ratio = rms / baseline[0], peak / baseline[1]
        item = {"case": label, "expected_amplitude_ratio": expected,
                "rms": rms, "peak": peak, "rms_ratio": ratio, "peak_ratio": peak_ratio,
                "live": live}
        report["cases"].append(item)
        print(label, "RMS ratio", round(ratio, 6), "peak ratio", round(peak_ratio, 6), flush=True)
        assert abs(ratio - expected) < 0.025, item
        assert abs(peak_ratio - expected) < 0.002, item

    try:
        client.call("substance.derive", name=material, based_on="Wood (Oak)")
        client.call("substance.set", name=material, fields=fields([0.25] * 6))
        for index, offset in enumerate((4, 7)):
            wall = client.call("scene.add_primitive", type="cube",
                               name=f"AcousticAcceptanceWall{index}_{time.time_ns()}", size=1)
            walls.append(wall)
            place(wall, offset, True)
            client.call("substance.set_object", object=wall, binding={"substance": material})
        with tempfile.TemporaryDirectory(prefix="rt_acoustics_") as folder:
            tone = Path(folder) / "tone.wav"
            with wave.open(str(tone), "wb") as wav:
                wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(b"".join(struct.pack("<h", round(6000 * math.sin(
                    2 * math.pi * 1000 * n / 48000))) for n in range(48000)))
            asset = client.call("audio.asset_import", path=str(tone))["id"]
            source = client.call("audio.create", name="Acoustic acceptance tone", asset_id=asset,
                                 position=[camera[0] + 10, camera[1], camera[2]],
                                 spatial=True, rolloff=0, loop=True, gain=0.5)["id"]
            for invalid in ({}, {"id": 7}, {"id": "__missing_acoustic_source__"},
                            {"id": source, "extra": True}):
                try:
                    client.call("audio.acoustics_debug", **invalid)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(("Invalid debug parameters accepted", invalid))
            client.call("audio.select", id=source)
            client.call("render.set_settings", debug_view=15)
            state = client.call("render.get_settings")
            assert state["debug_view"] == 15 and state["debug_view_name"] == "acoustic_surface"
            visualizer = client.call("audio.acoustics_debug", id=source)["surface_visualizer"]
            assert visualizer["active"] and visualizer["source_selected"]
            assert visualizer["source_eligible"] and not visualizer["calibrated_spl"]
            assert visualizer["extra_gpu_readback_bytes"] == 0
            assert visualizer["model"] == "direct_solid_geometry_proxy"
            ok, _ = client.try_call("render.set_settings", debug_view=16)
            assert not ok and client.call("render.get_settings")["debug_view"] == 15
            client.call("render.set_settings", debug_view=render_settings["debug_view"])
            client.call("audio.playback_update", mode="continuous")
            client.call("audio.play", id=source)
            sample("baseline disabled", 1)
            configure()
            sample("clear", 1, "clear", 0)
            place(walls[0], 4)
            sample("one closed object", 0.5, "evaluated_object_barrier", 2)
            trace = client.call("audio.acoustics_trace", id=source, meters_per_unit=2,
                                boundary_mode="object_barrier")
            assert abs(trace["distance_meters"] - 20) < 1e-5, trace
            assert abs(trace["delay_seconds"] - 20 / 343) < 1e-7, trace
            report["scaled_trace"] = trace
            place(walls[1], 7)
            sample("two separate closed objects", 0.25, "evaluated_object_barrier", 4)
            place(walls[1], 7, True)
            configure(model="thin_surface")
            sample("closed cube as two thin boundaries", 0.25, "evaluated_thin_surface", 2)
            configure()
            client.call("substance.set", name=material, fields=fields([0.25] * 6, False))
            sample("unknown material bypass", 1, "unknown_material", 2)
            ts = [0.04, 0.09, 0.16, 0.25, 0.36, 0.49]
            client.call("substance.set", name=material, fields=fields(ts))
            sample("unequal energy bands", math.sqrt(sum(ts) / 6), "evaluated_object_barrier", 2)
            before = client.call("audio.acoustics_preview_status")
            for patch in ({"meters_per_unit": 0}, {"enabled": 1}, {"boundary_mode": "bogus"}):
                params = {"id": source, "enabled": True, **patch}
                ok, error = client.try_call("audio.acoustics_preview", **params)
                assert not ok, params
                after = client.call("audio.acoustics_preview_status")
                assert [(e["id"], e["meters_per_unit"], e["boundary_mode"])
                        for e in before] == [(e["id"], e["meters_per_unit"], e["boundary_mode"])
                                              for e in after]
                report["cases"].append({"case": "invalid config rejected", "params": patch,
                                         "error": error})
            client.call("substance.set", name=material, fields=fields([0] * 6))
            sample("zero transmission silence", 0, "evaluated_object_barrier", 2)
            configure(False)
            sample("disabled restores PCM level", 1)
            configure()
            client.call("audio.remove", id=source)
            source = None
            assert client.call("audio.acoustics_preview_status") == []
            report["cases"].append({"case": "source removal clears preview", "passed": True})
            client.call("audio.asset_remove", id=asset)
            asset = None
        report["passed"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        for method, params in (
            [("audio.remove", {"id": source})] if source else []
        ) + ([("audio.asset_remove", {"id": asset})] if asset else []) + [
            ("scene.delete", {"name": wall}) for wall in reversed(walls)
        ] + [("render.set_settings", {"debug_view": render_settings["debug_view"]}),
             ("audio.playback_update", {"mode": mode}),
             ("substance.remove", {"name": material})]:
            ok, result = client.try_call(method, **params)
            report["cleanup"].append({"method": method, "ok": ok, "result": result})
        ok, result = client.try_call("scene.list_objects")
        report["remaining_objects"] = result if ok else {"unavailable": result}
        report["original_objects"] = objects
        ok, result = client.try_call("audio.list")
        report["final_audio"] = result if ok else {"unavailable": result}
        report["note"] = "Playback mode restored; explicit-mode flag and next_id may advance. Undo history preserved."
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        client.close()


if __name__ == "__main__":
    main()
