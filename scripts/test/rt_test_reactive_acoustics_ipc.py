"""Post-build external acceptance for physics/signal-driven spatial acoustics.

Requires a disposable geometry/audio-empty scene. Injects controlled flow signals;
this validates the shared reaction bridge, NOT an actual fluid solver or real-time
physical sound synthesis. No Studio launch, project save or build is performed.
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
    c = RtIpc()
    assert not c.call("scene.list_objects"), "Use a geometry-empty scene"
    initial = c.call("audio.list")
    assert not initial["emitters"] and not initial["clips"], "Use an audio-empty scene"
    camera = c.call("camera.get")["position"]
    telemetry = c.call("audio.telemetry_get")
    workspace = c.call("audio.workspace_get")
    source = reference = asset = wall = None
    material = "ReactiveAcoustics_" + str(time.time_ns())
    key = "@test/reactive_acoustics/" + str(time.time_ns())
    report = {"passed": False, "cases": [], "cleanup": [],
              "scope": "injected flow controls, live GPU paths and native PCM"}
    clear = [camera[0] - 8, camera[1], camera[2]]
    blocked = [camera[0] + 8, camera[1], camera[2]]

    def drive(position, value=1):
        return c.call("audio.signal_set", key=key, flow_kg_s=value * 2,
                      speed_mps=1, position=position, ttl_seconds=10)

    def read():
        result = c.call("batch", calls=[
            {"method": "audio.source_runtime", "params": {"id": source}},
            {"method": "audio.acoustics_preview_status"}, {"method": "audio.mixer_get"}])
        assert all("result" in item for item in result), result
        runtime, entries, mixer = (item["result"] for item in result)
        entry = next(e for e in entries if e["id"] == source)
        master = next(bus for bus in mixer["buses"] if bus["name"] == "Master")["meter"]
        return runtime, entry, master

    def wait_for(predicate, seconds=12):
        deadline = time.monotonic() + seconds
        while True:
            state = read()
            if predicate(*state):
                return state
            assert time.monotonic() < deadline, state
            time.sleep(0.03)

    def level(position, expected):
        drive(position)
        state = wait_for(lambda runtime, entry, meter:
            runtime["playing"] and runtime["position"] == position and
            entry.get("report", {}).get("source_position") == position and
            abs(entry["amplitude_gain"] - expected) < 1e-4)
        time.sleep(0.15)
        values = []
        for _ in range(12):
            values.append(read()[2]["rms"] ** 2)
            time.sleep(0.015)
        return math.sqrt(statistics.mean(values)), state

    try:
        c.call("audio.playback_update", mode="continuous")
        c.call("viewport.set_shading", mode="material")
        c.call("substance.derive", name=material, based_on="Stone")
        fields = {"acoustic_defined": True, "acoustic_scattering": 0}
        for band in (125, 250, 500, 1000, 2000, 4000):
            fields[f"acoustic_absorption_{band}"] = 0.2
            fields[f"acoustic_transmission_{band}"] = 0.04
        c.call("substance.set", name=material, fields=fields)
        wall = c.call("scene.add_primitive", type="cube", name=material, size=1)
        c.call("scene.set_transform", name=wall,
               translation=[camera[0] + 4, camera[1], camera[2]], scale=[0.3, 12, 12])
        c.call("substance.set_object", object=wall, binding={"substance": material})
        with tempfile.TemporaryDirectory(prefix="rt_reactive_acoustics_") as folder:
            tone = Path(folder) / "tone.wav"
            with wave.open(str(tone), "wb") as wav:
                wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(b"".join(struct.pack("<h", round(8000 * math.sin(
                    2 * math.pi * 1000 * i / 48000))) for i in range(48000)))
            asset = c.call("audio.asset_import", path=str(tone))["id"]
            authored = [camera[0] + 20, camera[1], camera[2]]
            source = c.call("audio.create", name="Reactive flow acoustic probe", asset_id=asset,
                spatial=True, loop=True, gain=0.15, rolloff=0, position=authored,
                reaction={"driver": "signal", "signal_key": key, "automatic": True,
                          "follow_position": True, "response_seconds": 0,
                          "input_min": 0, "input_max": 1})["id"]
            reference = c.call("audio.create", name="Manual independent probe", asset_id=asset,
                               spatial=True, position=clear, rolloff=0, gain=0.1)["id"]
            for voice in (source, reference):
                c.call("audio.acoustics_preview", id=voice, enabled=True, spectral=True)
            assert c.call("audio.telemetry_get") == telemetry, "Binding enabled collection"
            for params in ({}, {"id": source, "extra": 1}, {"id": "__missing__"}):
                ok, _ = c.try_call("audio.source_runtime", **params)
                assert not ok, params
            baseline, state = level(clear, 1)
            assert baseline > 0.001, state
            assert state[0]["position_origin"] == "reaction_world", state
            assert c.call("audio.get", id=source)["position"] == authored
            attenuated, state = level(blocked, 0.2)
            assert abs(attenuated / baseline - 0.2) < 0.035, (baseline, attenuated, state)
            report["cases"].append({"case": "automatic flow pose and blocked PCM",
                                    "ratio": attenuated / baseline, "state": state})
            # Parent transform must not transform an already-world-space measured position again.
            c.call("audio.update", id=source, parent=wall)
            _, state = level(blocked, 0.2)
            assert state[0]["position"] == blocked, state
            report["cases"].append({"case": "world-space following survives parent", "state": state})
            # One invalid/nonspatial preview must not suppress valid sources in the same batch.
            c.call("audio.update", id=reference, spatial=False)
            _, state = level(blocked, 0.2)
            entries = c.call("audio.acoustics_preview_status")
            manual = next(e for e in entries if e["id"] == reference)
            assert manual["status"] == "source_unavailable", manual
            assert state[1]["report"]["solver"]["state"] == "ready", state
            report["cases"].append({"case": "invalid source isolation", "manual": manual})
            seen = set()
            for index in range(18):
                position = [blocked[0], blocked[1] + index * 0.025, blocked[2]]
                drive(position)
                runtime, entry, _ = read()
                snapshot = entry.get("report", {})
                if snapshot.get("pose_policy") == "sampled_runtime_world":
                    seen.add(tuple(snapshot["source_position"]))
                    assert snapshot["result_age_ms"] <= 1000, snapshot
                time.sleep(0.025)
            assert len(seen) >= 2, ("Moving-source completion starvation", seen)
            report["cases"].append({"case": "moving snapshots complete", "positions": sorted(seen)})
            # Reflection acceptance uses the same controlled source and runtime position.
            c.call("scene.set_transform", name=wall,
                translation=[camera[0] + 4, camera[1] + 10, camera[2]], scale=[40, 0.3, 40])
            c.call("audio.acoustics_preview", id=source, enabled=True,
                   spectral=True, reflections=True)
            drive(blocked)
            state = wait_for(lambda runtime, entry, meter: bool(entry["reflection_paths"]))
            report["cases"].append({"case": "reactive reflection paths", "state": state})
            # Expiry stops automation; explicit Stop All still prevents automatic restarts.
            c.call("audio.signal_set", key=key, value=1, position=blocked, ttl_seconds=0.1)
            state = wait_for(lambda runtime, entry, meter: not runtime["playing"] and
                             runtime["effective_gain"] == 0)
            report["cases"].append({"case": "expiry stops source", "state": state})
            c.call("audio.stop_all")
            drive(blocked)
            time.sleep(0.2)
            assert not read()[0]["playing"]
            c.call("audio.resume_automatic")
            wait_for(lambda runtime, entry, meter: runtime["playing"])
            assert c.call("audio.telemetry_get") == telemetry
            # Free decoded voices before temporary test files are removed.
            for voice in (source, reference):
                c.call("audio.remove", id=voice)
            source = reference = None
            c.call("audio.asset_remove", id=asset)
            asset = None
        report["passed"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        actions = [("audio.remove", {"id": voice}) for voice in (source, reference) if voice]
        actions += [("audio.asset_remove", {"id": asset})] if asset else []
        actions += [("scene.delete", {"name": wall})] if wall else []
        actions += [("substance.remove", {"name": material}),
                    ("audio.workspace_update", workspace),
                    ("audio.playback_update", {"mode": initial["playback_mode"]})]
        for method, params in actions:
            ok, result = c.try_call(method, **params)
            report["cleanup"].append({"method": method, "ok": ok, "result": result})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        c.close()


if __name__ == "__main__":
    main()
