"""Post-build external acceptance for physics/signal-driven spatial acoustics.

Requires a disposable geometry/audio-empty scene. Injects controlled flow signals;
this validates the shared reaction bridge and procedural PCM with injected controls,
NOT an actual fluid solver or physical calibration. No Studio launch, save or build.
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
    duplicate_sources = []
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
        # Malformed requests must reject before typed JSON access or mutation.
        pristine = c.call("audio.list")
        malformed = [(method, {}) for method in (
            "get", "source_runtime", "update", "duplicate", "remove", "play", "pause",
            "stop", "asset_remove", "asset_import", "physical_impact",
            "physical_history_start", "physical_replay_start", "physical_bake_start",
            "preview_mute", "mixer_update", "signal_set", "thunder_update")]
        malformed += [("create", patch) for patch in (
            {"gain": "loud"}, {"name": None}, {"loop": 1}, {"reaction": []},
            {"reaction": {"driver": 1}}, {"reaction": {"automatic": "yes"}},
            {"reaction": {"input_max": "high"}}, {"position": [0, "x", 0]},
            {"generator": []}, {"generator_history": 1})]
        for method, params in malformed:
            ok, error = c.try_call("audio." + method, **params)
            assert not ok, (method, params, error)
            assert c.call("audio.list") == pristine, (method, params)
        report["cases"].append({"case": "malformed request rejection without author mutation",
                                "requests": len(malformed)})
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
            for params in ({}, {"id": None}, {"id": 1}, {"id": ""},
                           {"id": source, "extra": 1}, {"id": "__missing__"}):
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
        # Reuse this acceptance fixture for file-free continuous source generation.
        # These controls are injected, not measured solver contacts or splashes.
        for model in ("liquid_flow", "gas_flow", "friction", "rolling", "contact_activity",
                      "deformation", "fracture_activity"):
            source = c.call("audio.create", name="Procedural " + model,
                generator={"model": model, "seed": 0, "brightness": 0.5},
                loop=True, gain=0.15, spatial=True, rolloff=0, position=clear,
                reaction={"driver": "signal", "signal_key": key, "automatic": True,
                          "follow_position": True, "response_seconds": 0,
                          "input_min": 0, "input_max": 1})["id"]
            c.call("audio.acoustics_preview", id=source, enabled=True, spectral=True)
            drive(clear)
            state = wait_for(lambda runtime, entry, meter:
                             runtime["playing"] and meter["rms"] > 1e-6)
            assert state[0]["input_kind"] == "procedural", state
            before = c.call("audio.get", id=source)
            assert before["asset_id"] == "" and before["generator"]["version"] == 1
            for patch in ({"generator": {"model": "unknown"}},
                          {"generator": {"model": model, "seed": -1}},
                          {"generator": {"model": model, "brightness": 2}},
                          {"loop": False}):
                ok, error = c.try_call("audio.update", id=source, **patch)
                assert not ok, patch
                assert c.call("audio.get", id=source) == before
            c.call("audio.stop_all")
            drive(clear)
            time.sleep(0.15)
            assert not c.call("audio.source_runtime", id=source)["playing"]
            c.call("audio.resume_automatic")
            wait_for(lambda runtime, entry, meter: runtime["playing"])
            c.call("audio.signal_set", key=key, value=0, position=clear, ttl_seconds=0.1)
            wait_for(lambda runtime, entry, meter: not runtime["playing"])
            assert c.call("audio.telemetry_get") == telemetry
            assert not c.call("audio.list")["assets"]
            report["cases"].append({"case": "file-free procedural PCM/control", "model": model})
            c.call("audio.remove", id=source)
            source = None
        for resonator in ("wood", "metal", "stone", "glass"):
            source = c.call("audio.create", name="Impact " + resonator, loop=True,
                gain=0.15, position=clear, rolloff=0,
                generator={"model": "impact", "resonator": resonator,
                           "size_m": 0.1, "decay_seconds": 0.8})["id"]
            c.call("audio.acoustics_preview", id=source, enabled=True, spectral=True)
            event = c.call("audio.physical_impact", id=source, sequence=1, energy_j=0.5)
            assert event["accepted"] and not event["energy_calibrated"], event
            state = wait_for(lambda runtime, entry, meter:
                             runtime["playing"] and meter["rms"] > 1e-6)
            duplicate = c.call("audio.physical_impact", id=source, sequence=1, energy_j=0.5)
            assert not duplicate["accepted"] and duplicate["reason"] == "duplicate_or_out_of_order"
            for sequence, energy in ((-1, 1), (2, 0), (2, -1)):
                ok, error = c.try_call("audio.physical_impact", id=source,
                                      sequence=sequence, energy_j=energy)
                assert not ok, (sequence, energy)
            # Rejected events did not consume the next sequence number.
            assert c.call("audio.physical_impact", id=source, sequence=2, energy_j=0.1)["accepted"]
            c.call("audio.pause", id=source)
            assert not c.call("audio.physical_impact", id=source,
                              sequence=2, energy_j=0.1)["accepted"]
            assert not c.call("audio.source_runtime", id=source)["playing"]
            assert c.call("audio.physical_impact", id=source, sequence=3, energy_j=0.1)["accepted"]
            assert c.call("audio.source_runtime", id=source)["playing"]
            released = wait_for(lambda runtime, entry, meter:
                runtime.get("generator_runtime", {}).get("state") == "not_resident", seconds=15)
            assert not released[0]["playing"], released
            assert not c.call("audio.physical_impact", id=source,
                              sequence=3, energy_j=0.1)["accepted"]
            assert c.call("audio.source_runtime", id=source)["generator_runtime"]["state"] == "not_resident"
            assert c.call("audio.physical_impact", id=source, sequence=4, energy_j=0.1)["accepted"]
            report["cases"].append({"case": "explicit energy impact resonance/dedup",
                                    "resonator": resonator, "state": state})
            c.call("audio.remove", id=source)
            source = None
        # Shared dry DSP bake: runs at the consolidated acceptance gate only.
        source = c.call("audio.create", name="Physical bake probe", loop=True, gain=0.15,
                        generator={"model": "impact", "resonator": "metal"})["id"]
        with tempfile.TemporaryDirectory(prefix="rt_physical_bake_") as folder:
            hashes = []
            def bake_ready():
                deadline = time.monotonic() + 15
                while True:
                    job = c.call("audio.physical_bake_status")
                    if job["state"] not in ("queued", "running"):
                        assert job["state"] == "completed", job
                        return job
                    assert time.monotonic() < deadline, job
                    time.sleep(0.02)
            for index in range(2):
                path = Path(folder) / f"impact_{index}.wav"
                c.call("audio.physical_bake_start", id=source, path=str(path),
                       duration_seconds=0.1, events=[{"frame": 0, "sequence": 1, "energy_j": 0.1}])
                bake_ready()
                assert not path.exists(), "Bake published without explicit apply"
                applied = c.call("audio.physical_bake_apply")
                asset = applied["asset"]["id"]
                assert applied["state"] == "applied" and path.exists()
                assert applied["bake"]["frames"] == 4800 and applied["bake"]["peak"] > 0
                assert not applied["bake"]["acoustic_processing"]
                hashes.append(applied["bake"]["pcm_fnv1a64"])
                c.call("audio.asset_remove", id=asset)
                asset = None
            assert hashes[0] == hashes[1], hashes
            stale_path = Path(folder) / "stale.wav"
            c.call("audio.physical_bake_start", id=source, path=str(stale_path),
                   duration_seconds=0.1, events=[{"frame": 0, "sequence": 1, "energy_j": 0.1}])
            bake_ready()
            before = c.call("audio.physical_bake_status")
            ok, error = c.try_call("audio.physical_bake_start", id=source,
                path=str(Path(folder) / "invalid.wav"), duration_seconds=0.1,
                events=[{"frame": 4800, "sequence": 1, "energy_j": 0.1}])
            assert not ok and c.call("audio.physical_bake_status") == before
            c.call("audio.update", id=source, gain=0.16)
            assert c.call("audio.physical_bake_apply")["state"] == "stale"
            assert not stale_path.exists()
            stored = c.call("audio.update", id=source, generator_history={"version": 1,
                "duration_seconds": 0.1, "events": [{"frame": 0, "sequence": 1, "energy_j": 0.1}]})
            assert stored["generator_history"]["version"] == 1
            ok, error = c.try_call("audio.update", id=source, generator_history={"version": 99,
                "duration_seconds": 0.1})
            assert not ok and c.call("audio.get", id=source) == stored
            duplicate = c.call("audio.duplicate", id=source)
            duplicate_sources.append(duplicate["id"])
            assert duplicate["generator_history"] == stored["generator_history"]
            c.call("audio.remove", id=duplicate["id"])
            duplicate_sources.remove(duplicate["id"])
            stored_path = Path(folder) / "stored.wav"
            c.call("audio.physical_bake_start", id=source, path=str(stored_path))
            bake_ready()
            applied_stored = c.call("audio.physical_bake_apply")
            asset = applied_stored["asset"]["id"]
            assert applied_stored["bake"]["frames"] == 4800
            c.call("audio.asset_remove", id=asset)
            asset = None
            # Group mixing is stable under input ID permutation; crop prerolls DSP.
            duplicate = c.call("audio.duplicate", id=source)
            duplicate_sources.append(duplicate["id"])
            mix_hashes = []
            full_pcm = None
            for index, ids in enumerate(([source, duplicate["id"]],
                                         [duplicate["id"], source])):
                mix_path = Path(folder) / f"mix_{index}.wav"
                c.call("audio.physical_bake_start", scope="group", ids=ids, path=str(mix_path))
                bake_ready()
                mixed = c.call("audio.physical_bake_apply")
                asset = mixed["asset"]["id"]
                assert mixed["bake"]["source_count"] == 2
                assert math.isclose(mixed["bake"]["peak"],
                                    applied_stored["bake"]["peak"] * 2, rel_tol=1e-6)
                mix_hashes.append(mixed["bake"]["pcm_fnv1a64"])
                full_pcm = mix_path.read_bytes()[-4800 * 4:]
                c.call("audio.asset_remove", id=asset)
                asset = None
            assert mix_hashes[0] == mix_hashes[1]
            crop_path = Path(folder) / "mix_crop.wav"
            c.call("audio.physical_bake_start", scope="group",
                   ids=[source, duplicate["id"]], path=str(crop_path),
                   start_frame=1200, end_frame_exclusive=3600)
            bake_ready()
            cropped = c.call("audio.physical_bake_apply")
            asset = cropped["asset"]["id"]
            assert cropped["bake"]["frames"] == 2400
            assert crop_path.read_bytes()[-2400 * 4:] == full_pcm[1200 * 4:3600 * 4]
            c.call("audio.asset_remove", id=asset)
            asset = None
            c.call("audio.update", id=duplicate["id"], generator_history={"version": 1,
                   "duration_seconds": 0.05,
                   "events": [{"frame": 0, "sequence": 1, "energy_j": 0.1}]})
            tail_path = Path(folder) / "short_history_tail.wav"
            c.call("audio.physical_bake_start", scope="group",
                   ids=[source, duplicate["id"]], path=str(tail_path), start_frame=2400)
            bake_ready()
            tail = c.call("audio.physical_bake_apply")
            asset = tail["asset"]["id"]
            assert tail_path.read_bytes()[-2400 * 4:] == stored_path.read_bytes()[-2400 * 4:]
            c.call("audio.asset_remove", id=asset)
            asset = None
            stale_group_path = Path(folder) / "stale_group.wav"
            c.call("audio.physical_bake_start", scope="group",
                   ids=[source, duplicate["id"]], path=str(stale_group_path))
            bake_ready()
            before = c.call("audio.physical_bake_status")
            for patch in ({"ids": [source, source]}, {"start_frame": 4800},
                          {"end_frame_exclusive": 4801}):
                request = {"scope": "group", "ids": [source, duplicate["id"]],
                           "path": str(Path(folder) / "bad_group.wav")}
                request.update(patch)
                ok, error = c.try_call("audio.physical_bake_start", **request)
                assert not ok and c.call("audio.physical_bake_status") == before
            c.call("audio.update", id=duplicate["id"], gain=0.17)
            assert c.call("audio.physical_bake_apply")["state"] == "stale"
            assert not stale_group_path.exists()
            c.call("audio.remove", id=duplicate["id"])
            duplicate_sources.remove(duplicate["id"])
            report["cases"].append({"case": "group dry mix permutation and exact crop",
                                    "pcm_hashes": mix_hashes})
            # Same typed schedule runs in native replay; seek/preroll stays on the worker.
            c.call("audio.physical_replay_start", id=source, duration_seconds=2,
                   start_frame=1200, events=[{"frame": 0, "sequence": 1, "energy_j": 0.1},
                                             {"frame": 72000, "sequence": 2, "energy_j": 0.1}])
            deadline = time.monotonic() + 15
            while True:
                replay = c.call("audio.physical_replay_status")
                if replay["state"] not in ("queued", "running"):
                    assert replay["state"] == "completed", replay
                    break
                assert time.monotonic() < deadline, replay
                time.sleep(0.02)
            assert not c.call("audio.source_runtime", id=source)["playing"]
            applied_replay = c.call("audio.physical_replay_apply")
            assert applied_replay["state"] == "applied" and applied_replay["start_frame"] == 1200
            runtime = c.call("audio.source_runtime", id=source)
            assert runtime["generator_runtime"]["playback_input"] == "scheduled_history", runtime
            assert runtime["generator_runtime"]["history_frame"] >= 1200, runtime
            ok, error = c.try_call("audio.physical_impact", id=source, sequence=42, energy_j=0.1)
            assert not ok, error
            original_source = c.call("audio.get", id=source)
            ok, error = c.try_call("audio.update", id=source, pitch=1.5)
            assert not ok and c.call("audio.get", id=source) == original_source
            deadline = time.monotonic() + 15
            while c.call("audio.source_runtime", id=source)["generator_runtime"]["state"] != "not_resident":
                assert time.monotonic() < deadline, c.call("audio.physical_replay_status")
                time.sleep(0.03)
            c.call("audio.physical_replay_cancel")
            c.call("audio.physical_history_start", id=source, duration_seconds=1)
            c.call("audio.physical_impact", id=source, sequence=1, energy_j=0.1)
            time.sleep(0.12)
            history_path = Path(folder) / "history.wav"
            finished = c.call("audio.physical_history_finish", path=str(history_path), store=True)
            assert not finished["sample_exact_capture"] and finished["history"]["events"]
            assert finished["stored"] and c.call("audio.get", id=source)["generator_history"]["events"]
            bake_ready()
            applied = c.call("audio.physical_bake_apply")
            asset = applied["asset"]["id"]
            assert applied["state"] == "applied"
            c.call("audio.asset_remove", id=asset)
            asset = None
            c.call("audio.physical_bake_cancel")
            c.call("audio.physical_history_cancel")
            c.call("audio.remove", id=source)
            source = None
            assert not list(Path(folder).glob("rtp*.tmp")), "Bake leaked a temporary file"
            report["cases"].append({"case": "deterministic dry bake, stale rejection, sampled history",
                                    "pcm_hashes": hashes})
        report["passed"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        c.try_call("audio.physical_replay_cancel")
        c.try_call("audio.physical_bake_cancel")
        c.try_call("audio.physical_history_cancel")
        actions = [("audio.remove", {"id": voice}) for voice in (source, reference) if voice]
        actions += [("audio.remove", {"id": voice}) for voice in duplicate_sources]
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
