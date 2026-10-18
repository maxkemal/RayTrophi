"""Post-build, external named-pipe acceptance. Run in an empty disposable scene.

Checks concurrent sources, spectral PCM contrast, verified reflection delay and
native mixer PCM changes. Does not launch Studio or compile anything.
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
    parser.add_argument("--sources", type=int, choices=range(2, 33), default=4)
    args = parser.parse_args()
    client = RtIpc()
    initial = client.call("audio.list")
    initial_workspace = client.call("audio.workspace_get")
    assert not initial["emitters"] and not initial["clips"], "Use an audio-empty scene"
    assert not client.call("scene.list_objects"), "Use a geometry-empty scene"
    assert not client.call("audio.status")["preview_muted"], "Unmute preview before testing"
    camera = client.call("camera.get")["position"]
    sources, assets, wall = [], [], None
    material = "AcousticPlayback_" + str(time.time_ns())
    report = {"passed": False, "cases": [], "cleanup": []}
    bands = (125, 250, 500, 1000, 2000, 4000)

    def preview(source, **patch):
        return client.call("audio.acoustics_preview", id=source, enabled=True,
                           spectral=True, reflections=False, **patch)

    def ready(source, predicate=lambda entry: entry["status"] != "pending"):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            entry = next(e for e in client.call("audio.acoustics_preview_status")
                         if e["id"] == source)
            if entry["status"] != "pending" and predicate(entry):
                assert entry["solver"]["state"] == "ready", entry
                return entry
            time.sleep(0.05)
        raise AssertionError(("Timed out waiting for current acoustic result", entry))

    def rms():
        time.sleep(0.25)
        values = []
        for _ in range(20):
            mixer = client.call("audio.mixer_get")
            master = next(bus for bus in mixer["buses"] if bus["name"] == "Master")
            values.append(master["meter"]["rms"] ** 2)
            time.sleep(0.02)
        return math.sqrt(statistics.mean(values))

    try:
        client.call("audio.playback_update", mode="continuous")
        client.call("substance.derive", name=material, based_on="Wood (Oak)")
        fields = {"acoustic_defined": True, "acoustic_scattering": 0}
        for band in bands:
            fields[f"acoustic_absorption_{band}"] = 0.2
            fields[f"acoustic_transmission_{band}"] = 0.7 if band <= 250 else 0
        client.call("substance.set", name=material, fields=fields)
        wall = client.call("scene.add_primitive", type="cube", name=material, size=1)
        client.call("substance.set_object", object=wall, binding={"substance": material})
        client.call("scene.set_transform", name=wall,
                    translation=[camera[0] + 5, camera[1], camera[2]], scale=[0.3, 8, 8])
        with tempfile.TemporaryDirectory(prefix="rt_acoustic_playback_") as folder:
            for frequency in (125, 4000):
                path = Path(folder) / f"{frequency}.wav"
                with wave.open(str(path), "wb") as wav:
                    wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                    wav.writeframes(b"".join(struct.pack("<h", round(8000 * math.sin(
                        2 * math.pi * frequency * n / 48000))) for n in range(48000)))
                assets.append(client.call("audio.asset_import", path=str(path))["id"])
            client.call("audio.acoustics_update", config={"enabled": True, "spectral": True,
                        "reflections": False, "diffuse": False, "meters_per_unit": 1,
                        "boundary_mode": "object_barrier"})
            for index in range(args.sources):
                sources.append(client.call("audio.create", name=f"Concurrent {index}",
                    asset_id=assets[index % 2], loop=True, spatial=True, rolloff=0,
                    gain=0.03, position=[camera[0] + 10, camera[1], camera[2]])["id"])
            # Saved scene defaults and source exceptions use the same core as UI.
            client.call("audio.acoustics_update", config={"enabled": True,
                        "spectral": True, "reflections": False})
            settings = client.call("audio.acoustics_get")
            assert settings["config"]["enabled"]
            assert all(e["policy"]["mode"] == "scene" and e["effective"]["enabled"]
                       for e in settings["sources"])
            client.call("audio.update", id=sources[0], acoustics={"mode": "custom",
                        "config": {"enabled": False}})
            client.call("audio.update", id=sources[1], acoustics={"mode": "off"})
            client.call("audio.acoustics_update", config={"reflections": True})
            settings = client.call("audio.acoustics_get")
            by_id = {e["id"]: e for e in settings["sources"]}
            assert not by_id[sources[0]]["effective"]["enabled"]
            assert not by_id[sources[1]]["effective"]["enabled"]
            assert client.call("audio.list")["acoustics"]["reflections"]
            before = client.call("audio.list")
            for method, params in (
                ("audio.acoustics_update", {"config": {"enabled": "yes"}}),
                ("audio.acoustics_update", {"config": {"unknown": 1}}),
                ("audio.update", {"id": sources[0], "acoustics": {"mode": "unknown"}}),
                ("audio.acoustics_reset_sources", {"scope": "group", "ids": ["missing"]})):
                ok, error = client.try_call(method, **params)
                assert not ok, (method, error)
                assert client.call("audio.list") == before
            client.call("audio.acoustics_reset_sources", scope="scene")
            assert all(e["policy"]["mode"] == "scene"
                       for e in client.call("audio.acoustics_get")["sources"])
            client.call("audio.acoustics_update", config={"enabled": False,
                        "reflections": False})
            assert not client.call("audio.acoustics_preview_status")
            client.call("audio.workspace_update", mixer_bus="Effects")
            assert client.call("audio.workspace_get")["mixer_bus"] == "Effects"
            ok, error = client.try_call("audio.workspace_update", mixer_bus="unknown")
            assert not ok and client.call("audio.workspace_get")["mixer_bus"] == "Effects"
            report["cases"].append({"case": "scene acoustic defaults, source exceptions, mixer focus"})
            missing = Path(folder) / "4000.wav"
            hidden = missing.with_suffix(".hidden")
            missing.rename(hidden)
            try:
                ok, error = client.try_call("audio.play_all")
                assert not ok, "Missing second asset should reject the prepared batch"
                assert not client.call("audio.status")["voices"], error
                report["cases"].append({"case": "preparation rollback", "error": error})
            finally:
                hidden.rename(missing)
            start = time.perf_counter()
            result = client.call("audio.play_all")
            report["prepare_and_dispatch_ms"] = (time.perf_counter() - start) * 1000
            assert set(result["ids"]) == set(sources), result
            time.sleep(0.2)
            voices = client.call("audio.status")["voices"]
            assert len(voices) == args.sources and all(v["playing"] for v in voices), voices
            cursors = [v["cursor_frames"] for v in voices]
            assert max(cursors) - min(cursors) < 2048, voices
            report["cases"].append({"case": "concurrent", "voices": voices, "rms": rms()})
            client.call("audio.acoustics_batch", scope="group", ids=sources,
                        action="configure", config={"enabled": True, "spectral": True})
            entries = [ready(source) for source in sources]
            assert len(entries) == args.sources
            report["cases"].append({"case": "concurrent spectral GPU batch",
                                    "sources": len(entries), "rms": rms()})
            revisions = {e["id"]: e["preview_revision"] for e in entries}
            authored = client.call("audio.list")
            for action in ("dry", "processed"):
                batch = client.call("audio.acoustics_batch", scope="group", ids=sources,
                                    action=action)
                assert batch["ids"] == sources
                assert all(e.get("bypassed", False) == (action == "dry")
                           for e in batch["entries"]), batch
                assert {e["id"]: e["preview_revision"] for e in batch["entries"]} == revisions
                assert client.call("audio.list") == authored, "A/B changed saved settings"
            for method, params in (
                ("audio.acoustics_batch", dict(scope="group", ids=sources + ["missing"],
                                              action="configure", config={"enabled": False})),
                ("audio.acoustics_batch", dict(scope="group", ids=sources,
                                              action="configure", config={"enabled": "yes"})),
                ("audio.transport_batch", dict(scope="group", ids=[sources[0], sources[0]],
                                              action="stop")),
            ):
                ok, error = client.try_call(method, **params)
                assert not ok, (method, params)
                assert {e["id"]: e["preview_revision"] for e in
                        client.call("audio.acoustics_preview_status")} == revisions
                assert all(v["playing"] for v in client.call("audio.status")["voices"])
            for action in ("pause", "play"):
                client.call("audio.transport_batch", scope="group", ids=sources, action=action)
                time.sleep(0.15)
                assert all(v["playing"] == (action == "play")
                           for v in client.call("audio.status")["voices"])
            report["cases"].append({"case": "scoped A/B, revision preservation, atomic rejection, transport"})
            for source in sources:
                client.call("audio.acoustics_preview", id=source, enabled=False)
            client.call("audio.stop_all")
            ratios = []
            for source in sources[:2]:
                client.call("audio.play", id=source)
                baseline = rms()
                assert baseline > 0.001, baseline
                preview(source)
                entry = ready(source)
                filtered = rms()
                ratios.append(filtered / baseline)
                report["cases"].append({"case": "spectral", "id": source,
                                        "ratio": ratios[-1], "entry": entry})
                client.call("audio.acoustics_preview", id=source, enabled=False)
                client.call("audio.stop", id=source)
            assert ratios[0] > 2 * ratios[1], ratios
            # Same-side source/listener, broad ceiling: finite, unobstructed image-source path.
            client.call("scene.set_transform", name=wall,
                        translation=[camera[0] + 5, camera[1] + 10, camera[2]],
                        scale=[40, 0.3, 40])
            for band in bands:
                fields[f"acoustic_transmission_{band}"] = 0
            client.call("substance.set", name=material, fields=fields)
            source = sources[0]
            client.call("audio.play", id=source)
            baseline = rms()
            client.call("audio.acoustics_preview", id=source, enabled=True,
                        spectral=True, reflections=True)
            entry = ready(source, lambda e: bool(e["reflection_paths"]))
            for path in entry["reflection_paths"]:
                expected = (path["distance_meters"] - entry["report"]["distance_meters"]) / 343
                assert abs(path["excess_delay_seconds"] - expected) < 1e-5, path
                assert 0 < expected <= 1, path
            reflected = rms()
            assert abs(reflected / baseline - 1) > 0.05, (baseline, reflected, entry)
            report["cases"].append({"case": "audible reflection", "baseline": baseline,
                                    "reflected": reflected, "entry": entry})
            # The existing ceiling fixture also supports same-side Lambert NEE.
            client.call("audio.acoustics_preview", id=source, enabled=True,
                        spectral=True, reflections=False, diffuse=True)
            zero = ready(source, lambda e: e.get("report", {}).get("diffuse", {}).get(
                "samples") == 64)
            assert not zero["reflection_paths"], zero
            fields["acoustic_scattering"] = 1
            client.call("substance.set", name=material, fields=fields)
            full = ready(source, lambda e: e.get("report", {}).get("diffuse", {}).get(
                "contributions", 0) > 0)
            assert 0 < len(full["reflection_paths"]) <= 8, full
            assert all(p["kind"] == "diffuse" for p in full["reflection_paths"]), full
            full_energy = sum(sum(p["energy"]) for p in full["reflection_paths"])
            fields["acoustic_scattering"] = 0.5
            client.call("substance.set", name=material, fields=fields)
            half = ready(source, lambda e: e.get("report", {}).get("diffuse", {}).get(
                "contributions", 0) > 0 and abs(sum(sum(p["energy"]) for p in
                e["reflection_paths"]) / full_energy - 0.5) < 1e-4)
            report["cases"].append({"case": "Lambert zero and linear scattering budget",
                                    "zero": zero, "full": full, "half": half})
            client.call("audio.update", id=source, cone_inner=90, cone_outer=90)
            before_cone = client.call("audio.acoustics_preview_status")
            try:
                ok, error = client.try_call("audio.acoustics_preview", id=source,
                                            enabled=True, diffuse=True)
                assert not ok and "omnidirectional" in error, error
                # The frame loop may publish source_unavailable independently;
                # the authoring rejection must not allocate a new preview revision.
                after_cone = client.call("audio.acoustics_preview_status")
                assert [e["preview_revision"] for e in before_cone] == [
                    e["preview_revision"] for e in after_cone]
            finally:
                client.call("audio.update", id=source, cone_inner=360, cone_outer=360)
            fields["acoustic_scattering"] = 0
            client.call("substance.set", name=material, fields=fields)
            for concurrent in sources:
                client.call("audio.acoustics_preview", id=concurrent, enabled=True,
                            spectral=True, reflections=True)
            client.call("audio.play_all")
            entries = [ready(concurrent, lambda e: bool(e["reflection_paths"]))
                       for concurrent in sources]
            voices = client.call("audio.status")["voices"]
            assert len(voices) == args.sources and all(v["playing"] for v in voices), voices
            report["cases"].append({"case": "concurrent reflected GPU batch",
                                    "sources": len(entries), "rms": rms()})
            client.call("audio.stop_all")
            client.call("audio.play", id=source)
            client.call("scene.set_transform", name=wall,
                        translation=[camera[0] + 10000, camera[1] + 10, camera[2]])
            cleared = ready(source, lambda e: not e["reflection_paths"])
            restored = rms()
            assert abs(restored / baseline - 1) < 0.03, (baseline, restored, cleared)
            report["cases"].append({"case": "removed candidate restores dry PCM",
                                    "rms": restored, "entry": cleared})
            before = client.call("audio.acoustics_preview_status")
            for patch in ({"spectral": 1}, {"reflections": "yes"}, {"diffuse": "yes"}):
                ok, error = client.try_call("audio.acoustics_preview", id=source,
                                            enabled=True, **patch)
                assert not ok and "boolean" in error, error
                assert client.call("audio.acoustics_preview_status") == before
            # Remove runtime before TemporaryDirectory tries deleting decoded source files.
            client.call("audio.stop_all")
            for source in sources:
                client.call("audio.remove", id=source)
            sources.clear()
            for asset in assets:
                client.call("audio.asset_remove", id=asset)
            assets.clear()
        report["passed"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        actions = [("audio.remove", {"id": source}) for source in sources]
        actions += [("audio.asset_remove", {"id": asset}) for asset in assets]
        actions += ([("scene.delete", {"name": wall})] if wall else [])
        actions += [("substance.remove", {"name": material}),
                    ("audio.acoustics_update", {"config": initial["acoustics"]}),
                    ("audio.workspace_update", {"mixer_bus": initial_workspace["mixer_bus"]}),
                    ("audio.playback_update", {"mode": initial["playback_mode"]})]
        for method, params in actions:
            ok, result = client.try_call(method, **params)
            report["cleanup"].append({"method": method, "ok": ok, "result": result})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        client.close()


if __name__ == "__main__":
    main()
