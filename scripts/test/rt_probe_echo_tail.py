"""External native-PCM dry/wet late-tail check for the saved echo lab."""
import json
from pathlib import Path
import statistics
import time

from rt_ipc import RtIpc


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / "x64/Release/assets/acoustic_echo_lab"
    state = json.loads((folder / "session.json").read_text())
    source = state["sources"]["hard"][0]
    c = RtIpc()
    report = {"measurement": "native mixer callback PCM; late window 190..450 ms",
              "source": source, "arms": {}, "passed": False}
    try:
        for ids in state["sources"].values():
            for voice in ids:
                c.call("audio.stop", id=voice)
        c.call("camera.set_position", position=state["rooms"]["hard"]["listener"])
        for wet in (False, True):
            c.call("audio.stop", id=source)
            c.call("audio.acoustics_preview", id=source, enabled=wet,
                   spectral=True, reflections=True)
            if wet:
                deadline = time.monotonic() + 15
                while True:
                    entry = next(e for e in c.call("audio.acoustics_preview_status")
                                 if e["id"] == source)
                    if entry.get("solver", {}).get("state") == "ready" and entry["reflection_paths"]:
                        report["reflection_paths"] = entry["reflection_paths"]
                        break
                    assert time.monotonic() < deadline, entry
                    time.sleep(0.05)
            c.call("audio.play", id=source)
            rows = []
            deadline = time.monotonic() + 7
            while time.monotonic() < deadline:
                batch = c.call("batch", calls=[{"method": "audio.status"},
                                                {"method": "audio.mixer_get"}])
                status, mixer = (item["result"] for item in batch)
                voice = next(v for v in status["voices"] if v["id"] == source)
                master = next(b for b in mixer["buses"] if b["name"] == "Master")["meter"]
                rows.append({"phase_seconds": voice["cursor_frames"] / 48000,
                             "rms": master["rms"], "peak": master["peak"]})
                time.sleep(0.01)
            late = [row["rms"] ** 2 for row in rows if 0.19 <= row["phase_seconds"] <= 0.45]
            assert late, "No samples landed in the late window"
            report["arms"]["wet" if wet else "dry"] = {
                "late_mean_square": statistics.mean(late), "late_samples": len(late), "rows": rows}
        dry = report["arms"]["dry"]["late_mean_square"]
        wet = report["arms"]["wet"]["late_mean_square"]
        report["late_energy_ratio"] = wet / max(dry, 1e-20)
        assert wet > max(dry * 10, 1e-10), (dry, wet)
        report["passed"] = True
    finally:
        c.call("audio.stop", id=source)
        c.call("batch", calls=[{"method": "audio.play", "params": {"id": voice}}
                               for voice in state["sources"]["hard"]])
        path = root / "docs/dev/acoustics_echo_tail_20261017.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        c.close()
    print(json.dumps({key: value for key, value in report.items() if key != "arms"}, indent=2))


if __name__ == "__main__":
    main()
