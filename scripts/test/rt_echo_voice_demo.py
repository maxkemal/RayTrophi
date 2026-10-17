"""Real CC0 voice audition in the existing echo lab; run externally with Studio open.

--install adds separate sources and saves EchoRoomsHorrorVoices.rtp.
--room corridor|hard|soft selects a room; --dry bypasses scene acoustics.
Dependencies for preparing files: numpy, scipy, soundfile (demo-local .tools supported).
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

from rt_ipc import RtIpc

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "x64/Release/assets/acoustic_echo_lab"
PACK = ROOT / "RayTrophiStudio/assets/audio/cc0_horror_test"
SESSION = LAB / "horror_session.json"
PAGES = {
    "yelling": "https://opengameart.org/content/male-gruntyelling-sounds",
    "moans": "https://opengameart.org/content/undead-moans",
}


def prepare():
    sys.path.insert(0, str(LAB / ".tools"))
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf
    PACK.mkdir(parents=True, exist_ok=True)
    originals = PACK / "originals"
    originals.mkdir(exist_ok=True)
    files = {}
    provenance = []
    for archive, member, author in (
        ("yelling", "yelling sounds/yell1.wav", "HaelDB"),
        ("yelling", "yelling sounds/1yell3.wav", "HaelDB"),
        ("moans", "undead-1.ogg", "AntumDeluge"),
        ("moans", "undead-3.ogg", "AntumDeluge"),
    ):
        with zipfile.ZipFile(LAB / "voice_originals" / (archive + ".zip")) as z:
            raw = z.read(member)
        (originals / Path(member).name).write_bytes(raw)
        samples, rate = sf.read(io.BytesIO(raw), always_2d=True)
        samples = samples.mean(axis=1)
        # Retain quiet edges around the event; no denoising, pitch effect or reverb.
        active = np.flatnonzero(np.abs(samples) > np.max(np.abs(samples)) * 0.008)
        samples = samples[max(0, active[0] - int(rate * 0.08)):
                          min(len(samples), active[-1] + int(rate * 0.12))]
        samples = resample_poly(samples, 160, 147) if rate == 44100 else samples
        assert rate in (44100, 48000)
        samples *= 0.7 / np.max(np.abs(samples))
        fade = min(144, len(samples) // 2)
        samples[:fade] *= np.linspace(0, 1, fade)
        samples[-fade:] *= np.linspace(1, 0, fade)
        files[Path(member).name] = samples
        provenance.append({"original": Path(member).name, "author": author,
                           "source_page": PAGES[archive], "license": "CC0-1.0",
                           "license_verified_date": "2026-10-17",
                           "sha256": hashlib.sha256(raw).hexdigest()})
    manifest = {"pack": "CC0 human voice echo tests", "originals": provenance,
                "transformation": "Mono mean; trim quiet edges; 48 kHz; peak 0.7; 3 ms fades; "
                                  "24-second cycle with silence. No added echo or reverb.",
                "assets": []}
    for name, members, onsets in (
        ("human_shouts", ["yell1.wav", "1yell3.wav"], [1, 13]),
        ("human_moans_a", ["undead-1.ogg", "undead-3.ogg"], [1, 13]),
        ("human_moans_b", ["undead-3.ogg", "undead-1.ogg"], [7, 19]),
    ):
        pcm = np.zeros(24 * 48000)
        for member, onset in zip(members, onsets):
            voice = files[member]
            start = onset * 48000
            pcm[start:start + len(voice)] = voice
        path = PACK / (name + "_spaced_mono.wav")
        sf.write(path, pcm, 48000, subtype="PCM_16")
        manifest["assets"].append({"name": name, "file": path.name,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "members": members, "onsets_seconds": onsets})
    with zipfile.ZipFile(LAB / "voice_originals/moans.zip") as z:
        (PACK / "CC0-1.0.txt").write_bytes(z.read("LICENSE.txt"))
    (PACK / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--room", choices=("corridor", "hard", "soft"), default="corridor")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.install:
        assert not SESSION.exists(), "Already installed; omit --install"
        prepare()
    c = RtIpc()
    try:
        if args.install:
            state = {"before": {"audio": c.call("audio.list"), "camera": c.call("camera.get"),
                                "status": c.call("audio.status"), "project": c.call("project.path"),
                                "preview": c.call("audio.acoustics_preview_status")},
                     "assets": {}, "sources": {}}

            def remember():
                SESSION.write_text(json.dumps(state, indent=2), encoding="utf-8")

            remember()
            for name in ("human_shouts", "human_moans_a", "human_moans_b"):
                state["assets"][name] = c.call("audio.asset_import",
                    path=str(PACK / (name + "_spaced_mono.wav")), license="CC0-1.0",
                    source=PAGES["yelling" if name == "human_shouts" else "moans"],
                    author="HaelDB" if name == "human_shouts" else "AntumDeluge",
                    tags=["voice", "horror", "mono", "echo-test"])["id"]
                remember()
            for room, x in (("hard", 100), ("soft", 220), ("corridor", 340)):
                state["sources"][room] = []
                first = "human_shouts" if room == "corridor" else "human_moans_a"
                for kind, position in ((first, [x, 2.8, 12]),
                                       ("human_moans_b", [x - 2, 2.8, -20])):
                    source = c.call("audio.create", name=f"VoiceLab {room} {kind}",
                        asset_id=state["assets"][kind], position=position, spatial=True,
                        loop=True, gain=0.55, min_distance=8, max_distance=180,
                        rolloff=0.6, bus="Effects")["id"]
                    state["sources"][room].append(source)
                    remember()
        else:
            state = json.loads(SESSION.read_text(encoding="utf-8"))
        audio = c.call("audio.list")
        names = {e["id"]: e["name"] for e in audio["emitters"]}
        for ids in state["sources"].values():
            assert all(names.get(source, "").startswith("VoiceLab ") for source in ids)
        c.call("audio.stop_all")
        for bus in ("Master", "Effects"):
            c.call("audio.mixer_update", bus=bus, gain=1, muted=False, solo=False)
        c.call("audio.preview_mute", muted=False)
        c.call("audio.playback_update", mode="continuous")
        x = {"hard": 100, "soft": 220, "corridor": 340}[args.room]
        c.call("camera.set_position", position=[x, 2.8, -4])
        c.call("camera.set_target", target=[x, 2.8, 20])
        c.call("camera.set_fov", fov=75)
        c.call("viewport.set_shading", mode="material")
        c.call("render.set_settings", debug_view=0)
        for source in state["sources"][args.room]:
            c.call("audio.acoustics_preview", id=source, enabled=not args.dry,
                   spectral=True, reflections=True)
        if not args.dry:
            deadline = time.monotonic() + 15
            while True:
                entries = {e["id"]: e for e in c.call("audio.acoustics_preview_status")}
                if all(entries[source].get("solver", {}).get("state") == "ready"
                       for source in state["sources"][args.room]):
                    break
                assert time.monotonic() < deadline, entries
                time.sleep(0.1)
        c.call("audio.select", id=state["sources"][args.room][0])
        c.call("audio.mixer_reset_peaks")
        result = c.call("batch", calls=[{"method": "audio.play", "params": {"id": source}}
                                        for source in state["sources"][args.room]])
        assert all("result" in item for item in result), result
        time.sleep(10)
        report = {"room": args.room, "dry": args.dry, "status": c.call("audio.status"),
                  "preview": c.call("audio.acoustics_preview_status"),
                  "mixer": c.call("audio.mixer_get")}
        (LAB / ("voices_" + args.room + ("_dry" if args.dry else "_wet") + ".json")).write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        if args.install:
            c.call("project.save", path=str(LAB / "EchoRoomsHorrorVoices.rtp"))
        print(json.dumps({"room": args.room, "dry": args.dry,
            "sources": [{"id": e["id"], "status": e["status"],
                         "delays_ms": [round(p["excess_delay_seconds"] * 1000, 1)
                                       for p in e["reflection_paths"]]}
                        for e in report["preview"] if e["id"] in state["sources"][args.room]],
            "meters": [{"bus": b["name"], "peak_hold_dbfs": b["meter"]["peak_hold_dbfs"],
                        "overload_blocks": b["meter"]["overload_blocks"]}
                       for b in report["mixer"]["buses"] if b["name"] in ("Master", "Effects")]},
            indent=2))
    finally:
        c.close()


if __name__ == "__main__":
    main()
