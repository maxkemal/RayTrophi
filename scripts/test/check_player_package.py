"""Read-only package integrity check; never launches the Player or a C++ build."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest().upper()


def check(root):
    root = root.resolve(strict=True)
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8-sig"))
    scenes = json.loads((root / "player_scenes.json").read_text(encoding="utf-8-sig"))["scenes"]
    entries = {(entry["scene"], entry["id"]): entry for entry in manifest["audio_assets"]}
    count = 0
    for item in scenes:
        scene = (root / item["path"]).resolve(strict=True)
        if not scene.is_relative_to(root):
            raise ValueError("scene path escapes package")
        content = json.loads(scene.read_text(encoding="utf-8-sig"))
        if not manifest["scene_audio"]:
            continue
        for asset in content.get("audio", {}).get("assets", []):
            if not asset.get("path"):
                continue
            audio = (scene.parent / asset["path"]).resolve(strict=True)
            if not audio.is_relative_to(scene.parent):
                raise ValueError("audio path escapes packaged scene")
            entry = entries[(scene.name, asset["id"])]
            if sha256(audio) != entry["sha256"]:
                raise ValueError("audio hash mismatch: " + asset["id"])
            count += 1
    if count != len(entries):
        raise ValueError("manifest and scene audio counts differ")
    if manifest["perception"]:
        runtime = root / "perception"
        model_manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
        if sha256(runtime / "models/pose_landmarker.task").lower() != model_manifest["model_sha256"]:
            raise ValueError("perception model hash mismatch")
        for file in ("python/python.exe", "bridge/bridge.py", "bridge/game_bridge.py"):
            if not (runtime / file).is_file():
                raise FileNotFoundError(file)
    for shader in ("player_depth_card.spv", "player_depth_card_frag.spv"):
        data = (root / "shaders" / shader).read_bytes()
        if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
            raise ValueError("invalid shader: " + shader)
    return {"status": "PASS", "scenes": len(scenes), "audio_assets": count,
            "perception": manifest["perception"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", nargs="?", type=Path, default=Path("dist/RayTrophiPlayer"))
    print(json.dumps(check(parser.parse_args().package)))
