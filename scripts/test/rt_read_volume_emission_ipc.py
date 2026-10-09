"""Read volume emission packets without editing the scene or viewport mode."""
import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="snapshot")
    args = parser.parse_args()
    client = RtIpc()
    try:
        snapshot = client.call("render.volume_slots")
    finally:
        client.close()
    snapshot["capture_tag"] = args.tag
    out = Path(__file__).resolve().parents[2] / "docs/dev/volume_emission_snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    print(f"Saved: {out}")
    slots = [s for b in snapshot.get("backends", []) for s in b.get("slots", [])]
    if slots and any("emission_mode" not in s for s in slots):
        print("Running build lacks emission diagnostics; rebuild C++ to expose them.")


if __name__ == "__main__":
    main()
