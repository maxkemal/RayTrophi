"""Game design save/load round trip in Studio (Paint/GameDesign, paint.wet_design).

The pastel workshop's paint surfaces, their settings, the splat rules and the
balloon demo are authored over IPC (the same calls the Studio Game Design panel
makes), saved into a COPY of the project, and the copy is reopened. PASS when
the reopened project restores exactly the same game design by itself.

Studio only (the Player cannot save). The workshop must already be open in
Studio, or pass --open to open it (that replaces the current scene: save first).
The original workshop.rtp is never written. The copy goes to --out (default: a
temp folder) and is left there so it can also be opened in the Player.

  python scripts/test/rt_game_design_roundtrip_ipc.py [--open] [--out DIR]
"""
import argparse
import json
import math
import tempfile
import time
from pathlib import Path

from rt_ipc import PIPE_NAME, RtIpc

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "RayTrophiStudio/assets/scenes/pastel-paint-workshop/workshop.rtp"
SURFACES = ["PaintFloor", "PaintBack", "PaintLeft", "PaintRight"]
BALLS = [f"PaintBall_{i:02d}" for i in range(12)]
# Distinctive, non-default values: a restored default would not match them.
SETTINGS = {"wetness": 0.9, "lifetime_seconds": 17.5, "diffusion": 0.13, "runoff": 0.77,
            "absorption": 0.04, "drip_head": 0.85, "paint_load": 0.93, "deposit_rate": 0.88,
            "pickup_rate": 0.21, "brush_radius": 6.5, "oil": False,
            "roughness_channel": True, "height_normal_channel": True,
            "wet_roughness": 0.11, "dry_roughness": 0.31, "height_contribution": 0.4,
            "normal_strength": 0.6}
SPLAT = {"consume_particles": True, "radius": 0.095, "amount": 0.9, "min_speed": 0.0,
         "facing_min": 0.0, "max_events_per_tick": 65536}
DEMO = {"seed": 417, "balloons": 12, "particles_per_balloon": 1800, "interval_min": 2.5,
        "interval_max": 5.0, "autostart": False, "visual_nodes": BALLS}


def same(a, b, path="design"):
    """Structural equality with float tolerance (float32 round trip)."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def wait_applied(ipc, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = ipc.call("paint.wet_design")
        if not state["pending"]:
            return state
        time.sleep(0.1)
    raise AssertionError("game design still pending after reopen")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", action="store_true", help="open the workshop first")
    parser.add_argument("--out", default=None, help="folder for the saved copy")
    args = parser.parse_args()
    out = Path(args.out or tempfile.mkdtemp(prefix="rt_game_design_"))
    out.mkdir(parents=True, exist_ok=True)
    copy = out / "workshop_game_design.rtp"
    ipc = RtIpc(PIPE_NAME)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)

    if args.open:
        ipc.call("project.open", path=str(ASSET))
    current = Path(ipc.call("project.path"))
    assert current.resolve() == ASSET.resolve(), f"open {ASSET} in Studio first (or --open)"

    # 1. Author the design through the panel's calls.
    ipc.call("paint.wet_clear_surfaces")
    for name in SURFACES:
        ipc.call("paint.wet_add_surface", object=name, resolution=512)
        ipc.call("paint.wet_set_settings", object=name, **SETTINGS)
    ipc.call("paint.wet_set_splat", **SPLAT)
    demo = ipc.call("paint.wet_demo", operation="configure", config=DEMO)
    check("1 demo prepared on existing surfaces (no surface list)",
          demo["prepared"] and "surfaces" not in demo["config"], demo.get("error", ""))
    surfaces_after = {s["object"]: s for s in ipc.call("paint.wet_status")["surfaces"]}
    lifetimes = {n: surfaces_after[n]["settings"]["lifetime_seconds"] for n in SURFACES}
    check("2 demo kept the authored surface settings (did not apply its own look)",
          all(math.isclose(v, 17.5, rel_tol=1e-5) for v in lifetimes.values()), str(lifetimes))

    before = ipc.call("paint.wet_design")
    design = before["design"]
    check("3 wet_design holds 4 surfaces + splat + demo",
          design and len(design["wet_paint"]["surfaces"]) == 4 and "balloon_demo" in design
          and "surfaces" not in design["balloon_demo"], json.dumps(design)[:200])

    # 2. Save a copy and reopen it.
    ipc.call("project.save", path=str(copy))
    raw = json.loads(copy.read_text(encoding="utf-8"))
    check("4 saved .rtp carries game_design", "game_design" in raw)
    if "game_design" in raw:
        try:
            same(design, raw["game_design"])
            check("5 file content == wet_design", True)
        except AssertionError as e:
            check("5 file content == wet_design", False, str(e))

    ipc.call("paint.wet_clear_surfaces")   # nothing may survive from before the reopen
    ipc.call("project.open", path=str(copy))
    after = wait_applied(ipc)
    check("6 apply report clean", after["status"].startswith("applied:"), after["status"])
    try:
        same(design, after["design"])
        check("7 reopened design == saved design", True)
    except AssertionError as e:
        check("7 reopened design == saved design", False, str(e))
    demo = ipc.call("paint.wet_demo")
    check("8 demo prepared after reopen, flat mesh balls",
          demo["prepared"] and demo["visual_mode"] == "flat_mesh"
          and all(b["visual_resolved"] for b in demo["balloons"]), demo.get("error", ""))

    # 3. The paint itself is not saved; a burst still paints after reload.
    burst = ipc.call("paint.wet_demo", operation="burst", config={"count": 1})
    time.sleep(2.0)
    coverage = ipc.call("paint.wet_coverage")
    check("9 a burst paints the restored surfaces", coverage["covered_texels"] > 0,
          f"covered {coverage['covered_texels']}, spawned {burst['events'][-1]['spawned']}")
    ipc.call("paint.wet_demo", operation="stop")

    failed = [c for c in checks if not c[1]]
    print(f"{len(checks) - len(failed)}/{len(checks)} PASS  copy: {copy}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
