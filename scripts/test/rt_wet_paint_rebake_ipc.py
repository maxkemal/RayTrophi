"""A paint surface follows its object when it is moved (geometry re-bake).

Studio, pastel workshop open (or any scene with a wet paint surface named by
--object, facing +Z by default). Moves the object up 1.5 m, dabs where it is NOW,
checks the dab landed and paint.wet_status geometry_rebakes rose, then moves it
back. Before the fix the dab at the new place painted nothing (positions were
baked once, at paint.wet_add_surface).

  python scripts/test/rt_wet_paint_rebake_ipc.py [--object PaintBack] [--normal 0 0 1]
"""
import argparse
import time

from rt_ipc import PIPE_NAME, RtIpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", default="PaintBack")
    parser.add_argument("--normal", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    args = parser.parse_args()
    ipc = RtIpc(PIPE_NAME)
    checks = []

    def check(name, ok, detail=""):
        checks.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)

    def covered():
        for s in ipc.call("paint.wet_coverage")["surfaces"]:
            if s["object"] == args.object:
                return s["covered_texels"]
        raise SystemExit(f"{args.object} is not a wet paint surface")

    info = ipc.call("scene.object_info", name=args.object)
    top = info["world_max"]
    centre = info["world_center"]
    original = ipc.call("scene.get_world_transform", name=args.object)
    moved = [original["translation"][0], original["translation"][1] + 1.5, original["translation"][2]]
    rebakes = ipc.call("paint.wet_status")["geometry_rebakes"]
    try:
        ipc.call("scene.set_transform", name=args.object, translation=moved)
        time.sleep(1.5)   # the geometry check runs twice a second
        status = ipc.call("paint.wet_status")
        check("1 surface re-baked after the move", status["geometry_rebakes"] > rebakes,
              f"{rebakes} -> {status['geometry_rebakes']}")
        # Just under the moved top: above the old top, so only a re-baked atlas has texels here.
        point = [centre[0], top[1] + 1.5 - 0.3, centre[2]]
        before = covered()
        ipc.call("paint.wet_dab", position=point, radius=0.25, color=[1.0, 0.0, 1.0], amount=1.0,
                 normal=args.normal)
        time.sleep(0.6)
        after = covered()
        check("2 dab above the OLD top lands on the moved surface", after > before,
              f"{before} -> {after} at y={point[1]:.2f}")
    finally:
        ipc.call("scene.set_transform", name=args.object, translation=original["translation"])
        time.sleep(1.5)
    status = ipc.call("paint.wet_status")
    check("3 moving back re-bakes again", status["geometry_rebakes"] >= rebakes + 2,
          str(status["geometry_rebakes"]))
    print(f"{sum(checks)}/{len(checks)} PASS")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
