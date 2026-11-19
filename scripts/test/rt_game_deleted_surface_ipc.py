"""After rebuild, run externally with a disposable Studio game scene open.

Needs the four BalloonPaint droplet systems. Adds/deletes one temporary cube;
does not save the project, reset paint or start the game. Leaves undo history.
"""
import time
import uuid

from rt_ipc import RtIpc


def main():
    ipc = RtIpc()
    name = ipc.call("scene.add_primitive", type="cube",
                    name="DeletedPaintProbe_" + uuid.uuid4().hex[:8], size=1.0)
    deleted = False
    try:
        ipc.call("paint.wet_add_surface", object=name, resolution=512)
        ipc.call("paint.wet_demo", operation="set_collider",
                 config={"object": name, "mode": "obb"})
        before = ipc.call("paint.wet_demo", operation="status")
        assert any(c["object"] == name and c["in_all_systems"]
                   for c in before["colliders"]), "fixture collider missing"
        ipc.call("scene.delete", name=name)
        deleted = True
        time.sleep(0.25)
        status = ipc.call("paint.wet_status")
        assert all(s["object"] != name for s in status["surfaces"]), status
        demo = ipc.call("paint.wet_demo", operation="status")
        assert all(c["object"] != name for c in demo["colliders"]), demo
        design = ipc.call("paint.wet_design")["design"]
        assert all(s["object"] != name
                   for s in design["wet_paint"]["surfaces"]), design
        print("PASS: deleted surface and collider absent from runtime and saved design")
    finally:
        ipc.call("paint.wet_demo", operation="set_collider",
                 config={"object": name, "mode": "none"})
        if any(s["object"] == name for s in ipc.call("paint.wet_status")["surfaces"]):
            ipc.call("paint.wet_remove_surface", object=name)
        if not deleted:
            ipc.call("scene.delete", name=name)


if __name__ == "__main__":
    main()
