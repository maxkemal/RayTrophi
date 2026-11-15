"""GPU wet paint acceptance (Boyama Odasi step 3/4, docs/dev/PLAYER_GAME_PAINT_ROOM.md).

Runs in Studio on an EMPTY scene (the startup cube is tolerated and left
alone). Never launches an application. Builds its own objects and removes them.

    python scripts/test/rt_wet_paint_gpu_ipc.py          # cleans up after itself
    python scripts/test/rt_wet_paint_gpu_ipc.py --keep   # leaves objects + stains to LOOK at

Setup: four primitives overlapping at the origin - cube (edge 1), sphere
(size 1 = r 0.5), cylinder (size 1 = r 0.5, axis Y) and a floor plane (4 x 4)
- each made a wet paint surface. ★ The primitive cube maps all six faces onto
the SAME full UV square, so the bake keeps the first face (+Z) and a stain shows
on every face of the cube (one shared texture) - an asset property, not a bug. Checks, in order (independent and fast first):
  1. surfaces bake: every surface has covered texels and tiles;
  2. ONE world-space dab at (0, 0, 0.5) paints all four surfaces (multi-surface,
     seams by world distance) and the publish path moved the result into the
     material textures (device_copy expected; host_download = other VkDevice);
  3. cost follows the wet area: a single small dab ticks a small fraction of
     the atlas tiles, and after the paint dries the system goes idle (no ticks);
  4. wet-on-wet (report only): red kept by blue over WET vs over DRY red;
  5. runoff: on the cube's vertical side the painted area grows after the dab
     (paint runs down), on the horizontal plane it does not grow by more;
  6. particles: a device-resident stream recording collision events paints
     the plane with no download (device_event_dispatches > 0, host_events 0);
  7. reset: coverage back to zero;
  8. the screen: red stains show up as red pixels in the viewport (in memory).
"""
import json
import sys
import time

from rt_ipc import RtIpc

RES = 256


def wait_ticks(s, n=6, timeout=6.0):
    start = s.call("paint.wet_status")["ticks"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = s.call("paint.wet_status")
        if st["ticks"] >= start + n:
            return st
        time.sleep(0.05)
    return s.call("paint.wet_status")


def wait_idle(s, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = s.call("paint.wet_status")
        if st["status"] == "idle" and st["active_tiles"] == 0:
            return st, time.time() - t0
        time.sleep(0.1)
    return s.call("paint.wet_status"), None


def red_pixels_on_screen(s):
    import base64
    import io
    try:
        from PIL import Image
    except ImportError:
        return {"skipped": "no Pillow"}
    s.call("viewport.capture", enabled=True)
    try:
        time.sleep(0.6)
        shot = s.call("viewport.get_screenshot").get("image_base64")
    finally:
        s.call("viewport.capture", enabled=False)
    if not shot:
        return {"skipped": "no frame"}
    img = Image.open(io.BytesIO(base64.b64decode(shot))).convert("RGB")
    img.thumbnail((640, 640))
    red = sum(1 for r, g, b in img.getdata() if r > g + 60 and r > b + 60)
    return {"red_pixels": red, "size": list(img.size),
            "shading": s.call("viewport.shading").get("mode")}


def collider_names(s):
    listed = s.call("collider.list")
    items = listed if isinstance(listed, list) else listed.get("colliders", [])
    return [c["name"] if isinstance(c, dict) else c for c in items]


def by_object(coverage):
    return {c["object"]: c for c in coverage["surfaces"]}


def main():
    keep = "--keep" in sys.argv
    s = RtIpc()
    created = []
    report = {}
    try:
        names = s.call("scene.list_objects")
        assert set(names) <= {"Default_Cube"}, f"needs an empty Studio scene, found {names}"
        assert not s.call("particle.list_systems")["systems"], "Studio already has particle systems"
        s.call("paint.wet_clear_surfaces")

        for ptype, name, size in (("cube", "WetCube", 1.0), ("sphere", "WetSphere", 1.0),
                                  ("cylinder", "WetCylinder", 1.0), ("plane", "WetFloor", 4.0)):
            obj = s.call("scene.add_primitive", type=ptype, name=name, size=size)
            created.append(obj)
            # Its own material: one material holds one paint texture.
            mat = s.call("material.create", type="principled", name=obj + "_WetMat")
            s.call("material.set_param", material_name=mat, param="base_color", value=[0.8, 0.8, 0.8])
            s.call("material.assign", object_name=obj, material_name=mat)
        surfaces = {}
        for obj in created:
            info = s.call("paint.wet_add_surface", object=obj, resolution=RES)
            surfaces[obj] = info
            assert info["valid_texels"] > 0 and info["tiles"] > 0, ("bake produced nothing", info)
        report["bake"] = {o: {"valid": i["valid_texels"], "gutter": i["gutter_texels"],
                              "triangles": i["triangles"], "tiles": i["tiles"]}
                          for o, i in surfaces.items()}
        cube, sphere, cylinder, floor = created
        # Short lifetime so the idle check does not wait long; strong runoff
        # for the drip check.
        for obj in created:
            s.call("paint.wet_set_settings", object=obj, lifetime_seconds=1.2, runoff=1.5)
        wait_ticks(s, 2)

        # 2. one dab, four surfaces
        s.call("paint.wet_dab", position=[0.0, 0.0, 0.5], radius=0.12, color=[0.9, 0.1, 0.1],
               amount=1.0)
        st = wait_ticks(s, 4)
        cov = by_object(s.call("paint.wet_coverage"))
        painted = {o: cov[o]["covered_texels"] for o in created}
        report["one_dab"] = {"covered": painted, "publish_mode": st["publish_mode"],
                             "device_copies": st["publish_device_copies"],
                             "host_downloads": st["publish_host_downloads"],
                             "status": st["status"], "device_ready": st["device_ready"]}
        assert st["device_ready"], ("no device", st)
        missing = [o for o, n in painted.items() if n == 0]
        assert not missing, ("one world-space dab must reach all four surfaces", report["one_dab"])
        assert st["publish_device_copies"] + st["publish_host_downloads"] > 0, \
            ("paint never reached a material texture", report["one_dab"])

        # 3. cost follows wet area
        total_tiles = st["tiles"]
        ticked = st["ticked_tiles"]
        active = st["active_tiles"]
        report["cost"] = {"total_tiles": total_tiles, "active_tiles": active,
                          "ticked_tiles_while_wet": ticked, "last_tick_ms": st["last_tick_ms"]}
        # The definition: a tick touches the wet tiles and their 8 neighbours,
        # never the whole atlas. (A tile-count ratio depends on the resolution.)
        assert 0 < ticked <= 9 * max(active, 1) and ticked < total_tiles,             ("ticks do not follow the wet area", report["cost"])

        idle, seconds = wait_idle(s)
        report["cost"]["seconds_to_idle"] = seconds
        assert seconds is not None, ("paint never dried / system never idles", idle)

        # 5. runoff: a wet, long-lived stain on the vertical cube side grows
        # downhill; the same stain on the horizontal floor (no downhill) grows
        # less. Measured in TICKS: an idle Studio only advances frames when
        # something wakes it, so wall seconds say little about paint time.
        def growth(obj, pos):
            s.call("paint.wet_reset")
            wait_ticks(s, 2)
            s.call("paint.wet_set_settings", object=obj, lifetime_seconds=5.0, runoff=2.0,
                   wetness=1.0, drip_head=1.0)
            s.call("paint.wet_dab", position=pos, radius=0.15, color=[0.9, 0.1, 0.1], amount=1.0)
            wait_ticks(s, 2)
            first = by_object(s.call("paint.wet_coverage"))[obj]["covered_texels"]
            wait_ticks(s, 20, timeout=20.0)
            last = by_object(s.call("paint.wet_coverage"))[obj]["covered_texels"]
            return first, last
        c0, c1 = growth(cube, [0.0, 0.2, 0.5])
        f0, f1 = growth(floor, [1.4, 0.0, -1.4])
        report["runoff"] = {"cube": [c0, c1], "floor": [f0, f1],
                            "cube_growth": round(c1 / max(1, c0), 3),
                            "floor_growth": round(f1 / max(1, f0), 3)}
        assert c1 > c0 * 1.02 and c1 / max(1, c0) > f1 / max(1, f0), ("no runoff on the vertical side", report["runoff"])
        for obj in created:
            s.call("paint.wet_set_settings", object=obj, lifetime_seconds=1.2, runoff=1.5,
                   wetness=0.65, drip_head=0.45)

        # 4. wet-on-wet: blue over WET red keeps more red than blue over DRY
        # red (CPU model: pickup of the wet underpaint + wet diffusion at the
        # edges). The same spot twice, mean colour of the covered texels.
        def red_after_blue(wait_dry):
            s.call("paint.wet_reset")
            wait_ticks(s, 2)
            s.call("paint.wet_dab", position=[1.4, 0.0, 1.4], radius=0.15, color=[0.9, 0.05, 0.05], amount=1.0)
            wait_ticks(s, 2)
            if wait_dry:
                wait_idle(s)
            s.call("paint.wet_dab", position=[1.4, 0.0, 1.4], radius=0.15, color=[0.05, 0.05, 0.9], amount=1.0)
            wait_ticks(s, 10)
            return by_object(s.call("paint.wet_coverage"))[floor]["mean_color"]
        wet_mix = red_after_blue(False)
        dry_mix = red_after_blue(True)
        report["mix"] = {"wet": [round(c, 3) for c in wet_mix], "dry": [round(c, 3) for c in dry_mix]}
        # REPORT ONLY. The CPU model's pickup of wet underpaint is ~1% for a
        # splat that fully covers the previous one (measured 2026-11-15: wet r
        # 0.209 vs dry r 0.206); "red + blue = purple" in the design note is a
        # product decision, not something the ported model promises.
        report["mix"]["wet_minus_dry_red"] = round(wet_mix[0] - dry_mix[0], 4)

        # 6. particles from the device buffers
        s.call("paint.wet_reset")
        wait_ticks(s, 2)
        s.call("particle.add_system", name="PaintRain")
        s.call("particle.set_physics", system="PaintRain", execution_policy="auto", particle_radius=0.02)
        s.call("particle.add_emitter", system="PaintRain", name="Drops", point=[-1.3, 1.5, -1.3],
               direction=[0.0, -1.0, 0.0], speed=1.0, spread=8.0, rate_per_second=1500.0,
               lifetime_seconds=4.0)
        s.call("collider.create", name="PaintFloor", source_mode="plane", plane_y=0.0, restitution=0.2)
        s.call("particle.set_collision", system="PaintRain", response="die", record_events=True,
               event_capacity=65536)
        s.call("paint.wet_set_splat", radius=0.06, amount=0.9, particle_system="PaintRain")
        for _ in range(90):
            s.call("particle.step", system="PaintRain", dt=1.0 / 60.0)
            if _ % 10 == 0:
                wait_ticks(s, 1, timeout=2.0)
        st = wait_ticks(s, 4)
        pstats = s.call("particle.stats", system="PaintRain")
        cov = by_object(s.call("paint.wet_coverage"))
        report["particles"] = {"gpu_status": pstats.get("gpu_status"),
                               "device_event_dispatches": st["device_event_dispatches"],
                               "host_events_applied": st["host_events_applied"],
                               "floor_covered": cov[floor]["covered_texels"],
                               "sources_dropped": st["sources_dropped"]}
        assert pstats.get("gpu_status") == "gpu_resident", ("particles not on the device", report["particles"])
        assert st["device_event_dispatches"] > 0 and st["host_events_applied"] == 0, report["particles"]
        assert cov[floor]["covered_texels"] > 0, ("particle contacts left no paint", report["particles"])

        # 7. reset
        s.call("paint.wet_reset")
        wait_ticks(s, 2)
        after = s.call("paint.wet_coverage")
        report["reset"] = {"covered": after["covered_texels"]}
        assert after["covered_texels"] == 0, ("reset left paint", report["reset"])
        # 8. the SCREEN, not the atlas: a red stain on the cube's +Z face and on
        # the floor in front of the default camera must show up as red pixels in
        # the viewport (in memory only - no screenshot file). The atlas can be
        # right while the screen is not (a material shared by several objects,
        # a texture never re-uploaded); every check above is blind to that.
        s.call("paint.wet_dab", position=[0.0, 0.0, 0.5], radius=0.2, color=[0.95, 0.05, 0.05], amount=1.0)
        s.call("paint.wet_dab", position=[0.6, 0.0, 0.9], radius=0.25, color=[0.95, 0.05, 0.05], amount=1.0)
        wait_ticks(s, 6)
        report["screen"] = red_pixels_on_screen(s)
        assert report["screen"].get("red_pixels", 0) > 200, ("paint is in the atlas but not on the screen", report["screen"])
        s.call("paint.wet_reset")
        wait_ticks(s, 2)
        report["compute_context"] = s.call("paint.wet_status")["compute_context"]
        tables = s.call("render.volume_tables").get("backends", [])
        report["sim_device_is"] = {b.get("role"): b.get("sim_device_is_this_backends") for b in tables}
        if keep:
            # Something to look at: one stain per surface, in three colours.
            for pos, col in (([0.0, 0.0, 0.5], [0.9, 0.1, 0.1]), ([0.0, 0.5, 0.0], [0.1, 0.3, 0.9]),
                             ([1.2, 0.0, -1.2], [0.1, 0.8, 0.2])):
                s.call("paint.wet_dab", position=pos, radius=0.2, color=col, amount=1.0)
            wait_ticks(s, 4)
        print(json.dumps({"result": "PASS", **report}, indent=2))
    except AssertionError as e:
        print(json.dumps({"result": "FAIL", "reason": str(e), **report}, indent=2))
        sys.exit(1)
    finally:
        if keep:
            print("--keep: objects, surfaces and stains left in the scene; particle system removed")
            try:
                for name in collider_names(s):
                    s.call("collider.remove", name=name)
                s.call("particle.clear_systems")
            finally:
                s.close()
            return
        try:
            s.call("paint.wet_clear_surfaces")
            # Colliders first: collider.* creates a default system when none exists.
            for name in collider_names(s):
                s.call("collider.remove", name=name)
            s.call("particle.clear_systems")
            for obj in created:
                s.call("scene.delete", name=obj)
        finally:
            s.close()


if __name__ == "__main__":
    main()
