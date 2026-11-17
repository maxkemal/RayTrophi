"""GPU wet paint PBR channels: roughness and height->normal follow the paint.

Runs in Studio on an EMPTY scene (the startup cube is tolerated and left
alone). Never launches an application. Builds its own objects and removes them.

    python scripts/test/rt_wet_paint_channels_ipc.py
    python scripts/test/rt_wet_paint_channels_ipc.py --keep   # leave the stains to LOOK at

Two floor planes side by side, both wet paint surfaces. ChannelFloor gets the
channel outputs, PlainFloor stays colour-only (the control). Checks, in order:
  1. colour-only: a dab changes no roughness/normal texel and runs no channel
     dispatch (the opt-in costs nothing while it is off);
  2. roughness: after roughness_channel=true the SAME existing stain changes
     roughness texels, only where there is paint (far fewer than the surface),
     and no normal texel; the control surface changes nothing;
  3. height->normal: after height_normal_channel=true the existing stain gets
     normal texels (the height is seeded from the paint already there);
  4. a new dab grows both counts, and the channel planes reached the textures
     (pbr_device_copies > 0, or host downloads on a second VkDevice);
  5. reset restores every channel texel to the original.
Reported, not asserted: tick ms colour-only vs channels, extra atlas bytes.
"""
import json
import sys
import time

from rt_ipc import RtIpc

RES = 256


def wait_ticks(s, n=4, timeout=8.0):
    start = s.call("paint.wet_status")["ticks"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = s.call("paint.wet_status")
        if st["ticks"] >= start + n:
            return st
        time.sleep(0.05)
    return s.call("paint.wet_status")


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
        s.call("paint.wet_clear_surfaces")

        for name, x in (("ChannelFloor", -2.5), ("PlainFloor", 2.5)):
            obj = s.call("scene.add_primitive", type="plane", name=name, size=4.0)
            created.append(obj)
            mat = s.call("material.create", type="principled", name=obj + "_Mat")
            s.call("material.set_param", material_name=mat, param="base_color", value=[0.8, 0.8, 0.8])
            s.call("material.set_param", material_name=mat, param="roughness", value=0.8)
            s.call("material.assign", object_name=obj, material_name=mat)
            s.call("scene.set_transform", name=obj, translation=[x, 0.0, 0.0])
        channel, plain = created
        for obj in created:
            info = s.call("paint.wet_add_surface", object=obj, resolution=RES)
            assert info["valid_texels"] > 0, ("bake produced nothing", info)
            # Long-lived so the stain is still wet when the channels turn on.
            s.call("paint.wet_set_settings", object=obj, lifetime_seconds=30.0, runoff=0.0)
        wait_ticks(s, 2)

        def dab(x, z=0.0):
            s.call("paint.wet_dab", position=[x, 0.0, z], radius=0.3, color=[0.9, 0.1, 0.1], amount=1.0)

        # 1. colour only
        dab(-2.5)
        dab(2.5)
        st = wait_ticks(s, 4)
        cov = by_object(s.call("paint.wet_coverage"))
        report["colour_only"] = {
            "covered": {o: cov[o]["covered_texels"] for o in created},
            "roughness_texels": {o: cov[o]["roughness_texels"] for o in created},
            "normal_texels": {o: cov[o]["normal_texels"] for o in created},
            "pbr_dispatches": st["pbr_dispatches"], "pbr_buffer_bytes": st["pbr_buffer_bytes"],
            "last_tick_ms": st["last_tick_ms"]}
        assert all(cov[o]["covered_texels"] > 0 for o in created), ("dab painted nothing", report)
        assert st["pbr_dispatches"] == 0 and st["pbr_buffer_bytes"] == 0, \
            ("channels cost something while off", report["colour_only"])
        assert all(cov[o]["roughness_texels"] == 0 and cov[o]["normal_texels"] == 0 for o in created), \
            ("channel texels changed while off", report["colour_only"])

        # 2. roughness on the existing stain
        s.call("paint.wet_set_settings", object=channel, roughness_channel=True,
               wet_roughness=0.1, dry_roughness=0.3)
        st = wait_ticks(s, 4)
        cov = by_object(s.call("paint.wet_coverage"))
        c, p = cov[channel], cov[plain]
        report["roughness"] = {"roughness_texels": c["roughness_texels"], "normal_texels": c["normal_texels"],
                               "covered": c["covered_texels"], "valid": c["valid_texels"],
                               "control_roughness_texels": p["roughness_texels"],
                               "pbr_dispatches": st["pbr_dispatches"],
                               "pbr_device_copies": st["pbr_device_copies"],
                               "host_downloads": st["publish_host_downloads"],
                               "status": st["status"]}
        assert c["roughness_texels"] > 0, ("roughness did not follow the paint", report["roughness"])
        assert c["roughness_texels"] < c["valid_texels"] // 2, \
            ("roughness changed outside the painted area", report["roughness"])
        assert c["normal_texels"] == 0, ("normal changed with only roughness on", report["roughness"])
        assert p["roughness_texels"] == 0 and p["normal_texels"] == 0, \
            ("the colour-only surface changed", report["roughness"])

        # 3. height -> normal, seeded from the paint already there
        s.call("paint.wet_set_settings", object=channel, height_normal_channel=True,
               height_contribution=0.6, normal_strength=2.0)
        st = wait_ticks(s, 4)
        cov = by_object(s.call("paint.wet_coverage"))
        c = cov[channel]
        report["height_normal"] = {"normal_texels": c["normal_texels"],
                                   "roughness_texels": c["roughness_texels"],
                                   "status": st["status"]}
        assert c["normal_texels"] > 0, ("no normal from the existing paint's height", report["height_normal"])
        seeded_normal, seeded_rough = c["normal_texels"], c["roughness_texels"]

        # 4. a new dab grows both; the planes reached the textures
        dab(-3.5, 1.0)
        st = wait_ticks(s, 6)
        cov = by_object(s.call("paint.wet_coverage"))
        c = cov[channel]
        report["new_dab"] = {"normal_texels": c["normal_texels"], "roughness_texels": c["roughness_texels"],
                             "pbr_device_copies": st["pbr_device_copies"],
                             "host_downloads": st["publish_host_downloads"],
                             "publish_mode": st["publish_mode"],
                             "pbr_buffer_bytes": st["pbr_buffer_bytes"],
                             "pbr_texture_bytes": st["pbr_texture_bytes"],
                             "last_tick_ms": st["last_tick_ms"]}
        assert c["normal_texels"] > seeded_normal and c["roughness_texels"] > seeded_rough, \
            ("a new dab did not extend the channels", report["new_dab"])
        assert st["pbr_device_copies"] + st["publish_host_downloads"] > 0, \
            ("channel planes never reached a material texture", report["new_dab"])
        assert cov[plain]["roughness_texels"] == 0, ("control surface changed", report["new_dab"])

        if keep:
            print(json.dumps({"result": "PASS (reset not run: --keep)", **report}, indent=2))
            return

        # 5. reset
        s.call("paint.wet_reset")
        wait_ticks(s, 3)
        cov = by_object(s.call("paint.wet_coverage"))
        c = cov[channel]
        report["reset"] = {"covered": c["covered_texels"], "roughness_texels": c["roughness_texels"],
                           "normal_texels": c["normal_texels"]}
        assert c["covered_texels"] == 0 and c["roughness_texels"] == 0 and c["normal_texels"] == 0, \
            ("reset left channel texels behind", report["reset"])
        print(json.dumps({"result": "PASS", **report}, indent=2))
    except AssertionError as e:
        print(json.dumps({"result": "FAIL", "reason": str(e), **report}, indent=2))
        sys.exit(1)
    finally:
        try:
            if not keep:
                s.call("paint.wet_clear_surfaces")
                for obj in created:
                    s.call("scene.delete", name=obj)
        finally:
            s.close()


if __name__ == "__main__":
    main()
