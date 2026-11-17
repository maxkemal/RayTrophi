"""GPU wet paint PBR channels on the scene that is ALREADY open (Player or Studio).

    python scripts/test/rt_player_wet_paint_channels_ipc.py                  # Player pipe
    python scripts/test/rt_player_wet_paint_channels_ipc.py --target studio

Creates nothing. Uses the wet paint surfaces the open scene already has (the
pastel workshop has four) and paints with paint.wet_dab at the first surface's
world position. Wet paint is session state: the script resets the paint and
turns the channels back off when it ends, but the generated channel textures
stay bound to the materials until the project is reopened.

Checks, in order:
  1. colour-only: a dab changes no roughness/normal texel, no channel dispatch;
  2. roughness on the first surface: roughness texels appear only under paint,
     no normal texel, the other surfaces stay untouched;
  3. height->normal: the existing stain gets normal texels;
  4. a new dab extends both and the planes REACH the textures
     (pbr_device_copies > 0; on one VkDevice no host bytes are downloaded);
  5. reset returns every channel texel to the original.
Reported, not asserted: last_tick_ms colour-only vs channels, extra bytes.
"""
import argparse
import json
import sys
import time

from rt_ipc import PIPE_NAME, PLAYER_PIPE_NAME, RtIpc

CHANNEL_KEYS = ("roughness_channel", "height_normal_channel", "wet_roughness", "dry_roughness",
                "height_contribution", "normal_strength", "lifetime_seconds", "runoff")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("player", "studio"), default="player")
    parser.add_argument("--keep", action="store_true", help="leave paint and channels on to LOOK at")
    args = parser.parse_args()
    s = RtIpc(PLAYER_PIPE_NAME if args.target == "player" else PIPE_NAME)
    report = {"target": args.target}
    saved = {}
    objects = []
    try:
        status = s.call("paint.wet_status")
        objects = [sf["object"] for sf in status.get("surfaces", [])]
        assert objects, "the open scene has no wet paint surface"
        assert status["device_ready"], ("wet paint device not ready", status["status"])
        for obj in objects:
            settings = s.call("paint.wet_get_settings", object=obj)
            assert "roughness_channel" in settings, "this binary predates the channel settings"
            saved[obj] = {k: settings[k] for k in CHANNEL_KEYS}
        first = objects[0]
        others = objects[1:]
        m = s.call("scene.get_world_transform", name=first)
        matrix = m.get("matrix", m) if isinstance(m, dict) else m
        centre = [matrix[0][3], matrix[1][3], matrix[2][3]]
        report["surface"] = first
        report["dab_centre"] = [round(v, 3) for v in centre]
        counters0 = {k: status[k] for k in ("pbr_dispatches", "pbr_device_copies",
                                            "publish_bytes_downloaded", "publish_device_copies")}

        s.call("paint.wet_reset")
        for obj in objects:
            s.call("paint.wet_set_settings", object=obj, roughness_channel=False,
                   height_normal_channel=False, lifetime_seconds=30.0, runoff=0.0)
        wait_ticks(s, 3)

        def dab(dx=0.0, dz=0.0):
            s.call("paint.wet_dab", position=[centre[0] + dx, centre[1], centre[2] + dz],
                   radius=0.3, color=[0.9, 0.1, 0.1], amount=1.0)

        # 1. colour only
        dab()
        st = wait_ticks(s, 4)
        cov = by_object(s.call("paint.wet_coverage"))
        c = cov[first]
        report["colour_only"] = {"covered": c["covered_texels"], "roughness_texels": c["roughness_texels"],
                                 "normal_texels": c["normal_texels"],
                                 "pbr_dispatches_delta": st["pbr_dispatches"] - counters0["pbr_dispatches"],
                                 "last_tick_ms": st["last_tick_ms"]}
        assert c["covered_texels"] > 0, ("the dab painted nothing on " + first, report)
        assert c["roughness_texels"] == 0 and c["normal_texels"] == 0, \
            ("channel texels changed while off", report["colour_only"])

        # 2. roughness on the existing stain
        s.call("paint.wet_set_settings", object=first, roughness_channel=True,
               wet_roughness=0.1, dry_roughness=0.3)
        st = wait_ticks(s, 4)
        cov = by_object(s.call("paint.wet_coverage"))
        c = cov[first]
        report["roughness"] = {"roughness_texels": c["roughness_texels"], "normal_texels": c["normal_texels"],
                               "covered": c["covered_texels"], "valid": c["valid_texels"],
                               "others_roughness": {o: cov[o]["roughness_texels"] for o in others},
                               "status": st["status"],
                               "pbr_dispatches_delta": st["pbr_dispatches"] - counters0["pbr_dispatches"]}
        assert c["roughness_texels"] > 0, ("roughness did not follow the paint", report["roughness"])
        assert c["roughness_texels"] < c["valid_texels"] // 2, \
            ("roughness changed outside the painted area", report["roughness"])
        assert c["normal_texels"] == 0, ("normal changed with only roughness on", report["roughness"])
        assert all(cov[o]["roughness_texels"] == 0 for o in others), \
            ("a colour-only surface changed", report["roughness"])

        # 3. height -> normal, seeded from the paint already there
        s.call("paint.wet_set_settings", object=first, height_normal_channel=True,
               height_contribution=0.6, normal_strength=2.0)
        st = wait_ticks(s, 4)
        c = by_object(s.call("paint.wet_coverage"))[first]
        report["height_normal"] = {"normal_texels": c["normal_texels"],
                                   "roughness_texels": c["roughness_texels"], "status": st["status"]}
        assert c["normal_texels"] > 0, ("no normal from the existing paint's height", report["height_normal"])
        seeded_normal, seeded_rough = c["normal_texels"], c["roughness_texels"]

        # 4. a new dab grows both; the planes reach the textures
        dab(0.9, 0.6)
        st = wait_ticks(s, 6)
        c = by_object(s.call("paint.wet_coverage"))[first]
        report["new_dab"] = {
            "normal_texels": c["normal_texels"], "roughness_texels": c["roughness_texels"],
            "pbr_device_copies_delta": st["pbr_device_copies"] - counters0["pbr_device_copies"],
            "colour_device_copies_delta": st["publish_device_copies"] - counters0["publish_device_copies"],
            "host_bytes_delta": st["publish_bytes_downloaded"] - counters0["publish_bytes_downloaded"],
            "publish_mode": st["publish_mode"], "pbr_buffer_bytes": st["pbr_buffer_bytes"],
            "pbr_texture_bytes": st["pbr_texture_bytes"], "last_tick_ms": st["last_tick_ms"],
            "status": st["status"]}
        assert c["normal_texels"] > seeded_normal and c["roughness_texels"] > seeded_rough, \
            ("a new dab did not extend the channels", report["new_dab"])
        assert report["new_dab"]["pbr_device_copies_delta"] + report["new_dab"]["host_bytes_delta"] > 0, \
            ("channels are computed but never reach a material texture "
             "(binary without the channel publish step?)", report["new_dab"])

        if args.keep:
            print(json.dumps({"result": "PASS (reset not run: --keep)", **report}, indent=2))
            return

        # 5. reset
        s.call("paint.wet_reset")
        wait_ticks(s, 3)
        c = by_object(s.call("paint.wet_coverage"))[first]
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
            if not args.keep and saved:
                s.call("paint.wet_reset")
                for obj, values in saved.items():
                    s.call("paint.wet_set_settings", object=obj, **values)
        finally:
            s.close()


if __name__ == "__main__":
    main()
