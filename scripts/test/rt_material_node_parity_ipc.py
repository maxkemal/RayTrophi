"""Material node graph: what the panel/script authors is what renders.

Three checks, each a defect found 2026-10-22 (KUM_TANE_GEOMETRISI_VE_MINERAL_KARISIMI.md §8b):

  pin_value   An unconnected Mix Color colour set with nodes.set_param must
              reach the compiled per-pixel program. Before the fix the program
              baked a 0.5 literal: dark and bright pins rendered the same grey.
  remove      nodes.remove_graph must drop the compiled program from the
              material. Before the fix a base_color written afterwards was
              overridden by the dead program.
  ramp_stops  Color Ramp stops are scriptable (nodes.set_property 'stops',
              rows [[pos, r, g, b], ...]); a value the node would cap (9 stops)
              or a wrong row width is refused and the old value kept; a rows
              value on a pin (nodes.set_param) is refused.

Measured with render.probe on the Vulkan RT viewport (luminance of a framed
cube), not by looking. Builds its own cube + material far from the scene,
restores the camera, deletes the cube. The material stays (no delete API).

Run with the app open, viewport in Rendered mode:
    python scripts\\test\\rt_material_node_parity_ipc.py
"""
import json
import sys
import time

from rt_ipc import RtIpc, RtIpcError

GT = "material"
SPOT = [0.0, 60.0, 0.0]


def main():
    ipc = RtIpc()
    results = {}
    cube = None
    camera = ipc.call("camera.get")
    try:
        assert ipc.call("viewport.shading")["mode"] == "rendered", "Rendered viewport required"
        ipc.call("viewport.capture", enabled=True)
        mat = ipc.call("material.create", type="principled", name="NP_Parity")
        cube = ipc.call("scene.add_primitive", type="cube", name="NP_ParityCube", size=1.0)
        ipc.call("scene.set_transform", name=cube, translation=SPOT)
        ipc.call("material.assign", object_name=cube, material_name=mat)
        ipc.call("camera.set_position", position=[SPOT[0] + 1.6, SPOT[1] + 1.0, SPOT[2] + 1.6])
        ipc.call("camera.set_target", target=SPOT)

        def luminance():
            # Accumulation restarts on every apply; wait for a settled frame.
            time.sleep(0.3)
            deadline = time.perf_counter() + 30
            while ipc.call("viewport.status")["samples"] < 16:
                if time.perf_counter() > deadline:
                    raise RuntimeError("viewport did not accumulate")
                time.sleep(0.05)
            st = ipc.call("viewport.status")
            w, h = st["width"], st["height"]
            probe = ipc.call("render.probe", x=int(w * 0.45), y=int(h * 0.45),
                             width=int(w * 0.1), height=int(h * 0.1))
            assert probe["available"], probe
            return probe["mean_luminance"]

        def mix_graph(colour):
            if mat in ipc.call("nodes.graphs", graph_type=GT):
                ipc.call("nodes.remove_graph", graph_type=GT, graph_name=mat)
            ipc.call("nodes.create_graph", graph_type=GT, graph_name=mat)
            nodes = ipc.call("nodes.list", graph_type=GT, graph_name=mat)
            out = next(n["id"] for n in nodes if n["type_id"] == "MatV2.Output")
            for n in nodes:
                if n["type_id"] == "MatV2.Color":
                    ipc.call("nodes.remove", graph_type=GT, graph_name=mat, node_id=n["id"])
            mix = ipc.call("nodes.add", graph_type=GT, graph_name=mat, type_id="MatV2.MixColor")
            ipc.call("nodes.set_param", graph_type=GT, graph_name=mat, node_id=mix,
                     pin_index=0, value=0.0)
            ipc.call("nodes.set_param", graph_type=GT, graph_name=mat, node_id=mix,
                     pin_index=1, value=colour)
            ipc.call("nodes.link_by_key", graph_type=GT, graph_name=mat, from_node=mix,
                     from_output="Color", to_node=out, to_input="Base Color")
            ipc.call("nodes.apply", graph_type=GT, graph_name=mat)
            return mix

        # pin_value
        mix_graph([0.03, 0.03, 0.03])
        dark = luminance()
        mix_graph([0.9, 0.9, 0.9])
        bright = luminance()
        results["pin_value"] = {"dark": dark, "bright": bright,
                                "pass": bright > dark * 2.5}

        # remove
        mix_graph([0.03, 0.03, 0.03])
        before = luminance()
        ipc.call("nodes.remove_graph", graph_type=GT, graph_name=mat)
        ipc.call("material.set_param", material_name=mat, param="base_color",
                 value=[0.9, 0.9, 0.9])
        after = luminance()
        results["remove"] = {"with_graph_dark": before, "after_remove_white": after,
                             "pass": after > before * 2.5}

        # ramp_stops
        ipc.call("nodes.create_graph", graph_type=GT, graph_name=mat)
        ramp = ipc.call("nodes.add", graph_type=GT, graph_name=mat, type_id="MatV2.ColorRamp")
        stops = [[0.0, 0.1, 0.05, 0.0], [0.5, 0.8, 0.6, 0.3], [1.0, 1.0, 1.0, 1.0]]
        ipc.call("nodes.set_property", graph_type=GT, graph_name=mat, node_id=ramp,
                 property="stops", value=stops)
        listed = {p["name"]: p for p in ipc.call("nodes.list_properties", graph_type=GT,
                                                 graph_name=mat, node_id=ramp)}
        got = ipc.call("nodes.get_property", graph_type=GT, graph_name=mat, node_id=ramp,
                       property="stops")
        refusals = {}
        for label, kwargs in (
                ("nine_stops", dict(method="nodes.set_property", property="stops",
                                    value=[[i / 8.0, 0.0, 0.0, 0.0] for i in range(9)])),
                ("wrong_width", dict(method="nodes.set_property", property="stops",
                                     value=[[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])),
                ("rows_on_pin", dict(method="nodes.set_param", pin_index=0,
                                     value=[[0.0, 1.0]]))):
            method = kwargs.pop("method")
            try:
                ipc.call(method, graph_type=GT, graph_name=mat, node_id=ramp, **kwargs)
                refusals[label] = "ACCEPTED"
            except RtIpcError as exc:
                refusals[label] = str(exc)
        kept = ipc.call("nodes.get_property", graph_type=GT, graph_name=mat, node_id=ramp,
                        property="stops")

        def close(a, b):
            return len(a) == len(b) and all(
                abs(x - y) < 1e-6 for ra, rb in zip(a, b) for x, y in zip(ra, rb))

        results["ramp_stops"] = {
            "listed_type": listed.get("stops", {}).get("type"),
            "roundtrip": close(got, stops),
            "refusals": refusals,
            "kept_after_refusals": close(kept, stops),
            "pass": (listed.get("stops", {}).get("type") == "rows" and close(got, stops) and
                     all(v != "ACCEPTED" for v in refusals.values()) and close(kept, stops)),
        }
        ipc.call("nodes.remove_graph", graph_type=GT, graph_name=mat)
    finally:
        try:
            if cube:
                ipc.call("scene.delete", name=cube)
            ipc.call("camera.set_position", position=camera["position"])
            ipc.call("camera.set_target", target=camera["target"])
        finally:
            ipc.close()
    print(json.dumps(results, indent=1))
    ok = all(r["pass"] for r in results.values())
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
