"""Install a per-grain mineral mix on a granular material (Object Info -> Random).

docs/dev/KUM_TANE_GEOMETRISI_VE_MINERAL_KARISIMI.md section 3 (mineral table).
Every grain draws one Random value; thresholds on it pick a component:

    quartz      r < 0.70   honey..milky, translucent (transmission 0.55), IOR 1.55
    feldspar    0.70-0.88  cream/pink, mostly opaque (transmission 0.12)
    lithic      0.88-0.97  grey-brown, opaque, rough
    heavy       r >= 0.97  near-black (magnetite/ilmenite), opaque

Colours are built per channel from Math nodes into Combine Color, because
Math is the one node whose constants (its a/b properties) the material program
compiles. Before the 2026-10-22 compiler fix an unconnected Mix Color colour
pin compiled as 0.5 grey on the GPU and in CPU per-pixel shading, even though
the folded material showed it correctly. The Color Ramp is no better: its stop
list is an array property, which nodes.set_property refuses.

The same graph serves both grain paths: the procedural sphere cloud feeds
Random from the sphere record, near-camera mesh grains from the instance
origin (stable only while the grain rests - K1b).

Usage (app open): python scripts\\test\\rt_grain_mineral_material.py --material K0_Dry_Sand
Removes and rebuilds the material's graph each run.
"""
import argparse
import json

from rt_ipc import RtIpc

GT = "material"

# (upper Random threshold, base colour, transmission, roughness); the last row
# takes everything above the previous threshold.
QUARTZ_HONEY = [0.80, 0.66, 0.42]
QUARTZ_MILKY = [0.90, 0.88, 0.82]
FELDSPAR = [0.86, 0.68, 0.58]
LITHIC = [0.33, 0.30, 0.27]
HEAVY = [0.045, 0.045, 0.05]
T_FELDSPAR, T_LITHIC, T_HEAVY = 0.70, 0.88, 0.97
TRANS_QUARTZ, TRANS_FELDSPAR = 0.55, 0.12
ROUGH_BASE, ROUGH_LITHIC = 0.35, 0.30


class Graph:
    def __init__(self, ipc, name):
        self.ipc, self.name = ipc, name

    def add(self, type_id, **props):
        node = self.ipc.call("nodes.add", graph_type=GT, graph_name=self.name, type_id=type_id)
        for key, value in props.items():
            self.ipc.call("nodes.set_property", graph_type=GT, graph_name=self.name,
                          node_id=node, property=key, value=value)
        return node

    def link(self, from_node, from_output, to_node, to_input):
        self.ipc.call("nodes.link_by_key", graph_type=GT, graph_name=self.name,
                      from_node=from_node, from_output=from_output,
                      to_node=to_node, to_input=to_input)

    # Math ops (MaterialNodesV2.h MathNode::Op order).
    ADD, SUB, MUL, FRACT, GREATER = 0, 1, 2, 12, 17

    def math(self, op, a, b):
        """a/b: (node, output_key) to link, or a float constant."""
        node = self.add("MatV2.Math", op=op,
                        **({"a": float(a)} if not isinstance(a, tuple) else {}),
                        **({"b": float(b)} if not isinstance(b, tuple) else {}))
        if isinstance(a, tuple):
            self.link(a[0], a[1], node, "A")
        if isinstance(b, tuple):
            self.link(b[0], b[1], node, "B")
        return (node, "Value")

    def lerp(self, a, b, t):
        """a + t * (b - a); a/b are (node, key) or constants, t is (node, key)."""
        if not isinstance(a, tuple) and not isinstance(b, tuple):
            return self.math(Graph.ADD, self.math(Graph.MUL, t, b - a), a)
        return self.math(Graph.ADD, a, self.math(Graph.MUL, t, self.math(Graph.SUB, b, a)))

    def combine(self, channels):
        node = self.add("MatV2.CombineColor")
        for key, ch in zip(("R", "G", "B"), channels):
            self.link(ch[0], ch[1], node, key)
        return (node, "Color")


def output_node(ipc, name):
    for node in ipc.call("nodes.list", graph_type=GT, graph_name=name):
        if node["type_id"] == "MatV2.Output":
            return node["id"]
    raise RuntimeError("graph has no Material Output")


def install(ipc, material, ior=1.55, transmission_scale=1.0):
    if material in ipc.call("nodes.graphs", graph_type=GT):
        ipc.call("nodes.remove_graph", graph_type=GT, graph_name=material)
    ipc.call("nodes.create_graph", graph_type=GT, graph_name=material)
    g = Graph(ipc, material)
    out = output_node(ipc, material)
    # The default graph wires a Color node into Base Color; it is replaced below.
    for node in ipc.call("nodes.list", graph_type=GT, graph_name=material):
        if node["type_id"] == "MatV2.Color":
            ipc.call("nodes.remove", graph_type=GT, graph_name=material, node_id=node["id"])

    r = (g.add("MatV2.ObjectInfo"), "Random")
    not_quartz = g.math(Graph.GREATER, r, T_FELDSPAR)
    lithic_up = g.math(Graph.GREATER, r, T_LITHIC)
    heavy = g.math(Graph.GREATER, r, T_HEAVY)

    # Quartz tone varies inside its class: a second, decorrelated draw.
    tone = g.math(Graph.FRACT, g.math(Graph.MUL, r, 7.31), 0.0)
    channels = []
    for c in range(3):
        ch = g.lerp(QUARTZ_HONEY[c], QUARTZ_MILKY[c], tone)
        ch = g.lerp(ch, FELDSPAR[c], not_quartz)
        ch = g.lerp(ch, LITHIC[c], lithic_up)
        channels.append(g.lerp(ch, HEAVY[c], heavy))
    colour = g.combine(channels)
    g.link(colour[0], colour[1], out, "Base Color")

    # transmission = quartz - (quartz - feldspar) * not_quartz - feldspar * lithic_up
    k = transmission_scale
    trans = g.math(Graph.ADD,
                   g.math(Graph.MUL, not_quartz, k * (TRANS_FELDSPAR - TRANS_QUARTZ)),
                   g.math(Graph.MUL, lithic_up, -k * TRANS_FELDSPAR))
    trans = g.math(Graph.ADD, trans, k * TRANS_QUARTZ)
    g.link(trans[0], trans[1], out, "Transmission")

    rough = g.math(Graph.ADD, g.math(Graph.MUL, lithic_up, ROUGH_LITHIC), ROUGH_BASE)
    g.link(rough[0], rough[1], out, "Roughness")
    ipc.call("nodes.set_property", graph_type=GT, graph_name=material, node_id=out,
             property="ior", value=float(ior))

    return ipc.call("nodes.apply", graph_type=GT, graph_name=material)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--material", required=True)
    parser.add_argument("--ior", type=float, default=1.55)
    parser.add_argument("--transmission-scale", type=float, default=1.0,
                        help="0 = opaque grains (isolates colour from refraction)")
    args = parser.parse_args()
    ipc = RtIpc()
    try:
        result = install(ipc, args.material, args.ior, args.transmission_scale)
        print(json.dumps(result, indent=1))
        nodes = ipc.call("nodes.list", graph_type=GT, graph_name=args.material)
        print("nodes:", len(nodes))
    finally:
        ipc.close()


if __name__ == "__main__":
    main()
