"""Build Workshop's four-piece playground with Blender 5.3.

Run: blender -b --factory-startup --python-exit-code 1 --python
     scripts/generate_workshop_playground.py -- [output.glb]
"""
import argparse
import json
import struct
import sys
from math import cos, radians, sin, tau
from pathlib import Path
from tempfile import NamedTemporaryFile

import bpy
from mathutils import Vector


WORKSHOP = (
    Path(__file__).resolve().parents[1]
    / "RayTrophiStudio/assets/scenes/pastel-paint-workshop"
)
PALETTE = {
    "cream": "FFF0D4",
    "coral": "EFA08D",
    "yellow": "F7CE82",
    "mint": "A8DCC6",
    "lavender": "C1B3E4",
    "wood": "D59D72",
    "green": "89BA85",
    "pink": "EFB8C9",
    "blue": "94C5E0",
    "sand": "F7D79F",
}
# Positions in Blender space (X, -Workshop Z, Workshop Y).
FLOOR = 0.033
ZONES = (
    ("Park_Slide", (-2.93, -0.15), 0.59 * 1.17, 40, "blue"),
    ("Park_Playhouse", (2.74, 0.35), 0.57 * 1.12, -28, "pink"),
    ("Park_MushroomGarden", (-0.25, 2.84), 0.61 * 1.27, 8, "mint"),
)
GROUPS = ("Park_Floor", *(zone[0] for zone in ZONES))


def add_material(obj, color, smooth=False):
    obj.data.materials.append(bpy.data.materials[color])
    if smooth:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    return obj


def round_edges(obj, width):
    modifier = obj.modifiers.new("Rounded edges", "BEVEL")
    modifier.width = width
    modifier.segments = 3
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    modifier = obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def box(name, loc, dimensions, material, bevel=0.08):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    add_material(obj, material, smooth=True)
    round_edges(obj, bevel)
    return obj


def ball(name, loc, scale, material, segments=12, rings=8):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    return add_material(obj, material, smooth=True)


def cylinder(name, loc, radius, depth, material):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24, radius=radius, depth=depth, location=loc
    )
    obj = bpy.context.object
    obj.name = name
    add_material(obj, material, smooth=True)
    round_edges(obj, min(radius, depth) * 0.20)
    return obj


def beam(name, start, end, width, material):
    start, end = Vector(start), Vector(end)
    direction = end - start
    obj = box(name, (start + end) / 2, (width, width, direction.length),
              material, bevel=width * 0.32)
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    return obj


def mesh(name, vertices, faces, material):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return add_material(obj, material, smooth=True)


def build_slide():
    box("Platform", (0, 0.59, 1.28), (1.1, 0.77, 0.17), "mint")
    for x in (-0.44, 0.44):
        for y in (0.30, 0.87):
            box("Platform post", (x, y, 0.7), (0.16, 0.16, 1.4), "wood")
            beam("Guard post", (x, y, 1.34), (x, y, 1.73), 0.09, "coral")
        beam("Side guard", (x, 0.30, 1.70), (x, 0.87, 1.70), 0.09, "coral")
        beam("Ladder rail", (x, 1.37, 0), (x, 0.84, 1.38), 0.12, "wood")
    for i in range(4):
        box("Ladder tread", (0, 1.24 - i * .12, .26 + i * .30),
            (.85, .29, .12), ("mint", "lavender")[i % 2], bevel=.05)

    # Closed cross-section with raised sides and a shallow exit.
    section = ((-.43, 0), (.43, 0), (.43, .20), (.34, .23),
               (.30, .07), (-.30, .07), (-.34, .23), (-.43, .20))
    vertices = []
    segments = 14
    for i in range(segments + 1):
        t = i / segments
        y = .26 - 1.72 * t
        z = .065 + 1.21 * (1 - t) ** 1.65
        vertices.extend((x, y, z + height) for x, height in section)
    width = len(section)
    faces = [tuple(reversed(range(width)))]
    for i in range(segments):
        for j in range(width):
            a = i * width + j
            b = i * width + (j + 1) % width
            faces.append((a, b, b + width, a + width))
    faces.append(tuple(range(segments * width, len(vertices))))
    chute = mesh("Curved yellow chute", vertices, faces, "yellow")
    round_edges(chute, 0.025)
    for x in (-.44, .44):
        ball("Guard cap", (x, .87, 1.72), (.09, .09, .09), "yellow")


def build_house():
    box("Cream walls", (0, 0, .75), (2.30, 1.94, 1.50), "cream", bevel=.13)
    box("Foundation", (0, 0, .11), (2.42, 2.03, .22), "lavender")
    profile = ((-1.35, 1.36), (-1.22, 1.66), (-.66, 2.06), (0, 2.25),
               (.66, 2.06), (1.22, 1.66), (1.35, 1.36))
    vertices = [(x, y, z) for y in (-1.12, 1.12) for x, z in profile]
    n = len(profile)
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))
    roof = mesh("Rounded coral roof", vertices, faces, "coral")
    round_edges(roof, .09)
    for x in (-1.09, 1.09):
        beam("Roof trim", (x, -1.13, 1.5), (x, 1.13, 1.5), .13, "yellow")

    box("Door", (-.46, -1.01, .56), (.69, .15, 1.04), "wood", bevel=.15)
    ball("Doorknob", (-.25, -1.10, .55), (.06, .035, .06), "yellow")
    box("Doorstep", (-.46, -1.15, .09), (.85, .38, .18), "yellow")
    for x, radius, depth, color in ((.53, .41, .095, "yellow"),
                                    (.53, .32, .125, "blue")):
        obj = cylinder("Porthole", (x, -1.02, .90), radius, depth, color)
        obj.rotation_euler.x = radians(90)
    beam("Window crossbar", (.26, -1.09, .90), (.80, -1.09, .90), .05, "cream")
    beam("Window mullion", (.53, -1.09, .64), (.53, -1.09, 1.16), .05, "cream")
    cylinder("Flowerpot", (.91, -1.15, .15), .18, .29, "pink")
    ball("Flowerpot foliage", (.91, -1.15, .34), (.22, .20, .19), "green")


def build_garden():
    for x, y, size, color in ((-.70, -.24, 1.0, "coral"),
                              (.59, -.52, .78, "yellow"),
                              (.16, .57, 1.16, "lavender")):
        cylinder("Mushroom stalk", (x, y, .23 * size), .19 * size, .46 * size, "cream")
        ball("Mushroom cap", (x, y, .53 * size), (.43 * size, .43 * size, .24 * size), color)
        for dx, dy, h, radius in ((-.16, -.09, .735, .078), (.10, .08, .755, .062),
                                   (.16, -.18, .717, .065)):
            ball("Cap spot", (x + dx * size, y + dy * size, h * size),
                 (radius * size, radius * size, .024 * size), "cream", 8, 5)

    for x, y, height, color in ((-1.02, .54, .47, "pink"), (.90, .44, .55, "yellow"),
                                 (-.10, -.76, .40, "lavender")):
        cylinder("Flower stem", (x, y, height / 2), .035, height, "green")
        ball("Flower leaf", (x + .10, y, height * .44), (.15, .078, .043), "mint")
        for index in range(5):
            angle = tau * index / 5
            ball("Petal", (x + .11 * cos(angle), y + .11 * sin(angle), height),
                 (.088, .088, .043), color, 8, 5)
        ball("Flower center", (x, y, height + .03), (.084, .084, .049), "yellow", 8, 5)


def join_group(name, objects, origin):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = obj.data.name = name
    bpy.context.scene.cursor.location = origin
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return obj


def build_zone(zone, builder):
    name, (x, y), scale, turn, _ = zone
    before = set(bpy.context.scene.objects)
    builder()
    parts = sorted(set(bpy.context.scene.objects) - before, key=lambda obj: obj.name)
    obj = join_group(name, parts, (0, 0, 0))
    obj.location = (x, y, FLOOR + 0.022)
    obj.scale = (scale,) * 3
    obj.rotation_euler.z = radians(turn)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return obj


def attach_edge_toy(obj, builder):
    before = set(bpy.context.scene.objects)
    builder()
    parts = sorted(set(bpy.context.scene.objects) - before, key=lambda part: part.name)
    join_group(obj.name, [obj, *parts], obj.location.copy())


def build_seesaw():
    x, y = -2.65, -1.96
    pivot = cylinder("Seesaw pivot", (x, y, FLOOR + .16), .16, .36, "yellow")
    pivot.rotation_euler.x = radians(90)
    start = (x - .69, y, FLOOR + .14)
    end = (x + .69, y + .14, FLOOR + .35)
    plank = box("Seesaw plank", (0, 0, 0), (1.42, .32, .11), "coral", bevel=.05)
    plank.location = (Vector(start) + Vector(end)) / 2
    plank.rotation_euler = (Vector(end) - Vector(start)).to_track_quat("X", "Z").to_euler()
    for px, py, pz in (start, end):
        for offset in (-.08, .08):
            box("Handle post", (px, py + offset, pz + .11), (.045, .045, .16),
                "mint", bevel=.015)
        beam("Handle grip", (px, py - .09, pz + .19), (px, py + .09, pz + .19),
             .05, "mint")


def build_climbing_stones():
    for x, y, width, height, color in (
        (2.15, -2.02, .43, .20, "coral"),
        (2.66, -1.88, .48, .31, "yellow"),
        (3.15, -1.58, .54, .43, "lavender"),
    ):
        stone = box("Climbing stone", (x, y, FLOOR + height / 2),
                    (width, width * .92, height), color, bevel=height * .34)
        stone.rotation_euler.z = radians(18)
    for x, y, size, color in (
        (3.57, -1.95, .15, "cream"),
        (3.50, -2.20, .11, "wood"),
        (1.92, -2.26, .12, "pink"),
    ):
        ball("Edge pebble", (x, y, FLOOR + size * .47), (size, size * .72, size * .45),
             color, 8, 5)


def build_paths():
    # Four cubic ribbons share the edges of the central square junction.
    half = .18
    z = FLOOR + .012
    center = Vector((0, -.90))
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    vertices = [(center.x + x, center.y + y, z) for x, y in corners]
    faces = [(0, 1, 2, 3)]
    branches = (
        ((1, 0), (0, -1), (0, -1.8), (-.74, -2.56), (-.33, -3.83), .31),
        ((2, 1), (1, 0), (1.54, -.90), (1.50, .52), (2.75, .28), .08),
        ((3, 2), (0, 1), (0, .36), (.76, 1.30), (-.27, 2.75), .05),
        ((0, 3), (-1, 0), (-1.23, -.90), (-2.03, .30), (-2.90, -.23), .08),
    )
    for edge, direction, control_a, control_b, end, flare in branches:
        start = center + Vector(direction) * half
        a, b, end = Vector(control_a), Vector(control_b), Vector(end)
        previous = edge
        for i in range(1, 25):
            t = i / 24
            u = 1 - t
            p = u**3 * start + 3 * u*u*t * a + 3 * u*t*t * b + t**3 * end
            tangent = (u*u*(a-start) + 2*u*t*(b-a) + t*t*(end-b)).normalized()
            left = Vector((-tangent.y, tangent.x)) * (half + flare * t * t)
            first = len(vertices)
            vertices.extend(((p.x + left.x, p.y + left.y, z),
                             (p.x - left.x, p.y - left.y, z)))
            faces.append((*previous, first + 1, first))
            previous = (first, first + 1)
    mesh("Curved rubber paths", vertices, faces, "sand")


def build_edge_flowers():
    for x, y, height, color in (
        (-3.44, -2.34, .23, "pink"),
        (-3.52, -2.10, .32, "cream"),
        (-3.28, -2.45, .17, "lavender"),
        (3.40, -2.47, .27, "lavender"),
        (3.59, -2.40, .20, "cream"),
    ):
        z = FLOOR + height
        vertices = [
            (x + dx, y + dy, stem_z)
            for stem_z in (FLOOR, z)
            for dx, dy in ((-.014, -.012), (.014, -.012), (0, .017))
        ]
        mesh("Stem", vertices, [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)], "green")
        for i in range(5):
            angle = tau * i / 5
            ball("Edge flower petal", (x + .062 * cos(angle), y, z + .062 * sin(angle)),
                 (.039, .026, .039), color, 6, 3)
        ball("Edge flower center", (x, y - .021, z), (.041, .030, .041),
             "yellow", 6, 3)


def build_floor():
    for _, (x, y), _, turn, color in ZONES:
        pad = cylinder("Rubber island", (x, y, FLOOR + .009), 1, .026, color)
        pad.scale = (.87, .82, 1)
        pad.rotation_euler.z = radians(turn)
    build_paths()
    for x, y, color in ((-2.65, -1.96, "mint"), (2.70, -1.92, "pink")):
        pad = cylinder("Low toy pad", (x, y, FLOOR + .005), .86, .018, color)
        pad.scale.y = .57
    build_edge_flowers()
    join_group("Park_Floor", list(bpy.context.scene.objects), (0, 0, 0))


def linear(channel):
    return channel / 12.92 if channel <= .04045 else ((channel + .055) / 1.055) ** 2.4


def palette_colors():
    return [
        tuple(linear(int(value[i:i + 2], 16) / 255) for i in (0, 2, 4)) + (1.,)
        for value in PALETTE.values()
    ]


def bake_paint_atlases():
    # Wet paint seeds its canvas from this albedo using the same non-overlapping UV.
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    for name, color in zip(PALETTE, palette_colors()):
        material = bpy.data.materials[name]
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        output = nodes.get("Material Output")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = color
        links.new(emission.outputs[0], output.inputs["Surface"])

    for name in GROUPS:
        obj = bpy.data.objects[name]
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        for layer in list(obj.data.uv_layers):
            obj.data.uv_layers.remove(layer)
        obj.data.uv_layers.new(name="PaintUV")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(island_margin=.009, margin_method="FRACTION")
        bpy.ops.object.mode_set(mode="OBJECT")

        image = bpy.data.images.new(f"{name}_Albedo", 512, 512, alpha=False)
        targets = []
        for source in obj.data.materials:
            nodes = source.node_tree.nodes
            target = nodes.new("ShaderNodeTexImage")
            target.image = image
            nodes.active = target
            targets.append((nodes, target))
        bpy.ops.object.bake(type="EMIT", margin=3, margin_type="EXTEND")
        image.pack()
        for nodes, target in targets:
            nodes.remove(target)

        material = bpy.data.materials.new(f"{name}_Paint")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        shader = nodes.get("Principled BSDF")
        shader.inputs["Roughness"].default_value = .81
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        material.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.material_index = 0


def verify_glb(path):
    with path.open("rb") as file:
        magic, version, size, json_size, chunk = struct.unpack("<4sIII4s", file.read(20))
        if (magic, version, size, chunk) != (b"glTF", 2, path.stat().st_size, b"JSON"):
            raise ValueError("Invalid GLB header")
        document = json.loads(file.read(json_size))

    nodes = [node for node in document["nodes"] if "mesh" in node]
    if {node["name"] for node in nodes} != set(GROUPS) or len(nodes) != 4:
        raise ValueError("Expected exactly the four named park nodes")
    if (len(document["meshes"]) != 4
            or len({node["mesh"] for node in nodes}) != 4
            or len(document["materials"]) != 4
            or len(document["images"]) != 4
            or any(image.get("bufferView") is None for image in document["images"])):
        raise ValueError("Expected four standalone meshes and four embedded paint albedos")
    total_triangles = 0
    materials = set()
    for node in sorted(nodes, key=lambda item: item["name"]):
        primitives = document["meshes"][node["mesh"]]["primitives"]
        if len(primitives) != 1 or primitives[0].get("mode", 4) != 4:
            raise ValueError(f"Expected one triangle primitive for {node['name']}")
        primitive = primitives[0]
        if "TEXCOORD_0" not in primitive["attributes"]:
            raise ValueError(f"Missing paint UV on {node['name']}")
        materials.add(primitive["material"])
        bounds = document["accessors"][primitive["attributes"]["POSITION"]]
        translation = node.get("translation", (0, 0, 0))
        limits = ((-4.04, 4.0), (.02, 1.52), (-3.83, 4.0))
        if any(key in node for key in ("rotation", "scale", "matrix")) or any(
            bounds["min"][i] + translation[i] < low
            or bounds["max"][i] + translation[i] > high
            for i, (low, high) in enumerate(limits)
        ):
            raise ValueError(
                f"{node['name']} falls outside the Workshop park area: "
                f"{bounds['min']}, {bounds['max']}, {translation}"
            )
        triangles = document["accessors"][primitive["indices"]]["count"] // 3
        total_triangles += triangles
        print(f"{node['name']}: {triangles:,} triangles", flush=True)
    if len(materials) != 4 or total_triangles > 18_500:
        raise ValueError(f"Paint material or triangle budget exceeded: {total_triangles:,}")
    print(f"Total: {total_triangles:,} triangles", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?",
                        default=WORKSHOP / "workshop_playground.glb")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if args.output.suffix.lower() != ".glb":
        parser.error("Output must have a .glb extension")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    for name, color in zip(PALETTE, palette_colors()):
        bpy.data.materials.new(name).diffuse_color = color
    build_floor()
    slide = build_zone(ZONES[0], build_slide)
    house = build_zone(ZONES[1], build_house)
    build_zone(ZONES[2], build_garden)
    attach_edge_toy(slide, build_seesaw)
    attach_edge_toy(house, build_climbing_stones)
    bake_paint_atlases()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".glb", dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        result = bpy.ops.export_scene.gltf(
            filepath=str(temporary.resolve()),
            export_format="GLB",
            export_yup=True,
            export_texcoords=True,
            export_animations=False,
        )
        if result != {"FINISHED"}:
            raise RuntimeError(f"Blender export failed: {result}")
        verify_glb(temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

