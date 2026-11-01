"""Read-only benchmark identity. Excludes only the Max Steps A/B variable."""
import hashlib
import json

RAY_CAMERA_KEYS = ("position", "target", "up", "fov", "focus_distance", "aperture",
                   "depth_of_field", "effective_lens_radius", "lens_distortion", "auto_lens")


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def capture(client, targets):
    frame = client.call("timeline.get_frame")
    camera = client.call("camera.get")
    fields, versions = [], {}
    strength = "float_bits"
    domains = sorted({t[len("matter:"):].split(":gas", 1)[0] for t in targets})
    for domain in domains:
        info = client.call("fluid.inspect_fog_fields", domain=domain)
        versions[domain] = info["state_version"]
        for field in info["fields"]:
            if field["material_target"] not in targets:
                continue
            if "density_hash" not in field or "temperature_hash" not in field:
                strength = "summary_only"
            fields.append({"target": field["material_target"], "grid": info["grid"],
                "snapshot_source": info.get("gas_snapshot"),
                **{k: field.get(k) for k in ("density_hash", "temperature_hash", "active_cells",
                    "density_sum", "temperature_min_kelvin", "temperature_max_kelvin",
                    "optical_bounds", "visible")}})
    if {f["target"] for f in fields} != set(targets):
        raise RuntimeError("Gas target membership changed during benchmark")
    material = {}
    for target in targets:
        values = dict(client.call("volume.get_material", target=target))
        values.pop("max_steps", None)
        material[target] = values
    slots = client.call("render.volume_slots")
    boxes = []
    for backend in slots.get("backends", []):
        if backend["role"] != "render":
            continue
        for slot in backend.get("slots", []):
            boxes.append({k: slot.get(k) for k in ("name", "aabb_min", "aabb_max", "is_active",
                "volume_type", "density_device_local", "temperature_device_local")})
    context = {"frame": frame, "camera": {k: camera.get(k) for k in RAY_CAMERA_KEYS},
               "fields": sorted(fields, key=lambda f: f["target"]), "materials": material,
               "gpu_boxes": sorted(boxes, key=lambda b: b.get("name") or "")}
    if client.call("timeline.get_frame") != frame:
        raise RuntimeError("Timeline changed while reading benchmark context")
    return {"signature": signature(context), "fingerprint_strength": strength,
            "state_versions": versions, "context": context}


def require_same(reference, current):
    if reference["signature"] != current["signature"]:
        changed = [key for key, value in reference["context"].items()
                   if value != current["context"].get(key)]
        raise RuntimeError("INVALID benchmark: scene context changed: " + ", ".join(changed))
