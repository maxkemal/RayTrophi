"""Phase 8 diagnostic: what impulse does each cluster ACTUALLY receive, and why.

Read-only. Fires nothing, breaks nothing — run it any time after the UI step.

Why this exists: the first stage-B run reported one aggregate `last_max_impulse`
and it was tempting to reason backwards from that single number to a story about
cluster sizes. That reasoning is worthless the moment the scene differs from the
one the script authored (a manual rescale is enough). This prints the per-cluster
geometry the pressure bridge actually projects, so the distribution is measured
instead of inferred.

The impulse column reproduces the bridge's own expression:

    J [N s] = dp [Pa] * A_projected [m^2] * dt [s] * coupling * falloff

with A_projected the group AABB projected onto the plane normal to the blast
direction — exact for an axis-aligned box:

    A = |n.x|*dy*dz + |n.y|*dx*dz + |n.z|*dx*dy
"""

import rt

BEAM = "Phase08_Beam"
GAS = "Phase08_BurnGas"
PULSE_KPA = 300.0
PULSE_SECONDS = 0.02
PULSE_COUPLING = 1.0

groups = []
for index in range(0, 33):
    name = BEAM if index == 0 else "%s__cluster_%d" % (BEAM, index)
    try:
        groups.append((name, rt.physics.fracture_group(name)))
    except RuntimeError:
        if index > 0:
            break

assert groups, "no fracture groups found - run stage A and its UI step first"

# The blast is placed the way stage B places it — MEASURED from the clusters and
# the gas domain, never from stored coordinates. A diagnostic that exists to stop
# people inferring geometry has no business hardcoding any.
centers = [info["world_center"] for _, info in groups]
spread = [max(c[a] for c in centers) - min(c[a] for c in centers) for a in range(3)]
AXIS = max(range(3), key=lambda a: spread[a])
domain = rt.fluid.get(GAS)
domain_mid = (domain["domain_min"][AXIS] + domain["domain_max"][AXIS]) * 0.5
beam_lo = min(c[AXIS] for c in centers)
beam_hi = max(c[AXIS] for c in centers)
burnt_coord = beam_lo if abs(domain_mid - beam_lo) < abs(domain_mid - beam_hi) else beam_hi
PULSE_CENTER = tuple(
    burnt_coord if a == AXIS
    else (max(c[a] for c in centers) + min(c[a] for c in centers)) * 0.5
    for a in range(3))
PULSE_RADIUS = max(spread[AXIS] * 0.45, 1e-3)

rows = []
for name, info in groups:
    cx, cy, cz = info["world_center"]
    ex, ey, ez = info["world_extent"]
    dx, dy, dz = cx - PULSE_CENTER[0], cy - PULSE_CENTER[1], cz - PULSE_CENTER[2]
    distance = (dx * dx + dy * dy + dz * dz) ** 0.5
    # Mirrors the bridge exactly, including the engulfed case: within 5% of the
    # box there is no meaningful blast DIRECTION, so the area is the direction
    # average (Cauchy, S/4) instead of a projection along a noise vector. This
    # script exists to be checkable against the engine — the moment the two
    # expressions drift, its whole purpose is gone.
    engulf_radius = max(ex, ey, ez) * 0.05 + 1e-4
    engulfed = distance <= engulf_radius
    if engulfed:
        nx, ny, nz = 0.0, 1.0, 0.0
        area = (ex * ey + ey * ez + ez * ex) * 0.5
    else:
        nx, ny, nz = dx / distance, dy / distance, dz / distance
        area = abs(nx) * ey * ez + abs(ny) * ex * ez + abs(nz) * ex * ey
    in_range = distance <= PULSE_RADIUS
    falloff = max(0.0, 1.0 - distance / PULSE_RADIUS) if in_range else 0.0
    impulse = PULSE_KPA * 1000.0 * area * PULSE_SECONDS * PULSE_COUPLING * falloff
    rows.append({
        "group": name,
        "shards": info["shard_count"],
        "broken": info["broken_count"],
        "center_axis": round((cx, cy, cz)[AXIS], 3),
        "distance": round(distance, 3),
        "extent": (round(ex, 3), round(ey, 3), round(ez, 3)),
        "projected_area_m2": round(area, 4),
        "engulfed": engulfed,
        "falloff": round(falloff, 3),
        "predicted_impulse_Ns": round(impulse, 1),
        "effective_threshold_Ns": round(info["effective_break_impulse"], 1),
        "mean_integrity": round(info["mean_integrity"], 4),
        # Whether that integrity is this cluster's own region or the whole-object
        # average substituted for it, and how many elements it was built from.
        # Identical means across clusters read as uniform damage and are usually
        # a run of fallbacks instead.
        "regional": info.get("integrity_regional"),
        "elements": info.get("integrity_sampled_elements"),
        "would_break": impulse >= info["effective_break_impulse"] and in_range,
    })

rows.sort(key=lambda r: r["distance"])
print("cluster report — sorted by distance from the blast")
print("(`broken` is RUNTIME state: a rewind to frame 0 un-shatters every group,")
print(" so a zero here after a stage-B run means the timeline was rewound, not")
print(" that the blast did nothing.)")
for r in rows:
    print(r)

# The two questions the aggregate number could not answer.
in_range_rows = [r for r in rows if r["falloff"] > 0.0]
if len(in_range_rows) >= 2:
    nearest = in_range_rows[0]
    hardest = max(in_range_rows, key=lambda r: r["predicted_impulse_Ns"])
    print("")
    print("nearest in range : %s  area %.4f  J %.1f" %
          (nearest["group"], nearest["projected_area_m2"],
           nearest["predicted_impulse_Ns"]))
    print("hardest hit      : %s  area %.4f  J %.1f" %
          (hardest["group"], hardest["projected_area_m2"],
           hardest["predicted_impulse_Ns"]))
    if hardest["group"] != nearest["group"]:
        print("")
        print("★ The hardest-hit cluster is NOT the nearest one. That happens when")
        print("  the near clusters have smaller AABBs — which is what finer")
        print("  thermal seeding produces. Finer shattering therefore REDUCES the")
        print("  impulse a burnt region receives, partly cancelling the threshold")
        print("  reduction the same damage causes. Decide whether that coupling")
        print("  is wanted; right now it is emergent, not designed.")

degenerate = [r["group"] for r in rows if max(r["extent"]) <= 0.0]
if degenerate:
    print("")
    print("★ ZERO-SIZE clusters: %s" % degenerate)
    print("  A group with no extent presents no area, receives no impulse, and")
    print("  can never break by pressure. Its box is not being built from shard")
    print("  geometry.")

print("")
print("Blast used for these predictions: axis %s, center %s radius %.2f, "
      "%.0f kPa, %.3f s"
      % ("XYZ"[AXIS], tuple(round(v, 3) for v in PULSE_CENTER), PULSE_RADIUS,
         PULSE_KPA, PULSE_SECONDS))
print("Placement is measured from the clusters and the gas domain, so it follows")
print("the scene if the beam is moved or rescaled. Pressure and duration are the")
print("only constants left at the top.")
