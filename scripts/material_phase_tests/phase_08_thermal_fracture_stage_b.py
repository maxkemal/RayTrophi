"""Phase 8 vertical gate, STAGE B: the blast must take off the end it hit.

Run AFTER phase_08_thermal_fracture_stage_a.py and its UI step. The scene must
still hold the clustered, breakable shards.

What this proves, and why each half matters:
  - Clustering: a pulse at one end of the beam breaks the clusters near it and
    leaves the far end standing. With a single group (the old behaviour) the
    whole beam would detach at once, and no amount of cut quality upstream can
    fix that — the grouping IS the destruction granularity.
  - Area-based impulse: the reported impulse is real newton-seconds derived from
    the cluster's own projected area, so it scales with the object instead of
    being a unitless number that had to be re-tuned whenever anything resized.
"""

import rt

GAS = "Phase08_BurnGas"
BEAM = "Phase08_Beam"

# Cluster 0 keeps the object's own name; the rest are suffixed. Probe rather
# than assume a count, so the test still reads correctly if the UI was driven
# with a different cluster setting.
groups = []
for index in range(0, 33):
    name = BEAM if index == 0 else "%s__cluster_%d" % (BEAM, index)
    try:
        groups.append((name, rt.physics.fracture_group(name)))
    except RuntimeError:
        if index > 0:
            break

assert len(groups) >= 2, (
    "expected several structural clusters; run stage A's UI step first "
    "(Structural Clusters > 1, then Make Breakable). Found: %r"
    % [g[0] for g in groups])

before = {name: info["broken_count"] for name, info in groups}
assert sum(before.values()) == 0, ("something is already broken", before)

# ── Where the beam actually lies, MEASURED ───────────────────────────────────
#
# ★ This block used to be the constant BURNT_END_X = -1.1 and an `abs(dx)`
# comparison, i.e. "the beam runs along X" baked into the test. Rescale the beam
# in the UI along any other axis and every spatial assertion below silently
# becomes nonsense: the clusters spread in Y while the test keeps measuring X,
# where they all sit within a few centimetres of each other. The gate then FAILS
# for a reason that has nothing to do with the physics it exists to check.
#
# The long axis is the one the cluster centres actually spread along.
centers = [info["world_center"] for _, info in groups]
spread = [max(c[a] for c in centers) - min(c[a] for c in centers) for a in range(3)]
AXIS = max(range(3), key=lambda a: spread[a])
AXIS_NAME = "XYZ"[AXIS]
assert spread[AXIS] > 1e-3, (
    "the cluster centres are coincident on every axis, so there is no long axis "
    "to reason about - was the beam actually split?", centers)

# Which END is the burnt one comes from the GAS DOMAIN, never from the integrity
# numbers: the domain is the authored CAUSE of the burn, integrity is its
# EFFECT, and deriving the expected end from the effect would make the
# localisation assertion below circular and unfailable.
domain = rt.fluid.get(GAS)
domain_mid = (domain["domain_min"][AXIS] + domain["domain_max"][AXIS]) * 0.5
beam_lo = min(c[AXIS] for c in centers)
beam_hi = max(c[AXIS] for c in centers)
burnt_coord = beam_lo if abs(domain_mid - beam_lo) < abs(domain_mid - beam_hi) else beam_hi

# The burnt end weakened, so its effective threshold must be the lower one.
weakest = min(groups, key=lambda g: g[1]["effective_break_impulse"])
strongest = max(groups, key=lambda g: g[1]["effective_break_impulse"])
assert (weakest[1]["effective_break_impulse"] <
        strongest[1]["effective_break_impulse"]), (
    "no cluster is thermally weaker than another - was the beam burnt, and was "
    "integrity_weakening left on?", weakest, strongest)

# ★ EVERY cluster must be able to see damage, not just the one that inherited
# the object's own name. MSF fields are keyed by object and fracture groups by
# cluster; when that distinction was missed, five clusters out of six reported a
# flat integrity of exactly 1.0 and were silently exempt from weakening. A run of
# identical 1.0s is the fingerprint of that bug, so refuse to pass on it.
suffixed = [(n, i) for n, i in groups if n != BEAM]
assert suffixed, "expected suffixed clusters"
assert any(i["mean_integrity"] < 0.9999 for _, i in suffixed), (
    "every suffixed cluster reports integrity 1.0, so they are not finding the "
    "MSF field at all - check that Make Breakable passed source_object",
    [(n, i["mean_integrity"]) for n, i in suffixed])

# And the weakening has to be LOCAL: the cluster nearest the flame must be
# weaker than the far one. A per-object average would give them all the same
# number and this comparison would be a coin flip.
def distance_to_burn(info):
    return abs(info["world_center"][AXIS] - burnt_coord)

assert distance_to_burn(weakest[1]) < distance_to_burn(strongest[1]), (
    "the weakest cluster is not the one nearest the flame, so integrity is not "
    "being measured per region", AXIS_NAME, burnt_coord, weakest, strongest)

# ★ A per-region reading, or the whole-object average wearing its clothes?
# When a cluster's region holds no MSF elements the summary falls back to the
# object figure, which is how six clusters came to report one identical mean and
# look like uniform damage. The engine now says which happened; refuse to draw
# localisation conclusions from a set of fallbacks.
regional = [(n, i) for n, i in groups if i.get("integrity_regional")]
assert len(regional) >= 2, (
    "fewer than two clusters got a REGIONAL integrity reading - the rest fell "
    "back to the whole-object average, so any per-cluster comparison here is "
    "meaningless. Check the cluster AABBs against the MSF element positions.",
    [(n, i.get("integrity_regional"), i.get("integrity_sampled_elements"))
     for n, i in groups])

# ★ THE THRESHOLD HAS TO BE IN RANGE OF THE BLAST, or this test is a fraud.
#
# The first run of this gate passed with impulses of ~170 N s against thresholds
# of ~4: every cluster inside the radius broke no matter how strong it was, and
# the 3-broke/3-survived split came purely from the distance cutoff. It looked
# like a localisation result and was really a radius result. Refuse to report
# PASS from that situation instead of quietly repeating it.
#
# The pulse sits on the burnt END of the measured long axis, and on the beam's
# own centre in the other two — placing it by hardcoded coordinates is what put
# the previous version's blast off to the side of a rescaled beam.
def _cross_axis_mid(axis):
    return (max(c[axis] for c in centers) + min(c[axis] for c in centers)) * 0.5

PULSE_CENTER = tuple(burnt_coord if a == AXIS else _cross_axis_mid(a)
                     for a in range(3))
# Sized from the beam, not from a constant: it must cover the burnt end and
# leave the far clusters outside, whatever length the beam actually is.
BEAM_LENGTH = spread[AXIS]
PULSE_RADIUS = max(BEAM_LENGTH * 0.45, 1e-3)
authored_threshold = strongest[1]["base_break_impulse"]

stats_before = rt.gas.structural_impulse_stats()

# One pulse at the burnt end. Radius deliberately smaller than the beam so the
# far clusters sit outside it: a blast that reaches everything cannot show that
# clustering localises anything.
rt.gas.pressure_pulse(GAS, center=PULSE_CENTER, radius=PULSE_RADIUS,
                      peak_pressure_kpa=300.0, duration_seconds=0.02,
                      coupling=1.0)
rt.physics.step(1.0 / 30.0)

stats = rt.gas.structural_impulse_stats()


def distance_to_pulse(info):
    cx, cy, cz = info["world_center"]
    dx, dy, dz = cx - PULSE_CENTER[0], cy - PULSE_CENTER[1], cz - PULSE_CENTER[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


after = {name: rt.physics.fracture_group(name) for name, _ in groups}
broken = [n for n in after if after[n]["broken_count"] > before[n]]
intact = [n for n in after if after[n]["broken_count"] == before[n]]
in_range = set(n for n, info in after.items()
               if distance_to_pulse(info) <= PULSE_RADIUS)
held = [n for n in intact if n in in_range]
out_of_range = [n for n in intact if n not in in_range]

assert stats["consumed"] > stats_before["consumed"], stats
# ★ No cluster may report a degenerate box. The group AABB was once built from
# shard CENTRES, which makes a single-shard cluster a POINT: extent (0,0,0),
# projected area 0, impulse 0 — permanently immune to pressure while reporting
# perfectly normal-looking integrity. Nothing else in this gate would notice,
# because "it did not break" is also what a strong cluster looks like.
degenerate = [(n, after[n]["world_extent"]) for n in after
              if max(after[n]["world_extent"]) <= 0.0]
assert not degenerate, (
    "cluster(s) report a zero-size world_extent, so the blast can put no area "
    "behind them and they can never break by pressure", degenerate)
assert stats["last_projected_area_m2"] > 0.0, (
    "impulse was computed with no area - the group AABB came out empty", stats)
assert broken, ("the pulse broke nothing; raise peak_pressure_kpa or lower the "
                "Break Threshold used in the UI step", stats, after)

# The authored threshold must be the same ORDER as the impulse the scene really
# produces. Far below it and strength is decorative; far above and nothing can
# ever break. Either way the run says nothing about weakening.
assert stats["last_max_impulse"] <= authored_threshold * 20.0, (
    "Break Threshold (%.1f N s) is far below the impulse this blast delivers "
    "(%.1f N s), so every cluster in range broke regardless of strength and "
    "thermal weakening was never exercised. Re-run the UI step with a threshold "
    "near the impulse above." % (authored_threshold, stats["last_max_impulse"]),
    stats)
assert held, (
    "every cluster within the blast radius broke, so the split you see is the "
    "RADIUS CUTOFF, not structural strength. Raise the Break Threshold toward "
    "%.0f N s and re-run." % stats["last_max_impulse"], stats, after)

print({
    "result": "PASS", "phase": "8b",
    "clusters": len(groups),
    "long_axis": AXIS_NAME,
    "beam_length": round(BEAM_LENGTH, 3),
    "burnt_end": round(burnt_coord, 3),
    "pulse_center": tuple(round(v, 3) for v in PULSE_CENTER),
    "pulse_radius": round(PULSE_RADIUS, 3),
    "regional_readings": "%d/%d" % (len(regional), len(groups)),
    "broken": broken,
    "held_in_range": held,          # survived the blast on strength
    "out_of_range": out_of_range,   # survived on distance only
    "authored_threshold_Ns": authored_threshold,
    "last_max_impulse_Ns": stats["last_max_impulse"],
    "last_projected_area_m2": stats["last_projected_area_m2"],
    "weakest_cluster": weakest[0],
    "weakest_effective_threshold": weakest[1]["effective_break_impulse"],
    "strongest_effective_threshold": strongest[1]["effective_break_impulse"],
})
print("")
print("`held_in_range` is the result that matters: those clusters were inside")
print("the blast and survived on strength. `out_of_range` survived on distance")
print("alone and proves nothing about weakening.")
