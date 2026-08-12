"""Phase 10: the whole fracture lifecycle, with nobody pressing a button.

Until physics.fracture_object existed this test could not be written. A script
could weaken an object and could register shards as breakable, but the CUT
itself was reachable only from the panel -- so every automated run stopped in
the middle and handed the rest to a human. On a one-person project that is the
same as not being tested.

What this covers, in order:
    1. cut an object into shards        (physics.fracture_object)
    2. register the clusters            (physics.fracture_cluster_groups)
    3. save the project                 (project.save)
    4. reopen it                        (project.open)
    5. the shards are still there, with the same names and the same clusters

★ Step 5 is the claim. .rtp has no geometry section, so the shards are restored
by RE-CUTTING from a saved recipe. If the sites were not stored the reopened
object would break somewhere else -- plausible-looking, and wrong, with the
rigid bodies silently rebound to different pieces.
"""

import os
import rt

OBJ = "Phase10_Block"
PROJECT = os.path.join(os.environ.get("TEMP", "."), "phase10_roundtrip.rtp")

# ── Start from a known state ────────────────────────────────────────────────
# ★ Not hygiene fussiness: add_primitive renames on a name collision, so a
# second run on a dirty scene creates OBJ.001 while every later call still
# addresses OBJ — and the test then measures the PREVIOUS run's object. That
# happened on the first real run of this file.
def object_names():
    return [o["name"] for o in rt.scene.objects()]


for existing in object_names():
    if existing.startswith(OBJ):
        try:
            rt.scene.delete(existing)
        except RuntimeError:
            pass

# A solid block. Exact clipping needs a closed mesh; a primitive cube is the
# cleanest guarantee of that, so a failure here is never "the asset was open".
rt.scene.add_primitive("cube", name=OBJ, size=2.0)
assert rt.scene.exists(OBJ), (
    "the block was renamed on creation, so every later call would address a "
    "different object", object_names())
rt.scene.set_transform(OBJ, translation=(0.0, 1.0, 0.0))

# ── 1. Cut ──────────────────────────────────────────────────────────────────
cut = rt.physics.fracture_object(OBJ, site_count=24, seed=4242, pattern=0,
                                 cluster_count=4, exact_surface=True)
assert cut["shard_objects"], ("the cut produced no shards", cut)
assert cut["cluster_count"] >= 2, (
    "asked for 4 structural clusters and got fewer than 2; a single group means "
    "any hit anywhere removes the whole object", cut)
assert len(cut["shard_clusters"]) == len(cut["shard_objects"]), (
    "cluster index list must be parallel to the shard list", cut)

shards_before = list(cut["shard_objects"])
clusters_before = list(cut["shard_clusters"])

# ── 2. Register the clusters exactly as "Make Breakable" would ──────────────
groups = rt.physics.fracture_cluster_groups(OBJ)
assert len(groups) == cut["cluster_count"], (groups, cut)
for entry in groups:
    info = rt.physics.make_fracture_group(
        entry["group"], entry["shard_objects"], break_velocity=5.0,
        source_object=OBJ)
    assert info["shard_count"] == len(entry["shard_objects"]), (entry, info)
    # ★ Mass, not toughness, is what makes the threshold real. Every shard used
    # to weigh exactly 1 kg regardless of size, and a group that reports its
    # shard count as its mass is showing exactly that bug again.
    assert info["group_mass_kg"] > 0.0, ("group has no mass", info)
    # ★★ THE ASSERT THIS TEST WAS MISSING. It passed while every shard weighed
    # exactly 1 kg, because "mass > 0" is true of a placeholder too. A group mass
    # equal to its shard count is the signature of the default never being
    # replaced -- and it stayed invisible until the numbers were read by eye.
    assert abs(info["group_mass_kg"] - info["shard_count"]) > 1e-3, (
        "group mass equals its shard count exactly, i.e. every shard still "
        "weighs the placeholder 1 kg. The volume-derived mass did not reach the "
        "body.", info)
    assert abs(info["base_break_impulse"] -
               info["base_break_velocity"] * info["group_mass_kg"]) < 0.01, (
        "the impulse threshold is not velocity x mass", info)

masses_before = {e["group"]: rt.physics.fracture_group(e["group"])["group_mass_kg"]
                 for e in groups}

# ── 3/4. Round trip ─────────────────────────────────────────────────────────
rt.project.save(PROJECT)
rt.project.open(PROJECT)

# ── 5. Everything came back ─────────────────────────────────────────────────
# ★ Compared as SETS of names, not counts. A different count is the obvious
# failure; the dangerous one is the same count with shifted names, because the
# rigid bodies are bound by name and would rebind to the wrong pieces without
# anything reporting an error.
after = rt.physics.fracture_cluster_groups(OBJ)
shards_after = [s for entry in after for s in entry["shard_objects"]]
assert set(shards_after) == set(shards_before), (
    "the reopened project did not reproduce the same shards. If the counts "
    "match but the names shifted, the saved sites were not used and the object "
    "was cut afresh.",
    sorted(set(shards_before) ^ set(shards_after)))

assert len(after) == len(groups), (
    "cluster count changed across the round trip", after, groups)

for entry in after:
    info = rt.physics.fracture_group(entry["group"])
    before_mass = masses_before.get(entry["group"])
    assert before_mass is not None, ("group name changed", entry, masses_before)
    assert abs(info["group_mass_kg"] - before_mass) < max(before_mass * 0.02, 1e-3), (
        "a group came back with a different mass, so the shards are not the "
        "same pieces even though the names line up", entry["group"],
        before_mass, info["group_mass_kg"])

print({
    "result": "PASS", "phase": "10",
    "shards": len(shards_before),
    "clusters": cut["cluster_count"],
    "sites_used": cut["site_count"],
    "group_masses_kg": {k: round(v, 2) for k, v in masses_before.items()},
    "project": PROJECT,
})
print("")
print("No button was pressed anywhere in this test. That is the point: the cut,")
print("the grouping, the save and the reopen are all reachable from a script,")
print("so this chain can be re-checked on every build instead of by hand.")
