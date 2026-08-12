"""Phase 9: the fire itself must bring the structure down.

Every earlier phase built one link of this chain and stopped one short:

    burn -> integrity loss -> damage-guided fracture -> structural clusters
         -> BLAST FROM THE COMBUSTION ITSELF -> the hit part detaches

The last arrow is what this gate covers, and until now it did not exist. A
script could fire rt.gas.pressure_pulse by hand and phase_08 passed on exactly
that — but nothing in the app ever produced a structural impulse, so a fire
could char a building for a minute and never move it. Two gaps caused it: no
producer (only the scripted pulse queued events) and no consumer (the app's own
simulation loop pumped contact fractures but never blast events).

★ THIS TEST MUST NOT CALL rt.gas.pressure_pulse. The whole point is that nobody
had to. If you find yourself adding one to make it pass, the feature is still
missing and the test has become a decoration.

Run order:
    1. phase_08_thermal_fracture_stage_a.py   (burn the beam)
    2. its UI step                            (generate shards, make breakable)
    3. this script
"""

import rt

GAS = "Phase08_BurnGas"
BEAM = "Phase08_Beam"

groups = []
for index in range(0, 33):
    name = BEAM if index == 0 else "%s__cluster_%d" % (BEAM, index)
    try:
        groups.append((name, rt.physics.fracture_group(name)))
    except RuntimeError:
        if index > 0:
            break

assert len(groups) >= 2, (
    "expected several structural clusters; run stage A and its UI step first",
    [g[0] for g in groups])

before = {name: info["broken_count"] for name, info in groups}
assert sum(before.values()) == 0, ("something is already broken", before)

# ── Arm the coupling, then let the fire do the work ──────────────────────────
# The authored threshold tells us what scale of blast is meant to be survivable,
# so the pressure scale is calibrated against it rather than guessed. This knob
# is explicitly a calibration: the solver's fuel and temperature are normalized,
# so nothing converts them to kPa on physical grounds.
threshold = max(i["base_break_impulse"] for _, i in groups)
rt.gas.set_settings(GAS,
                    structural_coupling_enabled=True,
                    structural_pressure_scale=400.0,
                    structural_min_intensity=0.05,
                    structural_event_interval=0.25)

settings = rt.gas.get_settings(GAS) if hasattr(rt.gas, "get_settings") else None
if settings is not None:
    assert settings["structural_coupling_enabled"], (
        "the coupling flag did not stick - script/IPC parity gap", settings)

stats_before = rt.gas.structural_impulse_stats()

# Keep burning. No pulse is injected anywhere in this loop: every event that
# arrives is produced by the combustion field itself.
for _ in range(90):
    rt.fluid.step(1.0 / 30.0)
    rt.physics.step(1.0 / 30.0)

stats = rt.gas.structural_impulse_stats()
after = {name: rt.physics.fracture_group(name) for name, _ in groups}
broken = [n for n in after if after[n]["broken_count"] > before[n]]
intact = [n for n in after if after[n]["broken_count"] == before[n]]

# ── The producer exists at all ──────────────────────────────────────────────
assert stats["queued"] > stats_before["queued"], (
    "the fire produced NO structural impulse. Either the combustion never got "
    "hot enough to pass structural_min_intensity, or the producer is not wired "
    "into the step. Check that the beam is actually burning (stage A reports "
    "mean_integrity well below 1.0).", stats)

# ── ...and the app consumed it, rather than piling it up in a queue ─────────
# ★ These are two different failures with one appearance. `queued` rising while
# `consumed` stays flat is exactly what the old code did: the only consumer was
# rt.physics.step, so blast events accumulated forever during playback.
assert stats["consumed"] > stats_before["consumed"], (
    "structural impulses were queued but never consumed - the consumer is not "
    "pumped by this loop", stats)

assert stats["last_projected_area_m2"] > 0.0, (
    "an impulse was computed with no area behind it - the group AABB came out "
    "empty", stats)

# ── ...and it was strong enough to matter, without being absurd ─────────────
assert broken, (
    "the fire loaded the structure but broke nothing. Raise "
    "structural_pressure_scale (currently %.0f) toward the impulse needed for "
    "the %.0f N-s threshold, or burn longer." % (400.0, threshold), stats, after)

assert intact, (
    "EVERY cluster broke. A blast that removes the whole object cannot show "
    "that the fire took off the part it actually burnt - lower "
    "structural_pressure_scale until the far end survives.", stats, after)

print({
    "result": "PASS", "phase": "9",
    "clusters": len(groups),
    "broken": broken,
    "intact": intact,
    "events_queued": stats["queued"] - stats_before["queued"],
    "events_consumed": stats["consumed"] - stats_before["consumed"],
    "last_max_impulse_Ns": stats["last_max_impulse"],
    "last_peak_pressure_kpa": stats["last_peak_pressure_kpa"],
    "authored_threshold_Ns": threshold,
})
print("")
print("No pressure_pulse call appears in this script. Every blast above came")
print("from the combustion field, which is the whole claim being tested.")
