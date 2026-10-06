"""Read-only canonical C6 snapshot/restore probe. Requires acceptance_metrics build.

Write a paused captured timeline frame with --write FILE; after save/load or cache
restore of that same frame, use --compare FILE. Does not create a bake or save.
"""
import argparse
import json
import math
from pathlib import Path

from rt_ipc import RtIpc
from rt_test_fluid_active_window_ipc import checked_call


def validate(inventory):
    metrics = inventory["acceptance_metrics"]
    assert metrics["measured"] and inventory["measured"]
    assert metrics["appearance_quantization"] == "normalized_saturation_upper_edge_v2", \
        "This snapshot uses an older appearance quantization policy"
    assert 0.001 <= metrics["appearance_full_saturation"] <= 1
    granular = metrics["granular"]
    bands = metrics["saturation_bands"]
    assert len(bands) == 8
    assert bands[0]["particles"] == metrics["exactly_dry_particles"], \
        "Positive saturation incorrectly remained in the dry appearance band"
    assert sum(b["particles"] for b in bands) == granular["particles"]
    assert granular["particles"] == inventory["models"][1]["particles"]
    for key in ["dry_mass_kg", "pore_water_kg", "capacity_kg", "kinetic_energy_j"]:
        assert math.isfinite(granular[key]) and granular[key] >= 0, key
        assert math.isclose(sum(b[key] for b in bands), granular[key], abs_tol=1e-7,
                            rel_tol=1e-7), key
    assert 0 <= granular["mean_saturation"] <= 1.00001
    assert math.isclose(granular["dry_mass_kg"] + granular["pore_water_kg"],
                        inventory["models"][1]["mass_kg"], abs_tol=1e-5, rel_tol=1e-6)


def compare(reference, current, position_tolerance):
    assert reference["control"]["frame"] == current["control"]["frame"], "Restore the same frame"
    before, after = reference["inventory"], current["inventory"]
    assert before["acceptance_metrics"]["appearance_quantization"] == \
        after["acceptance_metrics"]["appearance_quantization"]
    assert before["acceptance_metrics"]["appearance_full_saturation"] == \
        after["acceptance_metrics"]["appearance_full_saturation"]
    assert before["particle_id_hash"] == after["particle_id_hash"], "Canonical IDs/order changed"
    assert before["pore_exchange"]["settings"] == after["pore_exchange"]["settings"]
    for model_before, model_after in zip(before["models"], after["models"]):
        assert model_before["particles"] == model_after["particles"]
        assert math.isclose(model_before["mass_kg"], model_after["mass_kg"],
                            abs_tol=1e-5, rel_tol=1e-6)
    for a, b in zip(before["acceptance_metrics"]["saturation_bands"],
                    after["acceptance_metrics"]["saturation_bands"]):
        assert a["particles"] == b["particles"], "Spatial wet band population changed"
        for key in ["dry_mass_kg", "pore_water_kg", "capacity_kg"]:
            assert math.isclose(a[key], b[key], abs_tol=1e-5, rel_tol=1e-6), key
        for key in ["bounds_min", "bounds_max", "dry_center_of_mass"]:
            if a[key] is None or b[key] is None:
                assert a[key] == b[key], key
            else:
                assert all(abs(x-y) <= position_tolerance for x, y in zip(a[key], b[key])), key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", type=Path)
    mode.add_argument("--compare", type=Path)
    parser.add_argument("--position-tolerance", type=float, default=0.001)
    args = parser.parse_args()
    if not math.isfinite(args.position_tolerance) or args.position_tolerance < 0:
        parser.error("Position tolerance must be finite and nonnegative")
    client = RtIpc()
    try:
        control = checked_call(client, "sim.control_state")
        assert not control["playing"], "Pause first"
        inventory = checked_call(client, "fluid.matter_models", domain=args.domain)
        validate(inventory)
        assert checked_call(client, "sim.control_state") == control, "Control changed"
        snapshot = {"control": control, "inventory": inventory}
        if args.write:
            args.write.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        if args.compare:
            reference = json.loads(args.compare.read_text(encoding="utf-8"))
            assert reference["inventory"]["domain"] == args.domain
            validate(reference["inventory"])
            compare(reference, snapshot, args.position_tolerance)
        print("PASS canonical granular shape/mass/saturation bands" +
              (" and restore comparison" if args.compare else ""))
    finally:
        client.close()


if __name__ == "__main__":
    main()
