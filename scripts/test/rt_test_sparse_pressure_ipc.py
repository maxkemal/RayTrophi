"""Dense/sparse pressure regression after the user shader/C++ build.

External process only; empty paused scene, no concurrent IPC tests. The mixed
Water/Soil fixture exercises the current resident Matter pressure lane without
changing ghost-fluid physics to make a test pass. Cleans its temporary scene.
This gate covers pressure storage, not complete sparse liquid/gas storage.
"""

import json
import math
import uuid
import argparse

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viscosity", action="store_true")
    parser.add_argument("--transfer", action="store_true",
                        help="Require compact MAC P2G/G2P and FLIP baseline selection")
    args = parser.parse_args()
    client = RtIpc()
    suffix = uuid.uuid4().hex[:8]
    domain = "SparsePressure_" + suffix
    source = "SparsePressureSoil_" + suffix
    created = False
    source_created = False
    substance = "SparseTransferWater_" + suffix
    substance_created = False
    try:
        assert not client.call("sim.control_state")["playing"], "Pause the timeline"
        assert not client.call("fluid.list_domains")["domains"], "Use an empty scene"
        if args.viscosity or args.transfer:
            client.call("substance.derive", name=substance, based_on="Water")
            substance_created = True
            fields = {}
            if args.viscosity:
                fields["liquid_kinematic_viscosity"] = 0.02
            if args.transfer:
                fields["solver_flip_blend"] = 0.95
            client.call("substance.set", name=substance, fields=fields)
        client.call("fluid.create_domain", name=domain, type="matter",
                    domain_min=[0, 0, 0], domain_max=[2, 2, 2], voxel_size=0.05)
        created = True
        client.call("fluid.set_param", domain=domain, backend="vulkan", boundary="closed",
                    default_substance=substance if args.viscosity or args.transfer else "Water",
                    visible=False, solid_phase=False,
                    thermal_liquid_enabled=False)
        # The viscous fixture crosses tile planes x/z=8 and y=16 at h=.05.
        seed_min = [0.35, 0.75, 0.35] if args.viscosity or args.transfer else [0.45, 0.85, 0.45]
        seed_max = [0.55, 0.95, 0.55] if args.viscosity or args.transfer else [0.65, 1.05, 0.65]
        client.call("fluid.seed", domain=domain, seed_min=seed_min,
                    seed_max=seed_max, particles_per_cell=2, persistent=True)
        client.call("flow_source.create", name=source, domain=domain, phase="liquid",
                    source_mode="point", position=[1.5, 1.2, 1.5], radius=0.0001,
                    fluid_substance="Soil", velocity=[0, 0, 0], fluid_velocity_spread=0.0,
                    fluid_particles_per_second=60.0, use_particle_limit=True,
                    max_emitted_particles=1, fluid_temperature_override=True,
                    fluid_temperature_kelvin=293.15)
        source_created = True
        results = []
        for sparse in (False, True):
            client.call("gas.set_settings", domain=domain, use_sparse_tiles=sparse)
            client.call("fluid.reset")
            initial = client.call("fluid.state_digest", domain=domain)
            for _ in range(4):
                client.call("fluid.step", dt=1.0 / 60.0)
            stats = client.call("fluid.step_stats", domain=domain)
            models = client.call("fluid.matter_models", domain=domain)
            assert stats["measured"] and stats["pressure_on_gpu"], stats
            assert not models["mixed_execution"]["step_held"], models
            assert stats["pressure_sparse_used"] == sparse, stats
            if args.viscosity:
                assert stats["viscosity_sparse_used"] == sparse, stats
            if args.transfer:
                assert stats["p2g_on_gpu"] and stats["g2p_on_gpu"], stats
                assert stats["transfer_sparse_used"] == sparse, stats
                assert stats["flip_sparse_used"] == sparse, stats
                # S1 (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md): the mixed liquid
                # lane keeps its MAC velocity on the pages only. A dense arm or
                # a compact failure must never read as canonical.
                assert stats.get("transfer_sparse_canonical") == sparse, stats
                assert not stats.get("transfer_sparse_blocked"), stats
                if not sparse:
                    for field in ("transfer_sparse_active_tiles", "transfer_sparse_allocated_tiles",
                                  "transfer_sparse_resident_bytes"):
                        assert stats[field] == 0, stats
            results.append({"initial": initial, "stats": stats,
                            "digest": client.call("fluid.state_digest", domain=domain),
                            "models": models["models"]})
        dense, sparse = results
        assert dense["initial"] == sparse["initial"], "Fixture reset changed initial state"
        assert dense["digest"]["particles"] == sparse["digest"]["particles"]
        errors = {
            "centroid": max(abs(a - b) for a, b in
                            zip(dense["digest"]["centroid"], sparse["digest"]["centroid"])),
            "mean_speed": abs(dense["digest"]["mean_speed"] - sparse["digest"]["mean_speed"]),
        }
        assert all(math.isfinite(value) and value <= 2e-5 for value in errors.values()), errors
        for a, b in zip(dense["models"], sparse["models"]):
            assert a["model"] == b["model"] and a["particles"] == b["particles"]
            assert abs(a["mass_kg"] - b["mass_kg"]) <= max(1e-10, abs(a["mass_kg"]) * 1e-6)
            delta = max(abs(x - y) for x, y in
                        zip(a["momentum_kg_m_s"], b["momentum_kg_m_s"]))
            assert delta <= max(1e-8, a["mass_kg"] * 2e-5), (a, b)
        stats = sparse["stats"]
        assert 0 < stats["pressure_sparse_active_tiles"] * 512 < stats["full_grid_cells"], stats
        assert stats["pressure_sparse_allocated_tiles"] >= stats["pressure_sparse_active_tiles"]
        assert 0 < stats["pressure_sparse_resident_bytes"] < stats["full_grid_cells"] * 5 * 4
        if args.viscosity:
            assert 0 < stats["viscosity_sparse_active_tiles"]
            assert stats["viscosity_sparse_allocated_tiles"] >= stats["viscosity_sparse_active_tiles"]
            assert 0 < stats["viscosity_sparse_resident_bytes"] < stats["full_grid_cells"] * 3 * 4
        if args.transfer:
            active = stats["transfer_sparse_active_tiles"]
            allocated = stats["transfer_sparse_allocated_tiles"]
            assert 0 < active <= allocated, stats
            assert active * 512 < stats["full_grid_cells"], stats
            assert stats["transfer_sparse_resident_bytes"] >= allocated * 576 * 9 * 4, stats
            assert "canonical" in stats["transfer_sparse_status"], stats
            print("PASS: compact canonical MAC velocity/FLIP selected; no dense velocity bank")
        print(json.dumps({"errors": errors, "dense": dense["stats"], "sparse": stats}))
        print("PASS: sparse pressure GPU selection, dense parity, mass/momentum and compact storage")
    finally:
        try:
            if source_created:
                client.call("flow_source.remove", name=source)
            if created:
                client.call("fluid.remove_domain", domain=domain)
            if substance_created:
                client.call("substance.remove", name=substance)
        finally:
            client.close()


if __name__ == "__main__":
    main()
