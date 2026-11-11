"""Post-build test in a DISPOSABLE scene; Apply rebases the scene timeline to zero.

Creates/removes its own Matter domain. Does not launch Studio or compile code.
Run externally: python scripts/test/rt_test_matter_seed_sources_ipc.py
"""
import math
import uuid

from rt_ipc import RtIpc, RtIpcError


def main():
    client = RtIpc()
    domain = "LayerSeed_" + uuid.uuid4().hex[:10]
    created = False
    material = domain + "_water"
    created_material = False

    def listing():
        return client.call("fluid.get_seed_sources", domain=domain)

    def digest():
        return client.call("fluid.state_digest", domain=domain)

    def rejected(method, **params):
        try:
            client.call(method, domain=domain, **params)
        except RtIpcError:
            return
        raise AssertionError("accepted invalid request: " + method)

    try:
        client.call("timeline.pause")
        client.call("fluid.create_domain", name=domain, type="matter",
                    domain_min=[-1, -1, -1], domain_max=[1, 1, 1], voxel_size=0.1)
        created = True
        client.call("fluid.set_param", domain=domain, backend="vulkan", boundary="closed")
        client.call("fluid.set_seed_source_settings", domain=domain,
                    settings={"particles_per_cell": 4, "settle": False,
                              "auto_capacity": True})
        client.call("fluid.set_param", domain=domain, max_particles=1)
        rows = [
            {"name": "Water", "substance": "Water", "bounds_min": [-.8, -.8, -.8],
             "bounds_max": [.8, -.2, .8], "temperature_mode": "custom", "temperature_kelvin": 300},
            {"name": "Sand", "substance": "Sand", "bounds_min": [-.8, -.2, -.8],
             "bounds_max": [.8, .2, .8], "temperature_mode": "domain"},
            {"name": "Snow", "substance": "Snow", "bounds_min": [-.8, .2, -.8],
             "bounds_max": [.8, .8, .8], "temperature_mode": "material"},
        ]
        for settings in rows:
            client.call("fluid.add_seed_source", domain=domain, settings=settings)
        recipe = listing()
        assert len(recipe["sources"]) == 3
        assert math.isclose(recipe["sources"][0]["resolved_birth_kelvin"], 300)
        assert recipe["sources"][2]["resolved_birth_kelvin"] < 273.15
        assert recipe["pending_recipe_changes"]
        assert recipe["plan"]["particle_upper_bound"] > 1
        selected_id = recipe["sources"][0]["id"]
        client.call("fluid.select_seed_source", domain=domain, id=selected_id)
        assert listing()["selected_source_id"] == selected_id
        rejected("fluid.select_seed_source", id=999999)
        rejected("fluid.select_seed_source", id=1.5)
        assert listing()["selected_source_id"] == selected_id
        client.call("fluid.select_seed_source", domain=domain, id=0)
        assert listing()["selected_source_id"] == 0
        client.call("fluid.set_seed_source_settings", domain=domain,
                    settings={"auto_capacity": False})
        rejected("fluid.apply_seed_sources")
        client.call("fluid.set_seed_source_settings", domain=domain,
                    settings={"auto_capacity": True})
        client.call("fluid.apply_seed_sources", domain=domain)
        generated = listing()
        assert generated["max_particles"] == generated["plan"]["suggested_capacity"]
        models = client.call("fluid.matter_models", domain=domain)["material_resolution"]
        counts = {}
        for row in models["carriers"]:
            counts[row["substance"]] = counts.get(row["substance"], 0) + row["particles"]
        assert all(counts.get(name, 0) > 0 for name in ("Water", "Sand", "Snow")), counts
        assert not listing()["pending_recipe_changes"]
        first = digest()
        client.call("fluid.apply_seed_sources", domain=domain)
        second = digest()
        assert first["particles"] == second["particles"]
        assert first["position_hash"] == second["position_hash"]
        assert first["velocity_hash"] == second["velocity_hash"]

        # A draft reassignment must not let the live material disappear.
        client.call("substance.derive", name=material, based_on="Water")
        created_material = True
        water = next(row for row in listing()["sources"] if row["name"] == "Water")
        client.call("fluid.update_seed_source", domain=domain, id=water["id"],
                    settings={"substance": material})
        client.call("fluid.apply_seed_sources", domain=domain)
        client.call("fluid.update_seed_source", domain=domain, id=water["id"],
                    settings={"substance": "Water"})
        try:
            client.call("substance.remove", name=material)
        except RtIpcError:
            pass
        else:
            raise AssertionError("removed a material still carried by live particles")
        client.call("fluid.apply_seed_sources", domain=domain)
        client.call("substance.remove", name=material)
        created_material = False
        second = digest()

        snow = next(row for row in listing()["sources"] if row["name"] == "Snow")
        # Authoring changes preserve live state; bad generation does too.
        client.call("fluid.update_seed_source", domain=domain, id=snow["id"],
                    settings={"bounds_min": [-.8, -.4, -.8]})
        assert digest()["position_hash"] == second["position_hash"]
        rejected("fluid.apply_seed_sources")
        assert digest()["position_hash"] == second["position_hash"]
        client.call("fluid.update_seed_source", domain=domain, id=snow["id"],
                    settings={"bounds_min": [-.8, .2, -.8]})
        stable = listing()
        rejected("fluid.update_seed_source", id=snow["id"], settings={"temperature_kelvin": -1})
        after = listing()
        assert after["sources"] == stable["sources"]
        assert after["settings"] == stable["settings"]
        rejected("fluid.update_seed_source", id=snow["id"], settings={"id": 99})
        rejected("fluid.update_seed_source", id=1.5, settings={"enabled": False})
        rejected("fluid.add_seed_source", settings={"name": "Mud", "substance": "Mud"})
        assert len(listing()["sources"]) == 3
        rejected("fluid.set_seed_source_settings", settings={"particles_per_cell": 0})
        rejected("fluid.set_seed_source_settings", settings={"auto_capacity": 1})
        rejected("fluid.set_seed_source_settings", settings={"memory_guard": "yes"})
        rejected("fluid.seed", seed_min=[-.5, 0, -.5], seed_max=[.5, .5, .5])
        print("PASS automatic capacity, layer selection, water/DEM/MPM layers, temperatures, "
              "identity, deterministic replay, "
              "atomic failures and draft recipe isolation")
    finally:
        try:
            if created:
                client.call("fluid.remove_domain", domain=domain)
        finally:
            try:
                if created_material:
                    client.call("substance.remove", name=material)
            finally:
                client.close()


if __name__ == "__main__":
    main()
