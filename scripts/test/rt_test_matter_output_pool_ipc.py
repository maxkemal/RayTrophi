"""External IPC acceptance after a user build; requires no existing domains.

Leaves two editable test materials in the scene; removes the temporary sources/domain.
Does not launch the app or compile anything.
"""
import uuid

from rt_ipc import RtIpc


def main():
    client = RtIpc()
    domain = "MatterOutputPool_" + uuid.uuid4().hex[:8]
    sources = []
    created = False
    try:
        assert not client.call("fluid.list_domains")["domains"], "Use an empty scene"
        before = client.call("material.list")
        ok, _ = client.try_call("material.create", type="substance:NotInCatalogue")
        assert not ok
        assert client.call("material.list") == before, "Invalid preset allocated material"
        water = client.call("material.create", type="substance:Water", name="Pool Water")
        sand = client.call("material.create", type="substance:Sand", name="Pool Sand")
        transmission = client.call("material.get_param", material_name=water, param="transmission")
        roughness = client.call("material.get_param", material_name=sand, param="roughness")
        assert transmission > 0.9 and roughness > 0.8
        client.call("material.set_param", material_name=sand, param="roughness", value=0.7)
        assert abs(client.call("material.get_param", material_name=sand, param="roughness") - 0.7) < 1e-5
        client.call("fluid.create_domain", name=domain, type="matter",
                    domain_min=[0, 0, 0], domain_max=[2, 2, 2], voxel_size=0.125)
        created = True
        client.call("fluid.set_param", domain=domain, backend="vulkan", boundary="closed",
                    preset="water", max_particles=1000)
        client.call("fluid.set_whitewater", domain=domain, enabled=False)
        for substance, material, representation in [("Water", water, "sdf"),
                                                    ("Sand", sand, "splat")]:
            client.call("fluid.set_substance_material", domain=domain, substance=substance,
                        material=material, representation=representation)
        bindings = {b["substance"]: b for b in
                    client.call("fluid.get", domain=domain)["substance_materials"]}
        assert bindings["Water"]["material"] == water
        assert bindings["Sand"]["material"] == sand
        assert bindings["Water"]["effective_representation"] == "sdf"
        assert bindings["Sand"]["effective_representation"] == "splat"
        for i, (substance, model, weight) in enumerate([
                ("Water", "fluid", 1.0), ("Sand", "granular", 3.0)]):
            name = domain + "_" + substance
            client.call("flow_source.create", name=name, domain=domain, phase="liquid",
                        fluid_substance=substance, initial_constitutive_model=model,
                        position=[0.7 + i * 0.6, 1.2, 1], radius=0.15,
                        fluid_particles_per_second=240000, particle_pool_weight=weight)
            sources.append(name)
        original = client.call("flow_source.get", name=sources[0])
        for invalid in [0, -1, 1001]:
            ok, _ = client.try_call("flow_source.update", name=sources[0],
                                    particle_pool_weight=invalid)
            assert not ok
            assert client.call("flow_source.get", name=sources[0]) == original
        client.call("fluid.step", dt=1 / 24)
        rows = [client.call("flow_source.get", name=name) for name in sources]
        grants = [row["pool_granted_particles"] for row in rows]
        assert grants == [250, 750], grants
        assert all(row["pool_requested_particles"] >= row["pool_granted_particles"]
                   for row in rows)
        assert client.call("fluid.get", domain=domain)["particle_count"] <= 1000
        models = client.call("fluid.matter_models", domain=domain)
        assert models["mixed_transport_ready"]
        assert not models["mixed_execution"]["step_held"]
        print("PASS: editable presets, separate SDF/splat/material bindings, weighted pool, atomic rejection")
        print("Visual check remains: Water SDF + Sand splat in RT/Solid/RayFusion.")
    finally:
        for name in reversed(sources):
            client.call("flow_source.remove", name=name)
        if created:
            client.call("fluid.remove_domain", domain=domain)
        client.close()


if __name__ == "__main__":
    main()
