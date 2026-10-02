"""Live acceptance for the unified gas + liquid Matter domain."""

import uuid

from rt_ipc import RtIpc


def main():
    client = RtIpc()
    suffix = uuid.uuid4().hex[:8]
    domain = "MatterDomain_" + suffix
    gas_source = "MatterGas_" + suffix
    liquid_source = "MatterLiquid_" + suffix
    created_sources = []
    created_domain = False
    try:
        assert not client.call("fluid.list_domains")["domains"], (
            "Matter acceptance needs an otherwise empty scene")
        created = client.call(
            "fluid.create_domain", name=domain, type="matter",
            domain_min=[20.0, 0.0, 0.0],
            domain_max=[21.0, 1.0, 1.0], voxel_size=0.1)
        created_domain = True
        assert created["type"] == "matter"
        assert created["phases"] == ["gas", "liquid"]
        client.call("fluid.set_param", domain=domain, backend="cpu",
                    boundary="closed", visible=False)

        client.call(
            "flow_source.create", name=gas_source, domain=domain, phase="gas",
            position=[20.5, 0.55, 0.5], radius=0.2,
            density=1.0, temperature=0.2, fuel=0.0,
            velocity=[0.0, 0.2, 0.0])
        created_sources.append(gas_source)
        client.call(
            "flow_source.create", name=liquid_source, domain=domain,
            phase="liquid", position=[20.5, 0.75, 0.5], radius=0.16,
            fluid_particles_per_second=900.0,
            velocity=[0.0, -0.5, 0.0])
        created_sources.append(liquid_source)

        sources = {row["name"]: row for row in
                   client.call("flow_source.list")}
        assert sources[gas_source]["phase"] == "gas"
        assert sources[liquid_source]["phase"] == "liquid"

        for _ in range(8):
            client.call("fluid.step", dt=1.0 / 60.0)

        info = client.call("fluid.get", domain=domain)
        gas_stats = client.call("gas.step_stats", domain=domain)
        assert info["type"] == "matter"
        assert info["phases"] == ["gas", "liquid"]
        assert info["live_state"]
        assert info["particle_count"] > 0, "liquid phase did not emit"
        assert info["active_density_cells"] > 0, "gas phase did not emit"
        assert gas_stats["measured"], "gas solver did not step"
        assert len(client.call("fluid.list_domains")["domains"]) == 1
        print("PASS: one Matter domain advanced gas and liquid phases")
        print("particles:", info["particle_count"])
        print("gas_cells:", info["active_density_cells"])
    finally:
        for name in reversed(created_sources):
            client.call("flow_source.remove", name=name)
        if created_domain:
            client.call("fluid.remove_domain", domain=domain)
        client.close()


if __name__ == "__main__":
    main()
