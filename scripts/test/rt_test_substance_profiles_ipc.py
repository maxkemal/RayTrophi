"""Read-only acceptance for the canonical physical substance table."""

import math

from rt_ipc import RtIpc, RtIpcError


REQUIRED_FLUIDS = {"Water", "Gasoline", "Alcohol", "Oil", "Plastic (PE)", "Wax"}
NUMERIC_FIELDS = {
    "density", "liquid_density", "specific_heat", "conductivity",
    "liquid_kinematic_viscosity", "ignition_kelvin", "flash_kelvin",
    "autoignition_kelvin", "melt_kelvin", "boiling_kelvin",
    "latent_heat_fusion", "latent_heat_vaporization", "vaporization_rate",
    "cooling_power", "oxygen_dilution", "flame_persistence",
    "granular_friction_degrees", "granular_cohesion",
}


def main():
    client = RtIpc()
    try:
        names = client.call("msf.substances")["substances"]
        assert len(names) == len(set(names)), "duplicate substance ids"
        assert REQUIRED_FLUIDS.issubset(names), REQUIRED_FLUIDS - set(names)

        rows = {}
        for name in names:
            row = client.call("msf.substance", name=name)["substance"]
            assert row["name"] == name
            assert NUMERIC_FIELDS.issubset(row)
            assert all(math.isfinite(row[key]) for key in NUMERIC_FIELDS)
            assert row["density"] > 0.0
            assert row["liquid_density"] > 0.0
            assert row["specific_heat"] > 0.0
            assert row["liquid_kinematic_viscosity"] >= 0.0
            rows[name] = row

        assert math.isclose(rows["Plastic (PE)"]["liquid_kinematic_viscosity"],
                            0.30, rel_tol=0.0, abs_tol=1e-7)
        assert math.isclose(rows["Wax"]["liquid_kinematic_viscosity"],
                            5.0e-3, rel_tol=0.0, abs_tol=1e-8)
        assert math.isclose(rows["Iron"]["liquid_kinematic_viscosity"],
                            1.0e-6, rel_tol=0.0, abs_tol=1e-10)
        assert rows["Water"]["fluid_extinguishing"] is True
        for name in ("Gasoline", "Alcohol", "Oil", "Plastic (PE)", "Wax"):
            assert rows[name]["fluid_flammable"] is True
            assert rows[name]["flash_kelvin"] > 0.0
            assert rows[name]["autoignition_kelvin"] >= rows[name]["flash_kelvin"]

        try:
            client.call("msf.substance", name="__unknown_substance_contract__")
        except RtIpcError:
            pass
        else:
            raise AssertionError("unknown substance silently fell back")

        print("PASS: canonical substance table is complete and queryable")
        print("substances:", len(rows))
    finally:
        client.close()


if __name__ == "__main__":
    main()
