"""Substance library acceptance (docs/dev/MADDE_TIPLERI_TASARIMI.md T1).

Part 1 (read-only): the built-in table is complete, categorized and queryable.
Part 2 (derive): a project substance inherits from its base, stores only the
fields it overrides, follows later base edits, and every bad edit is rejected
without changing anything. Built-ins are read-only. Removal is refused while
anything still names the substance. Leaves the library as it found it.

    python scripts/test/rt_test_substance_profiles_ipc.py
"""

import math

from rt_ipc import RtIpc, RtIpcError


REQUIRED_FLUIDS = {"Water", "Gasoline", "Alcohol", "Oil", "Plastic (PE)", "Wax"}
EXPECTED_CATEGORY = {"Water": "liquid", "Sand": "granular", "Soil": "granular",
                     "Gasoline": "fuel", "Oil": "fuel", "Iron": "solid", "Wood (Oak)": "solid"}
NUMERIC_FIELDS = {
    "density", "liquid_density", "specific_heat", "conductivity",
    "liquid_kinematic_viscosity", "ignition_kelvin", "flash_kelvin",
    "autoignition_kelvin", "melt_kelvin", "boiling_kelvin",
    "latent_heat_fusion", "latent_heat_vaporization", "vaporization_rate",
    "flame_persistence", "liquid_cold_viscosity", "elastic_thermal_viscosity_power",
    "elastic_thermal_yield_power", "elastic_thermal_relaxation_power",
    "granular_friction_degrees", "granular_cohesion",
}
HONEY, DARK, COLLIDER = "T1_Honey", "T1_HoneyDark", "T1_SubstanceRefCollider"


def main():
    client = RtIpc()

    def rejected(method, **params):
        try:
            client.call(method, **params)
        except RtIpcError as error:
            return str(error)
        return None

    def get(name):
        return client.call("substance.get", name=name)

    try:
        # â”€â”€ Part 1: built-ins â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        listing = client.call("substance.list")["substances"]
        names = [row["name"] for row in listing]
        assert len(names) == len(set(names)), "duplicate substance ids"
        assert REQUIRED_FLUIDS.issubset(names), REQUIRED_FLUIDS - set(names)
        by_name = {row["name"]: row for row in listing}
        for name, category in EXPECTED_CATEGORY.items():
            assert by_name[name]["category"] == category, (name, by_name[name])
            assert by_name[name]["builtin"] and not by_name[name]["based_on"]
        stale = [n for n in names if not by_name[n]["builtin"] and n.startswith("T1_")]
        assert not stale, ("a previous run left project substances behind", stale)

        rows = {}
        for name in names:
            row = get(name)
            assert row["name"] == name
            fields = row["fields"]
            assert NUMERIC_FIELDS.issubset(fields)
            assert all(math.isfinite(fields[key]) for key in NUMERIC_FIELDS)
            assert fields["density"] > 0.0 and fields["liquid_density"] > 0.0
            assert fields["specific_heat"] > 0.0
            assert fields["liquid_kinematic_viscosity"] >= 0.0
            if fields["meltable"]:
                assert fields["liquid_cold_viscosity"] >= fields["liquid_kinematic_viscosity"]
            rows[name] = fields
        assert math.isclose(rows["Plastic (PE)"]["liquid_kinematic_viscosity"], 0.30, abs_tol=1e-7)
        assert math.isclose(rows["Wax"]["liquid_kinematic_viscosity"], 5.0e-6, abs_tol=1e-10)
        assert math.isclose(rows["Iron"]["liquid_kinematic_viscosity"], 1.0e-6, abs_tol=1e-10)
        for name in ("Snow", "Ice"):
            assert rows[name]["liquid_kinematic_viscosity"] == rows["Water"]["liquid_kinematic_viscosity"]
            assert rows[name]["liquid_density"] == rows["Water"]["liquid_density"]
        for name in ("Honey", "Oil", "Chocolate"):
            assert rows[name]["solver_internal_friction"] == 0.0
        assert math.isclose(rows["Chewing Gum"]["granular_residual_strength"], 0.02, abs_tol=1e-6)
        assert rows["Chewing Gum"]["elastic_thermal_yield_power"] == 1.0
        assert rows["Chewing Gum"]["elastic_thermal_relaxation_power"] == 1.0
        assert rows["Stone"]["liquid_kinematic_viscosity"] == 0.5
        assert rows["Stone"]["liquid_density"] == rows["Stone"]["density"]
        assert math.isclose(rows["Gasoline"]["flash_kelvin"], 230.37, abs_tol=1e-3)
        assert math.isclose(rows["Alcohol"]["flash_kelvin"], 285.93, abs_tol=1e-3)
        assert math.isclose(rows["Alcohol"]["autoignition_kelvin"], 638.15, abs_tol=1e-3)
        for name in ("Sand", "Gravel", "Soil"):
            for key in ("solver_flip_blend", "solver_internal_friction"):
                assert rows[name][key] == 0.0, (name, key, rows[name][key])
        for name in ("Sand", "Gravel", "Ice"):
            assert rows[name]["granular_transport"] == "dem", (name, rows[name])
        assert rows["Soil"]["granular_transport"] == "mpm"
        # Snow (MADDE_UI_TEK_OTORITE U5): a continuum that compacts; DEM
        # material lives in grain_* fields, liquids carry surface tension.
        assert by_name["Snow"]["category"] == "granular" and by_name["Snow"]["builtin"]
        assert rows["Snow"]["granular_transport"] == "mpm"
        assert rows["Snow"]["granular_snow_plasticity"] is True
        assert rows["Snow"]["granular_snow_hardening"] > 0.0
        assert 0.0 < rows["Snow"]["granular_snow_critical_compression"] < 1.0
        assert 0.0 < rows["Snow"]["granular_snow_critical_stretch"] < 1.0
        assert (0.0 < rows["Snow"]["granular_snow_min_hardening"]
                <= rows["Snow"]["granular_snow_max_hardening"])
        assert rows["Sand"]["granular_compaction_hardening"] == 0.0
        assert math.isclose(rows["Sand"]["grain_friction"], 0.5, abs_tol=1e-6)
        # Sand retains pore water; the built-in capacity is 10% of grain volume.
        assert math.isclose(rows["Sand"]["grain_water_capacity_fraction"], 0.1,
                            abs_tol=1e-6)
        assert math.isclose(rows["Water"]["liquid_surface_tension_n_m"], 0.072, abs_tol=1e-6)
        assert {"Honey", "Chocolate", "Mud"}.issubset(rows)
        # Elastic solids on the MPM lane (docs/dev/ELASTIK_MPM.md, 2026-11-02).
        for name in ("Gelatin", "Dough", "Chewing Gum", "Mud"):
            assert rows[name]["default_constitutive_model"] == "elastic", (name, rows[name])
            assert rows[name]["elastic_young_modulus"] > 0.0
        assert rows["Gelatin"]["elastic_yield_strain"] == 0.0
        assert rows["Dough"]["elastic_yield_strain"] > 0.0
        assert by_name["Mud"]["category"] == "solid"
        assert rows["Mud"]["elastic_yield_strain"] > 0.0
        assert rows["Mud"]["elastic_viscosity_pa_s"] > 0.0
        assert rows["Mud"]["elastic_relaxation_s"] == 0.0
        assert rows["Mud Slurry"]["default_constitutive_model"] == "fluid"
        assert rows["Mud Slurry"]["liquid_kinematic_viscosity"] > 0.0
        assert rows["Chewing Gum"]["elastic_relaxation_s"] > 0.0
        # P5 2026-10-19: the quench chemistry is retired; water is just not a fuel.
        assert rows["Water"]["fluid_flammable"] is False
        assert not {"fluid_extinguishing", "cooling_power", "oxygen_dilution"} & set(rows["Water"])
        for name in ("Gasoline", "Alcohol", "Oil", "Plastic (PE)", "Wax"):
            assert rows[name]["fluid_flammable"] is True
            assert rows[name]["flash_kelvin"] > 0.0
            assert rows[name]["autoignition_kelvin"] >= rows[name]["flash_kelvin"]
        assert rejected("substance.get", name="__unknown_substance_contract__"), \
            "unknown substance silently fell back"
        print(f"part 1 PASS: {len(rows)} built-ins, categories right", flush=True)

        # â”€â”€ Part 2: derive â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        water = rows["Water"]
        assert rejected("substance.set", name="Water", fields={"density": 1.0}), \
            "a built-in accepted an edit"
        assert rejected("substance.derive", name="Water", based_on="Oil"), "name collision accepted"
        assert rejected("substance.derive", name="T1_X", based_on="__nope__"), "unknown base accepted"

        client.call("substance.derive", name=HONEY, based_on="Water")
        honey = get(HONEY)
        assert honey["based_on"] == "Water" and not honey["builtin"]
        assert honey["fields"] == water and honey["overridden"] == [], honey

        before_transport = get(HONEY)
        assert rejected("substance.set", name=HONEY,
                        fields={"density": 1400.0, "granular_transport": "xpbd"})
        assert get(HONEY) == before_transport, "invalid transport patch was not transactional"
        client.call("substance.set", name=HONEY, fields={"granular_transport": "dem"})
        assert get(HONEY)["fields"]["granular_transport"] == "dem"
        client.call("substance.set", name=HONEY, fields={"granular_transport": None})
        assert get(HONEY) == before_transport, "transport Revert did not restore inheritance"

        client.call("substance.set", name=HONEY, fields={"liquid_kinematic_viscosity": 2.0e-3})
        honey = get(HONEY)
        assert math.isclose(honey["fields"]["liquid_kinematic_viscosity"], 2.0e-3, rel_tol=1e-6)
        assert honey["overridden"] == ["liquid_kinematic_viscosity"], honey["overridden"]
        stable = get(HONEY)
        assert rejected("substance.set", name=HONEY,
                        fields={"elastic_thermal_relaxation_power": -1.0})
        assert get(HONEY) == stable
        assert rejected("substance.set", name=HONEY,
                        fields={"liquid_cold_viscosity": 1.0e-3})
        assert get(HONEY) == stable, "rejected freeze curve edit changed the profile"
        assert {k: v for k, v in honey["fields"].items() if k != "liquid_kinematic_viscosity"} == \
            {k: v for k, v in water.items() if k != "liquid_kinematic_viscosity"}

        # Transactional: one bad field rejects the whole patch.
        assert rejected("substance.set", name=HONEY, fields={"density": 1400.0, "emissivity": 5.0})
        # Retired 2026-10-19: surface heating is a film against the body capacity.
        assert rejected("substance.set", name=HONEY, fields={"thermal_response": 3.0})
        assert rejected("substance.set", name=HONEY, fields={"fluid_extinguishing": True})
        assert rejected("substance.set", name=HONEY, fields={"solver_density_correction": 1.0})
        assert rejected("substance.set", name=HONEY, fields={"no_such_field": 1.0})
        assert rejected("substance.set", name=HONEY,
                        fields={"meltable": True, "melt_kelvin": 400.0, "boiling_kelvin": 300.0})
        assert get(HONEY)["fields"]["density"] == water["density"], "a rejected patch changed a field"
        assert get(HONEY)["overridden"] == ["liquid_kinematic_viscosity"]

        # Inheritance: DARK follows HONEY's later edits, except what it overrides.
        client.call("substance.derive", name=DARK, based_on=HONEY)
        client.call("substance.set", name=DARK, fields={"char_color": [0.2, 0.1, 0.05]})
        client.call("substance.set", name=HONEY, fields={"density": 1420.0})
        dark = get(DARK)
        assert dark["fields"]["density"] == 1420.0, "derived substance did not follow its base"
        assert math.isclose(dark["fields"]["liquid_kinematic_viscosity"], 2.0e-3, rel_tol=1e-6)
        assert dark["overridden"] == ["char_color"], dark["overridden"]

        # Revert: null drops the override and the field follows the base again.
        client.call("substance.set", name=HONEY, fields={"liquid_kinematic_viscosity": None})
        assert get(HONEY)["fields"]["liquid_kinematic_viscosity"] == water["liquid_kinematic_viscosity"]
        assert get(DARK)["fields"]["liquid_kinematic_viscosity"] == water["liquid_kinematic_viscosity"]

        # Removal refused while referenced: by another substance, then by a collider.
        assert rejected("substance.remove", name=HONEY), "removed a base with a dependant"
        assert rejected("substance.remove", name="Water"), "removed a built-in"
        colliders = {c["name"] for c in client.call("collider.list")["colliders"]}
        client.call("collider.update" if COLLIDER in colliders else "collider.create",
                    name=COLLIDER, source_mode="sphere", sphere_center=[0, -50, 0],
                    sphere_radius=.1, enabled=False, msf_substance=DARK)
        error = rejected("substance.remove", name=DARK)
        assert error and COLLIDER in error, ("removed a substance a collider uses", error)
        client.call("collider.update", name=COLLIDER, msf_substance="Wood (Oak)")
        client.call("substance.remove", name=DARK)
        client.call("substance.remove", name=HONEY)
        names_after = [row["name"] for row in client.call("substance.list")["substances"]]
        assert names_after == names, "library not back to its starting state"
        print("part 2 PASS: derive, override, inherit, revert, reject, refuse-remove", flush=True)
        print("PASS substance library", flush=True)
    finally:
        for name in (DARK, HONEY):
            try:
                client.call("substance.remove", name=name)
            except RtIpcError:
                pass
        try:
            client.call("collider.remove", name=COLLIDER)
        except RtIpcError:
            pass
        client.close()


if __name__ == "__main__":
    main()
