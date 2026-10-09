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
    "cooling_power", "oxygen_dilution", "flame_persistence",
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
        # ── Part 1: built-ins ──────────────────────────────────────────────
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
            rows[name] = fields
        assert math.isclose(rows["Plastic (PE)"]["liquid_kinematic_viscosity"], 0.30, abs_tol=1e-7)
        assert math.isclose(rows["Wax"]["liquid_kinematic_viscosity"], 5.0e-6, abs_tol=1e-10)
        assert math.isclose(rows["Iron"]["liquid_kinematic_viscosity"], 1.0e-6, abs_tol=1e-10)
        for name in ("Sand", "Gravel", "Soil"):
            for key in ("solver_flip_blend", "solver_internal_friction"):
                assert rows[name][key] == 0.0, (name, key, rows[name][key])
        for name in ("Sand", "Gravel", "Ice"):
            assert rows[name]["granular_transport"] == "dem", (name, rows[name])
        assert rows["Soil"]["granular_transport"] == "mpm"
        assert {"Honey", "Chocolate", "Mud"}.issubset(rows)
        assert rows["Water"]["fluid_extinguishing"] is True
        for name in ("Gasoline", "Alcohol", "Oil", "Plastic (PE)", "Wax"):
            assert rows[name]["fluid_flammable"] is True
            assert rows[name]["flash_kelvin"] > 0.0
            assert rows[name]["autoignition_kelvin"] >= rows[name]["flash_kelvin"]
        assert rejected("substance.get", name="__unknown_substance_contract__"), \
            "unknown substance silently fell back"
        print(f"part 1 PASS: {len(rows)} built-ins, categories right", flush=True)

        # ── Part 2: derive ─────────────────────────────────────────────────
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
        assert {k: v for k, v in honey["fields"].items() if k != "liquid_kinematic_viscosity"} == \
            {k: v for k, v in water.items() if k != "liquid_kinematic_viscosity"}

        # Transactional: one bad field rejects the whole patch.
        assert rejected("substance.set", name=HONEY, fields={"density": 1400.0, "emissivity": 5.0})
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
