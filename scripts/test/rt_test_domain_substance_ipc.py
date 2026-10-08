"""T3 domain material contract; run outside the app after the C++ build.

Creates a temporary domain and substance, performs no simulation steps, and
removes both. Checks rejected inputs, material readback and numerical tuning.
"""

import math
import os

from rt_ipc import RtIpc, RtIpcError


def main():
    client = RtIpc()
    domain = "T3_Domain_" + str(os.getpid())
    material = "T3_Material_" + str(os.getpid())
    created_domain = False
    created_material = False

    def get():
        return client.call("fluid.get", domain=domain)

    def reject(**patch):
        before = get()
        try:
            client.call("fluid.set_param", domain=domain, **patch)
        except RtIpcError as error:
            assert "substance" in str(error).lower(), str(error)
        else:
            raise AssertionError("accepted retired/invalid input: " + repr(patch))
        after = get()
        for key in ("default_substance", "kinematic_viscosity", "viscosity_sweeps",
                    "viscosity_wall_slip", "granular_enabled"):
            assert after[key] == before[key], (key, before[key], after[key])

    try:
        client.call("fluid.create_domain", name=domain, type="fluid",
                    domain_min=[-1, 0, -1], domain_max=[1, 2, 1], voxel_size=0.1)
        created_domain = True
        for name in ("Water", "Honey", "Chocolate", "Mud", "Sand", "Gravel", "Soil", "Wax"):
            client.call("fluid.set_param", domain=domain, default_substance=name)
            got = get()
            fields = client.call("substance.get", name=name)["fields"]
            assert got["default_substance"] == name, got
            assert got["granular_enabled"] == (
                fields["default_constitutive_model"] == "granular"), (name, got)
            assert got["viscosity_sweeps"] == round(fields["solver_viscosity_sweeps"])
            assert math.isclose(got["viscosity_wall_slip"],
                                fields["solver_viscosity_wall_slip"], abs_tol=1e-6)

        reject(default_substance="__unknown_T3_substance__")
        for key, value in {
            "preset": "water", "chemistry_preset": "water",
            "kinematic_viscosity": 0.2, "granular_enabled": True,
            "granular_cohesion": 100, "thermal_freeze_kelvin": 310,
            "thermal_cold_viscosity": 0.1,
        }.items():
            reject(**{key: value, "viscosity_sweeps": 3})

        client.call("substance.derive", name=material, based_on="Honey")
        created_material = True
        client.call("fluid.set_param", domain=domain, default_substance=material,
                    viscosity_wall_slip=0.25, viscosity_sweeps=11)
        client.call("substance.set", name=material,
                    fields={"liquid_kinematic_viscosity": 0.02,
                            "solver_viscosity_wall_slip": 0.75,
                            "solver_viscosity_sweeps": 17})
        got = get()
        assert math.isclose(got["kinematic_viscosity"], 0.02, abs_tol=1e-6), got
        assert got["viscosity_sweeps"] == 11, "material edit reapplied numerical hints"
        assert math.isclose(got["viscosity_wall_slip"], 0.25, abs_tol=1e-6), got
        try:
            client.call("substance.remove", name=material)
        except RtIpcError as error:
            assert domain in str(error), str(error)
        else:
            raise AssertionError("removed the domain's default substance")
        client.call("fluid.set_param", domain=domain, default_substance=material)
        got = get()
        assert got["viscosity_sweeps"] == 17, got
        assert math.isclose(got["viscosity_wall_slip"], 0.75, abs_tol=1e-6), got
        print("PASS domain substance: selection, rejection, live physics, hints, references")
    finally:
        try:
            if created_domain:
                client.call("fluid.remove_domain", domain=domain)
        finally:
            try:
                if created_material:
                    client.call("substance.remove", name=material)
            finally:
                client.close()


if __name__ == "__main__":
    main()
