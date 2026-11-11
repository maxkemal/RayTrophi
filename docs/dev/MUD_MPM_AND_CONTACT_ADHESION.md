# Mud profiles and usable MPM adhesion authoring

## Delivered behavior

`Mud` is now the existing elastoplastic MPM material law (Elastic transport owner),
with a finite deviatoric Hencky yield strain, Kelvin-Voigt viscous stress, and
existing tensile damage/tearing. It no longer uses an ordinary viscous liquid
plus global internal damping as its approximation to an unyielded paste.
`Mud Slurry` provides the fluid/slurry alternative. Both profiles are registered
in the same substance catalogue as Snow, Water and Soil. No solver or adhesion
kernel was added or changed for this work.

The values are initial authoring presets, not fits to a particular measured soil.
This is the engine's existing EVP law, not a new exact Bingham/Herschel-Bulkley
implementation or a water/soil phase-fraction mixture. Automatic moisture-based
transitions, interface fracture-energy history and random clump fields were not
introduced. The user confirmed the existing MPM adhesion actually works; the
immediate adhesion issue was the authoring control.

| Dense Mud property | Starting value |
|---|---:|
| Density | 1700 kg/m3 |
| Young modulus | 80000 Pa |
| Poisson ratio | 0.40 |
| Deviatoric Hencky yield strain | 0.025 |
| Kelvin-Voigt viscosity | 150 Pa s |
| Maxwell relaxation | off (no automatic sub-yield creep) |
| Fracture strain | 0.35 |
| Damage rate | 20 /s |
| FLIP blend | 0 (PIC/APIC) |

These fields already have the Substance UI, Python and IPC authoring/validation
surfaces. UI and both APIs resolve the same catalogue and constitutive core.
Built-ins remain read-only. A project material can be derived and edited:

```python
rt.substance.get("Mud")
rt.substance.derive("My Mud", "Mud")
rt.substance.set("My Mud", {"elastic_yield_strain": 0.04,
                             "elastic_viscosity_pa_s": 250.0})
```

IPC equivalents: `substance.get`, `substance.derive`, `substance.set`, with `name`,
`based_on`, and/or `fields` as appropriate. Existing range/unknown-field and
atomic-patch semantics are retained. Existing explicitly authored material or
source model overrides remain authoritative: reselect/check the material and
actual transport owner rather than assuming a built-in edit overrides them.

## Adhesion UI

The old logarithmic DragFloat had a numeric maximum of 1 MPa, but dragging near
zero advanced too slowly across that range. It is replaced by:

- An absolute logarithmic slider spanning 0 to 1,000,000 Pa.
- A normal editable numeric field, with 100 Pa step and 1000 Pa fast step; no
  Ctrl+click is needed to type a value.
- Units and the kPa conversion in the tooltip. This is stress, not a 0..1 factor.

Both edit the original `contact_adhesion_pa`. The existing `collider.update`
Python/IPC operations and their normal grid adhesion law are unchanged. Values
inside the UI range are clamped, non-finite numeric edits restore the prior value.
Contact friction and internal material cohesion remain separate quantities.

## Verification and user checklist

Completed: `python scripts/test/check_mud_material_contracts.py`, non-build source
contracts and `git diff --check`. New C++ core regression is
`scripts/test/mud_material_test.cpp`: below-yield response, above-yield plastic
return, and viscous stress. It was not compiled or run by the agent.

1. Build the application, including the newly registered MudMaterial.cpp. No
   simulation shader changed in this task, so this task needs no shader rebuild.
2. User-build/run mud_material_test.cpp with the project include directory,
   source/src/Physics/Fluid/MudMaterial.cpp and source/src/Math/Vec3.cpp linked.
3. In a disposable scene, run the existing external acceptance scripts:
   `rt_test_substance_profiles_ipc.py` and `rt_test_domain_substance_ipc.py`.
   They now cover Mud EVP plus Mud Slurry. These mutate their own test objects;
   do not run them in an unsaved production scene.
4. Compare equal-size blocks of Mud, Mud Slurry and Snow on a stationary plane.
   Inspect `fluid.matter_models` transport ownership/law. Dense Mud should yield
   and retain plastic deformation; Slurry should flow. Do not judge only render
   noise: material behavior and output representation are independent.
5. Drag the adhesion slider from zero into kPa/MPa values; type 25000 in the exact
   field with an ordinary click. Read `collider.get` to confirm the Pa value.
6. Test a sliding/tilting plate before a rotating cylinder/tire. Use existing
   Contact Friction and Contact Adhesion. Check attachment and separation without
   adding an artificial launch force. Compare adhesion 0 against a larger value.
7. Recheck CPU/Vulkan where supported, moving/deforming collider velocity, and
   save/reload of custom material and collider values. These are live acceptance
   checks, not claimed complete by source inspection.
