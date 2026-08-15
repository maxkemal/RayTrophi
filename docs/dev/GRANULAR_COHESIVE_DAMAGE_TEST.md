# Granular Cohesive Damage Test

> **Durum:** AKTIF — Kabul testi protokolu; kohezyonlu granuler asamasi surerken kullanilir.

This milestone covers bonded granular materials such as packed snow, damp sand,
soil clods and weak aggregates. It is a continuum mesoscale model, not literal
molecular dynamics.

## Brittle drop setup

- Backend: Vulkan Compute
- Friction Angle: `32 deg`
- Cohesion: `1500 Pa`
- Tensile Cutoff: `500 Pa`
- Young Modulus: `350000 Pa`
- Poisson Ratio: `0.22`
- Hardening: `1.5`
- Fracture Strain: `0.01`
- Damage Rate: `12`
- Allow Rebonding: off
- Max Granular Solver Substeps: `16`
- Reseeding: off (forced by granular mode)

Drop a compact block from about 2 m onto the closed domain floor. It should
carry shape briefly, develop local damage on impact, split into irregular pieces
and leave frictional debris. Damage weakens cohesion and tension only: broken
debris must still make a granular pile instead of becoming liquid or ballistic
dust.

Acceptance:

- `Granular damaged > 0`
- `Granular max damage > 0`
- `Granular invalid = 0`
- particle count remains constant
- `Granular detached` rises near cracks/free fragments, not across the entire
  block before impact
- severe damage remains localized: fewer than 50% of particles should exceed
  50% damage, fewer than 25% should exceed 90% damage, and fewer than 50%
  should be fully detached after the 90-step drop

The broad `damaged` count uses a very low `damage > 0.0001` threshold and is a
microdamage activity indicator, not a severe-fracture percentage. Use the 10%,
50%, 90% bands and mean damage before deciding that fracture is domain-wide.

`Fracture Strain` is the maximum irreversible Rankine bond-opening onset, not
an instant full-break threshold. The history stores the largest tensile
overstress strain a particle has experienced; it is not summed once per solver
substep. This makes fracture invariant to elastic substep count and prevents a
small persistent APIC/free-surface error from eventually breaking an unloaded
body. It is deliberately separate from accumulated
Drucker-Prager plastic strain: compression and frictional rearrangement may
yield, flow and harden without being mislabeled as a broken cohesive bond.
The opening history is driven by tensile overstress of the maximum principal
stress, so shear can still create a crack through its tensile principal
direction. Above the threshold, `Damage Rate` controls exponential softening as
`1-exp(-rate * excess opening)`. The return map removes only the failed
principal tensile component instead of shifting the entire hydrostatic stress.

The solver now substeps the complete P2G, grid-boundary/stress, G2P and particle
advection loop according to the elastic wave-speed CFL limit. Telemetry reports
the requested/effective Young modulus, required substeps and substeps actually
run. With this setup the required count is normally about `15`; a maximum of
`16` should therefore recover approximately `350000 Pa` instead of silently
softening the material. If the required count exceeds the authored maximum,
the remaining stiffness cap is intentional and visible in telemetry.

Velocity and affine damping are converted to their per-substep roots, preserving
the authored frame-level damping instead of applying it 15 times. Density/NanoVDB
and fracture statistics run only after the final substep. The current Vulkan
implementation prioritizes a physically complete test point; each substep still
crosses the host/device synchronization boundary, so performance optimization
of a device-resident multi-substep command sequence is the next compute task.
For closed/periodic domains the large stress, plasticity and damage arrays are
already uploaded only on the first substep and downloaded only on the last;
open domains intentionally synchronize them each pass because outflow can
remove/swap particle indices between substeps.

The elastic predictor is deformation-gradient based. Each substep updates the
recoverable `F`, extracts its polar rotation, evaluates elastic strain in that
corotated frame, applies the Drucker-Prager/Rankine return maps, and stores the
projected elastic deformation back on the particle. Large plastic motion stays
in particle positions and plastic history rather than accumulating as
non-objective Cauchy stress during rigid rotation.

Granular advection is Lagrangian: after G2P, particle positions advance with
their post-grid particle velocity. The liquid marker path still uses midpoint
sampling of the MAC field. Applying that liquid resampling a second time to an
MPM particle used a different interpolation kernel than G2P and introduced
free-surface/grid-crossing deformation during otherwise uniform free fall.

The IPC test samples steps 15/30/45/60/75/90. Step 30 is still before floor
impact for the authored height, so every damage count, including microdamage,
must remain zero. Recoverable elastic motion below `Fracture Strain` is not
damage: jittered particle quadrature and grid crossing are permitted only while
the mean stays below 50% of the onset and the peak below 90%. The safety margin
still catches a constitutive state approaching fracture before contact, while
the damage count catches any actual threshold crossing. This separates
constitutive drift from real impact fracture instead of diagnosing only the
completely settled frame.

An older instrumentation path collected granular counters before the GPU tail
call, then the tail reset the complete stats block. Its characteristic symptom
was visible separation with every granular counter exactly zero. Counters are
now collected after the final tail call.

Run the IPC contract and impact smoke test from a separate terminal:

`python scripts\test\rt_test_granular_damage.py`

`fluid.step` advances the complete `SimulationWorld`; it is not a target-domain
step operation. The test therefore refuses to run when another enabled fluid or
gas domain exists. Run it in an otherwise empty scene or disable the other
domains yourself. The test deliberately does not toggle them automatically,
because disabling a grid domain invalidates its runtime grid and would modify
the scene under test. When the terminal working directory is `x64\Release`, use
the canonical source path `python ..\..\scripts\test\rt_test_granular_damage.py`.

The test also verifies the seed lifecycle before the drop. `fluid.seed` now
accepts `persistent`: `false` creates live particles only, while `true` stores
the Seed Box as a reset-time initial-state recipe. `fluid.clear` accepts
`clear_seed`: `false` clears only the live particles; `true` also disarms the
Seed Box/Fill Level recipe. Seeding and clearing act on the unified grid-domain
particle state only; the legacy `FluidObject` remains an empty editor/render
mirror and is not stepped as a second copy.

## Rebonding comparison

Repeat with Allow Rebonding on and Healing Rate `0.5-1.0`. Compressed fragments
may form a new clump; separated airborne fragments must not heal. Dry Sand keeps
Cohesion and Tensile Cutoff at zero, Rebonding off, and should keep
`Granular damaged = 0` because it has no bonds to damage.

## Next solver gate

The next quality gate is multi-field MPM contact. Particles on different
fracture branches that occupy the same grid cell must retain separate velocity
fields. A single-field P2G transfer can otherwise average newly separated chunks
back together even after their bonds are broken. Multi-field contact is required
before arbitrary tearing and rejoining can be called production-ready.
