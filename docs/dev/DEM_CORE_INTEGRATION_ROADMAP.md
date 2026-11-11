# General DEM integration and time-cost roadmap

2026-11-10. Applies to every DEM material, not just bonded ice.

## Current delivery status

Phase 1 source refactor implemented: `DemContactTimeBounds.h` is consumed by
`MatterGrainGpu.cpp`. Normal, sliding, rolling and capillary stiffness limits
are explicit independent inputs; no preset or particle state enters the module.
The existing reference integrator, accuracy/travel limits, adaptive stiffness
policy, common-clock hooks and canonical publication remain in place.
The damped stability expression uses an algebraically equivalent reciprocal
form to avoid subtraction cancellation at large damping ratios.
No new implicit solver, local-time runtime or speedup is delivered by this refactor.
Standalone MSVC C++17 assertion-enabled time-bounds fixture compiled and passed
on 2026-11-10 after explicit user authorization. Five standalone tests passed;
small Sand/Gravel/Ice IPC smoke passed. See `DEM_CPP_AND_IPC_ACCEPTANCE.json`.
The network fixture did not link standalone; GPU fixture remains pending.

## Measured problem

59130 grains cost 485.90 ms Fluid Step even after all 335953 bonds have failed:
262 microsteps / 1310 dispatches remain. Intact case: 971.41 ms and 576–582
microsteps. These are different dynamical states, not a controlled speedup ratio.
See `DEM_BOND_COST_LIVE.json`. CPU host wait is not device kernel timing.

## Numerical and material contract

- Keep physical masses, moments of inertia and material stiffness. No hidden
  mass scaling, new stiffness softening or arbitrary microstep ceiling.
  Existing explicit adaptation is legacy behavior and must remain visible in
  diagnostics during comparison; it is not the optimization strategy.
- Preserve elastic normal response, tangential and rolling history, twisting,
  Coulomb sliding, restitution, capillary force/rupture and bonded force/moment
  laws. Implicit integration changes their discrete treatment, not their meaning.
- Wet bridges and irreversible damage are history variables. Nonlinear solver
  trial iterations must not commit history repeatedly. Commit accepted physical
  intervals once; failed attempts restore state, histories and energy ledgers.
- Connected contact/bond graphs determine interacting regions. Spatial cells
  alone do not define independently integrable regions. New contact candidates
  need conservative swept bounds; connected fluid pressure remains nonlocal.
- Adaptive rigidification is optional later work. It cannot replace the general
  DEM contact solver or silently remove internal stress needed for fracture.

## Phase 1: explicit reference and reproducible acceptance

Implemented common contact bounds and analytic C++ fixture:
`scripts/test/dem_contact_time_bounds_test.cpp`. It checks mass/stiffness
scaling, damping stability boundary, rolling/sliding/bridge limits and a highly
damped positive finite result. The standalone fixture compiled and passed with assertions enabled.

Next measurement uses existing GPU timestamp support: accumulate grain hash,
list construction, integration, bond and coupling kernel costs separately from
host packing, submission/wait and transfer. Compare identical initial state,
physical parameters and simulated interval; report repeated median and spread.

## Phase 2: material-preserving implicit contact reference

Derive the discrete problem before GPU rollout. Start with rotating sphere
pairs and sphere/wall, then frictional piles. Use six velocity degrees of freedom
per sphere with the actual inertia. Solve finite-compliance normal contacts,
tangential/rolling response and friction constraints to a measured residual.
An AVBD/augmented-Lagrangian formulation is a candidate, not an accepted backend
until it reproduces the project's contact laws and convergence tests.

Mandatory gates: normal restitution and contact-duration dt sweeps, oblique
impact/sliding/rolling, large mass/radius/stiffness ratios, resting piles, wet
bridge formation/rupture, bond bending/twist/fracture. Compare with analytic
pair solutions where available and refined explicit solutions otherwise.
Implicit stability alone does not permit unresolved impact or fracture events;
reduce the interval to satisfy accuracy/event criteria. Include numerical
energy dissipation in the acceptance budget.

## Phase 3: GPU contact graph and iterative solve

Persistent pair identity comes from stable particle IDs, not current ordering.
One canonical pair owner computes force/impulse and moment; distribute equal
and opposite contributions without duplicate history updates. Use reusable
contact candidates with displacement/shape invalidation, compact active lists,
GPU residual reductions and batch submission. Avoid host round trips inside
iterations. Define overflow rejection and atomic publication first.

## Phase 4: time and multiphysics coupling

Each interaction region requests an interval from motion, contact accuracy,
constitutive evolution and coupling time scales. Track interface interval/tau;
force coefficients, solid fraction and state are available on the fast clock.
Cheap drag/contact/pressure-force application runs on the required fast clock;
expensive pressure solves can run less frequently only after dt convergence.
Net impulse is an accounting output, not a substitute for resolving fast
coupling. Equal/opposite momentum by itself does not prove correct evolution.

Claude's snow/water result is a required counterexample: a long fully implicit
drag exchange conserved momentum yet produced a different solution because
solid contact with ground acted later. Include a supported granular bed in
fluid and multiple exchange rates in DEM acceptance as well.

Keep the synchronized implementation as a test reference. Do not silently
switch a failed local-time solve into a domain-wide runtime fallback. Explicit
rejection/interval refinement must restore all involved owners transactionally;
otherwise hold and report the failure. Pressure force must be integrated over
the same physical interval, with reactions reaching the correct owner once.
Do not use the liquid `var_fluid_phi` slot for Vulkan solid occupancy; coordinate
with the MPM/pressure owner before changing shared buffers or clocks.

## Delivery completeness

A selectable new solver/time policy requires one core service with UI, Python
and IPC bindings, validation, save/load and scene fingerprint. Canonical flat
particle data stays authoritative. Checkpoint/cache must include history and
accepted scheduler state. No UI-only toggle or unexposed runtime mode counts
as complete. This first internal refactor introduces no such authoring mode.

Build: user incremental C++ build (MatterGrainGpu.cpp); header-only module.
Standalone fixture example: compile dem_contact_time_bounds_test.cpp as C++17
with source/include on the include path and assertions enabled, then run it.
No shader changes in phase 1. Do not claim physics/performance acceptance from
source checks alone.

## Primary references

- Itasca: explicit stiffness/mass and kinematic timestep constraints:
  https://docs.itascacg.com/pfc600/pfc/docproject/source/manual/numerical_simulations_with_pfc/pfc_formulation/timestep_constraints.html
- Chrono: compliant versus non-smooth contact formulations:
  https://api.projectchrono.org/collisions.html
- Augmented Vertex Block Descent, SIGGRAPH 2025:
  https://graphics.cs.utah.edu/research/projects/avbd/
- Small Steps in Physics Simulation: larger steps/more iterations do not
  automatically win on accuracy or performance: https://mmacklin.com/smallsteps.pdf
