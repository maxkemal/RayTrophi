# Matter: local work and model-specific time integration

2026-11-09. Design boundary, not an implemented local-time solver. The active
runtime still uses one common continuum subcycle. GPU weight-slot remapping is
an independent completed source optimization and does not change that clock.

## User reference and actual bottleneck

User-provided frame 80: 100000 parcels, 27346 active fluid cells, total
1897.12 ms, MAC tiles 120 active / 175 allocated (4.62 MiB), pressure tiles
64 / 82 (0.97 MiB), last MGPCG solve 40 / 288 iterations, residual 7.34e-6,
5 dot checks / 1.29 ms. Viscosity off. Minor setup differences are possible.
Legacy P2G/pressure/G2P/advect zero-ms entries are not zero-cost GPU measurements.
Last-solve pressure iterations are not the sum over the 180 substeps.

`MatterGpuStep.inl` computes a global request from MPM wave/strain stability,
explicit solid viscosity and maximum particle CFL. `grid_stride` is based on
that same request; without a faster external clock it is one. Consequently a
180-step snow requirement runs liquid pressure at the same 180-step frequency.
Sparse pressure already limits much arithmetic to active pages, so reducing empty
cell work alone does not remove this temporal multiplication.

There are two separate projects:

1. Active cell/tile work lists: preserve numerical times but avoid empty work.
2. Model/region-specific clocks: avoid forcing water to use snow's elastic dt.

The user's stated primary concern is temporal over-coupling. A new work-list
optimization must not be presented as completion of that clock change.

## Numerical contract

Do not implement local time by simply skipping pressure on arbitrary iterations
or updating neighboring cells at unrelated times. The current contact kernel
reads BOTH lanes' face velocities, masses and mass gradients from the same
substep, applies equal/opposite impulses, and expects those values before G2P.
Reusing an old liquid field while MPM moves changes contact location, force
integration, FLIP history and divergence; equal/opposite velocity changes alone
do not establish consistent time integration or stability.

Primary references informing this boundary:

- [Batty, Bertails and Bridson: variational solid-fluid coupling](https://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/):
  pressure and solid interaction have a coupled system structure.
- [Chentanez et al.: simultaneous coupling](https://jamesobrien.com/papers/Chentanez-SCP-2006-08/index.html):
  simultaneous pressure/elastic constraints address stability and closed regions.
- [Matthew West: variational and asynchronous time integration](https://lagrange.mechse.illinois.edu/pubs/We2004/):
  independent local clocks require a specifically derived integrator, not a loop skip.

These sources motivate the engineering contract; they do not certify this
project's future implementation. MPM/APIC and staggered MAC adaptations still
need their own derivation, reference tests and acceptance.

## Intended architecture

A shared scheduler service owns lane/component time budgets and synchronization
requirements. Liquid CFL/viscous/boundary-motion limits and MPM wave/strain/viscous
limits are computed separately from canonical SoA and physical fields. Snow keeps
its conservative hardening/stiffness bounds; no artificial modulus reduction or
hard 32-step cap. Add active material/pressure-component graphs and conservative
motion/contact halos before claiming spatially local schedules.

At every interface event, both states must represent the same physical time.
Interface momentum is accumulated once, equal and opposite; a corrector closes
contact and divergence residuals. Force/source/phase-transfer integration uses
the lane's actual elapsed dt, and FLIP baselines belong to the corresponding
liquid solve interval. Stale occupancy/mass gradients must not drive contacts.
For stiff or unresolved interfaces, synchronize the affected connected region
or reject/retry its interval from a valid checkpoint. A fallback must be visible;
it must not silently reinstate a domain-wide clock under a multirate label.

Pressure support covers each connected fluid solve region with its correct
boundary conditions. It is not an independently solvable scalar per visible cell.
Active work topology needs P2G/G2P support, advection reach and contact/solid bands;
shared faces retain one owner. Unallocated, empty, dormant and outside-domain
states remain distinct. Canonical flat particle/mesh data remains authoritative.

## Delivery gates

1. Separate lane demand and active-work diagnostics through one core service.
   Keep the existing synchronized reference integrator selectable for comparison.
2. Independent disconnected regions and separate lane event budgets; conservative
   interface prediction, accumulation/corrector and explicit stable fallback.
   GPU kernels consume scheduled work lists instead of dense domain dispatches
   where the mathematical operator permits it. This is a numerical change.
3. Shared UI/Python/IPC policy and diagnostics, serialization, fingerprint,
   checkpoint/cache time state and transactional validation. A scheduler is not
   complete as an internal or UI-only flag.
4. Analytic uniform flow plus distant stiff snow; disconnected pools; zero force;
   hydrostatic water; impact on light/stiff snow; thin interfaces, tile crossing,
   moving glued colliders, pore/phase exchange and rewind. Check conservation,
   contact/pressure residuals and dt convergence, not only visual appearance.
5. User build and explicitly authorized live tests. Report liquid pressure-solve
   count, MPM update count, active work, interface sync/corrector count, fallbacks,
   real GPU timestamps and full-step time. Claim no speedup until measured.

The independent weight-upload part can share a C++ build with this work, but it
must not trigger a claim that the main 180-step coupling problem is solved.
