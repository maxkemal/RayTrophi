# GPU-first implementation decision

> **Durum:** AKTIF — Granuler cozucunun GPU-once icra plani.

The production path is Vulkan Compute on the RTX 3060. CPU is a small reference
backend, not the first performance target.

The current profile (100,000 particles, 22,011 active cells, 20.37 ms total)
already provides a useful budget: P2G 5.55 ms, pressure 9.92 ms, G2P 3.86 ms,
and density conversion 1.04 ms. Pressure dot synchronization alone is 8.75 ms,
so it must be measured separately from the constitutive work.

The revised order is:

1. GPU baseline: freeze the current profile and add granular counters/readback.
2. Vulkan particle state and Drucker–Prager constitutive kernel.
3. Vulkan integration with existing P2G/grid/G2P, then validate chute, pour and
   impact scenes on the GPU.
4. Vulkan detach, settling/repose, then persistent fragment identity and
   multi-field frictional contact. Two independently seeded cohesive blocks
   must collide, exchange momentum and remain non-interpenetrating; a
   single-field velocity average is not an accepted approximation.
   Before tuning presets, separate physical packing/rest volume from numerical
   particles-per-voxel sampling. The same dry-sand material at 4 and 8 PPC must
   converge to a similar repose angle and bulk compression; raising PPC may
   improve quadrature, but must not silently turn loose sand into a different,
   tightly compacted material.
5. Small CPU constitutive golden tests and CPU/GPU tolerance checks.
6. Fluid regression fixes, only where measurements show a shared-path defect.
7. API, IPC, presets and production acceptance tests.

The first GPU milestone is successful chute + pour + impact behavior with stable
mass/momentum and no NaN/Inf, while the extra constitutive kernel remains within
an explicit frame budget. CPU 100k scenes are not required for this milestone.

After the core physics/contact gate, surface reconstruction becomes
fragment-aware: intact bonds may form a continuous stretched dough/plastic SDF,
while broken fragment ids are forbidden from contributing to the same
Zhu-Bridson neighbourhood. This render phase must reuse the physics fragment id
and expose one canonical profile/parameter contract through UI, scripting and
IPC; render-only fragment guesses are not authoritative.
