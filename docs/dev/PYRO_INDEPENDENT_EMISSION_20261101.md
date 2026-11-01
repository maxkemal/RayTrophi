# Independent emission and receiver-aware fire sampling

Latest correction and post-build scene inspection:
PYRO_SOURCE_CHANNEL_HANDOFF_20261101.md. Independent emission/mask now bypass
the smoke density curve as well as its multiplier; remap/cutoff remain shared.
The pre-correction curve semantics below are historical where they conflict.

2026-11-01. Implements the follow-up to PYRO_FIRST_SCATTER_LIVE_20261101.md.
C++/CUDA/shader builds, post-build IPC acceptance and GPU images are pending.
The agent did not build, launch Studio, change the open scene or repeat timing
measurements during this change.

## Independent coefficient

New common material setting: **independent_emission**, boolean, default false.
UI: **Emission independent of smoke multiplier**.

When enabled the existing emission response evaluates density before the
effective smoke multiplier and produces an emission coefficient j. Temperature/
density powers, exposure, saturation, ramp/blackbody and density mask remain
controls. Masking precedes the existing thermal limiter (64). Increasing the
smoke multiplier does not increase j. Extinction and scattering still change,
so the final image can brighten or darken for transport reasons.

Remap/curve/cutoff and source density remain part of the sampled source
definition; this is not a separate emission grid. Graph-authored optical
dependencies are outside the multiplier-invariance acceptance. The fire
sampler still rejects scenes containing gas Volume Graph programs.

| Mode | Camera emission over a constant segment | Source coefficient |
| --- | --- | --- |
| Legacy | S * (1-exp(-sigma_t*dt)) | sigma_t * S |
| Independent | j * (1-exp(-sigma_t*dt))/sigma_t | j |

The independent formula uses a stable j*dt vacuum limit and a short series
near zero optical depth. At smoke multiplier zero, emission-only density reads
retain the source field, while optical density remains zero. Host fog bounds
and metadata rejection retain these emitters. No ambient replacement is added.

VolumeEmissionTransport.h provides shared scalar response, density policy and
integration for C++/CUDA/GLSL. Wiring covers Vulkan ordinary march and mixture
(including flight emission), RayFusion/preview, CPU VDB camera integration,
OptiX NanoVDB and gas-texture camera integration. Fire lighting uses the same
coefficient conversion; it does not multiply independent j by sigma_t again.
Existing backend field/color/extinction approximations have not become identical.

## Receiver-aware proposal

Source selection now uses receiving smoke samples, rather than one global
power-only draw before the march:

- Up to four candidates per local update; the total per-camera-segment budget
  remains fire_scatter_samples (1..32, default 8). With multiple enabled sources,
  the largest authored budget bounds the invocation.
- 75% uniform receiver-centred ball, sized to roughly four optical mean free
  paths and bounded by voxel/domain scale.
- 25% global proposal: 7/8 sparse float-leaf topology + 1/8 AABB, or just AABB
  when leaf indexing is unavailable.
- Every draw uses the full world-volume mixture PDF, including overlap,
  transform Jacobians and legitimate zero draws outside the source box.
- RIS weights include inverse distance and a positive approximate-transmission
  guide. The guide never replaces actual RGB visibility. A nonzero guide floor
  preserves support for sources reached through holes/clear paths.
- Movement/stride can refresh selection while budget remains. Changing the
  source resets held visibility. After budget exhaustion the retained sample
  can still be used; 32 candidates are not charged again at every march step.

Existing shadow quadrature, voxel near-field regularization and light/visibility
stride remain approximations. No persistent cache or full-domain RGB buffer
was introduced. Actual random walks do not add the first-scatter estimator
without MIS; their flight emission does consume the coefficient convention.

## Compatibility and authoring

Saved in VolumeEmissionResponse, strict boolean validation and legacy default
off. Presets retain it. UI, Python and IPC share the existing atomic material
service. No automatic intensity/exposure adjustment occurs when enabling it;
brightness units change, and existing intensity may need substantial retuning.

Python:

    rt.volume.set_material("gas:Nuclear Gas", independent_emission=True,
                           fire_scatter_enabled=True, fire_scatter_samples=8)

External IPC (separate process, never the embedded script editor):

    client.call("volume.set_material", target="gas:Nuclear Gas",
                settings={"independent_emission": True,
                          "fire_scatter_enabled": True, "fire_scatter_samples": 8})

volume.get_material round-trips all controls. The existing response packet
word 0 is flags: bit 0=response enabled, bit 1=independent j. Values 0/1 retain
legacy meaning; 2/3 enable j. **VkVolumeInstance stays 688 bytes / 172 words.**
All consumers must be rebuilt: an old shader interprets 2/3 incorrectly even
though the byte stride did not change.

The existing secondary radiance cache stores S plus external in-scatter.
Emitting independent-j materials bypassed it until the bounce cache batch of
2026-11-01 (see PYRO_SOURCE_CHANNEL_HANDOFF_20261101.md) and marched normally. Their aligned
mixture shared-walk optimization is also bypassed until its assumptions are
extended. This can cost work; no speed improvement is claimed. Non-emitting
materials retain these optimizations even if the new flag was saved on them.
Independent emissive Matter streams stay separate to retain source densities.

The legacy internal-light integral is disabled for independent j; its UI is
disabled when either j mode or new fire sampling is on. Serialized legacy
controls remain compatible. RT/RayFusion step defaults were not changed.

## Verification and acceptance

Completed: **97 non-build math/source-contract tests**, plus light-cache ABI
contract. Tests cover source-power invariance, vacuum/opaque limits, homogeneous
step partition invariance, guided RIS normalization, mixture support, zero-smoke
culling, common authoring, header registration and shader call arity.
These are not C++ compilation or GPU execution.

User-run C++ tests: volume_emission_transport_test.cpp,
volume_material_authoring_test.cpp and fog_optical_bounds_test.cpp.
Python syntax of the external probe was checked. After full C++/CUDA/shader build:

    python scripts/test/rt_test_volume_fire_scatter_ipc.py --target "gas:Nuclear Gas"

The probe checks controls/limits/atomic rejection and restores the material;
it was not run against the older open binary.

Visual acceptance: fixed explosion/frame/camera/post. The user's RT ~32 step
starting point can stay, with RayFusion quality separately maintained. Enable
independent emission and tune source exposure/intensity once, then freeze source
settings. Compare smoke multiplier 1/10/30/0: j must not grow; the image still
follows extinction/scattering. Zero extinction must retain visible emission.
Check colored absorption, frame/mask updates, save/load and presets.

For fire-to-smoke acceptance use Vulkan deterministic march, random walk off.
Compare sampler off/on with 8 then 32 total candidates; nearby smoke should
receive light through physical visibility, while blockers attenuate it. More
candidates must improve convergence rather than multiply energy. A raw
fire-light AOV remains useful future diagnostic work. Matching the reference
appearance is not claimed before build/image acceptance.
