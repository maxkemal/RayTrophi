# Vulkan raster requirements audit — 2026-11-20

Scope: current working-tree source and deployed `dist/RayTrophiPlayer/shaders`.
This is a static audit, not certification of any GPU or the remote EXE. No C++
build or application launch was performed. Driver features/limits on the AMD
machine and the provenance of its EXE remain unknown.

## Actual rendering route

`RuntimeSceneViewer.cpp:163` constructs `VulkanViewportBackend`, explicitly selects
MaterialPreview at line 167. This is the viewport subclass, not just the similar
base-adapter raster implementation in VulkanBackend.cpp.

`VulkanViewportBackend.cpp:3240` logs the requested mode before resource creation.
Therefore `Interactive viewport mode selected: 'MaterialPreview'` proves neither
successful pipeline creation nor a rendered frame. HDR MaterialPreview becomes
active only if its pipeline, descriptor set, material buffer, post pipeline and
HDR framebuffer all exist (`:3994`). Otherwise the draw path binds the Solid
pipeline (`:4109`, `:4346`). The warning says Matcap, but `useMatcap` is set from
the actual viewport mode (`:4356`); an unchanged MaterialPreview mode falls back
to Solid shading, not necessarily a Matcap preset.

Vulkan renders offscreen; CPU readback then `SDL_UpdateTexture` presents the
result (`RuntimeSceneViewer.cpp:636–645`). SDL uses an accelerated renderer
selected by SDL (`AppWindow.cpp:52`); there is no explicit OpenGL context/version
requirement in this Player route. HUD visibility does not certify Vulkan output.

## Version contract as shipped

| Layer | Current requirement / behavior | Evidence |
|---|---|---|
| Instance request | Fixed Vulkan 1.3; no loader-version negotiation | VulkanBackend.cpp:1264 |
| Physical GPU admission | Scores device type, RT extensions and VRAM; no API >=1.3 eligibility rejection | VulkanBackend.cpp:1340–1401 |
| Queue | Prefers one queue family supporting both graphics and compute; viewport rejects compute-only | VulkanBackend.cpp:1409–1434; VulkanViewportBackend.cpp:1033 |
| Bulk shader target | Vulkan 1.3 / optimization enabled for normal raster shaders | compile_shaders.ps1:155 |
| Deployed base shaders | solid, solid_frag, material_preview, material_preview_frag, material_preview_sky, raster_post are SPIR-V 1.6 | Binary headers inspected; source/dist hashes equal |
| Current usable device floor | Vulkan 1.3 for these shipped SPIR-V 1.6 modules, plus mode-specific features and limits | Khronos version mapping; shader inspection |

Instance creation with API 1.3 is not proof that the selected physical GPU
supports device API 1.3. Vulkan permits mixed instance/device API versions.
The application must match each shader module to the device's API support.
`spirv-val --target-env vulkan1.2 dist/.../solid.spv` rejects the current binary:
`Invalid SPIR-V binary version 1.6 ... SPIR-V 1.5 (under Vulkan 1.2 semantics)`.
This negative check is expected and is not a malformed-shader finding.

The packaging readme says a current Vulkan 1.2 driver is sufficient
(`scripts/package_player.ps1:259`). That promise does not match the deployed
SPIR-V 1.6 raster shaders. The separate simulation batch script targeting 1.2
does not make the renderer a Vulkan 1.2 renderer.

## Mode-specific contract and gaps

### Solid / Matcap

- Base SPIR-V modules declare only `Shader`, without descriptor indexing,
  Float64, Int64 or ray-tracing capabilities.
- The pipeline layout requests **176 bytes** of push constants
  (`VulkanViewportBackend.cpp:1245`). The actual current C++ draw block and GLSL
  block occupy **168 bytes** (`:3565`, solid.vert, SPIR-V offsets through 164).
  Both exceed 128. There is **no Solid maxPushConstantsSize gate** before layout
  creation. MaterialPreview fallback therefore is not universally safe.
- Base layout has one combined-image sampler. Base formats include
  R8G8B8A8_UNORM and D32_SFLOAT. Resource setup also creates HDR passes/images
  even for the shared base resources (`:1203–1234`, `:3084–3105`). This is not
  a stripped-down legacy renderer with only the LDR resources.

### MaterialPreview

- Needs **208-byte push-constant layout** (`:1882`), explicitly gated against
  maxPushConstantsSize (`:1635–1644`).
- Needs all enabled feature bits `runtimeDescriptorArray`,
  `shaderSampledImageArrayNonUniformIndexing`, `descriptorBindingPartiallyBound`
  (`VulkanBackend.cpp:1590–1603`). These are individual capabilities, not a
  conclusion to draw solely from a Vulkan version or extension name.
- SPIR-V declares ShaderNonUniform, RuntimeDescriptorArray and
  SampledImageArrayNonUniformIndexing, consistent with the texture array.
- Descriptor layout (`:1718–1854`) contains **13 fragment-stage storage
  buffers**, **one storage image**, and **N + 14 combined-image samplers**.
  N is the bindless material texture-array capacity, not the number of textures
  in a particular test scene. Empty scene textures do not shrink the layout.
- Current N calculation reserves **12**, not 14, against
  maxPerStageDescriptorSampledImages (`:1707–1711`). When that clamp saturates,
  the requested layout can exceed that limit by two. It also does not budget
  against per-stage sampler limits, descriptor-set limits, storage-buffer limits
  and maxPerStageResources here. Consequently descriptor indexing + push size
  alone is not a complete admission check.
- Four HDR color attachments: R16G16B16A16_SFLOAT, R16G16_SFLOAT,
  R16G16B16A16_SFLOAT, R32G32B32A32_SFLOAT, plus D32_SFLOAT depth.
  The primary color attachment blends while the others do not (`:1962–1996`),
  so independentBlend must be enabled. Fragment shader storage writes also
  require fragmentStoresAndAtomics. Device setup enables these only if supported
  (`VulkanBackend.cpp:1681–1684`); the MP eligibility gate does not check them.
- The shared viewport setup does not preflight all attachment usage/format
  limits. Allocation/pipeline failures can return false without a useful stage
  and numeric VkResult in SceneLog.

### Atmosphere / simulation / optional features

- `ensureInteractiveViewportResourcesImpl` unconditionally calls
  `ensureAtmosphereLUTPipeline` (`:1042`). That helper first calls aerial/cloud
  setup (`VulkanBackend.cpp:16674–16675`). A solid-color world does not suppress
  these startup attempts. Errors about these pipelines alone do not prove that
  the main raster pipeline failed.
- atmosphere_lut, cloud_noise, cloud_sample and raster_post declare
  StorageImageExtendedFormats; current device setup enables the associated
  feature opportunistically via VulkanFsrLoader.cpp:20, but these pipelines
  cannot assume unsupported features became available.
- Shader float atomics / Float64 belong to particular simulation kernels,
  not the simple Solid raster shader. Hardware ray tracing, BDA, ray query and
  CUDA are not required by the inspected base Solid/MP SPIR-V modules.
  Optional volumes/reflections/shadows add their own requirements.
- RT_VK_LEGACY reduces enabled features; it does not change SPIR-V 1.6 binaries,
  push-constant layouts or descriptor budgets, and is not a complete compatibility
  mode for this renderer.

## Why the remote log cannot settle this yet

`VK_INFO()` is defined as `VulkanNullLogHelper()` (VulkanBackend.h:1905).
Device name, device API version and feature-tier informational messages are
therefore discarded. The successful CPU-side 27-mesh upload is not proof of a
successful GPU draw or readback. The supplied AMD log lacks numeric pipeline
creation errors and measured device limits; exact hardware rejection cannot
be inferred from the GPU's marketing name.

## Required next engineering steps (not implemented by this audit)

1. Persist one device admission report: selected adapter/driver/device API,
   actual enabled features, push/sampler/SSBO/format limits, loaded shader versions.
2. Reject incompatible shader versions before module creation; report numeric
   VkResult and pipeline name for resource/pipeline failures.
3. Define a real basic-raster compatibility profile before changing targets:
   <=128-byte push constants, fixed small descriptor layout and genuinely lazy
   effect/HDR resource setup. Recompile all affected shader variants consistently.
4. Budget MP descriptors against the full layout and gate all used feature bits.
5. On AMD compare requested/effective mode, submitted/consumed frames, nonblack
   CPU readback and SDL upload result. Do not certify compatibility from HUD alone.

## Primary references

- https://docs.vulkan.org/guide/latest/versions.html
- https://docs.vulkan.org/spec/latest/chapters/initialization.html
- https://docs.vulkan.org/spec/latest/chapters/limits.html
- https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceVulkan12Features.html

Validation performed: binary SPIR-V version/capability inspection; source/dist
shader hashes; targeted Vulkan1.3 spirv-val checks; expected Vulkan1.2 rejection
of shipped Solid; source and layout arithmetic audit. No binary or shader changed.
