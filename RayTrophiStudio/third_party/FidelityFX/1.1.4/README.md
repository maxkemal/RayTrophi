# FidelityFX SDK 1.1.4 — Vulkan upscaler dependency

Official release: <https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v1.1.4>

This directory contains AMD's unmodified FFX API headers and MIT license notices.
Only the Vulkan runtime is staged; frame generation, DX12, SDK samples, import
libraries and SDK build tools are not dependencies of RayTrophi's B4 upscaler.

The DLL is **`amd_fidelityfx_vk.dll`**, from `PrebuiltSignedDLL/` in the release
archive. Its AMD Authenticode signature was valid on 2026-10-26. It is x64 and
exports `ffxCreateContext`, `ffxDestroyContext`, `ffxConfigure`, `ffxQuery` and
`ffxDispatch`. No downloaded executable or DLL was run during staging.

Its Windows file version is **1.0.1.41314**, product version **1.0.1.0**.
These are DLL version resources, not the SDK or selected upscaler provider
version. B4 must query the provider through the FFX API at runtime.

`manifest.json` records the source URL, archive size, observed SHA-256 hashes,
selected files, signature identity and runtime dependencies. The archive hash
was computed locally after downloading from the official release; it is not
represented as a separately published AMD checksum.

## Restore and deploy

From the repository root:

```powershell
# Reuse the already downloaded, pinned archive:
powershell -NoProfile -File scripts/setup_fidelityfx.ps1

# On a new checkout, download the pinned official package first:
powershell -NoProfile -File scripts/setup_fidelityfx.ps1 -Download
```

The archive cache is `x64/.sdk-cache/FidelityFX-SDK-v1.1.4.zip`. Both it and
DLL binaries remain ignored by the repository's existing rules. Headers,
licenses, setup script, manifest and deployment target are source controlled.
The script verifies the whole archive before extracting its explicit file list,
checks every extracted file, verifies the AMD signature, then copies:

- `runtime/amd_fidelityfx_vk.dll` → **`x64/Release/amd_fidelityfx_vk.dll`**;
- `LICENSE.txt` → **`x64/Release/FidelityFX-LICENSE.txt`**;
- the same files to `x64/Debug` when that output directory already exists.

`FidelityFX.targets`, imported by the application project, also copies the DLL
and notice to `$(OutDir)` after the user's x64 builds. It adds the API header
include directory. It does not link an import library or compile AMD sources.
Missing DLLs produce a deployment warning; B4's runtime loader must report
unavailability and retain the full-resolution path.

The PE import table requires `vulkan-1.dll`, the MSVC x64 C++ runtime
(`MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`) and Windows/UCRT
system libraries. These are standard application/driver prerequisites, not
additional DLLs fetched by this setup script.

Staging is complete; renderer integration and live acceptance are separate B4
steps documented in `docs/dev/RT_REALTIME_DENOISE_UPSCALE_PLAN.md`.
