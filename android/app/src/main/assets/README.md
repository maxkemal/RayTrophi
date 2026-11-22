# APK assets (batch A4)

Intentionally empty in the skeleton. Batch A4 (asset extraction + scene open) fills it at
packaging time:

- `shaders/*.spv`: the compiled SPIR-V shaders (Vulkan 1.2) the Player reads next to the exe on
  Windows (`x64/Release/shaders`). Copied verbatim.
- the scene: `.rtp`, `.rtp.bin`, sidecar files and audio.

On first launch the Player extracts these into the app internal storage
(`Platform::appDataRoot()`); after that the code reads them with `std::filesystem` exactly as on
Windows. Do not commit binaries here.
