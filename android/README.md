# RayTrophi Player - Android (touch build)

Plan and status: `docs/dev/PLAYER_ANDROID_PORT.md`. This folder is the Gradle + CMake project.
The native source list is NOT hand-written: it is generated from the same inputs as the Windows
Player (`RayTrophiStudio.vcxproj` + `scripts/player_include_manifest.json`) plus the Android overlay
`scripts/player_android_manifest.json` (exclusions, stubs, flags, dependency variables).

## Prerequisites (one-time)

- JDK 17, Gradle 8.7+ (or add a wrapper: `gradle wrapper --gradle-version 8.7` inside `android/`).
- Android SDK with platform 34 and build-tools; accept the licenses.
- **NDK r26 or newer** (`sdkmanager "ndk;26.3.11579264"`) and CMake 3.22.1 (`sdkmanager "cmake;3.22.1"`).
- SDL2 **source** tree (2.30.x) and SDL2_image source tree. They provide the `org.libsdl.app.*` Java
  classes (`<SDL2>/android-project/app/src/main/java`) and are built by this CMake via `add_subdirectory`.
- A device or emulator with Vulkan 1.2 (x86_64 image with gfxstream, or Mali-G57 class or better).

## Dependency variables

| CMake variable (Gradle property) | What |
|---|---|
| `RT_SDL2_DIR` (`rtSdl2Dir`) | SDL2 source tree |
| `RT_SDL2_IMAGE_DIR` (`rtSdl2ImageDir`) | SDL2_image source tree |

Jolt, ozz-animation, imgui, miniz and xatlas are compiled from `RayTrophiStudio/external`.

## Build

```
python scripts/gen_player_android_cmake.py          # regenerate android/app/src/main/cpp/player_sources.cmake
python scripts/gen_player_android_cmake.py --check  # staleness check
cd android
gradle assembleDebug -PrtSdl2Dir=<path> -PrtSdl2ImageDir=<path>
```

Rerun the generator after adding or removing a source in the Studio vcxproj, or after editing
`scripts/player_android_manifest.json`; `--check` fails when the checked-in fragment is stale.

## Assets

`app/src/main/assets/` is filled by batch A4 (shaders `.spv` + scene). See the README there.
