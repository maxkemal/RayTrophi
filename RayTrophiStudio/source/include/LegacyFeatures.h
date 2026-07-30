#pragma once

// The old GasVolume/GasSimulator implementation remains buildable only to keep
// its eventual physical removal separate from production solver work. It must
// never be created, loaded, saved, simulated or rendered at runtime.
namespace RayTrophiLegacy {
inline constexpr bool kGasVolumeRuntimeEnabled = false;
}
