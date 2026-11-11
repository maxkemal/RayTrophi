#pragma once
// Absolute temperatures are stored and edited in Kelvin; the panel also shows
// the Celsius value in parentheses (user request 2026-11-11: habits stay).
// Only for ABSOLUTE temperatures -- a temperature DIFFERENCE (offset, range)
// is the same number in both scales, and a render colour temperature is not a
// thermodynamic one.
#include <cstddef>
#include <cstdio>

namespace UiTemperature {

constexpr float kCelsiusOffset = 273.15f;

// ImGui format string for a Kelvin drag: "<number_format> K (<celsius> C)".
// The Celsius part is a literal computed from the current value.
inline const char* kelvinFormat(float kelvin, char* buffer, std::size_t size,
                                const char* number_format = "%.1f") {
    std::snprintf(buffer, size, "%s K (%.1f C)", number_format, kelvin - kCelsiusOffset);
    return buffer;
}

// Plain text: "353.0 K (79.9 C)".
inline const char* kelvinText(float kelvin, char* buffer, std::size_t size,
                              const char* number_format = "%.1f") {
    char fmt[48];
    std::snprintf(fmt, sizeof(fmt), "%s K (%%.1f C)", number_format);
    std::snprintf(buffer, size, fmt, kelvin, kelvin - kCelsiusOffset);
    return buffer;
}

} // namespace UiTemperature
