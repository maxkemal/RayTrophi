#pragma once

#include <cstdint>

enum class TextureCompressionTarget : uint8_t {
    None = 0,
    BC4,
    BC5,
    BC7
};
