// Single implementation unit for the header-only image codecs (stb_image,
// stb_image_write, tinyexr), shared by Studio and Player.
//
// Moved verbatim out of Main.cpp so both executables link one definition of
// stbi_* / LoadEXR. Every other translation unit includes the headers WITHOUT
// the *_IMPLEMENTATION macros. Do not define them anywhere else.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md (Faz 3b).

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1  // Use stb's zlib (both stb implementations are above)
// Fix Windows.h min/max macro conflicts
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
