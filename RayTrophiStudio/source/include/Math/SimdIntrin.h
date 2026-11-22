#pragma once

#if defined(__aarch64__)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#include <simde/x86/sse4.1.h>
#else
#include <immintrin.h>
#endif
