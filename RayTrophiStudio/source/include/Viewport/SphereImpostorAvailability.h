#pragma once
#include <atomic>

// Published only by the viewport owning the opaque sphere pipeline.
inline std::atomic<bool> g_sphere_impostor_ready{false};
