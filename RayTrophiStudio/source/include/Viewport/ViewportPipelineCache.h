#pragma once
// Persistent VkPipelineCache shared by the interactive raster/RayFusion
// viewport pipelines (O7d).
//
// Targets the ~9.9 s `first_frame` shader/pipeline compile time measured on
// Lenovo TB330FU (Mali-G52) after O7b moved first 3D rendering out of
// `game_design`. Reuses RayTrophiSim::SimulationPipelineCache under the
// "viewport_raster" prefix so Windows and Android share one validation,
// shader-stamp and disk-persistence path.

#include <vulkan/vulkan.h>

#include <string>

namespace Backend {

// Opens (or returns the already-open) persistent cache for `device`.
VkPipelineCache acquireViewportPipelineCache(VkDevice device,
                                             VkPhysicalDevice physical_device,
                                             const std::string& shader_dir);

// Returns the currently open cache handle for `device`, or VK_NULL_HANDLE.
VkPipelineCache currentViewportPipelineCache(VkDevice device);

// Marks the cache dirty after a pipeline is created (writes on save/close).
void markViewportPipelineCacheDirty(VkDevice device);

// Persists any newly compiled pipelines to disk (no-op if clean or disabled).
void saveViewportPipelineCache(VkDevice device);

// Saves and destroys the cache handle for `device`.
void closeViewportPipelineCache(VkDevice device);

} // namespace Backend
