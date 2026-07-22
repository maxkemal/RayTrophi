/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiInternal.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * Internal shared declarations for modular RtApi translation units.
 */

#pragma once

#include "Api/RtApi.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "SceneCommand.h"
#include <atomic>
#include <memory>
#include <string>

// Global C symbols from Main.cpp / Scene
extern SceneUI ui;
extern bool g_solid_viewport_active;
extern bool g_geometry_dirty;
extern std::atomic<uint64_t> g_scene_geometry_generation;
extern bool g_bvh_rebuild_pending;
extern bool g_optix_rebuild_pending;
extern bool g_vulkan_rebuild_pending;
extern bool g_viewport_raster_rebuild_pending;

namespace rtapi {

extern UIContext* g_ctx;
extern SceneHistory* g_history;
extern RenderJobInfo g_render_job;

inline Result notBound() { return Result::fail("rtapi is not bound to a UIContext"); }
inline bool renderJobActive() { return g_render_job.state == RenderJobState::Rendering; }
bool objectExists(const std::string& name);
void pollTerrainEvaluations();

} // namespace rtapi
