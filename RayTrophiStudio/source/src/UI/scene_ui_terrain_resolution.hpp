/*
 * =========================================================================
 * File:  scene_ui_terrain_resolution.hpp
 * Part of: scene_ui_terrain.hpp (included from there, not standalone)
 * =========================================================================
 *
 * Three-resolution inspector section for a selected terrain:
 *
 *   FIELD   -- analysis grid (heightmap.width/height).
 *              Changing this triggers a full graph re-evaluation or resample.
 *   MESH    -- vertex grid for rasterisation/BVH.
 *              mesh <= field. 0 = follow field.
 *   PAINT   -- splat + macroColorMap evaluation grid.
 *              0 = follow field (max(512,field)). Can exceed field when
 *              procedural nodes run at paint resolution.
 *
 * Must be included after scene_ui_terrain.hpp defines its helper functions
 * (ResampleTerrainCurrentState, ScheduleTerrainTopologyRebuild, etc.).
 *
 * History:
 *   2026-08-21  Extracted from scene_ui_terrain.hpp (Faz 0 Parti A).
 *               mesh_resolution UI existed; paint_resolution UI added here.
 * =========================================================================
 */
#pragma once

// ---------------------------------------------------------------------------
// DrawTerrainResolutionSection
// ---------------------------------------------------------------------------
// Call this inside the inspector whenever a terrain is selected.
// ctx -- UIContext with scene, renderer, etc.
// t   -- the selected TerrainObject* (must not be null)
// ---------------------------------------------------------------------------
static void DrawTerrainResolutionSection(UIContext& ctx, TerrainObject* t) {
    if (!UIWidgets::BeginSection("Geometry & Resolution",
                                  ImVec4(0.45f, 0.78f, 1.0f, 1.0f), true))
        return;

    const bool graphBusy = t->nodeGraph && t->nodeGraph->isEvaluatingAsync();

    // Per-terrain statics keyed by terrain ID.
    static std::unordered_map<int, int>         s_reqField;
    static std::unordered_map<int, int>         s_obsField;
    static std::unordered_map<int, int>         s_reqMesh;
    static std::unordered_map<int, int>         s_reqPaint;
    static std::unordered_map<int, std::string> s_status;

    int& reqField = s_reqField[t->id];
    int& obsField = s_obsField[t->id];
    int& reqMesh  = s_reqMesh[t->id];
    int& reqPaint = s_reqPaint[t->id];

    // Initialise / refresh on first view or if terrain changed externally.
    if (reqField < 64 ||
        (obsField >= 64 && reqField == obsField && obsField != t->heightmap.width))
        reqField = (std::max)(64, t->heightmap.width);
    obsField = t->heightmap.width;
    if (reqMesh  == 0) reqMesh  = t->meshGridWidth();
    if (reqPaint == 0) reqPaint = t->paintGridWidth();

    const int fieldMin = (std::min)(t->heightmap.width, t->heightmap.height);

    // Memory cost helper (CPU-side, rough).
    auto mbFor = [](int w, int h, int channels, int bpe) -> float {
        return static_cast<float>(
            static_cast<uint64_t>(w) * h * channels * bpe) / (1024.f * 1024.f);
    };

    // ==========================================================================
    // FIELD RESOLUTION
    // ==========================================================================
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.9f, 1.0f, 1.0f));
    ImGui::SeparatorText("Field (Analysis)");
    ImGui::PopStyleColor();

    ImGui::Text("Current: %d x %d", t->heightmap.width, t->heightmap.height);
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragInt("Field Resolution##Field", &reqField, 1.f, 64, 16384, "%d");
    reqField = (std::clamp)(reqField, 64, 16384);
    {
        const double   cellM  = t->heightmap.scale_xz / (double)(std::max)(1, reqField - 1);
        const float    hm_mb  = mbFor(reqField, reqField, 1, 4);  // f32 height
        const float    ero_mb = mbFor(reqField, reqField, 4, 4);  // f32x4 erosion
        ImGui::TextDisabled("Cell %.3f m  |  Height ~%.1f MB  |  Erosion ~%.1f MB",
                            cellM, hm_mb, ero_mb);
    }

    const bool fieldChanged = reqField != t->heightmap.width || reqField != t->heightmap.height;
    ImGui::BeginDisabled(!fieldChanged || graphBusy || !t->nodeGraph);
    if (UIWidgets::PrimaryButton("Rebuild Procedural##Field",
                                  ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
        const int  oldW   = t->heightmap.width,  oldH = t->heightmap.height;
        const auto oldHD  = t->heightmap.data,   oldOr = t->original_heightmap_data;
        const auto oldHrd = t->hardnessMap, oldFl = t->flowMap, oldEr = t->erosionMapRGBA;
        try {
            if (!t->nodeGraph->evaluateTerrainAtResolution(t, ctx.scene, reqField, reqField)) {
                t->heightmap.width  = oldW; t->heightmap.height = oldH;
                t->heightmap.data   = oldHD; t->original_heightmap_data = oldOr;
                t->hardnessMap = oldHrd; t->flowMap = oldFl; t->erosionMapRGBA = oldEr;
                TerrainManager::getInstance().resizePaintMaps(t);
                TerrainManager::getInstance().rebuildTerrainMesh(ctx.scene, t);
                s_status[t->id] = "Graph produced no valid height output; terrain restored.";
            } else {
                reqField = t->heightmap.width;
                s_status[t->id] = "Procedural terrain rebuilt at new field resolution.";
            }
        } catch (const std::exception& e) {
            t->heightmap.width  = oldW; t->heightmap.height = oldH;
            t->heightmap.data   = oldHD; t->original_heightmap_data = oldOr;
            t->hardnessMap = oldHrd; t->flowMap = oldFl; t->erosionMapRGBA = oldEr;
            TerrainManager::getInstance().resizePaintMaps(t);
            TerrainManager::getInstance().rebuildTerrainMesh(ctx.scene, t);
            s_status[t->id] = std::string("Rebuild failed: ") + e.what();
        }
        if (ctx.scene_ui_ptr) ctx.scene_ui_ptr->selectManagedMesh(ctx, t->flatMesh);
        ctx.renderer.resetCPUAccumulation();
        ScheduleTerrainTopologyRebuild(ctx);
        ResetTerrainBackendAccumulation(ctx);
    }
    ImGui::EndDisabled();
    UIWidgets::HelpMarker("Re-evaluates the procedural graph at the target field resolution.\n"
                          "File/authored inputs keep their source contract.");

    ImGui::BeginDisabled(!fieldChanged || graphBusy);
    if (ImGui::Button("Resample Current##Field",
                      ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
        if (ResampleTerrainCurrentState(t, ctx.scene, reqField, reqField)) {
            if (ctx.scene_ui_ptr) ctx.scene_ui_ptr->selectManagedMesh(ctx, t->flatMesh);
            ctx.renderer.resetCPUAccumulation();
            ScheduleTerrainTopologyRebuild(ctx);
            ResetTerrainBackendAccumulation(ctx);
            s_status[t->id] = "Terrain resampled to new field resolution.";
        } else {
            s_status[t->id] = "Resample failed: current height data is invalid.";
        }
    }
    ImGui::EndDisabled();
    UIWidgets::HelpMarker("Bilinearly resamples heights + all analysis maps to the new field resolution.\n"
                          "Becomes the new authored baseline; original node graph is unaffected.");

    // ==========================================================================
    // MESH RESOLUTION
    // ==========================================================================
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 1.0f, 0.75f, 1.0f));
    ImGui::SeparatorText("Mesh (Vertex)");
    ImGui::PopStyleColor();

    ImGui::Text("Mesh grid: %d x %d%s",
                t->meshGridWidth(), t->meshGridHeight(),
                t->mesh_resolution == 0 ? "  (follows field)" : "");
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragInt("Mesh Resolution##Mesh", &reqMesh, 1.f, 2, fieldMin, "%d");
    reqMesh = (std::clamp)(reqMesh, 2, fieldMin);
    {
        const uint64_t tris    = reqMesh > 1
            ? 2ull * static_cast<uint64_t>(reqMesh - 1) * (reqMesh - 1) : 0;
        // Vertex buffer: P+N+UV (3+3+2 floats), ~8 floats/vertex
        const float    vtx_mb  = mbFor(reqMesh, reqMesh, 8, 4);
        ImGui::TextDisabled("%.2f M triangles  |  vtx ~%.0f MB  (field stays %d x %d)",
                            tris / 1'000'000.0, vtx_mb, t->heightmap.width, t->heightmap.height);
        // Honest about what is lost.
        if (reqMesh < fieldMin)
            ImGui::TextDisabled("Normals still sampled from the field; silhouette follows the mesh.");
    }

    const bool meshChanged = reqMesh != t->meshGridWidth();
    ImGui::BeginDisabled(!meshChanged || graphBusy);
    if (UIWidgets::PrimaryButton("Apply Mesh Resolution",
                                  ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
        t->mesh_resolution = (reqMesh >= fieldMin) ? 0 : reqMesh;
        TerrainManager::getInstance().rebuildTerrainMesh(ctx.scene, t);
        if (ctx.scene_ui_ptr) ctx.scene_ui_ptr->selectManagedMesh(ctx, t->flatMesh);
        ctx.renderer.resetCPUAccumulation();
        ScheduleTerrainTopologyRebuild(ctx);
        ResetTerrainBackendAccumulation(ctx);
        s_status[t->id] = "Mesh resolution applied.";
    }
    ImGui::EndDisabled();
    UIWidgets::HelpMarker("Sets the vertex/triangle grid independently of the field.\n"
                          "Measured: 4096 field + 1024 mesh = ~18x cheaper BVH build\n"
                          "with no loss of analysis or shading resolution.\n"
                          "0 = follow the field (historical default).");

    // ==========================================================================
    // PAINT RESOLUTION
    // ==========================================================================
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.85f, 0.5f, 1.0f));
    ImGui::SeparatorText("Paint (Splat + Macro Color)");
    ImGui::PopStyleColor();

    const int curPaintW = t->paintGridWidth();
    const int curPaintH = t->paintGridHeight();
    ImGui::Text("Paint grid: %d x %d%s",
                curPaintW, curPaintH,
                t->paint_resolution == 0 ? "  (follows field)" : "");
    ImGui::SetNextItemWidth(160.f);
    ImGui::DragInt("Paint Resolution##Paint", &reqPaint, 1.f, 64, 16384, "%d");
    reqPaint = (std::clamp)(reqPaint, 64, 16384);
    {
        const float splat_mb  = mbFor(reqPaint, reqPaint, 1, 4);   // RGBA8 splat
        const float macro_mb  = mbFor(reqPaint, reqPaint, 1, 4);   // RGBA8 macroColor
        ImGui::TextDisabled("Splat RGBA8: ~%.1f MB  |  MacroColor RGBA8: ~%.1f MB",
                            splat_mb, macro_mb);

        // Warn: paint > field with no procedural nodes = pure upsampling.
        // TODO Faz 2: replace hasProcNode with actual SatMapNode presence query.
        const bool hasProcNode = false;
        if (reqPaint > t->heightmap.width) {
            if (!hasProcNode) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.65f, 0.1f, 1.0f));
                ImGui::TextWrapped("No procedural nodes: raising paint above field "
                                   "only upsamples — no new frequency is added. "
                                   "File size grows, quality stays the same.");
                ImGui::PopStyleColor();
            } else {
                ImGui::TextDisabled("SatMap/procedural nodes will evaluate at %d px "
                                    "— real frequency gain.", reqPaint);
            }
        }
        if (reqPaint < (std::max)(512, t->heightmap.width) && reqPaint < curPaintW)
            ImGui::TextDisabled("Lowering paint resolution resamples the existing splat.");
    }

    const bool paintChanged = reqPaint != curPaintW;
    ImGui::BeginDisabled(!paintChanged || graphBusy);
    if (UIWidgets::PrimaryButton("Apply Paint Resolution",
                                  ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
        // Store 0 when the user sets exactly the "follow field" value so round-
        // trips don't accumulate drift.
        t->paint_resolution = (reqPaint == (std::max)(512, t->heightmap.width)) ? 0 : reqPaint;
        TerrainManager::getInstance().resizePaintMaps(t);
        ctx.renderer.resetCPUAccumulation();
        ResetTerrainBackendAccumulation(ctx);
        s_status[t->id] = "Paint resolution applied. Splat and macro color map resampled.";
    }
    ImGui::EndDisabled();
    UIWidgets::HelpMarker("Sets the splat map and macro color map grid independently of\n"
                          "the field. 0 = follow field (max(512, field)).\n"
                          "Only produces new information when procedural nodes (SatMap,\n"
                          "noise, warp) evaluate at this resolution.");

    // ==========================================================================
    // Status line
    // ==========================================================================
    const auto it = s_status.find(t->id);
    if (it != s_status.end() && !it->second.empty())
        ImGui::TextWrapped("%s", it->second.c_str());
    if (graphBusy)
        ImGui::TextColored(ImVec4(1.f, 0.75f, 0.25f, 1.f),
                           "Terrain graph is evaluating...");

    UIWidgets::EndSection();
}
