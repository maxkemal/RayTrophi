#include "scene_ui.h"
#include "ui_modern.h"
#include "imgui_internal.h"
#include "HittableInstance.h"

#include "MeshModifiers.h"
#include "ProjectManager.h"
#include "globals.h"
#include "Triangle.h"
#include "Renderer.h"
#include "Backend/IViewportBackend.h"
#include "scene_data.h"
#include <SceneSelection.h>
#include "Paint/TerrainPaintAdapter.h"
#include "Paint/MeshPaintAdapter.h"
#include "TerrainManager.h"
#include "PrincipledBSDF.h"
#include "Volumetric.h"
#include "MaterialManager.h"
#include "Texture.h"
#include "stb_image_write.h"
#include "SceneCommand.h"
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <SDL.h>

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

namespace {

TerrainObject* resolvePaintTerrain(SceneUI& ui) {
    if (ui.terrain_brush.active_terrain_id == -1) {
        return nullptr;
    }
    return TerrainManager::getInstance().getTerrain(ui.terrain_brush.active_terrain_id);
}

std::shared_ptr<Triangle> resolvePaintMesh(UIContext& ctx) {
    if (ctx.selection.selected.type != SelectableType::Object || !ctx.selection.selected.object) {
        return nullptr;
    }

    auto tri = std::dynamic_pointer_cast<Triangle>(ctx.selection.selected.object);
    if (!tri || tri->terrain_id != -1) {
        return nullptr;
    }

    return tri;
}

std::shared_ptr<Triangle> findMeshTriangleForMaterial(
    const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& mesh_entries,
    uint16_t material_id) {
    for (const auto& pair : mesh_entries) {
        if (pair.second && pair.second->getMaterialID() == material_id) {
            return pair.second;
        }
    }
    return nullptr;
}

int countTrianglesUsingMaterial(
    const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& mesh_entries,
    uint16_t material_id) {
    int count = 0;
    for (const auto& pair : mesh_entries) {
        if (pair.second && pair.second->getMaterialID() == material_id) {
            ++count;
        }
    }
    return count;
}

static std::string getHittableNodeName(const std::shared_ptr<Hittable>& obj) {
    if (!obj) return "";
    if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) return tri->getNodeName();
    if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) return inst->node_name;
    return "";
}

std::vector<std::shared_ptr<Triangle>> collectTrianglesForNode(
    const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::string& nodeName) {
    std::vector<std::shared_ptr<Triangle>> triangles;
    for (const auto& obj : objects) {
        if (getHittableNodeName(obj) != nodeName) {
            continue;
        }

        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            triangles.push_back(tri);
        } else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (inst->source_triangles) {
                triangles.insert(triangles.end(), inst->source_triangles->begin(), inst->source_triangles->end());
            }
        }
    }
    return triangles;
}

} // End of anonymous namespace

void projectObjectUVsFromView(UIContext& ctx, SceneUI& ui, const std::string& nodeName) {
    auto it = ctx.scene.base_mesh_cache.find(nodeName);
    if (it == ctx.scene.base_mesh_cache.end()) {
        std::vector<std::shared_ptr<Triangle>> baseTriangles;
        for (const auto& obj : ctx.scene.world.objects) {
            if (getHittableNodeName(obj) == nodeName) {
                if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                    baseTriangles.push_back(tri);
                } else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                    if (inst->source_triangles) {
                        for (auto& t : *inst->source_triangles) baseTriangles.push_back(t);
                    }
                    break;
                }
            }
        }
        if (baseTriangles.empty()) return;
        ctx.scene.base_mesh_cache[nodeName] = baseTriangles;
        it = ctx.scene.base_mesh_cache.find(nodeName);
    }

    std::shared_ptr<Camera> camera = ctx.scene.getActiveCamera();
    if (!camera) return;

    Vec3 cam_forward = (camera->lookat - camera->lookfrom).normalize();
    Vec3 cam_right = Vec3::cross(cam_forward, camera->vup).normalize();
    Vec3 cam_up = Vec3::cross(cam_right, cam_forward).normalize();

    // 1. Project all vertices to camera plane (Perspective projection)
    struct ProjData { float px, py; bool in_front; };
    std::vector<std::vector<ProjData>> tri_projs;
    tri_projs.reserve(it->second.size());

    float min_x = FLT_MAX, max_x = -FLT_MAX;
    float min_y = FLT_MAX, max_y = -FLT_MAX;
    bool any_visible = false;

    for (auto& tri : it->second) {
        Matrix4x4 world_mat = tri->getTransformMatrix();
        std::vector<ProjData> pts;
        for (int i = 0; i < 3; ++i) {
            Vec3 world_v = world_mat * tri->getVertexPosition(i);
            Vec3 to_v = world_v - camera->lookfrom;
            float dist = Vec3::dot(to_v, cam_forward);
            
            bool in_front = (dist > 0.05f); // Safe near plane
            float p_dist = std::max(0.05f, dist);
            
            float px = Vec3::dot(to_v, cam_right) / p_dist;
            float py = Vec3::dot(to_v, cam_up) / p_dist;
            pts.push_back({px, py, in_front});
            
            if (in_front) {
                min_x = std::min(min_x, px);
                max_x = std::max(max_x, px);
                min_y = std::min(min_y, py);
                max_y = std::max(max_y, py);
                any_visible = true;
            }
        }
        tri_projs.push_back(pts);
    }

    if (!any_visible) { // Fallback if nothing in front
        min_x = -1.0f; max_x = 1.0f; min_y = -1.0f; max_y = 1.0f;
    }

    float span_x = std::max(1e-5f, max_x - min_x);
    float span_y = std::max(1e-5f, max_y - min_y);

    std::vector<TriangleUVSetState> before_states;
    std::vector<TriangleUVSetState> after_states;
    before_states.reserve(it->second.size());
    after_states.reserve(it->second.size());

    // 2. Map projection to UV [0, 1] range
    for (size_t t_idx = 0; t_idx < it->second.size(); ++t_idx) {
        auto& tri = it->second[t_idx];
        auto& projs = tri_projs[t_idx];
        std::array<Vec2, 3> projected_uvs;
        for (int i = 0; i < 3; ++i) {
            projected_uvs[i].u = (projs[i].px - min_x) / span_x;
            projected_uvs[i].v = (projs[i].py - min_y) / span_y;
        }
        int target_uv_set = 0;
        Material* material = MaterialManager::getInstance().getMaterial(tri->getMaterialID());
        if (auto* pbsdf = dynamic_cast<PrincipledBSDF*>(material)) {
            target_uv_set = std::max(0, pbsdf->selected_uv_set);
        }
        const size_t uv_set_index = static_cast<size_t>(target_uv_set);
        auto [old_uv0, old_uv1, old_uv2] = tri->getUVSetCoordinates(uv_set_index);
        before_states.push_back(TriangleUVSetState{ tri, uv_set_index, { old_uv0, old_uv1, old_uv2 } });
        after_states.push_back(TriangleUVSetState{ tri, uv_set_index, projected_uvs });
    }

    auto command = std::make_unique<UVProjectionCommand>(nodeName, std::move(before_states), std::move(after_states));
    command->execute(ctx);
    ui.history.record(std::move(command));

    ui.rebuildMeshCache(ctx.scene.world.objects);

    if (ui.paint_mode_state.enabled) {
        auto mesh_adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(ui.paint_mode_state.getAdapter());
        auto selected_tri = std::dynamic_pointer_cast<Triangle>(ctx.selection.selected.object);
        if (mesh_adapter && selected_tri && mesh_adapter->getNodeName() == nodeName) {
            ui.paint_mode_state.setAdapter(std::make_shared<Paint::MeshPaintAdapter>(&ctx.scene, selected_tri));
        }
    }

    g_ProjectManager.markModified();
    SCENE_LOG_INFO("Projected UVs for object '" + nodeName + "' from current view.");
}

UIWidgets::IconType getPaintToolIcon(Paint::BrushTool tool) {
    switch (tool) {
        case Paint::BrushTool::Paint:  return UIWidgets::IconType::PaintTool;
        case Paint::BrushTool::Erase:  return UIWidgets::IconType::EraseTool;
        case Paint::BrushTool::Soften: return UIWidgets::IconType::SoftenTool;
        case Paint::BrushTool::Stamp:  return UIWidgets::IconType::StampTool;
        case Paint::BrushTool::Fill:   return UIWidgets::IconType::FillTool;
        case Paint::BrushTool::Clone:  return UIWidgets::IconType::CloneTool;
        case Paint::BrushTool::Spray:  return UIWidgets::IconType::SprayTool;
    }
    return UIWidgets::IconType::PaintTool;
}

// Paint-style editor for a 0..1 falloff LUT (x = normalized distance-from-center
// inverse t, y = brush weight). Drag inside the box to reshape the curve; returns
// true when the LUT changed. Lazily initializes to a smoothstep on first use.
static bool drawFalloffCurveEditor(std::vector<float>& lut, float height = 104.0f) {
    constexpr int N = 64;
    if (static_cast<int>(lut.size()) != N) {
        lut.resize(N);
        for (int i = 0; i < N; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(N - 1);
            lut[i] = t * t * (3.0f - 2.0f * t); // smoothstep default
        }
    }
    bool changed = false;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const float w = std::max(80.0f, ImGui::GetContentRegionAvail().x);
    const ImVec2 size(w, height);
    ImGui::InvisibleButton("##falloffcurve", size);
    const bool active = ImGui::IsItemActive();

    dl->AddRectFilled(p0, ImVec2(p0.x + size.x, p0.y + size.y), IM_COL32(20, 22, 26, 255), 4.0f);
    for (int g = 1; g < 4; ++g) {
        const float gx = p0.x + size.x * static_cast<float>(g) / 4.0f;
        const float gy = p0.y + size.y * static_cast<float>(g) / 4.0f;
        dl->AddLine(ImVec2(gx, p0.y), ImVec2(gx, p0.y + size.y), IM_COL32(44, 47, 54, 255));
        dl->AddLine(ImVec2(p0.x, gy), ImVec2(p0.x + size.x, gy), IM_COL32(44, 47, 54, 255));
    }
    dl->AddRect(p0, ImVec2(p0.x + size.x, p0.y + size.y), IM_COL32(70, 75, 85, 255), 4.0f);

    if (active) {
        const ImVec2 m = ImGui::GetIO().MousePos;
        const float fx = std::clamp((m.x - p0.x) / size.x, 0.0f, 1.0f);
        const float fy = std::clamp(1.0f - (m.y - p0.y) / size.y, 0.0f, 1.0f);
        const int bin = std::clamp(static_cast<int>(fx * (N - 1) + 0.5f), 0, N - 1);
        for (int d = -2; d <= 2; ++d) { // soft brush so neighbours follow
            const int b = bin + d;
            if (b < 0 || b >= N) continue;
            const float wgt = 1.0f - std::abs(static_cast<float>(d)) / 3.0f;
            lut[b] = std::clamp(lut[b] * (1.0f - wgt) + fy * wgt, 0.0f, 1.0f);
        }
        changed = true;
    }

    for (int i = 0; i + 1 < N; ++i) {
        const ImVec2 a(p0.x + size.x * static_cast<float>(i) / static_cast<float>(N - 1),
                       p0.y + size.y * (1.0f - lut[i]));
        const ImVec2 b(p0.x + size.x * static_cast<float>(i + 1) / static_cast<float>(N - 1),
                       p0.y + size.y * (1.0f - lut[i + 1]));
        dl->AddLine(a, b, IM_COL32(255, 180, 70, 255), 2.0f);
    }
    return changed;
}

// Fill a falloff LUT from one of the built-in preset shapes (0..4 like applyFalloffCurve).
static void setFalloffLutPreset(std::vector<float>& lut, int preset) {
    constexpr int N = 64;
    lut.resize(N);
    for (int i = 0; i < N; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(N - 1);
        float v;
        switch (preset) {
        case 1:  v = t; break;                                              // Linear
        case 2:  v = t * t; break;                                          // Sharp
        case 3:  v = std::sqrt(std::max(0.0f, 1.0f - (1.0f - t) * (1.0f - t))); break; // Sphere
        case 4:  v = std::sqrt(t); break;                                   // Root
        default: v = t * t * (3.0f - 2.0f * t); break;                      // Smooth
        }
        lut[i] = std::clamp(v, 0.0f, 1.0f);
    }
}

UIWidgets::IconType getSculptToolIcon(SceneUI::SculptBrushTool tool) {
    switch (tool) {
        case SceneUI::SculptBrushTool::Grab:       return UIWidgets::IconType::GrabTool;
        case SceneUI::SculptBrushTool::Inflate:    return UIWidgets::IconType::InflateTool;
        case SceneUI::SculptBrushTool::Smooth:     return UIWidgets::IconType::SmoothTool;
        case SceneUI::SculptBrushTool::Flatten:    return UIWidgets::IconType::FlattenTool;
        case SceneUI::SculptBrushTool::Draw:       return UIWidgets::IconType::DrawTool;
        case SceneUI::SculptBrushTool::Layer:      return UIWidgets::IconType::LayerTool;
        case SceneUI::SculptBrushTool::Pinch:      return UIWidgets::IconType::PinchTool;
        case SceneUI::SculptBrushTool::Clay:       return UIWidgets::IconType::ClayTool;
        case SceneUI::SculptBrushTool::ClayStrips: return UIWidgets::IconType::ClayStripsTool;
        case SceneUI::SculptBrushTool::Crease:     return UIWidgets::IconType::CreaseTool;
        case SceneUI::SculptBrushTool::Scrape:     return UIWidgets::IconType::ScrapeTool;
        case SceneUI::SculptBrushTool::Mask:       return UIWidgets::IconType::MaskTool;
        case SceneUI::SculptBrushTool::DrawSharp:  return UIWidgets::IconType::DrawSharpTool;
        case SceneUI::SculptBrushTool::Nudge:      return UIWidgets::IconType::NudgeTool;
        case SceneUI::SculptBrushTool::Blob:       return UIWidgets::IconType::BlobTool;
        case SceneUI::SculptBrushTool::Fill:       return UIWidgets::IconType::SculptFillTool;
        case SceneUI::SculptBrushTool::SnakeHook:  return UIWidgets::IconType::SnakeHookTool;
        case SceneUI::SculptBrushTool::ElasticDeform: return UIWidgets::IconType::ElasticDeformTool;
    }
    return UIWidgets::IconType::Sculpt;
}

UIWidgets::IconType getMeshSelectModeIcon(MeshElementSelectMode mode) {
    switch (mode) {
        case MeshElementSelectMode::Vertex: return UIWidgets::IconType::VertexMode;
        case MeshElementSelectMode::Edge:   return UIWidgets::IconType::EdgeMode;
        case MeshElementSelectMode::Face:   return UIWidgets::IconType::FaceMode;
        case MeshElementSelectMode::Combined: return UIWidgets::IconType::Mesh;
        default: break;
    }
    return UIWidgets::IconType::Mesh;
}

UIWidgets::IconType getMeshActionIcon(const char* action) {
    if (!action) return UIWidgets::IconType::Mesh;
    const std::string name(action);
    if (name == "Add Face") return UIWidgets::IconType::AddFace;
    if (name == "Merge") return UIWidgets::IconType::MergeVertices;
    if (name == "Merge To Center") return UIWidgets::IconType::MergeVertices;
    if (name == "Weld by Distance") return UIWidgets::IconType::WeldVertices;
    if (name == "Dissolve Vertex") return UIWidgets::IconType::DissolveTopology;
    if (name == "Loop Cut") return UIWidgets::IconType::LoopCutTool;
    if (name == "Dissolve Edge") return UIWidgets::IconType::DissolveTopology;
    if (name == "Extrude Face") return UIWidgets::IconType::ExtrudeFaceTool;
    if (name == "Delete Face") return UIWidgets::IconType::DeleteFaceTool;
    if (name == "Shade Flat") return UIWidgets::IconType::ShadeFlatTool;
    if (name == "Shade Smooth") return UIWidgets::IconType::ShadeSmoothTool;
    return UIWidgets::IconType::Mesh;
}

bool drawMeshIconButton(const char* id,
                        const char* label,
                        const char* tooltip,
                        UIWidgets::IconType icon,
                        const ImVec4& accent,
                        bool active = false,
                        const ImVec2& size = ImVec2(0, 0),
                        bool enabled = true,
                        bool showLabel = false) {
    return UIWidgets::IconActionButton(id, icon, showLabel ? label : "", active, accent, size, tooltip, enabled);
}

bool drawPaintToolSelectorButton(const char* id,
                                 const char* tooltip,
                                 Paint::BrushTool tool,
                                 Paint::BrushTool& current_tool,
                                 float width = 56.0f,
                                 float height = 54.0f) {
    if (UIWidgets::IconActionButton(
            id,
            getPaintToolIcon(tool),
            "",
            current_tool == tool,
            ImVec4(0.28f, 0.90f, 0.82f, 1.0f),
            ImVec2(width, height),
            tooltip)) {
        current_tool = tool;
        return true;
    }
    return false;
}

bool drawSculptToolSelectorButton(const char* id,
                                  const char* tooltip,
                                  SceneUI::SculptBrushTool tool,
                                  SceneUI::SculptBrushTool& current_tool,
                                  float width = 56.0f,
                                  float height = 54.0f) {
    if (UIWidgets::IconActionButton(
            id,
            getSculptToolIcon(tool),
            "",
            current_tool == tool,
            ImVec4(0.22f, 0.55f, 0.88f, 1.0f),
            ImVec2(width, height),
            tooltip)) {
        current_tool = tool;
        return true;
    }
    return false;
}

UIWidgets::IconType getPaintBehaviorIcon(Paint::BrushPaintMode mode) {
    switch (mode) {
        case Paint::BrushPaintMode::Normal: return UIWidgets::IconType::PaintTool;
        case Paint::BrushPaintMode::Mix:    return UIWidgets::IconType::Brush;
        case Paint::BrushPaintMode::Smudge: return UIWidgets::IconType::SoftenTool;
        case Paint::BrushPaintMode::Wet:    return UIWidgets::IconType::Water;
        case Paint::BrushPaintMode::Oil:    return UIWidgets::IconType::ClayTool;
    }
    return UIWidgets::IconType::PaintTool;
}

ImVec4 getPaintBehaviorAccent(Paint::BrushPaintMode mode) {
    switch (mode) {
        case Paint::BrushPaintMode::Normal: return ImVec4(0.52f, 0.80f, 1.0f, 1.0f);
        case Paint::BrushPaintMode::Mix:    return ImVec4(0.78f, 0.66f, 1.0f, 1.0f);
        case Paint::BrushPaintMode::Smudge: return ImVec4(1.0f, 0.62f, 0.42f, 1.0f);
        case Paint::BrushPaintMode::Wet:    return ImVec4(0.34f, 0.84f, 1.0f, 1.0f);
        case Paint::BrushPaintMode::Oil:    return ImVec4(1.0f, 0.74f, 0.28f, 1.0f);
    }
    return ImVec4(0.52f, 0.80f, 1.0f, 1.0f);
}

void drawPaintBehaviorSelector(Paint::PaintModeState& paint_mode_state) {
    struct PaintBehaviorOption {
        Paint::BrushPaintMode mode;
        const char* label;
        const char* tooltip;
    };
    const PaintBehaviorOption paint_mode_options[] = {
        { Paint::BrushPaintMode::Normal, "Normal", "Normal\nDirect paint deposit" },
        { Paint::BrushPaintMode::Mix,    "Mix",    "Mix\nBlend brush and surface color" },
        { Paint::BrushPaintMode::Smudge, "Smudge", "Smudge\nDrag existing surface color" },
        { Paint::BrushPaintMode::Wet,    "Wet",    "Wet\nWatery paint flow" },
        { Paint::BrushPaintMode::Oil,    "Oil",    "Oil\nHeavy pigment body" },
    };

    ImGui::TextUnformatted("Paint Behavior");
    const float available_width = ImGui::GetContentRegionAvail().x;
    const float spacing = 2.0f;
    const float button_width = std::max(56.0f, (available_width - spacing * 4.0f) / 5.0f);
    for (int mode_index = 0; mode_index < IM_ARRAYSIZE(paint_mode_options); ++mode_index) {
        const PaintBehaviorOption& option = paint_mode_options[mode_index];
        const bool selected = paint_mode_state.brush.paint_mode == option.mode;
        if (UIWidgets::IconActionButton(
                option.label,
                getPaintBehaviorIcon(option.mode),
                "",
                selected,
                getPaintBehaviorAccent(option.mode),
                ImVec2(button_width, 42.0f),
                option.tooltip)) {
            paint_mode_state.brush.paint_mode = option.mode;
        }
        if (mode_index + 1 < IM_ARRAYSIZE(paint_mode_options)) {
            ImGui::SameLine(0.0f, spacing);
        }
    }
}

bool brushSupportsRaisedPaint(Paint::BrushTool tool) {
    switch (tool) {
        case Paint::BrushTool::Paint:
        case Paint::BrushTool::Erase:
        case Paint::BrushTool::Stamp:
        case Paint::BrushTool::Fill:
        case Paint::BrushTool::Spray:
            return true;
        case Paint::BrushTool::Soften:
        case Paint::BrushTool::Clone:
            return false;
    }
    return false;
}

void syncHeightMaskPaintToggles(Paint::PaintModeState& state) {
    if (state.active_channel != Paint::PaintChannel::Mask) {
        return;
    }

    state.brush.write_height_mask = true;
    state.auto_normal_from_height = true;
}

float raisedPaintHeightValue(float contribution) {
    return 0.5f + std::clamp(contribution, 0.0f, 1.0f) * 0.5f;
}

float wrapBrushAngleDegrees(float degrees) {
    float wrapped = std::fmod(degrees + 180.0f, 360.0f);
    if (wrapped < 0.0f) {
        wrapped += 360.0f;
    }
    return wrapped - 180.0f;
}

bool brushHasDirectionalShape(const Paint::BrushSettings& brush) {
    return brush.shape != Paint::BrushShape::Circle &&
           (brush.shape == Paint::BrushShape::Flat ||
            brush.shape == Paint::BrushShape::Rectangle ||
            brush.shape == Paint::BrushShape::Capsule ||
            std::abs(brush.shape_aspect - 1.0f) > 0.01f);
}

bool brushHasRotatableTexture(const Paint::BrushSettings& brush) {
    const bool has_imported_alpha =
        brush.use_imported_alpha &&
        brush.alpha_texture &&
        brush.alpha_texture->is_loaded();
    const bool has_paint_texture =
        brush.use_paint_texture &&
        brush.paint_texture &&
        brush.paint_texture->is_loaded();
    const bool has_procedural_direction =
        brush.alpha_preset == Paint::BrushAlphaPreset::Noise ||
        brush.alpha_preset == Paint::BrushAlphaPreset::Scratch ||
        brush.alpha_preset == Paint::BrushAlphaPreset::Cloud;
    bool has_channel_texture = false;
    for (const Paint::BrushChannelInput& input : brush.channel_inputs) {
        if (input.enabled &&
            input.use_paint_texture &&
            input.paint_texture &&
            input.paint_texture->is_loaded()) {
            has_channel_texture = true;
            break;
        }
    }
    return has_imported_alpha || has_paint_texture || has_channel_texture || has_procedural_direction;
}

bool brushSupportsViewportRotation(const Paint::BrushSettings& brush) {
    return brushHasDirectionalShape(brush) || brushHasRotatableTexture(brush);
}

Paint::BrushChannelInput* getBrushChannelInput(Paint::BrushSettings& brush, Paint::PaintChannel channel) {
    const size_t index = static_cast<size_t>(channel);
    if (index >= brush.channel_inputs.size()) {
        return nullptr;
    }
    return &brush.channel_inputs[index];
}

const Paint::BrushChannelInput* getBrushChannelInput(const Paint::BrushSettings& brush, Paint::PaintChannel channel) {
    const size_t index = static_cast<size_t>(channel);
    if (index >= brush.channel_inputs.size()) {
        return nullptr;
    }
    const Paint::BrushChannelInput& input = brush.channel_inputs[index];
    return input.enabled ? &input : nullptr;
}

Vec3 getPreviewChannelColor(const Paint::BrushSettings& brush, Paint::PaintChannel channel) {
    if (const Paint::BrushChannelInput* input = getBrushChannelInput(brush, channel)) {
        return input->color;
    }
    return brush.color;
}

std::shared_ptr<Texture> getPreviewChannelTexture(const Paint::BrushSettings& brush, Paint::PaintChannel channel) {
    if (const Paint::BrushChannelInput* input = getBrushChannelInput(brush, channel)) {
        if (input->use_paint_texture && input->paint_texture && input->paint_texture->is_loaded()) {
            return input->paint_texture;
        }
        return nullptr;
    }
    if (brush.use_paint_texture && brush.paint_texture && brush.paint_texture->is_loaded()) {
        return brush.paint_texture;
    }
    return nullptr;
}

float getPreviewChannelTintStrength(const Paint::BrushSettings& brush, Paint::PaintChannel channel) {
    if (const Paint::BrushChannelInput* input = getBrushChannelInput(brush, channel)) {
        return input->tint_strength;
    }
    return brush.paint_texture_tint_strength;
}

Paint::PaintTextureTintMode getPreviewChannelTintMode(const Paint::BrushSettings& brush, Paint::PaintChannel channel) {
    if (const Paint::BrushChannelInput* input = getBrushChannelInput(brush, channel)) {
        return input->tint_mode;
    }
    return brush.paint_texture_tint_mode;
}

bool beginBrushDockSection(const char* label, bool default_open = true) {
    ImVec4 accent(0.40f, 0.78f, 1.0f, 1.0f);
    const std::string section_label = label ? label : "";
    if (section_label.find("Tool") != std::string::npos) accent = ImVec4(0.30f, 0.88f, 0.82f, 1.0f);
    else if (section_label.find("Stroke") != std::string::npos) accent = ImVec4(0.62f, 0.82f, 1.0f, 1.0f);
    else if (section_label.find("Color") != std::string::npos) accent = ImVec4(1.0f, 0.58f, 0.66f, 1.0f);
    else if (section_label.find("Behavior") != std::string::npos) accent = ImVec4(0.96f, 0.76f, 0.36f, 1.0f);
    else if (section_label.find("Alpha") != std::string::npos) accent = ImVec4(0.86f, 0.68f, 1.0f, 1.0f);

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 7.0f));
    
    float fr = 10.0f;
    if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
        fr = ThemeManager::instance().current().style.frameRounding;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, fr);
    
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(accent.x * 0.22f, accent.y * 0.22f, accent.z * 0.22f, 0.92f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(accent.x * 0.30f, accent.y * 0.30f, accent.z * 0.30f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(accent.x * 0.34f, accent.y * 0.34f, accent.z * 0.34f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(accent.x, accent.y, accent.z, 0.24f));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.92f, 0.95f, 0.98f, 1.0f));
    const bool open = ImGui::CollapsingHeader(
        label,
        (default_open ? ImGuiTreeNodeFlags_DefaultOpen : 0) |
        ImGuiTreeNodeFlags_SpanAvailWidth);
    ImGui::PopStyleColor(5);
    ImGui::PopStyleVar(2);
    return open;
}

std::shared_ptr<Material> clonePaintableMaterial(const std::shared_ptr<Material>& src, const std::string& new_name) {
    if (!src) {
        return nullptr;
    }

    if (auto* pbsdf = dynamic_cast<PrincipledBSDF*>(src.get())) {
        auto clone = std::make_shared<PrincipledBSDF>(*pbsdf);
        clone->materialName = new_name;
        return clone;
    }

    if (auto* vol = dynamic_cast<Volumetric*>(src.get())) {
        auto clone = std::make_shared<Volumetric>(*vol);
        clone->materialName = new_name;
        return clone;
    }

    return nullptr;
}

uint16_t makeObjectSlotPaintMaterialUnique(UIContext& ctx,
                                           const std::string& obj_name,
                                           int slot_index,
                                           uint16_t source_material_id,
                                           std::vector<std::pair<int, std::shared_ptr<Triangle>>>& mesh_entries,
                                           std::vector<uint16_t>& slot_ids) {
    auto& mgr = MaterialManager::getInstance();
    const std::string paint_name = obj_name + "_PaintSlot_" + std::to_string(slot_index);

    if (mgr.hasMaterial(paint_name)) {
        const uint16_t existing_id = mgr.getMaterialID(paint_name);
        if (slot_index >= 0 &&
            slot_index < static_cast<int>(slot_ids.size()) &&
            slot_ids[slot_index] != existing_id) {
            for (auto& pair : mesh_entries) {
                if (pair.second && pair.second->getMaterialID() == source_material_id) {
                    pair.second->setMaterialID(existing_id);
                }
            }
            slot_ids[slot_index] = existing_id;
            ctx.renderer.updateMeshMaterialBinding(ctx.scene, obj_name, source_material_id, existing_id);
        }
        return existing_id;
    }

    std::shared_ptr<Material> source_material = mgr.getMaterialShared(source_material_id);
    std::shared_ptr<Material> clone = clonePaintableMaterial(source_material, paint_name);
    if (!clone) {
        return source_material_id;
    }

    const uint16_t new_id = mgr.addMaterial(paint_name, clone);
    for (auto& pair : mesh_entries) {
        if (pair.second && pair.second->getMaterialID() == source_material_id) {
            pair.second->setMaterialID(new_id);
        }
    }

    if (slot_index >= 0 && slot_index < static_cast<int>(slot_ids.size())) {
        slot_ids[slot_index] = new_id;
    }

    ctx.renderer.updateMeshMaterialBinding(ctx.scene, obj_name, source_material_id, new_id);
    return new_id;
}

void syncPaintBrushToTerrain(SceneUI& ui, TerrainObject* terrain) {
    if (!terrain || !ui.paint_mode_state.enabled) {
        return;
    }

    ui.terrain_brush.enabled = true;
    ui.terrain_brush.active_terrain_id = terrain->id;
    ui.terrain_brush.mode = 5;
    ui.terrain_brush.paint_channel = ui.paint_mode_state.active_layer_index;
    ui.terrain_brush.radius = ui.paint_mode_state.brush.radius;
    ui.terrain_brush.strength = ui.paint_mode_state.brush.strength;
    ui.terrain_brush.curve = 0.25f + (ui.paint_mode_state.brush.falloff * 3.75f);
    ui.terrain_brush.show_preview = ui.paint_mode_state.brush.show_preview;
}

bool isMeshPaintTargetHit(const HitRecord& rec, const Paint::MeshPaintAdapter& adapter) {
    if (!rec.triangle) {
        return false;
    }

    return rec.triangle->getNodeName() == adapter.getNodeName() &&
           rec.triangle->getMaterialID() == adapter.getMaterialID();
}

bool raycastMeshPaintTargetFallback(UIContext& ctx,
                                    const ImVec2& screen_pos,
                                    const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& mesh_entries,
                                    const Paint::MeshPaintAdapter& adapter,
                                    HitRecord& hit_record) {
    if (!ctx.scene.camera) {
        return false;
    }

    const ImVec2 display = ImGui::GetIO().DisplaySize;
    if (display.x <= 1.0f || display.y <= 1.0f) {
        return false;
    }

    if (mesh_entries.empty()) {
        return false;
    }

    const float u = std::clamp(screen_pos.x / display.x, 0.0f, 1.0f);
    const float v = std::clamp(1.0f - (screen_pos.y / display.y), 0.0f, 1.0f);
    Ray ray = ctx.scene.camera->get_ray(u, v);

    bool hit = false;
    float closest = 1e30f;
    HitRecord best;
    for (const auto& entry : mesh_entries) {
        const auto& tri = entry.second;
        if (!tri || !tri->visible || tri->getMaterialID() != adapter.getMaterialID()) {
            continue;
        }
        if (tri->getTransformPtr() && !tri->hasAnySkinWeights()) {
            tri->updateTransformedVertices();
        }

        HitRecord candidate;
        if (tri->hit(ray, 0.001f, closest, candidate, true)) {
            closest = candidate.t;
            best = candidate;
            hit = true;
        }
    }

    if (hit) {
        hit_record = best;
    }
    return hit;
}

Vec3 closestPointOnTriangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, float& out_u, float& out_v, float& out_w) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;
    const float d1 = ab.dot(ap);
    const float d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        out_u = 1.0f; out_v = 0.0f; out_w = 0.0f;
        return a;
    }

    const Vec3 bp = p - b;
    const float d3 = ab.dot(bp);
    const float d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        out_u = 0.0f; out_v = 1.0f; out_w = 0.0f;
        return b;
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        out_u = 1.0f - v; out_v = v; out_w = 0.0f;
        return a + ab * v;
    }

    const Vec3 cp = p - c;
    const float d5 = ab.dot(cp);
    const float d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        out_u = 0.0f; out_v = 0.0f; out_w = 1.0f;
        return c;
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        out_u = 1.0f - w; out_v = 0.0f; out_w = w;
        return a + ac * w;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        out_u = 0.0f; out_v = 1.0f - w; out_w = w;
        return b + (c - b) * w;
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    out_u = 1.0f - v - w;
    out_v = v;
    out_w = w;
    return a + ab * v + ac * w;
}

Vec2 interpolateTriangleUV(const Triangle& tri, float bu, float bv, float bw) {
    const auto [uv0, uv1, uv2] = tri.getUVCoordinates();
    return Vec2(
        uv0.u * bu + uv1.u * bv + uv2.u * bw,
        uv0.v * bu + uv1.v * bv + uv2.v * bw
    );
}

bool computeTriangleUvFrame(const Triangle& tri, const Vec3& normal_hint, Vec3& tangent, Vec3& bitangent) {
    Vec3 normal = normal_hint.length_squared() > 1e-8f ? normal_hint.normalize() : Vec3(0.0f, 1.0f, 0.0f);
    const Vec3 p0 = tri.getVertexPosition(0);
    const Vec3 p1 = tri.getVertexPosition(1);
    const Vec3 p2 = tri.getVertexPosition(2);
    const Vec3 edge1 = p1 - p0;
    const Vec3 edge2 = p2 - p0;
    const Vec2 deltaUV1 = tri.t1 - tri.t0;
    const Vec2 deltaUV2 = tri.t2 - tri.t0;
    const float det = deltaUV1.u * deltaUV2.v - deltaUV2.u * deltaUV1.v;
    if (std::abs(det) <= 1e-6f) {
        return false;
    }

    const float f = 1.0f / det;
    tangent = Vec3(
        f * (deltaUV2.v * edge1.x - deltaUV1.v * edge2.x),
        f * (deltaUV2.v * edge1.y - deltaUV1.v * edge2.y),
        f * (deltaUV2.v * edge1.z - deltaUV1.v * edge2.z));
    if (tangent.length_squared() <= 1e-8f) {
        return false;
    }
    tangent = (tangent - normal * tangent.dot(normal)).normalize();
    if (tangent.length_squared() <= 1e-8f) {
        return false;
    }

    Vec3 computed_bitangent(
        f * (-deltaUV2.u * edge1.x + deltaUV1.u * edge2.x),
        f * (-deltaUV2.u * edge1.y + deltaUV1.u * edge2.y),
        f * (-deltaUV2.u * edge1.z + deltaUV1.u * edge2.z));
    if (computed_bitangent.length_squared() <= 1e-8f) {
        return false;
    }
    computed_bitangent = (computed_bitangent - normal * computed_bitangent.dot(normal)).normalize();

    bitangent = normal.cross(tangent).normalize();
    if (bitangent.dot(computed_bitangent) < 0.0f) {
        bitangent = -bitangent;
    }
    return bitangent.length_squared() > 1e-8f;
}

Vec3 reflectVectorForMeshMirror(const Matrix4x4& local_to_world,
                                const Matrix4x4& world_to_local,
                                const Vec3& world_vector,
                                bool mx,
                                bool my,
                                bool mz) {
    Vec3 local_vector = world_to_local.transform_vector(world_vector);
    if (mx) local_vector.x *= -1.0f;
    if (my) local_vector.y *= -1.0f;
    if (mz) local_vector.z *= -1.0f;
    Vec3 reflected = local_to_world.transform_vector(local_vector);
    return reflected.length_squared() > 1e-8f ? reflected.normalize() : reflected;
}

Paint::BrushSettings makeMirroredBrushForHit(const Paint::BrushSettings& brush,
                                             const HitRecord& source_hit,
                                             const HitRecord& mirrored_hit,
                                             bool mx,
                                             bool my,
                                             bool mz) {
    Paint::BrushSettings mirrored_brush = brush;
    if (!source_hit.triangle || !mirrored_hit.triangle) {
        if (mx || mz) mirrored_brush.flip_alpha_x = !mirrored_brush.flip_alpha_x;
        if (my) mirrored_brush.flip_alpha_y = !mirrored_brush.flip_alpha_y;
        return mirrored_brush;
    }

    Vec3 source_tangent, source_bitangent;
    Vec3 target_tangent, target_bitangent;
    if (!computeTriangleUvFrame(*source_hit.triangle, source_hit.normal, source_tangent, source_bitangent) ||
        !computeTriangleUvFrame(*mirrored_hit.triangle, mirrored_hit.normal, target_tangent, target_bitangent)) {
        if (mx || mz) mirrored_brush.flip_alpha_x = !mirrored_brush.flip_alpha_x;
        if (my) mirrored_brush.flip_alpha_y = !mirrored_brush.flip_alpha_y;
        return mirrored_brush;
    }

    const Matrix4x4 local_to_world = source_hit.triangle->getTransformMatrix();
    const Matrix4x4 world_to_local = local_to_world.inverse();
    const Vec3 reflected_tangent = reflectVectorForMeshMirror(local_to_world, world_to_local, source_tangent, mx, my, mz);
    const Vec3 reflected_bitangent = reflectVectorForMeshMirror(local_to_world, world_to_local, source_bitangent, mx, my, mz);

    const float a00 = target_tangent.dot(reflected_tangent);
    const float a01 = target_bitangent.dot(reflected_tangent);
    const float a10 = target_tangent.dot(reflected_bitangent);
    const float a11 = target_bitangent.dot(reflected_bitangent);
    const float det = a00 * a11 - a01 * a10;

    if (det < 0.0f) {
        mirrored_brush.flip_alpha_x = !mirrored_brush.flip_alpha_x;
        const float r00 = -a00;
        const float r10 = -a10;
        if (std::isfinite(r00) && std::isfinite(r10) && (r00 * r00 + r10 * r10) > 1e-6f) {
            const float angle_degrees = std::atan2(r10, r00) * 57.2957795f;
            mirrored_brush.alpha_rotation_degrees =
                wrapBrushAngleDegrees(mirrored_brush.alpha_rotation_degrees + angle_degrees);
        }
    } else {
        if (std::isfinite(a00) && std::isfinite(a10) && (a00 * a00 + a10 * a10) > 1e-6f) {
            const float angle_degrees = std::atan2(a10, a00) * 57.2957795f;
            mirrored_brush.alpha_rotation_degrees =
                wrapBrushAngleDegrees(mirrored_brush.alpha_rotation_degrees + angle_degrees);
        }
    }

    return mirrored_brush;
}

bool resolveMirroredMeshPaintHit(UIContext& ctx,
                                 const Paint::MeshPaintAdapter& adapter,
                                 const HitRecord& source_hit,
                                 bool mx,
                                 bool my,
                                 bool mz,
                                 HitRecord& out_hit) {
    auto source_tri = adapter.getTriangle();
    if (!source_tri) {
        return false;
    }

    const Matrix4x4 local_to_world = source_tri->getTransformMatrix();
    const Matrix4x4 world_to_local = local_to_world.inverse();
    Vec3 local_pos = world_to_local.transform_point(source_hit.point);
    Vec3 local_normal = world_to_local.transform_vector(source_hit.normal).normalize();
    if (mx) { local_pos.x *= -1.0f; local_normal.x *= -1.0f; }
    if (my) { local_pos.y *= -1.0f; local_normal.y *= -1.0f; }
    if (mz) { local_pos.z *= -1.0f; local_normal.z *= -1.0f; }

    const Vec3 target_world = local_to_world.transform_point(local_pos);
    const std::string node_name = adapter.getNodeName();
    const uint16_t material_id = adapter.getMaterialID();

    float best_dist2 = FLT_MAX;
    std::shared_ptr<Triangle> best_tri;
    Vec3 best_point;
    Vec3 best_normal;
    Vec2 best_uv;

    for (const auto& obj : ctx.scene.world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (!tri || tri->terrain_id != -1) {
            continue;
        }
        if (tri->getNodeName() != node_name || tri->getMaterialID() != material_id) {
            continue;
        }

        float bu = 0.0f, bv = 0.0f, bw = 0.0f;
        const Vec3 a = tri->getV0();
        const Vec3 b = tri->getV1();
        const Vec3 c = tri->getV2();
        const Vec3 closest = closestPointOnTriangle(target_world, a, b, c, bu, bv, bw);
        const float dist2 = (closest - target_world).length_squared();
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_tri = tri;
            best_point = closest;
            best_normal = (tri->getN0() * bu + tri->getN1() * bv + tri->getN2() * bw).normalize();
            best_uv = interpolateTriangleUV(*tri, bu, bv, bw);
        }
    }

    if (!best_tri || best_dist2 > 0.25f) {
        return false;
    }

    out_hit = source_hit;
    out_hit.triangle = best_tri.get();
    out_hit.point = best_point;
    out_hit.normal = best_normal;
    out_hit.uv = best_uv;
    return true;
}

bool hasMeshPaintTargetLocked(SceneUI& ui) {
    if (!ui.paint_mode_state.enabled || !ui.paint_mode_state.hasValidTarget()) {
        return false;
    }

    auto adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(ui.paint_mode_state.getAdapter());
    if (!adapter || !adapter->isValid()) {
        return false;
    }

    Paint::PaintTextureSet* set = adapter->getTextureSet();
    return set && set->initialized;
}

Vec3 lerpColor(const Vec3& a, const Vec3& b, float t) {
    const float clamped = std::clamp(t, 0.0f, 1.0f);
    return Vec3(
        a.x + (b.x - a.x) * clamped,
        a.y + (b.y - a.y) * clamped,
        a.z + (b.z - a.z) * clamped
    );
}

Vec3 applyPreviewTint(const Vec3& sampled, const Vec3& tint, float tint_strength, Paint::PaintTextureTintMode tint_mode) {
    const float strength = std::clamp(tint_strength, 0.0f, 1.0f);
    Vec3 tinted = sampled;
    switch (tint_mode) {
        case Paint::PaintTextureTintMode::Multiply:
            tinted = Vec3(
                std::clamp(sampled.x * tint.x, 0.0f, 1.0f),
                std::clamp(sampled.y * tint.y, 0.0f, 1.0f),
                std::clamp(sampled.z * tint.z, 0.0f, 1.0f));
            break;
        case Paint::PaintTextureTintMode::Recolor: {
            const float luminance = std::clamp(sampled.x * 0.299f + sampled.y * 0.587f + sampled.z * 0.114f, 0.0f, 1.0f);
            tinted = Vec3(
                std::clamp(tint.x * luminance, 0.0f, 1.0f),
                std::clamp(tint.y * luminance, 0.0f, 1.0f),
                std::clamp(tint.z * luminance, 0.0f, 1.0f));
            break;
        }
        case Paint::PaintTextureTintMode::Overlay: {
            const auto overlay = [](float base, float blend) -> float {
                return base < 0.5f
                    ? (2.0f * base * blend)
                    : (1.0f - 2.0f * (1.0f - base) * (1.0f - blend));
            };
            tinted = Vec3(
                std::clamp(overlay(sampled.x, tint.x), 0.0f, 1.0f),
                std::clamp(overlay(sampled.y, tint.y), 0.0f, 1.0f),
                std::clamp(overlay(sampled.z, tint.z), 0.0f, 1.0f));
            break;
        }
    }
    return Vec3(
        sampled.x + (tinted.x - sampled.x) * strength,
        sampled.y + (tinted.y - sampled.y) * strength,
        sampled.z + (tinted.z - sampled.z) * strength);
}

float uiHash01(float x, float y) {
    const float value = std::sin(x * 91.173f + y * 37.719f) * 43758.5453f;
    return value - std::floor(value);
}

float uiBrushHashNoise(float x, float y) {
    const float value = std::sin(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return value - std::floor(value);
}

float uiSmoothNoise(float x, float y) {
    const float ix = std::floor(x);
    const float iy = std::floor(y);
    const float tx = x - ix;
    const float ty = y - iy;

    const float n00 = uiBrushHashNoise(ix, iy);
    const float n10 = uiBrushHashNoise(ix + 1.0f, iy);
    const float n01 = uiBrushHashNoise(ix, iy + 1.0f);
    const float n11 = uiBrushHashNoise(ix + 1.0f, iy + 1.0f);

    const float sx = tx * tx * (3.0f - 2.0f * tx);
    const float sy = ty * ty * (3.0f - 2.0f * ty);
    const float nx0 = n00 + (n10 - n00) * sx;
    const float nx1 = n01 + (n11 - n01) * sx;
    return nx0 + (nx1 - nx0) * sy;
}

void uiRotateBrushCoords(float& x, float& y, float degrees) {
    if (std::abs(degrees) <= 0.001f) {
        return;
    }

    const float radians = degrees * 3.14159265f / 180.0f;
    const float cs = std::cos(radians);
    const float sn = std::sin(radians);
    const float rx = x * cs - y * sn;
    const float ry = x * sn + y * cs;
    x = rx;
    y = ry;
}

float uiBrushShapeAspectScale(const Paint::BrushSettings& brush) {
    return std::sqrt(std::clamp(brush.shape_aspect, 0.1f, 8.0f));
}

float uiBrushShapeDistance(Paint::BrushShape shape, float x, float y, float roundness) {
    const float ax = std::abs(x);
    const float ay = std::abs(y);
    switch (shape) {
        case Paint::BrushShape::Circle:
            return std::sqrt(x * x + y * y);
        case Paint::BrushShape::Rectangle: {
            const float p = 8.0f + std::clamp(roundness, 0.0f, 1.0f) * 16.0f;
            return std::pow(std::pow(ax, p) + std::pow(ay, p), 1.0f / p);
        }
        case Paint::BrushShape::Capsule: {
            const float half_segment = 0.55f;
            const float qx = std::max(ax - half_segment, 0.0f);
            return std::sqrt(qx * qx + ay * ay);
        }
        case Paint::BrushShape::Flat: {
            const float p = 10.0f + std::clamp(roundness, 0.0f, 1.0f) * 18.0f;
            return std::pow(std::pow(ax, p) + std::pow(ay, p), 1.0f / p);
        }
    }
    return std::sqrt(x * x + y * y);
}

struct UIBrushFootprintSample {
    float x = 0.0f;
    float y = 0.0f;
    float dist_norm = 0.0f;
};

UIBrushFootprintSample uiSampleBrushFootprint(const Paint::BrushSettings& brush, float x, float y) {
    uiRotateBrushCoords(x, y, -brush.alpha_rotation_degrees);
    const float aspect_scale = uiBrushShapeAspectScale(brush);
    x /= aspect_scale;
    y *= aspect_scale;

    UIBrushFootprintSample sample;
    sample.x = x;
    sample.y = y;
    sample.dist_norm = uiBrushShapeDistance(brush.shape, x, y, brush.shape_roundness);
    return sample;
}

float uiSampleBrushAlpha(Paint::BrushAlphaPreset preset, float nx, float ny, float scale, float rotation_degrees, bool radial_gate) {
    float sx = nx * std::max(0.01f, scale);
    float sy = ny * std::max(0.01f, scale);
    uiRotateBrushCoords(sx, sy, rotation_degrees);
    const float radial = radial_gate ? std::clamp(1.0f - std::sqrt(nx * nx + ny * ny), 0.0f, 1.0f) : 1.0f;

    switch (preset) {
        case Paint::BrushAlphaPreset::SoftRound:
            return radial;
        case Paint::BrushAlphaPreset::HardRound:
            return radial > 0.2f ? 1.0f : 0.0f;
        case Paint::BrushAlphaPreset::Noise: {
            const float noise = uiBrushHashNoise((sx + 1.0f) * 17.0f, (sy + 1.0f) * 19.0f);
            return radial * (0.45f + noise * 0.55f);
        }
        case Paint::BrushAlphaPreset::Scratch: {
            const float streaks = std::abs(std::sin((sx * 18.0f) + (sy * 2.5f)));
            const float breakup = uiBrushHashNoise((sx + 2.0f) * 11.0f, (sy + 2.0f) * 23.0f);
            const float scratch = std::pow(1.0f - streaks, 3.0f) * (0.35f + breakup * 0.65f);
            return radial * std::clamp(scratch * 2.2f, 0.0f, 1.0f);
        }
        case Paint::BrushAlphaPreset::Cloud: {
            const float n1 = uiSmoothNoise((sx + 3.0f) * 3.0f, (sy + 5.0f) * 3.0f);
            const float n2 = uiSmoothNoise((sx + 11.0f) * 6.0f, (sy + 7.0f) * 6.0f) * 0.5f;
            const float n3 = uiSmoothNoise((sx + 19.0f) * 12.0f, (sy + 13.0f) * 12.0f) * 0.25f;
            const float cloud = std::clamp((n1 + n2 + n3) / 1.75f, 0.0f, 1.0f);
            return radial * (0.25f + cloud * 0.75f);
        }
    }

    return radial;
}

float uiSampleImportedBrushAlpha(const std::shared_ptr<Texture>& texture, float nx, float ny, float scale, float rotation_degrees) {
    if (!texture || !texture->is_loaded()) {
        return 1.0f;
    }

    float scaled_x = nx * std::max(0.01f, scale);
    float scaled_y = ny * std::max(0.01f, scale);
    uiRotateBrushCoords(scaled_x, scaled_y, rotation_degrees);
    const float u = std::clamp(scaled_x * 0.5f + 0.5f, 0.0f, 1.0f);
    const float v = std::clamp(scaled_y * 0.5f + 0.5f, 0.0f, 1.0f);
    return std::clamp(texture->sampleIntensity(u, v), 0.0f, 1.0f);
}

float uiSampleBrushMask(const Paint::BrushSettings& brush, float nx, float ny) {
    if (brush.flip_alpha_x) nx = -nx;
    if (brush.flip_alpha_y) ny = -ny;
    if (brush.use_imported_alpha && brush.alpha_texture && brush.alpha_texture->is_loaded()) {
        return uiSampleImportedBrushAlpha(brush.alpha_texture, nx, ny, brush.alpha_scale, brush.alpha_rotation_degrees);
    }

    const bool radial_gate = brush.shape == Paint::BrushShape::Circle;
    return uiSampleBrushAlpha(brush.alpha_preset, nx, ny, brush.alpha_scale, brush.alpha_rotation_degrees, radial_gate);
}

void drawBrushAlphaPreview(const Paint::BrushSettings& brush) {
    const ImVec2 size(92.0f, 92.0f);
    ImGui::TextDisabled("Alpha Preview");
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##BrushAlphaPreview", size);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), IM_COL32(20, 22, 26, 255), 6.0f);
    dl->AddRect(p, ImVec2(p.x + size.x, p.y + size.y), IM_COL32(70, 74, 82, 255), 6.0f);

    const int cells = 40;
    const float pad = 8.0f;
    const float inner_w = size.x - pad * 2.0f;
    const float inner_h = size.y - pad * 2.0f;
    const float cell_w = inner_w / static_cast<float>(cells);
    const float cell_h = inner_h / static_cast<float>(cells);

    for (int y = 0; y < cells; ++y) {
        for (int x = 0; x < cells; ++x) {
            const float u = ((static_cast<float>(x) + 0.5f) / static_cast<float>(cells)) * 2.0f - 1.0f;
            const float v = ((static_cast<float>(y) + 0.5f) / static_cast<float>(cells)) * 2.0f - 1.0f;
            const UIBrushFootprintSample fp = uiSampleBrushFootprint(brush, u, v);
            if (fp.dist_norm > 1.0f) {
                continue;
            }

            const float falloff = std::clamp(brush.falloff, 0.0f, 1.0f);
            const float inner = std::clamp(1.0f - falloff, 0.0f, 1.0f);
            float base = 1.0f;
            if (fp.dist_norm > inner) {
                const float t = std::clamp((fp.dist_norm - inner) / std::max(0.001f, 1.0f - inner), 0.0f, 1.0f);
                base = 1.0f - (t * t * (3.0f - 2.0f * t));
            }

            const float alpha = std::clamp(
                base * uiSampleBrushMask(brush, u, v),
                0.0f, 1.0f);
            const int shade = static_cast<int>(50.0f + alpha * 205.0f);
            const ImU32 color = IM_COL32(shade, shade, shade, 255);
            const float x0 = p.x + pad + static_cast<float>(x) * cell_w;
            const float y0 = p.y + pad + static_cast<float>(y) * cell_h;
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + cell_w + 0.5f, y0 + cell_h + 0.5f), color);
        }
    }
}

void drawBrushPaintTexturePreview(const Paint::BrushSettings& brush) {
    if (!brush.paint_texture || !brush.paint_texture->is_loaded()) {
        return;
    }

    const ImVec2 size(92.0f, 92.0f);
    ImGui::TextDisabled("Paint Preview");
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##BrushPaintPreview", size);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), IM_COL32(20, 22, 26, 255), 6.0f);
    dl->AddRect(p, ImVec2(p.x + size.x, p.y + size.y), IM_COL32(70, 74, 82, 255), 6.0f);

    const int cells = 40;
    const float pad = 8.0f;
    const float inner_w = size.x - pad * 2.0f;
    const float inner_h = size.y - pad * 2.0f;
    const float cell_w = inner_w / static_cast<float>(cells);
    const float cell_h = inner_h / static_cast<float>(cells);

    for (int y = 0; y < cells; ++y) {
        for (int x = 0; x < cells; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(cells);
            const float v = 1.0f - ((static_cast<float>(y) + 0.5f) / static_cast<float>(cells));
            Vec3 sampled = brush.paint_texture->get_color_bilinear(u, v);
            sampled = applyPreviewTint(sampled, brush.color, brush.paint_texture_tint_strength, brush.paint_texture_tint_mode);
            sampled.x = std::clamp(sampled.x, 0.0f, 1.0f);
            sampled.y = std::clamp(sampled.y, 0.0f, 1.0f);
            sampled.z = std::clamp(sampled.z, 0.0f, 1.0f);

            const ImU32 color = IM_COL32(
                static_cast<int>(sampled.x * 255.0f + 0.5f),
                static_cast<int>(sampled.y * 255.0f + 0.5f),
                static_cast<int>(sampled.z * 255.0f + 0.5f),
                255);
            const float x0 = p.x + pad + static_cast<float>(x) * cell_w;
            const float y0 = p.y + pad + static_cast<float>(y) * cell_h;
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + cell_w + 0.5f, y0 + cell_h + 0.5f), color);
        }
    }
}

std::string brushAlphaDisplayName(const std::string& path) {
    if (path.empty()) {
        return {};
    }

    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string brushAssetDisplayName(const std::string& path) {
    return brushAlphaDisplayName(path);
}

std::filesystem::path utf8PathFromString(const std::string& path) {
    const char8_t* begin = reinterpret_cast<const char8_t*>(path.data());
    return std::filesystem::path(begin, begin + path.size());
}

const char* paintChannelFileTag(Paint::PaintChannel channel) {
    switch (channel) {
        case Paint::PaintChannel::BaseColor: return "basecolor";
        case Paint::PaintChannel::Normal: return "normal";
        case Paint::PaintChannel::Roughness: return "roughness";
        case Paint::PaintChannel::Metallic: return "metallic";
        case Paint::PaintChannel::Emission: return "emission";
        case Paint::PaintChannel::Mask: return "mask";
        case Paint::PaintChannel::Transmission: return "transmission";
        case Paint::PaintChannel::Opacity: return "opacity";
    }
    return "channel";
}

const char* paintChannelDisplayName(Paint::PaintChannel channel) {
    switch (channel) {
        case Paint::PaintChannel::BaseColor: return "Base Color";
        case Paint::PaintChannel::Normal: return "Normal";
        case Paint::PaintChannel::Roughness: return "Roughness";
        case Paint::PaintChannel::Metallic: return "Metallic";
        case Paint::PaintChannel::Emission: return "Emission";
        case Paint::PaintChannel::Mask: return "Height Mask";
        case Paint::PaintChannel::Transmission: return "Transmission";
        case Paint::PaintChannel::Opacity: return "Opacity";
    }
    return "Channel";
}

TextureType inferBrushTextureType(Paint::PaintChannel channel) {
    switch (channel) {
        case Paint::PaintChannel::BaseColor:
            return TextureType::Albedo;
        case Paint::PaintChannel::Emission:
            return TextureType::Emission;
        case Paint::PaintChannel::Normal:
            return TextureType::Normal;
        case Paint::PaintChannel::Roughness:
            return TextureType::Roughness;
        case Paint::PaintChannel::Metallic:
            return TextureType::Metallic;
        case Paint::PaintChannel::Mask:
            return TextureType::Unknown;
        case Paint::PaintChannel::Transmission:
            return TextureType::Transmission;
        case Paint::PaintChannel::Opacity:
            return TextureType::Opacity;
    }
    return TextureType::Unknown;
}

bool isMeshPaintUiChannelEnabled(Paint::PaintChannel channel) {
    return channel != Paint::PaintChannel::Opacity;
}

std::vector<Paint::PaintChannel> getSelectedMaterialBrushChannels(const Paint::PaintModeState& state) {
    std::vector<Paint::PaintChannel> channels;
    if (isMeshPaintUiChannelEnabled(state.active_channel)) {
        channels.push_back(state.active_channel);
    }
    for (int i = 0; i < static_cast<int>(Paint::kPaintChannelCount); ++i) {
        const Paint::PaintChannel channel = static_cast<Paint::PaintChannel>(i);
        if (!isMeshPaintUiChannelEnabled(channel)) {
            continue;
        }
        if (channel == state.active_channel) {
            continue;
        }
        if (state.linked_channels[static_cast<size_t>(i)]) {
            channels.push_back(channel);
        }
    }
    return channels;
}

bool exportTextureToPng(const std::shared_ptr<Texture>& texture, const std::string& file_path) {
    if (!texture || texture->width <= 0 || texture->height <= 0 || texture->pixels.empty() || file_path.empty()) {
        return false;
    }

    const std::filesystem::path out_path = utf8PathFromString(file_path);
    if (std::filesystem::exists(out_path)) {
        std::error_code ec;
        std::filesystem::remove(out_path, ec);
        if (ec) {
            return false;
        }
    }

    return stbi_write_png(
        out_path.string().c_str(),
        texture->width,
        texture->height,
        4,
        texture->pixels.data(),
        texture->width * 4) != 0;
}

bool exportTextureSetChannels(const Paint::PaintTextureSet& set, const std::filesystem::path& directory) {
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        return false;
    }

    const struct ChannelExportEntry {
        Paint::PaintChannel channel;
        const char* suffix;
    } entries[] = {
        { Paint::PaintChannel::BaseColor, "basecolor" },
        { Paint::PaintChannel::Normal, "normal" },
        { Paint::PaintChannel::Roughness, "roughness" },
        { Paint::PaintChannel::Metallic, "metallic" },
        { Paint::PaintChannel::Emission, "emission" },
        { Paint::PaintChannel::Mask, "mask" },
        { Paint::PaintChannel::Transmission, "transmission" },
        { Paint::PaintChannel::Opacity, "opacity" }
    };

    bool wrote_any = false;
    for (const auto& entry : entries) {
        std::shared_ptr<Texture> texture = set.getTexture(entry.channel);
        if (!texture) {
            continue;
        }

        const std::string base_name = set.target_node_name.empty() ? "paintset" : set.target_node_name;
        const std::filesystem::path out_path =
            directory / (base_name + "_" + std::to_string(set.material_id) + "_" + entry.suffix + ".png");
        if (exportTextureToPng(texture, out_path.string())) {
            wrote_any = true;
        }
    }

    return wrote_any;
}

float estimateMeshPaintWorldRadius(const Triangle& tri, const Texture& texture, float radius_px) {
    const Vec3 v0 = tri.getV0();
    const Vec3 v1 = tri.getV1();
    const Vec3 v2 = tri.getV2();
    const Vec3 cross = (v1 - v0).cross(v2 - v0);
    const float world_area = 0.5f * std::sqrt(cross.length_squared());

    const auto [uv0, uv1, uv2] = tri.getUVCoordinates();
    const Vec2 duv1 = uv1 - uv0;
    const Vec2 duv2 = uv2 - uv0;
    const float uv_area = 0.5f * std::abs(duv1.u * duv2.v - duv1.v * duv2.u);
    if (world_area <= 1e-6f || uv_area <= 1e-8f || texture.width <= 0 || texture.height <= 0) {
        return 0.0f;
    }

    const float world_per_uv = std::sqrt(world_area / uv_area);
    const float baseline = 1024.0f;
    const float uv_radius = radius_px / baseline;
    return std::max(0.001f, uv_radius * world_per_uv);
}

ImU32 makePreviewColorU32(const Paint::BrushSettings& brush, Paint::PaintChannel channel, float nx, float ny, float alpha, bool ghost) {
    Vec3 color = getPreviewChannelColor(brush, channel);
    std::shared_ptr<Texture> paint_texture = getPreviewChannelTexture(brush, channel);
    if (paint_texture) {
        float sx = nx * std::max(0.01f, brush.alpha_scale);
        float sy = ny * std::max(0.01f, brush.alpha_scale);
        if (brush.flip_alpha_x) sx = -sx;
        if (brush.flip_alpha_y) sy = -sy;
        uiRotateBrushCoords(sx, sy, brush.alpha_rotation_degrees);
        const float u = std::clamp(sx * 0.5f + 0.5f, 0.0f, 1.0f);
        const float v = std::clamp(sy * 0.5f + 0.5f, 0.0f, 1.0f);

        if (channel == Paint::PaintChannel::BaseColor || channel == Paint::PaintChannel::Emission) {
            Vec3 sampled = paint_texture->get_color_bilinear(u, v);
            sampled = applyPreviewTint(
                sampled,
                color,
                getPreviewChannelTintStrength(brush, channel),
                getPreviewChannelTintMode(brush, channel));
            color = Vec3(
                std::clamp(sampled.x, 0.0f, 1.0f),
                std::clamp(sampled.y, 0.0f, 1.0f),
                std::clamp(sampled.z, 0.0f, 1.0f));
        } else {
            const float texture_value = paint_texture->sampleIntensity(u, v);
            const float tint_value = std::clamp((color.x + color.y + color.z) / 3.0f, 0.0f, 1.0f);
            const float grayscale = texture_value + (texture_value * tint_value - texture_value) *
                std::clamp(getPreviewChannelTintStrength(brush, channel), 0.0f, 1.0f);
            color = Vec3(grayscale, grayscale, grayscale);
        }
    } else if (channel != Paint::PaintChannel::BaseColor && channel != Paint::PaintChannel::Emission) {
        const float grayscale = std::clamp((brush.color.x + brush.color.y + brush.color.z) / 3.0f, 0.0f, 1.0f);
        color = Vec3(grayscale, grayscale, grayscale);
    }

    const int a = static_cast<int>(std::clamp(alpha, 0.0f, 1.0f) * (ghost ? 90.0f : 150.0f));
    return IM_COL32(
        static_cast<int>(std::clamp(color.x, 0.0f, 1.0f) * 255.0f),
        static_cast<int>(std::clamp(color.y, 0.0f, 1.0f) * 255.0f),
        static_cast<int>(std::clamp(color.z, 0.0f, 1.0f) * 255.0f),
        a);
}

void drawMeshPaintPreview(UIContext& ctx, const HitRecord& rec, const Paint::MeshPaintAdapter& adapter, const Paint::BrushSettings& brush, Paint::PaintChannel channel, bool ghost = false, bool show_alpha_rotate_ring = false) {
    if (!brush.show_preview || !rec.triangle || !ctx.scene.camera) {
        return;
    }

    Paint::PaintTextureSet* set = adapter.getTextureSet();
    std::shared_ptr<Texture> texture = set ? set->getTexture(channel) : nullptr;
    if (!texture || texture->width <= 0 || texture->height <= 0) {
        return;
    }

    const float world_radius = estimateMeshPaintWorldRadius(*rec.triangle, *texture, brush.radius);
    if (world_radius <= 0.0f) {
        return;
    }

    const Vec3 normal = rec.normal.length_squared() > 1e-6f ? rec.normal.normalize() : Vec3(0, 1, 0);
    Vec3 tangent, bitangent;
    
    bool has_uv_tangent = false;
    if (rec.triangle) {
        const Vec3 p0 = rec.triangle->getVertexPosition(0);
        const Vec3 p1 = rec.triangle->getVertexPosition(1);
        const Vec3 p2 = rec.triangle->getVertexPosition(2);
        const Vec3 edge1 = p1 - p0;
        const Vec3 edge2 = p2 - p0;
        const Vec2 deltaUV1 = rec.triangle->t1 - rec.triangle->t0;
        const Vec2 deltaUV2 = rec.triangle->t2 - rec.triangle->t0;
        
        const float f = 1.0f / (deltaUV1.u * deltaUV2.v - deltaUV2.u * deltaUV1.v);
        if (std::isfinite(f) && std::abs(deltaUV1.u * deltaUV2.v - deltaUV2.u * deltaUV1.v) > 1e-6f) {
            tangent.x = f * (deltaUV2.v * edge1.x - deltaUV1.v * edge2.x);
            tangent.y = f * (deltaUV2.v * edge1.y - deltaUV1.v * edge2.y);
            tangent.z = f * (deltaUV2.v * edge1.z - deltaUV1.v * edge2.z);
            tangent = tangent.normalize();
            
            // To prevent precision issues or degenerates
            if (tangent.length_squared() > 1e-6f) {
                // Ensure orthogonality
                tangent = (tangent - normal * tangent.dot(normal)).normalize();
                if (tangent.length_squared() > 1e-6f) {
                    bitangent = normal.cross(tangent).normalize();
                    // Determine handedness of bitangent relative to computed UV bitangent
                    Vec3 computed_bitangent;
                    computed_bitangent.x = f * (-deltaUV2.u * edge1.x + deltaUV1.u * edge2.x);
                    computed_bitangent.y = f * (-deltaUV2.u * edge1.y + deltaUV1.u * edge2.y);
                    computed_bitangent.z = f * (-deltaUV2.u * edge1.z + deltaUV1.u * edge2.z);
                    if (bitangent.dot(computed_bitangent) < 0.0f) {
                        bitangent = -bitangent;
                    }
                    has_uv_tangent = true;
                }
            }
        }
    }
    
    if (!has_uv_tangent) {
        tangent = std::abs(normal.y) < 0.95f ? normal.cross(Vec3(0, 1, 0)) : normal.cross(Vec3(1, 0, 0));
        if (tangent.length_squared() <= 1e-8f) {
            return;
        }
        tangent = tangent.normalize();
        bitangent = normal.cross(tangent).normalize();
    }

    auto project = [&](const Vec3& p) -> ImVec2 {
        Camera& cam = *ctx.scene.camera;
        ImGuiIO& io = ImGui::GetIO();
        const float win_w = io.DisplaySize.x;
        const float win_h = io.DisplaySize.y;
        Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
        Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
        Vec3 cam_up = cam_right.cross(cam_forward).normalize();
        float fov_rad = cam.vfov * 3.14159f / 180.0f;

        Vec3 to_p = p - cam.lookfrom;
        float depth = to_p.dot(cam_forward);
        if (depth <= 0.1f) {
            return ImVec2(-1000.0f, -1000.0f);
        }

        float half_h = depth * tanf(fov_rad / 2.0f);
        float half_w = half_h * (win_w / std::max(1.0f, win_h));
        float lx = to_p.dot(cam_right);
        float ly = to_p.dot(cam_up);
        float ndc_x = lx / half_w;
        float ndc_y = ly / half_h;
        return ImVec2((ndc_x * 0.5f + 0.5f) * win_w, (0.5f - ndc_y * 0.5f) * win_h);
    };

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    constexpr int segments = 40;
    const float inner_scale = std::clamp(1.0f - brush.falloff, 0.15f, 0.95f);
    const ImVec2 center_screen = project(rec.point + normal * 0.002f);
    const ImVec2 radius_screen = project(rec.point + tangent * world_radius + normal * 0.002f);
    const float approx_screen_radius = (center_screen.x < -900.0f || radius_screen.x < -900.0f)
        ? 24.0f
        : std::sqrt(
            (radius_screen.x - center_screen.x) * (radius_screen.x - center_screen.x) +
            (radius_screen.y - center_screen.y) * (radius_screen.y - center_screen.y));

    if (!ghost) {
        const int grid = std::clamp(static_cast<int>(approx_screen_radius * 1.15f), 36, 96);
        const float extent_scale = std::max(uiBrushShapeAspectScale(brush), 1.0f / uiBrushShapeAspectScale(brush));
        const float cell_span = (2.0f * extent_scale) / static_cast<float>(grid);
        for (int gy = 0; gy < grid; ++gy) {
            for (int gx = 0; gx < grid; ++gx) {
                const float bx = (((static_cast<float>(gx) + 0.5f) / static_cast<float>(grid)) * 2.0f - 1.0f) * extent_scale;
                const float by = (((static_cast<float>(gy) + 0.5f) / static_cast<float>(grid)) * 2.0f - 1.0f) * extent_scale;
                const UIBrushFootprintSample fp = uiSampleBrushFootprint(brush, bx, by);
                if (fp.dist_norm > 1.0f) {
                    continue;
                }

                const float falloff = std::clamp(brush.falloff, 0.0f, 1.0f);
                const float inner = std::clamp(1.0f - falloff, 0.0f, 1.0f);
                float base = 1.0f;
                if (fp.dist_norm > inner) {
                    const float t = std::clamp((fp.dist_norm - inner) / std::max(0.001f, 1.0f - inner), 0.0f, 1.0f);
                    base = 1.0f - (t * t * (3.0f - 2.0f * t));
                }

                const float sample_x = bx;
                const float sample_y = -by;
                const float mask_alpha = uiSampleBrushMask(brush, sample_x, sample_y);
                const float alpha = base * mask_alpha;
                if (alpha <= 0.025f) {
                    continue;
                }

                const float local_x = bx * world_radius;
                const float local_y = by * world_radius;
                const float half_cell = world_radius * cell_span * 0.42f;
                const Vec3 center = rec.point + tangent * local_x + bitangent * local_y + normal * 0.002f;
                const Vec3 right = tangent * half_cell;
                const Vec3 up = bitangent * half_cell;
                const ImVec2 c = project(center);
                const ImVec2 px = project(center + right);
                const ImVec2 py = project(center + up);
                if (c.x < -900.0f || px.x < -900.0f || py.x < -900.0f) {
                    continue;
                }
                const ImU32 fill = makePreviewColorU32(brush, channel, sample_x, sample_y, alpha, ghost);
                const ImVec2 p0 = project(center - right - up);
                const ImVec2 p1 = project(center + right - up);
                const ImVec2 p2 = project(center + right + up);
                const ImVec2 p3 = project(center - right + up);
                if (p0.x < -900.0f || p1.x < -900.0f || p2.x < -900.0f || p3.x < -900.0f) {
                    continue;
                }
                const ImVec2 points[4] = { p0, p1, p2, p3 };
                dl->AddConvexPolyFilled(points, 4, fill);
            }
        }
    }

    auto draw_ring = [&](float radius, ImU32 color, float thickness) {
        ImVec2 prev;
        bool has_prev = false;
        for (int i = 0; i <= segments; ++i) {
            const float angle = (static_cast<float>(i) / static_cast<float>(segments)) * 6.2831853f;
            const Vec3 offset = tangent * std::cos(angle) * radius + bitangent * std::sin(angle) * radius;
            const ImVec2 p = project(rec.point + offset + normal * 0.002f);
            if (p.x > -900.0f && p.y > -900.0f) {
                if (has_prev) {
                    dl->AddLine(prev, p, color, thickness);
                }
                prev = p;
                has_prev = true;
            } else {
                has_prev = false;
            }
        }
    };

    auto draw_shape_outline = [&](float scale, ImU32 color, float thickness) {
        const float extent_scale = std::max(uiBrushShapeAspectScale(brush), 1.0f / uiBrushShapeAspectScale(brush));
        ImVec2 prev;
        bool has_prev = false;
        for (int i = 0; i <= segments; ++i) {
            const float angle = (static_cast<float>(i) / static_cast<float>(segments)) * 6.2831853f;
            const float dir_x = std::cos(angle);
            const float dir_y = std::sin(angle);
            float lo = 0.0f;
            float hi = extent_scale * 1.25f;
            for (int step = 0; step < 10; ++step) {
                const float mid = (lo + hi) * 0.5f;
                const UIBrushFootprintSample fp = uiSampleBrushFootprint(brush, dir_x * mid, dir_y * mid);
                if (fp.dist_norm <= scale) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            const Vec3 offset = tangent * (dir_x * lo * world_radius) + bitangent * (dir_y * lo * world_radius);
            const ImVec2 p = project(rec.point + offset + normal * 0.002f);
            if (p.x > -900.0f && p.y > -900.0f) {
                if (has_prev) {
                    dl->AddLine(prev, p, color, thickness);
                }
                prev = p;
                has_prev = true;
            } else {
                has_prev = false;
            }
        }
    };

    const ImU32 outer = ghost ? IM_COL32(255, 196, 96, 120) : IM_COL32(255, 196, 96, 230);
    const ImU32 inner = ghost ? IM_COL32(255, 196, 96, 70) : IM_COL32(255, 196, 96, 120);
    draw_shape_outline(1.0f, outer, ghost ? 1.2f : 2.0f);
    draw_shape_outline(inner_scale, inner, 1.0f);

    if (show_alpha_rotate_ring && !ghost) {
        const float radians = brush.alpha_rotation_degrees * 3.14159265f / 180.0f;
        const Vec3 dir = tangent * std::cos(radians) + bitangent * std::sin(radians);
        const ImVec2 center = project(rec.point + normal * 0.004f);
        const ImVec2 tip = project(rec.point + dir * world_radius * 1.18f + normal * 0.004f);
        if (center.x > -900.0f && tip.x > -900.0f) {
            const ImU32 ring = IM_COL32(112, 220, 255, 230);
            const ImU32 fill = IM_COL32(112, 220, 255, 245);
            draw_ring(world_radius * 1.18f, ring, 2.0f);
            dl->AddLine(center, tip, ring, 2.4f);
            dl->AddCircleFilled(tip, 4.0f, fill, 16);
        }
    }
}

 // namespace

void SceneUI::ensurePaintBrushPresets() {
    if (paint_brush_presets_initialized) {
        return;
    }

    paint_brush_presets_initialized = true;
    paint_brush_presets.clear();

    auto add_preset = [&](const std::string& name, const Paint::BrushSettings& brush) {
        paint_brush_presets.push_back(PaintBrushPreset{ name, brush });
    };

    Paint::BrushSettings soft_paint;
    soft_paint.radius = 36.0f;
    soft_paint.strength = 0.6f;
    soft_paint.falloff = 0.8f;
    soft_paint.alpha_preset = Paint::BrushAlphaPreset::SoftRound;
    add_preset("Soft Paint", soft_paint);

    Paint::BrushSettings hard_stamp;
    hard_stamp.tool = Paint::BrushTool::Stamp;
    hard_stamp.radius = 48.0f;
    hard_stamp.strength = 1.0f;
    hard_stamp.falloff = 0.15f;
    hard_stamp.alpha_preset = Paint::BrushAlphaPreset::HardRound;
    hard_stamp.stamp_mode = Paint::StampPlacementMode::Single;
    add_preset("Hard Stamp", hard_stamp);

    Paint::BrushSettings flat_brush;
    flat_brush.radius = 42.0f;
    flat_brush.strength = 0.85f;
    flat_brush.falloff = 0.28f;
    flat_brush.spacing = 0.08f;
    flat_brush.alpha_preset = Paint::BrushAlphaPreset::HardRound;
    flat_brush.shape = Paint::BrushShape::Flat;
    flat_brush.shape_aspect = 4.0f;
    flat_brush.shape_roundness = 0.75f;
    flat_brush.follow_stroke_angle = true;
    add_preset("Flat Brush", flat_brush);

    Paint::BrushSettings oil_bristle = flat_brush;
    oil_bristle.strength = 0.7f;
    oil_bristle.falloff = 0.42f;
    oil_bristle.alpha_preset = Paint::BrushAlphaPreset::Scratch;
    oil_bristle.paint_mode = Paint::BrushPaintMode::Oil;
    oil_bristle.wetness = 0.62f;
    oil_bristle.wet_lifetime_seconds = 4.8f;
    oil_bristle.wet_diffusion = 0.16f;
    oil_bristle.wet_runoff = 0.06f;
    oil_bristle.wet_absorption = 0.08f;
    oil_bristle.paint_load = 0.93f;
    oil_bristle.pickup_rate = 0.12f;
    oil_bristle.deposit_rate = 0.88f;
    oil_bristle.wet_terminal_buildup = 0.72f;
    oil_bristle.wet_terminal_softness = 0.72f;
    add_preset("Oil Bristle", oil_bristle);

    Paint::BrushSettings marker;
    marker.radius = 18.0f;
    marker.strength = 0.8f;
    marker.falloff = 0.18f;
    marker.spacing = 0.06f;
    marker.alpha_preset = Paint::BrushAlphaPreset::Noise;
    marker.shape = Paint::BrushShape::Capsule;
    marker.shape_aspect = 3.0f;
    marker.shape_roundness = 0.65f;
    marker.follow_stroke_angle = true;
    add_preset("Marker Chisel", marker);

    Paint::BrushSettings grunge;
    grunge.tool = Paint::BrushTool::Stamp;
    grunge.radius = 56.0f;
    grunge.strength = 0.8f;
    grunge.falloff = 0.55f;
    grunge.alpha_preset = Paint::BrushAlphaPreset::Cloud;
    grunge.stamp_mode = Paint::StampPlacementMode::Continuous;
    grunge.spacing = 0.35f;
    add_preset("Grunge Stamp", grunge);

    Paint::BrushSettings wet;
    wet.radius = 32.0f;
    wet.strength = 0.7f;
    wet.falloff = 0.7f;
    wet.paint_mode = Paint::BrushPaintMode::Wet;
    wet.wetness = 0.75f;
    wet.paint_load = 0.9f;
    wet.pickup_rate = 0.3f;
    wet.deposit_rate = 0.7f;
    wet.wet_terminal_buildup = 0.58f;
    wet.wet_terminal_softness = 0.68f;
    add_preset("Wet Blend", wet);

    Paint::BrushSettings smudge;
    smudge.radius = 30.0f;
    smudge.strength = 0.55f;
    smudge.falloff = 0.75f;
    smudge.paint_mode = Paint::BrushPaintMode::Smudge;
    smudge.smudge_strength = 0.8f;
    add_preset("Smudge Soft", smudge);

    Paint::BrushSettings spray;
    spray.tool = Paint::BrushTool::Spray;
    spray.radius = 42.0f;
    spray.strength = 0.45f;
    spray.falloff = 0.6f;
    spray.alpha_preset = Paint::BrushAlphaPreset::SoftRound;
    spray.spray_particles = 16;
    spray.spray_spread = 0.75f;
    spray.spray_droplet_size = 0.18f;
    add_preset("Soft Spray", spray);
}

void SceneUI::ensureSculptBrushPresets() {
    if (sculpt_brush_presets_initialized) return;
    sculpt_brush_presets_initialized = true;
    sculpt_brush_presets.clear();

    auto add = [&](const std::string& name, const Paint::BrushSettings& b) {
        sculpt_brush_presets.push_back(PaintBrushPreset{ name, b });
        };

    Paint::BrushSettings soft;
    soft.radius = 0.3f;
    soft.strength = 1.0f;
    soft.falloff = 0.75f;
    soft.alpha_preset = Paint::BrushAlphaPreset::SoftRound;
    add("Soft Sculpt", soft);

    Paint::BrushSettings grab = soft;
    grab.radius = 0.42f;
    grab.strength = 1.35f;
    grab.falloff = 0.82f;
    add("Grab Form", grab);

    Paint::BrushSettings draw = soft;
    draw.radius = 0.18f;
    draw.strength = 2.1f;
    draw.falloff = 0.58f;
    draw.alpha_preset = Paint::BrushAlphaPreset::HardRound;
    add("Draw Build", draw);

    Paint::BrushSettings clay;
    clay.radius = 0.2f;
    clay.strength = 2.5f;
    clay.falloff = 0.45f;
    clay.alpha_preset = Paint::BrushAlphaPreset::HardRound;
    add("Clay", clay);

    Paint::BrushSettings layer = clay;
    layer.radius = 0.22f;
    layer.strength = 1.8f;
    layer.falloff = 0.55f;
    add("Layer", layer);

    Paint::BrushSettings clay_strips;
    clay_strips.radius = 0.12f;
    clay_strips.strength = 3.0f;
    clay_strips.falloff = 0.25f;
    clay_strips.alpha_preset = Paint::BrushAlphaPreset::HardRound;
    add("Clay Strips", clay_strips);

    Paint::BrushSettings smooth;
    smooth.radius = 0.4f;
    smooth.strength = 1.5f;
    smooth.falloff = 0.8f;
    smooth.alpha_preset = Paint::BrushAlphaPreset::SoftRound;
    add("Smooth", smooth);

    Paint::BrushSettings inflate = soft;
    inflate.radius = 0.24f;
    inflate.strength = 1.8f;
    inflate.falloff = 0.52f;
    inflate.alpha_preset = Paint::BrushAlphaPreset::HardRound;
    add("Inflate Volume", inflate);

    Paint::BrushSettings sharp;
    sharp.radius = 0.12f;
    sharp.strength = 3.0f;
    sharp.falloff = 0.25f;
    sharp.alpha_preset = Paint::BrushAlphaPreset::HardRound;
    add("Sharp Detail", sharp);

    Paint::BrushSettings flatten;
    flatten.radius = 0.5f;
    flatten.strength = 2.0f;
    flatten.falloff = 0.6f;
    flatten.alpha_preset = Paint::BrushAlphaPreset::SoftRound;
    add("Flatten", flatten);

    Paint::BrushSettings pinch;
    pinch.radius = 0.15f;
    pinch.strength = 2.0f;
    pinch.falloff = 0.5f;
    pinch.alpha_preset = Paint::BrushAlphaPreset::SoftRound;
    add("Pinch", pinch);

    Paint::BrushSettings crease;
    crease.radius = 0.09f;
    crease.strength = 2.6f;
    crease.falloff = 0.35f;
    crease.alpha_preset = Paint::BrushAlphaPreset::HardRound;
    add("Crease", crease);

    Paint::BrushSettings scrape;
    scrape.radius = 0.24f;
    scrape.strength = 2.2f;
    scrape.falloff = 0.55f;
    scrape.alpha_preset = Paint::BrushAlphaPreset::HardRound;
    add("Scrape", scrape);
}
void SceneUI::activateEditWorkspace(UIContext& ctx) {
    bool hasSelection = (ctx.selection.selected.type == SelectableType::Object &&
                         ctx.selection.selected.object != nullptr);
    if (!hasSelection) {
        return;
    }
    const std::string selectedNodeName = ctx.selection.selected.object->getNodeName();

    ensureCPUSyncForPicking(ctx);

    mesh_workspace_mode = SceneUI::MeshWorkspaceMode::Edit;
    mesh_overlay_settings.enabled = true;
    mesh_overlay_settings.edit_mode = true;
    sculpt_mode_state.enabled = false;
    sculpt_mode_state.active_target_name.clear();
    terrain_sculpt_proxy_active = false;
    ctx.selection.mesh_element_mode = MeshElementSelectMode::Vertex;
    clearEditableMeshSelection();
    active_mesh_edit_object_name = selectedNodeName;
    active_mesh_edit_object_ptr = ctx.selection.selected.object.get();
    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    ensureMeshEditLayer(ctx, selectedNodeName);
}

void SceneUI::activateSculptWorkspace(UIContext& ctx) {
    bool hasSelection = (ctx.selection.selected.type == SelectableType::Object &&
                         ctx.selection.selected.object != nullptr);
    if (!hasSelection) {
        return;
    }
    const std::string selectedNodeName = ctx.selection.selected.object->getNodeName();
    auto selectedTriangle = std::dynamic_pointer_cast<Triangle>(ctx.selection.selected.object);
    const bool selectedIsTerrain = selectedTriangle && selectedTriangle->terrain_id != -1;

    mesh_workspace_mode = SceneUI::MeshWorkspaceMode::Sculpt;
    sculpt_mode_state.enabled = true;
    mesh_overlay_settings.edit_mode = true;
    active_mesh_edit_object_name = selectedNodeName;
    active_mesh_edit_object_ptr = ctx.selection.selected.object.get();
    sculpt_mode_state.active_target_name = selectedNodeName;
    clearEditableMeshSelection();
    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    terrain_sculpt_proxy_active = selectedIsTerrain;

    if (!mesh_cache_valid) {
        rebuildMeshCache(ctx.scene.world.objects);
    }
    auto meshIt = mesh_cache.find(selectedNodeName);
    if (meshIt != mesh_cache.end()) {
        for (auto& entry : meshIt->second) {
            if (entry.second) {
                entry.second->updateTransformedVertices();
            }
        }
        objects_needing_cpu_sync.erase(selectedNodeName);
    }
    extern bool g_bvh_rebuild_pending;
    g_bvh_rebuild_pending = true;

    if (selectedIsTerrain) {
        terrain_brush.active_terrain_id = selectedTriangle->terrain_id;
        if (terrain_brush.mode < 0 || terrain_brush.mode > 4) {
            terrain_brush.mode = 0;
        }
    } else {
        ctx.selection.mesh_element_mode = MeshElementSelectMode::Object;
        ensureMeshEditLayer(ctx, selectedNodeName);

        // Repair degenerate shading normals on sculpt entry. A dense/subdivided mesh
        // whose stored per-vertex normals collapsed to zero (modifier averaging) enters
        // sculpt with zero-length normals: shading goes black in every mode and the brush
        // hit normal falls back to world-up (0,1,0), so the brush disc lies flat instead
        // of following the surface. This used to call applyMeshShadingSettings, but that
        // REWRITES every normal from the per-object MeshShadingSettings (default =
        // auto-smooth), silently flipping flat/custom-shaded meshes to smooth on entry.
        // repairSculptEntryShadingNormals fixes ONLY the broken normals and leaves valid
        // ones alone, so the object stays in whatever shading mode it already had.
        repairSculptEntryShadingNormals(ctx, selectedNodeName, true);
    }
}

void SceneUI::drawModifiersPanel(UIContext& ctx) {
    UIWidgets::PushControlSurfaceStyle(ImVec4(0.92f, 0.56f, 0.38f, 1.0f));

    if (UIWidgets::BeginSection("Overlays & Selection", ImVec4(0.5f, 0.8f, 1.0f, 1.0f))) {
        ImGui::Checkbox("Viewport Mesh Overlay", &mesh_overlay_settings.enabled);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Render edit cage overlays in the viewport.");
        }

        ImGui::Checkbox("X-Ray View Mode", &mesh_overlay_settings.xray_mode);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Render edit overlay lines through opaque geometry.");
        }

        ImGui::Checkbox("Show Face Normals", &mesh_overlay_settings.show_normals);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Render outward face normals for selected faces.");
        }
        if (mesh_overlay_settings.show_normals) {
            ImGui::SliderFloat("Normals Length", &mesh_overlay_settings.normals_length, 0.05f, 2.0f, "%.2f m");
        }

        ImGui::Spacing();
        ImGui::TextDisabled("Selection Tool:");
        bool isBox = (mesh_overlay_settings.selection_tool == 0);
        bool isLasso = (mesh_overlay_settings.selection_tool == 1);
        if (ImGui::RadioButton("Box Select##sidebar", isBox)) {
            mesh_overlay_settings.selection_tool = 0;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Switch to Box Select (Shortcut: B)");
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Lasso Select##sidebar", isLasso)) {
            mesh_overlay_settings.selection_tool = 1;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Switch to Lasso Select (Shortcut: L)");
        }
        ImGui::Spacing();

        ImGui::Checkbox("Soft Selection (Proportional)", &mesh_overlay_settings.proportional_edit);
        if (mesh_overlay_settings.proportional_edit) {
            ImGui::SliderFloat("Soft Radius", &mesh_overlay_settings.proportional_radius, 0.05f, 5.0f, "%.2f");
            static const char* falloffTypes[] = { "Smooth", "Linear", "Sharp", "Sphere", "Root" };
            ImGui::Combo("Falloff Shape", &mesh_overlay_settings.proportional_falloff_type, falloffTypes, IM_ARRAYSIZE(falloffTypes));
            ImGui::SliderFloat("Falloff Bias", &mesh_overlay_settings.proportional_falloff, 0.05f, 1.0f, "%.2f");
        }

        UIWidgets::EndSection();
    }

    UIWidgets::Divider();

    if (UIWidgets::BeginSection("Modeling", ImVec4(0.8f, 0.4f, 0.9f, 1.0f))) {
        bool hasSelection = (ctx.selection.selected.type == SelectableType::Object &&
                             ctx.selection.selected.object != nullptr);
        const bool editSessionActive =
            mesh_workspace_mode == SceneUI::MeshWorkspaceMode::Edit &&
            mesh_overlay_settings.enabled &&
            mesh_overlay_settings.edit_mode &&
            !active_mesh_edit_object_name.empty();

        if (!hasSelection) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.6f, 1.0f), "Please select a mesh object.");
            UIWidgets::EndSection();
        } else {
            const std::string selectedNodeName =
                hasSelection ? ctx.selection.selected.object->getNodeName() : std::string{};
            const std::string effectiveNodeName =
                editSessionActive ? active_mesh_edit_object_name : selectedNodeName;

            if (!selectedNodeName.empty()) {
                ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Selected: %s", selectedNodeName.c_str());
            }
            if (editSessionActive && effectiveNodeName != selectedNodeName) {
                ImGui::TextColored(ImVec4(0.95f, 0.78f, 0.40f, 1.0f), "Editing: %s", effectiveNodeName.c_str());
            }

            if (!mesh_cache_valid) {
                rebuildMeshCache(ctx.scene.world.objects);
            }

            auto meshEntriesIt = mesh_cache.find(effectiveNodeName);
            const bool hasMeshEntries = meshEntriesIt != mesh_cache.end();
            auto selectedTriangle =
                hasSelection ? std::dynamic_pointer_cast<Triangle>(ctx.selection.selected.object) : nullptr;
            const bool selectedIsTerrain = selectedTriangle && selectedTriangle->terrain_id != -1;
            const bool isVertexMode = (ctx.selection.mesh_element_mode == MeshElementSelectMode::Vertex);
            const bool isEdgeMode = (ctx.selection.mesh_element_mode == MeshElementSelectMode::Edge);
            const bool isFaceMode = (ctx.selection.mesh_element_mode == MeshElementSelectMode::Face);
            const bool isCombinedMode = (ctx.selection.mesh_element_mode == MeshElementSelectMode::Combined);
            // Auto-enter edit workspace only for newly-selected objects.
            // If the user manually exited edit mode for this object, respect that.
            if (hasSelection && !mesh_overlay_settings.edit_mode) {
                if (modifier_panel_exit_object != selectedNodeName) {
                    activateEditWorkspace(ctx);
                }
            }
            // Reset the exit flag when a different object is selected.
            if (!modifier_panel_exit_object.empty() && modifier_panel_exit_object != selectedNodeName) {
                modifier_panel_exit_object.clear();
            }

            if (mesh_workspace_mode == SceneUI::MeshWorkspaceMode::Edit) {
                if (ImGui::Checkbox("Viewport Mesh Overlay", &mesh_overlay_settings.enabled)) {
                    if (!mesh_overlay_settings.enabled) {
                        modifier_panel_exit_object = selectedNodeName;
                        resetMeshEditState(ctx);
                    }
                }
            }

            if (mesh_overlay_settings.edit_mode) {
                if (mesh_edit_layer.active && mesh_edit_layer.object_name == effectiveNodeName) {
                    ImGui::SeparatorText("Edit Layer");
                    bool layerEnabled = mesh_edit_layer.enabled;
                    if (ImGui::Checkbox("##LayerEnabled", &layerEnabled)) {
                        setMeshEditLayerEnabled(ctx, layerEnabled);
                    }
                    ImGui::SameLine();
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Layer 1");
                    ImGui::SameLine();
                    ImGui::TextDisabled(mesh_edit_layer.enabled ? "Active" : "Muted");

                    const float buttonWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
                    if (ImGui::Button("Apply", ImVec2(buttonWidth, 0))) {
                        applyMeshEditLayer(ctx);
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Bake the current edit layer changes into the mesh.");
                    ImGui::SameLine();
                    if (ImGui::Button("Discard", ImVec2(buttonWidth, 0))) {
                        discardMeshEditLayer(ctx);
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Throw away the current edit layer changes.");
                    ImGui::Spacing();
                } else if (effectiveNodeName == active_mesh_edit_object_name) {
                    ImGui::SeparatorText("Edit Layer");
                    if (drawMeshIconButton(
                            "EditLayerCreate",
                            "Create Edit Layer",
                            "Create Edit Layer\nStart a non-destructive edit layer for this mesh.",
                            UIWidgets::IconType::AddKey,
                            ImVec4(0.52f, 0.82f, 1.0f, 1.0f),
                            false,
                            ImVec2(UIWidgets::GetInspectorActionWidth(), 40.0f))) {
                        ensureMeshEditLayer(ctx, effectiveNodeName);
                    }
                    ImGui::Spacing();
                }
            }
            if (mesh_workspace_mode == SceneUI::MeshWorkspaceMode::Edit && mesh_overlay_settings.edit_mode) {
                // Select Mode and Mesh Tools are now in the right Tool Dock.

                // Tool Options Context Panel
                UIWidgets::Divider();
                if (UIWidgets::BeginSection("Tool Options", ImVec4(0.72f, 0.84f, 1.0f, 1.0f))) {
                    bool hasAnyOptions = false;

                    if (isVertexMode || isCombinedMode) {
                        ImGui::SliderFloat("Weld Distance", &mesh_vertex_weld_distance, 0.0001f, 0.5f, "%.4f", ImGuiSliderFlags_Logarithmic);
                        ImGui::TextDisabled("Weld threshold for collapsing nearby vertices.");
                        hasAnyOptions = true;
                    }

                    if (isEdgeMode || isCombinedMode) {
                        if (hasAnyOptions) ImGui::Spacing();
                        ImGui::SliderFloat("Loop Cut Pos", &mesh_loop_cut_position, 0.05f, 0.95f, "%.2f");
                        ImGui::TextDisabled("Relative position of the loop cut along ring edges.");

                        // Catmull-Clark edge crease authoring (like Blender's Shift+E).
                        // The slider is a free "value to apply" (not bound to the selection,
                        // so dragging is never clobbered); the selection's current crease is
                        // shown as text and applied on release / via the buttons.
                        ImGui::Spacing();
                        const float selCrease = getSelectedEdgesAverageCrease(ctx);
                        const bool hasEdgeSelection = (selCrease >= 0.0f);
                        ImGui::BeginDisabled(!hasEdgeSelection);
                        ImGui::SliderFloat("Crease", &mesh_edge_crease_value, 0.0f, 1.0f, "%.2f");
                        if (ImGui::IsItemDeactivatedAfterEdit() && hasEdgeSelection) {
                            applyCreaseToSelectedEdges(ctx, mesh_edge_crease_value);
                        }
                        if (ImGui::Button("Apply Crease##edge") && hasEdgeSelection) {
                            applyCreaseToSelectedEdges(ctx, mesh_edge_crease_value);
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Clear Crease##edge") && hasEdgeSelection) {
                            applyCreaseToSelectedEdges(ctx, 0.0f);
                        }
                        ImGui::EndDisabled();
                        if (hasEdgeSelection) {
                            ImGui::TextDisabled("Selected edge crease: %.2f", selCrease);
                            ImGui::TextDisabled("Crease is applied when you bake Catmull-Clark (below).");
                        } else {
                            ImGui::TextDisabled("Select edge(s) to author crease sharpness.");
                        }
                        hasAnyOptions = true;
                    }

                    if (isFaceMode || isCombinedMode) {
                        if (hasAnyOptions) ImGui::Spacing();
                        ImGui::SliderFloat("Extrude Dist", &mesh_face_extrude_distance, -2.0f, 2.0f, "%.3f");
                        ImGui::SliderFloat("Inset Amount", &mesh_face_inset_amount, 0.02f, 0.98f, "%.2f");
                        ImGui::TextDisabled("Distance and inset scaling factor for mesh extrusion.");
                        hasAnyOptions = true;
                    }

                    if (!effectiveNodeName.empty()) {
                        auto& shading = ensureMeshShadingSettings(effectiveNodeName);
                        if (hasAnyOptions) ImGui::Spacing();
                        
                        bool autoSmooth = shading.auto_smooth;
                        if (ImGui::Checkbox("Auto Smooth Angle", &autoSmooth)) {
                            shading.auto_smooth = autoSmooth;
                            if (autoSmooth) {
                                shading.flat_shading = false;
                            }
                            applyMeshShadingSettings(ctx, effectiveNodeName);
                        }
                        if (shading.auto_smooth) {
                            if (ImGui::SliderFloat("Angle Threshold", &shading.auto_smooth_angle_degrees, 1.0f, 180.0f, "%.0f deg")) {
                                applyMeshShadingSettings(ctx, effectiveNodeName);
                            }
                        }
                        hasAnyOptions = true;
                    }

                    if (!hasAnyOptions) {
                        ImGui::TextDisabled("No active options for the current mode.");
                    }

                    UIWidgets::EndSection();
                }


            }

            UIWidgets::Divider();

            if (mesh_workspace_mode == SceneUI::MeshWorkspaceMode::Edit) {
                // 1. Lazy initialize base mesh cache if not present
                if (ctx.scene.base_mesh_cache.find(selectedNodeName) == ctx.scene.base_mesh_cache.end()) {
                    std::vector<std::shared_ptr<Triangle>> baseTriangles;
                    if (hasMeshEntries) {
                        baseTriangles.reserve(meshEntriesIt->second.size());
                        for (const auto& entry : meshEntriesIt->second) {
                            if (entry.second) {
                                baseTriangles.push_back(entry.second);
                            }
                        }
                    }
                    if (!baseTriangles.empty()) {
                        ctx.scene.base_mesh_cache[selectedNodeName] = baseTriangles;
                    }
                }

                // Get reference to stack
                auto& modifierStack = ctx.scene.mesh_modifiers[selectedNodeName];

                bool stackChanged = false;
                int applyModifierIndex = -1;

                auto replaceEvaluatedMesh = [&](const std::vector<std::shared_ptr<Triangle>>& meshToShow,
                                                const std::vector<std::shared_ptr<Triangle>>& selectionFallback,
                                                const std::string& message) {
                    std::vector<std::shared_ptr<Hittable>> remainingObjects;
                    remainingObjects.reserve(ctx.scene.world.objects.size() + meshToShow.size());
                    for (const auto& obj : ctx.scene.world.objects) {
                        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                        if (!tri || tri->getNodeName() != selectedNodeName) {
                            remainingObjects.push_back(obj);
                        }
                    }

                    for (const auto& tri : meshToShow) {
                        remainingObjects.push_back(tri);
                    }

                    ctx.scene.world.objects = remainingObjects;

                    if (!meshToShow.empty()) {
                        ctx.selection.selectObject(meshToShow[0], -1, selectedNodeName);
                    } else if (!selectionFallback.empty()) {
                        ctx.selection.selectObject(selectionFallback[0], -1, selectedNodeName);
                    }

                    if (mesh_overlay_settings.edit_mode &&
                        mesh_workspace_mode == SceneUI::MeshWorkspaceMode::Edit &&
                        !selectedNodeName.empty()) {
                        active_mesh_edit_object_name = selectedNodeName;
                        active_mesh_edit_object_ptr =
                            (!meshToShow.empty() ? meshToShow[0].get()
                                : (!selectionFallback.empty() ? selectionFallback[0].get() : nullptr));
                        editable_mesh_cache = EditableMeshCache{};
                        mesh_overlay_cache = MeshOverlayCache{};
                        mesh_edit_layer = MeshEditLayer{};
                    }

                    rebuildMeshCache(ctx.scene.world.objects);
                    if (mesh_overlay_settings.edit_mode &&
                        mesh_workspace_mode == SceneUI::MeshWorkspaceMode::Edit &&
                        !selectedNodeName.empty()) {
                        ensureEditableMeshCache(ctx, selectedNodeName);
                        ensureMeshEditLayer(ctx, selectedNodeName);
                    }
                    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                    ctx.renderer.resetCPUAccumulation();
                    if (ctx.backend_ptr) ctx.renderer.rebuildBackendGeometry(ctx.scene);

                    // Replacing the object's triangles is a STRUCTURAL change (triangle
                    // count changes), so the Rendered-mode backends (OptiX / Vulkan RT)
                    // must do a full geometry rebuild — not an in-place refit. Without
                    // bumping the geometry generation + rebuild flags, the GPU kept the
                    // OLD BLAS alongside the new one, so the pre- and post-bake meshes
                    // rendered on top of each other until the next edit forced a rebuild.
                    g_geometry_dirty = true;
                    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    g_vulkan_rebuild_pending = true;

                    addViewportMessage(message);
                    g_ProjectManager.markModified();
                };

                // Steady-state safety net: if the visible mesh's triangle count drifts
                // from a fresh full-stack evaluation, resync it. This MUST be skipped
                // during an interactive subdivision drag: the live preview deliberately
                // runs at a clamped level (so its count differs from the full evaluation),
                // and replaceEvaluatedMesh resets editable_mesh_cache + the element
                // selection — doing that every drag frame yanked the gizmo's target away,
                // so a cage vertex/edge/face would nudge once and snap back. The drag's own
                // path (applySelectedMeshElementTranslation → refreshEditableDisplayMeshFromBase,
                // then endInteractiveSubdivisionPreview on release) keeps the display in sync.
                if (!selectedNodeName.empty() &&
                    !isInteractiveSubdivisionPreviewActiveForObject(selectedNodeName) &&
                    ctx.scene.base_mesh_cache.find(selectedNodeName) != ctx.scene.base_mesh_cache.end() &&
                    !ctx.scene.base_mesh_cache[selectedNodeName].empty() &&
                    !modifierStack.modifiers.empty()) {
                    const auto evaluatedPreview = modifierStack.evaluate(ctx.scene.base_mesh_cache[selectedNodeName]);
                    const bool previewMismatch =
                        !hasMeshEntries ||
                        meshEntriesIt->second.size() != evaluatedPreview.size();
                    if (previewMismatch) {
                        replaceEvaluatedMesh(evaluatedPreview, ctx.scene.base_mesh_cache[selectedNodeName], "Modifier Preview Synced");
                        meshEntriesIt = mesh_cache.find(effectiveNodeName);
                    }
                }

                // 2. Draw existing modifiers
                for (size_t i = 0; i < modifierStack.modifiers.size(); ++i) {
                    auto& mod = modifierStack.modifiers[i];

                    ImGui::PushID(static_cast<int>(i));
                    if (UIWidgets::BeginSection(mod.name.c_str(), ImVec4(0.4f, 0.6f, 0.9f, 1.0f))) {

                        if (ImGui::Checkbox("Enabled", &mod.enabled)) stackChanged = true;

                        ImGui::SameLine();
                        if (UIWidgets::DangerButton("Delete")) {
                            modifierStack.modifiers.erase(modifierStack.modifiers.begin() + i);
                            stackChanged = true;
                            UIWidgets::EndSection();
                            ImGui::PopID();
                            break; // iterator invalidated
                        }

                        ImGui::SameLine();
                        if ((mod.type == MeshModifiers::ModifierType::FlatSubdivision ||
                             mod.type == MeshModifiers::ModifierType::SmoothSubdivision) &&
                            UIWidgets::PrimaryButton("Apply")) {
                            applyModifierIndex = static_cast<int>(i);
                        }

                        if (mod.type == MeshModifiers::ModifierType::FlatSubdivision || mod.type == MeshModifiers::ModifierType::SmoothSubdivision) {
                            if (ImGui::SliderInt("Levels", &mod.levels, 1, 10)) stackChanged = true;
                        }

                        if (mod.type == MeshModifiers::ModifierType::SmoothSubdivision) {
                            if (ImGui::SliderFloat("Smooth Weight", &mod.smoothAngle, 0.0f, 1.0f)) stackChanged = true;
                        }

                        UIWidgets::EndSection();
                    }
                    ImGui::PopID();
                }

                UIWidgets::Divider();

                // 3. Add new modifier
                UIWidgets::ColoredHeader("Add Modifier", ImVec4(0.5f, 0.8f, 1.0f, 1.0f));
                if (UIWidgets::SecondaryButton("Flat Subdivision", ImVec2(-1, 0))) {
                    MeshModifiers::ModifierData newMod;
                    newMod.name = "Flat Subdivision";
                    newMod.type = MeshModifiers::ModifierType::FlatSubdivision;
                    modifierStack.modifiers.push_back(newMod);
                    stackChanged = true;
                }
                if (UIWidgets::SecondaryButton("Smooth Subdivision", ImVec2(-1, 0))) {
                    MeshModifiers::ModifierData newMod;
                    newMod.name = "Smooth Subdivision";
                    newMod.type = MeshModifiers::ModifierType::SmoothSubdivision;
                    modifierStack.modifiers.push_back(newMod);
                    stackChanged = true;
                }

                // True Catmull-Clark as a DESTRUCTIVE bake (not a live modifier). Editing a
                // CC limit surface live (gizmo move/scale while previewing) proved fragile,
                // so CC is applied on demand: it subdivides the cage (with authored creases)
                // and replaces the editable base with the result. After this you edit the
                // baked dense mesh directly — stable, no per-frame re-evaluation.
                UIWidgets::Divider();
                UIWidgets::ColoredHeader("Catmull-Clark (Bake)", ImVec4(0.55f, 0.8f, 1.0f, 1.0f));
                ImGui::SliderInt("CC Levels##ccbake", &mesh_cc_bake_levels, 1, 10);
                if (UIWidgets::SecondaryButton("Apply Catmull-Clark", ImVec2(-1, 0))) {
                    // Bake from the cage (base) so position-keyed creases line up; fall back
                    // to the current visible mesh if no base was captured.
                    std::vector<std::shared_ptr<Triangle>> src;
                    auto baseSrcIt = ctx.scene.base_mesh_cache.find(selectedNodeName);
                    if (baseSrcIt != ctx.scene.base_mesh_cache.end() && !baseSrcIt->second.empty()) {
                        src = baseSrcIt->second;
                    } else if (hasMeshEntries) {
                        for (const auto& entry : meshEntriesIt->second) {
                            if (entry.second) src.push_back(entry.second);
                        }
                    }
                    if (!src.empty()) {
                        MeshModifiers::EdgeCreaseFn creaseFn;
                        if (!modifierStack.edgeCreases.empty()) {
                            creaseFn = [&modifierStack](const Vec3& a, const Vec3& b) {
                                return modifierStack.getEdgeCrease(a, b);
                            };
                        }
                        auto ccMesh = MeshModifiers::CatmullClarkSubD(src, mesh_cc_bake_levels, creaseFn);
                        if (!ccMesh.empty()) {
                            // Baked CC becomes the new editable base; drop subdivision
                            // modifiers (they would re-subdivide an already-CC cage) and
                            // the now-stale crease keys.
                            ctx.scene.base_mesh_cache[selectedNodeName] = ccMesh;
                            modifierStack.modifiers.erase(
                                std::remove_if(modifierStack.modifiers.begin(), modifierStack.modifiers.end(),
                                    [](const MeshModifiers::ModifierData& m) {
                                        return m.type == MeshModifiers::ModifierType::FlatSubdivision ||
                                               m.type == MeshModifiers::ModifierType::SmoothSubdivision;
                                    }),
                                modifierStack.modifiers.end());
                            modifierStack.edgeCreases.clear();
                            const auto displayMesh = modifierStack.modifiers.empty()
                                ? ccMesh
                                : modifierStack.evaluate(ccMesh);
                            replaceEvaluatedMesh(displayMesh, ccMesh, "Catmull-Clark Baked");
                        }
                    }
                }
                ImGui::TextDisabled("Bakes a true Catmull-Clark surface from the cage (uses edge creases).");

                if (applyModifierIndex >= 0) {
                    SCENE_LOG_INFO("Applying Modifier Stack up to index " + std::to_string(applyModifierIndex) + " for '" + selectedNodeName + "'...");

                    const auto& baseMesh = ctx.scene.base_mesh_cache[selectedNodeName];
                    MeshModifiers::ModifierStack bakedStack;
                    bakedStack.modifiers.assign(
                        modifierStack.modifiers.begin(),
                        modifierStack.modifiers.begin() + applyModifierIndex + 1);

                    auto bakedMesh = bakedStack.evaluate(baseMesh);
                    ctx.scene.base_mesh_cache[selectedNodeName] = bakedMesh;
                    modifierStack.modifiers.erase(
                        modifierStack.modifiers.begin(),
                        modifierStack.modifiers.begin() + applyModifierIndex + 1);

                    auto displayMesh = modifierStack.modifiers.empty()
                        ? bakedMesh
                        : modifierStack.evaluate(ctx.scene.base_mesh_cache[selectedNodeName]);

                    replaceEvaluatedMesh(displayMesh, ctx.scene.base_mesh_cache[selectedNodeName], "Modifier Applied");
                }

                // 4. Evaluate stack if changed
                if (stackChanged) {
                    SCENE_LOG_INFO("Evaluating Modifier Stack for '" + selectedNodeName + "'...");

                    const auto& baseMesh = ctx.scene.base_mesh_cache[selectedNodeName];
                    auto newMesh = modifierStack.evaluate(baseMesh);

                    SCENE_LOG_INFO("Evaluated mesh '" + selectedNodeName + "': " + std::to_string(baseMesh.size()) + " -> " + std::to_string(newMesh.size()) + " triangles.");
                    replaceEvaluatedMesh(newMesh, baseMesh, "Modifiers Updated");
                }

                UIWidgets::Divider();
            }


        }

        UIWidgets::EndSection();
    }
    UIWidgets::PopControlSurfaceStyle();
}

void SceneUI::drawSculptPanel(UIContext& ctx) {
    UIWidgets::PushControlSurfaceStyle(ImVec4(1.0f, 0.58f, 0.34f, 1.0f));
    if (UIWidgets::BeginSection("Sculpt Mode", ImVec4(1.0f, 0.58f, 0.34f, 1.0f))) {
        bool hasSelection = (ctx.selection.selected.type == SelectableType::Object &&
                             ctx.selection.selected.object != nullptr);

        if (!hasSelection) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.6f, 1.0f), "Please select a mesh object or terrain.");
            UIWidgets::EndSection();
            UIWidgets::PopControlSurfaceStyle();
            return;
        }

        const std::string selectedNodeName = ctx.selection.selected.object->getNodeName();

        // Auto-activate sculpt workspace if selection is new/changed or not yet enabled
        if (!sculpt_mode_state.enabled || sculpt_mode_state.active_target_name != selectedNodeName) {
            activateSculptWorkspace(ctx);
        }

        std::shared_ptr<Triangle> mesh_triangle = resolvePaintMesh(ctx);
        if (!mesh_triangle) {
            auto adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter());
            mesh_triangle = adapter ? adapter->getTriangle() : nullptr;
        }
        if (!mesh_triangle &&
            ctx.selection.selected.type == SelectableType::Object &&
            ctx.selection.selected.object) {
            mesh_triangle = ctx.selection.selected.object;
        }

        drawSculptBrushControls(ctx, mesh_triangle, false);

        UIWidgets::Divider();
        ImGui::TextDisabled("Subdivision modifiers are available only in the Modeling tab.");
        ImGui::TextDisabled("Sculpt stays focused on brush-based surface editing.");

        UIWidgets::EndSection();
    }
    UIWidgets::PopControlSurfaceStyle();
}

void SceneUI::drawPaintPanel(UIContext& ctx) {
    UIWidgets::PushControlSurfaceStyle(ImVec4(0.32f, 0.88f, 0.82f, 1.0f));
    std::shared_ptr<Triangle> mesh_triangle = resolvePaintMesh(ctx);
    TerrainObject* terrain = nullptr;
    if (!mesh_triangle) {
        terrain = resolvePaintTerrain(*this);
    }

    const std::string next_target_name = mesh_triangle ? mesh_triangle->getNodeName() : std::string();
    if (!next_target_name.empty() && paint_mode_state.active_target_name != next_target_name) {
        paint_mode_state.active_material_slot = 0;
        paint_mode_state.stroke = Paint::PaintStroke{};
    }

    if (mesh_triangle) {
        paint_mode_state.enabled = true;
    } else if (terrain && terrain->splatMap) {
        paint_mode_state.enabled = true;
        auto terrain_adapter = std::dynamic_pointer_cast<Paint::TerrainPaintAdapter>(paint_mode_state.getAdapter());
        if (!terrain_adapter || terrain_adapter->getTerrain() != terrain) {
            paint_mode_state.setAdapter(std::make_shared<Paint::TerrainPaintAdapter>(terrain));
        }
    } else {
        paint_mode_state.clearAdapter();
        paint_mode_state.enabled = false;
    }

    if (!UIWidgets::BeginSection("Paint Mode", ImVec4(0.95f, 0.55f, 0.30f, 1.0f))) {
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    if (!terrain && !mesh_triangle) {
        ImGui::TextDisabled("Paint target: none");
        ImGui::TextDisabled("Select a terrain for splat paint or a regular mesh for texture-set paint.");
        UIWidgets::EndSection();
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    // Stylized Target Info Panel
    if (mesh_triangle) {
        ImGui::TextColored(ImVec4(0.28f, 0.90f, 0.82f, 1.0f), "Target: %s", mesh_triangle->getNodeName().empty() ? "(unknown)" : mesh_triangle->getNodeName().c_str());
        ImGui::TextDisabled("Type: Mesh     |  Mode: Painting");
    } else if (terrain) {
        ImGui::TextColored(ImVec4(0.28f, 0.90f, 0.82f, 1.0f), "Target: %s", terrain->name.empty() ? "(unknown terrain)" : terrain->name.c_str());
        ImGui::TextDisabled("Type: Terrain  |  Mode: Painting");
    }
    UIWidgets::Divider();

    if (mesh_triangle) {
        drawMeshPaintPanel(ctx, mesh_triangle);
    } else if (terrain) {
        drawTerrainPaintPanel(ctx, terrain);
    }
    UIWidgets::EndSection();
    UIWidgets::PopControlSurfaceStyle();
}

void SceneUI::drawTerrainPaintPanel(UIContext& ctx, TerrainObject* terrain) {
    if (!terrain) {
        return;
    }

    if (terrain->layers.empty() || !terrain->splatMap) {
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.45f, 1.0f), "Terrain paint needs initialized layers and a splat map.");
        if (UIWidgets::PrimaryButton("Initialize Terrain Layers", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
            TerrainManager::getInstance().initLayers(terrain);
            paint_mode_state.setAdapter(std::make_shared<Paint::TerrainPaintAdapter>(terrain));
            ctx.renderer.resetCPUAccumulation();
            if (ctx.backend_ptr) {
                ctx.renderer.updateBackendMaterials(ctx.scene);
                ctx.backend_ptr->resetAccumulation();
            }
        }
        return;
    }

    Paint::PaintSurfaceAdapterPtr adapter = paint_mode_state.getAdapter();
    paint_mode_state.syncLayersFromAdapter();
    paint_mode_state.clampActiveLayer();


    UIWidgets::ColoredHeader("Layers", ImVec4(0.95f, 0.65f, 0.35f, 1.0f));

    if (!paint_mode_state.ui_layers.empty()) {
        std::string active_layer_label = paint_mode_state.ui_layers[paint_mode_state.active_layer_index].name;
        active_layer_label += " (Channel ";
        active_layer_label += std::to_string(paint_mode_state.active_layer_index);
        active_layer_label += ")";

        if (ImGui::BeginCombo("Active Layer", active_layer_label.c_str())) {
            for (int i = 0; i < static_cast<int>(paint_mode_state.ui_layers.size()); ++i) {
                const bool selected = (paint_mode_state.active_layer_index == i);
                std::string item_label = paint_mode_state.ui_layers[i].name;
                item_label += " (Channel ";
                item_label += std::to_string(i);
                item_label += ")";
                if (ImGui::Selectable(item_label.c_str(), selected)) {
                    paint_mode_state.active_layer_index = i;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }

    if (terrain->splatMap && terrain->splatMap->is_loaded()) {
        if (UIWidgets::IconActionButton("FillActiveLayerMask", UIWidgets::IconType::FillTool, "Fill Active Layer",
                                        false, ImVec4(1.0f, 0.70f, 0.45f, 1.0f),
                                        ImVec2(UIWidgets::GetInspectorActionWidth(), 34.0f),
                                        "Fill the active terrain mask channel")) {
            auto& pixels = terrain->splatMap->pixels;
            for (auto& p : pixels) {
                if (paint_mode_state.active_layer_index == 0) p.r = 255;
                else if (paint_mode_state.active_layer_index == 1) p.g = 255;
                else if (paint_mode_state.active_layer_index == 2) p.b = 255;
                else if (paint_mode_state.active_layer_index == 3) p.a = 255;
            }
            terrain->splatMap->updateGPU();
            ctx.renderer.resetCPUAccumulation();
            if (ctx.backend_ptr) {
                ctx.renderer.updateBackendMaterials(ctx.scene);
                ctx.backend_ptr->resetAccumulation();
            }
            SCENE_LOG_INFO("Filled terrain paint mask for active layer on: " + terrain->name);
        }
    }

    UIWidgets::Divider();
    UIWidgets::ColoredHeader("Brush", ImVec4(1.0f, 0.70f, 0.45f, 1.0f));
    ImGui::SliderFloat("Radius", &paint_mode_state.brush.radius, 1.0f, 200.0f, "%.1f m");
    ImGui::SliderFloat("Strength", &paint_mode_state.brush.strength, 0.01f, 10.0f, "%.2f");
    ImGui::SliderFloat("Falloff", &paint_mode_state.brush.falloff, 0.0f, 1.0f, "%.2f");
    if (!paint_mode_state.compact_ui) {
        ImGui::SliderFloat("Spacing", &paint_mode_state.brush.spacing, 0.01f, 1.0f, "%.2f");
        ImGui::SliderFloat("Flow", &paint_mode_state.brush.flow, 0.1f, 2.0f, "%.2f");
    }
    ImGui::Checkbox("Show Brush Preview", &paint_mode_state.brush.show_preview);

    ImGui::Spacing();
    ImGui::TextDisabled("Viewport painting stays on the terrain brush path for now.");
    syncPaintBrushToTerrain(*this, terrain);

    UIWidgets::Divider();
    UIWidgets::ColoredHeader("Mask & Splat Tools", ImVec4(0.85f, 0.68f, 0.40f, 1.0f));

    ImGui::PushItemWidth(160.0f);
    SceneUI::DrawSmartFloat("mhmin_paint", "Height Start", &terrain->am_height_min, 0.0f, 500.0f, "%.1f", false, nullptr, 12);
    SceneUI::DrawSmartFloat("mhmax_paint", "Height End", &terrain->am_height_max, 0.0f, 500.0f, "%.1f", false, nullptr, 12);
    SceneUI::DrawSmartFloat("mslope_paint", "Slope Steep", &terrain->am_slope, 1.0f, 20.0f, "%.1f", false, nullptr, 12);
    SceneUI::DrawSmartFloat("mflow_paint", "Flow Thresh", &terrain->am_flow_threshold, 1.0f, 500.0f, "%.0f", false, nullptr, 12);
    ImGui::PopItemWidth();

    if (UIWidgets::PrimaryButton("Generate Mask##terrain_paint", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
        TerrainManager::getInstance().autoMask(terrain, 0.0f, 0.0f, terrain->am_height_min, terrain->am_height_max, terrain->am_slope);
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            ctx.renderer.updateBackendMaterials(ctx.scene);
            ctx.backend_ptr->resetAccumulation();
        }
        SCENE_LOG_INFO("Auto-mask generated for: " + terrain->name);
    }

    if (UIWidgets::IconActionButton("ImportSplatMap", UIWidgets::IconType::Assets, "Import Splat Map",
                                    false, ImVec4(0.92f, 0.72f, 0.42f, 1.0f),
                                    ImVec2(UIWidgets::GetInspectorActionWidth(), 34.0f),
                                    "Import a terrain splat map")) {
        std::string path = SceneUI::openFileDialogW(L"Image Files\0*.png;*.jpg;*.jpeg;*.bmp\0All Files\0*.*\0");
        if (!path.empty()) {
            TerrainManager::getInstance().importSplatMap(terrain, path);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.backend_ptr) {
                ctx.renderer.updateBackendMaterials(ctx.scene);
                ctx.backend_ptr->resetAccumulation();
            }
        }
    }

    static float bake_flow_threshold = 25.0f;
    ImGui::PushItemWidth(160.0f);
    ImGui::DragFloat("Flow Bake Threshold##terrain_paint", &bake_flow_threshold, 1.0f, 1.0f, 500.0f, "%.0f");
    ImGui::PopItemWidth();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum flow accumulation written to alpha.");

    if (UIWidgets::PrimaryButton("Bake Flow to Alpha##terrain_paint", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.7f, 30))) {
        if (terrain->splatMap && terrain->splatMap->is_loaded() && !terrain->flowMap.empty()) {
            int w = terrain->heightmap.width;
            int h = terrain->heightmap.height;
            int sw = terrain->splatMap->width;
            int sh = terrain->splatMap->height;

            for (int y = 0; y < sh; y++) {
                for (int x = 0; x < sw; x++) {
                    float u = (float)x / (float)(sw > 1 ? sw - 1 : 1);
                    float v = (float)y / (float)(sh > 1 ? sh - 1 : 1);
                    int fx = std::clamp((int)(u * (w - 1) + 0.5f), 0, w - 1);
                    int fy = std::clamp((int)(v * (h - 1) + 0.5f), 0, h - 1);

                    float flowVal = terrain->flowMap[fy * w + fx];
                    float flowNorm = fmaxf(0.0f, flowVal - terrain->am_flow_threshold);
                    float final_A = 1.0f - expf(-flowNorm * 0.4f);
                    final_A = std::clamp(final_A, 0.0f, 0.98f);

                    terrain->splatMap->pixels[y * sw + x].a = (uint8_t)(final_A * 255.0f);
                }
            }
            terrain->splatMap->updateGPU();
            ctx.renderer.resetCPUAccumulation();
            if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
            SCENE_LOG_INFO("Flow baked to splat alpha for: " + terrain->name);
        } else {
            SCENE_LOG_WARN("Please run erosion first to generate a flow map.");
        }
    }

    if (ImGui::Button("Export Splat Map##terrain_paint", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
        if (terrain->splatMap && !terrain->splatMap->pixels.empty()) {
            std::string path = SceneUI::saveFileDialogW(L"PNG Files\0*.png\0", L"png");
            if (!path.empty()) {
                TerrainManager::getInstance().exportSplatMap(terrain, path);
                SCENE_LOG_INFO("Splat map exported to: " + path);
            }
        }
    }
}

void SceneUI::drawMeshPaintPanel(UIContext& ctx, const std::shared_ptr<Triangle>& meshTriangle) {
    if (!meshTriangle) {
        ImGui::TextDisabled("Mesh target unavailable.");
        return;
    }

    const std::string obj_name = meshTriangle->getNodeName();
    auto mesh_cache_it = mesh_cache.find(obj_name);
    auto slots_it = material_slots_cache.find(obj_name);
    if (mesh_cache_it == mesh_cache.end() || slots_it == material_slots_cache.end() || slots_it->second.empty()) {
        ImGui::TextDisabled("Mesh slot cache unavailable.");
        return;
    }

    auto& mesh_entries = mesh_cache_it->second;
    auto& slot_ids = slots_it->second;
    paint_mode_state.active_material_slot = std::clamp(
        paint_mode_state.active_material_slot, 0, static_cast<int>(slot_ids.size()) - 1);

    uint16_t slot_material_id = slot_ids[paint_mode_state.active_material_slot];
    std::shared_ptr<Triangle> slot_triangle = findMeshTriangleForMaterial(mesh_entries, slot_material_id);
    if (!slot_triangle) {
        slot_triangle = meshTriangle;
        slot_material_id = slot_triangle->getMaterialID();
    }

    auto adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter());
    if (!adapter || adapter->getTriangle() != slot_triangle) {
        paint_mode_state.setAdapter(std::make_shared<Paint::MeshPaintAdapter>(&ctx.scene, slot_triangle));
        adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter());
    }

    if (!adapter) {
        ImGui::TextDisabled("Mesh adapter unavailable.");
        return;
    }

    auto syncPaintMaterials = [&]() {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            ctx.renderer.updateBackendMaterials(ctx.scene, ctx.backend_ptr);
            ctx.backend_ptr->resetAccumulation();
        }
        if (g_viewport_backend && g_viewport_backend.get() != ctx.backend_ptr) {
            ctx.renderer.updateBackendMaterials(ctx.scene, g_viewport_backend.get());
        }
    };

    ImGui::Text("Material: %s", adapter->getMaterialName().empty() ? "(unnamed)" : adapter->getMaterialName().c_str());
    UIWidgets::Divider();

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 1: TARGET & LAYERS (DCC Style)
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Target & Layers", ImVec4(0.35f, 0.78f, 1.0f, 1.0f))) {
        ImGui::TextDisabled("Material Slot Configuration");
        if (ImGui::BeginCombo("Material Slot", ("Slot " + std::to_string(paint_mode_state.active_material_slot)).c_str())) {
            for (int i = 0; i < static_cast<int>(slot_ids.size()); ++i) {
                const bool selected = (paint_mode_state.active_material_slot == i);
                const std::string slot_name = MaterialManager::getInstance().getMaterialName(slot_ids[i]);
                const std::string item_label =
                    "Slot " + std::to_string(i) + ": " + (slot_name.empty() ? "[Unnamed]" : slot_name);
                if (ImGui::Selectable(item_label.c_str(), selected)) {
                    paint_mode_state.active_material_slot = i;
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        int binding_mode = static_cast<int>(paint_mode_state.material_binding_mode);
        const char* binding_labels[] = { "Use Current Material", "Make Unique For Paint" };
        if (ImGui::Combo("Binding Mode", &binding_mode, binding_labels, IM_ARRAYSIZE(binding_labels))) {
            paint_mode_state.material_binding_mode = static_cast<Paint::MaterialBindingMode>(binding_mode);
        }

        if (paint_mode_state.material_binding_mode == Paint::MaterialBindingMode::MakeUniqueForPaint) {
            if (UIWidgets::SecondaryButton("Make Unique Now", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
                const uint16_t unique_material_id = makeObjectSlotPaintMaterialUnique(
                    ctx,
                    obj_name,
                    paint_mode_state.active_material_slot,
                    slot_material_id,
                    mesh_entries,
                    slot_ids);
                slot_triangle = findMeshTriangleForMaterial(mesh_entries, unique_material_id);
                if (slot_triangle) {
                    paint_mode_state.setAdapter(std::make_shared<Paint::MeshPaintAdapter>(&ctx.scene, slot_triangle));
                    adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter());
                }
                syncPaintMaterials();
                g_ProjectManager.markModified();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextDisabled("Texture Set & Channels");
        static const int resolutions[] = { 512, 1024, 2048, 4096 };
        auto* texture_set = adapter->getTextureSet();

        if (!texture_set || !texture_set->initialized) {
            Paint::PaintLayerStack* loaded_stack = adapter->getLayerStack();
            if (loaded_stack && !loaded_stack->empty() && loaded_stack->width() > 0) {
                const int res = loaded_stack->width();
                adapter->ensureTextureSet(res);
                for (int i = 0; i < static_cast<int>(Paint::kPaintChannelCount); ++i) {
                    const Paint::PaintChannel channel = static_cast<Paint::PaintChannel>(i);
                    if (loaded_stack->anyLayerHasPixels(channel)) {
                        adapter->assignTextureToChannel(channel);
                    }
                }
                texture_set = adapter->getTextureSet();
                if (texture_set && texture_set->initialized) {
                    loaded_stack->flattenInto(*texture_set);
                    adapter->bindTextureSetToMaterial();
                    for (int ch = 0; ch < static_cast<int>(Paint::kPaintChannelCount); ++ch) {
                        if (auto texture = texture_set->getTexture(static_cast<Paint::PaintChannel>(ch))) {
                            texture->markVulkanDirtyFull();
                        }
                    }
                    syncPaintMaterials();
                    ctx.start_render = true;
                }
            }
        }

        const bool has_set = texture_set != nullptr && texture_set->initialized;
        if (!isMeshPaintUiChannelEnabled(paint_mode_state.active_channel)) {
            paint_mode_state.active_channel = Paint::PaintChannel::BaseColor;
        }
        syncHeightMaskPaintToggles(paint_mode_state);
        const std::shared_ptr<Texture> active_channel_texture =
            texture_set ? texture_set->getTexture(paint_mode_state.active_channel) : nullptr;
        const int effective_resolution = active_channel_texture
            ? std::max(active_channel_texture->width, active_channel_texture->height)
            : (texture_set && texture_set->resolution > 0 ? texture_set->resolution : paint_mode_state.requested_texture_resolution);
        
        int current_resolution_index = 1;
        for (int i = 0; i < 4; ++i) {
            if (resolutions[i] == paint_mode_state.requested_texture_resolution) {
                current_resolution_index = i;
                break;
            }
        }
        const char* resolution_labels[] = { "512", "1024", "2048", "4096" };
        if (ImGui::Combo("Resolution", &current_resolution_index, resolution_labels, IM_ARRAYSIZE(resolution_labels))) {
            paint_mode_state.requested_texture_resolution = resolutions[current_resolution_index];
        }

        const Paint::PaintChannel ui_channels[] = {
            Paint::PaintChannel::BaseColor,
            Paint::PaintChannel::Normal,
            Paint::PaintChannel::Roughness,
            Paint::PaintChannel::Metallic,
            Paint::PaintChannel::Emission,
            Paint::PaintChannel::Mask,
            Paint::PaintChannel::Transmission
        };
        const char* channel_labels[] = { "Base Color", "Normal", "Roughness", "Metallic", "Emission", "Height Mask", "Transmission" };
        int channel_index = 0;
        for (int i = 0; i < IM_ARRAYSIZE(ui_channels); ++i) {
            if (ui_channels[i] == paint_mode_state.active_channel) {
                channel_index = i;
                break;
            }
        }
        if (ImGui::Combo("Active Channel", &channel_index, channel_labels, IM_ARRAYSIZE(channel_labels))) {
            const Paint::PaintChannel previous_channel = paint_mode_state.active_channel;
            paint_mode_state.active_channel = ui_channels[channel_index];
            if (previous_channel == Paint::PaintChannel::Mask &&
                paint_mode_state.active_channel != Paint::PaintChannel::Mask) {
                paint_mode_state.brush.write_height_mask = false;
            }
            syncHeightMaskPaintToggles(paint_mode_state);
        }

        ImGui::TextDisabled("Brush Multi-Channel Binding");
        for (int i = 0; i < static_cast<int>(Paint::kPaintChannelCount); ++i) {
            const Paint::PaintChannel channel = static_cast<Paint::PaintChannel>(i);
            if (!isMeshPaintUiChannelEnabled(channel)) {
                paint_mode_state.linked_channels[static_cast<size_t>(i)] = false;
                continue;
            }
            if (channel == paint_mode_state.active_channel) {
                continue;
            }
            bool enabled = paint_mode_state.linked_channels[static_cast<size_t>(i)];
            if (ImGui::Checkbox(paintChannelDisplayName(channel), &enabled)) {
                paint_mode_state.linked_channels[static_cast<size_t>(i)] = enabled;
            }
            if (channel == Paint::PaintChannel::BaseColor ||
                channel == Paint::PaintChannel::Roughness ||
                channel == Paint::PaintChannel::Emission) {
                ImGui::SameLine();
            }
        }

        const char* create_button_label = !has_set
            ? "Create Selected Channels"
            : (active_channel_texture ? "Ensure Selected Channels" : "Create Selected Channels");
        if (UIWidgets::PrimaryButton(create_button_label, ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
            uint16_t paint_material_id = slot_material_id;
            if (paint_mode_state.material_binding_mode == Paint::MaterialBindingMode::MakeUniqueForPaint) {
                paint_material_id = makeObjectSlotPaintMaterialUnique(
                    ctx,
                    obj_name,
                    paint_mode_state.active_material_slot,
                    slot_material_id,
                    mesh_entries,
                    slot_ids);
            }

            slot_triangle = findMeshTriangleForMaterial(mesh_entries, paint_material_id);
            if (slot_triangle) {
                paint_mode_state.setAdapter(std::make_shared<Paint::MeshPaintAdapter>(&ctx.scene, slot_triangle));
                adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter());
            }
            if (adapter) {
                adapter->ensureTextureSet(paint_mode_state.requested_texture_resolution);
                std::vector<Paint::PaintChannel> channels_to_create = getSelectedMaterialBrushChannels(paint_mode_state);
                if (channels_to_create.empty()) {
                    channels_to_create.push_back(paint_mode_state.active_channel);
                }
                for (Paint::PaintChannel channel : channels_to_create) {
                    adapter->assignTextureToChannel(channel);
                }
                syncPaintMaterials();
                g_ProjectManager.markModified();
            }
        }

        if (texture_set && texture_set->initialized && effective_resolution != paint_mode_state.requested_texture_resolution) {
            if (UIWidgets::SecondaryButton("Resize Texture Set", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
                if (adapter->resizeTextureSet(paint_mode_state.requested_texture_resolution)) {
                    texture_set = adapter->getTextureSet();
                    syncPaintMaterials();
                    g_ProjectManager.markModified();
                }
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Katman Paneli (Photoshop / Substance Painter tarzı dikey entegrasyon)
        drawPaintLayerPanel(ctx, adapter.get());

        UIWidgets::EndSection();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 2: BRUSH DETAILS (DCC Style)
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Brush Details", ImVec4(1.0f, 0.70f, 0.45f, 1.0f))) {
        drawPaintBrushControls(ctx, slot_triangle, false);
        UIWidgets::EndSection();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 3: BAKE & UTILITIES (DCC Style)
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Bake & Utilities", ImVec4(0.85f, 0.68f, 0.40f, 1.0f))) {
        auto* texture_set = adapter->getTextureSet();
        const bool has_height_mask = texture_set && texture_set->getTexture(Paint::PaintChannel::Mask) != nullptr;

        ImGui::TextDisabled("Height Map To Normal Map");
        ImGui::SliderFloat("Normal Strength##bake", &paint_mode_state.height_to_normal_strength, 0.1f, 32.0f, "%.2f");
        ImGui::Checkbox("Clear Height After Bake##bake", &paint_mode_state.clear_height_after_bake);

        ImGui::BeginDisabled(!has_height_mask);
        if (UIWidgets::SecondaryButton("Generate Normal From Height", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
            adapter->ensureTextureSet(paint_mode_state.requested_texture_resolution);
            adapter->assignTextureToChannel(Paint::PaintChannel::Normal);
            Paint::PaintTextureSet* refreshed_set = adapter->getTextureSet();
            std::shared_ptr<Texture> normal_texture = refreshed_set ? refreshed_set->getTexture(Paint::PaintChannel::Normal) : nullptr;
            const std::vector<CompactVec4> before_pixels = normal_texture ? normal_texture->pixels : std::vector<CompactVec4>{};
            if (adapter->generateNormalFromHeight(paint_mode_state.height_to_normal_strength)) {
                refreshed_set = adapter->getTextureSet();
                normal_texture = refreshed_set ? refreshed_set->getTexture(Paint::PaintChannel::Normal) : nullptr;
                if (normal_texture) {
                    history.record(std::make_unique<PaintTextureCommand>(
                        adapter->getNodeName(),
                        adapter->getMaterialID(),
                        normal_texture,
                        before_pixels,
                        normal_texture->pixels));
                }
                syncPaintMaterials();
                g_ProjectManager.markModified();
            }
        }
        if (UIWidgets::SecondaryButton("Bake Current Height To Normal", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
            adapter->ensureTextureSet(paint_mode_state.requested_texture_resolution);
            adapter->assignTextureToChannel(Paint::PaintChannel::Normal);
            Paint::PaintTextureSet* refreshed_set = adapter->getTextureSet();
            std::shared_ptr<Texture> normal_texture = refreshed_set ? refreshed_set->getTexture(Paint::PaintChannel::Normal) : nullptr;
            std::shared_ptr<Texture> height_texture = refreshed_set ? refreshed_set->getTexture(Paint::PaintChannel::Mask) : nullptr;
            const std::vector<CompactVec4> normal_before_pixels = normal_texture ? normal_texture->pixels : std::vector<CompactVec4>{};
            const std::vector<CompactVec4> height_before_pixels = height_texture ? height_texture->pixels : std::vector<CompactVec4>{};
            if (adapter->bakeHeightIntoNormal(
                    paint_mode_state.height_to_normal_strength,
                    paint_mode_state.clear_height_after_bake)) {
                refreshed_set = adapter->getTextureSet();
                normal_texture = refreshed_set ? refreshed_set->getTexture(Paint::PaintChannel::Normal) : nullptr;
                height_texture = refreshed_set ? refreshed_set->getTexture(Paint::PaintChannel::Mask) : nullptr;
                auto composite = std::make_unique<CompositeSceneCommand>(
                    "Bake Height To Normal " + adapter->getNodeName());
                if (normal_texture) {
                    composite->add(std::make_unique<PaintTextureCommand>(
                        adapter->getNodeName(),
                        adapter->getMaterialID(),
                        normal_texture,
                        normal_before_pixels,
                        normal_texture->pixels));
                }
                if (paint_mode_state.clear_height_after_bake && height_texture) {
                    composite->add(std::make_unique<PaintTextureCommand>(
                        adapter->getNodeName(),
                        adapter->getMaterialID(),
                        height_texture,
                        height_before_pixels,
                        height_texture->pixels));
                }
                if (!composite->empty()) {
                    history.record(std::move(composite));
                }
                syncPaintMaterials();
                g_ProjectManager.markModified();
            }
        }
        ImGui::EndDisabled();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextDisabled("Export Channel Data");
        if (texture_set) {
            drawPaintChannelTextureSlots(ctx, adapter.get());

            if (UIWidgets::SecondaryButton("Export All Channels", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
                const std::string folder_hint = SceneUI::saveFileDialogW(L"PNG Files\0*.png\0", L"png");
                if (!folder_hint.empty()) {
                    const std::filesystem::path folder_path = utf8PathFromString(folder_hint).parent_path() /
                        (texture_set->target_node_name.empty() ? "paint_export" : (texture_set->target_node_name + "_paint"));
                    if (exportTextureSetChannels(*texture_set, folder_path)) {
                        SCENE_LOG_INFO("Exported channels to: " + folder_path.string());
                    }
                }
            }

            Paint::PaintLayerStack* export_stack = adapter->getLayerStack();
            if (export_stack && export_stack->layerCount() > 1) {
                if (UIWidgets::SecondaryButton("Bake Layers & Export", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
                    const std::string folder_hint = SceneUI::saveFileDialogW(L"PNG Files\0*.png\0", L"png");
                    if (!folder_hint.empty()) {
                        export_stack->flattenInto(*texture_set);
                        adapter->bindTextureSetToMaterial();
                        const std::filesystem::path folder_path = utf8PathFromString(folder_hint).parent_path() /
                            (texture_set->target_node_name.empty() ? "paint_baked" : (texture_set->target_node_name + "_baked"));
                        if (exportTextureSetChannels(*texture_set, folder_path)) {
                            SCENE_LOG_INFO("Baked and exported layers to: " + folder_path.string());
                        }
                    }
                }
            }

            if (UIWidgets::SecondaryButton("Restore Original Material", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
                if (adapter->restoreOriginalMaterialTextures()) {
                    adapter->releaseLayerStackFromScene();
                    paint_mode_state.bindLayerStack(nullptr);
                    paint_mode_state.active_layer_index = 0;
                    paint_mode_state.active_layer_id = 0;
                    releaseLayerThumbnails();
                    texture_set = adapter->getTextureSet();
                    syncPaintMaterials();
                    g_ProjectManager.markModified();
                }
            }
        }
        UIWidgets::EndSection();
    }
}

// ======================== Paint Layer Panel ========================

// ---------- Layer Thumbnail helpers ----------

void SceneUI::releaseLayerThumbnails() {
    for (auto& entry : layer_thumb_cache) entry.release();
    layer_thumb_cache.clear();
}

namespace {

// Fast hash of pixel data for change detection (sample a few pixels).
uint64_t hashLayerChannel(const std::vector<CompactVec4>& pixels) {
    if (pixels.empty()) return 0;
    uint64_t h = pixels.size();
    const size_t step = std::max<size_t>(1, pixels.size() / 64);
    for (size_t i = 0; i < pixels.size(); i += step) {
        const auto& p = pixels[i];
        h ^= (static_cast<uint64_t>(p.r) << 24 | static_cast<uint64_t>(p.g) << 16 |
               static_cast<uint64_t>(p.b) << 8 | static_cast<uint64_t>(p.a)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

// Downsample layer channel pixels to a small RGBA buffer (thumb_size x thumb_size).
void downsampleToThumb(const std::vector<CompactVec4>& src, int src_w, int src_h,
                       std::vector<uint8_t>& dst_rgba, int thumb_size) {
    dst_rgba.resize(static_cast<size_t>(thumb_size) * thumb_size * 4);
    if (src.empty() || src_w <= 0 || src_h <= 0) {
        // Checkerboard pattern for empty layers
        for (int y = 0; y < thumb_size; ++y) {
            for (int x = 0; x < thumb_size; ++x) {
                const size_t idx = (static_cast<size_t>(y) * thumb_size + x) * 4;
                const uint8_t c = ((x / 4 + y / 4) % 2 == 0) ? 40 : 55;
                dst_rgba[idx + 0] = c; dst_rgba[idx + 1] = c;
                dst_rgba[idx + 2] = c; dst_rgba[idx + 3] = 255;
            }
        }
        return;
    }
    const float sx = static_cast<float>(src_w) / static_cast<float>(thumb_size);
    const float sy = static_cast<float>(src_h) / static_cast<float>(thumb_size);
    for (int y = 0; y < thumb_size; ++y) {
        for (int x = 0; x < thumb_size; ++x) {
            const int src_x = std::min(static_cast<int>(x * sx), src_w - 1);
            const int src_y = std::min(static_cast<int>(y * sy), src_h - 1);
            const auto& p = src[static_cast<size_t>(src_y) * src_w + src_x];
            const size_t idx = (static_cast<size_t>(y) * thumb_size + x) * 4;
            // Alpha-blend over checkerboard
            const float a = static_cast<float>(p.a) / 255.0f;
            const uint8_t bg = ((x / 4 + y / 4) % 2 == 0) ? 40 : 55;
            dst_rgba[idx + 0] = static_cast<uint8_t>(p.r * a + bg * (1.0f - a));
            dst_rgba[idx + 1] = static_cast<uint8_t>(p.g * a + bg * (1.0f - a));
            dst_rgba[idx + 2] = static_cast<uint8_t>(p.b * a + bg * (1.0f - a));
            dst_rgba[idx + 3] = 255;
        }
    }
}

} // anonymous namespace

SDL_Texture* SceneUI::getOrCreateLayerThumbnail(UIContext& ctx, Paint::PaintLayerData* layer,
                                                 Paint::PaintChannel channel) {
    if (!layer) return nullptr;
    constexpr int THUMB_SIZE = 36;

    const size_t ch_idx = static_cast<size_t>(channel);
    const auto& pixels = layer->channel_pixels[ch_idx];
    const uint64_t current_hash = hashLayerChannel(pixels);

    // Find existing cache entry
    for (auto& entry : layer_thumb_cache) {
        if (entry.layer_id == layer->id) {
            if (entry.content_hash == current_hash && entry.texture) {
                return entry.texture;
            }
            // Update existing
            std::vector<uint8_t> rgba;
            downsampleToThumb(pixels, layer->width, layer->height, rgba, THUMB_SIZE);
            if (!entry.texture) {
                SDL_Renderer* sdl_r = ctx.renderer.sdlRenderer;
                if (sdl_r) {
                    entry.texture = SDL_CreateTexture(sdl_r, SDL_PIXELFORMAT_RGBA32,
                                                      SDL_TEXTUREACCESS_STREAMING, THUMB_SIZE, THUMB_SIZE);
                }
            }
            if (entry.texture) {
                SDL_UpdateTexture(entry.texture, nullptr, rgba.data(), THUMB_SIZE * 4);
            }
            entry.content_hash = current_hash;
            return entry.texture;
        }
    }

    // Create new
    LayerThumbEntry entry;
    entry.layer_id = layer->id;
    entry.content_hash = current_hash;
    SDL_Renderer* sdl_r = ctx.renderer.sdlRenderer;
    if (sdl_r) {
        entry.texture = SDL_CreateTexture(sdl_r, SDL_PIXELFORMAT_RGBA32,
                                           SDL_TEXTUREACCESS_STREAMING, THUMB_SIZE, THUMB_SIZE);
        if (entry.texture) {
            std::vector<uint8_t> rgba;
            downsampleToThumb(pixels, layer->width, layer->height, rgba, THUMB_SIZE);
            SDL_UpdateTexture(entry.texture, nullptr, rgba.data(), THUMB_SIZE * 4);
        }
    }
    layer_thumb_cache.push_back(entry);
    return entry.texture;
}

void SceneUI::drawPaintLayerPanel(UIContext& ctx, Paint::MeshPaintAdapter* adapter) {
    if (!adapter) return;

    Paint::PaintTextureSet* tex_set = adapter->getTextureSet();
    if (!tex_set || !tex_set->initialized) {
        paint_mode_state.bindLayerStack(nullptr);
        ImGui::TextDisabled("Create a texture set to enable layers.");
        return;
    }

    Paint::PaintLayerStack& stack = adapter->ensureLayerStack();
    if (paint_mode_state.getBoundLayerStack() != &stack)
        paint_mode_state.bindLayerStack(&stack);

    auto markPaintTexturesVulkanDirtyFull = [&]() {
        if (Paint::PaintTextureSet* texSet = adapter->getTextureSet()) {
            for (int ch = 0; ch < static_cast<int>(Paint::kPaintChannelCount); ++ch) {
                if (auto texture = texSet->getTexture(static_cast<Paint::PaintChannel>(ch))) {
                    texture->markVulkanDirtyFull();
                }
            }
        }
    };

    auto refreshLayerMaterialPreview = [&]() {
        ctx.renderer.resetCPUAccumulation();

        auto syncBackend = [&](Backend::IBackend* backend) {
            if (!backend) return;
            markPaintTexturesVulkanDirtyFull();
            ctx.renderer.updateBackendMaterials(ctx.scene, backend);
            backend->resetAccumulation();
        };

        syncBackend(ctx.backend_ptr);
        if (g_viewport_backend && g_viewport_backend.get() != ctx.backend_ptr) {
            syncBackend(g_viewport_backend.get());
        }

        ctx.start_render = true;
    };

    const float panel_width = ImGui::GetContentRegionAvail().x;
    const int layer_count = stack.layerCount();
    const ImGuiStyle& style = ImGui::GetStyle();

    // -------- header row: "Layers" + blend + opacity --------
    Paint::PaintLayerData* active = stack.layerAt(paint_mode_state.active_layer_index);
    {
        ImGui::TextColored(ImVec4(0.55f, 0.78f, 1.0f, 1.0f), "Layers");
        if (active) {
            // Blend mode right of header
            ImGui::SameLine();
            const float remaining = ImGui::GetContentRegionAvail().x;
            const float combo_w = remaining * 0.5f;
            const float slider_w = remaining - combo_w - style.ItemSpacing.x;

            const char* blend_labels[] = { "Normal", "Add", "Multiply", "Screen", "Overlay" };
            int blend = static_cast<int>(active->meta.blend_mode);
            ImGui::SetNextItemWidth(combo_w);
            if (ImGui::Combo("##blend", &blend, blend_labels, Paint::kLayerBlendModeCount)) {
                active->meta.blend_mode = static_cast<Paint::LayerBlendMode>(blend);
                paint_mode_state.syncLayersFromStack();
                adapter->compositeAndUpload();
                refreshLayerMaterialPreview();
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(slider_w);
            float opacity = active->meta.opacity;
            if (ImGui::SliderFloat("##opacity", &opacity, 0.0f, 1.0f, "%.2f")) {
                active->meta.opacity = opacity;
                paint_mode_state.syncLayersFromStack();
                adapter->compositeAndUpload();
                refreshLayerMaterialPreview();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Opacity");
        }
    }

    // -------- layer list (top = highest, Photoshop order) --------
    constexpr float ROW_HEIGHT = 32.0f;
    constexpr float EYE_COL_W = 22.0f;
    constexpr float LOCK_COL_W = 20.0f;
    {
        // Clamp stored height
        paint_layer_list_height = std::clamp(paint_layer_list_height, 60.0f, 500.0f);
        ImGui::BeginChild("##layer_list", ImVec2(panel_width, paint_layer_list_height), true);
        ImDrawList* dl = ImGui::GetWindowDrawList();

        for (int i = layer_count - 1; i >= 0; --i) {
            Paint::PaintLayerData* ld = stack.layerAt(i);
            if (!ld) continue;

            ImGui::PushID(i);
            const bool is_active = (paint_mode_state.active_layer_index == i);
            const ImVec2 row_min = ImGui::GetCursorScreenPos();
            const float row_w = ImGui::GetContentRegionAvail().x;
            const ImVec2 row_max(row_min.x + row_w, row_min.y + ROW_HEIGHT);

            // --- Row background ---
            if (is_active) {
                dl->AddRectFilled(row_min, row_max, IM_COL32(50, 85, 140, 140), 3.0f);
                dl->AddRect(row_min, row_max, IM_COL32(90, 150, 255, 200), 3.0f);
            } else {
                dl->AddRectFilled(row_min, row_max, IM_COL32(42, 42, 48, 220), 2.0f);
            }

            // --- Eye toggle (real checkbox, not InvisibleButton) ---
            ImGui::SetCursorScreenPos(ImVec2(row_min.x + 3.0f, row_min.y + (ROW_HEIGHT - 16.0f) * 0.5f));
            bool vis = ld->meta.visible;
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
            if (ImGui::Checkbox("##vis", &vis)) {
                ld->meta.visible = vis;
                paint_mode_state.syncLayersFromStack();
                adapter->compositeAndUpload();
                refreshLayerMaterialPreview();
                g_ProjectManager.markModified();
            }
            ImGui::PopStyleVar();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(vis ? "Hide" : "Show");

            // --- Layer name (Selectable for row selection) ---
            ImGui::SameLine();
            ImGui::SetCursorScreenPos(ImVec2(row_min.x + EYE_COL_W + 2.0f, row_min.y));
            const float selectable_w = row_w - EYE_COL_W - LOCK_COL_W - 6.0f;

            ImGui::PushStyleColor(ImGuiCol_Header, IM_COL32(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, IM_COL32(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, IM_COL32(0, 0, 0, 0));
            // Dim hidden layers
            if (!ld->meta.visible) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.45f, 0.48f, 1.0f));
            if (ImGui::Selectable(ld->meta.name.c_str(), is_active,
                                  ImGuiSelectableFlags_AllowOverlap,
                                  ImVec2(selectable_w, ROW_HEIGHT))) {
                paint_mode_state.active_layer_index = i;
                paint_mode_state.active_layer_id = ld->id;
                if (!paint_mode_state.active_target_name.empty()) {
                    paint_mode_state.last_layer_id_by_target[paint_mode_state.active_target_name] = ld->id;
                }
            }
            if (!ld->meta.visible) ImGui::PopStyleColor();
            ImGui::PopStyleColor(3);

            // --- Lock toggle ---
            ImGui::SameLine();
            ImGui::SetCursorScreenPos(ImVec2(row_max.x - LOCK_COL_W,
                                              row_min.y + (ROW_HEIGHT - 16.0f) * 0.5f));
            bool lck = ld->meta.locked;
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
            if (ImGui::Checkbox("##lck", &lck)) {
                ld->meta.locked = lck;
                paint_mode_state.syncLayersFromStack();
            }
            ImGui::PopStyleVar();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(lck ? "Unlock" : "Lock");

            // Advance cursor
            ImGui::SetCursorScreenPos(ImVec2(row_min.x, row_max.y + 1.0f));
            ImGui::Dummy(ImVec2(0.0f, 0.0f));
            ImGui::PopID();
        }

        ImGui::EndChild();

        // Resize handle: drag bottom edge of layer list
        {
            const ImVec2 handle_min = ImGui::GetCursorScreenPos();
            const ImVec2 handle_max(handle_min.x + panel_width, handle_min.y + 6.0f);
            ImGui::InvisibleButton("##layer_resize", ImVec2(panel_width, 6.0f));
            if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            if (ImGui::IsItemActive()) {
                paint_layer_list_height += ImGui::GetIO().MouseDelta.y;
                paint_layer_list_height = std::clamp(paint_layer_list_height, 60.0f, 500.0f);
            }
            // Visual handle bar
            ImDrawList* hdl = ImGui::GetWindowDrawList();
            const float cx = (handle_min.x + handle_max.x) * 0.5f;
            const float cy = (handle_min.y + handle_max.y) * 0.5f;
            const ImU32 hcol = ImGui::IsItemHovered() ? IM_COL32(120, 160, 220, 200)
                                                       : IM_COL32(80, 80, 90, 150);
            hdl->AddLine(ImVec2(cx - 16.0f, cy), ImVec2(cx + 16.0f, cy), hcol, 2.0f);
        }
    }

    // -------- active layer name edit --------
    if (active) {
        char name_buf[128];
        snprintf(name_buf, sizeof(name_buf), "%s", active->meta.name.c_str());
        ImGui::SetNextItemWidth(panel_width);
        if (ImGui::InputText("##layer_name", name_buf, sizeof(name_buf))) {
            active->meta.name = name_buf;
            paint_mode_state.syncLayersFromStack();
            g_ProjectManager.markModified();
        }
    }

    // -------- bottom toolbar (centered) --------
    {
        const float btn_w = 26.0f;
        const float btn_h = 22.0f;
        const int btn_count = 7;
        const float spacing = 2.0f;
        const float total_w = btn_w * btn_count + spacing * (btn_count - 1);
        const float offset = (panel_width - total_w) * 0.5f;
        if (offset > 0.0f) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);

        float tb_round = 3.0f;
        if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
            tb_round = ThemeManager::instance().current().style.frameRounding;
        }
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, tb_round);

        if (ImGui::Button("+##add", ImVec2(btn_w, btn_h))) {
            const int idx = paint_mode_state.addLayerAboveCurrent();
            if (idx >= 0) {
                // A brand-new empty layer contributes nothing to the composite,
                // so skip compositeAndUpload() here: flattening the entire stack
                // would clobber any texture state that drifted away from the
                // stack (e.g. external edits, undo, or stale base layer seeded
                // before the current session's fills/strokes).
                g_ProjectManager.markModified();
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Add Layer");

        ImGui::SameLine();
        if (ImGui::Button("D##dup", ImVec2(btn_w, btn_h))) {
            if (paint_mode_state.duplicateCurrentLayer() >= 0) {
                adapter->compositeAndUpload();
                refreshLayerMaterialPreview();
                g_ProjectManager.markModified();
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Duplicate");

        ImGui::SameLine();
        ImGui::BeginDisabled(paint_mode_state.active_layer_index >= layer_count - 1);
        if (ImGui::Button("^##up", ImVec2(btn_w, btn_h))) {
            if (paint_mode_state.moveCurrentLayer(1)) {
                adapter->compositeAndUpload();
                refreshLayerMaterialPreview();
                g_ProjectManager.markModified();
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move Up");
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::BeginDisabled(paint_mode_state.active_layer_index <= 0);
        if (ImGui::Button("v##dn", ImVec2(btn_w, btn_h))) {
            if (paint_mode_state.moveCurrentLayer(-1)) {
                adapter->compositeAndUpload();
                refreshLayerMaterialPreview();
                g_ProjectManager.markModified();
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move Down");
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::BeginDisabled(paint_mode_state.active_layer_index <= 0);
        if (ImGui::Button("M##mrg", ImVec2(btn_w, btn_h))) {
            if (paint_mode_state.mergeCurrentLayerDown()) {
                adapter->compositeAndUpload();
                refreshLayerMaterialPreview();
                g_ProjectManager.markModified();
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Merge Down");
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::BeginDisabled(layer_count <= 1);
        if (ImGui::Button("F##flat", ImVec2(btn_w, btn_h))) {
            paint_mode_state.flattenAllLayers();
            adapter->compositeAndUpload();
            refreshLayerMaterialPreview();
            g_ProjectManager.markModified();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Flatten All");
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::BeginDisabled(layer_count <= 1);
        if (ImGui::Button("X##del", ImVec2(btn_w, btn_h))) {
            if (paint_mode_state.removeCurrentLayer()) {
                adapter->compositeAndUpload();
                refreshLayerMaterialPreview();
                g_ProjectManager.markModified();
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Delete Layer");
        ImGui::EndDisabled();

        ImGui::PopStyleVar(2);
    }
}

// ======================== Per-Channel Texture Slots ========================

void SceneUI::drawPaintChannelTextureSlots(UIContext& ctx, Paint::MeshPaintAdapter* adapter) {
    if (!adapter) return;

    Paint::PaintTextureSet* texture_set = adapter->getTextureSet();
    if (!texture_set || !texture_set->initialized) return;

    UIWidgets::ColoredHeader("Channel Textures", ImVec4(0.75f, 0.88f, 0.55f, 1.0f));

    // Compact channel overview — colored indicators for every channel
    {
        const char* short_labels[] = { "BC", "N", "R", "M", "E", "H", "T" };
        const char* long_labels[]  = { "Base Color", "Normal", "Roughness", "Metallic", "Emission", "Height Mask", "Transmission" };
        static_assert(IM_ARRAYSIZE(short_labels) == IM_ARRAYSIZE(long_labels),
                      "short and long channel labels must stay in sync");
        bool printed_any = false;
        for (int ch = 0; ch < static_cast<int>(Paint::kPaintChannelCount); ++ch) {
            const auto channel = static_cast<Paint::PaintChannel>(ch);
            if (!isMeshPaintUiChannelEnabled(channel)) {
                continue;
            }
            if (printed_any) ImGui::SameLine(0.0f, 4.0f);
            printed_any = true;
            const bool has_tex = texture_set->getTexture(channel) != nullptr;
            const bool is_active = (channel == paint_mode_state.active_channel);
            ImVec4 color = has_tex
                ? (is_active ? ImVec4(0.3f, 1.0f, 0.5f, 1.0f) : ImVec4(0.6f, 0.8f, 0.6f, 1.0f))
                : ImVec4(0.5f, 0.5f, 0.5f, 0.6f);
            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::Text("[%s]", short_labels[ch]);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                std::shared_ptr<Texture> t = texture_set->getTexture(channel);
                if (t) ImGui::SetTooltip("%s: %dx%d", long_labels[ch], t->width, t->height);
                else   ImGui::SetTooltip("%s: Empty", long_labels[ch]);
            }
        }
    }

    // Active channel detail — import / export for the selected channel only
    if (!isMeshPaintUiChannelEnabled(paint_mode_state.active_channel)) {
        paint_mode_state.active_channel = Paint::PaintChannel::BaseColor;
    }
    const auto active_ch = paint_mode_state.active_channel;
    std::shared_ptr<Texture> active_tex = texture_set->getTexture(active_ch);
    const char* ch_name = paintChannelDisplayName(active_ch);

    if (active_tex) {
        ImGui::Text("%s: %d x %d", ch_name, active_tex->width, active_tex->height);
    } else {
        ImGui::TextDisabled("%s: Empty", ch_name);
    }

    // Source info
    if (texture_set->wasSeededFromExisting(active_ch)) {
        const std::string& src_name = texture_set->getSourceTextureName(active_ch);
        if (!src_name.empty()) {
            std::string short_name = src_name;
            const auto last_sep = src_name.find_last_of("/\\");
            if (last_sep != std::string::npos)
                short_name = src_name.substr(last_sep + 1);
            if (short_name.size() > 24)
                short_name = "..." + short_name.substr(short_name.size() - 21);
            ImGui::TextDisabled("Source: %s", short_name.c_str());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", src_name.c_str());
        }
    }

    // Import / Export buttons side by side for active channel
    const float btn_w = (UIWidgets::GetInspectorActionWidth() - ImGui::GetStyle().ItemSpacing.x) * 0.5f;

    if (ImGui::Button("Import##ch_active", ImVec2(btn_w, 0))) {
        const std::string path = SceneUI::openFileDialogW(
            L"Image Files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.exr;*.hdr\0", "", "");
        if (!path.empty()) {
            auto imported = std::make_shared<Texture>(path, TextureType::Unknown);
            if (imported && imported->is_loaded() && !imported->pixels.empty()) {
                const int target_res = texture_set->resolution > 0 ? texture_set->resolution : 1024;
                if (imported->width != target_res || imported->height != target_res) {
                    std::vector<CompactVec4> resized(
                        static_cast<size_t>(target_res) * static_cast<size_t>(target_res));
                    for (int y = 0; y < target_res; ++y) {
                        for (int x = 0; x < target_res; ++x) {
                            const float u = static_cast<float>(x) / static_cast<float>(target_res - 1);
                            const float v = static_cast<float>(y) / static_cast<float>(target_res - 1);
                            const int sx = std::clamp(static_cast<int>(u * (imported->width - 1) + 0.5f), 0, imported->width - 1);
                            const int sy = std::clamp(static_cast<int>(v * (imported->height - 1) + 0.5f), 0, imported->height - 1);
                            resized[y * target_res + x] = imported->pixels[sy * imported->width + sx];
                        }
                    }
                    imported->pixels = std::move(resized);
                    imported->width = target_res;
                    imported->height = target_res;
                }

                std::shared_ptr<Texture>& target_ref = texture_set->getTextureRef(active_ch);
                if (!target_ref) {
                    target_ref = imported;
                } else {
                    target_ref->pixels = imported->pixels;
                    target_ref->width = imported->width;
                    target_ref->height = imported->height;
                    target_ref->m_is_loaded = true;
                }
                texture_set->setSourceInfo(active_ch, true, path);
                texture_set->setSourceTexture(active_ch, imported);

                adapter->bindTextureSetToMaterial();
                if (target_ref->isUploaded()) {
                    target_ref->updateGPU();
                } else {
                    target_ref->upload_to_gpu();
                }

                Paint::PaintLayerStack* layer_stack = adapter->getLayerStack();
                if (layer_stack && layer_stack->layerCount() > 0) {
                    layer_stack->setResolution(target_res, target_res);
                    Paint::PaintLayerData* base_layer = layer_stack->layerAt(0);
                    if (base_layer) {
                        auto& buf = base_layer->ensurePixels(active_ch);
                        buf = target_ref->pixels;
                    }
                }

                ctx.renderer.resetCPUAccumulation();
                if (ctx.backend_ptr) {
                    ctx.renderer.updateBackendMaterials(ctx.scene, ctx.backend_ptr);
                    ctx.backend_ptr->resetAccumulation();
                }
                if (g_viewport_backend && g_viewport_backend.get() != ctx.backend_ptr) {
                    ctx.renderer.updateBackendMaterials(ctx.scene, g_viewport_backend.get());
                }
                g_ProjectManager.markModified();
            }
        }
    }

    ImGui::SameLine();
    ImGui::BeginDisabled(!active_tex);
    if (ImGui::Button("Export##ch_active", ImVec2(btn_w, 0))) {
        const std::string path = SceneUI::saveFileDialogW(L"PNG Files\0*.png\0", L"png");
        if (!path.empty() && active_tex) {
            exportTextureToPng(active_tex, path);
        }
    }
    ImGui::EndDisabled();
}

bool SceneUI::shouldShowPaintBrushDock() const {
    if (active_properties_tab == 8 && show_hair_tab) {
        return true;
    }

    if (sculpt_mode_state.enabled &&
        (!sculpt_mode_state.active_target_name.empty() || terrain_sculpt_proxy_active) &&
        mesh_overlay_settings.edit_mode) {
        return true;
    }

    if (mesh_workspace_mode == MeshWorkspaceMode::Edit &&
        mesh_overlay_settings.enabled &&
        mesh_overlay_settings.edit_mode &&
        !active_mesh_edit_object_name.empty()) {
        return true;
    }

    if (!paint_mode_state.enabled || !paint_mode_state.hasValidTarget()) {
        return false;
    }

    const auto adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter());
    return adapter && adapter->isValid();
}

float SceneUI::getPaintBrushDockWidth() const {
    if (!shouldShowPaintBrushDock()) {
        return 0.0f;
    }
    return std::clamp(paint_brush_dock_width, 50.0f, 400.0f);
}

void SceneUI::drawPaintBrushControls(UIContext& ctx, const std::shared_ptr<Triangle>& meshTriangle, bool rightDockOnly) {
    syncHeightMaskPaintToggles(paint_mode_state);
    auto adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter());
    if (!adapter && meshTriangle) {
        paint_mode_state.setAdapter(std::make_shared<Paint::MeshPaintAdapter>(&ctx.scene, meshTriangle));
        adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter());
    }
    if (!adapter) {
        if (!rightDockOnly) {
            ImGui::TextDisabled("Select a paintable mesh target to edit brush settings.");
        }
        return;
    }

    if (rightDockOnly) {
        UIWidgets::PushControlSurfaceStyle(ImVec4(0.32f, 0.88f, 0.82f, 1.0f));
        const float tool_gap = 3.0f;
        const float tool_size = 44.0f;
        const float avail_w = ImGui::GetContentRegionAvail().x;
        const int columns = std::max(1, static_cast<int>((avail_w + tool_gap) / (tool_size + tool_gap)));
        int col_idx = 0;

        const struct ToolInfo {
            const char* id;
            const char* tooltip;
            Paint::BrushTool tool;
        } paint_tools[] = {
            { "PaintToolPaint", "Paint\nApplies color or texture to the active channel.", Paint::BrushTool::Paint },
            { "PaintToolErase", "Erase\nRemoves paint from the active channel.", Paint::BrushTool::Erase },
            { "PaintToolSoften", "Soften\nBlends and smooths harsh paint transitions.", Paint::BrushTool::Soften },
            { "PaintToolStamp", "Stamp\nPlaces a stamped alpha or texture imprint.", Paint::BrushTool::Stamp },
            { "PaintToolFill", "Fill\nFills the whole target with the current value.", Paint::BrushTool::Fill },
            { "PaintToolClone", "Clone\nCopies paint from a sampled source area.", Paint::BrushTool::Clone },
            { "PaintToolSpray", "Spray\nScatters many small droplets across the brush radius.", Paint::BrushTool::Spray }
        };

        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(tool_gap, tool_gap));
        for (const auto& t : paint_tools) {
            if (col_idx > 0 && (col_idx % columns) != 0) {
                ImGui::SameLine(0.0f, tool_gap);
            }
            drawPaintToolSelectorButton(t.id, t.tooltip, t.tool, paint_mode_state.brush.tool, tool_size, tool_size);
            col_idx++;
        }
        ImGui::PopStyleVar();
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    Paint::PaintTextureSet* texture_set = adapter->getTextureSet();

    auto savePresetUI = [&]() {
        ensurePaintBrushPresets();
        if (!paint_brush_presets.empty()) {
            int selected_preset_index = 0;
            for (int i = 0; i < static_cast<int>(paint_brush_presets.size()); ++i) {
                if (paint_brush_presets[i].name == paint_brush_preset_name) {
                    selected_preset_index = i;
                    break;
                }
            }
            std::vector<const char*> preset_names;
            preset_names.reserve(paint_brush_presets.size());
            for (const auto& preset : paint_brush_presets) {
                preset_names.push_back(preset.name.c_str());
            }
            if (ImGui::Combo("Brush Preset", &selected_preset_index, preset_names.data(), static_cast<int>(preset_names.size()))) {
                paint_mode_state.brush = paint_brush_presets[selected_preset_index].brush;
                std::snprintf(paint_brush_preset_name, sizeof(paint_brush_preset_name), "%s", paint_brush_presets[selected_preset_index].name.c_str());
            }
            ImGui::InputText("Preset Name", paint_brush_preset_name, sizeof(paint_brush_preset_name));
            if (UIWidgets::SecondaryButton("Save Brush Preset", ImVec2(-1.0f, 0))) {
                std::string preset_name = paint_brush_preset_name;
                if (preset_name.empty()) {
                    preset_name = "Custom Brush";
                }
                bool updated_existing = false;
                for (auto& preset : paint_brush_presets) {
                    if (preset.name == preset_name) {
                        preset.brush = paint_mode_state.brush;
                        updated_existing = true;
                        break;
                    }
                }
                if (!updated_existing) {
                    paint_brush_presets.push_back(PaintBrushPreset{ preset_name, paint_mode_state.brush });
                }
            }
        }
    };

    ImGui::PushItemWidth((std::min)(176.0f, ImGui::GetContentRegionAvail().x - 18.0f));

    savePresetUI();

    if (beginBrushDockSection("Stroke")) {
        ImGui::SliderFloat("Radius", &paint_mode_state.brush.radius, 0.1f, 256.0f, "%.2f px");
        ImGui::SliderFloat("Strength", &paint_mode_state.brush.strength, 0.01f, 10.0f, "%.2f");
        ImGui::SliderFloat("Falloff", &paint_mode_state.brush.falloff, 0.0f, 1.0f, "%.2f");
        if (!paint_mode_state.compact_ui) {
            ImGui::SliderFloat("Spacing", &paint_mode_state.brush.spacing, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Flow", &paint_mode_state.brush.flow, 0.1f, 2.0f, "%.2f");
        }
        ImGui::Checkbox("Show Brush Preview", &paint_mode_state.brush.show_preview);
    }

    if (beginBrushDockSection("Color & Texture")) {
        ImGui::TextDisabled("Fallback Source");
        float brush_color[3] = {
            paint_mode_state.brush.color.x,
            paint_mode_state.brush.color.y,
            paint_mode_state.brush.color.z
        };
        if (ImGui::ColorEdit3(
                paint_mode_state.brush.use_paint_texture ? "Tint" : "Paint Value",
                brush_color,
                ImGuiColorEditFlags_NoInputs)) {
            paint_mode_state.brush.color = Vec3(brush_color[0], brush_color[1], brush_color[2]);
        }

        if (ImGui::Button("Load Paint Texture")) {
            const std::string path = SceneUI::openFileDialogW(
                L"Image Files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.exr;*.hdr\0",
                "",
                "");
            if (!path.empty()) {
                auto paint_texture = std::make_shared<Texture>(path, inferBrushTextureType(paint_mode_state.active_channel));
                if (paint_texture && paint_texture->is_loaded()) {
                    paint_mode_state.brush.paint_texture = paint_texture;
                    paint_mode_state.brush.paint_texture_path = path;
                    paint_mode_state.brush.use_paint_texture = true;
                }
            }
        }
        if (paint_mode_state.brush.paint_texture && paint_mode_state.brush.paint_texture->is_loaded()) {
            ImGui::SameLine();
            if (ImGui::Button("Clear Paint Texture")) {
                paint_mode_state.brush.paint_texture.reset();
                paint_mode_state.brush.paint_texture_path.clear();
                paint_mode_state.brush.use_paint_texture = false;
            }

            ImGui::Checkbox("Use Paint Texture", &paint_mode_state.brush.use_paint_texture);
            const std::string paint_name = brushAssetDisplayName(paint_mode_state.brush.paint_texture_path);
            ImGui::TextDisabled("Paint Texture: %s", paint_name.empty() ? "Untitled" : paint_name.c_str());
            ImGui::TextDisabled("Loaded as: %s", paintChannelDisplayName(paint_mode_state.active_channel));
            int tint_mode = static_cast<int>(paint_mode_state.brush.paint_texture_tint_mode);
            const char* tint_mode_labels[] = { "Multiply", "Recolor", "Overlay" };
            if (ImGui::Combo("Tint Mode", &tint_mode, tint_mode_labels, IM_ARRAYSIZE(tint_mode_labels))) {
                paint_mode_state.brush.paint_texture_tint_mode = static_cast<Paint::PaintTextureTintMode>(tint_mode);
            }
            ImGui::SliderFloat("Tint Strength", &paint_mode_state.brush.paint_texture_tint_strength, 0.0f, 1.0f, "%.2f");
        }

        std::vector<Paint::PaintChannel> source_channels = getSelectedMaterialBrushChannels(paint_mode_state);
        if (source_channels.empty() && isMeshPaintUiChannelEnabled(paint_mode_state.active_channel)) {
            source_channels.push_back(paint_mode_state.active_channel);
        }
        if (!source_channels.empty()) {
            ImGui::Separator();
            ImGui::TextDisabled("Per-Channel Sources");
            ImGui::TextDisabled("Active and linked channel sources.");
            for (Paint::PaintChannel channel : source_channels) {
                Paint::BrushChannelInput* input = getBrushChannelInput(paint_mode_state.brush, channel);
                if (!input) {
                    continue;
                }
                const std::string channel_name = paintChannelDisplayName(channel);
                ImGui::PushID(static_cast<int>(channel));
                const bool is_active_channel = channel == paint_mode_state.active_channel;
                const std::string header = channel_name + (is_active_channel ? "  *" : "");
                if (ImGui::TreeNodeEx(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Checkbox("Override", &input->enabled);
                    if (!input->enabled) {
                        ImGui::BeginDisabled();
                    }

                    float channel_color[3] = { input->color.x, input->color.y, input->color.z };
                    if (ImGui::ColorEdit3("Value", channel_color, ImGuiColorEditFlags_NoInputs)) {
                        input->enabled = true;
                        input->color = Vec3(channel_color[0], channel_color[1], channel_color[2]);
                    }

                    if (ImGui::Button("Load Texture")) {
                        const std::string path = SceneUI::openFileDialogW(
                            L"Image Files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.exr;*.hdr\0",
                            "",
                            "");
                        if (!path.empty()) {
                            auto paint_texture = std::make_shared<Texture>(path, inferBrushTextureType(channel));
                            if (paint_texture && paint_texture->is_loaded()) {
                                input->paint_texture = paint_texture;
                                input->paint_texture_path = path;
                                input->use_paint_texture = true;
                                input->enabled = true;
                            }
                        }
                    }
                    if (input->paint_texture && input->paint_texture->is_loaded()) {
                        ImGui::SameLine();
                        if (ImGui::Button("Clear Texture")) {
                            input->paint_texture.reset();
                            input->paint_texture_path.clear();
                            input->use_paint_texture = false;
                        }
                        ImGui::Checkbox("Use Texture", &input->use_paint_texture);
                        const std::string texture_name = brushAssetDisplayName(input->paint_texture_path);
                        ImGui::TextDisabled("Texture: %s", texture_name.empty() ? "Untitled" : texture_name.c_str());
                        int tint_mode = static_cast<int>(input->tint_mode);
                        const char* tint_mode_labels[] = { "Multiply", "Recolor", "Overlay" };
                        if (ImGui::Combo("Tint Mode", &tint_mode, tint_mode_labels, IM_ARRAYSIZE(tint_mode_labels))) {
                            input->tint_mode = static_cast<Paint::PaintTextureTintMode>(tint_mode);
                            input->enabled = true;
                        }
                        if (ImGui::SliderFloat("Tint Strength", &input->tint_strength, 0.0f, 1.0f, "%.2f")) {
                            input->enabled = true;
                        }
                    }

                    if (!input->enabled) {
                        ImGui::EndDisabled();
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        if (brushSupportsRaisedPaint(paint_mode_state.brush.tool)) {
            ImGui::Separator();
            const bool height_mask_channel_active = paint_mode_state.active_channel == Paint::PaintChannel::Mask;
            if (height_mask_channel_active) {
                ImGui::BeginDisabled();
            }
            ImGui::Checkbox("Raised Paint", &paint_mode_state.brush.write_height_mask);
            if (height_mask_channel_active) {
                ImGui::EndDisabled();
            }
            if (paint_mode_state.brush.write_height_mask) {
                ImGui::SliderFloat("Height Contribution", &paint_mode_state.brush.height_contribution, 0.01f, 1.0f, "%.2f");
                if (height_mask_channel_active) {
                    ImGui::BeginDisabled();
                }
                ImGui::Checkbox("Auto Normal From Height", &paint_mode_state.auto_normal_from_height);
                if (height_mask_channel_active) {
                    ImGui::EndDisabled();
                }
                if (paint_mode_state.auto_normal_from_height) {
                    ImGui::SliderFloat("Normal Strength", &paint_mode_state.height_to_normal_strength, 0.1f, 32.0f, "%.2f");
                }
            }
        }
    }

    if (paint_mode_state.brush.tool != Paint::BrushTool::Erase &&
        paint_mode_state.brush.tool != Paint::BrushTool::Soften &&
        paint_mode_state.brush.tool != Paint::BrushTool::Stamp &&
        paint_mode_state.brush.tool != Paint::BrushTool::Fill &&
        paint_mode_state.brush.tool != Paint::BrushTool::Clone &&
        beginBrushDockSection("Behavior")) {
        auto wetParamTip = [](const char* text) {
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
                ImGui::TextUnformatted(text);
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
        };
        drawPaintBehaviorSelector(paint_mode_state);
        if (paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Mix) {
            ImGui::SliderFloat("Mix Amount", &paint_mode_state.brush.mix_amount, 0.05f, 0.95f, "%.2f");
        } else if (paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Smudge) {
            ImGui::SliderFloat("Smudge Strength", &paint_mode_state.brush.smudge_strength, 0.05f, 1.00f, "%.2f");
        } else if (paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Wet ||
                   paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Oil) {
            const bool oil_mode = paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Oil;
            static const char* wet_quality_labels[] = { "Auto", "Balanced", "High", "Ultra" };
            int wet_quality_index = static_cast<int>(paint_mode_state.brush.wet_simulation_quality);
            if (ImGui::Combo("Simulation Quality", &wet_quality_index, wet_quality_labels, IM_ARRAYSIZE(wet_quality_labels))) {
                wet_quality_index = std::clamp(wet_quality_index, 0, static_cast<int>(IM_ARRAYSIZE(wet_quality_labels)) - 1);
                paint_mode_state.brush.wet_simulation_quality = static_cast<Paint::WetSimulationQuality>(wet_quality_index);
            }
            wetParamTip("Controls how often Wet/Oil flow simulation updates. Balanced is intentionally much lighter, High is a middle ground, Ultra keeps the highest realtime fidelity, and Auto becomes much more aggressive on 2K/4K textures.");
            ImGui::SliderFloat(oil_mode ? "Body" : "Wetness", &paint_mode_state.brush.wetness, 0.05f, 1.00f, "%.2f");
            wetParamTip(oil_mode
                ? "How dense and heavy the brush body feels on the surface. Higher values keep oil strokes richer and slower to break apart."
                : "Fresh paint amount carried by the brush and left on the surface. Higher values keep the stroke wetter for longer.");
            ImGui::SliderFloat("Wet Lifetime", &paint_mode_state.brush.wet_lifetime_seconds, 0.20f, 15.00f, "%.2fs");
            wetParamTip(oil_mode
                ? "How long the oily stroke stays workable before setting. Longer lifetime keeps blending active without needing much runoff."
                : "How long a stroke stays active before it fully dries. Longer lifetime means more time for blending and runoff.");
            ImGui::SliderFloat("Wet Diffusion", &paint_mode_state.brush.wet_diffusion, 0.00f, 1.50f, "%.2f");
            wetParamTip(oil_mode
                ? "How much thick pigment smears sideways into neighboring paint. Higher values create buttery bristle drag."
                : "Sideways spreading and color blending while paint is still wet. Higher values soften edges faster.");
            ImGui::SliderFloat("Wet Runoff", &paint_mode_state.brush.wet_runoff, 0.00f, 1.50f, "%.2f");
            wetParamTip(oil_mode
                ? "How much gravity can still pull heavy pigment downhill. Oil usually wants much lower values than watercolor-like wet paint."
                : "How strongly gravity pulls wet paint downhill in world space. Use this to make paint visibly slide on sloped surfaces.");
            ImGui::SliderFloat("Absorption", &paint_mode_state.brush.wet_absorption, 0.00f, 1.00f, "%.2f");
            wetParamTip(oil_mode
                ? "How much pigment settles into the surface instead of continuing to smear. Lower values keep an oily, movable stroke."
                : "How much wet paint sinks back toward the surface as it dries. Higher values can create soil-like settling or a shallow dried center.");
            ImGui::SliderFloat("Drip Head", &paint_mode_state.brush.wet_drip_head, 0.00f, 1.50f, "%.2f");
            wetParamTip(oil_mode
                ? "Extra buildup at the trailing end of a dragged stroke. Use lightly for palette-knife style ridges, not watery drops."
                : "Extra buildup at the lower end of a running stroke. Higher values make drips end in a thicker droplet head.");
            if (paint_mode_state.brush.write_height_mask) {
                ImGui::SliderFloat("Dry-End Buildup", &paint_mode_state.brush.wet_terminal_buildup, 0.00f, 1.50f, "%.2f");
                wetParamTip("Adds a soft raised bead only where enough pigment accumulates at the end of a wet run. Higher values strengthen the dried droplet buildup.");
                ImGui::SliderFloat("Buildup Softness", &paint_mode_state.brush.wet_terminal_softness, 0.10f, 1.00f, "%.2f");
                wetParamTip("Controls how broadly the dried droplet bead spreads around the terminal pigment pocket. Lower values stay tight, higher values soften the mound.");
            }
            ImGui::SliderFloat("Load", &paint_mode_state.brush.paint_load, 0.05f, 1.00f, "%.2f");
            wetParamTip(oil_mode
                ? "How much dense pigment the brush starts with. Higher values make early dabs stay opaque and saturated longer."
                : "How much fresh paint the brush starts with at the beginning of a stroke.");
            ImGui::SliderFloat("Pickup", &paint_mode_state.brush.pickup_rate, 0.00f, 1.00f, "%.2f");
            wetParamTip(oil_mode
                ? "How much existing paint the bristles pick back up. Lower values keep oil strokes laying paint down instead of washing it away."
                : "How strongly the brush picks color back up from the surface while moving.");
            ImGui::SliderFloat("Deposit", &paint_mode_state.brush.deposit_rate, 0.05f, 1.00f, "%.2f");
            wetParamTip(oil_mode
                ? "How aggressively thick paint is left on the surface each dab. Higher values make oil feel weighty and opaque."
                : "How aggressively the brush leaves its carried paint on the surface each dab.");
        }
    }

    const bool has_tool_settings =
        paint_mode_state.brush.tool == Paint::BrushTool::Stamp ||
        paint_mode_state.brush.tool == Paint::BrushTool::Fill ||
        paint_mode_state.brush.tool == Paint::BrushTool::Clone ||
        paint_mode_state.brush.tool == Paint::BrushTool::Spray;
    if (has_tool_settings && beginBrushDockSection("Tool Settings")) {
        if (paint_mode_state.brush.tool == Paint::BrushTool::Stamp) {
            int stamp_mode = static_cast<int>(paint_mode_state.brush.stamp_mode);
            const char* stamp_mode_labels[] = { "Single", "Continuous" };
            if (ImGui::Combo("Stamp Mode", &stamp_mode, stamp_mode_labels, IM_ARRAYSIZE(stamp_mode_labels))) {
                paint_mode_state.brush.stamp_mode = static_cast<Paint::StampPlacementMode>(stamp_mode);
            }
            ImGui::TextDisabled(
                paint_mode_state.brush.stamp_mode == Paint::StampPlacementMode::Single
                    ? "Stamp places one imprint per click."
                    : "Stamp places repeated imprints while dragging.");
        } else if (paint_mode_state.brush.tool == Paint::BrushTool::Fill) {
            ImGui::TextDisabled("Fill writes the active channel across the whole texture.");
        } else if (paint_mode_state.brush.tool == Paint::BrushTool::Clone) {
            ImGui::TextDisabled("Ctrl + Click sets clone source. Paint copies from that source.");
            if (paint_mode_state.stroke.has_clone_source) {
                ImGui::TextDisabled("Clone Source: %.3f, %.3f", paint_mode_state.stroke.clone_source_u, paint_mode_state.stroke.clone_source_v);
            } else {
                ImGui::TextDisabled("Clone Source: not set");
            }
        } else if (paint_mode_state.brush.tool == Paint::BrushTool::Spray) {
            ImGui::TextDisabled("Spray scatters many small dabs across the brush radius.");
            ImGui::SliderInt("Particles", &paint_mode_state.brush.spray_particles, 1, 64);
            ImGui::SliderFloat("Spread", &paint_mode_state.brush.spray_spread, 0.1f, 1.0f, "%.2f");
            ImGui::SliderFloat("Droplet Size", &paint_mode_state.brush.spray_droplet_size, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("Size Jitter", &paint_mode_state.brush.spray_size_jitter, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Opacity Jitter", &paint_mode_state.brush.spray_opacity_jitter, 0.0f, 1.0f, "%.2f");
        }
    }

    if (beginBrushDockSection("Shape")) {
        ImGui::TextDisabled("Symmetry");
        ImGui::Checkbox("Mirror X", &paint_mode_state.brush.mirror_x); ImGui::SameLine();
        ImGui::Checkbox("Mirror Y", &paint_mode_state.brush.mirror_y); ImGui::SameLine();
        ImGui::Checkbox("Mirror Z", &paint_mode_state.brush.mirror_z);

        int brush_shape = static_cast<int>(paint_mode_state.brush.shape);
        const char* shape_labels[] = { "Circle", "Rectangle", "Capsule", "Flat" };
        if (ImGui::Combo("Brush Shape", &brush_shape, shape_labels, IM_ARRAYSIZE(shape_labels))) {
            paint_mode_state.brush.shape = static_cast<Paint::BrushShape>(brush_shape);
            if (paint_mode_state.brush.shape == Paint::BrushShape::Circle) {
                paint_mode_state.brush.shape_aspect = 1.0f;
            } else if (paint_mode_state.brush.shape == Paint::BrushShape::Flat &&
                       paint_mode_state.brush.shape_aspect < 2.0f) {
                paint_mode_state.brush.shape_aspect = 4.0f;
            }
        }
        if (paint_mode_state.brush.shape != Paint::BrushShape::Circle) {
            ImGui::SliderFloat("Aspect", &paint_mode_state.brush.shape_aspect, 0.25f, 8.0f, "%.2f");
            ImGui::SliderFloat("Roundness", &paint_mode_state.brush.shape_roundness, 0.0f, 1.0f, "%.2f");
        }

        int alpha_preset = static_cast<int>(paint_mode_state.brush.alpha_preset);
        const char* alpha_labels[] = { "Soft Round", "Hard Round", "Noise", "Scratch", "Cloud" };
        if (ImGui::Combo("Brush Alpha", &alpha_preset, alpha_labels, IM_ARRAYSIZE(alpha_labels))) {
            paint_mode_state.brush.alpha_preset = static_cast<Paint::BrushAlphaPreset>(alpha_preset);
        }
        if (ImGui::Button("Load Alpha Mask")) {
            const std::string path = SceneUI::openFileDialogW(
                L"Image Files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.exr;*.hdr\0",
                "",
                "");
            if (!path.empty()) {
                auto alpha_texture = std::make_shared<Texture>(path, TextureType::Unknown);
                if (alpha_texture && alpha_texture->is_loaded()) {
                    paint_mode_state.brush.alpha_texture = alpha_texture;
                    paint_mode_state.brush.alpha_texture_path = path;
                    paint_mode_state.brush.use_imported_alpha = true;
                }
            }
        }
        if (paint_mode_state.brush.alpha_texture && paint_mode_state.brush.alpha_texture->is_loaded()) {
            ImGui::SameLine();
            if (ImGui::Button("Clear Alpha")) {
                paint_mode_state.brush.alpha_texture.reset();
                paint_mode_state.brush.alpha_texture_path.clear();
                paint_mode_state.brush.use_imported_alpha = false;
            }
            ImGui::Checkbox("Use Imported Alpha", &paint_mode_state.brush.use_imported_alpha);
            const std::string alpha_name = brushAlphaDisplayName(paint_mode_state.brush.alpha_texture_path);
            ImGui::TextDisabled("Imported: %s", alpha_name.empty() ? "Untitled" : alpha_name.c_str());
        }
        if (paint_mode_state.brush.alpha_preset == Paint::BrushAlphaPreset::Noise ||
            paint_mode_state.brush.alpha_preset == Paint::BrushAlphaPreset::Scratch ||
            paint_mode_state.brush.alpha_preset == Paint::BrushAlphaPreset::Cloud ||
            paint_mode_state.brush.use_imported_alpha) {
            ImGui::SliderFloat("Alpha Scale", &paint_mode_state.brush.alpha_scale, 0.25f, 8.0f, "%.2f");
        }
        ImGui::SliderFloat("Brush Rotation", &paint_mode_state.brush.alpha_rotation_degrees, -180.0f, 180.0f, "%.0f deg");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Ctrl + right-drag in the viewport rotates alpha textures and directional brush shapes.");
        }
        ImGui::Checkbox("Follow Stroke Angle", &paint_mode_state.brush.follow_stroke_angle);
        if (paint_mode_state.brush.tool != Paint::BrushTool::Stamp) {
            ImGui::BeginDisabled();
        }
        ImGui::SliderFloat("Scatter Jitter", &paint_mode_state.brush.scatter_jitter, 0.0f, 1.0f, "%.2f");
        if (paint_mode_state.brush.tool != Paint::BrushTool::Stamp) {
            ImGui::EndDisabled();
            ImGui::TextDisabled("Scatter Jitter affects Stamp placement.");
        }
        if (paint_mode_state.brush.tool == Paint::BrushTool::Stamp) {
            ImGui::Checkbox("Random Rotate", &paint_mode_state.brush.stamp_random_rotation);
            ImGui::SliderFloat("Scale Jitter", &paint_mode_state.brush.stamp_scale_jitter, 0.0f, 0.9f, "%.2f");
        }

        drawBrushAlphaPreview(paint_mode_state.brush);
        if (paint_mode_state.brush.paint_texture && paint_mode_state.brush.paint_texture->is_loaded()) {
            ImGui::SameLine();
            drawBrushPaintTexturePreview(paint_mode_state.brush);
        }
    }

    if (paint_mode_state.brush.tool == Paint::BrushTool::Fill) {
        const std::vector<Paint::PaintChannel> fill_channels = getSelectedMaterialBrushChannels(paint_mode_state);
        bool can_fill = false;
        if (texture_set) {
            for (Paint::PaintChannel channel : fill_channels) {
                if (texture_set->getTexture(channel)) {
                    can_fill = true;
                    break;
                }
            }
        }
        if (UIWidgets::PrimaryButton("Apply Fill", ImVec2(-1.0f, 0)) && can_fill) {
            bool any_changed = false;
            auto composite = std::make_unique<CompositeSceneCommand>(
                "Fill " + adapter->getNodeName());
            const bool auto_height_fill =
                paint_mode_state.brush.write_height_mask &&
                brushSupportsRaisedPaint(paint_mode_state.brush.tool);
            for (Paint::PaintChannel channel : fill_channels) {
                adapter->assignTextureToChannel(channel);
                std::shared_ptr<Texture> channel_texture = texture_set ? texture_set->getTexture(channel) : nullptr;
                const std::vector<CompactVec4> before_pixels = channel_texture ? channel_texture->pixels : std::vector<CompactVec4>{};
                if (adapter->fillChannel(channel, paint_mode_state.brush, paint_mode_state.active_layer_index)) {
                    any_changed = true;
                    if (channel_texture) {
                        composite->add(std::make_unique<PaintTextureCommand>(
                            adapter->getNodeName(),
                            adapter->getMaterialID(),
                            channel_texture,
                            before_pixels,
                            channel_texture->pixels));
                    }
                }
            }
            if (auto_height_fill) {
                Paint::BrushSettings height_brush = paint_mode_state.brush;
                height_brush.use_paint_texture = false;
                height_brush.paint_texture.reset();
                height_brush.paint_texture_path.clear();
                height_brush.channel_inputs = {};
                const float h = raisedPaintHeightValue(height_brush.height_contribution);
                height_brush.color = Vec3(h, h, h);

                adapter->assignTextureToChannel(Paint::PaintChannel::Mask);
                std::shared_ptr<Texture> mask_texture = texture_set ? texture_set->getTexture(Paint::PaintChannel::Mask) : nullptr;
                const std::vector<CompactVec4> mask_before_pixels = mask_texture ? mask_texture->pixels : std::vector<CompactVec4>{};
                if (adapter->fillChannel(Paint::PaintChannel::Mask, height_brush, paint_mode_state.active_layer_index)) {
                    any_changed = true;
                    if (mask_texture) {
                        composite->add(std::make_unique<PaintTextureCommand>(
                            adapter->getNodeName(),
                            adapter->getMaterialID(),
                            mask_texture,
                            mask_before_pixels,
                            mask_texture->pixels));
                    }
                }

                if (paint_mode_state.auto_normal_from_height) {
                    adapter->assignTextureToChannel(Paint::PaintChannel::Normal);
                    std::shared_ptr<Texture> normal_texture = texture_set ? texture_set->getTexture(Paint::PaintChannel::Normal) : nullptr;
                    const std::vector<CompactVec4> normal_before_pixels = normal_texture ? normal_texture->pixels : std::vector<CompactVec4>{};
                    if (adapter->generateNormalFromHeight(paint_mode_state.height_to_normal_strength)) {
                        any_changed = true;
                        normal_texture = texture_set ? texture_set->getTexture(Paint::PaintChannel::Normal) : nullptr;
                        if (normal_texture) {
                            composite->add(std::make_unique<PaintTextureCommand>(
                                adapter->getNodeName(),
                                adapter->getMaterialID(),
                                normal_texture,
                                normal_before_pixels,
                                normal_texture->pixels));
                        }
                    }
                }
            }
            if (any_changed) {
                ctx.renderer.resetCPUAccumulation();
                if (ctx.backend_ptr) {
                    ctx.renderer.updateBackendMaterials(ctx.scene, ctx.backend_ptr);
                    ctx.backend_ptr->resetAccumulation();
                }
                if (g_viewport_backend && g_viewport_backend.get() != ctx.backend_ptr) {
                    ctx.renderer.updateBackendMaterials(ctx.scene, g_viewport_backend.get());
                }
                if (!composite->empty()) {
                    history.record(std::move(composite));
                }
                g_ProjectManager.markModified();
            }
        }
    }
    ImGui::PopItemWidth();
}

void SceneUI::drawSculptBrushControls(UIContext& ctx, const std::shared_ptr<Triangle>& meshTriangle, bool rightDockOnly) {
    UIWidgets::PushControlSurfaceStyle(ImVec4(1.0f, 0.58f, 0.34f, 1.0f));
    TerrainObject* terrain = terrain_sculpt_proxy_active && terrain_brush.active_terrain_id != -1
        ? TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id)
        : nullptr;
    if (!meshTriangle && !terrain) {
        if (!rightDockOnly) {
            ImGui::TextDisabled("Select a mesh or terrain object to configure sculpt brushes.");
        }
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    auto savePresetUI = [&]() {
        ensureSculptBrushPresets();
        if (!sculpt_brush_presets.empty()) {
            int selected_preset_index = 0;
            for (int i = 0; i < static_cast<int>(sculpt_brush_presets.size()); ++i) {
                if (sculpt_brush_presets[i].name == sculpt_brush_preset_name) {
                    selected_preset_index = i;
                    break;
                }
            }
            std::vector<const char*> preset_names;
            preset_names.reserve(sculpt_brush_presets.size());
            for (const auto& preset : sculpt_brush_presets) {
                preset_names.push_back(preset.name.c_str());
            }
            if (ImGui::Combo("Brush Preset##sculpt", &selected_preset_index, preset_names.data(), static_cast<int>(preset_names.size()))) {
                sculpt_mode_state.brush = sculpt_brush_presets[selected_preset_index].brush;
                std::snprintf(sculpt_brush_preset_name, sizeof(sculpt_brush_preset_name), "%s", sculpt_brush_presets[selected_preset_index].name.c_str());
            }
        }
    };

    if (!rightDockOnly) {
        ImGui::PushItemWidth((std::min)(176.0f, ImGui::GetContentRegionAvail().x - 18.0f));
        
        // Stylized Target Info Panel
        if (terrain) {
            terrain_sculpt_proxy_active = true;
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.4f, 1.0f), "Target: %s", terrain->name.empty() ? "(unknown terrain)" : terrain->name.c_str());
            ImGui::TextDisabled("Type: Terrain  |  Mode: Sculpting");
        } else {
            terrain_sculpt_proxy_active = false;
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.4f, 1.0f), "Target: %s", meshTriangle->getNodeName().empty() ? "(unknown)" : meshTriangle->getNodeName().c_str());
            ImGui::TextDisabled("Type: Mesh     |  Mode: Sculpting");
        }
        ImGui::Separator();



        savePresetUI();
    }

    if (terrain) {
        if (terrain_brush.active_terrain_id != terrain->id) {
            terrain_brush.active_terrain_id = terrain->id;
        }

        if (rightDockOnly) {
            // Terrain Sculpting Tools (Elevation / Modification)
            const std::vector<std::pair<const char*, std::pair<UIWidgets::IconType, std::pair<int, const char*>>>> terrain_tools = {
                { "TerrainRaise", { UIWidgets::IconType::DrawTool, { 0, "Raise\nSculpt terrain height upward." } } },
                { "TerrainLower", { UIWidgets::IconType::ScrapeTool, { 1, "Lower\nSculpt terrain height downward." } } },
                { "TerrainFlatten", { UIWidgets::IconType::FlattenTool, { 2, "Flatten\nLevel terrain heights toward a target altitude." } } },
                { "TerrainSmooth", { UIWidgets::IconType::SmoothTool, { 3, "Smooth\nSoften sharp terrain peaks and valleys." } } },
                { "TerrainStamp", { UIWidgets::IconType::StampTool, { 4, "Stamp\nApply a heightmap texture stamp to the terrain." } } }
            };

            const float spacing = 3.0f;
            const float tool_size = 44.0f;
            const float avail_w = ImGui::GetContentRegionAvail().x;
            const int columns = std::max(1, static_cast<int>((avail_w + spacing) / (tool_size + spacing)));
            int col_idx = 0;

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, spacing));
            for (const auto& t : terrain_tools) {
                if (col_idx > 0 && (col_idx % columns) != 0) {
                    ImGui::SameLine(0.0f, spacing);
                }
                const char* id = t.first;
                UIWidgets::IconType icon = t.second.first;
                int mode = t.second.second.first;
                const char* tooltip = t.second.second.second;
                if (UIWidgets::IconActionButton(id, icon, "",
                                                terrain_brush.mode == mode, ImVec4(0.22f, 0.55f, 0.88f, 1.0f),
                                                ImVec2(tool_size, tool_size), tooltip, true)) {
                    terrain_brush.mode = mode;
                }
                col_idx++;
            }
            ImGui::PopStyleVar();
            UIWidgets::PopControlSurfaceStyle();
            return;
        }

        if (!rightDockOnly) {
            if (beginBrushDockSection("Stroke")) {
                ImGui::TextDisabled("Only terrain-safe brush controls are active here");
                ImGui::SliderFloat("Radius", &terrain_brush.radius, 1.0f, 200.0f, "%.1f m");
                ImGui::SliderFloat("Strength", &terrain_brush.strength, 0.01f, 500.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Curve", &terrain_brush.curve, 0.25f, 4.0f, "%.2f");
                terrain_brush.show_preview = sculpt_mode_state.brush.show_preview;
                if (ImGui::Checkbox("Show Brush Preview", &terrain_brush.show_preview)) {
                    sculpt_mode_state.brush.show_preview = terrain_brush.show_preview;
                }
            }

            if (terrain_brush.mode == 2 && beginBrushDockSection("Flatten##terrain_sculpt_section")) {
                ImGui::Checkbox("Use Fixed Height", &terrain_brush.use_fixed_height);
                if (terrain_brush.use_fixed_height) {
                    ImGui::DragFloat("Altitude", &terrain_brush.flatten_target, 0.1f, -1000.0f, 5000.0f, "%.1f m");
                } else {
                    ImGui::TextDisabled("Samples the initial click height.");
                }
            }

            if (terrain_brush.mode == 4 && beginBrushDockSection("Stamp##terrain_sculpt_section")) {
                ImGui::SliderFloat("Rotation", &terrain_brush.stamp_rotation, 0.0f, 360.0f, "%.0f deg");
                if (UIWidgets::IconActionButton("LoadTerrainStamp", UIWidgets::IconType::StampTool, "Load Stamp",
                                                false, ImVec4(0.22f, 0.55f, 0.88f, 1.0f), ImVec2(140.0f, 34.0f),
                                                "Load terrain stamp texture")) {
                    std::string path = SceneUI::openFileDialogW(L"Image Files\0*.png;*.jpg;*.jpeg;*.bmp\0");
                    if (!path.empty()) {
                        terrain_brush.stamp_texture = std::make_shared<Texture>(path, TextureType::Albedo);
                    }
                }
                if (terrain_brush.stamp_texture) {
                    ImGui::TextDisabled("Stamp: %s", std::filesystem::path(terrain_brush.stamp_texture->name).filename().string().c_str());
                }
            }
        }

        if (!rightDockOnly) {
            ImGui::PopItemWidth();
        }
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    // Mesh Sculpting Tools (Deform / Clay / Polish)
    if (rightDockOnly) {
        struct ToolInfo {
            const char* id;
            const char* tooltip;
            SceneUI::SculptBrushTool tool;
        };

        const std::vector<ToolInfo> sculpt_tools = {
            { "SculptDraw", "Draw\nBuilds or digs along the surface normal.", SceneUI::SculptBrushTool::Draw },
            { "SculptClay", "Clay\nAdds soft clay-like buildup.", SceneUI::SculptBrushTool::Clay },
            { "SculptClayStrips", "Clay Strips\nLays down ribbon-like clay strokes.", SceneUI::SculptBrushTool::ClayStrips },
            { "SculptInflate", "Inflate\nExpands volume outward or inward.", SceneUI::SculptBrushTool::Inflate },
            { "SculptCrease", "Crease\nCarves a sharp groove and pinch.", SceneUI::SculptBrushTool::Crease },
            { "SculptPinch", "Pinch\nPulls vertices inward to tighten forms.", SceneUI::SculptBrushTool::Pinch },
            { "SculptGrab", "Grab\nPulls the surface by dragging points.", SceneUI::SculptBrushTool::Grab },
            { "SculptLayer", "Layer\nBuilds toward a capped height.", SceneUI::SculptBrushTool::Layer },
            { "SculptSmooth", "Smooth\nRelaxes and evens surface detail.", SceneUI::SculptBrushTool::Smooth },
            { "SculptFlatten", "Flatten\nLevels peaks toward a plane.", SceneUI::SculptBrushTool::Flatten },
            { "SculptScrape", "Scrape\nTrims high spots for hard-surface shaping.", SceneUI::SculptBrushTool::Scrape },
            { "SculptDrawSharp", "Draw Sharp\nCrisp ridges and creases with a tight falloff.", SceneUI::SculptBrushTool::DrawSharp },
            // Nudge temporarily hidden from the UI alongside Snake Hook (stroke-
            // direction push not clean enough yet). Tool code/enum/icon stay live.
            // { "SculptNudge", "Nudge\nPushes the surface along the stroke direction.", SceneUI::SculptBrushTool::Nudge },
            { "SculptBlob", "Blob\nSpherical swell; builds rounded bulges.", SceneUI::SculptBrushTool::Blob },
            { "SculptFill", "Fill\nFills valleys toward a plane (Ctrl deepens).", SceneUI::SculptBrushTool::Fill },
            // Snake Hook temporarily hidden from the UI again (2026-06-16): the tip still
            // grabs verts BEHIND the touched surface (dynamic capsule-sweep prime scans the
            // back face) and its follow isn't clean. Tool code/enum/icon all stay live, so
            // re-enabling is a one-line restore once the back-face capture is fixed.
            // { "SculptSnakeHook", "Snake Hook\nDrags the surface into hooks and tentacles.", SceneUI::SculptBrushTool::SnakeHook },
            { "SculptElastic", "Elastic Deform\nSoft, wide grab-style pull.", SceneUI::SculptBrushTool::ElasticDeform },
            { "SculptMask", "Mask\nPaints a protection weight; masked areas resist every brush.", SceneUI::SculptBrushTool::Mask }
        };

        const float spacing = 3.0f;
        const float avail_w = ImGui::GetContentRegionAvail().x;

        if (avail_w < 160.0f) {
            // Icon Grid Mode
            const float tool_size = 40.0f;
            const int columns = std::max(1, static_cast<int>((avail_w + spacing) / (tool_size + spacing)));
            int col_idx = 0;

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, spacing));
            for (const auto& t : sculpt_tools) {
                if (col_idx > 0 && (col_idx % columns) != 0) {
                    ImGui::SameLine(0.0f, spacing);
                }
                drawSculptToolSelectorButton(t.id, t.tooltip, t.tool, sculpt_mode_state.tool, tool_size, tool_size);
                col_idx++;
            }
            ImGui::PopStyleVar();
        } else {
            // Detailed List Mode (Blender style)
            const float icon_size = 36.0f;
            const float row_height = icon_size + 6.0f;

            for (const auto& t : sculpt_tools) {
                const bool is_active = (sculpt_mode_state.tool == t.tool);
                ImGui::PushID(t.id);
                ImGui::BeginGroup();

                ImVec2 pos = ImGui::GetCursorScreenPos();
                float row_width = ImGui::GetContentRegionAvail().x;

                // Invisible button to capture clicks on the row
                if (ImGui::InvisibleButton("##row_btn", ImVec2(row_width, row_height))) {
                    sculpt_mode_state.tool = t.tool;
                }
                const bool is_hovered = ImGui::IsItemHovered();

                ImDrawList* dl = ImGui::GetWindowDrawList();
                if (is_active) {
                    dl->AddRectFilled(pos, ImVec2(pos.x + row_width, pos.y + row_height), ImGui::ColorConvertFloat4ToU32(ImVec4(0.22f, 0.55f, 0.88f, 0.22f)), 4.0f);
                    dl->AddRect(pos, ImVec2(pos.x + row_width, pos.y + row_height), ImGui::ColorConvertFloat4ToU32(ImVec4(0.22f, 0.55f, 0.88f, 0.45f)), 4.0f, 0, 1.0f);
                } else if (is_hovered) {
                    dl->AddRectFilled(pos, ImVec2(pos.x + row_width, pos.y + row_height), ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.05f)), 4.0f);
                }

                // Draw Icon
                ImVec2 icon_pos = ImVec2(pos.x + 3.0f, pos.y + 3.0f);
                UIWidgets::DrawIcon(
                    getSculptToolIcon(t.tool),
                    icon_pos,
                    icon_size,
                    ImGui::ColorConvertFloat4ToU32(is_active ? ImVec4(0.38f, 0.82f, 1.0f, 1.0f) : ImVec4(0.8f, 0.8f, 0.8f, 1.0f)),
                    1.5f
                );

                // Parse tooltip to extract name and short description
                std::string tooltip_str(t.tooltip);
                size_t newline_pos = tooltip_str.find('\n');
                std::string name = (newline_pos != std::string::npos) ? tooltip_str.substr(0, newline_pos) : tooltip_str;
                std::string desc = (newline_pos != std::string::npos) ? tooltip_str.substr(newline_pos + 1) : "";

                // Draw Text
                float text_x = pos.x + icon_size + 10.0f;
                dl->AddText(ImVec2(text_x, pos.y + 3.0f), ImGui::ColorConvertFloat4ToU32(is_active ? ImVec4(1.0f, 0.8f, 0.4f, 1.0f) : ImVec4(0.9f, 0.9f, 0.9f, 1.0f)), name.c_str());
                if (!desc.empty()) {
                    if (desc.size() > 40) {
                        desc = desc.substr(0, 37) + "...";
                    }
                    dl->AddText(ImVec2(text_x, pos.y + 20.0f), ImGui::ColorConvertFloat4ToU32(ImVec4(0.55f, 0.55f, 0.55f, 1.0f)), desc.c_str());
                }

                ImGui::EndGroup();
                ImGui::PopID();
                ImGui::Spacing();
            }
        }

        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    if (!rightDockOnly) {
        if (beginBrushDockSection("Brush Properties")) {
            ImGui::Checkbox("Screen Space Radius", &sculpt_mode_state.use_screen_space_radius);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Toggle screen radius in pixels vs world radius in meters.");
            }
            if (sculpt_mode_state.use_screen_space_radius) {
                ImGui::SliderFloat("Radius", &sculpt_mode_state.screen_radius_px, 8.0f, 240.0f, "%.0f px");
            } else {
                ImGui::SliderFloat("Radius", &sculpt_mode_state.brush.radius, 0.01f, 20.0f, "%.3f m");
            }
            ImGui::SliderFloat("Strength", &sculpt_mode_state.brush.strength, 0.01f, 10.0f, "%.2f");
            ImGui::SliderFloat("Falloff", &sculpt_mode_state.brush.falloff, 0.0f, 1.0f, "%.2f");
            if (!sculpt_mode_state.compact_ui) {
                ImGui::SliderFloat("Spacing", &sculpt_mode_state.brush.spacing, 0.01f, 1.0f, "%.2f");
                ImGui::SliderFloat("Flow", &sculpt_mode_state.brush.flow, 0.1f, 2.0f, "%.2f");
            }
            ImGui::Checkbox("Show Brush Preview", &sculpt_mode_state.brush.show_preview);
            ImGui::Checkbox("Front Faces Only", &sculpt_mode_state.front_faces_only);
            ImGui::Checkbox("Accumulate Live", &sculpt_mode_state.accumulate_live);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("When off, sculpt updates stay local during the stroke and GPU/render sync waits for mouse release.");
            }
            
            ImGui::Separator();
            ImGui::TextDisabled("Symmetry Options");
            ImGui::Checkbox("Mirror X##sculpt", &sculpt_mode_state.mirror_x); ImGui::SameLine();
            ImGui::Checkbox("Mirror Y##sculpt", &sculpt_mode_state.mirror_y); ImGui::SameLine();
            ImGui::Checkbox("Mirror Z##sculpt", &sculpt_mode_state.mirror_z);

            ImGui::Checkbox("Radial##sculpt", &sculpt_mode_state.radial_symmetry);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Repeat the stroke as N rotated copies around an object-local axis.");
            }
            if (sculpt_mode_state.radial_symmetry) {
                ImGui::SetNextItemWidth(120.0f);
                ImGui::SliderInt("Count##radial", &sculpt_mode_state.radial_count, 2, 16);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(90.0f);
                const char* radialAxes[] = { "X", "Y", "Z" };
                ImGui::Combo("Axis##radial", &sculpt_mode_state.radial_axis, radialAxes, IM_ARRAYSIZE(radialAxes));
            }
        }

        if (beginBrushDockSection("Mask")) {
            ImGui::TextDisabled("Masked areas resist every brush.");
            ImGui::SliderFloat("Mask Strength", &sculpt_mask_state.paint_strength, 0.01f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Deposit per stroke for the Mask brush. Hold Ctrl while painting to erase.");
            }
            ImGui::Checkbox("Show Mask Overlay", &sculpt_mask_state.show_overlay);

            const float fullW = ImGui::GetContentRegionAvail().x;
            const float halfW = (fullW - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
            if (ImGui::Button("Clear##mask", ImVec2(halfW, 0.0f))) {
                applySculptMaskOperation(0);
            }
            ImGui::SameLine();
            if (ImGui::Button("Invert##mask", ImVec2(halfW, 0.0f))) {
                applySculptMaskOperation(1);
            }
            if (ImGui::Button("Fill##mask", ImVec2(halfW, 0.0f))) {
                applySculptMaskOperation(2);
            }
            ImGui::SameLine();
            if (ImGui::Button("Smooth##mask", ImVec2(halfW, 0.0f))) {
                applySculptMaskOperation(3);
            }
            if (ImGui::Button("Sharpen##mask", ImVec2(fullW, 0.0f))) {
                applySculptMaskOperation(4);
            }
        }

        if (beginBrushDockSection("Dynamic Clay")) {
            ImGui::Checkbox("Wet Clay", &sculpt_mode_state.wet_clay_enabled);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Additive brushes deposit WET clay: it keeps settling toward the\n"
                                  "surrounding surface after the stroke, then dries and locks.\n"
                                  "Off = instant, rigid deposits (classic behaviour).");
            }
            if (sculpt_mode_state.wet_clay_enabled) {
                // Presets bundle the knobs into recognizable materials. Tweak any slider
                // afterwards to fine-tune. (wetness, dry, settle, flow, yield, hetero, cohesion)
                auto applyWetPreset = [&](float wetness, float dry, float settleV, float flowV,
                                          float yieldV, bool het, float coh) {
                    sculpt_mode_state.wet_clay_wetness = wetness;
                    sculpt_mode_state.wet_clay_dry_rate = dry;
                    sculpt_mode_state.wet_clay_settle = settleV;
                    sculpt_mode_state.wet_clay_flow = flowV;
                    sculpt_mode_state.wet_clay_yield = yieldV;
                    sculpt_mode_state.wet_clay_hetero = het;
                    sculpt_mode_state.wet_clay_cohesion = coh;
                };
                ImGui::TextDisabled("Preset:");
                ImGui::SameLine();
                if (ImGui::SmallButton("Clay"))  applyWetPreset(0.70f, 0.50f, 0.60f, 0.40f, 0.15f, false, 0.60f);
                ImGui::SameLine();
                if (ImGui::SmallButton("Water")) applyWetPreset(0.90f, 0.20f, 0.20f, 1.00f, 0.00f, false, 0.20f);
                ImGui::SameLine();
                if (ImGui::SmallButton("Mud"))   applyWetPreset(0.80f, 0.40f, 0.30f, 0.70f, 0.25f, true,  0.40f);
                ImGui::SameLine();
                if (ImGui::SmallButton("Putty")) applyWetPreset(0.60f, 0.70f, 0.25f, 0.35f, 0.50f, false, 0.85f);

                ImGui::SliderFloat("Wetness", &sculpt_mode_state.wet_clay_wetness, 0.0f, 1.0f, "%.2f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("How soft fresh clay starts. Higher = settles/flows more before it dries.");
                }
                ImGui::SliderFloat("Dry Rate", &sculpt_mode_state.wet_clay_dry_rate, 0.05f, 4.0f, "%.2f /s");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("How fast the clay sets. Higher = locks sooner.");
                }
                ImGui::SliderFloat("Settle", &sculpt_mode_state.wet_clay_settle, 0.0f, 1.0f, "%.2f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Surface tension: how strongly wet clay relaxes toward the surrounding surface.");
                }
                ImGui::SliderFloat("Cohesion", &sculpt_mode_state.wet_clay_cohesion, 0.0f, 1.0f, "%.2f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Viscosity of the FLOWING clay. High = smooth bonded tongue (putty);\n"
                                      "low = rough, breaks into chunks (mud, esp. with Heterogeneous Density).");
                }
                ImGui::SliderFloat("Gravity Flow", &sculpt_mode_state.wet_clay_flow, 0.0f, 1.0f, "%.2f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Wet clay descends as a body along world-down, creeping down\n"
                                      "slopes and pooling in low spots. 0 = no gravity.");
                }
                if (sculpt_mode_state.wet_clay_flow > 0.0f) {
                    ImGui::SliderFloat("Yield", &sculpt_mode_state.wet_clay_yield, 0.0f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("How steep a slope must be before clay flows. Higher = holds its\n"
                                          "shape more (viscous clay); 0 = runs freely like water.");
                    }
                    ImGui::Checkbox("Heterogeneous Density", &sculpt_mode_state.wet_clay_hetero);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Vary flow speed across the surface (a density field) so the mud\n"
                                          "creeps down unevenly / marbled. Off = uniform sheet.");
                    }
                    if (sculpt_mode_state.wet_clay_hetero) {
                        ImGui::SliderFloat("Density Scale", &sculpt_mode_state.wet_clay_hetero_scale, 0.2f, 8.0f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("Density-noise frequency: higher = finer marbling.");
                        }
                    }
                }
                ImGui::Checkbox("Water Only (re-wet, no deposit)", &sculpt_mode_state.wet_clay_water_only);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Paint wetness onto existing geometry WITHOUT adding clay, so dried\n"
                                      "areas soften and settle/flow again (like adding water in pottery).");
                }
                ImGui::TextDisabled("Active wet verts: %zu", sculpt_wet_clay_state.active_list.size());
            }
        }

        if (beginBrushDockSection("Brush Behavior")) {
            ImGui::SliderFloat("Normal Strength", &sculpt_mode_state.normal_strength, 0.0f, 2.0f, "%.2f");
            static const char* falloffTypes[] = { "Smooth", "Linear", "Sharp", "Sphere", "Root", "Custom" };
            ImGui::Combo("Falloff Type", &mesh_overlay_settings.proportional_falloff_type, falloffTypes, IM_ARRAYSIZE(falloffTypes));
            if (mesh_overlay_settings.proportional_falloff_type == 5) {
                drawFalloffCurveEditor(mesh_overlay_settings.custom_falloff_lut);
                if (ImGui::SmallButton("Smooth##fcv")) setFalloffLutPreset(mesh_overlay_settings.custom_falloff_lut, 0);
                ImGui::SameLine();
                if (ImGui::SmallButton("Linear##fcv")) setFalloffLutPreset(mesh_overlay_settings.custom_falloff_lut, 1);
                ImGui::SameLine();
                if (ImGui::SmallButton("Sharp##fcv")) setFalloffLutPreset(mesh_overlay_settings.custom_falloff_lut, 2);
                ImGui::SameLine();
                if (ImGui::SmallButton("Root##fcv")) setFalloffLutPreset(mesh_overlay_settings.custom_falloff_lut, 4);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Drag inside the box to reshape the brush falloff curve.");
                }
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Stylized Tooltip Info Box
            ImVec4 hint_bg = ImVec4(0.12f, 0.14f, 0.16f, 0.8f);
            float hint_round = 4.0f;
            if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
                const auto& curTheme = ThemeManager::instance().current();
                hint_bg = curTheme.colors.surface;
                hint_round = curTheme.style.windowRounding;
            }
            ImGui::PushStyleColor(ImGuiCol_ChildBg, hint_bg);
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, hint_round);
            if (ImGui::BeginChild("ToolHintBox", ImVec2(0.0f, 96.0f), true, ImGuiWindowFlags_NoScrollbar)) {
                ImGui::TextColored(ImVec4(1.00f, 0.65f, 0.40f, 1.0f), "Brush Tip Info:");
                const char* tool_hint = "This sculpt pass reuses the edit/soft-selection falloff model.";
                switch (sculpt_mode_state.tool) {
                case SculptBrushTool::Clay:
                    tool_hint = "Clay builds a soft layer; hold Ctrl to carve instead of add.";
                    break;
                case SculptBrushTool::Layer:
                    tool_hint = "Layer builds toward a capped height; hold Ctrl to cut down.";
                    break;
                case SculptBrushTool::ClayStrips:
                    tool_hint = "Clay Strips stacks muddy ribbons; hold Ctrl for subtractive cuts.";
                    break;
                case SculptBrushTool::Crease:
                    tool_hint = "Crease combines inward pinch with a cut along the stroke normal.";
                    break;
                case SculptBrushTool::Scrape:
                    tool_hint = "Scrape trims only the peaks above the current stroke plane.";
                    break;
                case SculptBrushTool::Draw:
                    tool_hint = "Draw pushes along the surface normal; hold Ctrl to dig negative relief.";
                    break;
                case SculptBrushTool::Inflate:
                    tool_hint = "Inflate swells volume evenly; hold Ctrl to deflate inward.";
                    break;
                case SculptBrushTool::Grab:
                    tool_hint = "Grab pulls the surface by dragging points along the screen plane.";
                    break;
                case SculptBrushTool::Pinch:
                    tool_hint = "Pinch draws vertices toward the center of the brush cursor.";
                    break;
                case SculptBrushTool::Smooth:
                    tool_hint = "Smooth relaxes and averages local surface detail.";
                    break;
                case SculptBrushTool::Flatten:
                    tool_hint = "Flatten levels peaks and valleys toward a local plane.";
                    break;
                case SculptBrushTool::Mask:
                    tool_hint = "Mask paints a protection weight (cool tint); masked areas resist every other brush. Hold Ctrl to erase the mask.";
                    break;
                case SculptBrushTool::DrawSharp:
                    tool_hint = "Draw Sharp cuts crisp ridges/creases along the normal; hold Ctrl to incise inward.";
                    break;
                case SculptBrushTool::Nudge:
                    tool_hint = "Nudge pushes the surface tangentially along the stroke direction.";
                    break;
                case SculptBrushTool::Blob:
                    tool_hint = "Blob swells a rounded bulge; hold Ctrl to contract.";
                    break;
                case SculptBrushTool::Fill:
                    tool_hint = "Fill raises valleys toward the brush plane; hold Ctrl to deepen instead.";
                    break;
                case SculptBrushTool::SnakeHook:
                    tool_hint = "Snake Hook drags the surface along the stroke into hooks and tentacles.";
                    break;
                case SculptBrushTool::ElasticDeform:
                    tool_hint = "Elastic Deform is a soft, wide grab-style pull (drag to move the surface).";
                    break;
                default:
                    break;
                }
                ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x - 4.0f);
                ImGui::TextUnformatted(tool_hint);
                ImGui::Spacing();
                ImGui::TextDisabled("Shortcuts: Shift = Smooth, Ctrl = Invert Brush");
                ImGui::PopTextWrapPos();
            }
            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
        }

        if (beginBrushDockSection("Alpha Mask")) {
            int alpha_preset = static_cast<int>(sculpt_mode_state.brush.alpha_preset);
            const char* alpha_labels[] = { "Soft Round", "Hard Round", "Noise", "Scratch", "Cloud" };
            if (ImGui::Combo("Shape##sculpt", &alpha_preset, alpha_labels, IM_ARRAYSIZE(alpha_labels))) {
                sculpt_mode_state.brush.alpha_preset = static_cast<Paint::BrushAlphaPreset>(alpha_preset);
                sculpt_mode_state.brush.use_imported_alpha = false;
            }
            
            const float btnW = ImGui::GetContentRegionAvail().x;
            if (ImGui::Button("Load Custom Alpha Texture##sculpt", ImVec2(btnW, 30.0f))) {
                const std::string path = SceneUI::openFileDialogW(
                    L"Image Files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.exr;*.hdr\0",
                    "",
                    "");
                if (!path.empty()) {
                    auto tex = std::make_shared<Texture>(path, TextureType::Unknown);
                    if (tex && tex->is_loaded()) {
                        sculpt_mode_state.brush.alpha_texture = tex;
                        sculpt_mode_state.brush.alpha_texture_path = path;
                        sculpt_mode_state.brush.use_imported_alpha = true;
                    }
                }
            }
            if (sculpt_mode_state.brush.alpha_texture && sculpt_mode_state.brush.alpha_texture->is_loaded()) {
                if (ImGui::Button("Clear Loaded Alpha##sculpt_alpha", ImVec2(btnW, 26.0f))) {
                    sculpt_mode_state.brush.alpha_texture.reset();
                    sculpt_mode_state.brush.alpha_texture_path.clear();
                    sculpt_mode_state.brush.use_imported_alpha = false;
                }
                ImGui::Checkbox("Use Loaded Alpha##sculpt", &sculpt_mode_state.brush.use_imported_alpha);
                const std::string alpha_name = brushAlphaDisplayName(sculpt_mode_state.brush.alpha_texture_path);
                ImGui::TextDisabled("Texture: %s", alpha_name.empty() ? "Untitled" : alpha_name.c_str());
            }
            if (sculpt_mode_state.brush.alpha_preset == Paint::BrushAlphaPreset::Noise ||
                sculpt_mode_state.brush.alpha_preset == Paint::BrushAlphaPreset::Scratch ||
                sculpt_mode_state.brush.alpha_preset == Paint::BrushAlphaPreset::Cloud ||
                sculpt_mode_state.brush.use_imported_alpha) {
                ImGui::SliderFloat("Alpha Scale##sculpt", &sculpt_mode_state.brush.alpha_scale, 0.25f, 8.0f, "%.2f");
            }
            ImGui::SliderFloat("Alpha Rotation##sculpt", &sculpt_mode_state.brush.alpha_rotation_degrees, -180.0f, 180.0f, "%.0f deg");
            
            ImGui::Spacing();
            drawBrushAlphaPreview(sculpt_mode_state.brush);
        }

        ImGui::PopItemWidth();
    }

    UIWidgets::PopControlSurfaceStyle();
}

void SceneUI::drawEditToolControls(UIContext& ctx, const std::shared_ptr<Triangle>& meshTriangle, bool rightDockOnly) {
    UIWidgets::PushControlSurfaceStyle(ImVec4(0.38f, 0.72f, 0.92f, 1.0f));

    if (!meshTriangle) {
        if (!rightDockOnly) {
            ImGui::TextDisabled("Select a mesh to configure edit tools.");
        }
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    bool hasSelection = (ctx.selection.selected.type == SelectableType::Object &&
                         ctx.selection.selected.object != nullptr);
    const std::string selectedNodeName =
        hasSelection ? ctx.selection.selected.object->getNodeName() : std::string{};
    const std::string effectiveNodeName =
        !active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name : selectedNodeName;

    const bool isVertexMode = (ctx.selection.mesh_element_mode == MeshElementSelectMode::Vertex);
    const bool isEdgeMode = (ctx.selection.mesh_element_mode == MeshElementSelectMode::Edge);
    const bool isFaceMode = (ctx.selection.mesh_element_mode == MeshElementSelectMode::Face);
    const bool isCombinedMode = (ctx.selection.mesh_element_mode == MeshElementSelectMode::Combined);

    auto activateMeshSelectMode = [&](MeshElementSelectMode mode) {
        mesh_overlay_settings.edit_mode = true;
        mesh_overlay_settings.enabled = true;
        sculpt_mode_state.enabled = false;
        ctx.selection.mesh_element_mode = mode;
        clearEditableMeshSelection();
        active_mesh_edit_object_name = effectiveNodeName;
        active_mesh_edit_object_ptr =
            (hasSelection && selectedNodeName == effectiveNodeName) ? ctx.selection.selected.object.get() : nullptr;
        ensureMeshEditLayer(ctx, effectiveNodeName);
    };

    const size_t selectedVertexCount = editable_mesh_cache.selection.vertex_ids.size();
    const bool hasVertexTools = (isVertexMode || isCombinedMode) && !effectiveNodeName.empty() && selectedVertexCount >= 2;
    const bool canAddFace = hasVertexTools && selectedVertexCount >= 3;
    const bool canMergeVertices = hasVertexTools;
    const bool canWeldVertices = hasVertexTools;
    const bool hasSelectedFaces = (isFaceMode || isCombinedMode) && !editable_mesh_cache.selection.face_ids.empty() && !effectiveNodeName.empty();
    const bool hasSelectedEdges = (isEdgeMode || isCombinedMode) && !editable_mesh_cache.selection.edge_ids.empty() && !effectiveNodeName.empty();

    auto triggerAction = [&](int action_type, UIContext& ctx) {
        if (action_type == 0) {
            addFaceFromSelectedVertices(ctx);
        } else if (action_type == 1) {
            extrudeSelectedMeshFaces(ctx, mesh_face_extrude_distance);
        } else if (action_type == 2) {
            insetSelectedMeshFaces(ctx, mesh_face_inset_amount);
        } else if (action_type == 3) {
            loopCutSelectedEdges(ctx, mesh_loop_cut_position);
        } else if (action_type == 4) {
            mergeSelectedVerticesToCenter(ctx);
        } else if (action_type == 5) {
            weldSelectedVerticesByDistance(ctx, mesh_vertex_weld_distance);
        } else if (action_type == 6) {
            dissolveSelectedVertices(ctx);
        } else if (action_type == 7) {
            dissolveSelectedEdges(ctx);
        } else if (action_type == 8) {
            deleteSelectedMeshFaces(ctx);
        } else if (action_type == 9) {
            auto& shading = ensureMeshShadingSettings(effectiveNodeName);
            shading.flat_shading = true;
            shading.auto_smooth = false;
            applyMeshShadingSettings(ctx, effectiveNodeName);
        } else if (action_type == 10) {
            auto& shading = ensureMeshShadingSettings(effectiveNodeName);
            shading.flat_shading = false;
            shading.auto_smooth = false;
            applyMeshShadingSettings(ctx, effectiveNodeName);
        } else if (action_type == 11) {
            flipSelectedMeshNormals(ctx);
        } else if (action_type == 12) {
            recalculateMeshNormals(ctx, true);
        } else if (action_type == 13) {
            recalculateMeshNormals(ctx, false);
        }
    };

    struct EditToolInfo {
        const char* id;
        const char* label;
        const char* tooltip;
        const char* disabled_tooltip;
        UIWidgets::IconType icon;
        ImVec4 accent;
        bool enabled;
        int action_type;
    };

    std::vector<EditToolInfo> edit_actions = {
        { "VertexAddFace", "Add Face", "Add Face\nCreate a triangle, quad, or n-gon from selected vertices.", "Disabled: Requires at least 3 selected vertices.", UIWidgets::IconType::AddFace, ImVec4(0.86f, 0.78f, 0.36f, 1.0f), canAddFace, 0 },
        { "FaceExtrude", "Extrude Face", "Extrude Face\nPush selected faces along their normals.", "Disabled: Requires selected face(s).", UIWidgets::IconType::ExtrudeFaceTool, ImVec4(0.42f, 0.78f, 1.0f, 1.0f), hasSelectedFaces, 1 },
        { "FaceInset", "Inset Face", "Inset Face\nShrink selected faces inward to leave a border ring.", "Disabled: Requires selected face(s).", UIWidgets::IconType::FaceMode, ImVec4(0.52f, 0.86f, 0.62f, 1.0f), hasSelectedFaces, 2 },
        { "EdgeLoopCut", "Loop Cut", "Loop Cut\nInsert a new edge strip across the selected ring.", "Disabled: Requires selected edge(s).", UIWidgets::IconType::LoopCutTool, ImVec4(0.36f, 0.84f, 0.82f, 1.0f), hasSelectedEdges, 3 },
        { "VertexMerge", "Merge Center", "Merge Center\nCollapse selected vertices to their center point.", "Disabled: Requires at least 2 selected vertices.", UIWidgets::IconType::MergeVertices, ImVec4(0.78f, 0.70f, 1.0f, 1.0f), canMergeVertices, 4 },
        { "VertexWeldByDistance", "Weld Distance", "Weld Distance\nSnap nearby selected vertices together using the weld distance.", "Disabled: Requires at least 2 selected vertices.", UIWidgets::IconType::WeldVertices, ImVec4(0.72f, 0.84f, 1.0f, 1.0f), canWeldVertices, 5 },
        { "VertexDissolve", "Dissolve Vert", "Dissolve Vert\nRemove selected vertices, preserving surrounding faces.", "Disabled: Requires selected vertex/vertices.", UIWidgets::IconType::DissolveTopology, ImVec4(1.0f, 0.66f, 0.54f, 1.0f), (isVertexMode || isCombinedMode) && selectedVertexCount >= 1, 6 },
        { "EdgeDissolve", "Dissolve Edge", "Dissolve Edge\nRemove selected edges, preserving polygon flow.", "Disabled: Requires selected edge(s).", UIWidgets::IconType::DissolveTopology, ImVec4(1.0f, 0.68f, 0.52f, 1.0f), hasSelectedEdges, 7 },
        { "FaceDelete", "Delete Face", "Delete Face\nRemove selected faces, leaving an open boundary.", "Disabled: Requires selected face(s).", UIWidgets::IconType::DeleteFaceTool, ImVec4(1.0f, 0.48f, 0.42f, 1.0f), hasSelectedFaces, 8 }
    };



    if (rightDockOnly) {
        const float avail_w = ImGui::GetContentRegionAvail().x;
        const float spacing = 3.0f;

        if (avail_w < 160.0f) {
            // Icon Grid Mode
            const float tool_size = 40.0f;
            const int columns = std::max(1, static_cast<int>((avail_w + spacing) / (tool_size + spacing)));
            int col_idx = 0;

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, spacing));

            // Selection Modes
            struct SelModeInfo {
                const char* id;
                const char* tooltip;
                MeshElementSelectMode mode;
                bool active;
                ImVec4 accent;
            };
            std::vector<SelModeInfo> sel_modes = {
                { "MeshSelectVertex", "Vertex Mode\nSelect and edit control cage points.", MeshElementSelectMode::Vertex, isVertexMode, ImVec4(0.94f, 0.76f, 0.26f, 1.0f) },
                { "MeshSelectEdge", "Edge Mode\nSelect strips, loops, cuts, and dissolves.", MeshElementSelectMode::Edge, isEdgeMode, ImVec4(0.30f, 0.82f, 0.78f, 1.0f) },
                { "MeshSelectFace", "Face Mode\nSelect polygons for extrusion and deletion.", MeshElementSelectMode::Face, isFaceMode, ImVec4(0.38f, 0.72f, 0.92f, 1.0f) },
                { "MeshSelectCombined", "Combined Mode\nSelect vertices, edges, and faces simultaneously.", MeshElementSelectMode::Combined, isCombinedMode, ImVec4(0.82f, 0.66f, 0.96f, 1.0f) }
            };

            for (const auto& s : sel_modes) {
                if (col_idx > 0 && (col_idx % columns) != 0) {
                    ImGui::SameLine(0.0f, spacing);
                }
                if (UIWidgets::IconActionButton(s.id, getMeshSelectModeIcon(s.mode), "", s.active, s.accent, ImVec2(tool_size, tool_size), s.tooltip, true)) {
                    activateMeshSelectMode(s.mode);
                }
                col_idx++;
            }

            ImGui::PopStyleVar();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, spacing));

            col_idx = 0;
            for (const auto& action : edit_actions) {
                bool active = false;
                if (action.action_type == 9) {
                    auto& shading = ensureMeshShadingSettings(effectiveNodeName);
                    active = shading.flat_shading && !shading.auto_smooth;
                } else if (action.action_type == 10) {
                    auto& shading = ensureMeshShadingSettings(effectiveNodeName);
                    active = !shading.flat_shading && !shading.auto_smooth;
                }

                if (col_idx > 0 && (col_idx % columns) != 0) {
                    ImGui::SameLine(0.0f, spacing);
                }

                if (UIWidgets::IconActionButton(
                        action.id,
                        action.icon,
                        "",
                        active,
                        action.accent,
                        ImVec2(tool_size, tool_size),
                        "",
                        action.enabled)) {
                    triggerAction(action.action_type, ctx);
                }
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("%s", action.enabled ? action.tooltip : action.disabled_tooltip);
                }
                col_idx++;
            }
            ImGui::PopStyleVar();
        } else {
            // Detailed List Mode (Blender style)
            const float icon_size = 36.0f;
            const float row_height = icon_size + 6.0f;

            ImGui::TextColored(ImVec4(0.72f, 0.84f, 1.0f, 1.0f), "Selection Modes");
            ImGui::Separator();
            ImGui::Spacing();

            struct SelModeInfo {
                const char* id;
                const char* name;
                const char* desc;
                MeshElementSelectMode mode;
                bool active;
                ImVec4 accent;
            };
            std::vector<SelModeInfo> sel_modes = {
                { "MeshSelectVertex", "Vertex Mode", "Select and edit vertices.", MeshElementSelectMode::Vertex, isVertexMode, ImVec4(0.94f, 0.76f, 0.26f, 1.0f) },
                { "MeshSelectEdge", "Edge Mode", "Select edges, cuts, dissolves.", MeshElementSelectMode::Edge, isEdgeMode, ImVec4(0.30f, 0.82f, 0.78f, 1.0f) },
                { "MeshSelectFace", "Face Mode", "Select faces for extrude/delete.", MeshElementSelectMode::Face, isFaceMode, ImVec4(0.38f, 0.72f, 0.92f, 1.0f) },
                { "MeshSelectCombined", "Combined Mode", "Select vertex, edge, face.", MeshElementSelectMode::Combined, isCombinedMode, ImVec4(0.82f, 0.66f, 0.96f, 1.0f) }
            };

            for (const auto& s : sel_modes) {
                ImGui::PushID(s.id);
                ImGui::BeginGroup();

                ImVec2 pos = ImGui::GetCursorScreenPos();
                float row_width = ImGui::GetContentRegionAvail().x;

                if (ImGui::InvisibleButton("##row_btn", ImVec2(row_width, row_height))) {
                    activateMeshSelectMode(s.mode);
                }
                const bool is_hovered = ImGui::IsItemHovered();

                ImDrawList* dl = ImGui::GetWindowDrawList();
                if (s.active) {
                    dl->AddRectFilled(pos, ImVec2(pos.x + row_width, pos.y + row_height), ImGui::ColorConvertFloat4ToU32(ImVec4(s.accent.x, s.accent.y, s.accent.z, 0.22f)), 4.0f);
                    dl->AddRect(pos, ImVec2(pos.x + row_width, pos.y + row_height), ImGui::ColorConvertFloat4ToU32(ImVec4(s.accent.x, s.accent.y, s.accent.z, 0.45f)), 4.0f, 0, 1.0f);
                } else if (is_hovered) {
                    dl->AddRectFilled(pos, ImVec2(pos.x + row_width, pos.y + row_height), ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.05f)), 4.0f);
                }

                ImVec2 icon_pos = ImVec2(pos.x + 3.0f, pos.y + 3.0f);
                UIWidgets::DrawIcon(
                    getMeshSelectModeIcon(s.mode),
                    icon_pos,
                    icon_size,
                    ImGui::ColorConvertFloat4ToU32(s.active ? s.accent : ImVec4(0.8f, 0.8f, 0.8f, 1.0f)),
                    1.5f
                );

                float text_x = pos.x + icon_size + 10.0f;
                dl->AddText(ImVec2(text_x, pos.y + 3.0f), ImGui::ColorConvertFloat4ToU32(s.active ? ImVec4(1.0f, 0.8f, 0.4f, 1.0f) : ImVec4(0.9f, 0.9f, 0.9f, 1.0f)), s.name);
                dl->AddText(ImVec2(text_x, pos.y + 20.0f), ImGui::ColorConvertFloat4ToU32(ImVec4(0.55f, 0.55f, 0.55f, 1.0f)), s.desc);

                ImGui::EndGroup();
                ImGui::PopID();
                ImGui::Spacing();
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.38f, 0.72f, 0.92f, 1.0f), "Geometry Tools");
            ImGui::Separator();
            ImGui::Spacing();

            for (const auto& action : edit_actions) {
                ImGui::PushID(action.id);
                ImGui::BeginGroup();

                ImVec2 pos = ImGui::GetCursorScreenPos();
                float row_width = ImGui::GetContentRegionAvail().x;

                ImGui::BeginDisabled(!action.enabled);
                if (ImGui::InvisibleButton("##row_btn", ImVec2(row_width, row_height))) {
                    triggerAction(action.action_type, ctx);
                }
                const bool is_hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
                ImGui::EndDisabled();

                if (is_hovered) {
                    ImGui::SetTooltip("%s", action.enabled ? action.tooltip : action.disabled_tooltip);
                }

                bool active = false;
                if (action.action_type == 9) {
                    auto& shading = ensureMeshShadingSettings(effectiveNodeName);
                    active = shading.flat_shading && !shading.auto_smooth;
                } else if (action.action_type == 10) {
                    auto& shading = ensureMeshShadingSettings(effectiveNodeName);
                    active = !shading.flat_shading && !shading.auto_smooth;
                }

                ImDrawList* dl = ImGui::GetWindowDrawList();
                if (active) {
                    dl->AddRectFilled(pos, ImVec2(pos.x + row_width, pos.y + row_height), ImGui::ColorConvertFloat4ToU32(ImVec4(action.accent.x, action.accent.y, action.accent.z, 0.22f)), 4.0f);
                    dl->AddRect(pos, ImVec2(pos.x + row_width, pos.y + row_height), ImGui::ColorConvertFloat4ToU32(ImVec4(action.accent.x, action.accent.y, action.accent.z, 0.45f)), 4.0f, 0, 1.0f);
                } else if (is_hovered && action.enabled) {
                    dl->AddRectFilled(pos, ImVec2(pos.x + row_width, pos.y + row_height), ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.05f)), 4.0f);
                }

                ImVec2 icon_pos = ImVec2(pos.x + 3.0f, pos.y + 3.0f);
                UIWidgets::DrawIcon(
                    action.icon,
                    icon_pos,
                    icon_size,
                    ImGui::ColorConvertFloat4ToU32(active ? action.accent : (action.enabled ? ImVec4(0.8f, 0.8f, 0.8f, 1.0f) : ImVec4(0.4f, 0.4f, 0.4f, 1.0f))),
                    1.5f
                );

                std::string tooltip_str(action.tooltip);
                size_t newline_pos = tooltip_str.find('\n');
                std::string name = (newline_pos != std::string::npos) ? tooltip_str.substr(0, newline_pos) : tooltip_str;
                std::string desc = (newline_pos != std::string::npos) ? tooltip_str.substr(newline_pos + 1) : "";

                float text_x = pos.x + icon_size + 10.0f;
                dl->AddText(ImVec2(text_x, pos.y + 3.0f), ImGui::ColorConvertFloat4ToU32(active ? ImVec4(1.0f, 0.8f, 0.4f, 1.0f) : (action.enabled ? ImVec4(0.9f, 0.9f, 0.9f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f))), name.c_str());
                if (!desc.empty()) {
                    if (desc.size() > 40) {
                        desc = desc.substr(0, 37) + "...";
                    }
                    dl->AddText(ImVec2(text_x, pos.y + 20.0f), ImGui::ColorConvertFloat4ToU32(action.enabled ? ImVec4(0.55f, 0.55f, 0.55f, 1.0f) : ImVec4(0.35f, 0.35f, 0.35f, 1.0f)), desc.c_str());
                }

                ImGui::EndGroup();
                ImGui::PopID();
                ImGui::Spacing();
            }
        }
    }

    UIWidgets::PopControlSurfaceStyle();
}

void SceneUI::drawPaintBrushDock(UIContext& ctx) {
    if (!shouldShowPaintBrushDock()) {
        return;
    }



    const bool sculptDock = sculpt_mode_state.enabled &&
        (!sculpt_mode_state.active_target_name.empty() || terrain_sculpt_proxy_active);
    const bool paintDock = paint_mode_state.enabled && paint_mode_state.hasValidTarget();
    const bool editDock = mesh_workspace_mode == SceneUI::MeshWorkspaceMode::Edit &&
                          mesh_overlay_settings.enabled &&
                          mesh_overlay_settings.edit_mode &&
                          !active_mesh_edit_object_name.empty();
    const bool hairDock = (active_properties_tab == 8);
    const bool slimDock = sculptDock || paintDock || editDock || hairDock;

    std::shared_ptr<Triangle> mesh_triangle = slimDock ? resolvePaintMesh(ctx) : resolvePaintMesh(ctx);
    if (!mesh_triangle) {
        auto adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter());
        mesh_triangle = adapter ? adapter->getTriangle() : nullptr;
    }
    if (!mesh_triangle && slimDock &&
        ctx.selection.selected.type == SelectableType::Object &&
        ctx.selection.selected.object) {
        mesh_triangle = ctx.selection.selected.object;
    }

    ImGuiIO& io = ImGui::GetIO();
    const float menu_height = getMainMenuReservedHeight();
    const bool bottom_visible = show_animation_panel || show_scene_log || show_terrain_graph || show_anim_graph || show_asset_browser;
    
    bool bottom_docked = false;
    if (bottom_visible) {
        if (!docking_enabled) {
            bottom_docked = true;
        } else {
            ImGuiID dockspace_id = this->dockspace_id;
            for (const char* name : {"Timeline", "Console", "Terrain Graph", "AnimGraph", "Asset Browser"}) {
                ImGuiWindow* win = ImGui::FindWindowByName(name);
                if (win && win->Active && win->DockNode) {
                    ImGuiDockNode* node = win->DockNode;
                    while (node->ParentNode) {
                        node = node->ParentNode;
                    }
                    if (node->ID == dockspace_id) {
                        bottom_docked = true;
                        break;
                    }
                }
            }
        }
    }

    const float bottom_margin = bottom_docked ? (bottom_panel_height + 24.0f) : 24.0f;
    const float dock_width = getPaintBrushDockWidth();
    const float dock_height = (std::max)(260.0f, io.DisplaySize.y - menu_height - bottom_margin);

    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - dock_width, menu_height), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(dock_width, dock_height), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.97f);
    
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, slimDock ? ImVec2(3.0f, 6.0f) : ImVec2(10.0f, 8.0f));
    if (slimDock) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    }

    static bool s_brush_dock_focused = false;
    static bool was_focused = false;

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings;
    if (!s_brush_dock_focused) {
        flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;
    }
    if (slimDock) {
        flags |= ImGuiWindowFlags_NoTitleBar;
    }

    if (!ImGui::Begin("Brush Dock", nullptr, flags)) {
        if (slimDock) {
            ImGui::PopStyleVar(3);
        } else {
            ImGui::PopStyleVar();
        }
        ImGui::End();
        s_brush_dock_focused = false;
        was_focused = false;
        return;
    }

    bool is_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    if (is_focused && !was_focused) {
        ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
    }
    was_focused = is_focused;
    s_brush_dock_focused = is_focused;

    if (!slimDock) {
        ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.35f, 1.0f), "Brush Dock");
        ImGui::Separator();
    }
    if (sculptDock) {
        drawSculptBrushControls(ctx, mesh_triangle, true);
    } else if (editDock) {
        drawEditToolControls(ctx, mesh_triangle, true);
    } else if (hairDock) {
        hairUI.drawHairBrushDockContent(ctx.renderer.getHairSystem(), &ctx.renderer);
    } else {
        drawPaintBrushControls(ctx, mesh_triangle, slimDock);
    }
    paint_brush_dock_width = std::clamp(ImGui::GetWindowWidth(), 50.0f, 400.0f);
    ImGui::End();
    
    if (slimDock) {
        ImGui::PopStyleVar(3);
    } else {
        ImGui::PopStyleVar();
    }
}

void SceneUI::handleMeshPaint(UIContext& ctx) {
    if (!paint_mode_state.enabled || !paint_mode_state.hasValidTarget()) {
        paint_mode_state.stroke = Paint::PaintStroke{};
        return;
    }

    auto adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter());
    if (!adapter || !adapter->isValid()) {
        paint_mode_state.stroke = Paint::PaintStroke{};
        return;
    }
    if (!isMeshPaintUiChannelEnabled(paint_mode_state.active_channel)) {
        paint_mode_state.active_channel = Paint::PaintChannel::BaseColor;
    }
    syncHeightMaskPaintToggles(paint_mode_state);
    const std::vector<Paint::PaintChannel> paint_channels = getSelectedMaterialBrushChannels(paint_mode_state);
    const bool mask_channel_selected =
        std::find(paint_channels.begin(), paint_channels.end(), Paint::PaintChannel::Mask) != paint_channels.end();
    const bool auto_height_brush =
        paint_mode_state.brush.write_height_mask &&
        !mask_channel_selected &&
        brushSupportsRaisedPaint(paint_mode_state.brush.tool) &&
        paint_mode_state.brush.tool != Paint::BrushTool::Clone;
    const bool normal_from_height_active =
        paint_mode_state.auto_normal_from_height &&
        (auto_height_brush || mask_channel_selected);
    auto makeHeightBrush = [&](const Paint::BrushSettings& source_brush, float deposit_ratio) {
        Paint::BrushSettings height_brush = source_brush;
        height_brush.use_paint_texture = false;
        height_brush.paint_texture.reset();
        height_brush.paint_texture_path.clear();
        height_brush.channel_inputs = {};
        const float deposit_t = std::clamp((deposit_ratio - 0.05f) / 0.85f, 0.0f, 1.0f);
        const float deposit_coverage = deposit_t * deposit_t * (3.0f - 2.0f * deposit_t);
        
        // The core user request calculation: Scale the height mask application speed
        // by the exact amount of physical paint deposited during wet/mix modes.
        // By reducing the strength towards 0, the blending weight natively drops to 0,
        // cleanly preventing the erasure of existing geometry and blocking unearned spikes.
        height_brush.strength *= deposit_ratio * deposit_coverage;

        const float h = raisedPaintHeightValue(height_brush.height_contribution);
        height_brush.color = Vec3(h, h, h);
        return height_brush;
    };
    auto snapshot_channels_for_stroke = [&](std::vector<Paint::PaintChannel> channels) {
        if (auto_height_brush &&
            std::find(channels.begin(), channels.end(), Paint::PaintChannel::Mask) == channels.end()) {
            channels.push_back(Paint::PaintChannel::Mask);
        }
        if (normal_from_height_active &&
            std::find(channels.begin(), channels.end(), Paint::PaintChannel::Normal) == channels.end()) {
            channels.push_back(Paint::PaintChannel::Normal);
        }
        return channels;
    };

    ImGuiIO& io = ImGui::GetIO();
    const float simulation_dt = io.DeltaTime > 0.0f ? io.DeltaTime : (1.0f / 60.0f);
    if (adapter->tickWetPaint(
            paint_mode_state.brush,
            simulation_dt,
            normal_from_height_active,
            paint_mode_state.height_to_normal_strength)) {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            ctx.renderer.updateBackendMaterial(ctx.scene, adapter->getMaterialID());
            ctx.backend_ptr->resetAccumulation();
        }
    }
    static float last_vulkan_paint_sync_time = -1000.0f;
    auto finish_stroke = [&]() {
        if (!paint_mode_state.stroke.active) {
            return;
        }

        const bool keep_clone_source = paint_mode_state.stroke.has_clone_source;
        const float clone_source_u = paint_mode_state.stroke.clone_source_u;
        const float clone_source_v = paint_mode_state.stroke.clone_source_v;

        if (paint_mode_state.stroke.changed &&
            auto_height_brush &&
            paint_mode_state.auto_normal_from_height) {
            // Normals are now updated per-dab for real-time smoothness
        }
        adapter->endStroke();
        if (paint_mode_state.stroke.changed) {
            auto composite = std::make_unique<CompositeSceneCommand>(
                "Paint " + adapter->getNodeName());

            if (paint_mode_state.stroke.using_layers) {
                // Layer-aware undo: capture layer pixel changes + flat texture changes.
                Paint::PaintLayerStack* undo_stack = adapter->getLayerStack();
                Paint::PaintLayerData* undo_layer = undo_stack ? undo_stack->layerById(paint_mode_state.stroke.layer_id) : nullptr;

                for (int i = 0; i < static_cast<int>(Paint::kPaintChannelCount); ++i) {
                    const size_t idx = static_cast<size_t>(i);
                    const auto channel = static_cast<Paint::PaintChannel>(i);
                    std::vector<CompactVec4>& before_lp = paint_mode_state.stroke.before_layer_pixels[idx];
                    if (before_lp.empty() && !(undo_layer && undo_layer->hasPixels(channel))) {
                        continue; // Channel was not painted
                    }
                    // Get current (after) layer pixels
                    std::vector<CompactVec4> after_lp;
                    if (undo_layer && undo_layer->hasPixels(channel)) {
                        after_lp = undo_layer->getPixels(channel);
                    }
                    // Skip if nothing changed
                    if (before_lp.size() == after_lp.size() && before_lp == after_lp) continue;

                    composite->add(std::make_unique<PaintLayerCommand>(
                        adapter->getNodeName(),
                        adapter->getMaterialID(),
                        paint_mode_state.stroke.layer_stack_key,
                        paint_mode_state.stroke.layer_id,
                        channel,
                        std::move(before_lp),
                        std::move(after_lp)));
                }

                // Layer strokes can also mutate flat generated textures, most
                // importantly auto-normal-from-height. Capture those deltas so
                // undo/redo restores the generated texture as well as the layer.
                for (int i = 0; i < static_cast<int>(Paint::kPaintChannelCount); ++i) {
                    const size_t idx = static_cast<size_t>(i);
                    std::shared_ptr<Texture> texture_ref = paint_mode_state.stroke.texture_snapshot_refs[idx];
                    const std::vector<CompactVec4>& before_pixels = paint_mode_state.stroke.before_pixels_by_channel[idx];
                    if (!texture_ref || before_pixels.empty()) {
                        continue;
                    }
                    if (before_pixels.size() == texture_ref->pixels.size() &&
                        before_pixels == texture_ref->pixels) {
                        continue;
                    }
                    composite->add(std::make_unique<PaintTextureCommand>(
                        adapter->getNodeName(),
                        adapter->getMaterialID(),
                        texture_ref,
                        before_pixels,
                        texture_ref->pixels));
                }
            } else {
                // Flat texture undo (no layers)
                for (int i = 0; i < static_cast<int>(Paint::kPaintChannelCount); ++i) {
                    std::shared_ptr<Texture> texture_ref = paint_mode_state.stroke.texture_snapshot_refs[static_cast<size_t>(i)];
                    std::vector<CompactVec4>& before_pixels = paint_mode_state.stroke.before_pixels_by_channel[static_cast<size_t>(i)];
                    if (!texture_ref || before_pixels.empty()) {
                        continue;
                    }
                    composite->add(std::make_unique<PaintTextureCommand>(
                        adapter->getNodeName(),
                        adapter->getMaterialID(),
                        texture_ref,
                        before_pixels,
                        texture_ref->pixels));
                }
            }

            if (!composite->empty()) {
                history.record(std::move(composite));
                g_ProjectManager.markModified();
            }
        }
        paint_mode_state.stroke = Paint::PaintStroke{};
        if (keep_clone_source) {
            paint_mode_state.stroke.has_clone_source = true;
            paint_mode_state.stroke.clone_source_u = clone_source_u;
            paint_mode_state.stroke.clone_source_v = clone_source_v;
        }
        // Final backend sync so the painted texture is fully up-to-date after
        // the throttled mid-stroke updates. Paired with the 220 ms throttle in
        // the per-dab loop below. Both backends sync'd because g_viewport_backend
        // is the live raster MP viewport even when m_backend is Vulkan RT/OptiX.
        if (ctx.backend_ptr) {
            ctx.renderer.updateBackendMaterials(ctx.scene, ctx.backend_ptr);
            ctx.backend_ptr->resetAccumulation();
            last_vulkan_paint_sync_time = static_cast<float>(ImGui::GetTime());
        }
        if (g_viewport_backend && g_viewport_backend.get() != ctx.backend_ptr) {
            ctx.renderer.updateBackendMaterials(ctx.scene, g_viewport_backend.get());
        }
    };

    if (io.WantCaptureMouse) {
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && paint_mode_state.stroke.active) {
            finish_stroke();
        }
        return;
    }

    if (paint_mode_state.brush.tool == Paint::BrushTool::Fill) {
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && paint_mode_state.stroke.active) {
            finish_stroke();
        }
        return;
    }

    // Ensure CPU triangle positions match current GPU transforms before raycasting.
    // In TLAS mode, gizmo transforms update the GPU AS but defer CPU vertex sync
    // (objects_needing_cpu_sync). Without this, the paint raycast would use stale
    // triangle positions and produce UV drift when the object was recently moved.
    ensureCPUSyncForPicking(ctx);

    static bool alpha_rotate_drag_active = false;
    static ImVec2 alpha_rotate_anchor(0.0f, 0.0f);
    const bool can_rotate_brush_orientation = brushSupportsViewportRotation(paint_mode_state.brush);
    const bool alpha_rotate_shortcut =
        can_rotate_brush_orientation &&
        io.KeyCtrl &&
        ImGui::IsMouseDown(ImGuiMouseButton_Right) &&
        !ImGui::IsMouseDown(ImGuiMouseButton_Left);
    if (alpha_rotate_shortcut && !alpha_rotate_drag_active) {
        alpha_rotate_drag_active = true;
        alpha_rotate_anchor = ImGui::GetMousePos();
    }

    const ImVec2 paint_mouse_pos = alpha_rotate_drag_active ? alpha_rotate_anchor : ImGui::GetMousePos();
    HitRecord rec;
    bool has_hit = raycastViewportHit(ctx, paint_mouse_pos, rec);
    bool is_target_hit = has_hit && isMeshPaintTargetHit(rec, *adapter);
    if (!is_target_hit &&
        (!has_hit || (rec.triangle && rec.triangle->getNodeName() == adapter->getNodeName()))) {
        auto mesh_it = mesh_cache.find(adapter->getNodeName());
        if (mesh_it != mesh_cache.end()) {
            HitRecord fallback_rec;
            if (raycastMeshPaintTargetFallback(ctx, paint_mouse_pos, mesh_it->second, *adapter, fallback_rec)) {
                rec = fallback_rec;
                has_hit = true;
                is_target_hit = true;
            }
        }
    }

    if (alpha_rotate_shortcut) {
        const float delta = io.MouseDelta.x + io.MouseDelta.y;
        if (std::abs(delta) > 0.001f) {
            paint_mode_state.brush.alpha_rotation_degrees =
                wrapBrushAngleDegrees(paint_mode_state.brush.alpha_rotation_degrees + delta * 0.65f);
        }
        if (ctx.renderer.window) {
            SDL_WarpMouseInWindow(
                ctx.renderer.window,
                static_cast<int>(alpha_rotate_anchor.x),
                static_cast<int>(alpha_rotate_anchor.y));
            io.MousePos = alpha_rotate_anchor;
        }
    } else {
        if (alpha_rotate_drag_active && ctx.renderer.window) {
            SDL_WarpMouseInWindow(
                ctx.renderer.window,
                static_cast<int>(alpha_rotate_anchor.x),
                static_cast<int>(alpha_rotate_anchor.y));
            io.MousePos = alpha_rotate_anchor;
        }
        alpha_rotate_drag_active = false;
    }
    if (paint_mode_state.brush.tool == Paint::BrushTool::Clone &&
        io.KeyCtrl &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        is_target_hit) {
        paint_mode_state.stroke.has_clone_source = true;
        paint_mode_state.stroke.clone_source_u = rec.uv.u;
        paint_mode_state.stroke.clone_source_v = rec.uv.v;
        paint_mode_state.stroke.clone_offset_initialized = false;
        return;
    }
    if (has_hit && is_target_hit) {
        drawMeshPaintPreview(ctx, rec, *adapter, paint_mode_state.brush, paint_mode_state.active_channel, false, can_rotate_brush_orientation && io.KeyCtrl);
        for (int i = 1; i < 8; ++i) {
            const bool mx = (i & 1) && paint_mode_state.brush.mirror_x;
            const bool my = (i & 2) && paint_mode_state.brush.mirror_y;
            const bool mz = (i & 4) && paint_mode_state.brush.mirror_z;
            if ((i & 1) && !paint_mode_state.brush.mirror_x) continue;
            if ((i & 2) && !paint_mode_state.brush.mirror_y) continue;
            if ((i & 4) && !paint_mode_state.brush.mirror_z) continue;

            HitRecord mirrored_hit;
            if (resolveMirroredMeshPaintHit(ctx, *adapter, rec, mx, my, mz, mirrored_hit)) {
                Paint::BrushSettings mirrored_brush = makeMirroredBrushForHit(
                    paint_mode_state.brush,
                    rec,
                    mirrored_hit,
                    mx,
                    my,
                    mz);
                drawMeshPaintPreview(ctx, mirrored_hit, *adapter, mirrored_brush, paint_mode_state.active_channel, true);
            }
        }
    }

    if (alpha_rotate_shortcut || !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (paint_mode_state.stroke.active) {
            finish_stroke();
        }
        return;
    }

    adapter->ensureTextureSet(paint_mode_state.requested_texture_resolution);
    for (Paint::PaintChannel channel : paint_channels) {
        if (!adapter->assignTextureToChannel(channel)) {
            return;
        }
    }
    if (auto_height_brush) {
        if (!adapter->assignTextureToChannel(Paint::PaintChannel::Mask)) {
            return;
        }
    }
    if (normal_from_height_active) {
        if (!adapter->assignTextureToChannel(Paint::PaintChannel::Normal)) {
            return;
        }
    }

    if (!is_target_hit) {
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && paint_mode_state.stroke.active) {
            finish_stroke();
        }
        return;
    }

    if (paint_mode_state.brush.tool == Paint::BrushTool::Clone && !paint_mode_state.stroke.has_clone_source) {
        return;
    }

    if (!paint_mode_state.stroke.active) {
        Paint::PaintStrokeContext stroke_ctx;
        stroke_ctx.layer_index = paint_mode_state.active_layer_index;
        stroke_ctx.dt = io.DeltaTime > 0.0f ? io.DeltaTime : (1.0f / 60.0f);
        adapter->beginStroke(stroke_ctx);
        paint_mode_state.stroke.active = true;
        paint_mode_state.stroke.changed = false;
        paint_mode_state.stroke.channel = paint_mode_state.active_channel;
        paint_mode_state.stroke.remaining_paint_load = paint_mode_state.brush.paint_load;
        paint_mode_state.stroke.wet_color = getPreviewChannelColor(
            paint_mode_state.brush,
            paint_mode_state.active_channel);
        paint_mode_state.stroke.has_wet_color = true;
        Paint::PaintTextureSet* set = adapter->getTextureSet();
        const std::vector<Paint::PaintChannel> snapshot_channels = snapshot_channels_for_stroke(paint_channels);

        // Determine if layer-aware painting will be used for this stroke.
        Paint::PaintLayerStack* stroke_layer_stack = adapter->getLayerStack();
        const bool stroke_uses_layers = (stroke_layer_stack != nullptr && stroke_layer_stack->layerCount() > 0);
        paint_mode_state.stroke.using_layers = stroke_uses_layers;

        for (Paint::PaintChannel channel : snapshot_channels) {
            std::shared_ptr<Texture> active_texture = set ? set->getTexture(channel) : nullptr;
            const size_t index = static_cast<size_t>(channel);
            paint_mode_state.stroke.texture_snapshot_refs[index] = active_texture;
            paint_mode_state.stroke.before_pixels_by_channel[index] = active_texture ? active_texture->pixels : std::vector<CompactVec4>{};
        }

        // Capture layer pixel snapshots for layer-aware undo.
        if (stroke_uses_layers) {
            paint_mode_state.stroke.layer_stack_key = adapter->getNodeName() + "#" + std::to_string(adapter->getMaterialID());
            Paint::PaintLayerData* active_ld = stroke_layer_stack->layerAt(paint_mode_state.active_layer_index);
            paint_mode_state.stroke.layer_id = active_ld ? active_ld->id : 0;
            for (Paint::PaintChannel channel : snapshot_channels) {
                const size_t index = static_cast<size_t>(channel);
                if (active_ld && active_ld->hasPixels(channel)) {
                    paint_mode_state.stroke.before_layer_pixels[index] = active_ld->getPixels(channel);
                } else {
                    paint_mode_state.stroke.before_layer_pixels[index].clear();
                }
            }
        }
    }

    paint_mode_state.stroke.elapsed += simulation_dt;

    bool should_apply = true;
    if (paint_mode_state.brush.tool == Paint::BrushTool::Stamp) {
        if (paint_mode_state.brush.stamp_mode == Paint::StampPlacementMode::Single) {
            should_apply = !paint_mode_state.stroke.stamp_applied;
        } else if (paint_mode_state.stroke.has_last_uv) {
            Paint::PaintTextureSet* set = adapter->getTextureSet();
            std::shared_ptr<Texture> active_texture = set ? set->getTexture(paint_mode_state.active_channel) : nullptr;
            if (active_texture && active_texture->width > 0 && active_texture->height > 0) {
            const float dx = (rec.uv.u - paint_mode_state.stroke.last_u) * static_cast<float>(active_texture->width);
            const float dy = (rec.uv.v - paint_mode_state.stroke.last_v) * static_cast<float>(active_texture->height);
            const float distance_px = std::sqrt(dx * dx + dy * dy);
            
            const float baseline = 1024.0f;
            const float scaled_radius = paint_mode_state.brush.radius * (static_cast<float>(active_texture->width) / baseline);
            const float spacing_px = std::max(1.0f, scaled_radius * std::max(0.05f, paint_mode_state.brush.spacing));
            should_apply = distance_px >= spacing_px;
        }
        }
    } else if (paint_mode_state.stroke.has_last_uv) {
        Paint::PaintTextureSet* set = adapter->getTextureSet();
        std::shared_ptr<Texture> active_texture = set ? set->getTexture(paint_mode_state.active_channel) : nullptr;
        if (active_texture && active_texture->width > 0 && active_texture->height > 0) {
        const float dx = (rec.uv.u - paint_mode_state.stroke.last_u) * static_cast<float>(active_texture->width);
        const float dy = (rec.uv.v - paint_mode_state.stroke.last_v) * static_cast<float>(active_texture->height);
        const float distance_px = std::sqrt(dx * dx + dy * dy);
        
        const float baseline = 1024.0f;
        const float scaled_radius = paint_mode_state.brush.radius * (static_cast<float>(active_texture->width) / baseline);
        const float spacing_px = std::max(1.0f, scaled_radius * std::max(0.05f, paint_mode_state.brush.spacing));
        should_apply = distance_px >= spacing_px;
        if (!should_apply &&
            (paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Wet ||
             paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Oil) &&
            paint_mode_state.brush.tool == Paint::BrushTool::Paint) {
            const float dwell_interval = std::clamp(0.02f + spacing_px * 0.0009f, 0.02f, 0.12f);
            should_apply = paint_mode_state.stroke.elapsed >= dwell_interval;
        }
    }
    }

    if (!should_apply) {
        return;
    }

    Paint::BrushSettings dab_brush = paint_mode_state.brush;
    Vec2 dab_uv = rec.uv;
    if (paint_mode_state.brush.follow_stroke_angle && paint_mode_state.stroke.has_last_uv) {
        const float du = rec.uv.u - paint_mode_state.stroke.last_u;
        const float dv = rec.uv.v - paint_mode_state.stroke.last_v;
        if ((du * du + dv * dv) > 1e-8f) {
            const float stroke_angle = std::atan2(dv, du) * 57.2957795f;
            dab_brush.alpha_rotation_degrees = wrapBrushAngleDegrees(
                paint_mode_state.brush.alpha_rotation_degrees + stroke_angle);
        }
    }
    if (paint_mode_state.brush.tool == Paint::BrushTool::Stamp) {
        if (paint_mode_state.brush.stamp_random_rotation) {
            const float random_turn = uiHash01(rec.uv.u * 173.0f, rec.uv.v * 257.0f);
            dab_brush.alpha_rotation_degrees += (random_turn * 360.0f) - 180.0f;
        }
        if (paint_mode_state.brush.stamp_scale_jitter > 0.001f) {
            const float random_scale = uiHash01(rec.uv.u * 311.0f, rec.uv.v * 149.0f);
            const float jitter = (random_scale * 2.0f - 1.0f) * paint_mode_state.brush.stamp_scale_jitter;
            dab_brush.alpha_scale *= std::max(0.1f, 1.0f + jitter);
        }
        if (paint_mode_state.brush.scatter_jitter > 0.001f) {
            Paint::PaintTextureSet* set = adapter->getTextureSet();
            std::shared_ptr<Texture> active_texture = set ? set->getTexture(paint_mode_state.active_channel) : nullptr;
            if (active_texture && active_texture->width > 0 && active_texture->height > 0) {
                const float angle = uiHash01(rec.uv.u * 401.0f, rec.uv.v * 197.0f) * 6.2831853f;
                const float baseline = 1024.0f;
                const float scaled_radius = paint_mode_state.brush.radius * (static_cast<float>(active_texture->width) / baseline);
                const float radius = std::sqrt(uiHash01(rec.uv.u * 167.0f, rec.uv.v * 557.0f)) *
                    paint_mode_state.brush.scatter_jitter * scaled_radius;
                dab_uv.u = std::clamp(dab_uv.u + (std::cos(angle) * radius) / static_cast<float>(active_texture->width), 0.0f, 1.0f);
                dab_uv.v = std::clamp(dab_uv.v + (std::sin(angle) * radius) / static_cast<float>(active_texture->height), 0.0f, 1.0f);
            }
        }
    }    float current_deposit_ratio = 1.0f;

    if ((paint_mode_state.brush.tool == Paint::BrushTool::Paint ||
         paint_mode_state.brush.tool == Paint::BrushTool::Stamp) &&
        (paint_mode_state.active_channel == Paint::PaintChannel::BaseColor ||
         paint_mode_state.active_channel == Paint::PaintChannel::Emission)) {
        Paint::PaintTextureSet* set = adapter->getTextureSet();
        std::shared_ptr<Texture> active_texture = set ? set->getTexture(paint_mode_state.active_channel) : nullptr;
        
        if (active_texture && paint_mode_state.brush.paint_mode != Paint::BrushPaintMode::Normal) {
            const Vec3 sampled = active_texture->get_color_bilinear(rec.uv.u, rec.uv.v);
            
            if (!paint_mode_state.stroke.has_pickup_color) {
                paint_mode_state.stroke.pickup_color = sampled;
                paint_mode_state.stroke.has_pickup_color = true;
            } else {
                paint_mode_state.stroke.pickup_color = lerpColor(
                    paint_mode_state.stroke.pickup_color,
                    sampled,
                    0.35f);
            }
            const Vec3 effective_paint_color = getPreviewChannelColor(
                dab_brush,
                paint_mode_state.active_channel);
            auto syncActiveChannelColorOverride = [&](const Vec3& color) {
                if (Paint::BrushChannelInput* input = getBrushChannelInput(dab_brush, paint_mode_state.active_channel)) {
                    input->color = color;
                }
            };

            if (paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Mix) {
                dab_brush.color = lerpColor(
                    effective_paint_color,
                    paint_mode_state.stroke.pickup_color,
                    paint_mode_state.brush.mix_amount);
                syncActiveChannelColorOverride(dab_brush.color);
            } else if (paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Smudge) {
                dab_brush.color = lerpColor(
                    sampled,
                    paint_mode_state.stroke.pickup_color,
                    paint_mode_state.brush.smudge_strength);
                syncActiveChannelColorOverride(dab_brush.color);
                dab_brush.strength *= 0.75f;
                dab_brush.flow *= 0.9f;
            } else if (paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Wet ||
                       paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Oil) {
                const bool oil_mode = paint_mode_state.brush.paint_mode == Paint::BrushPaintMode::Oil;
                if (!paint_mode_state.stroke.has_wet_color) {
                    paint_mode_state.stroke.wet_color = effective_paint_color;
                    paint_mode_state.stroke.has_wet_color = true;
                }
                const float load = std::clamp(paint_mode_state.stroke.remaining_paint_load, 0.0f, 1.0f);
                const float wetness = std::clamp(paint_mode_state.brush.wetness, 0.0f, 1.0f);
                const float pickup = std::clamp(paint_mode_state.brush.pickup_rate, 0.0f, 1.0f);
                const float deposit = std::clamp(paint_mode_state.brush.deposit_rate, 0.0f, 1.0f);
                const float substrate_reservoir = std::clamp(adapter->sampleWetPickupReservoir(rec.uv), 0.0f, 1.0f);
                const float fresh_bias = std::clamp(
                    deposit * (0.82f + 0.18f * wetness) * (0.78f + 0.22f * load),
                    0.0f,
                    1.0f);
                const float underpaint_mix = std::clamp(
                        substrate_reservoir * wetness * (1.0f - fresh_bias * (oil_mode ? 0.92f : 0.82f)) *
                            ((oil_mode ? 0.08f : 0.18f) + pickup * (oil_mode ? 0.18f : 0.34f)) *
                            ((oil_mode ? 0.36f : 0.55f) - load * (oil_mode ? 0.10f : 0.18f)),
                    0.0f,
                        oil_mode ? 0.20f : 0.42f);
                
                const Vec3 loaded_paint = lerpColor(
                    paint_mode_state.stroke.wet_color,
                    effective_paint_color,
                    load);

                dab_brush.color = lerpColor(loaded_paint, sampled, underpaint_mix);
                syncActiveChannelColorOverride(dab_brush.color);
                
                paint_mode_state.stroke.wet_color = lerpColor(
                    paint_mode_state.stroke.wet_color,
                    sampled,
                    std::clamp(
                        substrate_reservoir * pickup * wetness *
                            ((oil_mode ? 0.04f : 0.10f) + (1.0f - deposit) * (oil_mode ? 0.08f : 0.18f)) *
                            (1.0f - load * (oil_mode ? 0.82f : 0.62f)),
                        0.0f,
                        oil_mode ? 0.18f : 0.45f));
                    
                paint_mode_state.stroke.remaining_paint_load = std::clamp(
                    paint_mode_state.stroke.remaining_paint_load - deposit * (oil_mode ? 0.010f : 0.020f) +
                        pickup * substrate_reservoir * (oil_mode ? 0.004f : 0.010f),
                    0.0f,
                    1.0f);
                dab_brush.paint_load = load;
                dab_brush.flow *= oil_mode ? (0.96f + 0.08f * wetness) : (0.9f + 0.2f * wetness);
                current_deposit_ratio = std::clamp(fresh_bias * (0.78f + 0.22f * load), 0.0f, 1.0f);
            }
            
            // Calculate the absolute visual difference representing strictly how much fresh external paint weight
            // was introduced exactly to this blended pixel by the engine algorithms vs standard background color.
            if (paint_mode_state.brush.paint_mode != Paint::BrushPaintMode::Wet &&
                paint_mode_state.brush.paint_mode != Paint::BrushPaintMode::Oil) {
                const float color_diff = (dab_brush.color - sampled).length() / 1.73205f;
                current_deposit_ratio = std::clamp(color_diff, 0.0f, 1.0f);
            }
        }
    }

    // Determine if layer-aware painting is active.
    Paint::PaintLayerStack* layer_stack = adapter->getLayerStack();
    const bool use_layers = (layer_stack != nullptr && layer_stack->layerCount() > 0);
    const int active_layer = paint_mode_state.active_layer_index;
    const float dab_dt = std::max(io.DeltaTime > 0.0f ? io.DeltaTime : (1.0f / 60.0f), paint_mode_state.stroke.elapsed);

    // Accumulated dirty rect for region-based compositing (layers only).
    Paint::PaintDirtyRect accumulated_dirty;

    // Helper lambdas: paint / clone through layers or direct depending on mode.
    // When using layers, returns true and accumulates the dirty rect.
    auto doPaintAtUV = [&](Paint::PaintChannel ch, const Vec2& uv, const Paint::BrushSettings& br, float dt) -> bool {
        if (use_layers) {
            Paint::PaintDirtyRect dirty = adapter->paintLayerAtUV(active_layer, ch, uv, br, dt);
            if (!dirty.empty()) { accumulated_dirty.expand(dirty); return true; }
            return false;
        }
        return adapter->paintAtUV(ch, uv, br, dt);
    };
    auto doCloneAtUV = [&](Paint::PaintChannel ch, const Vec2& dst, const Vec2& src, const Paint::BrushSettings& br, float dt) -> bool {
        if (use_layers) {
            Paint::PaintDirtyRect dirty = adapter->cloneLayerAtUV(active_layer, ch, dst, src, br, dt);
            if (!dirty.empty()) { accumulated_dirty.expand(dirty); return true; }
            return false;
        }
        return adapter->cloneAtUV(ch, dst, src, br, dt);
    };
    auto isWetManagedAuxChannel = [&](Paint::PaintChannel ch, const Paint::BrushSettings& br) -> bool {
        if (br.paint_mode != Paint::BrushPaintMode::Wet &&
            br.paint_mode != Paint::BrushPaintMode::Oil) {
            return false;
        }
        if (ch != Paint::PaintChannel::Roughness &&
            ch != Paint::PaintChannel::Metallic &&
            ch != Paint::PaintChannel::Transmission) {
            return false;
        }
        return getBrushChannelInput(br, ch) != nullptr;
    };
    auto updateAutoNormalAtUV = [&](const Vec2& uv, float brush_radius) -> bool {
        if (!paint_mode_state.auto_normal_from_height) {
            return false;
        }
        if (use_layers && !accumulated_dirty.empty()) {
            const Paint::PaintChannel mask_channel = Paint::PaintChannel::Mask;
            adapter->compositeAndUploadRegion(&mask_channel, 1, accumulated_dirty);
        }
        return adapter->updateNormalFromHeightArea(
            uv,
            brush_radius,
            paint_mode_state.height_to_normal_strength);
    };

    // Collect channels that were actually painted for selective compositing.
    std::vector<Paint::PaintChannel> painted_channels;
    Paint::PaintTextureSet* set = adapter->getTextureSet();
    std::shared_ptr<Texture> active_texture = set ? set->getTexture(paint_mode_state.active_channel) : nullptr;

    auto applyPaintAtHit = [&](const HitRecord& hit) -> bool {
        bool hit_changed = false;
        const Vec2 hit_uv = hit.uv;

        if (paint_mode_state.brush.tool == Paint::BrushTool::Spray) {
            if (!active_texture || active_texture->width <= 0 || active_texture->height <= 0) {
                return false;
            }

            const int particles = std::clamp(paint_mode_state.brush.spray_particles, 1, 64);
            const float spread = std::clamp(paint_mode_state.brush.spray_spread, 0.05f, 1.0f);
            const float size_jitter = std::clamp(paint_mode_state.brush.spray_size_jitter, 0.0f, 1.0f);
            const float opacity_jitter = std::clamp(paint_mode_state.brush.spray_opacity_jitter, 0.0f, 1.0f);
            for (int p = 0; p < particles; ++p) {
                const float seed_u = hit_uv.u * (137.0f + static_cast<float>(p) * 17.0f);
                const float seed_v = hit_uv.v * (251.0f + static_cast<float>(p) * 23.0f);
                const float angle = uiHash01(seed_u, seed_v) * 6.2831853f;
                const float radius = std::sqrt(uiHash01(seed_v + 13.0f, seed_u + 29.0f)) *
                    spread * paint_mode_state.brush.radius;
                const Vec2 spray_uv(
                    std::clamp(hit_uv.u + (std::cos(angle) * radius) / static_cast<float>(active_texture->width), 0.0f, 1.0f),
                    std::clamp(hit_uv.v + (std::sin(angle) * radius) / static_cast<float>(active_texture->height), 0.0f, 1.0f));
                Paint::BrushSettings spray_brush = dab_brush;
                const float droplet_base_size = std::clamp(paint_mode_state.brush.spray_droplet_size, 0.05f, 1.0f);
                const float size_noise = uiHash01(seed_u + 41.0f, seed_v + 97.0f) * 2.0f - 1.0f;
                const float opacity_noise = uiHash01(seed_u + 149.0f, seed_v + 211.0f) * 2.0f - 1.0f;
                const float droplet_size = std::max(0.05f, droplet_base_size * (1.0f + size_noise * size_jitter));
                const float droplet_strength = std::max(0.05f, 1.0f + opacity_noise * opacity_jitter);
                spray_brush.radius = std::max(1.0f, paint_mode_state.brush.radius * droplet_size);
                spray_brush.strength *= (1.0f / std::sqrt(static_cast<float>(particles))) * droplet_strength;
                for (Paint::PaintChannel channel : paint_channels) {
                    if (isWetManagedAuxChannel(channel, spray_brush)) {
                        continue;
                    }
                    if (doPaintAtUV(channel, spray_uv, spray_brush, dab_dt)) {
                        hit_changed = true;
                        painted_channels.push_back(channel);
                        if (channel == Paint::PaintChannel::BaseColor &&
                            (spray_brush.paint_mode == Paint::BrushPaintMode::Wet ||
                             spray_brush.paint_mode == Paint::BrushPaintMode::Oil)) {
                            adapter->noteWetDab(active_layer, spray_uv, spray_brush, dab_dt, current_deposit_ratio);
                        }
                        if (channel == Paint::PaintChannel::Mask &&
                            paint_mode_state.auto_normal_from_height) {
                            updateAutoNormalAtUV(spray_uv, spray_brush.radius);
                        }
                    }
                }
                if (auto_height_brush) {
                    const Paint::BrushSettings height_brush = makeHeightBrush(spray_brush, current_deposit_ratio);
                    if (doPaintAtUV(Paint::PaintChannel::Mask, spray_uv, height_brush, dab_dt)) {
                        hit_changed = true;
                        painted_channels.push_back(Paint::PaintChannel::Mask);
                        if (paint_mode_state.auto_normal_from_height) {
                            updateAutoNormalAtUV(spray_uv, spray_brush.radius);
                        }
                    }
                }
            }
            return hit_changed;
        }

        if (paint_mode_state.brush.tool == Paint::BrushTool::Clone) {
            if (!paint_mode_state.stroke.clone_offset_initialized) {
                paint_mode_state.stroke.clone_offset_u = paint_mode_state.stroke.clone_source_u - rec.uv.u;
                paint_mode_state.stroke.clone_offset_v = paint_mode_state.stroke.clone_source_v - rec.uv.v;
                paint_mode_state.stroke.clone_offset_initialized = true;
            }
            const Vec2 src_uv(
                hit_uv.u + paint_mode_state.stroke.clone_offset_u,
                hit_uv.v + paint_mode_state.stroke.clone_offset_v);
            for (Paint::PaintChannel channel : paint_channels) {
                if (doCloneAtUV(channel, hit_uv, src_uv, dab_brush, dab_dt)) {
                    hit_changed = true;
                    painted_channels.push_back(channel);
                }
            }
            return hit_changed;
        }

        for (Paint::PaintChannel channel : paint_channels) {
            if (isWetManagedAuxChannel(channel, dab_brush)) {
                continue;
            }
            if (doPaintAtUV(channel, hit_uv, dab_brush, dab_dt)) {
                hit_changed = true;
                painted_channels.push_back(channel);
                if (channel == Paint::PaintChannel::BaseColor &&
                    (dab_brush.paint_mode == Paint::BrushPaintMode::Wet ||
                     dab_brush.paint_mode == Paint::BrushPaintMode::Oil)) {
                    adapter->noteWetDab(active_layer, hit_uv, dab_brush, dab_dt, current_deposit_ratio);
                }
                if (channel == Paint::PaintChannel::Mask &&
                    paint_mode_state.auto_normal_from_height) {
                    updateAutoNormalAtUV(hit_uv, dab_brush.radius);
                }
            }
        }
        if (auto_height_brush) {
            const Paint::BrushSettings height_brush = makeHeightBrush(dab_brush, current_deposit_ratio);
            if (doPaintAtUV(Paint::PaintChannel::Mask, hit_uv, height_brush, dab_dt)) {
                hit_changed = true;
                painted_channels.push_back(Paint::PaintChannel::Mask);
                if (paint_mode_state.auto_normal_from_height) {
                    updateAutoNormalAtUV(hit_uv, dab_brush.radius);
                }
            }
        }
        return hit_changed;
    };

    const bool dab_changed = applyPaintAtHit(rec);

    // If using layers, composite the painted channels to the flat texture for display.
    if (dab_changed && use_layers && !painted_channels.empty()) {
        // Deduplicate painted_channels
        std::sort(painted_channels.begin(), painted_channels.end());
        painted_channels.erase(std::unique(painted_channels.begin(), painted_channels.end()), painted_channels.end());
        if (!accumulated_dirty.empty()) {
            adapter->compositeAndUploadRegion(painted_channels.data(), static_cast<int>(painted_channels.size()), accumulated_dirty);
        } else {
            adapter->compositeAndUploadChannels(painted_channels.data(), static_cast<int>(painted_channels.size()));
        }
    }

    if (dab_changed) {
        paint_mode_state.stroke.elapsed = 0.0f;
        paint_mode_state.stroke.changed = true;
        paint_mode_state.stroke.has_last_uv = true;
        paint_mode_state.stroke.last_u = rec.uv.u;
        paint_mode_state.stroke.last_v = rec.uv.v;
        if (paint_mode_state.brush.tool == Paint::BrushTool::Stamp &&
            paint_mode_state.brush.stamp_mode == Paint::StampPlacementMode::Single) {
            paint_mode_state.stroke.stamp_applied = true;
        }
        if (dab_brush.paint_mode == Paint::BrushPaintMode::Mix) {
            paint_mode_state.stroke.pickup_color = lerpColor(
                paint_mode_state.stroke.pickup_color,
                dab_brush.color,
                0.5f);
            paint_mode_state.stroke.has_pickup_color = true;
        } else if (dab_brush.paint_mode == Paint::BrushPaintMode::Smudge) {
            paint_mode_state.stroke.pickup_color = lerpColor(
                paint_mode_state.stroke.pickup_color,
                dab_brush.color,
                0.65f);
            paint_mode_state.stroke.has_pickup_color = true;
        } else if (dab_brush.paint_mode == Paint::BrushPaintMode::Wet ||
                   dab_brush.paint_mode == Paint::BrushPaintMode::Oil) {
            paint_mode_state.stroke.pickup_color = lerpColor(
                paint_mode_state.stroke.pickup_color,
                dab_brush.color,
                dab_brush.paint_mode == Paint::BrushPaintMode::Oil ? 0.18f : 0.35f);
            paint_mode_state.stroke.has_pickup_color = true;
        }
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            // Paint-time material sync should stay scoped to the painted
            // material; rebuilding the full backend material table every dab
            // scales with total scene material count and becomes visible in
            // crowded scenes.
            ctx.renderer.updateBackendMaterial(ctx.scene, adapter->getMaterialID());
            last_vulkan_paint_sync_time = static_cast<float>(ImGui::GetTime());
            ctx.backend_ptr->resetAccumulation();
        }

        // Mirror painting
        for (int i = 1; i < 8; ++i) {
            const bool mx = (i & 1) && paint_mode_state.brush.mirror_x;
            const bool my = (i & 2) && paint_mode_state.brush.mirror_y;
            const bool mz = (i & 4) && paint_mode_state.brush.mirror_z;
            if ((i & 1) && !paint_mode_state.brush.mirror_x) continue;
            if ((i & 2) && !paint_mode_state.brush.mirror_y) continue;
            if ((i & 4) && !paint_mode_state.brush.mirror_z) continue;

            HitRecord mirrored_hit;
            if (resolveMirroredMeshPaintHit(ctx, *adapter, rec, mx, my, mz, mirrored_hit)) {
                Paint::BrushSettings original_dab_brush = dab_brush;
                Paint::BrushSettings mirrored_brush = makeMirroredBrushForHit(
                    original_dab_brush,
                    rec,
                    mirrored_hit,
                    mx,
                    my,
                    mz);
                dab_brush = mirrored_brush;
                const bool mirror_changed = applyPaintAtHit(mirrored_hit);
                dab_brush = original_dab_brush;
                if (mirror_changed) {
                    paint_mode_state.stroke.changed = true;
                }
                if (mirror_changed && use_layers && !painted_channels.empty()) {
                    if (!accumulated_dirty.empty()) {
                        adapter->compositeAndUploadRegion(painted_channels.data(), static_cast<int>(painted_channels.size()), accumulated_dirty);
                    } else {
                        adapter->compositeAndUploadChannels(painted_channels.data(), static_cast<int>(painted_channels.size()));
                    }
                }
            }
        }
    }
}
