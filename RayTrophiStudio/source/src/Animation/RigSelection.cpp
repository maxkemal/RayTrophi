#include "Animation/RigSelection.h"
#include "scene_data.h"
#include <algorithm>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <utility>
namespace RigAuthoring {
namespace {
bool coherent(const ViewState& view) {
    return view.selection_character == view.character &&
           std::find(view.selected_bones.begin(), view.selected_bones.end(), view.bone) !=
               view.selected_bones.end();
}
}
bool isBoneSelected(const ViewState& view, const std::string& character, const std::string& bone) {
    if (view.character != character)
        return false;
    return coherent(view) ? std::find(view.selected_bones.begin(), view.selected_bones.end(),
                                      bone) != view.selected_bones.end()
                          : view.bone == bone;
}
std::vector<std::string> selectedBones(const SceneData& scene) {
    std::vector<BoneView> views;
    std::string error;
    if (!listBones(scene, scene.rigView.character, views, error))
        return {};
    if (std::none_of(views.begin(), views.end(),
                     [&](const auto& b) { return b.name == scene.rigView.bone; }))
        return {};
    std::vector<std::string> names;
    for (const auto& bone : views)
        if (isBoneSelected(scene.rigView, bone.character, bone.name))
            names.push_back(bone.name);
    return names;
}
bool setSelectionPivot(SceneData& scene, const std::string& mode, std::string& error) {
    error.clear();
    if (mode != "active" && mode != "center") {
        error = "rig_selection_invalid_pivot";
        return false;
    }
    scene.rigView.selection_pivot = mode;
    return true;
}
bool selectBones(SceneData& scene, const std::string& character,
                 const std::vector<std::string>& names, const std::string& active,
                 const std::string& mode, std::string& error, const std::string& anchorOverride) {
    error.clear();
    if (mode != "replace" && mode != "add" && mode != "toggle" && mode != "range") {
        error = "rig_selection_invalid_mode";
        return false;
    }
    if (names.size() > 4096) {
        error = "rig_selection_limit";
        return false;
    }
    if (scene.rigView.pose.active && scene.rigView.pose.character != character) {
        error = "rig_pose_character_locked";
        return false;
    }
    if (scene.rigView.edit_mode && scene.rigView.edit_character != character) {
        error = "rig_edit_character_locked";
        return false;
    }
    std::vector<BoneView> views;
    if (!listBones(scene, character, views, error))
        return false;
    if (views.size() > 4096) {
        error = "rig_selection_limit";
        return false;
    }
    std::unordered_set<std::string> valid, requested;
    for (const auto& b : views)
        if (!valid.insert(b.name).second) {
            error = "rig_selection_invalid_hierarchy";
            return false;
        }
    for (const auto& name : names) {
        if (!valid.count(name)) {
            error = "unknown_bone";
            return false;
        }
        if (!requested.insert(name).second) {
            error = "rig_selection_duplicate_bone";
            return false;
        }
    }
    if (!active.empty() && !valid.count(active)) {
        error = "unknown_bone";
        return false;
    }
    if (mode == "range" && names.size() != 1) {
        error = "rig_selection_invalid_range";
        return false;
    }
    std::unordered_set<std::string> result;
    std::string rangeAnchor;
    if (mode == "add" || mode == "toggle")
        if (scene.rigView.character == character)
            for (const auto& name : selectedBones(scene))
                result.insert(name);
    if (mode == "range") {
        std::string anchor =
            scene.rigView.selection_character == character ? scene.rigView.selection_anchor : "";
        if (!valid.count(anchor))
            anchor = scene.rigView.character == character && valid.count(scene.rigView.bone)
                         ? scene.rigView.bone
                         : names.front();
        rangeAnchor = anchor;
        // Stable full depth-first hierarchy order, including collapsed branches.
        std::vector<std::string> order;
        std::unordered_map<std::string, std::vector<std::string>> children;
        for (const auto& b : views)
            children[b.parent].push_back(b.name);
        std::function<void(const std::string&)> visit = [&](const std::string& name) {
            order.push_back(name);
            for (const auto& child : children[name])
                visit(child);
        };
        for (const auto& b : views)
            if (b.parent.empty() || !valid.count(b.parent))
                visit(b.name);
        if (order.size() != views.size()) {
            error = "rig_selection_invalid_hierarchy";
            return false;
        }
        auto a = std::find(order.begin(), order.end(), anchor),
             b = std::find(order.begin(), order.end(), names.front());
        if (a > b)
            std::swap(a, b);
        for (auto it = a; it != b + 1; ++it)
            result.insert(*it);
    } else
        for (const auto& name : names)
            if (mode == "toggle" && result.count(name))
                result.erase(name);
            else
                result.insert(name);
    if (result.size() > 4096) {
        error = "rig_selection_limit";
        return false;
    }
    std::string nextActive = active.empty() ? (names.empty() ? "" : names.back()) : active;
    if (!active.empty() && !result.count(active)) {
        error = "rig_selection_active_not_selected";
        return false;
    }
    if (!result.count(nextActive)) {
        nextActive = result.count(scene.rigView.bone) ? scene.rigView.bone : "";
        if (nextActive.empty())
            for (const auto& b : views)
                if (result.count(b.name))
                    nextActive = b.name;
    }
    auto staged = scene.rigView;
    staged.character = result.empty() ? "" : character;
    staged.bone = nextActive;
    staged.selected_bones.clear();
    for (const auto& b : views)
        if (result.count(b.name))
            staged.selected_bones.push_back(b.name);
    staged.selection_character = result.empty() ? "" : character;
    if (!anchorOverride.empty() && !result.count(anchorOverride)) {
        error = "rig_selection_anchor_not_selected";
        return false;
    }
    staged.selection_anchor =
        anchorOverride.empty() ? (mode == "range" ? rangeAnchor : nextActive) : anchorOverride;
    if (result.empty())
        staged.selection_anchor.clear();
    exchangeSelection(scene.rigView, staged);
    releaseIKHandleSelection(scene.rigView);
    return true;
}
void releaseIKHandleSelection(ViewState& view) {
    auto& pose = view.pose;
    if (pose.control.empty()) {
        return;
    }
    pose.control.clear();
    pose.controlHandle = "target";
    // Discard only the uncommitted IK gesture; committed IK and contacts stay active.
    if (pose.hasPreview) {
        pose.preview.clear();
        pose.previewIK.clear();
        pose.limitHits.clear();
        pose.hasPreview = false;
        pose.invalidateEvaluation();
    }
    ++pose.serial;
}
void exchangeSelection(ViewState& a, ViewState& b) {
    a.character.swap(b.character);
    a.bone.swap(b.bone);
    a.selected_bones.swap(b.selected_bones);
    a.selection_character.swap(b.selection_character);
    a.selection_anchor.swap(b.selection_anchor);
    a.selection_pivot.swap(b.selection_pivot);
}
}
