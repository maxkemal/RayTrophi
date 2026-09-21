#include "Animation/ClipBinding.h"
#include "Animation/Retarget.h"
#include <cmath>
#include <map>
#include <set>

namespace RigAuthoring {
bool buildSameRigClip(const AnimationData& clip, const RayTrophi::NodeHierarchy& source,
                      const RayTrophi::NodeHierarchy& target, const std::string& targetCharacter,
                      const std::string& outputName, ClipBindingReport& report,
                      std::shared_ptr<AnimationData>& output, std::string& error,
                      const std::map<std::string, std::string>& nodeMap,
                      const std::string& mode, float translationScale) {
    report = {}; output.reset(); error.clear();
    report.mode = mode; report.translation_scale = translationScale;
    if (mode != "same_rig" && mode != "rest_basis") { error = "invalid_retarget_mode"; return false; }
    if (!std::isfinite(translationScale) || translationScale <= 0 || translationScale > 10000) {
        error = "invalid_translation_scale"; return false;
    }
    if (mode == "same_rig" && translationScale != 1.f) { error = "translation_scale_requires_retarget"; return false; }
    report.source_character = clip.modelName; report.source_clip = clip.name;
    report.target_character = targetCharacter; report.output_clip = outputName;
    if (source.empty() || target.empty()) { error = "missing_node_hierarchy"; return false; }
    if (!std::isfinite(clip.duration) || !std::isfinite(clip.ticksPerSecond) || clip.duration <= 0 || clip.ticksPerSecond <= 0) {
        error = "invalid_clip_timing"; return false;
    }
    std::set<std::string> channels;
    for (const auto& entry : clip.positionKeys) if (!entry.second.empty()) channels.insert(entry.first);
    for (const auto& entry : clip.rotationKeys) if (!entry.second.empty()) channels.insert(entry.first);
    for (const auto& entry : clip.scalingKeys) if (!entry.second.empty()) channels.insert(entry.first);
    if (channels.empty()) { error = "clip_has_no_channels"; return false; }
    std::map<std::string, std::vector<const RayTrophi::SceneNode*>> sourceByName, targetByName;
    std::map<std::string, const RayTrophi::SceneNode*> sourceByKey, targetByKey;
    for (const auto& node : source.nodes) { sourceByName[node.name].push_back(&node); sourceByKey[node.uniqueName] = &node; }
    for (const auto& node : target.nodes) { targetByName[node.name].push_back(&node); targetByKey[node.uniqueName] = &node; }
    std::map<std::string, const RayTrophi::SceneNode*> mapping;
    for (const auto& entry : sourceByKey) {
        const auto found = targetByName.find(entry.second->name);
        if (found != targetByName.end() && found->second.size() == 1 && sourceByName[entry.second->name].size() == 1)
            mapping.emplace(entry.first, found->second.front());
    }
    for (const auto& entry : nodeMap) {
        if (!sourceByKey.count(entry.first)) { error = "unknown_source_node"; return false; }
        const auto to = targetByKey.find(entry.second);
        if (to == targetByKey.end()) { error = "unknown_target_node"; return false; }
        mapping[entry.first] = to->second;
    }
    std::set<std::string> usedTargets;
    for (const auto& entry : mapping)
        if (!usedTargets.insert(entry.second->uniqueName).second) { error = "duplicate_target_mapping"; return false; }
    for (const auto& key : channels) {
        const auto sourceNode = sourceByKey.find(key);
        if (sourceNode == sourceByKey.end()) { report.unmapped.push_back(key); continue; }
        const auto mapped = mapping.find(key);
        if (mapped == mapping.end()) {
            const auto candidates = targetByName.find(sourceNode->second->name);
            if (candidates != targetByName.end() && (candidates->second.size() > 1 || sourceByName[sourceNode->second->name].size() > 1)) report.ambiguous.push_back(key);
            else report.unmapped.push_back(key);
            continue;
        }
        const auto& from = *sourceNode->second; const auto& to = *mapped->second;
        report.matches.push_back({key, to.uniqueName, from.name});
        bool restDifferent = false;
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c)
            restDifferent |= std::fabs(from.localBind.m[r][c] - to.localBind.m[r][c]) > 1e-4f;
        if (restDifferent) ++report.rest_difference_count;
        // Ancestor closure must map too: animated helpers carry units/axes.
        const auto* a = &from; const auto* b = &to;
        std::set<const RayTrophi::SceneNode*> visited;
        while (a) {
            if (!visited.insert(a).second) { report.hierarchy_mismatches.push_back(key); break; }
            if ((a->parent < 0) != (b->parent < 0)) { report.hierarchy_mismatches.push_back(key); break; }
            if (a->parent < 0) break;
            if (static_cast<size_t>(a->parent) >= source.nodes.size() || static_cast<size_t>(b->parent) >= target.nodes.size()) { report.hierarchy_mismatches.push_back(key); break; }
            a = &source.nodes[a->parent]; b = &target.nodes[b->parent];
            const auto parent = mapping.find(a->uniqueName);
            if (parent == mapping.end() || parent->second != b) { report.hierarchy_mismatches.push_back(key); break; }
        }
    }
    report.ready = report.unmapped.empty() && report.ambiguous.empty() && report.hierarchy_mismatches.empty();
    if (!report.ready) return true; // Valid preflight report, incompatible mapping.
    output = std::make_shared<AnimationData>(clip);
    output->name = outputName; output->modelName = targetCharacter;
    output->positionKeys.clear(); output->rotationKeys.clear(); output->scalingKeys.clear();
    for (const auto& match : report.matches) {
        const auto p = clip.positionKeys.find(match.source); if (p != clip.positionKeys.end()) output->positionKeys.emplace(match.target, p->second);
        const auto r = clip.rotationKeys.find(match.source); if (r != clip.rotationKeys.end()) output->rotationKeys.emplace(match.target, r->second);
        const auto s = clip.scalingKeys.find(match.source); if (s != clip.scalingKeys.end()) output->scalingKeys.emplace(match.target, s->second);
    }
    if (mode == "rest_basis" && !applyRestBasisRetarget(clip, source, target, report, translationScale, *output, error)) {
        output.reset(); report.ready = false; return false;
    }
    return true;
}
}
