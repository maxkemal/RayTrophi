/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiAnim.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* Skeletal animation playback facade (Faz 5.6c).
*
* ★SCOPE — playback + parameters only, node-graph topology deliberately left
* out. See the header block in RtApi.h for the evidence; the short version is
* that three playback paths coexist (graph / Ozz / controller), the Ozz runtime
* is a declared future migration that is still a stub, and AnimationNodeGraph
* is not a NodeSystem::GraphBase so rt.nodes cannot reach it without either an
* invasive refactor of a moving system or a second node-scripting dialect.
* Everything exposed here keeps its meaning whichever runtime wins.
*
* Each character owns its own AnimationController (ImportedModelContext::animator).
* This file NEVER touches AnimationController::getInstance() — that singleton
* still exists but a scene has many characters, and driving the global one would
* move whichever character happened to share its state.
*/

#include "RtApiInternal.h"

#include <algorithm>
#include <string>
#include <vector>

#include "AnimationController.h"
#include "AnimationNodes.h"

namespace rtapi {
namespace {

// Mirrors the renderer's own choice: the per-character runtime clone is the
// graph that actually evaluates; `graph` is the editor-side original kept as a
// legacy alias. Reading the wrong one reports parameters nothing consumes.
std::shared_ptr<AnimationGraph::AnimationNodeGraph> activeGraphOf(
        const SceneData::ImportedModelContext& ctx) {
    return ctx.runtimeGraph ? ctx.runtimeGraph : ctx.graph;
}

SceneData::ImportedModelContext* findCharacter(const std::string& name) {
    for (auto& ctx : g_ctx->scene.importedModelContexts) {
        if (ctx.importName == name) return &ctx;
    }
    return nullptr;
}

Result requireCharacter(const std::string& name, SceneData::ImportedModelContext*& out) {
    if (!g_ctx) return notBound();
    out = findCharacter(name);
    if (!out) return Result::fail("animated character not found: " + name);
    if (!out->animator)
        return Result::fail("character has no animation controller: " + name);
    return Result::success();
}

Result requireLayer(int layer) {
    // AnimationController::MAX_LAYERS is 4; an out-of-range layer is silently
    // ignored deeper in, which would look like a no-op edit from a script.
    if (layer < 0 || layer > 3)
        return Result::fail("animation layer out of range: " + std::to_string(layer) +
                            " (0..3)");
    return Result::success();
}

AnimCharacterInfo infoFromContext(const SceneData::ImportedModelContext& ctx) {
    AnimCharacterInfo info;
    info.name = ctx.importName;
    info.has_animation = ctx.hasAnimation;
    info.clip_count = ctx.animator ? static_cast<int>(ctx.animator->getAllClips().size()) : 0;
    info.bone_count = static_cast<int>(ctx.weightedBoneCount);
    info.uses_graph = ctx.useAnimGraph;
    info.graph_asset_key = ctx.animGraphAssetKey;
    info.graph_follows_timeline = ctx.animGraphFollowTimeline;
    info.root_motion = ctx.useRootMotion;
    info.root_motion_bone = ctx.rootMotionBone;
    info.visible = ctx.visible;
    return info;
}

// A graph parameter written to a character that is not running its graph would
// be stored and never read. Report it instead of pretending it landed.
Result requireGraph(const std::string& character,
                    std::shared_ptr<AnimationGraph::AnimationNodeGraph>& out) {
    SceneData::ImportedModelContext* ctx = nullptr;
    if (Result r = requireCharacter(character, ctx); !r) return r;
    out = activeGraphOf(*ctx);
    if (!out)
        return Result::fail("character has no animation node graph: " + character);
    if (!ctx->useAnimGraph)
        return Result::fail("character is not running its animation graph "
                            "(use_graph is off): " + character);
    return Result::success();
}

} // namespace

std::vector<AnimCharacterInfo> listAnimCharacters() {
    std::vector<AnimCharacterInfo> out;
    if (!g_ctx) return out;
    for (const auto& ctx : g_ctx->scene.importedModelContexts) {
        if (ctx.importName.empty()) continue;
        // ★An ImportedModelContext exists for EVERY import, including a plain static mesh
        // (a cube, a plane) that has no skeleton and therefore no AnimationController. Listing
        // those made `for ch in rt.anim.characters(): rt.anim.clips(ch)` — the obvious way to
        // use this API — fail on an ordinary scene with no animation in it at all. The list
        // means "characters this module can drive", so the controller is the membership test.
        // getAnimCharacter() deliberately stays unfiltered: asking about a named import by
        // hand should still answer, and its info reports has_animation/clip_count honestly.
        if (!ctx.animator) continue;
        out.push_back(infoFromContext(ctx));
    }
    return out;
}

Result getAnimCharacter(const std::string& character, AnimCharacterInfo& out) {
    if (!g_ctx) return notBound();
    SceneData::ImportedModelContext* ctx = findCharacter(character);
    if (!ctx) return Result::fail("animated character not found: " + character);
    out = infoFromContext(*ctx);
    return Result::success();
}

Result listAnimClips(const std::string& character, std::vector<AnimClipInfo>& out) {
    SceneData::ImportedModelContext* ctx = nullptr;
    if (Result r = requireCharacter(character, ctx); !r) return r;
    out.clear();
    for (const AnimationClip& clip : ctx->animator->getAllClips()) {
        AnimClipInfo info;
        info.name = clip.name;
        info.duration_seconds = clip.getDurationInSeconds();
        info.ticks_per_second = clip.ticksPerSecond;
        info.loop = clip.loop;
        info.start_frame = clip.startFrame;
        info.end_frame = clip.endFrame;
        out.push_back(std::move(info));
    }
    return Result::success();
}

Result playAnimClip(const std::string& character, const std::string& clip,
                    float blend_seconds, int layer) {
    SceneData::ImportedModelContext* ctx = nullptr;
    if (Result r = requireCharacter(character, ctx); !r) return r;
    if (Result r = requireLayer(layer); !r) return r;
    if (blend_seconds < 0.0f) return Result::fail("blend_seconds must not be negative");
    // play() on an unknown clip is a silent no-op in the controller, which from
    // a script reads as "it worked but nothing moved".
    if (!ctx->animator->getClip(clip))
        return Result::fail("animation clip not found on " + character + ": " + clip);
    ctx->animator->play(clip, blend_seconds, layer);
    return Result::success();
}

Result stopAnimation(const std::string& character, float blend_out_seconds, int layer) {
    SceneData::ImportedModelContext* ctx = nullptr;
    if (Result r = requireCharacter(character, ctx); !r) return r;
    if (Result r = requireLayer(layer); !r) return r;
    if (blend_out_seconds < 0.0f)
        return Result::fail("blend_out_seconds must not be negative");
    ctx->animator->stop(layer, blend_out_seconds);
    return Result::success();
}

Result setAnimPaused(const std::string& character, bool paused) {
    SceneData::ImportedModelContext* ctx = nullptr;
    if (Result r = requireCharacter(character, ctx); !r) return r;
    ctx->animator->setPaused(paused);
    // The graph runs on its own clock, so pausing the controller alone would
    // leave a graph-driven character playing.
    if (auto graph = activeGraphOf(*ctx)) graph->setPlaybackPaused(paused);
    return Result::success();
}

Result setAnimTime(const std::string& character, float seconds, int layer) {
    SceneData::ImportedModelContext* ctx = nullptr;
    if (Result r = requireCharacter(character, ctx); !r) return r;
    if (Result r = requireLayer(layer); !r) return r;
    if (seconds < 0.0f) return Result::fail("time must not be negative");
    ctx->animator->setTime(seconds, layer);
    return Result::success();
}

Result setAnimSpeed(const std::string& character, float speed, int layer) {
    SceneData::ImportedModelContext* ctx = nullptr;
    if (Result r = requireCharacter(character, ctx); !r) return r;
    if (Result r = requireLayer(layer); !r) return r;
    ctx->animator->setSpeed(speed, layer);
    return Result::success();
}

Result setAnimLoop(const std::string& character, bool loop, int layer) {
    SceneData::ImportedModelContext* ctx = nullptr;
    if (Result r = requireCharacter(character, ctx); !r) return r;
    if (Result r = requireLayer(layer); !r) return r;
    ctx->animator->setLoop(loop, layer);
    return Result::success();
}

Result getAnimPlayback(const std::string& character, int layer, AnimPlaybackInfo& out) {
    SceneData::ImportedModelContext* ctx = nullptr;
    if (Result r = requireCharacter(character, ctx); !r) return r;
    if (Result r = requireLayer(layer); !r) return r;
    out.layer = layer;
    out.clip = ctx->animator->getCurrentClipName(layer);
    out.playing = ctx->animator->isPlaying(layer);
    out.paused = ctx->animator->isPaused();
    out.blending = ctx->animator->isBlending(layer);
    out.time = ctx->animator->getCurrentTime(layer);
    out.normalized_time = ctx->animator->getNormalizedTime(layer);
    return Result::success();
}

Result setAnimGraphFloat(const std::string& character, const std::string& name, float value) {
    if (name.empty()) return Result::fail("parameter name must not be empty");
    std::shared_ptr<AnimationGraph::AnimationNodeGraph> graph;
    if (Result r = requireGraph(character, graph); !r) return r;
    graph->setFloatParam(name, value);
    return Result::success();
}

Result setAnimGraphBool(const std::string& character, const std::string& name, bool value) {
    if (name.empty()) return Result::fail("parameter name must not be empty");
    std::shared_ptr<AnimationGraph::AnimationNodeGraph> graph;
    if (Result r = requireGraph(character, graph); !r) return r;
    graph->setBoolParam(name, value);
    return Result::success();
}

Result triggerAnimGraphParam(const std::string& character, const std::string& name) {
    if (name.empty()) return Result::fail("parameter name must not be empty");
    std::shared_ptr<AnimationGraph::AnimationNodeGraph> graph;
    if (Result r = requireGraph(character, graph); !r) return r;
    graph->triggerParam(name);
    return Result::success();
}

Result getAnimGraphPlayback(const std::string& character, AnimPlaybackInfo& out) {
    std::shared_ptr<AnimationGraph::AnimationNodeGraph> graph;
    if (Result r = requireGraph(character, graph); !r) return r;
    const AnimationGraph::AnimationNodeGraph::PlaybackStatus status = graph->getPlaybackStatus();
    out.clip = status.clipName;
    out.playing = status.isPlaying;
    out.paused = status.isPaused;
    out.blending = false;              // the graph reports blending per node, not globally
    out.normalized_time = status.normalizedTime;
    out.time = 0.0f;                   // graph time is per clip node; not a single value
    out.layer = 0;
    return Result::success();
}

} // namespace rtapi
