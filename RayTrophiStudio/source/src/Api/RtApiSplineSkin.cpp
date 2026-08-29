#include "RtApiInternal.h"

#include "GeometryNodesV2.h"
#include "MeshEdit/SplineObject.h"
#include "ProjectManager.h"
#include "TriangleMesh.h"

#include <cmath>

namespace rtapi {
namespace {

std::shared_ptr<MeshEdit::SplineObject> findSplineObject(const std::string& name) {
    if (!g_ctx) return {};
    for (const auto& object : g_ctx->scene.world.objects) {
        auto spline = std::dynamic_pointer_cast<MeshEdit::SplineObject>(object);
        if (spline && spline->nodeName == name) return spline;
    }
    return {};
}

bool sceneObjectExists(const std::string& name) {
    if (!g_ctx) return false;
    for (const auto& object : g_ctx->scene.world.objects) {
        if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object)) {
            if (mesh->nodeName == name) return true;
        }
    }
    return false;
}

void storeDisplaySettings(MeshEdit::SplineObject& object,
                          const SplineSkinSettings& settings,
                          const std::string& host) {
    auto& display = object.skin_display;
    display.enabled = true; display.host_name = host;
    display.custom_profile = settings.custom_profile; display.radius = settings.radius;
    display.path_samples = settings.path_samples;
    display.radial_segments = settings.radial_segments;
    display.cap_start = settings.cap_start; display.cap_end = settings.cap_end;
    display.use_point_radius = settings.use_point_radius;
    display.taper_start = settings.taper_start; display.taper_end = settings.taper_end;
    display.taper_falloff = settings.taper_falloff;
    display.twist_start_degrees = settings.twist_start_degrees;
    display.twist_end_degrees = settings.twist_end_degrees;
    display.wave_amplitude = settings.wave_amplitude;
    display.wave_cycles = settings.wave_cycles;
    display.wave_phase_degrees = settings.wave_phase_degrees;
    display.wave_noise = settings.wave_noise; display.wave_seed = settings.wave_seed;
    display.wave_axis = settings.wave_axis;
}

Result setStringProperty(const std::string& graph, unsigned int node,
                         const std::string& property, const std::string& value) {
    NodeParamValue param;
    param.kind = NodeParamValue::Kind::String;
    param.string_value = value;
    return setNodeProperty("geometry", graph, node, property, param);
}

Result setFloatProperty(const std::string& graph, unsigned int node,
                        const std::string& property, float value) {
    NodeParamValue param;
    param.kind = NodeParamValue::Kind::Float;
    param.floats[0] = value;
    return setNodeProperty("geometry", graph, node, property, param);
}

Result setIntProperty(const std::string& graph, unsigned int node,
                      const std::string& property, int value) {
    NodeParamValue param;
    param.kind = NodeParamValue::Kind::Int;
    param.int_value = value;
    return setNodeProperty("geometry", graph, node, property, param);
}

Result setBoolProperty(const std::string& graph, unsigned int node,
                       const std::string& property, bool value) {
    NodeParamValue param;
    param.kind = NodeParamValue::Kind::Bool;
    param.bool_value = value;
    return setNodeProperty("geometry", graph, node, property, param);
}

} // namespace

Result createSplineSkin(const std::string& splineName, const std::string& requestedOutput,
                        float radius, int pathSamples, int radialSegments,
                        bool capStart, bool capEnd, bool usePointRadius,
                        SplineSkinInfo& out) {
    SplineSkinSettings settings;
    settings.radius = radius;
    settings.path_samples = pathSamples;
    settings.radial_segments = radialSegments;
    settings.cap_start = capStart;
    settings.cap_end = capEnd;
    settings.use_point_radius = usePointRadius;
    return createSplineSkinAdvanced(splineName, requestedOutput, settings, out);
}

Result createSplineSkinAdvanced(const std::string& splineName,
                                const std::string& requestedOutput,
                                const SplineSkinSettings& settings,
                                SplineSkinInfo& out) {
    out = {};
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!std::isfinite(settings.radius) || settings.radius <= 0.0f)
        return Result::fail("radius must be finite and greater than zero");
    if (settings.path_samples < 2 || settings.path_samples > 1024)
        return Result::fail("path_samples must be between 2 and 1024");
    if (settings.radial_segments < 3 || settings.radial_segments > 256)
        return Result::fail("radial_segments must be between 3 and 256");
    if (!std::isfinite(settings.taper_start) || !std::isfinite(settings.taper_end) ||
        settings.taper_start < 0.0f || settings.taper_end < 0.0f ||
        !std::isfinite(settings.taper_falloff) || settings.taper_falloff < 0.01f)
        return Result::fail("taper scales must be non-negative and falloff must be at least 0.01");
    if (!std::isfinite(settings.twist_start_degrees) ||
        !std::isfinite(settings.twist_end_degrees) ||
        !std::isfinite(settings.wave_amplitude) || !std::isfinite(settings.wave_cycles) ||
        !std::isfinite(settings.wave_phase_degrees) || !std::isfinite(settings.wave_noise) ||
        settings.wave_noise < 0.0f || settings.wave_axis < 0 || settings.wave_axis > 2)
        return Result::fail("twist or wave settings are outside the supported range");
    const auto sourceObject = findSplineObject(splineName);
    if (!sourceObject) return Result::fail("spline object not found: " + splineName);
    const bool restoreSourceSelection =
        g_ctx->selection.selected.spline_object == sourceObject;
    std::string splinePayload;
    if (Result splineResult = getSpline(splineName, splinePayload); !splineResult)
        return splineResult;
    try {
        const auto spline = nlohmann::json::parse(splinePayload);
        if (spline.value("closed", false))
            return Result::fail("Spline Skin requires an open path spline");
        if (!spline.contains("points") || spline["points"].size() < 2)
            return Result::fail("Spline Skin requires at least two control points");
    } catch (const std::exception& error) {
        return Result::fail(std::string("invalid spline source: ") + error.what());
    }

    if (!settings.custom_profile.empty()) {
        if (settings.custom_profile == splineName)
            return Result::fail("custom_profile must be a different closed spline");
        std::string profilePayload;
        if (Result profileResult = getSpline(settings.custom_profile, profilePayload); !profileResult)
            return Result::fail("custom_profile: " + profileResult.error);
        try {
            const auto profile = nlohmann::json::parse(profilePayload);
            if (!profile.value("closed", false))
                return Result::fail("custom_profile must be a closed spline");
            if (!profile.contains("points") || profile["points"].size() < 3)
                return Result::fail("custom_profile requires at least three control points");
        } catch (const std::exception& error) {
            return Result::fail(std::string("invalid custom_profile source: ") + error.what());
        }
    }

    std::string hostName = sourceObject->skin_display.host_name;
    bool createdHost = false;
    if (hostName.empty() || !sceneObjectExists(hostName)) {
        const std::string desired = requestedOutput.empty() ? splineName + "_SkinPreview" : requestedOutput;
        if (Result created = addPrimitive("cube", desired, 0.01f, hostName); !created)
            return created;
        createdHost = true;
    }
    auto rollback = [&](const Result& failure) {
        g_ctx->scene.geometry_node_graphs.erase(hostName);
        if (createdHost) deleteObject(hostName);
        return failure;
    };
    // Rebuild the procedural graph on the same host. No new scene geometry object
    // is created while display parameters change.
    g_ctx->scene.geometry_caches.erase(hostName);
    g_ctx->scene.geometry_node_graphs.erase(hostName);
    if (Result graphResult = createNodeGraph("geometry", hostName); !graphResult)
        return rollback(graphResult);

    unsigned int source = 0, profile = 0, taper = 0, twist = 0, wave = 0;
    unsigned int skin = 0, output = 0, link = 0;
    if (Result result = addNode("geometry", hostName, "GeoV2.SplineObject", source); !result)
        return rollback(result);
    if (Result result = addNode("geometry", hostName, "GeoV2.CurveTaper", taper); !result)
        return rollback(result);
    if (Result result = addNode("geometry", hostName, "GeoV2.CurveTwist", twist); !result)
        return rollback(result);
    if (Result result = addNode("geometry", hostName, "GeoV2.CurveWave", wave); !result)
        return rollback(result);
    if (Result result = addNode("geometry", hostName, "GeoV2.CurveToMesh", skin); !result)
        return rollback(result);
    if (!settings.custom_profile.empty()) {
        if (Result result = addNode(
                "geometry", hostName, "GeoV2.SplineObject", profile); !result)
            return rollback(result);
    }
    if (Result result = addNode("geometry", hostName, "GeoV2.Output", output); !result)
        return rollback(result);
    if (Result result = setStringProperty(hostName, source, "object", splineName); !result)
        return rollback(result);
    if (profile != 0) {
        if (Result result = setStringProperty(
                hostName, profile, "object", settings.custom_profile); !result)
            return rollback(result);
    }
    if (Result result = setFloatProperty(hostName, taper, "start_scale", settings.taper_start); !result)
        return rollback(result);
    if (Result result = setFloatProperty(hostName, taper, "end_scale", settings.taper_end); !result)
        return rollback(result);
    if (Result result = setFloatProperty(hostName, taper, "exponent", settings.taper_falloff); !result)
        return rollback(result);
    if (Result result = setFloatProperty(hostName, twist, "start_degrees", settings.twist_start_degrees); !result)
        return rollback(result);
    if (Result result = setFloatProperty(hostName, twist, "end_degrees", settings.twist_end_degrees); !result)
        return rollback(result);
    if (Result result = setFloatProperty(hostName, wave, "amplitude", settings.wave_amplitude); !result)
        return rollback(result);
    if (Result result = setFloatProperty(hostName, wave, "frequency", settings.wave_cycles); !result)
        return rollback(result);
    if (Result result = setFloatProperty(hostName, wave, "phase_degrees", settings.wave_phase_degrees); !result)
        return rollback(result);
    if (Result result = setFloatProperty(hostName, wave, "noise", settings.wave_noise); !result)
        return rollback(result);
    if (Result result = setIntProperty(hostName, wave, "seed", settings.wave_seed); !result)
        return rollback(result);
    if (Result result = setIntProperty(hostName, wave, "axis", settings.wave_axis); !result)
        return rollback(result);
    if (Result result = setFloatProperty(hostName, skin, "radius", settings.radius); !result)
        return rollback(result);
    if (Result result = setIntProperty(hostName, skin, "path_samples", settings.path_samples); !result)
        return rollback(result);
    if (Result result = setIntProperty(hostName, skin, "profile_samples", settings.radial_segments); !result)
        return rollback(result);
    if (Result result = setBoolProperty(hostName, skin, "cap_start", settings.cap_start); !result)
        return rollback(result);
    if (Result result = setBoolProperty(hostName, skin, "cap_end", settings.cap_end); !result)
        return rollback(result);
    if (Result result = setBoolProperty(
            hostName, skin, "use_point_radius", settings.use_point_radius); !result)
        return rollback(result);
    if (Result result = linkNodes("geometry", hostName, source, 0, taper, 0, link); !result)
        return rollback(result);
    if (Result result = linkNodes("geometry", hostName, taper, 0, twist, 0, link); !result)
        return rollback(result);
    if (Result result = linkNodes("geometry", hostName, twist, 0, wave, 0, link); !result)
        return rollback(result);
    if (Result result = linkNodes("geometry", hostName, wave, 0, skin, 0, link); !result)
        return rollback(result);
    if (profile != 0) {
        if (Result result = linkNodes("geometry", hostName, profile, 0, skin, 1, link); !result)
            return rollback(result);
    }
    if (Result result = linkNodes("geometry", hostName, skin, 0, output, 0, link); !result)
        return rollback(result);

    auto graphIt = g_ctx->scene.geometry_node_graphs.find(hostName);
    if (graphIt == g_ctx->scene.geometry_node_graphs.end() || !graphIt->second)
        return rollback(Result::fail("Spline Skin graph was not created"));
    graphIt->second->liveCurvePreview = true;
    NodeGraphApplyInfo apply;
    if (Result result = applyNodeGraph("geometry", hostName, apply); !result)
        return rollback(result);
    ObjectInfo mesh;
    if (Result result = getObjectInfo(hostName, mesh); !result)
        return rollback(result);

    out.object_name = hostName;
    out.source_node = source;
    out.profile_node = profile;
    out.taper_node = taper;
    out.twist_node = twist;
    out.wave_node = wave;
    out.skin_node = skin;
    out.output_node = output;
    out.vertex_count = mesh.vertex_count;
    out.triangle_count = mesh.triangle_count;
    storeDisplaySettings(*sourceObject, settings, hostName);
    sourceObject->skin_display_status = "Live skin preview: " + hostName;
    if (restoreSourceSelection) {
        for (size_t i = 0; i < g_ctx->scene.world.objects.size(); ++i) {
            if (g_ctx->scene.world.objects[i].get() == sourceObject.get()) {
                g_ctx->selection.selectObject(
                    sourceObject, static_cast<int>(i), sourceObject->nodeName);
                break;
            }
        }
    }
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result finalizeSplineSkinPreview(const std::string& splineName, SplineSkinInfo& out) {
    out = {};
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    const auto source = findSplineObject(splineName);
    if (!source) return Result::fail("spline object not found: " + splineName);
    const std::string host = source->skin_display.host_name;
    if (!source->skin_display.enabled || host.empty() || !sceneObjectExists(host))
        return Result::fail("spline has no live skin preview to convert");
    NodeGraphApplyInfo apply;
    if (Result result = applyNodeGraph("geometry", host, apply); !result) return result;
    g_ctx->scene.geometry_node_graphs.erase(host);
    ObjectInfo mesh;
    if (Result result = getObjectInfo(host, mesh); !result) return result;
    out.object_name = host; out.vertex_count = mesh.vertex_count;
    out.triangle_count = mesh.triangle_count;
    source->skin_display.enabled = false;
    source->skin_display.host_name.clear();
    source->skin_display_status = "Converted to mesh: " + host;
    for (size_t i = 0; i < g_ctx->scene.world.objects.size(); ++i) {
        auto finalMesh = std::dynamic_pointer_cast<TriangleMesh>(
            g_ctx->scene.world.objects[i]);
        if (finalMesh && finalMesh->nodeName == host) {
            g_ctx->selection.selectObject(
                finalMesh, static_cast<int>(i), finalMesh->nodeName);
            break;
        }
    }
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result clearSplineSkinPreview(const std::string& splineName) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    const auto source = findSplineObject(splineName);
    if (!source) return Result::fail("spline object not found: " + splineName);
    const std::string host = source->skin_display.host_name;
    if (!host.empty()) {
        g_ctx->scene.geometry_node_graphs.erase(host);
        if (sceneObjectExists(host)) {
            if (Result deleted = deleteObject(host); !deleted) return deleted;
        }
    }
    source->skin_display.enabled = false;
    source->skin_display.host_name.clear();
    source->skin_display_status = "Skin preview removed.";
    ProjectManager::getInstance().markModified();
    return Result::success();
}

} // namespace rtapi
