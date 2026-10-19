#include "Audio/AudioWorkspaceState.h"
#include "Audio/AudioMixer.h"
#include <stdexcept>
#include <cmath>
#include <set>
#include <algorithm>

namespace audio {
using json = nlohmann::json;
json defaultAudioWorkspace() {
    return {{"mixer_bus", "Master"}, {"analysis_mode", "live"}, {"history_scope", "scene"},
            {"analysis_view", "line"}, {"analysis_zoom_x", 1.0}, {"analysis_zoom_y", 1.0},
            {"analysis_pan_x", 0.0}, {"analysis_pan_y", 0.0},
            {"acoustic_all_sources", false}, {"acoustic_band", -1},
            {"acoustic_reflection_samples", 1}, {"selected_clip", ""}, {"selected_clips", json::array()}, {"selected_track", ""}, {"snap", true}, {"waveforms", true},
            {"inspector_open", false}, {"designer_open", false}, {"designer_page", "generate"}, {"designer_edit_view", "source"}, {"designer_envelope", "none"},
            {"audition_start_seconds", 0.0}, {"audition_end_seconds", 5.0},
            {"designer_canvas_zoom", 1.0}, {"designer_canvas_pan_x", 0.0},
            {"designer_canvas_pan_y", 0.0}, {"designer_tools_open", true}, {"designer_properties_open", true},
            {"designer_zoom", 1.0}, {"designer_pan", 0.0}, {"designer_effect", 0}, {"designer_layer", 0},
            {"designer_eq_band", 0}, {"designer_eq_range_db", 24.0},
            {"designer_eq_auto_range", true}, {"designer_eq_height", 280.0},
            {"designer_panel_zoom", 1.0}, {"designer_panel_pan_x", 0.0},
            {"designer_panel_pan_y", 0.0}};
}

json updateAudioWorkspace(const json& current, const json& patch, const json& clips, const json& tracks) {
    auto candidate = current;
    for (auto entry = patch.begin(); entry != patch.end(); ++entry) {
        if (entry.key() == "history_scope") {
            if (!entry.value().is_string() ||
                (entry.value() != "scene" && entry.value() != "design")) {
                throw std::invalid_argument("history_scope must be scene or design");
            }
        } else if (entry.key() == "analysis_view") {
            if (!entry.value().is_string() ||
                (entry.value() != "line" && entry.value() != "filled" &&
                 entry.value() != "bands" && entry.value() != "spectrogram")) {
                throw std::invalid_argument("analysis_view must be line, filled, bands or spectrogram");
            }
        } else if (entry.key() == "analysis_zoom_x" || entry.key() == "analysis_zoom_y" ||
                   entry.key() == "analysis_pan_x" || entry.key() == "analysis_pan_y") {
            const bool zoom = entry.key() == "analysis_zoom_x" || entry.key() == "analysis_zoom_y";
            const double low = zoom ? 1.0 : 0.0;
            const double high = zoom ? 32.0 : 1.0;
            if (!entry.value().is_number() || !std::isfinite(entry.value().get<double>()) ||
                entry.value().get<double>() < low || entry.value().get<double>() > high) {
                throw std::invalid_argument("Analysis zoom must be 1..32; pan must be 0..1, finite");
            }
        } else if (entry.key() == "analysis_mode") {
            if (!entry.value().is_string() ||
                (entry.value() != "live" && entry.value() != "design")) {
                throw std::invalid_argument("analysis_mode must be live or design");
            }
        } else if (entry.key() == "mixer_bus") {
            if (!entry.value().is_string()) {
                throw std::invalid_argument("mixer_bus must be a defined bus name");
            }
            const auto& names = AudioMixer::names();
            if (std::find(names.begin(), names.end(), entry.value().get<std::string>()) == names.end()) {
                throw std::invalid_argument("Unknown mixer bus");
            }
        } else if (entry.key() == "audition_start_seconds" || entry.key() == "audition_end_seconds") {
            if (!entry.value().is_number() || !std::isfinite(entry.value().get<double>()) ||
                entry.value().get<double>() < 0 || entry.value().get<double>() > 86400) {
                throw std::invalid_argument("Audition range must be finite and within the day");
            }
        } else if (entry.key() == "selected_clips") {
            if (!entry.value().is_array() || entry.value().size() > 1024) {
                throw std::invalid_argument("selected_clips must be an array of at most 1024 IDs");
            }
            std::set<std::string> ids;
            for (const auto& id : entry.value()) {
                if (!id.is_string() || !ids.insert(id.get<std::string>()).second) {
                    throw std::invalid_argument("Selected clip IDs must be unique strings");
                }
                bool found = false;
                for (const auto& clip : clips) {
                    found = found || clip.at("id") == id;
                }
                if (!found) {
                    throw std::invalid_argument("Selected audio clip does not exist");
                }
            }
        } else if (entry.key() == "designer_envelope") {
            if (!entry.value().is_string()) {
                throw std::invalid_argument("designer_envelope must be a string");
            }
            const auto value = entry.value().get<std::string>();
            if (value != "none" && value != "gain" && value != "pan") {
                throw std::invalid_argument("designer_envelope must be none, gain or pan");
            }
        } else if (entry.key() == "selected_track") {
            if (!entry.value().is_string()) {
                throw std::invalid_argument("selected_track must be a string");
            }
            const auto id = entry.value().get<std::string>();
            bool found = id.empty();
            for (const auto& track : tracks) {
                found = found || track.at("id").get<std::string>() == id;
            }
            if (!found) {
                throw std::invalid_argument("Selected audio track does not exist");
            }
        } else if (entry.key() == "selected_clip") {
            if (!entry.value().is_string()) {
                throw std::invalid_argument("selected_clip must be a string");
            }
            const auto id = entry.value().get<std::string>();
            bool found = id.empty();
            for (const auto& clip : clips) {
                found = found || clip.at("id") == id;
            }
            if (!found) {
                throw std::invalid_argument("Selected audio clip does not exist");
            }
        } else if (entry.key() == "designer_edit_view") {
            if (!entry.value().is_string() ||
                (entry.value().get<std::string>() != "source" &&
                 entry.value().get<std::string>() != "arrange")) {
                throw std::invalid_argument("designer_edit_view must be source or arrange");
            }
        } else if (entry.key() == "designer_page") {
            if (!entry.value().is_string()) {
                throw std::invalid_argument("designer_page must be a string");
            }
            const auto page = entry.value().get<std::string>();
            if (page != "generate" && page != "edit" && page != "effects" && page != "analyze") {
                throw std::invalid_argument("Unknown sound designer page");
            }
        } else if (entry.key() == "acoustic_band" ||
                   entry.key() == "acoustic_reflection_samples") {
            const int low = entry.key() == "acoustic_band" ? -1 : 0;
            const int high = entry.key() == "acoustic_band" ? 5 : 2;
            if (!entry.value().is_number_integer() || entry.value().get<double>() < low ||
                entry.value().get<double>() > high) {
                throw std::invalid_argument("Acoustic band must be -1..5; reflection samples 0..2");
            }
        } else if (entry.key() == "designer_effect" || entry.key() == "designer_layer" ||
                   entry.key() == "designer_eq_band") {
            if (!entry.value().is_number_integer() || entry.value().get<double>() < 0 ||
                entry.value().get<double>() > 7) {
                throw std::invalid_argument("Designer module selection must be integer 0..7");
            }
        } else if (entry.key() == "designer_panel_zoom" || entry.key() == "designer_eq_height") {
            const bool panelZoom = entry.key() == "designer_panel_zoom";
            const double low = panelZoom ? 0.5 : 180;
            const double high = panelZoom ? 1.5 : 600;
            if (!entry.value().is_number() || !std::isfinite(entry.value().get<double>()) ||
                entry.value().get<double>() < low || entry.value().get<double>() > high) {
                throw std::invalid_argument("Panel zoom must be 0.5..1.5; EQ height 180..600");
            }
        } else if (entry.key() == "designer_eq_range_db") {
            if (!entry.value().is_number() || !std::isfinite(entry.value().get<double>()) ||
                entry.value().get<double>() < 6 || entry.value().get<double>() > 288) {
                throw std::invalid_argument("EQ graph range must be finite, 6..288 dB");
            }
        } else if (entry.key() == "designer_zoom" || entry.key() == "designer_pan" ||
                   entry.key() == "designer_canvas_zoom" || entry.key() == "designer_canvas_pan_x" ||
                   entry.key() == "designer_canvas_pan_y" || entry.key() == "designer_panel_pan_x" ||
                   entry.key() == "designer_panel_pan_y") {
            const bool zoom = entry.key() == "designer_zoom" || entry.key() == "designer_canvas_zoom";
            const double low = zoom ? 1 : 0;
            const double high = entry.key() == "designer_zoom" ? 2048 :
                entry.key() == "designer_canvas_zoom" ? 8 : 1;
            if (!entry.value().is_number() || !std::isfinite(entry.value().get<double>()) ||
                entry.value().get<double>() < low || entry.value().get<double>() > high) {
                throw std::invalid_argument("Designer view value outside finite bounds");
            }
        } else if (entry.key() == "acoustic_all_sources" || entry.key() == "snap" || entry.key() == "waveforms" ||
                   entry.key() == "inspector_open" || entry.key() == "designer_open" ||
                   entry.key() == "designer_tools_open" || entry.key() == "designer_properties_open" ||
                   entry.key() == "designer_eq_auto_range") {
            if (!entry.value().is_boolean()) {
                throw std::invalid_argument("Audio workspace toggles must be boolean");
            }
        } else {
            throw std::invalid_argument("Unknown audio workspace parameter");
        }
        candidate[entry.key()] = entry.value();
    }
    if (patch.contains("selected_clip") && !patch.contains("selected_clips")) {
        candidate["selected_clips"] = json::array();
        if (!candidate.at("selected_clip").get<std::string>().empty()) {
            candidate["selected_clips"].push_back(candidate.at("selected_clip"));
        }
    } else if (patch.contains("selected_clips") && !patch.contains("selected_clip")) {
        candidate["selected_clip"] = candidate.at("selected_clips").empty() ? json("") :
            candidate.at("selected_clips").front();
    }
    const double span = candidate.at("audition_end_seconds").get<double>() -
        candidate.at("audition_start_seconds").get<double>();
    if (span < 0.05 - 1e-7 || span > 30 + 1e-7) {
        throw std::invalid_argument("Audition range must span 0.05..30 seconds");
    }
    const auto primary = candidate.at("selected_clip").get<std::string>();
    const auto& selected = candidate.at("selected_clips");
    if ((!primary.empty() && std::find(selected.begin(), selected.end(), json(primary)) == selected.end()) ||
        (primary.empty() && !selected.empty())) {
        throw std::invalid_argument("Primary clip must belong to the selection");
    }
    return candidate;
}
void pruneAudioSelections(json& workspace, const json& clips, const json& tracks) {
    auto& selected = workspace["selected_clips"];
    if (!selected.is_array()) {
        selected = json::array();
    }
    selected.erase(std::remove_if(selected.begin(), selected.end(), [&](const json& id) {
        return std::none_of(clips.begin(), clips.end(), [&](const json& clip) {
            return clip.at("id") == id;
        });
    }), selected.end());
    const auto primary = workspace.at("selected_clip");
    if (std::find(selected.begin(), selected.end(), primary) == selected.end()) {
        workspace["selected_clip"] = selected.empty() ? json("") : selected.front();
    }
    const auto trackId = workspace.at("selected_track");
    if (std::none_of(tracks.begin(), tracks.end(), [&](const json& track) {
        return track.at("id") == trackId;
    })) {
        workspace["selected_track"] = "";
    }
}

}
