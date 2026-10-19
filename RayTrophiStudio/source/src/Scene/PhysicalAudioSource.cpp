#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/PhysicalAudioSource.h"
#include "Audio/PhysicalAudioDsp.h"
#include "Audio/PhysicalAudioSchedule.h"
#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace audio {
namespace {
using json = nlohmann::json;

void require(bool valid, const std::string& message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
}

bool isPhysicalAudioSource(const json& source) {
    return source.contains("generator") && !source.at("generator").is_null();
}

void validatePhysicalAudioSource(json& source) {
    if (!isPhysicalAudioSource(source)) {
        source["generator"] = nullptr;
        return;
    }
    const auto recipe = source.at("generator");
    require(recipe.is_object(), "generator must be an object or null");
    for (auto field = recipe.begin(); field != recipe.end(); ++field) {
        require(field.key() == "version" || field.key() == "model" ||
            field.key() == "seed" || field.key() == "brightness" ||
            field.key() == "resonator" || field.key() == "size_m" ||
            field.key() == "decay_seconds" || field.key() == "reference_energy_j" ||
            field.key() == "texture" || field.key() == "variation" || field.key() == "body" ||
            field.key() == "character_revision" || field.key() == "surface",
            "Unknown physical generator field");
    }
    const auto version = recipe.value("version", json(1));
    require(version.is_number_integer() && (version == 1 || version == 2),
        "Unsupported physical generator version");
    require(version == 2 || (!recipe.contains("texture") && !recipe.contains("variation") &&
        !recipe.contains("body") && !recipe.contains("character_revision") &&
        !recipe.contains("surface")), "Character controls require generator version 2");
    const auto character = recipe.value("character_revision", json(1));
    require(character.is_number_integer() && (character == 1 || character == 2),
        "character_revision must be integer 1 or 2");
    const auto model = recipe.value("model", json("liquid_flow"));
    require(model.is_string() && (model == "liquid_flow" || model == "gas_flow" ||
        model == "friction" || model == "rolling" || model == "impact" ||
        model == "contact_activity" || model == "deformation" || model == "fracture_activity"),
        "Unknown physical generator model");
    const bool surfaceControl = version == 2 && character == 2 && model == "friction";
    require(surfaceControl || !recipe.contains("surface"),
        "surface requires v2 friction character_revision 2");
    const auto seed = recipe.value("seed", json(1));
    require(seed.is_number_integer() && seed.get<double>() >= 0 &&
        seed.get<double>() <= 4294967295.0, "generator seed must be a uint32 integer");
    const auto brightness = recipe.value("brightness", json(0.5));
    require(brightness.is_number() && std::isfinite(brightness.get<double>()) &&
        brightness.get<double>() >= 0 && brightness.get<double>() <= 1,
        "generator brightness must be finite within 0..1");
    require(source.at("asset_id") == "", "generator and asset_id are mutually exclusive");
    require(source.at("loop").get<bool>(), "Continuous physical generators require loop=true");
    const auto& reaction = source.at("reaction");
    require(model == "impact" ? reaction.at("driver") == "manual" :
        (reaction.at("driver") == "domain_contact" || reaction.at("driver") == "signal"),
        "Impact requires manual driver; continuous models require domain_contact or signal");
    const bool resonant = model == "impact" || model == "contact_activity" ||
        model == "deformation" || model == "fracture_activity";
    require(resonant || (!recipe.contains("resonator") && !recipe.contains("size_m") &&
        !recipe.contains("decay_seconds")), "Resonator controls require a resonant generator");
    require(model == "impact" || !recipe.contains("reference_energy_j"),
        "reference_energy_j requires an explicit impact model");
    if (reaction.at("driver") == "domain_contact") {
        const std::string expected = model == "liquid_flow" ? "fluid_flow" :
            model == "gas_flow" ? "gas_flow" : model == "friction" ? "dem_friction" :
            model == "deformation" ? "mpm_deformation" :
            model == "fracture_activity" ? "mpm_fracture" : "dem_rolling";
        const bool contact = model == "contact_activity" &&
            (reaction.at("domain_channel") == "dem_contact" ||
             reaction.at("domain_channel") == "mixed_contact");
        require(contact || (model != "contact_activity" && reaction.at("domain_channel") == expected),
            "Physical generator model does not match the measured domain channel");
        const char* metric = model == "fracture_activity" ? "value" : "power_w";
        require(reaction.at("input_metric") == metric,
            "Domain generators require power_w, except fracture_activity which requires value");
    }
    source["generator"] = {{"version", version}, {"model", model}, {"seed", seed},
        {"brightness", brightness}};
    if (version == 2) {
        source["generator"]["character_revision"] = character;
        if (surfaceControl) {
            const auto surface = recipe.value("surface", json("organic"));
            require(surface.is_string() && (surface == "organic" || surface == "mineral" ||
                surface == "metal" || surface == "glass"), "Unknown friction surface family");
            source["generator"]["surface"] = surface;
        }
        for (const auto* key : {"texture", "variation", "body"}) {
            const auto value = recipe.value(key, json(0.5));
            require(value.is_number() && std::isfinite(value.get<double>()) &&
                value.get<double>() >= 0 && value.get<double>() <= 1,
                std::string("generator ") + key + " must be finite within 0..1");
            source["generator"][key] = value;
        }
    }
    if (resonant) {
        const auto resonator = recipe.value("resonator", json("wood"));
        require(resonator.is_string() && (resonator == "wood" || resonator == "metal" ||
            resonator == "stone" || resonator == "glass"), "Unknown physical resonator");
        source["generator"]["resonator"] = resonator;
        for (const auto* key : {"size_m", "decay_seconds", "reference_energy_j"}) {
            if (model != "impact" && std::string(key) == "reference_energy_j") {
                continue;
            }
            const double fallback = std::string(key) == "size_m" ? 0.1 :
                std::string(key) == "decay_seconds" ? 0.4 : 1.0;
            const double lower = std::string(key) == "reference_energy_j" ? 1e-9 : 0.005;
            const double upper = std::string(key) == "reference_energy_j" ? 1e9 : 10.0;
            const auto value = recipe.value(key, json(fallback));
            require(value.is_number() && std::isfinite(value.get<double>()) &&
                value.get<double>() >= lower && value.get<double>() <= upper,
                "Impact size/decay/reference energy is outside its finite range");
            source["generator"][key] = value;
        }
    }
}

struct PhysicalAudioRuntime::Source {
    struct Data {
        ma_data_source_base base;
        Source* owner = nullptr;
    } data{};
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    std::atomic<std::uint32_t> target{0};
    std::atomic<std::uint64_t> cursor{0};
    std::atomic<std::uint32_t> blockPeak{0};
    bool draining = false;
    float drainSeconds = 0;
    float drainAllowance = 1.25f;
    float lastGain = 0;
    PhysicalAudioDsp dsp;
    std::shared_ptr<PhysicalAudioSchedule> schedule;
    std::atomic<std::uint64_t> replayPosition{0};
    std::atomic<std::uint32_t> replayEnded{0};
    std::uint64_t frame = 0;
    bool initialized = false;
    // Single producer (service main thread), single consumer (data-source callback).
    std::array<float, 32> impulses{};
    std::atomic<std::uint32_t> written{0};
    std::atomic<std::uint32_t> consumed{0};
    bool hasSequence = false;
    double referenceEnergy = 1;
    explicit Source(const json& recipe, std::shared_ptr<PhysicalAudioSchedule> replay)
        : dsp(recipe), schedule(std::move(replay)) {
        const auto& model = recipe.at("model");
        if ((model == "contact_activity" || model == "deformation" ||
            model == "fracture_activity") && !schedule) {
            drainAllowance += recipe.at("decay_seconds").get<float>();
        }
        if (schedule) {
            frame = schedule->cursor();
            replayPosition.store(frame, std::memory_order_relaxed);
        }
        if (dsp.impactMode()) {
            referenceEnergy = recipe.at("reference_energy_j").get<double>();
        }
        static const ma_data_source_vtable table = {
            read, nullptr, format, position, nullptr, nullptr, 0};
        auto config = ma_data_source_config_init();
        config.vtable = &table;
        data.owner = this;
        if (ma_data_source_init(&config, &data.base) != MA_SUCCESS) {
            throw std::runtime_error("Could not initialize physical audio data source");
        }
        initialized = true;
    }

    ~Source() {
        if (initialized) {
            ma_data_source_uninit(&data.base);
        }
    }

    static Source& owner(ma_data_source* dataSource) {
        return *reinterpret_cast<Data*>(dataSource)->owner;
    }

    static ma_result read(ma_data_source* dataSource, void* output, ma_uint64 count,
                          ma_uint64* framesRead) {
        auto& source = owner(dataSource);
        auto* samples = static_cast<float*>(output);
        if (count == 0) {
            if (framesRead) {
                *framesRead = 0;
            }
            return MA_SUCCESS;
        }
        if (source.schedule) {
            const float peak = source.schedule->render(samples, static_cast<std::size_t>(count));
            source.frame += count;
            source.cursor.store(source.frame, std::memory_order_relaxed);
            source.blockPeak.store(std::bit_cast<std::uint32_t>(peak), std::memory_order_relaxed);
            source.replayPosition.store(source.schedule->cursor(), std::memory_order_relaxed);
            source.replayEnded.store(source.schedule->cursor() == source.schedule->frames(),
                                     std::memory_order_relaxed);
            if (framesRead) {
                *framesRead = count;
            }
            return MA_SUCCESS;
        }
        const float target = std::bit_cast<float>(source.target.load(std::memory_order_relaxed));
        float excitation = 0;
        if (source.dsp.impactMode()) {
            auto read = source.consumed.load(std::memory_order_relaxed);
            const auto write = source.written.load(std::memory_order_acquire);
            while (read != write) {
                excitation += source.impulses[read % source.impulses.size()];
                ++read;
            }
            source.consumed.store(read, std::memory_order_release);
            excitation = std::min(excitation, 1.0f);
        }
        const float peak = source.dsp.render(samples, static_cast<std::size_t>(count),
                                             target, excitation);
        source.frame += count;
        source.blockPeak.store(std::bit_cast<std::uint32_t>(peak), std::memory_order_relaxed);
        source.cursor.store(source.frame, std::memory_order_relaxed);
        if (framesRead) {
            *framesRead = count;
        }
        return MA_SUCCESS;
    }

    static ma_result format(ma_data_source*, ma_format* sampleFormat, ma_uint32* channels,
                            ma_uint32* rate, ma_channel* channelMap, size_t mapCapacity) {
        if (sampleFormat) {
            *sampleFormat = ma_format_f32;
        }
        if (channels) {
            *channels = 1;
        }
        if (rate) {
            *rate = 48000;
        }
        if (channelMap && mapCapacity > 0) {
            channelMap[0] = MA_CHANNEL_MONO;
        }
        return MA_SUCCESS;
    }

    static ma_result position(ma_data_source* dataSource, ma_uint64* position) {
        if (!position) {
            return MA_INVALID_ARGS;
        }
        *position = owner(dataSource).cursor.load(std::memory_order_relaxed);
        return MA_SUCCESS;
    }
};

PhysicalAudioRuntime::PhysicalAudioRuntime() = default;
PhysicalAudioRuntime::~PhysicalAudioRuntime() = default;

std::unique_ptr<ma_sound> PhysicalAudioRuntime::prepare(ma_engine& engine, ma_sound_group* bus,
                                                      const json& source) {
    const std::string id = source.at("id");
    const auto pending = preparedReplays_.find(id);
    auto data = std::make_unique<Source>(source.at("generator"),
        pending == preparedReplays_.end() ? nullptr : pending->second);
    auto sound = std::make_unique<ma_sound>();
    const auto result = ma_sound_init_from_data_source(&engine, &data->data.base,
        MA_SOUND_FLAG_NO_SPATIALIZATION, bus, sound.get());
    if (result != MA_SUCCESS) {
        throw std::runtime_error("Could not initialize physical audio voice");
    }
    try {
        sources_[id] = std::move(data);
        preparedReplays_.erase(id);
    } catch (...) {
        ma_sound_uninit(sound.get());
        throw;
    }
    return sound;
}

void PhysicalAudioRuntime::activity(const std::string& id, float value) {
    const auto found = sources_.find(id);
    if (found != sources_.end()) {
        const float safe = !found->second->draining && std::isfinite(value) ?
            std::clamp(value, 0.0f, 1.0f) : 0.0f;
        found->second->target.store(std::bit_cast<std::uint32_t>(safe),
                                    std::memory_order_relaxed);
    }
}

float PhysicalAudioRuntime::outputGain(const std::string& id, float value) {
    const auto found = sources_.find(id);
    if (found == sources_.end()) {
        return value;
    }
    auto& source = *found->second;
    if (source.schedule) {
        source.lastGain = 1;
        return 1;
    }
    if (!source.draining) {
        source.lastGain = value;
    }
    return source.lastGain;
}

bool PhysicalAudioRuntime::retire(const std::string& id, bool inactive, float dt) {
    const auto found = sources_.find(id);
    if (found == sources_.end()) {
        return false;
    }
    auto& source = *found->second;
    if (source.schedule) {
        inactive = source.replayEnded.load(std::memory_order_relaxed) != 0;
    } else if (source.dsp.impactMode()) {
        const auto write = source.written.load(std::memory_order_acquire);
        const auto read = source.consumed.load(std::memory_order_acquire);
        const float peak = std::bit_cast<float>(source.blockPeak.load(std::memory_order_relaxed));
        inactive = source.hasSequence && write == read && peak < 1e-5f;
    }
    if (!inactive) {
        source.draining = false;
        source.drainSeconds = 0;
        return false;
    }
    source.draining = true;
    source.target.store(0, std::memory_order_relaxed);
    source.drainSeconds += std::isfinite(dt) ? std::clamp(dt, 0.0f, 0.25f) : 0.0f;
    // Current acoustic history is <=1 second; allow excitation and filter decay too.
    return source.drainSeconds >= source.drainAllowance;
}

json PhysicalAudioRuntime::status(const std::string& id) const {
    const auto sequence = sequences_.find(id);
    const json lastSequence = sequence == sequences_.end() ? json() : json(sequence->second);
    const auto found = sources_.find(id);
    if (found == sources_.end()) {
        return {{"state", "not_resident"}, {"last_accepted_sequence", lastSequence}};
    }
    const auto& source = *found->second;
    const auto write = source.written.load(std::memory_order_acquire);
    const auto read = source.consumed.load(std::memory_order_acquire);
    return {{"state", source.draining ? "draining" : "resident"},
        {"last_accepted_sequence", lastSequence},
        {"drain_seconds", source.drainSeconds}, {"queued_impacts", write - read},
        {"drain_allowance_seconds", source.drainAllowance},
        {"held_reaction_gain", source.lastGain},
        {"playback_input", source.schedule ? "scheduled_history" : "live_controls"},
        {"history_frame", source.replayPosition.load(std::memory_order_relaxed)},
        {"history_frames", source.schedule ? source.schedule->frames() : 0},
        {"history_ended", source.replayEnded.load(std::memory_order_relaxed) != 0},
        {"block_peak", std::bit_cast<float>(source.blockPeak.load(std::memory_order_relaxed))}};
}

json PhysicalAudioRuntime::validateImpact(const json& source, const json& params) {
    require(isPhysicalAudioSource(source) && source.at("generator").at("model") == "impact",
        "physical_impact requires an impact generator source");
    const auto sequence = params.value("sequence", json());
    require(sequence.is_number_integer() && sequence.get<double>() >= 0 &&
        sequence.get<double>() <= 9007199254740991.0,
        "Impact sequence must be an integer within 0..2^53-1");
    const auto energy = params.value("energy_j", json());
    require(energy.is_number() && std::isfinite(energy.get<double>()) &&
        energy.get<double>() > 0 && energy.get<double>() <= 1e9,
        "Impact energy_j must be finite within (0, 1e9]");
    return {{"sequence", sequence}, {"energy_j", energy}};
}

json PhysicalAudioRuntime::impact(const std::string& id, const json& event) {
    const auto found = sources_.find(id);
    require(found != sources_.end() && found->second->dsp.impactMode(),
        "Impact source must be prepared before excitation");
    auto& source = *found->second;
    require(!source.schedule, "Live impacts cannot be injected into scheduled replay");
    const auto sequence = event.at("sequence").get<std::uint64_t>();
    if (duplicate(id, sequence)) {
        return {{"id", id}, {"accepted", false}, {"reason", "duplicate_or_out_of_order"}};
    }
    const auto write = source.written.load(std::memory_order_relaxed);
    const auto read = source.consumed.load(std::memory_order_acquire);
    require(write - read < source.impulses.size(), "Impact excitation queue is full (32)");
    const auto amplitude = std::sqrt(std::min(1.0,
        event.at("energy_j").get<double>() / source.referenceEnergy));
    // Reserve/update main-thread dedup state before publishing to the callback.
    sequences_.insert_or_assign(id, sequence);
    source.impulses[write % source.impulses.size()] = static_cast<float>(amplitude);
    source.written.store(write + 1, std::memory_order_release);
    source.hasSequence = true;
    source.draining = false;
    source.drainSeconds = 0;
    return {{"id", id}, {"accepted", true}, {"sequence", sequence},
        {"excitation_amplitude", amplitude}, {"energy_calibrated", false}};
}

void PhysicalAudioRuntime::forget(const std::string& id) {
    sources_.erase(id);
    sequences_.erase(id);
    preparedReplays_.erase(id);
}

bool PhysicalAudioRuntime::duplicate(const std::string& id, std::uint64_t sequence) const {
    const auto found = sequences_.find(id);
    return found != sequences_.end() && sequence <= found->second;
}

void PhysicalAudioRuntime::release(const std::string& id) {
    sources_.erase(id);
}

void PhysicalAudioRuntime::clear() {
    replayJobs_.cancel();
    sources_.clear();
    sequences_.clear();
    preparedReplays_.clear();
}

bool PhysicalAudioRuntime::isReplay(const std::string& id) const {
    const auto found = sources_.find(id);
    return found != sources_.end() && found->second->schedule != nullptr;
}

void PhysicalAudioRuntime::validateReplaySource(const json& source) const {
    if (!isReplay(source.value("id", std::string()))) {
        return;
    }
    require(source.at("pitch") == 1 && source.at("reaction").at("pitch_min") == 1 &&
        source.at("reaction").at("pitch_max") == 1, "Active physical replay requires unit pitch");
}
}
