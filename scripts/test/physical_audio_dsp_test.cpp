// Standalone post-build harness. Codex must not compile this project/test.
// Link PhysicalAudioDsp.cpp, PhysicalAudioNaturalDsp.cpp and PhysicalAudioCharacterDsp.cpp.
#include "Audio/PhysicalAudioDsp.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using json = nlohmann::json;
void require(bool valid, const char* message) {
    if (!valid) {
        throw std::runtime_error(message);
    }
}

json recipe(const std::string& model, int version) {
    json result = {{"version", version}, {"model", model}, {"seed", 17}, {"brightness", 0.6}};
    if (version == 2) {
        result.update({{"texture", 0.6}, {"variation", 0.7}, {"body", 0.4}});
    }
    if (model == "impact" || model == "contact_activity" || model == "deformation" ||
        model == "fracture_activity") {
        result.update({{"resonator", "wood"}, {"size_m", 0.1},
            {"decay_seconds", 0.1}});
    }
    if (model == "impact") {
        result["reference_energy_j"] = 1;
    }
    return result;
}

std::vector<float> render(const json& config, std::size_t block) {
    audio::PhysicalAudioDsp dsp(config);
    const bool impact = config.at("model") == "impact";
    const std::size_t boundaries[] = {0, 997, 4093, 10001, 22050, 48000};
    const float activity[] = {0, 0.1f, 1, 0.5f, 0};
    std::vector<float> samples(boundaries[5]);
    for (std::size_t segment = 0; segment < 5; ++segment) {
        auto at = boundaries[segment];
        while (at < boundaries[segment + 1]) {
            const auto count = std::min(block, boundaries[segment + 1] - at);
            const float excitation = impact && at == boundaries[segment] &&
                segment > 0 && segment < 4 ? 0.3f : 0;
            // Empty callbacks must not consume an excitation or advance random/DSP state.
            dsp.render(nullptr, 0, activity[segment], 0.9f);
            dsp.render(samples.data() + at, count, activity[segment], excitation);
            at += count;
        }
    }
    return samples;
}

double energy(const std::vector<float>& samples) {
    double sum = 0;
    for (float sample : samples) {
        require(std::isfinite(sample) && std::abs(sample) < 4, "Nonfinite/unbounded PCM");
        sum += double(sample) * sample;
    }
    return sum;
}
}

int main() {
    try {
        for (const auto* model : {"liquid_flow", "gas_flow", "friction", "rolling", "impact",
                                 "contact_activity", "deformation", "fracture_activity"}) {
            std::vector<float> legacy;
            for (int version : {1, 2}) {
                auto config = recipe(model, version);
                const auto reference = render(config, 1);
                require(energy(reference) > 1e-10, "Missing generated sound");
                for (std::size_t block : {7u, 64u, 257u, 1024u}) {
                    require(reference == render(config, block), "Callback partition changed PCM");
                }
                audio::PhysicalAudioDsp silent(config);
                std::vector<float> silence(4800);
                silent.render(silence.data(), silence.size(), 0);
                require(energy(silence) == 0, "Zero activity created sound");
                if (version == 1) {
                    legacy = reference;
                } else {
                    require(reference != legacy, "V2 did not select new DSP");
                    config["seed"] = 18;
                    require(reference != render(config, 64), "Seed had no effect");
                    config = recipe(model, version);
                    config["texture"] = 0.05;
                    config["body"] = 0.95;
                    require(reference != render(config, 64), "Texture/body had no effect");
                }
            }
            std::cout << "PASS " << model << ": finite, silent, seeded, partition invariant\n";
        }
        for (const auto* model : {"liquid_flow", "gas_flow", "friction", "impact", "fracture_activity"}) {
            auto config = recipe(model, 2);
            const auto original = render(config, 64);
            config["character_revision"] = 2;
            const auto tuned = render(config, 1);
            require(energy(tuned) > 1e-10 && tuned != original, "Tuned character is missing");
            for (std::size_t block : {7u, 64u, 257u, 1024u}) {
                require(tuned == render(config, block), "Tuned character depends on block size");
            }
            audio::PhysicalAudioDsp silent(config);
            std::vector<float> silence(4800);
            silent.render(silence.data(), silence.size(), 0);
            require(energy(silence) == 0, "Tuned character generated sound from zero activity");
        }
        for (const auto* material : {"metal", "glass"}) {
            auto config = recipe("impact", 2);
            config["resonator"] = material;
            const auto original = render(config, 64);
            config["character_revision"] = 2;
            require(original == render(config, 64), "Accepted metal/glass impact changed");
        }
        for (const auto* material : {"wood", "stone"}) {
            double tails[2]{};
            for (int revision : {1, 2}) {
                auto config = recipe("impact", 2);
                config.update({{"character_revision", revision}, {"resonator", material},
                    {"decay_seconds", 0.6}});
                audio::PhysicalAudioDsp dsp(config);
                std::vector<float> samples(24000);
                dsp.render(samples.data(), samples.size(), 0, 0.3f);
                tails[revision - 1] = energy(std::vector<float>(samples.begin() + 5760, samples.end()));
            }
            require(tails[1] < tails[0] * 0.2, "Wood/stone ringing tail was not damped");
        }
        std::vector<std::vector<float>> surfaces;
        for (const auto* surface : {"organic", "mineral", "metal", "glass"}) {
            auto config = recipe("friction", 2);
            config.update({{"character_revision", 2}, {"surface", surface}});
            auto samples = render(config, 64);
            for (const auto& previous : surfaces) {
                require(previous != samples, "Friction surface has no effect");
            }
            surfaces.push_back(samples);
        }
        std::cout << "PASS tuned characters, preserved metal/glass, wood/stone damping, friction surfaces\n";
        // Stronger explicit excitation must carry more unnormalized dry energy.
        const auto config = recipe("impact", 2);
        double energies[2]{};
        for (int i = 0; i < 2; ++i) {
            audio::PhysicalAudioDsp dsp(config);
            std::vector<float> samples(12000);
            dsp.render(samples.data(), samples.size(), 0, i == 0 ? 0.1f : 0.2f);
            energies[i] = energy(samples);
        }
        require(energies[1] > energies[0], "Impact energy ordering was lost");
        std::cout << "PASS impact energy ordering\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
