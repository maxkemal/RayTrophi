#include "Fluid/MatterFogStreams.h"
#include "MaterialStateField.h"
#include "VolumeShader.h"
#include "ParticleSimulation.h"
#include <stdexcept>

namespace RayTrophiSim::Fluid {

std::shared_ptr<VolumeShader>& matterGasFogShader(SimulationGridDomainDesc& domain) {
    FogFieldKey key;
    key.kind = FogFieldKind::GasAggregate;
    return ensureFogAppearance(domain.matter_fog_streams, key,
        domain.fluid_fog_shader).shader;
}

std::string fogFieldId(const FogFieldKey& key) {
    if (key.kind == FogFieldKind::GasAggregate) {
        return "gas";
    }
    return "carrier:" + std::to_string(key.substance_tag) + ":" +
        std::to_string(static_cast<unsigned>(key.phase));
}

std::string fogFieldName(const FogFieldKey& key) {
    if (key.kind == FogFieldKind::GasAggregate) {
        return "Vapor / Gas";
    }
    const auto* substance = tryFindSubstanceByTag(key.substance_tag);
    const std::string name = substance ? substance->name : "Matter";
    return name + (key.phase == MatterOutputPhase::Solid ? " [Solid]" :
        key.phase == MatterOutputPhase::Liquid ? " [Liquid]" : " [Unresolved]");
}

MatterFogAppearance& ensureFogAppearance(MatterFogAppearances& appearances,
    const FogFieldKey& key, const std::shared_ptr<VolumeShader>& legacy) {
    auto& appearance = appearances[key];
    if (!appearance.shader) {
        appearance.shader = legacy ? std::make_shared<VolumeShader>(*legacy) :
            VolumeShader::createSmokePreset();
        appearance.shader->name = fogFieldName(key);
        if (key.kind == FogFieldKind::Carrier) {
            appearance.shader->emission.mode = VolumeEmissionMode::None;
        } else {
            appearance.shader->scattering.random_walk = false;
        }
    }
    return appearance;
}

nlohmann::json writeFogAppearances(const MatterFogAppearances& appearances) {
    auto rows = nlohmann::json::array();
    for (const auto& [key, appearance] : appearances) {
        nlohmann::json row = {{"kind", static_cast<unsigned>(key.kind)},
            {"substance_tag", key.substance_tag},
            {"phase", static_cast<unsigned>(key.phase)}, {"visible", appearance.visible}};
        if (appearance.shader) {
            row["material"] = appearance.shader->toJson();
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

MatterFogAppearances readFogAppearances(const nlohmann::json& domain) {
    MatterFogAppearances result;
    if (!domain.contains("matter_fog_streams")) {
        return result;
    }
    const auto& rows = domain.at("matter_fog_streams");
    if (!rows.is_array() || rows.size() > 32) {
        throw std::invalid_argument("matter_fog_streams must be an array of at most 32 entries");
    }
    for (const auto& row : rows) {
        const auto kind = row.at("kind").get<unsigned>();
        const auto phase = row.at("phase").get<unsigned>();
        if (kind > static_cast<unsigned>(FogFieldKind::GasAggregate) ||
            phase > static_cast<unsigned>(MatterOutputPhase::Liquid)) {
            throw std::invalid_argument("invalid Matter Fog stream key");
        }
        FogFieldKey key{static_cast<FogFieldKind>(kind),
            row.at("substance_tag").get<uint32_t>(), static_cast<MatterOutputPhase>(phase)};
        if (key.kind == FogFieldKind::GasAggregate &&
            (key.substance_tag != 0 || key.phase != MatterOutputPhase::Unknown)) {
            throw std::invalid_argument("aggregate gas must have an anonymous gas key");
        }
        MatterFogAppearance appearance;
        appearance.visible = row.value("visible", true);
        if (row.contains("material")) {
            appearance.shader = std::make_shared<VolumeShader>();
            appearance.shader->fromJson(row.at("material"));
        }
        if (!result.emplace(key, std::move(appearance)).second) {
            throw std::invalid_argument("duplicate Matter Fog stream key");
        }
    }
    return result;
}
} // namespace RayTrophiSim::Fluid
