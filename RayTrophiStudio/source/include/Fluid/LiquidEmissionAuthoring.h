#pragma once

#include "ParticleSimulation.h"
#include <string>
#include <cmath>
#include <stdexcept>

namespace RayTrophiSim::Fluid {

inline const char* liquidEmissionName(SimulationFlowSourceDesc::LiquidEmission mode) {
    using Mode = SimulationFlowSourceDesc::LiquidEmission;
    switch (mode) {
    case Mode::Inflow:
        return "inflow";
    case Mode::Outflow:
        return "outflow";
    default:
        return "rate";
    }
}

inline bool parseLiquidEmission(const std::string& name,
                               SimulationFlowSourceDesc::LiquidEmission& mode) {
    using Mode = SimulationFlowSourceDesc::LiquidEmission;
    if (name == "rate") {
        mode = Mode::Rate;
    } else if (name == "inflow") {
        mode = Mode::Inflow;
    } else if (name == "outflow") {
        mode = Mode::Outflow;
    } else {
        return false;
    }
    return true;
}

inline bool setLiquidEmission(SimulationFlowSourceDesc& source,
                              const std::string& name, std::string& error) {
    auto mode = source.liquid_emission;
    if (!parseLiquidEmission(name, mode)) {
        error = "liquid_emission must be rate, inflow or outflow";
        return false;
    }
    if (mode != SimulationFlowSourceDesc::LiquidEmission::Rate) {
        if (source.phase != SimulationFlowSourceDesc::Phase::Liquid) {
            error = "inflow/outflow require a liquid source";
            return false;
        }
        if (source.source_mode == SimulationFlowSourceMode::MeshSurface) {
            error = "inflow/outflow require point or object_bounds geometry";
            return false;
        }
    }
    source.liquid_emission = mode;
    return true;
}

template<class Info>
inline bool configureLiquidEmission(const Info& info, SimulationFlowSourceDesc& source,
                                    std::string& error) {
    if (info.liquid_emission != "rate" &&
        (!std::isfinite(info.radius) || info.radius <= 0.0f ||
         !std::isfinite(info.velocity.x) || !std::isfinite(info.velocity.y) ||
         !std::isfinite(info.velocity.z))) {
        error = "inflow/outflow require positive finite radius and finite velocity";
        return false;
    }
    return setLiquidEmission(source, info.liquid_emission, error);
}

template<class Info>
inline void copyLiquidEmissionInfo(const SimulationFlowSourceDesc& source, Info& out) {
    out.liquid_emission = liquidEmissionName(source.liquid_emission);
    out.fluid_flux_m3_s = source.fluid_flux_m3_s;
    out.fluid_flux_target_m3_s = source.fluid_flux_target_m3_s;
    out.fluid_overpack_dropped_total = source.fluid_overpack_dropped_total;
    out.drained_mass_kg_step = source.drained_mass_kg_step;
    out.drained_mass_kg_total = source.drained_mass_kg_total;
    out.liquid_emission_status = source.liquid_emission_status;
}

// Project files store authored mode only. Runtime counters restart with simulation.
template<class Json>
inline void writeLiquidEmission(Json& out, const SimulationFlowSourceDesc& source) {
    out["liquid_emission"] = liquidEmissionName(source.liquid_emission);
    out["phase"] = static_cast<int>(source.phase);
}

template<class Json>
inline void readLiquidEmission(const Json& data, SimulationFlowSourceDesc& source) {
    if (data.contains("phase")) {
        if (!data.at("phase").is_number_integer()) {
            throw std::runtime_error("flow source phase must be an integer");
        }
        const int phase = data.at("phase").template get<int>();
        if (phase < 0 || phase > 1) {
            throw std::runtime_error("unknown flow source phase in project");
        }
        source.phase = static_cast<SimulationFlowSourceDesc::Phase>(phase);
    }
    if (!data.contains("liquid_emission")) {
        source.liquid_emission = SimulationFlowSourceDesc::LiquidEmission::Rate;
        return;
    }
    if (!data.at("liquid_emission").is_string()) {
        throw std::runtime_error("liquid_emission must be a string");
    }
    auto mode = source.liquid_emission;
    if (!parseLiquidEmission(data.at("liquid_emission").template get<std::string>(), mode)) {
        throw std::runtime_error("unknown liquid_emission in project");
    }
    source.liquid_emission = mode;
}

} // namespace RayTrophiSim::Fluid
