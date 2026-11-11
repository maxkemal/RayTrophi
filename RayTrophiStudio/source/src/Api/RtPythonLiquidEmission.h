#pragma once
#include "Api/RtApi.h"
#include <pybind11/pybind11.h>

inline void writeLiquidEmissionPython(pybind11::dict& out,
                                    const rtapi::SimulationFlowSourceInfo& source) {
    out["liquid_emission"] = source.liquid_emission;
    out["fluid_flux_m3_s"] = source.fluid_flux_m3_s;
    out["fluid_flux_target_m3_s"] = source.fluid_flux_target_m3_s;
    out["fluid_overpack_dropped_total"] = source.fluid_overpack_dropped_total;
    out["drained_mass_kg_step"] = source.drained_mass_kg_step;
    out["drained_mass_kg_total"] = source.drained_mass_kg_total;
    out["liquid_emission_status"] = source.liquid_emission_status;
}

inline void applyLiquidEmissionPython(rtapi::SimulationFlowSourceInfo& source,
                                     const pybind11::kwargs& kwargs) {
    if (kwargs.contains("liquid_emission")) {
        source.liquid_emission = pybind11::cast<std::string>(kwargs["liquid_emission"]);
    }
}
