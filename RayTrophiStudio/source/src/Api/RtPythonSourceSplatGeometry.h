#pragma once
#include "Api/RtApi.h"
#include <pybind11/pybind11.h>

void writeSourceSplatGeometry(pybind11::dict& out, const rtapi::SimulationFlowSourceInfo& source);
void applySourceSplatGeometry(rtapi::SimulationFlowSourceInfo& source,
    const pybind11::kwargs& kwargs);
void registerSourceSplatGeometry(pybind11::module_& flow_source);

void registerSubstanceSplatGeometry(pybind11::module_& fluid);
