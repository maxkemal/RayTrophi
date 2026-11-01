// User's standalone C++ test target; no device or application required.
#include "Fluid/FluidForceColumns.h"
#include <cassert>

using namespace RayTrophiSim;

int main() {
    PackedForceField field;
    field.type = 0;
    field.enabled = 1;
    field.fluid_surface_drag = 1;
    field.affect_mask = static_cast<uint32_t>(SimulationSystemKind::Fluid);
    assert(Fluid::packedForcesNeedFluidColumns({field}));
    field.enabled = 0;
    assert(!Fluid::packedForcesNeedFluidColumns({field}));
    field.enabled = 1;
    field.type = 1;
    assert(!Fluid::packedForcesNeedFluidColumns({field}));
    field.type = 0;
    field.affect_mask = static_cast<uint32_t>(SimulationSystemKind::Gas);
    assert(!Fluid::packedForcesNeedFluidColumns({field}));
    field.affect_mask = static_cast<uint32_t>(SimulationSystemKind::Fluid);
    field.fluid_surface_drag = 0;
    assert(!Fluid::packedForcesNeedFluidColumns({field}));
    assert(!Fluid::packedForcesNeedFluidColumns({}));
    assert(!Fluid::fluidForcesNeedColumns(nullptr, false));
    assert(Fluid::fluidForcesNeedColumns(nullptr, true));

    SimulationForceFieldComputeBuffer buffer;
    assert(!Fluid::fluidForcesNeedColumns(&buffer, false));
    buffer.buffer.id = 1;
    buffer.count = 1;
    buffer.size_bytes = sizeof(PackedForceField);
    assert(Fluid::fluidForcesNeedColumns(&buffer, false)); // Conservative external producer.
    buffer.fluid_surface_columns_required = false;
    assert(!Fluid::fluidForcesNeedColumns(&buffer, false));
    assert(Fluid::fluidForcesNeedColumns(&buffer, true));
}
