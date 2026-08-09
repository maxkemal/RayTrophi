#pragma once

#include <cstdint>
#include <string>

namespace RayTrophiSim {

// Storage-independent field metadata shared by MSF, paint, sculpt, terrain,
// fluid/gas and physics bridges. Solvers keep ownership of their specialised
// buffers; this contract prevents consumers from depending on those layouts.
enum class FieldDomain : uint8_t { SurfaceUV, Vertex, VolumeGrid, Terrain2D };
enum class FieldFormat : uint8_t { UNorm8, Float16, Float32, Vec2Float32, Vec3Float32 };
enum class FieldSemantic : uint16_t {
    Unknown = 0,
    Temperature, Moisture, FuelRemaining, Char, Melt, MassLoss,
    Integrity, FractureDamage, Support, Thickness,
    Selection, MaterialMask, Displacement, Stiffness, Pin, ErosionResistance,
    Height, Hardness, Wetness, Sediment, Flow, FuelLoad,
    ReleasedFuel, ReleasedSmoke, MoltenReservoir, AshReservoir
};

struct FieldDirtyRegion {
    bool full = true;
    uint32_t min_x = 0, min_y = 0, min_z = 0;
    uint32_t max_x = 0, max_y = 0, max_z = 0;
};

struct FieldHeader {
    std::string object_key;
    int32_t material_slot = -1;
    FieldSemantic semantic = FieldSemantic::Unknown;
    FieldDomain domain = FieldDomain::SurfaceUV;
    FieldFormat format = FieldFormat::Float32;
    uint64_t topology_generation = 0;
    uint64_t content_generation = 0;
    FieldDirtyRegion dirty;
};

const char* fieldDomainName(FieldDomain domain);
const char* fieldFormatName(FieldFormat format);
const char* fieldSemanticName(FieldSemantic semantic);

} // namespace RayTrophiSim
