#include "SurfaceField.h"

namespace RayTrophiSim {

const char* fieldDomainName(FieldDomain v) {
    switch (v) {
        case FieldDomain::SurfaceUV: return "surface_uv";
        case FieldDomain::Vertex: return "vertex";
        case FieldDomain::VolumeGrid: return "volume_grid";
        case FieldDomain::Terrain2D: return "terrain_2d";
    }
    return "unknown";
}

const char* fieldFormatName(FieldFormat v) {
    switch (v) {
        case FieldFormat::UNorm8: return "unorm8";
        case FieldFormat::Float16: return "float16";
        case FieldFormat::Float32: return "float32";
        case FieldFormat::Vec2Float32: return "vec2_float32";
        case FieldFormat::Vec3Float32: return "vec3_float32";
    }
    return "unknown";
}

const char* fieldSemanticName(FieldSemantic v) {
    switch (v) {
        case FieldSemantic::Temperature: return "temperature";
        case FieldSemantic::Moisture: return "moisture";
        case FieldSemantic::FuelRemaining: return "fuel_remaining";
        case FieldSemantic::Char: return "char";
        case FieldSemantic::Melt: return "melt";
        case FieldSemantic::MassLoss: return "mass_loss";
        case FieldSemantic::Integrity: return "integrity";
        case FieldSemantic::FractureDamage: return "fracture_damage";
        case FieldSemantic::Support: return "support";
        case FieldSemantic::Thickness: return "thickness";
        case FieldSemantic::Selection: return "selection";
        case FieldSemantic::MaterialMask: return "material_mask";
        case FieldSemantic::Displacement: return "displacement";
        case FieldSemantic::Stiffness: return "stiffness";
        case FieldSemantic::Pin: return "pin";
        case FieldSemantic::ErosionResistance: return "erosion_resistance";
        case FieldSemantic::Height: return "height";
        case FieldSemantic::Hardness: return "hardness";
        case FieldSemantic::Wetness: return "wetness";
        case FieldSemantic::Sediment: return "sediment";
        case FieldSemantic::Flow: return "flow";
        case FieldSemantic::FuelLoad: return "fuel_load";
        case FieldSemantic::ReleasedFuel: return "released_fuel";
        case FieldSemantic::ReleasedSmoke: return "released_smoke";
        case FieldSemantic::MoltenReservoir: return "molten_reservoir";
        case FieldSemantic::AshReservoir: return "ash_reservoir";
        case FieldSemantic::Unknown: break;
    }
    return "unknown";
}

} // namespace RayTrophiSim
