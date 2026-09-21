// Standalone CPU regression test; no Vulkan device or application required.
#include "../../RayTrophiStudio/source/include/Viewport/RasterMaterialVisibility.h"
#include "../../RayTrophiStudio/source/include/MaterialCoverage.h"
#include <cassert>
#include <limits>

struct Material {
    float transmission = 0.0f;
    float opacity = 1.0f;
    float tile_break_strength = 0.0f;
    uint32_t transmission_tex = 0, opacity_tex = 0, flags = 0;
};

int main() {
    MaterialCoverage authored;
    assert(!authored.alphaCutout);
    assert(authored.opacity(0.3f) == 0.3f);
    assert(!MaterialCoverage::validCutoutValue(0.5f));
    authored.setCutout(1);
    MaterialCoverage copied = authored;
    assert(copied.opacity(0.49f) == 0 && copied.opacity(0.5f) == 1);
    authored.setCutout(0);
    assert(authored.opacity(0.3f) == 0.3f);
    Backend::RasterMaterialUsage usage;
    Backend::RasterMaterialPrograms programs;
    std::vector<Material> materials(2);
    std::vector<uint32_t> ids{0, 0, 0};
    auto replay = [&] {
        return Backend::rasterMeshMayTransmit(usage, ids, 3, materials, 2, programs, false);
    };
    assert(!replay());
    // Value/texture edits must take effect WITHOUT invalidating membership.
    materials[0].transmission = 1;
    assert(replay());
    materials[0].transmission = 0;
    materials[0].opacity_tex = 4;
    assert(replay());
    materials[0].opacity_tex = 0;
    materials[0].transmission_tex = 4;
    assert(replay());
    materials[0].transmission_tex = 0;
    materials[0].opacity = 0.5f;
    assert(replay());
    materials[0].opacity = 1;
    materials[0].flags = 1u << 19u;
    assert(replay());
    materials[0].flags = 0;
    assert(!replay());

    // Programs can introduce opacity or transmission while scalar fields stay opaque.
    programs.update({2, 3, UINT32_MAX, 0});
    assert(replay());
    programs.update({2, UINT32_MAX, 3, 0});
    assert(!replay());
    programs.update({99}); // malformed metadata must retain draws
    assert(replay());
    programs.update({});
    assert(!replay());

    // Reassignment, including same-size edits, updates the unique-ID cache.
    materials[1].transmission = 1;
    ids[1] = 1;
    usage.invalidate();
    assert(replay());
    ids[1] = 0;
    usage.invalidate();
    assert(!replay());
    // Shader clamps out-of-range IDs to the last bound material.
    ids.assign(3, 123);
    usage.invalidate();
    assert(replay());
    // Impostor flag is part of the stream; it must not alias ordinary IDs.
    ids.assign(3, 0x80000001u);
    usage.invalidate();
    assert(!replay());
    ids[0] = 1;
    usage.invalidate();
    assert(replay());

    assert(Backend::rasterMeshMayTransmit(usage, ids, 3, materials, 2, programs, true));
    assert(Backend::rasterMeshMayTransmit(usage, ids, 3, materials, 0, programs, false));
    assert(Backend::rasterMeshMayTransmit(usage, ids, 3, materials, 9, programs, false));
    assert(Backend::rasterMeshMayTransmit(usage, {}, 3, materials, 2, programs, false));
    assert(!Backend::rasterMeshMayTransmit(usage, {}, 0, materials, 2, programs, false));

    using RasterMaterialPolicy::rasterMaterialMayTransmit;
    assert(!rasterMaterialMayTransmit(0.001f, 0.99f, false, false, false, false, false));
    assert(rasterMaterialMayTransmit(0.0011f, 1, false, false, false, false, false));
    assert(rasterMaterialMayTransmit(0, 0.989f, false, false, false, false, false));
    assert(rasterMaterialMayTransmit(std::numeric_limits<float>::quiet_NaN(),
                                    1, false, false, false, false, false));
    assert(!rasterMaterialMayTransmit(0, 0.2f, false, true, false, false, true));
    assert(rasterMaterialMayTransmit(1, 1, false, true, false, false, true));

    ids.assign(3, 0);
    usage.invalidate();
    materials[0] = Material{};
    auto covered = [&] {
        return Backend::rasterMeshHasExactCoverage(usage, ids, 3, materials, 2, programs, false);
    };
    assert(covered());
    materials[0].opacity_tex = 5;
    assert(!covered());
    materials[0].flags = MATERIAL_FLAG_ALPHA_CUTOUT;
    assert(covered());
    assert(!replay());
    materials[0].tile_break_strength = 0.1f;
    assert(!covered());
    materials[0].tile_break_strength = 0;
    programs.update({1, 2, 0});
    assert(!covered());
    programs.update({});
    materials[0].transmission = 1;
    assert(!covered());
    assert(replay());
    materials[0].transmission = 0;
    materials[0].flags |= 1u << 17u;
    assert(!covered());
    assert(SurfaceCoverage::materialCoverageOpacity(0.49f, true) == 0);
    assert(SurfaceCoverage::materialCoverageOpacity(0.5f, true) == 1);
    assert(SurfaceCoverage::materialCoverageOpacity(0.49f, false) == 0.49f);
}
