// Liquid FOG view on its own volume slot.
//
// A liquid domain draws its surface on the domain's primary volume and its fog
// here, so both can be on screen at once (FluidViewResolver decides which
// substances go where). Before this, one volume per domain meant a surface
// override silently swallowed the fog.
//
// The gas/fog and SurfaceSDF instances use separate TLAS mask bits, and a gas
// ray hands off to the nearest surface crossing, so a coincident fog + surface
// pair is the layering case VULKAN_GAS_FLUID_LAYERING.md already validates.

#include "scene_data.h"
#include "Fluid/MatterPhaseGrid.h"

#include "globals.h"
#include "Fluid/FluidFogDensity.h"
#include "Fluid/FluidFoam.h"
#include "Fluid/FluidViewResolver.h"
#include "VolumeShader.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace RayTrophiSim {
namespace Fluid {

// Liquid splat density is small (a packed cell is ~1, a lone particle leaves
// 1/ppc spread over 8 cells) where gas presets expect 1..10, hence the
// multiplier; the tinted absorption makes accumulated water read blue.
std::shared_ptr<VolumeShader> makeLiquidFogShader() {
    auto shader = std::make_shared<VolumeShader>();
    shader->name = "Liquid Fog";
    shader->density.multiplier = 50.0f;
    shader->density.cutoff_threshold = 0.01f;
    shader->scattering.color = Vec3(0.55f, 0.74f, 0.92f);
    shader->scattering.coefficient = 1.0f;
    shader->scattering.anisotropy = 0.0f;
    shader->absorption.color = Vec3(0.15f, 0.42f, 0.78f);
    shader->absorption.coefficient = 2.0f;
    shader->emission.mode = VolumeEmissionMode::None;
    return shader;
}

} // namespace Fluid
} // namespace RayTrophiSim

void SceneData::syncDomainFogVolume(ParticleSystemObject& system, std::size_t d,
                                    const RayTrophiSim::SimulationGridDomainState& state,
                                    RayTrophiSim::SimulationGridDomainDesc& desc,
                                    const RayTrophiSim::Fluid::FluidViewPlan& plan,
                                    VDBVolumeManager& mgr, int frame, bool force_sync,
                                    bool render_enabled) {
    using RayTrophiSim::Fluid::FluidView;
    if (d >= system.domain_fog_vdb_ids.size() || d >= system.domain_fog_volumes.size()) return;
    const bool matter_gas =
        desc.type == RayTrophiSim::SimulationDomainType::Matter;

    // No fog view (or the domain is not rendered at all): the resource goes.
    // plan.fog is stable across empty frames by construction, so this is a
    // real representation change, never a per-frame flicker.
    if ((!plan.fog && !matter_gas) || !render_enabled) {
        if (system.domain_fog_vdb_ids[d] >= 0 || system.domain_fog_volumes[d]) {
            removeFogDomainVolume(system, d);
        }
        return;
    }

    const auto& grid = matter_gas
        ? RayTrophiSim::Fluid::gasGrid(state)
        : RayTrophiSim::Fluid::liquidGrid(state);
    const std::size_t cells = static_cast<std::size_t>(grid.getCellCount());
    auto hide = [&]() {
        // Keep the slot and its id; an invisible volume leaves the packet
        // without a TLAS rebuild, and showing it again is an SSBO update.
        if (system.domain_fog_volumes[d] && system.domain_fog_volumes[d]->visible) {
            system.domain_fog_volumes[d]->visible = false;
            g_gas_volumes_dirty = true;
        }
    };
    if (!system.visible || !state.valid || grid.nx <= 0 || cells == 0) {
        hide();
        return;
    }

    // ── Density: whole-domain splat when every live parcel is fog, otherwise
    //    only the fog parcels, with the same per-particle weight. ───────────
    // Whitewater types routed to fog add their deposit (massless, drawn only:
    // one particle = volume_density parcels' worth). The slot is kept for a
    // fog ROUTE (e.g. mist) even with nothing live; hide it without walking.
    const auto ww_fog_types = plan.whitewaterTypesIn(FluidView::Fog);
    const bool ww_fog_routed = desc.fluid_foam_params.enabled && !state.foam.empty() &&
        (ww_fog_types[0] || ww_fog_types[1] || ww_fog_types[2]);
    const bool parcels_live = plan.anyLiveIn(FluidView::Fog);
    if (!matter_gas && !parcels_live && !ww_fog_routed) {
        hide();
        return;
    }
    const RayTrophiSim::Fluid::FluidViewSelection fog_selection{&plan, FluidView::Fog};
    const bool whole_domain = parcels_live && plan.allLiveIn(FluidView::Fog);
    std::vector<float> subset;
    const float* raw = nullptr;
    if (matter_gas) {
        if (state.active_density_cells > 0 && grid.density.size() == cells) {
            raw = grid.density.data();
        }
        if (parcels_live) {
            std::vector<float> fluid_fog;
            if (RayTrophiSim::Fluid::splatFogDensityForSelection(
                    state.particles, grid.nx, grid.ny, grid.nz, grid.origin,
                    grid.voxel_size, desc.fluid_params.particles_per_cell,
                    fog_selection, fluid_fog)) {
                if (raw) subset.assign(raw, raw + cells);
                else subset.assign(cells, 0.0f);
                for (std::size_t c = 0; c < cells; ++c) {
                    subset[c] += fluid_fog[c];
                }
                raw = subset.data();
            }
        }
    } else if (!parcels_live) {
        // whitewater only; filled below
    } else if (whole_domain) {
        if (state.active_density_cells > 0 && grid.density.size() == cells) {
            raw = grid.density.data();
        }
    } else if (RayTrophiSim::Fluid::splatFogDensityForSelection(
                   state.particles, grid.nx, grid.ny, grid.nz, grid.origin,
                   grid.voxel_size, desc.fluid_params.particles_per_cell,
                   fog_selection, subset)) {
        raw = subset.data();
    }
    if (ww_fog_routed) {
        const bool routed[3] = { ww_fog_types[0], ww_fog_types[1], ww_fog_types[2] };
        float weights[3];
        RayTrophiSim::Fluid::whitewaterTypeWeights(desc.fluid_foam_params, routed, weights);
        const float per_particle = std::max(0.0f, desc.fluid_foam_params.volume_density) /
            static_cast<float>(std::max(1, desc.fluid_params.particles_per_cell));
        std::vector<float> ww;
        if (RayTrophiSim::Fluid::splatFoamDensity(
                state.foam, grid.nx, grid.ny, grid.nz, grid.voxel_size, grid.origin,
                ww, per_particle, weights) > 0) {
            if (!raw || raw != subset.data()) {
                // grid.density (whole domain) or nothing: own a copy to add into.
                if (raw) subset.assign(raw, raw + cells);
                else subset.assign(cells, 0.0f);
            }
            for (std::size_t c = 0; c < cells && c < ww.size(); ++c) subset[c] += ww[c];
            raw = subset.data();
        }
    }
    if (!raw) {
        // Nothing to draw this frame (no fog parcels yet, or rewound empty).
        hide();
        return;
    }

    if (matter_gas) {
        if (!desc.fluid_fog_shader) {
            desc.fluid_fog_shader = VolumeShader::createSmokePreset();
        }
    } else if (!desc.fluid_fog_shader) {
        desc.fluid_fog_shader = RayTrophiSim::Fluid::makeLiquidFogShader();
    }
    const auto& shader = desc.fluid_fog_shader;

    const std::string volume_name =
        system.name + " Domain " + std::to_string(d) +
        (matter_gas ? " [Matter Gas]" : " [Fluid Fog]");
    int stride = 1;
    const long long grid_cells = static_cast<long long>(cells);
    if (grid_cells >= 160LL * 160 * 160) stride = 3;
    else if (grid_cells >= 104LL * 104 * 104) stride = 2;

    const int prev_id = system.domain_fog_vdb_ids[d];
    const bool do_update = force_sync || prev_id < 0 || (frame % stride) == 0;
    if (do_update) {
        // A Gaussian-spread copy: the raw trilinear splat reaches 8 cells per
        // particle and shows spray as isolated dots.
        std::vector<float> spread;
        const float* density = raw;
        if (!matter_gas && desc.fluid_fog_spread_voxels > 0.0f) {
            RayTrophiSim::Fluid::spreadFogDensity(raw, grid.nx, grid.ny, grid.nz,
                                                  desc.fluid_fog_spread_voxels, spread);
            density = spread.data();
        }
        // Blackbody / channel emission reads the parcels' Kelvin on the same
        // grid and spread, so a hot parcel glows where it is drawn.
        std::vector<float> kelvin;
        const float* temperature = nullptr;
        const bool wants_temperature =
            shader->emission.mode == VolumeEmissionMode::Blackbody ||
            shader->emission.mode == VolumeEmissionMode::ChannelDriven;
        if (matter_gas && wants_temperature && grid.temperature.size() == cells) {
            kelvin.resize(cells);
            constexpr float kHeatToKelvin = 3000.0f;
            for (std::size_t c = 0; c < cells; ++c) {
                kelvin[c] = grid.temperature[c] * kHeatToKelvin;
            }
            temperature = kelvin.data();
        } else if (wants_temperature &&
            RayTrophiSim::Fluid::splatFogTemperatureKelvin(
                state.particles, grid.nx, grid.ny, grid.nz, grid.origin,
                grid.voxel_size, desc.fluid_fog_spread_voxels, kelvin,
                whole_domain ? nullptr : &fog_selection)) {
            temperature = kelvin.data();
        }
        const int new_id = mgr.registerOrUpdateLiveVolume(
            prev_id, volume_name, grid.nx, grid.ny, grid.nz, grid.voxel_size,
            density, temperature, nullptr);
        // -1 means the conversion threw; keep the last good binding rather
        // than orphaning it (same rule as the surface slot).
        if (new_id >= 0) system.domain_fog_vdb_ids[d] = new_id;
        simulation_render_updated = true;
        g_gas_volumes_dirty = true;
    }
    const int id = system.domain_fog_vdb_ids[d];
    if (id < 0) return;
    mgr.clearLiveDenseGpuFields(id);

    const Vec3 world_min = grid.origin;
    const Vec3 world_max = grid.origin +
        Vec3(static_cast<float>(grid.nx) * grid.voxel_size,
             static_cast<float>(grid.ny) * grid.voxel_size,
             static_cast<float>(grid.nz) * grid.voxel_size);

    bool created = false;
    if (!system.domain_fog_volumes[d]) {
        auto vol = std::make_shared<VDBVolume>();
        vol->transient = true;
        vol->name = volume_name;
        system.domain_fog_volumes[d] = vol;
        addVDBVolume(vol);
        world.objects.push_back(vol);
        created = true;
    }
    auto& vol = system.domain_fog_volumes[d];
    if (!vol->visible) {
        vol->visible = true;
        g_gas_volumes_dirty = true;
    }
    vol->name = volume_name;
    vol->cpu_render_skip = false;
    // The TLAS bakes this flag into the instance mask (fog 0x02, surface 0x08);
    // a flip without a rebuild leaves the mask behind. A fog slot is never an
    // isosurface, so this only fires if something else wrote the flag.
    if (vol->render_as_isosurface && !created) {
        g_geometry_dirty = true;
        g_vulkan_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        g_gas_volumes_dirty = true;
    }
    vol->render_as_isosurface = false;
    vol->setShader(shader);  // live shader edits reach the volume
    vol->bindLiveVolume(id, grid.voxel_size, world_min, world_max);

    if (created) {
        // New hittable: GPU TLAS + CPU BVH must see it (same as the surface slot).
        g_geometry_dirty = true;
        g_vulkan_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        g_gas_volumes_dirty = true;
        g_bvh_rebuild_pending = true;
    }
}
