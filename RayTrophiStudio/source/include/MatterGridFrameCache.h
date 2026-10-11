#pragma once

#include "FluidGrid.h"
#include "LosslessFrameArray.h"

namespace RayTrophiSim::SimFrameCache {

// Layout/scalar metadata remains in State; all vector storage moves here.
struct MatterGridFrameCache {
    LosslessArray<typename decltype(FluidSim::FluidGrid::active_tiles)::value_type> active_tiles;
    LosslessArray<typename decltype(FluidSim::FluidGrid::tile_active_mask)::value_type>
        tile_active_mask;
    LosslessArray<typename decltype(FluidSim::FluidGrid::vel_x)::value_type> vel_x;
    LosslessArray<typename decltype(FluidSim::FluidGrid::vel_y)::value_type> vel_y;
    LosslessArray<typename decltype(FluidSim::FluidGrid::vel_z)::value_type> vel_z;
    LosslessArray<typename decltype(FluidSim::FluidGrid::density)::value_type> density;
    LosslessArray<typename decltype(FluidSim::FluidGrid::temperature)::value_type> temperature;
    LosslessArray<typename decltype(FluidSim::FluidGrid::fuel)::value_type> fuel;
    LosslessArray<typename decltype(FluidSim::FluidGrid::interaction)::value_type> interaction;
    LosslessArray<typename decltype(FluidSim::FluidGrid::pressure)::value_type> pressure;
    LosslessArray<typename decltype(FluidSim::FluidGrid::divergence)::value_type> divergence;
    LosslessArray<typename decltype(FluidSim::FluidGrid::solid)::value_type> solid;
    LosslessArray<typename decltype(FluidSim::FluidGrid::solid_vel)::value_type> solid_vel;
    LosslessArray<typename decltype(FluidSim::FluidGrid::solid_cells)::value_type> solid_cells;
    LosslessArray<typename decltype(FluidSim::FluidGrid::kinematic_touched_cells)::value_type>
        kinematic_touched_cells;
    LosslessArray<typename decltype(FluidSim::FluidGrid::kinematic_previous_solid)::value_type>
        kinematic_previous_solid;
    LosslessArray<typename decltype(FluidSim::FluidGrid::kinematic_previous_velocity)::value_type>
        kinematic_previous_velocity;
    LosslessArray<typename decltype(FluidSim::FluidGrid::substance_solid_cells)::value_type>
        substance_solid_cells;
    LosslessArray<typename decltype(FluidSim::FluidGrid::substance_solid_prev_cells)::value_type>
        substance_solid_prev_cells;
    LosslessArray<typename decltype(FluidSim::FluidGrid::substance_solid_prev_vel)::value_type>
        substance_solid_prev_vel;
    LosslessArray<typename decltype(FluidSim::FluidGrid::solid_gas_density)::value_type>
        solid_gas_density;
    LosslessArray<typename decltype(FluidSim::FluidGrid::solid_gas_temperature)::value_type>
        solid_gas_temperature;
    LosslessArray<typename decltype(FluidSim::FluidGrid::solid_gas_fuel)::value_type>
        solid_gas_fuel;
    LosslessArray<typename decltype(FluidSim::FluidGrid::solid_gas_flame)::value_type>
        solid_gas_flame;
    LosslessArray<typename decltype(FluidSim::FluidGrid::solid_gas_band)::value_type>
        solid_gas_band;
    LosslessArray<typename decltype(FluidSim::FluidGrid::u_weight)::value_type> u_weight;
    LosslessArray<typename decltype(FluidSim::FluidGrid::v_weight)::value_type> v_weight;
    LosslessArray<typename decltype(FluidSim::FluidGrid::w_weight)::value_type> w_weight;
    LosslessArray<typename decltype(FluidSim::FluidGrid::overlay_weight_restore_u)::value_type>
        overlay_weight_restore_u;
    LosslessArray<typename decltype(FluidSim::FluidGrid::overlay_weight_restore_v)::value_type>
        overlay_weight_restore_v;
    LosslessArray<typename decltype(FluidSim::FluidGrid::overlay_weight_restore_w)::value_type>
        overlay_weight_restore_w;
    LosslessArray<typename decltype(FluidSim::FluidGrid::fluid_phi)::value_type> fluid_phi;
    LosslessArray<typename decltype(FluidSim::FluidGrid::surface_dust_supply)::value_type>
        surface_dust_supply;

    void captureAndRelease(FluidSim::FluidGrid& grid) {
        active_tiles.captureAndRelease(grid.active_tiles);
        tile_active_mask.captureAndRelease(grid.tile_active_mask);
        vel_x.captureAndRelease(grid.vel_x);
        vel_y.captureAndRelease(grid.vel_y);
        vel_z.captureAndRelease(grid.vel_z);
        density.captureAndRelease(grid.density);
        temperature.captureAndRelease(grid.temperature);
        fuel.captureAndRelease(grid.fuel);
        interaction.captureAndRelease(grid.interaction);
        pressure.captureAndRelease(grid.pressure);
        divergence.captureAndRelease(grid.divergence);
        solid.captureAndRelease(grid.solid);
        solid_vel.captureAndRelease(grid.solid_vel);
        solid_cells.captureAndRelease(grid.solid_cells);
        kinematic_touched_cells.captureAndRelease(grid.kinematic_touched_cells);
        kinematic_previous_solid.captureAndRelease(grid.kinematic_previous_solid);
        kinematic_previous_velocity.captureAndRelease(grid.kinematic_previous_velocity);
        substance_solid_cells.captureAndRelease(grid.substance_solid_cells);
        substance_solid_prev_cells.captureAndRelease(grid.substance_solid_prev_cells);
        substance_solid_prev_vel.captureAndRelease(grid.substance_solid_prev_vel);
        solid_gas_density.captureAndRelease(grid.solid_gas_density);
        solid_gas_temperature.captureAndRelease(grid.solid_gas_temperature);
        solid_gas_fuel.captureAndRelease(grid.solid_gas_fuel);
        solid_gas_flame.captureAndRelease(grid.solid_gas_flame);
        solid_gas_band.captureAndRelease(grid.solid_gas_band);
        u_weight.captureAndRelease(grid.u_weight);
        v_weight.captureAndRelease(grid.v_weight);
        w_weight.captureAndRelease(grid.w_weight);
        overlay_weight_restore_u.captureAndRelease(grid.overlay_weight_restore_u);
        overlay_weight_restore_v.captureAndRelease(grid.overlay_weight_restore_v);
        overlay_weight_restore_w.captureAndRelease(grid.overlay_weight_restore_w);
        fluid_phi.captureAndRelease(grid.fluid_phi);
        surface_dust_supply.captureAndRelease(grid.surface_dust_supply);
    }

    void restore(FluidSim::FluidGrid& grid) const {
        active_tiles.restore(grid.active_tiles);
        tile_active_mask.restore(grid.tile_active_mask);
        vel_x.restore(grid.vel_x);
        vel_y.restore(grid.vel_y);
        vel_z.restore(grid.vel_z);
        density.restore(grid.density);
        temperature.restore(grid.temperature);
        fuel.restore(grid.fuel);
        interaction.restore(grid.interaction);
        pressure.restore(grid.pressure);
        divergence.restore(grid.divergence);
        solid.restore(grid.solid);
        solid_vel.restore(grid.solid_vel);
        solid_cells.restore(grid.solid_cells);
        kinematic_touched_cells.restore(grid.kinematic_touched_cells);
        kinematic_previous_solid.restore(grid.kinematic_previous_solid);
        kinematic_previous_velocity.restore(grid.kinematic_previous_velocity);
        substance_solid_cells.restore(grid.substance_solid_cells);
        substance_solid_prev_cells.restore(grid.substance_solid_prev_cells);
        substance_solid_prev_vel.restore(grid.substance_solid_prev_vel);
        solid_gas_density.restore(grid.solid_gas_density);
        solid_gas_temperature.restore(grid.solid_gas_temperature);
        solid_gas_fuel.restore(grid.solid_gas_fuel);
        solid_gas_flame.restore(grid.solid_gas_flame);
        solid_gas_band.restore(grid.solid_gas_band);
        u_weight.restore(grid.u_weight);
        v_weight.restore(grid.v_weight);
        w_weight.restore(grid.w_weight);
        overlay_weight_restore_u.restore(grid.overlay_weight_restore_u);
        overlay_weight_restore_v.restore(grid.overlay_weight_restore_v);
        overlay_weight_restore_w.restore(grid.overlay_weight_restore_w);
        fluid_phi.restore(grid.fluid_phi);
        surface_dust_supply.restore(grid.surface_dust_supply);
    }

    std::size_t bytes() const {
        return
            active_tiles.bytes() +
            tile_active_mask.bytes() +
            vel_x.bytes() +
            vel_y.bytes() +
            vel_z.bytes() +
            density.bytes() +
            temperature.bytes() +
            fuel.bytes() +
            interaction.bytes() +
            pressure.bytes() +
            divergence.bytes() +
            solid.bytes() +
            solid_vel.bytes() +
            solid_cells.bytes() +
            kinematic_touched_cells.bytes() +
            kinematic_previous_solid.bytes() +
            kinematic_previous_velocity.bytes() +
            substance_solid_cells.bytes() +
            substance_solid_prev_cells.bytes() +
            substance_solid_prev_vel.bytes() +
            solid_gas_density.bytes() +
            solid_gas_temperature.bytes() +
            solid_gas_fuel.bytes() +
            solid_gas_flame.bytes() +
            solid_gas_band.bytes() +
            u_weight.bytes() +
            v_weight.bytes() +
            w_weight.bytes() +
            overlay_weight_restore_u.bytes() +
            overlay_weight_restore_v.bytes() +
            overlay_weight_restore_w.bytes() +
            fluid_phi.bytes() +
            surface_dust_supply.bytes();
    }
};

inline std::size_t fluidGridAllocationBytes(const FluidSim::FluidGrid& grid) {
    return
        grid.active_tiles.capacity() * sizeof(typename decltype(grid.active_tiles)::value_type) +
        grid.tile_active_mask.capacity() * sizeof(typename decltype(grid.tile_active_mask)::value_type) +
        grid.vel_x.capacity() * sizeof(typename decltype(grid.vel_x)::value_type) +
        grid.vel_y.capacity() * sizeof(typename decltype(grid.vel_y)::value_type) +
        grid.vel_z.capacity() * sizeof(typename decltype(grid.vel_z)::value_type) +
        grid.density.capacity() * sizeof(typename decltype(grid.density)::value_type) +
        grid.temperature.capacity() * sizeof(typename decltype(grid.temperature)::value_type) +
        grid.fuel.capacity() * sizeof(typename decltype(grid.fuel)::value_type) +
        grid.interaction.capacity() * sizeof(typename decltype(grid.interaction)::value_type) +
        grid.pressure.capacity() * sizeof(typename decltype(grid.pressure)::value_type) +
        grid.divergence.capacity() * sizeof(typename decltype(grid.divergence)::value_type) +
        grid.solid.capacity() * sizeof(typename decltype(grid.solid)::value_type) +
        grid.solid_vel.capacity() * sizeof(typename decltype(grid.solid_vel)::value_type) +
        grid.solid_cells.capacity() * sizeof(typename decltype(grid.solid_cells)::value_type) +
        grid.kinematic_touched_cells.capacity() * sizeof(typename decltype(grid.kinematic_touched_cells)::value_type) +
        grid.kinematic_previous_solid.capacity() * sizeof(typename decltype(grid.kinematic_previous_solid)::value_type) +
        grid.kinematic_previous_velocity.capacity() * sizeof(typename decltype(grid.kinematic_previous_velocity)::value_type) +
        grid.substance_solid_cells.capacity() * sizeof(typename decltype(grid.substance_solid_cells)::value_type) +
        grid.substance_solid_prev_cells.capacity() * sizeof(typename decltype(grid.substance_solid_prev_cells)::value_type) +
        grid.substance_solid_prev_vel.capacity() * sizeof(typename decltype(grid.substance_solid_prev_vel)::value_type) +
        grid.solid_gas_density.capacity() * sizeof(typename decltype(grid.solid_gas_density)::value_type) +
        grid.solid_gas_temperature.capacity() * sizeof(typename decltype(grid.solid_gas_temperature)::value_type) +
        grid.solid_gas_fuel.capacity() * sizeof(typename decltype(grid.solid_gas_fuel)::value_type) +
        grid.solid_gas_flame.capacity() * sizeof(typename decltype(grid.solid_gas_flame)::value_type) +
        grid.solid_gas_band.capacity() * sizeof(typename decltype(grid.solid_gas_band)::value_type) +
        grid.u_weight.capacity() * sizeof(typename decltype(grid.u_weight)::value_type) +
        grid.v_weight.capacity() * sizeof(typename decltype(grid.v_weight)::value_type) +
        grid.w_weight.capacity() * sizeof(typename decltype(grid.w_weight)::value_type) +
        grid.overlay_weight_restore_u.capacity() * sizeof(typename decltype(grid.overlay_weight_restore_u)::value_type) +
        grid.overlay_weight_restore_v.capacity() * sizeof(typename decltype(grid.overlay_weight_restore_v)::value_type) +
        grid.overlay_weight_restore_w.capacity() * sizeof(typename decltype(grid.overlay_weight_restore_w)::value_type) +
        grid.fluid_phi.capacity() * sizeof(typename decltype(grid.fluid_phi)::value_type) +
        grid.surface_dust_supply.capacity() * sizeof(typename decltype(grid.surface_dust_supply)::value_type);
}

} // namespace RayTrophiSim::SimFrameCache
