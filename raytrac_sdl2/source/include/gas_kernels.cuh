#pragma once
#include <cuda_runtime.h>

namespace FluidSim {

// CUDA Kernels wrapper functions
void cuda_init_simulation(int nx, int ny, int nz, 
                         float** d_rho, float** d_temp, float** d_fuel,
                         float** d_vx, float** d_vy, float** d_vz,
                         float** d_p, float** d_div,
                         float** d_vort_x, float** d_vort_y, float** d_vort_z);

void cuda_free_simulation(float* d_rho, float* d_temp, float* d_fuel,
                         float* d_vx, float* d_vy, float* d_vz,
                         float* d_p, float* d_div,
                         float* d_vort_x, float* d_vort_y, float* d_vort_z);

void cuda_upload_data(int nx, int ny, int nz,
                     float* h_rho, float* h_temp, float* h_fuel,
                     float* h_vx, float* h_vy, float* h_vz,
                     float* d_rho, float* d_temp, float* d_fuel,
                     float* d_vx, float* d_vy, float* d_vz);

void cuda_download_data(int nx, int ny, int nz,
                       float* d_rho, float* d_temp, float* d_fuel,
                       float* d_vx, float* d_vy, float* d_vz,
                       float* h_rho, float* h_temp, float* h_fuel,
                       float* h_vx, float* h_vy, float* h_vz);

void cuda_step_simulation(
    int nx, int ny, int nz, float dt,
    float* d_rho, float* d_temp, float* d_fuel,
    float* d_vx, float* d_vy, float* d_vz,
    float* d_p, float* d_div,
    float* d_vort_x, float* d_vort_y, float* d_vort_z,
    // Settings
    float vorticity_str, float buoyancy_alpha, float buoyancy_beta,
    float ambient_temp, float density_dissipation, float vel_dissipation,
    float temp_dissipation, float fuel_dissipation,
    float ignition_temp, float burn_rate, float heat_release, 
    float smoke_gen, float expansion,
    int pressure_iters,
    // External forces
    float wind_x, float wind_y, float wind_z
);

void cuda_apply_emitter(int nx, int ny, int nz, float dt,
                        float* d_rho, float* d_temp, float* d_fuel,
                        float* d_vx, float* d_vy, float* d_vz,
                        int shape_type, float cx, float cy, float cz,
                        float sx, float sy, float sz,
                        float density_rate, float temp, float fuel_rate,
                        float emit_vx, float emit_vy, float emit_vz);

} // namespace FluidSim
