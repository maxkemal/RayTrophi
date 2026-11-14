#include "GasSimulator.h"
#include <stdexcept>

namespace FluidSim {
bool GasSimulator::cudaBackendAvailable() const {
    return false;
}
void GasSimulator::bindCudaThread() {
}
void GasSimulator::initCUDA() {
    cuda_initialized = false;
}
void GasSimulator::freeCUDA() {
    cuda_initialized = false;
    gpu_data_valid = false;
}
void GasSimulator::clearGPU() {
}
void GasSimulator::stepCUDA(float, const Matrix4x4&) {
    throw std::logic_error("CUDA gas solver is not linked in Player");
}
void GasSimulator::uploadToGPU() {
}
void GasSimulator::downloadFromGPU() {
}
void GasSimulator::uploadForceFieldsToGPU() {
}
void GasSimulator::freeGPUForceFields() {
    d_force_fields = nullptr;
    gpu_force_field_count = 0;
}
void GasSimulator::initFFTSolver() {
}
void GasSimulator::cleanupFFTSolver() {
    use_fft_solver = false;
    fft_solver = nullptr;
}
void GasSimulator::uploadAdvancedEmittersToGPU() {
}
void GasSimulator::freeGPUAdvancedEmitters() {
    d_advanced_emitters = nullptr;
    gpu_advanced_emitter_count = 0;
}
} // namespace FluidSim
