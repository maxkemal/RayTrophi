#include "GasSimulator.h"
#include "Utils/OpenVdbUnavailable.h"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace FluidSim {

#include "../../Physics/GasSimulatorSerialization.inl"

GasSimulator::GasSimulator() = default;
GasSimulator::~GasSimulator() = default;

void GasSimulator::initialize(const GasSimulationSettings&) {
    OpenVdbUnavailable::report();
}

void GasSimulator::step(float, const Matrix4x4&) {
    OpenVdbUnavailable::report();
}

void GasSimulator::reset() {}
void GasSimulator::shutdown() {}

int GasSimulator::addEmitter(const Emitter&) {
    OpenVdbUnavailable::report();
    return -1;
}

void GasSimulator::removeEmitter(int) {}

float GasSimulator::sampleDensity(const Vec3&) const {
    return 0.0f;
}

float GasSimulator::sampleTemperature(const Vec3&) const {
    return 0.0f;
}

float GasSimulator::sampleFlameIntensity(const Vec3&) const {
    return 0.0f;
}

float GasSimulator::sampleFuel(const Vec3&) const {
    return 0.0f;
}

Vec3 GasSimulator::sampleVelocity(const Vec3&) const {
    return Vec3(0.0f, 0.0f, 0.0f);
}

void GasSimulator::startBake(int, int, const std::string&, const Matrix4x4&) {
    OpenVdbUnavailable::report();
}

void GasSimulator::cancelBake() {}

bool GasSimulator::loadBakedFrame(int) {
    OpenVdbUnavailable::report();
    return false;
}

bool GasSimulator::exportToVDB(const std::string&) const {
    OpenVdbUnavailable::report();
    return false;
}

bool GasSimulator::exportSequenceToVDB(const std::string&, const std::string&,
                                       int, int, const Matrix4x4&) {
    OpenVdbUnavailable::report();
    return false;
}

bool GasSimulator::canUseFFTPressureSolver() const {
    return false;
}

void GasSimulator::applyPreset(const std::string&) {
    OpenVdbUnavailable::report();
}

void GasSimulator::uploadToGPU() {}
void GasSimulator::downloadFromGPU() {}

GasSimulator::StateSnapshot GasSimulator::captureState() const {
    return {};
}

void GasSimulator::restoreState(const StateSnapshot&) {
    OpenVdbUnavailable::report();
}

} // namespace FluidSim
