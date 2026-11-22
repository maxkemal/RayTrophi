#include "VDBVolumeManager.h"
#include "Utils/OpenVdbUnavailable.h"

VDBVolumeManager::VDBVolumeManager() : last_error(OpenVdbUnavailable::message) {}
VDBVolumeManager::~VDBVolumeManager() = default;

VDBVolumeManager& VDBVolumeManager::getInstance() {
    static VDBVolumeManager manager;
    return manager;
}

void VDBVolumeManager::initialize() {}
void VDBVolumeManager::shutdown() {}

int VDBVolumeManager::loadVDB(const std::string&) {
    OpenVdbUnavailable::report();
    return -1;
}

void VDBVolumeManager::unloadVDB(int) {}
void VDBVolumeManager::unloadAll() {}

bool VDBVolumeManager::updateVolume(int, const std::string&, void*) {
    OpenVdbUnavailable::report();
    return false;
}

int VDBVolumeManager::registerOrUpdateLiveVolume(
    int, const std::string&, int, int, int, float, const float*, const float*, void*,
    const float*) {
    OpenVdbUnavailable::report();
    return -1;
}

bool VDBVolumeManager::exportDenseGridToVDB(
    const std::string&, int, int, int, float, float, float, float, const float*,
    const float*, const float*, const float*) {
    OpenVdbUnavailable::report();
    return false;
}

VDBVolumeData* VDBVolumeManager::getVolume(int) {
    return nullptr;
}

const VDBVolumeData* VDBVolumeManager::getVolume(int) const {
    return nullptr;
}

bool VDBVolumeManager::uploadToGPU(int, bool silent, void*) {
    if (!silent) {
        OpenVdbUnavailable::report();
    }
    return false;
}

void VDBVolumeManager::freeGPU(int) {}
void VDBVolumeManager::freeAllGPU() {}

void* VDBVolumeManager::getGPUGrid(int) const {
    return nullptr;
}

void* VDBVolumeManager::getGPUTemperatureGrid(int) const {
    return nullptr;
}

void* VDBVolumeManager::getHostGrid(int) const {
    return nullptr;
}

size_t VDBVolumeManager::getHostGridSize(int) const {
    return 0;
}

void* VDBVolumeManager::getHostTemperatureGrid(int) const {
    return nullptr;
}

size_t VDBVolumeManager::getHostTemperatureGridSize(int) const {
    return 0;
}

void* VDBVolumeManager::getHostFlameGrid(int) const {
    return nullptr;
}

size_t VDBVolumeManager::getHostFlameGridSize(int) const {
    return 0;
}

uint32_t VDBVolumeManager::getContentVersion(int) const {
    return 0;
}

void VDBVolumeManager::setLiveDenseGpuFields(
    int, uint64_t, uint64_t, uint64_t, uint64_t, int, int, int, float, float, float,
    float, uint64_t, bool, const int[3], const int[3]) {}

void VDBVolumeManager::setLiveDenseMajorant(int, uint64_t, int, int, int, int, uint64_t, int) {}

void VDBVolumeManager::setLiveDenseHostMirror(
    int, std::vector<float>&&, std::vector<float>&&, uint64_t) {}

const std::vector<float>* VDBVolumeManager::liveDenseDensityMirror(int) const {
    return nullptr;
}

const std::vector<float>* VDBVolumeManager::liveDenseTemperatureMirror(int) const {
    return nullptr;
}

uint64_t VDBVolumeManager::liveDenseMirrorVersion(int) const {
    return 0;
}

void VDBVolumeManager::clearLiveDenseGpuFields(int) {}
void VDBVolumeManager::setLiveSurfaceUvwAddress(int, uint64_t, uint64_t) {}
void VDBVolumeManager::setLiveDenseContentActive(int, bool) {}

float VDBVolumeManager::sampleDensityCPU(int, float, float, float) const {
    return 0.0f;
}

float VDBVolumeManager::sampleTemperatureCPU(int, float, float, float) const {
    return 0.0f;
}

bool VDBVolumeManager::hasTemperatureGrid(int) const {
    return false;
}

VDBDensityStats VDBVolumeManager::analyzeDensityStats(int, std::size_t) const {
    return {};
}
