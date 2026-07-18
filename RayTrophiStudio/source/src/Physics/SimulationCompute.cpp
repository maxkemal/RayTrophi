#include "SimulationCompute.h"
#include "globals.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace RayTrophiSim {

namespace {

uint32_t usageBits(ComputeBufferUsage usage) {
    return static_cast<uint32_t>(usage);
}

bool rangeFits(std::size_t capacity, std::size_t offset, std::size_t size) {
    return offset <= capacity && size <= capacity - offset;
}

} // namespace

ComputeBufferUsage operator|(ComputeBufferUsage lhs, ComputeBufferUsage rhs) {
    return static_cast<ComputeBufferUsage>(usageBits(lhs) | usageBits(rhs));
}

ComputeBufferUsage operator&(ComputeBufferUsage lhs, ComputeBufferUsage rhs) {
    return static_cast<ComputeBufferUsage>(usageBits(lhs) & usageBits(rhs));
}

ComputeBufferUsage& operator|=(ComputeBufferUsage& lhs, ComputeBufferUsage rhs) {
    lhs = lhs | rhs;
    return lhs;
}

bool hasComputeBufferUsage(ComputeBufferUsage usage, ComputeBufferUsage flag) {
    return (usageBits(usage) & usageBits(flag)) != 0u;
}

ComputeBackendType CpuSimulationComputeBackend::type() const {
    return ComputeBackendType::CPU;
}

const char* CpuSimulationComputeBackend::name() const {
    return "CPU Simulation Compute";
}

ComputeBackendCaps CpuSimulationComputeBackend::caps() const {
    ComputeBackendCaps result;
    result.available = true;
    result.supports_async = false;
    result.supports_shared_graphics_interop = false;
    result.max_storage_buffer_bytes = static_cast<std::size_t>(std::numeric_limits<uint32_t>::max());
    result.max_threads_per_group = 1;
    return result;
}

ComputeBufferHandle CpuSimulationComputeBackend::createBuffer(const ComputeBufferDesc& desc) {
    if (desc.size_bytes == 0) {
        return {};
    }

    const uint64_t id = next_id_++;
    CpuBuffer buffer;
    buffer.desc = desc;
    buffer.bytes.resize(desc.size_bytes, 0u);
    buffers_.emplace(id, std::move(buffer));

    ComputeBufferHandle handle;
    handle.id = id;
    handle.backend = type();
    return handle;
}

bool CpuSimulationComputeBackend::destroyBuffer(ComputeBufferHandle handle) {
    if (handle.backend != type() || !handle.valid()) {
        return false;
    }
    return buffers_.erase(handle.id) > 0;
}

bool CpuSimulationComputeBackend::resizeBuffer(ComputeBufferHandle handle, std::size_t size_bytes) {
    if (handle.backend != type() || !handle.valid() || size_bytes == 0) {
        return false;
    }

    auto it = buffers_.find(handle.id);
    if (it == buffers_.end()) {
        return false;
    }

    it->second.desc.size_bytes = size_bytes;
    it->second.bytes.resize(size_bytes, 0u);
    return true;
}

std::size_t CpuSimulationComputeBackend::getBufferSize(ComputeBufferHandle handle) const {
    if (handle.backend != type() || !handle.valid()) {
        return 0;
    }

    const auto it = buffers_.find(handle.id);
    if (it == buffers_.end()) {
        return 0;
    }
    return it->second.bytes.size();
}

bool CpuSimulationComputeBackend::uploadBuffer(ComputeBufferHandle handle,
                                               const void* data,
                                               std::size_t size_bytes,
                                               std::size_t dst_offset_bytes) {
    if (handle.backend != type() || !handle.valid() || (!data && size_bytes > 0)) {
        return false;
    }

    auto it = buffers_.find(handle.id);
    if (it == buffers_.end() || !rangeFits(it->second.bytes.size(), dst_offset_bytes, size_bytes)) {
        return false;
    }

    if (size_bytes > 0) {
        std::memcpy(it->second.bytes.data() + dst_offset_bytes, data, size_bytes);
    }
    return true;
}

bool CpuSimulationComputeBackend::downloadBuffer(ComputeBufferHandle handle,
                                                 void* data,
                                                 std::size_t size_bytes,
                                                 std::size_t src_offset_bytes) const {
    if (handle.backend != type() || !handle.valid() || (!data && size_bytes > 0)) {
        return false;
    }

    const auto it = buffers_.find(handle.id);
    if (it == buffers_.end() || !rangeFits(it->second.bytes.size(), src_offset_bytes, size_bytes)) {
        return false;
    }

    if (size_bytes > 0) {
        std::memcpy(data, it->second.bytes.data() + src_offset_bytes, size_bytes);
    }
    return true;
}

SimulationComputeContext::SimulationComputeContext()
    : backend_(std::make_unique<CpuSimulationComputeBackend>()) {
}

void SimulationComputeContext::setBackend(std::unique_ptr<ISimulationComputeBackend> backend) {
    if (backend_) {
        backend_->synchronize();
    }
    if (backend) {
        backend_ = std::move(backend);
    } else {
        backend_ = std::make_unique<CpuSimulationComputeBackend>();
    }
}

ISimulationComputeBackend& SimulationComputeContext::backend() {
    return *backend_;
}

const ISimulationComputeBackend& SimulationComputeContext::backend() const {
    return *backend_;
}

ComputeBackendType SimulationComputeContext::backendType() const {
    return backend_->type();
}

const char* SimulationComputeContext::backendName() const {
    return backend_->name();
}

ComputeBackendCaps SimulationComputeContext::caps() const {
    return backend_->caps();
}

ComputeBufferHandle SimulationComputeContext::createBuffer(const ComputeBufferDesc& desc) {
    return backend_->createBuffer(desc);
}

bool SimulationComputeContext::destroyBuffer(ComputeBufferHandle handle) {
    return backend_->destroyBuffer(handle);
}

bool SimulationComputeContext::resizeBuffer(ComputeBufferHandle handle, std::size_t size_bytes) {
    return backend_->resizeBuffer(handle, size_bytes);
}

std::size_t SimulationComputeContext::getBufferSize(ComputeBufferHandle handle) const {
    return backend_->getBufferSize(handle);
}

bool SimulationComputeContext::uploadBuffer(ComputeBufferHandle handle,
                                            const void* data,
                                            std::size_t size_bytes,
                                            std::size_t dst_offset_bytes) {
    return backend_->uploadBuffer(handle, data, size_bytes, dst_offset_bytes);
}

bool SimulationComputeContext::downloadBuffer(ComputeBufferHandle handle,
                                              void* data,
                                              std::size_t size_bytes,
                                              std::size_t src_offset_bytes) const {
    return backend_->downloadBuffer(handle, data, size_bytes, src_offset_bytes);
}

void SimulationComputeContext::beginFrame(uint64_t frame_index) {
    backend_->beginFrame(frame_index);
}

void SimulationComputeContext::endFrame() {
    backend_->endFrame();
}

void SimulationComputeContext::synchronize() {
    backend_->synchronize();
}

void SimulationComputeContext::beginTransferBatch() {
    backend_->beginTransferBatch();
}

bool SimulationComputeContext::endTransferBatch() {
    return backend_->endTransferBatch();
}

void* SimulationComputeContext::nativeBufferPtr(ComputeBufferHandle handle) const {
    return backend_->nativeBufferPtr(handle);
}

bool SimulationComputeContext::dispatch(const ComputeDispatch& cmd) {
    return backend_->dispatch(cmd);
}

bool SimulationComputeContext::supportsDispatch() const {
    return backend_->supportsDispatch();
}

void logSimulationComputeInfo(const std::string& message) {
    SCENE_LOG_INFO(message);
}

void logSimulationComputeWarning(const std::string& message) {
    SCENE_LOG_WARN(message);
}

void logSimulationComputeError(const std::string& message) {
    SCENE_LOG_ERROR(message);
}

} // namespace RayTrophiSim
