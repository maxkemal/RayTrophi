#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace RayTrophiSim {

enum class ComputeBackendType {
    CPU,
    CUDA,
    VulkanCompute
};

enum class ComputeBufferUsage : uint32_t {
    None = 0u,
    Storage = 1u << 0,
    Upload = 1u << 1,
    Download = 1u << 2,
    Uniform = 1u << 3,
    ReadOnly = 1u << 4,
    ReadWrite = 1u << 5
};

ComputeBufferUsage operator|(ComputeBufferUsage lhs, ComputeBufferUsage rhs);
ComputeBufferUsage operator&(ComputeBufferUsage lhs, ComputeBufferUsage rhs);
ComputeBufferUsage& operator|=(ComputeBufferUsage& lhs, ComputeBufferUsage rhs);
bool hasComputeBufferUsage(ComputeBufferUsage usage, ComputeBufferUsage flag);

struct ComputeBufferDesc {
    std::string debug_name;
    std::size_t size_bytes = 0;
    ComputeBufferUsage usage = ComputeBufferUsage::Storage;
};

struct ComputeBufferHandle {
    uint64_t id = 0;
    ComputeBackendType backend = ComputeBackendType::CPU;

    bool valid() const { return id != 0; }
};

struct ComputeDispatchSize {
    uint32_t groups_x = 1;
    uint32_t groups_y = 1;
    uint32_t groups_z = 1;
};

// One compute kernel invocation: a backend-resolved kernel name, the thread-group
// counts, the storage buffers bound by slot index, and a small push-constant blob
// (the kernel parameters). The CUDA / Vulkan-compute backends resolve `kernel` to
// a registered launcher; the CPU backend leaves dispatch unimplemented (the CPU
// solver runs its own direct path instead).
struct ComputeDispatch {
    const char* kernel = nullptr;
    ComputeDispatchSize groups;
    const ComputeBufferHandle* buffers = nullptr;
    std::size_t buffer_count = 0;
    const void* constants = nullptr;
    std::size_t constants_size = 0;
};

struct ComputeBackendCaps {
    bool available = false;
    bool supports_async = false;
    bool supports_shared_graphics_interop = false;
    std::size_t max_storage_buffer_bytes = 0;
    uint32_t max_threads_per_group = 1;
};

class ISimulationComputeBackend {
public:
    virtual ~ISimulationComputeBackend() = default;

    virtual ComputeBackendType type() const = 0;
    virtual const char* name() const = 0;
    virtual ComputeBackendCaps caps() const = 0;

    virtual ComputeBufferHandle createBuffer(const ComputeBufferDesc& desc) = 0;
    virtual bool destroyBuffer(ComputeBufferHandle handle) = 0;
    virtual bool resizeBuffer(ComputeBufferHandle handle, std::size_t size_bytes) = 0;
    virtual std::size_t getBufferSize(ComputeBufferHandle handle) const = 0;

    virtual bool uploadBuffer(ComputeBufferHandle handle,
                              const void* data,
                              std::size_t size_bytes,
                              std::size_t dst_offset_bytes = 0) = 0;
    virtual bool downloadBuffer(ComputeBufferHandle handle,
                                void* data,
                                std::size_t size_bytes,
                                std::size_t src_offset_bytes = 0) const = 0;

    virtual void beginFrame(uint64_t frame_index) { (void)frame_index; }
    virtual void endFrame() {}
    virtual void synchronize() {}

    // Returns the raw backend pointer for a buffer (CUDA device pointer, etc.) or
    // nullptr if unsupported. Lets a backend-aware caller interop directly.
    virtual void* nativeBufferPtr(ComputeBufferHandle handle) const {
        (void)handle;
        return nullptr;
    }

    // Dispatch a registered compute kernel. Default: unsupported (CPU path).
    virtual bool dispatch(const ComputeDispatch& cmd) {
        (void)cmd;
        return false;
    }

    virtual bool supportsDispatch() const { return false; }
};

class CpuSimulationComputeBackend final : public ISimulationComputeBackend {
public:
    ComputeBackendType type() const override;
    const char* name() const override;
    ComputeBackendCaps caps() const override;

    ComputeBufferHandle createBuffer(const ComputeBufferDesc& desc) override;
    bool destroyBuffer(ComputeBufferHandle handle) override;
    bool resizeBuffer(ComputeBufferHandle handle, std::size_t size_bytes) override;
    std::size_t getBufferSize(ComputeBufferHandle handle) const override;

    bool uploadBuffer(ComputeBufferHandle handle,
                      const void* data,
                      std::size_t size_bytes,
                      std::size_t dst_offset_bytes = 0) override;
    bool downloadBuffer(ComputeBufferHandle handle,
                        void* data,
                        std::size_t size_bytes,
                        std::size_t src_offset_bytes = 0) const override;

private:
    struct CpuBuffer {
        ComputeBufferDesc desc;
        std::vector<uint8_t> bytes;
    };

    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, CpuBuffer> buffers_;
};

class SimulationComputeContext {
public:
    SimulationComputeContext();

    void setBackend(std::unique_ptr<ISimulationComputeBackend> backend);
    ISimulationComputeBackend& backend();
    const ISimulationComputeBackend& backend() const;

    ComputeBackendType backendType() const;
    const char* backendName() const;
    ComputeBackendCaps caps() const;

    ComputeBufferHandle createBuffer(const ComputeBufferDesc& desc);
    bool destroyBuffer(ComputeBufferHandle handle);
    bool resizeBuffer(ComputeBufferHandle handle, std::size_t size_bytes);
    std::size_t getBufferSize(ComputeBufferHandle handle) const;

    bool uploadBuffer(ComputeBufferHandle handle,
                      const void* data,
                      std::size_t size_bytes,
                      std::size_t dst_offset_bytes = 0);
    bool downloadBuffer(ComputeBufferHandle handle,
                        void* data,
                        std::size_t size_bytes,
                        std::size_t src_offset_bytes = 0) const;

    void beginFrame(uint64_t frame_index);
    void endFrame();
    void synchronize();

    void* nativeBufferPtr(ComputeBufferHandle handle) const;
    bool dispatch(const ComputeDispatch& cmd);
    bool supportsDispatch() const;  // true when the backend can run kernels (GPU)

private:
    std::unique_ptr<ISimulationComputeBackend> backend_;
};

// Implemented in the CUDA backend translation unit (SimulationComputeCuda.cu).
// Returns nullptr when no CUDA device is available.
std::unique_ptr<ISimulationComputeBackend> createCudaSimulationComputeBackend();

// Implemented in SimulationComputeVulkan.cpp.
// ctx.device / physical_device / compute_queue must be valid VkDevice /
// VkPhysicalDevice / VkQueue handles cast to void*. Returns nullptr on failure.
struct SimulationComputeVulkanContext;
std::unique_ptr<ISimulationComputeBackend>
createVulkanSimulationComputeBackend(const SimulationComputeVulkanContext& ctx);

// One-shot validation of the CUDA buffer + dispatch pipeline (alloc -> upload ->
// dispatch a scale kernel -> download round-trip). Logs the result.
bool selfTestCudaSimulationCompute();

void logSimulationComputeInfo(const std::string& message);
void logSimulationComputeWarning(const std::string& message);
void logSimulationComputeError(const std::string& message);

} // namespace RayTrophiSim
