#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace RayFusion {
struct ProbeMarker {
    std::array<float, 3> position{};
    float radius = 0.08f;
    // 0 pending; 1 published usable; 2 published but rejected inside geometry.
    uint32_t state = 0;
};

// Viewport-owned, depth-tested LDR markers. No ImGui foreground primitives,
// readbacks, descriptor updates or per-frame GPU buffer allocations.
class ProbeOverlay {
public:
    explicit ProbeOverlay(VkDevice device) : device_(device) {}
    ~ProbeOverlay();
    bool initialize(VkRenderPass pass, const std::string& shaderDir);
    void record(VkCommandBuffer cmd, const float viewProj[16],
                const std::vector<ProbeMarker>& markers);
    const std::string& reason() const { return reason_; }
    bool ready() const { return pipeline_ != VK_NULL_HANDLE; }
private:
    VkDevice device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    bool attempted_ = false;
    std::string reason_;
};
}
