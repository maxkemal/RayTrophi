#include "Backend/VulkanBackend.h"
#include "Camera.h"
#include "Atmosphere/AtmosphereClouds.h"
#include "Atmosphere/AtmosphereClimate.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace VulkanRT {
namespace {
constexpr uint32_t fieldWidth = 512;
constexpr uint32_t fieldHeight = 288;
constexpr uint64_t dropBytes = 256 * 32;
constexpr uint64_t snapshotBytes = 48 + dropBytes;
constexpr uint64_t stateBytes = sizeof(Backend::LensWeatherHeader) + dropBytes +
                                601 * snapshotBytes;
struct alignas(16) LensPush {
    float forward[4], up[4], originTime[4], windGlass[4], optics[4], right[4];
    uint32_t control[4];
};
static_assert(sizeof(LensPush) == 112, "Lens push ABI");

std::vector<uint32_t> loadShader(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return {};
    }
    const auto bytes = static_cast<std::streamoff>(stream.tellg());
    if (bytes < 20 || bytes % 4 != 0) {
        return {};
    }
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(words.data()), bytes);
    return stream && words.front() == 0x07230203u ? words : std::vector<uint32_t>{};
}

bool pipeline(VkDevice device, VkPipelineLayout layout, const std::vector<uint32_t>& words,
              VkPipeline& output) {
    if (output) {
        return true;
    }
    if (words.empty()) {
        return false;
    }
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = words.size() * 4;
    moduleInfo.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &module) != VK_SUCCESS) {
        return false;
    }
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.layout = layout;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    const bool ok = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info,
                                             nullptr, &output) == VK_SUCCESS;
    vkDestroyShaderModule(device, module, nullptr);
    return ok;
}

void barrier(VkCommandBuffer cmd, VkAccessFlags source, VkAccessFlags destination) {
    VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memory.srcAccessMask = source;
    memory.dstAccessMask = destination;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &memory, 0, nullptr, 0, nullptr);
}

float fieldFloat(uint16_t value) {
    const uint32_t exponent = (value >> 10u) & 31u;
    const uint32_t fraction = value & 1023u;
    float result = exponent == 0 ? std::ldexp(float(fraction), -24) :
        exponent == 31 ? std::numeric_limits<float>::infinity() :
        std::ldexp(float(1024u + fraction), int(exponent) - 25);
    return (value & 0x8000u) ? -result : result;
}
} // namespace

struct VulkanDevice::LensWeatherResources {
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline simulation = VK_NULL_HANDLE;
    VkPipeline draw = VK_NULL_HANDLE;
    BufferHandle state;
    ImageHandle field;
    ImageHandle solidField;
    nlohmann::json environment;
    uint32_t environmentRevision = 0;
    std::array<float, 3> temperatureProfile{};
    bool cleared = false;
    bool attempted = false;
};

void VulkanDevice::destroyLensWeather() {
    if (!m_lensWeather) {
        return;
    }
    auto& r = *m_lensWeather;
    if (r.simulation) {
        vkDestroyPipeline(m_device, r.simulation, nullptr);
    }
    if (r.draw) {
        vkDestroyPipeline(m_device, r.draw, nullptr);
    }
    if (r.layout) {
        vkDestroyPipelineLayout(m_device, r.layout, nullptr);
    }
    if (r.pool) {
        vkDestroyDescriptorPool(m_device, r.pool, nullptr);
    }
    if (r.descriptorLayout) {
        vkDestroyDescriptorSetLayout(m_device, r.descriptorLayout, nullptr);
    }
    if (r.field.image) {
        destroyImage(r.field);
    }
    if (r.solidField.image) {
        destroyImage(r.solidField);
    }
    if (r.state.buffer) {
        destroyBuffer(r.state);
    }
    m_lensWeather.reset();
}

bool VulkanDevice::ensureLensWeatherLayout() {
    if (m_lensWeather) {
        return m_lensWeather->set != VK_NULL_HANDLE;
    }
    m_lensWeather = std::make_shared<LensWeatherResources>();
    auto& r = *m_lensWeather;
    const auto fail = [&] {
        destroyLensWeather();
        return false;
    };
    BufferCreateInfo buffer;
    buffer.size = stateBytes;
    buffer.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
    buffer.location = MemoryLocation::GPU_ONLY;
    r.state = createBuffer(buffer);
    r.field = createImage2D(fieldWidth, fieldHeight, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    r.solidField = createImage2D(fieldWidth, fieldHeight, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!r.state.buffer || !r.field.view || !r.solidField.view) {
        return fail();
    }
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER :
                                             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    VkDescriptorSetLayoutCreateInfo descriptor{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor.bindingCount = 3;
    descriptor.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(m_device, &descriptor, nullptr,
                                    &r.descriptorLayout) != VK_SUCCESS) {
        return fail();
    }
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 1;
    pool.poolSizeCount = 2;
    pool.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(m_device, &pool, nullptr, &r.pool) != VK_SUCCESS) {
        return fail();
    }
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = r.pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &r.descriptorLayout;
    if (vkAllocateDescriptorSets(m_device, &allocate, &r.set) != VK_SUCCESS) {
        return fail();
    }
    VkDescriptorBufferInfo state{r.state.buffer, 0, stateBytes};
    VkDescriptorImageInfo field{};
    field.imageView = r.field.view;
    field.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo solidField{};
    solidField.imageView = r.solidField.view;
    solidField.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = r.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = bindings[i].descriptorType;
    }
    writes[0].pBufferInfo = &state;
    writes[1].pImageInfo = &field;
    writes[2].pImageInfo = &solidField;
    vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LensPush)};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &r.descriptorLayout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(m_device, &layout, nullptr, &r.layout) != VK_SUCCESS) {
        return fail();
    }
    auto cmd = beginSingleTimeCommands();
    if (!cmd) {
        return fail();
    }
    vkCmdFillBuffer(cmd, r.state.buffer, 0, VK_WHOLE_SIZE, 0);
    VkClearColorValue zero{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, r.field.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    vkCmdClearColorImage(cmd, r.solidField.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    barrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    endSingleTimeCommands(cmd);
    r.cleared = true;
    return true;
}

VkDescriptorSetLayout VulkanDevice::lensWeatherLayout() {
    return ensureLensWeatherLayout() ? m_lensWeather->descriptorLayout : VK_NULL_HANDLE;
}

bool VulkanDevice::loadLensWeatherPipelines(const std::string& directory) {
    if (!ensureLensWeatherLayout()) {
        return false;
    }
    auto& r = *m_lensWeather;
    if (!r.attempted) {
        r.attempted = true;
        pipeline(m_device, r.layout, loadShader(directory + "/lens_water_sim.spv"), r.simulation);
        pipeline(m_device, r.layout, loadShader(directory + "/lens_water_draw.spv"), r.draw);
    }
    return r.simulation && r.draw;
}

void VulkanDevice::noteLensWeatherEnvironment(const atmosphere::CloudState& clouds,
                                               const Vec3& wind) {
    if (!m_lensWeather || !m_lensWeatherCamera.settings.enabled) {
        return;
    }
    nlohmann::json environment;
    atmosphere::cloudsToJson(clouds, environment);
    environment["surface_wind_source"] = {wind.x, wind.y, wind.z};
    auto& r = *m_lensWeather;
    if (r.environment != environment) {
        r.environment = std::move(environment);
        ++r.environmentRevision;
    }
}

void VulkanDevice::setLensWeatherCamera(const Backend::CameraParams& camera) {
    m_lensWeatherCamera.settings = camera.lens_weather;
    m_lensWeatherCamera.origin = camera.origin;
    const auto direction = camera.lookAt - camera.origin;
    m_lensWeatherCamera.forward = direction.length() > 1e-6f ? direction.normalize() : Vec3(0, 0, -1);
    m_lensWeatherCamera.up = camera.up;
    m_lensWeatherCamera.blur_uv = camera.lens_weather_blur_uv;
    m_lensWeatherCamera.identity = camera.lens_weather_identity;
    if (camera.orthographic) {
        m_lensWeatherCamera.settings.enabled = false;
    }
}

void VulkanDevice::recordLensWeather(VkCommandBuffer cmd, VkPipelineLayout postLayout,
                                     uint32_t width, uint32_t height) {
    if (!m_lensWeather || !m_lensWeather->set) {
        return;
    }
    auto& r = *m_lensWeather;
    const auto& camera = m_lensWeatherCamera;
    const auto& settings = camera.settings;
    const bool enabled = settings.enabled && r.simulation && r.draw;
    if (enabled) {
        barrier(cmd, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
        if (m_cloudRtParamsBuffer.buffer) {
            const VkBufferCopy copy{offsetof(Backend::CloudParamsGPU, nearState), 0, 128};
            vkCmdCopyBuffer(cmd, m_cloudRtParamsBuffer.buffer, r.state.buffer, 1, &copy);
        } else {
            vkCmdFillBuffer(cmd, r.state.buffer, 0, 128, 0);
        }
        barrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        Vec3 right = Vec3::cross(camera.forward, camera.up);
        right = right.length() > 1e-6f ? right.normalize() : Vec3(1, 0, 0);
        const Vec3 up = Vec3::cross(right, camera.forward).normalize();
        LensPush push{};
        const auto vector = [](float* out, const Vec3& value) {
            out[0] = value.x;
            out[1] = value.y;
            out[2] = value.z;
        };
        vector(push.forward, camera.forward);
        vector(push.up, up);
        vector(push.right, right);
        vector(push.originTime, camera.origin);
        push.up[3] = settings.hood_deg > 0 ?
            std::cos(settings.hood_deg * 0.01745329252f) : -1.0f;
        push.originTime[3] = m_lensTime;
        push.windGlass[0] = m_lensWind[0];
        push.windGlass[1] = m_lensWind[1];
        // Area-preserving square UV chart for a circular front element (pi D^2 / 4).
        push.windGlass[2] = settings.diameter_mm * 0.001f * 0.88622692545f;
        push.windGlass[3] = settings.hydrophobic ? 0.0012f : 0.0025f;
        push.optics[0] = camera.blur_uv;
        push.optics[1] = float(std::max(width, 1u)) / float(std::max(height, 1u));
        const auto ambient = atmosphere::ambientSnapshot();
        const std::array<float, 3> profile = {ambient.climate.surface_temperature_k,
            ambient.climate.lapse_rate_k_per_m, ambient.altitude_offset_m};
        if (r.temperatureProfile != profile) {
            r.temperatureProfile = profile;
            ++r.environmentRevision; // Branch replay; preserve deposited water.
        }
        push.optics[2] = atmosphere::sampleClimate(ambient.climate,
            camera.origin.y + ambient.altitude_offset_m).temperature_k - 273.15f;
        push.control[3] = settings.seed;
        push.control[0] = 1;
        push.control[2] = settings.history_revision + r.environmentRevision;
        push.control[1] = settings.revision ^ uint32_t(camera.identity) ^
                          uint32_t(camera.identity >> 32u);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.simulation);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.layout,
                                0, 1, &r.set, 0, nullptr);
        vkCmdPushConstants(cmd, r.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.draw);
        vkCmdDispatch(cmd, fieldWidth / 8, fieldHeight / 8, 1);
        barrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        r.cleared = false;
    } else if (!r.cleared) {
        barrier(cmd, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT);
        VkClearColorValue zero{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, r.field.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        vkCmdClearColorImage(cmd, r.solidField.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        vkCmdFillBuffer(cmd, r.state.buffer, 0, VK_WHOLE_SIZE, 0);
        barrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        r.cleared = true;
    }
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, postLayout,
                            1, 1, &r.set, 0, nullptr);
}

nlohmann::json VulkanDevice::sampleLensWeather() {
    const bool resources = m_lensWeather && m_lensWeather->set;
    const bool pipelines = resources && m_lensWeather->simulation && m_lensWeather->draw;
    struct Readback {
        Backend::LensWeatherHeader header;
        std::array<float, 256 * 8> drops;
    } readback{};
    static_assert(sizeof(Readback) == sizeof(Backend::LensWeatherHeader) + dropBytes,
                  "Lens readback ABI");
    auto& header = readback.header;
    float fieldMaxCoverage = 0.0f;
    float fieldMaxShift = 0.0f;
    double fieldMeanCoverage = 0.0;
    uint32_t fieldWetPixels = 0;
    float fieldMaxSnow = 0.0f;
    float fieldMaxIce = 0.0f;
    if (resources) {
        waitIdle(); // Explicit measurement only; normal rendering never reads back.
        auto command = beginSingleTimeCommands();
        if (!command) {
            throw std::runtime_error("Cannot record lens measurement visibility barrier");
        }
        barrier(command, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
        endSingleTimeCommands(command);
        downloadBuffer(m_lensWeather->state, &readback, sizeof(readback));
        // Explicit probe only: measure the actual drawn optical field rather
        // than treating a positive simulation drop count as display acceptance.
        BufferCreateInfo stagingInfo;
        stagingInfo.size = uint64_t(fieldWidth) * fieldHeight * 8;
        stagingInfo.usage = BufferUsage::TRANSFER_DST;
        stagingInfo.location = MemoryLocation::GPU_TO_CPU;
        auto staging = createBuffer(stagingInfo);
        if (!staging.buffer) {
            throw std::runtime_error("Cannot allocate lens field measurement buffer");
        }
        copyImageToBuffer(m_lensWeather->field, staging);
        std::vector<uint16_t> pixels(size_t(fieldWidth) * fieldHeight * 4);
        downloadBuffer(staging, pixels.data(), stagingInfo.size);
        for (size_t i = 0; i < pixels.size(); i += 4) {
            const float coverage = fieldFloat(pixels[i + 3]);
            const float x = fieldFloat(pixels[i]);
            const float y = fieldFloat(pixels[i + 1]);
            fieldMaxCoverage = std::max(fieldMaxCoverage, coverage);
            fieldMaxShift = std::max(fieldMaxShift, std::sqrt(x * x + y * y));
            fieldMeanCoverage += coverage;
            fieldWetPixels += coverage > 0.0001f ? 1u : 0u;
        }
        fieldMeanCoverage /= double(fieldWidth) * fieldHeight;
        copyImageToBuffer(m_lensWeather->solidField, staging);
        downloadBuffer(staging, pixels.data(), stagingInfo.size);
        destroyBuffer(staging);
        for (size_t i = 0; i < pixels.size(); i += 4) {
            fieldMaxSnow = std::max(fieldMaxSnow, fieldFloat(pixels[i]));
            fieldMaxIce = std::max(fieldMaxIce, fieldFloat(pixels[i + 1]));
        }
    }
    std::uint64_t hash = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(readback.drops.data());
    for (size_t i = 0; i < sizeof(readback.drops); ++i) {
        hash = (hash ^ bytes[i]) * 1099511628211ULL;
    }
    double volume = 0;
    double snowVolume = 0;
    double iceVolume = 0;
    uint32_t snowCount = 0;
    uint32_t iceCount = 0;
    for (size_t i = 0; i < 256; ++i) {
        const double total = std::max(0.0f, readback.drops[i * 8 + 2]);
        if (total <= 0) {
            continue;
        }
        const double snow = std::clamp(double(readback.drops[i * 8 + 6]), 0.0, total);
        const double ice = std::clamp(double(readback.drops[i * 8 + 7]), 0.0, total - snow);
        volume += total * 4.188790204786;
        snowVolume += snow * 4.188790204786;
        iceVolume += ice * 4.188790204786;
        snowCount += snow > 0 ? 1u : 0u;
        iceCount += ice > 0 ? 1u : 0u;
    }
    const auto identity = m_lensWeatherCamera.identity;
    const uint32_t expectedKey = m_lensWeatherCamera.settings.revision ^
        uint32_t(identity) ^ uint32_t(identity >> 32u);
    const auto targetStep = std::floor(std::clamp(
        m_lensTime * 60.0f + 0.0001f, 0.0f, 5184000.0f));
    return {{"enabled", m_lensWeatherCamera.settings.enabled}, {"resources_ready", resources},
        {"pipelines_ready", pipelines}, {"initialized", header.meta[2] != 0},
        {"state_matches_camera", header.meta[3] == expectedKey}, {"capacity", 256}, {"history_steps", 601},
        {"step_seconds", 1.0 / 60}, {"step", header.meta[0]},
        {"drop_count", header.meta[1]}, {"impact_rate_per_s", header.stats[0]},
        {"coverage", header.stats[1]}, {"film_fraction", header.stats[2]},
        {"field_max_coverage", fieldMaxCoverage}, {"field_mean_coverage", fieldMeanCoverage},
        {"field_max_shift_uv", fieldMaxShift}, {"field_wet_pixels", fieldWetPixels},
        {"optical_blur_uv", m_lensWeatherCamera.blur_uv},
        {"temperature_source", "atmosphere"},
        {"atmosphere_temperature_c",
            atmosphere::sampleAmbient(m_lensWeatherCamera.origin).temperature_k - 273.15f},
        {"field_max_snow", fieldMaxSnow}, {"field_max_ice", fieldMaxIce},
        {"state_hash", std::to_string(hash)}, {"water_volume_m3", volume},
        {"capacity_merges", header.stats[3]}, {"history_misses", header.history[2]},
        {"forcing_revision", resources ? m_lensWeather->environmentRevision : 0},
        {"history_first_step", header.history[0]}, {"history_last_step", header.history[1]},
        {"epoch_step", header.history[3]},
        {"snow_fraction", volume > 0 ? snowVolume / volume : 0},
        {"ice_fraction", volume > 0 ? iceVolume / volume : 0},
        {"snow_water_volume_m3", snowVolume}, {"ice_water_volume_m3", iceVolume},
        {"liquid_water_volume_m3", std::max(0.0, volume - snowVolume - iceVolume)},
        {"snow_count", snowCount}, {"ice_count", iceCount},
        {"scope", "rain_snow_ice"}, {"target_step", targetStep}};
}
} // namespace VulkanRT

namespace Backend {
void VulkanBackendAdapter::updateLensWeather(const Camera& camera) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    copyLensWeatherCamera(camera, m_camera);
    if (m_device) {
        m_device->setLensWeatherCamera(m_camera);
    }
    // Reuse the canonical near-camera weather evaluation, also when airborne
    // near drops are disabled. No accumulation reset: this is display glass.
    if (m_cloudStateSet) {
        setCloudState(m_cachedClouds, m_cloudTime, m_cloudDrift, m_cloudWind);
    }
}

nlohmann::json VulkanBackendAdapter::sampleLensWeather() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    return m_device ? m_device->sampleLensWeather() :
        nlohmann::json{{"resources_ready", false}, {"pipelines_ready", false}};
}
} // namespace Backend
