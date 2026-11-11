#include "Fluid/MatterSeedPlanning.h"
#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/MatterGrain.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/MeshVolumeTest.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"
#include "ParticleSimulation.h"
#include "SimulationComputeVulkanContext.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
#include <vulkan/vulkan.h>

extern RayTrophiSim::SimulationComputeVulkanContext g_vulkan_sim_compute_ctx;

namespace RayTrophiSim::Fluid {
namespace {
uint64_t checked(double value) {
    // Keep arithmetic and downstream host/GPU indices representable.
    if (!std::isfinite(value) || value < 0 || value > UINT32_MAX) {
        throw std::runtime_error("Seed recipe exceeds supported 32-bit particle/grid indexing. "
                                 "Reduce layer size or grid resolution.");
    }
    return static_cast<uint64_t>(std::ceil(value));
}
} // namespace

nlohmann::json planMatterSeeds(const SimulationGridDomainDesc &domain,
                               const MatterSeedMeshResolver &meshes,
                               const FluidSim::FluidGrid &grid, float ambient_kelvin) {
    nlohmann::json rows = nlohmann::json::array();
    uint64_t total = 0;
    const double h = grid.voxel_size;
    if (!(h > 0) || !std::isfinite(h) || grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        throw std::runtime_error("Seed planning requires positive finite voxel size");
    }
    const Vec3 domainLow = grid.origin;
    const Vec3 domainHigh = grid.origin + Vec3(grid.nx * grid.voxel_size, grid.ny * grid.voxel_size,
                                               grid.nz * grid.voxel_size);
    const double gridCells = static_cast<double>(grid.nx) * grid.ny * grid.nz;
    checked(gridCells);
    for (const auto &source : domain.matter_seed_sources) {
        if (!source.enabled) {
            continue;
        }
        Vec3 lo = source.bounds_min;
        Vec3 hi = source.bounds_max;
        if (!source.mesh_object.empty()) {
            const auto mesh = meshes ? meshes(source.mesh_object) : nullptr;
            if (!mesh || mesh->empty()) {
                throw std::runtime_error(source.name + ": mesh is missing or empty");
            }
            lo = mesh->boundsMin();
            hi = mesh->boundsMax();
        }
        lo = Vec3::max(lo, domainLow);
        hi = Vec3::min(hi, domainHigh);
        const auto *profile = tryFindSubstance(source.substance);
        if (!profile) {
            throw std::runtime_error(source.name + ": unknown substance");
        }
        auto params = domain.fluid_params;
        params.default_substance = source.substance;
        const auto model =
            substanceBirthModel(substanceTag(source.substance), params,
                                matterSeedBirthKelvin(source, domain, ambient_kelvin));
        const bool dem = model == MatterConstitutiveModel::Granular &&
                         profile->granular_transport == MatterGranularTransport::Dem;
        double count = 0;
        if (lo.x < hi.x && lo.y < hi.y && lo.z < hi.z) {
            if (dem) {
                const double radius = matterGrainMaximumRadius(domain.fluid_params.grain);
                const double jitter = 0.03 * radius;
                const double spacing = (2.0002 * radius + 2.8285 * jitter) / std::sqrt(2.0);
                if (!(spacing > 0) || !std::isfinite(spacing)) {
                    throw std::runtime_error("DEM seed planning requires a positive grain radius");
                }
                const auto extent = [&](double low, double high) {
                    const double width = high - low - 2 * (radius + jitter);
                    return width < 0 ? 0.0 : std::min(1000000.0, std::floor(width / spacing) + 1);
                };
                count =
                    std::ceil(extent(lo.x, hi.x) * extent(lo.y, hi.y) * extent(lo.z, hi.z) / 2.0);
            } else {
                // Includes boundary cells and ignores solids / mesh holes. The
                // actual birth path filters these, so this is an upper bound.
                const auto cells = [&](float low, float high, float origin, int n) {
                    // Match seedBoxCellRange's float arithmetic at grid boundaries.
                    const double first = std::clamp(
                        static_cast<double>(std::floor((low - origin) / grid.voxel_size)), 0.0,
                        static_cast<double>(n - 1));
                    const double last = std::clamp(
                        static_cast<double>(std::floor((high - origin) / grid.voxel_size)), 0.0,
                        static_cast<double>(n - 1));
                    return std::max(0.0, last - first + 1);
                };
                count = cells(lo.x, hi.x, domainLow.x, grid.nx) *
                        cells(lo.y, hi.y, domainLow.y, grid.ny) *
                        cells(lo.z, hi.z, domainLow.z, grid.nz) *
                        domain.fluid_seed_particles_per_cell;
            }
        }
        const auto bound = checked(count);
        total = checked(static_cast<double>(total) + bound);
        rows.push_back({{"id", source.id}, {"particle_upper_bound", bound}});
    }
    const uint64_t capacity =
        total ? checked(std::min<double>(UINT32_MAX, std::ceil(total * 1.15) + 64))
              : std::max<uint64_t>(1, domain.fluid_max_particles);
    // Planning allowances cover multiple particle/state/contact/transfer copies
    // and grid scratch. They are deliberately conservative heuristics, not a
    // calibrated peak-allocation model. Driver fragmentation remains possible.
    const uint64_t plannedCapacity = capacity;
    const uint64_t gpu =
        total ? plannedCapacity * 2048 + static_cast<uint64_t>(gridCells) * 512 : 0;
    const uint64_t host =
        total ? plannedCapacity * 4096 + static_cast<uint64_t>(gridCells) * 1024 : 0;
    return {{"particle_upper_bound", total},
            {"suggested_capacity", std::max<uint64_t>(1, capacity)},
            {"layers", rows},
            {"gpu_planning_bytes", gpu},
            {"host_planning_bytes", host},
            {"gpu_required", domain.backend == SimulationDomainBackend::GPU_Vulkan},
            {"memory_guard", domain.matter_seed_memory_guard},
            {"estimate_kind", "conservative_bounds_and_memory_allowance"}};
}

bool checkMatterSeedMemory(nlohmann::json &plan, std::string &error) {
    struct Availability {
        bool host_measured = false;
        bool gpu_measured = false;
        uint64_t host_free = 0;
        uint64_t gpu_free = 0;
    };
    // UI asks every frame. Refresh measurements at most once per second, and
    // whenever the simulation device changes. All callers use the frame thread.
    static Availability memory;
    static auto last = std::chrono::steady_clock::time_point{};
    static void *lastDevice = nullptr;
    const auto now = std::chrono::steady_clock::now();
    const auto physical = static_cast<VkPhysicalDevice>(g_vulkan_sim_compute_ctx.physical_device);
    if (now - last > std::chrono::seconds(1) || lastDevice != physical) {
        memory = Availability{};
        last = now;
        lastDevice = physical;
#ifdef _WIN32
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            memory.host_measured = true;
            memory.host_free = status.ullAvailPhys;
        }
#endif
        if (physical) {
            uint32_t count = 0;
            vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> extensions(count);
            if (vkEnumerateDeviceExtensionProperties(physical, nullptr, &count,
                                                     extensions.data()) == VK_SUCCESS) {
                const bool supported =
                    std::any_of(extensions.begin(), extensions.end(), [](const auto &e) {
                        return std::string(e.extensionName) == VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
                    });
                if (supported) {
                    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
                    VkPhysicalDeviceMemoryProperties2 props{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
                    props.pNext = &budget;
                    vkGetPhysicalDeviceMemoryProperties2(physical, &props);
                    for (uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; ++i) {
                        if ((props.memoryProperties.memoryHeaps[i].flags &
                             VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                            budget.heapBudget[i] != 0) {
                            memory.gpu_measured = true;
                            memory.gpu_free += budget.heapBudget[i] > budget.heapUsage[i]
                                                   ? budget.heapBudget[i] - budget.heapUsage[i]
                                                   : 0;
                        }
                    }
                }
            }
        }
    }
    plan["memory"] = {{"host_measured", memory.host_measured},
                      {"gpu_measured", memory.gpu_measured},
                      {"host_available_bytes", memory.host_free},
                      {"gpu_available_bytes", memory.gpu_free},
                      {"available_fraction", 0.7}};
    const bool hostExceeded =
        memory.host_measured &&
        plan.at("host_planning_bytes").get<uint64_t>() > memory.host_free * 0.7;
    const bool gpuExceeded = plan.at("gpu_required").get<bool>() && memory.gpu_measured &&
                             plan.at("gpu_planning_bytes").get<uint64_t>() > memory.gpu_free * 0.7;
    if (hostExceeded || gpuExceeded) {
        const auto required =
            plan.at(hostExceeded ? "host_planning_bytes" : "gpu_planning_bytes").get<uint64_t>();
        const auto available = hostExceeded ? memory.host_free : memory.gpu_free;
        error = std::string("Seed planning allowance: ") +
                std::to_string(required / (1024 * 1024)) + " MiB; currently available " +
                (hostExceeded ? "system RAM: " : "GPU budget: ") +
                std::to_string(available / (1024 * 1024)) +
                " MiB. The guard reserves 30% headroom. Reduce layer volume, particles per voxel, "
                "or grid resolution. Existing particles were preserved.";
        return false;
    }
    error.clear();
    return true;
}

} // namespace RayTrophiSim::Fluid

namespace RayTrophiSim {
nlohmann::json
ParticleSimulationSystem::matterSeedPlan(const SimulationGridDomainDesc &domain) const {
    const auto found = std::find_if(grid_domains_.begin(), grid_domains_.end(),
                                    [&](const auto &item) { return item.name == domain.name; });
    const auto index = static_cast<std::size_t>(found - grid_domains_.begin());
    if (found == grid_domains_.end() || index >= grid_domain_states_.size() ||
        !grid_domain_states_[index].valid) {
        throw std::runtime_error("Seed preview is waiting for the domain grid");
    }
    return Fluid::planMatterSeeds(domain, mesh_volume_resolver_,
                                  Fluid::liquidGrid(grid_domain_states_[index]),
                                  fluidDomainAmbientKelvin(domain, world_thermal_));
}
} // namespace RayTrophiSim
