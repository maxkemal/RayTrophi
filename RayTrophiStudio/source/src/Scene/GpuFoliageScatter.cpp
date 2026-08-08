#include "GpuFoliageScatter.h"
#include "InstanceGroup.h"
#include "SimulationCompute.h"
#include "TerrainSystem.h"
#include "Texture.h"
#include "Transform.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace FoliageGPU {
namespace {

struct alignas(16) CandidateValues {
    float includeValues[4];
    float rejectValues[4];
    float terrainValues[4];
    uint32_t ids[4];
};
static_assert(sizeof(CandidateValues) == 64, "parity candidate ABI changed");

struct alignas(16) ParityConstants {
    uint32_t count;
    float heightMin;
    float heightMax;
    float slopeMax;
    float curvatureMin;
    float curvatureMax;
    float exclusionThreshold;
    float directionInfluence;
    uint32_t allowFlags;
    uint32_t padding[3];
};
static_assert(sizeof(ParityConstants) == 48, "parity push constants changed");

ParityStats g_lastStats;
TerrainScatterStats g_lastTerrainStats;

uint32_t evaluateCPU(const CandidateValues& c, const ParityConstants& pc) {
    constexpr uint32_t RejectNonFinite = 1u << 0;
    constexpr uint32_t RejectEdge = 1u << 1;
    constexpr uint32_t RejectHeight = 1u << 2;
    constexpr uint32_t RejectSlope = 1u << 3;
    constexpr uint32_t RejectCurvature = 1u << 4;
    constexpr uint32_t RejectDirection = 1u << 5;
    constexpr uint32_t RejectExclusionField = 1u << 6;
    constexpr uint32_t RejectExclusionSplat = 1u << 7;
    constexpr uint32_t RejectDensity = 1u << 8;

    uint32_t rejection = 0;
    const float includeWeight = std::clamp(c.includeValues[0], 0.0f, 1.0f) *
                                std::clamp(c.includeValues[1], 0.0f, 1.0f) *
                                std::clamp(c.includeValues[2], 0.0f, 1.0f);
    if (c.terrainValues[3] < 0.5f) rejection |= RejectNonFinite;
    if (c.terrainValues[2] < 0.0f) rejection |= RejectEdge;
    if (c.rejectValues[2] < pc.heightMin || c.rejectValues[2] > pc.heightMax)
        rejection |= RejectHeight;
    if (c.rejectValues[3] > pc.slopeMax) rejection |= RejectSlope;

    const bool ridge = c.terrainValues[0] < pc.curvatureMin;
    const bool gully = c.terrainValues[0] > pc.curvatureMax;
    const bool flatRegion = !ridge && !gully;
    if ((ridge && (pc.allowFlags & 1u) == 0u) ||
        (flatRegion && (pc.allowFlags & 2u) == 0u) ||
        (gully && (pc.allowFlags & 4u) == 0u)) rejection |= RejectCurvature;

    const float directionProbability = 1.0f +
        std::clamp(pc.directionInfluence, 0.0f, 1.0f) *
        (std::clamp(c.terrainValues[1], 0.0f, 1.0f) - 1.0f);
    if (candidateRandom01(c.ids[1], c.ids[0], 1u) > directionProbability)
        rejection |= RejectDirection;
    if (c.rejectValues[0] >= pc.exclusionThreshold) rejection |= RejectExclusionField;
    if (c.rejectValues[1] >= pc.exclusionThreshold) rejection |= RejectExclusionSplat;
    if (candidateRandom01(c.ids[1], c.ids[0], 0u) > includeWeight) rejection |= RejectDensity;
    return rejection;
}

} // namespace

const ParityStats& getLastParityStats() { return g_lastStats; }

ParityStats runParityTest(uint32_t candidateCount) {
    g_lastStats = {};
    candidateCount = std::clamp(candidateCount, 1u, 2000000u);
    g_lastStats.candidateCount = candidateCount;

    std::vector<CandidateValues> candidates(candidateCount);
    constexpr uint32_t seed = 0x51a7c3d9u;
    for (uint32_t i = 0; i < candidateCount; ++i) {
        auto& c = candidates[i];
        c.includeValues[0] = candidateRandom01(seed, i, 2u);
        c.includeValues[1] = candidateRandom01(seed, i, 3u);
        c.includeValues[2] = candidateRandom01(seed, i, 4u);
        c.includeValues[3] = candidateRandom01(seed, i, 5u);
        c.rejectValues[0] = candidateRandom01(seed, i, 6u);
        c.rejectValues[1] = candidateRandom01(seed, i, 7u);
        c.rejectValues[2] = candidateRandom01(seed, i, 8u) * 2400.0f - 200.0f;
        c.rejectValues[3] = candidateRandom01(seed, i, 9u) * 90.0f;
        c.terrainValues[0] = candidateRandom01(seed, i, 10u) * 8.0f - 4.0f;
        c.terrainValues[1] = candidateRandom01(seed, i, 11u);
        c.terrainValues[2] = candidateRandom01(seed, i, 12u) - 0.02f;
        c.terrainValues[3] = (i % 997u) ? 1.0f : 0.0f;
        c.ids[0] = i; c.ids[1] = seed; c.ids[2] = 0u; c.ids[3] = 0u;
    }

    ParityConstants pc{};
    pc.count = candidateCount;
    pc.heightMin = 0.0f; pc.heightMax = 1800.0f; pc.slopeMax = 52.0f;
    pc.curvatureMin = -1.5f; pc.curvatureMax = 1.75f;
    pc.exclusionThreshold = 0.72f; pc.directionInfluence = 0.63f;
    pc.allowFlags = 1u | 2u; // reject gullies in this corpus

    std::vector<ScatterDecisionGPU> cpu(candidateCount), gpu(candidateCount);
    const auto cpuStart = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < candidateCount; ++i) {
        cpu[i].candidateId = i;
        cpu[i].reserved = candidateRandomBits(seed, i, 13u);
        cpu[i].rejectionMask = evaluateCPU(candidates[i], pc);
        cpu[i].accepted = cpu[i].rejectionMask == 0u ? 1u : 0u;
    }
    g_lastStats.cpuMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cpuStart).count();

    std::lock_guard<std::recursive_mutex> lock(RayTrophiSim::sharedMeshComputeMutex());
    RayTrophiSim::ISimulationComputeBackend* backend = RayTrophiSim::acquireSharedMeshComputeBackend();
    if (!backend || !backend->supportsDispatch()) return g_lastStats;
    g_lastStats.gpuAvailable = true;

    RayTrophiSim::ComputeBufferDesc inputDesc{};
    inputDesc.debug_name = "foliage_parity_candidates";
    inputDesc.size_bytes = candidates.size() * sizeof(CandidateValues);
    inputDesc.usage = RayTrophiSim::ComputeBufferUsage::Storage;
    RayTrophiSim::ComputeBufferDesc outputDesc{};
    outputDesc.debug_name = "foliage_parity_decisions";
    outputDesc.size_bytes = gpu.size() * sizeof(ScatterDecisionGPU);
    outputDesc.usage = RayTrophiSim::ComputeBufferUsage::Storage;
    const auto input = backend->createBuffer(inputDesc);
    const auto output = backend->createBuffer(outputDesc);
    if (!input.valid() || !output.valid()) {
        if (input.valid()) backend->destroyBuffer(input);
        if (output.valid()) backend->destroyBuffer(output);
        return g_lastStats;
    }

    const auto gpuStart = std::chrono::steady_clock::now();
    bool ok = backend->uploadBuffer(input, candidates.data(), inputDesc.size_bytes);
    RayTrophiSim::ComputeBufferHandle buffers[2] = {input, output};
    RayTrophiSim::ComputeDispatch dispatch{};
    dispatch.kernel = "foliage_scatter_parity";
    dispatch.groups.groups_x = (candidateCount + 255u) / 256u;
    dispatch.buffers = buffers; dispatch.buffer_count = 2;
    dispatch.constants = &pc; dispatch.constants_size = sizeof(pc);
    if (ok) ok = backend->dispatch(dispatch);
    if (ok) backend->synchronize();
    if (ok) ok = backend->downloadBuffer(output, gpu.data(), outputDesc.size_bytes);
    g_lastStats.gpuDispatchReadbackMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - gpuStart).count();
    backend->destroyBuffer(input); backend->destroyBuffer(output);
    if (!ok) return g_lastStats;

    for (uint32_t i = 0; i < candidateCount; ++i) {
        if (gpu[i].candidateId != cpu[i].candidateId || gpu[i].reserved != cpu[i].reserved)
            ++g_lastStats.rngMismatches;
        if (gpu[i].accepted != cpu[i].accepted) ++g_lastStats.acceptanceMismatches;
        if (gpu[i].rejectionMask != cpu[i].rejectionMask) ++g_lastStats.rejectionMaskMismatches;
    }
    g_lastStats.completed = true;
    return g_lastStats;
}

const TerrainScatterStats& getLastTerrainScatterStats() { return g_lastTerrainStats; }

int scatterFillTerrainGPU(::InstanceGroup& group, ::TerrainObject* terrain, bool& attempted) {
    attempted = false;
    g_lastTerrainStats = {};
    if (!terrain || terrain->heightmap.width < 3 || terrain->heightmap.height < 3 ||
        terrain->heightmap.data.empty() || group.brush_settings.target_count <= 0) return 0;

    const size_t requiredHeightSamples =
        static_cast<size_t>(terrain->heightmap.width) * terrain->heightmap.height;
    // During project deserialization a terrain can briefly have dimensions but
    // not its complete payload. Never expose that partial range to Vulkan.
    if (terrain->heightmap.data.size() < requiredHeightSamples) return 0;

    std::lock_guard<std::recursive_mutex> lock(RayTrophiSim::sharedMeshComputeMutex());
    auto* backend = RayTrophiSim::acquireSharedMeshComputeBackend();
    if (!backend || !backend->supportsDispatch()) return 0;
    g_lastTerrainStats.gpuAvailable = true;

    const uint32_t width = static_cast<uint32_t>(terrain->heightmap.width);
    const uint32_t height = static_cast<uint32_t>(terrain->heightmap.height);
    const size_t gridCount = static_cast<size_t>(width) * height;
    const uint32_t target = static_cast<uint32_t>(group.brush_settings.target_count);
    const uint64_t requestedCandidates = static_cast<uint64_t>(target) * 100ull;
    const uint32_t candidateCount = static_cast<uint32_t>((std::min<uint64_t>)(
        requestedCandidates, (std::max<uint64_t>)(static_cast<uint64_t>(target) * 4ull, 8000000ull)));
    g_lastTerrainStats.candidateCount = candidateCount;

    std::vector<float> ones(gridCount, 1.0f), minusOnes(gridCount, -1.0f);
    auto resolveField = [&](const std::string& name, const std::vector<float>& fallback)
        -> const std::vector<float>& {
        if (name.empty()) return fallback;
        const auto it = terrain->analysisFields.find(name);
        if (it == terrain->analysisFields.end() || !it->second || it->second->size() != gridCount)
            return fallback;
        return *it->second;
    };
    const auto& density = resolveField(group.brush_settings.density_mask_attribute, ones);
    const auto& exclusion = resolveField(group.brush_settings.exclusion_mask_attribute, minusOnes);
    const auto& scaleFieldValues = resolveField(group.brush_settings.scale_mask_attribute, ones);

    struct SplatPair { float includeValue, excludeValue; };
    std::vector<SplatPair> splat(1, {1.0f, -1.0f});
    const bool hasSplat = terrain->splatMap && terrain->splatMap->is_loaded() &&
        terrain->splatMap->width > 0 && terrain->splatMap->height > 0 &&
        terrain->splatMap->pixels.size() >= static_cast<size_t>(terrain->splatMap->width) *
            terrain->splatMap->height;
    const int includeChannel = group.brush_settings.splat_map_channel;
    const int excludeChannel = group.brush_settings.exclusion_channel;
    auto channelValue = [](const CompactVec4& p, int channel) {
        if (channel == 0) return p.r / 255.0f;
        if (channel == 1) return p.g / 255.0f;
        if (channel == 2) return p.b / 255.0f;
        return p.a / 255.0f;
    };
    if (hasSplat && (includeChannel >= 0 || excludeChannel >= 0)) {
        const int sw = terrain->splatMap->width, sh = terrain->splatMap->height;
        splat.assign(static_cast<size_t>(sw) * sh, {1.0f, -1.0f});
        for (int sy = 0; sy < sh; ++sy) for (int sx = 0; sx < sw; ++sx) {
            const auto& pixel = terrain->splatMap->pixels[static_cast<size_t>(sy) * sw + sx];
            auto& out = splat[static_cast<size_t>(sy) * sw + sx];
            if (includeChannel >= 0 && includeChannel <= 3) out.includeValue = channelValue(pixel, includeChannel);
            if (excludeChannel >= 0 && excludeChannel <= 3) out.excludeValue = channelValue(pixel, excludeChannel);
        }
    }

    ScatterSettingsGPU settings{};
    const auto& bs = group.brush_settings;
    settings.meta[0] = static_cast<uint32_t>(ScatterMode::TerrainFill);
    settings.meta[1] = target; settings.meta[2] = static_cast<uint32_t>(bs.seed); settings.meta[3] = candidateCount;
    settings.terrain[0] = width; settings.terrain[1] = height; settings.terrain[2] = static_cast<uint32_t>(group.sources.size());
    uint32_t flags = 0;
    if (!bs.density_mask_attribute.empty() && &density != &ones) flags |= HasDensityMask;
    if (!bs.exclusion_mask_attribute.empty() && &exclusion != &minusOnes) flags |= HasExclusionMask;
    if (!bs.scale_mask_attribute.empty() && &scaleFieldValues != &ones) flags |= HasScaleMask;
    if (hasSplat && includeChannel >= 0 && includeChannel <= 3) flags |= HasSplatInclude;
    if (hasSplat && excludeChannel >= 0 && excludeChannel <= 3) flags |= HasSplatExclusion;
    if (bs.min_distance > 0.01f) flags |= CheckMinimumDistance;
    if (bs.align_to_normal) flags |= AlignToNormal;
    if (bs.allow_ridges) flags |= AllowRidges;
    if (bs.allow_flats) flags |= AllowFlats;
    if (bs.allow_gullies) flags |= AllowGullies;
    settings.terrain[3] = flags;
    settings.heightSlopeEdge[0] = bs.height_min; settings.heightSlopeEdge[1] = bs.height_max;
    settings.heightSlopeEdge[2] = bs.slope_max;
    const float cellX = terrain->heightmap.scale_xz / static_cast<float>((std::max)(1, terrain->heightmap.width - 1));
    const float cellZ = terrain->heightmap.scale_xz / static_cast<float>((std::max)(1, terrain->heightmap.height - 1));
    const float edgeMeters = bs.edge_margin < 0.0f ? 2.0f * (std::max)(cellX, cellZ) : bs.edge_margin;
    settings.heightSlopeEdge[3] = terrain->heightmap.scale_xz > 1e-6f ?
        std::clamp(edgeMeters / terrain->heightmap.scale_xz, 0.0f, 0.49f) : 0.0f;
    settings.curvature[0] = bs.curvature_min; settings.curvature[1] = bs.curvature_max;
    settings.curvature[2] = static_cast<float>((std::max)(1, bs.curvature_step));
    settings.direction[0] = bs.slope_direction_angle; settings.direction[1] = bs.slope_direction_influence;
    settings.scale[0] = bs.scale_min; settings.scale[1] = bs.scale_max;
    settings.scale[2] = bs.normal_influence; settings.scale[3] = terrain->heightmap.scale_y;
    settings.rotation[0] = bs.rotation_random_y; settings.rotation[1] = bs.rotation_random_xz;
    settings.rotation[2] = cellX; settings.rotation[3] = cellZ;
    settings.offsetMask[0] = bs.y_offset_min; settings.offsetMask[1] = bs.y_offset_max;
    settings.offsetMask[2] = bs.exclusion_threshold; settings.offsetMask[3] = bs.scale_mask_influence;
    settings.maskSlots[2] = hasSplat ? terrain->splatMap->width : 1;
    settings.maskSlots[3] = hasSplat ? terrain->splatMap->height : 1;

    struct Push { uint32_t count, settingsIndex, pad0, pad1; } push{candidateCount,0,0,0};
    std::vector<uint32_t> decisions(candidateCount);
    RayTrophiSim::ComputeBufferDesc desc[7]{};
    const size_t sizes[7] = {sizeof(settings), gridCount*sizeof(float), gridCount*sizeof(float),
        gridCount*sizeof(float), gridCount*sizeof(float), splat.size()*sizeof(SplatPair),
        decisions.size()*sizeof(uint32_t)};
    const char* names[7] = {"foliage_settings","foliage_height","foliage_density","foliage_exclusion",
        "foliage_scale","foliage_splat","foliage_decisions"};
    RayTrophiSim::ComputeBufferHandle buffers[7]{};
    bool ok = true;
    for (int i=0;i<7;++i) { desc[i].debug_name=names[i];desc[i].size_bytes=sizes[i];desc[i].usage=RayTrophiSim::ComputeBufferUsage::Storage;buffers[i]=backend->createBuffer(desc[i]);ok=ok&&buffers[i].valid(); }
    auto cleanup=[&](){for(auto h:buffers)if(h.valid())backend->destroyBuffer(h);};
    if(!ok){cleanup();return 0;}
    const auto gpuStart=std::chrono::steady_clock::now();
    backend->beginTransferBatch();
    ok=backend->uploadBuffer(buffers[0],&settings,sizeof(settings));
    ok=ok&&backend->uploadBuffer(buffers[1],terrain->heightmap.data.data(),sizes[1]);
    ok=ok&&backend->uploadBuffer(buffers[2],density.data(),sizes[2]);
    ok=ok&&backend->uploadBuffer(buffers[3],exclusion.data(),sizes[3]);
    ok=ok&&backend->uploadBuffer(buffers[4],scaleFieldValues.data(),sizes[4]);
    ok=ok&&backend->uploadBuffer(buffers[5],splat.data(),sizes[5]);
    ok=backend->endTransferBatch()&&ok;
    RayTrophiSim::ComputeDispatch dispatch{};dispatch.kernel="foliage_scatter_terrain_accept";
    dispatch.groups.groups_x=(candidateCount+255u)/256u;dispatch.buffers=buffers;dispatch.buffer_count=7;
    dispatch.constants=&push;dispatch.constants_size=sizeof(push);
    if(ok)ok=backend->dispatch(dispatch);if(ok)backend->synchronize();
    if(ok)ok=backend->downloadBuffer(buffers[6],decisions.data(),sizes[6]);
    g_lastTerrainStats.gpuMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-gpuStart).count();
    g_lastTerrainStats.uploadBytes=sizes[0]+sizes[1]+sizes[2]+sizes[3]+sizes[4]+sizes[5];cleanup();
    if(!ok)return 0;

    const auto compactStart=std::chrono::steady_clock::now();
    std::unordered_map<uint64_t,std::vector<Vec3>> occupied;
    auto cellKey=[](int x,int z){return(static_cast<uint64_t>(static_cast<uint32_t>(x))<<32u)|static_cast<uint32_t>(z);};
    if(bs.min_distance>0.01f)occupied.reserve(target);
    const float minDistSq=bs.min_distance*bs.min_distance;
    Matrix4x4 finalMatrix,normalMatrix;const bool transformed=terrain->transform!=nullptr;
    if(transformed){terrain->transform->updateFinal();finalMatrix=terrain->transform->final;normalMatrix=terrain->transform->getNormalTransform();}
    std::vector<InstanceTransform> generated;generated.reserve(target);
    for(uint32_t id=0;id<candidateCount&&generated.size()<target;++id){if((decisions[id]&0x80000000u)==0u)continue;++g_lastTerrainStats.acceptedBeforeSpacing;
        const float u=candidateRandom01(static_cast<uint32_t>(bs.seed),id,2u),v=candidateRandom01(static_cast<uint32_t>(bs.seed),id,3u);
        const float gx=u*(width-1),gz=v*(height-1);const int x0=static_cast<int>(gx),z0=static_cast<int>(gz);const int x1=(std::min)(x0+1,int(width)-1),z1=(std::min)(z0+1,int(height)-1);const float fx=gx-x0,fz=gz-z0;
        const auto& hm=terrain->heightmap;const float h00=hm.getHeight(x0,z0),h10=hm.getHeight(x1,z0),h01=hm.getHeight(x0,z1),h11=hm.getHeight(x1,z1);const float localH=(h00*(1-fx)+h10*fx)*(1-fz)+(h01*(1-fx)+h11*fx)*fz;
        const int step=(std::max)(1,bs.curvature_step);const int sx=std::clamp(static_cast<int>(gx+0.5f),step,int(width)-1-step),sz=std::clamp(static_cast<int>(gz+0.5f),step,int(height)-1-step);
        const float hl=hm.data[static_cast<size_t>(sz)*width+sx-step],hr=hm.data[static_cast<size_t>(sz)*width+sx+step],hu=hm.data[static_cast<size_t>(sz-step)*width+sx],hd=hm.data[static_cast<size_t>(sz+step)*width+sx];const float dx=((hr-hl)*hm.scale_y)/(2*cellX*step),dz=((hd-hu)*hm.scale_y)/(2*cellZ*step);
        Vec3 pos(u*hm.scale_xz,localH,v*hm.scale_xz),normal=Vec3(-dx,1,-dz).normalize();if(transformed){pos=finalMatrix.transform_point(pos);normal=normalMatrix.transform_vector(normal).normalize();}
        if(bs.min_distance>0.01f){const int cx=static_cast<int>(std::floor(pos.x/bs.min_distance)),cz=static_cast<int>(std::floor(pos.z/bs.min_distance));bool collision=false;for(int ox=-1;ox<=1&&!collision;++ox)for(int oz=-1;oz<=1&&!collision;++oz){auto it=occupied.find(cellKey(cx+ox,cz+oz));if(it!=occupied.end())for(const auto& p:it->second)if((p-pos).length_squared()<minDistSq){collision=true;break;}}if(collision)continue;occupied[cellKey(cx,cz)].push_back(pos);}
        InstanceTransform inst=group.generateRandomTransform(pos,normal);if((flags&HasScaleMask)&&bs.scale_mask_influence>0){const float s00=scaleFieldValues[static_cast<size_t>(z0)*width+x0],s10=scaleFieldValues[static_cast<size_t>(z0)*width+x1],s01=scaleFieldValues[static_cast<size_t>(z1)*width+x0],s11=scaleFieldValues[static_cast<size_t>(z1)*width+x1];const float sf=(s00*(1-fx)+s10*fx)*(1-fz)+(s01*(1-fx)+s11*fx)*fz;inst.scale=inst.scale*(1.0f-bs.scale_mask_influence*(1.0f-std::clamp(sf,0.0f,1.0f)));}generated.push_back(inst);
    }
    group.addInstances(generated);g_lastTerrainStats.spawned=static_cast<uint32_t>(generated.size());g_lastTerrainStats.cpuCompactMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-compactStart).count();g_lastTerrainStats.gpuPathUsed=true;
    attempted = true;
    return static_cast<int>(generated.size());
}

} // namespace FoliageGPU
