#include "OptixWrapper.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <chrono> // Süre ölçmek için
#include <unordered_map> // Gereken başlık
#include <algorithm>    // std::min ve std::max için
#include <cstring>      // memcpy for camera hash

#include <SpotLight.h>
#include <filesystem>
#include <imgui.h>
#include <imgui_impl_sdlrenderer2.h>
#include "Triangle.h"  

#define OPTIX_CHECK(call)                                                       \
    do {                                                                        \
        OptixResult res = call;                                                 \
        if (res != OPTIX_SUCCESS) {                                             \
            const char* name = nullptr;                                         \
            const char* desc = nullptr;                                         \
            /* OptiX 7/8/9 gibi sürümlerde genelde bu helper'lar mevcut */      \
            /* Bazı SDK'larda dönüş tipi/fonksiyon farklı olabilir, header'a bak */ \
            name = optixGetErrorName ? optixGetErrorName(res) : nullptr;        \
            desc = optixGetErrorString ? optixGetErrorString(res) : nullptr;    \
            SCENE_LOG_ERROR(                                                    \
                std::string("OptiX error ") +                                   \
                (name ? std::string(name) : std::string("UNKNOWN")) +           \
                " (" + std::to_string(static_cast<int>(res)) + ")" +            \
                " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) \
            );                                                                  \
            if (desc) SCENE_LOG_ERROR(std::string("[OptiX msg] ") + desc);      \
            std::terminate();                                                   \
        }                                                                       \
    } while (0)

// Dosyanın başına ekleyin (OptiX için OPTIX_CHECK zaten var, CUDA için eklenmeli)
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            SCENE_LOG_ERROR(                                                    \
                std::string("CUDA error: ") + cudaGetErrorString(err) +         \
                " (" + std::to_string(static_cast<int>(err)) + ")" +            \
                " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) \
            );                                                                  \
            std::terminate();                                                   \
        }                                                                       \
    } while (0)

OptixWrapper::OptixWrapper()
    : Image_width(image_width), Image_height(image_height), color_processor(image_width, image_height) //  işte burada!
{
    d_vertices = 0;
    d_indices = 0;
    d_bvh_output = 0;
    d_accumulation_buffer = nullptr;
    is_gas_built_as_soup = false;
    allocated_vertex_byte_size = 0;
    allocated_normal_byte_size = 0;
    last_vertex_count = 0;
    d_variance_buffer = nullptr;
    d_sample_count_buffer = nullptr;
    sbt.raygenRecord = 0;
    sbt.missRecordBase = 0;
    sbt.hitgroupRecordBase = 0;
    traversable_handle = 0;
   // initialize();
}
void OptixWrapper::partialCleanup() {
    // Sadece buildFromData'da oluşturulan kaynakları temizle
    // stream ve context gibi önemli yapılar korunur

    if (d_vertices) {
        cudaFree(reinterpret_cast<void*>(d_vertices));
        d_vertices = 0;
    }
    if (d_indices) {
        cudaFree(reinterpret_cast<void*>(d_indices));
        d_indices = 0;
    }
    if (d_normals) {
        cudaFree(reinterpret_cast<void*>(d_normals));
        d_normals = 0;
    }
    if (d_uvs) {
        cudaFree(reinterpret_cast<void*>(d_uvs));
        d_uvs = 0;
    }
    if (d_tangents) {
        cudaFree(reinterpret_cast<void*>(d_tangents));
        d_tangents = 0;
    }
    if (d_material_indices) {
        cudaFree(reinterpret_cast<void*>(d_material_indices));
        d_material_indices = 0;
    }
    if (d_bvh_output) {
        cudaFree(reinterpret_cast<void*>(d_bvh_output));
        d_bvh_output = 0;
    }
    if (d_temp_buffer) {
        cudaFree(reinterpret_cast<void*>(d_temp_buffer));
        d_temp_buffer = 0;
    }
    if (d_output_buffer) {
        cudaFree(reinterpret_cast<void*>(d_output_buffer));
        d_output_buffer = 0;
    }
    if (d_compacted_size) {
        cudaFree(reinterpret_cast<void*>(d_compacted_size));
        d_compacted_size = 0;
    }
    if (d_params) {
        cudaFree(reinterpret_cast<void*>(d_params));
        d_params = 0;
    }
    if (d_coords_x) {
        cudaFree(reinterpret_cast<void*>(d_coords_x));
        d_coords_x = 0;
    }
    if (d_coords_y) {
        cudaFree(reinterpret_cast<void*>(d_coords_y));
        d_coords_y = 0;
    }
    if (d_materials) {
        cudaFree(reinterpret_cast<void*>(d_materials));
        d_materials = nullptr;
    }
    
    if (d_accumulation_buffer) {
        cudaFree(reinterpret_cast<void*>(d_accumulation_buffer));
        d_accumulation_buffer = nullptr;
    }
    if (d_variance_buffer) {
        cudaFree(reinterpret_cast<void*>(d_variance_buffer));
        d_variance_buffer = nullptr;
    }
    if (d_sample_count_buffer) {
        cudaFree(reinterpret_cast<void*>(d_sample_count_buffer));
        d_sample_count_buffer = nullptr;
    }
    if (d_framebuffer) {
        cudaFree(reinterpret_cast<void*>(d_framebuffer));
        d_framebuffer = nullptr;
    }
    if (d_accumulation_float4) {
        cudaFree(reinterpret_cast<void*>(d_accumulation_float4));
        d_accumulation_float4 = nullptr;
        accumulation_valid = false;
        accumulated_samples = 0;
    }

    // SBT kaynakları
    if (sbt.hitgroupRecordBase) {
        cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase));
        sbt.hitgroupRecordBase = 0;
        sbt.hitgroupRecordCount = 0;
    }

    traversable_handle = 0;

    // Senkronizasyon: zorunlu değil ama debug için güvenli
    cudaDeviceSynchronize();
}

void OptixWrapper::clearScene() {
    traversable_handle = 0;
    params.handle = 0;
    
    if (d_bvh_output) {
        cudaFree(reinterpret_cast<void*>(d_bvh_output)); 
        d_bvh_output = 0;
    }
    if (d_vertices) { cudaFree(reinterpret_cast<void*>(d_vertices)); d_vertices = 0; }
    if (d_indices) { cudaFree(reinterpret_cast<void*>(d_indices)); d_indices = 0; }
    
    if (d_params) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(RayGenParams), cudaMemcpyHostToDevice));
    }
    SCENE_LOG_INFO("[OptiX] Scene cleared (Handle=0)");
}

void OptixWrapper::cleanup() {
    if (d_vertices) {
        cudaFree(reinterpret_cast<void*>(d_vertices));
        d_vertices = 0;
    }

    if (d_indices) {
        cudaFree(reinterpret_cast<void*>(d_indices));
        d_indices = 0;
    }

    if (d_bvh_output) {
        cudaFree(reinterpret_cast<void*>(d_bvh_output));
        d_bvh_output = 0;
    }

    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }

    if (context) {
        optixDeviceContextDestroy(context);
        context = nullptr;
    }
    if (d_accumulation_buffer) cudaFree(d_accumulation_buffer);
    if (d_variance_buffer) cudaFree(d_variance_buffer);
    if (d_sample_count_buffer) cudaFree(d_sample_count_buffer);
    if (d_framebuffer) cudaFree(d_framebuffer);
    if (d_accumulation_float4) cudaFree(d_accumulation_float4);
    d_accumulation_float4 = nullptr;
    traversable_handle = 0;
}

OptixWrapper::~OptixWrapper() {
    cleanup();
}



void OptixWrapper::initialize() {
    if (context != nullptr) {
      
        return;
    }

    cudaFree(0); // CUDA başlat

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = nullptr;
    options.logCallbackLevel = 4;

    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));

    cudaStreamCreate(&stream);

    //std::cout << "OptiX Successfully initialized.\n";
}


void updatePixeloptix(SDL_Surface* surface, int i, int j, const Vec3SIMD& color) {
    Uint32* pixel = static_cast<Uint32*>(surface->pixels) + (surface->h - 1 - j) * surface->pitch / 4 + i;

    // Linear to sRGB dönüşüm (basit approx veya doğru dönüşüm kullanabilirsin)
    auto toSRGB = [](float c) {
        if (c <= 0.0031308f)
            return 12.92f * c;
        else
            return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
        };

    int r = static_cast<int>(255 * std::clamp(toSRGB(color.x()), 0.0f, 1.0f));
    int g = static_cast<int>(255 * std::clamp(toSRGB(color.y()), 0.0f, 1.0f));
    int b = static_cast<int>(255 * std::clamp(toSRGB(color.z()), 0.0f, 1.0f));

    *pixel = SDL_MapRGB(surface->format, r, g, b);
}
bool OptixWrapper::isCudaAvailable() {
    try {
        oidn::DeviceRef testDevice = oidn::newDevice(oidn::DeviceType::CUDA);
        testDevice.commit();
        return true; // CUDA destekleniyor
    }
    catch (const std::exception& e) {
        return false; // CUDA desteklenmiyor
    }
}

// OIDN methods removed - all denoising now handled by Renderer::applyOIDNDenoising
// This eliminates code duplication and ensures consistent behavior



void OptixWrapper::validateMaterialIndices(const OptixGeometryData& data) {
    if (data.materials.empty()) {
        SCENE_LOG_ERROR("No material available!");
        return;
    }

    if (data.indices.empty()) {
        SCENE_LOG_ERROR("There are no triangles!");
        return;
    }

    const auto& material_indices = data.material_indices;

    if (material_indices.empty()) {
        SCENE_LOG_INFO("Material indices are empty, default material (0) will be used for all triangles.");
        return;
    }

    for (size_t tri_idx = 0; tri_idx < material_indices.size(); ++tri_idx) {
        int mat_idx = material_indices[tri_idx];

        if (mat_idx < 0 || mat_idx >= (int)data.materials.size()) {
            SCENE_LOG_WARN(
                "Invalid material index for triangle [" +
                std::to_string(tri_idx) + "]: " +
                std::to_string(mat_idx)
            );
        }
    }

    SCENE_LOG_INFO(
        "Material indices verified (" +
        std::to_string(material_indices.size()) +
        " triangles checked)"
    );
}


void OptixWrapper::setupPipeline(const char* raygen_ptx) {
    // 1. Compile options
    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 2;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "optixLaunchParams";

    char log[2048];
    size_t log_size = sizeof(log);

    // 2. PTX modül oluştur (OptiX 8 için güncellendi)
    // OptiX 8'de NVRTC ile derlenmiş PTX modülü kullanmak
    OPTIX_CHECK(optixModuleCreate(
        context,
        &module_options,
        &pipeline_options,
        raygen_ptx,
        strlen(raygen_ptx),
        log,
        &log_size,
        &module
    ));

    // 3. Program group: raygen
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module;
    raygen_desc.raygen.entryFunctionName = "__raygen__rg";

    OptixProgramGroupOptions pg_options = {};
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        context, &raygen_desc, 1, &pg_options, log, &log_size, &raygen_pg));

    // 4. Program group: miss
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module;
    miss_desc.miss.entryFunctionName = "__miss__ms";

    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        context, &miss_desc, 1, &pg_options, log, &log_size, &miss_pg));

    // 5. Program group: hit
    OptixProgramGroupDesc hit_desc = {};
    hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_desc.hitgroup.moduleCH = module;
    hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        context, &hit_desc, 1, &pg_options, log, &log_size, &hit_pg));

    // 6. Pipeline oluştur
    OptixProgramGroup program_groups[] = { raygen_pg, miss_pg, hit_pg };
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;

    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        context, &pipeline_options, &link_options,
        program_groups, 3,
        log, &log_size,
        &pipeline
    ));

    // 7. Shader Binding Table (SBT) hazırla
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptySbtRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    EmptySbtRecord raygen_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &raygen_record));
    cudaMalloc(reinterpret_cast<void**>(&sbt.raygenRecord), sizeof(EmptySbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(sbt.raygenRecord), &raygen_record, sizeof(EmptySbtRecord), cudaMemcpyHostToDevice);

    EmptySbtRecord miss_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &miss_record));
    cudaMalloc(reinterpret_cast<void**>(&sbt.missRecordBase), sizeof(EmptySbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(sbt.missRecordBase), &miss_record, sizeof(EmptySbtRecord), cudaMemcpyHostToDevice);
    sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.missRecordCount = 1;

    EmptySbtRecord hit_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hit_pg, &hit_record));
    cudaMalloc(reinterpret_cast<void**>(&sbt.hitgroupRecordBase), sizeof(EmptySbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(sbt.hitgroupRecordBase), &hit_record, sizeof(EmptySbtRecord), cudaMemcpyHostToDevice);
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.hitgroupRecordCount = 1;
}
void OptixWrapper::destroyTextureObjects() {
    int texture_obj_count = 0;
    int array_count = 0;
    
    // 1. Texture Object'leri yok et
    for (const auto& record : hitgroup_records) {
        const HitGroupData& data = record.data;

        if (data.albedo_tex) { 
            cudaDestroyTextureObject(data.albedo_tex); 
            texture_obj_count++;
        }
        if (data.roughness_tex) { 
            cudaDestroyTextureObject(data.roughness_tex); 
            texture_obj_count++;
        }
        if (data.normal_tex) { 
            cudaDestroyTextureObject(data.normal_tex); 
            texture_obj_count++;
        }
        if (data.metallic_tex) { 
            cudaDestroyTextureObject(data.metallic_tex); 
            texture_obj_count++;
        }
        if (data.transmission_tex) { 
            cudaDestroyTextureObject(data.transmission_tex); 
            texture_obj_count++;
        }
        if (data.opacity_tex) { 
            cudaDestroyTextureObject(data.opacity_tex); 
            texture_obj_count++;
        }
        if (data.emission_tex) { 
            cudaDestroyTextureObject(data.emission_tex); 
            texture_obj_count++;
        }
    }

    // 2. CUDA Array'leri serbest bırak (CRITICAL FIX!)
    for (auto& array : texture_arrays) {
        if (array) {
            cudaError_t err = cudaFreeArray(array);
            if (err != cudaSuccess) {
                SCENE_LOG_WARN("[GPU CLEANUP] cudaFreeArray failed: " + std::string(cudaGetErrorString(err)));
            }
            else {
                array_count++;
            }
            array = nullptr;
        }
    }
    texture_arrays.clear();

    // 3. SBT records'ları temizle
    hitgroup_records.clear();
    
    SCENE_LOG_INFO("[GPU CLEANUP] Destroyed " + std::to_string(texture_obj_count) + 
                   " texture objects and " + std::to_string(array_count) + " CUDA arrays.");
}

void OptixWrapper::buildFromData(const OptixGeometryData& data) {
    SCENE_LOG_INFO(" OptiX buildFromData is starting...");

    if (data.vertices.empty() || data.indices.empty()) {
        SCENE_LOG_ERROR(" Geometry data is empty!");
        return;
    }
    
    // CRITICAL FIX: Do NOT destroy texture objects here!
    // Texture handles stored in Triangle::textureBundle must remain valid.
    // destroyTextureObjects() should only be called when completely clearing the scene
    // (in create_scene with append=false), not during rebuilds.
    // The old code was destroying textures here, which invalidated the handles
    // stored in triangles, causing the second object to use garbage texture pointers.
    
    // destroyTextureObjects();  // REMOVED - causes texture corruption on rebuild!
    
    // Only clear the hitgroup records (SBT data), not the actual texture memory
    hitgroup_records.clear();
    
    partialCleanup();            // << Ardından tüm CUDA buffer'larını temizle  

    // 1. Tüm geometri verilerini GPU'ya gönder
    size_t v_size = data.vertices.size() * sizeof(float3);
    cudaMalloc(reinterpret_cast<void**>(&d_vertices), v_size);
    cudaMemcpy(reinterpret_cast<void*>(d_vertices), data.vertices.data(), v_size, cudaMemcpyHostToDevice);
    allocated_vertex_byte_size = v_size;
    is_gas_built_as_soup = false; // Initial build is Indexed from Assimp

    size_t i_size = data.indices.size() * sizeof(uint3);
    cudaMalloc(reinterpret_cast<void**>(&d_indices), i_size);
    cudaMemcpy(reinterpret_cast<void*>(d_indices), data.indices.data(), i_size, cudaMemcpyHostToDevice);

    d_normals = 0;
    if (!data.normals.empty()) {
        size_t n_size = data.normals.size() * sizeof(float3);
        cudaMalloc(reinterpret_cast<void**>(&d_normals), n_size);
        cudaMemcpy(reinterpret_cast<void*>(d_normals), data.normals.data(), n_size, cudaMemcpyHostToDevice);
        allocated_normal_byte_size = n_size;
    }

    d_uvs = 0;
    if (!data.uvs.empty()) {
        size_t uv_size = data.uvs.size() * sizeof(float2);
        cudaMalloc(reinterpret_cast<void**>(&d_uvs), uv_size);
        cudaMemcpy(reinterpret_cast<void*>(d_uvs), data.uvs.data(), uv_size, cudaMemcpyHostToDevice);
    }

    // Materyal indekslerini GPU'ya gönder
    std::vector<int> default_material_indices;
    const int* material_indices_ptr = nullptr;
    if (data.material_indices.empty() || data.material_indices.size() != data.indices.size()) {
        default_material_indices.resize(data.indices.size(), 0);
        material_indices_ptr = default_material_indices.data();
        SCENE_LOG_WARN("Warning: Material indexes have been rebuilt.");
    }
    else {
        material_indices_ptr = data.material_indices.data();
    }

   
    size_t mi_size = data.indices.size() * sizeof(int);
    cudaMalloc(reinterpret_cast<void**>(&d_material_indices), mi_size);
    cudaMemcpy(reinterpret_cast<void*>(d_material_indices), material_indices_ptr, mi_size, cudaMemcpyHostToDevice);

    // Tangent verilerini GPU'ya gönder
     d_tangents = 0;
    if (!data.tangents.empty()) {
        size_t tangent_size = data.tangents.size() * sizeof(float3);
        cudaMalloc(reinterpret_cast<void**>(&d_tangents), tangent_size);
        cudaMemcpy(reinterpret_cast<void*>(d_tangents), data.tangents.data(), tangent_size, cudaMemcpyHostToDevice);
    }

    // GpuMaterial verilerini GPU'ya gönder (YENİ!!)
    if (!data.materials.empty()) {
        size_t mat_size = data.materials.size() * sizeof(GpuMaterial);
        cudaMalloc(reinterpret_cast<void**>(&d_materials), mat_size);
        cudaMemcpy(reinterpret_cast<void*>(d_materials), data.materials.data(), mat_size, cudaMemcpyHostToDevice);
    }

    // 2. Her materyal için bir SBT kaydı oluştur
    int ray_type_count = 1; // primary, shadow
   
   
    for (int ray_type = 0; ray_type < ray_type_count; ++ray_type) {
        for (size_t mat_index = 0; mat_index < data.materials.size(); ++mat_index) {
            SbtRecord<HitGroupData> rec = {};

            if (mat_index < data.textures.size()) {
                const auto& tex = data.textures[mat_index];
                rec.data.albedo_tex = tex.albedo_tex;
                rec.data.roughness_tex = tex.roughness_tex;
                rec.data.normal_tex = tex.normal_tex;
                rec.data.metallic_tex = tex.metallic_tex;
                rec.data.transmission_tex = tex.transmission_tex;
                rec.data.opacity_tex = tex.opacity_tex;
                rec.data.emission_tex = tex.emission_tex;

                rec.data.has_emission_tex = tex.has_emission_tex;
                rec.data.has_albedo_tex = tex.has_albedo_tex;
                rec.data.has_roughness_tex = tex.has_roughness_tex;
                rec.data.has_normal_tex = tex.has_normal_tex;
                rec.data.has_metallic_tex = tex.has_metallic_tex;
                rec.data.has_transmission_tex = tex.has_transmission_tex;
                rec.data.has_opacity_tex = tex.has_opacity_tex;
               
            }
            else {
                SCENE_LOG_WARN("[SBT-WARN] Mat #" + std::to_string(mat_index) + " has no texture!");
            }
            rec.data.material_id = static_cast<int>(mat_index);
            rec.data.vertices = reinterpret_cast<float3*>(d_vertices);
            rec.data.indices = reinterpret_cast<uint3*>(d_indices);
            rec.data.normals = reinterpret_cast<float3*>(d_normals);
            rec.data.uvs = reinterpret_cast<float2*>(d_uvs);
            rec.data.has_normals = !data.normals.empty();
            rec.data.has_uvs = !data.uvs.empty();
            rec.data.tangents = reinterpret_cast<float3*>(d_tangents);
            rec.data.has_tangents = !data.tangents.empty();
            rec.data.emission = data.materials[mat_index].emission;

            OPTIX_CHECK(optixSbtRecordPackHeader(hit_pg, &rec));
            hitgroup_records.push_back(rec);
        }
    }


    CUdeviceptr d_hitgroup_records;
    size_t sbt_size = hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>);
    cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), sbt_size);
    cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records), hitgroup_records.data(), sbt_size, cudaMemcpyHostToDevice);

    sbt.hitgroupRecordBase = d_hitgroup_records;
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitGroupData>);
    sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());

    // 4. Build input
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = &d_vertices;
    build_input.triangleArray.numVertices = static_cast<uint32_t>(data.vertices.size());
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    build_input.triangleArray.indexBuffer = d_indices;
    build_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(data.indices.size());
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.sbtIndexOffsetBuffer = d_material_indices;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(int);
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(int);

    std::vector<unsigned int> triangle_flags(data.materials.size(), OPTIX_GEOMETRY_FLAG_NONE);
    build_input.triangleArray.flags = triangle_flags.data();
    build_input.triangleArray.numSbtRecords = static_cast<uint32_t>(data.materials.size());

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1, &buffer_sizes));

   
    cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), buffer_sizes.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_output_buffer), buffer_sizes.outputSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_compacted_size), sizeof(uint64_t));

    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = d_compacted_size;

    OPTIX_CHECK(optixAccelBuild(
        context, stream, &accel_options, &build_input, 1,
        d_temp_buffer, buffer_sizes.tempSizeInBytes,
        d_output_buffer, buffer_sizes.outputSizeInBytes,
        &traversable_handle, &emit_property, 1
    ));

    uint64_t compacted_size;
    cudaMemcpy(&compacted_size, reinterpret_cast<void*>(d_compacted_size), sizeof(uint64_t), cudaMemcpyDeviceToHost);

    if (compacted_size < buffer_sizes.outputSizeInBytes) {
        CUdeviceptr d_compacted_buffer;
        cudaMalloc(reinterpret_cast<void**>(&d_compacted_buffer), compacted_size);
        OPTIX_CHECK(optixAccelCompact(
            context, stream, traversable_handle,
            d_compacted_buffer, compacted_size,
            &traversable_handle
        ));
        cudaFree(reinterpret_cast<void*>(d_output_buffer));
        d_output_buffer = d_compacted_buffer;
    }


    // Temizlik
    cudaFree(reinterpret_cast<void*>(d_temp_buffer));
    d_temp_buffer = 0;
    cudaFree(reinterpret_cast<void*>(d_compacted_size));
    d_compacted_size = 0;
    d_bvh_output = d_output_buffer;

    cudaStreamSynchronize(stream);
    SCENE_LOG_INFO(
        "OptiX buildFromData completed successfully! " +
        std::to_string(data.materials.size()) +
        " SBT record(s) created for material."
    );
}
/*
void OptixWrapper::launch_tile_based_progressive(
    SDL_Surface* surface, SDL_Window* window, int width, int height,
    std::vector<uchar4>& framebuffer, SDL_Texture* raytrace_texture
) {
    cudaGetDeviceProperties(&props, 0);

    constexpr int TILE_SIZE = 256;
    constexpr int MAX_PASSES = 4;

    // Buffers
    uchar4* d_framebuffer = nullptr;
    float4* d_accumulation = nullptr;

    cudaMalloc(&d_framebuffer, width * height * sizeof(uchar4));
    cudaMemset(d_framebuffer, 0, width * height * sizeof(uchar4));

    cudaMalloc(&d_accumulation, width * height * sizeof(float4));
    cudaMemset(d_accumulation, 0, width * height * sizeof(float4));

    // Streams
    cudaStream_t render_stream, copy_stream;
    cudaStreamCreate(&render_stream);
    cudaStreamCreate(&copy_stream);

    // Host buffers
    std::vector<uchar4> tile_buffer(TILE_SIZE * TILE_SIZE);
    framebuffer.resize(width * height, make_uchar4(0, 0, 0, 255));

    Uint32* pixels = (Uint32*)surface->pixels;
    int row_stride = surface->pitch / 4;

    // Tile generation
    std::vector<Tile> tiles;
    for (int y = 0; y < height; y += TILE_SIZE) {
        for (int x = 0; x < width; x += TILE_SIZE) {
            Tile tile;
            tile.x = x;
            tile.y = y;
            tile.width = std::min(TILE_SIZE, width - x);
            tile.height = std::min(TILE_SIZE, height - y);
            tile.samples = 0;
            tile.variance = 1.0f;
            tile.completed = false;
            tiles.push_back(tile);
        }
    }

    static std::mt19937 rng(12345);
    std::shuffle(tiles.begin(), tiles.end(), rng);

    std::cout << "Tiles: " << tiles.size() << " (" << TILE_SIZE << "x" << TILE_SIZE << ")\n";

    // Params setup (sabit değerler)
    params.image_width = width;
    params.image_height = height;
    params.handle = traversable_handle;
    params.materials = reinterpret_cast<GpuMaterial*>(d_materials);
    params.max_depth = render_settings.max_bounces;
    params.use_adaptive_sampling = render_settings.use_adaptive_sampling;
    params.variance_threshold = render_settings.variance_threshold;
    params.framebuffer = d_framebuffer;
    params.accumulation_buffer = d_accumulation;

    // Atmosphere
    params.atmosphere.sigma_s = 0.01f;
    params.atmosphere.sigma_a = 0.01f;
    params.atmosphere.g = 0.0f;
    params.atmosphere.base_density = 0.001f;
    params.atmosphere.temperature = 300.0f;
    params.atmosphere.active = false;

    RayGenParams* d_params = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(RayGenParams));

    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_counter = 0;

    // Multi-pass progressive loop
    for (int pass = 0; pass < MAX_PASSES; pass++) {
        int samples_this_pass = 1 << pass;

        std::cout << "\n=== Pass " << (pass + 1) << "/" << MAX_PASSES
            << " | Samples: " << samples_this_pass << " ===\n";

        if (pass > 0 && render_settings.use_adaptive_sampling) {
            std::sort(tiles.begin(), tiles.end(),
                [](const Tile& a, const Tile& b) { return a.variance > b.variance; });
        }

        int tiles_rendered = 0;
        int tiles_skipped = 0;

        for (auto& tile : tiles) {
            if (tile.completed && tile.variance < render_settings.variance_threshold) {
                tiles_skipped++;
                continue;
            }

            // Params güncelle
            params.tile_x = tile.x;
            params.tile_y = tile.y;
            params.tile_width = tile.width;
            params.tile_height = tile.height;
            params.samples_per_pixel = samples_this_pass;
            params.current_pass = pass;
            params.frame_number = frame_counter++;

            // Sync copy sonra launch (önceki işlemin bitmesini bekle)
            cudaStreamSynchronize(render_stream);

            cudaMemcpyAsync(d_params, &params, sizeof(RayGenParams),
                cudaMemcpyHostToDevice, render_stream);

            OptixResult result = optixLaunch(
                pipeline,
                render_stream,
                reinterpret_cast<CUdeviceptr>(d_params),
                sizeof(RayGenParams),
                &sbt,
                tile.width,   // Launch width = tile width
                tile.height,  // Launch height = tile height
                1             // Depth = 1
            );

            if (result != OPTIX_SUCCESS) {
                std::cerr << "OptiX launch failed: " << result << std::endl;
                std::cerr << "Tile: " << tile.x << "," << tile.y
                    << " Size: " << tile.width << "x" << tile.height << std::endl;
                continue;
            }

            // Tile'ı kopyala
            size_t src_offset = tile.y * width + tile.x;
            uchar4* src_ptr = d_framebuffer + src_offset;

            cudaMemcpy2DAsync(
                tile_buffer.data(),
                tile.width * sizeof(uchar4),
                src_ptr,
                width * sizeof(uchar4),
                tile.width * sizeof(uchar4),
                tile.height,
                cudaMemcpyDeviceToHost,
                copy_stream
            );

            cudaStreamSynchronize(copy_stream);

            // Host framebuffer güncelle
            for (int ty = 0; ty < tile.height; ty++) {
                int dst_y = tile.y + ty;
                int src_idx = ty * tile.width;
                int dst_idx = dst_y * width + tile.x;
                std::memcpy(&framebuffer[dst_idx], &tile_buffer[src_idx],
                    tile.width * sizeof(uchar4));
            }

            // Display güncelle
            for (int ty = 0; ty < tile.height; ty++) {
                for (int tx = 0; tx < tile.width; tx++) {
                    int px = tile.x + tx;
                    int py = tile.y + ty;
                    int fb_index = py * width + px;

                    const uchar4& c = framebuffer[fb_index];
                    Vec3 raw_color(c.x / 255.0f, c.y / 255.0f, c.z / 255.0f);
                    Vec3 final_color = color_processor.processColor(raw_color, px, py);

                    Uint8 r = static_cast<Uint8>(std::min(255.0f * final_color.x, 255.0f));
                    Uint8 g = static_cast<Uint8>(std::min(255.0f * final_color.y, 255.0f));
                    Uint8 b = static_cast<Uint8>(std::min(255.0f * final_color.z, 255.0f));

                    int screen_index = (height - 1 - py) * row_stride + px;
                    pixels[screen_index] = SDL_MapRGB(surface->format, r, g, b);
                }
            }

            tiles_rendered++;
            tile.samples += samples_this_pass;

            if (tiles_rendered % 16 == 0 || tiles_rendered == static_cast<int>(tiles.size() - tiles_skipped)) {
                SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
                SDL_UpdateWindowSurface(window);

                float progress = 100.0f * tiles_rendered / (tiles.size() - tiles_skipped);
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

                std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                    << progress << "% | "
                    << tiles_rendered << "/" << (tiles.size() - tiles_skipped)
                    << " tiles | " << elapsed << "s     " << std::flush;
            }
        }

        std::cout << "\n";

        // Variance calculation...
        // (mevcut kodun aynısı)
    }

    // Final update
    SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
    SDL_UpdateWindowSurface(window);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "\n✓ Render completed in " << duration << "s\n";

    // Cleanup
    cudaFree(d_framebuffer);
    cudaFree(d_accumulation);
    cudaFree(d_params);
    cudaStreamDestroy(render_stream);
    cudaStreamDestroy(copy_stream);
}
*/

// ------------------------------------------------------------------
// CYCLES-STYLE ACCUMULATIVE RENDERING - Progressive refinement
// Camera stationary: accumulates samples up to max_samples
// Camera moves: resets and renders with 1 sample for fast preview
// ------------------------------------------------------------------

// Helper function to compute camera hash for change detection
uint64_t OptixWrapper::computeCameraHash() const {
    // Simple FNV-1a style hash of camera parameters
    uint64_t hash = 14695981039346656037ULL;
    auto hashFloat = [&hash](float f) {
        uint32_t bits;
        memcpy(&bits, &f, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    };
    
    hashFloat(params.camera.origin.x);
    hashFloat(params.camera.origin.y);
    hashFloat(params.camera.origin.z);
    hashFloat(params.camera.lower_left_corner.x);
    hashFloat(params.camera.lower_left_corner.y);
    hashFloat(params.camera.lower_left_corner.z);
    hashFloat(params.camera.horizontal.x);
    hashFloat(params.camera.horizontal.y);
    hashFloat(params.camera.horizontal.z);
    hashFloat(params.camera.vertical.x);
    hashFloat(params.camera.vertical.y);
    hashFloat(params.camera.vertical.z);
    
    return hash;
}

void OptixWrapper::launch_random_pixel_mode_progressive(
    SDL_Surface* surface,
    SDL_Window* window,
    SDL_Renderer* renderer,
    int width,
    int height,
    std::vector<uchar4>& framebuffer,
    SDL_Texture* raytrace_texture
) {
    using namespace std::chrono;
    rendering_in_progress = true;

    const int pixel_count = width * height;

    // ------------------ BUFFER ALLOCATION -----------------------
    // Framebuffer for display (uchar4)
    if (!d_framebuffer || prev_width != width || prev_height != height) {
        if (d_framebuffer) cudaFree(d_framebuffer);
        cudaMalloc(&d_framebuffer, pixel_count * sizeof(uchar4));
        prev_width = width;
        prev_height = height;
        accumulation_valid = false; // Force reset on resolution change
    }

    // High-precision accumulation buffer (float4: RGB + sample count)
    if (!d_accumulation_float4 || !accumulation_valid) {
        if (d_accumulation_float4) cudaFree(d_accumulation_float4);
        cudaMalloc(&d_accumulation_float4, pixel_count * sizeof(float4));
        cudaMemset(d_accumulation_float4, 0, pixel_count * sizeof(float4));
        accumulation_valid = true;
    }

    // ------------------ CAMERA CHANGE DETECTION -----------------------
    uint64_t current_camera_hash = computeCameraHash();
    bool camera_changed = (current_camera_hash != last_camera_hash);
    bool is_first_render = (last_camera_hash == 0);

    if (camera_changed) {
        // Camera moved - reset accumulation
        cudaMemset(d_accumulation_float4, 0, pixel_count * sizeof(float4));
        accumulated_samples = 0;

        // Log only when actually moving camera, not initial state
        if (!is_first_render) {
            // SCENE_LOG_INFO("Camera changed - resetting accumulation");
        }

        last_camera_hash = current_camera_hash;
    }

    // ------------------ DETERMINE SAMPLES TO RENDER -----------------------
    // Max samples from settings (default to 100 if not set)
    int target_max_samples = render_settings.max_samples > 0 ? render_settings.max_samples : 100;

    // If we've reached max samples, don't render more
    if (accumulated_samples >= target_max_samples) {
        rendering_in_progress = false;
        return;
    }

    // Samples per pass: 1 for smooth progressive updates
    // Could increase for faster convergence at cost of UI responsiveness
    int samples_this_pass = 1;

    // ------------------ SETUP PARAMS -----------------------
    params.framebuffer = d_framebuffer;
    params.accumulation_buffer = reinterpret_cast<float*>(d_accumulation_float4);
    params.image_width = width;
    params.image_height = height;
    params.handle = traversable_handle;
    params.materials = reinterpret_cast<GpuMaterial*>(d_materials);



    // Use 1 sample per pixel per pass for smooth progressive refinement
    params.samples_per_pixel = samples_this_pass;
    params.min_samples = render_settings.min_samples;
    params.max_samples = render_settings.max_samples;
    params.variance_threshold = render_settings.variance_threshold;
    params.max_depth = render_settings.max_bounces;
    params.use_adaptive_sampling = false; // Progressive mode handles this differently

    // Frame number is the accumulated sample count (for random seed variation)
    params.frame_number = accumulated_samples + 1;
    params.current_pass = accumulated_samples;
    params.is_final_render = render_settings.is_final_render_mode ? 1 : 0;
    params.grid_enabled = render_settings.grid_enabled ? 1 : 0;
    params.grid_fade_distance = render_settings.grid_fade_distance;
    params.clip_near = render_settings.viewport_near_clip;
    params.clip_far = render_settings.viewport_far_clip;
    params.temporal_blend = 0.0f; // We handle blending manually via accumulation buffer

    // Full image tiles
    params.tile_x = 0;
    params.tile_y = 0;
    params.tile_width = width;
    params.tile_height = height;

    // ------------------ UPLOAD PARAMS -----------------------
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(RayGenParams));
    cudaMemcpyAsync(reinterpret_cast<void*>(d_params), &params, sizeof(RayGenParams), cudaMemcpyHostToDevice, stream);

    // ------------------ LAUNCH RENDER -----------------------
    auto pass_start = high_resolution_clock::now();

    OPTIX_CHECK(optixLaunch(
        pipeline, stream,
        d_params, sizeof(RayGenParams),
        &sbt,
        width, height, 1
    ));

    cudaStreamSynchronize(stream);

    auto pass_end = high_resolution_clock::now();
    float pass_ms = duration<float, std::milli>(pass_end - pass_start).count();

    // Update accumulated sample count
    accumulated_samples += samples_this_pass;

    // ------------------ COPY BACK & DISPLAY -----------------------
    partial_framebuffer.resize(pixel_count);

    cudaMemcpyAsync(partial_framebuffer.data(),
        d_framebuffer,
        pixel_count * sizeof(uchar4),
        cudaMemcpyDeviceToHost,
        stream);

    cudaStreamSynchronize(stream);

    // Update SDL Surface
    Uint32* pixels = (Uint32*)surface->pixels;
    int row_stride = surface->pitch / 4;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int fb_index = j * width + i;
            const uchar4& c = partial_framebuffer[fb_index];
            int screen_index = (height - 1 - j) * row_stride + i;
            pixels[screen_index] = SDL_MapRGB(surface->format, c.x, c.y, c.z);
        }
    }

    // ------------------ PROGRESS DISPLAY -----------------------
    float progress = 100.0f * accumulated_samples / target_max_samples;
    std::string title = "RayTrophi - Sample " + std::to_string(accumulated_samples) +
        "/" + std::to_string(target_max_samples) +
        " (" + std::to_string(int(progress)) + "%) - " +
        std::to_string(int(pass_ms)) + "ms/sample";
    SDL_SetWindowTitle(window, title.c_str());

    // Cleanup temporary params
    cudaFree(reinterpret_cast<void*>(d_params));
    d_params = 0;
    
    rendering_in_progress = false;
}
    // Local helper to avoid linker collisions
static inline float3 optix_to_float3(const Vec3& v) {
    return make_float3(v.x, v.y, v.z);
}

void OptixWrapper::setCameraParams(const Camera& cpuCamera) {
    params.camera.origin = optix_to_float3(cpuCamera.origin);
    params.camera.horizontal = optix_to_float3(cpuCamera.horizontal);
    params.camera.vertical = optix_to_float3(cpuCamera.vertical);
    params.camera.lower_left_corner = optix_to_float3(cpuCamera.lower_left_corner);

    // DOF için yeni parametreler:
    params.camera.u = optix_to_float3(cpuCamera.u);
    params.camera.v = optix_to_float3(cpuCamera.v);
    params.camera.w = optix_to_float3(cpuCamera.w);

    params.camera.lens_radius = static_cast<float>(cpuCamera.lens_radius);
    params.camera.focus_dist = static_cast<float>(cpuCamera.focus_dist);
	params.camera.aperture = static_cast<float>(cpuCamera.aperture);
    params.camera.blade_count = cpuCamera.blade_count;
}
__host__ __device__ inline float optix_length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}


void OptixWrapper::setWorld(const WorldData& world) {
    params.world = world;
    // Sync legacy background color for now (optional, depends on shader)
    params.background_color = world.color; 
}
void OptixWrapper::setLightParams(const std::vector<std::shared_ptr<Light>>& lights) {
    std::vector<LightGPU> gpuLights;

    for (const auto& light : lights) {
        LightGPU l = {};
        const Vec3& color = light->color;
        float intensity = light->intensity*5 ;

        // Initialize default values for new fields
        l.inner_cone_cos = 1.0f;
        l.outer_cone_cos = 0.0f;
        l.area_width = 0.0f;
        l.area_height = 0.0f;
        l.area_u = make_float3(1, 0, 0);
        l.area_v = make_float3(0, 1, 0);

        if (auto pointLight = std::dynamic_pointer_cast<PointLight>(light)) {
            const Vec3& pos = pointLight->position;
            l.position = make_float3(pos.x, pos.y, pos.z);
            l.color = make_float3(color.x, color.y, color.z);
            l.intensity = intensity;
            l.radius = pointLight->getRadius();
            l.type = 0;
        }
        else if (auto dirLight = std::dynamic_pointer_cast<DirectionalLight>(light)) {
            Vec3 dir = -dirLight->direction.normalize();
            l.direction = make_float3(dir.x, dir.y, dir.z);
            l.color = make_float3(color.x, color.y, color.z);
            l.intensity = intensity;
            l.radius = dirLight->getDiskRadius();
            l.type = 1;
        }
        else if (auto areaLight = std::dynamic_pointer_cast<AreaLight>(light)) {
            const Vec3& pos = areaLight->position;
            l.position = make_float3(pos.x, pos.y, pos.z);
            Vec3 dir = areaLight->direction.normalize();
            l.direction = make_float3(dir.x, dir.y, dir.z);
            l.color = make_float3(color.x, color.y, color.z);
            l.intensity = intensity;
            l.radius = 0.0f;
            l.type = 2;
            
            // AreaLight ek parametreleri
            l.area_width = areaLight->getWidth();
            l.area_height = areaLight->getHeight();
            Vec3 u = areaLight->getU();
            Vec3 v = areaLight->getV();
            l.area_u = make_float3(u.x, u.y, u.z);
            l.area_v = make_float3(v.x, v.y, v.z);
        }
        else if (auto spotLight = std::dynamic_pointer_cast<SpotLight>(light)) {
            const Vec3& pos = spotLight->position;
            Vec3 dir = spotLight->direction.normalize();
            l.position = make_float3(pos.x, pos.y, pos.z);
            l.direction = make_float3(dir.x, dir.y, dir.z);
            l.color = make_float3(color.x, color.y, color.z);
            l.intensity = intensity;
            l.radius = 0.0f;
            l.type = 3;
            
            // SpotLight cone angle parametreleri
            float angleDeg = spotLight->getAngleDegrees();
            float angleRad = angleDeg * (M_PI / 180.0f);
            l.inner_cone_cos = cosf(angleRad * 0.8f);  // İç cone (daha dar)
            l.outer_cone_cos = cosf(angleRad);          // Dış cone (tam açı)
        }

        gpuLights.push_back(l);
    }

    // GPU'ya kopyala
    CUdeviceptr d_lights;
    size_t byteSize = gpuLights.size() * sizeof(LightGPU);
    cudaMalloc(reinterpret_cast<void**>(&d_lights), byteSize);
    cudaMemcpy(reinterpret_cast<void*>(d_lights), gpuLights.data(), byteSize, cudaMemcpyHostToDevice);

    params.lights = reinterpret_cast<LightGPU*>(d_lights);
    params.light_count = static_cast<int>(gpuLights.size());
}

bool OptixWrapper::SaveSurface(SDL_Surface* surface, const char* filename) {
    //  Sabit ve değişmeyen klasör yolu
    std::filesystem::path output_dir = "E:/visual studio proje c++/raytracing_Proje_Moduler/raytrac_sdl2/image";

    if (!std::filesystem::exists(output_dir)) {
        std::error_code ec;
        if (!std::filesystem::create_directories(output_dir, ec)) {
            SDL_Log("Could not create folder: %s", ec.message().c_str());
            return false;
        }
    }

    std::filesystem::path full_path = output_dir / filename;

    SDL_Surface* surface_to_save = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB24, 0);
    if (!surface_to_save) {
        SDL_Log("Couldn't convert surface: %s", SDL_GetError());
        return false;
    }

    int result = IMG_SavePNG(surface_to_save, full_path.string().c_str());
    SDL_FreeSurface(surface_to_save);

    if (result != 0) {
        SDL_Log("Failed to save image: %s", IMG_GetError());
        return false;
    }

    SDL_Log("Image saved to: %s", full_path.string().c_str());
    return true;
}

void OptixWrapper::resetBuffers(int width, int height) {
    if (prev_width != width || prev_height != height) {
        if (d_accumulation_buffer) { cudaFree(d_accumulation_buffer); d_accumulation_buffer = nullptr; }
        if (d_variance_buffer) { cudaFree(d_variance_buffer); d_variance_buffer = nullptr; }
        if (d_sample_count_buffer) { cudaFree(d_sample_count_buffer); d_sample_count_buffer = nullptr; }
        
        // Critical Fix: Also free float4 accumulation buffer to prevent crash on resize
        if (d_accumulation_float4) { cudaFree(d_accumulation_float4); d_accumulation_float4 = nullptr; }
        
        // Also invalidate framebuffer so launch functions re-allocate
        if (d_framebuffer) { cudaFree(d_framebuffer); d_framebuffer = nullptr; }

        cudaMalloc(&d_accumulation_buffer, sizeof(float) * width * height * 3);
        cudaMalloc(&d_variance_buffer, sizeof(float) * width * height);
        cudaMalloc(&d_sample_count_buffer, sizeof(int) * width * height);
        
        // Re-allocate float4 accumulation buffer used by progressive renderer
        cudaMalloc(&d_accumulation_float4, sizeof(float4) * width * height);

        prev_width = width;
        prev_height = height;
        
        accumulated_samples = 0;
        accumulation_valid = false;
    }

    cudaMemset(d_accumulation_buffer, 0, sizeof(float) * width * height * 3);
    cudaMemset(d_variance_buffer, 0, sizeof(float) * width * height);
    cudaMemset(d_sample_count_buffer, 0, sizeof(int) * width * height);
    
    if (d_accumulation_float4) {
        cudaMemset(d_accumulation_float4, 0, sizeof(float4) * width * height);
    }
    
    frame_counter = 1;
}

bool OptixWrapper::isAccumulationComplete() const {
    int target_max_samples = render_settings.max_samples > 0 ? render_settings.max_samples : 100;
    return accumulated_samples >= target_max_samples;
}

void OptixWrapper::resetAccumulation() {
    // Reset the Cycles-style accumulation buffer for new frame
    if (d_accumulation_float4 && prev_width > 0 && prev_height > 0) {
        cudaMemset(d_accumulation_float4, 0, prev_width * prev_height * sizeof(float4));
    }
    accumulated_samples = 0;
    last_camera_hash = 0;  // Force re-hash on next render
    accumulation_valid = true;
}

// ============================================================================
// CRITICAL: updateGeometry() Limitations
// ============================================================================
// This function ONLY updates vertex positions and normals.
// It does NOT update material indices buffer (d_material_indices).
// 
// USE FOR:
//   ✅ Object transformation (position/rotation/scale changes)
//   ✅ Animation playback (vertex deformation)
//
// DO NOT USE FOR:
//   ❌ Object deletion (material indices become misaligned)
//   ❌ Object addition (material indices need regeneration)
//   ❌ Material changes (SBT needs rebuild)
//
// For deletion/addition, use Renderer::rebuildOptiXGeometry() instead.
// See OPTIX_MATERIAL_FIX.md for detailed explanation.
// ============================================================================
void OptixWrapper::updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (objects.empty()) return;

    std::vector<float3> vertices;
    std::vector<float3> normals; 

    vertices.reserve(objects.size() * 3);
    normals.reserve(objects.size() * 3);

    for (const auto& obj : objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri) {
            Vec3 v0 = tri->getVertexPosition(0);
            Vec3 v1 = tri->getVertexPosition(1);
            Vec3 v2 = tri->getVertexPosition(2);

            vertices.push_back({ v0.x, v0.y, v0.z });
            vertices.push_back({ v1.x, v1.y, v1.z });
            vertices.push_back({ v2.x, v2.y, v2.z });
            
            // Normals
            Vec3 n0 = tri->getVertexNormal(0);
            Vec3 n1 = tri->getVertexNormal(1);
            Vec3 n2 = tri->getVertexNormal(2);
            
            normals.push_back({ n0.x, n0.y, n0.z });
            normals.push_back({ n1.x, n1.y, n1.z });
            normals.push_back({ n2.x, n2.y, n2.z });
        }
    }

    if (vertices.empty()) return;

    // 1. Update Vertex and Normal Buffers on GPU (SAFE RESIZING)
    size_t new_vertex_size = vertices.size() * sizeof(float3);
    if (!d_vertices || new_vertex_size > allocated_vertex_byte_size) {
        if (d_vertices) cudaFree(reinterpret_cast<void*>(d_vertices));
        cudaMalloc(reinterpret_cast<void**>(&d_vertices), new_vertex_size);
        allocated_vertex_byte_size = new_vertex_size;
        // Optimization: If we reallocated, pointers changed. We might need full build if UPDATE depended on fixed ptrs?
        // OptiX allows pointer changes in buildInputs for UPDATE.
    }
    cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices.data(), new_vertex_size, cudaMemcpyHostToDevice);

    size_t new_normal_size = normals.size() * sizeof(float3);
    if (!d_normals || new_normal_size > allocated_normal_byte_size) {
        if (d_normals) cudaFree(reinterpret_cast<void*>(d_normals));
        cudaMalloc(reinterpret_cast<void**>(&d_normals), new_normal_size);
        allocated_normal_byte_size = new_normal_size;
    }
    cudaMemcpy(reinterpret_cast<void*>(d_normals), normals.data(), new_normal_size, cudaMemcpyHostToDevice);

    // Update SBT Logic? 
    // Currently, SBT stores pointers to d_vertices. If d_vertices changed address, SBT is stale!
    // FIX: We must update SBT if we reallocated OR if we switched from Indexed to Soup (pointers change logic).
    // For now, let's assume SBT has pointer to d_vertices. If d_vertices changed, we need to refresh it.
    // Ideally, we should implement a `updateSBT()` method. 
    // But since this is a quick fix, if we realloc, we are in danger.
    // LUCKILY, cuMemcpy to EXISTING d_vertices (if size fits) works fine.
    // If we realloc, we MUST update SBT. 
    // AS A SAFEGUARD: If we realloc, we accept the overhead? No, invalid pointer crash.
    // I will add a simplified SBT pointer update here.
    
    // Update SBT Records with new pointers
    // Note: hitgroup_records is a member variable, so we can iterate it.
    for (auto& rec : hitgroup_records) {
        rec.data.vertices = reinterpret_cast<float3*>(d_vertices);
        rec.data.normals = reinterpret_cast<float3*>(d_normals);
        // Note: Indices become invalid/unused in Soup mode, but shader might read them? 
        // If shader uses indices for lookup, it will read garbage/old d_indices.
        // We really should provide dummy indices or ensure shader handles non-indexed.
    }
    if (allocated_vertex_byte_size != 0) { // Only copy if we have data
         CUdeviceptr d_hitgroup_records = sbt.hitgroupRecordBase;
         cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records), hitgroup_records.data(), 
                    hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>), cudaMemcpyHostToDevice);
    }


    // 2. Refit Logic
    bool allow_refit = is_gas_built_as_soup && (traversable_handle != 0) && (vertices.size() == last_vertex_count); 
    // Refit is possible if we are correctly in Soup mode, have a valid handle, and TOPOLOGY (vertex count) hasn't changed. 
    // Refit is possible if we are correctly in Soup mode and have a valid handle.
    // Also, strictly speaking, refit is only valid if number of primitives matches.
    // We assume if is_gas_built_as_soup is true, we built it as soup before.
    // We should also check if size matches to imply no topology change.
    // Actually, updateGeometry builds soup based on `objects.size()`.
    // If `objects.size()` changes, we MUST rebuild.
    // Let's add a loose check on size matching approximate primitive count.
    
    // Construct Build Input (Triangle Soup)
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    build_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    build_input.triangleArray.vertexBuffers = &d_vertices;

    // No Indices (Soup)
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    build_input.triangleArray.indexStrideInBytes = 0;
    build_input.triangleArray.numIndexTriplets = 0;
    build_input.triangleArray.indexBuffer = 0;

    std::vector<uint32_t> triangle_input_flags(hitgroup_records.size(), OPTIX_GEOMETRY_FLAG_NONE);
    build_input.triangleArray.flags = triangle_input_flags.data();
    build_input.triangleArray.numSbtRecords = static_cast<uint32_t>(hitgroup_records.size()); 
    build_input.triangleArray.sbtIndexOffsetBuffer = d_material_indices;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(int);
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(int);


    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    
    // DECIDE: BUILD or UPDATE
    if (allow_refit) {
        accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    } else {
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        is_gas_built_as_soup = true; // Next time we can try update
    }

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1, &gas_buffer_sizes));

    // Clean up temporary buffers
    if (d_temp_buffer) {
        cudaFree(reinterpret_cast<void*>(d_temp_buffer));
        d_temp_buffer = 0;
    }
    if (d_compacted_size) {
        cudaFree(reinterpret_cast<void*>(d_compacted_size));
        d_compacted_size = 0;
    }
    cudaDeviceSynchronize();
    
    // Re-allocate temp buffer if needed
    cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_compacted_size), sizeof(uint64_t));
    // No, update takes the `traversable_handle` as in/out.
    // But we also need scratch space.
    // And if operation is BUILD, we allocate output.
    // If operation is UPDATE, we usually don't need new output buffer allocation if size is same.
    // BUT optixAccelBuild output param is `d_output_buffer`.
    // For Update: "The output buffer... must be the same buffer that was used for the initial build."
    // So we should NOT free d_bvh_output if updating.
    
    if (accel_options.operation == OPTIX_BUILD_OPERATION_BUILD) {
         if(d_bvh_output) cudaFree((void*)d_bvh_output);
         cudaMalloc((void**)&d_bvh_output, gas_buffer_sizes.outputSizeInBytes);
    }
    // Else: Reuse d_bvh_output

    OPTIX_CHECK(optixAccelBuild(
        context,
        0,                  
        &accel_options,
        &build_input,
        1,                  
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_bvh_output,
        gas_buffer_sizes.outputSizeInBytes,
        &traversable_handle,
        nullptr,            
        0                   
    ));
    
    
    cudaDeviceSynchronize();

    last_vertex_count = vertices.size(); // Update tracking for next frame
}
