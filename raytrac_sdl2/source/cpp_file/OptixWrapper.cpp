#include "OptixWrapper.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <chrono> // Süre ölçmek için
#include <unordered_map> // Gereken başlık
#include <algorithm>    // std::min ve std::max için
#include <SpotLight.h>
#include <filesystem>
#include <imgui.h>
#include <imgui_impl_sdlrenderer2.h>

#undef min              // Eğer min bir yerde macro tanımlandıysa temizler
#undef max

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


OptixWrapper::OptixWrapper()
    : Image_width(image_width), Image_height(image_height), color_processor(image_width, image_height) //  işte burada!
{
    d_vertices = 0;
    d_indices = 0;
    d_bvh_output = 0;
    d_accumulation_buffer = nullptr;
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
    traversable_handle = 0;
}

OptixWrapper::~OptixWrapper() {
    cleanup();
}
float3 to_float3(const Vec3& v) {
    return make_float3(v.x, v.y, v.z);
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

inline float3 toFloat3(const Vec3& v) {
    return make_float3(v.x, v.y, v.z);
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
void OptixWrapper::setupOIDN(int width, int height)
{
    if (oidnInitialized && oidnLastWidth == width && oidnLastHeight == height)
        return; // Çözünürlük aynı -> Hiçbir şey yapma, en hızlı yol

    // Aşağısı sadece çözünürlük değişirse (ya da ilk kurulumda) çalışır:
    oidnInputBuffer = nullptr;
    oidnOutputBuffer = nullptr;
    oidnFilter = nullptr;

    if (!oidnDevice)
    {
        bool useCUDA = g_hasOptix;

        oidnDevice = useCUDA
            ? oidn::newDevice(oidn::DeviceType::CUDA)
            : oidn::newDevice(oidn::DeviceType::CPU);

        // Log
        if (useCUDA)
            std::cout << "[OIDN] Using CUDA device for denoising.\n";
        else
            std::cout << "[OIDN] Using CPU device for denoising.\n";

        oidnDevice.commit();
    }


    const size_t byteSize = width * height * 3 * sizeof(float);

    oidnInputBuffer = oidnDevice.newBuffer(byteSize);
    oidnOutputBuffer = oidnDevice.newBuffer(byteSize);

    oidnFilter = oidnDevice.newFilter("RT");
    oidnFilter.setImage("color", oidnInputBuffer, oidn::Format::Float3, width, height);
    oidnFilter.setImage("output", oidnOutputBuffer, oidn::Format::Float3, width, height);
    oidnFilter.set("hdr", false);
    oidnFilter.set("srgb", true);
    oidnFilter.commit();

    oidnLastWidth = width;
    oidnLastHeight = height;
    oidnInitialized = true;
}

void OptixWrapper::applyOIDNDenoising(SDL_Surface* surface, bool denoise, float blend)
{
    int width = surface->w;
    int height = surface->h;

    setupOIDN(width, height);  // Çok hızlı - çözünürlük değişmediyse boş

    if (!oidnInitialized)
        return;

    const size_t pixelCount = width * height;
    const size_t byteSize = pixelCount * 3 * sizeof(float);

    Uint32* pixels = static_cast<Uint32*>(surface->pixels);

    // Paylaşılan statik vektör (her kare yeniden allocate etmez)
    static std::vector<float> input;
    static std::vector<float> output;

    input.resize(pixelCount * 3);
    output.resize(pixelCount * 3);

    for (int i = 0; i < pixelCount; i++) {
        Uint8 r, g, b;
        SDL_GetRGB(pixels[i], surface->format, &r, &g, &b);
        input[i * 3 + 0] = r / 255.0f;
        input[i * 3 + 1] = g / 255.0f;
        input[i * 3 + 2] = b / 255.0f;
    }

    std::memcpy(oidnInputBuffer.getData(), input.data(), byteSize);

    oidnFilter.set("denoise", denoise);
    oidnFilter.commit();

    oidnFilter.execute();

    std::memcpy(output.data(), oidnOutputBuffer.getData(), byteSize);

    float clamped = std::clamp(blend, 0.0f, 1.0f);

    for (int i = 0; i < pixelCount; ++i) {
        Uint8 r0, g0, b0;
        SDL_GetRGB(pixels[i], surface->format, &r0, &g0, &b0);

        float rr = output[i * 3];
        float gg = output[i * 3 + 1];
        float bb = output[i * 3 + 2];

        float r = rr * clamped + (r0 / 255.0f) * (1.0f - clamped);
        float g = gg * clamped + (g0 / 255.0f) * (1.0f - clamped);
        float b = bb * clamped + (b0 / 255.0f) * (1.0f - clamped);

        pixels[i] = SDL_MapRGB(surface->format,
            (Uint8)(std::clamp(r * 255.0f, 0.0f, 255.0f)),
            (Uint8)(std::clamp(g * 255.0f, 0.0f, 255.0f)),
            (Uint8)(std::clamp(b * 255.0f, 0.0f, 255.0f))
        );
    }
}

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
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    SbtRecord raygen_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &raygen_record));
    cudaMalloc(reinterpret_cast<void**>(&sbt.raygenRecord), sizeof(SbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(sbt.raygenRecord), &raygen_record, sizeof(SbtRecord), cudaMemcpyHostToDevice);

    SbtRecord miss_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &miss_record));
    cudaMalloc(reinterpret_cast<void**>(&sbt.missRecordBase), sizeof(SbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(sbt.missRecordBase), &miss_record, sizeof(SbtRecord), cudaMemcpyHostToDevice);
    sbt.missRecordStrideInBytes = sizeof(SbtRecord);
    sbt.missRecordCount = 1;

    SbtRecord hit_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hit_pg, &hit_record));
    cudaMalloc(reinterpret_cast<void**>(&sbt.hitgroupRecordBase), sizeof(SbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(sbt.hitgroupRecordBase), &hit_record, sizeof(SbtRecord), cudaMemcpyHostToDevice);
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
    sbt.hitgroupRecordCount = 1;
}
void OptixWrapper::destroyTextureObjects() {
    for (const auto& record : hitgroup_records) {
        const HitGroupData& data = record.data;

        if (data.albedo_tex) cudaDestroyTextureObject(data.albedo_tex);
        if (data.roughness_tex) cudaDestroyTextureObject(data.roughness_tex);
        if (data.normal_tex) cudaDestroyTextureObject(data.normal_tex);
        if (data.metallic_tex) cudaDestroyTextureObject(data.metallic_tex);
        if (data.transmission_tex) cudaDestroyTextureObject(data.transmission_tex);
        if (data.opacity_tex) cudaDestroyTextureObject(data.opacity_tex);
        if (data.emission_tex) cudaDestroyTextureObject(data.emission_tex);
    }

    hitgroup_records.clear(); // artık geçmiş yok
}

void OptixWrapper::buildFromData(const OptixGeometryData& data) {
    SCENE_LOG_INFO(" OptiX buildFromData is starting...");

    if (data.vertices.empty() || data.indices.empty()) {
        SCENE_LOG_ERROR(" Geometry data is empty!");
        return;
    }
    destroyTextureObjects();      
    partialCleanup();            // << Ardından tüm CUDA buffer'larını temizle  

    // 1. Tüm geometri verilerini GPU'ya gönder
    size_t v_size = data.vertices.size() * sizeof(float3);
    cudaMalloc(reinterpret_cast<void**>(&d_vertices), v_size);
    cudaMemcpy(reinterpret_cast<void*>(d_vertices), data.vertices.data(), v_size, cudaMemcpyHostToDevice);

    size_t i_size = data.indices.size() * sizeof(uint3);
    cudaMalloc(reinterpret_cast<void**>(&d_indices), i_size);
    cudaMemcpy(reinterpret_cast<void*>(d_indices), data.indices.data(), i_size, cudaMemcpyHostToDevice);

    d_normals = 0;
    if (!data.normals.empty()) {
        size_t n_size = data.normals.size() * sizeof(float3);
        cudaMalloc(reinterpret_cast<void**>(&d_normals), n_size);
        cudaMemcpy(reinterpret_cast<void*>(d_normals), data.normals.data(), n_size, cudaMemcpyHostToDevice);
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
    cudaFree(reinterpret_cast<void*>(d_compacted_size));
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

void OptixWrapper::launch_random_pixel_mode_progressive(
    SDL_Surface* surface,
    SDL_Window* window,
    SDL_Renderer* renderer, // ⬅️ YENİ PARAMETRE
    int width,
    int height,
    std::vector<uchar4>& framebuffer,
    SDL_Texture* raytrace_texture
) {
    using namespace std::chrono;
	rendering_in_progress = true;
    cudaGetDeviceProperties(&props, 0);
    int max_threads_per_block = props.maxThreadsPerBlock;
    int num_sms = props.multiProcessorCount;

    int image_pixels = width * height;

    // ---------------- ADAPTIVE BATCH HESABI ----------------
    // Kurallar:
    // - Küçük görüntülerde: tek seferde komple render (overhead minimal).
    // - Büyük görüntülerde: tile >= 512x512 (262144) tercih et.
    const int MIN_TILE = 256 * 256;        // 262144
    const int MAX_TILE = 1024 * 1024;      // 1048576
   
    int pixels_per_launch = image_pixels;
    if (image_pixels > MIN_TILE) {
        // Target temel: her SM için bir miktar iş bırak (512 thread/SM hedef)
        int target_threads = num_sms * 512; // yaklaşık hedef thread sayısı
        // Pixel-per-thread 1 kabul ederek hedef pixel sayısı:
        int target_pixels = target_threads;
        // Clamp hedefi makul tile aralığına al
        int clamped = std::clamp(target_pixels, MIN_TILE, std::min(image_pixels, MAX_TILE));
        // Ayrıca image büyüklüğüne göre tile sayısını mantıklı böl (ör. 4 tile veya 8 tile)
        // Burada tercihen kaç tile yapılacağına göre ayarla:
        int desired_tiles = std::clamp(image_pixels / clamped, 1, 16);
        pixels_per_launch = std::min(image_pixels, std::max(clamped, image_pixels / desired_tiles));
    }
    else {
        // Küçük resim: tek seferde tamamla
        pixels_per_launch = image_pixels;
    }

    // display update interval: küçük çözünürlükte sık, büyükte seyrek
    int display_update_interval = (image_pixels <= MIN_TILE) ? 1 : std::clamp(image_pixels / (pixels_per_launch * 2), 1, 4);

    // ------------------ FRAMEBUFFER SETUP -----------------------
    cudaMalloc(&d_framebuffer, width * height * sizeof(uchar4));
    cudaMemset(d_framebuffer, 0, width * height * sizeof(uchar4));

    params.framebuffer = d_framebuffer;
    params.image_width = width;
    params.image_height = height;
    params.handle = traversable_handle;
    params.materials = reinterpret_cast<GpuMaterial*>(d_materials);

    // Atmosfer ve diğer parametreler aynen
    params.atmosphere.sigma_s = 0.01f;
    params.atmosphere.sigma_a = 0.01f;
    params.atmosphere.g = 0.0f;
    params.atmosphere.base_density = 0.001f;
    params.atmosphere.temperature = 300.0f;
    params.atmosphere.active = false;

    params.samples_per_pixel = render_settings.samples_per_pixel;
    params.min_samples = render_settings.min_samples;
    params.max_samples = render_settings.max_samples;
    params.variance_threshold = render_settings.variance_threshold;
    params.max_depth = render_settings.max_bounces;
    params.use_adaptive_sampling = render_settings.use_adaptive_sampling;

    params.frame_number = frame_counter;
    frame_counter++;
    params.temporal_blend = 0.95f;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(RayGenParams));

    // ------------------ PİKSEL KOORDİNATLARI ------------------
    std::vector<std::pair<int, int>> all_coords;
    all_coords.reserve(width * height);
    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i)
            all_coords.emplace_back(i, j);

    static std::mt19937 rng(12345);
    std::shuffle(all_coords.begin(), all_coords.end(), rng);

    // Dinamik batch vektörleri (maks pixels_per_launch kadar)
    std::vector<int> coords_x;
    std::vector<int> coords_y;
    coords_x.reserve(pixels_per_launch);
    coords_y.reserve(pixels_per_launch);

    cudaMalloc(reinterpret_cast<void**>(&d_coords_x), pixels_per_launch * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_coords_y), pixels_per_launch * sizeof(int));

    size_t total_pixels = all_coords.size();
    size_t rendered_pixels = 0;

    cudaStream_t render_stream;
    cudaStreamCreate(&render_stream);

    partial_framebuffer.resize(width * height);
    std::vector<std::pair<int, int>> accumulated_coords;
    accumulated_coords.reserve(pixels_per_launch * display_update_interval);

    // SDL pixel buffer
    Uint32* pixels = (Uint32*)surface->pixels;
    int row_stride = surface->pitch / 4;

    auto start_time = high_resolution_clock::now();
    auto last_present_time = start_time;
    double fps_ema = 0.0;
    const double ema_alpha = 0.15; // fps smoothing

    int batch_counter = 0;
   /* SCENE_LOG_INFO("OptiX progressive render launched.");
    SCENE_LOG_INFO("Device: " + std::string(props.name) +
        " | SMs: " + std::to_string(num_sms) +
        " | Max threads per block: " + std::to_string(max_threads_per_block));

    SCENE_LOG_INFO("Total pixels: " + std::to_string(total_pixels) +
        " | Pixels per launch: " + std::to_string(pixels_per_launch) +
        " | Display update interval: " + std::to_string(display_update_interval));*/

    size_t count;
    for (size_t offset = 0; offset < total_pixels; offset += pixels_per_launch) {
        count = std::min<size_t>(pixels_per_launch, total_pixels - offset);
        size_t last_logged_pixels = 0;
        coords_x.clear();
        coords_y.clear();
        for (size_t k = 0; k < count; ++k) {
            int index = offset + k;
            coords_x.push_back(all_coords[index].first);
            coords_y.push_back(all_coords[index].second);
            accumulated_coords.emplace_back(coords_x.back(), coords_y.back());
        }
        if (rendering_stopped_gpu)
        {
            rendering_stopped_gpu = false;

			break;
        }
            
           

        // Async memory transfer
        cudaMemcpyAsync(reinterpret_cast<int*>(d_coords_x), coords_x.data(),
            count * sizeof(int), cudaMemcpyHostToDevice, render_stream);
        cudaMemcpyAsync(reinterpret_cast<int*>(d_coords_y), coords_y.data(),
            count * sizeof(int), cudaMemcpyHostToDevice, render_stream);

        params.launch_coords_x = reinterpret_cast<int*>(d_coords_x);
        params.launch_coords_y = reinterpret_cast<int*>(d_coords_y);
        params.batch_pixel_count = static_cast<int>(count);

        cudaMemcpyAsync(reinterpret_cast<void*>(d_params), &params,
            sizeof(RayGenParams), cudaMemcpyHostToDevice, render_stream);

        // Render launch (count threads)
        OPTIX_CHECK(optixLaunch(
            pipeline, render_stream,
            d_params, sizeof(RayGenParams),
            &sbt,
            static_cast<unsigned int>(count), 1, 1
        ));

        // *** Daha verimli: sadece değişen piksellerin D2H kopyalanması çok daha hızlı olur.
        // Burada pratiklik için tüm framebuffer'ı kopyalıyoruz. İstersen sadece 'coords_x/y' ile
        // tek tek pikselleri CUDA kernel ile host-buffer'a yazdırıp D2H kopyasını küçültebiliriz.
        cudaMemcpyAsync(partial_framebuffer.data(),
            d_framebuffer,
            width * height * sizeof(uchar4),
            cudaMemcpyDeviceToHost,
            render_stream);

        cudaStreamSynchronize(render_stream);

        rendered_pixels += count;
        batch_counter++;

        bool should_update = (batch_counter % display_update_interval == 0) ||
            (offset + pixels_per_launch >= total_pixels);

        if (should_update && !accumulated_coords.empty()) {
            // Geri dönen pikselleri surface'a yaz
            for (const auto& coord : accumulated_coords) {
                int px = coord.first;
                int py = coord.second;
                int fb_index = py * width + px;
                const uchar4& c = partial_framebuffer[fb_index];

                Vec3 raw_color(c.x / 255.0f, c.y / 255.0f, c.z / 255.0f);
				raw_color = Vec3::clamp(raw_color, 0.0f, 1.0f);
                float gpu_gamma = 1.0f / 1.2f;
                uint8_t r = uint8_t(powf(raw_color.x, gpu_gamma) * 255.0f + 0.5f);
                uint8_t g = uint8_t(powf(raw_color.y, gpu_gamma) * 255.0f + 0.5f);
                uint8_t b = uint8_t(powf(raw_color.z, gpu_gamma) * 255.0f + 0.5f);

                int screen_index = (height - 1 - py) * row_stride + px;
                pixels[screen_index] = SDL_MapRGB(surface->format, r, g, b);
               
            }

            // Texture ve renderer güncelle
           
           
            accumulated_coords.clear();
        }
    }
    //SCENE_LOG_INFO("OptiX progressive render completed. Total pixels: " + std::to_string(total_pixels));
    // Temizlik
    cudaFree(reinterpret_cast<void*>(d_framebuffer));
    cudaFree(reinterpret_cast<void*>(d_params));
    cudaFree(reinterpret_cast<void*>(d_coords_x));
    cudaFree(reinterpret_cast<void*>(d_coords_y));
    cudaStreamDestroy(render_stream);
	rendering_in_progress = false;
   
}


void OptixWrapper::setCameraParams(const Camera& cpuCamera) {
    params.camera.origin = toFloat3(cpuCamera.origin);
    params.camera.horizontal = toFloat3(cpuCamera.horizontal);
    params.camera.vertical = toFloat3(cpuCamera.vertical);
    params.camera.lower_left_corner = toFloat3(cpuCamera.lower_left_corner);

    // DOF için yeni parametreler:
    params.camera.u = toFloat3(cpuCamera.u);
    params.camera.v = toFloat3(cpuCamera.v);
    params.camera.w = toFloat3(cpuCamera.w);

    params.camera.lens_radius = static_cast<float>(cpuCamera.lens_radius);
    params.camera.focus_dist = static_cast<float>(cpuCamera.focus_dist);
	params.camera.aperture = static_cast<float>(cpuCamera.aperture);
    params.camera.blade_count = cpuCamera.blade_count;
}
__host__ __device__ inline float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}


void OptixWrapper::setBackgroundColor(const Vec3& color) {
    params.background_color = make_float3(color.x, color.y, color.z);
}
void OptixWrapper::setLightParams(const std::vector<std::shared_ptr<Light>>& lights) {
    std::vector<LightGPU> gpuLights;

    for (const auto& light : lights) {
        LightGPU l = {};
        const Vec3& color = light->color;
        float intensity = light->intensity*10;

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
            l.radius = dirLight->getDiskRadius(); // disk ışık yarıçapı
            l.type = 1;
        }
        else if (auto areaLight = std::dynamic_pointer_cast<AreaLight>(light)) {
            const Vec3& pos = areaLight->position;
            l.position = make_float3(pos.x, pos.y, pos.z);
            Vec3 dir = areaLight->direction.normalize();
            l.direction = make_float3(dir.x, dir.y, dir.z);
            l.color = make_float3(color.x, color.y, color.z);
            l.intensity = intensity;
            l.radius = 0.0f; // gerekirse hesaplanır
            l.type = 2;
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
        if (d_accumulation_buffer) cudaFree(d_accumulation_buffer);
        if (d_variance_buffer) cudaFree(d_variance_buffer);
        if (d_sample_count_buffer) cudaFree(d_sample_count_buffer);

        cudaMalloc(&d_accumulation_buffer, sizeof(float) * width * height * 3);
        cudaMalloc(&d_variance_buffer, sizeof(float) * width * height);
        cudaMalloc(&d_sample_count_buffer, sizeof(int) * width * height);

        prev_width = width;
        prev_height = height;
    }

    cudaMemset(d_accumulation_buffer, 0, sizeof(float) * width * height * 3);
    cudaMemset(d_variance_buffer, 0, sizeof(float) * width * height);
    cudaMemset(d_sample_count_buffer, 0, sizeof(int) * width * height);
    frame_counter =1 ;
}

