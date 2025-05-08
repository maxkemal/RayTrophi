#include "OptixWrapper.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "sbt_record.h"
#include <sbt_data.h>
#include <chrono> // Süre ölçmek için
#include <unordered_map> // Gereken başlık
#include <algorithm>    // std::min ve std::max için
#undef min              // Eğer min bir yerde macro tanımlandıysa temizler
#undef max

#define OPTIX_CHECK(call) \
    do { OptixResult res = call; if (res != OPTIX_SUCCESS) { \
        std::cerr << "OptiX error: " << res << " at " << __FILE__ << ":" << __LINE__ << "\n"; std::abort(); } \
    } while(0)

OptixWrapper::OptixWrapper()
    : image_width(800), image_height(600), color_processor(800, 600) // 💥 işte burada!
{
    initialize();
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

    traversable_handle = 0;
}

OptixWrapper::~OptixWrapper() {
    cleanup();
}
float3 to_float3(const Vec3& v) {
    return make_float3(v.x, v.y, v.z);
}

void OptixWrapper::initialize() {
    cudaFree(0); // CUDA başlat

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = nullptr;
    options.logCallbackLevel = 4;

    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));

    cudaStreamCreate(&stream);
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
void OptixWrapper::applyOIDNDenoising(SDL_Surface* surface, int numThreads = 0, bool denoise = true, float blend = 0.8f) {
    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    int width = surface->w;
    int height = surface->h;

    // Renk verisini normalize ederek buffer'a aktar
    std::vector<float> colorBuffer(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        Uint8 r, g, b;
        SDL_GetRGB(pixels[i], surface->format, &r, &g, &b);
        colorBuffer[i * 3] = static_cast<float>(r) / 255.0f;
        colorBuffer[i * 3 + 1] = static_cast<float>(g) / 255.0f;
        colorBuffer[i * 3 + 2] = static_cast<float>(b) / 255.0f;
    }

    // CUDA veya CPU cihazını seç
    oidn::DeviceRef device;
    if (isCudaAvailable()) {
        device = oidn::newDevice(oidn::DeviceType::CUDA);
    }
    else {
        device = oidn::newDevice(oidn::DeviceType::CPU);
    }
    device.set("numThreads", numThreads);
    device.commit();

    // OIDN buffer'larını oluştur
    oidn::BufferRef colorOIDNBuffer = device.newBuffer(colorBuffer.size() * sizeof(float));
    // oidn::BufferRef normalOIDNBuffer = device.newBuffer(normalData.size() * sizeof(float)); // Normal buffer
    oidn::BufferRef outputOIDNBuffer = device.newBuffer(colorBuffer.size() * sizeof(float));

    std::memcpy(colorOIDNBuffer.getData(), colorBuffer.data(), colorBuffer.size() * sizeof(float));
    // std::memcpy(normalOIDNBuffer.getData(), normalData.data(), normalData.size() * sizeof(float));

     // Filtreyi yapılandır ve çalıştır
    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color", colorOIDNBuffer, oidn::Format::Float3, width, height);
    // filter.setImage("normal", normalOIDNBuffer, oidn::Format::Float3, width, height); // Normal verisini burada ekliyoruz
    filter.setImage("output", outputOIDNBuffer, oidn::Format::Float3, width, height);

    filter.set("hdr", false); // Normal map verisi için HDR (lineer veri)
    filter.set("srgb", true); // Gamma düzeltmesi uygulama
    filter.set("denoise", denoise);
    filter.commit();

    auto start = std::chrono::high_resolution_clock::now();
    filter.execute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Hataları kontrol et
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
        std::cerr << "OIDN error: " << errorMessage << std::endl;

    // Denoised veriyi al ve karıştır
    std::memcpy(colorBuffer.data(), outputOIDNBuffer.getData(), colorBuffer.size() * sizeof(float));
    for (int i = 0; i < width * height; ++i) {
        Uint8 r_orig, g_orig, b_orig;
        SDL_GetRGB(pixels[i], surface->format, &r_orig, &g_orig, &b_orig);

        Uint8 r = static_cast<Uint8>((colorBuffer[i * 3] * blend + r_orig / 255.0f * (1 - blend)) * 255);
        Uint8 g = static_cast<Uint8>((colorBuffer[i * 3 + 1] * blend + g_orig / 255.0f * (1 - blend)) * 255);
        Uint8 b = static_cast<Uint8>((colorBuffer[i * 3 + 2] * blend + b_orig / 255.0f * (1 - blend)) * 255);

        pixels[i] = SDL_MapRGB(surface->format, r, g, b);
    }
}
void OptixWrapper::validateMaterialIndices(const OptixGeometryData& data) {
    if (data.materials.empty()) {
        std::cerr << " HATA: Hiç materyal yok!\n";
        return;
    }

    if (data.indices.empty()) {
        std::cerr << " HATA: Hiç üçgen yok!\n";
        return;
    }

    const auto& material_indices = data.material_indices;

    if (material_indices.empty()) {
        std::cout << " Bilgi: Material indices boş, hepsi default 0 sayılacak.\n";
        return;
    }

    for (size_t tri_idx = 0; tri_idx < material_indices.size(); ++tri_idx) {
        int mat_idx = material_indices[tri_idx];

        if (mat_idx < 0 || mat_idx >= data.materials.size()) {
            std::cerr << " UYARI: Triangle [" << tri_idx << "] için geçersiz material index: " << mat_idx << "\n";
        }
    }

    std::cout << " Material indices doğrulandı! (" << material_indices.size() << " üçgen kontrol edildi)\n";
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
void OptixWrapper::buildFromData(const OptixGeometryData& data) {
    std::cout << " OptiX buildFromData başlıyor...\n";

    if (data.vertices.empty() || data.indices.empty()) {
        std::cerr << " Geometri verisi boş!\n";
        return;
    }

    // 1. Tüm geometri verilerini GPU'ya gönder
    size_t v_size = data.vertices.size() * sizeof(float3);
    cudaMalloc(reinterpret_cast<void**>(&d_vertices), v_size);
    cudaMemcpy(reinterpret_cast<void*>(d_vertices), data.vertices.data(), v_size, cudaMemcpyHostToDevice);

    size_t i_size = data.indices.size() * sizeof(uint3);
    cudaMalloc(reinterpret_cast<void**>(&d_indices), i_size);
    cudaMemcpy(reinterpret_cast<void*>(d_indices), data.indices.data(), i_size, cudaMemcpyHostToDevice);

    CUdeviceptr d_normals = 0;
    if (!data.normals.empty()) {
        size_t n_size = data.normals.size() * sizeof(float3);
        cudaMalloc(reinterpret_cast<void**>(&d_normals), n_size);
        cudaMemcpy(reinterpret_cast<void*>(d_normals), data.normals.data(), n_size, cudaMemcpyHostToDevice);
    }

    CUdeviceptr d_uvs = 0;
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
        std::cout << "Uyarı: Materyal indeksleri yeniden oluşturuldu.\n";
    }
    else {
        material_indices_ptr = data.material_indices.data();
    }

    CUdeviceptr d_material_indices;
    size_t mi_size = data.indices.size() * sizeof(int);
    cudaMalloc(reinterpret_cast<void**>(&d_material_indices), mi_size);
    cudaMemcpy(reinterpret_cast<void*>(d_material_indices), material_indices_ptr, mi_size, cudaMemcpyHostToDevice);

    // Tangent verilerini GPU'ya gönder
    CUdeviceptr d_tangents = 0;
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
    std::vector<SbtRecord<HitGroupData>> hitgroup_records;

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

    CUdeviceptr d_temp_buffer, d_output_buffer, d_compacted_size;
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
    std::cout << " OptiX buildFromData başarıyla tamamlandı! "
        << data.materials.size() << " materyal için SBT kaydı oluşturuldu.\n";
}
void OptixWrapper::launch_random_pixel_mode(
    SDL_Surface* surface, SDL_Window* window, int width, int height,
    std::vector<uchar4>& framebuffer
) {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    int max_threads_per_block = props.maxThreadsPerBlock;
    int num_sms = props.multiProcessorCount;
    int optimal_pixels = num_sms * max_threads_per_block * 5;
    int pixels_per_launch = std::min(optimal_pixels, width * height);

    const int samples_per_pixel = 100;
    params.samples_per_pixel = samples_per_pixel;

    // ------------------ FRAMEBUFFER SETUP -----------------------
    uchar4* d_framebuffer = nullptr;
    cudaMalloc(&d_framebuffer, width * height * sizeof(uchar4));
    params.framebuffer = d_framebuffer;
    params.image_width = width;
    params.image_height = height;
    params.handle = traversable_handle;
    params.background_color = make_float3(0.15f , 0.25f , 0.3f );
    params.materials = reinterpret_cast<GpuMaterial*>(d_materials);

    // Atmosfer
    params.atmosphere.sigma_s = 0.2f;
    params.atmosphere.sigma_a = 0.1f;
    params.atmosphere.g = 0.1f;
    params.atmosphere.base_density = 1.0f;
    params.atmosphere.temperature = 300.0f;
    params.atmosphere.active = false; // kapalı

    // ------------------ ADAPTIVE PARAMETRELER ------------------

    params.min_samples = 50;                 // minimum örnek
    params.variance_threshold = 0.002f;     // temel varyans
    // Frame sayısını arttır
    static int frame_counter = 1;
    params.frame_number = frame_counter;
    frame_counter++;

    // Temporal blend: yeni frame %95, eski %5 katkı
    params.temporal_blend = 0.95f;

    // ------------------ VARYANS & AKÜMÜLASYON BUFFERLARI ------------------
    static float* d_variance_buffer = nullptr;
    static float* d_accumulation_buffer = nullptr;
    static int* d_sample_count_buffer = nullptr;

    // Sadece ilk defa veya boyut değişince allocate edilir
    static int prev_width = 0, prev_height = 0;
    if (width != prev_width || height != prev_height) {
        if (d_variance_buffer) cudaFree(d_variance_buffer);
        if (d_accumulation_buffer) cudaFree(d_accumulation_buffer);
        if (d_sample_count_buffer) cudaFree(d_sample_count_buffer);

        cudaMalloc(&d_variance_buffer, sizeof(float) * width * height);
        cudaMalloc(&d_accumulation_buffer, sizeof(float) * width * height * 3); // RGB
        cudaMalloc(&d_sample_count_buffer, sizeof(int) * width * height);

        cudaMemset(d_variance_buffer, 0, sizeof(float) * width * height);
        cudaMemset(d_accumulation_buffer, 0, sizeof(float) * width * height * 3);
        cudaMemset(d_sample_count_buffer, 0, sizeof(int) * width * height);

        prev_width = width;
        prev_height = height;
    }

    params.variance_buffer = d_variance_buffer;
    params.accumulation_buffer = d_accumulation_buffer;
    params.sample_count_buffer = d_sample_count_buffer;

    // ------------------ PARAMS DEVICE KOPYALAMA ------------------
    CUdeviceptr d_params;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(RayGenParams));

    // ------------------ PİKSEL KOORDİNATLARI ------------------
    std::vector<std::pair<int, int>> all_coords;
    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i)
            all_coords.emplace_back(i, j);

    std::shuffle(all_coords.begin(), all_coords.end(), std::mt19937{ std::random_device{}() });

    std::vector<int> coords_x(pixels_per_launch);
    std::vector<int> coords_y(pixels_per_launch);

    CUdeviceptr d_coords_x, d_coords_y;
    cudaMalloc(reinterpret_cast<void**>(&d_coords_x), pixels_per_launch * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_coords_y), pixels_per_launch * sizeof(int));

    size_t total_pixels = all_coords.size();
    size_t rendered_pixels = 0;

    for (size_t offset = 0; offset < total_pixels; offset += pixels_per_launch) {
        size_t count = std::min<size_t>(pixels_per_launch, total_pixels - offset);

        for (size_t k = 0; k < count; ++k) {
            int index = offset + k;
            coords_x[k] = all_coords[index].first;
            coords_y[k] = all_coords[index].second;
        }

        cudaMemcpy(reinterpret_cast<void*>(d_coords_x), coords_x.data(), count * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(reinterpret_cast<void*>(d_coords_y), coords_y.data(), count * sizeof(int), cudaMemcpyHostToDevice);

        params.launch_coords_x = reinterpret_cast<int*>(d_coords_x);
        params.launch_coords_y = reinterpret_cast<int*>(d_coords_y);
        params.batch_pixel_count = static_cast<int>(count);

        cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(RayGenParams), cudaMemcpyHostToDevice);

        OPTIX_CHECK(optixLaunch(
            pipeline, stream,
            d_params, sizeof(RayGenParams),
            &sbt,
            static_cast<unsigned int>(count), 1, 1
        ));

        cudaStreamSynchronize(stream);

        cudaMemcpy(framebuffer.data(), d_framebuffer, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

        // ------------------ SDL GÜNCELLEME ------------------
        Uint32* pixel;
        int i, j;
        Vec3 final_color;
        for (j = 0; j < height; ++j) {
            for (i = 0; i < width; ++i) {
                const uchar4& c = framebuffer[j * width + i];

                Vec3 raw_color(c.x / 255.0f, c.y / 255.0f, c.z / 255.0f);
                final_color = color_processor.processColor(raw_color, i, j);

                int r = static_cast<int>(255.0f * std::clamp(final_color.x, 0.0f, 1.0f));
                int g = static_cast<int>(255.0f * std::clamp(final_color.y, 0.0f, 1.0f));
                int b = static_cast<int>(255.0f * std::clamp(final_color.z, 0.0f, 1.0f));

                pixel = (Uint32*)surface->pixels + (height - 1 - j) * surface->w + i;
                *pixel = SDL_MapRGB(surface->format, r, g, b);
            }
        }

        rendered_pixels += count;

        float percent = 100.0f * rendered_pixels / total_pixels;
        double elapsed = duration_cast<seconds>(
            high_resolution_clock::now() - start_time
        ).count();

        printf("  Progress: %.2f%%  |  Pixels: %zu/%zu  |  Time Elapsed: %.0f s\r",
            percent, rendered_pixels, total_pixels, elapsed);
        fflush(stdout);

        SDL_UpdateWindowSurface(window);
        SDL_PumpEvents();
    }

    // ------------------ OIDN DENOISE ------------------
    applyOIDNDenoising(surface, 0, true, 0.9f);
    SDL_UpdateWindowSurface(window);

    printf("\n Render tamamlandı. Toplam süre: %lld s\n",
        duration_cast<seconds>(high_resolution_clock::now() - start_time).count());

    if (SaveSurface(surface, "image/output_Optix.png")) {
        std::cout << "Image saved successfully!" << std::endl;
    }
    else {
        std::cerr << "Failed to save image." << std::endl;
    }

    // ------------------ TEMİZLİK ------------------
    cudaFree(reinterpret_cast<void*>(d_framebuffer));
    cudaFree(reinterpret_cast<void*>(d_params));
    cudaFree(reinterpret_cast<void*>(d_coords_x));
    cudaFree(reinterpret_cast<void*>(d_coords_y));
    // Not: variance, accumulation ve sample_count bufferları sabit tutuluyor (bir sonraki frame için reuse!)
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

        if (auto pointLight = std::dynamic_pointer_cast<PointLight>(light)) {
            const Vec3& pos = pointLight->position;
            const Vec3& inten = pointLight->intensity;
            l.position = make_float3(pos.x, pos.y, pos.z);
            l.intensity = make_float3(inten.x, inten.y, inten.z);
			l.radius = 0.1f;
            l.type = 0;
            l.intensity_magnitude = length(l.intensity);

        }
        else if (auto dirLight = std::dynamic_pointer_cast<DirectionalLight>(light)) {
            const Vec3 dir = (-dirLight->direction.normalize());
            const Vec3& inten = dirLight->intensity;
            l.direction = make_float3(dir.x, dir.y, dir.z);
            l.intensity = make_float3(inten.x, inten.y, inten.z);
            l.radius = 0.05f;
            l.type = 1;
            l.intensity_magnitude = length(l.intensity);
        }
        else if (auto areaLight = std::dynamic_pointer_cast<AreaLight>(light)) {
            const Vec3& pos = areaLight->position;
           /* const Vec3 inten = areaLight->getIntensity();
            l.position = make_float3(pos.x, pos.y, pos.z);
            l.intensity = make_float3(inten.x, inten.y, inten.z);
            l.type = 2;*/
            l.intensity_magnitude = length(l.intensity);
        }
        gpuLights.push_back(l);
    }

    // GPU'ya kopyala
    CUdeviceptr d_lights;
    cudaMalloc(reinterpret_cast<void**>(&d_lights), gpuLights.size() * sizeof(LightGPU));
    cudaMemcpy(reinterpret_cast<void*>(d_lights), gpuLights.data(), gpuLights.size() * sizeof(LightGPU), cudaMemcpyHostToDevice);

    params.lights = reinterpret_cast<LightGPU*>(d_lights);
    params.light_count = static_cast<int>(gpuLights.size());
}
bool OptixWrapper::SaveSurface(SDL_Surface* surface, const char* file_path) {
    SDL_Surface* surface_to_save = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB24, 0);
    /* int imgFlags = IMG_INIT_PNG;
     if (!(IMG_Init(imgFlags) & imgFlags)) {
         SDL_Log("SDL_image could not initialize! SDL_image Error: %s\n", IMG_GetError());
         SDL_Quit();
         return 1;
     }*/

    if (surface_to_save == NULL) {
        SDL_Log("Couldn't convert surface: %s", SDL_GetError());
        return false;
    }

    int result = IMG_SavePNG(surface_to_save, file_path);
    SDL_FreeSurface(surface_to_save);

    if (result != 0) {
        SDL_Log("Failed to save image: %s", IMG_GetError());
        return false;
    }

    return true;
}
