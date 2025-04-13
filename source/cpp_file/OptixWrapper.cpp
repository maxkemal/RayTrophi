#include "OptixWrapper.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

// Makro tanýmlamalarý
#define OPTIX_CHECK(call) \
    { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::cerr << "OptiX call " << #call << " failed: " << optixGetErrorString(res) << std::endl; \
            exit(1); \
        } \
    }

// Hata kontrol makrolarý
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)      \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA error");                       \
        }                                                                 \
    } while (0)


OptixWrapper::OptixWrapper() {
    // OptiX baþlatma
    setupOptiX();
}

OptixWrapper::~OptixWrapper() {
    // OptiX kaynaklarýný temizle
    if (pipeline) {
        optixPipelineDestroy(pipeline);
        pipeline = nullptr;
    }



    if (pipeline) {
        // SBT'nin kaynaklarýný serbest býrak
        cudaFree(reinterpret_cast<void*>(sbt.raygenRecord));
        cudaFree(reinterpret_cast<void*>(sbt.missRecordBase));
        cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase));
        sbt = {}; // SBT'yi sýfýrla
    }

    if (optixContext) {
        optixDeviceContextDestroy(optixContext);
        optixContext = nullptr;
    }


}
void OptixWrapper::createSBT() {

}


void OptixWrapper::setupOptiX() {
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0)); // Varsayýlan CUDA cihazýný ayarla
    CUDA_CHECK(cudaFree(0));      // CUDA runtime'ý baþlat

    // Initialize OptiX
    OPTIX_CHECK(optixInit());

    // Create OptiX device context
    OptixDeviceContextOptions options = {};
    CUcontext cuCtx = 0;  // Kullanýlacak CUDA baðlamý; mevcut baðlamý belirtirseniz bu deðeri ayarlayýn
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optixContext));



    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    // PTX kodunu derle
    const char* PTX_STRING = R"(
.version 8.0
.target sm_50
.address_size 64

.entry __raygen__rg()
{
  // Basit bir raygen programý
  .reg .b32 %r0;
  mov.u32 %r0, 0;
  ret;
}

.entry __miss__ms()
{
  // Basit bir miss programý
  ret;
}

.entry __closesthit__ch()
{
  // Basit bir closest hit programý
  ret;
}

.entry __anyhit__ah()
{
  // Basit bir any hit programý
  ret;
}
)";

    OptixModule module = nullptr;
    char log[2048];  // Log buffer boyutunu artýrýn
    size_t sizeof_log = sizeof(log);

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OPTIX_CHECK(optixModuleCreate(  // veya yeni fonksiyon adý
        optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        PTX_STRING,
        strlen(PTX_STRING),
        log,
        &sizeof_log,
        &module
    ));
    if (sizeof_log > 1) {
        std::cerr << "OptiX module creation log: " << log << std::endl;
    }
    // Create program groups
    OptixProgramGroup raygenPG, missPG, hitgroupPG;
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};

    // Raygen program group
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &raygenPG));

    // Miss program group (optional)
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &missPG));

    // Hitgroup program group (optional)
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &hitgroupPG));

    // Create pipeline
    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;
    OPTIX_CHECK(optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions, &raygenPG, 1, log, &sizeof_log, &pipeline));

    // Create shader binding table (SBT)
    OptixShaderBindingTable sbt = {};
    CUdeviceptr raygen_record, miss_record, hitgroup_record;
    const size_t record_size = sizeof(RayGenSbtRecord);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), record_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), record_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), record_size));

    // Pack SBT records
    RayGenSbtRecord rg_sbt;
    MissSbtRecord ms_sbt;
    HitGroupSbtRecord hg_sbt;

    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rg_sbt));
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &ms_sbt));
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG, &hg_sbt));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, record_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt, record_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), &hg_sbt, record_size, cudaMemcpyHostToDevice));

    // Set SBT entries
    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = hitgroup_record;
    sbt.hitgroupRecordCount = 1;

    // Save SBT and pipeline
    this->sbt = sbt;
    this->pipeline = pipeline;
}



void OptixWrapper::buildAccelerationStructure(const std::vector<std::shared_ptr<Hittable>>& objects) {
    // OptiX hýzlandýrma yapýsý oluþturulacak
}

void OptixWrapper::createModule() {
    // PTX modülü yükleniyor
}

void OptixWrapper::createPipeline() {
    // Pipeline oluþturuluyor
}


// Vec3'ten float3'e dönüþüm
inline float3 vec3ToFloat3(const Vec3& v) {
    return make_float3(v.x, v.y, v.z);
}

// float3'ten Vec3'e dönüþüm
inline Vec3 float3ToVec3(const float3& v) {
    return Vec3(v.x, v.y, v.z);
}
OptixHitResult OptixWrapper::performOptixIntersectionTest(const float3& origin, const float3& direction, float t_min, float t_max) const {
    OptixHitResult result = {};
    OptixRay ray = { origin, direction, t_min, t_max };
    HitRecord rec;
    float hit_t = 1.0f;  // Bu test için hardcoded, sonradan hesaplanmalý.
    int hit_objectID = 1;

    optixLaunch(
        pipeline,
        stream,
        0,
        0,
        &sbt,
        1280,
        720,
        1
    );
    // OptiX'ten elde edilen materyali kontrol et
    if (result.hasHit) {
        // OptiX sonucuna göre HitRecord güncelleme
        rec.t = result.t;
        rec.point = float3ToVec3(result.hitPoint);
        rec.normal = float3ToVec3(result.normal);
        rec.material = result.material ; // Materyal kontrolü

        // Sonuçlarý OptiXHitResult'e kaydet
        result.hasHit = rec.material != nullptr; // Örneðin, hasHit'e material var mý kontrolü
        result.t = rec.t;
        result.hitObjectID = result.hitObjectID; // ID'yi güncelle
    }

    return result;
}

bool OptixWrapper::hit_with_optix(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    float3 origin = make_float3(r.origin.x, r.origin.y, r.origin.z);
    float3 direction = make_float3(r.direction.x, r.direction.y, r.direction.z);

    OptixHitResult result = performOptixIntersectionTest(origin, direction, t_min, t_max);

    if (result.hasHit) {
        rec.t = result.t;
        rec.point = Vec3(result.hitPoint.x, result.hitPoint.y, result.hitPoint.z);
        rec.normal = Vec3(result.normal.x, result.normal.y, result.normal.z);
        // Materyali ayarla
        rec.material = result.material;
        return true;
    }

    return false;
}

void OptixWrapper::handleCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << file << ":" << line << std::endl;
        throw std::runtime_error("CUDA Error");
    }
}

void OptixWrapper::handleOptixError(OptixResult result, const char* file, int line) {
    if (result != OPTIX_SUCCESS) {
        std::cerr << "OptiX Error: " << optixGetErrorString(result) << " at " << file << ":" << line << std::endl;
        throw std::runtime_error("OptiX Error");
    }
}

void OptixWrapper::handleCUresult(CUresult result, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(result, &errorStr);
        std::cerr << "CUDA Driver API Error: " << errorStr << " at " << file << ":" << line << std::endl;
        throw std::runtime_error("CUDA Driver API Error");
    }
}
void OptixWrapper::cleanup() {
    if (optixContext) {
        // OptiX kaynaklarýný serbest býrak
        optixDeviceContextDestroy(optixContext);
        optixContext = nullptr;
    }
    cudaFree(reinterpret_cast<void*>(sbt.raygenRecord));
    cudaFree(reinterpret_cast<void*>(sbt.missRecordBase));
    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(miss_prog_group);
    optixProgramGroupDestroy(raygen_prog_group);
  
    // Diðer temizleme iþlemleri
}