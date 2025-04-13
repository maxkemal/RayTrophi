#include <iostream>
#include <vector>
#include "Hittable.h"
#include "Ray.h"
#include <cuda.h>  
#include <cuda_runtime.h>                // CUDA runtime API  
#include <optix.h>

struct OptixHitResult {
    bool hasHit;
    float t;
    int hitObjectID;
    float3 hitPoint;
    float3 normal;
    std::shared_ptr<Material> material;
};

// CUDA runtime API  
class OptixWrapper {
public:
    OptixWrapper();
    ~OptixWrapper();

    void buildAccelerationStructure(const std::vector<std::shared_ptr<Hittable>>& objects);
    void createModule();
    void createPipeline();
    void initOptixFunctionTable();
    void createSBT();
    void setupOptiX();
    OptixHitResult performOptixIntersectionTest(const float3& origin, const float3& direction, float t_min, float t_max) const;

    bool hit_with_optix(const Ray& r, double t_min, double t_max, HitRecord& rec) const;

    // Diðer public metodlar...

private:
    struct OptixRay {
        float3 origin;
        float3 direction;
        float tmin;
        float tmax;
    };
    struct RayGenSbtRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // RayGen programýnýz için gerekli diðer veriler...
    };
    OptixDeviceContext optixContext;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    CUstream cudaStream;
    OptixProgramGroup raygen_prog_group;
    OptixProgramGroup miss_prog_group;
    OptixProgramGroup hitgroup_prog_group;
    CUstream stream;
    CUdeviceptr d_gas_output_buffer;
    OptixTraversableHandle gas_handle;


    OptixPipelineCompileOptions pipelineCompileOptions;


    // Hata kontrol yardýmcý fonksiyonlarý
    static void handleCudaError(cudaError_t error, const char* file, int line);
    static void handleOptixError(OptixResult result, const char* file, int line);
    static void handleCUresult(CUresult result, const char* file, int line);

    void cleanup();

    struct HitGroupSbtRecord {
        // ... Hit group verileri
    };

    struct MissSbtRecord {
        // ... Miss verileri
    };
    // Diðer private üyeler ve metodlar...
};


// Makrolar
#define CUDA_CHECK( call )  OptixWrapper::handleCudaError( call, __FILE__, __LINE__ )
#define OPTIX_CHECK( call ) OptixWrapper::handleOptixError( call, __FILE__, __LINE__ )
#define CU_CHECK( call )    OptixWrapper::handleCUresult( call, __FILE__, __LINE__ )