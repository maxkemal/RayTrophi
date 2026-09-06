#pragma once
#include "PostProcess/Exposure.h"
#include "globals.h"
#include <cstring>
namespace {
__global__ void postHistogramKernel(const float* hdr,int width,int height,int stride,float centerWeight,uint32_t* out) {
    __shared__ uint32_t bins[256];unsigned lane=threadIdx.x;bins[lane]=0;__syncthreads();
    int nx=min(width,128),ny=min(height,128);
    for(unsigned i=lane;i<unsigned(nx*ny);i+=256) {
        int qx=i%nx,qy=i/nx,x=(2*qx+1)*width/(2*nx),y=(2*qy+1)*height/(2*ny);
        const float* p=hdr+(size_t(y)*width+x)*stride;
        if(!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2]))continue;
        float lum=.2126f*fmaxf(p[0],0)+.7152f*fmaxf(p[1],0)+.0722f*fmaxf(p[2],0);
        if(lum<=1e-8f)continue;
        int bin=max(0,min(255,int((log2f(lum)+16)*8)));
        float u=(qx+.5f)/nx*2-1,v=(qy+.5f)/ny*2-1,center=fmaxf(0,1-.5f*(u*u+v*v));
        atomicAdd(&bins[bin],uint32_t(1+255*((1-centerWeight)+centerWeight*center*center)));
    }
    __syncthreads();out[lane]=bins[lane];
}
struct CudaPostMeter {
    uint32_t* deviceBins=nullptr;
    uint32_t* hostBins=nullptr;
    cudaEvent_t complete=nullptr;
    bool pending=false;
    int device=-1;
    uint64_t generation=0;
    ~CudaPostMeter() { release(); }
    void release() {
        if(device<0)return;
        int previous=-1;cudaGetDevice(&previous);cudaSetDevice(device);
        if(pending)cudaEventSynchronize(complete);
        if(complete)cudaEventDestroy(complete);
        if(deviceBins)cudaFree(deviceBins);
        if(hostBins)cudaFreeHost(hostBins);
        if(previous>=0)cudaSetDevice(previous);
        complete=nullptr;deviceBins=nullptr;hostBins=nullptr;pending=false;device=-1;
    }
    void meter(const float* hdr,int w,int h,int stride,cudaStream_t stream) {
        int active=-1;if(cudaGetDevice(&active)!=cudaSuccess)return;
        if(device>=0 && device!=active)release();
        if(pending) {
            const auto status=cudaEventQuery(complete);
            if(status==cudaErrorNotReady)return;
            if(status!=cudaSuccess){release();return;}
            rtpost::MeterResult result;result.generation=generation;
            std::memcpy(result.bins.data(),hostBins,1024);pending=false;
            rtpost::submitMeter(result,"CUDA GPU histogram");
        }
        if(!rtpost::meterEnabled())return;
        if(!complete) {
            device=active;
            if(cudaMalloc(reinterpret_cast<void**>(&deviceBins),1024)!=cudaSuccess ||
               cudaMallocHost(reinterpret_cast<void**>(&hostBins),1024)!=cudaSuccess ||
               cudaEventCreateWithFlags(&complete,cudaEventDisableTiming)!=cudaSuccess){release();return;}
        }
        generation=rtpost::meterGeneration();
        postHistogramKernel<<<1,256,0,stream>>>(hdr,w,h,stride,rtpost::meterSettings().center_weight,deviceBins);
        if(cudaGetLastError()!=cudaSuccess)return;
        if(cudaMemcpyAsync(hostBins,deviceBins,1024,cudaMemcpyDeviceToHost,stream)!=cudaSuccess)return;
        if(cudaEventRecord(complete,stream)==cudaSuccess)pending=true;
    }
};
}
void launchPostHistogram(const float* hdr,int width,int height,int stride,cudaStream_t stream) {
    if(!hdr || width<=0 || height<=0 || stride<3)return;
    // One pending 1 KB transfer per calling render thread. Never blocks the frame
    // to read telemetry. Device switch / teardown is the only drain operation.
    static thread_local CudaPostMeter meter;
    meter.meter(hdr,width,height,stride,stream);
}

namespace {
__global__ void postFloat4Kernel(const float4* hdr,uchar4* display,int width,int height,
    OidnPostParamsDevice p,float lensAmount,float lensFalloff) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width || y>=height)return;
    size_t i=size_t(y)*width+x;
    float4 h=hdr[i];float3 c=make_float3(h.x,h.y,h.z);
    if(lensAmount>0) {
        float u=float(x)/width*2-1,v=float(y)/height*2-1;
        float f=fmaxf(0,1-lensAmount*powf(sqrtf(u*u+v*v)/1.414f,lensFalloff));
        c.x*=f;c.y*=f;c.z*=f;
    }
    c=rtApplyPost(c,p,(x+.5f)/width,(y+.5f)/height);
    display[i]=make_uchar4(uint8_t(c.x*255+.5f),uint8_t(c.y*255+.5f),uint8_t(c.z*255+.5f),display[i].w);
}
}
bool launchOptixDisplayPost(const void* hdr,void* display,int width,int height,cudaStream_t stream,float lensAmount,float lensFalloff) {
    if(!hdr || !display || width<=0 || height<=0)return false;
    launchPostHistogram(static_cast<const float*>(hdr),width,height,4,stream);
    const auto& p=g_display_post;
    OidnPostParamsDevice dp{p.exposure,p.camera_exposure,p.gamma,p.saturation,p.color_temperature,p.vignette_strength,
        uint32_t(p.tone_mapping),uint32_t(p.vignette_enabled)};
    postFloat4Kernel<<<dim3((width+15)/16,(height+15)/16),dim3(16,16),0,stream>>>(
        static_cast<const float4*>(hdr),static_cast<uchar4*>(display),width,height,dp,lensAmount,lensFalloff);
    return cudaGetLastError()==cudaSuccess;
}
