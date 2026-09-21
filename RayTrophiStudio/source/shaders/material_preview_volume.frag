#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Realtime participating-media pass. It deliberately consumes the same raw
// 624-byte VkVolumeInstance table as Vulkan RT (binding 20): simulation, VDB,
// scripting and both renderers therefore have one volume-data authority.

layout(location = 0) in vec2 vNdc;
layout(location = 0) out vec4 outColor;

#include "post_chain.glsl"
#include "material_preview_ray.glsl"

layout(set = 0, binding = 2) uniform sampler2D envMaps[2];
layout(set = 0, binding = 9) uniform sampler2D worldEnvironment;
layout(set = 0, binding = 13) uniform sampler2D worldIrradiance;
layout(set = 0, binding = 18) uniform sampler2D previewOpaqueDepth;

struct LightData {
    vec4 position;
    vec4 color;
    vec4 params;
    vec4 direction;
    vec4 area_u;
    vec4 area_v;
};
layout(set = 0, binding = 5, std430) readonly buffer PreviewLightBuffer {
    LightData sceneLights[];
};
layout(set = 0, binding = 6, std430) readonly buffer PreviewSceneGlobalsBuffer {
    uint sceneLightCount;
    uint sceneFlags;
    uint shadowedLightCount;
    uint worldMode;
    vec4 worldColor;
    vec4 worldParams;
    vec4 worldSun;
    vec4 atmosphereA;
    vec4 atmosphereB;
    vec4 postA;
    vec4 postB;
    vec4 postC;
};

#include "material_preview_shadow_data.glsl"

layout(set = 0, binding = 20, std430) readonly buffer PreviewVolumeRawBuffer {
    uint volumeWords[];
};

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;
    // lightDir0.w: opaque depth snapshot is valid
    vec4 lightDir0;
    vec4 lightDir1;
    vec4 lightDir2;
    // x=material count, y=quality, z=lighting preset, w=volume count
    uvec4 materialMeta;
} pc;

const float PI = 3.14159265358979323846;
const float INV_4PI = 0.07957747154594767;
const uint MAX_VOLUMES = 16u;
#include "material_preview_volume_fields.glsl"

float phaseHG(float c,float g){
    float g2=g*g, den=1.0+g2-2.0*g*c;
    return INV_4PI*(1.0-g2)/(den*sqrt(max(den,1e-6)));
}
float phaseDual(float c,uint vi){
    return mix(phaseHG(c,volFloat(vi,108u)),phaseHG(c,volFloat(vi,104u)),
               clamp(volFloat(vi,112u),0.0,1.0));
}

vec3 blackbody(float kelvin){
    float t=clamp(kelvin,1000.0,40000.0)/100.0;
    float r=t<=66.0?1.0:clamp(329.698727446*pow(t-60.0,-0.1332047592)/255.0,0.0,1.0);
    float g=t<=66.0?clamp((99.4708025861*log(t)-161.1195681661)/255.0,0.0,1.0):
                         clamp(288.1221695283*pow(t-60.0,-0.0755148492)/255.0,0.0,1.0);
    float b=t>=66.0?1.0:(t<=19.0?0.0:clamp((138.5177312231*log(t-10.0)-305.0447927307)/255.0,0.0,1.0));
    return vec3(r,g,b);
}
vec3 rampColor(uint vi,float t){
    int count=clamp(volInt(vi,276u),0,8);
    if(count<=0)return vec3(1);
    vec3 prev=vec3(volFloat(vi,320u),volFloat(vi,352u),volFloat(vi,384u));
    float prevP=volFloat(vi,288u);
    if(t<=prevP)return prev;
    for(int i=1;i<8;++i){
        if(i>=count)break;
        uint o=uint(i)*4u;
        float p=volFloat(vi,288u+o);
        vec3 c=vec3(volFloat(vi,320u+o),volFloat(vi,352u+o),volFloat(vi,384u+o));
        if(t<=p)return mix(prev,c,clamp((t-prevP)/max(p-prevP,1e-6),0.0,1.0));
        prevP=p;prev=c;
    }
    return prev;
}

vec3 emissionAt(uint vi,vec3 p,float density){
    int mode=volInt(vi,256u);
    if(mode<=0)return vec3(0);
    if(mode==1)return volVec3(vi,136u)*volFloat(vi,148u)*density;
    uint64_t tempAddress=volAddress(vi,240u);
    uint64_t flameAddress=volAddress(vi,536u);
    bool hasTemp=tempAddress!=0ul;
    float temp=hasTemp?sampleRawField(vi,tempAddress,p):density;
    if(hasTemp && volInt(vi,168u)==4 && volInt(vi,428u)==5)temp*=3000.0;
    // An authored temperature/flame field is authoritative: empty/cold cells
    // must not glow through the legacy density fallback.
    float flame=flameAddress!=0ul?clamp(sampleRawField(vi,flameAddress,p),0.0,1.0):0.0;
    if(hasTemp && temp<=0.0 && flame<=0.0)return vec3(0);
    float lo=max(volFloat(vi,488u),0.0);
    float hi=volFloat(vi,268u)>lo+1.0?volFloat(vi,268u):lo+1500.0;
    float kelvin=temp>20.0?clamp(temp,lo,hi):mix(lo,hi,clamp(temp,0.0,1.0));
    float u=clamp((kelvin-lo)/max(hi-lo,1.0),0.0,1.0);
    float scale=max(volFloat(vi,260u),0.001);
    vec3 c=volInt(vi,272u)!=0?rampColor(vi,clamp(u*scale,0.0,1.0)):blackbody(kelvin*scale);
    vec3 e=c*density*volFloat(vi,264u)*(1.0+flame);
    float l=dot(e,vec3(0.2126,0.7152,0.0722));
    return l>64.0?e*(64.0/l):e;
}

bool evalLight(uint index,vec3 p,out vec3 l,out vec3 radiance){
    LightData light=sceneLights[index];
    int type=int(light.position.w+0.5);
    radiance=light.color.rgb*light.color.w;
    if(type==1){l=normalize(light.direction.xyz);return true;}
    vec3 to=light.position.xyz-p; float d2=dot(to,to);
    if(d2<1e-6)return false;
    l=to*inversesqrt(d2); float atten=1.0/d2;
    if(type==2){vec3 n=normalize(cross(light.area_u.xyz,light.area_v.xyz));atten*=max(dot(-l,n),0.0)*max(light.params.y*light.params.z,0.0);}
    else if(type==3){float c=dot(-l,normalize(light.direction.xyz));atten*=smoothstep(light.direction.w,light.params.z,c);}
    radiance*=atten; return atten>0.0;
}

vec2 directionUv(vec3 d){d=normalize(d);return vec2(atan(d.z,d.x)/(2.0*PI)+0.5,acos(clamp(d.y,-1.0,1.0))/PI);}
vec3 ambientLight(vec3 d){
    uint preset=pc.materialMeta.z;
    if(preset==3u){
        if(worldMode==0u)return worldColor.rgb*worldParams.x;
        return textureLod(worldIrradiance,directionUv(d),0.0).rgb*worldParams.y;
    }
    return textureLod(envMaps[preset==2u?1:0],directionUv(d),5.0).rgb*0.35;
}

float selfShadow(uint vi,vec3 p,vec3 l,float maxDistance){
    float nearT,farT;
    if(!rayVolumeInterval(vi,p+l*1e-4,l,nearT,farT))return 1.0;
    float endT=min(farT,maxDistance);
    if(endT<=0.0)return 1.0;
    int requested=clamp(volInt(vi,160u),1,64);
    int cap=(pc.materialMeta.y&0xffu)<=1u?4:((pc.materialMeta.y&0xffu)==2u?8:12);
    int steps=min(requested,cap); float dt=endT/float(steps), tau=0.0;
    float sigma=max(volFloat(vi,100u)+volFloat(vi,132u),0.0);
    for(int i=0;i<12;++i){if(i>=steps)break;tau+=sampleField(vi,volAddress(vi,232u),p+l*(float(i)+0.5)*dt)*sigma*dt;}
    return exp(-tau*clamp(volFloat(vi,164u),0.0,1.0));
}

RtPostParams volumePostParams(){
    RtPostParams p; p.exposure=postA.x;p.gamma=postA.y;p.saturation=postA.z;
    p.colorTemperature=postA.w;p.vignetteStrength=postB.x;
    p.toneMapping=uint(postB.y+0.5);p.vignetteEnabled=uint(postB.z+0.5);
    p.cameraExposure=postB.w;return p;
}

void main(){
    // NDC comes from the fullscreen triangle, including when no depth
    // snapshot exists. A fallback 1x1 depth texture must not define the ray.
    vec2 uv=vNdc*0.5+0.5;
    mat4 invVP=inverse(pc.viewProj);
    vec3 ro,rd;
    rtPreviewWorldRay(invVP,vNdc,ro,rd);
    float visibleT=1e30;
    if(pc.lightDir0.w>0.5){
        float z=texture(previewOpaqueDepth,uv).r;
        if(z<0.999999){vec4 h=invVP*vec4(uv*2.0-1.0,z,1.0);visibleT=max(dot(h.xyz/h.w-ro,rd),0.0);}
    }

    uint ids[16]; float starts[16]; float ends[16]; int count=0;
    uint total=min(pc.materialMeta.w,MAX_VOLUMES);
    for(uint vi=0u;vi<MAX_VOLUMES;++vi){
        if(vi>=total)break;
        if(volInt(vi,172u)==0 || volInt(vi,428u)==4)continue;
        float a,b;if(!rayVolumeInterval(vi,ro,rd,a,b))continue;
        a=max(a,0.0);b=min(b,visibleT);if(b<=a)continue;
        int at=count;
        while(at>0 && starts[at-1]>a){ids[at]=ids[at-1];starts[at]=starts[at-1];ends[at]=ends[at-1];--at;}
        ids[at]=vi;starts[at]=a;ends[at]=b;++count;
    }
    if(count==0)discard;

    vec3 accum=vec3(0); vec3 trans=vec3(1);
    uint quality=pc.materialMeta.y&0xffu;
    // ★★★ THIS IS A COST CEILING, NOT THE STEP COUNT.
    //
    // MEASURED 2026-09-20: it used to be 48/96/160 and the line below re-clamped
    // `steps` to it AFTER honouring the authored step, so `dt` always came back
    // as length/cap. The authored step could only ever be raised, never lowered,
    // and refining a gas domain from 124 to 247 cells changed the image by
    // nothing at all — silhouette diff 20.5/255 between the two top presets
    // proves the march had not converged even at 160. The volume's own
    // max_steps is the real dial; this only stops one volume eating the frame.
    int cap=quality<=1u?96:(quality==2u?256:512);
    for(int interval=0;interval<16;++interval){
        if(interval>=count)break;
        uint vi=ids[interval];float length=ends[interval]-starts[interval];
        int budget=min(clamp(volInt(vi,156u),1,2048),cap);
        // Honour the authored step; stretch it only when covering the interval
        // would cost more than the budget. `steps` can then never truncate the
        // march, because skipped empty distance does not consume it.
        float dt=max(volFloat(vi,152u),length/float(budget));
        int steps=min(int(ceil(length/dt)),budget);
        float jitter=hash31(vec3(gl_FragCoord.xy,float(vi)));
        float cachedShadows[9];for(int k=0;k<9;++k)cachedShadows[k]=-1.0;
        int stride=clamp(volInt(vi,180u),1,16);
        // Live dense gas publishes a per-block density maximum. Walking the
        // empty part of the bounding box used to burn the step budget that the
        // cloud needed: the plume filled ~4% of the domain cells, so most
        // samples resolved nothing and still advanced the ray.
        float cutoff=max(volFloat(vi,248u),0.0);
        bool allowSkip=volInt(vi,168u)==4 && volInt(vi,428u)==5;
        float t=starts[interval]+jitter*dt;
        float tEnd=ends[interval];
        int shaded=0, iters=0, maxIters=steps*3+16;
        while(t<tEnd && shaded<steps && iters<maxIters){
            ++iters;
            if(max(max(trans.r,trans.g),trans.b)<0.01)break;
            vec3 p=ro+rd*t;
            if(allowSkip){
                float skip=denseEmptyBlockStep(vi,p,rd,dt,cutoff);
                if(skip>dt*1.01){t+=skip;continue;}
            }
            float density=sampleField(vi,volAddress(vi,232u),p);
            if(density<=0.0){t+=dt;continue;}
            int s=shaded; ++shaded; t+=dt;
            vec3 sigmaS=max(volVec3(vi,88u),vec3(0))*max(volFloat(vi,100u),0.0)*density;
            vec3 sigmaA=max(volVec3(vi,120u),vec3(0))*max(volFloat(vi,132u),0.0)*density;
            vec3 sigmaT=max(sigmaS+sigmaA,vec3(1e-6));
            vec3 sampleT=exp(-sigmaT*dt), oneMinusT=vec3(1)-sampleT;
            vec3 scatterWeight=oneMinusT*sigmaS/sigmaT;
            vec3 light=ambientLight(-rd)*0.18;
            uint lights=min(sceneLightCount,quality<=1u?2u:(quality==2u?4u:8u));
            for(uint li=0u;li<8u;++li){
                if(li>=lights)break;vec3 l,rad;if(!evalLight(li,p,l,rad))continue;
                if((s%stride)==0 || cachedShadows[li]<0.0){
                    bool deep=(sceneFlags&1u)!=0u && deepMeta.z!=0u && shadowRecords[li].meta.x!=0u;
                    float maxD=int(sceneLights[li].position.w+0.5)==1?1e30:distance(sceneLights[li].position.xyz,p);
                    cachedShadows[li]=rtPreviewShadow(li,p,vec3(0),l,sceneLights[li].position.xyz,quality);
                    if(!deep)cachedShadows[li]*=selfShadow(vi,p,l,maxD);
                }
                light+=rad*phaseDual(dot(rd,l),vi)*cachedShadows[li];
            }
            if(pc.materialMeta.z==3u && worldMode==2u && worldSun.w>0.0){
                vec3 l=normalize(worldSun.xyz);
                if((s%stride)==0 || cachedShadows[8]<0.0){
                    bool deep=(sceneFlags&1u)!=0u && deepMeta.z!=0u && shadowRecords[32].meta.x!=0u;
                    cachedShadows[8]=rtPreviewShadow(32u,p,vec3(0),l,vec3(0),quality);
                    if(!deep)cachedShadows[8]*=selfShadow(vi,p,l,1e30);
                }
                light+=vec3(1.0,0.95,0.86)*worldSun.w*phaseDual(dot(rd,l),vi)*cachedShadows[8];
            }
            vec3 emis=emissionAt(vi,p,density);
            accum+=trans*(scatterWeight*light + oneMinusT*emis);
            trans*=sampleT;
        }
    }
    float alpha=clamp(1.0-dot(trans,vec3(0.2126,0.7152,0.0722)),0.0,1.0);
    if(alpha<1e-5)discard;
    // *** GORUNTULEME DONUSUMU BURADAN SOKULDU (2026-09-06).
    //   Bu shader artik SCENE-LINEAR yaziyor; zincir (exposure -> operator ->
    //   grade -> vignette -> sRGB) tek yerde, `raster_post.comp` icinde kosuyor.
    //   Zorunluydu: alan derinligi bu shader'larin ciktisini BULANISTIRIR ve
    //   bokeh, parlak noktanin daire olarak acilmasidir. Tonemap'ten gecmis bir
    //   deger o noktayi zaten kirpmistir; onu bulanistirmak gri leke uretir.
    outColor=vec4(accum,alpha);
}
