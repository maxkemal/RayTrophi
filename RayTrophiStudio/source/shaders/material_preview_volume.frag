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
const uint VOLUME_STRIDE_WORDS = 156u;
const uint MAX_VOLUMES = 16u;

uint volWord(uint vi, uint byteOffset) {
    return volumeWords[vi * VOLUME_STRIDE_WORDS + (byteOffset >> 2u)];
}
float volFloat(uint vi, uint byteOffset) {
    return uintBitsToFloat(volWord(vi, byteOffset));
}
int volInt(uint vi, uint byteOffset) {
    return int(volWord(vi, byteOffset));
}
uint64_t volAddress(uint vi, uint byteOffset) {
    return uint64_t(volWord(vi, byteOffset)) |
           (uint64_t(volWord(vi, byteOffset + 4u)) << 32);
}
vec3 volVec3(uint vi, uint byteOffset) {
    return vec3(volFloat(vi, byteOffset), volFloat(vi, byteOffset + 4u),
                volFloat(vi, byteOffset + 8u));
}
vec3 transformPoint(uint vi, uint base, vec3 p) {
    return vec3(
        volFloat(vi, base) * p.x + volFloat(vi, base+4u) * p.y +
            volFloat(vi, base+8u) * p.z + volFloat(vi, base+12u),
        volFloat(vi, base+16u) * p.x + volFloat(vi, base+20u) * p.y +
            volFloat(vi, base+24u) * p.z + volFloat(vi, base+28u),
        volFloat(vi, base+32u) * p.x + volFloat(vi, base+36u) * p.y +
            volFloat(vi, base+40u) * p.z + volFloat(vi, base+44u));
}
vec3 transformVector(uint vi, uint base, vec3 p) {
    return vec3(
        volFloat(vi, base) * p.x + volFloat(vi, base+4u) * p.y + volFloat(vi, base+8u) * p.z,
        volFloat(vi, base+16u) * p.x + volFloat(vi, base+20u) * p.y + volFloat(vi, base+24u) * p.z,
        volFloat(vi, base+32u) * p.x + volFloat(vi, base+36u) * p.y + volFloat(vi, base+40u) * p.z);
}

bool rayVolumeInterval(uint vi, vec3 ro, vec3 rd, out float tNear, out float tFar) {
    vec3 o = transformPoint(vi, 184u, ro);
    vec3 d = transformVector(vi, 184u, rd);
    vec3 safe = vec3(
        abs(d.x) > 1e-8 ? d.x : (d.x < 0.0 ? -1e-8 : 1e-8),
        abs(d.y) > 1e-8 ? d.y : (d.y < 0.0 ? -1e-8 : 1e-8),
        abs(d.z) > 1e-8 ? d.z : (d.z < 0.0 ? -1e-8 : 1e-8));
    vec3 a = (volVec3(vi, 48u) - o) / safe;
    vec3 b = (volVec3(vi, 60u) - o) / safe;
    vec3 lo = min(a, b), hi = max(a, b);
    tNear = max(max(lo.x, lo.y), lo.z);
    tFar = min(min(hi.x, hi.y), hi.z);
    return tFar > max(tNear, 0.0);
}

// NanoVDB buffer-reference reader, matching volume_closesthit.rchit.
#define PNANOVDB_GLSL
#define PNANOVDB_BUF_CUSTOM
struct pnanovdb_buf_t { uint64_t address; };
layout(buffer_reference, std430, buffer_reference_align=4) buffer NanoVDBBlock {
    uint data[];
};
uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint offset) {
    return NanoVDBBlock(buf.address).data[offset >> 2u];
}
uvec2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint offset) {
    uint i = offset >> 2u;
    return uvec2(NanoVDBBlock(buf.address).data[i],
                 NanoVDBBlock(buf.address).data[i + 1u]);
}
void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint offset, uint value) {}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint offset, uvec2 value) {}
#include "PNanoVDB.h"

float sampleNano(uint64_t address, vec3 p) {
    if (address == 0ul) return 0.0;
    pnanovdb_buf_t buf; buf.address = address;
    pnanovdb_grid_handle_t grid; grid.address.byte_offset = 0u;
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
    pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
    pnanovdb_readaccessor_t acc; pnanovdb_readaccessor_init(acc, root);
    pnanovdb_vec3_t wp = pnanovdb_vec3_uniform(0.0);
    wp.x=p.x; wp.y=p.y; wp.z=p.z;
    pnanovdb_vec3_t ip = pnanovdb_map_apply_inverse(buf, map, wp);
    vec3 q = vec3(ip.x, ip.y, ip.z), f = fract(q);
    ivec3 base = ivec3(floor(q));
    float d[8];
    for (int c=0; c<8; ++c) {
        pnanovdb_coord_t coord;
        coord.x=base.x+((c&1)!=0?1:0);
        coord.y=base.y+((c&2)!=0?1:0);
        coord.z=base.z+((c&4)!=0?1:0);
        pnanovdb_address_t a = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, acc, coord);
        d[c] = pnanovdb_read_float(buf, a);
    }
    return mix(mix(mix(d[0],d[1],f.x),mix(d[2],d[3],f.x),f.y),
               mix(mix(d[4],d[5],f.x),mix(d[6],d[7],f.x),f.y),f.z);
}

layout(buffer_reference, std430, buffer_reference_align=4)
readonly buffer DenseGasFloatGrid { float values[]; };

float sampleDense(uint vi, uint64_t address, vec3 local) {
    if (address == 0ul) return 0.0;
    ivec3 res = ivec3(int(volFloat(vi,464u)+0.5),
                      int(volFloat(vi,468u)+0.5),
                      int(volFloat(vi,472u)+0.5));
    if (any(lessThanEqual(res, ivec3(0)))) return 0.0;
    vec3 origin = volVec3(vi, 476u);
    float voxel = max(volFloat(vi,176u), 1e-6);
    vec3 gp = (local-origin)/voxel-vec3(0.5);
    if (any(lessThan(gp,vec3(-0.5))) || any(greaterThan(gp,vec3(res)-vec3(0.5)))) return 0.0;
    ivec3 p0=clamp(ivec3(floor(gp)),ivec3(0),res-ivec3(1));
    ivec3 p1=min(p0+ivec3(1),res-ivec3(1));
    vec3 f=clamp(gp-vec3(p0),vec3(0),vec3(1));
    int xy=res.x*res.y;
    int i000=p0.x+p0.y*res.x+p0.z*xy, i100=p1.x+p0.y*res.x+p0.z*xy;
    int i010=p0.x+p1.y*res.x+p0.z*xy, i110=p1.x+p1.y*res.x+p0.z*xy;
    int i001=p0.x+p0.y*res.x+p1.z*xy, i101=p1.x+p0.y*res.x+p1.z*xy;
    int i011=p0.x+p1.y*res.x+p1.z*xy, i111=p1.x+p1.y*res.x+p1.z*xy;
    DenseGasFloatGrid grid=DenseGasFloatGrid(address);
    return mix(mix(mix(grid.values[i000],grid.values[i100],f.x),
                       mix(grid.values[i010],grid.values[i110],f.x),f.y),
               mix(mix(grid.values[i001],grid.values[i101],f.x),
                       mix(grid.values[i011],grid.values[i111],f.x),f.y),f.z);
}

float hash31(vec3 p) {
    p=fract(p*0.1031); p+=dot(p,p.yzx+33.33); return fract((p.x+p.y)*p.z);
}
float valueNoise(vec3 p) {
    vec3 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    float n000=hash31(i), n100=hash31(i+vec3(1,0,0));
    float n010=hash31(i+vec3(0,1,0)), n110=hash31(i+vec3(1,1,0));
    float n001=hash31(i+vec3(0,0,1)), n101=hash31(i+vec3(1,0,1));
    float n011=hash31(i+vec3(0,1,1)), n111=hash31(i+vec3(1,1,1));
    return mix(mix(mix(n000,n100,f.x),mix(n010,n110,f.x),f.y),
               mix(mix(n001,n101,f.x),mix(n011,n111,f.x),f.y),f.z);
}
float fbm(vec3 p) {
    float v=0.0, a=0.5;
    for(int i=0;i<4;++i){v+=a*valueNoise(p);p*=2.03;a*=0.5;}
    return v;
}

float sampleRawField(uint vi, uint64_t address, vec3 worldPos) {
    vec3 local=transformPoint(vi,184u,worldPos)-volVec3(vi,416u);
    int type=volInt(vi,168u), source=volInt(vi,428u);
    float d=0.0;
    if(type==0) d=1.0;
    else if(type==1) d=fbm(local*max(volFloat(vi,84u),1.0));
    else if(type==2) d=sampleNano(address,local);
    else if(type==3 || source==3) {
        vec3 span=max(volVec3(vi,60u)-volVec3(vi,48u),vec3(1e-5));
        vec3 q=(local-volVec3(vi,48u))/span;
        float coverage=clamp(volFloat(vi,432u),0.0,1.0);
        float base=fbm(vec3(q.x,q.y*0.55,q.z)*max(volFloat(vi,444u),1.0));
        float edge=smoothstep(0.0,max(volFloat(vi,448u),0.02),
            min(min(q.x,1.0-q.x),min(q.z,1.0-q.z)));
        d=max(base-(1.0-coverage),0.0)*edge;
    } else if(type==4 && source==5) d=sampleDense(vi,address,local);
    return max(d,0.0);
}

float sampleField(uint vi, uint64_t address, vec3 worldPos) {
    float d=sampleRawField(vi,address,worldPos);
    float low=volFloat(vi,76u), high=max(volFloat(vi,80u),low+1e-6);
    d=max((d-low)/(high-low),0.0);
    float cutoff=max(volFloat(vi,248u),0.0);
    if(d<=cutoff) return 0.0;
    float fade=cutoff>0.0?smoothstep(cutoff,cutoff*2.0,d):1.0;
    return max(d*volFloat(vi,72u)*fade,0.0);
}

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
    vec2 size=vec2(textureSize(previewOpaqueDepth,0));
    vec2 uv=gl_FragCoord.xy/max(size,vec2(1));
    mat4 invVP=inverse(pc.viewProj);
    vec4 farH=invVP*vec4(uv*2.0-1.0,1.0,1.0);
    vec3 ro=pc.cameraPos.xyz;
    vec3 rd=normalize(farH.xyz/farH.w-ro);
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
        a=max(a,0.001);b=min(b,visibleT);if(b<=a)continue;
        int at=count;
        while(at>0 && starts[at-1]>a){ids[at]=ids[at-1];starts[at]=starts[at-1];ends[at]=ends[at-1];--at;}
        ids[at]=vi;starts[at]=a;ends[at]=b;++count;
    }
    if(count==0)discard;

    vec3 accum=vec3(0); vec3 trans=vec3(1);
    uint quality=pc.materialMeta.y&0xffu;
    int cap=quality<=1u?48:(quality==2u?96:160);
    for(int interval=0;interval<16;++interval){
        if(interval>=count)break;
        uint vi=ids[interval];float length=ends[interval]-starts[interval];
        int authored=clamp(volInt(vi,156u),1,2048),steps=min(authored,cap);
        float dt=max(volFloat(vi,152u),length/float(steps));
        steps=min(int(ceil(length/dt)),cap);dt=length/float(max(steps,1));
        float jitter=hash31(vec3(gl_FragCoord.xy,float(vi)));
        float cachedShadow=1.0;int stride=clamp(volInt(vi,180u),1,16);
        for(int s=0;s<160;++s){
            if(s>=steps || max(max(trans.r,trans.g),trans.b)<0.01)break;
            vec3 p=ro+rd*(starts[interval]+(float(s)+jitter)*dt);
            float density=sampleField(vi,volAddress(vi,232u),p);
            if(density<=0.0)continue;
            vec3 sigmaS=max(volVec3(vi,88u),vec3(0))*max(volFloat(vi,100u),0.0)*density;
            vec3 sigmaA=max(volVec3(vi,120u),vec3(0))*max(volFloat(vi,132u),0.0)*density;
            vec3 sigmaT=max(sigmaS+sigmaA,vec3(1e-6));
            vec3 sampleT=exp(-sigmaT*dt), weight=(vec3(1)-sampleT)*sigmaS/sigmaT;
            vec3 light=ambientLight(-rd)*0.18;
            uint lights=min(sceneLightCount,quality<=1u?2u:(quality==2u?4u:8u));
            for(uint li=0u;li<8u;++li){
                if(li>=lights)break;vec3 l,rad;if(!evalLight(li,p,l,rad))continue;
                if((s%stride)==0){float maxD=int(sceneLights[li].position.w+0.5)==1?1e30:length;cachedShadow=selfShadow(vi,p,l,maxD);}
                light+=rad*phaseDual(dot(rd,l),vi)*cachedShadow;
            }
            if(pc.materialMeta.z==3u && worldMode==2u && worldSun.w>0.0){
                vec3 l=normalize(worldSun.xyz);
                if((s%stride)==0)cachedShadow=selfShadow(vi,p,l,1e30);
                light+=vec3(1.0,0.95,0.86)*worldSun.w*phaseDual(dot(rd,l),vi)*cachedShadow;
            }
            vec3 source=light+emissionAt(vi,p,density);
            accum+=trans*weight*source;
            trans*=sampleT;
        }
    }
    float alpha=clamp(1.0-dot(trans,vec3(0.2126,0.7152,0.0722)),0.0,1.0);
    if(alpha<1e-5)discard;
    // The opaque snapshot is already display transformed. Transform only the
    // medium radiance, then premultiplied-alpha composite in the load pass.
    vec3 display=rtApplyPost(accum,volumePostParams(),uv);
    outColor=vec4(display,alpha);
}
