#ifndef RT_PREVIEW_VOLUME_FIELDS
#define RT_PREVIEW_VOLUME_FIELDS

const uint VOLUME_STRIDE_WORDS = 156u;

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
    // Smoothstep the interpolation weights. Plain trilinear is only C0: the
    // value is continuous across a cell face but its GRADIENT jumps, and the
    // density cutoff in sampleField turns that kink into a visible level set —
    // the cell-shaped faceting on the plume silhouette. Refining the march
    // never removes it, because it is in the RECONSTRUCTION, not the sampling
    // rate. Three multiply-adds buy C1 continuity at the faces.
    f=f*f*(3.0-2.0*f);
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

// Per-block maximum density for live dense gas, published by
// sim_gas_majorant.comp and already carried in this very record (the RT march
// in volume_closesthit.rchit reads the same three fields). Returns the distance
// this ray may advance because the block it is standing in cannot contain
// anything the cutoff would keep; 0.0 means "march normally".
//
// ★★ A MISSING MAJORANT MAY NEVER READ AS "EMPTY". Address 0 or block edge < 1
// means no acceleration is published, and the caller must then sample every
// step — the opposite reading silently deletes the smoke.
layout(buffer_reference, std430, buffer_reference_align=4)
readonly buffer DenseGasMajorantGrid { float blocks[]; };

float denseEmptyBlockStep(uint vi, vec3 worldPos, vec3 worldDir,
                          float baseStep, float cutoff) {
    uint64_t address = volAddress(vi, 512u);
    float blockEdge = volFloat(vi, 532u);
    if (address == 0ul || blockEdge < 1.0) return 0.0;
    ivec3 bdim = ivec3(int(volFloat(vi,520u)+0.5),
                       int(volFloat(vi,524u)+0.5),
                       int(volFloat(vi,528u)+0.5));
    if (any(lessThanEqual(bdim, ivec3(0)))) return 0.0;

    // Exactly the mapping sampleRawField uses, PIVOT INCLUDED, so a block index
    // here addresses the cells sampleDense would have read. Dropping the pivot
    // offsets the skip against the sampler and erases smoke at the domain edge.
    vec3 local = transformPoint(vi,184u,worldPos) - volVec3(vi,416u);
    vec3 dir   = transformVector(vi,184u,worldDir);
    float voxel = max(volFloat(vi,176u), 1e-6);
    vec3 origin = volVec3(vi,476u);
    float blockWorld = blockEdge * voxel;
    ivec3 b = ivec3(floor((local - origin) / blockWorld));
    if (any(lessThan(b, ivec3(0))) || any(greaterThanEqual(b, bdim))) return 0.0;

    // The stored maximum is RAW grid density; sampleField rejects on the
    // REMAPPED value before the multiplier, so remap the block maximum the same
    // way and compare the same quantity.
    DenseGasMajorantGrid mg = DenseGasMajorantGrid(address);
    float blockMax = mg.blocks[b.x + b.y*bdim.x + b.z*bdim.x*bdim.y];
    float low = volFloat(vi,76u), high = max(volFloat(vi,80u), low+1e-6);
    if (max((blockMax-low)/(high-low), 0.0) > cutoff) return 0.0;

    vec3 bmin = origin + vec3(b) * blockWorld;
    vec3 bmax = bmin + vec3(blockWorld);
    float tExit = 1e30;
    for (int axis = 0; axis < 3; ++axis) {
        float d = dir[axis];
        if (abs(d) < 1e-9) continue;
        tExit = min(tExit, ((d > 0.0 ? bmax[axis] : bmin[axis]) - local[axis]) / d);
    }
    if (!(tExit > baseStep)) return 0.0;
    // Stop one voxel short of the boundary: the block maximum says nothing
    // about the neighbour, and landing exactly on the face would let a trilinear
    // tap straddle it.
    return max(baseStep, tExit - max(voxel, baseStep));
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


#endif
