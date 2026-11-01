#include "volume_density_channels.glsl"

#define PNANOVDB_GLSL
#define PNANOVDB_BUF_CUSTOM

struct pnanovdb_buf_t {
    uint64_t address;
};

// We define a buffer block matching NanoVDB scalar layout
layout(buffer_reference, std430, buffer_reference_align=4) buffer NanoVDBBlock {
    uint data[];
};

uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byte_offset) {
    NanoVDBBlock blk = NanoVDBBlock(buf.address);
    return blk.data[byte_offset >> 2];
}

uvec2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byte_offset) {
    NanoVDBBlock blk = NanoVDBBlock(buf.address);
    uint idx = byte_offset >> 2;
    return uvec2(blk.data[idx], blk.data[idx + 1]);
}

void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint byte_offset, uint value) {}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint byte_offset, uvec2 value) {}

#include "PNanoVDB.h"

// ── Persistent-accessor trilinear sampler ─────────────────────────────────
// Caller must have pre-fetched buf/mapH/rootH and called pnanovdb_readaccessor_init
// once. The accessor caches the last walked root→internal→leaf path; reusing it
// across spatially-coherent march steps (typical step ≪ leaf size = 8 voxels)
// skips the tree walk on cache hits. This collapses NanoVDB sampling cost in the
// hot loop from O(tree-depth) to O(1) for the common case.
float sampleNanoVDBFloatTrilinearAcc(
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc,
    vec3 worldPos)
{
    pnanovdb_vec3_t wPos = pnanovdb_vec3_uniform(0.0);
    wPos.x = worldPos.x; wPos.y = worldPos.y; wPos.z = worldPos.z;
    pnanovdb_vec3_t iPos = pnanovdb_map_apply_inverse(buf, mapH, wPos);

    vec3 idxPos = vec3(iPos.x, iPos.y, iPos.z);
    vec3 p0 = floor(idxPos);
    vec3 frac = fract(idxPos);

    float d[8];
    for (int i = 0; i < 8; ++i) {
        pnanovdb_coord_t coord;
        coord.x = int(p0.x) + ((i & 1) != 0 ? 1 : 0);
        coord.y = int(p0.y) + ((i & 2) != 0 ? 1 : 0);
        coord.z = int(p0.z) + ((i & 4) != 0 ? 1 : 0);
        pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, coord);
        d[i] = pnanovdb_read_float(buf, addr);
    }

    float dx00 = mix(d[0], d[1], frac.x);
    float dx10 = mix(d[2], d[3], frac.x);
    float dx01 = mix(d[4], d[5], frac.x);
    float dx11 = mix(d[6], d[7], frac.x);
    float dxy0 = mix(dx00, dx10, frac.y);
    float dxy1 = mix(dx01, dx11, frac.y);
    return mix(dxy0, dxy1, frac.z);
}

// Live gas domains expose their Vulkan compute fields directly to RT. The
// dense grid stores cell-centred floats in x-major order.
layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer DenseGasFloatGrid {
    float values[];
};

float sampleDenseGasFloat(uint64_t address, VkVolumeInstance vol, vec3 localPos) {
    if (address == 0) return 0.0;

    ivec3 resolution = ivec3(
        int(vol._ext_reserved[0] + 0.5),
        int(vol._ext_reserved[1] + 0.5),
        int(vol._ext_reserved[2] + 0.5));
    if (any(lessThanEqual(resolution, ivec3(0)))) return 0.0;

    float voxelSize = max(vol.voxel_size, 1e-6);
    vec3 gridOrigin = vec3(
        vol._ext_reserved[3],
        vol._ext_reserved[4],
        vol._ext_reserved[5]);
    vec3 gridPos = (localPos - gridOrigin) / voxelSize - vec3(0.5);
    if (any(lessThan(gridPos, vec3(-0.5))) ||
        any(greaterThan(gridPos, vec3(resolution) - vec3(0.5)))) {
        return 0.0;
    }

    ivec3 p0 = clamp(ivec3(floor(gridPos)), ivec3(0), resolution - ivec3(1));
    ivec3 p1 = min(p0 + ivec3(1), resolution - ivec3(1));
    vec3 f = clamp(gridPos - vec3(p0), vec3(0.0), vec3(1.0));
    DenseGasFloatGrid grid = DenseGasFloatGrid(address);
    int xyStride = resolution.x * resolution.y;

    int i000 = p0.x + p0.y * resolution.x + p0.z * xyStride;
    int i100 = p1.x + p0.y * resolution.x + p0.z * xyStride;
    int i010 = p0.x + p1.y * resolution.x + p0.z * xyStride;
    int i110 = p1.x + p1.y * resolution.x + p0.z * xyStride;
    int i001 = p0.x + p0.y * resolution.x + p1.z * xyStride;
    int i101 = p1.x + p0.y * resolution.x + p1.z * xyStride;
    int i011 = p0.x + p1.y * resolution.x + p1.z * xyStride;
    int i111 = p1.x + p1.y * resolution.x + p1.z * xyStride;

    float z0 = mix(
        mix(grid.values[i000], grid.values[i100], f.x),
        mix(grid.values[i010], grid.values[i110], f.x),
        f.y);
    float z1 = mix(
        mix(grid.values[i001], grid.values[i101], f.x),
        mix(grid.values[i011], grid.values[i111], f.x),
        f.y);
    return mix(z0, z1, f.z);
}

// ═══════════════════════════════════════════════════════════════════════════
// MATERIAL COORDINATE (UVW) SAMPLING
// ═══════════════════════════════════════════════════════════════════════════
// The isosurface has no UVs and cannot have them: there is no mesh to unwrap,
// and the surface is rebuilt from the field every frame, so there is nothing
// stable for a UV to be attached to. What CAN be stable is a coordinate carried
// by the liquid itself — each particle remembers where it was born, and that
// value is gathered onto a grid the shader can sample.
//
// So the projection below is still tri-planar (no tangent frame exists here
// either), but it is anchored in MATERIAL space instead of world space. The
// difference only shows once the liquid moves: a pour now carries its texture
// instead of sliding through a stationary pattern.
//
// ★ Identity at rest. The coordinate is seeded with the birth POSITION in world
// units, so for a body that has not moved, uvw == worldPos and this returns
// exactly what the old world-anchored path returned. That is deliberate: it
// makes "did this change anything it should not have?" answerable by rendering
// a still tank and diffing, rather than by judgement.
// ★★★ THE BUFFER HOLDS A DISPLACEMENT, NOT A COORDINATE. Cell c stores
// (uvw - centre(c)); the coordinate is rebuilt here as worldPos + d.
//
// Storing the sum instead would cap the WHOLE coordinate at the sim voxel size,
// because the position term — which is full-resolution and free right here in
// the shader — would have had to survive a trip through the grid. That is
// exactly what "material mode looks like one pixel per cell" was. Only the
// deformation is grid-limited now, and deformation is smooth.
//
// ★ Consequence worth knowing before debugging anything here: in still liquid
// d is identically zero, so this returns worldPos EXACTLY. A resting tank
// therefore has to match World mode pixel for pixel.
layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer MaterialResidualGrid {
    float values[];      // interleaved xyz displacement per cell
};

// Returns false when this volume publishes no coordinate field. Callers MUST
// then fall back to worldPos. Returning a zero vector instead would map the
// entire surface to one texel — a flat-tinted liquid, which looks like a
// material authoring mistake rather than like missing data.
bool sampleMaterialCoord(VkVolumeInstance vol, vec3 worldPos, out vec3 coord) {
    coord = worldPos;
    if (vol.uvw_residual_address == 0) return false;

    ivec3 res = ivec3(int(vol.uvw_dim[0] + 0.5),
                      int(vol.uvw_dim[1] + 0.5),
                      int(vol.uvw_dim[2] + 0.5));
    if (any(lessThanEqual(res, ivec3(0)))) return false;
    if (vol.uvw_voxel <= 0.0) return false;

    // ★★★ INDEXED IN WORLD SPACE, THROUGH THE GRID'S OWN ORIGIN AND CELL SIZE.
    //
    // This used to go through inv_transform and normalise by aabb_min/aabb_max,
    // by analogy with the density sampler. That analogy is false. On a live
    // fluid, aabb is the tight ACTIVE box of the dense/SDF grid — padded by a
    // cell and recomputed as the liquid moves — while this buffer spans the
    // whole SIM grid at the sim's own resolution. Mapping the first onto the
    // second scaled the field by the ratio of their extents and offset it by the
    // difference of their origins, and since the active box follows the liquid,
    // both errors changed every frame.
    //
    // ★ The symptom was NOT "wrong texture position", which somebody would have
    // reported immediately. A slowly varying scale error reads as the pattern
    // smearing along the flow and swimming over the surface — i.e. it looks like
    // a quality problem in the coordinate, and it survived a whole round of
    // genuine quality work sitting underneath it.
    //
    // The producer walks the sim grid in world space, so the inverse of that
    // walk is world space too. No transform: if the domain moves, its grid
    // origin moves with it and arrives here already updated.
    vec3 uvwOrigin = vec3(vol.uvw_origin[0], vol.uvw_origin[1], vol.uvw_origin[2]);
    vec3 gridPos = (worldPos - uvwOrigin) / vol.uvw_voxel - vec3(0.5);

    // CLAMP rather than reject out-of-range. The producer extrapolated the field
    // a few voxels past the supported region precisely so the surface — which
    // sits at the edge of it — reads valid data; failing out here at the domain
    // wall would undo that and re-introduce a world-anchored ring exactly where
    // liquid touches the boundary.
    ivec3 p0 = clamp(ivec3(floor(gridPos)), ivec3(0), res - ivec3(1));
    ivec3 p1 = min(p0 + ivec3(1), res - ivec3(1));
    vec3  f  = clamp(gridPos - vec3(p0), vec3(0.0), vec3(1.0));

    // ★ Quintic ease on the interpolation fraction (Perlin's 6t^5-15t^4+10t^3).
    // Plain trilinear is only C0: the value is continuous across a cell boundary
    // but its DERIVATIVE jumps, and the derivative is what a texture lookup
    // actually rides. Easing makes it vanish at the boundaries instead.
    //
    // ★★★ THIS IS ONLY LEGAL ON A RESIDUAL, and getting that backwards is what
    // produced the reported blockiness. Applied to an ABSOLUTE coordinate the
    // ease modulates the identity gradient itself: d(coord)/d(world) becomes
    // s'(f), which runs from 0.0 at every cell face to 1.875 at every cell
    // centre. The texture is then frozen on the cell boundaries and compressed
    // in the middles — a rectangular quilt of plateaus with pinched seams,
    // at exactly one tile per cell. The seam-removal tool was the seam.
    //
    // On a residual it does what it was meant to do: the identity gradient is
    // added outside this function and is untouched, so easing only softens the
    // small deformation term.
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    MaterialResidualGrid g = MaterialResidualGrid(vol.uvw_residual_address);
    int xy = res.x * res.y;
    int i000 = (p0.x + p0.y * res.x + p0.z * xy) * 3;
    int i100 = (p1.x + p0.y * res.x + p0.z * xy) * 3;
    int i010 = (p0.x + p1.y * res.x + p0.z * xy) * 3;
    int i110 = (p1.x + p1.y * res.x + p0.z * xy) * 3;
    int i001 = (p0.x + p0.y * res.x + p1.z * xy) * 3;
    int i101 = (p1.x + p0.y * res.x + p1.z * xy) * 3;
    int i011 = (p0.x + p1.y * res.x + p1.z * xy) * 3;
    int i111 = (p1.x + p1.y * res.x + p1.z * xy) * 3;

    vec3 c000 = vec3(g.values[i000], g.values[i000 + 1], g.values[i000 + 2]);
    vec3 c100 = vec3(g.values[i100], g.values[i100 + 1], g.values[i100 + 2]);
    vec3 c010 = vec3(g.values[i010], g.values[i010 + 1], g.values[i010 + 2]);
    vec3 c110 = vec3(g.values[i110], g.values[i110 + 1], g.values[i110 + 2]);
    vec3 c001 = vec3(g.values[i001], g.values[i001 + 1], g.values[i001 + 2]);
    vec3 c101 = vec3(g.values[i101], g.values[i101 + 1], g.values[i101 + 2]);
    vec3 c011 = vec3(g.values[i011], g.values[i011 + 1], g.values[i011 + 2]);
    vec3 c111 = vec3(g.values[i111], g.values[i111 + 1], g.values[i111 + 2]);

    vec3 z0 = mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y);
    vec3 z1 = mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y);

    // worldPos carries the full-resolution identity; the grid only bends it.
    coord = worldPos + mix(z0, z1, f.z);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPOSITION — which materials this point of the liquid is made of.
// ═══════════════════════════════════════════════════════════════════════════
// Per cell the producer stores the two dominant material slots and the weight
// of the second. Both slots are 1-based ids; 0 means "the built-in dielectric",
// which is a real choice for a substance and not an absence.
//
// ★★★ ONLY THE WEIGHT IS INTERPOLATED. An index is a NAME: the value halfway
// between material 2 and material 4 is material 3, an unrelated material that
// would appear as a band along every boundary. So the slots come from the
// NEAREST cell and only the blend fraction is filtered. In a binary mixture —
// the ordinary case — the pair is the same everywhere the mixture exists, so
// nothing is lost; with three or more, the pair switches at a cell boundary
// while the fraction stays smooth.
//
// ★ Falls back to the domain material when nothing is published, which is what
// makes a single-material domain cost exactly what it did before.
layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer CompositionGrid {
    float values[];      // interleaved (slotA, slotB, weightB) per cell
};

void sampleComposition(VkVolumeInstance vol, vec3 worldPos,
                       inout float slotA, out float slotB, out float weightB) {
    slotB = slotA;
    weightB = 0.0;
    if (vol.composition_address == 0) return;
    if (vol.uvw_voxel <= 0.0) return;

    ivec3 res = ivec3(int(vol.uvw_dim[0] + 0.5),
                      int(vol.uvw_dim[1] + 0.5),
                      int(vol.uvw_dim[2] + 0.5));
    if (any(lessThanEqual(res, ivec3(0)))) return;

    // Same world-space indexing as the residual field — same grid, same origin,
    // same cell size. Sharing the placement is what guarantees the mixture and
    // the coordinate describe the same point.
    vec3 uvwOrigin = vec3(vol.uvw_origin[0], vol.uvw_origin[1], vol.uvw_origin[2]);
    vec3 gridPos = (worldPos - uvwOrigin) / vol.uvw_voxel - vec3(0.5);

    ivec3 p0 = clamp(ivec3(floor(gridPos)), ivec3(0), res - ivec3(1));
    ivec3 p1 = min(p0 + ivec3(1), res - ivec3(1));
    vec3  f  = clamp(gridPos - vec3(p0), vec3(0.0), vec3(1.0));

    CompositionGrid g = CompositionGrid(vol.composition_address);
    int xy = res.x * res.y;

    // Nearest cell decides WHICH two materials; see the note above.
    ivec3 pn = ivec3(greaterThan(f, vec3(0.5))) * (p1 - p0) + p0;
    int   ni = (pn.x + pn.y * res.x + pn.z * xy) * 3;
    slotA = g.values[ni + 0];
    slotB = g.values[ni + 1];

    // The weight is a quantity and IS interpolated, so the transition between
    // two substances is smooth rather than stepping at cell boundaries.
    int i000 = (p0.x + p0.y * res.x + p0.z * xy) * 3 + 2;
    int i100 = (p1.x + p0.y * res.x + p0.z * xy) * 3 + 2;
    int i010 = (p0.x + p1.y * res.x + p0.z * xy) * 3 + 2;
    int i110 = (p1.x + p1.y * res.x + p0.z * xy) * 3 + 2;
    int i001 = (p0.x + p0.y * res.x + p1.z * xy) * 3 + 2;
    int i101 = (p1.x + p0.y * res.x + p1.z * xy) * 3 + 2;
    int i011 = (p0.x + p1.y * res.x + p1.z * xy) * 3 + 2;
    int i111 = (p1.x + p1.y * res.x + p1.z * xy) * 3 + 2;

    float z0 = mix(mix(g.values[i000], g.values[i100], f.x),
                   mix(g.values[i010], g.values[i110], f.x), f.y);
    float z1 = mix(mix(g.values[i001], g.values[i101], f.x),
                   mix(g.values[i011], g.values[i111], f.x), f.y);
    weightB = clamp(mix(z0, z1, f.z), 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// THE ONE ANCHOR. Every isosurface pattern goes through here — tri-planar
// textures, the resin interior march, the porosity lattice, and the opacity
// mask. Three spaces, ONE place they diverge.
// ═══════════════════════════════════════════════════════════════════════════
// ★ That single divergence point is the whole design. The consumers do not know
// which space they are in and must not: the moment two of them resolve the
// space separately, they are free to disagree, and a resin interior anchored
// differently from the albedo painted over it reads as "the shading is a bit
// off" rather than as a coordinate bug.
//
//   COORD_MATERIAL - the parcel's own coordinate. Carried BY the liquid, so a
//                    pour takes its pattern with it. Identity for anything that
//                    has not moved, so it is also the safe default.
//   COORD_DOMAIN   - the container's local space. Travels with a carried vessel
//                    while the liquid flows THROUGH the pattern: the right
//                    answer for something painted on the tank, and the wrong
//                    one for something that belongs to the substance.
//   COORD_WORLD    - nailed to the room. The behaviour that shipped before
//                    material coordinates existed. Kept because it is a real
//                    look (a projected pattern), and because it is the escape
//                    hatch from the material coordinate's one true weakness —
//                    it STRETCHES with the flow, so a violently deformed splash
//                    eventually maps chaotically and an artist needs a way out
//                    that is not "animate less".
//
// The bool form above stays separate rather than folded in here because "there
// is no coordinate field" and "the coordinate happens to equal the world
// position" are different facts, and a caller that ever needs to tell them
// apart must not have to re-derive it from a value that looks identical.
const uint COORD_MATERIAL = 0u;
const uint COORD_DOMAIN   = 1u;
const uint COORD_WORLD    = 2u;

vec3 materialAnchor(VkVolumeInstance vol, vec3 worldPos) {
    uint space = uint(clamp(vol._accel_reserved[3], 0.0, 2.0) + 0.5);

    if (space == COORD_WORLD) return worldPos;

    if (space == COORD_DOMAIN) {
        // Volume-local, through the same inverse transform the density sampler
        // uses. NOT normalised to [0,1]: keeping it in world-sized units means
        // uv_scale/pore_scale stay "world units per tile" in every space, so
        // switching space re-anchors the pattern without also resizing it.
        vec3 localPos;
        localPos.x = vol.inv_transform[0] * worldPos.x + vol.inv_transform[1] * worldPos.y
                   + vol.inv_transform[2] * worldPos.z + vol.inv_transform[3];
        localPos.y = vol.inv_transform[4] * worldPos.x + vol.inv_transform[5] * worldPos.y
                   + vol.inv_transform[6] * worldPos.z + vol.inv_transform[7];
        localPos.z = vol.inv_transform[8] * worldPos.x + vol.inv_transform[9] * worldPos.y
                   + vol.inv_transform[10] * worldPos.z + vol.inv_transform[11];
        return localPos;
    }

    // COORD_MATERIAL. Falls back to worldPos when no field is published, which
    // is what makes an unbuilt or unavailable coordinate degrade to the old
    // behaviour instead of collapsing the surface onto the origin.
    vec3 c;
    sampleMaterialCoord(vol, worldPos, c);
    return c;
}

layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer DenseGasMajorantGrid {
    float blocks[];
};

// [0] = emitter count (may exceed capacity; the producer keeps counting so the
// overflow is visible), [1..] = block indices into the majorant grid.
layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer DenseGasEmissiveList {
    uint entries[];
};

// Empty-space skip for LIVE DENSE gas (volume_type 4 / source_type 5), which
// has no NanoVDB hierarchy to walk: a dense domain is a solid box of cells and
// the march used to pay for every step of it even when the smoke occupied a
// corner. sim_gas_majorant.comp reduces the density field to one maximum per
// 8^3 block; a block whose maximum is at or below the authored cutoff cannot
// contribute, so the ray jumps straight to that block's exit.
//
// Returns the distance to advance along `worldDir`, or 0 when the current block
// has content (or no majorant is published — then this must return 0 so the
// caller marches normally; treating a missing majorant as empty would delete
// the smoke rather than merely slow the render down).
//
// The reduction already covers one cell past each far face, so trilinear
// reconstruction at the boundary of a skipped block still reads zero.
float denseGasEmptyBlockStep(VkVolumeInstance vol, vec3 worldPos, vec3 worldDir,
                             float cutoff) {
    // The panel's "Majorant availability %" reads these. It was showing 0/0 on
    // every scene because volumeRecordMajorantQuery() was defined and never
    // called — a published majorant grid and a missing one produced the same
    // blank reading, so the row could not distinguish "no acceleration
    // available" from "acceleration available and not helping".
    bool majorantPublished = vol.majorant_address != 0 && vol.majorant_block >= 1.0;
    volumeRecordMajorantQuery(majorantPublished);
    if (!majorantPublished) return 0.0;

    ivec3 bdim = ivec3(vol.majorant_dim[0] + 0.5,
                       vol.majorant_dim[1] + 0.5,
                       vol.majorant_dim[2] + 0.5);
    if (any(lessThanEqual(bdim, ivec3(0)))) return 0.0;

    // Same object-space mapping the dense sampler uses, so a block index here
    // addresses exactly the cells sampleDenseGasFloat would read.
    vec3 localPos;
    localPos.x = vol.inv_transform[0]*worldPos.x + vol.inv_transform[1]*worldPos.y + vol.inv_transform[2]*worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4]*worldPos.x + vol.inv_transform[5]*worldPos.y + vol.inv_transform[6]*worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8]*worldPos.x + vol.inv_transform[9]*worldPos.y + vol.inv_transform[10]*worldPos.z + vol.inv_transform[11];
    vec3 localDir;
    localDir.x = vol.inv_transform[0]*worldDir.x + vol.inv_transform[1]*worldDir.y + vol.inv_transform[2]*worldDir.z;
    localDir.y = vol.inv_transform[4]*worldDir.x + vol.inv_transform[5]*worldDir.y + vol.inv_transform[6]*worldDir.z;
    localDir.z = vol.inv_transform[8]*worldDir.x + vol.inv_transform[9]*worldDir.y + vol.inv_transform[10]*worldDir.z;

    float voxelSize = max(vol.voxel_size, 1e-6);
    vec3 gridOrigin = vec3(vol._ext_reserved[3], vol._ext_reserved[4], vol._ext_reserved[5]);
    float blockWorld = vol.majorant_block * voxelSize;   // block edge in object units
    vec3 blockPos = (localPos - gridOrigin) / blockWorld;
    ivec3 b = ivec3(floor(blockPos));
    if (any(lessThan(b, ivec3(0))) || any(greaterThanEqual(b, bdim))) return 0.0;

    DenseGasMajorantGrid mg = DenseGasMajorantGrid(vol.majorant_address);
    int bi = b.x + b.y * bdim.x + b.z * bdim.x * bdim.y;
    // The stored maximum is RAW grid density. sampleDensityAcc rejects on the
    // REMAPPED value before applying density_multiplier, so remap the block
    // maximum the same way and compare the same quantity — multiplying here
    // would reject blocks the sampler would have kept (or worse, skip blocks it
    // would have shaded) whenever the multiplier is not 1.
    float blockMax = mg.blocks[bi];
    float remapped = volumeDensitySupport(max((blockMax - vol.density_remap_low) /
        max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0),
        vol.density_curve, vol._retired_cloud[0]);
    if (remapped > cutoff) return 0.0;   // block has content — march it

    // Slab exit of this block, in object units (== world units for a rigid
    // transform; a scaled volume errs on the short side, which only costs a
    // redundant sample).
    vec3 bmin = gridOrigin + vec3(b) * blockWorld;
    vec3 bmax = bmin + vec3(blockWorld);
    float tExit = 1e30;
    for (int axis = 0; axis < 3; ++axis) {
        float d = localDir[axis];
        if (abs(d) < 1e-9) continue;
        float bound = d > 0.0 ? bmax[axis] : bmin[axis];
        tExit = min(tExit, (bound - localPos[axis]) / d);
    }
    if (tExit >= 1e29 || tExit <= 0.0) return 0.0;
    // Land just inside the next block so the following iteration cannot pick
    // this same block again and stall the march.
    return tExit + blockWorld * 0.01;
}

#include "volume_nano_optical.glsl"

// Trilinear interpolation of a FloatGrid
float sampleNanoVDBFloatTrilinear(uint64_t gridAddr, vec3 worldPos) {
    if (gridAddr == 0) return 0.0;
    
    pnanovdb_buf_t buf;
    buf.address = gridAddr;
    
    // Get Handles
    pnanovdb_grid_handle_t gridH; gridH.address.byte_offset = 0u;
    pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(buf, gridH);
    pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(buf, treeH);
    pnanovdb_map_handle_t  mapH  = pnanovdb_grid_get_map(buf, gridH);
    
    // Init Accessor
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, rootH);
    
    // World to Index mapped
    pnanovdb_vec3_t wPos = pnanovdb_vec3_uniform(0.0);
    wPos.x = worldPos.x; wPos.y = worldPos.y; wPos.z = worldPos.z;
    pnanovdb_vec3_t iPos = pnanovdb_map_apply_inverse(buf, mapH, wPos);
    
    vec3 idxPos = vec3(iPos.x, iPos.y, iPos.z);
    
    vec3 p0 = floor(idxPos);
    vec3 frac = fract(idxPos);
    
    float d[8];
    for (int i = 0; i < 8; ++i) {
        pnanovdb_coord_t coord;
        coord.x = int(p0.x) + ((i & 1) != 0 ? 1 : 0);
        coord.y = int(p0.y) + ((i & 2) != 0 ? 1 : 0);
        coord.z = int(p0.z) + ((i & 4) != 0 ? 1 : 0);
        
        // Fast leaf-level read
        pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, coord);
        d[i] = pnanovdb_read_float(buf, addr);
    }
    
    float dx00 = mix(d[0], d[1], frac.x);
    float dx10 = mix(d[2], d[3], frac.x);
    float dx01 = mix(d[4], d[5], frac.x);
    float dx11 = mix(d[6], d[7], frac.x);
    
    float dxy0 = mix(dx00, dx10, frac.y);
    float dxy1 = mix(dx01, dx11, frac.y);
    
    return mix(dxy0, dxy1, frac.z);
}

// ============================================================
// Blackbody RGB — Kim et al. approximation (matches OptiX blackbody_to_rgb)
// ============================================================
#include "volume_material_response.glsl"

vec3 blackbodyToRGB(float kelvin) {
    kelvin = clamp(kelvin, 1000.0, 40000.0);
    float temp = kelvin / 100.0;
    float r, g, b;
    // Red
    if (temp <= 66.0) { r = 1.0; }
    else { r = clamp(329.698727446 * pow(temp - 60.0, -0.1332047592) / 255.0, 0.0, 1.0); }
    // Green
    if (temp <= 66.0) { g = clamp((99.4708025861 * log(temp) - 161.1195681661) / 255.0, 0.0, 1.0); }
    else              { g = clamp(288.1221695283 * pow(temp - 60.0, -0.0755148492) / 255.0, 0.0, 1.0); }
    // Blue
    if (temp >= 66.0)      { b = 1.0; }
    else if (temp <= 19.0) { b = 0.0; }
    else { b = clamp((138.5177312231 * log(temp - 10.0) - 305.0447927307) / 255.0, 0.0, 1.0); }
    return vec3(r, g, b);
}

// OptiX parity: preserve hue while limiting tiny, extremely hot cells that
// otherwise dominate rare indirect paths and show up as volume fireflies.
vec3 clampVolumeRadiance(vec3 c, float maxLuma) {
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return (luma > maxLuma && luma > 1e-6) ? c * (maxLuma / luma) : c;
}

// ============================================================
// Color Ramp — linear interpolation over stop list
// ============================================================
vec3 sampleColorRamp(VkVolumeInstance vol, float t) {
    if (vol.ramp_stop_count == 0) return vec3(1.0);
    vec3 c0 = vec3(vol.ramp_colors_r[0], vol.ramp_colors_g[0], vol.ramp_colors_b[0]);
    if (t <= vol.ramp_positions[0]) return c0;
    int last = vol.ramp_stop_count - 1;
    vec3 cN = vec3(vol.ramp_colors_r[last], vol.ramp_colors_g[last], vol.ramp_colors_b[last]);
    if (t >= vol.ramp_positions[last]) return cN;
    for (int i = 1; i < vol.ramp_stop_count; ++i) {
        if (t < vol.ramp_positions[i]) {
            float f = (t - vol.ramp_positions[i-1]) / max(vol.ramp_positions[i] - vol.ramp_positions[i-1], 1e-6);
            vec3 a = vec3(vol.ramp_colors_r[i-1], vol.ramp_colors_g[i-1], vol.ramp_colors_b[i-1]);
            vec3 b = vec3(vol.ramp_colors_r[i],   vol.ramp_colors_g[i],   vol.ramp_colors_b[i]);
            return mix(a, b, f);
        }
    }
    return cN;
}

// ============================================================
// Temperature Sampling — secondary NanoVDB grid via vdb_temp_address
// ============================================================
float sampleTemperature(VkVolumeInstance vol, vec3 worldPos) {
    if (vol.vdb_temp_address == 0) return 0.0;
    vec3 localPos;
    localPos.x = vol.inv_transform[0]*worldPos.x + vol.inv_transform[1]*worldPos.y
               + vol.inv_transform[2]*worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4]*worldPos.x + vol.inv_transform[5]*worldPos.y
               + vol.inv_transform[6]*worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8]*worldPos.x + vol.inv_transform[9]*worldPos.y
               + vol.inv_transform[10]*worldPos.z + vol.inv_transform[11];
    
    // Safety check bound box instead of 0.5 cube
    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) return 0.0;
    
    // Pivot offset correction (OptiX parity)
    localPos.x -= vol.pivot_offset[0];
    localPos.y -= vol.pivot_offset[1];
    localPos.z -= vol.pivot_offset[2];

    if (vol.volume_type == 4 && vol.source_type == 5) {
        // Simulation heat is normalized; retain the old live-VDB Kelvin scale.
        return sampleDenseGasFloat(vol.vdb_temp_address, vol, localPos) * 3000.0;
    }

    return sampleNanoVDBFloatTrilinear(vol.vdb_temp_address, localPos);
}

// Combustion reaction rate (GridFluid's bounded `interaction` field) at a world
// position, from live dense gas or its published NanoVDB grid. Uses exactly the transform + pivot chain
// sampleTemperature applies, so the reaction lines up with the temperature it
// modulates instead of being offset by the pivot on a moved domain.
float sampleFlame(VkVolumeInstance vol, vec3 worldPos) {
    if (vol.flame_address == 0) return 0.0;
    vec3 localPos;
    localPos.x = vol.inv_transform[0]*worldPos.x + vol.inv_transform[1]*worldPos.y
               + vol.inv_transform[2]*worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4]*worldPos.x + vol.inv_transform[5]*worldPos.y
               + vol.inv_transform[6]*worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8]*worldPos.x + vol.inv_transform[9]*worldPos.y
               + vol.inv_transform[10]*worldPos.z + vol.inv_transform[11];
    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) return 0.0;
    localPos.x -= vol.pivot_offset[0];
    localPos.y -= vol.pivot_offset[1];
    localPos.z -= vol.pivot_offset[2];
    if (vol.volume_type == 4 && vol.source_type == 5) {
        return clamp(sampleDenseGasFloat(vol.flame_address, vol, localPos), 0.0, 1.0);
    }
    return clamp(sampleNanoVDBFloatTrilinear(vol.flame_address, localPos), 0.0, 1.0);
}

float applyMaterialDensityNoise(VkVolumeInstance vol, vec3 localPos, float density) {
    if (vol._ext_reserved[7] < 0.5 || density <= 0.0) return density;
    vec3 extent = max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
    vec3 p = (localPos - vol.aabb_min) / extent;
    float scale = max(vol._ext_reserved[8], 0.001);
    float strength = clamp(vol._ext_reserved[9], 0.0, 1.0);
    int detail = clamp(int(vol._ext_reserved[10] + 0.5), 1, 8);
    float seed = vol._ext_reserved[11];
    vec3 seedOffset = vec3(seed * 0.1031, seed * 0.11369, seed * 0.13787);
    float field = fbmNoise(p * scale + seedOffset, detail);
    return density * mix(1.0, field, strength);
}

// ── Persistent-accessor density sampler ───────────────────────────────────
// Same dispatch as sampleDensity but routes volume_type==2 NanoVDB reads through
// the caller's pre-initialized accessor. Other volume types (homogeneous, procedural,
// cloud) ignore the accessor params. Caller must guarantee buf/mapH are valid when
// vol.volume_type == 2 && vol.vdb_grid_address != 0; otherwise the accessor args are
// untouched.
float sampleDensityAccChannels(
    VkVolumeInstance vol,
    vec3 worldPos,
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc,
    bool applyOpticalControls,
    out float emissionDensity)
{
    emissionDensity = 0.0;
    #include "volume_density_read_gate.glsl"
    vec3 localPos;
    localPos.x = vol.inv_transform[0] * worldPos.x + vol.inv_transform[1] * worldPos.y
               + vol.inv_transform[2] * worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4] * worldPos.x + vol.inv_transform[5] * worldPos.y
               + vol.inv_transform[6] * worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8] * worldPos.x + vol.inv_transform[9] * worldPos.y
               + vol.inv_transform[10] * worldPos.z + vol.inv_transform[11];

    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) {
        return 0.0;
    }

    float density = 1.0;

    if (vol.volume_type == 0) {
        density = 1.0;
    } else if (vol.volume_type == 1) {
        vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
        vec3 noiseCoord = normPos * vol.noise_scale;
        density = fbmNoise(noiseCoord, 4);
        vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
        float edgeFalloff = min(min(edgeDist.x, edgeDist.y), edgeDist.z);
        density *= smoothstep(0.0, 0.1, edgeFalloff);
    } else if (vol.volume_type == 2) {
        if (vol.vdb_grid_address != 0) {
            vec3 vdbWorldPos = localPos;
            vdbWorldPos.x -= vol.pivot_offset[0];
            vdbWorldPos.y -= vol.pivot_offset[1];
            vdbWorldPos.z -= vol.pivot_offset[2];
            density = sampleNanoVDBFloatTrilinearAcc(buf, mapH, acc, vdbWorldPos);
        } else {
            vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
            vec3 noiseCoord = normPos * max(vol.noise_scale, 1.0);
            density = fbmNoise(noiseCoord, 4);
            vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
            density *= smoothstep(0.0, 0.1, min(min(edgeDist.x, edgeDist.y), edgeDist.z));
        }
    } else if (vol.volume_type == 4 && vol.source_type == 5) {
        density = sampleDenseGasFloat(vol.vdb_grid_address, vol, localPos);
    }

    // SurfaceSDF consumes the producer's surface-centred 0..1 proxy directly.
    // Density/Remap/Cutoff/volume-noise are optical fog controls: applying them
    // before the iso=0.5 test moves the geometry and grows a false skirt along
    // domain walls. Explicit surface modifiers (porosity and opacity mask) are
    // applied later by sampleIsoField and remain geometric by design.
    if (!applyOpticalControls) {
        return (isnan(density) || isinf(density)) ? 0.0 : max(density, 0.0);
    }

    density = applyMaterialDensityNoise(vol, localPos, density);
    float normalized = max((density - vol.density_remap_low) /
        max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0);
    vec2 channels = volumeDensityChannels(normalized, vol.density_curve,
        max(vol._reserved[0], 0.0), vol.density_multiplier, vol._retired_cloud[0]);
    emissionDensity = channels.y;
    return channels.x;
}

float sampleDensityAccMode(VkVolumeInstance vol, vec3 worldPos,
    pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc, bool applyOpticalControls) {
    float unusedEmission;
    return sampleDensityAccChannels(vol, worldPos, buf, mapH, acc,
        applyOpticalControls, unusedEmission);
}

float sampleDensityAcc(
    VkVolumeInstance vol,
    vec3 worldPos,
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc)
{
    return sampleDensityAccMode(vol, worldPos, buf, mapH, acc, true);
}

// ═══════════════════════════════════════════════════════════════════════════
// PROCEDURAL POROSITY — fermented dough, aerated batter, pumice, set foam
// ═══════════════════════════════════════════════════════════════════════════
// Returns the amount to SUBTRACT from the density before the ISO test.
//
// ★ Why subtract from the FIELD instead of cutting alpha at the surface.
// A pore made by alpha has no rim: the hole is punched in an otherwise flat
// shading point, so its edge carries the parent surface's normal, refracts
// wrongly and casts no self-shadow. Displacing the field makes the pore real
// geometry — the surface normal here IS the field gradient, so the rim picks
// up the noise derivative for FREE, from the six samples already being taken.
// That is the whole reason this is a field term and not a shading term.
//
// ★★ MUST be a pure function of world position and the domain's own
// parameters — nothing per-ray, nothing per-material, no state. It is called
// from TWO places: the shading march AND nearestSurfaceSDFCrossing, the
// arbiter that decides where gas hands the ray over to liquid. If those two
// disagree by so much as a term, the gas is clipped against a surface the
// shader never draws, and nothing reports it. That constraint is also why the
// parameters live on the domain instance: the arbiter runs for OTHER volumes
// and has no access to this domain's surface material.
//
// Worley rather than fbm, deliberately: a bubble is a CELL. The distance to
// the nearest feature point of a Worley lattice is already a packed-sphere
// field, so 1-d gives voids at the cell centres. fbm would give clouds.
float isoPoreOffset(VkVolumeInstance vol, vec3 worldPos) {
    float amount = vol._accel_reserved[0];
    if (amount <= 1e-4) return 0.0;                       // off: exact old path
    float scale  = max(vol._accel_reserved[1], 1e-4);     // world units per cell
    float detail = clamp(vol._accel_reserved[2], 0.0, 1.0);

    // MATERIAL space, not world space, and not voxel space.
    //
    // Not voxel space, because the crumb must keep its real size when the domain
    // resolution changes, or every resolution edit re-bakes the bread. That is
    // why `scale` is in world units and why the coordinate below is too.
    //
    // Not world space, because a world-anchored pore lattice is nailed to the
    // room: a rising dough would slide THROUGH its own bubbles, gaining and
    // losing pores as it moved. The bubbles belong to the material, so they are
    // addressed in the material's coordinate — which also means they now stretch
    // with it, the way real crumb does. Falls back to worldPos when the domain
    // publishes no coordinate field, i.e. exactly the previous behaviour.
    //
    // ★ This stays inside sampleIsoField's ONE-field contract: the arbiter
    // (nearestSurfaceSDFCrossing) reaches this through the same call and reads
    // the same vol, so it sees the identical displaced field. Anchoring here
    // would only break that if it depended on something per-ray or per-material,
    // which the coordinate grid is not — it is per-domain data, like `scale`.
    vec3 p = materialAnchor(vol, worldPos) / scale;

    float pore = 1.0 - clamp(rh_worley(p) * 2.0, 0.0, 1.0);
    if (detail > 1e-3) {
        // A second, finer cell size mixed in by MAX (union of voids, not an
        // average): this is what separates bread crumb, which has mixed bubble
        // sizes, from packing foam, which has one.
        float fine = 1.0 - clamp(rh_worley(p * 2.7 + vec3(19.3, 7.1, 31.7)) * 2.0, 0.0, 1.0);
        pore = max(pore, fine * detail);
    }
    return amount * pore;
}

// ═══════════════════════════════════════════════════════════════════════════
// OPACITY MASK — holes in the liquid, cut in the FIELD
// ═══════════════════════════════════════════════════════════════════════════
// Returns the amount to SUBTRACT from the density before the ISO test, so a
// masked-out region is a real hole in the level set.
//
// ★ Why not an any-hit shader, which is the obvious answer. The volume
// procedural hit group binds VK_SHADER_UNUSED_KHR for any-hit, and the AABB
// BLAS is already built non-opaque — so the acceleration structure was never
// the obstacle, and adding an any-hit here would have "worked". It would also
// have produced RIMLESS holes: a cutout at the shading point punches through an
// otherwise flat surface, so the edge carries the parent surface's normal,
// refracts as if the hole were not there, and casts no self-shadow. On a liquid
// that is glaring, because the edge is exactly where refraction is read. Cut in
// the field instead and the rim normal falls out of the gradient that six
// samples are already computing — the same reason porosity is a field term.
//
// ★★ MEANING, matched to the mesh path deliberately. On a triangle,
// `mat.opacity` as a SCALAR becomes transmission (closesthit.rchit: `opacity <
// 0.99 -> transmission = 1 - opacity`), while the opacity TEXTURE is a cutout
// mask. This keeps that split exactly: only the texture cuts geometry here, and
// the scalar is left to the transmission path further down. Making the scalar
// erode the body instead would mean one material reads as semi-transparent
// glass on a mesh and as a thinner object on a liquid — the same number meaning
// two things, which is how a "the material looks wrong on water" report becomes
// unfindable.
//
// ★★★ THRESHOLDED, NOT STOCHASTIC, and this is forced rather than chosen. The
// mesh dithers intermediate mask values across samples (shadow_anyhit does
// exactly that, and it is right there). This field CANNOT: it is read by
// nearestSurfaceSDFCrossing, the gas/liquid handover arbiter, which must agree
// with the shading march sample for sample. A randomised threshold would make
// gas clip against a surface that differs every sample — shimmering smoke edges
// near liquid, with nothing pointing at opacity as the cause. The field must be
// a pure function of position, so the mask is hard.
float isoAlphaOffset(VkVolumeInstance vol, vec3 worldPos) {
    if (vol.iso_material_index <= 0.5) return 0.0;          // no bound material
    Material am = materials.m[uint(vol.iso_material_index - 1.0)];
    if (am.opacity_tex == 0u) return 0.0;                   // off: not one fetch

    // Tri-planar with EQUAL weights, not the sharpened-normal blend the albedo
    // path uses. There is no normal available here and there cannot be: this
    // function runs inside the gradient taps that COMPUTE the normal, so asking
    // for one is circular. Equal thirds of a hard mask make the test a majority
    // vote across the three projections, which is deterministic and stable.
    //
    // ★ Consequence worth knowing before it is mistaken for a bug: the hole edge
    // and an albedo edge cut from the SAME image will not land in exactly the
    // same place on steep faces, because the two use different blends.
    vec3 p = materialAnchor(vol, worldPos);
    vec2 sc = vec2(abs(am.uv_scale_x) > 1e-6 ? am.uv_scale_x : 1.0,
                   abs(am.uv_scale_y) > 1e-6 ? am.uv_scale_y : 1.0);
    vec2 of = vec2(am.uv_offset_x, am.uv_offset_y);
    vec4 tx = textureLod(materialTextures[nonuniformEXT(am.opacity_tex)], p.zy * sc + of, 0.0)
            + textureLod(materialTextures[nonuniformEXT(am.opacity_tex)], p.xz * sc + of, 0.0)
            + textureLod(materialTextures[nonuniformEXT(am.opacity_tex)], p.xy * sc + of, 0.0);
    tx *= (1.0 / 3.0);

    float mask = ((am.flags & MAT_FLAG_OPACITY_RGBA) != 0u) ? tx.a : tx.r;
    if (mask >= 0.5) return 0.0;                            // solid here

    // 1.0 is enough to carry even a fully-interior cell (density 1.0) below the
    // 0.5 ISO threshold, so a masked region is gone THROUGH the body rather than
    // dimpled on its surface. A smaller push would erode the silhouette and
    // leave the interior intact — a hole that closes as the liquid thickens,
    // which reads as the mask "not working in deep water".
    return 1.0;
}

// The ONE field every isosurface consumer must read. Both ISO threshold sites
// and all six gradient samples go through here, so the surface that gas is
// clipped against, the surface that is shaded, and the normal used to shade it
// can never come from different fields.
// `acc` is inout for the same reason sampleDensityAcc takes it that way: it is
// a persistent NanoVDB read accessor whose node cache must survive the call.
float sampleIsoField(VkVolumeInstance vol, vec3 worldPos, pnanovdb_buf_t buf,
                     pnanovdb_map_handle_t mapH, inout pnanovdb_readaccessor_t acc) {
    return sampleDensityAccMode(vol, worldPos, buf, mapH, acc, false)
         - isoPoreOffset(vol, worldPos)
         - isoAlphaOffset(vol, worldPos);
}

#include "surface_sdf_empty.glsl"

// ============================================================
// Density Sampling — supports homogeneous and procedural noise
// ============================================================
float sampleDensityChannels(VkVolumeInstance vol, vec3 worldPos, out float emissionDensity) {
    emissionDensity = 0.0;
    // Transform world pos → object space
    vec3 localPos;
    localPos.x = vol.inv_transform[0] * worldPos.x + vol.inv_transform[1] * worldPos.y 
               + vol.inv_transform[2] * worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4] * worldPos.x + vol.inv_transform[5] * worldPos.y 
               + vol.inv_transform[6] * worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8] * worldPos.x + vol.inv_transform[9] * worldPos.y 
               + vol.inv_transform[10] * worldPos.z + vol.inv_transform[11];
    
    // Check against real bounding box instead of [-0.5, 0.5]^3
    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) {
        return 0.0;
    }
    
    float density = 1.0;
    
    if (vol.volume_type == 0) {
        // Homogeneous: constant density
        density = 1.0;
    } else if (vol.volume_type == 1) {
        // Procedural noise: convert localPos from world scale back to standard normalized coords
        // The procedural noise historically mapped [-0.5, 0.5] to fit bounds.
        // We remap the precise boundary.
        vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
        vec3 noiseCoord = normPos * vol.noise_scale;
        density = fbmNoise(noiseCoord, 4);
        
        // Smooth falloff near edges
        vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
        float edgeFalloff = min(min(edgeDist.x, edgeDist.y), edgeDist.z);
        density *= smoothstep(0.0, 0.1, edgeFalloff);
        
    } else if (vol.volume_type == 2) {
        // NanoVDB grid sampling.
        if (vol.vdb_grid_address != 0) {
            // Apply OptiX pivot parity since NanoVDB indexing assumes raw bounding spatial coordinates
            vec3 vdbWorldPos = localPos;
            vdbWorldPos.x -= vol.pivot_offset[0];
            vdbWorldPos.y -= vol.pivot_offset[1];
            vdbWorldPos.z -= vol.pivot_offset[2];
            
            density = sampleNanoVDBFloatTrilinear(vol.vdb_grid_address, vdbWorldPos);
        } else {
            // Fallback: procedural noise
            vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
            vec3 noiseCoord = normPos * max(vol.noise_scale, 1.0);
            density = fbmNoise(noiseCoord, 4);
            vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
            density *= smoothstep(0.0, 0.1, min(min(edgeDist.x, edgeDist.y), edgeDist.z));
        }
    } else if (vol.volume_type == 4 && vol.source_type == 5) {
        density = sampleDenseGasFloat(vol.vdb_grid_address, vol, localPos);
    }
    
    density = applyMaterialDensityNoise(vol, localPos, density);
    // Apply density remap (No upper clamp, matches OptiX fmaxf)
    float normalized = max((density - vol.density_remap_low) /
        max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0);
    vec2 channels = volumeDensityChannels(normalized, vol.density_curve,
        max(vol._reserved[0], 0.0), volumeEmissionSamplingMultiplier(
            vol.density_multiplier, vol._retired_cloud[0]), vol._retired_cloud[0]);
    emissionDensity = channels.y;
    return channels.x;
}

float sampleDensity(VkVolumeInstance vol, vec3 worldPos) {
    float unusedEmission;
    return volumeEmissionOpticalDensity(sampleDensityChannels(vol, worldPos, unusedEmission),
        vol.density_multiplier, vol._retired_cloud[0]);
}

// Accessor-aware lightMarch. Reuses the caller's density-grid accessor so the
// shadow-march inner loop also benefits from leaf-level cache hits.
#include "volume_rt_shadow_march.glsl"

// ============================================================
// Light March — estimate transmittance from scatter point toward light
// (OptiX vol_light_steps eşdeğeri)
// ============================================================
float lightMarch(VkVolumeInstance vol, vec3 pos, vec3 lightDir, float maxDist) {
    if (vol.shadow_steps <= 0) return 1.0;
    if (maxDist <= 1e-4) return 1.0;
    
    // [SHADOW FIX] Match OptiX: s_step_world = world_vol_extent / (shadow_steps * 2)
    // This covers half of maxDist with shadow_steps samples.
    // OLD code used  min(maxDist, step_size*2) / shadow_steps  which for a large cloud
    // (maxDist=10, step_size=0.1) would only march 0.2 world units — missing 98% of the
    // volume. That left dense cores fully lit → solid white.
    vec3 shadowAbsColor = max(vol.absorption_color, vec3(0.0));
    float shadowAbsPeak = max(
        shadowAbsColor.r, max(shadowAbsColor.g, shadowAbsColor.b));
    vec3 shadowAbsWeights = shadowAbsPeak > 1e-5
        ? shadowAbsColor / shadowAbsPeak : vec3(1.0);
    float sigma_t = vol.scatter_coefficient +
        vol.absorption_coefficient *
        dot(shadowAbsWeights, vec3(0.2126, 0.7152, 0.0722));
    if (sigma_t <= EPSILON) return 1.0;

    int reqSteps = clamp(vol.shadow_steps, 1, 64);
    float dMid = sampleDensity(vol, pos + lightDir * (0.5 * maxDist));
    float tauHint = max(0.0, dMid) * sigma_t * maxDist;
    if (tauHint <= 0.02) return 1.0;
    float stepScale = clamp(sqrt(tauHint), 0.25, 1.0);
    int steps = int(ceil(float(reqSteps) * stepScale));
    steps = clamp(steps, 3, min(reqSteps, 16));

    float stepSize = maxDist / (float(steps) * 2.0);
    // Clamp to avoid NaN/zero-step if shadow_steps was set very high
    stepSize = max(stepSize, 1e-5);
    float jitter = fract(sin(dot(pos, vec3(12.9898, 78.233, 37.719)) +
                             dot(lightDir, vec3(39.346, 11.135, 83.155))) * 43758.5453);
    
    // Accumulate optical depth (matches OptiX: density_sum += sigma_t * step)
    float s_trans = 0.0;
    uint measuredShadowSamples = 1u; // includes dMid used for adaptive step count
    for (int i = 0; i < steps; i++) {
        vec3 samplePos = pos + lightDir * (float(i) + jitter + 0.5) * stepSize;
        float d = sampleDensity(vol, samplePos);
        measuredShadowSamples++;
        s_trans += d * sigma_t * stepSize;
        if (s_trans > 10.0) break; // fully occluded
    }
    volumeRecordShadowSamples(measuredShadowSamples);
    
    // Multi-scatter shadow blend — matches OptiX:
    // Beer-Lambert shadow transmission. Shadow strength remains an artistic
    // blend between the physical result and an unshadowed result.
    float phys_trans = exp(-s_trans);
    
    float shadowStrength = clamp(vol.shadow_strength * 0.92, 0.0, 1.0);
    return 1.0 - shadowStrength * (1.0 - phys_trans);
}

// ============================================================
// Main — Volume Ray March Entry Point
// ============================================================
// Distance, in world-ray parameter units, from a point in the volume to the
// local AABB exit along the light direction.
