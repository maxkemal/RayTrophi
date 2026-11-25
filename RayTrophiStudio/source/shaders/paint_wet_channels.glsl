// Existing Paint channel resolve, included after colour/state resolve.
// Source plane 6 (optional, pc._p1 bit 3): hidden picture RGBA8.
// Atlas planes: colour / roughness / height mask / normal / optional opacity. Source planes
// additionally carry the two float-bit UV Jacobian distances baked by CPU Paint.
uint sourceChannel(uint plane, uint texel) {
    return base_color[plane * pc._p0 + texel];
}

// Paint thins out toward the rim of a splat. Blending the channels linearly by
// coverage made a barely tinted fringe already read as glossy and bumpy, so the
// stain looked wider in roughness/normal than in colour. The channels follow
// the BODY of the paint: nothing below a quarter coverage, full above 0.85.
float channelMask(float coverage) {
    return smoothstep(0.25, 0.85, coverage);
}

float paintCoverage(uint ti) {
    return clamp(unpackColor(color[ti]).a / 255.0, 0.0, 1.0);
}

float raisedHeight(Surface s, int x, int y) {
    uint ti = texelIndex(s, clamp(x, 0, int(s.width) - 1), clamp(y, 0, int(s.height) - 1));
    float original = float(sourceChannel(2u, ti) & 255u);
    float thickness = clamp(state[ti].z, 0.0, 1.0) * channelMask(paintCoverage(ti));
    float raised = floor(128.0 + thickness * 127.0 + 0.5);
    return max(original, raised);
}

float addedHeight(Surface s, int x, int y) {
    x = clamp(x, 0, int(s.width) - 1);
    y = clamp(y, 0, int(s.height) - 1);
    uint ti = texelIndex(s, x, y);
    if (pos_valid[ti].w < 0.5) return 0.0;
    return (raisedHeight(s, x, y) - float(sourceChannel(2u, ti) & 255u)) / 255.0;
}

// ★ Tile cache of addedHeight over the tile plus its 2-texel apron (2026-11-24).
//   The five-tap gradient read addedHeight 8 times per texel, ~5 global loads
//   each (validity, source height twice, state, colour): ~40 loads per texel,
//   neighbours re-reading the same values. CHANNELS was the most expensive wet
//   paint kernel (0.92 ms per step for a four-balloon pop on an RTX 3060, far
//   more on Mali). Filled once per workgroup with the SAME clamped addedHeight
//   calls, so the gradient is bit-identical.
const int CH_APRON = 2;
const int CH_SIDE = int(TILE) + 2 * CH_APRON;   // 36
shared float s_added[CH_SIDE * CH_SIDE];

// Fills s_added for this tile. Every invocation of the workgroup must call it
// (it ends with a barrier); the caller's branch is workgroup-uniform.
void cacheAddedHeight(Tile t, Surface s) {
    int ox = int(t.tx * TILE) - CH_APRON, oy = int(t.ty * TILE) - CH_APRON;
    for (uint i = gl_LocalInvocationIndex; i < uint(CH_SIDE * CH_SIDE); i += 256u) {
        s_added[i] = addedHeight(s, ox + int(i % uint(CH_SIDE)), oy + int(i / uint(CH_SIDE)));
    }
    barrier();
}

float cachedAdded(Tile t, int x, int y) {
    int lx = x - (int(t.tx * TILE) - CH_APRON);
    int ly = y - (int(t.ty * TILE) - CH_APRON);
    return s_added[ly * CH_SIDE + lx];
}

// Same five-tap gradient, 0.15 sensitivity, PD denominator floor and normal
// encoding as MeshPaintAdapter's existing height->normal path, except the
// green sign (see below). Heights come from the tile cache (cacheAddedHeight).
uint channelNormal(Tile t, Surface s, int x, int y, uint ti) {
    float gh = ((cachedAdded(t,x+2,y)-cachedAdded(t,x-2,y))*0.5 +
                 cachedAdded(t,x+1,y)-cachedAdded(t,x-1,y))*0.5;
    float gv = ((cachedAdded(t,x,y+2)-cachedAdded(t,x,y-2))*0.5 +
                 cachedAdded(t,x,y+1)-cachedAdded(t,x,y-1))*0.5;
    uint original = sourceChannel(3u, ti);
    if (abs(gh) + abs(gv) < 1e-8 || s.normal_strength == 0.0) return original;
    vec2 span = vec2(uintBitsToFloat(sourceChannel(4u, ti)),
                     uintBitsToFloat(sourceChannel(5u, ti)));
    vec2 derivative = vec2(gh,gv) / max(span,vec2(1e-5)) * s.normal_strength * 0.15;
    // gv is the height slope toward image-DOWN (texel y grows downward), but the
    // renderers' bitangent points toward +v = image-UP (material_preview_frag
    // applyNormalMap negates B for exactly this). A normal leans away from
    // rising height, so green is +gv. With -gv every bump was lit upside down
    // on walls and a running drip read as paint climbing.
    // (MeshPaintAdapter's CPU height->normal still writes -dy.)
    vec3 detail = normalize(vec3(-derivative.x,derivative.y,1.0));
    vec3 base = normalize(unpackColor(original).rgb / 127.5 - 1.0);
    vec3 normal = normalize(vec3(base.xy / max(0.15,base.z) +
                                 detail.xy / max(0.15,detail.z),1.0));
    return packColor(vec4(floor(clamp(normal * 0.5 + 0.5,0.0,1.0)*255.0+0.5),255.0));
}

void seedHeightTile(uint tile_index) {
    if (pc._p0 == 0u) return;
    Tile t = tiles[tile_index];
    Surface s = surfaces[t.surface];
    if ((s.flags & SURFACE_HEIGHT_NORMAL) == 0u) return;
    float height_gain = paintHeightGain(s);
    if (height_gain <= 0.0) return;
    for (uint k = 0u; k < TEXELS_PER_THREAD; ++k) {
        int x,y; tileTexel(t,k,x,y);
        if (!inSurface(s,x,y)) continue;
        uint ti = texelIndex(s,x,y);
        // Enable over existing paint without clearing/restamping its colour.
        float coverage = unpackColor(color[ti]).a / 255.0;
        if (coverage > 0.0 && state[ti].z == 0.0) {
            float base_height = paintBaseHeight(ti);
            float ceiling = mix(base_height, 1.0, height_gain);
            state[ti].z = buildPaintHeight(base_height, ceiling, coverage * s.paint_load);
        }
    }
}

uint tileChangedBits(Surface s, int tx, int ty) {
    if (tx < 0 || ty < 0 || tx >= int(s.tiles_x) || ty >= int(s.tiles_y)) return 0u;
    return tile_flags[tileChangedIndex(s.tile_offset + uint(ty) * s.tiles_x + uint(tx))];
}

// What this tile's channel planes need this step (CHANGED_* bits, see
// tileChangedIndex): CONTENT = its own colour/wetness changed (roughness and
// colour planes), HEIGHT = a height within one tile of it changed (its normals
// read 2 texels across the edge). 0 = every plane is still current.
uint channelTileNeeds(Tile t, Surface s) {
    if (!changedBitsAvailable()) {
        // Old host (no changed region): the previous 5x5 ACTIVE apron.
        for (int oy = -2; oy <= 2; ++oy)
            for (int ox = -2; ox <= 2; ++ox)
                if (tileActive(s, int(t.tx) + ox, int(t.ty) + oy))
                    return CHANGED_CONTENT | CHANGED_HEIGHT;
        return 0u;
    }
    uint own = tileChangedBits(s, int(t.tx), int(t.ty));
    uint needs = (own & CHANGED_CONTENT) != 0u ? CHANGED_CONTENT : 0u;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            if ((tileChangedBits(s, int(t.tx) + ox, int(t.ty) + oy) & CHANGED_HEIGHT) != 0u) {
                return needs | CHANGED_HEIGHT;
            }
        }
    }
    return needs;
}

uint channelDispatchBase() { return 2u * pc.tile_count; }
uint channelListBase() { return channelDispatchBase() + 3u; }

void resetChannelList() {
    uint args = channelDispatchBase();
    tile_flags[args] = 0u;
    tile_flags[args + 1u] = 1u;
    tile_flags[args + 2u] = 1u;
}

shared uint s_channel_count;
shared uint s_channel_base;

void compactChannelTiles() {
    uint lane = gl_LocalInvocationIndex;
    uint tile_index = gl_GlobalInvocationID.x;
    if (lane == 0u) s_channel_count = 0u;
    barrier();

    bool active_tile = false;
    if (tile_index < pc.tile_count) {
        Tile t = tiles[tile_index];
        active_tile = channelTileNeeds(t, surfaces[t.surface]) != 0u;
    }

    uint local_slot = 0u;
    if (active_tile) local_slot = atomicAdd(s_channel_count, 1u);
    barrier();
    // Reserve once per workgroup to avoid a global atomic for each active tile.
    if (lane == 0u && s_channel_count != 0u) {
        s_channel_base = atomicAdd(tile_flags[channelDispatchBase()], s_channel_count);
    }
    barrier();
    if (active_tile) {
        tile_flags[channelListBase() + s_channel_base + local_slot] = tile_index;
    }
}

void channelTile(uint tile_index, bool edited) {
    if (pc._p0 == 0u) return;
    Tile t = tiles[tile_index];
    Surface s = surfaces[t.surface];
    // The derivative reaches two pixels beyond colour resolve's tile apron.
    // Dispatch after resolve so it reads the final height across workgroups.
    // A forced pass (bit 0: atlas expansion / seeded height) redoes every
    // plane; otherwise the changed bits decide, uniformly per workgroup.
    const bool forced = edited || (pc._p1 & 1u) != 0u;
    // For a settings edit the mask addresses the five output planes directly.
    // Ordinary simulation and full atlas initialization keep their existing path.
    const uint planes = edited ? pc.source_kind : 31u;
    if (gl_LocalInvocationIndex == 0u) {
        s_gate = forced ? (CHANGED_CONTENT | CHANGED_HEIGHT) : channelTileNeeds(t, s);
    }
    barrier();
    const uint needs = s_gate;
    if (needs == 0u) return;
    // One surface per tile: the flag is uniform across the workgroup.
    const bool height_normal = (s.flags & SURFACE_HEIGHT_NORMAL) != 0u;
    // Normals (and their height cache) only where a height near this tile
    // moved: a drying, still stain changes roughness, not its relief.
    const bool redo_height = height_normal && (needs & CHANGED_HEIGHT) != 0u;
    if (redo_height && (planes & 8u) != 0u) cacheAddedHeight(t, s);
    for (uint k = 0u; k < TEXELS_PER_THREAD; ++k) {
        int x,y; tileTexel(t,k,x,y);
        if (!inSurface(s,x,y)) continue;
        uint ti = texelIndex(s,x,y);
        vec4 layer = unpackColor(color[ti]);
        float coverage = clamp(layer.a/255.0,0.0,1.0);
        // Recreate colour also after lazy atlas expansion, preserving old paint.
        if ((planes & 1u) != 0u) {
            out_color[ti] = surfaceOutput(s, ti, layer, state[ti].w);
        }
        if ((planes & 2u) != 0u) {
            uint roughness = sourceChannel(1u, ti);
            if ((s.flags & SURFACE_ROUGHNESS) != 0u && coverage > 0.0) {
                float wet = float(s.pbr_params & 255u) / 255.0;
                float dry = float((s.pbr_params >> 8u) & 255u) / 255.0;
                float target = mix(dry, wet, clamp(state[ti].x, 0.0, 1.0));
                float original = float(roughness & 255u) / 255.0;
                float value = floor(mix(original, target, channelMask(coverage)) * 255.0 + 0.5);
                roughness = packColor(vec4(value, value, value, 255.0));
            }
            out_color[pc._p0 + ti] = roughness;
        }
        if (redo_height || !height_normal) {
            if ((planes & 4u) != 0u) {
                uint height = sourceChannel(2u, ti);
                if (redo_height) {
                    float value = raisedHeight(s, x, y);
                    height = packColor(vec4(value, value, value, 255.0));
                }
                out_color[2u * pc._p0 + ti] = height;
            }
            if ((planes & 8u) != 0u) {
                out_color[3u * pc._p0 + ti] = redo_height ? channelNormal(t, s, x, y, ti)
                                                        : sourceChannel(3u, ti);
            }
        }
        if ((planes & 16u) != 0u && (pc._p1 & 2u) != 0u) {
            uint originalOpacity = 0xFFFFFFFFu;
            uint opacity = originalOpacity;
            if ((s.flags & SURFACE_REVEAL_OPACITY) != 0u) {
                float original = float(originalOpacity & 255u);
                float visible = pos_valid[ti].w > 0.5 ? coverage : 0.0;
                float value = floor(original * visible + 0.5);
                opacity = packColor(vec4(value));
            }
            out_color[4u * pc._p0 + ti] = opacity;
        }
    }
    // The host queues the edited planes explicitly, distinct from simulation dirties.
    if (!edited && gl_LocalInvocationIndex == 0u) markDirty(t);
}
