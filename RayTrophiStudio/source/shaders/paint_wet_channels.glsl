// Existing Paint channel resolve, included after colour/state resolve.
// Atlas planes: colour / roughness / height mask / normal. Source planes
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

// Same five-tap gradient, 0.15 sensitivity, PD denominator floor and normal
// encoding as MeshPaintAdapter's existing height->normal path, except the
// green sign (see below).
uint channelNormal(Surface s, int x, int y, uint ti) {
    float gh = ((addedHeight(s,x+2,y)-addedHeight(s,x-2,y))*0.5 +
                 addedHeight(s,x+1,y)-addedHeight(s,x-1,y))*0.5;
    float gv = ((addedHeight(s,x,y+2)-addedHeight(s,x,y-2))*0.5 +
                 addedHeight(s,x,y+1)-addedHeight(s,x,y-1))*0.5;
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
    for (uint k = 0u; k < TEXELS_PER_THREAD; ++k) {
        int x,y; tileTexel(t,k,x,y);
        if (!inSurface(s,x,y)) continue;
        uint ti = texelIndex(s,x,y);
        // Enable over existing paint without clearing/restamping its colour.
        float coverage = unpackColor(color[ti]).a / 255.0;
        if (coverage > 0.0 && state[ti].z == 0.0) {
            float original = float(sourceChannel(2u,ti)&255u);
            float contribution = float(s.pbr_params>>16u)/65535.0;
            state[ti].z = clamp(max(0.0,(original-128.0)/127.0) +
                coverage*s.paint_load*clamp(contribution*1.6,0.0,1.0),0.0,1.0);
        }
    }
}

void channelTile(uint tile_index) {
    if (pc._p0 == 0u) return;
    Tile t = tiles[tile_index];
    Surface s = surfaces[t.surface];
    bool update = pc._p1 != 0u;
    // The derivative reaches two pixels beyond colour resolve's tile apron.
    // Dispatch after resolve so it reads the final height across workgroups.
    for (int oy = -2; oy <= 2 && !update; ++oy)
        for (int ox = -2; ox <= 2; ++ox)
            update = update || tileActive(s,int(t.tx)+ox,int(t.ty)+oy);
    if (!update) return;
    for (uint k = 0u; k < TEXELS_PER_THREAD; ++k) {
        int x,y; tileTexel(t,k,x,y);
        if (!inSurface(s,x,y)) continue;
        uint ti = texelIndex(s,x,y);
        vec4 layer = unpackColor(color[ti]);
        float coverage = clamp(layer.a/255.0,0.0,1.0);
        // Recreate colour also after lazy atlas expansion, preserving old paint.
        out_color[ti] = composite(base_color[ti],layer);
        uint roughness = sourceChannel(1u,ti);
        if ((s.flags&SURFACE_ROUGHNESS)!=0u && coverage > 0.0) {
            float wet = float(s.pbr_params&255u)/255.0;
            float dry = float((s.pbr_params>>8u)&255u)/255.0;
            float target = mix(dry,wet,clamp(state[ti].x,0.0,1.0));
            float original = float(roughness&255u)/255.0;
            float value = floor(mix(original,target,channelMask(coverage))*255.0+0.5);
            roughness = packColor(vec4(value,value,value,255.0));
        }
        out_color[pc._p0+ti] = roughness;
        uint height = sourceChannel(2u,ti);
        uint normal = sourceChannel(3u,ti);
        if ((s.flags&SURFACE_HEIGHT_NORMAL)!=0u) {
            float value = raisedHeight(s,x,y);
            height = packColor(vec4(value,value,value,255.0));
            normal = channelNormal(s,x,y,ti);
        }
        out_color[2u*pc._p0+ti] = height;
        out_color[3u*pc._p0+ti] = normal;
    }
    if (gl_LocalInvocationIndex == 0u) markDirty(t);
}
