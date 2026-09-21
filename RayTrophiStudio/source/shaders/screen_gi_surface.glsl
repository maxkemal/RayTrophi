// Geometric surface reconstruction from this frame's coverage depth.
// Prefer the closer-depth neighbor to avoid crossing silhouette edges.
bool giPosition(ivec2 p, out vec3 world, out float depth) {
    world = vec3(0); depth = 1.0;
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, ivec2(pc.shape.xy)))) return false;
    depth = texelFetch(giDepth, p, 0).r;
    if (depth >= 1.0) return false;
    vec2 ndc = (vec2(p) + 0.5) / vec2(pc.shape.xy) * 2.0 - 1.0;
    vec4 h = pc.invViewProj * vec4(ndc, depth, 1.0);
    if (abs(h.w) < 1e-12) return false;
    world = h.xyz / h.w;
    return !any(isnan(world)) && !any(isinf(world));
}
bool giSurface(ivec2 p, out vec3 world, out vec3 normal, out float depth, out float bias) {
    normal = vec3(0); bias = 0.02;
    if (!giPosition(p, world, depth)) return false;
    vec3 a,b,c,d; float za,zb,zc,zd;
    bool va=giPosition(p+ivec2(-1,0),a,za), vb=giPosition(p+ivec2(1,0),b,zb);
    bool vc=giPosition(p+ivec2(0,-1),c,zc), vd=giPosition(p+ivec2(0,1),d,zd);
    if ((!va&&!vb)||(!vc&&!vd)) return false;
    vec3 dx = va && (!vb || abs(za-depth)<abs(zb-depth)) ? world-a : b-world;
    vec3 dy = vc && (!vd || abs(zc-depth)<abs(zd-depth)) ? world-c : d-world;
    vec3 n=cross(dx,dy);
    if (dot(n,n)<1e-16) return false;
    normal=normalize(n);
    vec2 ndc=(vec2(p)+0.5)/vec2(pc.shape.xy)*2.0-1.0;
    vec4 h=pc.invViewProj*vec4(ndc,max(depth-1e-6,0.0),1.0);
    if (abs(h.w)<1e-12) return false;
    vec3 towardCamera=h.xyz/h.w-world;
    if (dot(normal,towardCamera)<0.0) normal=-normal;
    bias=max(0.02,length(towardCamera)*3.0);
    return true;
}
