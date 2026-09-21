// RayFusion probe field — GPU consumer side.
//
// Post-1b consumer: eight centre-aligned neighbours, normal weighting and
// directional distance visibility. Producer/SSBO ABI remains unchanged.
//
// ABI, RayFusion::ProbeTexel ile birebir: 2 x vec4 = 32 bayt, irradiance sonra
// distance. Sıralama "y * kProbeSide + x". C++ tarafı RayFusion/ProbeField.h.

const uint RF_PROBE_SIDE = 8u;
const uint RF_PROBE_TEXELS = RF_PROBE_SIDE * RF_PROBE_SIDE;

struct RfProbeTexel {
    vec4 irradiance; // rgb = scene-linear gelen difüz ışıma, a = slot geçerli mi
    vec4 distance;   // x = ortalama mesafe, y = ortalama kare, zw ayrılmış
};

layout(std430, set = 0, binding = 21) readonly buffer RayFusionProbeField {
    RfProbeTexel rfProbeTexels[];
};

layout(std430, set = 0, binding = 22) readonly buffer RayFusionProbeGrid {
    ivec4 rfGridCounts;   // xyz = hücre sayısı, w = yayınlanmış probe sayısı
    ivec4 rfGridMinimum;  // xyz = pencerenin en küçük hücresi, w = alan etkin mi
    vec4  rfGridSpacing;  // x = spacing, y = traced horizon (0 = sky bake), zw reserved
};

// Oktahedral çözme. C++ üreticisindeki rfOctDecode ile AYNI formül olmak
// zorunda: aynı texel'e yazan yön ile okuyan yön ayrışırsa belirti "ambient
// biraz yanlış" olur, ve o belirti kimse tarafından bug diye raporlanmaz.
vec3 rfOctDecode(vec2 e) {
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        vec2 s = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

vec2 rfOctEncode(vec3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-8);
    if (n.z < 0.0) {
        vec2 s = vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * s;
    }
    return n.xy;
}

// ProbeField::slotFor ile birebir: MUTLAK hücre koordinatının torusal hash'i.
// Pencere kaydığında (scroll) aynı dünya hücresi aynı slota düşsün diye böyle.
uint rfSlotFor(ivec3 cell) {
    ivec3 n = rfGridCounts.xyz;
    ivec3 c = ((cell % n) + n) % n;
    return uint((c.z * n.y + c.y) * n.x + c.x);
}

uint rfDirectionalIndex(uint slot, vec3 direction) {
    vec2 oct = rfOctEncode(direction) * 0.5 + 0.5;
    ivec2 texel = clamp(ivec2(oct * float(RF_PROBE_SIDE)),
                        ivec2(0), ivec2(int(RF_PROBE_SIDE) - 1));
    return slot * RF_PROBE_TEXELS + uint(texel.y) * RF_PROBE_SIDE + uint(texel.x);
}

// Same bounded estimate as RayFusion::momentVisibility (world units).
float rfMomentVisibility(vec2 moments, float distanceToSurface) {
    if (any(isnan(moments)) || any(isinf(moments)) ||
        any(lessThan(moments, vec2(0.0)))) return 0.0;
    float meanSquared = moments.x * moments.x;
    if (moments.y + 1e-5 * max(meanSquared, 1.0) < meanSquared) return 0.0;
    if (distanceToSurface <= moments.x) return 1.0;
    float variance = max(moments.y - meanSquared, 0.0);
    float delta = distanceToSurface - moments.x;
    float visibility = variance / (variance + delta * delta);
    return visibility * visibility * visibility;
}

// false means no usable measurement, not occlusion. Valid blocked neighbours
// return true with black so the caller cannot reintroduce unoccluded sky.
bool rfSampleProbeField(vec3 worldPos, vec3 normal, out vec3 irradiance) {
    irradiance = vec3(0.0);
    if (rfGridMinimum.w == 0 || rfGridCounts.w == 0) return false;

    const float spacing = rfGridSpacing.x;
    if (!(spacing > 0.0) || isinf(spacing) ||
        any(lessThanEqual(rfGridCounts.xyz, ivec3(0))) ||
        any(isnan(worldPos)) || any(isinf(worldPos)) ||
        any(isnan(normal)) || any(isinf(normal)) ||
        dot(normal, normal) < 1e-12) return false;

    const ivec3 lo = rfGridMinimum.xyz;
    const ivec3 hi = lo + rfGridCounts.xyz;
    // ★★★★★ SINIRDA BIR HUCRELIK PAY. Eski hali burada sert bir `return false`
    //   idi ve belirti sudur: "duvarin yarisi mavi yarisi dogal renginde",
    //   arada gecis YOK.
    //
    // ★★★★ Nedeni, iki emniyetin AYNI YERDE susmasi: bu fonksiyon false
    //   donunce tuketici `worldIrradiance`a -- engelsiz gokyuzu kuresine --
    //   duser, ve `rfSpecularSkyVisibility` de tam ayni sinir testiyle 1.0
    //   doner. Yani izgaranin bir hucre disinda hem isima hem gorunurluk
    //   olcumu birden kayboluyor ve piksel MUMKUN OLAN EN PARLAK cevabi
    //   aliyor. Ikisi bagimsiz olsa bu bir gurultu olurdu; korelasyonlu
    //   olduklari icin KESKIN BIR CIZGI.
    //
    // ★★★ Bir hucre disaridaki yuzeyin isimasi, bir hucre iceridekinden
    //   farkli DEGIL: ayni oda, ayni duvar. Pay bu yuzden guvenli. Daha
    //   genisletmek guvenli DEGIL -- acik arazide gokyuzune geri dusmek
    //   DOGRU cevaptir, ve payi buyutmek odanin isimasini disari tasirdi.
    // Hucre KENETLENMEZ, SINIR genisler: kenetlemek testi totolojik yapar ve
    // acik arazideki bir pikseli de "izgara icinde" sayardi.
    const ivec3 cell = ivec3(floor(worldPos / spacing));
    if (any(lessThan(cell, lo - ivec3(1))) || any(greaterThan(cell, hi))) return false;
    // Asagidaki 8-komsu dongusu izgara DISINDAKI hucreleri zaten atliyor, yani
    // pay icindeki bir piksel yalnizca KENAR problarindan beslenir ve agirlik
    // toplami normalize edildigi icin deger sinirda SUREKLIDIR. Hicbir gecerli
    // komsu yoksa yine false doner -- pay bir olcum UYDURMAZ, var olani bir
    // hucre uzatir.

    const vec3 n = normalize(normal);
    // Producers place probes at (cell + 0.5) * spacing, not cell corners.
    const vec3 gridPosition = worldPos / spacing - vec3(0.5);
    const ivec3 base = ivec3(floor(gridPosition));
    const vec3 blend = fract(gridPosition);
    float baseWeightSum = 0.0;
    for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 2; ++x) {
        ivec3 offset = ivec3(x, y, z);
        ivec3 neighbour = base + offset;
        // Check before the toroidal hash: an outside cell aliases a live slot.
        if (any(lessThan(neighbour, lo)) || any(greaterThanEqual(neighbour, hi))) continue;
        vec3 weights = mix(vec3(1.0) - blend, blend, vec3(offset));
        float weight = weights.x * weights.y * weights.z;
        if (weight <= 0.0) continue;
        uint slot = rfSlotFor(neighbour);
        if (slot >= uint(rfGridCounts.w)) continue;
        RfProbeTexel packet = rfProbeTexels[rfDirectionalIndex(slot, n)];
        if (packet.irradiance.a < 0.5) continue;

        vec3 probeToSurface = worldPos - (vec3(neighbour) + vec3(0.5)) * spacing;
        float distanceToSurface = length(probeToSurface);
        vec3 direction = distanceToSurface > 1e-6 ? probeToSurface / distanceToSurface : n;
        // Wrapped normal avoids a zero denominator behind a surface. Visibility
        // still blocks those probes; no positive floor is applied to visibility.
        float facing = distanceToSurface > 1e-6 ? 0.5 * dot(n, -direction) + 0.5 : 1.0;
        weight *= max(facing * facing, 0.01);
        vec2 moments = rfProbeTexels[rfDirectionalIndex(slot, direction)].distance.xy;
        float visibility = rfMomentVisibility(moments, distanceToSurface);
        baseWeightSum += weight;
        irradiance += max(packet.irradiance.rgb, vec3(0.0)) * weight * visibility;
    }
    if (baseWeightSum <= 0.0) return false;
    // Normalize spatial/normal weights only. Normalizing visibility away would
    // restore a lone occluded probe to full brightness at the grid boundary.
    irradiance /= baseWeightSum;
    return true;
}
