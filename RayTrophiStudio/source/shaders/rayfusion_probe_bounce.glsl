// RayFusion 1b-beta — tek diffuse sicrama, GERCEK malzeme okumasiyla.
//
// Hit tablolari BLAS'in kullandigi AYNI flat raster buffer'larindan turetilir.
// customIndex tam olarak TLAS instance sirasi, primitiveIndex indekssizdir.
#include "surface_coverage.h"
struct RfHitInstance {
    uvec2 positions;
    uvec2 materialIds;
    uvec2 uvs;          // 0 = mesh'in UV'si YOK; skalar renge dusulur
    uvec2 indices;      // 0 = mesh INDEKSSIZ (flat SoA, kose basina bir vertex)
    uint vertexCount;
    uint triangleCount; // primitiveIndex siniri -- indeksli mesh'te != vertexCount/3
    uint contentHash;
    uint pad;
};
// RayFusion::BounceMaterial ile birebir: 8 x 16 = 128 bayt.
struct RfBounceMaterial {
    vec4  diffuse;       // rgb albedo TINTI (doku carpar), w = destekleniyor
    vec4  emission;      // rgb emission * strength, w = opaklik skalari
    uvec4 textures;      // albedo, emission, opacity, metallic
    uvec4 textures2;     // specular, VkGpuMaterial::flags, uv wrap, thin foliage
    vec4  uvScaleOffset; // scale.xy, offset.xy
    vec4  uvTiling;      // tiling.xy, donme (derece), metallic skalari
    vec4  scalars;       // specular, alpha cutoff, thin transmission, coat
    // ★★★★★ TEK SAYIM SOZLESMESI: .x >= 0.5 ise bu malzemenin emission'i
    //   emissive UCGEN listesinde temsil ediliyor ve NEE onu ekleyecek -- bu
    //   yuzden asagida yarim kure isini emission'i EKLEMEZ. Bu alan olmadan
    //   emissive NEE, ayni isigi iki kez saymak olurdu.
    vec4  emissive;      // x: emission NEE listesinde (0/1)
};
struct RfBounceLight { vec4 position; vec4 radiance; vec4 direction; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RfPositions { float p[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RfMaterialIds { uint id[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RfUVs { float t[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RfIndices { uint i[]; };
layout(set=0, binding=4, std430) readonly buffer RfHitTable { RfHitInstance rfHits[]; };
layout(set=0, binding=5, std430) readonly buffer RfMaterialTable { RfBounceMaterial rfMaterials[]; };
layout(set=0, binding=6, std430) readonly buffer RfLightTable { RfBounceLight rfLights[]; };
// ★★★★★ Emissive UCGEN tablosu. Bunun ekledigi sey bir lob degil, bir
//   KESTIRICI: emissive geometri bugune kadar orneklenebilir bir isik degildi,
//   yani bir lamba ancak yarim kure isininin SANSINA bulunuyordu. Belirtisi de
//   "isik yok" degildi -- seed kasitli olarak kare numarasindan bagimsiz oldugu
//   icin TITREME degil sabit bir NOKTALI DESEN goruluyordu, ustune 5x5 blur
//   yayiyordu ve o filtrede luminance clamp yok.
// v0.w = artan normalize kumulatif alan (CDF), v1.w = bu ucgenin alani.
// ★★★★ Bagli descriptor'i yalnizca ONU TANIMLAYAN tuketici alir. `rfEmissives`
//   statik olarak kullanilan bir kaynak: tabloyu her yerde BEYAN etmek, onu
//   baglamayan gecislerde (probe trace) gecersiz bir descriptor demek olurdu.
//   Bu yuzden kapi DERLEME ZAMANINDA: RF_EMISSIVE_COUNT tanimlanmadiysa ne
//   binding vardir ne de emission bastirmasi.
#ifdef RF_EMISSIVE_COUNT
struct RfEmissiveTriangle { vec4 v0; vec4 v1; vec4 v2; vec4 radiance; };
layout(set=0, binding=12, std430) readonly buffer RfEmissiveTable { RfEmissiveTriangle rfEmissives[]; };
#endif
// * RT/raster hattinin okudugu AYNI bindless dizi, AYNI slot numaralariyla:
//   VkGpuMaterial::albedo_tex vb. dogrudan bu diziye indekstir. Ayri bir doku
//   listesi cikarmak iki dogruluk kaynagi demek olurdu.
layout(set=0, binding=7) uniform sampler2D rfTextures[];
// *** Bu sayac bu dilimin KABUL ALETI. CPU malzeme sayisi "kac materyal
//   uygun" der; olcum sorusu ise "kac isin gercekten golgelendi". hits > 0
//   iken shaded == 0, bounce'un goruntuyu hic degistirmedigi anlamina gelir
//   ve goruntu 1b-alpha ile birebir ayni cikar -- dogru gorunen bir hiclik.
// ★★★★★ Sayaclar NEDEN elendigini de soyler. `hits > 0 && shaded == 0` dogru
//   ama ise yaramaz bir olcumdu: bes ayri cikis yolu ayni sifiri uretiyordu.
//   Toplamlari + rfShadedCount == rfHitCount olmali; olmuyorsa adlandirilmamis
//   bir cikis yolu daha var. ABI: RayFusion::BounceCounters (48 bayt).
layout(set=0, binding=8, std430) buffer RfBounceCounters {
    uint rfHitCount;
    uint rfShadedCount;
    uint rfAlphaTested;
    uint rfAlphaOccluded;
    uint rfBackFaceShaded;  // arka yuz isabeti (artik golgeleniyor)
    uint rfSkipBounceOff;
    uint rfRejectUnresolved;
    uint rfRejectUnsupported;
    uint rfRejectDegenerate;
};
#ifndef RF_BOUNCE_COUNT
#define RF_BOUNCE_COUNT(counter) atomicAdd(counter, 1u)
#endif
#ifndef RF_ENV_SCALE
#define RF_ENV_SCALE 1.0
#endif

#include "pbr_texture_policy.glsl"

const uint RF_MAT_OPACITY_IN_ALPHA = (1u << 8);
bool rfValidTexture(uint slot) { return slot > 0u && slot < uint(pc.params2.w); }

uint rfHash(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }

// ★★★★★ YARIM KURE INTEGRALI ICIN DOGRU PREFILTERED SEVIYE.
//
//   Bu fonksiyon 2026-09-13'te bir GURULTU KOKU olarak eklendi. Env iskalama
//   yolu prefiltered zinciri MIP 0'dan okuyordu -- yani en keskin seviyeden.
//   Yarim kure integralini N ornekle kestirirken mip 0, AYNI MALIYETE
//   secilebilecek EN YUKSEK VARYANSLI seviyedir.
//
// ★★★ Ve bu bir yaklastirma KAYBI degil, KAZANCI: koni ortalamasi, yarim kure
//   integraline nokta orneginden DAHA YAKINDIR. Nokta ornegi dogru ortalamaya
//   yalnizca N buyurken yakinsar; koni ortalamasi N=1'de bile yakindir.
//
// ★★ Belirtisi neden atmosferli sahnelerde ve DOLAYLI aydinlatilan alanlarda
//   goruluyordu: Nishita gokyuzunde radyans gradyani cok buyuk (parlak
//   ufuk/gunes bolgesi vs sonuk zenit) ve dolayli alanlarda env BASKIN terim --
//   onu maskeleyecek guclu bir dogrudan isik yok.
//
// N ornegin her biri yarim kurenin ~1/N'ini kapsar; o koni roughness ~1/sqrt(N)
// demektir ve prefiltered zincir roughness'i `roughness * RF_ENV_MIP_SCALE` ile
// mip'e ceviriyor -- fragment shader'in `reflectionRoughness * 8.0` literal'iyle
// AYNI olcek.
#ifndef RF_ENV_MIP_SCALE
#define RF_ENV_MIP_SCALE 8.0
#endif
float rfEnvHemisphereLod(uint sampleCount) {
    float n = max(float(sampleCount), 1.0);
    return clamp(RF_ENV_MIP_SCALE * inversesqrt(n), 0.0, RF_ENV_MIP_SCALE);
}
float rfRandom(inout uint seed) { seed = rfHash(seed); return float(seed >> 8) / 16777216.0; }

vec3 rfReadPosition(RfPositions vertices, uint index) {
    return vec3(vertices.p[3u*index], vertices.p[3u*index+1u], vertices.p[3u*index+2u]);
}
vec2 rfReadUV(RfUVs coords, uint index) {
    return vec2(coords.t[2u*index], coords.t[2u*index+1u]);
}

// closesthit.rchit'teki applyMaterialUVTransform ile AYNI olmak ZORUNDA.
// Ayrisirsa sicrama, gorunen yuzeyin gosterdiginden BASKA bir texel okur ve
// belirti "GI rengi biraz tuhaf" olur -- kimsenin bug diye raporlamayacagi bir
// sey. Bu, bu deponun "uretici != tuketici" hata sinifinin tam ornegi.
float rfWrapRepeat(float x) { float r = fract(x); return r < 0.0 ? r + 1.0 : r; }
float rfWrapMirror(float x) {
    float r = mod(x, 2.0);
    if (r < 0.0) r += 2.0;
    return (r > 1.0) ? (2.0 - r) : r;
}
vec2 rfApplyUVTransform(RfBounceMaterial m, vec2 originalUV) {
    vec2 uv = originalUV - vec2(0.5);
    uv *= m.uvScaleOffset.xy;
    float angleRad = radians(m.uvTiling.z);
    float c = cos(angleRad), s = sin(angleRad);
    uv = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
    uv += vec2(0.5);
    uv += m.uvScaleOffset.zw;
    uv *= m.uvTiling.xy;
    uint wrap = m.textures2.z;
    if (wrap == 1u) return vec2(rfWrapMirror(uv.x), rfWrapMirror(uv.y));
    if (wrap == 2u) return clamp(uv, vec2(0.0), vec2(1.0));
    if (wrap == 3u) return originalUV;
    // 0 ve taninmayan modlar repeat. Sampler zaten repeat: burada sarmak,
    // clamp/mirror modlarinin sampler'a degil MALZEMEYE ait olmasi icindir.
    return vec2(rfWrapRepeat(uv.x), rfWrapRepeat(uv.y));
}

// Bir isabetin malzeme kaydini ve donusturulmus UV'sini cozer.
// uvValid false ise mesh'in UV'si yoktur ve doku OKUNMAZ -- uydurma bir (0,0)
// koordinati, dokunun sol alt kosesini butun yuzeye boyamak olurdu.
// positionsAddr bir uvec2 olarak doner, buffer_reference olarak DEGIL: bir
// referans tipini out parametresi yapmak sürücüden sürücüye degisen bir
// alandir ve burada kazanci yoktur -- adresten referans kurmak tek satir.
bool rfResolveHit(uint instance, uint triangle, vec2 bary,
                  out RfBounceMaterial mat, out vec2 uv, out bool uvValid,
                  out uvec2 positionsAddr, out uvec3 corner) {
    mat = rfMaterials[0];
    uv = vec2(0.0);
    uvValid = false;
    positionsAddr = uvec2(0);
    corner = uvec3(0u);
    if (instance >= rfHits.length()) return false;
    RfHitInstance hit = rfHits[instance];
    if (triangle >= hit.triangleCount ||
        all(equal(hit.positions, uvec2(0))) || all(equal(hit.materialIds, uvec2(0)))) return false;
    // ★★★★★ primitiveIndex BLAS'in ucgenlenmesine gore cozulur. Welded (indeksli)
    //   mesh'te `triangle*3`u dogrudan vertex dizisine sokmak, depolama
    //   sirasindan UYDURULMUS bir ucgen okur -- hatasiz, ama var olmayan yuzey.
    corner = uvec3(triangle * 3u, triangle * 3u + 1u, triangle * 3u + 2u);
    if (!all(equal(hit.indices, uvec2(0)))) {
        RfIndices table = RfIndices(hit.indices);
        corner = uvec3(table.i[corner.x], table.i[corner.y], table.i[corner.z]);
        if (any(greaterThanEqual(corner, uvec3(hit.vertexCount)))) return false;
    }
    uint material = RfMaterialIds(hit.materialIds).id[corner.x] & 0x7fffffffu;
    if (material >= rfMaterials.length()) return false;
    mat = rfMaterials[material];
    positionsAddr = hit.positions;
    if (!all(equal(hit.uvs, uvec2(0)))) {
        RfUVs coords = RfUVs(hit.uvs);
        vec2 a = rfReadUV(coords, corner.x);
        vec2 b = rfReadUV(coords, corner.y);
        vec2 c = rfReadUV(coords, corner.z);
        vec2 interpolated = a * (1.0 - bary.x - bary.y) + b * bary.x + c * bary.y;
        interpolated.y = 1.0 - interpolated.y; // same flat UV convention as RT shadow/prepass
        uv = rfApplyUVTransform(mat, interpolated);
        uvValid = true;
    }
    return true;
}

// Coverage and bounce eligibility are separate: unsupported BSDFs still
// respect authored/preview cutout holes; covered unsupported hits stay dark.
bool rfCandidateOccludes(uint instance, uint triangle, vec2 bary) {
    RfBounceMaterial mat;
    vec2 uv; bool uvValid; uvec2 positionsAddr; uvec3 corner;
    if (!rfResolveHit(instance, triangle, bary, mat, uv, uvValid, positionsAddr, corner)) return true;
    uint opacityTex = mat.textures.z;
    // Coverage is independent of whether we can shade the hit's BSDF.
    if (mat.diffuse.w < 0.5 && (mat.textures2.y & MATERIAL_FLAGS_PREVIEW_CUTOUT) == 0u) return true;
    if (opacityTex == 0u || !uvValid) return materialCoverageOpacity(mat.emission.w, (mat.textures2.y & MATERIAL_FLAGS_PREVIEW_CUTOUT) != 0u) >= mat.scalars.y;
    if (!rfValidTexture(opacityTex)) return true;
    RF_BOUNCE_COUNT(rfAlphaTested);
    // textureLod: compute'ta ortulu turev yoktur, ortulu ornekleme tanimsizdir.
    vec4 texel = textureLod(rfTextures[nonuniformEXT(int(opacityTex))], uv, 0.0);
    float alpha = ((mat.textures2.y & RF_MAT_OPACITY_IN_ALPHA) != 0u || opacityTex == mat.textures.x) ? texel.a : texel.r;
    bool occludes = materialCoverageOpacity(alpha * mat.emission.w, (mat.textures2.y & MATERIAL_FLAGS_PREVIEW_CUTOUT) != 0u) >= mat.scalars.y;
    if (occludes) { RF_BOUNCE_COUNT(rfAlphaOccluded); }
    return occludes;
}

bool rfThinHit(uint instance, uint triangle, vec2 bary) {
    RfBounceMaterial mat;
    vec2 uv; bool uvValid; uvec2 positionsAddr; uvec3 corner;
    return rfResolveHit(instance, triangle, bary, mat, uv, uvValid, positionsAddr, corner)
        && mat.diffuse.w >= 0.5 && mat.textures2.w != 0u;
}

// Serbest yol testi. gl_RayFlagsOpaqueEXT KALDIRILDI: alpha maskeli yaprak ve
// perde, opak bayragiyla kati bir duvar gibi golge yapardi. Bedeli aday
// dongusudur ve olculur (rfAlphaTested / rfAlphaOccluded).
bool rfClear(vec3 p, vec3 d, float distance) {
    if (distance <= 0.001) return false;
    rayQueryEXT shadow;
    rayQueryInitializeEXT(shadow, rfScene, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsNoOpaqueEXT, 0xFFu,
        p, 0.001, d, distance);
    while (rayQueryProceedEXT(shadow)) {
        if (rayQueryGetIntersectionTypeEXT(shadow, false) ==
            gl_RayQueryCandidateIntersectionTriangleEXT) {
            if (rfCandidateOccludes(
                    uint(rayQueryGetIntersectionInstanceCustomIndexEXT(shadow, false)),
                    uint(rayQueryGetIntersectionPrimitiveIndexEXT(shadow, false)),
                    rayQueryGetIntersectionBarycentricsEXT(shadow, false)))
                rayQueryConfirmIntersectionEXT(shadow);
        } else if (rayQueryGetIntersectionTypeEXT(shadow, false) ==
            gl_RayQueryCandidateIntersectionAABBEXT) {
            rayQueryGenerateIntersectionEXT(shadow, rayQueryGetIntersectionTEXT(shadow, false));
        }
    }
    return rayQueryGetIntersectionTypeEXT(shadow, true) == gl_RayQueryCommittedIntersectionNoneEXT;
}

// ★★★★★ EMISSIVE UCGEN NEE.
//
//   Eklenen sey bir lob degil, bir KESTIRICI. Emissive yuzeylerin yakin alani
//   aydinlatmasi zaten vardi (asagida `emission` donuyor); eksik olan, emissive
//   geometrinin ORNEKLENEBILIR bir isik olmamasiydi -- bir lamba ancak yarim
//   kure isininin SANSINA bulunuyordu, 1-4 ornek/piksel ile. Belirtisi de "isik
//   yok" degildi: seed kasitli olarak kare numarasindan bagimsiz oldugu icin
//   TITREME degil sabit bir NOKTALI DESEN goruluyordu, ustune 5x5 blur
//   yayiyordu ve o filtrede luminance clamp yok.
//
// ★★★ `RF_EMISSIVE_COUNT` tuketici basina tanimlanir ve 0 ise bu yol TAMAMEN
//   kapalidir -- emission bastirmasi da kapali (asagiya bak). Ikisi AYNI kapiya
//   bagli olmak zorunda: biri acik biri kapali kalsa, o malzemenin emission'i
//   hem NEE'den (liste yok) hem yarim kure isinindan (bastirildi) duserdi.
#ifndef RF_EMISSIVE_COUNT
#define RF_EMISSIVE_COUNT 0u
#define RF_EMISSIVE_DISABLED 1
#endif

// Donus, analitik isik dongusuyle AYNI birimde: diffuse BRDF ile carpilmaya
// hazir (1/PI zaten uygulanmis).
vec3 rfEmissiveNee(vec3 p, vec3 n, inout uint seed) {
#ifdef RF_EMISSIVE_DISABLED
    return vec3(0.0);
#else
    uint count = RF_EMISSIVE_COUNT;
    if (count == 0u) return vec3(0.0);

    // Alan-agirlikli secim: CDF ikili aramasi. Uniform secim, buyuk bir
    // emissive duvari ile kucuk bir filamani AYNI olasilikla secerdi ve varyans
    // duvarin katkisinda patlardi.
    float u = rfRandom(seed);
    uint lo = 0u, hi = count - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (rfEmissives[mid].v0.w < u) lo = mid + 1u; else hi = mid;
    }
    RfEmissiveTriangle tri = rfEmissives[lo];
    float totalArea = tri.v2.w;
    if (!(tri.v1.w > 0.0) || !(totalArea > 0.0)) return vec3(0.0);

    vec3 a = tri.v0.xyz, b = tri.v1.xyz, c = tri.v2.xyz;
    // Ucgen uzerinde uniform nokta (karekok warp).
    float root = sqrt(rfRandom(seed));
    float b0 = 1.0 - root, b1 = rfRandom(seed) * root;
    vec3 point = a * b0 + b * b1 + c * (1.0 - b0 - b1);

    vec3 toLight = point - p;
    float d2 = dot(toLight, toLight);
    if (d2 < 1e-8) return vec3(0.0);
    float distance = sqrt(d2);
    vec3 l = toLight / distance;
    float cosSurface = dot(n, l);
    if (cosSurface <= 0.0) return vec3(0.0);

    vec3 lightNormal = cross(b - a, c - a);
    if (dot(lightNormal, lightNormal) < 1e-16) return vec3(0.0);
    lightNormal = normalize(lightNormal);
    // ★ Iki tarafli: bu depoda emissive yuzeylerin yonu yazari tarafindan
    //   garanti edilmiyor ve tek tarafli yapmak, lambanin yarisini sessizce
    //   sondururdu. `abs` bu karari GORUNUR kilar.
    float cosLight = abs(dot(lightNormal, -l));
    if (cosLight <= 0.0) return vec3(0.0);

    // ★ Golge isini ucgenin BIRAZ ONUNDE biter: tam mesafede bitirmek, isigin
    //   kendi yuzeyini occluder saymasidir ve belirtisi "lamba kendini
    //   golgeliyor", yani hic aydinlatmamasidir.
    if (!rfClear(p, l, max(distance - 0.01, 0.0))) return vec3(0.0);

    // ★★★★ FIREFLY KOKUNU KAYNAKTA KES. Gurultunun bu yoldaki kaynagi 1/d^2:
    //   yuzey emissive ucgene cok yaklastiginda katki sinirsiz buyur ve tek
    //   piksel patlar. Kesme SIHIRLI BIR SABIT DEGIL, boyutsal bir arguman:
    //   sonlu alanli bir isiga kendi LINEER OLCEGINDEN daha yakin bir noktada
    //   tek alan ornegi zaten gecerli degildir (ucgen artik nokta isik gibi
    //   davranmiyor). Bu yuzden d^2 asagidan ucgenin kendi alaniyla sinirlanir.
    // ★ Kesme yalnizca DEJENERE yakinlikta devreye girer; normal mesafelerde
    //   ifade birebir aynidir, yani "gurultuyu azalttim ama isik da azaldi"
    //   takasi yok.
    float safeD2 = max(d2, tri.v1.w);
    return tri.radiance.rgb * (cosSurface * cosLight / safeD2) * totalArea / RF_PI;
#endif
}

// Committed hit DEGERLERI gecilir; opak ray query bir cikis parametresi olarak
// asla gecilmez.
vec3 rfBounceRadiance(uint instance, uint triangle, mat4x3 objectToWorld, vec2 bary,
                      vec3 hitPosition, vec3 rayDirection, uint seed) {
    RfBounceMaterial m;
    vec2 uv; bool uvValid; uvec2 positionsAddr; uvec3 corner;
    if (!rfResolveHit(instance, triangle, bary, m, uv, uvValid, positionsAddr, corner)) {
        RF_BOUNCE_COUNT(rfRejectUnresolved); return vec3(0);
    }
    if (m.diffuse.w < 0.5) { RF_BOUNCE_COUNT(rfRejectUnsupported); return vec3(0); }

    // Only committed, coverage-tested hits reach shading.
    vec3 albedo = m.diffuse.rgb;
    if (rfValidTexture(m.textures.x) && uvValid)
        albedo *= max(textureLod(rfTextures[nonuniformEXT(int(m.textures.x))], uv, 0.0).rgb, vec3(0));
    vec3 emission = m.emission.rgb;
    if (rfValidTexture(m.textures.y) && uvValid)
        emission *= max(textureLod(rfTextures[nonuniformEXT(int(m.textures.y))], uv, 0.0).rgb, vec3(0));

    // * Enerji payi artik BURADA hesaplaniyor, CPU'da degil: metallic bir
    //   dokudan gelebiliyor ve texel basina degisiyor. CPU'da katlanan sabit,
    //   metallic haritasi olan bir malzemede sessizce yanlis olurdu.
    float metallic = m.uvTiling.w;
    if (rfValidTexture(m.textures.w) && uvValid)
        metallic = samplePackedMetallic(
            textureLod(rfTextures[nonuniformEXT(int(m.textures.w))], uv, 0.0), m.textures2.y);
    float specular = m.scalars.x;
    if (rfValidTexture(m.textures2.x) && uvValid)
        specular *= clamp(textureLod(rfTextures[nonuniformEXT(int(m.textures2.x))], uv, 0.0).r, 0.0, 1.0);
    float f0 = clamp(0.08 * specular, 0.0, 1.0);
    vec3 diffuse = albedo * ((1.0 - clamp(metallic, 0.0, 1.0)) * (1.0 - (f0 + (1.0 - f0) / 21.0)));
    diffuse *= 1.0 - 0.04 * m.scalars.w;

    RfPositions vertices = RfPositions(positionsAddr);
    vec3 a = objectToWorld * vec4(rfReadPosition(vertices, corner.x), 1);
    vec3 b = objectToWorld * vec4(rfReadPosition(vertices, corner.y), 1);
    vec3 c = objectToWorld * vec4(rfReadPosition(vertices, corner.z), 1);
    vec3 n = cross(b-a, c-a);
    if (dot(n, n) < 1e-16) { RF_BOUNCE_COUNT(rfRejectDegenerate); return vec3(0); }
    n = normalize(n);
    if (dot(n, rayDirection) > 0.0) n = -n;
    // Sample one of the two diffuse hemispheres. Probability equals lobe
    // weight, so throughput stays bounded without adding a second sky ray.
    float thinTransmission = clamp(m.scalars.z, 0.0, 1.0);
    if (thinTransmission > 0.0 && rfRandom(seed) < thinTransmission) n = -n;
    vec3 p = hitPosition + n * 0.005;
    RF_BOUNCE_COUNT(rfShadedCount);
    // One cosine-weighted sample: Lambert / PDF cancels to diffuse reflectance.
    // Secondary hits do NOT read probes or recurse: exactly one diffuse bounce.
    float u = rfRandom(seed), phi = 2.0 * RF_PI * rfRandom(seed);
    vec3 tangent = normalize(cross(abs(n.y) < 0.99 ? vec3(0,1,0) : vec3(1,0,0), n));
    vec3 d = normalize(tangent * (sqrt(u)*cos(phi)) + cross(n,tangent) * (sqrt(u)*sin(phi)) + n*sqrt(1.0-u));
    // ★★★ Ikinci sicrama HER ZAMAN tek cosine ornegi, yani koni TAM yarim
    //   kure: `rfEnvHemisphereLod(1u)`. Eskiden mip 0 okunuyordu ve tek ornekle
    //   yuksek frekansli bir gokyuzunu ornekleyen bu satir, GI gurultusunun
    //   ikinci kaynagiydi.
    vec3 incoming = rfClear(p, d, pc.params.z)
        ? max(textureLod(rfEnvRadiance, rfDirectionToUv(d),
                         rfEnvHemisphereLod(1u)).rgb, vec3(0)) * RF_ENV_SCALE : vec3(0);
    // ★★★★★ IKINCI sicramada da emissive NEE. Burada cift sayim YOK: bu
    //   noktadan cikan yarim kure isini (`incoming`, yukarida) yalnizca
    //   GOKYUZUNU topluyor -- `rfClear` engellenmisse sifir doner, ucuncu bir
    //   yuzeyin emission'ini hic okumaz.
    incoming += rfEmissiveNee(p, n, seed);
    uint lightCount = uint(pc.params2.z);
    if (lightCount > 0u) {
        uint selected = min(uint(rfRandom(seed) * float(lightCount)), lightCount-1u);
        RfBounceLight light = rfLights[selected];
        vec3 delta = light.position.w > 0.5 ? light.direction.xyz : light.position.xyz-p;
        float d2 = dot(delta,delta);
        if (d2 > 1e-8) {
            vec3 l = delta * inversesqrt(d2);
            float distance = light.position.w > 0.5 ? pc.params.z : max(sqrt(d2)-0.005, 0.0);
            float cosine = max(dot(n,l), 0.0);
            if (cosine > 0 && rfClear(p, l, distance)) {
                float falloff = light.position.w > 0.5 ? 1.0 : 1.0/d2;
                incoming += light.radiance.rgb * (cosine * falloff * float(lightCount) / RF_PI);
            }
        }
    }
    // ★★★★★ TEK SAYIM KAPISI. Bu malzemenin emission'i emissive ucgen
    //   listesinde temsil ediliyorsa NEE onu ZATEN bir kez ekledi; burada
    //   tekrar eklemek ayni isigi iki kez saymaktir ve belirtisi "lambalar iki
    //   kat parlak" degil "GI biraz fazla sicak" olur -- kimse buna cift sayim
    //   demez.
    // ★★ `RF_EMISSIVE_COUNT == 0u` olan tuketicide (listeyi hic baglamayan yol)
    //   bastirma KAPALI kalir. Iki kapiyi ayirmak, o yolda emission'in hem
    //   NEE'den hem buradan dusmesi olurdu.
    bool representedInNee = RF_EMISSIVE_COUNT > 0u && m.emissive.x >= 0.5;
    return (representedInNee ? vec3(0.0) : emission) + diffuse * incoming;
}

vec3 rfHairBounceRadiance(uint instance, vec3 hitPosition, vec3 rayDirection, uint seed) {
    uint packedBits = floatBitsToUint(pc.params2.z);
    uint hairCount = packedBits >> 16u;
    if (hairCount == 0u) return vec3(0.0);
    uint meshInstanceCount = uint(rfHits.length());
    uint hairIdx = 0;
    if (instance >= meshInstanceCount) hairIdx = instance - meshInstanceCount;
    uint matIdx = uint(rfMaterials.length()) - hairCount + (hairIdx % hairCount);
    RfBounceMaterial m = rfMaterials[matIdx];
    
    vec3 hairColor = m.diffuse.rgb;
    vec3 n = -rayDirection; 
    vec3 p = hitPosition + n * 0.001;

    float u = rfRandom(seed), phi = 2.0 * RF_PI * rfRandom(seed);
    vec3 tangent = normalize(cross(abs(n.y) < 0.99 ? vec3(0,1,0) : vec3(1,0,0), n));
    vec3 d = normalize(tangent * (sqrt(u)*cos(phi)) + cross(n,tangent) * (sqrt(u)*sin(phi)) + n*sqrt(1.0-u));
    
    vec3 incoming = rfClear(p, d, pc.params.z)
        ? max(textureLod(rfEnvRadiance, rfDirectionToUv(d),
                         rfEnvHemisphereLod(1u)).rgb, vec3(0)) * RF_ENV_SCALE : vec3(0);
    incoming += rfEmissiveNee(p, n, seed);
    
    uint lightCount = packedBits & 0xFFFFu;
    if (lightCount > 0u) {
        uint selected = min(uint(rfRandom(seed) * float(lightCount)), lightCount-1u);
        RfBounceLight light = rfLights[selected];
        vec3 delta = light.position.w > 0.5 ? light.direction.xyz : light.position.xyz-p;
        float d2 = dot(delta,delta);
        if (d2 > 1e-8) {
            vec3 l = delta * inversesqrt(d2);
            float distance = light.position.w > 0.5 ? pc.params.z : max(sqrt(d2)-0.001, 0.0);
            float cosine = max(dot(n,l), 0.0);
            if (cosine > 0 && rfClear(p, l, distance)) {
                float falloff = light.position.w > 0.5 ? 1.0 : 1.0/d2;
                incoming += light.radiance.rgb * (cosine * falloff * float(lightCount) / RF_PI);
            }
        }
    }
    return hairColor * incoming * 0.318309886;
}
