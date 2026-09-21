#pragma once
#include <cstdint>
#include <string>
namespace RayFusion {
// Hit tables are derived from the same flat raster buffers used by the BLAS.
// customIndex is the exact TLAS instance order, primitiveIndex is non-indexed.
struct alignas(16) HitInstance {
    uint64_t positions = 0, materialIds = 0;
    // Same vertex order as `positions`, two floats per vertex. 0 means the mesh
    // carries no UVs at all -- a real state, not an error: the bounce then uses
    // the material's scalar colour instead of guessing a coordinate.
    uint64_t uvs = 0;
    // ★★★★★ WELDED (indexed) geometry. The raster upload welds every static
    //   non-skinned mesh and keeps an index buffer (see the welded branch in
    //   VulkanViewportBackend.cpp) -- terrain and every large static prop take
    //   that path. The BLAS used to be built from `positions` alone with
    //   VK_INDEX_TYPE_NONE, so the world the rays saw was vertices 0-1-2,
    //   3-4-5 ... in STORAGE order: fabricated triangles that are not the
    //   surface. The real surface was absent (light leaked through it) and the
    //   fabricated slabs occluded (flat shadows that belong to nothing). No
    //   error, no counter moved -- only the image was wrong, and only for the
    //   welded half of the scene, which is why de-indexed foliage looked right.
    //   0 means genuinely non-indexed: flat SoA, one vertex per corner.
    uint64_t indices = 0;
    uint32_t vertexCount = 0;
    // primitiveIndex bound. This is NOT vertexCount/3 once `indices` is set --
    // conflating the two is what made the out-of-range guard useless above.
    uint32_t triangleCount = 0;
    // Flat material-ID stream hash. Material reassignment can rewrite the GPU
    // buffer in place, so the ID stream has to reach the publication signature.
    uint32_t contentHash = 0;
    uint32_t pad = 0;
};
struct alignas(16) BounceMaterial {
    float diffuse[4]{};       // rgb albedo TINT (texture multiplies it); w = supported
    float emission[4]{};      // rgb emission colour * strength; w = opacity scalar
    uint32_t textures[4]{};   // albedo, emission, opacity, metallic (0 = none)
    uint32_t textures2[4]{};  // specular, VkGpuMaterial::flags, uv wrap mode, thin foliage
    float uvScaleOffset[4]{}; // scale.xy, offset.xy
    float uvTiling[4]{};      // tiling.xy, uv rotation degrees, metallic scalar
    float scalars[4]{};       // specular scalar, alpha cutoff, thin transmission, coat
    // ★★★★★ TEK SAYIM SOZLESMESI. `.x >= 0.5` ise bu malzemenin emission'i
    //   emissive UCGEN listesinde TAM OLARAK temsil ediliyor, yani NEE onu bir
    //   kez ekleyecek -- ve bu yuzden yarim kure isini ayni yuzeye carptiginda
    //   emission'i EKLEMEZ.
    //
    // ★★★ Bu alan olmadan emissive NEE eklemek, ayni isigi IKI kez saymak
    //   olurdu (bir kez NEE, bir kez isinin sansen o ucgene carpmasiyla) ve
    //   belirtisi "lambalar iki kat parlak" degil, "GI biraz fazla sicak"
    //   olurdu. Kimse buna cift sayim demez.
    //
    // ★★ Karar MALZEME granulunde, ucgen granulunde DEGIL: bir malzemenin
    //   emission'i ya butunuyle listede ya da hic degil. Yoksa reddedilen
    //   ucgenler (transparan, welded mesh, cap disi) emission'larini
    //   kaybederdi -- yani "bazi lambalar GI'da yok" gibi gorunurdu.
    float emissive[4]{};      // x: emission NEE listesinde temsil ediliyor (0/1)
};
struct alignas(16) BounceLight {
    float position[4]{}; // w: 0 point, 1 directional
    float radiance[4]{};
    float direction[4]{}; // toward light
};
// ★★★★★ Emissive UCGEN -- orneklenebilir bir isik olarak.
//
//   Emissive yuzeylerin yakin alani aydinlatmasi zaten YAPISAL olarak vardi:
//   `rfBounceRadiance` donusu `emission + diffuse * incoming`. Eksik olan bir
//   yetenek degil, KESTIRICIYDI: emissive geometri orneklenebilir bir isik
//   olmadigi icin bir lamba ancak yarim kure isininin SANSINA bulunuyordu,
//   1-4 ornek/piksel ile.
//
// ★ Belirtisi de bu yuzden "isik yok" degildi: seed kasitli olarak kare
//   numarasindan bagimsiz oldugu icin TITREME degil, sabit bir NOKTALI DESEN
//   goruluyordu -- ustune 5x5 blur yayiyor ve o filtrede luminance clamp yok.
//
// Alan-agirlikli secim icin `cdf` kullanilir: liste kurulurken ucgen alanlarinin
// normalize edilmis kumulatifi yazilir, GPU ikili arama yapar. Uniform secim
// buyuk bir emissive duvari ile kucuk bir filamani AYNI olasilikla secerdi.
struct alignas(16) EmissiveTriangle {
    float v0[4]{};       // xyz dunya uzayinda, w = normalize kumulatif alan (artan)
    float v1[4]{};       // xyz dunya uzayinda, w = bu ucgenin alani
    float v2[4]{};       // xyz dunya uzayinda, w kullanilmiyor
    float radiance[4]{}; // rgb emission * strength, w kullanilmiyor
};
static_assert(sizeof(EmissiveTriangle) == 64u, "emissive triangle GLSL ABI");

// GPU-side counters, zeroed before every dispatch. This is the ONLY thing that
// can answer "did the bounce actually do anything": the material counts below
// describe the CPU table, and hit_fraction only says a ray met geometry.
struct alignas(16) BounceCounters {
    uint32_t hits = 0;        // probe rays that hit a triangle
    uint32_t shaded = 0;      // of those, hits whose material is inside the slice
    uint32_t alphaTested = 0; // shadow/environment candidates run through alpha
    uint32_t alphaPassed = 0; // of those, candidates that actually occluded
    // ★★★★★ NEDEN elendigini soyleyen sayaclar. `hits > 0 && shaded == 0`
    //   dogru ama ISE YARAMAZ bir olcumdu: bes ayri sebep ayni sifiri
    //   uretiyor ve hangisinin tetiklendigini ancak shader'a printf koyarak
    //   ogrenebiliyordun. Bir olcu aleti "bir sey yanlis" demekle yetinirse,
    //   teshis maliyetini sifirlamak yerine sadece yerini degistirir.
    //   Bu bes sayac + `shaded` toplandiginda `hits`i vermeli; vermiyorsa
    //   adlandirilmamis bir cikis yolu daha var demektir.
    uint32_t backFaceShaded = 0;       // arka yuz isabeti; normal cevrilip GOLGELENDI
    uint32_t skippedBounceDisabled = 0;// bounce kapali (params2.y < 0.5)
    uint32_t rejectedUnresolved = 0;   // rfResolveHit false: tablo/indeks/materyal adresi cozulemedi
    uint32_t rejectedUnsupported = 0;  // materyal dilimin disinda (diffuse.w < 0.5)
    uint32_t rejectedDegenerate = 0;   // sifir alanli ucgen
};
struct BounceStatus {
    bool requested = false, ready = false, active = false;
    uint32_t instances = 0, materials = 0, lights = 0;
    uint32_t supportedMaterials = 0;
    uint32_t unsupportedMaterials = 0, unsupportedLights = 0;
    // WHY a material fell outside the slice. A bare count cannot be checked
    // against the scene: "31 unsupported" in a scene with 31 materials reads
    // the same whether every material is genuinely out of scope or the gate is
    // rejecting everything. These name the clause that rejected it, so the
    // number can be compared with what the material panel shows. A material
    // can trip several clauses, so they do NOT sum to unsupportedMaterials.
    uint32_t rejectedTextured = 0;    // a texture slot the slice still cannot read
    uint32_t rejectedTransparent = 0; // opacity < 1 or transmission > 0
    uint32_t rejectedLayered = 0;     // subsurface / translucency / clearcoat
    uint32_t rejectedFlagged = 0;     // water/terrain/volume class flags
    // OR of the flags that tripped the flag clause. A count alone cannot say
    // WHICH bit did it, and most of that word is NOT material features -- see
    // kDisqualifyingFlags in RayFusionBounce.cpp.
    uint32_t rejectedFlagBits = 0;
    // ★★★ Emissive NEE olcusu. `emissiveTriangles` kac ucgenin ISIK olarak
    //   orneklenebilir oldugunu soyler; `emissiveDropped` cap yuzunden DISARIDA
    //   kalani. Ikinci sayi olmadan bir emissive arazi "36 ucgen" gibi gorunur
    //   ve eksik isigin sebebi hicbir yerde yazmaz.
    // ★ `emissiveRejectedTransparent`: pratikte lamba abajurlari transparandir
    //   ve bounce kumesinden eleniyor -- yani en cok beklenen emissive nesne
    //   tam olarak katki VERMEYEN nesnedir. Bu sayi o surprizi gorunur yapar.
    uint32_t emissiveTriangles = 0, emissiveDropped = 0;
    // ★★★ Klozu ADLANDIR, tek bir "dropped" sayisi yetmez. Welded (indeksli)
    //   mesh'in indeks tamponu yalnizca GPU'da yasiyor (RasterMeshBuffer'da
    //   cpuIndices YOK), yani ucgenleri CPU'da COZULEMEZ. Bunu genel bir cap
    //   dusmesi gibi raporlamak, "emissive lamba neden katki vermiyor"
    //   sorusunu yanlis yere yonlendirirdi.
    uint32_t emissiveSkippedIndexed = 0;
    uint32_t emissiveRejectedTransparent = 0;
    float emissiveArea = 0.0f;
    // Measured on the GPU, last dispatch. `shadedHits` is the acceptance
    // number: hits > 0 with shadedHits == 0 means every ray landed on an
    // occluder and the image is identical to the bounce being off.
    uint32_t hits = 0, shadedHits = 0, alphaTested = 0, alphaOccluded = 0;
    // ★★★★ Elenme sebepleri (ayni dispatch, GPU'da sayilir). Bunlarin toplami
    //   + shadedHits == hits olmali. `backFaceShaded` bu toplamin DISINDADIR:
    //   golgelenen isabetlerin bir ALT KUMESIDIR, ayri bir cikis degil.
    //   TASARIM sonucudur ve ic mekanda baskin cikar: tek yuzlu duvarlardan
    //   olusan kapali bir odada probe isinlarinin cogu ARKA yuze carpar.
    uint32_t backFaceShaded = 0, skippedBounceDisabled = 0;
    uint32_t rejectedUnresolved = 0, rejectedUnsupported = 0, rejectedDegenerate = 0;
    uint64_t signature = 0;
    // ★★★★ CPU cost of building the hit/material/light tables, every frame,
    //   whether or not the bounce is enabled. This used to be unmeasured: the
    //   table build re-hashed the whole scene's per-vertex material-ID streams
    //   on every frame, so it scaled with scene size and pinned one thread
    //   while the GPU waited. "CPU 6%" on a 16-thread machine IS one saturated
    //   core; without a number here that reads as "nothing is busy".
    double prepareMs = 0.0;
    // ★★★★★ 9,06 ms/kare OLCULDU (710 instance, 69 malzeme) ve bounce KAPALIYKEN.
    //   Tek bir toplam "pahali" der ama NEYIN pahali oldugunu soylemez, ve
    //   tahminle cache yazmak bayatlamaya yol acar -- ki o 9 ms'den beterdir.
    //   Bu yuzden toplam FAZLARA bolunuyor: hangi fazin kapiya ihtiyaci oldugu
    //   olculerek secilsin.
    // ★★ `uploadSkipped` ayri bir alan: mevcut imza kapisi YALNIZCA yuklemeyi
    //   atlar, onu ureten CPU isini atlayamaz (imza kurulan dizilerin hash'i).
    //   Bu bayrak, kapinin gercekte ne kadarini kurtardigini gorunur yapar.
    double prepareInstancesMs = 0.0;  // sahne instance listesi
    double prepareEmissiveMs = 0.0;   // TUM ucgenlerin emissive taramasi
    double prepareMaterialsMs = 0.0;  // malzeme siniflandirma + isik paketleme
    double prepareUploadMs = 0.0;     // GPU yuklemesi (imza tutarsa 0)
    bool   uploadSkipped = false;
    // Emissive taramasi onbellekten mi geldi. Bu bayrak olmadan 0,00 ms iki
    // ayri seyi anlatabilir: "cache isabet etti" ve "tarama hic calismadi".
    bool   emissiveCached = false;
    // Physical Sky gunesi sicrama isik tablosunda mi, ve tonu gercek
    // transmittance LUT'undan mi geldi. Ikincisi ayri bir alan cunku LUT yokken
    // shader'in sabit yedegi kullanilir ve o, ALCAK GUNESTE gorunur sekilde
    // daha az sicaktir -- yaklasim oldugu SOYLENMELI, sessizce kullanilmamali.
    bool   sunInBounce = false;
    bool   sunTintFromLut = false;
    // ── Hair bounce integration ──────────────────────────────────────────
    // Hair geometry is added to the RayFusion TLAS as opaque AABB occluders.
    // These counters let the UI panel confirm hair participates in bounce.
    uint32_t hairInstances = 0;   // hair AABB BLAS instances in TLAS
    uint32_t hairMaterials = 0;   // hair materials in the bounce table
    std::string reason;
};
static_assert(sizeof(HitInstance) == 48 && sizeof(BounceMaterial) == 128 &&
              sizeof(BounceLight) == 48 && sizeof(BounceCounters) == 48,
              "probe bounce scalar ABI");
}
