#pragma once

// ===========================================================================
// Raster viewport GPU culling + LOD split.
//
// NEDEN VAR: production instancing (global instance buffer) devreye girdiginde
// CPU frustum culling'i ve scatter proxy ayrimi TAMAMEN devre disi kaliyordu.
// `uploadVisibleRasterInstances` global yolda proxy/butce mantigina varmadan
// erken donuyor, ve `writeRasterInstanceTransformsToGlobal` mesh.instanceCount
// = TUM instance sayisi yaziyordu. Yani kamera nereye bakarsa baksin her
// instance her kare gonderiliyordu; gizli olanlar bile sifir matrisle dejenere
// ucgen olarak vertex shader'dan geciyordu.
//
// Bu modul o iki karari GPU'ya tasir ve CPU'daki kare-basi taramayi
// (worldBBox merkezleri, mesafe kareleri, nth_element + sort) tamamen kaldirir.
//
// VERI AKISI (kare basina):
//   1. counters vkCmdFillBuffer ile sifirlanir
//   2. pass 0: her mesh icin bir dispatch -- frustum testi + LOD ayrimi,
//      hayatta kalanlarin matrisi SIKISTIRILMIS cikti bolgesine kopyalanir
//   3. bariyer
//   4. pass 1: tek dispatch -- indirect draw komutlari yazilir, LOD mesafe
//      esigi guncellenir
//   5. bariyer (INDIRECT_COMMAND_READ + VERTEX_ATTRIBUTE_READ)
//   6. cizim: vkCmdDraw[Indexed]Indirect, vertex binding'i SIKISTIRILMIS
//      buffer'a bakar
//
// ★ Raster vertex shader'larinda HICBIR degisiklik gerekmez: instance matrisi
//   zaten per-instance vertex attribute olarak geliyor, biz yalnizca hangi
//   buffer'in baglandigini ve kac instance cizildigini degistiriyoruz.
//
// ★★ ANLAM DEGISIKLIGI: ucgen butcesi artik "en yakin N instance" siralamasiyla
//   degil, kareler arasi yakinsayan bir MESAFE ESIGI ile uygulanir. Ayrintili
//   gerekce `shaders/raster_cull.comp` basindaki yorumda. Bu yuzden cagiran
//   taraftaki alanin adi da `...TriangleTarget`tir, `...TriangleBudget` degil:
//   sert tavan degil, hedeftir.
// ===========================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace VulkanRT { class VulkanDevice; }

namespace Backend {

class RasterGpuCull final {
public:
    // `raster_cull.comp` icindeki MeshParam ile BIREBIR ayni duzen (std430,
    // hepsi uint => dizi adimi 48 bayt). Buradaki bir alani degistirirsen
    // shader'daki karsiligini da degistir; static_assert yalnizca boyutu
    // korur, SIRAYI koruyamaz.
    struct MeshBinding {
        uint32_t firstInstance        = 0;  // kaynak matris/bounds tabani
        uint32_t instanceCount        = 0;  // bu mesh'in instance sayisi
        uint32_t outBase              = 0;  // sikistirilmis ciktidaki taban
        uint32_t outCapacity          = 0;  // == instanceCount
        uint32_t drawSlot             = kNoSlot;
        uint32_t proxyDrawSlot        = kNoSlot;
        uint32_t flags                = 0;  // FLAG_LOD_SPLIT | FLAG_INDEXED
        uint32_t trianglesPerInstance = 0;
        uint32_t elementCount         = 0;  // indexli ise indexCount, degilse vertexCount
        uint32_t proxyElementCount    = 0;
        uint32_t proxyFlags           = 0;  // FLAG_INDEXED
        // Proxy'nin KENDI cikti bolgesinin tabani. Tam ve proxy ayri
        // bolgelere, ikisi de onden doldurulur; iki taban da CPU'da bilinir.
        uint32_t proxyOutBase         = 0;
    };

    static constexpr uint32_t kNoSlot       = 0xFFFFFFFFu;
    // ★★★ Ikisi AYRI olmak zorunda: butcenin uygulanmasi, proxy mesh'in var
    //   olmasina bagli DEGILDIR. Proxy yoksa butce disi instance CIZILMEZ --
    //   eski CPU davranisinin aynisi. Ikisini birlestirmek, proxy'si olmayan
    //   (ornegin flat SoA kaynakli) her scatter grubunda butceyi tamamen
    //   etkisiz birakir.
    static constexpr uint32_t kFlagLodSplit = 1u;  // butce/mesafe esigi uygulanir
    static constexpr uint32_t kFlagIndexed  = 2u;
    static constexpr uint32_t kFlagHasProxy = 4u;  // demote edilenler cizilir
    // cmds[] tek bir 5-uint kayit dizisidir; hem VkDrawIndirectCommand hem
    // VkDrawIndexedIndirectCommand bu kaydin BASINDAN itibaren okunur.
    static constexpr uint32_t kCommandStride = 20;

    // Bir mesh'in gecen karede olculen sonucu. state buffer'i HOST_VISIBLE
    // oldugu icin bu okuma bedavadir -- ama BIR KARE GERIDIR.
    struct MeshResult {
        float    lodDistanceSq = 0.0f;
        uint32_t fullInstances = 0;
        uint32_t proxyInstances = 0;
    };

    RasterGpuCull();
    ~RasterGpuCull();
    RasterGpuCull(const RasterGpuCull&) = delete;
    RasterGpuCull& operator=(const RasterGpuCull&) = delete;

    // Pipeline + tum buffer'lari hazirlar. Kapasite yetiyorsa no-op.
    // false donerse cagiran ESKI CPU yoluna dusmeli ve bunu RAPORLAMALI --
    // sessizce culling'siz cizmek bu modulun duzeltmeye calistigi arizanin ta
    // kendisidir.
    // instanceCapacity: KAYNAK instance sayisi (bounds + src matris indeksleme)
    // outCapacity:      SIKISTIRILMIS cikti yuvasi sayisi. Bu ikisi ayni DEGIL:
    //                   LOD ayrimi olan her mesh icin bir de proxy bolgesi
    //                   ayrilir, yani cikti kaynaktan buyuktur.
    bool ensure(VulkanRT::VulkanDevice& device,
                const std::string& shaderDir,
                uint32_t instanceCapacity,
                uint32_t outCapacity,
                uint32_t meshCount,
                uint32_t drawSlotCount);

    bool isReady() const;

    // --- kare basina girdiler (hepsi mapped yazma, Vulkan cagrisi yok) -----

    // planes: 6 x (nx, ny, nz, d), CPU'daki m_frustumPlanes ile ayni
    // normalize edilmis konvansiyon. Near plane (indeks 4) shader'da BILEREK
    // atlanir; CPU tarafi da atliyordu.
    void setGlobals(const float planes[24],
                    const float cameraPos[3],
                    uint32_t scatterTriangleTarget,
                    float lodMinDistanceSq,
                    float lodMaxDistanceSq,
                    uint32_t meshCount,
                    // Gecis bandinin yari genisligi, esigin carpani olarak.
                    // 0 => sert anahtar. Bkz. raster_cull.comp classify().
                    float lodBandWidth);

    void setMeshBindings(const std::vector<MeshBinding>& bindings);

    // Instance basina dunya uzayi sinirlayici kure: (cx, cy, cz, radius).
    // ★ radius < 0 => GIZLI instance; shader onu hic gondermez. Eskiden gizli
    //   instance sifir matrisle dejenere ucgen olarak yine cizilirdi.
    void setBounds(uint32_t firstInstance, const float* centerRadius4, uint32_t count);

    // --- kayit -------------------------------------------------------------

    // srcInstanceBuffer: global instance matris buffer'i (VkBuffer). Her kare
    // ayni olmayabilir (buyume ile yeniden olusur), degistiginde descriptor
    // kendiliginden guncellenir.
    void record(void* commandBuffer, void* srcInstanceBuffer, uint32_t meshCount);

    // --- cizim tarafinin ihtiyaci -----------------------------------------

    void* compactedInstanceBuffer() const;   // VkBuffer
    void* commandBuffer() const;             // VkBuffer (indirect)
    static uint64_t commandOffset(uint32_t drawSlot) {
        return static_cast<uint64_t>(drawSlot) * kCommandStride;
    }

    // Gecen karenin sonucu (telemetri icin). Bir kare geridir; yoklugu
    // sifir DEGILDIR -- false donerse hic olcum yok demektir.
    bool readMeshResult(uint32_t meshIndex, MeshResult& out) const;

    void destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Backend
