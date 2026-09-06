#pragma once

// Raster/Realtime viewport frame telemetry — VALUES only.
//
// ★★★ CLAUDE.md kural 1: bir yetenek yalnız panelden okunabiliyorsa test
// edilemez. Faz 0.5a asenkron sunum yolu tam olarak bu sınıfa girer: köprü
// doğru çalıştığında görüntü AYNI görünür, yalnız gecikme ve stall değişir.
// Yani "çalışıyor mu" sorusunun tek cevabı ölçümdür, bakmak değildir.
//
// ★ Bir alanın 0 olması "ölçtüm ve sıfırdı" demek DEĞİLDİR; `available` false
// iken bütün sayılar yokluk anlamına gelir. Bu ayrım olmadan bir ajan
// "senkron stall kalmadı" ile "raster viewport hiç koşmadı"yı ayırt edemez.

#include <cstdint>

namespace Backend {

struct RasterFrameTelemetry {
    // false = bu backend'de raster frame ring hiç kurulmadı (RT modu, Vulkan
    // yok, ya da henüz tek kare çizilmedi). Aşağıdaki sayılar YOKTUR, sıfır
    // değildir.
    bool available = false;

    // false = sürücü kalıcı frame kaynaklarını reddetti ve deterministik eski
    // senkron yol kullanılıyor. Kural 6: desteklenmeyen bir yol sessizce
    // yanlış görünmez, raporlanır.
    bool async_present = false;

    // Capture/probe açıkken sunum bilerek senkronlanır: bir ajan probe ettiği
    // karenin AZ ÖNCE çizilen kare olduğundan emin olabilsin diye. Gecikme
    // pahasına belirlilik.
    bool synchronous_present = false;

    std::uint32_t slot_count = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    // ── Son kare, milisaniye ────────────────────────────────────────────────
    double frame_ms = 0.0;            // renderInteractiveViewportImpl tamamı
    double cpu_record_ms = 0.0;       // command buffer kaydı
    double slot_wait_ms = 0.0;        // iki slot da meşgulken sınırlı bekleme
    double submit_ms = 0.0;           // vkQueueSubmit (beklemesiz)
    double image_readback_ms = 0.0;   // YALNIZ eski senkron yolda > 0
    double host_read_ms = 0.0;        // invalidate + mapped memcpy
    double present_ms = 0.0;          // SDL surface kopyası + UpdateTexture

    // ── Ring kurulduğundan beri sayaçlar ────────────────────────────────────
    std::uint64_t frames_submitted = 0;
    std::uint64_t frames_consumed = 0;
    // ★ En önemli sayaç. Bu kare raster işi YAPTI ama sunulacak yeni tamamlanmış
    // slot yoktu, yani ekrana eski pikseller gitti. Küçük ve sabit olması
    // beklenir; sürekli artması boru hattının kaymaya başladığını söyler.
    std::uint64_t stale_presents = 0;
    // Bir slotu yeniden kullanmadan önce beklemek zorunda kalınan kare sayısı.
    // Sıfıra yakın = GPU önde; kare sayısına yakın = GPU tıkalı, köprü artık
    // paralellik üretmiyor.
    std::uint64_t slot_waits = 0;
    // Bilerek bloke edilen kareler: mod/resize sonrası ilk tohum karesi ve
    // capture açıkken her kare.
    std::uint64_t blocking_seeds = 0;
    // Kaynak mutasyonu (buffer yeniden kurma, descriptor yazımı) için uçuştaki
    // kareleri boşaltma sayısı. Bu, köprünün ödediği gerçek bedeldir.
    std::uint64_t resource_drains = 0;
    // Sunulan karenin kaç kare geride olduğu (0 = bu kare, 1 = bir önceki).
    std::uint32_t present_latency_frames = 0;

    // ── Geometri gönderimi ──────────────────────────────────────────────────
    // ★★★ Bu blok, sunum telemetrisinin cevaplayamadığı soruyu ölçer:
    // "raster mı yavaş, yoksa CPU instance taraması mı?" Bu ayrımı ölçen bir
    // alet yokken hangi optimizasyonun işe yaradığı bir TAHMİNDİ.
    //
    // ★★ gpu_culling false + global_instance_buffer true = sahne CULLING'SİZ ve
    // PROXY'SİZ çiziliyor: her mesh'in bütün instance'ları, kamera nereye
    // bakarsa baksın. Bu kombinasyon bir ARIZADIR ve ekranda doğru görünür —
    // yalnız yavaştır. Tam olarak bu yüzden ölçülüyor.
    bool global_instance_buffer = false;
    bool gpu_culling = false;

    // Sahnedeki TOPLAM instance (culling öncesi) ve mesh sayısı.
    std::uint32_t total_instances = 0;
    std::uint32_t cull_mesh_count = 0;
    // Bu karede kaydedilen çizim çağrısı sayısı (proxy çizimleri dahil).
    std::uint32_t draw_calls = 0;

    // ★ gpu_culling açıkken bu sayılar GPU'dan okunur ve BİR KARE GERİDİR:
    // compute kareyi yazar, CPU bir sonraki karede okur. Kamera sabitken
    // fark yoktur; hızlı harekette bir kare gecikir.
    std::uint64_t visible_triangles = 0;
    std::uint64_t full_triangles = 0;
    std::uint64_t proxy_triangles = 0;
    std::uint32_t full_instances = 0;
    std::uint32_t proxy_instances = 0;

    // LOD hedefi. ★ Adı bilerek "target": GPU culling'de bu sert bir tavan
    // değil, kareler arası yakınsayan bir mesafe eşiğinin hedefidir; hızlı
    // kamera hareketinde bir-iki kare aşılabilir.
    std::uint64_t scatter_triangle_target = 0;
};

} // namespace Backend
