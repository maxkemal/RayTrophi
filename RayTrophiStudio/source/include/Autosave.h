#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// AUTOSAVE — aralikli ve ZORLAMALI oturum kaydi
//
// ★★★★★ 2026-09-16'da bulundu: autosave AYARI, GERI YUKLEME yolu ve Hub'daki
//   "Recover the last autosaved scene session" dugmesi VARDI, ama dosyayi
//   YAZAN HIC KIMSE YOKTU. `auto_save_enabled` varsayilan true, araligi 300 sn,
//   `StartupMode::RestoreAutosave` bir mod olarak seciliyor -- ve kurtarilacak
//   dosya hicbir zaman olusmuyordu. Yani panel, olmayan bir guvenligi
//   raporluyordu; bu deponun en pahali hata sinifi.
//
// ★★★★ `ProjectManager::saveProject(path, ...)` YAN ETKILIDIR: `current_file_path`
//   ve gerekirse `project_name` yazar, ve dosyayi "son projeler" listesine
//   ekler. Naif bir autosave bu yuzden kullanicinin PROJESINI CALAR -- bir
//   sonraki Ctrl+S gercek projeye degil autosave.rtp'ye gider ve kullanici
//   bunu ancak projesini kaybettiginde fark eder. Bu birim kimligi kaydin
//   ETRAFINDA geri yukler; tek dogru yer burasi, cagiranlar degil.
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <string>

struct SceneData;
struct RenderSettings;
class Renderer;

namespace raytrophi::autosave {

struct Status {
    bool enabled = false;
    int interval_sec = 0;
    std::string path;
    bool file_exists = false;
    uint64_t file_bytes = 0;
    double seconds_since_last_write = -1.0;  // <0 = bu oturumda hic yazilmadi
    double seconds_until_next = -1.0;
    double last_write_ms = 0.0;
    std::string last_reason;                 // "interval" | "vulkan_device_lost" | "manual" | ...
    bool last_ok = false;
    std::string last_error;
    uint64_t write_count = 0;
    // ★★ `is_modified` yanlissa autosave SESSIZCE hic yazmaz. O yuzden atlama
    //   sayisi ayri raporlanir: write_count 0 iken bu sayi buyuyorsa arizali
    //   olan autosave degil, degisiklik bayragidir.
    uint64_t skipped_unmodified = 0;
    bool scene_is_modified = false;
};

// Araliga bakar, gerekiyorsa yazar. Her karede cagrilabilir; ucuz.
// `busy` true iken hicbir sey yapmaz (sahne yukleniyor vb.).
void tick(SceneData& scene, RenderSettings& settings, Renderer& renderer, bool busy);

// Aralia ve `is_modified`'a BAKMADAN yazar. Cihaz kaybi yolu bunu kullanir.
bool writeNow(SceneData& scene, RenderSettings& settings, Renderer& renderer,
              const std::string& reason);

Status status();

}  // namespace raytrophi::autosave
