# Realtime hacim: tablo viewport backend'ine HİÇ yayımlanmıyordu

> **Durum:** AKTİF — kök neden bulundu ve düzeltildi, DERLENMEDİ/doğrulanmadı (2026-09-04)

## Belirti

Canlı gaz domain'i **Rendered (Vulkan RT)** modunda doğru çıkıyor,
**realtime / Material Preview** modunda **hiç** çıkmıyor. Ekrana bakarak
ayırt edilebilir hiçbir ipucu yok: hata mesajı yok, çökme yok, kısmi sonuç yok.
Aynı gate SurfaceSDF (sıvı izoyüzeyi) pass'ini de kapatıyor.

## Kök neden

`SceneUI::syncVDBVolumesToGPU` (`src/UI/scene_ui_vdb.cpp`) hacim paketini
**yalnızca `g_backend`'e** yayımlıyordu:

```cpp
Backend::IBackend* getVdbRenderBackend(UIContext& ctx) {
    if (g_backend) return g_backend.get();          // ← tek hedef
    ...
    return nullptr;                                  // g_viewport_backend HİÇ dönmüyor
}
```

Ama raster viewport **ayrı bir `VulkanBackendAdapter`** ve **ayrı bir
`VkDevice`**: `g_viewport_backend`, Vulkan varsa **koşulsuz** kuruluyor
(`Main.cpp::initializeViewportBackendIfAvailable`, açılışta ~2650 ve backend
değişiminde ~3250), ve `getRasterViewportBackend()` her zaman **önce onu**
döndürüyor.

Hacim SSBO'su cihaz başına (`VulkanDevice::m_volumeBuffer` / `m_volumeCount`).
Sonuç:

| viewport adapter'ında | değer |
|---|---|
| `m_volumeCount` | **0** (her sahnede, her zaman) |
| `m_volumeBuffer.buffer` | `VK_NULL_HANDLE` |
| `updateMaterialPreviewVolumeBinding()` | ilk satırda `return` → binding 20 **hiç bağlanmadı** |
| `recordMaterialPreviewVolumePass` | `SKIPPED` |
| `recordMaterialPreviewSdfSurfacePass` | `return` |
| `useMaterialPreview`'ın `m_volumeCount > 0u` terimi | `false` |

Malzeme, ışık, dünya, kamera, görünürlük ve raster mesh'lerin hepsi viewport
backend'ine **tek tek aynalanmıştı** (`Main.cpp` ~860 / ~2689 / ~3252).
Hacimler, kimsenin aynalamadığı **tek paketti** — ve eksikliğin yokluktan başka
hiçbir belirtisi yoktu.

## ★★★ Tripwire zaten cevabı vermişti, YANLIŞ OKUNDU

Bir önceki turda `recordMaterialPreviewVolumePass` içine beş kapıyı adıyla yazan
tripwire konmuştu ve çıktısı alınmıştı:

```
[MPVolume] pass gates: mode=1 pipeline=1 descSet=1 volumeCount=0 bound=0 -> SKIPPED
```

Bu "sahnede gaz yok, doğru okuma" diye yorumlandı. **Değildi.** O adapter'da
`volumeCount` **her sahnede** 0 okur. `mode=1 pipeline=1 descSet=1` zaten
material-preview yolunun tamamen canlı olduğunu söylüyordu; ayakta olan bir
yolun sıfır hacim görmesi tam olarak bu hatanın imzasıydı.

★ **Ders: bir tripwire'ın "beklenen" değeri, ancak o değeri üretebilecek TÜM
durumları saydıktan sonra beklenendir.** `volumeCount=0`'ın iki üreticisi vardı
(sahnede hacim yok / bu cihaza hacim hiç verilmedi) ve enstrüman ikisini
ayırmıyordu — [[feedback_measurement_is_not_visibility]] ile aynı aile.

## İkinci, gerçek ama ikincil hata: SINIRI GEÇEN İŞARETÇİ

Önceki turun bir numaralı şüphelisi de gerçekti, sadece **maskelenmişti**: canlı
gaz için `vdb_grid_address = dense_density_address`, yani simülasyon compute
context'inin ham `VkDeviceAddress`'i. O context **en son yaratılan `VkDevice`'a**
bağlanıyor (`VulkanDevice::createLogicalDevice` kendini koşulsuz
`g_vulkan_sim_compute_ctx`'e yazıyor, `VulkanBackend.cpp:1447`).

`VolumetricRenderer::syncVolumetricData` bu adresleri OptiX'e vermemek için
zaten korunuyordu, ama testi "tüketici Vulkan mı?" idi — **iki Vulkan cihazı
için de doğru, biri için yanlış.** Yukarıdaki tel takılır takılmaz bu hata
görünür hale gelecekti: adres çözülmez, her yoğunluk örneği 0 döner, hacim yine
boş çıkar — "burada duman yok"tan ayırt edilemez.

Bu yüzden ikisi **aynı partide** düzeltildi.

## Değişiklikler

| Dosya | Ne |
|---|---|
| `src/UI/scene_ui_vdb.cpp` | Hacim paketi `g_backend` **ve** `g_viewport_backend`'e ayrı ayrı yayımlanıyor (ayrı çağrı — paket hedefe göre adres çözüyor, ortak tampon olamaz) |
| `src/Render/VolumetricRenderer.cpp` | `liveDenseAddressesUsable`: canlı yoğun gaz adresleri **yalnızca** sim compute cihazı o backend'in kendi cihazıysa yayımlanır. Değilse `live_data = nullptr` → adapter başına NanoVDB yüklemesine düşer (`m_vdbBuffers`), yani baked VDB ve SurfaceSDF'in zaten doğru çalıştığı yol. Ayrıca değişimde bir `[VolumePublish]` satırı |
| `src/Viewport/MaterialPreviewSdfSurface.cpp` | Gaz pass'indekinin aynısı beş kapılı tripwire: `[MPSdf] pass gates: ... -> RECORDED \| SKIPPED` |
| `include/Api/RtApi.h`, `src/Api/RtApi.cpp` | `VolumeTablesInfo` / `volumeTables()` |
| `src/Api/RtIpc.cpp`, `src/Api/RtPython.cpp`, `scripts/ipc_descriptor_overlay.json` | `render.volume_tables` |
| `scripts/test/rt_test_volume_tables.py` (+ `x64/Release/...`) | Bölünmeyi script'ten yakalayan test |

`RtIpcSecurity.cpp` değişmedi: `render.*` prefiksi zaten `Render` yetkisine
düşüyor. `audit_ipc_capabilities.py` yeşil.

## Kalan iş: realtime'da GAZ hâlâ çizilmiyor (ölçüldü 2026-09-04)

★ **Önce "donmuş görünür" diye yazmıştım — ÖLÇÜM bunu yalanladı, gaz TAMAMEN
YOK.** Düzeltmeden sonra çalışan uygulamada, aynı kameradan, aynı karede:

| mod | sıvı SurfaceSDF | canlı gaz |
|---|---|---|
| Rendered (Vulkan RT) | ✔ | ✔ (alev + duman) |
| Material Preview | ✔ (bu partide düzeldi) | **hiç yok** |

`render.volume_tables` ikisinde de `instance_count=2`, yani gaz viewport
tablosunda **var**; çizilmeyen şey içeriği.

**Neden:** gaz canlı bir GPU yoğun grid'i. Viewport backend'i sim compute
cihazına sahip olmadığı için (ölçüldü: `sim_device_is_this_backends=false`)
yabancı adres doğru şekilde bastırılıyor ve NanoVDB yoluna düşüyor — **ama o
yolda grid yok**: `scene_data.h`'de host NanoVDB yeniden üretimi
`use_live_dense_gpu` doğruyken kapatılıyor (pahalı OpenVDB→NanoVDB turu), yani
GPU gaz domain'inin host grid'i hiç üretilmiyor. Sonuç `vdb_grid_address = 0` →
`volume_type` prosedürel gürültüye geriliyor → ekranda hiçbir şey.

Yani "donmuş" ara durumu **yok**: ya canlı adres, ya hiçbir şey.

**YAPILDI (2026-09-04, ikinci parti — DERLENMEDİ).** Yoğun grid o adapter'ın
kendi belleğine **düz `float` SSBO** olarak kopyalanıyor. NanoVDB turu yok:
tüketici zaten düz bir dense grid örnekliyor, dolayısıyla iş bir memcpy
şeklinde. Shader yolu (`sampleDense`, `volume_type=4`, `source_type=5`) aynen
kaldı — değişen tek şey adresin kimin belleğini gösterdiği.

Zincir, uçtan uca:

| katman | ne |
|---|---|
| `ParticleSimulationSystem::downloadGasDenseFields` | GPU-resident gaz grid'ini host'a okur (`gasGpuFieldView` ile **birebir aynı** kabul testi — ikisi aynı grid'i tarif etmeli) |
| `scene_data.h` | `setLiveDenseGpuFields` hemen ardından, `g_dense_gas_host_mirror_needed` ve `dense_version` kapılı olarak aynayı doldurur |
| `VDBVolumeManager` | `dense_density_mirror` / `dense_temperature_mirror` + `dense_mirror_version` |
| `GpuVDBVolume` | `dense_density_host` / `dense_temperature_host` / `dense_host_version` (host taşıma alanları; ★ varsayılan ilklendirici YOK) |
| `VulkanBackend_Volumes.cpp` | sahibi değilse aynayı `m_denseGasBuffers`'a yükler ve **kendi** adresini yayımlar |
| `render.volume_tables` | `dense_gas_mirror_buffers` — kopyanın gerçekten olup olmadığı |

★ Kasıtlı eksikler: majorant ve emissive listesi kopyalanmıyor. İkisi de diğer
cihazın tamponu, ve yoklukları **güvenli**: eksik majorant "boş" diye
okunamaz, shader her adımı yürür. Yavaş, doğru.

★ `g_dense_gas_host_mirror_needed` bir bayrak, çünkü readback gerçek iş: tek
backend'li bir oturum hiç kullanmayacağı bir kopyanın bedelini ödememeli.

## Doğrulama

`docs/dev/NEXT_BUILD_CHECKS.md` §1 ve §2.

İlgili: [[bugfix_sdf_volume_evicted_by_unrelated_tlas_rebuild]] ·
[[feedback_ipc_values_only_no_pointers]] ·
`docs/dev/` — realtime hacim açık notu bu notla kapanıyor.
