# Proje açılışında raster viewport'ta VK_ERROR_DEVICE_LOST

> **Durum:** AKTİF — AÇIK. Kök neden BULUNMADI. Üç statik hipotez kuruldu, **üçü de kullanıcı gözlemiyle öldü**. Çökme artık **kural yüzünden ulaşılamaz**, o yüzden ölçüm için kural KAPATILABİLİR yapıldı. Sıradaki adım tahmin değil, `Probe-DeviceLostOnProjectOpen.ps1`. (2026-09-07)
> **2026-09-09 EKİ:** dördüncü bir hipotez var ve bu sefer **ölçülmüş bir kardeş arızası** ile geliyor — yüklemeyi BAŞLATAN kare sökmeyle yarışıyor (aşağıda "YENİ ADAY"). Sınaması kalkanın kapatılmasını gerektirir.

## Belirti

```
[ERROR] [Viewport] Vulkan device lost while submitting a raster frame;
        no retry will be attempted on the lost queue.
```

Uygulama ayakta kalıyor — bu maske değil, tasarım: `Main.cpp:2777`
`g_viewport_backend->shutdown()` + `reset()` yapıp `use_vulkan=false` ile CPU'ya
düşüyor.

★★ İki device-lost yolu var, ayrım önemli:

| yer | mesaj | anlamı |
|---|---|---|
| `VulkanViewportBackend.cpp:3572` | "while **acquiring**" | kuyruk ZATEN ölü — arızanın SONUCU |
| `VulkanViewportBackend.cpp:4569` | "while **submitting**" | bu karede kaydedilen iş cihazı öldürdü — **bildirilen bu** |

## ★★★★★ KULLANICININ A/B TABLOSU — bu dosyanın en değerli kısmı

2026-09-07, iki turda toplandı:

| # | senaryo | sonuç |
|---|---|---|
| 1 | Boş/yeni uygulama + hub panel → **ağır sahne** aç. Bu yolda **realtime (Material)** modda açılıyor. | **ÇALIŞIYOR** |
| 2 | Ağır sahne AKTİF ve **realtime moddayken** → hub'dan **yeni ağır proje** aç | **ÇÖKÜYOR** |
| 3 | Ağır sahne aktif → **Solid'e geç** → hub'dan ağır sahne aç | **ÇALIŞIYOR** |
| 4 | Boş sahne → *Open Project* ile ağır proje (Solid açılır) → sonra realtime'a geç | **ÇALIŞIYOR** |
| — | Logda VRAM uyarısı | **YOK** |

### Bu tablonun tek başına elediği şeyler

- **Salt maliyet / TDR ÖLDÜ.** Senaryo 1: aynı ağır sahne, aynı Material karesi,
  sorunsuz çiziliyor. Pahalı olsaydı orada da çökerdi.
- **Proje içeriği (NaN transform, bozuk instance) ÖLDÜ.** Aynı dosyalar.
- **"Yükleme sırasında Material bağlı olması" TEK BAŞINA YETMİYOR.** Senaryo 1
  de Material'da yükleniyor ve çökmüyor. Önceki turda kurduğum bütün teori
  buna dayanıyordu — **yanlıştı**.
- **VRAM tahliyesi (`checkAndTrimVRAMThreshold`) ÖLDÜ.** O yol log basar
  (`"Evicted N textures from inactive backend"` / `"VRAM critical"`);
  kullanıcı logda VRAM uyarısı olmadığını doğruladı.

### Geriye kalan tek değişken

Senaryo 1 ile 2 arasındaki fark **yüklenen şey değil, önceden yüklenmiş
olan şey**: senaryo 2'de ortada **sökülecek dolu bir GPU durumu var** ve
sökme **Material aktifken** yapılıyor. Senaryo 3 bunu doğruluyor: aynı sökme,
Solid'de yapılırsa zararsız.

Kullanıcının ifadesi, ve muhtemelen doğru okuma:

> *"sorun realtime modunda iken yükleme, yani bir işlem bitmeden belleği
> geçersizleştiriliyor gibi"*

## Statik denetimde ELENENLER

1. **`openProject` viewport reset'ini atlıyor** — HAYIR. `openProject`
   (~`ProjectManager.cpp:1930`) `newProject` çağırıyor; oradaki
   `resetForProjectReload()` bloğu (1132-1153) `defer_backend_reset`'e bağlı
   değil. (Bu, `memory/project_viewport_project_reload.md`'deki eski uyarıyı
   eskitir.)
2. **`buildRasterGeometry` uçuştaki kareye karşı korumasız** — HAYIR.
   `destroyAllRasterMeshes()` öncesi hem `drainInteractiveViewportInFlight()`
   hem `m_device->waitIdle()` var (~5453).
3. **Toplu doku yüklemesi sırasında kare basılıyor** — HAYIR.
   `beginBatchedTextureUpload`/`endBatchedTextureUpload` arasındaki döngü
   (`VulkanBackend.cpp:12258-12350`) UI pompalamıyor.
4. **Yükleme sırasında kare basılıyor (progress callback)** — HAYIR.
   `scene_ui.cpp:8664`'teki callback yalnızca `scene_loading_progress` ve
   aşama etiketini yazıyor; Template Hub yolu
   (`TemplateHubUI.cpp:160/176`) callback'i hiç vermiyor. Yükleme boyunca
   kare gönderilmiyor.
5. **Kare halkası teardown'da asılı kalıyor** — HAYIR.
   `VulkanViewportBackend::destroyInteractiveViewportResourcesImpl` (~701)
   `m_rasterFrameRing->reset()` çağırıyor.
6. **`resetForProjectReload` sıralamayı bilmiyor** — HAYIR, tam tersi. Kuralı
   yorumunda birebir yazıyor ve uyguluyor (`VulkanBackend.cpp:15400`):
   `waitIdle` → `destroyInteractiveViewportResourcesImpl(false)` →
   `m_interactiveViewport = {}` → `rebuildAccelerationStructure()`.

## Hâlâ bakılmamış yüzey (sıradaki oturum için, ama ÖNCE validation)

### ★★★★★ YENİ ADAY (2026-09-09): YÜKLEMEYİ BAŞLATAN KARENİN KENDİSİ

Yukarıdaki 4. madde "yükleme sırasında kare basılıyor" hipotezini eledi ve
haklıydı: progress callback kare basmıyor. Ama elenen şey **yükleme boyunca**
basılan karelerdi — **yüklemeyi başlatan kare** hiç bakılmadı.

`performOpenProject` yükleyici ipliğini `ui.draw`'un **içinde** doğuruyor
(Template Hub `drawMainMenuBar` içinde çizilir, `scene_ui_menu.hpp:1304`), ve
`Main.cpp`'deki yükleme kapısı döngünün **başında** duruyor (~3674). Yani o
turun geri kalanı — `drawPanels`, overlay'ler, **ve render + submit + present
bloğunun tamamı** — yükleyici ipliği `newProject` → `resetForProjectReload`
ile viewport kaynaklarını yıkarken çalışmaya devam ediyor.

Bu, tablodaki tek kalan değişkenle örtüşür: fark "önceden yüklenmiş olan şey"
değil, **onun yüzünden o son karenin ne kadar UZUN sürdüğü**. Ağır sahnede o
kare 26–467 ms (bkz. `project_main_loop_cost_is_unmeasured`), yani sökmeyle
çakışma penceresi o kadar açık kalıyor. Senaryo 1'de (boş uygulama) sökülecek
bir şey yok; senaryo 3'te (Solid) o kare ucuz ve raster iş kaydetmiyor.

Aynı pencere, `DrawVolumePerformancePanel`'de **ölçülmüş bir erişim ihlali**
üretti (kullanıcı raporu, 2026-09-09):
[BUG_LOADER_THREAD_RACES_ITS_OWN_FRAME.md](BUG_LOADER_THREAD_RACES_ITS_OWN_FRAME.md).
O çökme bu notun aksine **teoride kalmadı, gerçekleşti** — yani pencerenin
varlığı artık kanıtlı; açık olan yalnızca device-lost'un da ondan geçip
geçmediği.

★ Düzeltme (iki kapı: `ui.draw` içinde menü çiziminden sonra, `Main.cpp`'de
`ui.draw`'dan sonra) yazıldı. **Sınama `NEXT_BUILD_CHECKS.md` madde 4'tür ve
kalkanın KAPATILMASINI gerektirir** — bu notun kendi kuralı: kalkan açıkken
senaryo 2 hiç çalıştırılmıyor, yani suskunluk yine hiçbir şey kanıtlamaz.


- `purgeUploadedTextureCacheLocked` (`VulkanBackend.cpp:7811`) yalnızca
  **RT** descriptor'larını temizliyor (`clearPendingRTTextureDescriptors`);
  material preview descSet'ine hiç dokunmuyor. Doğru sıralama bir fonksiyon
  aşağıda `releaseInactiveViewportTextureCache` (7838) içinde var ve gerekçesi
  yazılı. Purge'ün **altı** çağıranı var; ikisi (`uploadMaterials` 12065/12073)
  ve biri (terrain 13146) descSet teardown'u yapmıyor. Senaryo 2'de bunlardan
  birinin ateşleyip ateşlemediği **ölçülmedi**.
  ★ Slot tükenmesi dalı log basar: `"Texture cache near descriptor capacity"`.
  Çökmeden önce o satır var mı — bakılsın, ücretsiz.
- `updateMaterialPreviewTextureDescriptor` (`VulkanBackend.h:1947`) binding 1'e
  **drenajsız** yazıyor; bu backend'de descriptor yazan diğer 23 yerin hepsi
  `drainInteractiveViewportInFlight()` ile başlıyor.
  ★★ Ve `VulkanViewportBackend.cpp` ~1564'teki yorum boşluğu **lisanslıyor**:
  *"Updating the set between frames (not during command-buffer recording) is
  always valid without those flags."* **Kural bu değil** — kısıt *recording*
  değil **pending execution** üzerinedir. Bu yorum, doğru olsun ya da olmasın
  bu arızanın sebebi, düzeltilmeli: şu haliyle bir sonraki kişiyi de aynı
  yanlış güvene götürür.

## ★★★★★ KURAL ÇÖKMEYİ SAKLADI — ve tam bu yüzden KAPATILABİLİR

Aynı gün konulan kural (`enterSolidViewportForSceneLoad`, bkz.
`memory/feedback_scene_load_may_not_force_heavy_viewport_mode.md`) senaryo 2'yi
artık **hiç kullandırmıyor**: hub'dan açılan her proje önce Solid'e düşüyor.
Çökme gitti. Kök neden **gitmedi**.

★★★★ Ve bu, ölçüm açısından bir felakete çok yakındı. Aşağıdaki tripwire tam
olarak senaryo 2'de tetiklenmek üzere kuruldu; kural o senaryoyu kapattığı için
tripwire **sonsuza kadar susacaktı** — ve suskunluğu "o sınıf elendi" diye
okunacaktı. CLAUDE.md'nin kendi satırı:

> **Tripwire'ın susması yokluğu kanıtlamaz. Enstrümanın anahtarı, ölçtüğü şeyle
> çakışmamalı.**

Burada anahtar ile ölçülen şey **birebir aynı anahtardı**. Kullanıcı bunu
düzeltmenin hemen ardından gördü: *"solid mod açılışı bu sorunu oluşturmadığı
için derleme ile bunu artık ölçemeyiz."*

Çözüm: kalkan **kapatılabilir**, ve yalnızca IPC'den.

```powershell
Invoke-RtIpc viewport.set_scene_load_guard @{ enabled = $false }   # ariza penceresi ACIK
Invoke-RtIpc project.open @{ path = '<agir proje>' }               # senaryo 2
@(Invoke-RtIpc viewport.frame_telemetry)[-1] |
    Select-Object stale_descset_rebuilds, device_lost
Invoke-RtIpc viewport.set_scene_load_guard @{ enabled = $true }    # GERI AC
```

Hazır betik bunu sırayla yapar ve kalkanı `finally` içinde **her durumda** geri
açar:

```powershell
.\scripts\ipc\Probe-DeviceLostOnProjectOpen.ps1 -ProjectA <agir1> -ProjectB <agir2>
```

★ Kalkan **panele konmadı**. Arızalı yolu bir menüden seçilebilir bırakmak,
kuralı bir tercihe çevirirdi. Kapalıyken her açılış Scene Log'a uyarı yazıyor —
kapalı unutulmuş bir kalkan, aylar sonraki bir device-lost raporunun görünmeyen
sebebi olmasın diye.

## ★★★★ KURULAN TRIPWIRE: descriptor set'in doku kuşağı

`purgeUploadedTextureCacheLocked` `m_uploadedImages`'teki bütün `VkImage`'ları
yok ediyor ve `m_textureCacheGeneration`'ı artırıyor — ama material-preview
descriptor set'inin binding 1'ini yeniden yazmıyor. Böyle bir set'e karşı
çizilen ilk Material karesi ölü `VkImageView` örnekler ve **gönderimde**
device-lost üretir; bu, belirtinin "while submitting" varyantıyla birebir
uyuşuyor.

Binding 1 doldurulduğu anda kuşak damgalanıyor
(`materialPreviewDescSetTextureGeneration`), ve `renderInteractiveViewportImpl`
çizimden önce karşılaştırıyor. Uyuşmazsa: bir kez `SCENE_LOG_ERROR`, her olayda
sayaç, sonra drenaj + `waitIdle` + set'i yeniden kurma ve kareyi atlama.

Sonuç **değer** olarak okunur — bilerek, çünkü "log'a bak" insan adımıdır ve
tekrarlanamaz:

| alan | anlamı |
|---|---|
| `stale_descset_rebuilds > 0` | **KÖK NEDEN BULUNDU** — bir purge set'i öksüz bıraktı |
| `device_lost = true`, sayaç 0 | ariza gerçek ama sebebi bu **değil** → sınıf ELENDİ |
| ikisi de 0 | ariza **üretilemedi**; "düzeldi" demek DEĞİL |

★★ Bu iki alan `available == false` iken de yayınlanıyor. Sürücü kaybı ring'i
öldüren şeyin ta kendisi; onları `available` kapısının arkasına koymak, tam da
okunmaları gereken anda gizlerdi — ve bir ajan "ölçüm yok"u "sorun yok" diye
okurdu.

★ Purge içine **eager sökme yapılmadı**: descriptor set yalnızca
`keepPipeline == false` ile yıkılabiliyor ve o bütün viewport pipeline'larını da
götürüyor — Solid modda tamamen boşuna, ölçülmemiş bir maliyet.

## ★★★★★ SIRADAKİ ADIM — ve bu bir yöntem kararı

Bu oturumda kaynaktan **üç** hipotez kuruldu ve **üçü de** kullanıcının
gözlemiyle öldü. Bu hata sınıfı kaynaktan tahmin edilemiyor; validation layer
nesneyi ve yasadışı işlemi **adıyla** söyler. Ortam değişkeni mevcut
(doğrulandı, `VulkanBackend.cpp:7770`):

```powershell
$env:RAYTROPHI_VK_VALIDATION = "1"
.\scripts\ipc\Start-RayTrophi.ps1
# senaryo 2'yi tekrarla: agir sahne ac -> realtime'da kal -> hub'dan ikinci agir proje ac
```

Debug messenger warning+error yakalıyor. **Bir tek bu çıktı, yukarıdaki bütün
listeden daha fazlasını söyler.**

★ Yanına ücretsiz iki log kontrolü: çökmeden hemen önce
`"Texture cache near descriptor capacity"` satırı var mı, ve
`[ViewportRaster] buildRasterGeometry` satırı çökmeden önce mi sonra mı.

## Ders (bu oturumdan, kaydedilmeye değer)

★★★★ **Kullanıcının NEGATİF kolu, benim statik okumamdan daha çok eledi.**
"Ne zaman çöküyor" üç tur boyunca yanlış teori üretti; "aynı şey ne zaman
ÇÖKMÜYOR" (senaryo 1 ve 3) iki turda üç teoriyi birden öldürdü. Bu sınıfta
önce negatif kol sorulmalı.

## İlgili

- `docs/dev/REALTIME_CAMERA_MOTION_PERF.md` — aynı yolun ölçülmüş maliyeti.
- `memory/feedback_vulkan_tlas_destroy_inflight_trace.md` — device-lost / TDR
  ayrımı ve "tahminle ilerleme" kuralının ilk kaydı.
- `memory/project_viewport_project_reload.md` — proje-reload purge sözleşmesi.
