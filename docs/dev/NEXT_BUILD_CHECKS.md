# Sıradaki derleme kontrolleri — Player Android A1 (2026-11-22)

> **★ TAM DERLEME ŞART (2026-11-22, Claude) — ve ESKİ EXE'Yİ ÇALIŞTIRMA.**
> `ImageHandle` düzeni değişti (`arrayLayer`, `sharedImage`; `VulkanBackend.h` her yerde) →
> Rebuild, artımlı değil. **Viewport SPIR-V'si yeni ABI ile DAĞITILDI** (8 modül, iki yerde):
> eski exe bu shader'larla Material modda descriptor uyuşmazlığına düşer (siyah/çökme) —
> önce derle, sonra aç. Yeni .cpp'ler: `VulkanMaterialTextureStorage.cpp`,
> `VulkanTextureCompaction.cpp`, `VulkanMaterialTextureBinding.cpp` (Studio vcxproj +
> Player/Android üretildi). Android APK'yı yeniden paketle (shader staging).

## Android duraklatma menüsü: görüntü kalitesi + çıkış (2026-11-23, Sonnet)

> Yalnız Android başlatıcısı (`PlayerOperatorKidUI.cpp`) + `PlayerOperator.cpp` (Android korumalı). Windows derlemesini etkilemez.
> Yeni: duraklatma ekranında "Hızlı/Dengeli/Net" (render_scale 0.5/0.75/1.0) ve "Çıkış" düğmesi; `launcher.exit` IPC işlemi; ilk açılışta Android'de kayıtlı seçim yoksa render_scale 0.5.

1. **Menü düzeni (en hızlı).** Duraklatınca Oyunlar / Devam / Ses / Kalite / Çıkış düğmeleri taşmadan sığmalı. Bozuksa: genişlik sığdırma ölçeği (`pauseButtons`) çalışmıyor demektir.
2. **İlk açılış varsayılanı.** Temiz kurulumda `viewport.frame_timings` → `applied.width` tam çözünürlüğün ~yarısı, kalite düğmesi "Hızlı". Bozuksa: `display_restored` yolundaki Android varsayılanı uygulanmıyor.
3. **Kalite döngüsü.** Düğmeye bas: Hızlı → Dengeli → Net → Hızlı; her basışta `player.operator status` → `display.render_scale` 0.5/0.75/1.0 ve `applied.width` değişmeli. Bozuksa: `display.configure` reddediliyor (logcat).
4. **Kalıcılık.** Net seç, uygulamayı kapat/aç: Net kalmalı (yarıya dönmemeli). Bozuksa: varsayılan kayıtlı seçimi eziyor.
5. **Çıkış.** Düğme ve `launcher.exit` uygulamayı kapatmalı (etkinlik biter). Bozuksa: SDL_QUIT `quit_requested`'a düşmüyor.

★ En sinsi: kalite düğmesinin etiketi değişir ama `render_scale` gerçekte değişmez (etiket yerel aynadan gelir). Etikete güvenme: `viewport.frame_timings` içindeki `applied.width`'e bak.

## ★★ T1 partisi (2026-11-23 öğleden sonra, Claude) — TAM DERLEME ŞART (Studio + Player)

> **CANLI SONUÇ (2026-11-23 ~11:45, kullanıcı tam derledi + göz testi OK, RTX 3060 1080p workshop):**
> 1 PASS: `feature_spec {active:true, terrain/graph/weather:false}`, `tiled_gpu=false`. Boşta **68,7 Hz**, raster 13,1 ms.
> 2 PASS: Player `set_quality performance` → shadowed 5→4, geri `auto`; `lights.list` 6 ışık.
> 3 PASS: `environment upload_ms=127 format=rgba16f clamped_texels=0 size=4096x2048`.
> 4 PASS (kısmi): `paint_wet` çağrı başı **0,404 → 0,288 ms (−%29)**. Patlamada döngü **40–43 → 47–52 Hz**.
> Saniye başı toplam (150 ms/s) değişmedi, çünkü kare hızı arttıkça tick sayısı da arttı. Kalan
> maliyet büyük olasılıkla tam ızgara grup başlatma + stamp. Indirect aktif liste hâlâ aday.
> AÇIK: tablet ölçümü (tiled_gpu=true + uyarı, HDRI A/B, paint_wet), Studio terrain/graph yeniden kurulumu.

> Yaygın header'lar değişti: `VulkanBackend.h` (`sampleRasterAppliedState` artık **virtual**,
> `uploadEnvironmentTexture`, `m_uploadFloatAsHalf`), `RasterStageTimings.h` (`RasterAppliedState` +7 alan),
> `RtApi.h` (`RasterTimingInfo` +6 alan). **Artımlı derleme eski .obj bırakırsa erişim ihlali kod hatası
> sanılır: Rebuild.** Aşağıdaki Vulkan 1.1 ve O3 partileri de aynı derlemeye girer.
> /Zs Studio+Player ve NDK arm64 PASS. Descriptor'lar yeniden üretildi, `audit_ipc_capabilities` OK.
> Shader: `paint_wet.spv` iki yere dağıtıldı (eski exe ile de çalışır, ABI aynı).

1. **IPC alanları (en hızlı).** `viewport.frame_timings` → `applied.feature_spec`
   (`{active, terrain, graph, weather, rebuilds}`) ve `applied.tiled_gpu`. Windows workshop'ta beklenen
   `active=true, terrain/graph/weather=false, tiled_gpu=false`. Tablette `tiled_gpu=true` ve `warnings`
   içinde "tiled GPU: ... sky carries the main pass" satırı. ★ Sinsi: `active=false` ise pipeline spec'siz
   kurulmuş demektir. Tam shader çalışır ve görüntü doğru kalır, yalnız hız kazancı yoktur; log'daki
   `[MaterialSpec]` satırına bak.
2. **Player'da yeni A/B anahtarları.** Player IPC'den: `viewport.set_quality {preset:"performance"}`
   → `frame_timings.applied.quality_preset` "performance" ve `shadowed_lights` azalmalı.
   `viewport.set_preview_lighting`, `viewport.set_taa`, `lights.list`, `lights.set_intensity {index,value:0}`
   → o ışık kareden çıkmalı (`scene_lights` aynı, görüntüde ışık yok). Hata `unsupported_in_player`
   ise politika satırı derlemeye girmemiş. ★ Testten sonra kaliteyi `auto`'ya, ışık yoğunluğunu
   eski değerine geri yaz (ışık değişikliği projeyi "değişti" işaretler).
3. **O4a HDRI 16F.** HDRI'li sahneyi aç. Log: `[World] Vulkan environment upload_ms=... format=rgba16f
   clamped_texels=N`. Gökyüzü ve yansımalar öncekiyle aynı görünmeli. ★ Sinsi: `clamped_texels` büyükse
   (güneş diski 65504 üstü) güneş parlaklığı kırpılır. Gökyüzü makul görünür ama sahne biraz sönük
   olur. N'yi not et. Rendered (RT) modunda HDRI ışığı da aynı olmalı.
   Tablette HDRI A/B'si (Codex'in 4018/3204 ms ölçümü) tekrar alınmalı; beklenen fark küçülür.
4. **O12 ıslak boya (shader; derlemeden bağımsız, Player'ı yeniden başlatmak yeter).** Workshop'ta
   balonları patlat: boya, damlama, kuruma, pürüz ve normal kanalları önceki gibi olmalı.
   `perf.gpu_kernel_timings` → `paint_wet` çağrı başı ms. Önce **0,40 ms** (RTX 3060, patlama), patlama
   karesi ~3,4 ms. Beklenen belirgin düşüş. ★ Sinsi: boya **tile sınırında durur** ya da yeni boyanan
   bölgenin komşusu kurumaz/akmaz. Bu, workgroup kapısının (`s_gate`) yanlış karar verdiği anlamına gelir.
   Tile kenarına düşen damlaya yakından bak.
5. **Regresyon:** Studio'da Material modu, terrain'li bir sahne, proje aç/kapat. Device lost olmamalı.

## ★ Vulkan 1.1 cihaz desteği (Omix X700 Mali-G76, 2026-11-23, Claude) — ARTIMLI derleme

Değişen: `RayFusionViewportPasses.cpp` (kapı: 1.1 + `VK_EXT_scalar_block_layout` +
`VK_KHR_buffer_device_address`), `VulkanBackend.cpp` (1.1 cihazda scalar eklentisi istenir, yoksa özellik
kapalı), `SimulationComputeVulkan.cpp` (Android BDA adı KHR'ye düşer). Shader: yeni
`scripts/compile_player_android_vk11.py` → `x64/Release/shaders_vk11` (SPIR-V 1.3, 346 shader). Staging
önce onu alır, `Build-PlayerApk.ps1` staging'den önce kendisi çalıştırır. 12 ışın izleme aşaması 1.2
kopyasıyla kalır (1.1 cihaz yüklemez). /Zs Studio+Player ve NDK arm64 -fsyntax-only PASS.

1. **Windows regresyon (en hızlı).** Studio + Player normal açılmalı. Windows SPV'leri değişmedi, kapı
   1.2+ cihazda eskisi gibi geçer. Bozulursa ilk şüpheli scalar layout satırı (`VulkanBackend.cpp`).
2. **APK paketleme.** `.\scripts\Build-PlayerApk.ps1 -Abi arm64`. Betik artık varsayılan olarak **Release** derliyor ve debug keystore ile imzalıyor (`-Configuration Debug` eski yol); APK adı `...-arm64-release-<tarih>.apk`. Çıktıda `vk11: N compiled ... 0 errors`
   ve staging'de `shaders: 3xx from shaders_vk11`. `ERROR` satırı varsa o shader 1.1'e inemiyor, çıktıda adı yazar.
3. **Lenovo (Vulkan 1.3, Mali-G52) regresyon.** Workshop açılmalı, dokular + ıslak boya doğru olmalı.
   Artık 1.1 SPIR-V ile çalışıyor. ★ Sinsi: aynı görüntü ama hız farkı. SPIR-V sürümü kodu değiştirmez,
   `frame_gpu_mean_ms` aynı kalmalı. Belirgin fark varsa sürücü 1.3 modülü farklı derliyor demektir. Not et.
4. **Emülatörde 1.1 kapısı.** `adb shell setprop debug.raytrophi.force_device_api 1.1` → uygulamayı
   yeniden başlat. RET EKRANI ÇIKMAMALI (emülatör scalar+BDA eklentilerini ilan ediyorsa). Çıkarsa ret
   metnindeki eklenti adı eksik olandır. Sonra `setprop ... ""` ile geri al. Not: bu yalnız kapıyı sınar;
   sürücü yine 1.3 olduğu için cihaz-oluşturma 1.1 yolu ancak gerçek cihazda sınanır.
5. **Omix X700 (arkadaş telefonu).** Ret ekranı yerine kütüphane → oyun açılmalı. Olası yeni ret:
   SSBO/aşama ve `independentBlend` zaten var. `maxImageArrayLayers 1024` doku sayfasını kısar
   (beklenen, hata değil). İlk karede siyah ya da çökme varsa log'daki ilk `vkCreateShaderModule` /
   pipeline hatası 1.1 SPIR-V'nin reddedildiğini gösterir; hangi shader olduğunu kaydet.
   ★ Sinsi: oyun açılır ama **ıslak boya görünmez** (float atomik yok → GPU boya fallback'i). Bu
   bilinen bir kısıt; Adreno için Codex'in kapısı aynı yolu kullanıyor.

## ★ Perf O3 — sahne özelliğine göre özelleştirilmiş malzeme shader'ı (2026-11-23, Claude)

> **CANLI ÖLÇÜLDÜ (2026-11-23 10:22, RTX 3060 1080p, workshop):** log `built with terrain=0 graph=0 weather=0`.
> main_pass 8,65 → **5,01 ms**, transmission 5,6 → 4,64 ms, raster 17,5 → **13,1 ms**, döngü 49,6 → **63,8 Hz**
> (O10+O11+O3 birlikte). Madde 3–6 (sonradan gelen özellik, bulut, proje değiştirme, tablet) açık.

Artımlı derleme yeter. Değişen header yalnız `VulkanViewportBackend.h` (7 içerici); O10/O11 ile aynı
derlemede gelir. SPV dağıtıldı. Eski exe bile yeni SPV ile tam shader'ı çalıştırır (doğrulandı: görüntü normal).

1. **Log satırı (en hızlı).** Workshop'u aç. Log'da
   `[MaterialSpec] material pipeline built with terrain=0 graph=0 weather=0` görmelisin (workshop'ta
   terrain, graph ve bulut yok). Bir bit `1` ise sahnede o özellik var ya da CPU kopyası kanıtlayamadı
   (dış malzeme tamponu). Bu durum hata değil, ama kazanç o bit kadar azalır. Satır HİÇ yoksa pipeline
   bu yoldan kurulmamıştır (`VulkanBackend.cpp:17557` eski yolu, ya da durum yakalanamadı: "not capturable" WARN).
2. **Hız (PC, 1080p, `--pacing 0`).** `viewport.frame_timings` → `main_pass`. O10 sonrası 7,4 ms
   (gürültü ±1 ms). Beklenen belirgin düşüş. RTX'te registerlar bol olduğu için kazanç Mali'den küçük
   olabilir; asıl hedef tablet.
3. **Özellik sonradan gelince (Studio).** Terrain'siz sahnede Material moddayken bir terrain oluştur
   ya da bir malzemeye node graph bağla. Log: `[MaterialSpec] material pipeline rebuilt for ...` ve
   terrain/graph **o karede doğru görünmeli**. `perf.list` → `viewport.material_spec.rebuild` süresi
   (ilk yeniden kurulum maliyeti, masaüstünde yüzlerce ms olabilir). ★ **En sinsi durum:** yeniden kurulum
   olmadan terrain düz tek renk görünür (splat karışmıyor) ya da graph'lı malzeme graph'sız değerleriyle
   çizilir. Makul görünür, kimse hata diye bildirmez. Bunu gördüysen maske o özelliği kaçırıyor demektir
   (`materialPreviewSceneFeatureMask`).
4. **Bulut/hava.** Bulut katmanını aç: yeniden kurulum logu + yüzeyde ıslaklık/bulut gölgesi/şimşek
   görünmeli. Kapatınca ~240 kare sonra `rebuilt for none` (daralma gecikmeli, bilerek).
5. **Proje değiştir / pencere boyutu.** Çökme veya device lost olmamalı. Yeniden açılışta yine
   `built with` satırı gelir.
6. **Tablet (APK yeniden paketlendikten sonra).** Aynı log satırı. 960×600 ve 1920×1200'de
   `frame_gpu_mean_ms`'i eski değerlerle karşılaştır (960×600: 893–1465 ms; 1920×1200: ≈3,8–4,0 s).
   malioc beklentisi: ana shader stack 1664 → 704 B, LS çevrimi ~3× az.
7. **IPC açığı (bilinen).** Maske şimdilik yalnız `perf.list`'te var: `viewport.material_spec.mask.*`
   girdilerinden `seq`'i en yüksek olan canlı maskedir (yeniden kurulum olmadıysa girdi yok, log'a bak).
   `frame_timings.applied.feature_spec` alanı bir sonraki TAM derlemeye bırakıldı (RtApi.h / IBackend.h).

## ★ Perf O10 + O11 (2026-11-23 öğle, Claude) — ARTIMLI derleme yeter

O10 (shader) zaten dağıtıldı ve PC'de ölçüldü. O11 yalnız `Backend/VulkanViewportParticles.cpp`
değişikliği, header yok. Studio + Player artımlı derleme, sonra APK yeniden paketleme.

1. **O10 görüntü (derleme gerekmez, hızlı).** Workshop'ta balonlar yarı saydam ve renkli, duvar/zemin
   aynı tonda olmalı. Cam/reçine malzemeli bir Studio sahnesinde (transmission > 0, interior depth,
   bubble) önceki görünüm korunmalı. ★ Sinsi: **transmission'ı 0 olan ama `interior_depth` (transmission_density)
   değeri olan opak malzeme** kapının içinde kalır, değişmez. Ama **transmission = 0 ve interior = 0
   olan bir malzemede** görünür bir fark çıkarsa kapı yanlış bir tüketiciyi kesiyor demektir
   (`frag:1513` glassPath). Beklenen fark sıfır.
   Ölçülen (RTX 3060 1080p): `main_pass` 8,65 → **7,38 ms**, transmission 5,6 → 5,25 ms, döngü 49,6 → 53,7 Hz.
2. **O11 kare temposu (derlemeden sonra).** Workshop'u Player'da `--pacing 0 --continuous` ile aç,
   10 sn bekle, `agent.get_state_summary.runtime.loop.frame` oku. Beklenen: `advance_ms_avg` 12–15 ms'den
   **birkaç ms'ye** düşer, `loop_hz` GPU raster süresine yaklaşır (16 ms raster → ~60 Hz). `advance` hâlâ
   çözünürlükle ölçekleniyorsa (`display.configure render_scale` 0.5 → düşüyor) başka bir GPU
   beklemesi daha var. Sıradaki aday `Paint::wetPaintGpu().publish` / `collectReadback(wait=true)`.
   ★ Ölçek değiştirdiysen 1.0'a geri yaz (tercihe kaydediliyor).
3. **O11 doğruluk.** Balon patlat: damlalar (pulled parçacıklar) görünmeli, iz bırakmalı, kaybolmalı.
   Studio'da CPU parçacıklı bir sahne (billboard/additive) oynat: parçacıklar titremeden, eski
   kareden kalıntı olmadan çizilmeli. ★ Sinsi: **parçacık sistemi eklenip silinirken** ya da LUT
   (profil rengi) değişirken bir kare yanlış renk/boş çizerse kontrol fonksiyonu
   (`writesDeviceState`) bir yazımı kaçırıyor demektir. Belirti tek kare, kimse rapor etmez:
   profil rengini oynarken yakından bak. Device lost olursa ilk şüpheli bu.
4. **Tablet (APK yeniden paketlendikten sonra).** Aşağıdaki akşam listesi aynen geçerli. `advance_ms_avg`
   ve `frame_gpu_mean_ms`'i O10+O11 öncesi değerlerle (≈3,8–4,0 s GPU) karşılaştır.

## ★ Android performansı — akşam tablet oturumu (2026-11-23, Claude)

Plan ve gerekçe: [PLAYER_ANDROID_PERF_PLAN.md](PLAYER_ANDROID_PERF_PLAN.md). Bu partide kod
değişmedi. Aşağıdakiler **mevcut APK ile** yapılır ve kod partisinin (O1–O6) hangi sırayla
gideceğini belirler. Sıra: hızlı/bağımsız olanlar önce, diğerlerinin okunuşunu etkileyenler sonra.

1. **HDRI formatı destekleniyor mu (30 sn, uygulama gerekmez).** `adb -s HVA751MH shell cmd gpu vkjson`
   → `VK_FORMAT_R32G32B32A32_SFLOAT` (109) `optimalTilingFeatures` içinde
   `SAMPLED_IMAGE_FILTER_LINEAR` (0x1000) biti var mı? **Yoksa:** bugünkü HDRI örneklemesi geçersiz
   kullanım ve sonuç tanımsız. O4 (RGBA16F + mip) bir doğruluk düzeltmesine dönüşür ve listenin başına geçer.
2. **GPU saati ve ısıl durum.** Oyun 2 dk açıkken `adb shell cat /sys/class/misc/mali0/device/cur_freq`
   (okunamazsa `dumpsys thermalservice`). Ölçümü soğuk cihazda ve 5 dk sonra iki kez al. **Saat
   yarıya düşüyorsa:** 3,8 s'nin bir kısmı ısıl kısıtlamadır, shader optimizasyonunun kazancı da
   ısınmış cihazda ölçülmeli. ★ Sinsi: soğuk cihazda alınan tek ölçüm "düzeldi" gösterir, çocuk
   10 dk oynayınca kare yine çöker.
3. **`game_design` tik sayısı (B6).** Açılışta `agent.get_state_summary.runtime.scene_load`'u 1 sn
   aralıkla oku. `game_design` aşamasında `detail` ("waiting for collider SDF") kaç kez tekrar
   ediyor? Aynı aralıktaki `viewport.frame_timings` kare sayısıyla karşılaştır. **~10 kare × ~3,8 s ise**
   açılış kare süresine bağlı demektir ve O7 (yüklemede 3D çizme) açılışı tek başına ~30 s kısaltır.
   **Kare sayısı küçük, süre uzunsa** darboğaz SDF üretimidir (CPU) ve bu ayrı bir iş.
4. **Ekranda balon yok vs. balon var (B5, 960×600 sabit hedef).** Kamera balonsuz duvara bakarken ve
   ekranın yarısı balonken `frame_gpu_mean_ms` al. Fark balon başına transmission bedelidir; O5'in
   tavan kazancı budur. ★ Aşama tablosundaki `sky` sütununa **bakma**: TBDR'de ana geçişin
   tamamı oraya yazılıyor (B1).
5. ~~malioc~~ **YAPILDI (öğlen):** ana shader 64 register, %50 occupancy, **1632 B stack**
   (spill 664). PC'de `advance`'ın GPU'yu beklediği ölçüldü (parçacık yüklemesindeki drain), patlamada
   `paint_wet` ~3,4 ms/kare. Ayrıntı ve yeni öncelik: plan notunun §3b'si (O11 → O10 → O3 → O12).
   Bu akşam ek madde: **tablette `advance_ms_avg` ve `paint_wet` saniye başı ms** — `agent.get_state_summary.runtime.loop.frame`
   ve `perf.gpu_kernel_timings` (patlama sırasında). ★ `display.configure` ölçeği **tercihlere
   kaydeder**: ölçek değiştirdiysen testin sonunda 1.0'a geri yaz.

## Player HDR karşılaştırması (2026-11-23)

- Tek sahneli Android APK açılış kontrolü (yeni APK bekliyor). Player otomatik
  olarak tek oyunu seçmeli ve ilk oyun karesine kadar mevcut yükleme ekranını
  göstermeli. Dış IPC'de `player.operator {operation:"launcher.status"}`
  ile `loading → playing` geçişini kontrol et. Çok sahneli APK seçim ekranında
  beklemeli. Otomatik açılış ve kütüphane seçimi aynı yükleme yolunu kullanır.

- Android APK teslim klasörü `E:\rt_android_build`. ARM64 Release test paketi
  `RayTrophiPlayer-arm64-release-workshop-2026-11-23.apk` yalnız Boyama Odası'nı içerir.
- Lenovo ilk kare: Debug **163,6 s** → Release **63,8 s**. `game_design`
  **122,7 → 38,9 s**, `prepare_particles` **28,6 → 22,1 s**, render ayarlarını
  yükleme **2,28 → 0,62 s**. Bunlar birer açılış ölçümüdür, iki koşulda da HDRI açıktı.
- Dünya IPC doğrulandı. `world.get` modu döndürdü, `world.set_mode` ile yapılan
  HDRI → düz renk → HDRI testinde tam hedef GPU ortalamaları sırasıyla
  **4018 / 3204 / 3853 ms** (5 / 6 / 6 kare). Geçersiz mod reddedildi.
- Aynı Release oturumunda HDRI ve kamera sabitken **960×600: 1055 ms** (9 kare),
  **480×300: 339 ms** (22 kare). Ölçümler `viewport.reset_frame_timings` sonrasında
  `viewport.frame_timings.frame_gpu_mean_ms` alanından alındı. Asenkron Player
  döngüsündeki CPU render süresi, toplam GPU çizim maliyetini tek başına göstermez.
  Test sonunda HDRI ve %100 iç ölçek geri yüklendi. Tablet 1920×1200 ve 204 dpi.
- Açılış A/B için Studio'da ayrı bir proje kopyasını `solid` modunda kaydet.
  Dünya serileştiricisi bu modda boş HDR yolu yazar. İki projeyi aynı Player
  derlemesi ve render hedefiyle açıp `agent.get_state_summary.runtime.scene_load`
  içindeki toplamı ve aşamaları karşılaştır. Canlı dünya modu değişikliği,
  sahne açılırken yapılan HDR yüklemesini ölçmez.
- Release SceneLog için erişilebilir bir tanılama yolu doğrula. Yeni sayaçlar
  `HDRI read_decode_ms` ve `Vulkan environment upload_ms`; Release APK
  `run-as` ile özel uygulama dosyası okumaya uygun değildir.
- Sonraki render ölçümünde sabit kamera ve 960×600 hedefte gölgeleri, ışık
  döngüsünü ve transmission'ı birer birer değiştir. Mali ölçümündeki `sky`
  etiketini tek başına HDRI maliyeti olarak yorumlama.

## Doku dizisi C modeli adım 1+2 — tek depolama + viewport sayfa ABI'si

> **SONUÇ 2026-11-22 23:40 (Claude, Windows RTX 3060, bedroom.rtp):** 1 PASS (`packed=33..36
> pages=21..26 over_budget=0 ~204 MB`), 2 PASS (Material = Rendered dokuları; ilk derlemede
> FAIL'di: GPU_ONLY ref tablosu `initialData` yazılmıyordu → `uploadBuffer` ile düzeltildi),
> 4 PASS (yeniden açma, device_lost=0), 7 PASS (yansıma+GI `ready`, RT gölge 3,17 M ışın).
> AÇIK: 3 ıslak boya, 5 VRAM, 6 matcap, 8.
>
> **9 LENOVO (2026-11-23 00:05, APK 2340, Mali-G52 MC2): KABUL PASS, PERF FAIL.** Ret YOK
> (descriptor indexing gerekçesi gitti), workshop.rtp açıldı ve doğru çizildi; sıkıştırma
> `packed=35 pages=2 35 MB`. Ama 1920x1200'de **kare ≈3,8 s GPU**; piksele doğrusal
> (0.5→1,5 s, 0.25→0,37 s) = fragment maliyeti. En büyük aşamalar "sky" 2,6 s + "transmission"
> 1,1 s (TBDR'de ana pass süresi muhtemelen "sky"e yazılıyor). ★ 32 dallı switch ELENDİ:
> 2 dallı shader varyantı tablette ölçüldü → fark yok (3980 ms). Yükleme ≈125–129 s
> (`game_design` 88 s, `prepare_particles` 28 s). Ayrıca: IPC `launcher.select index=0`
> yüklemeyi başlatmadı (kullanıcı dokunarak başlattı) — ayrı bakılacak.
> **Ayrıştırma (960x600):** temel 1465 ms (sky 945, transmission 464); kamera düz duvara
> yakın (ekranın çoğu tek basit malzeme + 1 balon) **893 ms** (sky 678, transmission 166).
> → Maliyet içerik değil, HER pikselde çalışan temel aydınlatma yolu: sahne 6 ışık, 6'sı
> gölgeli, 3 kaskad, preset `scene`, kalite `auto`. Transmission ekrandaki balonla ölçekleniyor.
> Player IPC'si kalite/ışıklandırma/dünya anahtarlarını AÇMIYOR (`unsupported_in_player`) →
> cihazda ayrıştırma yalnız shader varyantıyla yapılabiliyor (CLAUDE.md §1 açığı).
 Sonraki artımlı derlemede: sıkıştırma
> önizleme set'inin kuşak damgasını basıyor → `STALE material-preview descriptor set` ERROR
> yanlış alarmı ve gereksiz set yeniden kurulumu gider (yalnız log; davranış zaten doğru).

Adım 2: viewport (material preview, RT gölge, yansıma, ekran GI, prob izi) artık 32
`sampler2DArray` sayfa + slot→ref tablosu okuyor; descriptor indexing İSTEMİYOR. RT render
(Rendered/F12) bindless kaldı. Aşağıdaki 1–6 iki adımı birlikte sınar; 7–9 adım 2'ye özgü.

Bu parti dokuları dizi sayfalarının katmanlarına taşır; RT ve viewport katman başına
2D view okur. **Doğru çalışıyorsa hiçbir görsel fark yoktur** — fark varsa hatadır.

1. **Log satırı (en hızlı).** workshop.rtp aç. Log'da
   `[TexturePages] compaction result=0 packed=N skipped=K pages=P` görmelisin; N ≈ sahnedeki
   doku sayısı, `over_viewport_budget=0` tercihen. `result` sıfırdan farklıysa sayfa
   ayırma/görünüm oluşturma başarısız (doku yine eski tekli image'da kalır, görünüm bozulmaz
   — ama C modeli devrede değil). Satır HİÇ yoksa sıkıştırma çağrılmıyor (batch yolu değişmiş).
2. **Material ve Rendered görünümü aynı mı?** Dokulu yüzeyler (albedo, normal, roughness)
   öncekiyle aynı. ★ Sinsi hata: **dokular birbirine karışmış** (bir yüzeyde başka
   malzemenin dokusu) = yanlış katman/view; **bulanık/keskin değişmiş** = mip aralığı veya
   sampler yanlış. İkisi de makul görünebilir; tanıdık bir yüzeye yakından bak.
3. **Islak boya (Boyama Odası, Studio veya Player).** Boyanın hem viewport'ta hem Rendered'da
   göründüğünü, uzaklaşınca (mip) da kaybolmadığını doğrula. Boya yanlış yüzeyde/katmanda
   çıkıyorsa boya kopyası `arrayLayer` almıyor.
4. **Proje yeniden aç / başka proje aç** (purge yolu). Çökme veya device lost olmamalı;
   ikinci açılışta log yine `compaction result=0`.
5. **VRAM.** Doku belleği öncekiyle aynı veya biraz az olmalı (sayfa = tam boyut). Belirgin
   artış = eski tekli image'lar silinmiyor.
6. **Matcap** (Solid/matcap modunda kullanıcı matcap'i): görünmeli, siyah değil.

7. **Yansıma / ekran GI / RT gölge (ray query'li masaüstü GPU).** Durum yüzeylerinde
   `supported=true` olmalı; reason artık "descriptor indexing" demez. Yansımada dokulu
   yüzeyler doğru dokuyla görünmeli. ★ Sinsi: yansımada/GI'da dokular **düz renk** = ref
   tablosu boş/bayat (binding 14) — log'daki `packed` sayısıyla karşılaştır.
8. **Sıkıştırmadan SONRA eklenen doku** (sahne açıkken yeni dokulu asset içe aktar):
   viewport'ta bir sonraki sahne doku batch'ine kadar dokusuz görünebilir — BİLİNEN
   sınır, hata değil. Rendered'da hemen görünmeli. Hiç görünmüyorsa (yeni batch de
   gelmiyor) not et: canlı yeniden sıkıştırma tetikleyicisi sonraki iş.
9. **Lenovo arm64 (`adb -s HVA751MH`), APK yeniden paketlendikten sonra.** RayFusion artık
   "descriptor indexing" ile REDDETMEMELİ. Ret ekranı başka bir gerekçeyle çıkarsa
   (`independentBlend`, push, bellek, pipeline) o gerekçeyi aynen kaydet — sıradaki eksik
   özellik budur. Açılırsa Boyama Odası'nda dokular + ıslak boya görünmeli; ilk kare süresi
   ve FPS'i kaydet. ★ Mali'de binding 25–27 (temporal/bulut) partially-bound OLMADAN
   yazılmamış kalırsa sürücü çökebilir — çökme ilk Material karesinde ise ilk şüpheli bu.

## Malzeme doku dizileri kabul sırası

1. **Windows Studio önce/sonra.** Workshop ve kids-play-room sahnelerini dış
   named-pipe IPC istemcisiyle aynı kamera, çözünürlük, örnek sayısı, seed,
   ışık, pozlama ve simülasyon durumunda render et. Piksel eşitliği, değişen
   piksel oranı, kanal başına maksimum mutlak fark ve RMSE'yi kaydet. Gürültü
   varsa aynı sürümle tekrar al. Yakın/uzak yüzeyleri, cutout ve canlı paint
   düzenlemesini ayrıca karşılaştır. Eşit ısınma ve örnek sayısıyla
   `raster_main_gpu_samples` ve sabit bütçeli probe compute sürelerini ölç.
   %5'i aşan artışta specialization varyantını ve pipeline hazırlama maliyetini
   değerlendir. Geçici kareleri sonunda sil.
2. **x86_64 emülatör.** İki sahnede ilk kareyi, gerçek `texture_arrays`
   sayaçlarını ve async düzenleme sırasında ilerleyen kare sayısını doğrula.
   Siyah veya durağan karede pipeline/format tanısını ve IPC snapshot'ını al.
3. **Lenovo arm64, `adb -s HVA751MH`.** İki sahnede ilk kareye süreyi ve sabit
   aralıkta çözünürlük, FPS ve frame time değerlerini kaydet. Sahne yükleme
   tanısını, varsa GPU ret nedenini ve Android yerel IPC üzerinden
   `texture_arrays` değerlerini al. Mali-G52 raster fragment ve probe compute
   GPU sürelerini ayrı ölç.

## Android kenardan kenara (kamera kesiği) — yalnız APK, C++ değişmedi

0. **APK tek komutla alınır:** `.\scripts\Build-PlayerApk.ps1 -Abi both`. Staging 512 dahil; APK'lar `E:/rt_android_build/` altına tarihli adla kopyalanır, `-Install` ile kurulur. Betik "FAILED at step: X" ile durursa hata o adımdadır: `tools` yol eksik demektir, `staging` dist paketinin eksik olduğunu, `gradle` NDK derleme hatasını gösterir (yukarıdaki ilk hata satırına bak).
1. **Emülatör:** APK'yı kur (`-Abi x86_64 -Install`) ve kütüphaneyi aç. Görünüm önceki gibi olmalı: durum ve gezinme çubuğu yok, siyah şerit yok. Bozuksa `PlayerActivity.onCreate` içindeki attribute ayarı SDL'nin pencere kurulumuyla çakışıyor demektir.
2. **Omix X700 (kamera deliği solda):** Sol kenardaki ~68 px siyah şerit kaybolmalı, gradyan ekranın kenarına kadar uzanmalı. **En sinsi durum:** Şerit gider ama logo veya bir düğme kamera deliğinin altında kalır. Ekran makul görünür, kimse bunu hata diye bildirmez. Logonun sol kenarı ile deliği karşılaştır; çakışıyorsa güvenli alan insetlerini JNI ile okumak gerekir.
3. **Oyun içi:** Aynı cihazda oyun açılmaz (Mali-G76, Vulkan 1.1 cihaz → ret ekranı beklenir). Ret ekranı tam genişlikte görünmeli. Delik metnin ilk harflerini örtüyorsa madde 2'deki inset işi buna da uygulanır.

## Android 512 doku staging

1. `python scripts/stage_player_android_assets.py --all --max-texture 512`.
   Beklenen doku sayısı workshop 35, kids-play-room 119. Gömülü yük sırasıyla
   118 439 B ve 1 058 553 B (yerel Pillow 10.0.1). Farklı sayıda doku, değişmiş
   sahne manifestini veya kayıp doku kaydını gösterir. JPEG/PNG kodlayıcı sürümü
   bayt toplamını değiştirebilir.
2. x86_64 APK'yı paketleyip emülatöre kur. İki sahneyi aç, ilk karede dokulu
   yüzeyleri ve boyanın normal/height tepkisini kontrol et. Geometri
   görünüp doku ters, siyah veya başka yüzeyden geliyorsa manifest ofseti,
   doku ID'si ve normal decode yolunu incele.

## Vulkan malzeme doku dizileri kabulü

1. Uygulama partisi teslim edildiğinde Studio ve Player için temiz tam build
   ve iki ABI için APK al. Eski Windows Studio sürümünde sabit kamera ve
   ayarlarla iki `render.start` baseline karesini ve A6
   `raster_main_gpu_samples` verisini önce kaydet.
2. Yeni Windows Studio'da aynı kareleri karşılaştır. Değişen piksel oranı,
   maksimum fark, RMSE ve raster GPU süre farkını yaz. %5 üstünde
   specialization constant ile kova dalı daraltmayı değerlendir.
3. x86_64 emülatörde iki sahne ve canlı malzeme güncellemesi. İlk kare,
   doğru doku/normal ve edit sırasında ilerleyen kare sayacı beklenir.
4. Lenovo arm64 `adb -s HVA751MH` ile iki sahnede ilk kare süresi,
   FPS, raster ve probe GPU zamanları ve `texture_arrays.allocated_bytes`
   kaydet. Siyah kare veya ret durumunda ilk gerçek driver/format/limit
   hatasını kaydet.

> **Durum:** CANLI — Windows Player Release x64 derlemesi ve demoda fareyle balon yakalama geçti. Studio derlemesi de kullanıcı onayıyla geçti. İki hedefte değişen C++ birimleri MSVC x64 `/Zs` PASS.

## Android Vulkan sürüm kapısı (2026-11-22)

1. **APK.** JDK 17 ve mevcut Android SDK ile aşağıdaki komutları depo kökünde çalıştır. Her derlemenin ardından oluşan APK'yı bir sonraki derlemeden önce kur veya kopyala. İki oyunlu kütüphaneden emülatörde bir oyun seç.
   ```powershell
   $env:JAVA_HOME = 'C:\Program Files\Java\jdk-17'
   python scripts/stage_player_android_assets.py --all
   if ($LASTEXITCODE) { throw "Varlık hazırlığı başarısız" }
   $rtGradleArgs = @('assembleDebug', '--no-daemon',
       '-PrtSdl2Dir=E:/SDL2-2.30.4/SDL2-2.30.11',
       '-PrtSdl2ImageDir=E:/SDL2-2.30.4/SDL2_image-2.8.2_src')
   android/gradlew.bat -p android @rtGradleArgs -PrtAbis=x86_64
   if ($LASTEXITCODE) { throw "x86_64 APK derlemesi başarısız" }
   adb -e install -r android/app/build/outputs/apk/debug/app-debug.apk
   android/gradlew.bat -p android @rtGradleArgs -PrtAbis=arm64-v8a
   if ($LASTEXITCODE) { throw "arm64 APK derlemesi başarısız" }
   Copy-Item android/app/build/outputs/apk/debug/app-debug.apk E:/rt_android_build/RayTrophiPlayer-arm64-2026-11-22.apk
   ```
2. **Emülatör, loader 1.1.** Vulkan 1.2+ ve RayFusion özelliklerini sağlayan emülatörde aşağıdaki ayarlarla başlat. İlk kare gelmeli. IPC `agent.get_state_summary.runtime` içinde `loader_api == "1.1"`, `device_api >= "1.2"` ve `frame_available == true` olmalı. IPC için `adb -e forward tcp:47391 localabstract:RayTrophiPlayer` ve `Connect-RtIpc -Port 47391` kullan.
   ```powershell
   adb -e shell setprop debug.raytrophi.fail_viewport_startup 0
   adb -e shell setprop debug.raytrophi.force_loader_api 1.1
   adb -e shell setprop debug.raytrophi.force_device_api 0
   adb -e shell am force-stop com.raytrophi.player
   adb -e shell am start -n com.raytrophi.player/.PlayerActivity
   ```
3. **Emülatör, cihaz 1.1.** `adb -e shell setprop debug.raytrophi.force_device_api 1.1` çalıştır, uygulamayı durdurup yeniden başlat ve aynı oyunu aç. Mevcut Türkçe/İngilizce GPU ret başlığı, GPU adı, Vulkan 1.1, 1.2 gereksinimi ve `loader 1.1, device 1.1` görünmeli. İlk kare gelmemeli. IPC'de iki alan `"1.1"`, `frame_available` false olmalı. Ardından iki `force_*` özelliğini `0` yap, uygulamayı yeniden başlat ve ilk kareyi tekrar doğrula.
4. **Windows ve telefon.** Studio ve Windows Player Release x64 artımlı derleme sonrası Windows Player'da normal bir sahne aç. `rt.player.runtime_status()` ile IPC runtime alanlarını karşılaştır. Arm64 APK'yı telefonda aç. İlk kare gelmezse ekranda GPU adı ve loader/cihaz sürümü çiftini kaydet.

## Android çocuk başlatıcısı — Windows Player derlemesi (2026-11-22)

Sıralama: bağımsız/hızlı olan önce.

1. **Windows regresyonu (önce):** Studio + Player derlenir; `RayTrophiPlayer.vcxproj` yeniden üretildi (+`PlayerOperatorKidUI.cpp`, Windows'ta boş TU). Player'da F2/F6 konsol, Esc, "Play experience", 6 nav sayfası, kart tıklama = seçim + "Load selected scene" eskisi gibi. Bozuksa: `playerOperatorMenuButtonHit` kaldırıldı (yalnız Android çağırıyordu), `playerConsoleOpen` eklendi; `console.status` artık `playerConsoleOpen()` döner.
2. **IPC:** `player.operator launcher.status` Windows'ta açık hata ("only in the Android Player") vermeli; `console.open/close/status` aynı.
3. **HUD:** Windows'ta varsayılan HUD AÇIK kalmalı (yalnız Android kapalı). Kapalı gelirse `PlayerMain.cpp` `#ifdef` yanlış.
4. **Android (isteğe bağlı):** `python scripts/stage_player_android_assets.py --all`, APK kur: kütüphane kartları, karta dokun → yükleme → oyun, Geri → duraklat. **En sinsi:** `install -r` sonrası eski oyun listesi görünürse AAsset okuması (`PlatformOs.cpp`) çalışmıyor demektir.

## Player Android oyun menüsü — Windows regresyonu önce (2026-11-22)

Paylaşılan dosyalara dokunuldu (`PlayerOperator*.cpp/.h`, `PlayerMain.cpp`, `AppWindow.cpp`, `PlayerOperator.h/UI.h`, i18n). Windows kodu `#ifdef __ANDROID__` dışında aynı; ölçek `S(x)` Windows'ta `x*1.0`. Player + Studio tam derleme (UI.h/Widgets.h değişti).

1. **Studio + Player derlenir ve Player linklenir.** Bozuksa: `playerOperatorScale`/`playerOperatorMenuButtonHit`/`playerOperatorPage` (PlayerOperatorUI.cpp) ya da `wrapped`/`tooltipText` (Widgets.h) eksik demek.
2. **Windows Player açılışı aynı:** konsol açık gelir, "OPERATOR CONSOLE / DEVELOPER EDITION", 6 nav sayfası (Camera & room, Players dahil), sağ altta Stop/Reset/Play, F2/F6/Esc çalışır. Boyut/yerleşim eskisiyle piksel piksel aynı olmalı; kaymışsa `S()` bir yerde yanlış. En sinsi hata: yerleşim makul ama 1–2 px kaymış, kimse bildirmez.
3. **Kütüphane:** kapak kartları, "Add folder" (COM) düğmesi, kart tıklama = seçim + "Load selected scene" (Windows'ta eskisi gibi, kart tek başına yüklemez).
4. **Dil:** kayıtlı `locale` yoksa Windows yine `en` ile açılır (Android dışında davranış değişmedi). Türkçe seçip yeniden başlat: korunmalı.
5. **IPC:** `player.operator {operation:"console.open", config:{page:"settings"}}` menüyü o sayfada açar, `console.status` `{open,page}` döner, `console.close` kapatır.
6. **Android (isteğe bağlı):** `python scripts/stage_player_android_assets.py --scenes pastel-paint-workshop,kids-play-room`, APK kur: açılışta kütüphane, karta dokun → oyun, Geri/menü düğmesi. `adb install -r` sonrası eski `player_scenes.json` kalıyorsa (bkz. PLAYER_ANDROID_PORT.md) önce `adb uninstall`.

## Android Vulkan viewport hata ekranı

**Durum.** Emülatörde zorlanan arm64 hatası ve yerel x86_64 normal açılışı geçti. ADB'siz telefon testi için E:/rt_android_build/RayTrophiPlayer-arm64-s24.apk kopyalandı. SHA-256: `56452A57FEC657E36554ED170BA2CAA2A9EE41544F2338099E3E18D04AC2AE35`.

1. **Emülatör, yapay hata.** Yeni APK kurulu iken aşağıdaki komutları çalıştır. Ortadaki kutu `vkCreateCommandPool`, `VkResult -3`, seçilen GPU ve driver version göstermeli. `Son hatalar ve uyarılar` başlığına dokun ve en yeni ERROR satırını gör. IPC `agent.get_state_summary` ile `runtime.last_open_error` ve `runtime.startup_log` aynı tanıyı taşımalı. GPU alanı boşsa tanılama adımı cihaz seçiminden önce çalışıyor ya da kayıt zinciri koptu.
2. **Normal açılış.** Özelliği `0` yap, uygulamayı durdurup yeniden başlat. Gizli Renkler'in ilk karesi gelmeli. **En sinsi hata** eski tanı kaydının yeni denemeye sızıp normal açılışı engellemesi veya seçilmiş GPU'daki hatanın farklı bir adaptörün yetenek reddi olarak gösterilmesi.
3. **ADB'siz telefon.** Yeni arm64 APK'yı kur ve normal aç. Ret kutusu çıkarsa log başlığını açıp ekran görüntüsü al. Adım, VkResult (Vulkan çağrısından dönmüşse), GPU ve driver satırlarını birlikte gönder. Yalnız genel başlık görülürse ekranın hata gövdesinin de göründüğünü ve kurulan APK'nın sürümünü kontrol et.

```powershell
adb shell setprop debug.raytrophi.fail_viewport_startup 1
adb shell am force-stop com.raytrophi.player
adb shell am start -n com.raytrophi.player/.PlayerActivity

adb shell setprop debug.raytrophi.fail_viewport_startup 0
adb shell am force-stop com.raytrophi.player
adb shell am start -n com.raytrophi.player/.PlayerActivity
```

## Player Android — Adreno float atomik

1. **Yeni APK ve log.** Değişen C++ birimlerinin Studio/Player /Zs ve NDK x86_64/arm64-v8a -fsyntax-only denetimleri geçti. Yeni x86_64 APK'yı al ve host GPU emülatörüne kur. SDK platform-tools PATH'te olsun. İlk komut grubuyla tanılamayı açıp Player'ı yeniden başlat. `SceneLog.txt` içinde `Float atomic add disabled by diagnostic override` ve `module.AtomicFloat32AddEXT_not_supported_on_this_GPU` satırları beklenir. Örnek kernel `sim_matter_p2g`, shader yolu ve VkResult `-8` aynı satırda olmalı. Satır yoksa yeni APK'yı, adb hedefini ve özelliğin açılış öncesinde ayarlandığını kontrol et.
2. **Görsel kabul.** Gizli Renkler açılınca bir balon patlat. Damlaların duvara temas ettiği yerde boya görünmeli. İstersen mevcut dış IPC istemcisinden `paint.wet_demo status` ve `paint.wet_status` ile olay ve etkin tile artışını da doğrula. **En sinsi hata** sahne açık ve damlalar görünürken duvarda boya veya boyayla açığa çıkan yüzey izinin olmaması. Bu durumda spawn/ballistic/collide ve paint_wet pipeline loglarını, çarpışma olay yayımını ve paint dispatch durumunu kontrol et.
3. **Anahtarsız karşılaştırma ve gerçek cihaz.** İkinci komut grubuyla özelliği sıfırla ve aynı emülatör oyunu tekrarla. NVIDIA destekliyorsa tanılama satırı ve atomik capability retleri kalkmalı. Ardından yeni arm64-v8a APK'yı S24 Ultra'da anahtar kapalıyken dene. Yerel özellik yoksa capability retleri, açık sahne ve duvarda boya bir arada görünmeli. Adreno sürücüsünün tekrar `OpAtomicFAddEXT` demesi başka bir shader kurulum yolunun kaldığını gösterir.

```powershell
adb shell setprop debug.raytrophi.force_no_atomic_float 1
adb shell am force-stop com.raytrophi.player
adb shell am start -n com.raytrophi.player/.PlayerActivity
adb shell run-as com.raytrophi.player cat files/SceneLog.txt

adb shell setprop debug.raytrophi.force_no_atomic_float 0
adb shell am force-stop com.raytrophi.player
adb shell am start -n com.raytrophi.player/.PlayerActivity
```

## Player Android A6a ve A6b

1. **Derle ve paketle.** Studio ve Player Release x64 için temiz derleme al. Ortak C++ ölçüm yapılarının boyutu değişti. Android için standart Player paketi üzerinden varlıkları tekrar hazırla ve x86_64/arm64-v8a APK'ları al. Yeni başlık görünmüyorsa paket içindeki `assets/i18n/en.json` ve `tr.json` sürümlerini kontrol et.
2. **Ölçüm birimi.** Mümkünse `timestampPeriod != 1.0` raporlayan Vulkan cihazı kullan. Capture/debugger üzerinden seçili dispatch ve raster sınırının sorgu tick farkını al. Sayaç genişliğine göre farkı maskele. Beklenen ms `delta_ticks × timestampPeriod / 1_000_000`. Örnek olarak 2.000.000 tick ve 0,5 ns/tick tam 1 ms eder. **En sinsi hata** 2 ms gibi makul görünen ama period çarpanı atlanmış bir değerdir.
3. **Windows Player.** Sahnenin yüklenmesini bekle ve Player başlangıç logunda IPC sunucusunun açıldığını doğrula. Aşağıdaki ilk komutu ayrı PowerShell sürecinde çalıştır. `Idle` sırasında giriş yapma, `Burst` başlarken birkaç balon patlat. Dört compute satırı ve raster ana geçişi için ilgili aşamada `Available=True` ve pozitif örnek sayısı beklenir. Süre sütunları boşsa destek bayrağını, sahnenin ilgili GPU işini çalıştırdığını ve Player odağını kontrol et.
4. **Android emülatör.** Yeni x86_64 APK'yı `-gpu host` ile çalıştır. Ekranı uyanık tut. İkinci komut adb TCP yönlendirmesini oluşturup kendi çıkışında kaldırır. Birden çok cihaz varsa `-Serial <adb-seri>` ekle. Windows ile aynı sütunlar dolmalı. Duraklatılmış SDL döngüsü IPC zaman aşımı olarak görünür.
5. **Ret ekranı.** Desteklenmeyen arm64 cihazda HUD kapalıyken başlangıç ret başlığını ve gövdedeki GPU adı ile Vulkan gereksinimini kontrol et. Android dillerini Türkçe ve İngilizce olarak ayrı açılışlarda dene. Descriptor indexing şartını karşılayıp 15–16 SSBO raporlayan cihaz `paint_wet (17)`, 15'in altı iki kernel gerekçesini göstermeli. Ret edilen aygıtta projenin açılması kabul hatasıdır.
6. **Boş veri ve API eşliği.** Timestamp desteği olmayan kuyrukta `Supported=False`, `Available=False` ve boş ms sütunları beklenir. Duran sahnede hiç çalışmayan kernelin `Available=False` olması geçerlidir. Studio Python'da `rt.perf.gpu_kernel_timings()` ve `rt.viewport.frame_timings()` ile aynı ek alanları gör. `wet_paint_fallback=True` ayrı boya bağlamını seçer.

```powershell
.\scripts	est\Measure-PlayerGpuPasses.ps1 -IdleSeconds 5 -BurstSeconds 8
.\scripts	est\Measure-PlayerGpuPasses.ps1 -Android -IdleSeconds 5 -BurstSeconds 8
```

Raporda ortalama, tepe ve nearest-rank p95 ayrı GPU örneklerinden hesaplanır. Raster örnekleri GPU'dan birkaç kare geriden okunabilir. Compute tanılama açıkken asenkron gönderimler senkronlaşır. Bu çalıştırmayı zamanlayıcı kapalı FPS ölçümünden ayrı kaydet.

## Player Android A5 — dokunmatik giriş (2026-11-22)

**Emülatör başlangıç ölçümü.** Mevcut APK, `-gpu host`, yatay ekran 1600×720. Kullanıcı fareyle taşıma ve patlatmayı doğruladı. Boşta yaklaşık 56 FPS, çoklu patlamada yaklaşık 21 FPS, ardından tekrar yaklaşık 56 FPS. İç render boyutu doğrulanmadı. Tekerlekle derinlik ve yeni APK'da iki parmakla sıkıştırma kabulü açık.

**Windows Player link düzeltmesi — PASS (kullanıcı, 2026-11-22).** Ozz `platform.cc`, eski RayTrophi `Platform.cpp` ile aynı `platform.obj` yolunu kullanıyordu. Ozz artık `OzzBasePlatform.obj` üretiyor. Visual Studio'da değişen projeyi yeniden yükle ve x64 Release Player derlemesini tekrar al. Linkte `OzzBasePlatform.obj` ile `PlatformOs.obj` görünmeli. `platform.obj` hâlâ link girdisiyse Player için Clean + Rebuild uygula.

1. **APK.** Repo kökünde varlıkları hazırla ve x86_64 APK'yı derle. Kurulumdan sonra `Medium_Phone_API_37.0` (`-gpu host`) üzerinde Gizli Renkler açılmalı. Sahne açılmazsa önce varlık hazırlama çıktısını ve Player başlangıç logunu kontrol et. Android HUD, sahnesiz durumda da proje bırakma metnini göstermemeli.

   ```powershell
   python scripts/stage_player_android_assets.py
   $env:JAVA_HOME = 'C:\Program Files\Java\jdk-17'
   Push-Location android
   .\gradlew.bat assembleDebug --no-daemon -PrtAbis=x86_64 -PrtSdl2Dir=E:/SDL2-2.30.4/SDL2-2.30.11 -PrtSdl2ImageDir=E:/SDL2-2.30.4/SDL2_image-2.8.2_src
   Pop-Location
   adb -e install -r android/app/build/outputs/apk/debug/app-debug.apk
   adb -e forward tcp:47391 localabstract:RayTrophiPlayer
   Import-Module ./scripts/ipc/RtIpc.psm1 -Force
   Connect-RtIpc -Port 47391
   ```

2. **Tut, sürükle, bırak.** Görünür bir balonun ortasından başlayan `adb -e shell input swipe X1 Y1 X2 Y2 1500` komutunu ayrı terminalde çalıştır. Swipe öncesinde, sürerken ve sonrasında `(Invoke-RtIpc paint.wet_demo @{operation='status'}).interaction` oku. Beklenen sıra `held=-1 → balon ID'si → -1`, sürüklerken `owner=finger:<id>` ve görünür konum değişimi. Yatay ekranda koordinatı o anki kareden al. Eski APK'da 1725,540 noktası bir balonu tuttu.
   **En sinsi hata:** Fare sentezi açık kalırsa tek dokunuş iki olay dizisi üretir. Balon tutulup hemen bırakılmış gibi görünebilir veya sahip `mouse` olur. Swipe sürerken ölç. Tap sonrasındaki `held=-1` tek başına arızayı göstermez.
3. **Sıkıştırma ve boya.** Emülatörün çoklu dokunma girişinde ilk parmakla tut, ikinciyle aralığı daralt. Aynı balon ve sahip korunurken `interaction.depth` artmalı. İkinci parmak kalkınca tutuş sürmeli. Ardından balonu duvara bastırıp patlat. Patlama sayacının artması ve duvarda görünür boya beklenir. Patlama olup boya görünmezse A4 boya hattındaki `active_tiles` ve `device_event_dispatches` değerlerine bak.
4. **Windows fare regresyonu.** Kullanıcı demoda fareyle balon yakalamayı doğruladı. Aşağıdaki ayrıntılı kontroller ayrıca raporlanmadı. Üç değişen birim Player x64 `/Zs` PASS. Önceki tam derleme gereksinimleri tamamlandıysa A5 için artımlı Player derlemesi yeterli. Sol tuşla tut ve sürükle, tekerlekle derinliği değiştir, sol tuşu bırak. Tutarken IPC sahibi `mouse` olmalı. Shift ile bırakma ve odak kaybı mevcut tutuşu iptal etmeli. Sahnesiz ve hatasız açılışta proje bırakma metni görünmeli.

## ★ ÖNCE OKU — bu Windows derlemesi TAM derleme olmalı (Android A1–A3 + NDK düzeltmeleri, 2026-11-22)

Yaygın başlıklar değişti: `GpuInteropTypes.h` (yalnız clang dalı), `globals.h` (`gamma` silindi), `JoltWorld.h`, `PaintLayerStack.h`,
`PlayerVideoCard.h`, `Platform/Platform.h`, `GasSimulator.h`, `VDBVolumeManager.h`. Artımlı derleme eski `.obj` bırakırsa erişim
ihlali kod hatası sanılır → **Rebuild/temiz derleme**, Studio + Player. Visual Studio "proje dışarıda değişti" derse yeniden yükle.

W1. **Link (önce bu).** ✔ Studio temiz derleme linklendi (2026-11-22, kullanıcı). Player Release x64 linki de Ozz düzeltmesinden sonra kullanıcı tarafından doğrulandı. Bozuksa: `g_*` tanımsız → `using ::g_x;` değişikliği (VulkanBackend*,
    VulkanViewportBackend, TerrainNodesV2, AnimationNodes) global bildirimi göremiyor; `gamma` tanımsız → silinen global'i
    kullanan bir birim kaldı (grep'te bulunmadı); `BCryptGenRandom` → Platform.cpp pragma.
W2. **Terrain node gamma / animasyon klipleri / viewport raster** her biri bir kez açılsın — `using ::g_x` değişen yollar bunlar.
    Sessizce makul görünen hata: yanlış global'e bağlanma (Windows'ta beklenmez, MSVC zaten global'e bağlıyordu).
W3. **Boya katmanı tohumu.** Paint Edit'te doku setinden katman tohumla (`seedFromTextureSet` varsayılan argümanı aşırı yüklemeye
    döndü). Katman boş gelirse aşırı yükleme varsayılan `PaintUVParams`'ı geçmiyor.
W5. **(Sonradan eklendi, artımlı derleme yeter)** `Platform.cpp` → `PlatformOs.cpp` (LNK4042 kaybolmalı), `RtIpc.cpp` Windows-only
    korumasız (önişlemci çıktısı Windows'ta aynı), `RayFusionViewportPasses.h` +3 fonksiyon (7 birim yeniden derlenir), `PlayerMain`
    başlangıç hata bildirimi, `AppWindow` Android dalları. Windows'ta görmen gereken: Player normal açılır, HUD aynı. Sahnesiz açılışta
    "Drop an .rtp project here" görünür. Hatalı bir .rtp sürüklenirse artık ORTADA "could not open the scene" kutusu çıkar (yeni davranış).
W4. Not (önceden var olan): `VulkanBackend.h` `<sstream>` include etmiyor; vcxproj'da başka bir başlık maskeliyor olabilir.

## Player Android A2c — OpenVDB (2026-11-22)

1. **En sinsi hata — Windows'ta sessizce boş VDB hacmi.** Studio'da önceki sürümde görünür olduğu bilinen bir `.vdb` aç. Hacmin görünmesini ve aynı sahnede yoğunluğun önceki sürümle eşleşmesini bekle. İçe alma tamamlanmış görünüyorsa ama hacim boşsa önce Windows Studio/Player derleme tanımlarında `RT_NO_OPENVDB` bulunmadığını ve VDB'nin gerçek yöneticiye ulaştığını doğrula.
2. **Android NDK linki.** SDL2 ve SDL2_image kaynaklarını vererek arm64-v8a ve x86_64 hedeflerini derle. OpenVDB, TBB, Blosc veya Imath ön eki gerekmemeli. VDB yönetici ya da gaz sembolü çözümlenmezse önce üretilen kaynak listesindeki iki Android stub kaydını kontrol et.
3. **Android davranışı.** A3 yerel IPC üzerinden `scene.import_vdb` ve sparse VDB backend seçimini dene. İkisi de `OpenVDB not available on Android` hatası vermeli ve sahneyi değiştirmemeli. Paketli balon sahnesi normal açılmalı. Dışa aktarma denemesi dosya yazarsa, zaman çizelgesini değiştirirse veya boş hacim oluşturursa koruma yanlış noktadadır.

## Player Android A3 — IPC güvenliği OpenSSL'siz (2026-11-22, Windows'u da etkiler)

1. **Mevcut token hâlâ geçiyor mu (önce bu, hızlı).** `RtIpcSecurity` SHA-256'yı artık OpenSSL yerine ağaç içi `Utils/Sha256.h` ile, rastgele baytları `BCryptGenRandom` ile üretiyor. Daha önce verilmiş bir uzak token ile (`Start-RayTrophi.ps1 -Remote`, kayıtlı `ipc_tokens.json`) bağlan. Görmen gereken: kimlik doğrulama geçer. Bozuksa: özet uyuşmazlığı → SHA-256 uygulaması (test vektörleri 5/5 PASS ölçüldü, yani beklenmez) ya da `hex` biçimi.
2. **Yerel pipe davranışı aynı.** `Invoke-RtIpc agent.discover` Studio ve Player'da cevap verir; `connection_id` 32 hex karakter (boş değil). Boşsa: `secureRandomBytes` başarısız → bcrypt link/pragma.
3. **Linkleme.** Studio/Player hatasız linklenir (`bcrypt.lib` `Platform.cpp` içinde pragma ile). `BCryptGenRandom` çözümlenmemiş sembol → pragma derleyiciye ulaşmamış.
4. **Android (NDK sonrası).** `adb forward tcp:47391 localabstract:RayTrophiPlayer`, `Connect-RtIpc -Port 47391`, `Invoke-RtIpc scene.list_objects`. ★ En sinsi: IPC'nin Android'de HİÇ açılmaması — TLS stub'ı önceden her durumda `false` dönüyordu ve `RtIpc.cpp` buna yerel taşımayı da kapatarak cevap veriyor; düzeltildi (uzak IPC istenmediyse `true`). Log'da `IPC server started` yoksa önce buna bak. Bağlantı reddediliyorsa `SO_PEERCRED` uid'i (adb shell = 2000) logla.

## Player Android A2b — SIMD (2026-11-22)

1. **En sinsi hata — x86 yolunun sessizce yavaşlaması.** Windows Studio'da aynı exe yapılandırması, sahne, kamera, örnek sayısı ve ısınma süresiyle önceki sürüm ile CPU render ve AABB/bounding-box sürelerini karşılaştır. Aynı sahne sonucunu ve benzer süreleri bekle. Çıktı aynıyken süre artıyorsa `/arch:AVX2`, include sırası ve seçilen SIMD yolunu incele. Android x86_64 emülatöründe `-mavx2 -mfma` ile ayrı bir açılış kontrolü yap.
2. **AArch64 sözdizimi ve kabul.** NDK kurulduğunda önce arm64-v8a hedefinde `Vec3SIMD.cpp` ile `Matrix4x4.h` kullanan `AABB.cpp` birimlerini Android API 29 için kontrol et. Intrinsic, hedef başlığı veya istisna modu hatası beklenmez. Android araç zinciriyle kullanıcı derlemesinden sonra gerçek arm64 cihazda paketli sahneyi aç ve önceki kabuldeki görünür varlıkları karşılaştır.

0. **Player linkleniyor mu (önce bu).** Player vcxproj yeniden üretildi; `UI/PlayerOperator*UI.cpp` (6 dosya) A1 üretiminde düşmüştü, manifest düzeltmesiyle geri geldi. Visual Studio "proje dışarıda değişti" derse yeniden yükle. Görmen gereken: Player hatasız linklenir, F2 operatör menüsü açılır. Bozuksa: `drawPlayerOperator`/`releasePlayerOperatorUI` çözümlenmemiş sembol → vcxproj yeniden yüklenmemiş ya da manifest satırı kaybolmuş (`python scripts/gen_player_vcxproj.py --check`).
1. **Player açılış logu.** Paketli Player'ı paketin dışındaki bir çalışma dizininden başlat. `[Player Startup]` satırındaki çalışma dizini exe'nin klasörü, paket shader dizini eski Windows paketindeki aynı `shaders/` dizini olmalı. İlk sahne karesi shader eksikliği hatası olmadan gelmeli. Farklı kök çalışma dizini yönlendirmesine, doğru kökte eksik shader dosyası paket içeriğine işaret eder.
2. **En sinsi hata — AssetRegistry sessizce boş sahne.** Aynı paket ve `.rtp` ile önceki kabulde varlık kullandığı bilinen bir sahneyi aç. Beklenen arşiv kökünü, görünen model/doku/önizlemeleri ve sahne nesne listesini önceki sürümle karşılaştır. Açılış başarılı görünürken varlıkların kaybolması, `assets/` kökü veya kayıt çözümlemesi hatasıdır. Workshop balon sahnesinde duvarda ıslak boyayı da gör.
3. **UTF-8 ve yan dosyalar.** Studio ve Player'da boşluk ve Türkçe karakter bulunan bir yoldan aynı `.rtp` dosyasını aç. Varsa `.rtp.bin` ve harici kaynakları okuyarak ilk kareyi üretmeli. Uzun ya da sistem ANSI kod sayfası dışındaki bir exe yolunu ayrıca dene. Yalnız bu durumda açılış farklıysa geniş karakterli exe veya dosya yolu tüketicisini incele.
4. **Windows yükleyicileri ve tanılama.** İki hedefte Vulkan başlat. Paketli isteğe bağlı FidelityFX ile FSR'yi aç, CUDA bulunan Studio'da Terrain FFT'yi çalıştır ve performans profilindeki çalışma kümesinin ölçüldüğünü doğrula. Vulkan başlatılamazsa sürücü DLL yolu ve logu, FSR/FFT düşerse isteğe bağlı DLL varlığı ve sembol sonuçları kontrol edilmeli.
5. **Yerel IPC ve ajan başlatma.** Açık uygulamanın pipe'ına ayrı terminalden `scripts/test/rt_ipc.py` ile ilgili Studio/Player pipe adını kullanarak `agent.discover` çağır. Önceki Windows yerel erişim davranışı korunmalı. Kurulu RayTrophiAgent yapılandırmasıyla Windows'ta ajanı başlat ve durdur. Android'de IPC ve cihaz hazır olduğunda alt süreç başlatma açık bir `not supported on this platform` hatası vermeli.

Android cihaz kabulü A2/A4 ve SDK/NDK sonrasında. Paket varlıkları SDL iç depolama köküne çıkarıldığında Player'ın aynı `shaders/`, `scenes/` ve `assets/` yerleşiminden okuduğunu doğrula.

---

# Sıradaki derleme kontrolleri — Player eski GPU çökmesi (2026-11-20)

> **Durum:** CANLI — derleme bekliyor (Player, GTX 850M / i7-4700 laptop)

Pipeline trace artik varsayilan KAPALI: normal acilista dosya acma/yazma yok.
Yalniz process baslatilmadan once `RT_SIM_PIPELINE_TRACE=1` verilirse etkin.
Env bir kez okunur. Bu degisiklik icin kullanici Player artimli build almali.
Normal acilista mevcut SimulationPipelineTrace.txt boyutu/tarihi degismemeli;
env=1 ile module/pipeline satirlari append edilmeli. Codex build calistirmadi.

SON DURUM: kullanici shader deneylerinin geri alinmasini istedi. Iki derleme
scripti eski ayarlarinda; forces SPV kaynak/runtime/dist orijinal SPV1.6/27620 byte,
SHA256 2225C782CD67556B8828F1E36452BC25BD3B58426254558CE2F4009925A98158.
Asagidaki Vulkan1.2/-O0 dagitim talimatlari gecersiz; yalniz tani gecmisi.
Hem SPV1.5/-O hem SPV1.5/-O0 ayni pipeline.begin'de kesildi. Sebep kesinlesmedi.

## Shader adini kesinlestirme (Codex, 2026-11-20)

GUNCEL ikinci test: laptop SPV1.5/27676 byte optimizasyonlu forces shader'ini
yukledi ama ayni pipeline.begin'de tekrar coktu. Hedef surumunu dusurmek yetmedi.
Simdi yalniz forces Vulkan1.2/-O0 ile derlendi: SPV1.5, 37944 byte, validation PASS.
Source/runtime/dist kopyalari yenilendi; iki bulk script bu ayari korur.
Laptopta shaders/sim_fluid_particle_forces.spv degistir, Player'i yeniden ac;
trace'te 37944 byte ve pipeline.end kontrol et. Kok sebep henuz kesin degil.

Trace sonucu: iki deneme de `sim_fluid_particle_integrate_forces` /
`shaders/sim_fluid_particle_forces.spv` pipeline.begin sonrasi kesildi.
Yalniz bu shader Vulkan1.2/-O ile SPIR-V1.5 olarak yeniden derlendi ve dogrulandi;
source/runtime/dist kopyalari hash esit. C++ rebuild bu shader testi icin gerekmez.
Laptopta ayni isimli shaders dosyasini degistirip Player'i tamamen yeniden ac.
Yeni trace'te ilgili pipeline.end result=0 beklenir; sonraki hata varsa tum trace'i
topla. Bu henuz cihazda test edilmedi ve kok sebep kaniti degildir. ZIP eski kaldi.

- Player artimli C++ build kullanici tarafindan alinmali; shader degismedi.
- `SimulationPipelineTrace.h` yalniz startup'ta module/pipeline cagrilarinin
  hemen oncesini ve sonrasini `SimulationPipelineTrace.txt` dosyasina append eder.
  Her satir driver'a girmeden once dosya kapatilarak yazilir. Dosya process'in
  calisma dizinindedir; yazma izni yoksa trace olusmaz, acilis devam eder.
- Laptopta sahneyi bir kez ac; cokerse trace + SceneLog + crash stack'i topla.
  Son `pipeline.begin` icin karsilik gelen `pipeline.end` yoksa driver o shader'i
  olustururken donmedi. Kayit kernel adi, gercek SPV yolu, SPIR-V surumu ve
  byte boyutunu icerir. `module.begin` / `module.end` ayni ayrimi modul icin yapar.
- Son shader kesinlesince dagitilan SPV'nin capabilities/extensions listesini
  GPU'da acik ozelliklerle karsilastir; sonra yalniz ilgili shader/fallback'i duzelt.
  Kernel adi tek basina donanim uyumsuzlugunun kesin sebebini kanitlamaz.
- Derleme veya uygulama Codex tarafindan calistirilmadi; kaynak diff kontrolu yapildi.

Kök bulgu: `StartupCrashStack.txt` (PlayerMain'e eklenen yakalayıcı) + PDB ile çözüldü.
Çökme `vkCreateComputePipelines` içinde, NVIDIA `nvoglv64.dll`; zincir
`deserializeParticleSimulation → createParticleRuntime → syncSimulationWorld →
VulkanSimulationComputeBackend::loadPipelines → createPipeline`
(`SimulationComputeVulkan.cpp`). Hipotez: float atomic / float64 özelliği açık
olmayan cihazda, o yeteneği bildiren SPIR-V derleniyordu (UB → sürücü çöker).

Değişenler: PlayerMain.cpp (pencere erken gösterilir + çökme yakalayıcı),
ProjectManager.cpp (`load step:` logları), SimulationComputeVulkan.cpp
(`kernelNeedsMissingFeature` hem dispatch'te hem pipeline oluşturmadan önce).

Sırayla:
1. **Studio/dev makinede açılış** (hızlı, bağımsız). Float atomic ve float64
   destekleyen GPU'da hiçbir kernel atlanmamalı: sıvı/kar sahnesi eskisi gibi
   çalışır. Bozuksa: kapı yanlışlıkla atlıyor demektir (`m_has_float_atomics`
   / `m_has_shader_float64` ctor'a yanlış geçiyor).
2. **Laptopta Player.** Pencere açılıp sahne gelmeli (render siyah olabilir).
   ★ Sinsi durum: açılıyor ama parçacık/sıvı hiç çalışmıyor — sessizce makul
   görünür. SceneLog'da atlanan kernel sayısı yok; gerekirse log eklenmeli.
3. Hâlâ çökerse yeni `StartupCrashStack.txt` + `SceneLog.txt`'deki son
   `load step:` satırı. Aynı `createPipeline` çıkarsa hipotez yanlış: kernel adı
   için pipeline döngüsüne geçici log koy ve sürücüyü güncelle (850M için 4xx
   değil, mevcut son Maxwell sürücüsü).
4. Fullscreen'den çıkılamıyor raporu: F11 ile `SDL_SetWindowFullscreen`
   (PlayerMain.cpp:432). Pencere çökmeden önce takılı kaldıysa ayrı değerlendir.
