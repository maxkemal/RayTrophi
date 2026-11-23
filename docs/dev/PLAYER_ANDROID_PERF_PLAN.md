# Player Android performansı — kod optimizasyon raporu (Mali-G52 MC2)

> **Durum:** AKTİF — 2026-11-23 (Claude): Lenovo TB330FU üzerindeki iki ajan ölçümünün koddaki karşılığı ve sıralı optimizasyon planı. Kod değişikliği YOK; bu not plan. Kural: kalite düşürülmez, ikinci render yolu açılmaz. Görüntüyü değiştirmeden aynı işi daha ucuza yap.

## 1. Ölçümler (iki rapor, aynı tablet)

| Ölçüm | Değer | Kaynak |
|---|---|---|
| 1920×1200 kare GPU (Material/RayFusion, HDRI) | **≈3,8–4,0 s** | Claude 2026-11-23 00:05, Codex Release |
| Çözünürlükle ölçek | 0.5 → 1,5 s, 0.25 → 0,37 s. **Piksele doğrusal**, yani fragment maliyeti | Claude |
| 960×600 temel / düz duvar (ekranda 1 balon) | 1465 ms / **893 ms** | Claude |
| 960×600 / 480×300 (Release) | 1055 ms / 339 ms | Codex |
| HDRI → düz renk → HDRI (tam hedef) | 4018 / **3204** / 3853 ms, yani HDRI ≈ **%20** | Codex |
| Aşama payı (1920×1200) | "sky" 2,6 s + "transmission" 1,1 s | Claude |
| Açılış (Release) | 63,8 s; `game_design` **38,9 s**, `prepare_particles` 22,1 s | Codex |

Sahne: 6 ışık, 6'sı gölgeli, preset `scene`, kalite `auto`. Kamera düz duvara bakarken bile 893 ms ölçülüyor. **Maliyetin kaynağı içerik değil**, her pikselde çalışan temel aydınlatma yolu.

## 2. Koddaki karşılıklar (bulgular)

### B1. "sky 2,6 s" gökyüzü değil, ANA GEÇİŞİN süresi (ölçü aleti yanılıyor)
`VulkanViewportBackend.cpp:4093`: `Sky` damgası `vkCmdBeginRenderPass` içinde, ana çizim de aynı render pass içinde (`MainPass` damgası `:4396`). Mali gibi TBDR GPU'lar render pass'ı tile tile toplu işler. Pass içindeki ara timestamp'ler işi bölmez: bütün pass ilk damgaya yazılır. `material_preview_sky.frag` 167 satır ve tek doku okumalı, o yüzden 2,6 s olamaz. **Doğru okuma şu: "sky + main_pass" ≈ 2,6 s ana geçiş, "transmission" ≈ 1,1 s balonlar.** Tabloda bu düzeltilmezse bir sonraki ölçüm turu yine gökyüzünü kovalar. Bu tam olarak "panelin yalan söylemesi" sınıfından bir hata.

### B2. Ana fragment shader'ı devasa bir uber-shader: 651 KB SPIR-V
`x64/Release/shaders/material_preview_frag.spv` = **651 244 bayt**. Gökyüzü shader'ı 10 KB. `material_preview_frag.frag` 1753 satır, `bsdf_scatter.glsl` (2175), `material_program.glsl` (823), hava/şimşek/prob/SSS/clearcoat/sheen/transmission/yansıma G-buffer include'ları ile birlikte derleniyor. Mobil derleyici register'ı **en kötü dala göre** ayırır. Sahnede clearcoat, sheen, SSS, şimşek yoksa da bu dalların register maliyeti her pikselde ödenir. Mali Bifrost'ta register sayısı yükselince eşzamanlı thread sayısı yarıya iner, sınırı aşınca **stack spill** başlar (her erişim belleğe gider).
Kaba hesap: 893 ms × ~2 çekirdek × ~0,85 GHz / 576 000 px ≈ **piksel başına ~2600 çekirdek çevrimi**. 6 ışık × 9 gölge tapı + ~50 doku okuması bu kadar etmez. Bu sayı spill ya da occupancy çöküşünü düşündürüyor. **Kesin cevabı Mali Offline Compiler (`malioc`) verir ve tablet gerektirmez** (O2).
Not: daha önce elenen şey yalnız 32 dallı doku switch'iydi (2 dallı varyant, fark yok). Özellik dalları hiç ölçülmedi.

### B3. Işık başına gölge: 6 × 3×3 PCF, kalite düğmesi bunu düşürmüyor
`material_preview_shadow_data.glsl:36–77`: Mali'de ray-query yok, bu yüzden her ışık atlas yolundan gidiyor. Işık başına `(2r+1)²` tap var: kalite < 3 iken r=1 (9 tap), kalite 3'te r=2 (25 tap). Hacim gölgesi açıksa (`deepMeta.z`) her ışık `rtDeepShadow` ile 4 × `rtDeepTexel` yapıyor, yani **12 SSBO okuması** daha. Kalite `auto`, Performance'ta bile tap sayısı 9'un altına inmiyor. `useFullPbr` (`frag:1215`) `scene` preset'inde her zaman true, yani kalite düğmesi BRDF'yi de değiştirmiyor.
Sonuç: 6 ışık × 9 = **54 karşılaştırmalı okuma**, hacim gölgesi açıksa +72 SSBO okuması. Bunlar her opak pikselde.

### B4. HDRI dokusu RGBA32F ve mip'siz
`VulkanSceneSync.cpp:46`: HDR ortam haritası `VK_FORMAT_R32G32B32A32_SFLOAT` olarak, `uploadTexture2D(...,false,true)` ile mip'siz yükleniyor. Texel 16 bayt. Mobilde 32-bit float süzme yarım/çeyrek hızda çalışır ya da hiç desteklenmez (destek yoksa linear sampler **geçersiz kullanım**, sonuç tanımsız). Mip olmadığından uzak/geniş konili okumalar önbelleği boşa harcar. Ölçülen etki: HDRI kapatılınca **−814 ms (%20)**. Bu maliyet gökyüzü pikselinden gelemez, çünkü gökyüzü pikseli tek okuma yapıyor. Asıl tüketici B5.

### B5. Transmission (balonlar) uber-shader'ı İKİNCİ kez çalıştırıyor, üstüne ham HDRI'yi 27 kez okuyor
`MaterialPreviewTransmission.cpp:348`: balonlar aynı `material_preview_frag`'ı faz 2 ile yeniden çiziyor, yani B2 + B3'ün bedeli balon piksellerinde ikinci kez ödeniyor. Balon pikseli başına ek iş:
- `sampleDispersedTransmission` (`frag:517`): 3 kanalın her biri `sampleTransmissionEnvironment` çağırıyor, o da roughness > 0 ise (1 + 4..8) koni tapı yapıyor. Toplam **15–27 okuma**, hepsi ham RGBA32F equirect HDRI'den (B4).
- `sampleOpaqueSceneRough` (`frag:588`), kırılma için 3 kanal × 5–9 tap.
- `tracePreviewOpaqueReflection` (`frag:552`): 8–16 adım derinlik yürüyüşü, ardından 5–9 tap.

Ortam için önceden süzülmüş harita zaten var ve ambient onu kullanıyor (`frag:1290` civarı, "Prefiltered ambient is now the path for BOTH producers"). O değişiklik masaüstünde 102 ms'lik karenin 60 ms'ini geri vermişti. **Transmission bu yola hiç taşınmamış.**

### B6. Açılıştaki `game_design` 39 s büyük olasılıkla KARE SÜRESİ × kare sayısı
`RuntimeSimulationClock.cpp:43` → `Paint::tickGameDesign` her kare bir kez çalışıyor ve `GameDesign.cpp` collider SDF'yi `kDemoWaitTicks` (60×60) tik boyunca bekliyor. Yükleme ekranının arkasında 3D sahne tam maliyetle çiziliyorsa her tik ≈3,8 s sürer. Bu durumda 39 s ≈ **10 kare** demektir. Render düzelince açılış da kendiliğinden kısalır. Ayrıca kart ekranı kapatırken sahneyi çizmek tamamen boşa iş.

### B7. Ana geçiş 4 renk eki yazıyor (doğrulanacak)
`frag:32–34`: HDR renk + `outRfNormal` + `outRfSpecular` + `outPreviousClip`. Mali tile belleği piksel başına sınırlı (G52'de tam tile boyutu için ~128 bit). Fazlası tile'ı küçültür ve dış bellek trafiğini artırır. Yansıma/TAA Mali'de çalışmıyorsa bu ekler boşa yazılıyor olabilir. Formatlar kontrol edilmeli: 16F×4 + 16F×2 + 16F×4 + ? zaten 128 biti aşar.

## 3. Optimizasyon planı (sıra: kazanç / risk, görüntü değişmeden)

| # | İş | Görüntü | Beklenen | Tablet gerekir mi |
|---|---|---|---|---|
| **O1** | Ölçü aletini düzelt: TBDR'de (ARM/Qualcomm/IMG vendor ID) pass içi aşamalar `main_render_pass` olarak birleşik raporlanır, ya da `sky` damgası pass dışına alınır. IPC alanı adı değişir (`tbdr_merged: true`) | aynı | 0 ms, ama sonraki her ölçüm doğru okunur | hayır |
| **O2** | `malioc` ile statik analiz: `material_preview_frag.spv` için Mali-G52'de work register sayısı, **stack spilling**, çevrim (arith/LS/tex). Varyantlar (özellikler kapalı) karşılaştırılır | — | B2'yi kanıtlar ya da eler | **hayır, bugün yapılabilir** (Arm Performance Studio indirmesi gerekir) |
| **O3** | Uber-shader'ı **specialization constant** ile buda: sahnede olmayan özellikler (clearcoat, sheen, SSS, iridescence, şimşek, hava yüzeyi, hacim gölgesi, ekran gölgesi, terrain katmanları, transmission fazı, kalite kademesi) derlemede sabit false olur. Tek kaynak, tek yol. Pipeline sahne özellik maskesine göre bir kez kurulur | **bit-aynı** (kapalı dal zaten çalışmıyor) | O2 spill gösterirse **en büyük kazanç** (×2–×5 aralığı olası) | doğrulama için evet |
| **O4** | HDRI: RGBA32F → **RGBA16F + mip zinciri** (B10G11R11 da aday). Linear filtre desteğini `vkGetPhysicalDeviceFormatProperties` ile kontrol et, desteklenmiyorsa açık hata ver | 16F'nin 10 bit mantissası HDR için yeterli, fark görünmez | HDRI payının (≈800 ms) çoğu, bellek ¼ | doğrulama için evet |
| **O5** | Transmission ortam okumasını önceden süzülmüş haritaya taşı: roughness → mip, kanal başına **1 okuma** (27 yerine 3). Ambient'ın 2026'daki aynı değişikliğinin devamı | ambient ile aynı yaklaşıklık. RT referansına karşı ölçülür | balon pikseli başına ~24 okuma eksik | evet |
| **O6** | Gölge PCF'sini `textureGather` ile yap. `params.w == 1` iken 3×3 komşu texel 4 gather'la tam kapsanır ve karşılaştırmalar aynıdır | **bit-aynı** | ışık başına 9 → 4 okuma (6 ışıkta 54 → 24) | evet |
| **O7** | Yükleme ekranı sahneyi kapatırken 3D sahneyi çizme (yalnız başlatıcı). `game_design` tik sayısını `scene_load` ayrıntısına yaz | aynı | `game_design` 39 s → SDF hazırlık süresi | evet |
| **O8** | Mali'de çalışmayan geçişlerin eklerini (yansıma G-buffer, TAA previousClip) yazma. Önce B7 formatlarını oku | aynı | tile bellek / bant genişliği | evet |
| **O9** | (Kullanıcı kararı, EN SON) Otomatik iç çözünürlük: GPU süresiyle histerezisli ölçek. Önce testte kendiliğinden oluşan `render_scale=0.25`'in sebebini bul | — | piksele doğrusal | evet |

★ **CLAUDE.md §1 açığı, O1 ile birlikte kapatılmalı:** Player IPC kalite / ışıklandırma / gölge anahtarlarını `unsupported_in_player` ile reddediyor. Bu yüzden cihazda A/B yalnız yeni APK ile yapılabiliyor. En azından `shadow` (aç/kapa), `lighting_preset` ve `raster_viewport_quality_preset` okunup yazılabilmeli (Codex `world.*` için bunu yaptı). Bu olmadan tablette her hipotez bir APK turuna mal oluyor.

### Neden bu sıra
- O1 ve O2 tablet gerektirmiyor. Bugün yapılırsa akşamki ölçüm doğru soruya bakar.
- O3, O2'nin sonucuna bağlı. Spill yoksa O3'ün getirisi küçük olur ve öncelik O4–O6'ya geçer.
- O4 ve O6 görüntüyü değiştirmiyor, risk düşük. O5 küçük bir yaklaşıklık değişikliği içeriyor, bu yüzden RT referansıyla karşılaştırılmalı.
- O9 kullanıcı kararıyla zaten en sonda.

## 3b. Ölçüm 2026-11-23 öğlen: malioc + Windows Player (RTX 3060, 1920×1080, pacing 0)

### malioc (Arm Performance Studio 2026.3, Mali-G52 r1p0, sürücü r51p0) — B2 KANITLANDI
| Shader | Work reg. | Stack (alloca + spill) | Sınır |
|---|---|---|---|
| `material_preview_frag` (ve `_covered`) | **64 (%50 occupancy)** | **1632 B (968 + 664)** | LS |
| `material_preview_sdf_surface` | 64 (%50) | 832 B (448 + 384) | LS |
| `material_preview_sky` | 32 (%100) | 0 | A |
| `material_preview_shadow_frag` | 30 (%100) | 0 | T |

Ana shader'da her piksel **1,6 KB per-thread stack** taşıyor ve Load/Store sınırlı. Mali'de stack önbellekten belleğe taşar. "Fragment maliyeti piksele doğrusal ve içerikten bağımsız" ölçümünün açıklaması bu.
Kaynak varyantlarıyla ayrıştırma (glslc `-O`, aynı include'lar; deneme kopyaları scratchpad'de, repo değişmedi):
- `vec3 regs[MP_REGISTER_COUNT=32]` (malzeme graph VM'i, `material_program.glsl:339`): alloca'nın **448 B**'ı. Sahnede graph olmasa da yer ayrılıyor.
- Tek başına kapatılan hiçbir dal (ışık döngüsü, terrain, cam bloğu, hava, graph, ambient) spill'i bitirmiyor (596–680 B arasında kalıyor). **Baskı bütün uber-shader'a dağılmış.**
- Workshop'un kullanmadıklarını birlikte kapatmak (terrain + graph + hava + tile-break + mikro detay + sheen + şimşek + balon filmi + opak fazın sabitlenmesi): **stack 1600 → 608 B**, LS 837 → 271 çevrim (**~3×**). Işık döngüsü de kapatılınca 384 B kalıyor. Kalan spill'in kaynağı büyük olasılıkla `GpuMaterial mat = materials[i]` ile **struct'ın bütün kopyası** (`frag:681`; terrain dalı `mat`'a yazdığı için referansa çevirme denemesi derlenmedi).
- **O3 doğrulandı**: specialization ile budama ana kazanç. Ama "spill = 0" için ayrıca malzeme struct'ını alan alan okumak gerekiyor (O3b).

### Yeni bulgu B8 — cam bloğu her opak pikselde çalışıyor
`frag:1482–1560`: `sampleTransmissionEnvironment` (5–9 koni okuması), `previewResinInterior` (FBM gürültüsü) ve `imageLoad`, `transmission == 0` olan duvar pikselinde de hesaplanıyor. Sonuç `transmission > 0.001` olmadığı için kullanılmıyor. `glassPath = bubble || transmission > 0 || interiorDepth > 1e-4` kapısı **bit-aynı** sonuç verir (kapı dışında `transmissionLit = 0`, `resinBaseWeight = 0`, `dielectricPass = 0`). malioc'ta tek başına küçük bir fark çıkıyor (LS 1030 → 979) ama duvar pikselinin dinamik maliyetinden 5–9 doku okumasını ve bir FBM'i düşürür. → **O10**, en ucuz iş.

### Windows Player ölçümü — "GPU sınırı olamaz" sezgisi DOĞRU, iki ayrı sebep var
- Boşta 49,6 Hz. Raster GPU **17,5 ms**: `main_pass` 8,6 + `transmission` 5,6 (balonlar) + `rt_shadow` 2,3 ms, sahnede yalnız **40 bin üçgen ve 27 draw**. RTX 3060'ta bu sahne 1–2 ms olmalı. Sebep Mali'dekiyle aynı shader.
- CPU tarafında `advance_ms_avg` **15,5 ms**, render kaydı + sunum ise 3,4 ms. İç ölçek 1 → 0,5 → 0,25 yapılınca `advance` **17,1 → 6,2 → 2,8 ms** düştü (GPU raster 19,0 → 7,1 → 3,1 ms). Yani `advance` **GPU'yu bekliyor**: CPU ile GPU üst üste binmiyor, kare = GPU + CPU.
  **B9 (kök neden adayı):** `RuntimeSceneViewer.cpp:573` → `uploadParticleBillboards` → `VulkanViewportParticles.cpp:668 drainInteractiveViewportInFlight()`. Sahnede parçacık sistemi var (9 sistem), bu yüzden **her kare** çağrılıyor, canlı parçacık 0 olsa bile. Ardından `GPU_ONLY` tampona `uploadBuffer` da muhtemelen senkron. → **O11**: billboard tamponları kare-yuvası başına çift tampon olsun (host-visible halka), ve sistem değişmediyse/boşsa yükleme atlansın. Drain kalksın. Beklenen etki: PC'de kare ≈ max(GPU, CPU). Tablette ARM CPU ile GPU paralel çalışır.
- **Patlamada (11 balon arka arkaya)** 50 → 40–43 Hz. `paint_wet` saniyede **146 ms GPU** (çağrı başına 0,40 ms, kare başına ~8 dispatch, yani ~3,4 ms/kare). Parçacık çekirdekleri <1 ms/s. Raster patlamada azalıyor (balon kalmadığı için transmission 5,6 → 2,9 ms). Düşüşün kaynağı ıslak boya. 8 yüzey × 1024² (`pbr_buffer_bytes` 302 MB). Mali'de aynı iş ~20× yavaş çalışacağı için açık ara ikinci darboğaz. → **O12**: `paint_wet` dispatch'lerini yalnız aktif tile'lara indir (groups = `tile_count`, yani bütün tile'lar dispatch ediliyor, kernel içeride eleniyor). Kare başına 8 modun birleştirilip birleştirilemeyeceğine bak.
- Açılış (PC): 3,7 s, `game_design` 2,2 s. Tablette 38,9 s: oran ~18×, kare süresi oranıyla (~200×) uyuşmuyor. B6 kısmen doğru; `game_design` içinde ayrıca CPU/SDF işi var. Tablette ölçülecek.
- ★ `player.operator display.configure` **tercihlere KAYDEDİYOR** (`PlayerOperator.cpp:751 s.save()`). Android'de "render_scale kendiliğinden 0.25'e düştü" gizeminin en olası açıklaması bir testin bu çağrıyla ölçeği kalıcı yazması. Testler ölçeği geri yazmalı ya da ölçüm için kaydetmeyen bir yol açılmalı.

### Uygulanan (2026-11-23 öğle)
- **O10 YAPILDI, CANLI ölçüldü:** `frag:1513` `glassPath` kapısı. `material_preview_frag` ve `_covered` SPV iki yere dağıtıldı. RTX 3060 1080p: `main_pass` 8,65 → **7,38 ms**, raster 17,5 → 16,1 ms, döngü 49,6 → 53,7 Hz. malioc'ta statik tablo neredeyse aynı (stack 1664 B): kazanç dinamik, yani duvar pikseli artık koni okumasını ve FBM'i yapmıyor. Spill'i O3 çözecek.
- **O11 YAZILDI, derleme bekliyor:** `uploadParticleBillboards` drain'i artık yalnız bir tampon yükleme ya da descriptor yeniden yazma olacaksa yapıyor (`writesDeviceState` lambda'sı, header değişmedi). /Zs Studio+Player PASS. Kalan her kare drain durumu: canlı CPU-quad grubu (O11b: kare yuvası başına halka tampon).

- **O3 YAZILDI (derleme bekliyor, SPV dağıtıldı):** `material_preview_frag`'a `RT_SPEC_TERRAIN/GRAPH/WEATHER` (constant_id 0–2, varsayılan true). Seçim deneylerle yapıldı: bu üçü stack kazancının neredeyse tamamı. Diğer beş dalı (tile-break, mikro detay, sheen, şimşek, balon filmi) eklemek yalnız ~20 B getiriyordu. malioc'ta üçü kapalıyken **stack 1664 → 704 B, LS 988 → 332**. Host (`MaterialPreviewCoverage.cpp`): maske shader kapılarının aynısı (malzeme `flags` bit16, `m_rasterMaterialPrograms.hasProgram`, `m_cloudShadowBound && m_cloudRtRendering`). Kanıtlanamazsa bit açık kalır. İlk kurulum özelleştirilmiş, covered varyantı aşama bilgisini kopyaladığı için miras alıyor. Yeni özellik görünürse **o karede** genişletilip yeniden kuruluyor, daralma 240 kare gecikmeli. Kurulum başarısız olursa tam maskeye düşülüyor. IPC: şimdilik `perf.list` (`viewport.material_spec.*`); `applied.feature_spec` bir sonraki tam derlemeye bırakıldı.
- **O3b (sıradaki shader işi):** kalan 424 B spill. Aday `GpuMaterial mat = materials[i]` struct kopyası. Terrain dalı `mat`'a yazdığı için önce o dalın yerel değişkenlere taşınması gerekiyor.

### T1 YAZILDI (2026-11-23 öğleden sonra) — tam derleme bekliyor
- **O1 + IPC:** `frame_timings.applied.tiled_gpu` + TBDR uyarısı, `applied.feature_spec`. Player'a kalite/ışık modeli/TAA/ışık yoğunluğu A/B anahtarları açıldı (ışık başına gölge anahtarı yok; gölgeli ışık sayısını kalite ön ayarı belirliyor: `rasterShadowLightBudget`).
- **O4a:** HDRI `uploadEnvironmentTexture` ile RGBA16F (clamp + log). Mip yok (sky dikişi).
- **O12, plan değişti:** indirect dispatch YERİNE shader içinde workgroup kapısı. Asıl israf grup sayısı değil, her iş parçacığının komşu bayraklarını ayrı ayrı okumasıydı: tick/resolve 9, channels 25 okuma × 256 iş parçacığı × 32768 grup. Şimdi thread 0 okuyor, `s_gate` ile yayınlıyor. Sonuç bit-aynı, host/binding değişmedi. Indirect liste gerekirse ölçümden sonra.

### (eski plan metni) Tam derleme partisi T1 — header değiştiren işler bir arada
Artımlı derlemeye sığmayanlar. Hepsi tek tam derlemede toplanacak:
- **O12 ıslak boya:** her mod `tile_count` grup başlatıyor (8 × 4096 tile × 256 iş parçacığı ≈ 8,4 M; pasif tile erken çıksa bile). Çözüm: rotate modunda aktif (genişletilmiş) tile listesini sıkıştır, tick/resolve/channels/stamp'ı `vkCmdDispatchIndirect` ile yalnız o listede çalıştır. Gerekli: `SimulationCompute.h` `ComputeDispatch`'e indirect alanı. Ölçüm için mod başına zaman etiketi (`paint_wet.tick` vb., kernel adı pipeline'ı seçtiği için ayrı bir `label` alanı) da aynı partiye girer. `Measure-PlayerGpuPasses.ps1` önek toplamına güncellenir.
- **O4a HDRI 16F:** `uploadTexture2D` bütün float RGBA dokuların ortak yolu (malzeme EXR, veri dokuları). Yalnız ortam haritası için `uploadEnvironmentTexture` (RGBA16F, 65504'te clamp + clamp sayısı logu). `VulkanSceneSync.cpp:46` önbellek format karşılaştırması da güncellenir. Mip AYRI iş: sky pass'i `texture()` ile equirect okuyor ve dikişte türev sıçraması mip'le görünür çizgi yapar. Önce `textureLod` / `textureGrad` gerekir.
- **IPC açıkları (CLAUDE.md §1):** `frame_timings.applied.feature_spec` (O3 maskesi + yeniden kurulum sayısı, `RtApi.h`/`IBackend.h`). Player'da kalite/ışık/gölge anahtarlarının `unsupported_in_player` reddi kalkar.
- **O1 TBDR etiketi:** `RasterStageTimings` / `applied`'a `tbdr_merged`. Mali'de sky+main_pass tek satır.
- **O7 (ölçüme bağlı):** yükleme sırasında render bilerek zorlanıyor (`PlayerMain.cpp:1039`). `game_design` tik sayısı tablette ölçülmeden kesilmez.

### Güncel öncelik (ölçümle)
1. **O11**: parçacık yüklemesindeki her kare drain'i (CPU↔GPU seri). Küçük ve görüntüyü değiştirmez.
2. **O10**: cam bloğu kapısı. Birkaç satır, bit-aynı.
3. **O3 + O3b**: specialization ile budama ve malzeme struct'ını alan alan okuma. Ana kazanç (~3× LS). Hedef spill = 0, malioc ile doğrulanır.
4. **O12**: `paint_wet` yalnız aktif tile'larda.
5. O4–O6, O1, O7 önceki sırada.

## 4. Sinsi tuzaklar
- **"sky" sütununu optimize etmek.** Gökyüzü shader'ını hızlandırmak sıfır kazanç getirir ve ölçüm yine 2,6 s gösterir (B1).
- **Specialization ile dalı yanlış kapatmak** (ör. sahnede clearcoat'lu tek malzeme var ama maske malzeme listesinden değil ilk malzemeden okunuyor). Görüntü "makul" kalır, yalnız o malzeme biraz mat görünür. O3 maskesi **bütün malzeme tablosundan** türetilmeli ve IPC'de okunabilmeli.
- **RGBA16F'de 65504 üstü değerler** (çok parlak güneş diski HDRI'leri) `inf` olur. Yükleme sırasında clamp uygulanmalı ve clamp sayısı loglanmalı.
