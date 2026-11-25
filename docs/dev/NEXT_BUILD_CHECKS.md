# Sıradaki derleme kontrolleri — Player Android A1 (2026-11-22)

## Studio Play ıslak boya yayını ve opak balonlar (2026-11-25)

- **Studio canlı PASS, `workshop`:** Play → Stop çökmesi giderildi. İlk kesintisiz Play aralığında 2.121 GPU renk ve 7.238 GPU kanal kopyası, sonraki Play aralığında 1.175 renk ve 4.031 kanal kopyası kaydedildi. Her iki aralıkta da boya dokusu RAM indirme sayaçları sabit kaldı.
- **Açık performans kalemi:** Stop → yeniden Play geçişinde editör önizlemesi eşitlenirken 826,25 MiB boya dokusu geri okundu. Play yeniden başladığında sayaçlar sabitlendi.
- **Slider:** Roughness, normal ve gizli resim ayarları artık yalnız etkilenen yüzeyi ve kanalları yeniliyor. Kullanıcı maliyetin düştüğünü doğruladı.
- **Opak balonlar:** Eski sahneler dahil Windows'ta opak açılış doğrulandı. Android cihazında eski sahne açılışı ve Oyun Ayarları'nda opaklık kontrolünün kaldırılması canlı kontrol bekliyor.

## O7d Viewport `VkPipelineCache` (2026-11-25 gece, Antigravity) — ARTIMLI (Studio + Player + Android APK)

- **Değişiklikler:** yeni `Viewport/ViewportPipelineCache.h/.cpp` (`viewport_raster_<vendor>_<device>.bin`, `VulkanViewportBackend` + `MaterialPreviewCoverage/Ibl/Shadow` + `RasterTaa` + `RasterGpuCull` bağlandı).
- **Statik kontroller:** `gen_player_vcxproj.py --check`, `gen_player_android_cmake.py --check`, MSVC `/Zs` (8/8, 0 uyarı), Android NDK arm64 `-fsyntax-only` PASS.
- **Derleme sonrası kabul (Android APK):**
  1. 1. açılışta `viewport_raster_*.bin` kaydedilmeli; 2. (sıcak) açılışta `first_frame` **9,92 s** → **<1–2 s**, toplam `scene_load` **24,7 s → ~14–16 s** bandına inmeli.
  2. Balonlar açılışta opak görünmeli. Yalnız balonlar transmission üretiyorsa geçiş ≈ 0 ms olmalı (boşta raster GPU ~180–190 ms / >5 FPS hedefi).

## O8 + Android 512 boya tavanı + küçültülmüş pencere kapısı (2026-11-24 gece, Antigravity) — ARTIMLI (Windows Player + tek Android APK paketi)

Bu paket önceki O7a + O7b + O7c ile birlikte tek Android Release APK derlemesine hazırdır (`.\scripts\Build-PlayerApk.ps1 -Abi arm64-v8a` veya `-Abi both`).
- **O8 (`RayFusionViewportPasses` + `VulkanViewportBackend`):** Ray-query olmayan GPU'larda (Mali-G52) HDR geçişinde kullanılmayan `reflectionNormal` + `reflectionSpecular` (96 bit/px) ekleri `VK_ATTACHMENT_UNUSED` / `DONT_CARE`. Log: `[RayFusion] HDR render pass omitting unused reflection G-buffer attachments...`.
- **Android 512 (`BalloonPlayerDefaults.h` + `GameDesign.cpp`):** Android Player açılışta ıslak boya yüzeylerini 512 ile sınırlar (`pbr_buffer_bytes` ~4× düşer, Mali boya süresi ~4× kısalır; F2/oyun ayarı yine üstüne yazabilir).
- **Küçültülmüş pencere (`AppFrame` + `PlayerMain`):** Windows'ta pencere simge durumuna küçültülünce (capture kapalıyken) kare üretimi durur, 20 Hz IPC canlı kalır.
- **Kabul:**
  1. **Windows Player:** Simge durumuna küçült → 2 sn bekle → `frame_telemetry.frames_consumed` / render sayısı artmamalı; geri açınca hemen normal FPS'e dönmeli; `RT_VK_VALIDATION=1` ile hata olmamalı.
  2. **Android APK (Lenovo TB330FU):** Soğuk + sıcak açılışta `scene_load` (`prepare_particles`, `game_design`, `elapsed_ms`), `[SimComputeCache] Loaded`, `[RayFusion] HDR render pass omitting...`, `paint.wet_status` (`resolution=512`), boşta raster GPU (önceki 697 ms) ve patlamada `paint_wet` sürelerini (önceki `tick 174 ms`, `channels 96 ms`) karşılaştır.

## O7c `game_design` hızlandırma: paralel `WetPaintGeometryBake` + gereksiz gutter atlama (2026-11-24 gece, Antigravity) — ARTIMLI (Studio + Player + Android APK)

- **Değişiklikler:** yeni `Paint/WetPaintGeometryBake.h/.cpp` (tek ortak OpenMP kodu; Studio `/openmp`, Windows Player `/openmp`, Android `-fopenmp -static-openmp`), `WetPaintGpu.cpp` (`bakeGeometry` çağrısı + paralel `packRgba8`), `GameDesign.cpp` (`[GameDesign] apply total=... ms (surfaces=..., hidden=...)` logu).
- **Statik kontroller:** `gen_player_vcxproj.py --check` PASS, `gen_player_android_cmake.py --check` PASS, MSVC `/Zs /openmp` 3/3 PASS, Android NDK arm64 `-fopenmp -fsyntax-only` PASS.
- **Derleme sonrası kabul:**
  1. `workshop.rtp` açılışında `scene_load` → `game_design` süresini önceki **2.103,5 ms** ile karşılaştır; `SceneLog.txt`'de `[GameDesign] apply total=... ms (surfaces=... ms/7, hidden=... ms)` satırını oku.
  2. Duvarlara ve gizli nesnelere balon patlat; `paint.wet_status` içinde `valid_texels`, `gutter_texels`, `tiles` değerlerinin ve boya yerleşiminin öncekiyle birebir aynı olduğunu doğrula.

## O7a + O7b açılış hızlandırma: sim compute pipeline önbelleği + `game_design` render kapısı (2026-11-24 gece, Antigravity) — ARTIMLI (Studio + Player + Android APK)

- **Değişiklikler:** yeni `Device/SimulationPipelineCache.h/.cpp` + `SimulationComputeVulkan.cpp` (273 kernel yerine yalnız 5 çekirdek kernel önceden ısıtılır, kalanı ilk dispatch'te tembel derlenir; diskte kalıcı `VkPipelineCache`), `PlayerMain.cpp` (`game_design` sırasında örtülü 3D raster atlanır), `GameDesign.cpp` / `RuntimeSceneViewer.cpp` / `PlayerSceneLoad.h` (`scene_load` `game_design` aşamasına `sdf_wait_ticks`).
- **Statik kontroller:** `gen_player_vcxproj.py --check` PASS, `gen_player_android_cmake.py --check` PASS, Player MSVC `/Zs` 5/5 PASS, Android NDK arm64 `-fsyntax-only` PASS.
- **Derleme sonrası kabul:**
  1. **Soğuk + sıcak açılış (Windows Player & Android APK):** İlk açılışta logda `[SimCompute] Registered 273 kernels; pre-warmed 5 in <ms> ms (cache=cold)` ve `[SimComputeCache] Saved (...)`; ikinci açılışta `cache=hit` ve `[SimComputeCache] Loaded (...)` görünmeli.
  2. **`scene_load` ölçümü:** `prepare_particles` ve `game_design` sürelerini önceki Lenovo 2257 profiliyle (`22,58 s` + `21,60 s`) karşılaştır; `game_design` satırında `sdf_wait_ticks=<N>` görünmeli, yükleme bitince ilk 3D kare gecikmesiz çizilmeli.
  3. **İlk patlama & tembel derleme:** Player'da balon patlat → boya/damla hatasız çalışmalı. Studio'da sıvı/MPM/DEM sahnesi çalıştır → ilk dispatch'te ilgili kerneller tembel derlenip çıkışta önbelleğe kaydedilmeli.

## ★★★★ Döşemeli duvarda tuval: `paint_base_tex` (2026-11-24, Claude yan oturum) — ★ TAM DERLEME (Studio + Player)

★ **BUG FIX (2026-11-24 gece — ✔ CANLI DOĞRULANDI):**
1. `MeshPaintAdapter.cpp`: `isGeneratedPaintTexture` yalnız tek renk / tamamen şeffaf
   (`isUniformOrEmptyPaintTexture`) generated tuvali atıyor; kayıtlı `.rtp`'deki gerçek duvar/zemin/normal
   dokuları korunuyor.
2. `WetPaintGpu.h/.cpp` (`cleanBaseTextureForSave`) + `MaterialManager.cpp` + `ProjectManager.cpp`:
   overlay, composite ve `@base` yüzeyleri kaydedilirken geçici şeffaf/beyazlatılmış/boyalı tuval yerine
   temiz taban dokusu yazılıyor.
3. `WetPaintGpu.cpp` (`updateCanvasModes` + resim-değişim reset): Materyal panelinden albedo dokusu
   değiştirilince/silinince yüzey canlı olarak yeni dokuyu benimsiyor (`@base` dahil); yeni resim yolu
   atanınca eski `state.w` sıfırlanıyor.
4. `VulkanBackend.cpp` (`uploadMaterials`): `getTexID()` içinde yüklenen `vulkan_dirty` yeni dokular döngü
   sonunda `compactMaterialTexturesLocked()` ile 2D-dizi sayfalarına + `refTable`'a alınıyor (RayFusion
   viewport yeni dokuyu anında gösteriyor).
5. `paint_wet.comp` (`paint_wet.spv` iki klasöre dağıtıldı, `spirv-val` Vulkan 1.2 PASS): `pictureKind`
   akromatik eşiği `sat < 0.10 || mx < 0.12`; `pictureRevealTarget` tek patlamada snap yerine bileşen
   renklerini %45'e kadar, tam ton/karışımı ıslak boya süresince kademeli olarak %100'e açıyor.

★ ÖNCE: bu parti YAYGIN STRUCT değiştiriyor → **tam derleme**, artımlı değil. `PrincipledBSDF.h` (+`paintBaseTexture`),
`Backend/IBackend.h` `MaterialData` (+`paintBaseTexture`), `Backend/vulkan_material_types.h` `VkGpuMaterial` 400→**416 B**
(blok 26), çekirdek `_core_pad0` → `paint_base_tex` (boyut aynı). Artımlı derleme eski .obj bırakırsa erişim ihlali /
saçma malzeme = derleme sorunu, kod hatası değil. SPV iki yerde (eski exe ile aynı çizer: eski host bu sözcüğe 0 yazıyor).
/Zs PASS 9 dosya, `audit_shader_struct_layout.py` OK. Not: `docs/dev/WET_PAINT_CANVAS_UV.md`.

1. **Döşemesiz yüzeyler değişmemeli (bağımsız, hızlı):** workshop'u aç; `paint.wet_status` her yüzeyde
   `canvas_mode: "composite"`, `base_texture: ""`. Duvarlar, oyuncaklar, boya, gizli resim eskisi gibi.
   Bozuksa: ortak yolun `paint_base_tex == 0` dalı değişmiş demektir (raster frag / closesthit).
2. **Döşemeli duvar (asıl kabul, atılabilir kopyada):** PaintBack'e döşemeli bir doku ver (ör. `wall3.png`) ve döşemeyi
   panelden ya da `python scripts/test/rt_wet_paint_tiled_canvas_ipc.py --object PaintBack --player --tile 6 --hold 30`
   ile 6'ya çek. Beklenen: `canvas_mode: "overlay"`, `base_texture` dokunun adı, `material.get uv_scale_x` = 6 KALIR;
   duvar kağıdı 6× tekrarlar, balon lekeleri gerçek boyda TEK kez, gizli resim duvarı TEK kez kaplar.
   Bozuksa: (a) duvar kağıdı tekrarlamıyor ama boya doğru → `paint_base_tex` shader'a ulaşmıyor (VulkanBackend
   `getTexID`/`maybeQueue`) ya da dönüşüm yine sıfırlandı (`bindTextureSetToMaterial` koruması); (b) boyasız yer SİYAH →
   tuval önceden çarpılmış/şeffaf değil: `paint_wet.spv` eski ya da `kSurfaceOverlay` bayrağı gitmiyor; (c) duvar doğru ama
   boya hâlâ küçük ve tekrarlı → eski `material_preview_frag.spv` / `_covered.spv`.
3. **Studio Rendered (Vulkan RT):** aynı duvar raster ile aynı görünmeli. Farklıysa `closesthit.spv` eski.
4. **Geri dönüş:** döşeme 1 → `composite`, duvar kağıdı eskisi gibi bir kez gerilmiş; o yüzeyin boyası sıfırlanır (bilinçli).
5. **Kaydet/aç:** betik `--roundtrip E:/tmp/tiled.rtp` → açılışta yine `overlay`, `uv_scale_x` 6.
6. ★ **SİNSİ — GI:** overlay'e geçince oda KARARMAMALI. RayFusion prob GI artık duvar kağıdını okur; boya GI'a girmez
   (bilinçli). Oda belirgin koyulaşırsa `RayFusionBounce.cpp` tabanı değil tuvali (boyasız yerde siyah) okuyor demektir —
   kimse bunu bug diye raporlamaz, "biraz loş" görünür.
7. **Taban dokusu olmayan tuval:** düz renk bir duvarda döşeme verilirse 1'e geri çekilir, logda
   `[WetPaint] ... reset to identity`; boya tekrarlanmaz.
8. **Panel:** Studio Game Design yüzey ayarlarında overlay iken "Canvas over tiled base: <doku>" (salt-okur).
9. **Android:** APK'dan önce `python scripts/compile_player_android_vk11.py` (değişen 5 raster/compute SPV'nin 1.1 kopyası).
Bilinen: CPU render ve OptiX (dondurulmuş) tuvali hâlâ döşer; döşemeli normal/roughness haritası kanal tuvalinde bir kez
gerilir; `@base` resim modu döşemesiz kalır (taban = resim).

## ★★ Gizli resim düzeltmeleri (2026-11-24 akşam, YALNIZ SHADER, SPV iki yerde — uygulamayı yeniden başlat)

1. Kontur/açık ton (tür 1) pikseli boyayı siliyordu (pastel duvarda boya bir an tutup kayboluyordu): boya artık
   `açılma × (1 − kontur)` kadar kalkar — kontur 1'de boya kalır, 0'da yerini resim alır.
2. Açılma kararlı: eşleşme×kaplama küçük bir eşiği geçince texel 1'e çıkar (`revealSnap`), sonraki farklı renk onu
   örtemez. ★ SİNSİ: hâlâ "sonraki renk siliyor" görünüyorsa `paint.wet_coverage.picture_revealed_texels` düşüyor mu bak;
   düşüyorsa bir state yazıcısı w'yi sıfırlıyor, düşmüyorsa görüntü/kompozit tarafı. Ayrıca demo 30 sn boşta kalınca
   odayı SIFIRLAR (`idle_reset_seconds`) — açılmalar da gider, bu hata değil.

## ★★★ Gizli resim: GÖRÜNMEZ mod + `@base` (2026-11-24, Claude + alt ajan) — header (WetPaintGpu.h, RtApiPaintWet.h) → Studio + Player derlemesi

SPV iki yerde (düzen aynı; eski exe ile çalışır ama yeni ayarlar ancak derlemeyle gelir). Yeni: `picture_outline` 0..1
(0 = resim tamamen görünmez; kontur/renksiz pikseller HERHANGİ boya ile, renkli bölgeler tonu tutan boya ile açılır),
`reveal_picture="@base"` (duvarın kendi dokusu gizli resim, görünen duvar `picture_blank` rengi, varsayılan beyaz; kapatınca
asıl doku geri gelir). Studio Game Design → Hidden picture: "Use wall texture as picture", "Blank wall colour",
"Outline visibility". ★ UV'ler 0..1 olmalı (elle büyütülmüş UV resmi küçültüp tekrarlar — tuval oturumu bunu ele alıyor).

1. **Görünmez mod:** outline 0, `@base` → duvar düz beyaz; balon patlayınca değen yerlerde duvarın kendi dokusu belirir,
   renkli bölgeler yalnız doğru renkle. `paint.wet_coverage.picture_ratio` artar.
2. ★ **SİNSİ:** `@base` kapatılınca asıl doku geri gelmeli (gelmezse `base_original` geri yüklenmiyor). Malzeme rengi
   tabanlı yüzeyde beyaz bir an malzeme rengine dönüp geri gelebilir (bilinen).
3. Outline 1 + kontur üstüne boya: kontur boyanın ÜSTÜNDE kalır (boyama kitabı; bilinçli).

## ★★★★ Gizli duvar resmi: renge göre açılan boyama (2026-11-24, Claude + 2 alt ajan) — ARTIMLI (2 HEADER: WetPaintGpu.h, RtApiPaintWet.h → onları içeren ~15 .cpp)

★ ÖNCE: derleme BİTTİKTEN SONRA `RayTrophiStudio/source/shaders/paint_wet.spv`'yi `x64/Release/shaders/`'a kopyala
(yeni SPV 18 sözcüklük istatistik bekliyor, eski exe 16 ayırıyor — erken kopya = tampon taşması). Ajan ya da ben kopyalarım.

Yeni: yüzey ayarı `reveal_picture` (resim yolu), `picture_tolerance` (ton derecesi 5..90, varsayılan 30), `picture_hint`
(0..1). Resim UV sayfasına oturur: saydam piksel = duvar; koyu/doygunluğu düşük opak = HEP görünür (kontur, bulut);
doygun = gizli, TONU tutan boya (karışım dahil) değince kalıcı açılır (`state.w`), oda sıfırlanınca kapanır. Kaynak atlas
düzlem 6, `_p1 & 8`, yüzey bayrağı bit4 + tolerans/ipucu baytları, istatistik adımı 18 (16 resim pikseli, 17 açılan).
IPC `paint.wet_set_settings` / Python / oyun tasarımı kaydı / Studio Game Design → "Hidden picture" (Browse). Örnek:
`local_assets/hidden_pictures/kids_house.png` (repoda değil). /Zs PASS (6 dosya), tarif üretici + denetim OK.

1. **Resimsiz yüzeyler değişmemeli (bağımsız, hızlı):** mevcut duvarlarda boya/kabartma/parlaklık aynı. Kod resimsiz
   yüzeyde eski `composite`'e düşer.
2. **Açılma:** Studio'da bir duvara (ör. PaintBack) resmi ver → konturlar ve bulutlar görünür, renkli bölgeler gizli.
   Sarı balon → güneş/çiçek göbeği açılır; kırmızı → ev; mavi → çatı; kırmızı+sarı karışımı → turuncu kapı/top.
   `paint.wet_coverage` → `picture_ratio` artmalı.
3. ★★ **SİNSİ — yanlış renk açıyor/açmıyor:** her renk her yeri açıyorsa tolerans/ton yanlış; hiçbir renk açmıyorsa
   (a) resim yüklenmedi → log'da `[WetPaint] hidden picture could not be loaded` (yol göreli ise proje klasörüne göre),
   (b) satır yönü ters → resim baş aşağı görünür (UV v ekseni), (c) `_p1 & 8` bitine bak.
4. **Sıfırlama:** oda sıfırlanınca resim tekrar gizlenmeli, konturlar kalmalı. Resim DEĞİŞTİRİLİNCE eski açılmalar
   sıfırlanmaz (bilinen, oda sıfırlaması temizler).
5. **Player:** oyun tasarımı kaydıyla resim Player'a gelir (göreli yol proje klasörüne göre çözülür).

## ★★★ paint_wet channels: tile DEĞİŞİM bayrağı (2026-11-24, Claude) — ✔ CANLI: göz testi PASS; 4 balon channels 0,76 → 0,69 ms/adım, channels/tick 1,63 → 1,15; resolve 0,26 → 0,44 (karşılaştırma okumaları) → adım toplamı ~%11 net düşüş (sürekli yeni damga olan pencerede; kurumada daha büyük olmalı)

★★ ÇÖZÜLDÜ (2026-11-24): kök = VERİ, `PaintRight` normali Studio'da da dışa bakıyordu; kullanıcı Studio'da Normals → Flip + kaydet → Player'da boya tutuyor. Açık kalan: Flip Normals'ın IPC'si yok (§1) ve yüz yönü reddi için sayaç yok (sessiz arıza). Eski not: SAĞ DUVAR (`PaintRight`) boyayı tutmuyor. IPC deneyi: `paint.wet_set_splat facing_min=-1` ile 3 balon sağ duvara +0,34 kaplama, `facing_min=0` ile ~0 → damga YÜZ YÖNÜ testinde eleniyor (kıvılcım = çarpışma olayı var). `PaintRight` = sol duvarın Y etrafında 180° döndürülmüşü (det +1); sol/arka birim dönüşüm ve sorunsuz. Aday 1: Player'daki dünya normali (`N`) odanın dışına bakıyor (kullanıcı: bazı nesnelerde normal Player'da Studio'nun tersi). Aday 2: ince OBB'de `resolveAABB` en yakın yüze iter (geliş yönüne bakmaz) → arka yüz normali. Sol duvar aynı OBB/hızla çalıştığı için aday 1 daha olası. Kod okumasında kayıt (N_orig yerel) ve yükleme (getNormalTransform) doğru görünüyor. Player'da normal okuyan IPC yok (scene.object_info/raycast unsupported_in_player).

Önceki ölçüm: channels'ın asıl maliyeti tile sayısı (aktif tile'ın 5×5'i, her düzlem her adım). Şimdi binding 10 sonuna
tile başına "değişti" sözcüğü (`3*tc+3+t`): bit0 içerik (renk/ıslaklık → kendi pürüzlülük/renk düzlemi), bit1 yükseklik
(kalınlık/kaplama → 3×3 komşu normalleri). STAMP ve RESET iki biti koyar, RESOLVE eski/yeni karşılaştırır, ROTATE
temizler. Channels yalnız gereken tile'ları işler; normal + yükseklik önbelleği yalnız yakında yükseklik değiştiyse.
★ SPV eski exe ile GÜVENLİ: `tile_flags.length()` bölgeyi görmezse yazmaz ve eski 5×5 kapısına düşer (yani yeni exe
derlenene kadar kazanç yok, risk de yok). /Zs PASS.

1. **Kazanç (bağımsız):** 4 balon → `paint_wet.channels` adım başına 0,76 ms'den belirgin aşağı; channels/tick oranı
   1,6'dan düşük. Kuruma ortasında (akış durmuş) channels çok ucuzlamalı (yalnız pürüzlülük).
2. ★★ **SİNSİ — eksik düzlem:** yeni damganın kabartması (normal) ve ıslak parlaklığı GELMELİ; kuruyan lekenin
   parlaklığı zamanla matlaşmalı. Leke rengi var ama kabartma/parlaklık yoksa ya da tile kenarında kabartma yarım
   kalıyorsa bir değişim bayrağı eksik (stamp/reset/resolve). Akan damlanın izi boyunca kabartma da akmalı.
3. **Sıfırlama:** oda sıfırlanınca kabartma ve parlaklık da temizlenmeli (RESET iki bit).
4. Proje açılışı / yüzey ekleme (atlas yeniden kurulumu, zorunlu channels) sonrası tüm düzlemler doğru.

## ★★ paint_wet: channels tile önbelleği + doğal sıçrama şekli (2026-11-24, Claude) — YALNIZ SHADER — CANLI ölçüldü (RTX 3060, 4 balon): channels 0,92 → 0,76 ms/adım ama tick de 0,62 → 0,47 düştü (bu turda ıslak alan küçük); channels/tick oranı 1,48 → 1,63 = RTX'te önbellek KAZANÇ GETİRMEDİ (okumalar zaten L1/L2'de). Asıl maliyet 5×5 tile genişlemesi (tick 3×3) + 4 düzlem yazımı. Mali'de ölçülmedi. stamp 0,117 → 0,106 ms (parmaklar ek maliyet getirmedi). Görsel kabul bekliyor

`paint_wet.spv` derlendi, iki yere dağıtıldı (binding/push AYNI, mevcut exe ile uyumlu; Player'ı yeniden başlatmak
yeter). Android vk11 seti `Build-PlayerApk.ps1` ile yeniden üretilir.
- `paint_wet_channels.glsl`: normal gradyanı `addedHeight`'ı texel başına 8 kez (~40 global okuma) çağırıyordu;
  artık tile + 2 kenar (36×36) shared önbellekten okur (texel başına ~6 okuma). Sonuç bit-aynı.
- `paint_wet.comp` `eventSplatWeight`: 14 eşit sivri ışın (`pow(sin(14θ),8)`, "buz kristali") KALDIRILDI; yerine
  3–8 düzensiz, kökte kalın/uca incelen, ucunda boncuk toplanan parmak (hız ↑ → çok/uzun; sıyırmada ileri toplanır);
  uydu damlacıklar uçuş yönünde uzamış gözyaşı, karışık boy.

1. **channels maliyeti (bağımsız):** `Measure-PlayerGpuPasses.ps1 -PopBalloons 4` → `paint_wet.channels` adım başına
   0,92 ms'den belirgin aşağı (beklenti ~yarısı ya da daha az). tick/resolve değişmemeli.
2. ★★ **SİNSİ — normal haritası:** kuru boya kabartması ve ıslak parlaklık ÖNCEKİYLE aynı görünmeli (bit-aynı
   tasarım). Tile sınırlarında (32 texel ızgarası) çizgi/dikiş ya da kabartmanın tile kenarında kopması = önbellek
   indeksi kaymış.
3. **Sıçrama şekli (göz):** yıldız/kristal yerine birkaç düzensiz parmak + uçta boncuk + gözyaşı damlacıklar.
   Kenarlarda kesik/kare kesilmiş parmak varsa sınır yarıçapı (2,1) yetmiyor.
4. `sources_dropped` / `stamped_texels` benzer kalmalı; stamp süresi (`paint_wet.stamp`) en çok biraz artabilir
   (texel-kaynak başına ≤8 parmak döngüsü).

## ★★★ Kuruma GPU'su: ıslak boya 60 Hz adım + mip yalnız kirli bölge (2026-11-24, Claude) — ✔ CANLI: göz testi PASS (kullanıcı, uzak/eğik dahil); 4 balon: en düşük döngü 122 → 273 Hz, 4 sn ortalama 158 → 329 Hz; tick 55/s (önce kare başına); boya GPU 266 → 150 ms/s; paint.wet.submit 1,1 → 0,04 ms; sources_dropped 65k → 13k

Ölçüm (Windows Player 1280×720, 4 balon art arda, biri sürpriz): döngü **122 Hz**, GPU dolu (slot_wait ~4 ms,
`paint.wet.submit` 1,1 ms = halka GPU'yu bekliyor). Aktif kare başına boya GPU: channels 0,73 · tick 0,52 · resolve 0,29
· stamp 0,11 ms, raster 1,46 ms; ~150 Hz'de saniyede ~530 ms GPU ÖLÇÜLMÜYORDU → her doku kopyası (7 yüzey × 5 düzlem)
1024² sayfanın TÜM mip zincirini blit ediyordu. Değişen:
- `VulkanBackend.cpp`: `regenerateMipChainRegion` (.cpp içi) — kopya yolu yalnız kopyalanan dikdörtgenin her
  seviyedeki izini blit eder (çift hizalı, 2:1 kutu → tam blit ile aynı texeller; tek kenarlı seviye bütün).
  Diğer çağıranlar (`regenerateMipChainAfterPartialUpdate`) tam görüntüyle aynı yardımcıyı çağırır, davranış aynı.
- `WetPaintGpu.cpp`: `kWetStepSeconds = 1/60`. Stamp+consume her kare; tick/resolve/channels/rotate/coverage +
  okuma + yayın yalnız adımda (dt = son adımdan beri geçen, 2,5/60 sınırı aynı). İstatistik penceresi okuma başına bir
  kez sıfırlanır (damga-yalnız karelerin sayaçları kaybolmaz). 60 fps altında her kare adım = eskisiyle aynı.
/Zs Studio+Player PASS.

1. **Ölçüm (bağımsız):** aynı 4 balon senaryosu → döngü 122 Hz'den belirgin yukarı; `Measure-PlayerGpuPasses.ps1`
   `paint_wet.tick` örnek sayısı saniyede ≤60. `paint.wet_status.status` kuruma sırasında `between_steps`/`stamping`/
   `running` arasında gidip gelir.
2. ★★ **SİNSİ — mip:** boyalı duvara UZAKTAN ve eğik açıyla bak (üst mip'ler okunur). Boya uzakta da yakındaki gibi
   görünmeli; kirli bölgenin kenarında **kare kare yama / eski boya izi / dikiş** varsa bölge hizalaması yanlış.
   Islak→kuru geçişte uzaktaki renk yakındakiyle aynı anda değişmeli.
3. ★ **SİNSİ — akış hızı:** damlaların akma/kuruma hızı ÖNCEKİYLE aynı görünmeli (formüller 60 Hz'e kalibre). Masaüstünde
   yüksek FPS'te akış önceden FARKLIYSA (çok ince adımlar) şimdi tasarlanan hıza döner — gözle farkı not et.
4. **Damga gecikmesi:** damla duvara değdiği an ile lekenin görünmesi arası en çok bir 60 Hz adımı; fark edilmemeli.
   `sources_dropped` ve `stamped_texels` sayaçları azalmamalı (pencere birikimi).
5. **Studio fırça boyaması:** fırça darbesi gecikmesiz görünmeli; sıfırlama (reset) anında temizlemeli (zorunlu adım).

## ★★★★ RayFusion TLAS yerleşim yenilemesi BEKLEMESİZ (2026-11-24, Claude) — ✔ CANLI: göz testi PASS (kullanıcı); recorded_refresh 28,9/s (host 0,074 ms), sync_rebuild 0, resource_drains 14 sn'de 1 (önce ~30/s); boşta en uzun kare 5–6 → 2,4–2,5 ms, döngü 465 → 517 Hz; patlamada en uzun kare 15–21 → 4–8,6 ms (artık GPU'ya bağlı: slot_wait 3–4,6 ms)

Kök (canlı ölçüm, `frame.worst_iteration`): her saniyenin en uzun karesi boşta 5–6 ms, patlamada 15–21 ms, neredeyse
tamamı `render_ms`; `resource_drains` ≈ 30/s. `rebuildRayFusionTLAS` her yerleşim değişiminde (30 Hz balon hareketi)
drain + TLAS destroy/recreate + senkron build yapıyordu. Değişen:
- `VulkanBackend.cpp` `createTLAS` KAYITLI mod (externalCmd): yalnız yerinde kurulum; instance dizisi
  `vkCmdUpdateBuffer` ile komut içinde; önce/ara/sonra bariyerleri; koşul tutmazsa HİÇBİR ŞEY kaydetmez,
  `m_tlasSupportsUpdate=false` yapar (çağıran senkron yola düşer). Instance buffer artık `TRANSFER_DST`.
- `RayFusionSceneAS.cpp`: aynı instance sayısında async tek-atış (bekleme yok), yoksa eski drain+senkron yol;
  `allowUpdate=true`. perf.list: `rayfusion.tlas.recorded_refresh` / `rayfusion.tlas.sync_rebuild`.
- `VulkanViewportBackend.cpp` `drainInteractiveViewportInFlight` artık async tek-atışları da bekler
  (`reclaimAsyncCommands(true)`): BLAS silme/refit uçuştaki TLAS kurulumuyla yarışmasın.
/Zs Studio+Player PASS. ★ Header yorumu (`createTLAS` "hair ping-pong, host işi inline") ESKİ; header'a
dokunmamak için düzeltilmedi, bir sonraki tam derlemede güncellenmeli.

1. **Yol alınıyor mu (bağımsız, hızlı):** balonlar süzülürken `perf.list` → `rayfusion.tlas.recorded_refresh`
   saniyede ~30 artmalı, `sync_rebuild` yalnız açılışta/görünürlük değişiminde. `sync_rebuild` 30/s artıyorsa
   kayıt reddediliyor: log'da `createTLAS recorded refresh refused: <sebep>` satırı (bir kez) sebebi söyler.
2. **Drain gitti mi:** `viewport.frame_telemetry.resource_drains` artık ~30/s ARTMAMALI.
3. **Takılma:** `player.display` → `frame.worst_iteration.total_ms` boşta ~5–6 → **~2 ms**, patlamada 15–21 →
   belirgin düşüş; `render_ms` artık büyük pay olmamalı. Döngü boşta 465 → daha yüksek.
4. ★★ **SİNSİ — gölge/izleme doğruluğu:** RT gölgesi balonla birlikte hareket etmeli (bir kare gecikme bile
   olmamalı; kurulum bu karenin kayıtlarından ÖNCE kuyrukta). Gölge balonun eski yerinde kalıyor/titriyorsa
   bariyer ya da sıralama yanlış. Balon patlayınca (küçültülür, sayı aynı) gölgesi de kaybolmalı.
5. **Dayanıklılık:** 3–5 dk patlat (Debug + validation varsa `SYNC-HAZARD` aranır). Device lost ya da
   `VUID-vkCmdUpdateBuffer` = instance buffer'ın usage'ı ya da bariyer. Proje değiştir / sahne aç-kapa:
   BLAS silme yolları artık async'i bekliyor, cihaz kaybı olmamalı.
6. **Studio:** Material Preview'da hareketli nesne (gizmo ile sürükle) aynı yoldan geçer: sürükleme akıcı,
   gölge takip eder. Rendered modda RT render yolu (externalCmd'siz) DEĞİŞMEDİ.

## ★★ Player en uzun iterasyon dökümü (2026-11-24, Claude) — artımlı (yalnız PlayerMain.cpp)

Önceki parti CANLI ölçüldü (11:31 exe, kullanıcının 14 elle patlaması, IPC'siz): drain toplamı **0,0 ms** ✔,
`spawn_ms` 0,7–1,4 ms, masaüstünde çoğu 1 karede (sürpriz 10 728 damla → 3 kare) ✔. AMA `frame_interval_ms_max`
hâlâ **18–29 ms** (sessiz anda sürpriz 14,2); o pencerelerde sim ≤7, boya ≤5, efekt ≤3, spawn ~1 ms → takılma
simülasyon DIŞINDA. Yeni: `player.display` → `frame.worst_iteration` = 1 sn penceresinin en uzun iterasyonu,
aşama aşama (`input_ms` [dokunmatik patlatma burada çalışır], `pump_ms` [IPC], `advance_ms`, `render_ms` +
backend record/slot_wait/submit, `present_ms` [UI dahil], `sleep_ms`, `unaccounted_ms`). /Zs Player PASS.

1. **Alan var mı:** `player.display` → `frame.worst_iteration.total_ms` ≈ boşta 2 ms civarı, `unaccounted_ms` ~0.
   Büyükse dökümde bir aşama eksik.
2. **Takılmanın yeri:** elle patlat, 1 sn içinde `player.display` oku (IPC pop KULLANMA: kendi 27 ms'si `pump_ms`'e
   düşer). 18–29 ms'nin hangi aşamada olduğu görülmeli. ★ SİNSİ: `sleep_ms` büyükse takılma değil pacing'dir
   (pacing_hz>0); ölçümü pacing 0 ile yap.

## ★★★ Patlama karesi bölündü: LUT drain'i + damla kuyruğu (2026-11-24, Claude) — artımlı (2 .cpp)

Değişen: `UI/ParticleBillboardBuilder.cpp` (CPU efekt sistemleri LUT satırlarını ÖLÜYKEN de yazar → LUT
patlamada değişmez → `viewport.particle_billboards.drain` tetiklenmez), `Paint/BalloonPaintDemo.cpp`
(patlamanın damlaları kuyruğa, kare başına bütçe: max(0,6 ms, karenin %10'u), en geç 4 karede biter; kuyruk
aralıklı sıralı, geç doğan damla gecikme kadar ileri konur, ömrü kısalır). Header yok. /Zs Studio+Player PASS.
Ölçüm öncesi (1280×720, IPC pop): takılma karesi normal 15,0 / sürpriz 18,7 ms; kullanıcı patlamalarında 18–25 ms;
drain patlama başına ~5 kez, çağrı başına 1,1–2,4 ms; pop karesinde damla üretimi 1,2–2,4 ms.
★ Ölçüm: IPC `pop` yanıtı demonun bütün `status()`'unu döndürür (round-trip ~27 ms, ana iş parçacığında);
takılmayı ŞİŞİRİR. Sıkı `perf.list` yoklaması da döngüyü ~60 Hz'e indirir. Takılma için IPC'siz bak: elle
patlat, sonra `paint.wet_demo` `events[-1]` oku.

1. **Drain gitti mi (bağımsız, hızlı):** birkaç balon patlat, `perf.list` → `viewport.particle_billboards.drain`
   sayısı patlama başına **0** olmalı (önce ~5). Artıyorsa LUT başka bir sebeple değişiyor (profil ekleme,
   ışık) ya da pulled buffer'ları yeniden ayrılıyor; `upload` sayacına bak.
2. **Damla yayılımı:** `paint.wet_demo` `events[-1]`: `spawn_frames` **1–4**, `spawn_done_ms` birkaç kare,
   `spawn_ms` (pop karesinin kendi maliyeti) öncekinin (1,2–2,4) **altında**; status `pending_drops` patlamadan
   kısa süre sonra **0**. `spawn_frames` hep 1 ise bütçe hepsini ilk karede bitiriyor (masaüstünde olabilir,
   sorun değil); 4'ü geçemez.
3. **Takılma:** elle normal ve sürpriz patlat, `events[-1].frame_interval_ms_max`. Beklenen: 15–25 ms'den belirgin
   düşüş. Hâlâ ~15 ms ise kalan iş başka yerde (pop karesindeki ses, mesh küçültme / TLAS, wet paint ilk stamp);
   bir sonraki adım kare kayıt cihazı (en kötü karenin bölüm dökümü).
4. ★ **SİNSİ — görüntü:** patlama şekli aynı görünmeli: jetler aynı anda çıkar, damla yoğunluğu 1–4 karede dolar.
   Jetlerden biri belirgin geç çıkıyorsa kuyruk aralıklama bozuk. Boya izi miktarı aynı kalmalı
   (`spawned` aynı anlamda: süzgeçten geçen damla). Sürpriz balonda gökkuşağı topakları tek renk kalmalı.
5. **Testler:** `rt_player_balloon_pop_ipc.py`, `rt_balloon_reset_size_ipc.py` `spawned` alanını okur (anlam
   değişmedi). Patlamadan HEMEN sonra canlı parçacık sayan bir test varsa 1–4 kare beklemeli.
   Oda sıfırlama (`reset`) kuyruktaki damlaları da atar: sıfırlamadan sonra boya gelmemeli.

## ★★★ Compute gönderim HALKASI 3 yuva (2026-11-24, Claude) — artımlı derleme (yalnız 2 .cpp)

Değişen: `Device/SimulationComputeVulkan.cpp` (`kSlots` 2 → 3, uçuşta en fazla 2 gönderim; submit yalnız iki
kare önceki yuvayı bekler), `Paint/WetPaintGpu.cpp` (`kMaxPendingStats` 2 → 3, yeni `paint.wet.submit` scope'u).
Header/struct değişmedi. /Zs Studio+Player PASS. Sebep (canlı ölçüm, 1280×720, opak balon, tek directional,
1 balon): boşta 555 Hz, patlama+kurumada **298 Hz**; `simulation.wet_paint` CPU **1,77 ms/kare**, ama boya GPU işi
~0,14–0,5 ms, raster değişmiyor (1,29 ms). Her async submit bir önceki karenin compute gönderimini bekliyordu
(tek kuyrukta render karesinin arkasında) → kare = CPU + GPU, GPU ~%45 boş.

1. **Ölçüm (bağımsız, hızlı):** aynı sahne, timing KAPALI, 1 balon patlat ve `perf.list` farkı al
   (`Measure-PlayerGpuPasses.ps1` timing'i açar ve açıkken her submit SENKRON olur, bu madde için kullanma).
   Görmen gereken: `paint.wet.submit` ≈ 0,0x ms, `simulation.wet_paint` 1,77 → **< 0,2 ms**, döngü 298 →
   **~450+ Hz**. `paint.wet.submit` hâlâ ~1,5 ms ise bekleme halkada değil, `vkQueueSubmit`'in kendisinde
   (WDDM) demektir.
2. **Bekleme başka yere taşındı mı:** `player.display` → `slot_wait_ms_avg` artabilir (render halkası GPU'yu
   bekler). Bu beklenen ve doğru: artık GPU'ya bağlıyız. Döngü düşmüyorsa sorun yok.
3. **10 balon art arda** (`-PopBalloons 10`): `wet_paint.stats.stats_readbacks_lost` **0** kalmalı. Artıyorsa
   4 okuma yuvası yetmiyor (3 bekleyen + 1 yeni).
4. ★ **SİNSİ:** boya izleri artık en fazla **iki kare geç** görünebilir (öncesi bir). Gözle fark edilmemeli.
   Bir darbenin hiç görünmemesi ya da yalnız bir sonraki darbeyle görünmesi = son bilet toplanmıyor.
   Ayrıca damla titremesi / boya izi kaybı (dünkü görsel PASS'in tekrarı) ve **Studio**'da sıvı/kar
   sahnesinde 512 dispatch'lik ara gönderimli (MAX_DESC_SETS) bir sim: davranış değişmemeli. Bozukluk
   = cihaz kaybı ya da titreyen alan → WAR bariyerini (`ensureRecording`) şüphelen, iki uçuşta gönderim
   artık mümkün.

## Android erken açılış logcat kaydı (2026-11-23)

Lenovo TB330FU'da 2222 Release APK sahne ve IPC açılmadan `EXIT_SELF`,
durum 1 ile kapanıyor. Yeni APK'yı kurup Player'ı yeniden açın. Sonra
`adb -s HVA751MH logcat -d -t 5000 -s RayTrophiStartup:I` çıktısını alın.
Etiket native giriş, asset hazırlığı, SDL başlatma, Vulkan pencere oluşturma,
ImGui ve IPC başlangıcını gösterir. SDL pencere hatasının metni diğer SDL
çağrılarından önce korunur. Özel uygulama dizini veya debug APK gerekmez.

2249 APK bu adımların tamamını geçip IPC başladıktan sonra kapandı.
Sonraki APK Android'in native sinyal yakalayıcısını korur. Yeniden üretimde
`adb -s HVA751MH logcat -b crash -d` ile signal, Build ID ve backtrace
alın. Release sembollerini APK ile aynı derlemeden eşleyin.

2253 APK'da Build ID eşleşmesi ilk ImGui karesindeki font SIGSEGV'ini
doğruladı. Bir sonraki APK'da Vulkan backend, Android'de de launcher
karesinden önce başlar. Lenovo'da tek sahneli otomatik açılışı, yükleme
ekranını ve ilk oyun karesini doğrulayın. Ardından uygulamayı arka plana
alıp geri açın ve Geri menüsünü kontrol edin. GPU başlatma hatasında
yerel diyalog ve `RayTrophiStartup` hata satırı görünmeli.

## ★★★ Islak boya istatistik okuması BEKLEMESİZ (2026-11-23 gece, Claude) — ✔ ÖLÇÜM CANLI (22:03 exe): stats_collect 0,005 ms, 68,5 kare/sn, artık GPU'ya bağlı; görsel PASS (kullanıcı)

Tek gönderim derlemesi canlı ölçüldü (1080p, IPC 10 balon): `simulation.timeline` **9,4 → 1,3 ms** ✔, ama bekleme bir
sonraki senkron noktaya taşındı: `paint.wet.stats_collect` **0 → 4,3 ms** (önceki tick'in okumasını `wait=true` bekliyordu,
o gönderim de artık raster karesinin arkasında). 64,5 → 68 kare/sn. Düzeltme: biletler kuyruğa (en fazla 2), hazır olanlar
beklemeden toplanır; kuyruk doluysa en eskisi beklenir. `applyStats` kirli dikdörtgenleri BİRLEŞTİRİR, geç gelen okuma
bir şey kaybettirmez, yalnız bir kare gecikir. /Zs Studio+Player PASS.

1. **Ölçüm:** aynı test. Beklenen: `stats_collect` ~0, bekleme raster `slot_wait`'e taşınır. GPU bütçesi (raster 10,3 +
   boya ~2,5 ms) sınırı ~78 kare/sn; oraya yaklaşmalı. Hâlâ ~68 ise sıradaki senkron noktası başka yerde (perf.list'te
   en büyük ms/s satırı).
2. ★ **SİNSİ:** boya izleri **bir kare geç** belirir (gözle fark edilmemeli). Bir fırça darbesinin hiç görünmemesi ya da
   ancak bir sonraki darbede görünmesi = son bilet toplanmıyor (boşta çıkış). `wet_paint.stats.stats_readbacks_lost`
   artıyorsa okuma slotları yetmiyor.

## ★★★ Player: kare başına TEK async compute gönderimi (2026-11-23 gece, Claude) — TAM derlemeyle birlikte

Kök (canlı ölçüm, 1080p, IPC ile 10 balon): patlamada `simulation_tick` 12 ms'nin ~9'u, `endFrame()` → `synchronize()`
host fence'iydi. Tek kuyrukta önündeki raster karesini de bekliyordu; sonrasında CPU kareyi bitirirken GPU boş kalıyordu.
Değişiklik (Player'a özel, `SimulationClockInput::defer_compute_submit`; Studio aynı):
sim adımı kaydı açık bırakır → ıslak boya aynı tampona yazar → tick sonunda `submitAsync()` (`simulation.compute_submit`).
Staging artık yuva başına ve fence'te geri sarılıyor. İki bariyer: sim kaydının başında `ALL_COMMANDS` (raster okumalarına
karşı WAR), raster karesinin başında compute/transfer → grafik okuma.
Dokunulan: `SimulationCompute.h` (yeni sanal metot → TAM), `SimulationCompute.cpp`, `SimulationComputeVulkan.cpp`,
`RuntimeSimulationClock.h/.cpp`, `RuntimeSceneViewer.cpp`, `RasterViewportFrameRing.cpp`.

1. **Ölçüm (önce, bağımsız):** Player balon sahnesi, 1080p:
   `scripts/test/Measure-PlayerGpuPasses.ps1 -PopBalloons 10 -SkipComputeTiming` (IPC patlatır, insan gerekmez).
   Önce: patlama ~63 kare/sn, `simulation.timeline` ~9,4 ms, `simulation_tick` ~12 ms. Beklenen: timeline 1–2 ms'ye
   düşer, bekleme raster `slot_wait`'e taşınır, kare/sn ~75–85. Değişmediyse: kare içinde başka bir `synchronize()`
   var (bak: `endTransferBatch` downloads, `paint.wet.coverage` ~2/sn).
2. **Görüntü:** balon patlarken damlalar titremeden, bir kare geriden değil aynı anda görünür. Islak boya izleri yerinde.
   ★ **SİNSİ:** bariyer eksikse damlalar tek tek karelerde yanlış yerde (yarı yazılmış konum) çakar; ekran görüntüsünde
   değil, hareket halinde görülür. Kayıt al, kare kare bak. Sync validation açıksa 0 hazard olmalı.
3. **Uzun koşu (10 dk otomatik patlatma):** `perf.list`'te bellek sabit. Staging artık fence'te geri sarılıyor;
   büyüme varsa yuva imleci serbest bırakılmıyor demektir. Device lost OLMAMALI (kırıntı izi açık).
4. **Studio aynı davranmalı** (bayrak kapalı): sıvı/gaz sahnesi, pause/play kenarı, önbellek geri yükleme.
5. **Android (APK):** tablet/telefon patlama kare/sn önce/sonra. TBDR'de kazanç PC'den büyük beklenir (host fence boru
   hattını boşaltıyordu); ölçülmeden sayı yok.

## ★★★ T2 cam specialization (2026-11-23 gece, Claude) — TAM DERLEME

⚠ **YAYGIN STRUCT DEĞİŞTİ:** `RasterAppliedState` (+`material_spec_glass`; IBackend.h/VulkanBackend.h üzerinden her yerde)
ve `rtapi::RasterTimingInfo` (RtApi.h). Artımlı derleme eski .obj bırakırsa `frame_timings` okumasında erişim ihlali
kod hatası gibi görünür → **TAM** Studio + Player derlemesi (zaten bekleyen tam derlemeyle aynı seferde).
`material_preview_frag`/`_covered` SPV dağıtıldı (`sdf_surface`/`volume` aynı kaynaktan yeniden derlendi, içerik
değişmedi). Eski exe ile güvenli: host 3 sabit gönderir, cam sabiti (id 3) varsayılan `true` kalır.
malioc Mali-G52, workshop spec (terrain/graph/hava kapalı): cam açık 432 B stack · A 403 · LS 322 · T 302 →
cam kapalı **336 B · A 242 · LS 209 · T 236**. Tam açık 1376 B (değişmedi).
PCF `textureGather` (T2-3) denendi ve GERİ ALINDI: Mali'de +224 B stack, LS +24, bit-aynı da değil (plan notu).

1. **Önce: piksel kapısı (Windows Player, TAA kapalı, `three_point` ve `scene`, camsız sahne).** Önce/sonra karesi
   bit-aynı olmalı (fark 0): cam kapalıyken shader transmission/kabarcık/Interior Depth'i zaten 0 olan değerlere
   zorluyor. Fark varsa maske kanıtı ile shader kapısı ayrışmış demektir; önce `feature_spec.glass` değerine bak.
2. **`viewport.frame_timings` → `applied.feature_spec.glass`.** Camsız sahnede (`workshop`) `false`. Studio'da bir
   malzemeye `transmission>0` verince bir sonraki karede `true`, `rebuilds` +1, log
   `[MaterialSpec] material pipeline rebuilt for ...glass`. Alan yoksa → eski exe/eski obj.
3. **Cam açılınca görüntü, altı giriş ayrı ayrı:** transmission skaler, transmission dokusu, opaklık<0.99 (eski cam),
   kabarcık (`bubble`), Interior Depth (`transmission_density`), Material Graph. Her biri ilk karede cam çizmeli.
   ★ **SİNSİ:** maske birini kaçırırsa o malzeme hatasız **opak** çizilir ve "makul" görünür. Kaçmaya en yatkın
   ikisi: yalnız opaklığı düşük eski cam ve yalnız Interior Depth girilmiş reçine.
4. **Camı kaldır → ~240 kare sonra** `glass=false` ve bir rebuild daha. Hemen düşmemesi doğru (debounce).
5. **Lenovo/Mali (APK, sonra):** camsız `workshop`'ta `main_pass` ms önce/sonra.

## ★★ T2-1 adım 2: Android swapchain sunumu (2026-11-23 akşam, Claude + Sonnet alt ajan) — Windows ARTIMLI + APK

Android Player artık Windows ile aynı yoldan sunuyor: `PlayerSwapchainPresenter` (Vulkan swapchain + ImGui_ImplVulkan),
SDL_Renderer yok, kare başına 4 kopya yok. Android'e özgü kısım:
(a) `preTransform=IDENTITY`. Telefonun doğal yönü dikey olduğundan döndürmeyi compositor yapıyor; bu yüzden gelen
`VK_SUBOPTIMAL_KHR` yeniden kurulum TETİKLEMEZ.
(b) FIFO.
(c) `SDL_APP_*BACKGROUND` → swapchain bırakılır; `DIDENTERFOREGROUND` → yeni native window için VkSurface yeniden
kurulur.
(d) "GPU desteklenmiyor" ve device lost artık yerel diyalog; son 20 hata/uyarı satırını (kırıntı izi dahil) içerir.
Diyalog kapanınca uygulama çıkar.
NDK arm64 -fsyntax-only: değişen 13 dosya PASS. MSVC /Zs: 10 dosya PASS, VulkanBackend.cpp kontrol ortamında
ffx başlık yolu yüzünden koşmadı (yalnız koruma satırları değişti).

1. **Windows Player artımlı derle (önce).** Açılış, F2 tıklama, F11, boyutlandırma: bir önceki partiyle aynı davranmalı.
   Değişen Windows kodu: PlayerMain'in başlangıç-notu sınıflandırması (`classifyStartupNotice`). Bozuksa: Windows'ta
   sahne hatası notu görünmüyor ya da çift görünüyor.
2. **APK (`-Scenes pastel-paint-workshop`) → Lenovo tablet.** Oyun açılır, görüntü eskisiyle aynı. Kütüphane küçük resimleri,
   logo, çocuk başlatıcısı görünür; dokunma çalışır. `viewport.frame_timings.present_path=swapchain`,
   `frame_telemetry.host_read_ms≈0`. Ana ekrana çık + geri dön, ekranı kapat + aç: görüntü geri gelmeli (yüzey
   yeniden kurulumu). Bozuksa (siyah ekran, ses devam ediyor): `onForeground` sonrası yüzey kurulamıyor → logcat
   "Player swapchain" / "surface".
3. **Telefonda (doğal yönü dikey) döndürme.** Görüntü yatay ve doğru yönde olmalı, ters ya da yan değil.
   LandscapeLeft ↔ Right çevir. FPS düşmemeli: her kare yeniden kurulum olursa takılır (SUBOPTIMAL kapısı).
4. **Omix.** Device lost tekrarlarsa artık ImGui ekranı değil, yerel diyalog çıkar. Diyalogdaki `[GpuBreadcrumb]`
   satırlarının ekran görüntüsünü iste (`one_shot ... FIRST UNRETIRED` satırı asıl hedef).

★ En sinsi: tablette (doğal yönü yatay) her şey doğru görünür, telefonda döndürme yanlış ya da kare başına yeniden
kurulum çıkar. 2 ve 3 AYRI cihazlarda doğrulanmalı. Tablet PASS'i telefonu kanıtlamaz.
★ Android'de readback yalnız capture için; `viewport.get_screenshot` hâlâ çalışmalı (A6 ölçüm betikleri buna bağlı).

## ★ T2-1 Windows: F2 konsolu girdi almıyor — DÜZELTİLDİ (2026-11-23 akşam, Claude) — ARTIMLI (tek .cpp)

Belirti (kullanıcı + canlı tekrar): F2 konsolu görünüyor ama tıklama ve üzerine gelme çalışmıyor; önce tam ekranda,
sonra pencere modunda da. Canlı ölçüm: `console.status` tıklamadan önce de sonra da `page=games`; düğme rengi
üzerine gelince değişmiyor; ekran güncelleniyor (sunum çalışıyor).
Kök: `PlayerSwapchainPresenter::clearSwapchain()` her swapchain yeniden kurulumunda `ImGui_ImplVulkan_Shutdown()`
çağırıyordu. Bu ImGui'de (docking) Shutdown → `DestroyPlatformWindows()` ana viewport'un `PlatformHandle`'ını NULL
yapıyor. `imgui_impl_sdl2` her olayı pencere kimliğiyle `FindViewportByPlatformHandle` üzerinden arıyor, bulamayınca
olayı atıyor. İlk boyut / tam ekran / küçültme değişiminden sonra girdi kalıcı olarak ölüyordu.
Düzeltme: ImGui arka ucu ve UI render pass'i swapchain yeniden kurulumunda KORUNUYOR, yalnız yüzey biçimi değişirse
yeniden kuruluyor. O yolda ve kapanışta (`releaseUiRenderer`) ana viewport handle'ı geri yazılıyor. MSVC /Zs PASS.

1. **Player'ı artımlı derle → aç → F2 → bir düğmeye tıkla.** Çalışmalı.
2. **F11 tam ekran ↔ pencere, pencereyi boyutlandır, küçült/geri aç → F2'de tekrar tıkla.** Her adımdan sonra çalışmalı.
   Bozuksa: yeniden kurulumda ImGui yine kapanıyor (log'da "Player UI render pass" tekrarları).
3. **Kütüphane küçük resimleri ve logo** yeniden kurulumdan sonra da görünmeli (eskiden ImGui ile birlikte yok oluyordu).

★ Sinsi: konsol GÖRÜNÜR olduğu için "donmuş" sanılır; asıl ölçüt `player.operator console.status` sayfasının tıklamayla
değişmesidir.

**Ayrı gözlem (kullanıcı, iş açılmadı):** 720p tam ekranda boşta ~170 FPS (CPU kare yuvası bekliyor); boyama
başlayınca bekleme kayboluyor ve FPS ~80'e iniyor. Yani boyama sırasında darboğaz CPU. Codex'in ölçümü:
`advance` ~10 ms = sim tick 5,5 + billboard yükleme ~3 ms. Sıradaki optimizasyon adayı.

## ★ Omix 2. tur: kırıntı izi genişletildi + Mali stack düzeltmesi (2026-11-23 akşam, Claude) — ARTIMLI derleme

**Omix sonucu (1. iz):** `raster_frame` 2. karenin 8/8 aşamasını, `sim_compute` 21. submit'in 20/20 dispatch'ini
bitirmiş. İzlenen işlerin HİÇBİRİ asılmadı. Kayıp ya tampon kuyruklarında (raster readback kopyası, sim indirmeleri)
ya da izlenmeyen bir submit'te. Akustik geçiş elendi (TLAS ister, Mali'de yok).
**Eklenen:** (a) `one_shot` izi: `endSingleTimeCommands`/`Async` (~160 çağrı yeri: yükleme, mip, IBL, sıkıştırma),
etiket = çağıran fonksiyonun adı (Android `dladdr`+demangle; Windows'ta adres). (b) Raster ve sim tamponlarının
SONUNA `<end>` işareti: "o submit BİTTİ" ile "kuyruğu bilinmiyor" artık ayrılıyor. (c) Rapor ilk bitmemiş
submit'in etiketini basar (`FIRST UNRETIRED`). NDK arm64 -fsyntax-only PASS (5 dosya). MSVC /Zs çalıştırılmadı.

**Shader (aynı APK'ya girer):** `material_preview_frag` + `_covered`, Mali stack. Workshop yapılandırmasında
704 → 432 B (spill 424 → 376, alloca 280 → 56); tam yapılandırmada 1664 → 1376 B. Değişiklik iki parça:
- Stüdyo/matcap ışık dizileri (`lightDirs[3]` …) yardımcı fonksiyonlara alındı.
- Dispersiyonda dinamik vektör indeksi yerine seçim (`channel==0 ? v.x : …`) kullanıldı.

Formüller aynı. SPV üç yere dağıtıldı (source, x64/Release/shaders, shaders_vk11).

1. ✔ **PASS (2026-11-23, Claude, workshop 1280×720, TAA kapalı).** `three_point`: eski ve yeni SPV aynı anda alınan kare **921.600/921.600 piksel bit-aynı**. `scene`: sahne canlı (aynı çalışmada 1 sn arayla kare farkı ort. 13). İki çalışmada da hareketsiz kalan 489 bin pikselin 25'i en fazla 2 seviye farklı; kümeler boya/parçacık bölgesinde. Dispersiyon yolu bu sahnede çalışmıyor (`dispersion` alanı yok), doğruluğu kod okumasına dayanıyor. Eski metin:
   **Windows piksel kapısı (en önce, C++ derlemesi gerektirmez).** Player'ı kapatıp eski SPV ile bir kare,
   yeni SPV ile aynı kare alın (`git stash` gerekmez: eski SPV = HEAD'deki `material_preview_frag.spv`).
   `lighting_preset` 0 (stüdyo) ve 2/diğer (matcap) ile ayrı ayrı alın, çünkü değişen kod yalnız bu ön ayarlarda çalışıyor.
   Beklenen: fark 0. Fark varsa yalnız ışık yönünde olur: `normalize(sabit)` derleme anı ile çalışma anı farkı. SPV'yi geri alın, söyleyin.
2. **Windows regresyon.** Studio + Player açılır, logda `[GpuBreadcrumb]` YOK (NVIDIA device_fault'lu).
   ✔ **PASS Player (2026-11-23, Claude, 16:06 exe, RTX 3060, workshop 1280×720, 10 sn pencere):** normal çalışmada
   `[GpuBreadcrumb]` satırı yok, ERROR 0. `RAYTROPHI_GPU_BREADCRUMBS=1` ile üç iz açılıyor (`one_shot`, `sim_compute`,
   `raster_frame`), ERROR 0. İz maliyeti: raster GPU ort. 6,82 → 6,95 ms (+0,13 ms, ~%2), p95 7,90 → 8,05 ms.
   Aşamalar: main_pass 2,45, transmission 2,54, rt_shadow 1,10 ms. Studio açılmadı.
3. **Omix APK.** `-Scenes pastel-paint-workshop` ile paketle, gönder. Kayıpta beklenen satırlar:
   `raster_frame ... 'readback_copy <end>'; that submission FINISHED` (veya kuyrukta takıldı),
   `sim_compute ... '<end>'`, ve asıl yeni satır `one_shot: ... FIRST UNRETIRED submission N '<fonksiyon adı> <end>'`.
   O fonksiyon adı suçlu submit'i söyler (ör. `VulkanRT::VulkanDevice::generateMipmaps`, IBL ön süzme).

★ En sinsi: üç iz de "FINISHED" derse kayıp izlenmeyen kalan yoldadır: `submitDenoiserCopyAsync`,
`submitTraceTonemapAsync`, bulut parametre submit'i, sim kopya submit'i. "İz temiz, sorun yok" diye okumayın.

## ★ Omix X700 device lost: GPU kırıntı izi (2026-11-23, Claude) — ARTIMLI derleme

Belirti: Omix X700 (Mali-G76, Vulkan 1.1) 13:01 APK'sında `VK_ERROR_DEVICE_LOST`, "observed at
RasterViewportFrameRing.cpp:489". Bu yalnız kaybı FARK EDEN fence beklemesi; sebep değil. Mali'de
`VK_EXT_device_fault` yok. Yeni `Backend/VulkanGpuBreadcrumbs` her geçiş sınırında tam bariyer +
`vkCmdFillBuffer` ile host-coherent bir sözcüğe "buraya kadar bitti" yazar; kayıpta hata ekranına
`[GpuBreadcrumb] raster_frame: ... STUCK IN the work between 'X' and 'Y'` ve aynısını `sim_compute`
(dispatch etiketiyle) için basar. Yalnız device_fault OLMAYAN cihazda açılır (RTX'te kapalı);
`RAYTROPHI_GPU_BREADCRUMBS=1/0` zorlar. Yeni .cpp: Studio vcxproj + Player/Android üretildi.
NDK arm64 -fsyntax-only PASS (4 dosya).

1. **Windows regresyon (en hızlı).** Studio + Player normal açılmalı; logda `[GpuBreadcrumb]` satırı
   OLMAMALI (NVIDIA device_fault'lu). Varsa kapı yanlış: cihaz fault kaydına girmemiş.
2. **Windows zorlamalı.** `$env:RAYTROPHI_GPU_BREADCRUMBS=1` ile Player'ı aç: logda iki `trail on`
   (raster_frame, sim_compute), görüntü ve boya aynı. Bozuksa (siyah/titreme) bariyer bir render pass
   içinde kaydediliyor demektir — validation'da `vkCmdFillBuffer ... inside render pass` arayın.
3. **Lenovo Mali (device_fault yok → iz AÇIK).** Kare süresi önceki ölçüme göre belirgin artmamalı
   (raster ~10 bariyer/kare, sim dispatch başına 1). Belirgin artış = izin maliyeti; ölçün, söyleyin.
4. **Omix APK (asıl hedef).** `.\scripts\Build-PlayerApk.ps1 -Abi arm64`, arkadaşa gönder. Kayıp
   tekrarlarsa ekran görüntüsündeki `[GpuBreadcrumb]` iki satırı sebebi daraltır:
   - `raster_frame ... between 'frame_start' and ...` → o karenin ilk işi; `never reached its first mark`
     → kareden ÖNCEKİ iş (çoğunlukla sim_compute satırına bakın).
   - `between 'TransmissionPrep' (veya 'ScreenGiFilter') and ...` → HDR sahne render pass'i
     (sky/depth/main/volume) — en olası aday ana malzeme shader'ı.
   - `sim_compute ... between 'A' and 'B'` → B etiketli compute çekirdeği.

★ En sinsi: iz satırı HİÇ çıkmazsa (yalnız eski `[DeviceFault] no device ...` satırı) APK eski
derlemedir ya da iz açılamadı (`trail off` uyarısına bakın) — "kırıntı boş" diye OKUMAYIN.

## Canlı billboard vertex tamponları (2026-11-23)

1. Ortak Vulkan başlığı değiştiği için Windows Player ve Studio'yu yeniden
   derleyin. Aynı 1280×720 Jolt açık sahnede compute timestamp kapalıyken
   11 patlama ve ıslak duvar penceresini tekrar ölçün.
2. `viewport.particle_billboards.vertex_write` kare yuvasına yazılan CPU
   quad verisinin duvar saati süresini verir. Önceki yoğun pencerede
   `viewport.particle_billboards.upload` 327 çağrı / 163 iterasyonda
   2,97 ms/iterasyondu. Yeni `upload` sayacı küre ve LUT gibi paylaşılan
   tamponları kapsar. Kare kaydı maliyetine taşınan `vertex_write` süresini
   karşılaştırmaya dahil edin.
3. Alpha ve additive efektlerin birlikte görünmesini, son efekt söndüğünde
   kadrajda eski quad kalmamasını ve yeni patlamada kapasite artışını kontrol
   edin. Pencereyi yeniden boyutlandırın, başka sahne açın ve canlı efekt
   varken capture ile senkron sunumu deneyin. Studio Solid/Matcap'te küre
   impostor ve device-pulled parçacıkların görünümünü de kontrol edin.
4. `player.advance.simulation_tick` içinde `simulation.balloon_demo`,
   `simulation.timeline` ve `simulation.wet_paint` CPU sürelerini ayrı
   okuyun. Bunlar tick toplamına dahildir. `paint.wet.stats_collect`
   ise `simulation.wet_paint` içinde gerçekleşir.

## Islak boya ve billboard bekleme ölçümü (2026-11-23)

1. Kullanıcı Player Release x64 artımlı derlemesinden sonra Gizli Renkler'i
   1280×720'de açın. Jolt ayarını sabit tutun ve FPS sınırını kapatın.
2. `scripts/test/Measure-PlayerGpuPasses.ps1 -SkipComputeTiming` ile boşta
   ve arka arkaya 11 patlamada ölçün. Bir sonraki pencerede patlatmadan,
   duvarlar hâlâ ıslakken aynı ölçümü alın.
3. `player.advance.simulation_tick` ortak simülasyon tick'inin CPU duvar
   saatini ölçer. `viewport.particle_billboards.build` CPU'da çizim
   verisinin hazırlığını, `viewport.particle_billboards.upload` gerçek
   tampon yükleme çağrılarını ölçer. Bir karede birden fazla yükleme
   olabildiği için `TotalMs` üzerinden karşılaştırın.
4. `paint.wet.stats_collect` önceki boya istatistiklerinin fence beklemesi
   ve hosta kopyasını, `viewport.particle_billboards.drain` billboard
   tamponu veya descriptor yazımından önce uçuşta olan render işini
   boşaltma süresini ölçer. `perf.list` ve `rt.perf.list()` aynı
   kayıtları verir. Betikte `Samples` ve `TotalMs` pencere farkıdır.
   CPU sayaçları oturum boyunca biriktiğinden pencere p95/tepe alanları boştur.
5. `paint.wet.stats_collect` simülasyon tick'inin içindedir. İkisini toplamayın.
   Boşta sayaçlar hiç çağrılmadıysa boş değer beklenir. CPU duvar saati
   sürelerini GPU timestamp süreleriyle toplamayın. GPU çekirdek tablosu
   için betiği `-SkipComputeTiming` olmadan ayrı çalıştırın.

## Islak boya channels aktif tile listesi (2026-11-23)

1. Ortak `ComputeDispatch` düzeni değiştiği için Studio/Player'ı yeniden
   derleyin. Vulkan 1.2 shader source ve Release dizinlerinde, Vulkan 1.1
   shader Android staging dizininde hazır. Android APK'yı güncellerken
   staging'i çalıştırın.
2. 1280×720'de aynı kamera, Jolt ve boya ayarlarıyla az boyalı ve çoğu
   duvarı hâlâ ıslak sahneleri ayrı ölçün.
   `scripts/test/Measure-PlayerGpuPasses.ps1` çıktısında
   `paint_wet.channels_compact` + `paint_wet.channels` toplamını önceki
   `paint_wet.channels` ile karşılaştırın. Yalnız `channels` satırına
   bakmak liste hazırlama maliyetini gizler.
3. Damlalar tile kenarını geçerken roughness, normal, kabartı ve opaklığı
   gözleyin. Boyayı temizleyip yeniden patlatın. Sonra boya dururken height
   veya normal kanalını değiştirin. Zorunlu kanal yenilemesi tüm atlası
   güncellemeli. Hatada `dispatch_failed` ve `device_lost` durumuna bakın.

## Balon patlamasında damla slotları (2026-11-23)

1. Güncel Player ile Gizli Renkler sahnesinde önce tek balon, sonra ısınmış
   sahnede yaklaşık 300 ms arayla 11 balon patlatın. Full HD karşılaştırmasında
   `spawn_ms`, `spawn_timing.particle_enqueue_sample_ms / sampled_drops` ve
   en uzun kare aralığını izleyin. Önceki ısınmış tepe 20,95 ms ve bu olayın
   örneklenen damla başına CPU kuyruğa ekleme süresi 3,74 µs idi.
2. Sürpriz balonda çoklu renkleri, peş peşe patlamada damla sayısını ve
   damlalar kaybolduktan sonra yeniden patlatıldığında boya ve efektleri
   kontrol edin. Slot sırası, CPU doğrulaması ve GPU resident yazıları
   standart tekli spawn yoluyla aynı olmalı.
3. Yalnız Gizli Renkler içeren imzalı APK için
   `scripts/Build-PlayerApk.ps1 -Scenes pastel-paint-workshop` çalıştırın.
   Tek sahne seçimi APK varlık listesini ve sahne manifestini birlikte üretir.

## Android Vulkan hata ekranı ve ekran görüntüsü (2026-11-23)

1. Omix X700’da yeni APK ile aynı sahneyi açın. Vulkan aygıtı kaybolursa HUD
   kapalıyken de `VK_ERROR_DEVICE_LOST` ve sürücünün hata döndürdüğü işlem
   ekranda görünmeli. İşlem bilgisi varsa mevcut Player runtime durumunda
   `device_lost_reason` olarak da okunur.
2. Hata ekranında ekran görüntüsü tuşlarını kullanın. Uygulama odağı geçici
   kaybolup geri geldiğinde duraklatma menüsü ekranı örtmemeli. Android Geri
   tuşu hata ekranının üstüne oyun menüsünü açmamalı.
3. Sağlıklı bir sahnede Geri ile menüyü açın ve Devam ile oyuna dönün.
   Desteklenmeyen GPU’da ret mesajının ve dokunarak açılan log bölümünün
   okunabildiğini kontrol edin.

Bu kabul, Vulkan aygıt kaybının altında yatan sürücü veya GPU sorunundan
bağımsız olarak hata ekranını sınar.

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
