# Vulkan RT + RayFusion: hareket halinde hız — denoise, temporal, upscale planı

> **Durum:** AKTİF — 2026-10-27. B1–B4b CANLI (B4b: kullanıcı duruş birikimi "çok kaliteli"; canlı IPC: hareket 20/20 FSR active, input svgf). **B4c yazıldı, DERLENMEDİ** (yalnız C++): sahne animasyonu SVGF/FSR için hareket.

## DEVİR NOTU — B4c (2026-10-27)

**ÖNCE:** `NEXT_BUILD_CHECKS.md` en üstündeki B4c listesi. Yalnız C++.
Kök (skin sahnesi, canlı): kamera dururken animasyon her karede birikimi sıfırlıyor,
SVGF bunu duruş sanıyor → 1 spp, geçmiş her karede siliniyor (ortalama 1.0),
11.8 ms/kare tam çözünürlük, FSR kapalı. Nesne hareketi algılaması
(`motionInstanceSerial` ↔ `m_motionSeenInstanceSerial`) policy anında hep eşitti:
ikisi aynı karenin render'ında ilerliyordu (üretici ≠ tüketici). Düzeltme:
`VulkanRealtimeDenoiserAdapter.cpp` — `submitRealtimeTrace` kamera dururken art
arda birikim sıfırlayan trace'leri sayar (`m_sceneResetStreak`), sonraki karenin
`configureRealtimeDenoiser`'ı ≥2'de "scene" hareketi verir. FSR policy'si SVGF'nin
`moving`'ini okuduğu için kendiliğinden izler. Tek değişiklik (slider, tek
transform, FSR extent geçişinin kendi reset'i) eşiğe takılmaz. Ayar
`scene_motion` (vars. true), stats `motion_source` / `scene_reset_streak`;
IPC/Python/panel/overlay. Açık soru: deforme mesh'te rijit hız yaklaşımı iz
bırakıyor mu (kontrol 5); bırakıyorsa hız sözleşmesi işi.
İkincil gözlem (ölçüldü, kaynağı aranmadı): skin sahnesinde epoch SVGF
çalışması başına ~4 artıyor → kare başına birden fazla `resetAccumulation`.

## DEVİR NOTU — B4b kaynak partisinden sonra (2026-10-27)

**ÖNCE:** `NEXT_BUILD_CHECKS.md` en üstündeki B4b listesi. Shader (`realtime_svgf.comp`,
iki varyant) + C++ derlemesi gerekir; glslangValidator sözdizimi denetimi iki
varyantta temiz, `.spv` üretilmedi. IPC audit PASS, descriptor üretildi.

**Teşhis (canlı, iç mekan yatak odası, bu oturum):** 0.01° salınımla 16 kare,
zamansal sapma: ham 1 spp sd 0.107–0.110; SVGF sd 0.008 (kolon) / 0.012 (tavan)
ama sapmanın **%66–93'ü 8×8 blok** ölçeğinde (ham: %12) → "yüzen leke" =
SVGF ince gürültüyü büyük lekeye çeviriyor. skew ≤ 0 ve firefly_clip 1.0
iyileştirmedi → ateşböceği DEĞİL. atrous 5 etkisiz → luminance edge-stop
geç geçişlerde daralıyor. Kalıcı çözüm girdi varyansı (radyans önbelleği /
ReSTIR GI). Makaleden sapma: Schied 2017 geçmişe ilk à-trous çıktısını geri
besler, bizimki ham temporal rengi saklıyor (`temporal()` nextColor) — ucuz
kaldıraç, bu partide YAPILMADI (Sonra listesi).

**1. Duruş sönmesi (kök: `rawBlend = sample/64` doğrusal).** Ekrandaki gürültü
≈ σ·√N/64 → örnek arttıkça BÜYÜYORDU; 64'ten sonra ham 64+ spp kalıyordu.
Yeni: `still_noise` (göreli standart hata, vars. 0.02; `static_samples` SÖKÜLDÜ,
ad değişti). Shader raygen'in Welford M2'sini (binding 15) okur; aynı materyal
3×3 ortalamasıyla RSE ≤ τ → ham, ≥ 2τ → filtreli, piksel başına. 8 örnek altı
M2'ye güvenilmez. Son çeyrekte ev sahibi rampası tabanı (target-1'de sıçrama
yok). Filtreli piksel sayısı slot başına sayaç (binding 16, eşlenmiş), slot
fence'inden sonra okunur; ≤ %0.1 → `still_complete`, duruş SVGF'si o epoch için
durur (temiz sahne bedava). Stats: `still_filtered_fraction`, `still_complete`.

**2. Sağlayıcı sözleşmesi.** raw yedek/geri yükleme SVGF'den
`Backend/VulkanRawGuard.*`'a taşındı (yeni .cpp, vcxproj'da). Adapter:
guard.save → [SVGF.begin varsa] → FSR/köprü → tonemap → guard.restore.
SVGF artık **isteğe bağlı sağlayıcı**: kapalıyken FSR ham birikimi büyütür.
Kalkan kapılar: `setUpscale` SVGF şartı, `setRealtimeDenoiser` "önce upscale
kapat", uygunlukta "SVGF is disabled". Yeni: `upscale_stats.input_provider`
("svgf"/"none"), panelde "Input:". Delik kapatıldı: hareket AOV'u artık FSR
açıkken de kapatılamaz (eskiden yalnız SVGF kuralıyla dolaylı korunuyordu).
Gelecek sağlayıcı (kare başı OIDN vb.) aynı guard altında raw'a compose eder
ve `m_upscaleInput`'u yazar; OIDN'in Vulkan↔OIDN harici bellek köprüsü YOK.

**Sonra:** filtreli geçmiş geri beslemesi (Schied); iç mekan varyans azaltma;
RayFusion için FSR (TAA'nın yerine; raster derinlik girdisi, matriste jitter,
overlay sırası) — ayrı plan bölümü.

> **B4 Auto geçişi çökme — 2026-10-27:** Kullanıcı ffxCreateContext içinde
> DLL → null function çağrısı bildirdi. SDK 1.1.4 kaynak denetimi, backend'in
> vkGetBufferMemoryRequirements2KHR adresini yüklediğini doğruladı; uygulama bu
> extension'ı açmıyor (Vulkan core sürümünü kullanıyor). FSR'ye verilen resolver,
> KHR adresi yoksa ABI eşdeğeri core vkGetBufferMemoryRequirements2'ye dönüyor.
> DLL çağrısından önce SDK'nin gerekli device komutları kontrol ediliyor;
> eksik komut durumunda adıyla tam çözünürlük fallback yapılıyor.
> Kök neden canlı debugger ile henüz doğrulanmadı. C++ Release yeniden derlenip
> uygulama yeniden açılarak Auto/Pro hareket testi yapılmalı; shader değişmedi.


> **B4 ilk canlı kontrol — 2026-10-27:** Kullanıcı derledi. 1920×1080,
> Auto / Ultra Performance ile durağan durumda giriş=çıkış (beklenen).
> Harici IPC kamera hareket turunda da active=false, dll_loaded=false,
> dispatches=0; genel uygunluk engeline takıldı. Kamera tur sonunda geri yüklendi.
> Kaynak denetimi: camera_mode==0 koşulu varsayılan Pro (1) modunu da dışlıyordu.
> Auto+Pro desteği ve engelleyen koşulu ayrı ayrı bildiren inactive_reason eklendi.
> Bu düzeltme için kullanıcı C++ Release derlemesi gerekir; FSR GPU kabulü henüz PASS değil.


## DEVİR NOTU (ARŞİV) — B4 kaynak partisinden sonra (2026-10-27)

> B4 o zamandan beri derlendi ve CANLI; aşağıdaki "SVGF şartı" ve "FSR açıkken
> SVGF kapatılamaz" sözleşmeleri B4b'de söküldü (yukarıdaki devir notu).

**ÖNCE:** `NEXT_BUILD_CHECKS.md` en üstündeki B4 listesi. Kod yazımı ve SDK hazırlığı
bitti; bu ajan C++/shader derlemesi, DLL çalıştırma veya uygulama açma yapmadı.
Canlı testler yalnız kullanıcının derlemesinden sonra, açık uygulamaya dış
`scripts/test/rt_ipc.py` ile tek turda yapılır. SDK kaynakları derlenmez.

**Checkpoint:** B1–B3 + gaz işleri `8c7be7f9` commit'inde. B4/SDK değişiklikleri
henüz commit edilmedi. Eski yedi devir notunun arşivi dosyanın sonunda korunuyor.

**Hazır SDK:** FidelityFX SDK 1.1.4 / resmî Vulkan `amd_fidelityfx_vk.dll`;
AMD Authenticode Valid, x64 ve 5 FFX export'u statik doğrulandı. Başlıklar,
MIT lisansları ve hash manifesti `RayTrophiStudio/third_party/FidelityFX/1.1.4/`.
DLL+lisans **x64/Release** ve mevcut Debug'a kopyalı. `FidelityFX.targets`
kullanıcı derlemelerinden sonra `$(OutDir)`'e kopyalar. DLL ve ZIP Git-ignore'da;
yeni checkout: `powershell -NoProfile -File scripts/setup_fidelityfx.ps1 -Download`.
DLL file version 1.0.1.41314, FSR provider sürümü değildir; provider API'den sorgulanır.

**Gerçek kaynak haritası (onaylı §6 planı uygulandı):**
- `Backend/UpscaleTypes.h`, `VulkanFsrLoader.*`: tipler, kesin exe dizininden güvenli
  dinamik DLL yükleme, export kontrolü ve mantıksal Vulkan cihaz özellikleri.
- `Backend/VulkanFsrUpscaler.*`: context/quality/jitter query, GPU timestamp,
  hareket/extent politikası ve etiketli duruş köprüsü. Plandaki ayrı Policy/History
  dosyaları bu odaklı (<2000 satır) modülde toplandı.
- `Backend/VulkanUpscaleInputs.*` + `shaders/upscale_inputs.comp`: depth/velocity/
  reactive dönüştürme, full-res compose ve eski pozisyon görüntüsü.
- `Backend/VulkanUpscaleAdapter.cpp` + mevcut `VulkanRealtimeDenoiserAdapter.cpp`:
  sunum bağlantısı, slot/fence/epoch geçişleri. Büyük backend dosyasında yalnız
  özellik/çağrı/trace-extent/kaynak yaşam döngüsü bağlantıları var.
- `Api/RtApiUpscale.*`, `RtIpcUpscale.*`, `RtUpscaleBindings.*`,
  `UI/UpscaleControls.cpp`: ortak servis, IPC, Python, panel; capability/overlay/
  descriptor üretimi ve vcxproj/filters kayıtları tamamlandı.

**Sözleşmeler:**
- `render.set_upscale {mode:off|fsr|auto, quality:quality|balanced|performance|
  ultra_performance, sharpness:0..1}`; atomik doğrulama, atlanan alan korunur.
  Oturumluk; varsayılan **off / quality / 0.2**. Kabul sonrasında auto kullanıcı
  tarafından açılır; bu kaynak partisinde varsayılan sessizce değiştirilmedi.
- `render.upscale_stats {synchronize:true|false}`; true GPU bekler, false poll.
  available/active ayrı; reason hata/kapı, inactive_reason bilerek tam çözünürlük.
  Context ilk uygun hareket karesinde tembel kurulur; duran karede ilk sorgunun
  available=false olması tek başına donanım uyumsuzluğu değildir.
- Yalnız görünür Vulkan Rendered + standart perspektif/lens + SVGF + motion AOV.
  Final/debug/ortho/distortion/CA/shake/raster tam çözünürlük; bunların son moving
  frame'inden kalan active bayrağı raster paneline taşınmaz. SDK kapısı mantıksal
  storage-image özellikleri, Vulkan >=1.2, formatlar ve context sonucuna bakar.
- Kapasite full display'dir; RT/SVGF aktif düşük dikdörtgende, FSR/tonemap/readback
  full boyutta. Motion sample özeti yalnız aktif RT dikdörtgenini okur.
  Her extent geçişinde eski frame slot'ları beklenir/atılır; geçmişler sıfırlanır.
- FSR query jitter'ı B2'nin authored Halton tercihini değiştirmeden zorlanır.
  B2 hızları **render pikseli** olduğundan motionVectorScale=(1,1); normalized
  vektör örneklerinin renderW/H çarpanı tekrar uygulanmaz. Yön/ölçek canlı kabulde
  1° dönüşle doğrulanacak; kaynak incelemesi görüntü kabulü değildir.
- R32 depth=near/viewZ, sky=0; ters-Z/sonsuz bayrakları. SDK'nın ters derinlik
  sırasıyla cameraNear=FLT_MAX, cameraFar=0.01. Reactive B2 z → R8, <=0.9.
- B3 mevcut kodu radyansı scene-linear tutuyor (`pushConst.exposureFactor=1`);
  FSR **preExposure=1**, kamera pozlaması yalnız mevcut display post'ta.
- FSR, SVGF'nin compose ettiği HDR'ı raw restore'dan önce tüketir; full-res
  çıktısı aynı raw sunum hedefine compose edilir, tonemap sonrası raw yedekten
  aynen geri gelir. Duruşta aynı poz/epoch ve uyumlu yüzeyde dondurulmuş FSR
  görüntüsü 64 örneğe kadar söner; 128'de katkı yok. Slider/resize/viewport
  değişiminde eski köprü taşınmaz. Target=1 için eski düşük görüntüde takılmayı
  önleyen bir tonemap-only son sunum var; 128 örnek native davranışı korunur.
- gpu_ms: inputs+FSR+compose+pozisyon kopyası GPU zamanıdır. svgf_filter_ms:
  temporal+variance+atrous. **SVGF total_ms'i ayrıca ekleme**: dış submission
  zaman aralığı FSR'ı da içerir. İlk shader/context kurulumunu hız kabulüne katma.
- DLL/context/shader/dispatch hatası reason ile full-res fallback. Hatalı düşük
  çözünürlüklü slot yayımlanmaz. Hata sonrası yeniden denemek için mode off/on.
  FSR açıkken SVGF'yi kapatmak reddedilir; önce upscale off.

**Bekleyenler:** kullanıcı derlemesi; soğuk DLL-yok kapısı; GPU süre/FPS, jitter,
ince çizgi/hayalet, gaz reactive, duruş ve bit-bit 128 referans. Hiçbiri PASS
olarak yazılmadı. Statik kontrol sonucu `NEXT_BUILD_CHECKS.md` B4 bölümüne yazılır.
Yeni bulgular §7'ye; B3 Sonra mikro işleri bu partiyi büyütmez.

## 0. Kapsam ve kural

**Hedef:** hareketli kamerada ve yoğun sahnelerde (iç oda, çok bounce,
hacim) etkileşimli kare hızı — durgun karede bugünkü kaliteyi KAYBETMEDEN.

**Kural (derleme ekonomisi):** proje büyük, derleme uzun. Plan dört derleme
partisine bölündü; her parti kendi içinde biter, kendi IPC yüzeyini getirir
ve sonunda tek bir IPC ölçüm turuyla kabul edilir. Parti ortasında
"bir derleyip bakalım" yok. Yeni bulgu partiyi büyütmez, **Sonra** listesine
gider (§7).

**Değişmez:** ilerlemeli birikim (progressive accumulation) referans olarak
kalır. Yeni her şey birikimin ÜSTÜNE görüntü tarafında oturur; durgun kamera
128 örneğe yakınsadığında görüntü bugünküyle aynı olmalı. OptiX dondurulmuş
(CLAUDE.md §6): yeni yetenekler yalnız Vulkan'a.

## 1. P0 — ölçülen taban (2026-10-26, CANLI)

İç oda, 1680×945, Vulkan Rendered, RTX 3060 (10de_2503), 2 ışık, hacim/bulut kapalı.

| Ayar | RT dispatch ms/örnek |
|---|---|
| bounce 1/1/1 | 10.8 |
| bounce 2/2/2 | 31.3 |
| bounce 4/4/4 | 81.8 |
| bounce 10/4/8 (sahne ayarı) | 95.7 |

| Denoiser | RT ms/örnek | OIDN ms/çağrı | örnek/sn |
|---|---|---|---|
| kapalı | 95.7 | — | ~11.2 |
| fast | 110.7 | 43.0 | ~9.8 |
| balanced | 119.9 | 63.3 | ~9.0 |

Okuma:
- **Primary ucuz, ikincil bounce pahalı** (~20–25 ms/bounce). Hacim yolu
  bu sahnede hiç koşmuyor; maliyet yüzey path tracing'in kendisi.
- **OIDN her örnekte koşuyor** (birikim sürerken), RT ile örtüşüyor ama
  aynı GPU'yu paylaştığı için RT dispatch'ini +15/+24 ms şişiriyor.
  Boşta (birikim bitmiş) çağrı yok — doğrulandı.
- **Hareket halinde** her kamera değişimi birikimi sıfırlar → 1 örnek/kare,
  96 ms + 43 ms OIDN. Bu planın asıl hedefi bu durum.
- Russian roulette bounce ≥ 3'ten başlıyor, throughput tabanlı; açık
  renkli iç mekânda throughput yüksek kaldığı için nadiren keser.

Ölçüm aletleri hazır: `render.set_gpu_timing` + `render.gpu_timing_stats`
(RT dispatch), `render.denoiser_stats` (OIDN), `perf.get('loop.denoise')`.
Birikimi sıfırlamak için kamerayı 0.0001 kıpırdat.

## 2. Mimari karar: neden FSR İLK adım değil

Upscaler (FSR 2/3, TAAU) üç girdi ister: **hareket vektörü, derinlik,
gürültüsüz renk**. Bugün bunlardan yalnız derinlik/pozisyon var. 1 örnekli
gürültülü path-trace çıktısı upscaler'a girerse gürültüyü büyütür ve
hareket halinde iz (ghosting) bırakır. Doğru sıra:

```
1 spp RT (düşük çözünürlük)
  → temporal yeniden kullanım (hareket vektörü + geçmiş, B2)
  → gerçek zamanlı denoise (B3)
  → upscale (B4)
  → tonemap / post
durgun + yakınsadı → OIDN bir kez (mevcut yol, B1 ile seyreltilmiş)
```

Hazır olan altyapı (kodda doğrulandı): raygen zaten `denoiserAlbedoImage`,
`denoiserNormalImage`, `denoiserPositionImage` (dünya pozisyonu + materyal id)
ve `varianceImage` yazıyor; `VulkanVolumeTemporal` hacimler için ping-pong
geçmiş + reprojection (en iyi piksel araması) taşıyor; raster TAA önceki
kare VP matrisini tutuyor. **Eksik olan tek temel yapı hareket vektörü.**

## 3. B1 — ucuz kazançlar (1 derleme, C++ + küçük shader)

Altyapı gerektirmeyen, bugünkü yolda ölçülebilir kazançlar.

**B1a. OIDN takvimi.** Birikim sürerken her örneği denoise etmek gereksiz:
görüntü yavaş değişir. `render.set_denoiser` yeni alan
`schedule = every_sample | progressive | converged_only`:
- `progressive` (yeni varsayılan): örnek 1, 2, 4, 8, 16, … ve son örnekte
  denoise; aradaki karelerde son denoise sonucu gösterilir (önbellekten
  kopya, OIDN yok). 128 örnekte 43 ms × 128 yerine 43 ms × 8.
- Hareket halinde (örnek 1'de kalırken) davranış bugünküyle aynı; o durum
  B3'ün işi.
- `denoiser_stats` + `skipped_by_schedule`.
- Kabul: denoiser açık, birikim sürerken örnek/sn ≥ 10.8 (bugün 9.8);
  yakınsamış son kare `every_sample` ile aynı (piksel farkı ~0).

**B1b. Hareket kalite profili.** Kamera hareket ederken bounce bütçesi
düşer, durunca tam bütçeyle birikim baştan başlar. `render.set_motion_profile
{ enabled, max_bounces, diffuse_bounces, transmission_bounces, settle_ms }`.
P0 tablosu kazancı zaten veriyor: 10/4/8 → 2/2/2 ≈ 96 → 31 ms (3×).
Kural: profil `RenderSettings`'i DEĞİŞTİRMEZ, backend'e giden etkin değeri
değiştirir; `render.get_settings` iki değeri ayrı raporlar (authored /
effective) — panel yalan söylemesin.

**B1c. Roulette başlangıcı ölçüm düğmesi.** `rr_start_bounce` (varsayılan 3,
bugünkü davranış) IPC'ye. Kalite kararı değil, ölçüm: 2'de maliyet ve
gürültü (komşu piksel farkı) ölçülür, varsayılan ancak ölçümden sonra değişir.

**B1 uygulama notları (2026-10-26).**
- Takvim önbelleği son denoise edilmiş GÖRÜNTÜ karesini (8 bit, ~6 MB) tutar;
  geçerliliği `IBackend::accumulationEpoch()` ile ölçülür — örnek sayısı
  hareket halinde her kare 1'e döndüğü için sıfırlamayı göstermez.
- Hareket kamera POZUNDAN tespit edilir (`g_camera_dirty` değil: birden çok
  tüketicisi var). Timeline oynatımı kamera animasyonluysa profil sürekli aktif
  kalır — bilerek.
- RR düğmesi yalnız Vulkan (OptiX dondurulmuş).

**B1d (bu partide zaten yapıldı).** Birikim bitmiş kareye bounce/denoiser
düzenlemesi ulaşıyor (2026-10-26 CANLI PASS).

## 4. B2 — hareket vektörü + geçmiş (1 derleme, shader + C++) ★ temel

**B2'ye eklenen düzeltme (B1 canlı testinden, 2026-10-26) — ÖNCE BU:**
`getDenoiserFrameGPU` her çağrıda bir ÖNCEKİ çağrının kopyasını döndürüyor
(fence-deferred ping-pong). B1 takvimi çağrıları seyrekleştirince bu kopya
önceki kamera pozundan olabiliyor: converged_only'de son denoise karesi 6 px
eski pozda ölçüldü (kullanıcı "kamera hafif sağa dönmüş" diye gördü).
Düzeltme: staging kopyasını `(accumulationEpoch, sampleCount)` ile etiketle;
istenen kareyle eşleşmiyorsa senkron kopya + fence bekle (seyrek çağrıda bir
kez, ~kopya süresi), eşleşiyorsa bugünkü asenkron yol. Kabul: aynı test
(kamerayı 0.25 kaydır, converged_only) → denoise karesi ham kareyle 0 px
kayma; every_sample'da örnek/sn düşmüyor. Bu aynı zamanda B2'nin geçmiş
tamponları için de kural: **her geçmiş/önbellek girdisi hangi epoch ve
örnekten geldiğini taşır.**

**Hız (velocity) AOV:** `rg16f`, piksel uzayında bu kare → önceki kare.
- *Statik geometri:* `denoiserPositionImage` dünya pozisyonu × önceki
  kare VP. Shader'a önceki kamera matrisi push constant olarak gider.
- *Rijit hareketli nesne:* TLAS instance başına **önceki dönüşüm** tamponu
  (instance id ile indekslenir). Closest-hit nesne uzayı vuruş noktasını
  önceki dönüşümle dünyaya çevirir. Instance yükleme yolu zaten her kare
  dönüşüm yazıyor; bir öncekini tutmak ek kopya.
- *Deforme (skinned, sim yüzeyi, fracture parçası):* B2'de rijit yaklaşım +
  ayrılma testi yakalar; hassas köşe-başı önceki pozisyon **Sonra**.
- *Hacim / şeffaf / miss:* güvenilir vektör yok → `reactive` maske
  (0..1). Hacim zaten kendi temporal yolunda; maske onu dışarıda tutar.

**Ayrılma (disocclusion) testi:** önceki karenin pozisyon + normal + materyal
id'si ile karşılaştırma (`VulkanVolumeTemporal`'ın arama/ret mantığı yeniden
kullanılır, ikinci kopya yazılmaz).

**RayFusion:** raster yolu hızı vertex shader'da önceki MVP ile üretir
(instance başına önceki model matrisi, aynı tampon). **Aynı `velocity` +
`reactive` sözleşmesi** — B3/B4 tüketicileri hangi yoldan geldiğini bilmez.
Bugünkü raster TAA yalnız kamera reprojeksiyonu yapıyor; bu onu nesne
hareketine de doğru hale getirir.

IPC: `render.motion_stats` (geçerli geçmiş %, ayrılma %, reactive %),
`debug_view` yeni değerler `motion_vectors`, `disocclusion`.

Kabul (iç oda):
- Durgun kamera: hız AOV her yerde 0; geçerli geçmiş %100.
- Yörünge (`camera.orbit` küçük adım): statik geometride geçerli geçmiş ≥ %95.
- Hareketli rijit nesne (`scene` transform IPC'si ile kaydır): nesne
  üzerinde vektör nesneyle aynı yönde; arkası açılan bölgede ayrılma > 0.
- ★ Sessiz arıza: vektör yarım piksel kaymış olur ve her şey "biraz
  bulanık" görünür. Ölçüt: statik sahnede yörünge sonrası reprojeksiyonla
  geri dönüş hatası < 0.1 px (sayı olarak `motion_stats`'ta).

**B2 uygulama notları (2026-10-26, yazıldı):**
- Hız = noktanın önceki kameraya izdüşümü − şimdiki kameraya izdüşümü (ikisi
  de push-constant vektörlerinden; jitter sadeleşir). Kamera matrisi değil,
  raygen'in kendi `lowerLeft/horizontal/vertical` eşlemesinin tersi.
- Önceki kamera + örnek adresleri `MotionParams` kaydında (112 B, 4 slotluk
  host-visible halka, push constant'ta adres). Her DISPATCH bir karedir.
- Önceki dönüşümler: her TLAS kurulumu bir önceki örnek dizisini saklar (CPU
  yolu CPU kopyasından memcpy, GPU yolu komut tamponunda kopya); nesne hareketi
  yalnız kurulumdan sonraki ilk dispatch'e uygulanır. Sıra güveni: aynı sayı +
  aynı kurulum yolu; değilse yalnız kamera hareketi.
- Geçmiş birikim sıfırlamasında SİLİNMEZ (VulkanVolumeTemporal'ın tersi).
- Payload'a `primaryInst` (gl_InstanceID) eklendi; hair/volume closesthit
  yazmaz → o pikseller yalnız kamera hareketi alır.
- Raster: `raster_taa.comp` aynı sözleşmeyi yazar, yalnız kamera hareketi,
  yalnız TAA açıkken. **Raster nesne hareketi (vertex'te önceki model matrisi)
  YAPILMADI → B3'ün başında ya da ayrı iş.**
- Halton(2,3) 8 faz, varsayılan KAPALI (yakınsamış AA'yı kabalaştırır).
- `render.motion_stats sample=` tek senkron okuma; her kare maliyeti yok.
- Deforme geometri (skinned/sim): rijit yaklaşım, geçmiş testi yakalar.

## 5. B3 — gerçek zamanlı denoiser (1 derleme, compute shader + C++)

SVGF ailesi (Schied 2017) — bilinen, sağlam, ~birkaç ms:
1. **Temporal birikim:** B2 geçmişi ile 1 spp renk + moment (varyans)
   birikimi; reactive ve ayrılma bölgesinde geçmiş ağırlığı düşer.
2. **Varyans tahmini:** yeni açılan bölgede uzaysal varyans.
3. **À-trous dalgacık filtresi:** 4–5 iterasyon, kenar durdurucular
   derinlik/normal/albedo (raygen zaten yazıyor). Albedo demodülasyonu:
   doku detayı filtrelenmez.

Politika (tek yerde, ana döngüde):
- Hareket → SVGF (her kare).
- Durgun + birikim sürüyor → birikim görüntüsü + B1a OIDN takvimi.
- Yakınsadı → OIDN bir kez (yüksek kalite).
- Geçiş: hareket bitince SVGF geçmişi birikim başlangıcını besler, böylece
  durunca ekran "gürültüye geri düşmez".

IPC: `render.set_realtime_denoiser { mode = off|svgf, atrous_iterations,
history_max, variance_clip }`, `render.realtime_denoiser_stats` (geçiş başına
GPU ms, geçmiş uzunluğu ortalaması).

Kabul (iç oda, 1680×945, 3060):
- SVGF toplam ≤ 6 ms (OIDN fast 43 ms'ye karşı).
- Hareket halinde gürültü ölçüsü (komşu piksel farkı) OIDN fast'e yakın;
  bugünkü ham 1 spp: 0.065, OIDN: 0.012 (2026-10-26 ölçümü).
- Hayalet testi: hareketli nesnenin bıraktığı iz 4 kareden kısa.
- Hacimli sahnede (nükleer/gaz) hacim bölgesi yıkanmıyor (reactive çalışıyor).

## 6. B4 — upscale (1 derleme) — karar noktası

RT'yi düşük çözünürlükte izle (ölçek 0.5–0.77), sonra büyüt. P0'a göre
0.5 ölçekte RT ~96 → ~24–30 ms beklenir (piksel sayısı ¼).

İki seçenek, **B3 ölçümünden sonra karar**:
- **A. AMD FidelityFX FSR (2.x / 3.1 upscaler), MIT, Vulkan arka ucu.**
  Kalitesi kanıtlı, vendor bağımsız, açık kaynak çizgisine uygun. Maliyet:
  SDK bağımlılığı vcxproj'a girer (derleme süresi), API sözleşmesi
  (renk, derinlik, hız, reactive, jitter, exposure). FSR3 **kare üretimi
  (frame generation) kapsam dışı** — gecikme ve UI katmanıyla çakışma.
- **B. Kendi TAAU'muz.** B2 geçmişi ve B3 filtresi zaten elde; jitter +
  geçmişi yüksek çözünürlükte tutmak ek iş. Bağımlılık yok, kalite FSR'nin
  altında kalabilir.

Öneri: **A**, eğer B2 sözleşmesi FSR girdileriyle birebir uyuyorsa (hedef
budur); B, SDK entegrasyonu derleme süresini kabul edilemez uzatırsa.
DLSS kapsam dışı (yalnız NVIDIA, lisans). FSR1 (yalnız uzaysal) yalnız
geçici geri dönüş yolu olarak.

Gerekli: kamera alt-piksel jitter (raygen'de, B2 ile birlikte yazılabilir),
UI/overlay tam çözünürlükte, upscale tonemap ÖNCESİ (HDR).

IPC: `render.set_upscale { mode = off|fsr|taau, scale, sharpness }`,
`render.upscale_stats` (girdi/çıktı çözünürlüğü, GPU ms).

Kabul: ölçek 0.67'de RT dispatch ≤ 0.5× tam çözünürlük; ince geometri
(tel, saç, yaprak) yörüngede titremiyor; hacim bölgesi reactive ile
hayaletsiz.

### B4 KARARI ve TASARIMI (2026-10-26, kullanıcı onayladı: FSR 3.1)

**Sürüm:** FidelityFX SDK **v1.1.4 / FSR 3.1.4**, Vulkan arka ucu, AMD'nin imzalı hazır DLL'i
(kaynaktan derleme YOK → derleme süresine etkisi bir başlık + DLL kopyası). SDK 2.x / FSR 4
Vulkan'ı desteklemiyor; FSR 4 zaten yalnız RX 9000. Kaynaklar: GPUOpen FSR 3.1.4 kılavuzu,
FidelityFX-SDK GitHub releases/README.

**Donanım kapısı ("GPU destekliyorsa devreye girsin"):** FSR 3.1 compute shader'dır, üreticiden
bağımsız; resmî şart Vulkan 1.x + GLSL 4.50 + `GL_EXT_samplerless_texture_functions` +
`GL_EXT_shader_image_load_formatted` (Vulkan özelliği `shaderStorageImageReadWithoutFormat`)
ve typed UAV load / `R16G16B16A16_UNORM` storage desteği. Çalışma zamanında sırayla:
1. DLL `LoadLibrary` ile DİNAMİK yüklenir (import lib ile bağlama yok → DLL yoksa uygulama
   açılır, FSR yalnız "kullanılamaz" olur);
2. cihaz özellikleri + format desteği sorgulanır;
3. `ffxCreateContext` başarılı mı.
Herhangi biri başarısızsa mod istenmiş kalır ama **etkin olmaz**, `reason` adıyla döner
(SVGF'deki `available/reason` deseni), görüntü bugünkü tam çözünürlük yoluna kalır. RTX 3060
şartları karşılar. (İkinci el "RTX 20+ gerekir" iddiası doğrulanmadı — kapı ölçümle karar
verir, kart listesiyle değil.)

**Akış (yalnız hareket sırasında — dinamik çözünürlük):** kamera/nesne hareket ederken RT
`scale × viewport` boyutunda izler (aynı görüntülerin sol-üst bölgesine; yeniden ayırma yok,
FSR render size < resource size'ı destekler) → SVGF render çözünürlüğünde → FSR 3.1 (HDR
bayrağı, lineer renk) tam çözünürlüğe → tonemap. Kamera durunca tam çözünürlükte birikim
yeniden başlar (bir sıfırlama); durağan kalite referansı değişmez. Ölçek değişimi motion
geçmişini ve SVGF geçmişini geçersiz kılar.
- **Girdiler:** renk = SVGF çıktısı (render çöz., HDR); hız = B2 `motionImage` xy (bu kareden
  önceki kareye, piksel, y aşağı, jitter'sız — FSR'ın beklediği yön; `motionVectorScale`
  piksel için ayarlanır, işaret/eksen 1° dönüş testiyle DOĞRULANIR); derinlik = position
  AOV'dan küçük bir compute ile ters-Z sonsuz (FFX bayrakları ona göre); reactive = B2 z
  (R8, ≤ 0.9'a kırp); exposure = otomatik bayrağı ya da bizim pozlama değeri.
- **Jitter:** FSR Halton jitter ister (uzunluk 8·(display/render)² civarı; FSR kendi faz
  sayısını verir). B2'de Halton jitter yolu var (`motion_halton_jitter`); FSR etkinken o
  zorunlu, faz uzunluğu FSR'dan alınır.
- **SVGF ile ilişki:** SVGF render çözünürlüğünde kalır (gürültüsüz renk FSR'ın şartı);
  B3f'in durma geçişi (dondurulmuş geçmiş) tam çözünürlüğe dönüşte yeniden düşünülmeli —
  ilk durağan karelerde FSR çıktısı köprü olabilir. Planda ayrı madde.

**IPC/yüzey:** `render.set_upscale {mode: off|fsr|auto, quality: quality|balanced|performance|
ultra_performance (ölçek 1/1.5, 1/1.7, 1/2, 1/3), sharpness 0..1}`, `render.upscale_stats
{available, reason, dll_loaded, ffx_version, render_w/h, display_w/h, gpu_ms, active}`.
Varsayılan: geliştirme sırasında `off`; kabulden sonra `auto` (destek varsa açık) —
kullanıcının isteği. Panel: Sampling altındaki Denoise bloğunun yanında "Upscale While
Moving (FSR 3.1)". Python + yetki + overlay + audit (5 katman).

**Kabul (iç oda + bulutlu dış sahne, RTX 3060):**
1. Kapı: DLL yokken uygulama açılır, `available=false` + reason; varken true.
2. Balanced'ta RT ms ≤ 0.5 × tam çözünürlük; hareket FPS artışı sayıyla.
3. FSR GPU ms (1680×945 çıkış) — SVGF + FSR toplamı OIDN fast'ın altında.
4. 1° dönüş: görüntü kayması yok (hız işareti/ölçeği doğru); ince geometri (pervaz, avize,
   hasır, saç) yörüngede titremiyor; hareketli nesne hayaleti < 4 kare.
5. Hacim/gaz sahnesinde reactive bölge hayaletsiz.
6. Durunca tam çözünürlüğe dönüş: 128 örnekte ekran upscale kapalıyla bit bit aynı.
7. ★ Sessiz arıza: hız vektörü ölçeği yanlışsa (piksel vs UV) FSR "biraz bulanık / yüzen"
   görüntü verir, kimse hata demez — 4. madde bunu sayıyla yakalar.

### B4 UYGULAMA PLANI — somut dosya/adım listesi (2026-10-26)

**Durum: 2026-10-27 kullanıcı planı onayladı; kaynak adımları uygulandı. Derleme/canlı kabul bekliyor. Gerçek modül haritası en üstteki B4 devir notunda.**
Devir kuralı: plan kullanıcıya gösterilir; onaydan sonra aşağıdaki işler TEK kaynak
partisinde tamamlanır, kullanıcı tek shader+C++ derlemesi alır, canlı testler en sonda.
SDK başlıkları doğrudan incelendi; Vulkan DLL tahmini değil doğrulanmış bağımlılıktır.

1. **Bağımlılık/Release — TAMAM.**
   `RayTrophiStudio/third_party/FidelityFX/1.1.4/` içinde `include/ffx_api/`,
   `runtime/amd_fidelityfx_vk.dll`, lisanslar, SHA-256 `manifest.json`, README ve
   `FidelityFX.targets`; `scripts/setup_fidelityfx.ps1` yalnız indirir/doğrular/kopyalar.
   Proje include yolu ve x64 build-sonrası `$(OutDir)` kopyası bu target'tadır.
   `x64/Release/amd_fidelityfx_vk.dll` ve `FidelityFX-LICENSE.txt` şu anda mevcut ve
   kaynakla hash eşit. SDK/DLL derlenmez; import lib kullanılmaz. Kaynak XML,
   PowerShell parser, pinli arşiv kurulumu ve Release/Debug hash kontrolü geçmiştir;
   MSBuild veya uygulama çalıştırılmamıştır.
2. **Loader, cihaz kapısı ve context.** Yeni
   `source/include/Backend/UpscaleTypes.h`, `Backend/VulkanFsrLoader.h/.cpp`,
   `Backend/VulkanFsrUpscaler.h/.cpp`.
   Loader uygulama dizinindeki kesin DLL yolunu güvenli `LoadLibraryExW` ile açar;
   beş export zorunlu. `ffxCreateBackendVKDesc` + `ffxCreateContextDescUpscale`
   zinciri yalnız mevcut Vulkan cihazını kullanır; üretici/kart adı allowlist'i yok.
   Özellik/format kontrolü, context oluşturma ve provider-version sorgusu ayrı
   hata aşamaları olarak raporlanır. DLL file-version provider-version yerine
   yazılmaz. Resize/device-loss/context teardown yalnız ilgili fence'ler boşalınca.
3. **Hareket politikası ve etkin boyut.** Yeni
   `Backend/VulkanUpscalePolicy.h/.cpp` ve `Backend/VulkanUpscaleAdapter.cpp`.
   Mevcut kamera/nesne hareket kararını ortak kullan; ikinci bağımsız zamanlayıcı yok.
   Display/allocation extent ve active render extent ayrı değerlerdir. Quality
   modlarının render boyutunu SDK query'si verir (quality/balanced/performance/
   ultra_performance). `off`, final render, debug/AOV veya kapı başarısızlığı tam
   çözünürlüktedir. Hareket başlangıcı/duruşu/quality/resize değişimi epoch, motion,
   SVGF ve FSR history resetlerini tek geçişte tutarlı uygular. Düşük çözünürlük
   radyansı tam çözünürlük progressive tahminine veya OIDN girdisine karışmaz.
4. **Raygen, motion ve SVGF active extent.** Mevcut
   `VulkanMotionHistory.*`, `VulkanRealtimeDenoiser.*`, `raygen.rgen` ve
   `realtime_svgf.comp` üzerinde odaklı değişiklikler. Kapasite display boyutunda
   kalır; aktif dikdörtgen dispatch/copy/kernel sınırlarına ayrıca verilir.
   Shader'ların `imageSize`/allocation ile aktif viewport'u karıştırması önlenir;
   komşu örnekleri de aktif sınıra clamp edilir. B2 hız vektörü aktif render pikseli
   olarak kalır. Halton faz sayısı ve jitter offset'i `ffxQuery` içindeki
   `ffxQueryDescUpscaleGetJitterPhaseCount/GetJitterOffset` ile gelir; bugünkü sabit
   8 faz FSR etkinliğinde kullanılmaz. Authored jitter tercihi değiştirilmez.
5. **FSR girişleri.** Yeni `shaders/upscale_inputs.comp` ve
   `Backend/VulkanUpscaleInputs.cpp`. Position AOV + etkin kamera → view-space
   pozitif derinlikten ters-Z/sonsuz-far R32 depth; sky far=0. Motion xy RG16F,
   reactive R8 (B2 z, 0..0.9); FSR jitter'sız piksel vektörüne uygun ölçek açıkça
   tanımlanır ve 1° dönüşte ölçülür. Normalized motion örneğinin renderW/H çarpanı
   bizim piksel vektörüne körlemesine uygulanmaz. HDR preExposure ve exposure
   girdileri mevcut raygen/tonemap pozlaması izlenerek tek kez uygulanır.
6. **Aynı submission'da SVGF → FSR → tonemap.** Yeni
   `Backend/VulkanUpscalePresentation.cpp` mevcut
   `VulkanRealtimeDenoiserAdapter.cpp` sunum bağlantısını yönetir. `VulkanDevice`
   trace boyutu ile display/tonemap/readback boyutu ayrılır. FSR Vulkan komutlarını
   mevcut frame-slot command buffer'ına kaydeder. Filtreli düşük çözünürlük rengi
   raw restore'dan ÖNCE tüketilir; tam çözünürlük FSR çıktısı tonemap'e gider.
   Tonemap kaynak descriptor'ı in-flight güncellenmez: slot başına descriptor
   veya mevcut raw-backup/restore sözleşmesine uygun ayrı compose geçişi kullanılır;
   nihai seçim mevcut descriptor/readback kodu incelenerek bu modülde sabitlenir.
   Başarısız dispatch düşük çözünürlük köşesini tam görüntü diye sunmaz: o slot
   yayımlanmaz, reason kaydedilir, sonraki kare tam çözünürlük resetine döner.
7. **Duruş köprüsü ve durgun referans.** Yeni `Backend/VulkanUpscaleHistory.cpp`
   son tam display FSR görüntüsünü epoch/pose/extent ile etiketler. İlk full-res
   karelerde görüntü köprüsü yalnız aynı poz/geçerli yüzey geçmişiyle çalışır;
   B3f'in düşük çözünürlüklü momentleri doğrudan full-res geçmiş gibi okunmaz.
   Birikim sayısı boyunca köprü söner; 128 örnekte katkısı sıfır, raw HDR ve son
   sunum upscale-off referansıyla aynı olmalıdır. Sabit kamera slider değişikliği,
   yeni sahne ve resize eski FSR görüntüsünü taşımamalı. Bu adım geçişteki kararma/
   gürültü sıçraması regresyonunu önler; B3 küçük Sonra işleri bu partiyi büyütmez.
8. **Ortak API, IPC, Python ve panel.** Yeni `Api/RtApiUpscale.h/.cpp`,
   `Api/RtIpcUpscale.h/.cpp`, `Api/RtUpscaleBindings.h/.cpp`,
   `UI/UpscaleControls.cpp`; mevcut büyük dosyalara yalnız include/kayıt/çağrı.
   `render.set_upscale {mode:off|fsr|auto, quality, sharpness:0..1}` atomik
   doğrulanır; panel de aynı servisi çağırır. Stats: requested mode/quality,
   effective scale ve active; available/reason/dll_loaded/ffx_version; render ve
   display boyutları; gerçek GPU timestamp ms, reset/dispatch/failure sayaçları.
   `RtIpcSecurity`, overlay ve capability audit aynası; descriptor üretimi.
   Yeni kaynakların vcxproj/filters kayıtları. Geliştirme varsayılanı off;
   kabul sonrasında kullanıcı auto'yu açar; etkinlik kapısı cihaz/context sonucudur.
9. **Parti sonunda doğrulama.** Statik kaynak/ABI/IPC/dağıtım kontrolleri; dış
   `scripts/test/rt_probe_upscale_ipc.py` ve Release aynası hazırlanır. Kullanıcı
   bütün shader'ları ve C++'ı BİR kez derler. Sonra dış named-pipe ile tek canlı
   tur: DLL yok kapısı; balanced RT <=0.5x taban ve toplam SVGF+FSR ms; jitter/hız
   işaret-ölçek; ince geometri ve hareketli nesne; gaz reactive; duruş/resize;
   128 örnekte referans eşitliği. Sonuçlar `NEXT_BUILD_CHECKS.md` başına yazılır;
   test ortasında yeni özellik eklenmez. Derleme ve uygulama açma ajan tarafından
   yapılmaz; test için açık uygulama kullanıcıdan gelir.

**B4 dışı:** frame generation, DX12/OptiX, FSR4/SDK2 yükseltmesi, raster viewport'u
FSR'a geçirmek, SVGF §7 mikro-optimizasyonları. İlk B4 yalnız Vulkan Rendered yolu.

## 7. Sonra (bu planın dışında, sırasız)

- (B3'ten) SVGF: sağ dikey pervazda hafif noktalanma; hareket karesi +%1–2 parlak; à-trous 6.8 ms
  (fp16 ara görüntü / paylaşılan bellek); hayalet, hacim reactive, bulutlu sahne ölçülmedi;
  timeline oynatımı SVGF'yi tetiklemiyor (gökyüzü bulutu hız vektörsüz → sürünme riski).

- Deforme geometri için köşe başı önceki pozisyon (skinned + sim yüzeyi).
- Solid/Matcap için TAA/hız desteği ve Material per-mesh fallback için rijit nesne hızı (B3 Material/RayFusion global instance yolunu kapsar).
- B2 hız AOV maliyetinin geçmiş yazımı / önceki konum / register alt kalemlerini GPU profiler ile paylaştırma (ölçüm bekliyor).
- ReSTIR DI (çok ışıklı sahnelerde gölge ışını maliyeti) — bu sahnede 2 ışık, öncelik değil.
- OIDN'i yarım çözünürlükte koşmak / fp16 girdi.
- İkincil bounce'larda daha ucuz malzeme değerlendirmesi (ölçülmeden öneri değil).

## 8. Derleme takvimi özeti

| Parti | Derleme | İçerik | Kabul ölçümü |
|---|---|---|---|
| B1 | 1 | OIDN takvimi, hareket profili, RR düğmesi | örnek/sn, son kare eşitliği |
| B2 | 1 (+shader) | hız AOV, önceki dönüşümler, ayrılma, RayFusion hızı, jitter | motion_stats, debug view |
| B3 | 1 (+shader) | raster nesne hızı + TAA uyanması + SVGF | GPU ms, gürültü ölçüsü, hayalet |
| B4 | 1 | FSR ya da TAAU | ölçekli RT ms, titreme |

Her parti sonunda: `NEXT_BUILD_CHECKS.md` sıralı liste + tek IPC turu.

## Ek — önceki devir notları (arşiv, yeniden eskiye)

### Önceki devir notu — B3e canlı, B3f yazıldı (2026-10-26)

Durma anında kararma/gürültü ölçüldü (0.487→0.465): durağan mod hareket geçmişini atıyordu.
B3f: `realtime_svgf.comp` durağan dalı geçmişi dondurup `nh/(nh+N)` ile harmanlar. Liste
`NEXT_BUILD_CHECKS.md` en üstte (★ eski poz / çift kontur kontrolü).

### Önceki devir notu — B3d canlı, B3e yazıldı

Koyu noktalar/ince hat kırılması ölçüldü: geçmiş debug %98 yeşil → kök à-trous luminans
eşiğinin ham piksel varyansı (makaledeki 3×3 ön-filtre yoktu). B3e yalnız shader; liste
`NEXT_BUILD_CHECKS.md` en üstte. Ölçüm aleti kalıcı: `debug_history`.

### Önceki devir notu — B3c canlı, B3d yazıldı

Sonuçlar `NEXT_BUILD_CHECKS.md` "B3c SONUÇLARI". B3d listesi en üstte (3 madde). Açık iş:
hareket halinde ince/izole özelliklerde koyu tekil noktalar. İki tahmin denendi (2×2/3×3
arama, kamera piksel izi), ikisi de gidermedi → bir sonraki adım ÖLÇÜM: `debug_history`
ile o pikseller geçmiş buluyor mu. Kırmızıysa tutarlılık testi (normal 0.8 / malzeme / alfa
kesikli hasır), yeşilse demodülasyon / à-trous tarafı.

### Önceki devir notu — B3b canlı, B3c yazıldı (2026-10-26)

B3b ölçüldü (`NEXT_BUILD_CHECKS.md` "B3b SONUÇLARI"): karartma çözüldü, referans korunuyor.
Kalan: ince/izole özelliklerde koyu tekil noktalar → B3c yazıldı (piksel izi aynı malzemeli
komşu yoksa kameradan; SVGF push 48 bayt; `VulkanMotionHistory::cameraFootprint`). B3c
listesi aynı dosyanın en üstünde (5 madde, ★ hayalet kontrolü).

Durağan mod da B3c'de YAZILDI (`static_samples`, varsayılan 0): OIDN kapalıyken birikim
üzerinde yalnız uzaysal SVGF, limit hedef örnekle kırpılır. Varsayılan limit B3c
listesinin 8. maddesindeki kesişme ölçümüyle seçilecek. Panel düzeni de B3c'de.

### Önceki devir notu — B3b yazıldı (2026-10-26), derleme + test bekliyor

Kontrol: `NEXT_BUILD_CHECKS.md` en üst B3b listesi (9 madde). Değişen dosyalar:
`shaders/realtime_svgf.comp` (temporal yeniden yazıldı: kırpma yok, 2×2 ayak izi +
3×3 arama raygen motionHistory kayıtlarıyla, geçmişsiz pikselde ateş böceği kırpması),
`VulkanRealtimeDenoiser.*` (15 binding, `historyFlags`, `begin(..., rtHistory, rtWriteSlot)`),
adapter, `RealtimeDenoiserTypes.h` (enabled=true, `firefly_clip`), API/IPC/Python/overlay/
probe (yeniden adlandırma), `RealtimeDenoiserControls.cpp` + `scene_ui.cpp` (kontrol
Light Paths'e taşındı, Motion Vectors API üzerinden). Raygen değişmedi.

Açık kalanlar (B3b dışı): timeline oynatımı SVGF'yi tetiklemiyor (tetikleyici kamera +
TLAS; gökyüzü bulutu hız vektörsüz → oynatımda sürünme riski, ayrı karar); yavaş
sahnede (kare > 150 ms) hareket penceresi kare süresinden kısa; à-trous 6.7 ms;
hayalet / hacim reactive / final render / çoklu instance culling ölçülmedi; bulutlu
sahnede kare RT'ye bağlı (ilk örnek ~470 ms) — B4 ölçeği asıl kaldıraç.

### Önceki devir notu — B3 ölçüldü (2026-10-26), sıradaki B3b

Sonuçlar ve B3b önerisi: `NEXT_BUILD_CHECKS.md` en üst "B3 SONUÇLARI". Özet: politika,
raster hız, referans eşitliği PASS; **hareket halinde karartma FAIL** — kök
`realtime_svgf.comp` temporal geçmiş kırpması (variance_clip ile parlaklık doğrudan
oynuyor, à-trous oynamıyor). UI: SVGF kontrolü yalnız viewport OIDN açıkken görünüyor
ama SVGF ondan bağımsız. B3b: kırpma düzeltmesi + UI taşıma + varsayılan açık kararı.
İkinci FAIL (bulutlu sahne): siluetlerde parlak nokta hattı = her kare geçmiş reddi
(jitter'lı tek piksel testi) → B3b'ye 2×2 ayak izi + 3×3 arama. Timeline oynatımı SVGF'yi
tetiklemiyor (tetikleyici yalnız kamera + TLAS) — oynatım desteği ayrı karar (gökyüzü bulutu
hız vektörsüz → sürünme riski).
Ölçülmeyenler: hayalet (<4 kare), hacim sahnesi reactive, final render, çoklu instance
culling. Ölçüm script'leri yok (ajan scratchpad'i); yöntem: hareket sırasında
`viewport.get_screenshot` ortalama luma vs yakınsamış OIDN.

### Önceki devir notu (B3 kaynak partisinden sonra)

**B3 derlenmedi, canlı test edilmedi.** Derlemeyi kullanıcı yapar. Testler bütün
adımlar bittikten sonra tek turda: `NEXT_BUILD_CHECKS.md` başındaki B3 listesi.
İLK madde B2'den kalan raster `historyFlags` kamera-hız düzeltmesidir.
B2'nin ölçülmüş 11/12 sonuçları aynı dosyada aşağıda korunur.

**Yazılan kaynaklar:**
- Raster Material/RayFusion vertex yolu instance başına önceki model matrisini
  okur. Global instance tamponu önceki çizilen kareyi GPU kopyasıyla saklar;
  GPU culling mevcut ve önceki matrisleri birlikte sıkıştırır. HDR dördüncü
  renk attachment önceki modelin güncel kameradaki clip konumunu taşır.
  TAA aynı B2 piksel/y-aşağı sözleşmesine çevirir ve geçmiş derinliğini sınar.
  `motion_stats.devices[].raster_objects_available` hazır olma durumudur.
  Material per-mesh fallback kamera hareketi taşır. Solid/Matcap mevcut
  tasarımda TAA çalıştırmaz; bekleme kapısı ve raporlar bu sınıra bağlandı,
  eski Material hız görüntüsü bu modlarda örneklenmez. Deforme köşe geçmişi
  bu partide eklenmez.
- TAA çizim fonksiyonu zaten daha fazla kare istiyordu; dış
  `needsViewportRender()` bunu bildirmiyordu. İstek kapıya bağlandı; sahne
  dirty'sine çevrilmedi, dolayısıyla her kare birikim sıfırlanmaz.
- `VulkanRealtimeDenoiser.*`: temporal renk/moment, spatial variance, 1..5
  à-trous, normal/pozisyon/materyal/albedo durdurucuları, albedo demodülasyonu,
  reactive ağırlığı. GPU timestamp ve senkron istatistikte ortalama geçmiş.
  Geçmiş slotları epoch/örnek/dispatch etiketi taşır.
- **Referans koruması:** ham HDR aynı komut tamponunda yedeklenir, yalnız
  tonemap sırasında filtreli görüntü gösterilir, ardından ham tampon geri
  yüklenir. Geçişte ilk 8 örnekte görüntü köprüsü söner; filtreli radyans
  Welford'a veya OIDN HDR girdisine beslenmez. Bu, §5'teki geçiş niyetini
  referans tahminini yanlı hale getirmeden uygular.
- Tek politika ana döngüden ortak servise gider: kamera/rijit nesne hareketi
  SVGF, durgun birikim mevcut B1 OIDN takvimi, final render SVGF dışında.
  Viewport OIDN ile SVGF aynı hareket karesinde birlikte çalışmaz.
- Ortak API + IPC + Python + panel + yetki + descriptor overlay hazır.
  `render.set_realtime_denoiser {mode:'off'|'svgf', atrous_iterations:1..5,
  history_max:1..64, variance_clip:0.1..10}`; atomik doğrulama, varsayılan
  off / 4 / 24 / 3. Ayarlar B2 motion kontrolü gibi oturumluk.
  `render.realtime_denoiser_stats {synchronize:true|false}`: true GPU bekler;
  false panel/poll için. GPU total backup/compose/tonemap/restore dahil.
  SVGF açıkken motion'u kapatmak API'den reddedilir.

**Henüz ölçülmeyenler:** ≤6 ms, OIDN'e yakın gürültü, <4 kare iz, hacim
reactive kalitesi, 128 örnek referans eşitliği; bunların hiçbiri PASS değildir.
B2 hız AOV ~3.5 ms maliyetinin alt kalemleri de henüz ölçülmedi. SVGF bu
geçmişi tükettiği için ölçümden önce B2 geçmiş yazımını kaldırmadık.

**Kurallar ve tuzaklar:** shader'lar dahil TEK derleme; kullanıcı derler.
Dış IPC (`scripts/test/rt_ipc.py`), embedded script'te kare bekleme yok.
Viewport OIDN `viewport_enabled` ile açılır (`render_enabled` final içindir).
Vec3 dizidir. GPU timing yeni pencere `set_gpu_timing enabled=true` ile.
Hız görüntüsünü hareket çağrısının hemen ardından oku; sonraki dispatch onu
üstüne yazar. Duran RT'de ~%4 kenar reddi B2'de normal ölçüldü. Doğrulama
resimleri `.captures/` altında geçici ve iş sonunda silinir.

**Sonra:** RR 2/1 gürültü karşılaştırması (varsayılan 3); geçici projede
schedule/rr/motion-profile kayıt-yükleme testi; yeni B3 bulguları §7'ye.
