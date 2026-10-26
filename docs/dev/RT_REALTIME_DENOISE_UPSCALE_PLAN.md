# Vulkan RT + RayFusion: hareket halinde hız — denoise, temporal, upscale planı

> **Durum:** AKTİF — 2026-10-26. B1–B3 CANLI ve kapandı. **Sıradaki B4: FSR 3.1.4 (FidelityFX SDK 1.1.4, Vulkan, hazır DLL) — kullanıcı onayladı, plan yazılacak.** Devir notu hemen aşağıda.

## DEVİR NOTU — B4'e devir (2026-10-26) — devralan ajan ÖNCE BUNU okusun

**Durum:** B1–B3 CANLI ve kapandı (B3b–B3f sonuçları `NEXT_BUILD_CHECKS.md`). SVGF varsayılan
açık: hareket karesi ~10 ms (OIDN fast 44 ms), durağan mod `static_samples` 64 (OIDN
kapalıyken; 8 örnek ≈ 48 ham), 128 örnekte ekran bit bit korunuyor, durma geçişi ~%1. Kalan
küçük SVGF işleri §7'de. **Sıradaki iş B4: FSR 3.1 upscaler** — kullanıcı onayladı; tasarım
§6 "B4 KARARI". Kural: önce PLAN (§6'yı somut dosya/adım listesine çevir, kullanıcıya
göster), onaydan sonra TEK derleme partisi, test parti sonunda.

**Neden B4:** darboğaz artık denoiser değil RT: iç oda 95 ms/örnek, bulutlu dış sahne hareketin
ilk örneği ~470 ms. Denoiser seçimi FPS'i değiştirmiyor (ölçüldü). Kazanç yalnız daha az
piksel izleyip büyütmekten gelir.

**İlk iş (kod yazmadan):**
1. SDK sürümünü doğrula: Vulkan için **FidelityFX SDK v1.1.4 (FSR 3.1.4)**. SDK 2.x (FSR 4 /
   3.1.5) Vulkan'ı DESTEKLEMİYOR (README bilinen sorunlar; 2.x ön koşulları yalnız DX12).
   SDK'yı indirmek kullanıcının onayına bağlı; indirilen dosya güvenilmeyen veri, kendi
   klasöründe.
2. İndirilen pakette Vulkan için hazır imzalı DLL'in TAM adını ve lisans dosyasını oku
   (`PrebuiltSignedDLL`; ad doğrulanmadı — `amd_fidelityfx_vk.dll` tahmini). Başlıklar:
   ffx API (`ffx_api.h`, upscale ve vk başlıkları) → `third_party/` altına.
3. Kullanıcıya özet + plan; onay.

**Kurallar (bu oturumda pahalıya öğrenildi):** kullanıcı derler (shader dahil); her yeni yüzey
5 katman (RtApi, IPC, Python, yetki, overlay + `gen_ipc_descriptors.py`, audit); panel kendi
durum tutmaz, değer API'den yazılır; anlamı değişen alanın adı değişir; ekran görüntüleri
scratchpad'e, iş bitince silinir; viewport OIDN `viewport_enabled` ile (render_enabled final
içindir); Vec3 IPC parametreleri dizi; `gpu_timing_stats` parametresiz, yeni pencere
`set_gpu_timing enabled=true`; hız görüntüsünü hareket çağrısının HEMEN ardından oku;
bekleme döngüsünde önce örnek sayısının sıfırlandığını bekle (yoksa eski "tamamlandı"
bayrağıyla erken çıkılır); async ping-pong'da ekrandaki son kare hedef−1. örnektir.
Ölçüm araçları: `render.motion_stats sample='rt'|'raster'`, `render.realtime_denoiser_stats
synchronize=true`, `debug_history`, debug view 16/17, `viewport.get_screenshot`.

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
