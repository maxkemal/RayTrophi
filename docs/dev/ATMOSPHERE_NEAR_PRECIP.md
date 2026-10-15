# Kamera önü yağış: damla alanı, perdeyle dikiş, lens (Atmosfer Faz 10)

> **Durum:** AKTİF — 2026-10-15. N1 + N1b DERLENDİ (kullanıcı): maliyet iki
> backend'de düştü. N1c kullanıcı tarafından DERLENDİ; 2026-10-15 dış IPC probu
> tamamen geçti. Koyuluk/rüzgâr/dikiş için yeni görsel kabul ayrıca bekleniyor.
> N2–N4 plan; kar + lens bu belgenin ajanında (tek sahip), lens tasarımı
> `ATMOSPHERE_LENS_WATER.md`.
> Yerine geçtiği yapı: `WeatherParams`'ın havadaki kısmı (elle tip/şiddet,
> ekran overlay'i, gök tonu, duvar saatiyle önizleme) — N1'de söküldü.

## ★ DEVİR NOTU (her parti sonunda güncellenir)

- **RT Play yenilenmesi:** Kullanıcı RayFusion'da perde hareketini gördü fakat
  RT'de çoğu kare sabit kaldı. Main cloud push → aynı döngüde RT render wake
  bağlantısı eklendi (ayrı geç playback-frame kontrolüne bağımlılık kaldırıldı).
  Perde quadrature'ı sabittir, alanın timeline/wind girdileri dinamik kalır.
  Eski binary ölçüm anında durmuştu; Play canlı kabulü yeni C++ build'de bekliyor.

- **Kamera perde içinde gürültü:** Kullanıcı duraklatılmış sahnede kaynağı
  perdeye ayırdı. Ortak `cloud_rt.glsl` kamera perdesi sabit orta nokta quadrature
  kullanıyor; jitter aralık sınırlarından kaldırıldı, eksik ilk mesafe düzeltildi.
  Yakın perde ince deseni yumuşatıldı, ortalama extinction korundu. Yalnız shader
  paketi derlemesi ve RT/RayFusion göz kabulü bekliyor; tam aralık/homojen
  extinction sayısal kontrolü geçti. Yakın partiküller bu partide değiştirilmedi.

- **L2 kar/buz kaynak teslimi:** Kamera camında kar tutunma/birikim/erime,
  buzlu formlar ve donan yağmur eklendi; `glass_temperature_c` ortak UI/script/IPC
  ayarıdır. Dört lens/post shader ve C++ derlemesi, `--snow` probe/göz kabulü
  bekleniyor. Ayrıntı `ATMOSPHERE_LENS_WATER.md` §11. L1 kullanıcı derlemesinde
  çalıştı; hafif eliptiklik görsel ayarı açık bırakıldı.

- **L1 kamera camı ilk kaynak teslimi:** sıvı yağmur/çisenti için ayrı, varsayılan
  kapalı kamera seçeneği; GPU damlalar + RT/RayFusion ortak HDR kırılma, UI/script/
  IPC ve proje ayar kaydı yazıldı. Henüz derlenmedi. Gerçek 256-damla / 10 s history
  sınırı, yeni dört shader ve probe `ATMOSPHERE_LENS_WATER.md` §10'dadır. Kar/buz L2.
- **2026-10-15 devam kontrolü:** kullanıcı N1c'yi sorunsuz derlediğini bildirdi.
  Açık Studio'da `scripts/ipc/Probe-NearPrecip.ps1` dış süreçten çalıştı: ALL PASS.
  Oran/tip tek kaynağı, kuru nokta, shutter–çizgi ilişkisi, zorlanmış kar, perde
  başlangıcı, atomik shutter doğrulaması ve yüzey weather sözleşmesi doğrulandı.
  Geçici iklim/bulut ayarları geri yüklendi. Bu probe görüntü okumaz; %47 mercek
  geçirme payının görsel kabulü ve RT/RayFusion görünüm karşılaştırması değildir.
  Camdaki tepki henüz yok; L1/L2/L3 tasarımından uygulamaya geçilecek.
- **N1 yazıldı (2026-10-14), derlenmedi.** Kamera önündeki yağış artık
  atmosferden geliyor: `cloudPrecipSampleForm(kamera)` → oran + 8 tip
  karışımı. Shader `near_precip.glsl`, RT (`raygen.rgen`) ve RayFusion
  (`cloud_scene.comp`) aynı fonksiyonu çağırıyor. Perde march'ı kamera
  ışınlarında `near_range_m`'den başlıyor (çift sayım yok).
- Eski overlay CPU, OptiX, Vulkan RT ve RayFusion'dan söküldü; `visibility`,
  `visual_mode`, `Mist`, "Realtime Weather Preview" ve `VulkanWeatherClock`
  gitti. `WeatherParams` artık yalnız **elle yüzey kaplaması** (ıslaklık,
  kar/toz birikimi); onu `ATMOSPHERE_SURFACE_COUPLING.md` adım 2 bağlayacak.
- Enstantane tek: `CloudState::shutter_s` (şimşek darbesi + damla çizgisi).
  Eski `lightning.shutter_s` buraya taşındı.
- Ölçüm: `world.sample_clouds {mode:"near_precip"}`, panelde okuma.
- **N1b (2026-10-14 gece) — maliyet, YAZILDI derlenmedi.** Kullanıcı:
  fırtına sahnesinde RayFusion 130 → 340 ms. Canlı ölçüm: ~210 ms uzak
  perdeden (`ground_fraction` 0 ↔ 1), yakın damla kamera kuruyken ~0–10 ms.
  Perde: örnek sayısı segment boyuna bağlı, güneş gölgesi mesafeyle
  yenileniyor. Yakın damla: kamera durumu (oran, tip, gölge) artık
  `cloud_sample.comp` mode 7 ile parametre gönderiminde TEK iş parçacığında
  hesaplanıp `CloudParamsGPU::nearState`'e yazılıyor; çizim yalnız okuyor
  (önce her pikselde hesaplanıyordu). `world.cloud_stats` →
  `near_state_dispatches`.
- **2026-10-15 kullanıcı (derledi):** performans RT ve RayFusion'da arttı.
  Şimdilik yalnız yağmur görünümü var; iki gözlem — **koyu**, ve **rüzgârda
  fazla hareket ediyor**. Kodda kontrol EDİLMEDİ, ölçülmedi; kök adayları:
  1. **Koyuluk = damla modeli.** `nearPrecipComposite` damlayı opak bir
     yayıcı gibi çiziyor: `L = Es·gölge·faz + ambTop·(0.275 + 0.35·gölge)`
     (perdenin İÇ SAÇILMA formülü). Fırtına altında gölge ≈ 0 → L ≈ 0.26 ×
     açık gök LUT texeli; arkadaki gökten/zeminden koyu → damla karartır,
     peçe de karartır. Gerçek sıvı damla bir mercek: ~165° görüş alanındaki
     ORTAMI kırarak gösterir (Garg–Nayar: radyans ≈ ortam ortalaması, arka
     plandan genelde PARLAK). Düzeltme yönü: sıvı formlarda (yağmur,
     çisenti, donan yağmur) damla radyansı = kırılmış ortam (gök yarım
     küresi, bulut altında bulut tabanının parlaklığı — `ambTop` açık gök
     LUT'u, örtüyü bilmiyor) + güneş parıltısı; buz formları (kar, graupel,
     pelet, dolu) dağınık beyaz kalabilir. Peçe de radyansı aynı olmalı.
     Önce sayıyla doğrula: damla piksellerinin parlaklığı < arka plan mı.
  2. **Fazla hareket = rüzgâr + pozlama.** Damla hızı `v = tam iklim yüzey
     rüzgârı + düşüş`. Perde ise taban altı havayı iklim rüzgârının ~%60'ı
     kabul ediyor (`kSurfaceWindFraction` = 0.6, `makeCloudParamsGPU`) —
     iki tanım tutarsız; yakın damla da aynı payı kullanmalı (tek sabit).
     Ayrıca bu sahnede `shutter_s` = 0.135 (kullanıcı ayarı; varsayılan
     1/30): çizgi boyu = |v| × 0.135 → 10 m/s rüzgârda ~1.6 m, neredeyse
     yatay uzun çizgiler. Timeline oynarken kafes `v·t` ile kayar; hız
     doğruysa hareket de fiziksel.
- **Devir (2026-10-15):** kar ve kamera camı (lens) etkisi aynı ajana
  (bu belgenin sahibi) verildi; alanda tek ajan. Kar: aynı
  `near_precip.glsl` (form tablosu `kNearRadius/Sway`, buz = dağınık);
  lens: N3 → `ATMOSPHERE_LENS_WATER.md` (tasarım, L1–L3).
- **N1c (2026-10-15) — yukarıdaki iki madde, YAZILDI derlenmedi:**
  1. Koyuluk: sıvı formlar (yağmur, çisenti, donan yağmur; sulu sepken
     yarı yarıya) artık mercek: kaplanan arka planın `kNearLensKeep` =
     0.47'sını geçirir (τ 0.94 × görüşün yarısı = piksel), kalan yarıyı
     ortamdan + güneş parıltısından alır (`nearParticleLight`). Peçe de
     aynı. Buz formları eski dağınık-beyaz model. Arayüz değişmedi
     (`acc + T·arka`), callable/çağrı yerleri aynı.
  2. Rüzgâr: yakın damla rüzgârı = iklim rüzgârı ×
     `kSurfaceWindFraction` (0.6) — perdenin taban-altı varsayımıyla tek
     tanım (`makeCloudParamsGPU`). Probe adım 4 buna göre.
  - Bilinen: sıvı damla arka planı daha az kestiği için yakın alanın
    ortalama sönümü perdenin aynı aralıktaki sönümünden biraz az (perde
    ileri lobla benzer bir payı zaten geri saçıyor); dikiş testi 6'da
    fark görülürse `kNearLensKeep` ayarlanır. Sahnedeki `shutter_s` 0.135
    kullanıcı ayarı: çizgi boyu ona göre uzun kalır (doğru).
- Sıradaki: L1 (lens yağmuru), sonra N2 (barınak/örtü haritası + sıçrama
  + kamera hızı).

---

## 0. Bugün (kod okuması, 2026-10-14)

| Katman | Durum | Kaynak |
|---|---|---|
| Uzak perde | **atmosferden**: `cloudPrecipMarch`, oran + tip karışımı, ışık bulut sütunundan, 40 km | `cloud_rt.glsl`, `cloud_common.glsl` |
| Kamera önü | **eski sistem**: `WeatherParams` (tip None/Rain/Snow/Dust/Mist, `intensity`, `density` 0..1), 3 sabit derinlikte ekran desen katmanı, **sabit renk** (`weatherTintColor`), ışıktan bağımsız | `weather_fast.glsl`, `ray_color.cuh`, `Renderer.cpp` |
| Gök tonu | eski sistem: tipe göre gök rengi karıştırma | `rt_sky.glsl`/`sky_model.cuh`/`World.cpp` |
| Önizleme | "Realtime Weather Preview": viewport'ta zaman = `SDL_GetTicks` (duvar saati), her kare birikim sıfırlanır — **zaman-saf değil** | `Main.cpp`, `VulkanWeatherClock.cpp` |
| Yüzey | eski sistem, elle: ıslaklık / birikim / çökelme / yükseklik | `weather_surface.glsl`, `material_scatter.cuh` |

İki yağış vardı: atmosferdeki (perde) ve elle girilen (kamera önü). Birbirini
bilmiyorlardı: fırtınanın altında kamera önü kuru, açık havada yağmur
çizgileri olabiliyordu. Bu deponun bilinen hata sınıfı (iki tanım, sessiz
ayrışma).

## 1. İlkeler

1. **Tek kaynak.** Kamera önündeki damlanın oranı, tipi, boyu ve hızı
   `cloudPrecipSampleForm`'dan gelir — perdenin kullandığı fonksiyon. Elle
   "yağmur yağsın" demek atmosferin yağışını ayarlamaktır
   (`precipitation.rate_mm_h`, `form`); ikinci bir düğme yok.
2. **Ortalama korunur.** Yakın damlaların ortalama kaplaması, perdenin aynı
   aralıktaki optik derinliğine eşittir (`N·πr²·Δz`, `N = σ/(2πr²)`).
   Perde march'ı kamera ışınında `near_range_m`'den başlar.
3. **Işık aynı.** Damla radyansı perdenin örnek ışığıyla aynı formül:
   `S = Es·gölge·faz + ortam·(0.275 + 0.35·gölge)` (+ şimşek). Sabit renk yok.
   Sıvı damla (N1c) bu ışığın yarısını ortamdan alır, diğer yarısı arkasındaki
   piksel (mercek, Garg–Nayar); buz dağınık beyaz kalır.
   Sonuç fiziksel: yağmur arkadan ışıkta parlar, karanlık zeminde zor
   görünür (Garg & Nayar). Bu bir hata değil, "oyun yağmuru" olmamanın kendisi.
4. **Zaman-saf, tohumlu.** Damla konumu `(hücre hash, timeline zamanı)`'nın
   saf fonksiyonu. Duvar saati yok. Timeline durunca damlalar donar (doğru).
5. **İki backend, tek GLSL.** OptiX dondurulmuş: yeni katman almaz, eski
   overlay'i söküldü.

## 2. Model (N1)

### 2.1 Durum, kamerada
`nearPrecipForms(kamera)` = `cloudPrecipSampleForm(kamera)`: oran R (mm/h),
karışım. Kare başına BİR KEZ hesaplanır (N1b): `cloud_sample.comp` mode 7,
bulut parametre gönderiminin içinde, `CloudParamsGPU::nearState`'e; kamera
gölgesi de orada (taban şekli, güneşe 48 adım). Karışımın **en büyük iki** tipi çizilir (genelde 1–2 tip var).
Yakın alan ~15 m; oran km ölçeğinde değişir, kameradaki değer yeter.

### 2.2 Tip tablosu (tip başına)
Mevcut tablolar: düşüş hızı `kPrecipFall*`, sönüm `kPrecipSigma*` ve üssü,
ileri lob `kPrecipFwd*`. Yeni: eşdeğer yarıçap (yağmur 1 mm, çisenti
0.2 mm, kar 2.5 mm, sulu sepken/graupel/pelet 1.2–1.5 mm, dolu 6 mm, donan
yağmur 1 mm) ve buz payı (beyaz dağınık / sıvı kırıcı). Sayı yoğunluğu
`N = pay · σ(R) / (2πr²)` (Q_ext = 2). 10 mm/h yağmurda N ≈ 170 /m³ —
Marshall–Palmer'ın D > 1 mm payıyla (~250) aynı mertebe.

### 2.3 Kafes
- Dünyaya sabit 3B kafes, **damlayla birlikte hareket eden** çerçevede:
  `q = p − v·t`, `v = (rüzgâr_x, −düşüş, rüzgâr_z)`. Kamera dönse de yürüse
  de paralaks doğru; RT ve RayFusion aynı damlayı görür.
- Hız tipe sabit (yalnız iklim rüzgârı değişir): hız kameranın konumuna
  bağlı olsaydı `v·t` yüzlerce metrelik ofsete dönüşür, kamera kıpırdayınca
  bütün damlalar sıçrardı.
- Hücre bir eksende `v`'ye hizalı ve uzun: `h_v ≥ |v|·shutter·1.25`, çizgi
  hücresinden taşmaz → hücre başına tek test, komşu yok.
- Bantlar: [0.25, 0.5, 1, 2, 4, 8, 16] m, enine hücre `h = bant başı / 5`.
  Bant başına ~8 hücre geçilir → tip başına ≤ ~72 adım (sabit üst sınır).
- Hücre başına beklenen damla `w = N·V`. `w < 1`: hücre `w` olasılıkla dolu.
  `w > 1`: tek temsilci çizgi `w` damla taşır.

### 2.4 Çizgi
- Pozlama boyunca damla `baş − v·shutter → baş` doğrusunu süpürür.
- Çizilen yarıçap `ρ = max(r, piksel_yarıçapı(z))`, alan
  `A = 2ρ(L⊥ + 2ρ)`, opaklık `α = min(α_max, w·πr²/A)`.
  Tek damla için bu tam olarak Garg–Nayar'ın zaman × alan seyrelmesi.
- `α_max = 0.35`'i aşan pay bantta **düzgün peçe** olarak eklenir — ortalama
  korunur, uzak bantta kalın ip gibi çizgiler oluşmaz.
- Kar: yarıçap büyük, hız küçük, hücre içinde salınım (genlik ≤ kenar payı,
  hücre dışına taşmaz).

## 3. Ayarlar ve IPC (kural 1)

| Alan | Yer | Anlam |
|---|---|---|
| `precipitation.near_enabled` | `world.set_clouds` | yakın damla alanı (perde başlangıcı da buna bağlı) |
| `precipitation.near_range_m` | `world.set_clouds` | 2..60 m; açık damla aralığı |
| `shutter_s` | `world.set_clouds` | kare pozlaması (şimşek + çizgi), 0..0.5 s |
| ölçüm | `world.sample_clouds {mode:"near_precip", points:[...]}` | nokta başına `rate_mm_h`, `forms`, `number_density_per_m3`, `streak_length_m`, `curtain_start_m` |

Panel: Clouds → Precipitation → "Near camera" (açık/kapalı, aralık,
enstantane) + kameradaki durumun salt okunur satırı. Panel ve IPC aynı
`CloudState` alanlarını yazar.

## 4. Bilinen sınırlar (N1)

- **Barınak yok:** çatı altında, odada da yağar. Eski overlay'de de yoktu.
  N2'nin işi (yukarıdan örtü haritası, yüzey bağlamayla ortak).
- **Kamera hızı yok:** araba/uçuş kamerasında damla kameraya akmaz.
- **Yerel ışık yok:** damla güneş, gök ve şimşekle aydınlanır; sokak
  lambası damlayı parlatmaz (Garg–Nayar'ın en görünür durumu) → N2.
- Ses köprüsü hâlâ global oranı okuyor (`AudioWeatherBridge`); kameradaki
  oran ancak GPU okuma geri bildirimiyle gelir → N2.

## 5. Sonraki fazlar

- **N2 — barınak ve sıçrama:** yukarıdan derinlik/örtü haritası (kamera
  çevresi, yüzey bağlamayla ORTAK, `ATMOSPHERE_SURFACE_COUPLING.md` §1);
  damla haritanın altında ölür; derinlik isabetinde sıçrama; kamera hızı
  `v_rel = v − v_kamera` (animasyondan, zaman-saf); yakın nokta ışıkları;
  kameradaki durumun asenkron geri okunması (ses, panel).
- **N3 — lens (kamera seçeneği, göz damla görmez):** tasarım
  `ATMOSPHERE_LENS_WATER.md` (çarpma akısı `v_rel`'den, durumlu zaman-saf
  simülasyon + checkpoint, kırılma composite iki backend tek GLSL; kar
  yapışır/erir, donan yağmur buz tabakası).
- **N4 — yüzey:** `WeatherParams` tümüyle söker; ıslaklık ve birikim yer
  yağış haritasından (`ATMOSPHERE_SURFACE_COUPLING.md` adım 2–3).

## 6. N1 uygulaması (2026-10-14)

- `CloudState::shutter_s` (lightning'den taşındı), `CloudPrecipitation::
  near_enabled`, `near_range_m`; JSON, doğrulama, lerp.
- GPU paketi: `precip2.x` = shutter (yakın alan kapalıysa 0), `precip3.w` =
  aralık, `cirrusDirection.w` / `emissionSource.w` = yüzey rüzgârı x/z m/s.
  `cirrusDirection.z` = timeline zamanı (eski "weather clock" yerine
  `m_cloudTime`).
- `near_precip.glsl` (yeni), `cloud_rt.glsl` perde başlangıcı,
  `raygen.rgen` + `cloud_scene.comp` çağrı, `cloud_sample.comp` mode 6.
- Söküm: `weather_fast.glsl`, `VulkanWeatherClock.cpp`, `WeatherClock.h`,
  CPU/OptiX overlay ve gök tonu, `WeatherParams::visibility/visual_mode`,
  `WEATHER_MIST`, `realtime_weather_preview`, ilgili keyframe alanları.
