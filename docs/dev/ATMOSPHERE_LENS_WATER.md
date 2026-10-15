# Kamera camı: damla, akış, kar, buz (Atmosfer Faz 10 / N3)

> **Durum:** AKTİF — 2026-10-15. L1 ilk kaynak teslimi yazıldı; kullanıcı C++/
> shader derlemesi tamamlandı; temel dış IPC probu geçti. Islak sahne ve görsel
> kabul bekleniyor. L2 kaynak teslimi eklendi; derleme/canlı kabul bekliyor. L3 tasarım. Üst belge:
> `ATMOSPHERE_NEAR_PRECIP.md` §5 N3. Kanonik yön: kamera seçeneği, göz
> damla görmez; yağış atmosferden gelir, elle "yağmur efekti" yoktur.

## ★ DEVİR NOTU

- **Ortak sıcaklık (son sözleşme, §12):** Ayrı `glass_temperature_c` kaldırıldı.
  Cam, atmosferin kamera konumundaki sıcaklığını kullanır. Yüzey kaplaması da
  aynı iklim profiline bağlandı; yerel weather sorgusu sıcaklık/termal uygunluk
  döndürür. Önceki §11 bağımsız sıcaklık ayarı tarihsel teslimdir. C++/shader
  paketi derlemesi ve yeni canlı probe bekleniyor.

- **L2 kaynak teslimi:** Kar, sulu sepken, graupel, dolu, buz peletleri ve donan
  yağmur için tutunma/faz davranışı eklendi. Ortak kamera servisine
  `glass_temperature_c` (-80..80, varsayılan 0) eklendi; cam sıcaklığı hava
  sıcaklığından bağımsızdır. Yeni binary/shader ve canlı kabul bekleniyor.
  Güncel L2 sözleşmesi §11'de; eski L1-only notları tarihsel kapsamı anlatır.

- **Görünürlük teşhisi:** Son shader düzeltmesine rağmen kullanıcı 30 damlada etki
  görmedi. DoF kapalı/pinhole kameraya fallback f/2 CoC uygulanması düzeltildi;
  C++ yeniden derlenmeli. Sample artık gerçek optik alanın maksimum/ortalama
  coverage, maksimum UV kayması ve dolu piksel sayısını da okur. Bu kontrol
  simülasyon sayacını görüntü kabulünden ayırır. Yeni sonuç henüz doğrulanmadı.

- **L1 kaynak teslimi (2026-10-15):** UI/script/IPC ortak servis, kamera başına
  kalıcı ayarlar, GPU sıvı damla simülasyonu ve RT/RayFusion ortak HDR composite
  yazıldı. İlk teslim 256 damla ve 601 sabit adımın GPU snapshot geçmişidir
  (yaklaşık 10 s); aşağıdaki 4096 damla/checkpoint hedefi henüz uygulanmadı.
  Kullanıcı derledi; dış viewport probunda ayar/atomik doğrulama ve GPU kaynak/
  pipeline kontrolleri geçti. Açık kamerada yağış 0 mm/saat olduğundan ıslak
  simülasyon/geri sarma ve göz kabulü bekliyor. Kesin sözleşme ve sınırlar §10'da.
- **2026-10-15 devam:** N1c kullanıcı tarafından derlendi; açık Studio'da mevcut
  yakın yağış probu tamamen geçti. Bu belge hâlâ tasarımdır: cam simülasyonu veya
  composite kodu teslim edilmiş sayılmaz. N1c sayısal kabulü camdaki fiziksel tepkiyi
  doğrulamaz. Yeni görünüm/dikiş için kullanıcı görsel kabulü ayrıca bekleniyor.
- 2026-10-15: tasarım yazıldı. Önce yağmurun iki açık maddesi kapatıldı
  (`ATMOSPHERE_NEAR_PRECIP.md` devir notu, NEXT_BUILD_CHECKS 2026-10-15
  öğlen). Sıradaki: L1 (yağmur simülasyonu + kırılma composite + `v_rel`).

---

## 1. Neden ayrı katman

Havadaki damla `(hücre hash, timeline zamanı)`'nın saf fonksiyonudur
(`near_precip.glsl`). Camdaki damla öyle değildir:

1. **Durumlu.** Yapışır, büyür, eşik aşınca kayar, iz bırakır, birleşir,
   buharlaşır/erir. Bugünkü hali geçmişin integralidir.
2. **Odak dışı kırıcı mercek.** Ön merceğe ~0–3 cm; arkasındaki sahneyi
   büyütülmüş, ters ve bulanık gösterir. Rengi sahneden gelir.

Sonuç: damla rengi BİTMİŞ görüntüden kırılarak örneklenir → ışık sahneyle
kendiliğinden tutarlıdır (sabit renk yok, ayrı aydınlatma modeli yok).

## 2. İlkeler

1. **Tek kaynak.** Çarpma akısı, tip karışımı, damla boyu ve rüzgâr
   `nearState`'ten (kare başına bir kez, `cloud_sample.comp` mode 7). Cam
   sıcaklığı = atmosferin yüzey sıcaklığı (`precip3.x` + lapse, kamera
   yüksekliğinde). İkinci bir "yağış şiddeti" düğmesi yok.
2. **Zaman-saf (yeniden üretilebilir).** Durum timeline karesinin
   deterministik fonksiyonudur: sabit adım, tohum `hash(seed, kare, slot)`,
   checkpoint + yeniden simülasyon (§5). Duvar saati yok.
3. **Birikimi bozmaz.** RT'de composite birikmiş ORTALAMAYA uygulanır
   (tonemap öncesi), birikimi sıfırlamaz; damlalar ilerlese de yol izleme
   yakınsaması sürer.
4. **İki backend, tek GLSL.** Simülasyon ve composite backend'den bağımsız
   compute; RT ve RayFusion aynı shader'ı aynı girdilerle çağırır. OptiX
   dondurulmuş, almaz.
5. **Kural 1.** UI, script ve IPC aynı `LensWeather` servis çekirdeğini
   çağırır.

## 3. Fizik modeli

### 3.1 Çarpma akısı
```
Φ [damla/s] = Σ_form N_f · max(0, −v_rel·n) · A_lens · (1 − hood(θ))
v_rel = v_damla − v_kamera,   v_damla = yüzey rüzgârı + (0, −düşüş, 0)
n = kamera ileri yönü (dışa bakan cam normali ileri)
```
- `N_f`, düşüş, yarıçap, yüzey rüzgârı: `nearState` + `cloudSurfaceWind()`
  (havadaki damlayla aynı değer, aynı %60 yüzey payı).
- Rüzgâra dönük / yukarı bakan kamera çok ıslanır; sırtı dönük az; hızlı
  araç kamerası önden yoğun alır (`v_kamera` timeline'dan, iki karenin
  farkı — zaman-saf).
- `hood(θ)`: parasoley açısı; `v_rel` ile ileri eksen arası açı θ_hood'dan
  büyükse çarpma kesilir. Barınak (çatı altı) N2 örtü haritasıyla.
- Örnekleme: kare başına beklenen sayı `Φ·Δt` → Poisson, deterministik
  tohum; boyut dağılımı Marshall–Palmer (yağmur), tip tablosundan
  (diğerleri). Çarpmada büyük damla sıçrar: r > 1.5 mm → 1 ana + 2–4 küçük.

### 3.2 Camdaki damla (sıvı)
Durum (damla başına 32 B): ekran uv (2), hacim (1), hız uv (2), yaş/iz
tohumu (1), tip/bayrak (1), buz payı (1).

- **Yapışma–kayma:** temas açısı histerezisi. Kayma eşiği yarıçapı
  `r_c` (kaplama: hidrofobik ~1.2 mm, çıplak cam ~2.5 mm). `r < r_c`:
  sabit; `r ≥ r_c`: kayar.
- **Kuvvet (ekran düzleminde):** `g_⊥ = g − (g·n)n` (kamera pitch/roll'a
  göre; yukarı bakan kamerada ≈ 0 → kaymaz) + hava sürüklemesi
  `k·(v_rel)_⊥·|v_rel|` (araçta damla yukarı/yanlara kaçar).
- **Hareket:** dur-kalk (eşik civarında gürültülü sürtünme), değer
  gürültüsüyle kıvrılma (yol tohumlu, deterministik).
- **İz:** kayan damla adım başına küçük kütle bırakır (iz damlacıkları,
  r ~0.2–0.5 mm, yapışık); kütle korunur.
- **Birleşme:** üst üste binen damlalar `r³` toplanarak birleşir; hız
  momentumla. Birleşme eşiği aşmayı tetikler → gerçekteki zincirleme akış.
- **Buharlaşma:** yavaş, nem ve rüzgâra bağlı (kuru havada dakikalar).
- **Su filmi:** kaplama oranı ~%35'i aşınca (yoğun yağmur, birleşmiş
  damlalar) düzgün film; görüntü düşük
  frekanslı bulanıklaşır, film içinde akan dereler.

### 3.3 Kar, buz, karışık formlar
Form bilgisi `nearState` karışımından (en büyük iki tip):

| Form | Camda |
|---|---|
| Yağmur / çisenti | kırıcı damla; çisenti küçük, çok, nadiren kayar |
| Kar | yapışır (cam T ≤ ~1 °C); yarı saydam beyaz dağınık leke, kırmaz; birikir |
| Kar, T > 0 °C | erime hızı ∝ (T − 0)·ısı akısı; aynı su kütlesiyle damlaya döner (yoğunluk ~0.1 → yarıçap küçülür), sonra §3.2 |
| Sulu sepken | yarısı yapışır, hızlı erir |
| Graupel / pelet | çoğu seker; küçük buz kalıntısı, hızla erir |
| Dolu | yapışmaz; çarpmada sıçrama damlacıkları (sıvı) |
| Donan yağmur | T < 0: çarpınca donar → buz tabakası (pürüzlü, buzlu cam saçılması), kademeli kalınlaşır |

Kar lekesi görünümü: dağınık beyaz, ışığı görüntünün yerel ortalamasından
(düşük mip) alır, kenarı yumuşak, merkezi daha opak; erime ilerledikçe
saydamlaşıp damla görünümüne geçer (buz payı 1 → 0).

### 3.4 Optik (composite)
Kamera parametreleri: diyafram (f-sayısı), odak uzaklığı, odak mesafesi,
sensör. Cam ~kamera düzleminde → her zaman odak dışı.
- **CoC:** ince mercek, cam ön mercekte (d_g ≈ 0) → camdaki bir nokta
  giriş göz bebeği boyutunda bir disk olarak görünür: ekrandaki bulanıklık
  çapı ≈ (f / N) / (sensör yüksekliği) × (görüntü yüksekliği px) ×
  ölçek_lens (kalibrasyon, varsayılan 0.25). Geniş açı + küçük diyafram:
  keskin damla; tele + geniş diyafram: yalnız bulanık lekeler.
- **Yükseklik alanı:** damlalar yarım çözünürlükte bir yükseklik/kalınlık
  dokusuna yazılır (küresel başlık profili, CoC ile bulanıklaştırılmış).
- **Kırılma:** normal = ∇h; örnek uv = uv + k·∇h·kalınlık (ters görüntü
  için işaret büyük damlada döner). Bulanıklık: kalınlık + CoC'den mip.
- **Kenar:** tam iç yansıma kaynaklı ince koyu halka; parlak kaynaklarda
  (güneş, şimşek, lamba) speküler parıltı (görüntünün parlak piksellerinden).
- **Film / buz:** düşük frekanslı bulanıklaştırma + hafif kontrast kaybı;
  buzda saçılma (pus).
- Yer: HDR'de, DoF sonrası, bloom/tonemap ÖNCESİ (lens etkileri bloom'u
  doğru beslesin). RT: `tonemap.comp` girişindeki birikmiş ortalama
  üzerinde ayrı pass. RayFusion: `raster_post.comp` zincirinde DoF sonrası.

## 4. Veri ve maliyet

Bu bölüm ileri tasarım hedefidir; ilk L1'in gerçek kapasitesi ve maliyeti §10'da.

- Damla listesi: sabit kapasite 4096 × 32 B = 128 KB (GPU, kalıcı).
- Yükseklik dokusu: yarım çözünürlük R16F (+ buz payı R8).
- Simülasyon: 4096 iş parçacığı, kare başına 1–4 alt adım; birleşme için
  ekran ızgarası (hücre başına liste, 64×36).
- Composite: piksel başına 1 doku + 1–2 kırılmış örnek. Hedef: 1080p'de
  ≤ 0.5 ms; kamera kuru ve cam boşsa pass hiç çalışmaz.

## 5. Zaman-saflık ve önbellek

Bu checkpoint/ısınma tasarımı henüz teslim edilmedi. İlk L1 sabit adım ve sınırlı
tam snapshot geçmişi kullanır; kayıtlı olmayan eski yağışı uydurmaz (§10).

- Sabit adım `Δt = 1/60 s`; kare indeksine bağlı. Timeline oynarken her
  kare ileri simüle edilir.
- Checkpoint her 24 karede (128 KB + doku yok, doku listeden yeniden
  çizilir). Geri sarma / atlama: en yakın önceki checkpoint'ten yeniden
  simülasyon (≤ 23 adım).
- Isınma: ilk karede cam kuru başlamaz; son ~30 s'lik geçmiş `nearState`
  kaydı yoksa sabit durumla hızlı ön simülasyon (60 adım, büyük Δt).
- Timeline dururken durum donar; viewport'ta kamera hareketi durumu
  değiştirmez (cam kamerayla birlikte; yalnız `v_kamera` animasyondan).
- Önbellek anahtarı: (seed, ayarlar, atmosfer parametre özeti); değişince
  checkpoint'ler geçersiz.

## 6. Ayarlar, script, IPC (kural 1)

| Alan | Anlam |
|---|---|
| `enabled` | lens suyu açık/kapalı (kamera başına) |
| `coating` | `bare` / `hydrophobic` → kayma eşiği, temas açısı |
| `hood_deg` | parasoley açısı (0 = yok) |
| `seed` | damla tohumu |
| `wiper` (L3) | yok / aralıklı / sürekli |

- IPC: `camera.get_lens_weather`, `camera.set_lens_weather {...}`
  (doğrulama: aralık dışı → hata, durum değişmez),
  `camera.sample_lens_weather` → `{impact_rate_per_s, drop_count,
  coverage, film_fraction, ice_fraction, snow_fraction}`.
- Script: aynı servis çağrıları (`rt.camera.lens_weather`).
- Panel: sağ dock → Kamera → "Lens weather" (ayrı modül dosyası), salt
  okunur ölçüm satırı.

## 7. Kod yerleşimi (2000 satır kuralı)

| Dosya | İş |
|---|---|
| `include/Atmosphere/LensWeatherState.h` | ayarlar, doğrulama, JSON, lerp |
| `src/Atmosphere/LensWeatherService.cpp` | checkpoint, adım planı, ölçüm |
| `src/Backend/VulkanLensWeather.cpp` | buffer/doku, dispatch (iki backend ortak) |
| `shaders/lens_water_sim.comp` | çarpma + simülasyon + birleşme |
| `shaders/lens_water_draw.comp` | listeden yükseklik dokusu |
| `shaders/lens_water_composite.comp` | kırılma, kar/buz, film |
| `src/Api/RtApiLensWeather.cpp`, `src/UI/scene_ui_lens_weather.cpp` | IPC/script, panel |

Büyük dosyalara (`VulkanBackend.cpp`, `raygen`, `RtIpcMethodDescriptors`)
yalnız bağlantı satırı girer.

## 8. Fazlar

- **L1 — yağmur:** çarpma akısı (`v_rel`, hood), sıvı simülasyon (yapışma,
  kayma, iz, birleşme), kırılma composite, ayarlar + IPC + probe.
- **L2 — kar ve buz:** kar yapışma/birikim/erime (kütle korunumu), sulu
  sepken, graupel/pelet/dolu davranışı, donan yağmur buz tabakası.
- **L3 — bağlam:** N2 örtü haritası (barınak), kamera hızı kaynağı,
  silecek/üfleme, yerel ışıkta parıltı.

## 9. Kabul (probe taslağı, `Probe-LensWeather.ps1`)

1. Kamera kuru → `impact_rate 0`, `drop_count` zamanla 0'a iner.
2. Yağış altında yukarı bakış > yatay > aşağı bakış (çarpma oranı).
3. Rüzgâra karşı > rüzgâr arkası.
4. Geri sar → aynı karede `drop_count`, `coverage` birebir aynı.
5. Kar → T > 0'da `snow_fraction` düşerken toplam su kütlesi korunur
   (± %2).
6. Aralık dışı ayar reddedilir, durum değişmez.
7. ★ Sinsi olan: sayılar doğru ama görüntüde damla yok → composite pass
   bağlı değil (RT'de tonemap girişi, RayFusion'da DoF sonrası). Göz
   kontrolü şart.

## 10. L1 ilk kaynak teslimi — gerçek sözleşme (2026-10-15)

### Kullanıcı akışı ve ortak servis

Sağ Kamera özelliklerinde **Lens weather → Rain on camera glass** varsayılan
kapalıdır. Camı ıslatmak için kamera atmosferik yağmur/çisenti altında olmalı ve
timeline ilerlemelidir. İlk etkinleştirme/Clear glass kuru başlar; duvar saati veya
uydurulmuş 30 saniyelik ön geçmiş yoktur. Kamera yağmurdan çıkınca mevcut su
kayma/buharlaşmayla azalır. Ortografik görünüm ve OptiX bu katmanı uygulamaz.

Kamera nesnesi ve iki proje kayıt yolu aynı ayarları taşır; eski projelerde eksik
alan kapalı gelir. Geçici damlalar/history proje dosyasına yazılmaz. Her cihaz yalnız
aktif kameranın geçici cam durumunu tutar; kamera değiştirince yeni kuru epoch açılır,
etkin olmayan kameralar arka planda simüle edilmez. UI ve Python/IPC
`LensWeatherService` ve aynı doğrulayıcı üzerinden çalışır; UI-only yağış modeli yoktur.
Güneş/yağmur şiddeti için ikinci bir kontrol eklenmedi. Lens girdisi atmosferin
GPU `nearState` ölçümüdür; **Drops around the camera kapalıyken de** lens bu ölçümü alır.

| İşlem | Sözleşme |
|---|---|
| `camera.get_lens_weather` | Parametresiz, aktif kamera ayarları, Read |
| `camera.set_lens_weather` | Tek `patch` nesnesi; atomik doğrulama, SceneWrite |
| `camera.reset_lens_weather` | Parametresiz; sonraki display pass'te kuru epoch, SceneWrite |
| `camera.sample_lens_weather` | `backend=viewport|render` (viewport varsayılan); gerçek GPU readback, Read |

Python: `rt.camera.get_lens_weather()`, `set_lens_weather(**alanlar)`,
`reset_lens_weather()`, `sample_lens_weather(backend="viewport")`.

```python
# Dış IPC istemcisi; uygulamanın embedded script alanında IPC çalıştırma.
c.call("camera.set_lens_weather", patch={
    "enabled": True, "coating": "hydrophobic", "hood_deg": 0,
    "diameter_mm": 50, "seed": 17,
})
print(c.call("camera.sample_lens_weather", backend="viewport"))
```

`enabled` yalnız boolean; `coating=bare|hydrophobic`; `hood_deg=0..89` (0 parasoley
yok); `diameter_mm=10..200`; `seed=uint32`. Bilinmeyen alanlar, yanlış tip, boolean
yerine sayı ve sonlu olmayan/sınır dışı değerler mevcut ayarı değiştirmeden
reddedilir. Kaplama/hood değişimi mevcut suyu koruyup history'yi dallandırır;
enabled/seed/cam çapı değişimi veya Clear glass yeni kuru epoch açar.

### Simülasyon ve optik

- Aynı GPU forcing: ilk iki baskın tipin sayı yoğunluğu, yarıçap ve düşüş hızı;
  atmosferin %60 yüzey rüzgârı. L1 yalnız **rain/drizzle** kabul eder. Kar, sulu
  sepken ve donan yağmur bu sürümde sıvı yağmur gibi gösterilmez.
- Çarpma beklentisi `N max(0, -dot(v_drop-v_camera, forward)) A dt`;
  dairesel ön eleman alanı `pi D²/4`, alan koruyan kare UV eşlemesi. Kamera hızı
  timeline konum farkından; her bileşen ±300 m/s ile sınırlı. Durmuş timeline'da
  elle gezinme hız üretmez. Poisson hedefinin yerine bu ilk sürüm bounded,
  deterministik stratified/Bernoulli varış kullanır. Çok yüksek akıda temsilci
  damla ağırlığıyla beklenen su hacmi korunur.
- 1/60 s sabit adım, dispatch başına en fazla 8 adım. Kaplama eşiği bare 2.5 mm /
  hydrophobic 1.2 mm; cam düzlemine yansıtılan yerçekimi ve hava sürüklemesiyle
  kayma, hacim aktarımıyla iz, örtüşen damlaların hacim/momentumla birleşmesi.
  Kapasite dolunca yeni su mevcut damlaya katılır; `capacity_merges` ölçülür.
- Buharlaşma ilk sürümde sabit 180 s zaman sabitidir; gerçek nem/sıcaklık/kaplama
  kalibrasyonu değildir. Film, kaplama tahmininden türetilen optik bulanıklıktır;
  tam sıvı film/rivulet çözücüsü yoktur. `coverage` örtüşme için Poisson tahminidir.
- 512×288 RGBA16F alan, bulanık küresel başlık gradyanı, f-sayısı/odak uzaklığı/
  sensör yüksekliğinden CoC yaklaşımı; ortak GLSL ile 5 HDR örneği ve 0.94 geçirme.
  Kalibre edilmiş tam Snell/çok katmanlı lens izleyicisi değildir. Kaynak rengi
  sahnedendir; sabit beyaz yağmur overlay'i eklenmedi.
- RT'de iki tonemap resolve yolunun birikmiş HDR girdisi okunur, accumulation
  görüntüsüne yazılmaz; RT debug görüntüsü kırılmaz. RayFusion'da DoF sonrası,
  tonemap öncesi uygulanır. RayFusion kırılmış örnekleri aerial/fog uygulanmış HDR'den
  alır; her kırılmış örnek için tam DoF yeniden hesaplanmaz. Büyük bokeh + damla
  etkileşimi manuel kabul ister. Ham HDR/debug/CPU post/export yollarının tamamında
  eşdeğerlik iddiası yoktur; kayıt/export görünümü ayrıca kontrol edilmelidir.

### Geçmiş, sınırlar ve yaşam döngüsü

GPU state 256×32 B; tam snapshot halkası 601 adım (son 10 s), toplam state yaklaşık
4.73 MiB; optik alan 1.125 MiB; **cihaz başına yaklaşık 5.86 MiB**, iki VkDevice ayrı
kopya taşır. Yeni shader yoksa durum `pipelines_ready=false`, UI ölçümünde hata
görünür; rastgele veri okunmaz. Kaynaklar resize'da yeniden boyutlandırılmaz;
cihaz kapanışında GPU işi tamamlandıktan sonra serbest bırakılır.

Timeline dururken simülasyon ilerlemez. Aynı atmosfer/ayar girdileriyle son 10 s
içindeki kayıtlı adıma dönüş snapshot'ı geri yükler. Daha eski/uzak atlama kuru
epoch açıp `history_misses` artırır; görünmeyen tarih yeniden kurulmuş sayılmaz.
İleri yakın atlama 8 adımlık dispatch bütçesiyle yetişir; `step` ve `target_step`
eşit değilken sonuç hedef kareye yetişmemiştir. Atmosfer/iklim rüzgârı veya hood/
kaplama değişikliği eski replay aralığını keser, mevcut suyu korur. Kamera animasyon
eğrisi/poz geçmişi yeniden yazıldıysa Clear glass ile yeni dal başlatılmalıdır;
tam keyframe-input geçmişi ve sınırsız deterministik offline seek henüz yoktur.

Üst modeldeki GPU ızgaralı birleşme yerine ilk sürüm bounded sıralı 256-damla
birleşmesi kullanır. 0.5 ms hedefi **ölçülmedi ve kabul edilmiş performans değildir**.
Barınak haritası, buz/kar, silecek, yerel ışık parıltısı ve çok katmanlı kaplama L2/L3.

### Yeni derleme ve kabul

Codex build/shader derlemesi veya Studio açılışı yapmadı. Kullanıcı derlemesi:

- Studio C++ (yeni modüller VS projesinde kayıtlı).
- `lens_water_sim.comp → lens_water_sim.spv`
- `lens_water_draw.comp → lens_water_draw.spv`
- `tonemap.comp → tonemap.spv`
- `raster_post.comp → raster_post.spv`

`.glsl` ortak include'ları bu dört shader'a dahildir. Yeni binary/shader'larla
Studio'yu yeniden aç; eski çalışan uygulama yeni lens pipeline'ını yüklemiş sayılmaz.

```powershell
python scripts/test/rt_test_lens_weather_ipc.py --backend viewport
# Önce yağışlı sahneyi ve kameranın ıslak noktasını seç; timeline seek sahneyi etkiler:
python scripts/test/rt_test_lens_weather_ipc.py --backend viewport --wet
```

`render` backend için aynı komut, Rendered görünümü aktifken. Probe eski ayarları,
kamera target'ını ve timeline karesini geri yazar; geçici lens geçmişini temizler.
Seeking'in fizik/animasyon yan etkilerini geri almaz; kayıtlı test sahnesinde çalıştır.
Hatalı patch'in atomik reddi, ayar round-trip'i, gerçek GPU readiness/bütçe; `--wet`
ile pozitif çarpma/su hacmi, pause ve yakın geçmişte bit eşitliği denetlenir. Henüz
derlenmediği için probe koşturulmadı; statik kaynak/API kontrolleri canlı kabul değildir.

Göz: aynı yağışta RT ve RayFusion; yukarı/rüzgâra dönük kamera daha çok ıslanmalı;
yağış kesilince su aniden kaybolmamalı; kaplama kaymayı değiştirmeli; camdaki
şekiller sahneyi gerçekten kırmalı. Duraklatma, Clear/off, yakın geri sarma, yeniden
boyutlandırma, scene değişimi, proje save/reopen (ayarlar kalır, damlalar kuru başlar),
DoF, final çıktı ve GPU süre/bellek ölçümü ayrı kabul kapılarıdır.

Statik denetim: 671 IPC işleminin capability/security mirror ve descriptor güncelliği,
yeni kaynak/header VS kayıtları/XML, modül boyutları/delimiter'lar, dört shader'ın
include kapanışı, Python probe syntax, UTF-8 ve diff whitespace kontrolleri geçti.
Bu sonuçlar C++/GLSL derlemesini veya görünür lens etkisini doğrulamaz. `--wet`
probundan önce global timeline duraklatılmalıdır.

## 12. Tek atmosfer sıcaklığı — kamera ve yüzeyler

**Bu bölüm §11'in bağımsız cam sıcaklığı ayarının yerine geçer.** Yeni ayar veya
ikinci sıcaklık otoritesi yoktur: `AtmosphereClimate::ClimateState` tek sahibidir.
UI Atmosphere/Climate, Python `rt.world.set_climate` ve IPC `world.set_climate`
mevcut ortak core'dan geçer. `surface_temperature_k` Kelvin; yerel sıcaklık scene
Y + atmosphere altitude offset ile lapse profili üzerinden türetilir. 11 km'de
lapse durur. `world.sample_climate` aynı konumdaki kanonik ölçümdür.

- Kamera panelindeki sıcaklık girişi kaldırıldı; **Atmosphere at camera** yalnız
  ölçüm gösterir. L2 faz değişimi bu sıcaklığı kullanır. Ayrı termal atalet/cam
  ısıtıcısı yoktur; su/kar/buzun erime-donma hızı yine sabit-adım L2 modelidir.
- Sıcaklık profili değişirse mevcut cam kütlesi korunur, replay geçmişi dallanır.
  `camera.sample_lens_weather` → `temperature_source: atmosphere`,
  `atmosphere_temperature_c`. Eski `glass_temperature_c` ayar alanı yoktur.
- Eski proje/legacy kamera JSON'unda `glass_temperature_c` yalnız yüklemede
  atılır; sonraki save'e yazılmaz. Script/IPC üzerinden aynı alan gönderilirse
  `world.set_climate` kullanımını açıklayan hata gelir; patch atomik kalır.
- Vulkan RT/RayFusion Weather Surface termal tepkisi: kar örtüsünün uygunluğu
  ≤0 °C'de 1, ≥2 °C'de 0, arada smoothstep geçişlidir. Sıcak kar ayarı ıslak
  görünüm verir; soğuk ıslak yüzey buzumsu BRDF tepkisi alır. Miktar hâlâ mevcut
  surface ayarından gelir: yeni kalıcı kar/buz kütle deposu, erime suyu akışı veya
  tüm yüzeyleri otomatik yağışa bağlayan N4 simülasyonu teslim edilmedi.
- `world.sample_weather_local` / `rt.world.sample_weather_local(points=...)`
  her nokta için `air_temperature_c` ve `frozen_surface_share` döndürür. İkincisi
  kütle/coverage değildir, aynı termal uygunluk fonksiyonudur. Diğer sistemler
  C++'ta `sampleAmbient(position)` veya `World::sampleClimate(position)` kullanır.

GPU sıcaklık formülü `atmosphere_temperature.glsl` ile bulut yağış türü ve yüzeyler
arasında paylaşılır. Raster packet `visual.yzw` iklim profilini taşır. RT world
packet'ında emekli iki overlay padding alanı sea-level C / lapse float bitleri
için kullanılır; offset zaten world altitude alanındadır. GPU blok boyutu değişmez.

**Doğrulama/derleme:** C++ + tüm shader paketi, Studio yeniden açılmalı. IPC
capability/descriptor kontrolü, probe syntax/include kapanışı ve sayısal CPU/GPU
profil eşliği (yükseklik offset'i, inversiyon, tropopause dahil) geçti; gerçek GPU
kabulü bekleniyor. Dış `rt_test_lens_weather_ipc.py` ortak sıcaklığı üç yükseklikte
sorgular; emekli alan reddini denetler. `--snow` artık atmosfer sıcaklığını geçici
-5/20/-20 °C yaparak birikim/erime/donmayı ölçer ve eski sea-level sıcaklığı geri
yazar. İklim animasyonu olmayan kayıtlı test sahnesinde çalıştır; otomatik bulut
türetilmesi de bu değişimleri izler. Önceki kamera/frame/seek uyarıları geçerlidir.

## 11. L2 kar/buz kaynak teslimi — gerçek sözleşme

Kamera seç → **Camera glass weather (Lens weather)** → **Precipitation on camera
glass**. Yağış hâlâ atmosferin kamera konumundaki en büyük iki formundan gelir.
`glass_temperature_c` cam sıcaklığıdır (-80..80 °C, varsayılan 0); hava sıcaklığını
değiştirmez ve hava sıcaklığından otomatik türetilmez. UI, Python ve IPC aynı
atomik patch servisini kullanır. Ayar proje/legacy kamera JSON'una yazılır; eski
projelerde 0 olur. Sıcaklık değişimi mevcut birikimi silmez, replay geçmişini dallandırır.

| Form | İlk L2 davranışı |
|---|---|
| Yağmur/çisenti | Sıvı damla; negatif cam sıcaklığında zamanla donar |
| Kar | ≤1 °C tam tutunma, daha sıcak camda %35; kar lekesi, ısınınca erime |
| Sulu sepken | %50 tutunma; tutunan kütlenin yarısı kar, yarısı sıvı |
| Graupel / buz peleti | %15 tutunma; buz kalıntısı |
| Dolu | %8 tutunma; negatif sıcaklıkta buz, diğer durumda sıvı sıçrama kalıntısı |
| Donan yağmur | Negatif cam sıcaklığında doğrudan buz; diğer durumda sıvı |

Kar/buz fazları mevcut 32 bayt parçacığın son iki float'ında su-eşdeğer kütle
olarak saklanır. Snapshot boyutu ve 256 parçacık/601 adım sınırı değişmedi.
Birleşme ve kapasite taşmasında tüm faz kütleleri toplanır. Katı pay ≥%25 ise
parçacık tutunur; eridikten sonra mevcut kayma/iz yolu çalışır. Buharlaşma yalnız
sıvıyı azaltır; kar/buz süblimasyonu bu teslimde yoktur.

Faz değişimi deneysel birinci derece modeldir: kar erimesi `0.04*T /s`, buz
erimesi `0.012*T /s`, sıvı donması `0.05*abs(T) /s`; T=0'da değişmez. Faz
değişimi toplam su-eşdeğer hacmi korur. Ayrı bir enerji/latent ısı/cam ısı kapasitesi
çözücüsü değildir. Kar yoğunluğu 100, graupel 400, kompakt buz 917 kg/m³
varsayımıyla görsel parçacık boyutundan su-eşdeğer kütleye geçilir.

Yeni ikinci 512×288 RGBA16F alan kar örtüsü ve buz pusunu taşır (+1.125 MiB /
VkDevice, toplam lens kaynakları yaklaşık 6.99 MiB). Kar çoğunlukla akromatik,
grenli saçıcı örtü; buz bulanık, düşük kontrastlı katmandır. Ortak HDR composite
RT/RayFusion'da kullanılır. Yerel ekran radyansından yaklaşık aydınlatılır;
tam ortam ışığı entegrasyonu, kristal geometrisi ve fiziksel buz filmi çözücüsü yoktur.

`camera.sample_lens_weather`: `scope=rain_snow_ice`; `water_volume_m3` tüm fazların
su-eşdeğer toplam hacmidir. `liquid_water_volume_m3`, `snow_water_volume_m3`,
`ice_water_volume_m3`, `snow_count`, `ice_count`; `snow_fraction`/`ice_fraction`
toplam su kütlesi içindeki paydır, ekran coverage'ı değildir. `field_max_snow` ve
`field_max_ice` doğrudan çizilmiş katı optik alanından ölçülür. Ölçüm açık istemde
senkrondur; normal karelerde CPU readback yoktur.

**Derleme:** C++ + `lens_water_sim.comp`, `lens_water_draw.comp`, `tonemap.comp`,
`raster_post.comp`; ortak include'lar değişti, dördü birlikte yenilenmeli. Studio'yu
yeniden aç. Codex derleme/uygulama başlatma yapmadı.

**Dış probe:** Kayıtlı, kamerada ≥1 mm/saat kar olan sahne; global timeline
duraklatılmış, test edilen backend görünür:

```powershell
python scripts/test/rt_test_lens_weather_ipc.py --backend viewport
python scripts/test/rt_test_lens_weather_ipc.py --backend viewport --snow
```

`--snow` kamerayı geçici yukarı çevirir, 30 kare biriktirir, gerçek kar optik
alanı/pause/replay hash'ini kontrol eder; sonra aşağı çevirip camı 20 °C yapar,
8 karede kar hacmi azalırken sıvı hacmi artmasını ve toplamın %2 içinde kalmasını
ister. Ardından -20 °C'de 8 kare daha ilerleyerek sıvının buza geçmesini ve gerçek
buz optik alanının dolmasını denetler. Kamera/frame/ayarlar geri yüklenir; lens geçmişi temizlenir. Seeking'in diğer
sahne simülasyonlarına etkisi geri alınmaz. Yağmur için `--wet` regresyonunu da
çalıştır. Rendered görünümde `--backend render` kullan.

**Göz kabulü:** Soğuk kar altında tutunma/birikim, ılık camda erime ve sıvıya
geçiş; donan yağmurda buz pusu; Clear/off, pause, yakın replay, scene/save/reopen.
DoF kapalı/açık ve 16:9/1:1/dikey görünümde iki backend'i karşılaştır. Kullanıcının
bildirdiği hafif eliptiklik hâlâ açık görsel nottur; L2 bunu kapatmış sayılmaz.

L2 kaynak kontrolünde IPC capability/descriptor denetimi (671 işlem), Python
syntax/JSON, dört shader include kapanışı, faz değişiminin sayısal kütle/pozitiflik
kontrolü ve whitespace denetimi geçti. Bunlar GPU üzerinde canlı L2 kabulü değildir.
