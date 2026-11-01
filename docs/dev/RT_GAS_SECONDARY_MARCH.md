# Vulkan RT: ikincil sekmelerde gaz march'ı

> **Durum:** §24 doğal GPU probu düzeltildi; aynı hash ile 256/16=611/572ms, kalan per-entry maliyet açık. §23 ölçüm kimliği eklendi; reload sonrası gaz dağılımı ve kamera farkı doğrulandı, kök açık. §21 VRAM doğrulandı, birlikte 280 ms. §22 karışım Max Steps düzeltmesi kaynakta hazır, shader DERLENMEDİ. §20 geri alındı. §20.2 CANLI: kare 50, buhar kutusu %51.5, birlikte 573 ms. Kalan maliyet açık. §19 CANLI: kutu %93 → %11; birlikte GPU 329 ms (önceki oturum 1052 ms). Görsel kabul açık. §17 canlı: cache çalışıyor ama toplam maliyet yüksek. §18.1 küçük düzeltmeler DERLENMEDİ. Önceki ölçüm — 2026-10-28: çakışan iki gaz akışının maliyeti (§15–16b). §16 derlendi, canlıda bu sahneye ETKİSİZ: sahne genel karışım yolunda. Sıradaki iş aşağıdaki devir notunda.

## Devir notu (2026-10-28 — bu işi devralan ajan buradan başlasın)

**İş:** iki canlı gaz akışı (Matter domain'inin kaynak başına yayınladığı ayrı
RT hacimleri: "Vapor / Unattributed Gas" + "Point Flow Source 4") üst üste
binince RT 1 fps'e düşüyor. Ayrı ayrı 598 + 112 ms/örnek, birlikte 1157 ms
(§16b). Hedef: birlikte ≈ ayrı ayrı toplamı.

**Sahne ve ölçüm (derlemesiz):** `E:\rt_k\matter4emiter.rtp`, kare ~38–39,
Vulkan Rendered viewport, kamerayı oynatma. Ölçüm tek komut:
`python scripts/test/rt_probe_gas_overlap_cost_ipc.py` (görünürlüğü
`volume.set_material visible` ile değiştirir, sonunda geri alır). Kabul:
"both" satırında `secondary_density_samples` ~0 (bugün 3.33/yol) ve GPU ms
"sum" sütununa yakın.

**Kök (kodda doğrulandı):** çakışan segmentte iki yol var.
- Hızlı companion yolu: `volume_closesthit.rchit`, march gövdesi. §16 burada
  radyans önbelleğini çakışmaya açtı (derlendi, doğru, statik kontrolle
  sabitlendi) ama bu sahne oraya HİÇ girmiyor.
- Genel karışım: `shadeFogMixture` (`volume_mixture.glsl` ~544). Çağrı
  `volume_closesthit.rchit` ~4097'de; kapı `matterFogNeedsGeneralMixture`
  (`volume_mixture.glsl` ~85–107). Üyelerden biri emissifse
  (`emission_mode != 0`; bu sahnede ikisi de blackbody = 2), eğri uyarlamalı adım
  isterse, random walk/program varsa ya da faz/adım ayarları farklıysa bu yola
  gider. Bu yol sekme ışınlarında radyans önbelleğini kullanmıyor; ikincil
  örnekler ve maliyet buradan.

**Seçenekler (kullanıcıya sorulmadı, sıra önerisi):**
1. Derlemesiz deneme: buharın emisyonunu kapat (kullanıcı "buhar blackbody
   istemez" dedi): `volume.set_material target="matter:Physics Domain 1:gas"
   settings={"emission_mode": 0}` (anahtar adını `volume.get_material`
   çıktısından doğrula). Ama eğri (`density_curve[9]` uyarlamalı) da kapıyı
   açabilir; probe ile hangi yola girdiğini sayaçlardan anla
   (`radiance_cache_marches` artar mı). Kalıcı çözüm değil: duman gerçekten
   emissif olursa (yangın) yine genel yol.
2. Kalıcı: `shadeFogMixture` sekme ışınlarında (kamera segmenti değil,
   `worldData.w.volumeMarchLod` bit 21 açıkken) üyelerin radyans önbelleğini
   N üyeyle okusun: §16'daki birleştirme σ = Σσᵢ, J = Σσᵢ·Jᵢ / σ, her üye kendi
   yuvası (`volumeLightCacheSlot(index)`), kutusu dışında sıfır. Örnek:
   `volume_radiance_cache.glsl` → `volumeRadianceCacheMarch` (iki üyeli; döngüyü
   N'e genelle). Karışımın üye listesi `mf*` globallerinde (`mfCount`,
   MF_CAPACITY 16). Emisyon önbellekte J'nin içinde zaten var.

**Kurallar (bu dosyada tekrar öğrenildi):**
- Doldurma (`volumeRadianceCacheFetch` → `vrcComputeVertex`, ışık döngüsü)
  TEK çağrı yerinden: üye döngüsü + tablodan (`volumes.v[i]`) okuma. İki çağrı
  yeri ışık döngüsünü gömer. Taban: derleme 206 sn, `volume_closesthit.spv`
  982,912 B (`render.rt_pipeline_status` → `compile_seconds`); >%10 artış FAIL.
  Ağır yeni kod callable'a (RT_SHADER_COMPILE_LOAD.md).
- Bir ortam için tek köşe varyantı (companion'a özel varyant aynı hücreleri
  paylaşır, birbirini yeniden doldurur).
- OptiX dondurulmuş: dokunma. Derlemeyi (C++ ve `compile_shaders.bat`)
  kullanıcı alır. Kontrol: `check_volume_march_lod_contracts.py`; partiyi
  NEXT_BUILD_CHECKS'e sıralı listeyle bitir.
- Geçmiş: aynı sorun 10-24'te ölçüldü (MATTER_VOLUME_STREAMS_PLAN_20261024.md
  "Live RT regression"); kaynak başına malzeme denemesi 10-21'de geri alındı
  (GAS_SOURCE_MATERIAL_RENDER_AUDIT.md, derleme 314 → 495 sn). Tek hacim (B
  yolu) kullanıcı kararıyla ertelendi: üç+ akış gerekirse.

**Ayrı açık (bu işin dışı):** buhar tek başına 366–598 ms (yol başına 19–35
kamera örneği; soluk pus 41k hücrede, `occupiedLocalBox` `> 0` eşiği). Host
tarafı `viewport.render_frames` ms/kare GPU'nun 5–10×'i, kökü ölçülmedi.

Eski devir notu (2026-10-25, büyüyen adım / ~20 M kamera örneği): §14.

## 1. Sorun

Sahne: Physics Domain 1, kare 28, Vulkan RT. Fazlar `fluid.set_phase_output`
ile değiştirildi ve sonra geri yüklendi. Değerler yol başına.

| | yalnız splat | yalnız gaz | splat + gaz |
|---|---:|---:|---:|
| ms/örnek | 27 | 97 | 295 |
| hacme giriş | 0 | 1.75 | 2.29 |
| yoğunluk örneği | 0 | 58 | 33 |
| gölge yoğunluk örneği | 0 | 12.3 | 5.0 |
| ışık önbelleği isabet / kaçırma | – | 19.6 / 0.002 | 5.8 / 0.11 |

Splat + gaz, yalnız gazdan **daha az** örnek alıyor ama 3 kat pahalı.

- Gaz eklemenin farkı (295 − 27 = 268 ms) 33 örneğe bölününce örnek başına
  ~8 ms ediyor.
- Yalnız gazda aynı hesap (97 − 27 = 70 ms, 58 örnek) ~1.2 ms veriyor.
- Yani **örnek başına fiyat ~6 kat.**

Aday açıklama tutarlılık (coherence). Yalnız gazda örneklerin çoğu kamera
ışınından gelir: komşu pikseller aynı hücreleri, aynı sırayla okur. Splat
+ gazda örneklerin çoğu splat'lardan seken ışınlardan gelir. Bu ışınların
yönü rastgele, adımları warp içinde ayrışır ve bellek okumaları dağılır.

Bu açıklama henüz kanıtlanmadı: sayaçlar kamera ve ikincil örnekleri ayırmıyor
(§4 adım 0).

## 2. Ayarlarla yapılan deney (derlemesiz)

Gaz malzemesi `volume.set_material` ile değiştirildi, ölçüldü, sonra geri
yüklendi. Ayarlar bütün ışınlara uygulandı; kamera ışınları da dahil.

| | yalnız gaz | splat + gaz |
|---|---:|---:|
| temel (adım 0.5 voksel, en çok 256 adım, gölge 8/4) | 114 | 488 |
| adım ×4 | 55 | **246** |
| en çok 16 adım | 51 | 328 |
| gölge 2 örnek / her 16 adımda bir | 80 | 386 |

- Bu partide temel değerler (114 / 488), sayaç ölçümünden (97 / 295) yüksek
  çıktı. Aday sebep: her `set_material` ışık önbelleğinin dönemini sıfırlıyor,
  önbellek ölçüm penceresinde yeniden doluyor. Yalnız oranlara güven.
- **Adım sayısı asıl düğme.** Dört kat seyrek adım splat + gazı yarıya
  indiriyor. Gölge march'ı ikincil kalıyor (−%21).
- Bu düğmeler kamera ışınını da kabalaştırıyor. Görüntünün doğrudan görünen
  gazı bozulur. Hedef aynı kazancı yalnız ikincil sekmelerden almak.

## 3. Tasarım: ikincil segment için march LOD'u

İkincil = `PL_CAMERA_SEGMENT` biti kapalı olan iz. Raygen bu biti kamera
segmentinin her izinde kurar (`raygen.rgen` ~840, `bounce == 0`; ücretsiz
geçişler bounce'u artırmaz), sonraki sekmelerde kurmaz. Hacim closest-hit'i bunu
zaten okuyabiliyor. Yeni yük alanı ya da yeni iz yok. Emsal: `miss.rmiss` aynı
biti kamera ışınına tam bulut yürüyüşü, sonraki sekmelere sınırlı yürüyüş
vermek için kullanıyor (cloud_rt.glsl).

### 3.1 Adım: seyrek ve sınırlı

```
bool secondary = (payload.primaryMeta & PL_CAMERA_SEGMENT) == 0u;
float stepSize = max(vol.step_size, 1e-4) * (secondary ? lod.stepScale : 1.0);
int   budget   = secondary ? min(authoredStepBudget, lod.maxSteps) : authoredStepBudget;
```

- Varsayılan önerisi: `stepScale = 4`, `maxSteps = 32`.
- Jitter aynı kalır. Sabit adımlı quadrature'ın hatası adımla büyür, ama
  dolaylı ışık çok örnekte ortalanır. Hata görünürse banding olarak değil,
  ince gaz yapısının dolaylı katkısında yumuşama olarak görünür.
- Boş blok atlama (dense ve NanoVDB) zaten `baseStep` biriminde çalışıyor;
  ayrı dokunuş gerekmez.

### 3.2 Gölge: yalnız önbellek

İkincil segmentte sahne ışığı ve güneş için önbellek kaçırılırsa, strided
gölge march'ı yerine o örnekte `vlcFillCell` doldurma bütçesi kullanılır.
Bütçe bittiyse son geçerli değer kullanılır. Kaçırma oranı yol başına 0.11 ve
etki küçük, ama ikincil ışınlarda geri dönüş yolunun kendisi ayrışmayı büyütür.

### 3.3 Ambient gölgesi: ikincilde seyrek

Ambient gölgesi önbellekte yok. Yönü ışına bağlı (`0.55·up + 0.45·rayDir`),
o yüzden önbellek kanalına girmiyor. Her 8 adımda bir tam `lightMarchAcc`
çalışıyor. İkincil segmentte `ambientShadowStride` adım bütçesine eşitlenir:
ışın başına tek değerlendirme, ilk yoğun örnekte. Ambient yavaş değişen ışık;
ikincil sekmede tek değer yeterli.

### 3.4 Dokunulmayanlar

- Kamera segmenti, birebir bugünkü gibi.
- Random walk ve genel karışım (`matterFogNeedsGeneralMixture`) dalları. Bu
  sahnede çalışmıyorlar. Kendi adım bütçeleri var; ölçü göstermeden açılmaz.
- SDF izoyüzey dalı ve emisyon NEE seçimi.
- Yüzey gölge ışınının gaz geçirgenliği (`computeVolumeShadowTransmittance`).
  Zaten giriş noktasında önbellekten okunuyor.

## 4. Uygulama sırası

0. **Sayaç ayrımı (ilk iş, tek başına derlenebilir).** `density_samples`,
   `shadow_density_samples` ve `volume_rays` kamera ve ikincil olarak ayrı
   sayılır. Struct büyür; `render.volume_stats` iki alan grubu döndürür.
   Kabul: splat + gazda ikincil örnek payı > %60 ise §3 doğru hedefte demektir.
   Payı küçükse §3 yanlış hedef: ayrışma kamera ışınında, ve tasarım yeniden
   açılır.
1. **§3.1 ve §3.3, render düzeyinde bir ayarla.** Malzeme ayarı değil, render
   kalitesi politikası: aynı gaz farklı sahnelerde farklı LOD istemez.
   - Çekirdek: `rtapi::setVolumeSecondaryMarch(float step_scale, int max_steps)`
     ve okuyucusu. `step_scale` 1–16, `max_steps` 4–256. `step_scale = 1` ve
     `max_steps = 256` bugünkü davranışla birebir.
   - IPC: `render.volume_secondary_march` (Read) ve
     `render.set_volume_secondary_march` (Write). Python binding, panel
     kaydırıcısı (ışık önbelleği kaydırıcısının yanında), overlay satırı,
     descriptor üretimi.
   - GPU: world verisinde iki alan (VulkanWorldData'da boş bir pad varsa o).
     Shader aynaları yeniden adlandırılır.
   - Bu bir açma/kapama anahtarı değil, kalite ayarı. Kabul edildiğinde de
     kalır (ışık önbelleğindeki `max_cells` gibi).
2. **§3.2**, ancak adım 0 ikincil gölge kaçırma maliyetini önemli gösterirse.

## 5. Karar bekleyen

- **Varsayılan değerler.** Önerim 4 / 32. Daha agresif değerler (8 / 16) kazancı
  büyütür, ince yapının dolaylı katkısını daha çok yumuşatır.
- **Kamera segmentindeki ikinci geçişler.** Raygen kamera bitini "teleport" ve
  geçişlerde de kuruyor. Kamera ışını gazdan çıkıp yeniden girerse tam kalite
  alır. Bu doğru: görünen gaz.

## 6. Ölçüm ve kabul

- Aynı sahne, kare 28. `splat_gas_counters.py` düzeni: yalnız splat, yalnız
  gaz, splat + gaz; fazlar sonra geri yüklenir.
- **Beklenen:**
  - Splat + gaz 295 ms'den 150–200 ms bandına iner. Tahminin dayanağı, §2'deki
    adım ×4 deneyinin kazancının çoğunun ikincil örneklerden gelmesi.
  - Yalnız gaz neredeyse değişmez, çünkü örneklerinin çoğu kamera ışınından.
  - Yalnız splat değişmez.
- **Görsel kabul:** `step_scale = 1` ile karşılaştırma. Gaz içindeki
  splat'ların dolaylı aydınlanması ve gazın kendisi aynı görünmeli.
- ★ **Sinsi başarısızlık:** kamera biti ikincil izlerde de açık kalırsa LOD hiç
  devreye girmez. Görüntü aynı, maliyet aynı ve "kazanç yokmuş" diye
  okunur. Adım 0'ın sayaç ayrımı tam olarak bunu yakalar: ikincil örnek
  sayısı sıfırsa bit yanlış.

## 7. Adım 0 sonucu (2026-10-25, canlı): ölçüt kaldı, maliyetin yarısı ışık hesabı

Aynı sahne, kare 28. Toplamlar kare örneği başına, binlik birimde. Yol başına
değerler iki sahne arasında karşılaştırılamaz: `paths_traced` farklı (gas_only
~120 k, splat_gas ~324 k yol/örnek). §1'deki "örnek başına 6×" okuması bu
yüzden YANLIŞTI.

### 7.1 Sınıf ayrımı

| | gas_only | splat_gas |
|---|---:|---:|
| ms/örnek | 93 | 275 |
| hacim girişi kamera / ikincil | 120 / 89 | 232 / 513 |
| yoğunluk örneği kamera / ikincil | 4737 / 2512 | 6026 / 5036 |
| ikincil örnek payı | %34.7 | **%45.5** |

- Sınıflar toplamı tam kapatıyor (sınıfsız örnek 0). Kamera biti doğru.
- Girişlerin %69'u ikincil, ama ikincil girişler kısa. Örneklerin çoğu hâlâ
  kamera ışınında.

### 7.2 Ayrıştırma deneyleri (`volume.set_material`, geri yüklendi)

Her ölçümden önce 16 örnek beklendi; ışık önbelleği yeniden doldu ve temel
değerler tekrarlandı (275 / 286 / 286).

| splat_gas | ms | yoğunluk örneği | gölge örneği |
|---|---:|---:|---:|
| temel | 275–286 | 11062 | 1528 |
| adım ×4 | 150 | 3633 | 1107 |
| in-scatter yok, σt aynı | **148** | 12108 | 76 |
| shadow_steps 0 | 206 | 8686 | 0 |

Kabaca bütçe (splat_gas ~286 ms):

| kalem | ms | dayanak |
|---|---:|---|
| splat ve yüzeyler (hacimsiz) | ~31 | splat_only |
| yoğunluk march'ı | ~117 | in-scatter yokken 148 − 31, 12.1 M örnek → ~9.7 ms/M |
| in-scatter (ışık hesabı + gölge) | **~138** | 286 − 148 |
| bunun gölge/önbellek kısmı | ~55 | shadow_steps 0: −80 ms, ~25'i örnek düşüşü |

- shadow_steps 0 yüzey sis gölgesini de kapatıyor (hacim girişi 746 → 639),
  bu yüzden o satır kaba.
- Gölge dışındaki ışık hesabı her örnekte çalışıyor: ışık örneklemesi, segment
  ters kare ortalaması, faz, güneş geçirgenlik LUT'u, ambient ve yüksek mertebe
  gökyüzü. Bu ~80 ms, kamera ve ikincil örneklerde eşit dağılıyor.
- Adım ×4'ün kazancı (−125 ms) hem yoğunluk hem ışık maliyetini düşürdüğü
  için büyük.

### 7.3 §3 tek başına ne verir

İkincil örnekler ~5.0 M × (9.7 + ~12.5) ms/M ≈ 110 ms. ×4 seyreltme bunun
~2/3'ünü alır: **~−70 ms, 286 → ~215 ms.** Değerli ama hedefin yarısı.
Kamera tarafındaki ~6 M örneğin ışık hesabına dokunmaz.

### 7.4 Yeniden açılan tasarım (karar bekliyor)

- **B1 — ışık adımı seyreltme (kamera + ikincil).** Yoğunluk ve geçirgenlik
  her adımda kalır. In-scatter yalnız her `k` adımda bir değerlendirilir; arada
  son değer tutulur ve `k·dt` ile ağırlıklandırılır.
  - Önbelleklenen gölgeler zaten `shadow_stride` ile seyreltiliyor; B1 aynı
    fikri ışık hesabının geri kalanına uygular.
  - Jitter'lı başlangıç seyrek değerlendirmenin bantlaşmasını örnekler
    arasında dağıtır.
  - Beklenen: ışık maliyetinin (~138 ms) kabaca `1 − 1/k`'sı. `k = 4` için
    ~−100 ms.
  - Risk: ince, keskin aydınlık sınırlarında (gölge kenarı, ışık konisi) kamera
    görüntüsü yumuşar. Bu, `k`'yı kamera için küçük, ikincil için büyük tutmayı
    gerektirebilir.
- **B2 — §3'ün ikincil LOD'u**, B1'in üstüne. Ek kazanç esas olarak yoğunluk
  march'ından gelir (ikincil ~5 M × 9.7 ms/M ≈ 49 ms'nin 2/3'ü, ~−33 ms).
- **Önerim:** B1 önce, render düzeyinde iki ayarla: kamera `k` (varsayılan 2)
  ve ikincil `k` (varsayılan 4). İkisi 1'e ayarlanınca bugünkü davranışla
  birebir. B2 ölçümden sonra.
- Derlemesiz ön kontrol yok: hiçbir malzeme ayarı ışık hesabını yoğunluktan
  ayrı seyreltmiyor.

## 8. B1 + B2 yazıldı (2026-10-25, derlenmedi; shader + C++)

Tek ayar, dört alan; render düzeyinde, projeyle kaydedilmez:

| alan | aralık | varsayılan | ne yapar |
|---|---|---:|---|
| `camera_light_stride` | 1–16 | 2 | kamera segmentinde in-scatter her k örnekte bir |
| `secondary_light_stride` | 1–16 | 4 | sonraki sekmelerde in-scatter her k örnekte bir |
| `secondary_step_scale` | 1–16 | 4 | sonraki sekmelerde yoğunluk adımı çarpanı |
| `secondary_max_steps` | 0, 4–511 | 32 | sonraki sekmelerde adım üst sınırı (0 = malzemeninki) |

**1 / 1 / 1 / 0, bu ayardan önceki march'ın kendisi.** Ayrı bir kod yolu
değil; karşılaştırma görüntüsü bununla alınır.

- **Taşıma:** dünya tamponunda tek kelime, `volumeMarchLod`. Eskiden kullanılmayan
  `_aerialPad0` alanıydı; offset değişmedi, beş shader aynası yeniden adlandırıldı.
  Paketleyen fonksiyon `Backend::packVolumeMarchLod`: bit 0–3 kamera adımı − 1,
  4–7 ikincil adım − 1, 8–11 adım çarpanı − 1, 12–20 adım sınırı. Kelimenin 0
  olması referans march demek.
- **B1 (ışık adımı):** `if (sigma_s_local > 0.0)` bloğundaki in-scatter yalnız
  her k örnekte bir hesaplanır, aradaki örnekler son değeri kullanır. Yoğunluk,
  opaklık ve geçirgenlik her örnekte kalır. Boş alan atlamasından sonra (adım
  aralığı k veya fazla) tutulan değer kullanılmaz, yeniden hesaplanır.
- **Gölge yenileme:** önbelleklenen dört gölge (sahne ışığı, emitter, güneş,
  ambient) `step % stride == 0` ile yenileniyordu. k > 1 iken ışık hesabı bu
  adımları atlayabilir ve gölge bütün ışın boyunca ilk değerinde kalırdı.
  Artık `volumeShadowDue(step, stride, k)` kullanılıyor: pencere başına bir
  yenileme. k = 1 için tam olarak eski koşul.
- **B2:** kamera segmenti dışında `stepSize` adım çarpanıyla çarpılır, adım
  bütçesi sınırla kesilir. `baseStep` ve boş blok atlamaları bundan türer.
- **Sınıf:** gelen yükün `PL_CAMERA_SEGMENT` bitine bakılır (volume_closesthit
  main()). Sayaç ayrımıyla aynı kaynak.
- **Dokunulmayanlar:** random walk, genel karışım, SDF izoyüzey, yüzey gölge
  ışınının gaz geçirgenliği, emisyon (her örnekte kalır).
- **Arayüz:**
  - IPC: `render.volume_march_lod` (Read) ve `render.set_volume_march_lod`
    (Render; alanlar isteğe bağlı).
  - Python: `rt.render.volume_march_lod()` / `set_volume_march_lod(...)`.
  - Panel: ışık önbelleği kaydırıcısının altında dört kaydırıcı, bırakınca
    uygulanır.
- **Kontroller:** `scripts/test/check_volume_march_lod_contracts.py` (OK).
  A/B ölçümü `scripts/test/rt_volume_march_lod_ab_ipc.py` (ikisi de
  `x64/Release` altına kopyalandı).
- **Beklenen** (§7 bütçesinden): splat_gas ~286 → ~150–180 ms; gas_only
  ~93 → ~65 ms (kamera k = 2 ışık maliyetini yarıya indirir); splat_only
  değişmez.

## 9. Canlı A/B (2026-10-25)

### 9.1 Faz durumları (`rt_volume_march_lod_ab_ipc.py`, LOD 2/4/4/32)

Toplamlar kare örneği başına, binlik birimde.

| | ref ms | LOD ms | kamera örneği | ikincil örneği |
|---|---:|---:|---|---|
| yalnız splat | 12.8 | 12.3 | – | – |
| yalnız gaz | 93.0 | **58.1 (−%38)** | 4989 → 5095 | 2631 → 769 |
| splat + gaz | 286.6 | **152.0 (−%47)** | 6098 → 6171 | 5007 → 1639 |

Kamera örnek sayısı değişmedi; ikincil örnekler üçte bire indi. Kullanıcının
gözlemi, yani maliyetin splat ve katı yüzeylerden seken ışınların gaza yeniden
girip march etmesi olduğu, sayaçla doğrulandı. splat + gazda hacim
girişlerinin %69'u ikincil.

### 9.2 Tam sahne, alan alan (follow / sdf / fog, 128 örnek, aynı RNG)

| ayar (kamera k / ikincil k / adım / sınır) | ms | farklı piksel (>3/255) |
|---|---:|---:|
| 1/1/1/0 referans | 368 | – |
| 2/1/1/0 yalnız kamera ışık adımı | 374 | %0.54 |
| 1/4/1/0 yalnız ikincil ışık adımı | 342 | %1.52 |
| 1/1/4/32 yalnız ikincil adım | 273 | %2.83 |
| 2/4/4/32 hepsi | **223 (−%39)** | %4.23 |

- **Kamera ışık adımı (k = 2): kazanç yok, artefakt var.** Kutunun gölgeli
  alt yüzünde yatay dalgalı bantlar çiziyor. Varsayılan 1'e çekildi. Tam
  sahnede kazancın olmaması, kamera tarafındaki ışık maliyetinin önbellek ve
  gölge adımıyla zaten seyreltilmiş olmasıyla uyumlu; ölçülmedi.
- **İkincil adım asıl kazanç** (−95 ms). İkincil ışık adımı −26 ms.
- **Görünür fark sıvıda.** SurfaceSDF sıvının içinden ve üstünden görülen
  bölge LOD'da daha açık, koyu benekler azalıyor (ortalama +6.6/255; değişen
  piksellerin hepsi bu bölgede). Kırılan ve yansıyan ışınlar ikincil segmentte,
  bu yüzden gazı kaba adımla geçiyorlar. Seyrek adımlı Beer-Lambert ortalaması
  (Jensen) geçirgenliği yukarı yanlılar, sonuç açılma. Gazın kamera görüntüsü
  ve splat + gaz durumu neredeyse aynı: değişen piksel %0.77, ortalama
  −2.1/255.
- **Karar (kullanıcının):** sıvıdaki açılma kabul mü? Kabul değilse seçenek:
  SDF devrinden gelen ışınlarda (kırılma) ikincil adımı kapatmak; ölçüm bunun
  maliyetini gösterir.

## 10. Zemin düzlemi: nükleer patlama sahnesi (2026-10-25, canlı)

Kullanıcı gözlemi: domain'in altında bir düzlem açılınca maliyet hemen ikiye
katlanıyor. Sahne "Nuclear Detonation (Cinematic)": emisyonlu gaz, yere yayılan
etek, `Plane_1` (144 m, y=0), kamera y≈5.4. Toplamlar kare örneği başına,
binlik birimde.

| durum / ayar | ms | yol | kamera / ikincil giriş | kamera / ikincil örnek |
|---|---:|---:|---|---|
| düzlem gizli, 1/1/1/0 | 91 | 169 | 169 / 0 | 5542 / 0 |
| düzlem gizli, 4/2/2/32 | 76 | 178 | 178 / 0 | 6162 / 0 |
| düzlem açık, 1/1/1/0 | **438** | 1090 | 296 / 186 | 19566 / 13080 |
| düzlem açık, 1/4/4/32 | 221 (−%50) | 1093 | 299 / 188 | 19849 / 3830 |
| düzlem açık, 4/2/2/32 | 223 | 1104 | 310 / 191 | 20996 / 4182 |
| düzlem açık, 4/4/4/32 | 211 | 1104 | 310 / 191 | 20995 / 3890 |
| düzlem açık, 4/8/8/16 | **189 (−%57)** | 1105 | 311 / 191 | 21114 / 2077 |

- Düzlem maliyeti 4.8 katına çıkarıyor, yalnız sekmeler yüzünden değil.
  Ekranın alt yarısı artık zemin: yol sayısı 169 → 1090 bin. Zemine giden
  kamera ışınları yere yayılan gaz eteğinden yatık geçiyor; kamera örnekleri
  3.5 katına çıkıyor (5.5 → 19.6 M).
- Düzlemden seken ışınlar gaza giriş başına ~70 örnek alıyor: etek boyunca
  yatay march. LOD bunu üçte bire indiriyor. Kullanıcının hipotezi bu kısım
  için doğru.
- LOD'dan sonra kalan maliyetin çoğu kamera örnekleri (~20 M). Kamera ışık
  adımı 4 burada ~%5–10 kazandırıyor (211'e karşı 221). Kullanıcı yatay
  blokları kabul edilebilir buldu, ama önceki sahnede kazanç yoktu ve bant
  çizmişti. Varsayılan şimdilik 1.
- **Yan bulgu:** panelde görünür yapılan düzlem ilk denemede RT render'a
  girmedi; ikinci açışta girdi. Kullanıcı bunun ara sıra olduğunu söylüyor.
  Görünürlüğü IPC'ye açan yan oturuma bildirildi.
- **Sıradaki aday:** zemine yatık bakan kamera ışınlarının gaz eteği march'ı.
  Örnek başına maliyet değil, örnek sayısı büyük. Etek ince ve geniş; bu
  RT_GAS_OPTICAL_PERFORMANCE'taki "ince/geniş içerik için çoklu AABB"
  adayıyla aynı aile.

## 11. Yoğunluğa göre uyarlanan adım (2026-10-25, derlenmedi; shader + C++)

**Neden bu, ve neden mesafe LOD'u ya da çoklu AABB değil.** Nükleer sahnede
ölçüldü (Nuclear Gas):
- Voksel 0.133 m, adım 0.7 voksel = 9.3 cm, adım sınırı 192.
- Kamera ~54 m uzakta, FOV 30.8°: pikselin kapladığı genişlik ~3 cm, adımdan
  küçük. Mesafe/piksel LOD'u bu kamerada hiç devreye girmez.
- Segmentlerin %27'si boş alan atlamasıyla zaten geçiliyor ve atlanan
  segmentler örnek sayılmıyor. Çoklu AABB kalan ~20 M kamera örneğini
  azaltmaz; onlar gazın dolu kısmında.
- Kalan örneklerin büyük kısmı ince, seyrek etekte. Orada tek adımın optik
  kalınlığı küçük, yani adım büyütülebilir.

**Kural.** Her örnekte segment uzunluğu, kafes adımının 1, 2, 4 veya 8 katı:
- Segmentin tepe kanal optik kalınlığı (`max(σt_rgb) · baseStep · k`) bütçenin
  (`adaptive_tau`) altında kaldıkça k ikiye katlanır; en çok
  `adaptive_*_max_stride`'a kadar.
- Kafes ve jitter değişmez; sonraki örnek yine bir kafes noktasına düşer
  (`step += segStride`).
- Beer-Lambert segment boyunca analitik. Hata segment başına τ ≤ bütçe ile
  sınırlı.
- Yoğun çekirdekte ve emisyonlu bölgede τ büyük olduğu için adım kafes adımında
  kalır.

**Gölge yenileme tuzağı.** Atlamalı march, `volumeShadowDue`'nun pencere
testini ışın boyunca hiç tutturamayabilir; gölge ilk değerinde donar. Uyarlanan
adım açıkken test "son ışık değerlendirmesinden bu yana pencere sınırı geçildi
mi" oluyor (`heldInscatterStep`'e göre). Kapalıyken eski koşul aynen duruyor,
bu yüzden referans birebir.

**Ayar** (`render.set_volume_march_lod`, mevcut LOD'a eklendi):

| alan | değerler | varsayılan |
|---|---|---:|
| `adaptive_camera_max_stride` | 1 (kapalı), 2, 4, 8 | 4 |
| `adaptive_secondary_max_stride` | 1, 2, 4, 8 | 4 |
| `adaptive_tau` | 0.0079–0.99 (1/128 adımla saklanır) | 0.05 |

Paketleme: bit 21–22 kamera log2, 23–24 sekme log2, 25–31 τ·128. Referans:
1/1/1/0 ve iki uyarlanan adım 1.

**Not:** Malzemedeki `adaptive_stepping` alanı Vulkan RT'de hiçbir şey
yapmıyor; yalnız `primaryStep()` ile adım boyunu vokselden türetiyor. Bu ayar
onun yerine geçmez, render düzeyindedir.

**Beklenen:** Düzlemli nükleer sahnede kamera örnekleri 19.6 M'den belirgin
aşağı, ms 221'den (1/4/4/32) aşağı. Etek ve dumanın ince kenarları en çok
yararlanır.

**Risk:** İnce ama parlak emisyonlu gaz (düşük yoğunluk, yüksek sıcaklık)
seyrek örneklenir; ince sıcaklık ayrıntısı yumuşayabilir. A/B görüntüsünde
patlamanın etek ateşine bakılmalı.

## 12. Canlı ölçüm: uyarlanan adım GERİLEME, sekme LOD'unda sınırın bedeli (2026-10-25)

Nükleer sahne, `Plane_1` görünür (`scene.set_visible` ile), aynı oturumda
arka arkaya. Kare ve kamera önceki ölçümlerle aynı olmayabilir; yalnız bu
tablonun kendi içindeki karşılaştırmalar geçerli. 128 örnek. "Ortalama fark",
referansa göre işaretli luma farkı /255 (yanlılık); farklı piksel yüzdesinin
çoğu RNG ayrışmasından gelen gürültü.

### 12.1 Uyarlanan adım (§11)

| ayar | ms | kamera örneği (M) | yol (k) | ortalama fark |
|---|---:|---:|---:|---:|
| referans | 451 | 20.0 | 1101 | – |
| yalnız uyarlanan 4 / τ 0.05 | 410 | **23.8** | 1173 | +2.3 |
| LOD 1/4/4/32 | 241 | 20.3 | 1104 | +8.1 |
| LOD + uyarlanan 4 | **249** | 23.9 | 1174 | +9.2 |
| LOD + uyarlanan 8 / τ 0.1 | **260** | 24.1 | 1189 | +10.7 |
| düzlem gizli, referans | 108 | 9.1 | 179 | – |
| düzlem gizli, LOD + uyarlanan 4 | **136** | 13.8 | 254 | – |

**Uyarlanan adım örnek sayısını AZALTMADI, ARTIRDI** (kamera 20.0 → 23.9 M),
görüntüyü de açtı. Kök neden: segment uzunluğu segmentin BAŞINDAKİ yoğunluğa
göre seçiliyor. İnce kenardan yoğun bölgeye giren segment o bölgenin
söndürmesini atlıyor. Gaz şeffaflaşıyor, ışınlar daha derine iniyor, daha çok
örnek ve daha çok yol ortaya çıkıyor (yol 1101 → 1174 k). Ateş kenarları açılıyor.
Doğru bir sürüm segmentin üst sınır yoğunluğunu (majorant) okumalı; o ayrı bir
tasarım. **Öneri: sök** (CLAUDE.md kural 5; ölü yol yaşatılmaz). Canlıda
1/1 ile kapatıldı.

### 12.2 Sekme LOD'u, alan alan (uyarlanan kapalı)

| ayar (sekme ışık adımı / adım çarpanı / sınır) | ms | ikincil örnek (M) | ortalama fark |
|---|---:|---:|---:|
| referans 1/1/0 | 404 | 13.2 | – |
| 4/1/0 | 336 | 13.2 | **+0.02** |
| 1/2/0 | 330 | 8.8 | +0.8 |
| 1/4/0 | 262 | 5.6 | +3.8 |
| 1/1/32 | 232 | 4.3 | **+8.1** |
| 4/2/128 | 282 | 8.7 | +1.0 |

- **Sekme ışık adımı 4 bedava:** −68 ms, yanlılık yok.
- **Zemini açan 32 adım sınırı.** Sınır, uzun bir kirişi 32 parçaya böler.
  Eteği yatay geçen sekme ışınında adım 0.09 m'den ~0.7 m'ye çıkıyor. Ateşli
  ve heterojen gazda seyrek Beer-Lambert ortalaması geçirgenliği yukarı
  yanlılar (Jensen), sonuç zemin ~%6 aydınlanıyor. Önceki sahnede (homojen
  sis) bu etki küçüktü.
- **Önerilen yeni varsayılan: 1 / 4 / 2 / 128** (kamera ışık adımı / sekme
  ışık adımı / sekme adım çarpanı / sekme sınırı). 404 → 282 ms (−%30), +1/255.
  Agresif seçenek 1/4/4/32: −%47, +8/255.

### 12.3 Karar ve uygulama (2026-10-25)

Kullanıcı kararı: uyarlanan adım söküldü, varsayılan 1/4/2/128.
- Shader: `segStride`, uyarlanan adım çözümü ve pencere geçişli gölge testi
  kalktı. `volumeShadowDue(step, stride, lightStride)` eski haline döndü.
  Yerine, neden kaldırıldığını anlatan bir yorum kondu.
- C++/IPC/Python/panel: üç `adaptive_*` alanı kalktı. Paketlemede 21–31.
  bitler boş.
- Kontrol script'i, uyarlanan adımın majorant sınırı olmadan geri gelmesini
  FAIL eder.

## 13. Bulut march'ından iki fikir (2026-10-25, derlenmedi; shader + C++)

Kullanıcı sorusu: atmosfer bulutu RT kapalıyken (RayFusion) makul kalitede ve
ucuz; gaz neden pahalı? Bulutun kodundan çıkan karşılaştırma:

| | bulut (RayFusion) | gaz (yol izleyici) |
|---|---|---|
| karede march edilen piksel | 1/4 (`realtime_interleave = 2`) + yeniden yansıtılan geçmiş | her piksel, her örnek |
| sekme ışını | yok; çoklu saçılma analitik (`CLOUD_MS_OCTAVES`) | gaza yeniden girer |
| RT'de buluta giren sekme ışını | march yok, panorama (`rt_secondary_full_march = false`) | tam march |
| adım | uzaklıkla ve T düştükçe büyür (`deepGrow`, 2.5× kadar) | sabit 0.7 voksel |
| yoğunluk | ucuz şekil, ayrıntı yalnız T > 0.15 | her örnekte tam NanoVDB |
| güneş gölgesi | 3–6 adet, giderek açılan dokunuş | ışık önbelleği |
| ambient | yukarı/aşağı 2 dokunuş | her 8 adımda bir tam march |

Fark algoritmadan çok mimaride. Bulutun RT'deki iki kararı gaza taşındı.

### 13.1 Geçirgenliğe göre büyüyen adım (bulutun `deepGrow`'u)

- Kafes adımı T > 0.35 iken 1; T 0.35'ten 0.05'e düştükçe `*_transmittance_growth`
  değerine (en çok 4) kadar büyür:
  `stride = 1 + round((G-1) · (1 − smoothstep(0.05, 0.35, T)))`.
- Hatası kalan katkıyla (≤ T) sınırlı. §12'de sökülen uyarlanan adımın tersi:
  o, segmentin başındaki yoğunluğa bakıyor ve ince kenardan yoğun bölgeye
  girişi atlıyordu.
- Büyüme açıkken önbellekli gölgelerin yenilenmesi pencere geçiş testine
  dönüyor. Kapalıyken eski koşul aynen duruyor.
- Varsayılan kamera 3, sekme 3 (bulut 2.5'e kadar çıkıyor).

### 13.2 Sekme ışınları için radyans önbelleği (bulutun panorama kararının karşılığı)

- **Yeri:** Işık önbelleği ızgarasında, köşe başına dört yeni kanal:
  - 6: yoğunluk;
  - 7–9: kaynak radyansı J (RGB).
  Kanal sayısı 6 → 10. Işık eşlemesi artık ayrı `VLC_LIGHT_CHANNELS = 6`'ya
  bağlı; yoksa ışık 5–8 radyans kanallarına düşerdi.
- **J'nin anlamı:** Bir segment `T · J · (1 − exp(−σt·dt))` katkı verir; tam
  march'ın `transmittance · (inscatter + emis) · one_minus_sampleT` ifadesiyle
  aynı biçim. Bileşenleri:
  - güneş ve ışık 0–4: yalnız izotropik faz; gölgeleri ışık önbelleğinden;
  - gökyüzü ambient'i: yukarı doğru iki yoğunluk dokunuşuyla gölgelenir, üstüne
    yüksek mertebe gökyüzü görünürlüğü eklenir;
  - emisyon: march ile aynı fonksiyonlar.
- **Doldurma:** Işık önbelleği gibi tembel ve CAS ile.
  - Sahiplik yoğunluk kanalının kelimesinde. Kaynak kanalları önce, yoğunluk en
    son yazılır.
  - Gereken bir ışık önbelleği okuması henüz yoksa sahiplik geri bırakılır ve
    ışın tam march'a düşer.
  - Etiket şunların karışımı: slot anahtarı, kanal, ışık durumu (güneş + ışık
    0–4, çağrı başına bir kez hash'lenir) ve malzeme parametreleri.
- **Okuma:** Yalnız kamera segmenti dışında. Uygunluk düz march'ın kendi kısa
  yollarıyla aynı: companion yok, malzeme programı yok, gökyüzü bulutu değil.
  - Kutu hücre boyunda adımlarla trilinear okumayla geçilir (en çok 256 adım).
  - Bir köşe eksikse ışın tam march'a düşer.
  - Sayaçlar: `radiance_cache_marches` ve `radiance_cache_fallbacks`.
- **Yaklaşımlar** (yalnız sekme ışınlarında):
  - izotropik faz;
  - alan ışıkları ve 5. ışıktan sonrakiler yok;
  - emitter NEE terimi yok (ateşin kendi dumanını aydınlatması);
  - ambient gölgesi kaba.
  Bunlar düzlemli sahnede zeminin dolaylı aydınlanmasını değiştirebilir;
  görüntü karşılaştırması şart.
- **Bellek:** Slot başına 64³ × 10 × 8 B ≈ 21 MB (önce 12.6 MB). 512 MB bütçe
  aşılırsa hücre sayısı mevcut kuralla küçülür.

### 13.3 Ayar (`render.set_volume_march_lod`'a eklendi)

| alan | değerler | varsayılan |
|---|---|---:|
| `secondary_radiance_cache` | bool | açık |
| `camera_transmittance_growth` | 1 (kapalı)–4 | 3 |
| `secondary_transmittance_growth` | 1–4 | 3 |

Paketleme: bit 21 önbellek, 22–23 kamera büyümesi − 1, 24–25 sekme büyümesi − 1.
Referans: 1/1/1/0, önbellek kapalı, büyüme 1/1.

### 13.4 Beklenen

- **Düzlemli nükleer sahne:**
  - Sekme örnekleri neredeyse sıfıra iner; geri düşme sayısı birkaç örnekten
    sonra küçülmeli.
  - Kamera örnekleri, yoğun başlıktan sonra büyüyen adım sayesinde azalır.
- **Risk:** Zeminin ateşten aldığı dolaylı ışık. Önbellekte emitter NEE yok ama
  emisyon var. Zemin referansa göre koyu ya da açık kayarsa ilk şüpheli J
  bileşenleri.

## 14. Canlı ölçüm: radyans önbelleği kazanç, büyüyen adım değil (2026-10-25)

Nükleer sahne, `Plane_1` görünür, aynı oturum. Kamera bir ara elle oynatıldı,
ama konum ölçümden önceki değerle aynı çıktı ve fark görüntülerinde kenar
kayması yok. Süreler iki koşuda alındı (ikincisi ters sırada); sayaçlar iki
koşuda birebir aynı. Toplamlar kare örneği başına; zemin/başlık farkı luma /255
(zemin referansı 111).

| ayar | ms (koşu 1 / 2) | kamera örneği (M) | sekme örneği (M) | önbellek march / geri düşme (k) | zemin | başlık |
|---|---|---:|---:|---|---:|---:|
| referans | 501 / 515 | 19.9 | 13.4 | – | – | – |
| 1/4/2/128 | 307 / 363 | 20.0 | 8.8 | – | +2.4 | 0.0 |
| **yalnız önbellek** | **247 / 239** | 20.2 | **0.0** | 191 / **0** | +5.3 | 0.0 |
| yalnız büyüme 3/3 | 421 / 468 | 20.2 | 13.2 | – | −0.4 | **−1.8** |
| varsayılan (hepsi) | 252 / 237 | 20.5 | 0.0 | 191 / 0 | +5.4 | −1.8 |

**Radyans önbelleği:**
- Sekme örneklerinin tamamını kaldırıyor. Ölçüm penceresinde önbellek tam
  doluydu: geri düşme sıfır.
- Referansa göre −%53, 1/4/2/128'e göre −%25…−%35.
- Bedeli zeminin ~%5 aydınlanması. Normal bakışta yan yana görüntüde fark
  seçilemiyor. Aday kaynaklar ölçülmedi: izotropik faz (gaz g=0.3 ileri +
  geri lob), yukarı doğru iki dokunuşlu ambient gölgesi (geniş, alçak etekte
  gölgeyi az tahmin edebilir).
- Kayıp emitter NEE terimi zemini koyulaştırırdı; görülen etki ters yönde.

**Geçirgenliğe göre büyüyen adım:**
- Kamera örneği azalmadı (19.9 → 20.2 M).
- Yoğun başlıkta T 0.35'ten 0.001'e birkaç adımda iniyor; büyüme o kısa
  aralıkta neredeyse hiç adım kurtarmıyor. Örneklerin çoğu T > 0.35 olan
  ince bölgelerde. Bulutta işe yaramasının nedeni bulutun km ölçeğinde kalın
  ve yumuşak olması.
- Başlıkta ölçülebilir bir kararma bırakıyor (−1.8/255).
- Süre farkı (−%9…−%16) iki koşu arasındaki gürültü kadar.
- **Öneri: sök** (kural 5).

### 14.1 Karar ve uygulama (2026-10-25)

Kullanıcı kararı: büyüyen adım söküldü, radyans önbelleği varsayılan açık.
- Shader: `growMax`, `growStride` ve pencere geçişli gölge testi kalktı.
  `volumeShadowDue(step, stride, lightStride)` sabit kafes biçimine döndü.
  Segmentin yanındaki yorum, denenip sökülen iki değişken adımı anlatıyor.
- C++/IPC/Python/panel: iki `*_transmittance_growth` alanı kalktı. Paketlemede
  22–31. bitler boş.
- Kontrol script'i, iki değişken adımdan biri majorant sınırı olmadan geri
  gelirse FAIL eder.

## 15. Canlı ölçüm: kaynak başına gaz akışları üst üste binince maliyet 2.5× (2026-10-28)

Sahne: kullanıcının 4 emitterli Matter sahnesi (`E:\rt_k\matter4emiter.rtp`,
86³ ızgara, 5.8 cm), kare 39, Vulkan RT viewport. Kullanıcı gözlemi: gaz
görünmezken 50 fps, görünürken 2 fps; buhar ve duman domain'in küçük bir
kısmında. Matter domain'i gazı **kaynak başına iki ayrı RT hacmi** olarak
yayınlıyor: "Vapor / Unattributed Gas" (artık: kaynama buharı + atfedilmemiş
gaz) ve "Point Flow Source 4" (duman). İkisinin de malzemesi aynı (yoğunluk
çarpanı 11.1, kesme 0.01, eğri ilk segmentte ~2.8× büyütüyor, blackbody).

Ölçüm (`render.gpu_timing_stats` kamera RT dispatch'i, bir örnek;
`volume.set_material visible` ile A/B):

| görünür | GPU ms/örnek | kamera yoğunluk örneği/yol | ikincil/yol |
|---|---|---|---|
| hiçbiri | 12.5 | 0 | 0 |
| yalnız duman | 43 | 2.1 | 0.0 |
| yalnız buhar | 265 | 19.0 | 0.05 |
| **ikisi** | **790** | 21–22 | 1.3–2.1 |

- **Toplamın 2.5 katı:** örnek sayısı toplama yakın (21 ≈ 19 + 2), ama süre
  265 + 43 = 308 yerine 790. Üst üste binen iki hacimde **örnek başına
  maliyet ~3×**. İkincil (sekme) yoğunluk örnekleri yol başına 0.05 → 2.1
  (45×). Kök ölçülmedi; adaylar: çakışan hacimlerde her adımın iki ızgara ve
  iki malzeme okuması, gaz↔gaz devirleri (`gas_handoffs` 0.2 → 0.3,
  `teleports` 0.39 → 0.47), ikinci hacmin emisyonu yüzünden sekme ışınlarının
  hacme girmesi.
- **Buhar kutusu neredeyse bütün domain:** `occupiedLocalBox`
  (VDBVolume.cpp) kutuyu `> 0` olan her hücreyle büyütüyor; buhar 41,283
  hücrede, ortalama 0.0015 (duman 3,622 hücre, ortalama 0.069). Ama asıl
  maliyet kutu değil: buharın kesmesini 0.01 → 0.2'ye çıkarmak örnek
  sayısını değiştirmedi (20.95 → 20.74), süreyi 798 → 563 ms indirdi. Yani
  march bu adımları atıyor; süre kesmeyi geçen örneklerin gölgelendirmesinden.
  8³ majorant atlaması (`empty_segments_skipped` 19.5/yol) zaten çalışıyor.
- İlk karede tek seferlik ~2.4 s/kare (ışık önbelleği dolumu / yükleme);
  sonraki örnekler yukarıdaki gibi.
- Kural 1 açığı: örnek başına maliyetin hangi hacim sayısında alındığı
  (1 / ≥2 hacim çakışması) sayılmıyor; kök bu yüzden aday listesinde.

Sonraki ölçüm: çakışma sayacı (adım başına kaç gaz hacmi örneklendi) ve
ikincil ışınların hangi hacme girdiği; düzeltme adayı iki akışı RT'de tek
hacim + akış başına malzeme ağırlığı olarak örneklemek.

### 15b. Kullanıcı tahminleri sınandı (2026-10-28, aynı sahne, ısınmalı 6 kare ortalaması)

| yol başına | buhar | duman | toplam | ikisi |
|---|---|---|---|---|
| GPU ms/örnek | 366 | 86 | 451 | **964** |
| hacim ışını | 1.22 | 0.32 | 1.54 | 1.42 |
| sekme sınırına takılan yol | 0.028 | 0.042 | 0.070 | 0.023 |
| kamera yoğunluk örneği | 34.6 | 3.7 | 38.3 | 39.3 |
| **ikincil yoğunluk örneği** | **0** | **0** | **0** | **3.33** |
| majorant sorgusu | 2.7 | 0.8 | 3.5 | 5.2 |

- "Her gaz hacmi için tekrar ışın" ÇÜRÜDÜ: ikisi birlikte hacim ışını toplamdan az.
- "Işık iki hacim arasında sekme sınırına kadar" ÇÜRÜDÜ: sınıra takılan yol en az
  ikisi birlikteyken (%2.3).
- Fark ikincil (sekme) march'ta: tek hacimde sekme ışını radyans önbelleğiyle
  karşılanıyor (0 yoğunluk örneği); iki hacim birlikteyken 3.3 örnek/yol, yani
  bir gaz hacminde başlayan sekme ışını ötekini açık march'la, ışık adımlarıyla
  yürüyor (aday, doğrudan ölçülmedi: önbellek hacim başına). Tek hacim + akış
  başına malzeme ağırlığı bunu da kaldırır (önbellek birleşik ortamı kapsar).
- Ölçüm notu: ilk koşuda (§15) buhar 19 kamera örneği/yol, burada 34.6; kamera
  ya da yakınsama farkı. Host ms/kare (render_frames) GPU'nun 5–10×'i (4.7 s);
  görünürlük değişiminin tetiklediği yeniden kurulum mu, ölçüm yöntemi mi, açık.

## 16. Çakışmada radyans önbelleği (A yolu, 2026-10-28; yalnız shader, derlenmedi)

§15b'nin kökü kodda: `volume_closesthit.rchit` önbelleği `!hasCompanion`
kapısıyla çakışmada kapatıyordu; sekme ışınları ikinci gaz hacmini açık march'la
yürüyordu. Boş bölge atlaması ise çakışmada zaten iki alanı birlikte sınıyor
(`volume_companion_empty.glsl`); kamera örnekleri bu yüzden toplama eşitti.
Kullanıcı kararı: önce A (bu yol), tek hacim (B) ancak üç ve daha fazla akış
gerekirse.

- `volumeRadianceCacheMarch(slot, companionSlot, slots, ...)`: her ortam kendi
  ışık önbelleği yuvasından okur; adımda σ = σ₁ + σ₂,
  J = (σ₁J₁ + σ₂J₂)/σ (toplanan ortamlar için kesin). Kutusu dışında companion
  sıfır sayılır (köşe araması kenara kıstırırdı).
- Tek doldurma çağrı yeri: iki ortam iki adımlık bir döngüde aynı
  `volumeRadianceCacheFetch` çağrısından geçer, hacim tablodan okunur. İki
  çağrı yeri ışık döngüsünü bu aşamaya iki kez gömerdi.
- Bilinen yaklaşıklıklar (yalnız sekme ışını): her J yalnız kendi ortamıyla
  gölgelenir (tam march companion'ın ışık march'ını çarpar); companion'ın kendi
  emisyonu J'sinde vardır, tam kamera march'ı onu atlar (overlap sınırı).
  Companion'a özel ikinci bir köşe varyantı bilerek yok: aynı hücreleri paylaşır,
  iki varyant birbirini yeniden doldururdu.
- Taban: `volume_closesthit.spv` 977,104 B (2026-10-27 23:08). Pipeline derleme
  süresi `render.rt_pipeline_status` → `stage_seconds` ile ölçülecek.
- Kontrol: `check_volume_march_lod_contracts.py` (çakışma kapısı ve tek çağrı
  yeri sabitlendi).

### 16b. Canlı (2026-10-28): derleme iyi, kazanç YOK — sahne genel karışım yolunda

- Pipeline derleme 206 sn (taban 314 sn), `volume_closesthit.spv` 977,104 →
  982,912 B (+%0.6). Tek çağrı yeri kuralı tuttu.
- `rt_probe_gas_overlap_cost_ipc.py`: buhar 598, duman 112, ikisi 1157 ms;
  ikincil yoğunluk örneği hâlâ 3.33/yol, `radiance_cache_fallbacks` 0.
- Kök: `matterFogNeedsGeneralMixture` (volume_mixture.glsl) iki üyeden biri
  emissifse (`emission_mode != 0`) ya da eğri uyarlamalı adım isterse çakışmayı
  `shadeFogMixture`'a yollar. Bu sahnede buhar da duman da blackbody → §16'nın
  düzelttiği companion yolu hiç çalışmıyor. §16 doğru ama bu sahneye dokunmuyor.
- Sıradaki iş: `shadeFogMixture` sekme ışınlarında radyans önbelleğini N üyeyle
  okusun (§16'daki σ ağırlıklı birleştirme, üye döngüsüyle; tek çağrı yeri).
  Ayrıca buharın blackbody istememesi (malzeme emisyonu kapalı) onu emissif
  saymaktan çıkarır; bu tek başına sahneyi companion yoluna geri alabilir.


## 17. Genel karışımın sekme önbelleği — 2026-10-28 (kaynak hazır, derlenmedi)

§16b açığı için tek ortak giriş `volume_cached_segment.glsl` eklendi.
Closest-hit, genel karışıma ayrılmadan önce bu girişi çağırır. Kamera,
malzeme programı ve random-walk üyeleri tam integratörde kalır. Emisyon
ve adaptif kamera adımı artık sekme önbelleğini tek başına engellemez.
Genel karışım 16 yuvaya kadar adayları aynı konumda okur; her üyenin kendi
kutusu dışında katkısı sıfırdır. Extinction ve sigma-ağırlıklı kaynak
birleşir. Önbellek eksikse tam karışım çalışır; yarım radyans kullanılmaz.

Doldurma hâlâ TEK march/fetch çağrı yerinden geçer. Dinamik aggregate
array eklenmedi; ilk iki üyenin sabitleri scalar olarak tutulur, ek üyeler
scalar fallback kullanır. Emisyon önbellekteki J'dedir. Önceki cache
bedelleri sürer: izotropik faz, üyenin kendi gölgesi, emitter NEE eksikliği.
Yeni hız veya derleme süresi ölçülmedi; §16b tabanı 206 sn / 982,912 B.

Aynı partide Matter Fog yayıncısı, yeni snapshot'ta bulunmayan akışın canlı
VDB/manager/world kaydını kaldırır. Kalıcı `matter_fog_streams` materyali
saklanır; Fog'a dönünce geri kullanılır. Böylece Fog → SDF/Splat geçişinde
VDB panelde artık kayıt kalmaz; kaldırılmış transient seçim de temizlenir.
UI/Python/IPC aynı yayıncıdan geçer, yeni API düğmesi eklenmedi.

Statik: march LOD, mixture adaptive (10), material (10), optical (22),
empty rejection (7), light cache ve overlap/cache lifecycle (3) geçti.
Derleme ve canlı kabul: NEXT_BUILD_CHECKS'in ilk partisi.


## 18. Canlı §17: cache çalışıyor, toplam maliyet hâlâ yüksek — 2026-10-28

Kullanıcı derledi ve uygulama açıkken dış named-pipe IPC ile ölçüldü.
Pipeline ready, hata yok, compile_seconds = 219.734 sn (206 sn tabana +%6.7).
1680×945 RT viewport. Malzemeler: scatter_coefficient=0, absorption=1.46,
emission_mode=2, adaptive=true, voxel step=.5, density multiplier=11.111.

| yol başına / GPU örneği | buhar | duman | ayrı toplamı | birlikte |
|---|---:|---:|---:|---:|
| GPU ms | 771 | 185 | 956 | 1052 |
| kamera density | 37.518 | 7.057 | 44.575 | 46.093 |
| ikincil density | 0 | 0 | 0 | 0 |
| radiance cache march | .498 | .209 | .707 | .595 |
| cache fallback | 0 | 0 | 0 | 0 |

Önceki 'genel karışım cache'i atlıyor' açığı kapandı. Ancak kullanıcı
şikâyeti çözülmedi. Eski sahne/kamera/pencereyle doğrudan hız farkı
hesaplanamaz; aynı koşudaki birlikte maliyet ayrı toplamından yaklaşık %10
fazla. Buhar tek başına bile 771 ms. Yeni kök kanıtı olmadan cache yolunu
tekrar değiştirmeyin.

Kısa A/B (ayarlar her partinin sonunda finally ile geri yüklendi):
- baseline: 1052 ms, kamera 45.77 örnek/yol;
- emission_mode=0 (ikisi): 1139 ms, 45.67 örnek/yol; belirgin kazanç yok;
- adaptive=false: 802 ms, 14.19 örnek/yol;
- adaptive=false + emission=0: 780 ms, 12.34 örnek/yol;
- gaz görünmez: 31.35 ms;
- yeniden baseline: 1153 ms, 45.77 örnek/yol (süre gürültüsü var).

**Ölçüm sınırı:** `primaryStep()` adaptive=false iken voxel multiplier yerine
legacy world `step_size` kullanır. Bu A/B yalnız adaptif hesap maliyetini
ölçmez, örnek aralığını da değiştirir. Bu nedenle adaptive=false üzerinde
voxel multiplier=4 denemesi etkisizdi (14.19 örnek/yol); bunu '8× daha
seyrek march' diye yorumlamayın. Kayıtlar:
`rt_gas_camera_cost_20261028.json`, `rt_gas_camera_floor_20261028.json`.

### 18.1 Kaynakta bulunan iki gereksiz iş (DERLENMEDİ)

- mfEvaluate, graph yokken density=0 olsa bile sıcaklık ve optik terimleri
  hesaplıyordu. Kanonik sampler sonrası kesin sıfırda erken dönüş eklendi;
  graph'ların boş kaynakta yoğunluk üretmesi korunuyor, sayaç aynı yerde.
- mfAdaptiveSpan ilk iki düz kamera üyesinin ray/box dönüşümlerini her adımda
  tekrar kuruyordu. Başta hazırlanan mfCameraRay/Interval kullanılıyor;
  random-walk yeni uçuş için eski dönüşüm hesabını koruyor.

Bunlar kaliteyi veya örnek kafesini değiştirmez. Henüz kazanç ölçülmedi;
ana yavaşlığın tamamını çözdükleri iddia edilmiyor. Statik optical (22),
mixture adaptive (10), overlap/cache (5), march LOD geçti.


## 19. Materyale göre güvenli Fog kutusu — 2026-10-28 (C++, DERLENMEDİ)

Canlı gözlem (§18.1 kullanıcı tarafından derlendi, anlamlı kazanç yok):
86³ domain'de buhar raw-positive 51,374 hücre (%8.08); AABB [0,0,0] →
[5,4.651163,5], domain hacminin %93.02'si. Duman kutusu daha küçük.
Pipeline compile_seconds=223.908 sn. Bu durum kutunun geniş olduğunu kanıtlar;
toplam GPU maliyetinin tamamının yalnız kutudan geldiğini kanıtlamaz.

`Fluid/FogOpticalBounds.h` ortak çekirdeği, raw > 0 yerine kanonik
remap → monoton yoğunluk eğrisi → cutoff sırasıyla aday köşeleri sınırlar.
Multiplier cutoff'tan SONRA geldiği için cutoff karşılaştırmasına çarpılmaz;
kesin sıfır multiplier/scale katkısız sayılır. CPU/GPU yuvarlama sınırında
küçük güvenlik payı vardır. Trilinear ham değer köşelerin konveks birleşimidir;
monoton dönüşümde aktif örneğin en az bir aktif köşesi olmalıdır. Bu köşelerin
bir voksel komşuluğu korunur, kutu 8'lik bloklara dışarı yuvarlanır. Fizik
alanları, sıcaklık, yoğunluk verisi ve step ayarları değiştirilmez.

Graph, aktif materyal programı, noise, edge falloff, geçersiz/nonmonoton
parametre veya remap edilmiş pozitif background varsa tam grid korunur.
Tüm hücreler kesin katkısızsa canlı kayıt saklanır ama optik görünürlük kapatılır;
yazarın `appearance.visible` tercihi değiştirilmez. Aynı state.version'da bile
yoğunluk politikası/scale değişirse yayıncı anahtarı kutuyu tekrar hesaplatır.
Böylece cutoff azaltma/eğri değiştirme eski dar kutuda takılmaz. Materyal
anahtarı yalnız başarılı yayında güncellenir.

Mevcut UI/Python/IPC materyal servisi aynı yayıncıya gider. Yeni ayar eklenmedi.
Mevcut `fluid.inspect_fog_fields` / `rt.fluid.inspect_fog_fields` aynı çekirdekten
`optical_bounds` döndürür: `first_voxel`, exclusive `end_voxel`, `empty`,
`conservative_fallback`. `render.volume_slots` yayımlanmış gerçek AABB'yi verir.

Statik/nümerik Python kontrolleri geçti. Gerçek C++ çekirdek regresyonu
`scripts/test/fog_optical_bounds_test.cpp` eklendi, derleme yasağı nedeniyle
çalıştırılmadı. Canlı kazanç henüz ölçülmedi. Görünür katkılı köşeler gerçekten
bütün alana dağılmışsa tek AABB yine geniş kalabilir; böyle bir sonuçta keyfi
threshold yükseltmek yerine blok bazlı optik boşluk atlaması değerlendirilir.


### 19.1 Canlı: dar kutu yayımlandı, GPU maliyeti belirgin düştü (2026-10-28)

Kullanıcı C++ build aldı; `rt_probe_fog_bounds_ipc.py` çıktısı hem render hem
viewport backend'de yeni AABB'leri doğruladı. Buhar kutusu domain'in
%11.269'u (önce %93.02), duman %2.656; ikisinde conservative_fallback=false.
Raw-active oranlar sırasıyla %7.852 ve %1.050. Buhar voxel kutusu [0,0,16] →
[80,16,72]; duman [64,0,24] → [86,24,56].

Açık uygulamada dış IPC, `rt_probe_gas_overlap_cost_ipc.py --frames 4`:

| | yalnız buhar | yalnız duman | ayrı toplam | birlikte |
|---|---:|---:|---:|---:|
| GPU ms/RT örneği | 261 | 84 | 345 | 329 |
| kamera density/yol | 6.536 | 2.839 | 9.375 | 10.745 |
| ikincil density/yol | 0 | 0 | 0 | 0 |
| cache fallback/yol | 0 | 0 | 0 | 0 |

Önceki oturum 771/185/1052 ms ile kıyaslandığında birlikte yaklaşık %69
azalma (3.2 kat), kamera density 46.09 → 10.75/yol. Bu, aynı build'de
eski-kutu/yeni-kutu toggle A/B değildir; güncel raw occupancy biraz farklı.
Kesin nedensel hız oranı diye sunulmaz. Güncel tek koşuda birlikte maliyet
ayrı toplamından büyük değil; çakışma kabulü geçti. Görüntü/ince kenarlar,
materyal değişiminde yeniden genişleme ve hareketli plume görsel kabulü
henüz yapılmadı. Görünürlükler probe finally bloğunda tekrar açıldı.
Kayıt: `rt_gas_optical_bounds_live_20261028.json`.


## 20. Zayıf blokta bütçeli optik integrasyon (2026-10-28, shader; DERLENMEDİ)

Amaç: cutoff'u geçen ama optik katkısı küçük bölgelerde tek noktadaki
"düşük yoğunluk" tahmini yerine BÜTÜN leaf/tile majorant'ını kullanmak.
Yeni modül: `volume_mixture_optical_blocks.glsl`. Hacim verisi değiştirilmez,
zayıf bölgenin emilimi/emisyonu sıfırlanmaz; geniş segment tek midpoint'te
Beer–Lambert ile integre edilir. Mevcut kesin boşluk atlaması önce çalışır.

Uygunluk bilinçli dar: live Matter NanoVDB (source_type=6, live sparse
metadata), adaptive açık, scatter_coefficient TAM sıfır, random-walk kapalı,
graph/noise yok, nonlinear emission response yok, flame yok. Güncel ölçülen
sahnenin gazları scatter=0 ve legacy blackbody olduğundan uygundur. Saçılmalı
gazda gelen ışığa güvenli üst sınır olmadığı için mevcut integratör korunur;
"her gaz optimize edildi" iddiası yok.

- Leaf maksimumu HAS_MIN_MAX ile doğrulanır; live inactive tolerance 1e-6
  üst sınırına dahil edilir. Sabit tile tek değeriyle sınırlanır.
- Leaf/tile'ın son voksel slab'ı hariç tutulur; komşu leaf'teki ince yapı
  veya sıcak kaynak görünmeden geçilemez. Her üyenin AABB giriş/çıkışı
  segmenti keser; bütün örtüşen üyelerin sigma/source üst sınırı toplanır.
- Monoton remap/curve/cutoff/fade/multiplier sırası majorant'a uygulanır.
  Sigma üst sınırı = density_max * absorption_coefficient (RGB ağırlığı ≤1).
- Legacy constant emission J üst sınırı = max(color)*strength*density_max.
  Blackbody RGB ve normalize T^4 ≤1 olduğundan üst sınır intensity*density_max;
  HDR ramp stoplarının kanal maksimumu da hesaba katılır. Bu bound sıcaklık
  örneği okumaz; dar sıcak bir pik bu yüzden kaybolamaz. Bilinmeyen response
  veya geçersiz source bound varsa normal adım.
- Ayrı hata bütçeleri: optik derinlik 0.001, yerel yayılan radyans kanal
  başına 0.001. İkisi de düz hacim SEGMENTİ boyunca tüketilir, her blokta
  yenilenmez. h ≤ tau_left/sigma_max ve h ≤ emission_left/source_rate_max.
  Her iki integral de [0, bound*h] aralığında olduğundan midpoint hatası
  aralığın genişliğiyle sınırlıdır. Delta T ≤ delta tau.
- Bu SON piksel parlaklık hatası garantisi değildir: sonraki çok parlak bir
  yüzeyin radyansı küçük transmittance hatasını büyütebilir. Bir yol birden
  fazla hacim segmentine girerse bütçe her segmentte ayrı geçerlidir.
- En fazla 16*baseStep; kanıtlı span baseStep'ten büyük değilse eski yol.
  Bütçe bitince normal adım geri gelir. Optik olarak kalın blok uzatılamaz.

Tek non-scattering Matter hacmi de aynı midpoint integratörüne bağlandı;
blok kanıtı çıkmazsa authored max_steps ve sabit baseStep korunur. Bu tek
hacimde örnek konumlarını eski jitter kafesinden midpoint kafesine taşır;
yoğunluk homojen değilken aynı bit görüntü beklenmez, görsel kabul gerekir.
Karışımda eski hücre-adaptif yol fallback'tir. Kamera/sekme cache ayrımı
korundu, random-walk flights bu bütçeyi kullanmaz. Ağır cache fill hâlâ tek
çağrı yerinde; yeni modülde ışık/temperature döngüsü yok.

UI/Python/IPC mevcut `adaptive_stepping` materyal işlemiyle aynı politikayı
kullanır; yeni ayar/komut veya yeni validation semantiği eklenmedi. Tooltip
kapsamı anlatacak biçimde güncellendi. OptiX koduna dokunulmadı.

Statik/nümerik: `check_volume_optical_block_contracts.py` (4; dar spike,
pozitif emisyon bound'u, üç üye ve tekrarlanan blok bütçesi, source wiring),
mixture adaptive (10), optical (22), march LOD geçti. Shader derlenmedi ve
GPU kazancı ölçülmedi. Özellikle 0.001'lik koruyucu bütçe nedeniyle her
cutoff-üstü bölgenin uzatılması beklenmez; önemli emisyon/emilim korunur.


### 20.1 CANLI FAIL — geri alındı (2026-10-28)

Kullanıcı aynı sahnede §20'yi derledi. Dış IPC sonuçları:

| | yalnız buhar | yalnız duman | ayrı toplam | birlikte |
|---|---:|---:|---:|---:|
| GPU ms/örnek | 545 | 134 | 679 | 766 |
| kamera density/yol | 14.900 | 4.481 | 19.381 | 24.794 |
| majorant query/yol | 4.270 | 4.372 | 8.642 | 11.542 |
| ikincil density / cache fallback | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

§19 referansı birlikte 329 ms / 10.745 kamera density / 4.463 majorant query
idi. §20 süreyi yaklaşık 2.33 katına çıkardı. Optik hata sınırının matematiksel
olarak güvenli olması performans iyileştirmesi anlamına gelmedi. Kodda ek
blok sorguları ve tek hacmi başka integratöre yönlendirme vardı; ikisi birlikte
geri alındı. Hangisinin yüzde kaç maliyet yarattığı ayrı ölçülmedi.

- `volume_mixture_optical_blocks.glsl` silindi; include, bütçe, span çağrıları
  ve `matterFogBoundedSingle` yönlendirmesi çıkarıldı.
- Tek hacim yeniden önceki sabit march yolunda. Genel karışım önceki
  hücre-adaptif yolunda. UI tooltip önceki doğru açıklamaya döndü.
- §19 materyale göre dar kutu ve §17 sekme cache'i KORUNDU. Erken sıfır
  dönüşü ve kamera transform reuse de korundu.
- `check_volume_optical_block_contracts.py` artık bu geri almayı ve başarılı
  yolların korunmasını denetler; geçmeyen özelliği başarılı göstermez.

Geri alma kaynakta hazır, kullanıcı shader derlemesi sonrası canlı doğrulanmalı.
329 ms'ye dönüldüğü henüz ölçülmedi. Bu başarısız denemeyi yeniden eklemeyin.
Bir sonraki mimari değişiklikten önce GPU profiler/sayaçla sıcak iş ayrılmalı;
yeni bir spekülatif optimizasyon veya yeni kalite kaybı eklenmedi.
Kayıt: `rt_gas_optical_blocks_regression_20261028.json`.


### 20.2 Geri alma sonrası canlı: kare 50, genişlemiş buhar kutusu (2026-10-28)

Kullanıcı geri almayı derledi, maliyet hâlâ yüksek dedi. Dış IPC okumasında
kare 50; buharın kanonik kutusu [0,0,16] → [80,64,80], yani domain'in
%51.52'si. Önceki 329 ms snapshot'ında kutu [0,0,16] → [80,16,72], %11.27
idi. Materyal ayarları aynı; fallback=false, genişleme gerçek optik sınır
hesabından geliyor. Yoğunluk alanının uzamsal dağılımı aynı kabul edilemez.

`rt_probe_gas_overlap_cost_ipc.py --frames 4` güncel tek koşu:
- buhar 446, duman 78, toplam 524, birlikte 573 ms;
- kamera density 20.100 / 2.717 / 23.827 (birlikte);
- majorant query 2.343 / 1.247 / 3.605;
- ikincil density ve cache fallback bütün koşullarda sıfır.

§20 kullanıcı çıktısı 766 ms / 11.542 majorant query idi. Son sorgular
geri alma sonrası 3.605'e indi. Fakat bu da kontrollü aynı-frame eski/yeni
shader A/B değildir. §20.1'in '329 -> 766 tamamen kod regresyonu' yorumu
fazla kesindi; eldeki kanıt maliyet artışını gösterir, alan/kare farkının
payını ayırmaz. Başarısız deneme geri alınmış kalır.

Kaynakta emekli semboller yok; source ve x64/Release volume_closesthit.spv
SHA256 aynı, boyut 993,388 B. Pipeline ready/hatasız. IPC yüklü shader hash'i
vermediğinden bunlar runtime bytecode kimliğinin doğrudan kanıtı değildir.
Kullanıcının timeline'ı değiştirilmedi; ölçüm öncesi/sonrası kare 50 idi.
Görünürlükler geri açıldı. Kayıt: rt_gas_rollback_frame50_20261028.json.

Sonuç: 329 ms'ye dönüş genel bir garanti değil; artan optik kutu kamera
march işini artırıyor. Kalan yavaşlık hâlâ açık. Yeni shader denemesi yapılmadı.


## 21. NanoVDB örnekleme belleği: host heap yerine device-local (2026-10-28, C++; DERLENMEDİ)

Kullanıcı aynı sahnede OptiX total süreyi ~110 ms ölçtü. OptiX yolu dondurulmuş,
sadece okundu. ray_color.cuh VDB yolu ile Vulkan optik hesapları birebir aynı
değil: OptiX örnekleme kodunda Vulkan yoğunluk eğrisi ve joint-mixture yolunun
eşdeğeri yok; legacy emisyon tepkisi de farklı. Bu yüzden 110 ms eşit görüntü
kalitesinde kontrollü backend benchmark'ı diye sunulmaz; çok değerli bir ipucu.

Kullanıcı Vulkan'a geçti (kare 50), dış IPC ile aynı oturumda üç ölçüm yapıldı.
Authored 537 ms; no-emission/fixed companion 514 ms, no-emission/fixed phase
mismatch 458 ms. Son ikisinde density/ray/probe sayıları BİREBİR aynıydı
(kamera density=6.851/yol, solid probes=.9655/yol), o yüzden bu ölçüm genel
karışımın daha pahalı olduğuna kanıt üretmedi. Saçılma sıfır olduğundan faz
farkı optik sonucu etkilemez; tüm materyaller/sayaçlar finally ile geri alındı,
get_material eşitliği doğrulandı. Kayıt: rt_gas_integrator_parity_20261028.json.

### 21.1 Kaynakta somut bellek yerleşimi açığı

VulkanBackend_Volumes.cpp yoğunluk ve sıcaklık NanoVDB buffer'larını
CPU_TO_GPU ile açıyordu. translateMemoryLocation bu isteği yalnız
HOST_VISIBLE|HOST_COHERENT olarak çeviriyor. findMemoryType ilk uyanı seçiyor;
device-local tercih etmiyor. Yorumdaki 'host-visible, device-local BAR'
varsayımı kod tarafından garanti edilmiyordu.

Kullanıcının RTX 3060 cihazında read-only Vulkan memory enumeration:
- type 3: flags=6 (host-visible/coherent), heap 1, device-local DEĞİL;
- type 4: flags=14, heap 1, yine host;
- type 5: flags=7, heap 2, host-visible device-local.

Dolayısıyla eski seçim, buffer memoryTypeBits type 3'e izin veriyorsa,
VRAM yerine sistem belleği seçiyor. Bu, her NanoVDB ağaç/değer okumasına PCIe
maliyeti yükleyebilir. Eski allocation'ın gerçek memoryTypeBits/heap'i o
build'de IPC'ye açık değildi; 'aktif eski buffer kesin RAM'de' iddiası yok.
Kaynak ve cihaz sıralaması bu yolu doğrular. Kayıt: rt_gas_memory_types_20261028.json.

### 21.2 Düzeltme ve doğrulama yüzeyi

İki sıcak NanoVDB buffer allocation'ı GPU_ONLY (=DEVICE_LOCAL zorunlu)
yapıldı. uploadBuffer'ın mevcut staging/copy yolu kullanılır; content version,
50% over-allocation, senkronizasyon, lifetime/sweep aynı kalır. Buffer verisi,
shader, örnek sayısı, opacity/emission ve geometrik atlama politikası değişmez.
OptiX koduna dokunulmadı.

Yeni küçük VulkanVolumeResidency.cpp, allocator'ın mevcut tracked heap bilgisini
okur. `render.volume_slots` ve Python `rt.render.volume_slots()` her slotta
`density_device_local` / `temperature_device_local` döndürür: 1 device-local,
0 host heap, -1 unknown/not managed NanoVDB. Gerçek yayımlanmış GPU adresi
buffer haritalarıyla eşleştirilir; VDB id churn'ü yanlış yerleşim raporlamaz.
Yeni kullanıcı ayarı veya authored işlem yok; mevcut tanı API'si genişledi.

Statik: residency (4), Fog bounds (4), material (10), march LOD geçti.
C++ build ve gerçek bellek/süre kabulü açık. İlk upload ve sim/playback'te
staging submit/wait maliyeti artabilir; bunu steady GPU dispatch süresinden
ayrı ölçmek gerekir. Bu parti yeni boş-kutu algoritması eklemez; önce
mevcut örneklerin bellek erişiminin maliyetini düzeltir. Yeni shader build
gerekmiyor. Ajan build çalıştırmadı ve uygulama başlatmadı.


## 22. Genel karışım Max Steps ayarını uygulamıyordu (2026-10-28; shader, DERLENMEDİ)

Kullanıcı 256 → 16 değişiminin hız/kaliteye etkisiz olduğunu bildirdi.
Kaynakta doğrulandı: shadeFogMixture, max_steps alanını hiç okumadan
baseStep'i 2048 ve adaptif döngüyü 8192 limitleriyle kuruyordu. UI/Python/IPC
ayarı taşıyordu; normal tek-hacim yolu zaten authored cap'i kullanıyordu.
Ayrıca son VRAM çıktısı iki backend/dört grid için device_local=1;
birlikte son GPU 280 ms (buhar185/duman84), kamera density9.482/yol,
ikincil density/fallback0. Bu kullanıcı ölçümü §21 canlı tabanıdır.

Yeni düz karışım yürüyüşünde ray interval'ine gerçekten değen üyelerin en
düşük max_steps değeri ortak değerlendirme bütçesidir. Her joint sample her
üyeyi değerlendirdiği için bu seçim her üyenin iş tavanına uyar. Off-ray
üyeler bütçeyi düşürmez. Gölge ve random-walk event budget'ları ayrıdır.

Base kafes bu bütçeye göre kurulur. Adaptif örnek kalan aralığı kalan sample
sayısına bölen coverageFloor'dan kısa olamaz. Son örnek tüm kuyruğu kapsar;
bütçe azaldığında gazın sonu kesilmez, çözünürlük düşer. Kesin boş atlamalar
optik örnek bütçesi tüketmez. Adaptif hedef bütçe yüzünden kabalaştırıldığında
mevcut step_budget_exhausted sayacı bunu raporlar. Yerel voxel hata sınırı
bu durumda garanti edilmez: yazarın açık work cap'i daha önceliklidir.

UI/script/IPC aynı mevcut max_steps ayarını kullanır; validation16..1024
aynı kaldı, yeni API/ayar yok. Sekme radyans cache'i ayrı cell-walk politikası
kullanır; bu cap düz kamera yoğunluk integrasyonunu sınırlar. Tek hacimde
zaten 16'dan az gereken ışın varsa 256→16 fark yaratmayabilir. Sayaçtaki
density_samples bir joint sample başına birden çok üye sayabilir; ray başına
ham okuma sayısını ortak sample limitiyle karıştırmayın.

Statik/nümerik testler: mixture_step_budget(4), adaptive(10), march LOD,
rollback(2) geçti. Yeni A/B: rt_probe_gas_max_steps_ipc.py; 256/16/256_repeat,
sonunda orijinal cap'ler ve instrumentation geri yüklenir. GPU/kalite kabulü
kullanıcı shader build'inden sonra; ajan derlemedi.


## 23. Oturumlar arasında farklı gaz snapshot'ları / ölçüm kimliği (2026-10-28)

Kullanıcı §22 sonucu: 256=625.61 ms,16=488.73 ms,256-repeat=603.80 ms;
kamera density25.6192185689 →8.8473335853 →25.6192185689. İki256 koşusunun
ray/sample/budget/fallback sayaçları BİREBİR aynı. Bu koşu kendi içinde
yapısal olarak stabil, ~%3.5 GPU süre farkı var; 16cap işlevsel. Oturumlar
arasında ise kutuların/alanların aynı olduğuna dair koruma eksikti.

İlk read-only mevcut durum, kare50: buhar active51,105, density_sum118.19437,
box[0,0,16]→[80,64,72] (%45.08). İkinci okumada server stopping döndü;
uygulama yeniden başlatılmadı. Kullanıcı yeniden yükledi ve daha hızlı dedi.
Yeni read-only iki snapshot kendi arasında stabil:
- kare50, buhar active63,072, density_sum126.15447;
- buhar box[0,0,8]→[86,8,64] (%6.06);
- duman active16,789, density_sum358.4213;
- kamera pozisyonu aynı, target önce[.2827,-.4387,.3550], sonra
  [1.0305,-.3562,2.0471]; focus_distance da farklı (DoF kapalı).

Demek ki aynı dosya/kare numarası aynı render girdisini garanti etmemiş:
hem gaz dağılımı hem kamera değişmiş. Yeniden simülasyon/cache/backend restore
hangisinin dağılım farkını yarattığı henüz kanıtlanmadı; bunu 'boş hücre eleme
rastgele çalışıyor' veya 'yalnız kamera farkı' diye kesinleştirmeyin. Daha çok
raw-active hücre varken kutu çok küçülebilir: cutoff üstü hücrelerin KONUMU
ve dağılımı belirleyicidir. Yeni kayıt: rt_gas_reloaded_snapshot_20261028.json.

### 23.1 Ölçüm koruması (Python hazır; tam voxel hash'i için C++ build açık)

`gas_probe_snapshot.py`: kare, ray-kamera ayarları, gaz alanı özeti/hash'i,
materyal (A/B değişkeni max_steps hariç), gerçek render-backend AABB'leri ve
residency kimliği alınır. VDB id/upload serial/state-version churn'u kimlik
olarak kullanılmaz; state_versions ayrıca kaydedilir. Fark varsa MaxSteps
probu INVALID diyerek abort eder ve finally ayarları geri yükler. Her ölçüm
öncesi/sonrası kimlik, GPU çözünürlüğü denetlenir. Raporlar benzersiz dosya
adlarına kaydedilir; eski oturum dosyasının üzerine yazılmaz.

Mevcut eski executable'da hash yoksa açıkça summary_only raporlanır. Bu mod
farkı yakalayabilir ama eşit toplamın eşit voxel düzeni olduğunu kanıtlamaz.
Mevcut `fluid.inspect_fog_fields` / Python eşdeğerine density_hash ve
temperature_hash eklendi (C++ DERLENMEDİ): tüm float voxel bitlerinden,
source-empty sıcaklıklar dahil, sabit byte sıralı FNV-1a64. Yalnız on-demand
inspection'da hesaplanır; per-frame renderer'a ek maliyet eklemez. Yeni
Fluid/FogFieldDigest.h projeye kayıtlı. Fizik/render algoritması değişmedi.

`check_gas_probe_snapshot.py` 4 test: cap/id değişimi kabul, eşit toplamda
farklı field hash ret, kamera değişimi ret, eski build summary etiketleme.
Yeni shader denemesi eklenmedi. Sonraki performans oranları yalnız aynı
snapshot kimliği/çözünürlükte raporlanmalı; sim/cache restore determinizmi
ayrı araştırma olarak açık.


## 24. Güncel ayar doğrulaması ve ölçüm yolu düzeltmesi (2026-10-28)

Kullanıcı açık sahnede yüksek maliyet/etkisiz quality bildirdi. Kare49,
1680×945, Vulkan ready, iki NanoVDB grid çifti device_local=1. Aynı full voxel
hash'leri/kamera/materyal (max_steps hariç)/AABB ile kontrollü A/B yapıldı;
scene_signature=74cc36820de778bae7da58a9d8f1fc7712778f7e471067ea95225f634c3e35fe.

İlk eski prob yöntemi:256=733,16=603,256repeat=746ms, density32.642→10.191→32.642.
Emisyon0/16=573ms, density0=24ms. Sayaç off/on/off=732/755/755ms; atomikler
bu farkın tek kökü değil. Low power-state anlık gözlendi (210/810MHz), ama
başka örneklerde1980/7301MHz'e yükseldi; power-state tek kök diye sunulmaz.

**Ölçüm kusuru:** viewport.render_frames, ana iş parçacığında
renderer.render_progressive_pass(nullptr, nullptr, scene,1,0) çağırıyor;
normal Vulkan display döngüsü beklenmiyor. Timestamp'ler GPU işi olsa da bu
çağrılar normal sürekli RT akışına ek iş/idle aralıkları karıştırabiliyor.
Eski probun baştan beri saf interaktif RT benchmark'ı gibi sunulması yanlıştı.
Doğal GPU/counters-off8 sample ortalaması567ms ve çok daha dar aralık verdi.

Yeni gas_probe_sampling.py doğrudan timestamp capture sayısını bekler,
8warmup sonrası örnek alır; hiçbir viewport.render_frames çağrısı yok.
Süre pass'inde volume counters kapalı, sayaçlar AYRI doğal pass'te toplanır.
Frame/timeout kontrolleri ve finally cleanup var. İki ana gas probu ortak
helper'a geçti, overlap probu gerçek eski visibility/instrumentation durumunu
restore eder. Yeni SamplingContracts2 ve SnapshotContracts4 test geçti.

Düzeltilen probun AYNI voxel/kamera kimliğiyle canlı sonucu:
-256:610.668ms, camera density32.8763/yol;
-16:571.648ms, camera density10.3093/yol;
-256repeat:605.929ms, camera density32.8763/yol.
İkincil density/fallback0. Orijinal materyaller geri yüklendi (RESTORED True).
Kayıt rt_gas_max_steps_1793185733859251700.json.

Sonuç: MaxSteps GPU'da çalışıyor, okuma sayısını~%69 düşürüyor; doğal akışta
süre kazancı yalnız~%6. Kalan sabit/per-entry GPU maliyeti açık. Shadow/stride
ayarları scatter_coefficient=0 sahnede ana in-scatter işini değiştirmez.
Sadece MaxSteps düşürerek kalan500+ms'yi çözme varsayımı geçerli değil.
Yeni shader veya C++ optimizasyonu yapılmadı; bu tur sadece ölçüm düzeltmesi.


## 25. Import VDB / OptiX yön bağımlı kaybolma: adım uzayı ve coverage audit'i

Kullanıcı: density curve kapatınca düzelmedi; Max Steps yüksekken de aynı;
üst/yan bakışta hacim yok gibi, alttan dolu. Negatif yön AABB kodu okundu:
min/max slab hesabı işaretleri koruyor; basit negatif-yön hatası bulunmadı.

Somut ayrı defect adayı: VolumetricRenderer renderVoxelSize'ı Vulkan'da
worldVoxelSize, OptiX'te native getVoxelSize seçiyor; gv.step_size bundan
üretiliyor. OptiX raymarch_vdb_volume ise requested_step'i world chord ile
bölüp required_steps/march_step_limit hesaplıyor. SONRA quality clamp,
base_step'i world_voxel_size veya world_extent/16'ya küçültebiliyor;
march_step_limit buna göre yeniden hesaplanmıyor. Kaynakta bu sıra doğrulandı.

Sayısal örnek (sahnenin ölçülmüş ölçeği değil): world chord8, native voxel100,
scale.001, multiplier.207. Requested20.7, required_steps3. Son clamp step.5;
3 adım yalnız1.5/8 kapsıyor. Cap16/256/1024 aynı sonucu veriyor. Yoğunluk
kolonun alt1.5 birimindeyse alttan görünür, üstten prefix boş kalır. Kullanıcı
gözlemiyle uyumlu bir mekanizma; mevcut VDB'nin actual scale/chord değerleri
henüz doğrudan IPC'de okunamadığından bu sahnedeki tek kök diye sunulmaz.

`scripts/test/check_optix_vdb_step_space_audit.py`3 read-only/nümerik audit
geçti; test implementasyonun hâlâ bu sırada olduğunu belgeliyor, düzeltme testi
değil. OptiX/CUDA/renderer koduna değişiklik yapılmadı, build çalıştırılmadı.
Native/world step birliği ve SON adım uzunluğuna göre kalan aralığın tam
kapsanması birlikte ele alınmalı; sadece authored cap artırmak yeterli değil.
