# Vulkan RT: ikincil sekmelerde gaz march'ı

> **Durum:** AKTİF — 2026-10-25; march LOD + sekme ışınları için radyans önbelleği varsayılan açık (düzlemli nükleer sahne −%53). Geçirgenliğe göre büyüyen adım SÖKÜLDÜ (§14); derleme bekliyor.

## Devir notu

- **Nerede kaldı:** büyüyen adım söküldü (kod; canlıda IPC ile kapatıldı),
  radyans önbelleği varsayılan açık. Derleme bekliyor.
- **Sıradaki (açık kalemler):**
  - zemindeki +%5'in kaynağı (izotropik faz ya da iki dokunuşlu ambient
    gölgesi), ölçülmedi;
  - kalan ~20 M kamera örneği. Doğru bir değişken adım, segmentin üst sınır
    yoğunluğunu (majorant) okumalı.
- **Dokunulacak dosyalar:**
  - `volume_closesthit.rchit` (adım kurulumu ~4109–4119, ambient gölgesi ~4697);
  - `volume_instrumentation.glsl` ve `VulkanVolumeInstrumentation.h`;
  - RtApi + IPC + Python + panel (dört dokunuş);
  - descriptor overlay.
- **Bağlam:** önceki adım (iç içe katı izi, "Seçenek A") gerileme yaptı ve geri
  alındı. Bkz. RT_GAS_OPTICAL_PERFORMANCE.md §"Seçenek A ölçüldü". Ders: hacim
  closest-hit'ine iz ya da büyük yük ekleme. Bu tasarım yalnız döngü sınırlarını
  ve bir dalı değiştirir.

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
