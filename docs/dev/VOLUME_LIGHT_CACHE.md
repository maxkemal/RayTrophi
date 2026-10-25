# Hacim ışık geçirgenlik önbelleği (volume light cache)

> **Durum:** REFERANS — 2026-10-25; V1 canlı kabul edildi (~%14 hız, görüntü kullanıcı gözüyle aynı), `enabled` A/B anahtarı söküldü (bu son adım derlenmedi). Ölçümler §10–§11.

## Devir notu (2026-10-25)

Yazıldı, derlenmedi. Dokunulan yerler:
- Shader: `volume_light_cache.glsl` (okuma, ortak), `volume_light_cache_fill.glsl`
  (doldurma, yalnız hacim shader'ı), `volume_closesthit.rchit` (sahne ışığı + güneş
  in-scatter), `volume_surface_shadow.glsl` + `closesthit.rchit` + `bsdf_scatter.glsl`
  (yüzey gölgesine `lightIndex`), `volume_instrumentation.glsl` (+4 sayaç).
  `raygen/hair_closesthit/rt_sky`: yalnız dünya struct'ında iki ad.
- C++: `VulkanBackend_Volumes.cpp` (`publishVolumeLightCache`, ayar),
  `VulkanBackend.cpp` (dünya adresi, teardown, TLAS sonrası başlık),
  `vulkan_world_data.h`, `VulkanVolumeInstrumentation.h` (208→224 B).
- Yüzey: `rtapi::volumeLightCache/setVolumeLightCache`, IPC
  `render.volume_light_cache` (Read) / `render.set_volume_light_cache` (Render),
  Python aynı adlar, panel: Performance → "Volume Light Cache" + "Light Cache Cells".
- Testler: `scripts/test/check_volume_light_cache_contracts.py` (kaynak, OK),
  `scripts/test/rt_volume_light_cache_ab_ipc.py` (canlı A/B).

Kabul edildi; `enabled` anahtarı, A/B betiği ve panel onay kutusu söküldü
(§12). Önbellek her zaman açık, ayar yalnız `max_cells`. Kapsam dışı bırakılanlar (§1) ayrı iş: ambient, emitter NEE,
alan ışıkları, companion ve random walk NEE.

## 1. Sorun ve hedef

Vulkan RT'de bir hacim örneğinin ışığa olan geçirgenliği `T(x → ışık)` kameradan,
pikselden ve bounce'tan bağımsızdır. Bugün buna rağmen her ışın için yeniden
march ediliyor:

- **In-scatter** (`volume_closesthit.rchit`, march döngüsü): her (stride'lı) yoğunluk
  örneği için seçilen 2 sahne ışığına + güneşe `lightMarchAcc` (≤12–16 örnek).
- **Yüzey gölgesi** (`volume_surface_shadow.glsl`, `computeVolumeShadowTransmittance`):
  gazın içinden/arkasından gelen her yüzey NEE gölge ışını her hacmi ≤16 örnekle
  march ediyor. "Çevreden seken ışığın domaine tekrar girmesi" bu kalemdir.

2026-10-25 ölçümünde (kare 28, çoklu emitter) bu iki kalemin payı onlarca ms; 22
Ekim sahnesinde scatter yolunun ~%60'ı. Hedef: ikisini de **önbellekten tek
trilinear okumaya** indirmek, görüntüyü değiştirmeden (aynı ölçek ve doğruluk sınıfında).

Kapsam DIŞI (bilerek): ambient yönü (`ambientLightDir` ışın yönüne bağlı,
sabit bir ışık değil), emitter NEE (yayıcı ışın başına rastgele seçiliyor), alan
ışıkları (ışık üzerindeki örnek noktası her örnekte değişir), companion
(örtüşen ikinci ortam) gölgesi. Bunlar mevcut march'ta kalır.

## 2. Karar: ne saklanır

Hücre başına, kanal başına **yoğunluk integrali** `I = ∫ρ ds` (dünya birimi),
`σ_t` ile ÇARPILMADAN. Gerekçe: iki tüketici farklı `σ_t` ve farklı
shadow_strength formülü kullanıyor (march: renk ağırlıklı soğurma ve `0.92·strength`;
yüzey: `σ_s+σ_a` ve `strength`). İntegral ikisine de hizmet eder ve
scatter/absorption değişince geçersizleşmesi gerekmez.

★ Önceki "source shadow cache" denemesi haritalanmış extinction'ı interpole
ettiği için nonlinear cutoff'u bozmuştu. Burada interpolasyon yine **integral**
üzerinde. İntegrali üreten örnekleyici ise curve/remap/cutoff/multiplier/graph
noise'u zaten uygulayan `sampleDensityAcc`'in kendisi. Yani nonlinearlik integralin
İÇİNDE kalır, interpolasyon yalnız düzgün bir alanı karıştırır.

## 3. Karar: tembel (lazy) doldurma, compute pass YOK

Önbelleği ayrı bir compute pass'le doldurmak, `sampleDensityAcc`'i (NanoVDB index
yolu, dense gaz, Volume Graph gürültüsü, cutoff tüyü) compute'a kopyalamak
demektir. İki örnekleyici ayrışırsa gölge yoğunluğu ile görünen yoğunluk farklı
olur ve bu hatasız, ipucusuz bir sapmadır. Bu yüzden **hücreyi, ona ilk ihtiyaç
duyan RT çağrısı kendisi doldurur**; aynı shader, aynı örnekleyici.

- Hücre değeri **deterministiktir**: hücre merkezinden ışığa doğru, sabit jitter (0.5)
  ve hücre boyutuna bağlı sabit adım sayısıyla. Aynı hücreyi iki çağrı doldurursa
  aynı biti yazar.
- **Sahiplenme (claim):** geçersiz hücreye denk gelen çağrı 32-bit `atomicCompSwap`
  ile etiketi "doluyor" durumuna çeker. Kazanan doldurur, diğerleri **bugünkü
  march'a** düşer. İlk kare böylece bugünkü maliyet + hücre doldurma kadar sürer,
  katlanmaz. Bütçe **closest-hit çağrısı başına 4 hücre** (bakma başına değil:
  bakma başına bütçe soğuk önbellekte her adımı × her ışığı ödetirdi). Hücre
  doldurma 4..32 örnek, yani çağrı başına en kötü ek ≤128 örnek.
- **Yayın:** tek 64-bit yazma `(etiket | değer<<32)`. Okuyucu tek 64-bit okuma yapar
  ve etiketi kontrol eder. Yarım yazılmış bir değeri geçerli sanmak mümkün değildir.
  Bayat okuma en kötü ihtimalle "kaçırma" olur, yanlış değer olmaz.
- Yüzey shader'ı (`closesthit.rchit`) **yalnız okur**, doldurmaz. Onun örnekleyicisi
  (`ch_volDensity`) farklıdır, o yüzden doldurma hakkı yalnız hacim shader'ında.
  Hücre geçersizse yüzey bugünkü march'ını yapar.

## 4. Karar: geçersizleşme etiketle, kirli bayrakla değil

Etiket = hash(hacim anahtarı, kanal, ışık geometrisi), bit0 = 0, bit1 = 1;
"doluyor" = etiket | 1.

- **Hacim anahtarı** (host): yayınlanan `VkVolumeInstance` baytları + içerik
  sürümleri (NanoVDB `content_version`, dense `dense_version`,
  `dense_mirror_version`) + hücre boyutları + bölge ofseti. 0 = bu slotta önbellek
  kapalı. Sürüm bilinmeyen yayın yollarında (TLAS sırası yeniden dizme, eski gaz
  yolu) anahtar artan bir epoch'la zorla değiştirilir: güvenli taraf yeniden doldurmaktır.
- **Işık geometrisi** (shader, okuma anında): güneş yönü; yönlü ışıkta yön; nokta
  ve spot ışıkta konum. Işık taşınınca host'ta hiçbir yol kirlenmek zorunda değil;
  etiket kendiliğinden tutmaz.

Bellek sıfırla başlar. 0 hiçbir geçerli etiketle eşleşmez, ve bit1=1 kuralı bunu
garanti eder.

## 5. Yerleşim

Tek GPU_ONLY buffer (yalnız TLAS kuran, yani RT tüketicisi olan adapter'da):

```
[0..15]    uint slotCount, uint channelCount, uint64 reserved
[16..271]  16 × uvec4 { entryOffset, dims(10:10:10), volumeKey, 0 }
[512..]    entry[ (entryOffset + cell*K + kanal) ] : uint64 (etiket | I<<32)
```

- K = 6 kanal: 0 = Nishita güneşi (yalnız hacim shader'ı kullanır), 1..5 = sahne
  ışığı indeksi 0..4 (nokta/spot/yönlü). İndeks ≥5 veya alan ışığı → march.
- Hücre ızgarası hacmin yerel kutusu `[aabb_min, aabb_max]` üzerinde; en uzun eksen
  `min(max_cells, ceil(uzunluk/(2·voxel)))`, diğerleri orantılı, her eksen ≥ 4.
  Varsayılan `max_cells = 64` → 64³ × 6 × 8 B = 12.6 MB/hacim.
- Adres, dünya tablosundaki iki boş kelimede: `_aerialPad1/_aerialPad2` →
  `volumeLightCacheLo/Hi` (uint). `VkVolumeInstance` (672 B) **büyümüyor**. Struct
  büyütmek 4 RT shader'a ve 672 bayt sabitiyle okuyan 3 raster pass'ine dokunmak
  demekti, ve bu deponun en sessiz arıza sınıfıdır.
- Başlık her volume yayınında `uploadBuffer` (sıralı transfer) ile yeniden yazılır.
  Buffer yalnız büyür; büyüyünce waitIdle → yeni buffer → `vkCmdFillBuffer 0` →
  dünya tablosu yeniden yüklenir (adres değişti).

## 6. Tüketiciler

1. **In-scatter, sahne ışığı** (`ls` döngüsü): `!hasCompanion`, ışık alan ışığı değil,
   `li < 5` ise önbellekten okunur. Okuma ucuz olduğu için **stride beklenmez, her
   adımda** okunur; bu kalite KAZANCI (stride önbellekteki gölgeyi bayatlatmıyordu,
   march'ı seyreltiyordu). Kaçırmada eski strided `lightMarchAcc` yolu aynen çalışır.
2. **In-scatter, güneş**: kanal 0, aynı kural.
3. **Yüzey gölgesi**: `computeVolumeShadowTransmittance(origin, dir, maxDist, lightIndex)`.
   Imza ışık kimliğini taşır; yön eşleştirerek ışık tahmin etmek yasak. Hacmin
   girdiği noktada (içerideyse başlangıç noktasında) okuma yapılır. Bu yaklaşım
   değil: hücre ışığa aynı doğru üzerinde ölçülmüştür. Kaçırmada mevcut march.

Dönüşüm: march `T = 1 − 0.92·strength·(1 − e^{−σ_t,w·I})`, yüzey
`T = 1 − strength·(1 − e^{−(σ_s+σ_a)·I})`. Bugünküyle aynı formüller, yalnız
`I`'nın kaynağı değişiyor.

## 7. Yetenek yüzeyi (kural 1)

- API: `rtapi::volumeLightCache()` / `setVolumeLightCache(enabled, max_cells)`.
- IPC: `render.volume_light_cache` (durum: ayar, slot başına boyut/bayt, toplam bayt,
  buffer var mı), `render.set_volume_light_cache {enabled?, max_cells?}`.
- Python: `rt.render.volume_light_cache()`, `rt.render.set_volume_light_cache(...)`.
- Panel: volume sayaç panelinde aç/kapa + max hücre.
- Sayaçlar (`render.volume_stats`): `light_cache_hits`, `light_cache_misses`,
  `light_cache_fills`, `light_cache_fill_samples`. ★ `hits` tek başına bir şey
  kanıtlamaz: `misses` ile birlikte okunur. İlk karede misses > 0 normal, durağan
  karede ~0'a inmeli. Inmiyorsa etiket her kare değişiyor demektir (anahtara
  her kare değişen bir alan girmiş).
- Ayar projeye KAYDEDİLMEZ. Bu bir A/B anahtarıdır: kabul edilince `enabled`
  sökülür, yalnız `max_cells` kalır (kural 5).

## 8. Bilinen bedeller ve tuzaklar

- **Oynatımda** her sim karesi anahtarı değiştirir, önbellek her kare yeniden dolar.
  Maliyet ≈ bugünkü + doldurma. Kazanç durağan karede ve accumulation boyunca.
- **Gölge yumuşaması**: hücre = 2 voxel'den kaba gölge kenarı. Yoğun çekirdeğin
  hemen yanındaki ince dumanda fark görülebilir. A/B ile bakılmalı.
- **Sessizce makul görünen arıza**: anahtar host'ta bayat kalırsa (yoğunluk değişti
  ama yayın yapılmadı) gölge bir önceki kareye ait olur. Hata yok, görüntü
  inandırıcı. Kontrol: oynatımda `light_cache_fills` her karede > 0 olmalı.
- Doldurma örnek sayısı hücre boyutuna bağlı (4..32). Bugünkü march ≤16. Yani
  önbellek değeri ortalamada bugünkünden daha İYİ örneklenmiş bir gölgedir, birebir
  aynı değildir. "Piksel eşitliği" kabul ölçütü değil, görsel eşitlik ölçüttür.

## 9. Kabul

`NEXT_BUILD_CHECKS.md`'de sıralı. Özet: (1) aç/kapa A/B aynı kare/kamera ms;
(2) durağan karede misses → ~0; (3) gaz gölgesinin zemindeki şekli aç/kapa
arasında görsel aynı; (4) ışık taşınınca gölge güncellenir (etiket); (5) oynatımda
fills her kare > 0.

## 10. Canlı ölçüm 1 — 2026-10-25 (ilk build): önbellek HİÇ dolmadı

Kare 28, Rendered, pipeline ready (soğuk derleme 176 s). `render.volume_light_cache`:
buffer 1.88 MB, slot 0 "Snow [Solid]" ve slot 1 "Vapor / Gas" 25³ (voxel
sınırı), slot 2 SurfaceSDF önbelleksiz (doğru). A/B (`rt_volume_light_cache_ab_ipc.py`):

| Durum | ms/örnek | hits | misses | fills |
|---|---:|---:|---:|---:|
| off | 275 | 0 | 0 | 0 |
| on | 279 | 0 | 248 286 | 0 |
| off tekrar | 325 | 0 | 0 | 0 |
| on tekrar | 328 | 0 | 248 286 | 0 |

Kök: iki slot AYNI kutuyu ([0,5]³) paylaşıyor, yani her gaz ışınında companion
var. V1 in-scatter yolu `!hasCompanion` şartıyla önbelleği hiç çağırmıyordu.
Kimse doldurmadığı için yüzey gölgesi de (yalnız okur) hep kaçırdı. Görülen
misses'ın tamamı yüzey okuması. Düzeltme: companion KENDİ slotundan okunur
(`companionLightMarch` da kendi gridini march ediyordu), iki geçirgenlik
çarpılır. Yalnız `volume_closesthit` shader'ı yeniden derlenmeli.

★ Ölçüm aracı notu: davranışı aynı olan iki pencerenin sayaçları 15 haneye kadar
eşit çıkıyor. Bu bayat okuma DEĞİL. Aynı kare + accumulation sıfırlaması aynı
örnek indekslerini, yani aynı ışınları üretiyor (deterministik RNG). Sayaç
sıfırlama ayrıca denendi, çalışıyor. 22 Ekim stride16 satırındaki eşitlik de
büyük ihtimalle bu (stride'ın sayaca etkisi yoksa).

## 11. Canlı ölçüm 2 — 2026-10-25 (companion düzeltmesi derlendi): ÇALIŞIYOR

Aynı kare 28, aynı sahne, pipeline ready (soğuk derleme 189 s).

| Durum | ms/örnek | hits | misses | fills | shadow/vr | density/vr |
|---|---:|---:|---:|---:|---:|---:|
| off | 287 | 0 | 0 | 0 | 19.9 | 18.5 |
| on | 256 | 56.9 M | 9 | 47 | 8.3 | 18.7 |
| off tekrar | 294 | 0 | 0 | 0 | 20.0 | 18.5 |
| on tekrar | 247 | 56.9 M | 9 | 47 | 8.3 | 18.7 |

- Kazanç ~**35–47 ms/örnek (~%14)**. Sabahki tahmin aralığında (20–45 ms).
- Kaçırma oranı 1.6·10⁻⁷: durağan karede etiket kararlı (madde 4 geçti).
- Sayaç penceresinde fills 47 = önbellek pencere öncesinde zaten dolmuştu.
- Kalan 8.3 shadow/vr kapsam dışı kalemler: ambient march (stride 8), emitter,
  alan ışığı ve ilk örneklerdeki doldurma. Sıradaki aday ambient.

Görsel A/B (96 spp, aynı kare, sayısal; görüntüler scratchpad'de üretilip silindi):
- off ile off tekrar birebir aynı (0 fark): deterministik, gürültü tabanı sıfır.
- off→on: ortalama parlaklık 0.5400 → 0.5365 (**−%0.66**). Değişen piksellerin
  %99.7'si koyulaşıyor. |fark| > 0.05 olan pikseller %0.23, p99 |fark| 0.039.
- Fark bulut şeklini izliyor: kutu ya da hücre deseni yok.
- **Açık soru (göz ile bakılmalı):** fark haritasında karın üst kenarı hizasında
  düz, ince bir çizgi var. Ya yüzey gölgesinin girdiği noktada sınır hücresi
  kenetlemesi, ya da gerçek bir kenar (katı yüzeyin gölgesi).
- Koyulaşmanın yönü beklenen tarafta: önbellek her adımda okuyor (stride yok) ve
  hücre başına 4..32 örnek alıyor. March ise ≤12–16 adaptif örnekle orta nokta
  yoğunluğuna göre seyreltiyor ve ince yoğun bölgeyi kaçırıp gölgeyi azaltabiliyor.
  Hangisinin "doğru" olduğu referans olmadan kanıtlanmış değil: march'ın örnek
  tavanı 12/16'da sabit, önbellek de 25³ hücreyle voxel sınırında.

## 12. Kabul ve anahtarın sökülmesi — 2026-10-25

Kullanıcı panelde aç/kapa yaptı: görüntü gözle aynı. Kabul edildi. Kural 5
gereği A/B anahtarı söküldü: `VolumeLightCacheConfig` → `volumeLightCacheMaxCells()`,
`rtapi::setVolumeLightCache(int max_cells)`, IPC `render.set_volume_light_cache`
artık yalnız `max_cells` (zorunlu) alır, Python aynı. Panelde onay kutusu
kalktı, "Light Cache Cells" kaldı. `rt_volume_light_cache_ab_ipc.py` silindi
(kapatılacak bir şey yok). Contract check, anahtarın geri gelmesini FAIL eder.

★ Bir karşılaştırma gerekirse doğru araç kapatmak değil SAYAÇLAR: kaçırma zaten
eski march'ı çalıştırıyor, yani bozuk bir önbellek `light_cache_misses` olarak
görünür, farklı bir görüntü yolu olarak değil.

Açık kalan: §11'deki kar kenarı çizgisi (kullanıcı gözüyle sorun görmedi);
kapsam dışı kalemler (§1) ayrı iş, sıradaki aday ambient march.

## 13. Kanal sayısı 6 → 10: radyans önbelleği (2026-10-25)

Aynı ızgara, sekme ışınları için dört kanal daha taşıyor: yoğunluk ve kaynak
radyansı RGB (RT_GAS_SECONDARY_MARCH.md §13, `volume_radiance_cache.glsl`).
Işık eşlemesi artık `VLC_LIGHT_CHANNELS = 6`'ya bağlı. Slot başına bellek
~1.67 katına çıkıyor.
