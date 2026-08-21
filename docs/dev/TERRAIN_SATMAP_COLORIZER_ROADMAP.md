# Terrain SatMap Kolorizer Yol Haritası

> **Durum:** AKTİF — Ek A ölçüm katmanı CANLI DOĞRULANDI; sıradaki iş Faz 0. 2026-08-21
> (rev. 2026-08-21: çözünürlük ayrıştırması Faz 0 olarak eklendi, 8k reddi geri
> alındı, arazi tipi preset kütüphanesi Faz 2b, 4k üretim maliyeti Ek A).
>
> **İlerleme (2026-08-21, parti 1-2, ikisi de derlendi ve canlı doğrulandı):**
> Ek A'nın ölçüm aleti (`rt.perf`), tahsisat yolu ve tembel hızlandırma yapısı
> ölçümü tamam. **Ölçümün cevabı: 4k maliyetinin %5'i mesh doldurmada, %90'ı
> Solid raster geometrisi + Embree BVH kurulumunda.** Faz 0'ın kazancı
> ölçüldü: **~18×**. Faz 0 artık sıradaki iş; Faz 1-5 tasarım.

Gaea'nın SatMap'ine denk bir **makro renk** katmanı: terrain analiz alanlarını bir
LUT'tan geçirip per-texel RGB üretmek ve onu mevcut 4 katmanlı splat blend'inin
üstüne **modülasyon olarak** uygulamak.

---

## 0. Neden bu iş küçük

SatMap bir doku sistemi değil, bir **kolorizer**. Gaea'da detay satmap'ten gelmez;
satmap düşük frekanslı rengi taşır, detay render tarafında tiled dokudan gelir.
Yani Gaea'nın çıktısı **bizim shader'ımızın girdisinin bir eksiğidir** — bizde
tiled PBR katman zaten var, renk yok.

Elimizde hazır olanlar:

| Gerekli | Bizdeki karşılığı |
|---|---|
| Analiz alanları | `TerrainAnalysis` (Slope/Concavity/Convexity/Valley/Wetness), `TerrainObject::flowMap`, `erosionMapRGBA` (wear/deposit/flow/influence), `Lithology`, `SoilDepth`, `WetnessMap`, `Climate` |
| Maske kompozisyonu | `SurfaceComposer`, `BiomeComposer`, `MaskCombine` / `Remap` / `MaskAdjust` / `ChannelExtract` |
| Tiled PBR detay | 4 katmanlı splat blend — `closesthit.rchit` §2b, katman başına albedo/roughness/metallic/normal/transmission + `layer_uv_scale` |
| Renk semantiği | `NodeSystem::ImageSemantic::Albedo` ve `::AO` **zaten tanımlı**, terrain hiç kullanmıyor |

Maske kalitesi doğrulanmış kabul ediliyor: flow / soil / snow maskeleri şu anda
üretim kalitesinde. Yani bu yol haritası **yeni analiz üretmiyor**, var olanı
renge çeviriyor. Tek istisna AO (Faz 4).

Ama bir önkoşul var, ve önceki taslak onu bir kısıt yerine bir erdem sanmıştı:
**tek bir çözünürlük sayısı üç ayrı işi birden yapıyor.** Faz 0 bunu ayırır.

---

## Faz 0 — ÜÇ çözünürlüğü ayır (önkoşul)

### Bugünkü durum: `heightmap.width` üç şeyin birden adı

| Rol | Nerede | Bugünkü bağ |
|---|---|---|
| **Alan (field)** — analiz ızgarası | `heightmap.data`, `hardnessMap`, `flowMap`, `erosionMapRGBA`, `analysisFields` (`TerrainSystem.h:123-160`) | `heightmap.width × height` |
| **Mesh** — vertex ızgarası | `updateTerrainMesh` (`TerrainManager.cpp:922`) | **birebir** `w × h` vertex, `2·(w-1)·(h-1)` üçgen |
| **Boya (paint)** — splat / makro renk | `resizeSplatMap` (`TerrainManager.cpp:1451`) | `max(512, heightmap.width)` |

Bu üçü farklı yönlere çekiyor:

- **Analiz yüksek ister.** Flow accumulation, ufuk AO, erozyon — hepsi ızgara
  frekansına duyarlı; 1k'da vadi ağı kaba çıkar.
- **Mesh düşük ister.** 2048² = 4.19M vertex ve **8.38M üçgen**, tek bir terrain
  için. Aynı sahnede 512² mesh 0.52M üçgen — 16× ucuz.
- **Boya yüksek ister.** Splat 4 katmanı ayıran maske; makro renk düşük frekanslı
  ama üstüne binen splat kenarları keskin olmalı.

Bunları tek sayıya bağlamak, üçünden ikisini her zaman yanlış yerde bırakır.

### Uygulama durumu (2026-08-21, parti 3 + Parti A)

| Kadran | Durum |
|---|---|
| `field_resolution` | ✅ zaten vardı (`heightmap.width/height`) |
| `mesh_resolution` | ✅ **CANLI DOĞRULANDI** — 0 = alanı takip et; `terrain.set_mesh_resolution` + `terrain.create(mesh_resolution=)` + panel kadranı |
| `paint_resolution` | ✅ **YAZILDI (Parti A)** — `TerrainSystem.h` + `TerrainManager.h/cpp` + `scene_ui_terrain_resolution.hpp`; 0 = alanı takip et; `resizePaintMaps` her iki haritayı birlikte taşır |
| `macroColorMap` + `macro_color_strength` | ✅ **ALANLARI YAZILDI (Parti A)** — `TerrainSystem.h`'e eklendi, serialize/deserialize edildi; texture upload ve shader bağlantısı Parti B'de |
| Alan çözünürlüğünde normal map | ⏳ ertelendi — normaller **alandan örnekleniyor** (bedava kısmı alındı), ama vertex ARASI detay hâlâ kayıp |
| Scatter'ın UV'den örneklemesi | ⏳ ertelendi — vertex aynası korundu, ama artık **yeniden örnekleniyor** (sessiz silme düzeltildi) |

★★★ **`paint_resolution` neden ertelendi:** maske grafiği o çözünürlükte
değerlendirilmeden kadranı açmak yalnızca yukarı örnekleme satın alır. Kadranın
kendisi panelin yalanı olur — bu dokümanın en çok uyardığı şeyin ta kendisi.
Kadran, değerlendirme yoluyla **birlikte** gelmeli.

★★ **Uygulamada çıkan tuzak:** analiz alanları vertex attribute'una
aynalanırken boyut testi `vertexCount`'a karşı yapılıyordu. Bu yalnızca iki
ızgara aynı olduğu için doğruydu; ayrıldıkları anda **her maske testi düşer ve
alan sessizce silinirdi** — foliage/scatter maskesiz kalır, hata yok, log yok.
Test artık alan hücre sayısına karşı, ve alanlar vertex ızgarasına bilinear
yeniden örnekleniyor.

★ İkinci tuzak: `updateDirtySectors` vertex dizisini **alan koordinatlarıyla**
indeksliyor. Ayrık mesh'te yanlış vertex'leri oynatırdı ve belirtisi bir
**sculpt hatası** gibi görünürdü. Ayrık mesh'te tam yola düşüyor.

### ★★★★★ Faz 0 ölçüldü (2026-08-21): 6 323 ms → 33 ms

Aynı 4096² alan, Solid viewport:

| | mesh = alan (4096) | **mesh 1024** |
|---|---|---|
| `accel.vulkan_solid.raster_geometry` | 5 587 ms, +7.4 GB | **422 ms, −2.5 GB** |
| `accel.cpu.embree_build` | 2 821 ms, +6.6 GB | **155 ms, +0.28 GB** |
| `scene.sync_transformed_vertices` | 53 ms | **3 ms** |
| kare döngüsü tıkanması | **6 323 ms** | **33 ms** |

Ve alan hiç kıpırdamadı: `resolution` 4096×4096, `sample_height` 103.227 →
103.227 (birebir aynı). Analiz çözünürlüğünden **sıfır** kayıp.

★ Negatif RSS bir hata değil: eski büyük mesh'in tamponları serbest bırakıldı.

★★ **`terrain.create` de `mesh_resolution` alıyor.** İlk ölçümde şu görüldü:
önce tam çözünürlükte kurup sonra küçültmek, o pahalı hızlandırma yapısını
**bir kez yine de kurduruyor**. Kadranı yalnızca sonradan sunmak, kullanıcının
şikâyet ettiği ilk üretimi hiç iyileştirmezdi.

### Karar: üç bağımsız kadran

```cpp
// TerrainObject
int field_resolution = 2048;   // analiz + heightmap ızgarası (otorite)
int mesh_resolution  = 1024;   // vertex ızgarası, alandan filtrelenerek örneklenir
int paint_resolution = 4096;   // splatMap + macroColorMap
```

Kurallar:

- **`mesh_resolution ≤ field_resolution`.** Mesh alanın *tüketicisi*. Alandan
  yüksek bir mesh, olmayan bilgiyi enterpolasyonla uydurur.
- **`paint_resolution` alandan bağımsız**, aşağı da yukarı da serbest.
- Üçü de sidecar'a (`.rtp.bin`) **ayrı ayrı** yazılır. Eski alan adı
  (`heightmap.width`) alan çözünürlüğü olarak kalır; mesh ve boya **yeni
  isimlerle** girer, ki eski proje "mesh = 2048" diye okunmasın.

### ★★★ Kritik: boya çözünürlüğü bir YENİDEN ÖRNEKLEME HEDEFİ DEĞİL, bir DEĞERLENDİRME çözünürlüğüdür

Splat'ı 4k'ya çıkarıp maskeleri hâlâ alan çözünürlüğünde hesaplayıp yukarı
örneklemek **hiçbir bilgi eklemez**. Sonuç daha yumuşak, daha "pahalı" görünür
ve kimse bunu bug diye raporlamaz — bu deponun en pahalı hata sınıfı: makul
görünen sonuç.

Bu yüzden `paint_resolution` maske grafiğinin **evaluate çözünürlüğüdür**:

- Analizden gelen girdiler (`Slope`, `flowMap`, `Wetness`) alan çözünürlüğünde
  yaşar ve boya ızgarasında **bilinear örneklenir** — burada yeni bilgi yok, ve
  olmaması normal.
- Node'un **kendi** ürettiği yüksek frekans (noise, warp, dither, threshold
  keskinliği) boya çözünürlüğünde hesaplanır — kazanç **buradadır.**
- Bir `SurfaceComposer` zincirinde tek bir prosedürel node bile yoksa,
  `paint_resolution > field_resolution` yalnızca dosya boyutu satın alır.
  **Panel bunu söylemeli** (bkz. çıkış testi).

### ★★ Düşük mesh, gölgeleme detayını geri vermek zorunda

`mesh_resolution < field_resolution` seçildiğinde vertex normalleri de düşer.
Alanın taşıdığı detay geri gelmezse terrain yassılaşır — ve suç **satmap'e**
atılır, tıpkı Faz 4'teki AO tuzağı gibi.

Bu yüzden Faz 0'ın parçası: alan çözünürlüğünde bir **terrain normal map**
üretilip mesh'in tangent-space normali olarak bağlanır. Mesh 512 iken alan 2048
ise, gölgeleme 2048'lik detayı görür.

★ **Dürüst sınır:** normal map **silueti geri getirmez.** Ufuk çizgisi ve
kenar profili `mesh_resolution`'a bağlı kalır. Kamera terrain'e paralel bakıyorsa
düşük mesh görünür. LOD / adaptif tessellation / displacement bu yol haritasının
işi değil; burada verilen söz "gölgeleme detayı korunur", "siluet korunur" değil.

### ★★ Foliage / scatter mesh'ten değil ALANDAN örneklemeli

Bugün `analysisFields`, `updateTerrainMesh` içinde **flatMesh vertex
attribute'una** aynalanıyor ve scatter oradan barycentric okuyor
(`TerrainSystem.h:145`). Mesh'i düşürdüğün an scatter maskesi de düşer — sessizce,
belirtisiz.

Faz 0 bunu çevirir: scatter/foliage alanı **UV üzerinden** örnekler
(`hitUV` → alan ızgarası, bilinear). Vertex aynası ya kalkar ya da yalnızca bir
hızlandırma önbelleği olarak, alanla aynı değeri döndürdüğü doğrulanarak kalır.

### 8k sınırı: kaldırıldı, yerine ölçülmüş maliyet

Önceki taslak 8k'yı reddediyordu — gerekçesi "8k = 64M vertex" idi. **O gerekçe
mesh'in alana kilitli olmasından geliyordu, ve Faz 0 tam olarak o kilidi
söküyor.** Alan 8k iken mesh 512 kalabilir; ret gerekçesi ortadan kalkar.

Yerine sabit sayı değil, kullanıcıya gösterilen tablo:

| Çözünürlük | splat RGBA8 | makro renk RGBA8 | heightmap f32 | erosionRGBA f32×4 | mesh üçgen |
|---|---|---|---|---|---|
| 1024 | 4 MB | 4 MB | 4 MB | 16 MB | 2.1 M |
| 2048 | 16 MB | 16 MB | 16 MB | 64 MB | 8.4 M |
| 4096 | 64 MB | 64 MB | 64 MB | 256 MB | 33.5 M |
| 8192 | 256 MB | 256 MB | 256 MB | 1 GB | 134 M |
| 16384 | 1 GB | 1 GB | 1 GB | 4 GB | 537 M |

- **Tek gerçek tavan cihazdan gelir:** `VkPhysicalDeviceLimits::maxImageDimension2D`
  (yaygın olarak 16384). Panel bu değeri **sorgulayıp gösterir**, kodda sabit
  yazmaz — sabit yazılmış bir limit, cihaz değiştiğinde yalan söyler.
- Kadranın üstünde canlı bir "bu ayar ~X MB" satırı. Kullanıcı 16k istiyorsa
  16k alır; **maliyeti gizlemek reddetmekten kötüdür.**
- Alt sınır 64 olarak kalır (bugünkü `scene_ui_terrain.hpp:259`).

### Heightmap içe aktarma: dosya çözünürlüğüne bağlılık kalkar

Bugün iki ayrı içe aktarma yolu var ve **ayrışmışlar**:

| | `createTerrainFromHeightmap` (`TerrainManager.cpp:358`) | `HeightmapInputNode::loadHeightmapFromFile` (`TerrainNodesV2.cpp:447`) |
|---|---|---|
| Bit derinliği | **8-bit** (`stbi_load`, `/255.0f`) → **256 seviye** | 16-bit (`stbi_load_16`, `/65535.0f`) |
| Ölçek | tamsayı `stride`, yalnızca **aşağı** | tamsayı `strideX/Y`, yalnızca **aşağı** |
| Hedef | `w/stride` — dosyadan türer | `w/strideX` — dosyadan türer |

Üç ayrı arıza:

1. **8-bit yol teraslama üretir.** 16-bit bir heightmap'i `createTerrainFromHeightmap`
   ile açmak yükseklikleri 256 seviyeye çöktürür; sonuç "kademeli yamaç" olarak
   görünür ve genelde erozyona veya normal kalitesine atfedilir. **İki yol tek
   16-bit yola indirilir** (kural 5: iki kod yolunu her ihtimale karşı yaşatma).
2. **Tamsayı stride hem kırpar hem aspect'i kaydırır.** 4097 px bir dosya
   `stride=2` ile 2048'e iner ve son satır düşer; 3000×2000 bir dosya
   `strideX=2, strideY=1` ile **oranı bozar.**
3. **Yukarı örnekleme yok.** 1024'lük bir dosya, 2048'lik bir alan grafiğini
   besleyemez.

Karar: **alan çözünürlüğünü kullanıcı seçer, içe aktarma dosyayı o hedefe
yeniden örnekler** — aşağı Lanczos (mevcut `ImageResample::lanczos_resample_u16`),
yukarı bicubic. Dosya boyutu artık yalnızca bir *kaynak kalitesi* bilgisidir,
bir *hedef* değil. Panel dosyanın kendi boyutunu "kaynak: 1024×1024" olarak
gösterir ve hedefin üstündeyse "yukarı örnekleniyor — yeni detay üretilmez"
uyarısı verir. ★ Bu uyarı, boya çözünürlüğündeki uyarının aynısı: **büyütmek
bilgi eklemez.**

### Faz 0 çıkış testleri

- **Kimlik turu:** üç çözünürlük de eşitken (bugünkü hal) render **bit-bit aynı**.
  Değişiyorsa örnekleme yolu Faz 0 öncesiyle uyuşmuyordur, ve sonraki her ölçüm
  bu farkın üstüne biner.
- **Mesh düşür, alan sabit:** alan 2048 sabit, mesh 2048 → 512. Üçgen sayısı 16×
  düşmeli, **gölgeleme detayı gözle ayırt edilemez kalmalı** (normal map çalışıyor).
  Yassılaştıysa normal map bağlanmamıştır. Siluet farkı **beklenen**, bug değil.
- **★ Boya yükselt, prosedürel node YOK:** salt analiz zincirinde splat 2048 → 4096.
  Maske **ölçülebilir biçimde daha keskin olmamalı** (yalnızca yeniden örneklendi).
  Keskinleştiyse alan zaten daha yüksek bilgi taşıyordu, yani bir yerde erken
  düşürülüyor demektir.
- **Boya yükselt, prosedürel node VAR:** aynı zincire bir noise ekle. **Şimdi**
  keskinleşmeli. Keskinleşmiyorsa graph hâlâ alan çözünürlüğünde değerlendiriliyor
  ve `paint_resolution` yalnızca bir resize hedefi olarak bağlanmıştır — bu,
  bütün Faz 0'ın sessizce boşa çıktığı hâldir.
- **Scatter parite:** aynı foliage sahnesi mesh 2048 ve mesh 512 ile. Bitki
  dağılımı **aynı** kalmalı. Kabalaştıysa scatter hâlâ vertex attribute'undan
  okuyor.

---

## Faz 1 — Veri: makro renk haritası

`TerrainObject`'e ekle:

```cpp
std::shared_ptr<class Texture> macroColorMap; // RGBA8, paint_resolution
float macro_color_strength = 0.0f;            // 0 = kapalı (varsayılan)
```

Kurallar:

- Makro renk **splat ile aynı çözünürlükte** (`paint_resolution`) tutulur ve
  **aynı yeniden boyutlandırma fonksiyonundan** geçer. İkisi arasında kayabilen
  ikinci bir örnekleme yolu, kazandıracağı hiçbir şeye değmez.
- `resizeSplatMap` Faz 0'da `resizePaintMaps(terrain)` hâline gelir: hedef artık
  `heightmap.width` değil `paint_resolution`, ve iki haritayı birlikte taşır.
  ★ Fonksiyonun **adı da değişmeli** — anlamı değişti (kural 5).
- Varsayılan `strength = 0.0f`, ve harita yoksa da 0. Yani var olan sahneler
  bit-bit aynı render edilir.
- Proje sidecar'ına (`.rtp.bin`) yazılır; splat ile aynı kap.

**Çıkış testi:** haritayı elle düz kırmızıya doldur, `strength = 1` → terrain
kırmızıya boyanır ama tile detayı (normal/roughness varyasyonu) **görünür kalır**.
Detay kaybolduysa Faz 3'teki modülasyon formülü yanlış bağlanmıştır.

## Faz 2 — Node'lar: SatMap + ColorOutput

İki yeni `NodeType` (enum sonuna eklenir, serileştirme kararlılığı için):

**`SatMap`** — kolorizer.

- Girdi: `Primary` (Image2D/Mask), `Secondary` (Image2D/Mask, opsiyonel).
- Çıktı: `Color` (Image2D, `ImageSemantic::Albedo`, 3 kanal).
- LUT iki kaynaktan biri:
  1. **Gradient stop'ları** — node içinde serileştirilir, panelden düzenlenir.
  2. **LUT görüntüsü** — 256×256 PNG import. Gaea satmap'leri literal olarak budur.
- `Secondary` bağlıysa LUT 2B örneklenir (`u = Primary`, `v = Secondary`),
  bağlı değilse `v = 0.5` sabit.
- Parametreler: `contrast`, `saturation`, `value_bias`, `primary_remap` (min/max).
- ★ **`paint_resolution`'da değerlendirilir.** Girdi maskesi alan çözünürlüğünden
  geliyorsa bilinear yükseltilir; LUT örneklemesi ve `contrast` boya ızgarasında
  yapılır — kontrast eğrisi keskin bir renk sınırı üretebildiği için bu, yeniden
  örnekleme değil gerçek kazançtır.

**`ColorOutput`** — `SplatOutput`'un ikizi. Girdi `Color`, `terrain->macroColorMap`'e
yazar. `strength` burada değil terrain'de yaşar: o bir görünüm kadranı, graph
verisi değil.

★ **Lisans notu:** Gaea'nın kendi LUT görüntüleri depoya konulmaz ve dağıtılmaz.
Kullanıcı kendi lisanslı dosyasını import eder; depoya yalnızca kendi yazdığımız
gradient preset'leri girer.

**Çıkış testi:** `TerrainAnalysis.Slope` → `SatMap.Primary` → `ColorOutput`,
iki stop'lu (yeşil→gri) gradient. Dik yamaçlar gri, düzlükler yeşil. Graph
kaydedilip yüklendiğinde gradient stop'ları aynı.

## Faz 2b — Arazi tipine göre ön tanımlı SatMap kütüphanesi

Boş bir gradient editörü kullanışlı değil: kullanıcı "alpin" ister, sekiz stop'un
konumunu değil. Ön tanımlılar bu yüzden bir konfor özelliği değil, **SatMap'in
asıl arayüzü**.

### ★★ Bir preset yalnızca gradient DEĞİLDİR

`SatMap` iki eksenli örneklendiği için (`u = Primary`, `v = Secondary`), bir renk
rampasını hangi alanın sürdüğünü söylemeyen preset **anlamsızdır**. Aynı stop
listesi `Slope`'a bağlanırsa kayalık yamaç, `Height`'a bağlanırsa kar çizgisi
üretir. O yüzden preset kaydı şunları birlikte taşır:

```jsonc
{
  "id": "alpine_snowline",
  "label": "Alpin — kar çizgisi",
  "primary":   { "field": "terrain.height",  "remap": [0.35, 0.95] },
  "secondary": { "field": "terrain.slope",   "remap": [0.0, 0.6] },   // yoksa v = 0.5
  "interp": 0,
  "stops": [ [0.00, 0.28,0.34,0.22], [0.45, 0.45,0.44,0.40], [0.72, 0.72,0.73,0.75], [1.00, 0.96,0.97,1.00] ],
  "contrast": 1.0, "saturation": 0.9, "value_bias": 0.0
}
```

★ **`remap` preset'in parçası olmak zorunda.** Kar çizgisi mutlak yükseklikte
değil, terrain'in **kendi** min/max aralığında bir orandadır; `heightmap.min_value`
/ `max_value` zaten hesaplanıyor (`TerrainManager.cpp:938`). Remap'siz bir preset,
`scale_y` değişen her sahnede sessizce yanlış yere düşer — ve bu, "renkler biraz
kaymış" diye geçiştirilecek türden bir arıza.

### Mevcut `ColorRampNode` yeniden kullanılır — ama sınırı miras ALINMAZ

`MaterialNodesV2.h:2716`'da tam ihtiyacımız olan şey duruyor:
`ColorRampNode::Stop { float pos; float col[3]; }`, `interpolation` (0 Linear /
1 Constant), `[pos,r,g,b]` JSON biçimi ve `scene_ui_materialnodes.hpp` içinde
çalışan bir stop editörü widget'ı.

SatMap **aynı Stop yapısını, aynı JSON biçimini ve aynı widget'ı** kullanır.
İkinci bir gradient tipi yazmak, iki serileştirme biçimi ve iki editör demek.

★★★ **Ama `kMaxStops = 8` miras ALINMAZ.** Gaea satmap'leri sürekli rampalardır;
8 stop bir kanyon katmanlanması için az. SatMap 16'ya çıkar, **ve paylaşılan
widget'taki 8 sabiti bir parametre hâline gelir.** Sabit kalırsa 16 stop'lu bir
preset yüklenir, sessizce 8'e kırpılır (`deserializeParams` içindeki `break`
tam olarak bunu yapıyor) ve kullanıcı yalnızca "preset biraz sönük çıkmış"
görür. Bu, kimsenin bug diye raporlamayacağı bir kayıptır.

### Kütüphane

Depoya girecek ilk set — hepsi **bizim yazdığımız** gradyanlar (Faz 2'deki lisans
notu: Gaea LUT'ları depoya girmez):

| Preset | Primary | Secondary | Karakter |
|---|---|---|---|
| `alpine_snowline` | Height | Slope | Çayır → kaya → kar; dik yüzeylerde kar tutmaz |
| `desert_dunes` | Height | Wetness | Sıcak kum → açık tepe; çukurlarda koyu nem |
| `canyon_strata` | Height | Lithology | Sert yatay katmanlar (`interp = Constant`) |
| `temperate_forest` | Wetness | Slope | Yeşil vadi → kuru sırt |
| `tundra_permafrost` | Height | Wetness | Gri-kahve likenli, donmuş turba lekeleri |
| `volcanic_basalt` | SoilDepth | Occlusion | Koyu bazalt → kızıl skorya; AO ile çatlaklar |
| `coastal_cliffs` | Height | Flow | Tuz beyazı kenar, ıslak koyu taban |
| `eroded_badlands` | Erosion (wear) | Deposition | Aşınan sırt açık, biriken taban koyu |

- `canyon_strata` **`interpolation = 1` (Constant)** olmalı: katmanlı kaya
  yumuşak geçişle inandırıcı olmuyor. Bu, mevcut `ColorRampNode` alanının
  zaten desteklediği bir mod — yeni kod değil.
- `volcanic_basalt` **Faz 4'e bağımlı** (Occlusion). Faz 4 gelmeden bu preset
  kütüphaneye girmez; `Secondary`'yi sessizce `Convexity`'ye düşürmek, preset'in
  neden yassı göründüğünü açıklanamaz hâle getirir. ★ Eksik alanı olan preset
  **listede görünür ama devre dışıdır**, ve nedeni yazar.

### Nerede yaşarlar

`assets/terrain/satmap_presets/*.json` — her preset tek dosya, kütüphane
dizini tarar. Uygulanan recipe tam ramp/ayar snapshot'ını graph JSON'una yazar;
asset daha sonra kaldırılırsa kaydedilmiş proje aynı görünümü korur. Derlenmiş
bir tabloya gömülmez:

- Kullanıcı kendi preset'ini dosya bırakarak ekler; yeni bir preset **derleme
  gerektirmez.**
- Kullanıcının kendi lisanslı LUT'undan türettiği bir rampa depoya karışmaz.
- ★ Bir preset dosyası bozuksa **atlanır ve loglanır**; sessizce varsayılan
  gradient'e düşmek, "preset uygulandı ama hiçbir şey değişmedi" hâlini üretir.

**Çıkış testi:** `alpine_snowline` uygulanır, sonra `scale_y` iki katına çıkarılır.
Kar çizgisi **terrain'in oranına göre aynı yerde** kalmalı. Aşağı/yukarı kaydıysa
`remap` mutlak yüksekliğe bağlanmıştır. ★ İkinci tur: 16 stop'lu bir preset
kaydedilip yüklenir; stop **sayısı** aynı mı — 8'e düştüyse `kMaxStops` mirası
kalmıştır.

## Faz 3 — GPU ve shader

`VkTerrainLayerData` şu anda 48 byte ve içinde `uint32_t _pad[2]` duruyor.
İki alan tam oraya sığar, **`static_assert` boyutu değişmez**:

```cpp
uint32_t macro_color_tex;   // eski _pad[0]
float    macro_strength;    // eski _pad[1]
```

Dokunulacak yerler:

- `IBackend::TerrainLayerData` (CPU tarafı ikizi),
- `VulkanBackendAdapter::uploadTerrainLayerMaterials`,
- `VulkanViewportBackend::uploadTerrainLayerMaterials` — **ikisi de**; birini
  unutmak viewport ile render'ın ayrıştığı, hatasız bir duruma yol açar,
- `closesthit.rchit` §2b, blend döngüsünden **sonra**.

★ Mesh UV'si `0..1` normalize üretiliyor (`updateTerrainMesh`, `x/divW`), yani
shader tarafı boya çözünürlüğünden **bağımsızdır** — Faz 0 shader'da hiçbir
değişiklik gerektirmez. Bu, üç kadranın ayrılabilmesinin asıl sebebi.

Shader, blend bittikten sonra:

```glsl
if (tl.macro_color_tex > 0u && tl.macro_strength > 0.0) {
    vec3  macro      = texture(materialTextures[nonuniformEXT(int(tl.macro_color_tex))], hitUV).rgb;
    float detailLuma = max(luminance(blendAlbedo), 1e-3);
    vec3  relit      = macro * (blendAlbedo / detailLuma);
    blendAlbedo      = mix(blendAlbedo, relit, tl.macro_strength);
}
```

★★★ **Buradaki tek kritik karar:** makro renk albedo'yu **değiştirmez, modüle
eder**. `mat.albedo = macro` yazmak derlenir, çalışır ve uzaktan güzel görünür —
ama tile detayının tamamını siler ve sonuç yakından plastikleşir. Bu, kimsenin
bug diye raporlamayacağı türden bir başarısızlıktır: makul görünen sonuç.
Yukarıdaki formül detayın **kendi ortalamasına oranını** korur, makro'dan
yalnızca ton ve seviye alır.

**Çıkış testi:** aynı sahne `strength = 0` ve `strength = 1` ile render edilir.
0'da görüntü Faz 1 öncesiyle **bit-bit aynı** olmalı. 1'de renk değişir ama
yakın plan doku detayı ölçülebilir biçimde korunur.

## Faz 4 — Occlusion (ayrılabilir)

SatMap'in "detaylı" hissi büyük ölçüde ikinci LUT ekseninde AO'dan gelir.
Bizde `Concavity` / `Convexity` var, gerçek ufuk taraması yok.

- `terrain_horizon_ao.comp` — heightmap üzerinde N yönlü ufuk taraması,
  **alan çözünürlüğünde** (mesh'te değil — AO bir analiz, bir gölgeleme değil).
- Sonuç `analysisFields["occlusion"]` olarak yayınlanır; `TerrainFieldsOutput`
  üzerinden hem `SatMap.Secondary`'ye hem foliage/scatter'a açılır.

Bu faz **bağımsızdır**: onsuz da SatMap çalışır, yalnızca daha yassı görünür.
Faz 1-3 doğrulanmadan buraya girme — yassılığın kaynağını AO yokluğuna
atfetmek, Faz 3'teki bir modülasyon hatasını **veya Faz 0'daki eksik normal
map'i** gizleyebilir. ★ Bu partide yassılığın artık **üç** olası kökü var.

## Faz 5 — Script/IPC yüzeyi (kural 1: dört dokunuş)

| Metot | İş |
|---|---|
| `terrain.set_resolutions` | `field` / `mesh` / `paint` — üçü tek çağrıda, kısmi ayar da serbest |
| `terrain.get_resolutions` | Üç değer + tahmini bellek + cihazın `maxImageDimension2D` değeri |
| `terrain.bake_satmap` | Graph'ı değerlendirip makro renk haritasını üretir |
| `terrain.list_satmap_presets` | Kütüphaneyi döndürür — id, etiket, sürücü alanlar, **devre dışıysa nedeni** |
| `terrain.apply_satmap_preset` | Bir preset'i `SatMap` node'una uygular (stop'lar + remap + sürücü alanlar) |
| `terrain.get_satmap_gradient` / `set_satmap_gradient` | Stop listesini ham okur/yazar — ★ **16 stop kırpılmadan geri gelmeli** |
| `terrain.import_color_lut` | LUT PNG'sini bir `SatMap` node'una bağlar |
| `terrain.set_macro_color_strength` | `strength` kadranı |
| `terrain.export_color_map` | Üretilen haritayı diske yazar — **görsel doğrulamanın ölçüm ucu** |
| `terrain.export_field` | Bir `analysisFields` alanını diske yazar — boya/alan ayrımının ölçüm ucu |

Her biri için: `RtApi.h` + `RtApi*.cpp` → `RtIpc*.cpp` dispatch → `RtPython*.cpp`
binding → `RtIpcSecurity.cpp` yetkisi → `gen_ipc_descriptors.py` + overlay satırı.
Dördü yapılıp biri unutulursa metot **sessizce reddedilir** (fail-closed).

★ `terrain.import_heightmap` da bu partide değişir: `target_resolution` parametresi
alır ve dosya boyutundan bağımsızlaşır. Eski imza dosya boyutunu hedef sanıyordu.

★ Tersi de geçerli: üç çözünürlük kadranı, `strength` ve gradient stop'ları
panelden de düzenlenebilir olmalı. Yalnızca script'ten ayarlanabilen bir kadran,
panelden eklenen bir `SatMap` node'unu görünmez biçimde etkisiz bırakır.

---

## Ek A — İlk üretim neden yavaş (4k), ve ne yapılabilir

Şikâyet: 4k'da ilk terrain üretimi uzun sürüyor. Kod okundu; **hesap değil,
tahsisat ve topoloji maliyeti.** Sıcak yol `updateTerrainMesh`
(`TerrainManager.cpp:922`) + `gridToFlatMesh` (`:182`).

4096² için sayılar: **16.78 M vertex**, **33.55 M üçgen**, **100.66 M indeks**.

| Tampon | Nerede | 4k'da |
|---|---|---|
| `positions` + `normals` (geçici) | `:961-965` | 402 MB |
| `uvs` (geçici) | `:1073` | 134 MB |
| `indices` (geçici) | `:1084` | 403 MB |
| `P`, `N`, `P_orig`, `N_orig` | `gridToFlatMesh` | 805 MB |
| `uv` + `materialID` | `gridToFlatMesh` | 168 MB |
| `geometry->indices` | `gridToFlatMesh:231` | 403 MB |
| **tepe** | | **≈ 2.3 GB** |

Bu boyutta iş, aritmetikle değil **sayfa hatalarıyla** sınırlıdır. Dört ayrı
iyileştirme var, kazanç sırasına göre:

### Durum özeti (parti 1)

| Adım | Durum |
|---|---|
| Ölçüm aleti (`rt.perf`) | ✅ **CANLI DOĞRULANDI** |
| 1. Faz 0 (mesh/alan ayrımı) | ✅ **`mesh_resolution` YAZILDI** (parti 3) — `paint_resolution` bilerek ertelendi |
| 2. Geçici tamponları kaldır | ✅ **DOĞRULANDI** — 4k mesh 380 ms / +1.3 GB (staging yok) |
| 3. İndeks kopya döngüsü | ✅ **DOĞRULANDI** |
| 4. `P_orig` / `N_orig` sök | ❌ **REDDEDİLDİ** — ölçüldü, sözleşme gerçek |
| 5. Çift mesh doldurma (`createTerrain`) | ✅ **DOĞRULANDI** — `count` 2 → 1 |
| 6. Tembel hızlandırma yapısı ölçümü | ✅ **DOĞRULANDI** (parti 2) — kök: `accel.vulkan_solid.raster_geometry` 6.4 s / +7.4 GB |

### 0. ✅ Ölçüm aleti: `rt.perf` (YAZILDI)

Hiçbir optimizasyon seçilmeden önce ölçüm gerekiyordu, ve mevcut alet ölçmüyordu:
`MeshProfileTimer.h` sonuçlarını yalnızca Scene Log'a yazıyordu **ve makrosu
`((void)0)`'a derleniyordu**. İkisi birlikte bu projenin test modelini tamamen
boşa çıkarıyor — ajan Scene Log'u okuyamaz, kapalı makro zaten hiçbir şey ölçmez.

Yerine gelen: `PerfProfile.h` / `PerfProfile.cpp` — bölümler artık **değer**.

- `RTPERF_SCOPE("ad")`, işi kim yapıyorsa oraya konur; kayıt defteri kendi
  kilidini taşır, worker thread'den de yazılabilir.
- Her bölüm: `last_ms`, `total_ms`, `max_ms`, `count`, **çalışma kümesi deltası**
  (`last_rss_delta_mb`) ve monotonik `seq`. ★ Bellek rakamı zorunlu: bu iş
  tahsisat sınırlı, ve yanına bellek konmamış bir süre yanlış yarıyı optimize
  ettirir.
- `MeshProfileTimer.h` **silindi**; 10 çağrı noktası `RTPERF_SCOPE`'a çevrildi
  (kural 5: iki mekanizmayı birlikte yaşatma). Bunların ikisi zaten tam
  ihtiyacımız olan yerdeydi: `Renderer::rebuildBVH(...)` ve
  `Renderer::rebuildBackendGeometry(GPU)`.
- Terrain tarafına eklenen bölümler: `terrain.graph.evaluate` / `.height` /
  `.aux_outputs` / `.finalize_mesh`, `terrain.mesh_fill` / `.create` / `.update`
  / `.publish_fields`, `terrain.splat_resize`.

Script/IPC yüzeyi (kural 1, dört dokunuş + descriptor): `perf.list`, `perf.get`,
`perf.reset`, `perf.set_logging` → `rt.perf.*`. Yetki: dördü de `Read`.

★★ **Okumalar kare döngüsü kuyruğuna girmiyor.** Diğer bütün IPC sorguları
sahne durumu okuduğu için kuyruğa alınır; burada bu yanlış olurdu — "bu yapım
zamanı nereye harcıyor" sorusunun en değerli anı UI thread'i **meşgulken**, ve
kuyruğa alınmış bir okuma tam da tarif etmek istediği işin arkasında beklerdi.

★ `perf.get` bulunmayan bölüm için **sıfır değil `found:false`** döner. Sıfır
"ölçtüm, bedavaymış" diye okunur — eksik ölçümün yanlış ölçüme dönüştüğü nokta
tam olarak burasıdır.

★★★ **İsim `perf.`, `profile.` değil:** `mesh.profile.sweep.*` zaten var ve
orada "profile" bir **profil eğrisi** demek. Tek metot tablosunda bir kelimenin
iki alakasız anlamı bir okuma tuzağıdır.

### 1. ★★★ Faz 0 zaten bu işin optimizasyonudur

Kullanıcının şikâyeti Faz 0'ın gerekçesidir: alan 4096 kalıp mesh 1024 olursa
vertex sayısı **16× düşer** (16.78 M → 1.05 M), üçgen 33.55 M → 2.1 M, geometri
belleği ~1.37 GB → ~86 MB. Analiz çözünürlüğü hiç kaybedilmeden.

★ Ve asıl kazanç muhtemelen mesh doldurmada bile değil: `updateTerrainMesh`
bitişinde `g_bvh_rebuild_pending` / `g_optix_rebuild_pending` /
`g_vulkan_rebuild_pending` **hepsi** işaretleniyor (`:1085-1092`). 33.55 M üçgenlik
bir BVH/BLAS yapımı, mesh doldurmadan büyük olasılıkla **daha pahalıdır**.

★★★ **Bu yüzden ilk iş ölçmek:** mesh doldurma, BVH yapımı ve graph
değerlendirmesi ayrı ayrı zamanlanmadan hiçbir optimizasyon seçilmemeli.
"Yavaş" bir belirtidir, bir ölçüm değil — ve bu depoda yanlış bileşeni optimize
etmek daha önce oldu.

### 2. ✅ Geçici tamponları tamamen kaldır (~940 MB) (YAZILDI)

`positions` / `normals` / `uvs` / `indices` doldurulup `gridToFlatMesh` içinde
**ikinci kez** kopyalanıyordu. Artık her iki dal da `resize_vertices` +
`add_attribute` sonrası doğrudan `P` / `N` / `P_orig` / `N_orig` / `uv` /
`materialID` işaretçilerine yazıyor; dört geçici vektör ve ikinci geçiş düştü.

★ `calculateNormal()` heightmap'i doğrudan okuduğu için geçici pozisyonlara
bağımlı bir şey yoktu — kaldırma güvenliği buradan geliyor.

★ `gridToFlatMesh` (TerrainManager kopyası) böylece **ölü kaldı ve silindi**
(kural 5). River/Water kendi kopyalarını taşıyor, onlar bu partinin dışında.

### 3. ✅ İndeks kopyası: eleman eleman döngü → doğrudan üretim (YAZILDI)

```cpp
tm->geometry->indices.resize(indices.size());
for (size_t i = 0; i < indices.size(); ++i) tm->geometry->indices[i] = indices[i];
```

`gridToFlatMesh:230-231`. 4k'da bu **100.66 M yinelemelik seri bir döngüydü** —
`geometry->indices` hizalı bir ayırıcı kullandığı için düz bir vektör ataması
kabul etmiyordu. İndeksler artık doğrudan `geo.indices` içine paralel üretiliyor;
hem geçici 403 MB hem seri döngü kalktı.

### 4. ❌ `P_orig` / `N_orig`: 402 MB birebir kopya — REDDEDİLDİ

Terrain transform'u pratikte saf ötelemedir, yani `P_orig ≈ P` ve `N_orig == N`.
Dört tam Vec3 dizisinin ikisi, bir matris çarpımı uzağındaki veriyi saklıyor.

★ **Ölçüldü, ve cevap "sökme".** `P_orig` / `N_orig` depoda **46 dosyada 296
kez** okunuyor — `VulkanBackend`, `VulkanViewportBackend`, `EmbreeBVH`,
`MeshModifiers`, `RtApiSculpt`, `ParticleRenderBridge`, `Triangle`, `OptixWrapper`
dahil. Bu 402 MB bir israf değil, **yayılmış bir sözleşme**. Terrain için
atlamak, o okuyucuların hepsinde terrain'e özel bir "yoksa P kullan" dalı
gerektirirdi — yani tek bir tahsisatı kaldırıp yerine kırk yerde sessiz bir
ayrışma riski koymak.

★ Bu maddeyi ölçmeden uygulamak, bu partinin en pahalı hatası olurdu: tasarruf
gerçek, gerekçe yanlıştı.

### Ölçekten bağımsız iki not

- **`std::minmax_element`** (`:939`) 16.78 M float üzerinde seri koşuyor. Tek
  başına küçük, ama paralel bir azaltma bedava.
- **İptal ve ilerleme.** 4k'da hangi optimizasyon yapılırsa yapılsın iş saniyeler
  sürer. `defer_mesh_updates` bayrağı zaten ana-thread finalize modeli kuruyor;
  üretim ilerlemesinin panele bir ilerleme çubuğu olarak çıkması, "uygulama
  dondu" algısını kaldırır. ★ Ama **ölçüm yerine geçmez** — ilerleme çubuğu
  yavaşlığı görünür kılar, azaltmaz.

### ★★★★★ ÖLÇÜLDÜ (2026-08-21): mesh doldurma darboğaz DEĞİL

Canlı uygulamada IPC ile ölçüldü, 4096² düz terrain:

| Faz | Süre | Nasıl ölçüldü |
|---|---|---|
| `terrain.create` çağrısı (kuyruklu) | **721 ms** | dışarıdan duvar saati |
| ├ `terrain.mesh_fill.create` | 362 ms, **+1312 MB** | `perf.list` |
| └ `terrain.mesh_fill.update` (gereksiz) | 156 ms | `perf.list` |
| **Çağrı DÖNDÜKTEN sonra kare döngüsü tıkalı** | **6684 ms** | kuyruklu sondaj |
| bunun ölçülen kısmı | **0 ms** | — |

**Toplam ~7.4 s, ve ölçüm bunun %6'sını görüyordu.**

★★★ Yani "4k terrain uzun sürüyor" şikâyetinin **%90'ı mesh doldurmada değil**,
çağrı döndükten sonra kare döngüsünde tembel kurulan hızlandırma yapılarında.
Mesh tahsisat yolunu optimize etmeye devam etmek, sürenin onda birinde
uğraşmak olurdu.

★★ **Ölçüm tekniği kaydedilmeye değer:** `perf.list` kuyruğa **girmez**,
`scene.list_objects` girer. İkisinin duvar saati farkı doğrudan **kare
döngüsünün meşgul olduğu süredir**. Tıkanma başka türlü görünmezdi — uygulama
donmuş gibi durur, hiçbir sayaç bir şey söylemez.

★ **Tripwire'ın susması yokluğu kanıtlamaz.** İlk turda hiçbir BVH bölümü
görünmedi ve bu "BVH bedava" diye okunabilirdi. Gerçek sebep: kare döngüsü
`Renderer::rebuildBVH`'yi **çağırmıyor** — `EmbreeBVH` doğrudan bir async
lambda içinde kuruluyor (`Main.cpp:6873`), ve Vulkan tarafı hiç ölçülmüyordu.

**Parti 2'de kapatılan delik** — çağrı noktalarına değil, **işi yapan
fonksiyonlara** kondu (her çağıranı yakalar):

| Bölüm | Yer |
|---|---|
| `scene.sync_transformed_vertices` | `SceneUI::syncAllTransformedVertices` — ★ ANA THREAD'de, async BVH gönderilmeden önce |
| `accel.cpu.embree_build` | `EmbreeBVH::build` |
| `accel.vulkan_rt.rebuild` | `VulkanBackendAdapter::rebuildAccelerationStructure` |
| `accel.vulkan_rt.update_geometry` | `VulkanBackendAdapter::updateGeometry` |
| `accel.vulkan_rt.raster_geometry` | `VulkanBackendAdapter::buildRasterGeometry` |
| `accel.vulkan_solid.raster_geometry` | `VulkanViewportBackend::buildRasterGeometry` |

★ Hızlandırma yapıları **tembel** kurulur (önce Solid/Vulkan RT, diğerleri
gerekirse), yani hangi bölümün dolduğu **aktif viewport moduna bağlıdır**. Boş
bir bölüm burada "yapılmadı" demektir, "bedava" değil.

### ✅ Bulundu ve düzeltildi: her scripted terrain mesh'i İKİ KEZ kuruyordu

`terrain.mesh_fill` sayacı `count=2` gösterdi. Sebep: `rtapi::createTerrain`
önce `TerrainManager::createTerrain`'i çağırıp mesh'i kurduruyor, sonra
`heightmap.scale_y`'yi atayıp **`updateTerrainMesh`'i tekrar** çağırıyordu —
tek bir skaleri sonradan uygulamak için tam bir vertex+normal geçişi.
4k'da **156 ms**, ve çözünürlüğün karesiyle büyüyor.

`createTerrain` artık `height_scale`'i **parametre olarak** alıyor; ikinci geçiş
silindi.

★ Bu ancak sayaç `count` tuttuğu için görüldü. Yalnızca "son süre" raporlayan
bir profiler bunu **yapısal olarak gösteremezdi** — iki çağrının ikincisi
tamamen makul bir süre raporluyordu.

### ★★★★★ DELİK KAPANDI (parti 2 doğrulandı): maliyet SOLID RASTER GEOMETRİSİNDE

4096² terrain, aynı ölçüm tekrarlandı:

| Bölüm | Süre | RSS |
|---|---|---|
| `accel.vulkan_solid.raster_geometry` | **6394 ms** | **+7.4 GB** |
| `accel.cpu.embree_build` (worker thread) | 3278 ms | +6.4 GB |
| `terrain.mesh_fill` | 380 ms | +1.3 GB |
| `scene.sync_transformed_vertices` | 54 ms | — |
| kare döngüsü tıkanması (kuyruklu sondaj) | 7142 ms | |

★ Kayıtlı toplam (10.5 s) duvar saatinden (7.7 s) **büyük** — çünkü Embree bir
worker thread'de çakışık koşuyor. Bu bir hata değil, ölçümün doğru okunma
biçimi: bölümler birbirini dışlamaz.

### Çözünürlük ölçeklemesi (ölçüldü, 2026-08-21)

| Çözünürlük | Üçgen | `mesh_fill` | `solid_raster` | `embree_build` | Döngü tıkanması |
|---|---|---|---|---|---|
| 512 | 0.5 M | 6 ms | 110 ms | 34 ms | **163 ms** |
| 1024 | 2.1 M | 21 ms | 324 ms | 149 ms | **398 ms** |
| 2048 | 8.4 M | 94 ms | 1 407 ms | 788 ms | **1 609 ms** |
| 4096 | 33.5 M | 380 ms | 6 394 ms | 3 278 ms | **7 142 ms** |

Hepsi üçgen sayısıyla **doğrusal**. Ve bu tablo Faz 0'ın kazancını bir tahmin
olmaktan çıkarıyor:

> ★★★★★ **Alan 4096 + mesh 1024 → 7 142 ms yerine ~398 ms. Yaklaşık 18×**,
> ve analiz çözünürlüğünden hiçbir şey kaybedilmeden.

★★★ **Faz 0'ın gerekçesi böylece DEĞİŞTİ.** "Mesh doldurma pahalı" değil —
mesh doldurma 4k'da toplam maliyetin %5'i. Gerçek gerekçe: **üçgen sayısı
hızlandırma yapılarını pahalılaştırıyor**, ve `mesh_resolution` onu doğrudan
kesen tek kadran.

★★ **Ayrı bir kırmızı bayrak:** Solid raster geometrisi 33.5 M üçgen için
**+7.4 GB** istiyor — üçgen başına ~220 byte. Bu, indeksli mesh'in kaynak
başına açıldığına (unwelded soup) işaret ediyor. Kendi başına bir iş; bu yol
haritasının kapsamı değil ama **ölçüldü ve kayda geçti**, çünkü Faz 0 bunu
16× küçültür ama **ortadan kaldırmaz.**

★ **Tembel kurulum doğrulandı:** ilk turda yalnızca `accel.vulkan_solid.*`
doldu, `accel.vulkan_rt.*` hiç görünmedi — çünkü viewport Solid moddaydı. Boş
bir bölüm "bedava" değil, **"bu modda çalışmadı"** demektir.

### ★★★★ Rendered modunda ölçüldü: SOLID RASTER YİNE KURULUYOR

Viewport `rendered`'a alınıp aynı 4096² ölçümü tekrarlandı:

| Bölüm | Süre | RSS |
|---|---|---|
| `accel.vulkan_solid.raster_geometry` | **5915 ms** | **+7.4 GB** |
| `accel.vulkan_rt.update_geometry` | 2999 ms | +225 MB |
| `accel.cpu.embree_build` | 2620 ms | +6.4 GB |
| `scene.sync_transformed_vertices` | **502 ms** | — |
| `terrain.mesh_fill` | 354 ms | +1.3 GB |
| `accel.vulkan_rt.rebuild` | 0.6 ms | — |
| kare döngüsü tıkanması | **9858 ms** | |

Üç şey çıktı:

1. ★★★ **Solid raster geometrisi Rendered modunda da kuruluyor** — 5.9 s ve
   7.4 GB, ekranda gösterilmeyen bir temsil için. Toplam tıkanma bu yüzden
   Solid'deki 7.1 s'den **9.9 s'ye çıkıyor**, RT yolu üstüne binerek.
   ★ Bunu "gereksiz" diye yazmıyorum çünkü **ölçmedim**: Solid raster'ı
   moddan bağımsız tutan bir sebep olabilir (moda geri dönüş, seçim/overlay).
   Ama 7.4 GB'lık bir eager build için o sebebin **yazılı olması gerekir** —
   şu anda yok. Bakılacak ilk soru bu.
2. `accel.vulkan_rt.rebuild` **0.6 ms** — asıl RT işi `update_geometry`'de.
   İsme bakıp "rebuild pahalı" demek yanlış iz olurdu.
3. `scene.sync_transformed_vertices` 54 ms → **502 ms**. Ana thread'de, async
   BVH gönderilmeden önce. Solid'de neden onda biri olduğu ayrıca bakılacak.

---

## Sıralı kontrol listesi (uygulandığında)

0. **★★★ 4k üretim profili** — graph eval / mesh fill / BVH+BLAS ayrı ayrı
   zamanlanır (Ek A). Her şeyden önce, çünkü hangi optimizasyonun anlamlı olduğunu
   **yalnızca bu belirler**; sırayla değil, ölçümle karar verilir.
1. **Faz 0 kimlik turu** — üç çözünürlük eşitken render bit-bit aynı mı. Bağımsız,
   hızlı ve bu bozuksa sonraki her ölçüm kirlidir.
2. **Faz 0 mesh düşürme** — üçgen sayısı düştü mü, gölgeleme detayı durdu mu.
   Yassılaştıysa normal map bağlı değil.
3. **★ Faz 0 boya yükseltme, prosedürel node YOK** — maske keskinleşmemeli.
   Keskinleştiyse alan bilgisi bir yerde erken düşürülüyor.
4. **★★★ Faz 0 boya yükseltme, prosedürel node VAR** — şimdi keskinleşmeli.
   Keskinleşmiyorsa `paint_resolution` yalnızca bir resize hedefi olarak
   bağlanmış, yani bütün Faz 0 sessizce boşa çıkmış demektir. **Bu, bu partinin
   en sinsi başarısızlığıdır:** her şey çalışıyor görünür, dosyalar büyür,
   görüntü biraz yumuşar, kimse şikâyet etmez.
5. **Scatter paritesi** — mesh 2048 vs 512, foliage dağılımı aynı mı.
6. **Heightmap import turu** — 16-bit bir dosya, hedeften küçük ve büyük iki
   çözünürlükte. Teraslama yoksa 8-bit yol gerçekten sökülmüş; aspect bozulmuyorsa
   stride yolu gerçekten gitmiş.
7. **Faz 1 çıkış testi** — düz renk, detay hayatta mı.
8. **Faz 3 `strength = 0` regresyonu** — eski sahne bit-bit aynı mı.
9. **Viewport ↔ render paritesi** — iki `uploadTerrainLayerMaterials` yolu aynı
   şeyi mi yüklüyor. Ayrışırsa belirti yok, yalnızca iki farklı görüntü.
10. **Faz 2 serileştirme turu** — kaydet/yükle sonrası gradient ve üç çözünürlük aynı mı.
10b. **★ Preset turu** — `alpine_snowline` uygula, `scale_y`'yi ikiye katla: kar
    çizgisi oransal olarak yerinde mi (remap doğru bağlanmış mı). Ardından 16
    stop'lu bir preset kaydet/yükle: stop **sayısı** korunuyor mu (`kMaxStops`
    mirası kalmamış mı). İkisi de sessiz kayıp üretir.
10c. **Mesh yeniden üretim profili** — Faz 0 sonrası mesh 1024 iken 0. maddedeki
    üç sayı tekrar alınır. Mesh fill düştüğü hâlde toplam düşmüyorsa darboğaz
    BVH'dedir; mesh optimizasyonuna devam etmek zaman kaybıdır.
11. **`terrain.export_color_map` + render karşılaştırması** — otomatik görsel
    doğrulamanın kapanış halkası.
12. **Faz 4 AO**, yalnızca 1-11 yeşilse.

## Açık sorular

- **`mesh_resolution` varsayılanı ne olmalı?** Bugünkü davranışla aynı kalması
  (= `field_resolution`) sürprizsizdir ama kimse kadranı bulmaz; otomatik bir
  düşük varsayılan (ör. `min(field, 1024)`) kazancı hemen verir ama var olan
  sahnelerin siluetini değiştirir. Kimlik turunu bozmamak için ilk sürümde eşit
  başlaması, panelde "önerilen: 1024" ipucu ile birlikte, muhtemelen doğrusu.
- **`erosionMapRGBA` alan çözünürlüğünde f32×4** — 8k'da 1 GB. Erozyon alanlarını
  f16'ya indirmek 512 MB kazandırır; kayıp ölçülmeli, varsayılmamalı.
- **Terrain normal map hangi çözünürlükte?** Alan çözünürlüğü doğal cevap ama
  boya çözünürlüğüne bağlamak splat kenarlarıyla hizalar. Muhtemelen alan;
  kararı ölçümle ver.
- Gradient stop editörü mü, LUT görüntüsü mü **birincil** yol olacak? İkisi de
  desteklenecek ama panelin varsayılanı ne olmalı — stop'lar keşfedilebilir,
  LUT import'u Gaea kullanıcısına tanıdık.
- Makro renk yalnızca albedo'yu mu modüle etmeli, yoksa roughness'a da zayıf bir
  bağ mı? (Islak/kumlu geçişler için cazip, ama ikinci bir gizli bağ.)
