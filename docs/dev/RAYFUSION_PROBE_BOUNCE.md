# RayFusion 1b-β — ilk tek sıçrama dilimi

Durum: **DOKU + ALPHA DİLİMİ YAZILDI, DERLENMEDİ**. Kapı hatası (RESIN_OBJ_SPACE)
canlı IPC ile bulundu ve düzeltildi; doku okuma ve alpha kesme bu partide eklendi.
Gas mod geçişi kullanıcı tekrarında sorunsuz; kalıcı hata olarak kabul edilmedi.
Physical Sky backend yön farkı kullanıcının kararıyla ertelendi. Bu parti bu
iki alana dokunmaz.

## Kapsam

Probe ışını opak üçgene isabet edince gerçek raster material ID'sinden diffuse
reflectance ve emission okunur. Bir cosine-weighted ortam ışını ve en fazla
bir point/directional ışık gölge ışını atılır. Işık, görünür ışık listesinden
eşit olasılıkla seçilir; katkı seçim olasılığına bölünür. Tek yüzey sıçraması
vardır; ikinci isabet probe cache'i okumaz ve tekrar sıçrama üretmez.
Ortam ışıması × diffuse reflectance, doğrudan ışık için Lambert 1/π,
metallic ve ortalama Fresnel enerji payı uygulanır. Sonuç eski probe texel
irradiance/π sözleşmesiyle yayınlanır; raster tüketici değiştirilmedi.

Geometri kaynağı BLAS'ın kullandığı flat raster vertex/material-ID buffer'larıdır.
TLAS `customIndex` sırası, gizli instance'lar ve cap elemesinden SONRA kaydedilir;
shader primitive ID ile aynı üçgenin verisini okur. Per-face Triangle facade
koleksiyonu kullanılmaz. Kaynaklar aynı viewport VkDevice'ına aittir.

Adres ABI'si: instance 32, malzeme 32, ışık 48 bayt. GLSL adresi `uvec2` olarak
okur; ek shaderInt64/scalarBlockLayout özelliği istemez.
[Khronos buffer-reference uvec2 sözleşmesi](https://github.khronos.org/Vulkan-Site/glslext/latest/glslext/ext/GLSL_EXT_buffer_reference_uvec2.html).

Malzeme/ışık içeriği, instance adresleri ve flat material-ID değişimleri producer
signature'a katılır. Değişimde eski probe'lar invalid edilir. Tablo büyümesi ve
yazımı öncesi mevcut in-flight drain kullanılır; descriptor her partide güncellenir.
Kaynak değişmediğinde aynı GPU tabloları yeniden yüklenmez.

## Kontrol ve gözlem

- UI: RayFusion development → **Single diffuse bounce (1b-beta)**.
- Python: `rt.rayfusion.set_probe_bounce(True)`.
- IPC: `rayfusion.set_probe_bounce {"enabled": true}`; Render capability.
- Üçü aynı `rtapi::setRayFusionProbeBounce` servisine gider. Bu bir geliştirme
  kontrolüdür; session-local, varsayılan kapalı, proje kaydına eklenmedi.
- `Trace probe rays` / `rayfusion.set_probe_producer {"traced":true}` da açık olmalı.
- IPC ve Python yalnız boolean kabul eder; IPC eksik/ek alanı reddeder.
  Hazır viewport/donanım yoksa `applied=false`. `applied` isteğin kabulüdür,
  GPU başarısı değildir; `active` son yayınlanan alanın durumudur.
- `rayfusion.probe_field` ve `rt.rayfusion.probe_field()` yeni alanları aynı
  isimlerle döndürür: `bounce_requested`, `bounce_ready`, `bounce_active`,
  `bounce_reason`, `bounce_signature`, `bounce_instances`, `bounce_materials`,
  `bounce_lights`, `bounce_supported_materials`, `bounce_unsupported_materials`,
  `bounce_rejected_textured`, `bounce_rejected_transparent`,
  `bounce_rejected_layered`, `bounce_rejected_flagged`,
  `bounce_rejected_flag_bits`, `bounce_unsupported_lights`,
  `bounce_hits`, `bounce_shaded_hits`, `bounce_alpha_tested`,
  `bounce_alpha_occluded`.

Probe başına en fazla 64 ana + 64 ortam + 64 gölge = **192 ışın**. Scheduler,
kalite preset'inin toplam ışın tavanını korumak için daha az probe seçer.
Alpha yolunda gerçek 64 ışın/probe kullanılır; önceki preset sayısı shader'ın
sabit 64 ışınıyla uyuşmuyordu. `core_status` bütçeleri kontrol çekirdeğinin
planlanan değerleridir; dispatch ölçümü değildir. `trace_ms` senkron dispatch
ve readback wall time'ıdır, yalnız GPU süresi değildir.

## Doku + alpha dilimi (2026-09-08)

Sıçrama isabet noktasında artık gerçek malzemeyi okuyor:

| Slot | Durum |
|---|---|
| albedo, emission, opacity, metallic, specular | **okunuyor** |
| roughness, normal, height | kasıtlı okunmuyor |
| transmission | dilim dışı |

Roughness okunmuyor çünkü bu dilimde yalnız diffuse lob var ve roughness
sonuca hiç girmiyor. Normal ve height okunmuyor çünkü probe texel'i bir yarım
küre ortalamasıdır; teğet uzayda bir tedirginlik onun frekansının altındadır.
Bunlar sessiz yaklaşıklık değil, yazılı kapsam kararlarıdır — geometrik normal
bu dilimin baştan beri ilan ettiği modeldir.

UV, hit tablosuna eklenen `uvs` adresinden (raster `uvBuffer`, aynı vertex
sırası) barycentric ile interpole edilir ve `closesthit.rchit`'teki
`applyMaterialUVTransform` ile **birebir aynı** dönüşümden geçer. Paketli
metallic kanalı ortak `pbr_texture_policy.glsl` ile çözülür. Ayrı bir kopya
yazmak, "üretici != tüketici" hata sınıfını davet etmek olurdu: sıçrama görünen
yüzeyin gösterdiğinden başka bir texel okur ve belirti yalnızca "GI rengi biraz
tuhaf" olurdu.

Enerji payı (metallic/specular Fresnel) CPU'dan shader'a taşındı. Eskiden
skalar metallic ile CPU'da katlanıyordu; metallic bir dokudan gelebildiği anda
o sabit sessizce yanlış olurdu.

Alpha maskesi gerçek: hem birincil probe ışınından hem gölge ışınından
`gl_RayFlagsOpaqueEXT` kaldırıldı ve adaylar opaklıkla eleniyor (bit 8 ile `.a`
veya `.r`, malzeme opaklığıyla çarpılıp 0.5 eşiğinden geçirilerek). Bedeli aday
döngüsüdür ve tahmin edilmiyor, **ölçülüyor**: `bounce_alpha_tested` /
`bounce_alpha_occluded`.

Mesh'in UV'si yoksa (`hit.uvs == 0`) doku okunmaz ve skalar renge düşülür.
Uydurma bir (0,0) koordinatı, dokunun sol alt köşesini bütün yüzeye boyamak
olurdu.

### Kabul ölçüsü: `bounce_shaded_hits`

Bu dilimin eklediği en önemli şey bir özellik değil, bir **ölçü aleti**.
Bugüne kadarki bütün sayılar CPU malzeme tablosuna aitti — *ne gölgelenebilir*.
`bounce_shaded_hits` GPU'da, son dispatch sırasında ölçülür: *ne gölgelendi*.

`bounce_hits > 0` iken `bounce_shaded_hits == 0`, her ışının bir occluder'a
çarptığı ve yayınlanan alanın bounce kapalıyken üretilenle **birebir aynı**
olduğu anlamına gelir. Malzeme sayıları bunu asla söyleyemez. `hit_fraction`
da söyleyemez: o yalnızca ışının geometriye değdiğini ölçer.

## Canlı ölçüm — 2026-09-08 akşamı (doku dilimi DERLENDİ)

Kullanıcı derledi ve masaya alpha dokulu bir zambak koydu. IPC ile ölçülen:

```
bounce_materials 40   supported 32   rejected: textured 0, transparent 8, layered 1
bounce_hits 216       bounce_shaded_hits 122      hit_fraction 0.105
bounce_alpha_tested 0 bounce_alpha_occluded 0
scene_as: 76 BLAS / 76 instance, 0 skipped, 0 hidden
bounce_instances 67 -> 76   (zambağın 9 alt mesh'i tabloya girdi)
```

**Doku dilimi çalışıyor.** 40 materyalin 32'si kapsamda; reddedilen 8'i cam,
1'i katmanlı. `rejected_textured` **0** — yani hiçbir materyal artık "dokusu
var" diye elenmiyor. 216 isabetin 122'si gölgelendi.

### Sayaçların canlı olduğu KANITLANDI

Bounce kapalıyken bir dispatch koştu ve `hits 216 / shaded 0` verdi; açıkken
aynı geometri `216 / 122`. Aynı isabet sayısı, farklı gölgeleme sayısı — sayaç
gerçekten o dispatch'i ölçüyor, tabloyu yankılamıyor.

### ⚠ ALPHA KOLU HÂLÂ DENENMEDİ, ve sebebi malzeme değil GEOMETRİ

Zambağın 4 materyalinde gerçek opacity dokusu var (2 taç yaprak + 2 yaprak,
`opacity` skaları 1.0), hepsi destekleniyor, hepsi izlenen sahnede. Buna rağmen
`alpha_tested = 0`: **hiçbir probe ışını çiçeğe değmiyor.** Çiçek eklendikten
sonra `hits` tam 216'da kaldı.

Aritmetik: çiçek (1.00, 1.26, 2.06) konumunda, ölçek 0.01. En yakın probe
(1.5, 1.5, 1.5), yani **0.79 birim** uzakta. ~0.5 m boyunda bir zambak bu
mesafeden ~%2 katı açı kaplar → probe başına 64 ışından **1-2 tanesi**; üstelik
zambak çoğunlukla ince sap ve seyrek yaprak, gerçek kapsama bunun çok altında.
Yani 0 isabet beklenen sonuçtur.

`alpha_tested = 0` bu partide "alpha bozuk" demek DEĞİL, "alpha hiç
çalıştırılmadı" demektir. İkisi ayrı sonuçtur ve karıştırılmamalıdır.

### ★★★ IPC ile okunan sayaç KARE DÖNGÜSÜNE BAĞLIDIR

`rayfusion.set_probe_bounce` isteği **uygulanıyor** (`bounce_requested`
anında dönüyor), ama `producer_signature`, `valid` ve `traced_publishes`
kıpırdamıyor: probe alanı yalnız viewport kare ürettiğinde servis ediliyor.
Uygulama boştayken toggle'dan hemen sonra sayaç okumak **önceki partiyi**
ölçer. Bu deponun kayıtlı kuralının bu yüzeydeki karşılığı: bir IPC yazısı ile
onun ölçüsü arasında viewport'un tiklemesi gerekir.

Pratik kural: toggle → viewport'ta bir şeye dokun (kamerayı kıpırdat) →
`traced_publishes` arttığını gör → **sonra** sayaç oku.

### Probe ızgarasının şekli ayrı bir soru

Izgara 4x2x4 hücre, 3 birim aralık, minimum hücre (-2,-1,-2). Probe merkezleri
x,z ∈ {-4.5, -1.5, 1.5, 4.5}, y ∈ {-1.5, 1.5}. Yani **Y'de yalnız iki katman
var ve -1.5 zeminin altında** — pratikte 32 probe'un 16'sı iş görüyor. Bir odayı
temsil etmek için bu ızgara seyrek. Bu bir hata değil, bir tasarım kararının
ölçülmüş sonucu; 1b sonrası kalite diliminin girdisi.

## ⚠ AÇIK: doğrudan aydınlatılmayan yüzeylerde YOĞUN MAVİ (2026-09-08)

**Takip:** Kullanıcı onayıyla tüketici görünürlüğü ve sekiz komşu harmanı
[1c diliminde yazıldı](RAYFUSION_PROBE_SAMPLING.md); derleme/görsel kabul
bekliyor. Aşağıdaki tek-probe ve okunmayan moment bulguları düzeltme öncesini
anlatır. Izgara yerleşimi ve hiç ölçüm olmayan bölgedeki sky fallback hâlâ açık.

Kullanıcı gözlemi: bazı yüzeylerde yoğun mavi atmosfer rengi, özellikle
doğrudan aydınlatılmayanlarda, ve RT'den çok farklı. **Normal değil**, ve
sebebi bu partinin doku dilimi DEĞİL — 1a'dan beri duran üç şeyin bileşimi.

### 1. Odanın DIŞINDAKİ probe, odanın İÇİNİ aydınlatıyor

`scene.raycast` ile y=1.5 katmanındaki 16 probe'un her birinden 5 yöne ışın
atıldı (yukarı, ±X, ±Z):

```
probe(x,z)      up     +X     -X     +Z     -Z
(-4.5,-4.5)      -      -      -      -      -     <- tamamen acik hava
(-4.5,-1.5)      -      -      -      -      -     <- tamamen acik hava
(-1.5,-4.5)      -      -      -      -      -     <- tamamen acik hava
( 1.5,-4.5)      -      -      -     3.5     -
( 1.5, 1.5)     1.5    1.8     -     2.8    2.5    <- gercekten cevrelenmis
( 1.5, 4.5)     1.5    0.1    0.4    0.1    0.2    <- gercekten cevrelenmis
```

Oda kabaca `x ∈ [0,5]`, `z ∈ [0,6]`, tavan `y = 3.0`. Izgara ise dünya
orijininde sabit `x,z ∈ [-6,6]`, `y ∈ [-3,3]`. Yani **16 kullanılabilir
probe'un 6'sı binanın dışında, açık havada** ve her yönde gökyüzü görüyor.
`hit_fraction`'ın %10.5 olması bundan; kapalı bir oda için bu sayı 1.0'a yakın
olmalıydı.

Tüketici hücreyi `floor(worldPos / 3.0)` ile seçiyor. `x ∈ [-3,0)` aralığındaki
bir iç yüzey, `x = -1.5`'teki probe'u okur — **duvarın dışındaki, gökyüzüne
bakan probe'u**. Sonuç: saf gökyüzü ışıması × albedo = yoğun mavi.

### 2. Görünürlük verisi ÜRETİLİYOR ama TÜKETİLMİYOR

`rayfusion_probe_trace_beta.comp` her texel için `meanDistance` ve
`meanSquare` yazıyor — bu bir Chebyshev/varyans görünürlük çiftidir ve tam
olarak "bu probe duvarın öbür tarafında mı" sorusunu yanıtlamak için üretilir.
`probe_field.glsl` içinde `packet.distance` **hiçbir yerde okunmuyor.**

Ölçü aleti çalışıyor, kimse bakmıyor. Bu deponun tekrar eden hata sınıfı.

### 3. Harmanlama yok — hata HÜCRE KUANTALI

Tek probe, en yakın hücre, trilineer harmanlama yok. Bu yüzden hata
"her yer biraz mavi" değil, **"bazı yüzeyler çok mavi"** biçiminde görünür ve
sınırlar 3 birimin katlarında keskin durur. Kullanıcının tarifi bununla birebir
uyuşuyor.

### ✔ Kullanıcı gözlemi teşhisi DOĞRULADI (kontrol deneyi)

Masa örtüsünün masadan sarkan yüzünde `Trace probe rays` açılınca mavilik
temizleniyor. Hücre matematiği bunu birebir açıklıyor: masa (1.42, 1.04, 1.89),
sarkan kısım y ~ 0.7 → `floor(pos/3)` = (0,0,0) → probe merkezi
**(1.5, 1.5, 1.5)** — raycast'te gerçekten çevrelenmiş çıkan iki probe'dan biri.

Yani ayrım tam olarak şu: çevrelenmiş probe okuyan yüzey traced üreticiyle
düzeliyor, açık havadaki probe okuyan yüzey düzelmiyor. **Üretici sağlam;
arıza, hangi probe'un okunduğunda.** Bu, düzeltmeyi ızgara/oklüzyon tarafına
sabitler ve üretici tarafında arama yapılmasına gerek bırakmaz.

### Bu parti bunu KÖTÜLEŞTİRMEDİ

1b-α'da isabet eden yön siyah katkı veriyordu: çevrelenmiş probe karanlık,
açık havadaki probe gök mavisi — yani komşu hücreler arasındaki karşıtlık
**daha yüksekti**. Doku dilimi çevrelenmiş probe'ları aydınlattığı için farkı
bir miktar azaltır. Kök neden ızgara yerleşimi + eksik görünürlük testidir.

### RT ile farkın sebebi

RT piksel başına gerçek, oklüzyonlu GI çözer. Buradaki ambient tek bir
oklüzyonsuz probe okumasıdır. İkisinin ayrışması beklenen sonuçtur; kapanması
için tüketicinin görünürlüğü hesaba katması gerekir.

### Çözüm yolu — ve neden bu bir KARAR

Üçü de tüketici tarafında (`probe_field.glsl`):

1. **Chebyshev görünürlük testi**: `packet.distance.xy` momentleriyle, probe ile
   yüzey arasındaki mesafeyi karşılaştır; probe duvarın arkasındaysa ağırlığını
   düşür. Veri zaten yazılıyor, tek satırlık bir tüketici eksik.
2. **8 probe üzerinden trilineer harman**, görünürlük ve normal yönü ile
   ağırlıklandırılmış. Hücre kuantalı keskin sınırları bu kaldırır.
3. Izgara yoğunluğu/yerleşimi ayrı soru: 3 birim aralıkla bir odaya 16 probe
   düşüyor ve Y'de tek kullanışlı katman var.

★★★ **Bu bir detay değil, kayıtlı bir kararın bozulması.** `probe_field.glsl`
başlığı tüketicinin KASITLI olarak dondurulduğunu söylüyor: üretici değişirken
tüketici sabit kalsın ki görüntüdeki fark yalnızca üreticiden gelsin. Tüketiciyi
değiştirmek o ölçüm zeminini kaldırır. Yapılacaksa bilerek yapılmalı ve
1b-α/1b-β karşılaştırmaları o noktadan sonra yeniden temellendirilmelidir.

## Açık kapsam ve kabul sınırı

Dokusuz opak diffuse/emission, geometric normal, ortam ve en fazla 64 görünür
point/directional ışık. Desteklenmeyen malzemeler **opaque occluder** kalır,
uydurma albedo ile bounce üretmez. Sayılar CPU malzeme/ışık tablosuna aittir;
isabet edilen desteklenmeyen yüzey sayısı değildir.

Texture/normal map/material graph parity, alpha/cam, area/spot, analitik
Physical Sky güneşi, SDF/gas/VDB/hair katılımı tamamlanmadı. Fiziksel güneş
ortam prefilter'ından çıkarıldığı için bu dilim analitik güneş sıçraması iddia
etmez. 200 birim sonlu görünürlük aralığı ve 0.005 normal bias kullanılır;
ince/çok büyük sahneler ayrı kabul testidir. Sabit örnekler tek güncellemede
iz bırakabilir; jitter/çoklu temporal güncelleme sonraki kalite dilimidir.
Skinning/sculpt AS refit doğrulaması önceki AS altyapısının açık kabul borcudur.
Bu ilk dilim, 1b-β'nin tüm materyallerle tamamlandığı anlamına gelmez.

## Derleme ve test

Kaynak kontrolleri:

```
python scripts/audit_rayfusion_bounce.py
python scripts/audit_ipc_capabilities.py
```

Kullanıcı derlemesi:

1. Shader derlemesi yeni `rayfusion_probe_trace_beta.comp` dosyasından
   **rayfusion_probe_trace_beta.spv** üretmeli. Farklı dosya adı eski alpha
   SPV'nin yanlışlıkla beta başarı sayılmasını önler. Eksikse producer reason
   eksik dosyayı bildirir ve sky-bake fallback sürer.
2. C++ projesini derle; `RayFusionBounce.cpp` projeye eklendi. Bu partide RT
   path-tracer payload'ı değişmedi; yalnız probe compute zinciri değişti.
3. Dokusuz kırmızı duvar + nötr zemin + point/directional ışık veya HDRI ile
   sabit kamera kur. Trace açıkken bounce kapalı/açık görüntüleri karşılaştır.
   Duvarı gören zeminde sınırlı kırmızı dolaylı katkı beklenir.
4. Işığı kapat/aç, rengini/şiddetini değiştir; duvar rengini değiştir ve farklı
   materyal ata. Signature değişmeli ve eski ışık alanı yenilenmeli.
5. Araya opak engel ekle: gölge ışını katkıyı kesmeli. Siyah albedo + emission
   kapalı yüzey bounce üretmemeli; emission açık yüzey ışık yaymalı.
6. İkinci obje, shared mesh, taşı, gizle, sil/undo ve tümünü sil/yeniden ekle:
   customIndex/material eşleşmesi bozulmamalı, TDR/validation hatası olmamalı.
7. UI/Python/IPC aynı requested/active sonucunu vermeli; `enabled:1`, eksik
   enabled ve ek parametre IPC'de hata olmalı. Kapalı traced ile bounce isteği
   alınabilir ama aktif olarak raporlanmamalı.
8. Performance/balanced/quality/full için toplam planlanan ışın sınırı,
   yakınsama ve `trace_ms` kaydedilmeli. GPU'da görsel/enerji doğrulaması geçmeden
   bu dilim DOĞRULANDI veya ÖLÇÜLDÜ sayılmaz.

## Ölçüm bulgusu (2026-09-07) — "31 materyal desteklenmiyor"

`old_room.rtp` sahnesinde panel **31 materyalin 31'ini** desteklenmiyor
gösterdi. Canlı IPC ile sahne tablosu okundu (`material.list`,
`material.textures`, `material.get_param`):

| Sınıf | Adet |
|---|---|
| Dokulu (base_color/roughness/metallic/normal/specular) | 23 |
| Dokusuz ama `transmission = 1` (cam) | 7 |
| Dokusuz, opak, `transmission = 0` (`default_material`) | **1** |

Yani sayının **30** olması gerekirdi. Sahne gerçekten bu dilimin dışında —
"31 desteklenmiyor" bir arıza değil, neredeyse doğru bir cevap — ama tam
olarak desteklenmesi gereken tek materyal kadar sapma var. Kullanıcının
tek materyalli (default solid) sahnede de aynı satırı görmesi aynı yöne
işaret ediyor.

Canlı deney: dokusuz `1_glass` materyalinin `transmission` değeri IPC'den
0'a çekildi. Bu materyal o anda kapının okunabilen **her** maddesini
sağlıyordu (doku yok, `opacity = 1`, subsurface/translucency/clearcoat = 0),
ama `bounce_unsupported_materials` 31'de kaldı. Değer geri alındı.

Okunamayan tek madde `m.flags == 0u`. `VkGpuMaterial::flags` yalnız malzeme
özelliği taşımıyor: 8-11. bitler doku kanal seçicileri, 12-15. bitler
metallic/roughness paketli kanal override'ı, 16+ terrain/water/volume. Yani
panelde **düpedüz sade görünen** bir materyal burada reddedilebilir ve bunun
hiçbir belirtisi yoktur — bu deponun bilinen hata sınıfı: bir bayrak alanının
iki farklı anlamı.

Bu partide kapı **davranış olarak değiştirilmedi** (NaN dahil aynı sonucu
verir); sadece hangi maddenin reddettiği sayılır hale getirildi:
`bounce_rejected_textured|transparent|layered|flagged` ve reddeden bayrakların
OR'u olan `bounce_rejected_flag_bits`. Tek başına "31 desteklenmiyor" **ölçüm
değildir**: baştan sona dokulu bir sahne ile her şeyi reddeden bir kapı aynı
sayıyı basar. Ayrıca `bounce_supported_materials` 0 ise panel ve `bounce_reason`
bunu açıkça söyler — çünkü o durumda üretilen görüntü 1b-α ile **birebir
aynıdır** ve sessiz kalırsa arıza gibi okunur.

### KÖK NEDEN (aynı gün, ölçüldü)

Yeni sayaç tek turda cevabı verdi: `bounce_rejected_flag_bits = 0x200000`,
yani **bit 21 = `VK_MAT_FLAG_RESIN_OBJ_SPACE`**, ve `rejected_flagged` =
tablodaki materyal sayısı. Projedeki 31 materyalin 31'inde
`resinObjectSpace: true` — bu bayrak **varsayılan olarak açık**.

`m.flags == 0u` maddesi bu yüzden **hiçbir sahnede, hiçbir materyal için**
geçmiyordu; 1b-β şimdiye kadar tek bir piksel bile değiştirmemişti ve
"ışıkları görmüyor" belirtisi de buradan geliyordu (shader
`m.diffuse.w < 0.5` görünce `vec3(0)` döndürür — ışık kodu hiç çalışmaz).

`flags` bir özellik kümesi değil, üç ayrı anlamın paketlendiği bir kelime:
bit 8-15 doku kanal çözme/seçme bitleri, bit 21 reçine iç hacminin
**koordinat uzayı**, bit 16+ gerçek malzeme sınıfı. Kapı artık "sıfır değil"
diye çıkarım yapmıyor, bu dilimin gölgelemesini gerçekten değiştirecek
bitleri **adıyla** sayıyor: TERRAIN, WATER, WATER_FFT_READY, BUBBLE,
MARBLE_VOLUME, WATER_LAKE, WATER_RIVER, VOLUME. Kanal seçici bitleri gerekli
değil, çünkü dokulu materyali zaten `textured` maddesi eliyor.

Bir sonraki dilimin ölçüm borcu: bu sayılar hâlâ CPU malzeme tablosuna ait.
"Kaç probe ışını desteklenen bir malzemeye çarptı" sorusunu ancak shader
tarafında bir sayaç yanıtlar; `hit_fraction` yalnız geometriye çarpmayı ölçer.
