# Terrain Road and Path Network Roadmap

> **Durum:** AKTİF — Faz 0, 1, 2 ve 3 tamam; Faz 4 (yol yüzey mesh'i) ve Faz 5
> (hydrology crossing) **yazıldı, DERLENMEDİ** (2026-08-31). Faz 6 (otomatik
> routing) ve Faz 7 (settlement) başlamadı ve ayrı bir proje ölçeğindedir.
>
> ★★★ Açık sözleşme borcu **kapatıldı**: kazılan yolun akarsu yatağı sanılması
> artık bir kadranla değil, **kesitle ve bir sınıflandırma dışlamasıyla**
> cevaplanıyor — aşağıdaki bölüme bak. Altıncı canonical alan
> `infrastructure.ditch` eklendi.
>
> [PROFILE_SPLINE_NEXT_PHASE_ROADMAP.md](PROFILE_SPLINE_NEXT_PHASE_ROADMAP.md)
> Faz 3.6'daki genel amaçlı `Curve to Mask` + `Road Carve` altyapısının
> tüketicisidir. Ayrı bir spline temsili veya ikinci bir curve-to-terrain
> çözüm yolu kurmaz.

## Ürün kararı

Yol ve patikalar render mesh olmadan önce terrain analiz ürünüdür:

```text
SplineObject / DataType::Curve
        -> Curve to Mask                 (Faz 3.6 sahibi)
        -> Road Carve                    (Faz 3.6 sahibi)
        -> graded height + ölçüm pinleri
        -> Faz B publication
        -> mevcut renk/blend graph'ı + foliage + opsiyonel yol mesh'i
```

Manuel ve gelecekte otomatik üretilen yollar aynı `SplineObject` / `Curve`
otoritesine ve aynı `Road Carve` hesabına girmelidir. Otomatik sistem yalnızca
güzergâh seçer; terrain deformasyonu, maskeler ve tüketici semantiği için ikinci
bir uygulama yazılmaz.

Canonical scene geometrisi flat `TriangleMesh` / DNA SoA'dır. Yol yüzey mesh'i
üretilirse bu yoldan yayınlanır; per-face `Triangle` facade koleksiyonları
geometri kaynağı veya otorite olamaz.

## Mevcut altyapının sahipliği

Bu plan şu parçaları yeniden tasarlamaz:

- `SplineObject` kontrol noktası, handle, gizmo, undo, serialization ve yaşam
  döngüsünün tek otoritesidir.
- `DataType::Curve` graph içindeki curve taşıma sözleşmesidir.
- `spline.*` script/IPC işlemleri curve geometrisini düzenleyen tek yüzeydir.
- Faz 3.6 `Curve to Mask` ile curve -> terrain field dönüşümünün sahibidir.
- Faz 3.6 `Road Carve` ile mask/profile -> graded height dönüşümünün sahibidir.
- `Publish Field`, yazara ait bestelenmiş maskeleri adlandırmanın mevcut yoludur.

Bu nedenle eski taslaktaki bağımsız `RoadNetworkData` kontrol noktaları,
`TerrainRoadNetworkCore` rasterizer'ı, ayrı road undo/gizmo/serialization hattı
ve `terrain.road.set_points` operasyonu kaldırılmıştır.

River authoring burada yeniden kullanılabilir bir spline -> field altyapısı
sayılmaz. `RiverSpline` terrain'i okuyarak görsel spline üretir;
`RiverSplineOutputNode` field -> spline yönündedir. Yolun gereken
**non-destructive, graph'ta değerlendirilen, alan yayınlayan** spline -> field +
height yolu Faz 3.6'da ilk kez kurulur ve sıfırdan solver işidir.

★ Düzeltme (2026-08-30): spline -> yükseklik yolunun *imperatif* bir uygulaması
zaten var — `terrain.carve_river` (`RtApiTerrain.cpp:1456`) nehir spline'ını
örnekleyip `TerrainManager::carveRiverBed` / `carveRiverBedNatural` ile doğrudan
heightmap'e yazar ve `TerrainSnapshot` ile geri alır. `Road Carve` canlıya
geçtiğinde aynı işin iki uygulaması olacaktır. Faz 3.6 kapanmadan bu üçünden
biri seçilmelidir: carve'ı `Road Carve`'ın bir profiline dönüştür (tercih
edilen), açıkça "yıkıcı tek seferlik authoring aracı" diye etiketleyip graph
yolundan ayır, ya da kural 5 gereği sök. Karar verilmemesi üçüncü sessiz yol
demektir. Ayrıntı ve göç planı:
[PROFILE_SPLINE_NEXT_PHASE_ROADMAP.md](PROFILE_SPLINE_NEXT_PHASE_ROADMAP.md)
Faz 3.7.

## Road üst katmanının veri modeli

Yol katmanı curve geometrisini kopyalamaz. Yalnızca mevcut `SplineObject`'e
referans veren yol semantiğini taşır:

```text
RoadAssignment
  spline_object_id
  profile_id
  enabled
  priority
  crossing_mode   Auto | Terrain | Bridge | Ford | Tunnel
```

Birden çok spline ağ olarak yorumlandığında topoloji tablosu yine spline ID'lerine
referans verir; kontrol noktalarını tekrar tutmaz. Silinen/bulunamayan referans
açık diagnostic üretir, sessizce boş maske üretmez.

### Profil sözleşmesi

`RoadProfile`, curve geometrisinden bağımsız ve veri güdümlüdür:

- core ve shoulder genişliği
- ditch genişliği ve derinliği
- crown/camber
- maksimum longitudinal grade
- maksimum cut/fill eğimi ve miktarı
- grading blend mesafesi ve terrain conformity
- foliage hard-exclusion ve density-falloff mesafesi
- opsiyonel yüzey mesh ayarları

İlk profiller: `Footpath`, `DirtRoad`, `MainRoad`. Patika ve ana yol ayrı ölçüm
türleri değildir; aynı road ölçümünün farklı profilleridir.

## Faz A / Faz B zorunlu sözleşmesi

Terrain graph değerlendirmesi iki fazlıdır:

- **Faz A:** height zinciri, background thread. `Road Carve` graded height ve
  ilişkili alanları burada bir kez çözer.
- **Faz B:** publication sink'leri, main thread. Başlangıçta
  `terrain->analysisFields.clear()` çağrılır.

Bu ayrım ilk implementasyondan itibaren bağlayıcıdır:

1. `Road Carve` solver'ı Faz A'da yalnızca bir kez çalışır.
2. Graded height ve maskeler aynı immutable `RoadCarveResult` snapshot'ında
   tutulur; snapshot evaluation/input revision ile etiketlenir.
3. Faz B publisher aynı snapshot'ı yayınlar; solver'ı yeniden çalıştırmaz.
4. `analysisFields.clear()` publication set'ini temizler, Faz A snapshot'ını
   yok etmez.
5. Faz B'de geçerli snapshot yoksa açık hata üretir; eski alanı veya nötr sıfırı
   başarı gibi sunmaz.
6. Height ile maskeler farklı terrain varsayımlarından üretilemez.

Graph'ın pull cache'i kullanılabilir, fakat node-owned sonucu Faz A/Faz B
sınırında taşıyan açık snapshot/revision sözleşmesi yine gereklidir.

## ✅ ÇÖZÜLDÜ: kesme/dolgu zarfı (2026-08-30, canlı doğrulandı)

İlk sürüm max-grade sınırlayıcısını uyguluyordu ama **sapmayı hiçbir şey
sınırlamıyordu**: %12 eğimli bir yol bir dağa yeterince hızlı tırmanamaz, profil
giriş yüksekliğinde kalır, ve rasterizer bunun gerektirdiğini sadakatle inşa
eder — zirvede kanyon, vadide dağ yüksekliğinde dolgu. Matematik doğruydu;
**eksik olan kısıttı.** Bu belgenin "must not silently generate extreme walls"
cümlesi yazılıydı ve uygulanmamıştı.

`RoadCarveSettings`: `maxCutMeters` (8 m), `maxFillMeters` (6 m). Çözücü grade
sınırlayıcı ile zarf kısıtını **dönüşümlü** uygular (8 geçiş; ikisi de zemine
çektiği için yakınsar). Çatıştıklarında **zarf kazanır** — biraz fazla dik bir yol
yoldur, havada asılı duran bir yol arızanın kendisidir — ve o durum
`gradeExceededSamples` olarak sayılıp panelde raporlanır
(`peakCut/FillMeters`, `envelopeClampedSamples` ile birlikte).

★★ Yeni bir kadran eklerken **cache anahtarına da ekle**: iki değer
`TerrainRoadCarveNode` key hash'ine eklenmeseydi kadranı çevirmek hiçbir şey
yapmaz ve "düzeltme çalışmıyor" gibi görünürdü.

Ölçüm node'a ikinci bir sayı yüzeyi olarak konmadı: `infrastructure.cut` /
`fill` yayınlandığında `terrain.field_stats` min/max verir.

## Road Carve hesabı

```text
route_height  = spline boyunca grade-limitli longitudinal profil
target_height = route_height + cross_section(lateral_distance)
graded_height = blend(input_height, target_height, grading_support)
cut           = max(input_height - graded_height, 0)
fill          = max(graded_height - input_height, 0)
```

Longitudinal profil field-scale gürültüyü takip etmemeli; `Max Grade` ve profil
sınırlarını uygulamalıdır. Limit aşımı diagnostic üretir. Aşırı duvar üretmek,
sessizce kırpmak veya spline'ı olduğu gibi height'a damgalamak kabul edilmez.

## Kalıcı ölçüm alanları

İlk sürüm yalnızca çıkış kriterlerinin kullandığı beş canonical alanı rezerve
eder:

| Stable field | Ölçüm |
|---|---|
| `infrastructure.road_core` | Yol/patika yüzey desteği |
| `infrastructure.shoulder` | Core dışındaki shoulder/geçiş desteği |
| `infrastructure.cut` | Grading ile çıkarılan terrain miktarı/desteği |
| `infrastructure.fill` | Grading ile eklenen terrain miktarı/desteği |
| `infrastructure.foliage_exclusion` | Sert foliage yerleşim yasağı |
| `infrastructure.ditch` | Hendek desteği / drenaj iletkenliği |

`path` ve `road` için ayrı canonical alan yoktur; ayrım `profile_id`'dir.

★★★ **Hendek altıncı canonical alan oldu (2026-08-31)** ve bu bir kapsam
genişlemesi değil, aşağıdaki AÇIK maddenin kapatılmasının **zorunlu yarısı**:
`road_core` kanal sınıflandırmasından dışlanıp hendek yayınlanmazsa sonuç makul
görünür ama drenaj ağı sessizce kopar. İkisi tek partide gelmek zorundaydı.

Distance, tangent, wear ve falloff önce node çıkış pini veya diagnostic olarak
kalır. Gerçek tüketici kanıtlanırsa contract değerlendirmesiyle canonical
ölçüme yükseltilir. Yazar bunları adlandırmak isterse `Publish Field` kullanır.

## Materyal ve foliage kapsamı

SatMap output `analysisFields` tüketmez; yalnızca macro color ve strength alır.
Bu nedenle ayrı road SatMap sistemi, preset kütüphanesi veya maskeleri yeniden
yorumlayan SatMap kodu yazılmaz.

`Road Carve` core/shoulder/cut/fill alanlarını çıkış pinleri olarak verir. Mevcut
renk, ramp ve blend node'ları bu maskelerden macro color üretir. Hazır kullanım
yalnızca mevcut node'ları bağlayan recipe/setup olabilir; yeni materyal hesap
yolu değildir.

Foliage `infrastructure.foliage_exclusion` alanını mevcut tek exclusion slotundan
okur. Çoklu dışlama yeni slotlarla çözülmez; Math -> `Publish Field` ile
bestelenir. Density falloff ilk sürümde canonical değildir; Road Carve çıkışı
Publish Field üzerinden foliage density maskesine bağlanabilir.

## Değerlendirme sırası

```text
macro terrain
-> watershed / lake / river analysis
-> Curve to Mask
-> Road Carve (Faz A: height + tek snapshot)
-> surface relief/detail
-> Faz B: beş canonical alanı aynı snapshot'tan yayınla
-> mevcut material/SatMap renk graph'ı
-> foliage/scatter
-> opsiyonel road surface mesh
```

Otomatik routing hydrology'yi okur. Manuel road carve da accepted lake veya
river kanalını ancak açık crossing kararıyla etkileyebilir. Bridge span boyunca
road sürekliliği korunur fakat terrain grading kapatılır.

## ✅ ÇÖZÜLDÜ: kazılan yol AKARSU YATAĞI sanılıyordu (2026-08-31)

Kullanıcı raporu (2026-08-30): "yol ile kazılan alanlar SatMap veya flow
çözücülerinde akarsu yatağı olarak algılanabiliyor."

**Ve bu tam olarak bir hata değildi — eksik olan bir OTORİTE BEYANIYDI.** Kazılmış
bir yol doğrusal bir çukurdur; akış çözücüsünün görüşünden mükemmel bir kanaldır.
Fiziksel olarak da yanlış değil: yarılmış bir yol gerçekten su toplar — hendek
ve menfez tam bu yüzden vardır. Sistemin bilmediği şey, o çukurun **arazi mi
altyapı mı** olduğuydu. Yükseklik alanı bunu söyleyemez.

★★★ **Kök, düşünülenden bir kat daha aşağıdaydı: kesitin kendisi yalandı.**
`RoadCarveSettings` içinde ne crown ne hendek vardı — profil sözleşmesi ikisini
de vaat ediyor olmasına rağmen. Yani üretilen şey *bir yola benzeyen çukur*
değildi; **düz tabanlı doğrusal bir hendekti**, ve akış çözücüsü ona doğru
baktı. Bir bayrakla "burası yol" demek, yanlış geometriyi doğru etiketlemek
olurdu.

**Uygulanan çözüm iki parçalı ve ikisi de zorunlu:**

1. **Geometri (asıl otorite).** `crownMeters`, `ditchWidthMeters`,
   `ditchDepthMeters` kesite girdi. Yüzey artık dışbükey — suyu yana atar — ve
   omzun dışındaki hendek onu aşağı taşır. **Yol iter, hendek taşır**, gerçek
   yol mühendisliğinin aynı sebeple yaptığı şey. Bu bir kadran değil, çukurun
   ne olduğunu değiştiren bir şekildir.
2. **Sınıflandırma.** `TerrainV2.RiverNetwork`'e eklenen `Channel Exclusion`
   (opsiyonel, sona eklendi) pinindeki hücreler **kanal olarak ADLANDIRILMAZ**.
   `Road Network`'ün `Road Core` çıkışı buraya bağlanır.

★★★ **Sinsi başarısızlık yapısal olarak imkânsız kılındı.** Dışlama yalnızca
sınıflandırmaya dokunur: akış yönü, ebeveynlik ve akümülasyon **hiç
değişmez** (`TerrainRiverNetworkCore.cpp`, `result.active` satırı). Yani
dışlanan bir hücrenin aldığı su yine aynı aşağı havzaya gider. Routing'i kesmek
tam olarak yol haritasının uyardığı makul görünen arıza olurdu: yolda nehir yok,
ve aşağı havza sessizce eksik besleniyor.

Altıncı canonical alan `infrastructure.ditch` yayınlanır ve
`RoadFieldsOutput`'ta **zorunlu** bir pindir — hendeği yayınlamadan yolu
dışlamak mümkün olmasın diye.

Erosion tarafı için ayrı kod yazılmadı, çünkü gerek yoktu: `Hydraulic Erosion`
zaten `Area` ve `Hardness` maskesi alıyor. Tarif: `infrastructure.road_core`'u
`Hardness`'a (veya tersini `Area`'ya) bağla — yol oyulmaz.

★ Ölçüm hâlâ yapılabilir ve yapılmalı: `terrain.field_stats` ile
`hydrology.accumulation`ı carve ÖNCE ve SONRA oku. Bu partide beklenen sonuç
**değişmemesi**; değişiyorsa sonraki bir geçiş yüksekliği yeniden okuyordur ve
o tüketici de `Channel Exclusion`a bağlanmalıdır.

## ✅ ÇÖZÜLDÜ: Curve Input tek spline taşıyordu (Faz 2 ile, 2026-08-30)

Kullanıcı raporu (2026-08-30). Doğru: `TerrainCurveInputNode` tek bir
`splineObject` adı tutuyor ve `CurveValue` = `shared_ptr<CurveNodeData>`,
`CurveNodeData` da **tek** bir `BezierSpline` taşıyor. On yol segmenti = on
Curve Input + on Curve to Mask.

`CurveNodeData`'yı çoklu yapmak **çözüm değil**: o tip Geometry graph'ıyla
paylaşılıyor ve orada "tek eğri" doğru modeldir — bir sweep path'i bir eğridir,
bir Curve Deform bir eğri okur. Paylaşılan tipi bir tüketicinin ihtiyacı için
genişletmek, bu deponun "aynı isim ≠ aynı iş" dersinin tekrarı olur.

**Önerilen: `Curve Set to Mask` node'u.** Bir seçim kuralı alır (açık çoklu
seçim veya ad öneki) ve eşleşen bütün snapshot'ları **mevcut**
`TerrainCurveMask` rasterizer'ıyla tek maskeye yazar. Yeni pin tipi yok, ripple
yok, node patlaması yok. `Curve Input → Curve to Mask` tek-eğri yolu, eğrinin
kendisine ihtiyaç duyan tüketiciler için aynen kalır.

★ Karşı argüman ve cevabı: "iki yoldan maske üretmek" gibi görünür, ama iki
**uygulama** değil — ikisi de aynı rasterizer'ı çağırır; biri pinden, diğeri
seçim kuralından beslenir. Alternatif (yeni bir curve-set payload tipi) başka
bir graph'la paylaşılan tipi değiştirmek demek, ve o gerçekten iki yol yaratır.

★ Ad öneki seçim kuralı seçilirse: eşleşme **sıfır spline** buluyorsa bu bir
**diagnostic** olmalı, sessiz boş maske değil. Boş maske "yol yok" ile "adı
yanlış yazdım"ı ayırt edilemez kılar.

**Uygulanan çözüm önerilenden farklı ve daha iyi.** `Curve Set to Mask`
yazılmadı, çünkü kök Curve Input'un tek isim taşıması değildi: **graph'ta hiçbir
şey bir eğrinin YOL olduğunu bilmiyordu.** Bir ad öneki kuralı bunu bir isim
uzlaşımına çevirirdi — taşıyan bir etiket, yine.

`RoadNetworkRegistry` (Faz 2) bu soruyu cevaplar, ve `TerrainV2.RoadNetwork`
düğümünün **Curve pini yoktur**: kaydı okur, terrain context'in zaten aldığı
immutable snapshot'ları çeker, hepsini **tek** `solveRoadNetworkCarve` çağrısıyla
çözer. N yol = bir düğüm, bir solve, bir snapshot.

★ Yan kazanç: bütün yollar **tek** nearest-distance geçişi paylaşır, yani
örtüşmeler yakınlıkla çözülür — kavşak iki kez kazılmaz ve dar bir patika geniş bir
yolun içinden oluk geçirmez. Kazanan yolun profili piksele **birlikte taşınır**;
tek bir paylaşılan settings bloğundan okumak patikaya ana yolun omzunu giydirirdi.

`CurveNodeData` değiştirilmedi. Tek-eğri `Curve Input → Curve to Mask` yolu,
eğrinin kendisine ihtiyacı olan tüketiciler için aynen duruyor.

## Fazlar

### Faz 0 — Ölçüm ve sözleşme kapısı

Kodlamadan önce curve sahipliği, profil birimleri, beş field adı, error modeli
ve Faz A/B snapshot sözleşmesi dondurulur.

> **2026-08-30 canlı durum:** `terrain.field_stats` aynı rtapi core üzerinden
> Python ve IPC'ye indi; descriptor `Read` capability ile keşfedildi. Geçici
> 64x64 procedural fixture'da 4096 finite / 0 non-finite örnek, sekiz kovası
> toplamda 4096 eden histogram, üç nokta örneği ve missing-field / bad-bin /
> out-of-range refusal yolları IPC'den geçti. Probe nesnesi `finally` temizliği
> sonrası sahnede kalmadı. Fixture kurulumu ayrıca mevcut
> `snowy_mountain_valley + biome_temperate` birleşiminde
> `Surface Relief.Height -> Soil Depth.Height: would create a cycle` wiring
> fault'unu ölçtü; bu field-stats arızası değildir, ayrı preset graph borcudur.

`TerrainV2.RoadCarve` da bu dilime eklendi. `RoadCarveResult` tek solve
revision'inda non-destructive Height, Road Core, Shoulder, Cut (metre), Fill
(metre) ve Foliage Exclusion uretir. Boyuna profil `maxGradePercent` ile iki
yonlu sinirlanir. Ayri output pull'lari ayni node snapshot cache'ini kullandigi
icin publisher sayisi solver cagri sayisini artirmaz. Canli IPC fixture'i
`scripts/test/rt_test_terrain_road_carve.py` icindedir.

Faz B uygulama notu (2026-08-30): `TerrainV2.RoadFieldsOutput` beş field pini
ile `Snapshot Revision` pininin doğrudan aynı `TerrainV2.RoadCarve` düğümünden
gelmesini zorunlu tutar. Boyut, semantic, unit, finite değer ve mask 0..1
doğrulamalarının tamamı geçmeden hiçbir canonical alan yazılmaz. Başarılı
commit kopyasız olarak `infrastructure.road_core`, `infrastructure.shoulder`,
`infrastructure.cut`, `infrastructure.fill` ve
`infrastructure.foliage_exclusion` adlarını yayınlar. Yeni foliage layer'ları ve
standart biome foliage kurulumu bu son alanı mevcut hard-exclusion slotuna
bağlar; alan yayınlanmayan terrain'lerde eski nötr davranış korunur.

Deliverable'lar:

- `RoadAssignment`, `RoadProfile`, `RoadCarveResult` contract'ları
- `SplineObject` ID yaşam döngüsü ve dangling-reference davranışı
- flat, side-slope, ridge, diagonal ve S-bend sentetik fixture'ları
- `terrain.field_stats` script + IPC ölçüm ucu
- method descriptor, validation/error semantics ve statik API kontrolleri

`terrain.field_stats` en az şunları döndürmelidir:

- terrain/field adı, width, height ve channel sayısı
- min, max, mean, nonzero count/fraction
- sabit alan (`min == max`) tanısı
- istenebilir histogram/bin sayıları
- UV veya field koordinatında toplu nokta örnekleri
- bulunamayan alan ve aralık dışı sample için açık hata

Çıkış kriteri:

- Fixture beklentileri sayısal olarak tanımlıdır.
- `terrain.field_stats` mevcut yayınlanmış bir alanı gözle bakmadan ölçer.
- UI/Python/IPC aynı field ölçüm core'unu kullanır.
- Yeni kontrol noktası deposu veya spline edit API'si olmadığı statik audit ile
  doğrulanır.

### Faz 1 — Faz 3.6 ortak altyapısı: Curve to Mask + Road Carve

Bu fazın implementasyon sahibi
[PROFILE_SPLINE_NEXT_PHASE_ROADMAP.md](PROFILE_SPLINE_NEXT_PHASE_ROADMAP.md)
Faz 3.6'dır. Bu belge yalnızca road kabul kriterlerini tanımlar.

Uygulama notu (2026-08-30): `TerrainV2.CurveInput` ve
`TerrainV2.CurveToMask` eklendi. Scene spline'ları main thread'de immutable
snapshot olarak alınır; worker graph scene'yi dereference etmez. Stroke genişliği
metredir, kontrol noktası `userData1` ile çarpılabilir; closed-fill ve metrik
falloff aynı genel raster core'unu kullanır. Node ayarları graph serialization ile
birlikte mevcut `nodes.get_property/set_property` script ve IPC yüzeylerinden
erişilir. Derleme/canlı doğrulama kullanıcı isteğiyle faz sonuna ertelenmiştir.

Deliverable'lar:

- genel amaçlı `Curve to Mask`
- grade-limitli, non-destructive `Road Carve`
- tek evaluation'da tek `RoadCarveResult` snapshot'ı
- graded height + road_core/shoulder/cut/fill/exclusion pinleri
- bir mevcut `SplineObject` ve tek `DirtRoad` profil fixture'ı

Çıkış kriteri:

- Düz, diagonal, bend, side-slope ve ridge fixture'ları deterministiktir.
- Height, cut ve fill aynı snapshot/revision'dan geldiğini raporlar.
- Faz B solver çağrı sayısını artırmaz.
- Degenerate curve, boyut uyuşmazlığı ve imkânsız grade açık diagnostic üretir.

### Faz 2 — Manuel road authoring dikey dilimi

Mevcut spline authoring akışına yalnızca road profili ve crossing semantiği
eklenir.

Deliverable'lar:

- mevcut `SplineObject` seçip road profili atama/kaldırma
- `Footpath`, `DirtRoad`, `MainRoad` profil kütüphanesi
- road assignment serialization ve dangling-reference diagnostics
- mevcut spline undo/gizmo/edit akışının korunması
- UI, scripting ve IPC'nin aynı road-assignment core'unu çağırması

Script/IPC kapsamı curve noktalarını tekrar etmez:

```text
terrain.road.assign_profile
terrain.road.clear_profile
terrain.road.get_assignment
terrain.road.list_assignments
terrain.road.set_crossing_mode
terrain.road.get_diagnostics
```

Curve düzenleme `spline.*` operasyonlarında kalır.

Çıkış kriteri:

- UI ve IPC ile aynı spline'a profil atamak aynı serialized assignment'ı üretir.
- Save/load ve undo/redo curve geometrisini çift sahipli hale getirmez.
- Profil değiştirmek curve noktalarını değiştirmeden alanları yeniden üretir.

### Faz 3 — Publication, materyal graph'ı ve foliage

> **Durum: TAMAM (2026-08-31).** Altı canonical alan `RoadFieldsOutput`'tan tek
> snapshot'ta yayınlanıyor; foliage hard-exclusion slotu
> `infrastructure.foliage_exclusion`'a bağlı (`TerrainNodesV2.cpp:12733`).
> Materyal tarafı için **yeni kod yazılmadı ve yazılmamalıydı**: mevcut
> color/ramp/blend zinciri maskeleri okuyor. Yol yüzeyini erozyondan korumak da
> bir tariftir, kadran değil — `infrastructure.road_core` → Hydraulic Erosion
> `Hardness`.
>
> ★ Pin görünürlüğü düzeltildi: `RoadFieldsOutput`'un **zorunlu** girişleri
> Diagnostic olarak gizleniyordu. Bağlanması şart olan ve görünmeyen bir pin,
> bu dosyanın tarttığı iki arıza yönünden kötü olanıdır.

Deliverable'lar:

- beş canonical alanı Faz B'de aynı snapshot'tan yayınlama
- stable node pin keys ve Primary/Optional görünürlük sınıfları
- mevcut color/ramp/blend zinciriyle örnek path/dirt/main-road recipe'leri
- foliage hard exclusion bağlantısı
- falloff pinini `Publish Field` ile density maskesine bağlayan örnek
- CPU/GPU foliage parite ölçümleri

Çıkış kriteri:

- `terrain.list_fields` beş canonical adı raporlar.
- `terrain.field_stats` core/shoulder/cut/fill sınırlarını sayısal ölçer.
- Materyal rengi grading desteğinin dışında taşmaz.
- CPU ve GPU scatter hard exclusion içinde sıfır instance üretir.
- Field çözünürlüğü sabitken mesh çözünürlüğünü değiştirmek field
  istatistiklerini değiştirmez.

### Faz 4 — Opsiyonel yol yüzey geometrisi

> **Durum: YAZILDI, DERLENMEDİ (2026-08-31).** `TerrainRoadMesh.{h,cpp}` +
> `RtApiRoadMesh.cpp`; `terrain.road.build_mesh` / `clear_mesh` /
> `get_route`, Python ve panel karşılıkları.
>
> ★★★ Mesh **çözülmüş rotadan** üretilir — araziyi kazan örneklerin ta
> kendisinden. Eğriyi ikinci kez örneklemek aynı soruya ikinci bir cevap
> üretirdi ve ikisi önce **virajlarda** ayrışırdı; yani hizasızlık bir örnekleme
> sorunu olduğu halde bir modelleme sorunu gibi görünürdü.
>
> ★★ Sahiplik: atama, ürettiği nesnenin adını taşır (`meshObject`, serialize
> edilir) ve yeniden üretim o nesnenin **geometrisini değiştirir**. Her seferinde
> yeni nesne yayınlamak, sahnede üst üste binmiş ve her biri doğru görünen
> bayat yollar bırakmanın yoludur.
>
> Hendek **mesh'lenmez**: hendek arazidir. Zemin zaten oradayken ona ikinci bir
> yüzey koymak iki yüzey demektir.

Aynı `SplineObject`, profil ve çözülmüş route örneklerinden render mesh üretilir.

Deliverable'lar:

- UV/normal/material slot taşıyan ribbon mesh
- profile göre shoulder/curb varyantları
- flat `TriangleMesh` / DNA SoA publication
- stable spline/assignment -> generated mesh ownership
- export, scene lifecycle ve script/IPC diagnostics

Çıkış kriteri:

- Mesh, grading ve field maskeleri bend/side-slope üzerinde hizalıdır.
- Mesh kapatıldığında terrain-only yol geçerli kalır.
- Tekrarlı regeneration duplicate/stale scene objesi bırakmaz.

### Faz 5 — Hydrology crossing

> **Durum: YAZILDI, DERLENMEDİ (2026-08-31).** `crossing_mode` artık
> **saklanan bir kayıt değil, çözücünün okuduğu bir beyan**. Çözünürlük
> **örnek başınadır**: bir yol vadiyi tüm uzunluğunca değil, yalnızca gereken
> açıklıkta geçer.
>
> - `Terrain` — bugünkü davranış, zarf bağlar.
> - `Bridge` — su üstünde **veya** gereken dolgu `max_fill_meters`'ı aştığı
>   yerde açıklık. Açıklıkta arazi **hiç** değişmez, tabliye iki ayak arasında
>   **düzleştirilir** (sarkmayan bir tabliye).
> - `Tunnel` — gereken kesme `max_cut_meters`'ı aştığı yerde. Arazi değişmez ve
>   **hiçbir yüzey maskesi yayınlanmaz** — tünelin üstünde orman büyür.
> - `Ford` — su hattında yatağı takip eder; **asla doldurmaz** (dolduran bir
>   geçit sessizce bir barajdır) ve hendek/crown uygulanmaz.
> - `Auto` — `Water` pini bağlıysa su üstünde köprü, değilse **tam olarak
>   Terrain**. Auto'nun varsayılan olması sebebiyle burada iddialı davranmak,
>   kimsenin istemediği rotalarda arazinin sessizce kazılmayı bırakmasıyla
>   sonuçlanırdı.
>
> ★★★ Su **tek gözlemdir**: `Road Network`'ün `Water` pini bağlı değilse bir
> Ford çözülemez ve bu `crossing_diagnostic` olarak **raporlanır**, sessizce
> normal yola düşürülmez — sessizce düşen bir ford viewport'ta tamamen makul
> görünür.
>
> ★★ Açıklıklar `roadWidthMeters` kadar **genişletilir** (abutment) ve yakın
> olanlar birleşir: tam dolgu limitinin bağladığı yerde başlayan bir köprünün
> ayağı yoktur, ve üç örnek dolgu ile ayrılmış iki açıklık iki köprü değildir.
>
> Ölçüm: `terrain.road.get_route` — `bridge_samples`, `ford_samples`,
> `tunnel_samples`. Bir beyanın gerçekten çözüldüğünü script'in kanıtlama yolu
> budur.

Deliverable'lar:

- lake/channel/shoreline/flood-risk routing ve grading kısıtları
- explicit `Terrain`, `Bridge`, `Ford`, `Tunnel` davranışı
- bridge span altında grading suppression
- ford için sınırlı terrain/material profili
- çözülemeyen crossing için açık diagnostic
- script/IPC crossing kontrolleri ve ölçümü

Çıkış kriteri:

- Normal terrain segmenti accepted lake veya korunan channel'ı doldurmaz.
- Bridge river bed'i kesmez, doldurmaz veya düzleştirmez.
- Ford yalnızca ilan edilen kesitte terrain'i etkiler.

### Faz 6 — Otomatik path/road network

Otomatik sistem ikinci bir yol temsili üretmez. Sonuçları scene-authoritative
`SplineObject`'ler ve bunlara bağlı `RoadAssignment` kayıtlarıdır.

Deliverable'lar:

- slope, cliff, wetland, cut/fill ve crossing cost field'ı
- coarse-to-fine route search
- spline fit, simplification ve curvature limitleri
- locked/manual spline'ları koruyan regeneration
- seed/version determinism ve scripting/IPC generation controls

Çıkış kriteri:

- Aynı input ve seed aynı spline/assignment sonuçlarını üretir.
- Generated ve manual spline aynı Faz 1-5 yolundan geçer.
- Locked manual spline regeneration sırasında değişmez.

### Faz 7 — Settlement temeli

Deliverable'lar:

- slope, water access, flood risk ve road access suitability alanı
- city/village anchor'ları ve road network bağlantı talepleri
- settlement footprint/exclusion alanları
- ileride streets/parcels/buildings için ayrılmış contract'lar
- scripting/IPC operations ve diagnostics

Çıkış kriteri:

- Settlement geçersiz eğim, accepted lake ve flood-risk alanından kaçınır.
- Anchor değişikliği yalnızca etkilediği route taleplerini deterministik üretir.

## Modül ve UI sınırları

Yeni solver ve servis mantığı odaklı `.h/.cpp` modüllerinde yaşar. 2000 satırı
aşmış dosyalara yalnızca küçük include, node registration, çağrı veya IPC route
wiring'i eklenebilir. UI, script ve IPC ayrı business logic taşıyamaz.

- Faz 3.6 modülleri genel curve -> field ve profile/grade -> result sahibidir.
- Road assignment/profile modülü spline ID -> semantik metadata taşır.
- Road mesh builder opsiyonel flat geometry üretir.
- Road routing gelecekte cost search sonucunu `SplineObject` olarak yayınlar.
- Nokta/handle düzenleme mevcut `SplineObject` overlay/gizmo'sunda kalır.
- Profile/crossing sağ contextual dock'ta, analiz node'ları alt Terrain graph'ta
  yaşar; büyük kalıcı road shelf eklenmez.

## İlk uygulama partisi

1. Faz 0 contract'larını ve beş canonical field adını dondur.
2. Önce `terrain.field_stats` ölçüm ucunu tamamla.
3. Faz 3.6 içinde `Curve to Mask` sentetik fixture'larını kur.
4. Faz A'da bir kez çalışan ve Faz B'ye immutable snapshot taşıyan en dar
   `Road Carve` çözümünü yaz.
5. Tek mevcut `SplineObject` + `DirtRoad` ile graded height, core, shoulder,
   cut, fill ve exclusion'ı ölç.
6. Bu sayısal kapı geçmeden profile UI, material recipe, mesh veya routing'e
   geçme.

Codex build veya uygulama başlatmaz; her fazın derleme ve canlı doğrulamasını
kullanıcı yapar.
