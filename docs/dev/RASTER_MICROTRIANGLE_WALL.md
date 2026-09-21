# Raster duvarı MİKRO-ÜÇGEN — ve LOD bütçesi mesh başına ÇARPILIYOR

> **Durum:** AKTİF — kök neden kaynakta bulundu, düzeltildi ve **CANLI
> DOĞRULANDI** (2026-09-08, derlenmiş binary, 10 030 instance / ~1 milyar
> instance üçgeni olan orman sahnesi). §7 tam maliyet ayrışmasını,
> §8 asıl darboğazı taşıyor. ★ §2-3'ün "primitive rate" iddiası §7'de
> ölçümle **daraltıldı**.
> Önceki not [RASTER_SCENE_LIGHTING_FRAGMENT_COST.md](RASTER_SCENE_LIGHTING_FRAGMENT_COST.md)
> "üçgenle doğrusal → overdraw" diyordu; o çıkarım **eksik belirlenmişti**,
> aşağısı onu düzeltiyor.

## 1. Düzeltme: "üçgenle doğrusal" tek başına overdraw demek DEĞİL

Aynı doğrusallığı üç mekanizma üretir ve ikisi overdraw değil:

★★★★ **2026-09-08 akşamı ÖLÇÜMLE seçildi: hangi satırın geçerli olduğu artık
biliniyor.** Derinlik ön geçişi yazıldı ve A/B'si yapıldı — **hiçbir şey
kazandırmadı, ~22 ms ekledi** (dört tur, KAPALI 324 / AÇIK 346). Yani elenecek
gizli fragment yok: baskın mekanizma **quad kuantalaması**, oklüzyon overdraw'ı
değil. Ayrıntı: [RAYFUSION_RT_SHADOWS_AND_TRANSPARENCY.md](RAYFUSION_RT_SHADOWS_AND_TRANSPARENCY.md) §6c.

| Mekanizma | Maliyeti ne belirler | Depth prepass düzeltir mi |
|---|---|---|
| Klasik overdraw (gizli fragment) | kaplanan piksel × katman | **Evet** (erken-Z) |
| **Quad kuantalama** (piksel-altı üçgen) | üçgen sayısı × 4 | **Hayır** |
| **Primitive rate** (setup/raster) | üçgen sayısı | **Hayır — iki katına çıkarır** |

## 2. Mikro-üçgen rejimi KOŞULSUZ

45M üçgen / 1,59M piksel = **piksel başına 28 üçgen**, foliage ekranın
tamamını kaplasa bile. Ortalama üçgen pikselin 1/28'inden küçük.

- **Quad:** fragment gölgelendirme türev için 2×2 quad hâlinde yapılır. Bir
  pikselin 1/28'ini kaplayan üçgen yine **4 fragment** çalıştırır (3'ü helper).
  45M üçgen → **≥180M shader çağrısı**, 1,59M piksel için. ~113x israf.
  ★ Çözünürlüğü düşürmek bunu **iyileştirmez** — üçgenler daha da küçülür.
- **Primitive rate:** SOLID modun kendi sayısı 45,0M / 98,9 ms =
  **455M üçgen/sn**. Modern bir GPU büyük üçgenlerde bunun 10-20 katını yapar.
  Yani **trivial shader'lı geçiş bile shader'a değil, üçgen sayımına takılı.**

## 3. Bu yüzden depth prepass'in kolu kısa

Prepass gizli quad'ları erken-Z ile gerçekten öldürür (218 ms'lik gölgelendirme
teriminin çoğu). Ama **geometriyi bir kez daha gönderir**:

```
316 ms  ->  ~99 (prepass primitive) + ~99 (ana geçiş primitive) + az gölgelendirme
        ~=  200-220 ms      = 1,4-1,5x
```

Gerçek bir kazanç, ama taban ~200 ms'de çakılı kalır ve RT'den hâlâ ~10x uzak.
★ Prepass **doğru bir iş ama baş aday değil**; üçgen sayısı düştükten sonra
ucuz bir ek olarak girer.

## 4. ★★★★ KÖK NEDEN: LOD bütçesi mesh başına ÇARPILIYOR

`raster_cull.comp` `finalize()`, her mesh'in LOD mesafe eşiğini çözdüğü yer:

```glsl
uint allowed = max(1u, g.scatterTriangleTarget / p.trianglesPerInstance);
uint seen    = max(1u, full);
float scale  = clamp(float(allowed) / float(seen), 0.5, 1.5);
st.x = clamp(d * scale, g.lodMinDistSq, g.lodMaxDistSq);
```

`g.scatterTriangleTarget` **tüm sahnenin** hedefi (29,6M), ama bu satır
**her mesh için ayrı** koşuyor. 35 cull mesh'in her biri bütün bütçeyi
kendisinin sanıyor → efektif tavan **35 × 29,6M**.

Ölçülen telemetriyle birebir tutuyor:

```
45,9M üçgen / 722 full instance  ~= 63,5k üçgen/instance
allowed = 29,6M / 63,5k          ~= 466 instance   (MESH BASINA)
mesh basina gercek instance      ~= 1034 / 35 = 30
30 << 466  ->  scale her kare 1,5'e yapisir
           ->  esik lodMaxDistSq'e (1e18) kacar
           ->  HICBIR SEY proxy'ye dusmez
```

Proxy'ye düşen **28** instance, instance başına ~1M üçgeni olan tek bir yoğun
mesh — `allowed`'ın ancak orada instance sayısının altına indiği yer.

★★★ **Sonuç: bütçe hiç uygulanmadı.** Bugüne kadar ölçülen her raster sayısı
*LOD'suz* raster. `scatter_triangle_target` telemetride raporlanıyor ve makul
görünüyor — **ölçü aleti hedefi gösteriyor, uygulananı değil.**

★ Ve arıza sessiz: LOD devreye girmeyince sahne **doğru** çizilir, yalnız
yavaştır. [RASTER_GPU_CULLING_NEVER_ENABLED.md](RASTER_GPU_CULLING_NEVER_ENABLED.md)
ile aynı imza — resmi olmayan arıza.

### Önerilen düzeltme: hedefi TALEBE ORANTILI paylaştır

Mesh başına pay, CPU'da bir kare geriden:

```
demand_i = instanceCount_i * trianglesPerInstance_i
share_i  = target * demand_i / sum(demand)
```

`MeshBinding`'e bir `triangleTarget` alanı, shader `g.scatterTriangleTarget`
yerine `p.triangleTarget` okur. Mesh başına tek adımlık çözüm aynen kalır;
yalnızca her mesh **kendi payına** yakınsar. Üstüne mevcut
`m_rasterScatterTargetScale` geri beslemesi zaten var.

⚠ Bütçe düşünce proxy'ler ilk kez gerçekten görünür olacak — ve kullanıcı
proxy kalitesinin **düşük** olduğunu bildirdi. Yani bu düzeltme hızı getirir
ve **görsel sorunu açığa çıkarır**; ikisi ayrı iş.

## 7. ÖLÇÜLDÜ — düzeltme sonrası, orman sahnesi

10 030 instance, ~1 milyar instance üçgeni, yoğun opacity dokusu. 1680×945.
Yöntem: IPC'den kamerayı 14-16 kez oynat, geçen süreyi `frames_submitted`
deltasına böl.

### 7a. Düzeltme tuttu

| | önce | sonra |
|---|---|---|
| `proxy_instances` | 28 | **4 965** |
| `visible_triangles` | 45,9M (hedefin %55 üstü) | **32,1M** (hedef 29,6M) |

★ Görsel kabul de yapıldı: **en yakın ağaçlar tam kalite** (kullanıcı, ekranda).
Bu partinin en sinsi başarısızlığı — sayılar kusursuzken yakın planın da
impostor'a düşmesi — **gerçekleşmedi**. Mesafe eşiği ve geçiş bandı
(`kRasterLodBandWidth = 0.35`) doğru yerde.

### 7b. Solid ÇOK daha fazla üçgen çiziyor ve çok daha hızlı

| | ms/kare | üçgen | ms/Müçgen |
|---|---|---|---|
| **solid** | **97,9** | **84,9M** | **1,15** |
| **scene/fusion** | **476,6** | **32,2M** | **14,8** |

★★★★ **DÜZELTME (§2-3'e):** solid burada **867M üçgen/sn** çıkarıyor. Yani
primitive rate bu sayılarda **bağlayıcı kısıt değil**. Önceki notta 455M/sn
"primitive-bound" diye yorumlanmıştı; o sahne daha küçüktü ve iddia fazla
genişti. Duvar, **18,4 üçgen/piksel ile çarpılan fragment maliyeti**.

### 7c. Scene maliyeti üçgenle DOĞRUSAL (kalite preset süpürmesi)

| preset | ms/kare | üçgen | ms/Müçgen |
|---|---|---|---|
| performance | 319,7 | 21,0M | 15,2 |
| balanced | 452,9 | 29,0M | 15,6 |
| quality | 616,9 | 43,5M | 14,2 |

★ Yani **üçgeni yarıya indirmek maliyeti yarıya indiriyor.** Proxy fusion'da
etkisiz *değil* — §8'de görüleceği gibi **tükenmiş**.

### 7d. Tam ayrışma (~29-30M üçgende)

| Katman | ms | pay |
|---|---|---|
| geometri + trivial shading (solid) | 42 | %9 |
| + materyal / opacity değerlendirmesi | **+139** | %30 |
| + scene ambient (sky + probe yolu) | **+162** | %36 |
| + yönlü ışık & 3 kaskad gölge | **+113** | %25 |
| **toplam** | **456** | ölçülen 455,6 ✓ |

Kollar: `viewport.set_shading`, `viewport.set_preview_lighting`,
`lights.set_visible`.

### 7e. RayFusion bu sahnede maliyet DEĞİL

Speküler görünürlük A/B'si (traced ↔ sky_bake): **443,9 vs 452,2 ms**,
ms/Müçgen 15,15 vs 15,08 — **gürültü içinde**. Küçük sahnede ölçülen ~28 ms'lik
(%16) pay burada kayboldu.

★ Difüz probe tüketicisinin kapatma kolu hâlâ yok, **ama ölçüm boşluğu artık
kapalı**: `rfSampleProbeField` fragment'in hücresi probe penceresinin dışındaysa
`probe_field.glsl:92`'de hemen `false` döner. Pencere 4×2×4 hücre × 3 m = 12×6×12
metre; orman sahnesi yüzlerce metre. `hit_fraction = 0,055`, yani fragment'lerin
**%94,5'i erken çıkıyor.** Tüketici bu maliyeti üretemez.

## 8. ★★★★★ ASIL DARBOĞAZ: LOD İKİLİ, arada hiçbir kademe yok

```
full  : 28.761.921 tris /  292 inst =  98.500 tris/instance
proxy :    476.640 tris / 4965 inst =      96 tris/instance
oran  : 1 : 1.026
```

**98 500 üçgen ya da 96. Arada hiçbir şey yok.** Üç gözlemi birden bu açıklıyor:

- **"proxy çok kalitesiz"** — 98 500 üçgenlik bitkinin yerine 96 üçgenlik kart.
- **"proxy fusion'da etkisiz"** — 4 965 proxy instance toplam üçgenin yalnızca
  **%1,6**'sını taşıyor; kalan **292** full instance **%98,4**'ünü taşıyor.
  Proxy tükenmiş: daha fazlası ancak yakın plandaki bitkileri kart yapmakla
  olur, o da kabul edilemez.
- **"solid modda optimum"** — solid'de 1,15 ms/Müçgen, 98 500 üçgen sığıyor;
  fusion'da 15 ms/Müçgen, sığmıyor.

★★★★ **Cluster LOD'un somut gerekçesi budur** — primitive rate değil, bu
uçurum. Sürekli (ya da hiç değilse 4-5 kademeli) bir LOD, ekran hatasına göre
98 500 → 25 000 → 6 000 → 1 500 → 96 verir. §7c doğrusallığı, üçgendeki her
azalmanın fusion maliyetine **birebir** yansıdığını gösteriyor.

## 8b. ★★★★★ Uçurumu doldurmanın tutamağı KAYNAKTA HAZIR

Sahne: tek scatter grubu "Forest Foliage Assets", **10 000 instance**, iki
kaynak, `triangle_count = 992 232 310`.

`.glb` başlıklarından materyal başına üçgen (doğrudan glTF JSON'undan):

| Ağaç | toplam üçgen | iğne + twig payı |
|---|---|---|
| Pinus ponderosa | 27 691 | **%95,2** |
| Pine Jeffreyi | 170 356 | **%82,5** |
| **ortalama** | **99 024** | **%84,3** |

★ Ortalama, ölçülen 98 500 / 99 223 üçgen/instance ile birebir tutuyor —
scatter iki türü de 1,0 ağırlıkla kullanıyor.

★★★★ **Üçgenlerin %84'ü alfa-test'li iğne/twig kartı, ve bunlar AYRI materyal
ID'sinde:** `Pin pond need 1/2`, `Pinus pond twig`, `Pin jef need 1/2`,
`Pin jef twig`. Gövde/dal ayrı (`trunk`, `br 01`, `br cut`, `dead br`,
`peeling br`).

Bunun anlamı: **ara LOD için genel bir QEM decimator'a gerek yok.** Foliage'da
QEM zaten yanlış araçtır (yaprak kartları kopuk quad'lardır; kenar çökertme çöp
üretir). Doğru işlem **kart seyreltme**: materyal ID'sinden iğne/twig üçgenlerini
seç, bir kısmını at, kalanları biraz büyüt. Kaba merdiven:

| Kademe | işlem | üçgen | fusion tahmini* |
|---|---|---|---|
| L0 | tam | 99 000 | 433 ms |
| L1 | iğne kartlarının %60'ı atılır | ~48 600 | ~210 ms |
| L2 | %85 atılır + dal sadeleştirme | ~28 000 | ~120 ms |
| L3 | mevcut impostor | 96 | — |

\* §7c'deki 15 ms/Müçgen doğrusallığından; **tahmin, ölçüm değil.**

★★ Ve kazanç **iki terimde birden**: iğne kartları alfa-test'li olduğu için
seyreltme, §7d'deki 139 ms'lik materyal/opacity terimini de aynı oranda keser.
Kullanıcının "opacity çok maliyet üretiyor" gözlemi ve bu tutamak aynı şeye
bakıyor.

⚠ İki tür çok farklı: 27,7k ve 170,4k. Asıl yük Jeffreyi'de; merdiven tür başına
ayarlanmalı, sabit oranla değil.

### Yapısal engel: LOD zinciri İKİ kademeye sabitlenmiş

`RasterGpuCull::MeshBinding` tek bir proxy taşıyor (`proxyDrawSlot`,
`proxyElementCount`, `proxyOutBase`, `proxyFlags`) ve `classify()` bir **bool**
seçiyor. N kademe için: kademe dizisi + mesafe eşiği dizisi, `classify()` bir
**indeks** seçer, mesh başına 2 yerine N draw slot. Sınırlı ama gerçek bir
değişiklik — ve L1/L2 üretimi olmadan tek başına işe yaramaz, tersi de.

## 9. Yön: cluster LOD, ve piksel-altını RT'ye devretmek

Kullanıcının önerdiği boru hattı
(`Original Geometry -> GPU Clusters -> Screen Error -> Select/Compact -> Raster`)
duvarın kendisine vuruyor: 45M → 3M üç terimi birden çökertir (primitive rate,
quad israfı, overdraw). 10x oradan gelir.

★ Nanite'ın işin **yarısı** budur; diğer yarısı seçimden sonra mikro-üçgenleri
**compute shader'da bir software rasterizer**'a vermektir — donanım quad'ı o
rejimde yenilemediği için. Unreal oraya ray tracing koymadı, kendi
rasterleyicisini yazdı.

★★★★ **Ama bu depoda RT arka ucu Unreal'dekinden ucuz.** Unreal RT'yi seçmedi
çünkü BVH'yi bu iş için ayrıca kurup güncel tutmak pahalıydı. Burada **AS zaten
var, zaten bakımlı, ve aynı sahnede 10x hızlı ÖLÇÜLDÜ.** "Piksel-altı bölgeyi
ray'e devret" burada zaten ödenmiş bir arka uca devretmek demek — ve **kötü
impostor sorununu da çözer**: uzak foliage'ın yerine kalitesiz bir kart değil,
gerçek geometri koyuyorsun.

★ İkisi alternatif değil: "screen error evaluation" her iki yolun da aynı ön
ucu. Fark seçimden *sonra*. Ortak parçayı önce kur, arka uç kararını ölçümden
sonra ver.

⚠ Hibritin gerçek maliyeti hız değil **dikiş**: iki üretici, tek tüketici.
Raster ile trace aynı materyal değerlendirmesine varmazsa mesafe bandında gözle
görünür bir çizgi çıkar. Bu deponun kendi dersi — RayFusion dikişi de tam
olarak bu şekilde çözülmüştü (tüketici sabit, üretici değişir).

## 10. Sıra

| # | İş | Boyut | Durum / neden |
|---|---|---|---|
| 0 | Mesh başına bütçe payı (§4) | küçük | **BİTTİ, canlı doğrulandı** (§7a) |
| 1 | **Ara LOD kademeleri** — 98 500 ↔ 96 uçurumunu doldur | orta | ★★★★★ §8/§8b. Cluster'dan **önce**: aynı terime vurur, tutamak kaynakta hazır (%84 iğne kartı, ayrı materyal ID), opacity terimini de keser, ve "proxy kalitesiz" şikâyetini kapatır. İki parça: (a) LOD zincirini N kademeye aç, (b) kart seyreltmeyle L1/L2 üret |
| 2 | Cluster + screen-error ön ucu | büyük | Duvarın sürekli çözümü |
| 3 | Arka uç: uzak küme → trace, yakın → raster | büyük | AS bedava; dikiş asıl risk |
| 4 | Fragment yolunu ucuzlat (§7d: opacity 139 + ambient 162 + gölge 113) | orta | Üçgenden bağımsız ikinci eksen; 15 → 6 ms/Müçgen bile 2,5x |
| 5 | Depth prepass | orta | Üçgen sayısı düştükten sonra ucuz ek |

## İlgili

- [RASTER_SCENE_LIGHTING_FRAGMENT_COST.md](RASTER_SCENE_LIGHTING_FRAGMENT_COST.md) — fragment ölçümleri (overdraw çıkarımı burada düzeltildi)
- [RASTER_GPU_CULLING_NEVER_ENABLED.md](RASTER_GPU_CULLING_NEVER_ENABLED.md) — aynı yolun bir önceki sessiz arızası
- [RAYFUSION_FOLIAGE_FRAME_COST.md](RAYFUSION_FOLIAGE_FRAME_COST.md) — aynı sahnenin CPU kök nedeni
