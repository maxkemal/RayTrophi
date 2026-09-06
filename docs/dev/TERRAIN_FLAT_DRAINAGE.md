# Düzlükte akarsu davranışı — Garbrecht & Martz akış gradyanı

> **Durum:** REFERANS — 2026-09-03'te uygulandı ve **canlı doğrulandı**
> (ölçümler aşağıda). Kök neden analizi kalıcıdır.

## Belirti

Çukur sınırlarında ve düşük eğimli düzlüklerde akış **düz ve köşeli** çıkıyordu:
0 ve 45 derecelik uzun çizgiler, dikdörtgen basamaklar, birleşmeyen paralel
kollar. Aynı haritanın gerçek eğimli kısımları kusursuz görünüyordu. Çözünürlük
(1K–4K) ve arazi tipi değiştirilerek denendi; **desen aynı sınırlarda üretildi**,
yani arıza manzarada değil algoritmadaydı.

## Kök neden — tek bir şey değil, kilitlenmiş bir çift

### 1. Doldurulmuş düzlükte gradyan YOKTUR

Bu, sistemin kabul etmesi gereken temel gerçek: bir düzlükte veya doldurulmuş
çukurda akışın gittiği yön **arazinin değil, koşullandırma algoritmasının**
özelliğidir. Eski şema her taşma halkasını birkaç ulp kaldırıyordu
(`kFillEpsilon`, dither'lı). Sonuç bir **geodezik mesafe alanı**dır — ve doldurma
kardinal ile çapraz adımı **aynı** ücretlendirdiği için o metrik Chebyshev'dir,
seviye eğrileri **KAREDIR**. Kare bir mesafe alanında en dik iniş 0 ve 45
derecede akar. Belirtinin şekli buydu.

**★★★ Dither bunu çözemez ve çözmediği ölçülebilir.** Dither hücre başına bir
*çarpan*dır, adım ise yol boyunca **toplanır**; N adımdan sonra bağıl saçılım
1/√N ile küçülür. Uzun bir düzlükte yüzey, geodezik mesafenin lineer bir
fonksiyonuna geri yakınsar. Dither **dokuyu** bozar, **şekli** bozmaz — ve sorun
şekildeydi.

### 2. Düzlük kendini yeniden şekillendiremiyordu

`terrain_lem_incise` asınmayı `incisionSafety * (alıcıya düşüş)` ile sınırlar.
Birkaç ulp **sayısal olarak sıfır düşüş**tür, yani düzlük **hiçbir parametre
ayarıyla oyulamıyordu**. İlk kesim olmayınca `A^m` pozitif geri beslemesi hiç
başlamıyor: nehri ovadan ayıran mekanizma yok. Dolayısıyla **ilk drenaj
çözümünün düzlüğe çizdiği desen NİHAİ desendi.**

Bu tuzağın diğer yarısı zaten belgelenmişti — `maxDepositionMeters` notu düz
hücreyi "yalnızca yükselebilen bir cırcır" diye tarif ediyor. Çökeltme yarısı
kapatılmış, **oyma yarısı açık kalmıştı.**

### 3. Yardımcı etkenler

- **Piramit bütçesi ters çalışıyordu.** `drainageFillPasses >> (deepest-1-l)`
  seviye 0'da 2K ve üstünde 12'ye çakılıyordu: **harita büyüdükçe bütçe
  küçülüyordu**, yani düzlükler hücre cinsinden genişledikçe. 12 pass = bilgi 12
  hücre yol alır; daha geniş her düzlük kaba seviyenin **nearest-neighbour**
  prolongasyonunu aynen koruyordu. O parça parça sabit yüzeyde bloklar birebir
  eşit olduğu için ağırlık satırları sıfır çıkıyor, akış yalnızca blok
  kenarlarında adım atabiliyordu. Dikdörtgen basamaklar buradan.
- **ULP kuantalanması MFD'yi D8'e çöktürüyordu.** Düzlükte komşu farkları
  yalnızca birkaç ayrık float değeri alıyor; `slope^1.5` sonra 255'e kuantalanınca
  ağırlık satırı neredeyse ikili oluyor. MFD'nin yayılma işi tam da gerektiği
  yerde ölüyordu.
- **Sabit beraberlik yanlılığı.** `terrain_lem_weights` yuvarlama artığını
  `power[d] > maxPower` kazananına ekliyordu; katı `>` ile her yakın beraberlik
  `d = 0`'da, sol-üst köşegende kalıyordu. Düzlükte sekiz güç zaten yakın
  beraberdir, yani sabit yönde kalıcı bir itme.
- **Ölü kod, canlı yorum.** `terrain_flow_fill.comp` başlığı coherent noise'un
  artefaktı çözdüğünü söylüyordu; LEM ve macro yolları `noiseAmplitude = 0`
  geçiyordu. (Üçüncü tüketici — watershed/flow-mask yolu — onu **gerçekten**
  0.03 ile kullanıyor, bu yüzden parametre sökülmedi, yorumu düzeltildi.)

## Çözüm

### Doldurma artık yüzeye DOKUNMUYOR

`eps = 0`. Bir havzanın her hücresi **birebir aynı float**'a oturur. "Düzlük"
böylece iyi tanımlı bir küme olur ve `filled` yalnızca tek bir soruyu yanıtlar:
**durgun su nereye erişiyor.** Tüketicilerinin ondan istediği hep buydu.

### Akış gradyanı İKİ mesafe alanından kurulur

`terrain_lem_flat_{seed,relax,ramp}.comp` (ve CPU'da `resolveFlats` /
`buildRouteSurface`):

- `dLow` — düzlüğün **çıkışlarına** geodezik mesafe. Eski merdivenin taşıdığı terim.
- `dHigh` — **yüksek arazinin düzlüğe değdiği yere** geodezik mesafe. Eksik olan
  terim buydu ve birleşmeyi sağlayan odur: ovaya giren bir dere, çıktığı duvardan
  uzaklaşmaya devam eder; hepsi birden çıkış halkalarına dönüp yan yana akmaz.

```
route = filled + flatStep * ( 2*dLow + (highRef - dHigh) )
```

Mesafeler **hücrenin on ikide biri** cinsindendir, böylece çapraz adım 17/12 =
1.4167 (≈√2) ödeyebilir. Kardinal ile çapraza aynı ücreti kesmek alanı kare
yapan şeydi.

**★★★ Katsayılar bir kanıttır, zevk ayarı değil.** Gevşetmenin kendi ebeveyn
bağı boyunca `dLow` en az bir kardinal adım (1 hücre) düşerken `dHigh` en fazla
bir çapraz adım (1.4167) oynar:

```
Δ(route) ≤ -2·1 + 1·1.4167 = -0.583 < 0
```

Yani **her çözülmüş düzlük hücresinin kesinlikle daha alçak bir komşusu vardır**
— kaç pass koşulduğundan bağımsız olarak. Yüzeyi gürültüyle dürtmek (dither'ın
yaptığı) bu garantiyi veremez, çünkü düzlüğün ortasında komşusuz çukurcuklar
üretir. Burada perturbe edilen **mesafedir**, yüzey değil.

**★★★★ `highRef` bir SINIRDIR ve sınır kanıtı GPU'da ayakta tutan şeydir.**
İki cephe birlikte yakınsamaz (bir ova çıkışından 20, duvarından 900 hücre
uzakta olabilir), dolayısıyla bazı hücreler "cephe gelmedi" nöbetçisini tutarken
komşuları gerçek mesafe tutar. Sınır olmasa bu iki değer yan yana gelir, `dHigh`
terimi 1.4167'den çok daha fazla sıçrar ve **ovanın ortasında yerel bir çukuk**
belirir — yani bu geçişin sökmek için var olduğu kusur, kendi teşhis boşluğundan
geri girer. `highRef ≤ flatResolvePasses - 2` tutulduğunda çözülmemiş her hücre
**ve tüm komşuları** terimin sınırlanmış (=0) bölgesinde kalır; sıçrama ancak iki
*çözülmüş* hücre arasında olabilir, onlar da en fazla bir çapraz adım farklıdır.

Sınır aynı zamanda daha iyi bir modeldir: G&M bu terimi düzlük başına normalize
eder; sınırsız bir mesafe bin hücrelik bir ovayı tek duvardan uzağa eğip çıkış
terimini boğardı. Terimin işi, derenin yamacı terk ederken **nereye nişan
aldığına** karar vermektir — bu bir sınır civarı etkisidir.

### İKİ yüzey, ve ayrılmaları gerekiyordu

| Alan | Anlamı | Okuyanlar |
|---|---|---|
| `lakeSurface` | Doldurulmuş yüzey, düzlükte tam düz. **Su nerede DURUYOR.** | göl kapısı, `lakeDepth`, water level |
| `route` | `lakeSurface` + düzlük gradyanı. **Su nereye GİDİYOR.** | ağırlıklar, alıcı seçimi, akış yönü |

Eskiden tek alandı ve merdiveni taşıyordu — bu onu hem kötü bir göl yüzeyi
yapıyordu (merdiven durgun su sayılıyordu; `lakeCells` haritanın çeyreğini
okuyup hiçbir şey ifade etmeyebiliyordu) hem de daha kötü bir akış yüzeyi.
Ayırmak **sıfır bellek** maliyetlidir: `filledA`/`filledB` zaten bir ping-pong
çiftiydi. Geodezik mesafe alanları da `areaA`/`areaB`'yi scratch olarak kullanır
(akümülasyona kadar boştalar), uint32'e iki uint16 paketlenerek.

### Oyma kilidi açıldı

```glsl
float bedDrop   = max(hCur - heightIn[receiver], 0.0);   // gerçek rölyef
float routeDrop = max(rCur - routeField[receiver], 0.0); // düzlükte: flat gradient
float dropNorm  = bedDrop + routeDrop;
```

Gerçek rölyefi olan yerde `routeDrop = 0`, davranış birebir aynı. Düzlükte
`bedDrop = 0` ve sınır artık **mutlak veto değil, bir limit**. Anti-çukur
garantisi ayakta: akış yüzeyi alıcıya doğru kesinlikle iner, her hücre kendi
akış düşüşünün en fazla `incisionSafety` kadarını keser, dolayısıyla ovaya
oyulan kanal aşağı doğru monoton kalır. Milimetrenin altında ters döndüğü yerde
bir sonraki doldurma onu fiziksel olarak ne ise o şekilde okur — **su tutan bir
kanal** — ve `lakeEps` (1 cm) bunu göl eşiğinin altında tutar.

**★★★ Asıl nokta budur:** ramp bir cevap değil, **geri besleme döngüsü için bir
tohum**. Kanal yatağını açar açmaz yatağın kendi rölyefi devralır ve sentetik
gradyan önemsizleşir. Kıvrılma ve birleşme buradan gelir, ramp'tan değil.

### Ölçüm — çünkü çözülmemiş düzlük İNANDIRICI görünür

`unresolved_flat_cells` (GPU'da atomik sayaç, IPC ve panelde görünür): çıkış
cephesinin hiç ulaşamadığı düzlük hücreleri. Bunlar yönlendirme gradyanı
taşımaz, **terminal çukuktur** ve üstlerindeki her havzayı sessizce kırpar —
render'da sıradan zemin gibi görünürler. Geri okuma başarısız olursa **-1**
yazılır: "ölçülmedi", ölçülmüş sıfırdan **ayırt edilebilir** olmak zorunda.

## Canlı ölçüm (2026-09-03, 1024² / 4000 m / snowy_mountain_valley, GPU)

Her tur öncesi grafik yeniden değerlendirildi, yani **aynı zemin**; tek değişken
`flat_gradient`. `terrain.erode` + `terrain.erosion_stats` üzerinden.

| | A `grad = 0` | B `grad = 2e-4` |
|---|---|---|
| eroded | 2103.1 | 2230.3 |
| deposited | 1885.0 | 2054.2 |
| drainage_density | %2.98 | **%6.74** |
| max_drainage_area_fraction | %18.6 | **%69.6** |
| lake_area_fraction | %10.01 | %10.09 |
| unresolved_flat_cells | 0 | 0 |
| mass_error_fraction | −9.0e−7 | −1.2e−6 |

Okunuşu: aşınma arttı (düzlük artık oyuyor), kanal yoğunluğu ikiye katlandı, ve
**gövde havzası haritanın %19'undan %70'ine çıktı** — yani düzlük çukurları
havzaları kırpmayı bıraktı. Göl oranı sabit kaldı, yani ramp sahte göl
üretmiyor; kütle defteri iki turda da kapanıyor.

**★ `grad = 0` eski BUILD değildir.** Doldurma artık merdiven de eklemediği için
o ayarda düzlükte hiç gradyan kalmaz; A bir **taban**dır, bir "öncesi" değil.
Gerçek öncesi/sonrası için eski commit'i derlemek gerekir.

### Teşhis aleti ateş edebiliyor mu?

Hep sıfır raporlayabilen bir sayaç ölçüm değildir. `flat_resolve_passes`
düşürülünce sayaç **monotonik** olarak yükseliyor ve gövde havzası onunla
birlikte çöküyor — yani belgelenen nedensellik (çözülmemiş düzlük = terminal
çukuk = kırpılmış havza) doğrudan ölçülüyor:

| `flat_resolve_passes` | unresolved flats | haritanın | trunk |
|---|---|---|---|
| 4 | 96 974 | %9.25 | %19.3 |
| 32 | 67 576 | %6.45 | %35.7 |
| 512 | **0** | %0 | **%50.5** |

4 pass'te çözülmemiş oran (%9.25) haritanın düzlük oranına (%9.73,
`landform_stats.flat_fraction`) neredeyse eşit — yani o bütçede pratikte
**hiçbir** düzlük çözülmüyor. Tutarlı.

## Kadranlar

| Ad | Varsayılan | Anlamı |
|---|---|---|
| `flat_gradient` | `2e-4` m/m | Doldurulmuş düzlük için varsayılan eğim. **0 = ramp kapalı**, eski (donmuş) davranış — yani kullanılabilir bir A/B kontrolü. |
| `flat_resolve_passes` | 256 (Balanced) | GPU, pass başına bir hücre: **çözülebilen en geniş düzlük, hücre cinsinden.** |

`flat_gradient` kalite preset'ine **dahil değildir** ve olmamalıdır: bir bütçe
değil, fiziksel bir varsayımdır. `flat_resolve_passes` bir bütçedir, preset'e
dahildir (Draft 128 / Balanced 256 / High 512).

## Ayrıca düzeltilenler

- Piramit seviye 0 dolgu bütçesi kaydırmadan **muaf** tutuldu.
- `terrain_flow_fill` merdiven adımı `dist8` ile ölçekleniyor (kare → sekizgen).
  Bu, `eps > 0` geçen diğer iki tüketiciyi de biraz daha az köşeli yapar.
- `terrain_lem_weights` beraberlik tarayıcısı hash'lenmiş bir ofsetten başlıyor;
  yuvarlama artığı artık sabit olarak sol-üst köşegene gitmiyor.
- Akış **yönü** (`directionX/Y`) artık `route`'tan türetiliyor. Göl yüzeyi
  düzlükte birebir düz olduğu için hiçbir yön veremezdi.

## ★★★★ İKİNCİ KATMAN: maske düzeltmeyi hiç görmüyordu (2026-09-03, aynı gün)

Yukarıdaki düzeltme canlı doğrulandıktan **sonra** kullanıcı hâlâ "flow
maskında bazı çukurlar düz ve o alana inen flow uçları düzleşiyor" dedi. Ölçüm
tek çağrıda söyledi:

    terrain.flow_authority -> { "erosion_unwired": true,
                                "discharge_measured": false,
                                "source": "derived_erosion_unwired" }

`FlowMaskNode::compute` iki yolludur: `Discharge` girişi bağlıysa ölçülmüş alanı
kullanır, değilse **kendi** priority-flood'unu koşar:

```cpp
const float routedHeight = std::max(filledHeight[nIdx], centerFilled + eps);
```

Kardinal ve çapraza **aynı eps** — yani motordaki **dördüncü** flow-fill
uygulaması ve düzeltilmemiş olanı. Kullanıcının baktığı maske buradan geliyordu;
LEM düzeltmesi ona hiç uğramıyordu. Belirti birebir aynı olduğu için "düzeltme
işe yaramadı" gibi okunuyor — oysa iki ayrı üretici var.

### Ama kabloyu takmak da çözmüyordu — pin YANLIŞ alanı taşıyor

| pin | etiket | beyan | `compute` dönüşü |
|---|---|---|---|
| out[1] | Wear | Mask | erosion |
| out[2] | Deposits | Mask | deposition |
| out[3] | **"Flow"** | PhysicalScalar | **`sediment`** (log-normalize 0..1) |
| — | *(pin yok)* | — | `fields.discharge` ← yalnızca RGBA önizlemenin mavi kanalı |

`HydraulicErosion.Flow` → `FlowMask.Discharge` bağlantısı `flow_authority`'yi
`"measured"` yapıyor ama **sediment** ölçtürüyor: dürüst `unwired` durumundan
daha kötü bir **sahte yeşil**.

★★★ Depo bunu zaten yarı biliyordu ve yanlış sonuca varmıştı:
`TerrainNodePortPresentation.cpp`'deki legacy slot tablosu *"Sediment Flux is the
field the new Flow pin actually carries"* diyor, sonra *"Discharge does NOT --
it is published by River Hydraulics and has no equivalent here"* diye ekliyor.
Pin listesi için doğru, **çözücü için yanlıştı**: LEM `fields.discharge`'ı en
baştan beri hesaplıyordu, yayımlayacak yeri yoktu.

### Yapılan

1. out[3] **"Flow" → "Sediment"** (kural 5: anlam sessizce değişmesin diye ad da
   değişir). Kayıtlı graflar güvende: yükleyici önce stableKey arar, bulamayınca
   ve port listesi **uzamışsa** indekse düşer — eski `"flow"` anahtarı yine
   indeks 3'e, aynı veriye iner.
2. Yeni out[4] **"Discharge"**, `PhysicalScalar` + `CubicMetersPerSecond`,
   `fields.discharge` **ham** olarak (normalize DEĞİL: FlowMask alanın kendi
   min/max'ına göre sınıflandırıyor, önceden normalize edilmiş bir girdi tam da
   ihtiyacı olan ölçeği atardı).
3. `wireFlowAuthority` artık watershed pin'i yoksa **HydraulicErosion.Discharge**'a
   düşüyor, ve FlowMask kuran **her** setup'tan çağrılıyor (eskiden yalnızca
   river-network setup'ından çağrılıyor, pin 0 ise sessizce dönüyordu).
   `outputs.size() >= 5` kontrolü bir doğruluk koruması: dört çıkışlı eski bir
   node'da indeks 3 sediment'tir, onu bağlamak sahte yeşil üretirdi.
4. Legacy slot tablosunda **Discharge artık `Removed` değil, slot 4**. Aksi halde
   budamadan önce kaydedilmiş projeler o bağlantıyı sessizce düşürürdü.
5. `compact()` listeleri pin **adına** göre eşleşiyor; `"Flow"` orada bırakılsaydı
   iki pin de Products'tan optional bölüme düşer ve düzeltme **görünmez** giderdi.

★★★ Ders: *aynı isim ≠ aynı iş*, ve bu sefer ismi taşıyan pin ile onu tüketen
otoritenin **ikisi de** doğru şeyi söyleyen yorumlar taşıyordu — sadece
birbirlerine bakmıyorlardı.

★ Ayrıca bir parite açığı: IPC'den `nodes.link` var, **unlink yok**. Test için
kurulan bir kablo script'ten geri alınamıyor.

## Açık kalan

- **Motorda DÖRT ayrı flow-fill uygulaması var** ve yalnızca LEM'inki düzeltildi:
  (1) `TerrainErosionLem` CPU, (2) `terrain_flow_fill.comp`, (3) TerrainManager'ın
  watershed/macro CPU yolları (merdiven + **yüzey gürültüsü 0.03**, ki CPU
  `priorityFlood` yorumu bunu açıkça "yanlış hamle" diye işaretliyor),
  (4) `FlowMaskNode::compute`'un kendi priority-flood'u. Kablolama düzeltmesi
  (4)'ü artık ölçülmüş alana yönlendiriyor, ama **kablosuz durumda hâlâ yalan
  söylüyor**. Dördünü tek bir servise indirmek ayrı bir iştir.
- **IPC'den link silinemiyor** (`nodes.link` var, unlink yok).
- **Kıvrılma (menderes) hâlâ türeyen bir davranıştır**, modellenmiş değil.
  Yanal aşınma / kıyı göçü yok; sinüozite `useBedMax` avülsiyon + çökeltme
  eşleşmesinden geliyor. Düzlükte gerçek menderes isteniyorsa sıradaki iş
  **yanal aşınma**dır, bu ramp değil.
