# Arazi sekli: Noise Generator manzara degil FRAKTAL uretiyordu

> **Durum:** AKTİF — 2026-08-29. Birinci tur derlendi (0 FAIL) **ama test tek
> yonluydu ve arazi hedefi asarak bozuldugu halde yesil kaldi**. Ikinci tur
> (kaya + Math operandi + cift yonlu test) derlendi ve **canli arazide
> dogrulandi**. Ucuncu tur (makro-mikro uyum olcusu) yazildi, **DERLENMEDI**. Kabul testi `scripts/terrain_landform_shape_check.py`,
> kontrol listesi `NEXT_BUILD_CHECKS.md`.

## Sikayet

> "mevcut yapı iyi ama geniş arazi yerine engebeli yoğun arazi üretiyor —
> daha seyrek dağlar, tepeler, küçük düz araziler gibi"

Bu bir zevk meselesi gibi duruyor. Degil: soylenen sey **olculebilir** ve
olculdugunde tek bir cumleye iniyor.

## Olcum (duzeltmeden once)

Kurulum: `terrain.create` 4096 m / 512, `height_scale` 1000. Grafik yalnizca
**Noise Generator -> Height Output** — erozyon yok, Surface Relief yok, cunku
soru jeneratorun kendi ciktisi hakkinda. Node varsayilanlari, Orogenic,
Feature Size 600 m, Relief 140 m, 10 bant.

| Olcu | Deger | Gercek arazide beklenen |
|---|---|---|
| duz zemin (<3 derece) | **4.4 %** | daglik bir karoda bile onemli bir pay |
| yerel kabartma p90/p10 (64 karo) | **1.92** | 4-10x (masif ve havza) |
| en dusuk 1/5 elevasyon bandi | **7.0 %** alan | cokelme ile **agir** |
| orta 1/5 bant | **39.5 %** alan | orta bant agir olmamali |
| medyan egim | **12.4 derece** | 4 km'de 140 m icin cok yuksek |
| son pencere katlamasi (1 km -> 2 km) | **+5 %** kabartma | hala buyumeli |

Uc ayri okuma, **tek** bir sey soyluyor: bu bir **homojen fraktal**, bir
manzara degil.

★ En cok bilgi tasiyan satir sonuncusu. Pencere genisligini 1 km'den 2 km'ye
cikardiginda kabartma yalnizca %5 artiyor — yani alan **ceyrek karoda zaten
korelasyonsuz**. 4096 m'lik bir arazide hicbir sey ~1 km'den genis degildi.
Relief kadranini 140'tan 1400'e cikarmak bunu duzeltmez, sadece ayni dokuyu
buyutur. "Geniş arazi yok" cumlesinin sayisal karsiligi budur.

★★ Medyan egim 12.4 derece, tum karo boyunca toplam kabartma 140 m iken.
4096 m'de 140 m ortalama %3.4 egim demek. Yerel egimin bunun **dort kati**
olmasi, kabartmanin tamaminin **yuksek frekansa** harcandiginin dogrudan
kanitidir.

## Iki kok neden

### 1. Feature Size mutlak metre, arazi boyutu degil

Node varsayilani sabit **600 m**. Arazi boyutu sabit degil. 1 km'lik bir
karoda 600 m genis bir vadidir; 4 km'lik bir karoda **yedi tekrardan biridir**,
ve o noktadan sonra karoda 600 m'den genis hicbir sey yoktur — Relief ne
olursa olsun.

Panelin tavsiyesi de bu bandi mesrulastiriyordu: *"terrain size / 2 to / 20"*.
`/20` ucu tam olarak sikayet edilen araziyi uretir.

**Duzeltme:** `autoFeatureSize` (varsayilan acik) en genis yer sekillini
arazinin kendisine gore olceklendiriyor (~yarim karo). Metre yazmak isteyen
kutucugu kapatir; panel cozulen degeri gosterir.

★ Geriye uyum tuzagi: yeni bayragin varsayilani `true`. `deserializeFromJson`
onu duz `j.value(...)` ile okusaydi, bayragi olmayan **her eski node** Auto'ya
gecer ve kaydedilmis dalga boyu sessizce atilirdi. Bu yuzden anahtarin
**yoklugu bir cevap**: `featureSizeMeters` var ve `autoFeatureSize` yoksa, o
node metre yazmistir.

### 2. Dissection her yerde yariyi gecemiyordu

Orogenic'in bilesimi soyleydi:

```cpp
height = shape * (0.50f + uplift * 0.95f) + (uplift - 0.5f) * 0.55f;
```

`shape` (parcalanmis multifraktal) carpani **hicbir yerde 0.50'nin altina
inmiyor**. `uplift` de ortasi 0.5 olan genis bantli bir fBm, yani pratikte
carpan 0.85-1.10 arasinda geziniyor. Olculen yerel kabartma orani **1.92** tam
olarak bu.

Sonuc: masif de yok, havza da yok, uzerine bir sey koyulacak duz zemin de yok.
**Her yer esit engebeli.**

**Duzeltme:** kutle organizasyonu. Uplift alani kendi **kantiline** gore
kesiliyor; kesme noktasi yazilan **Lowland alan orani** (varsayilan 0.35).
Altinda kalan alan `dissectFloor`'a duser, ustunde smoothstep ile tirmanir —
kenari olmayan yumusak bir gecis.

Ayrica **detay kabartmayi takip ediyor**: ayni cekirdek ve ayni baslangic
koordinati ile *daha az oktavli* ikinci bir alan uretiliyor ve masif gucune
gore harmanlaniyor. Ova ince oktavlari birakip **genis yuvarlak tepeler**
tasiyor; dik ulke puruzlu kaliyor. Gercek zeminin yaptigi da budur.

★ `broadField` **ayri fit edilmez**, `shapeField`'in araligiyla olceklenir.
Kendi istatistigine gore fit edilseydi uc oktav tam kabartmaya kadar buyur ve
harman "ayni yer sekli, ince detayi alinmis" olmaktan cikip "iki esit
yukseklikte arazi capraz gecisi" olurdu.

## Yeni olcum yuzeyi: `terrain.landform_stats`

Bu partinin ikinci yarisi bir duzeltme degil, bir **alet**.

Neden bir node diagnostigi degil: bir node yalnizca kendisinden **istenen**
seyi raporlayabilir. Bu deponun en pahali hata sinifi tam olarak istenen ile
olan arasindaki bosluktur. Bu yuzden `terrain.landform_stats` **pisirilmis
yukseklik alanini** olcer, hicbir jeneratorun defterini okumaz.

Yayinladigi sey:

- `relief_window_meters` / `relief_window_relief` — **merdivenin kendisi**.
  Turetilmis tek bir "landform scale" sayisi, olcum kilığında bir esiktir;
  egrinin sekli asil cevaptir, o yuzden ham hali de yayinlanir.
- `broad_growth` — son pencere katlamasinin ekledigi kabartma. Esik
  gerektirmeden okunur: 1.0'a yakinsa alan ceyrek karoda zaten korelasyonsuz.
  Olculen: Feature 600 m'de **1.05**, 2048 m'de **1.37**.
- `landform_scale_meters` — merdivenin buyume dizi. Oktav kuantali, **kaba**,
  ve yazilan Feature Size'a **esit olmasi beklenmez**.
- `flat_fraction`, `gentle_fraction`, `median_slope_deg`, `p95_slope_deg`
- `local_relief_p10/p90/ratio` — masif/havza kontrasti
- `lowland_fraction`, `midland_fraction`, `hypsometric_integral`
- `realised_hurst` — yalnizca `landform_scale_meters` **altindaki**
  pencerelerden fit edilir.

### ★★★ Bu aleti kurarken kendi olcumum iki kez yanildi

**Birincisi.** "Doygunluk = kabartmanin karonun toplam kabartmasinin %90'ina
ulastigi pencere" diye tanimlamistim. Ortalama pencere kabartmasi karonun
kuyruk-kuyruga kabartmasina **hicbir zaman yaklasmaz** (140 m'lik bir karoda
en genis pencere 82 m verdi), yani bu olcut **her arazi icin** tum karoyu
raporluyordu — hicbir sey olcmuyordu. Buyume dizisiyle degistirildi.

**Ikincisi, ve daha ogretici.** Roughness kadranini feature 600 m'de olcup
"tamamen olu" diye okudum: iddia edilen 1.03 -> 0.67 araligina karsi
gerceklesme 0.542 -> 0.490 cikmisti. Yanlisti. Merdiven feature size'in
**uzerine** cikinca egri yapisi geregi duzlesiyor ve regresyonu her ayarda
0.5'e cekiyor. Ayni olcum feature 2048 m'de yapildiginda kadran **yasiyor**:
0.79 -> 0.68.

> Bir kadranin olu gorundugu her durumda, once **aletin bandini** sorgula.

Gercek bulgu daha yumusak ve hala acik: kadran olu degil, **kalibresiz** —
0.36'lik bir aralik vaat edip ucte birini teslim ediyor. `realised_hurst`
artik olculdugu icin kalibrasyon bir sonraki turda yapilabilir.

## Kabul testi

`scripts/terrain_landform_shape_check.py` — jeneratoru tek basina kurar,
olcer, ve **kadranlari geri cevirerek** olcunun canli oldugunu kanitlar:

1. Feature Size'i 600 m'ye sabitle -> `broad_growth` cokmeli
2. `lowlandFraction` = 0 -> duz zemin ve masif kontrasti kaybolmali
3. varsayilanlari geri koy -> olcum **aynen** donmeli

★ Kontroller esiklerden daha onemlidir. Yesil bir sonuc + olu bir kontrol,
sayilarin baska bir sebeple oynadigi anlamina gelir.

## Denetim

`scripts/audit_terrain_noise_contract.py`, uc kurali da kizartilarak sinandi:

1. Noise Generator'in yazilan her alani cizilir, kaydedilir **ve** geri okunur.
   (Kaydedilmeyen bir alan `nodes.set_property`'ye de gorunmez — serilestirilen
   JSON uzerinden calisiyor. Yani "panel-only" ile "script-only" ayni satirda.)
2. Auto acikken metre atamasi olu koddur — preset'te ikisi birden duramaz.
3. Her `terrain.*` IPC metodunun ayni adli Python baglantisi vardir (bugun
   25/25; depo genelinde 66 metot bu sartı saglamiyor, o yuzden kural bilerek
   bu namespace ile sinirli).

## Sirada ne var

- **Roughness kalibrasyonu.** `realised_hurst` artik olculuyor; esleme
  egrisini teslim edilen aralikla hizala. Her mevcut araziyi oynatir, o yuzden
  ayri bir tur.
- **Erozyon presetlerinin yeniden kalibrasyonu.** Droplet parametreleri eski
  homojen tabana gore ayarlanmisti; taban artik ova/masif ayrimi tasiyor.
- **Continental'in karakteri.** Eski `plain * 0.78` terimi kaldirildi, ovanin
  kabartmasi `dissectFloor` uzerinden geliyor. Sayilar tutuyor; goz de tutmali.
- Bu duzeltme yalnizca `NoiseGeneratorNode`'a dokundu. `MountainRangeNode` ve
  `TerrainDetailNode` ayni sorunu tasiyor olabilir — **olculmedi**.


---

# Ikinci tur: asiri duzeltme, kaya, ve testin kendisi

## ★★★★ Tek yonlu esik bir test degil, circirdir

Birinci turun kabul testi **0 FAIL** verdi. Ayni build'de olculenler:

| Olcu | Olculen | Testin sordugu |
|---|---|---|
| duz zemin (<3 derece) | **56.1 %** | `> 0.15` ✔ |
| medyan egim | **2.20 derece** | sorulmadi |
| ovada yerel kabartma p10 | **1.6 m** / 134 m | sorulmadi |
| masif/havza orani | **45.75** | `> 3.0` ✔ |
| 40 dereceden dik alan | **0.00 %** | sorulmadi |

Her sinir "eskisinden daha iyi mi" diye soruyordu. **Gittigin yonde
basarisiz olamayan bir sinir olcmez.** Arazi fraktal olmaktan cikti ve
ovaya donustu; test bunu alkisladi.

Kullanici gordu, test gormedi. Butun siniral artik **aralik**.

## Kullanicinin ikinci raporu: "dağlarda sanki hiç kaya yok"

Hipotezim yanlisti. `landformErodedFbm`'in damping terimi her oktavi birikmis
egimin yuksek oldugu yerde soneltiyor, ve bunun puruzu dik zeminden
**kaldirdigini** varsaymistim. Olculen: dik besteligin puruzu yumusak
besteligin **17-23 katı**. Damping masum.

Gercek olcum ikiye ayrildi:

- **Varsayilan ayarda (140 m / 4 km) kaya OLAMAZ**: maksimum egim 25.3
  derece, 40 derece ustu alan **%0.00**. 4 km'de 140 m zaten yumusak zemindir.
  Kabul testi artik alp kabartmasinda (900 m) olcuyor — yoksa yanlis araziyi
  olcerdi.
- **Alp kabartmasinda geometri dik ama yuzey duz**: 40 derece ustu alan %18,
  ama hucre olcegindeki puruz 0.34 m. Buyuk duzlemsel yuzler. Kayayi kaya
  yapan sey **sirt keskinligi**, yumusak kubbe degil.

Bilesimdeki suclu: masif terimi (`massif * 0.90`) smoothstep'lenmis 3 bantli
bir alandi ve butun dissection kadar aralik tasiyordu. Ova ise `dissectFloor`
0.10 ile **1.6 m** kabartma tasiyordu.

**Duzeltme:**
- `dissectFloor` 0.10 -> 0.28, `upliftWeight` 0.90 -> 0.62, `shapeWeight`
  0.80 -> 0.95 (Orogenic; digerleri orantili). Bunlar **kalibrasyon** — yonu
  olcum belirledi, buyuklugu ben sectim, yargici 3. maddenin araliklari.
- ★ **Crest keskinligi artik masife bagli.** `mfRidge` sabitti; simdi masif
  gucuyle 0.40x-1.25x arasinda. Kaya yukselen ve siyrilan zeminde cikar; ova
  kendi molozuyla ortulur. Bunun icin pass sirasi degisti: masif alani
  **once** hesaplaniyor, cunku sekil onu okumak zorunda.

## Yeni olcu: kaya olculebilir hale geldi

`terrain.landform_stats` iki alan daha yayinliyor:

- `cliff_fraction` — 40 dereceden dik alan. Toprak ve moloz o acida durmaz,
  yani ciplak kaya olarak okunmak **zorunda** olan pay.
- `roughness_slope_ratio` — en dik beste / en yumusak beste hucre puruzu.

★★ **Ikisi birlikte okunur.** Hic dik zemini olmayan bir arazi de yuksek
oran verir, cunku "en dik bestelik" sadece "en az duz bestelik"tir. Tek
basina okunan oran, olcmedigi seyi onaylayan bir sayidir.

## Math node: ayni bayrak iki ayri sey soyluyordu

Dorduncu partide her terrain node'una varsayilan exposure profili verdim:
`optional` pin yalnizca bagliyken cizilir. `MathNode.B` `optional` ilan
edilmisti — ama "birakabilecegin bir degistirici" oldugu icin degil, eksikken
bir `factor` skalerine dussun diye.

Sonuc: Math **tek** soketle cikti ve ikinci operand **hic baglanamadi**.

Tarama: blanket profil alan 31 sinifta **tek** operand vakasi buydu; kalanlar
gercekten opsiyonel Mask/modifier pinleri. B zorunlu yapildi, `factor` ve
onun ikinci kod yolu sokuldu.

★ Denetim kurali 4 bunu kilitliyor: tek harfli operand pini (`A`, `B`)
`optional` ilan edilemez.

## Mountain Range: yarim goreli node

Kullanici "Gaea'daki mountain node gibi hizli arazi" dedi. O node zaten var —
ve varsayilaninda **kullanilamaz**: 4096 m karoda 420 m genislik, medyan egim
**0.01 derece**. Duz levha uzerinde ince bir kabartı.

Uzunluk zaten `1.05x terrain` diye goreliydi. **Yalnizca genislik mutlak
metreydi.** `widthMeters` -> `widthFraction`.

★★ Eski projelerin donusumu **1500 m referans karo** varsayimiyla yapiliyor.
Bu sayi bes preset'in genislik/uzunluk oranindan cikiyor: alpine 0.28 (dar
zincir), volkanik 0.51 / uzunluk 0.62 (neredeyse dairesel), col 0.22 (dar).
Besinin de kendi adina uymasi referansi belirliyor — tercih degil cikarim.
Yine de cok farkli bir karoda yazilmis proje bir kez bakilmali.

## Sirada

- `BasinValleyNode.widthMeters` (180 m, "Valley Width") ayni hastalikta
  gorunuyor — **olculmedi**.
- Mountain Range preset'leri sayisal olarak tutuyor, **gozle dogrulanmadi**.
- Roughness kalibrasyonu, erozyon presetlerinin yeniden kalibrasyonu.


---

# Ucuncu tur: makro-mikro uyumu bir OLCUDUR

Kullanici, kopuk kablolarin bir test sahnesine ait oldugunu soyleyip asil
olcutu adlandirdi: **makro-mikro uyumu**. Bu ölçülebilir bir sey.

## Altinci turun canli dogrulamasi

46 node'luk sahne, 1000 m karo, 2048 (0.49 m hucre), kabartma 110.5 m:

| Olcu | Deger | Hedef |
|---|---|---|
| kaya alani (>40 derece) | **%21.4** | 0.05-0.35 ✔ |
| dik/yumusak puruz orani | **8.39** | >2 ✔ |
| masif/havza orani | **5.82** | 2.5-15 ✔ |
| hipsometri | ova %54.7 / orta %9.0 | dip-agir ✔ |

Kaya calismasi tuttu. Duz zemin %8.8 kabul testimin 0.15 alt sinirinin
altinda, ama o sinir **ciplak jenerator** icin yazildi; burada tam erozyon
zinciri var ve erozyon dikleştirir. Yanlis uygulanmis bir esigi arıza diye
raporlamak, olcunun kendisini bozmak olurdu.

## Uyum olculdu: 16 katlik aralikta dikis yok

Merdivenden cikan yerel ustel:

| pencere | buyume | yerel Hurst |
|---|---|---|
| 4 -> 8 m | 1.670 | 0.740 |
| 8 -> 16 m | 1.750 | 0.807 |
| 16 -> 31 m | 1.801 | 0.849 |
| 31 -> 63 m | 1.793 | 0.843 |
| 63 -> 125 m | 1.796 | 0.845 |
| 125 -> 250 m | 1.673 | 0.742 |
| 250 -> 500 m | 1.440 | 0.526 |

8-125 m arasi plato **0.836 ± 0.041**. Ust uctaki dusus karo bittigi icin,
arazinin bir ozelligi degil.

## ★★★ Ama merdiven 4 m'de basliyordu

Hucre 0.49 m. **0.5-4 m arasi hic olculmuyordu** — ve Surface Relief tam
orada calisir, goz "kaya dokusu"nu tam orada okur. Alet, uyumun en cok
bozulabilecegi banda kordu.

Merdiven **2 hucreden** basliyor artik.

## Yeni olcu: `spectrum_kink`

Egim, kabartma ve hipsometri **olceklere bakmaz**. Bir detay katmani
altindaki yer sekliyle anlasmadiginda ucu de saglikli kalir — bu yuzden uyum
kendi okumasini hak ediyor.

- `spectrum_kink` — herhangi bir oktavin fit edilen ustel yasadan en buyuk
  sapmasi, `realised_hurst` ile ayni birimde. ~0.05'in altinda olcekler tek
  bir manzaradir.
- `spectrum_kink_signed` — **yonu soyler**: negatif o olcek detaydan yoksun
  (dusen bant), pozitif fazla tasiyor (yapistirilmis katman).
- `spectrum_kink_meters` — hangi olcekte. Sucluyu **boyundan** bulursun.

★ `realised_hurst` de duzeltildi: eskiden `landform_scale_meters` altindaki
her pencereyi aliyordu, ama merdivenin ust ucu **karo bittigi icin** duzlesir.
Artik karonun sekizde birinden genis pencereler fit'in disinda.

## ★★ Mountain Range gocu: calisti, ama sonucu daraltti

Kullanicinin kaydedilmis `widthMeters` degeri 420'ydi; 1500 m referansla
`widthFraction` 0.28 oldu. Arazi **1000 m** oldugu icin gercek genislik
420 m -> **280 m**. Goc dogru calisti; referans varsayimi bu karo icin dusuk
kaldi.

Bu, "cok farkli bir karoda yazilmis proje bir kez bakilmali" uyarisinin
gerceklestigi durumdur — ve neden sessiz bir donusum yerine **beyan edilmis
bir varsayim** oldugunu gosteriyor. Eski gorunum icin `widthFraction` = 0.42.

## Sirada

- **Fluvial cok dik yuzleri oyuyor.** Akarsu oyugu bir egim/alan esiginin
  ustunde durmali; o bolge yamac/termal alani. `FluvialErosion.mask` girisi
  tam bunun icin var ve bagli degil. **Olculmedi.**
- `BasinValleyNode.widthMeters` (180 m) ayni mutlak-metre hastaliginda.
- Roughness kalibrasyonu; erozyon presetlerinin yeniden kalibrasyonu.
