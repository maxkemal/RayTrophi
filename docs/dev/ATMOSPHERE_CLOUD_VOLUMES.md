# Bulut hacimleri: sütun profili + arka plan katmanları + hücre hacmi (Atmosfer Faz 9)

> **Durum:** AKTİF — 2026-10-14. §5 kararları öneri yönünde onaylandı.
> V0+V1+V2 CANLI PASS, V2b–V2e canlı (2026-10-14); V3 yazıldı, derlenmedi. Yerine geçeceği yapı:
> `ATMOSPHERE_CELL_CLOUDS.md` §1 fırtına kabuğu (parti 14 yüzey modeli).
> Kabuk, V5 kabul edilene kadar yan yana yaşar, sonra sökülür (CLAUDE.md
> kural 5).

## ★ DEVİR NOTU (her parti sonunda güncellenir)

- **Kodda (derlenmedi, 2026-10-14):** V0 + V1.
  - V0: `atmosphere::columnLevels()`. `stormColumnSpan` artık onu
    çağırıyor; `get_clouds.column_levels` salt okunur.
  - V1: `CloudState.volumes`, GPU paketinde `volume` + `volumeSlots[16]`
    (CloudParamsGPU 17552 B), RGBA16F atlas, `cloud_volume_gen.comp`
    (TEST elipsoidi). Kabuk yuvası 3 içinde, kutunun içindeki noktalar
    hacimden okunuyor. IPC: `sample_clouds mode=volume_sdf`,
    `cloud_stats.backends[].volumes`. Panel: Storms → "Cell volumes".
- **2026-10-14 derlendi, V0+V1 CANLI PASS** (Probe-CloudVolumes, viewport):
  LCL/tropopoz eşit; 8 yuva, 6 hacim; yükseklik 8252/8259 m, genişlik
  3373/3375 m (±2 voksel); iç sönüm sabit 0.261 = 0.261; kapatınca 5 taban
  noktası bit bit aynı. Göz: test elipsoidi "Arrival gemisi" gibi
  (beklenen; şekil V2'de). Panel hatası: hava/şimşek yaması `volumes`'u
  göndermiyordu, kutu geri kapanıyordu → düzeltildi (yalnız C++).
- **2026-10-14 akşam YAZILDI, derlenmedi: V2 + RT bant düzeltmesi.**
  - V2 üretici: gövde ve turret kabarcık zincirleri, düz LCL tabanı,
    duvar bulutu, tropopoz örsü (lifli), eğim. Turret'ler yeni
    `cloud_turrets.glsl` ile hücre haritasıyla ortak. Yuva paketi 96 B
    (CloudParamsGPU 18064 B); G kanalı = detay türü.
  - RT bant kökü: kolon aralığı ~400 km, bütçe adımı ~460 m, jitter ise
    27 m'ydi. Düzeltme: kolon aralığı hücre haritası karesine kırpıldı
    (`cloudStormMapSpan`); jitter gerçek adıma göre; kutu içinde SDF ile
    boş alan atlama (yalnızca li == 3).
  - Probe adım 3 V2'ye göre: yükseklik bağımlılığı, düz taban, determinizm.
- **2026-10-14 gece: V2 CANLI PASS** (profil farkı 422 m, düz taban 4/4,
  sabit sönüm, kapatınca aynı). Göz: kuleler + örs çıktı. Üç kusur
  bulundu, V2b ile düzeltildi (YAZILDI, derlenmedi):
  - katman `cell.r` diskine göre çekiliyordu → kulenin alt kısmı iki kez
    doluyordu, katman tepesinde sert çizgi vardı. Artık kutu içinde hacmin
    sdf'sine göre çekiliyor (+300 m);
  - AO dikey tap'leri duvarı ve kabarcık bellerini laciverte boyuyordu.
    Hacim içinde artık `tau = σ·(−sdf)·1.5`;
  - "tespih tanesi" gövde → konik çekirdek sütun, yarım yarıçap aralıklı
    kabarcıklar, 7 alt kabarcık.
- **V2b CANLI (göz):** bel lekeleri gitti, gövde tek kütle.
- **V2c YAZILDI, derlenmedi:**
  - kuleler yer yer saydamdı: eğim (kesme) mesafeyi büyütüyordu ve SDF
    atlaması ince kısımları atlıyordu → mesafe `1 + |s|·d(eğim)/dy`'ye
    bölünüyor;
  - kullanıcı: "katman fırtınaya eskisi gibi sarılmıyor" → katman eteği
    (sdf < 2.5 km: örtü, tepe ve tür artıyor, `cell.a` ile sınırlı;
    majorant `cell.a`'yı ekliyor); katman yalnızca hacmin içinde çekiliyor.
- **V2c CANLI (göz, kullanıcı):** saydamlık GÖLGE tarafında, güneşle yer
  değiştiriyor → atlama değil ışık. Şekil hâlâ tek kule; çoklu hücre,
  parçalı çevre, farklı kabarmalar yok.
- **V2d YAZILDI, derlenmedi:**
  - saydamlık kökü: ortam ışığı ufku BAKIŞ azimutundan okuyordu → gölge
    yüzü arkadaki gök gradyanını kopyalıyordu. Artık zenit + 4 sabit azimut
    (bakıştan bağımsız; tüm katmanları etkiler). Hacim yüzeyinde SDF
    normali (gök/zemin payı) + 2 sondalı açıklık (girinti koyusu) + gri
    karşılıklı yansıma (`cloudVolumeNormal/Openness`, cloud_common.glsl);
  - üretici: 4 kuleli yan hat (inflow arka kanadı, merdiven yükseklik,
    ayrı nabız), besleyici kümülüs halkası (1.5 km ızgara, fırtınaya yakın
    daha uzun/sık), ~650 m Worley kabarmaları (±77 m), alt kabarcıklar
    duvara da iniyor ve boyları geniş dağılıyor; shader kabarcık dent'i
    80 → 120 m. Lipschitz: `(1+eğim)(1+170/650)`.
- **V2d CANLI:** kullanıcı "şekiller çok daha iyi" (ışık yönü değişik kare).
  Probe PASS (4 hücre, R 3.4–4.4 km, kutu 18.8 km): profil farkı 258 m,
  düz taban 4/4 @578 m, sönüm 0.261 sabit, kapatınca aynı. Probe hatası
  düzeltildi: `Crossings` kutu dışı (1e6) → bulut geçişini atlıyordu; taban
  kutu tabanından ~60 m yukarıda (< örnek adımı) olunca gövde çıkışı taban
  sanılıyordu. Ölçülmedi: kare maliyeti, yeniden boyama takılması.
- **V2e YAZILDI, derlenmedi** (kullanıcı: "gölge yüzü hâlâ saydam gibi";
  IPC ekran görüntüsü, kamera güneşe bakıyor): gölge yüzü gök kadar açık;
  kule tabanında kutu genişliğinde düz kahverengi levha, kutu kenarında
  kesik. Düzeltmeler: ortamın yarısı normal yönündeki gökten; etek kare
  düşüş + kırık örtü + kutu kenarı sönümü (`cloudVolumeLastEdge`); kutu
  içi katman AO tap'leri yalnız katmanın kendi yoğunluğu (tap kuleye
  düşüp gök ışığını öldürüyordu → kahverengi).
- **V2e CANLI** (kullanıcı: "çok daha doğal"; IPC karesi): kahverengi
  levha ve kutu kenarı çizgileri gitti, gölge yüzü gri + gümüş kenar,
  güverte kuleye kırık kümülüslerle kabarıyor. Karede kalanlar (V3'e
  girdi): (a) uzak kulelerin tabanında güverte hizasında soluk, düz üstlü
  bir şerit, alt kısımlarından ufuk seçiliyor; (b) kuleler çok eğik,
  duman sütunu gibi (kesme/eğim payı büyük); (c) örs çevresinde havada
  kopuk küçük parçalar.
- **V3 YAZILDI, derlenmedi:** kutular kolonun `kCloudVolumeHangM` (900 m,
  ≥ 100 m irtifa) altına iniyor; `d.x/d.y` kutu tabanından ölçülüyor;
  marcher kolonun alt aralığını `volumeSlots[0].a.y`'ye uzatıyor. Üretici:
  sarkan taban lobları (2B Worley, ≤ 420 m), duvar bulutu ≤ 450 m, 28 scud
  (inflow yarısı, ömür boyunca içe+yukarı, sin büyüme), besleyiciler de
  içe akıyor, eğim 1.6 → 0.8, örs ≥ 250 m. Probe 3b → "taban kutu tabanı
  ile LCL arasında".
- **V3 derlendi.** Kullanıcı: gölgede ufuk altı kahve / üstü gök (iki
  backend) → **V3b YAZILDI:** hava perspektifi gök değeri ufkun üstüne
  kısıldı (`cloudAerialSky`); `cloudSegmentMayContain` kutu tabanını
  biliyor. Kullanıcı ayrıca: çap büyüyünce küçük parçalar çıkıyor ve örs
  birkaç hücreye tek şapka oluyor (açıklama: kutu büyüyünce besleyici
  halka ve örs kesilmeden görünür; örsler birleşir — MCS'de gerçekçi;
  istenirse örs yarı ekseni sınırlanır). Obje–bulut sırası: kod yolu doğru,
  teyit testi yapılmadı.
- **V3b derlendi; kullanıcı burada DURDU** ("aynı yerde fazla patinaj").
  Son düzeltme (YAZILDI, derlenmedi): şimşek yükseklikleri hücrenin kendi
  kolonuna ölçekleniyor (`CloudLightningParams.h`, `cellDepth`); peak
  düşünce açık gökte çakıyordu. V3/V3b/şimşek commit'lenmedi.
- **Sıradaki adım (kullanıcı açınca):** V3/V3b derleme → kontrol listesi → V4 (hacimsel cirrus
  bandı; örs onun kalınlaşması) / uzak kule tabanı şeridi.
- **Açık kararlar:** yok. Not: `.vdb` dışa aktarma V2'ye kaldı.
- **Dokunulan dosyalar:** AtmosphereClouds.h/.cpp, CloudParams.h,
  CloudLightningParams.h, VulkanBackend.h/.cpp, VulkanDeviceClouds.cpp,
  VulkanRTPipelineBuild.cpp (binding 37), RtApiClouds.cpp, RtPython.cpp,
  CloudMaterialUI.cpp, shaders: cloud_common.glsl, cloud_rt.glsl,
  cloud_sample.comp, raygen/cloud_raster/scene/shadow (binding define),
  YENİ cloud_volume_gen.comp; overlay JSON + üretilmiş descriptor'lar.
- **Kullanıcıyla konuşulan:** fırtına şeklini kullanıcı kendisi yeniden
  açtı. Kabuğa yama yok; fazlı ve ölçümlü yeni temel. Her kod partisinin
  sonunda bu blok güncellenir.

## 0. Neden (2026-10-14 canlı ölçüm)

Fırtına sahnesi, uzak kule karesi. Üç ayrı bulgu çıktı; yalnızca biri bu
notun konusu, ama üçü de aynı gün teşhis edildi ve birbirine karışmamalı.

1. **Kule duvarında dikey çizgiler: yapısal.**
   - Kabuk (`cloudShellDensity`) bir **yükseklik alanı**. Hücre haritası her
     (x, z) için bir tepe yüksekliği veriyor, yoğunluk o yüzeyin altındaki
     derinlik.
   - Kabarcık (±320 m) ve karnabahar girintisi (60–90 m) yüzeye **düşey
     metre** olarak ekleniyor. Tepede bu çalışıyor. Dik duvarda ise düşey
     kaydırma duvarı yerinden oynatmıyor: 3B gürültü kayboluyor, geriye
     yalnızca (x, z) değişimi kalıyor ve o da yukarı uzatılıyor.
   - Ölçüm: mature hücrede (tower 0.87) `world.sample_clouds density`
     yatay profilleri y = 3500 m ve y = 4500 m'de **nokta nokta aynı**
     (0.135, 0.175, 0.199, 0.209, …). Kule içi 2B bir alanın yukarı
     çekilmiş hali.
   - Aynı gün yazılan "kubbe karesini sakla" düzeltmesi
     (`cloud_cells.comp`) haritanın texel kırışıklarını giderdi; çizgiler
     kaldı. Yükseklik alanıyla dik duvar yapılamaz.
2. **Toz/duman rengi: malzeme, kod değil.** Ortak malzeme
   `scattering 0.14 / absorption 0.39 / density 0.23` → albedo 0.26.
   Referans yol izleyici 3 ve 24 sıçramada aynı görüntüyü verdi (0.26³ =
   %1.8). 1/0/1'e çekilince beyaz-krem ve mavi-gri gölge.
3. **Yağış görünmüyor: sahne ayarı.** `ground_fraction = 0` (elle). Her
   perde tabanda buharlaşan virga; 28×20 km ızgarada yağış her yerde 0.

Ek gözlem (kabuk modeli): kule yoğunluğu hücre örtüsüyle **ölçekleniyor**
(`towerD = … × cell.r`). Zayıf bir hücre daha küçük bir kule değil, daha
saydam bir kule oluyor. Gerçek konvektif bulutta sıvı su içeriği her
hücrede aynı mertebede (≈0.5–2 g/m³); hücre şiddeti yükseklik ve
genişliği belirler, optik yoğunluğu değil.

## 1. Gerçek fırtına: bağımsız katmanlar değil, sütunun seviyeleri

- **LCL (yoğunlaşma seviyesi) = taban.** Bölge boyunca hemen hemen
  sabit; tabanlar gerçekten düz. Hata düz taban değil, kusursuz düz
  taban ve altının boş olması.
- **Karnabahar = yükselen kabarcıklar.** Yükselme akımı tek bir sütun
  değil, art arda termik kabarcıklardır. Her biri çevre havayı içine
  çeken bir girdap halkası; yükseldikçe genişler. Doku her yönde oluşur,
  duvarda da.
- **Donma seviyesi.** Üstünde su buza döner; doku sertten lifliye geçer.
- **Tropopoz = örs tavanı.** Bulut kararlı stratosfere çarpıp yayılır,
  üst rüzgâr örsü tek yöne sürükler. Çekirdek tavanı biraz aşar.
- **Soğuk havuz → hamle cephesi → raf bulutu**, kenarında yeni hücreler.
- **Süperhücre.** Rüzgâr kesmesi akışı döndürür. Yükselme akımının altında
  duvar bulutu oluşur; içeri akan scud ve beslenme bantları ("küçük
  bulutların merkeze çekimi") ve yağışsız düz bir taban görülür.
- **Siklon** bunun sinoptik ölçekli (≈1000 km) üst düzeyidir:
  orta enlemde virgül biçimi ve cepheler, tropikte göz ve sarmal bantlar.
  Bu notun kapsamı dışında; `ATMOSPHERE_STORMS.md` §5c sinoptik arka plan.

**Tasarım dersi:** "katman" yüksekliği elle girilen bir tabaka değil,
iklimden türeyen bir seviye olmalı. Bulut bu seviyeler arasında yerel
zorlamanın izin verdiği kadar yer kaplar.

## 2. Endüstri (hafızadan, kaynakları V1 öncesi doğrulanacak)

- **Nubis 2015–2017** (Guerrilla, Horizon Zero Dawn). Bugünkü modelimizin
  atası: 2B hava haritası, 3B gürültü, tür başına yükseklik profili.
  Gökyüzü için çok iyi; kahraman ölçekte fırtına bununla olmuyor. Biz de
  aynı tavana çarptık.
- **Nubis Evolved / Nubis³ 2022–2023** (Horizon Forbidden West).
  Uçulabilen kahraman bulutlar **voksel bulut** oldu. Büyük şekil
  modellendi ya da simüle edildi, düşük çözünürlüklü bir hacme yazıldı.
  Render sırasında sınır prosedürel gürültüyle aşındırıldı; boş alan
  mesafe alanıyla atlandı. Gökyüzünün geri kalanı prosedürel kaldı.
- **Flight Simulator 2020/2024.** Gerçek hava verisiyle sürülen 3B voksel
  ızgara, üstüne detay gürültüsü.
- **Film (Houdini, Moana, Weta).** Baştan VDB: pyro yoğunlaşma
  simülasyonu ya da prosedürel hacim, sonra yol izleyici.
- **Unreal / Frostbite** katman modelinde kaldı; fırtına kuleleri yok.

Ortak çizgi: **arka plan prosedürel, kahraman bulut hacim.**

## 3. Mimari: iki ölçek, tek marcher

### 3.1 Sütun profili (iklimden, CPU)

`atmosphere::ColumnLevels { lcl_m, freezing_m, tropopause_m }`. Kaynağı
climate: yüzey sıcaklığı, nem ve lapse rate. Bugün kodda zaten var:
yağış biçimi için `precip3` (deniz seviyesi °C, lapse), tropopoz için
`stormColumnSpan`, LCL için `deriveWeather` (cloud_base_m).

- Tek fonksiyonda toplanır ve `CloudState`'e türetilmiş, salt okunur alan
  olarak yazılır. IPC: `world.get_clouds.column_levels`.
- Katmanlar yüksekliklerini elle de alabilir (bugünkü gibi). Ama rol
  verilmiş bir katman (`formation_role`) varsayılan olarak seviyesine
  bağlanır: taban rolü LCL'ye, örs/cirrus rolü tropopoza.
- Donma seviyesi bulut optiğine girer: su ile buz arasında faz fonksiyonu
  ve detay türü. Bu, `ATMOSPHERE_CELL_CLOUDS.md` §2'nin yeri.

### 3.2 Arka plan katmanları (bugünkü prosedürel model, kalır)

Stratus, kümülüs alanları, altostratus ve cirrus. Ucuz ve sonsuz;
karnabaharı zaten 3B aşındırmayla üretiyor (kullanıcının beğendiği
şablon kümülüs bu yol).

- Hacimli hücrenin ayak izinde bugünkü gibi kenara çekilir
  (`shellMask` mantığı, majorant korunur).
- **Cirrus yeniden yazılır** (V4): şu an hacimsiz ve gölgesiz bir 2B
  tabaka (`cloudCirrus`: eğik yoldan opaklık hesaplıyor). Yerine tropopoz
  altında ince, hacimli bir buz bandı gelir: liflenmiş, rüzgâra çekilmiş
  gürültü; gölge ve şimşek ışığı alır. Örs bu bandın hücre üstünde
  kalınlaşmış hali olur.

### 3.3 Hücre hacmi (yeni)

Her **görünür ve gelişmiş** hücre için bir hacim (growing/mature/
dissipating, `tower > ~0.15`).

**Format: GPU'da üretilen 3D doku atlası** (öneri; karar §5-1).

- Bir `VkImage` 3D dizisi ya da tek bir 3D atlas, hücre başına bir yuva.
  Yuva boyutu 128 × 128 × 128 (xz: hücrenin örs dahil sınır kutusu, y: LCL
  tabanının 300 m altından tropopoz + overshoot'a kadar).
  - Voksel boyutu: xz ≈ 60–120 m, y ≈ 100 m.
  - İnce ölçek (karnabahar girintisi, lif) render'da detay gürültüsünden
    gelir, Nubis'teki gibi.
- Kanallar (**RGBA16F**; RG16F storage image yazmak cihazda açık olmayan
  `shaderStorageImageExtendedFormats` istiyor):
  - **R = işaretli mesafe** (metre, ±kırpılmış). Sınır bilgisini ve boş
    alan atlamayı tek değer taşır.
  - **G = detay türü** (0 sert karnabahar / yükselen çekirdek …
    1 lifli buz / dağılan).
  - Optik yoğunluk ızgarada **yok**: içi sabit sıvı su içeriği, şiddet
    yükseklik ve genişliği belirler (bkz. §0 ek gözlem).
- Donanım trilineer filtre. NanoVDB'nin elle örneklemesine göre örnek
  başına çok daha ucuz; NanoVDB GPU'da kurulamıyor da.
- Bellek: 128³ × 8 B = 16 MB/yuva. Varsayılan 8 yuva = 128 MB, yalnızca
  hacim açıkken ayrılır (kapalıyken 2×2×1 yer tutucu). Yuva sayısı bütçe
  parametresi; GPU paketi en fazla 16 taşır (std430 dizi boyu, ölçek tavanı
  değil).

**Üretici** (compute, `cloud_volume_gen.comp`). Hücrenin **ve zamanın saf
fonksiyonu**; zaman çubuğunda ileri geri gitmek güvenli.

- **Kule = termik kabarcık zinciri.** Hücrenin bugünkü turret listesi
  (`cloud_cells.comp`'taki hash ve yaşam döngüsü) ortak bir include'a
  taşınır; kabarcıklar her turret'in yükselme yolu boyunca doğar, yükselir
  ve genişler (yarıçap ∝ yükseklik). Mesafe alanı küre birleşimidir
  (smooth-min). İki ölçek kabarcık (kabarcık yüzeyinde alt kabarcıklar)
  üreticide; üçüncü ölçek render'da aşındırma.
- **Taban:** LCL düzlemi ile kesişim. Yükselme akımının altında alçalan
  duvar bulutu, aralarda geri çekilen taban, aşağı sarkan loblar.
- **Scud ve beslenme:** tabanın 100–600 m altında küçük küreler. Zamanla
  yükselme akımına doğru içe ve yukarı akarlar (saf fonksiyon: id + yaş).
- **Örs:** tropopozda kama biçimli tabaka; üst rüzgâr yönünde uzar,
  yaşla genişler. Altında mammatus küreleri.
- **Ne zaman çalışır:** hücre parametresi ya da bulut zamanı değişince.
  Oynatmada kare başına bir yuva sırayla. 128³ × ~30 küre ≈ 60 M işlem,
  ≈ 1 ms bütçe hedefi (ölçülecek).

### 3.4 Marcher'a bağlanış

**Uygulanan (V1): fırtına kolonu yuvası 3'ün içinde, yeni yuva yok.**
Kutu, kolonun irtifa aralığını birebir kaplıyor. Bu yüzden kolonun hazır
altyapısı hacmi de taşıyor:

- aralıklar katman boşluklarında `cloudIntervals` yuva 3'ten, katman
  aralığının içinde `cloudLayerDensityResolved` toplamından geliyor;
- majorant `cloudLayerMajorant`'tan; kutu 1.5 kiremit genişletiliyor ve
  sınır `σ_su`;
- katmanlar hücrede bugünkü gibi kenara çekiliyor (`cell.r`).

`cloudShellDensity`, kutunun içindeki bir noktada `cloudVolumeDensity`'ye
dönüyor; kutu dışındaki hücreler kabukta kalıyor. Böylece bulutun bütün
ışığı aynen kalıyor: çoklu saçılım yaklaşımı, atmosfer ve aerial
perspektif, şimşek, yağış perdesi, RayFusion, gölge haritası. İlk taslaktaki
"yuva 5 + kutu aralıkları" planı, katman aralıklarıyla örtüşmeyen aralık
kuralını bozacağı için bırakıldı.

- **Yoğunluk:** `σ = σ_su × smoothstep(0, deri, −sdf + aşındırma)`.
  Aşındırma bugünkü detay dokusu: G kanalına göre karnabahar ya da lif.
  Majorant `σ_su` (sabit) olur; hücre başına kaba bir mip gerekmez.
- **Boş alan atlama:** dışarıda `adım = max(sdf − aşındırma_genliği, dt_min)`
  (sphere tracing). Bugün her örnekte hücre haritası + kubbe + kabuk mantığı
  çalışıyor; yerine tek bir trilineer okuma ve büyük adımlar geliyor.
- **Yağış:** footprint, damlanın çıktığı noktadaki hacim tabanından okunur.
  Ayrı bir iş olan yağış haritası buna bağlanır.
- **Binding:** RT (raygen, callable, closest-hit segmentleri) ve RayFusion
  (`cloud_raster`, `cloud_scene`, `cloud_shadow`, `cloud_sample`) için yeni
  bir 3D sampler. Bugünkü 35/36 ve 5/16 düzeninin yanında; ayrıntı V1'de.

### 3.5 Hacim ↔ arka plan dikişi

Hacim kutusunun kenarında hacim yoğunluğu sıfıra iner. Arka plan ise
`cell.r` ile çekilir; ikisi aynı ayak izinden türediği için boşluk ya da
çift yoğunluk olmamalı. Kabul testi: kutunun kenarından geçen yatay bir
profilde süreksizlik olmaması (V1 #3).

### 3.6 Script/IPC (kural 1)

- Durum `world.cloud_stats.backends[].volumes` altında (yeni metot yerine;
  hacim cihaz başına, satır da cihaz başına): generator_ready, atlas_slots,
  atlas_bytes, live_slots, repaints_last_update / repaints_total,
  allocations, boxes[]. Üretim süresi (ms) V2'de ölçülecek.
- `world.sample_clouds` hacimleri zaten kapsar (aynı `cloudDensity`).
  Yeni `mode = volume_sdf`: işaretli mesafeyi döndürür; şekil sayıyla
  doğrulanır.
- (V2) `world.export_cloud_volume {cell, path}` → `.vdb`. Mevcut OpenVDB yolu
  CPU'da (`VDBVolumeManager`). Hacim hem göz hem başka araçlarla (Houdini,
  Blender) incelenebilir olur. VDB üretim hattımızla da bağlantı noktası bu.
- `CloudState.volumes {enabled, slots, voxel_m}` + panel. Her alan
  panelden de düzenlenebilir.
- Dört dokunuş + descriptor üreticisi + overlay satırları + audit.

### 3.7 Kabuğun sökülmesi

V2 + V3 canlıda kabuktan iyi bulunursa kabuk sökülür (`cloudShellDensity`,
yuva 3, hücre haritası yaprak 1'in tepe kanalı, yaprak 2'nin örs alanları).
Hücre haritasında yalnızca örtü, yağış ve majorant kalır. Eski
`storms.column` alanı silinir; anlamı değiştirilmez (kural 5).

## 4. Fazlar (her biri ayrı parti, her birinin aç/kapa düğmesi var)

| Faz | İçerik | Kabul (sayıyla önce) |
|---|---|---|
| **V0** | Sütun profili (§3.1), IPC'de salt okunur | `column_levels` LCL'si `deriveWeather` tabanıyla, tropopozu `stormColumnSpan` ile aynı |
| **V1** | Hacim altyapısı: atlas, yuva yönetimi, marcher (kolon yuvası 3 içinde), IPC, panel. Üretici yalnızca **test şekli** (elipsoit) | `volume_sdf` analitik elipsoitle ±1 voksel; kutu kenarında yoğunluk süreksizliği yok; kapalıyken görüntü bit bit aynı |
| **V2** | Kule üreticisi: kabarcık zinciri, LCL tabanı, tropopoz örsü | 3500 m ve 4500 m profilleri **farklı** (bugün aynı); duvarda dikey şerit yok (göz) |
| **V3** | Taban: duvar bulutu, sarkan loblar, scud, merkeze akış | Scud konumu zamanla yükselme akımına yaklaşıyor (iki zamanda `volume_sdf`); taban altı 300 m'de yoğunluk > 0 |
| **V4** | Hacimli cirrus bandı; örs = bandın kalınlaşması | Cirrus gölgesi zeminde ölçülüyor (`cloud_shadow`) |
| **V5** | Kabuk sökülür (§3.7) | Audit + aynı sahne karesi; RT örnek süresi düşüyor ya da aynı |
| V6 (sonra) | Gaz çözücü aynı atlasa yazar: yoğunlaşma + gizli ısıyla simüle edilen hücre | — |

## 5. Kullanıcı kararları (2026-10-14: dördü de öneri yönünde onaylandı)

1. **Format:** GPU'da üretilen 3D doku atlası (öneri) mi, NanoVDB mi?
   - Atlas: donanım trilineer, GPU'da üretim kolay, hücre başına sabit
     bellek.
   - NanoVDB: seyrek ve içe/dışa aktarmaya hazır, ama GPU'da kurulamıyor ve
     örnekleme elle yapılıyor.
   - Öneri: render ve üretim atlasla, VDB yalnızca dışa aktarma ve
     (ileride) içe alma için.
2. **Voksel boyutu:** xz 60–120 m, y 100 m. Daha ince, yakın plan için
   daha iyi ama yuva başına bellek kübik büyür.
3. **Yuva bütçesi:** varsayılan 8 (64 MB).
4. **Sıra:** V0 → V1 → V2. Yağış haritası (maliyet) bağımsız; araya
   girebilir.

## 6. Riskler

- **Şekil iterasyonu yine gerekecek** (V2). Fark: artık doğru temel
  üzerinde. Gerçek 3B mesafe alanı var; duvar sorunu yapısal olarak yok.
- **Bant sınırı ve kutu kenarı dikişleri:** §3.5, V1 kabulünde ölçülür.
- **Üretim maliyeti** oynatmada: kare başına bir yuva; `world.cloud_volumes`
  süreyi raporlar.
- **Kabuk ile hacim yan yana** yaşarken çift yoğunluk: hacim açıkken kabuk
  o hücrede kapanır (tek anahtar).
