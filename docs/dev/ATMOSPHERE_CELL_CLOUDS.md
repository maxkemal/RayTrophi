# Fırtına hücresi bulutları: tam sütun yapı ve yoğunluğa bağlı optik (Atmosfer Faz 6)

> **Durum:** AKTİF — 2026-10-12. Kullanıcı isteği: "basınç alanlarındaki
> bulutlar tam doğal değil; bu alanlar katman yüksekliklerini ve her katmanı
> kullanmalı, yoğunluğa göre scatter/absorption ayrılmalı". §6 kararları
> öneri yönünde onaylandı. §1 fırtına kabuğu: parti 14'ün yüzey modeli
> esas. 15, 15b ve 16 canlıda daha kötü çıktı; 16b ile geri alındı (§8.1),
> derlenmedi. §2 optik sonra. Yol haritası: `ATMOSPHERE_STORMS.md` §8.

## 0. Bugün neden doğal değil (kod okuması)

- **Hücre katmanı aşamıyor.** Hücre her katmana yalnızca "örtü tabanı" ve
  "tip tabanı" olarak yazılıyor (`cloud_common.glsl`, `cell.r`/`cell.g`).
  - Kule, bulunduğu katmanın tepesinde kesiliyor. 1.5–2 km'lik bir kümülüs
    katmanında Cb, 2 km'lik bir blok oluyor.
  - Gerçek bir Cb, LCL'den (~1 km) denge seviyesine, yani tropopoza
    (10–14 km) kadar tek bir sütun.
- **Katmanlar birbirinden habersiz.** Üst katmana (Ac/Ci) örs düşmüyor.
  Alt katmanda duvar bulutu (wall cloud), raf bulutu (shelf) ya da yırtık
  taban bulutu (scud) yok.
  - `validateClouds` örtüşen katmanları zaten reddediyor.
  - Bağlı fırtına bölgeleri (`formation_role` base/tower/anvil +
    `formation_group`) var, ama hücreleri değil kendi kafes noktalarını
    kullanıyor.
- **Optik katman başına sabit.** `extinction_per_m`, malzeme ve
  `droplet_diameter_um` (tek g) her yerde aynı.
  - Gerçekte kule tabanı su damlası, tepe ve örs buz kristali.
  - Yağış çekirdeğinde iri damla ve graupel var, yani daha karanlık ve daha
    ileri saçılım.
  - Damla boyu (etkin yarıçap), sıvı su miktarıyla (LWC) büyüyor.

## 1. Model: hücre kendi sütununu taşır ("fırtına kabuğu")

İki seçenek var:

| | A: katmanları hücreye göre bük | B: ayrı fırtına kabuğu (öneri) |
|---|---|---|
| Ne | Her katmanın tepe/taban/rolü hücre haritasından değişir | Hücrelerin olduğu yerde dolu, başka her yerde boş bir sanal katman: taban = LCL, tepe = denge seviyesi + aşma |
| Katman sınırları | Kabuklar sabit kalmalı (march sınırı); kule yine kabukta kesilir | Kabuk troposferi kapsar; boşluk atlama majorantı hücre haritasından (A kanalı) gelir, hücre dışında maliyet sıfır |
| Örtüşme | Kullanıcı katmanlarıyla çakışır | Hücre içinde kullanıcı katmanlarının yoğunluğu `1 − hücre maskesi` ile söner; çift yoğunluk olmaz |
| Örs, duvar bulutu, raf bulutu | Katman rolleriyle sınırlı | Sütunun kendi yükseklik profili, hepsi bir yerde |

**B'nin yükseklik profili** (hücre evresine göre, `h` = taban → tepe):

- **Taban (0–0.1):**
  - Yağışsız taban updraft altında, yağış çekirdeğinin dışında.
  - Duvar bulutu: updraft altında taban 100–300 m iner.
  - Scud: yağış perdesinde yırtık parçalar.
- **Kule (0.1–0.75):** Parti 11'in updraft kuleleri, artık tam yükseklikte.
  Girdap aynı.
- **Örs (0.75–1):**
  - Örtü yükseklikle genişler ve rüzgâr kesmesi yönünde uzanır.
  - Altında mamatus: aşağı sarkan keseler. Bu, örsün alt yüzünde ters
    yönde kabarma olarak yapılır.
- **Aşma tepesi:** En güçlü kule örsün 0.5–1 km üstüne taşar.
- **Raf / arcus bulutu:** Hamle cephesi yayı boyunca alçak, yatay bir
  silindir. Konum ve güç `ATMOSPHERE_LOCAL_WINDS_FOG.md`'deki çıkış
  rüzgârından gelir.

Kabuk, katman dizisinde 4. bir yuva olarak durur (`CloudLayerG` aynı
yapıda). Böylece march, gölge, yağış ayağı ve şimşek kaynağı ek kod yolu
olmadan çalışır. Bedeli: kabuk boyunca uzun boş march'lar. Majorant
segmentleri bunu atlamalı, ölçülecek.

## 2. Yoğunluğa ve faza bağlı optik

Bugün örnek başına sabit olan dört değer, örnek başına türetilir:

| Değer | Neyden | Görünüm |
|---|---|---|
| Faz (su/buz) | sıcaklık: donma seviyesi (`freezing_level_m`) ile −20 °C arası karışık, üstü buz | kule tepesi ve örs lifli, "camsı"; buz faz fonksiyonu (cirrus'unki) |
| Etkin damla boyu | LWC ∝ yoğunluk × yükseklik (adiabatik) | çekirdekte iri damla: kütle başına daha az sönüm, daha yüksek g (gümüş kenar azalır, çekirdek koyulaşır) |
| Saçılım / soğurma | tek saçılım albedosu: bulut ~0.9999, yağış çekirdeği ve graupel biraz düşük | yağmur perdesi ve çekirdeği gri-koyu; "kirli" taban seçeneği (toz, duman) |
| Sönüm ölçeği | buzda kütle başına düşük | örs yarı saydam kenarlı |

Malzeme sistemi değişmiyor: `CloudMaterial`, temel değerler ve sanatçı
çarpanı olarak kalıyor. Fiziksel modülasyon onun üstüne çarpan olarak
geliyor ve bir anahtarla kapatılabiliyor. Maliyet örnek başına birkaç
ALU işi. Ancak çoklu saçılım oktavları g'ye bağlı, bu yüzden g'nin örnek
başına değişmesi gölge march'ını da etkiler. Ölçülecek.

## 3. RayFusion paritesi

Aynı `cloud_common.glsl`, aynı kabuk. RayFusion interleave'i ve geçmişi
değişmez. Gölge haritası kabuğun gölgesini de alır: örs, yere geniş bir
gölge düşürür.

## 4. IPC (kural 1)

- `world.set_clouds {storms: {column: {enabled, overshoot_m, anvil_spread,
  wall_cloud, mammatus}, optics: {physical: bool}}}`
- `world.sample_clouds` her nokta için `phase` (water/ice/mixed),
  `effective_radius_um` ve `albedo` döndürür. Kabul testi bunlar üzerinden.

## 5. Kabul

- Bir olgun hücrenin yoğunluk profili tabandan tropopoza kesintisiz
  (`sample_clouds` dikey tarama). Örs genişliği kule genişliğinin ≥ 2 katı,
  rüzgâr kesmesi yönüne kaymış.
- Hücre içinde kullanıcı katmanının yoğunluğu ≈ 0 (çift sayım yok).
- Donma seviyesinin üstünde `phase = ice`.
- Hücre yokken kare süresi aynı (kabuk boş, march girmiyor).

## 7. Parti 13 uygulaması (§1)

- **Kabuk = GPU katman yuvası 3** (`CloudParamsGPU.layers[4]`,
  `packStormColumn`).
  - Taban: en alçak etkin katmanın tabanı eksi duvar bulutu payı
    (300 m × `wall_cloud`).
  - Tepe: `column_top_m` + 800 m aşma payı.
  - Görünüm (gürültü ölçekleri, erozyon, malzeme, faz) ev sahibi katmandan
    kopyalanır.
  - `f.w = 1` kabuk bayrağıdır.
- **Çakışma tasarımı:** Kabuk katmanlarla örtüşür, ama march aralıkları
  ÇAKIŞMAZ. Bu, delta tracking ve önden arkaya birleştirme için şart.
  - Kabuk yalnızca katmanların kapsamadığı irtifa boşluklarında kendi
    aralığını alır (`cloudIntervals`).
  - Bir katmanın aralığında o katmanın yoğunluğu kabuğu da taşır
    (`cloudLayerDensityResolved` = `cloudLayerOwnDensity` +
    `cloudShellDensity`); majorantlar toplanır.
  - `cloudDensity` her kaynağı bir kez sayar.
- **Hücre içinde katmanlar çekilir:** örtü × (1 − maske(cell.r)). Hücre
  artık katman örtüsünü yükseltmez.
- **Hücre haritası 256×768.** Üçüncü yaprak:
  - R: örs örtüsü (rüzgâr yönünde büyüyen elips)
  - G: duvar bulutu
  - B: hamle cephesi için ayrıldı
  - A: örs majorantı. Elips mesafesi 1/min(yarı eksen) Lipschitz olduğu
    için genişletme analitik.
- **Yükseklik profili:**
  - Kule tepesi `flow.z`'den: < 0.75 ise tropopoza oranlı, üstü aşma.
  - Örs tropopozun ~1 km altında. Altında mamatus keseleri (gürültü
    eşiği).
  - Örste dikey ölçek ×2.2 (yassı) ve lifli erozyon.
- **Rüzgâr kesmesi:** Kabukta eğim 3 km yükseklikle sınırlı.
  Troposfer boyu karesel eğim kuleyi yatırıyordu.
- **Kapalı** (`storms.column = false`): eski davranış, hücreler katmanları
  boyar. Kabuk canlı doğrulanınca bu yol sökülecek (kural 5, karar 3).

### 7.1 İlk canlı bakış ve düzeltmeler (parti 13b)

Varsayılana yakın ayarlarla alınan viewport görüntüsünde dört kusur
görüldü; hepsi düzeltildi:

1. **Örs uçan daire gibiydi.** Sabit kalınlıkta bir bant dikey kenarlı bir
   disk çiziyordu ve kuleden kopuktu.
   - Örs artık bir kama: kulenin üstünde ~1.5 km kalın, rüzgâr yönündeki
     uca doğru incelir; alt yüzü uzaklıkla yükselir, tepesi düz.
   - Örtüsü çekirdeğe sabitlendi: örs kulenin tepesinden büyür.
2. **Kuleler iğne ve koni gibiydi.** Dar kulelerde tepe profili yarıçap
   boyunca düz bir eğriyle (smoothstep) sönüyordu.
   - Kule artık kubbe (`sqrt(1 − x²)`) ve 0.3–0.48 r genişliğinde.
   - Gövde de kubbe.
3. **Alt bant parlak beyazdı ve katman tepesinde keskin bir çizgi
   vardı.** Katman aralığındaki bir sütun örneğinin ortam ışığı örtülmesi
   (AO) yalnızca katmanın içine bakıyordu, üstteki kuleyi görmüyordu.
   - Hücre içindeki her örnek AO'yu artık sütunun kendi açıklığı
     üzerinden ölçüyor (her yönde en fazla 3 km).
4. **Şimşek yalnızca yataydı** ve ince ev sahibi katmanın içinde kalıyordu.
   - Sütun varken flaşlar sütunun yük bölgelerinden üretiliyor:
     - %50 dikey bulut içi: ana negatif yük (sütunun %30–45'i)
       → örs altındaki pozitif yük (%72–87)
     - %20 tabana inen
     - %30 örs gezgini: yatay, rüzgâr yönünde
   - Yere düşenler de ana yük bölgesinden başlıyor.
   - `kind = cloud_vertical` (stage.z = 3). Dallar kanal boyunca açılıyor,
     iki uçta yatay örümcek ağı var.

### 7.2 Kule bir YÜZEYDİR (parti 13c, kullanıcının referans fotoğrafı)

Referans, kümülüs konjestus fotoğrafı:
- neredeyse dikey duvarlar
- basamaklı, yuvarlak taçlar
- yüzeyin her yeri birkaç ölçekte sert karnabahar kabarcıklarıyla kaplı
- yoğun bir iç ve düz bir taban

"Örtü × gürültü" (katman modeli) yüzeysiz, yumuşak pamuk koniler
çiziyordu. Kabuk kulesi artık şöyle kuruluyor:

- **Yüzey:** kule kubbe alanının (sqrt) belirlediği tepe yüksekliği.
  Büyük loblar (Worley oktavları, ~1.8 km ölçek) yüzeyi ±320 m YUKARI
  iter. Kubbe alanı gürültü vektörüyle ±380 m kaydırılmış noktadan
  okunur, bu da duvarları YANA iter.
- **Yoğunluk:** yüzeyin altındaki derinliğin 70 m'lik bir smoothstep'i ×
  tabanda 120 m'lik bir yumuşama × `cell.r`. `cell.r` çarpanı majorant
  sınırını korur.
- **Detay:** yalnızca çıkarır. Küçük kabarcıkların arasında yüzey en
  fazla 90 m içeri girer (~320 m ölçek). Yükselen kulede 60 m.
- **Ölçek bağımsızlığı:** kule ölçekleri ev sahibi katmanın gürültü
  ölçeğinden bağımsız. İnce stratokümülüs ev sahibi artık kuleyi
  "kabarcıklı" yapmaz.
- **Örs:** örtü ve gürültü modelinde kaldı (lifli buz).
- **Turret'sız hücre:** `turrets 0` iken hücre tek bir kubbedir; artık
  kenara kadar düz bir plato çizilmiyor.

### 7.3 Devin çevresi (parti 15)

Canlı bakış: parti 13+14 derlendi. Karnabahar kule, örs ve büyük biçim
iyi. Kullanıcının saydığı eksikler (referans fotoğrafı):
- devin altında ve çevresinde küçük bulutlar
- çevre katmanlarla kule arasında bir bağlantı
- kulenin çevresinde ince, yassı stratus tabakaları

Eklenenler (kabuk yoğunluğunda; her terim bir örtü okumasıyla sınırlı,
majorant korunuyor):
- **Uydular:** hücre haritası 3. yaprak B kanalı = hücre çevresinde bir
  halka (0.8–2.2 r), giriş tarafında güçlü.
  - Halkada küçük kümülüs kubbeleri ve fractus var. Giriş tarafındakiler
    daha yüksek: kanat hattı (flanking line).
  - Majorant: halka d = 1.25 çevresinde tek tepeli, bu yüzden
    genişletilmiş maksimum [d − e, d + e] aralığında 1.25'e en yakın
    noktadadır (A kanalı).
- **Velum:** iki nemli seviyede (troposferin %30 ve %55'i, ±150 m)
  kuleye yapışık, 60–130 m kalınlığında, yırtık, yarı saydam (0.45)
  yassı tabakalar. Uydu halkasının üstüne uzanır, kuleyi çevreye bağlar.
- **Scud:** tabanın 380 m altına kadar, yağış (cell.b) olan yerlerde
  yırtık parçalar.
  - Kabuk artık ev sahibi tabanın 400 m altından başlıyor
    (`kStormBelowBaseM`).
  - Duvar bulutu tabanı 300 m × `column.g` kadar indiriyor.

### 7.4 Parti 15b: ayrı parçalar söküldü, tek bulut

Canlı bakışta parti 15, öncekinden KÖTÜ çıktı. Görülenler:
- velum tabakaları kuleyi delen düz, tek düze diskler
- uydu halkası tabanda düz bir etek diski
- parçalar zaman zaman ana buluttan kopuk

Kullanıcının tarifi (doğrusu): katmanlar ayrı ayrı değil, tek bir bulut
üretimi. Gövde yükselir; alt alan genişler ve parçalanarak çevre
bulutlara bağlanır.

Ders: Ayrı bir şekil fonksiyonuyla eklenen "parça" (disk, kubbe), komşu
katmanın dokusuyla hiç akraba değildir. Gözle ayrı bir nesne olarak
okunur. Çevre bulutu, çevredeki katmanın KENDİSİ olmalı.

Uygulama:
- **Söküldü:** velum, uydu kubbeleri, scud.
- **Kaide:** kule, hücre genişliğinde (`cell.r`, ~1.15 r) parçalı bir
  kümülüs kütlesine oturur. Bu kütle çekirdekte dolu, kenarda kırık;
  tepeleri dışa doğru alçalır (400–2200 m × `cell.r`). Gövde aşağıda
  genişler.
- **Halka = ev sahibi katman:** halka bölgesinde (3. yaprak B) katmanın
  kendi örtüsü, gelişimi ve türü (≥ 0.66, kümülüs) yükselir. Çevre
  bulutları katmanın dokusunu korur, hücreye doğru yoğunlaşıp uzar ve
  kaideye bağlanır.
  - Katman majorantı kabuk açıkken `column.a`'yı (genişletilmiş halka)
    da kapsar.

**Döşeme (kullanıcı: "aynı desen birbirini takip ediyor").** Kule
kabarcıkları dokunun döşeme periyodunda okunuyordu: kabarcıklar 320 m,
loblar 1.8 km. 10 km'lik bir kulede aynı desen sıra sıra tekrar ediyordu;
örsteki "beyin" deseni de aynı sebepten. Düzeltme:
- Bütün kabuk şekil okumaları ~7 km ölçekte bir 3D bükmeden geçiyor
  (±900 m).
- Kabarcık oktavı döndürülmüş bir eksen takımından okunuyor (Quilez
  ortonormal matrisi) ve ölçeği 410 m oldu.

Yeterli olmazsa sıradaki adım: detay dokusunu 64³'ten 128³'e çıkarmak
(+6 MB), ya da ayrı bir düşük frekanslı değişim dokusu eklemek.
cuFFT/CUDA çözüm DEĞİL: sorun dokunun periyodu, üretim yöntemi değil.
Vulkan birincil yol; doku zaten compute shader'da üretiliyor.

## 8. KARAR (2026-10-12): hücre bir şekil değil, yerel hava durumudur (parti 16)

**Neden.** Parti 13–15'in her denemesi daire, koni, disk ya da plaka
üretti. Kullanıcı: "alt tepe neden dairesel disk, rastgelelik yok; bu
yapıyla ilerlemek imkânsız".

**Kök neden.** Hücreyle ilgili her şey, merkeze olan uzaklığın düzgün bir
fonksiyonuydu: zarf, kubbe, kaide, halka, örs elipsi. Gürültü yalnızca bu
dairelerin yüzeyine doku ekliyordu, ana hatlarını değiştirmiyordu.
Gerçek bulutta ana hattı gürültü çizer; fırtına yalnızca yoğunluğu ve
yüksekliği değiştirir.

**Karar** (kullanıcı onayladı):
- Fırtına alanı yerel bir hava durumudur: nem/örtü, kararsızlık/tür,
  yükselme/tepe ve kesme/örs yönü.
- Bulut, mevcut katman üreticisinden bu yerel değerlere göre çıkar.

**Parti 16:**
1. **Bükülmüş hücre alanları** (`cloud_cells.comp`). Her hücre alanı,
   fraktal bükülmüş koordinatta (6 km ±1800 m, 2 km ±600 m) ölçülen
   uzaklıktan hesaplanır. Bükme bulut sürüklenmesine sabitlenir
   (`CloudCellsGPU.frame`). Tepelere 1.5 km ölçekte ±%15 kaos eklenir.
   - Majorantlar bükülmemiş uzaklık − (dilation + |warp| sınırı) ile
     hesaplanır. Güvenli bir sınırdır, ikinci geçiş gerekmez.
2. **Kabuk = katman üreticisi** (`cloudLayerOwnDensity`, li = 3, sütun
   modu):
   - **Taban:** ev sahibi taban; duvar bulutu payı kadar iner.
   - **Örtü:** yalnızca hücre alanları.
   - **Kule:** `cell.r` yerel tepeye kadar (son %6'da söner). Yerel tepe,
     bükülmüş kule alanından gelir.
   - **Örs:** örs alanı, tropopoz altında kalınlığı alanla değişen bir
     tabaka.
   - **Tür:** ≥ 0.7 (kule kümülüs .. Cb).
   - **Şekil:** ev sahibinin fazı ve gürültüsü; 7 km'lik 3D bükme
     döşemeyi kırar.
   - Örtü ve majorant aynı kalır (`max(cell.a, column.a)`).
3. **Söküm:** el yapımı kubbeler, kaide, uydu diskleri, velum, scud ve
   kabuk kule yüzey modeli (kural 5).

**Sırada (parti 17):**
- örs ve soğuk havuz, alandan ve kesme yönünden
- hücrelerin sıcaklık/nem sapması olarak iklim ve basınca bağlanması
- yerel sis ile rüzgârın aynı alandan beslenmesi

### 8.1 Canlı sonuç: parti 16 GERİ ALINDI (parti 16b, 2026-10-12)

Kullanıcı derledi ve baktı:
- Ana kule dağ gibi çıktı. Parti 14'ün karnabahar detayları kayboldu.
- RT modda maliyet çok arttı (yol izleyici kapalıyken bile).
- Kullanıcının hükmü: parti 14'ün ana şekli daha iyiydi.

Geri alınanlar (kural 5, ölü yol bırakılmadı):
- `cloud_cells.comp` parti 14 haline döndü: bükme ve `frame` yok, uydu
  halkası yok. 3. yaprak B yeniden "reserved".
- Kabuk yeniden yüzey modeli (§7.2, 13c): kubbe alanı tepe yüksekliği,
  billow tepeyi ve duvarı iter, detay yalnız oyar.
- Katman üreticisinin sütun modu ve nem halkası söküldü. Hücre içinde
  katman yine kenara çekilir; çevrede kendi örtüsü kalır.
- `storms2.z` yeniden duvar bulutu payı (`kStormWallCloudM` × wall_cloud).
  `CloudCellsGPU` 32 B başlığa döndü.

Korunanlar:
- 15b'nin döşeme düzeltmesi, yalnız şekil okumalarında: 7 km'lik 3D
  bükme ±900 m ve döndürülmüş kabarcık oktavı. Kabarcık ölçeği parti
  14'teki gibi 320 m.
- Parti 14 yağış tipleri ve parti 15 graupel payı.

**Ders.** Kabuğa ya da katmana eklenen ayrı biçimler (15: disk, 15b:
kaide) ve kabuğu katman üreticisine çevirmek (16: dağ) denendi. Üçü de
parti 14'ün yüzey modelinden kötü çıktı. Devin çevresindeki küçük
bulutlar, üst katmana bağlantı ve ince stratus için yeni bir girişim,
bundan sonra parti 14'ün yüzeyine DOKUNMADAN yapılacak. Her adım ayrı
ayrı açılıp kapatılabilecek ve canlı görülecek.

## 6. Kararlar (2026-10-12: hepsi öneri yönünde onaylandı)

1. A mı B mi? Öneri: **B** (fırtına kabuğu).
2. Fiziksel optik varsayılan açık mı? Öneri: açık, panelde kapatılabilir.
3. Mevcut `formation_role` ve `formation_group` (elle kurulan fırtına
   bölgeleri) kalsın mı? Kabuk aynı işi hücrelerle yapıyor. Öneri: kabuk
   canlı doğrulanınca sök (kural 5).
