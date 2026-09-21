# Nükleer patlama preset'i ve gaz domain'inin katmanlaşması (stratification)

> **Durum:** AKTİF — katmanlaşma, alan kaybı ve yüzey tozu DERLENDİ ve CANLI ÖLÇÜLDÜ (2026-09-20); sonlu toz rezervi + sıkıştırılmış kare cache'i yazıldı, DERLENMEDİ. Kabul için `scripts/ipc/Test-NuclearPreset.ps1` ve `docs/dev/NEXT_BUILD_CHECKS.md`.

## Neden bu iş açıldı

Mevcut `Fireball (mushroom)` preset'i mantar şeklini **doğru mekanizmayla**
üretiyordu: güçlü termal kaldırma + vortisite. Eksik olan tek şey, yükselen
kütlenin **nerede duracağıydı**.

`sim_gas_buoyancy.comp` ve `GridFluidSolver.cpp::addBuoyancy` aynı formülü
çalıştırıyordu:

```
force = density_weight*d + heat_weight*(t - ambient_temperature)
```

`ambient_temperature` **tek bir skalerdi** ve `ParticleSimulation.cpp`'de
`0.0f` sabitleniyordu. Yani sıcak kütle yükseldikçe kaldırma kuvveti hiç
azalmıyordu: kolon **domain tavanına çarpana kadar** çıkıyor, orada yassılıyordu.

★★★ Bunun sinsi tarafı: tavana çarpmış bir kolon **doğru görünür.** Gerçek bir
inversiyon katmanı da şapkayı tam olarak öyle yassılaştırır. Yani şekil
"kutunun" sonucuydu ama bug gibi görünmüyordu — domain'i yeniden boyutlandırana
kadar. Bu, bu deponun `Volume` varsayılanı ve `fire_enabled`'dan okuyan gaz
shader'ıyla aynı sınıf: **ölçü aleti makul bir yalan söylüyordu.**

## Çözüm: potansiyel sıcaklık katmanlaşması

Yeni alan: `SimulationGridDomainDesc::gas_ambient_stratification`
(çözücüde `SolverParams::ambient_stratification`, API'de
`GasDomainSettings::ambient_stratification`, panelde **Stratification**).

```
ambient(y) = ambient_temperature + stratification * (y - domain_floor)
```

### ★ İşaret ters sezgiseldir ve gerekçesi budur

Bu **meteorolojik lapse rate DEĞİL**, potansiyel sıcaklık (θ) gradyanıdır ve
kararlı atmosfer için **pozitiftir** — "yukarı çıkınca hava soğur" sezgisinin
tersi.

Gerekçe: bu çözücüde **adyabatik soğuma yoktur**. Yükselen bir hücre
sıcaklığını olduğu gibi taşır. Bu tam olarak bir parselin θ koordinatlarında
yaptığı şeydir, ve θ koordinatlarında kararlı bir ortamın değeri **yükseklikle
artar**. Lapse rate olarak yazılsaydı `(t - ambient(y))` yükseldikçe *büyürdü*
ve kolon sonsuza kadar hızlanarak çıkardı — yani düzeltme, düzeltmeye çalıştığı
şeyi kötüleştirirdi.

### ★ Geri çağırma terimi parselin KENDİ anomalisiyle sınırlı

```cpp
excess = t - ambient;
env    = stratification * max(height_above_floor, 0);
heat_term = max(excess - env, -abs(excess));
```

`env`'i koşulsuz çıkarmak, **boş havaya** (excess == 0) yükseklikle büyüyen bir
aşağı kuvvet verirdi: domain çapında bir hayalet iniş akımı. Bu, shader'daki
`has_temperature` yedeğinin var olma sebebiyle **aynı hata sınıfı**, ama daha
kötüsü: "duman güzelce çöküyor" gibi görünür ve kimse bug diye raporlamaz.

`-abs(excess)` sınırı bunu eşiksiz/maskesiz kapatır: ortam tam olarak sıfır
hisseder, soğuk gaz stratifikasyonla daha da batırılmaz, sıcak parsel ise
aşabildiği kadar aşar ve geri oturur (tepe aşımı + geri düşme — gerçek
davranış).

### Yükseklik domain TABANINDAN ölçülür

Dünya sıfırından değil. Domain'i taşıyınca atmosferi de taşınır, ve şapka
yüksekliği `h* ≈ anomali / stratification` **yazarın seçtiği bir sayı** olarak
kalır — kutunun nerede durduğunun sonucu değil.

## İki preset, tek tarif

`NuclearCinematic` ve `NuclearPhysical`. Tek `case` bloğu, tek ölçek çarpanı
`S` (1 m / 80 m) ve tek zaman çarpanı `T`.

★ **Ayrı preset olmaları bilinçli.** "Boyut kadranı olan tek preset" her
parametreye gizli bir "hangi ölçekteyim" anlamı yükler ve hızlı olan çürür,
çünkü kimse çalıştırmaz. Sinematik olan **iterasyon için** ayarlı (22×34×22 m,
~1.5M hücre, birkaç saniye); fiziksel olan 1.76×2.72 km, **aynı hücre sayısı**.

★ Zaman `S` ile ölçeklenmez. Çözücünün saniyesi timeline'ın saniyesidir;
kimsenin oturup izleyemeyeceği bir çekim daha iyi bir çekim değildir.

★ Fiziksel preset **daha ince değil, daha BÜYÜK**. Hücre sayısını sabit tutmak
ikisini karşılaştırılabilir kılar: şapka domain yüksekliğinin farklı bir
oranına oturursa, konuşan **ölçektir, çözünürlük değil.**

### Aşamalar (zaman pencereli flow source'lar)

| # | Kaynak | Pencere | Rolü |
|---|---|---|---|
| 1 | Detonation Core | 0 – 0.05 | Patlama. `temperature=10`, `fuel=16`, `fire_expansion=1.15` şoku üretir |
| 2 | Fireball Rise | 0.05 – 0.55 | Ateş küresinin kendi yanması, zaten yükselirken |
| 3 | Stem Afterwind | 0.45 – 4.5 | **Sap.** Yalnızca RÜZGÂR — taşıdığı madde yerden kalkar |

★ Dördüncü bir kaynak vardı (`Ground Dust Skirt`) ve **kaldırıldı**: toz artık
yüzeyden şokun kendi rüzgârıyla kalkıyor. Aşağıya bak.

★ **4. aşamanın zamanlaması işin bütün püf noktası.** Sap patlama enkazı
değildir; ateş küresinin arkasından içeri çekilen havanın (afterwind) saniyeler
sonra kaldırdığı tozdur. Patlamayla aynı anda başlatılırsa sap olmaz, patlamanın
parçası olur.

★ **3. aşamaya sıcaklık vermek tek adımda her şeyi bozar:** etek yükselir ve
etek/şapka ayrımı kaybolur, sonuç tek biçimsiz bir bulut olur.

★ Enkaz partikülleri **yakıt taşımaz** (`grid_fuel_deposit = 0`). Yakıt ilk
flaşta biter. Fireball preset'indeki gibi yakıt bıraksaydı kolon sürekli yeniden
tutuşur ve sap tekrar bir **alev sütununa** dönerdi — Fireball'un tam da istediği
şey, bunun asla istemediği şey.

### Görünüm

`createExplosionPreset()` üzerine: `blackbody_intensity=60`,
`temperature_max=6000` (çekirdek turuncu diske *clip* olmasın diye), parlak ve
zayıf soğuran ileri-saçılımlı duman.

★ Bu **yoğuşma (Wilson bulutu) GÖRÜNÜMÜ**, faz değişimi değil. Gaz grid'inde nem
kanalı yok; hiçbir şey yoğuşmuyor. Doğru görünen bir görünüm iddiası, fizik
iddiası değil.

## Test edilebilirlik: iki yeni metot

Preset'ler daha önce **yalnızca panelden** kurulabiliyordu
(`scene_ui_forcefield.hpp` tek çağıran). Yirmi bağlı sayıdan oluşan bir tarifin
tek dürüst testi "çalıştır ve bulutu ölç"tür, ve o döngü ilk adımı bir insanın
düğmeye basması olduğu sürece kapanmaz.

- **`particle.add_preset`** — slug ile preset kurar (`nuclear_cinematic`, ...).
- **`gas.measure_plume`** — canlı gaz alanını ölçer: sütun sınırları, tabandan
  yükseklik, **en geniş dilim ve o dilimin yüksekliği**, tepe/ortalama ısı,
  doluluk oranı, ve `touching_ceiling`.

★★★ `measured` bayrağı süs değildir. Her uzunluk hem **gerçekten boş** bir
domain'de hem de **örneklenemeyen** bir domain'de (canlı runtime yok, hiç
adımlanmamış, density kanalı kapalı) 0 okur. Bu ayrım olmasa, bulutu izleyen bir
script "yükseklik 0" okuyup **patlama olmadı** sonucuna varırdı — makul ve
tamamen yanlış, yani en kötü cinsinden.

★★ `touching_ceiling` bu işin **ölçü aletidir**. True ise şapkanın şeklini
stratification değil kutu belirliyordur ve yanındaki bütün sayılar kırpılmış bir
bulutu tarif ediyordur.

★ En geniş dilimin **yüksekliği** şapkayla kolonu ayırır: hâlâ tırmanan bir
kolon **başından** en geniştir; oturmuş bir mantar başının **hemen altından**.
Tek başına genişlik bu ikisini ayıramaz.

## Dokunulan katmanlar

| Katman | Dosya |
|---|---|
| GPU compute | `shaders/sim_gas_buoyancy.comp` (push-const 36 → 44 B) |
| GPU ABI/dispatch | `src/Physics/ParticleSimulation.cpp`, `src/Device/SimulationComputeVulkan.cpp` |
| CPU parity (dense + sparse VDB) | `src/Physics/GridFluidSolver.cpp`, `include/GridFluidSolver.h` |
| Domain tanımı | `include/ParticleSimulation.h` |
| Serileştirme | `src/Core/ProjectManager.cpp` |
| Çekirdek API | `include/Api/RtApi.h`, `src/Api/RtApiFluid.cpp`, `src/Api/RtApiParticle.cpp` |
| IPC | `src/Api/RtIpc.cpp` |
| Python | `src/Api/RtPython.cpp` |
| Yetki | `src/Api/RtIpcSecurity.cpp` (`gas.measure_plume` → **Read**) |
| Ajan tarifi | `scripts/ipc_descriptor_overlay.json` + üretici |
| Panel | `src/UI/scene_ui_simulation_domains.cpp`, `src/UI/scene_ui_forcefield.hpp` |

**Render backend'lerine dokunulmadı ve dokunulması gerekmiyor.** Kaldırma
kuvveti *simülasyon* compute yolundadır; RayFusion/VulkanRT/OptiX/CPU dördü de
alanın **tüketicisidir**. Görünüm tarafında yeni bir özellik yok: blackbody ve
color ramp `VolumeShader::toGpu` üzerinden zaten dördüne de gidiyor.

## Henüz yapılmayanlar

- **Rüzgâr makaslaması** (yükseklikle dönen/güçlenen rüzgâr) — gerçek şapkaların
  eğilmesi ve kuyruk oluşturması bundan gelir. Force field'larda **zaman
  penceresi yok**, o yüzden kalıcı bir alan olarak eklenmesi gerekir.
- **Nem/yoğuşma kanalı.** Şu anki beyaz şapka bir görünüm ayarı.
- **Blast → yapı kuplajı** preset'te kapalı (`structural_coupling_enabled`).
  Açılması ayrı bir kalibrasyon turu ister (`structural_pressure_scale`).
- **`rt.perf` ölçümü yok**: fiziksel preset'in kare maliyeti ölçülmedi.


---

# 2026-09-20 — İLK CANLI ÖLÇÜM: katmanlaşma ÇALIŞTI, bulut yine de kayboldu

Sinematik preset derlendi ve 250 kare (24 fps, ~10.4 s) cache'lendi. Domain
canlı 124×191×124 (voxel 0.178 — profil çözünürlüğü yazılan 0.22'den yukarı
çekti, ayrı bir not).

## Doğrulanan: şapkanın yüksekliği artık fizik

`touching_ceiling` **hiçbir karede true olmadı.** Tavan 34 m, bulut 24.7 m'de
durdu. Kapak kutunun değil stratification'ın işi — bu kayıt bu dosyanın açılış
iddiasını kapatıyor.

## Bulunan: alan kaybı bulutu SİLİYORDU

| kare | ~s | top (m) | peakT | meanT | aktif hücre |
|---|---|---|---|---|---|
| 15 | 0,6 | 5,34 | **9,79** | 1,25 | 26 300 |
| 110 | 4,6 | 21,2 | 1,5 | 0,31 | 220 000 |
| 180 | 7,5 | 24,0 | 0,24 | 0,065 | **311 276** |
| 250 | 10,4 | 24,7 | **0,036** | 0,025 | **8 930** |

Kök neden `ParticleSimulation.cpp`'deki `base_params`:

```cpp
density_dissipation = (mode == Gas) ? viscosity*0.35f : 0.5f;  // preset Spark => 0.5
temperature_dissipation = 0.5f;
fuel_dissipation = 0.25f;
```

Bunlar **global**; per-domain override bloğu yanmayı, kaldırmayı, vortisiteyi,
türbülansı ve stratification'ı set ederken dissipation'a **hiç dokunmuyordu**.

Hesap tutuyor: `exp(-0.5 × 10.4) = 0.0055`; ölçülen oran 0.0037.

★★★ **Üstel sönüm düzgün solmaz, UÇURUMDAN atlar.** 311k hücre 8.9k'ya tek
seferde düşüyor, çünkü bütün bulut görünürlük eşiğini neredeyse aynı anda
geçiyor. Belirti bu yüzden "kayıp oranı" gibi değil, **"tepe soğuyup kayboldu"**
gibi okunuyor — yani semptom kök nedeni tarif etmiyor.

★★ `0.5f` bir tercih değil, **başka bir iş için seçilmiş sabit**: kıvılcımların
taşıdığı ince duman. Spark fizik modunu seçen her hibrit efekt onu sessizce
miras alıyor.

★ Ve panelde **"Flame Dissipation"** kadranı var — dissipation ayarlamak isteyen
herkesin ilk uzanacağı düğme, ve yanlış olan: alevin interaction alanını
söndürür, dumanı ve ısıyı değil. **Doğru isimli yanlış düğme.**

## Düzeltme: domain başına alan kaybı

`gas_dissipation_override` (bool) + `gas_density_dissipation` /
`gas_temperature_dissipation` / `gas_fuel_dissipation`. Kapalıyken davranış
**aynen** eskisi; alan varsayılanları eski globalleri yansıtıyor, yani kutuyu
işaretlemek tek başına hiçbir şeyi değiştirmiyor.

★ Sentinel değil ayrı bayrak: `0.0` geçerli ve işe yarar bir orandır (hiç kayıp
yok), o yüzden aynı anda "devral" anlamına gelemez. (bkz. `aperture==0` dersi.)

Preset değerleri: duman 0.012/s, ısı 0.22/s, yakıt 0.5/s.

## ★★ Yeni bağlaşım: ısı kaybı ↔ stratification

Bu ikisi **birlikte türetilmek zorunda**. Yavaş soğuyan bir sütun kaldırma
kuvvetini daha uzun korur ve **daha yükseğe** oturur. Soğumayı 0.5 → 0.22'ye
indirip stratification'ı yerinde bırakmak kapağı doğrudan tavana sokardı — yani
tam da bu parametrenin kaldırmak için eklendiği arıza.

Preset'teki yeni değer `6.5 / 26` **ölçülmedi, ESKİ koşudan ekstrapole edildi.**
`Test-NuclearPreset.ps1` artık ölçülen buluttan ima edilen stratification'ı
yazdırıyor; **yorumdaki sayıya değil ona güven.**

## Ayrıca görülen, henüz düzeltilmeyen

- **Sap ~4.5 s'de aniden kesiliyor.** `Stem Afterwind`'in `end_time`'ı. f110'da
  sap var, f180'de yok. Tasarım gereği ama geçiş sert; kaynağın sönümlenerek
  bitmesi gerek (flow source'ta ramp yok).
- **Şapka en geniş yerini 13 m'de yapıyor, tepe 24.7 m'de** (oran 0.53).
  Testin 0.45–0.98 aralığını geçiyor ama şapkadan çok "geniş orta gövde".
  Vortisite torusu daha yukarıda kurulmalı.
- **f210'da genişlik 20.1 m, domain 22 m.** Bulut yanal olarak kutuya değmek
  üzere; open boundary'den akıp kütle kaybediyor. Domain genişliği veya yanal
  yayılım gözden geçirilmeli.
- **voxel_size yazılan 0.22 yerine 0.178 okundu** — kalite profili çözünürlüğü
  yukarı çekiyor. İki preset'in "aynı hücre sayısı" iddiası bu yüzden
  doğrulanmadı.


---

# 2026-09-20 (3. parti) — toz artık KALDIRILIYOR, konmuyor

Kullanıcının isteği: *"zeminden şok dalgaları ile toz kalkmalı."* Haklı, çünkü
preset'te öyle olmuyordu.

## Kaldırılan taklit

`Ground Dust Skirt` bir flow source'tu: orijinde, yarıçapı **yazarın yazdığı**
bir halka. Şoku takip etmiyordu, araziyi takip etmiyordu, 5. karede de 100.
karede de aynıydı. Şekli doğru görünüyordu ve **hiçbir şeyin sonucu değildi.**

Aynı şekilde `Stem Afterwind` hem rüzgârı hem dumanı taşıyordu; yani sap, yerden
kalkan madde değil, tabana enjekte edilen dumandı.

## Yerine gelen: yüzey kesmesi eşiği (saltation)

`addSurfaceDust` — bir yüzey, üzerinden esen **yatay** rüzgâr
`surface_dust_threshold`'u aşana kadar hiçbir şey vermez; aştıktan sonra
**fazlayla orantılı** verir.

★ **Eşik sert olmak zorunda.** Durgun havanın tozlu zemin üstünde berrak
olmasının sebebi bu, ve şok eteğinin yumuşak bir pus değil **keskin kenarlı ve
cepheyle birlikte ilerleyen** bir halka olmasının da.

★ **Yalnızca YATAY bileşen okunur.** Normal bileşeni de katsaydık yükselen her
sütun kendi yukarı akışından toz üretirdi — "bol bol güzel toz" gibi görünen,
hiç durmayan bir geri besleme döngüsü.

★ Yüzey = domain'in taban katmanı + katı üstünde duran her hücre. Katı taraması
domain'de collider yoksa **tamamen atlanır**, yani yaygın durum O(nx·nz).
Bu bir **yüzey** geçişi, o yüzden bilerek GPU karşılığı yok: MAC alanını yükleyip
indirmek, kazandıracağı taramadan pahalı.

★★ **Bilinen basitleştirme: yüzeyin kaynağı SONSUZ.** Gerçek bir patlama bir
noktayı temizler ve sonraki rüzgârlar oradan daha az kaldırır. Burada esen
rüzgâr üretmeye devam eder, tek sınır `surface_dust_max_density`. İzlenecek
belirti: **hiç incelmeyen bir sap.**

## Preset'e etkisi

- `Ground Dust Skirt` **silindi**.
- `Stem Afterwind` artık `density = 0` — yalnızca rüzgâr. Sap, altındaki zeminden
  kalkan tozdan yapılıyor. ★ Buraya duman geri konursa **yer kuralının çalışıp
  çalışmadığı görünmez olur**: sap iki durumda da doğru görünür.
- Domain: `gas_surface_dust_enabled = true`, eşik 4 m/s (varsayılan 6'dan düşük,
  çünkü afterwind'in yer seviyesindeki girişi şoktan yumuşak ve onun da barajı
  geçmesi lazım), yield 0.6, tavan 2.5.

## ★★★ Diskler ve saçaklar: TOZ DEĞİL, SU — ve henüz yapılmadı

Kullanıcının tarif ettiği *"bazı yükseltilerde oluşan diskler"* ve *"merkez
hattındaki düşük basınçlı hızlı yükselişin çevresindeki saçaklar"* **yoğuşma
bulutlarıdır**, toz değil:

- Şokun arkasındaki **seyrelme** basıncı düşürür → nemli hava doymayı geçer →
  geçici bir küre/disk yoğuşur, basınç toparlanınca buharlaşır (Wilson bulutu).
- Yükselirken kolonun çevresine giren nemli hava **yoğuşma seviyesinde** beyaz
  bir yaka/etek yapar. Birden fazla nemli katman = birden fazla disk.
- Aynı mekanizma şapkanın beyazlığını da verir.

★ Yani üç ayrı görünüm, **tek bir mekanizma**: basınca ve sıcaklığa bağlı
doyma. Bunları tozdan yapmaya çalışmak yine taklit olurdu — ve bu dosyanın
kaldırmaya çalıştığı şeyin aynısı.

★ Trinity (çöl, kuru) zayıf, Bikini (deniz, nemli) muhteşem yoğuşma bulutları
verdi. Fark nem. Sahnedeki Physical Sky'ın `humidity` alanı bu modelin ortam
buharı için doğal kaynağı — ve o kuplaj olmadan disk "her zaman var" olur, ki
yine yanlış.

### Yapılmadan önce ÖLÇÜLMESİ gereken şey

Yoğuşma basınç düşüşünden tetiklenir. Ama bu çözücü **sıkıştırılamaz**: 
`grid.pressure` bir projeksiyon Lagrange çarpanı, pascal değil. `fire_expansion`
bir diverjans HEDEFİ verdiği için basınç alanında şok/seyrelme yapısı
**olabilir** — ama bu bir varsayım.

Bu yüzden `gas.measure_plume` artık `pressure_min` / `pressure_max` /
`pressure_min_height` raporluyor (yoğunluk eşiğinden bağımsız, **bütün domain**
taranarak — çünkü seyrelme bölgesi eşiğin altındaki havada oturur), ve
`Test-NuclearPreset.ps1` sonucu yazdırıyor.

- `pressure_min < 0` çıkarsa: gerçek bir düşük basınç bölgesi var, yoğuşma
  modeli ona bağlanabilir.
- Çıkmazsa: model başka bir sürücüye ihtiyaç duyar (yükseklik + sıcaklık), ki o
  da fiziksel olarak savunulabilir ama **Wilson küresini vermez**.

Faz B'ye bu ölçüm alınmadan başlanmamalı.

### Faz B tasarımı (yazılmadı)

İki yeni kanal: `Vapor` ve `Condensate`.
`cond = max(0, vapor - q_sat(T, p))`, gizli ısı salınımı, basınç toparlanınca
geri buharlaşma. Yoğuşuk **ayrı bir kanal** olmalı, dumana katılmamalı — bütün
mesele disklerin **beyaz**, tozun **kahverengi** olması; tek alanda birleşirlerse
ayırt edilemezler ve amaç kaybolur.

Maliyet tahmini: `fuel` kanalının izleri — çözücüde 11, GPU'da 17, kanal
bayrağında 16 nokta; iki kanal için bunun iki katı, artı yeni bir yoğuşma
çekirdeği (CPU+GPU), artı render bağlaması.


---

# 2026-09-20 (4. parti) — CANLI ÖLÇÜM: üç düzeltme tuttu, biri görünümü bozdu

Kullanıcı 160×248×160 domain'de 100 kare bake etti. Ölçüm:

| kare | aktif hücre | top | genişlik @ yükseklik | peakT | meanT | pMin @ yükseklik |
|---|---|---|---|---|---|---|
| 3 | 5 199 | 3,58 | 2,75 @ **1,86** | 6,95 | 1,10 | **−23,7** @ 0,48 |
| 15 | 26 620 | 5,09 | 8,53 @ **0,21** | 9,91 | 2,63 | −16,7 @ 0,34 |
| 40 | 93 075 | 11,0 | 10,59 @ 0,07 | 8,31 | 2,11 | **−41,6** @ 6,39 |
| 99 | 527 412 | 19,80 | 16,23 @ 12,58 | 4,84 | 1,19 | **−27,6** @ **15,06** |

## ✔ Alan kaybı düzeltmesi tuttu

f99'da meanT 1,19 / peakT 4,84. Önceki koşuda aynı zamanda 0,31 / 1,5'ti ve
bulut ölüyordu. Artık ölmüyor.

## ✔ Katmanlaşma tuttu

`touching_ceiling` yine hiç true olmadı; top 19,8 / tavan 34. Ve tahmin
tutuyor: peakT 4,84 ÷ stratification 0,25 = **19,4 m**, ölçülen top 19,8 m.
Yani kapak yüksekliği artık **önceden hesaplanabilir bir sayı.**

## ✔✔ Faz B'nin önündeki belirsizlik KALKTI

`pressure_min` her karede **kuvvetle negatif** ve düşük basınç bölgesi
**yükselen kolonla birlikte tırmanıyor**: 0,48 m → 6,39 m → 15,06 m.

★ Bu tam olarak kullanıcının tarif ettiği "merkez hattında düşük basınç"tır,
ölçülmüş hali. Projeksiyon çarpanı pascal değil, ama **şekli** taşıyor — yani
yoğuşma modeli ona bağlanabilir. Faz B'ye girilebilir.

## ✘ Yüzey tozu görünümü BOZDU — ve sebebi "bilinen basitleştirme"ydi

Kullanıcı: *"bir önceki yapı bundan daha iyi idi."* Haklı. Görüntüde alt yarı
şişmiş, zeminde geniş koyu bir toz **halısı** var ve sap şapka kadar kalın.

Ölçüm kanıtı: f15–f40 arası en geniş dilim **0,07–0,21 m**'de sabit ve
genişliği 8,5 → 10,6 m. Bu ilerleyen bir cephe değil, **yayılan bir tabaka**.

Kök: 3. partide "★★ KNOWN SIMPLIFICATION" diye yazdığım şey — **yüzeyin kaynağı
sonsuzdu**. Gerçek bir şok, geçtiği yeri **temizleyip gider**; arkasında toz
kalmadığı için halka olur. Sonsuz kaynakta eşiği bir kez aşan her sütun, rüzgâr
estiği sürece üretmeye devam ediyor ve halka **kendi içini dolduruyor.**

★★★ Ders: *bilinen bir basitleştirmeyi yazmak, onu zararsız yapmaz.* Bu
basitleştirme özelliğin tek görsel iddiasını — "halka cepheyle büyür" — sessizce
tersine çevirdi, ve testteki `skirt EXPANDED` maddesi bunu **geçer**, çünkü
yayılan bir tabaka da genişler.

### Düzeltme: sonlu yer rezervi

`FluidGrid::surface_dust_supply` — sütun başına kalan oran, yenilenme yok.
`gas_surface_dust_supply` = bir sütunun toplam verebileceği yoğunluk, 0 =
sınırsız (eski davranış).

★ Rüzgârın istediği kadar değil, **kalan kadar** verilir. Yeterince güçlü bir
hamle sütunu tek adımda sıyırır ve o sütun biter; bunun yerine *oranı*
kısmak, bütün sütunları sonsuza kadar daha düşük seviyede üretir durumda
bırakırdı — yani halı, ağır çekimde.

★ Reset rezervi **DOLDURUR**, sıfırlamaz. Sıfırlasaydı çekim zaten sıyrılmış
zeminde başlar ve etek hiç çıkmazdı — sessiz, makul görünen bir yokluk.

## Bellek: kare cache'i 218 MiB/kare tutuyordu

Kullanıcı 100 kareden fazlasını bake edemedi. Ölçüm:

`sim_frame_cache_` her kare için **FluidGrid'in tam kopyasını** tutuyordu —
9 float dizisi: density, temperature, fuel, interaction, **pressure,
divergence, vel_x, vel_y, vel_z**. 160×248×160'ta 57,2M float = **218 MiB/kare**,
100 kare = **21,3 GiB**.

Üç ayrı arıza:

★ **Uygulama zaten daha iyisini biliyordu.** `SimCache.cpp`'deki DİSK formatı
tam olarak dört dizi yazıyor ve başlığında "velocity & affine are left
zero-sized-safe" diyor. Yani aynı program, dokuzdan beşinin oynatım için
gereksiz olduğunu çoktan kabul etmiş; RAM yolu o dersi devralmamış.

★ **Alanlar neredeyse boş.** Ölçülen bulut en büyük halinde domain'in
**%8,3**'ünü kaplıyor. Kalan %91,7'yi yoğun saklamak maliyetin çoğu, ve alan
*listesini* kısaltmak bunu çözmez.

★★★ **Ve bütçe KARE sayısıyla tutuluyordu** (`kMaxCachedSimFrames = 600`). Kare
sabit boyutlu olsaydı bu bir sınır olurdu: 64³'te kare ~4 MiB, 600 kare 2,4 GiB.
160×248×160'ta aynı 600 kare **137 GiB**. Sınırladığı şeyle birlikte büyüyen bir
sınır, sınır değildir.

★★ Dahası `estimateSimCacheBytes()` — "cache büyüyor, diske bak" uyarısını
besleyen tek sayı — grid durumlarını **kasten hariç tutuyordu** ("per-cell size
isn't cheaply known"). Yani uyarı, tam olarak şişen duruma **yapısal olarak
kördü.**

### Düzeltme: `SimFrameCompress.h`

8³ tile-seyrek + IEEE half. Boş tile hiç veri tutmaz. Ölçülen kare
**218 MiB → ~5 MiB**.

- **pressure / divergence hiç saklanmıyor.** Basınç her adımda sıfırdan
  çözülüyor (yalnızca SOR için başlangıç tahmini), divergence scratch. Saklamak
  hiçbir şey korumuyordu, karenin %22'sine mal oluyordu.
- **Hız yalnızca her 25. karede.** Sarma ve render hızı hiç okumaz; ama
  **sarılan kareden devam etmek** okur.

★★★ Buradaki ayrım, keyframe'lerin yarattığı asıl mesele: *bir kareyi geri
yüklemek* ile *ondan ileri gitmek* iki ayrı taleptir. Render-only bir kareye
sarıp oynatmak gazı **durgundan** başlatır — sütun birkaç kare duraklar sonra
yeniden hızlanır, ki bu bir çözücü tökezlemesi gibi görünür ve asla cache hatası
diye raporlanmaz. `nearestVelocityKeyframeAtOrBelow` + `sim_live_velocity_valid_`
bu deliği kapatıyor; ★ diskten yükleme **zaten hep hızsızdı** (SimCache.h kendi
başlığında söylüyor), o da artık kayıtlı.

### Ölçülebilirlik

`sim_cache.status` artık `ram_bytes`, `budget_bytes`, `budget_reached`
döndürüyor. ★ Bütçe dolunca **tahliye değil, RED** ediliyor: kullanıcının bake
edilmiş sandığı kareleri sessizce düşüren bir cache, "geri sardığımda sim
değişti" diye okunur ve çözücü hatasından ayırt edilemez.

## Disk cache hakkında

`force_disk_cache` — Cinema profilinin set ettiği bayrak — **hiçbir yerde
okunmuyor**. `ProjectManager`'da serileştiriliyor,
`scene_ui_simulation_domains.cpp`'de dört yerde yazılıyor, okuyan sıfır. Yani
"Cinema profiline geç, diske yazsın" **ölü konfig**. Bu partide dokunulmadı;
sıkıştırmadan sonra 100 kare ~0,5 GiB olacağı için önceliği düştü, ama bayrağın
yalan söylemesi ayrı bir iş olarak duruyor.


---

# 2026-09-20 (5. parti) — sıkıştırma DOĞRULANDI, iki yeni kök bulundu

## ✔ Kare cache'i: 218 MiB → **3,22 MiB** / kare

Canlı ölçüm (`sim_cache.status`): 104 kare, `ram_bytes` = 350 876 160 B = **334,6 MiB**.
Kare başına **3,22 MiB**; tahminim 5 MiB'tı, tile atlama ondan da iyi çalıştı.
Eskisi aynı sahnede 218 MiB/kare, yani 104 kare = **22,7 GiB** isterdi.

★ Kullanıcının gözlemi — *"görev yöneticisi toplam 16 GB diyordu ama uygulama
2,8 GB gösteriyordu"* — bunun doğrudan sonucu: 22,7 GiB'lık bir tahsis 16 GB'lık
makinede **çoğunlukla takasa düşer**. Görev yöneticisinin uygulama sütunu
*working set* (yerleşik sayfalar) gösterir, commit'i değil; sayfa dışına atılan
kısım o sütunda görünmez. Yani cache uygulamanın belleğiydi, sadece **RAM'de
değildi.** Toplamın tavan yapıp uygulamanın mütevazı görünmesi tam olarak bu
tablonun imzasıdır.

## ✘✘✘ Çözücü parametreleri bake'i GEÇERSİZLEŞTİRMİYOR

Canlı kanıt: `gas.set_settings` ile `turbulence_persistence` değiştirildi,
timeline sarıldı, dönen kare **bit bit aynı** (top 6,33 · width 9,64 @ 0,07 ·
36 805 aktif hücre) ve `config_signature` hiç değişmedi.

`computeSimConfigSignature()` grid domain'lerden **yalnızca ayrıklaştırmayı**
hash'liyordu: ad, tip, backend, sınır, bounds, çözünürlük, voxel. Yanındaki
2026-08-17 yorumu bunu anlatıyor — o düzeltme grid geometrisini kapattı ve
**orada durdu.** Kaldırma, vortisite, türbülans, yanma, dissipation,
stratification, yüzey tozu: hiçbiri hash'te yok.

★★★ Bu, bayat bir cache'in alabileceği **en kötü şekil**. Hata vermez, uyarmaz,
ve yanlış bile görünmez: **parametrenin işe yaramadığı gibi görünür.** Baked bir
timeline üzerinde kalibrasyon yapan herkes, hiçbir şeye bağlı olmayan bir kadranı
çeviriyordu. Bu oturumda eklediğim stratification / dissipation / yüzey tozu
kadranlarının hepsi bu durumdaydı.

Düzeltme: 28 yazarlı çözücü alanı imzaya eklendi.

★ Ve **bit-tam** hash'le (`bf`), `qf` ile değil: `qf` 1/1000'e kuantalıyor, ki
canlı poz jitter'ı için doğru ama küçük aralıklı bir kadran için yanlış —
fiziksel preset'in stratification'ı 0,003125, `qf` onu ve bütün komşularını aynı
3'e eşler ve bake, kapağı metrelerce oynatan bir değişikliği atlatırdı.

## ✘✘ Türbülans spektrumu TERSİNE: en küçük oktavlar en güçlüsü

`analyticCurlFbm`'de curl, frekansla çarpan bir terim getiriyor:

```glsl
vec3 c = frequency * vec3(cos(...) - cos(...));
result += c * amplitude;     // amplitude *= persistence, frequency *= lacunarity
```

Yani oktav *o*'nun hız genliği `scale × (lacunarity × persistence)^o`.

| oktav | frekans | dalga boyu | voxel/dalga | göreli genlik |
|---|---|---|---|---|
| 0 | 1,60 | 3,927 m | 28,6 | 1,00 |
| 2 | 6,40 | 0,982 m | 7,1 | 1,25 |
| 4 | 25,60 | **0,245 m** | **1,78** | **1,57** |

`lacunarity × persistence = 2 × 0,56 = 1,12 > 1`, yani genlik **oktavla
büyüyor**, ve en güçlü bileşen **voxel'in altında** (1,78 hücre/dalga).

★★★ Ve alanın yorumu `// amplitude decay per octave` diyordu — **yanlış**.
Varsayılan 0,5'te `L×P = 1,0`, yani "sönüm" etiketli kadran **hiçbir şeyi
söndürmüyor**; 0,5'in üstünde *yükseltiyor*. Kolmogorov benzeri bir kaskad
(v ~ k^−1/3) için doğru değer `lacunarity^(−4/3) = 0,397`.

Yapılanlar:
- Yorum düzeltildi, varsayılan 0,5 → **0,40**. ★ Yüklenen projeler etkilenmez:
  serileştirici alanı her zaman açıkça yazıyor, varsayılanı yalnızca **yeni**
  domain okur (bkz. `Volume` varsayılanı dersi).
- Nükleer preset 0,56 → 0,40.
- `effectiveTurbulenceOctaves()`: dalga boyu **4 hücrenin** altına düşen oktavlar
  kırpılır. ★ Nyquist değil 4 hücre, çünkü mesele dalganın var olup olmadığı
  değil, semi-Lagrangian advection'ın onu **taşıyıp taşıyamadığı**.
- ★★ Kırpma **sessiz kalmıyor**: `gas.get_settings` artık
  `turbulence_octaves_effective` döndürüyor. 5 yazıp 3 koşturan bir kadran,
  çekirdeğe ulaşmayan bir değeri raporlayan panelle aynı sınıftır.

Ölçülen domain'de (voxel 0,1375, scale 1,6): **istenen 5, taşınabilir 3.**

## Diğer presetler

Hepsi 0,50–0,56 aralığında, yani hiçbirinin spektrumu sönümlenmiyor. Bu partide
**dokunulmadı** — nükleer preset üzerinde çalışılıyordu ve diğerlerinin
görünümünü istenmeden değiştirmek ayrı bir karar. Ama hepsi aynı durumda.

## Ölçülen ama düzeltilmeyen: mantar KISA ve GENİŞ

f100'de top 20,37 m, genişlik **16,79 m**. Yani yükseklik/genişlik ≈ 1,2.

Kapak yeri doğru: peakT 4,79 ÷ stratification 0,25 = 19,2 m, ölçülen 20,37 m —
tahmin tutuyor. Sorun kapağın yeri değil, **bulutun yana yükselmekten daha hızlı
yayılması**: f20'de genişlik 9,64 m, yükseklik 6,33 m — daha 20. karede genişlik
yüksekliği geçmiş.

Sürükleniyor değil: kütle merkezi kayması 0,46 m (bulutun %2'si), asimetri
%8–10. Kullanıcının gördüğü "bir tarafa çöküyor", %10 asimetrinin pütürlü bir
kapakta okunuşu.

Silüeti düzeltmek için üç kol, hiçbiri denenmedi:
- `ambient_stratification` ↓ (0,25 → ~0,16 kapağı ~30 m'ye taşır)
- `fire_expansion` ↓ (1,15 çok yüksek; yanal yayılımı o sürüyor)
- `gas_buoyancy_heat` ↑

---

## 2026-09-20 — Tepe bulutun çökmesi: ÖLÇÜLDÜ, kök neden kesin

**Belirti (kullanıcı):** "belli bir kare sonra tepe bulut yere doğru çökmeye başlıyor,
135. kareden sonra hızla."

### Ölçüm

Stratification 0.25 K/m. Denge yüksekliği `h* = anomali / stratification`.

| kare | peakT | h* = T/0,25 | centroid Y | top | max genişlik | bounds_max.x |
|---|---|---|---|---|---|---|
| 80 | 5,76 | 23,03 | 13,18 | 18,17 | 12,94 | 5,52 |
| 110 | 4,38 | 17,50 | **14,85** | 21,61 | 17,34 | 8,55 |
| 120 | 3,99 | 15,96 | 14,75 | **22,16** | 19,13 | 9,65 |
| 135 | 3,48 | 13,91 | 14,08 | 21,89 | 20,51 | **11,02** |
| 150 | 3,03 | 12,13 | 13,02 | 21,75 | **22,02** | 11,02 |
| 175 | 2,41 | 9,64 | 10,77 | 20,65 | 22,02 | 11,02 |
| 200 | 1,92 | 7,67 | 8,47 | 20,37 | 22,02 | 11,02 |

★★★ **Centroid, h*'ı bire bir takip ediyor** (yaklaşık 1 m gecikmeyle, ataletten).
Bulut çökmüyor — **denge yüksekliği çöküyor, bulut onu izliyor.**

### Kök neden

`temperature_dissipation = 0,22 /s`, ve bu alan aynı zamanda **kaldırma kuvvetini taşıyan
alan**. Potansiyel sıcaklık alanının tanımı gereği **korunması** gerekir: bir parsel denge
yüksekliğine çıkınca orada kalır, çünkü θ'sı sönmez. Burada θ sönüyor, dolayısıyla
stratification terimi parseli yavaşça yere geri çekiyor.

★ Yani bu, stratification teriminin hatası değil — tam olarak yazıldığı gibi çalışıyor.
Hata, **sönümlü bir alana korunumlu bir alanın işini vermek.**

### 135. karenin ayrı bir anlamı var

Aynı karede `bounds_max.x` 11,02'ye **sabitleniyor** — domain duvarı. f150'den sonra
`max_width` 22,02, yani domain'in tam genişliği. Yani tepe o karede **duvara dayanıyor**
ve yanal yayılma durunca tek gidecek yön aşağısı kalıyor. İniş f110'da başlıyor ama
**görünür hale f135'te geliyor** — kullanıcının işaret ettiği kare tam olarak bu.

★ Bu, daha önce not edilen "sap kısa, tepe geniş" belirtisini de açıklıyor: tepe kısmen
geniş çünkü duvara sıkışmış.

### Seçenekler (yazarlık kararı, bu partide UYGULANMADI)

1. **Doğru olan:** kaldırma skalerini emisyon sıcaklığından ayır — biri korunumlu, diğeri
   sönümlü. Yeni bir alan ve derleme gerektirir.
2. **Tek kadran, hemen denenebilir:** `temperature_dissipation`'ı 0,22'den ~0,02'ye indir,
   sonra yeni peakT platosunu ölç ve `ambient_stratification = platoT / istenen_yükseklik`
   olarak kalibre et. İki geçişli; `Test-NuclearPreset.ps1` kalibrasyon satırını zaten basıyor.
   **Bake'i geçersiz kılar.**
3. **Bağımsız ve her hâlükârda gerekli:** domain'i genişlet. Tepe 22 m'lik domain'de
   22,02 m'ye ulaşıyor, yani en az iki katı genişlik istiyor.

⚠ 2 ve 3 önbelleği siler (202 kare).

---

## 2026-09-21 — Tepe bulutunun çökmesi: h* = peakT / stratification

Belirti: bulut yükseliyor, f120'den sonra hızla zemine iniyor, "hâlâ çok sıcak
olmasına rağmen".

### Ölçüm (strat = 0.25, temperature_dissipation = 0.22)

| kare | tepe | merkez | peakT | h\* = T/0.25 | merkez − h\* |
|---|---|---|---|---|---|
| 60 | 20.40 | 14.65 | 6.65 | 26.6 | −11.96 |
| 90 | 25.67 | 18.47 | 5.05 | 20.2 | −1.73 |
| 110 | 27.20 | 18.52 | 4.21 | 16.8 | +1.69 |
| 130 | 27.20 | 17.09 | 3.50 | 14.0 | +3.09 |
| 160 | 26.86 | 13.58 | 2.66 | 10.6 | +2.94 |
| 200 | 26.86 | 8.52 | 1.84 | 7.4 | +1.15 |

Son sütun her karede ±3 m. **Bulut denge yüksekliğinde duruyor; inen şey
dengenin kendisi.** Ölçülen sönüm oranı ln(6.65/1.84)/5.83 s = 0.220/s, yani
`temperature_dissipation` ile birebir — 24 fps varsayımıyla da tutarlı.

★ Bu bir regresyon değil: `ambient_stratification` solver'a ilk kez bu
derlemede kopyalanıyor (bkz. GAZ_ADIMI_TASIMA_MALIYETI.md, geri yüklenen
eşlemeler). Önceki derlemelerde tabakalaşma hiç uygulanmıyordu.

### ★★ Yanlış çıkan ilk düzeltme: dissipation 0.22 → 0.03

Gerekçem şuydu: emission yolunda gaz domain'i için `temp *= 3000` var ve
`temperature_max = 5000`, dolayısıyla T > 1.67 olan her hücre azami parlaklığa
kıstırılıyor; peakT o pencerede 6.65 → 1.84, ikisi de clamp'in üstünde; demek
ki sönüm görsel olarak bedava.

**Yanlıştı.** peakT'ye bakıp bütün alan hakkında hüküm verdim. Gözün gördüğü
sönme `mean_temperature`'dan geliyor: 0.22'de meanT 2.73 → 0.54 (kelvin 1620,
u ≈ 0.08, yani karanlık), 0.03'te meanT ~6'da kalıyor ve **bütün hücreler
clamp'in üstünde** — bulut domain'i doldurup masif beyaz bir bloğa dönüştü
(f200 fill_fraction 0.68). Ders: *tepe değeri bir alanın tamamı hakkında
konuşmaz.*

### Ölçülen en iyi hal: dissipation 0.22, stratification 0.25 → 0.09

| kare | tepe | peakT | meanT | h\* = T/0.09 | fill | lid |
|---|---|---|---|---|---|---|
| 60 | 22.27 | 6.62 | 2.73 | 73.6 | 0.140 | hayır |
| 90 | 31.11 | 5.03 | 2.06 | 55.9 | 0.300 | hayır |
| 110 | 34.00 | 4.19 | 1.62 | 46.5 | 0.414 | **EVET** |
| 200 | 34.00 | 1.83 | 0.54 | 20.4 | 0.678 | **EVET** |

h\* bütün çekim boyunca bulutun üstünde kalıyor, yani geri çağırıcı kuvvet hiç
devreye girmiyor: bulut **itildiği için değil, soğuduğu için** duruyor. Çöküş
tamamen kayboldu (f200'de tepe 8.52 yerine 34), sütun koyu, zemin eteği koyu
ve kabarıyor (surface dust ilk kez çalışıyor).

### ⚠ Kalan kısıt: domain tavanı

f110'dan itibaren `touching_ceiling = true`. Domain 22.1 × **34** × 22.1
(130×200×130, voxel 0.17). h\* f110'da 46.5 — tavanın üstünde.

★★★ Ve bu bir parametre sorunu değil, **yapısal bir kısıt**: sıcaklık çekim
boyunca 3.6 kat düşüyor, dolayısıyla h\* de 3.6 kat aralıkta geziyor. Bu
aralığın üstü tavana çarpıyorsa altı zemine çöküyor. Tek bir stratification
değeri ikisini birden kurtaramaz.

Gerçek çözüm ikisinden biri:

1. **Domain'i yükselt** (~55 m → y'de 324 hücre, ~5.5M hücre). Ucuz değil ama
   tek satır.
2. ★★★ **Kaldırma kuvvetini, görünüm için sönümlenen alandan ayır.** Emission
   ayrı bir fade kanalından (flame/fuel) okunmalı ki ısı görüntüyü yakmadan
   kalabilsin. Bu partinin dışında; asıl doğru olan bu.

---

## ★★★★★ 2026-09-21 — Soğuyan bulut neden hâlâ beyaz: emisyon sıcaklığa BAĞLI DEĞİL

Soru: "blackbody emisyon rengini temperature'den almıyor mu ki soğuyan yapı
hâlâ beyaz oluyor?"

**Renk alıyor. Parlaklık almıyor.**

### Ölçüm

f200'de `blackbody_intensity` 16.667 → 0 yapıldı: bulut **tamamen siyah**
bir siluete döndü. Yani gördüğümüz beyaz, güneş alan duman değil, emisyon.
(Yan bulgu: hacim neredeyse hiç saçılan ışık almıyor — ayrı bir konu.)

16.667 → 2.0 (8 kat düşük): kapak **yine beyaz**, yalnızca kenarlarda turuncu
bir halka belirdi. Yoğunluk sekizde birken bile clip ediyor.

### Kök neden

```glsl
vec3 blackbodyToRGB(float kelvin){
    if (temp <= 66.0) { r = 1.0; }        // <6600 K: KIRMIZI HEP 1.0
    ...
    if (temp >= 66.0) { b = 1.0; }        // >6600 K: MAVİ HEP 1.0
}
...
vec3 eEmis = eColor * ed * vol.blackbody_intensity;   // ← magnitude YOK
```

Bu Tanner Helland yaklaşımı bir **kromatiklik** döndürür: her sıcaklıkta en az
bir kanal 1.0'a sabitli. Yalnızca yoğunlukla çarpılınca:

| | 1620 K | 5000 K |
|---|---|---|
| kırmızı kanal | **1.00** | **1.00** |
| yeşil | 0.45 | 0.83 |
| mavi | 0.00 | 0.69 |

**Aynı parlaklık.** Soğumak tonu değiştirir, enerjiyi değiştirmez. Isısının
%90'ını kaybetmiş bir bulut, ışın boyunca yeterince kalınsa kırmızı kanalı
zaten clip etmiştir; her ek adım yeşili ve maviyi de clip'e sürükler → beyaz.
Ton yalnızca sütunun ince olduğu yerde, yani siluet kenarında hayatta kalır —
iki ölçüm görüntüsünün ikisinde de tam olarak bu görülüyor.

★ `volume_closesthit.rchit` içindeki mevcut yorum bu arızayı zaten tarif
ediyordu, sebebini bilmeden: *"Temperature alone cannot tell a flame from the
hot smoke sitting above it: both are hot, so the whole plume glows."* İkisi de
parlak, çünkü **parlaklık sıcaklığın fonksiyonu değildi.**

### Düzeltme

Planck-integre radyans T⁴ ile gider (Stefan-Boltzmann). Yazarın belirlediği
`temperature_max`'e normalize edildi, böylece **en sıcak gaz bugünkü
parlaklığını aynen korur**, yalnızca soğuk gaz kararır:

```glsl
e_radiance = pow(clamp(authoredKelvin / rangeMax, 0.0, 1.0), 4.0);
```

1340/5000 aralığında en soğuk görünür hücre çekirdekten ~190 kat sönük.

Dört çağrı yerine uygulandı: raster march, Vulkan RT ana march, RT emitter
bloğu (ışık yayan hacmin komşu yüzeyleri aydınlatması) ve CPU/OptiX yolu.
**Yalnızca blackbody koluna** — yazılmış bir renk rampası nihai renk hakkında
bir beyandır, onu T⁴ ile ölçeklemek yazarla kavga etmek olurdu.

### ★★ Bu, yükseklik/parlaklık gerilimini de çözüyor

Önceki bölümde "ısıyı tutarsan bulut beyaza patlıyor, sönümlersen bulut
çöküyor, ikisi aynı alan" diye bir çıkmaz tarif edilmişti. O çıkmaz bu
arızanın sonucuydu: sönüm, parlaklığı düşürmenin **tek** yoluydu. T⁴ ile
parlaklık sıcaklığa dik biçimde bağlı olduğundan, artık ısı tutulup bulut
havada kalırken parlaklık kendi başına sönebilir.

⚠ Dolayısıyla `temperature_dissipation` ve `ambient_stratification` T⁴
derlendikten SONRA yeniden ayarlanmalı. Bu bölümden önceki bütün kalibrasyon
sayıları ölü emisyon modeline göre seçildi.

---

## ★★★★★ Kullanıcının ayrımı teşhisi kapattı: SADECE nükleer preset çöküyor

Kullanıcı 2026-09-21'de şunu bildirdi: *"diğer mushroom preset tepe bulut
çökmüyor sadece nükleer presette oluyor."*

Tek satırda doğrulandı:

```
ParticleSimulation.h:420   float gas_ambient_stratification = 0.0f;   // VARSAYILAN
SceneDataParticlePresets.cpp:933   dom.gas_ambient_stratification = 6.5f / (26.0f * S);
```

**Bütün depoda bu alanı yazan TEK yer nükleer preset.** Ve `buoyantAnomaly`
ilk satırında erken dönüyor:

```cpp
if (params.ambient_stratification == 0.0f) return excess;
```

Yani diğer bütün preset'lerde env terimi, clamp, geri çağırıcı kuvvet —
hiçbiri **hiç çalışmıyor.** Bulutları çökmüyor çünkü onları aşağı itecek bir
mekanizma yok; ısıları bitene kadar yükseliyorlar.

### Bu iki şeyi birden kanıtlıyor

1. **Çöküşün sebebi tabakalaşma terimi**, başka hiçbir şey değil. Diferansiyel
   deney kendiliğinden yapılmış oldu: aynı çözücü, aynı render, tek fark bu
   alan.
2. **2026-09-21 kaldırma kuvveti düzeltmesinin etki alanı bir preset.**
   `gas_ambient_stratification` varsayılanı 0 olduğu için presence kapısı da
   clamp'in kaldırılması da yalnızca nükleer sahneyi (ve elle bu dial'e
   dokunulmuş sahneleri) etkiler. Diğer preset'lerde tek bir kod yolu bile
   değişmiyor.

### ★ Ve satırın kendisi hatayı taşıyor

```cpp
dom.gas_ambient_stratification = 6.5f / (26.0f * S);
```

Niyet açık: *"6.5'lik bir anomali 26·S metrede dengelensin."* Ama anomali
**6.5'te kalmıyor** — 0.22/s ile üstel sönüyor, ve h\* = anomali/strat olduğu
için tavan da onunla birlikte iniyor. Formül, sönümün anında geçersiz kıldığı
bir zirve-sıcaklık varsayımını kodluyor.

Zaten satırın üstündeki yorum bunu yarı yarıya itiraf ediyormuş:

> ★ The 6.5 is EXTRAPOLATED from the 2026-09-20 run, NOT measured under the
> new rate.

**Ekstrapolasyon bir ölçüm değildir** — ve burada ekstrapole edilen şey,
zamanla sabit olmayan bir büyüklüğün tek bir anıydı.

---

## ★★★★★ Denge bir HÂL değil, bir ANDI — sönüm hedefi yanlış yerdeydi

Kullanıcı 2026-09-21'de bildirdi: *"ambient_stratification 0.01'in üstüne
çıkınca geç de olsa çöküyor, 0.03'te 170'lerde yavaş da olsa çöküyor."*

Yani presence kapısı ve clamp'in kaldırılması çöküşü **yavaşlattı ama
kaldırmadı.** Canlı sahnede strat = 0.03 ile ölçüm (meanT üzerinden,
`buoyancy_heat` 2.6):

| kare | merkez | meanT | env | anomali | kuvvet |
|---|---|---|---|---|---|
| 90 | 25.6 | 2.04 | 0.77 | +1.27 | +3.30 |
| **130** | **30.0** | 0.88 | 0.90 | **−0.02** | **−0.06** |
| 160 | 30.4 | 0.41 | 0.91 | −0.50 | −1.31 |
| 200 | 29.7 | 0.25 | 0.89 | −0.64 | −1.66 |

★ **f130'da bulut gerçekten nötr kaldırmaya ulaşıp 30 m'de park etti** — doğru
davranış, bir kapağın yapması gereken tam olarak bu. Sonra sıcaklığı dengenin
**içinden geçip** düşmeye devam etti, anomali negatife gömüldü ve orada kaldı:

```
anomaly = excess − strat·h     ve     excess → 0
⇒ anomaly → −strat·h  = SABİT, hiç bitmeyen aşağı kuvvet
```

Yakınsadığı değer −strat·h·buoyancy_heat = −2.34 m/s². Denge bir **hâl**
değil, bulutun içinden geçtiği bir **an**dı. Yazarın denediği hiçbir
stratification değerinin çöküşü kaldıramamasının sebebi bu: 0.01 ve 0.03
terimi kaldırmıyor, sadece yavaşlatıyor.

### Kök: sönüm GLOBAL ambient'a çekiyor, kaldırma YEREL çevreye bakıyor

`dissipate()` sıcaklığı tek bir global değere çekiyordu. `buoyantAnomaly` ise
parseli yükseklikle **artan** bir çevreye karşı ölçüyor. İkisi baştan beri
farklı bir "sıfır" tanımlıyordu ve bu uyuşmazlık kalıcıydı.

`dissipateTemperatureStratified()` hedefi yerel çevre yaptı:

```
excess → strat·h·presence     ⇒     anomaly → 0
```

Artık kaldırma teriminin aradığı sabit nokta var: bulut durduğu yerde kalır.

★ Hedef de **presence ile kapılı**, ve bunun sebebi fizik değil: aynı alan
**blackbody emisyonunu sürüyor.** Çevre profilini boş hücrelere yazmak açık
havayı parlatırdı. *Üreticinin tek tüketicisi kendisi değil.*

⚠ Büyük stratification'da kalan sıcaklık ihmal edilebilir değil: 0.25'te 26
m'de park eden bir hücre excess 6.5 taşır ve bu parlaktır. Kapak yüksekliğini
**büyük stratification ile değil, küçük stratification + yüksek domain** ile
kurmanın bir sebebi daha.

★ Ayrıca iki yol baştan beri farklı sönüyormuş: yoğun yol 0'a, seyrek VDB yolu
`ambient_temperature`'a. İkisi de yeni hedefe çekildi.

---

## ★★★★★ Nihai biçim: çevre kaldırmayı İPTAL EDER, TERSİNE ÇEVİREMEZ

Gevşeme-hedefi denemesi de çöküşü kaldırmadı. Ölçüm (strat = 0.05) sebebini
verdi:

| kare | hücre | meanT | hedef | toplam ısı |
|---|---|---|---|---|
| 130 | 1391748 | 1.319 | 1.485 | 1836235 |
| 160 | 1538403 | 0.844 | 1.499 | 1298773 |
| 200 | 1959861 | 0.770 | 1.476 | 1508226 |

Toplam ısı **sönmüyor** — gevşeme hedefi çalışıyor. Ortalamanın düşmesi
soğuma değil **seyrelme**: hücre sayısı 1.39M → 1.96M.

★ Ve arıza tam buradaydı: genişleyen bulut sürekli **T = 0 ile yeni hücre
dolduruyor**, o hücrelerin hedefi ise `strat·h`. Yani büyüyen kenar kendi
aşağı kuvvetini imal ediyor ve 0.22/s'lik gevşeme buna yetişemiyor. Alanı
çevre profiline eşitlemeye çalışmak, genişleyen bir bulutla kalıcı olarak
kavga eder. **Yanlış kaldıraçtı; geri alındı** (kural 5 — tek iş için iki
mekanizma yaşatma).

### Doğru biçim

```cpp
return excess - std::min(env, std::max(excess, 0.0f));
```

Denenip ölçülen üç biçim:

| biçim | sonuç |
|---|---|
| `max(excess-env, -abs(excess))` | sıralama ters: ılık batar, soğuk asılı kalır |
| `excess - env` | denge bir AN; anomali −strat·h'ye oturur, kalıcı iniş |
| **`excess - min(env, max(excess,0))`** | **h\* üstünde kuvvet YOK; bulut durduğu yerde kalır** |

★★ Gerekçe dial'in ne için var olduğunda: preset notu açıkça yazıyor —
*"gives the plume a ceiling of its OWN"*. **Tavan bir yükselişi durdurur,
aşağı itmez.**

★ Ve hiçbir şey kaybedilmiyor: soğuk ağır dumanın aşağı kuvveti **zaten ayrı
ve doğru işaretli bir terim olarak var** — `buoyancy_density * d`.
Tabakalaşmanın da aşağı itmesi, mevcut bir kuvveti yanlış işaret
konvansiyonuyla ikinci kez uygulamaktı.

★ Sınır sonuca değil `env`'e konuldu ki tabakalaşmasız hâl ile **sürekli**
kalsın: `excess <= 0` için çevre tam olarak sıfır katkı verir ve anomali
`excess`'tir — `ambient_stratification == 0` erken çıkışıyla bit-aynı. Diğer
bütün preset'ler bu alanı 0'da bıraktığı için burada bir basamak görmemeleri
şart.

---

## ★★★★★ Son eksik parça: duman hiç TEMİZLENMİYORDU

Kullanıcı: *"180 frame gibi dumana dönüşen sütun çökmese de yükselip
zayıflamıyor... ısı kaybedip zayıflama veya yükselip zayıflaması yok."*

Kaldırma kuvveti artık doğruydu ama bulut **duruyor ve yalnızca yayılıyordu**.

### Kaynaklar zaten kapalıydı

`flow_source.list` okundu:

| kaynak | start → end (s) | kare |
|---|---|---|
| Detonation Core | 0.00 → 0.05 | 0 → 1 |
| Fireball Rise | 0.05 → 0.55 | 1 → 13 |
| Stem Afterwind | 0.45 → 4.50 | 11 → **108** |

f108'den sonra **domain'e tek gram yeni kütle girmiyor.** Yani aktif hücrenin
1.61M → 2.60M büyümesi üretim değil, saf **yayılma**.

### Kök: `density_dissipation = 0.012/s`

8.3 saniyede dumanın yalnızca **%9'u** temizleniyor. Kaynak kapalı, temizlik
yok: bulut sonsuza kadar orada duruyor ve yayıldıkça domain'i dolduruyor.
Doluluk hiçbir kararlı hâle ulaşmıyordu (0.25 → 0.48 → 0.64 → 0.77).

### 0.012 → 0.18

| kare | hücre | fill | tepe | merkez |
|---|---|---|---|---|
| 90 | 1 042 832 | 0.309 | 32.1 | 25.0 |
| 110 | 1 288 238 | 0.381 | 34.0 | 28.5 |
| 140 | 1 187 734 | 0.351 | 34.0 | 30.2 |
| 170 | 1 164 291 | 0.344 | 34.0 | 30.5 |
| 200 | 1 188 893 | 0.352 | 34.0 | 30.3 |

Hücre sayısı f110'da tepe yapıp **düşüyor ve sabitleniyor**: yayılma ile
temizlik dengeye geliyor. Görsel olarak koyu gri sütun, incelen yarı saydam
duman, közlü kapak — istenen "yükselip zayıflama" bu ayarla geliyor.

★ Yan etki ve hoş olanı: duman inceldikçe `presence` kapısı düşüyor, dolayısıyla
çevre terimi zayıflıyor ve incelen duman kaldırma kuvvetini geri kazanıyor.
İki mekanizma aynı yoğunluk ölçüsünü paylaştığı için bu bedava geliyor.

### Sahnede doğrulanan değerler

```
ambient_stratification = 0.05
density_dissipation    = 0.18      (preset'te 0.012)
temperature_dissipation= 0.22
temperature_max        = 20000     (preset'te 5000)
```

⚠ Hiçbiri `SceneDataParticlePresets.cpp`'ye girmedi.

⚠ Kalan tek kusur: kapak hâlâ f110'dan sonra domain tavanında (34 m).
Parametreyle çözülmez.
