# Node Tabanlı Simülasyon Mimarisi ve Madde/Termodinamik Katmanı — Karar Planı

## Amaç

Mevcut Fluid, Gas, Foam, Vulkan ve shader altyapısını koruyarak iki katman
kazandırmak:

1. **Node kontrol katmanı** — Houdini benzeri, fakat RayTrophi'nin mevcut UI ve
   veri modeline bağlı bir simülasyon orkestrasyonu.
2. **Madde ve termodinamik katmanı** — objelerin kimyasal yapısı olacak (demir,
   selüloz, buz, mum...); ısıya ve ateşe madde özelliklerine göre tepki
   verecekler: demir erir, ahşap kömürleşip küle döner, ıslak yüzey geç tutuşur.

İkinci katman birincinin *içeriği*, birinci katman ikincinin *arayüzü*. Bu
dokümanda önce madde katmanı tanımlanır (çünkü node tipleri ondan türer), sonra
node mimarisi.

## Temel karar önerisi

Node graph solver'ın yerine geçmeyecek; mevcut solver'ları yöneten derlenmiş bir
simulation orchestration katmanı olacak. Böylece mevcut UI, API, serialization,
cache ve GPU kaynak yönetimi korunacak.

```text
Node Graph → Graph Compiler → Dependency Scheduler → Existing Solvers → Vulkan
```

---

# BÖLÜM A — Madde ve Termodinamik Katmanı

## A.1 Mevcut durum: çekirdek zaten var

`shaders/sim_gas_collider_source.comp` içinde çalışan tam bir piroliz modeli var:

- Yüzey sıcaklığı birinci mertebe termal ataletle komşu gaz sıcaklığını takip
  eder (`transfer = 1-exp(-3*dt)`), tek frame'lik ani sıçramayla tutuşmaz.
- `ignition_temperature` eşiği aşılınca `remaining` yanabilir kütle tüketilir.
- Yanan kütle komşu açık hücreye fuel + duman (`*0.30`) + ısı olarak salınır;
  asıl yanmayı domain'in normal combustion geçişi yapar.

Yani **"ağaç yanar" fiziğinin çekirdeği çalışıyor.** Bu katman sıfırdan
yazılmayacak, taşınacak ve genişletilecek.

## A.2 Kök sorun: yanma durumu ızgarada yaşıyor, objede değil

`surface_state[]` gaz domaininin **voxel indeksiyle** adreslenmiş. Üç sonucu var:

1. Obje domainden çıkarsa yanma durumu yok olur.
2. Voxel çözünürlüğünde tutuluyor — render'ı (kararma) veya geometriyi (erime)
   süremez.
3. Temas kesilince kod `surface_state[fuel_state_id] = -1` yazarak **kömürleşmeyi
   kasten siler**. Animasyonlu collider'ın kaynak davranışı için doğru; kalıcı
   malzeme hasarı için kabul edilemez.

Dolayısıyla ilk yapısal iş, durumu ızgaradan objeye taşımaktır.

## A.3 Material State Field (MSF)

Obje başına kalıcı, önbelleklenebilir yüzey durumu katmanı. Kodda iki prototipi
zaten var ve MSF ikisinin genellemesidir:

- `SculptWetClayState` (`scene_ui.h`) — per-vertex skaler + `active_list` +
  kuruyan vertex'i tahliye ederek maliyeti aktif kümeye sınırlama.
- `WetSimulationState` (`Paint/MeshPaintAdapter.h`) — per-texel ıslaklık,
  kalınlık, pigment + dirty-rect + UV seam köprüleri.

Wet clay fırçası zaten *"ıslaklık = geçici mobilite, kuruyunca kilitlenir"*
modelini kuruyor. **Erime bunun ısı sürümüdür**; aynı akış çekirdeği viskozitesi
`melt` fraksiyonundan gelecek şekilde yeniden kullanılır.

### Kanallar

| Kanal | Anlam | Süreceği şey |
|---|---|---|
| `temperature` | Yüzey sıcaklığı (K) | Her şey |
| `moisture` | Nem / ıslaklık 0..1 | Tutuşma gecikmesi, sönme, shading |
| `fuel_remaining` | Kalan yanabilir kütle | Piroliz |
| `char` | Kömürleşme 0..1 | Shading (kararma), yapısal zayıflama |
| `melt` | Sıvı fraksiyonu 0..1 | Geometri akışı, emission |
| `mass_loss` | Kaybedilen kütle | Büzülme, kül, fluid'e devir |

### Depolama kararı

Birincil depolama **per-vertex, flat SoA üzerinde**. Gerekçe: geometriyi
etkileyen kanallar (`melt`, `mass_loss`) zaten vertex uzayında çalışmak zorunda
ve sculpt sistemi flat SoA-native. Tek otorite tutmak iki senkron kaynağından
iyidir.

Bilinen sınır: `char` detay çözünürlüğü tesselasyona bağlı kalır. Kabul edilen
takas — ileride yalnızca *shading* kanalları için opsiyonel per-texel char
maskesi eklenebilir (Paint altyapısı hazır), ama ilk sürümde yok.

### Yaşam döngüsü

MSF **runtime durumudur, mesh verisi değildir.**

- Mesh'e yazılmaz; obje üzerinde override katmanı olarak durur.
- Frame 0'a sıfırlanabilir olmalı.
- `SimCache`'e serialize edilir ve **cache imza hash'ine girer.** `scene_data.h`
  içindeki collider hash'i bugün `gas_ignition_temperature` vb. içeriyor; madde
  profili de girmeli, aksi halde maddeyi değiştirince bayat yanma cache'i oynar.
- Aktif küme mantığı: ortam sıcaklığındaki ve üstüne ısı gelmeyen obje termal
  adımın içine hiç girmez (wet-clay `active_list` tahliyesiyle aynı kalıp).

## A.4 Madde kütüphanesi (SubstanceProfile)

Bugün `ParticleSimulation.h` içindeki collider `gas_ignition_temperature`,
`gas_surface_fuel_capacity`, `gas_surface_burn_rate` elle girilen üç serbest
float. Bunlar bir profil arkasına alınacak — materyalde "Demir" seçince türeyecek:

```text
SubstanceProfile {
    density, specific_heat, conductivity, emissivity,
    ignition_T, melt_T, boiling_T, latent_heat_fusion,
    burn_rate, char_yield, ash_yield,
    char_color, molten_emission_curve, melt_viscosity
}
```

Başlangıç presetleri: Demir, Çelik, Bakır, Selüloz/Ahşap, Kağıt, Buz, Mum,
Plastik, Taş, Kumaş, Et.

Geriye uyum: mevcut üç float, profil seçilmediğinde aynen çalışmaya devam eder
(`Custom` profili).

## A.5 Birim otoritesi — çözülmesi zorunlu çelişki

Kod tabanında bugün üç uyuşmaz sıcaklık konvansiyonu var:

| Yer | Birim | Durum |
|---|---|---|
| `World.h` `AtmosphereParams.temperature` | Celsius (−50..+50) | Yaşıyor, ama **render** parametresi (atmosferik saçılma) |
| `GasSimulator.h` `ambient_temperature=293`, `max_temperature=6000` | Kelvin | Doğru tasarlanmış, ama **ölü yol** |
| `RtApi.h` `fire_max_temperature = 10.0f` | Normalize | **Canlı grid-domain yolu** |

Karar: **yazım (authoring) birimi Kelvin.** Domain seviyesinde tek bir
`temperature_scale` solver'ın normalize alanına eşler. Madde eşikleri her zaman
Kelvin yazılır; normalize değere dönüşüm tek noktada olur.

`AtmosphereParams.temperature` bu iş için **kullanılmayacak** — o bir render
parametresidir; termal otorite yapılırsa gökyüzü görünümü değiştirildiğinde sahne
yanmaya başlar. Ayrı bir termal durum eklenir, UI'da opsiyonel "atmosfer
sıcaklığını takip et" bağı sunulur.

## A.6 Ortam koşullarının yeri: World mü, domain mi?

**World, simülasyon durumunu değil sınır koşullarını taşır.** Katmanlama:

```text
World (varsayılan, her yerde tanımlı)
  → Domain override (kendi sınırları içinde ezer)
    → Thermal Field (hacimsel yerel etki, domain sınırından bağımsız)
      → Obje MSF (yüzey durumu)
```

`WorldThermalState`: `ambient_temperature_K`, `oxygen_availability`,
`convection_coefficient`, global rüzgâr/yerçekimi bağı. Statik veya keyframe'li,
ucuz.

Bunun asıl kazancı: **hiçbir domainin içinde olmayan objenin de tanımlı bir
sıcaklığı olur.** Yanan kütük duman domaininden dışarı taşındığında sönmez;
ortamla ışınım/taşınım dengesine göre soğur. Domain-bağımlı tasarımda bu senaryo
çalışmaz ve sistem "sadece domain içinde fizik var" hissi verir.

Isı alanı için **yeni varlık üretilmeyecek** — mevcut force field sistemine
`Thermal` modu eklenir. Isı alanı uzamsal olarak skaler yüklü bir force field'dır;
ayrı sistem kurmak binding + IPC dispatch + capability olmak üzere üç yerde
ikinci bakım yükü demektir.

## A.7 Kuplajlar

| Yön | Ne yapar | Maliyet |
|---|---|---|
| Gaz → MSF | Yüzey sıcaklığı gazı termal ataletle takip eder | **Var** |
| MSF → Gaz | Piroliz: fuel + duman + ısı salımı | **Var** |
| MSF → Render | `char` → base color kararması + roughness ↑; `melt`/`T` → blackbody emission (kızıl demir) | Düşük |
| Fluid → MSF | Su teması `moisture` yazar; nem tutuşmayı geciktirir, yanmayı söndürür | Düşük, **getirisi yüksek** |
| MSF → Geometri | `melt` → wet-clay akış çekirdeği, viskozite `melt`'ten; `ash_yield` → büzülme | Orta |
| MSF → Fluid | Eriyen kütle eşiği aşınca APIC domain'ine parçacık tohumlar, mesh o kütleyi kaybeder | Yüksek |

### Erime konusunda kapsam sınırı

Katı bir demir çubuğun **topolojik olarak eriyip birikinti yapması vertex
taşıyarak yapılamaz.** İş ikiye bölünür:

- (a) Yüzey sarkma/parlama etkisi — wet-clay tarzı teğetsel taşıma, mesh korunur.
- (b) Gerçek akma — eriyen kütle **fluid solver'a parçacık olarak devredilir**,
  mesh o kadarını kaybeder. APIC solver'ın viskozite ve sıcaklık taşıyabilmesi bu
  yolu ucuzlatıyor.

Katı→sıvı yeniden meshleme **kapsam dışı.** Denenirse orada tıkanılır.

Ahşap→kül daha kolay: `char` gölgelendirme + büzülme, `fuel_remaining == 0`
olunca kül parçacık sistemi doğurulur ve yüzey çökertilir.

---

# BÖLÜM B — Node Mimarisi

## B.1 Kullanıcı modeli

Kullanıcı önce domain ve kaynakları oluşturur, sonra fiziksel akışı node'larla
bağlar. Gelişmiş kullanıcılar mevcut özellik panellerine erişmeye devam eder.
Node'lar aynı ayarları tekrar tanımlamaz; mevcut objelere, field'lara ve API
parametrelerine bağlanır.

## B.2 Node kategorileri

### Simulation

- Fluid Domain / APIC Solver
- Gas Domain / Combustion Solver
- Foam From Fluid
- Collider
- Fluid-to-Gas Coupling
- Gas-to-Fluid Ignition

### Material / Chemistry *(yeni — Bölüm A'nın arayüzü)*

- **Substance** — madde profili seçer/override eder
- **Thermal Transfer** — ortam/gaz/alan ↔ yüzey ısı alışverişi
- **Pyrolysis** — tutuşma, yanma, kömürleşme
- **Phase Change** — erime/donma/buharlaşma
- **Char / Ash Output** — kütle kaybı, kül doğurma, büzülme
- **Moisture** — nem yazımı ve sönme

### Field ve veri

- Density, Temperature, Fuel, Flame
- Velocity, Surface SDF, Foam Density
- Field Combine, Remap, Smooth, Advect
- Particle Set ve Field Cache

### Render

- Liquid Material
- Foam Material
- Smoke Volume
- Fire / Blackbody
- **Char / Molten Surface** *(yeni)*
- Volume Material Output

## B.3 Tip güvenliği

Socket türleri açıkça tanımlanmalı: `FloatField`, `VectorField`, `SDFField`,
`ParticleSet`, `DomainRef`, `Material`, **`SurfaceField`** (MSF referansı),
**`Substance`**. Uyumsuz bağlantılar graph derleme aşamasında reddedilmeli veya
kontrollü conversion node'u önermeli.

## B.4 Yürütme akışı

1. Graph değişikliği dirty işaretler.
2. Compiler bağlantıları doğrular ve execution plan üretir.
3. Scheduler yalnızca değişen node ve downstream bağımlılıklarını çalıştırır.
4. GPU kaynakları domain/field/MSF kimliğiyle yeniden kullanılır.
5. Cache node'u frame ve parametre imzasına göre kayıt/okuma yapar.
6. UI, node durumunu: hazır, dirty, simüle ediliyor, cache'li veya hatalı olarak
   gösterir.

## B.5 Karara bağlanan tartışma maddeleri

| Soru | Karar |
|---|---|
| Ayrı editor mü, panel modu mu? | Mevcut UI içinde `Simulation Graph` **paneli** olarak başla; ayrı pencere sonra |
| Parametreler objeye mi yazılsın? | **Hayır, override katmanı.** MSF runtime durumudur, frame 0'a sıfırlanabilir kalmalı; mesh'e yazılırsa geri alınamaz |
| Cache otomatik mi? | Kullanıcı kontrollü bake + otomatik imza doğrulama |
| Tek graph mı, alt-graph mı? | Tek graph ile başla; alt-graph/preset sonra |
| GPU scheduler ilk sürümde mi? | Hayır — CPU dependency planı ile başla |
| Ortam sıcaklığı nerede? | **World** (sınır koşulu), domain kendi sınırında ezer — bkz. A.6 |
| Sıcaklık birimi? | **Kelvin** yazım birimi, domain seviyesinde normalize eşleme — bkz. A.5 |

Shader graph ve gelişmiş GPU fusion ilk sürüme alınmayacak; mevcut shader
parametreleri önce Material node'larına bağlanacak.

---

# BÖLÜM C — Faz Sırası

Faz 3'e kadar solver'lara neredeyse hiç dokunmadan görünür sonuç alınır.

| Faz | İş | Durum |
|---|---|---|
| **1** | **MSF iskeleti** — obje başına kalıcı yüzey durumu, GPU-sahipli, gather-only. Voxel yolunun yanında bayrakla çalışır | **✅ DOĞRULANDI** |
| **2** | **SubstanceProfile + Kelvin birim katmanı** — 12 maddelik kütüphane, tek dönüşüm noktası | **✅ DOĞRULANDI** |
| **2b** | **MSF tek otorite** — `sim_msf_scatter`/`resolve` gaza salar; voxel `surface_state`, üç serbest float ve `Custom` **söküldü** | **YAZILDI** (SimCache hariç) |
| **3a** | **Per-texel char maskesi** — MSF örnekleme kümesi üçgenden texel'e terfi; alan-bazlı çözünürlük bağımsızlığı | **✅ DOĞRULANDI** |
| **3b** | **Render bağlama** — char maskesi per-instance, blackbody molten emission | **✅ DOĞRULANDI** |
| **3c** | **Hasar kontrolü** — obje başına / sahne geneli "Clear Damage" | **YAZILDI** |
| **3d** | **Kızıl ısı ışıması** — maske G kanalı mutlak Kelvin'e bağlandı, eşik Draper noktası | **YAZILDI** |
| **3e** | **Madde alan başına** — `step()`'ten profil parametresi söküldü; her alan kendi maddesini çözer | **✅ DOĞRULANDI** |
| **4** | **WorldThermalState + Thermal force field modu** — ortam sıcaklığı, domain override | **✅ DOĞRULANDI** |
| **4b** | **MSF frame cache** — RAM timeline + disk bake; hasar artık scrub'da hayatta kalıyor | **✅ DOĞRULANDI** |
| **5** | **Nem / söndürme** — fluid teması MSF'ye `moisture` yazar | **YAZILDI** |
| **6b** | **Erime durumu** — `melt` fraksiyonu + gizli ısı kilidi (geometri YOK) | **✅ DOĞRULANDI** |
| **6a** | **Texel→vertex eşlemesi** — vertex→UV→texel→melt, uzaysal weld YOK | **YAZILDI** |
| **6c** | **Geometri slump** — türetilmiş deplasman, weld'li, yerçekimiyle çökme | **YAZILDI** |
| **6c-2** | **Yanal akış + gölet** — wet-clay çekirdeği, hacim korunumu | |
| **7** | **Kül / kütle kaybı** — büzülme + kül parçacık doğurma | |
| **8** | **Fluid'e kütle devri** (demir dökümü) — en riskli, en sona | |
| **9** | **Node katmanı** — Bölüm B; alttaki sistem oturduktan sonra üstüne arayüz | |

---

# ⏸ OTURUM NOTU — 2026-08-02 sonu

## Test edilmeyi bekleyenler (derlenmedi)

Bu turda yazılan her şey **derlenmedi ve test edilmedi**:

1. **Panel per-object istatistikleri** — `This object` altındaki burning / surface T
   / char / fuel artık yalnızca seçili alanı katlıyor; sahne toplamları ayrı
   **"All objects"** bölümünde. (Bir önceki turda bu satırlar globaldi ve kağıdın
   sıcaklığını "Iron" başlığı altında gösteriyordu.)
2. **Collider ↔ viewport seçim senkronu** — çift yönlü, yalnızca değişimde.
   Yardımcılar `ForceFieldUI::selectedObjectNodeName` /
   `selectSceneObjectByNodeName`.
3. **Faz 3c** Clear Damage / Clear All Damage (+ accumulation reset).
4. **Faz 3d** Kelvin/Draper ışıma.
5. **Fixed-point düzeltmesi** (1<<22 + yuvarlama).

★ **İKİ SHADER DEĞİŞTİ, .spv yeniden derlenmeli:**
`closesthit.rchit` (Kelvin/Draper ışıma) ve `sim_msf_scatter.comp` (yuvarlama).

## Yarın ilk bakılacak: demir yeterince ısınıyor mu?

Son ölçüm sahne toplamıydı (658 K) ve büyük olasılıkla **kağıda** aitti. Panel
artık per-object olduğu için ilk iş: **demiri seçip kendi `surface T`'sini
okumak.**

- Erime için gereken **1811 K**. Şu an tahminen %20–35 bandında.
- Işıma için gereken **798 K** (Draper). Panel `GLOWING` / `below` yazıyor.
- Eğer demir 798 K'ye bile çıkmıyorsa **önce ısı bütçesi** çözülmeli (alev
  sıcaklığı / domain `max_temperature` / `thermal_response`), Faz 6'ya girmeden.

★ Bunu doğrulamadan melt'e başlamak riskli: melt büyük bir iş (aşağıda) ve
bitirdikten sonra "zaten erime sıcaklığına hiç çıkmıyormuş" demek istemeyiz.

## Faz 6a — texel→vertex eşlemesi (YAZILDI, test edilmedi)

> **Bu bölümün eski hali yanlış bir problemi çözüyordu.** "Resolver paylaşılan
> index buffer'ı olmayan bir üçgen çorbası veriyor, o yüzden uzaysal weld
> gerekiyor, o yüzden 6a riskli" deniyordu. Weld hiç gerekmiyor.

**★ Kök içgörü: vertex KİMLİĞİ değil, melt DEĞERİ lazım.** Soru "bu vertex hangi
elemandır?" değil, "bu vertex'in altındaki yüzey ne kadar eridi?". Vertex zaten
bir UV taşıyor ve MSF elemanı zaten bir **texel** (`texel_index`). Yani zincir:

```
vertex → UV → texel → melt
```

Uzaysal eşleştirme yok, dolayısıyla bu kod tabanını daha önce ısırmış olan ayna
simetrisi hayalet eşleşmesi sınıfı **hiç ortaya çıkmıyor**.

**★ Üstelik UV'den gitmek yalnızca güvenli değil, ZORUNLU.** Renderer maskeyi tam
olarak bu zincirle örnekliyor (`closesthit.rchit`, `rawUV`). Başka herhangi bir
eşleme, deplase edilmiş geometri ile akkorluk/char gölgelemesinin yüzeyin NEREDE
eridiği konusunda anlaşmazlığa düşmesine izin verirdi.

### Uygulama

- `MaterialStateField` iki dizi kazandı: `melt_texel` (res²) + `melt_covered`.
- **Maskeyle LOCKSTEP kuruluyor** — aynı geçiş, aynı texel, aynı 1-texel
  dilation. Ayrışsalardı geometri, gölgelemenin erimediği bir yerde erirdi.
- Seam'de `max` (char ile aynı gerekçe): daha çok erimiş komşu kazanır, yoksa
  seam üstündeki vertex komşularıyla birlikte çökmez.
- `sampleMeltAtUV()` UV'yi **wrap** eder, clamp etmez: tiling bir layout meşru ve
  clamp her aralık-dışı vertex'i kenar texel'ine yığıp UV kenarı boyunca fiziksel
  sebebi olmayan bir erime şeridi üretirdi.
- **`false` = "geometriye DOKUNMA", `melt = 0` DEĞİL.** UV'si olmayan mesh
  (`mask_resolution == 0`) ya da boş UV alanına düşen vertex için deplasman
  yapılmaz. Sessizce erimemek görünür ve açıklanabilir; uydurma bir değerle
  sessizce deplase etmek değil.
- `clearField` `melt_texel`'i de sıfırlar (`covered` HAYIR — o UV layout'unu
  tarif eder, hasarı değil).

### ★ Bilinen ve KABUL EDİLEN sınır: aynalı UV

UV adaları aynalı/çakışıksa iki farklı yüzey noktası aynı texel'i paylaşır ve
birlikte erir. Bu yeni bir kusur **değil**: char'ın Faz 3a'dan beri davranışı bu
ve renderer zaten iki tarafı aynı gölgeliyor. Geometrinin gölgelemeyle uyuşması
tutarlı olan cevap; burada "düzeltmek" ikisini birbirinden ayırırdı.

### Panel

`melt lookup (UV): N / M texels reachable (%)` — bir vertex sorgusunun melt'e
ULAŞIP ulaşamayacağını söyler, erime değerlerinden AYRI raporlanır çünkü "hiçbir
şey erimedi" ile "eridi ama UV'den erişilemiyor" farklı arızalar.

## Faz 6c — geometri (SLUMP YAZILDI, akış AÇIK)

### ★ Deplasman TÜRETİLİR, biriktirilmez

Tek karar üç şeyi birden ödüyor:

- `melt` monoton (6b onu yalnızca artırır) ⇒ türetilmiş deplasman da monoton,
  yani erime hâlâ **geri döndürülemez** okunuyor.
- `melt` zaten **Faz 4b frame cache**'inde ⇒ timeline scrub geometriyi bedavaya
  oynatıyor. Yeni cache YOK, yeni serileştirme YOK.
- Clear Damage `melt`'i sıfırlar ⇒ mesh kendiliğinden rest'e döner.

Canlı vertex'lere biriktirseydik üçünün de elle kurulması gerekirdi, üstelik sim
farklı `dt` ile koştuğu her karede birikim farklı olurdu.

### ★★ Bu fazın ASIL tuzağı: UV seam'i mesh'i yırtar

UV seam'i tam olarak **bir uzamsal vertex'in iki farklı UV taşıdığı** yerdir. Her
SoA vertex'ini kendi UV cevabıyla deplase etmek mesh'i her seam boyunca **açar**.
Bu yüzden: örnekleme SoA vertex başına (herkesin kendi UV'si var — zaten amaç bu),
ama sonra **unique vertex'e `max` ile indirgeme**. Texel scatter'ın seam kuralıyla
aynı kural, dolayısıyla ikisi anlaşıyor.

Weld için `rebuildWeldCache(node, map)` — `rebuildSoftWeldCache`'in cache haritası
parametreleştirildi. ★ Eriyen obje `soft_weld_cache_`'e **konmaz**: oraya konan
non-soft bir node'u soft body'nin freeze/reset/frame-cache yolları soft body
sanardı (`rigid_bake_cache_`'in ayrı tutulmasıyla aynı gerekçe). Aynı welder,
ayrı defter. Yazma da ortak: `applyDeformedVertsToCache` soft body'nin
doğrulanmış normal/scatter yazıcısından ayrıldı — iki yazıcı, local/world veya
normal işlemenin ayrışacağı iki yer demekti.

★ Node hem soft body hem eriyen ise melt **geri çekilir**: iki yazıcı her karede
birbirini ezerdi.

### ★ Kare başına çağrı yeri: step değil, RENDER senkron noktası

`Main.cpp`'de `syncMaterialStateFieldMasks`'in hemen yanında. Gerekçe: **scrub
`melt`'i adım ATMADAN geri yükler**; step site'ına asılı geometri, oynatma
devam edene kadar erimemiş kalırdı — Faz 3c/4b'nin tuzağının aynısı. Artık "bu
karenin erimesini göster"in iki yarısı birlikte oluyor, gölgeleme ile şekil
anlaşıyor.

### ★ KAPSAM, açıkça: bu SLUMP, akış DEĞİL

Vertex'ler yerçekimiyle çöker ve rest bbox tabanına yassılır. **Yanal yayılma
yok, hacim korunmuyor.** Taban kelepçesi (`max(rest_min_y, ...)`) objenin
üstünde durduğu yüzeyin yaklaşımı — onsuz eriyen küre zeminin içine geçerdi,
onunla yassılıp gölete dönüşüyor. Panel bunu "slumping under gravity (no lateral
flow / pooling yet)" diye yazıyor ki doğru sonuç bozuk sanılmasın.

Sag miktarı objenin **kendi boyuyla** ölçekleniyor (aynı madde fincanda ve
katedralde aynı davransın) ve `1 - melt_viscosity` ile — yani madde
kütüphanesinin authored viskozitesi nihayet bir işe yarıyor. **Yeni UI parametresi
YOK**, dolayısıyla script/IPC parite yüzeyi de yok.

**AÇIK (6c-2):** yanal akış + gölet + hacim korunumu; çekirdek olarak sculpt
wet-clay sistemi.

## Açık borçlar (değişmedi)

- ~~**SimCache'e MSF serileştirme**~~ → Faz 4b, DOĞRULANDI.
- ~~**Faz 4 WorldThermalState**~~ → aşağıda, DOĞRULANDI.
- **`kMaskKelvinRange`** C++ ile shader arasında ELLE senkron (3000 K).
- **★ Parlama rampası metalleri gösteremiyor** (`closesthit.rchit`, Faz 3d):
  `g = (K-798)/(2400-798)` sonra `g*g*g`. Rampa zaten Draper'da sıfırlanmış,
  üstüne küp alınması dikliği **iki kez** sayıyor. Demir kendi erime noktasında
  — katı halde ulaşabileceği en yüksek sıcaklık — sadece `g = 0,25`; bakır
  1358 K'de `g = 0,043`, yani erimiş bakır neredeyse görünmüyor. Üstelik 2400 K
  tavanı çoğu metalin erime noktasının üstünde, hiçbiri rampanın tepesine
  çıkamıyor. Önerilen düzeltme: üs 3 → 1,5 (`g*sqrt(g)`) — demir 0,25→0,50,
  bakır 0,04→0,21; Wien kuyruğunun dikliği korunur. **Görsel kalibrasyon,
  kullanıcı kararı bekliyor.**
- `resolveLightweightObjectOBBForSimulation` + `simulation_local_bounds_` artık
  **ölü kod** (rotate regresyonu nedeniyle devre dışı bırakıldı); rotate testi
  doğrulandığına göre sökülebilir.

## Faz 1 — YAZILDI (test edilmedi)

### İki tasarım düzeltmesi

**1. `solid_gas_owner` kanalı gereksiz çıktı.** İlk keşifte "ızgarada obje kimliği
yok, eklenmeli" diye not edilmişti. Tasarım vertex/eleman-merkezli kurulunca bu
düşüyor: MSF elemanı zaten hangi objeye ait olduğunu bilir, kendi dünya
konumundan hücresini bulur ve gaz sıcaklığını **gather** ile okur. Izgaranın
sahiplik bilmesi hiçbir adımda gerekmiyor. `FluidGrid` değişmedi.

**2. Örnekleme birimi vertex değil, ÜÇGEN AĞIRLIK MERKEZİ.** Collider mesh
resolver (`setColliderMeshResolver`) paylaşılan index buffer'ı olmayan dünya
uzayında bir üçgen çorbası veriyor. Bundan vertex çıkarmak uzamsal weld ister —
bu kod tabanında daha önce ısırmış bir sınıf (ayna simetrisi hayalet eşleşmeleri).
Ağırlık merkezi weld istemez ve `area`'yı bedavaya taşır.

> **Faz 6 borcu:** erime geometriyi hareket ettirir ve gerçek bir vertex
> eşlemesi ister. Bu terfi Faz 6'nın ilk işi; sahte bir vertex dedup'ı ile
> şimdiden gizlenmedi.

### Ne yapıldı

- `MaterialStateField.h/.cpp` — obje başına kalıcı alan, GPU-sahipli
  (`gpu_centers` vec4 xyz+area, `gpu_state` vec4 T/fuel/char/moisture),
  sync-pass tabanlı yaşam döngüsü (senkronlanmayan obje serbest bırakılır),
  topology generation değişince state sıfırlanır.
- `sim_msf_gather.comp` (+ kernel tablosu kaydı, 4 SSBO / 64B push-const) —
  eleman başına en yakın AÇIK hücreyi arar, termal atalet + tutuşma + kömürleşme
  integre eder.
- `ParticleSimulationSystem` üzerine bayrak + stats + readback API'si; hook
  `stepGridDomains` içinde collider gaz kaynağının hemen ardında.
- Collider panelinde toggle + canlı okuma; gaz stats panelinde `gpu_msf_ms`.

### Faz 1'in kasıtlı sınırı

MSF **gaza hiçbir şey yazmıyor** (scatter yok). Voxel yolu kaynak enjeksiyonunun
tek sahibi olarak dokunulmadan çalışıyor. Bu sayede bayrağı açmak simülasyonu
**değiştiremez** — A/B karşılaştırması ancak böyle anlamlı olur. `sim_msf_scatter`
ve `atomicAdd` ile gaza besleme, voxel yolu emekliye ayrılırken gelecek.

### Faz 1'de kapsam dışı bırakılanlar

`SimCache` serialize + imza hash'i ve ayrı Debug Visualizer view'ı **yapılmadı**.
Gözlem için stats paneli okuması yeterli oldu; MSF henüz simülasyonu etkilemediği
için bayat cache riski de yok. İkisi de Faz 3'te (MSF görsele bağlanınca) zorunlu
hale gelir ve orada yapılacak.

### Test noktası

GPU gaz domaini + mesh objesi üstünde `Enable Gas Surface Source` +
`Ignite on Contact` + `Material State Field`. Beklenen: aleve yaklaşınca
`surface T` yükselir, eşik aşılınca `burning` sayısı artar, `char mean` tırmanır
ve `fuel left` düşer. **Kritik ayırt edici:** objeyi alevden uzaklaştır — voxel
yolu kömürleşmeyi sıfırlar, MSF `char mean` değerini korumalı.

## Faz 2 — YAZILDI (test edilmedi)

### Kelvin katmanı

`MaterialTemperatureScale` = `ambient_kelvin` (293) + `kelvin_per_unit` (350).
Dönüşüm **tek noktada**: `MaterialSubstance::fromProfile`. Bu noktanın
aşağısında hiçbir şey Kelvin görmez; shader normalize dünyada kalır.

Varsayılanlar geriye uyumu bilerek koruyor: ahşabın gerçek tutuşma noktası
573 K, bu ölçekle **0.8 normalize**'e düşüyor — yani `ParticleColliderDesc`'in
eskiden beri gelen varsayılanı. Mevcut sahnelerin davranışı değişmiyor. Demirin
erime noktası 1811 K → 4.34 normalize, varsayılan `max_temperature`=10 içinde
rahatça duruyor.

### Madde kütüphanesi

12 profil: Custom, Iron, Steel, Copper, Wood (Oak), Paper, Cloth, Plastic (PE),
Wax, Ice, Stone, Flesh. Ölçülebilir büyüklükler (yoğunluk, özgül ısı, iletkenlik,
erime/kaynama noktası, füzyon gizli ısısı) gerçek fiziksel sabitler; simülasyon
ayar düğmeleri (`fuel_capacity`, `burn_rate`, `char_rate`, `melt_viscosity`)
birbirine göre sıralanacak şekilde seçildi — kağıt hızlı yanıp az bırakır, meşe
yavaş yanıp çok kömürleşir, çelik hiç yanmaz.

Erime/kül/optik alanları **şimdi yazıldı, sonra tüketilecek** (Faz 3 char_color +
molten_emission, Faz 6 melt_kelvin + melt_viscosity, Faz 7 ash_yield). Presetleri
bir kez gerçek değerlerle yazmak, her fazda geriye dönüp doldurmaktan iyi.

`Custom` **ilk sırada ve varsayılan**: collider'ın üç serbest float'ı otorite
kalır, eski sahneler bit-birebir aynı çalışır. Bilinmeyen isim de `Custom`'a
düşer — daha yeni bir sürümde kaydedilmiş proje yüklenmeyi reddetmez.

### UI kararı

Panel hem Kelvin'i hem normalize karşılığını gösteriyor. Tek başına Kelvin
göstermek, yanlış ölçeklenmiş bir domain'de her şeyin sessizce tutuşmasını (ya da
hiç tutuşmamasını) gizlerdi; iki sayı yan yana olunca ölçek hatası görünür olur.

### Faz 2'de kapsam dışı bırakılanlar

**Script/IPC yüzeyi eklenmedi.** Kural gereği yeni yazma yüzeyi üç yere birden
gider (binding + IPC dispatch + capability), ama mevcut `SimulationColliderInfo`
binding'i piroliz ailesinin **hiçbirini** (`gas_ignition_temperature`,
`gas_surface_fuel_capacity`, `gas_surface_burn_rate`) zaten taşımıyor. Tek başına
`msf_substance` eklemek yarım bir yüzey olurdu. Ailenin tamamı MSF'nin API
tasarımıyla birlikte, Faz 9'da eklenecek.

`MaterialTemperatureScale` şimdilik runtime seviyesinde tek; **Faz 4** onu
`WorldThermalState`'e taşıyıp domain override'ı ekleyecek.

## Faz 2b — YAZILDI (SimCache hariç)

### Üç geçişli zincir

`sim_msf_gather` (eleman durumu + salınan kütle) → `sim_msf_scatter` (uint
sabit-nokta `atomicAdd` ile hücre biriktiricilerine) → `sim_msf_resolve` (gaz
alanlarına katıp biriktiricileri temizler). `atomicAdd(float)` kullanılmadı:
`GL_EXT_shader_atomic_float` opsiyonel bir uzantı, Vulkan ise birincil backend —
donanım bağımsızlığı kolaylıktan önce gelir.

Resolve **domain başına bir kez**, tüm alanlar scatter ettikten sonra çalışır;
alan başına çalıştırmak aynı birikimi birkaç kez uygulardı. Biriktiriciler
resolve'un kendisi tarafından temizlendiği için ayrı bir sıfırlama dispatch'i yok.

### ★ Sıralama tuzağı

MSF bloğu **host gaz anlık görüntüsünün upload'ından SONRA** çalışmak zorunda.
Kodda zaten aynı kural için bir yorum vardı (APIC sıvı yüzey yanması); MSF'yi
collider kaynağının hemen ardına koymak, sonraki publication upload'ının
GPU-only birikimi üzerine yazmasına yol açıyordu. Blok taşındı.

### Sökülenler

`sim_gas_collider_source.comp`'un piroliz yarısı (14→10 binding), `surface_state`
tamponu, `solid_gas_ignition/fuel_capacity/burn_rate` ızgara kanalları,
`ParticleColliderDesc`'in `gas_ignition_temperature` /
`gas_surface_fuel_capacity` / `gas_surface_burn_rate` üçlüsü ve serialization'ları,
`Custom` profili, sistem geneli MSF aç/kapa bayrağı.

`gas_ignite_on_contact` **kaldı** — artık "bu obje piroliz yapar" anahtarı.
Collider'ın düz emisyon kanalları da kaldı; onlar madde değil, emitter.

### Per-object override (kullanıcı geri bildirimi sonrası eklendi)

`Custom` sökülünce tek tek malzeme ayarlama imkânı da gitmişti. Kullanıcı haklı
olarak "kağıdın daha düşük ısıda yanmasını isteyebilirim" dedi ve bunu domain
ölçeğinden yapmaya çalıştı — ama **`Kelvin per unit` bir kalibrasyondur, sanatsal
kontrol değil**; onunla oynamak sahnedeki her malzemeyi aynı anda kaydırır.

Çözüm `Custom`'ı geri getirmek değil (o ikinci bir kod yoluydu), profilin üstüne
**`SubstanceOverride`**: `override_ignition` + `ignition_kelvin`,
`burn_rate_scale`, `fuel_capacity_scale`. Varsayılanları no-op.

- Override'lar **Kelvin cinsinden ve dönüşümden ÖNCE** uygulanır — sistemde hâlâ
  tek bir Kelvin→normalize dönüşüm noktası var.
- **Yanmayan maddeler override'ları tamamen yok sayar**; yanma hızını ölçeklemek
  demiri sessizce yakamamalı. Panel bunu açıkça yazar, sessizce yutmaz.
- Panelde iki bölüm ayrıldı: obje override'ı ve "tüm malzemeleri etkiler"
  etiketli domain kalibrasyonu.

Ayrıca Faz 2'de atlanmış bir eksik yakalandı: `msf_substance` **`SceneSerializer`
tarafına hiç yazılmıyordu** (sadece `ProjectManager`'a eklenmişti). İkisi de
tamamlandı.

### ★ Kalan borç: SimCache

MSF durumu **hâlâ cache'e yazılmıyor.** Faz 1'de "MSF simülasyonu etkilemiyor"
gerekçesiyle ertelenmişti; o gerekçe bu fazda sona erdi. Sonuç: timeline'da
geriye sarınca veya bake edilmiş bir kareyi oynatınca **kömürleşme sıfırlanır**.

Zor olan kısım yapısal: MSF runtime seviyesinde ve obje adıyla anahtarlanmış,
cache ise domain başına `SimulationGridDomainState` saklıyor — ikisi birbirine
oturmuyor. Ya cache girdisine paralel bir MSF anlık görüntüsü eklenecek ya da MSF
sahipliği domain altına taşınacak. Kendi başına bir dilim.

## Faz 2b — orijinal karar notu

Proje henüz sürüm çıkarmadı, dolayısıyla **geriye uyum bir tasarım kısıtı değil.**
Faz 1 doğrulandığına göre iki sistemi yan yana tutmanın tek gerekçesi (kendi
kodumu doğrulamak) ortadan kalktı. Geriye kalan tek şey çift defter tutmak.

Yapılacaklar:

- **`sim_msf_scatter.comp`** — eleman başına salınan yakıtı hedef gaz hücresine
  yazar. Birden çok eleman aynı hücreye düşeceği için toplama gerekiyor;
  `GL_EXT_shader_atomic_float` garanti olmadığından **uint sabit-nokta
  `atomicAdd` + ayrı bir çözme geçişi** kullanılacak (taşınabilir çekirdek GLSL).
- Piroliz artık MSF'de: `sim_msf_gather` salınan kütleyi hesaplar, scatter gaza
  aktarır.
- **Sökülecekler:** `sim_gas_collider_source.comp`'un `surface_state` yarısı,
  `gas_surface_state` tamponu, `ParticleColliderDesc`'in
  `gas_ignition_temperature` / `gas_surface_fuel_capacity` /
  `gas_surface_burn_rate` üçlüsü, `Custom` profili ve ona bağlı ikinci kod yolu.
- `msf_substance` **zorunlu** hale gelir, varsayılan `Wood (Oak)`.
- MSF artık simülasyonu etkilediği için **`SimCache` serialize + imza hash'i
  burada zorunlu** (Faz 1'de kasıtlı ertelenmişti).

> Not: collider'ın *kaynak* kanalları (`gas_density_rate`, `gas_temperature_rate`,
> `gas_fuel_rate`, `gas_flame_rate`, `gas_surface_band_voxels`) kalır — onlar
> piroliz değil, düz emisyon; MSF'nin işi değil.

## Faz 3a — YAZILDI (test edilmedi)

### Anahtar fikir: elemanlar artık texel

Faz 1'de eleman = üçgen ağırlık merkeziydi. Termal durum için doğruydu, görsel
için ölümcül: küpte 12 düz yama. Çözüm maskeyi elemanlardan **rasterize etmek
değil** (bu bilgi eklemez, 12 değeri daha yüksek çözünürlükte saklar) — **örnekleme
kümesini texel'e terfi etmek.**

Her texel kendi dünya konumunu (baricentrik enterpolasyonla) taşır, kendi gaz
hücresini okur, kendi sıcaklığını integre eder. 128×128 maske = ~16k örnek.

**`sim_msf_gather` ve `sim_msf_scatter` hiç değişmedi** — zaten "eleman başına bir
invocation" yazılmışlardı. Doğru soyutlamanın bedava getirisi.

### ★ Çözünürlük bağımsızlığı

Yakıt ve yanma hızı artık **birim ALAN başına**, eleman başına değil. Olmasaydı
maske çözünürlüğünü artırmak sahnedeki yanabilir kütleyi çarpardı: aynı küp,
sırf maskesi inceldiği için daha uzun yanar ve daha çok duman salardı. Üçgenin
gerçek dünya alanı onu kaplayan texel'lere bölünüyor; shader `centers[id].w`'yi
okuyup hem yanma hızını hem kapasiteyi ölçekliyor.

### UV tuzakları ve ele alınışları

- **UV'si olmayan mesh** → üçgen ağırlık merkezi yedeğine düşer (`mask_resolution
  = 0`). Bloklu ama çalışır; sessizce boş alan üretmez. Eşik: üçgenlerin yarıdan
  azı kullanılabilir UV taşıyorsa tamamı yedeğe düşer — birkaç UV'li üçgenden
  oluşan ada-ada bir maske, dürüst bloklu yedekten kötüdür.
- **Dikiş çatlağı** → iki savunma: rasterizasyonda küçük negatif kenar toleransı,
  ve okuma sonrası bir texel'lik dilatasyon. Onsuz her UV adası sınırında saç
  teli kalınlığında bir çizgi kalır ve filtrelenince göze çarpar.
- **Texel'den küçük üçgen** → ağırlık merkezi örneği olarak korunur, alandan
  düşmez.
- **Aynı texel'e düşen iki üçgen** → `max`, üzerine yazma değil; yanmış olan
  kazanır.
- **Örtüşen UV** (mimari modellerde yaygın) → hâlâ hayalet iz üretir. Bu bir
  yedek yolla çözülemez; UV'nin kendi sorunu. Belgelendi.

### Test noktası

Panelde `elements` 12'den binlere çıkmalı. Asıl kanıt **`char max` ile
`char mean`'in AYRIŞMASI**: alevi küpün bir yüzüne tut — `max` 1'e tırmanırken
`mean` düşük kalmalı. Eski per-üçgen örneklemede ikisi birlikte hareket ederdi.
`Char Mask Res`'i 64↔256 arasında değiştirince `fuel left` **değişmemeli** —
çözünürlük bağımsızlığının testi bu.

## Faz 3b — render bağlama (karar alındı, yazılmadı)

### Keşif 1: ABI büyümesine gerek yok

`VkGpuMaterialCore` içinde tam doğru yerde bir `_core_pad0` (float) duruyor;
char maske handle'ı oraya oturuyor ve struct **160 byte'ta kalıyor**. Bu, bilinen
"struct büyüyünce tüm `.spv`'leri yeniden derle, ABI beş yerde" tuzağını
tamamen atlatıyor. `VkGpuMaterialExt`'teki `_ext_pad0/1/2` de char rengi +
molten emission için yeterli.

### Keşif 2 → karar: bağlama per-INSTANCE olmalı

MSF obje başına, materyal ise **paylaşımlı**. Char'ı materyal slotuna yazmak,
aynı materyali kullanan iki kutudan biri yandığında ikisini birden karartırdı —
Faz 3 kararında "mevcut texture'a yıkıcı yazma" için verilen gerekçenin aynısı,
sadece texture yerine materyal slotunda.

**Karar: TLAS instance `customIndex` üzerinden per-instance tablo.** Paylaşılan
materyal doğru çalışır. `VkVolumeInstance` zaten aynı kalıbı kuruyor, oradan
örnek alınacak.

> ★ O kalıbın bilinen tuzağı da geçerli: instance tablosunun uzunluğu TLAS
> eşlemesini takip etmeli, per-frame içeriği değil — boş bir paket eşlemeyi
> silmemeli.

### Keşif 3: yeni tablo gerekmiyor, mevcut `VkInstanceData` genişletilecek

Binding 5'te zaten per-instance bir tablo var (`materialIndex`, `blasIndex`,
8 byte). Char alanları oraya eklenecek → 16 byte. **Bu struct DÖRT yerde tanımlı**
(`VulkanBackend.h` + `closesthit.rchit` + `raygen.rgen` + `shadow_anyhit.rahit`)
ve byte-birebir aynı kalmak zorunda; dosyalarda bu kural için zaten uyarı var.

Obje anahtarı da mevcut: `m_instanceSources`, `m_vkInstances`'a paralel ve
`shared_ptr<Hittable>` tutuyor — instance'tan obje adına, oradan MSF alanına.

Katmanlama: `VulkanBackend`'in simülasyon runtime'ına erişimi yok ve olmamalı.
Sahne katmanı backend'e bir **çözücü callback** verecek
(`obje adı → bindless texture index`), backend instance kurarken onu çağıracak.

### MSF tarafı — YAZILDI

Maske artık **iki kanallı unorm8**: R = char, G = normalize yüzey sıcaklığı.
Tek texture bilinçli: ikisi her zaman birlikte örnekleniyor, ayrı upload
per-frame maliyeti boşuna ikiye katlardı. `mask_revision` ile değişmeyen maske
yeniden yüklenmiyor. Sıcaklık, domain tavanına göre nicemlendiği için o değer
readback ile birlikte taşınıyor — render anında tahmin edilmiyor.

### Vulkan bağlama — YAZILDI

- `VkInstanceData` 8→16 byte, **dört yerde birden** güncellendi (C++ + üç shader).
- `TLASInstance`'a `msfCharTex` / `msfCharPacked`.
- **Tek doldurma pass'i** (`refreshMaterialStateFieldInstances`): instance listesi
  üzerinde bir kez yürür, çözücüyü çağırır. Her instance kurulum yerine tek tek
  eklemek yerine bu seçildi — yeni bir kurulum yeri eklendiğinde sessizce
  atlanamaz. Çözücü yoksa alanlar **sıfırlanır**, bayat bindless index alakasız
  bir texture örneklerdi.
- Katmanlama korundu: backend simülasyonu görmüyor, sahne katmanı
  `setMaterialStateFieldMaskResolver` ile bir kapanış veriyor.
- `closesthit.rchit`: maske **ham mesh UV'siyle** örnekleniyor (materyal UV
  scale/offset yanık izini yanmayan yere kaydırırdı). Char → base color'ı
  `char_color`'a doğru karartır, roughness'ı 1'e çeker, metallic'i düşürür —
  kurum gözenekli saçar, sadece koyulaştırmak siyah boya gibi okunurdu.
  Isı → 0.35 eşiğinin üstünde küpsel eğriyle blackbody (kırmızı→turuncu→ak).
  Eşik şart: her şey oda sıcaklığında yayıyor, eşiksiz her MSF objesi hafifçe
  parlardı.

### ★ DERS: köprünün son halkası olay-güdümlü kapıların arkasındaydı

Zincirin ilk dördü (`fields → mask → texture → bindless index`) baştan çalıştı,
ama ekranda hiçbir şey görünmedi. Nedeni isim uyuşmazlığı değildi — telemetriye
eklenen `Instance lists walked: 0` satırı gösterdi ki `refreshMaterialStateFieldInstances`
**hiç çalışmıyordu**. İki ayrı kapı üst üste binmişti:

1. Backend'deki altı çağrı yerinin hepsi **olay-güdümlü** (sahne yeniden kurulumu,
   materyal ataması, hair ekleme). Hiçbiri her kare çalışmıyor.
2. Sahne katmanındaki köprü `syncWorldDataToBackend` içindeydi, o da
   `g_world_dirty` kapısının arkasında.

Kritik nokta: **MSF alanı sim çalışınca doğuyor, yani son olaydan çok sonra.**
Dolayısıyla çözücü kuruluyor ve bir daha hiç sorulmuyordu. Dahası `g_world_dirty`
kapısı tam olarak obje yanarken kapalı: dünya değişmiyor, değişen sim.

Çözüm iki parçalı:
- `syncMaterialStateFieldBindings()` — her kare çalışan giriş noktası. İsimleri
  yeniden çözer ama instance SSBO'sunu **yalnızca maske index'i/paketlenmiş renk
  gerçekten değiştiğinde** yeniden yükler (`refreshMaterialStateFieldInstances`
  artık `bool` döndürüyor). Maske texture'ı yerinde güncellendiği için bindless
  index sabit; upload pratikte alan doğduğunda/öldüğünde/yeniden boyutlandığında
  bir kez olur. Öncesinde `drainInFlightTraces()` — yeniden yaratılan buffer
  uçuştaki bir trace'in descriptor set'inden okunuyor.
- Köprü `syncWorldDataToBackend`'den çıkarılıp ana döngüde **her dirty kapısının
  dışına** alındı.

Genellenebilir ders: **sim durumu hiçbir `g_*_dirty` bayrağının tarif ettiği şey
değildir.** Sahne kaynaklı senkron kapılarına sim çıktısı asılırsa, kapı tam da
simülasyon ilginçleştiği anda kapanır. Bir de: "0" üç farklı arızayı gizliyordu
(pass hiç çalışmadı / isim çıkarılamadı / isim eşleşmedi); ayırt edici sayaçlar
eklenene kadar yanlış hipotez (isim uyuşmazlığı) kovalanıyordu.

Doğrulandığında telemetri: `fields 1 → mask 1 → tex 1 → idx 1 → lists 1 →
named 1 / unnamed 4097 → bound 1`. `unnamed 4097` normal: parçacık/scatter
instance'ları isimsiz, MSF'yi ilgilendirmiyorlar.

## Faz 3c — Hasar kontrolü (YAZILDI)

MSF hasarı **materyalde de UV texture'ında da yaşamıyor**; objeye ait, kalıcı,
runtime durumu. Bunun doğrudan sonucu: onu geri almanın hiçbir dolaylı yolu yok
— materyal parametresi sıfırlamak, texture silmek, materyali değiştirmek hiçbir
şey yapmaz. O yüzden açık bir kontrol şart:

- `MaterialStateFieldSystem::clearField(object_key)` — tek obje.
- `resetState()` — hepsi (zaten timeline sıfırlamada kullanılıyordu).
- Panelde **Clear Damage** / **Clear All Damage**.

★ Kritik ayrıntı: durum sıfırlamak **yetmez**. Renderer'ın okuduğu şey maske ve
maske yalnızca readback sırasında `scatterCharMask` ile yeniden kuruluyor. Eski
`resetState()` maskeye dokunmuyordu, yani obje bir sonraki readback'e kadar
yanık görünmeye devam ederdi. `clearField` maskeyi de sıfırlıyor **ve
`mask_revision`'ı artırıyor** — artırmazsa köprü "değişmemiş" diye upload'ı
atlar ve temizleme hiç görünmez.

★ İkinci ayrıntı (kullanıcı raporu): temizlemek sahnede *başka* hiçbir şeyi
oynatmıyor, dolayısıyla biriken görüntüyü geçersiz kılan hiçbir yol yok — obje
alakasız bir şey render'ı tetikleyene kadar yanık görünmeye devam ediyordu.
Butonlar artık `resetCPUAccumulation` + `resetAccumulation` + `start_render`
tetikliyor. (Bu, "post-process asla accumulation reset etmez" kuralıyla
çelişmiyor: burada gerçekten yüzeyin görünümü değişti, tonemap değil.)

## Faz 3d — Kızıl ısı ışıması (YAZILDI)

Kararma çalışıyordu ama **hiçbir şey parlamıyordu**, ve sebebi bir birim hatasıydı:
maskenin G kanalı sıcaklığı **domain'in `max_temperature` tavanına oranla**
nicemliyordu. Varsayılan tavan 10 normalized; kullanıcının gerçekten sıcak olan
yüzeyi 1.212 normalized (≈717 K) idi → maskede 0.12 → shader'ın 0.35 eşiğinin
çok altında → hiç ışıma yok.

Daha derin sorun oran değil, **otorite**: akkorluk sıcaklığın fiziksel bir
özelliği, solver ayarının değil. Tavana oranla nicemlemek, aynı sıcak objenin
bir domain slider'ı kaydırıldı diye parlamasına ya da sönmesine yol açıyordu.

- Maske G artık **mutlak Kelvin**: `MaterialStateField::kMaskKelvinRange = 3000 K`
  (8 bit → ~12 K adım; çeliğin kaynama noktasını da kapsar).
- Shader eşiği **Draper noktası (798 K)** — ısıtılan bir yüzeyin gözle görülür
  şekilde kızarmaya başladığı gerçek fiziksel sabit. Draper→2400 K bandı donuk
  kırmızıdan ak-sıcağa kadar okunabilir aralığın tamamı.
- `scatterCharMask` artık `MaterialTemperatureScale` alıyor (tek Kelvin dönüşüm
  noktası ilkesi korunuyor), `readback_max_temperature_` yerini `readback_scale_`
  aldı.
- Panel normalize sayı yerine **Kelvin** gösteriyor, ayrıca maddenin erime ve
  tutuşma noktasını ve Draper eşiğini yanına yazıyor. "Parlamıyor" ile "henüz
  yeterince sıcak değil" ancak böyle ayırt edilebiliyor.

★ Uyarı: `closesthit.rchit` içindeki `kMaskKelvinRange` sabiti C++ tarafıyla
elle senkron; biri değişirse ikisi birden değişmeli.

## ★ Faz 3a'nın gizli bedeli: fixed-point ölçeği terfiyle birlikte ayarlanmadı

Kullanıcı "yanan obje neden tutuşmuyor, o mekanizmayı mı söktük" diye sordu.
Mekanizma sökülmemişti — gather→scatter→resolve zinciri, `flame_level`,
`smoke_yield`, `heat_release` hepsi yerindeydi. Sorun **sayısaldı**.

`kFixedPointScale` 1<<16 idi ve bu değer eleman = **üçgen** iken seçilmişti.
Faz 3a elemanları **texel**'e terfi ettirince eleman başına salınan kütle ~3 kat
büyüklük düştü, ölçek gözden geçirilmedi. Üstüne scatter shader'ı `uint(x)` ile
**kesiyordu** (yuvarlamıyordu), yani her eleman her adımda tek yönlü kayıp
veriyordu:

| maske elemanı | fuel | duman | ısı |
|---|---|---|---|
| 6.336 | −10% | **−14%** | −3% |
| 25.000 | −49% | **−100%** | −36% |
| 100.000 | −100% | −100% | −100% |

Yani obje kararıyor ama gaza **hiçbir şey salmıyordu**; alev beslenecek yakıtı
bulamıyordu. Kullanıcının gözlemi ("domain density düşük kalınca alevleri
görememişim") bu hatanın tam olarak beklenen belirtisi.

★ Daha sinsi tarafı: bu, Faz 3a'nın kendi **çözünürlük bağımsızlığı** garantisini
bozuyordu. Alan ölçeklemesi toplam kütleyi sabit tutuyor, ama eleman başına
kesme kaybı eleman sayısıyla büyüyor — yani ince maske daha az duman veriyordu.
Garantinin ihlali fizik katmanında değil, nicemleme katmanındaydı.

Düzeltme: ölçek 1<<22, ve scatter'da `uint(x + 0.5)` (yuvarlama). Kesme hatası
%14 → %0; taşma için bir hücreye tek adımda 1024 kütle-birimi gerekir ki
en-yakın-açık-hücre araması bunu üretemez.

**Genel ders:** bir örnekleme kümesinin çözünürlüğünü N kat artırmak, o kümeden
türeyen her **sabit nicemleme ölçeğini** N kat daha kaba hale getirir. Terfi
ederken fizik doğru ölçeklendi, nicemleme unutuldu — ve semptom "fizik yanlış"
gibi göründü.

### Madde artık alan başına (✅ DOĞRULANDI)

Test: iki obje, biri Iron biri Paper. Sonuç `Objects 2 / 14464`, `Fields 2`,
`Masks 2`, `bound 2`; **demir kararmıyor, kağıt kararıyor.** Düzeltme öncesi
ikisi de ilk collider'ın maddesiymiş gibi davranıyordu.


`ParticleSimulation.cpp` içindeki MSF döngüsü profili **yalnızca ilk
collider'dan** alıyordu (`if (!profile)`) ve `step()` o tek profili bütün
alanlara uyguluyordu. Tek maddeli sahnelerde görünmez, ama farklı maddelerden
iki collider varsa ikincisi birincinin maddesiymiş gibi yanıyordu — ve sessiz
başarısızlık iki yönlüydü: önce demir çözülürse `profile.combustible == false`
olduğu için **scatter dispatch'i komple atlanıyor**, yani yanındaki tahta hiç
duman/alev vermiyordu.

Düzeltme, parametreyi taşımak yerine **otoriteyi alana taşımak**:

- `MaterialStateField` artık `substance_name` **ve** `overrides` taşıyor.
- `step()`'ten `profile`/`substance` parametreleri **kaldırıldı**; her alan kendi
  maddesini `findSubstance(field.substance_name)` ile çözüyor ve `fromProfile`'ı
  kendi override'larıyla uyguluyor.
- Çağrı yerindeki "ilk collider'dan profil topla" mantığı tamamen silindi.

★ Genel kalıp: obje-başına bir durum kümesi varsa, o kümeyi süren parametre de
obje-başına olmalı. Domain seviyesinden tek bir değer indirmek, tek-nesneli
testte doğru görünüp çok-nesneli sahnede sessizce yanlış sonuç veren bir yapı
kuruyor.

★★ Aynı kalıbın **okuma** tarafı da vardı ve ilk düzeltmede gözden kaçtı: panel
`This object: Iron` yazarken altındaki sıcaklık/char/fuel satırları hâlâ **tüm
alanların toplamıydı**. Yani kağıdın 658 K'si "Iron" başlığı altında gösterilip
demirin 1811 K erime noktasıyla kıyaslanıyordu. Obje-başına bir durum, yalnızca
toplamda okunuyorsa değersizdir. Panel artık seçili alanın host aynasını kendisi
katlıyor; sahne toplamları ayrı bir **"All objects"** bölümünde duruyor.

Ayrıca collider listesi ile viewport seçimi çift yönlü bağlandı: collider'ın
kendi seçim kimliği yok, `source_name` ile bir objeye bağlı — panel bir
collider'ı düzenlerken gizmo başka bir objede olabiliyordu. Senkron **yalnızca
değişimde** yapılıyor (her karede zorlamak bir tarafı otoriter yapar ve
kullanıcının listeden seçimini anında geri alırdı), ve collider'ı olmayan bir
obje seçmek panelin satırını bozmuyor.

### ★ Bilinen maliyet: readback + yeniden yükleme

Maske GPU'da üretiliyor ama RT tarafında **örneklenebilir bir VkImage** olması
gerekiyor ve ikisi ayrı bağlamda. Şimdilik durum okunup texture yeniden
yükleniyor. `mask_revision` ile değişmeyen obje atlanıyor ve maliyet panelde
"readback" satırında **görünür** — gizlenmiş bir varsayım değil.

Round-trip'i tamamen atlayan "compute doğrudan image'a yazar" yolu takip işi
olarak duruyor; şu an yapılmadı, çünkü bağlamlar arası image paylaşımı kendi
başına bir dilim.

### Kalan — (tamamlandı)

1. `VkInstanceData` genişletme (1 C++ + 3 shader, byte-birebir).
2. `TLASInstance`'a char alanları + çözücü callback.
3. Maskeyi `uploadTexture2D` ile yükle, `updateTexture2DInPlace` ile tazele
   (paint yolunun per-dab kalıbı; VkImage'i her frame yeniden yaratmaz).
4. `closesthit.rchit`: maskeyi örnekle, base color'ı `char_color`'a doğru karart,
   roughness'ı yükselt, G kanalından blackbody emission ekle.
5. RT `.spv`'lerini yeniden derle.

Tek parça halinde yapılacak — yarısı uygulanmış bir `VkInstanceData` değişikliği
derlenmeyen bir build demek.

## Faz 3 kararı — per-texel char maskesi

Küp testi somut sayıyı verdi: **12 üçgen = 12 örnekleme noktası.** Termal durum
için yeterli, görsel kararma için kullanılamaz (her üçgen tek renk olur).

Karar **A**: MSF ikiye ayrılır.

- **Shading kanalları** (`char`, `moisture`) UV uzayında bir maskeye yazılır —
  çözünürlük mesh yoğunluğundan bağımsızlaşır.
- **Geometrik kanallar** (`melt`, `mass_loss`) elemanda kalır, çünkü geometriyi
  onlar hareket ettirecek.

Paint altyapısı bu işi zaten çözmüş durumda ve yeniden kullanılacak:
`WetSimulationState` (per-texel katmanlar), `PaintDirtyRect` (dokunulan bölge),
`wet_seam_links_` (UV seam köprüleri — maske olmadan yanık izi seam'de kesilir).

Bedeli dürüstçe: MSF iki uzaylı olur ve seam bakımı gerekir. Alternatifler
(eleman yoğunlaştırma, vertex'e taşıma) çözünürlüğü mesh'e bağlı bıraktığı için
sorunu ötelerdi, çözmezdi.

## Faz 4 — WorldThermalState + Thermal alan (YAZILDI, test edilmedi)

### ★ Kök bulgu: MSF bloğu domain başına çalışıyordu

Faz 4'ün asıl kazancı ortam sıcaklığı *değil*, onu eklerken ortaya çıkan yapısal
hata oldu. MSF sync + gather bloğu gaz domaini döngüsünün **içindeydi**, ve
gather'ın soğutma dalı "gömülü" ile "bu kutunun içinde hiç değil" durumlarını
**aynı dala** düşürüyordu. Sonuç:

- Her obje, **her gaz domaini için bir kez** yeniden sync ediliyordu (mesh
  resolve + upload).
- Her obje, **her gaz domaini için bir kez soğutuluyordu** — objeyi içermeyen
  domain dahil. Yani sahneye ikinci bir duman kutusu koymak, alakasız bir objenin
  ulaşabileceği sıcaklığı yarıya indiriyordu.

Tek domainli test sahnesinde tamamen görünmez. "Demir yeterince ısınmıyor"
şikâyetinin bir bileşeni büyük olasılıkla buydu.

Düzeltme yapısal: **soğutma gather'dan söküldü**, kare başına bir kez çalışan
ayrı bir `sim_msf_ambient` geçişine taşındı; gather artık domain sınırları
dışındaki elemanın state'ine **hiç dokunmuyor** (erken `return`).

### Katmanlama artık gerçekten çalışıyor

```text
World (her yerde tanımlı)  →  Domain override (kendi AABB'si içinde)
    →  Thermal field (uzamsal, yerel)  →  obje MSF
```

- `WorldThermalState`: `ambient_kelvin`, `kelvin_per_unit`,
  `convection_coefficient`, `oxygen_availability`.
- `MaterialTemperatureScale` **silinmedi** — artık türetilmiş dönüşüm nesnesi
  (`world.scale()`), yani tek Kelvin→normalize noktası ilkesi korunuyor.
- Domain override: `thermal_override_enabled` + `thermal_ambient_kelvin` +
  `thermal_oxygen`. **Eleman başına** AABB testi — bir obje domain duvarını
  gerçekten yarı yarıya kesebilir.

### ★ `kelvin_per_unit` için domain override'ı BİLEREK YOK

Maske akkorluğu **mutlak Kelvin** cinsinden nicemleniyor (Faz 3d). İki domain bu
eşleme üzerinde anlaşmazsa, aynı obje hangi kutuda durduğuna göre farklı
sıcaklık raporlar ve farklı parlar. Ambient ve oksijen sınır koşuludur; birim
eşlemesi değildir. Panelde bu açıkça yazıyor, çünkü kullanıcı onu orada arayacak.

### Üstel gevşeme, lineer çıkarma değil

Eski `max(0, T - rate*dt)` yalnızca sıfıra doğru soğuyabiliyordu. Sıfırdan farklı
bir ambient ile bu, yüzeyi oda sıcaklığının **altına** sürer ve mutlak sıfırda
kilitler; dahası bir Thermal alanın içinde duran objeyi **ısıtamaz**. Yeni:
`T = amb + (T - amb) * exp(-k*dt)`, hiçbir zaman aşmaz.

### Thermal alan: yeni varlık üretilmedi

`ForceFieldType::Thermal` mevcut force field sistemine eklendi (A.6 kararı).
Kuvvet uygulamıyor; parametresi `thermal_delta_kelvin` (ambient'in **üstüne**
eklenir — iki ocak bir ocaktan sıcaktır, negatif değer soğuk kaynak yapar).

★ **Kuvvet yollarından tek satırla ayrıldı:** `affectMaskForField` Thermal için
`0` döndürüyor. CPU (`evaluateAt`) ve GPU (`sim_force_fields.glsl`,
`sim_fluid_particle_forces.comp`) zaten `affect_mask & system_mask` testi
yapıyordu — **hiçbir shader'a dokunulmadı** ve Thermal alan yanlışlıkla dumanı
itemez.

Kapsam sınırı, sessizce değil açıkça: Box/Cylinder/Cone şekilleri radyal profile
düşüyor (alanın tam yerel dönüşümünü GPU'ya taşımak gerekirdi). Panel bunu yazıyor.

### Readback artık kare başına BİR kez

`readback()` `step()`'in sonundaydı, yani **domain başına** çalışıyordu: iki
domainli sahne kare başına iki kez pipeline'ı durduruyor ve her char maskesini
iki kez yeniden kuruyordu. Dahası gaz domaini olmayan sahnede hiç çalışmıyordu —
panel, ambient geçişi tarafından gayet iyi simüle edilen bir obje için boş
okuyordu. Yeni: `flushReadback()`, domain döngüsünden **sonra**, bir kez.

### Sıra kuralı

Ambient geçişi domain döngüsünden **ÖNCE**. Sonra çalışsaydı gather'ın az önce
yüzeye yatırdığı gaz ısısının bir kısmını aynı adımda geri alırdı ve obje,
fizikten çok geçiş sırasına bağlı bir sıcaklıkta otururdu.

### Bilinen kapsam sınırı

`stepGridDomains` sahne**de hiç domain yoksa** en başta `return` ediyor, yani
"sıfır domain + yanan obje" halinde ambient geçişi de çalışmıyor. Doğru ifade:
*her domainin **dışında** olan obje simüle edilir; sahnede en az bir domain
bulunmak koşuluyla.* Sökülmesi kolay ama bu turun konusu değil.

### Değişen dosyalar

- `MaterialStateField.h/.cpp` — `WorldThermalState`, `ThermalSource`,
  `AmbientZone`, `stepAmbient()`, `flushReadback()`, `step()`'e oksijen.
- `sim_msf_ambient.comp` **(YENİ)** + kernel tablosu (4 SSBO / 48B).
- `sim_msf_gather.comp` — domain-dışı erken çıkış, soğutma söküldü.
- `ForceField.h/.cpp` — `Thermal` tipi + `thermal_delta_kelvin` + JSON.
- `SimulationWorld.cpp` — Thermal için affect mask 0.
- `ParticleSimulation.h/.cpp` — domain thermal override, `world_thermal_`,
  MSF bloğunun döngü dışına taşınması.
- `scene_data.h` — cache imzasına world thermal + domain override + Thermal alan
  (★ `affects_fluid` kapısı Thermal'ı elerdi, ayrı dal eklendi).
- `ProjectManager.cpp` + `SceneSerializer.cpp` — **ikisi birden**.
- `RtApi.h` + `RtApiForceField.cpp` + `RtIpc.cpp` + `RtPython.cpp` — parite.
- Paneller: force field (Thermal parametreleri), collider (World thermal +
  ambient telemetrisi), domain (Thermal Override).

★ **`sim_msf_ambient.spv` ve `sim_msf_gather.spv` yeniden derlenmeli**
(`compile_shaders.bat` `*.comp` joker'i kullandığı için yeni dosya otomatik
alınır).

### İlk çalıştırma sonucu (kullanıcı raporu)

`Objects 1/6336 · Ambient pass on | 1 thermal field · surface T max 947 K ·
GLOWING · %52 erime yolunda · char/fuel 0 (Iron yanmaz, doğru)`.

**Isı bütçesi sorusu cevaplandı: evet, çıkabiliyor.** Yüzey alanla dengeye
oturuyor, yani tepe sıcaklık ≈ yerel ambient. 293 K + ~650 K ⇒ 947 K, yani
varsayılan 600 K offset. Demiri eritmek için ≈ **1520 K offset** (veya aynı
offset'te `Convection x` düşürmek) gerekiyor. Faz 6'ya girmenin önündeki
"zaten hiç erime sıcaklığına çıkmıyormuş" riski kalktı.

### ★ `ambient_kelvin` ile `kelvin_per_unit` aynı knob DEĞİL

`Kelvin = ambient_kelvin + normalized × kelvin_per_unit`.

- **Ambient = toplamsal kaydırma.** Sıcak/soğuk ayırmadan her elemana aynı sabiti
  ekler, yani kontrastı yok eder. ★ Daha önemlisi ambient pass'in **hedefi** odur:
  obje oraya geri döner ve altına inemez. Alev sönse bile yüzey sonsuza dek kızıl
  kalır — ve bu bir "soğuma çalışmıyor" bug'ı gibi görünür.
- **`kelvin_per_unit` = çarpımsal kazanç.** Yalnızca gerçekten ısınmış elemanı
  büyütür; soğuk gövde ambient'te kalır, `max/mean` ayrışması korunur. Eşikler de
  aynı `MaterialTemperatureScale`'den geçtiği için (`fromProfile`) erime/kaynama
  noktaları kendiliğinden yeniden ölçeklenir — tutarlı bir kalibrasyon, hile değil.

**Kural: gain'i `kelvin_per_unit`'ten al, ambient'i 293 K'de bırak** (sahne
gerçekten fırın değilse). Biri sahneyi ısıtır, diğeri ateşi.

Yan etki: domain `max_temperature` tavanı **normalize** birimde, dolayısıyla kpu
büyüdükçe aynı tavan çok daha yüksek bir Kelvin'e karşılık gelir ve üst sınır
olarak koruması zayıflar. Ambient'i erime noktasının üstüne çıkarmak da objeyi
hiçbir ısı kaynağı olmadan eritir (fiziksel olarak doğru, kazara yapılırsa
"kendiliğinden eriyor" gibi görünür).

### ★ İki UI arızası — ikisi de Faz 4'ün yarattığı iş akışının açığa çıkardığı

**1. Panel, viewport'taki kuvvet alanı seçimini her karede siliyordu.**
`drawForceFieldPanel` içinde Simulation paneli Domains/Colliders/RigidBody
bölümündeyken `clearForceFieldSelection()` **koşulsuz** çağrılıyordu. Viewport'ta
bir alanı seçmek bir sonraki UI karesinde geri alınıyordu — "seçim reddediliyor"
belirtisi. Niyet doğruydu (bölümden çıkınca seçim asılı kalmasın) ama bu bir
**geçiş** eylemi; kare başına eylem yapılınca panel, görünür olduğu sürece
seçimi rehin alıyor. Düzeltme: `section_changed` bayrağı, tek atımlık temizleme.

★ Faz 4 bunu yaşanmaz hale getirdi: Thermal alanı objenin yanına taşırken aynı
anda Colliders bölümünden yüzey sıcaklığını okumak gerekiyor.

★ Kontrol edildi, **suçlu değil**: collider↔viewport seçim senkronu. Kuvvet alanı
seçilince `selectedObjectNodeName` boş dönüyor, `findColliderFor("")` eşleşmiyor
ve panel satırına dokunulmuyor.

**2. Thermal alanda `Affected Targets` kutuları ölü ama canlı görünüyordu.**
Affect mask Thermal için toptan sıfırlandığından o dört kutu hiçbir şey yapmıyor;
kullanıcı alanın çalışmadığını sanınca ilk oraya baktı ("etkilenecek alanları
seçmemiş"). Artık disabled + "bu alan kuvvet uygulamaz, hedefi obje seçer:
'Ignite on Contact' açık her collider ısınır" notu.

**Genel ders:** bir panelin görünür olması, sahne seçimi üzerinde otorite kurma
hakkı vermez; ve ölü bir kontrolü ekranda bırakmak, kullanıcıyı arızayı kendi
ayarlarında aramaya yollar.

### Render köprüsü — ✅ SAĞLAM (yanlış alarmdı)

İlk okumada `Instance lists walked: 0` görünmüştü. Seçim düzeltmesinden sonraki
tur: `lists 1 → named 1 / unnamed 4097 → bound 1` — Faz 3b'de belgelenen sağlıklı
telemetrinin birebir aynısı. Yani kopukluk yoktu; RT sahnesi o an kurulmamıştı.

★ Ama telemetri bunu söyleyemiyordu: sayaç `m_vkInstances.empty()` `return`'ünden
SONRA artıyordu, dolayısıyla "geçiş hiç çağrılmadı" ile "çağrıldı, liste boştu"
aynı `0`'ı üretiyordu — Faz 3b'nin "tek 0 üç arızayı gizler" dersinin bir kat
aşağıda tekrarı. `instance_lists_empty` sayacı eklendi; bir dahaki sefere bu
ayrım panelde yazılı olacak.

### ★ Ayar kuralı: tepe sıcaklık = yerel ambient

Yüzey Thermal alanla **dengeye oturuyor**, yani tepe sıcaklık alanın o noktadaki
ambient'ine eşitleniyor:

```text
tepe ≈ world_ambient + (thermal_delta_kelvin × yüzeydeki falloff)
```

Ölçülen: 796 K ⇒ `offset × falloff = 503 K`. Hedef için offset'i
`(hedef − 293) / 503` ile ölçekle — parlama (900 K) ×1.2, demir erime (1811 K)
**×3.0**. Daha ucuz ikinci yol: `Inner Radius Core` objeyi kapsasın; iç yarıçap
tam güç bölgesidir, falloff kaybı sıfırlanır.

`max 796 / mean 382` ayrışması ayrıca uzamsallık kanıtı: ısınma alana bakan yüzde
toplanmış, tüm objeye yayılmamış (Faz 3a'nın `char max/mean` testinin sıcaklık
kanalındaki karşılığı).

### Test noktası

1. **Thermal alan tek başına:** gaz domaini olmadan (ama sahnede bir domain
   varken) objenin üstüne Thermal alan koy, `thermal_delta_kelvin = 1600`.
   Beklenen: panel `surface T` tırmanır, 798 K'yi geçince `GLOWING`. Bu, ısı
   bütçesini gaz solver'ından **bağımsız** ölçmenin yolu — demirin erime
   noktasına çıkıp çıkamayacağı böyle ayrı ayrı sınanır.
2. **Convection:** `Convection x` 0'a çekilince obje hiç soğumamalı (termos).
3. **Oksijen:** `Oxygen` 0 iken tahta kararmayı durdurmalı, ama ısınmaya
   devam etmeli.
4. **★ Kritik ayırt edici (çoklu domain):** aynı sahneye ikinci, uzak bir gaz
   domaini ekle. Objenin ulaştığı tepe sıcaklığı **değişmemeli**. Düzeltme
   öncesi düşerdi.
5. Panelde `Ambient pass: on | N thermal fields, M domain overrides` satırı ve
   `Cost` altında ayrı `ambient` süresi görünmeli.

## Faz 4b — MSF frame cache (YAZILDI, test edilmedi)

Faz 2b'den beri duran borç. Neden Faz 5/6'dan **önce**: her yeni faz aynı deliğe
daha çok state ekliyor. Faz 6 `melt` + `mass_loss` yazacak, yani **geri
döndürülemez geometri değişikliği** — onu scrub'da kaybetmek char'ı kaybetmekten
çok daha kötü, ve retrofit maliyeti her fazda büyüyor.

### Yapısal karar: kardeş, çocuk değil

Cache domain başına `SimulationGridDomainState` tutuyor; MSF obje başına ve
runtime seviyesinde. Faz 4 bu asimetrinin hangi yöne çözüleceğini kesinleştirdi:
**domain dışındaki obje de simüle ediliyor**, dolayısıyla MSF domain'in altına
taşınamaz. Çözüm, `rigid_frame_cache_` / `soft_frame_cache_` /
`particle_frame_cache_` kalıbı: lockstep paralel harita
(`msf_frame_cache_`), ve disk tarafında domain listesinin **yanında** ayrı blok.

### Ham stride değil, isimli kanallar

Diske `kStateStride` bloğu yazılmıyor; altı isimli kanal yazılıyor
(`temperature/fuel/char/moisture/melt/mass_loss`). Gerekçe: stride'da
`released_this_step` gibi **kare içi geçici** ve bir `reserved` slot var. Dosya
formatını stride'a bağlamak, ileride bir scratch slotu eklendiği gün her cache'i
sessizce geçersiz kılardı. `released_this_step` kasten yazılmıyor — gather onu
her adımda sıfırlıyor, scatter aynı karede tüketiyor.

Türetilebilir olan da yazılmıyor: `centers`/`texel_index` syncField tarafından
canlı mesh'ten yeniden kuruluyor, `char_mask` durumdan yeniden üretiliyor.
İkisini saklamak, obje kımıldadığı anda dosyayı **yanlış** yapardı.

### ★ Restore ertelenmiş — çünkü eleman kümesini sadece syncField bilir

Scrub, sim alanlarını yeniden kurmadan ÖNCE bir kareye iniyor; snapshot'ın ait
olduğu eleman kümesi o an var olmayabilir. Bu yüzden:

1. Alan zaten varsa ve uyuyorsa **anında** uygulanır (çalışan bir sim'i
   scrub'lama — yaygın durum).
2. Yoksa `pending_restore_`'a park edilir; `syncField` o objeyi yeniden kurarken
   talep eder. **Rebuild upload'ından SONRA** uygulanır, yoksa taze sıfırlanmış
   state snapshot'ın üstüne yazardı.

★ Eleman sayısı / maske çözünürlüğü uyuşmazsa snapshot **düşürülür**, yeniden
eşlenmez: eleman↔state karşılığını bozan şey tam olarak retopoloji ve maske
çözünürlüğü değişimidir, ve yanık izini alakasız yüzeye bulaştırmak onu
kaybetmekten kötüdür.

★ `applySnapshot` maskeyi de yeniden kuruyor. Scrub bir adım tarafından takip
edilmek zorunda değil (timeline duraklamış inebilir) ve renderer **maskeyi**
okuyor, state'i değil — Faz 3c'deki Clear Damage tuzağının aynısı.

### Diğer ayrıntılar

- `captureSnapshot` host aynasını **kendisi tazeliyor** (`refreshHostState`,
  `readback()`'ten ayrıldı). Aksi halde snapshot, son panel-tetikli readback'in
  bıraktığını kaydederdi — yani yalnızca stats paneli açıkken doğru olurdu.
- `SimCache::kVersion` 1→2. Eski cache reddedilir, çağıran resimülasyona düşer;
  sürüm çıkmadığı için v1 okuyucu tutulmadı.
- Bozuk **tek** MSF girdisi frame'i reddetmiyor, sadece o objeyi düşürüyor: bir
  objenin yanık izini kaybetmek kurtarılabilir, sağlam bir fluid/gas bake'ini
  birlikte çöpe atmak değil.
- `estimateSimCacheBytes()` MSF'yi sayıyor: eleman = maske texel'i, yani 128²
  maske ≈ 6k eleman ≈ obje başına kare başına ~150 KB. Buradaki en hızlı büyüyen
  kalem, "diske bake et" uyarısında görünmesi gerekiyor.
- Rigid düzenlemesinde (`rigid_frame_cache_.clear()`) MSF **temizlenmiyor** —
  fluid cache ile lockstep, rigid ile değil.

### Test noktası

1. Bir objeyi yak, timeline'ı geri sar, ileri sar: `char mean` **korunmalı**.
   Düzeltme öncesi sıfırlanıyordu.
2. Bake al, cache'ten oynat: yanık izi kareler boyunca **tutarlı** olmalı.
3. `Char Mask Res`'i bake sonrası değiştir → snapshot düşmeli (yanık kaybolur),
   sahne **çökmemeli**. Uyuşmazlığın kasıtlı davranışı bu.
4. Frame 0'a reset → hasar gitmeli ve park edilmiş snapshot geri **getirmemeli**.

## Faz 5 — Nem / söndürme (YAZILDI, test edilmedi)

### ★ Sahiplik kuralı: kaynak domain başına, yutak kare başına

Faz 4'ün dersinin doğrudan uygulaması. Nemin üç dokunma noktası var ve her
sürecin **tek** sahibi:

| Süreç | Nerede | Sıklık |
|---|---|---|
| Islanma (**kaynak**) | `sim_msf_wet` | sıvı domaini başına |
| Kuruma (**yutak**) | `sim_msf_ambient` | kare başına BİR |
| Yanmayı bastırma (**okuma**) | `sim_msf_gather` | gaz domaini başına |

Islanmanın domain başına olması doğru, çünkü **monoton bir kaynak**: üst üste
binen iki su tankının aynı kalası ıslatması fiziksel olarak sorunsuz, kalas yine
ıslak. Faz 4'te soğutmanın döngüden çıkarılma sebebi tam tersiydi — bir
**gevşeme** domain sayısıyla sessizce ölçekleniyordu. Ayrım kuralın kendisi,
istisna değil. Buharlaşmayı gather ile ambient arasında bölmek çift-soğutma
hatasının aynısını üretirdi; o yüzden yutak tek yerde.

### Fizik: su kaynama noktasında kilit

"Islaksa yanmaz" diye bir bayrak yok. Su önce kaynamak zorunda ve kaynarken
yüzeyi **373.15 K'de tutuyor** — gelen ısının tamamı faz değişimine gidiyor,
dolayısıyla yüzey tutuşma sıcaklığına kuruyana kadar çıkamıyor. Klasik ıslak
odun davranışı bedavaya geliyor.

Eşik ofseti yerine **sıcaklık kilidi** olarak yazıldı, çünkü gerçekten aşağıda
tutulan şey sıcaklık — ve render köprüsü akkorluk için o kanalı okuyor. Fırındaki
ıslak demir çubuk parlamamalı.

`kWaterBoilingKelvin` **tek Kelvin dönüşüm noktasından** geçiyor
(`MaterialSubstance::boil_normalized`); shader sabiti olsaydı her domain
kalibrasyonunda başka bir sıcaklık anlamına gelirdi — Faz 3d akkorluk hatasının
tekrarı.

Söndürme ayrıca **kademeli**: yanma hızı `(1 - moisture)` ile sönümleniyor, yani
hortumlanan ateş bir anda kesilmiyor, geri çekiliyor.

### Madde katmanı

İki yeni alan: `absorbency` (suyu ne hızla emer — metal 0, kumaş 1.0) ve
`dry_rate` (pasif kuruma). Kuruma yüzeyin **kendi sıcaklığıyla** hızlanıyor
(ateşin yanındaki kalas saniyeler içinde, soğuk odadaki çok daha yavaş),
kaynama noktasına oranla normalize edildiği için eğri her kalibrasyonda aynı
şeyi ifade ediyor. Emici olmayan madde ıslanma dispatch'ini **tamamen atlıyor** —
demire çarpan su bedava olmalı, texel başına no-op geçiş değil.

### ★ Bilinen sıra bağımlılığı (dürüstçe)

Islanma, fluid dalının **sonunda** çalışıyor çünkü okuduğu doluluk o domainin
density splat'ı — daha erken çalıştırmak bir önceki karenin suyunu test ederdi.
Sonucu: domain listesinde Gas, Fluid'den **önce** geliyorsa gather o kare bir
önceki karenin nemini görür. Nem karelerle kıyaslandığında yavaş değiştiği için
pratikte önemsiz; yeniden yapılandırmadım, yazıyorum.

### Kapsam dışı

`moisture` **render'a bağlanmadı**. Maske iki kanallı (R=char, G=ısı) ve üçüncü
kanal texture formatını değiştirir. Islak yüzeyin koyulaşması/parlaklığı ayrı bir
dilim.

### Test noktası

1. Tahta objeyi yak, üstüne fluid domaini boşalt → `moisture` yükselmeli,
   `burning elements` **düşmeli**, panel "SUPPRESSED" yazmalı.
2. Suyu kes → nem düşer, obje yeniden tutuşur. **Kritik:** `char` bu süre boyunca
   korunmalı (hasar kalıcı, nem geçici).
3. Demir objeyi ıslat → `absorbency 0`, hiçbir şey olmamalı; panel "water runs
   off" yazmalı.
4. Sıcak yüzey ıslakken **parlamamalı** (`surface T` 373 K'de kilitli).
5. `Wetting liquid domains` 0 iken nem varsa → cache'ten gelmiştir, sudan değil.

## Faz 6b — Erime durumu (✅ DOĞRULANDI)

**Ölçüm (2026-08-03):** demir tepe sıcaklığı tam **1811 K'de takıldı**, 312 eleman
`melting`, 0 `fully molten`, `melt` max 0,137. Kilit tutuyor — olmasaydı Thermal
alan yüzeyi 2000 K üstüne iterdi. Aşımın kilitten önce ölçülmesi de doğrulandı:
`melt` sıfırda donmuyor, ilerliyor. Ayrıca **Faz 4b cache** doğrulandı: maske her
karede üretiliyor ve timeline'da ileri/geri oynatmada doğru maske geliyor.

`melt` yavaş büyüyor çünkü artış hızı aşımla, aşım da kilitle sınırlı — bu
fiziksel olarak doğru: erime hızını ısı akısı sınırlar. Hızlandırmak için tepe
sıcaklığı değil **akı**yı büyütmek gerekir (Thermal offset ↑ veya Inner Radius).

Faz 6 üçe bölündü. **6b (durum)** burada; **6a (texel→vertex)** ve **6c (geometri
akışı)** ayrı. Gerekçe: yarım bırakılmış geometri kodu derlenmeyen bir build ya
da bozulmuş mesh demek. 6b kendi başına tutarlı ve build'i yeşil bırakıyor.

> Bölmenin ikinci gerekçesi — "6a uzaysal weld ister, o yüzden riskli" —
> **yanlış çıktı**; 6a bölümüne bak. Weld hiç gerekmiyordu, çünkü lazım olan
> vertex kimliği değil melt değeriydi ve UV o değeri doğrudan veriyor.

### ★ Nem kilidiyle bilerek AYNI ŞEKİL

İkisi de aynı yasa: birinci mertebe faz değişimi sabit sıcaklıkta enerji yutar.
Su kaynarken yüzeyi 373 K'de, demir erirken 1811 K'de tutuyor. Farklı yazmak tek
bir yasanın iki modeli olurdu.

```glsl
melt += (T - T_melt) * melt_rate * dt;   // aşım ÖNCE ölçülür
if (0 < melt < 1) T = mix(T_melt, T, melt);   // gizli ısı kilidi
```

★ **Aşım kilitten ÖNCE ölçülüyor.** Sonra ölçülseydi yüzey tam erime noktasına
sabitlenir, aşım sıfırlanır ve erime %50'de sonsuza dek durur — kendi kendini
iptal eden bir döngü, ve belirtisi "erime birden duruyor" olurdu.

Kilit **kademeli çözülüyor** (`mix(T_melt, T, melt)`): tamamen erimiş yüzey
serbestçe aşırı ısınabiliyor, ki erimiş demirin kaynama noktasına tırmanmasını
sağlayan bu.

### Erimeyen madde için bayrak değil sentinel

`meltable == false` → `melt_normalized = 1e9`. Shader'da tek karşılaştırma, ve
bir bayrakla bir eşiğin birbiriyle çelişmesi imkânsız.

### Gizli ısı → hız eşlemesi (dürüstçe: ayar, türetme değil)

Gerçek entalpi modeli eleman başına kütle ve özgül ısı ister; bu katman onları
taşımıyor. Taşıdığı şey fiziksel olarak anlamlı **sıralama**: buz (3.34e5) inatçı,
mum (2.1e5) kolay teslim. Referans demirinki, yani demir normalize derece başına
saniyede 1.0 hızla eriyor. Bir ayar eşlemesi olduğu için öyle yazıldı.

### Görsel sonuç bedava geliyor

Eriyen yüzey tanımı gereği erime sıcaklığında, o da **zaten** maskenin ısı
kanalından blackbody olarak parlıyor (Faz 3d). Yeni bir render bağlaması
yapılmadı — üçüncü bir maske kanalı texture formatını değiştirirdi.

★ **Geometri hareket ETMİYOR** ve panel bunu açıkça yazıyor, ki "molten" okuyup
sert duran mesh bir bug gibi okunmasın.

### Test noktası

1. Demire Thermal alan (offset ≈1520 K, Faz 4 ayar kuralı) → `surface T` 1811 K'ye
   yaklaşınca `melt` yükselmeli.
2. **★ Kritik:** erime sırasında `surface T` 1811 K'de **takılı kalmalı** (gizli
   ısı). `melt` 1.0'a ulaşınca sıcaklık tekrar tırmanmaya başlamalı.
3. Tahta objede `melt` **hep 0** kalmalı (sentinel).
4. Buz vs mum: aynı ısıda mum belirgin şekilde daha hızlı erimeli.

## Çalışma kuralları

- **Tur başına tek değişken.** MSF taşıma ile görsel kuplajı aynı turda yapılırsa
  hangi tarafın bozuk olduğu ayırt edilemez.
- MSF grow-only bir tampon olacaksa GPU'ya `.size()` ile değil **eleman sayısıyla**
  yüklenir (bilinen sessiz CPU'ya düşme kök nedeni).
- Her yeni yazma yüzeyi üç yere birden gider: script binding + IPC dispatch +
  capability.

## Kazanımlar

- Mevcut güçlü solver altyapısı korunur.
- Objeler malzeme kimliği kazanır; ateş/ısı sahne genelinde anlamlı hale gelir.
- Ateş ↔ su etkileşimi nem kanalı üzerinden neredeyse bedavaya gelir.
- Kullanıcı karmaşık domain/coupling ayarlarını görsel olarak yönetir.
- Sadece değişen bölüm yeniden hesaplanır.
- Gelecekte yeni solver ve shader tipleri eklendiğinde UI yeniden yazılmaz.
