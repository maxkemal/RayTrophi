# Madde tipleri, hal ve preset'ler — tek tanım tablosu

> **Durum:** AKTİF — 2026-10-08. Kullanıcı build başarılı; T1/T3 hızlı IPC kabulü PASS. T2 kısmi, tam T3 kabulü açık. Tane/MPM birleşmesi
> (H1 B11) ve birleşik madde domain'i bu tablonun üstüne oturur; önce bu.

İlgili: [BIRLESIK_MADDE_DOMAIN_TASARIMI.md](BIRLESIK_MADDE_DOMAIN_TASARIMI.md) (domain),
[MATTER_H1_GRAIN_ROADMAP.md](MATTER_H1_GRAIN_ROADMAP.md) (B11 üç sahip).

---

## 1. Sorun (koddan envanter, 2026-10-07)

### 1a. Aynı "madde" kavramı beş listede

| Liste | Kod | Nerede seçilir | Kapsam |
|---|---|---|---|
| Substance Profile | `SubstanceProfile` / `substanceLibrary()` (`MaterialStateField.cpp`) | flow source, MSF nesnesi | parçacık (tag) / nesne |
| Fluid Preset | `APICSolverParams::FluidPreset` + `applyPreset` | domain paneli (`drawFluidPresetCombo`) | **bütün domain** |
| Chemistry Preset | `FluidChemistryPreset` | domain paneli | bütün domain |
| Initial Model | `SimulationFlowSourceDesc::initial_constitutive_model` | flow source | kaynak |
| Constitutive Model | madde bağlaması (`setFluidSubstanceMaterial`) | domain madde satırı | domain × madde |

Tekrarlar ve kategori hataları:
- Water, Oil, Wax, Plastic üç listede; Sand, Gravel, Soil iki listede.
- **Lava** = Stone'un erimiş hali (`melt_kelvin = 1473` Stone'da zaten var). **Molten Plastic**
  = Plastic, **Wax** (sıvı preset) = Wax maddesi. Bunlar madde değil, **hal**.
- **Wet Sand** = kum + gözenek suyu, yani bir **durum**; pore exchange bunu hesaplıyor.
- **Ice** ile **Water** iki ayrı madde, ama aynı maddenin iki hali. Ice'ın
  `default_constitutive_model` değeri **Fluid** (yalnız Sand/Gravel/Soil'e model atanmış).
- Substance listesinde yanan katılar (Wood, Iron, Paper, Flesh: MSF/gaz yangını) ile
  dökülebilir maddeler karışık; kum emitter'ında "Iron" seçilebiliyor.
- Fluid Preset domain'e **tek** malzeme yazar; çok maddeli domain'le çelişir.
- Kütüphane yerleşik ve salt-okunur; kullanıcı madde **ekleyemez**. Yalnız nesne başına
  sapma (per-object override, `fromProfile()` içinde delta) var.

### 1b. Fiziksel hal yedi ayardan, hiçbiri diğerinden türemiyor

1. domain "Contents": Gas / Liquid / Matter
2. flow source "Emission Phase": Gas / Liquid (`SimulationFlowSourceDesc::Phase`)
3. `GridPhase`: Gas / Liquid
4. bağlamada `SubstancePhase`: Liquid / Solid ("blocks flow")
5. `MatterConstitutiveModel`: Auto / Fluid / Granular / Elastic
6. gaz emitter'ında `FuelPhase`: Gas / Liquid / Solid
7. parçacıkta `kParticleFlagFrozen` + `thermal_liquid_enabled` + `melt_kelvin`/`boiling_kelvin`

Ek bulgular (T2 okuması, 2026-10-07):
- **Donma sıcaklığı domain'den okunuyor, maddeden değil:** `FluidThermalLiquid.cpp` donmayı
  `params.thermal_freeze_kelvin` ile yapıyor (varsayılan 330 K = Wax'ın değeri). Maddenin kendi
  `melt_kelvin`'i bu yolda okunmuyor. Su domain'inde termal zincir açılırsa su 330 K'de donar.
- **İki ayrı "katı" mekanizması:** Frozen bayrağı parçacığı yerinde çiviler (hız 0);
  bağlamadaki `SubstancePhase::Solid` ise etiketi grid'in solid maskesine basar (ve
  `fluid_solid_phase_enabled` ana anahtarına bağlı). Aynı fiziksel durum, iki yol.
- **Doğumda model dört katmanlı zincirle seçiliyor** (`MatterDomainSources.inl`): kaynağın
  Initial Model'i → domain bağlaması → maddenin `default_constitutive_model`'i → domain'in
  `granular_enabled` bayrağı (FluidPreset'ten). Başlangıç sıcaklığı zaten var
  (`fluid_temperature_override` / `fluid_temperature_kelvin`).

Çelişkili bir kombinasyon (örneğin "Solid" fazında "Fluid" modelli Ice) hata vermeden kurulur.
CLAUDE.md'deki "panel yalan söyler" sınıfının malzeme versiyonu bu.

---

## 2. Kararlar (kullanıcı, 2026-10-07)

1. **Uzmanlaşmış domain tipleri (Gas, Liquid) KALIR.** Birleşik Matter domain'i (a) tam
   kurulmadan ve (b) uzman domain kadar hızlı olduğu **ölçülmeden** kaldırılmaz. Bugün
   "yalnız gaz içeren bir Matter domain'i eski Gas domain'ini yakalıyor mu" bile ölçülmüş
   değil → §6 T0 ilk iş.
2. **Maddeler kopyalanıp özelleştirilebilir.** Biçimi ajanın kararı (§4).
3. **Yerleşik olan temel malzemelerdir, gerisini kullanıcı türetir.** Yanan katılar ayrı bir
   liste değil, aynı tabloda kategori (ajan önerisi; odun hem yanan nesne hem dökülen talaş,
   yani aynı madde).

---

## 3. Model: üç katman

Endüstri çizgisi (Houdini MPM/FLIP/Pyro ayrımı): **malzeme ne olduğunu, hal ne durumda
olduğunu, çözücü nasıl hesaplandığını** söyler. Kullanıcı yalnız ilkini seçer.

### 3.1 Madde (Substance): tek tablo, tek kimlik

Mevcut `SubstanceProfile` alanları (termal, yanma, nem, erime/kaynama, gizli ısılar,
`liquid_density`, `liquid_kinematic_viscosity`, granüler sürtünme/kohezyon) korunur.
Yeni alanlar:

| Alan | Değerler | Not |
|---|---|---|
| `category` | `liquid` / `granular` / `solid` / `fuel` / `gas` | UI gruplaması ve emitter filtresi. Davranışı **değiştirmez**. |
| `solid_behavior` | `elastic` / `granular` / `obstacle` | Katı haldeyken constitutive model. Bugünkü `default_constitutive_model`'in yerini alır. |
| `granular_transport` | `dem` / `mpm` | Yalnız `granular` için. B11 sahiplik kuralı buradan okunur. |
| `liquid_yield_stress_pa` | ≥ 0 | Bingham (çamur, bal). 0 = Newtonian. Bugün Mud preset'inde örtük. |
| `gas_channel` | `vapor` / `smoke` / `none` | Kaynama/yanma ürünü gaz şeridinde hangi kanala gider. |

`category` yalnız bir etiket; davranış her zaman faz başına alanlardan okunur. "Granular"
kategorisi DEM'i açmaz, `solid_behavior = granular` açar.

★ **Kural (T1'de uygulandı):** bir alan tabloya **tüketicisiyle aynı partide** girer.
Okuyanı olmayan bir alan panelde düzenlenir ama hiçbir şeyi değiştirmez; bu,
`Volume` varsayılanı ve `fire_enabled` okuyucusuyla aynı "panel yalan söyler" sınıfı.
T1 yalnız `category`'yi ekledi (tüketicisi: seçici gruplaması). `solid_behavior` T2'de,
`granular_transport` T4'te, `liquid_yield_stress_pa` T3'te, `gas_channel` T5'te gelir.

### 3.2 Hal (State): seçilmez, türetilir

- Parçacığın hali = sıcaklığı ile maddesinin `melt_kelvin` / `boiling_kelvin` değerlerinin
  karşılaştırması. Gizli ısı ledger'ı (zaten var) geçişin yumuşaklığını taşır.
- Kaynak tarafında yazılan tek şey **başlangıç sıcaklığı**. "Katı / sıvı başlat" kısayolu
  yalnız sıcaklığı yazar.
- Constitutive model ayrı bir ayar olmaz: `hal + madde → model` tek fonksiyondan çıkar.
  Bugünkü 4 (`SubstancePhase`), 5 (`MatterConstitutiveModel` seçimi) ve 7 (Frozen bayrağı)
  bu fonksiyonun çıktısına dönüşür. Bayrak kalabilir ama **yazarı tek** olur.
- `MatterConstitutiveModel` enum'u çözücü içi değer olarak yaşar; kullanıcıya açılan kopyası
  (Initial Model, bağlamadaki Constitutive Model) kalkar.

### 3.3 Çözücü ve gösterim: türetilir

| Hal + davranış | Çözücü |
|---|---|
| sıvı | APIC şeridi |
| katı + granular + dem | DEM grain |
| katı + granular + mpm | MPM (Drucker–Prager) |
| katı + elastic | MPM elastik (karma yolda **henüz yok**; bkz. §6 T5) |
| katı + obstacle | solid mask (bugünkü `SubstancePhase::Solid`) |
| gaz | Euler gaz grid'i |

"Drawn as" (splat / SDF / fog, `SubstanceRepresentation`) ayrı eksen olarak kalır.

### 3.4 Preset = tarif

Preset yalnız kurulum yapan bir tariftir (Campfire, Explosion; `SceneDataParticlePresets`):
madde + kaynak + domain oluşturur. Malzeme tipi tanımlamaz. Fluid Preset ve Chemistry Preset
listeleri bu anlamda preset değil, maddenin kopyası; §5'te sökülür.

---

## 4. Türetilmiş maddeler (karar 2'nin biçimi)

**Seçim: kalıtım, kopya değil.** Türetilmiş madde = `{name, based_on, overrides}`. Yalnız
değişen alanlar saklanır, gerisi tabandan okunur.

Gerekçe: kopya, tabandaki bir düzeltmeyi (örneğin `ProjectManager` ve `SceneSerializer`'da kayıtlı "Honey etiketli ama viskozitesiz"
hatası gibi bir kalibrasyon) türetilmişlere ulaştırmaz. Ayrıca iki ayrı tam kopya "aynı davranışta iki
tanım" sorununu kullanıcı tarafında yeniden üretir. Houdini'deki "preset + override" ve
USD'nin "inherits" katmanı bu çizgide.

Çözümleme sırası (tek fonksiyon, bugünkü `fromProfile()` delta yoluyla aynı mantık):

    yerleşik taban  →  türetilmiş madde (proje)  →  nesne başına sapma (mevcut)

Kurallar:
- **Yerleşikler salt-okunur.** "Düzenle" = "türet". Böylece yerleşik değerler sürümle
  güncellenebilir, proje bozulmaz.
- **İsim kimliktir** (`substance_tag` = isim hash'i, bugünkü gibi). Yeniden adlandırma yeni
  madde demektir; panel bunu söyler. Var olan parçacıklar eski tag'i taşır.
- Türetilmiş maddeler **projeyle** kaydedilir (`ProjectManager`, `substances` dizisi).
  `based_on` bulunamazsa yükleme hata verir ve maddeyi adıyla raporlar. Sessiz yedek yok
  ("yok ≠ silinmiş").
- Türetilmişten türetme serbest; döngü yasak ve yazarken reddedilir (node döngüsü dersi).
- Değişiklik sonrası etkilenen domain'ler `invalidateScriptSimulation` alır. Malzeme fiziği
  değişti, eski kareler artık bu sahneyi anlatmıyor.

---

## 5. Yerleşik taban seti ve eski tanımların eşlemesi

**Yerleşik (temel):** Water, Oil, Gasoline, Alcohol, Sand, Gravel, Soil, Stone, Wood (Oak),
Paper, Cloth, Plastic (PE), Wax, Iron, Steel, Copper, Flesh.

| Eski | Yeni |
|---|---|
| Ice (madde) | **Water**, katı hal (sıcaklık < 273.15 K, `solid_behavior = elastic`). Yükleyici "Ice"ı Water + başlangıç sıcaklığına çevirir. ★ T2'de doğrula: Ice bugün MSF nesne malzemesi olarak da kullanılıyor; nesnenin başlangıç sıcaklığı erime noktasının altında değilse buz bloğu ilk karede su olur. |
| FluidPreset Water / Oil / Wax / Sand / Gravel | aynı adlı madde |
| FluidPreset Honey, Chocolate | Water/Oil'den türetilmiş, **yerleşik örnek** olarak gelir (`based_on` mekanizmasını gösterir, eski sahneler buna eşlenir) |
| FluidPreset Mud | Soil'den türetilmiş örnek: `granular_transport = mpm`, `liquid_yield_stress_pa > 0` |
| FluidPreset Cohesive Soil | Soil (kohezyon zaten Soil'de 800 Pa) |
| FluidPreset Lava | Stone, sıvı hal (sıcaklık > 1473 K) |
| FluidPreset Molten Plastic | Plastic, sıcak hal |
| FluidPreset Wet Sand | Sand + gözenek suyu (pore exchange). Madde değil. |
| FluidChemistryPreset (tümü) | kimya maddenin alanı; domain düzeyindeki seçim kalkar |
| Initial Model (kaynak) | kalkar; kaynakta başlangıç sıcaklığı |
| Constitutive Model + Phase (bağlama) | kalkar; `hal + madde` fonksiyonu |

★ Sinsi risk: FluidPreset domain'in **parametrelerine** (viskozite, wall slip, damping,
packing) yazıyordu. Madde tablosuna taşınan yalnız fizik alanları olmalı; wall slip ve
damping gibi **sayısal** ayarlar domain'de kalır. Bunu karıştırmak, bir maddeyi seçince
domain'in sayısal ayarlarının sessizce değişmesine yol açar.

---

## 6. Sıra

Her adım kendi kabulüyle kapanır ve CLAUDE.md §1 dört dokunuşunu (RtApi, IPC, Python,
yetki + descriptor overlay) taşır. Panel ile script aynı alanları düzenler.

| Adım | İş | Kabul |
|---|---|---|
| **T0** ◐ ilk sonuç §6b | **Ölçüm, kod yok.** Aynı gaz sahnesi: eski Gas domain'i vs yalnız gaz taşıyan Matter domain'i; aynı şekilde sıvı. ms/kare (`rt.perf`), GPU bellek, sonuç farkı. | Tablo `docs/dev/`'de. Karar 1'in kapısı bu sayılar; fark büyükse önce o kapatılır. |
| **T1** ◐ BUILD + HIZLI IPC PASS 2026-10-08, save/open açık | `category` + türetme (`based_on`/overrides), proje serializer (`substances` dizisi), `substance.list/get/derive/set/remove` (eski `msf.substances`/`msf.substance` söküldü), `rt.substance`. Kütüphane kilitsiz okunan anlık görüntü (`SubstanceLibrary.cpp`). Panel: flow source ve collider seçicilerinin altında madde editörü; seçici kategoriye göre **gruplu** (filtre yok: Wax/Plastic bugün sıvı olarak dökülüyor, filtre T2'deki başlangıç sıcaklığına kadar bekler). | `rt_test_substance_profiles_ipc.py` PASS; türetilmiş madde kaydedilir/yüklenir; IPC'den okunan değer panelle aynı. Tabanı bulunamayan proje hata verir. |
| **T2 ön koşulu (karar)** | Etiketsiz parçacıklar bugün "domain'in tek malzemesi"ni, yani FluidPreset'i kullanıyor. Hal maddeden türeyecekse etiketsiz parçacığın da bir maddesi olmalı: domain'e **Varsayılan Madde** (FluidPreset combo'sunun yerine). Bu, T2 ile T3'ü birbirine bağlar; ikisi aynı partide ya da T3 önce. | — |
| **T2** ◐ KISMİ 2026-10-08 | Hal türetme tek fonksiyon: `hal + madde → model`. Ice → Water eşlemesi. Bağlamadaki Phase/Constitutive ve kaynaktaki Initial Model kalkar; kaynakta başlangıç sıcaklığı. | Su 263 K'de doğar → katı; ısınınca sıvı. Donma/erime maddenin `melt_kelvin`'inden (domain `thermal_freeze_kelvin` sökülür). Frozen bayrağı ile solid mask tek "katı" yoluna iner. Aynı madde iki halde, tek tanım. |
| **T3** ◐ BUILD + HIZLI IPC PASS 2026-10-08, legacy/görsel kabul açık | FluidPreset + FluidChemistryPreset sökülür; eski projeler yükleyicide §5 tablosuyla eşlenir (alan adları da değişir, kural 5). | Eski bir Honey/Lava sahnesi aynı ν ile açılır; domain'in sayısal ayarları değişmez. |
| **T4** | H1 B11: `granular_transport` ile üç sahip (sıvı + DEM + MPM) ve MPM↔tane teması. | B11 kabul matrisi. |
| **T5** | Elastik MPM'i karma yola almak; gaz↔tane bağlantısı. | Ayrı notlar. |
| **T6** | Domain tipleri: yalnız T0 tablosu eşitlik gösterdiğinde ve birleşik yol tamamsa. | Karar 1. |

---

## 6a. T3 (+ T2'nin donma yarısı) uygulama kararları — 2026-10-07

Kullanıcı onayı: domain'e **Varsayılan Madde**, T2 ve T3 birlikte. Okumadan çıkan iki gerçek:
(1) `chemistry_preset` kodda zaten "etiketsiz parçacığın maddesi" olarak kullanılıyor
(`resolveFluidSubstanceProfile(tag, chemistry_preset)`, ~25 dosya) — Varsayılan Madde bunun
adı konmuş hali. (2) `applyPreset` her malzeme için ~25 alan yazıyor; bunların bir kısmı fizik,
bir kısmı o malzeme için ayarlanmış **sayısal** değer (taneli malzemede `flip_blend` 0 olmak
zorunda; bal su ayarlarıyla akmaz).

| Karar | Gerekçe |
|---|---|
| `APICSolverParams::default_substance` (isim) `current_preset` + `chemistry_preset`'in yerine geçer; iki enum ve `applyPreset`/`applyChemistryProfile` sökülür. | Tek kimlik. |
| **Fizik her adımda maddeden çözülür** (`resolveDomainSubstancePhysics`): ν, taneli bünye seti, donma (= maddenin `melt_kelvin`), donma yakını viskozite eğrisi, parsel iletimi, yakıt profili, `granular_enabled` (= maddenin modeli). Domain bu alanları **tutmaz**; panel/IPC/script onları maddeden düzenler. | Kopyalanan fizik, madde düzeltildiğinde domain'e ulaşmaz — "aynı davranışın iki tanımı" geri gelir. |
| **Sayısal ayar = maddenin "solver hints" grubu**, domain Varsayılan Maddeyi seçtiğinde bir kez domain'e yazılır ve domain'de düzenlenebilir kalır (FLIP/APIC karışımı, sönümler, sweep sayısı, wall slip, alt adım tavanı, termal zincir anahtarı ve soğuma hızları). | Bugünkü preset davranışını birebir korur; kullanıcının domain'deki ince ayarı ezilmez. Panel "madde seçilince uygulanır" der, ayrıca "Uygula" düğmesi. |
| Su gibi çözülemeyecek kadar küçük ν: çözücü ν·dt/h² < 1e-3 ise viskoz çözümü atlar (ν=0). | Water preset'i ν=0 yazıyordu ("render edilebilir hiçbir voxelde görünmez"); madde artık gerçek 1e-6'yı taşır, maliyet artmaz. Kural fiziksel: sayısal difüzyon baskın. |
| Çakışan sayılarda **çözücüye kalibre edilmiş preset değeri** kazanır: Sand 35°, Gravel 43° + preset'in E/ν/dilatans seti, Oil ν 1e-4, Wax ν 5e-6. | Preset değerleri yığın derinliğine göre kalibre edilmiş (yorumlarda yazıyor); tablo değerleri değildi. ★ Wax 5e-3 → 5e-6: erimiş mum MSF sahnesi 1000× daha akışkan olur; kontrol listesinde sinsi madde. |
| Yeni yerleşikler: **Honey, Chocolate, Mud** (sıvı). Lava / Molten Plastic / Wet Sand / Cohesive Soil yerleşik **olmaz**: eski projede kullanılmışsa yükleyici bir proje maddesi üretir (`Lava (legacy)` ← Stone, `Molten Plastic (legacy)` ← Plastic (PE), `Wet Sand (legacy)` ← Sand, `Cohesive Soil (legacy)` ← Soil). | Temel set temel kalır (karar 3); eski proje birebir aynı fiziği alır. |
| **Göç:** eski projede `default_substance` yoksa, domain'in kayıtlı fizik değerleri eşlenen tabandan farklıysa bir proje maddesi (`<Domain> material`) türetilir ve fark override olarak yazılır. | Elle ayarlanmış (Custom) domain'ler sessizce değer kaybetmez. |
| T2'nin kalan yarısı (Initial Model, bağlamadaki Phase/Constitutive, Frozen↔solid mask birleşmesi) **sonraki parti**. | Bu parti zaten domain fiziğinin sahibini değiştiriyor; ikisini aynı build'de ayırt edilemez kılmak riskli. |

## 6b. T0 ölçümü — ilk sonuç (2026-10-07, canlı, Vulkan)

Betik: `scripts/test/rt_t0_matter_vs_specialized_perf.py` (2×2×2 m, voxel .04 = 50³, 96 kare,
3 tur dönüşümlü sıra). Log: `docs/dev/t0_matter_vs_specialized_live.json`.

| | Uzman | Matter (tek faz) | Oran |
|---|---|---|---|
| Gaz çözücüsü `total_ms` (medyan) | 19.73 | 19.91 | **1.01×** |
| Sıvı çözücüsü `total_ms` (medyan) | 30.01 | 38.62 | 1.29× (aşağıdaki ★'a bak) |
| Sıvı GPU kernel süresi/kare, ısınmış (60 kare dolum sonrası, 2 tekrar) | 7.00 / 5.61 | 5.96 / 5.76 | **≈1.0×** |

- **Gaz: eşit.** Matter'ın kullanılmayan sıvı fazı adım atmıyor (`measured: false`), maliyeti yok.
- **Sıvı: GPU işi eşit**, aynı kod yolu ve aynı grid. Çözücü `total_ms` farkı (+8.6 ms) host
  beklemesi (`batch_end_ms` 10 → 20, `synchronize_ms` 1.0 → 2.4); kernel süresi değişmiyor.
- **Bellek ölçülmedi:** `perf.get_gpu_memory` simülasyon compute tamponlarını izlemiyor
  (domain oluşturmak toplamı değiştirmiyor). "0 MiB" değil, "izlenmiyor".

★★ **Ölçüm dersi: IPC istemcisinin duvar saati çözücüyü ÖLÇMEZ.** Her IPC çağrısı UI karesinde
sırayla işlenir; `fluid.step` etrafındaki süre uygulamanın kare temposudur ve viewport'un o an ne
çizdiğine bağlıdır. Kanıt: kod aynıyken aynı kol 43 ms'den 98 ms'ye çıktı, ve iki adım arasına
herhangi bir `*.step_stats` okuması girince adım ~586 ms oldu (sıvı/gaz/Matter fark etmeden).
İlk T0 sürümü bu yüzden önce "gaz 1.45× yavaş", sonra "sıvı 1.48× yavaş" gösterdi; ikisi de
artefakttı. Karşılaştırmada yalnız çözücünün kendi `total_ms`'i ve `perf.gpu_kernel_timings`
kullanılır; duvar saati yalnız bilgi.

**Karar 1 için durum:** gaz tarafında kapı geçildi. Sıvı tarafında GPU işi eşit; geriye
host bekleme farkı kalıyor. Bu fark kontrollü koşulda (viewport boşta, kullanıcı uygulamaya
dokunmuyor) tekrar ölçülmeli. Kapanmadan T6'ya geçilmez.

## 7. Açık sorular

- `gas_channel`: gaz şeridi tek "smoke" yoğunluğu mu, yoksa tür başına kanal mı? Buhar ile
  dumanı ayırmak gaz grid'ine kanal ekler. T5'e kadar ertelenebilir.
- Karışım (çamur = toprak + su): bugün türetilmiş madde olarak öneriliyor (§5). Gerçek karışım
  (kütle oranı alanı) yalnız gerekçesi ölçülürse.
- MSF nesne malzemeleri (yanan katılar) bugün `findSubstance` ile **yedekli** arıyor (bulamazsa
  ilk profile düşüyor). T1'de türetilmiş maddeler eklenince bu yedek sessiz yanlış madde demek;
  `tryFindSubstance`'a geçirilmeli.


## 6c. T3 kaynak teslimi ve devir kapatma — 2026-10-08

Önceki oturumun kalan 1–5 işleri tamamlandı: iki node betiği
`viscosity_wall_slip` kullanıyor (opt-in, geri alma ve tiksiz alanın korunması
kontrolleri sürüyor); `rt_test_granular_presets.py` söküldü, Sand/Gravel/Soil
FLIP ve internal friction sıfır sözleşmesi madde testinde. Wax beklentisi
`5e-6` olarak düzeltildi. Descriptor overlay, üretici ve yetki audit'i
`default_substance` sözleşmesini denetliyor; yalnız hata döndüren eski girdiler
sunulan parametre olarak tarif edilmiyor.

`FluidDomainSubstance.{h,cpp}` vcxproj'a ve IDE filters'a eklendi; kütüphane
dosyaları da filters'ta. Değişen/yeni Python kaynaklarının Release kopyaları
eşitlendi; kaldırılan granular test Release'ten de kaldırıldı.

Ek kaynak düzeltmeleri: eski proje göçü küçük viskozite farklarını artık birim
büyüklüğünde mutlak toleransla yutmuyor. `substance.set` sonrası fizik ve
kimya readback'i, türetilmişleri de kapsayacak şekilde hemen yenileniyor;
sayısal solver hints tekrar uygulanmıyor. Legacy FluidObject varsayılan
maddesi de silme-refuse kontrolüne dahil. Rheology ve MSF auto-transfer
betiklerindeki eski preset/viscosity/chemistry okumaları güncellendi.

Yeni dış kabul: `scripts/test/rt_test_domain_substance_ipc.py` geçici domain ile
madde seçimi, eski anahtarların mutasyonsuz reddi, madde düzenlemesinin anında
fiziğe ulaşması, numerical tuning'in korunması, açık yeniden seçimin hints'i
uygulaması ve referans varken silmenin reddini sınar. **Canlı çalıştırılmadı.**

T2 tam kapanmadı: Initial Model, binding Phase/Constitutive ve Ice→Water göçü;
parçacık başına sıcaklık/madde çözümlemesi, Frozen↔solid mask tek yazarlı yol
sonraki partidir (§6a kararı). `FluidThermalLiquid.cpp` hâlâ domain'in çözülmüş
donma değerini kullanır; Water+Wax aynı domain'de ayrı donma eşiği kabulü henüz
verilmez. Descriptor'deki `fluid_flammable` vb. kimya aynaları da fiziksel
olarak sökülmedi; seçme/düzenleme sırasında eşitlenir. T3 yazıldı etiketi kaynak
teslimini belirtir, derleme/legacy migration/görsel kabul PASS anlamına gelmez.

Sıralı kullanıcı kontrol listesi: [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md).


## 6d. Kullanıcı build ve hızlı canlı kabul — 2026-10-08

Kullanıcı build'i tamamladı, açık uygulamaya dış `rt_ipc.py` ile bağlanıldı.
`rt_test_substance_profiles_ipc.py`: iki part PASS (21 yerleşik).
`rt_test_domain_substance_ipc.py`: PASS (seçim, eski girdilerin mutasyonsuz
reddi, anında fizik, numerical tuning/hints ve referans koruması).
Son dış sorgu: geçici T1/T3 domain, madde ve collider kalmadı. Testler
simülasyonu adımlamadı veya projeyi kaydetmedi; fiziksel kabul değildir.

§6c'deki "derlenmedi/canlı çalıştırılmadı" ifadeleri kaynak tesliminin tarihsel
durumudur; güncel durum bu bölüm ve sıra tablosudur. T1 save/open ve T3 eski
proje göçü/Wax görsel kabulü açık; T2 kalan yarısı başlanacak kaynak partisidir.
Devam noktası ve dosyalar: [MADDE_T2_T3_HANDOFF.md](MADDE_T2_T3_HANDOFF.md).
Kanıt: [madde_t1_t3_live_2026-10-08.json](madde_t1_t3_live_2026-10-08.json).
