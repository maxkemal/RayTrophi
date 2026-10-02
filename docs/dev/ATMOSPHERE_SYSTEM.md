# Atmosfer sistemi — tek veri modeli: gökyüzü, bulut, hava durumu, iklim

> **Durum:** AKTIF — 2026-09-29 açıldı; **2026-10-01: Faz 1–3 (3a/3b/3c dahil) DERLENDİ (kullanıcı teyidi); kontrol listelerinin sonuçları bu notta işlenmedi — "derlendi" ≠ "doğrulandı". Faz 4 TASLAK (`ATMOSPHERE_WEATHER.md`), Faz 2b açık.** Aşağıdaki "YAZILDI, DERLENMEDİ" ifadeleri bu tarihten önceki oturumlardan kalma, bayat. Faz 0 bitti; Faz 1a (iklim otoritesi) 2026-09-30 CANLI DOĞRULANDI; **Faz 1b (aerial froxel + yükseklik sisi) 2026-09-30 DERLENDİ, SAYISAL KABUL GEÇTİ (gökyüzü paritesi birebir); göz kontrolleri açık — "Kaldığımız yer".** Eski `volumetric_cloud_layer_roadmap.md` bunun yerine ARŞİV'e alındı.

## ▶ Kaldığımız yer (2026-09-30 gece; 2026-10-01'de derleme teyidi eklendi) — yeni oturum BURADAN başlar

**2026-10-01:** bu alandaki tüm değişiklikler (Faz 1b, 2, 3a, 3b, 3c, güneş kuralı)
derlendi. Sıradaki: NEXT_BUILD_CHECKS listelerinin sonuçlarını işle, Faz 2b, Faz 4 kararı.

**Faz 3b yazıldı, derlenmedi (2026-09-30):** Vulkan RT yol izlemeli bulutlar +
yer gölgesi; eski hacim Vulkan'dan söküldü. `ATMOSPHERE_CLOUDS.md` "Kaldığımız yer".

**Faz 3a yazıldı, derlenmedi (2026-09-30):** bulut otoritesi (`CloudState`) +
GPU bulut alanı + ölçüm IPC'si; ayrıntı `ATMOSPHERE_CLOUDS.md` "Kaldığımız yer".

**Faz 2 yazıldı, derlenmedi (2026-09-30):** iklim → fizik bağlantısı, kontrol
listesi NEXT_BUILD_CHECKS en üstte; kararlar §7 Faz 2.

**Güneş kuralı (2026-09-30, derlenmedi):** dünya güneşi yüzeyi YALNIZ directional
ışıkla aydınlatır, iki backend'de de. RayFusion onu ayrıca ışık sayıyordu (sync
açıkken çift güneş) — söküldü; Nishita'ya geçişte directional yoksa "Sun"
eklenir (`world.ensure_sun_light`). Kontrol: NEXT_BUILD_CHECKS en üst bölüm.
Açık: RT `miss.rmiss` diski ikincil ışınlara da veriyor.

**Faz 1b derlendi ve sayısal kabulü geçti (2026-09-30):** iki cihazda boru
hattı hazır, Probe-Climate 10/10 (LUT refactor'ü sağlam), Probe-Aerial 1–4,
gökyüzü paritesi birebir (NEXT_BUILD_CHECKS Faz 1b başındaki ölçümler).
★ Yüzey paritesi bu probla ölçülemiyor: RayFusion yüzeyleri Rendered'dan ~×1,45
parlak (froxel'den bağımsız, hafızadaki açık "realtime parlak" hatası); o
kapanınca `-Region` yüzey bölgesiyle tekrar koş. Açık: NEXT_BUILD_CHECKS Faz 1b
madde 4, 7–13 (göz, eski proje, animasyon render'ı, OptiX).
Aşağıdaki "Sıradaki iş" metni derleme öncesi yazıldı. Sıradaki iş:
kullanıcı `compile_shaders.bat` + msbuild alır → `NEXT_BUILD_CHECKS.md` en
üstteki "Atmosfer Faz 1b" listesi SIRAYLA (13 madde) → `Probe-Aerial.ps1`
(önce bölgesiz, sonra uzak zemin `-Region` ile parite).

Önce bak (ucuz, bağımsız): liste madde 2 — LUT shader'ı ortak başlığa
(`atmosphere_common.glsl`) taşındı; Faz 1a'nın 0,8065 / 0,8272 ölçümü
aynen tekrarlanmalı. Tekrarlanmazsa geri kalan her şey o kaymanın üstüne
kurulur.

Faz 1b'nin nasıl kurulduğu §7'de. Tasarımda bilinçli kararlar (yeniden açma):
- Froxel görüş-uzayı derinliğiyle dilimlenir (raster derinliği yalnız bunu
  verir); 32 dilim karesel, 62,5 m – 64 km. 64 km ötesi son dilim (az pus,
  fazla değil).
- Gökyüzü pikselleri yalnız SİS bloğunu okur (sky-view LUT havayı zaten
  sonsuza kadar içeriyor; üstüne eklemek çift pus).
- Hava terimi sky-view LUT'un modelinin AYNISI (`skyRadiance`: aynı saçılma,
  aynı görüş sönümü, aynı güneş birimi) — uzak dağ gökyüzü rengine solar.
- Sis aydınlatılan ortam: güneş (transmittance LUT, HG `fog_anisotropy`) +
  sky-view yarıküre ortalaması. Nishita dışı modlarda `fog_albedo` radyans
  olarak okunur (eski düz karışımın aynısı).
- Adım 1'deki "önce bugünkü aerial'i ölç" YAPILMADI: eski formül söküldü ve
  kabul ölçütü eski↔yeni değil RT↔RayFusion. Eski görünümle kıyas isteniyorsa
  derlemeden önceki exe gerekir.

**Faz 1a'dan hâlâ açık üç kontrol** (NEXT_BUILD_CHECKS Faz 1a madde 6, 7, 9):
climate keyframe, kaydet/aç, OptiX nem.

**Sonra: Faz 2** (iklim alanları + fizik bağlantısı, §7). Başlamadan önce
§0.10'daki "render yolu keyframe uygulayıcısı eksik" maddesine karar ver —
Faz 2 de keyframe'lenen alan ekleyecek.

★ Kurallar hatırlatma: `.spv` DERLEME (kullanıcı `compile_shaders.bat` koşar).
CRLF dosyaları yalnız Edit aracıyla ya da bayt korumalı betikle. Bash aracı
heredoc içindeki ters eğik çizgiyi ve tırnakları bozabiliyor — uzun betiği
önce Write ile dosyaya yaz, sonra çalıştır; C++ string'ine `\n` gerekiyorsa
Edit kullan.

Hedef: atmosferi (fiziksel gökyüzü, aerial perspective, bulutlar, hava durumu ve
**iklim alanları** — rüzgar, sıcaklık, nem) **tek bir otoriteden** üretmek; Vulkan
RT, RayFusion (realtime raster) ve fizik sistemleri bu otoriteyi yalnızca
**okur**. Bugün aynı büyüklüğün 3–6 ayrı sahibi var ve birbirlerinden habersizler.

Bu not iki şey içerir: (1) kod okunarak **doğrulanmış** bugünkü durum — her
maddenin `dosya:satır` dayanağı var; (2) faz planı — tasarımdır, değişebilir.

---

## 0. Bugünkü durum (kod okuması, 2026-09-29)

### 0.1 ★★★★ Eski bulut yol haritası koda hiç inmemiş

`volumetric_cloud_layer_roadmap.md` "Güncel Durum" bölümünde şunları **yapıldı**
diye yazıyordu; üçü de kodda yok:

| Not diyordu | Kod |
|---|---|
| `NishitaSkyParams` iki `CloudLayerParams layers[2]` taşıyor, düz `cloud_*` alanları kaldırıldı | `CloudLayerParams` **deponun hiçbir yerinde yok**; düz `cloud_*`/`cloud2_*` alanları `World.h:68-111`'de duruyor |
| Katman başına ayrı hacim (`..._L1`, `..._L2`) | Tek hacim: `kInternalSkyCloudVolumeName` (`Render/VolumetricRenderer.cpp:57`). İki katman açıksa **yoğunluğu büyük olan seçilir, diğeri sessizce düşer** (`:213`) |
| İç bulut hacmi artık `world.objects`/TLAS'a eklenmiyor | Ekleniyor: `scene.world.objects.push_back(volume)` (`:198`) |

Bu, "planın yapıldı dediği şey kodda yok" sınıfıdır (bkz. SSS planı). Bu nottaki
her "durum" iddiası bu yüzden dayanaklı yazılır.

### 0.2 Bulut = genel hacim yoluna iliştirilmiş prosedürel VDB

Bulut ±50–500 km'lik, identity transformlu bir dünya dilimi
(`VolumetricRenderer.cpp:223-235`). Genel hacim yolundan geçer
(`volume_intersection.rint`, `volume_closesthit.rchit`, `VulkanBackend_Volumes.cpp`)
— bunlar birleşik madde domain'i işinin de dokunduğu dosyalar. March varsayılanı
8 adım, zamansal birikim yok.

★ "Kamera hareketine tepkisiz" şikâyeti **incelendi, hata değil**: 2 km
yükseklikteki bulutun birkaç metrelik kamera kaymasında paralaksı fiziksel olarak
küçük. Bu maddeyi yeniden açma.

### 0.3 Aerial perspective: 3D LUT Vulkan'a hiç yüklenmiyor

`AtmosphereLUTData` 3D `aerial_perspective_lut` tanımlıyor (`World.h:46`) ama
Vulkan yükleme yolu atlıyor (`VulkanBackend.cpp:20207` "Currently skipping 3D
aerial perspective LUT upload"). Vulkan RT ve OptiX (`ray_color.cuh:1560`)
mesafe tabanlı analitik bir formülle idare ediyor. **RayFusion'da aerial yok.**
→ Faz 1b: Vulkan'da iki backend de froxel okur; 3D LUT alanı OptiX/CPU'nun
donmuş yolunda kaldı (Vulkan slot [3] artık froxel atlası).

### 0.4 Weather: global skaler kümesi, mekânsal değil, türetilmiyor

`WeatherParams` (`World.h:168-185`): tek anda tek tip (`WEATHER_RAIN/SNOW/DUST/MIST`),
global `intensity/density`, elle girilen `visibility` ve `surface_*_output`
alanları. Buluttan bağımsız — yağmur buluttan değil düğmeden gelir. Vulkan RT
okuyor (`closesthit.rchit:299-312`, `weatherActive()` `:1017`). **RayFusion'da yok.**

### 0.5 RayFusion'da atmosfer yok

`material_preview_*`, `rayfusion_*`, `post_chain.glsl` içinde `cloud`, `weather`,
`aerial` hiç geçmiyor. Gökyüzü dışındaki atmosferik etkinin tamamı eksik.
→ Faz 1b: aerial + yükseklik sisi `raster_post.comp`'ta (DoF'tan önce, her
DoF örneğine kendi derinliğiyle). Bulut ve weather hâlâ yok (Faz 3/4).

### 0.6 ★★★ İklim büyüklüklerinin birbirinden habersiz sahipleri

| Büyüklük | Sahip | Değer / birim |
|---|---|---|
| Ortam sıcaklığı | `World.h:143` `nishita.temperature` | °C, yalnız gökyüzü görünümü |
| | `GasSimulator.h:213` `ambient_temperature` | 293 K (20 °C) — gökyüzü 15 °C derken |
| | `MaterialStateField.h:59,103` `ambient_kelvin` | 293 K |
| | `GridFluidSolver.h:55` `ambient_temperature` | **0** — normalize uzay, K değil |
| Nem | `World.h:142` `nishita.humidity` | 0..1, yalnız pus görünümü |
| | MSF nem (Faz 5) | kendi alanı |
| Rüzgar | `WeatherParams.wind_direction/speed` (`World.h:174`) | "m/s benzeri ölçek" |
| | `ForceField` `Wind` tipi (`ForceField.h:48,192`) | kuvvet; APIC'e sürükleme ile bağlı (`:234-248`) |
| | `FoliageWindSystem` grup başına (`InstanceGroup.h:87`, `FoliageWindSystem.cpp:109`) | birimsiz strength/speed |
| | FFT okyanus materyali (`material_gpu.h:100`) | m/s |
| | `CloudManager.h:48` | **20 m/s sabit kodlu** |

Aynı sahnede gökyüzü 15 °C, gaz 20 °C; bulut 20 m/s ile kayarken ağaçlar
ve su yüzeyi başka yönde sallanabilir. Hiçbiri hata vermez.

### 0.7 ★★★★ Nem kaydırıcısı ÖLÜYDÜ (Faz 1a sırasında bulundu, 2026-09-30)

`nishita.humidity` panelde, IPC'de (`world.set_atmosphere`) ve keyframe'de
vardı; **hiçbir renderer okumuyordu**. Vulkan `makeAtmosphereLUTParamsGPU`
onu `weather.x`'e yazıyordu, `atmosphere_lut.comp` `weather.x`'i hiç
okumuyordu; CPU `AtmosphereLUT` (OptiX + CPU) alana hiç bakmıyordu.
Değer değişince LUT imzası değiştiği için LUT **yeniden kuruluyordu** — yani
maliyet ödeniyor, etki sıfır. "Panelin yalan söylemesi" sınıfının saf örneği.
Faz 1a'da higroskopik Mie büyümesi olarak canlandı (§2).

### 0.8 ★★★ Render yolu nem/sıcaklık keyframe'lerini uygulamıyordu

Dünya keyframe'lerini iki yer uyguluyor: `TimelineWidget.cpp` (sürükleme) ve
`Renderer.cpp` (animasyon render'ı). İkincisinin `need_nishita_update`
listesinde nem ve sıcaklık **yoktu**: timeline'da görünen değişim, render
edilen animasyonda uygulanmıyordu. Faz 1a'da iki yol da iklim bloğunu
`World::setClimate` üzerinden uygular.

### 0.9 LUT parametre paketleyicisi iki kopyaydı

`makeAtmosphereLUTParamsGPU` hem `VulkanBackend.cpp` hem
`VulkanDevicePipelines.cpp` içinde birebir kopyaydı (ikisi de ölü nemi
paketliyordu). Faz 1a'da tek başlığa taşındı: `Backend/AtmosphereLutParams.h`.

### 0.10 Faz 1b sırasında bulunanlar (2026-09-30)

- ★★★ **`fog_height` ölüydü.** Panelde "Fog Height", keyframe'de, JSON'da
  vardı; `calculateHeightFogFactor` parametreyi alıp kullanmıyordu (yoğunluk
  kameranın `y`'sine göreydi). Faz 1b'de gerçek anlamına kavuştu: altında
  düzgün katman, üstünde `fog_falloff` ile incelme.
- ★★★ **Sis ve aerial IPC'de HİÇ yoktu** (`src/Api` altında tek `fog_`
  yok). Kural 1 ihlali; Faz 1b kabul testi bu yüzden koşulamazdı. Artık
  `world.get_aerial/set_aerial`.
- ★★★ **İki dünya keyframe uygulayıcısı vardı ve kaymıştı** (§0.8'deki nem
  hatası tek değildi). `TimelineWidget::draw` her şeyi uyguluyordu;
  `Renderer::updateAnimationState` içindeki kopya sis, godrays, multi-scatter,
  aerial, ozon gücü, bulut ışığı, 2. bulut katmanı, overlay'i atlıyordu.
  Kullanıcı: eski animasyon render'ı (`render_Animation` worker) emekliye
  ayrılmış, sequence render artık viewport yolundan (`g_seq_save_*`) geçiyor.
  Doğrulandı: `render_Animation`'ın çağıranı yoktu. **Söküldü** (766 satır +
  yalnız onun tükettiği `pending_anim_transform_updates` ve
  `SceneData::updateVDBVolumesFromTimeline`), `updateAnimationState`'teki
  dünya kopyası da söküldü. Tek uygulayıcı: `TimelineWidget::draw` — yeni
  keyframe'li dünya alanı YALNIZ oraya eklenir.
  AÇIK (atmosfer dışı): `&& !g_seq_save_active` ile korunan "worker sahipliği"
  kapıları (TimelineWidget `render_owns_timeline`, scene_ui `render_owns_sim`,
  Main.cpp `skip_backend_for_anim`, RtApi) her iki giriş `g_seq_save_active`
  kurduğu için pratikte hep false — ölü görünüyor ama sequence BİTİŞİNDE
  bayrak sırası izlenmeden sökülmedi.
- ★★ **RT binding 8 yazımı ardışık sayıyla yapılıyordu** (`validCount` kadar
  eleman, 0'dan). Boş bir LUT slotu varken [3] bir alt slota kayardı. Artık
  eleman başına yazılıyor (`writeRtAtmosphereDescriptors`).
- ★★ **Güneşin Mie çekirdeği yalnız RT'deydi.** Sky-view LUT Mie fazını 2,0'de
  kırpıyor; kırpılan tepe (`max(0, phaseM − 2)`) yalnız `miss.rmiss`'te
  analitik ekleniyordu. RayFusion geniş haleyi (LUT'ta) gösterip çekirdeği
  göstermiyordu — kullanıcı gözüyle buldu. Ortak `sky_sun_corona.glsl`.
  (Terim hâlâ CPU referansının `mie_density * 0.15` sabitini taşıyor; toz/nem
  ile büyümüyor. LUT alfa kanalına Mie integralini yazıp oradan okumak doğru
  çözüm — Faz 3 bulutla birlikte.)
- Froxel görüntüsü `m_lutImages[3]`'e KONMADI: iki LUT yeniden kurulum döngüsü
  (`i < 4`) onu da yok ederdi. Ayrı üye: `m_aerialFroxel`.

---

## 1. İlkeler

1. **Tek otorite:** `AtmosphereState` çekirdekte yaşar. Backend'ler ve fizik onu
   okur, **yazmaz**. UI kendi kopyasını tutmaz.
2. **Üretici tek, tüketiciler ince.** Pahalı iş (LUT'lar, bulut march'ı, gölge
   haritası) backend'den bağımsız compute geçişlerinde bir kez yapılır; Vulkan RT
   ve RayFusion aynı dokuları **örnekler**. Bu, iki backend arasındaki pariteyi
   **ölçülebilir** yapar — ayrı ayrı yazılmış iki atmosfer ancak göz kararı
   karşılaştırılır.
3. **Fiziksel birimler alan adında:** `temperature_k`, `wind_mps`,
   `relative_humidity`, `pressure_pa`. Birimsiz "strength" iklim modeline girmez.
   (Tüketicinin birimini oku — impulse→hız alanı dersi.)
4. **Türetilen türetilir.** Görüş mesafesi, yüzey ıslaklığı, yağış tipi
   (yağmur/kar) elle girilen alan değildir; nem, sıcaklık ve weather map'ten çıkar.
   Sanatçı kontrolü türetmenin **girdisine** uygulanır, çıktısına değil.
5. **Script/IPC'ye ilk günden açık** (CLAUDE.md kural 1). Her faz, IPC ile sayıyla
   doğrulanabilir bir kabul testiyle biter.
6. **Geriye uyum yükü yok:** eski `cloud_*`, `WeatherParams` ve `nishita.temperature/humidity`
   sahne anahtarları okunmaz; alan adları değişir ki eski veri yanlış okunmasın.
   Eski sahnelerde atmosfer varsayılana döner — bu bilinçli.

---

## 2. Veri modeli

```text
AtmosphereState                      (çekirdek, sahneye serileşir, keyframe'lenir)
├── sky          : fiziksel atmosfer — Rayleigh/Mie/ozon yoğunluk ölçekleri,
│                  güneş (yön, açısal çap, ışınım), gezegen yarıçapı
├── climate      : iklim — HER NOKTADA örneklenebilir alanlar
│   ├── surface_temperature_k, lapse_rate_k_per_m      (sıcaklık profili)
│   ├── surface_relative_humidity, humidity_scale_height_m
│   ├── surface_pressure_pa                            (hava yoğunluğu türetilir)
│   └── wind : base_mps (vec3, zemin), shear (yükseklikle artış),
│              gust (spektrum genliği + frekans), turbulence
├── weather_map  : dünyaya sabitli 2D alan (doku veya prosedürel üretici)
│   kanallar: cloud_coverage, cloud_type, precipitation, wetness_accum
│   rüzgarla advekte edilir (climate.wind'in bulut katmanı yüksekliğindeki değeri)
└── cloud_layers[N] : taban/tepe yüksekliği, tip profili (cumulus/stratus/cirrus),
                      erozyon, faz fonksiyonu, yoğunluk
```

Türetilen büyüklükler (saklanmaz, hesaplanır):

- `sampleClimate(world_pos, time) -> { temperature_k, relative_humidity,
  pressure_pa, air_density, wind_mps }` — **fizik sistemlerinin tek giriş kapısı.**
- Yağış tipi: bulut tabanı ile zemin arasındaki sıcaklık profilinden
  (0 °C izotermi) → yağmur / kar / sulu kar.
- Görüş / pus: nem + aerosol → Mie yoğunluğu → aerial perspective.
- Yüzey ıslaklığı: yerel `precipitation` × süre, buharlaşma sıcaklık ve nemle.

★ `weather_map` kamera merkezli **değil**, dünyaya sabitlidir; yoksa bulutlar
kamerayla birlikte yürür ve 0.2'deki "tepkisizlik" gerçekten hata olur.

---

## 3. Fizik sistemleriyle konuşma — sözleşme

**Evet, iklim alanları fizik sistemlerine bağlanacak — ama tek yönlü.**

### 3.1 Yön: atmosfer → simülasyon (sınır koşulu)

Atmosfer simülasyonlara **ortam** ve **sınır koşulu** sağlar. Simülasyonlar
global atmosfere geri yazmaz. Gerekçe ölçek: atmosfer kilometre ölçeğinde,
simülasyon domain'leri metre ölçeğinde. Bir yangının ısısını global sıcaklık
alanına yazmak ya fiziksel olarak anlamsız (etki km ölçeğinde sıfır) ya da
bir kutunun içinde "yerel iklim" icat etmek demek — bu iş zaten simülasyonun
kendi alanının işi. MSF'nin "ambient zone" kavramı (`MaterialStateField.h:355`)
bu ayrımın doğru tarafında duruyor: yerel override simülasyonda yaşar.

### 3.2 Tüketiciler ve ne okuyacakları

| Tüketici | Bugün | Atmosferden okuyacağı |
|---|---|---|
| Gaz çözücü (`GasSimulator`) | sabit 293 K, stratification ayrı parametre | ortam T(y) ve **lapse rate** — stratification artık gerçek atmosferden gelir; rüzgar açık sınırda giriş hızı |
| `GridFluidSolver` | `ambient_temperature = 0` (normalize) | ★ normalize uzayda kalır; dönüşüm ölçeği ve ofseti atmosferden alır. **Tuzak:** K değerini doğrudan yazmak kaldırmayı 293 kat büyütür — hata vermez, "çok güçlü" görünür |
| MSF (termal + nem) | 293 K, kendi nemi | ortam T, bağıl nem → pasif soğuma hedefi, buharlaşma/kuruma hızı |
| APIC sıvı | yalnız `ForceField Wind` ile sürükleme | global rüzgar, zemin yüksekliğindeki değer; aynı sürükleme modeli |
| Partiküller | force field'lar | global rüzgar ile hava sürüklemesi (kütleye/boyuta göre) |
| Foliage rüzgarı | grup başına birimsiz ayar | global rüzgar × grup ölçeği (ölçekler kalır, **kaynak** değişir) |
| FFT okyanus | materyal başına m/s | global rüzgar (materyal override'ı opsiyonel) |
| Weather yüzey tepkisi | elle girilen `surface_*_output` | türetilen ıslaklık / kar birikimi → MSF nem kanalına **birikim** olarak |
| Jolt rigid/cloth | yok | Faz 5+: aerodinamik sürükleme (cloth önce) |

### 3.3 Force field'larla ilişki

Global rüzgar bir **taban alanıdır**; `ForceField Wind` yerel bir eklemedir
(bina arası rüzgar tüneli, pervane). İkisi toplanır, biri diğerini **silmez**.
Varsayılan global rüzgar `0 m/s` → mevcut sahnelerin fiziği değişmez. Global
rüzgarı taklit etmek için sahneye kocaman bir Wind force field konmuşsa, göç
sonrası rüzgar **iki kez** sayılır — göç notunda ve NEXT_BUILD_CHECKS'te
açıkça işaretlenecek.

### 3.4 Tüketici başına kapı: `inherit_atmosphere`

★★ **Karar (2026-09-30, kullanıcı):** atmosfer yalnız **ortak veri** üretir
(`atmosphere::ambientSnapshot` / `sampleAmbient`); fiziğe çözücü çözücü elle
bağlanmaz. Birleşik madde domain'i (`BIRLESIK_MADDE_DOMAIN_TASARIMI.md`) bu
veriyi **tek bir ortam aşamasında** okur: domain başına tek kapı (atmosfer /
yerel), içindeki bütün çözücüler çözülmüş ortamı oradan alır. Faz 2'nin
APIC / partikül / gaz stratification üzerindeki ayrı `inherit_atmosphere`
bayrakları **geçicidir**, birleşik domain geldiği gün sökülür. Madde domain'i
olmayan görsel tüketiciler (foliage, okyanus) ortak veriyi doğrudan okur.
Yeni çözücü-başı bağlantı (ör. gazın sınır rüzgârı) EKLENMEZ; ortam aşamasına
yazılır.

Aşağıdaki metin kararın öncesidir:

Her tüketicinin kendi değeri kalır, ama yanında `inherit_atmosphere` bayrağı
olur. Açıkken (yeni domain'lerin **varsayılanı**) değer atmosferden örneklenir;
kapalıyken tüketicinin kendi değeri kullanılır (laboratuvar koşulu, iç mekân).
Panel hangi değerin **uygulandığını** gösterir, yalnız kendi alanını değil —
yoksa panel yalan söyler (Volume varsayılanı dersi). IPC'de de aynı: `get`
etkin değeri ve kaynağını (`"atmosphere"` / `"local"`) döndürür.

### 3.5 Zaman ve kare döngüsü

`sampleClimate` zamanı açıkça alır (timeline zamanı). Rüzgar esintisi ve weather
map advekti deterministiktir: aynı kare = aynı değer; önbellekli simülasyon
yeniden oynatılınca atmosfer farklı değer vermez. Keyframe'ler `AtmosphereState`
üzerinde yaşar (bugünkü `WorldKeyframe` alanları yeni adlarla taşınır).
★ Script testleri kare döngüsüne kördür — atmosferin simülasyona gerçekten
ulaştığı, bir kare **oynatılarak** doğrulanır, yalnız `get` ile değil.

---

## 4. Render tarafı

### 4.1 Ortak compute geçişleri (Vulkan, backend'den bağımsız)

★ Düzeltme (2026-09-30): Vulkan RT ve RayFusion **iki ayrı VkDevice**
üzerinde koşar; bir dokuyu fiziksel olarak paylaşamazlar. "Ortak" burada
**aynı shader, aynı parametre paketi, her cihazda bir kez** demektir.
Parite garantisi paketin tek tanımından gelir (`AtmosphereLutParams.h`),
dokunun paylaşımından değil. RayFusion bugün transmittance/sky-view/multi-scatter
LUT'larını zaten okuyor (`material_preview_sky.frag` binding 10-12).

| Geçiş | Çıktı | Ne zaman |
|---|---|---|
| Transmittance / multi-scatter LUT | 2D | `sky` kirlenince (var, yeniden kullanılır) |
| Sky-view LUT | 2D | güneş / `sky` / kamera yüksekliği değişince (var) |
| **Aerial perspective froxel** | 3D (kamera frustum'u, ~32×32×32) | her kare (ucuz) |
| **Bulut gölge haritası** | 2D, dünyaya sabitli, güneş yönünde | güneş veya bulut değişince |
| **Bulut march'ı** | RayFusion: ¼ çözünürlük + zamansal yeniden izdüşüm; RT: miss/atmosfer yolunda tam | her kare |

### 4.2 Tüketiciler

- **Vulkan RT:** miss = sky-view + bulut; closest-hit = aerial froxel + bulut
  gölgesi; weather yüzey tepkisi mevcut `weatherSurfaceActive()` yolundan, ama
  türetilen değerlerle.
- **RayFusion:** gökyüzü geçişi sky-view + bulut örnekler; composite aerial froxel
  uygular; gölge haritası ışık hesabına çarpan. Hafızadaki karar ile aynı
  çizgi: dikiş **lookup**'tır, viewport'ta yeni AS kodu yok.
- **OptiX (donduruldu):** yeni özellik almaz. Karar (2026-09-29, kullanıcı onayı):
  **seçenek (a)** — `AtmosphereState`'ten OptiX'in eski alanlarına salt okunur
  bir eşleme doldurulur, eski bulut/sky kodu çalışmaya devam eder. Eşleme tek bir
  fonksiyondur ve yalnız OptiX başlığına yazar; çekirdekte eski alan adı yaşamaz.

### 4.3 Bulut artık hacim instance'ı değil

Faz 3'te `__RayTrophi_Internal_SkyCloudVolume` ve onu `world.objects`'e sokan kod
**sökülür**. Bulut atmosfer geçişinde çizilir; genel hacim shader'larına
dokunulmaz → birleşik madde domain'i işiyle dosya çakışması yok.

---

## 5. Sınır: birleşik madde domain'i ile

`BIRLESIK_MADDE_DOMAIN_TASARIMI.md` yerel, simüle edilen maddeyi (sıvı, gaz,
katı faz) birleştiriyor. Atmosfer bulutu oraya **girmez**: kilometre ölçeğinde,
parametrik/prosedürel, simüle edilmeyen bir alandır. "Bulut da su buharı" diye
ikisini birleştirmek, farklı ölçek ve farklı çözücüyü aynı depoya koymak olur —
whitewater kararındaki "aynı ad ≠ aynı fizik" dersi.

Temas noktası tek ve tek yönlü: madde domain'i ortam koşulunu `sampleClimate`
üzerinden alır (§3). Görsel geçiş (sim dumanının uzakta pusa karışması) aerial
froxel'in simülasyon hacmine de uygulanmasıyla olur — ayrı iş, Faz 4 sonrası.

---

## 6. IPC / Python yüzeyi

Mevcut `world.get_atmosphere/set_atmosphere` (gökyüzü ortamı) ve
`world.get_thermal` adlandırmasıyla tutarlı olsun diye düz `world.*` adları
kullanılır (2026-09-30 kararı; ilk taslaktaki `world.atmosphere.*` iç içe ad
alanı terk edildi).

| Metot | Durum |
|---|---|
| `world.get_climate` / `world.set_climate` | ✎ Faz 1a — `applied_*` alanları RENDER PAKETİNDEN okunur (ayna kopması ölçülebilir) |
| `world.sample_climate { position }` | ✎ Faz 1a — ISA profili, fiziğin giriş kapısı |
| `world.set_atmosphere` | humidity/temperature anahtarlarını artık **reddeder** |
| `world.get_weather_map`, `world.set_weather_map`, `world.sample_weather_map` | Faz 3 |
| `world.get_clouds`, `world.set_cloud_layer`, … | Faz 3 |
| `world.get_aerial` / `world.set_aerial` | ✎ Faz 1b — hava aç/kapa + sis (yoğunluk 1/m, yükseklik m, falloff 1/m, mesafe m, albedo, anizotropi). Emekli `aerial_*`, `fog_color`, `fog_sun_scatter` REDDEDİLİR |
| `world.atmosphere_stats` | ✎ Faz 1b — cihaz başına (render / viewport): `lut_ready`, `froxel_available`, `froxel_active`, `froxel_dispatches`. ms YOK: dispatch kare komut tamponunda, ayrı zaman sorgusu kurulmadı (Faz 3 bulut march'ıyla birlikte) |

Tüketici tarafında her domain'in `get`'i etkin ortam değerini ve kaynağını
döndürür (§3.4). Her yeni namespace: `RtIpcSecurity.cpp` yetkisi,
`gen_ipc_descriptors.py` + overlay özeti, `audit_ipc_capabilities.py`.

---

## 7. Fazlar ve kabul testleri

Her faz sayıyla doğrulanır; görüntü en fazla bir mozaik.

### Faz 0 — plan ✔ (bu not)

### Faz 1a — iklim otoritesi ✔ CANLI DOĞRULANDI (2026-09-30)

Ölçümler (exe 00:34, `atmosphere_lut.spv` yeniden derlendi, 12544 bayt):

| Kontrol | Sonuç |
|---|---|
| `Probe-Climate.ps1` | 10/10 PASS — RH 0,75 → Mie ×2,0; 300 K → 26,85 °C; p(1000 m) 89874,8 Pa; ρ₀ 1,2250 |
| `Probe-AtmosphereCost.ps1 -Field surface_relative_humidity` | kontrol 32,0 ms / nem 29,2 ms, drenaj 0 — LUT düzenlemesi bedava |
| `render.probe` ortalama parlaklık, RH 0,1 → 0,9 → 0,1 | Rendered (Vulkan RT) 0,8065 → 0,8272 → 0,8065; Realtime (RayFusion) 0,7990 → 0,8201 → 0,7991; NaN 0 |
| Kullanıcı gözle | "nem artık doğru davranıyor" |

İki backend aynı yönde, neredeyse aynı büyüklükte (+0,0207 / +0,0211) ve
deterministik geri dönüyor: aynı LUT shader'ı iki cihazda aynı paketi okuyor.

Kapsam kararı: gökyüzü alanlarının (`nishita.*`) toptan taşınması **ertelendi**.
`WorldData` bugün hem otorite hem GPU paketi; ~20 dosya doğrudan okuyor ve bulut
alanları Faz 3'te zaten sökülecek. Taşınan yalnızca **sahibi belirsiz** olanlar:
sıcaklık, nem, rüzgar (+ yeni: lapse rate, basınç).

- `atmosphere::ClimateState` — `include/Atmosphere/AtmosphereClimate.h`,
  `src/Atmosphere/AtmosphereClimate.cpp`. ISA örnekleme, doğrulama (red,
  kırpma yok), higroskopik Mie ölçeği, JSON, keyframe yardımcısı.
- `World::setClimate/getClimate/sampleClimate`; paket alanları `derived_*`
  adıyla yalnız `World::syncClimatePacket()` tarafından yazılır ve
  `setNishitaParams`/`setWeatherParams` sonrasında yeniden senkronlanır (eski
  kopyadaki bayat değer otoriteyi ezemez).
- Nem canlandı: `(1-RH)^-0.5` (Hänel, γ=0,5; RH ≤ 0,95). Vulkan LUT'ta
  `weather.x`, CPU/OptiX LUT'ta `dust_density` çarpanı — aynı fonksiyon.
- Proje JSON: `atmosphere.climate`. Eski `nishita.humidity/temperature`,
  `weather.wind_*` okunmaz. Keyframe: tek `has_climate` grubu (`fclim`, `ctk`…);
  eski `fhum/ftmp/hum/tmp/wtw/wtws` okunmaz.
- OptiX eşlemesi (karar a): OptiX paket alanlarını okur, kodu davranış değiştirmedi.
- Kabul: `scripts/ipc/Probe-Climate.ps1` (ayna, ISA, emekli anahtar, red).

### Faz 1b — aerial froxel + yükseklik sisi ✎ YAZILDI, DERLENMEDİ (2026-09-30)

Eski durum: RT aerial'i sanatsal bir formüldü (`pow(transmittance, distFactor)`
+ `aerial_min/max_distance` rampası + `aerial_density`), sis ayrı bir analitik
düz renk karışımıydı; RayFusion'da ikisi de yoktu.

Kurulan:
- **Üretici (tek):** `shaders/atmosphere_aerial_froxel.comp`. 32×32 hücre ×
  32 dilim, hücre başına bir iş parçacığı, dilim başına 4 adım (128 adım,
  adım başına 1 transmittance örneği). Çıktı 1024×128 RGBA16F atlas, dört
  blok: hava+sis in-scatter / geçirgenlik, yalnız sis in-scatter / geçirgenlik.
  Adım integrali Hillaire'in enerji korunumlu formu.
- **Ortak ortam:** `shaders/atmosphere_common.glsl` — `atmosphere_lut.comp`
  artık bunu include ediyor (yoğunluk, faz, UV eşlemeleri tek yerde).
- **Tüketici sözleşmesi:** `shaders/aerial_froxel.glsl` — atlas düzeni,
  dilim eşlemesi, örnekleme, RT için görüntü düzlemine izdüşüm.
- **Parametre bloğu + görüntü düzlemi:** `Backend/AerialFroxelParams.h`.
  `renderProgressive`'in kamera düzlemi formülü buraya taşındı
  (`makeCameraImagePlane`); RayFusion froxel'ini aynı fonksiyondan kurar.
- **Kayıt:** `VulkanDevice::recordAerialFroxelPass` kare komut tamponuna
  (foton geçişi deseni). Parametreler `vkCmdUpdateBuffer` ile — eşlenmiş bir
  host yazımı önceki karenin okumasıyla yarışırdı. Yeniden kurulum yalnız
  girdi imzası değişince (kamera düzlemi, atmosfer bloğu, sis, LUT varlığı).
- **RT:** raygen `atmosphereLUTs[3]`; `aerialFroxelReady` bayrağı dünya
  paketinde eski `aerialEnabled` slotunda (ofsetler değişmedi, beş shader).
- **RayFusion:** `raster_post.comp` binding 3; her HDR okumasına (merkez + DoF
  örnekleri) kendi derinliğiyle uygulanır. Yalnız Material Preview modunda
  koşar (post geçişi zaten yalnız orada).
- **Veri:** `fog_color` → `fog_albedo`, `fog_sun_scatter` → `fog_anisotropy`;
  `aerial_density/min/max_distance` sökülü (World, keyframe, UI, JSON).
- **OptiX/CPU:** donmuş eşleme — sökülen alanların eski varsayılanları
  sabit, `fog_albedo` renk, `fog_anisotropy` güneş parlaması kazancı olarak.

**Kabul:** `scripts/ipc/Probe-Aerial.ps1` (sayaç davranışı + red + gidiş-dönüş;
`-Region` ile RT/RayFusion farkı %5 içinde). Ayrıntılı sıra:
`NEXT_BUILD_CHECKS.md` "Atmosfer Faz 1b".

### Faz 2 — iklim alanları + fizik bağlantısı ✎ YAZILDI, DERLENMEDİ (2026-09-30)

**Yapıldı** (ayrıntı + kontrol listesi: NEXT_BUILD_CHECKS "Faz 2"):
`atmosphere::publishAmbient` / `ambientSnapshot` — `World::syncClimatePacket`
tek yayın noktası, fizik tarafı bunu okur (worker thread'de güvenli).
MSF/dünya ısısı, gaz ortam T + stratification, foliage, okyanus, APIC sıvı ve
partikül sürüklemesi bağlandı; her birinde `inherit_atmosphere` kapısı ve
API'de etkin değer + kaynak. Sökülen: `InstanceGroup::updateWind` (ölü),
`CloudManager` (ölü). Yeni: `scatter.get_wind/set_wind`, `water.get_wind/set_wind`,
`world.set_thermal` `inherit_atmosphere`/`reference_kelvin`, gaz/partikül/sıvı
ayarlarında `inherit_atmosphere`; kabul betiği `Probe-ClimateCoupling.ps1`.

**Kararlar (plandan sapma değil, plan bunları açıkta bırakmıştı):**
- ★ **Kalibrasyon sıfırı ≠ ortam.** `WorldThermalState.reference_kelvin`
  (normalize 0 = kaç K) ortamdan ayrıldı. Ortam iklimi izlerse ve sıfır onunla
  kayarsa her saklı eleman sıcaklığı her karede yeniden yorumlanırdı — yanan
  bir kütük tam iklim deltası kadar sıçrardı, hata vermeden. Eski projeler
  `reference = ambient` ile yüklenir.
- **Kapı varsayılanı iki sınıf.** MSF/gaz/foliage/okyanus **kapalı** (açmak
  görünür değişiklik: ortam 293→288 K, sakin dünyada foliage durur, okyanus
  düzleşir). APIC/partikül **açık** (rüzgar 0 iken formül birebir eski).
- **Gaz stratification fizikseldir ve minik** (g/cp − L)/kpu ≈ 1e-5/m; preset
  değerleri 0,05–0,25. Kapı bu yüzden kapalı ve panelde uyarılı.
- **Okyanus rüzgârı yalnız GPU malzemesinden geçer** (CUDA FFT emekli); bu
  yüzden `WaterManager::update` her kare malzemeyi yeniler.
- **Sıvı yüzey sürüklemesi** iklim rüzgârının %3'üne (klasik yüzey akıntısı
  oranı) hedeflenir, Wind force field'ın kendi yüzey-sürükleme modeliyle.

**Açık (Faz 2b, plan §3.2'de vardı):** gazın açık sınırda rüzgâr girişi;
`sim.world_thermal` düğümünde inherit; rüzgâr kayması + esinti (ClimateState'e
alan gerekir); `sample_climate` yükseklikle bağıl nem (bugün sabit).

#### Faz 2 özgün plan metni (referans)

- `climate` alanları, `sampleClimate`, `inherit_atmosphere` kapıları.
- Bağlantı sırası (bağımsız ve hızlı görülen önce): MSF ortam T → gaz ortam T +
  lapse rate → foliage rüzgarı → APIC rüzgarı → partikül sürüklemesi → FFT okyanus.
- `CloudManager`'ın sabit 20 m/s'i sökülür.
- **Kabul:** `sample_climate` 0 m ve 1000 m'de lapse rate kadar farklı sıcaklık
  döndürür; gaz domain'i `inherit_atmosphere` açıkken `get` ile aynı ortamı
  raporlar ve **bir kare oynatıldıktan sonra** ortam hücreleri o sıcaklıktadır;
  global rüzgar 0 iken mevcut test sahnelerinin sayısal çıktısı değişmez.

### Faz 3 — weather map + yeni bulut

★ Ayrıntılı tasarım: **`ATMOSPHERE_CLOUDS.md`** (2026-09-30, TASLAK — alt
partiler 3a–3d, karar bekleyenler §9). Aşağıdaki özgün madde listesi referans.

- Weather map (prosedürel üretici + doku), rüzgarla advekt.
- Bulut march'ı (RT tam, RayFusion ¼ + zamansal), gölge haritası.
- Eski iç VDB bulut yolu ve `cloud_*` alanları sökülür.
- **Kabul:** iki katman aynı anda görünür (bugünkü sessiz düşme kapanır —
  `world.clouds.get` iki katman, `stats` iki katmanın march'ını sayar);
  zemindeki bulut gölgesi RT ve RayFusion'da aynı konumda; bulut kaldırılınca
  `world.objects` sayısı değişmez (TLAS'a girmediğinin kanıtı).

### Faz 4 — türetilen hava durumu

- Yağış = weather map `precipitation` > 0 **ve** bulutun altı; tip 0 °C
  izoterminden. Görüş nemden. Yüzey ıslaklığı birikimden → MSF nem kanalı.
- RT'deki weather görselleri RayFusion'a taşınır (aynı türetilmiş değerler).
- `WeatherParams`'ın elle girilen `*_output` alanları sökülür.
- **Kabul:** açık gökyüzü altında yağış sayacı 0; bulut altındaki bir noktada
  ıslaklık zamanla artar ve yağış durunca sıcaklık/neme bağlı hızla azalır.

### Faz 5+ — sonrası

Cloth/rigid aerodinamik sürükleme; sim hacimlerine aerial uygulanması;
fırtına/cephe gibi zamanla değişen weather map üreticileri.

---

## 8. Riskler ve açık sorular

- **Dünya birimi:** `sampleClimate` metre varsayar. Sahne biriminin metre
  olmadığı durumlar (`scene scale`) açıkça dönüştürülmeli — yoksa lapse rate
  ve rüzgar, küçük ölçekli sahnede sessizce yanlış olur.
- **RayFusion zamansal yeniden izdüşüm:** `RasterTaa` hareket vektörleri var;
  bulut geçişinin bunları okuyabildiği Faz 3 başında doğrulanmalı.
- **Çift sayılan rüzgar** (§3.3) göç notunda.
- **`GridFluidSolver` normalize sıcaklık uzayı** (§3.2) — birim dönüşümü tek
  yerde yazılır, çağıranların her birinde değil.
- **RayFusion derinlik hassasiyeti:** D32 standart-Z (near 0,01, far 1e6);
  ~5 km ötesinde görüş derinliği kaba adımlarla gelir → çok uzak zeminde pus
  bantlanabilir. Görülürse reversed-Z ayrı iş (Faz 1b'yi bloklamaz).
- **RayFusion v ekseni:** froxel v alttan yukarı, raster uv yukarıdan aşağı
  varsayıldı (`fuv.y = 1 - uv.y`). Ters çıkarsa sis tabakası ekranın üstüne
  çıkar — NEXT_BUILD_CHECKS Faz 1b madde 5.
- **Keyframe göçü:** `WorldKeyframe`'in `hum/tmp` JSON anahtarları yeni adlarla
  değişir; eski keyframe'ler okunmaz (bilinçli, ilke 6).
