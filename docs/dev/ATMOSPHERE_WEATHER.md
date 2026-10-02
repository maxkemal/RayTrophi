# Hava olayları: türetilmiş hava, yağış, şimşek, kuvvet alanları (Atmosfer Faz 4)

> **Durum:** AKTİF — 2026-10-01. §8 kararlarının hepsi öneri yönünde onaylandı
> ("senin önerilerinle ilerle, eskiye uyumluluk gerekmiyor"). 4a YAZILDI (derlenmedi).

Önceki fazlar: `ATMOSPHERE_SYSTEM.md` (tek AtmosphereState, iklim → fizik tek
yönlü), `ATMOSPHERE_CLOUDS.md` (CloudState + `cloudMarch`, iki modda ortak).

---

## 0. Bugün (kod okuması, 2026-10-01)

| Konu | Durum | Dosya |
|---|---|---|
| İklim | `ClimateState`: yüzey T, lapse rate, RH, basınç, rüzgâr (yön+hız). Kaynak bu; fizik tek yönlü okur (Faz 2) | `include/Atmosphere/AtmosphereClimate.h` |
| Bulut | `CloudState` (3 katman + cirrus + weather map + kalite), `cloudMarch` RT + RayFusion ortak, gölge (RT doğrudan, RayFusion 512² harita) | `ATMOSPHERE_CLOUDS.md` |
| Weather map | 2048², R örtü, G tip, **B boş** ("yağış, Faz 4" diye ayrıldı), A 1 | `shaders/cloud_noise.comp` mode 3 |
| Eski hava | `WeatherParams` (tip, "sanatsal" yoğunluk, görüş, yüzey ıslaklık çıktıları) → **yalnız Vulkan RT raygen/miss**'te ekran üstü yağış overlay'i + gök tonu | `World.h:WeatherParams`, `raygen.rgen applyWeatherPrecipitationOverlay`, `miss.rmiss applyWeatherSky` |
| Kuvvet alanları | `ForceField.h` (vortex dahil) — gaz/akışkan/parçacık tüketir; bulut tüketmez | `include/ForceField.h` |
| Parçacık | GPU parçacık sistemi + görünüm profili LUT'u (id ile adresli) | `ParticleSimulation.h`, commit 7dbd72c |
| Şimşek | **Yok** | — |

Sonuç: yağış bugün **yalnız RT'de, ekran uzayında bir overlay** — geometriyle,
ışıkla, buluttan bağımsız. RayFusion'da hiç yok. Bu faz onu söker (§7).

---

## 1. İlkeler

1. **İklim tek kaynak, her şey türetilir.** Hava bir "mod" (yağmur/kar
   düğmesi) değil; T, RH, kararsızlık, rüzgârdan çıkan bir **durum**. Kullanıcı
   iklimi (ya da bir hava preset'ini — o da iklim değerleridir) anahtarlar;
   bulut tipi, taban, yağış, şimşek sıklığı türetilir. Elle geçersiz kılma
   mümkün ama açıkça "override" olarak (Faz 2'deki `inherit_atmosphere` gibi).
2. **Tek tanım, iki mod.** Faz 3'ün dersi: aynı GLSL fonksiyonu RT'de ve
   RayFusion'da. Yağış perdeleri `cloudMarch`'ın içinde, yakın yağış ortak
   parçacık sistemiyle — iki ayrı "yağmur" yazılmaz.
3. **Fiziksel birim, sanatsal çarpan yok.** Yağış mm/h, damla çapı mm, kar
   yoğunluğu, görüş km. "intensity 0..1" gibi birimsiz alanlar söküm listesinde.
4. **Script/IPC'den erişilebilir** (CLAUDE.md kural 1) — her yeni durum
   `world.*` altında okunur/yazılır, ölçüm uçları sayı döndürür.
5. **Kuvvet alanları buluta tek yönlü etki eder.** Bulut simüle edilmez;
   alanlar bir **sürüklenme alanına** yazar, bulut onu okur. Bulut hiçbir
   çözücüye geri yazmaz (Faz 2 kuralı).

---

## 2. Türetilmiş hava (4a)

### 2.1 Model

Girdi `ClimateState` + iki yeni alan:
- `instability` (0–1): konvektif potansiyel (CAPE'in boyutsuz vekili). Lapse
  rate'ten türetilebilir ama sahne ölçeğinde elle vermek daha anlaşılır —
  karar §8-1.
- `precipitation_mm_h` **türetilir**, girdi değil.

Türetme (`atmosphere::deriveWeather(ClimateState) -> DerivedWeather`, saf fonksiyon):
- **Bulut tabanı** = kaldırma yoğuşma seviyesi (Espy): `LCL ≈ 125 m × (T − Td)`,
  Td Magnus formülüyle RH'dan. Nemli hava → alçak taban.
- **Tip**: kararsızlık düşük → stratiform (0–0.33), orta → kümülüs, yüksek +
  nemli → kümülonimbüs (1.0).
- **Örtü**: RH'ın 0.6 üstü doğrusal yükselişi (≥0.95 → kapalı).
- **Kalınlık**: tip ile (stratus ~400 m, Cb ~8–10 km, tropopoza kadar).
- **Yağış**: yalnız tip ≥ ~0.6 ve örtü yüksekken; `P ∝ örtü × kalınlık × RH`.
  Kar/yağmur ayrımı: donma seviyesi (T ve lapse rate'ten) zeminin üstünde mi.
- **Şimşek sıklığı**: Cb ve kararsızlığa bağlı (dakikada flaş).

### 2.2 Bulutla bağ

`CloudState`'e `derive_from_climate` (varsayılan **açık** yeni projelerde,
eski projelerde kapalı). Açıkken katman 0'ın taban/kalınlık/tip/örtüsü ve
weather map B kanalının ölçeği türetilmişten gelir; panel bu alanları
**salt okunur** gösterir ("iklimden"). Kullanıcı kapatınca değerler kopyalanır
ve düzenlenebilir olur — sessiz anlam değişikliği yok.

Weather map **B kanalı = yağış potansiyeli** (0–1): üretici (`cloud_noise.comp`
mode 3) örtü ve tip alanlarından türetir — yoğun kümelerin çekirdeği. Yerel
yağış = `P_türetilmiş × B`. Böylece yağış **bulutun altında** olur, gökyüzünün
her yerinde değil.

---

## 3. Yağış ve toz (4b)

Üç ölçek, üçü de aynı yerel yağış değerini okur (`precipAt(xz)` =
`P × B(xz)`, `cloud_common.glsl`'e eklenir):

### 3.1 Uzak alan — yağış perdeleri (hacimsel, `cloudMarch` içinde)
- Bulut tabanından zemine, `precipAt(xz) > 0` olan sütunlarda düşük yoğunluklu
  bir hacim: sönüm `σ ∝ P^0.63` (Marshall–Palmer damla dağılımından optik
  sönüm), rüzgârla eğik (düşüş hızı yağmur ~6 m/s, kar ~1 m/s → eğim = rüzgâr /
  düşüş hızı). Virga: taban altında buharlaşma → yükseklikle incelen yoğunluk
  (RH düşükse zemine ulaşmaz).
- Işık: bulutla aynı march; faz yağmurda neredeyse izotrop + hafif ileri,
  karda izotrop. Bedava: bulutun gölge haritası perdeleri de karartır.
- Aralık hesabı: `cloudIntervals`'a bir "yağış kabuğu" (zemin → en alçak
  katman tabanı) eklenir; majorant = `σmax × B` majorant haritasından (§ majorant
  haritası B'nin de maksimumunu tutar — mode 4 genişler).

### 3.2 Yakın alan — parçacıklar (kamera çevresi, iki mod)
- Kamera merkezli, kamerayla kayan bir kutu (~40 m); GPU parçacık sistemi
  (mevcut) üzerinden, yeni bir **yağış yayıcısı**: yoğunluk `precipAt(kamera)`,
  hız = düşüş hızı + iklim rüzgârı, tip (yağmur çizgisi / kar tanesi / toz)
  türetilmişten.
- Görünüm: mevcut görünüm profili LUT'una iki profil (yağmur: hareket
  bulanıklığıyla uzamış, kırılmalı; kar: yumuşak disk, yavaş salınım).
- Çarpışma: zemin yüksekliği (terrain) ile basit ölüm; sıçrama sonraya.
- **Tek sistem, iki mod:** parçacıklar zaten RT ve RayFusion'da çiziliyor —
  ekran overlay'i sökülür.

### 3.3 Yüzey — ıslaklık ve birikim
- Mevcut `surface_wetness/accumulation` "çıktıları" gerçek alan olur: zamanla
  entegre edilen, dünyaya sabitli kaba bir 2D harita (512², kamera çevresi,
  RayFusion bulut gölgesi haritası gibi) — yağış artırır, güneş/rüzgâr
  kurutur (Faz 2 `dryingScale`), kar ısıyla erir.
- Tüketiciler: malzeme ıslaklığı (pürüzlülük ↓, albedo ↓ — mevcut weather
  surface yolu), kar örtüsü (normal yukarı bakan yüzeylerde), su birikintisi
  sonraya.
- **Toz:** aynı model, kaynak rüzgâr hızı eşiği + kuru zemin (RH düşük);
  uzak alanda alçak, yoğun, sarımsı perde (haboob), yakın alanda parçacık.

---

## 4. Şimşek ve yıldırım (4c)

- **Olay üretici:** Cb katmanı + `precipAt` yüksek + şimşek sıklığı → Poisson
  süreci, **timeline zamanına bağlı ve tohumlu** (aynı kare aynı flaş —
  render tekrarlanabilir olmalı). Olay: konum (Cb çekirdeği), tip (bulut içi /
  bulut-yer), başlangıç zamanı, süre (~200 ms, 3–4 tekrar darbesi).
- **Kanal geometrisi:** orta nokta kaydırmalı dallanan yol (tohumlu), bulut
  tabanından zemine (bulut-yer) ya da bulut içinde yatay. Çizim: ince emissive
  şerit (RayFusion: raster çizgi + bloom; RT: emissive kapsül/çizgi primitive).
- **Bulut içi aydınlatma:** `cloudMarch` içinde, aktif flaş varsa her
  örnekte kanalın birkaç noktasından **mesafeye bağlı** ışık (ters kare +
  bulutun kendi optik derinliği tahmini). Bulut içten parlar, flaş yayılır.
- **Sahne ışığı:** bulut-yer flaşında geçici bir ışık (konumlu, çok güçlü,
  kısa) — RT ve RayFusion ışık listesine girer, olay bitince çıkar. Gölgeler
  normal ışık yolundan.
- IPC: `world.lightning_events(t0, t1)` (olay listesi), `world.trigger_lightning`
  (test ve sanatsal tetik).

---

## 5. Kuvvet alanlarıyla etkileşim (4d)

- **Bulut sürüklenme alanı:** weather map kapsamında kaba bir 2D vektör
  alanı (256², RG16F), her karede güncellenir:
  `offset(t+dt) = advect(offset(t), u) + u·dt`, `u` = iklim rüzgârı + sahnedeki
  `ForceField`'lerin yatay bileşeni (vortex → teğet hız). Semi-Lagrangian,
  compute; iki cihazda aynı girdiyle aynı sonuç.
- `cloudWeatherAt` ve temel gürültü koordinatları bu ofsetle okunur. Sonuç:
  vortex bulutları çevresinde döndürür, kasırga spirali yapar; alan kaldırılınca
  sürüklenme kalır (fiziksel — bulut "yerine dönmez").
- **Deterministik zaman:** alan kare kare entegre edildiği için timeline'da geri
  gitmek sorunlu. Çözüm: sabit adımlı entegrasyon + kare başına anahtar-kare
  önbellek (her N karede bir anlık görüntü); scrub en yakın anlık görüntüden
  ileri sarar. Karar §8-4.
- **Hortum hunisi:** ayrı küçük analitik yoğunluk (kesik koni, dönen gürültü),
  bir vortex alanına bağlı; tabanda toz/kalıntı parçacıkları (4b tozuyla).
- Birleşik madde domain'i (`BIRLESIK_MADDE_DOMAIN_TASARIMI.md`) ortam aşaması
  geldiğinde aynı sürüklenme alanını rüzgâr sınırı olarak okuyabilir — ters yön yok.

---

## 6. IPC yüzeyi (kural 1)

| Metot | Ne |
|---|---|
| `world.get_weather` | türetilmiş hava (taban, tip, örtü, yağış mm/h, kar/yağmur, şimşek sıklığı) + override durumu |
| `world.set_weather` | kararsızlık, override'lar, yüzey modeli parametreleri |
| `world.sample_precipitation` | noktalarda yerel yağış (mm/h) ve perde sönümü — GPU'da, `sample_clouds` gibi |
| `world.surface_wetness` | noktalarda ıslaklık / kar derinliği |
| `world.lightning_events`, `world.trigger_lightning` | §4 |
| `world.cloud_flow_stats` | sürüklenme alanı: max hız, ofset, üretim sayacı |

Kabul betikleri **kısa** (kullanıcı kuralı 2026-09-30: az token): faz başına 3–5
sayısal kontrol.

---

## 7. Söküm listesi

- `raygen.rgen applyWeatherPrecipitationOverlay` ve `applyWeatherAtmosphere`
  (ekran uzayı yağış/görüş), `miss.rmiss applyWeatherSky` tonu.
- `WeatherParams`: `intensity`, `density`, `precipitation_scale`, `visibility`,
  `visual_mode` (birimsiz/sanatsal). `type` türetilmiş duruma dönüşür (adı da
  değişir — eski veri yanlış okunmasın, kural 5). Yüzey çıktıları §3.3'ün
  gerçek alanına taşınır.
- RT world struct'ındaki weather bloğu (5 shader kopyası — Faz 3b'deki bulut
  bloğu sökümüyle aynı yöntem).
- Eski projeler: hava tipi → iklim preset'ine bir kez eşlenir (yağmur → RH 0.95
  + kararsızlık 0.6, kar → T < 0 °C, toz → RH 0.1 + rüzgâr 15 m/s).

---

## 8. Kararlar (2026-10-01, hepsi öneri yönünde onaylandı)

**4a uygulama notları:** `atmosphere::deriveWeather` (AtmosphereClimate.cpp),
`World::rederiveClouds` (her iklim/irtifa yazımında `syncClimatePacket`'ten),
`CloudState.derive_from_climate` (preset kapatır), weather map B = yağış
potansiyeli (`smoothstep(0.6,0.95,cov)`), `world.get_weather`, panel.
Anahtar kare: `instability` iklim anahtarına HENÜZ girmedi (mevcut değer korunur).
Eski bulut içe aktarımı silindi (uyumluluk yok).

**4b-1 uygulama notları (perdeler):** `CloudState.precipitation` {rate_mm_h, snow,
ground_fraction} (türetmede iklimden: virga = RH 0.45–0.8), `CloudParamsGPU.precip/precip2`
(240 B; sönüm yağmur 2.5e-4·R^0.63, kar 2.6e-3·R^0.7; eğim = rüzgâr / düşüş hızı 6.5 | 1 m/s),
`cloudPrecipAt/cloudPrecipDensity` (cloud_common), `cloudPrecipMarch` (cloud_rt, `cloudMarch`
sonunda; 40 km, karesel adım). `world.sample_clouds mode=precipitation` = §6'daki
`sample_precipitation` (ayrı metot açılmadı). Sınır: yalnız gök ışınları; arazi önündeki
perde ve yakın parçacık, yüzey ıslaklığı, overlay sökümü → 4b-2.

1. **Kararsızlık girdisi:** ayrı `instability` alanı mı (öneri — sahne
   ölçeğinde anlaşılır), lapse rate'ten mi türetilsin (fiziksel ama kullanıcı
   lapse rate'i düşünmez)?
2. **Türetme varsayılanı:** yeni projede `derive_from_climate` açık (öneri),
   eski projede kapalı.
3. **Yakın yağış:** mevcut GPU parçacık sistemi (öneri — iki modda zaten
   çiziliyor) mı, ayrı hafif bir "yağış çizgisi" shader'ı mı?
4. **Sürüklenme alanında scrub:** anlık görüntü önbelleği (öneri) mi, yoksa
   alanı yalnız ileri oynatmada entegre edip scrub'da sıfırlamak mı (basit ama
   geri sarınca bulutlar zıplar)?
5. **Sıra:** 4a → 4b → 4c → 4d (öneri). 4a diğerlerinin girdisi; 4b en büyük
   görsel kazanç; 4c 4a+4b'nin üstüne oturur; 4d en karmaşık ve birleşik
   domain ile kesişir.

---

## 9. Alt partiler (her biri bir derleme)

| Parti | İçerik | Kabul |
|---|---|---|
| **4a Türetilmiş hava** | `deriveWeather`, `CloudState.derive_from_climate`, weather map B kanalı, panel salt-okunur alanlar, `world.get/set_weather` | "nemli + kararsız" iklim → Cb, taban LCL ile ±%10; RH 0.3 → açık gök |
| **4b Yağış + toz** | `precipAt`, perdeler `cloudMarch`'ta, yakın parçacık yayıcısı, yüzey ıslaklık/kar haritası, eski overlay sökümü | iki modda aynı perde konumu; `sample_precipitation` bulut altında > 0, açıkta 0 |
| **4c Şimşek** | olay üretici, kanal, bulut içi aydınlatma, geçici sahne ışığı | aynı kare → aynı olay listesi; flaş sırasında bulut parlaklığı ↑ |
| **4d Kuvvet alanları** | sürüklenme alanı, vortex → spiral, hortum hunisi, scrub önbelleği | vortex çevresinde bulut ofseti teğet; scrub ileri-geri aynı kare aynı gök |

Faz 3'ten kalan (bu fazdan bağımsız, sıra kullanıcıda): 3d ışık huzmeleri
(froxel'e gölge haritası), RayFusion parametre değişiminde titreme, oktav
kalibrasyonu (referans yol izleyiciye karşı).

---

## Kaynaklar

- Espy LCL yaklaşımı; Magnus çiğ noktası (Alduchov & Eskridge 1996).
- Marshall & Palmer 1948 — damla boyu dağılımı, yağış sönümü.
- Schneider, "Nubis³" (SIGGRAPH 2022) — hava haritası, yağış kanalı.
- Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite" (2016).
- Kim et al., "Physically-based lightning" / orta nokta kaydırmalı kanal yöntemleri.
- Stam, "Stable Fluids" (1999) — semi-Lagrangian taşıma.
