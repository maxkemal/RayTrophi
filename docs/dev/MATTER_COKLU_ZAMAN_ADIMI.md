# Madde domain'inde çoklu zaman adımı (sıvının kendi saati)

> **Durum:** CANLI — 2026-11-11: sıvının kendi saati KALICI (anahtar ve senkron temas yolu söküldü, Adım 3); bağ ayrıntısı ve açık yakınsama sorunu `MATTER_SIVI_MPM_BASINC_BAGLASIMI.md` §12.

## 1. Sorun ve ölçüm

Aynı Matter domain'inde MPM (kar), DEM, sıvı ve gaz birlikte koşar. Bugün hepsi **en kısıtlayıcı
malzemenin** istediği ortak alt adım saatinde ilerliyor (`resolveMatterCommonSubsteps`,
`MatterGpuStep.inl`).

**Ölçülen sahne (kullanıcı sahnesi, 2026-11-09, 3 kare `fluid.step`, `step_stats.mixed_stages`):**

| | |
|---|---|
| Parçacık | 100 000 — Snow 62 578 (MPM Stomakhin), Water 37 422 |
| Snow | E 1.4e5 Pa, ρ 400, ξ 10, sertleşme tavanı 5 |
| Voksel | 2.78 cm |
| Alt adım | **180** (`granular_wave_substeps`; strain-rate yalnız 1–2 istiyor) |
| Kare | **~2.1 s** (`total_ms` 2099–2327), 64 000 dispatch |

| Aşama (host duvar saati, bekleme dahil) | ms/kare | dispatch | senkron | not |
|---|---|---|---|---|
| **pressure** (sıvı MGPCG) | **1374** | 52 236 | 1767 | 180 çözüm; 154 MB upload, 1407 download |
| **liquid_transfer** (sıvı P2G) | **634** | 2 700 | 180 | her alt adım bir senkron |
| mpm_transfer | 14 | 5 040 | 0 | GPU işi yukarıdaki beklemelerin içinde |
| gather, advection, occupancy, flip, forces, contact | ~15 | ~3 800 | 0 | |

Bölüşüm kabadır (host duvar saati GPU beklemesini ilk senkronlayan aşamaya yazar), ama
yön nettir: **karenin ~%95'i sıvı şeridinin 180 kez çözülmesi.** Sıvının kendi CFL'i bu
sahnede kare başına ~1 adım ister (en hızlı parçacık 0.39 m/s, h 2.78 cm).

**180 nereden:** dalga CFL'i dt < 0.35 h / √(E/ρ), E = `snowSizingYoungModulus` = E × 5
(tavan). c = √(7e5/400) = 41.8 m/s → dt 2.3e-4 s → 1/24 s'de 180.

**Neden `grid_stride` kurtarmıyor:** `grid_stride = substeps / ceil(request)` ve `request`
elastik isteği (180) içerdiği için 1'e düşer. Mekanizma "basıncı DEM sıklığında çözme" için
yazılmış; MPM ızgara tabanlı olduğundan her alt adımda ızgara aktarımı ister, sıvı da onunla
sürüklenir.

**Ölçülmüş yan bulgular:**
- Dinlenik kar 51 720 uyuyordu; ilk `fluid.step` sonrası `granular_sleeping` 0. Su karışımı
  karı uyanık tutuyor olabilir — ayrıca bakılmalı (180 adımı artık bütün kar ödüyor).
- Yüklemede raporlanan J_P (min 0.44, max 4.0, ort. 1.22) ilk adımdan sonra 0.89–1.26 /
  ort. 1.0003 oldu: **yükleme sonrası istatistik bayat**, adım atmadan okunmamalı.

## 2. Elenen yol

**"Alt adımı geçen karenin ölçülen sertleşmesine göre boyutla"** — kodda denenmiş ve
reddedilmiş (`GranularConstitutive.h`, `snowSizingYoungModulus` notu): ölçüm bir kare bayat,
çarpma karesi seyrek adımla koştu, 7646 parselin modülü kırpıldı, kar aşırı sıkıştı
(ort. J_P 0.74). Kazancı da küçük olurdu (bugünkü en sert parsel ~1.7–2.9× → 106–136 adım).

## 3. Mimari: iç içe saat

Endüstri karşılığı CFD-DEM alt çevrimidir (EDEM+Fluent, CFDEM/LIGGGHTS): akışkan büyük
adımda, katı fazı içinde kendi kararlılık adımıyla.

```
dış adım  Δt_L  = sıvı CFL'i (bu sahnede 1–2/kare)
  1. sıvı P2G (+ karışık hücrede MPM kütlesi, bkz. §4)
  2. ORTAK basınç projeksiyonu        ← birlikte çözülür
  3. sıvı G2P; sıvı MAC hızı u_L donar
  4. iç adımlar  δt = Δt_L / n_MPM  (n_MPM = elastik + strain + CFL isteği)
       MPM P2G → gerilme → collider → sürükleme(u_L sabit) → G2P → adveksiyon
       sürükleme itmesi I_k biriktirilir
  5. Σ I_k sıvıya EŞİT VE ZIT itme olarak defterden (MatterExchangeLedger)
  6. ısı / faz dönüşümleri: dış adımda bir kez, defterden (aynı madde iki kez çözülmez)
```

DEM aynı kalıba oturur (bugünkü `grid_stride` zaten DEM'i iç saatte tutuyor). Gaz kendi
CFL'iyle ayrı bir dış saat; sıvıyla buharlaşma/yoğuşma alışverişi dış adım sınırında.

## 4. Neler birlikte çözülmeli, neler alt adımlanabilir

**★ Eklenmiş kütle tuzağı.** Kar (400 kg/m³) sudan (997) hafif. Hafif katı yoğun sıvıda
iken iki tarafı ayrı çözüp yalnız sınırdan alışveriş yapan (gevşek, partitioned) bağlaşım
eklenmiş kütle etkisiyle kararsızdır (Causin, Gerbeau & Nobile 2005). Bu yüzden:

| Birlikte (dış adımda, monolitik) | Gerekçe |
|---|---|
| Karışık hücrede sıkıştırılamazlık: MPM kütlesi basınç sistemine girer (karışım/hacim kesri, Gao ve ark. 2018; Tampubolon ve ark. 2017) | eklenmiş kütle; iki şerit birbirini ayrı ayrı itmesin |
| Aynı şeridin kendi teması (MPM–MPM ortak ızgarada) | zaten tek ızgara momentumu |
| Ani büyük ısı akısı (şok sönümleme) | açık alışveriş aşar |

| Alt adımlanabilir (iç saatte) | Koşul |
|---|---|
| MPM elastisite / plastisite | kendi dalga + strain CFL'i |
| DEM–DEM teması | kendi temas adımı |
| Kinematik collider teması (her şerit) | collider hareketi şerit saatinde örneklenir |
| Sürükleme (MPM ← donmuş u_L) | itme biriktirilir, dış adımda eşit-zıt |
| Isı iletimi, faz dönüşümü | dış adımda bir kez; difüzyon kendi CFL'iyle |
| Gaz adveksiyonu | kendi CFL'i |

**Üçlü/dörtlü karışım kuralı:** basınç tek ve ortak; her şeridin iç fiziği kendi saatinde;
şeritler arası her alışveriş dış adım sınırında, defterden, bir kez.

## 5. Bundan bağımsız, hemen yapılabilir (senkron yolu da hızlandırır)

**0. (YAZILDI 2026-11-09, derlenmedi) Basınç çözümü her alt adımda SOĞUK başlıyordu** (p = 0,
~64 iterasyon). Aynı kare içinde önceki basınçtan sıcak başlatma eklendi; ölçüt soğuk b'ye göre.
Kontrol listesi NEXT_BUILD_CHECKS.md.

1. **Basınç statikleri her çözümde yeniden yükleniyor** (~0.85 MB/çözüm, 543 upload/kare).
   Aday: sıvı duvar bandı `LiquidBandRestore` her çözümde `pressure_statics_uploaded = false`
   yapıyor (`FluidGpuPressure.inl`). Doğrulanmadı.
2. **Çözüm başına ~10 senkron** (1767/180): yakınsama kontrolü her seferinde host'a okunuyor.
   Kontrolü seyrelt ya da cihazda koşullu dur.

## 6. Kabul (senkron çözüm referans)

Aynı sahne, aynı başlangıç (kaydedilmiş proje), iki yol yan yana:

| Ölçüt | Eşik |
|---|---|
| Kütle defteri (madde başına) | birebir (yalnız ledger kayıtları kadar fark) |
| Toplam momentum (yerçekimi + duvar itmesi çıkarılmış) | ±%1 |
| Batan/yüzen kar kütlesi, karışım yüksekliği 2 s'de | ±%5 |
| Dış adım yarıya → sonuç farkı | yarıya iner (birinci derece yakınsama) |
| Kararlılık | 10 s'de max hız ve enerji sınırlı, `bad_kinematics` sıfırlaması yok |
| Maliyet | `mixed_stages` aşama tablosu; hedef pressure+liquid_transfer < 50 ms/kare |

★ Sessizce makul görünen başarısızlık: gevşek bağlaşımda kar suyun içinde **hafifçe titrer**
(eklenmiş kütle) ama patlamaz; görsel olarak "su dalgası" sanılır. Momentum defterini ve
kar parsellerinin hız varyansını kare kare oku.

## 7. Sıra

1. §5 iki hızlı iş (ölçüm: pressure ms ve upload bayt).
2. Sıvı dış saati, **önce gevşek bağlaşım + tanı**: eklenmiş kütle titreşimi görünüyor mu.
3. Karışık hücrede ortak projeksiyon (monolitik); kabul tablosu.
4. Uyku: suyla karışan karın neden uyanık kaldığı (bölüm 1 yan bulgu).
5. Bölgesel zaman adımı (Fang ve ark. 2018) yalnız 1–4'ten sonra uyanık sert kısım azınlıksa.

## 8. İlk canlı sonuç (2026-11-09) — ★ §9 bu bölümün yorumunu GERİ ALDI

`rt_test_matter_liquid_multirate_ipc.py` 7/7 (kar bloğu + çöken su sütunu, h 2.5 cm, 200 alt adım):

| | senkron | sıvı kendi saatinde |
|---|---|---|
| kare | 819 ms | **123 ms** |
| basınç | 546 ms | **17 ms** (4 çözüm/kare) |
| kütle (model başına parsel, kg) | korunuyor | korunuyor, birebir |
| şekil p05/p95, kar kütle merkezi | — | ±2 voksel / ±1 voksel içinde |
| sıvı momentumu 1 s'de | **0.09** kg·m/s | **1.73** |

Hangisi fizik? Aynı tankta karsız saf su (tek şerit, kendi CFL saatinde) 1 s'de |p| = **5.5**,
2 s'de hâlâ 7.4: su saniyelerce çalkalanır. Senkron yol suyu 1 s'de durduruyor: kare başına 200
P2G/G2P aktarımı (her birinde PIC payı + afin sönüm) sayısal viskozite olarak birikiyor.
**Yani senkron yol doğruluk referansı DEĞİL**; kabul ölçütü "referansla aynı" değil, şu olmalı:
- saf sıvının kendi sönümüne (aynı tank, karsız) yakın momentum düşüşü;
- `liquid_stride` yarıya inince sonucun değişmemesi (zaman adımı yakınsaması);
- kütle birebir, kar dinlenik şekli ±1 voksel.

★ Bu bulgu senkron karma sahnelerin (kullanıcının kar+su sahnesi dahil) suyunun bugüne kadar
"ağır, çabuk duran" görünmesinin sebebi de olabilir.
## 9. Yakınsama taraması (2026-11-09) — §8'in yorumu yanlıştı

`fluid.step` dt'si küçültülerek sıvı adımı değiştirildi (MPM alt adımı dalga hızından gelir, değişmez).
Sıvı momentumu |p| (kg·m/s), aynı tohum:

| koşu | sıvı dt | 0.25 s | 0.5 s | 1.0 s |
|---|---|---|---|---|
| saf su | 1/24 | 18.4 | 5.1 | 5.0 |
| saf su | 1/96 | 17.7 | 7.1 | 7.4 |
| saf su | 1/384 | 16.9 | 7.3 | 8.3 |
| kar+su, kendi saati | 1/96 | 17.0 | 3.7 | 1.8 |
| kar+su, kendi saati (çağrı 1/96) | 1/96 | 15.6 | 3.8 | 2.6 |
| kar+su, kendi saati | 1/384 | 14.9 | 2.4 | 0.8 |
| kar+su, senkron | 1/4800 | 13.4 | 1.5 | 0.08 |

- **Saf su aktarım sayısıyla sönmüyor** (384 aktarım/s'de de çalkalanıyor): §8'deki "senkron yol aşırı
  sönümlü, sebep aktarım sayısı" yorumu yanlış.
- **Kar+su sıvı dt küçüldükçe tek yönlü olarak senkron sonuca gidiyor.** Yani senkron yol, bugünkü
  temas modelinin yakınsamış limiti; kendi saatinde koşan yol bağlaşımı EKSİK uyguluyor ve sonuç
  sıvı adımına bağlı. Anahtar varsayılan yapılamaz.
- Kar kütle merkezi her koşuda ±3 mm içinde (kar zaten dinlenik).

**Muhtemel kök:** `sim_matter_contact` yalnız yaklaşan (closing > 0) yüzlerde normal itme +
Coulomb sürtünme (μ = tan φ_kar = 0.7, sürtünme ≤ μ·normal itme) uygular. Senkron yolda her MPM alt
adımında sıvı MAC'i yerçekimi alır → kapanma her alt adımda yeniden oluşur → normal ve sürtünme
itmesi sürekli. Kendi saatinde sıvı MAC'i aralık boyunca donmuş: ilk temas kapanmayı sıfırlayınca
sonraki alt adımlarda itme yok.

**Sıradaki deney (derleme istemez, yalnız .inl + belki bir shader):** aralık boyunca sıvı MAC
yüzlerine dış kuvveti (g·dt) her MPM alt adımında uygula; basınç projeksiyonu yine yalnız sıvı
adımında. Kabul: kar+su |p| sıvı dt 1/96 ile 1/384 arasında ±%10 ve senkron sonuca yakın.

**Ayrı fizik sorusu:** su–kar arayüzünde Coulomb sürtünmesi (μ 0.7) fiziksel değil; gerçek
mekanizma sınır tabakası (kaymazlık) + gözenekli kar içinde Darcy sürüklemesi. Senkron sonucun
suyu 1 s'de durdurması bu modelin sonucu olabilir — maliyet işinden ayrı değerlendirilmeli.
## 10. Sürtünme deneyi (2026-11-09) — sürtünme ana sebep DEĞİL

Kar–su teması sürtünmesi domain'in çözülmüş `granular_friction_angle`'ından gelir; kar+su
domain'inde bu Water'ın alanı (35°; Snow 30°). İkisi de 0° yapılmış türetilmiş maddelerle (derleme
yok), sıvı |p| (kg·m/s), h 2.5 cm, dt 1/24:

| koşu | 0.25 s | 0.5 s | 1.0 s |
|---|---|---|---|
| saf su (kar yok) | 18.4 | 5.1 | 5.0 |
| senkron, μ 0.7 | 13.4 | 1.5 | 0.08 |
| senkron, μ 0 | 14.2 | 2.1 | **0.41** |
| kendi saati, μ 0.7 | 17.0 | 3.7 | 1.8–4.9 * |
| kendi saati, μ 0 | 17.3 | 3.8 | 3.2 |

\* aynı kurulum iki koşuda 1.75 ve 4.87: tek anlık |p| çalkalanma fazına ve GPU atomik sırasına
duyarlı; senkron yol ise tekrarlanabilir (0.083 / 0.080).

- Sürtünme sıfırken de senkron yol suyu 1 s'de neredeyse durduruyor: **ana sönüm sürtünme değil.**
- Kalan aday: temas çekirdeği yaklaşan her yüzde **tam esnek olmayan** normal itme uygular ve bunu
  her alt adımda (saniyede 4800 kez) yapar. Gerçek suda duvara doğru hız, sıkıştırılamazlık
  basıncıyla teğete YÖNLENDİRİLİR, yok edilmez. Kar ile suyun üst üste bindiği/değdiği her hücrede
  bu itme, alt adım sayısıyla büyüyen bir sürükleme gibi davranıyor — sayısal, fiziksel değil.
- Bu, §4'teki tasarım kararıyla aynı yere çıkıyor: sıvı ile kar arasındaki normal bağlaşım tekrar
  eden temas itmeleri yerine **sıvının basınç projeksiyonunda** çözülmeli (kar = hareketli, gözenekli
  katı: hacim kesri + katı hızı, tepki kuvveti karın ızgarasına). Bunun altyapısı collider için zaten
  var (varyasyonel katı ağırlıkları + katı yüz hızları, `sim_matter_liquid_wall`). Hem fiziği hem
  sıvının kendi saatini aynı anda düzeltir.