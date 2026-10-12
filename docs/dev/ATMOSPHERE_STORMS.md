# Fırtına sistemleri: hücreler, yağış tipleri, şimşek yerleşimi, basınç alanları (Atmosfer Faz 5)

> **Durum:** AKTİF — 2026-10-12. Kararlar kullanıcıyla verildi ("önerdiğin gibi
> yap"). 5a CANLI (parti 10: elle hücre doğru çalışıyor). Parti 11 YAZILDI,
> derlenmedi: hücre içi yapı §3.1, basınç → türetilmiş hava §5 (ilk adım),
> viewport gizmosu §7, yukarı yönlü yıldırım. 5b TASLAK.

Önceki fazlar: `ATMOSPHERE_SYSTEM.md` (tek AtmosphereState, iklim → fizik tek
yönlü), `ATMOSPHERE_CLOUDS.md` (CloudState + `cloudMarch`),
`ATMOSPHERE_WEATHER.md` (türetilmiş hava, yağış perdeleri, şimşek).

---

## 0. Neden (canlı gözlem, 2026-10-12)

- **Yağmur bulutun her yerinde yağıyor.** "Burada yağar mı" bilgisi
  hava haritasının B kanalından geliyor: `smoothstep(0.6, 0.95, örtü)`
  (`cloud_noise.comp` mode 3). Yoğun örtünün olduğu her yer yağar. Fırtına
  çekirdeği, hücre yaşı ya da yağış tipi diye bir kavram yok.
- **Şimşek bulut alanının dışına kaçıyor.** Olaylar `lightning.center_m ±
  extent_m/2` karesinde düzgün dağılımla seçiliyor. Açık gökte kalanları
  yalnızca GPU'daki örtü kapısı (`cloudLightningActivity`) eliyor.
  RayFusion zemin flaşında bu kapı hiç yok (bulut alanı bağlı değil), yani
  orada flaş açık gökten de zemini aydınlatabiliyor.
- Gerçek fırtınalar **organize**: hücreler doğar, olgunlaşır, dağılır.
  Sağanak olgun çekirdekte, hafif ve geniş yağış dağılan hücrede ya da
  cephede, şimşek yalnızca olgun Cb'de olur. Hücreler rüzgârla yürür.
  Soğuk havuzun hamle cephesi yeni hücreleri tetikler.

## 1. Ölçek gerçeği

Sinoptik basınç sistemleri (alçak/yüksek) yüzlerce–binlerce km. Sahne
40–100 km. Bu yüzden iki katman:

| Katman | Ölçek | Sahnede ne | Nerede yaşar |
|---|---|---|---|
| **Sinoptik arka plan** (5c) | sahne dışı | gradyan rüzgârı (yön/hız, alçak etrafında dönüş), nem/örtü eğimi, cephe hattı | iklim durumu (`ClimateState`), tek yönlü |
| **Mezoskala hücreler** (5a, 5b) | 2–10 km | konvektif hücreler (yaşam döngüsü), soğuk havuz mezo-yükseği, hamle cephesi | `CloudState.storms` |

## 2. İlkeler

1. **Durum CPU'da, alan GPU'da.** Hücre listesi az sayıda (≤ 24) ve
   deterministik. CPU tutar, IPC okur/yazar (kural 1). Alan GPU'da her kare
   256² bir hücre haritasına sentezlenir (`cloud_cells.comp`). Bulut örneği
   ona **tek doku okumasıyla** bakar; hücre döngüsü sıcak yolda YOK.
2. **Zamanın saf fonksiyonu.** Hücrenin konumu, yarıçapı ve evresi
   `(seed, t)`'den hesaplanır; entegrasyon yok. Timeline'da ileri-geri aynı
   gök (ATMOSPHERE_WEATHER §8-4 sorununu bu katmanda hiç doğurmaz).
   Hücre rüzgârla yürür: `pos(t) = doğum + rüzgâr × (t − t_doğum)`. Bu,
   hava haritasının sürüklenmesiyle aynı hız; hücre ile altındaki bulut
   dokusu birlikte kayar.
3. **Tek tanım.** Yağış, şimşek yerleşimi ve taban flaş ışığı aynı hücre
   listesinden çıkar. RT ve RayFusion aynı haritayı okur. CPU şimşeği zaten
   olgun hücre çekirdeğine koyduğu için RayFusion'daki kapı açığı kapanır.
4. **Majorant kırılmaz.** Hücre örtüyü YÜKSELTEBİLİR. Delta tracking ve
   boş-uzay atlaması `yoğunluk ≤ örtü × sönüm` sınırına güvenir. Hücre
   haritası bu yüzden kendi majorantını taşır: örtü profili mesafeyle monoton
   azaldığı için, `δ` kadar genişletilmiş maksimum = profilin `d − δ`'deki
   değeri. Analitik, ek geçiş gerektirmez. `δ` = bir majorant segmenti +
   iki hücre texeli.
5. **İklim → fizik tek yönlü.** 5b'nin çıkış rüzgârı sürüklenme ve kuvvet
   alanlarına yazar, hiçbir şey hücrelere geri yazmaz.

## 3. Hücre modeli (5a)

`CloudState.storms`:

| Alan | Anlam |
|---|---|
| `enabled` | hücre sistemi açık. Kapalıyken yağış eski B kanalı (yaygın örtü yağışı) olur, şimşek YOK |
| `derive_from_climate` | aşağıdaki dört alan iklimden türetilir (panelde salt okunur) |
| `center_m`, `extent_m` | fırtına alanı (dünya xz). Eski `lightning.center_m/extent_m` buraya taşındı |
| `max_cells` | aynı anda yaşayan otomatik hücre (0–16) |
| `cell_radius_m` | olgun hücre yarıçapı (ortalama) |
| `lifetime_s` | hücre ömrü (ortalama) |
| `intensity` | olgunluktaki tepe şiddeti (0–1) |
| `stratiform_rain` | hücre dışındaki yoğun örtü altında geniş, hafif yağış (tepe hızın oranı) |
| `cells[]` | elle konan hücreler (≤ 8): konum, yarıçap, şiddet, doğum, ömür |

**Otomatik hücreler:** `max_cells` şerit, şimşek şeritleri gibi kaydırmalı.
Şerit k'nin hücre indeksi `floor((t − ofset_k) / periyot)`, oradan seed →
doğum konumu (alan içinde), yarıçap, şiddet.

**Evreler (ömür oranı `s`):**

| Evre | s | Örtü | Kule (tip) | Yağış | Şimşek |
|---|---|---|---|---|---|
| büyüme | 0–0.25 | 0 → 1 | yükselir | yok | yok |
| olgun | 0.25–0.7 | 1 | tam Cb | konvektif, çekirdekte tepe | evet, şiddetle orantılı |
| dağılma | 0.7–1 | yavaş söner | iner (örs kalır) | hafif, geniş (stratiform) | yok |

Yağış çekirdeği hücre merkezinden rüzgâr yönünde `0.3 r` öndedir (ileri
kanat); dağılmada çekirdek genişler ve zayıflar.

**GPU:** `CloudCellsGPU` (ayrı SSBO) → `cloud_cells.comp` → hücre haritası
RGBA16F 256², alan merkezli:

| Kanal | İçerik |
|---|---|
| R | örtü tabanı (katman örtüsü yerel olarak en az bu) |
| G | kule (yerel tip en az bu) |
| B | konvektif yağış potansiyeli |
| A | R'nin `δ` genişletilmiş majorantı |

Yoğunlukta: `cov = max(remap(w.r), R · mix(0.7, 1, w.r))`,
`type = max(type, G)`, gelişim `= max(dev, G)`. Yağışta:
`B_etkin = max(B_hücre, stratiform_rain · w.b)`. Majorantta:
`max(eski, A)`.

**Şimşek:** olay şeritleri artık bir kareden değil, **olgun hücrelerden**
seçilir. Olasılık şiddetle orantılı, konum çekirdek içinde, olay hücreyle
birlikte yürür. Olgun hücre yoksa şimşek yok.

### 3.1 Hücre içi yapı (parti 11)

Canlı gözlem: elle konan hücre "tek düze" görünüyordu. Hücre örtüyü 1'e,
tipi Cb'ye çekiyordu; katman kalınlığı da sabit olduğu için hücre, katmanı
dolduran düz bir blok oluyordu (`topFrac` = 1). Hücre haritasına ikinci bir
alan eklendi: 256×512 tek görüntü, alt yarı (ayrı binding yok).

| Kanal | İçerik |
|---|---|
| RG | girdap: SHAPE okumasının yatay kayması (m) |
| B | yerel tepe, katmanın oranı olarak |
| A | updraft, yani kule çekirdeği ağırlığı |

- **Kuleler:** Hücre başına 5 updraft. Her biri kendi döngüsünde (~15 dk)
  doğar, kendi yüksekliğine (katmanın %62–100'ü) yükselir, sonra çöker.
  Yeniler giriş (rüzgâra karşı) tarafına yakın doğar. Kuleler arasındaki
  gövde katmanın yarısında kalır; örtü orada %70'e iner (≤ zarf, majorant
  kırılmaz). `(id, age)`'in saf fonksiyonu.
- **Girdap (mezosiklon):** Şekil okuması hücre merkezi etrafında, çekirdekte
  tepe yapan ve dışa sönen bir açıyla döndürülür (`rotation × tower × 1.4`
  rad). Gürültü spiral bantlara sarılır. Kule yükselince artar, dağılınca
  gevşer. Yalnız SHAPE etkilenir; örtü ve hava haritası etkilenmez.
- **Updraft:** Kule çekirdeğinde erozyon yarıya iner (sert karnabahar, tül
  yok), dikey ölçek ×0.75 (daha uzun kabarmalar), taç detayı açılır.
- **Parametreler:** `storms.turrets` (0 = eski düz kolon, 1 = tam kule
  yapısı) ve `storms.rotation` (−1..1, + = yukarıdan saat yönü tersine).
- **Sınır:** Kule katman kalınlığını aşamaz (kabuk sınırı). Gerçek bir Cb
  için kalın katman gerekir; iklimden türetilen kararsız hava bunu zaten
  veriyor.

## 4. Soğuk havuz ve çıkış rüzgârı (5b, TASLAK)

> 2026-10-12: Genişletildi ve `ATMOSPHERE_LOCAL_WINDS_FOG.md`'ye taşındı.
> Yeni kapsam: giriş, updraft, downdraft ve hamle cephesi; tüm fiziğe taban
> rüzgâr alanı; hava uygunluğu, son yağış ve yerel sis. Aşağısı ilk taslak.

Olgun ve dağılan hücrenin altında yağışla soğuyan hava → mezo-yüksek
(+1–3 hPa), dışa doğru akış (10–20 m/s) ve bir hamle cephesi (yay). Yeni
otomatik hücreler bu yayın üzerinde doğar (çok hücreli/hat organizasyonu).
Rüzgâr alanı ATMOSPHERE_WEATHER §5'teki sürüklenme alanına ve
`ForceField`'lere tek yönlü yazar. Ölçüm: `world.sample_wind`.

## 5. Sinoptik arka plan (5c, TASLAK — basınç ilk adımı parti 11'de yazıldı)

**İlk adım (parti 11):** Climate panelindeki basınç kaydırıcısı yalnızca hava
yoğunluğuna gidiyordu, türetilmiş havaya hiç girmiyordu. Kullanıcı "basınç
etkisiz" diye gözlemledi. Deniz seviyesi basınç sapması artık sinoptik
sistemin yerine geçiyor: `synoptic_lift` = +1 (~983 hPa, alçak) .. −1
(~1038 hPa, yüksek).

| Etki | Alçak basınç | Yüksek basınç |
|---|---|---|
| Kararsızlık | +0.3 | −0.6 |
| Örtü | +0.3 | ×0.3 |
| Tepe | — | inversiyonla bulut tabanının ~1.5 km üstünde kesilir |
| Hücre sayısı | ×1.5 | × (1 − yüksek) |
| Yağış | stratiform yağışa +0.4 | ×0.2 |

Bu, `derive_from_climate` üzerinden etki eder. Elle kurulmuş bir katman
basınçtan etkilenmez; fırtınalar `storms.derive_from_climate` açıksa yine
etkilenir.

İklime eklenir: alçak/yüksek merkezinin sahneye göre yönü ve uzaklığı.
Türetilen: gradyan rüzgârı (alçak etrafında siklonik dönüş, yarıküreye göre
yön), örtü/nem eğimi, cephe hattı (soğuk cephe → hücre dizisi, sıcak cephe →
geniş stratiform). 5a'nın otomatik doğum dağılımı cepheye yaslanır.

## 6. Ek bulgu: kare başı yükleme beklemesi (parti 10'da düzeltildi)

`CloudParamsGPU` GPU-only bir tampondaydı. Her `setCloudState` ve
`setWeatherTime` (oynatmada her kare, iki backend) çağrısı: staging tamponu
AYIR → tek seferlik komut → **fence ile CPU BEKLER** → staging'i sil.
Bütün kareler aynı compute kuyruğunda olduğu için bu bekleme önceki karenin
GPU işinin bitmesini de bekliyordu; CPU/GPU örtüşmesi sıfırlanıyordu.
Yerine: kalıcı staging halkası + önceden ayrılmış komut tamponu, kuyruğa
sıralı gönderim, CPU beklemez (yalnız halka dolarsa). Hücre haritası
compute'u aynı gönderimde koşar, kare kayıt noktalarına dokunulmaz.

## 7. IPC yüzeyi (kural 1) ve viewport gizmosu

- `world.get_clouds` / `world.set_clouds {storms:{...}}`: yapılandırma ve
  elle hücreler.
- `world.cloud_stats` → `storms.cells[]`: o anki canlı hücreler (konum,
  yarıçap, evre, örtü/kule/yağış/şimşek potansiyeli). Hücre haritasının
  girdisi tam olarak budur.
- `world.sample_clouds mode=precipitation`: hücre altında > 0, hücre dışında
  (stratiform 0 iken) 0. Kabul testinin ölçüsü.
- `world.cloud_stats.lightning[].kind`: `ground`, `cloud` veya `upward`.

**Gizmo (parti 11, `StormGizmoUI`):**
- Her canlı hücre, yerden bulut tabanına uzanan bir kolon olarak çizilir.
  Renk evreyi gösterir: büyüyen mavi, olgun turuncu, dağılan gri, kalıcı
  sarı. Fırtına alanı yerde bir kare olarak görünür.
- Elle konan hücreler ve alan merkezi tutamakla seçilir, XZ düzleminde
  taşınır. Yazma `rtapi::setCloudsJson` üzerinden yapılır; panel ve IPC ile
  aynı yol, UI kendi durumunu tutmaz.
- Sürüklenen (ömürlü) bir hücrenin tutamağı onun ŞİMDİKİ konumundadır.
  Bırakınca doğum konumu, rüzgâr kayması geri çıkarılarak yazılır.
- "Show cells in viewport" yalnızca bir görüntü tercihidir.
- **Otomatik hücreler tek tek düzenlenmez** (şeritlerin saf fonksiyonu,
  saklanan hali yok). Kullanıcı bunları seçmeye çalıştı ve takıldı.
  Çözüm dondurmadır: `freezeStormCells` o an canlı otomatik hücreleri aynı
  doğum, konum, yarıçap, tepe değeri ve ömürle elle konmuş hücreye çevirir.
  Gök değişmez; otomatik şeritler durur. Bunu üç yol tetikler:
  - viewport'ta otomatik bir hücreye tıklamak (tıklanan seçili kalır)
  - paneldeki "Convert live automatic cells to authored" düğmesi
  - IPC: `world.set_clouds {storms:{freeze_automatic:true}}`

## 8. Yol haritası (2026-10-12)

| Sıra | Faz | Not | Kapsam |
|---|---|---|---|
| ✔ | 5a | bu not | hücreler, kuleler, girdap, gizmo, dondurma, basıncın ilk adımı |
| 1 (parti 13 yazıldı) | 6 | `ATMOSPHERE_CELL_CLOUDS.md` | hücre tam sütun "fırtına kabuğu" (LCL → tropopoz, örs, duvar bulutu, mamatus) + yoğunluk ve faza bağlı optik |
| 2 | 5b | `ATMOSPHERE_LOCAL_WINDS_FOG.md` §1 | giriş, updraft, downdraft ve hamle cephesi rüzgârı → bulut şekli, yağış eğimi, `sampleWind` |
| 3 (parti 14 yazıldı; rüzgârlardan önce alındı) | 8 | `ATMOSPHERE_PRECIP_TYPES.md` | tipler (çisentiden doluya), kar çizgisi, görsel tablo |
| 4 | 7 | `ATMOSPHERE_LOCAL_WINDS_FOG.md` §2–4 | hava uygunluğu (orografi, ısınma), son yağış, yerel sis |
| 5 | 5c | bu not §5 | sinoptik merkez ve cepheler, gün içi döngü, senaryolar |
| 6 | ayrı | `ATMOSPHERE_SURFACE_COUPLING.md` | yer yağış haritası → ıslaklık, kar, sıçrama, akış, dolu |

1 ve 2 birbirini besliyor: raf bulutu, rüzgârın hamle cephesinden geliyor.
İkisi ardışık iki parti olarak yapılmalı.

## 8. Alt partiler

| Parti | İçerik | Kabul |
|---|---|---|
| **5a** (parti 10) | hücre modeli, hücre haritası, yağış/örtü/kule entegrasyonu, majorant, şimşek hücreden, yükleme beklemesinin kaldırılması | şimşek olaylarının %100'ü olgun hücre çekirdeğinde (`cloud_stats`); yağış hücre dışında 0 (stratiform 0); aynı `t` → aynı hücre listesi |
| **5b** | soğuk havuz, çıkış rüzgârı, hamle cephesinde doğum, rüzgâr alanı | olgun hücre çevresinde dışa dönük rüzgâr; yeni doğumların yay üstünde yığılması |
| **5c** | sinoptik arka plan, cephe | alçak tarafına doğru örtü artışı; cephe hattına dizili hücreler |
