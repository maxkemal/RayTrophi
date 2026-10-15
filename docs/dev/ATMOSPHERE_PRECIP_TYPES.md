# Yağış tipleri, yerleşimi ve görsel tanımları (Atmosfer Faz 8)

> **Durum:** AKTİF — 2026-10-12. Parti 14 YAZILDI, derlenmedi: uzak perde
> tipleri (§6). Yakın parçacıklar ve arazi okuması (kar çizgisi zeminde)
> sonra. Yol haritası: `ATMOSPHERE_STORMS.md` §8.

## 0. Bugün

- Tip ikili: `snow = Tc ≤ 1 °C` (yer sıcaklığı), sahnenin her yerinde aynı.
- Yerleşim: hücre çekirdeği (konvektif) + dağılan hücre + stratiform pay.
  Doğru, ama tipsiz.
- Görsel: yağmur ya da kar için tek bir çizgi ve sönüm modeli
  (`cloudPrecipSample`, near-field parçacıklar).

## 1. Tip nereden gelir: sütunun sıcaklık profili

Tip, yağışın DÜŞTÜĞÜ sütunda belirlenir. Girdiler: iklim (T0, lapse rate,
inversiyon), donma seviyesi, yerel arazi yüksekliği ve hücre gücü.

| Tip | Koşul |
|---|---|
| Çisenti | stratus / stratokümülüs, ince ve doygun; konvektif değil |
| Yağmur | donma seviyesi yerin > ~300 m üstünde |
| Sulu sepken | donma seviyesi yerin 0–300 m üstünde (erime tamamlanmıyor) |
| Kar | sütun baştan sona soğuk. Tane boyu sıcaklıkla değişir: 0..−5 °C iri yapışkan küme, < −10 °C ince toz |
| Donan yağmur / buz peleti | üstte sıcak katman, yerde soğuk (negatif lapse veya inversiyon); sıcak katman kalınlığı ikisini ayırır |
| Graupel | olgun hücre, orta güç, soğuk |
| Dolu | çok güçlü kule (tower × updraft × şiddet eşiği), çekirdeğin updraft'a bitişik kenarında dar sütun; tane 0.5–5 cm |
| Virga | var (`ground_fraction`), korunur |

**Arazi:** Donma seviyesi bir yükseklik olduğu için aynı yağış vadide
yağmur, zirvede kar olur. Kar çizgisi kendiliğinden çıkar. Bunun için
arazi yüksekliği okunur (`ATMOSPHERE_LOCAL_WINDS_FOG.md` karar 2 ile aynı
kopya).

## 2. Görsel tanımlar (tip başına parametre tablosu)

| Tip | Tane / damla | Düşüş hızı | Uzak perde | Yakın parçacık |
|---|---|---|---|---|
| Çisenti | 0.2–0.5 mm | 1–2 m/s | ince, homojen pus; görüş düşer, çizgi yok | çok yoğun, çok kısa, saydam |
| Yağmur | Marshall–Palmer dağılımı (hızdan) | 4–9 m/s | çizgili perde; şiddetle koyulaşır; yağmur ayağı | uzun çizgi; yere sıçrama (yüzey planı) |
| Sağanak | iri damla ağırlıklı | 7–9 m/s | keskin kenarlı, koyu sütun | kalın çizgi |
| Sulu sepken | karışık | 2–5 m/s | yağmur ve kar karışımı | iki tür karışık |
| Kar | küme 2–20 mm | 0.5–1.5 m/s | yumuşak, beyaz, rüzgârla eğik | salınım, dönme, yavaş |
| Graupel | 2–5 mm yuvarlak | 2–4 m/s | beyazımsı | sert ve sekmeli |
| Dolu | 5–50 mm | 10–40 m/s | dar, açık renkli sütun (buz saçar) | parlak, sekmeli, sesli |

- Uzak perde: `cloudPrecipSample`'ın sönümü ve faz fonksiyonu tip başına
  değişir. Kar daha izotrop saçar; yağmurun ileri saçılımı güçlü (gün ışığı
  arkadan gelince parlak perde). Gökkuşağı ayrı bir iş, sonra.
- Erime bandı (bright band): donma seviyesinin hemen altında, perdede ince
  parlak bir şerit.
- Yakın parçacıklar: aynı tip tablosundan beslenir; iki temsil tek kaynak.

## 3. GPU ve veri

- Hücre haritasının B kanalı (yağış potansiyeli) kalır. Tip konumdan ve
  yükseklikten hesaplanabilir: sıcaklık profili sabit uniform'lardan, arazi
  yüksekliği 2D haritadan geliyor. Yalnız dolu ve graupel hücreye özgü; onlar
  için hücre haritasında bir kanal gerekir (yapı yarısındaki A = updraft
  zaten tam bunu taşıyor).
- `CloudParams` içine tip tablosu girer: ~8 tip × 4 vec4.

## 4. IPC (kural 1)

- `world.sample_clouds {mode:"precipitation"}` → `rate_mm_h`, `type`,
  `particle_size_mm`, `fall_speed_mps`
- `world.cloud_stats.precip_types`: sahnedeki tiplerin dağılımı (kabul)

## 5. Kabul

- `lapse_rate` < 0 ve yer < 0 °C iken `type = freezing_rain`.
- Yer 0 m'de 5 °C, zirve 2500 m: vadide `rain`, zirvede `snow`.
- Dolu yalnız eşiği aşan hücrede ve çekirdeğin 0.3 r içinde.

## 6. Parti 14 uygulaması (uzak perde)

**Profil.** `precip3` = deniz seviyesi °C, lapse K/m ve zorlanmış tip.
`World::rederiveClouds` bunları HER senkronda iklimden yazar; katman 0
türetilmese de yazılır. `CloudPrecipitation.air_temperature_c`,
`lapse_rate_k_per_m` ve `snow` salt okunur türetilmiş alanlardır. `snow`
artık "sahne zemininde donmuş mu" demek; ses köprüsü bunu okur, anlamı
eskisiyle aynı.

**Sınıflandırma** (`cloudPrecipFormAt`). Her nokta, sütunu boyunca
izlenir:
- **Erime:** bulut tabanından noktaya kadar sıfırın üstünde geçen
  derece·metre. 30–400 °C·m arasında kar → sulu sepken → yağmur.
- **Sıcak bulut:** taban 0 °C'nin üstünde ve updraft yok. Baştan sıvı
  yağar.
- **Sıfır altında sıvı:** donan yağmur. Erime katmanının altındaki soğuk
  katmanda 1500–3000 °C·m'yi geçince buz peleti.
- **Updraft** (`flow.w`): 0.2–0.45 graupel, 0.45–0.8 dolu. Graupel
  donma seviyesinin ~1 km altında erir; dolu uzun düşüşte bile yaşar.
- **Çisenti:** konvektif çekirdekten gelmeyen pay × sığ tür
  (katman tipi < 0.25–0.5).

**Optik** (tip başına sabit tablo). Karışımın ağırlıklı toplamı
kullanılır:
- sönüm katsayısı ve hız üssü
- düşüş hızı: eğim buna göre ölçeklenir (`precip.zw` artık 1 m/s'lik bir
  düşen için verilir, shader yerel hıza böler)
- ileri lob payı (faz)
- çizgi kontrastı: çisenti homojen pus, dolu keskin çizgi

**IPC:** `world.sample_clouds {mode:"precipitation_form"}` → `values`
(kimlikler) ve `forms` (adlar).

**Panel:** "Snow" kutusu yerine "Form" seçicisi (Auto + 8 tip). Altında
türetilmiş hava sıcaklığı ve donma seviyesi gösterilir.

**Canlı ölçüm (2026-10-12, parti 14 derlendi):** Olgun çekirdek altında
15 °C'de `rain` ✓. Ama −5 °C'de ve inversiyonda (yer −3 °C, lapse −0.004)
baskın tip **graupel** çıktı; beklenen kar ve donan yağmurdu. Sebep:
updraft ağırlıkları dışlayıcıydı, çekirdeğin bütün kütlesini graupel ve
dolu alıyordu. Düzeltme (parti 15): dolu ≤ %60 (updraft 0.55–0.85),
graupel ≤ %40 (0.3–0.6). Kütle payı olarak.

**Henüz yok:**
- ~~yakın alan parçacıkları (WeatherParams Rain/Snow, ayrı sistem)~~ →
  2026-10-14 `ATMOSPHERE_NEAR_PRECIP.md` N1: kamera önü damlalar bu tip
  karışımından ve bu tablolardan (düşüş hızı, sönüm, ileri lob) besleniyor;
  eski WeatherParams overlay'i söküldü
- zemindeki arazi yüksekliği: kar çizgisi şimdilik perdenin kendi
  yüksekliğinden çıkıyor, arazi okunmuyor
- erime bandının parlaklığı
