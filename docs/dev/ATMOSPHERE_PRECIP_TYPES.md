# Yağış tipleri, yerleşimi ve görsel tanımları (Atmosfer Faz 8)

> **Durum:** TASLAK — 2026-10-12. Yol haritası: `ATMOSPHERE_STORMS.md` §8.

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
