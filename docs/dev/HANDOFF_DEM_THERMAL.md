# DEM taneleri için sıcaklık — devir notu

> **Durum:** TASLAK — 2026-10-10, birleşik madde ısı katmanı (BIRLESIK_MADDE_DOMAIN_TASARIMI §4.7) bitti; DEM taneleri bu katmanın DIŞINDA.

## Neden

Madde domain'inde sıvı, MPM granular (kar), gaz ve collider kabuğu (MSF) artık
maddenin kendi `c, k, ρ, L, M, Tb` değerleriyle ısı alışverişi yapıyor. DEM taneleri
(`MatterGrainGpu`, `sim_matter_grain.glsl`) bunun hiçbirine katılmıyor: ıslak kum
ısınmaz, sıcak taş suyu kaynatmaz, buz taneleri erimez.

## Hazır olan (yeniden kullan, kopyalama)

- `Fluid::parcelHeatCapacity(particles, p, profile, h, ppc)` — parçacık ısı sığası.
- `Fluid::conductThermalLiquid` — hücre ısı sığası + harmonik k, kesin iki cisim çözümü.
  Taneler aynı hücre ızgarasına girerse iletim kendiliğinden çalışır; ŞART: tane
  taşıyıcıları `particles.temperature` dizisinde gerçek sıcaklık taşımalı ve host
  kopyası adım sonunda güncel olmalı.
- `exchangeLiquidGasSurface` / `exchangeLiquidSolidContact` — sütun tepesi ve temas
  hücresi mantığı; taneler için aynı film modeli.
- Termal alan: `collectThermalSources` + `thermalSourceWeight` (CPU), aynı profil GPU'da.
- Doğum: `substanceAmbientBirthKelvin` (taneler de kaynaktan doğar).

## Yapılacak (önerilen sıra)

1. Tane sıcaklığı durumu: DEM GPU yerleşik taşıyıcılarda sıcaklık kanalı (yoksa ekle),
   host'a indirme (uyku/yerleşiklik ile uyumlu — S1b/uyku revizyonlarını bozma).
2. Tane ↔ sıvı: tane hücresindeki sıvıyla iki cisim alışverişi (film katsayısı katı–sıvı
   500 W/m²K, `exchangeLiquidSolidContact` ile aynı form).
3. Tane ↔ tane iletim: temas listesi zaten var; temas başına `G = k_harm · a_temas`
   (a_temas ≈ π·r_temas²; Hertz temas yarıçapından).
4. Tane erimesi: `meltable` taneler (Ice) Tm üstünde sıvı parçacığa dönüşür — kütle ve
   enerji defterle (`Melting`).
5. IPC: `fluid.matter_models` grain bölümüne `min/mean/max_kelvin`, `heat_to_liquid_j`.
6. Kabul: sıcak taş (Stone, 600 K) ile su kabı → su ısınır, taş soğur, enerji farkı ≈ 0.

## Sınırlar / dikkat

- `sim_matter_grain.glsl` ve `MatterGrainGpu.cpp` diğer ajanın alanıydı (uyku
  revizyonları 22–24, S1b) — onlarla çakışma.
- Sıcaklık bayat okunursa (yerleşik GPU durumu) iletim "çalışıyor görünür" ama enerji
  kaybolur: kabul testinde enerji farkını ölç.
