# DEM taneleri için sıcaklık — devir notu

> **Durum:** AKTİF — 2026-10-19: 1, 2, 3 ve 5 kaynakta (`Fluid/MatterGrainThermal.cpp`, DERLENMEDİ); 4 (Ice erimesi) mevcut iskelet yolundan geçiyor, canlı kanıtı yok; 6 = `scripts/test/rt_test_grain_heat_ipc.py`. Ayrıntı: BIRLESIK_MADDE_DOMAIN_TASARIMI "Parti 2026-10-19".

## 2026-10-19 uygulama özeti

- Plandan sapma: sıcaklık için GPU DEM kanalı EKLENMEDİ. DEM sıcaklığa ihtiyaç
  duymuyor. Taneler kanonik host dizisinde taşıyor, termal blok tane adımından
  önce bütün `state.particles` üzerinde koşuyor.
- Tane↔sıvı ve tane↔hava filmleri tanenin iç direnciyle seri
  (h_eff = 1/(1/h + r/(5k))). Tane↔tane hücre içinde G = k·4πr. Hücreler arası
  iletim zaten vardı.
- Tane domain'inde sıvı donmaz, aşırı soğur (`freeze_held_by_grains`).
- IPC: `fluid.list_domains` → `thermal_grain_*`.

(Aşağısı 2026-10-10 taslağı.)

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
