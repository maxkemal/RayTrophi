# Islak kum doğrulaması (DEM sıvı köprüleri)

> **Durum:** AKTİF — 2026-10-23; ilk ölçüm yapıldı, seed ıslaklığı yazıldı (derlenmedi). Sıradaki: kolon + kaya testi.

Hedef (kullanıcı): hafif ıslak, sıkıştırılmış kum bloğu ayakta durmalı ve
kuvvetle topak topak dağılmalı. Kar çalışmasının (KAR_STOMAKHIN_MPM.md) aynı
yöntemi: IPC ile kur, sayıyla ölç, gerekirse görsele bak.

## Mevcut model
- Sand ön ayarı DEM'dir (`granular_transport dem`). Islak tane = tanede su
  (`grain_water_capacity_fraction` > 0) ve domain'de sıvı ya da ıslak doğum.
- Köprü: Willett (2000) pendüler köprü, F = 2πRγcosθ / (1 + 2.1s + 10s²),
  kopma S = V^(1/3). Kaba taneleme: temsilî R tane, gerçek r'nin (R/r)² köprüsü
  → F × R/r; yığın yapışma gerilmesi korunur (2026-10-21 düzeltmesi).
- MPM yolu (gözenek suyu, `fluid.set_pore_exchange`) ayrı: kılcal kohezyon
  4S(1−S)·tepe — tepe S=0.5'te, gerçekte düşük doygunlukta; ayrıca eski kardaki
  DP zayıflıkları (mükemmel plastisite, iyileşme). Bu not DEM yolunu izler.

## 2026-10-23 ölçüm (exe 16:10)
- Sand'in su kapasitesi 0 → ıslanamaz. Test için `WetSandProbe` (kapasite 0.1).
- Islak tane yalnız dökülerek doğuyordu; `fluid.seed` hep kuru.
- Kalıp testi: 3600 ıslak tane (r 12.5 mm) kalıba döküldü, wet_grains true,
  merkez 0.277 m. Kalıp collider'ları kapatılınca sim 0. kareye sarıldı → 0
  parçacık. Sim sürerken düzenleme = geri sarma; testler tek kurulum olmalı.

## Değişiklik
`SimulationGridDomainDesc::fluid_seed_grain_saturation` (0..1 kapasite): seed
DEM tanelerine doğum suyu verir (`initMatterGrainBirthWater`), ownership ıslak
doğum sayar. IPC `fluid.seed grain_saturation`, Python, panel, kayıt, hash.
Probe: `scripts/ipc/Probe-WetSandBlock.ps1` (kolon + kaya).

## Sonra
- Kılcal kohezyon eğrisi (MPM yolu) düşük doygunlukta tepe yapmalı.
- Sim sürerken kalıp kaldırma gibi zamanlanmış olaylar: collider `enabled`
  anahtarlanabilir (keyframe) mi? Değilse kalıp testleri yazılamaz.

## 2026-10-23 canlı (exe 18:09, seed ıslaklığı derlendi)

Kolon (seed'li 0.3×0.5×0.3 m blok, r 12.5 mm, 1800 tane): kuru ve ıslak
(0.1/0.3/0.8) HEPSİ tek tane kalınlığına yayıldı (merkez 0.25 → ~0.02).
Köprüler çalışıyordu (8834 köprü, 0.44 kg su). İki kök:

1. **Yuvarlanma direnci 0.02 (yuvarlak tane).** Döküm yığını, aynı malzeme:

   | μ_r | kuru merkez → açı | ıslak (0.5) merkez → açı |
   |---|---|---|
   | 0.02 (Sand ön ayarı) | 0.020 → ~8° | 0.053 |
   | 0.2 | 0.051 → ~30° | 0.076 → ~47° |

   (Koni varsayımı: merkez = H/4, hacim 1800·V_tane/0.6.) Gerçek: kuru kum
   30–35°, nemli daha dik. Sand ve Gravel köşeli → 0.2 / 0.25.
2. **Seed basit kübik kafes (hacim kesri 0.39).** Kayma direnci yok; ilk 5
   adımda ortalama hız 0.52 m/s (serbest düşüşe yakın) — blok bulut gibi
   çöküyor. FCC'ye çevrildi (0.556, en yakın 2.2002R, jitter 0.07R → ≥2.0021R).
   `matter_grain_birth_test.cpp` 125 → 172 tane.

Ayrıca Sand'in su kapasitesi 0 → 0.1 (ıslanabilsin). Yan etki: sıvılı Sand
sahnelerinde taneler artık ıslanır (köprü), tasarım gereği.

Kullanıcıya bırakılan sahne: SandTest (nemli döküm 0.5) + SandDry (kuru döküm),
`WetSandProbe` (μ_r 0.2, kapasite 0.1). Sıradaki build: FCC seed + Sand ön ayarı
→ `Probe-WetSandBlock.ps1` kolon/kaya.

## 2026-10-23 canlı (exe 18:39: FCC + Sand μ_r/kapasite)
FCC seed çalışıyor (aynı blok 1800 → 2813 tane). Kolon 0.5 m: kuru 0.247→0.030,
ıslak 0.3 → 0.039; kayada ikisi de yere iner. Kısa kolon 0.2 m (merkez 0.091):
kuru 0.020, ıslak 0.3 → 0.029, ıslak 1.0 → 0.042 — ayakta kalmıyor.
Sebep (hesap): köprü modeli tutarlı ama yapışma ~0.43 kPa (r = 0.25 mm) →
dikey duvar kritik yüksekliği ~0.11 m; gerçek nemli kumun görünür yapışması
1–3 kPa. Gerçek tane yarıçapı yapışmayı R/r ile ölçekler: ıslak 0.3 ile
r = 0.1 mm → 0.081, r = 0.04 mm → 0.083 (kolon AYAKTA, hafif çöküş).
Karar bekliyor: Sand ön ayarında r'yi küçültmek (ısı filmi de r'yi kullanır)
ya da ayrı bir yapışma çarpanı. Seed boşluğu (0.2R) köprüyü ~23× zayıflatır;
ince kumda yine de tutuyor.

## 3. parti: teğet seed (2026-10-23, derlenmedi)
FCC aralığı d = 2.0002R + 2.83·j, j = 0.03R → d = 2.0851R, boşluk 0.085R,
doluluk 0.653 (gerçek sıkı kum 0.62–0.64). Neden: 0.2R boşluk seed'i olduğundan
hacimli gösteriyor, ilk karelerde sahte çöküş enerjisi üretiyor, köprüleri ~23×
zayıf başlatıyordu; seed settle her adım hızları sıfırladığından boşluklu
seed'de adım başı aynı ~1.4 mm düşüşü görüp plateau kuralıyla erken duruyordu.
MPM/sıvı seed'leri süreklidir (kütle = ρ·hacim/ppc), bu boşluk sınıfı yalnız DEM.

## DEVİR: tane başına gerçek yarıçap + harmonik ortalamalı köprü (yazılmadı)
Başka ajan için iş tanımı:
- Sorun: `grain_real_radius_m` maddeye ait tek değer, domain'de tek tane
  malzemesi. Kum+çakıl karışımı tek gerçek yarıçapla yapışır.
- Fizik: köprü F = 2πγcosθ·r*, r* = 2r₁r₂/(r₁+r₂) (küçük tane belirler);
  Bond sayısı ∝ 1/r² → ince tane yapışkan, iri çakıl değil; su ince taneye
  gider, karışımın yapışmasını ince oranı belirler.
- Kaba taneleme: simüle R, gerçek r → F = 2πγcosθ·R*·(R*/r*), R*, r* çiftin
  harmonik ortalamaları. Şu an `sim_matter_grain.glsl::bridge()` tek
  `real_radius` (masses[2·meta.x]) ve `prefactor = wet.y·max(1, r/real)`
  kullanıyor; `MatterGrainGpu.cpp` cohesion_scale = maximum_radius /
  represented_grain_radius_m (CFL'deki köprü sertliği de buna bağlı).
- Yapılacak: tane başına gerçek yarıçap kanalı (madde + boyut dağılımından;
  per-substance tag ya da per-particle float), köprü ve CFL köprü sertliği
  çift başına; ısı filmi (`MatterGrainThermal.cpp` film_radii) aynı kanalı
  okusun. Shader revizyonu 30 → 31 (ABI). IPC/Python/panel: tane başına
  gerçek yarıçap okunabilir olsun.
- Kabul: aynı domain'de Sand (0.1 mm etkin) + Gravel (4 mm) karışımı; nemli
  kum matrisi çakılı tutmalı, saf ıslak çakıl yapışmamalı. Ölçüm
  `scripts/ipc/Probe-WetSandBlock.ps1` deseniyle (kısa kolon, merkez y).
- Not: 0.2 m ıslak kolon r = 0.25 mm'de çöküyor, 0.1 mm'de ayakta (bu notun
  2026-10-23 canlı bölümü). Görünür yapışma gerçek nemli kumda 1–3 kPa.
