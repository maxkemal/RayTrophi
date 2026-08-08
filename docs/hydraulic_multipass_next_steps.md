# Hydraulic Multi-Pass — Devam Notu

Tarih: 4 Ağustos 2026

## Tamamlanan durum

- Hydraulic node CPU/Vulkan ortak parametre sözleşmesine geçirildi.
- Gerçek solver alanları üretiliyor:
  - Erosion
  - Deposition
  - Discharge
  - Sediment Flux
  - Flow Direction
- Vulkan yolu 9 storage buffer ve 100-byte push constant kullanıyor.
- Tek node içinde resident GPU multi-pass akışı kuruldu:
  1. Conditioning
  2. Incision
  3. Maturation
- Pass aralarında height ve alan buffer'ları GPU'da kalıyor; readback finalde yapılıyor.
- CPU fallback üç aşamayı çalıştırıp alanları aggregate ediyor.
- Multi-pass presetleri eklendi: Balanced, Alpine, Humid, Arid, Custom.
- Hydraulic properties UI fiziksel gruplara ayrıldı ve JSON serileştirmesi güncellendi.
- Son değerlendirme süresi ve solver-space alan integralleri node UI'da gösteriliyor.
- Snow, River, Biome ve Foliage hızlı kurulumları Hydraulic alanlarıyla uyumlu hale getirildi.
- Terrain Fields Output kalıcı alanları:
  - `erosion.hydraulic`
  - `erosion.deposition`
  - `hydrology.hydraulic_discharge`
  - `erosion.sediment_flux`
- Eski dekoratif Sediment Deposit / Alluvial Fan / Delta Formation / Erosion Wizard üretim yolları registry, factory, import ve menüden çıkarıldı.
- Hydraulic shader `glslc` doğrulamasından ve kaynaklar `git diff --check` kontrolünden geçti.

## İlk test

1. `compile_shaders.bat` çalıştır.
2. Projeyi derle.
3. Aynı terrain ve seed üzerinde Single Pass ile Multi-Pass Balanced karşılaştır.
4. Erosion, Deposition, Discharge ve Sediment Flux önizlemelerini kontrol et.
5. Maturation'ın kanalları silmeden düşük eğim ve eteklerde çökelme üretmesini doğrula.
6. Alpine, Humid ve Arid presetlerini aynı terrain üzerinde karşılaştır.
7. Snowy Mountain Valley → River → Biome → Foliage hızlı kurulum sırasını test et.
8. Last Evaluation süre ve integral değerlerini kaydet.

## Sonraki faz: fiziksel Fluvial bağlantısı

Fluvial ayrı solver olarak kalacak. Sıradaki çalışma:

- Hydraulic Sediment Flux için Fluvial'e `Sediment Supply` girişi eklemek.
- Hydraulic Deposition/Discharge alanlarını normalize grafik maskesinden ayrı raw solver alanları olarak taşımak.
- Rainfall rate ve simulation duration parametrelerini fiziksel birimlerle tanımlamak.
- Hücre alanına bağlı su hacmi ve sediment kütlesi kullanmak.
- Telemetry'ye şu bütçeleri eklemek:
  - Eroded mass
  - Deposited mass
  - Carried sediment
  - Sediment leaving domain
  - Sediment mass error
  - Water input
  - Evaporated water
  - Water leaving domain
- Multi-pass sonraki droplet başlangıçlarını önceki pass discharge alanına göre importance-sample etmek.
- Watershed yönünü ana nehirler için otoriter tutmak; Hydraulic yönünü lokal rill/yamaç yönlendirmesi olarak kullanmak.

## Beklenen mimari

`Base Height → Hydraulic Multi-Pass → Watershed → Fluvial → River/Lake → Snow/Biome/Foliage`

