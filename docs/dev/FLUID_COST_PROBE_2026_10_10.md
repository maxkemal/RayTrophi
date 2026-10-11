# Açık su sahnesi maliyeti ve proje açılış karesi — 2026-10-10

Kullanıcı isteğiyle açık uygulamada dış named-pipe IPC ölçümü. Build veya shader
derlemesi yapılmadı. Başlangıç: kare 0, durmuş sahne; Physics Domain 1, Matter,
Vulkan, Water, thermal_liquid_enabled=false; tek sıvı kaynak 10000 parçacık/s.

## Ölçüm

5 ilk adım + 30 kayıtlı adım, dt=1/60. Son nüfus 5833. Son 10 medyanı:

| Sayaç | Değer |
|---|---:|
| Sıvı total_ms | 58.208 ms |
| P2G | 9.5175 ms |
| Pressure | 19.00295 ms |
| Density | 3.53155 ms |
| Advect | 1.9905 ms |
| Transfer batch end | 31.0575 ms |
| Upload | 26094384 byte/adım |
| Download | 59499912 byte/adım |

Gaz tüm örneklerde measured=false; bu sahnede termal zincir kapalı. Grid 128³
(2097152 hücre). Compact P2G/FLIP açık fakat transfer_sparse_canonical=false;
durum "compact P2G/FLIP with dense projection/contact publication".
Basınç aktif pencere kullanıyor; pressure_sparse_used=false.

Bu sahnenin mevcut yükü gaz sıcaklık/erime/kaynama çalışmasına atfedilemez:
termal zincir kapalı ve gaz adımı ölçülmemiş. Ana gözlenen yük sıvı dense MAC
yayını/aktarımları, submit beklemeleri ve pressure aşamasıdır. Batch end süreleri
P2G/pressure sürelerinin içinde de yer alır; tablo satırları toplanmaz. Bunlar
host duvar süreleri, ayrı GPU kernel timestamp ölçümü değildir. Eski exe ile
A/B yapılmadığı için bugünkü değişikliklerin regresyonu olmadığı iddia edilmez.

Kaynakta FluidGpuPressure.inl saf-liquid projected MAC hızlarını host'a indirir;
Matter model resident yolu bu indirmeyi atlar. FluidGpuTransferStages.inl,
grid_velocity_is_current=false ise MAC hızlarını tekrar yükler. Saf-liquid
residency/sparse publication işinin açık olması bu sayaçlarla uyumludur.
Ham kayıt: [fluid_cost_probe_2026_10_10.json](fluid_cost_probe_2026_10_10.json).

## Ek ölçüm — termal açık, kare 118

Kullanıcı önceki performansıyla farkı sorgulayınca salt okunur yeniden ölçüldü.
Bu örnek ilk termal-kapalı ölçümden farklıdır: thermal_liquid_enabled=true,
49166 parçacık; sıvı step_stats measured=false (bu okuma sıvı adım süresi vermez).
Gazın yayımlanan son adımı total_ms=823.321, inventory_advection_ms=712.777
(yaklaşık %86.6). GPU pressure 34.201 ms, source upload 10.049 ms,
host sync 7.566 ms. Evaporated son-adım sayacı 0 olmasına rağmen gaz fiziksel
envanteri 90.69336 kg / 346427519.9 J ve 617379 aktif hücre; sıvı 373.15–378.12 K.
Bu nedenle "bu adımda buharlaşma yok" envanterin boş olduğu anlamına gelmez.

FluidMistPhaseExchange.cpp::advectGasPhaseInventory iki host kg/J dizisini
GridFluid::advectPassiveScalarField ile ayrı ayrı taşır; her biri için ayrıca
tam alan sanitize/toplam/normalizasyon döngüleri vardır. ParticleSimulation.cpp
gaz çözümünün ardından nihai host hız alanıyla bunu çalıştırır. Termal aktif
sıvı gaz adımını uyandırır. Bu sahnenin termal-açık aşırı yükünde baskın ölçülen
süre PCIe kopyası değil CPU envanter advection'dır; kopya/senkron maliyeti ek yük.
Önceki termal-kapalı sıvı darboğazı ayrı kalır. Aktif destek/GPU envanter taşıma
yolu ve termal-açık/kapalı eş koşullu A/B sonraki düzeltmede önceliklidir;
envanteri silmek veya enerji muhasebesini kapatmak kabul edilebilir çözüm değildir.

## Proje açılışı düzeltmesi (kaynak; build bekliyor)

ProjectManager::openProject artık kayıtlı render playback/current frame ve
timeline scrub frame yerine 0 kullanır; playback kapalıdır. Track/keyframe
verileri korunur. Main ana-thread yükleme finalizasyonunda TimelineWidget reset
edilir; eski projenin UI karesi yeni sahneye taşınmaz. API/Python/IPC project.open
aynı ProjectManager yolunu kullanır ve mevcut UI senkronuyla 0'ı yayınlar.
New Project mevcut scene.clear/RenderSettings/TimelineWidget reset yolunu korur.

Kullanıcı build sonrası kabul:

1. Kare 120'de durmuş ve playing durumlarından File > New Project: kare 0,
   playing=false, eski kareye simülasyon replay yok.
2. Kare 120'de kaydedilmiş simülasyonlu proje: File > Open, recent/startup ve
   project.open (Python/IPC) ile aç; ilk kontrol kare 0, playing=false.
3. Keyframe trackleri, timeline aralığı ve emitter tarifleri korunmalı; kullanıcı
   açıkça kare 120'ye scrub edince olağan replay/cache davranışı çalışmalı.
4. Ana planın saf-liquid residency kabulünde aynı sahneyle byte/adım ve total_ms
   karşılaştırılmalı; termal açık bir referans ayrıca ölçülmeli.
