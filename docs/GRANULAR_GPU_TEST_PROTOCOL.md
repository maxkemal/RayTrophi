# Granular GPU acceptance test

Bu protokol, Vulkan granular path P2G/G2P'ye bağlandıktan sonra uygulanır.
Build ve shader compilation kullanıcı tarafından çalıştırılır.

## Ortak ayarlar

- Backend: Vulkan Compute
- Preset: `sand`
- Gravity: `(0, -9.81, 0)`
- Voxel size: önce `0.05 m`, sonra `0.025 m`
- Particles: yaklaşık `50k` ve `100k`
- Reseed: test boyunca kapalı
- Render: particles/spheres; surface smoothing kapalı veya sabit
- Capture: başlangıç, 30, 60, 120 ve 240 frame

Her koşuda Fluid Step telemetry kaydedilir: total, P2G, pressure, G2P,
constitutive, yield count, detach count, invalid count, particle count,
active cells, mass and momentum.

## Test A — Elastic/yield shear

Kapalı bir kutuda yatay tabakayı eğik hareket eden üst plaka ile kes. İlk kısa
harekette malzeme elastik kalmalı; kritik shear aşıldığında yield count artmalı
ve üst tabaka kaymaya başlamalıdır.

Kabul: invalid count = 0, parçacık sayısı sabit, yield count sıfırdan pozitif
değere geçer, tabaka tek rijit blok gibi kilitlenmez.

## Test B — Pour/repose angle

Bir hazneden düz zemine sabit debiyle kum dök. Akış kesildikten sonra 240 frame
bekle.

Kabul: yığın belirgin bir repose angle ile durur; tüm malzeme sıvı gibi yayılmaz;
settling son 60 frame'de sınırlı kalır; detach count yalnızca serbest uçlarda
görülür.

## Test C — Impact/disintegration

Kompakt kum hacmini kısa mesafeden sert zemine bırak. Çarpışma anında shear ve
yield yükselmeli, üst/kenar parçacıkları ayrılmalı ve sonra tekrar tek viskoz
hacim halinde birleşmemelidir.

Kabul: impact frame'lerinde yield/detach artışı, sonrasında kütle korunumu,
invalid count = 0 ve parçacıkların kalıcı olarak tek blob'a dönmemesi.

## Çözünürlük karşılaştırması

Test B'yi 0.05 m ve 0.025 m voxel size ile tekrarla. Repose angle ve yığın
yüksekliği çözünürlükle tamamen değişmemeli; fark raporlanmalı, gizlenmemelidir.

## Rapor formatı

Her test için backend, voxel, particle count, frame time ortalaması, pressure
dot-sync süresi, constitutive süresi, yield/detach toplamı, final particle count,
mass error, momentum error ve kısa görsel not gönder.
