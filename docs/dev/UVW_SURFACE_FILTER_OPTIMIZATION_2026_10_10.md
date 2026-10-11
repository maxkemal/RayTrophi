# UVW sparse çalışma ve yüzey merkezi filtresi

Kaynak hazır, kullanıcı C++ derlemesi ve canlı hız/kalite kabulü bekliyor.
Shader değiştirilmedi; build çalıştırılmadı. Eski binary'de ölçülen 47–51 ms
UVW ve 8–9 ms SDF bu partinin hızlanma sonucu değildir.

## UVW

`buildMaterialCoordinateGrid` odaklı `FluidMaterialCoordinates.cpp` modülüne
taşındı. Gather artık tüm domain üzerinde çalışmıyor: seçili, geçerli ve
tükenmemiş kanonik SoA parçacıklarının etrafındaki SurfaceWorkTiles kullanılır.
Parçacık CSR'ı fiziksel kernel genişliğinde SurfaceParticleBins'e kurulur.
Extrapolation ve iki Jacobi displacement smoothing pass'i aynı tile planında
çalışır. Smoothing scratch tam 3*N grid kopyası yerine çalışma-lane Vec3
alanıdır; okuma/yazma fazları ayrı OpenMP bariyerleriyle sıralanır.

Displacement `uvw - position`, iki jenerasyonun per-parcel karışımı, zero
residual'un geçerliliği ve selection/mass kuralı korunur. Texture coordinate
gerekliliği yalnız “albedo texture var mı” diye kapatılmadı; procedural/normal/
diğer materyal tüketicileri de bu alanı okuyabilir. Bu parti mevcut UVW
davranışını hızlandırır, materyal bağımlılıklarına göre authoring'i kesmez.

Alanı dışa uzatma minimum üç sweep; filtrelenmiş merkez ve anisotropik kernel
desteği daha genişse tüketicinin interpolation halo'suna göre artırılır.
Bu artış amaçlı doğruluk düzeltmesidir. Aynı extension sayısı için dense ve
sparse algoritma matematik oracle'ında eşleşir. Son yoğun UVW çıktı/mask ABI'si
korunur; tam sparse render storage migration değildir.

## Yüzey kalitesi ve anisotropi maliyeti

Mevcut `position_smoothing` spherical kernel'de yok sayılıyordu. Şimdi hem
spherical hem anisotropic üretici geçici render merkezlerini yerel weighted
mean'e yaklaştırır. Canonical position, velocity, mass veya solver state'i
değiştirilmez. `anisotropy_neighbor_min` altında merkez değiştirilmez, spherical
droplet fallback'i korunur. Spherical yolda covariance/eigensolve yapılmaz.
UI'daki mevcut radius/smoothing/min-neighbour kontrolleri aniso kapalıyken de
görünür; script/IPC aynı core parametreleri kullanır. Yeni authoring parametresi
eklenmedi. **position_smoothing=0 raw-centre yolunu geri verir.**

Merkezler filtrelendikten sonra output CSR tekrar kurulur. Böylece query
yarıçapına worst-case merkez kayması eklenmez. Output bin genişliği kernel
desteğine göre seçilir; güçlü aniso açıkken küçük binlerde geniş, çoğu boş
stencil taraması azaltılır. İlk neighbourhood CSR'ı centre/covariance hesabına
hizmet etmeye devam eder. `surface_bin_cells` output CSR hücre sayısını raporlar.
Legacy `surface_anisotropy_ms` artık particle preparation (centre filter ve
opsiyonel covariance) zamanıdır; anisotropi kapalıyken de sıfır olmayabilir.

Varsayılan position_smoothing=.9, önceden spherical yolda etkisizdi; artık
mevcut sahnenin siluetini değiştirebilir. Bu render filtresi geometrik hacmi
tam koruyan bir constraint değildir. Damlacık/jet/sheet/tank görünümü ve hacim
hatası yeni binary'de değerlendirilmeden “kalite kabul edildi” denmez.
Bu değişiklik SDF formülünü gerçek-distance reinitialization ile değiştirmez
ve shader normal/marching kodunu değiştirmez.

## Kontroller ve sonraki kullanıcı build'i

Kaynak/math PASS: sparse/dense UVW extrapolation ve smoothing (3/5/8 sweep),
identity/translation residual, isolated centre, zero filter, symmetry/tolerance,
physical bins, refinement bağımsızlığı, source ownership ve panel/proje XML.
Cache, WW ve gaz yüzey regresyonları PASS. Source delimiter taraması PASS.
Kontroller native C++/GPU çalıştırmaz.

Tek C++ derlemesinde yeni SurfaceParticleBins.cpp ve FluidMaterialCoordinates.cpp
proje kayıtları dahil olmalı; shader derlemesi gerekmez. Açık test sahnesinin
aynı cache karesinde:

1. Default spherical m1, position smoothing 0 ve .9 görüntü/siluet/hacim
   karşılaştırması. İzole damla ve iki ayrı stream yanlış birleşmemeli; tank
   sınırı/ince sheet aşırı daralmamalı. Normal/pütür ve frame-to-frame jitter
   ayrı kontrol edilir.
2. `perf.list` içinden `sim.render.surface.material_coordinates`,
   `sim.render.surface.level_set`, `sim.timeline.render_sync` karşılaştırılır.
   Parent/child süreler üst üste eklenmez. Yeni hızlanma miktarı ölçülmeli.
3. Aniso on/off ve kalite m1/m2; output CSR, particle preparation ve gather
   maliyetleri, toplam kare/RAM/VRAM okunur. Blank UI capture kapalı olmalı;
   görsel capture ayrı alınır ve ayarı geri yüklenir.
4. Doku/procedural advection, UVW nesil karışımı, cache rewind ve paused
   label/material edits. Existing dış-terminal fixture'ları:

   ```powershell
   python scripts/test/rt_test_fluid_material_coords.py
   python scripts/test/rt_test_fluid_surface_fullness.py
   ```

   Bu fixture'lar test rig'i kurabilir; disposable sahnede çalıştırılır.
   Kullanıcının açık projede authored ayarları bu kaynak turn'ünde değiştirilmedi.
