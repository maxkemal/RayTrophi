# SDF yüzey üretimi: fiziksel bin ve sparse çalışma tile'ları

Kaynak hazır; kullanıcı C++ derlemesi ve canlı maliyet/kalite kabulü bekliyor.
Shader değiştirilmedi, build çalıştırılmadı. Mevcut sahne Splat, 70,000
parçacık/kare 168 olduğundan bu turn'de yeni üretici için canlı hızlanma ölçülmedi.

## Değişiklik

`FluidLevelSet.cpp` parçacık CSR'ını rafine SDF gridine kuruyordu. x2/x4 kalite
offset/count/cursor belleğini 8/64 kat büyütüyor, her yüzey örneğinin büyük
ölçüde boş refined komşuluk stencil'ini taramasına yol açıyordu. CSR artık
fiziksel kernel genişliğinde, sim gridinden daha ince olmayan binlere kuruluyor.
Kalite çarpanı fiziksel parçacık komşuluğunu büyütmüyor. Kernel ve parçacık
yarıçapı metre olarak aynı; kalite parametrelerinin anlamı değişmedi.

Tam rafine domain üzerinde üç occupancy dilation taraması ve tam-domain
aktif-liste taraması kaldırıldı. Yeni `SurfaceWorkTiles` yalnız kanonik SoA
parçacıklarının etkilediği sabit-grid 8³ tile'ları sıralayıp tek yazarlı çalışma
listesi oluşturuyor. Domain dışında, seçimin dışında veya tükenmiş parçacıklar
bu yüzeye katkı yapmıyor. Tile grid origin'i ve voxel örnek merkezleri aynı.

Yumuşatma desteğine iteration*voxel halo ekleniyor. Altı-komşulu Jacobi
güncellemesi yalnız tile alanında; geçici alan tam rafine domain boyutunda
değil, çalışma alanı boyutunda. Her sweep'in okuma ve yazma fazları ayrı OMP
bariyerleriyle sıralanıyor. Gerçek domain sınırındaki komşu sayısı korunuyor.
Aktif/surface sayaçları da tile alanından hesaplanıyor.

Kalite düzeltmesi: anisotropi kernel desteği G'nin en büyük semiaxis sınırıyla
hesaplanıyor; eski 4x reach tavanı kaldırıldı. Smoothed-centre kaymasının
üst sınırı source-bin sorgusuna ekleniyor. Önceden kaymış/uzamış kernel'in
kuyruğu eksik aday sorgusu nedeniyle kesilebiliyordu. İzole parçacık spherical
fallback'i, kütle yarıçapı ve surface offset davranışı korunuyor. Komşuluk
toplama sırası değiştiğinden floating-point sonuçta bit-eşitlik beklenmez;
geometrik toleransla karşılaştırılır. Ani kernel desteğinin tamamlanması
anisotropi açık sahnede amaçlı şekil farkı yaratabilir.

## Maliyet ve sınırlar

128³, kernel=2 sim voxel örneğinde CSR 64³ bin kullanır; m=1/2/3/4'te aynı
262,144 bin. Eski m=4 CSR 512³ bin kullanıyordu. Count/offset/cursor toplamı
kabaca 2.5 GiB yerine 5 MiB (64-bit size_t). Occupancy'nin iki tam-domain
byte tamponu da kalktı. Tile/smoothing alanı parçacıkların yayılımına bağlıdır;
tüm domain doluysa küçülme garanti değildir.

**Final `sdf_out` hâlâ mevcut dense render ABI'sinde.** 128³/m4'te tek float
çıktı hâlâ 512 MiB; başlatma, density proxy dönüşümü, NanoVDB dönüşüm/upload
işleri bu partide tam sparse storage'a taşınmadı. CPU yüzey üreticisidir;
bu değişiklik GPU shader üreticisi eklemez. Realtime m4 hızı veya belirli
yüzde hızlanma vaat edilmez. Cache/frame/solver'ın fizik state'i değişmedi.

## API ve ölçüm

Native FluidDomainInfo, Python ve IPC get/list ortak core stats'ı yayımlar:
`surface_work_cells` (halo ve tile padding dahil), `surface_bin_cells`,
`surface_bin_ms`, `surface_anisotropy_ms`, `surface_schedule_ms`,
`surface_gather_ms`, `surface_smooth_ms`. Mevcut `surface_build_ms` output
başlatması dahil üretici toplamını ölçer; downstream VDB/upload/render dahil
değildir. UI kalite alanları ve bütün existing script/IPC authoring yolları
aynı buildLevelSet servisini kullanmaya devam eder. Yeni authoring düğmesi yok.

Read-only dış IPC snapshot:

```powershell
python scripts/test/rt_probe_surface_cost_ipc.py --label sdf_m1
```

## Yapılan ve bekleyen kontroller

Python oracle PASS: disconnected/isolated drop, domain duvarı, odd tile
sınırları ve 0/1/2/4 smoothing sweep için dense/sparse alan eşitliği; fiziksel
bin adaylarının all-particle sorguyla eşitliği; refinement bağımsızlığı ve
native/script/IPC/proje sözleşmeleri. Source delimiter taraması, cache,
gas/WW ve background policy regresyonları PASS; descriptor --check PASS.
Bu kontroller C++ veya native uygulama çalıştırmaz.

Tek kullanıcı C++ derlemesinden sonra aynı durmuş parçacık state'inde m1/m2,
anisotropi kapalı/açık ve SDF/Splat karşılaştırılır. `surface_bin_cells` kaliteyle
artmamalı; build ve stage süreleri, toplam kare süresi, RAM ve VRAM ayrı okunur.
İnce jet/sheet, iki ayrık damla, dolu havuz, domain duvarı ve hızlı hareketten
sonra yüzey/normal/titreşim kontrol edilir. Eski/yeni üretici farkı toleransla,
çözünürlük ve fiziksel yarıçap sabit tutularak değerlendirilir. Timeline geri
sarım ve paused kalite/material/label editleri de test edilir. Büyük m4 test
öncesi dense output için bellek payı göz önünde tutulur. Native kabul bekliyor.
