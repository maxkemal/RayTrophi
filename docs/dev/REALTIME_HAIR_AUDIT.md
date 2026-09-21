# Realtime hair — malzeme ve maliyet ilk incelemesi

> **Durum:** AKTİF — kaynak incelemesi tamamlandı; uygulama kodu değiştirilmedi, hair sahnesinde performans ölçülmedi (2026-09-06).

**Kapsam düzeltmesi:** kullanıcı performans şikâyetinin hair'e özel olmadığını,
genel Realtime'da **kamera hareketinde düşük FPS** olduğunu açıkladı. Bu
notun performans maddeleri yalnız hair kaynak denetimidir; kullanıcının
darboğazı olarak yorumlanmamalıdır. Hair işi malzeme desteğidir. Genel
inceleme: [REALTIME_CAMERA_MOTION_PERF.md](REALTIME_CAMERA_MOTION_PERF.md).

## Kesin kaynak bulguları

1. `shaders/hair_viewport_frag.frag`, malzeme veya ışık tamponu okumaz.
   Sabit kahverengi kök / altın uç rengi ve alpha=1 üretir. RT hair malzeme
   parametreleri bu shader'a ulaşmaz. Bu durum parametre tuning sorunu değildir.
2. `VulkanViewportBackend.cpp` hair pipeline'ı LINE_LIST, lineWidth=1,
   depthWrite=false, tek örnekli raster oluşturur. Vertex akışı yalnız
   position + kökten uca koordinattır; tangent, radius, root UV ve groom
   material index yoktur. Gerçek saç shading'i yalnız fragment shader
   değiştirerek tamamlanamaz.
3. `Renderer::uploadHairToGPU()` raster bölümünde görünür groom'ların bütün
   procedural children'ını CPU'da `generateChildStrand()` ile üretip her
   segmenti iki adet 16-byte vertex'e açar. 1 milyon segment yaklaşık 32 MB
   ham çizgi verisidir; geçici vector kapasitesi ve child üretim maliyeti ayrıca gelir.
   Bu bölüm viewport quality, mesafe veya ekran kaplamasına göre LOD yapmaz.
4. `uploadHairViewportLines()` her çağrıda in-flight drain yapıp bütün çizgi
   verisini upload eder. Kapasite yettiğinde allocation tekrarlanmaz; buna
   rağmen upload tekrarlanır. **Her kare çağrıldığı henüz kanıtlanmadı.**
5. Aynı `uploadHairToGPU()` çağrısı raster çizgilerini yükledikten sonra render
   backend'inin RT hair yoluna da devam eder. Topolojiye göre GPU guides /
   CPU strand ve BLAS işi oluşabilir; raster modda bu maliyetin ertelenip
   ertelenemeyeceği bütün çağıranlarda incelenmeli. Kör bir erken return,
   Rendered'a geçişte veya animation/refit'te eski veri bırakabilir.
6. UI dirty/transform sync, raster rebuild, animasyon, API ve mod geçişleri
   upload çağırır. Mevcut aynı-frame coalescing guard render backend'inin
   `hairGpuActive()` durumuna bağlıdır; raster içeriği revision cache'i değildir.
7. Raster mesh gölge üretimi `m_rasterMeshes` çizer; hair overlay bu caster
   listesine katılmaz. Hair shader gölge atlası da okumaz. Dolayısıyla mevcut
   hair için gölge ekleme ayrı bir pipeline entegrasyonudur.
8. Hair draw ayrı `vkCmdDraw(hairLineVertexCount)` çağrısıdır. Mesh üçgen/
   instance sayaçları hair segment yükünü anlatmaz. Hair'e özel üretim,
   upload byte/ms, çizilen segment ve GPU pass zamanı olmadan darboğaz
   kesinleştirilemez.

## Açık uygulama okuması

Salt okunur IPC: `viewport.status` shading=rendered, `hair.list` boş dizi.
`viewport.frame_telemetry` daha önceki raster karelerden değerler taşıyor;
bunlar hair benchmark'ı olarak kullanılmadı. Sahne/mod değişmedi, uygulama
başlatılmadı, derleme yapılmadı.

## Önerilen çalışma sırası

1. Hair sahnesinde sabit kamera, kamera hareketi, materyal değişimi ve groom
   geometri değişimini ayrı ölç. Üretim/upload sayısı ve byte miktarını
   shader/draw maliyetinden ayır; önce ihtiyaç varsa ortak UI/API/IPC üzerinden
   hair ölçüm alanlarını ekle.
2. Geometry, transform ve material revision'larını ayır. Renk değişiminde
   çocuk telleri yeniden üretme; değişmeyen geometriyi yeniden aktarma.
   Raster için RT BLAS işini erteleme ancak Rendered moduna geçiş dahil
   bütün çağıranların tazelik sözleşmesi kurulunca yapılmalı.
3. Mevcut groom/RT malzeme otoritesinden tangent, radius, root UV ve material
   ID taşıyan raster yolunu kur. Kalınlık/coverage ve hair'e özgü ışık
   değerlendirmesi aynı yolun parçasıdır; sabit gradient'i cilalamak yeterli olmaz.
4. Çocuk tel ve segment LOD'sini mevcut viewport quality'ye bağla; kararlı
   seçim kullan, kamera hareketinde teller rastgele değişmesin. Full modunun
   mevcut tam-geometri anlamını koru. Farklı Vulkan cihazlarının RT buffer
   adreslerini viewport'a doğrudan taşıma.
5. Hair cast/receive shadow ve yoğun tel kümelerinin görünürlüğünü ekle;
   ekran kaplaması/overdraw maliyetini ayrıca ölç.

Bu sıra bir uygulama teslimi değildir. Mevcut saç üretimindeki Triangle facade
yolu yeni özellikler için çoğaltılmamalı; scalp geometri işi gerekirse flat
TriangleMesh/DNA verisi esas alınır. 2000 satır üzerindeki Renderer ve backend
dosyaları yalnız entegrasyon alır; yeni implementation odaklı modüllere gider.
