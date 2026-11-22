# RayFusion Play kaynak kurulum planı

Player tek çizim yolu olarak RayFusion MaterialPreview kullanır. Studio ile aynı
PBR materyalleri, ışıklandırma ve yüzeye işlenen ıslak boya korunur. Kaynakları
aktif sahne ve oyun ihtiyacı belirler. Donanım gereksinimleri bu kaynaklardan
türetilir ve [Vulkan denetimi](VULKAN_RASTER_REQUIREMENTS_AUDIT.md) ile karşılaştırılır.

## Mevcut kurulum

| Kaynak | Kod yolu ve karar |
| --- | --- |
| RayFusion ve sunum | `RuntimeSceneViewer` MaterialPreview seçer. Ortak HDR ve LDR render pass kurulumu `RayFusionViewportPasses` modülüne taşındı. Player shader dizinini MaterialPreview üzerinden bulur. Video derinlik kartı ve billboard çizimi post sonrası LDR geçişinde devam eder. |
| Materyal ve boya | `RuntimeSceneViewer` materyal buffer'ını yükler ve her kare değişen boya kaynaklarını yayımlar. Materyal verisi, UV atlası, yüzey ıslaklığı, GPU damla çarpışması ve kaplama ölçümü aynı servislerden devam eder. |
| Editör | Player Solid/Matcap, grid, edit mesh, seçim maskesi ve outline kaynaklarını kurmaz. Seçim maskesi readback çağrısı Studio gizmo UI'a ait. Player zaten groom sahnelerini kabul etmediği için Studio hair-line kurulumu da ayrıldı. SurfaceSDF ve volume için RayFusion pipeline'ları korunur, Solid varyantları Studio'da kurulur. |
| Atmosfer | Player fiziksel gökyüzü LUT pipeline'ını ilk Nishita LUT üretiminde, aerial froxel'i aerial perspektif veya yükseklik sisi istendiğinde, bulut pipeline'larını sahne hava çizimi veya açık bir bulut örnekleme isteğinde kurar. Lens weather compute pipeline'ları ilk etkin kamerada kurulur. Post shader'ın kullandığı lens descriptor seti geçerli kaynaklarla hazırlanır. Studio'nun kurulumu korunur. |
| İsteğe bağlı geçişler | Player GPU skinning'i ilk skinning verili raster mesh ile, volume pipeline'ını ilk GPU hacmiyle, SurfaceSDF'yi ilk aktif ve geçerli yüzey slotuyla hazırlar. Kontroller aynı boyuttaki erken dönüşten önce çalışır. Transmission, yarı geçirgen balonların kırılması için mevcut kurulumuyla devam eder. |

## Uygulama sırası

1. **Studio kurulumu ile ortak RayFusion kurulumunu ayır.** Kaynak değişikliği
   hazır, birleşik canlı kabul bekliyor. Player MaterialPreview kurulamazsa
   kurulum aşaması mevcut runtime hata alanına yansır. Studio'nun mod geçişleri
   aynı pipeline'ları kullanmaya devam eder.
2. **Sahne kaynaklarını gerektiğinde hazırla.** Dünya modu, aktif bulut/hava ayarı,
   animasyon ve sahnede kullanılan materyal/geometri özelliği üzerinden tetikle.
   İlk tüketimden önce hazırla. Aynı çözünürlükte sahne değişimi, özellik açıp
   kapatma ve resize senaryolarını kapsa. Kullanımdaki kaynakları ancak GPU kareleri
   tamamlandıktan sonra emekliye ayır. Desteklenmeyen isteğe bağlı bir özellikte
   hata aşamasını mevcut runtime durumunda görünür kıl.
3. **Gerçek Vulkan sözleşmesini çıkar.** MaterialPreview çizim bloğu
   208 bayttan 128 bayta indi. Kamera view matrisi ve rim ışığı set 0,
   binding 28'deki 80 baytlık frame UBO'suna taşındı. Ana çizim, transmission,
   gölge, sky ve hacim geçişleri yeni çizim ABI'sini paylaşır. Descriptor bütçesi
   artık RayFusion layout'undan hesaplanıyor. Fragment kaynak hesabına dört
   renk hedefi dahil. Log, cihazın seçilen doku dizisi kapasitesini gösterir.
   Kurulum, gerçekten etkin descriptor indexing, independentBlend ve
   fragmentStoresAndAtomics özelliklerini kontrol eder. Player adaptörü,
   fiziksel Vulkan sürümünü, driver değerini ve push constant limitini kaydeder.
   Format koşullarını kullanılan geçişler üzerinden doğrula.
   Manifestteki 40 zorunlu ve koşullu shader Vulkan 1.2 hedefinde doğrulandı.
   Player instance sürümünü loader'a göre 1.2 veya 1.3 seçer. Fiziksel
   adaptör seçiminde RayFusion'ın temel Vulkan özellikleri ve birleşik
   graphics/compute queue kontrol edilir. Simülasyon uyumsuz SPIR-V ve push
   aralıklarını shader modülü oluşturmadan atlar. Tam 1.2 kabulü bekler.

## Kabul ve build sırası

20 Kasım 2026 kullanıcı kabulü. Yeni Player build'inde balon, hacim, bulut ve
iskelet sahneleri doğru açıldı. Önceki build'de sentetik kamera kartı ve
derinlik sırası canlı doğrulandı. Pipeline tahsis sayıları ölçülmedi.

- Kullanıcı Player ve Studio hedeflerini bir artımlı build oturumunda alır.
  Aynı kabul oturumunda Player'ı yeni EXE ile aç, harici terminalden
  `python scripts/test/rt_player_balloon_paint_ipc.py` çalıştır. Bu prob test
  sahnesini açtığı için ayrı test oturumu kullan.
- Balon odasında aynı proje ve kamerayla referans görünümü, patlama sonrası
  damlaları, ıslak boyayı, kaplama sayacını, reseti ve kamera derinlik kartını
  kontrol et. Aynı oturumda pencere boyutunu değiştir ve projeyi yeniden aç.
  Volume pipeline değişikliği için `python scripts/test/rt_player_volume_ipc.py`
  ile hacim sahnesini doğrula.
- Studio'da MaterialPreview, Solid ve Matcap arasında geçiş yap. Grid, edit
  vertex/edge/face overlay ve seçim outline'ını doğrula.
- Balon odasından aynı çözünürlükte önce volume, sonra skinned karakter
  sahnesine geç. Hacim çizimini ve karakter deformasyonunu doğrula. Açılış
  tanısında, balon odasının bu iki pipeline'ı kurmadığını ve sahne geçişinden
  sonra ilgili pipeline'ın hazırlandığını kontrol et.
- Atmosfer ayrımı için düz renk balon odasından aynı çözünürlükte Nishita ve
  bulut sahnesine geç. `python scripts/test/rt_player_clouds_ipc.py` ile bulut
  çizimini ve açık örnekleme yolunu kontrol et. HDRI veya renk dünyasında
  yükseklik sisini ayrıca doğrula. Katmanlı bulut sahnesi kullanıcı testinde
  geçti, yükseklik sisi ayrı kabul bekliyor.
- Lens weather kapalı bir projede aç, aynı kamerada IPC ile özelliği etkinleştir
  ve ilk yağışlı kareleri doğrula. Ardından kapatıp post görüntüsünün normale
  dönmesini ve tekrar etkinleştirince pipeline'ların kullanılmasını kontrol et.
- SurfaceSDF ayrımı için balon odasından normal hacme, sonra aynı boyutta gerçek
  isosurface sahnesine geç. `render.volume_slots` ile aktif, grid adresi olan
  `source_type=4`, `volume_type=2` slotunu kontrol et ve yüzeyi görsel doğrula.
  Balon odasında yarı geçirgen materyalin kırılmasını karşılaştır.
- Yeni C++ EXE'yi güncellenmiş shader setiyle birlikte kullan. Vulkan 1.2
  doğrulamasını `./scripts/package_player.ps1 -ShaderInventoryOnly` ile
  kontrol et. Yeni EXE'den sonra tek Player paketi mevcut paketleme hattından
  üretilir. Geometri referansı düz TriangleMesh/DNA SoA olarak kalır.
- Son Player Release build'inde Vulkan 1.3 makinede workshop, kamera kartı,
  katmanlı bulut ve normal hacmi doğrula. Vulkan 1.2 fiziksel cihazda aynı
  workshop açılışı, patlama, boya ve kart akışını tekrarla. Log'da instance
  sürümünü ve seçilen cihazın API sürümünü ayrı kontrol et. Yalnız 1.2
  fiziksel cihaz görülmesi, 1.2 loader dalını tek başına doğrulamaz.
