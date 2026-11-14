# Player: ilk .rtp → Vulkan görüntüsü

## 2026-11-14 akşam — canlı görüntü kabulü PASS (Claude)

Kullanıcı Player'ı derleyip açtı (exe 18:38). `python scripts/test/rt_player_scene_ipc.py
--player-only` → **PASS** (çıkış 0). EchoRoomsLab iki kez yüklendi, kamera arada
taşındı: 26 flat mesh / 312 üçgen, 26 draw call, 156 görünür üçgen, 223 submit = 223
consume, `device_lost=false`, `error=""`. Ret sahnesi beklendiği gibi
`unsupported_in_player_scene: instances` döndü. Ekran görüntüsü elle incelendi:
koridor, duvarlar, karolu zemin ve gökyüzü doğru perspektifte (görüntü scratchpad'de,
repoya girmedi). Device-loss düzeltmesi tuttu.

★ Sessizce makul görünen tek nokta: `material_count=1`, `default_material_used=true`.
EchoRoomsLab bir akustik test sahnesi ve görüntü tek tip gri; sahnenin gerçekten tek
materyalli olduğu **kanıtlanmadı**. Sonraki kabul sahnesi birden çok renkli materyal
içermeli; orada da 1 görülürse materyal tablosu yüklenmiyor demektir.

## 2026-11-14 canlı ilk kare sonucu ve düzeltme

Kullanıcı Player build/open PASS bildirdi. İlk IPC durumunda sahne yüklenmemişti
(0 nesne / 0 render çağrısı). Dış kabul betiği EchoRoomsLab'ı yükledi: parser
272 ms, 26 flat mesh / 312 üçgen / 3 ışık; raster build 68 ms. İlk MaterialPreview
karesi ana thread'i uzun süre meşgul etti; CDB salt okunur stack örneğinde
NVIDIA nvgpucomp64/nvoglv64 sürücü derleyici çağrıları görüldü. Debugger qd ile
ayrıldı. Sonraki stack'te thread yeniden FrameCycle beklemesindeydi; fakat IPC
`device_lost=true`, 1 submit / 0 consume / frame_available=false bildirdi.
**Canlı görüntü kabulü FAIL.** Kalıcı bir dosya okuma beklemesi olduğu söylenemez;
cihaz kaybının kesin GPU kök nedeni henüz kanıtlanmadı.

Kaynakta iki hazırlık eksikliği düzeltildi:
- Lazy Vulkan başlangıcından sonra SceneTextureManager gerçek device yetenekleriyle
  yeniden initialize edilir. İlk snapshot'ta görülen Vulkan=no/MaterialPreview=no
  değerleri Player tarafından zamanında güncellenmiyordu.
- Örnekte materyal tablosu boştu; Renderer boş tabloyu GPU'ya yüklemeden dönüyordu.
  ID 0 için açık loglu standart gri PBR materyali eklenir. Diğer eksik materyal
  referansları doğrudan flat SoA vertex ID'lerinden doğrulanıp hata döndürülür.
  GPU material buffer gerçekten oluşmadıysa render'a geçilmez. Durumda
  material_count/default_material_used/material_buffer_ready alanları eklendi;
  dış kabul betiği buffer'ı da doğrular. Test cleanup hatası ilk hatayı maskelemez.

Düzeltme /Zs PASS; Player include audit 644/644 temiz; Python syntax PASS.
Yeni binary/canlı test henüz yok. **Bu düzeltmeyi denemek için yalnız Player'ı
kapatıp Player artımlı Build/link ve yeniden açılış yeterli.** Studio'nun çalışan
viewer host'u olmadığı için bu kabul için Studio rebuild gerekmiyor. Sürücü
pipeline hazırlığının ana thread'i bloke etmesi ayrıca açık performans/UX borcu;
bu düzeltme onu asenkron hale getirdiğini iddia etmez.


Durum: kaynak ve sözdizimi kontrolleri tamamlandı; yeni exe linki ve canlı görüntü
henüz doğrulanmadı. Ara derleme istemeden AppFrame, konsolsuz Release ve statik
sahne görüntüleyici aynı kabul paketinde toplandı.

## Bağlanan yol

`project.open` → `rtapi::openProject` → Player'a bağlı `RuntimeSceneViewer` →
ortak `ProjectManager::openProject` → flat `TriangleMesh` / DNA SoA → ortak
`VulkanSceneSync` → mevcut `VulkanViewportBackend` MaterialPreview → SDL texture.
Player'ın sürükle-bırak ve başlangıç dosyası da aynı `rtapi::openProject` çağrısını
kullanır. Studio'nun mevcut proje açma finalizasyonu korunur; host bağlanmadığında
aynı eski yol çalışır. UTF-8 dosya yolu denetimi ortaklaştırıldı.

Studio dünya/HDRI/atmosfer senkronunun gövdesi `VulkanSceneSync.cpp`'ye taşındı;
raster geometri + skin başlangıç hazırlığı da iki hedefte aynı işlevi kullanır.
Yeni bir shader, shader ABI veya registry kaydı yoktur.

Bu ilk dilim **statik** flat mesh, materyal, kamera, ışık ve çevre görüntüsüdür.
Animasyon/simülasyon tick'i henüz Player'da çalışmaz. Scatter, VDB/gas, gerçek su
yüzeyleri, parçacık sistemleri, saç ve fracture reçeteleri destekleniyor sayılmaz.
Bilinen metadata sistemleri yükleme öncesinde açık hatayla reddedilir; parser
sonrasında görülen uyumsuz geometri/runtime da hata üretir. Boş su ayarları
(`water_surfaces: []`) geçerlidir. Per-face Triangle facade kaynağına düşülmez.

Metadata reddi mevcut sahneyi korur. Yükleyici başladıktan sonra oluşan hata
transactional geri alma sağlamaz: scene_ready=false ve açık hata döner; eski
sahnenin render edildiği iddia edilmez. Device loss için otomatik kurtarma bu
ilk dilimde yoktur; açık hata ve Player'ı yeniden başlatma gerekir.

## Ortak script ve IPC yüzeyi

| İşlem | Script | IPC |
|---|---|---|
| Yükleme | `rt.project.open(path)` | `project.open {path}` |
| Durum | `rt.agent.state_summary(False)["runtime"]` | `agent.get_state_summary {include_probe:false}` sonucunda `runtime` |
| Kamera | `rt.camera.orbit(...)` | `camera.orbit {yaw,pitch}` |
| Görüntü yakalama | mevcut `rt.viewport` capture/probe işlemleri | `viewport.capture`, `viewport.get_screenshot` |

Script agent binding zaten aynı `dispatchAgentMethod` hizmetini çağırır; yeni
Python veya IPC iş mantığı yazılmadı. Player'ın içine Python bağlanmadı; Player
üzerindeki otomasyon dış Python sürecinden named-pipe istemcisiyle yürür.

`agent.discover` Player için `runtime_stage=static_scene_viewer` bildirir;
scene_loading/rendering alanları yetenek bildirimi olarak true olur. İzin listesi
14 metoda genişledi; list_methods/describe aynı listeyi görür. Editör işlemleri
ve batch hâlâ `unsupported_in_player` hatası döndürür.

`runtime.scene_ready` yükleme/hazırlık sonucudur; görüntü kanıtı değildir.
`frame_available` ve `frame_scene_revision == scene_revision` mevcut sahneden
ölçülmüş kareyi belirtir. Render çağrıları, GPU ring submit/consume sayaçları,
draw_calls, visible_triangles, ölçüm availability ve device_lost ayrı verilir.
Ring bulunmayan senkron fallback'te ring sayaçlarının 0 olması normaldir.
`pixel_signature` örneklenmiş görüntü imzasının ondalık string temsilidir.

Odak dışı/minimize Player normalde GPU işi göndermez. Açık frame capture isteği,
dış testin odağı çalmadan görüntü alabilmesi için bu kapıyı aşar. TAA/yenileme işi
bitmiş statik sahne yeniden render edilmez. Resize yeni texture/kare ister;
önceki sahnenin veya eski boyutun görüntüsü yeni kare diye yayımlanmaz.

## Tek derleme ve canlı kabul

1. Player ve Studio'yu kapat, iki VS projesini reload et; **Release x64 için
   ikisini bir kez artımlı Build/link et**. Yeni ortak kaynaklar projelere kayıtlı.
   Bu parti yaygın struct/ABI değiştirmedi; ayrıca tam Rebuild istenmiyor.
2. İkisini aç. Player Release'te konsol açılmamalı. Dosya sürüklenebilir veya
   `RayTrophiPlayer.exe --project "E:\scenes\simple.rtp"` kullanılabilir.
3. Depo kökünden dış terminalde çalıştır:

   ```powershell
   python scripts/test/rt_player_scene_ipc.py
   ```

   Varsayılan olarak mevcut `x64/Release/assets/acoustic_echo_lab/EchoRoomsLab.rtp`
   seçilir. Bu dosya yoksa `--project "tam-yol.rtp"` verilir. Örneğin metadata ve
   .bin varlığı kontrol edildi; yeni Player'da render edildiği henüz doğrulanmadı.
4. Betik iki exe'nin IPC kimliğini, gerçek görüntü/probe ve GPU çizim verilerini,
   kamera hareketini, yeniden açmayı ve desteklenmeyen metadata reddinin mevcut
   sahneyi koruduğunu denetler. Yalnız Player'da sahne/kamera değiştirir; Studio'ya
   yazmaz, proje kaydetmez. PNG bellekte çözümlenir; screenshot dosyası oluşturmaz.
   Geçici ret-fixture'ı test sonunda silinir. Toplam timeout varsayılan 240 s.
5. Ayrıca Player resize/minimize/geri dönüş, TAA sonrası boşta tüketim ve normal
   kapanışı kontrol et. Studio'da proje açma, HDRI/sky, resize ve viewport mod
   geçişleri korunmalı. Kapanış sonrası işlemin gerçekten bittiğini doğrula.

Named-pipe sandbox error 5 olursa aynı dış probu escalation ile tekrarla;
embedded script workspace veya başka taşıma kullanma. Test timeout nedeniyle
zorla kesilirse `viewport.capture {enabled:false}` ile capture kapatılabilir.

## Tamamlanan kaynak kontrolleri

- Player /Zs: AppFrame, RuntimeProjectHost, VulkanSceneSync, RuntimeSceneViewer,
  PlayerMain, RtApi, RtIpcAgentDiscovery; Studio /Zs: Main, VulkanSceneSync,
  RuntimeSceneViewer. Gerçek tlog ayarlarıyla syntax-only; obje/exe üretilmedi.
- Player include audit: 644/644 temiz, CUDA/OptiX/Embree/OIDN/Python sınırları 0.
- Player proje generatoru ve IPC descriptor --check PASS; IPC capability audit PASS.
- Include audit birim testleri 2/2; Python acceptance syntax/help ve whitespace PASS.

Tam tek kare/sim döngüsü hedefi kapanmadı: Studio sim/animasyon tick'i ve diğer
runtime sistemlerinin finalizasyonu sonraki dilimlerdir. Bu paket ilk gerçek
statik yükleme → Vulkan raster → görüntü → IPC ölçüm zincirini hedefler.
