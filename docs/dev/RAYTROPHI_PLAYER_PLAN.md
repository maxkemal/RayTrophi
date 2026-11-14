# RayTrophi Player — oyun çalışma zamanı ayrımı

> **Durum:** AKTİF — 2026-11-13: Studio ile aynı kaynak ağacından, OptiX/CUDA/Embree/OIDN/Python içermeyen ikinci bir exe (`RayTrophiPlayer`). Faz 1k'ye kadar başlık/backend ayrımı yazıldı; CUDA/OptiX, Embree, OIDN ve Python include sınırları sıfırlandı; eski su sistemi Player kapsamından çıkarıldı; kullanıcı tam Studio build/link kontrolünü sorunsuz tamamladı; kısmi canlı smoke geçti; Faz 2a Studio launch ayrımı ve Faz 2b ortak pencere/SDL yaşam döngüsü (`AppRuntime`) derlendi, Studio açılışı PASS; Faz 3a–3d: Player VS projesi (üretilmiş), ortak runtime state, Host'lar ve SceneUI durum host'u ile **ilk Player linki GEÇTİ** (2026-11-13; exe yasaklı DLL'lere bağlı değil, pencere + HUD açılıyor). Statik .rtp yükleme ve ortak Vulkan raster sunumu kaynakta hazır; yeni link/canlı kabul bekler. Tam ortak sim/animasyon kare döngüsü hedefi açık.

## Devir notu (her partide güncellenir)

### İlk oyun hedefi — Boyama Odası (2026-11-14, Claude)

Kullanıcı ilk Player projesini belirledi: boyama odası, uçan balonlar, genel kamera
(özel donanım yok) ile çocukların konumu/hareketi ve maskeli görüntüsü, patlayan
balondan renkli boya partikülleri, çarpışmalar GPU partikülden, boya mevcut ıslak
boya modelinden (GPU'ya ve çok yüzeye taşınarak). Tasarım ve sıra:
[PLAYER_GAME_PAINT_ROOM.md](PLAYER_GAME_PAINT_ROOM.md) (TASLAK). Kod yazılmadı.
Sıra yine buradaki 0 (ilk kare kabulü) ve 1 (ortak kare döngüsü) ile başlar.
Hedef pazar: AVM'lerde 6–10 yaş kapalı etkileşimli oyun alanları (çok çocuk, oturum,
gözetimsiz kiosk, yansıtma/ekran). **2026-11-14 akşam: ilk kare canlı PASS**
(rt_player_scene_ipc.py --player-only; device-loss yok).

**Adım 1a — ortak simülasyon saati, kaynak hazır (2026-11-14, Claude):** sim tick'i
panelin çizim fonksiyonundaydı (ImGui DeltaTime/IsAnyItemActive okuyordu), Player'da
hiç koşmuyordu. Gövde `Core/RuntimeSimulationClock` (tek süreç saati) içine taşındı;
SceneUI UI gerçeklerini verip sonucu uygular (playhead, toast). Main'in raster deform
refit dalı `VulkanSceneSync::refitRasterDeformedNodes` oldu. Player `advanceRuntime`
ile saati canlı modda sürer ve raster write-back'i (refit / transform / tam rebuild)
servis eder; `runtime.simulation` IPC bloğu sayaçları verir. Player hâlâ partikül
sahnelerini reddeder (render köprüsü Codex dosyalarında; adım 2 ile). Animasyon
(karakter/anim graf) tick'i Main'de kaldı → adım 1b. /Zs 8/8, include audit PASS;
build + canlı kabul (`rt_player_sim_clock_ipc.py`) bekliyor.
2026-11-14 ilk canlı tur: Studio ortak saatle PASS (kutu düşüp duruyor, scrub doğru),
EchoRoomsLab regresyonu PASS. Player FAIL → kök: `playing` timeline oynatımı demek,
serbest akış yalnız `live_mode && !playing`; Player `playing=false` ile düzeltildi.
**Adım 1a CANLI PASS (19:28 Player):** düşen kutu 2.96 → 0.25 m, 230 raster refit / 1
rebuild (yükleme).

**Reddedilen yapıları kapatma sırası (kullanıcı onayı 2026-11-14):** materyal → partikül
sistemi → scatter/instance → volume (mevcut `syncVolumetricData`) → saç/su/fracture
ihtiyaç olunca. İlke: Studio'nun çift yönlü mevcut çağrıları kullanılır, Studio'ya özgü
maliyet (iki aygıt, OptiX/CPU dalları) Player'a taşınmaz.
**Materyal:** canlı ölçüm statik yolun zaten tam olduğunu gösterdi (5 materyal + doku,
Studio ile aynı görüntü). Kapatılan iki boşluk: editör ızgarası (`grid_opacity=0`) ve
yükleme sonrası toplu materyal/ışık kirliliği servisi; Player IPC'ye `material.*`
değerleri. **CANLI PASS (19:41 Player):** ızgara yok, `material.set` görüntüye ulaşıyor,
sim saati + EchoRoomsLab regresyonu PASS.
**Partikül kabulü — kaynak hazır (2026-11-14):** red kaldırıldı; ortak saat + Studio'nun
billboard yolu (`AppRuntime::uploadParticleBillboards`, SceneUI artık bunu çağırır).
`UI/ParticleBillboardBuilder.cpp` UI'sızdı, manifestle Player'a dahil (Codex dosyası,
değişmedi; iş bölümü panoda). RT partikül örnekleri (`syncParticleRenderInstances`) ve
mesh partikülleri Player'da yok (raster billboard/sphere/pulled yeter).
**CANLI PASS (19:51):** 1183 partikül, tek pulled çizim (GPU-resident), görüntü doğru. IPC tarif borcu: `particle.set_appearance`
`profile_id` tarifte float, dispatch int istiyor.
**Scatter/instance — kaynak hazır (2026-11-14):** Vulkan'da scatter native InstanceGroup
olarak kalır, raster build onu `InstanceManager`'dan doğrudan okur; Player'ın engeli
yalnız iki ret satırıydı (kaldırıldı). Çalışma sırasında scatter değişimi
`refreshRasterScatterInstances` ile. Rüzgâr animasyonu Player'da yok (animasyon adımı).
**CANLI PASS (20:11):** 150 instance, 6 draw call, görüntü doğru; regresyon 4/4.
**Volume (VDB/gaz) — kaynak hazır (2026-11-14):** önceki yön uygulandı: SceneUI funnel'ı
(`syncVDBVolumesToGPU`) `AppRuntime::publishVolumes` oldu, Player'daki saplama aynı
fonksiyonu tek aygıtla çağırıyor; yeni volume sistemi yazılmadı. Yanında kural-1 açığı:
VDB içe aktarma yalnız paneldeydi → `rtapi::importVdbVolume` + IPC `scene.import_vdb` +
Python + yetki + tarif. Animasyonlu VDB dizisi Player'da yok.
**CANLI PASS (20:33):** VDB Player'da görünüyor; regresyon 6/6. Kararlaştırılan ret
kapatma sırası (materyal → partikül → scatter → volume) TAMAMLANDI; kalan retler
saç / su / fracture (ihtiyaç olunca).

### Kullanıcı yönü ve sonraki oturum başlangıcı — 2026-11-14

**Player bir dosya gösterici değildir: oyun motorunun çalışma zamanı/yürütücüsüdür.**
`.rtp` açıp görüntülemek yalnız ilk uçtan uca kabul basamağıdır; ürünün hedefi bu
basamağa indirgenmeyecek. Nihai çalışma zamanı sahne yaşam döngüsü, girdi, oyun
mantığı/scripting yüzeyleri, animasyon, fizik/simülasyon, ses ve render yürütümünü
Studio ile aynı çekirdek üzerinden çalıştırmalıdır. Studio authoring kabuğudur;
Player, o çekirdeğin editörsüz yürütücüsüdür.

Kullanıcı yarın buradan devam etmek istiyor; bu oturumda yeni uygulama/derleme işi
istenmiyor. Yeni oturumda önce bu not ve koordinasyon panosu okunacak.

Başlangıç işi: RuntimeSceneViewer içindeki ikinci sahne hazırlama/render sırasını
büyütmek yerine Studio'nun mevcut çalışan akışını ortaklaştır. Mevcut hizmetleri
ara ve kullan; yeni modül gerektiğinde çalışan gövdeyi taşı, iş mantığını yeniden
yazma. ProjectManager::openProject ve mevcut Vulkan backend zaten kullanılıyor;
materyal/kamera/ışık/dünya/volume güncelleme sırası da tek kaynak olmalı.
VDB/gas için çekirdek zaten `VolumetricRenderer::syncVolumetricData`; SceneUI
funnel'ının backend yönlendirmesini ayırıp bu hizmeti bağla. Player'da VDB'yi
kapatıp yeni volume sistemi yazmak hedef değildir. Scatter için de mevcut flat
runtime yolu esas alınacak; Triangle facade koleksiyonuna geri dönülmeyecek.

Son canlı durum: örnek .rtp 26 flat mesh/312 üçgen/3 ışık olarak yüklendi; ilk GPU
kare device-loss ile başarısız, görüntü kabulü geçmedi. Lazy capability yayını,
boş materyal tablosu ve GPU material buffer doğrulaması düzeltmeleri kaynakta;
henüz yeni build/canlı doğrulaması yok. Kullanıcının son yönü nedeniyle hemen
tekrar build isteme: önce ortak runtime mimarisi düzeltmesini anlamlı tek pakette
topla. Soğuk pipeline hazırlığının ana thread'i bloke etmesi açık borçtur.


- **2026-11-14 canlı yükleme PASS / ilk görüntü FAIL; düzeltme kaynak hazır:**
  EchoRoomsLab 26 flat mesh/312 üçgen/3 ışık olarak yüklendi. İlk GPU kareden
  önce NVIDIA sürücü pipeline derlemesi ana thread'i bloke etti; sonra device
  lost (1 submit/0 consume), görüntü yok. Debugger ayrıldı; Player IPC hata durumuyla
  yeniden cevap veriyor. Lazy capability refresh + boş materyal tablosuna default
  ID 0 + flat materyal ID/buffer validasyonu eklendi. GPU kök nedeni kesinleşmedi;
  yeni canlı kabul bekler. Yalnız Player artımlı build; Studio için bu kabulde
  yeniden build istemiyoruz. /Zs ve include audit PASS. Detay kabul belgesinde.


- **İlk statik .rtp → Vulkan viewer, birleşik paket kaynak hazır (Codex):**
  Kullanıcı ara derleme istemedi; AppFrame ve GUI subsystem değişikliği üzerine
  RuntimeProjectHost/RuntimeSceneViewer/VulkanSceneSync bağlandı. Başlangıç dosyası,
  drop ve IPC aynı `rtapi::openProject` → ProjectManager flat parser yolunu kullanır.
  Studio world/HDRI/sky sync ve raster build hazırlığı ortak modüle çıkarıldı.
  Player MaterialPreview render/surface/texture sunumu ve mevcut viewport capture
  hattını kullanır; statik sahne oturunca idle, arka planda yalnız açık capture
  isteği GPU çalıştırır. Script/IPC ortak agent özeti `runtime` ölçümlerini taşır;
  yeni method/descriptor veya Python bağımlılığı eklenmedi. Bilinen desteklenmeyen
  runtime sistemleri açık hata; boş water ayarları kabul edilir. Static slice,
  tam oyun/sim/animasyon döngüsü değildir. **Yeni exe henüz build/link/canlı PASS
  değil.** Bir kez iki hedef artımlı build ardından dış uçtan uca test hazır:
  `python scripts/test/rt_player_scene_ipc.py` (mevcut EchoRoomsLab varsayılanı).
  /Zs kontrolleri, 644/644 include audit ve proje/IPC generator denetimleri PASS.
  [Ayrıntı ve kabul](PLAYER_STATIC_VIEWER_ACCEPTANCE.md).


- **Faz 2c ilk çıkarım — ortak kare I/O kaynak hazır (2026-11-13, Codex):**
  Yeni `Core/AppFrame.h/.cpp` / `AppRuntime::FrameCycle`, olay sınıflandırması,
  yükleme engelli rtapi kuyruğu, ImGui begin/end/draw, SDL present ve Player
  bootstrap pacing/redraw politikasını taşır. İki exe aynı implementasyonu çağırır.
  Studio Main yalnız minimal include/declaration/call wiring aldı; mevcut çağrı
  sırası, loader kapısı ve perf scope'ları korunur. 3 begin/5 end/4 draw+present
  dalı bağlandı. PlayerMain'deki tekrarlı pacing/foreground/redraw gövdesi silindi.
  Projeler kayıtlı; generator PASS, include audit 641/641, XML kaynak kayıtları PASS.
  **Yeni kod henüz build/link/canlı doğrulanmadı; ikisi artımlı build bekler.**
  Tam tek kare döngüsü HEDEFİ HENÜZ KARŞILANMADI: sim/animasyon tick ve Vulkan
  scene sync hâlâ Main/SceneUI'da. Sonraki çıkarım bu servis sınırıdır. Player
  `project.open` hâlâ desteklenmez; sahne yükleniyor/render ediliyor iddiası yok.


- **2026-11-13 iki hedef canlı IPC PASS; konsolsuz Player kaynak hazır:**
  Studio/Player birlikte açıkken dış kabul betiği PASS. Kullanıcı idle düzeltmesi
  sonrası boşta CPU/GPU tüketiminin görünmediğini bildirdi. Player konsolu IPC
  gereği değil, Console subsystem mirasıydı. Generator Release/ReleaseDev'i
  Windows subsystem yapar; Debug Console. Mevcut SDL2main korunur. Yalnız Player
  proje reload + Release Build/link ve konsolsuz açılış kabulü bekler.


- **2026-11-13 Player idle düzeltmesi kaynak hazır:** Kullanıcı 3e Player build
  PASS bildirdi; açık Player dış IPC bootstrap kabulü PASS. Sınırsız, VSync
  olmayan SDL döngüsü boşta da sürekli HUD çizip sunuyordu. PlayerMain artık
  statik HUD'ı yalnız olay/durum değişiminde çiziyor; odak dışı sunum yok;
  odakta 60 Hz, arka planda 20 Hz uyuyan IPC/olay pompası var. Yalnız Player
  artımlı build ve performans/canlı kapanış kabulü bekliyor. Studio değişmedi.


- **Faz 3e kaynak hazır — Player IPC sınırı (2026-11-13, Codex):**
  `RT_PLAYER` yalnız üretilen Player proje ayarına eklendi. Ortak
  `RtPlayerIpcPolicy.h` pipe kimliğini ve izin verilen metotları tek yerden belirler;
  transport oluşturma ve kapanış uyandırması aynı hedef pipe'ını kullanır.
  Studio `\\.\pipe\RayTrophiStudio`, Player `\\.\pipe\RayTrophiPlayer`.
  Player bootstrap aşamasında yalnız `agent.discover`, `agent.describe`,
  `agent.list_methods`, `agent.search_capabilities` açıktır. Diğer tüm IPC çağrıları
  (batch dahil), handler/parametre erişiminden önce `unsupported_in_player: <method>`
  hatası döndürür. Bu geçici sınır ortak rtapi servislerini silmez; gerçek Player
  servisi bağlandıkça liste genişletilmeli. Python Player'da halen yoktur.
  Registry aynı politikayla süzülür; arama Studio workflow'larını önermez;
  discovery `runtime_stage=bootstrap`, `scene_loading=false`, `rendering=false`
  bildirir. Kapanış istendiğinde Player artık son bir IPC/render karesi çalıştırmaz;
  mevcut stop → unbind → state destroy sırası ve bekleyen istek iptali korunur.
  **C++ build/link ve canlı IPC/kapanış testi yapılmadı.** Generator `--check`,
  include audit 640/640 ve Python kontrolleri PASS; canlı kabul betiği
  `scripts/test/rt_player_bootstrap_ipc.py`. Bu parti ortak kare döngüsü veya
  `.rtp`/Vulkan sunumu içermez. Sonraki sıra: bu sınırın kullanıcı build/canlı
  kabulü, ortak kare çekirdeği, basit mesh/materyal/kamera/ışık sahnesinin `.rtp`
  yüklenip Vulkan'da gösterilmesi; scatter ve VDB sonraki kabul sahneleridir.


- **★ KİLOMETRE TAŞI — İLK PLAYER LİNKİ GEÇTİ (2026-11-13 22:47, kullanıcı build'i):**
  Faz 3d sonrası link 39 → **0**; `x64\Release\RayTrophiPlayer.exe` üretildi, açıldı,
  beklenen HUD göründü (kullanıcı). Ajan doğrulaması: link logunda 0 LNK hatası;
  `dumpbin /dependents` çıktısında `cudart`/`cufft`/`nvcuda`/`OpenImageDenoise`/
  `embree4`/`python311` **YOK**. Bağlı DLL'ler: libssl/libcrypto, openvdb, Imath,
  SDL2, SDL2_image, tbb12, vulkan-1 (delay-load), CRT/OpenMP. Boyut 23.8 MB (Studio
  34.3 MB). Link yolculuğu: 364 → 168 → 39 → 0, dört tur.
  - **Doğrulanmayan:** Player IPC cevabı (`agent.discover`), editör-bağımlı metotların
    Player'da HATA döndüğü (başarı taklidi yok), temiz kapanış, Studio'nun bu
    partilerden sonra tam regresyonu (yalnız açılış + HDR + çözünürlük + Türkçe giriş
    + docking doğrulandı).
  - **Faz 3d anındaki öneri (3e kaynak durumu yukarıda):** (1) Player'a ayrı IPC pipe adı — şu an Studio ile
    aynı ad, ikisi birlikte açılamıyor ve Player'ın QA yüzeyi ancak Studio kapalıyken
    test edilebiliyor. (2) Player'ın ilk gerçek işi: `.rtp` aç + Vulkan/RayFusion ile
    göster (Faz 3 hedefi) — bunun için scatter→sahne ve VDB→GPU senkronunun SceneUI
    static'lerinden çekirdeğe taşınması gerekir. (3) Faz 2'nin kalan parçası: kare
    döngüsü/sim tick'in Main.cpp'den `AppRuntime`'a ayrılması (Player şu an kendi
    küçük döngüsünü koşuyor; "tek kare döngüsü" hedefi henüz karşılanmadı).

- **Faz 3c ÖLÇÜLDÜ + Faz 3d YAZILDI (2026-11-13, Claude):** Player linki **168 → 39**,
  derleme 0 hata; yeni alınan MeshEdit/Paint/UV/Template/Import dosyaları ek bağımlılık
  getirmedi. Kalan 39'un tamamı ortak kodun Studio editörüne (SceneUI) uzandığı
  yerler + Embree 4.
  - **★ Bulgu (ölçüldü):** 18 ortak API dosyası `ui` (SceneUI) NESNESİNE ~170 yerde
    erişiyor; çoğu veri: `timeline` 19, `mesh_cache_valid` 23, `viewport_settings` 5,
    panel görünürlük bayrakları (RtApiUiState), `editable_mesh_cache` 14, fracture
    haritaları 8. Linker `ui`'yi yalnız ilk obj'de raporluyor; sayı aldatıcıydı.
    Bu, CLAUDE.md §1'in "UI durum tutmaz" borcunun somut listesidir.
  - **Çözüm (Studio koduna dokunmadan):** `UI/SceneUIStateHost.cpp` — Player kendi
    `SceneUI ui` nesnesini **durum taşıyıcı** olarak tanımlar (hiç çizilmez;
    `UIContext::scene_ui_ptr = &ui`, undo `ui.history`, Studio ile aynı bağlama).
    Mantığı editörde yaşayan 22 SceneUI üyesi + `enterSolidViewportForSceneLoad` +
    `UvEditorUI::open` + `TemplateHubUI` (boş hub) **açık ret** olarak tanımlı: bool
    olanlar false döner (çağıran API hataya çevirir), void olanlar işlem adı başına
    bir kez loglar; fracture kaydı olan proje "UNFRACTURED yüklendi" uyarısı verir.
    Hiçbiri başarı taklidi yapmaz. Deneme derlemesiyle ölçüldü: bu nesnenin
    kurulması yalnız PaintModeState, AgentChatPanel, Vec3, GraphBase ister.
  - `Render/EmbreeBVHHost.cpp`: boş BVH (hit/occluded false, tek uyarı). PlayerMain
    `use_embree=false` kurar → Renderer Embree'siz `ParallelBVHNode` kurar, CPU ışın
    sorguları gerçek kalır.
  - Manifest `+` (SceneUI'a bağlı olmayan, yalnız ImGui): `ui_modern.cpp`,
    `ui_section_style.cpp`, `scene_ui_agent_chat.cpp`, `SimulationTransport.cpp`
    (içinde `TimelineWidget::setPlayback`), `GizmoOcclusion.cpp`;
    `SplineSurfaceAuthoring.cpp` hariç listesinden çıktı (yalnız ImGui kullanıyor).
  - PlayerMain artık ortak `color_processor`/`active_model_path` globallerini kullanır.
  - Doğrulama: `/Zs` 4/4 + 7 deneme derlemesi (obj üretip çözülmemiş dışları ölçtüm,
    sonra sildim); include audit **640/640**, 0/0/0/0; Player **811 kaynak (632
    birinci taraf)**, `--check` PASS. Studio'da iki None kaydı (derleme gerekmez).
  - **Beklenen:** 39 sembolün hepsinin karşılığı var → link GEÇEBİLİR ya da ikinci
    katman (yeni alınan beş UI dosyasının istedikleri) birkaç sembol çıkarır.
  - **★ Açık borç listesi (Player'da çalışmayan, mantığı editörde olan):** mesh edit
    operatörleri, fracture (kes/geri al/tarif tekrarı), duplicate, geometry graph
    evaluate + processAnimations, **scatter → sahne senkronu**, **VDB hacim GPU
    senkronu** (son ikisi Player'ın render edeceği şeyler: Faz 4'ten önce çekirdeğe
    taşınmalı), UV editör penceresi, son projeler listesi.

- **Faz 3b ÖLÇÜLDÜ + Faz 3c YAZILDI (2026-11-13, Claude):** kullanıcı Studio'yu
  derledi (HDR/EXR sorunsuz); Player linki **364 → 168** (Vulkan, globaller,
  stb/tinyexr, konsol grupları sıfırlandı).
  - **Karar (kullanıcı):** UI içermeyen MeshEdit/Paint/UV/Template/Import kodu
    Player'a girer; yalnız editör UI'sı Studio'da kalır (§1 tablosu güncellendi).
    Manifest: `+src/MeshEdit/* +src/Paint/* +src/UV/* +src/Template/* +src/Import/*
    +src/UI/SceneSelection.cpp`; hariç: ProfileSplineEditor/Overlay/PointGizmo,
    GeometryCurveNodes, SplineSurfaceAuthoring, TemplateSession, UserTemplateManager,
    TemplateUiStateAdapter, RtApiTemplates, RtIpcTemplates.
  - Yeni Host'lar: `WaterSystemHost.cpp` (WaterManager/RiverManager'ın ortak koddan
    çağrılan 14 üyesi; "legacy su yok": lookup nullptr, create reddeder, serialize
    boş liste, deserialize kayıtları uyarıyla yok sayar — pişmiş mesh'ler yüklenir),
    `StudioOnlyIpcHost.cpp` (`dispatchTemplateIpc` → false; istek mevcut
    bilinmeyen-metot hatasına düşer). `PythonUnavailableHost.cpp`'ye dört addon
    fonksiyonu eklendi (RtIpc.obj).
  - Doğrulama: `/Zs` 7/7 PASS (üç Host + SceneSelection, FlatMeshEditService,
    UvXatlasBackend, ModelImport Player görünümünde). Include audit **632/632**,
    0/0/0/0. Player projesi **803 kaynak (624 birinci taraf)**, `--check` PASS.
    Studio'da yalnız iki None kaydı (derleme gerekmez, reload yeter).
  - **Beklenen kalan (3d):** SceneUI'a bağlı API çağrıları (~32 + `ui` globali +
    `enterSolidViewportForSceneLoad`, AgentChatPanel 4, UvEditorUI::open,
    UIWidgets::HelpMarker, TimelineWidget::setPlayback,
    SimulationTransport::setTimelineMode, GizmoOcclusion::stats,
    TemplateHubUI::addRecentProject) ve Embree 4. Yeni alınan `scene_ui.h` içeren
    MeshEdit/UV servisleri (FlatMeshEditService, FlatMeshGeometryCommand,
    ProfileAuthoringService, SplineObjectLifecycle/Service, UvAttributeUpdate,
    UvGeometryTransaction) ek SceneUI sembolü isteyebilir. Tasarım önerisi:
    ortak başlıkta soyut **EditorBridge** (Studio SceneUI'a iletir, Player'da null →
    metot açık hatayla reddeder); `ui` globali yerine `UIContext::scene_ui_ptr`.
    Bu, "panel çekirdek mantığı tutuyor" borcunun (CLAUDE.md §1) görünür listesidir.

- **Faz 3a ilk Player linki ÖLÇÜLDÜ (2026-11-13, kullanıcı build'i):** derleme 0 hata
  (IntDir `RayTrophiStudio\x64\Release\Player\`, 589 obj — Studio obj'leriyle karışmıyor;
  hata listesindeki `RayTrophiStudio\X.obj` yolu VS gösterimi). Link **364 çözülmemiş
  sembol**, gruplar: Vulkan backend parçaları 109 (`src/Viewport/*` manifest dışı) ·
  MeshEdit 56 · Main/scene_ui globalleri 48 · SceneUI/editör UI 32 · Paint 28 ·
  stb/tinyexr/TextEditor 26 (impl. Main.cpp'de; TextEditor = RtConsole) · Template 18 ·
  Python 16 (RtConsole 14, RtApi 2) · Su/nehir 14 · Seçim 8 · Embree 4 · OIDN 2 ·
  CUDA sim 1 · import 3.
  - **Faz 3b YAZILDI (2026-11-13, Claude), derleme bekler:**
    - `Core/RuntimeState.cpp`: 65 globalin TEK tanımı (Main.cpp'deki dirty/deferred-
      rebuild/seq-save/yükleme bayrakları, `g_backend`, `g_viewport_backend`,
      `color_processor`, `active_model_path`, `stylize_redisplay`, `markWorldDirty`;
      scene_ui.cpp'den `g_scene_load_solid_guard` + iki RAM sorgusu;
      scene_ui_modifiers.cpp'den `g_dense_mesh_as_hittable`; `g_uiClipsRef` —
      scene_ui_animgraph.hpp'de `inline` tanımdan `extern` bildirime döndü).
      Main.cpp yalnız globals.h'nin bildirmediği 18 adı `extern` ile görür. Çift
      tanım taraması temiz (scene_ui.cpp'deki `static active_model_path` dosya içi).
    - `Utils/ImageCodecs.cpp`: stb_image/stb_image_write/tinyexr implementasyonları
      Main.cpp'den taşındı (aynı makrolar); başka hiçbir TU `*_IMPLEMENTATION` tanımlamaz.
    - Host'lar (Studio'da None, Player'da derlenir): `RendererOidnHost.cpp` (boş
      OidnState), `SimulationComputeCudaHost.cpp` (nullptr → çağıran Vulkan'a düşer),
      `PythonUnavailableHost.cpp` (`isInitialized=false`, `executeFile` açık hata).
    - Manifest: `+src/Viewport/*`, `-src/Api/RtConsole.cpp` (Python konsolu/TextEditor).
    - Doğrulama: `/Zs` 14/14 PASS (Main, RuntimeState ×2, ImageCodecs ×2, scene_ui,
      scene_ui_modifiers, scene_ui_hierarchy, ProjectManager ×2, AnimationNodes, üç
      Host); Player dökümlerinde SDK yok. Include audit **583/583**, 0/0/0/0. Player
      projesi **755 kaynak (576 birinci taraf)**, üretici `--check` PASS, XML PASS.
    - Beklenen Player link etkisi: Viewport ~109, globaller ~48, stb/tinyexr 11,
      TextEditor+Python konsolu ~29, OIDN/CUDA/Python host 5 → ~200 sembol kapanmalı;
      RtConsole'un çıkması kendi çağıranlarını yeni hata olarak gösterebilir.
    ★ Kullanıcı derlemesi sürerken kaynak düzenlenmez: 2026-11-13 12:40'ta bu parti
    çalışan Studio rebuild'ine denk geldi, UI dosyalarındaki taşıma o an geri alındı,
    derleme bittikten sonra yeniden uygulandı.
  - **Karar isteyen gruplar (3c):** MeshEdit/Paint/Template/SceneUI çağrıları API
    dosyalarından geliyor (RtApiMeshEdit, RtApiPaint, RtApiSpline, RtIpcMeshTools,
    RtApi*Templates, RtIpcAgentDiscovery, RtUvBindings): Player manifestinden çıkarmak
    IPC kayıt tablosunun yapısına bağlı. Su/nehir: ProjectManager/TerrainNodesV2/
    RtApiAtmosphere/RtApiTerrain/Renderer çağrıları servis ayrımı ister. Embree:
    Renderer.cpp CPU BVH kuruyor — `Renderer`'ın Player'daki rolü kararı.

- **Faz 3a, Player VS projesi iskeleti (2026-11-13, Claude):**
  `RayTrophiStudio/RayTrophiPlayer.vcxproj` (+ `.filters`) **üretilmiş dosyadır**:
  `python scripts/gen_player_vcxproj.py` Studio vcxproj'unu (ayarlar, dosya başı
  metadata) ve `player_include_manifest.json`'u okur; `--check` bayatlığı FAIL eder.
  Player = Studio'nun derlediği üçüncü taraf (Jolt/ImGui/ozz/miniz/xatlas, 179)
  + manifest içindeki Studio ClCompile birinci taraf kaynakları + manifestin yeni
  `player_link_alternatives` listesi (Studio'da None: 10 Host adaptörü,
  `TextureHost.cpp`, `PlayerMain.cpp`) = **723 kaynak, 544 birinci taraf**.
  Ayar dönüşümü: CUDA/OptiX/OIDN/Embree/Python include/lib/delay-load, CUDA build
  customization ve gömülü Python kopyası çıkarılır; `RT_NO_CUDA` tanımlanır; IntDir
  `x64\<Cfg>\Player\` (obj paylaşılmaz), OutDir Studio ile aynı (shader/DLL ortak).
  Kalan yasak token kalırsa üretici durur. Kök `.sln`'de **yalnız ActiveCfg**:
  "Build Solution" Studio'yu derler, Player sağ tık → Build ile ayrıca derlenir.
  - `PlayerMain.cpp`: AppRuntime pencere + ImGui HUD (FPS, IPC durumu), kendi
    SceneData/Renderer/ColorProcessor/SceneSelection/SceneHistory'si ile UIContext,
    `rtapi::bind` + `rtipc::start`, döngüde `pollEvent` + `drainMainThreadQueue`;
    kapanış Studio sırasıyla. **Sahne yükleme ve Vulkan/RayFusion sunumu YOK.**
    Player nesneleri bilinçli olarak Studio global adlarını (`scene`, `ui`,
    `ray_renderer`) taşımaz: ortak kodun hâlâ uzandığı Studio globalleri link
    hatası olarak görünmeli, burada gölgelenmemeli.
  - `TextureHost.cpp`: dört `Texture::cuda*` üyesinin no-op karşılığı (plan maddesi).
  - Manifest include'una `src/simdjson.cpp` eklendi (üçüncü taraf, IPC JSON).
  - Doğrulama: `/Zs` 5/5 PASS (Main, AppWindow ×2, PlayerMain, TextureHost); Player
    dökümlerinde SDK başlığı yok. Include audit **551/551**, 0/0/0/0. Üretici
    `--check` PASS, dört proje dosyası XML parse PASS. **Player hiç derlenmedi/
    linklenmedi.** Beklenen: ilk Player linki çok sayıda LNK2001/LNK2019 verir —
    ortak kodun Main.cpp/UI/Viewport/MeshEdit/Import globallerine ve sembollerine
    uzandığı yerlerin **haritası budur**; hatalar kümelenip sırayla ayrılacak.
  - Disk/proje farkı: manifest diskte 551 aday sayar, Studio projesi bunların
    7'sini derlemiyor (RigFingerFK, RtApiImport, RtIpcImport, FoamSurface,
    OptixAccelManager_MaterialUpdate, TerrainSatMapNodes, AssimpLoader); Player
    projesi de almaz. Yetim mi yarım iş mi, ayrıca bakılmalı.
  - IPC pipe adı sabit (`\\.\pipe\RayTrophiStudio`): Studio ve Player aynı anda
    açıksa ikincisi IPC başlatamaz (log'da uyarı). Player'a ayrı pipe adı sonraki iş.

- **Faz 2c, olay pompası + sunum hedefi yeniden kurulumu (2026-11-13, Claude):**
  `AppRuntime::pollEvent(e, EventInfo)` = SDL_PollEvent + ImGui SDL2 iletimi +
  sınıflama (`quit_requested`, `focus_gained`); her olay (IPC'nin SDL_PushEvent
  uyandırması dahil) kabuğa geri döner. Studio döngüsü quit'i hâlâ `ui.tryExit()`
  (kaydet sorusu) ile karşılar; kısayollar Main'de. `reset_render_resolution`'ın
  yüzey/doku yok et–kur–siyah sun adımları `AppRuntime::recreatePresentationTargets`
  oldu; aspect/renderer/backend yeniden boyutlandırması Main'de kaldı. Hata yolunda
  eskisi gibi erken dönülür, `g_original_surface_needs_sync` yalnız başarıda kurulur.
  Arka plan `SDL_WaitEventTimeout` beklemeleri Main'de (politika Viewport'ta).
  - `/Zs` 3/3 PASS, Player dökümü SDK'sız, include audit 548/548 (0/0/0/0).
    Yeni dosya yok (proje kaydı değişmedi). **Kullanıcı (2026-11-13): build PASS,
    viewport çözünürlük değişimi sorunsuz.** IPC uyanması/kapatma sorusu ayrıca
    raporlanmadı.
  - Sim tick ayrımı bu partide YAPILMADI: döngü gövdesi Studio durumuyla (timeline,
    anim graph, editör bayrakları) iç içe; ayrı parti olarak sırada.
- **Faz 2b kullanıcı kabulü (2026-11-13):** Türkçe klavye girişi ve panel
  docking/ayırma sorunsuz (kontrol 3–5 PASS).

- **Faz 2b, ortak pencere/SDL yaşam döngüsü (2026-11-13, Claude):** Yeni
  `Core/AppWindow.h/.cpp` (`AppRuntime` namespace) SDL platformunu, ana pencereyi,
  SDL renderer'ı, iki RGBA32 CPU yüzeyini, sunum dokusunu ve ImGui bağlamı +
  SDL2/SDLRenderer2 backend'lerini kurar ve aynı sırayla söker:
  `initPlatform` → `createMainWindow(WindowConfig)` → `initImGui(ImGuiConfig)` …
  `destroyMainWindowAndImGui` → `shutdownPlatform`. Beş pencere globali
  (`window/renderer/surface/original_surface/raytrace_texture`) artık bu modülde
  **tanımlı**; Main yalnız `extern` görür (tek tanım grep PASS). Studio'ya ait olan
  Main'de kaldı ve modülün **etrafında** çağrılır: splash, OptiX donanım tespiti,
  pencere başlığı/Segoe UI/docking seçimi, tema, Python/rtapi/OptiX/viewport backend
  kapanışı, terrain node önizleme dokusu (renderer sökülmeden önce bırakılır).
  - Davranış farkı tek: IME hint'i artık splash "Creating main window..." çiziminden
    sonra, pencere oluşturmadan hemen önce kuruluyor (yalnız pencere oluşturmayı
    etkiler). Başlangıç breadcrumb metinleri aynı; kapanışta handle'lar sıfırlanır.
  - `/Zs` **3/3 PASS** (Main + AppWindow Studio, AppWindow Player `RT_NO_CUDA`);
    Player include dökümü yalnız SDL/ImGui/globals, yasaklı SDK yok. Proje/filter
    tekil kayıt + XML parse PASS. Include audit **548/548** (yeni modül Player
    adayında), 0/0/0/0. Shader/struct/IPC/script değişmez. **Kullanıcı: Faz 2a+2b
    build/link PASS, Studio sorunsuz açıldı (2026-11-13).** Font/IME/kapanış/CLI
    maddeleri ayrıca doğrulanmadı.
  - Sıradaki: kare döngüsünün olay pompası (SDL_PollEvent/resize/quit) ve sim
    tick'i; ardından Python/offline capability sınırı. `reset_render_resolution`
    yüzey/doku yeniden kurulumu henüz Main'de — döngü ayrımında AppWindow'a taşınmalı.

- **Faz 2a, Studio launch policy (2026-11-13, Codex):** Main.cpp içindeki
  proje/script/offline render/sequence/headless argüman çözümleme yeni
  `Core/StudioLaunchOptions.h/.cpp` modülüne taşındı. StudioShell namespace,
  Studio projesinde tek kayıt ve Player manifestinde açık dışlama kullanılır.
  Başlangıç çalışma dizini değişmeden önce parse çağrısı aynı yerde kalır;
  hata metinleri, varsayılanlar ve mevcut seçenek öncelikleri korunur.
  Parser gövdesi kaynak eşitliği ve proje/filter tekil kayıtları PASS;
  Main + yeni kaynak `/Zs` 2/2 PASS, include audit 547/547 ve 0/0/0/0.
  Shader/struct/IPC/script değişmez. **Bu yeni parti build/link bekler.**
  Bu ilk parça AppRuntime/kare döngüsü ayrımını tamamlamaz. Sonraki parça:
  pencere/SDL yaşam döngüsü ve Studio UI başlangıç/kapanış hizmetleri;
  ardından kare döngüsü/sim tick ve Python/offline capability sınırı.

- **Canlı smoke devri (2026-11-13):** Kullanıcının tam derlediği Faz 1k exe'sinde
  üç backend render/geçiş, 17 Python çağrısı, IPC probları, proje roundtrip ve
  Vulkan OIDN GPU 17/17 geçti. Kullanıcı kalan canlı testleri devraldı;
  render-target resize/temiz kapanış ve kapsamlı sayısal kabul tamamlanmadı.
  Ayrıntı: [canlı rapor](PLAYER_LIVE_SMOKE_2026_11_13.md).


- **Kullanıcı kabul bildirimi (2026-11-13):** Tam Studio derlemesi/link sorunsuz
  tamamlandı. Bu, Faz 1k'ye kadar biriken Studio kaynak/proje bağlantılarının build
  kontrol noktasıdır; önceki parti notlarındaki yapılmamış Studio build durumunu
  günceller. Kısmi canlı smoke üstteki raporda; sayısal karşılaştırma ve Player hedef/link
  kabulü bekler. Kalan canlı kontroller kullanıcıda; liste NEXT_BUILD_CHECKS.md başında.

- **Faz 1k, Python bind / ortak IPC ayrımı (2026-11-13, Codex):**
  On karma kaynakta Python fonksiyonları yeni `RtPython*Bindings.cpp` dosyalarına
  taşındı; servis/IPC/JSON/descriptor kodu ortak kaynakta kaldı. Lens/object/weather
  başlıkları pybind include yerine ileri bildirim taşır; RtIpc.cpp sızıntısı kalktı.
  UV işlem listesi, lens invoke ve weather hata dönüşümü iki transport için ortak.
  Beş saf Python sahibi (addon loader, addon UI, PyScene, realtime-denoiser/upscale
  Python bağlamaları) Player manifestinden dışlandı; genel ses/ImGui/SVGF/FSR kalır.
  - Güncel **547 aday / 547 temiz**: CUDA/OptiX **0**, Embree **0**, OIDN **0**,
    Python **0**; dört baseline sıfır. -5 aday saf Python sınıflandırması; on karma
    dosya + transitif RtIpc temizliği +11 temizdir. Studio yeni on modülü derler;
    Player mevcut RtPython* dışlama kuralıyla bunları almaz.
  - `/Zs` **34/34 PASS**; **11 Player** dökümünde Python/pybind ve diğer yasaklı
    SDK yok. **32** binding/IPC/servis/helper/sözlük kaynak eşitliği PASS. XML tek
    kayıt/include audit/diff-check PASS. IPC capability audit PASS: 757 dispatch,
    743 descriptor güncel (724 belgeli/583 parametreli); descriptor çıktısı değişmedi.
  - Dokuz sözleşme suite'i **86/89**. Üç hata parti öncesi API snapshot'ıyla da
    aynı: timing w/h beklentisi, gas cache v18 beklentisi, emission alanlarını eski
    params.h'de arama. Ayrı post-v2 JS guard'ı dokunulmayan OptixWrapper'daki eski
    launchOptixDisplayPost beklentisinde FAIL. Başarılı sayılmadılar; üretim kodu
    bu eski beklentilere uydurulmadı. Üç Python taşıma source testi yeni dosya ve
    boşluk toleransına uyarlandı; servis/metot assertion'ları korundu.
  - Shader/struct/IPC/script imzası değişmedi; shader derlemesi gerekmez. Ana
    build/link ve canlı kabul yok. Include=0, script.run_file/addon gibi Python
    isteyen IPC/API yollarının link/capability ayrımını çözmez. Sırada **Main.cpp
    (8043 satır) ortak AppRuntime / StudioShell ayrımı**, ardından Player VS hedefi.
  - Tam kaynak tablosu, test hataları ve kullanıcı kabulü:
    [Python sınır raporu](PLAYER_PYTHON_BOUNDARY_AUDIT_2026_11_13.md).

- **Faz 1j, Embree başlık/saç CPU ayrımı (2026-11-13, Codex):**
  Ortak `EmbreeHandles.h` opaque tipleri sağlar; `EmbreeBVH.h` SDK başlığı taşımaz.
  Gerçek `EmbreeBVH.cpp` SDK include'unu kendi alır ve Player adayından dışlanır.
  `HairEmbree.cpp` CPU BVH/ray/occlusion/volumetric brush sorgularını tutar; Studio
  bunu derler. Player `HairQueryHost.cpp` seçmeli (Studio'da None); offline sorgular
  hit üretmez, ortak strand üretimi/dynamics/Vulkan GPU hazırlığı aynı kalır.
  - Denetleyici BOM'lu ilk include'u atlıyordu: `utf-8-sig` ile düzeltildi, iki
    regresyon testi eklendi. Başlangıç Embree sayısı önceki raporda 6, düzeltilmiş
    okuyucuda **7** idi (gizlenen kaynak EmbreeBVH.cpp). CUDA/OIDN/Python sayıları
    değişmedi. Artış baseline'a yazılmadı; ayrım sonrası Embree=0 kilitlendi.
  - Güncel **552 aday / 536 temiz**: CUDA/OptiX **0**, Embree **0**, OIDN **0**,
    Python **16**. +1 Host/-1 offline kaynakla aday sayısı aynı. Önceki raporlanan
    530 temiz, BOM düzeltmesinden sonra gerçek başlangıçta 529 idi.
  - Seçili aktif kaynaklar `/Zs` **14/14 PASS**; altı Player include dökümünde
    CUDA/OptiX/Embree SDK yok. AssimpLoader.cpp aktif proje/tlog'da yok; /Zs yok.
    Dört CPU gövdesi + cleanup ve kalan ortak saç kodu kaynak eşitliği PASS.
    BOM testleri 2/2, XML/kayıt/include audit/diff-check PASS. ParticleRenderBridge
    çalışma çakışması nedeniyle düzenlenmedi, iki dalda yalnız denetlendi.
  - Struct alanı/yerleşimi, shader/ABI ve IPC/script imzaları değişmedi. Yeni
    shader yok. Ana build/link veya canlı/sayısal kabul yapılmadı; SDK'sız başlıklar
    offline sınıf sembollerini linkten kendiliğinden kaldırmaz. Sonraki parti:
    **16 Python girişini ortak API/IPC'den ayırma**; ardından runtime/loader/link sınırı.
  - Ayrıntı/kullanıcı kabulü:
    [Embree sınır raporu](PLAYER_EMBREE_BOUNDARY_AUDIT_2026_11_13.md).

- **Faz 1i, kalan 12 CUDA/OptiX adayı (2026-11-13, Codex):**
  Dokuz ortak kaynakta SDK işleri ayrıldı/kullanılmayan include'lar kaldırıldı;
  üç gerçek OptiX implementasyonu offline manifestine ayrıldı. Vulkan OIDN interop,
  gaz CUDA/FFT/doku, VDB CUDA kopyaları, terrain legacy CUDA/foliage ve renderer
  OptiX çağrıları hedefe göre seçilir. **9 Studio + 8 Host = 17 yeni modül**;
  Studio projede Host dosyaları yalnız None. Player karşı tarafı seçmeli.
  `scene_ui_nodeeditor.hpp` beklemesi ortak offline hook'a bağlı; hesaplar ortak
  kaldı. Yeni kullanıcı işlemi, struct alanı, IPC veya shader registry değişmedi.
  - Güncel **552 aday / 530 temiz**, CUDA/OptiX **0**, Embree **6**, OIDN **0**,
    Python **16**; baseline 0/6/0/16, include audit PASS. +5 aday = sekiz Host
    eklenmesi eksi üç gerçek offline dosyanın çıkarılması. OIDN düşüşü de offline
    OptixWrapper sınıflandırmasındandır; tüm iyileşme kaynak temizliği diye sayılmaz.
  - SDK'lı/CUDA'sız ortak ve adaptör kaynakları `/Zs` **35/35 PASS**; **17 Player**
    gerçek include dökümünde CUDA/OptiX SDK yok. **31** taşınan gövde/parametre
    kaynak eşitliği kontrolü PASS; **81** alan offset'i ve boyut/hizalama iki dalda
    `/Zs` **2/2 PASS**. Yeni XML/kaynak kayıtları ve diff-check PASS.
  - İki etkilenen CUDA shader doğrulandı: erosion PTX birebir aynı, kaynak/Release
    kopyaları hash PASS; Stylize PTX anonim kaynak-yolu sembol hash'i dışında aynı.
    İkisi ptxas PASS. Stylize exe'ye CudaCompile nesnesiyle bağlanır; native nesne
    sonraki kullanıcı Studio derlemesinde yenilenecek. Yeni shader yok.
  - **Ana build/link, Player hedefi ve canlı/sayısal kabul yok.** CUDA include=0
    link ayrımını kanıtlamaz. Host gas CPU fallback'i kullanır; realtime bütçe
    garantisi verilmez. Renderer'ın CPU/offline/Embree/Python/editör link sınırı ve
    ortak loader/kare döngüsü işi sürüyor. Sonraki başlık hedefleri Embree 6/Python 16.
  - Tam dosya/çift seçimi, doğrulama ve kullanıcı kabul listesi:
    [CUDA sınır raporu](PLAYER_CUDA_BOUNDARY_AUDIT_2026_11_13.md).

- **Kapsam kararı, eski Water/River sistemi (2026-11-13, kullanıcı):**
  Mevcut su sistemi olgun/realtime uygun görülmediği için Player'a taşınmayacak;
  realtime su ileride ayrı tasarlanacak. `WaterManager.cpp`, `RiverManager.cpp` ve
  `fft_ocean.cu` Player aday manifestinden dışlandı. Studio sistemi korunur;
  bu turdaki geçici WaterManager CUDA/Host ayrımı geri alındı, yeni modül kalmadı.
  Pişmiş `TriangleMesh` / DNA SoA geometri ve ortak malzeme gösterimi korunacak;
  Player yüklemesi su/nehir yöneticisini, authoring modifier'ını, dalga tick'ini veya
  CUDA FFT kaynaklarını yeniden kurmayacak. Bu karar Fluid/MPM/DEM simülasyonunu
  kapsam dışına çıkarmaz; onlar ayrı sistemlerdir.
  - Ölçüm **547 aday / 514 temiz**, CUDA/OptiX **12**, Embree **6**, OIDN **1**,
    Python **16**; include audit PASS, baseline 12. 549/515 → 547/514 ve 13 → 12
    farkı kapsam daralmasıdır, kaynak bağımlılığı temizleme kazanımı değildir.
  - **Link ayrımı henüz tamamlanmadı:** Player projesi henüz yok. Faz 2/3'te mevcut
    `Main.cpp` su tick/cleanup; `ProjectManager.cpp` water/rivers load/save/clear;
    `SceneSerializer.cpp` WaterSurface bind; `Renderer.cpp` FFT texture bridge;
    `RtApiAtmosphere.cpp` su rüzgârı ve `RtApiTerrain.cpp` nehir erişimleri Studio
    servislerine ayrılmalı. Atmosferin kendisi Player'da kalır; tüm atmosfer API
    dosyası sırf su çağrısı taşıyor diye dışlanmaz. TerrainNodesV2 authoring çağrıları
    Studio tarafında kalmalı. Player paketlemesi pişmiş flat mesh'i taşımalı.
  - Player'ın capability/IPC listesi eski Water/River işlemlerini sunmamalı;
    desteklenmeyen authoring işlemleri mevcut hata sözleşmesiyle reddedilmeli.
    Yeni realtime su geldiğinde ortak servis + script/IPC sözleşmesi ayrıca tasarlanır.
  - Bu son kapsam partisinde C++/shader/struct/IPC kodu değişmedi. Geçici ayrımın
    `/Zs` sonuçları nihai Player su desteği kabulü değildir. Build/link/canlı kabul yok.

- **Faz 1h, World host vektör matematiği (2026-11-13, Codex):**
  `World.cpp` artık `vec3_utils.cuh` içermez. Yalnız kullandığı dot/normalize
  işlemleri anonim namespace içindeki iki host yardımcıya taşındı; çarpım/toplam,
  sqrtf ve bileşen başına bölme sırası eski host formülüyle aynı (sıfır vektör
  davranışı dahil). Yardımcı/include dönüşümü geri çevrilince kaynak HEAD ile
  birebir aynı: PASS. Doku upload/getter ve LUT kaynak seçimi değişmedi.
  - Studio SDK'lı ve Player `RT_NO_CUDA` `/Zs` **2/2 PASS**; Player komutunda CUDA
    include dizini çıkarıldı ve `/showIncludes` dökümünde CUDA SDK başlığı yok.
    Tlog komutuna projenin ortam tabanlı include dizinleri eklendi; ilk eksik
    SDL_image include yolu düzeltildikten sonra iki dal da geçti.
  - Güncel **549 aday / 515 temiz**: CUDA/OptiX **13** (önce 14), Embree **6**,
    OIDN **1**, Python **16**. Include audit PASS; baseline 13'e sıkılaştırıldı.
  - Shader, struct ABI, IPC ve script sözleşmeleri değişmedi. C++ build/link,
    uygulama açma ve canlı/sayısal kabul yapılmadı. Sonraki kullanıcı kabulü:
    Nishita güneş yönü, reset/proje yeniden açma ve CPU/Vulkan/OptiX gökyüzü.
  - Sonraki WaterManager incelemesi kullanıcı kapsam kararıyla kapatıldı (üstte).
    VulkanBackend CUDA bağımlılığı Faz 1i ile ayrıldı (üstte).

- **Faz 1g, atmosfer LUT kaynak ayrımı (2026-11-13, Codex):**
  `AtmosphereLUT.cpp` ortak CPU hesabı, örnekleme ve Vulkan'ın okuyacağı host
  önbelleklerini tutar. CUDA initialize/cleanup ve üç doku yükleme çağrısı yeni
  `AtmosphereLUTCuda.cpp`'ye taşındı. Yeni `AtmosphereLUTHost.cpp` Player için
  initialized durumunu yönetir; CUDA handle'ları sıfır kalır ve CUDA upload yapılmaz.
  İki implementasyon link seçimidir; ikisi aynı hedefte derlenmez.
  - Studio vcxproj CUDA modülünü `ClCompile` olarak alır; Host modülü yalnız `None`
    olarak görünür. Gelecekteki Player projesi Host modülünü derlemeli, CUDA modülünü
    almamalı. Manifest CUDA modülünü dışlar; Host aday kümeye dahildir.
  - Hesaplama gövdesi ve CUDA lifecycle/upload gövdeleri kaynak karşılaştırmasında
    değişmedi: PASS. Yeni private metotlar host-only sınıf içinde; struct düzeni,
    shader kodu/ABI ve IPC değişmedi, etkilenen shader varyantı yok.
  - Studio SDK'lı ortak+CUDA ve Player `RT_NO_CUDA` ortak+Host `/Zs` **4/4 PASS**.
    Player `/showIncludes` dökümlerinde CUDA SDK başlığı yok. Link/canlı ve sayısal
    atmosfer kabulü yapılmadı; bu kontroller onları kanıtlamaz.
  - Güncel **549 aday / 514 temiz**: CUDA/OptiX **14** (önce 15), Embree **6**,
    OIDN **1**, Python **16**. Önceki 548/512'ye göre +1 aday/+1 temiz yeni Host
    modülünden gelir; diğer +1 temiz ortak atmosfer hesabının SDK'dan ayrılmasıdır.

- **Faz 1f, ortak hacim verisi (2026-11-13, Codex):** `GpuVDBVolume` ve
  `GpuGasVolume` tanımları yeni `include/GpuVolumeTypes.h` içine birebir taşındı;
  `params.h` bu başlığı içerir. Yeni başlık yalnız `GpuInteropTypes.h` ve `<cstdint>`
  alır; OptiX SDK istemez. `VolumetricRenderer.cpp` ve `VulkanBackend_Volumes.cpp`
  artık ortak hacim başlığını kullanır; scalar `make_float3(1)` çağrısı aynı değerli
  üç bileşenli çağrı yapıldı. Yeni başlık proje/filters'a kayıtlı.
  - Alan tanımları birebir karşılaştırma PASS. İki yapının toplam **149 alan offset'i**,
    boyut ve hizalama karşılaştırmaları MSVC `/Zs` ile SDK'lı ve `RT_NO_CUDA`
    dallarında PASS. Üç tüketici (`VolumetricRenderer`, `VulkanBackend_Volumes`,
    `OptixWrapper`) `/Zs` **3/3 PASS**.
  - Etkilenen **raygen / miss_kernels / hitgroup_kernels** PTX, bulk scriptindeki
    CUDA 12.6 / OptiX 9 / compute_50 / fast-math / maxrregcount=64 ayarlarıyla seçili
    derlendi. NVCC, giriş noktası, hedef ve hash kontrolü PASS; üçü de önceki
    `RayTrophiStudio/ptx` çıktısıyla **bayt bayt aynı**. Kaynak `ptx/` ve `x64/Release/ptx/`
    kopyaları hash ile doğrulandı. `x64/Debug` klasörü yok; Debug dağıtımı yapılmadı. Yeni shader yok, bulk listesi değişmedi.
  - Güncel **548 aday / 512 temiz**: CUDA/OptiX **15** (önce 17), Embree **6**,
    OIDN **1**, Python **16**. Struct/shader ABI, runtime davranışı ve IPC değişmedi.
    Ana C++ build/link veya canlı uygulama testi yapılmadı.

- **Faz 1e, kullanılmayan SDK include'ları (2026-11-13, Codex):**
  `OptixWrapper.h`'de kullanılmayan OIDN include'u kaldırıldı; yalnız SDK tipini
  kullanan `OptixWrapper.cpp` kendi include'unu aldı. `EmbreeBVH.h` yalnız
  `OptixGeometryData` için `OptixTypes.h` alır; bütün wrapper'ı taşımaz.
  `InstanceManager.cpp` ve `WaterManager.cpp` kullanılmayan wrapper include'unu bıraktı.
  - Güncel **548 aday / 510 temiz**: CUDA/OptiX **17** (önce 21), Embree **6**,
    OIDN **1** (önce 12), Python **16**. OIDN kalan tek aday `OptixWrapper.cpp`;
    ileride offline kütüphaneye ayrılacak. CUDA'sız temiz sayısı aynı kaldı çünkü
    temizlenen adayların başka yasaklı bağımlılıkları (örn. Embree/FFT CUDA) sürüyor.
  - MSVC `/Zs` **6 / 6 aktif kaynak PASS**: material API, ParticleRenderBridge,
    InstanceManager, WaterManager, OptixWrapper, EmbreeBVH. Include grafiğinde etkilenen
    `AssimpLoader.cpp` mevcut vcxproj'a kayıtlı değil, tlog komutu yok; denetlenmedi.
    Ölçüm diskteki aday kaynakları sayar, yalnız projede derlenen kaynaklar değildir.
  - C++ davranışı, struct düzeni, shader ABI/kodu, IPC değişmedi. Tam link/canlı kabul yok.

- **Git/devir durumu (2026-11-13):** gözlenen dal `main`; ayrı Player dalı henüz
  oluşturulmadı, bu çalışma commit edilmedi. Ürün ayrımı aynı kod tabanında iki VS
  projesidir; kullanıcı kararıyla ayrıca Player dalı açılmayacak. İleride CMake de
  aynı kaynak/kütüphane sınırlarını iki hedefe bağlayabilir; bu aşamada CMake eklenmiyor.
  Ortak çalışma ağacında ajan varken dal değiştirilmemeli.
  Her parti sonunda bu not, `NEXT_BUILD_CHECKS.md` ve `AJAN_KOORDINASYON.md` birlikte
  güncellenir: kapsam, ölçüm, koşulan/koşulmayan kontroller ve kalan iş açıkça yazılır.

- **Faz 1d, OIDN başlık ayrımı (2026-11-12, Codex):** `Renderer.h` OIDN SDK'sını
  artık içermez. Altı SDK reference üyesi gizli `OidnState` içine taşındı; özel deleter
  SDK tanımını ortak başlıkta gerektirmez. Denoiser metotları yeni offline
  `src/Render/RendererOidn.cpp` içinde (vcxproj/filters kayıtlı, Player adayından hariç).
  Kalite varsayılanları aynı sayısal değerleri taşır ve SDK ile static_assert edilir;
  SDK enum kullanan `Main.cpp` kendi SDK include'unu alır.
  - Taşınan metot gövdeleri yalnız SDK durumuna erişim öneki geri çevrilerek önceki
    kaynakla birebir karşılaştırıldı: PASS. SDK reference bırakma sırası korundu.
  - MSVC `/Zs` **7 / 7 PASS**: yeni modül, Renderer, Main, realtime-denoiser API,
    domain-creation API, PostSurface, VulkanBackend. Tüm kaynaklar yeniden taranmadı;
    link ve canlı denoiser kabulü henüz yok.
  - Güncel **548 aday / 510 temiz**; CUDA/OptiX **21**, Embree **6**, OIDN **12**
    (önce 71), Python **16**. Toplam taranan çeviri birimi 775.
  - `Renderer` C++ nesne yerleşimi değişti: sonraki kullanıcı TAM derleme partisine dahil.
    Bu başlık hijyenidir; `Renderer.cpp` hâlâ offline/Studio bağımlılıkları taşır.
    Player'ın ortak render servisleri/link ayrımı tamamlanmış değildir.

- **Politika değişikliği ölçümü (kullanıcı kararı, 2026-11-12):** ImGui/SDL/ses/Jolt
  Player'da kalır. Denetim yalnız CUDA/OptiX, Embree, OIDN ve Python'ı yasak sayar:
  **548 aday / 454 temiz**, sırasıyla **21 / 6 / 71 / 16**. Önceki 355 → 454 farkı
  ölçüt düzeltmesidir, ilave kaynak temizliği değildir. Aşağıdaki ImGui sayıları tarihsel.
  Bunlar gerekli altyapıya örnektir, kapalı bir izin listesi değildir. Kullanıcı
  saymasa da Player'ın çalışması için gereken diğer kütüphaneler korunur; kaldırma
  kararı gerçek offline render/Studio üretim aracı kullanımına göre verilir.

- **Faz 1c, API başlık ayrımı (2026-11-12, Codex):** `UIContext` yeni
  `include/UIContext.h` içinde; üye sırası, tipler ve varsayılanlar korundu.
  `RtApiInternal.h` artık `scene_ui.h` yerine bu hafif başlığı ve `SceneHistory.h`'yi içeriyor.
  Gerçek editör kullanıcıları `scene_ui.h`'yi, renderer/selection/color üyelerine erişenler
  ilgili tanım başlığını kendileri içeriyor. `RtManualSimulationStep.h` kendi editör include'unu aldı.
  - `RT_NO_CUDA` tiplerinde `float4/int4/uint4` 16, `uchar4` 4 bayt hizalı yapıldı;
    boyut + hizalama assert'leri eklendi. İç içe yapı yerleşimi ve bağımsız `UIContext.h`
    probu hem CUDA SDK hem `RT_NO_CUDA` ile MSVC `/Zs` PASS (obj/exe üretilmedi).
  - Güncel aday kümesi **548**: temiz **355**, CUDA/OptiX **21**, Embree **6**,
    OIDN **71** (önce 113), Python **16**, ImGui **176**. Toplam tarama 774 birim
    (yeni `TextureCuda.cpp` dahil); önceki 773 sayısı tarihsel kapsamdır.
  - Bu adım yeni runtime/API davranışı eklemez; script/IPC sözleşmeleri ve shader ABI aynı.
  - Doğrudan API iç başlığı tüketicilerinin **89 / 89'u MSVC `/Zs` PASS**.
    Eksilen manuel-step/terrain-graph/renderer tanımları kendi kullanan dosyalarına eklendi.
    Tam derleme/link ve canlı kabul henüz yapılmadı; `NEXT_BUILD_CHECKS.md` başına bak.

- **Önceki parti (2026-11-12, Faz 1a, YAZILDI — TAM derleme bekliyor):**
  - Yeni `include/GpuInteropTypes.h`: CUDA vektör tiplerinin tek giriş noktası. Studio'da birebir
    `<cuda_runtime.h>` + `<vector_types.h>`; `RT_NO_CUDA` tanımlıysa aynı adlarla düz yapılar.
  - `material_gpu.h`, `sbt_data.h`, `World.h`, `OptixTypes.h`, `AtmosphereLUT.h`, `Texture.h` onu içeriyor.
  - `Texture.h`: dört satır içi CUDA gövdesi yeni `src/Render/TextureCuda.cpp`'ye taşındı
    (`cudaUploadFull/cudaUpdateFull/cudaUpdateRegion/cudaRelease`); koruma mantığı başlıkta kaldı.
  - `Hair/HairSystem.h`: `<embree4/rtcore.h>` yerine `RTCScene` ileri bildirimi.
  - `Triangle.h`: `<OptixWrapper.h>` yerine `OptixTypes.h` + OptixWrapper'ın geçişli sağladığı birinci taraf başlıklar.
  - `scripts/audit_player_includes.py` + `scripts/player_include_manifest.json` (ölçüm ve cırcır; taban kilitlendi).
  - **Faz 1b (aynı parti):** `ParallelBVHNode.h` / `ParallelBVHNode_optimized.h` artık `OptixWrapper`'ı
    ileri bildirimle görüyor (`scene_data.h → HittableList.h → ParallelBVHNode.h → OptixWrapper.h`
    zinciri koptu). İki dosya adları dolaylı görüyormuş, kendi include'ları eklendi:
    `scene_ui_mesh_overlay.cpp` (+`OptixWrapper.h`) ve `VolumetricRenderer.cpp` (+`params.h`); ikisi de `/Zs` PASS.
- **Ölçülen etki (Player aday kümesi, 548 birim):** temiz 217 → **355**; CUDA/OptiX 313 → **21**; OIDN 274 → 113; Embree 113 → 6; ImGui 176 ve Python 16 değişmedi.
- **Bulgu (Faz 4 için):** `GpuVDBVolume` / `GpuGasVolume` `params.h`'de tanımlı ve `params.h`
  `<optix.h>` içeriyor. Vulkan'ın da kullandığı hacim yapıları bir OptiX başlığında yaşıyor;
  `VolumetricRenderer.cpp` Player'a girecekse bu yapılar OptiX'siz bir başlığa ayrılmalı.
- **Karar düzeltmesi (2026-11-12, kullanıcı):** ImGui, SDL, mevcut ses sistemi ve Jolt
  Player'ın ortak runtime altyapısıdır. ImGui ile HUD, ödül ekranı, bilgi paneli ve debug UI
  kullanılabilir. Studio panelleri ve authoring araçları ayrı tutulur; bir ImGui include'u
  tek başına editör bağımlılığı veya Player engeli değildir. `NodeCore.h`'yi yalnız ImGui
  içerdiği için ayırma işi öncelikten çıkarıldı.
- **Sıradaki:** kalan OIDN/CUDA/OptiX/Embree/Python girişlerini ve gerçek Studio servis
  bağımlılıklarını ayırmak. `UIContext` ayrımı korunur; runtime servis/link ayrımı henüz bitmedi.
  OIDN başlık sızıntısı kapandı; kalan tek OIDN adayı offline `OptixWrapper.cpp`.
  Ortak hacim tiplerinin `params.h`'ye bağımlılığı 1f'de kaldırıldı. CUDA/OptiX kalan
  14 adayda gerçek CUDA çağrıları ve offline backend bağlantıları var. Atmosfer LUT
  hesabı 1g'de ayrıldı; `World.cpp → vec3_utils.cuh` yalnız vektör matematiği için SDK
  taşıyan sıradaki küçük düğümdür. Ardından gerçek CUDA gas/FFT/interop işlemlerinin
  ortak runtime hizmetlerinden ayrılması ve 16 Python girişinin IPC'den ayrılması gerekir.
  ImGui/SDL/ses/Jolt ve diğer gerekli runtime kütüphaneleri korunur.
- **Player'da gereken karşılık:** `Texture::cuda*` dört üyesinin no-op tanımı (Faz 3'te
  `TextureNoCuda.cpp`; `#ifdef` değil link seçimi).

## 1. Amaç ve kararlar

Aynı `.sln` içinde iki exe:

| | Studio | Player |
|---|---|---|
| Rol | içerik üretimi, offline render | oyun çalışma zamanı |
| Render | hepsi | RayFusion; donanım destekliyorsa Vulkan RT + SVGF + FSR |
| OptiX / CUDA / Embree / OIDN | var | **yok** |
| ImGui | var | **var** — HUD, ödül/bilgi panelleri, debug UI |
| SDL pencere/girdi, mevcut ses sistemi, Jolt fizik | var | **var**, ortak runtime altyapısı |
| Studio panelleri, editör overlay/gizmo/node-editor UI'sı, SceneUI'a bağlı oturumlar (template hub) | var | **yok** |
| UI içermeyen MeshEdit/Paint/UV/Template/Import veri + servis kodu | var | **var** (karar 2026-11-13: yükleyici bunlara muhtaç; sınır "authoring" değil "editör UI") |
| Python | var | **yok** (karar 2026-11-12) |
| IPC | var | **var, zorunlu** (karar 2026-11-12; CLAUDE.md kural 1 — IPC'siz Player test edilemez) |
| Sahne | `.rtp` | önce doğrudan `.rtp`, paketleme Faz 5 |

VS 2022'de seçim: çözümde iki proje (`RayTrophiStudio`, `RayTrophiPlayer`) olur; başlangıç projesi
olarak hangisi seçilirse o derlenir/çalışır. Ortak kod statik kütüphanede bir kez derlenir.

### Sim kademeleri (karar 2026-11-12)

| Kademe | Kullanım | Player'da |
|---|---|---|
| Parçacık sistemi (emitter + GPU parçacık) | oyun VFX'inin büyük kısmı | canlı, varsayılan |
| Pişmiş sim önbelleği (`SimCache`, mevcut realtime oynatım) | yıkım, patlama, dökülen sıvı | Studio çözer, Player oynatır |
| Canlı çözücü | rigid body her zaman; madde/sıvı yalnız bütçeli küçük domain | isteğe bağlı |

Gerekçe: dinlenik MPM karesi parçacık sayısına değil dispatch zincirine bağlı (384 ve 100k parçacık
aynı ~100 ms GPU); hareketli karede CPU yüzey çıkarımı çözücüden pahalı. Yani "küçük domain" bugün
16 ms'e sığmıyor. Çözücüler Player'dan **link sınırıyla dışlanmaz** (Vulkan compute); canlı sim bir
derleme kararı değil sahne başına bütçe kararıdır: domain'e `canlı / pişmiş / kapalı` modu ve ms
bütçesi, ikisi de IPC'den okunur.

## 2. Mimari: link sınırı, `#ifdef` serpiştirme değil

| Kütüphane | İçerik | Studio | Player |
|---|---|---|---|
| `RtCore` | sahne, DNA/flat SoA, animasyon, Jolt fizik/sim, ses, malzeme, serileştirme | ✔ | ✔ |
| `RtRenderVk` | VulkanBackend, RayFusion, Vulkan RT, SVGF, FSR, atmosfer | ✔ | ✔ |
| `RtRuntime` | SDL pencere/girdi, cihaz, **tek kare döngüsü**, IPC pompası, ImGui oyun UI | ✔ | ✔ |
| `RtOffline` | OptiX, Embree, OIDN, CPU yol izleyici | ✔ | — |
| `RtEditor` | Studio panelleri, MeshEdit, Paint, import/export, node editörleri | ✔ | — |

- Player yasaklı kütüphanelere bağlanmadığı için sızıntı **link hatası** olur, sessiz kalamaz.
- **Tek kare döngüsü** iki exe'de ortak: ayrı döngü, Studio'da IPC ile test edilenin Player'da
  koşan şey olmaması demektir ("script testi kare döngüsüne kör" dersi).
- Tek istisna `GpuInteropTypes.h` içindeki `RT_NO_CUDA`: GPU paylaşımlı yapılar CUDA tip adlarıyla
  yazılı ve OptiX donduruldu (CLAUDE.md §6), o yapıları yeniden adlandırmak yasak bölgeye dokunur.
  **Başka hiçbir yerde `#ifdef RT_NO_CUDA` yazılmaz.** CUDA *fonksiyonu* çağıran başlık Player
  kümesine ait değildir; o kod offline `.cpp`'ye taşınır.

## 3. Ölçüm (2026-11-12, değişiklikten ÖNCE)

Aşağıdaki ilk ölçümler tarihsel olarak ImGui'yi de yasak sayıyordu; 217/355 temiz
sayıları bu eski ölçüte aittir. Yeni ölçüt yalnız CUDA/OptiX, Embree, OIDN ve Python'ı
engeller. Sayılardaki politika kaynaklı artış kod temizliği kazanımı olarak yorumlanmamalı.
SDL/ses/Jolt da izinli altyapıdır. Studio kaynak/link sınırı ayrıca denetlenir.

`python scripts/audit_player_includes.py` — geçişli include grafiği, 773 çeviri birimi.
Player aday kümesi 548 birim; yalnız **217'si temiz**.

| Kategori | Kirli birim | Giriş noktası (en yakın) |
|---|---|---|
| CUDA/OptiX | 313 | `sbt_data.h` 103, `Texture.h` 102, `World.h` 50, `material_gpu.h` 38 |
| OIDN | 274 | `OptixWrapper.h` 180 (`Triangle.h` üzerinden), `Renderer.h` 94 |
| ImGui | 176 | `ui_modern.h` 92, `NodeSystem/NodeCore.h` 70 |
| Embree | 113 | `Hair/HairSystem.h` 107 |
| Python | 16 | `Api/` altında tek tek bağlama dosyaları |

Okuma: sızıntı yüzlerce dosyaya yayılmış görünür ama **dokuz başlık** neredeyse hepsini taşıyor.
Dizin bazında en çarpıcı satır `src/Physics`: 184 birimin 112'si CUDA, 107'si OIDN görüyor ve
hiçbiri kullanmıyor.

Düzeltmeler (ilk elle tarama yanlıştı): `Ray.h` temiz — CUDA kısmı zaten `__CUDACC__` içinde.
`Triangle.h` yalnız `OptixGeometryData::TextureBundle` için bütün `OptixWrapper.h`'yi çekiyordu.

★ Scriptin sınırı: yalnız `__CUDACC__` / `RT_NO_CUDA` koşulları değerlendirilir, diğer her
`#if`'in iki dalı da alınır. Sonuç üst sınırdır (gerçek derlemenin atladığını raporlayabilir, aldığını gizleyemez).
Bu yüzden `GpuInteropTypes.h` üzerinden geçen birimler `RT_NO_CUDA` yokken hâlâ "CUDA görüyor"
sayılırdı; script artık Player görüşünde çalışıyor (`__CUDACC__` tanımsız, `RT_NO_CUDA` tanımlı).
"Temiz" = yasaklı başlığa ulaşılmıyor; link edilebilirliği kanıtlamaz.

## 4. Riskler

- **TDR kurtarması.** Studio'da cankurtaran OptiX. Player'da yok; Vulkan cihazını yeniden kurma
  yolu Player'ın kendi işidir ve sağlamlığı ölçülmedi.
- **RT pipeline ilk derlemesi** sürücüde 168 sn ölçülmüştü. Oyunda kabul edilemez: pipeline cache
  paketle gelmeli ya da kurulumda ısıtılmalı.
- **`get_frame` realtime oynatımı takip etmiyor** — pişmiş sim kademesinin IPC gözlemlenebilirliği
  Player'dan önce düzeltilmeli.
- **Parçacık Batch B** yazıldı, derlenmedi; Player'ın varsayılan VFX yolu ona dayanıyor.
- **FSR** (B4, FSR 3.1.4) başka ajana devredilmişti; canlı durumu doğrulanmalı.
- **Vegetation varlıkları** dağıtım kısıtlı; paketleme (Faz 5) bunu ele almalı.

## 5. Fazlar

**Faz 1 — Başlık hijyeni (tek projede, Studio davranışı değişmez).**
- 1a ✍ CUDA tip başlığı, `Texture.h` CUDA gövdeleri, `AtmosphereLUT.h`, Hair/Embree,
  Triangle/OptixWrapper (yazıldı, tam derleme bekliyor).
- 1b `EmbreeBVH.h` / `ParallelBVHNode*.h` → `OptixWrapper.h` zinciri ve `Renderer.h` OIDN üyeleri
  (pimpl, ya da `Renderer` bütünüyle `RtOffline`'a — önce kimin neye bağlı olduğu ölçülecek).
- 1c API/editör servis sınırı: Studio panel ve authoring bağımlılıklarını ayır; ImGui ortak kalır.
  `Api/` içinde IPC dispatch ile pybind bağlamasının ayrılması.
- Kabul: `audit_player_includes.py --check` PASS ve sayılar her partide düşüyor.

**Faz 2 — `Main.cpp` (8042 satır) ayrımı.** `AppRuntime` (pencere, cihaz, kare döngüsü, sim tick,
IPC) ve `StudioShell` (Studio panelleri, editör araçları). Studio hâlâ tek exe; kabul = mevcut IPC testleri aynen geçer.

**Faz 3 — Kütüphane bölünmesi + `RayTrophiPlayer.vcxproj`.** İlk Player: bir `.rtp` aç, RayFusion
ile göster, IPC'ye cevap ver. `Api/` içinde IPC dispatch ile Python bağlaması aynı dosyalarda
karışık (16 birim); IPC tarafı Player'a gelir, `RtPython*` gelmez.

**Faz 4 — Render katmanı.** `RenderCapabilities` ile kademe seçimi (RT → Vulkan RT + SVGF + FSR;
değilse RayFusion + FSR). Birikimli varsayımlar yerine sabit kare bütçesi, dinamik çözünürlük.

**Faz 5 — Oyun çalışma zamanı.** Sabit zaman adımı, girdi, Studio'da "Play" (aynı `RtRuntime`),
sim kademeleri ve bütçe, paketleme (pişmiş sahne + `.spv` + pipeline cache), cihaz kaybı kurtarma.

## 6. Faz 1a doğrulama yöntemi

Derleme kullanıcının; bu parti iki yolla denetlendi:

1. **Include grafiği** (`audit_player_includes.py`): yukarıdaki sayılar.
2. **MSVC `/Zs`** (yalnız sözdizimi, `.obj`/`.pdb` üretmez): MSBuild'in kaydettiği gerçek komut
   satırları (`CL.command.1.tlog`) ile 395 / 773 birim: 1a'da 157 (değişen adları kullanan 80'in tamamı dahil),
   1b'de `OptixWrapper.h`'yi kaybeden 238 birimin tamamı. 1b taraması 2 gerçek hata yakaladı
   (yukarıda). Kalan birimler denetlenmedi. Ayrıntı `NEXT_BUILD_CHECKS.md` başında.
   `/Zs` şablon örneklemesi ve link hatasını yakalamaz; tam derleme hâlâ asıl kanıt.
