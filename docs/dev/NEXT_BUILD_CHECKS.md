# NEXT BUILD CHECKS

## Player volume (VDB / gaz) + `scene.import_vdb` — 2026-11-14, Claude

İki iş tek partide:
1. **Kural-1 açığı kapandı:** VDB içe aktarma yalnız panelde, dosya diyaloğuna bağlıydı
   (hiçbir test sahneye VDB koyamıyordu). Gövde `rtapi::importVdbVolume`
   (`Api/RtApiVdbVolume.h/.cpp`, RtApi.h'ye dokunulmadı = tam derleme yok) oldu; panel
   onu çağırıyor. IPC `scene.import_vdb {path, orientation}`, Python `scene.import_vdb`,
   yetki `FilesRead|SceneWrite` + uzak yol denetimi, overlay tarifi; descriptor üreticisi
   ve `audit_ipc_capabilities.py` (Python aynası dahil) OK.
2. **Player volume:** `SceneUI::syncVDBVolumesToGPU` gövdesi `AppRuntime::publishVolumes`
   (VulkanSceneSync) oldu. Studio render + viewport, Player yalnız tek viewport aygıtı
   (Player'daki "editörde yok" saplaması artık ortak fonksiyonu çağırıyor; ortak API'nin
   `RtApiFluid`/`RtVolumeMaterialBindings` çağrıları da Player'da çalışır hale geldi).
   Ret kalktı; volume Hittable'ları düz-mesh denetiminden muaf; hazırlıkta ve
   `g_gas_volumes_dirty` servisinde aynı çağrılar. IPC `runtime.vdb_volumes / gas_volumes`.

**CANLI PASS (20:33 Player / 20:38 Studio):** `scene.import_vdb` Studio'da çalıştı (101 km bulut → 5 birim, GPU upload); Player `vdb_volumes=1`, volume'lu/volume'suz fark 944 piksel (bulut ölçekle küçük, ortalama fark 0.43 bu yüzden), görüntüde gölgeli bulut kümecikleri gözle doğru. Regresyon 6/6 PASS (bootstrap dahil). Madde 1 (panelden elle içe aktarma) kullanıcı gözü bekliyor. **Parti kapandı.** /Zs 12/12 PASS, include audit + üretici PASS. Ret örneği `rt_player_scene_ipc.py`:
`vdb_volumes` → `fracture_recipes`. **İki projeyi reload (yeni .cpp) → ikisini artımlı build.**
Header: yalnız yeni `RtApiVdbVolume.h`, `VulkanSceneSync.h`.

1. Studio panelinden VDB içe aktarma eskisi gibi (diyalog, toast, seçim, ölçek/hizalama).
   Bozuksa: `SceneUI::importVDBVolume` artık yalnız diyalog + API + seçim.
2. Fixture: `python scripts/test/rt_player_volume_ipc.py --make-fixture` (Studio açık, başlangıç
   sahnesi) → `scene.import_vdb` bilgi döner, `volume.rtp` kaydedilir.
3. Player: `python scripts/test/rt_player_volume_ipc.py` → PASS: `vdb_volumes=1`, aynı kamerayla
   volume'lu/volume'suz kare farkı > 500 piksel (çeyrek çözünürlük).
   - `vdb_volumes=0` → ProjectManager VDB'yi Player'da yüklemedi (dosya yolu / GPU upload).
   - 1 ama fark yok → tablo yayınlandı ama MaterialPreview volume dalı çizmiyor:
     ★ sinsi: `[MPVolume] volumeCount=0 bound=0 -> SKIPPED` logu "sahnede gaz yok" diye
     okunmuştu (2026-09-04); Player logunda bu satıra bak.
4. Regresyon: scatter / particles / materials / sim_clock / scene (`--player-only`) PASS.
5. Bilinen sınır: animasyonlu VDB dizisi (timeline `updateFromTimeline`) Player'da sürülmüyor.

## Player scatter / instance kabulü — 2026-11-14, Claude

Ölçüm: Vulkan aygıtında yükleyici scatter gruplarını sahneye açmıyor (native
InstanceGroup, `gpu_dirty`), raster build (`VulkanViewportBackend::buildRasterGeometry`)
grupları `InstanceManager`'dan doğrudan okuyor; Player'ın çağırdığı `buildRasterScene`
bunun içinde. Yani taşınacak gövde yoktu; engel yalnız iki ret satırıydı. Kaldırıldı;
çalışma sırasında scatter değişimi Studio'nun aynı `refreshRasterScatterInstances`
çağrısıyla servis ediliyor; IPC `runtime.scatter_groups / scatter_instances`.
Taban: eski Player fixture'ı `unsupported_in_player_scene: instances` ile reddediyor
(doğrulandı). `rt_player_scene_ipc.py` ret örneği `instances` → `vdb_volumes`.
**CANLI PASS (20:11 Player):** 1 grup / 150 instance, 622 sarı piksel, 174k görünür üçgen / 6 draw call (instancing); görüntü gözle doğru (gölgeli, ölçek değişken). Regresyon 4/4 PASS. **Parti kapandı.** /Zs 2/2 PASS.

1. `python scripts/test/rt_player_scatter_ipc.py` → PASS: 1 grup, ≥75 instance,
   görüntüde sarı çakıl > 40 piksel.
   - `scatter_instances` 0 → binary sidecar'daki instance'lar okunmadı
     (`deserializeBinaryInstances`).
   - instance var, sarı yok → raster build grupları atladı: ★ sinsi aday, kaynak mesh
     (`Pebble`, y=-20) çizilip kopyalar çizilmiyorsa görüntüde HİÇBİR şey yanlış görünmez.
2. Regresyon: particles / materials / sim_clock / scene (`--player-only`, yeni ret örneği) PASS.
3. Bilinen sınır: scatter rüzgârı (`InstanceManager::updateWind`) Studio'da timeline'a bağlı;
   Player'da sürülmüyor — sonraki animasyon/tick adımıyla.

## Player partikül kabulü — 2026-11-14, Claude

Player artık `particle_simulation` içeren sahneyi reddetmiyor. Partiküller ortak sim
saatinde adımlanır, Studio'nun aynı billboard yoluyla çizilir:
`SceneUI::uploadParticleBillboards` gövdesi `AppRuntime::uploadParticleBillboards`
(VulkanSceneSync) oldu; SceneUI yalnız çağırır. `UI/ParticleBillboardBuilder.cpp`
UI'a bağlı değildi, yalnız klasörü yüzünden Player dışındaydı → manifestle dahil
(dosya değişmedi, Codex'in dosyası). Partikül compute bağlamı Player aygıtında zaten
yayınlanıyor (`VulkanDevice::publishAsSimComputeContext`). /Zs 5/5 (builder Player
bayraklarıyla dahil), include audit + üretici --check PASS. Taban: eski Player fixture'ı
`unsupported_in_player_scene: particle_simulation` ile reddediyor (doğrulandı).

**CANLI PASS (19:51 Player / 19:55 Studio):** link temiz; 1183 canlı partikül, `particle_draws=1` (tek pulled çizim = GPU-resident yol, host kopyası yok), 426 magenta piksel; görüntü gözle doğru. Regresyon 3/3 PASS. Partiküller zeminden geçiyor (fixture'da çarpıştırıcı yok; adım 2'nin konusu). **Parti kapandı.** İlk not: **İki projeyi reload (Player'a yeni dosya) → ikisini artımlı build.** Header:
yalnız `Core/VulkanSceneSync.h`, `Core/RuntimeSceneViewer.h` (az TU).

1. **Player linki.** Builder ilk kez Player'da: LNK2019 çıkarsa builder'ın uzandığı
   bir sembol Studio-yalnız dosyada demektir (adını söyle, host/taşıma kararı).
2. Studio: partikül sahnesinde Solid/Material görünümde billboard'lar eskisi gibi;
   Debug görünümünde (display mode 1) billboard yok, nokta overlay var.
3. `python scripts/test/rt_player_particles_ipc.py` (Player açık) → PASS: `particles_alive > 200`,
   `particle_draws > 0`, görüntüde magenta piksel > 50.
   - alive 0 → emitter Player'da yüklenmedi ya da saat partikül sistemini adımlamıyor.
   - alive var, draws 0 → sistem `emitter_only` ya da `visible=false` kaydedilmiş.
   - draws var, magenta yok → billboard pass Player'ın MaterialPreview karesinde
     çizilmiyor. ★ Sinsi: GPU-resident sistemde "pulled" çizim aygıt eşleşmesine bağlı;
     eşleşmezse CPU quad yoluna düşer — çalışır, ama `particle.stats` host senkronu artar.
4. Regresyon: `rt_player_materials_ipc.py`, `rt_player_sim_clock_ipc.py`,
   `rt_player_scene_ipc.py --player-only` PASS.

## Player materyal tam desteği — 2026-11-14, Claude

**Canlı ölçüm (derlemeden ÖNCE, 19:28 Player):** çok materyalli + dokulu fixture
(`rt_player_materials_ipc.py --make-fixture`: dama dokulu zemin, kırmızı/yeşil/mavi
küp) Player'da 5 materyal, `default_material_used=false`; görüntü Studio'nunkiyle
gözle aynı (doku, renk, gölge). Statik materyal/doku yolu zaten tamam: Player
Studio'nun çift yönlü `Renderer::updateBackendMaterials/updateBackendMaterial`
çağrısını kullanıyor; tek aygıtta tek yükleme.

İki boşluk kapatıldı (yalnız `RuntimeSceneViewer.cpp` + `RtPlayerIpcPolicy.h`):
- Editör ızgarası Player'da çiziliyordu (proje `grid_opacity` 0.4 taşıyor) → Player
  yüklemede `grid_opacity=0`.
- Yükleme sonrası toplu materyal/doku/ışık kirliliği servis edilmiyordu →
  `advanceRuntime` Studio'nun aynı çağrılarıyla tüketiyor.
- Player IPC: `material.list/get/of_object/set` (oyun kuralı ve boya için; çekirdek yol).

**19:41 Player build ile CANLI PASS:** materyal testi (renkler, dama kontrastı, `grid_band_std=0.0`, `material.set` kırmızı→sarı görüntüde), sim saati ve EchoRoomsLab regresyonu PASS; görüntü gözle kontrol edildi. **Parti kapandı.** /Zs 3/3 PASS. **Yalnız Player artımlı build yeterli** (Studio'da davranış değişmedi;
politika başlığı Studio'da 4 IPC TU'yu yeniden derletir, gerekmez).

1. `python scripts/test/rt_player_materials_ipc.py` (Player açık) → PASS: renkler,
   dama kontrastı, `grid_band_std < 1`, `material.set` ile kırmızı küp sarıya döner.
   - `editor grid visible` → `grid_opacity` sonradan yeniden yükleniyor (ör. proje
     ayarının ikinci uygulanması) ya da backend opaklık 0'da da çiziyor.
   - `material.set did not reach the Player image` → per-slot yol viewport'a
     ulaşmıyor; ★ sinsi: test geçer görünür ama renk **bir sonraki tam yüklemede**
     gelir — bekleme döngüsü render_calls artışına bakıyor, görüntüye değil.
2. `python scripts/test/rt_player_sim_clock_ipc.py` ve `rt_player_scene_ipc.py --player-only`
   regresyonu PASS.

## Ortak simülasyon saati (Player adım 1) — 2026-11-14, Claude

**İlk canlı tur (19:11/19:14 exe'leri):** Studio maddeleri 1–3 PASS (fixture üretildi;
Studio'da ortak saatle kutu 3.0 → 0.25 m düşüp zeminde durdu, scrub kare 0/20/99 =
3.0/0.318/0.25 geri yükledi). Madde 5 PASS (EchoRoomsLab; sim'siz sahnede saat boşta).
**Madde 4 FAIL → düzeltildi, yalnız Player artımlı build bekliyor:** Player
`playing=true` veriyordu; `updateSimulationTimeline` serbest akışı yalnız
`live_mode && !playing` için koşar, timeline yolu playhead 0'da hiç adım atmaz
(184 tick, kutu y=3'te). `RuntimeSceneViewer.cpp` artık `playing=false`. /Zs PASS.
Bootstrap testinin sabit metot listesine iki yeni salt okunur metot eklendi.
**19:28 Player build ile madde 4 PASS:** kutu 2.96 → 0.25 m, 231 tick, 230 raster refit, 1 rebuild (yükleme), piksel imzası değişti. Madde 5 tekrar PASS. **Parti kapandı.**

### İlk partinin listesi

Simülasyon tick'i panelden (`scene_ui.cpp`) `Core/RuntimeSimulationClock`'a taşındı;
sim'in hareket ettirdiği gövdelerin raster refit'i Main'den `VulkanSceneSync`'e.
Player her döngüde aynı saati canlı modda sürüyor. Yeni .cpp iki projeye kayıtlı
(Player vcxproj üretici ile). /Zs 8/8 (Player 4, Studio 4) ve include audit PASS.
Önceki "ilk kare device-loss" maddesi 2026-11-14 akşam canlı PASS oldu.

**İki projeyi reload et (yeni dosya) → ikisini artımlı Release x64 Build/link.**
Header değişikliği yalnız `Core/RuntimeSceneViewer.h`, `Core/VulkanSceneSync.h` ve
`Api/RtPlayerIpcPolicy.h` (az TU); yaygın struct düzeni değişmedi.

Sıra (bağımsız ve hızlı önce):

1. **Studio açılır, boş sahnede Play/Pause çalışır.** Panel yalnız saati çağırıyor.
   Bozuksa: `scene_ui.cpp` çağrı bloğu (`AppRuntime::simulationClock().tick`).
2. **Studio'da var olan bir sim sahnesi (rigid/gaz) timeline'da eskisi gibi oynar,
   scrub geri yükler, sim parametresi değişince "Sim changed - cache reset" toast'ı
   çıkar ve playhead başa döner.** Bozuksa: `rewind_to_start` sonucu UI'da uygulanmıyor.
   ★ En sinsi olası hata: **Timeline Start/End değiştirmek simülasyonu sıfırlıyorsa**
   aralık-düzenleme koruması ilk tick'te kaçmıştır (saat ilk çağrıda aralığı yalnız
   öğrenir). Hata vermez, yalnız ağır sahne başa atlar.
3. **Fixture (Studio açık, BOŞ sahne):** `python scripts/test/rt_player_sim_clock_ipc.py --make-fixture`
   → `docs/dev/.captures/player-sim-clock/rigid_drop.rtp` yazılır, Studio sahnesi
   yeniden boşalır. Bozuksa: fixture sorunu, Player değil.
4. **Player kabulü:** `python scripts/test/rt_player_sim_clock_ipc.py` → PASS:
   kutu ≥1.5 m düşer, zeminde (y 0–0.6) durur, `driven_ticks` artar,
   `raster_refits + raster_rebuilds > 0`, piksel imzası değişir.
   - `box did not fall` + `driven_ticks` artıyor → solver adımlıyor ama rigid gövde
     kaydolmadı (ProjectManager rigid_bodies yüklemesi / syncSimulationWorld).
   - `driven_ticks` artmıyor → `advanceRuntime` çağrılmıyor ya da
     `anySimulationRuntimeEnabled()` false.
   - kutu düşüyor ama `motion never reached raster` / imza aynı → deform refit
     yolu: `setDeformRefitActive(true)` etkisiz, `markDirty(Geometry)` yolu servis edilmiyor.
   - ★ Sessizce makul: kutu düşüyor, imza değişiyor ama **her karede
     `raster_rebuilds` artıyor** → refit hep reddedilip tam raster kuruluyor; çalışır
     görünür, büyük sahnede kare süresi çöker. Beklenen: rebuild ~0–1, refit çok.
5. **EchoRoomsLab regresyonu:** `python scripts/test/rt_player_scene_ipc.py --player-only` PASS
   (sim'siz sahnede saat hiçbir şey yapmamalı; `simulation.clock_ticks` 0 kalır).
6. Player penceresi arka plandayken sim sürer (kiosk kararı); render yalnız ön planda
   ya da capture açıkken. Arka planda 20 Hz pompada dt 1/30'a kırpıldığı için sim
   gerçek zamandan yavaş akar — bilinen, kabul edilen sınır.

## Player ilk kare device-loss düzeltmesi — 2026-11-14

İlk canlı yükleme PASS (26 mesh/312 üçgen), ilk GPU görüntü FAIL: uzun NVIDIA
pipeline hazırlığından sonra device_lost=true, 1 submit/0 consume. Hazırlıkta geç
capability yayını ve boş materyal tablosu görüldü; RuntimeSceneViewer bunları
artık düzeltir/doğrular. Kesin GPU kök nedeni yeniden canlı kabul bekler.

**Yalnız Player'ı kapat → Player artımlı Build/link → aç.** Bu kabul için yeniden
Studio build istenmiyor. Sonra dış terminalde `python scripts/test/rt_player_scene_ipc.py
--player-only` veya ikisi de açıksa aynı komutun --player-only olmayan biçimi.
Yeni material_buffer_ready true, material_count>0, gerçek kare/draw ve reload kabulü
geçmeli. /Zs, 644/644 include ve Python syntax PASS. Ayrıntı ve açık cold-pipeline
UI borcu: PLAYER_STATIC_VIEWER_ACCEPTANCE.md başındaki canlı sonuç.


## Birleştirilmiş Player .rtp / Vulkan kabul paketi — kaynak hazır

Önceki AppFrame ve konsolsuz Release partileriyle birlikte statik scene viewer
bağlandı. Ara build checkpoint'leri yerine **iki projeyi reload edip ikisini bir
kez artımlı Release x64 Build/link et**. Yalnız kaynak/sözdizimi denetimi yapıldı;
yeni exe için link, shaderların runtime yüklenmesi ve görüntü henüz ölçülmedi.

İkisi açıkken `python scripts/test/rt_player_scene_ipc.py` dış terminalden çalışır.
Varsayılan fixture mevcut EchoRoomsLab.rtp; alternatif `--project "tam-yol.rtp"`.
Betik gerçek görüntü/GPU sayaçları, kamera, yeniden açma ve hata semantiğini ölçer;
Studio'ya yazmaz. Sonra resize, boşta tüketim, konsolsuz açılış ve temiz kapanış
kontrol edilir. Ayrıntı: [PLAYER_STATIC_VIEWER_ACCEPTANCE.md](PLAYER_STATIC_VIEWER_ACCEPTANCE.md).

Scope statik flat mesh/material/camera/light/environment MaterialPreview'dır.
Scatter/VDB/gas/su/particle/hair/fracture runtime yok; açık ret uygulanır. Tam ortak
sim/animasyon tick'i bu partiye dahil değildir. Yeni shader/ABI değişikliği yok.
Player 644/644 include temiz, generator/descriptor/capability audit PASS; yeni ve
bağlanan kaynakların /Zs kontrolleri PASS. Eski aşağıdaki notlar tarihsel; bu
birleştirilmiş kabul paketi build sırasını supersede eder.


## Ortak kare işlemleri — AppFrame (2026-11-13, kaynak hazır)

**İki proje reload + ikisi için artımlı Build/link.** Yeni `Core/AppFrame.cpp`
ikisinde de ClCompile; Main.cpp ve PlayerMain.cpp yalnız çağrı bağlantısı alır.
Önceki konsolsuz Release Player proje ayarı bu generator çıktısında korunur.
C++ layout/shader/IPC sözleşmesi değişmedi; Codex C++ build veya app launch yapmadı.

- Studio'nun 3 ImGui kare başlangıcı, 5 Render kapanışı, 4 draw/present yolu;
  olay pompası ve mevcut konumundaki IPC pompası ortak FrameCycle kullanır.
  Scene-loading sırasında IPC engeli, sunum renkleri/texture copy ve perf kapsamları
  yerinde kalır. Studio'ya Player'ın foreground/idle/pacing politikası uygulanmaz.
- Player bootstrap aynı ortak operasyonları kullanır. Değişmeyen HUD boşta
  çizilmez; odak dışı sunum yok; 60/20 Hz uyuyan pompa korunur.
- Bu yalnız ortak kare I/O katmanıdır: sim tick, geometri/kamera/malzeme/VDB
  senkronu ve Vulkan render halen Studio döngüsünde. `.rtp` Player'a açılmadı;
  Player discovery/izin listesi bu partide genişletilmedi.

Kullanıcı kabulü:
1. İki exe birlikte açıldığında dış terminalde
   `python scripts/test/rt_player_bootstrap_ipc.py` PASS olmalı.
2. Player sabit odaklı/odaksız/minimize halde tüketimi düşük kalmalı; geri dönüş,
   resize ve pencere kapatma çalışmalı. Yeni GUI subsystem ile konsol açılmamalı.
3. Studio sahne yükleme ekranı, normal viewport, çözünürlük değişimi ve yüklü
   sahneyle IPC cevapları korunmalı. Yükleme sırasında kuyruğa alınan çağrı,
   yükleme bittikten sonra çalışmalı; loader sahnesine arada yazmamalı.
4. Rendered/RayFusion mod geçişi ve normal kapanış mevcut davranışı korumalı.

Statik: generator --check PASS; iki XML'de AppFrame.cpp tam birer kez kayıtlı;
include audit 641/641 temiz, dört yasaklı kategori 0; include audit testleri 2/2;
diff whitespace kontrolü PASS. Bunlar yeni C++ link/canlı kabul yerine geçmez.


## Player konsol penceresi — 2026-11-13

Studio + Player birlikte açıkken dış IPC bootstrap kabul betiği PASS.
Kullanıcı idle düzeltmesini iki hedefte derlediğini ve Player boşta CPU/GPU
tüketiminin görünmediğini bildirdi. Scene render henüz yok; statik HUD görüntüsü
yeniden çizilmeden korunur.

Player generatoru Release/ReleaseDev için `SubSystem=Windows` üretir; Debug
Console kalır. SDL2main zaten link listesinde; SDL_main girişini WinMain üzerinden
çağırır. IPC named-pipe sunucusunun konsola ihtiyacı yoktur. **Player kapat, VS
projesini reload et, Release x64 için Build/link al.** Studio rebuild gerekmez.
Beklenen: konsol açılmadan Player/HUD başlar; tam dış IPC betiği yine geçer.
Generator/XML kontrolü PASS; yeni GUI subsystem link/canlı açılışı bekler.


## Player bootstrap boşta tüketim düzeltmesi — 2026-11-13

Kullanıcı Player 3e build PASS bildirdi. Açık Player üzerinde dış Python IPC
bootstrap kabul testi **PASS** (sandbox error 5 sonrası aynı prob escalation ile).
Bu test Studio'yu veya kapanışı doğrulamaz.

Yeni PlayerMain değişikliği için yalnız **Player artımlı build** gerekir; Studio
kaynakları değişmedi. Player kapatıldıktan sonra derle. Sabit HUD artık yalnız
pencere/girdi olayı veya IPC durum değişiminde çizilir; boşta NewFrame/clear/draw/
present çağrılmaz. Odak dışı/minimize sunum tamamen atlanır; IPC kuyruğu en çok
yaklaşık 50 ms aralıkla işlenir. Odakta olay patlamaları yaklaşık 60 FPS ile
sınırlı; boşta olay/IPC kontrolü yaklaşık 60 Hz, arka planda 20 Hz uyuyarak çalışır.
FPS yazısı sürekli çizim gerektirdiği ve bootstrap için anlamlı olmadığı için
"Scene: not loaded" durumuyla değiştirildi. Bu politika oyun simülasyonuna
uygulanmış değildir; bootstrap döngüsüne aittir.

Kabul: odakta fareyi sabit tutup CPU/GPU tüketimini karşılaştır; sonra odağı
başka pencereye geçir ve minimize et. Sürekli GPU sunumu durmalı, tüketim
belirgin azalmalı. Geri dönünce/resize sonrası HUD yenilenmeli. Arka planda da
`python scripts/test/rt_player_bootstrap_ipc.py` cevap vermeli (Studio hazırken).
Yeni performans davranışı henüz derlenmedi veya canlı ölçülmedi.


## Player Faz 3e — ayrı IPC ve bootstrap capability sınırı (kaynak hazır)

1. Visual Studio proje değişikliğini reload et. **Player Rebuild** yap: yeni
   `RT_PLAYER` hedef tanımı nedeniyle eski Player objeleri kullanılmamalı.
   Studio için değişen dört IPC kaynağını içeren **artımlı build** yeterli;
   C++ layout/shader ABI değişmedi. Codex C++ build veya uygulama başlatmadı.
2. Studio ve Player'ı birlikte aç. Ayrı terminalden:
   `python scripts/test/rt_player_bootstrap_ipc.py`
   Beklenen: Studio kendi kimliğini/geniş kataloğunu, Player bootstrap kimliğini
   ve dört keşif metodunu bildirir; mesh/scatter/fracture/project/script/batch
   Player'da `unsupported_in_player` hatası verir. Sayısal `agent.describe.method`
   parametresi hatayla reddedilir. Betik uygulama açmaz ve sahneyi değiştirmez.
3. Player'ı pencere kapatma düğmesiyle kapat. Task Manager'da Player işleminin
   bittiğini kontrol et (pencerenin kaybolması tek başına yeterli değil).
   Sonra `python scripts/test/rt_player_bootstrap_ipc.py --studio-only` çalıştır:
   Studio hâlâ cevap vermeli. Player'ı yeniden açıp tam betiği tekrar çalıştır.
4. Bağlı bir IPC istemcisi varken, ayrıca art arda `agent.discover` istekleri
   giderken Player'ı kapat; işlem kısa sürede bitmeli. İstemcide pipe kapanış
   hatası normaldir, Player'ın asılı kalması değildir. Bu yarış canlı doğrulanmadı.

Yerel named-pipe varsayılanları içindir. İki uygulamada birden isteğe bağlı TLS
etkinleştirilirse mevcut ortak port yapılandırması ayrıca ayrılmalıdır.
Canlı prob sandbox Windows error 5 verirse aynı dış Python probunu escalation
ile tekrar çalıştır; alternatif taşıma veya embedded script kullanma.

Statik sonuç: generator `--check` PASS; include audit 640/640 (dört yasaklı
kategori 0); mevcut include audit birim testleri 2/2; Python syntax/help PASS.
Bu sonuçlar C++ build/link, IPC cevabı veya temiz kapanış kanıtı değildir.


## Güncel kabul durumu — 2026-11-13

Kullanıcı tam Studio derlemesini sorunsuz tamamladığını bildirdi: Faz 1k'ye kadar
biriken değişikliklerde **Studio TAM build/link PASS (kullanıcı bildirimi)**.
Önceki Studio build bekleme notları Faz 1k'ye kadar bu bildirimle güncellendi.
Kısmi canlı smoke geçti; [rapor](PLAYER_LIVE_SMOKE_2026_11_13.md). Kullanıcı kalan
canlı testleri devraldı. Gerçek AOV resize/temiz kapanış ve sayısal kabul bekler.

## Player Faz 3d — SceneUI durum host'u + Embree host'u (Claude, 2026-11-13)

**Kullanıcı (2026-11-13 22:47): Player derlendi, linklendi, açıldı, HUD göründü
(madde 1 ve 3'ün görsel kısmı PASS). Ajan: madde 2 PASS — `dumpbin /dependents`
çıktısında yasaklı DLL yok. AÇIK: madde 3'te temiz kapanış, madde 4 (IPC cevabı ve
editör-bağımlı metotların hata döndürmesi).**

Studio'da yalnız iki None kaydı: **reload yeter, Studio derlemesi gerekmez.**
Studio kaynak koduna bu partide dokunulmadı. Sıralı kontrol:

1. **Player derleme + link (sağ tık → Build).** Beklenen: link GEÇER ya da tek haneli
   sayıda yeni sembol kalır. Log: `RayTrophiStudio\x64\Release\Player\RayTrophiPlayer.log`.
   LNK2005 (çift tanım) çıkarsa: bir Host dosyası ile gerçek implementasyon aynı anda
   Player'a girmiş demektir (adı ver).
2. (Link geçerse) **Bağımlılık kontrolü.** `dumpbin /dependents x64\Release\RayTrophiPlayer.exe`
   çıktısında `cudart`, `cufft`, `nvcuda`, `OpenImageDenoise`, `embree4`, `python311`
   **olmamalı**. Varsa ayrım sızdırıyor.
3. **Çalıştır.** Studio KAPALIYKEN `x64\Release\RayTrophiPlayer.exe`: siyah pencere,
   sol üstte HUD ("RayTrophi Player", "IPC: running", FPS). Pencere X'i ile temiz
   kapanır. Açılışta çöküyorsa `SceneLog.txt` son satırlarını ver.
   ★ Studio açıkken "IPC: stopped" görünür: pipe adı ortak, beklenen.
4. **IPC cevabı.** Player açıkken `Import-Module .\scripts\ipc\RtIpc.psm1 -Force` ve
   `Invoke-RtIpc agent.discover` (ya da herhangi bir okuma metodu) cevap verir.
   ★ Sinsi: `physics.fracture_object`, `mesh.*` operatörleri, `scatter.*` Player'da
   HATA dönmeli (ya da log'da "[Player] '...' is implemented in the Studio editor").
   Sessizce `ok` dönüyorsa bir stub başarı taklidi yapıyor demektir — bana bildir.

## Player Faz 3c — UI'sız authoring kodu Player'da + su/template host'ları (Claude, 2026-11-13)

**Kullanıcı (2026-11-13): Player derleme 0 hata, link 168 → 39.**

Studio'da yalnız iki None kaydı var: **reload yeter, Studio derlemesi gerekmez.**
Player projesi yeniden üretildi (803 kaynak; ~48 yeni dosya derlenecek).

1. **Player derleme (sağ tık → Build).** Derleme hatası beklenmez; yeni alınan
   MeshEdit/Paint/UV/Template/Import dosyalarından derleme hatası çıkarsa ilk
   20 satır yeter (yalnız 4'ü `/Zs` ile denetlendi).
2. **Player link.** 168'den belirgin düşmeli; MeshEdit/Paint/Template/su/seçim/
   import/Python grupları kapanmalı. Kalan beklenen: SceneUI'a bağlı çağrılar,
   AgentChatPanel, Embree 4 (+ yeni dosyaların getirdiği SceneUI sembolleri).
   Log: `RayTrophiStudio\x64\Release\Player\RayTrophiPlayer.log` — yolu vermen yeter.
   ★ Sinsi: sayı düşmek yerine ARTARSA başarısızlık değildir — yeni alınan
   servis dosyaları SceneUI'a uzanıyor demektir; liste EditorBridge kapsamını belirler.

## Player Faz 3b — ortak runtime state + codec + host'lar (Claude, 2026-11-13)

**Kullanıcı (2026-11-13): Studio build PASS, HDR/EXR sorunsuz; Player linki 364 → 168.**

Studio projesine 2 yeni ClCompile (`Core/RuntimeState.cpp`, `Utils/ImageCodecs.cpp`)
ve 3 None (Host) eklendi → **VS çözümünü reload et**. TAM rebuild GEREKMEZ: artımlı
build Main.cpp, scene_ui.cpp, scene_ui_modifiers.cpp, scene_ui_hierarchy.cpp,
ProjectManager.cpp (scene_ui_animgraph.hpp) ve iki yeni dosyayı derler. Shader/
struct düzeni/IPC değişmedi. Sıralı kontrol:

1. **Studio link.** LNK2005 (çift tanım) ya da LNK2001 (çözülmemiş) **yok**. Bu
   partide bir globalin tanımı Main/scene_ui'den `RuntimeState.cpp`'ye taşındı;
   LNK2001 `g_*`/`stbi_*`/`LoadEXR` → yeni dosya projeye girmemiş (reload);
   LNK2005 → aynı adın ikinci tanımı kalmış (adı bana ver).
2. **Studio açılış + proje aç.** Normal açılır, bir proje yüklenir, viewport
   çizilir. HDRI/EXR ortam dokusu yüklenir (tinyexr), PNG ekran görüntüsü/kaydet
   çalışır (stb_image_write). ★ Sinsi: EXR'li sahne açılıyor ama gökyüzü düz renk →
   tinyexr yapılandırma makrosu kaybolmuş olabilir; log'da EXR yükleme hatası ara.
3. **Animasyon grafiği.** Anim graph panelinde clip node'u sahnedeki clip'leri
   listeler (`g_uiClipsRef`). Liste boşsa panel pointer'ı kurmuyor demektir.
4. **Player link (sağ tık → Build).** Hata sayısı 364'ten belirgin düşmeli (~160
   civarı beklenir). Çıktıyı / `RayTrophiStudio\x64\Release\Player\RayTrophiPlayer.log`
   yolunu ajana ver; kalanlar MeshEdit/Paint/Template/SceneUI/su/Embree kararları.

## Player Faz 3a — ilk Player derleme/link denemesi (Claude, 2026-11-13)

Studio projesine yalnız iki None kaydı eklendi (PlayerMain.cpp, TextureHost.cpp);
Studio derlemesi etkilenmez ama VS çözümü **reload** ister (yeni proje). Player
"Build Solution"a dahil DEĞİL. Sıralı kontrol:

1. **Studio hâlâ tek başına derleniyor.** Build Solution → yalnız RayTrophi Studio
   derlenir. Player da derlenmeye başlıyorsa sln'de Build.0 satırı oluşmuş demektir.
2. **Player projesi yükleniyor.** Solution Explorer'da "RayTrophi Player" görünür,
   yüklenemedi uyarısı yok. Uyarı varsa: `python scripts/gen_player_vcxproj.py --check`.
3. **Player derleme (Release|x64, sağ tık → Build).** Derleme aşamasında hata
   beklenmez (`/Zs` geçen dosyalar dışındakiler denetlenmedi; derleme hatası çıkarsa
   ilk 20 satırı yeter). **Link aşamasında çok sayıda LNK2001/LNK2019 BEKLENİR** —
   başarısızlık değil, ölçüm: çıktı penceresinin tamamını (ya da
   `x64\Release\Player\*.log`) ajana ver; hatalar sembol/dosya bazında kümelenip
   sırayla ayrılacak.
   ★ Sinsi durum: link **temiz geçerse** ve exe CUDA/OIDN/Embree/Python DLL'i
   istemiyorsa harika; ama `dumpbin /dependents x64\Release\RayTrophiPlayer.exe`
   çıktısında `cudart`, `OpenImageDenoise`, `embree4`, `python311` görünürse dönüşüm
   bir bağımlılığı kaçırmış demektir.
4. (Yalnız link geçerse) Player'ı çalıştır: siyah pencere + sol üstte HUD
   ("RayTrophi Player", IPC: running, FPS). Studio açıkken IPC "stopped" görünür —
   pipe adı ortak, beklenen.

## Player Faz 2c — olay pompası + çözünürlük değişimi (Claude, 2026-11-13)

**Kullanıcı (2026-11-13): build PASS, viewport çözünürlük değişimi sorunsuz.**

Yalnız Main.cpp ve AppWindow.h/.cpp değişti; artımlı build yeter, reload gerekmez.
Shader/struct/IPC değişmedi. `/Zs` 3/3, include audit 548/548 PASS. Sıralı kontrol:

1. **IPC uyanması.** Uygulamayı arka plana al, `Invoke-RtIpc` ile herhangi bir metodu
   çağır: cevap hızlı gelmeli. 10× yavaşlık → pompa SDL_PushEvent olayını yutuyor
   (bkz. IPC'nin ana döngüyü uyandırmaması dersi).
2. **Kapatma.** Pencere X'i: kaydedilmemiş sahnede kaydet sorusu gelmeli. Soru
   gelmeden kapanıyorsa quit_requested → `ui.tryExit()` bağı kopmuştur.
3. **Kısayollar.** Ctrl+S, Delete, H; animasyon render'ında ESC/P çalışmalı.
4. **Odak.** Başka pencereye geçip geri dön: viewport donmuş kalmamalı
   (focus_gained yolu).
5. **Çözünürlük.** Render ayarlarında çözünürlüğü değiştirip Apply: kısa siyah kare,
   sonra yeni boyutta görüntü. ★ Sinsi durum: görüntü eski boyutta gerilmiş kalırsa
   doku yeniden kurulmamış ama hata da loglanmamış demektir; log'da
   "Failed to create SDL surfaces" varsa kurulum başarısız ve erken dönülmüştür.

## Player Faz 2b — ortak pencere/SDL yaşam döngüsü (Claude, 2026-11-13)

**Kullanıcı (2026-11-13): 3–5 PASS** (Türkçe giriş, docking/ayırma).

Yeni `Core/AppWindow.h/.cpp` proje/filter'a kayıtlı → **VS'de projeyi reload et**.
Faz 2a ile aynı artımlı build/link partisine girer (Main.cpp + iki yeni .cpp).
Header/struct düzeni değişmedi; shader derlemesi gerekmez. `/Zs` 3/3, include audit
548/548 PASS. **Kullanıcı bildirimi (2026-11-13): Faz 2a+2b build/link PASS,
Studio sorunsuz açıldı** (madde 1–2). Kalan 3–8 ayrıca doğrulanmadı.
Sıralı kontrol (bağımsız/hızlı önce):

1. **Link.** Beklenen: `window`/`renderer`/`surface`/`original_surface`/`raytrace_texture`
   için LNK2005 (çift tanım) veya LNK2001 (çözülmemiş) **yok**. Varsa: AppWindow.cpp
   projeye girmemiş (reload) ya da başka bir .cpp bu globalleri hâlâ tanımlıyor.
2. **Normal açılış.** Splash görünür, durum metinleri sırayla akar, ana pencere
   "RayTrophi Studio" başlığıyla maximize açılır. Pencere hiç görünmüyorsa veya
   splash takılı kalıyorsa: createMainWindow sonrası SDL_ShowWindow yolunu kontrol et.
3. **Yazı tipi ve Türkçe karakter.** Panellerde Segoe UI ve ş/ğ/ı doğru. Kare/soru
   işareti görüyorsan: ImGuiConfig.font_path geçmiyor → ImGui varsayılan fontuna düştü.
   ★ Sinsi olan bu: varsayılan font da okunaklıdır, yalnız Türkçe glifler bozulur.
4. **Docking + klavye.** Panel sürükle/yerleştir çalışır, Tab ile alan gezinmesi var.
5. **IME/metin girişi.** Bir ad alanına Türkçe harf yaz; karakter gelmiyorsa
   SDL_StartTextInput çağrısı kaybolmuştur.
6. **`--startup-diagnostics`** ile aç: StartupCrash/emergency logda
   `SDL_Init done … ImGui SDL init done` satırları eskisiyle aynı metin ve sırada.
7. **Temiz kapanış.** Pencereyi kapat: süreç takılmadan çıkar, `StartupCrash.log`
   silinir, SceneLog'da kapanışta hata yok. Kapanışta AV → terrain node önizleme
   dokusu renderer'dan sonra bırakılıyor demektir (sıra Main'de korunmalı).
8. **`--headless`/gizli açılış.** Pencere gösterilmeden IPC hazır olur ve kapanır.

## Player Faz 2a — Studio launch ayrımı

Main.cpp ve yeni Core/StudioLaunchOptions.cpp kullanıcı artımlı Studio build/link
partisine dahil. Yeni .cpp proje/filter kayıtlı; VS projeyi reload et.
`/Zs` 2/2, parser gövde eşitliği, XML tekil kayıtları ve include audit PASS.
Player 547/547 temiz; SDK sınırları 0/0/0/0. Shader derlemesi gerekmez.

Sonraki build'de normal açılış/pozisyonel proje ve --project, --script;
--render/--spp/--frame, --frames/--frames-dir ve --headless davranışını kontrol et.
Eksik argüman, bozuk/taşan integer ve bulunamayan proje hata metinleri korunmalı.
Bu parça yeni exe üzerinde canlı denenmedi; şu an açık exe Faz 1k build'idir.

## Player Faz 1k: Python include 16 → 0 — 2026-11-13 (Codex)

Sonraki kullanıcı TAM build/link partisine dahil; şimdi build talebi değildir.
Studio yeni on RtPython*Bindings.cpp dosyasını ortak servis/IPC modülleriyle
birlikte derler. Player yalnız ortak tarafı alır. İmzalar, IPC descriptor/schema,
struct ve shader alanları değişmedi; shader derleme/dağıtım ihtiyacı yok.

- `/Zs` **34/34 PASS**; **11 Player** include dökümü SDK'sız. **32** kaynak eşitliği,
  proje/filter kayıtları ve diff-check PASS. Include audit **547/547 temiz**,
  CUDA/OptiX/Embree/OIDN/Python = **0/0/0/0**. IPC capability/descriptor audit PASS.
- Kaynak sözleşmeleri **86/89**: timing boyutu, cache v18 ve eski params.h alan
  beklentileri parti öncesi API snapshot'ında da FAIL. Ayrı post-v2 JS denetimi
  dokunulmayan OptixWrapper launchOptixDisplayPost guard'ında FAIL. Ayrıntı ve
  test dosyaları [Python raporunda](PLAYER_PYTHON_BOUNDARY_AUDIT_2026_11_13.md).
- Kullanıcı kabulü: proje reload/tam Studio link, rt importu ve ses/UV/volume/
  lens-weather/object-substance/post/cutout/particle-label/weather/timing Python
  çağrıları; karşılık gelen IPC metotları; bozuk tür, eksik context/kamera,
  NaN/Inf reddi. Önceki partilerin kabul listesi korunur. Canlı/nümerik kabul yok.
- Player gerçek linki yapılmadı. Script/addon IPC yolları ve Python sahibi API/UI
  tick'leri runtime/capability sınırında ayrılmalı; include=0 DLL'siz exe kanıtı değil.

## Player Faz 1j: Embree include sınırı 0 — 2026-11-13 (Codex)

Sonraki kullanıcı TAM derleme/link partisine dahil; şimdi build talebi değildir.
Studio `HairEmbree.cpp`, Player `HairQueryHost.cpp` seçer; iki implementasyon aynı
hedefte linklenmez. EmbreeBVH.cpp SDK include'unu kendi alır. EmbreeBVH.h artık
SDK tanımı sağlamaz; rtc*/somut RTC kullanan kaynak kendi SDK include'unu almalı.

- `/Zs` 14/14 PASS: beş ortak kaynak iki dal, üç Studio kaynağı (Main dahil) ve
  Player Host. Altı Player dökümü SDK'sız. AssimpLoader.cpp aktif kayıt yok; /Zs yok.
  Taşınan dört CPU gövdesi/cleanup ve kalan ortak saç kodu kaynak eşitliği PASS.
- Include audit BOM ilk-satır hatası düzeltildi; iki regresyon testi PASS. Embree
  gerçek başlangıç 7 (önce 6 raporlanmıştı), son 0. Son sayılar **552 aday/536
  temiz**, CUDA/OptiX 0, Embree 0, OIDN 0, Python 16. Baseline sıfıra kilitlendi.
- Kabul: CPU Embree render/material remap, instance/particle bridge; saç lazy BVH,
  brush picking ve shadow occlusion; Vulkan groom animasyonu/GPU upload; proje
  yeniden açma, clear ve kapanış. Önceki partilerin kontrolleri korunur.
- Ana build/link ve canlı/sayısal kabul yok. Shader, struct alanı veya IPC değişmedi;
  shader derleme ihtiyacı yok. Kaynak seçimi ve ayrıntı
  [Embree sınır raporunda](PLAYER_EMBREE_BOUNDARY_AUDIT_2026_11_13.md).

## Player Faz 1i: CUDA/OptiX include 12 → 0 — 2026-11-13 (Codex)

Sonraki kullanıcı TAM derleme/link partisine dahildir; bu kayıt şimdi build talebi
değildir. **17 yeni adaptör** Studio projede kayıtlı: 9 Studio ClCompile, 8 Host
None. Projeyi reload et; iki alternatif aynı hedefte linklenmemeli. Tam seçim
[CUDA sınır raporunda](PLAYER_CUDA_BOUNDARY_AUDIT_2026_11_13.md).

- SDK'lı/CUDA'sız `/Zs` **35/35 PASS**; 17 Player include dökümünde CUDA/OptiX yok.
  31 taşınan kaynak eşitliği ve 81 alan/sizeof/alignof (iki dal) PASS. Include audit:
  **552 aday / 530 temiz**, CUDA/OptiX **0**, Embree 6, OIDN 0, Python 16.
- Erosion PTX derlendi, eski gövdeyle birebir aynı; kaynak + Release hash PASS.
  Stylize CUDA PTX kaynak-yolu sembol hash'i dışında aynı, iki ptxas PASS.
  Stylize native CudaCompile nesnesi sonraki kullanıcı derlemesinde yenilenmeli.
  Yeni shader/binding/push/struct alanı/IPC değişikliği yok.
- Kabul: Vulkan RT/raster; VDB sequence ve canlı gaz; Studio CPU/CUDA gaz; erosion
  Vulkan/CPU fallback; OptiX foliage/saç/malzeme; OIDN GPU-direct/resize/geçiş/kapanış;
  Stylize, proje yeniden açma ve kaynak temizliği. Önceki partilerin kabulü korunur.
- C++ proje build/link ve canlı/sayısal kabul yapılmadı. Player hedefi kurulunca
  gerçek SDK kütüphanesiz link testi gerekir; Renderer/loader/IPC editör ve offline
  sembol ayrımı henüz tamamlanmadı. Gas Host CPU fallback performansı ölçülmedi.

## Player kapsam kararı: eski Water/River dışarıda — 2026-11-13 (kullanıcı)

Mevcut su/nehir ve CUDA FFT ocean Player'a taşınmayacak; realtime su ayrı yazılacak.
Studio su kaynakları aynı kaldı; geçici CUDA/Host ayrımı geri alındı. Player aday
manifesti WaterManager/RiverManager/fft_ocean.cu'yu dışlar; include audit PASS:
547 aday / 514 temiz, CUDA/OptiX 12, Embree 6, OIDN 1, Python 16. Azalma kapsam
kararından gelir. Bu parti C++/shader değişikliği veya yeni derleme talebi içermez.

Player hedefi/link ayrımı hazırlanırken su tick/cleanup, proje yükleme, serializer
bind, render FFT bridge ve atmosfer/terrain su API çağrıları Studio'ya ayrılmalı
(tam liste `RAYTROPHI_PLAYER_PLAN.md` devir notunda). Pişmiş flat su geometrisi ve
malzemesi yüklenmeli; Water/River runtime nesnesi, CUDA FFT ve eski su authoring
API'si oluşmamalı. Atmosfer ve ayrı Fluid sim korunur. Player henüz linklenmedi;
manifest dışlaması tek başına çalışır Player kanıtı değildir.

## Player Faz 1h World vektör matematiği — 2026-11-13 (Codex; temizlik sürüyor)

`World.cpp` CUDA `vec3_utils.cuh` yerine aynı formüllü iki yerel host yardımcı kullanır.
SDK'lı ve CUDA'sız `/Zs` 2/2 PASS; Player include dökümünde CUDA SDK yok.
Yardımcı/include değişimini geri çeviren kaynak eşitliği kontrolü PASS.
Include audit PASS: 549 aday / 515 temiz, CUDA/OptiX 14 → 13; baseline güncel.
Shader/struct ABI ve IPC değişmedi; shader derlemesi gerektiren değişiklik yok.

Sonraki toplu kullanıcı derlemesinde `World.cpp` derleme/link kabulü; ardından
Nishita güneş yönü (elevation/azimuth), reset, proje yeniden açma ve CPU/Vulkan/OptiX
atmosfer görünümü kontrol edilmeli. Ana build/link ve canlı/sayısal test yapılmadı.
Bu kayıt şimdi derleme talebi değildir; önceki TAM derleme listesi korunur.

## Player Faz 1g atmosfer kaynak ayrımı — 2026-11-13 (Codex; temizlik sürüyor)

Studio: ortak `AtmosphereLUT.cpp` + yeni `AtmosphereLUTCuda.cpp` derlenir. Player:
ortak dosya + `AtmosphereLUTHost.cpp` derlenmeli; iki kaynak implementasyonu aynı
hedefe konmamalı. Host dosyası Studio projesinde yalnız `None` öğesidir.

SDK'lı ortak/CUDA ve `RT_NO_CUDA` ortak/Host `/Zs` 4/4 PASS; Player include dökümleri
CUDA SDK'sız. Hesaplama ve eski CUDA gövdeleri birebir kaynak karşılaştırması PASS.
Shader/struct ABI değişmedi, shader derlemesi gerektiren değişiklik yok.

Include denetimi: 549 aday / 514 temiz; CUDA/OptiX 15 → 14. Yeni Host dosyası aday
ve temiz sayısını birer artırır; ortak atmosfer dosyasının temizlenmesi de +1 temizdir.
Sonraki toplu kullanıcı kabulünde atmosfer parametre değişimi, Vulkan host LUT
yüklemesi, CPU gökyüzü ve OptiX gökyüzü, proje yeniden açma/kapanış kontrol edilmeli.
Ana build/link veya canlı/sayısal kabul yapılmadı; bu kayıt şimdi build talebi değil.

## Player Faz 1f ortak hacim başlığı — 2026-11-13 (Codex; temizlik sürüyor)

`GpuVDBVolume`/`GpuGasVolume` artık `GpuVolumeTypes.h` içinde; `params.h` bunları
oradan alır. Vulkan volume upload ve ortak volumetric yayıncı OptiX launch parametrelerini
çekmez. Alan tanımları, 149 alan offset'i, sizeof/alignof karşılaştırmaları PASS;
SDK'lı ve CUDA'sız başlık probları PASS. Üç tüketici `/Zs` 3/3 PASS.

Etkilenen üç PTX (raygen, miss_kernels, hitgroup_kernels) derlendi: NVCC/entry/target
kontrolü PASS; eski kaynak çıktılarıyla bayt bayt aynı. Kaynak/Release `ptx/` kopyaları hash PASS; Debug klasörü yok, oraya dağıtım yapılmadı; ilgisiz shader'lar derlenmedi. Shader ABI değişmedi.

Include denetimi: 548 aday / 512 temiz; CUDA/OptiX 17 → 15, diğer sayılar aynı.
Ana C++ build/link ve canlı kabul yapılmadı; sonraki toplu kullanıcı doğrulamasında
Vulkan RT ve viewport'ta VDB + gas hacmi, ardından OptiX hacmi görünümü kontrol edilmeli.
Bu kayıt şimdi ana uygulama derlemesi talebi değildir.

## Player Faz 1e include temizliği — 2026-11-13 (Codex; temizlik sürüyor)

`EmbreeBVH.h` artık `OptixWrapper.h` sağlamaz; export veri tipi için `OptixTypes.h`
alır. `OptixWrapper.h` OIDN SDK sağlamaz; tek gerçek kullanıcısı kendi include'unu alır.
InstanceManager/WaterManager kullanılmayan wrapper include'larını bıraktı.

- MSVC `/Zs` 6/6 aktif kaynak PASS: material API, ParticleRenderBridge, InstanceManager,
  WaterManager, OptixWrapper, EmbreeBVH. Eski AssimpLoader.cpp projede kayıtlı değil,
  derleme komutu bulunmadı; denetlenmedi. Tüm proje tekrar taranmadı.
- Include denetimi: 548 aday / 510 temiz; CUDA/OptiX 21 → 17, OIDN 12 → 1,
  Embree 6 ve Python 16 aynı. SDK include'u ortak başlıklara geri eklenmemeli.
- Shader/struct/IPC davranışı değişmedi. Sonraki kullanıcı derleme partisine dahil;
  bu not şimdi derleme talebi değil. Önceki OIDN/OptiX/CPU hair kontrolleri korunur;
  instance scatter ve su sahnesinin görünümü de o kabulde kontrol edilir.

## Player Faz 1d OIDN başlık ayrımı — 2026-11-12 (Codex; temizlik sürüyor)

Bu kayıt şimdi derleme talebi değildir; sonraki kullanıcı TAM derlemesine dahildir.
`Renderer` nesne yerleşimi değişti. Yeni `RendererOidn.cpp` Studio projesi/filters'a
kayıtlı; unresolved `createOidnState`/deleter/denoise sembolü görülürse proje reload ve
yeni kaynak kaydı kontrol edilmeli. Ortak başlığa OIDN SDK include'u geri eklenmemeli.

- `RendererOidn.cpp`, `Renderer.cpp`, `Main.cpp`: MSVC `/Zs` PASS; yeni modül kalite
  varsayılanlarını SDK enum değerleriyle static_assert eder. Taşınan metot gövdeleri
  durum erişim öneki hariç önceki kaynakla birebir aynı doğrulandı.
  Realtime-denoiser API, domain-creation API, PostSurface, VulkanBackend de PASS:
  toplam **7 / 7**. Bütün kaynaklar yeniden sözdizimi taramasından geçirilmedi.
- Include denetimi: 548 aday / 510 temiz, OIDN 71 → 12; diğer yasaklı kategoriler aynı.
- Sonraki canlı kabul: host OIDN + GPU OIDN, kalite değişimi, çözünürlük değişimi,
  OptiX ↔ Vulkan geçişi, GPU denoiser release ve uygulama kapanışı. Bu partide koşulmadı.
- ImGui/SDL/ses/Jolt ve gerektiği ortaya çıkan diğer runtime kütüphaneleri korunur;
  bu örnekler kapalı izin listesi değildir.

**Player kapsam düzeltmesi (2026-11-12, kullanıcı):** ImGui, SDL, mevcut ses sistemi
ve Jolt Player'da korunur; Studio panelleri/authoring araçları ayrı tutulur. Güncel
include denetimi 548 aday / 454 temiz, CUDA/OptiX 21, Embree 6, OIDN 71, Python 16 PASS.
Aşağıdaki 355 temiz / ImGui bağımlılığı notları eski ölçüte aittir. Bu karar yeni C++
değişikliği veya şimdi derleme gereksinimi oluşturmaz; temizlik sürüyor.

## Player Faz 1c API başlık ayrımı — 2026-11-12 (Codex; kullanıcı TAM derlemesi bekliyor)

Önceki 1a/1b partisiyle birlikte derlenmeli. `UIContext` artık `UIContext.h` içinde;
üye sırası/tipler/varsayılanlar kaynak karşılaştırmasıyla aynı doğrulandı.
`RtApiInternal.h → scene_ui.h` bağımlılığı kaldırıldı. Editör/renderer/selection/color
üyelerini kullanan kaynaklar ilgili başlığı açıkça içeriyor; yeni runtime davranışı yok.

- CUDA'sız tiplerde float4/int4/uint4 hizalaması 16, uchar4 4 bayt olarak düzeltildi.
  Yerleşim/aggregate probu gerçek CUDA SDK ve `RT_NO_CUDA` ile `/Zs` PASS.
- `RtApiInternal.h`'yi doğrudan kullanan **89 / 89 kaynak MSVC `/Zs` PASS**.
  `RtManualSimulationStep.h`, `RtApiSpline.cpp`, `PostSurface.cpp` eksik tanım
  bağımlılıkları düzeltildi, tekrar kontrol edildi. İlk binding denemelerindeki
  eksik Python include yolu denetim ortamında tamamlandı; son binding kontrolleri PASS.
  Bu kontrol obj/exe üretmez; link veya canlı davranış kabulü değildir.
- Include denetimi: 548 aday, 355 temiz; OIDN 113 → 71, diğer kategoriler aynı.
  ImGui hâlâ `scene_data.h → GeometryNodesV2.h → NodeCore.h` yoluyla geliyor.
- Derlemede eksik `SceneUI`/`Renderer`/`SceneSelection`/`ColorProcessor` tanımı görülürse
  kullanıldığı kaynak kendi başlığını almalı; `RtApiInternal.h`'ye editör include'u geri eklenmemeli.
- Canlı kabul: IPC/Python üzerinden sahne sorgusu ve bir geri alınabilir düzenleme;
  manuel fluid/rigid adım sonrası timeline'ın eski kareyi geri yüklememesi;
  seçme/undo-redo ve simülasyon API smoke kontrolleri. Önceki OptiX doku kontrolleri de geçerli.

## Player ayrımı Faz 1a — 2026-11-12 gece (Claude; YAZILDI, kullanıcı TAM derlemesi bekliyor)

Plan: `RAYTROPHI_PLAYER_PLAN.md`. Bu parti yalnız **include grafiğini** değiştirir; davranış,
struct düzeni, shader ABI, IPC aynı kalmalı. Yaygın başlıklar değiştiği için **tam derleme** olur.

**★ Listenin başında uyarı:** `World.h`, `Texture.h`, `Triangle.h`, `material_gpu.h`, `sbt_data.h`
değişti. Artımlı derleme hemen hemen her şeyi yeniden derler; "Build" yeterli, ama garip bir
erişim ihlali görürsen önce Rebuild.

**Ön denetim (ajan, derleme değil):** bütün `src/**/*.cpp` MSVC `/Zs` ile (yalnız sözdizimi,
`.obj` üretmez; komut satırları `CL.command.1.tlog`'dan). Sonuç: **157 / 773 birim denetlendi, 157 PASS, 0 FAIL.** Kapsam: değişen adları (`OptixWrapper`, `optix*`, `oidn`, `rtc*`, `cuda*`, doku yükleme üyeleri) metninde geçiren 80 birimin TAMAMI + alfabetik ilk 77 birim. Kalan 616 birim denetlenmedi (tam tarama ~1.5 saat sürüyordu); bunlar o adları doğrudan kullanmıyor ama bir başlık üzerinden dolaylı bağımlılık olabilir — tam derleme asıl kanıt.

**Faz 1b aynı partiye eklendi:** `ParallelBVHNode.h` / `ParallelBVHNode_optimized.h` artık
`OptixWrapper`'ı ileri bildirimle görüyor. `OptixWrapper.h`'yi kaybeden **238 birimin tamamı**
`/Zs` ile denetlendi: 236 PASS, 2 gerçek hata. `scene_ui_mesh_overlay.cpp` düzeltildi (PASS).

`VolumetricRenderer.cpp` düzeltildi (`params.h` eklendi, PASS). Dosyayı kilitleyen 12:45'ten kalma
17 asılı `cl.exe` kullanıcı onayıyla sonlandırıldı.

Sıra: hızlı ve bağımsız olan önce.

1. **Derleme biter mi.** Beklenen: 0 hata. İlk saniyelerde `GpuInteropTypes.h` hatası =
   benim hatam, dur ve bildir. `unresolved external Texture::cudaUploadFull/cudaUpdateFull/
   cudaUpdateRegion/cudaRelease` = `TextureCuda.cpp` projeye girmemiş (vcxproj'a ekledim; VS açıkken
   eklendiyse "Reload" ister). `C2027/C2079 OptixWrapper` veya `rtc*` tanımsız = bir dosya bu
   adları `Triangle.h`/`HairSystem.h` üzerinden görüyormuş; o dosyaya kendi include'u eklenir.
2. **`python scripts/audit_player_includes.py --check`** → `PASS`. FAIL = biri çekirdek başlığa
   yasaklı include ekledi (sayı tabandan yüksek).
3. **Uygulama açılır, Vulkan RT / RayFusion'da dokulu bir sahne doğru görünür.** Bozuksa: bu
   yoldaki tek değişiklik include sırası; doku yanlışsa `Texture.h` bölünmesine bak.
4. **OptiX backend'e geç, dokulu sahne.** Asıl dokunulan yol burası: CUDA doku yüklemesi artık
   `TextureCuda.cpp`'de. Görmen gereken: dokular eskisi gibi. Dokular **siyah/eksik** =
   `cudaUploadFull` çağrılmıyor ya da erken dönüyor. Boyama (mesh paint / terrain splat) sonrası
   OptiX'te fırça izi görünmeli (`cudaUpdateRegion`).
5. **OptiX'te doku silme/değiştirme, sonra proje kapatıp açma.** `cudaRelease` yolu. Çökme ya da
   "yanlış doku yanlış mesh'te" = serbest bırakma sırası bozulmuş (bildirim → senkron → yok etme).
6. **Saç (hair) olan sahne, CPU render.** `HairSystem.h` artık Embree'yi ileri bildirimle görüyor;
   CPU saç kesişimi eskisi gibi çalışmalı.

**★ En sinsi başarısızlık:** OptiX'te dokuların *yüklenmeyip düz renk* görünmesi. Vulkan birincil
yol olduğu için günlük kullanımda fark edilmez, hata da vermez; OptiX ancak TDR sonrası cankurtaran
olarak açıldığında ortaya çıkar. Madde 4'ü atlama.

**Bu partide YAPILMAYAN:** Player exe'si yok, `RT_NO_CUDA` hiçbir yerde tanımlı değil; uyum
başlığının CUDA'sız dalı hiç derlenmedi (Faz 3'te ilk kez derlenecek).


## Ölçek partisi 2 — 2026-11-11 (AJAN DERLEDİ 19:29, canlı doğrulandı; kullanıcı derlemesi gerekmiyor)

Kullanıcı izniyle derlemeyi ajan aldı (ölçek planı süresince). Bu liste, kullanıcının kendi
açılışında gözle bakabileceği ve otomatik testin göremediği şeyler içindir.

**Bu derlemede (hepsi canlı koşuldu):**
- Domain bütçesi: varsayılan 4096 MiB. 1M kar 1 GiB bütçede bile bekletilmiyor (tahmin
  2048 → 384 B/parçacık).
- S1 dilim A (`MatterGpuStep.inl`): yayınlamada kalıcı ara tampon + swap, paralel NaN taraması,
  tüm-MPM'de sidecar sütun swap'ı; hazırlıkta tag başına madde, ıslak tepki kapalıyken tek
  yükleme.
  - ★ İlk derlemede ÇÖKTÜ: `thread_local` dizi OpenMP döngüsünde ada göre kullanılmıştı; her
    işçi iş parçacığı kendi boş kopyasını görüyor (null + 4·i). Döngüden önce işaretçiye
    bağlanarak düzeltildi; kural kodda yorum olarak duruyor.
- S5 blok P2G + sıralama (yeni kayıtlar 21/64 ve 5/64; cihazda paylaşımlı float atomik açık).
- `sim_sparse_mac_mark` karo kutusu; `sim_surface_centres` kutu sırası.

**Ölçülen:**
- 1M kar 2.05 → 1.76 s/kare; 1M kar+su 1.38 → 1.33 s.
- smoke MPM 30/30, multirate 7/7, yüzey paritesi geçti.
- A/B fiziği (`RT_P2G_BLOCK_OFF`): ≤ 0.3 mm.

**21:08 derlemesi (ajan):** hücre kümeli P2G denendi; daha yavaş çıktığı için söküldü. Exe,
19:29'daki blok P2G haline döndü. Smoke elastic + snow_support PASS.

**22:12 derlemesi (ajan):**
- `RtApi.cpp` `enqueue` artık ana döngüyü uyandırıyor (arka plandaki pencerede IPC 500 ms
  uykuyu bekliyordu).
- P2G permütasyon denemesi söküldü (exe değişmedi, yalnız `.spv`).
- Doğrulama: smoke 30/30, multirate 7/7, yüzey 8/8.

**15:11 derlemesi (ajan, 2026-11-12): yalnız parsel tek damla ("uydu" kümeleri).**
- Görsel çocuklar (parsel başına 8 küçük küre) yığındaki boşlukları doldurmak için var; havada
  tek kalan parselde ana noktanın çevresinde sıkı bir küme olarak görünüyordu. Artık çözücünün
  Sprey etiketini taşıyan parsel, yığındaki çocuklarla AYNI BOYDA tek küre çiziliyor.
  (14:40 sürümü ana parselin hacminde tek küre çiziyordu; küçük çocukların yanında iri top
  gibi durduğu için kullanıcı gözlemiyle değişti.) Köprü bunu point-sphere
  örneğinde NEGATİF `scale.y` ile işaretler (`fluidRenderProxyIsolated`).
- Yalnız çizim tarafı; çözücüye, adım süresine ve cache'e yük yok.
- ★ İlk deneme dizi komşuluğuna bakıyordu ve yığının yarısını da büyük küreye çevirdi:
  parseller doğum sırasını korur, karıştıktan sonra dizi komşusu uzayda komşu değildir.
- Canlı (10 648 su parseli, 84 sprey): yığın içi düzgün küçük küreler, havadakiler tek damla.
- ⚠ Kapsam dışı: raster "resident" GPU yolu (`fluid_sphere_proxy.vert`) etiketi göremiyor,
  o yol devredeyken eski kümeler kalır. Bu sahnede CPU impostor yolu çalışıyordu.
- Codex'in Field samples yolu ölçüldü (87k parsel): her adımda
  "transport unavailable or CFL budget exceeded", 0 örnek, adım başına 0.4–6 s CPU. Kapalı kalmalı.
- 1. Gözle: sıçrama sahnesi, Solid ve Rendered. Havadaki damlalar tek küre, yığın eskisi gibi.
- 2. Damla yığına geri düşerken küçük bir "pop" (tek küre → 8 çocuk) beklenir; sprey görsel
  olarak hacminin 1/çocuk sayısı kadarını gösterir (ince serpinti). Pop rahatsız
  ediyorsa etiket histerezisini (ayrılma ≤ 2, katılma ≥ 6 komşu) gevşet.
- 3. ★ Sinsi durum: mesh splat (icosphere vb.) kullanan bir domainde negatif ölçek geometriyi
  ters çevirirdi. İşaret yalnız procedural kürede konuyor; mesh splat sahnesinde spreyin
  normal göründüğünü bir kez kontrol et.

**13:43 derlemesi (ajan, 2026-11-12): DEM alt adımı ölçülen sönümden.**
- Kararlılık sınırı sönümü "bütçedeki bütün temaslar tek doğrultuda" sayıyordu (14 temasta
  toplam sönüm oranı 1.14, adımda 2.7× ceza). Shader artık tane başına gerçek sönüm oranını
  ölçüyor (`diagnostics[47]`), host geçen karenin en büyüğünü 1.5× payla kullanıyor; dört
  doğrusal temasın altına inmez. Kuvvet yasası DEĞİŞMEDİ, yalnız adım boyu.
- ★ Arayüz: tane shader revizyonu 43, tanı başlığı 48 kelime (ses telemetrisi bir kelime kaydı).
  Beş tane shader'ı derlendi, `.spv` iki yerde.
- 74k kum (R = 1 cm): 496 → 382 alt adım/kare, toplam sönüm oranı 1.14 → 0.51. Artık doğruluk
  sınırı (çarpışma başına 24 adım) belirliyor. Yığın oturuyor: en hızlı tane 2 s'de 0.02 m/s.
  IPC: `fluid.matter_models` → `cfl_terms.measured_damping_rate_s`.
- Testler: hızlı duman 16/16, grain_sleep 3/3, grain_inflow 6/6, tane sözleşmesi PASS.
- ⚠ İki FAIL, bu değişiklikle bağı DOĞRULANMADI (önceki exe ile koşulmadı):
  `grain_heat` 7/8 (`cools_at_physical_rate`; ısıl bir geçit) ve `sleep_transfer`
  (ilk adımda `sim.control_state` eşitlik assert'i, fizik koşmadan düşüyor).
- 1. Gözle: bilinen bir kum sahnesini oynat. Yığın eskisi gibi oturmalı. Titreyen ya da
  yavaşça "kaynayan" yığın görürsen ölçülen sönüm payı yetmiyor demektir:
  `DemContactTimeBounds.h`'de 1.5 çarpanını büyüt.
- 2. Hızlı çarpma (yüksekten dökülen kum, hızlı kollider): patlama OLMAMALI. Olursa çarpma
  karesi önceki karenin seyrek temas ölçümünü kullanmıştır; taban değeri (4) yükselt.
- 3. ★ Sinsi durum: yığın biraz daha "yumuşak" sönerse kimse hata demez. Aynı sahnede
  `cfl_terms.substeps` ile birlikte yığının son yüksekliğini eski kayıtla karşılaştır.

**13:24 derlemesi (kullanıcı, 2026-11-12): DEM tane nozulu (inflow).**
- Tane maddesi artık `liquid_emission=inflow` ile dökülüyor (`planGrainInflow`): akışa dik,
  titretilmiş altıgen katmanlar, katman yüzeyi geçince doğar. Debi = alan × hız, kare hızından
  bağımsız; tıkalı nozul borç biriktirmez.
- Canlı: `rt_test_grain_inflow_ipc.py` 6/6 (458 tane, beklenen 466; kare başına en çok 19,
  ortalama 14.8). Bağımsız planlayıcı testi 3/3: en yakın çift 2R × 1.05–1.07, sayı %1 içinde.
- Katı oranı nozul kesitine göre ~%27 (taneler duvardan bir yarıçap içeride doğar).
- 1. Gözle: panelden Sand kaynağını inflow yap, hız ver. Kesintisiz bir sütun dökülmeli.
  Kare başına ayrı öbekler görürsen katman fazı kareler arasında taşınmıyor demektir.
- 2. Nozulu zemine ya da yığına göm, birkaç saniye beklet, çek. Çekince topluca boşalma
  OLMAMALI. Olursa reddedilen doğumlar birikime geri yazılıyor demektir.
- 3. ★ Sinsi durum: kutu (object bounds) nozulu yalnız kutunun orta düzleminden döker;
  ince uzun bir kutuda hız ekseni yanlışsa debi makul görünür ama beklenenden düşüktür.
  `fluid_flux_target_m3_s` ile `total_emitted_particles` artışını karşılaştır.
- Ölçüm (değişiklik yok): 74k kum, R = 1 cm, 496 alt adım/kare, 348 ms/kare
  (`sim_matter_grain_step` 0.70 ms/çağrı). Sınırlayan kararlılık: 14 temas varsayımı ve
  toplam sönüm oranı 1.14 (aşırı sönümlü). Doğruluk sınırı tek başına 381 alt adım ister.
  Komşu listesi her alt adımda yeniden kuruluyor (22 ms/kare).

**12:06 derlemesi (ajan, 2026-11-12): kollider yüz ağırlıklarında tam kesin kısayol.**
- `computeSolidFaceWeights`: yüz merkezindeki SDF mesafesi Lipschitz payından uzaksa yüz
  tamamen içte (0) ya da tamamen dışta (255) sayılır, 16 örnek atlanır. Kesin bir sınır,
  yaklaşım değil: dem1'de 16 örnekle karşılaştırıldı, ~400k yüzün ~%83'ü kısayoldan,
  **0 uyuşmazlık**.
- dem1 (hareketli 2.5 m mesh-SDF küre): `sim.fluid.solid_face_weights` 24 → ~8 ms/kare.
  Adım: uyuyan karelerde 55 → ~40 ms.
- Tam GPU taşıması yapılmadı. Ağırlıkları dört host tüketicisi okuyor: yoğun basınç
  statiği yüklemesi, sparse kaynak paketleyici, CPU yolları ve gözenek geri yüklemesi.
- Testler: collider_phi 12/12, mpm_collider 8/8.
- 1. Gözle kontrol: kollider kara değerken temas biçimi öncekiyle aynı mı? Kar ya içine
  sızıyorsa ya da önünde boşluk kalıyorsa, Lipschitz payı kısa kalmış demektir.
  `sdf_lipschitz` 1.1 güvenlik çarpanını büyüt.
- 2. ★ Sinsi durum: ağırlıklar sessizce 0/255'e kayarsa sonuç makul görünür, ama kollider
  yüzeyi bir hücre kalınlaşır ya da incelir. İnce duvarlı mesh'te (kutu kabuğu) suyun duvardan
  geçmediğini kontrol et.

**11:48 derlemesi (ajan, 2026-11-12): kar yanında fırlayan su ÇÖZÜLDÜ.**
- Kök: gözenek sürüklemesi. Su yüz kütleleri yalnız sıvı adımında yenileniyor, parseller ise
  her MPM alt adımında ilerliyor. Sıçrama parselleri kütlesi ~0 kalmış yüzlere giriyor; sürükleme
  o yüzü kar hızına çekiyor; FLIP/delta bunu parsele tam kütlesiyle aktarıyor. Momentum
  korunmuyor, 50 m/s tavanına pompalanıyordu.
- Düzeltme: `sim_matter_liquid_drag.glsl` yalnız su kütlesi ≥ %5 · ρ_sıvı·h³ olan yüzlerde.
  Push 32 → 48 B, iki kayıt satırı güncellendi.
- Eleme sırası: sürükleme kapalı / yalnız su yazımı kapalı → yok; gözenekli, tepki, delta kapalı
  → var. Denenip sökülenler: φ ≥ %5 kapısı, MPM hız kelepçesi, sıvı adımında sürüklemeyi
  atlamak, MPM malzeme (FLIP taban) hızı.
- bal: en yüksek hız 9.1 m/s (önce 44–50), sıvı alt adımı 17–36 (önce 180'e kadar), sıçrama
  karesi 0.55–0.72 s (önce 1.6–2.4 s).
- Testler: smoke 30/30, multirate PASS, sıcak su + kar PASS.
  `mpm_porous` floats/near_surface ve `liquid_drag` converged FAIL, ama düzeltmesiz eski
  shader'la da aynı FAIL; önceden var olan açık konular.

**10:36 derlemesi (ajan, 2026-11-12): hareketli kollider dinlenik karı uyutmuyordu.**
- `MatterMpmSleep.cpp`: domain'in HERHANGİ bir yerinde hareketli kollider (solid_vel ≠ 0)
  dinlenik karı uyanık tutuyordu; voksel imzası her karede değiştiği için dinlenme sayacı da
  her karede sıfırlanıyordu.
- Artık yalnız parsel kutusunun yakını (3 hücre + 2 × kollider kare kayması) ve oraya varacak
  ayak izi sayılıyor.
- dem1: temastan önce 150 → **55 ms** (uykuda), kollider ~0.65 m kala uyanıyor, temas
  davranışı aynı.
- Smoke 30/30, `collider_phi` 12/12.
- ★ Sessiz başarısızlık: hızlı bir kollider uyuyan karın içine bir kare geç girerse kar o kare
  tepkisiz kalır. Bakılacak: hızlı kollider (> 1 hücre/kare) karın içine dalarken ilk temas
  karesi.

**00:13 derlemesi (ajan, 2026-11-12): hareketli kollider maliyeti.**
- `computeSolidFaceWeights` tek iş parçacığındaydı. Hareketli 2.5 m mesh-SDF küre her karede
  ~3M SDF örneği çekiyordu: 205 ms/kare, karla temastan bağımsız.
- Üç yüz döngüsü OpenMP ile paralelleştirildi (her yüz tek yazıcı, sonuç aynı): dem1'de
  **205 → 24 ms**.
- ★ Smoke testleri bu derlemede KOŞULMADI (açık sahne vardı). Sırada: `mpm_collider` ve
  `collider_phi` testleri.
- Not: dem1'deki tek seferlik seed projeyle kaydedilmiyor; proje yeniden açılınca domain boş.
- Sonraki kaldıraç (yazılmadı): yüzeyden uzak yüzde 16 yerine 1 örnek (SDF mesafesiyle) ya da
  GPU'ya taşıma.

**23:57 derlemesi (ajan): aşırı soğumanın iki kalan kökü kapandı + sıçrama maliyetinin kökü bulundu.**
- Bal sahnesinde geçici sondayla bulundu (sökülen `THERMAL-PROBE`):
  - (a) Kuru havada buharlaşmanın çiğ noktası ~0 K: buna **yaş termometre tabanı** eklendi
    (psikrometrik; kuru 293 K → 277.6 K, %50 → 286.2 K). CPU ve GPU.
  - (b) `conductThermalLiquid` hücre ΔE'sini her parsele TEKDÜZE ekliyordu: karışık hücrede
    kar parseli komşularının altına iniyordu. Artık yalnız yeni ortalamanın öbür yanındaki
    parsellerden, c_p|T − T_yeni| ağırlığıyla dağıtılıyor (enerji birebir, sınır aşılmıyor).
    Hücre içi düzeltme de aynı kuralla.
  - Sonuç: bal sahnesi 60 kare boyunca T_min = 263.1 K (karın kendisi). Donma hâlâ var
    (kare 60'ta 865 parsel). Su −10 °C karla karışıp donuyor; fiziksel olabilir, erime hızıyla
    birlikte bakılacak.
- **Sıçrama maliyeti (kare 22–48, 1–2.4 s):**
  - Birkaç SU parseli tabanda, kar yığınının olduğu köşede 20–50 m/s'ye fırlıyor
    (`SPEED-PROBE`, hâlâ kodda, yalnız > 10 m/s'de log yazar).
  - Sıvı CFL'si 17 → 180 adım, viskozite PCG kare GPU'sunun ~%65'i.
  - Termal kapalı da aynı; collider yapışması 0 da aynı; **kar kaynağı kapalıyken YOK**
    (vmax 5 m/s, 16 adım) → kök su–kar bağında. Sürükleme örtük, aday değil.
  - Adaylar: gözenekli projeksiyon / basınç tepkisi / aralık sonu saf-delta G2P. AÇIK.

**23:10 derlemesi (ajan): termal duraklama + Celsius.**
- Kök: sıcak su buharlaşmada tek adımda ~30 K'ye soğuyup donuyordu; donmuş parsel karma GPU
  adımını durduruyordu.
- Düzeltmeler:
  - Buharlaşma geriye Euler (CPU `GasSurfaceExchangeCpu.cpp` + GPU
    `sim_inventory_surface_exchange.comp`).
  - Donmuş parseller şerit dışında sabit (`kMatterGpuHeldModel`).
- Test `rt_test_matter_thermal_hot_water_snow_ipc.py`: 353 K su + kar, termal açık, en düşük
  T = 263 K, bekletilen kare yok. Smoke 30/30, multirate 7/7.
- Panel: mutlak Kelvin alanları "353.0 K (79.9 C)" gösteriyor (`UI/TemperatureLabel.h`).
- Bakılacaklar:
  1. Bal sahnesinde termal açık + özel su sıcaklığı: sim durmamalı, `thermal_min_kelvin`
     karın sıcaklığının altına inmemeli.
  2. Panelde sıcaklık alanları Celsius'u parantezde göstermeli (termal paneli, seed kaynağı,
     collider MSF, gaz, madde düzenleyici).
  - ★ Donmuş parsel yolu (gerçek donma) bu testte TETİKLENMEDİ; kar yanında su gerçekten
    donarsa sabit kalmalı ve sim devam etmeli.

**22:39 derlemesi (ajan): kar ölçülen sertleşmeyle boyutlanıyor.**
- Kar, tavan (×5) yerine kare başında ölçülen en sert parsel × 1.5 ile boyutlanıyor. Kare
  aşarsa tavanla yeniden koşuluyor.
- 1M kar 1.76 → 1.22 s/kare.
- Bakılacak: kar sahnesinde çarpma ve sıkışma davranışı öncekiyle aynı mı.
  ★ Sessiz başarısızlık: çarpma karesinde kar hafif "yumuşak" görünür (fazla sıkışma).
  Ölçüt: düşüş testinde J_p en küçük ≈ 0.54, ortalama ≈ 0.92 (tavanla koşuyla aynı).

0. Uygulamayı kendin aç, pencereyi arka plana al ve bir IPC testi koş: artık yavaşlamamalı.
   Bozuksa (ping ~500 ms): uyandırma olayı ulaşmıyor.

**Kullanıcının kendi sahnesinde bakması gerekenler (sırayla):**
1. Bal/kar sahnesini aç ve oynat: kar ve su davranışı öncekiyle aynı olmalı.
   Bozuksa: blok P2G karo dışı katkıyı ya da FLIP taban çizgisini yanlış topluyor. Uygulamayı
   `RT_P2G_BLOCK_OFF=1` ortam değişkeniyle aç; davranış düzeliyorsa kök blok yolu.
2. Sahnede kar, kum ya da elastik var ve "Rendered" (Vulkan RT) modunda yüzey düzgün
   görünüyor: yüzey merkez çekirdeği değişti, parite testi geçti.
   ★ Sessiz başarısızlık: yüzey biraz pürüzlü ama "makul" görünür; parite sayıları bunu yakalar,
   göz yakalamaz.
3. Eski bir projeyi aç: domain bütçesi kayıtlı değerini korumalı.

Ayrıntı: `docs/dev/OLCEK_MILYON_PARCACIK.md` §3.3–3.4.
