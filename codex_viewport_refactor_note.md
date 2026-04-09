# Solid Viewport Refactor Note

Amac:
Solid/Matcap/MaterialPreview viewport'unu secili render device'dan ayirmak.
Rendered modu OptiX / Vulkan RT / CPU backend'e bagli kalabilir, ama edit viewport kendi raster backend'i ile yasamali.

Tamamlanan ana isler:
- `IViewportBackend` eklendi.
- `g_viewport_backend` artik `std::unique_ptr<Backend::IViewportBackend>` olarak tutuluyor.
- `VulkanViewportBackend` eklendi ve `Main.cpp` bunu dedicated viewport backend olarak olusturuyor.
- `getRasterViewportBackend()` artik once `g_viewport_backend` donduruyor; sadece fallback olarak `g_backend` uzerinden `IViewportBackend` cast ediyor.
- `VulkanBackendAdapter` icindeki interactive viewport helper'lari `virtual ...Impl` seam'e acildi.
- Su implementasyonlar `VulkanViewportBackend.cpp` tarafina tasindi / override edildi:
  - `ensureInteractiveViewportResourcesImpl`
  - `destroyInteractiveViewportResourcesImpl`
  - `renderInteractiveViewportImpl`
  - `setInteractiveViewportMatcapImpl`
  - `setInteractiveViewportMatcapPresetImpl`
- `VulkanViewportBackend.cpp` icine yerel SPIR-V loader eklendi; artik `loadSPV`, `MAX_PATH`, `GetModuleFileNameA` gibi `VulkanBackend.cpp` veya Windows API bagimliliklari yok.
- `VulkanBackend.cpp` tarafinda bazi raster sync kararlarinda eski `m_viewportMode == Solid/Matcap` kontrolleri `shouldUseInteractiveViewport()` tabanina cekildi.

UI / sync tarafinda yapilan kritik ayrismalar:
- `scene_ui_mesh_overlay.cpp`
  - `processPendingMeshEditGpuSync()` artik viewport raster sync ile render backend sync'ini ayri yurutuyor.
  - `ctx.backend_ptr` viewport backend olabilir; render backend olarak artik `g_backend.get()` kullaniliyor.
  - `g_viewport_backend` varsa ona ayri raster patch/update gidiyor.
  - Raster mesh yoksa fallback olarak `buildRasterGeometry(...)` cagrilip patch tekrar deneniyor.
  - OptiX targeted sync artik `ctx.backend_ptr` degil `g_backend.get()` uzerinden cozuluyor.
- `scene_ui_terrain.hpp`
  - Terrain sculpt / paint sync yolunda RT partial update ile viewport raster update ayrildi.
  - `g_backend` Vulkan RT backend ise `updateTerrainBLASPartial(...)` oraya gidiyor.
  - Raster / Solid viewport sync ise `g_viewport_backend` veya fallback `ctx.backend_ptr` uzerinden gidiyor.
- `scene_ui_selection.cpp`
  - Bazi stale `use_vulkan` rebuild kapilari temizlendi.
- `scene_ui_gizmos.cpp`
  - Rebuild kararlari artik `use_vulkan/use_optix` yerine gercek backend nesnelerine bakiyor.
- `scene_ui.cpp`
  - VDB append sonrasi rebuild karari eski flag yerine gercek Vulkan backend / viewport backend varligina gore veriliyor.
- `scene_ui_lights.cpp`
  - World/LUT sync yardimcisi dosya scope'una tasindi.
  - Gerekli yerlerde `g_viewport_backend` de world sync aliyor.

Onemli mimari not:
- `ctx.backend_ptr` her zaman render backend degil. Solid/Matcap/MaterialPreview sirasinda bu pointer viewport backend olabilir.
- Render device ile ilgili sync / acceleration / RT guncellemeleri gerekiyorsa oncelikle `g_backend` dusunulmeli.
- Viewport raster / interactive shading / matcap / solid cache isleri icin oncelikle `g_viewport_backend` dusunulmeli.
- UI'da hangi render aygiti seciliyse sadece o render backend scene verisiyle sicak tutulmali.
- Secili olmayan render backend icin her frame / her edit / her sculpt sync yapilmamali.
- Render device degisince:
  - eski render backend'in scene cache / BLAS / TLAS / descriptor / instance state'i ya tamamen birakilmali ya da stale kabul edilmeli
  - yeni secilen render backend scene'den full sync alarak tek gecerli render backend state'i olmali
- Solid mod icin gerekli olan sey "tum render backend'leri guncel tutmak" degil:
  - `g_viewport_backend` raster state'ini guncel tutmak
  - aktif render backend degisecekse o anda yeni backend'i tam sync etmek

Onerilen hedef mimari:
- `g_viewport_backend`
  - Solid / Matcap / MaterialPreview raster verisi
  - her zaman ayri ownership
  - mesh/terrain/live edit sync burada kalmali
- `g_backend`
  - o an UI'da secili tek render aygiti
  - OptiX veya Vulkan RT veya CPU render backend
  - sadece bu backend incremental/full render sync almali
- Secili olmayan backend'ler
  - scene mutation aninda hic sync almamali
  - backend switch oldugunda lazy-create veya full-resync ile aktive olmali

Onerilen bayrak ayrimi:
- `g_viewport_raster_rebuild_pending`
  - sadece `g_viewport_backend` icin
- `g_render_backend_rebuild_pending`
  - aktif `g_backend` icin genel render rebuild istegi
- Istege gore backend-spesifik alt durumlar:
  - `g_optix_rebuild_pending`
  - `g_vulkan_rebuild_pending`
  - ama bunlar sadece aktif backend o aygit ise anlamli olmali

Buffer/sahne senkron prensibi:
- Mesh/terrain/scatter/water/vdb gibi scene mutasyonlarinda:
  - once viewport raster gerekiyorsa `g_viewport_backend` sync edilir
  - sonra sadece aktif render backend sync edilir
  - aktif olmayan render backend icin hicbir sey yapilmaz
- Backend switch oldugunda:
  - yeni backend full `updateGeometry + materials + volumes + lights + camera + resetAccumulation`
  - eski backend stale/stateful kalacaksa render edilmeyecegi garanti edilmeli
  - ideal durumda eski backend release/shutdown veya en azindan "inactive cache" moduna alinmali

Solid -> Rendered gecisinde dogru davranis icin onerilen kural:
- Solid mod boyunca sadece `g_viewport_backend` zorunlu olarak guncel kalir.
- Eger secili render backend stale ise Rendered'a gecis aninda bir kere full sync/rebuild yapilir.
- Bu rebuild viewport raster rebuild ile ayni bayragi paylasmamali.

Su anki durum:
- Derleme basariliydi son bildirilen durumda.
- Interactive viewport resource/render/matcap akisi buyuk olcude `VulkanViewportBackend` sinirina cekildi.
- UI tarafinda davranisi etkileyen bazi eski `use_vulkan` varsayimlari temizlendi.
- Buna ragmen sculpt/sync tarafinda halen davranis bozukluklari var.
- Bu turda yeni bir ayrim daha yapildi:
  - `Main.cpp` icine `g_viewport_raster_rebuild_pending` eklendi.
  - Frame-end rebuild akisinda viewport raster rebuild ile Vulkan RT full rebuild birbirinden ayrilmaya baslandi.
  - Mesh sculpt raster sync yolunda viewport backend icin `resetAccumulation()` eklendi.
  - Mesh/terrain sculpt raster sync basarisizsa artik `g_viewport_raster_rebuild_pending` set ediliyor; sadece render backend Vulkan ise ayrica `g_vulkan_rebuild_pending` set ediliyor.
  - Viewport backend init / backend switch / shading mode degisiminde eski `g_vulkan_rebuild_pending` yerine viewport rebuild icin yeni bayrak kullanilmaya baslandi.

Bugunku sonuc:
- Solid mod artik oncekine gore daha izole, ama TAM izole degil.
- Ana raster draw/resource yolu artik dedicated viewport backend'e dayaniyor.
- Buna karsin scene mutation sonrasi rebuild scheduling katmaninda halen render-backend'e bagli kalan noktalar var.
- Kullanici testinden gelen yeni semptomlar:
  - Solid modda terrain eklendikten sonra hemen gorunmuyor; Rendered'a gecip geri donunce geliyor.
  - Terrain sculpt canli degil; degisim mouse birakinca gorunuyor.
  - New Project sonrasi eski terrain solid modda gorunur kalabiliyor.
  - Solid modda add/import ile eklenen normal mesh objeler de hemen gorunmeyebiliyor.
- Bu semptomlar icin bu turda yapilan duzeltmeler:
  - Terrain create/delete/clear yollarinda viewport icin `g_viewport_raster_rebuild_pending` set edilmeye baslandi.
  - Terrain live sculpt sirasinda viewport raster sync, RT backend seciminden bagimsiz hale getirildi.
  - OptiX seciliyken bile terrain sculpt artik live olarak viewport raster backend'e sync olmaya calisiyor.
  - New Project akisinda eski `g_vulkan_rebuild_pending` yanina viewport rebuild icin de yeni bayrak set ediliyor.
  - Asset append/import ve procedural add yollarinda da viewport raster rebuild bayragi set edilmeye baslandi.
  - Bu akislarda `ctx.backend_ptr` yerine mumkun oldugunca aktif render backend (`g_backend`) sync edilip viewport rebuild ayri tutuluyor.
  - Import yolunda ek bir hata bulundu:
    - bazi import akislarinda `Renderer::create_scene(...)` ve `ProjectManager::importModel(...)` cagrilarina `ctx.backend_ptr` veriliyordu
    - Solid modda bu pointer viewport backend olabildigi icin import pipeline yanlis backend baglaminda ilerliyordu
    - bu turda import yolu aktif render backend (`g_backend`) uzerine cekildi
  - Main loop'taki scene-loading/import finalize blogunda da render backend sync'i `ui_ctx.backend_ptr` yerine `g_backend` uzerinden yapilmaya baslandi.
  - Yeni tespit:
    - Vulkan RT mesh edit incremental update yolu transform-handle kullanan objelerde local/world space ayrimini bozuyor olabilir.
    - Belirti: CPU/OptiX dogru gorurken Vulkan RT'de mesh edit vertexleri beklenenden farkli yere gidiyor.
  - Sonraki turda bu koruma daha hedefli hale getirildi:
    - `VulkanBackend.cpp::updateMeshBLASPartial()` icinde hedef BLAS secimi sadece node name ile degil
      ilk triangle pointer'i + triangle count ile de guclendirildi.
    - Boylece ayni/adimsi node name tasiyan farkli instance/BLAS'lara yanlis patch gitme ihtimali azaltildi.
    - Bunun ardindan mesh edit icin Vulkan incremental yol tekrar acildi; performans fallback'ten geri alinmaya calisiliyor.
  - Beyaz viewport icin ek onlem:
    - Solid -> Rendered gecisinde yapilan senkron Vulkan rebuild sonrasi `g_vulkan_rebuild_pending` temizleniyor.
    - Amaç ayni frame'de ikinci/full tekrar rebuild tetiklenip beyaz viewport olasiligini arttirmamasini saglamak.
  - Vulkan gecis yavasligi / kararsizlik icin ek ayrim:
    - `VulkanBackend::updateGeometry()` icindeki eski
      `shouldUseInteractiveViewport() || !m_rasterMeshes.empty()` fallback'i daraltildi.
    - Vulkan RT backend artik Rendered modda kendi raster mesh cache'ini rebuild etmiyor.
    - Eger elinde eski raster cache kaldiysa temizleniyor; raster sahipligi dedicated viewport backend'e birakiliyor.
    - Amaç:
      - Vulkan RT backend'e gereksiz solid/raster geometri upload etmemek
      - backend switch maliyetini dusurmek
      - RT backend ile viewport backend arasindaki cift-state kararsizligini azaltmak

Halen acik kalan problemler:
- Mesh sculpt bazen Solid modda ilk darbede objeyi sahneden kaldiriyor / gizliyor gibi davraniyor.
- Solid modda sculpt firca tepkisi olsa bile mesh her zaman dogru canli geometri guncellemesini gostermiyor.
- Render device Vulkan seciliyse davranis daha iyi; bu da bazi yollarin hala render-device-secimine bagimli kaldigini gosteriyor.
- Vulkan RT Rendered moduna gecince sahne bazen geliyor ama kamera hareketinde siyah ekran olabiliyor.
- `g_vulkan_rebuild_pending` artik eskiye gore daha dar anlamda kullaniliyor, ama kod tabaninin baska bolumlerinde halen viewport rebuild yerine de set ediliyor olabilir.

Tam izolasyon icin halen temizlenmesi gereken bagimliliklar:
- `scene_ui_selection.cpp`
  - delete / selection-driven mutation sonrasi `g_viewport_backend` var diye dogrudan `g_vulkan_rebuild_pending` set eden yollar var.
  - Bunlar viewport raster rebuild pending ve render backend rebuild pending olarak ayrilmali.
- `scene_ui_gizmos.cpp`
  - gizmo release sonrasi `active_interactive_viewport_backend` kontrolunde sadece `g_vulkan_rebuild_pending` set ediliyor.
  - Solid/Matcap icin burada `g_viewport_raster_rebuild_pending` tercih edilmeli.
- `scene_ui.cpp`
  - VDB append/import gibi akislarda `g_viewport_backend` varligi ile `g_vulkan_rebuild_pending` ayni sepete atiliyor.
  - Volumetric / append senaryolarinda viewport rebuild ile RT rebuild ayrimi tekrar gozden gecirilmeli.
- `SceneCommand.cpp`
  - command/undo/redo mutasyonlari halen sadece `g_vulkan_rebuild_pending` uzerinden rebuild schedule ediyor.
  - Bu katman genel scene mutation merkezi oldugu icin yeni ayrimin en kritik eksik parcasi burasi olabilir.
- Backend switch akisinin nihai hedefi:
  - aktif backend disindakilere live scene sync yok
  - switch aninda tam resync var
  - Solid/Rendered gecis mantigi "tum backend'leri sicak tut" degil "viewport sicak, aktif render backend gerekince sicak" olmali
- `globals.h`
  - yeni `g_viewport_raster_rebuild_pending` extern'i eklenmediyse ortak/global kullanim ihtiyacinda derleme veya ileride bakim sorunu cikarabilir.

Bir sonraki chat icin onerilen ilk giris noktasi:
1. `scene_ui_mesh_overlay.cpp::processPendingMeshEditGpuSync()` tekrar incelensin.
2. `SceneCommand.cpp`, `scene_ui_selection.cpp`, `scene_ui_gizmos.cpp`, `scene_ui.cpp` icinde `g_vulkan_rebuild_pending` set eden yerler yeni modele tasinsin.
3. Mesh sculpt sirasinda:
   - viewport raster patch/update
   - render backend incremental update
   - full rebuild pending bayraklari
   birbirinden net ayrilsin.
4. `scene_ui_terrain.hpp` icindeki terrain sculpt / commit yollarinda benzer stale rebuild mantigi var mi tekrar kontrol edilsin.
5. Gerekirse su ayrim her yerde zorunlu hale getirilsin:
   - viewport raster rebuild pending
   - Vulkan RT full scene rebuild pending

Dikkat edilmesi gerekenler:
- Ghost object fix bozulmamali.
- `g_backend` ve `ctx.backend_ptr` birbirine karistirilmamali.
- `g_viewport_backend` varsa Solid/Matcap/MaterialPreview path'i oraya akmali.
- OptiX ve Vulkan RT icin Rendered backend sync'i viewport update basarili oldu diye erken `return` ile atlanmamali.
- Amac sadece derlemek degil; Solid sculpt ve Rendered'a gecis davranisi stabil olmali.

Yeni chatte devam etmek icin iyi acilis cumlesi:
`codex_viewport_refactor_note.md notundaki son duruma gore mesh/terrain sculpt ile Vulkan RT siyah ekran sorununu devam ettir`

2026-04-03 ek not:
- Vulkan RT Rendered modunda kamera cok kucuk hareket edince sahne verisinin kaybolup beyaz alana dondugu yeni bir belirti raporlandi.
- Inceleme sonucu iki zincir bulundu:
  - `resetAccumulation()` GPU output/variance goruntulerini temizliyor ve `m_forceClearOnNextPresent = true` ile host framebuffer'i de zorla sifirlatiyordu.
  - `renderProgressiveImpl()` icinde `!isRTReady() || !hasTLAS()` durumunda dogrudan `presentBackgroundOnly()` cagriliyordu.
- Bu kombinasyon, kamera hareketi sonrasi RT/TLAS hazirligi bir frame gecikirse son gecerli render yerine beyaz/background-only goruntu dusmesine yol acabiliyordu.
- Uygulanan dar koruma:
  - Vulkan Rendered modunda `m_forceClearOnNextPresent` host clear'i artik hemen uygulanmiyor; yalniz interactive viewport yolunda anlik clear devam ediyor.
  - Rendered modda ilk basarili RT frame gelene kadar son host goruntusu korunuyor ve o anda bayrak temizleniyor.
  - `!isRTReady() || !hasTLAS()` durumunda artik sadece ilk frame/sample yoksa background-only present yapiliyor; once render alinmisken gecici readiness kaymasinda mevcut goruntu korunuyor.
- Beklenen etki:
  - Vulkan RT sahne ilk dogru gelirken kamera hareketinde beyaza dusme sikligi azalacak.
  - Bu patch correctness/stability odakli; kok neden muhtemelen hala RT readiness / TLAS hayat dongusunde.

2026-04-03 daha sonraki ilerleme:
- Refactor sirasinda bazi dosyalarda `extern` bildirimleri anonymous namespace icine dusup
  yanlis internal-linkage uretiyordu.
- Bu siniftaki compile sorunlari temizlendi:
  - `SceneCommand.cpp`: `g_bvh_rebuild_pending`
  - `scene_ui_water.hpp`: `g_viewport_raster_rebuild_pending`, `g_optix_rebuild_pending`, `g_vulkan_rebuild_pending`
  - `scene_ui_scatter.cpp`: ayni rebuild bayraklari
  - `Main.cpp`: `g_backend` icin helper-oncesi global `extern` bildirimi namespace disina alindi
- Yani son rebuild hatalari daha cok refactor artigi declaration/linkage kaynakliydi; mimari karar degil.

Solid -> Vulkan RT Rendered gecisi icin ek sertlestirme:
- `Main.cpp` icindeki `solid/matcap -> rendered` Vulkan gecis blogu eskiden agirlikla
  `rebuildAccelerationStructure + updateGeometry` yapiyordu.
- Bu, OptiX -> Vulkan backend switch kadar tam bir render-backend scene push degildi.
- Gecis blogu su ek sync'lerle genisletildi:
  - `syncVDBVolumesToGPU(ui_ctx)`
  - `updateBackendGasVolumes(scene)`
  - `uploadHairToGPU()`
  - `updateBackendMaterials(scene)`
  - `syncCameraToBackend(*scene.camera)`
  - `setLights(scene.lights)`
  - `setWorldData(&wd)`
  - `uploadAtmosphereLUT(al)`
- Hipotez:
  - OptiX -> Vulkan gecisi temiz cunku tam backend-switch yolu zaten full sync yapiyor.
  - Solid -> Vulkan RT Rendered gecisinde ise viewport backend sicak ama render backend stale kalabiliyor;
    ilk Rendered frame geometry dogru gelse bile world/light/gas/camera/LUT/deskriptor tarafi eksik olabiliyor.

Vulkan beyaz viewport sorununda yeni bulgu:
- Kullanici testi ile sorun daha daraltildi:
  - Vulkan RT beyaza dusme her zaman olmuyor.
  - Sorun ozellikle `viewport denoiser` acikken tekrar uretilebiliyor.
  - `viewport denoiser` kapaliyken semptom belirgin sekilde azaliyor / kayboluyor.
- Bu, RT readiness/TLAS gecikmesi disinda bir ikinci zincir oldugunu gosteriyor:
  - Vulkan backend AOV/denoiser frame uretimi
  - Main loop'ta `getDenoiserFrame()` + OIDN apply zamani
  - kamera hareketi sonrasi reset/ilk sample anlari

Bu bulgu icin uygulanan dar korumalar:
- `VulkanBackend.h/.cpp`
  - `m_hasPresentedRenderedFrame` eklendi.
  - AmaÃ§:
    - `m_currentSamples == 0` oldugu her durumu "ilk kez hic frame yok" saymamak.
    - Kamera hareketi sonrasi resetten sonra RT/TLAS bir frame gec hazir olursa son gecerli host frame korunabilsin.
  - `renderProgressiveImpl()` icinde:
    - `!isRTReady() || !hasTLAS()` ise background-only present artik sadece
      `m_currentSamples == 0 && !m_hasPresentedRenderedFrame` durumunda yapiliyor.
    - Bir basarili Rendered frame indirildiginde `m_hasPresentedRenderedFrame = true` oluyor.
    - Device init / shutdown / output resize gibi gercekten "ilk frame'e donen" durumlarda bu bayrak sifirlaniyor.

2026-04-03 delete / tombstone ek not:
- Obje silmede yeni semptom:
  - Solid modda silinen obje bazen viewportta kalmaya devam ediyor.
  - Vulkan RT'ye gecince ayni obje silinmis gorunuyor.
  - OptiX'te silinen obje bazen hicbir backend tarafinda silinmis kabul edilmiyor.
  - Pathtrace modlarda delete sonunda gereksiz rebuild hissi olusuyor.
- Kok neden:
  - `editor_pending_delete_object_names` ile soft-delete/tombstone modeli baslatildi ama
    backend switch / viewport raster rebuild / render backend rebuild sonrasinda bu set merkezi olarak yeniden uygulanmiyordu.
  - Bu yuzden hangi backend en son full sync aliyorsa delete state'i onun uzerinde kaybolabiliyordu.
- Yeni duzeltme:
  - `Main.cpp` icine `applyPendingDeleteVisibilityToBackend(...)` yardimcisi eklendi.
  - Bu helper su noktalardan sonra cagriliyor:
    - backend switch sonrasi yeni `g_backend` full scene sync'i bitince
    - `g_viewport_backend` scene/world/camera sync'i bitince
    - OptiX async rebuild tamamlaninca
    - Solid viewport `buildRasterGeometry(...)` tamamlaninca
    - Vulkan RT `updateGeometry(...)` / full rebuild tamamlaninca
  - Boylece tombstone visibility artik rebuild/switch sonrasi tekrar uygulanmis oluyor.
- Delete scheduling tarafi da yumusatildi:
  - `scene_ui_selection.cpp` ve `SceneCommand.cpp` object delete/undo yolunda
    GPU render backend aktifse artik dogrudan CPU BVH async rebuild zorlanmiyor.
  - Bunun yerine:
    - obje visibility hemen backend'lere itiliyor
    - viewport backend varsa `g_viewport_raster_rebuild_pending` set ediliyor
    - CPU tarafi icin `g_cpu_sync_pending` ile "gerektiginde sync et" davranisina geciliyor
    - CPU backend aktifse yine `g_bvh_rebuild_pending` kullaniliyor
- Beklenen etki:
  - Delete state'i Solid / Vulkan RT / OptiX arasinda daha merkezi ve tutarli olacak.
  - GPU pathtrace modlarda delete sonrasi gereksiz full rebuild zinciri azalacak.
  - Solid'de obje gorunurlugu rebuild/switch sonrasi tekrar canlanmamali.
- `Main.cpp`
  - Vulkan aktifken viewport denoiser icin ek guard eklendi:
    - kamera yeni hareket ettiyse (`camera_moved_recently`)
    - veya aktif drag varsa
    - veya sample sayisi henuz `<= 1` ise
    - viewport OIDN denoiser uygulanmiyor
  - AmaÃ§:
    - kamera hareketi sonrasi bos / yari hazir AOV'larla host goruntuyu ezmemek
    - Vulkan RT + viewport denoiser acikken beyaza dusme ihtimalini azaltmak

Bu yeni Vulkan/denoiser korumalari icin not:
- `m_hasPresentedRenderedFrame` korumasi dusuk riskli ve genel stabilite icin tutulmali.
- `Main.cpp` tarafindaki Vulkan viewport denoiser guard'i daha deneysel:
  - Sorunu keserse tutulabilir ama davranis/reaksiyon hissi bozarsa daha sonra yumusatilabilir.
  - Ornegin `sample_count > 1` esigi ya da `camera_moved_recently` penceresi yeniden ayarlanabilir.

Su an icin en guncel test odagi:
1. Vulkan RT sec
2. Viewport denoiser ACIK test et
3. Kamera kucuk hareket
4. Sonra ayni testi denoiser KAPALI test et
5. Fark belirginse bir sonraki hedef `getDenoiserFrame()` / AOV-ready dogrulamasi olmali

Su an acik kalan ana problem:
- Vulkan RT + viewport denoiser kombinasyonunda beyaza dusme kok sebebi tam cozulmus degil;
  sadece semptomu daraltan korumalar eklenmis durumda.
- Eger halen tekrar ederse bir sonraki mantikli giris noktasi:
  - `Main.cpp`deki OIDN apply kapisi
  - `VulkanBackend::getDenoiserFrame()`
  - denoiser AOV'larin kamera resetinden hemen sonra gecerli olup olmadiginin dogrulanmasi

2026-04-03 selection regression notu:
- Selection/picking performansini iyilestirme amaciyla `scene_ui_selection.cpp` icinde
  `rebuildTriToIndex()` odakli hafif cache ve `resolvePickedTriangleFromScene(...)` gibi ek fallbackler denendi.
- Sonuc:
  - ilk secimden sonra hizlanmak yerine her secim daha pahali hale geldi
  - bazi asamalarda secim tamamen bozuldu
  - Solid modda realtime transform / tasima da olumsuz etkilendi
- Karar:
  - bu optimizasyonlar geri alindi
  - selection akisi tekrar "ilk tıkta gerekirse tam `rebuildMeshCache()`, sonra stabil secim" modeline donduruldu
  - delete/tombstone/backend visibility duzeltmeleri korunuyor; sadece picking optimizasyon regresyonlari geri alindi
