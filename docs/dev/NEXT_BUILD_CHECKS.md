# NEXT BUILD CHECKS

## ★ TAM REBUILD — taşınan nesne + geri sarma geçmişi — 2026-11-05 öğle (YAZILDI, derlenmedi)

**★ Tam Rebuild:** `SceneData` büyüdü (`MeshVolumeCacheEntry` poz/sınır alanları,
`sim_rewind_history_`) ve `rtapi::SimControlStateInfo` (`rewind_history`).

**Kullanıcı (canlı):**
- ✔ İlk Play'de seed artık kaybolmuyor.
- ✔ Mesh hacmi doğru doluyor (primitive'lerle denendi).
- ✘ Seed'den önce nesne taşınırsa eski konum dolduruluyordu.
- `last_rewind` okundu: 43. kare, `fluid 'Physics Domain 1' thermal/solver/seed`, `absorbed: false`.
  Bu büyük ihtimalle kullanıcının sonraki bir düzenlemesi; ilk Play'deki kaymanın kaydının
  üzerine yazdı.

**Kök (taşınan nesne):** `SurfaceMeshCache.version` her yeniden kurulumda aynı sahne çapı sayacı.
Taşıma onu artırmıyor. Hacim testi önbelleği sürüm + üçgen sayısına bakıyordu, bu yüzden eski
pozdaki test kullanılıyordu.

**Yazılan:**
- Önbellek anahtarına nesnenin dönüşüm matrisi ve yüzey sınırları da girdi.
- `sim.control_state.rewind_history`: son 8 olay, eskiden yeniye. Sonraki bir düzenleme önceki
  kaymanın kaydını silmiyor.
- Akışkan imza bölümü ikiye ayrıldı: `fluid 'X' thermal` ve `fluid 'X' solver/seed`.
- Teste `moved_object` kapısı eklendi: küre 0,4 m taşınınca `seed_top_y` 0,4 m kaymalı.

**Sırayla:**
1. `python scripts/test/rt_test_seed_mesh_volume_ipc.py` → **6/6** (`moved_object` ve
   `not_truncated` yeni).
2. **Elle:** projeyi aç, Seed, Play. Ajan `rewind_history`'yi okur.
   - `absorbed: true` olan satırın `first_changed_part`'ı, ilk Play kaymasının bölümünü söyler.
   - O bölümdeki türetilmiş alan imzadan çıkarılır ve koruma sökülür.
   - Hiç satır yoksa kayma artık oluşmuyor demektir. O durumda kök büyük ihtimalle önceki
     partilerden birinde kapandı ve koruma güvenle sökülebilir.

## Mesh Volume seed — canlı sonuç ve küçük düzeltme — 2026-11-05 (düzeltme YAZILDI, artımlı)

**Canlı (kullanıcı tam Rebuild aldı, boş sahne): `rt_test_seed_mesh_volume_ipc.py` 4/4 PASS.**
- **Su:** 0,5 m kürede 8 PPC ile 33 086 parçacık doğdu; beklenen 33 510 (−%1,3, kürenin
  üçgenlere bölünmesinden). `seed_top_y` 0,4968; küre tepesi 0,5.
- **Gizli nesne:** birebir aynı sayı. Kum kale akışı gizli mesh'le çalışıyor; açık risk kapandı.
- **Kum (DEM):** 5 102 tane, tepe 0,448.
- Test kendi domain'ini ve küresini temizledi.

**Bulunan kusur, düzeltildi:** `seed_requested_particles` 42 624 idi. Kısmen içeride kalan sınır
hücreleri tam sayılıyordu; bu da `seed_truncated`'ı ve yanlış "bütçe yetmedi" uyarısını tetikler.
Mesh seed'inde kesilme artık bütçenin kendisinden okunuyor. Kesilme yoksa `requested = emitted`.
Teste `not_truncated` kapısı eklendi.

**Derledikten sonra:** test 5/5 olmalı. `not_truncated` FAIL ise `MatterDomainSynchronization.inl`
içinde `mesh_seed` dalı.

**Hâlâ bekleyen (üstteki "imza kayması" partisi):** projeyi aç, Seed, Play, ardından
`sim.control_state.last_rewind`'ı oku. Bu oturumda henüz tetiklenmedi (`null`).

## ★ Mesh Volume seed (kum kale) — 2026-11-05 (DERLENDİ, CANLI 4/4 — üstteki bölüm)

**★ Kullanıcının o sırada süren derlemesinin ORTASINDA yazıldı.** O derleme bazı dosyaları eski,
bazılarını yeni haliyle almış olabilir. Bu parti için **yeniden tam Rebuild** al. `SceneData` ve
`SimulationGridDomainDesc` büyüdü (`fluid_seed_mesh_object`, `mesh_volume_cache_`), ayrıca
`SimulationSeedStats` (`error`).

**Yazılan:**
- `Fluid/MeshVolumeTest.h` (yeni, vcxproj'da):
  - Nesnenin dünya üçgenleri üzerinde `ColliderMeshBVH` ve üç ışınlı parite oylaması.
    Collider SDF pişirmesiyle aynı içeride kuralı.
  - Hücre maskesi: her hücre bir kez paralel sınıflanıyor; nokta testi yalnız sınır hücrelerinde.
- `FluidSeedMode::MeshVolume`:
  - `seedBox` (sıvı/MPM) yalnız içeride kalan alt-hücre slotlarını doğuruyor; sınır basamaklı
    değil, alt-voxel.
  - `seedMatterGrainBox` (DEM) yalnız içeride kalan FCC doğumlarını kabul ediyor.
  - Nesne yoksa ya da içi boşsa hiçbir şey seed'lenmiyor ve `seed_error` nedenini söylüyor.
- Sahne: `resolveMeshVolumeForSimulation` testi `SurfaceMeshCache.version` değişene kadar
  önbellekte tutuyor.
- İmza: yalnız nesne ADI hash'e giriyor. Geometri sürümü bilerek dışarıda; türetilmiş alan tuzağı.
- Katmanlar:
  - API `seedFluidParticlesFromMesh`;
  - IPC `fluid.seed_mesh {domain, object, particles_per_cell, replace, persistent, grain_saturation}`;
  - Python `rt.fluid.seed_mesh`;
  - `fluid.get` artık `seed_mode: mesh_volume`, `seed_mesh_object` ve `seed_error` veriyor;
  - overlay güncel, descriptor'lar yeniden üretildi, audit OK.
- Kayıt/yükleme: ProjectManager ve SceneSerializer.
- Panel:
  - Seed Mode'da "Mesh Volume (fill an object)" seçeneği ve "Seed Object" seçicisi;
  - Seed Fluid Now bu kipte nesnenin içini dolduruyor;
  - hata olursa panelde "Seed failed: …" satırı çıkıyor.

**Sırayla:**
1. **`python scripts/test/rt_test_seed_mesh_volume_ipc.py`** (boş, duraklatılmış sahne).
   4 kapı var: `missing_object`, `liquid_count`, `liquid_shape`, `hidden_object`. `sand` kolu
   bilgi amaçlı.
   - `liquid_count` düşük ve `seed_error` boşsa maske fazla dışarıda sayıyor. Sınır hücresi eşiğine
     (`0.8661 h`) bak.
   - `liquid_shape` FAIL ama sayı doğruysa: seed maske yerine kutuyu doldurmuş. `mask` parametresi
     seed fonksiyonuna ulaşmıyor.
   - ★ **En sinsi:** `hidden_object` FAIL. Gizli nesne `getSurfaceMeshCacheForObject`'ten null
     dönüyor demektir. Kum kale akışı tam olarak gizli mesh ister; ilk düzeltilecek madde budur.
2. **Kum kale (elle).**
   - Kale mesh'i, Matter domain, Default Substance Sand, Seed Mode Mesh Volume, Seed Object = kale.
   - Seed Fluid Now'a bas, kaleyi gizle. Kum kale şeklini almalı.
   - Islak kum için "Seed Grain Wetness" > 0 ve Settle Seed at Rest açık olsun.
   - Kale çöküyorsa bu seed değil, malzeme konusu (kohezyon/ıslaklık).
3. **Emitter inflow/outflow (bu partide YOK):** aynı `MeshVolumeTest`'le sonraki partide.

## ★ TAM REBUILD — İlk seed'i silen imza kayması: teşhis + koruma — 2026-11-04 gece 3 (YAZILDI, derlenmedi)

**★ Tam Rebuild gerekli:** `SceneData`'ya üye eklendi (`SimRewindRecord`, imza bölümleri, koruma).
Bu yapı neredeyse her yerde include ediliyor. Artımlı derleme eski `.obj` bırakırsa alakasız bir
yerde erişim ihlali görürsün.

**Canlı ölçüm (kullanıcı projeyi yeniden açtı, Seed, Play; izleyici 0,2 s aralıkla okudu):**
- Yüklemeden sonra imza `10057163858437137944` idi, seed 87 552 parçacık.
- Play'den sonra **ilk adımın hemen ardından** imza `17420775324187008440` oldu. Aynı anda n=0 ve
  `dropped_seeds` doldu. Arada hiçbir düzenleme yoktu.
- İkinci değer, önceki oturumlarda adım attıktan sonraki kararlı imzanın aynısı.
- Yani runtime ilk senkronda/adımda hash'e giren bir alanı türetip geri yazıyor. Sonraki seed'lerde
  değer zaten türetilmiş olduğundan imza değişmiyor ("yalnız ilk sefer").
- Önceki partinin `adoptSeedAsSimulationStart`'ı doğru çalıştı (seed'den sonra playhead 0'a
  geldi), ama bu kayma cache'i temizleyen yolu tetikledi.

**Yazılan:**
- `computeSimConfigSignature(parts)`: bölüm sınırlarında birikimli ara hash'ler.
  - Domain grid/gas, emitters/colliders, fluid thermal/solver/seed, flow sources,
    gas/force field, rigid.
  - İzleyici, son eşleşen kareyle farkı bulup ilk değişen bölümü
    `sim.control_state.last_rewind.first_changed_part` olarak kaydediyor. API, IPC ve Python
    katmanlarında; overlay güncel, audit OK.
- **Seed-start drift guard:** `adoptSeedAsSimulationStart` ilk 8 kare için tek seferlik bir
  koruma kuruyor. Bir adımın hemen ardından görülen kayma kabul ediliyor: bake korunuyor,
  `absorbed=true` kaydediliyor, log'a WARN yazılıyor. Koruma kökü gizler; kök alan bulunup
  imzadan çıkarılınca bu koruma da sökülecek.

**Sırayla:**
1. **Projeyi yeniden aç, Seed, Play (elle).** Su kalmalı, akmalı.
2. **Ajan:** `sim.control_state` → `last_rewind`.
   - Beklenen: `absorbed: true` ve `first_changed_part` kaymanın bölümünü söylüyor, örneğin
     `"domain 'Physics Domain 1' grid/gas"` ya da `"fluid '...' thermal/solver/seed"`.
   - O bölümdeki türetilmiş alanı bulup imzadan çıkarmak bir sonraki partinin işi.
   - `absorbed: false` ve su yine boşaldıysa: kayma korumanın penceresi dışında oldu
     (ilk 8 kare). `frame` alanı ne zaman olduğunu söyler.
3. ★ **En sinsi:** Seed ile Play arasında gerçek bir fizik düzenlemesi yaparsan (örn. viskozite)
   o düzenleme hâlâ başa sarmalı. Sarmıyorsa koruma fazla geniş.
4. Mesaj kusuru (bilinen, düzeltilmedi): seed'den sonra "Sim changed - cache reset" uyarısı
   çıkıyor. Seed playhead'i başa çekerken aynı istek kanalını kullanıyor.

## Splat damlaları da Default Substance materyali — 2026-11-04 gece 2 (YAZILDI, derlenmedi; artımlı)

**Önceki partinin canlı sonucu:** RayFusion yeni materyali alıyor ✔. RT'de canlı (ilk oynatma)
karelerde yüzey yeni materyalle görünüyor, ama eski su materyali de görünür kalıyor. Cache
karelerinde sorun yok.

**Ölçülen (frame 110, RAM cache):** rota `gpu_live`, tek volume ve composition var. Yani yüzey yolu
iki tarafta da aynı. Canlı oynatmada hareket eden parçacıklar spray/foam etiketi alıyor ve
label route ile **splat** olarak çiziliyor. `ParticleRenderBridge` splat materyalini etikete göre
seçiyor ve etiketsiz parçacığı Water bağlamasına hiç eşlemiyordu. Composition'daki kök sınıfın
aynısı. Damlalar domain materyaliyle, yani eski suyla çiziliyordu.

**Yazılan:**
- `ParticleRenderBridge.cpp`: etiketsiz parçacık Default Substance bağlamasının splat
  materyalini alıyor.
- `surface_sdf_shadow_query.glsl` (RayFusion RT gölgesi, `rayfusion_rt_shadow.comp`): aynı
  is_active ofset hatası düzeltildi (168 → 172).

**Derlenecek:** C++ artımlı ve `rayfusion_rt_shadow.comp`.

**Sırayla:**
1. **Rendered, cache temiz, ilk oynatma (elle).** Eski su görünümü kalmamalı. Su sıçrarken
   damlalar da yeni materyalle çizilmeli.
   - Hâlâ görünüyorsa: yüzeyin içinde mi (derinlik) yoksa ayrı damlalar mı, bunu söyle. İçerideyse
     kök splat değil; composition'ın desteksiz hücreleri olabilir, sondayla ölçerim.
2. **Seed → Play boşalması (kök bulundu, düzeltme YAZILDI).** Kullanıcı "Recreate Seed on Reset"
   kutusunun kapalı olduğunu söyledi, yani seed tek seferlikti.
   - **Kök:** seed RAM cache'i siliyordu ve sim'i o anki playhead karesine (ör. önceki koşunun
     sonu) park ediyordu, ama seed'li durumu hiçbir kareye kaydetmiyordu. Play başa sardı,
     `restoreSimFrame` başarısız oldu, `resetSimulationToStart` tek seferlik seed'i sildi.
     İkinci denemede playhead zaten baştaydı ve Play ileri adımladı.
   - **Düzeltme:** `adoptSeedAsSimulationStart` (`scene_data.h`). Elle tetiklenen seed 0. kare
     olarak cache'e alınıyor, playhead başa çekiliyor, rigid'ler 0'dan yeniden pişiyor.
   - **Test (elle):** önceki bir koşunun sonunda dur, kutu kapalıyken "Seed Fluid Now"a bas.
     Playhead başa gelmeli. Play'de su yerinde kalmalı ve akmalı. Döngüde başa dönünce seed
     korunmalı.
   - ★ Bir config düzenlemesi (cache düşüren) tek seferlik seed'i HÂLÂ siler. Bu tasarım gereği.
   - Bozuksa: playhead başa gelmiyorsa `consumeSimRewindRequest` tüketilmiyor. Su yine boşalıyorsa
     `captureSimFrame(0)` RAM bütçesinden reddedilmiş olabilir (log'da "snapshot rejected").
   - `rt_matter_smoke.py --quick` 16/16 kalmalı. `fluid.seed` artık playhead'i başa çekiyor;
     ileride bir kareye göre seed'leyen script varsa o farkı görür.

## Etiketsiz sıvı = Default Substance + Output'ta materyal seçimi sim'i sıfırlamıyor — 2026-11-04 gece (YAZILDI, derlenmedi; artımlı yeter, struct düzeni değişmedi)

**Sondanın bulduğu kök** (kullanıcının sahnesi, canlı): 87 552 parçacığın HEPSİ etiketsiz
(`fluid.get` → `substances: [{name:"", tag:0}]`). Output listesi bu parçacıklar için domain'in
`default_substance`'ını (Water) isim olarak gösteriyor, ama iki composition üreticisi de (CPU ve GPU)
etiketsiz parçacığı doğrudan slot 0'a, yani domain fallback'ına gönderiyordu. Water bağlaması seed
parçacıklarına hiç ulaşmıyordu. Cache'ten oynatınca parçacıklar gerçek etiketi taşıdığı için doğru
görünüyordu, ilk oynatmada yanlış. Kullanıcı tam olarak bunu bildirdi: "cache'lenince düzeliyor,
ilk oynatmada çıkıyor". RayFusion'daki 1–2 haftalık gerilemenin de en olası kökü bu: Output
listesine Default Substance satırı eklendiğinde başladı.

**Yazılan:**
- `collectSubstanceMaterialEntries` (`FluidLevelSet.h`): bağlama tablosundan composition
  girdilerini kuran TEK yer. Default Substance bağlamasına `kSubstanceUntagged` altında ikinci bir
  giriş ekliyor. Üç çağıran da bunu kullanıyor: CPU, GPU ve `fluid.surface_parity`. İki üreticideki
  "etiketsiz → slot 0" kısa devresi söküldü.
- `resolveFluidViews`: etiketsiz parçacığın Output rotası (Splat/SDF/Fog) da Default Substance
  bağlamasını izliyor.
- Sim imza (`scene_data.h`): Default Substance için varsayılan fizikli (viskozite inherit, liquid)
  bir satır artık hash'e girmiyor. Miscibility imzadan çıktı (yalnız yüzey alanını şekillendiriyor).
  Output'tan İLK materyal seçimi satırı yarattığı için sim sıfırlanıyordu.
  `MatterSubstanceState` tane sahibi aramasında bu satırı atlıyor; aksi halde sıralama değişirdi.
- `Main.cpp`: duraklatılmış karede Rendered → raster geçişi render resync istiyor. Sonda bunu
  canlı gördü: Material'a geçince rota `gpu_live`'da kaldı, raster cihazında composition yoktu.
- `material_preview_sdf_surface.frag`: is_active kontrolü 168 yerine 172'yi okuyor.

**Derlenecek:** C++ (artımlı) ve `material_preview_sdf_surface.frag`.

**Sırayla (sahnen açık; ajan IPC ile okur, sen gözle bakarsın):**
1. **Rendered, ilk oynatma (elle, en hızlı).** Cache'i temizle, Play'e bas. Yanlar da üst yüzey de
   `Default_Cube_Material` olmalı.
   - Hâlâ yalnız üstte doğruysa composition değil, yerleşim sorunu: probe ile kenar hücrelere bak.
2. **Sim sıfırlanmıyor (elle).** Bir kare ilerlet. Output'ta Water için materyali "Domain
   material" ↔ başka bir materyal arasında değiştir. Kare ve parçacıklar yerinde kalmalı.
   - Sıfırlıyorsa başka bir imza bu tabloya bakıyor. Bana söyle, `computeSimConfigSignature` /
     `computeFluidCouplingSignature`'a ve bake cache imzasına bakarım.
   - ★ Viskozite ya da phase değiştirmek HÂLÂ sıfırlamalı. Bu doğru davranış, fizik değişiyor.
3. **Duraklatılmışken Rendered → Material/RayFusion (ajan: `render.volume_slots`,
   `render.probe_surface_composition`).** `viewport` satırında `has_composition` true olmalı ve
   probe `device` == `host` vermeli. RayFusion yüzeyi materyalle görünmeli.
   - `has_composition` false: resync tüketilmedi.
   - true ama yüzey dielektrik: frag okuması bozuk. Probe değerine bak.
4. **`python scripts/test/rt_test_surface_gpu_ipc.py`.** `bound` kolu dahil parite PASS kalmalı.
   - ★ **En sinsi:** parite PASS ama iki taraf da yanlış. Alias girişi üç çağırana da gidiyor, parite
     bu hatayı göremez. Gerçek ölçü 1. ve 3. maddeler.

## RayFusion composition sondası — 2026-11-04 akşam (YAZILDI, derlenmedi; artımlı yeter)

**Composition partisinin canlı sonucu** (kullanıcının sahnesi, Matter / Water → Output listesinde
`Default_Cube_Material`, 87 k parçacık):
- ✔ GPU yolu çalışıyor: `surface_route` `gpu_host`, `surface_gpu_status` boş. Yüzey kurulumu
  122 → **32.5 ms** (host bekleme dahil).
- ✔ `fluid.surface_parity` kusursuz: max |Δφ| 1e-5 h; işaret, bant ve composition uyuşmazlığı
  0; uvw farkı 0. GPU 65 ms, CPU 170 ms (iki ölçüm de host bekleme dahil).
- ✘ RayFusion/Material hâlâ dielektrik. Raster cihazında `has_composition` true ve
  `preview_material_count` 4.
- Kontrollü deney (sonra geri alındı): aynı materyal domain yolundan verildiğinde
  (`iso_material_slot` 1, composition yok) raster opak gri, yani DOĞRU çiziyor. Bozuk olan
  yalnız composition okuması. Frag kodu ve ofsetler (146/154. word'ler) dağıtılan `.spv`'de
  doğru.

**Yazılan:** `render.probe_surface_composition {point:[x,y,z]}`. Frag'in o noktada okuduğu
composition hücresini her Vulkan cihazının KENDİ tamponundan geri okuyor, host kopyasıyla
yan yana veriyor.

**Derledikten sonra (ajan):** yüzeyin üst katmanında birkaç nokta sorgulanır.
- `viewport.device` ≠ `host` ise raster cihazına yükleme bozuk: ofsetli `uploadBuffer`,
  bellek eşlemesi.
- `device` == `host` == (1, 0, 0) ise veri doğru, frag yanlış okuyor: adres ya da
  buffer_reference.
- `host[0]` == 0 ise o hücre desteksiz. RT neden doğru çiziyor, o zaman ona bakılır.
- `address_offset` ≠ `uvw_bytes` ise composition adresi yanlış hesaplanmış.

## ★ TAM REBUILD — GPU composition + RayFusion teşhis alanları — 2026-11-04 öğleden sonra (YAZILDI, derlenmedi)

**Kullanıcının ilk derlemesinden sonra canlı okunan** (Matter domain, Water, 87 k parçacık,
Output listesinden Water → `Default_Cube_Material`):
- `surface_route` `cpu` idi. `surface_gpu_status`: "substance materials need the composition
  grid". Output listesinden materyal bağlanmış her domain GPU yolunu reddediyordu, hızlanma
  bu yüzden yoktu. Yüzey 122 ms'de kalıyordu (merkez yumuşatma 99 ms).
- RayFusion yüzeyi hâlâ dielektrik çiziyordu, sahne görüntüsünde doğrulandı. Raster SDF
  shader'ı yeni `.spv`'yle composition okuyor, ama sonuç değişmedi. Kök kod okuyarak
  BULUNAMADI.

**Yazılan:**
- `sim_surface_composition.comp`: `buildCompositionGrid`'in GPU ikizi. GPU yolu artık madde
  materyallerini reddetmiyor; host'a inen rotada composition da iniyor, canlı rotada
  adresiyle yayınlanıyor (`surface_live_composition_address`).
- `fluid.surface_parity` composition'ı da karşılaştırıyor; testte yeni `bound` kolu var.
- **Teşhis:** `render.volume_slots`'a eklenenler:
  - slot başına `iso_material_slot`, `has_uvw`, `has_composition`, `uvw_dim`;
  - backend başına `preview_material_count`.
  RT ve raster ayrı cihazlar; ikisi yan yana okunur.

**Derlenecek:** tam Rebuild (`VDBVolumeData`, `VolumeSlotDiag` ve `FluidSurfaceGpuBuffers`
büyüdü), yeni `sim_surface_composition.comp`. Aşağıdaki listenin geri kalanı geçerli.

**Derledikten sonra (ajan, sahnen açıkken yalnız okuyarak):**
1. RayFusion modunda `render.volume_slots`:
   - `viewport` satırında `has_composition` false ise composition raster cihazına
     gitmiyor. Bu durumda VolumetricRenderer'ın uvw kapısına ya da backend'in uvw bloğuna
     bakılır.
   - `has_composition` true ama `preview_material_count` 0 ise raster'ın materyal tablosu
     boş, hiçbir materyal geçemez (domain materyali de).
   - İkisi de doluysa kök shader'da: ya hit noktasında composition hücresi desteksiz
     (fallback 0), ya da materyal okuması başka bir yerde eziliyor.
2. Senin sahnende `fluid.get`: `surface_route` `gpu_host` (RayFusion) ya da `gpu_live`
   (Rendered) olmalı, `surface_gpu_status` boş, yüzey kurulumu ~120 ms'den birkaç ms'ye
   inmeli.

## ★ TAM REBUILD — GPU yüzey level set'i (Y1+Y2) + RayFusion madde materyali — 2026-11-04 (DERLENDİ: GPU yolu madde materyalli domain'de çalışmadı, RayFusion düzelmedi; üstteki parti)

Tasarım ve devir notu: [GPU_YUZEY_LEVEL_SET.md](GPU_YUZEY_LEVEL_SET.md).

**★ Tam Rebuild gerekli.** Şu yapıların düzeni değişti ve çok yerde include ediliyorlar:
- `LevelSetParams` (`gpu_build`), her domain tanımının içinde;
- `LevelSetStats` (`route`, `gpu_status`, baytlar);
- `rtapi::FluidDomainInfo`;
- `VDBVolumeData` (`surface_live_uvw_address`).

Artımlı derleme eski `.obj` bırakırsa alakasız bir yerde erişim ihlali görürsün.

**Derlenecekler:**
- C++. İki yeni `.cpp`, ikisi de vcxproj'da: `Physics/Fluid/FluidSurfaceGpu.cpp`,
  `Api/RtApiFluidSurface.cpp`.
- Shader'lar (`compile_shaders.ps1`):
  - 10 yeni `sim_surface_*.comp`;
  - değişen `volume_closesthit.rchit`: üç varyantı da (base, fog, surface);
  - `material_preview_sdf_surface.frag`.

  `.spv`'ler `x64/Release/shaders`'a da gitmeli. Eksik bir `sim_surface_*.spv` hiçbir şeyi
  bozmaz, yalnız GPU yüzeyi CPU'ya düşer (`surface_gpu_status`: "dispatch refused").

**Ölçülen hedef (derleme öncesi):** hareketli 100 k kar karesinde CPU yüzeyi 235 ms
(level set 184 + uvw 51). Solver 128 ms. `loop.frame` ~420 ms.

**Sırayla** (boş ve duraklatılmış sahnede, Vulkan):

1. **RayFusion madde materyali (hızlı, bağımsız; elle).** Bir domain'de bir maddeye
   (Water/Snow) madde materyali olarak bir sahne materyali bağla, RayFusion/Material
   moduna geç. Yüzey o materyalle görünmeli, artık domain varsayılanında kalmamalı.
   - Hâlâ varsayılandaysa: sen materyali domain'in kendi "Surface Material" seçicisinden
     atamışsındır, madde bağlamasından değil. O yol zaten `iso_material_index`'i okuyor.
     Kök başka (raster volume tablosu yenilenmiyor olabilir). Bana söyle.
   - Düzeltme yalnız raster SDF shader'ında: composition ızgarasından baskın materyali
     okuyor.
2. **`python scripts/test/rt_test_surface_gpu_ipc.py`: 7/7 hedef.**
   - `water_parity` / `snow_parity` FAIL ise `sign_mismatch` ve `max_abs_phi_over_h`'a bak.
     - `max_abs_phi_over_h` ≈ 0.5: kutulama ya da hücre merkezi yarım voxel kaymış.
     - Ölçek hatası gibi büyükse merkez yumuşatma tutmuyor (`reach`, `lambda`).
     - `uvw_max_abs` büyük ama φ iyi: genişletme taraması sayısı ya da düzleştirme.
   - `route_gpu` FAIL ve `surface_gpu_status` "dispatch refused": `.spv` dağıtılmamış.
   - `cpu_build_gone` FAIL: GPU yolu sessizce CPU'ya düşüyor. Nedeni `surface_gpu_status`'ta.
   - `faster` eşiği ×0.25: GPU yolunda render_sync içinde yalnız ~3 MB yükleme, kernel'ler
     ve hash kalmalı.
   - ★ **En sinsi:** parite PASS, ama `route_gpu` `gpu_host` ve sen Rendered moddasın.
     Canlı rota açılmıyor demektir (`surface_live_possible`): `render_settings.use_vulkan`,
     `g_solid_viewport_active` ya da `domain_vdb_ids[d] < 0`. Görüntü yine doğru çıkar, ama
     her kare proxy geri iniyor ve ölçüm farkı gizler.
3. **Rendered (Vulkan RT) gözle (elle).** Kar ya da su sahnende panelde "Build Surface on
   GPU"yu aç/kapat. Yüzey birebir aynı kalmalı, panel "Built on: GPU (RT reads device
   buffers)" demeli.
   - Rendered'da yüzey yok ama Solid'de var: canlı rota örneklemesi bozuk. Volume tablosu
     type 4 / source 4, yerleşim `uvw_*` alanlarından geliyor. Anahtarı kapatınca dönüyorsa
     kök orada.
   - Yüzey var ama hiç kırılma yok, sis gibi: backend `liveDenseSurface` dalına girmemiş,
     `source_type` 5'e çevrilmiş.
   - ★ **En sinsi:** yüzey yarım voxel kaymış ya da biraz incelmiş görünür, ama "güzel"
     durur. Aynı karede anahtar açık/kapalı iki render al ve sayısal farkına bak; gözle
     bakmak yetmez. Yerleşimde −0.5 hücre merkezi kuralı (NanoVDB yolu +0.5 öteliyor).
4. **Solid / Material / RayFusion (elle).** Panel "GPU (read back for viewport)" demeli ve
   yüzey Rendered ile aynı olmalı.
   - Duraklatılmış bir karede Rendered → Solid geçişi yap. Solid o anki yüzeyi göstermeli
     (`host_stale` yeniden kurulumu tetikler). Eski kare görünüyorsa tetik çalışmıyor.
5. **Maliyet.** Hareketli kar sahnende `perf.list`:
   - `sim.render.surface.level_set` ve `material_coordinates` çalışmamalı;
   - `sim.render.surface.gpu` birkaç ms olmalı;
   - `loop.frame` ~420 ms'den belirgin düşmeli.
6. **`python scripts/test/rt_matter_smoke.py --quick` (arka planda): 16/16 kalmalı.**

**Bilinen risk (ölçülmedi):** domain yeniden boyutlanırsa canlı yayınlanan proxy tamponu
yeniden ayrılır. Bir önceki karenin RT işi hâlâ o adresi okuyorsa device-lost riski var.
Canlı gazda da aynı risk bulunuyor. Domain boyutunu oynatırken çökme görürsen ilk aday bu.

## Önceki partiden açık (elle, MPM domain uykusu — canlı kod)

1. Kar sahnende Settle on Seed açık, kar dinlenince `fluid.step_stats` `gpu_status`
   `matter_asleep` olmalı ve kare hızlanmalı. Collider oynatınca ya da kartopu atınca hemen
   uyanmalı.
2. ★ Uyuyan yığına yaslanıp DURAN bir collider: yığın 3 kare sonra yeniden uyur. Collider
   kara o anda itme uyguluyorsa bu yanlış (bağlam özeti temas kuvvetini görmüyor).
3. Kuvvet alanı (rüzgâr vb.) varken domain hiç uyumamalı.
4. Kıpırdama yeniden görünürse `fluid.particle_extremes` → `fastest` sebebini söyler
   (`granular_pressure` ~0 + destekli + uyanık → settle sönümü desteğe bağlanmalı).
