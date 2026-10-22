# NEXT BUILD CHECKS

> **Durum:** CANLI — 2026-10-22 (3. parti); materyal programı pin düzeltmesi + K0 yakın mesh havuzu boşluğu. Kaynakta, derlenmedi.

Önceki partinin açık deneyi (RT gaz phase-cache var/yok iki build) bu partiyle
çakışmıyor; ayrıntısı `RT_GAS_OPTICAL_PERFORMANCE.md` sonunda.

Build: C++ yeterli (shader değişmedi). Değişen: `MaterialNodesV2.h` (derleyicide
`inPin`: değeri pinde yaşayan 25 giriş artık pin değerini derler),
`ParticleRenderBridge.cpp` (yakın mesh havuzu kapasite kuralı
`nearPoolCapacityFor`). Gerçek shard + mineral ölçümü bu build'den önce canlı
koşuldu (KUM_TANE…md §8b).

Sıra: bağımsız ve hızlı görülen önce.

1. **Mix Color rengi render'a ulaşıyor.** Yeni principled materyal → node
   grafiği: Mix Color (Fac 0, Color1 kırmızı, bağlantısız) → Base Color;
   bir küpe ata, Rendered (Vulkan RT). Küp **kırmızı** olmalı.
   Ajan yolu: `rt_grain_mineral_material.py` yerine aynı grafiği
   `nodes.set_param` ile kurup `render.probe`/ekran görüntüsü.
   *Bozuksa:* gri (0.5) = derleyici hâlâ literal basıyor (eski exe — zaman damgası).
2. **Eski sahneler.** Mix Color / Hue-Sat / Combine Color kullanan kayıtlı bir
   materyali aç. ★ **En sinsi sonuç:** render'ı DEĞİŞEBİLİR — bu düzeltmenin
   kendisi: artık panelde görünen pin değeri çiziliyor, eskiden 0.5/1.0 literal.
   Değişen sahne bug değil; değişmeyen ama panelde farklı renk gösteren sahne bug.
   *Bozuksa:* bağlı girişli bir node'un rengi değiştiyse `inPin` bağlantıyı
   atlıyor (olmamalı — ilk satırı `inOr` ile aynı).
3. **Küre yolu değişmedi.** Near-mesh kapalı bir kum sahnesinde
   `render.instance_stats` ve ms/örnek önceki build'le aynı.
   *Bozuksa:* değişiklik ana havuza sızmış (yalnız yakın havuz kuralı değişti).
4. **Boşluk azaldı.** Ajan fixture'ı kurar ve tam K0'ı koşar:
   `python scripts	est
t_k0_grain_mesh_near_ipc.py --domain K0_Sand --apply --distance 1.5 --library fracture`.
   Beklenen hepsi-mesh vakası (4591): `tlas_instances` ≈ 6.1 bin (önce 12321);
   1.5 m (780): ≈ 1.0–1.6 bin (önce 3105). Script artık kullanıcının near
   `sources` listesini de geri yükler (`near_restored` true).
   *Bozuksa:* hâlâ ~12 bin = eski exe ya da kapasite yolu eski kuralı kullanıyor.
5. **Kamera hareketi.** Fixture'da kamerayı birkaç kez az oynat (aynı mesafe
   bandında): her harekette yapısal yeniden kurma olmamalı (`perf.list`
   `accel.vulkan_rt.rebuild` sayısı hareket başına artmamalı, yalnız kapasite
   aşılınca artar).
   ★ ms/örnek boşluk azaldıktan sonra hiç değişmezse "boşluk bedava" diye
   okunur — opak shard ölçümü (§8b) zaten bunu söylüyor; ama yalnızca
   `tlas_instances` gerçekten düştüyse geçerli. Önce slot sayısına bak.

6. **Grafik silme + Color Ramp durakları + pin değeri (tek script).**
   `python scripts	est
t_material_node_parity_ipc.py` → `PASS`
   (pin_value, remove, ramp_stops). Kendi küpünü 60 m yukarıda kurar, `render.probe`
   ile ölçer, kamerayı geri yükler.
   ★ **ESKİ exe'de bu script uygulamayı ÇÖKERTTİ** (2026-10-22): `nodes.set_property`
   iç içe dizi değeri (`[[...],...]`) alınca pipe koptu, süreç kapandı. Eski
   `nodeParamFromJson` 3 elemanlı diziyi Vector3 sanıp `get<float>()`'u bir dizi
   üzerinde çağırıyordu; atılan `json::type_error` yakalanmıyor gibi. Yeni build'de
   bu yol `Rows` olarak ayrıştırılıyor. Yine de kontrol et: IPC dispatch'i
   `std::exception` (yalnız `runtime_error` değil) yakalıyor mu? Yakalamıyorsa
   herhangi bir metoda bozuk tipli değer göndermek uygulamayı düşürür.

Doğrulama (ajan): değişiklikler derlenmedi. Derleyici hatası canlıda bulundu
(kırmızı Mix Color gri çizildi), düzeltme sonrası koşulmadı.

---

## Ayrı parti (aynı build): fracture shard'ları flat SoA

Kaynakta, derlenmedi. Değişen: `scene_ui_menu.hpp` (shard üretimi + silme),
`scene_data.h` (`nodeWorldCenter`, `getRigidBodySourcePivot`, kayıt öncesi
rest snapshot'ı, yeni `forgetBodyMeshCaches`), `ThermalFractureBridge.cpp`
(`nodeMeshVolume` artık `indices` üzerinden). Yeni .cpp yok, IPC yüzeyi değişmedi.
Her shard artık **tek** `TriangleMesh` (ardışık üçlü, `indices[i]==i`); facade üretilmiyor.

Sıra: bağımsız ve hızlı görülen önce.

1. **Shard tek flat mesh.** `scene.add_primitive cube` → `physics.fracture_object`
   (site_count 24) → bir `shard_objects` adına `scene.object_info`.
   Görmen gereken: `meshes: 1`, `triangles` > 0, `has_bounds: true`.
   *Bozuksa:* `meshes: 0` = hâlâ facade üretiliyor (eski exe — zaman damgasına bak).
2. **Görünüm/malzeme aynı.** Viewport'ta preview gap'li shard'lar; exact modda dış
   yüz kaynak dokusunu taşır, kesik yüzler `<obje>_Interior` malzemesinde.
   *Bozuksa:* hepsi tek renk/doku (0,0) = materialID/uv özniteliği yazılmamış.
3. **Seçim/picking.** Bir shard'a viewport'ta tıkla → hiyerarşide o shard seçilir;
   Rigid Body listesinde tıklamak da viewport'ta seçer.
   *Bozuksa:* `rebuildMeshCache` flat shard'ı `direct_mesh_nodes`'a almıyor.
4. **Re-fracture + unfracture.** Aynı objeyi farklı seed'le ikinci kez kır, sonra
   `physics.unfracture_object`. Beklenen: ikinci kesimde eski shard'lardan **hiçbiri**
   kalmaz (obje sayısı = yeni shard sayısı + diğerleri); unfracture orijinali geri
   koyar, `__shard_` adlı obje kalmaz.
   *Bozuksa:* çift shard = `fractureEraseNodes_` flat'i silmiyor.
5. **Make Breakable → kır → simüle et.** `physics.make_fracture_group` (veya panel),
   `physics.break_fracture_group`, birkaç kare oynat. Shard'lar ayrılıp düşmeli,
   grup kütlesi hacme göre (24 shard'lık 2 m küp ≈ 8 m³ × yoğunluk), **24 kg değil**.
   Patlama merkezden dışa doğru olmalı.
   *Bozuksa:* hepsi aynı yöne/orijinden uçuyorsa `nodeWorldCenter` flat'i okumuyor;
   kütle = shard sayısı kg ise `nodeMeshVolume` 0 dönüyor.
6. **Re-fracture SONRASI simülasyon.** 5'i yap, sonra objeyi re-fracture et ve tekrar
   kır/oynat. Çökme veya eski kesimin pozunda donmuş shard olmamalı.
   *Bozuksa:* rigid bake cache'i ölü `TriangleMesh*` tutuyor (`forgetBodyMeshCaches`
   çağrılmamış) — bu UB, bazen "çalışır" görünür.
7. **Kayıt/açma.** `scripts/material_phase_tests/phase_10_fracture_roundtrip.py`
   (script değişmedi, Release kopyası aynı). PASS + yeniden açılışta shard adları ve
   cluster'lar aynı.
8. **★ Simülasyon ortasında kayıt (en sinsi).** 5'teki sahnede shard'lar havadayken
   (kare ~30) `project.save`, sonra `project.open`. Açılışta shard'lar **dinlenik
   (kesilmiş) pozda** olmalı, havadaki pozda değil; kayıttan sonra da canlı sahne
   havadaki pozda kalmalı.
   *Bozuksa:* açılışta shard'lar dağınık duruyorsa kayıt öncesi rest snapshot'ı flat
   mesh'i atlıyor — hata vermez, "kaydettiğim yer" gibi makul görünür. Bu
   add_primitive ile eklenen her flat rigid body için de geçerliydi (eski hata).
9. **Termal kırılma.** Yanmış bir objede `pattern=2` ile kır; panel mesajında
   "no recorded burn damage" yalnız gerçekten hasar yoksa çıkmalı; kırılma bölgesi
   yanık bölgeye yoğunlaşmalı.
10. **K0 tane örneği olarak shard.** (Diğer oturumun scripti; o oturum facade reddini
    kaldırınca) `rt_k0_grain_mesh_near_ipc.py --library fracture`: `exemplars` = shard
    sayısı, `exemplar_triangles` > 0, `status: active`.
    *Bozuksa:* `missing source` = `gatherNormalizedSceneFaces` shard'ı bulamıyor.


## 2026-10-22: prosedürel tane kütüphanesi ve DEM/MPM/fluid görünümü

Derlemeyi kullanıcı yapar; ajan build veya uygulama başlatmadı. Yeni
`UI/GrainMeshLibraryUI.cpp` proje/filters listelerine kaydedildi.

1. Üstteki materyal/K0 maddeleri 1–6'yı yeni executable ile tamamla.
2. Domain'i Splat/Particles görünümünde, geometry=icosphere olarak hazırla;
   birkaç carrier doğduktan sonra timeline'ı duraklat, kamerayı Near Distance
   içine getir. Ayrı terminalde:
   `python scripts/test/rt_grain_library_options_ipc.py --domain K0_Sand --render`
   Beklenen PASS: rock ve voronoi için near_instances>0; yanlış string/sayı/
   iç içe dizi sonrası IPC hâlâ cevap verir. Probe ayarları geri yükler.
3. 2'yi küçük **DEM, MPM ve fluid** sahnelerinde ayrı çalıştır. Virtual Grains
   8 iken mesh açık → primary_visual_children=1; kapalı → eski ayar geri
   gelmeli (MPM Physical Carriers açıksa primary=1 kalması doğrudur).
   Whitewater sayısı etkilenmemeli. Rendered RT ve Solid/Matcap görünümünde
   aynı carrier hem küre hem mesh olarak çift çizilmemeli.
4. Near Grain Meshes → Grain Shape: Procedural Rock / Rounded ve Procedural
   Voronoi / Angular kaynak obje seçmeden çalışmalı. Scene obje sayısı sabit
   kalmalı. Near Distance/bütçe dışında kalan carrier'lar küre kalmalı.
5. Scene objects seç; iki farklı materyalli kaynak (mümkünse bir kaynakta iki
   yüz materyali) ekle. Use Exemplar Face Materials açıkken yüz materyalleri
   ve UV'ler korunmalı; kapalıyken domain/substance/ıslak bantlar kullanılmalı.
   Kaynakta materyal atamasını değiştir: paused sahnede yakın mesh güncellenmeli.
   Uzak küreler bu seçenekten etkilenmez; yakın/uzak renk farkı bu modda beklenir.
6. Kaydet/aç: shape ve source_materials aynı kalmalı. Eski projede scene/false.
   K0 A/B ölç: near_restored=true; benchmark kendi shape/materyal seçimlerini
   geri yüklemeli. K0 fixture önce yeniden kurulmalı, önceki sahne kaydedilmemişti.
7. `scripts/test/grain_mesh_library_test.cpp` kaynak testi kullanıcı test
   derlemesinde çalıştırılmalı; ajan proje kuralı gereği derlemedi.
8. Fiziksel boyut dağılımı sonraki iştir. Önceki emitter durması kullanıcı
   tarafından sahne kurulumuna bağlandı; çözücü bug'ı olarak açık tutulmuyor.

Ajanın build gerektirmeyen kontrolleri: Python AST, JSON/XML okuma,
IPC descriptor üretimi/--check, ilgili dosyalarda diff whitespace kontrolü.
Canlı render, kayıt round-trip ve C++ kabul testleri çalıştırılmış sayılmaz.


## 2026-10-22: Add Mesh / Grains–Crystals üreticisi

Ayrıntılı sözleşme: `docs/dev/GRAIN_CRYSTAL_GENERATOR.md`.

1. Kullanıcı derlemesinde yeni `Scene/GrainGenerator.cpp`, `UI/GrainGeneratorUI.cpp`
   ve `Api/Rt{Api,Ipc,Python}GrainGenerator.cpp` modülleri alınmalı (vcxproj/filters
   kaydedildi; CMake source glob'ları kapsıyor). Ajan derlemedi/uygulamayı başlatmadı.
2. Add Mesh → Procedural Generator → Grains / Crystals: Rock/Sand, Voronoi,
   Snow Crystal ve Mineral Crystal. Seed, adet, biçim, çap aralığı ve ağırlıklı
   materyaller kullanıcı kontrolünde olmalı. Yeni kütüphane gerçek flat objeler olmalı.
3. Ayrı terminal: `python scripts/test/rt_grain_generator_ipc.py` → PASS.
   `--keep` geometri/materyal/kayıt incelemesi için örnekleri bırakır.
4. Mesh Grains → Create Grain / Crystal Library… aynı pencereyi doğru hedefle
   açmalı; Generate seçili domain'e kaynakları ve kendi materyallerini bağlamalı.
   Tek Undo/Redo objeleri ve önceki domain kaynak ayarlarını birlikte geri getirmeli.
5. Karı alt/üst/yandan incele: kapalı, kalınlıklı dallı plaka; uç/kenar delikleri
   ve üst üste iç dal yüzleri olmamalı. Salt: 4 kenar/0 uç; Quartz/Ice: 6 kenarlı
   prizma; Sugar: seçilebilir uzama/uç. Aspect ve seed sonuçta görülmeli.
6. Üretilen örnekleri içeren proje kopyasını kaydet/aç. İsimler, geometri,
   yerleşim, UV ve materyaller korunmalı. Küçük çaplı kar örnekleri de Mesh Grains
   kaynağı olmalı (SoA yüz okumasında mutlak minik-alan elemesi kaldırıldı).
7. Eski near_mesh_shape=rock/voronoi sahneleri açılmalı; yeni UI'de ayrı sabit
   üretici dropdown'ı görünmemeli. Scene object listesi tasarımı ertelendi.
8. Kullanıcı kaynak testlerinde `grain_generator_test.cpp` + mevcut
   `grain_mesh_library_test.cpp` çalıştırılmalı. Fiziksel boyut dağılımı hâlâ ayrı iş.

## Voronoi edge rounding acceptance (2026-10-22)

- User build: GrainGenerator.cpp and GrainGeneratorUI.cpp integration.
- Generate Voronoi with Examples=1: an irregular cut cell, not an uncut cube.
- Change seed: different geometry; repeat seed: deterministic geometry.
- Compare Roundness=0/0.6 with Edge Detail=6 under lighting and wireframe.
- Compare Edge Detail=2/8: real triangle density must change when rounded.
- Aspect=(0.7,1.6,1): verify smooth normals and closed surface.
- Repeat via Python and scene.generate_grains IPC with identical parameters.
- User compile/run scripts/test/grain_generator_test.cpp: new single-cell,
  seed, closed-surface, smooth-normal and detail regressions.
- Recheck Snow Crystal, materials, Undo/Redo and save/load.

Codex did not build or launch the application; visual acceptance remains pending.

## DEM physical size distribution (2026-10-22, user build/live checks pending)

- Build C++ and DEM / grain-fluid / grain-MPM shaders together (DEM revision 28).
- Build/run scripts/test/matter_grain_size_test.cpp and existing birth/stages tests.
- In a paused empty scene, external terminal: python scripts/test/rt_grain_size_ipc.py.
- UI: min radius .01 m, max .02 m, Sand + large stones, large fraction .10.
  Verify Sphere and Near Mesh agree, larger stone resting heights, impacts,
  sleep/wake, source spacing, reset replay, project save/load and cache scrub.
- Recheck uniform/log-uniform, max=0 legacy fixed size, invalid patches unchanged,
  and all five size fields locked while particles are live.
- Recheck wet/liquid/gas/thermal/MPM momentum/heat regressions for mixed sizes.
- Dense 1:2 and 1:4 piles: inspect explicit bucket/list/contact capacity holds;
  existing 16 bucket / 32 list / 24 history limits remain, no silent truncation.
- See docs/dev/DEM_SIZE_DISTRIBUTION.md for settings and physical semantics.

No project/shader compilation or application/live IPC verification performed by Codex.

## DEM size controls and visible refusal feedback follow-up

- Pending user build: size edits are staged together, applied atomically with
  Apply size settings; invalid ranges are shown beside controls; reset required
  blocks applying sizes while preserving the draft. Discard restores saved sizes.
- Large stone share now uses percent with small/large count labels and estimated
  volume/mass dominance; optional 95% sand / 5% stones preset. Physics RNG unchanged.
- Rejected settings and held solver steps display prominent orange/red wrapped text.
- Verify two radius edits do not apply individually; Apply/Discard; external IPC
  settings refresh the draft; pending size edits do not alter other solver changes.
- Live IPC diagnosis in Physics Domain 1: damping bound requested 529 substeps,
  cap 512. Raised cap to 1024; steps dt=1/60 and dt=1/24 succeeded, held=false.
  Same 1143 grains retained; center Y .828615 -> .808913 -> .747721 m.
  Timeline remained paused at frame 22; sizes and source unchanged, no reset/save.
- This verifies the current executable; pending UI source changes need a user build.

## DEM mixed-size contact pile / runtime neighbour lists (revision 29)

Live reproduction at paused frame 17, Physics Domain 1: minimum radius .026 m,
maximum .083 m, bimodal large fraction .07, seed 1360, max_substeps 1024.
One external dt=1/24 step held with "grain neighbour list full (>32 ...)" at
4289 grains. This is a separate capacity failure from the previous CFL limit.
No reset, size/source edits or save were performed during this reproduction.

Pending user build:
- Rebuild C++ and all sim_matter_grain shader variants together (revision 29).
- Fixed-size domains retain 32-list baseline; mixed sizes start with 128 slots.
- Standalone DEM retries list-only overflow with a doubled budgeted buffer from
  unchanged host state; shared MPM/liquid clocks refuse replay to avoid double steps.
- Allocation includes old+new peak memory and checks shader index width.
- IPC diagnostics: runtime.neighbour_list_capacity / neighbour_list_retries.
- With the failing pile open and paused, external terminal:
  python scripts/test/rt_grain_neighbour_capacity_ipc.py --steps 8
- Check actual floor contact, dense mixed pile, fixed-size baseline, storage/working
  budget failures, and shared-clock mixed cases. Actual contact history remains 24;
  this change removes the 32-candidate ceiling, not all other solver budgets.

Codex performed source checks only for revision 29; no compilation or live run of
new code. The open executable still has revision 28 and cannot use this fix yet.

## DEM contact history overflow at floor impact (revision 30)

Live read-only diagnosis: paused frame 29, Physics Domain 1, 94,915 grains,
minimum radius .011 m, maximum .036 m, large count 995 (1% authored), seed 1360,
max_substeps 2048. Status: contact count/history-in-flight exceeds legacy 24.
No reset, setting mutation, extra tick or save performed during this diagnosis.

Pending user build and acceptance:
- Rebuild C++ plus all sim_matter_grain shader variants together, revision 30.
- Mixed-size history starts at 64 records, fixed-size remains 24. Four 32-bit
  mask words support records across boundaries 31/32, 63/64 and 95/96 up to 127.
- Block metadata is now owner + four masks + rest + compensation (7 words),
  following the capacity-sized identity->block map. Sleep/audit offsets changed.
- Standalone DEM history overflow retries from unchanged canonical host state
  with capacity doubled to at most 128, charged to actual/peak memory budget.
- Shared MPM/liquid clocks cannot replay and continue to refuse overflow.
- Adaptive CFL retries use observed maximum contacts + 4, bounded by allocated
  history; shared clocks use the full allocated history bound.
- runtime.contact_history_capacity / contact_history_retries expose allocation.
- External terminal with the failed pile paused:
  python scripts/test/rt_grain_neighbour_capacity_ipc.py --steps 8
- Verify floor impact and sustained dense packing beyond 24 contacts, sleep/wake,
  changing neighbour identities, population reorder, GPU-resident fixed-size baseline,
  friction/rolling persistence, conservative memory refusal, and mixed-owner safety.
- This raises mixed-size memory use (64 records for every grain initially);
  it is not sparse per-grain contact storage and does not establish million-grain
  cinematic performance. At 128 records a million-grain history alone is ~3.34 GiB,
  potentially above a device's single-buffer limit. The checked budget must refuse it.

Revision-30 work received source checks only. No build/shader compilation or live
physics acceptance of the new code was performed by Codex.

## Size-aware contact CFL bound (host-only follow-up, shader ABI stays revision 30)

Live diagnosis at frame 26: .011/.036 m bimodal 1%, 86,462 grains, cap 2048,
held at damping bound 2081. Existing bound combined globally smallest mass with
largest radius stiffness. This pair does not describe a physical grain.

Source change: for each inertial mass mi and radius ri, bound incident stiffness
by alpha*max(ri,2*ri*Rmax/(ri+Rmax)); reduce maximum ki/mi across particles.
Then omega^2 <= 2*contact_budget*max(ki/mi), retaining conservative damping and
accuracy algebra. Wet bridge bound remains conservative. Mixed-size rolling
uses the conservative unequal-radius rotational factor. Equal dry sizes retain
the old normal bound. No stiffness, restitution, size or substep cap is modified.

Validation:
- Analytical wall/pair inequalities checked for .011/.016/.026/.036 m; new
  normal-rate bound is .468085 of the old fictitious min-mass/max-k bound.
- Added C++ regression for unequal pairs, walls and equal-size parity in
  scripts/test/matter_grain_size_test.cpp (not compiled/run by Codex).
- Live revision-30 executable, unchanged authored settings: 12 external ticks
  at dt=1/48 succeeded, 36..42 peak contacts, history 64, neighbour list 128,
  no hold. This validates shorter dt under old bound, NOT the new host code.
- User rebuild C++, no shader ABI change. Repeat original dt=1/24 dense impact
  and sustained pile, inspect cfl_stiffness_mass_rate_s2 and substeps.
- Repeat fixed-size convergence, high rolling friction and wet/shared-clock
  cases. Lower substep count is accepted only with stable/converged physics.
- No reset, material/size edits, substep cap changes or scene save by Codex;
  12 manual half ticks advanced the live simulation, timeline left paused.

## Explicit reset clears stale DEM CFL hint

Live check confirmed contact_resolution=8 was applied; a NEW dt=1/24 step still
held on damping (2129 > 2048), not accuracy. Lowering accuracy cannot remove a
damping limit. The new size-aware bound remains pending a user C++ build.

Source audit additionally found resetGridDomainStates cleared host particles but
only invalidated generic field copies; retained matter_runtime->grain kept the
previous dense pile's cfl_contact_hint, publication identities and sleep context.
Added invalidateMatterGrainAfterReset in the focused DEM module and minimal calls
in both explicit reset buffer loops. Reusable allocations/growth requests remain;
CFL hint resets to baseline, history/residency/sleep/motion publication is cold.
Do NOT call this on per-step invalidateDeviceCopies (would erase valid history).

User build and regression:
- C++ rebuild only, shader ABI stays revision 30.
- After a dense pile/error, reset to sparse births and verify first-step
  cfl_contacts_per_grain starts at baseline instead of the old peak.
- Test fixed/mixed sizes, empty reset, repeated reset, and births restarting IDs.
- C++ reset-state regression added to matter_grain_size_test.cpp, not executed.
- Source checks passed. No reset or authored setting changes made in the live
  scene by Codex in this follow-up; one failed probe tick was attempted.

## Live close-out 2026-10-22: DEM continuation and MPM/DEM fixture

User-built executable: existing 100,000-grain DEM scene completed six dt=1/24
steps without holds, 1162..1260 substeps, max contacts 33, mean step_total 1358 ms.
This closes the reported short continuation failure, not long-run/scale acceptance.

Created MPM Sand + DEM Stones in the same scene. Previous domain/source retained
but disabled; editing setup reset previous live particles. New fixture completed
86 dt=1/120 ticks with 12,288 MPM carriers and 10 DEM stones, nonzero bidirectional
contact events and sampled momentum residual ~1e-8 N*s. It remains paused at frame 0.
Actual penetration/support/convergence acceptance remains next work.

Details: docs/dev/DEM_MPM_ACCEPTANCE_20261022.md
Raw measurements and previous configuration: docs/dev/dem_mpm_acceptance_20261022.json
No compilation or application launch by Codex; live tests used external named pipe.

## Seed material UX follow-up

- User C++ build: existing MatterModelControls.cpp now provides the seed panel's
  shared domain-default substance picker and granular MPM/DEM/fluid model label.
- Minimal integration call in scene_ui_simulation_domains.cpp; no feature body
  added to the over-2000-line panel file. No shader changes.
- Verify Seed/Matter pickers remain synchronized; liquid/MPM/DEM labels match
  substance.get; errors are visible; reset+seed and save/load preserve material.
- UI, Python and IPC all use existing updateFluidDomain; no UI-only material state.
- Current live bed is granular MPM, NOT fluid (0 fluid / 12288 MPM / 10 DEM).
  Fluid-like appearance is not dismissed; pile/slope/penetration acceptance remains.
- Source checks passed, no compilation and no live acceptance of the new UI yet.


### DEM seed ownership and physical packing (2026-10-22)

- First Sand seed regression: resolve substance ownership and material before
  birth, after replacement clears old particles. Previously a first seed could
  become MPM and lock that owner until reset.
- UI, Python `rt.fluid.seed`, IPC `fluid.seed` share this core. DEM seeds use
  sphere-volume mass and a maximum-radius lattice independent of voxel/PPC.
  Variable radii keep deterministic IDs; smaller grains begin with gaps and settle.
  This is conservative initial packing, not dense equilibrium packing.
- DEM requested count is a lower bound when truncated (emitted + 1). Candidate
  work is bounded; capacity/scan cutoff has a separate DEM warning.
- User: C++ build only, no shader changes. Codex did not compile or run C++ tests.
- Run updated `scripts/test/matter_grain_birth_test.cpp`: physical spacing/mass,
  granular model, voxel independence, append collision rejection, budget, small box.
- Live: empty domain, builtin Sand, seed replace=true BEFORE any step.
  `fluid.matter_models`: DEM > 0, MPM = 0, reset_pending=false. Repeat after
  reset/reseed. Replace existing incorrectly born MPM particles.
- Identical box/radius with different voxel/PPC must give identical DEM count.
  Custom granular_transport=mpm and Water retain voxel/PPC sampling and owners.
  MPM default + DEM source must seed MPM, even with domain grains enabled.
- Drop Sand above floor to test impact scattering. A zero-velocity resting seed
  is not a bounce test. Restitution and friction were not changed.


### DEM seed column symmetry regression (2026-10-22)

Live read-only evidence after the ownership fix: Physics Domain 1 has 34,512
DEM grains, zero MPM/fluid owners, reset_pending=false, dry grains, zero liquid
bridges, restitution=0.5. Horizontal bounds remain the seed footprint; horizontal
momentum and spin energy are zero, maximum contacts per grain=2. This matches
perfectly aligned columns compressing/rebounding without lateral excitation,
not an adhesive material or MPM ownership fallback.

- Seed now uses deterministic bounded position jitter (10% of maximum radius
  per coordinate) with spacing 2.2002 R. The worst adjacent jitter extremes
  retain at least 2.0002 R separation along their separating axis. Bounds include
  the jitter margin. No velocity kicks or changes to material forces.
- Existing size_seed controls reproducibility through the same UI/script/IPC
  core. Seed positions remain independent of voxel/PPC and particle budget.
- Initial count is lower than the old perfectly touching lattice; this remains
  conservative packing, not a dense packing generator. The box needs room for
  the maximum diameter plus its jitter margin.
- Updated C++ regression assertions cover pair separation, containment, layer
  misalignment, seed repeatability/change, append rejection, budget-prefix stability.
  Codex did not compile/run C++ tests or launch the application.
- User build: ordinary C++, no shader change. Clear/replace and seed again: old
  live positions cannot acquire the new birth jitter. Run with settle-on-seed
  disabled for an observable drop. Verify horizontal spreading and nonzero spin
  after impact, no coherent column bounce, no held step or capacity overflow.
  Repeat with settle enabled and with radius variation; check a final stable pile.
