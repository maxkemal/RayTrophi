# Güncel kabul: MPM + grain GPU temas devamı

> **2026-10-09 yeni kapsam: build bekletiliyor.** Kullanıcı sıvı ve gaz sparse
> sisteminin tamamını kaynakta bitirip tek build/test istedi. Aşağıdaki eski
> parti checklist'leri son build için saklıdır; consumer migration kapanmadan
> build istenmez. Güncel çalışma:
> [MATTER_SPARSE_FULL_STORAGE_IMPLEMENTATION.md](MATTER_SPARSE_FULL_STORAGE_IMPLEMENTATION.md).

## Son kaynak partisi: compact MAC transfer + FLIP (revision 8)

1. Kullanıcı `RayTrophiStudio/source/shaders/compile_sim_shaders.bat` ve C++
   build alır. Yeni `SparseMacTransferGpu.cpp` proje kayıtlarında. On bir yeni
   `sim_sparse_mac_*` SPV ve ortak gövde kullanan dense/indexed P2G/G2P SPV'leri
   aktif runtime shader dizinine birlikte deploy edilmelidir. Codex derlemedi.
   Bu çalışma ağacında root `compile_shaders.bat`'ın çağırdığı
   `compile_shaders.ps1` mevcut değil; burada mevcut simulation batch yolu
   veriliyor. Root driver eksikliği bu aktarım partisinde değiştirilmedi.
2. Kaynak denetimleri:
   `python scripts/test/check_sparse_mac_transfer_contracts.py` ve
   `python scripts/test/check_sparse_mac_transfer_math.py` PASS. İkinci test
   CPU oracle'dır, GPU/GLSL testi değildir. Existing dispatch/pressure/viscosity,
   Matter GPU/grain ve domain panel source denetimleri de PASS.
   Capability/descriptor freshness PASS (663 method). Genel
   `verify_descriptor_claims.py` enum list/string uyumsuzluğunda durdu; sonucu
   PASS değildir. Yeni transfer alanları ayrı source contract ile denetlendi.
3. Build sonrası uygulama kullanıcı tarafından açılır. Boş/durmuş sahnede,
   başka IPC testi koşmazken ayrı terminalden sırayla:
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer`
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer --viscosity`
   Her iki probe dense/sparse reset parity, mass/momentum, centroid/hız ve
   `transfer_sparse_used`/`flip_sparse_used` seçimini ister. Silent dense
   fallback FAIL'dir. Geçici domain/source/madde cleanup edilir.
4. Tek geniş paket iki yeni probe'u da içerir:
   `python scripts/test/rt_test_matter_acceptance_ipc.py --extended`.
   Named-pipe Windows error 5 sandbox kaynaklıysa dış probe escalation ile
   tekrar koşulur; embedded script workspace kullanılmaz.
5. Tam sparse MAC kabulü henüz yok. Transfer havuzu compact, fakat host MAC,
   projection/contact device publication, weight ve FLIP fallback scratch
   hâlâ dense. Pool bytes ilave capacity'dir; VRAM/FPS kazancı raporlanmaz.
   Moving wall, topology değişimi, büyük 2D GPU, tüm sparse consumer'lar ve gaz
   depolaması sonraki kapılardır. Son iki probe henüz canlı koşulmadı.

## 2026-10-09 tamamlanan canlı sparse probe'lar

rt_test_sparse_pressure_ipc.py ve --viscosity dış süreçte sıralı PASS.
Dense/sparse mass/momentum ve centroid/hız eşitliği geçti. Son sahne boş/durmuş;
geçici domain/source/madde kaldırıldı. matter_sparse_live_2026-10-09.json raporu.
Pressure pool 13,364 byte; viscosity fixture pressure 99,492 + RHS 56,300 byte.
Host süreleri karışık (pressure 17.26/35.95 ms, visc 22.67/19.95 ms dense/sparse);
native GPU/FPS/cinematic kabulü yapılmadı. Full MAC/gas ve büyük 2D gate açık.

## Shader çıktı yazma devamı — kullanıcı build aracı

1. Hata footer'ı RayTrophiStudio/compile_shaders.bat'ten geliyor. Script artık
   compile_shaders.ps1'e delege eder; shader önce GUID geçici dosyaya derlenir,
   başarılı SPIR-V header kontrolünden sonra mevcut output değiştirilir.
   Compiler hatası eski SPV'yi truncate etmez; publish kısa I/O kilidini retry eder.
2. Aynı root için yeni ana script'in eşzamanlı ikinci koşusu named mutex ile
   reddedilir. Doğrudan glslc/diğer legacy batch'ler bu mutex'e katılmaz.
3. PowerShell 5 -ValidateOnly source plan PASS (275 entry, compiler çağrılmadı).
   check_shader_output_publish.ps1 yalnız AST publication fonksiyonunu çalıştırır;
   valid replace, invalid çıktı ve kilitli target korunması, unlock sonrası retry.
   Bu shader derlemesi veya GPU test değildir. Build kullanıcıda.
4. Kullanıcı aynı RayTrophiStudio/compile_shaders.bat dosyasını tekrar çalıştırır.
   Publish kalıcı olarak başarısızsa artık compiler syntax hatası diye gizlenmez;
   target yolu ve işletim sistemi hatası görünür. Runtime deployment da aynı
   güvenli publication yolunu kullanır. Her SPV atomiktir; bütün batch tek
   transaction değildir ve derleme sonuçları ancak kullanıcı koşusunda doğrulanır.

## Ek kaynak partileri: sparse tile temel ve GPU pressure

### Son devam: P2G/G2P/FLIP 2D dispatch

1. Kullanıcı son shader batch + C++ build: sim_dispatch.glsl kullanan P2G/G2P
   (normal + indexed), clear/normalize/window, sim_matter_copy/clear/contact
   SPV'leri birlikte güncel olmalı. Shader ABI'leri değişmedi; old SPV + yeni
   C++ 2D planı birlikte kullanılmaz. Core revision 7.
2. `check_fluid_gpu_dispatch_contracts.py` PASS; C++ test hedefinde
   fluid_gpu_dispatch_test.cpp beklenen çıktı:
   `PASS balanced GPU dispatch coverage, padding and 32-bit boundary`.
   Codex C++ testini derlemedi/çalıştırmadı.
3. İlk iki-satır sınırı 65535*256 + 1 eleman: 32768x2 logical workgroup; ikinci
   satır kaybolmamalı, aynı yüz/particle iki defa işlenmemeli. Sinsi: küçük
   fixture PASS olup >16.7M elementli gerçek dispatch'in hâlâ bozuk olması.
   C++ shape testi GPU kernel kabulü değildir; bu büyük-case GPU gate'i açık.
4. Diğer ajan DEM dosyalarında devam eder; bu kaynak partisi o dosyaları
   değiştirmez. Son IPC paketi iki tarafın source/build partileri tamamlandıktan
   ve diğer ajanın mevcut canlı testleri bittikten sonra bir kez sıralı koşulur.
5. Ana MAC/FLIP depolaması hâlâ dense. Bu kaynak hazırlığı memory/FPS veya tam
   sparse solver kabulü değildir; native timestamp ve gerçek resident byte gerekir.

### Son devam: sparse viscosity MAC RHS

1. Kullanıcı shader + C++ build: sim_sparse_viscosity clear/mark/capture/sweep
   ABI 13/68 ve dense sim_fluid_viscosity_rbgs ABI 11/52 ortak GLSL gövdesinden
   yeniden derlenir. SparseViscosityGpu.cpp/.h proje kayıtlarında; Codex derlemedi.
2. `python scripts/test/check_sparse_viscosity_contracts.py` PASS. C++/shader
   runtime testi değildir; fizik eşitliği sonraki gate'te ölçülür.
3. Son sıralı runner sparse_viscosity işini de içerir. Doğrudan probe:
   `python scripts/test/rt_test_sparse_pressure_ipc.py --viscosity`.
   Boş durmuş sahne, diğer agent testleri bitmiş ve son build alınmış olmalı.
   Nu=0.02 geçici maddeyle dense/sparse mass/momentum, hız/centroid ve pool
   resident byte eşitlik/savings aranır; sahne ve türetilmiş madde cleanup edilir.
4. Sinsi: tile sınırındaki pozitif MAC yüzünün sahibi dahil edilmezse viscosity
   yalnız bazı yüzlere uygulanır. Sparse bayrağı tek başına kabul değildir.
   Hareketli duvar ve yüksek nu/dt yakınsaması son fizik kapısında kalır.
5. Ana velocity, P2G/FLIP ve gas grid hâlâ dense. Yeni ölçü sadece viscosity RHS
   pool'udur. Full sparse solver/FPS başarısı iddia edilmez. Core revision 6.

### Güncel devam: Vulkan sparse pressure bağlı, tam sparse MAC/gaz açık

1. Kullanıcı `source/shaders/compile_sim_shaders.bat` + C++ build: yeni on
   sim_sparse_pressure_* shader ABI 16/80; değiştirilmiş divergence/gradient ve
   CG/window reduction SPV'leri birlikte güncel olmalı. SparsePressureGpu.cpp
   proje kayıtlarında. Codex build veya shader derlemesi çalıştırmadı.
2. Aşağıdaki S0 C++ testi ve `check_sparse_pressure_contracts.py` önce gelir.
   Son statik audit PASS; C++ testi henüz derlenip çalıştırılmadı.
3. Diğer ajanın canlı testleri bittikten sonra tek son paket:
   `python scripts/test/rt_test_matter_acceptance_ipc.py --extended`.
   Yeni sparse_pressure probe dense/sparse aynı fixture, mass/momentum,
   centroid/hız ve pressure resident memory karşılaştırır; scene cleanup yapar.
4. Kullanılınca UI/script/IPC pressure_sparse_used=true; aktif/allocated tile ve
   gerçek resident byte görünür. Sessiz sinsi sonuç: CPU/dense fallback başarılı
   görünüp sparse diye kabul edilmesi. Probe sparse flag ve pressure_on_gpu ister.
5. Bu devam yalnızca pressure scratch'ı sparse yaptı. MAC/viscosity/host grid,
   gaz storage/pressure ve GFM sparse portu açık. Full sparse/FPS kabulü sayılmaz.

### Önceki S0 kaynak kontrolü

1. Kullanıcı C++ test hedefinde `scripts/test/sparse_tile_grid_test.cpp`:
   `PASS sparse tile storage, MAC ownership, remap and swept support` beklenir.
   FAIL adres sahipliği, tile remap veya seyahat halo'su sorununu gösterir.
   Codex C++ derleme/test çalıştırması yapmadı.
2. `python scripts/test/check_domain_panel_fields.py`: PASS beklenir. Initial
   Model/Phase/Constitutive Model madde şemasına taşındı; panelde geri eklenmez.
3. S0 host temelinin aktif solver'da bellek/FPS kazancı göstermesi beklenmez.
   En sinsi yanlış kabul: eski dense storage + window dispatch'i gerçek sparse
   bellek sonucu saymak. S1–S4 açık; takip MATTER_SPARSE_TILE_GRID.md.
4. Aşağıdaki ortak saat/dinamik destek kabulü aynı son test paketinde korunur.

> **Durum:** AKTİF — 2026-10-08. Önceki ortak state partisi kullanıcı build PASS.
> Bu devamın C++/shader ve canlı kabulü açık. Ana plan kapanmadı; build kullanıcıda.

## Son kaynak partisi: dinamik destek + tek kabul paketi

Kaynak partileri tamamlanınca kullanıcı shader + C++ build alır; devam eden
testlerle paralel yeni IPC koşusu açılmaz. Bu devam derlenmedi/canlı kabul edilmedi.

1. Sekiz `sim_grain_fluid_*` shader (clear/hash/cells/solid/refresh/delta/reaction/apply)
   ABI 21/96; eski 13/32 SPV ile yeni kaynak test edilmez. C++ coupling pool release
   ve yeni prepare params/motion imzası kaynakta. Proje build'i kullanıcıda.
2. Kullanıcı C++ hedefinde matter_common_clock_test.cpp ve matter_grain_coupling_test.cpp.
   Güncel konumda Fluid destek edinir, uzaklaşınca bırakır, geri dönünce edinir.
   Partikül kimliği/kütlesi değişmeden trilinear lump mass partition korunmalı.
3. Boş durmuş sahnede tek dış komut:
   `python scripts/test/rt_test_matter_acceptance_ipc.py --extended`.
   Tek rapor `docs/dev/matter_acceptance_<UTC>.json`; H1 detayları koşu başına
   ayrı snapshot. Extended H1 fixture'ları disabled olarak incelemeye bırakılır.
4. Dinamik probe: uzaktaki Water grain desteğine GIRER, sonra ÇIKAR; destek
   device_rebin_each_tick, ortak saat enabled, held=false, fluid/grain/mpm=1/1/1.
   Liquid impulse residual <1e-6 N s. Sinsi: sıvı hareket eder fakat eski hücre
   eşlemesi yüzünden coupling hep 0 ya da hep aktif kalır. İkisi FAIL.
5. Native GPU kernel timings ayrı; host stage wall time, dispatch/transfer byte,
   working set ve alt adımlar ayrı ölçülür. Profiler unsupported ise maliyet
   ölçülmemiştir; 24 fps PASS sayılmaz. Native profiling instrumentation açık diye
   raporlanır, production performance proof değildir.
6. Tam kapanışa eksik kaynak/kabul: üç-owner porous pressure reaction, MPM/grain
   deformasyon-angular ve yüksek hızlı penetration/dt yakınsaması; 1.3M+ cinematic
   ölçek/boundary gerçek GPU maliyeti; termal enerji/elastik/gaz son kapıları.
   Runner implemented_gates_passed dönebilir, all_unified_matter_gates_closed=false
   kalır. Fizik terimlerini kapatarak veya hidden cap ekleyerek maliyet kapısı geçilmez.

## Önceki kaynak partisi: ortak saat

1. Kullanıcı shader + Release x64 build: altı `sim_grain_mpm_*` (count dahil,
   ABI 13/32, revision 2), beş `sim_grain_fluid_*` (ABI 13/32). Yeni
   MatterGrainFluidGpuCoupling.cpp/.h ve MatterCommonClock.h proje kayıtlarında.
   Codex build/uygulama başlatmadı; eski SPV'lerle yeni kaynak kabul edilmez.
2. Kullanıcı C++ test hedefi: `scripts/test/matter_common_clock_test.cpp`.
   Even ping-pong, authored cap, 100000 adımın keyfi 4096 ret almaması,
   NaN/Inf/index width ve grid interval'in CFL üst sınırını aşmaması.
3. Boş durmuş sahnede dış terminal: `python scripts/test/rt_test_matter_transport_ipc.py`.
   Common clock enabled, held=false, contact impulse >0; transport_steps >=
   continuum_grid_steps>0. Water eklenince sahipler 1/1/1, liquid_reaction_on_gpu=true,
   liquid momentum residual <1e-6 N s. Sinsi hata: host ikinci tepki ekler veya
   fluid adveksiyonu contact-corrected canonical hızı kullanmaz. İkisi FAIL.
4. MPM bloğu→kum ve kum→MPM havuzu: 24/60/120 fps, penetration/mass/momentum;
   saf DEM ve grain'siz MPM regresyonu. CFD-DEM destek weights hâlâ frame sampled.
5. Sinematik ölçek: 1.3M+ sahne, authored budget ve gerçek GPU buffer kapasitesi.
   Coordinator'da 1M particle/64-neighbour ret kapısı olmamalı. GPU kernel timestamp
   ile maliyet ölç; host_wait tek başına 24 fps PASS değildir. Porous owner izolasyonu,
   deformasyon support/angular yakınsama ve ana plan final kapıları açık.

Kaynak denetimi: grain/common-clock ABI, GPU indexed kontratları, IPC descriptor ve
capability, Python AST/proje/filters/JSON PASS. C++/shader/canlı kabul değildir.

## Önceki kaynak partisinin listesi (tarihsel)

1. **Final kaynak partisi sonunda shader + Release x64 build (kullanıcı):**
   `RayTrophiStudio/source/shaders/compile_sim_shaders.bat` beş `sim_grain_mpm_*`
   SPIR-V üretmeli. C++ projesinde `MatterGrainMpmContact.cpp/.h` kayıtları var.
   Eksik/bayat shader teması hazır gösterip geçiştirilmez, adım açık hatayla tutulur.
2. **Hızlı sahibi/modeli kontrol et:** Substance → granular_transport: Sand/Gravel/Ice
   dem, Soil mpm. Grain açıkken MPM material satırları görünmeli. Active Matter sayıları
   fluid/grain/MPM olarak ayrı. Yanlışsa solver değil doğum/transport çözümlemesi bozuk.
3. **Boş durmuş sahnede dış IPC smoke:**
   `python scripts/test/rt_test_matter_transport_ipc.py`. Beklenen grain=1, mpm=1,
   ready=true, held=false; GPU contact events ve grain impulse >0, residual
   <max(1e-7, impulse*1e-4). Ardından Water eklenince üç sayının hepsi 1 kalmalı.
   Test kaynaklarını temizler, kaydetmez. ★ En sinsi hata: ready=true/held=false ama
   temas dürtüsü sıfır; ya da Soil sessizce ikinci DEM tanesine dönüşür. İkisi FAIL.
4. **Core kaynak regresyonları (kullanıcının test hedefi):**
   `matter_substance_state_test.cpp`, `matter_grain_coupling_test.cpp`,
   `matter_grain_params_test.cpp` (MPM spin/pile dışında). MPM, sıvı
   mass/volume/momentum bin'ini değiştirmemeli; parcel_cell=-1, drag/su exchange
   yalnız Fluid sahibine gitmeli. Kimlik ve doğum kütlesi regresyonlarını koru.
5. **Regresyon ve fizik yakınsaması:** saf DEM H1 suite; grain kapalı Soil MPM;
   kum→Soil havuzu ve hareketli Soil bloğu→kum. 24/60/120 fps karşılaştır, kütle/
   momentum ve en derin penetration ölç. Normal temas var diye şekil/maliyet PASS
   sayma: MPM pozisyonu DEM frame'i boyunca sabit olduğundan yüksek hızda risk var.
6. **Üç sahip + collider:** Fluid/MPM/grain aynı domain. Fluid drag+kaldırma açık,
   porous projection MPM varken kapalı olduğu UI/IPC raporunda görünür. Ayrı owner
   weight'leri yapılana kadar B5'in üç-sahip eşitliği kapanmaz. Disk cache/render
   hazırlığı/bridge başka ajanın alanı; bu partide değiştirilmedi.
7. **Ana plan final kapıları:** ortak üç-sahip scheduler; deformasyonla temas
   desteği/angular momentum; per-substance MPM parametre tüketimi; termal enerji;
   karma elastik MPM/gaz-grain; save/open/legacy/numerical tuning; T0 uzman/Matter
   eşitlikleri. T6 uzman domain sökümü bu kabul sonuçları olmadan yapılmaz.

Kaynak denetimi: grain/MPM ABI ve dispatch sözleşmesi PASS; IPC capability ve
663 metot descriptor eşliği PASS; AST, Release aynaları, proje/filters ve JSON PASS.
Bunlar C++/shader derlemesi veya canlı fizik kabulü değildir.

## Ek parti (ayrı ajan): gizli tavanlar söküldü — grid 512, DEM 100k/1M, çarpıştırıcı 4096, Max Particles 10M

Kural: kodda ayar tavanı yok; panelde yalnız sürükleme aralığı (Ctrl+tık ile
üstü yazılır). Kalan tek tavanlar **indeks genişliği** (doğruluk sınırı) ve
kendi mesajlarıyla raporlanan bellek/cihaz sınırları.

0. **Shader derle:** `compile_sim_shaders.bat` (sim_matter_grain*.spv). Shader
   revizyonu 16→17. Eski .spv ile adım **held** olur ve "revision" der — bozuk
   değil, derlenmemiş demektir.
1. **Grid tavanı (hızlı, bağımsız):** IPC `fluid.set_param {domain, max_auto_resolution: 768}`
   → `fluid.get` `max_auto_resolution` 768 kalmalı (önceden 512'ye geri yazılıyordu).
   Enforce budget kapalıyken `resolution` en büyük eksende ~768'e çıkmalı.
   Panelde Max Auto Resolution'a Ctrl+tık 3000 yazılabilmeli.
   Bozuksa: değer 512'ye dönüyorsa eski binary.
2. **Max Particles:** `fluid.set_param max_particles: 50000000` → `fluid.get` 50M;
   panel 50M göstermeli (önceden 10M gösteriyordu).
3. **DEM >1M (2D dispatch):** Sand emitter'lı sahneyi Max Particles 3M ile oynat.
   1M'yi geçince taneler hareket etmeye devam etmeli; `fluid.matter_models`
   `step_held` false. Durursa mesajı oku: bütçe / storage-buffer / 2^24 indeks
   sınırından hangisi olduğunu ayrı söyler.
   ★ Sinsi: 1M üstünde **taneler hareket ediyor ama bir kısmı çift/titrek** —
   bu 2D katlamanın yanlış olduğu anlamına gelir (y satırları aynı indeksi
   okuyor). Tane sayısı ile `pile.grains` eşit, `scattered_fraction` 100k
   altındakiyle benzer olmalı.
4. **Çarpıştırıcı >4096 yüz:** yoğun bir mesh (≥50k üçgen, ör. subdivided zemin)
   çarpıştırıcı yap. Adım held olmamalı; `runtime.collider_faces` gerçek sayı.
   `host_ms` ilk karede BVH kurulumunu öder, statik çarpıştırıcıda sonraki
   karelerde ödemez. Hareketli büyük mesh her kare yeniden kurar (beklenen).
   Bozuksa: ilk kare saniyeler sürüyorsa patch birleştirme hâlâ O(N²).
5. **C++ test:** `scripts/test/matter_grain_collider_bvh_test.cpp` PASS
   (patch birleştirme union-by-size'a geçti; eşitlik/eşitsizlik iddiaları aynı).

6. **Render instance tavanları kalktı:** fluid splat 1M ve klasik parçacık
   toplamı 300k (`ParticleRenderBridge.cpp`) söküldü. Tek tavan cihazın TLAS
   `maxInstanceCount`'u (genelde 16.7M): `VulkanDevice::createTLAS` fazlasını
   kırpar ve bir kez ERROR loglar. 1M+ splat'lı sahnede Rendered modda tüm
   parçacıklar görünmeli; log'da `exceed the device maxInstanceCount` yoksa
   sınıra gelinmemiştir. ★ Sinsi: parçacıkların bir kısmı RT'de yok ama
   raster'da var → bu log'a bak.

Kalan bilinen tavanlar: SurfaceSDF modunun raster uyumluluk proxy'si 32768
(seyreltme/LOD, parçacık sınırı değil), eski `GasSimulator` 512/256³ (ayrı iş).
Diğer ajanın `sim_matter_contact.comp`'u da 1D dispatch: ~16.7M thread'de aynı
sınır; aynı `setLinearGroups` + 2D katlama kalıbı uygulanmalı.

### Canlı sonuç (2026-10-08, RTX 3060 12 GB, build 13:54) — `scripts/test/rt_test_scale_ceilings_ipc.py`

- **A grid PASS:** `max_auto_resolution` 600 ve `max_particles` 50M yazıldığı gibi
  kaldı; voxel .007 ile grid **572×29×572** kuruldu, düğme 512'ye geri yazılmadı.
- **B DEM:** 32 258 yüzlü düz terrain çarpıştırıcı (eski 4096) ve **1 009 066 tane**
  held olmadan adımlandı (eski 100k / 1M). r=8 mm'de 2622 substep, ~40 s/kare.
  Enerji sınırda, taneler domain içinde → 2D katlama tutarlı.
- **Buldu, düzeltildi (derleme bekliyor):**
  1. 1.09M'de held: "grain buffer of 2224 MiB exceeds the device storage-buffer
     limit of 2048 MiB". Sayının kendisi sığıyordu (history 1.09M×1536 B ≈ 1.6 GB);
     taşıran büyüme payıydı (1.5×). Pay artık cihaz sınırına göre kırpılıyor
     (`MatterGrainGpu.cpp`). Yeniden: `--grains 1300000` held olmamalı.
  2. Tane doğumu kaynak başına kare başına **262 144 deneme** ile sınırlıydı
     (`MatterDomainSources.inl`) → kare başına ~200k doğum. Artık istekle ölçekleniyor
     (büyük isteklerde 4×). Yeniden: `--window-frames 2 --grains 600000 --source-radius 1.7
     --grain-radius .008` ilk karede ~600k doğurmalı.
- **Kalan gerçek tavan (bu GPU):** contact history tek buffer'da tane başına 1536 B;
  Vulkan storage-buffer sınırı 2 GiB → **~1.39M DEM tanesi**. Aşmak için history'yi
  birden çok buffer'a bölmek gerekir (ayrı iş). Mesajı artık bunu söylüyor.

### Canlı sonuç 2 (kullanıcı build'i, aynı gün)

- Doğum denemesi düzeltmesi tuttu (kare başına 200k → 329k), ama her kare **tam
  329 272** doğdu: APIC sıvısının aşırı paketleme koruması (`spawn_cells × 16`,
  1.7 m küre / 0.1 m voxel) tane doğumuna da uygulanıyordu. Tanede atlanıyor
  (`MatterDomainSources.inl`; tane çakışma filtresi zaten var). Sıvı doğumu
  değişmedi. Yeniden: aynı komut, 3 karede 1.3M'ye ulaşıp held olmadan adımlamalı.
- Kalıcı değil, yalnız bu sahnede: 987 816 tanede 6 kare, enerji sınırda.

### Canlı sonuç 3 — PASS
1.3M tane 3 karede doğdu, 6 kare held olmadan adımlandı (büyüme payı kırpması
dahil), enerji sınırda, 32k yüzlü çarpıştırıcı. ~50 s/kare: 2622 substep, sınır
`damping` (r=8 mm'de mutlak 4 N·s/m kayma sönümü × 24 temas slotu); son substep'te
yalnız 15 702 temas, tane başına en çok 3. Substep başına maliyet ~19 ms/1.3M tane.

### Parti 4: DEM substep — kayma sönümü kararlılık sınırından çıkarıldı (yalnız C++, shader değişmedi)

`MatterGrainGpu.cpp`: host CFL kayma dashpot'unu (7 c_t × 24 temas) sayıyordu;
shader o dashpot'u zaten temas başına kesiyor (bütçe bir substep'te kaymayı
ters çeviremez) → koşulsuz kararlı. Sınırda artık yalnız normal (restitution) dashpot.

1. **Maliyet (hızlı):** aynı ölçek komutu (1.3M, r=8 mm). Beklenen: substep
   2622 → **~712**, `substep_limit` yine `damping` (artık normal dashpot),
   kare ~50 s → **~14 s**. Hâlâ 2622 ise eski binary.
2. **Fizik regresyonu (asıl kabul):** `python scripts/test/rt_h1_grain_suite.py`
   — settle, coexist, repose (yığın açısı), G2 kollarının hepsi önceki gibi PASS.
   Kayma sönümü artık daha seyrek substep'te kesmeye takılabilir (r=8 mm'de etkin
   ~1.45 N·s/m; kayma ~0.5 ms'de söner, kare 16 ms) — görsel fark beklenmez.
   ★ Sinsi: yığın açısı birkaç derece **düşerse** (taneler daha uzağa kayıyorsa)
   kesme sönümü fazla zayıflatıyor demektir; sayı PASS bandında kalsa da
   önceki koşudaki repose_angle_deg ile karşılaştır.
3. Sonraki kaldıraç (yapılmadı): normal dashpot da 24 temasın hepsi aynı yönde
   varsayımıyla sayılıyor; doğruluk sınırı ~340 substep. Ölçüm 2'den sonra.
- **Parti 4 madde 1 PASS (16:18 build):** 1.3M tane, r=8 mm: substep 2622 → **712**,
  kare ~50 s → **~15.3 s** (3.3×). Serbest düşüş eğrisi öncekiyle aynı
  (com_y 2.1661 / 2.1656, KE 2540 / 2541 J). Madde 2 (süit) koşuyor.
- **Parti 4 madde 2 — süit (16:18 build): 14/17, kalan üçü bu partiyle ilgisiz.**
  settle: substep 164 → 138, sınır artık `accuracy`; kalan enerji 4.9e-13 J/tane (önce 4.5e-13).
  repose (önce → şimdi): μr .05 18.2→24.7 (tepe alçaldı: açı uydurması bu kolda gürültülü),
  .1 23.1→22.3, .3 34.8→35.3, .1 küçük tane 24.4→22.2. Tane boyutu bağımlılığı
  1.3° → 0.1°. Hesap: repose ve motion sahnelerinde kayma kesmesi **hiç devreye
  girmiyor** (dt eşiğin 4-6× altında) → fark yalnız dt (3.4e-5 → 4.1e-5 s) ve
  koşudan koşuya gürültü; "sönüm zayıfladı" etkisi yok.
  Düşenler: contracts (bayat 4096 dizesi — düzeltildi, tek başına PASS);
  panel_fields (diğer ajanın 493995a'sında kalkan 'Initial Model', 'Phase',
  'Constitutive Model' — SUBSTANCE_MIGRATIONS'a girmeli); motion (önceden de
  düşüyordu, 493995a kaydı sweep −0.015 m: küre çapı 0.15, merkez y .15 → tek
  katman yığının 2.5 cm üstünden geçiyordu. Merkez .07 yapıldı → **PASS**, sweep +0.067 m).

### Parti 5: DEM kararlılık sınırı ölçülen temas sayısıyla (yalnız C++, shader değişmedi)

`MatterGrainGpu.cpp`: Gershgorin sınırı 24 temas varsayıyordu; normal dashpot
yaylarla orantılı olduğundan toplam ζ √temas ile büyür → 712 substep'i o
belirliyordu. Artık temas = geçen karenin ölçülen maksimumu + 4, [12, 24]
(12 = eşit kürelerde öpüşme sayısı; süitte en yoğun yığın 8-10). Karede daha
fazlası ölçülürse kare yayımlanmaz, 24 ile yeniden koşulur (temas geçmişi o
karede sıfırlanır; `cfl_contact_budget_retry`). MPM/ortak saatle bağlı adım 24'te kalır.
Restitution / sönüm katsayıları değişmedi.

1. **Maliyet:** 1.3M komutu. `cfl_contacts_per_grain` 12 (ilk kare 24), substep
   712 → **~384**, kare ~15 s → **~8 s**. `fluid.matter_models grain_diagnostics.runtime`.
2. **Fizik:** `python scripts/test/rt_h1_grain_suite.py --only settle repose dense static convergence`.
   repose açıları ve settle kalan enerjisi Parti 4 değerleriyle ±2° / aynı mertebe.
   `cfl_contact_budget_retry` dense ve repose kayıtlarında **nadiren** true olmalı
   (her karede true ise +4 payı yetmiyor → hint sürekli 24'e döner, hız kazancı kaybolur).
3. ★ Sinsi: yığın kayıp patlarsa / KE tane başına yükselirse ama step held değilse,
   bir karede temas sıçraması yakalanmadı demektir — `max_contacts_per_grain` >
   `cfl_contacts_per_grain` olan ve retry=false bir satır arayın (olmamalı).
- **Parti 5 canlı (21:56 build): etkisiz** — `cfl_contacts_per_grain` 24 kaldı, substep 712.
  Sebep: tane adımı `mpm_contact` nesnesini **her zaman** alıyor (MPM parseli yokken
  de); koşul nesnenin varlığına bakıyordu. Artık `mpm_contact->active()`. Yeniden derle.

### Parti 6: DEM maliyet sayaçları (SHADER DEĞİŞTİ → compile_sim_shaders.bat; revizyon 17→18)

`grain_diagnostics.runtime.cost_last_substep` (son substep, tüm tanelerin toplamı):
`scanned_bucket_slots`, `grain_pairs`, `history_probes`, `contactless_grains`,
`collider_nodes_visited`. Tane sayısına bölerek okunur.
1. Shader derlenmeden koşarsa adım held: "shader revision mismatch" — beklenen, derle.
2. 1.3M komutu: Parti 5 beklentileri (cfl 12, ~384 substep) + sayaçlar dolu.
   Okunacak oranlar: scanned/grain_pairs (komşu listesi kazancı), history_probes/
   temas (geçmiş arama), contactless_grains/grains (havadaki pay → uyutma/multirate).
- **Parti 5+6 canlı (22:15 build, 22:17 shader):** 1.3M dökülen tane: `cfl` 12,
  substep **384** (ilk kare 712, ipucu 24'ten başlıyor), kare ~15 s → **~9.4 s**,
  yörünge aynı (KE 2539.6/2540.3 J), retry yok.
  Sayaçlar (son substep): bucket slotu 15.04M (11.6/tane), tane-tane temas 15 064 →
  **bulunan temas başına ~1000 slot okuması**; temassız tane %98.9; geçmiş araması
  temas başına ~1; BVH 1 düğüm/tane. GPU 7.2 s/kare (~19 ms/substep), host 1.4 s.
  → Komşu listesi (Verlet) en büyük kaldıraç. Yığın oranları süit kayıtlarında.
- **Parti 5 fizik (alt küme 5/6):** static, convergence, settle, dense, repose PASS.
  Uyarlamalı temas: dense cfl 12-14 (maks ölçülen 10), repose 12-13 (maks 9),
  **retry 0** (dense 0/246, repose 0/300). motion FAIL: küre itiyor (+0.022 m) ama
  eşik .03; önceki koşu +0.067 — tek katman yığının yerleşimi koşudan koşuya
  değişiyor, kütle merkezi ölçüsü bu test için kararsız (dt'den bağımsız).
- **Repose A/B (aynı build, `--repose-arms mu_r_.05 mu_r_.1 --repose-contact-resolution 51`
  = Parti 4 dt'si, 290 substep):** tepe .05: 0.255 (Parti 4: 0.254, Parti 5: 0.237);
  .1: 0.273 (Parti 4: 0.286, Parti 5: 0.267). Aynı dt'de iki koşu arası fark
  (0.286 / 0.273) gözlenen kaymayla aynı büyüklükte → dt etkisi tek koşuyla ayırt
  edilemiyor; Parti 5 kabul. Açı aynı dt'de 24.7° / 17.3° → **açı uydurması ±4°'lik
  kapıya göre fazla gürültülü**; fiziği değiştiren partiler için çok tohumlu koşu
  veya tepe/profil ölçüsü gerekir.
- settle: tanelerde kalan dönme enerjisi dünkü koşudan çok daha yavaş sönüyor
  (kare 480: 1.2e-4 J / dün 1.9e-16 J); bu sahnede dt değişmedi (138 substep).
  Dikey eksende dönme yalnız burulma sürtünmesiyle söner (varsayılan 0) — kaotik
  olabilir. Kapı yalnız "büyüyor mu"ya bakıyor. Açık gözlem, regresyon değil.

### Parti 7: DEM Verlet komşu listesi (SHADER + C++; revizyon 18→19) — `docs/dev/DEM_VERLET_LISTESI.md`

Önce `compile_sim_shaders.bat` (yeni: `sim_matter_grain_list_clear`, `sim_matter_grain_list_build`),
sonra build. Üç dönen bucket tablosu → tek tablo + tane başına 32'lik liste;
liste GPU'da bayrakla yeniden kurulur (pay 0.5 r), kare başında zorunlu kurulum.
Temas kümesi aynı olmalı (yalnız toplama sırası değişir) → fizik farkı float gürültüsü kadar.

1. **Revizyon:** shader derlenmeden koşarsa "shader revision mismatch" → derle.
   Kernel tablosu 17 bağlama bekliyor; eski .spv ile pipeline kurulmaz.
2. **Maliyet (1.3M):** `cost_last_substep.neighbour_candidates` ≈ taneler × liste boyu
   (dökmede çoğu 0) — önceki 15.04M bucket slotuna karşı. `neighbour_list_builds`
   kare başına ≥1; serbest düşüşte birkaç-onlarca, durgun yığında ~1.
   Kare süresi ~9.4 s'den aşağı. Substep 384 aynı kalmalı.
3. **Fizik (kısa):** `python scripts/test/rt_h1_grain_suite.py --only base history static settle dense wet`
   — hepsi PASS; settle kalan enerji ve dense sayıları Parti 5 ile aynı mertebe.
   ★ Sinsi: liste bir çifti kaçırırsa taneler birbirinin İÇİNDEN geçer ama adım
   held olmaz. Belirti: yığın normalden alçak/yayvan, `grain_pairs` aynı sahnede
   Parti 6'dakinden belirgin az. dense ve settle bunu yakalar.
4. Taşma: "grain neighbour list full" → 0.5 r kesmede 32'den fazla komşu (aşırı
   sıkışma); normal yığında olmamalı.

**Parti 7 sonuç (00:03 build, 2026-10-09 09:45–09:55):** madde 2 PASS (önceki oturum).
Madde 3: `base history static settle dense wet` → **6/6 PASS**, taşma mesajı yok.
- base: g = 9.8101 / 9.8100 (iki dt).
- static: tutma kolu sürünme 8.8e-10 m; statik kapalı kol 0.146 m kayıyor (ayırt ediyor).
- settle: kalan enerji 5.6e-13 J/tane (Parti 4: 4.9e-13). Dönme enerjisi kare 480'de
  1.9e-4 J (Parti 5: 1.2e-4; dün gece aynı build'le yarım kalan koşuda 8e-19) →
  koşudan koşuya kaotik, listeyle ilgisiz; açık gözlem olarak duruyor.
- wet: kuru rms 0.324 m / ıslak 0.205 m (2652 köprü); su kayması 6.5e-7 kg.
- dense: 4096 → 17.3 ms (p95 27.5), 16384 → **34.0 ms** (p95 46.6; Parti 3 kapısı ≤41.1);
  substep 194, sınır accuracy, maks temas/tane 9, kuru kütle sapması 0.
  Önceki koşuların dense temas sayısı kayıtlı değildi (log üzerine yazılıyor) — bu koşunun
  4873 / 72659 temas değerleri bundan sonrası için referans.

### Parti 8: DEM temas geçmişi tek bank, yerinde (SHADER + C++; revizyon 19→20) — `docs/dev/DEM_VERLET_LISTESI.md` "Sonraki 1"

Geçmiş tane başına **1536 B → 672 B** (+ 12 B blok haritası/sahip/maske; 1544 → 684).
İki bank + her yeniden sıralamada cihazda toplama (permute) yerine: her tanenin kalıcı
bir **geçmiş bloğu** var, host her kare tane→blok haritasını tutuyor (yalnız sıra
değişince / doğumda 4 B/tane yüklenir). Shader kayıtları yerinde günceller; bulunan kayıt
kendi slotuna, yeni temas alt adım başında boş olan slota yazılır. `sim_matter_grain_clear`,
`_permute`, `_permute_copy` söküldü (.comp/.spv, kernel tablosu, bat, vcxproj).
Fizik birebir aynı olmalı (yay değerleri aynı, yalnız saklama yeri değişti).
Not: plandaki "liste konumuna bağla + çift başına tek kayıt" yapılmadı — tek bankta j'nin
i'nin kaydını okuması i'nin yazımıyla yarışır; liste 32 slot > 24 geçmiş slotu.

1. **Revizyon:** önce `compile_sim_shaders.bat`. Eski .spv ile "shader revision mismatch"
   (19 ≠ 20) → derle. Eski `sim_matter_grain_clear/permute*.spv` dosyaları
   `x64/Release/shaders`'ta kalabilir; artık yüklenmiyor, zararsız.
2. **Kontrat (derlemesiz, PASS):** `python scripts/test/check_matter_grain_contracts.py`.
3. **Fizik (asıl kabul):** `python scripts/test/rt_h1_grain_suite.py --only base history static settle dense wet`
   — 6/6 PASS ve sayılar Parti 7 sonucuyla aynı mertebe (settle 5.6e-13 J/tane,
   dense 4873 / 72659 temas, 16384 ≤ 41 ms).
   ★ Sinsi: geçmiş her kare kaybolursa hiçbir şey çökmez, yığınlar yine durur.
   **static** bunu yakalar: tutma kolunda sürünme 8.8e-10 m'den mm'lere çıkar (kapı
   5e-4). history arm'ında `history_remapped` true iken static'in hâlâ tutması = blok
   haritası sıralamayı doğru izliyor. FRESH bayrağı temizlenmezse de aynı belirti.
4. **Maliyet (1.3M, `scripts/test/rt_test_scale_ceilings_ipc.py --grains 1300000`):**
   `working_set_bytes` ~1.1 GB düşmeli; substep süresi Parti 7 ile aynı. `history_probes`
   temas başına ~1'den birkaça çıkabilir (artık sabit sıra yok, dolu slot maskesi
   taranıyor) — oranı not et; maliyet farkı görünmüyorsa sorun değil.
5. **Tavan (opsiyonel, VRAM izin verirse):** `--grains 2500000` storage-buffer
   sınırına takılmadan adımlamalı (eski tavan ~1.39M; yeni hesap ~3.1M, en büyük
   buffer artık geçmiş 672 B × N). Takılırsa mesaj hangi buffer olduğunu söyler.
6. Taşma: "…or 24 history records were held in one substep" yeni mesaj — eski temaslar
   kaybolurken aynı alt adımda yenileri gelip toplam 24'ü aşarsa. Normal yığında görülmemeli.

### Parti 9: DEM host kare maliyeti (yalnız C++; shader değişmedi) — Parti 8 ile aynı build'de

**Ölçüm (2026-10-09, 00:03 build, 1.3M dökülen tane r=8 mm, 384 substep):** kararlı karede
duvar ~3.3 s, bunun yalnız **~0.95 s'i `gpu_wait`**. Tane host aşamaları ~1.7 s:
merge ~670, order ~525, prepare ~475, publish ~55 ms. Kalan ~0.6 s tane adımı dışında.
Komut: `python scripts/test/rt_test_scale_ceilings_ipc.py --skip-grid --grains 1300000
--grain-radius .008 --source-radius 1.7 --window-frames 2 --steps 6` (satırlar artık
`host_ms` ve `working_set_mb` basıyor).

Kaldırılan işler (hepsi gereksiz iş; fizik ve sıra bit düzeyinde aynı kalmalı):
- **order:** bölmedeki kimlik `stable_sort`'u (hemen ardından (hücre, kimlik) toplam
  sırasıyla yeniden sıralanıyordu); hücre sıralamasının eşitlik bozucusu artık anahtarda
  (sort içinden `particle_id`'ye rastgele erişim yok); `selectMatterParticles` tam kopya +
  compact + parçacık başına 31 dizi kopyası yerine sütun bazlı `gatherFrom`.
- **merge:** 1.3M düğümlü `unordered_set` kimlik kontrolü → düz `MatterParticleIdIndex`;
  parçacık başına kopya → sütun bazlı `copyRangeFrom`.
- **prepare:** yayımlanmış kimlik eşlemesi `unordered_map` → aynı indekste kalan tane için
  arama yok, gerekirse düz indeks. Kuru karede (ıslak/kuplaj/kuvvet alanı/ortak saat yok)
  48 B/tane sıfır kuplaj satırı artık kurulmuyor/yüklenmiyor (62 MB/kare); cihazda zaten sıfır.
- `FluidParticles` kopya yolları tek sütun listesinden (`forEachColumnPair`) geçiyor.

1. **Derleme:** `FluidParticles.h` şablon değişikliği birçok dosyaya dokunur; hata çıkarsa
   ilk bakılacak yer `forEachColumnPair` / `copyRangeFrom` / `gatherFrom`.
2. **Birim (hızlı):** `scripts/test/matter_grain_coupling_test.cpp` (bölme/seçme/birleştirme,
   değişen nüfus reddi) — PASS kalmalı.
3. **Maliyet (asıl):** yukarıdaki 1.3M komutu. Beklenen kaba hedef: merge ≤ ~150 ms,
   order ≤ ~250 ms, prepare ≤ ~350 ms; duvar ~3.3 s → **~2.2 s**. `gpu_wait` değişmemeli
   (~0.95 s). Hangi aşama beklenenden yüksek kalırsa bir sonraki kaldıraç odur.
4. **Fizik:** Parti 8'in süit komutu (base history static settle dense wet). Sıra aynı
   toplam sıralama olduğundan sayılar Parti 8 ile aynı mertebe.
   ★ Sinsi: sütun kopyasında bir dizi yanlış hizalanırsa (örn. `granular_*` ya da
   `pore_water_mass_kg` başka taneye kayarsa) hiçbir şey çökmez. **wet** arm'ı su
   kütlesi kaymasını (6.5e-7 kg mertebesi) ve köprü sayısını ölçer; o sayı büyürse buradan.

**Parti 9 ek (aynı build):**
- Yeni zaman alanları (`grain_diagnostics.runtime.host_ms`): `upload` (durum yükleme
  partisi), `download` (geri okuma; artık önüne bir `synchronize()` kondu, yani yalnız
  okuma — GPU hesabı `gpu_wait` içinde kalır), `step_total` (tüm tane-domain adımı;
  ölçek testindeki `wall_s` eksi bu = tane adımı dışında geçen süre, 1.3M'da ~0.65 s
  açıklanmamıştı). `cell_order: {disorder, resorted}`.
- **Koşullu hücre sıralaması:** popülasyon geçen yayımla aynıysa ve komşu çiftlerin
  ≤ %2'si sıra dışıysa (`kMatterGrainResortDisorder`) yeniden sıralanmıyor → durum
  cihazda kalıyor (`state_resident` true, ~88 MB yükleme yok). Doğum/emme varsa her zaman
  sıralanır. Durgun yığın zaten sıralıydı (önceden de resident); kazanç yavaş akan yığında.
5. Ölçek komutunda satır başına `resident` ve `cell_order` okunur. Dökme sırasında
   `resorted` true (doğum + düşüş) beklenir. ★ Sinsi: `resident` true iken yığın
   donuyorsa (taneler kare kare aynı) host durumu cihazla ayrışmış demektir —
   settle'ın `tail_rms_range_m` (≈5e-12) ve dense yığın sayıları bunu yakalar.

### Parti 10: DEM uyuyan taneler (SHADER + C++; revizyon 20→21, push constant 112→128 B) — `docs/dev/DEM_UYUYAN_TANELER.md`

Parti 8–9 ile aynı build. **Varsayılan AÇIK** (`sleep=true`, 2 mm/s, 0.2 s); A/B için
`fluid.set_grain_settings(domain=..., sleep=False)`. Blok başına 3 kelime (sahip/maske/
dinlenme); geçmiş haritası buffer'ı 4 kelime/tane. IPC tarifi yeniden üretildi.

1. **Revizyon:** `compile_sim_shaders.bat` (dört grain kernel'i). Eski .spv → "shader
   revision mismatch" (20/21). Push constant 128 B ile pipeline kurulamıyorsa kernel
   tablosu (`SimulationComputeVulkan.cpp`, 17, 128) ile shader bloğu uyuşmuyor demektir.
2. **Kontrat (derlemesiz, PASS):** `check_matter_grain_contracts.py`.
3. **Fizik, uyku AÇIK:** süit alt kümesi (Parti 8 komutu) + `repose`.
   Beklenen: settle/static/dense PASS; settle'da `sleeping_grains` kuyrukta 256'ya
   yakın, kalan enerji daha da düşük (uyuyanın hızı tam 0). static'in **sürünen** kolu
   (statik sürtünme kapalı) hâlâ ≥1e-2 m kaymalı — kaymıyorsa tane uyumuş demektir
   (eşik/süre fazla gevşek). repose açıları Parti 5/7 ile ±2° içinde.
   ★ Sinsi: **yığın havada donar** (uyuyan tane, altı çekilince düşmez) ya da
   **çığ erken durur** (repose açısı birkaç derece yükselir) — hiçbir şey çökmez,
   yığın sadece "biraz dik" görünür. repose'u `sleep=False` ile bir kez daha koşup
   açıları karşılaştır; fark ±2°'yi aşarsa eşik/süre ayarı ya da uyandırma kuralı yanlış.
4. **Maliyet:** 1.3M komutunu `--steps 30` ile (yığın otursun). `sleeping_grains`
   artmalı, `gpu_wait` düşmeli. Aynı komutu `--no-sleep` ile koşup karşılaştır.
5. Kapalı durumlar: hareketli çarpıştırıcılı sahne (`rt_grain_motion_ipc.py`) ve sıvı
   kuplajlı coexist kolları uyku yüzünden değişmemeli (`sleeping_grains` 0 ya da
   yalnız kuplajsız tanelerde).

### Sparse S1: karışık sıvı şeridinde compact kanonik MAC hızı (SHADER + C++) — `docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md`

Parti 8–10 (DEM) ile **aynı build**. Diğer ajanın son kaynak partisi (compact MAC
aktarım + FLIP, sparse grid storage/GPU transaction) de bu build'de ilk kez derleniyor.

1. **Shader derlemesi:** `compile_sim_shaders.bat` + `compile_fluid_window_shaders.bat`.
   13 yeni `sim_sparse_mac_*` girişi; gövdesi `.glsl`'e taşınan yoğun ikizler
   (divergence ×3, subtract_gradient ×2, matter_contact, zero_solid_faces, advect_tail,
   viscosity) da yeniden derlenmeli. glslc hatası `sim_mac_lane.glsl`'i gösterirse:
   ortak adres yardımcısı, `pc.nx/ny/nz` isteyen her gövde onu `pc`'den sonra include eder.
2. **Kontratlar (derlemesiz, hepsi PASS):** `check_sparse_mac_canonical_contracts.py`,
   `check_sparse_mac_transfer_contracts.py`, `check_fluid_gpu_dispatch_contracts.py`,
   `check_fluid_window_contracts.py`, `check_matter_gpu_contracts.py`,
   `check_sparse_viscosity_contracts.py`, `check_sparse_pressure_contracts.py`.
3. **Yoğun yol regresyonu (önce bu):** sparse KAPALI bir sıvı sahnesi ve karışık
   Su/Toprak — davranış öncekiyle aynı olmalı. Yoğun kernel'ler artık ortak gövdeden
   derleniyor; adres formülü aynı (kâhin doğruladı) ama derlenmiş kodun ilk koşusu bu.
4. **Canlı kabul (boş/durmuş sahne, dış process, sıralı):**
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer`
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer --viscosity`
   Beklenen: sparse kolda `transfer_sparse_canonical` true, `transfer_sparse_blocked`
   false, durum metninde "canonical"; dense/sparse centroid/hız farkı ≤2e-5, model
   kütle/momentum assertion'ları PASS (sınır değişmedi).
   ★ Sinsi: `transfer_sparse_blocked` true ise compact yol hiç koşmamış, şerit yoğun
   kalmıştır — prob bunu FAIL eder; panelde "Dense (compact blocked: neden)" görünür.
   ★ İkinci sinsi: parite PASS ama `transfer_sparse_canonical` false — eski binary
   ya da eski .spv (sahiplik hiç kurulmadı). Exe/spv zaman damgasına bak.
5. **Bellek (bilgi):** sparse kolda yoğun `GridDomainVelX/Y/Z` ve `ScratchVel*` tamponları
   yok. `perf.get_gpu_memory` ile iki kolu karşılaştır; kazanç yüz bankalarının
   yarısından azdır (hesap S1 notunda), sayfa havuzu ekstradır.
