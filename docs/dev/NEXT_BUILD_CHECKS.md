# Sıradaki derlemede kontrol edilecekler

> **Durum:** CANLI — her partide üzerine yazılır. Önceki sürüm (particle
> authoring + SSS) git geçmişinde: `git show 220bed8:docs/dev/NEXT_BUILD_CHECKS.md`.
>
> ★ En üstteki (kayıt 122 s: doku yeniden kullanımı) YENİ. Altındakiler
> önceki derlemelere girdi; doğrulanan maddeleri ✔ işaretli, kalanlar açık.

## Particle Faz 0 düzeltmesi + Faz 1 ilk parti: sistem adresleme, render IPC (2026-09-25)

> Parçacık ajanının partisi. Aşağıdaki "donma/autosave" bölümü diğer ajanın,
> ikisi aynı derlemeye girer ve birbirinden bağımsızdır.

**Ölçülen (13:09 exe, canlı IPC):** Faz 0 kontrol 1 ✔ (208/208), kontrol 5 ✔
(GPURequired adımı reddetti, parçacık y=5'te kaldı). **Kontrol 6 en sinsi
sonucu verdi:** bütün Auto satırları `backend_not_vulkan` — GPU yolu hiç
koşmamıştı; tablo "1.07x hızlanma" gösteriyordu, ikisi de CPU'ydu.
Kök: `syncSimulationWorld()` compute backend'ini yalnızca **grid domain**
backend'lerinden seçiyordu; parçacık politikası seçime hiç katılmıyordu, IPC
`particle.step` de seçimi hiç çalıştırmıyordu. CPU referansı:
`docs/dev/particle_baseline_2026-09-25_cpu.json`.

Değişen: `scene_data.h` (backend seçimi + sistem id sayacı),
`ParticleSimulation.cpp` (addEmitter uid tekilliği), `RtApi.h`,
`RtApiParticle.cpp`, `RtIpc.cpp` / `RtPython.cpp` (parçacık blokları
**çıkarıldı**), **yeni** `RtIpcParticle.cpp/.h`, `RtPythonParticle.cpp/.h`
(vcxproj'a eklendi), `scene_ui_forcefield.hpp` (sistem adı alanı), üretilmiş
descriptor'lar, overlay, `ipc_test_client.py`, `rt_api_smoke_test.py`,
`Probe-ParticleBaseline.ps1`. Audit derlemeden önce geçti (602 metot).

1. **Derleme / bağlama.** *Bozuksa:* `dispatchParticleIpc` ya da
   `registerParticleBindings` çözümlenemedi → iki yeni `.cpp` vcxproj'da yok.
   `rt.particle` modülü Python'da "has no attribute" → `RtPython.cpp`'deki
   kayıt çağrısı çalışmıyor.
> ✔ 2026-09-26 (23:21 exe, canlı): 2 ✔ (bütün `phase1:` satırları OK; 4 FAIL =
> 3'ü 23:40 sonrası eklenen deposit sayaçları, 1'i sahnede Default_Cube yok),
> 3 ✔ (`force[gpu]`, down = 12×capacity), 4 ✔ (baseline alındı, ballistic
> Δ ≤ 2.4e-7 m, plane Δ 0 — ama ilk koşunun plane satırları GEÇERSİZDİ:
> bulut düzleme hiç ulaşmıyordu, probe düzeltildi ve yeniden ölçüldü).
> Kalan: 1 (derlendi ✔), 5, 6, 7, 7b, 7c, 8.

2. **IPC testi** — `python scripts/ipc_test_client.py`. Bütün `phase1:`
   satırları OK olmalı. Asıl sinyal **`phase1: writes landed in B only`**:
   *Bozuksa* (`A=2 B=0`) açık bir `system_id` sessizce aktif sisteme düşüyor —
   bu partinin düzelttiği şeyin ta kendisi. `uid survives index shift` FAIL →
   uid çözümü index'e bakıyor.
3. **★ Faz 0 kontrol 6'yı tekrarla** —
   `.\scripts\ipc\Probe-ParticleBaseline.ps1 -Counts 2048 -Scenarios ballistic -Samples 10`
   Auto satırı artık **`force[gpu]`** demeli ve `down` ≈ `12 × capacity` bayt.
   *Bozuksa:* hâlâ `backend_not_vulkan` → `syncSimulationWorld`'ün parçacık
   dalı çalışmadı (Vulkan compute oluşturulamadı ve kilit düştü — SceneLog'a
   bak). Sahnede panelden bilerek "GPU (CUDA)" seçilmiş bir domain varsa bu
   değer DOĞRUDUR: kuvvet çekirdeği yalnız Vulkan'da; probe'u boş sahnede koş.
   `no_dispatch_support` → Vulkan compute bağlamı var ama dispatch yok.
   Probe artık bu durumda sarı uyarı basar ve `speedup`'ı boş bırakır.
4. **Tam baseline** — `-OutputPath .\particle_baseline.json`, JSON'u bana
   gönder. ★ **En sinsi sonuç:** `force[gpu]` ama `ballistic`
   `max_position_delta` büyük (>1e-4 m) → GPU kuvvet çekirdeği CPU'dan farklı
   fizik hesaplıyor; hız farkı "GPU hızlı" diye okunur, oysa sonuç yanlıştır.
5. **Python smoke** — uygulama içinden `rt_api_smoke_test.py`; son satırlardan
   biri `[rt-smoke] rt.particle phase-1 system addressing + render + uid: OK`.
6. **Panel ↔ çekirdek** — Simulation > Particles > System sekmesinde combo'nun
   altında **ad alanı**. Adı değiştir, Enter → hiyerarşide yeni ad. Başka bir
   sistemin adını yaz → kırmızı "already in use" ve alan eski ada döner.
   Tersini de dene: `particle.set_system {"system_id": N, "name": "X"}` →
   panel X göstermeli. `particle.set_render {"shape":"cube"}` → panelde
   "Ray Trace Shape" Cube olmalı.
7. **Davranış değişikliği (bilerek):** domain'siz bir parçacık sahnesi Auto'da
   artık Vulkan compute kullanır. Bir kıvılcım emitter'ı ile timeline'ı 3–4
   kez Oynat/Duraklat → TDR yok (`VULKAN_PARTICLE_PRESET_PAUSE_TDR.md`'deki
   çit düzeltmesi bu yolu da kapsıyor olmalı). Küçük sayılarda kare süresi
   hafif artabilir: kısmi GPU yolu her adımda hızı indiriyor, Faz 5'e kadar
   beklenen maliyet; `particle.stats.gpu_force_ms` ile görünür.
7b. **Domain backend combo'su** (Fluid/Gas domain paneli): sıra artık CPU →
   "GPU (Vulkan Compute - Recommended)" → "GPU (CUDA - Alternative, NVIDIA)".
   Eski bir projede CUDA seçili domain hâlâ CUDA göstermeli, Vulkan seçili olan
   Vulkan. *Bozuksa* (seçim kaymış görünüyorsa): combo sırası `backend_values`
   ile eşleşmiyor — kayıtlı değer değişmez, yalnızca etiket yanlış olur.
7c. **Deposit sayaçları IPC'de** — kamp ateşi sahnesinde oynatırken
   `particle.stats` → `grid_deposit_landed` > 0 olmalı. Zemin collider'ı
   olmayan spark emitter'da `grid_deposit_dropped_no_domain` büyük çıkar:
   parçacıklar y<0'a düşüp gaz kutusundan çıkıyor (2026-09-25 canlı
   sahnede alt sınır y=-3.06 ölçüldü). *Bozuksa* (üçü de 0, oranlar > 0):
   stats son adımı değil başka bir sistemi okuyor.
8. **Kaydet/aç** — bir emitter'ın `uid`'ini oku, kaydet, projeyi yeniden aç,
   `particle.get_emitter {"emitter_uid": <uid>}` aynı emitter'ı bulmalı;
   `list_systems` id'leri aynı kalmalı.
   *Beklenen, hata değil:* `clear_systems` sonrası yeni sistemler artık
   "Particle System 1"den değil kaldığı sayıdan devam eder (id'ler tekrar
   kullanılmaz — eski id'yi tutan script yeni sisteme yazmasın diye).

## "Bazen donup kalıyor" = otomatik kayıt 122 s ana iş parçacığında (2026-09-25)

**Ölçülen (canlı, sahneyi IPC ile kare kare sürerek):** 175 karede sim en
fazla ~170 ms/kare (ilk kare 1.3 s: collider ağırlıklarının ilk kurulumu).
Donma anında `cdb` ile ana iş parçacığı yığını: `autosave::tick → writeNow →
saveProject → serializeTextures → fwrite`. `project.autosave_status`:
`last_write_ms 122793`. SceneLog: `embedded: 64, png re-encoded: 64,
previous-embed-reused: 0, bin-bytes: 1224050340`, `serializeTextures 122586 ms`.
Sim DEĞİL — her 300 sn'de bir, CPU tek çekirdekte 64 dokuyu baştan kodluyordu.
Basit sahnede görünmemesinin nedeni az doku.

**Kök (iki kat):**
1. Önceki kaydın doku baytlarını aynen kopyalayan yol, manifesti ana `.rtp`
   JSON'undan okuyordu; format 3.0'dan beri dokular `.rtp.shared`'da → indeks
   HER ZAMAN boş → her kayıt (Ctrl+S dahil) her dokuyu yeniden kodluyordu.
2. `Texture::upload_to_gpu()` yükleme bekleyen dokuyu `markVulkanDirtyFull()`
   ile işaretliyordu, o da `save_dirty` kuruyordu → açılışta hiç
   dokunulmamış dokular "kaydedilmemiş değişiklik" sayılıyordu.

**Düzeltme:** manifest `.shared`'dan okunur (yoksa eski düzen); otomatik kayıt
/ "Farklı kaydet" kaynak projenin `.bin`'ini de indekse ekler; yeniden
kullanılan blob bayt bayt kopyalanır (JPG'ye yeniden sıkıştırma denemesi —
PNG decode — kaldırıldı); `markVulkanUploadPending()` yalnız GPU bayrağını
kurar. **Otomatik kayıt artık `save_dirty`'ye DOKUNMAZ** (aşağıda 3) ve
**oynatma sırasında ertelenir**, durdurunca yazar.
Değişen: `ProjectManager.cpp/.h`, `Autosave.cpp`, `Texture.h`, `Main.cpp`.
Yeni araçlar: `scripts/test/drive_fluid_frame_timing.py` (timeline'ı IPC ile
kare kare sürer, her karenin aşama dökümü), `probe_fluid_frame_spikes.py`.

**Ek (aynı parti, kullanıcı isteği):** otomatik kayıt artık **arka planda**
(Ctrl+S gibi) ve **`ProjectManager::saveProjectCopy`** ile yazar — proje yolu,
adı, son projeler, `is_modified`, doku `save_dirty` bayrakları HİÇ değişmez
(eski "yaz sonra geri al" ana iş parçacığında doğruydu, arka planda Ctrl+S'i
autosave.rtp'ye yazdırırdı). Bütün kayıtlar tek mutex'le sıraya girer. Ayar
açılıp kapatılabilir: **File > Project Save Options > Autosave** (+ aralık),
Template Hub'da da; IPC `project.autosave_set {enabled, interval_sec}`,
Python `rt.project.autosave_set`. HUD: "Autosaving..." → "Autosaved (x s)" /
"Autosave failed: ...". Ek değişen: `Autosave.h`, `RtApi*`, `RtIpc*`,
`RtPython.cpp`, `scene_ui.cpp`, `scene_ui_menu.hpp`, `TemplateHubUI.cpp`.

0. **Arka plan + HUD:** `project.autosave_set {"interval_sec": 60}`, bir şeyi
   değiştir, 1 dk bekle (oynatma KAPALI) → HUD "Autosaving..." sonra
   "Autosaved"; o sırada viewport dönmeye devam etmeli. `enabled:false` →
   `write_count` artmamalı; menüdeki onay kutusu da kapalı görünmeli.
   *Bozuksa:* HUD yoksa `progress().finished` kenarı kaçıyor.
0b. **★ SİNSİ OLAN — kimlik:** otomatik kayıt yazarken Ctrl+S → proje KENDİ
   dosyasına kaydedilmeli (`project.path` hâlâ senin .rtp'n), son projeler
   listesinde autosave.rtp görünmemeli.
1. **Hızlı:** projeyi aç, dokunmadan Ctrl+S. SceneLog (`x64/Release/SceneLog.txt`)
   `previous-embed-reused: 64`, `png re-encoded: 0` demeli; `serializeTextures`
   saniyeler (1.2 GB kopya), 122 s değil. *Bozuksa:* reused 0 ise indeks hâlâ
   boş ("Loaded N previous embedded texture entries" satırına bak); reused
   birkaç, re-encoded çoğunluk ise dokular açılışta hâlâ `save_dirty`
   (başka bir `markVulkanDirtyFull` yükleme yolu var).
2. **Otomatik kayıt:** IPC `project.autosave_now` → `last_write_ms` saniyeler.
   Oynatırken `project.autosave_status.seconds_until_next` negatife düşer ama
   yazma olmaz; durdurunca bir kez yazar.
3. **★ SİNSİ OLAN — boyama kaybı:** bir dokuyu boya, `project.autosave_now`
   (veya 5 dk bekle), sonra Ctrl+S, projeyi yeniden aç → boyama DURMALI. Hata
   vermeden kaybolursa: otomatik kayıt `save_dirty`'yi temizlemiş ve Ctrl+S
   projenin eski `.bin`'indeki boyanmamış baytları kopyalamıştır.
4. **Aynı dosyaya aynı dosyadan kopya:** Ctrl+S, projenin kendi `.bin`'inden
   okuyup geçici dosyaya yazar, sonra yerine koyar. İki kez art arda Ctrl+S →
   ikinci de reused 64, proje açılıyor ve dokular doğru.
5. **Uçtan uca:** `python scripts/test/drive_fluid_frame_timing.py 240 250`
   (mum sahnesi açık) → ilk kare dışında 1 s'yi aşan kare olmamalı ve betik
   `dispatch timeout` ile kesilmemeli (bu partide 175. karede kesildi = kayıt).

## Ara ara donma: sim zamanlaması + collider OBB önbelleği (2026-09-25)

**Şikâyet:** önbellek düzeltmesinden sonra çok hızlandı ama "bazen donup
kalıyor, CPU bekliyor". **Ölçülen (canlı, derleme öncesi):**
`sim.fluid.solid_face_weights` genelde ~0 ms, en fazla 840 ms (önbellek ara
ara yeniden kuruluyor); `loop.ui_draw` en fazla 7.4 s, `loop.frame_tail` en
fazla 3.1 s — ama sim adımının kendisinde zamanlayıcı olmadığı için HANGİ
aşama olduğu okunamadı. Collider'lar statik; SDF oynatmada yeniden
pişirilmiyor; timeline tick başına en fazla 1 kare ilerliyor (8 adımlık
yakalama döngüsü aday değil).
**Bulunan sürekli israf:** `resolveObjectOBBForSimulation` her çağrıda mesh'in
BÜTÜN üçgenlerini kopyalayıp iki kez tarıyordu — collider başına, adımda
birkaç kez. Voxel önbelleği isabet ederken bile `voxelize_colliders` ~8 ms
buydu. **Düzeltme:** OBB, yüzey önbelleği girdisi yeniden kurulana/silinene
kadar ezberleniyor (`sim_obb_memo_`). **Yeni zamanlayıcılar:**
`sim.timeline.update` (ebeveyn), `.config_sig`, `.source_poses`, `.step`,
`.capture_frame`, `.render_sync`, `.restore_frame`, `.restore_frame_disk`;
`sim.collider.obb_resolve`, `sim.collider.surface_cache_rebuild`.
Değişen: `scene_data.h`. Yeni: `scripts/test/probe_fluid_frame_spikes.py`.

1. ✔ (ölçüldü: <0.5 ms) **Hızlı:** oynat, `sim.fluid.voxelize_colliders` artık ~0-1 ms olmalı
   (önceden ~8). *Bozuksa:* `sim.collider.surface_cache_rebuild` her adımda
   sayıyorsa yüzey önbelleği her kare yeniden kuruluyor (geometri kuşağı her
   kare artıyor demek) — OBB ezberi de onunla birlikte her kare düşer.
2. ✔ (IPC sürüşüyle ölçüldü: sim ≤170 ms/kare, donma otomatik kayıttı — üstteki bölüm) **Asıl ölçüm:** terminalden
   `python scripts/test/probe_fluid_frame_spikes.py 300 250` çalıştır, sonra
   uygulamada oynat ve donmayı bekle. Her donma aralığı için hangi scope'un
   büyüdüğünü basar. Çıktıyı bana ver. *Okuma:* `sim.timeline.step` büyük ama
   `sim.fluid.*` küçükse çözücünün zamanlanmamış kısmı (basınç/GPU bekleme);
   `render_sync` büyükse yüzey (SDF) inşası; `capture_frame` büyükse kare
   önbelleğine kopyalama; hiçbiri büyük değil ama `loop.frame` büyükse sim
   dışı bir yol (untimed satırı).
3. **★ SİNSİ OLAN:** OBB ezberi, collider'ı hareket ettirince ESKİ kutuyu
   verirse hata vermez — sıvı objenin eski yerine çarpar. Kontrol: oynatma
   durmuşken collider objesini gizmo ile kaydır, tekrar oynat → sıvı yeni
   konuma çarpmalı; collider gizmo kutusu da objeyle birlikte gitmeli.

## Donmuş mum collider önbelleğini her karede öldürüyordu (2026-09-25)

**Şikâyet:** mum sahnesi collider'a değince CPU patlıyor. **Ölçülen (canlı):**
`loop.frame` 2586 ms; çözücü aşamalarının toplamı ~60 ms (pressure 44, P2G 4,
G2P 2.4). Sahnede iki `mesh_sdf` collider: makine + masa örtüsü (domain tabanını
kaplıyor). 992/1000 parçacık donmuş, 242 katı hücre.

**Kök:** katı örtü (donmuş mum) yüzlerini collider yüz ağırlıklarının ÜSTÜNE
yazıyordu; bu yüzden örtü varken her karede `collider_weights_init = false`
yapılıyordu → her karede tam reset + her mesh collider'ın analitik
süper-örneklemesi. Donma temasla başladığı için "collider'a değince" göründü.
**Düzeltme:** örtü kapattığı yüzleri eski değerleriyle kaydediyor
(`FluidGrid::overlay_weight_restore_*`), sonraki adımın başında geri yüklüyor;
collider önbelleği hep saf collider ağırlıklarını görüyor. Ayrıca sıvı adımına
`RTPERF_FRAME_SCOPE`: `sim.fluid.voxelize_colliders`, `.thermal_cool_freeze`,
`.solid_overlay`, `.solid_face_weights`, `.thermal_viscosity_field`.
Değişen: `FluidGrid.h`, `ParticleSimulation.cpp`. Yeni dosya yok.

1. ✔ (kullanıcı: "çok hızlandı") **Aynı sahne, aynı an** — `perf.list`: donmuş mum varken
   `sim.fluid.solid_face_weights` birkaç ms olmalı (önbellek isabeti), saniye
   değil; `loop.frame` yüzlerce ms'ye inmeli. *Bozuksa:* `solid_face_weights`
   hâlâ büyükse başka bir yol önbelleği bozuyor (collider imzası her kare
   değişiyor mu — `sim.fluid.voxelize_colliders` de büyükse evet). Başka bir
   scope büyükse darboğaz oradadır; ismini bana yaz.
2. **★ SİNSİ OLAN: geri yükleme yanlışsa** — iki yönde de hatasız görünür:
   (a) collider yüzleri açık kalır → sıvı makineden/örtüden hafifçe SIZAR;
   (b) örtü yüzleri kapalı kalır → mum eridikten sonra havada görünmez bir
   duvar kalır, sıvı "hiçbir şeye" çarpar. Kontrol: mumu dök, sonra termal
   zinciri kapat (donmuşlar serbest kalır) → sıvı makineden sızmamalı ve
   eski donmuş bölgeden serbestçe akmalı.

## Termal sıvı (mum) + yüzey ayarları IPC'de (2026-09-25)

**Amaç:** fotoğraf makinesinin üstüne erimiş mum dökmek. Mum, viskoz olduğu
için değil **soğuyup donduğu** için mum gibi görünür; motorda bu yön yoktu
(ısı zinciri yalnızca erime yönündeydi, sıvı parçacıklar gaz grid'i dışında
hiç soğumuyordu, emitter parçacıkları **0 K** doğuruyordu).

Kurulan:
- `APICSolverParams::thermal_*` + yeni `Fluid/FluidThermalLiquid.h/.cpp`:
  yüzeyde havaya, collider/kapalı duvara temasta daha hızlı soğuma (Newton);
  donma noktasının üstünde log-ölçekli ν(T) rampası (substance viskozite alanı
  kanalından, CPU ve GPU aynı alanı görür); donma noktasının altında
  **desteğe değen** parçacık donar (`kParticleFlagFrozen`), sabitlenir (v=0)
  ve mevcut katı-faz örtüsüne girer → üstüne dökülen mum katman katman birikir.
  Destek şartının gerekçesi başlıkta: havada donan parçacık askıda kalırdı.
- **Wax** preset'i (sıcakta ν=5e-6, 330 K'de donar, 25 K kalınlaşma bandı).
  ★ Wax dışındaki her preset zinciri KAPATIR (Water 293 K'de doğar, 330 K'nin
  altındadır — kapanmasa su donardı).
- Flow source **Pour Temperature (K)** (anahtar: `fluid_temperature_override`
  + `fluid_temperature_kelvin`). Kapalıyken artık domain ortamında doğar;
  `seedBox` de ortamda doğurur (ikisi de 0 K yazıyordu).
- Yüzey ayarları IPC/Python'da: `surface_resolution_multiplier`,
  `kernel_radius_voxels`, `particle_radius_voxels`, `narrow_band_voxels`,
  `smoothing_iterations`, `anisotropy_*`, `position_smoothing` →
  `fluid.set_param` (aralık dışı REDDEDİLİR) ve `fluid.get` (+ ölçülen
  `surface_grid_dim` / `surface_build_ms`).
- Panel: "Thermal Liquid" bölümü + canlı ölçüm; **Heat Conduction** satırı
  (bu alan script'ten yazılabiliyordu, panelde hiç yoktu); preset combo'suna
  **Molten Plastic** (eksikti — o preset'teki domain "Custom" görünüyordu) ve
  **Wax**.
- ★ Okurken bulunan hata: `fluid.set_substance_material` GAZ lookup'ı
  kullanıyordu → sıvı domain'de yalnızca "gas domain not found" dönebilirdi.
  Sıvı lookup'ına çevrildi. **Canlıda ölçülmedi** (uygulama kapalıydı) — 1.
  maddenin 6. fazı bunu ölçer.

Değişen: `APICFluidSolver.h/.cpp`, `FluidParticles.h`, `FluidLevelSet.h/.cpp`,
`ParticleSimulation.h/.cpp`, `RtApi.h`, `RtApiFluid.cpp`, `RtApiSimNodes.cpp`,
`SimulationNodes.h`, `RtIpc.cpp`, `RtPython.cpp`, `ProjectManager.cpp`,
`SceneSerializer.cpp`, `scene_data.h` (bake imzası), `scene_ui_forcefield.hpp`,
`scene_ui_simulation_domains.cpp`, üretilmiş `RtIpcMethodDescriptors.cpp`,
overlay JSON. **Yeni `.cpp`: `Physics/Fluid/FluidThermalLiquid.cpp`** —
`.vcxproj` + `.filters`'a eklendi. Shader değişmedi. Yeni IPC metodu yok
(var olanlara anahtar eklendi) → yetki tablosu değişmedi; audit geçti.

> ✔ 1 GEÇTİ (canlı, dışarıdan): 353 K doğuyor, ortam altı yok, 30. karede 241 /
> 240. karede 4660 donmuş, ν 1.1e-5..5e-2, kapatınca 0; set_substance_material
> artık "material not found" diyor (gaz lookup hatası canlıda doğrulandı).
> ★ IPC test script'leri uygulamanın script workspace'inden ÇALIŞTIRILMAZ —
> ana thread'i tutup kendi isteğini bekler; rt_ipc artık bunu açıkça reddediyor.

1. **Otomatik test** — boş/az dolu bir sahnede
   `python scripts/test/rt_test_thermal_wax_ipc.py`. Kendi rig'ini kurar
   ("WaxProbe"), sonunda PASS/FAIL. Fazlar bağımsız sırayla:
   - 1 yüzey anahtarları gidip geliyor, 5 reddediliyor. *Bozuksa:* anahtar
     dispatch'e ya da `fluid.get`'e ulaşmadı.
   - 2 wax→water zinciri kapatıyor. *Bozuksa:* `applyPreset` sıfırlaması.
   - 3 döküm sıcaklığı gidip geliyor.
   - 4 **dök–soğut–dondur**: sıcak doğuyor (≈353 K), ortam altına inen yok,
     ortalama düşüyor, ν alanı ≥10× aralık, 4 s sonunda `frozen > 0`.
     *Bozuksa:* "colder than ambient" → hâlâ 0 K doğuran bir yol var;
     "nothing froze" + `cold_unsupported > 0` → destek testi; `frozen=0` ve
     `cold_unsupported=0` → hiç soğumuyor.
   - 5 kapatınca donmuş parçacık kalmıyor.
   - 6 `set_substance_material` sıvı domain'i kabul ediyor.

2. **Panel** — Simulation > domain > APIC Solver: preset listesinde
   Molten Plastic ve Wax var. Thermal Environment gerçek etkin ambient'i;
   Thermal Liquid bölümü freeze/release eşiklerini, `tau=1/rate`, kaynakların
   ambient/custom dağılımını ve canlı sıcaklık farkını gösteriyor. Flow source
   `Initial Temperature = Domain Ambient | Custom` kullanıyor ve her durumda
   **Effective Birth Temperature** yazıyor. Şunları ayrıca doğrula:
   - birth = ambient iken “soğuyacak sıcaklık farkı yok” uyarısı görünür;
   - freeze > ambient ve kaynak ambient iken “hemen donabilir” uyarısı görünür;
   - custom 353 K, ambient 293 K, freeze 303 K iken tahmini contact/air eşik
     süreleri görünür ve sıcaklık telemetrisi zamanla düşer;
   - art arda eklenen point/object flow source'lar benzersiz ad alır; IPC'de
     `flow_source.get/update` ile ayrı ayrı hedeflenebilir.
   *Bozuksa:* eski exe (zaman damgası) veya yeni
   `scene_ui_fluid_thermal.cpp` proje girdisi eksik.

> ✔ 3 kullanıcı manuel doğruladı (2026-09-25): donma oluyor, sıcaklıkla değişiyor.
> Katman yığılması ve yan yüzeyde sabitleme ayrıca raporlanmadı.

3. **Asıl sahne: makineye mum** — domain'i makinenin üstünü saran dar kutu
   yap, preset **Wax**, makineyi **mesh collider** olarak ekle, flow source
   Pour Temperature ≈ 350 K, oynat. Görmen gereken: dökülen mum önce akar,
   yavaşlar, makinenin üstünde ve yan yüzünde **durur**; sonraki döküm
   öncekinin üstüne yığılır. Panelde Frozen artmalı.
   ★ **SİNSİ OLAN:** Frozen sayısı artıyor ama mum yine de akıp gidiyorsa
   (`fluid.get` → `solid_phase_cells` 0 kalıyor), donmuş katman bir hücreyi
   dolduracak kadar kalın değil — örtü eşiği (`solid_phase_fill`) seed
   yoğunluğuna bağlı. Görüntü "mum yapışmıyor" der, sayılar "dondu" der; bug
   raporu olarak gelmez. Çare: `solid_phase_fill` düşür ya da çözünürlük.
   İkinci sinsi olan: yan yüzeyde donmuş katman aşağı kayıyorsa sabitleme
   (pin) GPU yolunda kaybolmuş demektir — `Fluid::step`'teki kuvvet sonrası
   sıfırlama ya da G2P geri yüklemesi.

4. **Yüzey inceliği** — aynı sahnede Surface Detail 2–3 + Anisotropic Kernel.
   `fluid.get`: `surface_grid_dim` = sim grid × çarpan, `surface_build_ms`.
   Sim maliyeti (`fluid.step_stats`) değişmemeli; yalnızca yüzey kurulumu artar.

5. **Kaydet/aç** — Wax preset'li domain + döküm sıcaklığı olan kaynak →
   kaydet, aç. `fluid.get` thermal_* ve `flow_source.get` pour alanları aynı.
   *Bozuksa, ÖNCE bak:* ProjectManager'daki "parseDomain ... fluid block"
   tripwire log satırı. O satır varsayılanları basıyorsa bu, 2026-08-16'dan
   beri açık olan `.rtp` fluid_params kaybıdır — bu partinin hatası değil.

6. **Maliyet** — donmuş parçacık varken GPU advect tail atlanır (katı
   parçacık istisnası host'ta), sıvı başına kare başına bir grid gidiş-dönüşü
   geri gelir. `fluid.step_stats` baytlarını not et; anlamlıysa sıradaki iş.

Bilinen sınırlar: cache playback sıcaklığı/donmayı taşımaz (render için
önemsiz, cache'ten *devam* etmek donmuş katmanı sıfırlar). Donmuş parçacık
hareketli collider'la birlikte gitmez (v=0 sabitlenir).

## Granüler grid hızı cihazda kalıyor (2026-09-25)

> ✔ 1–3 doğrulandı (canlı ölçüm): upload 1330→200 MB, download 582→25 MB,
> P2G 131→1.6 ms, G2P 163→25 ms, alt adım başına 6.25 MB; kullanıcı tek
> çekirdek beklemesini artık görmüyor. **4 ve 5 açık.**

**Şikâyet:** fluid domain + 2 collider, sim play ile cache'lenirken bazı
karelerde tek çekirdek meşgul, uzun bekleme.
**Ölçülen (derlemeden önce, canlı uygulama):** 114³ granüler domain,
**1000 parçacık**, bir karede **1.33 GB upload + 582 MB download**.
Her elastik alt adımda (32'ye kadar; collider yığını zorladıkça artıyor =
"bazı kareler") P2G alanı indiriliyor, solid-yüz kıskacı host'ta koşuyor,
G2P için geri yükleniyordu; kare boyunca sabit olan solid hızı da her alt
adımda 3 alan olarak yeniden yükleniyordu. Hepsi grid boyutuyla ölçekli,
tek iş parçacıklı memcpy.

Değişen: `ParticleSimulation.cpp`, `SimulationComputeVulkan.cpp` (kayıt
satırı), **yeni shader** `shaders/sim_fluid_zero_solid_faces.comp`
(4 buffer, 36 bayt push constant). Yeni `.cpp` yok.
Yeni prob: `scripts/test/probe_granular_transfer.py`.

1. **Shader derlendi mi** — `compile_shaders.bat` sonrası
   `sim_fluid_zero_solid_faces.spv` exe'nin shader klasöründe olmalı.
   *Bozuksa:* Scene Log'da bir kez `sim_fluid_zero_solid_faces unavailable`
   uyarısı çıkar ve eski yol koşar (2. maddedeki sayılar değişmez). Çökme
   OLMAMALI.

2. **Transfer baytları** — aynı sahneyi aç, timeline'ı oynat,
   `python scripts/test/probe_granular_transfer.py "Grid Domain 1" 20`.
   Görmen gereken (114³, 32 alt adım): **download ≈ 18 MB/kare** (3 yüz
   alanı, kare sonunda bir kez) + parçacık boyutlu; **upload ≈ 32 × 5.9 MB
   (fluid mask) + 18 MB (solid hız, bir kez) ≈ 200 MB/kare**. Önce 1330 / 582.
   `up/substep MB` sütunu ~6 civarında olmalı, ~41 değil.
   *Bozuksa:* hâlâ ~1.3 GB → yerleşik yol devreye girmedi. Kapı koşulları:
   viskozite > 0, katı-faz madde etiketi (solid substance), Periodic sınır,
   CUDA backend, GPU kuvvet entegrasyonu kapalı. Bunlardan biri doğruysa eski
   yol BEKLENEN davranıştır.

3. **Kare süresi** — collider temasında (alt adım sayısı 32'ye çıkarken)
   tek çekirdek beklemesi belirgin kısalmalı. `perf.get loop.frame`
   `last_ms` / `max_ms`; önce temas karelerinde yüzlerce ms.
   *Bozuksa:* 2 geçip 3 geçmiyorsa darboğaz transfer değilmiş — sıradaki
   aday alt adım başına `buildFluidMaskFromParticles` (tam grid fill) ve
   parçacık indirmeleri.

4. **★ SİNSİ OLAN: davranış aynı mı** — kum yığını collider'ın üstünde
   durmalı, içinden geçmemeli; domain duvarları tutmalı. Cihaz kıskacı
   uygulanmazsa sonuç **hatasız ve makul** görünür: kum biraz "daha akışkan",
   collider'a biraz gömülüyor. Kimse bunu bug diye raporlamaz.
   A/B: `sim_fluid_zero_solid_faces.spv`'yi geçici olarak yeniden adlandır
   (eski yol zorlanır), aynı kareye kadar oynat, `fluid.get`'ten
   `granular_sleeping`, `granular_yielded`, `granular_mean_accumulated_plastic`
   al; .spv'yi geri koy, tekrarla. Aynı mertebede olmalı. Bit-bit beklenmez:
   iki kıskaç aynı yüzleri sıfırlıyor ve float gidiş-dönüşü kayıpsız, ama
   P2G'nin float atomic'leri AYNI yolun iki koşusunda bile toplama sırasını
   değiştirir. Referans için önce eski yolu iki kez koş: aradaki fark gürültü
   tabanıdır.
   *Bozuksa:* kıskaç cihazda maskeyi yanlış okuyor (maske yüklemesi ile
   dispatch sırası) ya da indeks düzeni `FluidGrid::vel*Index` ile uyuşmuyor.

5. **Sıvı domain değişmedi** — collider'lı bir SIVI (granüler değil) domain
   eskisi gibi davranmalı; bu partide onun yolu değişmedi (`runGpuFluidG2P`
   içindeki "solid yok" kapısı çağırana taşındı, çağıran zaten veriyordu).
   *Bozuksa:* sıvı collider'dan sızıyorsa G2P artık host kıskacını görmeyen
   cihaz alanını örnekliyor → çağıran taraftaki `hasAnySolid()` kapısına bak.

## Particle Faz 0 — ölçüm katmanı + collider IPC (2026-09-24)

> 13:09 derlemesinde ölçüldü: 1 ✔, 5 ✔, 6 ✗ (`backend_not_vulkan` — kökü ve
> düzeltmesi en üstteki parçacık bölümünde; 6–8 bir sonraki derlemede tekrar).
> 2 (smoke), 3, 4 kullanıcıda.

Değişen: `ParticleSimulation.h/.cpp`, `RtApi.h`, `RtApiParticle.cpp`,
`RtIpc.cpp`, `RtIpcSecurity.cpp`, `RtPython.cpp`, `ProjectManager.cpp`,
`scene_ui_forcefield.hpp`, üretilmiş `RtIpcMethodDescriptors.cpp`.
Yeni `.cpp` yok, shader değişmedi. Audit derlemeden önce geçti.

Sıra: bağımsız ve hızlı olanlar önce; 6–8 ancak 1–5 temizse anlamlı.

1. **Yeni alanlar var mı** — `python scripts/ipc_test_client.py`.
   `particle.stats(phase0 fields)` satırında "missing" FAIL olmamalı,
   `collider.*` altı test OK olmalı.
   *Bozuksa:* "missing" → eski exe; `collider.create` "unknown method" → eski
   exe ya da dispatch; "not authorized" → `collider.` namespace'i yetkiye
   ulaşmadı.

2. **Smoke test** — `python scripts/rt_api_smoke_test.py` sonuna kadar geçmeli.
   Yeni assert'ler: `stage_backends` beş anahtar; `forces_on_gpu` ile
   `gpu_force_status == "gpu"` birebir; CPU politikasında
   `gpu_force_status == "cpu_policy"` ve `force_download_bytes == 0`.
   *Bozuksa:* CPU politikasında indirme byte'ı > 0 → politika kapısı GPU
   bloğunu atlamıyor.

3. **Panel ↔ çekirdek** — Simulation > Particles > Physics sekmesinde
   **Execution** combo'su ve altında gri "Last step: forces CPU/GPU (durum),
   backend adı" satırı. Combo'yu CPU yap → timeline'ı birkaç kare oynat →
   satır `cpu_policy` demeli; IPC'den `particle.get_physics` aynı değeri
   (`cpu`) dönmeli. Tersini de dene: `particle.set_physics
   {"execution_policy":"auto"}` → combo Auto'ya dönmeli.
   *Bozuksa:* panel ile IPC farklıysa panel başka bir runtime'ı okuyor.

4. **Kaydet/aç** — politikayı CPU yap, projeyi kaydet, kapat, aç →
   combo hâlâ CPU. *Bozuksa:* `ProjectManager` `physics.execution_policy`
   yazmıyor/okumuyor; alan açılışta sessizce Auto'ya düşer.

5. **GPU Required gerçekten reddediyor mu** — sim compute backend'i Vulkan
   değilken (ya da Vulkan'da iken `particle.stats.compute_backend` neyse)
   `set_physics {"execution_policy":"gpu_required"}`, bir `particle.spawn`,
   `particle.step`, `particle.stats`:
   Vulkan değilse `step_blocked: true`, `gpu_force_status:
   "gpu_required_blocked"`, `stage_backends.forces: "blocked"` ve parçacık
   **hareket etmemiş** olmalı (`get_state_sample` pozisyonu değişmez).
   *Bozuksa:* parçacık düştüyse GPURequired sessizce CPU'ya düşüyor —
   yol haritasının açıkça yasakladığı şey.

6. **★ Auto satırı gerçekten GPU mu** — boş sahnede, timeline durmuşken:
   `.\scripts\ipc\Probe-ParticleBaseline.ps1 -Counts 2048 -Scenarios ballistic -Samples 10`
   Auto satırındaki `force[...]` **`gpu`** olmalı.
   ★ **En sinsi sonuç:** `backend_not_vulkan` / `no_dispatch_support` /
   `buffers_not_ready`. Script hata vermez, Auto satırı CPU satırıyla aynı
   süreyi gösterir ve "GPU hızlandırması yok" diye okunur — oysa GPU yolu
   **hiç koşmamıştır**. Bu durumda karşılaştırma tablosu anlamsızdır; önce
   nedeni (status değeri) bana yaz.

7. **Transfer maliyeti görünür mü** — aynı koşuda Auto satırında
   `down` (force_download_bytes/adım) ≈ `3 × 4 × capacity` byte olmalı
   (üç hız bileşeni, **kapasite** boyutunda — alive değil), `force_sync_calls`
   ≈ 1. `mirror` her iki politikada ≈ `(9×4 + 1) × capacity` byte.
   *Bozuksa:* 0 → sonda (probe scope) transferleri görmüyor; beklenenden çok
   büyükse uploadToCompute adım başına iki kez tam yükleniyor (emit
   `data_version_`'ı artırdığında beklenen, ama sayıyı not et).

8. **Tam baseline** — `.\scripts\ipc\Probe-ParticleBaseline.ps1 -OutputPath
   .\particle_baseline.json`. 3 senaryo × 3 sayım × 2 politika. Sonda
   "CPU vs other policy" tablosu çıkar.
   Beklenen: `ballistic` için `max_position_delta` çok küçük (~1e-4 m ve
   altı). `plane` ve `self_collision`'da fark büyüyebilir — çarpışma küçük
   float farklarını büyütür, bu tek başına hata değildir; ama `ballistic`
   büyükse GPU kuvvet çekirdeği ile CPU yolu **farklı fizik** hesaplıyor.
   `nonfinite_max` her satırda 0 olmalı.
   JSON'u bana gönder: Faz 5'in kabul toleransları ve hız kapıları bu
   sayılardan türetilecek.

### Bu partide bulunan ve ölçülmesi gereken şeyler

- GPU kuvvet bloğu (upload + dispatch + sync + hız indirme)
  `integrate_start`'tan **önce** çalışıyordu; hiçbir aşama zamanlayıcısı
  onu görmüyordu. Artık `gpu_force_ms`.
- İndirme doğrudan `buffers_`'a yapılıyordu: y bileşeni başarısız olursa x
  zaten GPU'da entegre edilmiş kalıyor ve CPU yolu kuvveti x'e **ikinci kez**
  uyguluyordu. Artık üçü de gelirse takas ediliyor, yoksa host dokunulmamış.
- Transferler **kapasite** boyutunda; ölü slotlar da taşınıyor.
- `spawn()` her parçacıkta `findDeadSlot()` ile baştan tarıyor (O(kapasite)).
  Büyük burst'lerde `emit_ms`'e bak.
- Çarpıştırıcılar yalnızca Python'daydı; `collider.*` artık IPC'de.
