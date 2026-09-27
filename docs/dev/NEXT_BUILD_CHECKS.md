# Sıradaki derlemede kontrol edilecekler

> **Durum:** CANLI — her partide üzerine yazılır. Önceki sürüm (particle
> authoring + SSS) git geçmişinde: `git show 220bed8:docs/dev/NEXT_BUILD_CHECKS.md`.
>
> ★ En üstteki dört bölüm (sıvı Fog modu; splat havuzu raster'da + Virtual
> Particles söküldü, 2026-09-27) YENİ. Altındakiler önceki derlemelere girdi; doğrulanan
> maddeleri ✔ işaretli, kalanlar açık.

## Fog Spread + sisin blackbody'si parçacık sıcaklığından (2026-09-27, 5. parti)

> Ham splat parçacık başına yalnız 8 hücreye değiyor; seyrek sprey tek tek
> lekelere dönüşüyordu. Eski "geniş bulut" görüntüsü gerçek yoğunluk değil,
> SDF'nin yüzey yardımcısıydı (siyah küp de ondan). Yeni alan
> `fluid_fog_spread_voxels` (varsayılan 1.5, 0..6): sis rotası grid yerine
> Gauss ile yayılmış KOPYAYI yükler (`Fluid::spreadFogDensity`, yeni
> `FluidFogDensity.cpp` — vcxproj'a eklendi). Çözücü grid'i ve
> `active_density_cells` DEĞİŞMEZ. IPC/Python: `fluid.set_fog spread_voxels`,
> okuma `fluid.get fog_spread_voxels`.

1. **Derleme.** Yeni `.cpp` projede; link hatası varsa vcxproj girdisi.
2. **Script.** `python scripts	estt_test_fluid_fog_mode_ipc.py "Grid Domain 1"`
   — 4. adım: 2.5 geri okunur, 7 reddedilir.
3. **Görsel, fog modunda.** Panelde "Volume Material (Fog)" altında
   "Fog Spread (voxels)". 0 → bugünkü lekeler; 1.5 → sürekli bulut; 3+ →
   yumuşak, geniş. Kaydırıcı DURAKLATILMIŞ karede de anında etki etmeli.
   *Etki etmiyorsa:* köprü yeniden yüklemiyor (resync isteği yutuluyor).
4. **Zemin kenarı.** Havuz domain duvarına kadar sönükleşmeden gitmeli (çekirdek
   duvarda yeniden normalize ediliyor). *Kenarda koyu şerit:* normalize çalışmıyor.
5. **Maliyet.** Büyük domain'de (≥200³) spread 6 ile kare süresine bak;
   ayrılabilir 3 geçiş + OpenMP. *Belirgin takılma:* yayma her karede değil
   yalnız yükleme karesinde çalışmalı — sık çalışıyorsa bana söyle.
6. **★ SİNSİ: yoğunluk tepe değeri düşer.** Yayma kütleyi korur ama tepeyi
   indirir; ince sprey daha soluk görünür. Bu hata değil — Density çarpanıyla
   dengelenir. Aynı spread'de "önceki kadar yoğun değil" normal.
7. **★★ SİNSİ: bugün parlayan sıvı sönebilir.** Blackbody/ChannelDriven emisyonlu
   sis artık PARÇACIK sıcaklığını (Kelvin) okuyor. Önceden sıvıda sıcaklık kanalı
   yoktu ve shader `temperature <= 0` görünce YOĞUNLUĞU sıcaklık sayıyordu —
   "sıcak akışkan" görüntüsü o yedekten geliyordu. Oda sıcaklığındaki sıvı
   (≈293 K) şimdi neredeyse hiç parlamaz (radyans ~ (T/Tmax)⁴). Bu DOĞRU
   davranış; hata sanma. Sıcak sıvı için Flow Source'ta sıcaklık override'ı ver.
8. **Sıcaklık verisi.** `fluid.get`: `particle_kelvin_measured: true`,
   `particle_min_kelvin` / `particle_max_kelvin`. Script'in 4. adımı bunu
   kontrol eder. *false:* parçacıklar 0 K (yazılmamış) doğuyor — emitter yolu.
9. **Görsel: lav.** Flow Source sıcaklık override 1400 K, sis modu, Blackbody:
   kaynaktan çıkan sıvı turuncu-sarı parlar. Termal zinciri aç: soğudukça
   koyu kırmızıya döner ve söner. *Tüm bulut tek renk:* sıcaklık kanalı
   yüklenmiyor (köprüde `splatFogTemperatureKelvin` false dönüyor).
10. **Kenar rengi.** Bulutun ince kenarı çekirdekle aynı sıcaklık rengini
    taşımalı (kütle ağırlıklı yayma). *Kenarlar kırmızı, çekirdek beyaz ve
    hepsi aynı sıcaklıktaysa:* sıcaklık yoğunlukla birlikte bulanıklaşıyor.

## ★ Fog düzeltmesi: sıvının yoğunluk sayacı her adımda sıfırlanıyordu (2026-09-27, 4. parti)

> İlk fog derlemesinde sis hiçbir modda çizilmedi; VDB panelindeki Fog seçimi de
> artık çizmiyordu (o seçim şimdi aynı domain moduna yazıyor). CANLI ÖLÇÜLDÜ:
> 104k parçacık, splat her adımda çalışıyor (`[FluidGPU DensitySplat]` logu),
> ama `active_density_cells = 0`. Kök: `stepGridDomains` sonundaki analiz
> geçişi sayacı HER domain için sıfırlayıp yoğunluğu yalnız gazda yeniden
> sayıyordu ("fluid domains don't touch grid.density"). Fog kapısı
> `active_density_cells > 0` istediği için kendini kapattı. Sınırlar
> (`active_density_min/max`) da sıvıda hiç dolmuyordu. Düzeltme: tarama sıvıya
> da açıldı (`ParticleSimulation.cpp`, analiz geçişi). Shader değişikliği YOK.

1. **Sayaç.** Fog modunda bir kare ilerlet: `fluid.get` →
   `active_density_cells > 0`, `max_density` ≈ 1 civarı (hücre başına
   parçacık/ppc). *Hâlâ 0 ise:* exe eski, ya da sim o karede adım atmadı
   (önbellekten geldi) — Play ile birkaç adım at.
2. **Log.** SceneLog'da `[VolumeGate 0] ... route=VolumeFog` satırı artık
   `RENDERABLE` ve `active_cells>0` demeli. *NOT renderable + active_cells>0:*
   kapıda başka bir koşul var, bana log'u getir.
3. Sonra aşağıdaki bölümün 1–7. maddeleri (hepsi bu sayaca bağlıydı).

## Sıvı için Volumetric Fog modu + VDB paneli domain'e yazıyor (2026-09-27, 3. parti)

> Kök neden: VDB panelindeki "Fog ↔ Refractive" combo'su domain'in her karede
> yeniden ürettiği hacme yazıyordu; köprü onu domain'den geri yazıyordu
> (`render_as_isosurface = fluid_surface_route`). Sıvı yoğunluk üreticisi zaten
> her adımda çalışıyor (`sim_fluid_density_splat`), yani Ağustos'taki "sıvı
> yoğunluk splat'lemez" gerekçesi geçersizdi. Yeni kayıt değeri `VolumeFog = 4`;
> kayıtlı `Volume = 0` anlamını KORUYOR (sıvıda → SDF), eski sahneler değişmez.
> Otomatik test: `python scripts\test\rt_test_fluid_fog_mode_ipc.py "Grid Domain 1"`.

1. **Script.** `fluid.set_param render_mode=fog` → `fluid.get` `"fog"`;
   `active_density_cells > 0`. *0 ise:* üretici sorunu, render değil.
2. **★ Asıl hata: kare değişince düşme.** Script'in 3. adımı: üç kare boyunca
   mod `fog` kalır. Elle: VDB panelinde domain hacmini seç — "Owned by liquid
   domain …" satırı görünmeli; Fog seç, timeline'ı sür: sis kalmalı.
   *Düşüyorsa:* panel hâlâ hacme yazıyor (`fluidDomainOwningVolume` null
   dönüyor) ya da başka bir yol domain modunu geri yazıyor.
3. **Panel.** Liquid Display'de üçüncü seçenek "Volumetric Fog / Gas". Seçince
   "Volume Material (Fog)" bölümü açılır; slider'lar sisi değiştirir.
   "Now drawing: Fog volume: untagged".
4. **Görsel: Material ve Rendered.** Sıvı sis olarak görünür, domain shader'ı
   (yoğunluk/saçılma/emilim) ile. İlk geçişte `Liquid NanoVDB Preview` preset'i
   uygulanır (yoğunluk ×50, mavi emilim).
5. **Eski sahne.** Sıvı için `Volume` (0) kaydedilmiş eski bir .rtp: SDF açılmalı,
   sis DEĞİL. *Sis açılıyorsa:* dekoder 0'ı yanlış çözüyor.
6. **★ SİNSİ: SDF override + fog.** Bir maddeye SurfaceSDF override ver, domain
   fog'da kalsın: sis ÇİZİLMEZ (tek domain hacmi yüzeye gider). Panel bunu
   turuncu uyarıyla söylemeli; `fluid.get` `effective_representation` "sdf".
   Uyarı yoksa kullanıcı "sis bozuk" sanır.
7. **VDB IOR/roughness/foam.** Domain hacminde bu üçü artık domain alanlarını
   yazar; kare değişince korunur.

## Splat havuzu raster'da, foliage LOD'undan muaf; Virtual Particles söküldü (2026-09-27)

> Kök nedenler canlı sahnede ölçüldü (88k su, granüler kapalı), düzeltmeler
> derlenmedi. Özet: `docs/dev/GRANULAR_HEIGHTFIELD_GORUNUM.md` "Neden söküldü".
> Otomatik test: `python scripts	estt_test_fluid_splat_raster_ipc.py "Grid Domain 1"`
> (domain'de parçacık olmalı; mod/subdiv/shading'i geri yükler).

Sıra: bağımsız ve hızlı olanlar önce; 5–7 birbirini maskeler, 1–4 geçmeden bakma.

1. **Derleme.** Beş `GranularVirtual*` dosyası silindi ve vcxproj/filters'tan
   çıkarıldı. *Bozuksa:* "cannot open GranularVirtual…" = eski bir include
   kalmış; bana satırı söyle.
2. **Eski proje açılışı.** `virtual_particles` ile kaydedilmiş bir projeyi
   aç (şu anki sahne öyle). `fluid.get` → `render_mode: "particles"`.
   *Bozuksa:* `"volume"`/`"surface"` = değer 3, `fluidRenderModeFromStored`
   üzerinden okunmuyor (ProjectManager / SceneSerializer).
3. **Script reddi.** `fluid.set_param render_mode=virtual_particles` hata
   döner, mod değişmez. `fluid.set_splat_geometry subdivisions=4` reddedilir.
   *Bozuksa:* kabul ediyorsa eski exe (zaman damgasına bak).
4. **Panel.** Liquid Display combo'sunda iki seçenek var. Splat Spheres →
   "Splat Geometry & Preview" açılır; Smooth Surface → "Surface SDF Settings".
   Yeni oluşturulan domain'de Sphere Subdivision Detail = 0.
   *Bozuksa:* Splat seçiliyken SDF ayarları açılıyorsa `current_mode_idx`
   eşlemesi kaymış (eskiden Virtual seçilince tam olarak bu oluyordu).
5. **Solid: impostor gerçekten çiziyor mu.** Test script'inin 3. adımı.
   `viewport.frame_telemetry` → `raster_sphere_groups ≥ 1`,
   `sphere_impostors_uploaded > 0`, `sphere_impostors_drawn == uploaded`.
   *Bozuksa:* uploaded>0 ama drawn=0 → çizim kapısı kapalı (pipeline/buffer/
   matcapDescSet/mod), `recordParticleBillboards`. groups≥1 ama uploaded=0
   → havuzda görünür parçacık yok ya da builder yüklemiyor.
   ✔ **Sayaçlar kökü buldu (2026-09-27, 2. derleme):** uploaded=56666,
   drawn=0, ready=true, Solid ve Matcap'te aynı. Kapalı kapı `matcapDescSet`:
   teardown onu `keepPipeline`'dan bağımsız yok ediyordu, yani **ilk panel/
   pencere yeniden boyutlandırmasından sonra** set kalıcı NULL; yalnız
   pipeline sıfırdan kurulurken yeniden yaratılıyordu. Solid geometri set'i
   `!= NULL` korumasıyla bağlayıp çizmeye devam ettiği için hiçbir şey bozuk
   görünmüyordu. Düzeltme her iki backend'de (viewport + base) teardown'ı
   `!keepPipeline` arkasına aldı. **Doğrulama:** Solid'de splat'ler görünür;
   sonra bir paneli sürükleyip viewport'u YENİDEN BOYUTLANDIR — hâlâ
   görünür olmalı ve `sphere_impostors_drawn == uploaded`. Yan etki olarak
   kullanıcının yüklediği matcap dokusu da resize sonrası artık kaybolmamalı.
6. **Material / RayFusion: havuz raster listesinde.** Script'in 4. adımı:
   `full_instances > 1`, `proxy_instances == 0`. Sonra sim'i oynat.
   ★ `total_instances`'a BAKMA: boş havuz slotları artık maskeli olarak listede
   duruyor, yani o sayı slot sayar, parçacık değil — boş havuzda da büyük çıkar.
   *Görmen gereken:* küreler hep görünür, kart/düz şerit yok, oynatırken
   titreme yok. *Bozuksa:* full_instances=1 → havuz çizilmiyor;
   log'da `buildRasterGeometry early-out` + `stampedBy` satırına bak.
7. **★ SİNSİ: yeni doğan parçacıklar.** Boş domain'den emitter ile başlat,
   Material modunda oynat. Akan su **kesintisiz** görünmeli. Eski kusur
   tam olarak "seyrek, delikli akış" gibi görünüyordu — kimse bunu bug diye
   raporlamaz, emisyon ayarı sanılır. Havuz kademesi büyümeden (aynı kapasitede)
   dolan slotlar artık `mask` ile açılıyor; delik görürsen bu madde bozuk.
8. **Foliage regresyonu.** Yoğun foliage sahnesi, Auto preset:
   `proxy_instances > 0` hâlâ olmalı. *Bozuksa:* muafiyet foliage'a da
   sızmış (`group.transient` yanlış true) — kare maliyeti patlar.
9. **Boş havuz maliyeti.** Cache temizle (0 parçacık, havuz dolu kalır),
   Material modunda `frame_ms` ve `visible_triangles`. Havuz slotları artık
   raster'da duruyor ama `mask=0` → GPU cull atlar. Eski ölçüm: 1M ölü slot
   = 101,9 ms. *Görmen gereken:* visible_triangles ≈ sahnenin kendisi,
   frame_ms normal. *Bozuksa:* ölü slotlar çiziliyor → cull bounds `w<0`
   yazılmıyor (`writeRasterInstanceBound`).
10. **Vulkan RT (Rendered) değişmedi.** Aynı sahnede RT'ye geç: splat'ler
    görünür, hız önceki gibi. subdiv 0'ın (20 üçgen) görünümü yakın planda
    kabul edilebilir mi — değilse Splat Geometry'den 1'e çek; bu yalnız
    default değişikliği.
11. **Flat kaynaklı scatter senkronu.** Flat SoA kaynaklı bir foliage grubu
    olan sahnede bir instance'ı gizmo ile taşı: yerinde kalmalı. Senkron
    artık `scatterSourceTransform`'u uyguluyor; eskiden ilk senkronda
    merkezleme ofseti kadar kayıyordu. *Bozuksa:* obje taşıma anında zıplar.
12. **1M boş slot maliyeti (ÖLÇÜLDÜ, düzeltme derlenmedi).** Canlı ~520k sabitken
    havuz 524k → 1M olunca Material karesi (sim oynarken) **49 → 91 ms** çıktı.
    Boş slotlar ücretsiz değilmiş. Düzeltme: transform senkronu boş kalan
    slotu matris kurmadan atlıyor. *Görmen gereken:* aynı deneyde fark
    ≲5 ms. *Hâlâ ~40 ms ise:* maliyet CPU senkronu değil GPU/yükleme tarafı
    (`uploadRasterInstanceBuffer` 1M matris yazıyor) — o zaman dirty aralık
    yüklemesi gerekir. Deney: domain `surface`→`particles` (havuzu sıfırlar),
    ~507k tohum, Material, timeline'ı kare kare ilerlet, `frame_ms` izle.
13. **TDR (591k splat, Material→RT) TEKRARLANAMADI.** IPC ile denenenler, hepsi
    temiz: RT 65k/135k/270k/540k; Material→RT 540k; RT'de ve Material'da
    kare kare oynatma; Material'da havuz büyümesi (524k→1M) + hemen RT.
    Sürücü kaydı: `nvlddmkm` olay 153 (TDR zaman aşımı). Log'da kayıp,
    RT "yield" satırından ÖNCE gözlendi. Denenemeyen tek koşul **sürekli
    Play** (IPC'de play yok; set_frame kare kare adım atıyor). *Tekrar
    ederse:* SceneLog'u koru ve hangi modda Play'e bastığını not et.

## Vulkan RT: çok splat varken oynatmada yavaşlama — TLAS refit + orijindeki ölü slotlar (2026-09-26)

> ✔ **Kullanıcı doğruladı (derlendi):** RT oynatma yavaşlaması kayboldu. Havuz
> kademesi büyürken tam kurulum kısa bir takılma yapıyor (1M'de kabul edilebilir).
> ⚠ Aynı derlemede 1M splat ile Solid→Rendered'da **yeni bir TDR** — sessiz
> render cihazı ölümü, bkz. `GRANULAR_HEIGHTFIELD_GORUNUM.md` "AÇIK" bölümü.

> Kullanıcı gözlemi: RT'de play sırasında (havuz büyüyüp instance sayısı
> değiştikten sonra) render çok yavaşlıyor, Solid→Rendered yapınca hemen
> hızlanıyor. İki kök, ikisi de derlenmedi:
>
> 1. **TLAS hiç yeniden kurulmuyordu, yalnız refit ediliyordu** (`createTLAS`
>    + GPU `recordGpuTLASUpdate`, sayı değişmedikçe `MODE_UPDATE`). Refit ilk
>    kurulumun hiyerarşisini tutar, kutuları büyütür. Havuz kademesi büyüyünce
>    tam kurulum olur, o anda slotların çoğu ölü; sonraki karelerde onlar domain'de
>    doğdukça ağaç bozulur. Solid→Rendered tek taze kurulumdu. Artık iki yol da
>    **aynı nesneye yerinde `MODE_BUILD`** yapıyor: handle/adres/descriptor
>    değişmez, destroy yok (TDR riski eklemez).
> 2. **Ölü slotlar `identity + mask 0` idi** = orijinde birim küre, hepsi üst
>    üste, domain de orijinde. Mask gölgelemeyi eler ama BVH özdeş kutuları
>    ayıramaz: orijinden geçen her ışın o yaprakları tek tek ziyaret eder. Artık
>    `parkedTLASInstanceTransform()` (1 mm, y=-10 km) — CPU iki yol + GPU
>    `instance_prepare.comp`. Kostik hedef kutusu mask 0'ı atlıyor (eskiden
>    ölü slotları orijinde sayıyordu).
>
> ★ Kare atlamalı IPC A/B (build@20 → refit@143) fark göstermedi (~31 ms/örnek
> her iki kolda) — çünkü atlama havuz kapasitesini değiştirip zaten tam
> kurulum yaptırıyor. Kullanıcının koşulu **canlı play**; o IPC'den sürülemiyor.
> Ayrıca: `viewport.status.ms_per_sample` her durumda 0 dönüyor (ölü alan).

1. **`instance_prepare.comp` → `.spv` derlendi mi.** Shader değişti. Eski spv
   ile GPU yolu ölü slotları hâlâ orijine koyar; CPU yolu park eder. Bozuksa:
   yavaşlama yalnızca GPU scatter güncellemesi olan karelerde sürer.
2. **Aynı senaryo: RT'de play, havuz birkaç kademe büyüsün.** Beklenen: hız
   oynatma boyunca sabit kalır, Solid→Rendered artık fark yaratmaz. Bozuksa
   (hâlâ Solid'e geçip dönünce hızlanıyor): üçüncü bir kaynak var — BLAS
   (splat icosphere değil, foam/sphere GAS) veya volume tarafı.
3. **Görsel: splat'lar, köpük, cam/su kostikleri doğru yerde.** Bozuksa
   (orijinde ya da -10 km yönünde bir artefakt): park dönüşümü mask 0 olmadan
   bir yere sızıyor.
4. **Maliyet: sahne hareketsizken (pause) örnek hızı eskisi kadar.** Her kare
   BUILD, refit'ten pahalı; ama hareketsiz karede TLAS hiç çağrılmamalı.
   ★ Sessiz başarısızlık: pause'da da kare süresi birkaç ms yüksekse biri
   TLAS'ı değişiklik olmadan her kare güncelliyor demektir — bu düzeltmeden
   önce de öyleydi, sadece refit ucuz olduğu için görünmüyordu.
5. **TDR yok.** RT'de play + pause + Solid↔Rendered birkaç tur. Yerinde
   BUILD da bir yazma; önceki refit ile aynı senkronizasyonu kullanıyor.

> OptiX IAS'ında aynı refit deseni duruyor (`OptixAccelManager.cpp` ~1850);
> OptiX dondurulduğu için dokunulmadı.
>
> Ölçüm: `render.fluid.splat_instances` / `render.fluid.foam_instances` perf
> bölümleri eklendi (splat köprüsünün CPU maliyeti, granüler gösterim kararı
> için).

## Fluid Particles modu: çift çizim (mavi diskler) söküldü (2026-09-26)

> Particles modundaki fluid domain'leri hem render bridge'in küre
> instance'larıyla hem de `ParticleBillboardBuilder::appendGridDomainParticles`
> ile (domain rengi, yani mavi, billboard) iki kez çiziliyordu. Billboard'lar
> canlı konumu okuyor, instance'lar bir adım geriden geliyordu. Oynatırken her
> kürenin önüne mavi bir disk çıkıyor, duraklatınca disk kürenin içinde
> kayboluyordu. IPC ekran görüntüsüyle yeniden üretildi: duraklatılmışta yok,
> adım sırasında var. Billboard kopyası söküldü.

1. Particles modunda bir su sütunu oynat. Mavi halka/disk **olmamalı**,
   küreler beyaz/materyal renginde kalmalı. Hâlâ varsa başka bir yol daha
   çiziyordur (overlay'in debug noktaları: `particle_display_mode`).
2. Emitter'lı bir particle sistemi (fluid'siz) billboard olarak görünmeye
   devam etmeli; o yol değişmedi.

## Sim compute: kernel tablosu binding sayısı ≠ shader (validation ile bulundu)

> `RAYTROPHI_VK_VALIDATION=1` açılışta `VUID-VkComputePipelineCreateInfo-layout-07988`
> bastı. Uyuşmayan iki kernel:
> - `sim_fluid_granular_stress_update`: tablo 14, shader ve dispatch 15
>   (`bond_scale`);
> - `sim_fluid_surface_combustion`: tablo 6, shader ve dispatch 9.
>
> Pipeline layout eksik binding'le kurulup üstüne daha büyük bir descriptor
> set bağlanıyordu. Bu tanımsız davranış; NVIDIA pratikte tolere ettiği için
> kum testi çalışıyordu. Tablo düzeltildi. `dispatch` artık sayı uyuşmazlığında
> reddediyor ve kernel adını `[SimCompute] ... dispatch refused` diye bir kez
> log'a yazıyor.
> Kalıcı denetim: `python scripts/audit_sim_kernel_bindings.py` (0 uyuşmazlık).

1. Validation açıkken başlat: `07988` artık **hiç** görünmemeli.
2. Kum testi (granüler) ve yanan sıvı (yüzey yanması) önceki gibi çalışmalı.
   Log'da `dispatch refused` satırı varsa başka bir kernel'in çağrı yeri
   tablodan ayrışmış demektir; satır kernel'in adını söyler.

## TDR: sim çalışırken RT'ye geçiş — KÖK BULUNDU, düzeltme derlenmedi (2026-09-26)

> Yeniden üretim: sim canlı moddayken, parçacık havuzu büyürken RT'ye geçiş.
> Validation açıkken yakalandı; device lost'tan önce **hiç validation hatası
> yok**. Yani sorun API kullanımı değil, zamanlama.
>
> Üç çöküşteki ortak dizi:
> `RayFusion ... yielded` → `SWITCHING to Vulkan RT` → `RayFusion scene AS built … 11 KB`
> → device lost. Viewport'un RayFusion build'leri hep 4 KB; 11 KB'lık olanlar
> başka bir cihazın, yani **render backend'in** build'i.
>
> Kök: `scene_ui_procamera.cpp` ve `scene_ui_selection.cpp`'deki "AS ısıtıcıları"
> `ensureRayFusionSceneAS`'i `ctx.backend_ptr` (render backend) üzerinde de
> çağırıyordu. RayFusion TLAS'ı cihazın TEK TLAS slotuna kurar
> (`m_device->createTLAS`). Render backend'de bu slot path tracer'ın TLAS'ıdır.
> - Parçacık havuzu büyüyünce instance imzası değişiyor ve yeniden kurulum
>   tetikleniyor.
> - Bu RT'ye geçiş karesine denk gelirse path tracer'ın TLAS'ı, uçuştaki
>   trace'ler onu kullanırken 65.536 instance'lık, farklı sıralı bir TLAS ile
>   değiştiriliyor.
> - `drainInteractiveViewportInFlight` yalnız viewport karelerini bekliyor,
>   trace slotlarını beklemiyor.
>
> Düzeltme:
> - Isıtıcılar yalnız `g_viewport_backend`'i ısıtıyor.
> - Build log'u artık `device=` yazıyor.
>
> Ek olarak yan bulgu: iki sim kernel'inin binding sayısı, ayrı bölümde.

1. **Tanıyı doğrula:** RT'ye her geçişte log'da `[RayFusion] scene AS built`
   satırı varsa `device=` değeri viewport cihazınınki olmalı. Render cihazında
   bir build görünüyorsa başka bir çağıran kalmış demektir.
2. **Tekrar dene:** canlı mod, çok parçacık, havuz büyürken RT'ye birkaç kez
   geç (validation açık kalabilir). Device lost **olmamalı**.
3. ★ Sinsi başarısızlık: TDR biter ama Solid/MaterialPreview'da gölge ya da
   RayFusion etkisi kaybolur. O zaman viewport ısıtması başka bir sebeple
   render backend'e bağımlıymış; bunu `render.probe` ile ölç.
4. Açık kalan yapısal risk: RayFusion ile path tracer'ın aynı cihazda aynı
   TLAS slotunu paylaşması. Tek backend'li kurulumda yalnız mod kapısı
   (`m_viewportMode != Rendered`) koruyor.

## Animasyon: timeline'a bağlı graph zamanı + döngüyü açan root motion (2026-09-26)

> Teşhis (yürüyen Mixamo karakteri, 34 karelik döngü):
> - Graph klipleri timeline'ı DEĞİL biriken delta'yı izliyordu
>   (`graph_follows_timeline=true` iken bile). Kare atlayınca poz donuyordu,
>   geri sarınca aynı kare farklı poz veriyordu.
> - Root motion, kök kemiğin ötelemesini TAMAMEN sıfırlıyordu; kalça
>   yüksekliği de sıfırlandığı için karakter yere gömülüyordu. Yatay hareket
>   de her karede nesnenin transform'una `position += delta` diye kalıcı
>   yazılıyordu, bu yüzden geri sarınca ve oynatma yolu değişince karakter
>   son kaldığı konumdan başlıyordu.
>
> Yeni model (`Animation/RootMotionUnroll`):
> - Klip zamanı = (kare − başlangıç) / fps; graph timeline'ı izliyorsa bu
>   mutlak zaman kullanılır.
> - Root motion = kök kemiğe `tamamlanan döngü × döngü başına yol` eklenir.
>   Yol = son konum anahtarı − ilk konum anahtarı; kemiğin ebeveyn uzayında
>   eklendiği için iskelet ölçeği hiyerarşide kendiliğinden uygulanır.
> - Kemik sıfırlanmaz, transform'a yazılmaz, nesnenin konum anahtarları
>   uygulanır.
>
> Sökülenler: `RootMotionDelta`, iki çıkarım yolu, iki transform itme bloğu,
> blend düğümlerindeki root-motion lerp'leri, UI'nin "root motion açıksa konum
> anahtarını atla" kuralı.
> Yeni IPC: `anim.set_root_motion {character, enabled, bone?}`. `anim.character`
> artık `root_motion_resolved_bone`, `root_motion_cycle_travel` ve
> `root_motion_travel_valid` döndürüyor.
> Yeni dosyalar: `RootMotionUnroll.h/.cpp` (vcxproj + filters'a eklendi).

1. **Derleme.** `AnimationController::getAnimatedGlobalTransform` imzası
   değişti (döngü sayaçları). `RootMotionDelta`'yı kullanan başka bir yer
   kalmışsa derleyici söyler; grep temiz çıktı.
2. **Root motion kapalıyken** (varsayılan) karakter eskisi gibi yerinde
   döngü atmalı, döngü sonunda geri sıçramalı. Tek fark: graph timeline'ı
   izliyorsa artık kare atlamaları doğru poza gidiyor. Kapalıyken bir şey
   değiştiyse, zaman kaynağı sırası bozulmuştur.
3. **Probe:** test sahnesini aç, sonra
   `python scripts/test/rt_probe_root_motion_timeline_ipc.py 1 "1 Kinematic Preview" 110`.
   Root motion'ı AÇIK bırakır. Görmen gereken:
   - `bone:` boş olmayan bir kemik, `cycle travel` sıfır olmayan bir vektör ve
     `valid: True`;
   - `scrub` satırında her kare tek bir konum;
   - `walk` satırında ≥ 3 m yatay yol, en büyük kare adımı ~0,05–0,15 m
     (eski döngü sıçraması 1,6 m idi);
   - `rewind offset ≈ 0`, sonunda PASS.
   `valid: False` ise otomatik seçilen kemiğin (Armature/RootNode) konum
   anahtarı yok: `anim.set_root_motion character=1 enabled=true
   bone=<Hips'in tam adı>` ile sabitle.
4. **Görsel:** oynat. Karakter domain boyunca kesintisiz yürümeli ve y'de
   gömülmemeli. Durdurup 0'a sarınca başa dönmeli; sahnenin kayıtlı
   transform'u değişmemeli. ★ Sinsi başarısızlık: karakter yürüyor ama
   bir noktada viewport'tan kayboluyor. Nesnenin sınır kutusu hâlâ başlangıç
   yerinde kaldığı için culling ediliyor olabilir. Bunu yalnız göz görür.
5. **Kum/su ile:** `rt_probe_kinematic_foot_stamps_ipc.py "1 Kinematic Preview" 110`.
   `max` solver speed artık 36–40 m/s sıçrama göstermemeli (döngü ışınlanması
   bitti); ayak başına sıfır hücreli kare olmamalı.
6. **Kalan bilinen eksik:** state machine ve blend geçişleri hâlâ delta ile
   ilerliyor; timeline'a bağlı değiller. Klip değişince (`play`) döngü sayacı
   sıfırlanır ve karakter o klibin başlangıç konumuna döner.

## Kinematic collider: ayak kutusu mesh'ten + çözücü damga telemetrisi (2026-09-26)

> Teşhis (yürüyen karakter, ~1 voxel su, 5,9 cm voxel): ayak kutusu **bileğe**
> ortalanmış sabit 3,75×2,5×6,25 cm idi; basan ayakta alt yüzü y≈0,07'de,
> suyun ancak üstüne değiyordu. Ayaklar karede 1–2 hücre damgalıyordu.
> `autoFit` içindeki "bilek→parmak kutusu" dalı **ölü koddu**: `isBodyAnchor`
> foot/toe/head'i daha önce yakalıyor. Değişiklikler:
> - `collectKinematicJointPoses` her joint için baskın olduğu (ağırlık ≥ 0,5)
>   rest-pose skin köşelerinin kemik-yerel sınırını ölçer; `fitLeaf` foot/toe
>   kutusunu bu sınırdan kurar (topuk+taban+parmak). Köşesi olmayan joint eski
>   sabit kutuya düşer. Ölü foot ve head dalları söküldü.
> - Yeni `physics.collider.proxy_set.solver_stamps` (+ Python
>   `proxy_set.solver_stamps()`): son grid adımında proxy×domain başına
>   `stamped_cells` ve çözücünün `solid_vel`'e yazdığı hız. Sayım zaten dönen
>   damga döngüsünde bir artırım; ek geçiş yok.
> - `proxy_set.sample` notuna "hız OKUMA anındaki geçmişe göre, çözücününki
>   değil" uyarısı eklendi.

1. **Derleme.** Yeni dosya yok; `KinematicColliderVoxelizer.h` artık
   `ParticleSimulation.h`'ten include ediliyor. `FluidGrid` class/struct ileri
   bildirimi C4099 uyarısı verirse zararsız (önceden de vardı).
2. **Telemetri tek başına** (refit ETMEDEN, önce eski kutuyla ölç):
   `python scripts/test/rt_probe_kinematic_foot_stamps_ipc.py "1 Kinematic Preview" 24`.
   Görmen gereken: her karede dört ayak satırı, `frames stepped 24/24`, ayak
   hücreleri ortalama ~1–2,5 (önceki elle ölçümle aynı). Bu, sayacın doğru
   saydığının kanıtı. `steps` artmıyorsa log hiç dolmuyor — çağrı noktası
   bağlanmamış.
3. **Refit** — aynı komut `--refit` ile (set'in proxy'lerini YENİDEN ÜRETİR).
   Görmen gereken: foot/toe kutularının `local_position`'ı sıfır değil (Mixamo'da
   kemik ekseninde ileri ve aşağı), `half_extents` ayağın gerçek boyu kadar
   (ayak ~10+ cm uzunluk). Ardından basan ayakta **hücre sayısı belirgin artar**
   ve `zero-frames` düşer. Hâlâ ~1–2 ise: kutu yanlış uzayda (placement /
   globalInverse zinciri) — `local_position`'ı viewport overlay'de gözle kontrol et.
4. **Viewport overlay** (kinematic preview açık): ayak kutuları topuktan
   parmak ucuna uzanmalı ve tabana oturmalı. ★ Sinsi başarısızlık: kutu doğru
   BOYUTTA ama bilekte ya da ayağın üstünde — telemetri hücre sayısını artmış
   gösterir ama su yine yalnız üstten ezilir. Bunu yalnız overlay gösterir.
5. **Görsel**: splat ve granüler testinde ezilme/dağılma. Su 1 voxel
   derinliğindeyse etki yine sınırlı kalır — domain'i yürüme alanına indirip
   voxel'i küçültmek (sahne ayarı) ayrı bir kol.

## Particle Faz 1.5 Batch B: cihazda kalan balistik adım + vertex pulling (2026-09-26)

> Parçacık ajanının partisi. Tasarım: `PARTICLE_SYSTEM_GPU_ROADMAP.md` Faz 1.5
> "Batch B design". Faz 0'ın "kısmi GPU" yolu (her adım tüm SoA upload +
> sync + hız indirme) **silindi**; sistem ya tamamen cihazda ya tamamen CPU
> referansında koşar. `particle.stats` alanları yeniden adlandırıldı:
> `gpu_force_status`→`gpu_status` (`gpu_resident`), `forces_on_gpu`→
> `device_resident`, `force_*`/`mirror_*`/`upload_ms` yok; yeni: `residency`,
> `step_*`, `snapshot_*` (kümülatif), `nonfinite_measured`.
> Yeni dosyalar: `ParticleDeviceResidency.cpp`, `sim_particle_ballistic.comp`,
> `sim_particle_spawn.comp`, `particle_viewport_pull.vert`,
> `include/particle_appearance_lut.glsl`. Silinen: `sim_particle_force_integrate.*`.

1. **Shader + log.** `compile_shaders.bat` (12:07'de zaten koşmuş, üç yeni
   `.spv` var). Açılışta konsolda `particle_viewport_pull.spv missing`
   **olmamalı**. Varsa: pull pipeline yok, cihazdaki her sistem viewport'ta
   görünmez (sim çalışır).
2. **IPC testi** `python scripts/ipc_test_client.py` — `phase1.5B:` satırları.
   Kritik olan: `CPU and Auto trajectories agree` (|dp| ≤ 1e-4 m, 16 adım).
   Büyükse ballistic kernel ile CPU yolunun işlem sırası ayrışmış.
   `Auto did not go resident (...)` basıyorsa sebep parantezde — makinede
   Vulkan compute yoksa beklenen, varsa hata.
3. **`rt_api_smoke_test.py`** (Python yüzeyi aynı alanlar).
4. **Panel**: Physics sekmesi → `Last step: GPU (gpu_resident), Vulkan ...,
   state on device`. Collider ekle → `CPU (host_consumer_colliders)`, kaldır →
   tekrar GPU. **Parçacıklar geçişte sıçramamalı** (cihaz durumu `cpu_path`
   snapshot'ıyla eve geliyor).
5. **Viewport görsel (vertex pulling)**: Campfire/kıvılcım, Auto. Billboard'lar
   görünür ve **hareket eder**; tek sistemde additive alev + alpha duman
   ikisi de çizilir (row lookup'taki blend biti).
   ★ **En sinsi hal: parçacıklar görünür ama doğdukları yerde DONUK.** Bu,
   builder'ın CPU quad yolundan bayat host pozisyonlarını çizdiği anlamına
   gelir (pull edilemedi ve snapshot da alınmadı). `particle.stats` →
   `snapshot_last_reason` = `foreign_device` ve her karede artıyorsa: sim ile
   viewport **farklı VkDevice** (`render.volume_tables` ile teyit) — o zaman
   donukluk bir hatadır; artmıyorsa `residentDrawBuffers` false dönüyor.
6. **Zaman çizelgesi oynat + geri sar**: oynarken `snapshot_mirrored_steps`
   her karede artmalı, `snapshot_sync_count` yalnızca ilk karede bir kez
   artmalı (sonrasında önbellek talebi adımın kendi fence'ine biniyor). Her
   karede artıyorsa talep→ayna mekanizması çalışmıyor: oynatma eski kısmi
   yol kadar pahalı. Geri sarınca doğru parçacıklar görünmeli
   (restore → `residency host` → tam upload).
7. **Politika geçişi oynarken**: Auto → CPU → Auto. Sıçrama yok. `step_blocked`
   yalnızca GPU Required + collider/self collision'da (sebep `gpu_status`'ta).
8. **Render in Raytrace açık** sistem: RT instance'ları parçacıkların GÜNCEL
   yerinde (`snapshot_last_reason` `raytrace_instances`). Debug display
   (nokta) modu: noktalar hareket eder (`debug_dots`).
9. **Kapı ölçümü** `.\scripts\ipc\Probe-ParticleBaseline.ps1 -Scenarios
   ballistic -Counts 1024,8192,32768` (timeline durmuş, sahnede başka
   render yükü yokken). Auto satırları `step[gpu_resident]`, `down 0 B`.
   **Faz 5 kapısı: 32k resident < 1,18 ms (Faz 0 CPU).** `down` > 0 ise
   betik uyarır: bir tüketici her kare host durumu çekiyor (madde 5/8).
   plane/self_collision satırları bilerek `host_consumer_*` (Faz 6).

## Particle Faz 1.5 Batch A: görünüm profili + GPU LUT, eski start/end yolu söküldü (2026-09-26)

> Parçacık ajanının partisi; kinematic collider bölümü (dosyanın sonunda)
> diğer ajanın, ikisi aynı derlemeye girer ve birbirinden bağımsızdır.
> Plan ve sözleşme değişiklikleri: `PARTICLE_SYSTEM_GPU_ROADMAP.md` → Phase 1.5.

**Ne değişti:** parçacık başına renk/boyut/opaklık artık SoA'da tutulmuyor ve
CPU her adımda lerp yapmıyor. Emitter bir `appearance_profile_id` taşıyor.
Profil 64 örneklik bir LUT'a pişiriliyor ve raster shader, RT instance boyutu,
gaz deposit ağırlığı ile debug noktaları AYNI LUT'u okuyor. Billboard blend'i
sistemden profile taşındı. Eski projeler yüklenirken iki anahtarlı profile
çevriliyor.
Yeni dosyalar (vcxproj'a eklendi): `ParticleAppearanceProfile.h/.cpp`,
`ParticleSimulationAppearance.cpp`, `UI/ParticleBillboardBuilder.h/.cpp`,
`UI/ParticleAppearanceUI.h/.cpp`, `Viewport/ParticleBillboardData.h`,
`Backend/VulkanViewportParticles.cpp`. Silinen: `scene_ui_fluid_billboards.hpp`.
Shader: `particle_viewport.vert` yeniden yazıldı (frag aynı).

> ✔ 2026-09-26 (11:06 exe + 11:08 spv, canlı IPC ve viewport ekran görüntüsü):
> - **1 ✔**
> - **2 ✔:** 277/277 PASS, 26 `phase1.5:` satırının hepsi OK.
> - **3 ✔:** varsayılan profil sarı→kırmızı, küçülüyor.
> - **4 ✔ (IPC ile):** additive alev ve alpha duman aynı sistemde. Duman profili
>   canlı olarak maviye + emisyon 1→3 çevrildi; LUT yeniden yüklendi ve ekrana yansıdı.
> - **5 ✔:** kaydet→aç→kaydet'te id'ler aynı ve profil çoğalmıyor; dosyada legacy
>   anahtar 0. Sentetik v2 projesinde açık değerler birebir geçti, anahtarsız
>   emitter eski varsayılanları aldı, sistem `blend_mode=1` → iki profil alpha.
> - **Not:** `render_in_raytrace` açıkken RT küreleri viewport'ta opak çizilip
>   billboard'ları örtüyor. Bu eskiden de böyleydi, bu partinin hatası değil.
> - **Kalan:** 4'ün panel tarafı (Duplicate/önizleme şeridi), 6, 7, 8, 10.

1. **Önce shader'ları derle** (`compile_shaders.bat`), sonra exe.
   `particle_viewport.spv` eski kalırsa ne olur: pipeline 8 float'lık vertex +
   128 baytlık push constant bekler, eski shader ise 9 float + 144 bayt bekler.
   Parçacıklar hiç görünmez ya da dev ve anlamsız üçgenler olarak çizilir.
   **Bozuksa:** önce `x64/Release/shaders/particle_viewport.spv`'nin zaman
   damgasına bak.
2. **IPC testi** — `python scripts/ipc_test_client.py`. Bütün `phase1.5:`
   satırları OK olmalı. Asıl sinyaller:
   - `set_emitter(start_size) → error` ve `set_system(blend_mode) → error`.
     *Bozuksa:* anahtarlar sessizce yutuluyor; eski bir script görünümü
     değiştirdiğini sanıp hiçbir şey yapmaz.
   - `profiles landed in B only`. *Bozuksa:* `system_id` etkin sisteme
     düşüyor demektir.
   Python tarafı: `rt_api_smoke_test.py` →
   `[rt-smoke] rt.particle phase-1.5 appearance profiles: OK`.
3. **Görsel eşdeğerlik (en hızlı göz testi).** Boş bir sisteme "Add Point
   Emitter" ekle, render ayarlarında emitter_only'yi kapat, oynat. Görünüm eski
   varsayılanın aynısı olmalı: sarımsı başlar, kırmızıya döner, küçülür ve söner.
   Explosion presetini de dene: çekirdek büyük ve parlak, kıvılcımlar küçük.
   **Bozuksa:**
   - Her şey beyazsa ve boyutu 1 m ise parçacıklar fallback satırına (satır 0)
     düşüyor, yani `lut_row` eşleşmiyor.
   - Boyut doğru ama renk sabitse shader LUT'u yanlış indeksliyor.
4. **Tek sistemde iki blend** (bu partinin gerekçesi). Aynı sisteme ikinci bir
   emitter ekle. Emitter > Spawning Appearance Dynamics > **Duplicate** bas,
   Blend'i Alpha yap, rengi koyu gri ver. Parlak (additive) ve koyu (alpha)
   parçacıklar aynı anda görünmeli. Profil önizleme şeridi viewport'la aynı
   renkleri göstermeli.
5. **Eski proje migrasyonu.** Bu partiden önce kaydedilmiş, parçacık emitter'lı
   bir `.rtp` aç. Görünüm aynı olmalı. Panelde her emitter için
   "<ad> Appearance" profili görünmeli. Eski projede sistem Alpha idiyse
   profiller de alpha olmalı. Sonra kaydet (yeni ada), yeniden aç: aynı
   görünüm, aynı profil id'leri. Kaydedilen `.rtp`'de `start_size` geçmemeli
   (`findstr /c:"start_size" dosya.rtp` boş dönmeli).
   *Bozuksa* (ikinci açılışta profiller çoğaldıysa): migrasyon idempotent
   değil demektir, yani emitter `appearance_profile_id` olmadan yazılmış.
6. **Viewport boyutunu değiştir** (paneli sürükle). Parçacıklar kaybolmamalı.
   Boyut değişince LUT tamponu atılıp yeniden yükleniyor. **Bozuksa:**
   descriptor stale kalmış; `writeParticleLutDescriptor` çağrılmıyor.
7. **Fluid domain, Particles render modu** (bir sıvı domain'inde
   `fluid_render_mode = Particles`). Billboard'lar domain renginde, eski
   boyutta ve alpha ile çizilmeli. Bu yol da artık aynı LUT'tan geçiyor.
8. **RT render (Vulkan RT) parçacık instance'ları.** Parçacıklar ömürleri
   boyunca küçülmeli (boyut LUT'tan geliyor). "Inherit color" açıkken malzeme
   rengi = ilk emitter profilinin doğum rengi. Profilin rengini değiştirince
   RT malzemesi yeniden kurulmalı.
9. **Bilerek yapılan davranış değişiklikleri:**
   - Ash debris artık kendi "Ash Debris" profiliyle ve **alpha** ile çiziliyor.
     Eskiden hedef sistemin blend'ini alıyordu; koyu kül additive'de
     görünmezdi.
   - Profil düzenlemesi sim cache'ini **yalnızca opaklık eğrisi değişince**
     temizler. Renk sürüklerken sim sıfırlanmamalı.
10. **★ SİNSİ OLAN — gaz kuplajı.** Kamp ateşi / ignited fuel jet sahnesinde
    oynat; `particle.stats` → `grid_deposit_landed` ve duman sütunu önceki
    derlemeyle aynı büyüklükte olmalı. Deposit ağırlığı artık opaklığı
    lerp'lenmiş SoA'dan değil LUT'tan okuyor. İki anahtarlı doğrusal profilde
    sonuç birebir aynı olmalı. Duman hafifçe zayıflamışsa kimse bunu bug diye
    raporlamaz, "kalibrasyon" sanılır. Zayıflama varsa ilk şüpheli
    `sampleAppearance`'ın gördüğü profil id'si: 0 ise fallback'in 1→0
    opaklığı kullanılıyor demektir.

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
## Kinematic Collider Sources K1/K2 — build/live checks (2026-09-26)

> First K1 build live result: IPC CRUD/validation PASS. Character `1` exposed
> 80 bones; old auto-fit produced 64/64 resolved and moving proxies, but its
> arbitrary input order kept finger chains before the right lower leg/foot.
> Body-first + detail-opt-in is a post-build source fix and is the first check
> for the next build. A cleaned 22-proxy preview set was left in the live scene.
> The 2026-09-26 scale probe then found a second concrete unit bug: the 1%
> Mixamo rig turned the nominal 0.025-0.20 m auto-fit limits into 0.0008-0.002 m
> sampled radii. Auto-fit now evaluates limits and bone lengths in world metres
> and stores the converted bone-local dimensions; this change needs a rebuild.

1. Build once after the K1 source additions. Open a scene with one evaluated
   rig visible in `rig.list_characters`.
2. Open Simulation > Colliders > Kinematic Collider Sources. Select the rig,
   add a set, then run Auto Fit Skeleton. Confirm the list contains bounded
   sphere/capsule/box proxies, includes both feet, excludes finger/eye/skirt/
   twist/end detail bones by default, and no unresolved proxy appears at the
   world origin. Run `rt_probe_kinematic_rig_ipc.py 1`; expected PASS.
3. With viewport gizmos enabled, confirm cyan analytic outlines follow the
   animated bones. Toggle Show in Viewport and confirm the overlay disappears
   without disabling solver participation. The overlay already projects world
   dimensions; after rebuilding, rerun
   `python scripts/test/rt_setup_kinematic_preview_ipc.py 1`. It reuses and
   refits the exact `1 Kinematic Preview` set instead of accumulating copies.
4. From a separate terminal run
   `python scripts/test/rt_test_kinematic_collider_ipc.py`. Expected: `PASS`.
   In the Codex sandbox the named pipe may require an escalated run; Windows
   error 5 is not evidence that the app listener is absent.
5. For the real rig, call `physics.collider.proxy_set.sample` twice across two
   animation poses. Confirm `resolved=true`, centers follow their bones, and
   the second sample reports finite velocity. Scrub backwards, then later K2
   must reset authoritative history before any solver consumes it.
6. Save to a disposable `.rtp`, reopen it, and verify set/proxy IDs, shapes,
   local values, consumer mask, contact values and `viewport_visible` survive.
   A malformed or duplicate ID must reject the kinematic section instead of
   partially loading it.
7. Run `python scripts/test/rt_inspect_kinematic_scale_ipc.py` and compare one
   thigh plus each foot with the viewport. Sampled capsule segment/radius must
   be in world units. On the live 1% rig, a thigh/leg radius should be on the
   order of centimetres (roughly 0.09 m for a 0.52 m segment), not the measured
   0.002 m. Foot boxes must also remain proportional after the refit.
8. Put a low/medium-resolution Fluid domain around the legs, seed water, keep
   Fluid + Granular consumers enabled, and play the walk. The leg/foot cells
   must block liquid and transfer local limb velocity; a moving proxy must not
   leave solid ghost cells behind it. Scrub backwards and replay: no velocity
   explosion on the first resumed step. Shortcut:
   `python scripts/test/rt_setup_kinematic_water_ipc.py`. For an existing
   scene, run `rt_inspect_kinematic_fluid_scene_ipc.py` first. Do not infer the
   visible free-surface height from the domain AABB: `fluid.get` currently
   reports domain bounds and aggregate particle counts, not the particles'
   world-space height distribution. Confirm water depth in the viewport (or a
   future particle-bounds diagnostic) before judging foot contact.
   **2026-09-26 live partial PASS:** on the Vulkan water scene, an IPC-driven
   frame 0 -> 6 produced 6/6 collider-voxelization calls (6.981 ms total),
   retained 57,800 particles, and followed 0.26-0.34 m foot motion. The user
   confirmed visible foot gizmos and local water interaction. Rewind then
   reported `dropped_seeds=['Grid Domain 1']`; that run is invalid for
   ghost-cell acceptance. Refill using a persistent FillLevel/reseed recipe,
   then rerun `rt_probe_kinematic_rewind_ipc.py`. Remaining: moving-stamp ghost
   check and rewind/replay velocity.
   **Authoring bug found:** `Seed Fluid Now` always calls the SeedBox service;
   that service changes the runtime mode to `SeedBox`, even when the panel says
   `Fill Domain`. With `Recreate Seed on Reset` off, rewind then drops the tank.
   Until the shared core/API/IPC fix lands, use Fill Domain with Auto Reseed on
   Edit and do not press Seed Fluid Now, or explicitly arm Recreate Seed on
   Reset before testing rewind.
9. Repeat with a smoke-filled Gas domain and Gas enabled. CPU gas and Vulkan
   gas must both part smoke locally around the moving limbs; Vulkan uses the
   same host-stamped `solid`/`solid_vel` upload contract. If the whole column
   follows the character centroid, the old rigid-collider path is being used.
   Shortcut: `python scripts/test/rt_setup_kinematic_smoke_ipc.py`.
10. Disable Fluid or Gas in the set consumer mask while leaving Show in
    Viewport enabled. The outline must stay visible but the disabled solver
    must ignore the proxies. Re-enable it without recreating the set.
