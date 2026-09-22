# Fluid ilk ölçüm: sayaç denetimi ve IPC taban çizgisi

Durum: 2026-09-22, statik denetim, canlı taban çizgisi ve RAM kök nedeni tamamlandı.
Bu çalışma fizik veya çözücü davranışını değiştirmiyor.

## Canlı taban çizgisi (153³ Vulkan sahnesi)

Kullanıcının açık test sahnesinde tek `Grid Domain 1` vardı. IPC'de
`backend=vulkan`, `live_state=true`, 153³ ızgara (3.58 milyon hücre) ve
ölçüm sırasında 495.578 partikül okundu. `Probe-FluidBaseline.ps1` ile 5
ısınma + 30 adım, `dt=1/60` koşuldu. Ham veri:
[`fluid_baseline_153cube.json`](fluid_baseline_153cube.json).

| Sayaç | medyan | P90 |
|---|---:|---:|
| `particle.stats.grid_domain_ms` | 202.3 ms | 216.8 ms |
| IPC `fluid.step` çağrı süresi | 219.8 ms | 714.0 ms |

18–25. örneklerde IPC çağrısı yaklaşık 700–840 ms sürdü; iç domain adımı
çoğunlukla 200–206 ms kaldı (iki iç sıçrama: 331 ve 273 ms). Bu nedenle IPC
P90 değeri çözücü maliyeti olarak kullanılmaz. Isınma sonrasında partikül
sayısı 487.245'ten 495.578'e çıktı; 30 kayıtlı örnekte 495.578 olarak kaldı.
Backend alanı Vulkan kaldı, fakat bu alan gerçek GPU fallback durumunu
kanıtlamaz.

`perf.get_gpu_memory`: VRAM kullanımı koşu öncesi ve sonrası yaklaşık
1587 MiB (1,55 GiB); izlenen device-local 533 MiB,
izlenmeyen 1054 MiB. Bu 1054 MiB henüz `simulation` kategorisine
ayrılmadığından fluid tamponlarına doğrudan atfedilemez.

**Sonuç:** Bu sahnede yaklaşık 202 ms'lik bir domain adımı ölçüldü. Mevcut
ölçüler bunun ne kadarının kernel, upload, readback veya host taraması olduğunu
göstermiyor. Sıradaki ölçüm bu dört parçayı ayrı sayaçlarla görünür kılmalı;
tek toplamdan optimizasyon hedefi seçilmez.

## Ana bellek bulgusu: fluid kareleri sayaç dışında

Aynı sahnede süreç `WorkingSet=11.541 MiB`, `Private=13.196 MiB` düzeyindeydi.
`sim_cache.status` 114 RAM karesi için yalnız **151 MiB** bildiriyordu.
Zaman çizelgesi 113→114 ilerletildiğinde cache sayacı **1,83 MiB**, süreç özel
belleği yaklaşık **100 MiB** arttı. Oynatma başı yeniden 113'e alındı;
oluşan 114. kare cache'de kaldı. Genel partikül SoA'sında `alive_count=0` ve
`capacity=0`: yük ayrı genel partikül sisteminden gelmiyor.

Kök kod yolu `SceneData::compressGridDomainStates`: `c.meta = st` fluid
partiküllerini derin kopyalıyor. Sonra yalnız grid'in dokuz büyük float dizisi
`meta`dan çıkarılıyor. `CachedGridDomain::bytes()` ise önceden `sizeof` ve
sıkıştırılmış alanları sayıyordu; `meta.particles`, `meta.foam`, solid/weight
gibi kalan grid dizilerini saymıyordu. Yaklaşık 500 bin liquid parçacığın
temel SoA'sı tek başına kare başına onlarca MiB eder. Dolayısıyla bildirilen
151 MiB ve 4 GiB cache sınırı gerçek RAM'i temsil etmiyordu.

`SimFrameCacheMemory.h` bu kopyalarda kalan bütün grid, fluid particle ve foam
vektörlerinin **capacity** tahsislerini sayar. Sıkıştırılmış `Field` de
`size` yerine `capacity` ile hesaplanır. Mevcut `sim_cache.status` ve
Python yüzeyi aynı `SceneData` sayacını okuduğu için ikisi birlikte düzelir;
4 GiB sınırı da gerçek tutulan frame verisine göre çalışır. Bu bir **muhasebe
ve üst sınır düzeltmesi**; tek karenin kopyasını sıkıştırmaz. Sınır dolunca
yeni kareler RAM cache'e alınmaz ve timeline scrub eksik aralığı yeniden
simüle edebilir.

## Yeni derleme: Solid vekili ve SDF bellek kontrolü

Kullanıcının yeniden derleyip açık bıraktığı aynı 153³, yaklaşık 495 bin
parçacıklı sahnede (`render_mode=surface`, Vulkan) süreç özel belleği yaklaşık
**6064 MiB**, çalışma kümesi **4814 MiB** idi. `sim_cache.status` 43 RAM karesi
için **3.06 GiB** (`3290386475` bayt) bildirdi; 4 GiB bütçe henüz dolmamıştı.
Bu, önceki derlemede 114 kare için yalnız 151 MiB gösteren eksik muhasebenin
düzeldiğini canlı olarak doğruluyor. Önbellek sayacı tek başına süreç özel
belleğinin yaklaşık yarısını açıklıyor. GPU toplam kullanımı yaklaşık 1112 MiB,
bunun yaklaşık 788 MiB'si hâlâ kategorisizdi; kategorisiz kısmı fluid'e
atfetmiyoruz.

Başlangıç görünümü `material` idi. `solid` moduna geçince özel bellek yaklaşık
6065 MiB oldu; `material` moduna dönünce yaklaşık 6065 MiB kaldı. Önbellek
baytı (`3290386475`) ve GPU bellek sayacı (`1165942784`) değişmedi. Görünüm
başlangıçtaki `material` moduna geri alındı. Bu mod geçişindeki yaklaşık 1 MiB
fark, çift üretimin bu çalışmadaki RAM büyüklüğünü tek başına izole etmiyor:
`ParticleRenderBridge` Solid için oluşturduğu instance havuzunu Material'a
dönüşte silmiyor, yalnız ölçeklerini sıfırlayıp tekrar kullanım için tutuyor.

Kod yolu eşzamanlı üretimi doğruluyor: SurfaceSDF domain, görünüm modundan
bağımsız olarak `buildLevelSet` çağırıyor; Solid/Matcap ayrıca raster SDF vekili
olarak parçacık instanceları hazırlıyor (`scene_data.h` 2916–3045,
`ParticleRenderBridge.cpp` 1155–1205 ve 1373–1405). Yani kullanıcının şüphesi
geçerli; ancak **bu sahnedeki birkaç GiB ana bellek baskısının ana açıklaması
değil**. 153³ ve yüzey ayrıntısı 1x varsayımıyla tek float SDF alanı 13.7 MiB;
UVW alanı doluysa 41.0 MiB. Bu sahnede substance material binding olmadığı
için composition alanı boş kalır. Yaklaşık 500 bin `InstanceTransform` (40 bayt)
19 MiB düzeyindedir; render backend kopyaları ve geçici SDF scratch bu hesaba
dahil değildir. Yüzey ayrıntısı 2–4x seçilmişse SDF hücre maliyeti kübik
artar; bu ayar IPC'de okunmadığından sayıları kesin canlı tahsis olarak
sunmuyoruz.

Sonraki bellek işi, canlı SDF/UVW/scratch ve splat instance kapasitesini ayrı
sayaçlarla raporlamak; ardından cache karelerinin particle SoA kopyasını
sıkıştırmak veya bütçeyi kullanıcıya görünür ayarlanabilir yapmak. Solid
vekilini doğrudan silmek mevcut havuzun tekrar kullanım ve backend yeniden
kurulum davranışını değiştirir; ölçüm olmadan buna geçilmemeli.

## Gaz taşıma dersinin fluid GPU yoluna uygulanması (statik denetim)

`GAZ_ADIMI_TASIMA_MALIYETI.md` yeniden okundu. Gazda aşama kronometresi
upload, GPU kuyruğu ve sonraki `synchronize` beklemesini birbirine karıştırmıştı;
bir aşamanın 0 ms görünmesi GPU işinin bittiği anlamına gelmiyordu. Fluid'de
de `APICSolverStats` yalnız bütün aşamanın CPU duvar süresini tutuyor. Gerçek
GPU kernel süresi ve transfer baytları henüz ayrı, canlı sayaçlarda yok.

Mevcut kaynakta Vulkan fluid yolu şu host sınırlarını içeriyor (153³ ve
494.881 parçacıkla yaklaşık miktarlar). **Bunlar canlı ölçüm değil, ilgili
GPU dalları başarılı çalışırsa kaynak kodunun istediği baytlardır.**

| Kod yolu | Host → cihaz | Cihaz → host | Açıklama |
|---|---:|---:|---|
| Parçacık kuvvetleri | 30.2 MiB | 5.7 MiB | Position, velocity, affine, mass fraction yüklenir; velocity geri alınır. |
| P2G | 30.2 MiB | 41.3 MiB | Aynı dört parçacık alanı tekrar yüklenir; üç yüz hız alanı indirilir. |
| MGPCG giriş | 54.9 MiB | — | Üç yüz hızı ve host'ta kurulan fluid mask tekrar yüklenir. İç pressure transferleri ayrıca sayılmadı. |
| G2P sonu | — | 22.7 MiB | Parçacık velocity ve affine geri alınır. |
| GPU advection sonu | — | 11.3 MiB | Position ve velocity geri alınır. |
| Density splat | 5.7 MiB | 13.7 MiB | Position yeniden yüklenir, tüm density grid'i indirilir. |

Bu satırların toplamı yaklaşık **216 MiB/adım**; MGPCG içi, FLIP snapshot,
foam, başka alanlar ve transferlerin staging kopyaları hariç. Gerçek yolun
aktifliği yalnız `fluid.get.backend=vulkan` ile kanıtlanmaz. En açık tekrar
P2G'nin indirdiği yüz hızlarının hemen ardından MGPCG girişinin bunları
yeniden yüklemesi. Fakat arada `Fluid::step` sınır işlemleri var: gazdaki gibi
bu çifti kaldırmadan önce **host'un gerçekten hangi alanlara yazdığı ve
hangi tüketicinin host kopyasını okuduğu** ayrı saptanmalı.

Ek host işi: `buildFluidMaskFromParticles` her GPU pressure adımında maskeyi
temizleyip `grid.solid` için 3.58 milyon hücre tarıyor, sonra parçacıkları
işliyor. Bu sahnede collider yok; `enforceGridSolidFaceBoundaries` ise
`hasAnySolid()` ile erken çıkabiliyor. Bu iki işlem aynı `pressure_ms`
yakınında olduğundan adı `boundary` olan satırdan maliyet çıkarılamaz.

Bellek tarafında `ensureGridDomainComputeBuffers`, domain tipini ayırmadan
dört tam hücrelik **gas solid mask/velocity** tamponu ayırıyor; 153³ için
yaklaşık **54.7 MiB**. Fluid combustion state de koşulsuz iki hücrelik alan
(**27.3 MiB**) alıyor; bu sahnede substance binding yok. Bunlar kaynakta
kesin tahsis istekleri, fakat `perf.get_gpu_memory` compute kategorisi
olmadığından canlı VRAM kullanımına ayrı ayrı bağlanamıyor. Gas ajanı da
GPU bellek muhasebesi üzerinde çalıştığı için ortak `ensure` ve allocator
yoluna burada müdahale edilmedi.

**Sıradaki ölçüm eşiği:** fluid adımında gerçek GPU/CPU yolunu ve fallback'i
raporlayan `measured` durumu; aşama başına upload/download baytı ve süresi;
`endTransferBatch`/`synchronize` beklemesi; mask kurma, yüz sınırı ve
particle redistribüsyonunun host süresi. GPU timestamp yoksa `dispatch_ms`
kernel süresi diye adlandırılmamalı. Bu sayaçlar gelmeden defter, MGPCG veya
alt adımlama değişikliği için hız iddiası kurulmaz.

### 2026-09-22: fluid adımı transfer sayacı eklendi, canlı veri derleme bekliyor

`SimulationComputeContext` artık yalnız ölçüm kapsamı açıkken upload/download
**istenen baytları**, çağrı sayılarını ve host duvar sürelerini topluyor.
`SimulationTransferProbeScope`, tek fluid domain adımını kapsıyor; aynı
kapsamın içindeki MSF coupling çağrıları da sayılır. Gas adımında kapsam kapalı
ve ek kronometre çağrısı yok. Sonuç `fluid.step_stats(domain)` ile hem IPC hem
Python'da aynı `RtApi` çekirdeğinden okunuyor. Boşta veya cache'ten geri
yüklenmiş frame için `measured=false`; `fluid.get.backend` yerine
`gpu_status` ve aşama `*_on_gpu` bayrakları gerçek son adımı bildiriyor.

Sayaçlar **kernel süresi değildir**: `dispatch_call_ms` CPU'nun dispatch
çağrısında geçirdiği süre, `batch_end_ms` ve `synchronize_ms` kuyruk/fence
beklemesini içerebilir. `upload_call_ms` ve `download_call_ms` backend'in
çağrı süresidir; baytlar başarılı transfer değil, istenen transfer miktarıdır.
Başarısız GPU girişimi ve CPU fallback aynı adımda olursa iki yolun çağrıları
beraber görünür; `gpu_status` ile birlikte yorumlanmalı.

Kullanıcının **yeni derlemesinden sonra** açık sahnede oynatma durdurularak:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/ipc/Probe-FluidBaseline.ps1 `
  -Domain 'Grid Domain 1' -Warmup 5 -Samples 30 -TransferStats `
  -OutputPath docs/dev/fluid_transfer_153cube.json
```

Bu komut `fluid.step` ile **yeni** adımlar üretir; cache'teki kareleri gezmek
transfer ölçmez. Her satırdaki `step_stats.measured=true`, parçacık sayısı ve
GPU bayrakları kontrol edilmeli. `measured=false` ise adım koşmamıştır;
`gpu_status` fallback söylüyorsa 216 MiB statik GPU tahminiyle karşılaştırma
geçersizdir. `dispatch_call_ms` küçükken `batch_end_ms` büyükse işi GPU'dan
kaldırılmış sanmamalı; kuyruk o çağrıda beklenmiş olabilir.

### Yeni derlemede canlı sonuç: 500 bin parçacık, 153³

Kullanıcı derledikten sonra açık sahnede önce 5+30 adım çalıştırıldı. Emitter
parçacık sayısını 41.666'dan 291.666'ya çıkardığı için o seri geçiş ölçümü
olarak saklandı: [`fluid_transfer_153cube.json`](fluid_transfer_153cube.json).
Ardından 30 ısınma + 30 örnek yapıldı; örneklerin tamamında parçacık sayısı
**500.000**, çözünürlük 153³ ve GPU durumu
`P2G/pressure(MGPCG)/G2P/density/advect+boundaries; forces GPU` idi.
Ham veri: [`fluid_transfer_153cube_500k.json`](fluid_transfer_153cube_500k.json).

| Son adım sayacı | 30 örnekte değer / medyan |
|---|---:|
| `particle.stats.grid_domain_ms` | 225.4 ms (P90 234.3 ms) |
| İstenen host → cihaz | 220.105.804 bayt = 209.9 MiB/adım |
| İstenen cihaz → host | 142.846.140 bayt = 136.2 MiB/adım |
| **İki yön toplamı** | **346.1 MiB/adım** |
| Upload / download çağrısı | 20 / 15 |
| Dispatch / batch bitiş / explicit sync | 168 / 14 / 2 |
| `upload_call_ms` | 20.1 ms |
| `download_call_ms` | 0.95 ms |
| `batch_end_ms` | 109.4 ms |
| `synchronize_ms` | 1.55 ms |
| `dispatch_call_ms` | 0.61 ms |

Statik 216 MiB tahmin **eksikmiş**: FLIP snapshot, pressure içindeki ek
alanlar ve diğer yolları dışarıda bırakmıştı. 30 örnekte transfer baytları
birebir aynı kaldı. `batch_end_ms` içindeki 109 ms, GPU işlerinin o noktada
tamamlanmasını beklemeyi de içerir; bunu PCIe kopya süresi veya 109 ms saf
transfer kazancı diye yorumlamak yanlış olur. Aşama medyanları P2G 24.1,
pressure 52.2, G2P 21.1, advection 21.6, density 6.3 ms idi; bunlar da
izole kernel süresi değildir.

`perf.get_gpu_memory` koşu başı/sonu yaklaşık 1019 MiB toplam VRAM ve
789 MiB kategorisiz bellek bildirdi. Bir ek yeni adımda
`sim_cache.status` öncesi/sonrası **0 RAM frame** gösterdi: IPC `fluid.step`
cache karelerini tüketmiyor ve bu koşu cache'i doldurmuyor. Bu durum
transfer ölçümünü geçersiz kılmaz; yalnız timeline cache maliyeti ayrı.
Tek adımlık cache kontrolü: [`fluid_transfer_cache_check.json`](fluid_transfer_cache_check.json).

**Uygulama kararı:** Artık GPU yerleşikliği için yeterli büyüklükte gerçek bir
hedef var; körlemesine bütün indirmeleri silmek güvenli değil. Tam GPU
yerleşikliğinin ilk mimari adımı
domain başına particle position/velocity/affine ile MAC velocity için
host-current/device-current defteri ve boyut/nesil damgası olmalı. Emitter,
reseed, collider recovery, CPU kuvveti, cache restore ve domain resize host
yazarı; GPU force, P2G, pressure, G2P ve advection cihaz yazarı olarak
işaretlenmeli. Önce her tüketici `ensureHost/ensureDevice` kapısından geçmeli;
sonra yalnız **hiçbir host yazarı araya girmeyen** bir tekrar kaldırılmalı ve
referans fizik kareleriyle A/B yapılmalı. Gas ajanının ortak compute
altyapısındaki çalışmasıyla çakışmamak için bu defter fluid'e özgü modülde
yaşamalı. GPU bellek tarafında 55 MiB gas-solid ve 27 MiB combustion
tahsislerinin gerekliliği, `simulation` kategorisi canlı olduktan sonra
ayrıca sınanmalı.

### İlk kesinti: aynı adımın P2G parçacık yeniden yüklemesi

Ölçümden sonra `FluidGpuParticleUpload.h` ile fluid'e özgü, adım sınırında
sıfırlanan bir upload damgası eklendi. Kuvvet aşaması dört parçacık akışını
başarıyla yükledikten sonra, P2G aynı buffer handle'ları, aynı parçacık sayısı
ve yeterli kapasiteyi görürse ikinci yüklemeyi atlar. Aradaki collider
recovery host'ta pozisyon değiştirirse mevcut tazeleme yeniden damga basar;
reallocation, backend değişimi, eksik handle veya başarısız yükleme tam upload
yoluna döner. Çoklu granular substep'te yalnız ilk substep yeniden kullanım
isteyebilir. Step sınırı ve buffer release damgayı geçersiz kılar.

Bu değişiklik **yalnız aktarımı** kaldırır; shader, parçacık aritmetiği,
pressure veya render belleği değişmez. 500 bin parçacıkta dört akış
`position(12)+velocity(12)+affine(36)+mass_fraction(4) = 64` bayt/parçacık,
yani başarılı durumda **32.000.000 bayt = 30.5 MiB/adım** beklenir.
Karşılaştırma hedefi eski 220.105.804 upload baytından **188.105.804**
bayta, 20 upload çağrısından **16**'ya ve 14 batch bitişinden **13**'e
düşmesidir. Download baytı **142.846.140** kalmalı. Bu rakamlar yalnız
aynı 500 bin parçacıklı GPU yolu, tek substep ve collider recovery olmayan
sahnede beklenir; süre kazancı yeniden derleme sonrası ölçülmelidir.

Yeni derlemeden sonra üstteki `-TransferStats` komutu aynı sahnede tekrar
çalıştırılır. `measured=true`, bütün GPU bayrakları açık ve 500 bin parçacık
sabit olmalı. Upload 32 milyon bayt azalmıyorsa yeniden kullanım kapısı
çalışmıyor; daha fazla azalıyorsa beklenmeyen bir aşama atlanmış olabilir.
Download veya fizik durumunda sistematik bir sapma varsa optimizasyon geri
alınmalı ve aradaki host yazarı bulunmalı.

### Derleme sonrası doğrulama (2026-09-22)

Açık 153³ Vulkan sahnesinde emitter 500.000 parçacığa ulaştıktan sonra
30 sabit yük adımı ölçüldü. Her adımda parçacık sayısı 500.000,
`measured=true`, `ok=true` ve P2G/pressure/G2P/density GPU bayrakları açıktı.

| Sayaç, 30 adım | Önce | Yeniden kullanım sonrası |
|---|---:|---:|
| Host → GPU | 220.105.804 bayt | 188.105.804 bayt |
| Upload çağrısı | 20 | 16 |
| GPU → host | 142.846.140 bayt | 142.846.140 bayt |
| Download çağrısı | 15 | 15 |
| Batch bitişi | 14 | 13 |
| `grid_domain_ms` medyan | 225,88 ms | 214,52 ms |
| `grid_domain_ms` P90 | 234,28 ms | 234,28 ms |
| P2G aşaması medyan | 24,11 ms | 19,20 ms |

**Aktarım sonucu kesin:** dört parçacık akışının 32.000.000 baytlık ikinci
yüklemesi her adımda atlandı; GPU → host trafiği değişmedi. Süreler ayrı
koşulardan geldiği için 11,36 ms medyan farkının tamamı bu değişikliğe
atfedilemez. P90 değişmedi. Sahne cache'e yazılmadı (`ram_frames=0`);
görsel/fizik eşdeğerliği bu sayaçlarla kanıtlanmıyor.

Ham veriler: `fluid_transfer_153cube_500k.json` (önce) ve
`fluid_transfer_153cube_500k_reuse_steady.json` (sonra). İlk yeni derleme
koşusu `fluid_transfer_153cube_500k_reuse.json` emitterin 250.000'den
500.000'e doluşunu içerir; sabit yük süre karşılaştırmasına katılmadı.

## Sayaçların gerçek kapsamı

| Gösterilen satır | Kodun ölçtüğü aralık | Yorum sınırı |
|---|---|---|
| `particle.stats.grid_domain_ms` | `ParticleSimulationSystem::step` içindeki bütün `stepGridDomains` çağrısı | Bütün domain'ler ve domain ön/son işleri dahil. Tek domain sahnesinde ilk toplam için kullanılır. |
| `APICSolverStats.total_ms` | `Fluid::step` girişinden advection bitimine (`stage_end`) | Sonraki `redistributeParticles` ve `advanceMaterialCoordinates` hariç. GPU yolunda ikinci, kısa `Fluid::step` çağrısının süresi; tam fluid adımı değildir. |
| UI `Step total` | `max(fs.total_ms, p2g+pressure+viscosity+g2p+advect+density)` | Gerçek duvar saati toplamı değildir. Örtüşen kapsamlar ve atlanan işler var. |
| `p2g_ms` | CPU'da `particleToGrid`; GPU'da `ensureGridDomainComputeBuffers` + `runGpuFluidP2G` çevresi | GPU satırı aktarım, dispatch, senkronizasyon ve olası tahsisi ayırmaz. Başarısız girişimde süre sıfır kalabilir. |
| `boundary_ms` | CPU `enforceSolidBoundaries` | P2G sonrası FLIP snapshot kopyası bunun dışında. GPU bölünmüş yolun ilk çağrısında oluşur, fakat ikinci çağrı stats'ı sıfırladığı için panelde korunmaz. |
| `viscosity_ms` | CPU'da maske yapımı + çözücü + ikinci boundary; GPU'da `runGpuFluidViscosity` + `enforceGridSolidFaceBoundaries` | İzole kernel süresi değildir. |
| `pressure_ms` | CPU projection; GPU `runGpuFluidMGPCGPressure` + host boundary | GPU satırı CG dot readback ve başka transferleri içerebilir. `pressure_cg_dot_ms` bunun alt kümesidir. |
| `g2p_ms` | CPU'da `gridToParticle` + granular işlem + air drag + damping; GPU'da `runGpuFluidG2P` çevresi | GPU timer'ı solid parcel velocity restore döngüsünden önce biter. |
| `advect_ms` | CPU `advectParticles`; GPU `runGpuFluidAdvectTail` | CPU reseed ayrı ve zamanlanmamış; GPU fonksiyonu transfer/senkronizasyon içerebilir. |
| `density_ms` | `runGpuFluidDensitySplat` veya CPU fallback çevresi | `ensureGridDomainComputeBuffers` ve fallback denemesi dahil olabilir. |

Kaynaklar: `ParticleSimulation.cpp` 10175–10424, 10767–10790, 11935–12119;
`APICFluidSolver.cpp` 3054–3577; `scene_ui_simulation_domains.cpp` 3895–3919.

## İlk canlı ölçüm

Kullanıcı uygulamayı açar, simülasyon oynatımını durdurur ve **tek fluid
domain'li** bir sahneyi kaydedilmiş başlangıç durumundan yükler. Domain'de
partikül ve `live_state=true` bulunmalı. Aynı sahnede gaz veya başka grid domain
olmamalı; `fluid.step` tüm `SimulationWorld`'ü ilerletir. Önce büyük çözünürlük:

```powershell
.\scripts\ipc\Probe-FluidBaseline.ps1 -Domain Water -Warmup 5 -Samples 30 -OutputPath .\fluid_large.json
```

Küçük çözünürlük için aynı fizik, aynı başlangıç ve yaklaşık 20 kat az hücreyle
ayrı sahne yüklenir. Yeniden açmak simülasyon durumunu sıfırlar:

```powershell
.\scripts\ipc\Probe-FluidBaseline.ps1 -Domain Water -Warmup 5 -Samples 30 -OutputPath .\fluid_small.json
```

Komutlar `Water` yerine gerçek domain adını kullanmalı. Probe yazarlık
ayarlarını değiştirmez fakat `fluid.step` ile simülasyon durumunu ilerletir.
Sonuçlar `grid_domain_ms` ve IPC duvar süresi için medyan/P90, her örnekte
partikül sayısı/backend, ön/son GPU bellek dökümünü içerir.
`fluid.get.backend` yapılandırma bilgisidir; gerçek GPU fallback durumunu tek
başına kanıtlamaz. Her koşuda Fluid Step Stats panelindeki `Compute` satırının
ve aşama sürelerinin ekran görüntüsü de alınmalı.

**Beklenen:** `live_state` ve partikül sayısı dolu, backend koşu boyunca aynı,
örnek süreleri pozitif. `grid_domain_ms` IPC süresinden genellikle küçük;
aralarındaki fark GPU transferi diye etiketlenmez, IPC ve UI kuyruğu da vardır.
`untracked_bytes` ölçümü bellek kategorileri tamamlanana dek çözücü tamponu
olarak yorumlanmaz.

**Bozuksa:** `live_state=false` veya sıfır partikül → çözücü ölçülmedi;
backend değişimi veya panelde GPU fallback → karşılaştırma geçersiz; P90 çok yüksek ve medyan düşük →
başlangıç/tahsis/işletim sistemi gürültüsü veya kararsız iş yükü; örnekler
boyunca partikül sayısı çok değişirse çözünürlük karşılaştırması aynı iş yükü
değildir. Sayılar makul görünse bile `grid_domain_ms` tek domain doğrulaması
olmadan o fluid domain'in süresi sayılmaz.

## İkinci ölçüm için gereken sayaçlar

Mevcut satırlardan host↔cihaz bayt ve `synchronize()` bekleme süresi çıkarılamaz.
İlk iki JSON, büyük/küçük ölçek davranışını gösterir ve hangi sahnenin ayrıntılı
instrumentasyona uygun olduğunu seçtirir. Sonraki kod partisi transferi
`upload / dispatch / sync / download` olarak, özellikle P2G, pressure, G2P,
advection ve density içinde ayrı saymalı; `total_ms` gerçek domain adımını
kapsamalı ve bu ölçüler script ile IPC'de aynı çekirdek verisinden okunmalı.
Bu ayrım yapılmadan MGPCG veya advection alt adımlama değiştirilmez.
