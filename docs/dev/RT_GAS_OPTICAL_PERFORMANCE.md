# RT gas optical traversal performance

> **Durum:** AKTİF — 2026-10-25; kutu daraltması canlı doğrulandı; iç içe katı izi (Seçenek A) ölçüldü, gaz march'ını ~2.8× yavaşlattı ve GERİ ALINDI. Gaz içi splat maliyetinin asıl kaynağı devir izi değil march'ın kendisi. Sıradaki: ince/geniş içerik için blok başına çoklu AABB.

Kaynak başına materyal sistemi geri alındı ve kullanıcı eski davranışı doğruladı.
Bu parti yalnız mevcut RT maliyetini azaltır. Fog'un gas emitter blackbody
ayarlarını alması sonraki ayrı iştir. Fizik, panel, materyal schema, IPC/Python
operasyonları, OptiX ve VkVolumeInstance ABI değiştirilmedi. Backend mevcut
_uvw_pad[0] reserved float üzerinden yalnız [LIVE SIMULATION] dense-to-sparse
üreticisini işaretler; yeni alan/stride yok. C++ ve shader birlikte build gerekir.

## Taşınan fikir ve uygulama

Separate yolundaki optik görünürlükten boş alan elemesi eski NanoVDB yoluna
uyarlandı. Önceki nanoEmptyTileStep yalnız topolojik inaktif tile atlıyordu;
pozitif ama cutoff altındaki leaf'ler tam örnekleniyordu. Yeni yardımcı mevcut
NanoVDB min/max istatistiğini kullanır. Leaf max aynı remap/monotone curve ile
cutoff altında kalıyorsa sadece trilinear desteğin o leaf içinde kaldığı
[L,L+7) bölgesini atlar. Son voxel slab ve eksik min/max istatistiği normal yola
kalır. Leaf max yalnız aktif değerleri kapsadığı için optik leaf elemesi sadece
zero-background live producer üzerinde etkin; 1e-6 copyFromDense tolerance da
üst sınıra katılır. Arbitrary import VDB inactive değerleri leaf max dışında
kalabilir; bunlarda optik leaf elemesi yapılmaz, affine sampler kullanılabilir. density_leaf/majorant_segments_skipped sayacı bu elemenin kanıtıdır.

Primary ve gölge march aynı affine ışının world/local/index dönüşümünü bir kez
hesaplar. Aynı 8 köşeli interpolasyon, AABB testi, remap/curve/cutoff feather ve
multiplier korunur. Graph veya material density noise varsa mevcut sampler'a
dönülür. Primary overlap halinde skip yapılmaz; companion görünürlüğü korunur.
Gölge adımları atlanırken başlangıç quadrature kafesi korunur; adım bütçesi,
ışık seçimi ve opacity estimator değiştirilmedi. Zero-density primary örneğinde
companion/graph yoksa kullanılmayacak sıcaklık okumaları yapılmaz.

Önceki source shadow cache doğrudan taşınmadı: mapped extinction interpolasyonu
nonlinear cutoff davranışını değiştiriyordu. Bu parti yeni CPU grid/cache/upload
üretmeden mevcut NanoVDB verisini kullanır. Hızlanma oranı henüz ölçülmedi.
Dense live gas majorant yolu mevcut haliyle kalır; bu kazanım öncelikle Matter
host NanoVDB yayınını hedefler. Dolu/emissive bölgelerdeki march ve SDF probe
maliyeti hâlâ devam eder; her sahnede 5x hızlanma beklentisi verilmez.

## Doğrulama

Agent build veya shader derlemesi çalıştırmaz. Non-build kontrol scripti:
`scripts/test/check_rt_gas_optical_contracts.py`; release mirror aynı.
CPU sayısal bound/lattice kontrolleri GLSL execution kanıtı değildir.
Mevcut `render.volume_counters` / `render.volume_stats` IPC ve Python üzerinden
aynı sayaçları verir; yeni authoring yeteneği/ayar eklenmedi.

Build sonrası aynı paused kare/kamera/çözünürlük/spp ile karşılaştır; gazın
kaybolması hızlanma değildir. Önce sayaçla density_samples/volume_rays,
majorant_segments_skipped, shadow_density_samples/volume_rays ve ray sayısını
kaydet. Sonra sayaç kapalı gerçek frame sürelerini ölç. Yeni modül isimleri
volume_nano_optical.glsl, volume_rt_shadow_march.glsl, volume_rt_empty_sample.glsl.

## 2026-10-22 — Canlı maliyet ölçümü (kare 101)

Kullanıcının açık 4-emitter sahnesi, 1680x945 viewport, Vulkan RT pipeline ready.
Başlangıç scatter=0, density multiplier=11.111, absorption=1.46, shadow_steps=8,
shadow_stride=4, max_steps=256; curve enabled, emission mode2, random_walk=false,
internal_light_steps=0. Aynı timeline kare ve kamera korundu. Materyal ve sayaç
ayarları finally geri yüklendi ve eşitlik doğrulandı. Scene save yapılmadı;
transient API material editleri project modified bayrağını set etmiş olabilir.

**Geçerli RT süre yöntemi:** viewport.render_frames kullanılmadı; normal main-loop
Vulkan backend sample sayacı 8'den 40'a ilerlerken geçen wall time ölçüldü.
Sayaçlar kapalıydı. Bunlar UI/scheduler dahil ms/RT sample değeridir, saf GPU
 timestamp değildir. Ham sonuç: rt_gas_cost_natural_rt_20261022.json.

| Durum | ms/RT sample |
|---|---:|
| Scatter0 | 108.96 |
| Scatter0.001 | 274.35 |
| Scatter1 | 250.09 |
| Scatter1, shadow_steps0 | 177.40 |
| Scatter1, density_multiplier0 | 89.27 |
| Scatter0 tekrar | 105.19 |

0.001 ve1 aynı pahalı scattering yolunu açar. Aralarındaki ~%10 farkı tek koşuyla
katsayıya bağlı hızlanma diye yorumlamıyoruz; optical depth/termination/ölçüm
oynaması da etkiler. Scatter1, scatter0 referansına göre ~2.3x; density0'a göre
~2.8x. Shadow_steps0 ile scatter1 süresi ~%29 azalır, ancak scattering'in ek işi
kalmaya devam eder. Density0 bir görünür-gaz performans baseline'ı değildir.

Ayrı instrumented display-loop sayıları (rt_gas_cost_live_20261022.json):
volume ray başına density ~34.96 (scatter0), ~34.77 (scatter1), ~34.99 (density0).
Scatter1 shadow samples/ray ~5.37, scatter0.001 ~4.19; stride16 ~2.34. Optical
skip ~9.85–9.89 milyon segment; walk_paths0; solid probes ~1.68 milyon. Optical
skip çalışıyor. Density0 bile raw density sampler'a giriyor; output0 olmak
volume traversal'ı tamamen kaldırmıyor. Sayaçların hepsi normal display-loop
Vulkan akışından; requested CPU pass adedine göre per-frame sayılmamalı.

**Ölçüm aracında saptanan açık:** viewport.render_frames, backend kontrolünden
sonra Renderer::render_progressive_pass çağırıyor; bu fonksiyon CPU renderer.
İlk batch wall/submission süreleri RT için GEÇERSİZ olarak JSON'da işaretlendi.
Ayrıca VulkanBackendAdapter::getMillisecondsPerSample sabit0 döndürüyor; status
ms_per_sample alanı ölçüm olarak kullanılmadı. perf.list snapshot da saklandı;
reset edilmediği için tarihsel max/total sahneye özel yeni pencere değildir.

Sonraki performans işi: scatter branch'teki ışık/sky hesapları ve gölge sorguları.
closesthit.rchit::computeVolumeShadowTransmittance ayrıca surface-to-light gaz
sönümünü march ediyor, mevcut shadow_density_samples bunu saymıyor; toplam gölge
maliyetinin yalnız5.37 örnek/ray olduğu söylenemez. Şimdilik render koduna yeni
optimizasyon uygulanmadı. Fog/blackbody ayrımı ayrı sonraki iş olarak korunur.

## 2026-10-22 — Domain / splat / solid probe maliyet ayrımı

Kullanıcı: gaz maliyeti, diğer çözücülerin splat sphere maliyeti ve domain'in
kamerayı kaplaması birlikte render adımını uzatıyor. Hipotezin kapsama kısmı
doğru: büyük AABB çok sayıda kamera/secondary ışınını volume closest-hit'e sokar.
Boş blok skip'i march örneklerini azaltır; volume entry, solid probe ve handoff
maliyetlerini kendiliğinden kaldırmaz.

Mevcut101kare ölçümünde arbiter_gate_open0, arbiter_candidates0,
arbiter_no_crossing0: SDF araması bu sahnede çalışmamış. Gas+SurfaceSDF birlikte
aktif olduğunda nearestSurfaceSDFCrossing ayrı candidate grid march'ı yapar;
bu akış gazın her density adımında çağrılmaz, segment başında çalışır.

Solid probe ise scatter0/1/.001 durumlarında ~1.678m kez çalışmış ve ~691k hit
bulmuş. Aynı kayıtta volume_rays~987k. Sayaçlar farklı erken-return yollarını
kapsadığı için bu oranı her rayda1.7 aynı probe yapılıyor diye yorumlamıyoruz.
Mask0xF5 gas/fog0x02 ve SDF0x08'i dışlar, splat sphere0x04'ü tutar. Probe
triangle/sphere yüzeyini bulup march'ı o mesafede keser; handoff aynı yüzeyi
shade etmek için tekrar traversal'a sokabilir. Probeları körlemesine kapatmak
hacim içindeki sphere/zemin/yüzeyleri kaybettirir. Primary probe bu nedenle
ışın başına/segment başına, optik kalınlık tahminiyle iptal edilmiyor.

Splat/foam küreler compact procedural sphere AABB BLAS yolunu kullanıyor;
her küre ayrı TLAS instance değil. Yine de probe ray'leri analytic sphere
intersection shader'ını çalıştırabilir. Gerçek sphere intersection adedi ve bu
BLAS'ın GPU zamanı mevcut volume sayaçlarında ayrı ölçülmüyor. Rakamları sphere
çiziminin süresi olarak sunamayız.

Sonraki optimizasyon öncelikleri:
1. Kamera ışını ve aynı yönde transparent handoff boyunca geçerli en yakın
   solid mesafesi tekrar kullanılabilir mi araştır. Yön/segment sınırı değişince
   cache geçersiz olmalı; refraksiyon, SDF ve splat görünürlüğü korunmalı.
2. Volume bounds optik destekle daraltılabilir mi ölç; gas data'sız AABB girişini
   azaltmak march skip'inden farklı kazançtır. Cutoff/material edit, sıcaklıkla
   emission, graph/noise ve animasyon bounds geçerliliği birlikte ele alınmalı.
3. Saçılma light/sky maliyeti ve surface-to-light volume shadow sorgularını
   ayrı instrument et. Ek sayaç varsa ortak Vulkan/IPC/Python yüzeyinden sun.
4. Mevcut sahnede gaz görünürken sphere representation A/B ve ROI bazlı karşılaştır;
   solver adımını ve kamera/kareyi değiştirmeden geometri/render etkisini ayır.

Bu devam partisi source audit ve ölçüm yorumudur; shader/physics kodu değişmedi.

## 2026-10-22 — Doğruluk önce: embedded solid mesafesi / gaz ön tabakası

Kullanıcı RT'de solid önündeki gazın kaybolduğunu/arkada kaldığını, domain içi
gazın sınırdakinden daha kırpılmış göründüğünü bildiriyor. RayFusion doğru.
Optimizasyon önceliği bu gözlem nedeniyle doğruluk olarak değiştirildi.

İki kesin kod kusuru düzeltildi; gözlemin tamamını çözdüğü henüz iddia edilmiyor:
- Gas ve SurfaceSDF embedded-solid probe TerminateOnFirstHit ile any-hit'te
  terminateRay kullanıyordu. BVH candidate sırası mesafe sırası değildir.
  Yeni ortak helper candidate'ı kabul eder, sentinel'i korur, .z found flag'i
  ve minimum distance tutar. Caller first-hit flag'ini kaldırır; traversal
  accepted mesafeyle daralarak daha yakın hit'i arar. Triangle/sphere ve hair
  aynı sözleşmeyi kullanır. Shadow miss .w=1 yazsa da .z hit bilgisi bağımsızdır.
- Gaz solidT-0.01'e kadar entegre ediliyordu: her solid/splat önünde sabit1cm
  gaz tabakası siliniyordu. tFar artık min(tFar,solidT). Handoff zaten orijinal
  ışın başlangıcından gas maskesini dışlayarak yeniden trace ediyor; bu geçiş
  için gazı1cm erken bitirmek gerekli değildi. Sphere bit0x04 korunur.

Yeni özellik/API/IPC ayarı veya payload/descriptor ABI eklenmedi. Gerçek gas
şiddeti, scattering, fizik ve RF yolu değiştirilmedi. Önceki optical skip ve
index sampler korunuyor; bunların görsel etkisi de build sonrası kontrol
edilmeli. Sıralama düzelip iç gaz seyrekliği kalırsa skip/step/denoiser etkisi
ayrı izole edilmeli; tek bulguyla her şeyi çözüldü sayma.

## 2026-10-22 — Boş gaz içinde kaybolan procedural splat sphere

Kullanıcı nearest-solid build sonrası gas sıralaması/yoğunluğu doğru, ancak
gaz olmayan hücrelerde splat sphere kayboluyor; SDF yüzeyi görünür bildiriyor.
Kök: VulkanRTPipelineBuild sphere hit group any-hit slotunu UNUSED bırakıyordu.
Yeni nearest-solid probe .z found flag'ini any-hit'te yazıyor; procedural sphere
intersection kabul edilse bile SkipClosestHit nedeniyle bu flag hiç yazılmıyor.
Probe bir sphere'e çarptığı halde gaz yolu no-solid sanıp box exit'e ilerliyor.
Bu önceki nearest-hit düzeltmesinde kaçırılan sphere SBT tüketicisidir.

Sphere group mevcut shadow_anyhit stage'ine bağlandı. Sphere hit-kind dispatch
triangle inst/vertex buffer okumalarından ÖNCE yeni sphere_shadow_probe.glsl'ye
gider. Volume sentinel aynı nearest-distance helper'ını kullanır; .z density'den
bağımsız yazılır. Camera/bounce hit kabul edilir; gerçek shading mevcut sphere
closesthit'te kalır. Ordinary sphere shadow occlusion korunur. SSS sentinel için
nearest distance + analytic sphere normal; bevel probe tint gibi işlenmez.
Fizik/BSDF/material/API/IPC değişikliği yok; payload ve SSBO stride aynı.

C++ pipeline + shadow_anyhit shader birlikte build gerekir; agent çalıştırmadı.
Kabul: gazsız hücrede ve density multiplier0 iken sphere görünür; gas ön tabakası
ve sphere arkasında attenuation sıralaması doğru; SDF/triangle/hair korunur.

## 2026-10-22 — Düzeltilmiş sphere/solid build referansı

Kullanıcı sphere/gas görünürlüğünü doğruladı. Açık rendered viewport ve ready
pipeline'da101karede yeniden natural Vulkan örnek8→40 ölçümü yapıldı. Başlangıç
scatter0, density11.111, absorption1.46, shadow_steps8, shadow_stride4 idi.
Orijinal materyal ve sayaç durumu geri yüklendi (JSON equality true), kare101.
Sahne kaydedilmedi. Render kaynakları bu ölçüm partisinde değiştirilmedi.

| Durum | ms/RT sample (wall) | Density/volume ray | Shadow/volume ray |
|---|---:|---:|---:|
| Authored scatter0 | 164.91 | 26.68 | 0 |
| Scatter0 ikinci | 149.25 | 26.68 | 0 |
| Scatter0.001 | 402.28 | 26.68 | 8.08 |
| Scatter1 | 417.67 | 26.21 | 8.76 |
| Scatter1 shadow_steps0 | 298.64 | 26.10 | 0 |
| Scatter1 shadow_stride16 | 358.79 | 26.26 | 3.58 |
| Scatter1 density0 | 149.56 | 26.42 | 0 |
| Authored tekrar | 169.90 | 26.68 | 0 |

Raw: rt_gas_cost_postsolid_20261022.json. Sayaçsız süre ve ayrı instrumented
window kullanıldı. Wall süre UI/poll/main-loop scheduling içerir; saf GPU
 timestamp değildir. Scatter0 kontrolü149–170ms oynadı; küçük yüzde farkları
kesin kazanç sayılmamalı. .001 ve1 aynı pahalı scattering yoluna giriyor.
Scatter1'i density0 ile kıyaslayınca ~2.79x. Shadow_steps0 ~%28.5, stride16
~%14.1 daha kısa. Bu knob değişimleri görüntüyü değiştirebilir; optimizasyon
teslimi olarak sunulmadı ve kalıcı tutulmadı.

Önceki eksik-sphere build'den density/ray35→26.7 düşerken solid probes/path
~.53→1.04 oldu. Bunlar düzeltmeyle daha yakın sphere yüzeyinin bulunması ve
ardından gerçek geometri shading/continuation yapılmasıyla uyumlu. Farkı saf
bir probe süresi gibi yorumlamıyoruz; eski build görüntüsü doğru değildi.

Density0 hâlâ26.42 density/ray ve~1.06 solid probes/path taşıyor: sampler sonunda
multiplier0 oluyor ama AABB entry/solid probe ve raw lookup maliyetleri kalıyor.
Bu baseline gaz görünürkenki maliyeti tahmin etmek için tek başına kullanılamaz.
SDF arbiter mevcut kayıtta0; yeni örnekte random-walk da kapalı.

Öncelik: scattering light/sky hesaplarının ve surface-to-light volume shadow
march'ının ayrı maliyetini instrument edip tekrarları azaltmak; ardından
same-direction solid-distance reuse ve optical bounds. Probeları kaldırmak
veya gölge kalitesini gizlice düşürmek hedef değil. Fog/emitter blackbody ayrımı
sonraki ayrı doğruluk işi olarak duruyor.

Tekrar ölçüm aracı: scripts/test/rt_probe_gas_cost_ipc.py (release mirror aynı).
Default salt okuma, --apply bütün kısa A/B'leri yapar ve finally restore eder.
Pipeline ready + rendered ister, timeline frame değişirse durur. Per-device
counter state farklıysa atomik restore mümkün olmadığı için yazmadan reddeder.
viewport.render_frames çağırmaz; CPU reference benchmark tuzağına düşmez.

## 2026-10-22 — Kaliteyi koruyan ilk tekrar azaltma partisi

Kullanıcı: gözle görünmeyen çok zayıf scattering adımları elenebilir; yoğun alanda
adım uzatılması düşünülebilir; boş alan doğru elenmeli. Karar: katsayıya bağlı
gizli minimum scattering eşiği ve yoğunluğa bağlı kaba adım eklenmedi. HDR ışık,
ince kontrast/emission ve uzun yol boyunca toplam katkı, küçük katsayıyı görünür
kılabilir. Adaptif adım ancak kaynak/ışık/optik değişim hata sınırıyla ele alınmalı.

İlk küçük uygulama volume_phase_cache.glsl: ray başına seçilen scene light pair
slot0/1 için yalnız directional type1, sun slot2 için sabit single-medium
HG phase tekrar kullanılır. Graph veya companion varsa cache kapalı. Point/area
ışıkları konuma bağlı olduğu için hesap devam eder. Multiple-scattering/albedo
blend örnek başına aynı ifadeyle korunur. RNG çağrıları ve sırası değişmedi;
density/temperature/empty skip/shadow stride/march steps değiştirilmedi. Cache
lazy doldurulur; ışık katkısı olan ilk örnek gelene kadar phase hesaplanmaz.

Bu sabit raw phase'in tekrarını kaldırır; driver zaten benzer hoist yapıyorsa
kazanç küçük veya0 olabilir. Register/branch maliyeti de ölçülmeli. Kullanıcı
build'inden önce hızlanma veya görsel eşitlik iddiası yok. Native GPU execution
ve RT/RF görüntü karşılaştırması hâlâ kabul kapısı. C++/API/IPC/schema değişmedi;
render volume_closesthit shader/include build'i yeterli.

### Phase cache compiler düzeltmesi

Kullanıcı glslc unexpected INVARIANT hatası bildirdi. Helper parametresi GLSL
reserved qualifier olan invariant adını kullanıyordu; canReuse olarak düzeltildi.
Statik wiring/nümerik testler GLSL parser değildir ve bunu yakalamamıştı; regression
kontrolüne bu reserved parameter eklendi. Agent shader compile çalıştırmadı.

Kullanıcı volume_closesthit derlemesinin uzun sürdüğünü de bildiriyor. Mevcut
compile_shaders.ps1 bütün stage'leri -O ile derliyor; volume_closesthit yaklaşık
5k satır ve dahil edilen material/BSDF/volume yollarını aynı modülde optimize
ediyor. Bu kaynak boyutu/optimizasyon baskısına işaret eder; gerçek compile
profilimiz yok. Önceki pipeline driver süreleri glslc süresi değildir. -O
kapatılmadı; compile hızını render performansını düşürerek düzeltme yapılmadı.
Phase cache'in kazancı ölçülmedi; compile/register/render maliyeti artarsa
bu küçük deney geri alınmalı, faydası kanıtlanmış gibi tutulmamalı.

## 2026-10-22 — Phase-cache build: ölçüm, geri alma yok

Kullanıcı uzun driver compile bildirip kör geri almayı reddetti. Kod geri
alınmadı ve bu ölçüm partisinde render/pipeline kaynakları değiştirilmedi.

Tamamlanan IPC render.rt_pipeline_status: ready, build_count1, compile_seconds
495.2853215, deferred_threads13. Önceki doğru-sphere build kaydı313.5575651sn;
şimdiki toplam~%58 daha uzun. Stage_seconds tüm stage'lerde null ve cache_hit
null: bu driver çalıştırmasında per-stage creation feedback geçerli değil.
Dolayısıyla toplam495sn volume shader'a kesin olarak atanamaz. cache_loaded
false/loaded_bytes0; bu kayıttaki ilk pipeline build'inde disk blob yüklenmiş
olduğu doğrulanmıyor. Diskte build sonrası~11MB cache mevcut. SPIR-V değişince
hash/stamp invalidasyonu beklenir; bu tek başına bug değildir.

Mevcut derlenmiş SPIR-V salt okuma opcode analizi:
- volume_closesthit:4,648,372byte,252,254instruction,1function; en büyük function
  247,732instruction,674LoopMerge,7,585SelectionMerge,16,145Phi.
- closesthit:1,637,168byte,87,021instruction,1function.
- cloud_callable:1,616,688byte,88,344instruction,1function.
- shadow_anyhit:34,600byte,2,008instruction.
Volume module, monolithic inline/SSA büyüklüğü nedeniyle kuvvetli adaydır;
SPIR-V boyutu driver zamanı değildir ve önceki modül boyutu elimizde yok.
Aynı fonksiyonu farklı .glsl include'lara taşımak bunu bölmez; glslc -O
çıkışında tüm yollar tek main fonksiyona açılmıştır. Hash/opcode kanıtı
rt_phase_cache_spirv_20261022.json, pipeline snapshot
rt_phase_cache_pipeline_20261022.json altında. Agent derleme çalıştırmadı.

Aynı101karede natural RT sayaçsız örnek8→40 tekrar ölçümü:
| Durum | ms/RT sample wall |
|---|---:|
| Authored scatter0 | 185.26 |
| Scatter0 ikinci | 222.72 |
| Scatter0.001 | 484.73 |
| Scatter1 | 462.58 |
| Scatter1/shadow_steps0 | 337.22 |
| Scatter1/stride16 | 397.61 |
| Scatter1/density0 | 159.28 |
| Authored tekrar | 187.62 |

rt_gas_cost_phasecache_20261022.json. Original materyal önceki ölçümle JSON
olarak aynı. Kontrol zamanı da oynuyor ve density/ray önce26.68 şimdi27.53;
bu nedenle iki app çalıştırması arasında %10 fark cache'in kesin etkisi gibi
sunulamaz. Kazanç kanıtı yok; driver compile süresi artışı ise ölçülmüş bir
regresyon gözlemidir. Materyal/sayaç restore true, sonframe101.

Bir sonraki belirleyici deney: yalnız phase-cache var/yok kontrollü kullanıcı
build A/B. Her iki SPIR-V hash/boyut/opcode sayısı, ilk pipeline total ve varsa
stage feedback, aynı sahne RT kontrol süreleri saklanmalı. Çalışan shader/cache
silinmemeli; app'i yeniden açıp aynı binary/SPIR-V ile warm pipeline süresi de
ayrı ölçülmeli. Phase-cache kodunu kaldırma veya -O kapatma bu partide yapılmadı.

## 2026-10-25 — Çoklu emitterli domain sahnesi: maliyet katmanları

Kare 28, rendered, `gas:Physics Domain 1` (authored scatter 2, density x10,
shadow_steps 8/stride 4). Aynı araç (`rt_probe_gas_cost_ipc.py --apply`) +
ayrı bir `visible=false` koşusu; materyal restore true, kare 28 korundu.
Wall ms/RT örnek; authored tekrarları 237–292 oynadı (±25 ms gürültü).

| Durum | ms | density/vr | SDF arbiter/vr | shadow/vr | probe/vr |
|---|---:|---:|---:|---:|---:|
| gaz gizli (`visible=false`) | 125 | – | – | – | – |
| scatter1 density0 | 160 | 22.8 | 31.1 | 3.1 | 1.60 |
| scatter0 | 247 | 24.5 | 31.1 | 6.6 | 1.59 |
| scatter1 | 266 | 25.8 | 32.4 | 15.4 | 1.50 |
| scatter1 shadow_steps0 | 259 | 25.9 | 32.4 | 7.4 | 1.50 |
| authored (scatter2) | 291 | 26.4 | 33.0 | 16.1 | 1.48 |

Okuma (22 Ekim sahnesinden FARKLI):
- Gazın toplam payı ~125→~290 ms. Içeriden ışığa march (scatter0→scatter1/2,
  shadow_steps0) bu sahnede yalnız ~20–45 ms, gürültü sınırında. 22 Ekim'deki
  "scatter 2.5x" bulgusu bu sahneye genellenemez.
- density0 bile +35 ms: kutu girişi, march iskeleti, solid probe ve **SDF
  arbiter**. Arbiter bu sahnede her volume ışınında açık (gate_open =
  volume_rays) ve ışın başına ~31 örnekle gaz density örneğinden FAZLA iş
  yapıyor (280M vs 224M). 22 Ekim'de 0'dı; SurfaceSDF+gaz örtüşmesi yeni aday.
- Görünür gazın kendisi +87 ms (density0→scatter0). scatter0'da da 6.6
  shadow/vr var; bunlar in-scatter değil, yüzey NEE gölgesinin gazı march
  etmesi (`computeVolumeShadowTransmittance`, density0'da `continue`).
  "Çevreden seken ışığın domaine tekrar girmesi" bu kalemdir; ayrı zaman
  ölçümü yok.
- ★ Ölçüm aracı açığı: stride16 satırının sayaçları scatter1 ile 15 haneye
  kadar AYNI. Enstrümanlı pencere tazelenmemiş olmalı; o satırın sayaçları
  geçersiz (ms değeri ayrı ölçüldüğü için geçerli).

Sonuç: ışık geçirgenlik hacmi doğru ilk adım, çünkü hem in-scatter hem yüzey
gölgesi kalemlerini tek okumaya indirir; ama bu sahnede beklenen kazanç
~2x değil, onlarca ms. SDF arbiter ayrı ve aynı büyüklükte bir kalem.

RT pipeline cache (aynı gün): disk cache SİLİNMEMİŞ. Açılışta 11 199 KB blob
aynı shader damgası ve driver UUID ile yüklendi, driver yine 168.4 s derledi,
kaydedilen blob birebir aynı boyutta (driver yeni girdi eklemedi). Yani
VkPipelineCache tasarlandığı gibi çalışıyor ama NVIDIA RT pipeline'ın bir
kısmını (muhtemelen link/stage-arası optimizasyon) her açılışta yeniden
yapıyor; soğuk derleme 313–495 s idi. Aynı gün ikinci açılışta da ~168 s
çıkıp çıkmadığı ölçülmedi.

## 2026-10-25 — Yalnız splat sahnesi: görünmeyen sis akışı karenin %80'i

Kullanıcı: domain içinde yalnız splat ve bir collider küp varken bile, kamera
domain'e yakınsa RT çok pahalı. Ölçüm aracı: `scripts/test/rt_domain_layer_cost_ipc.py`
(her hacim hedefini tek tek gizler / yoğunluğu 0 yapar; `vdb:` kimlikleri her
yayında değiştiği için yalnız kalıcı `matter:` hedeflerini kullanır).

| Durum (kare 28) | ms/örnek |
|---|---:|
| hepsi görünür | 161–189 |
| "Snow [Solid]" taşıyıcı Fog akışı gizli | **31** |
| aynı akış yoğunluk 0 (metadata ile etkisiz) | **31** |
| hepsi gizli | 23 |

Baseline sayaçları: yol başına 3.2 hacim ışını, 2.42 devir (handoff), hacim
ışını başına 1 solid probe (%76 isabet), 16.7 yoğunluk örneği, 18.8 boş atlama.
Görüntü A/B (64 spp): akış açık/kapalı farkı (ortalama |Δ| 0.0009) aynı ayarla
iki çekim arasındaki gürültüden (0.0034) küçük. Yani akış görüntüye bir şey
katmıyor.

Kök (iki ayrı şey):
1. **Rota:** `label_routes.spray = fog`. Sınıflandırıcı kümeden kopan 217
   parçacığı "spray" etiketliyor; 52'si katı kar ve Fog'a gidiyor
   (`fluid.inspect_fog_fields`: 384 dolu hücre, en yüksek yoğunluk 0.10).
   Kullanıcının "Hidden" seçimi GAZ fazının rotası; etiket rotasını kapsamıyor.
   Karar kullanıcıda (katı faz spray/mist rotasını izlemeli mi).
2. **Yapı:** Matter Fog akışı tam ızgara kutusuyla bağlanıyordu (`bindLiveVolume`).
   Yoğun gazda "dolu kutu" daraltması vardı, ama yalnız SSBO kırpma kutusunu
   daraltıyordu; TLAS kutusu `getWorldBounds()` = tam ızgara. NanoVDB topolojisi
   yalnız ÖRNEK atlatır. Işının kutuya girmesini, yani closest-hit + probe +
   boş march + devir izini engelleyemez. Kameraya yakın domain'de bu her yolda
   her sekmede tekrarlanıyor.

Düzeltme (yazıldı, derlenmedi): `VDBVolume::shrinkLiveBoundsToOccupied`. Yerleşim
aynı, yerel kutu dolu hücrelere daralıyor (+1 hücre trilinear payı, 8 hücrelik
bloğa dışa yuvarlama ki kutu her karede değişip TLAS güncellemesi istemesin).
TLAS kutusu, SSBO kırpma kutusu, raster aralığı ve OptiX paket kutusu aynı
kaynaktan türediği için hepsi birlikte daralıyor. Yalnız Matter Fog akışlarına
uygulandı. CPU Embree BVH kutuyu kurulumda okuduğu için kutu BÜYÜDÜĞÜNDE
`g_bvh_rebuild_pending` kalkıyor. Sonra (takip): eski domain sis yolu
(`FluidDomainFogVolume.cpp`, adımlı güncelleme) aynı daraltmayı almadı.

### Devam: ilk sahne, faz rotalarıyla ölçüm (Fog daraltması derlenmiş build)

Fog akışı kutuları daralmış geldi ("Snow [Solid]" [0.8–4.8]×[0.8–4.0]×[0–4.8],
gaz [0–5]×[0–2.4]×[0–5]). Ama içerik domain'in çoğunu kaplıyor, kazanç küçük.
Kullanıcının şikâyeti: "her şey splat / hidden yapınca maliyet düşmüyor".
`fluid.set_phase_output` ile ölçüldü (sonunda geri yüklendi):

| solid / liquid / gas | ms/örnek | kalan hacim slotu |
|---|---:|---|
| follow / sdf / fog (kayıtlı hali) | 253–303 | kar Fog, gaz, sıvı SDF |
| splat / **sdf** / hidden | **135** | yalnız sıvı SurfaceSDF, kutu [0,5]³ |
| splat / splat / hidden | **26** | yok |
| splat / hidden / hidden | 25 | yok |
| hidden / hidden / hidden | 25 | yok |
| splat / splat / fog | 262 | gaz |
| hidden / hidden / fog | 84 | gaz |

- Rotalar maliyeti kapatıyor, ama ÜÇ faz için de seçilince. Kayıtlı sahnede
  sıvı = sdf: eriyen kar (9 301 parsel) bir SurfaceSDF hacmi yayınlıyor ve kutusu
  tüm domain. Katı splat + gaz hidden durumunda kalan 111 ms'nin tamamı bu.
  Kullanıcının "hidden maliyeti kapatmıyor" gözlemi buydu.
- Splat'lar bir gaz kutusunun içindeyse gazın maliyeti ~3 katına çıkıyor
  (84 → 262): her splat sekmesi kutunun içinden başlıyor, probe + devir izi ödüyor.
- ★ Önceki "hepsi gizli = 46 ms" yanıltıcıydı: `visible=false` karı Fog'a
  yönlendirilmiş halde gizliyordu, yani splat de çizilmiyordu.

Düzeltme (yazıldı, derlenmedi): SurfaceSDF / host gaz yayın yolunda
(`scene_data.h`, yükleme bloğu) dolu kutu yüklemede hesaplanıyor (köpük kanalı
da dolu sayılıyor), `VDBVolume::live_occupied_*` üzerinde saklanıyor ve her
`bindLiveVolume`'dan sonra `applyLiveOccupiedBounds()` ile uygulanıyor.
(Yükleme imza kapılı, bağlama her karede ve kutuyu tam ızgaraya sıfırlıyor.)
Canlı GPU gazında uygulanmıyor: o yol cihazdaki yoğun tamponu okur, host
kopyasından türetilen kutu canlı dumanı kırpabilir.

### Canlı doğrulama: SurfaceSDF kutu daraltması (derlenmiş build)

| solid / liquid / gas | önce | sonra |
|---|---:|---:|
| follow / sdf / fog | 253–303 | 301 |
| splat / sdf / hidden | 135 | **77** |
| splat / splat / hidden | 26 | 31 |
| hidden / hidden / hidden | 25 | 31 |

SDF slotu [0,5]³ → [0,5]×[0,2.4]×[0,5]. x/z tam kalıyor, çünkü erime suyu küpü
sarıyor ve zeminde düzensiz birikintilerle domain zemininin çoğuna yayılıyor.
Görüntüde kırpılma ya da düz kesik yok (yalnız su rotası, 48 spp, göz kontrolü).
Kalan ~46 ms suyun gerçek yüzeyi artı tek kutunun üst kısmındaki hava: ince ve
geniş bir birikinti, tek AABB'nin en kötü durumu. Sonraki yapısal adım (yazılmadı):
dolu bloklara bölünmüş çoklu AABB (BLAS'ta blok başına bir kutu). Tam kare
(follow / sdf / fog) değişmedi, çünkü kar Fog'u ve buhar gerçekten domain'i
kaplıyor.

Kullanıcı kabulü (aynı gün): buhar Hidden maliyeti kapatıyor; yalnız splat
(katı/sıvı splat, gaz hidden) sahnesinde hiçbir hacim slotu yayınlanmıyor,
kare 28'de 25–27 ms/örnek (önceki şikâyette 135–161 ms).

## 2026-10-25 — Gaz kutusu içindeki splat'lar gazı ~3 kat pahalılaştırıyor: kök

Aynı sahne, kare 28, `fluid.set_phase_output` ile (geri yüklendi). Yol başına:

| | yalnız splat | yalnız gaz | splat + gaz |
|---|---:|---:|---:|
| ms/örnek | 31 | 94 | **311** |
| sekme izi (diffuse+specular) | 2.48 | 0.77 | 2.07 |
| hacim girişi | 0 | 1.74 | 2.53 |
| probe izi | 0 | 1.74 | 2.54 |
| probe isabeti = devir izi | 0 | 0.74 | **1.92** |
| teleport (boş kutu → yeniden iz) | 0 | 1.01 | 0.62 |
| yoğunluk örneği | 0 | 57 | 39 |
| ışık önbelleği isabet / kaçırma | – | 19.3 / 0.01 | 7.7 / 0.11 |

Gaz tek başına ucuz, çünkü ışınların çoğu kutuyu boş geçiyor. Splat kutunun
içindeyse HER sekme: kutu girişi + closest-hit → splat'ı bulan probe izi →
splat'a kadar march → aynı splat'ı gölgelemek için raygen'den devir izi →
gölgeleme. Yol başına iz ~2.5'ten ~7'ye çıkıyor; bu tam olarak gözlenen ~3 kat.
March/gölge örneği payı küçük; yüzey gölgeleri önbellekten okunuyor.

Probe ve devir AYNI yüzeyi iki kez buluyor: probe en yakın mesafeyi bulur
(`shadow_anyhit` sentinel'i, `NoOpaque` yüzünden her aday için any-hit), devir
aynı ışını orijinden tekrar izler. Seçenekler (karar bekliyor):
- **A (yapısal):** probe'u gölgeleyen iz yap. Hacim closest-hit'i katıyı ana
  yük tipinde (ikinci bir yük konumu) iç içe izler; yüzey closest-hit'i
  gölgelemeyi orada yapar; gaz katının mesafesine kadar march edilir; sonuç
  `T·yüzey + gaz` olarak birleşir. Devir izi ve raygen'deki ikinci geçiş kalkar
  (yol başına −1.9 iz). Bedel: özyineleme derinliği +1, büyük bir yük yığında,
  ıskalama durumunda miss shader'ının kısaltılması gerekir, ve çekirdek iz
  döngüsünün sekme bütçesi/AOV sözleşmesine dokunulur.
- **B (ucuz):** gaz devrine SDF devrinin kullandığı `plSolidHandoffWord` tMin
  ipucunu ver (yalnız tMin; tMax değil, çünkü alfa kesiği probe'un kabul ettiği
  yüzeyi reddedebilir). Devir izi kalır ama solidT'den önceki BVH düğümlerini
  atlar. Kazanç sınırlı.

### Seçenek A yazıldı (2026-10-25, derlenmedi; shader + C++): iç içe katı izi

Hacim closest-hit'i (düz march yolu) artık probe yerine katıyı ANA yük tipinde
(konum 2, `solidPayload`) iç içe izliyor. Yüzey closest-hit'i onu orada
gölgeliyor, dönüşte `volumeTraversalMask`'a `gl_HitTEXT` yazıyor
(`PL_NESTED_SOLID`, `primaryMeta` bit 23). Gaz o mesafeye kadar march ediliyor
ve `volumeAdoptNestedSolid` sonucu `gaz + T·yüzey`, verim `T·yüzey_verimi`,
saçılma ise yüzeyinki olarak döndürüyor. Raygen'deki devir izi bu yolda yok.

- Miss: bayrak varsa gökyüzü hesaplanmadan döner (maske 0 = katı yok).
- Fog karışımı ve random walk eski mesafe probe'unu kullanmaya devam ediyor:
  kendi devirlerini solidT'den yapıyorlar, iç içe gölgeleme orada boşa giderdi.
  SDF (izoyüzey) dalının probe'una dokunulmadı.
- Alfa kesiği iç içe izde kamera ışınındaki gibi any-hit'te çözülüyor.
  Eski probe `NoOpaque` ile her adayı kabul ediyordu.
- Özyineleme derinliği 2 → 3 (raygen → hacim CH → iç içe yüzey CH → gölge).
  Yığın dinamik değil; driver varsayılan yığını derinliğe göre büyütür.
- ★ Davranış farkı: eski yolda katının önündeki gaz geçişi BOUNCE_SPECULAR
  olarak bir sekme harcıyordu ("Specular includes the gas march continuation").
  Artık tek iz = yüzeyin sekmesi. Gaz içindeki yollar bir sekme daha derine
  gidebilir, görüntü bazı yerlerde hafifçe parlaklaşabilir. Bu hata değil,
  eski yolun fazladan sekme harcamasının kalkmasıdır.
- Sayaç: `gas_handoffs` artık "katı bulundu ve çözüldü" anlamında (iç içe
  gölgeleme dahil). Raygen'deki ikinci iz yok, bu yüzden `charged_specular`
  ve `free_passes` düşmeli.
- Kaynak kontrolü: `scripts/test/check_nested_solid_contracts.py` → OK.

### Seçenek A ölçüldü: gerileme, geri alındı (2026-10-25)

Aynı sahne, kare 28, `fluid.set_phase_output` (geri yüklendi). Kullanıcı
gözlemi "maliyet arttı gibi" doğru çıktı. Yol başına:

| | yalnız splat | yalnız gaz | splat + gaz |
|---|---:|---:|---:|
| ms/örnek, A'dan önce | 31 | 94 | 311 |
| ms/örnek, A ile | 24.7 | **265.3** | **405.6** |
| hacim girişi | 0 | 1.74 | 2.27 |
| probe = iç içe iz | 0 | 1.74 | 2.27 |
| katı bulundu | 0 | 0.73 | 1.74 |
| teleport | 0 | 1.01 | 0.53 |
| yoğunluk örneği | 0 | 58.4 | 34.5 |
| sekme izi (diffuse+specular) | 2.40 | 0.77 | 2.07 |

Okuma:
- **İş miktarı değişmedi, iş başına süre değişti.** gas_only'de giriş, probe,
  isabet, teleport ve yoğunluk örneği A'dan önceki değerlerle aynı. Kare süresi
  2.8 kat. Aynı sayıda işin pahalılaşması iz sayısıyla değil, shader'ın kendi
  hızıyla ilgili.
- **Artış yoğunluk örneğiyle orantılı.** gas_only +171 ms / 58 örnek ≈ 2.9 ms
  örnek başına. Aynı oran splat_gas'ın 34.5 örneğine ~+100 ms verir. Ölçülen
  +94 ms; yani devir izinin kalkması en fazla ~10 ms kazandırdı. Aday kök:
  march döngüsü yavaşladı. Hacim closest-hit'i içine ikinci bir 22 dword'lük
  ana yük konumu ve tam yüzey gölgelendiricisini çağıran bir iz eklemek,
  register baskısını ya da yığın kullanımını march döngüsünün tamamına yaydı.
  Kanıt dolaylı: sürücünün shader istatistiği okunmadı.
- **splat_only etkilenmedi** (hacim slotu yok, kod yolu hiç çalışmıyor). Bu da
  maliyetin özyineleme derinliği ya da pipeline geneli değil, hacim
  closest-hit'inin kendisi olduğunu destekliyor.
- **Teşhisin düzeltmesi:** önceki bölümdeki "yol başına ~7 iz, tam olarak ~3
  kat" okuması yanlıştı. Devir izi ucuzdu: kalkınca ~10 ms kazanıldı. Gaz
  içindeki splat'ların maliyeti, her sekmenin gaza YENİDEN girip march etmesi
  (yol başına 2.3–2.5 giriş). Hedef iz sayısı değil, ikincil sekmelerdeki
  march'ın fiyatı.

Geri alındı (shader + özyineleme 2; `check_nested_solid_contracts.py` silindi).
Ders: hacim closest-hit'ine yeni bir iz ya da büyük yük eklemek march'ın
fiyatını değiştirir. Ekleme ucuz görünse bile önce yalnız gaz durumunu ölç.

Sıradaki aday (karar bekliyor): ikincil sekmelerde (bounce ≥ 1) march'ı
ucuzlatmak. Seçenekler: daha seyrek adım, ya da in-scatter'ı ışık
önbelleğinden okuyup yalnız geçirgenliği march etmek.

Geri alma canlı doğrulandı (aynı gün, aynı sahne/kare): splat_only 27.4,
gas_only 97.0, splat_gas 294.5 ms/örnek. Sayaçlar A'dan önceki değerlerde
(giriş 1.75 / 2.29, yoğunluk örneği 58 / 33). Gerileme tamamen A'dandı.
