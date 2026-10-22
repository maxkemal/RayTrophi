# RT gas optical traversal performance

> **Durum:** AKTİF — 2026-10-22; tek materyalli NanoVDB yolu, kullanıcı build/ölçümü bekliyor.

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
