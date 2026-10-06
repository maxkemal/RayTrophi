# C6 ilk doygunluk modeli — kaynak partisi, 2026-10-05

Kullanıcı derlemesi temel wet/pore/mixed GPU problarını geçti. Tam mekanik,
mekânsal ve kalıcılık kabulü açık. Tam pore-pressure projection yoktur.

## Kanonik veri ve çözüm
S = clamp(pore_water_mass_kg / pore_capacity_kg, 0, 1). Dry rest mass ve identity
aynı SoA slotunda kalır; eklenen su mevcut C5 transport mass yoluna girer.
MatterWetResponse dört float / 16 byte: friction multiplier, dilatancy
multiplier, additive capillary cohesion Pa, positive pore pressure Pa.
Core bu alanı bir kez kare başında türetip GPU'ya yollar. Tüm ortak substep'ler
aynı alanı kullanır; kare sonunda C5 exchange yayınlanır. Dolayısıyla yeni
absorbed water sonraki fizik karesinde etki eder; görünüm güncel sidecar'ı okur.
Bu split dt yakınsamasıyla ayrıca kabul edilmelidir.

- Friction multiplier = 1 + (wet_friction_scale - 1)*S.
- Dilatancy multiplier = 1 + (wet_dilatancy_scale - 1)*S.
- Capillary cohesion = capillary_cohesion_pa * 4*S*(1-S); authored thermal bond
  cohesion ve tensile strength'e eklenir. Damage/hardening mevcut constitutive
  mekanizmasında uygulanır. Ayrı bir hydraulic fracture modeli değildir.
- Pore pressure = pore_pressure_scale * rho_water * gravity_m_s2 * h * S^2.
- Drucker-Prager shear strength ve rebonding gate, max(compression-u, 0) kullanır.
  Stress tensor'un mean kısmı yeniden bir pressure PDE ile çözülmez.
- Grid inertia dry+pore mass; stress impulse -dt*dry_volume*(stress*grad W).
  dry_volume=rest_mass*mass_fraction/canonical dry density; pore water bunu büyütmez.
  Tag bulunmayan legacy stress volume 1600 kg/m3 fallback kullanır.

Bu katsayılar ampirik authoring modelidir. Hücre boyu h açıkça pressure head'e
etki eder; çözünürlükten bağımsız sonuç vaat edilmez. Water hydrostatic pressure,
contact/drag ve immersed buoyancy kabulü ayrı açık testlerdir. CPU wet strength
fallback yok; wet physics Closed Vulkan Matter mixed path'e yönlenir.

## Authoring ve doğrulama
Mevcut rt.fluid.set_pore_exchange(domain, **patch) ve IPC
fluid.set_pore_exchange aynı setMatterPoreExchange servisini çağırır; UI de bu
servisi kullanır. Settings pore_exchange.settings altında geri okunur.

| Alan | Varsayılan | Sonlu aralık |
|---|---:|---|
| wet_response_enabled | false | bool |
| wet_appearance_enabled | false | bool |
| wet_friction_scale | 0.6 | 0..1 |
| wet_dilatancy_scale | 0.25 | 0..1 |
| capillary_cohesion_pa | 250 | 0..100000 Pa |
| pore_pressure_scale | 1 | 0..10 |
| wet_color_scale | 0.55 | 0.05..1 |
| wet_roughness_scale | 0.65 | 0.05..1 |
| wet_appearance_full_saturation | 0.05 | 0.001..1 |

Unknown key, wrong type, nonfinite/out-of-range ve wet porosity edit tam patch'i
reddeder. Serializer eski belgeleri wet flags kapalı yükler. Cache v11 sidecar
formatı aynı; settings hash/policy revision eski bake'i geçersiz kılar.
Authoring simulation cache'i invalidates: kontrollü bilanço sırasında config
edit yapılmamalı; A/B testleri ayrı reset koşuları olmalıdır.

## Görünüm
Görünüm doluluğu A=clamp(S/wet_appearance_full_saturation,0,1).
Varsayılan %5 pore doluluğunda tam wet görünüm; 1 seçilirse eski tam-pore
tepki ölçeğine döner. Bu bir görünüm ayarıdır; fizik ve kg sürekli gerçek S'yi kullanır.
PrincipledBSDF granular splat source material ceil(7*A) bandına yönlenir. Bin 0
original dry material; 1..7 editable dry BSDF'den türetilmiş 7 snapshot'tır.
S=0 band 0; her S>0 en az band 1. Önceki round(7*S) seçimi %7.14 altındaki
ıslanmayı kuru materyalde bırakıyordu. Canlı sahnede maksimum S=%4.75 ve 2292
wet carrier olmasına rağmen tüm 8333 granular carrier band 0'da kaldı; bu kaynak
düzeltmesi kullanıcı C++ derlemesinde canlı doğrulandı. Shader değişmedi.
Upper-edge quantization A'yı en fazla 1/7 aşar. .55 color multiplier ile ilk
bant %6.43, son bant %45 daha koyudur. Artık düşük pore doluluğu yalnız ilk
banda sıkışmaz: S=%1 -> band 2; S=%4.75 -> band 7 (full threshold=%5).
appearance_quantization=normalized_saturation_upper_edge_v2 ve
appearance_full_saturation raporlanır; farklı
politikaların snapshot band sayıları birbirine restore kanıtı olarak karşılaştırılmaz.
RT A/B: 2924 wet/6909 dry carrier'lı paused sahnede yalnız band 1 materyali
geçici magenta yapıldı. RT'de ayrı wet küreler magenta oldu, kuru kum aynı kaldı;
renk geri yüklendi ve timeline control sabit kaldı. Per-sphere materyal aktarımı
PASS; düşük-S varsayılan %6.43 koyulaşmasının algılanabilirliği ayrı görünüm
kalitesi konusudur. Bu deney genel repose/cache/çözünürlük kabulünü kapatmaz.
Kullanıcı Sand1'i manuel koyulaştırınca RT'de doğru sonucu gözledi.
Bu gözlem sonrası full-wet sensitivity UI/Python/IPC, JSON/cache hash ve aynı
renderer/metrics helper'ına eklendi. Yeni kaynak partisi kullanıcı C++
derlemesini bekler; shader değişmedi. Türetilmiş Sand1 değişikliği yerine
Pore Water threshold ve saturated color/roughness authored ayarları kullanılır.
Color/roughness saturated multiplier'a doğrusal yaklaşır. Fizik continuous S,
görünüm 8 bant kullanır. Scene material source kalır; generated variants fizik
ve emitter bağlarını değiştirmez. Scene registry generation ve kuru materyal
snapshot değişimi variants'i yeniler; material/source ID'leri interpolasyon görmez.
Canonical particle ID/cache sidecar restore'dan aynı spatial band türetilir.
Raster uniform sphere impostor wet görünümde kullanılmaz; material kullanan
instance yolu seçilir. Bu ek render maliyeti kabul testinde ölçülmelidir.
Granular SDF/Fog wet blend bu ilk kapsamda yok; wet checkbox granular splat içindir.

## Kontroller ve kullanıcı kabulü
2026-10-05 kullanıcı derlemesi: boş sahnede wet/pore drainage/mixed GPU probları
PASS; 25 + 180 manuel adımda ikinci pencerede su +0.32 mg, kuru Sand sabit.
7 wet variant üretildi. Ham kayıt ve kalan matris: MATTER_C5_HANDOFF.md.
Non-build: check_matter_wet_contracts.py, check_matter_gpu_contracts.py,
check_matter_pore_contracts.py, check_matter_transfer_contracts.py,
check_matter_solver_stages.py, check_matter_pore_batch_math.py.
C++ regression (henüz çalıştırılmadı): matter_wet_response_test.cpp ve
matter_wet_palette_test.cpp;
model endpoints, volume independence, canonical identity/mass purity,
legacy Auto, invalid pore mass, transactional JSON, old defaults, hash revision;
dry/damp/saturated spatial material routing ve missing-source fallback.

Kullanıcı shader + Release build yapar; stress update/P2G ABI'ları 18/68 ve 9/52.
Yeni wet flags ile kaynak/kütle test sahnesi reset edilir. rt_test_matter_wet_ipc.py
sadece paused finite field/GPU publication kontrolüdür; repose veya spatial
appearance ispatı değildir. Ayrı dry/damp/saturated collapse/repose; dry control;
iki dt ve çözünürlük; wet save/load+bake/scrub ve renderer karşılaştırması gerekir.
