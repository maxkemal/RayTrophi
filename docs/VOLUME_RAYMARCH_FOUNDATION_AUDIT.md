# Volume Raymarch Temel İşleme Denetimi

Tarih: 2026-07-23

Kapsam: Legacy/baked VDB primary entegrasyonu, volume self-shadow ve
Vulkan RT / OptiX / CPU paritesi. Atmosphere aerial/LUT bu raporun dışında ve
önceden kararlaştırıldığı gibi en son ele alınacak.

## Sonuç

Mevcut sorun tek bir kalite katsayısı veya backend hatası değildir. Üç backend
aynı UI parametrelerini alsa da aynı raymarch algoritmasını uygulamıyor.
Özellikle geometrik ilerleme, optik-depth adaptasyonu, örnek bütçesi ve
stokastik yoğunluk elemesi birbirine bağlanmış durumda.

En kritik hata şudur:

1. `base_step`, ışının `Max Steps` içinde hacim çıkışına ulaşmasını sağlayacak
   şekilde hesaplanıyor.
2. Yoğun örnekte optical-depth adaptasyonu gerçek ilerlemeyi
   `base_step * 0.25` değerine kadar küçültüyor.
3. Küçük adım yine bir tam `Max Steps` örneği tüketiyor.
4. Döngü örnek bütçesi dolduğunda `t_exit` değerine ulaşmadan sonlanıyor.

Bu nedenle sonuç fiziksel entegrasyon değil, ışının hangi yoğunlukları önce
gördüğüne bağlı bir kırpma haline geliyor. Kamera yönü değişince hacim içindeki
ray uzunluğu ve yoğun voxel sırası değiştiği için OptiX'teki agresif,
bakışa-bağlı clipping doğrudan bu tasarımdan kaynaklanıyor.

## Mevcut ortak akış

Üç backend kabaca şu işlemleri yapıyor:

1. Dünya ışınını volume AABB ile kesiştirip `t_enter/t_exit` buluyor.
2. Kaynak voxel boyutu ve `Voxels / Sample` ile istenen step hesaplıyor.
3. Ray uzunluğu authored `Max Steps` değerini aşacaksa cover step üretiyor.
4. İlk örneği bir step içinde jitter ediyor.
5. Density sample, remap ve multiplier uyguluyor.
6. Düşük optical-depth örneklerini rastgele kabul veya reddediyor.
7. Kabul edilen yoğun örneklerde step'i optical-depth nedeniyle küçültüyor.
8. Primary örnek içinden bir veya daha fazla shadow march çalıştırıyor.
9. Transmittance küçükse erken çıkıyor.

Bu listenin 3, 6 ve 7. maddeleri birlikte güvenli değildir. Cover step yalnızca
ilerleme daha sonra küçültülmezse ışının tamamını kapsar. Stokastik red ise
yalnızca scattering olayını değil extinction ve emission katkısını da atıyor.

## Backend farkları

### Vulkan RT

- Upload aşamasında native voxel boyutu obje transformunun en küçük basis
  ölçeğiyle dünya uzayına çevriliyor.
- Primary march tamamen world-space `t` kullanıyor.
- NanoVDB hierarchy skip yalnızca legacy/baked density için aktif.
- Empty-tile skip şu anda primary `step` sayacını tüketiyor. Büyük boş tile
  fiziksel density sample maliyeti taşımadığı halde authored bütçeden düşüyor.
- Density için olasılıksal `scatter_keep` testi extinction, emission ve lighting
  entegrasyonunun tamamını atlayabiliyor.
- Yoğun bölgede `dt` dört kata kadar küçülüyor fakat budget yeniden
  paylaştırılmıyor. Düşük Max Steps değerlerinde hacim sonuna ulaşamama riski var.
- Her kabul edilen primary sample en fazla iki scene light shadow march ve ayrıca
  Nishita sun shadow march çalıştırabiliyor. Gerçek maliyet yaklaşık
  `primary samples * shadow samples * sampled lights` ölçeğinde.

### OptiX

- Upload aşamasında `step_size` native voxel boyutundan üretiliyor; world voxel
  daha sonra shader içinde inverse transformun yalnızca ilk basis uzunluğundan
  türetiliyor.
- Bu scalar ölçek anisotropic transformlarda ray yönünü temsil etmiyor.
- `base_step`, `max(world_voxel_size, ray_extent / 16)` ile tekrar üstten
  sınırlandırılıyor. Bu ek clamp UI kalite ve Max Steps semantiğini değiştiriyor.
- Primary sample başına yeni NanoVDB accessor ve trilinear sampler kuruluyor.
  Aynı durum shadow örneklerinde de tekrarlanıyor; accessor cache avantajı
  kaybediliyor.
- Vulkan'daki hierarchy skip OptiX'te yok.
- Optical-depth adaptasyonu geometrik ilerlemeyi küçültüp aynı step budgetini
  tüketiyor. Kamera yönüne bağlı clipping'in ana nedeni bu.
- `scatter_keep` rastgele testi extinction dahil tüm örneği atlıyor. RNG dizisi
  kamera ve ray sırasına bağlı olduğu için boşluklar bakışla değişebiliyor.
- Sun shadow yönü local uzaya çevriliyor fakat world shadow mesafesi primary
  raydan alınmış `dir_length` ile local mesafeye çevriliyor. Anisotropic scale
  altında light yönünün kendi transform uzunluğu kullanılmadığı için gölge
  adımlaması kamera yönüne bağlanabiliyor.

### CPU

- Voxel boyutu obje transformunun en küçük axis ölçeğiyle world-space'e
  çevriliyor. Bu kaliteyi korur fakat anisotropic veya çok küçültülmüş objelerde
  tüm ray yönlerini en sıkıştırılmış eksene göre örnekleyerek aşırı maliyet
  üretir.
- Primary loop Vulkan/OptiX gibi yoğun bölgede gerçek ilerlemeyi dört kata kadar
  küçültür ve aynı `Max Steps` bütçesini tüketir.
- `rand()` ile jitter ve `scatter_keep` uygulanır. Density kabul edilmezse
  extinction ve emission da uygulanmaz.
- Her kabul edilen primary örnekte sahnedeki bütün ışıklar dolaşılır.
- Her ışık için BVH shadow ray atılır; volume vurulursa ayrıca density shadow
  march yapılır. CPU maliyet düşüşünün ana çarpanı budur.
- CPU'da NanoVDB/OpenVDB active-tile skip primary ve shadow yollarında yoktur.

## Semantik olarak hatalı veya belirsiz noktalar

### `Max Steps`

Tek sayaç şu anda üç farklı işi temsil ediyor:

- Gerçek density sample sayısı
- Boş hierarchy traversal sayısı
- Optical-depth substep sayısı

Bunlar aynı bütçe değildir. UI'daki değer bu yüzden maliyeti ve coverage'ı aynı
anda güvenilir biçimde kontrol edemez.

### `Voxels / Sample`

Scalar minimum-axis voxel boyutu kaliteyi koruyan muhafazakâr bir yaklaşım olsa
da ekrandaki footprint'i ve ray yönünü bilmez. Çok küçültülmüş veya uzaktaki bir
VDB kaynak grid çözünürlüğünde render edilmeye devam eder.

Doğru dünya step'i ray yönüne bağlıdır. Anisotropic transform için tek bir
`min(scaleX, scaleY, scaleZ)` veya inverse transformun ilk satırı yeterli
değildir.

### Optical-depth adaptasyonu

Mevcut kod dense bölgede geometrik step'i küçültüyor. Fixed budget altında bu,
entegrasyon doğruluğunu artırmak yerine ray coverage'ını kaybettiriyor.

Optical-depth kontrolü iki güvenli biçimden biriyle yapılmalıdır:

- Geometrik segment değişmeden analitik Beer-Lambert entegrasyonu yapmak.
- Segmenti substep'lere bölmek fakat substep sayısını ayrı budget ile takip
  etmek ve primary rayın çıkışa ulaşmasını garanti etmek.

### Stokastik density elemesi

`scatter_keep` testi mevcut haliyle unbiased delta tracking değildir. Örnek
reddedildiğinde extinction, emission ve direct-light katkısı da yok sayılıyor.
Bu özellikle ince dumanı bakışa ve RNG'ye bağlı biçimde boşaltır.

Density cutoff deterministik olmalıdır. Stokastik seçim gerekiyorsa yalnızca
path-scattering olayı için, majorant ve ağırlık düzeltmesi olan ayrı bir
delta/ratio tracking algoritması kullanılmalıdır.

## Önerilen tek referans entegratör

Tüm backend'ler önce aynı sade referans algoritmaya dönmelidir:

1. Ray direction normalize edilir; bütün `t` değerleri world distance olur.
2. AABB kesişimi tek canonical world interval üretir:
   `L = max(0, t_exit - t_enter)`.
3. Ray yönüne bağlı world voxel footprint hesaplanır.
4. `requested_ds = voxel_footprint * voxels_per_sample`.
5. `N = min(ceil(L / requested_ds), MaxSteps)`.
6. `ds = L / N`; böylece tam interval kesin olarak kapsanır.
7. Jitter her frame/ray için stratified sample konumunu değiştirir fakat
   segment coverage'ını değiştirmez.
8. Her segmentte density deterministik örneklenir.
9. Extinction ve emission her segment için analitik olarak entegre edilir.
10. Density cutoff yalnızca açıkça tanımlı deterministik eşik olur.
11. Optical-depth substepping ilk parity aşamasında kapalı tutulur.
12. Shadow march ayrı ve kesin bir sample budget kullanır.
13. Sparse hierarchy traversal segmentleri atlar fakat density sample budgetini
    tüketmez; yalnızca ayrı traversal guard sayacını tüketir.

Bu referans önce trilinear, fixed-step ve deterministic olmalıdır. Üç backend
aynı görüntüye ulaştıktan sonra optimizasyonlar tek tek ve ölçümlü eklenmelidir.

## Performans modeli

Yaklaşık maliyet:

`primary density samples`
`+ primary lit samples * sampled lights * shadow density samples`
`+ temperature samples`
`+ graph evaluation`

Bu nedenle yalnızca primary `Max Steps` düşürmek yeterli değildir. Self-shadow
her primary örnekte çalıştığında maliyet çarpımsaldır.

İlk üretim hedefi:

- Primary density: kesin budget
- Shadow density: kesin ve primary'den bağımsız budget
- En fazla bir sun ve sınırlı sayıda sampled local light
- Legacy NanoVDB için persistent accessor
- Primary ve shadow için hierarchy traversal
- Graph evaluation yalnızca graph bağlı olduğunda
- Temperature grid yalnızca emission gerçekten kullandığında

## Düzeltme sırası

### Aşama A — parity baseline

1. Optical-depth kaynaklı geometrik step küçültmesini üç backend'te kapat.
2. `scatter_keep` ile extinction/emission örneği atlamayı kaldır.
3. `N` ve `ds = L / N` ile kesin full-interval coverage kur.
4. Sparse skip'in primary sample bütçesini tüketmesini engelle.
5. OptiX'in ek `ray_extent / 16` kalite clamp'ini kaldır.
6. Backend başına effective `L`, `ds`, primary sample ve shadow sample debug
   sayaçları ekle.

### Aşama B — transform parity

1. Direction-aware world voxel footprint için ortak CPU yardımcı fonksiyonu ve
   eşdeğer GPU formülü tanımla.
2. Primary ve light ray yönleri için ayrı local/world distance dönüşümü kullan.
3. Uniform, anisotropic, negatif ve çok küçük scale regression sahneleri ekle.

### Aşama C — kontrollü optimizasyon

1. OptiX persistent NanoVDB accessor.
2. OptiX ve CPU empty-space traversal parity.
3. Shadow sample caching veya düşük frekanslı transmittance.
4. Ekran-space footprint/LOD; küçük ve uzak hacimlerde kaynak voxel
   çözünürlüğünü gereksiz yere korumama.
5. Graph-aware occupancy metadata.

### Aşama D — gelişmiş entegrasyon

Parity baseline sabitlendikten sonra optical-depth substeps, ratio tracking,
majorant grid ve filtered LOD ayrı özellikler olarak eklenebilir. Her özellik
Vulkan, OptiX ve CPU referans görüntüsüne karşı bağımsız kabul testi geçmelidir.

## Kabul matrisi

Her sahne aynı kamera ve seed ile Vulkan, OptiX ve CPU üzerinde karşılaştırılır:

- Uniform density cube
- İnce shell / ortası boş patlama
- Küçük ölçekli legacy VDB
- Anisotropic scale VDB
- Kamera volume dışında ve içinde
- Önden, yandan ve çapraz bakış
- Shadow kapalı
- Yalnız directional sun shadow
- Bir point light ve sun
- Constant emission ve temperature emission
- Graph kapalı legacy VDB ve graph-driven density

Önce transmittance/depth parity, sonra lighting parity değerlendirilmelidir.
Aerial perspective ve display LUT bu testlerde kapalı tutulmalıdır.

## Vulkan referans uygulaması — başlangıç

İlk parity baseline yalnız Vulkan RT üzerinde başlatıldı:

- Primary interval kesin sayıda eşit world-space segmente bölünüyor.
- Dense bölgede geometrik ilerlemeyi küçülten optical-depth step kaldırıldı.
- `scatter_keep` artık density/extinction/emission örneğini rastgele atmıyor;
  yalnızca deterministik ince-scatter ağırlığı olarak kalıyor.
- `Shadow Update Stride` self-shadow sonucunun kaç primary segment boyunca
  yeniden kullanılacağını açıkça kontrol ediyor.
- Shadow Steps tek shadow rayının iç çözünürlüğünü, Shadow Update Stride ise
  primary ray boyunca kaç shadow rayı açıldığını kontrol ediyor.
- Draft ve Medium stride 4, High stride 2, Ultra stride 1 kullanıyor.

OptiX ve CPU bu baseline kabul edilmeden aynı değişikliklere geçirilmeyecek.

### Sequence/UI Vulkan resource safety

`updateVDBVolumes()` now takes the same backend render mutex used by material
program updates. This prevents a sequence frame upload from rewriting a
host-visible NanoVDB buffer while an RT dispatch traverses its device address.
Binding-9 volume SSBO growth also waits for the queue before destroying the old
buffer. This remains a resource-lifetime safety improvement for sequence
playback.

The mutex alone does not prove that an already submitted RT dispatch has
finished. On a VDB content-version change, Vulkan now performs one queue-idle
per synchronization batch before overwriting any existing density or
temperature NanoVDB buffer. This is the correctness baseline for rapid manual
sequencer scrubbing. A per-frame ring/triple-buffer upload is the planned
follow-up for removing this stall without reintroducing device-address hazards.

The subsequently observed access violation inside
`vkCreateRayTracingPipelinesKHR` was not caused by the volume changes. Its root
cause was a hair pipeline descriptor-stage mismatch: `hair_shadow_anyhit`
accessed bindings 10/11 while those bindings did not include
`VK_SHADER_STAGE_ANY_HIT_BIT_KHR`. The hair fix is the authoritative resolution
for that pipeline-creation crash. Volume-side SPIR-V header/module-result checks
remain as defensive validation.
