# Matter sıvı/gaz sparse tile grid

> **Genişletilmiş kapsam:** sıvı ve gaz tamamlanacak, sonra tek kullanıcı build
> ve sıralı IPC kabulü yapılacak. Ortak storage/transaction çekirdeği kaynakta;
> canonical runtime consumer migration hâlâ açık. Çalışma noktası:
> [MATTER_SPARSE_FULL_STORAGE_IMPLEMENTATION.md](MATTER_SPARSE_FULL_STORAGE_IMPLEMENTATION.md).

> **Durum:** AKTİF — 2026-10-09. S0, Vulkan pressure/RHS havuzları ve compact
> MAC P2G/G2P/FLIP aktarımı kaynakta. Projection/contact publication ve host MAC
> hâlâ dense. Son aktarım partisi için kullanıcı build/GPU kabulü açık.

## 2026-10-09 kaynak devamı: compact MAC transfer ve FLIP

`SparseMacTransferGpu.h/.cpp` sıvı Vulkan aktarımını ortak 8³/576-face tile
adreslemesine bağlar. Hız XYZ, fiziksel ağırlık XYZ ve FLIP baseline XYZ aynı
map/list'i paylaşır. `sim_sparse_mac_address.glsl` iç yüzün pozitif tile sahibi
ile son domain yüzünün son hücre sahibi kuralını tek yerde tutar; kırpılmış tile
ve padding yüzleri ikinci fiziksel durum değildir.

Topology GPU'daki canonical parcel pozisyonlarından tüm quadratic MAC desteğini
üretir. Mixed Matter için tüm parcel desteği konservatif olarak dahil edilir;
indexed scatter yalnız kendi modelini işler. Yeni host occupancy bitmap veya
grid/particle readback yok; yalnız dört byte tile count indirilir. Her P2G yeni
clear/mark/reset yapar: slot sırası veya eski FLIP sayfası sonraki alt adıma
taşınmaz. G2P aynı pozisyonlarda, advection'dan önce yapılır. Bu geçici transfer
topology'si pressure/travel/gas desteği yerine kullanılmaz.

Normal/indexed P2G mevcut APIC shader gövdesini kullanır. Fiziksel parcel kütlesi
ve mass gradient'in dense contact indeksi korunur. Normalize ardından GPU
publication bütün dense yüzleri yazar; eksik tile'da hız ve ağırlık sıfırlanır.
Pressure, viscosity, porous/solid sınırları ve contact dense yayınla çalışır.
G2P öncesinde gerçek projection/contact sonrası alan GPU'da aynı compact hız
sayfalarına alınır. APIC/PIC/FLIP gather mevcut ortak G2P gövdesini kullanır.

Pure-liquid baseline mevcut post-P2G/pre-boundary anında; mixed baseline mevcut
solid-face clamp sonrasında alınır. Viscosity/projection sonrası baseline olmaz.
Host-visible source overwrite'dan önce capture lifetime fence korunur. Capture
bütün sahip yüzleri alır: sıfır P2G ağırlıklı ama boundary/contact hızı olan yüz
atlanmaz. FLIP=0/PIC de compact post field'dan gather yapabilir.

Vulkan nonperiodic liquid ve mixed liquid lane desteklenir. CUDA, periodic,
gas ve granular stress P2G mevcut dense yolunu korur. Invalid layout, allocation,
shader dispatch veya authored budget hatasında compact P2G drained/released
edilir; dense P2G bütün alanı yeniden kurar. Snapshot capture hatası dense
snapshot'a döner. Compact post gather hatası mevcut caller recovery/held-step
politikasını kullanır. Shader indeks genişliği ve cihaz SSBO kapasitesi kontrol
edilir; yeni keyfi parcel/tile tavanı eklenmez. Pool kapasitesi yeniden kullanılır;
authored budget düşerse trim edilir.

UI, Python fluid stats ve IPC `fluid.step_stats` aynı core üzerinden şu alanları
verir: `transfer_sparse_used`, `flip_sparse_used`,
`transfer_sparse_active_tiles`, `transfer_sparse_allocated_tiles`,
`transfer_sparse_resident_bytes`, `transfer_sparse_status`.
Transfer bayrağı başarılı compact P2G'yi, FLIP bayrağı pozitif blend ile başarılı
compact GPU gather'ı gösterir. Byte sayısı gerçek 11 buffer capacity toplamıdır:
map/list ve dokuz float page bankı. Dense/CPU seçiminde kullanılan-pool sayaçları
sıfırdır; kullanılmayan retained kapasite bu sayaçların kapsamına girmez.

**Bu parti canonical solver MAC depolamasının tamamını sparse yapmaz.** Dense
host MAC, device projection/contact publication, weight ve dense FLIP fallback
scratch hâlâ ayrılır. Transfer havuzu ilave resident bellektir; pool byte sayısı
net VRAM tasarrufu değildir. Dense clear/publication maliyeti hâlâ vardır.
Sonraki S2 işi velocity/mask/solid/fraction/porous consumer'larını ortak sparse
adreslemesine taşıyıp dense bankları kaldırmak; S3 gaz ayrıca açıktır.

On bir yeni SPV: clear/mark/reset/normalize/publish/capture/gather 8/36,
P2G 7/36, indexed P2G 12/40, G2P 11/68, indexed G2P 13/72. Dense ve mevcut
indexed giriş ABI'leri değişmedi; ortak gövdeler yeniden derlenmelidir.
Yeni modül vcxproj/filters ve simulation shader batch'e kayıtlı. Matter core
bake revision 8. DEM shader/GPU kaynaklarına edit yapılmadı.

`check_sparse_mac_transfer_contracts.py` ve bağımsız CPU adres/APIC/FLIP oracle
`check_sparse_mac_transfer_math.py` PASS. Mevcut dispatch, pressure, viscosity,
Matter GPU/grain ve panel denetimleri PASS. Bunlar C++/GLSL derlemesi veya GPU
fizik kabulü değildir. Build/uygulama çalıştırılmadı.
Capability/descriptor freshness audit PASS (663 method). Ayrı
`verify_descriptor_claims.py` denetimi enum listesini string olarak okuyup
`AttributeError: 'list' object has no attribute 'split'` ile durdu; bu genel
prose denetimi tamamlanmış sayılmaz. Yeni field/API wiring source denetimi PASS.

Kullanıcı build sonrasında, boş/durmuş sahnede dış process'te sıralı:
`python scripts/test/rt_test_sparse_pressure_ipc.py --transfer` ve
`python scripts/test/rt_test_sparse_pressure_ipc.py --transfer --viscosity`.
Probe FLIP=.95 türetilmiş madde, tile sınırı fixture'ı, dense/sparse model
mass/momentum ve centroid/hız parity ile gerçek compact yol seçimini kontrol
eder. Son runner iki probe'u içerir. Canlı sonuç henüz yok; moving wall,
büyük 2D GPU ve cinematic maliyet kapıları açık.

## 2026-10-09 canlı kabul: basınç ve viscosity parity PASS

Kullanıcının build sonrası isteğiyle iki dış IPC probe sıralı koşuldu; uygulama
başlatılmadı, build yapılmadı. Başlangıç ve bitiş: boş domain listesi, playing=false,
script_driving=false. Geçici kaynak/domain/türetilmiş madde cleanup edildi.
Rapor: [matter_sparse_live_2026-10-09.json](matter_sparse_live_2026-10-09.json).

- Basınç: 40³ grid, 129 parcel; centroid farkı 1.386e-9 m, mean speed farkı
  2.772e-9 m/s. Model mass/momentum assertion'ları PASS. Sparse pressure 1 tile,
  13,364 byte pool capacity. Son step host wall dense 17.26 ms / sparse 35.95 ms.
- Viscosity: aynı grid, nu=0.02 m²/s, 251 parcel; tile düzlemlerini geçen fixture.
  Centroid farkı 5.937e-10 m, mean speed farkı 2.422e-8 m/s; mass/momentum PASS.
  Pressure 8 tile / 99,492 byte; viscosity RHS 8 tile / 56,300 byte.
  Son step host wall dense 22.67 ms / sparse 19.95 ms.

Bu tek küçük-fixture karşılaştırması FPS kabulü değildir. İlk testte sparse daha
yavaştı; ikinci testte daha hızlıydı. Native GPU kernel timestamps ölçülmedi.
Tam MAC/FLIP/gas sparse storage, >16.7M elemanlı 2D GPU kabulü, yüksek nu/dt
yakınsaması/moving wall ve cinematic maliyet kapıları açık kalır.

## Önceki devam: P2G/G2P/FLIP dispatch ölçek hazırlığı

İş sahipliği ayrı: diğer ajan sim_matter_grain.glsl / MatterGrainGpu.cpp ve DEM
Verlet/geçmiş/maliyet alanında; bu devam bu iki dosyayı değiştirmez. Bizde MAC
aktarımı, shared float copy/clear, pressure/viscosity ve ileride gas storage var.
Diğer ajan kendi bellek ve maliyet hedeflerini docs/dev altında ölçülebilir
bir tasarım notuyla başlatabilir; fizik kabulü aynı son test paketinde kalır.

FluidGpuDispatch.h + sim_dispatch.glsl ortak 256-thread 2D planı verir.
Vulkan P2G clear/scatter/normalize, normal ve indexed G2P, FLIP snapshot copy,
mixed MAC clear/copy/contact bu adreslemeye bağlı. Pressure/viscosity pool
dispatch'leri de dengeli rectangle kullanır. CUDA P2G/G2P mevcut launch planını
korur. CPU'ya yeni grid veya particle indirmesi eklenmedi. Mevcut FLIP copy
sonundaki ReBAR/host overwrite lifetime fence'i kaldırılmadı.

Statik maliyet hedefi: 65536 logical workgroup için 32768x2, padding sıfır;
65535x2 gibi neredeyse iki kat launch yok. Genel padding rows-1'den fazla olmaz.
Shader padded GROUP'u çarpımdan önce eler; UINT_MAX yakınında lane'in başa
sararak ilk particle/face'i ikinci kez işlemesi engellenir. Bu bir partikül veya
VRAM tavanı değildir. Field/shader ABI ve backend capacity anlamları aynı kalır;
P2G/G2P signed count'a sığmayan girdi sessiz truncate edilmez. Diğer particle
stages/word offsets için bütün cinematic ABI denetimi hâlâ ayrı kabul kapısıdır.

Bu partinin bellek hedefi ek grid/particle depolaması olmaması, maliyet hedefi
logical dispatch kapsamını koruyup rectangle padding'i sınırlandırmaktır.
VRAM/FPS tasarrufu ölçülmedi. Canonical velocity ve FLIP snapshot hâlâ dense;
bir sonraki source adımı bunları ortak sparse MAC adreslemesine taşımaktır.
Bu hazırlık sparse MAC depolamasının tamamlandığı anlamına gelmez.

check_fluid_gpu_dispatch_contracts.py ve mevcut sparse/Matter/grain/panel kaynak
denetimleri PASS. fluid_gpu_dispatch_test.cpp kullanıcı C++ testinde 65535
sınırını, ilk ikinci satırı ve UINT_MAX kapsamını kontrol eder; henüz derlenmedi.
Core revision 7. Canlı uygulama/test ve shader/C++ derleme çalıştırılmadı.

## Son devam: viscosity MAC RHS tile pool

SparseViscosityGpu.cpp normal Vulkan liquid ve mixed Matter sıvı hattına bağlandı.
Üç tam scratch2 MAC kopyası, implicit viscosity başlangıç hızı için artık gerekli
yüz tile'larına ayrılır. Velocity'nin canonical üç dizisi hâlâ dense'dir; bu
aşama P2G/FLIP/pressure gradient veya gas MacCormack depolamasını taşımadı.
Gas kanalları olan grid'in MacCormack scratch2 bankları korunur; liquid dense
referansın RHS bankları yalnız gerektiğinde ayrılır.

Dense 11/52 ve sparse 13/68 girişleri aynı sim_fluid_viscosity.glsl gövdesini
kullanır. classify/relax, simetrik variable nu, hareketli solid velocity,
wall slip ve stress-free air denklemi tek yerde kalır. Sparse MARK her fluid
hücrenin kendi tile'ını ve pozitif MAC yüzlerinin sahip tile'larını dahil eder.
İç tile'ın fazladan yüz satırı ikinci sahip olmaz; domain son yüzü son hücrenin
tile'ında tutulur. Her fiziksel fluid yüz tek RBGS invocation'ında yazılır.
Tile sınırı yüzü kaybolmaz ve aynı parity içinde yeni yazma yarışı oluşmaz.
Periodic mevcut dense yolunda kalır; fiziği kapatılarak kabul aranmaz.

Dört yeni shader: clear/mark/capture/sweep, ABI 13 storage / 68 push byte.
Dense viscosity shader da ortak gövde için yeniden derlenmelidir. Başlangıç
hızı GPU'da capture edilir; bir uint tile sayısı dışında bu aşama yeni grid
veya particle indirmez. Lookup hâlâ domain tile sayısı kadar; sınıflandırma
hücre maskesini tarar. Pool yeniden kullanılır, gerçek capacity raporlanır ve
authored budget küçülürse trim edilir. Keyfi particle/tile sınırı eklenmedi.

UI, script ve IPC viscosity_sparse_used/active_tiles/allocated_tiles/resident_bytes
ile yalnız bu RHS pool'unu gösterir. Dense/CPU/no-op sonucu sparse diye raporlanmaz.
Core bake revision 6. Sparse viscosity ABI/shared-body, Matter/grain, panel ve
source yapı kontrolleri PASS; C++/shader veya canlı fizik testi çalıştırılmadı.

Son runner'a `rt_test_sparse_pressure_ipc.py --viscosity` eklendi. Geçici Water
türevinin nu=0.02 m²/s alanı, Water/Soil fixture'ında dense/sparse reset-parity,
mass/momentum ve bellek karşılaştırmasına girer; fixture üç tile düzlemini geçer.
classify/relax/nu/solid-velocity fonksiyonları HEAD kaynaklarıyla byte eşit
olarak denetlendi. Bu kaynak denetimi runtime fizik kabulü değildir.
Canlı PASS henüz yoktur;
moving wall, yüksek nu/dt yakınsaması ve cinematic native maliyet kapıları açık.

## Son kaynak devamı: Vulkan tile basınç çözümü

SparsePressureGpu.cpp artık FluidGpuPressure yoluna bağlı. Vulkan, sparse açık,
GFM kapalı ve periodic olmayan mevcut basınç denkleminde beş dense CG scratch
ve dense partial/scalar ayırımı kaldırılır; servis gerçek maskeden GPU tile listesi
çıkarır. Altı compact float alan (basınç/r/z/search/As/diag), double partial/scalars
kalıcı pool'dadır. Map/list domain tile sayısına göre ayrılır: hâlâ yoğun tile
lookup vardır, fakat hücre sayısı boyutunda yeni host occupancy bitmap yoktur.

Yeni on shader ABI 16 SSBO / 80 byte. Tile sayısı için dört byte, yakınsama için
sekiz iterasyonda 56 byte indirilir; bu aşama particle/grid download eklemez.
İlk sınıflandırma ve eski basıncın publication clear'i hâlâ domain hücrelerini
tarar. P2G, velocity/mask/solid fields ve yayınlanan pressure dense
kalır. Dolayısıyla bu **tam sparse solver belleği** veya O(active-only) kare
maliyeti değildir. Dense publication field eski G2P/contact kullanıcılarına
mevcut indeks sözleşmesini verir; CG buna göre sparse alanla çözer.

Kesirli yüz ağırlıkları, moving-solid/porous divergence ve density correction
önceki matrisle eşleşir. GFM ve periodic fizik seçenekleri kapatılmaz; mevcut
referans/fallback yolu korunur. Mixed Matter'ın mevcut liquid modeli zaten
GFM kullanmadığı için bu yolu kullanabilir; default GFM pure-liquid için GPU
sparse basınç henüz desteklenmiyor. Gaz basıncı bu servise bağlanmadı.

Pool capacity byte muhasebesi gerçek buffer boyutlarından gelir. Yazılmış
working budget uygulanır; budget sıfırsa yeni keyfi limit yoktur. Budget
küçülürse oversized scratch kapasitesi trim edilir. Parent mixed çalışma seti
kontrolü ayrı ve konservatif kalır; yalnızca bu pool için kontrol total scene
bütçesinin yeni bir ölçümü diye sunulmaz. Buffer release/backend değişimi
mevcut lifetime hizmetiyle uyumludur. Büyük dispatch 2D lane/reduction indeksleri
kullanır; eski gradient/divergence ve pencere reduction yardımcıları da buna
uyarlanır. Son GPU/kernel maliyeti henüz ölçülmedi.

UI, `fluid.step_stats` IPC ve Python aynı core stats üzerinden
pressure_sparse_used/active_tiles/allocated_tiles/resident_bytes verir. Bu
sayılar yalnızca pressure pool'udur. Kullanılmayan yol sıfır raporlanır.

`check_sparse_pressure_contracts.py`, mevcut Matter/grain/panel kontratları ve
capability/descriptor denetimi PASS. C++/shader build veya canlı test yapılmadı.
`rt_test_sparse_pressure_ipc.py` aynı Water/Soil rig'ini resetleyerek dense/sparse
centroid/hız, model mass/momentum ve allocation karşılaştırır; final acceptance
runner'a eklendi. Henüz canlı PASS değildir. Core authoring bake revision 5.

## İş paylaşımı ve sıra

DEM sayaçları → Verlet/geçmiş → GPU sıralama/indirme → uyuyan kümeler diğer
ajanda. Bu iş sıvı/gaz sparse tile grid adımıdır. Sonraki B10 MPM gövde/DEM
yüzey bunun yerine geçmez. Diğer ajanın GPU kapasite ve boundary maliyet
değişiklikleri korunur; gizli partikül/bellek tavanı eklenmez.

## Mevcut yolun gerçek durumu

FluidGrid::resize tüm hücre/skaler/MAC dizilerini ayırıyor. ActiveTile listesi
ve FluidActiveWindow/FluidActivePressure yalnızca işlem kapsamını daraltıyor;
yoğun depolamayı kaldırmıyor. MatterPhaseConfig::syncGrid de bu resize yolunu
kullanıyor. Host SparseTileGrid bunların yerine henüz bağlanmadı.
`use_sparse_tiles` artık uygun Vulkan basınç ve viscosity RHS pool'larını seçer;
tam FluidGrid/MAC/gas belleğinin sparse olduğu anlamına gelmez.

## S0: ortak depolama temeli

`Fluid/SparseTileGrid.h` yoğun grid'den bağımsızdır. Topology yalnızca çağıranın
verdiği yarı açık hücre kutularını 8³ tile'lara dönüştürür; domain hacmi boyutunda
bitmap/reserve yoktur. Tile sırası deterministiktir. Field'lar aynı immutable
topology'yi paylaşır; her kanalın ayrı topology kopyası yoktur.

Hücre alanı tile başına 512 float, MAC bileşeni 576 float ayırır. İç yüzün sahibi
pozitif taraftaki tile; domain son yüzünün sahibi son hücrenin tile'ıdır. Fazladan
yüz satırı padding'dir, ikinci fiziksel durum değildir. Kırpılmış son tile ve tam
8 hücrede biten domain bu kurala dahildir. Background kanal başına açık verilir:
yoğunluk için sıfır, sıcaklık için ambient gibi. Eksik tile'a yazmak hata verir.

Field::rebind koordinatla sayfa taşır; slot sırası değişmesi kimliği bozmaz.
Background dışı değeri olan bir sayfayı atmak reddedilir ve o alanın eski durumu
korunur. Bu, otomatik uyuma/LOD sistemi değildir. İleride tüm kanalların remap'i
tek solver transaction'ında yayınlanmalıdır; alan başına işlem yeterli değildir.

sweptSupport quadratic transfer + basınç komşusu + hız×dt seyahat halo'su üretir.
Bu yalnızca parcel destek tohumudur; gaz basınç domain'ini temsil etmez. Emitter,
hareketli collider, termal yayılım ve gözenekli reaction kapsamı ayrıca eklenir.
İndeks/host adres sınırları dışında keyfi kapasite tavanı yoktur. Bu temel host
size_t indekslidir; Vulkan tabloları eklenirken gerçek shader indeks genişliği ve
cihaz buffer kapasitesi ayrıca doğrulanacaktır.

## Sonraki kaynak kapıları

1. **S1 kalan device topology/pool:** pressure lane GPU tile üretimi/compact pool
   bağlı; tam MAC ve gaz için GPU'da occupied tile üretimi/sıralama ve resident
   slot map; dispatch 2D; yeniden kullanım, gerçek capacity byte muhasebesi ve
   explicit authored budget. Kare başına tüm grid/particle indirme yapılmaz.
   EnsureGridDomainComputeBuffers büyük ParticleSimulation.cpp içinde yeni
   uygulama almaz; yalnızca yeni modül çağrısı alır.
2. **S2 sıvı:** FluidGpuP2G/TransferStages/Pressure/Viscosity ve MatterGpuStep
   aynı sparse adresleme ile çalışır. Fluid mask, solid phi/face fractions,
   yoğunluk ve porous reaction birlikte taşınır. Basınç bağlantısı/component
   nullspace ve MAC extrapolation korunur; yoğun referansla eşitlik aranır.
3. **S3 gaz:** advection, combustion, heat, divergence/pressure, vorticity ve
   moving boundaries birlikte taşınır. Görünmeyen hava basınç taşıdığı için
   yalnızca smoke density threshold ile tile silinmez. Kapalı gaz domain'inde
   tam basınç kapsamı gerekebilir; bunu sparse başarı diye küçültmeyiz.
4. **S4 dış yüzey/kabul:** mevcut shared core üzerinden script/IPC/UI gerçek
   storage mode, active/allocated/retained tile ve resident byte raporlar.
   Hata/budget anlamı aynı olur. Diğer ajanın render/cache işine doğrudan edit
   yapılmaz; yeni alanın snapshot/export sözleşmesi onunla eşleştirilir.

## Tek son kabul paketine eklenecek kontroller

Önce `sparse_tile_grid_test.cpp` kullanıcı C++ test hedefinde: birbirinden uzak
iki pool için yalnızca iki sayfa; ambient background; remap ve atomik red;
kırpılmış grid üzerinde tüm MAC yüzleri dense adres referansıyla eşit; CFL halo.
Bu test solver doğruluğu veya GPU performans testi değildir.

Basınç probe'u aynı son runner'da, diğer ajanın testi bittikten ve kullanıcı son
shader/C++ build aldıktan sonra koşulur. S2/S3 sonrası aynı başlangıçla
dense/sparse: kütle, momentum, termal enerji,
divergence residual ve dt yakınsaması; tile sınırını geçen jet/pool, ayrı havuzlar,
kapalı gaz kutusu, hareketli duvar ve üç-owner porous sahne. Sayfa girip çıkarken
basınç sıçraması veya makul görünen kütle/ısı kaybı özellikle FAIL'dir.

Native GPU kernel süreleri, dispatch sayısı, upload/download byte, allocated ve
resident capacity ayrı raporlanır. Cinematic ölçek kapısı fizik terimlerini
kapatmadan yapılır. S0 için runtime bellek/FPS kazancı henüz iddia edilmez.
