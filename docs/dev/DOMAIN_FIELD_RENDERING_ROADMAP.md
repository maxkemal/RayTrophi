# Domain alanlarından tutarlı Splat, Fog ve SurfaceSDF

Tarih: 2026-11-09; kapsam güncellemesi: 2026-11-12. Durum: kaynak/doküman
incelemesi ve araştırmaya dayalı tasarım;
uygulama tamamlanmış değildir. Build, shader derlemesi ve canlı uygulama testi yapılmadı.

2026-11-12 uygulama başlangıcı: [P0/P1 iç altyapı ve açık entegrasyon işleri](DOMAIN_FIELD_IMPLEMENTATION.md).
Snapshot sözleşmesi ve parent'sız CPU transport/reseed çekirdeği aktif splat
köprüsüne bağlandı. Açık route `cpu_soa_reconstruction`; MPM için sıvı MAC
kullanılmaz. UI/script/IPC, iki proje serializer'ı, RAM checkpoint ve SimCache v21
kaynak bağlantıları mevcut. Mevcut sahne mesh paleti aynı örnekleri flat DNA
bankalarıyla tüketir. Kullanıcı C++ build'i/canlı kabul bekliyor. GPU transport,
projected lane alan yayını, yeni ağırlıklı palet/LOD hysteresis ve ortak Fog/SDF
adapter'ları açık işlerdir; tüm yol haritası tamamlandı sayılmaz.

## Karar ve kapsam

Görsel örneklerin parent taşıyıcıya bağı kaldırılacak. Örnekler geçerli domain
alanından üretilecek, kendi konumlarında örnekledikleri hızla taşınacak. Splat,
Fog ve SurfaceSDF aynı sürümlenmiş fiziksel yayın kaynağını kullanacak; birbirinin
görüntüleme çıktısını fiziksel otorite kabul etmeyecek.

Bu karar APIC/MPM fiziksel taşıyıcılarını kaldırmaz. Fiziksel partikül SoA'sı ile
grid, hibrit çözücünün farklı ve gerekli durumlarını taşır. Ortak yayın katmanı
bunlara salt okunur erişim sağlar. Yeni görsel örnekler çözücüye kütle, momentum,
basınç, sıcaklık veya kuvvet yazmaz.

Tek bir evrensel endüstri standardı algoritması yoktur. Aşağıdaki mimari,
yerleşik hibrit çözüm, alanla adveksiyon, sparse volume ve yüzey reconstruction
uygulamalarının projeye uyarlanmış mühendislik önerisidir. Sayısal eşikler bu
dokümanın kabul hedefleridir; literatür garantisi değildir.

## Mevcut kaynak bulguları

| Kaynak | Bulgular ve geçiş etkisi |
| --- | --- |
| `Fluid/FluidRenderProxy.h/.cpp` | Bilinçli render-only çoğaltma; child = parent + offset. Komşular dizide ±1/2/4/8/16; uzamsal arama değil. Offset parent çevresine kısıtlanıyor. |
| `shaders/fluid_sphere_proxy.vert` | Aynı desen çizim sırasında yeniden hesaplanıyor; hücre hızı, dt veya kalıcı örnek durumu kullanılmıyor. |
| `UI/ParticleBillboardBuilder.cpp`, `Backend/VulkanBackend_Spheres.cpp` | Raster ve Vulkan RT parent tabanlı geometrik çoğaltmayı kullanıyor. Ortak örnek tamponuna taşınmalı. |
| `Fluid/FluidLevelSet.h/.cpp`, `FluidSurfaceGpu.cpp` | Kanonik fiziksel partiküllerden yüzey reconstruction zaten var. Bu bilgi görsel örnekler veya kaba hücre ortalamasıyla değiştirilmemeli. |
| `Fluid/SurfaceParticleBins.h/.cpp` | Gerçek uzamsal CSR desteği mevcut; parent indeks penceresinin yerine kullanım uygunluğu değerlendirilmeli. |
| `Fluid/FluidDomainFogVolume.cpp`, `MatterFogFields.h` | Carrier ve gaz alanları, madde/faz/kaynak seçimi mevcut; ortak snapshot'a adapter ile bağlanmalı. |
| `FluidGrid.h`, `SparseGridStorage.h` | MAC yüz hızları, cell-centered skalerler ve sparse snapshot/background sözleşmeleri mevcut. Render için ikinci çözücü kurulmaz. |

İlgili mevcut yön belgeleri:

- [Sparse sıvı geçişi](MATTER_SPARSE_S1_SIVI_GPU.md),
  [storage sözleşmesi](MATTER_SPARSE_FULL_STORAGE_IMPLEMENTATION.md).
- [Anisotropik yüzeyin mevcut doğruluk sınırı](SDF_ELLIPSOID_FIELD_2026_10_10.md),
  [sparse yüzey desteği](SDF_SPARSE_SUPPORT_AND_ICE.md),
  [realtime SDF tüketicisi](REALTIME_SDF_SURFACE.md).
- [Bağımsız volume stream tasarımı](MATTER_VOLUME_STREAMS_PLAN_20261024.md),
  [sonraki gas-source uygulaması](GAS_SOURCE_VOLUME_STREAMS.md).

Eski tasarım notlarındaki açık işler sonraki uygulama belgeleriyle birlikte
okunmalı; tarihsel blocker güncel kodda hâlâ varmış gibi kabul edilmemeli.
Özellikle CPU/GPU residence ve sparse consumer kapsamı her adapter için kaynakta
yeniden doğrulanacak; storage sınıfının varlığı runtime migration kanıtı değildir.

## Araştırmanın tasarıma etkisi

1. [Disney APIC](https://www.disneyanimation.com/publications/the-affine-particle-in-cell-method/)
   parçacık-grid transferinde affine hız bilgisini korur. Çıkarımımız: fiziksel
   taşıyıcılar korunmalı; render örnekleri ayrı, pasif bir katman olmalı.
2. [SideFX POP Advect by Volumes](https://www.sidefx.com/docs/houdini/nodes/dop/popadvectbyvolumes.html)
   velocity-volume ile konum taşıma ve CFL kontrollü tracing sunar. Konumu ve
   hızı iki integratörle tekrar uygulama tehlikesini de açıklar. Önerimiz:
   tek adveksiyon sahibi, MAC örnekleme ve kontrollü alt adımlar.
3. [Yu ve Turk, anisotropik reconstruction](https://faculty.cc.gatech.edu/~turk/my_papers/particle_surfaces_tog.pdf)
   uzamsal komşuluk/PCA ile ince akış ve yüzey özelliklerini iyileştirir.
   Önerimiz: ortak veri yayını, fakat yüzey için fiziksel subcell bilgi korunur.
4. [OpenVDB overview](https://www.openvdb.org/documentation/doxygen/overview.html)
   inactive voxel'in boş olmak zorunda olmadığını gösterir. Önerimiz: sparse
   activity, fiziksel doluluk ve veri geçerliliği ayrı kavramlar olmalı.
5. [OpenVDB LevelSetUtil](https://www.openvdb.org/documentation/doxygen/LevelSetUtil_8h.html)
   proper distance-field işlemleri ile fog dönüşümünü ayırır. Mevcut implicit
   residual otomatik olarak gerçek mesafe kabul edilmemeli.
6. [PBRT transmittance](https://www.pbr-book.org/4ed/Volume_Scattering/Transmittance)
   optik sönümün yol boyunca integralini temel alır. Fog yoğunluğu ile sıvı hacim
   oranı veya görsel küre sayısı aynı nicelik değildir.

## Ortak veri yayını

Önerilen yeni modüller: `DomainFieldSnapshot`, `DomainFieldSampling`,
`DomainVisualSamples`, `DomainRenderFieldService`. İsimler öneridir; mevcut
eşdeğer servisler varsa genişletmek tercih edilir. 2000 satırı aşan kaynaklara
yalnız gerekli include/call/registration eklenir.

Snapshot kimliği domain UID, fiziksel zaman, step/substep, state generation,
topology generation ve transform revision içerir. Kanal manifesti location,
birim, background, geçerlilik, sahip lane/faz ve CPU/GPU residence bildirir.
Sürümler arası tamponlar karıştırılmaz. GPU publication fence/semaphore ile
tamamlandıktan sonra immutable handle yayınlanır; tüketiciler bitene kadar eski
buffer/topology yaşar. Başarısız yayın current diye sunulmaz.

Asgari alanlar:

| Alan | Sözleşme |
| --- | --- |
| Hız | Faz/lane'e uygun projected ve boundary-resolved MAC yüz alanı, uzunluk/saniye. Granüler ve sıvı hızları körlemesine ortalanmaz. |
| Doluluk / hacim | Fiziksel hacim ağırlıklı reconstruction; binary fluid mask veya P2G weight otomatik hacim oranı değildir. |
| Yoğunluk | Fiziksel mass density ve mevcut gaz render density ayrı anlam/birim etiketleri taşır. |
| Madde/faz/kaynak | Mevcut seçim ve provenance sözleşmeleri korunur; katkılar toplamı otoriter alanı aşmaz. |
| Sıcaklık | Mevcut fiziksel sıcaklık, açık birim dönüşümü; eksik veri için uydurma değer yok. |
| Solid phi / sınırlar | Hareketli duvar hızı, open/closed/periodic davranış ve örnekleme desteği. |
| Fiziksel SoA görünümü | Yüzey ve gerekirse materyal aktarımı için salt okunur subcell konum/hacim/kimlik. |

Hacim tahmini parçacık sayısından bağımsız olmalı: uygun faz için Vp = mp/rhop
ve normalize reconstruction ağırlıkları kullanılır. Eşit hacimli taşıyıcı
varsayımı yalnız gerçekten sağlandığı lane'de geçerlidir. Hacim oranını [0,1]'e
kırpmak tek başına korunum sağlamaz; taşan hacim raporlanır ve seçilen conservative
reconstruction yöntemiyle ele alınır. Binary doluluktan ince damla geri kazanılmaz.

## Sparse geçerlilik ve örnekleme

`ValidData`, `KnownBackground`, `Unavailable`, `OutsideDomain` ayrı sonuçlardır.
Allocated/active bayrağı tek başına madde varlığını belirlemez. Narrow-band SDF
içi inactive negatif değer taşıyabilir; bunu boş veya pozitif yapmak yasaktır.

Yeni örnek için geçerli madde desteği, uygun hız stencili ve sınır bilgisi gerekir.
KnownBackground sıfır yoğunluk geçerli boşluktur; Unavailable değildir. Eksik
velocity face sıfır hız diye sessizce yorumlanmaz. Ghost/halo veya açıkça tanımlı
dar bant extrapolation hazırlanır ve geçerliliği işaretlenir. Pressure topology
render görünürlüğünden türetilmez.

MAC bileşenleri kendi staggered koordinatlarında interpolate edilir. Bir hücrenin
ortalama hızını bütün noktalara vermek veya hücre merkezine çekmek yoktur.

## Parent'sız görsel örnekler

SoA örnek durumu: stable sample UID, konum, önceki konum, temsil edilen hacim,
stream/faz kimliği, doğum zamanı ve validity. Hücre indeksi geçici bin bilgisidir;
örnek başka hücreye geçince kimliğini kaybetmez. Parent UID/index alanı bulunmaz.

1. Fiziksel alt adım tamamlanınca mevcut örnekler tek kez taşınır. Varsayılan
   midpoint RK2; yerel CFL ile alt adım. Aynı kareyi tekrar çizmek taşıma yapmaz.
   Zaman içinde iki snapshot yoksa frozen-velocity RK2'nin zamansal doğruluk
   sınırı açıkça raporlanır; gerçek ikinci mertebe zaman iddiası yapılmaz.
2. Adveksiyon çözümlenmiş hızı kullanır. Yerçekimi/basınç/kuvvet ikinci kez
   uygulanmaz. Spray/foam gibi ayrı fizik isteyen örnekler ayrı sisteme aittir.
3. Taşıma sonrası spatial bin kurulur. Hedef sayı yaklaşık
   Ncell = alpha * h^3 / target_sample_volume ile hesaplanır. Kesirli kısım
   deterministik dağıtılır; toplam bütçe ve stream adaleti korunur.
4. Eksikler yalnız madde desteği içinde stratified/blue-noise benzeri dağılımla
   tamamlanır. Küresel kabuk veya parent çevresi yoktur. Fazla örnekler hysteresis
   ile azaltılır; taşınan örneklerin her kare yeniden dağıtılması yasaktır.
5. Örnek başına ağırlıklar güncel temsil edilen hacme normalize edilir. Görsel
   bütçe azalınca fiziksel hacim değişmez. Küre hacmi toplamının korunması tek
   başına birleşim geometrisinin veya piksel opaklığının korunduğu anlamına gelmez.
6. Solid kesişimi swept kontrol/projection ve duvar hızlarıyla ele alınır.
   Open çıkış emekli edilir, periodic wrap uygulanır. Unavailable alan örneği
   parent'a döndürmez; adveksiyon reddi/sınırlı bekletme ve diagnostic uygulanır.
7. Doğum/ölüm görünüm geçişleri ve hareket bulanıklığı yeni doğan/retired örnekler
   için tanımlanır; yoğunluğu iki kere sayan fade kullanılmaz.

## Yoğun gösterim ve fizik çözünürlüğü

Hedef, daha az fiziksel taşıyıcı ile çözülen uygun bir madde alanını daha çok
pasif görsel örnekle göstermektir. Görsel çoğaltma fizik çözünürlüğünü artırmaz:
tek tek tane teması, ince akış veya küçük ölçekli yığılma ayrıntısı üretmez.
Taşıyıcı sayısını azaltmanın doğruluğu ayrı fizik yakınsama testleriyle belirlenir;
yoğun bir görüntü fiziksel doğruluk kanıtı değildir. Bu plan çözücü azaltımını
otomatik yapmaz ve ölçüm öncesi toplam hızlanma vaat etmez.

İki açık temsil modu bulunur:

- **Fiziksel birebir:** özellikle gerçek tane teması/DEM için bir fiziksel tane
  bir görsel örnekle eşleşir; fiziksel konum, yarıçap ve varsa yönelim otoritedir.
  Ek görsel taneler fiziksel temasın yerine geçirilmez.
- **Alan tabanlı yoğun gösterim:** uygun continuum lane'de kalıcı, parent'sız
  görsel örnekler alanı izler. Fiziksel tane kimliği veya temas garantisi yoktur.
  Desteklenmeyen lane açıkça reddedilir; sessizce bu moda geçirilmez.

Fiziksel taşıyıcı bütçesi, toplam görsel örnek bütçesi ve ayrıntılı mesh örnek
bütçesi birbirinden ayrıdır. Render ayarı çözücü kapasitesini veya fiziksel
durumu değiştirmez. Kullanıcıya bu üç sayı ve etkin temsil modu gösterilir.

## Sahne objelerinden geometri paleti

Küre ve seçilmiş sahne mesh'leri aynı görsel örnek katmanının farklı geometri
temsilleridir. Objeler parent çevresinde child kümeleri oluşturmaz. Birden çok
kaynak obje ağırlıklı bir palete eklenebilir; örnek UID + seed üzerinden kararlı
bir palet girdisi seçilir. Dizi sırası, hücre geçişi veya kamera hareketi bu
seçimi değiştirmez. Palet düzenlemesindeki yeniden eşleme politikası tanımlanır.

Kaynak geometri yalnız kanonik flat `TriangleMesh` / DNA SoA yolundan okunur;
per-face `Triangle` koleksiyonları otorite değildir. Bir obje birden çok mesh
ve materyal parçası taşıyorsa bütünlüğü korunur. Aynı kaynak geometri paylaşılır;
örnek başına vertex/index kopyası oluşturulmaz. Objeyi palete almak onu fiziksel
çarpıştırıcıya dönüştürmez; sahnedeki mevcut collider rolü ayrıca yönetilir.

Palet girdisi kalıcı kaynak kimliği, geometri revision'ı, ağırlık, yerel pivot,
yön ekseni, boyut normalizasyonu ve materyal politikasını taşır. Dünya transform'u
ile kaynak yerel geometrisi ayrılır; kaynak ölçeği iki kez uygulanmaz. Kaynağın
sahnede ayrıca görünmesi açık bir ayardır. Kaynak edit/silme, undo ve save/reopen
referansları sürümlü handle ile yönetilir; eksik kaynak diagnostic verir ve
tanımlı küre temsiline geçer, çözücüye veya örnek kimliğine dokunmaz.

İlk kapsam statik mesh paletidir. Skinned/deforming kaynaklar desteklenmiyorsa
işlem açıkça reddedilir; eski deforme geometri sessizce çizilmez. Materyal seçimi
kaynak materyallerini koruma veya domain görünümünü uygulama olarak açıkça
tanımlanır; çok materyalli kaynakların alt-mesh eşlemesi korunur.

### Boyut ve yönelim

Örnek durumu geometri seçimi ve quaternion yönelimle genişletilir; önceki yönelim
motion blur/cache için saklanır. Mesh yönelimi her redraw'da hash ile yeniden
üretilmez. Birebir mod fizik yönelimini kullanır. Alan modunda lane'e uygun
velocity-gradient dönmesi veya açıkça seçilen sabit yönelim politikası kullanılır;
görsel açısal hareket çözücüye geri yazılmaz. Hız gradyanı unavailable ise
tanımlı bekletme/diagnostic uygulanır, rastgele dönme üretilmez.

Kapalı ve geçerli mesh için normalize edilmiş kaynak hacmi ile örneğin temsil
ettiği hacim ölçeği belirler. Açık/non-manifold veya sıfır hacimli mesh fiziksel
hacim hesabına sokulmaz; açık bir görsel boyut/bounds politikası gerekir.
Sanatçı boyut çarpanı fiziksel hacmi değiştirmez ve görünür doluluğu değiştirebilir.
Küre hacim toplamı, mesh hacim toplamı, birleşim hacmi ve piksel örtücülüğü aynı
ölçü değildir. Palet değişiminde görünür boyut ve doluluk ayrı doğrulanır.

### Hibrit LOD ve maliyet

Yakında paylaşılan gerçek mesh, uzakta analitik küre gibi ucuz temsil kullanılır.
Karar ekran boyutu, geometri karmaşıklığı ve bütçeye dayanır; hysteresis ile
kamera sınırında gidip gelme önlenir. Temsil değişiminde sample UID, merkez,
geometri seçimi ve yönelim korunur; kalıcı örnekler yeniden seed edilmez.
Küre proxy bounds'u kaynağı kapsamalı; şekil/siluet farkı kabulde ölçülmelidir.
Geçiş iki opak kopyayı üst üste çizerek yapılmaz; raster/RT gölge ve motion-blur
davranışları birlikte tanımlanır. Yakın örnek sayısı bütçeyi aşarsa kullanılan
temsile ait düşürme kararı ve sayıları raporlanır.

Palet başına BLAS/geometri paylaşımı hedeflenir; RT instance sayısı, TLAS
update/build süresi, raster draw sayısı, mesh üçgenleri, sample taşıma/reseed
süresi ve toplam VRAM ayrı ölçülür. Küre yolunun tek kompakt BLAS avantajı mesh
paletinde otomatik sağlanmış sayılmaz. Per-sample CPU mesh üretimi ve her kare
tam geometri upload'u normal güncelleme yolu olamaz. Raster, Vulkan RT ve
RayFusion destek/LOD farklarını açıkça raporlar; desteklenmeyen route gizlenmez.

## Üç tüketici

**Splat:** Raster, Vulkan RT ve RayFusion ortak sample buffer veya aynı sürümlü
host adapter'ı okur. Shader içinde parent'tan geometri üretimi kaldırılır.
Boyut/anisotropy support ve temsil edilen hacimden türetilir. Hash kimliği sample
UID'dir; dizi sıralaması ve hücre değişimi renk değiştirmez.

**Fog:** Gas alanı doğrudan mevcut otoriter gas/provenance alanından; carrier fog
fiziksel SoA/hacim reconstruction'dan hazırlanır. Görsel splat örnekleri fog kaynağı
olmaz. Optik extinction uzunluk^-1, yoğunluktan materyal katsayılarıyla türetilir;
mevcut artistik yoğunluk kanalları migrasyonda sessizce SI kabul edilmez. Ayrı
stream'ler mevcut VDB kimliklerini, fiziksel sıcaklığı ve malzemelerini korur.
SDF yüzeyi ile iç medium birlikte geçerlidir; aynı medium hem fog stream hem
surface interior olarak iki kez entegre edilmez. Imported VDB/cloud kapsamı korunur.

**SurfaceSDF:** Mevcut fiziksel partikül tabanlı anisotropik reconstruction
korunur; ortak snapshot/support/stream seçimini kullanır. Yalnız hücre ortalamasına
zorunlu geçiş yoktur. Implicit alan ve proper SDF ayrı metadata taşır. Proper SDF
gereken yolda sıfır yüzeyi korunarak narrow-band redistancing yapılır; smoothing
ve volume correction'ın yüzey/hacim etkisi ölçülür. Approximate implicit alanda
garantili sphere tracing varsayılmaz; conservative step/bracketing/refinement
ile kaçırılan kesişimler denetlenir. Nihai march güvenliği ayrı kabul kapısıdır.

## Lifecycle, maliyet ve dış API

Pause'ta redraw örnekleri değiştirmez. Rewind/seek ya snapshot/checkpoint'ten
örnek durumunu yükler ya deterministic replay yapar. Kalıcı RNG seed tek başına
geçmiş adveksiyon durumunu yeniden kurmaz. Reset/domain removal/undo/topology
değişimi ve GPU device loss için buffer sahipliği açık olmalı. Cache version
artışı ve eski cache için açık yeniden üretim politikası gerekir.

Bellek hesabı sample SoA kapasitesi, çift tamponlar, bins, halo, field kanalları,
RT kayıtları/AS ve snapshot'ın tuttuğu eski tamponları kapsar. Overflow ve bütçe
kontrolü allocation'dan önce yapılır. Görsel bütçe fiziksel adım kapasitesini
değiştirmez. CPU fallback ve readback varsa sebep/byte/time raporlanır.

Önerilen operasyonlar (henüz uygulanmadı):

- `domain.render_fields.describe/inspect`: kanal manifesti, sürümler, route,
  validity, memory ve publication hatası.
- `domain.visual_samples.get_settings/set_settings/get_stats`: etkinlik, hedef
  örnek hacmi veya yoğunluğu, max samples, seed, reseed hysteresis ve diagnostics.
- `domain.visual_samples.inspect`: sınırlandırılmış/paged örnek okuma; normal
  render güncellemesinde büyük readback yok.
- `domain.visual_samples.get_palette/set_palette`: sahne kaynakları, ağırlıklar,
  pivot/ölçek normalizasyonu ve materyal politikası; tüm palet atomik doğrulanır.
  Kaynak döngüsü, eksik/uygunsuz geometri, negatif veya toplamı sıfır ağırlık
  reddedilir; sıfır ağırlıklı tek girdi devre dışı kabul edilir.

`get_settings/set_settings` temsil modu, yönelim politikası, görsel hedef hacim,
max samples, max mesh samples ve LOD eşikleri/hysteresis'i kapsar. `get_stats`
fiziksel taşıyıcı, toplam örnek, mesh/küre sayıları, eksik kaynak, bütçe baskısı,
route ve AS/upload/bellek maliyetlerini bildirir. Bunlar önerilen API adlarıdır;
mevcut eşdeğerlerle nihai adlandırma P0'da netleştirilir. Palet kaynakları,
yönelim, LOD ayarları ve örnek durumu proje/cache şemasına dahil edilir.

UI, Python ve IPC aynı `DomainRenderFieldService` işlemlerini çağırır. Ayar yazımı
transactional; NaN/Inf, negatif değer, desteklenmeyen faz/route ve bütçe hatasında
kısmi değişiklik yok. `Unavailable` ile boş sonuç ayrılır. Eski
`visual_children` alanının yeni samples-per-cell anlamına sessiz dönüşümü yok;
versioned migration ve legacy readback açıklaması gerekir. Yeni kalıcı raf yok;
mevcut domain bağlamsal paneli kullanılır.

## Uygulama sırası ve kapanış kapıları

| Aşama | Teslim | Kapanış koşulu |
| --- | --- | --- |
| P0 | Runtime consumer/field envanteri, snapshot sözleşmesi, baseline | Aktif CPU/GPU/sparse yollar kaynakta eşlenir; birim ve otorite belirsizliği kalmaz. |
| P1 | Ortak snapshot, MAC sampling, halo/validity ve yayın servisi | Dense/sparse ve stale-buffer oracle'ları; aynı generation tüm tüketicilere görünür. |
| P2 | Parent'sız kalıcı sample SoA, transport/reseed/cache ve açık temsil modları | Analitik taşıma, pause/seek, sınır ve bütçe testleri; DEM birebir semantiği korunur; API/IPC aynı aşamada. |
| P3a | Raster/RT/RayFusion ortak sample ve küre tüketimi | Aynı snapshot için merkez/yarıçap/UID eşliği; alan modunda parent offset aktif yoldan çıkar. |
| P3b | Flat mesh kaynak paleti, boyut/yönelim ve materyal politikası | Çok materyalli obje bütünlüğü, kararlı seçim, kaynak lifecycle, UI/script/IPC ve cache kabulü. |
| P3c | Hibrit mesh/küre LOD ve ayrı render bütçeleri | Kamera geçişi, renderer eşliği ve gerçek cihaz AS/VRAM maliyeti ölçülür; fizik checksum'ı değişmez. |
| P4 | Fog ve SDF adapter'larının ortak snapshot'a taşınması | Stream sahipliği, ince yüzey ve optik regresyonlar; görsel sample bütçesinden bağımsızlık. |
| P5 | Migration, profiler, doküman ve canlı kabul | Kullanıcı build'i ve dış IPC matrisi tamam; açık route/fallback raporu. |

Her aşama gerekli UI/script/IPC, hata sözleşmesi ve testleriyle birlikte teslim
edilir. Ara aşama tamamlandı diye tüm runtime GPU-resident veya production-ready
ilan edilmez. İlk kod işi P0/P1 olmalı; fog renderer'ı ve çözücüyü aynı anda
yeniden yazmak bu geçişin önkoşulu değildir.

## Doğrulama ve kullanıcı build/test listesi

Yeni matematik testleri implementasyonu taklit etmek yerine analitik/oracle
sonuçlarını sınamalı. Static source check compiler veya görüntü kabulü değildir.

1. Uniform hız: x(t)=x0+u*t; sabit dizi sırası değişiminde sonuç aynı. Sıfır hızda
   redraw drift yok. Sabit gridde başlangıç hedefi konum hatası <= 1e-4 voxel.
2. Solid-body rotation ve affine shear: analitik yörünge ve dt küçülmesinde hata
   yakınsaması. MAC face interpolation dense/sparse eşliği <= 1e-5 relatif
   tolerans, sıfıra yakın değerler için ayrıca mutlak tolerans tanımlanır.
3. Yeni tile/halo, negatif koordinat, periodic wrap, moving wall, open çıkış,
   KnownBackground ve Unavailable ayrı oracle'larla sınanır.
4. Still pool, düşen jet, ince sheet, izole damla, ayrı iki madde ve karışık faz:
   parent kümeleri yok; fiziksel carrier sayısını değiştirmeden örnek bütçesi
   1x/4x değiştirildiğinde fog/SDF field checksum'ı aynı snapshot'ta aynı kalır.
5. Uniform fog slab: T=exp(-sigma_t*L), farklı voxel ve march step'lerinde
   yakınsama. Başlangıç hedefi referansa <= %1 transmittance hatası; sample sayısı
   bunu etkilemez. Çoklu stream toplamı ve kayıt sırası bağımsızlığı sınanır.
6. Sphere/plane SDF oracle: sıfır yüzeyi, işaret, narrow-band background ve
   |grad phi| ölçülür. Gradient değerlendirmesi medial axis/singular bölgelerden
   uzak yapılır. Thin sheet ve küçük damlada redistancing hacim kaybı raporlanır.
7. Raster/RT/RayFusion aynı yayın; fiziksel checksum önce/sonra aynı. Pause,
   tekrar redraw, save/reopen, rewind, cache seek, domain silme ve GPU buffer
   retirement testleri tamamlanır. Performans kabulü gerçek cihazda baseline
   üzerinden yapılır; ölçüm olmadan FPS/VRAM iyileşmesi iddia edilmez.
8. Geometri paleti: farklı pivot/ölçekli en az üç kaynak, çok materyalli obje,
   açık mesh, eksik/silinen kaynak, kaynak düzenleme, undo ve save/reopen sınanır.
   UID tabanlı seçim dizi permütasyonunda değişmez. Birebir tane yönelimi fiziksel
   veriyle eşleşir; alan modunda sabit hızda beklenmedik spin ve redraw drift yoktur.
9. Kamera yakın/uzak LOD sınırını geçer; örnek kimliği/merkezi değişmez,
   hysteresis çalışır, çift çizim oluşmaz. Aynı fizik snapshot'ında örnek ve mesh
   bütçesi ayrı ayrı değiştirilir; fizik/fog/SDF checksum'ları aynı kalır.
   Mesh/küre sayıları tüm backend'lerde açıklanır. Küre baseline'a karşı sample,
   draw, AS update/build, upload ve VRAM maliyeti raporlanır. Düşük taşıyıcı + yoğun
   gösterim ayrıca yüksek çözünürlüklü fizik referansıyla hareket/temas açısından
   karşılaştırılır; yalnız görüntü yoğunluğuyla çözücü azaltımı kabul edilmez.
10. Kullanıcı C++ build alır; Codex yalnız etkilenen shader varyantlarını derler,
   doğrular, source/runtime klasörlerine dağıtır ve bulk kaydını tamamlar.
   Kullanıcı sonra Studio'yu açar; IPC yoklamaları ayrı
   süreçte `scripts/test/rt_ipc.py` üzerinden çalışır. Mevcut
   `fluid.surface_parity` ve fog/stream readback'leri yeni testlerle birlikte
   kullanılır. Yeni IPC testlerinin dosya/komutları uygulanınca bu bölüme eklenir;
   henüz var olmayan test komutu çalışırmış gibi verilmez.

Canlı doğrulamada ekran görüntüleri yalnız ignored `docs/dev/.captures/` içinde
tutulur ve iş sonunda silinir. Codex build veya Studio launch çalıştırmaz.
