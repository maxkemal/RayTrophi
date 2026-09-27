# Virtual Particles ve Granüler Sanallaştırılmış Temsil — Yol Haritası

> **Durum:** ARŞİV — 2026-09-27'de söküldü. Kod (`GranularVirtual*`,
> `FluidRenderMode::VirtualParticles`) kaldırıldı; aşağıdaki plan referans olarak duruyor.

## Neden söküldü (2026-09-27)

Canlı sahnede ölçüldü (88k su parçacığı, granüler kapalı):

- **Sıvı için mod bir no-op'tu.** Köprüde `VirtualParticles`, `Particles` ile
  birebir aynı yolu izliyordu; sınıflandırıcı yalnızca `granular_enabled`
  iken devreye giriyordu. Kullanıcıya "adaptif" diye sunulan seçim sıvıda
  hiçbir şey değiştirmiyordu — ve panel onu seçince Splat Geometry yerine
  **SDF ayarlarını** açıyordu (kapı combo indeksi 1'e bakıyordu).
- **Kullanıcının gördüğü "yoğunken düz mesh / sürekli değişen render" bu
  kodun ürünü değildi.** Raster, akışkan havuzunu foliage scatter'ı gibi
  işaretliyordu; üçgen bütçesi aşılınca uzak splat'ler kart şerit proxy'sine
  düşüyor, Auto preset'in kare-süresi geri beslemesi sınırı her kare
  kaydırıyordu. Düzeltmesi `RasterMeshBuffer::scatterLodExempt`.
- **Solid/Material'da görünmeme de ondan bağımsızdı.** Raster, havuzun o anki
  dolu slotlarını kuruluşta sıkıştırıyor, sonradan dolan slotlar yalnızca
  transform senkronuna kalıyordu (ki var olmayan instance'ı güncelleyemez);
  gereken yapısal rebuild de `g_scene_geometry_generation` kapısında
  yutuluyordu (log: 99 kez `early-out: gen=39`). Düzeltme: geçici havuzların
  tüm slotları raster'da, boşlar `mask=0`.
- Granüler kısmı G2/G3'te kaldı; histerezis yoktu, yani sınıf titremesi
  garantiliydi. Yeniden ele alınırsa bu plan başlangıç noktasıdır; G5
  (procedural AABB + intersection) ayrı ve hâlâ geçerli bir RT hedefidir.

## Ürün hedefi

Kullanıcıya yalnız granüler malzemelere özgü ikinci bir parçacık sistemi
sunulmaz. Kanonik seçim **Virtual Particles**'dır: fluid, gas, foam ve genel
parçacık görünümleri için backend'in en hafif doğru temsili seçmesine izin
veren ortak render sözleşmesi. Granüler domain bu sözleşmenin özel bir
politikasıdır; desteklenmiş iç kütleyi transient heightfield/seyrek yüzeye
devreder, yalnız yüzey ve kopmuş taneleri aktif splat olarak tutar.

`Particles` modu her parçacık için açık ve tam geometri isteyen tanılama/yakın
plan yoludur. `Virtual Particles` ise kullanıcı niyetidir; billboard, vertex
pulling, procedural AABB veya kontrollü sphere fallback seçimi renderer'a
aittir. Böylece gelecek backend iyileştirmeleri sahne dosyasına yeni bir mod
veya granular'a özel UI seçeneği eklemez.

Kum, kuru/ıslak kar, çamur ve kopan taneler tek bir “milyon tane küre” çizim
yolu olmamalı. Hedef, maliyeti toplam simülasyon parçacığı sayısından mümkün
olduğu kadar ayırıp **görünen yüzey + gerçekten kopmuş/hareketli tane** sayısına
bağlamaktır.

Bu “maliyetsiz” bir sistem değildir. Doğru iddia şudur:

- içeride kalan sakin kütle tek tek geometri taşımaz;
- yüzey, malzemenin davranışına uygun daha seyrek bir temsil kullanır;
- havadaki/kopan taneler kompakt bir aktif listede kalır;
- aynı sınıflandırma Solid, Material Preview, RayFusion, Vulkan RT, OptiX ve
  Embree yollarına veri sağlar;
- sonraki aşamada fizik de aynı aktif/pasif ayrımıyla sanallaştırılır.

## Canlı sahne temel ölçümü

Kullanıcının açık sahnesindeki domain `wet_sand` yapıldı, parçacık çiziminde
kalındı ve kalıcı dolgu 1.000.000 parçacık sınırına ulaştı. Grid 100³, voxel
0,05; granüler Young modülü 250.000 ve gerekli/uygulanan alt adım sayısı 30.

İki timeline karesi istendi; ölçüm penceresinde bir gerçek solver adımı oluştu:

| Kalem | Ölçüm |
|---|---:|
| `sim.timeline.step` | 1.162,9 ms |
| P2G | 151,2 ms |
| G2P | 493,6 ms |
| Advect | 96,8 ms |
| Batch sonlandırma | 592,0 ms |
| GPU yükleme | 2.152.000.000 bayt |
| GPU indirme | 2.276.120.000 bayt |
| Dispatch | 603 |
| `render.fluid.splat_instances` | 46,6 ms / 2 çağrı (~23,3 ms/çağrı) |
| `loop.viewport_render` | 1,69 ms / 1 çağrı |
| Timeline frame yakalama | 81,5 ms |

Sonuç: hafif görünüm bugün yaklaşık 23 ms’lik splat köprüsünü ve ağır geometriyi
azaltır; fakat 1M tanede ana darboğaz 30 alt adımlı çözücü ve her alt adımda
tekrarlanan host/device trafiğidir. Bu yüzden proje iki paralel ama aynı veri
sözleşmesine bağlı eksende ilerlemelidir:

1. **Render sanallaştırması:** hemen uygulanır.
2. **Fizik sanallaştırması:** doğruluk ve korunum testlerinden sonra uygulanır.

## Temsil piramidi

### 1. Taşınan iç kütle

Gevşek kum ve kar yığını için 2,5B heightfield en ucuz ilk temsildir. Ancak
yükseklik “sütundaki en yüksek parçacık” değildir. Havada uçan tek tane bu
şekilde yığının parçası olur ve kendi kendini gizler.

Önce yoğun hücreler bulunur. Zemine veya gerçek collider hücresine bağlı yoğun
bileşen taşınan kütledir. Yalnız bu bileşenin sütun tepeleri heightfield’a girer.
Tünel, saçak ve katlanan yüzey gerektiğinde heightfield yerine seyrek 3B yüzey
seçilir.

### 2. Yüzey kabuğu

Taşınan kütlenin üst bandındaki taneler ayrı sınıftır. Yakın planda malzeme
detayı için deterministik örneklenmiş impostor/surfel olarak çizilebilir;
uzaklaştıkça yalnız yüzey kalır. Yüzey kabuğu bütçesi ekrana kaplanan alan ve
mesafeye göre belirlenir, toplam parçacık sayısına göre değil.

### 3. Kopmuş taneler

Destek bileşenine bağlı olmayan parçacıklar kompakt aktif buffer’a yazılır.
Bunlar sıçrayan, saçılan, yuvarlanan veya havadaki gerçek tanelerdir. Raster
yolunda vertex pulling + indirect draw; RT yolunda procedural AABB + intersection
shader hedeflenir. Bir parçacık başına yüksek üçgenli sphere mesh/BLAS instance
taşınmaz.

### 4. Kohezif yüzey

Islak kar, sıkışmış çamur veya kırılan kabuk heightfield’ın temsil edemediği
saçak/oyuklar üretir. Bu malzemeler için seyrek yoğunluk/SDF bloklarından yüzey
üretilir. Bu, sıvı SurfaceSDF’nin körlemesine yeniden kullanılması değildir;
granüler kopma, plastik deformasyon ve tane detayını koruyan ayrı bir yüzey
politikasıdır.

## Ortak veri sözleşmesi

İlk CPU referansı şu dosyalardadır:

- `source/include/Fluid/GranularVirtualRepresentation.h`
- `source/src/Physics/Fluid/GranularVirtualRepresentation.cpp`

Girdi her zaman kanonik düz SoA’dır: `FluidParticles::position` ve solver’ın
`FluidGrid` verisi. Per-face `Triangle` koleksiyonları kaynak veya otorite
değildir.

Çıktı:

- `HiddenBulk`: yüzey tarafından temsil edilen iç kütle;
- `SurfaceGrain`: üst kabuktaki görünür tane adayları;
- `DetachedGrain`: destek bileşeninden kopmuş aktif tane;
- geçerlilik maskeli XZ yükseklik alanı;
- yüzey ve kopmuş tane kompakt indeks listeleri;
- `measured` bayraklı sayımlar ve yapı süresi.

CPU ve GPU uygulamaları aynı sahnede eşdeğer sınıf üretmelidir. Renderer kendi
“aktif tane” tanımını icat edemez. Fizik sanallaştırması da bu sınıfları tüketir,
fakat onları render sonucundan geri okumaz.

## Backend politikası

| Backend/mod | İç kütle | Yüzey tanesi | Kopmuş tane |
|---|---|---|---|
| Solid / Matcap | Düz SoA yüzey mesh’i | GPU impostor | GPU impostor |
| Material Preview | Aynı mesh | Vertex pulling | Vertex pulling |
| RayFusion | Aynı mesh/BLAS refit | Procedural AABB | Procedural AABB |
| Vulkan RT | Aynı mesh/BLAS refit | Procedural AABB | Procedural AABB |
| OptiX | Aynı mesh güncellemesi | Özel primitive | Özel primitive |
| Embree CPU | Aynı mesh | Bütçeli sphere/point primitive | Bütçeli primitive |

Mesh shader zorunlu değildir. Mevcut vertex-pulling ve indirect draw altyapısı
önce kullanılır. Mesh shader daha sonra desteklenen GPU’larda ek hızlandırma
olabilir; ürün sözleşmesi veya tek çalışır yol olamaz.

## Aşamalar

### G0 — Ölçüm ve emniyet (tamamlandı)

- `render.fluid.splat_instances` ve solver sayaçları canlı sahnede ölçüldü.
- 1M wet-sand temel değeri kaydedildi.
- `scripts/test/rt_probe_granular_render_baseline.py` tekrar ölçümü yapar.
- `scripts/test/rt_setup_granular_baseline_ipc.py` sahneyi aynı temele getirir.
- Cache temizlendikten sonra sıfır parçacıkla 1M splat havuzunun raster listesinde
  kaldığı ölçüldü: 524.289 görünür instance, 44,6M üçgen ve 101,9 ms GPU karesi.
  Ölü slotların raster compaction’a girmemesi ve Solid redraw bayrağının
  kendini açık tutmaması düzeltildi; A/B probu
  `scripts/test/rt_probe_empty_fluid_pool_ipc.py` içindedir.
- 1M triangle-sphere ile Rendered moda geçiş yapılmaz: daha önce sürücü TDR’ı
  gözlendi. Limit yükseltmek çözüm değildir.

### G1 — CPU referans sınıflandırma (çekirdek eklendi)

- O(N + grid) yoğunluk ve destek bağlantısı.
- `kSolidCollider` destek sayılır; parçacıktan türetilen
  `kSolidSubstance` dairesel kanıt olarak kullanılmaz.
- Geçersiz/alan dışı parçacık güvenli biçimde `DetachedGrain` olur.
- Heightfield yalnız desteklenmiş sütunlardan kurulur; boş sütun ayrı maskedir.
- Yüzey bandı, kopmuş tane listesi ve ölçüm istatistikleri üretilir.

Sonraki doğrulama: sentetik düz yatak, tek havadaki tane, collider üstündeki
yığın, iki ayrı yığın, domain dışı tane ve boş domain testleri.

### G2 — Yaşam döngüsü ve ortak servis

- Domain başına transient `GranularVirtualRepresentation` saklanır.
- Simülasyon değişmediyse yeniden sınıflandırılmaz; generation/dirty sözleşmesi
  kullanılır.
- Düz `TriangleMesh`/DNA SoA yüzey mesh’i sabit topolojiyle yaratılır; yalnız
  `P/N` güncellenir ve mevcut geometri-dirty/BLAS-refit yolu kullanılır.
- Üretilmiş yüzey proje dosyasının kalıcı otoritesi olmaz; yüklemede sim
  durumundan yeniden kurulur.
- Domain/mod silme ve proje temizleme, sahiplik sırasını güvenli kapatır.

### G3 — Kullanıcı yüzeyi, scripting ve IPC (ilk dikey dilim eklendi)

Tek çekirdek servisin işlemleri:

- `render_mode = particles | virtual_particles | surface`;
- yüzey çözünürlüğü/LOD;
- surface-grain yoğunluğu ve bütçesi;
- detached-grain bütçesi;
- kohezif yüzey tercihi;
- debug görünümü: bulk/surface/detached.

`virtual`, `adaptive` ve `virtual_particles` aynı kanonik moda yazılır; okuma
her zaman `virtual_particles` döndürür. Aynı doğrulama ve hata metinleri UI,
Python scripting ve IPC’den çağrılır.
`fluid.get` en az şu telemetriyi döndürür:

- `granular_virtual_measured`;
- `granular_virtual_build_ms`;
- `granular_virtual_surface_columns`;
- `granular_virtual_hidden_bulk`;
- `granular_virtual_surface_grains`;
- `granular_virtual_detached_grains`;
- istenen ve gerçek render bütçeleri.

Bu yüzeyler birlikte teslim edilmeden özellik “tamamlandı” sayılmaz.

### G4 — Raster GPU yolu

- Sınıflandırma sonucu kompakt indeks buffer’ına yazılır.
- Yüzey ve kopmuş taneler mevcut Vulkan parçacık vertex-pulling yoluyla çizilir.
- Indirect draw sayacı GPU’da kalır; milyonluk liste CPU’da her kare dolaşılmaz.
- Yakın/uzak LOD ekran boyuna göre seçilir; görünmeyen/interior taneler draw’a
  girmez.
- Heightfield vertex buffer’ı kalıcıdır, topoloji yeniden oluşturulmaz.

### G5 — Ray tracing yolu

- Aktif taneler tek tek triangle sphere değildir.
- Kompakt tane AABB buffer’ı procedural geometry BLAS’ına girer; intersection
  shader gerçek küre/kapsül kesişimini çözer.
- Yüzey mesh’i ayrı BLAS’ta refit edilir.
- RayFusion ve Vulkan RT aynı aktif buffer’ı tüketir; OptiX/Embree karşılıkları
  aynı bütçe ve görünürlük semantiğini uygular.
- İlk RT karesi için süre bütçesi ve TDR koruması vardır. Bütçe aşıldığında
  sessiz kilitlenme yerine telemetri + kontrollü LOD düşüşü olur.

### G6 — Tam GPU sınıflandırma

- Occupancy, destek flood/propagation, sütun yüksekliği ve compact aşamaları
  compute’a taşınır.
- Host’a milyon parçacık indirmek yerine yalnız küçük telemetri ve gerekiyorsa
  yüzey grid’i iner.
- CPU referansı doğruluk oracle’ı olarak kalır.
- Mevcut başka host tüketicileri taşınmadan solver indirmesi kaldırılamaz;
  transfer kazancı sayaçlarla kanıtlanır.

### G7 — Kohezif seyrek 3B yüzey

- Heightfield’ın yapamadığı oyuk/saçak için aktif tile tabanlı yoğunluk veya SDF.
- Sadece yüzeye yakın bloklar polygonize edilir.
- Gevşek kum varsayılan olarak heightfield’da kalır; 3B yüzey otomatik olarak
  her granüler malzemeye yüklenmez.

### G8 — Fizik sanallaştırması

Amaç sakin iç kütleyi her alt adımda tam MPM parçacığı gibi dolaşmaktan
çıkarmaktır:

- sakin ve gömülü bölge continuum/grid özeti olarak tutulur;
- temas bandı, serbest yüzey, yüksek deformasyon ve kopan bölgeler parçacık
  çözünürlüğünde kalır;
- collider yaklaşınca yerel bölge deterministik biçimde parçacığa terfi eder;
- sakinleşen iç bölge tekrar özet temsile iner;
- terfi/indirme kütle, momentum, sıcaklık, plastik hacim, hasar ve malzeme
  etiketini korur;
- histerezis ve minimum yaşam süresi sınıf titremesini önler.

İlk fizik hedefi alt adım başına host/device batch sayısını ve G2P maliyetini
azaltmaktır. Young modülünü yapay biçimde düşürerek hız kazanmak fizik
sanallaştırması sayılmaz.

## TDR ve bellek güvenliği

- 1M mevcut üst sınır test sınırıdır; kullanıcı artırabilse bile
  `virtual_particles` RT yolu hazır olmadan Rendered/RayFusion için sınırsız sphere yolu
  açılmaz.
- Render bütçesi açık bir değerdir; gerçek kullanılan değer IPC’den okunur.
- Havuz kapasitesi, canlı tane sayısı ve çizilen tane sayısı ayrı ölçülür.
- GPU device-lost durumu render durumuna taşınır; “sample artmıyor” tek teşhis
  yolu olamaz.
- Proje temizleme sırasında simülasyon sistemleri, transient yüzey ve backend
  kaynaklarının sahiplik sırası test edilir.

## Kabul ölçütleri

1. Tek havadaki tane heightfield’a katılmaz ve kaybolmaz.
2. Karakter ayağı yığını ezer; iz yüzeyde, sıçrayan taneler aktif listede görünür.
3. `hidden_bulk > 0`; yüzey ile tüm splat kümesi yanlışlıkla üst üste çizilmez.
4. 1M parçacıkta render maliyeti toplam N yerine yüzey/aktif bütçeyle ölçeklenir.
5. Solid, Material Preview, RayFusion, Vulkan RT, OptiX ve Embree aynı sahne
   semantiğini korur; desteklenmeyen backend kontrollü fallback bildirir.
6. CPU ve GPU sınıflandırması örnek sahnelerde tolerans içinde eşleşir.
7. UI, Python ve IPC aynı ayarı yazar, aynı doğrulama hatasını verir ve aynı
   ölçümleri okur.
8. Cache/replay ve proje yeniden açılışında yüzey deterministik yeniden oluşur.
9. Fizik sanallaştırması açıldığında toplam kütle ve momentum sapması test
   toleransını aşmaz.

## Ayrı ama ilişkili işler

- Kinematik collider, bu sistemin temas girdisidir; sınıflandırıcı collider
  hücrelerini destek olarak tanır. Kemik proxy overlay ölçeği ayrı UX işidir.
- `GRANULAR_SIMULATION_ROADMAP.md` içindeki constitutive/MPM doğruluğu korunur;
  bu yol haritası onun görünüm ve ölçeklenme katmanıdır.
- Sıvı SurfaceSDF, gevşek granüler yığının varsayılan görünümü değildir.
- Timeline geri sarma/Fill Domain yaşam döngüsü hataları bu temsilden bağımsız
  izlenir; yeni transient kaynaklar o yaşam döngüsünü daha da karmaşıklaştırmamalıdır.
