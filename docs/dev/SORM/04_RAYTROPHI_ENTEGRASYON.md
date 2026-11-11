# SORM — RayTrophi entegrasyon ve görev planı v0.3

**İnceleme tarihi:** 11.11.2026. Yerel kaynak incelemesi yapılmıştır; uygulama
çalıştırılmamış, C++ veya shader derlenmemiştir. Yeni modül ve komutlar öneridir;
uygulanmış özellik veya canlı kabul sonucu değildir.

## Karar ve kapsam

Hesaplama C++ çekirdek/ortak servistedir; Python scriptleri deney dizisi, tohumlar,
kontroller ve analizi yönetir. Python ve IPC aynı servise gider, ikinci bir
bilimsel çekirdek oluşturulmaz. Kodlama öncesi yöntem önerileri `05` belgesinde;
her ajanın her ilerleme ve sonucu belgeleme yükümlülüğü README/AGENTS ve `06`
günlüğünde tanımlıdır. Bu kayıtlar işin tamamlanma koşuludur.

RayTrophi altyapısında, Matter/DEM/MPM/FLIP çözücülerinden mantıksal olarak ayrı
küçük araştırma çekirdeği. CPU referans, ham telemetri, ortak Python/IPC servisi ve
render gerektirmeyen batch yolu ve erken sahne gözlemi (K1-G) ilk teslimde birlikte
bulunur. RayTrophi bu araştırmanın görsel laboratuvarıdır. GPU compute yalnız
profil gerekçelendirirse eklenir; CPU durumunu GPU viewport’ta çizmek bunun için
beklemez. Çoklu duyusal sensörler daha sonradır; ilk sürümün sahnede otonom eylem
yetkisi yoktur.

## Kaynakta doğrulanan mimari harita

Aşağıdaki yollar `RayTrophiStudio/source/` altındadır. `ARCHITECTURE.md` ve
`ARCHITECTURE_TR.md` okunmuştur; somut sınırlar kaynak uygulamasından çıkarılmıştır.
Yerel ağaç başka ajanların devam eden değişikliklerini içerdiğinden kodlama
öncesinde koordinasyon panosu ve ilgili dosyalar taze okunur.

| Kaynak / işlem | Kullanım ve sınır |
|---|---|
| `include/SimulationCompute.h`: `ISimulationComputeBackend`, `SimulationComputeContext` | Buffer yaşam döngüsü, upload/download, dispatch, transfer sayaçları, bellek ve GPU timestamp ölçümü hazır. CPU backend dispatch çalıştırmaz; CPU referans doğrudan adımlar. |
| `src/Device/SimulationComputeVulkan.cpp`: `dispatch`, kernel kayıt tablosu | Mevcut cihaz/buffer/descriptor/barrier hizmetleri kullanılır. Yeni Vulkan cihazı veya shader yükleyici kurulmaz. |
| `include/SimulationWorld.h`: `ISimulationSystem`, `SimulationWorld::addSystem`, `compute` | İleride sahne entegrasyonu için uzantı noktaları; K1'in sahne sistemi olarak kaydolması gerekmez. |
| `src/Physics/SimulationWorld.cpp`: realtime adımlama ve `executeStep` | Sabit adım var; realtime biriktirici alt adım sınırında zamanı atabiliyor. Bilimsel koşu duvar saatinden bağımsız tam adım sayısıyla yürütülür. |
| `include/Fluid/FluidParticles.h` | Düz SoA ve kalıcı kimlik için örnek; akışkan/malzeme durumunu SORM'a taşımak gerekmez. |
| `include/Fluid/DemBondNetwork.h` | Fiziksel gövde, bağ ve kırılma semantiği SORM belleği değildir; ilişkisel iz deposuna dönüştürülmez. |
| `src/Api/RtIpc.cpp`: `enqueueQuery`, `enqueueResult`, `waitForFuture` | Named-pipe/JSON yolu ana iş parçacığı kuyruğuna gider. Bekleyen istek için 30 saniyelik dispatch sınırı var; başlamış işlem gerçek sonucu bekler. Tek uzun çağrı yerine kısa adım paketleri/durum sorgusu kullanılır. |
| `include/Api/RtApi.h`: `enqueue`; `src/Core/Main.cpp`: `drainMainThreadQueue` | API işleri ana döngüde yürür. Embedded script içinde IPC beklenmez. |
| `src/Api/RtPythonDomainWalls.cpp`, `RtIpcDomainWalls.cpp`, `include/Api/DomainWallsApi.h` | Python ve IPC'nin aynı rtapi servisini çağırdığı modüler bağlama örneği. |
| `src/Api/RtIpcMethodRegistry.h`, `RtIpcMethodDescriptors.cpp` | Mevcut keşif, parametre doğrulama ve yetki metaverisi kullanılır; registry çalışma kuyruğu değildir. |
| `src/Core/Main.cpp`: CLI parse, `run_hidden`, `runScriptFile` | Mevcut `--headless`, `--render` veya `--frames` gerektirir ve SDL/UI başlangıcından geçer. Genel render'sız deney modu henüz yoktur. Başlangıç script'i IPC sunucusundan önce çalışır. |
| `src/Api/RtIpc.cpp`: `fluid.step_stats`; `include/SimulationCompute.h`: `TransferStats`, `GpuKernelTiming` | Altyapı performans ölçümü kullanılabilir. SORM bilimsel çıktısı ayrı şema alır; fluid istatistiklerine eklenmez. CPU enqueue süresi GPU çalışma süresi değildir. |
| `include/UI/ParticleBillboardBuilder.h`, `include/Viewport/ParticleBillboardData.h` | K1-G için mevcut CPU billboard aktarım adayı. Builder bugün sahne parçacık/akışkan sistemlerini toplar; SORM snapshot adaptörü gerekir. LUT/age tabanlı renk yolu fiziksel yaş olarak yorumlanmaz. |

Kök `scripts/test/rt_ipc.py` dış süreç named-pipe istemcisidir;
`scripts/test/rt_test_dem_bodies_ipc.py` mevcut IPC kabul örneğidir. SORM ham zaman
serisi, manifest ve bilimsel metrik şeması henüz yoktur. JSON/CSV çıktısı küçük
deney modülünde sağlanır; yeni veritabanı veya transport gerekmez.

## K0/K1 dosya bazlı minimum değişiklik planı

M0'ın kesin matematik ve deney referansı `07_M0_SOZLESMESI.md` dosyasıdır.
CPU/Python/IPC/viewport bu tek sözleşmeyi kullanır. K0 M0 tanımı tamamlandı;
gerçek JSON şeması, alan/hata adları ve çıktı codec'i uygulamada belgelenecek.

Yollar `RayTrophiStudio/source/` altındadır; dosyalar henüz oluşturulmadı.

| Önerilen yeni modül | Sorumluluk |
|---|---|
| `include/Sorm/SormCore.h`, `src/Sorm/SormCore.cpp` | Düz durum dizileri, sabit yerel komşuluk, CPU M0 ve Heun adımlama. |
| `include/Sorm/SormExperiment.h`, `src/Sorm/SormExperiment.cpp` | Manifest doğrulama, sabit uyarı alanları, RNG, deney evreleri, checkpoint, ham JSON/CSV ve durum makinesi. |
| `include/Api/SormApi.h`, `src/Api/RtApiSorm.cpp` | Tek ortak servis: başlat, sınırlı adım ilerlet, durum, sonuç ve durdur. |
| `src/Api/RtPythonSorm.h/.cpp`, `src/Api/RtIpcSorm.h/.cpp` | Aynı servise ince bağlar; ayrı iş mantığı içermez. |
| `include/Sorm/SormCli.h`, `src/Sorm/SormCli.cpp` | Aynı servisi render başlatmadan batch olarak yürüten CLI adaptörü. |
| `include/Sorm/SormObservation.h`, `src/Sorm/SormObservation.cpp` | Adım/zaman damgalı salt-okunur snapshot ve kayıt oynatma; bilimsel durumun ikinci sahibi değildir. |
| `include/UI/SormViewport.h`, `src/UI/SormViewport.cpp` | Snapshot → mevcut billboard/overlay çizimi, lejant, seçim ve gözlem kontrolleri. |

Kök `scripts/test/rt_test_sorm_ipc.py` mevcut dış istemciyle kabul yapar. Sayısal
fikstürler aynı çekirdeği kullanır; ayrı Python bilimsel model uygulaması yazılmaz.
Python analiz katmanı yalnız kayıtları değerlendirir.

Mevcut büyük dosyalara yalnız küçük include/kayıt/çağrı bağlantıları eklenir:

- `src/Core/Main.cpp`: örneğin `--sorm-experiment <manifest.json>` için mevcut
  render davranışından ayrı erken çıkış; donanım/SDL/UI başlangıcından önce aynı
  CPU servisine yönlendirme. Kesin komut adı K0 sözleşmesinde sabitlenir.
- `src/Api/RtPython.cpp`: `rt.sorm` modül kaydı.
- `src/Api/RtIpc.cpp`: modüler SORM dispatcher bağlantısı.
- Kök `scripts/gen_ipc_descriptors.py` ve `scripts/ipc_descriptor_overlay.json`:
  modüler handler keşfi/metaverisi gerektiği kadar güncellenir; mevcut üretim
  yoluyla `RtIpcMethodDescriptors.cpp` doğrulanır.
- `RayTrophiStudio/RayTrophiStudio.vcxproj` ve `.filters`: yeni modüller kaydedilir.

İlk teslimde Matter struct/cache şeması, fizik çözücüsü veya DEM graph değişikliği
planlanmıyor. Mevcut CPU billboard/LUT ve çizgi yolları önceliklidir; shader
değişikliği gerekmediği adaptör incelemesiyle doğrulanır, peşinen garanti edilmez. 2.000 satırı aşan dosyalara özellik gövdesi eklenmez.

## Yürütme ve hata sözleşmesi

Batch CLI, Python ve IPC aynı servis/core'u kullanır. IPC'de dış düzenleyici kısa
adım paketleri gönderip durum sorgular; ilk sürüm için ayrı genel job altyapısı
veya uzun worker görevi kurulmaz. GPU eklenirse mevcut cihazın thread/queue
sahipliği ayrıca doğrulanır; CPU tasarımından GPU worker güvenliği varsayılmaz.

Önerilen durumlar: hazır, çalışıyor, tamamlandı, iptal edildi, başarısız.
İşlemler koşu kimliğiyle adreslenir; aktif koşuda reset/başlat çakışması reddedilir.
İlerletme tam adım sayısına bağlıdır; çağrı paket boyutu sonucu değiştirmemelidir.

Geçersiz manifest, sonlu olmayan sayı, desteklenmeyen model/sürüm, geçersiz adım
ve durum geçişi mutasyon öncesi reddedilir. Ortak servis aynı hata nedenini
Python'da exception, IPC'de mevcut hata zarfı ve CLI'da başarısız çıkış olarak
sunar. Kesin alan/hata kodları K0'da belgelenir. İptal/hata başarı sayılmaz;
checkpoint yalnız tamamlanmış adımda alınır. Çıktı yazma hatası görünürdür;
mevcut sonuç dosyası sessizce ezilmez.

Checkpoint model/şema sürümünü, tüm dinamik durumları, simülasyon zamanını,
uyarı evresi/sırasını ve RNG durumlarını içerir. Manifest ara durum checkpoint'i
değildir. Sonuçlar ayrı deney çıktı dizininde tutulur; sahne cache'i kullanılmaz.

## Güncellenmiş uygulama sırası

| Parti | Kapsam ve kapı |
|---|---|
| K0 | Kesin M0/manifest, RNG, sürümlü kayıt/checkpoint, API ve hata sözleşmesi; açık sayısal kararları kapat. |
| K1 | CPU M0, 1.000 birim, sıfır/A/B; ham telemetri, ortak Python/IPC, render'sız CLI ve K1-G sahne gözlemi birlikte. Kabul `03` belgesindeki kapıdan geçer. |
| K2 | Tek kesin M1 adayı, aynı hızlı denklemin donmuş-iz kontrolü ve nedensel ablasyonlar. |
| K3 | K1 kayıtları üzerine çoklu tohum analizi, güç analizi ve dondurulmuş doğrulama. Telemetri buraya ertelenmez. |
| K4 | M2: pozitiflik, enerji akısı, besleme/kayıp/yapı maliyeti, muhasebe ve yakınsama. M3 ayrı karşılaştırma kararıdır. |
| K5 | Uzun koşu orkestrasyonu gerekirse kısa-adım API'sini genişlet; ilk IPC yüzeyi buraya ertelenmez. |
| K6 | Profil gerekçesiyle Vulkan; mevcut compute, SoA/kompakt komşuluk, ping-pong ve CPU/GPU kıyası. |
| K7 | K1-G üzerine gelişmiş karşılaştırma, akı, grup ve büyük ölçek görsel analizi; ilk görselleştirme burada başlamaz. |
| K8 | Salt-okunur görüntü/ses/fizik olay adaptörleri; kapalı döngü otonomi ayrı karar. |

K0/K1 uygulamasından önce model/protokol açık kararları kapatılır ve kodlama için
kullanıcı yetkisi alınır. Bu dokümantasyon revizyonu kodlama yetkisi değildir.

## GPU ve ölçek bütçesi

N=100000 ve ortalama derece 32 için simetrik grafikte yaklaşık 1,6 milyon
benzersiz kenar veya 3,2 milyon yönlü komşu kaydı vardır. Benzersiz kenar başına
32 bayt varsayımı yalnız kenar verisi için yaklaşık 51,2 MB eder; indeksler,
durumlar, yönlü kopyalar ve geçici tamponlar dahil değildir. Bu tahmindir;
mevcut SORM GPU düzeni veya ölçülmüş tüketim değildir. Hedef O(Nk).

Shader gerektiğinde `source/shaders/compile_sim_shaders.bat` ve mevcut Vulkan
kernel kayıt tablosu kullanılır. Yalnız etkilenen varyantlar derlenir, SPIR-V
doğrulanır ve `RayTrophiStudio/source/shaders/` ile `x64/Release/shaders/` içine
kopyalanır. ABI değişiklikleri koordinasyon panosuna yazılır.

## Doğrulama ve çalışma sınırları

C++ derlemesini kullanıcı yapar. Codex statik kontrolleri ve kendi değiştirdiği
shaderların derleme/doğrulamasını yürütür. Uygulama o turda açıkça istenmeden
başlatılmaz. Canlı API kabulü dış terminalde `scripts/test/rt_ipc.py` üzerinden
yapılır; sandbox named-pipe hata 5 için gerekli escalation kullanılır.

Derleme, sayısal CPU kabulü, headless/IPC kabulü, GPU eşdeğerliği ve bilimsel başarı
ayrı raporlanır. Bu revizyonda bunların hiçbiri çalıştırılmış değildir. Mevcut
Matter/DEM/MPM/FLIP çözücüleri yeniden tasarlanmaz; diğer ajanların değişiklikleri
geri alınmaz.


## K1-G gözlem katmanı ve mevcut çizim yolları

Akış: SormCore → tamamlanmış adım snapshot'ı → telemetri ve SormViewport.
Snapshot koşu kimliği, model sürümü, adım/zaman, konum/kimlik, x, I, komşuluk ve
modelin gerçekten tanımladığı diğer alanları taşır. Çizim state'e yazmaz, RNG
kullanımını değiştirmez ve çekirdeği kendiliğinden ilerletmez. Görünüm kapalıyken
deney aynı biçimde çalışır. Oynatma tamponu sınırlıdır; tam kayıt ayrı çıktıdır.

`ParticleBillboardBuilder::build` ve `ParticleBillboardUpload` CPU kaynaklı
veriyi zaten GPU çizimine taşır. SORM verisi sahte bir fluid domain veya DEM
malzemesi oluşturulmadan odaklı adaptörle mevcut upload'a eklenmelidir. Mevcut
appearance LUT yolunun renk/işaret/ölçek ihtiyacını karşıladığı doğrulanır; tek
birim başına sahne mesh nesnesi üretilmez. Tam entegrasyon noktası uygulama öncesi
builder çağrıları taze okunarak seçilir; yalnız küçük bağlantı eklenir.

`src/UI/scene_ui_gizmos.cpp` içinde `ProjectPoint` ve ImGui `AddLine` kullanan
mevcut dünya→ekran çizgi örnekleri bulunur. K1'de yaklaşık 3.000 benzersiz sabit
kenar için odaklı SormViewport overlay'i bu yaklaşımı kullanabilir. Yakın düzlem
kırpması, örtülme ve ekran koordinatları doğrulanmalıdır. Overlay X-ray ise
lejantta açıkça belirtilir; derinlik testli 3B bağ çizimi varmış gibi sunulmaz.
Eşik/seyreltme uygulanırsa yalnız görsel filtre olur ve görünen/toplam kenar sayısı
raporlanır. Bu dosyaya yeni özellik gövdesi eklenmez.

K1 görünümü x, I ve sabit komşuluktur. M1'de h ile etkili katsayılar, M2'de gerçek
e ve F, M3'te a görünür. Düğüm izi varken kenar belleği icat edilmez. Ham değer,
sabit renk aralığı, ölçek birimi, eşik ve simülasyon zamanı seçilen birimde
incelenebilir. Uzmanlaşma/küme renkleri sonradan hesaplanan analizdir; başlangıç
etiketi veya çekirdeğe geri besleme değildir.

UI mevcut viewport ve bağlamsal alanlara yerleşir; büyük kalıcı laboratuvar rafı
kurulmaz. Mevcut zaman çizelgesi gözlem/oynatma için adaptörle kullanılabilir;
playhead doğrudan deneyin sayısal saati olmaz. Duraklat/tek adım/uyarı/tekrar
kontrolleri ortak servise, renk/filtre/kamera kontrolleri yalnız gözlem durumuna
gider. Snapshot okuma, oynatma seçimi ve gözlem ayarları da Python/IPC'den aynı
servisler üzerinden erişilebilir olmalıdır.

K1-G teslim kapısı sayısal kabul ile canlı görsel kabulü ayrı raporlar. Kamera,
filtre ve çizim hızının canonical sonuca etki etmediği doğrulanır. Uygulama
açılmadan yalnız kaynak incelemesine dayanarak görsel başarı ilan edilmez.

İleride aynı nesnenin görüntüsü, çarpışması ve sesi ortak zaman damgasıyla salt
okunur giriş olabilir. Dönüştürücüler sabittir; nesne-ses doğru eşleşmesini veya
hedef temsili çekirdeğe vermez. Eşzamanlılık etkisi eşleştirilmemiş/geciktirilmiş
olay kontrolleriyle ayrıca sınanır.
