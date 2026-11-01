# Ana plan kapanış çalışması — 2026-10-28

## Kullanıcı yönlendirmesi sonrası ana kaynak partisi — 2026-10-29

- D2 nested-clock ceiling düzeltmesi: uniform continuum*ceil(DEM/continuum)
  ürünü authored limit'i aşabiliyordu (40*ceil(512/40)=520 >512). Clock hook
  artık DEM isteğini de taşır. Yalnız taşmada, quotient sınırları üzerinden
  ilk uygun even continuum sayısı seçilir; CFL aralığı büyümez, fizik
  sertliği değiştirilmez, DEM ping-pong bank0 ve toplam zaman korunur.
  512/40/512 →64*8=512; 502/40/512 →42*12=504; normal 80/40/512 aynı.
  Probe ve mixed acceptance artık measured DEM total<=authored limit'i de
  denetler. 2..128 bütün limit/istek/continuum kombinasyonlarında bağımsız
  exhaustive minimum-feasible oracle PASS; C++ test kaynağına da eklendi,
  derlenmedi. Grain/GPU kaynak audit PASS; canlı ölçüm sona bırakıldı.

- D2 ortak saat/kuvvet maliyeti: SimulationWorld packed snapshot'ından yalnız
  Fluid'i etkileyen enabled Wind + fluid_surface_drag alanı için kolon ihtiyacı
  yayınlar. Atmosfer rüzgârı ayrı korunur. Particle force stage kolon clear/build
  dispatch'lerini yalnız ihtiyaçta gönderir; yoksa column_nx/nz=0 ile shader'ın
  önceden hazırlanmadığı halde kolon okuması da kapanır. Ordinary body-force
  denklemine, gravity/MPM/DEM'e veya shader ABI'ye dokunulmadı. Harici buffer
  üreticilerinde host flag varsayılanı conservative=true. C++ predicate testi
  hazır, koşulmadı; GPU/phase/grain static audit PASS. Bu kaynak değişikliği
  henüz derlenmedi; hız kazancı ölçülmedi. DEM contact-only temporal azaltımı
  tamamlanmış sayılmaz, mevcut multirate/Verlet/sleep altyapısı korunur.

- T6 E2 loader sıra hatası kapatıldı: ProjectManager ve SceneSerializer
  domain'leri kaynaklardan ÖNCE synchronize ederek legacy type'ı kaybediyordu.
  İlk synchronize bütün flow source kayıtları eklendikten sonraya taşındı;
  kaynak fazı ve shader göçü aynı core'da, bütün input hazırken çalışır.
  Disabled kaynak fazını kasıtlı yanlış tutan mevcut legacy roundtrip testi
  bu durumu kapsar. Statik sıra denetimi, lanes/phase ve project timeline
  audit'leri PASS. Son başarılı kullanıcı build'inden sonra; canlı koşulmadı.

- T6 loader tipi ortak core parser'a bağlandı. ProjectManager'ın bilinmeyen
  her string'i LegacyGas sayması ve SceneSerializer'ın bilinmeyen enum sayısını
  kabul etmesi kaldırıldı. Geçerli eski Gas/Fluid, 0/1, yeni Matter/2 ve eksik
  alanda Gas varsayılanı korunur. C++ kontrat testine geçerli/invalid ad ve sayı
  durumları eklendi (koşulmadı). Python AST runtime type-filter audit'i bütün
  scripts/test/*.py için PASS; tek eski-type eşleşmesi açıkça legacy fixture.
  Lane ve phase kaynak audit'leri PASS. Yeni fizik ölçümü/tuning yapılmadı.

Hassas su/cephe testleri ve tuning sona bırakıldı. Mevcut FAIL'ler korunur.
T6 E6 kaynak düzeltmesi: `MatterCacheRestore.h` RAM metadata'yı ve disk
frame'lerini bütün sistemlerde commit öncesi doğrular. Eski akışta
setGridDomainStates red verse bile MSF/particle sidecar restore ve üst
katmanın true dönüşü devam edebiliyordu; disk read başarısızlığı runtime'ı
silip yine kare restore başarılı görünebiliyordu. Artık lane/count veya
read hatasında hiçbir grid/MSF commit'i yapılmaz; disk cache invalid +
rebake log ve false sonucu. Yeni odaklı header, büyük scene_data'da yalnız
çağrı/taşıma wiring'i, proje kayıtları ve shared lane validator kullanır.
Core C++ testine iki sistemli geç hata/eksik dosya/başarılı commit fixture'ları
eklendi; ajan derlemedi. Son kullanıcı build'inden SONRA eklendi.

Önceki kısa teşhis ölçümü (yeniden koşulmayacak, tuning kararı değil):
Water gerçek hints flip=.97/apic=.95/tau=.05. Density off p95=4.458 mm;
APIC off 3.104 mm; PIC(0/0) 0.718 mm peak, son 0.275 mm;
FLIP .3 1.177 mm peak. Baseline 4.327 mm peak. Bu veri tau'nun tek neden
olmadığını gösterir; henüz kesin fizik düzeltmesi veya kabul değildir.

## En yeni canlı checkpoint — kullanıcı yeni derlemesi, 2026-10-29

Dokuz seçili kol 7/9 PASS. Yeni transport 8/8, lanes 7/7, substance,
melt 4/4, floor 2/2, mixed clock 5/5, run readback PASS. Snow-only 300/mixed
600; floor status'taki density failure kalktı. Water rest p95=4.311 mm/kare
FAIL; Ritter 0.3/0.9 hızları 1.1093/1.9494 m/s, referans 2.4261 m/s:
ikisi <=%15 doğruluk kapısını geçmedi, 0.3 ayrıca belirgin regresyon. Water
ön ayarı değiştirilmedi. Son sahne boş/Pause/frame0. Tam sayılar ve log yolu
NEXT_BUILD_CHECKS en üstte. Eski "yeni build bekliyor" notları tarihsel;
bu turdan kalan source önceliği liquid rest/density, ardından geniş kabul.

## Test sonrası kaynak düzeltmesi — 2026-10-29

- Güncel scene_data.h'daki `preserveScriptSimulationPreview()` seed yolunda
  vardı, fluid.step/run sonunda yoktu. World step sonrası bu mevcut servise
  bir entegrasyon çağrısı eklendi: ilk UI tick authoring/config değişimini
  frame-zero restore olarak yorumlayıp tamamlanan adımı silmemeli. Manual
  claim ilerlemesi ayrı korunur. Snow/melt ilk boş okuma kapıları yeni build
  olmadan PASS sayılmadı. Physics.step davranışı bu ek çağrıyla değiştirilmedi.
- Sleep-transfer'ın ayrı static audit'i de eski 5-word/4-stride beklentilerinden
  8-word/7-stride güncel metadata'ya geçirildi; bağımsız impulse/energy/time
  spesifikasyonlarıyla PASS. Süreç denetiminde RayTrophiStudio çalışmıyordu;
  ajan uygulama başlatmadı.

- D2 ortak saat ölçümü: `rt_probe_grain_clock_ipc.py` artık tam host_ms,
  continuum/grid/DEM schedule, contactless ve upload/download baytlarını
  kaydeder; ayrı summary dosyasında örneklenmiş median/p95 bulunur. Süreler
  iç içe olabildiğinden toplam alınmaz. `--require-clock` held/eksik clock/
  DEM=continuum*per veya grid<=continuum ihlalinde fail verir. Shared-clock
  kabul testine aynı bağımsız schedule gate eklendi; null diagnostics eksik
  ölçüm olarak ele alınır. Aritmetik/missing-data oracle uygulamasız PASS;
  canlı 4-emitter maliyet kazancı ölçülmedi ve kabul verilmedi.

- D2.6 kaynak eklendi: `UI/SimulationTransport.cpp` TimelineWidget Play/Pause
  ve iki Simulation Mode panelinin ortak servisi. Mevcut pause GPU drain/audio
  davranışı bu modüle çıkarıldı; UI, Python ve IPC aynı core çağrılarını kullanır.
  `timeline.play/pause/transport`, `sim.get_mode/set_mode` gerçek viewport
  döngüsünü sürer. mode yalnız timeline/live, invalid patch mutasyondan önce red;
  API final render sırasında mutation reddeder. Read state cache valid/baking/
  RAM range ve manual claim'i de verir. Project XML, security ve descriptor
  kayıtları eklendi. Eski "D2.6 gerçekten açık" checkpoint'i bu kaynaktan önceydi.
- `rt_test_simulation_transport_ipc.py` Play/Pause ve Live birth kontrolünü
  fluid.step/run olmadan dış polling ile yapar. Mode geri yüklenir, test
  domain/source temizlenir. Yeni kullanıcı build'inden önce canlı koşulmadı.

- Son kabul runner'ına matter_lanes/grain_sleep/ritter_flip full kolları eklendi;
  legacy_project yalnız `--with-legacy-project` ile seçilebilir. Quick sabit.
  TimeoutExpired bytes çıktısının str'e eklenirken crash etmesi düzeltildi;
  boş/yanlış/kapalı --only artık app erişiminden önce exit 2 verir, 0/0 PASS
  olamaz. Mock subprocess/cleanup ile bu hata yolları uygulamasız doğrulandı.
- D2 source re-audit: continuum kendi grid_stride'ında; liquid refresh/react
  continuum tick başına, iç DEM adımları per çarpanıyla çalışıyor. Host dökümü
  mevcut diagnostics runtime.host_ms'de. Temas olmayan taneler hâlâ fused
  integrator'a girer; contact-only temporal optimization ve 4-emitter maliyet
  kabulü tamamlandı sayılamaz.
- D2.6 kontrolü GERÇEKTEN açık: RtIpc/RtPython aramasında timeline.play/pause
  ve Live Update setter bulunmadı. Yalnız manual claim düzeltmesi bu gereği
  karşılamaz. TimelineWidget pausePlayback var, start setter yok; g_sim_timeline_mode
  mevcut UI live_mode kaynağı. Ortak core/UI/Python/IPC kontrolü kaynakta
  tamamlanmalı; eski yapılacaklar listesinin "varlığını kontrol et" notu artık
  bu somut eksiklik olarak okunmalı.

- DEM metadata kaynak hatası düzeltildi: history_blocks shader guard'ı eski
  `5*capacity` yerine map+owner+4 masks+rest+elapsed = 8 kelimeyi doğrular.
  Çarpma taşmasını önlemek için length/8 karşılaştırılır. Host/shader revision
  32→33; descriptor/push ABI aynı 17/128. Son kullanıcı build'inden sonra
  eklendiği için **C++ + sim shader kullanıcı derlemesi gerekir**. Sleep audit
  bu guard'ın history map okunmadan önce olmasını da doğrular. Grain/sleep/GPU
  kaynak kontrolleri PASS. Hata ancak eksik/uyumsuz metadata tamponunda
  belirginleşir; geçerli revision-32 koşularındaki jitter'ın nedeni olduğu
  iddia edilmez.

- T6 `rt_test_matter_legacy_roundtrip_ipc.py` eklendi: güncel proje kabuğunda
  eski Gas/Fluid schema fixture üretir, gerçek ProjectManager load/save/open
  yolundan iki domain, disabled source phase, tek-grid, gaz shader ve ikinci
  canonical açılışı denetler. Yalnız harcanabilir boş proje; bütün dosyalar yeni
  TEMP klasöründe, aktif proje yolu değişir. Fixture-transform `--self-test`
  PASS; gerçek load/save kabulü henüz koşulmadı. Tarihsel bir kullanıcı dosyası
  ile uyumluluğun tüm sürümlerini kanıtlamaz.

- D2 FLIP ölçüm altyapısı: `rt_test_ritter_flip_ipc.py` 0.3/0.9 için
  türetilmiş Water profilleriyle aynı dam-break kolonunu çalıştırır. Yerleşik
  ön ayarı değiştirmez; profil/domain cleanup yapar. Leading carrier x-max
  kuyruğundan hız, Ritter ideal kuru sürtünmesiz yatak referansıyla karşılaşır.
  Referans: ETH VAW Mitteilung 222, §2.5.1:
  https://ethz.ch/content/dam/ethz/special-interest/baug/vaw/vaw-dam/documents/das-institut/mitteilungen/2010-2019/222.pdf
  3D finite-depth startup ve leading-carrier outlier hassasiyeti açıkça
  belirtilir. Hız hatası <=%15, 0.3 regresyon <=5 yüzde puan; rest/column
  sonuçları olmadan preset kararı verilmez. Aritmetik self-test PASS;
  canlı benchmark henüz koşulmadı.
- Ortak `getFluidStateDigest` canonical host positions'dan ölçülen min/max
  merkez koordinatları verir; Python/IPC aynı core veriyi sunar. Empty/unstepped
  için bounds_measured=false ve sıfır placeholder; nonfinite position red.
  Run-readback testi artık centroid'in bounds içinde kalmasını da doğrular.

- `check_matter_grain_contracts.py` ve `check_matter_grain_sleep_contracts.py`
  güncel kaynak sözleşmeleriyle PASS. Eski sabit SLOTS/LIST yerine host-upload
  metadata/shader okuması, dört history mask word'ü ve map+7 metadata word,
  radius-cubed birth mass, çift yarıçaplı bridge/rolling, renderer'ın canonical
  tane yarıçapı ve mevcut raporlanan CFL adaptation/cap politikası denetlenir.
  Substance material paneli read-only denetimi korunur; ayrı domain size editor
  için ortak rtapi çağrısı doğrulanır. Physics kodu bu audit partisinde değişmedi.
  Bu sonuç sayısal jitter/sleep-transfer veya CFL adaptation doğruluğu kabulü
  değildir; bunların canlı kapıları açık kalır.

- Density adapter GPU politika denetimi: scratch solve boyunca
  `matter_model.enabled=false`; böylece sıfır host hız alanının yüklenmesi ve
  düzeltme alanının host'a okunması resident-model kısa yoluyla atlanamaz.
  Önceki enabled değeri RAII ile geri gelir; pressure statics kirli işaretlenir
  (scratch zero solid velocity sonraki fizik çözümüne taşınmaz). Mevcut
  `runMatterGpuStep` zaten ResetViews kullanır; bu ek koruma tek başına son
  water-rest başarısızlığının nedeni olarak kanıtlanmış değildir.
- `fluid_liquid_advection_test.cpp`: affine korunumu, backend exception'da
  grid/position rollback ve frozen-only solve callback'inin hiç çağrılmaması
  senaryoları eklendi. C++ test derlenmedi/koşulmadı; kullanıcı build kuralı.

- `RtManualSimulationStep.h`: physics.step'in mevcut timeline claim yayını
  ortak yardımcıya çıkarıldı; fluid.step/run da her tamamlanan world step sonrası
  bunu çağırır. Böylece Pause viewport tick'i parked frame 0 durumunu geri
  yüklememeli. Cache/config invalidasyonu ayrı kalır; bu değişiklik otomatik
  olarak Snow veya melt kabulü anlamına gelmez.
- Run-readback kabulü artık particle/mass yanında script_driving ve ilerleyen
  frame'i de zorunlu tutar. MPM/DEM testi run içi son particle_count ile dış
  digest'i karşılaştırır, önce/sonra control state basar. Canlı teşhis denemesi
  pipe error 2 ile hiç bağlanamadı; uygulama başlatılmadı.

- `MatterDensityProjection.inl`: GPU fiziksel hız geri yüklemesi yalnız basınç
  callback'i gerçekten GPU operatörünü denediyse yapılır. Hareketli sıvı yoksa
  (özellikle Frozen ice + Sand) no-op artık kullanılmamış GPU tamponlarına
  upload denemez. Başarısız operatör denemesinden sonra geri yükleme korunur.
- Substance kabulü Sand'in kaynakta tanımlı 0.1 pore-water kapasitesini doğrular;
  eski sıfır beklentisi kaldırıldı. Malzeme fiziği değiştirilmedi.
- Bu iki değişiklik son kullanıcı derlemesinden SONRADIR; canlı kabul kanıtı
  değildir. Sonraki kullanıcı derlemesinde substance ve P9 floor tekrar koşulmalı.
  Snow-only boş readback, melt ilk kütle ve water rest hâlâ açık.

## Son canlı checkpoint — 2026-10-29 kullanıcı derlemesi

T6 lanes 7/7 PASS; quick smoke 7/11. Yeni düz P9 floor 2/2 (kum min-y
0.2099565 m, 960 ice face), fakat status density position projection failed.
Substance Sand water-capacity beklentisi, liquid_at_rest (4.31 mm p95),
p9_melt ilk kütle okuması 0 ve snow-only digest 0 açık. Ayrıntılar
NEXT_BUILD_CHECKS ilk bölüm. Sahne temiz, Pause/frame0. Eski "henüz derlenmedi"
ifadeleri kaynak checkpoint tarihidir; bu build yalnız koşulan kapılara kanıt.

**DEVAM EDİYOR. Henüz derleme/kapanış teslimi değildir.** Kullanıcı ana planın
bitirilmesini istedi; kapsam D2 kalanları + D3/T6. Yeni yan iş ekleme, küçük
adımlarda onay isteme. Build/shader derlemesi kullanıcıda; canlı test en sonda.
Son gözlenen 4-emitter sahnesi Pause'daydı. Bu kaynak partisinde uygulamaya
dokunulmadı. Kullanıcı: diğer ajan rigging planında; derleme bu yüzden askıda.
Kaynak işi bitene kadar build/canlı test isteme veya çalıştırma.


## 2026-10-29 kaynak devamı — D3 ve buz yüzleri

- `SimulationDomainLanes.{h,cpp}`: tek canlı Matter kimliği; eski enum isimleri
  LegacyGas/LegacyFluid ve yalnız giriş/göç amaçlı. Descriptor/state şerit
  anahtarları, eski kaynak phase + gas shader göçü, atomik lane validasyonu,
  tek gaz alışveriş sahibi ve RAM cache layout red servisi.
- HasGas/HasLiquid bütün tüketicilerde descriptor/state üzerinden. Tek şerit
  birincilde, yalnız dual Matter ikincil sıvı/foam/render buffer'ını kullanır.
  Kapalı gaz için solver/sources/rain/grain drag kapıları var; absent grid silinir.
- E1: `resolveMatterGasExchangeOwners` kendi örtüşen gazını önce seçer;
  liquid-only ilk etkin örtüşen gaza gider. Surface heat, phase transfer ve
  ambient-film kararı aynı owner vektörünü kullanır; domain sırası kendi gazını
  bir dış gaza kaptıramaz.
- UI Lane preset, Python `get_lanes/set_lanes` ve dış IPC aynı rtapi/core
  servisinde. Canlı içerik, kaynak hedefi, DEM zorunluluğu ve boş şerit seti
  mutasyonsuz red; no-op resetlemez. Kaynak auto phase yalnız açık şeridi seçer.
- Phase JSON v2 lane flags; disk cache v19 lane baytları. Legacy Gas/Fluid
  frame'leri gerekçeli red, eski dual Matter cache'leri okunabilir. GPU gas-only
  primary live view korunur; shader geçişleri son gaz look'unu taşır.
- Ölü ve zaten kayıt/çağrısı kalmamış beş GranularVirtualScene/Representation/
  Surface dosyası silindi. Aktif DEM virtual grains/mesh grains kaldırılmadı.
- Tip temelli script filtreleri phases'a geçti; gas/fluid creation string'leri
  compatibility preset olarak geçerli. Eski pre-T3 preset/physics-key testlerinin
  ayrıca bayat olması T6 tarafından düzeltilmiş diye iddia edilmez.
- Buz geometri işi `MatterIceCollider.{h,cpp}` modülünde: integer grid node
  koordinatları (komşu kenarlar bitwise aynı), negatif yüzlerde dışa winding.
  BVH zaten bütün ice face'lerini fingerprint'e katıyordu. Yeni C++ test
  outward normals / internal face removal / duplicate parcel / unfrozen cases.
- P9 floor testi artık küresel donmuş yığın yerine 180 K, domain-wide düz seed
  slab kullanır; eski min-y gate yanlardan meşru kaymayı tunnelling sayıyordu.
  Gate gevşetilmedi; ayrıca step_held olmaması aranır. Canlı sonuç YOK.
- Yeni kabul kaynakları: `rt_test_matter_lanes_ipc.py` (7 kapı),
  `matter_domain_lanes_test.cpp`, `matter_ice_collider_test.cpp`. C++ testleri
  derlenmedi/koşulmadı. `MATTER_T6_LANES.md` E1–E10, migration ve sözleşme.
- Statik PASS: lane contracts, phase contracts, 13 GPU ABI contracts, IPC
  capability audit + descriptor freshness (712 metot), değişen Python AST/XML.
  Yeni C++ modüllerinde 100 kolon kontrolü temiz; whitespace kontrolü temiz.

**Devam:** D3 kaynaklarını özellikle preset varsayılanları ve legacy native
render/material köşeleri için gözden geçir. Tam legacy `.rtp` save/open kabulü
son kullanıcı build'inde; dar C++ fixture bunu kanıtlamaz. D2 FLIP/Ritter,
yalnız temas tanelerinde mikro adım, host breakdown, bayat grain/sleep statik
kontrolleri ve density-pass kaynak/diagnostic incelemesi hâlâ açık. Kullanıcı
rigging çalışması bittiğinde bütün kaynak hazırsa tek build + toplu kabul.

## Bu kapanış partisinde yazılanlar (henüz derlenmedi)

- Snow library kabulü artık eski DP compaction alanlarını değil gerçek Snow
  plastisite/hardening/compression/stretch sözleşmesini denetliyor.
- Ortak saat testi boş Snow popülasyonunu düşmüş sayamıyor; pozitif ve eşit
  popülasyon olmadan centroid çıkarımı PASS veremiyor.
- `FluidDomainSubstance`: granular fizik alanları ortak yardımcıya çıkarıldı;
  MPM parçacığının gerçek maddesi (Snow) domain Water olsa da constitutive
  parametreleri seçiyor. Sayısal hints/default substance değiştirilmez.
  Birden çok farklı MPM maddesi için ayrı constitutive batch henüz yok;
  ilk MPM maddesini seçen mevcut model başına parametre kısıtı ayrıca denetlenmeli.
- Tek-model granular Matter da `requiresMatterModelBatch` ile Matter yolunda;
  tek Snow'nun genel domain-malzeme yoluna sapması engelleniyor.
- CPU RK2 `FluidLiquidAdvection.h`: frame liquid support dışında kendi hızını
  koruyor (Vulkan D1b ile aynı kural). Ana >2000 satırlık dosyada yalnız
  include/nesne/call wiring. CPU pressure mevcut sayaçlara iterasyon/son
  göreli kalıntı yazıyor.
- `FluidDensityProjection.h` + `MatterDensityProjection.inl`: domain fiziksel
  pressure/G2P'de density relaxation kapalı; frame sonunda aynı CPU/GPU
  pressure operatorüyle sıfır hız alanından yalnız konum düzeltmesi. Hız,
  affine, fiziksel MAC hızları korunur. Frozen/static/granular parseller
  taşınmaz; shift en çok 0.25h, solid hedefe girmez. GPU policy restore ve
  particle-upload invalidation var. Yeni shader eklenmedi.
  **Bu yeni yol nümerik/canlı doğrulanmadı.** Özellikle compact MAC sonrası
  fiziksel GPU field restore, solid velocity scratch ve maliyet incelenmeli.
- `fluid_liquid_advection_test.cpp`: CPU support/ballistic fallback ve konum
  düzeltmesinin hız alanını koruması/başarısız callback rollback testi yazıldı;
  kullanıcı build kuralı gereği C++ test derlenmedi.
- GPU source ABI kontrolünün stress_update pini shader/registry ile 100 B'ye
  güncellendi; kontrol PASS. Bu bir ABI implementasyon değişikliği değildir.

## Güncel kalan kabul / kararlar — 2026-10-29

1. Kullanıcı C++ + sim shader revision 33 derlemesi. Yeni transport, bounds,
   density scratch ve manual preview kaynakları son başarılı build'den sonra.
2. Density/rest: liquid_density, liquid_column, seed_settle. İkinci çözüm
   statik olarak incelendi; hız pompasının kalktığı/volume korunduğu ölçülmedi.
3. FLIP 0.3/0.9 Ritter testi artık VAR. Aritmetik self-test geçti; gerçek
   cephe/rest/column sonuçları olmadan Water ön ayarı değişmedi.
4. Buz düz levha kabulü son eski build'de geçti; yeni helper/metadata guard
   ve density status için yeniden kabul gerekiyor. Küresel yığın eski false
   fail fixture'ı artık kullanılmıyor.
5. Ortak saat: continuum grid stride ve DEM iç döngüsü var. Host süre dökümü
   API/probda var. Yalnız temaslılara zaman daraltma ve 4-emitter hız kazancı
   tamamlanmış kabul edilmez; yeni build profili sonraki optimizasyon kararını
   belirleyecek. --require-clock schedule/missing-data kapısı hazır.
6. Play/Live ortak UI/Python/IPC servisi artık VAR; gerçek frame-loop kabulü
   bekliyor. CPU PCG ölçümleri ve RK2 fallback kaynakta, yeni canlı kabul yok.
7. Grain/sleep/sleep-transfer statik kontrolleri güncel 8-word/4-mask düzende
   PASS; gerçek weak/strong transfer, jitter ve large-N kabulü ayrı.
8. D3/T6 E1–E10 kaynakta, lanes 7/7 önceki build'de geçti. Synthetic legacy
   project load/save/open testi hazır, canlı koşulmadı. T0 yeniden, cache red
   mesajı ve tarihsel kullanıcı dosyası/görsel kabul hâlâ açık.
9. Tek son tur komutları NEXT_BUILD_CHECKS başında. Kaynak veya statik PASS'i
   canlı fizik PASS'i olarak kullanma; goal henüz tamamlanmadı.

## Şimdiye kadarki statik doğrulama

- `check_matter_phase_contracts.py`: PASS.
- `check_matter_gpu_contracts.py`: PASS (13 ABI + ordering/budget/commit/API).
- Değişen Python AST ve vcxproj/filters XML: PASS.
- `git diff --check`: whitespace error yok (repo CRLF uyarıları var).
- C++/shader build ve bu yeni partinin canlı testi yapılmadı.

Önceki canlı emisyon kabulü ayrı: 6/6 + ledger PASS, quick smoke 7/11;
4-emitter 204 adımda 89.571863067 kg source/domain withheld eşit, sarı pill
gözle PASS. Bunlar yukarıdaki yeni kodun kanıtı değildir.
