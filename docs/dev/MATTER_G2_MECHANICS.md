# G2 — granül mekaniği, mevcut sahne ve ilk kaynak düzeltmesi

2026-10-05. Öncelik: kuru bağsız kumun yük/çökme davranışı; sonra H1 tekil
tane contact/rolling/spin. Koyu materyal ve küre görünümü mekanik kabul değildir.

## Açık sahne — salt okunur inceleme

Physics Domain 1, paused frame 250: Young=2 MPa, friction=49°, cohesion=0,
tensile_cutoff=25 kPa; 43 required/43 applied substeps, invalid=0.
10343 granular carrier, dry mass=2068.600031 kg, stored water=2.223816 kg.
İki sürekli Water/Sand kaynağı açık; wet physics etkin. Bu, kuru ve bağsız
kum referansı değildir. Tensile cutoff bağımsız çekme dayanımı sağlar; cohesion=0
olması bütün bağların kapalı olduğu anlamına gelmez. Görsel güçlü kolon ve
destek üzerindeki yığın içeriyor: granular_current_inspection.jpg.
Granular physical carriers checkbox aktif; mevcut yol MPM'dir, DEM değildir.

## Bulunan açık ve kaynak düzeltmesi

Mixed GPU, tüm taşıyıcılardan measureLoad çağırıyordu: suyun affine/height ve
softening değerleri granül CFL'ye girebiliyor; extent/softening bilgisi elastic
planner'a taşınmıyor, load/wave/strain sayaçları varsayılan sıfır/1 kalıyordu.

MatterGranularLoad.cpp/.h canonical constitutive model'e göre yalnız granular
taşıyıcıları okur; Auto domain legacy modeliyle çözülür. Aynı legacy 1600 kg/m3,
9.81 m/s2 reference ve rho*g*vertical_extent proxy korunur. Bu hydrostatic
extent zarfıdır; havadaki/unsupported granülleri de içerir, gerçek destek/contact
basıncı ölçümü değildir. Çok maddeli profile density/mekanik authoring göçü ayrıdır.

Mixed elastic planner strain/load/softening girdilerini alır; authored Young
azaltılmaz. Load/CFL sayıları ve measured bayrağı yayınlanır. Cached/default
sayılar measured=false olarak ayrılır. Active Matter UI, mevcut Python
rt.fluid.matter_models ve fluid.matter_models IPC aynı runtime yayınını okur.
Yeni granular_mechanics raporu load_proxy ve contact_pressure_measured=false
ile sınırını açıkça belirtir. Fiziksel granül hareketi hâlâ MPM; H1 uygulanmadı.

## Build / kabul

- Normal kullanıcı C++ build; shader ve GPU binding ABI değişmedi.
- C++ regression: scripts/test/matter_granular_load_test.cpp; water height=100 m,
  affine=10000 ve softening=0 granular ölçümünü etkilemez; Auto ve pure kimlik
  kontrolü. Kaynak hazır, test target çalıştırılmadı.
- GPU/wet/transfer/stages statik kontrolleri + probe Python/XML PASS.
- Yeni binary için dış terminal, paused açık sert sahnede:
  `python scripts/test/rt_test_matter_granular_ipc.py "Physics Domain 1" --step --expect-load`
  --step tüm aktif domainleri bir kez ilerletir; kaynaklar açıksa emisyon sürebilir.
  Bu probe G2 mekaniğini tamamlandı ilan etmez.

## Sonraki mekanik kapı

Kullanıcı yeni derlemesi, paused frame 171: granular-only yayın canlı PASS;
extent load estimate=78195.38 Pa, load Young proxy=781953.81 Pa, authored/effective
Young=2 MPa, wave=43/strain=3/applied=43, invalid=0. Bu proxy destek basıncı değil.
Wet publication PASS; stored water=.977807 kg, wet carrier=2956, maximum
capillary cohesion=17.295 Pa, maximum pore pressure=.303 Pa. Görünüm policy v2.
Kullanıcı wet bölgenin dağılıp döküldüğünü gözledi. Önceki frame 250 ve yeni
frame 171/sürekli emisyon doğrudan A/B değildir; fiziksel neden kanıtı verilmez.
Wave 43 iki ölçümde de aynı; G2 sayaç düzeltmesi stiffness'i azaltmadı.
Ham kayıt: matter_g2_after_build_2026-10-05.json.

Ayrı kuru referans: yalnız canonical Sand; finite source veya kontrollü seed;
cohesion=0, tensile_cutoff=0, rebonding=false, stored pore water=0, thermal
softening kapalı. Sahneye keyfi settle damping veya cohesion ekleyerek geçiş yok.
İki başlangıç kolon yüksekliği; aynı fiziksel hacim/yük ile iki grid/dt;
emisyon sonrası dry kg, COM yüksekliği, runout/RMS radius, kinetik enerji,
repose ve destek yükü ölçülür. Sürekli düşen kaynak kolon yüksekliğinden
statik pile pressure/repose kabulü verilmez. G2/H1/C6/C7 açık.


## Sonlu üç koşu — 2026-10-05

Dış IPC, h=.1 m, dt=1/60 s, 150 adım=2.5 s; her koşuda 360 Sand,
72.000001073 kg; E=2 MPa, friction=49°, cohesion/tensile=0, rebonding=false.
Water koşularında 360 sonlu parcel, kaynak .5–.8 s; kapalı sınırlar, collider
geçici kapalı. Kaynak sonrası 60–150 adımlarda kuru kütle farkı sıfır;
en büyük su sapması water-only .654 mg, wet-response .315 mg.

| Son ölçüm | Kuru | Su, wet fizik kapalı | Su, wet fizik açık |
|---|---:|---:|---:|
| COM y (m) | .08895 | .07885 | .07725 |
| RMS yayılma (m) | .30144 | .36055 | .36051 |
| Kinetik enerji (J) | .01373 | .28059 | .31128 |
| Ortalama S | 0 | .12489 | .12386 |

Kütle kapısı PASS; mekanik nedensel A/B KABUL EDİLMEDİ. Su başlamadan
25. adımda wet-on kuru COM y=.31406, wet-off=.29454 m. Anahtar saf granular
ile ortak Matter solver seçimini değiştiriyor; S=0 response nötr olsa da
entegrasyon yolu aynı değil. Su gelişi de wet-off yolu sonradan değiştiriyor.
FluidDomainStep.inl minimal routing düzeltmesi: pore exchange açık Matter
baştan ortak solver kullanır. Pore kapalı legacy yol korunur; shader/ABI yok.
Pore-enabled saf kuru koşunun performansı da ortak yol maliyetini taşır.
Statik pore/wet/GPU kontratları PASS; yeni binary canlı tekrar bekler.

Tekrar: uygulama açık/duraklatılmışken dış terminalden
`python scripts/test/rt_g2_dry_wet_compare_ipc.py`.
Probe önce authoring yedeği alır; sonunda kaynak/domain/collider enabled ve
visible ayarlarını geri yükler. Runtime resetlenir; timeline frame 0 kalır.
Yeni karşılaştırma domain/sources sahnede disabled/hidden kalır; proje kaydedilmez.
Ön su COM/RMS farkı >1 mm ise nedensel kabul reddedilir. Tek grid/dt,
2.5 s ölçümü repose/yerleşme, doygun çökme, DEM veya G2 tam kabulü değildir.
Ham ölçüm: matter_g2_dry_wet_compare_2026-10-05.json.


## Routing düzeltmesi sonrası kullanıcı derlemesi — canlı PASS

Aynı sonlu probe üç koşuyu tamamladı. Su öncesi 25. adımda maksimum COM y
farkı .103 mm, RMS farkı .0093 mm; 1 mm nötr kuru eşik geçti. Önceki yaklaşık
19.5 mm solver farkı giderildi. Kuru kütle tüm örneklerde 72.000001073 kg;
kaynak sonrası su max sapması .726 mg (water-only), .609 mg (wet).

| 2.5 s son ölçüm | Kuru | Su, wet kapalı | Su, wet açık |
|---|---:|---:|---:|
| COM y (m) | .09260 | .08686 | .08365 |
| RMS yayılma (m) | .30047 | .34599 | .35173 |
| Kinetik enerji (J) | .06009 | 2.04222 | .41756 |
| Ortalama S | 0 | .13003 | .12989 |

Wet-on granular 360 taşıyıcının tamamı ıslak; max capillary=245.31 Pa,
pore pressure=182.15 Pa. Wet-on/off son RMS farkı yaklaşık %1.66; tek
tekrar ve hareketli su teması nedeniyle bu bir genel malzeme kalibrasyonu değil.
Kuru yığın son .5 s içinde hâlâ RMS .29490→.30047 m değişiyor. Su kontrolü
enerji .0984→2.0422 J yükseliyor; statik denge/repose kabulü verilmez.
Sonraki G2 kapısı: daha uzun kuru yerleşme, iki yük/kolon yüksekliği ve
iki dt/grid yakınsaması; gerçek yuvarlanma H1/DEM kapsamında açık.
Authoring geri yüklendi, frame0/paused; test domain/sources disabled/hidden.
Ham yeni binary kaydı: matter_g2_dry_wet_after_routing_2026-10-05.json.
Genel probe logu son koşuyla güncellenir; önceki RED bölümünün tablosu tarihsel.


## Sıradaki kuru matris — hazır, canlı bağlantı bekler

`python scripts/test/rt_g2_dry_matrix_ipc.py` (default 8 s/koşu).
Dört koşu: h=.1/dt=1/60/72 kg; dt=1/120/72 kg;
h=.08/dt=1/60/yaklaşık71.9872 kg; h=.1/dt=1/60/144 kg.
Yük oranı emitter parcel sayısıyla kurulur; iki kontrollü statik başlangıç
kolonu yerine geçmez. h=.08 count703 nedeniyle nominal kg farkı %0.0178;
canlı raporda gerçek kg ölçülür, eşit kütle varsayılmaz.

Ayrı bağsız referans: E=2 MPa, poisson=.3, friction35°, dilatancy/hardening=0,
cohesion/tensile/damage=0, rebonding=false; thermal/wet response kapalı,
pore açık ortak solver, su yok, collider kapalı, kapalı domain tabanı.
Önceki 49° kısa wet karşılaştırmasından ayrı fizik tanımıdır.
Her saniye dry kg/count/pore=0, invalid0, authored stiffness/CFL kontrolü;
son iki saniyede COM y aralığı<=1 mm, RMS aralığı<=%1 ve örneklenmiş
maksimum Ekin/kg<=.001 J/kg yerleşme ön kapısı. Bu eşikler önceden belirlenmiş
test kriteridir; deneysel Sand kalibrasyonu veya DEM kanıtı değildir.
COM/RMS dt-grid farkı<=%5, yük farkı<=%0.1 raporlanır; iki taraf yerleşmeden
statik yakınsama kabulü yok. Repose ve gerçek destek basıncı hâlâ ayrı açık.

Probe Python AST/CLI PASS; uygulama IPC bağlantısı Windows error2 verdi.
Canlı adımlar başlamadı, mevcut sahneye müdahale edilmedi; yeni C++/shader
build gerekmez. Uygulama açık/duraklatılmışken dış terminalden tekrar.
Probe sonunda authoring enabled/visible durumları geri yüklenir; runtime reset,
frame0 kalır; yeni test domain/source disabled/hidden kalır, proje kaydedilmez.


## Alt adım sönümü — yeni kaynak düzeltmesi

Canlı kuru matris sırasında kaynakta bulundu: FluidDomainStep saf granular
velocity/affine multiplier için n'inci kök kullanıyor; MatterGpuStep ortak
subcycle aynı multiplier'ı n kez uyguluyordu. Örneğin affine=.98, n=17:
outer-step hedef .98 yerine .70932; velocity=.999 için .98314 yerine .999.
GranularStepPolicy::substepDamping ortak yardımcı oldu; saf yol ve iki Matter
lane aynı outer-step çarpımını korur. Sayısal stiffness/CFL kaynaklı ilave
sönüm kaldırılır; malzeme Young/friction/cohesion, sleep veya pressure
parametreleri değiştirilmedi. Yeni kullanıcıya açık operasyon yok;
mevcut UI/Python/IPC authored parametreleri aynı çekirdeğe ulaşır.

Per-outer-step authored damping semantiği korunur; fiziksel saniye başına
normalizasyon henüz yapılmadı. Bu nedenle dt yakınsamasının tek nedeni veya
tam çözümü olduğu iddia edilmez. Sabit sleep threshold ve dış kuvvetlerin
outer-step kick'i sonraki incelemede ayrı değerlendirilmeli.
Shader/ABI/serializer/cache schema değişmedi; davranış değiştiği için eski
bake kabul karşılaştırmasında yeniden üretilmeli.
GPU/wet/pore/stages statik kontrolleri ve bağımsız float32 çarpım referansı PASS.
C++ regression source: scripts/test/matter_granular_damping_test.cpp;
test target derlenmedi. Kaynak değişikliği halen çalışan eski binary'yi etkilemez;
kullanıcı C++ build ve kuru/ıslak problarını tekrar çalıştırmalı.


## Kuru matris canlı sonuç — sönüm düzeltmesi öncesi binary

Dört koşu × 8 s tamamlandı. Sonuç: sampled dry mass/count, pore=0,
invalid=0, authored Young2 MPa ve CFL kapıları PASS; dört yerleşme kapısı RED.

| Koşu | Kuru kg | COM y (m) | RMS (m) | Son2s COM aralığı (mm) | Son2s RMS aralığı |
|---|---:|---:|---:|---:|---:|
| h=.1, dt=1/60 | 72.000001 | .07254 | .40804 | 1.370 | %5.398 |
| h=.1, dt=1/120 | 72.000001 | .06248 | .30880 | 5.744 | %0.723 |
| h=.08, dt=1/60 | 71.987198 | .06962 | .40574 | 2.196 | %4.858 |
| h=.1, 144 kg | 144.000002 | .07916 | .42163 | .283 | %6.011 |

Dt farkı: RMS %24.321, COM y %13.866 -> %5 yakınsama kapısı RED.
Grid farkı: RMS %0.566, COM y %4.024, kg %0.0178 -> snapshot eşiği içinde;
iki yığın da yerleşmediği için statik grid yakınsaması kabulü verilmez.
Düşük kinetik enerji yerleşme kanıtı değil; son2s tüm koşuların sampled max
Ekin/kg değeri .001 J/kg altında, şekil değişmeye devam ediyor.
144 kg koşusunda profil büyümesi sınırlı; aynı spawn hacminde iki kat parcel
iki kontrollü kolon/yük desteği deneyi yerine geçmez. Rest-volume/başlangıç
packing ve temas davranışı ayrı incelenmeli; render sphere DEM teması değildir.

Authoring geri yüklendi; frame0/paused, test domain/source disabled/hidden.
Ham sabit baseline: matter_g2_dry_matrix_before_damping_2026-10-05.json.
Sonraki adım kullanıcı normal C++ build (shader yok), nötr kuru/wet kontrolü
ve aynı dört koşunun tekrar ölçümü. Yeni sönümün canlı etkisi henüz ölçülmedi;
per-outer-step damping, sleep eşikleri, force kick ve fiziksel başlangıç hacmi
kalan inceleme başlıklarıdır. G2/H1 açık.


## Alt-adım çarpımı build sonrası kuru matris

Kullanıcı yalnız Sand preset/E200 kPa sahnesini açtı; Water emitter disabled,
Fluid count0, Sand10323, frame250/paused. Bu sahne authoring'i korunarak ayrı
E2 MPa/dilatancy0 bağsız dört koşu yeniden ölçüldü. İlk baseline'dan domain/source
indeksleri farklı; eski-yeni şekil farkı tek başına nedensel A/B sayılmaz.
Aynı yeni matris içindeki dt/grid karşılaştırması esas alınır.

| 8 s son ölçüm | Kuru kg | COM y (m) | RMS (m) | Son2s RMS aralığı |
|---|---:|---:|---:|---:|
| Referans | 72.000001 | .07495 | .43035 | %6.157 |
| dt yarım | 72.000001 | .06119 | .31556 | %0.663 |
| h=.08 | 71.987198 | .07054 | .42946 | %4.623 |
| İki yük | 144.000002 | .08118 | .44387 | %6.481 |

Kuru sampled mass/count/pore0/CFL/invalid0 PASS; dört yerleşme kapısı RED.
Dt RMS farkı %26.673, COM %18.358; grid RMS %0.208 fakat COM %5.887 ->
yakınsama eşiği RED. Önceki düzeltme tek başına G2'yi kapatmadı.
Authoring restore canlı doğrulandı: Water disabled, Sand enabled; frame0/paused.
Sabit kayıt: matter_g2_dry_matrix_before_time_scaling_2026-10-05.json.

## Yeni kaynak partisi: granül sönümünü fiziksel zamana ölçekleme

Aynı outer-step retention 120 Hz'de saniyede iki kat uygulanıyordu; alt adım
n'inci kökü bu farkı çözmez. GranularStepPolicy::timeScaledSubstepDamping:
`factor = multiplier^(frame_dt / (1/60) / substeps)`.
1/60 referans çarpımı korunur; 24/60/120 Hz eşit sürede aynı granül velocity ve
affine retention verir. Pure granular ve Matter granular lane ortak yardımcı.
Liquid lane eski per-outer-step semantiğini korur, yalnız subcycle çarpımını
koruyan n'inci kök kullanır. APICSolverParams yorumları granül birimini açıklar.
Bu numerical politika mevcut preset/core yolundadır; yeni authoring işlemi yok.
Young/friction/cohesion/sleep/force kick ayarları değiştirilmedi.
Sleep eşikleri, dış kuvvet kick'i, taban teması ve başlangıç rest-volume packing
hâlâ ayrı inceleme başlıkları; bu değişiklik tam yakınsama kanıtı değildir.
Shader/ABI/schema yok; kullanıcı C++ build bekler. GPU/wet/pore/stages kontratları
ve bağımsız float32 eşit-zaman retention referansı PASS (max error5.16e-5).
C++ damping regression 24/60/120Hz ×1/9/17/43 substeps kaynak hazır;
test target derlenmedi. Build sonrası kuru matris aynı komutla tekrar.


## Ağır çekim / hafif tane gözlemi — yeni time-scaled binary

Orijinal frame103/paused: 4254 canonical Sand, dry850.800013 kg, pore0;
Water emitter disabled; friction35°, cohesion/tensile0, E200 kPa, forcefield yok.
Bu küreler MPM taşıyıcı/virtual grain görünümüdür; DEM bireysel kütle/yuvarlanma
kanıtı vermez. Kütleyi keyfi yükselterek hissi düzeltme yapılmadı.

Dış IPC tek taşıyıcı ve64 taşıyıcı, dt1/60 ve1/120; .2 s temassız COM düşüşü.
Aynı Sand E200 kPa preset, kapalı ama uzaktaki sınırlar, collider geçici kapalı;
her koşuda yaklaşık9.536/9.560/9.742/9.746 m/s² aşağı ivme. Önceden belirlenmiş
9.81 ±%5 gate4/4 PASS; sayısal sönüm/air drag mevcut, tam vakum g testi değildir.
Bu kısa serbest uçuş testidir; collider sekmesi/yuvarlanması/yığın kabulü değildir.
Probe: rt_g2_free_fall_ipc.py; ham kayıt matter_g2_free_fall_2026-10-05.json.

Orijinal sahnede3 dış IPC step(dt1/60): solver total57.728/227.440/21.322 ms,
656 dispatch/step; simüle edilen süre16.667 ms. IPC wall571/250/48 ms ayrıca UI,
transport ve scheduling içerir; solver FPS veya normal playback hızı diye
kullanılmaz. Üç kısa/warm-up örneği hesaplama maliyetini gösterir, uzun C7
performans kabulü değildir. matter_dry_slow_motion_scene_2026-10-05.json ve
matter_dry_slow_motion_timing_2026-10-05.json.

Time-scaled binary kuru8 s matris: dry mass/count/pore0/CFL PASS;
yerleşme4/4 RED. Dt RMS farkı%24.995, COM%16.209 -> RED.
Grid RMS%0.556/COM%2.868 snapshot toleransında; yerleşmedikleri için statik
konverjans değil. Son2s RMS aralığı reference%5.762, dt-half%0.707,
grid%4.581, 144kg%6.543; dt-half COM5.0 mm hâlâ düşüyor.
Kayıt: matter_g2_dry_matrix_time_scaled_2026-10-05.json.

## Yeni kaynak: boş sıvı lane işlerini atlama

MatterGpuStep, canonical model scan ile has_fluid belirler; pure Granular
pore-enabled domain ortak granular solver'da kalır. Boş fluid occupancy,
gradient/P2G/boundary, FLIP snapshot, viscosity/pressure, intermodel contact,
fluid gather/tail ve fluid particle readback atlanır. Üç persistent fluid
velocity grid'i önce sıfırlanır; eski su alanı yayınlanmaz. Granular algoritma,
dt/CFL/sönüm ve kimlik/kütle publication korunur; pore exchange yine çalışır.
Sıvı emisyonu veya drainage birth sonrası sonraki adım canonical scan tekrar
sıvıyı görür, tam ortak yol etkinleşir. İki lane allocation/budget korunur;
bu aşama buffer memory tasarrufu veya pure-fluid pruning iddiası değildir.

GPU flag pressure_on_gpu=false ve status granular-only gerçek yapılan işi
raporlar; mixed_transport_ready ortak backend hazır olduğu için true kalır.
UI/Python/IPC aynı raporu okur; yeni authoring işlemi/schema/shader yok.
GPU/pore/wet/stages source checks ve3 probun Python AST PASS. Normal kullanıcı
C++ build ve canlı hız/eşdeğerlik/empty→water→empty tekrar bekler.
--expect-empty-fluid-skipped opsiyonu freefall/dry matrix/dry-wet problarına
eklendi; dry-wet son dry_after_water koluyla sıcak buffer geçişini de test eder.
Önceki binary'nin performans/şekil sonucu yeni pruning sonucu sayılmaz.
Authoring restore ayrıca IPC ile doğrulandı: Sand enabled, Water disabled;
frame0/paused, test domain/source disabled/hidden, proje kaydedilmedi.
G2 temas/rest-volume packing/sleep/force subcycling ve H1 gerçek tane fiziği açık.


## 2026-10-05 — boş sıvı hattı kullanıcı build / canlı kabul

RAM cache kullanıcı izniyle temizlendi. Orijinal Pore Water OFF sahne legacy
pure-granular yolundadır; pruning ortak pore-enabled Matter yolunu hızlandırır.
Freefall 1/64 taşıyıcı ×60/120 Hz 9.536–9.746 m/s², 4/4 PASS.
Dry→water-only/wet→dry-after-water mass/nötr başlangıç ve tekrar kuru yayın PASS;
su max sapma .887 mg, Sand sabit. Kuru pressure=false/contact=0; su geldiğinde
pressure/contact yeniden aktif. Bu reset/replay geçişidir; tüm drainage geçişlerinin
veya C6 persistence'ın tam kabulü değildir.

8 s matris dört kol kütle/CFL/boş hat kapıları PASS, yerleşme dört kol RED.
Reference son RMS .429034 m; dt-half RMS fark %27.28, COM-y %19.08.
Grid-finer RMS %1.88, COM-y %5.18. Denge kurulmadığından repose kabulü yok.
Reference dispatch eski 1810→453, dt-half 962→245, finer 2340→583.
Yeni median total_ms sırasıyla 16.29/21.73/16.71/15.92 (reference/dt/finer/load).
Arşiv koşulları kontrollü C7 A/B değildir; dispatch azalması hız yüzdesi değildir.
Allocation/bütçe azaltılmadı, legacy pure-granular fizik değişmedi.

Sıradaki araştırma temas, outer-step force entegrasyonu, sleep ve rest-volume
packing; dt etkisinin sebebi henüz kanıtlanmadı. G2 açık, H1 DEM ayrı açık.
Original authoring geri yüklendi, frame0/paused; test alanları disabled/hidden,
proje kaydedilmedi. Yeni build bekleyen kaynak değişikliği yok.
Kanıt: matter_g2_dry_matrix_empty_lane_skipped_2026-10-05.json;
matter_g2_free_fall_2026-10-05.json; matter_g2_dry_wet_compare_2026-10-05.json.
