# Matter C5 devir notu — 2026-10-05

## Yetki ve durum
Kullanıcı C5'i tek büyük kaynak partisi olarak istedi; derlemeyi kendisi yapar.
Codex C++/shader derlemesi, uygulama açılışı ve canlı IPC testi çalıştırmadı.
Son kullanıcı derlemesi C5 öncesi başarılıydı. Bu C5 kaynak partisi yeni ve henüz
kullanıcı tarafından derlenmedi. Alt ajan çalıştırılmadı. Commit/push yapılmadı;
çalışma ağacında önceki C4/H1 değişiklikleri de vardır, korunmalıdır.

## Uygulanan kaynak partisi
- MatterPoreExchange.h/.cpp + sim_matter_pores.comp: Vulkan hücre sahipliğinde
  Darcy benzeri hidrolik yük farkıyla emilim/drenaj. CPU yalnız hücre CSR dizini,
  doğrulama ve kare yayınını yapar; canlı CPU emilim çözücüsü/fallback yoktur.
- Kanonik FluidParticles dört sidecar taşır: pore_water_mass_kg,
  pore_capacity_kg, pore_porosity, pore_water_energy_j. Saturation mass/capacity
  oranıdır. Clear/reserve/emit/swap-remove/copy/resize ve bellek hesabı bağlıdır.
- Granül transport kütlesi kuru kütle + gözenek suyudur; ayrı GPU transport
  fraction buffer'ı kanonik kuru rest_mass/mass_fraction değerlerini korur.
  Exchange kapatıldığında mevcut gözenek suyu yine taşınır.
- Drenaj yeni kimlikli kanonik Water/Fluid parçacığı üretir. Havuzda yer yoksa
  su gözenekte tutulur. Islak taşıyıcılar drenaj slotlarında önce gelir.
- Kütle, momentum ve su termal enerjisi kontrolü geçmeden aday kare yayınlanmaz.
  Ledger Absorption/Drainage olayları yalnız kanonik commit sonrasında yazılır.
  Kinetik enerji korunumu vaat edilmez: hız karışımı dissipatiftir.
- MatterPoreAuthoring ortak doğrulama/JSON servisi, RtApiMatterPores ortak
  authoring işlemi. UI Matter/Pore Water, Python rt.fluid.set_pore_exchange ve
  IPC fluid.set_pore_exchange aynı API'yi çağırır. Bilinmeyen anahtar/tür/aralık
  hataları yazmadan reddedilir; ıslakken porosity değişimi reddedilir.
- fluid.matter_models.pore_exchange: settings, measured/held/status, depolanan
  su/kapasite/max saturation, son absorbed/drained, drainage births, slot
  bekleyen taşıyıcı sayısı ve korunum residual'ları.
- Serializer + proje JSON ayarları, cache v11 dört sidecar ve ayar fingerprint'i.
  Eski v10 bake okunmaz; yeniden bake gerekir. Emitter pool weight de iki cache
  fingerprint yoluna eklendi. Eski projelerde C5 varsayılan kapalıdır.
- Vcxproj/filters, shader batch ve Vulkan kayıt: 13 binding, 40 byte constants.
  sim_matter_pores.spv kullanıcı shader derlemesinde üretilmelidir.
- IPC descriptor yeniden üretildi: 656 yöntem / 637 belgelenmiş yöntem.

## Kesin kapsam / açık kabul
Bu C5 kaynak teslimidir; fiziksel kabul kapatılmadı. İlk kapsam Closed Vulkan
Matter, kanonik büyük/küçük harfe duyarlı Water + Sand/Gravel/Soil, mobil
çözülmüş Fluid/Granular rejimleri. Frozen/Elastic desteklenmez. İlk kabulde
reaksiyon/yanma/faz geçişi ve collider kullanılmamalıdır.

Exchange kare başına bir hücre-local finite-volume adımıdır; tam gözenek basıncı
çözümü veya komşu-hücre temas stencil'i değildir. Hücre başına 1024 taşıyıcı
sınırı GPU iş yükünü sınırlar; aşımda kare tutulur. Drenaj konumu taşıyıcının
0.5 voxel altı, domain tabanı üstüne clamped; collider-aware birth kabulü açık.
Her drenaj taşıyıcısı/kare yeni parcel üretir: uzun koşuda havuz dolabilir ve su
pores içinde kalır. Birth birleştirme/geri kullanım optimizasyonu açık.
C6 saturation -> friction/cohesion/effective stress/pore-pressure ve ıslak
materyal görünümü uygulanmadı. Render seçimi/substance materyalleri önceki
partiden mevcut; C5 yeni ıslak görünüm yapmaz.

Önceki C4 canlı testinde source adları water/sand ve model override kullanıldı;
kanonik katalog Water/Sand testine eşdeğer değildir. Rejim/GPU yolu gözlendi,
katalog fiziği C5 kabulünde yeniden doğrulanmalıdır.

## Yapılmış kontroller (derleme değil)
python scripts/test/check_matter_gpu_contracts.py — PASS (13 GPU ABI)
python scripts/test/check_matter_transfer_contracts.py — PASS (cache v11)
python scripts/test/check_matter_solver_stages.py — PASS
python scripts/test/check_matter_pore_contracts.py — PASS
Yeni dış probe Python AST kontrolünden geçti; çalıştırılmadı.
Bu kontroller C++/GLSL derlenebilirliğini veya fiziksel yakınsamayı kanıtlamaz.

## Kesin sonraki iş / kullanıcı derleme sonrası kabul
1. Kullanıcı normal shader + Release derlemesini yapar (yeni pores SPV dahil).
2. Eski bake temizlenir. Closed Vulkan Matter'da kanonik Water + Sand,
   yeterli boş particle slotu, collider/yanma/faz geçişi olmadan temas sahnesi.
3. Pore Water etkin: porosity 0.35, permeability 1e-8 m2, viscosity 0.001,
   gravity 9.81, drainage_scale 0. Birkaç temas karesi, sonra pause.
   Uzak kuru kumda gözenek suyu oluşmaması ayrıca görsel/konum incelemesiyle
   doğrulanır; mevcut aggregate probe uzaktaki taşıyıcıyı tek başına kanıtlamaz.
4. Harici terminal: python scripts/test/rt_test_matter_pores_ipc.py "DOMAIN"
   --expect absorption. Emission durmuş iki snapshot'ta serbest su kaybı =
   pore water kazancı ve toplam water korunumunu ayrıca karşılaştır.
5. Drainage scale 1, ilerlet/pause; probe --expect drainage. Serbest Water
   births, pore azalması, kapasite 0..1 ve ledger conservation izlenir.
6. Havuz dolu durumda drenaj kaybı yok; disable/enable wet transport; save/load
   + bake/scrub sidecar/kimlik round-trip; iki dt ve çözünürlük yakınsaması.
7. Hata varsa önce C5 düzelt, ardından C6 büyük parti. C7 kalite kabulü hâlâ açık.

Uygulama açık olsa bile embedded script workspace'de IPC probe çalıştırma.
Windows named-pipe error 5 alırsan harici Python probe'u sandbox escalation ile
tekrar çalıştır; yeni transport icat etme. AGENTS.md gereği kullanıcı bu turda
istemeden uygulamayı açma veya herhangi bir derleme çalıştırma.

## Önceki ertelenmiş kalite notları
Domain taşıma/splat ve domain sınırı dışı çizim cache kozmetiği; gas varken SDF
liquid body materyalinin bazı karelerde default'a düşmesi; preset gas RT
kararlılığı son kalite testlerinde açık. Bunlar C5 kabulüyle çözülmüş sayılmaz.


## 2026-10-05 canlı C5 incelemesi — kullanıcı derlemesinden sonra
Kullanıcı açık ikili emitter sahnesini ölçmeye izin verdi. Uygulama açılmadı,
derleme yapılmadı; dış scripts/test/rt_ipc.py ile named-pipe escalation kullanıldı.
Sahne C4_Mixed_GPU, Closed Vulkan, 32^3, 50.000 particle limit. İlk durumda
frame 0/boş runtime, C5 disabled; C4_Sand emitter aslında Soil (Granular),
C4_Water emitter küçük harf water (Fluid). Soil için domain custom cohesion
1807 Pa, Young 706400 Pa; bu kuru Sand kabul sahnesi değildir.

Mevcut ayarlar ile ilerletmede GPU üç stage etkin, held=false; temas sayaçları
199,529,1562 oldu. Contact pair sayısı kuvvet transferinin tüm doğruluğunu,
batmayı veya hidrostatik buoyancy/drag kabulünü kanıtlamaz.

Test sırasında Water emitter etiketi Water'a düzeltildi; aynı Obj_1_Material /
SDF binding Water için eklendi (eski water binding korunuyor). Soil ve kaynak
konumları değiştirilmedi. C5 enabled, permeability=1e-8, drainage_scale=0 ile
emilim; sonra drainage_scale=1 ile drenaj denendi. Bu ayarlar sahnede kaldı,
dosyaya save yapılmadı. Son playhead 68, playing=false.

Kaydedilmiş canlı örneklerde pore water 0.0134 -> 0.0354 -> 0.2959 -> 3.3068 kg,
max saturation 0.3204; absorption last-step 0.4237 kg. Mass residual yaklaşık
-7.2e-8 kg, momentum residual 4.6e-7 kg m/s. Drenaj örneği: 0.00049856 kg /
66 yeni Water parcel, mass residual -8.1e-9 kg. Bunlar anlık solver örnekleridir,
aynı durdurulmuş sistemin before/after bilançosu veya çözünürlük kabulü değildir.
Timeline set_frame asenkron resync yapar; config değişimi cache/resync tetikler.
JSON'daki frame requested playhead'dir; her örneğin solver yaşını garanti etmez.
Örnekler: matter_c5_live_2026-10-05.json.

Önemli açık sorun: sonraki ölçümde 50.000 limit doldu: 45.425 Fluid + 4.575
Granular. 3.524 pore taşıyıcısında drainage slotu yok, drained_kg=0; su tutuldu.
Kararlı son query öncesi/sonrası sim.control_state epoch=0, frame=68,
playing=false idi. Önceki query ile bu query arasında çok sayıda drainage birth
oluştu; timeline-resync/step sayımı sonraki kontrollü testte izlenmeli.

Harici mixed GPU --contact probe PASS (78.098 contacts). C5 --expect drainage
probe son karede FAIL: havuz dolu olduğundan drained_kg=0, bu aşamada beklenen
budget koşulu; önceki kayıtlı örnekte drenaj doğumu görüldü. C5 genel finite /
saturation / GPU publication probe ayrıca çalıştırıldı. C5 tam kabul açık.

## Öncelikli sonraki düzeltmeler
1. FluidPhysicalMass.cpp densityForParticle yalnız liquid_density kullanıyor.
   Sand/Gravel/Soil bu alanı override etmiyor (default 1000); dry density alanı
   Sand 1600, Gravel 1750, Soil 1450. Gözlenen Soil 0.125 kg/parcel, Water
   ~0.124625 kg/parcel. Rejim-aware mass initialization yapılmalı: Granular dry
   density, Fluid liquid density; exact phase-transfer masses korunmalı. CPU/
   GPU ortak helper, legacy Auto fallback ve yeni-emission yolları denetlenmeli.
   Bu turda fizik kaynak kodu değiştirilmedi; mevcut exe bu kusuru taşıyor.
2. C5 tiny drainage parcel her taşıyıcı/her step üretimi havuzu hızlı dolduruyor.
   GPU hücre/model bazlı birleştirme veya kontrollü birth batching/reuse tasarla;
   mass/momentum/thermal energy/ID/cache/ledger korunumu birlikte sürmeli.
3. Fluid-only pressure ile unilateral Coulomb grid-contact tam hidrodinamik
   drag/buoyancy doğrulaması değildir. C4 roadmap iki yönlü contact/drag kabulünü
   (ayrı referans, yoğunluk kontrastı, submerged bed, pressure/drag transfer)
   açık tut. C6 wet effective stress/cohesion uygulanmadan batma kabulü verme.
4. Yukarıdakilerden sonra gerçek canonical Water+Sand (zero cohesion) paused
   isolated test; emitters stop, epoch/time checked balances ve dt/resolution.


## 2026-10-05 C5 yoğunluk ve drenaj kaynak düzeltmesi
- FluidPhysicalMass artık explicit Granular için profile.density (kuru yoğunluk),
  explicit Fluid için liquid_density kullanır. Auto CPU/GPU partition gibi domain
  legacy rejimini izler. Domain step ve thermal/mist/combustion ilk-kütle yolları
  legacy bilgisini geçirir. Geçerli rest_mass ve phase-transfer kesin kütleleri
  yeniden yazılmaz. Dolayısıyla mevcut dolu runtime reset edilmeden düzelmez.
- GPU drenaj çıkışı hücre başına nominal rho_water * h^3 / 8 kg ile sınırlıdır.
  Aynı hücrede bunun altında kütleye sahip Fluid/Water varsa onun boş kapasitesi
  kullanılır; yoksa hücre başına tek yeni slot ayrılır. Islak hücreler önceliklidir.
  Slot/alıcı yoksa su gözenekte tutulur. Küçük su için minimum bekleme eşiği yoktur.
- GPU hidrolik exchange hesaplar; host yalnız GPU kayıtlarını hücre bazında
  toplar ve kanonik kareyi yayınlar. Kütle, momentum, termal enerji gate'i ve
  taşıyıcı başına ledger korunur. Refill mevcut particle_id'yi korur; yeni batch
  tek yeni kimlik alır. Mevcut Water konumu refill sırasında korunur; yeni batch
  kütle ağırlıklı taşıyıcı konumunun 0.5 voxel altında doğar (taban clamp sürer).
- Ortak fluid.matter_models raporuna drainage_refills eklendi (Python/IPC ortak
  API); drainage_births yalnız gerçek yeni kimlik sayısıdır. Dış probe artık
  drained_kg > 0 için births + refills > 0 koşulunu denetler.
- Cache mass-policy ve pore-settings fingerprint revision değişti. v11 sidecar
  biçimi değişmedi. Eski bake yeniden üretilmeli; eski SPV bu kaynakla uyumlu
  davranışı sağlamaz, normal kullanıcı shader derlemesi zorunludur.
- Havuzun hiç dolmayacağı garanti edilmez: hareketle hücre değiştiren su veya
  nominal kütleye ulaşan alıcı yeni slot gerektirebilir. Dolu havuzda yalnız uygun
  yerel alıcısı olan hücreler drenaj yapabilir. Collider-aware birth hâlâ açıktır.

### Kullanıcı derleme ve kabul listesi
1. Normal shader + Release derlemesi; sim_matter_pores.spv yeniden üretilsin.
2. Eski bake/runtime reset; Closed Vulkan, kanonik Water + Sand, zero cohesion,
   collider/reaksiyon/faz geçişi kapalı. Ayrıca Soil yoğunluk karşılaştırması.
   h=0.1, ppc=8 için Sand=0.2, Gravel=0.21875, Soil=0.18125 kg kuru parcel;
   Water kendi liquid_density değerini kullanır. Aktif emitters sonra durdurulsun.
3. Paused epoch/time kontrollü önce/sonra: serbest Water + pore water sabit;
   emilim kaybı=pore kazancı. Uzak kuru kumda su oluşmaması ayrıca incelensin.
4. Drainage scale 1: dış terminalden
   python scripts/test/rt_test_matter_pores_ipc.py "DOMAIN" --expect drainage
   Son karede births veya refills görülmeli. Aynı hücrede küçük alıcı olduğunda
   refill ID korunmalı ve particle count artmamalı. Birçok taşıyıcı aynı hücrede
   drenaj yaptığında en fazla bir birth; birleşen su sıcaklığı/hızı ağırlıklı olmalı.
5. Dolu havuz: uygun küçük yerel Water alıcısıyla refill sürmeli; alıcı yokken
   drenaj suyu gözenekte kalmalı. Save/load + bake/scrub mass/ID round-trip ve
   iki dt/çözünürlük kabulü yapılmalı; yeni cap drenaj hızını etkileyebilir.
6. scripts/test/matter_physical_mass_test.cpp gerçek helper için regression
   kaynağıdır; kullanıcının test hedefinde assert'ler açık çalıştırılmalıdır.
   Codex bu C++ testini derlemedi/çalıştırmadı. Python source contract ve batch
   matematik referansı GPU/C++ yürütmesini veya fizik kabulünü kanıtlamaz.

Bu düzeltme turunda uygulama açılmadı, canlı IPC ve build çalıştırılmadı;
C6 ve fiziksel C5 kabulü henüz tamamlanmış sayılmaz.


## 2026-10-05 kullanıcı derlemesi sonrası kontrollü canlı kabul
Kullanıcı açık sahnede dış IPC ölçümünü ve gerekirse kabul sahnesi kurulmasını
yetkilendirdi. Uygulama açılmadı, build çalıştırılmadı. İlk okuma frame=250,
paused, C4_Mixed_GPU runtime=0 particles, C5 disabled; C4_Sand gerçekte Soil,
cohesion=1807 ve iki kaynak sınırsızdı. Bu ilk durum korunum kabulü sağlamaz.
Özgün domain/source ayarları ölçüm JSON'una kaydedildi; silinmedi. Eski domain
ve kaynaklar disabled, domain görünürlüğü kapalı olarak açık projede duruyor.

Yeni C5_WaterSand_Acceptance: Closed Vulkan Matter, h=0.1, 32^3; kanonik Sand
ve Water, cohesion/tensile=0, Young=50000 Pa, solid phase ve thermal liquid
kapalı. Pore porosity=0.35, permeability=1e-8, drainage_scale=1. Kaynaklar
0..0.3 saniye sonlu; Sand/Water temas kaynağı başına 360, uzak kuru kontrol
Sand kaynağı 60 parçacık üst sınırı. Su/kum 293.15 K ve ilk hız sıfır.
Uzak kuru kontrolün su almaması aggregate IPC ile ayrı ayrı kanıtlanmadı.

Dış fluid.step gerçek SimulationWorld scheduler'ını 205 kez dt=1/60 ile
çalıştırdı. Timeline frame=0, paused, epoch=0 kaldı; bu 205 timeline karesi
olarak raporlanmamalıdır. Sonlu kaynaklar 25. adımda kotasını doldurmuştu.
Sonraki 180 adımda hiçbir config değişmedi. Her 10 adımda alınan son 120-adım
ölçümlerinde step_held/pore held görülmedi.

| Bilanço | 25. adım | 205. adım |
|---|---:|---:|
| Serbest Water kg | 43.514443790 | 42.041961949 |
| Pore water kg | 1.350558031 | 2.823039162 |
| Toplam su kg | 44.865001821 | 44.865001111 |
| Kuru Sand kg | 84.000001252 | 84.000001252 |
| Parçacık sayısı | 949 | 1234 |

Toplam su değişimi -7.095e-07 kg; örneklenen maksimum
sapma 7.295e-07 kg. 420 Sand kuru parcel başına yaklaşık 0.2 kg:
yoğunluk düzeltmesi canlıda gözlendi. İlk 360 Water yaklaşık 44.865 kg.
Son adım drenaj 0.030523216 kg, 1 birth + 129 refill, budget blocked=0;
980 contacts, P2G/pressure/G2P GPU ve held=false. Dış pore drainage, mixed
GPU contact, model identity/transfer probları PASS. Parçacık sayısı sıfır
büyüme vaadi değildir; bu aralıkta +285, son 120 adımda +44.

Bu sonlu kabul sahnesinde su üretimi gözlenmedi. Uzun havuz-dolum, collider,
save/load/bake kimlik round-trip, iki dt/çözünürlük ve spatial dry-control
kabulü henüz kapanmadı. C6 fizik kabulü yapılmadı.

Kullanıcı açık projeyi farklı bir adla kaydedebilir; Codex project.save yapmadı.
Sahne paused bırakıldı. Tekrar: timeline başlangıcına dön/reset ve play; üç
finite emitter otomatik durur. Harici yeniden kurulum/ölçüm aracı:
scripts/test/rt_c5_acceptance_scene_ipc.py --setup --steps 25
sonra --steps 60. Setup açık sahnede kaynak/domain ayarlarını değiştirir.
Ham kayıt: docs/dev/matter_c5_acceptance_live_2026-10-05.json.


## 2026-10-05 görsel kabul sahnesi ve eski output binding temizliği
Kullanıcı tekrar kullanılabilir sahnenin görünümünü ve Matter Output'da kalan
küçük harfli water kaydını bildirdi. Canlı query: C5 kabul domaininde Sand/Water
iki binding; üçüncü water eski C4_Mixed_GPU domainindeydi. Binding emitterden
bağımsız saklanıyordu, silinmiş fiziksel madde veya yeni su üretimi değildi.
Mevcut fluid.set_substance_material(domain="C4_Mixed_GPU", substance="water")
ile eski binding kaldırıldı; eski domain Soil/Water, yeni domain Sand/Water.

MatterModelControls.cpp içine Remove output binding düğmesi eklendi. UI mevcut
setFluidSubstanceMaterial ortak API'sinin tüm opsiyonları boş silme yolunu
çağırıyor; Python rt.fluid.set_substance_material ve IPC zaten aynı işlemle
parite sağlar. Emitter referansı varsa madde fallback görünümüyle listede kalır.
Yeni düğme henüz derlenmedi; mevcut exe'de canlı temizlik API ile yapıldı.
Mevcut transfer/pore source contract kontrolleri PASS; build yapılmadı.

Açık sahne: C5 Sand Preview sıcak kum rengi, C5 Water Preview mavi; nötr gri
zemin ve yakın kamera. Eski Default_Cube kadraj dışına taşındı (silinmedi).
Zemin collider değildir; fizik Closed domain duvarlarıyla sürer. Orijinal objelerin
transformları docs/dev/c5_visual_original_transforms.json içinde saklandı.
SDF denemesinde bu seyrek/düşük çözünürlüklü su dağılımında bütün su açıkça
izlenemedi; test görünümü Splat bırakıldı. Bu sahne sayısal/tanısal fixture'dır;
fotogerçekçi ıslak kum veya C6 wet appearance kabul sahnesi değildir.

Görsel ayarlardan sonra reset + 85 dış scheduler adımı; 420 Sand + 770 Water,
toplam su yaklaşık 44.865002316 kg; son drenaj 0.045462139 kg / 0 birth,
129 refill, held=false. Pore --expect drainage PASS. Yeni reset koşuları ham
logda önceki 25..205 bilançosundan sonra eklenmiştir; önceki summary özgün
koşunun ölçümüdür. Son görüntü c5_acceptance_visual.jpg; son açık sahne ayarları
c5_visual_scene_state.json. Kullanıcı kaydeder; Codex project.save çağırmadı.
