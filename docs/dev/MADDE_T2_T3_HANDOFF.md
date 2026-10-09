# Madde T1/T3 kabulü ve T2 devam noktası

> **Durum:** AKTİF — 2026-10-08. Kullanıcı build'i başarılı; T1/T3 hızlı dış IPC kabulü PASS. T2 kısmi, tam T3 kabulü açık.

## Son devam: sparse pressure GPU kaynağı

Güncel devam: P2G/G2P/FLIP transfer/copy ve mixed MAC clear/contact dengeli 2D
dispatch'e geçti; padded group UInt lane çarpımından önce elenir. Source ABI
denetimi PASS; C++ dispatch unit ve büyük-case GPU kabulü açık. Core revision 7.
DEM agent'in sim_matter_grain.glsl/MatterGrainGpu.cpp dosyalarına dokunulmadı.
Canonical MAC/FLIP/gas hâlâ dense; bu yalnız ölçek hazırlığıdır. Derleme/canlı
test başlatılmadı. Son source/test listesi NEXT_BUILD_CHECKS.md içinde.

Sonraki kaynak partisi: SparseViscosityGpu ile üç MAC RHS kopyası aktif yüz
tile'larına taşındı; normal liquid + mixed Matter aynı classify/relax shader
gövdesini kullanır. Dört yeni shader 13/68, dense giriş 11/52; core revision 6.
UI/script/IPC ayrı viscosity pool ölçüsü verir. Kaynak kontratları PASS;
kullanıcı build + --viscosity dense/sparse probe açık. Canlı test başlatılmadı.
Canonical velocity/P2G/FLIP/gas hâlâ dense; tam sparse storage kapanmadı.

Vulkan pressure scratch tile pool ve on yeni shader (16/80) bağlandı. Statik
kontratlar PASS; kullanıcı build + dense/sparse parity henüz açık. Tam MAC/gaz
storage kapanmadı. Ayrıntı: [MATTER_SPARSE_TILE_GRID.md](MATTER_SPARSE_TILE_GRID.md).

## 2026-10-08 dinamik sıvı desteği ve toplu kabul — son kaynak checkpoint

Kullanıcı kararı: devam eden testleri kesmeden kaynak partilerini birleştir;
fizik, yakınsama ve maliyeti tek sıralı komutta ölç. Bu devam kaynakta, derlenmedi.

- CFD-DEM desteği her ortak tick'te canonical GPU konumlarından yeniden binlenir;
  Di Felice, trilinear lump partition, pore voidage/submergence ve Archimedes
  hesabı CPU referansıyla aynı. Tepki aynı tick'in PRE-contact ağırlıklarıyla
  dağıtılır; hareket eden su eski destek hücresine bağlı kalmaz.
- Solid volume ve tepki sekiz fiziksel destek hücresine GPU CAS ile scatter
  edilir. Float-atomic extension/komşu kırpma/retry ceiling yok. Dry-cell voidage
  de korunur; tamamen kuru taşıyıcıda gereksiz komşuluk taraması yapılmaz.
- İlk hash temizliği sonrası yalnız önceki touched buckets temizlenir. Coupling
  tamponları domain runtime'ında tutulur; gerçek allocated byte budget'a yazılır,
  domain release'te bırakılır. Backend değişiminden önce grid hazırlanarak ödünç
  grain runtime referansının geçersizleşmesi önlenir. Partikül cap eklenmedi.
- Yeni shader ABI 21 storage / 96 push byte; sekiz sim_grain_fluid_* aşaması
  (clear/hash/cells/solid/refresh/delta/reaction/apply). API/UI support readback
  device_rebin_each_tick. Islak su transferi son canonical konumda yeniden binlenir.
- Dış probe suyun uzak hücreden grain desteğine girip çıkmasını, eşit-zıt dürtüyü
  ve üç sahibin korunmasını ölçer. --kernel-timings native GPU sürelerini ayrıca
  toplar; host wait GPU kernel süresi diye sunulmaz. Profiler ayarı geri yüklenir.
- Tek komut: python scripts/test/rt_test_matter_acceptance_ipc.py --extended.
  Koşular sıralı, tek JSON ve H1 detay snapshot'ları. Extended H1 fixture'ları
  kendi suite davranışıyla disabled kalır. Başka IPC testiyle paralel çalıştırma.

**Açık:** üç-sahip porous projection/pressure reaction, deformasyon+angular
MPM/grain yakınsaması ve cinematic ölçek native GPU maliyeti. Runner bunları
remaining_gates ve all_unified_matter_gates_closed=false olarak raporlar;
küçük regresyon PASS'i ana planı kapatmaz. Test/build şimdi çalıştırılmadı.

## Önceki checkpoint: 2026-10-08 ortak GPU saat

Önceki MPM/grain sürümü kullanıcı build ve dış IPC smoke PASS: 0.063435 N s
temas, 4.66e-10 N s residual, üç ayrı sahip. Bu yeni devam kaynakta; derlenmedi.

- MPM+DEM bulunan domain'de geometri/temas ortak mikro saatle ilerler; continuum
  P2G/pressure/G2P kendi CFL/elastic aralığında yenilenir. Basınç DEM sıklığında
  zorlanmaz. Sıvı ve MPM contact-corrected canonical hızla her tick advect edilir.
- Sıvı topak hızları ve eşit-zıt drag/kaldırma tepkisi GPU'da her tick güncellenir;
  alt adım başına tam particle state readback/upload yok, host ikinci tepki eklemez.
- UI/Python/IPC `runtime.common_clock`: enabled, transport_steps,
  continuum_grid_steps, liquid_reaction_on_gpu, liquid_support.
- Önceki contact modülünün keyfi 1M taşıyıcı/64 komşu retleri kaldırıldı.
  Gevşeme measured graph degree'lerinden türetilir, event sayacı 64 bit, yeni
  dispatch'ler 2-D. Yalnız gerçek shader indeks genişliği/cihaz buffer kapasitesi
  ve açık authored domain bütçesi/alt-adım üst sınırı kontrol edilir.
- Diğer ajanın DEM damping/kapasite/2-D ve render/cache değişiklikleri korunur.

Sınırlar: mevcut CFD-DEM destek hücreleri/weights ve CPU-only kuvvetler frame
başına örneklenir. Üç-owner porous projection, deformasyonla contact support,
angular kabulü ve 24 fps büyük sahne maliyeti açık. Yeni clock C++ test kaynağı
ve dış IPC probe hazır; final kullanıcı build'inden sonra çalıştırılır.

## Önceki checkpoint: 2026-10-08 MPM + grain temas kaynak devamı

Önceki ortak state/sahiplik partisi kullanıcı tarafından derlendi. Bu devamda
`MatterGrainMpmContact` Vulkan yolu eklendi; yeni C++ ve beş shader henüz kullanıcı
build/canlı kabulünden geçmedi. Eski MPM+grain readiness ret kapısı kaldırıldı.
Panel grain açıkken MPM iskeletini gizlemez; transport madde başına dem|mpm'dir.

- Continuum frame'i sonrası her DEM alt adımında iki sahip GPU hashinden aynı
  çifti okur. Ayrı gather/apply dispatchleri, eşit-zıt unilateral normal ve Coulomb
  dürtüsü; 64 komşu sınırı aşılırsa canonical yayın reddedilir. Yalnız metadata
  yüklenir, kompakt dürtüler alınır; temas için tam particle sidecar transferi yok.
- MPM taşıyıcısı sıvı drag/viskozite/su emme alanına katılmaz. Üç sahip birlikte
  çalışır; MPM varken sıvı porous projection kapalı, drag+kaldırma açık. Ortak MAC
  weight'leri MPM'ye ikinci tepki vermesin diye bu sınır UI/IPC'de açıklanır.
- `fluid.matter_models.grain_diagnostics.runtime.mpm_contact`: parcels, events
  (sıfır olmayan çift dürtüleri), grain_impulse_magnitude_n_s, max_neighbours,
  momentum_residual_n_s ve schedule. Aynı core UI/Python/IPC tarafından okunur.
  Grain pile/spin tanısı artık MPM'yi DEM küresi saymaz.
- Dış IPC smoke sıfır olmayan temas, momentum kalıntısı, üç ayrı sahip ve canlı
  transport edit ret kontrolüne çevrildi. Core regresyon MPM'nin sıvı alanından
  dışlanmasını kapsar. Final canlı testler kullanıcıda, uygulama başlatılmadı.

**B11/ana plan kapanmadı:** continuum geometrisi DEM frame'i boyunca end-state'te
sabit; temas hızı DEM alt adımlarında değişir. Ortak üç-sahip zamanlayıcısı,
deformasyonla değişen MPM temas desteği, angular contact/spin kabulü, ayrı porous
owner weight'leri, büyük yığın/24 fps maliyeti ve yüksek hızlı blok yakınsaması açık.
MPM materyal parametrelerinin madde başına solver tüketimi de final denetim ister.
Gaz/grain, karma elastik MPM, termal hal geçişi ve T0/T6 final kapıları açık kalır.
Cache/render prep/SimCache/render bridge dosyalarına dokunulmadı.

## Güncel kaynak checkpoint — 2026-10-08 (bu parti derlenmedi)

Kullanıcı ana birleşik domain planının kalan işlerini topluca onayladı. Bu parti,
önceki kaynak/build ve canlı kabul kaydını değiştirmez; ana plan **tamamlanmadı**.

- Yeni `MatterSubstanceState.{h,cpp}`: kaynak/seed doğum modeli, adım başında
  madde+sıcaklıktan model çözümü, statik tag/frozen ortak engel denetimi.
  Eritilebilir granül/elastik madde erime eşiğinde Fluid modeline geçer; kimlik,
  kütle ve hız korunur; eski deformasyon/stres yeni katılaşmaya taşınmaz.
- Ice ayrı madde olarak korunur (onaylı karar). Soğuk Ice granül, sıcak Ice sıvı
  modelidir. Water'ın havada hareketli buz taşıyıcısına dönüşümü **henüz yok**;
  destekli/pinli donma yolu devam eder. Gizli ısı/mekanik enerji kabulü açık.
- Water/Wax ν(T) artık parçacığın kendi erime eşiği, sıcaklığı, sıcak/soğuk
  viskozitesi ve aralığından hesaplanıp hücreye toplanır. Erime histerezisi de
  maddenin aralığını kullanır; granül/elastik taşıyıcı sıvı eğrisine katılmaz.
- `granular_transport=dem|mpm` kütüphanede, UI/IPC/Python aynı substance
  servisi üzerinden düzenler; kalıtım/override/proje kaydı mevcut ortak yolu
  kullanır. Sand/Gravel/Ice DEM, diğer maddeler MPM; grain kapalıysa MPM kalır.
- Doğum ve eksik kütle tamamlaması yalnız DEM sahibi için küre kütlesi kullanır.
  Soil gibi MPM sahibi sessizce DEM'e çevrilmez. `fluid.matter_models`
  `transport_owners` aynı core'dan dört sahip sayısını ve readiness'i raporlar;
  Active Matter paneli aynı çözümü gösterir.
- T4 **yalnız sahiplik ve koruma kapısı**: MPM+grain birlikteyse temas henüz
  olmadığı için adım açık hatayla tutulur. B11 faz 2/3, MPM↔tane eşit-zıt temas,
  karma elastik MPM ve gaz↔grain bağlantısı **uygulanmadı**. Bu koruma, üç sahipli
  fizik kabulü veya ana plan kapanışı değildir.
- Canlı taşıyıcı varken `granular_transport` düzenlemesi, türetilmiş referansları
  da kapsayarak tüm yama uygulanmadan reddedilir; domain reset gerekir.
- Yeni CPP regresyon kaynağı ve dış IPC ownership-gate testi hazır, çalıştırılmadı.
  Eski wax betiğinin kalmış `preset` readback'i `default_substance` olarak düzeltildi.
  Build ve canlı kabul kullanıcıda. Disk cache/render prep/SimCache/render bridge
  dosyalarına dokunulmadı; yalnız authoring bake imzasına core revizyonu eklendi.

Kaynak denetimi: descriptor üretimi/audit (663 metot) PASS, grain kaynak/ABI
sözleşmesi PASS, Python AST ve yedi Release eşliği PASS, proje/filters/JSON
kayıtları PASS. Bunlar derleme veya canlı fizik kabulü değildir.

Sonraki zorunlu kaynak sırası: T2 hareketli soğuk Water/Ice ve gerçek termal
geçiş bilançosu → B11 GPU MPM↔tane contact ve üç sahip → karma elastik MPM/gaz
bağlantısı → final T0/legacy/save-open/fizik matrisi → eşitlik geçerse T6.

## Devir: sonraki ajan için (2026-10-08, son durum)

**Kullanıcı tercihleri (bağlayıcı):**
- Önce ana planı bitir, testleri en sona bırak. Sık build/test döngüsü hem token hem derleme süresi maliyeti. Ölçüm ve test scripti yazabilirsin ama çalıştırma sırası sonda.
- Build'i kullanıcı alır (CLAUDE.md §2). Kod yaz, derlemeyi bekle; derleme sonucunu kullanıcı bildirir.
- Diğer ajan **fluid disk cache** ve **render hazırlık süresi** üzerinde çalışıyor. O alana (render prep, disk cache, SimCache, render bridge) dokunma; çakışmayı kullanıcıya sor.
- Güncel kullanıcı onayı (2026-10-08): ana birleşik domain planının kalan işleri topluca onaylı. Küçük adımlarda yeniden onay sorma. Önce kaynak planı, testler en sonda; build kullanıcıda.

**Bu oturumda kaynakta yapılanlar (derlendi, hatasız — kullanıcı bildirdi):**
1. `FluidThermalLiquid.cpp` `updateThermalFreeze`: donma/erime eşiği parçacığın maddesinden (T2a). Eritilemeyen madde donmaz; Hata B düzeltildi (eritilemeyen için bayrak hiç yazılmaz/temizlenir).
2. Söküm partisi: akış kaynağı `initial_constitutive_model` ve bağlama `constitutive_model` API/IPC/Python/UI/kayıt/karmadan çıktı. Doğum modeli yalnızca maddeden (`MatterDomainSources.inl`). Bağlama `phase` = statik katı blok işareti (etiket değişti, enum ve tag yolu aynı). Eski anahtarlar okunmaz.
3. Betikler (9 test, iki kopya) güncellendi; descriptor'lar `gen_ipc_descriptors.py` ile yenilendi; `audit_ipc_capabilities.py` OK (663 metot).
4. Ölçüm kapsamları: `sim.matter.gpu_partition_upload` (MatterGpuPartition.cpp), `sim.matter.emit` (MatterDomainSources.inl). Sweep: `scripts/test/rt_matter_perf_sweep.py` (+ Release kopyası). Sweep sahne parametrelerini değiştirir, kaydetmeden çalıştırma.
5. Ice preset (`MaterialStateField.cpp`): model Granül, sürtünme 2.3° (μ≈0.04, GEÇİCİ), E=9.3 GPa, ν=0.32, çekme 1 MPa. Karar: [BUZ_MODEL_KARARI.md](BUZ_MODEL_KARARI.md) onaylı.

**Ölçüm bulguları (ertelendi, kanıt zayıf):**
- Sweep (voxel 0.035, ppc 8): en kötü kare ~1.8 s; hipotez: bellek tahsisi / rebuild. Kanıt yok; disk cache/render prep ajanı bu alana zaten girdi, oradan bakılabilir.
- `capture_frame` parçacık başına 0.73 → 0.96 µs büyüyor: host kopyası şüphesi (H1-C7 ihlali olabilir).
- `sim.matter.gpu_partition_upload` saf su sahnesinde hiç çalışmaz (yalnız mixed/granül) → model yükleme hipotezi karışık sahnede test edilmeli.

**Açık işler (öncelik sırasıyla):**
1. Build sonucu bekleniyor: söküm partisi + Ice preset. Hata çıkarsa ilk bakılacak yer: `setFluidSubstanceMaterial` çağrı sayısı (7 argüman), kaldırılan alanlara kalan referanslar.
2. **Ice MPM maliyeti:** Ice bir domain'in varsayılan maddesi olup granül MPM seçilirse E=9.3 GPa adaptif alt adımı şişirir (`APICFluidStep.inl:304`, `FluidDomainStep.inl:518`). Ölçülmeden Ice'i MPM varsayılanı yapma; grain (DEM) taşıyıcı olarak kullan.
3. Ice ölçümü (kullanıcı onayladı): tek buz parçası ve yığın dökümü (H1 grain sahnesi); statik sürtünme ve repose ölçülecek.
4. Katı blok (statik) ile donmuş bayrağı ayrı kalır (bkz. aşağıda "Eski katı bağlama göçü"). `frozen` ve `solid_substance_tags` iki üretici, tek tüketici (`APICFluidStep.inl` `solid_particle`). Tekilleştirme: ortak yardımcı, çözücü tarafında.
5. T2b-1: `constitutive_model` yazıcılarını tek fonksiyonda topla (doğum noktaları `MatterDomainSynchronization.inl`, `MatterDomainSources.inl`). Davranış değişmez. Diğer ajanın dosyalarına dokunmadan önce koordinasyon.
6. Ana plan kalanı: T4 (karma MPM, tam gaz/granül bağlantısı), T5, T6 (domain tipleri). Önce T2 kapanmalı.

**Kontrol listesi:** `NEXT_BUILD_CHECKS.md` güncel parti (söküm). Ice ölçümü ve Ice preset kabulü henüz listeye eklenmedi — eklenmeli.

**Eski katı bağlama göçü (karar):** Eski katı bağlamalar donmuş başlamaz; statik katı blok tag yolunda kalır. Gerekçe: donmuş bayrağı termal zincir kapalıyken silinir ve eritilemeyen madde ilk adımda erir (kontrol sonucu). Kullanıcı "donmuş başlat" seçmişti; bu kontrolle daraltıldı ve kullanıcı onayladı.

**Dikkat:** Bu oturumda birkaç dosya başka bir ajan/kullanıcı tarafından diskte değiştirildi (RtPython.cpp, MatterModelControls.cpp, NEXT_BUILD_CHECKS.md, BUZ_MODEL_KARARI.md). Düzenlemeden önce mutlaka yeniden oku.

---

## Eski devir (T1/T3 kabulü, tarihsel)

Kanonik madde planı: [MADDE_TIPLERI_TASARIMI.md](MADDE_TIPLERI_TASARIMI.md).
Ana domain planı: [BIRLESIK_MADDE_DOMAIN_TASARIMI.md](BIRLESIK_MADDE_DOMAIN_TASARIMI.md).
Kanıt: [madde_t1_t3_live_2026-10-08.json](madde_t1_t3_live_2026-10-08.json).

## Bu oturumda doğrulanan

Kullanıcı C++ build aldığını ve uygulamanın açık olduğunu bildirdi. Ajan build
almadı veya uygulama başlatmadı. Testler ayrı Python process'inde
`scripts/test/rt_ipc.py` ile çalıştı; sandbox'ta Windows error 5 sonrası
named-pipe erişimi için escalation kullanıldı.

| Kabul | Sonuç | Kapsam |
|---|---|---|
| `rt_test_substance_profiles_ipc.py` | PASS | 21 yerleşik, kategori, türetme, override, kalıtım, Revert, hatalı yamanın reddi, referans varken silmenin reddi; Wax `5e-6`, granular solver hints kontrolü |
| `rt_test_domain_substance_ipc.py` | PASS | Water/Honey/Chocolate/Mud/Sand/Gravel/Soil/Wax seçimi, eski girdilerin mutasyonsuz reddi, madde editinden sonra hemen fizik readback'i, numerical tuning korunumu, yeniden seçimde hints, referans koruması |
| Son temizlik sorgusu | PASS | Geçici T1/T3 domain, madde ve collider kalmadı |

Simülasyon adımlanmadı, proje dosyası kaydedilmedi. Authoring işlemleri mevcut
mekanizma gereği simülasyon cache'ini geçersiz kılabilir; bu test fiziksel
hareket, cache replay veya görsel kalite kabulü değildir.

## Kapanan kaynak partisi

- `SubstanceLibrary` ve `FluidDomainSubstance` proje/filters kayıtları tamam.
- FluidPreset/FluidChemistryPreset yerine domain `default_substance` kullanır.
  Fizik madde tablosundan, sayısal ayarlar seçimde uygulanan solver hints'ten gelir.
- UI/Python/IPC aynı core yoluna bağlı; eski fizik setter'ları açık hata verir.
- Node testleri domain fizik alanı yerine `viscosity_wall_slip` kullanır.
- Eski granular preset testi kaldırıldı; madde sözleşmesine taşındı.
- Descriptor/audit ve panel alan denetimi yeni sahipliği doğrular; kaynak
  kontrolleri geçti. Scriptlerin Release kopyaları eşitlendi.
- Göç toleransı küçük ν farklarını yutmaz. Madde editinden sonra fizik/kimya
  aynaları hemen yenilenir; numerical hints tekrar uygulanmaz.

## Tam kapanıştan önce kalan kabul

Sıra [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md)'de. Hızlı testleri tekrar
istemek gerekmez; yeni kaynak değişikliği veya regresyon varsa tekrar çalışır.

1. Türetilmiş maddeli proje save/open ve yeni projede kütüphane izolasyonu.
2. Eski Honey/Lava/Wet Sand/Cohesive Soil/Molten Plastic ve özel ayarlı domain
   göçü; aynı fizik, numerical tuning ve tekrar yüklemede tek madde.
3. Panel görsel kontrolü; node opt-in/Clear Overrides geri alma kabulü.
4. Wax/MSF `5e-3` → `5e-6` viskozite değişimi, termal zincir ve rheology görseli.
5. Önceki H1 hareketli collider/suite regresyonları gerektiğinde.

## Kodda tam kaldığımız yer: T2 sonraki kaynak partisi

T3'ü yeniden yazma. Tasarım §6a kalan T2'yi ayrı parti olarak tanımlar.

- Kaynağın `initial_constitutive_model` authoring alanı ve binding
  `SubstancePhase`/Constitutive seçimi hâlâ var. Başlıca girişler:
  `ParticleSimulation.h`, `MatterDomainSources.inl`, `RtApiFluid.cpp`,
  `MatterModelControls.cpp`, domain paneli, iki serializer ve API bindings.
- ✔ T2a (2026-10-08, kaynak): `FluidThermalLiquid.cpp` `updateThermalFreeze` donma/erime
  eşiğini parçacığın maddesinden okur (eritilebilir → `melt_kelvin`, eritilemez → donmaz,
  etiketsiz → domain değeri). Karışık Water/Wax domain'i artık ayrılır. Kabul:
  `NEXT_BUILD_CHECKS.md` 2–5. `buildThermalViscosityField` ν(T) eğrisi hâlâ domain
  eşiğini kullanıyor (açık).
- ★ Karar bekleniyor: 263 K'de "doğuşta katı" kuralı, desteksiz donmuş parçacığın
  asılı kalmasını gerektirir (bkz. `NEXT_BUILD_CHECKS.md` açık T2).
- ✔ Karar (2026-10-08, kullanıcı): **Ayrı `Ice` maddesi.** Ice, Water ile erime eşiği
  (273.15 K) ve gizli ısı üzerinden bağlanır. Mevcut Ice preset'i (`MaterialStateField.cpp`)
  korunur; katı davranış parametreleri (sürtünme, E/ν, kırılma) henüz YOK ve kaynaksız
  uydurulmayacak. Açık: katı buz granül (Drucker-Prager) mi, rijit gövde mi olarak
  çözülecek — bu karar kendi ölçümüyle verilir.
- ✔ Karar (2026-10-08, kullanıcı): **Phase türetilir.** Kullanıcı seçimi olmaktan çıkar;
  faz = f(sıcaklık, melt_kelvin). Binding Phase combo'su ve IPC/Python `phase` parametresi kalkar.
- ★ Bulgu: binding `phase == Solid` bugün `FluidDomainStep.inl:284` içinde ızgara katı maskesine
  çevriliyor (akışı engelleyen İKİNCİ katı yolu) ve `MatterGrainParams.cpp:262`'de granül
  taşıyıcı seçiminde kullanılıyor. Yani "Phase kalkar" bir silme değil: katı maske tek yola
  (donmuş parçacıklar) indirilmeli. Açık: eski sahnelerdeki katı bağlamaların göçü
  (doğuşta donmuş mu, sıvıya mı çevrilsin, yoksa reddedilsin?). Karar bekleniyor.
- ★ Söküm kapsamı: `initial_constitutive_model` ~16 yerde (RtApi.h, RtApiFluid, RtIpc, RtPython,
  descriptor'lar, SceneSerializer, ProjectManager, UI, scene_data hash, ParticleSimulation.h,
  MatterDomainSources.inl). Binding `constitutive_model` ayrıca ~12 yerde. Derlemeden
  doğrulanamayacağı için tek partide yapılacak; descriptor'lar `gen_ipc_descriptors.py` ile yenilenir.
- ✔ Söküm partisi (2026-10-08, kaynak): Initial Model ve bağlama `constitutive_model` kalktı; doğum modeli
  yalnızca maddeden. Bağlama `phase` = statik katı blok işareti (sıvı fazı sıcaklıktan türer). Hata B
  düzeltildi (eritilemeyen madde donmaz). Kabul adımları: `NEXT_BUILD_CHECKS.md`.
  Eski katı bağlama göçü: statik blok tag yolunda kalır (donmuş bayrağına bağlanmaz — termal zincir
  kapalıyken bayrak silinir ve eritilemeyen parça ilk adımda erir; kontrol sonucu).
- ★ Faz yazıcısı: `frozen` bayrağı zaten tek yazıcı (`updateThermalFreeze`). `constitutive_model`
  ise doğumda birkaç yerden yazılıyor (`MatterDomainSources.inl`, `MatterDomainSynchronization.inl`).
  Bunların tek çözücüde toplanması, faz routing'i değiştiği için T2b-3 ile birlikte yapılır.
- ★ Maliyet: `MatterGpuPartition.cpp` her dispatch'te tüm model dizisini (4 B/parçacık) yükler.
  Revizyon sayacı ve yalnız değişimde yükleme ölçümden sonra eklenecek (T2b-2).
- Frozen bayrağı ile binding solid mask iki ayrı yol; tek `madde + sıcaklık →
  hal/model` core çözümü ve tek bayrak yazarı henüz uygulanmadı.
- Ice→Water yükleme göçü, soğuk başlangıç sıcaklığının korunması ve ilgili
  testler açık. T2 kabulü: aynı madde 263 K'de katı doğar, ısınınca sıvıya geçer;
  UI/Python/IPC/serializer aynı hizmet ve hata semantiğini paylaşır.
- `fluid_flammable`, `fluid_extinguishing` vb. domain kimya aynaları hâlâ
  saklanıyor; seçme/edit sırasında eşitleniyor. Sökmek gerekiyorsa explicit
  combustion enable kontrollerini madde fiziğiyle karıştırmadan ele al.
- Elastik MPM karma yol, tam gaz/granül bağlantısı ve uzman Gas/Liquid
  domain'lerini kaldırma bu hızlı kabul ile tamamlanmış sayılmaz (T4/T5/T6).

Yeni feature logic 2000 satır üstü dosyalara eklenmez; odaklı yeni core modülü,
bu dosyalarda yalnız entegrasyon çağrıları. Build kullanıcıya aittir.
# 2026-10-08 sparse grid devamı

Vulkan pressure scratch sparse tile pool kaynağa bağlandı; on yeni shader ABI
16/80, SparsePressureGpu.cpp + stat/UI/IPC/Python aynı core. GFM/periodic yolun
fiziği kapatılmadı. Parent mixed bake revision 5. Kaynak kontratları PASS;
kullanıcı build + dense/sparse IPC parity testi açık. Diğer ajanın canlı testleri
sırasında yeni IPC koşusu açılmadı. Tam MAC/gaz sparse storage henüz bitmedi;
durum ve kalan kapılar [MATTER_SPARSE_TILE_GRID.md](MATTER_SPARSE_TILE_GRID.md).
