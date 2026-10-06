# C4b karma GPU test noktası — 2026-10-04

Bu kayıt önceki C4b “henüz bağlı değil” notlarının güncel durumunu değiştirir.
Karma Fluid/Granular transport kaynakta canlı Matter adımına bağlıdır. Codex
uygulama veya shader derlemesi yapmadı; çalışma zamanı kabulü henüz verilmedi.

## Kaynakta bağlı akış

- Model listeleri kanonik düz parçacık indekslerini taşır; kimlik ve substance
  yeniden doğmaz veya emitter bazlı ayrı otoriteye dönüşmez.
- Model başına persistent GPU grid ve FLIP alanı kullanılır. Indexed P2G/G2P,
  granular stress/settle, occupancy ve advection mevcut çekirdek matematiğini
  ortak GLSL kaynaklarından paylaşır; legacy ABI varsayılan olarak korunur.
- Indexed P2G rest_mass_kg * mass_fraction ile fiziksel kütle toplar. Granular
  stress impulse'u aynı fiziksel kütle metriğine ölçeklenir.
- Ortak CFL/elastic alt adımda sıvı MGPCG basıncı ve granular stress ayrıdır;
  GPU temas ikisi hazır olduktan sonra, iki G2P'nin önünde çalışır.
- Temas tuple'ı ayrık MAC yüzlerini sahiplenir; normal ve Coulomb impulse'ları
  eşit/ters, yüz-kütlesi metriğinde enerji azaltıcıdır. Bu metrik kanonik
  parçacık momentum/parity fiziksel kabul testinin yerine geçmez.
- Model partikülleri/granular state ve grid hızları alt adımlar arasında GPU'da
  kalır. Host sonuçları başarılı frame sonunda birlikte yayınlanır. Mevcut
  MGPCG'nin küçük convergence readback'leri sürer; “tamamen readback yok” iddiası yok.
- Frame boyunca sabit collider/viscosity alanları her ortak alt adımda yeniden
  yüklenmez. İkinci model grid/pressure scratch çalışma seti tahmini domain
  resource_budget_mb sınırına bağlıdır; aşımda stiffness düşürülmez.
- Karma GPU başarısızlığı adımı tutar ve gerekçeyi bildirir; tek-model CPU
  fiziğine sessiz düşüş yoktur. CPU domain açıkça seçilirse referans batch çalışır.
- Tek model kaldığında seçim kanonik constitutive_model'den yapılır; Auto
  yalnız legacy domain fallback'idir. Özellikle ilk su çıkmadan önce yalnız
  granül bulunması tek-model sıvı basıncına yönlenmez.

## Son derleme ve tek sahne kabulü

1. Uygulama C++ derlemesi ve `RayTrophiStudio/source/shaders/compile_sim_shaders.bat`
   ile simulation shader derlemesini kullanıcı yapar. Yeni `sim_matter_*.spv`
   dosyaları çalışan uygulamanın shader klasöründe bulunmalı; uygulama yeni
   binary/shaderlarla yeniden açılmalıdır.
2. Küçük bir Matter domain (ör. 32³/64³), Vulkan compute, Closed duvarlarla
   başlanır. Bir Sand/granular emitter ile Water/fluid emitter farklı konumlarda
   kurulup su granül yatağına veya karşı akışına ulaşacak şekilde yönlendirilir.
3. İlk yalnız-granül kareleri ve sonra iki modelin birlikte bulunduğu birkaç
   temas karesi ilerletilir. Active Matter tablosunda iki model korunmalı;
   durum “Mixed Vulkan” göstermeli, common substeps >= 1 olmalı. Temaslı bir
   karede contact pairs > 0 beklenir. Kaynaklar tam üst üste aynı hızda ise
   kapanan temas veya ayrıştırılabilir normal oluşması beklenmeyebilir.
4. Ayrı terminalden salt okunur prob:
   `python scripts/test/rt_test_matter_mixed_gpu_ipc.py "DOMAIN ADI" --contact`.
   Bu prob GPU stage/alt adım/kimlik/sonlu kütle kabulünü kontrol eder;
   görsel davranış, momentum grafiği veya performans kabulü yerine geçmez.
5. Sonra Open outflow ile parçacık sayısının düşmesi, bir model kalınca doğru
   tek-model yoluna geçmesi, cache rewind ve tek-model sıvı/granül smoke
   kontrol edilir. Model kaynakları durdurulduğunda cache kimlikleri ve
   materyal UVW koordinatları sürekliliğini korumalıdır.

## Açık kabul sınırları

- Vulkan karma Periodic, frozen ve Elastic taşıyıcılar şu test noktasında
  desteklenmez; adım açık hata ile tutulur. CUDA karma yol eklenmedi.
- GPU temas 19 storage binding gerektirir. Descriptor kapasitesi 24'e açıldı;
  cihaz limitleri altında layout'lar oluşturulur ve eski 16-binding cihazlarda
  legacy compute yolu korunur. Eksik Matter kernel yeni adımı açık hata ile tutar.
- CPU hücreden MAC'a referans lift ile GPU yüz tuple teması farklı ayrık
  operatörlerdir; CPU/Vulkan parity henüz kabul edilmiş değildir.
- Kimyanın iki emitere domain üzerinden uygulanması, ayrı splat materyalleri,
  gas+liquid body materyal davranışı ve domain taşıma/cache kozmetiği mevcut
  kalite notlarında bekliyor. Bu patch bunları giderdiği iddiasında değildir.
- C4 fiziksel/performance kabulü ve H1/C5/C6/C7 yol haritası tamamlanmadı.

Kaynak kontrolleri: `check_matter_transfer_contracts.py`,
`check_matter_solver_stages.py`, `check_matter_gpu_contracts.py` ve
`gen_ipc_descriptors.py --check`. Bunlar C++/GLSL çalışma zamanı testleri değildir.

## 2026-10-04 canlı karma sahne ve Output başlangıcı

Kullanıcı uygulama ve shader derlemelerinin tamamlandığını bildirdi. Dış
`rt_ipc.py` ile 32³ Closed/Vulkan `C4_Mixed_GPU` sahnesi kuruldu: birbirine
yönelen Sand/Granular ve Water/Fluid kaynakları. 12. karede her modelde 900
parçacık, 14 ortak alt adım, 992 contact pair ve 46.415.872 byte çalışma seti
raporlandı. P2G/pressure/G2P GPU bayrakları true, step_held false idi.
`rt_test_matter_mixed_gpu_ipc.py C4_Mixed_GPU --contact` geçti. Kullanıcı iki
maddenin kendi davranışını koruduğunu gözlemledi. Bu fiziksel yakınsama veya
CPU/Vulkan parity kabulü değildir; C5/C6 emilim/doygunluk henüz bağlı değildir.

Matter Output paneli mevcut ortak `fluid.set_substance_material` API'sini
kullanarak emitter maddelerini, bağımsız SDF/Splat/Fog seçimini ve sahne
materyali atamasını görünür kılar. Emitter ve render için ikinci bir fizik
otoritesi oluşturmaz. Yeni panel henüz kullanıcı tarafından derlenmedi.

Panel kabulü: water=SDF, sand=Splat seç; farklı mevcut sahne materyalleri ata.
Madde bazlı ayarları `fluid.get` substance_materials ile geri oku. Temas
sayaçları ve model kimlikleri korunurken RT/Solid/RayFusion görünümünü kontrol
et. Her iki maddeyi sırayla SDF/Splat/Fog yap; state-label rotalarının spray/
foam/mist için görünümü ayrıca yönlendirdiğini dikkate al. Fog mevcut domain
volume shader'ını kullanır; madde başına ayrı fog shader'ı ve preset'ten
materyal üretimi bu ilk panel değişikliğinde eklenmemiştir.

## Output / ortak havuz kod partisi (derleme bekliyor)

- Output sırası tekrar Liquid Display ile başlar. Matter Output bunun içinde
  madde bazlı yönlendirme/material düzenleyicisidir; gaz shader bloğu sıvı
  görünüm ayarlarının arkasındadır. Matter için eski Substance Look tekrar
  gösterilmez; legacy liquid sahnelerde eski panel korunur.
- `material.create(type="substance:Water", name="My Water")` ve
  `substance:Sand`, `substance:Iron`, `substance:Paper` gibi kanonik katalog
  adları normal, düzenlenebilir Principled sahne materyali üretir. Preset yalnız
  görsel öneridir; fizik/faz/kimya değiştirmez. Bilinmeyen madde adı materyal
  ayrılmadan reddedilir. UI aynı `createMaterial` ve substance binding API'sini
  çağırır; mevcut sahne materyali seçimi ve domain materyaline dönüş korunur.
- `flow_source.create/update(..., particle_pool_weight=1.0)` kalan canlı
  kapasitenin paylaşım ağırlığıdır. [0.001,1000] aralığında sonlu olmalı; geçersiz
  istek atomik reddedilir. Alan proje/scene serialization ve Python/IPC'de
  aynıdır. Eski dosyalarda 1.0 olur. Bu canlı sahiplik kotası veya ömür boyu
  üretim limiti değildir; emitter lifetime limit ayrı kalır.
- Enjeksiyon başlamadan bütün uygun liquid kaynaklarının talebi hesaplanır.
  Düşük talebin kullanmadığı pay tekrar dağıtılır, tamsayı eşitlikleri kareye
  göre döner. Havuzun 1% dead-band kontrolü plan başında bir kez uygulanır.
  Böylece listedeki ilk emitter kapasiteyi tüketerek diğerini aç bırakmaz.
  Reddedilen tam parçacık talepleri sonraki kareye birikmez.
- `flow_source.get/list` read-only `pool_requested_particles` ve
  `pool_granted_particles` son enjeksiyon tahsisini bildirir. Bu gerçek
  doğmuş veya hâlâ yaşayan emitter parçacık sayısı değildir: geçersiz spawn
  konumu nedeniyle ayrılan slot kullanılmayabilir. Rewind bu sayaçları temizler.
  Domain sekmesindeki Shared Particle Pool ortak canlı toplamı, ağırlıkları
  ve tahsisi gösterir; madde kimliği emitter kökenine dönüştürülmez.

Kabul: kullanıcı C++ derler; bu partide shader değişikliği yok. Boş sahnede dış
terminalden `python scripts/test/rt_test_matter_output_pool_ipc.py` çalıştırılır.
Prob geçici domain/kaynakları kaldırır, iki düzenlenebilir test materyalini
sahnede bırakır. Preset/editability, Water=SDF + Sand=Splat farklı materyal
bağlamaları, geçersiz ağırlık atomikliği ve 1:3 paylaşımda 250/750 tahsis
kontrol edilir. Görsel RT/Solid/RayFusion kontrolü ayrıca yapılır. Saf allocator
için `scripts/test/matter_emission_budget_test.cpp` eşit/ağırlıklı/düşük talep/
tie rotation/korunum testlerini taşır; Codex bunu derlemedi veya çalıştırmadı.

Önemli: katalog adı tam eşleşir (`Water`, `Sand`). Önceki canlı ilk probdaki
küçük harfli `water`/`sand` kaynakları explicit constitutive override ile iki
rejimi doğruladı; katalogdan yoğunluk/kimya doğruladığı anlamına gelmez. Yeni
kabul probu kanonik adları kullanır. Madde başına ayrı fog shader ve C5/C6
emilim-doygunluk bu partide hâlâ açık kalır.
