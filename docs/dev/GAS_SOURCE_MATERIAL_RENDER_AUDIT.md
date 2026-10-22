# Gaz kaynağı / veri kanalı malzemesi — render incelemesi

> **Durum:** ARŞİV — 2026-10-21; kullanıcı kararıyla kaynak başına materyal uygulaması geri alındı. Aşağıdaki uygulama ve ölçüm notları tarihçedir.

## Render yolu

1. `FluidDomainFogVolume.cpp::SceneData::syncDomainFogVolume` Matter için
   `gasGrid(state).density` alanını kullanıyor. Fog görünümündeki sıvı parselleri
   ve whitewater da yoğunluğa eklenebiliyor. Tek malzeme `fluid_fog_shader`.
2. `VolumetricRenderer.cpp::syncVolumetricData` her VDB için bir `GpuVDBVolume`
   üretiyor. Optik katsayılar ve material program hacim başına bir kez atanıyor.
   Canlı gaz adresleri yalnız cihazı paylaşan Vulkan tüketicisine veriliyor;
   diğer Vulkan cihazı host aynasını kullanıyor.
3. `VulkanBackend_Volumes.cpp` paketi TLAS sırasına eşleyip `VkVolumeInstance`
   tablosuna çeviriyor. Canlı dense gaz `source_type=5`, `volume_type=4`;
   NanoVDB `volume_type=2`. Ayna tüketicisi kendi cihazında buffer oluşturuyor.
4. `volume_closesthit.rchit::sampleDenseGasFloat` hücre merkezli float alanı
   trilinear okuyor. Yoğunluk dönüşümü `sampleDensityAccMode`, `sampleDensityAcc`
   ve `sampleDensity` üzerinden yürüyor. Birincil march, gölge ve random walk
   aynı kanal karışımının sönümünü kullanmalı.

## Kaynak anlamı ve alınan karar

Kullanıcı gerçek kaynak ayrımını seçti ve kapsamı üretici/taşınım/cache yoluna
genişletti. Aşağıdaki iki sözleşmeden ikincisi uygulanacak; kanal bağlamayla
yetinilmeyecek. Isı hesabı değiştirilmez; yalnız provenance bağlantıları eklenir.

`grid.density` saf emitter izi değildir. GPU yüzey buharlaşması
`sim_inventory_surface_exchange.comp` içinde hem buhar kütlesini (`m += dm`)
hem gaz yoğunluğunu (`density += ...`) artırıyor. CPU karşılığı
`GasSurfaceExchangeCpu.cpp` aynı işlemi yapıyor. Birikim üstten sınırlanıyor;
bugünkü buhar kütlesini yoğunluktan çıkarmak emitter izini geri getirmez.
Alanların taşınımı ve birimleri de farklıdır.

Ayrı envanterler `gas_phase_mass_kg` (buhar), `gas_phase_condensate_kg`
(yoğuşmuş sis), `gas_phase_foreign_kg` (yoğuşmayan gaz). GPU sahibi
`GasInventoryDevice`, host/cache karşılıkları domain state.
`captureGasInventoryHostImage` mevcut okuma yüzeyidir; her render sync'inde
tam envanter indirmek performans açısından ayrıca değerlendirilmelidir.

- **Kanal bağlama:** ortak gaz yoğunluğu bu adla sunulur; buhar ve yabancı gaz
  envanterleri ayrı render kanalları olur. Ortak yoğunluk emitter diye
  adlandırılmaz. Buhar/sis optik anlamı ve çift sayımı önleme kuralı tanımlanır.
- **Gerçek kaynak bağlama:** emitter katkısı üretildiği andan itibaren ayrı
  izlenir ve taşınır. Mevcut birleşik alandan kesin türetilemez;
  üretici/taşınım/cache değişikliği gerekir. Isı/fizik dosyalarına dokunmama
  sınırı altında bu sözleşme tamamlanamaz.

## Uygulama kapıları

Üç coincident VDB çözüm değildir: `volume_overlap_selection.glsl` üçüncü üyeyi
atlıyor; companion emission ve material programını çalıştırmıyor. Kanal
karışımı tek domain hacminin içinde değerlendirilmelidir. Sönüm ve saçılma
yoğunlukla ağırlıklı toplanmalı; emission ve faz fonksiyonu katkıları korunmalı.
Majorant/empty-space skip tüm etkin kanalları kapsamalı; ortak yoğunluğun sıfır
olması envanteri boş saydırmamalıdır.

`RtVolumeMaterialBindings.cpp` mevcut `volume.list_materials/get_material/
set_material` ve Python yüzeyini taşıyor. Yeni bağlar ortak core servise
eklenmeli; panel aynı servisi çağırmalı. `RtApi.h`, IPC, Python, yetki, overlay
ve üretilmiş descriptor birlikte teslim edilmeli. `gas:<domain>` hedefinin
anlamı sessizce değişmemeli. Proje kaydet/aç, cache/replay ve farklı Vulkan
cihazı aynı kanal sözleşmesini korumalı.

2000 satırı aşan renderer/backend/shader/panel/proje dosyalarında yalnız
include/call/serialization wiring; özellik odaklı yeni modüllerde uygulanmalı.
OptiX'e özellik eklenmez. Kullanıcının kapsam genişletme kararından sonra
fizik kaynaklarına yalnız kaynak izi/taşınım/cache bağlantıları eklendi;
ısı ve kuvvet formülleri değiştirilmedi.

## Yazılan ilk temel (kaynak malzemesi henüz yok)

- `FluidGrid::emitter_density`: yalnız flow source ve parçacık emitter
  enjeksiyonlarının doldurduğu bağımsız, boyutsuz alan. Buharlaşma bu alanı
  beslemez. Fiziksel yoğunluk, sıcaklık ve envanter alanlarına geri besleme yok.
- `GasEmitterProvenance.h/.cpp`: CPU pasif skaler taşıma, Vulkan'da mevcut
  scalar/MacCormack çekirdekleriyle cihazdaki hız alanından taşıma. Sonuç ancak
  başarılı okuma sonrası commit edilir; başarısız GPU aşaması CPU yoluna düşer.
  Dissipation bir kez uygulanır, katı hücrelerde iz temizlenir.
- RAM cache alanı lossless saklar; Matter grid cache'i ve bellek hesabı kapsar.
  Disk cache v17 alanı ve `emitter_density_known` bilgisini saklar. v11–v16
  dosyaları kaynak izi bilinmiyor olarak okunur; ortak yoğunluktan türetilmez.
  Aynı layout'ta solver scratch hydration alanı ve known bilgisini korur.
- SparseVDB adımı henüz kaynağa özgü taşımayı yapmıyor; provenance `unknown`
  işaretlenir. Vulkan birincil yoldur. OptiX kaynaklarında değişiklik yok.

## Kaynak malzemesi bağlantısı — test noktasına teslim

`GasSourceMaterials` tek doğrulama/ayar servisidir. Matter descriptor'ında
`gas_source_materials` saklanır. Panel `rtapi::setGasSourceMaterials` çağırır;
IPC ve Python aynı API'den geçer. `gas.get_source_materials(domain)` ve
`gas.set_source_materials(domain, settings)` exact unique Matter adı kullanır.
Final render kilidi, strict anahtar/tür/aralık kontrolü ve atomik patch vardır.
Proje ve SceneSerializer yolları aynı schema'yı saklar/geri yükler.

Kaynak modu opt-in, varsayılan kapalıdır. Her kaynak bağımsız optik
malzeme taşır: density_multiplier/cutoff, scatter_color/coefficient,
absorption_color/coefficient, emission_color/strength, anisotropy,
back_anisotropy, lobe_mix, multi_scatter. Emission, düşük veya sıfır extinction durumunda da analitik segment integraliyle
korunur. Scatter/absorption RGB [0,1], emission RGB [0,10000],
katsayılar [0,1e6], g [-.99,.99], karışımlar [0,1]. Kaynak modu Volume Graph,
random walk uygulamaz; kaynak başına blackbody ve ramp desteklenir.
Domain shader'ı adım/gölge kalite ayarlarının otoritesidir. Çoklu saçılma ve
faz karışımı bounded march yaklaşımıdır; tam çoklu ortam random walk değildir.

- emitter: enjeksiyonda kaydedilen boyutsuz yoğunluk izi (emitter kimliği
  başına değil, emitter kaynak sınıfı). Buhar katkısından türetilmez.
- steam: (buhar + askıdaki yoğuşmuş su) / hücre hacmi, kg/m3. Yağmur dahil
  edilmez. Bu malzeme görünür buhar için yazarlık kontrolüdür; saf buharın
  otomatik fiziksel optik modeli olduğu iddia edilmez.
- foreign: yoğuşmayan gaz envanteri / hücre hacmi, kg/m3.

`GasSourceRender` fizik alanlarını değiştirmeden derived render paketini
üretir. GPU envanteri için mevcut snapshot okuyucusu kullanılır; bu yeni yolun
host kopya ve bellek maliyeti canlı testte ölçülmelidir. Ham kaynakları tek
hacimde toplar; ortak gaz yoğunluğu tekrar eklenmez. Sıfır içerik paketten
çıkarılır; eski shared-density majorant ve topology skip kaynak modunda
kullanılmaz. Grid'in tamamı traversal proxy'sidir. Eksik provenance/readback
hatası halinde domain kaynak render'ı bastırılır, `render_error` görünür.

`gas_source_medium.glsl` hem RT hem RayFusion için tek sampler/optik toplam ve
gölge hesabıdır. RT ışık march'ı, surface shadow ve companion kaynak emission'ı
bu toplamı kullanır. RayFusion kendi bounded çizimini aynı kaynak örneğiyle
yapar; tüm görüntünün bire bir eşitliği vaat edilmez. VkVolumeInstance 688
byte; kaynak pointer'ı offset 672. Dört structured shader ve üç raw reader
birlikte güncellendi. Her Vulkan cihazı kendi kaynak buffer'ını yükler.
Frozen OptiX `GpuVDBVolume` ABI'sine kaynak alanı eklenmedi; synchronous Vulkan
host metadata ayrı tutulur ve upload sonrası temizlenir. OptiX'e yeni optik
özellik eklenmedi.

Probe: `scripts/test/rt_test_gas_source_materials_ipc.py --domain NAME` salt
okur; `--apply` bağımsız patch/atomik hata/restore kontrolü yapar. Python:

```python
rt.gas.set_source_materials("Matter", {
    "enabled": True,
    "sources": {"steam": {"scatter_color": [1, 1, 1], "density_multiplier": 4}}
})
rt.gas.get_source_materials("Matter")
```

## Sıralı kabul

1. Geçersiz kaynak/hedef/tür/malzeme ve render kilidi atomik hata vermeli;
   mevcut bağ korunmalı.
2. Panel/IPC/Python aynı kanonik bağları göstermeli; yetki/descriptor audit'leri
   geçmeli. Yalnız panelde görünen seçim kabul değil.
3. Kaydet/aç, domain çoğalt/sil/yeniden oluştur bağları doğru yerde tutmalı;
   eski cihaz adresleri yaşamamalı.
4. Ortak gaz, buhar ve yabancı gaz tek tek denenmeli: malzeme değişikliği yalnız
   seçilen kanalın katkısını değiştirmeli. Build ve shader derlemesi kullanıcıda.
5. Üç kanal aynı hücrelerde: extinction, emission ve gölge korunmalı; kanal
   sırası sonucu değiştirmemeli. **Sinsi hata:** üçüncü kanalın kaybına rağmen
   makul görünen tek renk duman.
6. Pause/cache/replay, ikinci Vulkan cihazı, sıvı yüzeyiyle örtüşme ve boş alan
   atlama aynı ayrımı korumalı.
7. En son mevcut Matter duman süiti ve maliyet ölçümü. Canlı IPC dış
   `scripts/test/rt_ipc.py` sürecinden, kullanıcı uygulamayı açtıktan sonra.

Sayısal C++ regression harness'i `scripts/test/gas_emitter_provenance_test.cpp`
ve Release aynasında. Derleme kullanıcıda; bu harness henüz çalıştırılmadı.
Build, shader derleme ve uygulama başlatma yapılmadı. Bu partinin build
kontrolleri `NEXT_BUILD_CHECKS.md` içindedir.

Parti sonu statik kontroller: proje/filter XML ve tekil modül kayıtları,
Release test aynası, yalnız provenance alanına yazım ve readback öncesi commit
olmaması, v17 disk sırası/legacy unknown işaretlemesi PASS. İlgili tracked
dosyalarda `git diff --check` geçti. Bunlar sayısal veya canlı kabul değildir.

Takip partisinin statik kabulü: 4 yeni source ABI/wiring ve 10 mevcut volume
material testi PASS; descriptor güncel (683 metot), capability mirror audit PASS.
Yeni C++/GLSL delimiter, Python AST, XML ve Release test aynası kontrolleri PASS.
C++ harness, shader build ve dış IPC/RT/RayFusion görsel kabulü kullanıcıda.


## Kaynak başına termal emission — 2026-10-21 (Active, build bekliyor)

Dördüncü `combustion` slotu mevcut reaction/fuel alanına bağlanır; duman
yoğunluğundan yangın kimliği çıkarılmaz. Aynı hücrede buhar ve yangın kendi
malzemelerini kullanır. Buhar varsayılan `none`, yanma `blackbody/reaction`.
Bu sınıf ayrımıdır; ayrı emitter veya yakıt türü kimliği takibi değildir.

Her slot none/color/blackbody/ramp, Kelvin aralığı, sıcaklık ölçeği/gücü,
yoğunluk gücü, exposure, saturation ve 2..8 RGBA ramp stop sunar. Reaction/fuel
driver yalnız combustion içindir; fuel seçimi sıcak yanmamış yakıtı da bilinçli
olarak ışıtabilir. Panel, Python ve IPC aynı atomik servisi çağırır. Eski RGB
kayıtlar color olarak göç eder. Opt-in kapalı projeler aynı kalır.

Termal alan mevcut ownership ledger üzerinden salt okunur snapshot edilir.
Render, fiziğin `airKelvinFromVirtual` dönüşümünü kullanır; ısı denklemleri
değişmez. Hücrede tek fiziksel hava sıcaklığı vardır; optik cevap malzeme
başınadır. Payload v2: 1184 byte header, 32 byte/hücre; materyal 288 byte.
VkVolumeInstance 688 byte kalır. RT ve RayFusion ortak emission helper kullanır.
Yeni C++ sayısal örnekleri hazırlanmıştır; derleme/canlı kabul kullanıcıdadır.

## Güncel model: çıktı bağı ve tek materyal editörü (payload v3)

Kaynak sınıfı paleti artık normal UI değildir. Yeni Matter domain varsayılan
olarak ayrı volume çıktıları kullanır. Eski projelerde mevcut görünüm için
legacy yol korunur; Separate volume outputs ile geçilir ve eski cache için
reset/rebake gerekir. Legacy dört slot API explicit binding_mode=legacy_sources
ile uyumluluk için durur; sources-only patch de bu yolu seçer.

- `gas:<domain>`: mevcut gas VolumeShader varsayılanı.
- `output:<domain>:gas/<uid>`: flow emitter gazı. İlk edite kadar gas varsayılanını
  miras alır; ilk edit aynı VolumeShader ayarlarından bağımsız override kurar.
  Particle emitter deposit kimlikleri high-bit namespace kullanır; flow kimliğiyle
  çakışmaz. Konumsal yakınlıktan emitter kimliği tahmin edilmez.
- `output:<domain>:steam`: buhar+yoğuşma; gas emission mirası yok.
- `output:<domain>:foreign`: yabancı gaz; gas emission mirası yok.
- `output:<domain>:fog/<substance>`: yalnız authoritative view resolver'ın fog'a
  gönderdiği parseller. Karın parçacık Kelvin'i kullanılır; gas Kelvin'i değil.
- `output:<domain>:secondary/spray|foam|bubble`: yalnız fog'a yönlendirilen
  secondary tipler. SDF/splat BSDF bağları değişmez.
- `output:<domain>:fog`: bu fog hedeflerinin varsayılanı. Override varsa korunur.

VDB paneli Simulation volume outputs listesini ve mevcut VolumeShader editörünü
kullanır. Domain panelindeki ikinci optik/blackbody ağacı kaldırıldı. Aynı
`volume_material::execute` servisi UI, Python ve IPC tarafından çağrılır.
`volume.list_materials` hedef ve inherited durumunu listeler. `volume.set_material`
çıktı hedefinde `inherit:true` ile override'ı kaldırır. `gas.set_source_materials`
outputs map'inde null da aynı mirası geri getirir. Proje serileştirmesi mevcut
source settings noktasından kaydeder; yeni bir yan kayıt otoritesi yoktur.

Emitter yoğunluk izleri UID ile bağımsız taşınır, sönümlenir ve solid maskesine
uyar; kaynama yoğunluğu bunlara katılmaz. Tek emitter'da aggregate transport
sonucu paylaşılır. Çok emitter'da mevcut scalar carrier ayrı izleri taşır.
Disk cache v18 ve RAM lossless cache kimlikleri saklar; eski cache per-emitter
kimliği uydurmaz. Bu izler gas yoğunluğu kökenidir: sıfır yoğunluklu yakıtın
per-emitter reaksiyon kökeni henüz ayrı taşınmıyor; legacy combustion kanalı
reaksiyon/fuel görünümünü sunmaya devam eder. Tam termokimyasal kaynak ayrımı
iddia edilmez.

Payload v3: 48 byte sabit başlık + kanal başına 336 byte native/legacy optik
veri; hücre başına 4 byte kayıt indeksi, 8^3 blok occupancy tablosu ve yalnız
aktif hücreler için `(2*kanal_sayısı+2)*4` byte. Maksimum 32 kanal. Her kanalın
kendi yoğunluğu ve sıcaklığı vardır. Native remap/curve, phase, blackbody/ramp
ve emission response korunur; kar fog'u gas ramp'ını okuyamaz. Volume graph,
noise ve random-walk kaynak karışımında uygulanmıyor; bounded march kullanılır.
VkVolumeInstance 688 byte ve frozen OptiX packet ABI değişmedi.

Maliyet düzeltmesi: aynı state/inventory sürümünde GPU readback ve fog splat
tekrarlanmaz. Materyal değişikliği yalnız native başlığı günceller; Vulkan
adapter field_version aynıysa sadece bu başlığı upload eder. Eski birleşik
Matter NanoVDB'nin önce üretilip sonra atılması kaldırıldı. RT/RayFusion ortak
occupancy üzerinden bütün kaynakları ve interpolation halo'larını kapsayan
boşluk atlar. RT ana ve companion örneklemesindeki yinelenen source sample
kaldırıldı. Shadow extinction yolu blackbody/ramp değerlendirmez.

**Kalan maliyet:** değişen fizik adımında CPU snapshot/seyrek paketleme hâlâ
var. GPU tüketimi boyunca yazmayı önleyen waitIdle korunur; güvensizce kaldırılmadı.
Bu teslim tam GPU-resident/fence-ring üretim değildir. render_pack_count,
render_cache_hits ve render_packet_bytes IPC'de ölçülebilir. Hızlanma yüzdesi
henüz ölçülmedi. Fiziksel ısı/yanma denklemleri değiştirilmedi.

Yeni kontrol: `scripts/test/rt_test_volume_outputs_ipc.py --domain NAME`
salt okur; `--apply` (timeline pause) bağımsız output edit, atomik ret, inherit,
materyal-only cache ve finally restore sınar. Yeni binary olmadan çalıştırılması
bu teslimi doğrulamaz. Sıralı kullanıcı kabulü NEXT_BUILD_CHECKS.md'dedir.


### glslc process crash düzeltmesi — 2026-10-21

Kullanıcı volume_closesthit.rchit derlemesinde -1073741571/0xC00000FD bildirdi.
Bu süreç stack overflow durumudur; hangi compiler pass'inin tetiklediği eldeki
çıktıdan kesinleşmez. Derleyici IR/inlining yükünü azaltmak için array içeren
336 byte GasSourceMaterial değer kopyaları ve parametreleri kaldırıldı; SSBO
buffer-reference ve material index kullanılıyor. Optics-only density/shadow
fonksiyonları emission çağrı ağacından ayrıldı. RT nested light marches artık
blackbody/ramp örnekleyicisini density hesabı için büyütmez. GPU payload ve optik
formüller korunur. Derleme agent tarafından çalıştırılmadı; kullanıcı teyidi açık.


### Gaz march maliyeti, buhar hedefi ve preview düzeltmeleri — 2026-10-21

Kullanıcı karşılaştırması: gaz verisi bulunan aynı durumda separate materyaller
tek hacme göre yaklaşık 2x; gazın yokluğuna göre toplam RT gaz işi 10x üstü.
İlk kapatma karesinde gazın kaybolması karşılaştırmayı yanıltmıştı. Yeni setter,
Separate aç/kapat sırasında timeline ilerlemeden render üreticisini force-resync
eder. Bu performans oranları Codex tarafından doğrulanmış hız ölçümleri değildir.

Salt canlı inceleme: açık sahnede native outputs, overrides boş, 5,359,532 byte
paket; pack_count136/cache_hits1042. Cache çalışıyordu. Bölgesel instrumentasyon
[.35,.25,.65,.8] anlık toplamında paths524160, volume_rays424019,
density_samples39091755, empty_segments9203607 görüldü. Otomatik display-loop
kareleri de sayaçlara girebilir; talep edilen bir kare bu sayıların yalnız tek
kareye ait olduğunu kanıtlamaz. API submission37.76ms GPU-only/FPS benchmark
sayılmaz. Source shadow yolu sayaç yazmadığı için shadow_density_samples0
ölçümü eksikti. Pipeline ready ve driver creation225.95s raporladı. Mod ve sayaç
ayarları probe sonunda geri yüklendi; fizik verisi ilerletilmedi.

Düzeltmeler:

- Ham pozitif nem/taşıyıcı varlığı artık optik occupancy anlamına gelmez.
  Native remap/monotone density curve/cutoff ve gerçek scatter/absorption/emission
  desteği üzerinden görünür hücreler işaretlenir. Trilinear halo korunur.
  Ham kayıtlar silinmez: materyal/cutoff değişince snapshot almadan occupancy
  yeniden üretilir. GPU'ya başlıkla beraber yeni visibility tablosu yüklenir.
- Bounds 8-voxel bloklara oturur; destek blokları değişince geometry kirlenir,
  aynı blok içi hareket TLAS değişikliği yaratmaz.
- Source shadow march RGB extinction cache'inden 8 komşu okur; her gölge örneğinde
  kanal sayısı kadar remap/absorption hesabı tekrar edilmez. Cache fizik veya
  materyal değişiminde host raw kayıtlardan güncellenir; materyal-only upload
  başlık + occupancy + cache tail taşır. Ana ışın kaynak materyalini değerlendirmeye
  devam eder. Gölge cache'i mapped katsayıların trilinear yaklaşımıdır; nonlinear
  cutoff/remap'i her gölge noktasında tekrar uygulamakla bire bir aynı değildir.
  Native curve'in kullanılmayan 9..11 slotları material0'da opsiyonel tail metadata
  taşır. Payload v3, Material336, VkVolumeInstance688 değişmez. Cache yoksa eski
  optics yolu kullanılır (ör. raw tail başlangıcı16M float üstünde).
- Boş blok ve boş stencil kontrolleri kanallar okunmadan yapılır. Sıfır katkılı
  kanal işi atlanır. Source shadow örnekleri ve source occupancy query'leri mevcut
  render.volume_stats sayaçlarına yazılır.
- Solid/Matcap source yolu workbench aydınlatmasını artık atlamaz. RF world ambient
  yüzey yoluyla aynı intensity kuralını kullanır: düz renkte worldColor.w, HDRI'da
  worldParams.y, Physical Sky bake'inde ek HDRI çarpanı yok. Yeni görsel teyit açık.
- Mixed Matter VDB nesnesinin yanıltıcı tek-shader editörü yerine aynı panelde
  output seçici açılır; ilk seçim steam'dir. Proxy VDB hedefi volume.list_materials
  içinde bağımsız materyal diye listelenmez; canonical gas/output hedefleri kullanılır.
  Hedef listesi GPU volume oluşmasını beklemez ve ilk GPU bake öncesinde düzenlenebilir.

Yeni sayısal fixture gas_source_visibility_test.cpp; nem/cutoff, halo, RGB gölge
katsayısı ve materyal-only geri kazanımı sınar. Yeni harici
rt_probe_gas_render_cost_ipc.py salt okur; --apply aynı paused frame üzerinde
separate/shared/separate sayaçlarını ve fizik veri sabitliğini karşılaştırır.
Instrumentasyon timing'i çıplak GPU render süresi değildir. 10x ortak gaz maliyetinin
çözüldüğü veya 2x ek maliyetin kaldırıldığı iddia edilmez; yeni build ile kabul açık.

### 2026-10-21 — VDB input identity and duplicate editor repair

User reports separated RT about 5x faster than the previous implementation, still
about 2x slower than no gas. Shared/disabled mode is now more expensive than
separated mode. These are user observations, not controlled GPU measurements.
The disabled mode still uses FluidDomainFogVolume's legacy NanoVDB producer and
shared shader; source optical occupancy/cache does not accelerate that path.
No change to its march policy is included in this UI repair.

The global output editor was repeated in selected proxy properties. Remove the
second full editor and retain a direction to the global selector. The shader UI
uses shader.get() for ImGui IDs, but the caller allocated a fresh draft every
frame. Keep a stable per-target draft address and refresh its values from the
canonical service every frame. All writes still use volume_material::execute;
no new script/IPC operation or independent material authoring state is introduced.
Build and interactive input verification remain with the user.

### 2026-10-21 — Kullanıcı kararıyla geri alma

Kaynak/emitter/output materyal bağları, Separate UI, source IPC/Python işlemleri,
ayar serialization, emitter provenance/taşınım/RAM cache, v18 disk writer ve sparse
source render/upload/shader yolu kaldırıldı. Domain tek VolumeShader ve mevcut
NanoVDB/dense gaz yoluna döndü. VDB paneli eski tek editörüne döndü. Ortak volume
response/density curve/internal light ve bağımsız fizik çalışmaları korundu.
VkVolumeInstance source pointerları kaldırıldı: 672 byte / 168 uint; tüm Vulkan
RT ve RayFusion okuyucuları birlikte güncellendi. Disk writer tekrar v16;
v17/v18 deney cache'leri desteklenmez, simülasyon yeniden bake edilmelidir.
Kaydedilmiş sahnelerde gas_source_materials artık okunmaz/yazılmaz; eski ortak
fluid_fog_shader korunur. Output override'ları uygulanmaz.

Performans aktarımı bu partinin kapsamında yapılmadı. Görünür optik destekten
boş blok elemesi ve gölge için önhesaplanmış extinction fikri tek materyalli
yola uyarlanabilir; source packet kodu doğrudan eski sampler'a bağlanamaz.
Geri alma öncesi çalışma ağacı OS geçici klasöründe
raytrophi-source-material-before-rollback-20261021.zip adıyla saklandı.
