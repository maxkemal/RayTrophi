# Pyro optik veri yolu planı

2026-11-01 kaynak güncellemesi: PYRO_INDEPENDENT_EMISSION_20261101.md.
Opt-in bağımsız j, ortak CPU/CUDA/GLSL response/integral, sıfır extinction limiti,
alıcı yakınından kaynak seçimi ve toplam aday bütçesi uygulandı. Yeni build/GPU
kabulü bekliyor. RT/RayFusion default ayrımı bu partiye katılmadı; kalıcı cache yok.

2026-10-31 ek uygulama: `PYRO_FIRST_SCATTER.md`. Kullanıcının yönlendirmesiyle
performans yeniden ölçülmedi; mevcut kayıtlar esas alındı. Maskelenmiş ortak
emisyon değerlendiricisini kullanan source-space RIS ilk saçılma yolu yazıldı.
Tek hacim ve mixture kamera march'ı, opt-in UI/Python/IPC; kalıcı cache yok.
C++/shader derlemesi ve GPU kabulü bekliyor. Ham kaynak AOV tanısı hâlâ açık;
bu katkı tam multiscatter veya referans görünümün tamamlandığı anlamına gelmiyor.

2026-10-31. Kaynak: RT_EXPLOSION_MATERIAL_AUDIT_20261031.md canlı karşılaştırmaları.
Hedef: simülasyondaki density/temperature/flame ayrıntısını koruyan, RT ve diğer
render tüketicilerinin aynı kaynak tanımını kullandığı duman/alev sistemi.

## Aşamalar ve kabul

Öncelik güncellemesi (2026-10-31): kullanıcının Houdini mask referansı üzerine
density emission mask görsel partisi HDR/histogram aşamasının önüne alındı.
Kaynakta uygulandı; `PYRO_DENSITY_EMISSION_MASK.md` ve NEXT_BUILD_CHECKS başı.
Koyu kurum/alev ayrımı hedefi, tüm planın veya iç scatter sisteminin tamamlandığı anlamına gelmez.

1. **Sıcaklık sözleşmesi (başlandı).** Kaynak birimini açık taşı; canlı Kelvin
   verisini normalize sanma; açık birimli soğuk/eksik alan density'den alev üretmesin.
   Ortak scalar çekirdek, UI + volume.get/set_material scripting/IPC, serialization,
   cache invalidation ve regression kontrolleri birlikte teslim edilir.
   Eski ithal VDB'ler için legacy davranış korunur; normalize/Kelvin açık seçilebilir.
   Kabul: canlı NanoVDB/dense yolunda 20 -> 20.01 süreksizliği yok; soğuk hücreler
   parlamaz; normalize ithal VDB ve kaydet/aç aynı sözleşmeyi korur.
2. **Kaynak tanısı ve HDR aralık.** Density/temperature/flame histogramları, ramp
   endpoint ve emisyon limit sayaçları; ham alan, extinction, emission, transmittance
   AOV'leri. Optik ölçek ve kaynak birimi raporlanır. Keyfi 64 sınırının yerine
   ölçülebilir/denetlenebilir kaynak politikası tasarlanır; körlemesine kaldırılmaz.
3. **Ortak optik örnek.** Tek çekirdek sigma_s, sigma_a, j ve phase üretir. Mevcut
   legacy source-function S ile hacimsel emissivity j açıkça ayrılır; j integrali
   sigma_t -> 0 limitinde j*dt olur. Kamera, NEE, iç integral, cache aynı tanımı okur.
   Density=0, emission>0 alanlar kaybolmaz. Eski sahneler için sürümlü migration.
4. **Flame ve duman ayrımı.** Mevcut combustion/reaction/flame alanının anlamı,
   adveksiyon/ömrü ve cache'i denetlenir; uygun kanallar simden NanoVDB/dense'e kadar
   korunur. Smoke extinction ve flame strength/color ayrı eğrilerden gelir.
   Yeni tam-domain grid eklemeden önce sparse/SoA maliyet hesabı yapılır.
5. **İç ışık.** NanoVDB dahil ortak j'den güç ağırlıklı emisyon örnekleme. Kaba
   radiance cache yüksek mertebeyi taşır; doğrudan kamera ve geçirgenlik ayrıntılı
   kalır. Cache tamper, sıcaklık/flame/material/ışık değişimlerinde yenilenir.
   Yaklaşık saçılma ile gerçek sekmeler aynı enerjiyi iki kez saymaz.
6. **Kabul/preset.** Aynı frame/kamera/spp ile alev kabuğu, soğuk pus, ışık sızıntısı,
   katı gölgesi, emissive-only ve eski VDB matrisi. Zaman ve görüntü birlikte ölçülür.

## Uygulama kuralları

- Yeni iş mantığı odaklı modüllerde; 2000+ satırlı dosyalara yalnız entegrasyon.
- UI, scripting ve IPC aynı material service/configure yolunu kullanır.
- Sıcaklığı örnek büyüklüğünden tahmin etme yalnız eski/kimliği belirsiz VDB uyum
  modunda kalır. Render politikası simülasyonun termodinamik alanını değiştirmez.
- Yeni cache anahtarları kaynak birimi ve tüm optik girdileri kapsar.
- Agent build/shader derlemesi çalıştırmaz. Yeni binary/SPV olmadan canlı kabul
  iddiası yapılmaz. Her aşamanın kaynak/test ve kullanıcı checklist'i ayrı kaydedilir.

## Bu tur kapsamı

İlk aşamanın uygulanması ve kaynak doğrulaması. Aşama 2–6 bu turda tamamlanmış
sayılmaz; tam optik ayrım veya emissive random walk iddiası yoktur.

## İlk parti: kaynakta tamam, derleme/canlı kabul bekliyor

- `VolumeTemperatureMapping.h`: C++/CUDA/GLSL ortak scalar sözleşme. Explicit
  Kelvin/normalize alanları eksik/soğuk olduğunda density fallback yok.
- `volume.get_material` / `volume.set_material` (`rt.volume` ve IPC):
  `temperature_units` integer 0..2. UI aynı configure çekirdeğini kullanır.
  Serialization mevcut emission_response içine eklenir; preset birimi korur.
- Canlı NanoVDB metadata'sı source-default'u Kelvin olarak çözer; ithal eski
  VDB legacy inference kullanır. GPU pakette mevcut density_curve[10] kullanılır,
  stride/offset büyümez. Yoğunluk eğrisi ve temperature policy ayrı anlamdadır.
- RT kamera/NEE/iç ışık ortak emission değerlendiricisi kullanır. Preview ve
  CPU/CUDA sıcaklık eşlemesi aynı scalar çekirdeğe bağlıdır. Mevcut renderer'ların
  diğer emisyon denklemleri bu partide tamamen eşitlenmiş sayılmaz.
- Radiance cache unit/min/max/scale değişimini izler; merge key optik kimliği
  de içerir. Bununla uyumlu iki merged kaynağın birlikte düzenlenmesi de yeniden yayılır.
- `check_volume_temperature_contracts.py` (7), material (10), optical bounds (5),
  light-cache kaynak kontrolleri geçti. C++ scalar ve harici IPC post-build testleri
  teslim edildi, agent tarafından build/run yapılmadı.
- Canlı kabul ve tam build sırası `NEXT_BUILD_CHECKS.md` başındadır.

### Build sonrası geri bildirim

Kullanıcı belirgin kalite artışı görmedi. Açık binary'de unit=1 ve yeni SPV scalar
sembolü doğrulandı; özel cold-field kabulü hâlâ açık. Frame 74 backend kıyasında
adaptif kapalı OptiX ~212 ms, Vulkan ~500 ms; SVGF kapalı Vulkan ~490 ms/iterasyon.
Bu saf GPU kıyası değil, doğal viewport duvar saati ölçümüdür. Kamera density
örnek yükü yüksek, secondary-cache fallback çok düşük. Aşama 2'de kaynak clamp/
histogram tanısına ek olarak filtre açık/kapalı görsel A/B yapılmalı: ana denoiser
kapalı olsa bile ayrı SVGF yolu ilerleme sırasında aktif olabilir. Detaylar ve
ölçüm sınırları audit notunun sonundadır. Sıcaklık doğruluğu paketi tam alev
kalitesi iyileştirmesi olarak kabul edilmedi.

### Kaynak şekli denemeleri

`PYRO_SOURCE_SHAPE_DIAG_20261031.md`: aynı sahnede altı kontrollü ve iki keşif
renderı alındı, SVGF kapalı tutuldu ve ayarlar tamamen restore edildi. Maske
güçlü etki ediyor; yalnız duman görüntüsünde yapısal ayrıntı var. Parametre
denemeleri referansın iç parıltısını oluşturmadı. Bunlar beauty renderlar olduğu
için ham iç emisyon dağılımı kesin teşhis edilmedi. Sonraki kaynak işi ortak
değerlendiriciden alan/AOV tanısı ve kamera için ayrı fire in-scatter cache
bileşeni olarak daraltıldı; secondary J'yi primary'ye ekleyerek ışığı çift sayma.
