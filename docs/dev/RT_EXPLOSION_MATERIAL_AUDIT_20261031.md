# Açık patlama sahnesi: RT malzeme incelemesi

2026-10-31. Harici `scripts/test/rt_ipc.py` ile canlı inceleme. Build yapılmadı.
Kullanıcının yaklaşık 70 milyon hücreli olarak belirttiği sahne. Hücre sayısı bu
incelemede yeniden sayılmadı. Render edilen kaynak tek NanoVDB slotu, sıcaklık mevcut.

## Gözlem ve kontrollü karşılaştırmalar

Başlangıç görüntüsü: 1680x945, 128 spp. Baş kısmen kadraj dışında. Alev lobları geniş
krem/beyaz alanlara dönüşüyor; kırmızı/siyah sınırlar belirgin, iç hacim yerine kabuk
izlenimi var. Denoiser kapalı; bu görünüm denoiser'a bağlanamaz.

Özgün malzeme: density multiplier 13.194, blackbody intensity 16.667, ramp açık,
temperature min/max 1340/20000, scale 1.337, response kapalı, anisotropy 0.99,
back anisotropy -0.3, lobe mix 0.7, scatter 0.84, absorption 1.72,
multi_scatter 0.85, random_walk false, internal_light_steps 0, max_steps 192,
voxel_step_multiplier 0.7. Post: AgX, exposure 1, gamma/look 0.648, saturation 1.113.

Yalnız blackbody intensity değiştirilerek:
- 4.16675: beyazlık bir miktar azalıyor ama geniş doygun alanlar sürüyor (52 spp).
- 0.5: sıcak loblarda iç ton/kıvrım farkları daha okunur; kabuk görünümü sürüyor (61 spp).

Yalnız density multiplier 3.2985 yapıldığında alev daha beyaz ve yumuşak, duman daha
ince oluyor (52 spp). Density düşürmek tek başına çözüm değil. Bu karşılaştırmalar
malzeme etkisini görmek içindir; eşit spp performans veya kalite ölçümü değildir.

Tüm denemeler sonunda intensity 16.667 ve multiplier 13.194 geri okuma ile doğrulandı.
Capture tekrar kapalı. Proje kaydedilmedi. Geçici ekran görüntüleri silindi.

## Kaynakta doğrulanan sınırlar

1. `volume_fog_shading.glsl`: tek hacim random walk yolu `emission_mode == 0` ister.
   Bu emissive slotta yalnız random_walk açmak gerçek hacim çoklu saçılmasına geçirmez.
   `multi_scatter` normal march'ta gökyüzü kaynaklı yüksek mertebe yaklaşımına katkıdır;
   alevin çoklu saçılmalı taşınımının genel çözümü değildir.
2. Aynı dosyadaki emitter NEE yolu emissive list ve majorant adreslerine bağlıdır.
   Bunların renderer yayını dense alanlar üzerinden gelir. NanoVDB kaynağında aynı
   desteğin varlığı varsayılamaz. Genel iç integral mevcut ama bu sahnede steps=0.
   Katıların alevle aydınlanması, iç duman taşınımının tamamlandığını kanıtlamaz.
3. `volume_internal_rt.glsl`: ramp açıkken legacyRadiance=1; sıcaklığın ayrıca
   parlaklık yanıtı yalnız response açıkken uygulanır. Bu sahnede response kapalı;
   source = rampRGB * mappedDensity * intensity. Sonuç ayrıca luminance 64 ile sınırlı.
   Ramp alpha `sampleColorRamp` tarafından okunmaz. Renk rampı ile emisyon gücü eğrisi
   aynı şey değildir; alpha'nın bir güç eğrisi olması isteniyorsa açık bir sözleşme gerekir.
4. temperature scale 1.337 ramp koordinatına uygulanıp clamp edilir: u >= ~0.748
   son renk durağına düşer. Bu, mevcut Kelvin dönüşümünde yaklaşık 15297 K üstünün
   aynı ramp RGB'sini almasıdır. Kaç hücrenin bu aralıkta olduğu ölçülmedi.
5. `max_steps=192` bütün kesişim aralığına dağıtılır; gereken adım sayısı bütçeyi
   aşınca gerçek adım büyür. Yüksek voxel çözünürlüğü render'da otomatik korunmaz.
   Bunu doğrulamak için aynı materyal/kamerada yalnız max_steps 192/384/768 A/B gerekir.
6. NanoVDB gas yayınında `scene_data.h` normalize heat alanını 3000 ile çarpar.
   Bu sanal Kelvin ölçeğidir; termodinamik sıcaklığın doğruluğu bu incelemede kanıtlanmadı.

## Öncelik

Önce nötr look ile emisyon yanıtını ve ramp dinamik aralığını ayır; sonra kamera march
çözünürlüğünü A/B ile doğrula. Ardından NanoVDB dahil aynı emisyon kaynağını kullanan
iç hacim ışık taşınımını tamamla. Emissive random walk desteği ayrı integratör işi;
mevcut koşulu kaldırmak tek başına yeterli değildir. Daha çok voxel veya bounce
vermek bu eksikleri kendiliğinden çözmez. İnceleme kapsamında kaynak kodu değiştirilmedi.

## Sönümleme karşılaştırması (2026-10-31)

Kullanıcının onayıyla yalnız scatter/absorption katsayıları birlikte ölçeklendi.
Emisyon, density multiplier, ramp ve diğer malzeme ayarlarının değişmediği her
varyantta get_material ile doğrulandı. Baseline ve iki varyant 1680x945, 128 spp.
İşlem sırasında timeline.get_frame=69 okundu; simülasyona/kameraya komut verilmedi.

| Ölçek | Scatter | Absorption | Baş ROI ortalama luma | Luma >=235 payı |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.84 | 1.72 | 165.52 | %9.66 |
| 0.5 | 0.42 | 0.86 | 184.46 | %18.13 |
| 0.25 | 0.21 | 0.43 | 193.10 | %26.52 |

ROI: JPEG'de x=[650,1300), y=[0,460), aynı ekran bölgesi. Luma Rec.709 ağırlıklı
8-bit görüntü değeridir; HDR enerji, gerçek clipping veya yalnız alev maskesi
ölçümü değildir. Arka plan da ROI'ye dahildir.

Gözlem: daha düşük sönümleme alev loblarını daha beyaz/tekdüze yaptı; beklenen iç
katman ayrımı ortaya çıkmadı. Duman ve zemin daha geçirgen/aydınlık oldu. Bu,
yalnız optik kalınlığı düşürmenin çözüm olmadığını gösterir; opaklığın hiç etkisi
olmadığını kanıtlamaz. Parlaklık/ramp sıkışması iç detay değerlendirmesini sınırlıyor.

Önemli deney sınırı: shader'da katkı `T * emis * (1-exp(-sigma_t*dt))` olarak
birikir. Emisyon AYARLARI sabit olsa da katsayıları değiştirmek bu ağırlığı da
değiştirir. Bu bir sabit hacimsel emissivity (j) deneyi değildir; mevcut kaynak
fonksiyonu sözleşmesinin fiziksel j ile aynı olduğu varsayılamaz. Önceki "emisyonu
sabit tutmak" ifadesi bu ayrımla okunmalıdır. Parametre azaltımıyla kesin bir
integratör hatası veya yalnız simülasyon hatası teşhis edilmedi.

Sonraki anlamlı kontrol: doygunluğu azaltılmış sabit bir emisyon/look altında
aynı sönümleme matrisi; ardından yalnız march bütçesi 192/384/768 karşılaştırması.
Kalıcı kontrol veya integratör değişikliği bundan ayrı ele alınmalı.

İşlem finally bloğunda özgün scatter/absorption geri yüklendi ve malzeme JSON'u
başlangıçla tamamen eşit bulundu. Capture başlangıçtaki kapalı durumuna döndü.
Proje kaydedilmedi. Geçici script, JSON ve bu deneyde üretilen görüntüler silindi.

## Ayrıntı kaybı matrisi (2026-10-31)

Harici IPC, aynı frame 69/kamera, 1680x945, denoiser kapalı. Her karede timeline
frame ve değişmemesi gereken materyal anahtarları kontrol edildi. Emisyon 0.5,
post gamma/look 1 ve saturation 1; AgX ve diğer post ayarları korundu. Hedef 64 spp,
poll/capture gecikmesi nedeniyle 64–66 spp. Bunlar görüntü karşılaştırmalarıdır;
GPU performans ölçümü veya bit düzeyinde eşit örnekli referans değildir.

| Varyant | spp (capture öncesi) | ROI luma ort. | RGB MAE / düşük-emisyon tabanı |
| --- | ---: | ---: | ---: |
| Taban: 192 adım, özgün katsayılar | 65 | 133.683 | 0 |
| Sönümleme 1/2 | 64 | 155.384 | 20.644 |
| Sönümleme 1/4 | 65 | 173.343 | 38.040 |
| 384 adım | 64 | 133.081 | 1.308 |
| 768 adım | 64 | 133.083 | 1.306 |
| Temperature scale 1.337 yerine 0.7 | 66 | 113.972 | 17.830 |

ROI öncekiyle aynı, JPEG RGB 0–255; HDR veya alev maskesi ölçümü değil.
384 ile 768 adımın birbirine RGB MAE'si 0.093, kanal mutlak fark p99=2.
Gözle 192→384 ufak fark, 384→768 belirgin yeni iç yapı sağlamıyor. Bu kadraj ve
malzemede adım bütçesi baskın kayıp gibi görünmüyor; farklı kamera veya daha ince
alev yapısı için genellenemez. Düşük emisyon altında da sönümlemeyi düşürmek
lobları daha açık/tekdüze yapıyor. Temperature scale değişikliği tonları ve
loblardaki renk ayrımını adım artışından çok daha fazla etkiliyor; salt detay
metriği değildir, parlaklık/renk değişimini de içerir.

### Düzleştirme nerede?

- **Veri dönüşümü:** canlı NanoVDB yayınında FloatGrid → createNanoGrid kullanılıyor.
  Sparse tolerans density 1e-6, temperature 1e-3; bu kaynakta fp16/8-bit quantization
  veya tüm sıcaklıkları tek değere indiren normalizasyon bulunmadı. Bu statik
  denetim, GPU voxel değerlerinin sim ile bit eşitliğini ölçmüş sayılmaz.
- **Ramp clamp:** `volume_internal_rt.glsl:15` u*scale koordinatını 0..1'e kırpar.
  Özgün min/max 1340/20000 ve scale 1.337 ile ~15296.62 K üzerindeki tüm sıcaklıklar
  son RGB durağını alır. Response kapalı olduğundan ramp yolunda legacyRadiance=1;
  sıcaklığın ayrıca bir güç eğrisi yok. Bu aralıktaki sıcaklık farklarının kaynak
  emisyonunda kaybolması koddan doğrudan çıkar; bu hücrelerin oranı ölçülmedi.
- **Kaynak luminance clamp:** aynı dosya satır 20 kaynak emisyonunu luminance 64
  ile sınırlar. Son ramp rengi (1,1,0.8), intensity 16.667, multiplier 13.194
  için ham density >=0.29530 sınırı aşar (remap nötr, eğri kapalı ve cutoff fade=1).
  Renk sabitse üstündeki yoğunluk farkları emisyon kaynağında tamamen eşitlenir;
  extinction yine farklıdır. gas.get raporu max_density=2.32884,
  active_density_cells=1460481, voxel_size=0.1328125. Bu alan istatistikleri
  yayınlanan grid'in histogramı değildir; clip edilen hücre sayısı ölçülmedi.
  Emisyon 0.5 deneyinde aynı max density varsayımıyla kaynak en fazla ~15.364
  olabilir; dolayısıyla 64 sınırı bu deneyde baskın olamaz.
- **Birim tahmini / soğuk hücre hatası adayı:** `kelvinUnits` yalnız dense
  volume_type=4/source_type=5 için true. NanoVDB canlı yayını Kelvin ölçeğinde
  olmasına rağmen shader değer <=20 ise normalized kabul ediyor. Mevcut aralıkta
  temperature=5 veya 20 son beyaz durağa, 20.01 ise ilk siyah durağa gider.
  Bu kesin bir fonksiyon süreksizliğidir; sahnede hangi pikselleri etkilediği
  ölçülmedi. Ayrıca temperature<=0 density ile ikame edilir; eksik/soğuk heat
  alanı dolu density üzerinden yapay alev üretebilir. Live bridge'in heat*3000
  sözleşmesiyle bu tahmin uyumsuzdur. İthal VDB'lerin normalize/Kelvin ayrımını
  koruyacak açık birim metadatası gerekir; yalnız koşulu silmek yeterli değildir.
- **Taşınım:** T hızla sıfıra yaklaşınca arkadaki farklar örtülür; emisyon da
  1-exp(-sigma_t*dt) ile ağırlanır. Sim verisi değişmese de görüntü düzleşebilir.

Sonuç: kaynakta doğrulanmış düzleştirmeler ramp endpoint ve emisyon clamp'idir.
Float grid'in baştan düzleştirildiği kanıtlanmadı. Öncelik canlı sıcaklık birimini
ve sıfır heat fallback sözleşmesini düzeltmek, sonra kaynak emisyonunun dinamik
aralığını ölçerek korumak. Clamp'i körlemesine kaldırmak gürültü/firefly ve eski
sahne görünümü riskleri taşır; histogram/clip sayaçlarıyla değerlendirilmelidir.
Ardından emisyon–sönümleme bağı ve emissive iç ışık taşınımı ele alınmalıdır.

Özgün materyal ve post JSON'larının tamamı finally sonrası başlangıçla eşit
doğrulandı. Capture önceki kapalı durumuna döndü; proje kaydedilmedi. Bu partide
uygulama/shader kaynak kodu değiştirilmedi ve build çalıştırılmadı. Geçici script,
JSON ve altı doğrulama görüntüsü görev sonunda silindi.

## İlk paket sonrası canlı durum ve backend maliyeti (2026-10-31)

Kullanıcı build sonrası belirgin görsel fark görmediğini, Vulkan'ın OptiX'ten
yavaş olduğunu bildirdi. Yeni binary `temperature_units=1` döndürüyor. Diskteki
`x64/Release/shaders/volume_closesthit.spv` içinde `volumeMapTemperature` sembolü
var; Vulkan pipeline ready. Bunlar yeni yolun kurulumu için kanıt, özel soğuk
hücre fixture'ının piksel düzeyinde kabulü yerine geçmez. İlk paket ramp endpoint
ve kaynak 64 limitini kaldırmadığı için büyük beyaz alev bölgelerinin düzelmesi
beklenmemeli; bu kalite hedefi hâlâ açık.

Bu tur frame **74**, önceki frame 69 ölçümleriyle birebir süre kıyası değil.
Density multiplier artık 10.417; birim 1. `response_enabled=false` iken
temperature_power=1.555, density_power=0.25, emission_exposure_ev=7.333 ve
emission_saturation=1.583 denkleme katılmaz. Bunları değiştirmek bağımsız response
açılmadan etkisizdir. +7.333 EV ile response'u doğrudan açmak doğru bir nötr A/B
değildir; ölçümde bu ayarlara dokunulmadı.

### Zaman ölçümü

Aynı kamera, frame, malzeme; doğal viewport ilerlemesinde sample 8→32, iki tekrar.
Capture kapalı, sayaçlar kapalı. Aşağıdakiler **duvar saati / ilerleme iterasyonu**,
saf GPU event süresi veya eşit görüntü hatasında hız kıyası değildir. Backend'lerin
integratörleri farklıdır. Adaptif kapalıyken erken durdurma etkisi ayrıldı.

| Backend / durum | Tekrar 1 ms | Tekrar 2 ms |
| --- | ---: | ---: |
| OptiX, adaptif açık | 185.58 | 185.58 |
| OptiX, adaptif kapalı | 212.21 | 212.25 |
| Vulkan, adaptif kapalı, SVGF açık | 500.00 | 499.38 |
| Vulkan, adaptif kapalı, SVGF kapalı | 489.58 | 490.25 |

Bu sahne/kod yolunda tam piksel örneklemede Vulkan ~2.35x, SVGF kapalıyken ~2.31x
daha yavaş. Filtre tek başına farkı açıklamıyor. OptiX adaptif açıkken global
iteration 36 civarında piksel örnek ortalaması 4.647; kapalıyken mean/max=36.
Bu yüzden global sample sayısı tek başına eşit ışın işi demek değildir.

Ayrı Vulkan GPU timestamp geçişi, başlangıçtaki adaptif açık durumda:
- kamera RT dispatch ortalaması 377.897 ms (8 warm-up sonrası 16 örnek),
- kamera density samples/path 31.319,
- secondary density samples/path 0.000114,
- shadow density samples/path 2.081,
- majorant queries/path 13.018,
- radiance cache marches/path 0.15134, fallback/path ~0.00000182.

Ana aday kamera volume march ve örnek başına NanoVDB/temperature işidir; sayaçlar
tek başına bu işin instruction/bellek maliyetini ayırmaz. Sekme cache fallback'i
bu sahnede baskın değil. Önce emisyon/temperature fetch ve shadow/cache-fill
maliyetini ayrı A/B ile ayır; performans farkını doğrudan yeni scalar unit
kontrolüne veya Vulkan API'sine bağlama. Eski/yeni binary performans A/B yapılmadı.

### Denoiser değerlendirmesine düzeltme

Önceki "denoiser kapalı" tespiti ana `render.get_denoiser` ayarına dayanıyordu.
Ayrı `render.realtime_denoiser_stats` SVGF'nin açık olduğunu gösterdi. Adaptif
açık son geçişlerde ~%89.6 filtered fraction ve ~11.90 ms toplam GPU maliyeti
raporladı. Tamamlanmış görüntü sorgusunda presenting=false iken ilerleme sırasında
presenting/static_active=true görüldü. Filtrenin açık olması tek başına son
128-spp görüntünün filtreli olduğunu kanıtlamaz; 64-spp önceki karşılaştırmaların
ham render diye yorumlanmaması gerekir. Görsel flattening etkisi ayrı filtre
açık/kapalı, aynı spp görüntü A/B ile doğrulanmalı.

Ham raporlar: `rt_pyro_vulkan_cost_20261031.json`,
`rt_pyro_backend_compare_20261031.json`.
Her ölçüm sonunda frame/kamera/malzeme eşitliği doğrulandı. Başlangıçtaki Vulkan,
adaptif açık ve SVGF açık geri yüklendi; UI/API ile kontrol edildi. Proje
kaydedilmedi. Kaynak veya shader değişikliği/build yapılmadı. Geçici probe silindi.
