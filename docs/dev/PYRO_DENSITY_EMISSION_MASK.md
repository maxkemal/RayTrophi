# Pyro density emission mask — 2026-10-31

Durum: kaynakta uygulandı; C++/CUDA/shader build ve görsel kabul kullanıcıda.
Referans: https://www.sidefx.com/docs/houdini/nodes/vop/pyroadvancedshader.html
Hedef: yoğun kurum lobunu karanlık bırakıp daha ince aralıklarda alevi korumak.
Bu ilk sürüm density maskesidir; ayrı baked scatter alanı veya custom mask rampı değildir.

## Ortak sözleşme

Mask input = renderer'ın değerlendirdiği shading density / effective density multiplier.
Density remap/curve/cutoff ve varsa graph sonucu bu örneğin içindedir. Multiplier'ın
kendisi maske aralığını kaydırmaz. Bu ham simülasyon histogramı veya ayrı soot alanı değildir.
Cutoff/curve ayrıntıyı zaten silmişse maske onu geri getiremez.

```
lo = center - width / 2
hi = center + width / 2
factor = 1 - strength * smoothstep(lo, hi, mask_input)
emission_source *= factor       // radiance limitinden ÖNCE
```

`VolumeEmissionMask.h` aynı scalar çekirdeği CPU/CUDA/GLSL için tanımlar. Maske
sigma_s, sigma_a, transmittance veya sim verisini değiştirmez. Ek voxel grid yoktur.
Kamera, NEE, iç ışık integrali, radiance cache ve viewport aynı maskelenmiş kaynak
anlamını kullanır. Constant ve thermal emisyona uygulanır; thermal RT yolunda zaten
ortak olan volumeEmissionAt kullanılır. Mevcut renderer'ların diğer emisyon/ışık
denklemleri arasındaki farklar bu partide tümüyle giderilmiş sayılmaz.

Kapalı varsayılan eski sahne görünümünü korur. Maske `response_enabled`'dan bağımsızdır.
Strength=0 tam no-op; center yükselince daha fazla alev görünür, width geçişi yumuşatır.
Yeni sahnede doğru center/width alanın dağılımına göre seçilmeli; tek evrensel eşik yoktur.

## UI / scripting / IPC

Physical volume / fire response bölümündeki Density emission mask aynı
`volume_material::configure` servisini kullanır. `volume.get_material` hepsini döndürür.
`volume.set_material` ve `rt.volume.set_material` aynı ayarları atomik olarak doğrular:

| API ayarı | Tip / aralık | Varsayılan |
| --- | --- | --- |
| emission_mask_enabled | bool | false |
| emission_mask_center | finite number 0..100000 | 0.2 |
| emission_mask_width | finite number 0.000001..100000 | 0.1 |
| emission_mask_strength | finite number 0..1 | 1 |

Geçersiz tip/aralık veya bilinmeyen anahtar işlemi reddeder; kısmi malzeme değişmez.
Proje dosyasında mevcut emission_response nesnesinin mask_* alanlarında saklanır;
eski dosyada alanlar yoksa kapalıdır. Malzeme presetleri maskeyi ve sıcaklık birimini korur.

Build sonrası deneme (in-process rt örneği; canlı ölçümler harici IPC'den yapılmalı):

```python
rt.volume.set_material("gas:Nuclear Gas",
    emission_mask_enabled=True,
    emission_mask_center=0.2,
    emission_mask_width=0.1,
    emission_mask_strength=1.0)
```

Bu değerler başlangıç denemesidir. Alev fazla kaybolursa center yükselt; geniş
beyaz yüzeyler sürerse maske aralığını ve emisyon gücünü ayrı değerlendir.
Sahnedeki response_enabled=false ise +7.333 EV gibi saklı response ayarlarını
sırf maskeyi kullanmak için açma; maskenin kendi checkbox'ı yeterlidir.

## Çoklu kaynak ve cache

mask(a+b) doğrusal değildir; density toplayıp sonradan maskelemek bağımsız kaynakların
görüntüsünü değiştirebilir. Aktif maskeli emissive kaynaklar bu sürümde birleşmez.
Panel/IPC merge status nedeni açıklar; örtüşen çoklu kaynakta maliyet artabilir.
Tek hacimli patlamada ek slot yoktur. Maskeler kapalı/strength=0 iken gizli mask
parametre farkları merge identity'den çıkarılır; diğer optik ayarlar uygunsa birleşme döner.
Mask parametreleri radiance cache material stamp'e katılır.

## ABI / doğrulama

VkVolumeInstance sonuna byte 672'de dört float eklendi. Yeni stride **688 byte / 172 word**.
Dört GLSL struct, üç raw-word tüketicisi, iki backend paket yolu ve üç shader
publisher güncellendi. Eski binary ve yeni shader birlikte çalıştırılmamalı.
Tam C++/CUDA ve tam shader paketi gerekir; VolumeOnly yeterli değildir.

Non-build kontroller: mask (8), material (10), temperature (7), RT optical (24),
optical bounds (5), stage split (7): toplam 61 test; light-cache sözleşmesi de geçti.
C++ scalar testi `scripts/test/volume_emission_mask_test.cpp` teslim edildi, derlenmedi.
Harici post-build test `rt_test_volume_emission_mask_ipc.py` tip/aralık/atomiklik/
restore kontrolü yapar. Görsel A/B, kaydet/aç ve çoklu-source kabulü NEXT_BUILD_CHECKS başında.

## Vulkan / OptiX maliyetinin durumu

Önceki frame 74 ölçümü: adaptif kapalı OptiX ~212 ms, Vulkan ~500 ms; SVGF kapalı
Vulkan ~490 ms/viewport iterasyonu. Vulkan kamera RT GPU timestamp ~378 ms,
camera density samples/path ~31.3; secondary cache fallback çok düşük.
Yeni maskenin bu oranı düzelttiği iddia edilmiyor; build sonrası ölçülmeli.

Statik somut fark: OptiX NanoVDB ek ışık döngüsü doğrudan ışık katkısını ekler;
Vulkan sahne ışıkları için hacim gölge yürüyüşü/cache yolu kullanır. Optik denklemler
ve adım politikaları da birebir aynı değil. Bu farkın 2x içindeki payı ölçülmedi;
yalnız API/driver veya sıcaklık eşleme düzeltmesi sebep diye gösterilemez.

Termal ayrıştırma için `scripts/test/rt_probe_pyro_thermal_cost_ipc.py` eklendi.
Graph olmayan sahnede thermal strength=0 ve constant strength=0 sıfır emisyon
üretir; sigma ve density sabit kalır. ABBA GPU timestamp kıyası thermal fetch/
değerlendirme ve ilgili cache/emitter politikasının toplam farkını ölçer; tek
başına NanoVDB fetch fiyatı değildir. Kamera/kare değişirse kıyas reddedilir;
malzeme ve instrumentation geri yüklenir. Vulkan adaptif durumunu API'den okuyamadığı
için bunu doğrulanmamış olarak raporlar; sabit piksel yükü için UI'da kapatılmalı.

Bu tur uygulama kapalıydı (process ve named pipe yok); agent yeniden başlatmadı.
Yeni probe ve mask görsel kabulü koşulmadı. Bir sonraki ölçümde termal yol farkı
küçükse kamera lighting/majorant ve shader register/bellek fiyatı ayrı incelenmeli.

## Kullanıcının ayarladığı sahne incelemesi — frame 74

Canlı Vulkan, 128 spp, gas:Nuclear Gas ve vdb:108 malzemeleri aynı. Response ve
mask açık. Parametre değiştirilmeden render capture alınıp incelendi; capture
başlangıçtaki kapalı duruma döndü, geçici görüntü silindi.

Görüntüde koyu loblar ve kırmızı yüzey kıvrımları belirgin. Hedef referanstaki
yaygın çatlak/iç parıltı yerine ince, yüzeye işlenmiş çizgi/kabuk hissi sürüyor.
Bu önceki kadrajla kontrollü A/B değildir; kullanıcının kamera/malzeme ayarları değişmiş.

Ölçülen ayarlar ve cebirsel sonuçları:
- Density ramp ilk knot: x=1/7, y=0.826923; başlangıç eğimi 5.78846.
- Mask center/width 0.2/0.1, strength 1. Remap 0..1, cutoff 0.007. Geçiş aralığı
  mapped density 0.15..0.25 iken ilk ramp segmentinde ham density 0.025914..0.043189.
  Bu bölgede cutoff fade zaten 1. Daha büyük ham yoğunluklar tamamen maskelenir.
  Daha küçük yoğunluklar tam/aşamalı emisyon taşır; maske yalnız bir bandı açık tutmaz.
- Multiplier 17.361; mask merkezindeki shading density 3.4722. Response density power
  3.972 bu değerden ~140.37 çarpan üretir.
- Emission exposure +15.333 EV = ~41275.56 çarpan, blackbody strength 84.028.
  Temperature power 4.722 soğuk değerleri bastırırken diğer çarpanlar büyütüyor;
  bu karşılıklı telafi ince/kararlı bir ton aralığı kurulduğunu kanıtlamaz.
- Temperature range 0..20000, scale 6.15: yaklaşık 3252.03 K üzerinde renk rampı
  son durağa sabitlenir. Bu bölgede renk sıcaklık ayrımını taşımaz; power hâlâ
  şiddeti değiştirebilir. Kaç voxel/örneğin bu durumda olduğu sayılmadı.
- Internal light steps 0; bağımsız iç ışık integrali kapalı. Kaynak/camera glow
  varlığı, ayrı scatter/iç aydınlatma katmanının tamamlandığı anlamına gelmez.

Önemli tasarım sınırı: ilk mask girdisi opacity için remap edilmiş shading
density'den türetiliyor. Kullanıcının opacity ramp'ı maskeyi de daraltmış.
Sonraki düzeltme: bir voxel fetch'ten raw ve shaded density'yi birlikte taşımak;
maskeye bağımsız ham-source remap vermek, opacity ramp'ını bundan ayırmak.
Ardından kontrollü emisyon aralığı ve iç scatter katmanı değerlendirilmelidir.
Bu incelemede ayar veya uygulama kaynağı değiştirilmedi.

### Kullanıcının sonraki ayarları: eğri kapalı

Frame 74, Vulkan 128 spp, vdb:108. Density curve disabled; dolayısıyla önceki
eğri kaynaklı mask daralması bu görüntü için geçerli değil. Mask center/width/strength
0.2/0.1/1 ile fade aralığı tekrar raw density 0.15..0.25 (remap nötr, cutoff fade=1).
Bu durumda raw/shaded mask girdisi ayrımı tek başına görünümü değiştirmeyecektir.

Yeni multiplier 9.028, density power 3.778, temperature power 5.333, EV +9.778,
blackbody intensity 15.972, scale 1.2, temperature range 0..20000. Response/mask açık.
EV ~877.96 çarpan; ramp endpoint artık ~16666.67 K. Internal light steps=0, walk=false.
Görüntü önceki incelemeye göre daha yumuşak, tabandaki beyaz alev azalmış; koyu loblar
korunuyor. Parıltı hâlâ ağırlıkla lobların yüzeyinde/konturunda okunuyor; referansın
çatlaklardan gelen iç parıltısına ulaşıldığı söylenemez. Birden çok parametre birlikte
değiştiği için etki tek bir ayara veya implementasyona atfedilmedi.

İki bağımsız test adayı: density power 3.778 -> 1 (diğerleri sabit), ardından iç ışık
0 -> düşük adım sayısı (diğerleri sabit). Maskenin inverse yanıtıyla pozitif density
power'ın çarpımı dar bir yayıcı yoğunluk bölgesi seçebilir; gerçek alan dağılımı
ölçülmeden yalnız renderdan yeni bir veri kaybı teşhisi konulmadı. Ayarlar değiştirilmedi.
