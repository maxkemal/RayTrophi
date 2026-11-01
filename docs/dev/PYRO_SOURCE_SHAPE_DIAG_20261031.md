# Pyro kaynak şekli: canlı karşılaştırmalar

2026-10-31, kullanıcının sahne üzerinde deneme yetkisiyle harici rt_ipc istemcisi.
Frame 74, aynı kamera, Vulkan, 1680x945. SVGF tüm karşılaştırmalarda kapalı.
Global iteration hedefi 64; capture öncesi sayaç 65–68. Adaptif örnekleme durumuna
müdahale edilmedi; bunlar eşit-piksel-spp veya eşit-hata performans kıyasları değil.
Uygulama/shader derlemesi ve kaynak kodu değişikliği yapılmadı.

## Altı kontrollü durum

Sigma, density alanı/curve/multiplier, kamera ve post sabit. Başlangıç response:
density power 3.778, temperature power 5.333, emission EV +9.778, intensity 15.972;
mask center 0.2, width 0.1, strength 1. Density curve kapalı. Internal light steps=0.

| Durum | Global sample | Baş ROI luma ort. | Luma >=235 payı |
| --- | ---: | ---: | ---: |
| Özgün maskeli | 65 | 108.886 | %0 |
| Yalnız maske kapalı | 65 | 186.244 | %41.92 |
| Maske kapalı, emission EV=0 | 65 | 160.143 | %15.83 |
| Maske kapalı, emission EV=-14 | 65 | 103.443 | %0 |
| Emission mode=0, yalnız duman | 68 | 102.895 | %0 |
| Özgün maskeli, yalnız density power=1 | 68 | 104.527 | %0 |

ROI x=[720,1170), y=[145,550), Rec.709 ağırlıklı 8-bit JPEG değerleri. Arka plan
da içerir; HDR enerji veya shader limitine takılan örnek oranı değildir.

- Maskesiz özgün ayar lobların tamamını çok parlak gösteriyor. Maske etkisiz değil.
- EV=0 ile kıvrımlar daha okunur, geniş parlak loblar hâlâ var.
- EV=-14 görüntüsü yalnız dumana çok yakın, birkaç sıcak iz kalıyor.
- Yalnız duman görüntüsünde loblar ve ince kıvrımlar mevcut. Tüm ayrıntının kaynak
  grid'de kaybolduğu iddiası bu görüntülerle desteklenmiyor.
- Density power=1, özgün maskeli görüntüyü ağırlıkla karartıyor; beklenen iç
  aydınlanmayı ortaya çıkarmıyor. Bu tek parametreyle sorun çözülmedi.

Domain'in raporladığı max_density=1.729606 ve multiplier=9.028 ile shading density
üst sınırı ~15.61488. Eğri/remap nötr kabulüyle EV=-14 kaynak üst sınırı birim renk
için ~31.4868; konservatif 2x renk sınırıyla ~62.9736. Aynı alan yayınlandığı ve
graph/noise density üretmediği varsayımında 64 limiter bu tanıda baskın olamaz.
Bu voxel histogramı veya gerçek GPU clip sayacı ölçümü değildir.

## İki keşif adayı

Çok parametreli bir blackbody denemesi: density power=0, temperature power=2,
EV=0, saturation=1, intensity=6, temperature range=1340..8000, scale=1, ramp kapalı;
mask center/width=0.35/0.5. Önce özgün multiplier=9.028, ardından yalnız multiplier=3.
İkisi de 65 global örnek. Geniş açık/gri loblar ürettiler; hedefteki koyu kurum
aralarındaki sıcak iç parıltıyı vermediler. Kalıcı preset olarak seçilmediler.
Bu keşif çok parametreli olduğu için tek bir dönüşümün hatası/başarısı olarak yorumlanamaz.
İlk çalışma host-double/C++-float karşılaştırması nedeniyle geçersiz sayılıp restore
edildi; scalar değerler float32'ye çevrildikten sonraki çalışma tamamlandı.

## Kanıt sınırı

Bu sekiz görüntü **beauty render**dır. Extinction korunmuştur; ham 3B emisyon
kesiti, transmittance-free emission AOV veya gerçek mask AOV değildir. Dolayısıyla
kamerada görülmeyen iç kaynak dağılımı veya fiziksel doğruluk kesin teşhis edilmedi.
Maske kapatılınca beyazlık görmek, içeride doğru ışık taşındığını kanıtlamaz.

Kaynak kodunda mevcut radiance cache'in kamera segmentinde kullanılmadığı ve
emitter-NEE katkısını içermediği açık. Ayrı internal integrali kapalı olan bu
sahnenin son görünümü tam bir alev→duman saçılma çözümü diye kabul edilmemeli.
Fakat yeni cache'in aynı dar/yanlış kaynağı örnekleyerek hedefi otomatik vereceği
de varsayılmamalı.

## Daraltılmış sonraki geliştirme

1. Aynı optik değerlendirmeden raw density, gerçek temperature birimi, mask,
   limiter öncesi/sonrası kaynak ve transmittance tanısı. Ham kesit/alan örneği ile
   görüntüdeki örtülmeyi ayır. Ayrı CPU emisyon denklemi kopyalama.
2. Kamera yolu için yalnız **alevden gelen in-scatter** taşıyan ayrı cache bileşeni.
   Mevcut secondary radiance J'sini primary'ye doğrudan ekleme: doğrudan emisyon
   ve harici aydınlatma iki kez sayılabilir.
3. Emisyon desteğine/gücüne göre örnekleme veya seyrek yayılım. Mevcut internal
   integral bbox boyunca uniform örnekler; ince maskelenmiş kaynakları kaçırma
   ihtimali ve yüksek maliyeti var. Bu mekanizmanın sayısal payı henüz ölçülmedi.
4. Kaynak aynı canonical masked emission/optics sözleşmesinden gelir. Mevcut
   modelde source-function S ve j=sigma_t*S ilişkisi açık tutulur; sim termal
   enerjisi doğrudan ışık olarak kullanılmaz. Sanatçı fire/scatter bileşenleri
   ayrılacaksa enerji ve görüntü sözleşmesi açıkça tanımlanır.
5. Coarse/sparse cache, hücre/stride bazlı lookup ve kaynak/material/frame
   invalidation. Yeni tam-domain RGB dense grid veya her fine camera adımında
   yeniden çoklu cache-tree yürüyüşü varsayılan çözüm değil.

Kayıtlar: `rt_pyro_source_shape_20261031.json`, `rt_pyro_blackbody_candidate_20261031.json`.
Özgün materyal, kamera, frame ve post tam eşitlikle doğrulandı; capture ve SVGF
başlangıç durumuna geri döndü. Proje kaydedilmedi. Sekiz geçici görüntü ve geçici
script/JSON dosyaları görev sonunda silindi.
