# RayFusion — speküler sky görünürlüğü

Durum: **YAZILDI, DERLENMEDİ** (2026-09-08).

## Kullanıcı testi ve canlı inceleme

İlk teslimdeki derlenmedi etiketi tarihsel kayıttır. Kullanıcı yeni sürümde
çoğu yüzeyde mavi specular tabakanın kaybolduğunu, sandalye kıvrımlarında
neredeyse kalmadığını, koltukta ise sürdüğünü bildirdi. Tam kabul henüz yok.

Canlı IPC: quality preset, traced aktif, **bounce kapalı**, 32/32 valid,
pending=0, hit_fraction=0.10546875. Güncel görüntü
`tmp/rayfusion_specular_live.jpg`; geçici capture eski durumuna döndürüldü.
Önceki görüntüyle kamera ve bounce durumu aynı değil; nicel A/B sayılmaz.

Kamera ışınlarıyla salt okunur `scene.raycast` örnekleri:

| Yüzey | Dünya konumu (yaklaşık) | Probe alanında |
|---|---|---|
| 1_sofa_02_Base | (2.962, 1.110, -0.299) | Evet |
| 1_sofa_02_Seat | (2.168, 0.994, 0.003) | Evet |
| 1_Ottoman_01 | (1.724, 0.827, 0.558) | Evet |
| 1_dining_chair_02.002 | (1.630, 1.408, 1.025) | Evet |

Bu noktalardaki kalıntıyı pencere dışı fallback açıklamaz. Alan içinde olmak,
komşu probe'un aynı odada veya yansıma yönünün doğru görünürlükte olması
anlamına gelmez. Düşük roughness/metalik değerler kapsam kapısı değildir.
Kalan etki için probe-origin parallax/yön çözünürlüğü ile raster/RT speküler
lob farkı ayrıştırılmalıdır; henüz kök neden olarak biri seçilmedi.

Kullanıcı aynı sahnede materyal specular değerini sıfırlayınca mavi tabakanın
yaklaşık %98 kaybolduğunu bildirdi. Bu görsel kullanıcı tahminidir; piksel
ölçümü değildir. Kaynakta diffuse probe görünürlüğü uygulanırken environment
specular katkısının engellenmeden kalmasıyla uyumludur.

## Uygulama

`rayfusion_specular_visibility.glsl`, sekiz komşunun yansıma yönündeki mesafe
momentlerinden ayrı bir sky görünürlük tahmini üretir. Probe-yüzey bağlantısı
da momentlerle ağırlıklandırılır. Işık rengi hâlâ speküler environment yolundan
gelir; diffuse irradiance yansıma olarak kullanılmaz. Materyal specular/F0,
doğrudan ışık parlamaları, transmission ve geçerli ekran uzayı yansıması değişmez.
Scene lighting (`lightingPreset=3`) yolundaki environment specular, clearcoat
katkısı dahil, bu tahminle çarpılır. Studio lighting bundan etkilenmez.

CPU gerçek traced yayın için ortak `kProbeTraceDistance=200` değerini mevcut
grid uniform'ının ayrılmış `spacing.y` kanalına yazar. Sky bake için sıfırdır.
Sadece traced isteğini okumaz; fallback sky bake yanlışlıkla engellenmez.
Buffer boyutu ve descriptor düzeni değişmedi; uniform kanal semantiği genişledi.

## Kontrol yüzeyleri ve sınırlar

Yeni kullanıcı işlemi yoktur. Mevcut ortak üretici servisi üzerinden UI
**Trace probe rays**, Python `rt.rayfusion.set_probe_producer(True/False)` ve
IPC `rayfusion.set_probe_producer {"traced":true/false}` düzeltmeyi de kontrol
eder. Mevcut boolean doğrulaması ve hata semantiği değişmedi.
`rayfusion.probe_field` / Python `probe_field()` içindeki gerçek `producer`,
`pending`, `valid` ve `traced_publishes` ile yayın doğrulanır. Traced kapatmak
diffuse üreticiyi de değiştirir; bu karşılaştırma saf specular A/B testi değildir.

Bu, **probe merkezinden yönlü moment tahmini**, pikselden kesin RT değildir.
200 birim ötesindeki engeller, seyrek ızgara, parallax ve küçük pencere açıklığı
çözülmüş değildir. Alan dışında veya kullanılabilir probe yoksa eski sky
fallback korunur. Sabit yön çözünürlüğü ve tek yansıma yönü kullanılır;
roughness lobu boyunca görünürlük integrali ve clearcoat için ayrı görünürlük
lobu yoktur. Özellikle kaba yüzeylerde fazla kararma veya dar yansımalarda
basamak görülebilir. Yerel sahne yansımaları eklenmedi; kapalı sky yerine
eksik yerel yansımanın gelmesi bu dilimin dışında.

## Kontrol ve kullanıcı build listesi

Kaynak kontrolü: `python scripts/audit_rayfusion_specular_visibility.py`.
Sayısal referans/kaynak denetimidir; GPU çalıştırması değildir.

1. C++ projesini derle; grid uniform'ına gerçek traced horizon yayını eklendi.
2. `material_preview_frag.frag` dosyasından kullanılan
   `material_preview_frag.spv` dosyasını yeniden üret. Yeni GLSL include gerekli.
   Trace compute kaynakları değişmedi; ortak horizon hâlâ 200.
3. Mevcut oda sahnesinde materyalin özgün specular değerini koru. Aynı kamera,
   pozlama ve ışıklarla koltuk/puf/sandalye kıvrımlarını karşılaştır. Kapalı
   yöndeki mavi sky azalmalı, gerçek doğrudan ışık parlamaları kalmalı.
4. Açık sky sahnesinde specular kaybolmamalı. Pencereli odada açıklık yönü,
   kapalı yön ve farklı roughness/clearcoat değerlerini kontrol et; aşırı
   kararma veya titreşim/katman oluşumu kabul bulgusu olarak kaydedilmeli.
5. Traced kapalı sky bake, alan dışı ve eksik probe fallback davranışlarını
   kontrol et. Traced/bounce değişiminden sonra viewport karesi ürettir ve
   `traced_publishes` artmadan ekranı yeni sonuç olarak değerlendirme.
6. Ekran uzayı yansımasının ve nokta/yönlü ışık spekülerlerinin korunmasını
   kontrol et. Aynı çözünürlükte capture kapalı kare süresini karşılaştır:
   ek tüketici maliyeti en fazla 16 probe texel okumasıdır; ölçülmedi.
