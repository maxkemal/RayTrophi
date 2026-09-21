# RayFusion — görünürlük ve 8-probe tüketicisi

Durum: **YAZILDI, DERLENMEDİ** (2026-09-08). Kullanıcı 1b-β sonrasında
hacim fazından önce bu düzeltmeyi seçti. GPU/görsel kabul bekliyor.

## Kullanıcı yeniden testi (2026-09-08)

Kullanıcı değişiklik sonrası maviliğin azaldığını, ancak dolaylı aydınlatılan
yüzey kıvrımlarında tabaka gibi göründüğünü bildirdi. Merkeze yakın, traced
probe kapsamındaki bölgelerde etki çok az; bu bölgelerde değişiklikten önce
de neredeyse yoktu. Bu geri bildirim görsel iyileşmeyi destekler; tam kabul
ve GPU süre ölçümü değildir. Yukarıdaki derlenmedi etiketi ilk teslim kaydıdır.

Kaynak incelemesi: alan hâlâ sabit 4×2×4, aralık 3; kapsam x/z için [-6,6),
y için [-3,3). Alan dışında veya kullanılabilir komşu olmadığında global sky
fallback sürer. Ayrıca sekiz komşu harmanı dışarıdaki sky ağırlıklı bir probe'u
eskiden yalnız iç probe okuyan yüzeye katabilir. Mesafe momentleri bunu ancak
yaklaşık bastırır. Irradiance ve moment yönleri hâlâ 8×8 haritadan tek texel
seçer; normal/yön değişiminde kıvrım üzerinde basamak üretmesi mümkündür.

Kök neden görüntüyle ayrıştırılmadı. Kapsamı büyütmenin tek başına tabakayı
kaldıracağı doğrulanmış değildir. Sonraki teşhis aynı sahnede kapsam/fallback,
komşu görünürlüğü ve yön örnekleme sınırlarını ayırmalıdır. Izgarayı aynı 32
probe ile daha geniş alana yaymak örnekleme aralığını büyütüp sorunu artırabilir.

### Açık sahnenin IPC/görüntü incelemesi

2026-09-08: material shading, traced ve bounce aktif; `valid=32`, `total=32`,
`pending=0`, `in_flight=0`, `hit_fraction=0.10546875`, `bounce_hits=216`,
`bounce_shaded_hits=122`. Alanın güncellemeyi bitirmesini beklemek kapsamı
genişletmez. 1680×945 yakalamada sol koltuk/puf kıvrımlarında ve sandalye
kenarlarında mavi parlaklık görüldü. Kanıt: `tmp/rayfusion_sampling_live.jpg`.
Kamera/sahne/üretici değiştirilmedi; geçici capture eski kapalı durumuna döndü.

`material_preview_frag.frag` içinde probe yalnız diffuse irradiance'ı değiştirir.
`envSpecular` ve clearcoat environment, `worldPrefiltered` üzerinden probe
görünürlüğünden bağımsız kalır. Bu nedenle kalan kıvrım parlaklığı için
speküler sky katkısı da adaydır; görüntü tek başına diffuse/specular ayrımını
kanıtlamaz. Probe kapsamını genişletmek bu ayrı yansıma yolunu düzeltmez.
Capture açıkken alınan süreler interaktif performans ölçümü olarak kullanılmadı.

`probe_field.glsl` artık üreticinin `(cell + 0.5) * spacing` merkezlerine göre
sekiz komşuyu okur. Trilineer ve yumuşak normal ağırlıkları birlikte kullanılır.
Irradiance yüzey normali yönünden; mesafe momentleri **probe'dan yüzeye**
doğru okunur. Moment görünürlüğü CPU `momentVisibility` sözleşmesindeki
kübik Chebyshev tahminidir. Pencere dışı hücreler toroidal slot hesabından önce
elenir; yayınlanmamış/geometri içinde reddedilmiş probe'lar harmana katılmaz.

Normalizasyon yalnız uzamsal/normal ağırlıklarına uygulanır. Görünürlük
paydada yoktur: tek kalan, engellenmiş probe'un küçük katkısını tekrar tam
parlaklığa yükseltmez. Bu konservatif tercih, kısmen engellenmiş komşulukta
standart görünürlükle normalize edilen interpolasyondan daha karanlık olabilir.
Geçerli ölçüm olup tüm katkılar engellendiyse siyah döner; global sky eklenmez.
Hiç kullanılabilir komşu yoksa veya piksel alan dışındaysa önceki sky fallback
sürer. Bu kapsama açığı ve seyrek ızgara bu değişiklikle çözülmüş sayılmaz.

Üretici, GPU ABI, ızgara ayarları ve ışın bütçesi değişmedi. Yeni kullanıcı
işlemi yoktur; mevcut UI, `rt.rayfusion.*` ve `rayfusion.*` IPC işlemleri aynı
üreticileri ve ortak raster tüketicisini kullanır. Eski tek-probe tüketicisiyle
yapılmış 1b-α/β görüntü karşılaştırmaları bu sürümde yeniden alınmalıdır.

Kaynak kontrolü: `python scripts/audit_rayfusion_probe_sampling.py`.
Bu sayısal referans ve kaynak bağlantısı denetimidir, GLSL çalıştırmaz.

## Kullanıcının derleme ve kabul sırası

1. `material_preview_frag.frag` shader'ını yeniden derleyip kullanılan
   `material_preview_frag.spv` dosyasını güncelle. Bu dilimde C++ değişmedi.
2. Sabit pozlama/kamera ile sky-bake, traced alpha ve traced beta görüntülerini
   yeniden al. Her üretici değişiminden sonra viewport karesi ve artan
   `traced_publishes` ile güncel alanı doğrula.
3. Kapalı oda ve dışarıda parlak sky: duvar arkasındaki probe'un mavi katkısı
   azalmalı; tamamen engellenmiş geçerli komşuluk sky'a geri düşmemeli.
4. Kırmızı duvar/nötr zemin ve pencereli oda: açıklıktan gelen sınırlı renk
   katkısı korunmalı. Pencere kapanınca ışık azalmalı; aşırı kararma kaydedilmeli.
5. Eski üç birimlik hücre sınırlarında yüzey sürekliliğini, negatif dünya
   koordinatlarını ve alan kenarını kontrol et. Alan dışı/hiç ölçülmemiş bölge
   hâlâ sky fallback kullanır; bu bölgeyi görünürlük başarısı diye sayma.
6. Aynı çözünürlük ve sahnede önce/sonra viewport zamanını ölç. Tüketici artık
   en fazla 8 irradiance + 8 moment texel'i okur; GPU maliyeti ölçülmedi.

Mesafe momentleri yaklaşık görünürlüktür; 64 ışın ve 8×8 yön çözünürlüğü ince
duvar/small opening kaçağını tamamen kaldırmayı garanti etmez. Izgara yoğunluğu,
yerleşimi, relocation ve yönler arasında filtreleme sonraki kalite işleridir.
