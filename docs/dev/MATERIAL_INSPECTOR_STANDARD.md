# Materyal inspector standardı ve animasyon envanteri

Tarih: 2026-11-03  
Durum: UI kaynak dönüşümü uygulandı — build ve canlı doğrulama bekleniyor.

## Uygulanan yapı

Eski Surface Core–Procedural Detail arasındaki uzun özellik dizisi yedi odaklı
modüle taşındı: Surface, Transparency, Interior, Subsurface, Coat, Emission, Detail.
Dar dock'ta taşan yatay sekmeler yerine tek bir bölüm seçicisi kullanılır;
checkbox ile işaretlenen birden fazla bölüm birlikte çizilir. Menü checkbox
tıklamasında kapanmaz; işaret kaldırılana kadar bölüm açık kalır. Bölümler aynı
sırada, kendi başlıkları ve ayrı widget ID kapsamlarıyla görünür. Boş seçimde
diğer panel içerikleri erişilebilir kalır. İlk açılış Surface; önceki tekli
seçim state'i varsa aynı bölümden başlanır. Seçim mevcut ImGui window state'i içinde caller ve
material ID kapsamındadır; yeni proje/solver state'i değildir.

`MaterialInspectorSchema.h` içinde 46 benzersiz sunum tanımı vardır: alan kimliği,
görünen ad, grup, kontrol türü ve mevcut inspector keyframe bağlantısı.
Bu tanımlar materyal depolama, validasyon veya fizik aralığı için ikinci bir
otorite değildir. Yeni authoring işlemi veya public API eklenmedi.
Mevcut servis işlemleri, setter'lar, değer aralıkları ve preset değerleri korunur.

`MaterialInspectorControls.h` bütün çıkarılan slider, integer, enum, checkbox,
vektör ve renk alanlarını ortak satırlarla çizer. Etiketler solda; dar alanda
üsttedir. Sağdaki animasyon yuvası, animasyonsuz alanlarda da ayrılır.
Renk kontrolleri `UI/ColorControls.h` üzerinden Fog View'un eski standart
ImGui `ColorEdit3` düzenini kullanır: RGB girişleri, küçük kare önizleme ve etiket
aynı satırdadır. Özel genişlik eşiği, zorunlu alt satır, soldaki ayrı renk etiketi
ve ColorButtonPosition override'ı yoktur. Önizleme panelle büyümez; picker ve
RGB/RGBA sürükle-bırak ImGui'nin kendi davranışıdır. Materyal Properties aynı
kontrolü kullanır ve animasyon düğmesi için ek yer ayırır. Mevcut setter'lar ve
değer uygulama callback'leri korunur.

Önceki uyarlamalı düzen kullanıcı ekran görüntüsünde aynı panelde tutarsız
satırlara yol açtı; bu nedenle kaldırıldı. Build sonrası manuel kabul: geniş/dar
dock'ta tek satır, picker, RGB girişi ve Properties <-> Fog View renk sürükle-bırak.
Yalnız C++ derlemesi gerekir; canlı UI doğrulaması bekleniyor.
Presets dar dock'ta satıra yayılır. Interior preset seçicisi farklı materyale
geçilince önceki materyalin preset adını doğruymuş gibi göstermemelidir;
uygulama eylemi olarak “Choose a preset...” gösterir.

UV Workflow, texture slotları ve volumetric editor bu dilimde yeniden tasarlanmadı.
Onlar erişilebilir kalır; bu teslim tüm materyal UI'sının tamamlandığı anlamına gelmez.

### Texture slotları — sonraki dilim

Texture Maps artık küçük, en-boy oranını koruyan örneklenmiş önizleme ve kanal
adından oluşan satırları kullanır. Boş slotta `+ / Empty` görünür; tıklama mevcut
dosya diyaloğunu açar. Dolu slot tıklaması veya sağ tık Replace, Copy, Paste ve
Remove from material menüsünü açar. Remove yalnız bağlantıyı kaldırır; kaynak
dosyayı silmez. Değiştirilen texture'lar mevcut graveyard güvenliğine teslim edilir.

Hover gecikmesi sonrası yaklaşık 256px büyük örneklenmiş önizleme, kaynak
çözünürlüğü, dosya yolu ve seçili kanal gösterilir. Renk haritaları RGB,
Roughness/Metallic mevcut Auto/R/G/B seçimine göre gri tonlu görüntülenir.
Opacity alpha dama zemine bindirilir. HDR önizlemesi tone-map edilmiş olarak
etiketlenir. Paint override ve varsa kaynak bilgisi korunur.

Önizleme CPU verisinden sınırlı örnekleme yapar: küçük slot 24x24, hover 80x80;
orijinal texture çözünürlüğünde bir image viewer değildir. Yeni GPU kaynağı,
dosya tekrar okuması veya kalıcı thumbnail cache'i oluşturmaz. CPU görüntüsü
bulunmuyorsa hover açıkça Preview unavailable gösterir.

Build sonrası: boş yükleme/iptal, dolu Replace/iptal, Copy/Paste (dolu slot dahil),
Remove, farklı en-boy oranları, HDR, alpha, Paint güncellemesi ve packed map
kanal değişimlerini dene. Dar dock ve ekran kenarında tooltip yerleşimini kontrol et.
Build ve canlı doğrulama henüz yapılmadı. Dosya sürükleyip bırakma bu dilimde eklenmedi.

## Animasyon: mevcut durum ve bulunan açık

| Alan | Bu teslimdeki davranış |
|---|---|
| Base Color, Metallic, Roughness, Specular | Mevcut keyframe işlemi korunur. |
| IOR, Transmission, Opacity | Mevcut keyframe işlemi Transparency altında korunur. |
| Emission color | Mevcut emission grup işlemi korunur; capture doğruluğu aşağıdaki audit kapsamındadır. |
| Emission intensity | Değer düzenlenebilir; aktif keyframe düğmesi eklenmedi. |
| Dispersion, Interior, Bubble ve diğer ek alanlar | Değerler düzenlenebilir; bağlanmamış animasyon yuvası pasif ve açıklamalıdır. |
| Subsurface, Coat | KeyframeSystem'de bazı alan/flag'ler mevcut; bu inspector'ın mevcut toggle callback'i bunları kapsamıyor. Uçtan uca destek var diye işaretlenmedi. |

Önemli audit: `MaterialKeyframe(const GpuMaterial&)` emission_strength'i 1 ile
başlatıyor. `drawPrincipledBSDFEditor` içindeki insert callback GPU snapshot'ı
kullanıyor ve gerçek `emissionProperty.intensity` değerini ayrıca aktarmıyor.
Dolayısıyla intensity'ye aynı emission flag'i üzerinden bir düğme eklemek yeterli
değil. Renk/şiddet ayrımı ve GPU'daki birleşmiş değerler birlikte denetlenmeli.

Sonraki animasyon dilimi, UI-local callback'e daha fazla bool eklemek yerine:

1. Gerçek materyalden canonical animation snapshot/capture ve per-field kimlikleri.
2. Yeni alanlar için serialize/load, evaluate, interpolation ve curve editor denetimi.
3. Ortak key ekle/kaldır/sorgula servisi; UI, scripting ve IPC aynı işlemleri çağırmalı.
4. Round-trip ve iki-frame değerlendirme testleri; özellikle intensity'nin 1'e dönmemesi,
   aynı frame'de diğer property key'lerinin silinmemesi ve çoklu materyal hedefleri.

Yeni alanların hepsi keyleniyor diye kabul verilmez; bu dilimde yalnız mevcut
bağlantılar korunmuştur. Sunum metadatasındaki `keyAction`, backend'in tüm teorik
animasyon kabiliyetini değil, bu inspector'da bağlı işlemi ifade eder.

## Kaynak denetimi

- Önceki yedi grubun material setter çağrıları ve doğrudan yazım envanteri karşılaştırıldı.
- 46 alan kimliğinin benzersizliği ve çıkarılan primitive kontrollerin ortak wrapper
  üzerinden geçmesi kontrol edildi.
- Yeni dosyalarda delimiter/string ve 100-column format denetimi uygulandı.
- Build, shader derlemesi veya uygulama başlatma yapılmadı.

## Kullanıcı build ve canlı kabul listesi

1. Normal build'i çalıştır; Material Properties bölüm seçicisini aç.
2. Yedi bölümün her birine geç; eski kontrollerin tamamının erişilebilirliğini doğrula.
3. 240/320/440 logical px ve desteklenen UI ölçeklerinde etiket/kontrol/animasyon
   yuvasının taşmadığını kontrol et. Renk şeridi, RGB kutuları gibi daralmamalı.
4. Her kontrol türünde mouse ve klavye ile değer değiştir; görüntü güncellenmeli.
5. Base Color/Emission picker'larını aç; sayısal giriş, kapatma ve mouse capture.
6. Mevcut sekiz keyframe alanını ekle/kaldır; frame değiştir, save/load yap.
   Mevcut emission capture açığını yeni doğru davranış gibi kabul etme.
7. Interior Depth ve Transmission 0.5 eşiğiyle resin/stone/glass alt gruplarını;
   Dust Style, Shards/Rainbow ve Bubble alt kontrollerini doğrula.
8. SSS yöntemi değişince Walk Limit'in doğru enabled/disabled davranışını kontrol et.
9. Interior, SSS, Coat ve Detail preset'lerini dar panelde dene; değerler korunmalı.
10. Materyal/nesne ve bölüm değiştir; yanlış hedefe yazım veya başka materyalin
    preset adını taşıyan yanıltıcı etiket olmamalı.
11. UV ve Texture Maps yollarını dene; bu dönüşümde işlev kaybı olmamalı.
12. Birleşik domain çalışmasıyla birlikte build et; bu değişiklikler domain kaynaklarını
    değiştirmiyor. Paylaşılan genel PropertyLayout bileşeni önceki UI diliminden geliyor.
13. Bölüm menüsünde Surface, Transparency ve Emission'ı birlikte işaretle. Menü
    açık kalmalı; her üç bölüm de görünmeli. Birinin işaretini kaldırınca diğerleri
    açık kalmalı. Hiçbiri seçili değilken UV/Texture Maps ve dış panel eylemleri
    çalışmaya devam etmeli. Materyal değiştirip dönünce seçim karışmamalı.

Güncel kompakt düzen: Texture Maps font ölçeğine göre yeterli genişlikte iki,
dar dock'ta bir sütun kullanır. Normal Strength Normal slotunun, packed channel
seçicileri kendi Roughness/Metallic slotlarının altında kalır. Dar hücrelerde
uzun dosya adı yerine çözünürlük gösterilir; tam yol hover önizlemesindedir.

Gereksiz `Open Audio workspace` footer düğmesi, iki Scene çağıranı ve artık
kullanılmayan `AudioUI::drawProperties` girişi kaldırıldı. Audio ana sekmesinin
erişimi korunur. Önceki listede dış eylemlerin korunması bu düğmeyi kapsamaz.

Ek kabul: iki sütunda dokuz slotun eksiksizliği, 1/2 sütun geçişi, Normal Strength,
R/G/B/Auto seçimi, hover ve context menüleri, iki Scene yerleşiminde footer'ın
kaybolması ve ana Audio sekmesine erişim. Build ve canlı doğrulama bekleniyor.
