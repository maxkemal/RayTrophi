# RayFusion — derinlikli probe overlay ve kamera takibi

Durum: **YAZILDI, DERLENMEDİ** (2026-09-08). Kullanıcı talebi: işaretler sahne
derinliğini kullansın, ImGui panellerinin üstüne taşmasın; mevcut hacim yoluna
ölçülmemiş ek maliyet getirilmesin.

## Kullanım ve ortak kontrol

RayFusion development panelinde iki varsayılan kapalı seçenek vardır:

| UI | Python | IPC |
|---|---|---|
| Show probes (depth tested) | `rt.rayfusion.set_probe_overlay(True)` | `rayfusion.set_probe_overlay {"enabled":true}` |
| Follow camera position | `rt.rayfusion.set_probe_follow_camera(True)` | `rayfusion.set_probe_follow_camera {"enabled":true}` |

Üç yüzey aynı `rtapi` servisine ve viewport probe sahibine gider. Her iki
yazma IPC'de Render capability ister. `enabled` gerçek boolean olmalı;
eksik/ek parametre ve sayı/string reddedilir. Python dönüşümü kapalıdır.
Hazır probe sahibi yoksa `applied=false`. İstek kabulü GPU başarı iddiası
değildir. Ayarlar proje kaydına eklenmedi; probe sahibi yeniden yaratıldığında
sıfırlanır. Overlay ve kamera takibi birbirinden bağımsızdır.

`rayfusion.probe_field` / Python `probe_field()`:
`overlay_requested`, `overlay_ready`, `overlay_markers`, `overlay_reason`,
`follow_camera`. Ready, son kayıt sırasında pipeline'ın hazır olmasıdır;
markers kaydedilen marker sayısıdır, **derinlikten geçip görünen piksel/probe
sayısı değildir**. Rendered moduna geçişte önceki raster kayıt durumu kalabilir;
bu alan aktif viewport modu yerine kullanılamaz. Shader eksikliği açık reason
üretir; shader derlemesi ardından uygulamayı yeniden başlat.

## Derinlik ve panel sırası

ImGui yalnız kontrol ve açıklamayı çizer. İşaretler gerçek probe merkezinde
küçük üç boyutlu octahedron'lardır. Vulkan raster'ın post sonrası LDR geçişinde,
aynı sahne depth attachment'ına `LESS_OR_EQUAL` testiyle çizilir; depth yazmaz.
Ekrana/projeksiyona göre kesilir ve viewport görüntüsünün parçası olarak ImGui
panellerinin altında kalır. Foreground draw-list veya CPU depth readback yoktur.

Yeşil: yayınlanmış kullanılabilir probe. Amber: henüz yayınlanmamış. Kırmızı:
geometri içinde doğduğu için kullanılmaz olarak yayınlanmış. Derinlik testi
kırmızı işareti de gizleyebilir; duvar arkasını gösteren x-ray modu yoktur.
Opak/sahne depth otoritesi kullanılır; transparan ve depth yazmayan hacim
görünürlüğü bu overlay'in ayrı bir ray-tracing çözümü değildir.

İşaretler yalnız ayrı Vulkan raster viewport yolunda çizilir. Rendered yolunda
başka GPU/device depth'i ödünç alınmaz ve sahne tekrar render edilmez.
Kapalıyken marker listesi, pipeline kurulumu ve draw işi yoktur. Açıkken ilk
kullanımda pipeline oluşturulur; her probe için 24 vertex, toplam 32 draw vardır.
Vertex buffer, descriptor güncellemesi veya ilave GPU readback yoktur. GPU
maliyeti ölçülmedi. Pipeline mevcut viewport kaynak temizliğinde bırakılır.

## Otomatik yerleşimin ilk dilimi

Follow camera, mevcut **32 probe / 3 birim aralık** bütçesini büyütmez.
Kamera konumu hücrelere kuantize edilir; bir hücrelik tolerans küçük sınır
salınımını bastırır, kamera alanın dışına çıkarsa pencere kayar. Kamera dönüşü
tek başına kaydırmaz. `ProbeField::scroll` dünyada ortak kalan hücrelerin
değerlerini korur, yalnız yeni hücreleri sıfırlar. Producer origin'leri, upload
hücreleri, GPU grid minimum'u ve overlay aynı güncel grid'i kullanır.

Mevcut kalite/ray bütçesi yeni hücreleri partiler halinde doldurur. Başarılı
partiden sonra iş kaldıysa frame pump ve cached-frame kapısı yakınsamayı
tamamlatır; başarısız üretim sonsuz retry döngüsüne alınmaz. Takibi kapatmak
alanı o anki konumunda dondurur; dünya orijinine atlatmaz.

Bu dilim **sahne sınırlarına göre yoğunluk seçimi veya relocation değildir**.
Duvar içindeki probe taşınmaz; mevcut rejection kullanılır. Büyük sahneyi
birden kaplamak veya dar odada örneklemeyi sıklaştırmak teslim edilmedi.
Overlay, sonraki yerleşim kararlarını gözlemlemek için eklendi.
Hacim/froxel/otomatik yol seçici çalışması ölçülmüş ihtiyaç çıkana kadar ertelendi.

## Kaynak kontrolleri ve kullanıcı build/test sırası

`python scripts/audit_rayfusion_probe_overlay.py` ve
`python scripts/audit_ipc_capabilities.py`: kaynak bağlantıları, depth ayarları,
yaşam döngüsü, kontrol yüzeyleri ve sayısal toroidal scroll sözleşmesi.
Bu kontroller C++/GLSL derlemesi veya GPU doğrulaması değildir.

1. C++ projesini derle. Yeni `RayFusionProbeOverlay.cpp` vcxproj'a eklendi.
2. `rayfusion_probe_overlay.vert` → `rayfusion_probe_overlay.spv` ve
   `rayfusion_probe_overlay_frag.frag` → `rayfusion_probe_overlay_frag.spv`
   üret. Farklı stem'ler derleme script'inde dosyaların birbirini ezmesini önler.
3. Material viewport'ta overlay aç: duvar önündeki marker görünsün, arkasındaki
   gizlensin. Kamerayı döndür ve ortografik görünümü dene. Panel/popup/dock'u
   marker'ın üzerine getir: panel üstte kalmalı. Resize ve viewport kalite
   değişiminde validation/device-loss hatası olmamalı.
4. UI, Python ve IPC'den aç/kapat. Eksik enabled, `enabled:1`, `enabled:"true"`
   ve ek alan IPC'de reddedilmeli; Python sayı/string'i reddetmeli. İstekten
   sonra viewport karesi oluşmalı, panel ve probe_field aynı sonucu vermeli.
5. Kamera takibini aç. Kamerayı yalnız döndür: minimum sabit kalmalı. Yavaşça
   taşı: minimum tam hücrelerle değişmeli, toplam=32 ve spacing=3 kalmalı.
   Kamerayı durdur: pending sıfıra inmeli. Eski hücre ışığı yeni konuma
   yapışmamalı; alan dışında yeni probe amber → yeşil/kırmızı ilerlemeli.
6. Takibi kapat ve kamerayı taşı: minimum sabit kalmalı. Büyük kamera sıçraması,
   negatif koordinatlar, geometri değişimi ve proje yeniden açmayı dene.
7. Overlay kapalı/açık aynı kamera ile capture kapalı frame süresini karşılaştır.
   Hacim yolunun görüntü ve performans davranışı bu partide değişmemeli.
