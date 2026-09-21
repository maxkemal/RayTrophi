# Realtime — katkısız shader işi ve seçim upload beklemesi

> **Durum:** AKTİF — üç dar optimizasyon yazıldı; derleme, görüntü ve FPS doğrulaması bekliyor (2026-09-07).

Genel performans ölçümleri ve ana Physical Sky işi
[REALTIME_CAMERA_MOTION_PERF.md](REALTIME_CAMERA_MOTION_PERF.md) içindedir.
Oraya sonradan eklenen 102 → 38.6 ms sky açık/kapalı karşılaştırması bu
partinin ölçümü değildir. Buradaki değişikliklerin o farkı kapattığı iddia
edilmiyor. Proje açılışı/Realtime TDR ve template HUD başka ajanın kapsamıdır.

## Yazılan değişiklikler

- `material_preview_frag.frag`: transmission replay, kesin opak fragment'i
  UV/terrain/texture/graph işinden önce eler. Transmission veya opacity map,
  graph programı, bubble veya kısmi opacity varsa geç eleme korunur. Opacity'nin
  metal olmayan yüzeyde transmission'a dönüşmesi özellikle kapsandı.
  Impostor zaten opak sınıflandığı için replay'de doğrudan elenir. Ana opaque
  geçişindeki cam back-depth atomikleri korunur. Vertex/draw tekrarı sürer.
- Aynı shader: diffuseColor tam sıfırken (tam metal/tam transmission dahil)
  diffuse sky filtresi hesaplanmaz. Translucency sıfırken arka environment
  örneği alınmaz. Kısmi materyallerin filtre örnek sayısı düşürülmedi.
- `SelectionOutlineUpload.cpp`: seçim matrislerinin gerçek byte içeriği
  cache'lenir; aynı veriyle kamera hareketinde host write/drain yapılmaz.
  Değişen veri ve tampon büyümesi önce in-flight drain yapar. Cache yalnız
  başarılı map/copy sonrası geçerli olur. Draw listesi, mesh kimliği ve seçim
  önceliği her çağrıda mevcut instance verisinden çözülür; nodeName kimlik
  cache'i yapılmadı. Realtime composite ve Rendered mask readback aynı
  resolver/uploader'ı kullanır. Konturun iki geometri çizimi hâlâ vardır.

Yeni kullanıcı ayarı veya API yok; mevcut Realtime ve seçim işlemleri
optimize edildi. Büyük backend dosyası yalnız ortak uploader çağrısı ve eski
upload bloğunun kaldırılmasıyla değişti; yeni mantık ayrı modüldedir.

## Kaynak doğrulaması

31.104 replay sınıflandırma kombinasyonunda erken elenen fragment'in eski yol
tarafından da elendiği kontrol edildi. Erken kapının texture işinden önce
oluşu, tek graphOffset/drawPhase tanımı, iki seçim çağıranının ortak uploader'a
ulaşması, cache/drain/map sırası ve vcxproj XML kaydı kontrol edildi.
Bunlar shader derlemesi veya GPU doğrulaması değildir.

## Açık ana iş

Physical Sky'ın diffuse ve reflection konileri hâlâ katkı veren loblarda
merkez + 4/8 çevre örneği alır. Bunlar iki lob etkinse 10/18 canonical world
değerlendirmesidir; çok pürüzsüz reflection tek örnek yolundadır. Genel sky
maliyeti için overlay'i de içeren prefilter gerekir. LUT içeriği aynı image'a
yeniden yazılabilir; yalnız image handle cache anahtarı olamaz. Kamera
hareketinde pahalı yeniden üretim yapılmaması ve güneş transmittance'ının
kamera yüksekliğine bağımlılığı ayrıca ele alınmalı. HDRI için mevcut GGX
prefilteri kullanmak eski koninin görünümüyle kendiliğinden aynı sonucu vermez;
görüntü ve enerji karşılaştırması kabul kapısıdır.

## Kullanıcının derleme ve kabul sırası

1. C++ derlemesi ve `material_preview_frag.frag` shader derlemesi gerekir.
   Yeni `SelectionOutlineUpload.cpp` vcxproj'a kayıtlıdır. Uygulamayı ajan
   derlemedi veya başlatmadı. NEXT_BUILD_CHECKS'in önceki kamera partisindeki
   "shader gerekmez" cümlesi bu ek parti için geçerli değildir.
2. Opak, texture'lı bir sahnede aynı çözünürlük/kalite/kamera hareketini
   karşılaştır. Görüntü değişmemeli. Kazanç ölçülmeden FPS yüzdesi yazma;
   vertex/draw tekrarı hâlâ bulunduğundan üçgen maliyeti bitmiş sayılmaz.
3. Scalar transmission=0 iken transmission map ile cam; yalnız kısmi opacity
   ile cam; graph transmission/opacity; bubble; terrain ve alfa kesimli yaprak
   kontrolü. Hepsi eskisi gibi görünmeli. **En sinsi hata:** sayılar ve opak
   sahne doğruyken map/opacity camının sessizce opaklaşması veya kaybolması.
4. Tam metal, tam cam, kısmi metal/transmission ve translucent malzemeleri
   Physical Sky altında dene. Kısmi lobların parlaklığı ve arka aydınlatma
   değişmemeli; yalnız sıfır katkılı hesap atlanır.
5. Statik nesne seçiliyken orbit yap. Seçimden kaynaklanan `resource_drains`
   her kare artmamalı. Diğer sistemlerin drain'leri ayrıca var olabilir.
   Sonra nesneyi taşı, seçimi/referans sırasını değiştir, çoklu seçimi büyüt,
   nesneyi sil/yeniden ekle ve viewport'u yeniden boyutlandır. Kontur doğru
   nesneye ve yeni dönüşüme oturmalı; eski matris görürsen cache geçersizdir.
6. Rendered modundaki mask/selection yolunu da kontrol et; Realtime onayı
   ikinci çağıranı kapsamaz. Açılış/TDR işinin kabul testleri ayrı tutulur.
