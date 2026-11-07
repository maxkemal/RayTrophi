# UV canlı kabul ve mouse etkileşimi — 2026-11-04

## Açık kullanıcı build'inde doğrulananlar

- Harici corner IPC probe ve `script.run_file` üzerinden Python API: split,
  stale revision reddi, undo/redo geçti.
- Genel UV authoring IPC probe: pack/unwrap, izolasyon, normalize, sayfalama,
  stale write, undo/redo ve silinen hedef filtresi geçti.
- Vertex paylaşan torus: 561 → 711 vertex, 150 split; 1024 triangle korundu.
  Topoloji undo/redo tam eski/yeni revision'a döndü.
- Geçici test nesneleri silindi. Proje history/modified durumu testlerden etkilenir.

Sphere unwrap mevcut binary'de kutuplardaki dejenere yüzler için atomik hata verdi.
Kaynak düzeltmesi bu yüzlerin önceki corner UV'lerini korur; geçerli yüzlerde chart
eksikliği yine hatadır. Sonuç `atlas_preserved_faces` içerir ve panel bunu bildirir.
Bu düzeltme için yeni build ve sphere probe tekrarı bekleniyor.

## Yeni kaynak değişiklikleri; henüz canlı UI kabulü yapılmadı

UV Editor'de Select / box, Move, Rotate, Scale araçları bulunur. Kutuya en az bir
UV köşesi giren island seçilir; Shift ekler, Ctrl çıkarır. Island seçildikten sonra
Move/Rotate/Scale aracında sol mouse sürüklenir. Önizleme detached layout üzerinde
aynı `UV::transform` fonksiyonunu kullanır; bırakma aynı `UV::request("transform")`
servisine tek transaction gönderir. Escape preview'ı iptal eder. Yazma için mevcut
Allow texture changes tercihi gerekir. Henüz vertex/edge/lasso seçimi yoktur.

Panel açıkken kısayol sahibi UV editörüdür: G/R/S araç seçer, A tüm island'ları
seçer, Alt+A temizler, F çerçeveler. Yazı/numerik alanına yazarken bu kısayollar
çalışmaz. Viewport object/rig G/R/S, scene seçim/silme/mod değiştirme, numpad kamera
ve sürekli WASD kontrolleri panel açıkken bu tuşları alamaz. Panel kapanınca mevcut
viewport davranışı geri gelir. Ctrl+Z/Y ortak history'yi kullanır; Audio history
panel açıkken tuşu devralmaz. Save/render gibi uygulama komutları korunur.

Background menüsü mesh'in flat materialID kanalından kullanılan materyalleri bulur.
Albedo, roughness, metallic, specular, normal, opacity, transmission, emission ve
height texture slotları varsa seçilebilir. Saydamlık değiştirilebilir; seçili
island dolgusu alttaki texture'ı gizlemez. Envanter ortak `previewTextures` helper'ı
üzerinden UI ve `uv.info` / `rt.uv.info` içindeki `textures` listesine sunulur.
Listede label/material/name/width/height bulunur. Texture değiştirilirse Refresh ile
envanter yenilenebilir; mevcut texture pixel boyamaları yeniden yükleme gerektirmez.

Şimdiki gösterim görünür alanı en fazla 128×128 örnekle çizen CPU önizlemesidir;
tam çözünürlüklü GPU gösterim değildir. Ham UV 0 / 0–1 tile gösterir; materyalin
tiling/offset/node graph dönüşümleri, packed kanal ayrıştırma ve UDIM henüz yoktur.
Tam çözünürlüklü gösterim ve vertex/edge seçimi bir sonraki UI kabul kapılarıdır.

## Kullanıcının yeni build sonrası kontrol listesi

1. UV Editor açıkken sahnede bir nesne seçili olsun: G/R/S yalnız UV aracını
   değiştirmeli; A/Alt+A yalnız UV seçimini değiştirmeli; Delete nesneyi silmemeli;
   S kamerayı yürütmemeli. Paneli kapatınca viewport kısayolları geri gelmeli.
2. Kutuyla birden çok island seç, Shift ekle/Ctrl çıkar. G/R/S sonrası sürükle;
   seçilmeyen island değişmemeli. Bırak → tek undo; Escape → UV/history değişmemeli.
3. Materyalde asimetrik albedo/normal/roughness texture ile Background seç;
   yukarı/aşağı yönü, pan/zoom, saydamlık ve seçili ada üstünde görünürlüğü kontrol et.
4. Ayrı terminalde `python scripts/test/rt_test_uv_corner_ipc.py` ve
   `python scripts/test/rt_test_uv_authoring_ipc.py` çalıştır. Güncellenen corner
   probe torus ve sphere split/preserved-face/undo vakalarını da kapsar.
5. İstenirse `tests/uv` bağımsız native hedefini kullanıcı derleyip çalıştırır;
   geometric/UV-degenerate face ve welded cube atlas regresyonları eklendi.

Ajan build çalıştırmadı ve uygulama başlatmadı. Yeni mouse/texture/kısayol değişiklikleri
açık eski binary'de doğrulanmış sayılmaz. Non-build kontrat kontrolü 8/8 geçti.
