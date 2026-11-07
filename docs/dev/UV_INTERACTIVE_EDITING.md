# UV seçim ve native texture gösterimi

2026-11-04. Kullanıcı build'inde corner/vertex IPC ve Python, sphere/torus atlas
regresyonları geçti. Mouse island rotation ve tek undo ile özgün revision'a dönüş
canlı doğrulandı. Aşağıdaki Edit UVs ve texture polling düzeltmeleri için yeni build
bekleniyor. Bağımsız native test hedefi ajan tarafından derlenmedi.
Önceki canlı sonuçlar [kabul kaydında](UV_EDITOR_LIVE_ACCEPTANCE_2026_11_04.md)
korunur; aşağıdaki ekler o binary'de doğrulanmış değildir.

## Panel

Selection menüsünde Island / Vertex / Edge seçilir. Mod değişince önceki seçim
temizlenir. Click en yakın vertex'i veya edge'i 8 ekran pikseli içinde seçer;
ayrı indeksler aynı UV'de olsa bile birleştirilmez. Kutu seçimi vertex konumlarını,
edge için iki ucun da kutu içinde olmasını kullanır. Island kutu seçimi en az bir
corner'ın içeri girmesine dayanır. Shift ekler, Ctrl çıkarır; A tüm görünür hedef
mesh'in ilgili elemanlarını seçer, Alt+A temizler.

G/R/S araç seçer. Seçili elemanı sol mouse ile sürüklemek canlı preview üretir;
bırakmak tek history işlemi, Escape iptaldir. Döndürme ve ölçekleme seçim UV
bounds merkezini kullanır. Sayısal transform alanları kendi Pivot değerini kullanır.
Vertex/Edge modunda boş seçim yazma işlemlerini etkinleştirmez. Island modundaki
boş seçim mevcut sözleşme gereği bütün island'ları kapsar.

Manipülasyon başlangıçta gizli kalan `Allow texture changes` tercihine bağlıydı.
Canlı kullanıcı sorunu bu tercih açılınca çözüldü. Kaynak düzeltmesinde Move aracı
ve Edit UVs varsayılan açıktır. Edit UVs görünür üst toolbar'dadır; kapatılırsa
panel açıkça kilit mesajı gösterir. Yeni bir eleman da aynı mouse basışıyla seçilip
sürüklenebilir; önce seçip ikinci kez basmak gerekmez. Ortak API'nin zorunlu
allow_texture_change alanı korunur; UI normal authoring işlemlerinde bunu gönderir.

Vertex hareketi onu kullanan tüm triangle corner'larına yansır; 3D konum, normal,
indeksler ve diğer attribute'lar değişmez. Edge hareketi iki endpoint vertex'inin
birleşimini taşır. Bu işlem seam mark/cut/stitch değildir. Ayrı corner UV vermek
gerektiğinde mevcut Triangle corners / set_corners remap yolu kullanılır.

## Script ve IPC

`uv.transform`, `uv.normalize`, `uv.flip_u`, `uv.flip_v`, `uv.swap_uv` ve karşılık
gelen `rt.uv.*` yöntemleri opsiyonel `vertices` alanını kabul eder. UV vertex ID'leri
`uv.layout` face kayıtlarının `vertex_indices` alanından alınır. Edge düzenlemek
için endpoint ID'leri verilir; UI edge sıra numarası bir kalıcı scene kimliği değildir.

```python
layout = rt.uv.layout(object="Plane", mesh=0, limit=10)
edge = layout["faces"][0]["vertex_indices"][:2]
rt.uv.transform(
    object="Plane", mesh=0, revision=layout["revision"],
    vertices=edge, offset=[0.25, -0.5], allow_texture_change=True,
)
```

`vertices` verilirse boş olamaz; tamsayı, sıfır veya pozitif ve mevcut vertex
aralığında olmalıdır. Nonempty `islands` ile birlikte verilemez. Boş vertices'in
yanlışlıkla bütün mesh'i dönüştürmesine izin verilmez. Geçersiz seçim hiçbir UV
yazmadan reddedilir. Mevcut revision/render/shared-geometry/texture acknowledgment
kuralları korunur. `vertex_selection` ve `edge_endpoint_transform` capability'leri
eklendi; IPC/Python keşfi aynı 14 yöntemi, edit şeması yeni opsiyonel alanı sunar.

## Texture

Background, materyal kanallarını aynı flat inventory'den seçer. Eski 128×128 CPU
örnek çizimi kaldırıldı. Yeni odaklı `UI/UvTexturePreview.cpp`, görüntüyü kendi
çözünürlüğünde SDL texture'a yükleyip UV tile üzerine tek AddImage ile çizer.
Pan/zoom texture'ı yeniden yüklemez. Seçim, boyut, HDR formatı veya renderer
değişince cache yenilenir; panel kapatılınca kaynak bırakılır. Renderer boyut
limitini aşan görüntü açık hata verir; sessizce küçültülmez.

Texture slot envanteri, global geometri generation'ına bağlı olmadan 250 ms'de bir
yeniden okunur. Sonradan atama/değiştirme/silme için objeyi yeniden seçmek gerekmez.
Bu güncelleme UV layout'unu/selection'ı yenilemez ve mouse gesture'ını iptal etmez.
Hedefte ilk texture otomatik arka plan olur; kullanıcı Checker only veya başka kanal
seçerse tercihi korunur. Yeni hedefte otomatik seçim yeniden etkinleşir.

Refresh texture veya üstteki Refresh, aynı texture'ın değiştirilmiş CPU pixel'lerini
yeniden yükler. Her frame tam görüntü taranmaz; paint/undo sonrası gerekirse bu
düğme kullanılır. LDR RGB byte'ları korunur, HDR görünüm tone map edilir. Kaynak
alpha yerine editör opacity'si uygulanır. Ham UV 0 / 0–1 tile gösterilir; materyal
tiling/node dönüşümleri, ayrı packed kanal gösterimi ve UDIM henüz eklenmedi.

## Kabul

- Non-build kontrat kontrolleri: 9/9. Build/native test çalıştırılmadı.
- `tests/uv` native hedefinde vertex/edge endpoint izolasyonu, rotation/scale pivot,
  invalid/mixed/empty scope atomik ret vakaları eklendi; kullanıcı çalıştırmalıdır.
- IPC ve Python corner probe'larına endpoint transform, seçilmeyen UV/3D yüzey
  koruması ve undo/redo eklendi. Probe başında capability aranır; eski binary'de
  yanlış scope ile yazmaya başlamaz.
- Kullanıcı build'inde ayrı terminalden corner IPC/Python, vertex scope,
  sphere/torus remap/undo regresyonları geçti. UI'de Default_Cube üzerinde allow
  tercihi açıldı; mouse island rotation commit etti ve undo özgün revision'a döndü.
  Kullanıcının UV geometri verisi test sonrası korundu. Edit kilidi açık bırakıldı.
- Kaynak düzeltmelerinden sonraki kullanıcı build'inde ayrı terminalde
  `python scripts/test/rt_test_uv_corner_ipc.py` ve genel regression için
  `python scripts/test/rt_test_uv_authoring_ipc.py` çalıştırılır.
- UI kabulü: vertex/edge click/kutu/Shift/Ctrl; tek ve çoklu seçim drag; Escape;
  tek undo; normal/albedo üzerinde asimetrik texture yönü ve pixel ayrıntısı;
  pan/zoom, texture refresh, panel kapat/aç; UV paneli açıkken viewport G/R/S
  ve WASD'nin tetiklenmemesi. Büyük mesh drag maliyeti ayrıca ölçülmelidir.

## Güncel erişim ve viewport kararı

2026-11-05: UV-only yazmalar için [attribute update yolu](UV_ATTRIBUTE_UPDATES.md)
eklendi. Genel geometry scheduler yerine hedefli shading buffer upload kullanılır;
topoloji split ve unsupported resident layout'ların rebuild ayrımı bu kayıttadır.
Bu ek için yeni build ve canlı GPU kabulü bekleniyor.

UV Editor alt bardaki kalıcı sekmeden kaldırıldı. View > UV Editor ve materyal
properties içindeki Open UV Editor aynı paneli açar; script/IPC uv.open korunur.
View menüsü seçili flat mesh'i başlangıç hedefi olarak kullanır. Panelin mevcut
dock/ayrı pencere davranışı korunur.

UV düzenleme kilidi kaldırıldı. Normal mouse/sayısal işlemler doğrudan çalışır;
script/IPC tarafındaki allow_texture_change doğrulaması korunur.

Sonraki seçim gösterimi edit mesh modundaki yeni GPU overlay'i yeniden kullanacak.
Viewport'ta UV seçimini gösterme kullanıcıya bağlı, varsayılan kapalı bir görünüm
seçeneği olacak. İlk teslim yalnız UV seçiminin objede vurgulanmasıdır; otomatik
edit-mode geçişi veya zorunlu viewport → UV seçim senkronizasyonu yoktur. Bu overlay
henüz teslim edilmedi; çalışan backend olmadan etkisiz bir checkbox eklenmedi.

Sonraki aşama isteğe bağlı GPU seçim vurgusu, sahnede seam mark/clear ve seam'e uyan unwrap'tir.
