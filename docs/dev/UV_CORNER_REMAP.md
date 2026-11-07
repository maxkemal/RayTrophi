# UV köşeleri ve attribute koruyan vertex remap

Tarih: 2026-11-04. Corner ve torus canlı kabulü geçti; dejenere yüz düzeltmesi ve
yeni panel etkileşimleri için yeni build bekleniyor. Ayrıntılar:
[04 Kasım kabul kaydı](UV_EDITOR_LIVE_ACCEPTANCE_2026_11_04.md).

## Yeni davranış

Paylaşılan bir geometrik vertex'in farklı üçgen köşelerine farklı UV verilebilir.
İlk UV mevcut vertex slotunu kullanır; başka UV isteyen köşeler için sona vertex
kopyaları eklenir. Aynı kaynak vertex + aynı UV tekrarları tek slot paylaşır.
Farklı kaynak vertex'ler, konumları/UV'leri çakışsa bile kendiliğinden kaynak edilmez.
Kullanılmayan mevcut slotlar, face ve corner sırası korunur.

DNA remap her attribute'u kendi C++ tipiyle toplar. P/P_orig, N/N_orig, materialID,
bütün ek UV setleri ve custom kanallar aynı source map'i kullanır. Skin weight
listeleri de çoğaltılır. Desteklenmeyen attribute remap'i, yanlış buffer uzunluğu
veya bozuk indeks veriyi düşürmek yerine açık ret üretir. Aktif delta sonucu ve
in-place aktif yazmalar yeni snapshot'a alınır; önceki geometri/delta stack korunur.

xatlas artık UV vertex bölünmesi gereken geçerli çıktıları bu katmana verir.
Face/corner sırasının değişmesi ve çoklu atlas hâlâ açık hata verir. Manuel seam
işaretleme/silme, pin, conformal solve ve texture rebake sonraki aşamalardır.

## Ortak UI / script / IPC işlemi

UV Editor > **Triangle corners** bölümünde bir üçgene tıklanır veya triangle index
girilir. Üç U/V çifti **Apply triangle UVs** ile uygulanır; diğer köşeler korunur.
Texture değişimine açık tercih gerekir. Başka domain'in güncellenmesi, hedef UV
revision'ı değişmedikçe girilmekte olan köşe alanlarını sıfırlamaz.

Python `rt.uv.set_corners`, IPC `uv.set_corners` aynı UV::request servisini çağırır:

```python
layout = rt.uv.layout(object="Plane", mesh=0, limit=10)
corners = [list(uv) for face in layout["faces"] for uv in face["uv"]]
corners[3] = [0.25, 0.5]
result = rt.uv.set_corners(
    object="Plane", mesh=0, revision=layout["revision"],
    corners=corners, allow_texture_change=True,
)
```

Örnek küçük mesh'in bütün layout'unu varsayar. Büyük mesh'te `next_face` null olana
kadar bütün sayfalar okunmalıdır. Sayfa okumasında revision değişirse yeniden başlanır.
Payload face-major sırada üçgen başına üç corner içerir; en fazla 60000 corner.
Eksik/yanlış/non-finite payload ve stale revision hiçbir veri yazmadan reddedilir.

Layout kayıtları artık `vertex_indices` döndürür. Mutation sonucunda güncel vertex/
island sayısı, `vertex_splits`, `topology_changed` ve revision bulunur. `corner_remap`
ve `set_corners` true, seam/texture preservation false; keşifte 14 yöntem vardır.
Mesh NumPy görünümleri topoloji değişiminden sonra yeniden alınmalıdır; testler eski
görünümleri `.copy()` ile saklar.

## Transaction ve kaynak sınırı

Tüm attribute'ları hazır detached geometri tek heavy SceneHistory komutuyla yayınlanır.
Undo/redo eski/yeni GeometryDetail pointer'ını geri koyar; önceki delta durumu korunur.
UV-only değişiklikler mevcut hafif yolda kalır. No-op yeni undo adımı oluşturmaz.
Local BVH, pointiness, material attribute, surface ve UI cache'leri invalid edilir;
CPU/backend rebuild ortak scheduler'dan istenir. Shared direct geometry ret kuralı
korunur; transient simülasyon çıktıları authoring için yazılamaz.

`GeometryDetail.h` içine tipli remap virtual/override sözleşmesi ve bir friend
bildirimi eklendi. Veri sözleşmesi bu header'ı kullanan dosyaların yeniden derlenmesini
gerektirir. Uygulama yeni `MeshEdit/GeometryRemap.cpp`, `UV/UvCornerRemap.cpp` ve
`UV/UvGeometryTransaction.cpp` içindedir. SceneUI header'ı ve domain implementasyonu
değiştirilmedi. Vertex-ID tabanlı UI seçimleri split sonrası yeniden seçilmelidir.

## Kullanıcı build / kabul sırası

1. Normal Studio build'ini al. Vertex paylaşan iki üçgenli plane'de yalnız ikinci
   üçgenin ortak UV köşesini değiştir: tek vertex kopyası oluşmalı; ilk üçgen, 3D konum
   ve normal korunmalı. Ctrl+Z/Ctrl+Y ile sayı/layout geri dönmeli. Checker/normal map,
   save/reload, ek UV setleri ve skinned mesh kontrolü yap.
2. Uygulama açıkken **ayrı terminalde** çalıştır:

   ```powershell
   python scripts/test/rt_test_uv_corner_ipc.py
   python scripts/test/rt_test_uv_authoring_ipc.py
   ```

   İlk probe IPC split/undo/redo/stale revision ve Python API yüzeyini denetler.
   `script.run_file` child'ı yalnız rt API'sini kullanır; uygulama içinden pipe açmaz.
   Probe'lar geçici nesneleri siler; history/proje modified durumunu etkiler.
3. Ek UV/custom non-trivial attribute, aktif delta yazması, skin weight ve bozuk map
   vakaları için bağımsız native hedef eklendi. Aşağıdakiler **kullanıcının çalıştıracağı**
   build komutlarıdır; ajan çalıştırmadı:

   ```powershell
   cmake -S tests/uv -B tests/uv/out
   cmake --build tests/uv/out --config Release
   ctest --test-dir tests/uv/out -C Release --output-on-failure
   ```

İlk kaynak tesliminde Studio/native testler derlenmedi ve uygulama açılmadı.
Kullanıcının build sonrası açık uygulamasında canlı probe sonuçları üstteki kayıttadır.
Non-build kontrat kontrolleri native davranış testlerinin yerine geçmez.
`python -B scripts/test/check_uv_authoring_contracts.py`: 8/8 non-build kontrolü geçti.
