# UV Editor — ilk uygulama dilimi

Tarih: 2026-11-03. Durum: ilk dilim kullanıcı tarafından derlendi; temel canlı IPC/Python
kabulü geçti. Boyutlandırılabilir tools ve silinmiş-hedef düzeltmesi kaynakta;
bu son değişikliklerin build/görsel kabulü bekleniyor.
Bkz. [Canlı test kaydı](UV_EDITOR_LIVE_ACCEPTANCE_2026_11_03.md).
2026-11-04: [Materyal UV Workflow taşıması](UV_MATERIAL_WORKFLOW_MIGRATION.md) uygulandı;
materyalde sampling ayarları ve kısayol, editörde bütün mesh UV araçları bulunur.
Devam dilimi: [UV corner remap](UV_CORNER_REMAP.md) kaynakta eklendi; xatlas vertex
bölünmesi ve Triangle corners authoring kullanıcı build/kabulünü bekliyor.
Tam UV editörü tamamlanmış değildir. Ana kapsam: [UV yol haritası](UV_AUTHORING_AUDIT_ROADMAP.md).

## Teslim edilen kaynak

- Alt bar `UV Editor` girişi ve Material > Texture Coordinates > `Open UV Editor` kısayolu.
- Mevcut alt dock'a açılan, taşınabilir bağımsız ImGui penceresi. Dock kapalıyken float.
- Island Tools için yatay/dikey sürüklenebilir ayırıcı, çift tıkla sıfırlama ve Tools
  ile gizleme. Dar pencerede üst/alt yerleşim; Transform/Atlas katlanabilir bölümler.
- Dock işlemi SceneUI host callback'inde kalır; UV görünümü `scene_ui.h` içermez,
  private host metodunu dışarıdan çağırmaz. Ortak header'daki kullanıcı düzeltmesi korunur.
- Doğrudan flat TriangleMesh hedef listesi; çok materyalli sibling'ler ayrı ordinal ile seçilir.
- Tam indeksli UV canvas, checker, ada renkleri, click/Shift/Ctrl seçimi, zoom/pan/frame.
- Ada seçimine veya tüm UV'lere sayısal offset/scale/rotation ve uniform normalize.
- UV değişimleri SceneHistory üzerinden undo/redo; seçim dışı UV'ler korunur.
- MIT upstream xatlas sabit commit ile eklendi; auto unwrap ve pack-only backend bağlantısı.
- Ortak flat servis üzerinden Python `rt.uv.*` ve main-thread enqueue kullanan IPC `uv.*`.
- IPC keşfinde 14 yöntemin parametreleri, yetkileri ve sınırlamaları kayıtlıdır.

Yeni çekirdek `source/include/UV` ve `source/src/UV`; arayüz `UI/UvEditorUI`;
adaptörler `Api/RtUvBindings` / `RtIpcUv` içinde. Birleşik domain/MPM implementasyonu
değiştirilmedi. Ortak API, UI ve proje dosyalarına yalnız UV bağlantıları eklendi;
diğer ajanın bu dosyalardaki değişiklikleri korunmalıdır.

## Bu dilimin açık sınırları

- Mevcut `uv` kanalı/set 0 düzenlenir; yeni UV-set yaşam döngüsü henüz yoktur.
  Eski materyal panelinin ek seti `uv` içine kopyalama davranışı bu dilimde değiştirilmedi.
- Adalar mevcut vertex indekslerini paylaşan ters yönlü manifold kenarlardan çıkarılır.
  İçe aktarılmış triangle soup'ta geometrik olarak komşu fakat ayrı indeksli yüzler
  kendiliğinden kaynak edilmez. Kalıcı corner topolojisi sonraki aşamadır.
- Seam/pin, manuel conformal unwrap, cut/stitch, GPU UV renderer, box selection ve
  interaktif G/R/S sürükleme henüz yoktur. Canvas şimdilik CPU ImDrawList kullanır;
  milyon yüz için performans kabulü yapılmış değildir.
- Güncel kaynakta xatlas UV vertex split ihtiyacı attribute koruyan remap ile
  karşılanır. Face/corner sırası değişirse işlem hâlâ reddedilir. Bkz. corner remap notu.
- Atlas işleri bu dilimde senkrondur ve 20000 üçgen ile sınırlıdır. Bu eşik bir süre
  garantisi değildir; background job/cancel sonraki aşamadır. Tek atlas desteklenir.
- Texture rebake yoktur. Mesh UV değişimi mevcut texture görünümünü değiştirebilir.
  UI'da ilgili seçenek, API'de `allow_texture_change=True` açıkça gereklidir.
- Seçim dışı adalarla aynı vertex'i paylaşan seçimler ve başka direkt mesh'le aynı
  GeometryDetail'i paylaşan hedefler değişmeden hata verir. Instance authoring bu
  ilk hedef listesinde yoktur; source UV değişimi instance görünümünü de etkileyebilir.
- Panel görünürlüğü ve editör seçimi henüz proje UI-state şemasına eklenmedi.

## API

Python işlemleri keyword argümanlarıyla çağrılır; IPC adları aynı işlemin `uv.*` karşılığıdır.

```python
import rt

rt.uv.open(object="Cube")
targets = rt.uv.targets()
info = rt.uv.info(object="Cube", mesh=0)
page = rt.uv.layout(object="Cube", mesh=0, first_face=0, limit=1000)
rt.uv.transform(
    object="Cube", mesh=0, revision=info["revision"],
    islands=[0], offset=[0.1, 0.0], scale=[1.0, 1.0],
    pivot=[0.5, 0.5], degrees=0.0, allow_texture_change=True,
)
```

Diğer işlemler: `capabilities`, `normalize`, `pack`, `unwrap`, `flip_u`, `flip_v`,
`swap_uv`, `project_view`, `set_corners`. Transform/normalize/flip/swap
için boş `islands` bütün adaları hedefler; atlas işlemleri bütün mesh üzerinde çalışır.
Her yazma güncel `revision` gerektirir; bu değer opaque'tir ve oturumlar arasında
saklanmamalıdır. Layout sayfaları 1..10000 üçgen ile sınırlıdır; `next_face` null
olana kadar devam edilir. Her düzenleme sonrası yeni revision alınmalıdır.

Python hataları RuntimeError; IPC hataları mevcut `__error` zarfıyla taşınır.
Önemli hatalar: `uv_stale_revision`, `uv_render_locked`, `uv_history_unavailable`,
`uv_texture_change_ack_required`, `uv_invalid_island`, `uv_shared_vertex`,
`uv_shared_geometry`, `uv_backend_unavailable`, `uv_vertex_split_required`,
`uv_atlas_remap_required`, `uv_async_required`.

## Lisans ve kaynak kaydı

`external/xatlas/SOURCE.txt` sabit upstream SHA ve orijinal dosya hash'lerini içerir.
MIT LICENSE ve thekla_atlas bildirimleri korunmuştur. İki yerel güvenlik düzeltmesi
`RayTrophi:` yorumlarıyla işaretlidir: boundary'siz chart bbox ve planar normal drift.
SculptForge'un GPL AutoUv, UV solver, komut veya panel kodu kopyalanmadı.

## Kontroller ve kullanıcı kabulü

Çalıştırılan non-build kontrol:

```powershell
python -B scripts/test/check_uv_authoring_contracts.py
```

Bu yedi kontrol proje/XML kayıtlarını, API/IPC-yetki eşleşmesini, flat modül sınırını,
UV görünümünün private host/header bağımsızlığını ve vendor kaynak kaydını doğrular.
C++ derleme veya davranış testi yerine geçmez.
Genel `audit_ipc_capabilities.py` ilk çalıştırmada UV okuma mirror eksiklerini yakaladı;
UV kuralları düzeltildi. Aynı çalıştırmada bağımsız `fluid.sample_solid_phi` mirror
drift'i ve genel generated descriptor güncelliği de raporlandı; domain tarafı bu işte
değiştirilmedi. UV descriptor'ları adaptörde kalıcı kayıtlıdır; generator bu delegasyonu
yeniden üretip duplicate kayıt oluşturmaması için atlar.

Kullanıcının sırası:

1. VS2022'de `RayTrophiStudio.sln` için kendi normal x64 build'ini al.
   CMake kullanılıyorsa yeni UV kaynak glob'u için yeniden configure et.
2. Uygulamayı aç; Material > Texture Coordinates > Open UV Editor ve alt bar girişini kontrol et.
3. Küpte ada seç, offset/rotation/scale uygula; Ctrl+Z/Ctrl+Y ile geri al/yinele.
   Checker, frame, zoom/pan, float/dock ve dar pencere düzenini kontrol et.
   Tools/canvas ayırıcısını iki yönde sürükle; çift tıkla sıfırla; Tools'u gizleyip
   tekrar aç. Transform/Atlas katlanmalı, dar alanda etiketler/sayılar erişilebilir olmalı.
4. Ayrı terminalde çalıştır:

   ```powershell
   python scripts/test/rt_test_uv_authoring_ipc.py
   ```

   Probe geçici benzersiz küp oluşturur ve siler; proje/history modified olur.
   Embedded script workspace içinde çalıştırma. Probe seçim izolasyonu, paging,
   stale revision, normalize, undo/redo, pack/unwrap ve silinen nesnenin UV hedeflerinden
   çıkarılmasını kontrol eder. Son deletion regression kontrolü güncel kaynak build'ini
   gerektirir; ilk derlenmiş binary'de hata canlı olarak yeniden üretildi.
5. Çok materyalli importta sibling hedeflerini ayrı düzenle; ortak geometry için
   açık ret ve UV'siz hedef için açıklayıcı hata kontrolü yap.
6. Save/reload ile değişen `uv` verisinin korunduğunu, textured nesnede appearance
   değişiminin açık seçime bağlı olduğunu kontrol et. Mevcut ek-set davranışı
   için roadmap'teki set-0 koruma işi hâlâ gereklidir.

İlk uygulama turunda build veya canlı probe çalıştırılmadı. Sonraki kullanıcı build'i
üzerinde IPC/Python kontrolleri çalıştırıldı; bu ajan build/shader compilation çalıştırmadı.
