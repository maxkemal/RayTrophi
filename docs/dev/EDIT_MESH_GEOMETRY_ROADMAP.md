# Edit Mesh ve İleri Geometri Yol Haritası

> Durum: statik kod incelemesi, 2026-08-20
>
> Kapsam: edit mesh, kaliteli profil geometri, boolean/SDF, spline surface ve polygon sistemi.
> Bu belge derleme sonucu değil; kaynak kod ve mevcut teknik notlar üzerinden hazırlanmış durum
> değerlendirmesi ve uygulama sıralamasıdır.

## Yönetici özeti

RayTrophi'nin edit mesh çekirdeği başlangıç seviyesinde değildir. `MeshEdit::HalfEdgeMesh`
gerçek n-gon topolojisi, explicit boundary half-edge'leri, non-manifold raporlaması ve temel
Euler operasyonlarını taşıyor. UI tarafında da flat SoA mesh'i welded editable cache'e bağlayan
bir köprü var.

Buna karşılık sistem henüz Blender/Maya seviyesinde bir üretim editörü değildir. Temel darboğaz
operatör sayısı değil, edit topolojisinin kalıcı ve kanonik olmaması, operasyon sonrası cache/SoA
senkronizasyonunun güvenilir olmaması, topology undo'nun facade snapshot'larına dayanması ve
geometri işlemlerinin scripting/IPC yüzeylerine henüz taşınmamış olmasıdır.

Yaklaşık mesafe değerlendirmesi:

| Alan | Mevcut seviye | İleri editör hedefinden mesafe |
|---|---:|---:|
| Edit topoloji çekirdeği | %45 | Yüksek |
| Günlük edit araçları | %30 | Çok yüksek |
| Kalıcı polygon/corner veri modeli | %25 | Yüksek |
| Undo/redo ve transaction güvenilirliği | %35 | Yüksek |
| Profil/sweep geometri | %10 | Çok yüksek |
| Spline surface | %10 | Çok yüksek |
| Mesh boolean | %5 | Çok yüksek |
| SDF tabanlı boolean/remesh | %15 (altyapı parçaları var) | Yüksek |
| Script + IPC geometri authoring | %10 | Çok yüksek |

Bu oranlar özellik sayısı değil, üretim kalitesi, veri sahipliği, hata toleransı ve otomasyon
paritesini birlikte ifade eden planlama tahminleridir.

Ürün hedefi, başka bir DCC'ye geçişi önlemek olmalıdır: kullanıcı primitive/profile/curve,
polygon modeling, topology edit, bevel/extrude/inset, spline surface, boolean/SDF, UV/material
ve mesh cleanup işlerini RayTrophi içinde tamamlayabilmelidir. Blender/Maya gibi araçlar yalnızca
referans ve import/export kaynağı olabilir; zorunlu üretim bağımlılığı olmamalıdır.

## Modern modelleme UI'sı

Modelleme UI'sı viewport-first kalır. Kalıcı, büyük bir araç rafı yerine mevcut ürün düzeni
kullanılır:

- **Sol context rail:** Edit / Profile / Curve / Surface / Boolean / Cleanup çalışma alanları.
- **Viewport:** doğrudan vertex/edge/face/polygon seçimi, GPU overlay, depth-tested gizmo,
  snapping, transform orientation, live operation preview ve operation confirmation.
- **Sağ contextual dock:** aktif moda göre yalnız ilgili araç ayarları, selection statistics,
  topology warnings, attribute transfer ve apply/cancel controls.
- **Bottom editor:** non-destructive operation stack, profile/curve editor, node graph,
  boolean history, preview/final quality ve background job progress.

### UI çalışma alanları

| Çalışma alanı | Ana görsel araçlar |
|---|---|
| Edit | vertex/edge/face/poly mode, linked/loop/ring/boundary selection, move/slide, extrude, inset, bevel, knife, dissolve, merge, normals |
| Profile | 2D profile canvas, control points, corner radius, holes/nesting, resolution, symmetry, saved profile assets |
| Curve | Bézier/Catmull-Rom/B-spline control points, handles, tangent/normal frame, twist/bank, arc-length and closed-loop controls |
| Surface | loft/rail/sweep/revolve, patch boundaries, trim loops, tessellation tolerance and continuity diagnostics |
| Boolean | operand picker, union/intersection/difference, exact/SDF backend, voxel resolution, tolerance, material policy and repair report |
| Cleanup | weld, fill holes, fix winding, remove degenerate, non-manifold report, triangulate/quadify, remesh and attribute transfer |

### Ortak tool kontratı

Her görsel araç aynı metadata ile register edilir:

```cpp
struct MeshToolDescriptor {
    ToolId id;
    WorkspaceId workspace;
    SelectionDomain domain;
    OperationCapabilities capabilities;
    bool previewable;
    bool undoable;
    bool scriptable;
    bool ipc_exposed;
};
```

UI yalnızca descriptor ve parametre modelini çizer; geometri üretmez. Preview, commit, undo,
CPU fallback, GPU dispatch, diagnostics, Python ve IPC aynı `GeometryEditService` operasyonuna
bağlanır. Böylece bir araç UI'da, addon'da, script'te veya node graph'ta farklı sonuç üretmez.

### Ajanlar: IPC + kanonik eğitim bilgisi

Modelleme sistemi ajanlar tarafından da kullanılacağı için IPC yalnızca düşük seviyeli bir komut
kapısı olmamalıdır. İki tamamlayıcı teslimat gerekir:

1. **Makine yüzeyi:** discovery ile listelenen IPC metotları, capability/selection-domain
   kontratı, typed params, validation error kodları, progress/cancel, operation id ve undo grup
   kimliği.
2. **Ajan bilgi yüzeyi:** tool seçimini, doğru sırayı, riskleri ve sonuç doğrulamasını anlatan
   sürümlü kanonik rehber, cookbook ve örnek iş akışları.

Önerilen IPC alanları:

- `mesh.tools.list` / `mesh.tools.describe`
- `mesh.selection.get` / `mesh.selection.set` / `mesh.selection.query`
- `mesh.operation.preview` / `mesh.operation.commit` / `mesh.operation.cancel`
- `mesh.operation.status` / `mesh.operation.diagnostics`
- `mesh.asset.snapshot` / `mesh.asset.validate`
- `mesh.undo` / `mesh.redo`
- `mesh.profile.*`, `mesh.curve.*`, `mesh.surface.*`, `mesh.boolean.*`, `mesh.cleanup.*`

Her mutasyon sonucu `{ok, operation_id, revision, changed_counts, warnings, errors, undo_group}`
şeklinde makinece okunabilir rapor vermeli. Ajan doğrudan GPU buffer veya facade triangle
listesiyle çalışmamalı; capability keşfi, selection/preview, commit, validate ve diagnostics
akışını kullanmalıdır.

Agent-facing bilgi paketi şu belgeleri içermeli:

- `docs/dev/AGENT_MESH_MODELING_GUIDE.md`: çalışma alanları, selection domain'leri, tool seçimi,
  CPU/GPU fallback, preview/commit ve hata semantiği.
- `docs/dev/AGENT_MESH_MODELING_COOKBOOK.md`: primitive → profile/sweep → edit → boolean →
  cleanup → validate → save/render uçtan uca örnekleri.
- IPC method descriptor kayıtları: parametre tipleri, varsayılanlar, limitler, undoable,
  previewable, cancelable, gerekli selection ve beklenen sonuç.
- Her yeni operasyon için script + IPC smoke testi ve ajanın takip edeceği doğrulama akışı.

Bu paket statik model eğitimi iddiası değil; uygulamayla birlikte sürümlenen agent
context/reference kaynağıdır. UI, Python veya IPC tool kontratı değiştiğinde rehber ve örnekler
aynı değişiklikte güncellenmelidir.

### Görsel kalite ve geri bildirim

- Seçili domain renkleri ve overlay yoğunluğu sabit, tema üzerinden değiştirilebilir olmalı.
- Edge/face occlusion render depth ile test edilmeli; arka taraftaki elemanlar ayrı soluk renkte
  gösterilebilmeli.
- Live preview sırasında yeni/etkilenen topology, boundary, non-manifold ve attribute kaybı
  anında diagnostics olarak görünmeli.
- Her destructive commit'ten önce sağ dock'ta etkilenen vertex/edge/face/poly sayısı,
  üçgen sayısı, memory tahmini ve undo maliyeti gösterilmeli.
- Uzun boolean/SDF/surface işlerinde progress, cancel, preview quality ve final quality ayrı
  kontrol edilmeli.
- Render mode'da overlay ve picking, raster mode ile aynı görsel ve seçim kontratını kullanmalı.

### İlk UI vertical slice

İlk uygulanacak görsel paket şu sırada olmalı:

1. Edit workspace rail + vertex/edge/face/poly mode switcher.
2. GPU depth-tested edit overlay: point, edge, face fill, active element, hidden/occluded state.
3. Transform gizmo, snapping, orientation/pivot ve selection statistics dock.
4. Tool options dock: Move, Extrude, Inset, Loop Cut ve seçili Edge Bevel.
5. Live preview + Apply/Cancel + one-step undo transaction.
6. Render/path-traced overlay ve ID/depth picking parity.
7. Profile workspace için 2D profile canvas ve Curve workspace için temel control-point editor.
8. Boolean workspace için operand listesi, Exact/SDF seçimi ve diagnostics.
9. Aynı vertical slice için IPC method descriptors, agent guide/cookbook ve smoke test akışı.

Bu paket tamamlanmadan yeni araç eklemek yerine UI/core contract stabilitesi ölçülmelidir. Yeni
araçlar bundan sonra aynı descriptor, operation service ve addon registry üzerinden eklenir.

Hedef çıtası modern DCC'lerin edit mesh ve mesh üretim hattıdır: yüksek polygon sayısında
çalışan seçim/overlay, gerçek topology authoring, güvenilir undo, n-gon/corner attribute
koruması, non-destructive üretim, profil/curve/surface araçları ve exact/SDF boolean
seçenekleri. Bu hedefe göre mevcut sistem bir prototip değil, fakat production editörün erken
altyapı aşamasındadır.

## GPU compute kararı: hibrit edit pipeline

GPU compute edit mesh için güçlü bir hızlandırıcıdır, fakat topology'nin tek ve geri dönüşsüz
otoritesi olmamalıdır. GPU için en uygun işler yoğun ve düzenli veri paralelliği olanlardır:
vertex transform/deform, proportional editing, normal/tangent üretimi, attribute/mask
işlemleri, selection ID/depth buffer, SDF voxelization, marching/dual-contour preview,
subdivision preview ve büyük mesh overlay buffer üretimi.

CPU tarafında kalması gerekenler ise düzensiz topology cerrahisi ve doğrulama işleridir:
edge split/collapse/flip, face merge/split, n-gon loop değişimi, non-manifold kararları,
stable id/remap, attribute domain transferi, transaction/undo ve save/load. Bu işlemler GPU'da
araştırılabilir ama ilk üretim mimarisinde GPU-only yapmak; hata ayıklama, determinism ve
farklı GPU davranışları açısından gereksiz risk taşır.

Kanonik veri kuralı workspace talimatıyla uyumlu biçimde flat `TriangleMesh` / DNA SoA'dır.
Yeni bir topology cache veya operation index'i eklenebilir, ancak facade `Triangle` koleksiyonu
kanonik sahne geometrisi olamaz. GPU ve CPU aynı flat SoA + topology metadata kontratını tüketir;
sonuç yine flat SoA'ya publish edilir. Böylece addon'lar da aynı operation service'e bağlanır.

Önerilen üç çalışma seviyesi:

1. **CPU authoritative:** tüm operasyonlar CPU'da; referans, fallback, deterministic test ve
   düşük/orta yoğunluklu mesh yolu.
2. **GPU accelerated:** CPU topology kararını verir; GPU compute vertex/attribute/preview
   işini yürütür; revision ve fence ile sonucu flat SoA'ya geri bağlar.
3. **GPU preview / deferred commit:** interaktif sürüklemede GPU yalnız geçici preview üretir;
   mouse release veya Apply anında CPU authoritative operation commit eder. Büyük boolean/SDF
   işlemleri progress/cancel destekli asenkron job olabilir.

Her operasyon için `OperationCapabilities` yayınlanmalı: `cpu`, `gpu_preview`, `gpu_commit`,
`deterministic`, `supports_cancel`, `max_memory_hint`. GPU yolu yoksa sessizce farklı sonuç
üretmek yerine aynı parametrelerle CPU fallback'e geçmeli ve diagnostics üretmelidir. GPU/CPU
sonuçları toleranslı geometri karşılaştırmasıyla düzenli olarak çapraz doğrulanmalıdır.

Bu yapı addon modelini de mümkün kılar: addon yeni bir tool/node/operation tanımlar, fakat
selection, validation, transaction, publish, Python ve IPC yüzeyleri ortak core service'ten
gelir. Addon ister CPU kernel, ister GPU compute kernel, ister yalnızca UI ve parametre tanımı
sağlayabilir.

## Mevcut yapı

### Güçlü taraflar

- `MeshEdit::HalfEdgeMesh` index tabanlıdır; vertex, half-edge, edge ve face ayrı tutulur.
- Boundary loop'ları explicit half-edge olarak temsil edilir; `validate()` yapısal bütünlüğü
  kontrol eder.
- `splitEdge`, `splitFace`, `flipEdge`, `collapseEdge`, `dissolveEdge`, `extrudeFace`,
  `insetFace` ve quad edge loop/ring yürüyüşleri vardır.
- `EditableMeshCache` vertex/edge/triangle/polygon seçimlerini ve welded vertex ilişkilerini
  tutar. Flat SoA seam kopyalarını `flat_soa_offsets`/`flat_soa_data` ile birlikte günceller.
- `ensureEditableHalfEdge()` half-edge'i polygon cache'ten lazy olarak üretir.
- UI'da translation, transform, extrude, inset, loop cut, dissolve, vertex dissolve, normal
  flip ve shading işlemleri bulunur. Seçili edge bevel kullanıcıya açık/çalışan bir edit
  operasyonu değildir.
- Geometry Nodes tarafında Extrude, Inset, Bevel ve Remesh prototipleri vardır. Flat SoA için
  half-edge'e giriş/çıkış köprüsü de vardır.
- SDF/isosurface/marching-cubes altyapısı özellikle fluid, volume ve remesh yollarında zaten
  kullanılmaktadır. Bu, boolean için yararlı bir temel ama henüz genel mesh boolean sistemi
  değildir.

### Kritik eksikler ve riskler

1. **İki gerçeklik var.** Render'ın kanonik verisi flat `TriangleMesh` SoA iken edit cache ve
   bazı topology undo yolları `vector<shared_ptr<Triangle>>` facade/cage saklıyor.
2. **Polygon bilgisi flat mesh'te kalıcı değil.** Quad/ngon çoğu zaman triangulation sonrası
   kayboluyor; node tarafı coplanar triangle çiftlerini heuristik olarak yeniden birleştiriyor.
3. **Edit işlemleri aynı veri kontratından geçmiyor.** Bazı operasyonlar half-edge + yeniden
   üretim, bazıları triangle soup, bazıları doğrudan SoA kullanıyor.
4. **Topology undo snapshot/delta modeli tamamlanmamış.** Büyük meshlerde bellek ve kimlik
   stabilitesi sorunu çıkarabilir.
5. **Edit bevel yok.** Bevel bugün object/geometry modifier tarafında kullanılabilir. Seçili
   edge'lere edit-mode bevel uygulayan tamamlanmış bir kullanıcı özelliği yoktur. Eski kaynakta
   buna yönelik fonksiyon/çekirdek denemeleri bulunması, özelliğin mevcut olduğu anlamına gelmez;
   bunlar çalışır ürün kapsamına alınmamalıdır.
6. **Boolean çekirdeği yok.** Kodda genel mesh Boolean node/operatörü bulunmuyor; mevcut boolean
   referansları çoğunlukla flag veya fizik/volume bağlamında.
7. **Spline yalnızca eğri matematiği seviyesinde.** `BezierSpline` değerlendirme, tangent,
   arc-length ve sample üretir; profile sweep, frame transport, cap, seam ve surface patch
   sistemi yok.
8. **Authoring API paritesi eksik.** Mesh buffer okuma/yazma var; topology, profile, boolean,
   surface ve polygon operasyonlarının aynı core service üzerinden Python + IPC kontratı yok.

9. **Overlay yalnızca görüntü yolunun bir parçası.** Edit mesh overlay GPU raster viewport
   yolunda çalışıyor. Render/path-traced viewport'ta aynı edit overlay katmanı yok; mevcut
   seçim maskesi/ImGui fallback'i edit vertex/edge/face authoring overlay'inin karşılığı değil.
   Bu nedenle render modunda edit işlemleri görünür topoloji geri bildirimi olmadan yapılabiliyor.
10. **Triangle sınırı gerçek ürün sınırı.** CPU raster seçim outline'ında
    `MAX_TRIS_FOR_RASTER = 1000000` kapısı var. Bu, edit mesh'in çözmesi gereken bir sınırlama
    değil; yoğun mesh için GPU overlay veya decimated display proxy gerekir. Modern DCC hedefinde
    kullanıcıya uygulanan sabit triangle limiti kaldırılacak; yalnızca bellek/performans ve
    isteğe bağlı preview LOD politikaları kalacak.

## Kütüphane değerlendirmesi ve karar

Tek bir hazır kütüphane modern DCC hattının tamamını sağlamıyor. Bu nedenle “her şeyi kendimiz
yazalım” da “kütüphaneyi doğrudan kanonik mesh yapalım” da doğru değil. Doğru strateji: kendi
`EditableGeometryAsset` ve operation/undo/publish kontratımızı koruyup, uygun kütüphaneleri
izole backend olarak kullanmak.

| Kütüphane | Kullanılabileceği alan | Karar |
|---|---|---|
| OpenMesh | BSD-3 half-edge/topology ve property yaklaşımı | Mevcut half-edge'e alternatif olarak spike; tek başına bevel/boolean/DCC hattı değil |
| Geometry Central | Genel/manifold surface mesh, polygon ve property erişimi | Akademik/işlem backend'i için incelenebilir; kendi asset kimlik/undo kontratımızın yerine geçmez |
| PMP | MIT polygon mesh data structure, remesh, subdivision, smoothing, decimation | En uygun permissive yardımcı kütüphane adayı; operasyon kapsamı ayrıca doğrulanacak |
| libigl | MPL-2 tabanlı geometri işleme ve bazı boolean/processing algoritmaları | Seçili algoritmalar için adapter; `copyleft` parçaları lisans incelemesi olmadan kullanılmaz |
| CGAL | En güçlü exact predicates, mesh processing ve robust boolean seçenekleri | GPL veya ticari lisans nedeniyle ancak ticari lisans kararıyla; doğrudan bağımlılık olarak şimdilik yok |
| Manifold | Manifold mesh boolean backend'i | SDF ve exact mesh boolean spike'ında ayrıca benchmark; genel edit topology sahibi değil |

İlk teknik spike üç parçalı olmalı: PMP/OpenMesh/Geometry Central ile topology operasyonu,
Manifold ile boolean, mevcut SDF ile voxel boolean. Aynı corpus üzerinde non-manifold, UV seam,
material/corner attribute, 1M+ triangle, self-intersection ve undo/publish ölçülür. Spike sonucu
başarısızsa ilgili algoritma kendi focused modülümüzde yazılır; kütüphane API'si kanonik veri
modeline sızmaz.

Lisans notu: CGAL resmi olarak GPL veya ticari lisanslıdır; libigl çekirdeği MPL-2 olarak
belirtilse de bazı copyleft algoritmalar ayrı namespace'tedir. PMP MIT, OpenMesh BSD-3 olarak
yayınlanır. Bu nedenle ticari dağıtım için kütüphane seçimi yalnızca teknik değil, lisans
uyumluluğu kararıdır.

## Mimari kararlar

### 1. Kanonik edit varlığı: `EditableGeometryAsset`

Yeni sistemde edit mesh, geçici UI cache değil, sahne nesnesinin topology sahibi olmalıdır.
Önerilen veri:

- welded vertex positions ve stable vertex id
- polygon loops / half-edge adjacency
- corner domain: UV, normal, tangent, material, custom attributes
- edge flags: crease, sharp, seam, bevel weight, selection group
- face material ve polygon id
- source/derived revision ve topology generation

Flat `TriangleMesh` SoA bunun render/cache çıktısıdır. Edit işlemi önce asset üzerinde çalışır,
sonra deterministik triangulation ile SoA'ya publish edilir. UI, script ve IPC aynı geometry
service'i çağırır.

### 2. Kimlik ve attribute kuralları

- Vertex/edge/face/polygon id'leri operation içinde mümkün olduğunca korunur; silinenler
  tombstone olur, publish/compact aşamasında remap tablosu üretilir.
- UV, normal, tangent ve material vertex değil corner-domain verisi olarak ele alınmalıdır.
- Her polygon için kalıcı `poly_id`; her üçgen için `poly_id` ve `corner remap` tutulmalıdır.
- Triangulation görünmez bir veri kaybı olmamalıdır.
- Non-manifold ve açık yüzeyler reddedilmek yerine açık hata/uyarı raporu ile işlenmelidir.

## Uygulama yol haritası

### Faz 0 — DCC hedefi, ölçüm ve güvenlik kapısı

- Half-edge self-test'i genişlet: cube, grid, ngon, hole, open shell, bow-tie, non-manifold,
  UV seam, degenerate ve inverted winding.
- Her topology operasyonu için invariant raporu: face/edge/vertex sayıları, boundary loop,
  manifold durumu, flipped/zero-area yüzler, attribute kaybı.
- Kütüphane spike benchmark corpus'u: 10K, 100K, 1M ve 10M triangle; UV seam, n-gon, hole,
  non-manifold, self-intersection ve animated transform vakaları.
- Edit overlay performans bütçesi ve sabit triangle cap'in kaldırılması için GPU buffer/LOD
  tasarımı.
- UI'dan bağımsız `GeometryOperationReport` ve deterministic test fixtures ekle.
- Derleme kullanıcı tarafından yapılacak; Codex doğrulaması statik/test kaynaklarının incelenmesi
  ile sınırlı kalacak.

### Faz 1 — Edit mesh'in kanonikleştirilmesi ve limitsiz ölçek

- `EditableGeometryAsset` ve `GeometryEditService` modüllerini yeni focused `.h/.cpp` dosyaları
  olarak ekle.
- Flat SoA ↔ asset dönüşümünü tek köprüye indir; polygon/corner/material/UV bilgisini koru.
- `EditableMeshCache` yalnızca viewport hızlandırma/index cache'i olsun; topology sahibi olmasın.
- `TopologySnapshot` veya block/delta tabanlı undo ekle; tek kullanıcı hamlesi tek transaction olsun.
- Publish sonrası CPU BVH, Vulkan/OptiX geometri, selection ve overlay aynı revision ile
  yenilensin.
- Edit display için triangle sayısına dayalı reddetme kaldırılmalı. Büyük meshler indexed GPU
  vertex/edge/face buffers, visibility culling, selection ID buffer ve gerektiğinde yalnızca
  görüntü amaçlı decimated proxy ile çizilmeli; proxy topology sahibi olmamalı.

Çıkış ölçütü: edit edilen quad/ngon, UV seam, material id, undo/redo ve save/load döngüsünde
  polygon kimliğini ve görünümünü korur.

### Faz 2 — Güvenilir günlük edit araçları ve render overlay

- Selection graph: vertex/edge/face/poly, linked, boundary, shortest path, loop/ring, angle,
  material ve normal tabanlı seçim.
- Move/slide: vertex slide, edge slide, face slide, multi-edge parallel slide.
- Topology: subdivide, knife/project cut, connect vertices, poke, triangulate, quadify,
  dissolve limited by angle/planar region, merge by distance.
- Extrude region, individual faces, along normal, along connected normal; inset region ve
  individual faces; normal/transform orientation seçenekleri.
- Edit-mode edge bevel'i sıfırdan asset üzerinde çalıştır; seçili edge setinden başlayıp yeni
  edge/face/poly kimlikleri üret, sonra overlay'i aynı publish path'i ile yenile. Boundary,
  clamp overlap, miter, segments, profile, harden normals ve material continuity testleri ekle.
- GPU edit overlay'i render/path-traced moda taşı: scene depth/ID ile occlusion doğru çalışmalı,
  vertex/edge/face selection ayrı ID kanallarından okunmalı, selection highlight ve transform
  gizmo render görüntüsünün üstünde doğru kompozitlenmeli. CPU ImGui çizimi yalnız fallback/debug
  olmalı.
- Render modunda edit picking için render-depth readback veya düşük maliyetli ID/depth pass
  kullanılmalı; körlemesine dünya/ekran projeksiyonlu işlem yapılmamalı.

Çıkış ölçütü: tüm araçlar aynı topology service'i kullanır; facade-only operasyon kalmaz.

### Faz 3 — Profil geometri ve kaliteli sweep

- `Profile2D`: kapalı/açık loop, winding, delik/nesting, corner/continuity, resolution ve
  per-point radius/attribute.
- Primitive profile üreticileri: circle, rectangle, rounded rectangle, gear, custom polygon.
- `CurvePath`: polyline, Bézier, Catmull-Rom/B-spline; arc-length parametrizasyonu.
- Frame sistemi: Frenet fallback + parallel transport; twist, banking, up-vector ve closed-loop
  seam correction.
- `Sweep/Loft`: profile along path, variable scale/twist, caps, miter/round corners, UV
  generation, material regions, adaptive sampling.
- `Revolve/Lathe`: profile axis, angular segments, poles, seam weld ve cap.

Çıkış ölçütü: aynı profil/path hem interactive edit hem geometry node hem Python/IPC'den üretir;
  seam, normals, UV ve cap kalitesi deterministik olur.

### Faz 4 — Spline surface ve patch mesh

- Curve network ve surface boundary graph.
- Bézier/NURBS benzeri 4x4 patch değerlendirme; continuity diagnostics (G0/G1/G2).
- Loft, ruled surface, network surface, Gordon-style patch ve rail sweep.
- Trim loop, hole, tessellation tolerance, adaptive triangulation ve polygon preservation.
- Edit cage ↔ evaluated surface ayrımı; control point, handle, knot ve trim seçimleri.

Çıkış ölçütü: surface edit, tessellation ve mesh publish birbirinden ayrıdır; yeniden tessellation
edit asset'in kimlik/attribute kontratını bozmaz.

### Faz 5 — Boolean ve SDF geometri

İki yol birlikte tasarlanmalı:

1. **Exact mesh boolean:** kapalı/manifold meshlerde intersection curve çıkarma, face split,
   inside/outside sınıflama, coplanar politika, winding repair, attribute transfer ve yeniden
   triangulation. Bu yol keskin CAD yüzeyleri ve kontrollü topology için.
2. **SDF boolean/remesh:** mesh/primitive/surface'i signed distance field'e voxelize etme,
   union/intersection/difference/smooth variants, narrow-band evaluation ve dual contouring
   veya marching cubes ile yüzey çıkarma. Bu yol organik, self-intersect ve bozuk input için.

Ortak `BooleanResult` şunları raporlamalıdır: input validity, open/non-manifold warnings,
  voxel resolution/error estimate, removed/created regions, material policy, attribute transfer,
  topology quality ve preview/final quality.

SDF yolunu ilk üretim boolean olarak tercih etmek hızlı ve toleranslıdır; ancak CAD doğruluğu
  gereken sonuçlarda exact mesh boolean zorunlu ikinci backend olarak kalmalıdır. SDF yalnızca
  mevcut fluid yüzey kodu kopyalanarak yapılmamalı; genel `GeometrySDF` asset/cache katmanı
  kurulmalıdır.

### Faz 6 — Non-destructive geometry graph

- Profile, sweep, surface, boolean, remesh ve edit operations node olarak aynı core service'i
  kullanmalı.
- Preview/final kalite profilleri ve cached evaluation.
- Apply/commit sınırı: graph çıktısını kanonik edit asset'e dönüştürür, undo tek transaction olur.
- Node çıktılarında source map, poly id, material/UV transfer ve diagnostics saklanır.

### Faz 7 — Script, IPC ve üretim kalitesi

- `rt.mesh.edit`, `rt.profile`, `rt.curve`, `rt.surface`, `rt.boolean` dar ve isim tabanlı API'ler.
- Her mutation validation/error contract ile `Result` döndürmeli ve undoable olmalı.
- IPC aynı method isimlerini ve hata kodlarını taşımalı; binary mesh aktarımı için mevcut JSON
  dışı/handle tabanlı yol genişletilmeli.
- Smoke test matrisi: topology round-trip, boolean validity, profile seam, surface tessellation,
  attribute transfer, undo/redo, save/load ve IPC/Python parity.

## Öncelik sırası

En doğru yatırım sırası şudur:

1. Kütüphane spike + limitsiz GPU edit display mimarisi.
2. Kanonik edit asset + polygon/corner kimlikleri.
3. Topology transaction/undo ve tek publish path.
4. Render/path-traced edit overlay ve doğru picking.
5. Seçili edge bevel dahil günlük edit araçlarının tamamlanması.
6. Profile2D + CurvePath + sweep/revolve.
7. Genel GeometrySDF ve SDF boolean/remesh.
8. Exact mesh boolean.
9. Spline surface/patch/trim.
10. Node, Python ve IPC parity; final kalite ve performans optimizasyonları.

Boolean'a doğrudan başlamak cazip görünse de mevcut veri modeliyle sonuçlar kalıcı polygon,
attribute ve undo sorunlarını büyütür. Önce edit asset ve publish kontratı sabitlenirse boolean,
profil ve surface sistemleri aynı geometri omurgasına oturur.

## RayTrophi içinde tam mesh üretim DCC'si için Go/No-Go kapısı

Mevcut yol haritası gerçek ihtiyacı karşılıyor; ancak aşağıdaki kapılar tamamlanmadan “kullanıcı
mesh üretimi için harici DCC'ye ihtiyaç duymaz” hedefi tamamlanmış sayılmamalı:

- Flat `TriangleMesh` / DNA SoA kanonik ve facade-free authoring yolu olarak doğrulanmış olmalı.
- Polygon, corner, UV seam, material, crease/sharp ve custom attribute kimlikleri topology
  operasyonlarından sonra korunmalı.
- Tek bir operation service; UI, GPU preview, CPU fallback, addon, Python ve IPC tarafından
  paylaşılmalı.
- CPU authoritative path her operasyonda çalışmalı; GPU yolu aynı sonuç kontratına ve revision
  sistemine bağlanmalı.
- Render/path-traced modda depth-tested vertex/edge/face overlay ve picking çalışmalı; edit
  işlemi yalnız raster viewport'a bağlı kalmamalı.
- Sabit triangle reddi kalkmalı. 1M+ triangle için indexed GPU buffer, culling ve isteğe bağlı
  display proxy çalışmalı; proxy topology sahibi olmamalı.
- Seçili edge bevel, loop cut, extrude region, inset region, dissolve, split/knife, merge,
  normal/shading ve temel selection modelleri aynı publish/undo yolundan geçmeli.
- Her topology işlemi undo/redo, save/load, material/UV transfer, non-manifold warning ve
  deterministic CPU testinden geçmeli.
- En az bir gerçek addon; yeni bir selection/tool/operation olarak çekirdeği değiştirmeden
  register olup UI + Python + IPC'den çalıştırılmalı.
- PMP/OpenMesh/Geometry Central/Manifold/SDF spike sonuçları lisans, kalite, hız, attribute
  transfer ve bakım maliyetiyle belgelenmiş olmalı.

Bu kapıların anlamı “ilk sürümde Blender'ın tüm araçlarını bitirmek” değildir. Anlamı, RayTrophi
içinde üretim yapılabilecek minimum tam döngünün kanıtlanmasıdır: primitive/profile/curve ile
başla, edit et, topology değiştir, boolean/surface üret, UV/material ata, kaydet ve render et.
Bu nedenle ilk vertical slice yalnızca seçili edge bevel değil; seçili edge bevel + 1M+ indexed
mesh + render-mode overlay/picking + CPU/GPU parity + undo/save/load + basit profile sweep ve
SDF boolean içermelidir.

## Kaynak referansları

- `RayTrophiStudio/source/include/MeshEdit/HalfEdgeMesh.h`
- `RayTrophiStudio/source/src/MeshEdit/HalfEdgeMesh.cpp`
- `RayTrophiStudio/source/include/scene_ui.h` — `EditableMeshCache`
- `RayTrophiStudio/source/src/UI/scene_ui_mesh_overlay.cpp`
- `RayTrophiStudio/source/include/GeometryNodesV2.h`
- `docs/dev/flat_mesh_facade_audit.md`
- `docs/dev/refactoring_implementation_plan.md` — mevcut bevel/remesh/flat-half-edge notları
- `docs/dev/API_SCRIPTING_ROADMAP.md`

## Uygulama durumu / takip panosu

### Tamamlandi

- [x] Flat `TriangleMesh` / DNA SoA mesh validation: finite vertex/normal, index bound, alignment ve degenerate triangle raporu.
- [x] Mesh tool registry: workspace, selection domain, availability, CPU/GPU capability, preview ve undo metadata.
- [x] IPC discovery: `mesh.tools.list`, `mesh.tools.describe` ve `mesh.asset.validate`.
- [x] Ortak operation preflight kontrati: `MeshOperationRequest` / `MeshOperationPlan`.
- [x] `mesh.operation.plan`: object/tool/backend/faz/selection/revision kontrolleri ve CPU fallback raporu.
- [x] Yeni moduller proje dosyalarina eklendi; buyuk mevcut dosyalara yalnizca entegrasyon wiring'i yapildi.

### Devam ediyor - Batch 1

- [x] `MeshTopologyTransaction` source module: copy-on-write working topology, validation gate, atomic publish, cancel ve operation report.
- [x] Deterministic in-memory topology self-test ve `mesh.operation.self_test` IPC endpoint eklendi ve derlenmis uygulamada dogrulandi.
- [x] Half-edge -> flat DNA SoA publisher eklendi: topology validation, indexed triangulation, normal rebuild, transform bake ve mevcut vertex attribute kopyalama.
- [x] Flat publisher self-test yeni derleme sonrasi IPC ile dogrulandi.
- [x] Flat `GeometryDetail` undo/redo snapshot modulu ve memory estimate eklendi.
- [x] Undo/redo snapshot self-test yeni derleme sonrasi IPC ile dogrulandi.
- [x] `FlatMeshEditService` eklendi: topology publish, flat validation, SceneHistory command ve ortak scene refresh tek service akışında.
- [x] `FlatMeshGeometryCommand` eklendi; flat geometry undo/redo SceneHistory komutu olarak temsil ediliyor.
- [x] Python `rt.mesh.validate()` ve `rt.mesh.plan_operation()` eklendi; IPC ile aynı `rtapi` preflight hattını kullanıyor ve uygulama içinde smoke testten geçti.
- [x] İlk gerçek undoable mutation yüzeyi eklendi: IPC `mesh.operation.commit_positions` ve Python `rt.mesh.set_positions_undoable()`.
- [x] Undoable position commit yeni derleme sonrasi Python + IPC undo/redo ile doğrulandi: 3 PASS, 0 FAIL.
- [ ] UI operasyonlarÄ±nÄ±n mevcut overlay kodundan bu service'e taşınması.
- [x] Flat SoA edit modunda sağ tool dock'un boş kalmasına neden olan facade-null guard düzeltildi; flat mesh için active object adı artık yeterli UI context.
- [x] Sağ edit dock için registry tabanlı context header eklendi: canonical path, CPU authority, selection mode, V/E/F istatistikleri ve hazır/planned tool sayıları.
- [x] Edit dock açılışı active facade-name state’ine bağımlı olmaktan çıkarıldı; flat SoA edit mode artık seçili object state’iyle operator dock’u gösterebilir.
- [ ] **UI blocker / tekrar incelenecek:** Edit mode sağ dock penceresi açılıyor ancak operator ikonları ve selection actions (Vertex, Edge, Face, Extrude, Delete, Merge, Dissolve vb.) görünmüyor. Şüpheli zincir: `drawPaintBrushDock()` → `drawEditToolControls()` çağrı koşulu, `mesh_triangle`/flat `TriangleMesh` tür geçişi, dock content clipping/width veya `UIWidgets::IconActionButton` çizim yolu. Ayrı görsel debug batch’inde ImGui item trace ve aktif dock screenshot/probe ile doğrulanacak.
- [x] Ortak neden doğrulandı: `resolvePaintMesh()` yalnızca Triangle facade döndürüyor; flat TriangleMesh’te edit/sculpt target yok sanılıp erken çıkılıyordu. Sculpt guard da active object name fallback’iyle düzeltildi.
- [x] UI batch sonrası core IPC regresyonu: 11 PASS, 0 FAIL; Python commit/undo/redo: 3 PASS, 0 FAIL.
- [x] Ortak saÄŸ dock clipping nedeni bulundu: splitter `InvisibleButton` panel yÃ¼ksekliÄŸi kadar cursorâ€™Ä± aÅŸaÄŸÄ± taÅŸÄ±yordu; content cursorâ€™Ä± geri yÃ¼kleniyor. Edit tool shelf flat targetâ€™ta facade olmadan da Ã§iziliyor; derleme sonrasÄ± gÃ¶rsel smoke bekliyor.
- [x] Ana Properties panelinin ortak boÅŸ gÃ¶rÃ¼nme nedeni bulundu: tam yÃ¼kseklikli sidebar splitter `Button` sonrasÄ± `PropContentArea` cursorâ€™Ä± panel altÄ±nda kalÄ±yordu; content Y konumu splitter sonrasÄ± geri yÃ¼kleniyor.
- [ ] Undo/redo snapshot ve operation revision sistemi.
- [ ] Python binding'in `mesh.operation.plan` ve ortak operation raporuna baglanmasi.
- [ ] Mevcut HalfEdge Euler operasyonlarinin bu service contract'a alinmasi.

### Siradaki buyuk batch'ler

- [ ] Batch 2: selected edge bevel, selection publish ve 1M+ indexed GPU edit display.
- [ ] Edit selected-edge bevel: deneysel facade yolu durduruldu; ters yön, köşe temizliği ve çoklu seçim sorunları nedeniyle production UI'dan çıkarıldı.
- [x] Bevel replace hatasi bulundu ve merkezi `replaceSceneObjectsForNode()` flat-aware yapildi: eski TriangleMesh artik yeni facade meshinin yaninda kalmiyor; replacement tekrar tek flat TriangleMesh olarak publish ediliyor.
- [x] Bevel konkav vertex patch artifakti icin fan triangulation kaldirildi; yeni `MeshEdit::triangulatePlanarPolygon()` ear-clipping + 3D winding kontrolu kullaniliyor.
- [x] Bevel inset yönü winding bağımlılığından çıkarıldı: yüz merkezine projeksiyon ile gerçek yüz-içi yön kullanılıyor; ters bevel vakası için yeni düzeltme derleme sonrası smoke bekliyor.
- [x] Flat bevel undo snapshot düzeltildi: `ReplaceMeshGeometryCommand` facade listesini değil, before/after `GeometryDetail` snapshot'ını `FlatMeshGeometryCommand` ile saklıyor; undo/redo sonrası UI mesh cache invalidation eklendi.
- [ ] Bevel için hazır topology kernel değerlendirmesi: lisans, flat SoA adaptörü, multi-edge/multi-face bevel, corner join, winding, undo ve CPU/GPU stratejisi karşılaştırılacak.
- [ ] Batch 3: render/path-traced mode depth-tested overlay ve ID picking.
- [ ] Batch 4: profile/curve/sweep, spline surface ve SDF boolean vertical slice.
- [ ] Öncelik değişikliği: Batch 4 içinde profile/spline mesh üretimi, bevel/library kernel'den önce gelecek; viewport-first UI kontratı `docs/dev/PROFILE_SPLINE_MESH_UI_ROADMAP.md` ile izlenecek.

Batch 1 preflight + topology transaction + flat publisher + undo/redo IPC smoke testi kullanici derlemesinden sonra tamamlandi: 11 PASS, 0 FAIL. Python mesh binding smoke testi: 1 PASS, 0 FAIL.
Her batch, kullanici derlemesinden sonra tek toplu IPC/regresyon testiyle kapatilacak.
