# Profile / Spline Mesh UI Roadmap

## Ürün kararı

Profile ve spline ile mesh üretimi viewport-first çalışmalıdır. Kullanıcı iki
panel arasında sürekli gidip gelerek yalnızca slider değiştirmemeli; spline
noktalarını, tangent handle'larını ve path/profile ilişkisini doğrudan viewport
üzerinde düzenleyebilmelidir.

Sağ dock ikincil ayar ve bilgi alanıdır. Geometri üretiminin ana etkileşimi
viewport üzerindeki seçim, sürükleme, gizmo, snapping ve doğrudan numeric input
ile yapılır.

## Etkileşim modeli

- Profile ve Path ayrı seçim domain'leri olarak gösterilir.
- Nokta, segment ve tangent handle seçimi aynı edit selection altyapısını kullanır.
- Drag sırasında viewport overlay canlı preview üretir; commit bırakma anında yapılır.
- `G` profile/path noktasını taşır, `R` tangent yönünü döndürür, `S` tangent uzunluğunu değiştirir.
- `E` yeni spline noktası/segment ekler, `X` seçili nokta veya segmenti siler.
- `Ctrl` snapping, `Shift` hassas sürükleme, `Alt` simetri/bağımsız handle davranışı için ayrılır.
- Aktif numeric değer viewport yanında küçük bir modal/value bubble ile düzenlenir;
  kullanıcı panel açmadan kesin değer girebilir.
- Undo/redo drag başlangıcı ve commit arasında tek bir operation transaction olur.

## UI yerleşimi

Viewport üzerinde:

- profile/path control points;
- tangent handles ve normal/binormal frame göstergesi;
- sweep preview, section density ve twist yönü;
- başlangıç/bitiş cap göstergesi;
- snapping, symmetry ve axis constraint işaretleri.

Sağ contextual dock içinde:

- aktif spline/profile adı ve selection summary;
- Operation: Sweep, Extrude, Lathe, Loft, Ribbon;
- preview quality ve CPU/GPU backend durumu;
- section count, path sampling, twist, scale, cap ve UV seçenekleri;
- material/profile presetleri.

Alt editor bölgesinde:

- spline graph/curve timeline;
- numeric key/value history;
- operation stack ve non-destructive parameter snapshots.

Kalıcı büyük toolbar veya sürekli açık slider rafı eklenmez. Context dock aktif
tool'a göre değişir ve viewport iş akışını kapatmaz.

## Teknik servis sınırı

UI yalnızca ortak operation request oluşturur. Sweep/profile üretimi:

1. spline verisini doğrular;
2. deterministic sampling ve frame transport uygular;
3. section topology üretir;
4. cap/UV/normal/material kanallarını oluşturur;
5. flat `TriangleMesh` / DNA SoA'ya publish eder;
6. `FlatMeshGeometryCommand` ile undo/redo kaydeder;
7. aynı sonucu Python ve IPC'ye raporlar.

GPU compute canlı preview için kullanılabilir; commit ve başarısız GPU durumunda
CPU yolu deterministik referans olarak kalır.

## İlk vertical slice

- [x] 2D closed profile spline + 3D open path spline veri modeli
- [x] Editable başlangıç primitive'leri: circle, rectangle, open line, open arc
- [ ] Viewport point/handle selection ve drag gizmosu
- [x] Sweep mesh üretimi, cap, transported frame ve deterministic triangulation çekirdeği (`MeshEdit/ProfileSweep`)
- [x] Flat SoA publish, validation, undo/redo ile AddObjectCommand scene commit
- [ ] Contextual dock ve numeric value bubble
- [x] Python + IPC `mesh.profile.sweep.preview` ve `mesh.profile.sweep.self_test`
- [x] Rotational profile preview: editable cup/bottle radial profile + Screw/Revolve
- [x] IPC `mesh.profile.revolve.preview` ve `mesh.profile.revolve.self_test`
- [x] Python + IPC `mesh.profile.sweep.commit`
- [x] Python + IPC `mesh.profile.revolve.commit`
- [ ] CPU/GPU preview ayrımı ve self-test fixture'ları

## 2026-08-20 ilerleme notu

`ProfileSweep` ilk geometri çekirdeği olarak eklendi. Kapalı profil ve açık path
doğrudan `DNA::GeometryDetail` kanallarına (`P_orig`, `P`, `N_orig`, `N`, `uv`,
`materialID`) yazar; üçgen indeksleri, başlangıç/bitiş kapakları ve alan ağırlıklı
normaller aynı deterministik üretim adımındadır. Preview yanında scene publish
servisi, undo/redo için `AddObjectCommand` ve Python/IPC commit yüzeyleri de eklendi;
viewport UI ve edit edilebilir spline oturumu sonraki paketin işidir.

`Add > 2D Spline` artık yalnızca mesh olmayan `SplineObject` authoring kaynağı
oluşturur. Circle, Rectangle, Open Line ve Open Arc XY/XZ/YZ plane seçimiyle
oluşturulabilir. Revolve/Sweep üretimi bu kaynağın modifier/Geometry Node evaluation
katmanında kalır; Add işlemi geometri üretmez.

UI placement correction: profile authoring controls are removed from the Edit Mesh
right dock. Add will create a dedicated 2D spline authoring object; it must not
create a TriangleMesh immediately. Screw/Revolve/Sweep belong to modifier and
Geometry Node evaluation on that spline source, with axis/plane selection exposed
by the profile editor.

Scene-authored generated meshes now receive a real `Transform` handle and are
selected after commit; this is required for hierarchy selection, gizmo transforms
and modifier-panel targeting. The normal channels are also published in the
renderer-facing direction, so manual normal flipping is no longer part of the
expected workflow.

Primitive üretimi de `SplinePrimitive` modülüne ayrıldı. Primitive yalnızca başlangıç
kontrol noktalarını ve tangent handle'larını kurar; bundan sonraki viewport editleri
aynı `BezierSpline::points` verisini değiştirir. Böylece çember/dikdörtgen gibi hazır
şekiller kilitli preset değil, normal düzenlenebilir spline olarak davranır.

İlk test kapısı olarak ortak `ProfileSweepService` eklendi. UI, Python ve IPC aynı
preview üreticisini çağırıyor; IPC için `mesh.profile.sweep.self_test`, parametreli
`mesh.profile.sweep.preview/commit`, Python için `rt.mesh.profile_sweep_preview` ve
`rt.mesh.profile_sweep_commit` hazır. Revolve tarafında aynı preview/commit yüzeyleri
cup/bottle presetleriyle kullanılabilir.

IPC güvenlik/audit eşlemesi de tamamlandı: preview ve self-test Read yetkisiyle,
scene publish yapan commit operasyonları SceneWrite yetkisiyle
sınıflandırıldı. `audit_ipc_capabilities.py` sonucu: 330 metod, descriptor tablosu
güncel, sınıflandırma ve mirror kontrolü PASS.

Revolve kararı: bardak/şişe gibi dönel ürünlerde authoring kaynağı tek vertex
extrude zinciri değil, yarıçap-yükseklik düzleminde kapalı ve düzenlenebilir bir
profil spline'ıdır. Screw/Revolve bu spline'ı Y ekseni etrafında örnekler. Mesh edit
extrude/merge araçları daha sonra serbest biçimli profil düzeltmeleri için aynı
authoring verisine bağlanacaktır.
