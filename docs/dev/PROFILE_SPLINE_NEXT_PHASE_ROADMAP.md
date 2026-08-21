# Profile Spline Next Phase Roadmap

Bu roadmap yeni, River'dan bağımsız 2D Spline authoring çekirdeği içindir. River
sistemi bu çekirdeğin sahibi değildir; ileride ortak spline verisini tüketen ayrı
bir kullanıcı olarak bağlanabilir.

## Mevcut temel

- [x] Mesh-free `SplineObject`, XZ / Y-up authoring düzlemi ve temel primitive'ler.
- [x] Viewport point overlay, seçim ve tek sahipli point gizmo transform akışı.
- [x] Edit mode, selected-point dock kontrolleri ve point index overlay'i.
- [x] Bezier point handle verisi ve otomatik tangent yenileme.
- [ ] Bezier adıyla sınırlı olmayan ortak spline veri modeli.

## Faz 1 — Ortak 2D Spline modeli ve edit servisi

- [x] UI mantığından bağımsız `SplineEditService` başlangıç modülü.
- [x] Bezier segmentine hit parametresinde control point ekleme altyapısı.
- [x] De Casteljau tabanlı Bezier segment insert/subdivide altyapısı.
- [x] Açık spline endpoint extrude altyapısı.
- [ ] Eğri değerlendirme tipini point tipinden ayır: `Linear`, `Bezier`, `BSpline`.
- [ ] Linear evaluation ve linear insert/subdivide.
- [ ] Uniform cubic B-spline evaluation, knot/degree doğrulaması ve insert politikası.
- [ ] Anchor, `InHandle`, `OutHandle` seçim kimliklerini ortaklaştır.

## Faz 2 — Viewport authoring araçları

- [x] UI araç seçimi: `Select`, `Insert Point`, `Subdivide`, `Extrude`.
- [x] Insert Point aracında spline üzerinde mouse hover hit testi ve preview marker.
- [x] Sol tıkla hit parametresine yeni nokta ekleme; eklenen noktayı seçili bırakma.
- [x] Subdivide paneli: cut sayısı ve seçili segment hedefi.
- [x] Sabit yükseklikte scroll’lu control-point listesi.
- [x] Ctrl ile çoklu point selection ve viewport/list selection senkronu.
- [x] Tek seçili segment veya çoklu seçili segment batch subdivide.
- [x] Edit mode dışında seçili segment subdivide operasyonu.
- [x] Bezier handle ve B-spline control-point düzenleme panelleri.
- [ ] Extrude yalnızca açık spline’ın geçerli ilk/son anchor’ında etkin olsun.
- [ ] Extrude sonrası yeni anchor’ın viewport mouse hareketiyle taşınması ve sol tıkla onayı.
- [ ] Closed spline için endpoint extrude’ı kapat; insert/subdivide yine çalışsın.
- [x] Hover, preview, invalid-state, API undo/redo transaction ve dirty işareti.

## Faz 3 — Değerlendirme ve üretim tüketicileri

- [ ] Ortak evaluation service: position, tangent, normal, arc-length ve closest-hit.
- [ ] Screw/Revolve ve Sweep’in bu servisi kullanması; UI içinde geometri üretimi olmaması.
- [ ] Evaluated sonucu canonical flat `TriangleMesh` / DNA SoA yoluna yayınla.
- [x] Plane, closed state, curve type ve handles güvenli serialization; seçim durumu
      transient kabul edilip yüklemede güvenli biçimde sıfırlanır.
- [ ] Deterministic normals, UV, material ve modifier output policy.

## Faz 4 — Scripting, IPC ve doğrulama

- [x] Insert, subdivide, extrude ve serialization için scripting API ve IPC.
- [x] UI, scripting ve IPC’nin aynı edit core yolunu kullanması.
- [x] Validation/error semantiği, capability audit kaydı ve IPC descriptor dokümantasyonu.
- [ ] B-Spline knot insertion, modifier evaluation API ve pure core self-testleri.
- [ ] UI smoke checklist ve build/manual validation kullanıcı tarafından çalıştırılacak.

## Tasarım kararları

`CurveType` ile `PointRole` aynı enum değildir. Bezier anchor noktası `Anchor`, tangent
kolları `InHandle` ve `OutHandle` rolündedir. B-spline kontrol noktası eğri üzerinde
bulunmak zorunda olmadığından Bezier handle alanları B-spline verisine zorunlu eklenmez.

Subdivide şekli koruyan bir geometri operasyonudur: Bezier için De Casteljau, linear için
segment bölme, B-spline için knot insertion kullanılacaktır. Mouse hit testi evaluation
service'in segment ve `t` sonucuna dayanmalı; ekran örneklemesi yalnızca ilk hover kabuğudur.

## Manual validation gate

- Insert Point hover/sol tık, subdivide cut sayısı ve açık uç Extrude akışını doğrula.
- Closed spline’da endpoint Extrude’ın engellendiğini doğrula.
- Multi-select transform, handle transform ve undo/redo sınırlarını doğrula.
- Save/reload sonrası curve type, handles, knots ve seçim güvenliğini doğrula.
