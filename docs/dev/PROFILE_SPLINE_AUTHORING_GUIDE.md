# 2D Spline Authoring Guide

Bu belge Profile Spline çekirdeğinin UI, Python, IPC ve proje dosyası arasındaki
ortak sözleşmesini anlatır. River sistemi bu API’nin sahibi değildir; River,
ortak spline verisini tüketen ayrı bir kullanıcıdır.

## Curve type

- `linear`: Kontrol noktaları doğru parçalarının uçlarıdır.
- `bezier`: Anchor noktaları ve relative `tangent_in` / `tangent_out` handle’ları kullanır.
- `bspline`: En az dört kontrol noktasıyla uniform cubic B-spline değerlendirmesi yapar.

Curve type ile point role aynı kavram değildir. B-spline kontrol noktaları handle
değildir; seçilen kontrol noktasının pozisyonu doğrudan düzenlenir.

## Python

```python
import rt

print(rt.spline.list())
payload = rt.spline.get("OpenLineSpline")
rt.spline.insert_point("OpenLineSpline", segment=0, t=0.5)
rt.spline.subdivide("OpenLineSpline", segments=[0, 1], cuts=2)
rt.spline.extrude("OpenLineSpline", endpoint=1, position=(4.0, 0.0, 2.0))
```

`rt.spline.get()` versioned JSON metni döndürür. Değiştirilmiş payload,
`rt.spline.set(name, payload)` ile doğrulanarak uygulanabilir.

## IPC

Her çağrı JSON-RPC gövdesi kullanır:

```json
{"id": 1, "method": "spline.subdivide",
 "params": {"name": "OpenLineSpline", "segments": [0, 1], "cuts": 2}}
```

Temel yöntemler `spline.list`, `spline.get`, `spline.set`, `spline.insert_point`,
`spline.subdivide` ve `spline.extrude`’dur. Başarısız doğrulamalar `__error`
alanıyla döner; UI, Python ve IPC aynı `SplineEditService` yolunu kullanır.
Mutasyonlar `SceneHistory` snapshot komutuna kaydedilir; bu nedenle `rt.undo()` ve
`rt.redo()` Python/IPC çağrılarından sonra da aynı spline durumunu geri getirir.

## Serialization

Proje JSON’unda `spline_objects` dizisi altında `rt.spline.v1` payload’ları
saklanır. Payload; isim, plane, curve type, closed state, transform, anchor
pozisyonları, Bezier handle’ları ve user data’yı içerir. Seçim ve geçici hover
durumu kalıcı veri değildir; yüklemede temizlenir.

## Modifier hazırlığı

Spline authoring payload’ı modifier’lara doğrudan source olarak bağlanır:

1. `evaluation service` position/tangent/normal/arc-length üretir.
2. Screw/Revolve profili açısal örnekler.
3. Sweep path frame’i taşır.
4. Loft aynı parametre domain’inde birden fazla spline’ı eşler.
5. Skin çoklu spline kesitlerini ortak ring topology’ye yayınlar.
6. Tüm sonuçlar canonical flat `TriangleMesh` / DNA SoA yoluna gider.

Modifier UI, Python ve IPC geometri matematiğini tekrar etmez; aynı evaluation ve
publish servislerini çağırır.
