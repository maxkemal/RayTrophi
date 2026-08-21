# Agent Mesh Modeling Guide

Bu belge, RayTrophi mesh modeling araçlarını kullanan ajanlar için kanonik referanstır.
Uygulama tool descriptor ve IPC method listesini her zaman bu belgeden daha güncel kabul eder.

## Temel kurallar

- Geometri kaynağı flat `TriangleMesh` / DNA SoA'dır; facade `Triangle` listesine yazılmaz.
- Önce capability keşfi, sonra selection, preview, commit, validate ve diagnostics yapılır.
- Destructive operasyonlar undo transaction içinde çalıştırılır.
- GPU desteklenmiyorsa aynı parametrelerle CPU fallback kullanılır; sonuç sessizce değiştirilmez.
- Non-manifold, açık yüzey, degenerate yüz, UV seam ve attribute kaybı raporlanmadan işlem başarılı
  kabul edilmez.
- Büyük meshlerde preview quality kullanılabilir; display proxy topology sahibi değildir.

## Standart işlem akışı

1. `mesh.tools.list` ile uygun workspace/tool keşfedilir.
2. `mesh.tools.describe` ile selection domain, parametre, GPU/CPU yetenekleri ve undo bilgisi
   okunur.
3. `mesh.selection.query` ile mevcut seçim ve aktif element doğrulanır.
4. Gerekirse `mesh.selection.set` ile vertex/edge/face/poly seçimi yapılır.
5. `mesh.operation.preview` çağrılır; değişen element sayısı ve warnings incelenir.
6. Kullanıcı niyeti doğrulandıktan sonra `mesh.operation.commit` çağrılır.
7. `mesh.asset.validate` ve `mesh.operation.diagnostics` ile sonuç kontrol edilir.
8. Gerekirse `mesh.undo` ile tek transaction geri alınır.

## Workspace seçimi

| İhtiyaç | Workspace |
|---|---|
| Mevcut mesh topology düzenleme | `edit` |
| Profil çizip path boyunca üretme | `profile` + `curve` |
| Loft, sweep, patch veya trim surface | `surface` |
| Union, difference, intersection | `boolean` |
| Hole, winding, degenerate, remesh | `cleanup` |

Her mutation sonucu `ok`, `operation_id`, `revision`, `changed_counts`, `warnings`, `errors` ve
`undo_group` alanları okunmalıdır. `ok=true` olsa bile warnings varsa kullanıcıya bildirilir.
