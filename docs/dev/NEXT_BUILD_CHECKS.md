# Sıradaki build kontrolleri

> **Durum:** CANLI — her partide üzerine yazılır. Son güncelleme: 2026-09-17 (14. parti).

## Silme kaynağı BULUNDU: Ctrl+Z tuş işleyicisi

Sayaçlar tam olarak doğru yere götürdü. `sculpt.cache_wipe.rebuildMeshCache_ran`
ölçümde **hiç görünmedi** → `ensureEditableMeshCache` içindeki yol koşmamış.
`rebuildMeshCache` 6 kez koşmuş, 3'ünde korunacak cache vardı, `restore_refused`
toplamı **0** → koruma/geri yükleme **çalışıyordu**. SceneLog da
`object(''->'Plane_1')` diyor, yani cache gerçekten sıfırlanmış. Log sırası:

```
History: Undo - Sculpt Plane_1
Undo: Sculpt Plane_1
[ensureEditableMeshCache] rebuild reason: object(''->'Plane_1') ...
```

Kaynak `scene_ui.cpp`'deki Ctrl+Z işleyicisi:

```cpp
history.undo(ctx);
rebuildMeshCache(ctx.scene.world.objects);
mesh_overlay_cache = MeshOverlayCache{};
editable_mesh_cache = EditableMeshCache{};   // ← koşulsuz siliyor
```

Komut ne yaptıysa yapsın cache atılıyor — ve `rebuildMeshCache`'in cache'i
özenle saklayıp geri yüklemesini de bir sonraki satır iptal ediyor. Koruma
"çalışıyor" ölçülüyordu çünkü gerçekten çalışıyordu; sonra çöpe atılıyordu.

**Düzeltme:** `SceneCommand::handlesUiCacheSync()` (varsayılan **false**).
`FlatSculptEditCommand` bunu `apply()` içinde gerçekten ne olduğuna göre
döndürüyor: `adoptExternalFlatSoaEdit` başardıysa true, rebuild yedeğine
düştüyse false. `SceneHistory::undo/redo` bunu çağırana bildiriyor, tuş
işleyicisi de yalnızca false ise eski toptan geçersizleştirmeyi yapıyor.
Denetlenmemiş her komut eskisi gibi davranıyor.

## CPU seçiliyken Solid'de maliyet artışı — ÖLÇÜLDÜ

| bölüm | çağrı | çağrı başına |
|---|---|---|
| `loop.viewport_render_cpu` | 100 | **132 ms** |
| `loop.viewport_render` (GPU) | 815 | 2,8 ms |

47 kat. CPU render backend'i seçiliyken her sculpt darbesi `start_render`
kurduğu için CPU yolu görüntüyü baştan izliyor — Solid gösterilirken o sonuç
ekrana **gitmiyor** bile. Yani bu harcanan iş.

**Bu partide düzeltilmedi.** Doğru kapı "viewport Solid gösterirken CPU izleme
tetiklenmesin" ama `start_render` çok yerden kuruluyor ve CPU sonucunun render
önizleme paneli gibi başka tüketicileri olabilir; undo düzeltmesiyle aynı
build'de test edilmesini istemiyorum.

Ayrıca not: `accel.cpu.bvh_refit` 43 çağrı / **0,66 ms toplam** — 12. partinin
CPU BVH düzeltmesi sağlam, artış oradan gelmiyor.

Değişen dosyalar: `include/SceneCommand.h`, `src/Utils/SceneCommand.cpp`,
`include/SceneHistory.h`, `src/Utils/SceneHistory.cpp`, `include/scene_ui.h`,
`src/UI/scene_ui.cpp`.

---

## 1. Derleme

**Ne görmen gerek:** temiz derleme.
**Bozuksa:** `invalidateUiCachesAfterHistoryStep` bulunamadı → `scene_ui.h`
bildirimi sınıf gövdesine girmemiş.

## 2. Ctrl+Z hızı

Sculpt paneline gir, uzun bir darbe, Ctrl+Z.
**Ne görmen gerek:** bekleme yok; bir darbe atmakla karşılaştırılabilir
(komutun kendisi ~14 ms ölçülüyor).
```powershell
Invoke-RtIpc perf.list @{} | Where-Object { $_.name -like 'ensureEditableMeshCache*' -or $_.name -like 'sculpt.cache_rebuild.*' -or $_.name -like 'sculpt.undo.*' } | Format-Table name,count,total_ms,max_ms -AutoSize
```
**Ne görmen gerek:** `ensureEditableMeshCache.build[sculpt]` sayısı
**Ctrl+Z ile artmıyor** (panele ilk girişteki 1 kurulumda kalıyor).
**Bozuksa ne demek:** hâlâ artıyorsa cache'i başka bir yer siliyor; SceneLog'daki
`rebuild reason:` satırı yine kimin ne değiştirdiğini yazacak.

## 3. ★★★ UNDO DOĞRULUĞU — bu partinin riski burada

Artık undo'dan sonra cache **silinmiyor**, yani doğruluk tamamen
`adoptExternalFlatSoaEdit`'in yeniden tohumlamasına bağlı. Sıra:

1. Darbe vur → **Ctrl+Z** → **yeni bir darbe vur**: eski şekil geri gelmemeli.
2. Ctrl+Z → **Ctrl+Y** (redo) → şekil geri gelmeli, sonra yine Ctrl+Z.
3. Ctrl+Z'den sonra **aynı yere** darbe vur: fırça tutmalı (PBVH sınırları).
4. Undo sonrası **darbenin dış kenarına** bak: ince bir iz kalmamalı.

**Bunlar bozuksa** eskiden toptan silme hatayı örtüyordu — şimdi örtmüyor. Hangi
adımda bozulduğunu yaz, `adoptExternalFlatSoaEdit` içindeki tohumlamayı ona göre
düzelteceğim. Tuş işleyicisini geri almak çözüm değil, sadece 1 saniyeyi geri
getirir.

## 4. Sculpt DIŞI undo geriye gitmedi mi

Bir objeyi taşı → Ctrl+Z. Bir obje sil → Ctrl+Z. Materyal değiştir → Ctrl+Z.
**Ne görmen gerek:** hepsi eskisi gibi. Bu komutlar `handlesUiCacheSync()`
varsayılanını (false) kullandığı için eski toptan geçersizleştirmeden geçiyor.
**Bozuksa ne demek:** bir komut yanlışlıkla true dönüyor.

## 5. Kalan sıra

- **CPU seçiliyken Solid'de 132 ms'lik boşa CPU izleme** (yukarıda). Sıradaki
  en büyük kalem.
- `sculpt.enter.pbvh` max 420 ms — panel girişinin ikinci yarısı.
- `raster.solid.soa_refit.diff` — hâlâ tüm mesh'i tarıyor.
- Edit overlay sculpt sırasında da çiziliyor (`edit_mode=true`); ölçümde büyük
  çıkmadı ama `ui.editable_mesh_overlay` izlenmeli.
- Undo geçmişi belleği: vuruş başına ~186 bin vertex × 52 bayt ≈ 9,7 MB; sınır
  yok.
