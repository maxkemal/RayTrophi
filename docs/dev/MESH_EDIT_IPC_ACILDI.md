# Poligon edit operatörleri IPC'ye açıldı

> **Durum:** AKTİF — kod yazıldı, **runtime doğrulanmadı**. Kabul testi:
> `docs/dev/NEXT_BUILD_CHECKS.md` + `scripts/ipc/Probe-MeshEdit.ps1`.

**Tarih:** 2026-09-20

---

## Belirti

Bir sandalye modellemesi denenirken (referans görselden, tamamen IPC üzerinden)
uygulama tanınabilir bir sandalye üretti ama yalnızca **primitif + transform**
ile. Poligon düzenleme hiç kullanılamadı.

`mesh.tools.list` şunu döndürüyordu:

```json
{ "id": "edit.extrude", "availability": "implemented",
  "scriptable": true, "ipc_exposed": true }
```

Beş araç için de aynısı. Ama bu araçları **çağıracak hiçbir metot yoktu**.
Dispatch'te yalnızca `mesh.tools.list/describe`, `mesh.operation.plan/self_test`
ve `mesh.profile.*` vardı.

---

## ★★★ Kök: katalog kendi NİYETİNİ ölçüyordu

`ipc_exposed: true` bir **açıklama** değil, bir **iddia**ydı — ve hiçbir şey
onu doğrulamıyordu. Araç kaydı `MeshTool.cpp`'de elle yazılmıştı; oraya
`true` yazmak bir metodun var olmasını sağlamıyor.

Bu, bu deponun tekrar eden hata sınıfının aynısıdır:

> **Ölçü aleti doluluk raporluyordu.**

Elle tutulan IPC kataloğu 299 boş satır çıkmıştı ve o yüzden üretilir hale
getirildi. Araç kataloğu aynı hatayı, ters yönden bir kez daha yaptı.

### Ters yönü de vardı

`edit.edge_bevel` katalogda **`Planned`** kayıtlıydı. Oysa
`SceneUI::bevelSelectedEdges` yaklaşık 200 satırlık dolu bir implementasyondu.
`Planned` araçlar varsayılan listelemede **gizlenir**, yani edit workspace'inin
tek gerçek yuvarlatma operatörü her çağırana görünmezdi.

Yani katalog dört aracı **olduğundan fazla**, bir aracı **olduğundan az**
gösteriyordu. İkisinin ortak nedeni tek: availability alanı koda değil,
yazarın o günkü niyetine bakıyordu.

---

## ★★ İkinci kök: operatör var, seçim yok

Sekiz operatörün tamamı `editable_mesh_cache.selection`'ı tüketir. Seçim
API'si olmadan operatörleri açmak, **hiçbir şeye bağlı olmayan bir kol**
takmak olurdu: her çağrı "seçim yok" diye dönerdi ve dıştan bakınca **bozuk
operatör** gibi görünürdü, eksik yetenek gibi değil.

Bu yüzden seçim bu partinin parçası, altındaki bir detay değil.

### Ve id listesi tek başına yetmez

Bir ajan yüz id'si bilmez. "Üstteki yüz" ya da "şu kutunun içindekiler" bilir.
`mesh.edit.select_by_normal` ve `mesh.edit.select_by_box` bu yüzden var:
onlarsız id listesi **pratikte ulaşılamaz** ve operatörler teoride kalırdı.

---

## Ne eklendi

| Katman | Dosya |
|---|---|
| Çekirdek API | `Api/RtApi.h` + `Api/RtApiMeshEdit.cpp` |
| IPC dispatch | `Api/RtIpcMeshEdit.cpp` / `.h` (+ `RtIpc.cpp`'de tek satır) |
| Python binding | `Api/RtPythonMeshEdit.cpp` |
| Yetki | `Api/RtIpcSecurity.cpp` + `scripts/audit_ipc_capabilities.py` aynası |
| Ajan tarifi | `scripts/ipc_descriptor_overlay.json` (15 giriş) |

**Metotlar:** `mesh.edit.{begin,get_state,select,select_by_normal,select_by_box,
clear_selection,get_selection}` ve `mesh.{extrude,inset,bevel,loop_cut,
dissolve_edges,dissolve_vertices,merge_vertices,weld_vertices}`.

Araç kataloğu da gerçeğe çekildi: `edit.edge_bevel` → `Implemented`, ve
`edit.{dissolve_vertices,merge_vertices,weld_vertices}` kaydedildi.

### Yanında: `scene.object_info` artık sınır kutusu döndürüyor

Ajan parçaların birbirine **değip değmediğini ölçemiyordu** — yalnızca ekran
görüntüsünden göz kararı tahmin ediyordu. Üçgen/vertex sayısı "ne kadar ağır"
sorusunu cevaplar, "nerede" sorusunu asla.

İki ek karar:

- `has_bounds=false` iken sınır alanları **yok**, sıfır değil. Sıfırlanmış bir
  kutu "orijinde bir nokta" diye okunur — bir ölçüm olmayan şey ölçüm gibi
  görünmemeli.
- `meshes` alanı eklendi. `getObjectInfo` eskiden **ilk eşleşen** mesh'i
  okuyordu; çok materyalli import TEK objedir ama BİRKAÇ flat mesh'tir, yani
  sayılar objenin bir dilimini bütünüymüş gibi raporluyordu.

---

## ★ Kapatılmayan delik

`editable_mesh_cache` (topoloji + seçim) hâlâ `SceneUI` üzerinde yaşıyor.
Bu, kuralın doğrudan ihlali:

> UI kendine ait durum TUTMAZ — bir paneli scriptleme isteği duyduğun an, o
> panel çekirdeğe ait bir durumu tutuyordur.

Bu parti cache'i taşımadı. Yaptığı şey durumu **dışarıdan okunur ve yazılır**
kılmak: `mesh.edit.get_state` artık çekirdeğin gördüğü sayıları verir, yani
çekirdek ile panel bundan sonra **sessizce** ayrışamaz. Taşıma ayrı bir iştir.

---

## Bağlantılı

- `docs/dev/NEXT_BUILD_CHECKS.md` — kabul testi, sıralı
- `scripts/ipc/Probe-MeshEdit.ps1` — ölçen probe (sayı karşılaştırır, ekran
  görüntüsüne bakmaz)
