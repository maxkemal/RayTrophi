# Siradaki build kontrolleri — TERRAIN FIRCA + SECIM KIMLIGI

> **Durum:** REFERANS — 2026-08-29, dokuzuncu parti. **Hepsi canli dogrulandi**,
> asagisi bundan sonraki degisiklikler icin REGRESYON listesidir.
>
> Sekizinci parti: terrain sculpt clamp'i (dogrulandi). Dokuzuncu parti:
> (1) terrain splat paint fircasi cikmiyordu — iki kok, **dogrulandi**,
> (2) sag firca dock'u sol panelin kopyasiydi — kaldirildi, **dogrulandi**,
> (3) "kup secili ama terrain tasiniyor" — iki yapisal kok, **dogrulandi**
> (dogru obje ekleniyor/seciliyor, terrain tasinmiyor).

Bu partide degisen dosyalar:

- `UI/scene_ui.cpp` — ★★★ Terrain Graph paneli **acik olmasi** degil **odakta
  olmasi** fircayi kapatiyor (iki yerde); `allow_paint_bridge` eski haline dondu
- `UI/scene_ui_modifiers.cpp` — terrain paint dock'u kaldirildi;
  `syncPaintBrushToTerrain` artik **radius kopyalamiyor**; terrain paint paneli
  `terrain_brush.radius`'u (metre) duzenliyor
- `Physics/TerrainManager.cpp` — ★★★ `registerTerrainMeshOnce` zaten kayitliysa
  **yerinden oynatmiyor**; her world.objects mutasyonunda `g_mesh_cache_dirty`
- `UI/SceneSelection.cpp` — ★★★ `object_index` esitligi artik tek basina KIMLIK
  degil; spline oğeleri icin null==null tuzagi kapatildi
- `scripts/terrain_scene_selection_check.py` (+ `x64/Release/scripts/`) — yeni

Yeni `.cpp` yok, `.vcxproj` degismedi. IPC yuzeyi degismedi
(`python scripts/gen_ipc_descriptors.py --check` yine `up to date` demeli).

---

## 0. Derleme

**Ne gormen gerek:** temiz build.
**Bozuksa ne demek:** `TerrainManager.cpp`'de `g_mesh_cache_dirty` icin
"declared but not defined" (C7631) cikarsa `extern` yine `namespace {}` icine
kaymis demektir — dosya kapsaminda durmali (kod icinde not var).

## 1. ★★★ Terrain splat paint fircasi (asil sikayet)

Terrain sec → **Paint** sekmesi → panel `Target: <terrain>` demeli.

**Ne gormen gerek:** viewport'ta **sari** firca cemberi; sol tik surukleyince
splat degisiyor. **Sag tarafta ek bir firca paneli CIKMAMALI.**

**Bozuksa ne demek — sirayla:**
1. Cember hic yoksa: `terrain_brush.enabled` yine her kare kapaniyordur.
   ★ Kok buydu: `Terrain Graph` panelini **acik** birakmak yetiyordu; o blok
   `handleTerrainBrush`'tan SONRA calisip bayragi siliyordu. Sculpt etkilenmiyordu
   cunku o `terrain_sculpt_proxy_active` uzerinden geciyor — "sculpt calisiyor,
   paint calismiyor" asimetrisinin tek sebebi buydu.
2. Cember **cok kucuk / nokta gibi**yse: yaricap birimi yine mesh fircasindan
   kopyalaniyordur. `Paint::BrushSettings::radius` obje birimidir (presetler
   0.09–0.30), `terrain_brush.radius` **metredir**; 1 km'lik arazide 0.25 m
   fircadir ve "firca cikmiyor" diye gorunur.
3. Panelde `Paint target: none` yaziyorsa secim terrain'i baglamiyordur
   (`terrain_brush.active_terrain_id == -1`).

★ **En sinsi hali:** cember var, boyuyor gibi ama splat degismiyor — o zaman
`radius` degil `strength`/kanal bakilir. Ölçum:

```powershell
Invoke-RtIpc terrain.paint_splat @{ name='Terrain_1'; dabs=@(@(0,0),@(20,0)); channel=1; radius=50 }
```
`coverage_delta > 0` olmali.

## 2. Regresyon: mesh boyama ve mesh firca dock'u

Bir kup sec → Paint → boya.

**Ne gormen gerek:** sag firca dock'u ESKISI GIBI aciliyor (mesh paint), aletler
ve katmanlar calisiyor.
**Bozuksa ne demek:** dock kapisi (`shouldShowPaintBrushDock`) fazla daraltilmis.

★ Ayrica: mesh boyadiktan sonra terrain'e gec, sonra tekrar mesh'e don.
**Ne gormen gerek:** her iki fircanin yaricapi kendi olceginde kaliyor
(mesh ~0.25, terrain ~5–50 m). Birbirine bulasiyorsa radius yine paylasiliyor.

## 3. ★★★ "Kup secili, terrain tasiniyor" — once OLC

```powershell
python scripts/terrain_scene_selection_check.py
```

**Ne gormen gerek:** hepsi PASS. Ozellikle `the TERRAIN did NOT move`.
**Bozuksa ne demek:** ariza **kimlik hattinda** (isim/indeks/transform handle) —
UI'de degil. O zaman `select.list` ciktisi hangi nesnenin gercekten secili
oldugunu soyler.
**Hepsi PASS ise:** deger katmani temiz; kalan ariza **hiyerarsi tiklamasi →
secim → gizmo** zincirindedir (scene_ui_hierarchy.cpp / scene_ui_gizmos.cpp).

Sonra UI'da tekrarla: terrain-only sahne → `Add > Mesh > Cube` → hiyerarside
kupe tikla → tasi.

**Ne gormen gerek:** kup tasiniyor, terrain duruyor; hiyerarside YALNIZ kup
satiri secili gorunuyor.
**Bozuksa ne demek:**
- Iki satir birden secili gorunuyorsa `object_index` esitligi hala kimlik
  sayiliyordur (`SceneSelection::isSelected`).
- Kup satiri secili ama terrain tasiniyorsa gizmo'nun tuttugu
  `sel.selected.object->getTransformHandle()` yanlis nesnenin handle'idir —
  o zaman kupun flat mesh'inin `transform`'una bak (`scene.get_transform`).

★ Terrain'in mesh cozunurlugunu degistirip (Terrain > Resolution > Apply)
tekrar dene: eski kod bu anda `world.objects`'i yeniden siraliyordu ve
UI'nin cache'ledigi her slot indeksi kayiyordu.

## 4. Terrain mesh yeniden kaydi hala tek kopya

Terrain graph'i birkac kez Evaluate et, sonra:

```powershell
Invoke-RtIpc scene.list_objects | Select-String Chunk
```

**Ne gormen gerek:** `<terrain>_Chunk` **bir kez** listeleniyor.
**Bozuksa ne demek:** `registerTerrainMeshOnce`'in "zaten kayitli => dokunma"
kisayolu cift kayit birakiyor demektir (eski kod her seferinde sil+ekle yapip
bunu maskeliyordu).

## 5. Sculpt regresyonu (gecen parti)

```powershell
python scripts/terrain_brush_check.py
```
**Ne gormen gerek:** hepsi PASS (`raise RAISES the ground it touches`).

## 6. ✔ Kapandi: "terrain-only sahnede Add gorunmuyor"

Ayri bir cizim arizasi degildi. Nesne sahnedeydi; **kimligi** yanlis nesneye
aitti (bkz. §3). Bir parti raster hattinda bosa arandi — ders: "gorunmuyor"
raporunda once nesnenin KIMLIGINI dogrula (outliner + secim + transform),
sonra render hattina in.
