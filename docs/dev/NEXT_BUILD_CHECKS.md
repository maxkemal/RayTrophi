# Bir sonraki build — Faz 3 BİTTİ: Assimp söküldü

> **Durum:** CANLI — **derlendi ve import yolları doğrulandı** (kullanıcı,
> 2026-09-06): “import yapılarını test ettim, herhangi bir eksiklik tespit
> etmedim”. Yani **§0, §1, §2 geçti.** ★ Aşağıdakiler import yolu DİŞINDA
> kaldığı için ayrıca bakılmadı — kapandı saymayın: **§3** varlık tarayıcısı,
> **§4** OptiX render, **§5** gömülü dokulu dosya, **§6** dll gerçekten düştü mü,
> **§7** arazi macro color + semantic map.

**Önceki parti doğrulandı.** Kullanıcı derledi ve **FBX, GLB, OBJ** üçünü de yeni
okuyucularla açtı: sorunsuz. Bu, Faz 3'ün üç artışını da kapattı, ve bu parti
son adımı yapıyor: **Assimp projeden tamamen çıkarıldı.**

Önceki glTF kontrolleri [arşivde](IMPORT_GLTF_CHECKS_ARCHIVE.md).
Devir notu ve tuzak listesi: [FAZ3_DEVIR_NOTU.md](FAZ3_DEVIR_NOTU.md).

## Ne yapıldı

| İş | Sonuç |
|---|---|
| `AssimpLoader.h` (3400 satır) + `AssimpLoader.cpp` | **silindi** |
| `assimp-vc143-mt.lib` / dll kopyası | vcxproj **ve** CMakeLists'ten söküldü |
| `Texture(const aiTexture*)` + `decode_raw` + `decode_compressed` | silindi — her okuyucu gömülü görüntüyü kendi çözüp byte buffer veriyor, o kurucu zaten var |
| `AssimpLoader::convertTrianglesToOptixData` (334 satır) | **taşındı** → `buildOptixMaterialTables()`, `OptixMaterialTables.h/.cpp`. İçinde tek satır Assimp yoktu |
| `AssetRegistry` FBX/OBJ metadata | Assimp yerine `probeModel()` |
| `GltfProbe` | → `ModelProbe` (kural 5: anlamı genişledi, adı değişti); `probeFbx` (ufbx) ve `probeObj` eklendi |
| `set_fbx_reader` / `set_obj_reader` + File menüsü | **kaldırıldı** — format başına tek okuyucu kaldı, seçenek ölü |
| `RtApiImport` / `RtIpcImport` / `RtPythonImport` / `ImportSettings*` | silindi, IPC 416 → **412 metot**, audit yeşil |

Manuel test; yeni test scripti yok.

---

## §0 — ★★★ Derleme: bu partinin EN OLASI hata sınıfı

★★★ **Beklenen hata türü tek bir şey: eksik include.** `AssimpLoader.h` yalnızca
bir yükleyici değildi; `Triangle.h`, `Camera.h`, `Material.h`, `MaterialManager.h`,
`Texture.h`, `Light.h`, `globals.h`, `EmbreeBVH.h`, `sbt_data.h` gibi başlıkları
da **onu include eden herkese geçişli olarak** taşıyordu. Include edenlerin
listesi 13'tü. Şimdi o başlık yok.

Include'unu değiştirdiğim dosyalar: `Renderer.h`, `scene_data.h`,
`AnimationController.h`, `ProjectManager.cpp`, `scene_ui.cpp`,
`scene_ui_animgraph.hpp`, `NodeHierarchy.cpp`, `OzzRuntime.cpp`,
`GltfDirectReader.cpp`, `UfbxReader.cpp`.

**Bozuksa ne demek:** `undefined identifier` / `incomplete type` alırsan bu odur.
**Çaresi:** o dosyaya eksik başlığı **doğrudan** ekle. Geri alma, ve "eskisi gibi
her şeyi taşıyan bir başlık" kurma — gizli bağımlılığı görünür yapmak bu işin
amacıydı.

★★ **Bu partide tarama iki kez dar kaldı, ikisi de kullanıcı build'inde veya
son kontrolde yakalandı — kayda geçiyor:**

1. `scene_ui_animgraph.hpp` silinmiş başlığı include ediyordu; ilk taramam
   yalnızca `*.h`/`*.cpp` bakıyordu, `.hpp` uzantısı kaçmıştı.
2. **`scene_ui.cpp` içinde İKİNCİ bir Assimp yolu vardı**
   (`computeAssetPreviewBounds`): önizleme sınır kutusu için elle yazılmış bir
   `aiNode` ağacı yürüyüşü. Kullanıcının derlemesinde `aiProcess_*`,
   `aiMatrix4x4`, `aiNode` hataları olarak çıktı. Kalibrasyon hatası bendeydi:
   `aiScene|aiMesh|aiMaterial|aiTexture` diye **isim listesi** taramıştım,
   `ai[A-Z]` **deseni** yerine — `aiProcess_*`, `aiNode` ve `aiMatrix4x4` o
   listede yoktu.

**Ders:** bir bağımlılığı sökerken **isim listesiyle değil desenle** tara
(`\bai[A-Z][A-Za-z0-9_]*\b`), ve uzantı listesine `.hpp/.inl/.cu/.cuh` ekle.
Son tarama ikisiyle de tekrarlandı ve temiz.

★ O ikinci yol **portlanmadı, silindi**: `probeModel(applyNodeTransforms=true)`
zaten tam olarak o kutuyu (sahne grafiğine yerleştirilmiş sınırlar) tanımlıyor.
Yani önizleme çerçeveleme artık FBX/OBJ'de **ilk kez** bu yoldan geçiyor — §3'ü
atlama.

## §1 — En hızlı bağımsız kontrol: menü ve script

**Yap:** File menüsüne bak. Sonra script'ten `scene.get_fbx_reader()` çağır.

**Ne görmen gerek:** **FBX Reader / OBJ Reader menüleri YOK.** Script metodu
`unknown method` diyor. `agent.discover` 412 metot raporluyor.

**Bozuksa ne demek:** menüler duruyorsa eski exe çalışıyordur — zaman damgasına bak.

## §2 — ★★ Üç formatın da hâlâ açılması

**Yap:** geçen partide açtığın **aynı** FBX, GLB ve OBJ dosyalarını tekrar aç.

**Ne görmen gerek:** geçen seferkiyle **birebir aynı** sonuç — obje sayısı,
konum, ölçek, dokular, skinli FBX'te animasyon.

★ Bu parti hiçbir okuma davranışını değiştirmedi; yalnızca **artık çağrılmayan**
kodu sildi. Bir fark görürsen o silme yanlış bir şeye dokunmuş demektir, ve en
olası aday `Texture.h`'den çıkan gömülü görüntü çözücüleridir — o durumda belirti
**gömülü dokulu bir dosyada doku gelmemesi** olur (harici dokular etkilenmez).

## §3 — Varlık tarayıcısı (metadata artık Assimp'siz)

**Yap:** varlık tarayıcısında **FBX ve OBJ** klasörlerine gir. Üçgen/mesh
sayılarına ve önizleme çerçevelemesine bak.

**Ne görmen gerek:** sayılar dolu ve makul; önizleme objeyi çerçeveliyor.
★ Tarama **belirgin biçimde hızlanmış** olmalı: eski yol her dosyanın bütün
buffer'larını çözüp `aiProcess_ImproveCacheLocality` (Tipsify yeniden sıralama)
çalıştırıp **bir sayı ve bir kutu** üretiyordu. Yeni yol ufbx'te gömülü
görüntüleri atlıyor, OBJ'de düz metin tarıyor.

**Bozuksa ne demek — belirtiden sebebe:**

| Belirti | Sebep |
|---|---|
| Bütün sayılar **0**, boyut yok | `probeModel` false döndü. Log'da `model probe failed` satırı olmalı — ★ probe **sessizce başka bir yola düşmüyor**, çünkü probe'un okuyamadığı dosyayı **importer da okuyamaz**; açılamayacak bir dosya için makul sayı göstermek tam da "panel yalan söylüyor" hatası |
| FBX sayıları doğru ama **tarama yavaş** | `opts.ignore_embedded` düşmüş; probe gömülü görüntüleri çözüyor demektir |
| OBJ üçgen sayısı **import'takinden farklı** | Probe da okuyucu da n-gon'u fan ile bölüyor (köşe−2). İkisi ayrılırsa tarayıcı ile sahne farklı sayı gösterir |
| Önizleme çerçevesi FBX/OBJ'de **bozuk** | `probeModel(..., applyNodeTransforms=true)`. Artık glTF dışı formatlarda da çalışıyor — eskiden yalnızca glTF'ti, bu yeni bir yetenek, yani ilk kez burada test ediliyor |

## §4 — OptiX yolu (taşınan 334 satır)

**Yap:** OptiX/CUDA backend'i ile bir sahne render et. Mümkünse **flat SoA**
(proje dosyasından açılmış) bir sahne.

**Ne görmen gerek:** normal render.

★★ **Bozuksa ne demek:** `buildOptixMaterialTables` yanlış taşınmış olabilir.
Fonksiyonun kritik özelliği adında değil: **üçgen listesi boş olsa bile
çağrılmak zorunda.** Tablolar `MaterialManager`'dan geliyor; geometri çıkarımı
`nTris>0` ile kapılı. Flat bir sahne buraya **boş** liste ile gelir ve tabloları
yine de ister — çağrılmazsa materyal buffer'ı null olur ve `optixLaunch`
**CUDA 700 illegal memory access** ile ölür. Yeni başlıkta bu yazılı; "boş liste
için atlayalım" diye optimize etme.

## §5 — Gömülü dokulu dosya (silinen çözücüler)

**Yap:** dokuları **içine gömülü** bir GLB ve mümkünse bir FBX aç.

**Ne görmen gerek:** dokular geliyor.

**Bozuksa ne demek:** `Texture.h`'den `decode_raw`/`decode_compressed` silindi.
Bunlar Assimp'in `aiTexture`'ından çözüyordu. Okuyucular gömülü içeriği kendileri
alıp `Texture(std::vector<char>, ...)` kurucusuna veriyor — cgltf buffer view,
ufbx `content`, OBJ'de gömülü görüntü yok. Doku gelmiyorsa o yol kopmuştur.
★ Harici dosya dokuları bu maddeden **etkilenmez**, o yüzden §2'de doku görmen
bu maddeyi geçtiğin anlamına gelmez.

## §6 — Bağımlılığın gerçekten düştüğü

**Yap:** çıktı klasöründe `assimp-vc143-mt.dll` var mı bak. Temiz bir klasöre
build alıp çalıştır.

**Ne görmen gerek:** dll **kopyalanmıyor** ve uygulama onsuz açılıyor.

**Bozuksa ne demek:** hâlâ bir yerde linkleniyor. vcxproj'da iki configuration
ve CMakeLists'te dört satır temizlendi; başka bir yer kalmış olabilir.

## §7 — Arazi doku haritaları (silinen kurucunun son kullanıcıları)

**Yap:** iki şeyi de çalıştır:
1. arazi SatMap / macro color düğümü (arazi renklendirmesi) → `MacroColorMap`
2. arazi **semantic map** üretimi ve PNG'den yüklenmesi → `TerrainSemanticMap`

**Ne görmen gerek:** ikisi de eskisi gibi çalışıyor, ve log'da artık
**`Texture pointer null, skip` uyarısı YOK.**

★ **Neden burada:** iki çağıran boş bir doku elde etmek için
`Texture(nullptr, TextureType::Albedo, "MacroColorMap")` çağırıyordu — yani
**Assimp kurucusunu null bir `aiTexture` ile**. Kurucu silinince derleme
`std::construct_at` / `C2665` ile patladı (hata `xutility`'de görünür, çünkü
`make_shared` oradan kurar; gerçek yer çağıran dosyadır).

Zaten var olan `Texture(name, w, h, type)` kurucusuna çevrildi. **İki gözle
görülür fark:**

1. Her arazi boyamasında düşen `Texture pointer null, skip` uyarısı kalktı.
2. ★ Doku artık gerçekten **`"MacroColorMap"` adını taşıyor**. Eski kurucu
   null kontrolünden `name` atamasından **ÖNCE** dönüyordu, yani o ad sessizce
   çöpe gidiyordu. Ad'a göre arayan bir şey varsa ilk kez bulacak.

★★ **Tarama dersi, üçüncü kez:** bu maddedeki dört çağırandan ikisi
(`TerrainSemanticMap.cpp`) ilk taramamda **kaçtı**, çünkü `nullptr` çağrının
**bir sonraki satırındaydı** ve satır bazlı grep çok satırlı çağrıyı görmez.
Bu partide tarama üç kez zayıf kaldı: (1) uzantı listesi `.hpp` içermiyordu,
(2) desen yerine isim listesi kullandım, (3) satır bazlı aradım.
Son araç (`scratchpad/scan_removed.py`) üçünü de karşılıyor: yorumları söküp
**dosyanın tamamında desenle** arıyor. ★ Bir tarama aracı kendi
dokümantasyonunu bulgu diye raporluyorsa, o araç bozuktur.

★ Ayrıca not: `TerrainSatMapNodes.cpp` **iki yerde** var (`src/Physics/` 1802
satır, `src/Scene/` 178 satır) ve vcxproj'da yalnızca Physics olanı kayıtlı —
ama CMake recursive glob kullandığı için diğerini de derler. İkisini de
düzelttim; hangisinin ölü olduğuna bakmak ayrı bir iş (kural 5).

---

---

## Bu partide KAPANMAYAN

- **Rule 1 borcu:** `scene.import_model` hâlâ bir şey döndürmüyor. Okuyucular
  `ImportStats`'ı ölçüyor (parse / materyal / geometri / animasyon ayrı ayrı),
  ama script tarafına açılmadı. ★ Import IPC dosyaları bu partide silindi;
  bu iş yeni bir `RtIpcImport` ile geri gelecek, ve doğal yeri orası.
- glTF açık maddeleri: `KHR_texture_transform` okunmuyor; 7 paylaşılan mesh için
  gerçek instancing (~4.39M kopya üçgen); LOD0-only import seçeneği.
- Yazıcının non-conformant skin sözleşmesi (okuyucu `generator` sniff'i ile
  telafi ediyor) — kendi export'umuz Blender'da yanlış açılıyor.
- UV düzenlemesi Vulkan RT'de tam rebuild tetikliyor.
- `vcpkg/ports/assimp` duruyor: o vcpkg'nin **kendi port kayıt ağacı**, bizim
  manifestimiz değil. Bize ait bir bağımlılık değil, dokunulmadı.
