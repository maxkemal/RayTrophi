# Import/export göçü — devir notu (Faz 3'ü devralan ajan için)

> **Durum:** ARŞİV — Faz 3 bitti, Assimp 2026-09-06'da söküldü. Tuzak
> listesi (§3) hâlâ geçerli referans: yeni bir okuyucu yazan herkes okusun. — Faz 2 doğrulandı; ilk ufbx artışı kullanıcı tarafından derlendi, Twin Towers import logu alındı. Görsel kabul açık. 2026-09-05.

**Güncel devam noktası:** [UFBX_INCREMENT1.md](UFBX_INCREMENT1.md) ve
[NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md). Statik FBX için opt-in ufbx,
ortak dispatch, UI/Python/IPC okuyucu seçimi ve build kayıtları eklendi.
Skinning/animasyon, OBJ ve Assimp'in sökülmesi henüz yapılmadı.

**Kullanıcı doğrulaması:** önceki §7, §8 ve §9(a) kapalı. §9(b) çift kayıt veya
yanlış sayım değil: 977 primitif / 8.778.982 üçgen düğüm toplamı doğru.
Yalnızca 7 kaynak mesh tekrar ediyor (biri 315 kez); yaklaşık 4.39M üçgenlik
kopya veri var. Eşik ve açık import seçeneğiyle InstanceGroup dönüşümü ayrı
kalıyor. Önceki kontrol maddeleri [arşivde](IMPORT_GLTF_CHECKS_ARCHIVE.md).

Bu not, glTF/GLB'yi Assimp'ten söken üç iş partisinin **devri** için yazıldı.
Kod okunarak öğrenilemeyecek şeyleri içerir: hangi iddia doğrulandı, hangisi
sadece çözümlendi, ve bu alanda hangi tuzaklara **zaten** düşüldü.

Önce `CLAUDE.md`'yi oku — bu depodaki kurallar tercih değil, her biri bir kez
pahalıya öğrenildi. Bu not onların yerine geçmez.

---

## 0. §7 doğrulandı — büyük Unreal GLB açılıyor

7. partinin düzeltmesi (`cgltf_validate` artık kapı değil) **kullanıcı tarafından
doğrulandı (2026-09-05):** "açıldı test ettim sorunsuz geçti". Önceki
derlenmemiş/test edilmemiş durumu kapandı. Bu sonuç aşağıdaki varlığın testidir;
bütün spec dışı dosyalar için güvenlik kanıtı değildir.

Tamamlanan kontrol: `docs/dev/IMPORT_GLTF_CHECKS_ARCHIVE.md` **§7**. Test dosyası
kullanıcıda: `E:\3d model_arsiv\fab_3d_models\village_house_and_barn_glb.glb`
(627 MB, Unreal Engine 5.2.1, 745 görüntü, 1140 materyal, `TEXCOORD_1` taşıyor).
Bu tek dosya aynı anda §2 (ekstra UV setleri), §5 (paralel doku prefetch'i) ve
§7'yi (spec dışı dosya) gerçek ölçekte sınıyor.

★ Bu değişiklik artık **spec dışı dosyaları okumaya çalışıyor**. Çökme olursa
`validateReadSafety()` eksik kalmış demektir; bu ilk şüphelin olsun.

---

## 1. Nerede kaldık

**Faz 2 KAPANDI ve kullanıcı tarafından doğrulandı.** glTF/GLB hem okuma hem
yazma tarafında Assimp'siz, ve **tur kapanıyor** — kendi export'umuz geri
açılıyor:

| | Durum |
|---|---|
| Skinli animasyon, doku yönü (V flip), fotometrik ışık (683 lm/W), ekstra UV setleri | ✅ doğrulandı |
| `EXT_mesh_gpu_instancing` scatter: doğru katman sayısı, doğru konum ve ölçek | ✅ doğrulandı |
| Çok materyalli obje **tek obje** kalıyor (okuyucu + yazıcı) | ✅ doğrulandı |
| Bitki kütüphanesi + animasyon klibi doğrudan okuyucudan (`Import/ModelImport.h`) | ✅ doğrulandı |
| Import hızı — kullanıcı: *"Assimp paralel yapıdan bile çok hızlı"* | ✅ doğrulandı |
| `cgltf_validate` artık kapı değil (Unreal'in boş primitifleri) | ✅ kullanıcı testi geçti (2026-09-05) |

**Faz 3 ilk artış DERLENDİ; ufbx importu logla doğrulandı.** Twin Towers FBX:
776 mesh / 187.404 üçgen / 13 materyal, toplam 52,092 ms. Doku sayısı 0;
görsel kabul ve diğer testler açık. `external/ufbx/ufbx.c`,
`UfbxVendor.cpp` üzerinden tek kez derlemeye bağlandı. `UfbxReader` statik FBX
geometri/materyal/hiyerarşisini ImportedModel'e üretir. File → FBX Reader,
Python ve IPC aynı oturum ayarını kullanır; varsayılan Assimp. Destek kapsamı
ve uyarılı eksikler [ilk artış notunda](UFBX_INCREMENT1.md).

### Assimp'in kalan yüzeyi — üç yer

| Yer | Ne yapıyor |
|---|---|
| `source/include/AssimpLoader.h` (3430 satır) | asıl okuyucu; artık **yalnızca FBX + OBJ** |
| `source/include/Texture.h` | `Texture(const aiTexture*)` kurucusu; başka çağıranı yok, loader ile ölür |
| `source/src/Scene/AssetRegistry.cpp` | FBX/OBJ metadata için kendi `Assimp::Importer`'ı (glTF dalı `probeGltf`'e geçti) |

★ `AssimpLoader.h` hâlâ **ölü glTF kodu** taşıyor (`isRayTrophiGltfFile`,
`loadGltfLightMetadataMap`, `isGLTF`). glTF oraya artık hiç uğramıyor. Kural 5
gereği sökülmeli — ucuz ve bağımsız bir temizlik, Faz 3'ün iyi bir §0'ı.

---

## 2. Faz 3 planı (üç artış, her biri ayrı derlenip test edilir)

1. **ufbx okuyucu: geometri + materyal + düğüm hiyerarşisi.** Hedef şekli
   `Import/ImportedModel.h` — glTF okuyucusuyla **aynı sözleşme**, yeni bir tip
   uydurma. Bu aşamada `AssimpLoader` FBX yolu **durur**, ufbx opt-in olur.
   (Faz 2'de doğrulanmış desen: önce yan yana, sonra eskisini sök.)
2. **Skinning + animasyon.** ✅✅ **DOĞRULANDI (2026-09-06).** Kullanıcı
   skinli/animasyonlu FBX'i **hem Assimp hem ufbx** ile açtı: fark yok,
   animasyonlar doğru oynadı. ★ Örneklem küçüktü (az sayıda skinli FBX), yani
   "yol doğru davranıyor" kaydı — "skinning bitti" değil. Yazılanlar:
   `buildSkeleton` (bone kapanışı + `geometry_to_bone` offset'leri), per-corner
   ağırlıklar (`vertex_indices` üzerinden), `ufbx_bake_anim()` ile baked klipler,
   skinli mesh base'i **birim**. Kontrol listesi §7. Özgün plan notu: Kullanıcının FBX kullanımı ağırlıklı olarak
   **skinli karakter**, yani bu ertelenebilir bir ekstra değil, işin yarısı.
   Referans: `GltfDirectReader::buildSkins()` — iskelet **kapanışını** kurar
   (eklemler + her atası), `AssimpLoader::buildBoneData` ile aynı şekil.
3. **Sökme.** ✅✅ **TAMAMLANDI (2026-09-06), derlenmedi.**
   FBX/GLB/OBJ üçü de yeni okuyucularla kullanıcı tarafından doğrulandıktan
   sonra Assimp tamamen çıkarıldı: `AssimpLoader.h`/`.cpp`, `aiTexture`
   kurucusu ve gömülü görüntü çözücüleri, `assimp-vc143-mt.lib` (vcxproj +
   CMake), ve geçici okuyucu seçenekleri. `convertTrianglesToOptixData`
   **taşındı** (`buildOptixMaterialTables`), silinmedi — içinde Assimp yoktu.
   `AssetRegistry` artık `probeModel()` kullanıyor. Detay: NEXT_BUILD_CHECKS.

   ★ Geçmiş kayıt (parti içindeyken yazılan):
   - **OBJ/MTL okuyucusu yazıldı** (`Import/ObjReader.*`), `File > OBJ Reader`
     ile opt-in. Vendor kitaplık YOK: OBJ satır tabanlı metin, çözdüren ~300
     satır; tinyobjloader eklemek kendi sözleşmesi + dönüşüm katmanı demekti.
   - **`AnimationData` + `BoneData` `AssimpLoader.h`'den çıkarıldı** →
     `Animation/AnimationData.h`. ★★★ **Aslında sökmenin ÖN KOŞULUYDU:** iki
     çekirdek tip, `<assimp/scene.h>` ile açılan 3400 satırlık başlıkta
     yaşıyordu, yani Assimp'i **değiştirmek için yazılan okuyucular bile**
     Assimp'i include etmek zorundaydı. `MeshInstance` taşınmadı, **silindi**:
     depoda tek bir okuyucusu yoktu (kural 5).
     Include edenler 13 → 7'ye düştü.
   - **Kalan:** `Renderer`'daki iki canlı Assimp çağrısı
     (`clearTextureCache`, `convertTrianglesToOptixData`), AssetRegistry FBX/OBJ
     metadata dalı, sonra `AssimpLoader.h` + `Texture.h`'daki `aiTexture`
     kurucusu + `assimp-vc143-mt.lib`.

Dispatch tek yerde: **`rtimport::loadModel()`** (`source/src/Import/ModelImport.cpp`).
ufbx eklemek orada tek bir dal. Yeni çağıran ekleme — mevcutları oraya bağla.

---

## 3. ★★★ Bu alanda ZATEN düşülmüş tuzaklar

Hepsi bu üç partide gerçekten oldu. Tekrar etme.

### 3.1 `nodeName` bir KİMLİK değil, GRUPLAMA anahtarıdır
Çok materyalli import **TEK mantıksal objedir**; bir düğümün bütün
`TriangleMesh` kardeşleri **aynı** `nodeName`'i paylaşır. Okuyucuya `_primN` eki
koymak scatter'ı, seçimi ve proje kaydını sessizce böldü. Benzersizlik
**düğümler arasında** gerekir (`resolveUniqueMeshNodeName`), düğüm içinde değil.
Mesh başına kimlik isteyen her şey **işaretçiyle** anahtarlıyor
(`"[DirectMesh]-<ad>-<ptr>"` BLAS anahtarı, raster `meshKey`).
Aynı hata **yazıcıda da** vardı: `planGeometry` artık `flat_` girdilerini
`(nodeName, transform işaretçisi)` ile gruplayıp tek mesh + tek node yazıyor.

### 3.2 Bir dalı silmek, o dala giden yolları kapatmaz
Faz 2 "glTF artık Assimp'e uğramıyor" diye kapatılmıştı. **Yanlıştı**: iddia tek
çağırma noktasında doğrulanmıştı, oysa `loadModelToTriangles`'ın **üç** çağıranı
vardı ve ikisinde glTF dalı yoktu. Bütün bitki kütüphanesi `.glb` olduğu için
serpilen her ağaç ters UV ile geliyordu. **"Artık X olmuyor" bir davranış
iddiasıdır ve çağıran başına doğrulanır.**

### 3.3 Kurulmuş + loglanmış ≠ çiziliyor
Import scatter grubu kurdu, log "20 placement" dedi, sahnede hiçbir şey yoktu:
her backend `if (group.instances.empty() || group.sources.empty()) continue;`
ile başlıyor ve ben yalnızca eski `source_triangles`'ı doldurmuştum. **Yeni bir
alt sistemi dışarıdan doldururken tüketicinin giriş kapısındaki erken `continue`
koşulunu oku.**

### 3.4 `flat_meshes` mi facade mi — ikisi denk DEĞİL
- `flat_meshes` → Vulkan mesh **başına** bir BLAS, her biri kendi materyaliyle;
  ama BLAS'ı **dünya-nesnesi geçişinin kaydından** alıyor (`m_meshRegistry`).
  Kayıt yoksa `continue` → **görünmez orman**.
- `centered_triangles_ptr` → tek BLAS, materyali **yalnızca üçgen 0'dan**.

Bu yüzden kütüphane bitkisi (sahnede dünya nesnesi **değil**) facade şeklinde
kalmak zorunda, ve okuyucuya `emitSingleFacadePerMesh = false` geçiliyor —
`InstanceManager` merkezlenmiş kopyaları `source.triangles`'ı **tek tek gezerek**
üretiyor, temsilci facade verilirse bütün ağaç tek üçgene düşer.

### 3.5 Karşı taraf bir terimi pişirdiyse, sen tekrar uygulama
Yazıcı yerleşimlere `translation(-mesh_center) * sourceWorld` gömüyor. Okuyucu
bu yüzden `mesh_center = Vec3(0)` bırakıyor. Aynı arıza şekli skinli mesh'te de
yaşandı (düğüm transformu iki kez uygulandı).
⚠ **Açık ama tetiklenmemiş:** tur hâlâ bu terimi iki kez uygulayabilir —
okuyucu prototip mesh'lerini **kendi node transform'larıyla** geri veriyor ve
backend terimi tekrar uyguluyor. Prototip origin'de + birim ölçekliyse görünmez
(doğrulanan testte öyleydi). Çaresi: `ScatterSource`'a "yerleşimler nihai"
bayrağı + `InstanceManager::buildOneSource`'un `mesh_center`'ı ezmemesi + dört
backend'de `sourceToScatter`'ın birim alınması.

### 3.6 Simetrik bir hatayı round-trip ÖLÇEMEZ
UV V çevirmesi hem okuyucuda hem yazıcıda eksikti, o yüzden
glTF→RayTrophi→glTF turu doğru görünüyordu. Konvansiyon doğrulaması **tek yönlü**
referans ister: dışarıda üretilmiş bir dosya, ya da bizim dosyamızın Blender'da
açılması. **Düzeltme de iki tarafta birden yapılır.**

### 3.7 Kaybedilen bir optimizasyon HİÇBİR ŞEY olarak raporlanır
`AssimpLoader::prefetchTextures` paralel doku çözmeyi zaten yapıyordu; doğrudan
okuyucuya **taşınmadı**. Hiçbir şey bozulmadı, çıktı aynı, sadece tek çekirdek.
**Bir okuyucuyu/bağımlılığı değiştirirken eskisinin performans geçişlerini de
envanterle.** ufbx'e geçerken aynı soruyu sor: AssimpLoader'ın FBX yolunda
prefetch/cache/paralel ne varsa listele.

### 3.10 Eski okuyucunun okudugu ALANLARI da envanterle, sadece davranisini degil
glTF'te her `textureInfo` bir `texCoord: N` tasir (hangi TEXCOORD_n). Duz
sartname, uzanti degil. Dogrudan okuyucu bunu **hic okumuyordu**; AssimpLoader
ise HEP okuyordu (`AI_MATKEY_UVWSRC` -> `selected_uv_set`). Olculdu: bir Unreal
varliginda dokusu olan 248 canli materyalin **241'i texCoord=1** kullaniyor, yani
neredeyse hepsi yanlis UV setinden ornekleniyordu.
★ Belirti "yanlis doku" degil **"doku gelmedi"** gibi gorunur — yanlis set
genelde atlasin bos bolgesine duser ve kullanici bunu "asset'te texture yokmus"
diye yorumlar. ufbx'e gecerken ayni soruyu sor: AssimpLoader FBX yolunda hangi
ALANLARI okuyor (UV kanali, ikinci UV, vertex renkleri, materyal anahtarlari)?

### 3.8 "Geçerli mi" ile "güvenle okunur mu" ayrı sorulardır
`cgltf_validate` hard gate yapılmıştı; Unreal'in boş primitifleri yüzünden
627 MB'lık geçerli bir varlık hiç açılmadı. Üçüncü parti bir doğrulayıcıyı kapı
yapmadan önce sor: **o hangi soruyu cevaplıyor, ben hangisini soruyorum?**
Ayrıca **"rejected the file" teşhis edilemez bir mesajdır** — reddeden her kapı
sebebini yazmalı.

### 3.9 Ölçüyü toplayıp atma
`ImportStats` faz sürelerini (`seconds_parse/materials/geometry/animation`)
**zaten ölçüyordu**, ama yalnızca toplam basılıyordu. "Import yavaş" şikâyeti her
seferinde yeniden enstrümantasyon gerektiriyordu. Şimdi dördü de basılıyor.

---

## 4. Açık maddeler (öncelik sırasıyla)

1. **Bu partiyi derle ve test et:** güncel NEXT_BUILD_CHECKS. ★ Derlenmemiş;
   özellikle **§0** (çekirdek tip taşıması geçişli include zincirini kırmış
   olabilir) ve **§2** (eski okuyucular hiç değişmemeli) önce.
   ufbx artış 1 ve 2 doğrulandı. glTF §9(b) geometri paylaşımı ayrı açık iş.
2. **Varlık tarayıcısı:** `.glb` üçgen/mesh sayısı ve önizleme çerçevelemesinin
   ayrı gözle doğrulaması önceki notlarda açık kalmıştı. FBX/OBJ metadata
   taraması halen Assimp; son sökme artışında değişecek.
3. **Kural 1 borcu:** `ImportStats` ölçüyor ama `scene.import_model` hiçbir şey
   döndürmüyor — import sonucunun **IPC yüzeyi yok**. `CLAUDE.md` kural 1 gereği
   borç; beş dokunuş + `python scripts/gen_ipc_descriptors.py`.
4. **`KHR_texture_transform` okunmuyor.** Blender'da Mapping node kullanan her
   varlık bu uzantıyla gelir; offset/scale/rotation sessizce yok sayılıyor.
   ★ Motorun `applyMaterialUVTransform`'u **(0.5,0.5) merkezli**, glTF'inki
   **(0,0) çıpalı**, üstüne import'taki V çevirmesi var — üçünü birden çözen
   gerçek bir dönüşüm gerekiyor, kopyala-yapıştır değil.
5. **Yazıcının skin sözleşmesi non-conformant.** Kendi export ettiğimiz skinli
   glTF standart semantiği sağlamıyor; okuyucu bunu
   `generator == "RayTrophi Studio"` sniff'i ile telafi ediyor. Yazıcı
   düzelmedikçe bu telafi **kalıcı** olur ve dosyalarımız Blender'da yanlış açılır.
6. **UV düzenlemesi Vulkan RT'de tam rebuild** (raster'da sorun yok). Kullanıcı
   bilerek erteledi. Daraltılacak yer: `refreshGeometryAfterUvChange` →
   `rebuildBackendGeometry(scene)` yerine tek nesnenin BLAS refit'i.
7. **Import'ta kalan iki paralellik fırsatı** — ölçülmedi, yapılmadı:
   `buildMeshes` primitif ekseninde (önce materyalleri seri kur, `materialIdFor`
   paylaşılan `MaterialManager`'a yazıyor), ve skin ağırlıklarının toplu okunması.
   ★ **Önce faz kırılımına bak**: `geometry` süresi `materials` yanında küçükse
   ikisi de zaman kaybıdır.
8. Seçim gizmosu bazı objelerde çizilmiyor; oynatım hızı düzeltmesinin AĞIR
   sahnede doğrulanması. (İkisi de bu göçten önce açıktı.)

---

## 5. Çalışma biçimi — pazarlık konusu değil

`CLAUDE.md`'de yazılı, ama en çok unutulanlar:

- **Build'i KULLANICI alır.** `msbuild`/`cmake`/`dotnet build` çalıştırma. Kodu
  yaz, ne test edileceğini söyle, bırak. Yeni `.cpp` eklediysen `.vcxproj`'a da ekle
  (CMake recursive glob yapıyor, oraya elle eklemek gerekmiyor).
- **Kullanıcı manuel test ediyor, test scripti istemiyor.** Bunu açıkça söyledi.
  Tanı amaçlı tek seferlik bir okuma scripti yazacaksan **depoya değil**,
  scratchpad'e yaz.
- **Her partiyi sıralı kontrol listesiyle bitir** (`docs/dev/NEXT_BUILD_CHECKS.md`,
  her partide üzerine yazılır). Sıralama: bağımsız ve hızlı görülen önce,
  başkalarının sonucunu maskeleyen sonra. Her madde için hem "ne görmen gerek"
  hem "bozuksa ne demek" yaz, ve **sessizce makul görünen sonucu ayrıca işaretle**.
- **Geriye uyum yükü yok** (kural 5). İki kod yolunu "her ihtimale karşı"
  yaşatmak bu depoda tekrar tekrar sessiz arızaya dönüştü. Ama sessizce **anlam**
  değiştirme: alanın adı da değişmeli (`GltfReadOptions` → `ImportOptions` bu
  yüzden yeniden adlandırıldı).
- **Vulkan birincil GPU yolu.** Bir şey yalnızca CUDA/OptiX'te çalışıyorsa eksik.
- **Geometri her zaman flat SoA** (`TriangleMesh` + `DNA::GeometryDetail`).
  İndeks tamponu **yok**, `Triangle` eski facade. Sahnede nesne sayan/arayan kod
  yalnızca facade tararsa flat mesh'leri "yok" sayar ve bunun belirtisi olmaz.
- **Kullanıcı Türkçe yazıyor ve Türkçe cevap bekliyor.** Kod yorumları İngilizce.

★ Ve bu deponun en pahalı hata sınıfı: **makul görünen sonuç.** Onu kimse bug
diye raporlamaz. Bu üç partideki her hata tam olarak o şekildeydi — dosya
geçerliydi, log sağlıklıydı, sahne doluydu, ve yanlıştı.

---

## 6. İlgili notlar

- [ASSIMP_IMPORT_REPLACEMENT_BRIEF.md](ASSIMP_IMPORT_REPLACEMENT_BRIEF.md) —
  göçün asıl planı, faz faz ölçümler ve kapanışlar
- [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) — **CANLI**, sıradaki build'de
  bakılacaklar (ufbx ilk artış)
- [UFBX_INCREMENT1.md](UFBX_INCREMENT1.md) — güncel kapsam, API ve eksikler
- [IMPORT_GLTF_CHECKS_ARCHIVE.md](IMPORT_GLTF_CHECKS_ARCHIVE.md) — önceki §7–§9 dahil kabul geçmişi
