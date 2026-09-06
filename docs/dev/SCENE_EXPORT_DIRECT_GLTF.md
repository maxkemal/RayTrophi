# Sahne export'u: Assimp turunu bırakıp doğrudan glTF yazmak

> **Durum:** AKTİF — yazıcı DERLENDİ ve büyük sahnede doğrulandı. Sonraki doğrulama turunda bulunan iki arızanın düzeltmesi henüz derlenmedi; bkz. §6.

Tarih: 2026-09-04. Önceki tur (2026-09-03/04) için bkz. bu dosyanın sonundaki
"Önceki turda ne yapıldı" bölümü.

---

## 1. Belirti

Kullanıcı ölçtü: **36M üçgenli, instance'ı olmayan** bir sahne export'u

- dakikalarca sürdü ve **bitmedi**,
- sürenin ~%99'unu **tek çekirdekte** geçirdi,
- **~15 GB RAM** kullandı.

Bir önceki turda eklenen paralellik bu vakada **hiç işe yaramadı**.

## 2. Kök: bayt zaten doğru biçimdeydi, kod onu kendine dönüştürüyordu

RayTrophi'nin bellek içi geometrisi **glTF'in disk düzeniyle bayt bayt aynı**:

| RayTrophi | glTF accessor |
|---|---|
| `Vec3` (3 sıkı float, 12 B) | `VEC3` / `FLOAT` (5126) |
| `Vec2` (2 sıkı float, 8 B) | `VEC2` / `FLOAT` (5126) |
| `DNA::GeometryDetail::indices` (`uint32_t`) | `SCALAR` / `UNSIGNED_INT` (5125) |

Eski yol bu baytları **kendilerine çevirmek için** şunları yapıyordu:

1. Flat SoA mesh'i üçgen başına bir `ExportTriangle` cephe nesnesine patlatıyordu
   (36M × 32 B ≈ **1,15 GB**, vector büyümesiyle geçici olarak iki katı), ve her
   üçgen için `nodeName + "_" + mat_id` **string'i kuruyordu** (36M malloc).
2. Zaten indeksli olan köşeleri bir hash map ile **yeniden dedup ediyordu**
   (108M'e kadar düğüm; MSVC'de düğüm başına ayrı allocation ⇒ **~6-7 GB**).
3. Sonucu `aiMesh`'e kopyalıyordu (**+3,9 GB**), ve `aiFace` yüzünden **üçgen
   başına bir `new unsigned int[3]`** yapıyordu — 36M küçük allocation.
4. Assimp'in glTF2 exporter'ı bunların üstüne **kendi üçüncü kopyasını** kurup
   yazıyordu.

Toplam ≈ 16 GB; bildirilen 15 GB ile örtüşüyor.

**Ve hepsi seriydi**, çünkü paralellik **obje ekseninde** bölünüyordu:
`num_threads = min(hardware_concurrency, objectCount)`. Tek dev mesh ⇒ `= 1`.

> ★ Genel ders: **paralellik işin eksenine göre bölünür, kabın eksenine göre
> değil.** "Objeleri thread'lere dağıt" ifadesi bir sahne için doğru, bir mesh
> için anlamsızdır — ve ikisi aynı koddan geçtiği için fark hatasız görünür.

## 3. Yeni yapı: `GltfDirectWriter`

`source/include/GltfDirectWriter.h` + `source/src/Utils/GltfDirectWriter.cpp`.
Assimp export yolu **tamamen söküldü** (`createAssimpMaterial`,
`patchExportedGltfExtras`, `exportSkeleton`, `ExportTriangle`/`MeshBatch`,
base64 gömme — ~1500 satır silindi). Assimp yalnızca **import**'ta kaldı.

### 3.1 Önce PLANLA, sonra YAZ

Bir GLB, JSON chunk'ını BIN chunk'ından önce yazmak zorundadır; ama JSON'un
içinde BIN'deki her bayt ofseti geçer. İki bilinen çözüm de kötü: BIN'i RAM'de
tamponlamak (bellek) veya iki kez yazmak (I/O).

Bunun yerine yazıcı **tüm ikili yerleşimi önce planlar** — her uzunluk ucuz bir
sayım geçişinden bilinir — bitmiş JSON'u yazar, sonra parçaları planlanan sırada
**akıtır**. Tek geçiş, geçici dosya yok, birleştirme yok.

Sonuç: tepe RAM `O(JSON + kodlanmış doku + küçük üretilmiş dizi)`; **sahne
geometrisiyle orantılı değil.** 36M üçgenli mesh pozisyon/normal/uv yolunda
**sıfır** heap allocation üretir — diziler zaten oldukları yerden yazılır.

### 3.2 Malzeme başına primitive, köşe kopyası YOK

glTF mesh'i primitive listesidir ve primitive'ler **aynı attribute
accessor'larını paylaşıp yalnızca `indices`'te ayrılabilir**. Yeni yazıcı bunu
kullanır: N malzemeli bir mesh **bir** köşe kümesi + N indeks aralığı yazar.
Eski yol her malzeme batch'i için köşe dizisinin tamamını çoğaltıyordu.

### 3.3 Üçgen ekseninde paralellik

`parallelRange(count, minPerThread, fn)` — her sıcak döngü **üçgen/köşe**
eksenini böler. Malzeme histogramı, POSITION min/max indirgemesi, doku sRGB
dönüşümü ve metallic-roughness paketleme hep bu eksende.

Küçük mesh'ler mesh listesi üzerinde paralel; `>= 65536` üçgenli mesh'ler
**kendi üçgenleri** üzerinde paralel, teker teker. (İç içe paralellik yok.)

### 3.4 Yan etki: iki sessiz veri kaybı düzeldi

- **Roughness/metallic dokuları**: glTF bunları **tek** dokunun G ve B
  kanallarında ister. Eski yol iki ayrı standart-dışı slot yazıyordu; uyumlu
  hiçbir görüntüleyici okumaz. Yeni yazıcı paketliyor.
- **UV ölçek/kaydırma**: eski yol tamamen düşürüyordu. Artık
  `KHR_texture_transform` ile taşınıyor. ★ Kayıp UV tiling "doğru ama yanlış"
  görünür — kimsenin bug diye raporlamadığı sınıf.

Ayrıca `.gltf` metin modu artık base64 gömmek yerine **yanında `.bin` dosyası**
yazıyor (%33 daha küçük, RAM'e tek parça string sığdırmıyor) ve GLB'nin 4 GB
tavanı yok. Instancing artık iki modda da çalışıyor (eski yol yalnız GLB).

### 3.5 Uzunluk doğrulaması (yeni tuzak, yeni koruma)

Ham akıtma yaptığımız için bir **işaretçi yeterli değil**:
`get_positions_orig_count()` `vertex_count`'ten **kısa** olabilir (yeniden üreyen
yüzeyler: sıvı/gaz izoyüzeyi, partikül mesh'i). `vertexCount` eleman okumak
tamponu aşar ve hasar **sessizce dosyaya** yazılır.

- `DNA::GeometryDetail::get_core_attribute_count(Attr)` eklendi (mevcut
  `get_positions_orig_count()`'un genel hali).
- Yazıcı akıttığı **her** tamponu uzunluk kontrolünden geçirir; kısa bir P_orig
  anlık görüntüsü canlı tampona düşer ve bunu log'lar. Normaller kabul edilen
  pozisyon anlık görüntüsüyle **eşleştirilir** (bind-pose pozisyon + canlı normal
  karışımı yanlış gölgeleme verir).
- `sizeof(Vec3)==12` / `sizeof(Vec2)==8` artık **`static_assert`**. Bu düzen
  bozulursa dosya değil **build** patlar.

## 4. Kural 1: export'un artık script yüzeyi var

Export yıllarca **hiç** IPC/script yüzeyi olmadan yaşadı — bu arızanın ancak elle
export edip Task Manager'a bakarak bulunabilmesinin sebebi tam olarak budur.

`scene.export_gltf` eklendi (beş dokunuş tamam: `RtApi.h`/`RtApi.cpp`,
`RtIpc.cpp`, `RtPython.cpp`, `RtIpcSecurity.cpp`, overlay + üretici;
`audit_ipc_capabilities.py` yeşil).

Yetki **`FilesWrite`**, `SceneWrite` değil: `scene.` namespace'inde ama sahneyi
okuyup **dosya** yazıyor. Hem C++ hem Python aynası bu istisnayı adıyla taşıyor.

Yanıt **ölçülen** maliyeti döndürür — faz faz saniye ve yazıcının **kendi** tepe
heap'i (`peak_writer_mb`, process RSS değil) — yani export maliyeti artık
script'ten regresyona sokulabilir:

```powershell
Invoke-RtIpc scene.export_gltf @{ path = 'E:\tmp\scene.glb' }
```

## 6. Doğrulama turu (derlendikten SONRA, IPC'den sürülerek)

Yazıcı derlendi; kullanıcının 36M üçgenli sahnesi **hızlıca export edildi**.
Ardından `scene.export_gltf` ile bir doğrulama turu yapıldı — ve kural 1'in
gerekçesi ilk kullanımda kendini kanıtladı: **iki gerçek arıza** çıktı.

### 6.1 Doğrulanan kapsam (ölçüldü)

| Test | Sonuç |
|---|---|
| `cesium_man.gltf` (rig + animasyon) round-trip | ✅ 19 joint, JOINTS_0/WEIGHTS_0 geçerli, ağırlıklar 1'e toplanıyor, IBM sayısı = joint sayısı, iskelet hiyerarşisi zincir, 57 kanallı T/R/S animasyonu |
| Kendi `.glb`'mizi geri import | ✅ mesh/üçgen sayıları birebir |
| Kamera + `KHR_lights_punctual` | ✅ |
| `.gltf` yan dosya modu | ✅ `<isim>.bin` |

Yapısal doğrulama `glb_check.py` ile: her accessor'ın baytları tamponun içinde,
her indeks köşe aralığında, POSITION min/max **gerçek** min/max ile uyumlu, PNG
sihirli sayıları doğru.

### 6.2 ★★ Arıza A: scatter Vulkan yolunda export'a hiç girmiyordu

`scatter.fill` → 1000 instance. `scene.export_gltf` → **`instances: 0`**.

`SceneUI::syncInstancesToScene()` Vulkan backend'inde **erken dönüyor** —
kasten:

> "Vulkan RT and Vulkan raster both consume the canonical flat source meshes
> plus InstanceGroup::instances directly. Never expand that data back into one
> CPU HittableInstance facade per placement."

Export ise yalnızca `scene.world.objects`'i tarıyordu. Yani **birincil GPU
yolunda scatter hiç yazılmıyordu** ve dosya bu haliyle tamamen geçerliydi.

★★★ Bu, CLAUDE.md'deki **"üretici ≠ tüketici"** dersinin ders kitabı örneği.
Bir önceki tur "scattered objeler artık destekleniyor" diye not düşmüştü; ama
eklenen destek yalnızca **CPU-compat açılımını** kapsıyordu — Vulkan onu hiç
üretmiyor. *Bir yolu desteklemek, o yolun gerçekte kullanıldığı anlamına gelmez.*

Çare: yazıcı `InstanceManager::getGroups()`'u doğrudan okuyor ve
**VulkanBackend.cpp ile birebir aynı** bileşimi kullanıyor:

```
final = instance.toMatrix() * translation(-source.mesh_center) * sourceWorld
```

Kaynak mesh hem sahne objesi hem scatter kaynağıysa **tek** glTF mesh'i paylaşır
(`flatMeshIndex_`), yani köşe verisi çoğaltılmaz.

### 6.3 Arıza B: silinen nesneler dosyaya sızıyordu

`scene.list_objects` 2 nesne dedi, dosyada **4 mesh** vardı. Editörde silinen
nesne `world.objects`'te kalıyor (fiziksel temizlik kayıt anında);
`rtapi::listObjects` bunları `isEditorPendingDeleteObjectName` ile eliyor, export
elemiyordu. **Eski Assimp yolunda da aynıydı** — regresyon değil, yıllardır
oradaydı ve export'un script yüzeyi olmadığı için görülemiyordu.

★ Sinsi olan: dosya "çalışıyor", sadece sahnede olmayan şeyler içeriyor.

## 5. Açık kalanlar

- **Import hâlâ Assimp.** Kullanıcı bunu da sorun olarak işaretledi ve bir miktar
  optimize etti (`aiProcess_ImproveCacheLocality` / `CalcTangentSpace` kaldırıldı,
  glTF'de `JoinIdenticalVertices` atlandı). Sıradaki tur: glTF/GLB import'u
  `cgltf` ile doğrudan flat SoA'ya okumak — `vcpkg/ports/cgltf` zaten mevcut ve
  tek başlıklı. glTF import'u doğrudan yapmak, bu depoda **aynı simetriyi**
  kurar: dosyadaki bayt düzeni ile `GeometryDetail`'inki aynı.
- `bake_transforms` ayarı hiçbir zaman uygulanmadı (eski yolda da). Node
  transform'u yazmak glTF'in doğru deyimi olduğu için kasten böyle bırakıldı.
- Terrain bake artık "bu malzeme bir batch'te kullanılıyor mu" ön kontrolünü
  yapmıyor (batch kavramı kalktı). Sahnedeki her splat'lı terrain bake edilir.

## Önceki turda ne yapıldı (2026-09-03/04, hâlâ geçerli)

Bunlar Assimp yolundaki iyileştirmelerdi; ikisi yeni yazıcıda da yaşıyor:

- Scatter/foliage `HittableInstance` objeleri eskiden export'ta **tamamen sessizce
  atlanıyordu**; artık kaynak başına bir kez geometri + instance başına hafif
  node (yeni yazıcıda doğrudan `EXT_mesh_gpu_instancing`).
- Export popup'ı ön tahmin ve büyük sahne için onay kutusu gösteriyor. Tahmin
  modeli yeni yazıcıya göre güncellendi: flat SoA mesh'ler artık tepe RAM'e
  **katkı yapmıyor**, yalnızca legacy `Triangle` cephesi ve instance kaynakları
  materyalize ediliyor.
