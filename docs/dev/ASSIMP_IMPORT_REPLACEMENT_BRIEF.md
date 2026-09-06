## c.glb: confirmed legacy export skin-space mismatch - 2026-09-05

Read-only inspection of the user's c.glb establishes a different, concrete
cause from the earlier missing-track finding. Asset generator is exactly
"RayTrophi Studio". It contains 50 skinned mesh nodes and 33 joints. Mesh nodes
store 0.1 scale with +90 degree X rotation. Joint bind world * inverse bind
instead contains 10 scale with -90 degree X rotation. Dropping the mesh matrix
therefore produces the user's tenfold growth and sideways character.

Measured directly from file accessors: body source extents [1.7275, 1.2634,
0.3436] become [17.2750, 3.4362, 12.6339] under skinning alone, and return to
source extents after the authored mesh correction. Across all 50 meshes and
33 joints, max absolute error in meshWorld * jointBindWorld * IBM - identity
is 7.50251e-6.

GltfDirectReader now retains meshWorld ONLY when generator is RayTrophi Studio
and every joint satisfies this cancellation within 0.002 (finite values).
Standard glTF retains identity mesh base. No guessed global axis/unit fix.
Log: [glTF] legacy RayTrophi skin-space correction retained for '...'

The existing writer stores mesh transforms separately and only emits indexed
skeleton nodes; legacy output is not proof of standard glTF skin semantics.
This reader compatibility fix does not repair the exporter's format contract
or guarantee recovery of every legacy file with different bind-space layouts.

Manual validation: rebuild normally, reimport c.glb, verify size/up axis at
import, playback and scrub; confirm the compatibility log. No build/application
launch or test script was performed. Numeric work only read the user's file.
The separate static empty-parent issue still requires its actual sample file.

---

## Follow-up after failed manual build check - 2026-09-05

User rebuilt: BOTH reported defects persisted. Earlier changes are not runtime
acceptance evidence. A further C++ animation defect is now corrected:

- OzzRuntime::buildRawAnimation left missing component tracks empty. The local
  vendored Ozz animation_builder.cc CopyRaw explicitly fills empty tracks with
  identity keys; it does not inherit skeleton rest transforms. Missing parent
  rotation/scale therefore disappeared even with a complete imported hierarchy.
  Empty tracks now receive the corresponding local bind-pose component.
- AnimationController::calculateNodeTransform also started every component at
  identity when a clip existed. It now preserves defaultTransform verbatim for
  unkeyed nodes and retains bind components for partially keyed nodes.
- Source evidence: bundled cesium_man.gltf node 0 has a -90 degree X matrix and
  no animation channel. Previously Ozz replaced that root matrix with identity.

Static empty-parent placement is STILL UNDER INVESTIGATION. Assimp's
getGlobalTransform and cgltf_node_transform_world both compose the full parent
chain, and both readers keep source vertices local. The user's failing asset
paths have been requested to locate the first actual divergence; do not claim
that this animation fix resolves an entirely static file.

No build, application launch or test script. Source diff whitespace check passed.
Manual check: reimport the character, verify initial pose/playback/scrubbing,
then compare an existing FBX. Static-file diagnosis awaits the concrete asset.

---

## Takeover update - 2026-09-05 (source checked; manual verification pending)

The earlier identity mesh base + identity global inverse correction and joint
ancestor closure remain in place. This follow-up completes additional gaps:

- Direct glTF resolves one unique node identity for geometry, hierarchy, skin
  joints and animation channels. Duplicate names (including empty parents) no
  longer overwrite another node's animated world matrix. The synthetic root
  also reserves its own unique identity.
- Nodes referenced outside the active scene retain their complete hierarchy;
  the BoneData safety pass now includes technical ancestors, not only indexed
  joints, and assigns fallback indices in stable source order.
- Skin classification is shared by transform selection and weight decoding,
  per primitive. A static primitive in a mixed mesh keeps its full node world
  transform instead of inheriting the skinned primitive's identity base.
- Transform::setBase with a pivot and setPivotOffset(..., true) preserve the
  exact affine matrix. Decomposing/recomposing it previously discarded shear
  from rotated, non-uniformly scaled empty-parent chains. Explicit TRS edits
  still use the existing TRS representation.

Manual checks after your normal build (no build/application run by Codex):
1. Import the previously failing animated character GLB: check orientation,
   size and position immediately, then play, scrub and return to frame zero.
2. Import Blender static meshes under nested empties with translation,
   rotation and non-uniform scale. Compare placement and proportions; change
   the pivot with world preservation and verify geometry stays stationary.
3. Check duplicate node names and a multi-primitive mesh containing both
   skinned and static geometry, if available.
4. Append two characters, then save/reopen RTP; verify each rig stays separate.
5. Recheck an existing FBX and pivot editing, since Transform is shared.

No test scripts were added or executed. Runtime correctness remains pending
these manual checks. This update supersedes the earlier mesh-only naming
policy and nodeMeshIsSkinned references below.

---

# Assimp import'tan çıkış — devretme notu

> **Durum:** AKTİF — **Faz 2 BİTTİ (yazıldı, derlenmedi — 2026-09-05):
> glTF/GLB artık Assimp'e HİÇ uğramıyor.** Doğrudan okuyucu geometri, ekstra UV
> setleri, materyal/doku, skin, animasyon, kamera, `KHR_lights_punctual` ve
> `EXT_mesh_gpu_instancing` okuyor; Assimp fallback'i **kaldırıldı** (kural 5).
> `AssetRegistry` ve `scene_ui` de glTF için kendi `Assimp::Importer`'larını
> bırakıp `rtimport::probeGltf`'e geçti. **Assimp hâlâ FBX + OBJ için duruyor**
> (`AssimpLoader.h` ve `Texture.h`'daki `aiTexture` kurucusu) — lib'in düşmesi
> Faz 3'e bağlı. Doğrulama adımları `NEXT_BUILD_CHECKS.md`.
> Yeni bir oturumun ilk okuyacağı dosya budur.

Bu not, konuyu sıfırdan çıkarım yapmadan devralabilmek için yazıldı. Aşağıdaki
her iddia bu depoda **doğrulandı**; tarih ve dosya verildi.

---

## Bağlam: export bitti, import kaldı

Export tarafı 2026-09-04'te Assimp'ten tamamen çıkarıldı: `GltfDirectWriter`
(plan-sonra-yaz, tek geçiş, flat SoA'yı doğrudan akıtır) — kullanıcı 36M üçgenli
sahneyi başarıyla export etti. Bkz. `SCENE_EXPORT_DIRECT_GLTF.md`.

**Import hâlâ Assimp.** Ve projenin kendi `.rtp` paketi modelleri `.glb` olarak
sakladığı için glTF import **sıcak yol** — yani en çok kazanç orada.

---

## ★★★ Asıl engel: Assimp bir YÜKLEYİCİ değil, bir VERİ TİPİ olmuş

Bu, işin gerçek maliyetini belirleyen bulgu. Assimp yalnızca dosya okumuyor;
tipleri çekirdek sahne yapılarının İÇİNDE yaşıyor:

| yer | ne tutuyor | dosya |
|---|---|---|
| `BoneData::boneNameToNode` | **`aiNode*`** — ham Assimp işaretçisi | `AssimpLoader.h:429` |
| `AnimationData` | `std::vector<aiVectorKey>` / `aiQuatKey` (pozisyon/rotasyon/ölçek anahtarları) | `AssimpLoader.h:272-274` |
| `AnimatedObject::interpolate*` | imzaları `aiVectorKey` / `aiQuatKey` alıyor | `AnimatedObject.h:196,225,240` |
| `AnimationNodes` | `sampleVectorKey` / `sampleQuatKey` aynı tipleri alıyor | `AnimationNodes.h:45-46` |
| bind pose | `aiMatrix4x4 aiDef` | `AssimpLoader.h:291` |

★ Yani "Assimp'i çıkar" demek, **animasyon anahtar tipini bütün animasyon
sisteminde değiştirmek** demek. Yükleyiciyi değiştirmek işin küçük yarısı.

`assimp|aiScene|Importer` geçen dosya sayısı: **20+** (`Renderer.h`,
`scene_data.h`, `Texture.h`, `Triangle.h`, `ProjectData.h`, `ProjectManager.cpp`,
`AssetRegistry.cpp`, `OptixWrapper.cpp` dahil).

---

## Elde ne var

- `tiny_gltf.h` **zaten depoda**: `external/ozz-animation/src/animation/offline/gltf/extern/`
  (ozz'un offline araç zinciri için). Yani bir glTF okuyucu ağaçta mevcut.
- `cgltf` bir **vcpkg portu** olarak erişilebilir (`vcpkg/ports/cgltf`); kurulu
  olup olmadığı doğrulanmadı.
- Yazma tarafında `GltfDirectWriter` var — glTF ABI'sini bu depoda zaten bilen,
  denenmiş kod. Okuyucu onun ayna görüntüsü olarak tasarlanabilir.

---

## Önerilen faz sırası (karar VERİLMEDİ, tartışılacak)

**Faz 0 — Ayrıştırma. ✅ YAZILDI (2026-09-05), DERLENMEDİ.**

`RayTrophi::VectorKey` / `QuatKey` (`source/include/Animation/AnimationKeys.h`)
artık animasyon sisteminin tek anahtar tipi. `AnimationData`, `AnimationNodes`,
`AnimationController`, `OzzRuntime`, `.rtp` serileştirici, `TimelineWidget` ve
`GltfDirectWriter` hepsi bu tipi konuşuyor. **Davranış değişmedi ve `.rtp` dosya
şeması da değişmedi** (anahtarlar hâlâ `time/x/y/z/w`), yani eski projeler
aynen açılıyor.

Yol boyunca sökülen iki şey — ikisi de göç değil, **ölü yol**:

- **`BoneData::boneNameToNode` (`aiNode*`) SİLİNDİ.** Sadece Assimp tipi değil,
  **sarkan** bir işaretçiydi: `aiScene` importer kapsamdan çıkınca ölüyor,
  `BoneData` ondan uzun yaşıyor (Renderer haritayı importlar arası birleştiriyordu
  bile). Hiçbir yer okumuyordu; taşıdığı her değer zaten yanı başında
  `boneDefaultTransforms` / `boneParents` içine kopyalanıyordu.
- **`AnimatedObject.h` SİLİNDİ.** Hiçbir yerde örneklenmiyordu (Renderer.cpp'de
  zaten "wrappers removed (were unused)" notu vardı), ama Assimp tipli üç giriş
  noktasını (`interpolatePosition/Rotation/Scaling`) tutuyordu. ★ Ölü kodu
  göç ettirmek, tip değişikliğini üç katı büyük gösterirdi. Ayrıca
  `interpolateRotation` `Quaternion(x, y, z, w)` kuruyordu — kurucu `(w, x, y, z)`
  aldığı için kullanılsaydı asla doğru olamazdı.

`aiVectorKey`/`aiQuatKey` artık **tek bir yerde** yaşıyor: `AssimpLoader.h`
içindeki `appendAnimationKeys` / `assignAnimationKeys` dönüştürücüleri —
yani yükleyici sınırında. Aynı şekilde `aiMatrix4x4::Decompose()` çağrıları
`RayTrophi::decomposeTRS()` ile değişti (`AnimationData` ve `OzzRuntime`).

★ Bu faz bitmeden okuyucu yazmak, iki temsili yan yana yaşatmak olurdu; bu
depoda o hata sınıfı defalarca ısırdı (bkz. `feedback_flat_soa_is_the_geometry_model`).

**★★★★ Faz 0.5 — DÜĞÜM HİYERARŞİSİNİ SAHİPLEN. Brief bunu KAÇIRMIŞTI, ve
okuyucudan önce gelmesi zorunluydu.**

Bu notun ilk hâli `aiNode*` engelini `BoneData::boneNameToNode` diye tarif
ediyordu. Asıl engel daha büyüktü ve daha derindeydi:

    Renderer::updateAnimationWithGraph / updateAnimationState
        modelCtx.loader->calculateAnimatedNodeTransformsRecursive(
            modelCtx.loader->getScene()->mRootNode, ...)   // HER KARE

Yani animasyon çalışma zamanı **canlı aiScene düğüm ağacını her karede
geziyordu**, ve `ImportedModelContext::loader` esas olarak bunun için tüm
`aiScene`'i oturum boyunca canlı tutuyordu. cgltf ile yüklenen bir modelin
aiScene'i olmayacağı için **animasyon hiç oynamazdı — hatasız, uyarısız.**

★ Yürüyüşün `aiNode`'dan istediği tek şey şuydu: bir ad, bir yerel bind
transformu, ve çocuklar. Bağımlılık hiçbir zaman Assimp'e değildi; sadece bu üç
alanı kimse yazmamıştı.

Yapılan: `RayTrophi::NodeHierarchy` (`Animation/NodeHierarchy.h`) — düz dizi +
çocuk indeksleri, kopyalanabilir, sahiplik sorusu yok.
`computeAnimatedGlobalTransforms()` yürüyüşü devraldı; `aiNode` sürümü
söküldü (kural 5). `uniqueName` **inşa anında** çözülüyor: onu yürüyüş sırasında
türetmek, sıcak yola bir loader işaretçisi sokan şeyin ta kendisiydi.

Yan sonuç: `ImportedModelContext::loader` artık **yazılıyor ama hiç
okunmuyordu** → kaldırıldı. aiScene import biter bitmez serbest kalıyor.

**Faz 1 — Doğrudan glTF/GLB okuyucu. ✅ YAZILDI (2026-09-05), DERLENMEDİ.**

`Import/GltfDirectReader.{h,cpp}` — cgltf 1.15 (vendored, `external/cgltf`).
Kapsam: geometri (doğrudan flat SoA), düğüm hiyerarşisi, materyaller +
dokular (gömülü bufferView ve harici URI), skin (JOINTS_0/WEIGHTS_0 +
inverseBindMatrices), animasyon, kameralar, `KHR_lights_punctual`.

Ortak sonuç yapısı `Import/ImportedModel.h` — **sanal arayüz değil, düz veri**.
Gerekçe orada yazılı; kısaca: format seçimi tek dallanma, ve parity probe'un
karşılaştırdığı şey zaten bu struct'ın iki dolumu. Alanlar `create_scene`'in
TÜKETTİĞİNDEN türetildi, `AssimpLoader`'ın üyelerinden değil — "tarafsız" bir
tipi tek bir uygulamadan tasarlamak, Assimp'in bu kod tabanında veri tipine
dönüşme sebebinin ta kendisi.

Assimp fallback var ama **sessiz değil**: başarısızlıkta sebebiyle birlikte
`SCENE_LOG_WARN`. Aksi halde "yeni okuyucu çalışıyor" ile "başarısız oldu ve
Assimp örttü" dışarıdan aynı görünürdü.

### Faz 2 kapanışı (2026-09-05) — dört yerin İKİSİ kapandı

- `AssetRegistry` ve `scene_ui` artık glTF için `rtimport::probeGltf` kullanıyor.
  ★ Bu yalnızca bağımlılık temizliği değil: ikisi de bir SAYI ve bir KUTU için
  dosyanın tamamını çözüyordu (AssetRegistry üstelik `aiProcess_ImproveCacheLocality`
  ile bir Tipsify pass'i ödüyordu). glTF `POSITION` accessor'ında min/max'ı
  **şartname gereği zorunlu** kılıyor — kutu zaten JSON'da, probe tamponları hiç
  yüklemiyor.
- `Renderer::create_scene` glTF için Assimp'e **düşmüyor**. Başarısızlık artık
  başarısızlık: iki okuyucuyu bir format için canlı tutmak (kural 5) fallback'in
  gerçek hataları örtmesiyle sonuçlanıyordu.
- `EXT_mesh_gpu_instancing` okunuyor. ★ Yazıcı bunu **hep üretiyordu**; okunmadığı
  için RayTrophi'nin kendi scatter export'u geri açıldığında prototip mesh ve BOŞ
  bir orman geliyordu — geçerli bir dosya, hatasız. "Formatı destekliyoruz" ile
  "kendi çıktımızı geri açabiliyoruz" ayrı iddialar.
- Ekstra UV setleri (`TEXCOORD_1..n`) okunuyor. Bu olmadan fallback'i kaldırmak
  çok-UV'li her varlık için **sessiz bir geriye gidiş** olurdu.

Kalan iki yer (`AssimpLoader.h`, `Texture.h`'daki `aiTexture` kurucusu) FBX/OBJ'e
bağlı ve Faz 3'te düşer.

### ✅ glTF YOLU KAPANDI (2026-09-05, kullanıcı derledi ve ölçtü)

Üç tur doğrulama sonunda glTF/GLB **hem okuma hem yazma** tarafinda Assimp'siz ve
tur kapanıyor:

- Skinli animasyon, doku yönü, fotometrik ışık şiddeti, ekstra UV setleri ✓
- `EXT_mesh_gpu_instancing` scatter: **doğru katman sayısı, doğru konum, doğru
  ölçek** — kendi export'umuz geri açılıyor ✓
- Çok materyalli obje **tek obje** kalıyor (okuyucuda `_primN` eki söküldü,
  yazıcıda `flat_` gruplama eklendi) ✓
- Bitki kütüphanesi ve animasyon klibi de artık doğrudan okuyucudan geçiyor
  (`Import/ModelImport.h` seam'i) ✓
- ★ Hız: kullanıcının ölçümü **"Assimp paralel yapıdan bile çok hızlı"**. Paralel
  doku prefetch'i (Assimp'ten devralınmamıştı), yerinde vertex unpack ve toplu
  indeks okuma.

Sıradaki: **Faz 3'ün asıl işi** — ufbx ile FBX okuyucusu, sonra küçük bir OBJ
ayrıştırıcı, en son `assimp-vc143-mt.lib`'in sökülmesi.

✅ **Canlı doğrulama (2026-09-05, kullanıcı derledi):** glTF/GLB Assimp'siz
açılıyor, skinli animasyonlar doğru, dokular doğru yönde, ışık şiddetleri
fotometrik dönüşümle geliyor, ve **scatter instance objeler doğru import
ediliyor** — yani RayTrophi'nin kendi scatter export'u artık geri açılabiliyor.
★ Bu son madde Faz 2'nin asıl iddiasıydı: "formatı destekliyoruz" değil,
**tur kapanıyor.**

★ Bu turda ilk denemede okuma çalışıp sahne boş kaldı; sebebi grubun
`sources` alanının boş olmasıydı (her backend orada `continue` ediyor). Arıza
şekli kayıtlı, çünkü tekrar edecek: **kurulmuş + loglanmış ≠ çiziliyor.**

### ★★★★★ DÜZELTME: FAZ 2'NİN İDDİASI TEK ÇAĞIRMA NOKTASINDA DOĞRULANMIŞTI (2026-09-05, Faz 3 başlangıcı)

Faz 2 kapanışında "glTF/GLB artık Assimp'e hiç uğramıyor" yazdım. **Yanlıştı.**
Doğrusu: *`Renderer::create_scene` için* Assimp'e uğramıyordu. Faz 3'ün ilk adımı
olarak `loadModelToTriangles`'ın çağıranlarını saydım — üç tane var, ve ikisinde
glTF dalı yoktu:

| Çağıran | glTF dalı |
|---|---|
| `Renderer.cpp` | ✔ |
| `FoliageAssetLibrary.cpp` | ✘ |
| `scene_ui.cpp` (animasyon klibi) | ✘ |

Bedeli somut: **bütün bitki kütüphanesi `.glb`** (`assets/vegetation/**`). Yani
Faz 2'den sonra aynı dosya sahneye nasıl girdiğine göre iki farklı sonuç
veriyordu — model olarak sürüklenince doğru UV, bitki olarak dikilince
Assimp'ten ters V (`AssimpLoader` hiçbir yerde `aiProcess_FlipUVs` kullanmıyor).
Serpilen her ağaç atlasını baş aşağı örnekliyordu, hata mesajı yok.

★★★ **Ders, kod hatasından daha değerli: "artık X olmuyor" bir DAVRANIŞ
iddiasıdır ve çağIRMA NOKTASI BAŞINA doğrulanır.** Bir dalı kaldırmak, o dala
giden bütün yolları kapatmaz. Doğru ölçüm şu: eski giriş noktasını grep'le,
çıkan her çağıranı tek tek oku.

**Çare:** dispatch tek yere taşındı — `Import/ModelImport.h` /
`rtimport::loadModel()`. Artık çağıranlar bir OKUYUCU değil bir DOSYA istiyor,
ve ufbx eklemek bu dosyada tek satır olacak. `create_scene` bilerek `readGltf`'i
doğrudan çağırmaya devam ediyor (ışık/kamera/instance grubu/istatistik tüketiyor):
bu ikinci bir okuyucu değil, aynı okuyucunun **daraltılmış** sarmalayıcısı.

Ayrıca `GltfReadOptions` → **`ImportOptions`** olarak yeniden adlandırıldı ve
`Import/ImportedModel.h`'ye taşındı (kural 5: anlamı genişleyen alanın adı da
değişir), ve `external/ufbx/` vendor'landı (MIT) — henüz derlemeye dahil değil.

### ★★★★★ ASSIMP'İN KALAN YÜZEYİ DÖRT YERDE, BİR YERDE DEĞİL (2026-09-05 ölçümü)

"Assimp'i sök" işi `AssimpLoader` ile bitmiyor. Depoda `assimp/` başlığını gerçekten
kullanan **dört** yer var (geri kalan eşleşmeler yorum satırı):

| Yer | Ne yapıyor | Söküm için ne gerekiyor |
|---|---|---|
| `include/AssimpLoader.h` (147 kullanım) | asıl okuyucu; artık yalnız **FBX + OBJ** | Faz 3: ufbx + küçük bir OBJ ayrıştırıcı |
| `include/Texture.h` | `Texture(const aiTexture*)` + `decode_raw`/`decode_compressed` | Başka çağıranı yok; loader ile birlikte ölür |
| `src/Scene/AssetRegistry.cpp` | **KENDİ `Assimp::Importer`'ı**: varlık tarayıcısı için mesh/materyal/animasyon sayısı + bbox | Karar gerekiyor — aşağıya bak |
| `src/UI/scene_ui.cpp` (~276) | ÜÇÜNCÜ bağımsız aiScene yürüyüşü, yine bbox için | Aynı karar |

★★ **Son ikisi kimsenin "importer"ı değil, ve tam da bu yüzden kataloglanmamışlardı.**
`AssimpLoader`'ı kusursuz şekilde ufbx ile değiştirsen bile bu ikisi
`assimp-vc143-mt.lib`'i ayakta tutar — yani DLL düşmez ve derleme zamanı geri gelmez.
Bu, briefin ilk sürümünün Faz 0.5'i kaçırmasıyla aynı hata sınıfı: **bağımlılığı
"import" adı geçen dosyalarda aramak.**

★★★ **İkisi de AYNI İŞİ İKİ KEZ yapıyor, ve o iş zaten yapılmış durumda.** Her ikisi
de bir dosyayı TAM ayrıştırıp (AssetRegistry üstelik `aiProcess_ImproveCacheLocality`
ile, yani sadece bbox için bir Tipsify passı ödüyor) saydığı şeyler
`rtimport::ImportStats`'in zaten TAŞIDIĞI şeyler: mesh/vertex/üçgen/materyal/
animasyon sayıları. Doğru hamle "AssetRegistry'yi ufbx'e taşımak" değil,
**AssetRegistry'nin okuyucuya SORMASI** — ölçümü üreten yer zaten okuyucu.

> Bu bir tasarım kararıdır, sessizce yapılmamalı: varlık tarayıcısı bugün
> **yüklemeden** metadata çıkarabiliyor. Okuyucuya sormak, ya "sadece istatistik"
> modu olan bir okuyucu API'si (`GltfReadOptions`'da zaten `loadGeometry=false`
> var) ya da bir sidecar önbellek gerektirir. Hangisi olduğu ölçülerek seçilmeli.

### ★★★★★ Faz 1'in yerleşim hatası — sökülen terim değil, SADELEŞEN terim

Faz 1 ilk testinde `.glb` skinli karakterler yanlış konumda, yanlış hizada ve
yanlış BOYUTTA geldi; basit bir küp ise kusursuzdu. İki belirti tek köke çıktı.

Motorun skinning zinciri (`TriangleMesh::applySkinning` + `Renderer` 2487):

```
world = transform.base × globalInverse × jointWorld × IBM × v
```

`jointWorld` **zaten tüm ata zincirini taşır**. Okuyucu skinli mesh'e
`base = nodeWorldMatrix(node)` verip `globalInverse = identity` beyan edince
kök matris **iki kez** uygulanıyordu. `cesium_man.gltf` en küçük vaka: düğüm 0
bir −90° X (Z-up→Y-up) düzeltmesi, skinli mesh onun çocuğu.

★★ **Assimp'te görünmemesi bir doğruluk değil, bir SADELEŞMEYDİ:**
`globalInverseTransform = inverse(kök mTransformation)` ve tek köklü bir glTF'te
Assimp düğüm 0'ı aiScene köküne terfi ettiriyor — yani `base × globalInverse`
tam olarak birime sadeleşiyordu. Assimp sökülünce **sadeleştiren terim gitti,
sadeleşen terim kaldı.**

> **Ders (bu dosyanın en pahalı satırı):** bir karşılıklı sadeleşen çiftin
> yalnızca bir yarısını devralırsan hata çıkmaz — **makul görünen yanlış bir poz**
> çıkar. Yeni okuyucu yazılırken sorulacak soru "bu alanı dolduruyor muyum" değil,
> "**bu alan kimin terimini sadeleştiriyordu**" olmalı.

Düzeltme, şartnamenin zaten söylediği şey: glTF 2.0 §Skins — *skinli mesh'i
referanslayan düğümün transformu YOK SAYILIR*. `base = identity`.

Aynı partide kapatılan iki yan açık:
- **Mesh düğüm adları tekilleştirilmiyordu.** `AssimpLoader::resolveUniqueNodeName`
  aynen taşındı; oradaki yorum zaten şunu söylüyordu: *"ikinci düğüm sessizce
  birincinin transformunu kullanır ve yanlış dünya konumuna yerleşir."*
- **`transform->base = M` yerine `setBase(M)`.** Ham atama `position/rotation/scale`
  bileşenlerini 0/0/1'de bırakıyordu: render doğru, **panel yalancı**, ve ilk gizmo
  dokunuşu objeyi orijine ışınlıyordu.

**Faz 1'de KAPSAM DIŞI kalanlar (bilerek):** `.glb`/`.gltf` için Assimp'i baypas et.
- `EXT_mesh_gpu_instancing` — **yazıcı bunu ÜRETİYOR**, yani RayTrophi'nin kendi
  export ettiği bir scatter sahnesi şu an geri okunduğunda instance'ları
  gelmez. Faz 2'nin ilk maddesi olmalı.
- Morph target / blend shape.
- CUBICSPLINE animasyon: değerleri örnekleniyor, teğetler yok sayılıyor
  (LINEAR gibi davranıyor) — ve bu bir uyarı satırı basıyor, sessiz değil.
- Ekstra UV setleri (`TEXCOORD_1+`), vertex renkleri, tangent'lar.
- KHR_texture_transform (yazıcı üretiyor, okuyucu henüz uygulamıyor).

★ Skinli mesh'ler Assimp yolundaki gibi **yüz başına facade** üretmeye devam
ediyor. Renderer'ın import-flat collapse'ı onları dışarıda bırakıyor ("SoA
skinning is a later increment"); okuyucunun bundan sapması, hangi mesh'lerin
flat'e gittiğini okuyucu değiştirmenin yan etkisi olarak değiştirirdi.

**Faz 2 — Kapsam eşitleme.** skin, animasyon, instancing
(`EXT_mesh_gpu_instancing`), punctual lights, kamera, materyal uzantıları.
Doğrulama zaten var: `scripts/validate_gltf_export.py` + export→import
gidiş-dönüşü.

**Faz 3 — Assimp'i FBX/OBJ/DAE'ye indir.** ★ **Tamamen SÖKME.** FBX kullanılan
bir yetenek ve `aiProcess_GlobalScale` özel durumu var. Kural 5 ("ölü yolu sök")
burada geçerli değil, çünkü yol ölü değil.

---

## ★★★ KÜTÜPHANE KARARI (2026-09-05) — artık karar VERİLDİ

Önceki revizyonda "karar verilmedi, tartışılacak" yazıyordu. Aşağıdaki her madde
bu depoda doğrulandı.

### Kapsam önce daralt: DAE YOK

Import dosya filtresi (`scene_ui_menu.hpp:609`) tam olarak şu:

    *.gltf;*.glb;*.fbx;*.obj

**DAE hiç sunulmuyor.** Yani Assimp'in tamamen sökülmesi üç format demek, dört
değil. Bu, "tamamen sökme" kararını önceki revizyonda sanıldığından ucuz yapıyor.

### Faz 1 (glTF/GLB): **cgltf** — port ağaçta, v1.14

| Aday | Karar | Gerekçe |
|---|---|---|
| **cgltf** | ✅ **SEÇİLDİ** | JSON'u tanımlayıcılara (accessor/bufferView) ayrıştırır ve sana `.bin` bloğuna işaretçi verir. Kendi nesne grafiğini KURMAZ. Bu, `GltfDirectWriter`'ın tersi: aynı bellek-düzeni=disk-düzeni mantığı. Tek header + tek `.c`, MIT, `vcpkg/ports/cgltf` zaten ağaçta |
| fastgltf 0.8.0 | dürüst alternatif | Modern C++17, SIMD base64, daha iyi hata raporu. Ama bu kullanım için kazancı küçük ve derleme maliyeti ~1M satırlık bir projede gerçek. **Ancak** base64 gömülü `.gltf` ayrıştırma süresi bir ÖLÇÜMDE görünürse geç |
| tiny_gltf | ❌ **TUZAK** | Ağaçta olması onu cazip gösteriyor — değil. (a) Her tamponu `std::vector`'a çözer ve kendi mesh/primitive grafiğini kurar: export tarafının dakikalar + 15 GB ödeyerek kaçtığı **tam olarak o üçüncü kopya**. (b) Ağaçtaki kopya ozz'un offline aracına ait **özel vendored** bir dosya (`external/ozz-animation/src/animation/offline/gltf/extern/`); ona bağlanmak import'u üçüncü parti bir submodule'ün iç dizinine bağlar |

★ cgltf'in `cgltf_accessor_unpack_floats` / `cgltf_accessor_read_index`
yardımcıları normalize int, stride ve sparse accessor gibi çirkin durumları
kapatıyor — ama sıkı paketlenmiş float durumunda (yaygın olan) seni onlardan
geçmeye ZORLAMIYOR, doğrudan işaret edebiliyorsun. Okuyucunun hızlı yolu bu.

### Faz 3 (FBX): **ufbx** — vendored, çünkü vcpkg portu YOK

| Aday | Karar | Gerekçe |
|---|---|---|
| **ufbx** | ✅ **SEÇİLDİ** | MIT, tek `.c` + `.h`. Skin cluster, blend shape, animation stack/layer kapsıyor. ★ Ve `target_axes` / `target_unit_meters` ile **kendi birim/eksen dönüşümünü yapıyor** — bu, brief'in "sökmeyi engelliyor" dediği `aiProcess_GlobalScale` özel durumunun doğrudan karşılığı |
| OpenFBX | ❌ | Portu ağaçta var (2024-05-08) ama animasyon/skinning kapsamı ufbx'ten belirgin şekilde zayıf. Bu depoda FBX riglenmiş karakter için kullanılıyor, yani tam o zayıf tarafta |
| Autodesk FBX SDK | ❌ **LİSANS** | `LICENSE.txt` = MIT. SDK'nın yeniden dağıtım şartları MIT dağıtılan bir uygulamayla uyumlu değil. Başka bir ajan bunu önerirse cevap budur |

### OBJ: kütüphane alma

OBJ'yi Assimp'e bağlı tutmak, önemsiz bir format için tüm Assimp bağımlılığını
(`assimp-vc143-mt.lib`) canlı tutmak demek. Ya doğrudan flat SoA'ya yazan bir
ayrıştırıcı yaz (~200 satır) ya da tinyobjloader (MIT, tek header). Bu, Assimp'in
son kullanıcısı olduğu için **`.lib`'i düşürmenin son adımı**.

### ★★★★ Sıralama tavsiyesi: FBX'i EN SONA bırak

Assimp'in gerçek değeri FBX'te. glTF temiz bir spesifikasyon; FBX 25 yıllık,
üretici tuhaflıklarıyla dolu bir ikili format ve Assimp'in FBX okuyucusu bu
projenin **zaten başarıyla import ettiği** dosyalara karşı denenmiş.

Ve kazanç en düşük orada: `.rtp` paketi modelleri `models/0001/*.glb` olarak
saklıyor, yani **glTF sıcak yol**, FBX tek seferlik bir import.

    Faz 1 (cgltf)  →  Faz 2 (kapsam eşitleme + parity ölçümü)  →  ÖLÇ  →  sonra FBX

Assimp `.lib`'i FBX ve OBJ kanıtlanana kadar linkli kalsın. Bağımlılığı düşürmek
son adım, ilk adım değil.

### ★★★ Okuyucuyu yazacak ajanın bilmesi gereken TUZAK

`DNA::GeometryDetail::skin_weights` şu tipte:

```cpp
std::vector<std::vector<std::pair<int, float>>> skin_weights;   // köşe BAŞINA vektör
```

glTF `JOINTS_0`/`WEIGHTS_0`'ı **sabit 4 genişlikte** verir. Naif bir okuyucu
1M köşeli bir karakterde **1M küçük heap ayırması** yapar — export tarafının
bu turda kaçtığı maliyet sınıfının aynısı, bu sefer import tarafında.

`indices` ise gerçek bir düz indeks tamponu
(`std::vector<uint32_t, AlignedAllocator<32>>`), yani okuyucu indeksli glTF
primitive'ini **doğrudan** yazabilir; üçgen açmaya gerek yok.

---

## Ölçüm/kabul (kural 1 ve 3)

- Import'un IPC yüzeyi olmalı: dosya yükle → nesne/üçgen/skin/anim sayıları geri
  dön. Aksi halde import **test edilemez** sayılır.
  **✅ Animasyon yarısı açıldı (2026-09-05):** `anim.source_clips` ve
  `anim.source_channels` YÜKLEYİCİNİN ürettiğini raporluyor — bir karakterin
  `AnimationController` durumunu değil (o `anim.clips`). Klip başına kanal/anahtar
  sayısı, tick cinsinden süre ve anahtar zaman aralığı. `scene.export_estimate`
  de eklendi, böylece panelin ÖN tahmini ile yazıcının ÖLÇÜLEN sonucu script'ten
  karşılaştırılabiliyor.
  ★ `scripts/probe_import_export_parity.py` bu sayıları bir temel dosyaya yazıp
  sonraki koşuda diff'liyor. Faz 0 saf tip göçü olduğu için **bu sayıların hiçbiri
  oynamamalı**; Faz 1'de oynayan her sayının açıklanması gerekir.
- Kabul testi: aynı `.glb` için Assimp yolu ile yeni yol **aynı** vertex/üçgen/
  joint/kanal sayısını ve aynı bbox'ı vermeli. Fark varsa hangi tarafın doğru
  olduğu ölçülerek kararlaştırılmalı — "yeni yol farklı, demek ki daha iyi"
  denmemeli.
- ★ Bu turda öğrenilen ders geçerli: **kare boyutu / sayı dizisi, göz kararından
  üstün.** Import için karşılığı, iki yolun sayaç tablosunu yan yana basmak.

---

## Faz 0 sonrası SIRADAKİ

1. **Tam rebuild + `NEXT_BUILD_CHECKS.md` §3.** Faz 0 derlenmedi; dört dosya
   silindi/yeniden adlandırıldı ve `BoneData` **ortadan** küçüldü — kısmi
   derlemede eski düzene göre derlenmiş bir çeviri birimi ekleme noktasından
   sonraki her alanı yanlış offset'ten okur.
2. **Temel çıkar:** aynı `.glb`/`.fbx` ile `probe_import_export_parity.py --write`.
   Faz 1'in kabul testi bu dosya.
3. Faz 1'e ancak (1) ve (2) bittikten sonra geç.

★ `scene_ui.cpp:~290`'daki `std::function<void(aiNode*, const aiMatrix4x4&)> walk`
Faz 0 kapsamında DEĞİL ve bilerek bırakıldı: kendi yerel `Assimp::Importer`'ı
içinde, tek fonksiyon kapsamında bir asset bbox sondası. Assimp FBX/OBJ/DAE için
kalacağına göre (Faz 3) bu sınırlı kullanım meşru.

---

## Devretmeden önce KAPANMAMIŞ iş (2026-09-04 oturumundan)

`docs/dev/NEXT_BUILD_CHECKS.md` §0a **henüz koşturulmadı.** Kullanıcı tam
rebuild yapmadı ve bu partide dört struct **ortadan** büyüdü
(`VDBVolumeData`, `VulkanDevice`, `VulkanBackendAdapter` — üçünde de ekleme
noktasından sonra alan var). Kısmi derlemede eski düzene göre derlenmiş bir
çeviri birimi, ekleme noktasından sonraki her alanı yanlış offset'ten okur ve bu
"aynı sorunlar duruyor"dan **ayırt edilemez**.

⇒ Hacim/viewport konusuna dönüldüğünde ilk iş **tam rebuild + §0a kare-boyutu
dizisi**. O ölçüm alınmadan yeni hipoteze geçilmemeli.
