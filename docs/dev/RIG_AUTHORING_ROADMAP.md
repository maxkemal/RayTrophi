# Rig ve hareket yazarlığı — görsel iskelet, auto-rig ve animasyon

> **Status (2026-09-16):** Multipart fit/bind, rest mirror, Pose/FK, hand/foot IK,
> joint overlays and the first in-place humanoid walk are user-confirmed. Timed
> IK/contact/bake, chain/spline/aim controls and gait body-motion refinements are
> source-delivered through common UI/API/Python/IPC, with remaining runtime gates
> recorded below. Live inspection found normalized but anatomically broad
> arm/clavicle weights; category-level bounded capsule reweight preview/apply is
> reweight and pose-following viewport capsule preview are source-delivered and
> await user deformation checks. Next: per-bone radius overrides/handles,
> footprint/contact phases and root-motion spline walk/run. New
> test scripts remain deferred. Weighted deletion/remap/transfer, general weighted
> rest editing and visibility/interior or heat weighting remain open.

Hedef: insan, hayvan ve böcek karakterler için güçlü bir **iskelet ve hareket
yazarlığı sistemi** kurmak. Kullanıcı modeli seçer, anatomik şablonu yerleştirir,
eklemleri viewport'ta düzeltir, ağırlıkları üretir ve pozlar veya adım hedefleri
üzerinden animasyon hazırlar. Ajan aynı işlemleri script ve IPC ile hızlıca
yapabilir; ürettiği rig ve hareket kullanıcı tarafından görsel olarak düzenlenebilir.

**2026-09-12 kapsam güncellemesi:** İlk teknik fazlar korunmuştur. İnsan dışı
şablonlar, IK/FK, poz ve klip yazarlığı, adım tabanlı hareket ve retargeting artık
ürün hedefinin parçasıdır; aşağıdaki ek fazlar tasarımdır, uygulanmış özellik değildir.

Not iki şey içeriyor: (1) kod okunarak **doğrulanmış** mevcut durum ve onun
yapısal sorunları, (2) faz planı. Faz planı tasarımdır ve değişebilir;
1. bölümdeki bulgular kod okumasıdır ve her birinin dosya:satır dayanağı var.

---

## 1. ★ Önce bulunan sorunlar

These are findings from the initial source audit. Dated status notes below
represent current delivery; the original explanations are historical rationale.
A resolved issue must not be read as an open task. Original impact order is retained.

### 1.1 ★★★★ GPU İLK dört influence'ı alıyor, EN BÜYÜK dördü değil — ve yeniden normalize etmiyor

**Status (2026-09-13): RESOLVED FOR IMPORT PRODUCERS; UNIVERSAL WRITER COVERAGE PARTIAL.**
New glTF/ufbx/Assimp flat imports share SkinWeightContract duplicate merge,
strongest-four selection, sort and normalization. User confirmed build/runtime.
weight_stats/get_weights exist in source. Native/copy/topology repair and future
live weight producers are not declared universally complete.

**Initial audit rationale:**

`GeometryDetail::skin_weights` vertex başına **sınırsız** influence tutuyor
(`std::vector<std::pair<int,float>>`, `DNA/GeometryDetail.h:311`). Dört tüketici
bunu farklı okuyor:

| Tüketici | Davranış |
|---|---|
| CPU skinning (`Scene/TriangleMesh.cpp:35`) | **Hepsini** toplar, `totalWeight` ile **yeniden normalize eder** (`:79`) |
| Vulkan RT BLAS (`Backend/VulkanBackend.cpp:8342`) | `influence < 4` — ilk dördü, **sıralamadan**, normalize etmeden |
| Raster viewport (`Backend/VulkanViewportBackend.cpp:6803`) | aynı: ilk dördü, sıralamadan |
| `shaders/skinning.comp` | tam 4 slot; toplam < 0.001 ise identity. Başka **normalizasyon yok** |

Yani bir vertex'in influence'ları `[(5,0.1), (9,0.1), (2,0.1), (7,0.1), (3,0.6)]`
ise CPU render doğru deforme eder; GPU **toplam ağırlığı 0.4 olan** bir karışım
uygular ve vertex bind pozuna doğru çöker.

Initial assumptions that imports already honored the contract proved incomplete:
ufbx could retain >4 influences and glTF did not renormalize after filtering.
Shared import-producer normalization now closes these two gaps.

**Neden sinsi:** ilk prosedürel ağırlık üreticisi (ısı difüzyonu vertex başına
8-20 influence üretir) bu kuralı ilk ihlal eden olacak, ve belirti "ağırlıklar
yanlış" değil **"viewport ile render farklı görünüyor"** olacak — yani
ağırlıkları değil, backend'leri şüpheli yapan bir belirti.

**Karar:** budama + yeniden normalizasyon **üreticinin** sözleşmesidir,
tüketicinin değil. `skin_weights` yazan her yol en fazla N (varsayılan 4)
influence, ağırlığa göre azalan sıralı, toplamı 1 bırakmak zorunda. Kural
yazıldığı anda bir tripwire ister: `rig.weight_stats` → `max_influences`,
`min_weight_sum`, `max_weight_sum`.

### 1.2 - Bone indices live inside weights: weighted deletion needs remap/transfer

**Status (2026-09-13): PARTIALLY RESOLVED.** Current deletion only supports
owned meshless/unweighted clip-free leaf joints. Surviving scene bone indices
remain stable; getBoneIndexCapacity supports gaps. This subset does not remap
weights or transfer deleted influences because weighted editing is out of scope.

skin_weights stores BoneData::boneNameToIndex indices, not names. Compacting indices
without updating vertex rows or dropping a deleted influence without renormalizing
silently changes deformation; CPU/GPU out-of-range checks can conceal that error.

**Delivered:** RigEditing::stageDeleteRigBone in source/src/Animation/RigEditing.cpp
stages canonical hierarchy/BoneData removal with common finish/undo, root/leaf,
cross-rig and anatomy guards. RigWeights::hasFlatSkinReferences now checks actual
flat scene weight buffers for any owned hierarchy index reference, even if skin
metadata is missing. Rest/topology staging, Edit entry and actor placement reject
such references, including zero/invalid entries and malformed extra rows. Checks
run at mutation staging/Edit entry, not on every inspector/viewport frame.

**OPEN:** weighted/imported deletion, atomic global remap/slot policy, parent weight
transfer, shared top-four normalization, CPU/GPU invalidation and undo/reopen
regression. weight_stats detects unknown/ambiguous/foreign indices; detection is
not repair. Do not close all of 1.2 before these operations are delivered.

### 1.3 - Overlay must use joint globals, not getFinalBoneMatrices skinning output

**Status (2026-09-13): RESOLVED IN SOURCE.** Controller exposes the pre-offset
cache through getJointGlobalTransforms (source/include/AnimationController.h).
Graph FinalPoseNode also produces jointGlobalTransforms. Renderer captures graph,
controller and ozz globals through RigView::captureGlobals/captureRuntimeGlobals;
ozz capture happens before skin inverse/offset conversion. RigView::listBones uses
bind/joint globals plus actor/mesh placement, never skinning matrices as joint
positions. UI/Python/IPC share that BoneView data path.

**Initial audit rationale:**

    boneMatrix = globalInv * animatedGlobal * offset

Skin matrices transform vertices. Joint visualization needs animatedGlobal before
offset, composed with external placement. Using skin matrices can appear correct
in bind while moving the overlay outside the mesh during playback. The previously
missing accessor and snapshot wiring now exist. This closes the data separation
issue, not Pose/IK or weighted rest-edit authoring.

Evidence: source/src/Animation/AnimationController.cpp, AnimationNodes.cpp,
RigView.cpp, and source/src/Render/Renderer.cpp (graph/controller/ozz capture).
Acceptance remains animated joint world positions independent of skin offset,
including Rest/Animated and actor placement checks.

### 1.4 - Rig edits must refresh derived representations together

**Status (2026-09-13): PARTIALLY RESOLVED.** Owned unweighted authoring has a
common rebuild/native snapshot path. Initial "no rebuild path" findings no longer
apply to this subset. Initial owned actual-flat binding now has a common
weight/offset/membership/geometry transaction with undo and backend invalidation.
General weighted edit/rebind is still open.

| Representation | Current delivery | Remaining boundary |
|---|---|---|
| BoneData | Common RigEditing::finish derives local/default/parent/index/offset records from canonical owned NodeHierarchy; offset = inverse(global rest) | Imported weighted bind and weight/index migration open |
| skeletonNodes | Same finish calls rebuildSkeletonRepresentation | No shared weighted-edit transaction yet |
| NodeHierarchy | Canonical owned rest/topology state with stage/undo/native serialization | Not a general imported weighted hierarchy rebind service; legacy fallback cannot recover all missing scene data |
| Ozz AnimationSet | Same finish builds runtime skeleton/index-joint mapping; restoreOwnedRigRuntime handles static owned reopen | Clip migration/weighted topology changes open |
| CPU/GPU skin/influence geometry | Import normalization; initial RigBinding flat bind P/N and influence transaction, CPU BVH/backend invalidation, undo/redo | General live weight edit/delete/remap and weighted rest rebind remain open |

Evidence: source/src/Animation/RigEditing.cpp (finish, restoreOwnedRigRuntime),
RigSerialization.cpp, source/src/Api/RtApiRigEditing.cpp (common undo),
source/src/Core/ProjectManager.cpp (native hierarchy save/open), and
source/src/Backend/VulkanBackend_Pose.cpp (existing pose replay).

rig.list_bones reports four separate representation flags. Owned unweighted
create/add/rest/reparent/delete, undo/redo and native reopen must preserve all
four. Canonical hierarchy-to-derived-state flow is deliberate; no second hierarchy/
BoneData business-logic copy is introduced. Replaying cached pose is NOT rebuilding
weight buffers. Initial RigBinding now refreshes flat bind P/N, influences,
CPU BVH and GPU backend state together. General weighted edit/delete/remap/rebind
must use a common transaction before 1.4 is fully closed.

### 1.5 ★★ ozz var, ama auto-rig/auto-weight ozz'da YOK

`ozz_animation_offline` zaten vcxproj'da derleniyor — `raw_skeleton.cc`,
`skeleton_builder.cc`, `animation_builder.cc`. Yani **sıfırdan iskelet kurmak
için altyapı hazır** (`RawSkeleton` → `SkeletonBuilder` → runtime `Skeleton`),
ve IK iş yükleri de derleniyor (`ik_aim_job.cc`, `ik_two_bone_job.cc`).

Ama ozz bir çalışma zamanıdır, rigging kütüphanesi değil: kemik yerleştirme de
ağırlık çözümü de yok. O iki algoritma bu depoda yazılacak. Bunu baştan söylemek
gerekiyor, çünkü "ozz'umuz var" cümlesi Faz 4 ve 5'in maliyetini sıfır gibi
gösteriyor.

---

## 2. Initial inventory (historical source audit)

This inventory predates authoring delivery. Use the dated issue status notes and
current phase checkpoint for delivered rig/API/IPC capabilities; this section
retains the original starting point, not the current feature inventory.


**Kullanılabilir durumdakiler:**

- `SkeletonNode` editör temsili — parent/children + `boneIndex` + `weightedBone`
  + **`globalBindTransform` zaten hesaplanmış** (`scene_data.h:385`). Overlay'in
  bind-pozu verisi hazır.
- Salt-okunur iskelet ağacı paneli — `drawSkeletonHierarchyTree`
  (`UI/scene_ui_hierarchy.cpp:73`), `[skinned]`/`[anim]` ayrımını zaten gösteriyor.
- Overlay çizim kalıbı — `drawLightGizmos` / `drawForceFieldGizmos`
  (`UI/scene_ui_gizmos.cpp`), ImGuizmo entegrasyonu, world→screen dönüşümü.
- CPU ve GPU skinning yolları çalışıyor; gereksiz deformasyonu eleyen pose-hash
  kapısı da var (`TriangleMesh.cpp:52`).
- `.rtp` kemik serializasyonu tam: index, offset, local bind, parent, weighted set.

**Eksik olanlar:** 1.4 tablosunun ✘ satırları, ağırlık üreten herhangi bir yol,
kemik seçimi, kemik düzenleme API'si, `rt.rig` namespace'i.

---

## 3. Faz planı

Sıralama ölçütü: **ölçü aleti algoritmadan önce.** Faz 0-2 yapılmadan 3-4
yapılırsa çıktı ölçülemez, ve bu depoda ölçülemeyen şey "makul görünüyor"a
dönüşüp kalibrasyon turuna gömülüyor.

### Faz 0 — Kemik gerçek bir düzenlenebilir varlık olur

Algoritma yok. 1.4'teki üç ✘ satırının kapatılması ve düzenleme gövdesinin kurulması.

**İş:**
- Yeni `source/src/Animation/RigEdit.cpp` (+ **vcxproj kaydı**): `BoneData`
  üstünde `addBone` / `deleteBone` / `renameBone` / `reparentBone` /
  `setBoneBindTransform` / `mirrorBones`.
- `boneOffsetMatrices` **hiçbir zaman elle yazılmaz** — bind pozu her
  değiştiğinde `inverse(globalBind)` olarak yeniden türetilir.
- `NodeHierarchy`'yi `BoneData`'dan kuran tek gövde (okuyucularınkiyle aynı
  sözleşme: `name` + `uniqueName` + `localBind` + parent).
- `AnimationSet`'i yeniden derleyen yol; `sceneBoneToRuntimeJoint` tazelenir.
- `deleteBone`'un indeks remap'i + ağırlık devri (1.2).
- 1.3'ün erişimcisi: eklem dünya transformu dışarı verilir — `globalTransformCache`
  zaten tutuyor.
- `rt.rig` IPC yüzeyi (4. bölüm).

**Kabul aleti:** script'ten kemik ekle → `rig.list_bones` onu **dört temsilde de**
ayrı ayrı raporlasın: `in_bonedata`, `in_skeleton_nodes`, `in_node_hierarchy`,
`in_ozz_skeleton`. Dört bayrak, tek "ok" değil. Projeyi kaydet/aç, hâlâ orada.
Kemik sil → `rig.weight_stats`'ta `min_weight_sum` 1.0'da kalsın.

**Bozuksa ne demek:** dört bayraktan biri false ise o temsilin yeniden kurma yolu
eksiktir — ve onu okuyan yol (ör. ozz örnekleyici) düzenlemeye **yapısal olarak
kör** kalır.

★ **En sinsi başarısızlık:** kemik eklenir, overlay'de görünür, `.rtp`'ye yazılır
— ama `AnimationSet` tazelenmediği için animasyon oynatıldığında hiçbir şey
olmaz. Hata yok, log yok. `in_ozz_skeleton` bayrağı tam olarak bunu ölçmek için var.

**Serialization prerequisite delivery:** Project save/open already persists BoneData,
canonical TRS clips (including bound/retargeted output) and AnimGraph JSON. Audit
found NodeHierarchy missing from context persistence. It is now saved as versioned
node records and restored with authored/unique names, parent indices and exact
local matrices; children are rebuilt. Older files fall back to available BoneData.
New native hierarchy round-trip and legacy compatibility await user verification:
`docs/dev/RIG_SERIALIZATION_BUILD_CHECKS.md`.
Manual mapping options/presets, preview time/camera and overlay selection are not
persisted yet. Do not mark the entire authoring session as serializable.

**Editable-rig first delivery (source, user build pending):** Existing bottom
Animation editor now has a Rig tab. `rig.create` creates an owned meshless Root
or chain3 template; `rig.add_bone` appends a child, `rig.set_rest_transform` edits
local position/rotation. UI, Python and IPC share RigEditing and one undo command.
NodeHierarchy is canonical rest/topology state for owned rigs; BoneData indices,
local/default/global-offset data, skeletonNodes and ozz bridge rebuild together.
`rig.list_bones` reports local_rest_transform, ownership, template, revision and
four separate representation flags. Existing indices remain stable on append.
Project save/open preserves ownership/template/revision and hierarchy; static
owned rigs also rebuild their ozz bridge on reopen.

First pass only edits owned, unskinned rigs without canonical clips. Imported,
weighted and clip-bound rigs are rejected; no scale/shear, reparent/delete, mesh
binding or humanoid auto-placement in this subset. Root/chain height uses scene
units. Next: reparent/delete and anatomical templates, then safe weighted bind
editing through the flat geometry path. Native hierarchy save/reopen correction
and this new authoring pass both await user build/runtime validation.

User requested no new per-step test files until infrastructure is established;
new regression files are deferred. Only static source/XML/descriptor audits are
performed now. User build checklist: Animation > Rig -> create chain3 -> add
child -> select/edit rest -> undo/redo -> save/reopen and check all four flags.

Python/IPC operations: `rig.create(character, template_id="chain3", height=1.8)`,
`rig.add_bone(character, name, parent, rest_transform)` and
`rig.set_rest_transform(character, bone, rest_transform)`. Parent/bone use full
unique keys; transform is a row-major 16-float rigid local matrix, identity default
for add. Rig/joint names use ASCII letters/digits/underscore/hyphen (1-128 chars).
Failures use named service errors; typed parameter/shape errors are binding errors.
Mapping/preview session persistence remains a separate follow-up.

### Faz 0B — Klip importuyla birlikte bağlama ve temel retarget

**Manual mapping delivery:** Optional `node_map` overrides are shared by the
ClipBinding core, hierarchy panel, Python and IPC. Unique node names and full
parent/helper chains are validated; unknown nodes and duplicate targets fail.
Automatic same-rig playback is user-confirmed; manual mapping awaits build/runtime
checks in `docs/dev/CLIP_BINDING_BUILD_CHECKS.md`. This copies absolute local TRS.
Rest-basis delta conversion is now implemented in focused `Animation/Retarget`:
`mode=rest_basis`, explicit `translation_scale`, shared dry-run/bake logic in
UI/Python/IPC. Matching parent chains, positive uniform scales only. See
`docs/dev/RETARGET_BUILD_CHECKS.md`; hierarchy build/playback user-confirmed. Manual panel visibility
is user-confirmed; manual mapping/playback deferred to IPC.

**2D mapping delivery:** The existing bottom Animation editor now has Graph /
Retarget tabs. Source/target skeletons, orthographic views, picking, linked channel
selection, node filtering and synchronized seconds preview use shared services.
`anim.sample_clip_binding` exposes the same poses to Python/IPC without changing
scene playback. Prior rest-basis hierarchy workflow and the new 2D mapping/pose-preview UI
build/runtime are user-confirmed. Dedicated script/IPC regression checks remain
pending in RIG_MAPPING_UI_BUILD_CHECKS.md.

**Planned mapping UI polish (deferred):**

- Mouse-wheel zoom anchored at the cursor, preserving the point under the pointer.
- Middle-mouse drag to pan the mapping canvas; keep Fit as a reset action.
- A subtle, hover-revealed draggable splitter between the left controls and canvas
  (the user's "ghost scrollbar" suggestion). Drag changes panel width, with
  minimum widths for readable controls and a usable canvas.
- Canvas gestures remain local to the editor and do not trigger viewport controls
  or change bone mappings. Preserve the current bottom editor dock/float layout.

These are planned UX refinements, not implemented or required before the next
rig/template infrastructure milestone.

**Next authoring milestone:** editable canonical rig and skeleton template creation
(Phases 0/4/6), with script/IPC operations and flat-data bind/weight contracts.
Mapping presets/revisions and rest alignment remain UI follow-ups; advanced
contact IK does not gate fast skeleton creation.


**2026-09-12 ilk bağlama teslimi:** Kullanıcı aynı Mixamo karakterinin skinli
ve skinsiz iki exportunu import etti; ikinci klip ayrı karakter/prefix taşıdığı
için skinli hedefte oynamadı. Yeni `Animation/ClipBinding` aynı exported rig için
authored-name/ancestor eşleme, preflight raporu ve kaynak korunarak target-bound
klip kopyası üretir. Hedef controller/ozz runtime sahne değişmeden stage edilir;
bind undo/redo destekler. UI mevcut karakter hierarchy bölümündedir;
Python/IPC `anim.preview_clip_binding` ve `anim.bind_clip` aynı çekirdeği kullanır.
Bu same-rig doğrudan TRS transferidir; rest farkları raporlanır, düzeltilmez.
Genel retarget solver ve manuel role mapping hâlâ bekliyor. Build/runtime
checklist: [`CLIP_BINDING_BUILD_CHECKS.md`](CLIP_BINDING_BUILD_CHECKS.md).

Bu faz Faz 10'a ertelenmez: dış animasyon kullanımı için Faz 0'ın bağımsız rig
varlığı üzerinde kurulur. Önce meshsiz kaynak iskelet+klip korunur; sonra hedef
rig'e kanal bağlama ve aynı anatomik aile için temel retarget teslim edilir.

Üç açık yol vardır: aynı iskelete doğrudan kanal bağlama; farklı ama aynı
anatomide role-based retarget; uyumsuz anatomi için desteklenmeyen eşlemeyi
raporlama. Kemik adı benzerliği yalnız eşleme önerisi üretir; rest pose, parent
ilişkisi ve kanal uzayı doğrulanmadan klip hedefe bağlanmaz.

Ortak `Animation/Retarget` servisi kaynak/hedef rest-global transformlarını,
local rotation basis farklarını, root yön/ölçek dönüşümünü ve translation
politikasını açıkça ele alır. Quaternion/TRS ve world/local uzayı sözleşmeleri
fixture'larla doğrulanır. Kaynak rig değişmez; sonuç hedef rig'e bağlı yeni
canonical klibe bake edilir. Mevcut oynatıcıların tükettiği klip ve runtime
cache'leri tek commit yoluyla yenilenir.

UI: kaynak rig/klip → hedef rig → otomatik rol eşleme → elle düzeltme → rest
pose hizalama → eş zamanlı hareket önizlemesi → yeni klibe bake. Eşleme preset'i
sürümlü kaydedilir; kaynak ve hedef bone revision'ı değişince tekrar doğrulanır.
Script ve IPC aynı preview/commit servisini kullanır; hata/eksik rol listesi,
ölçek, eşleme kapsamı ve bake sonucu makine-okunur döner.

**Kabul:** identity retarget pozu tolerans içinde korur; farklı isim ve bone
index sırasındaki aynı rig doğru eşlenir; A/T rest-pose farkı, farklı root
ekseni/ölçeği ve farklı uzuv oranları fixture'ları kontrol edilir. Unknown bone,
duplicate target, uyumsuz hierarchy ve stale revision mutasyonsuz reddedilir.
Bake, preview ve save/reload eşleşir. Ayak temasını koruyan IK iyileştirmesi
Faz 7/10'da genişler; ilk temel retarget ayak kaymasını raporlar.

### Faz 1 — 4-influence sözleşmesi + ölçü aleti

1.1'in kapatılması. Kemik düzenlemeden bağımsız; bugün de yapılabilir.

**İş:** `skin_weights` yazan her yol için tek normalizasyon/budama gövdesi (en
büyük N, azalan sıralı, toplam 1). GPU tarafındaki üç kopya o sözleşmeye
**dayanır**, kendi budamasını yapmaz. `rig.weight_stats` +
`rig.get_weights(object, vertex)`.

**Kabul aleti:** mevcut bir `.glb` import et → `max_influences <= 4`, her
vertex'te `weight_sum` 1±1e-4. Sonra elle 5. influence yaz → üretici sözleşmesi
onu budasın, tripwire sussun.

**Bozuksa ne demek:** `weight_sum < 1` görüyorsan GPU o vertex'i bind pozuna
doğru çekiyordur — ve bunu CPU render'da göremezsin.

### Faz 2 — Kemik seçimi, overlay ve inspector

#### 2A — Çoklu kemik seçimi ve toplu rest düzenleme

**2026-09-13 kaynak teslimi tamamlandı; kullanıcı build/runtime kabulü bekliyor.**
Ortak çoklu seçim, aktif kemik, pivot, viewport/hiyerarşi etkileşimi, Rig Edit kutu
seçimi ve tek undo adımlı toplu rest Move/Rotate teslim edildi. UI/Python/IPC aynı
servisleri kullanır. Mirror 2B ve ağırlıklı rest/topoloji düzenleme bu teslim değildir.

**Ürün sözleşmesi:** Viewport ve
hiyerarşide Ctrl ile ekle/çıkar, Shift ile aralık seçimi ve kutu seçimi; tek rig
içinde seçili kemik kümesi ve ayrı aktif kemik ortak scene seçim servisinde tutulur.
Kalıcı unique kemik kimlikleri kullanılır; skinning indeksleri seçim kimliği olmaz.
Mevcut tek-kemik işlemleri aktif kemikle uyumlu kalır. Inspector, gizmo ve ağırlık
haritası aktif kemiği gösterir; diğer seçili kemikler ayrıca vurgulanır.

Toplu Move/Rotate için aktif kemik veya seçim merkezi pivotu açıkça seçilir.
Parent ve child birlikte seçildiğinde dünya uzayındaki hedefler önce hesaplanır,
local rest dönüşümleri hedef parent uzayına çevrilir; child iki kez taşınmaz.
Escape tüm işlemi iptal eder; tek drag tek undo adımıdır. İlk kapsam mevcut
owned/unskinned/clip-free rest edit sınırıdır; weighted rebind kapısı açılmadan
bağlı rig'in rest düzenlemesi serbest bırakılmaz. Toplu silme/reparent ayrı
topoloji işlemidir ve 1.2'nin ağırlık aktarma/remap koşullarını devralır.

**Kabul:** viewport/hiyerarşi/UI/Python/IPC aynı seçim ve aktif kemiği raporlar;
parent+child seçimi, pivot, iptal, undo/redo ve rig değişiminde stale seçim kontrolü.
Seçim geçicidir; yapılan rest değişiklikleri native snapshot'ta korunur.

#### Planlanan 2B — Mirror rest düzenleme ve simetrik kemik oluşturma

Çoklu seçim ve ortak toplu rest işleminden sonra teslim edilir. Mirror düzlemi
rig model uzayında tanımlanır (varsayılan X=0); actor'ın dünya rotasyonu/ölçeği
düzlemi değiştirmez. Anatomik pair metadata eşleşmenin otoritesidir; ad son eki
tahmini yerine eksik eşler kullanıcıya önerilir ve onaylı eşleme kaydedilir.

İki işlem ayrılır: mevcut karşı tarafa **mirror düzenleme** (tek seferlik veya
canlı simetrik edit) ve eksik karşı uzvu **mirror oluşturma**. Sol→sağ/sağ→sol,
merkez kemik davranışı ve iki eşin aynı anda seçilmesindeki kaynak taraf açık olur;
çift uygulama yapılmaz. Joint konumu ve yönelimi birlikte aynalanır; reflection
matrisi veya negatif scale doğrudan local rest olarak saklanmaz. Parent uzayları,
kemik eksenleri, pair/chain metadata ve yeni kimlik/indeksler birlikte doğrulanır.
Oluşturma kaynak kemiğin kimliğini/skin indeksini kopyalamaz; bound/weighted
topoloji ve rebind sınırlamaları aynen geçerlidir.

**Kabul:** hareket etmiş/dönmüş actor altında düzlem korunur; farklı parent
uzaylarında eşlenmiş zincir, merkez kemik, eksik/çakışan eşleme ve iki taraflı seçim
kontrol edilir. Mirror iki kez uygulandığında rest tolerans içinde geri döner;
UI/Python/IPC, undo/redo, dört rig temsili ve native reopen sonuçları eşleşir.
Önerilen yeni seçim/toplu/mirror API adları tasarım aşamasında kesinleştirilecek;
mevcut `rig.mirror` tasarım satırı uygulanmış operasyon sayılmaz.

**2026-09-12 uygulama durumu:** Görünüm/seçim alt adımı kaynakta teslim edildi:
`Animation/RigView` ortak servis; `SceneData::rigView` ortak seçim/visibility;
AnimGraph Final Pose, controller ve ozz'dan skin offset öncesi joint-global
snapshot; yeni `UI/scene_ui_rig_overlay.cpp` çizim/picking; `UI/RigViewUI.cpp`
hiyerarşi seçimi ve read-only seçim özeti. `rt.rig` / `rig.*` list_characters,
list_bones, select_bone, get_selected_bone, clear_selection ve overlay visibility
yüzeyleri aynı çekirdeği kullanır. Descriptor/capability audit PASS (448 metot,
34 namespace); source registration ve Python syntax kontrolleri PASS.
Bu teslim Faz 0/1/2'nin tamamını kapatmaz: bind edit/gizmo, ağırlık özeti ve
iskelet authoring operasyonları hâlâ bekliyor. Kullanıcı build/runtime checklist:
[`RIG_VIEW_BUILD_CHECKS.md`](RIG_VIEW_BUILD_CHECKS.md). Canlı test yapılmadı.

**Kullanıcı doğrulaması:** Sonraki build'de kemik overlay çizimi, kemik seçimi
ve overlay görünüm aç/kapat kullanıcı tarafından çalışır olarak doğrulandı.
Animasyon overlay, tüm runtime yolları ve save/reopen checklist'i ayrıca izlenir.

**İş:**
- Alt-seçim kanalı: `(character, boneIndex)`. ★ Bu durum **sahnede** yaşar,
  panelde değil — yoksa IPC'den okunamaz ve panel tekrar otorite olur.
- `drawSkeletonOverlay()` → `scene_ui_gizmos.cpp`; oktahedron/çizgi, seçili kemik
  vurgulu, weighted/anim ayrımı renkte. Poz kaynağı **1.3'ün erişimcisi**,
  `getFinalBoneMatrices()` değil.
- Hiyerarşi ağacına tıkla-seç + sağ tık menüsü (ekle/sil/adlandır) + sürükle-reparent.
- Kemik inspector'ı: bind transform, parent, ağırlık özeti. ★ Ters yön kuralı:
  `rig.set_bind_transform` script'ten yapılabiliyorsa panelden de yapılabilmeli,
  yoksa script'ten kurulan rig panelden düzeltilemez.
- ImGuizmo'yu seçili kemiğin global matrisine bağla.

**Kabul aleti:** `rig.select_bone` → `viewport.capture` + `get_screenshot` ile
overlay'in o kemiği vurguladığı görsel olarak doğrulanır.
`rig.get_selected_bone` **değer** döndürür.

★ **En sinsi başarısızlık:** overlay rest pozunda doğru, animasyonda kayık (1.3).
Kabul testi bu yüzden **animasyon oynarken** yapılır, bind pozunda değil.

### Faz 3 — Otomatik ağırlık, mesafe yolu

İlk prosedürel üretici. Kasten basit olan önce, çünkü Faz 1'in tripwire'ını ilk
zorlayacak olan bu.

**İş:** kemik segmentine mesafe⁻ᵖ + görünürlük testi (vertex'ten kemiğe giden
ışın mesh'in içinden mi geçiyor — kolun gövdeye yapışmasını bu engeller). Çıkış
Faz 1 sözleşmesinden geçer.

**Kabul aleti:** `rig.auto_weights` sonrası `weight_stats` temiz; bir kol kemiğini
döndür → gövde vertex'lerinin yer değiştirmesi eşiğin altında kalsın. Bu sayıyı
**görünürlük testi** üretir, mesafe üretmez — yani testin ölçtüğü şey, yolun
zor olan yarısıdır.

### Faz 4 — Otomatik kemik yerleştirme (humanoid şablon)

**İş:** kanonik humanoid şablon iskeleti (repo'da asset); mesh'in bbox'ı +
simetri düzlemi + yükseklik dilimlerinden çıkan siluet genişlik profiline göre
ölçekle ve eklemleri kaydır, medial-eksen kısıtıyla iç hacimde tut. Pinocchio
(Baran-Popović) yaklaşımının pratik yarısı.

**Ön koşullar — açıkça raporlanır, varsayılmaz:** T/A-poz, up-axis ve yön
tespiti, kabaca kapalı yüzey. Sağlanmıyorsa **reddet**, tahmin etme.

**Kabul aleti:** eklem başına oturma artığı (fit residual) + "eklem iç hacimde mi"
testi, `rig.auto_rig` çıktısında sayı olarak.

★ **En sinsi başarısızlık:** iskelet makul görünür ama bir eklem yüzeyin
dışındadır; ağırlık üreticisi oradan itibaren saçmalar ve suç ağırlıklara atılır.
"İç hacimde mi" testi tam olarak bu ikisini ayırmak için var.

İnsan dışı anatomik şablonlar Faz 6'da ele alınır. Genel mesh contraction → curve
skeleton yolu daha sonraki araştırmadır: tek başına anatomik rol ve retargeting
eşlemesi sağlamaz.

### Faz 5 — Isı difüzyonu ağırlıkları

İç voxel gridi üzerinde Laplace çözümü; collider voxelizasyon önbelleği gridi
üretebilir, üstüne CG. Kemik başına bir sağ taraf.

Faz 3'ün kabul aleti burada da geçerli ve bir karşılaştırma sayısı verir: aynı
sahnede mesafe yolu vs ısı yolu, aynı eşikle.

---

## 4. `rt.rig` — kural 1'in beş dokunuşu

Mevcut `rt.anim` bilerek yalnızca playback (`Api/RtApi.h:2030`: "deliberately the
PLAYBACK + PARAMETER half only"). Rig düzenlemesi ayrı namespace ister.

    rig.list_bones / rig.get_bone
    rig.add_bone / rig.delete_bone / rig.rename_bone / rig.reparent
    rig.set_bind_transform / rig.mirror
    rig.select_bone / rig.get_selected_bone        ← DEĞER, panelden okuma değil
    rig.auto_rig(character, template="humanoid")   → fit residual raporu
    rig.auto_weights(object, method, max_influences)
    rig.get_weights / rig.set_weights / rig.weight_stats

| Katman | Dosya |
|---|---|
| Çekirdek API | `source/include/Api/RtApi.h` + yeni `source/src/Api/RtApiRig.cpp` (**vcxproj'a ekle**) |
| IPC dispatch | `source/src/Api/RtIpc.cpp` |
| Python binding | `source/src/Api/RtPython*.cpp` |
| Yetki | `source/src/Api/RtIpcSecurity.cpp` — **unutulursa metot sessizce reddedilir** (fail-closed) |
| Ajan tarifi | `python scripts/gen_ipc_descriptors.py` + `scripts/ipc_descriptor_overlay.json` |

Yeni namespace eklendiği için `scripts/audit_ipc_capabilities.py` çalıştırılmalı.

---

## 5. Görsel üretim akışı ve ortak veri sözleşmesi

Ana akış: **Model seç → Anatomi seç → İşaretleri yerleştir → İskeleti önizle ve
düzelt → Bağla → Poz ver → Hareket ekle → Klip olarak kaydet.** Kullanıcı her
adımda geri dönebilir; önizleme sahneye bağlanmadan iptal edilebilir.

- UI yerleşimi `TEMPLATE_HUB_UX_ROADMAP.md` yönünü izler: viewport merkezde;
  sol bağlam rayında Rig/Animate araçları; sağ bağlamsal dock'ta aktif adım,
  anatomi, inspector ve kalite raporu; alt editörde timeline, klipler ve adımlar.
  Büyük kalıcı shelf veya ayrı bir ana çalışma yüzeyi eklenmez.
- Viewport'ta eklem işaretleri, kemikler, simetri düzlemi, IK hedefleri ve ayak
  temasları düzenlenir. Rest/bind düzenleme ile animasyon pozu ayrı modlardır;
  poz vermek bind matrislerini değiştirmez.
- Tek ortak servis: UI, Python ve IPC aynı doğrulama ve mutasyon gövdesini çağırır.
  Geometri kaynağı yalnız flat `TriangleMesh` / DNA SoA'dır. Şablon ve hareket
  tanımları otorite olan rig/klip verisine dönüştürülür; runtime temsilleri türetilir.
- Anatomik roller (pelvis, spine, limb, foot, antenna vb.), taraf, zincir ve
  simetri eşleri şablon metadatasıdır. Kalıcı kemik kimliği ile yoğun skinning
  indeksi ayrılır; indeks tek türetilmiş eşleme olur. Klip, seçim ve kontrol
  referansları kemik silme/yeniden adlandırmada açıkça güncellenir.
- Çok meshli karakter, aksesuar, rijit parça ve ayrı karakter sınırları açık
  tanımlanır; kemik silme remap'i ilgili rig'e bağlı meshlerle sınırlıdır.
- Düzenleme bir transaction ve undo/redo işlemi olur. Hata aktif veriyi yarım
  bırakmaz. Rig revision değişince hierarchy, ozz, pose cache ve GPU skin
  buffer'ları tutarlı yenilenir. `.rtp` yeni metadata için sürümlü migration ister.
- Otomasyon preflight/preview/commit ayrımını destekler. Belirsiz anatomi veya
  işaret için güven skoru ve düzeltme isteği döner; sessizce rastgele rig bağlanmaz.
  Uzun fit/weight işleri progress, cancel ve eski revision'a commit reddi ister.

## 6. Genişletilmiş teslim fazları

Faz 0–5 temel rig ve ağırlık hattıdır. Aşağıdaki sıra bağımlılık sırasıdır;
Faz 6'nın şablon sözleşmesi humanoid Faz 4 tasarlanırken de kullanılmalıdır.

### Faz 6 — Anatomik şablonlar ve yönlendirilmiş auto-rig

İlk paketler: humanoid, dört ayaklı hayvan ve altı bacaklı böcek. Ortak şablon
şeması eklem rolleri, parent ilişkileri, simetri eşleri, uzuv zincirleri, joint
limitleri, fit işaretleri ve varsayılan kontrolleri taşır. Kuyruk, kanat, anten,
parmak ve değişken segmentler modüler eklenir; her hayvan tek şablona zorlanmaz.

Auto-rig: mesh preflight → tür/yön/ölçek ve işaretler → şablon fit → görsel
düzeltme → auto-weight → deformasyon önizlemesi → commit. Açık yüzey veya zor
anatomi için desteklenen manuel işaret/bağlama yolu sunulur; otomatik çözümün
reddi manuel rig düzenlemeyi engellemez.

**Kabul:** insan, dört ayaklı ve böcek örneklerinde UI ve ajan aynı şablonla
aynı iskeleti üretir; başarısız fit sahneyi değiştirmez. Her örnekte eklem fit
raporu, ağırlık istatistikleri, uzuv bükme testi ve save/reload kontrolü vardır.

### Faz 7 — Poz, kontroller ve IK/FK

**Faz 2 devamı:** aynı çoklu seçim/aktif kontrol ve pivot sözleşmesi pose/FK/IK
kontrollerine uyarlanır. Seçili poz/kontrolleri aynalama `rig.mirror_pose` kapsamıdır;
rest/bind verisini değiştirmez. Rig pair metadata, pose evaluator ve limit/constraint
kuralları kullanılır; tek işlem/undo ve UI/Python/IPC paritesi zorunludur.

Deform kemikleri ve animatör kontrolleri ayrılır. İki kemikli uzuv IK, pole
vector, aim, eklem limitleri, FK ve poz korunarak IK/FK geçişi kurulur. Çok
segmentli böcek uzuvları için ayrı zincir çözümü ve limit davranışı tanımlanır;
ozz two-bone işi tüm anatomiler için yeterli varsayılmaz.

Poz kütüphanesi, poz aynalama ve kontrol reset ortak servis üzerinden çalışır.
Kontrol/constraint değerlendirme sırası tek pose evaluator içinde belirlenir;
mevcut üç playback yoluyla sahiplik çakışması çözülmeden authoring teslim edilmez.

**Kabul:** IK hedefi izlenir, limit ihlali raporlanır, IK/FK geçişinde poz
tolerans içinde korunur; script ile verilen poz viewport ve render'da aynıdır.

### Faz 8 — Poz ve animasyon klibi yazarlığı

Klip oluştur/sil/kopyala, kemik/kontrol kanalına key ekle/sil, pozdan key üret,
interpolasyon, zaman ölçekleme, loop ve bake. Mevcut object keyframe işlemleri
iskelet kanalı yazarlığı yerine geçirilmez; quaternion ve zaman birimi sözleşmesi
açık olur. Timeline ve eğri editörü aynı canonical klip verisini düzenler.

**Kabul:** iki pozdan üretilen klip ara karelerde doğru örneklenir; quaternion
geçişi, loop sınırı, undo/redo, runtime yeniden üretimi ve `.rtp` round-trip
kontrol edilir. UI/script/IPC aynı kanal ve key sayılarını raporlar.

### Faz 9 — Adım ve hareket tarifi üzerinden animasyon

Kullanıcı viewport'ta ayak basma hedeflerini veya yolunu yerleştirir; hız, adım
uzunluğu, temas süresi ve gait fazlarını ayarlar. İlk hareketler yürüme, koşma ve
dönmedir. İnsan için iki ayak, dört ayaklı için walk/trot, böcek için tripod gait
ayrı temas çizelgeleri kullanır; hepsi uzuv rolleri üzerinden tanımlanır.

Tarif; root hareketi, gövde dengesi, swing/stance ve IK hedefleri üretir. Zemin
örnekleme, ayak sabitleme, ulaşılamayan adım ve temas hataları raporlanır. Canlı
önizleme ile klibe bake aynı değerlendirmeyi kullanır; tarif ve baked klip
arasındaki ilişki ve yeniden üretim politikası açıkça saklanır.

**Kabul:** ayak kayması ve zemin penetrasyonu sayısal ölçülür; hedef dışı adım
açık hata/uyarı üretir. Üç anatomide temas çizelgesi doğrulanır; bake edilen
klip önizlemeyle tolerans içinde eşleşir ve projeyi açınca aynı oynar.

### Faz 10 — Gelişmiş retargeting ve ajan üretim reçeteleri

Faz 0B'deki temel rol/rest-pose/ölçek eşlemesi üzerine IK ile ayak temaslarını
koruma ve gelişmiş root motion uyarlaması eklenir. İnsan hareketinin böceğe doğrudan aktarılması garanti
edilmez; uyumsuz roller preflight'ta raporlanır.

Ajan reçetesi: karakteri incele → şablonu/işaretleri seç → fit raporunu kontrol
et → bağla → test pozu uygula → gait/poz tarifi üret → bake → kalite raporunu al.
Batch işlemler deterministik seed, revision kontrolü ve idempotency sözleşmesi
ister. Görsel kontrol verinin sayısal doğrulamasını tamamlar.

**Kabul:** desteklenen aynı-aile iki karakter arasında klip aktarılır;
eşlenmeyen roller raporlanır. Bir reçete UI'ya tıklama bağımlılığı olmadan IPC
ile tamamlanır; sonuç kullanıcı tarafından kemik, kontrol ve key düzeyinde düzenlenir.

## 7. Ek API ve tamamlanma sözleşmesi

Aşağıdaki adlar **öneridir**, mevcut callable metotlar değildir. Her fazda kesin
imzalar, script/IPC paritesi, capability kaydı ve descriptor'lar birlikte teslim edilir.

| Yetenek | Önerilen ortak operasyonlar |
|---|---|
| Şablon ve fit | `rig.list_templates`, `rig.preflight`, `rig.preview_fit`, `rig.set_landmarks`, `rig.commit_fit` |
| Bağlama ve işler | `rig.bind`, `rig.unbind`, `rig.validate`, `rig.job_status`, `rig.cancel_job` |
| Kontrol ve poz | `rig.create_controls`, `rig.set_ik_target`, `rig.set_ik_fk`, `rig.get_pose`, `rig.set_pose`, `rig.mirror_pose` |
| Klip yazarlığı | `anim.create_clip`, `anim.set_bone_key`, `anim.remove_bone_key`, `anim.key_pose`, `anim.bake_clip` |
| Hareket tarifi | `motion.create_recipe`, `motion.set_steps`, `motion.set_gait`, `motion.preview`, `motion.bake`, `motion.validate` |
| Retarget | `anim.preview_retarget`, `anim.commit_retarget` |

Hatalar ortak makine-okunur kod taşır: örneğin invalid hierarchy, unknown bone,
unsupported anatomy, ambiguous orientation, stale revision, unreachable target.
Validation toleransları, birimler, koordinat uzayı ve mutasyon/undo semantiği
her operasyonun dokümantasyonunda yer alır. Eksik ağırlıklı vertex, NaN,
duplicate influence, geçersiz bone index ve sıfır toplam ayrı raporlanır.

Her fazın teslimi: focused core modülleri + UI + Python + IPC + security ve
descriptor kaydı + hata sözleşmesi + anlamlı regresyon testleri + kullanıcı/ajan
dokümanı. 2000 satırı aşan mevcut dosyalara yalnız küçük entegrasyon wiring'i
eklenir. Yeni özellik gövdeleri ayrı modüllerde tutulur.

**İlk uçtan uca kilometre taşı:** Faz 0–4 ile tek humanoid'i ajanla kurup
bağlamak, viewport'ta düzeltmek ve test pozu ile deformasyonu doğrulamak.
**İlk animasyon kilometre taşı:** Faz 7–9 ile aynı karaktere görsel adım hedefleri
üzerinden düzenlenebilir yürüyüş klibi üretmek. Isı ağırlıkları ve sonraki anatomi
paketleri ilk basit animasyonun zorunlu ön koşulu değildir.

## 8. Panel tasarımı — kurulum, animasyon ve yüz

Ürün kararı: profesyonel joint/control/constraint mantığı korunur; ilk kullanım
yönlendirilmiş şablon akışıdır. İleri kullanıcı tekil kemik, zincir, limit ve
ağırlık düzenlemeye aynı çalışma alanından geçer. Ajan için ayrı UI yapılmaz:
ajan servis durumunu değiştirir, kullanıcı aynı sonucu panelde görür ve düzeltir.

| Alan | İçerik ve görev |
|---|---|
| Sol hierarchy | Karakter altında Skeleton, Controls, Mesh Bindings, Clips ve Face; mesh yokken de karakter görünür ve seçilir |
| Sol bağlam rayı | Rig ve Animate; aktif bağlama göre küçük araç grupları |
| Sağ dock / Setup | Şablon seçimi, anatomi modülleri, eksen/ölçek, işaretler, fit önizleme ve kalite raporu |
| Sağ dock / Bind | Bağlanacak mesh listesi, ağırlık yöntemi, heatmap, etkilenen vertex özeti, normalize/mirror ve yerel düzeltme |
| Sağ dock / Pose | FK/IK, seçili kontrol, pole vector, limitler, reset ve poz kütüphanesi |
| Sağ dock / Motion | Yürüyüş tarifi, gait, temaslar, adım/yol hedefleri ve bake |
| Sağ dock / Face | Yüz eşlemesi, ifade ve viseme şablonları, yoğunluk ve bölgesel kontroller |
| Alt editör | Zaman, kontrol/kemik kanalları, eğriler, klipler ve ayak temas çizelgesi |
| Viewport | İşaret, eklem, kontrol ve adım hedeflerinin doğrudan manipülasyonu; ağırlık/limit/temas görselleştirmesi |

Sağ dock sekmeleri aktif işe göre açılır; hepsi aynı anda uzun bir forma
dönüşmez. Setup ve Bind adımlarında Preview/Apply/Cancel, Pose ve Motion'da
key/bake eylemleri bulunur. Bind ve Pose modlarının renk/adı açıkça farklıdır.
Seçim `(character, stable bone/control id)` ile ortak sahne durumunda yaşar.
Meshsiz rig'de Bind eylemi mesh seçilene kadar açıklamasıyla devre dışıdır;
iskelet düzenleme, poz verme ve klip oynatma kullanılabilir kalır.

### 8.1 Hızlı mimik yazarlığı — ayrı bir deformasyon hattı

Yüz ifadeleri yalnız gövde iskelet şablonuyla çözülmez. Yüz sistemi facial bone,
morph/blendshape veya hibrit sürücüleri destekleyen semantic kanal katmanı ister.
İlk sürüm mevcut yüz kemikleri veya mevcut morph hedeflerini eşler; herhangi bir
yüz mesh'inden otomatik kaliteli morph üretildiği varsayılmaz.

İlk akış: yüz kabiliyetlerini incele → jaw/eye/brow/lip veya morph rollerini eşle
→ neutral pozu doğrula → ifade/viseme paketini önizle → yoğunluğu ayarla → key/bake.
Örnek paketler smile, frown, blink, surprise ve konuşma ağız pozlarıdır. Şablon
bir semantic kanal karışımıdır; hedefte olmayan kanalları raporlar. Otomatik
ses/lip-sync ilk ifade şablonu tesliminin zorunlu parçası değildir.

Panelde küçük ifade önizlemeleri, yoğunluk slider'ı, sağ/sol ve üst/alt yüz
kontrolleri, neutral/reset ve key düğmesi bulunur. İnsan, hayvan ve böcek için
aynı mimik paketi zorlanmaz; türe özgü kanal setleri kullanılır.

**Teknik kapı:** canonical flat geometride morph hedefleri, import/serialization,
CPU/GPU deformasyon eşitliği ve morph+skeletal değerlendirme sırası önce audit
edilir. Eksikse focused modüllerle kurulur. İfade preset'i scene revision ve
kanal eşleme verisiyle saklanır; UI slider'ı ayrı otorite olmaz.

Önerilen script/IPC yüzeyi: `face.inspect`, `face.set_mapping`,
`face.list_presets`, `face.preview_expression`, `face.apply_expression`,
`face.set_channel`, `face.key_expression`, `face.bake`. Bunlar tasarım adlarıdır.

**Kabul:** neutral'a dönüş, eksik kanal raporu, ifadelerin karışımı, morph+bone
birleşimi, UI/script/IPC paritesi, key/bake ve save/reload. Yüz deformasyonu
viewport ve render'da karşılaştırılır. Bu hat ilk gövde rig milestone'unu bloke
etmez; temel pose/clip servisini kullanır.

### 8.2 Meshsiz dış iskelet ve klip importu — öncelikli temel gereksinim

Kullanıcı bulgusu: mesh olmayan dosyalardan iskelet ve animasyonlar mevcut
deneyimde alınamıyor. **2026-09-12 statik inceleme:** kısmi destek mevcut;
uçtan uca çalıştığı henüz doğrulanmadı. Bu yüzden eski bölümdeki animation-only
temsil varlığı runtime kabulü olarak yorumlanmamalıdır.

- İlk incelemede `Import/UfbxReader.cpp` yalnız skin cluster'larını ve baked
  animation hedeflerini seed ediyordu. **İlk değişiklik:** yeni
  `Import/UfbxSkeletonSeeds.h` FBX bone attribute taşıyan düğümleri ve atalarını
  toplar; okuyucu bu yolu kullanır. Skinsiz+animasyonsuz joints artık kaynak
  toplama kapsamındadır; uçtan uca import henüz doğrulanmadı.
- `Import/GltfDirectReader.cpp` skin joint'leri ve animation target'larını
  topluyor. Skin ve animasyon olmayan sıradan node hiyerarşisi kendiliğinden
  iskelet sayılmaz; açık kullanıcı seçimi/format metadatası gerekir.
- `Render/Renderer.cpp` mesh çıktısı boşken yanıltıcı “scene loading failed”
  logu yazıyordu; ilk değişiklik bunu non-mesh içerik işlendiğini belirten bilgi
  mesajıyla değiştirdi. Blok zaten erken dönmüyordu; bu değişiklik tek başına
  tüm import sorununun çözüldüğü anlamına gelmez.
- `scene_data.h` içinde `animationOnlyImport = members.empty() && hasAnimation`
  bulunuyor. Animasyonsuz saf iskelet için ayrı durum gerekir; rig varlığı
  weighted mesh veya klip varlığına bağlanmamalıdır.

**Desteklenecek üç bağımsız girdi:** sadece iskelet; iskelet+klipler ve sıfır
mesh; mevcut hedef iskelete açık eşleme ile sadece klipler. Import seçenekleri
Geometry / Skeleton / Animation olarak ayrılır. Başarı koşulu seçilen içeriğin
geçerli gelmesidir; `triangle_count > 0` değildir. Desteklenen formatlar ve
format başına rig/node ayırma kuralları dokümante edilir; BVH gibi ek formatlar
ayrı reader işi olarak planlanır, mevcut destek gibi sunulmaz.

İskelet sahnede mesh'ten bağımsız kimlik, root transform, bounds, hierarchy ve
clip binding taşır. Viewport overlay, picking, frame selection ve timeline
mesh olmadan çalışır. Başka uygulamadan gelen kontrol rig'i/constraint grafiği
ile deform iskeleti ayrılır: ilk import garantisi joint hierarchy, rest pose ve
desteklenen baked TRS klipleridir; yabancı uygulama kontrol grafiği otomatik
çalışır kabul edilmez. Daha sonra yeni mesh bu iskelete ortak Bind servisiyle bağlanır.

**Öncelik:** Faz 0 ile birlikte bağımsız rig varlığı ve import sözleşmesi kurulur;
auto-rig algoritmalarından önce kapanmalıdır. UI/Python/IPC import yollarında
aynı seçenekler, hata kodları ve sonuç sayıları kullanılır.

**Kabul fixture'ları:** skinsiz animasyonsuz FBX iskeleti; sıfır meshli FBX/glTF
iskelet+klip; mevcut rig'e clip-only eşleme; uyumsuz kanal/eksen/ölçek; append
isim çakışması. Her birinde joint/clip/key sayısı, görünür seçilebilir overlay,
klip oynatma, save/reload ve sonradan flat mesh bind kontrol edilir. Başarısız
import aktif sahneyi korur. Bu incelemede build veya runtime testi yapılmadı.

### 8.3 İlk kaynak değişikliği ve kalan işler

İlk batch bir mevcut import düzeltmesidir; yeni authoring API'si sunmaz.
UI importu ve `rt.scene.import_model` / `scene.import_model` IPC aynı
`ProjectManager → Renderer → loadSceneModel → readUfbx` yolunu kullanır.
Yeni helper vcxproj include listesine kaydedildi. CPU regression kaynağı
`scripts/test/fbx_unskinned_skeleton_test.cpp` explicit joint toplama, ancestor
closure, sıradan düğüm dışlama ve tekrar toplama davranışını kontrol eder.
Test derlenmedi/çalıştırılmadı; kullanıcı build doğrulaması bekleniyor.

İlk kullanıcı build'i yapıldı. Denenen harici FBX, mevcut unsupported
blendshape/cache kontrolünde reddedildi; bu sonuç saf iskelet testini doğrulamaz.
Hata mesajına ayrı `blend_shapes` ve `geometry_caches` sayıları eklendi.
Mesh/skin/anim içermeyen üç joint ASCII FBX fixture'ı
`scripts/test/fixtures/rig_authoring/unskinned_three_joint.fbx` altında hazırlandı;
statik yapısı kontrol edildi, compiled reader/runtime parse doğrulaması bekliyor.

**İkinci kullanıcı testi:** hem fixture hem meshsiz Mixamo FBX, okuyucunun
sonundaki `loadGeometry && model.objects.empty()` kontrolünde “no polygon
geometry” ile reddedildi. İlk inceleme bu son gate'i atlamıştı; collector testi
bu gate'i kapsamıyordu. Kontrol artık imported skeleton joints veya animation
clips varsa meshsiz importu kabul eder. İçeriksiz dosya reddi korunur. glTF son
okuma yolunda eşdeğer mesh zorunluluğu bulunmadı; OBJ'nin faces zorunluluğu
iskelet/animasyon taşımayan format olduğu için korunur.
`scripts/test/rt_test_unskinned_fbx_import.py` production script importu ve
saved joint/hierarchy doğrulamasıyla save/reopen regresyonunu kapsamak üzere
eklendi. Syntax kontrolü yapıldı; build/runtime testleri kullanıcıya aittir.

Statik kontrol: proje XML'i, tek helper kaydı, vendor include yolu ve focused
reader entegrasyonu PASS. `audit_ipc_capabilities.py` mevcut descriptor çıktısı
dispatch kaynaklarıyla eşleşmediği için STALE döndü (441 metot / 33 namespace).
Bu batch dispatch/binding değiştirmiyor; descriptor uyumsuzluğu ayrı kapanacak.

Import sonrası kullanıcı saf iskeleti hiyerarşide, meshsiz FBX klibini de
AnimGraph'ta doğruladı. Ardından inspection/selection API ve IPC ile
animated-global overlay/picking kaynakta yazıldı; yeni görünüm batch'inin
build/runtime doğrulaması bekliyor. Kalan: authoring işlemleri, bağımsız rig
root-motion placement, meshsiz import round-trip canlı kontrolü ve Faz 0B
retarget servis/UI/binding/test teslimi. Bu batch bu fazları tamamlamaz.

## 9. Sonraki araştırmalar

- Şablonsuz genel curve-skeleton çıkarımı ve otomatik anatomik rol keşfi.
- Her topoloji/poz için işaretsiz tek tık auto-rig garantisi.
- Anatomik aileler arası genel hareket aktarımı ve fizik tabanlı locomotion.

## 10. Kullanıcının build ve runtime doğrulama listesi

Yol haritası ve ilk meshsiz FBX kaynak düzeltmesi yazıldı; build veya uygulama çalıştırılmadı.
Implementasyon fazları teslim edildiğinde build'i kullanıcı yapar; ardından:

1. Yeni modüllerin proje kaydı ve Python/IPC capability audit sonuçları kontrol edilir.
2. İnsan, dört ayaklı ve böcek fixture'larında ilgili fazın sayısal kabul testleri çalıştırılır.
3. Kemik ekle/sil/reparent ve weight değişiminde CPU, GPU viewport ve RT deformasyonu karşılaştırılır.
4. Poz/IK, key ve gait düzenlemesi UI, Python ve IPC üzerinden aynı sonuçla doğrulanır.
5. Undo/redo, iptal, hatalı preflight, stale revision ve save/reload kontrolleri yapılır.

İlk batch için: CPU regression testini assertions açık olarak derleyip çalıştır;
skinsiz/animasyonsuz FBX joint zincirini UI ve script/IPC importundan ayrı ayrı
al; Characters / Models → Skeleton altında joints ve atalarını kontrol et.
Skinned FBX'te deformasyon/bone eşlemesi regresyonu kontrol et; meshsiz
iskelet+klip importunda kaynak kanal/key sayısı, oynatma ve save/reload'u doğrula.

Kullanıcının Developer PowerShell'de CPU test komutu (Codex çalıştırmadı):

```powershell
cl /nologo /std:c++17 /EHsc scripts\test\fbx_unskinned_skeleton_test.cpp /Fo:tmp\fbx_unskinned_skeleton_test.obj /Fe:tmp\fbx_unskinned_skeleton_test.exe
.\tmp\fbx_unskinned_skeleton_test.exe
```


### Rig interaction modes (2026-09-12)

The user confirmed owned rig creation, joint addition and local rest editing compile and work.
Added Scene / Rig Edit interaction modes through shared core, Python `rt.rig.set_mode`
and IPC `rig.set_mode`; `get_mode` reads the transient mode. Rig Edit locks viewport
picking to one owned, unskinned, clip-free rig, suppresses object transform gizmos,
mesh picking, marquee and object delete/duplicate. Skeleton overlay cannot be hidden
while editing. Exit an existing mesh/sculpt/paint session before entering Rig Edit
(`viewport_edit_mode_conflict`); choosing Modeling/Sculpt/Paint returns to Scene.
The left context rail now has Edit Bone next to Modeling, reusing the existing Rig
editor in the inspector; viewport bone selection synchronizes the inspector.
Mode is transient, resets on scene replacement, and is excluded from rig undo state.
Scene-mode scripting rest operations remain available; in Edit, other rig writes
and selection return `rig_edit_character_locked`, and rig creation returns `rig_edit_active`.
Pose mode, viewport rest transform gizmo, click-to-add and child-position/length tools
remain subsequent steps. Rest topology editing is distinct from animated pose/key editing.
No new test files, build or app run in this step, per user instructions.
User checks after build: overlap a mesh and an owned rig, enter Edit Bone / Rig Edit,
pick joints and empty space, verify mesh/object gizmos/delete/duplicate cannot take
selection; change rest values and undo/redo; return to Scene and verify mesh selection;
check Python/IPC get_mode and rejection of another rig selection while locked.


### Viewport rest transform gizmo (same batch)

Added a focused RigViewportUI module. In Rig Edit, Move/Rotate and World/Local
use ImGuizmo with the viewport projection. Parent-space conversion feeds the
existing `setRigRestTransform` common service; Python/IPC use the same
`rig.set_rest_transform` operation. During drag, a copied canonical hierarchy
is sampled by `sampleRigPose` to preview the selected joint and descendants,
without mutating live rig/runtime state. Release commits once to history;
Escape cancels. Selection, rig revision, scene load, mode and availability
changes discard a pending drag. Scale remains unsupported.
User checks: translate parent and child, verify descendant preview and final
rest positions; rotate in World and Local; Ctrl+Z/Ctrl+Y; Escape during drag;
switch mode/selection mid-drag; compare perspective and orthographic alignment.
No new test files; build and viewport validation remain user-owned.


### Consecutive joint creation (2026-09-12)

User confirmed viewport rest translation builds and works. First follow-up in
Phase 0 adds shared collision-aware `rig.get_next_bone_name(character, seed="Joint1")`
(Python and IPC). A free seed is returned unchanged; occupied names increment
the trailing integer, or append 1. Suggestions do not reserve names. `rig.add_bone`
now accepts an empty authored name for automatic Joint1/Joint2 naming in the
same staged operation; explicit duplicate names remain errors. Names are bounded
to 128 ASCII characters and integer overflow returns `bone_name_exhausted`.
The Rig inspector defaults to automatic naming, displays the selected parent,
and prepares the next free name after addition. The canonical add operation
selects each new child, so repeated Add child joint extends a chain; selecting
an earlier joint creates a branch. Disable automatic names for exact manual names.
Existing local position is relative to the selected parent's frame. Each add
remains one undo command and uses existing native hierarchy serialization.
Next delivery: rename/reparent/safe delete, then imported meshless rig adoption,
then safe weighted/clip-bound editing. No new test files or builds in this step.
User checks after build: Root -> Joint1 -> Joint2 -> Joint3 by repeated Add;
select Root and add a branch; create a duplicate explicit name with auto off;
undo/redo additions; save/reopen and verify parents and rest transforms.


### Rename / reparent / leaf delete (2026-09-12)

User confirmed consecutive bone additions build and work. Added three Phase 0
operations through shared staged core, UI, Python and IPC: `rig.rename_bone`,
`rig.reparent_bone`, `rig.delete_bone`. Owned unskinned clip-free eligibility
remains; imported, weighted and animated rig editing is still deferred.
Rename accepts an authored ASCII name, rekeys canonical hierarchy and BoneData,
updates child parent references, preserves scene bone index and selects the new key.
Reparent requires an existing same-rig parent, rejects self/descendant cycles and
root reparent, and computes new local rest from old world globals so the entire
subtree keeps its world rest pose. Children arrays are rebuilt after topology edits.
Delete accepts only a non-root leaf, removes its canonical node/BoneData entries,
repairs hierarchy parent indices and selects its parent. No recursive delete or
weight transfer. All edits rebuild skeletonNodes and Ozz through the existing
finish path and native hierarchy serialization; each operation is one undo command.
Cross-rig references are rejected; unchanged rename/reparent returns rig_edit_no_change.
Deletion preserves other character indices, leaving a gap. Added BoneData index
capacity accessor and minimal animation/controller/renderer matrix-sizing and
import-offset integration. Reverse lookup is cleared before rebuild so gaps have
no stale names. UI bone topology controls reuse the current Edit Bone inspector
and are disabled during an active viewport gizmo drag. Visual polish stays deferred.
Next: imported meshless skeleton adoption into an owned editable asset; weighted
or clip-bound bind editing requires its own migration/weight/animation work.
No new test files, build or app run. Static audits only; user runtime checks:
1. Rename a parent and verify children/selection/overlay/Ozz; reject duplicate names.
2. Reparent a rotated chain to another branch: all world rest poses stay unchanged.
3. Reject self/descendant parent; reject root delete/reparent and non-leaf delete.
4. Delete a leaf, then undo/redo all operations and save/reopen the topology.
5. With a skin animated character imported after the owned rig, delete an owned leaf
   and verify that character still deforms correctly; import another rig afterward.


### Imported meshless skeleton -> editable copy (2026-09-12)

User confirmed leaf deletion, undo, additions and unaffected animation of another
skinned character in the same scene. Rename/reparent and save/reopen remain
pending runtime checks. Phase 0 now gains `rig.copy_from(source_character, character)`
in UI, Python and IPC through the same staged service and creation undo command.
It creates a distinct owned clip-free meshless rig from source canonical rest
hierarchy. Source skeleton and clips/runtime are never modified or transferred.
Only skeletonNodes and required ancestors are included; missing canonical entries
fail. Parent-first order and children links are rebuilt; a forest gets an identity
SceneRoot. Positive uniform scale is baked into world joint positions while
proper rest rotations are preserved, then local rigid matrices are recomputed.
Nonuniform scale/shear/reflection are rejected, not approximated. Source posed
animation snapshots are ignored. Empty/unsupported ASCII name characters and
duplicates are sanitized deterministically to valid unique authored names; the
operation returns source_bone -> target_bone entries for every copied source node.
This returned map is not a persistent/live link; later rename/delete can change
the target. Rig uses import_copy template marker and existing native owned-rig
hierarchy persistence/Ozz rebuild. No source loader, mesh or animation graph is copied.
UI: Edit Bone -> Copy imported skeleton -> Source skeleton / Copy as -> Create
editable copy, then Enter Rig Edit. Mesh-bearing/weighted source extraction is
not yet supported. Copying requires Scene mode and a conflict-free target name.
No new test files, build or application run in this batch; static checks only.
User checks: import animated meshless FBX, make copy and edit/move/add/rename/delete;
source clip still plays; copy has no clips; undo/redo creation; save/reopen both;
check copied rest alignment for uniform-scale FBX; reject duplicate target and
weighted/mesh-bearing input; confirm Python/IPC returned bone_map.

#### Follow-up dependency order

1. Validate copy + topology save/reopen and rename/reparent runtime paths.
2. Add durable anatomy roles, symmetry pairs and limb chains with templates.
3. Fit a humanoid template to canonical flat TriangleMesh/DNA geometry.
4. Add explicit bind/weight generation before enabling weighted bind edits.
5. Add pose/FK/IK and key authoring with distinct Pose mode and clip migration.
6. Generalize anatomy families and improve final UI after canonical services work.
Imported skin/clip-bound edits require an explicit rebinding/animation migration
policy; enabling them by merely lifting ownership checks is not acceptable.


### Rest / Pose contract and batched validation

Rig Edit means canonical rest/bind editing, not a universal T-pose reset. Rest
may be T-pose, A-pose, animal stance or insect rest. Future Pose mode owns a
transient evaluated pose and animation keys. Entering Rig Edit from playback or
Pose must pause evaluation, leave transient pose state distinct from rest, and
show the actual stored rest pose. Leaving must not silently turn edited rest into
animation keys or resume an incompatible clip. Weighted/clip-bound rest edits
need explicit preview/commit for rebinding and animation migration; lifting the
current safety gates is insufficient. These semantics are core requirements,
not only UI polish. Current Edit mode still rejects clip-bound/weighted rigs;
Pose/evaluator transition logic is not implemented in this batch.
Per user request, extend runtime/test cadence to milestone batches, not each
small capability. Do not add new test files now. Group anatomy + templates,
then fitting, then bind/weights and Pose/IK milestones. Continue necessary
non-build source/schema integration checks; user performs builds/application checks.

### Canonical anatomy schema foundation (2026-09-12)

Added focused RigAnatomy core/API/UI modules. Owned rigs carry version 1 anatomy:
family custom/humanoid/quadruped/insect; named role->bone assignments; disjoint
left/right symmetry pairs; named ordered contiguous parent limb chains.
Family is a tag, not a generated or automatically detected anatomical guarantee.
`rig.get_anatomy(character)` reads; `rig.set_anatomy(character, anatomy)` replaces
through shared core validation and one existing rig undo command, in UI/Python/IPC.
Set remains owned/unskinned/clip-free and respects scoped edit character lock.
Unknown fields/types/version, unknown keys, duplicate roles, duplicate pair bones,
invalid/duplicate chain names, repeated/disconnected chain joints and limits
are rejected before live scene mutation. 4096 roles/pairs, 1024 chains, max 4096
joints per chain; ASCII role/chain IDs 1..128 with underscore/hyphen/dot allowed.
Native save/load stores anatomy alongside NodeHierarchy; legacy absent anatomy
loads as custom/empty. Load validates metadata against the restored hierarchy.
Rename updates references; copy remaps all anatomy keys; reparent that would break
a declared chain fails; anatomy-referenced leaf deletion fails until records are
removed. Topology rebuild validates metadata so no dangling reference is published.
The existing Edit Bone inspector now has a Rig anatomy section: family, selected
bone role assignment, symmetry counterpart, ordered chain draft and record removal.
No permanent panel or new workspace. Template packets will use this schema before
humanoid fitting. Limits/markers/control recipes and generated humanoid/quadruped/
insect templates remain next work; this is schema foundation, not auto-rig delivery.

Example Python/IPC anatomy payload for an existing chain3 rig named Rig:

```json
{"version":1,"family":"custom","roles":[{"role":"spine.base","bone":"Rig_Joint1"}],"symmetry":[],"chains":[{"name":"spine","bones":["Rig_Root","Rig_Joint1","Rig_Joint2"]}]}
```

Grouped user milestone checks after anatomy/templates build: assign/read metadata
through UI and Python/IPC; invalid pair/chain rejection; rename references; chain
reparent and referenced leaf-delete rejection; clear records and retry; copy
metadata remap; undo/redo; native save/reopen. Copy and rename/reparent runtime
validation from preceding batches remains open. No new test files/build/app run.


### Shared Rest / Animated view and Rig tab removal (2026-09-12)

Removed the temporary Rig tab from AnimationWorkspace; Graph and Retarget remain.
Rig editing/anatomy stay in existing left Edit Bone inspector. No copy UI expansion.
Graph now exposes per-character Rest/Animated display through `rig.set_pose_view`
and `rig.get_pose_view` (UI/Python/IPC same core). Choices are transient, reset on
scene replacement, not saved, not undo state and do not mutate canonical bind/clip
assets. Scoped Rig Edit always has effective Rest, while requested view is preserved.
Rest skips chosen rig graph/controller/Ozz/root-motion evaluation. Canonical
NodeHierarchy sampler supplies bind globals; skin matrices use per-model inverse,
global rest and bone offsets. New geometry path reads/mutates flat TriangleMesh
only, never Triangle facades. Animated rigs' unweighted flat node meshes return to
stored hierarchy rest transforms in Rest. Other characters continue animation.
Switching back refreshes skin display even for a paused clip; pending CPU restore
survives GPU-only evaluation. Rest and Animated do not mean T-pose reset or new bind.
Rest shows the actual authored rest; changing bind remains gated from skin/clip rigs.
Rig pose view is a viewport/runtime evaluation switch, not a render-sequence/export
preset. Explicit existing force_bind_pose preview takes precedence where used.
Retarget source and target have separate Rest checkboxes, passed to shared
`anim.sample_clip_binding` source_pose_view/target_pose_view selectors. These are
read-only local preview choices, independent of scene switches, keep binding report
and clip data unchanged, and return source/target pose source labels. Both Rest
stops preview clock advancement; default calls remain animated for compatibility.
No new test files, project build or app run. Checks remain grouped per user request.
Next grouped user checks: weighted animated character Rest/Animated with paused and
playing clips; two characters with only one in Rest; CPU/GPU viewport transitions;
rest rig overlay and Rig Edit lock; Retarget each side and both Rest; invalid selector
via IPC/Python; verify only Graph/Retarget tabs and native reopen defaults Animated.
Anatomical template generation remains the next authoring milestone; imported skin
bind editing and Pose/IK/keys are not unlocked by the display switch.


Rest view runtime follow-up: user confirmed Rest works, but skin viewport presentation
waited for camera movement. Pose-view API now wakes start_render and resets CPU/backend
accumulation only after a successful actual mode change. Shared UI/Python/IPC path;
failed or idempotent requests do not disturb accumulation. No builds/new test files.
Grouped runtime check: switch Rest/Animated with camera fixed and a paused clip.


### Shared anatomical rest templates v1 (2026-09-12)

User confirmed Rest display works; viewport wake follow-up remains to be checked
with camera fixed. Next authoring milestone supplies a scene-independent catalogue
and template builder used by `rig.create`, `rig.list_templates`, `rig.get_template`
in UI/Python/IPC. Existing Edit Bone template combo reads the same catalogue; no
Rig tab or new permanent panel. Root/chain3 preserve original positions and empty
anatomy. New layouts: humanoid 27 joints (T rest, spine/head, arms/hands and legs/
feet/toes), quadruped 26 (standing rest, spine/head, four legs and tail), insect6
36 (body/head, six four-segment legs and two antenna chains). Anatomy roles,
disjoint symmetry pairs and ordered contiguous chains are produced with hierarchy.
All layouts: +Y up, +Z forward, +X character-left; neutral identity joint rotations,
unit scale, ground-origin to highest endpoint height. Insect height includes
antennae. Heights are scene-unit sizes, not automatic anatomy fitting/normalization.
Template read payload is version 1 with Template_ prefixed preview keys; actual
creation prefixes requested character. Catalogue reports version, count, family
and suggested height (human 1.8, quadruped 1.0, insect 0.3); creation's old API
default height remains 1.8. Shared core rejects unknown IDs, nonfinite/nonpositive/
>10000 heights and invalid template anatomy before publishing live rig state.
The existing finish stages all four representations, runtime, one undo command and
native snapshot hierarchy/anatomy persistence. Generated bone anatomy references
must be removed before leaf deletion, and reparent must keep declared chains valid.
No production joint-axis orientation, optional fingers/wings/modules, joint limits,
fit markers, constraints, auto-fit, weights, IK or gait controls yet. Existing rig
snapshots are canonical and do not regenerate implicitly from changed recipes.
Next: fit markers / preflight and explicit template-to-flat-TriangleMesh/DNA fit
preview/commit, then bind/weight generation before weighted bind editing.
No new test files, build or application launch. One combined source/registration/
IPC schema audit for the package; user checks stay grouped:
1. Create each new family at two heights; inspect layout, role/pair/chain counts.
2. Read list_templates/get_template via Python/IPC; get_template creates no objects.
3. Move/edit template joints, rename anatomy refs, undo/redo creation and edits.
4. Save/reopen hierarchy, metadata and Ozz; other skinned character stays animated.
5. Root/chain3 compatibility; unknown ID and invalid height leave scene unchanged.
6. Carry forward camera-fixed Rest/Animated and earlier copy/anatomy runtime checks.


### Multiple owned rigs and independent scene placement (2026-09-12)

User confirmed all new templates build and generate. Second UI creation reused
Rig and hit rig_name_conflict; there is no single-rig-per-session core limit.
Edit Bone now defaults to collision-aware automatic rig names from a seed via
shared rig.get_next_name. Manual names retain explicit collision errors. Scoped
Rig Edit must be exited before creating/selecting another rig; the panel explains
this lock rather than silently leaving an active edit.

Scene mode: select any joint or bone segment of an owned meshless rig, then use
Move/Rotate World/Local gizmo or whole-rig inspector position/rotation. This edits
ImportedModelContext.rigSceneTransform, not canonical NodeHierarchy rest matrices,
bone offsets, anatomy or clip curves. rig.get_scene_transform/set_scene_transform
expose the same operations to Python/IPC. Meshless owned rigs with clips may be
placed; imported/weighted scene placement remains a separate future integration.
Bone selection clears object selection; picking an object gives its gizmo back.
Rig Edit keeps individual rest manipulation; root parent and drag preview include
actor placement, so edits remain correct after moving/rotating the entire rig.
Drag preview is transient, Escape cancels, release records one undo command.
Native project snapshots persist rigid actor placement; older snapshots use identity.
Scale/shear/reflection are rejected by the common service.

No new test files/build/app launch. Next grouped runtime checks: create three rigs
without manually changing names; separate them in Scene via a non-root joint;
rotate one, enter Edit and move its root/child; undo/redo placement/rest edits;
save/reopen all three and their placements; check mesh selection and skinned rig
animation remain functional. Read/write matrices and invalid rigid transforms via
IPC/Python when agent verification is available.

### Detailed template profiles

Keep current layouts as lightweight defaults. Add versioned basic/standard/detailed
recipes with optional modules and bounded segment counts through the same template
builder/catalogue, UI, Python and IPC: humanoid fingers and configurable spine/neck;
quadruped paws/toes, spine and tail; insect antenna/leg segment counts and optional
wings. Generate stable names, roles, pairs and contiguous chains together. Store the
concrete hierarchy/anatomy snapshot plus recipe/profile/version; never regenerate
existing rigs silently when catalogue versions change. Joint orientation and limits
must be solved before promising production IK/twist/bend control. Detailed profiles
follow multi-rig placement stabilization and precede or accompany family fitting.

**Detailed profile v2 source delivery (2026-09-19):** The existing 27-joint
`humanoid` recipe remains unchanged for compatibility. The separate
`humanoid_detailed` recipe has a denser Spine01/02/03 and Neck01/02 chain plus
four-joint Thumb, Index, Middle, Ring and Pinky branches for 70 total joints.
Every digit has stable roles, a contiguous Hand-to-tip anatomy chain and a left/right
symmetry pair for every joint. Anatomy schema v6 stores validated derived fit rules:
intermediate spine/neck joints interpolate between primary torso guides, while the
three internal joints of each digit interpolate between Hand and its editable tip.
Alignment exposes Primary, Hands and All-guides filters; Ctrl/Shift selection moves
multiple editable guides by a shared plane delta. Derived joints remain visible but
are regenerated from their guide endpoints rather than becoming contradictory manual
inputs. The profile
uses the existing template catalogue/create surfaces, so UI, Python and IPC expose the
same hierarchy without a second construction path. Existing rigs are never regenerated.
The current parent-owned envelope contract treats the new digit joints as deforming
segments. Their `*_hand.*` roles use a distinct `digit` category whose default radius
is 30% of the extremity radius, clamped to the existing minimum; each segment remains
individually editable through the existing per-bone profile service. This avoids using
the palm/foot-sized default capsule on closely spaced fingers. Build, viewport shape,
manual fitting density and bound-hand deformation remain user verification gates.
Configurable segment counts, twist joints, authored joint axes, box selection and
grouped finger FK remain follow-ups; this delivery does not claim them.


### Uniform actor scale and anatomical 2D fitting direction (2026-09-12)

User confirmed multiple rigs and Scene placement build and work. Scene Rotate
already exists (R / Rotate); this step adds S / Scale and numeric Uniform rig scale.
Scale handles all adjust one positive uniform actor scale, preserving template
proportions and proper rotation. Rig Edit still allows only rigid individual rest
Move/Rotate. Placement validation, Python/IPC rig.set_scene_transform and native
restore share the same translation/rotation/uniform-scale contract (scale range
0.0001..10000). Nonuniform scale, zero/negative scale, reflection, shear and
nonfinite values are rejected. Undo/redo and transient release-only drag preview
reuse the existing placement command; hierarchy, anatomy and clips stay unchanged.
Uniform scale applies to evaluated model-space motion too, as an actor transform.

Next fitting milestone uses canonical flat TriangleMesh / DNA geometry, without
mutating the target mesh: preflight bounds and explicit axes/symmetry; synchronized
front/side orthographic 2D silhouettes; anatomical landmarks for pelvis, shoulder,
elbow, wrist, knee, ankle and family equivalents; template joint-position fitting
with symmetry and interior/medial constraints. A single 2D projection cannot infer
depth reliably, so side-view/3D confirmation and fit residual/interior reports are
required before commit. Uniform actor placement gives coarse initial alignment;
body/limb proportions are solved as fitted joint positions, not nonuniform actor
scale. UI, scripting and IPC share read-only preview and explicit undoable commit,
with revision checks. Fitting is followed by bind/weights. No nearest-vertex-only
placement or distance-only weighting is the anatomical quality criterion. The 2D
fitting UI/solver is planned, not delivered by this scale change.

No new test files, build or app launch. Grouped user check: Scene G/R/S on two rigs,
scale then enter Rig Edit and move root/child, cancel and undo/redo, native reopen,
invalid matrices via IPC/Python. Confirm rest edits and the other rig are unchanged.


## Phase checkpoint and agent handoff (2026-09-13)

This checkpoint supersedes older implementation-status notes, not the numbered
phase design. No phase is closed by template generation alone.

| Roadmap phase | Current delivery | Remaining gate / next work |
|---|---|---|
| 0 - Editable bones | Owned meshless create/add/rename/reparent/leaf delete/rest edit, scoped Edit, shared undo/API/IPC/native snapshot | Imported weighted original bind editing/rebind and clip migration; grouped reopen/API verification |
| 0B - Clip binding / basic retarget | Meshless clip import, same-rig/manual/rest-basis binding, synchronized preview | Dedicated agent regression, advanced contacts and reusable mapping workflows |
| 1 - Influence contract / measurements | Shared top-four import normalization (user-confirmed), flat weight_stats/get_weights and index/ownership diagnostics | Preceding package user-confirmed; live editing/repair and copy/topology writer enforcement not universally complete |
| 2 - Selection / overlay / inspector | Canonical joint-global overlay, selection/rest tools, weight tint; joint limits and pose-following capsule envelope display source-delivered | Per-bone envelope radius handles and weighted rest rebind remain open |
| 3 - Distance weights baseline | Initial actual-flat multipart owned binding, nearest-segment top-four weights, shared UI/API/IPC, undo/redo/native registry and CPU/GPU invalidation delivered in source | User confirms build, Apply, bind and apparently correct initial weights; bind undo/reopen/deformation quality checks pending; visibility constraints and general weighted mutation remain open |
| 4 - Humanoid auto-placement | Flat preflight, multipart targets, front/side manual landmarks and preview/commit; humanoid v3 recipe | Automatic placement, interior/visibility constraints and measured residual remain open; manual fit is not automatic |
| 5 - Weight quality | Bounded category capsule reweight and pose-following viewport preview source-delivered over authored bindings | User deformation checks, per-bone radius overrides and visibility/interior or heat diffusion comparison |
| 6 - Anatomical templates / guided auto-rig | Basic humanoid/quadruped/insect6/avian, role/pair/chain metadata and multipart manual fitting | Apply/bind first path user-confirmed; prioritize Pose and bone keys after 2B; detailed modules and joint-axis/limit design remain open |
| 7 - Pose / controls / IK/FK | 7A Pose/FK, hand/foot two-bone IK and animator control layer v1 runtime-confirmed; blend/position pins and joint limits/overlay source-delivered | Grouped finger FK, foot pivots/roll and representative contact/limit/save-reopen checks; weighted rest/rebind remains separate |
| 8 - Pose / clip authoring | 8A explicit bone keys and Auto Key user-confirmed | Curves, clip operations, bake and grouped native reopen verification |
| 9 - Step / movement recipes | Humanoid in-place Walk Recipe v1 runtime-confirmed; body-motion refinement source-delivered | 9A footprints/contact phases; 9B root-motion arc-length spline walk/run; 9C spline flight |
| 10 - Advanced retarget / agent recipes | Planned beyond Phase 0B | Contact-aware IK, motion quality reports and agent workflow recipes |

### Current bounded delivery: humanoid basic recipe v2 (Phase 4/6 preparation)

27 joints and all names/parents/roles/pairs/chains stay stable. Positions are authored
height fractions for a generic neutral T-rest, not a certified anthropometric
standard. Shoulders and hips narrowed, shoulder line raised and made level; arm
segments rebalanced; head origin lowered to leave a meaningful head terminal
segment. Knees/ankles share the hip X/Z for straight neutral legs; future IK must
use explicit bend preferences rather than relying on a template kink.

| Landmark / segment | Height-normalized recipe v2 |
|---|---|
| Pelvis / hip pivots | Pelvis Y .530; hip Y .515; hip X +/- .055 |
| Spine / chest / neck / head / crown | Y .620 / .735 / .835 / .875 / 1.000 |
| Shoulder / elbow / wrist / hand end | X +/- .115 / .285 / .425 / .500, level Y .800 |
| Clavicle base | X +/- .035, Y .800 |
| Knee / ankle | Y .285 / .045, X +/- .055, Z 0 |
| Toe / toe end | Y .020, Z .095 / .140 |

Recipe version is reported by list_templates/get_template/list_bones and saved
as ImportedModelContext.rigTemplateVersion. Missing legacy version defaults to 1.
Existing snapshot hierarchy/anatomy/placement never regenerate on load; only new
humanoid creation uses v2. Root/chain3/quadruped/insect remain recipe v1. Payload
hierarchy/anatomy schema versions remain 1; recipe version is a separate concept.
Detailed fingers/spine/twist/limits and control rigs are future Phase 6/7 work,
not part of this geometry-only proportion correction.

### Next agent task: Phase 4/6 preflight and fitting foundation

Start with rig.preflight and explicit target mesh identity/axis selection over
canonical flat TriangleMesh / DNA positions. Report finite bounds, topology and
surface limitations, symmetry/pose assumptions; do not infer anatomical certainty
from bbox or nearest vertices. Reuse existing context panel/dock/bottom editor,
then add synchronized front/side projections and canonical anatomical landmarks.
Deliver a read-only preview with residual/interior diagnostics and explicit
undoable commit; reject stale mesh/rig revision before mutation. Mesh geometry
stays unchanged; fitting writes owned rig joint rest positions through one common
service. Allow positive uniform actor placement for coarse alignment; convert
landmarks correctly between world and model coordinates. Do not enable weighted
bind edits or generate weights until their independent gates are satisfied.

Read these focused modules before editing:
- Animation/RigTemplates.h/.cpp: shared recipes/catalogue/builder.
- Animation/RigEditing.h/.cpp: staging, ownership/clip gates, rest/runtime rebuild.
- Animation/RigAnatomy.h/.cpp: role/pair/chain schema and topology validation.
- Animation/RigView and RigPosePreview: actor placement vs evaluated joint globals.
- UI/RigEditingUI and RigViewportUI: existing inspector and transient drag preview.
- Api/RtApiRigEditing, RtApiRigTemplates, RtPythonRig, RtIpcRig: shared surfaces.
- Core/ProjectManager: minimal snapshot wiring only (file exceeds 2000 lines).

Constraints for handoff: obey AGENTS.md; no builds/compilation or app launch (user
builds), no new test files per step per user's instruction; use grouped runtime
checks. Existing >2000-line files get only minimal integration wiring. Every new
operation must include common core, Python, IPC, validation/error docs and regenerated
scripts/ipc_descriptor_overlay.json / RtIpcMethodDescriptors.cpp. Never use per-face
Triangle facades as geometry authority. Avoid malformed BOM prefixes; save source
as valid UTF-8. Do not expand editable-copy UI; future Edit/Pose targets the original
rig only after rebind/migration service exists.

Pending grouped user checks: Scene G/R/S and scaled root/child rest edit;
new humanoid v2 proportions at two heights; old v1 rig unchanged after reopen;
undo/redo, version reporting and snapshot persistence; other skinned rig animation;
camera-fixed Rest/Animated refresh and earlier outstanding IPC/Python checks.
No project build, application run or new test files in this delivery.


### Phase 6 continuation: revised animals and avian basic template (2026-09-12)

Supersedes the preceding checkpoint's recipe catalogue version summary. New
quadruped basic v2 keeps its 26 keys/parents/roles/pairs/chains: narrower limb
pivots, a less steep torso and adjusted neck/head, paws and paw endpoints now at
Y=0 rather than floating at .08 height. The generic fore/hind bends stay distinct;
species-specific hocks/paws/toes remain detailed-profile work. Insect6 basic v2
keeps 36 keys and metadata: hind leg attachments moved closer to body, proximal
leg endpoints raised to form a supporting arch, distal endpoints redistributed.
Ground-to-highest-antenna height normalization remains unchanged. These are
authored fitting seeds, not certified universal animal anatomy.

New template ID/family avian, recipe v1: 32 joints, 32 roles, 10 symmetry pairs,
7 contiguous chains (spine, beak, tail, two wings, two legs). Spread-wing rest
layout includes shoulder/elbow/wrist/tip/end, head/crown and beak, three-joint tail,
paired thigh/knee/ankle/toe/end. +Y up / +Z forward / +X left, ground to crown
height 1.0 normalized, suggested height .6 scene units. All rest rotations identity
and scale unit. This is a generic bird; bat finger-supported membranes and insect
wing attachments require separate future modules/profiles, not this same anatomy.
Feather chains, folding constraints, aerodynamic/flight animation and IK are not
implemented. Root/chain3 remain unchanged v1; humanoid remains revised v2.

Shared catalogue/builder serves UI/Python/IPC list_templates/get_template/create;
anatomy validation and family inspector now accept avian. Existing snapshot
hierarchies stay untouched; recipe version persistence and legacy v1 fallback
reuse the prior package. Rename/reparent/delete anatomy protection still applies.

Handoff status: Phase 6 basic catalogue expanded; Phase 4/6 fitting/preflight is
still next, not complete. Phase 1 remains mandatory before weight generators.
Detailed animal segments and wing modules belong to Phase 6, controls/folding to
Phase 7, authored flight clips to Phase 8/9. No new test files, builds or app run.
Grouped user check: create quadruped/insect6/avian at two heights; inspect feet,
mirrored wings and chains; Scene G/R/S then rest root/child edit; undo/redo and
save/reopen versions; confirm older rigs and other skinned animation unaffected.


### Humanoid sagittal correction and mesh-binding priority (2026-09-13)

User compiled animal/avian templates and reports organic layouts; humanoid v2
became too planar after knee depth removal. Humanoid basic recipe v3 restores
model-space +Z depth: lumbar +.008 height, chest -.012, neck 0, head/crown +.012;
hip -.004, knee +.014, ankle 0. Arms stay level/symmetric with shoulder/wrist
-.012 and elbow -.004, giving a small forward bend hint. These are authored
neutral fitting defaults, not anatomical measurements or a solved bind pose.
IK still needs explicit bend/pole preferences later. All 27 keys, topology and
anatomy metadata stay unchanged. Newly generated humanoids use v3 across
UI/Python/IPC; saved v1/v2 snapshots stay unchanged. Other families unchanged.

Current usefulness boundary: generated rigs are standalone skeleton authoring
assets; they do not yet deform a target mesh. Do not describe templates as a
completed auto-rig. Stop extending the basic catalogue before the next end-to-end
mesh milestone: Phase 4/6 flat TriangleMesh/DNA preflight, explicit mesh target,
front/side landmarks and fit preview/commit; Phase 1 shared influence pruning/
normalization and diagnostics; Phase 3 initial bind/weights then Phase 5 quality
improvement. Binding must incorporate actor translation/rotation/uniform scale
correctly in rest globals and offsets, preserve canonical flat geometry, and
invalidate CPU/GPU skinning consistently. Acceptance is a fitted humanoid that
actually deforms in viewport/render, with undo and native save/reopen, using the
same service through UI/Python/IPC. Existing weighted imports are a separate path.

This correction delivers no mesh fitting/binding or weight generator. No builds,
app launch or new test files. Grouped user check: a NEW humanoid viewed from side
shows modest knee/spine/elbow depth; feet retain forward +Z, left/right symmetry
and crown height stay unchanged; old saved humanoids retain their original poses.


UI follow-up (2026-09-13): removed duplicate Move/Rotate/Scale and World/Local
radio buttons from Edit Bone. Rig viewport continues to consume the existing
global manipulator mode/space; inspector numeric authoring fields remain.
No core/API/IPC operation changes, new tests, builds or app launch.


### Phase 4/6 mesh preflight foundation and UI labels (2026-09-13)

User-facing template labels no longer display v1/v2/v3, and the recipe version row
is removed from Edit Bone. Internal recipe provenance remains in native snapshots
and agent payloads for compatibility; it is not a release/maturity label.

The next bounded fitting foundation is delivered: rig.preflight(mesh) shared core,
Python, IPC and Edit Bone Mesh setup. Explicit exact mesh nodeName selection reads
canonical flat TriangleMesh/DNA current P, checks actual buffer size before raw
access, applies object base transform once, reports finite bounds, counts, invalid
triangle indices/numerical areas, zero-area triangles, trailing indices and skin.
Missing/ambiguous mesh identity and malformed buffers/transforms return named errors;
geometric limitations return a report. There is no target mesh/rig persistent link
or binding yet. Snapshot diagnostics are refreshed explicitly after geometry edits.
can_start_landmarks means finite volumetric unskinned geometry, not fit acceptance;
fit_ready stays false, and axes/pose/symmetry/closed-surface/interior are unverified.
Animated current transforms are not evaluated; weighted meshes cannot start this
owned unskinned setup. UI/API do not claim anatomical certainty from bounding boxes.

Next agent task is still Phase 4/6: canonical landmark storage, front/side 2D views,
explicit axis/rest/symmetry choice, quality-reported fit preview and revision-safe
undoable commit. Phase 1 normalization/diagnostics precede Phase 3 bind/weights.
Use Animation/RigPreflight for flat geometry reads, Api/RtApiRigTemplates for shared
API entry, RtPythonRig/RtIpcRig and existing Edit Bone target/report integration.
No builds, app launch or new test files. Grouped checks: translated/scaled unskinned
mesh diagnostics, unknown/ambiguous target, planar/empty/invalid geometry reports,
existing skin rejected for landmark readiness; Python/IPC same report and no mutation;
labels show names without versions. This is a foundation, not a completed auto-rig.


### Phase 4/6 manual fitting foundation - evening handoff (2026-09-13)

User requested completing the current fitting foundation before tomorrow. Delivered
manual alignment loop: explicit target mesh preflight -> get_fit_setup -> front
(X/Y) and side (Z/Y) landmark editing -> preview_fit -> commit_fit. Existing Edit
Bone Mesh setup contains a collapsible Fit skeleton to mesh section; no new
permanent shelf or Animation Rig tab. Prepare alignment gives coarse height/center
placement, preserves the seed's proportions/depth and supplies at most 8192 flat
mesh sample points for projection. This is projected mesh sampling, not an inferred
anatomical silhouette/medial-axis solver. Drag landmarks in either view; the third
coordinate stays unchanged, and numeric world position is available for the selected
joint. Draft edits never change scene rest until Apply manual alignment.

Caller confirms +Y up, +Z forward and rest pose. preview_fit requires complete
finite coordinates for all canonical node keys, rejects zero-length segments,
reports lengths and bounds containment. Out-of-bounds points block commit.
Bounds containment is NOT a surface interior test; interior_verified stays false,
and the UI explicitly presents this as manual alignment. Automatic anatomical
fitting, interior/medial constraints and measured auto-fit residual remain open
Phase 4 work. Do not mark the entire numbered Phase 4/6 complete.

Shared core is Animation/RigFitting.h/.cpp. UI, Python and IPC expose
rig.get_fit_setup(character, mesh), rig.preview_fit(character, mesh, landmarks,
axes_confirmed=false), rig.commit_fit(character, mesh, preview). Drafts are request
payloads, not persisted scene state; successful canonical rest results persist
through existing native hierarchy/anatomy/placement snapshots. Commit validates
rig revision and flat mesh content token (P, indices, base transform), recomputes
all checks and ignores untrusted can_commit flags. Preserves joint global rotations,
converts world landmarks through inverse actor placement, derives local rest and
uses the existing common stage/finish/undo rebuild. UI, script and IPC do not
implement their own fitting or mutation logic. Imported weighted originals and
clip-bound owned rigs stay blocked by existing rest-edit gates. Mesh geometry is
unchanged, and no binding/weights are delivered.

This bounded manual foundation is source-delivered, pending user build/runtime.
Tomorrow's next end-to-end goal: Phase 1 influence normalization/measurements and
Phase 3 bind/weight baseline over the fitted owned rig and flat target mesh, with
correct actor/rest/bind spaces and CPU/GPU invalidation. Improve automatic fitting
and interior diagnostics in Phase 4 without calling this manual mode automatic.
Detailed wing modules/controls remain Phase 6/7; do not expand the catalogue now.

Grouped user verification (no Codex builds or app launch, no new test files):
1. Choose an unskinned static mesh and owned humanoid; Prepare alignment, confirm
   axes/rest, adjust matching knees/hips/shoulders in front and side, Preview, Apply.
2. Move/rotate/scale actor beforehand; world landmarks must still produce correct
   model rest. Check actual viewport skeleton after Apply, undo/redo and native reopen.
3. Repeat animal/avian; all chains, names and anatomy remain stable after fit.
4. Outside bounds, zero-length, missing/nonfinite marks and unconfirmed axes fail
   without scene changes. Changing rig or mesh after Preview gives stale preview.
5. Python/IPC use the same setup/preview/commit; previews do not mutate the scene.
   Existing skinned animated character remains unaffected; target mesh is not deformed.


### Multipart manual fitting and anatomy UI simplification (2026-09-13)

User compiled manual fitting; some targets fail and anatomy controls are crowded.
Specific failing-target error code is requested; do not claim every failure fixed.
Two unnecessary manual-stage barriers addressed: isolated zero-area faces are now
warnings when valid faces remain, and boundary checks have explicit scale/float
precision tolerance. Nonfinite/bad indices/trailing indices/no valid faces/invalid
bounds/existing skin still block and preflight.blockers states reasons. Bind/weight
stages must define their own stricter geometry contract, not inherit manual warnings.

New shared rig.list_fit_targets catalogue supplies exact individual mesh names and
model:<importName> groups. Group scope uses canonical flat scene objects associated
with model flat members or exact NodeHierarchy node keys; no Triangle facade geometry
is read. All source transforms are applied once into a transient flat diagnostic
snapshot; scene objects remain separate and unchanged. Preflight and get/preview/
commit_fit accept the same group ID through UI/Python/IPC. Any source part skin blocks
this unskinned setup. Combined transformed positions/indices participate in stale
preview token; group count/part names are reported. No subset filter or persisted
merged/bound model is created. Accessories are included in whole-import groups, so
users can select the body alone when props/hair affect bounds. Future multipart bind
must bind actual canonical parts individually to one rig, never the diagnostic mesh.

Anatomy inspector now shows family/count summary and notes templates already supply
roles/pairs/chains. Roles, Symmetry pairs and Limb chains are collapsed bounded lists;
one remove action for selected record, not a button on every row. Edit anatomy
metadata reveals family/creation tools; assign/pair/build forms are nested collapsed
sections. Selected chain shows compact base-to-tip path. Same set_anatomy service,
validation, undo and serialization; metadata is not required to be manually matched
for normal template fitting. This UX cleanup creates no UI-only business logic.

Phase checkpoint: Phase 2 inspector usability improved, Phase 4/6 manual foundation
now supports whole imported-character groups. Full auto-fit/interior tests and mesh
binding/weights still open. No build/app launch or new test files. Grouped user
check: body/head/clothes import chooses Character (N parts), per-part transforms
remain separate; source skin blocks group; zero-area warnings and blockers visible;
manual fit/undo/reopen and stale group preview; anatomy lists selected remove and
advanced assign/pair/build still validate through common service.


### Skinned startup Rest and render-idle scheduling (2026-09-13)

User reports automatic first-clip playback after import/native reopen keeps Vulkan
RT/other renderers resetting. Animation initialization invokes shared
initializeRestPoseDefaults: valid weighted rigs without explicit transient pose
choice enter Rest through setPoseView, using actual bind hierarchy. Explicit views
survive append/reinitialization; standalone meshless clips keep animated defaults.
Native scene replacement clears transient choices, so skin reopens in Rest.
Clips and graphs remain registered for explicit Animated view selection.

Main autonomous graph wake skips effective Rest models. Three file-animation
scheduling decisions use needsFileAnimationEvaluation: pending view transitions
are applied once; clips all belonging to Rest models do not force ongoing updates
or resets. Animated models and unresolved legacy clips retain evaluation. Manual
camera/light/world timeline and simulation paths stay independent. Cached Rest
skinning suppresses repeated application. UI/Python/IPC use existing pose-view API.

No builds/app run/new test files. Grouped user check: skinned import/native reopen
with camera fixed accumulates in Rest; explicit Animated resumes, Rest settles;
append preserves previously explicit Animated rig while new skin starts Rest;
meshless clips and keyed camera/light/simulation continue. Invalid bind/hierarchy
keeps validated pose-view failure semantics; no invented bind pose.

### Rest pose: backend rebuild skinning repair

- Solid/Matcap and Vulkan RT geometry rebuilds upload bind-space vertices. Reapply the current final bone matrices immediately after rebuild, including backend switches, without advancing clips. FBX unit conversion in those matrices must remain active in Rest. OptiX full scene synchronization also reapplies the current pose.
- Rest evaluation repairs its model-scoped matrix slots if the renderer buffer was reset independently of the cached pose snapshot; unchanged matrices do not trigger continuous accumulation resets.
- Pending user verification: fresh FBX import and project reopen in Rest; Solid/Matcap/Vulkan RT/OptiX switches at a fixed camera; Animated -> Rest retains scale; accumulation settles. No new test files or builds run for this fix.

### Rest pose: Vulkan rebuild follow-up

- Vulkan pose synchronization restores skinning after constructing replacement BLAS/TLAS buffers; the previous slow path skinned old buffers before replacing them. A focused VulkanBackend_Pose module caches current skin matrices and reapplies them after RT/raster geometry rebuilds, including direct UI rebuilds, without advancing animation.
- Legacy project migration is explicitly out of scope at user request. The inspected older skin_animation.rtp has no persisted nodeHierarchy and contains an Armature helper with scale 100 and a 90-degree rotation; no compatibility migration was added. Validate current imports and newly saved/reopened projects instead.
- Pending runtime verification: fresh skinned FBX import in Solid then Vulkan RT without playback; Animated/Rest switches retain size and orientation; save a new project and reopen in Rest; fixed-camera accumulation settles. No builds, application launches or new test files performed.

### Rest pose: first Vulkan render pipeline readiness

- Skinning compute pipeline creation is deferred until the first Vulkan render. Pose dispatch before pipeline readiness returns without skinning; Rest idles thereafter, while New Project works because the pipeline already exists. On successful first pipeline creation, replay the cached current pose onto the existing BLAS/TLAS before tracing.
- Reuse an already valid skinning pipeline during output-resource initialization rather than recreating it. This preserves existing skinning descriptors and avoids unnecessary setup.
- Pending user check: cold application launch, fresh skinned FBX import, switch directly from Solid to Vulkan RT without starting animation or using New Project; compare scale and orientation. Then repeat after New Project and with Animated/Rest toggles. No builds or application launches performed.

### Rest pose: CPU render and picking synchronization

- GPU-to-CPU synchronization reapplies current skin matrices to canonical flat mesh P/N from bind channels before rebuilding the CPU BVH. This does not advance animation and runs once per CPU synchronization request.
- Generic picking transform synchronization and legacy triangle transform wiring skip weighted parent meshes so they cannot overwrite canonical skinned vertices with raw bind positions.
- Pending user check: cold import in Rest, Solid/Vulkan RT/OptiX/CPU mode changes with a fixed camera, mesh picking at visible body position, Animated/Rest and repeat GPU-to-CPU transitions. No builds, launches or new test files performed.


### Phase 1 producer contract and diagnostics checkpoint (2026-09-13)

Continued from multipart manual fitting. Shared Animation/SkinWeightContract.h
canonicalizes newly imported flat weights in glTF, ufbx and Assimp publication
paths: removes negative IDs/nonfinite/nonpositive weights, merges duplicate IDs
in double precision, deterministically retains strongest four (bone ID tie-break),
normalizes retained weights. Empty rows remain empty; no invented fallback bone.
ufbx previously allowed more than four influences; glTF did not normalize after
filtering. Existing native snapshots are not migrated or rewritten by this delivery.

rig.weight_stats(mesh) is shared by core, API, Python, IPC and existing Edit Bone
Mesh setup collapsed diagnostics. It inspects actual individual flat TriangleMesh/
DNA weights, never merged diagnostic targets or Triangle facade geometry. Reports
max_influences, min/max sums for nonempty rows (null if none), empty rows, invalid
entries, duplicates, order violations, over-limit/unnormalized vertices and extra
rows. contract_valid allows empty rows; fully_weighted is independent. Tolerance
is 1e-5. Bone index ownership/range is explicitly not verified. Exact mesh names
only; multipart binding must operate on actual parts.

This bounded foundation is source-delivered, not the entire Phase 1 or Phase 3:
get/set_weights, ownership validation, visualization, live edits/undo/backend
invalidation and initial binding/weight generation remain open. Existing copy,
topology and native-load paths were audited but are not claimed to repair arbitrary
weights universally; future producers must use the common contract. Next: finish
remaining Phase 1 contracts and bind fitted owned rigs over actual flat target
parts with explicit bind spaces and shared mutation/undo/invalidation.

No builds, application launch or new test files. User build/runtime checklist:
1. Build RayTrophiStudio using your normal configuration.
2. Fresh weighted FBX/glTF/Assimp import: choose each individual mesh in Mesh setup;
   Inspect skin weights. Expect max <= 4, nonempty sums near 1, zero invalid,
   duplicate, unsorted, over-limit and unnormalized counts; empty rows are separate.
3. Python rt.rig.weight_stats(mesh="exactNodeName") and IPC rig.weight_stats with
   {"mesh":"exactNodeName"} must match UI and leave rest/pose/history unchanged.
4. Unskinned targets report null sums and unweighted rows; unknown, ambiguous and
   group-only names fail without changes. New project/reimport clears UI snapshots;
   inspect again after edits. Existing native project snapshots stay unchanged.
5. Compare viewport/CPU/Vulkan RT/OptiX deformation on an imported asset with >4
   source influences; verify shared top-four output and Rest/Animated behavior.


### GLB Animated helper-root correction (2026-09-13)

User confirmed Phase 1 build/runtime. Separate GLB Rest/Animated scale-axis bug:
read-only binary inspection of sit.glb shows Armature static scale .1 and +90deg X;
skin_anim.glb has two Armature roots at scale .01 and +90deg X. Their animation
channels start at hips and do not animate these helper roots. glTF import retains
ancestor closure and unique node identities, including duplicated authored bone
names in skin_anim.glb. FinalPoseNode instead used identity for unindexed helpers
absent from extraTransforms, discarding those root transforms only in Animated.
This predicts 10x/100x enlargement and missing axis correction. No file rewrites.

Focused Animation/GraphRestDefaults module now supplies canonical local rest TRS
for indexed/extra clip nodes before keyed component overrides, and exact local rest
matrices for missing graph helper transforms. This matches controller missing-track
semantics and retains root unit/axis transforms through graph hierarchy evaluation.
AnimationNodes.cpp receives only include/call wiring and obsolete-comment removal;
new module registered in vcxproj. Existing UI/Python/IPC pose-view operations share
the repaired graph; no new authoring operation or binding duplication.

Source/static XML/integration checks only; no builds, app run or new test files.
User checklist: fresh import each GLB in Rest; switch Animated without moving actor,
check size and orientation stable (pose/motion may change). Pause, scrub several
times, Rest/Animated repeat; Solid/Vulkan RT/OptiX/CPU mode checks. Save a new native
project and reopen. Recheck working FBX and multiple independent animated models.
This is a runtime graph bug diagnosis, not full glTF asset conformance certification.


### Phase 1 vertex reads and ownership diagnostics (2026-09-13)

User opted to continue standard authoring delivery instead of chasing every GLB
export variant; previous graph fix awaits their normal build. Focused canonical
Animation/RigWeights service now owns weight_stats and get_weights(object, vertex).
UI existing Mesh setup, Python and IPC call the same API/service. get_weights reads
exact stored flat DNA vertex rows without sorting/normalizing, exposes bone index,
unique name, numeric validity, ambiguity and character membership. Nonfinite weights
serialize as null; empty/missing rows stay empty and row presence is explicit.
Object is an exact individual mesh nodeName, vertex zero-based uint64. Groups are
fitting diagnostic targets, not authoritative skin geometry. Out-of-range including
uint64 max fails rig_vertex_out_of_range; IPC negative/bool/float invalid_parameter,
Python typed conversion semantics are documented. Both methods require Read only.

weight_stats keeps existing numeric/top-four contract_valid semantics and adds
unknown/ambiguous/foreign index counters. Forward BoneData index map is authoritative;
index gaps and multiple names sharing an index fail index_range_valid. Mesh character
ownership uses direct flat members or exact NodeHierarchy unique keys, never facade
geometry/authored-name/prefix guesses. Multiple matching contexts report ambiguous;
missing context/hierarchy remains unverified rather than inventing ownership.
bone_indices_verified additionally requires resolved hierarchy membership and no
foreign/unknown/ambiguous indices. weights_valid combines contract_valid, fully_weighted
and bone_indices_verified; it does not certify bind matrices or deformation quality.
UI shows aggregate ownership errors and one selected flat vertex influence table.
Snapshots refresh explicitly; new project/target resets them.

Existing rig_view_contract.py and test_rig_view_ipc.py now contain optional shared
live weight checks (no new test files): bounds including uint64 max, repeat read and
failed-query immutability, selection/mode/overlay preservation, ownership/coverage.
No live test run, builds or app launch. Static syntax/XML/registration and generated
IPC capability audits only. Existing native snapshots are untouched.

User grouped checklist after normal build:
1. Fresh weighted character, Mesh setup individual body/clothes target -> inspect
   weights; known owner/indices and zero foreign/unknown/ambiguous counts expected.
2. Inspect flat vertex 0/middle/last; names and sums consistent with statistics.
   Unskinned mesh returns empty influences; invalid vertex fails without mutation.
3. Pause animation. Embedded Python: load scripts/test/rig_view_contract.py and call
   run_weights("exactMeshNodeName"). External IPC (app already running):
   python scripts/test/test_rig_view_ipc.py --character Character --mesh exactMeshNodeName
4. Both APIs should match UI, preserve selection/mode/rest/pose and leave undo unchanged.
   Multiple imported rigs remain independent; native save/reopen preserves weights.

Phase 1 read/diagnostic surfaces delivered, live editing/repair and universal copy/
topology writer enforcement not claimed complete. Next user-visible milestone is
Phase 3 explicit owned-rig mesh binding and initial nearest-segment weights over
actual canonical flat parts, with actor/rest/bind-space checks, shared undo and
CPU/GPU invalidation. Do not expand template catalogue or call manual fit automatic.


### Issues 1.2/1.3/1.4 audit and unskinned mutation guard (2026-09-13)

User requested current issue closure status rather than only appended delivery logs.
Updated status banner, original issue subsections and phase checkpoint in place:
1.3 joint-global/skin separation resolved in source; 1.2 stable-index unskinned leaf
removal delivered but weighted transfer/remap open; 1.4 owned unweighted rebuild/
persistence delivered but live weighted influence invalidation/rebind open.
Existing runtime pose replay is explicitly not an influence-buffer edit service.

Audit also found unskinned mutation staging relied on metadata without checking
actual flat weight references. Added shared RigWeights::hasFlatSkinReferences to
owned rig edit staging, placement staging and Edit entry. Scans actual flat scene
geometry once per distinct DNA buffer; any stored reference to an owned hierarchy
index blocks, even zero/invalid entries or malformed extra rows. No per-face source,
no index renumbering/transfer and no automatic data repair. Normal capability/UI
frame queries do not scan every skin buffer. Existing named errors unchanged:
rig_edit_requires_unskinned / rig_placement_requires_meshless_unskinned. UI/Python/
IPC use the same gate through existing authoring services. This closes a prevention
gap in the supported subset, not the full weighted deletion milestone.

No builds, app launch or new test files. Source/header/registration and descriptor
checks only. User checklist: normal chain3 add/rest/reparent/leaf delete (indices of
survivors stable), undo/redo/native reopen and four representation flags; weighted
imports still reject owned edit; if a developer injects a flat influence referencing
an owned rig index while clearing weighted metadata, Edit entry/rest/delete/actor
placement must reject before scene/history mutation, including zero/invalid entries.
Unrelated imported skin weights must not block an unreferenced owned rig. Next:
Phase 3 canonical actual-part bind/weight transaction and influence invalidation,
then weighted delete/remap/transfer through that same mutation core.


### Initial flat multipart binding and nearest-segment weights (2026-09-13)

User confirmed all preceding work compiled and ran without observed issues. This
new bounded Phase 3 package is source-delivered; build/runtime acceptance pending.
Focused RigBinding/RigBindMath/RigBindingScope modules own preview, staging and
canonical ownership; RtApiRigBinding owns one production history transaction.
Existing inspector adds a collapsible binding section after manual alignment.
Python/IPC rig.preview_bind(character,mesh,axes_confirmed=False),
rig.bind_mesh(character,mesh,preview), rig.get_binding(character) call the same API.
Preview and get_binding require Read; bind_mesh requires SceneWrite. Descriptor
and capability mirror updated together.

Scope: initially meshless, clip-free authoring-owned rig and static unskinned flat
parts, exact individual nodeName or model:<importName> group. Resolve actual scene
TriangleMesh pointers, not temporary merged diagnostics or Triangle facades. Keep
part names, separate meshes, indices, material IDs, UV and named attributes. Clone
DNA; convert current P by inverse(A)*M into common rig bind space (A actor placement,
M source base). Transform N by inverse-transpose and normalize. Rest world position
is invariant: A * (inverse(A)*M*P) = M*P. Common actor Transform, per-model inverse I,
offsets inverse(global joint rest) give identity skin at Rest. Normals must exist
and be complete/finite/nonzero. Reject source animation, existing skin, modifiers,
geometry graphs/deltas, pending deletions, invalid geometry and singular/reflected
transforms. Degenerate faces warn if usable faces remain. No automatic axis fix.

Segments connect parent to child and weight the parent bone. Anatomy root role is
nondeforming; each bone uses nearest outgoing segment, never sums duplicate segments.
Inverse distance-squared regularization uses extent*1e-4 (minimum 1e-9), strongest
four, deterministic ID tie-breaking, common merge/normalize and negligible-weight
pruning. Preview reports coverage/sum bounds, weighted bone count, distances, outside
bounds warnings and at most 16 sampled rows per part. Caller must inspect rest axes
and alignment; surface interior, visibility and alignment verification are false.
Limits: 256 parts, 2M vertices, 100M vertex-segment evaluations, 4096 joints.

Commit regenerates weights and checks revision plus rig/mesh input tokens, ignoring
caller-submitted weights/can_bind. Stale input rejects before scene/history changes.
Transaction publishes geometry/transforms, offsets, weighted metadata, hierarchy-derived
skeleton/ozz runtime, selected target view and affected source membership vectors.
Undo/redo swap these snapshots and invalidate animation groups, CPU BVH, OptiX,
Vulkan and raster geometry/influence buffers via existing mutation refresh service.
Other rig skin matrices and pose-view choices remain independent. Exact
ImportedModelContext.rigBoundMeshes registry persists with the native snapshot;
restoration attaches actual flat parts and removes former source memberships.
Ownership diagnostics/rest skinning respect registry before original source keys.
Missing/deleted parts retain names with present=false, without fabricated geometry.

This does not close full Phase 3 quality acceptance or issues 1.2/1.4. No weight
painting, repeated bind append, imported weighted rebind, weighted delete/transfer,
weighted rest editing, Pose/IK or automatic interior/visibility solve is delivered.
Bound rest/topology editing stays gated. Compatible clip binding remains the existing
animation path; Rest/Animated is a view switch, not pose authoring.

No builds or app launch. Existing rig_pose_preview_test.cpp extended with scaled
actor/world-rest invariance, singular/reflected rejection, segment clamping, duplicate
segment/tie order and negligible-weight checks; link RigBindMath.cpp when user builds
that existing standalone test. Existing rig_view_contract.py check_binding is shared
between Python/IPC and checks preview/error immutability, ignored forged weights,
part identity, numeric/index/ownership validity and production undo/redo. Embedded
run_binding(character,mesh) mutates a disposable aligned fixture only when called.
Existing rt_test_rig_roundtrip.py run_binding(new_output.rtp,character) checks actual
native registry, weight rows, ownership and four skeleton representations after reopen.
No new test files and no live tests run by Codex.

User grouped acceptance after normal build:
1. Disposable unskinned body/clothes model, create owned rig, manually align in both
   fitting views. Binding section: confirm axes/rest, Preview initial weights.
   Preview must preserve mesh/rig/selection/history; inspect part/vertex counts.
2. Change a landmark/rest or source transform after preview: old Apply must reject
   stale preview. Re-preview and Bind: preserve world Rest size/orientation/position,
   names, separate part/material/UV identity. Weight inspector should report valid
   top-four coverage, owned indices and zero foreign/unknown/ambiguous entries.
3. Undo/redo repeatedly in CPU, Solid/raster, Vulkan RT and OptiX; geometry and skin
   buffers must restore together. Another independent animated rig must keep working.
4. Save to a new native .rtp, reopen and repeat Rest/Animated and weight diagnostics.
   Use existing compatible clip mapping to inspect actual deformation where applicable;
   nearest-segment quality is baseline and may need later correction.
5. Optional embedded checks: load existing rig_view_contract.py, call
   run_binding("OwnedRig","model:SourceModel") on a fresh disposable fixture; then
   load rt_test_rig_roundtrip.py and run_binding("NEW_OUTPUT.rtp","OwnedRig").

   External IPC alternative on a separate fresh disposable aligned fixture:
   python scripts/test/test_rig_view_ipc.py --character OwnedRig --bind-target model:SourceModel
   This explicit flag performs binding and production undo/redo; ordinary invocation
   keeps its existing inventory/read-only weight checks.

Static delivery checks passed: existing Python test AST, descriptor JSON, project XML,
all ten new header/source registrations, focused source include/delimiter and shared
UI/Python/IPC wiring checks. Generated descriptors/capability mirror agree for 480
methods; git diff whitespace check passed. Build and runtime checks remain with user.


### Selected bone weight display; sculpt tint reuse (2026-09-13)

User proposed reusing the existing sculpt mask display for selected bone weights,
with weight painting later. Delivered a display-only checkbox in the existing rig
inspector: Show selected bone weights. Follows canonical bone selection for owned
and imported weighted flat meshes; independent of skeleton overlay visibility.
Transient ViewState.weight_map_visible defaults off, is not persisted or recorded
in undo and does not mark the project modified. Sculpt protection values, show flag,
attribute and workspace/tool selection are unchanged.

Shared RigWeights::boneWeightField resolves exact mesh ownership and unique forward
bone index. Returns one scalar per actual DNA vertex, missing rows zero, sums matching
duplicate positive/zero finite entries in double, ignores and counts matching negative/
nonfinite entries, clamps only display values to [0,1]. It never repairs stored skin.
API/Python/IPC rig.get_weight_map(mesh,character,bone),
rig.set_weight_map_visible(visible), rig.get_weight_map_visible() are delivered with
named errors and descriptors. Read operations require Read; toggle uses existing
rig SceneWrite policy despite being transient. Owner/index/foreign failures do not
silently show another rig's weights. rig.weight_stats remains numeric/ownership
validation, not replaced by a display field.

New focused ScalarFieldOverlay triangle primitive is shared by existing sculpt tint
and new RigWeightMapUI. Existing over-2000-line sculpt/gizmo modules receive only
include/call wiring and removal of the now unused original tint helper. Weight display
reads actual current flat deformed P/N plus object transform, never editable-cache
Triangle facade refs or a merged geometry source. Blue alpha gradient matches sculpt
mask rendering and follows animation. Background ImGui layer stays behind panels.
Like the existing sculpt tint, this is not depth-buffer-tested: backfaces are culled,
but overlapping front surfaces can bleed through. The inspector states that limit;
display is bounded to 2M vertices/120k examined faces and dense parts can be omitted.
No GPU shader/build change or new sculpt/weight paint mode is delivered.

Existing rig_view_contract.py extends the shared live contract with check_weight_map
and embedded run_weight_map(mesh,character,uniqueBoneKey); validates raw vertex row
agreement/clamping, repeat read, ownership errors, selection/mode/weight immutability
and independent toggle restoration. Existing test_rig_view_ipc.py supports
--character Character --mesh exactMesh --weight-bone Character_UniqueBone.
No new test files. Static syntax/project registration/IPC descriptor/capability
checks only; no compilation or application launch by Codex.

User grouped checks after normal build:
1. Select a weighted bone, enable Show selected bone weights. Strongest areas should
   be blue; select another bone and check body/clothes maps change appropriately.
2. Rest/Animated, pause/scrub, orbit and compare Solid/CPU/Vulkan RT/OptiX. Tint must
   follow current mesh deformation/object placement; overlapping-surface limitation
   is expected for this existing overlay path, not certified surface visibility.
3. Toggle off; sculpt protection mask and weights remain unchanged. Switch rigs,
   including imported rigs and unrelated skinned models; only selected owner tints.
4. Optional shared Python/IPC checks above on a paused exact flat mesh. Existing bind
   and undo/redo/reopen checklist remains pending user validation for the prior package.

Next weight-paint capability can use this same scalar display and canonical read
service, but writes must be a shared skin mutation service with bone-owner validation,
common strongest-four/normalization semantics, undo and CPU/GPU influence invalidation.
Do not treat the sculpt protection array as authoritative skin weights or implement
UI-only writes. General weighted delete/remap/transfer and visibility/quality still open.

Delivery static checks passed: existing Python test AST, descriptor JSON/project XML,
five new module registrations/headers/delimiters, common scalar primitive used in both
overlays, script/IPC parity and no weight/sculpt writes in display path. Descriptor
and capability audit passes for 483 methods. Build/runtime acceptance remains pending.


### Multi-selection and mirror phase assignment (2026-09-13)

User is currently compiling; this request adds planning only, no new implementation
or runtime acceptance claim. Phase 2A covers canonical selection set/active bone and
batch rest transforms; Phase 2B covers paired mirror rest editing and mirror creation.
They precede further rest inspector polish and remain inside existing ownership/skin
edit gates. Pose/control mirroring stays Phase 7. Weight-map mirroring belongs to the
future common weight-paint/edit service after Phase 1/3 writer and invalidation rules,
with vertex correspondence and bone-pair mapping; it is not the Phase 2 rest operation.
Selection, rest, pose and weight operations must expose the same core through UI,
Python and IPC with named validation errors, preview where appropriate and undo.


### Segment selection / joint identity correction (2026-09-13)

User compiled and reported working weight display, then noticed the elbow-to-wrist
segment selected Hand and displayed hand-region weights. Static audit confirmed the
viewport used incoming parent->child lines but assigned child identity/highlight.
RigBinding consistently assigns outgoing segments to their starting parent joint:
Forearm at elbow -> Hand at wrist, then Hand -> HandEnd for the hand segment.
The scalar map correctly read the selected Hand index; this is a visual/picking
ownership mismatch, not evidence of shifted stored skin weights.

Focused scene_ui_rig_overlay now uses the starting joint for segment picking,
weighted color and selected highlight. Joint circles keep their own exact identity
and take precedence over segment hits globally, avoiding endpoint ambiguity.
Labels/gizmo remain at the selected joint. Existing select_bone UI/API/IPC core
and canonical bone IDs/hierarchy/weights are unchanged; no renaming or rebind.
Branching joints own all outgoing links; terminal joints remain pickable as dots.

Source/delimiter and whitespace checks only; no compilation/application launch.
User checklist: click forearm segment middle -> Forearm, orange elbow->wrist,
elbow pivot and forearm scalar map. Click wrist dot -> Hand; select Hand through
hierarchy/Python/IPC -> orange wrist->HandEnd and hand map. Repeat upper arm/leg,
branch/terminal joints and Rest/Animated; scoped Edit/gizmo/UI input locks preserved.
User subsequently compiled and confirmed the segment/joint selection correction
now works correctly. This targeted acceptance does not certify all bind/undo/native/
backend cases, which retain their separate checklist.


### Phase 2A complete source delivery: multi-selection and batch rest (2026-09-13)

The interrupted turn delivered core files only, not a completed phase. User asked
explicitly to finish. Remaining UI/gizmo/API/history/registration/docs are now delivered;
this is source completion and awaits user compilation/runtime acceptance.

RigSelection owns a transient scene selection set, separate active bone, range anchor
and active/center pivot preference. Names are canonical unique joint keys, not skin
indices. One character only. Legacy select_bone replaces with one member;
get_selected_bone remains active bone; clear_selection clears complete set/anchor.
Ctrl toggles and Shift selects an inclusive depth-first hierarchy range in viewport
and tree, including collapsed branches. Another character starts a new set in Scene;
scoped Edit rejects switching character. Rig Edit Shift-drag blank space box-selects
joint origins, Ctrl+Shift box adds; Escape/input locks/load/mode changes cancel box.
Terminal joint dots and starting-parent ownership of segments stay correct. Active
orange, remaining selected green; active inspector and weight map stay single-bone.

RigViewportUI rest gizmo captures immutable hierarchy, selection, active/pivot and
revision at drag start. Active pivot or arithmetic mean of selected world joint origins;
center orientation stays active joint basis, existing World/Local option preserved.
Translate/Rotate only in Rig Edit; Scene G/R/S still moves the complete owned meshless
actor. Pure RigBatchRest helper conjugates rigid world delta through actor placement,
sets selected original globals as targets, derives locals using updated parent globals.
Selected parent+child move once; unselected descendants inherit their normal hierarchy
motion. Every preview is baseline-to-target, not accumulated per-frame. Escape, loss
of available viewport, scene load/revision/selection/pivot/mode/operation changes cancel.
Commit is one existing RigEditCommand transaction. Full selection snapshots now exchange
with model/bone data for undo/redo (including initial binding command integration).
Shared authoring history allocates creation capacity and records before canonical
execute; no fallible history allocation after publishing the edit.

Owned meshless/unweighted/clip-free authoring gate and actual flat skin-reference scan
still apply to batch rest. Common finish rebuilds BoneData, skeletonNodes and static
controller/ozz runtime from canonical NodeHierarchy, preserves indices, increments
revision, and native snapshot preserves changed rest. Selection/pivot are not serialized;
existing native clearSelection resets set/active on reopen. No Triangle facade geometry,
GPU shader changes, weighted rebind, batch topology operations, pose editing or mirror.

New shared API/Python/IPC surfaces:
- rig.select_bones(character,bones,active="",mode="replace",anchor="")
- rig.get_selection() -> {character,bones,active,anchor,pivot}
- rig.set_selection_pivot(mode="active" or "center")
- rig.transform_rest(character,bones,world_delta,rig_revision)

transform_rest takes explicit unique target keys and expected revision from list_bones,
not implicit UI selection; target list becomes selection, existing active target retained
else last input active. Delta is 16 row-major finite numbers, rigid affine; rotation
about a chosen point uses T(p)*R*T(-p). Old revision rejects rig_edit_stale_revision;
empty/duplicate/unknown targets, invalid active/mode/pivot, scale/reflection/nonfinite,
no-op and existing ownership/skin/clip gates reject with documented named errors before
scene publication. get_selection is Read, mutators use existing rig SceneWrite policy.
Descriptors/generated capability mirror updated together.
The optional anchor restores the full transient selection, including the starting
bone for subsequent Shift ranges. Batch rest preserves a valid existing anchor.

Existing tests extended, no new test files and none executed live by Codex:
- rig_pose_preview_test.cpp: independent parent+child translation/rotation, unselected
  ancestor/descendant inheritance, source immutability, scaled rotated actor/world delta,
  duplicate/empty/unknown target and scale/reflection rejection. Link RigBatchRest.cpp
  and RigBindMath.cpp for this existing standalone test when user builds it.
- rig_view_contract.py: check_multiselection / run_multiselection(character) transient
  modes/active/legacy/error/pivot checks with selection restoration. check_batch_rest /
  run_batch_rest(character) explicit disposable owned fixture checks one world delta,
  stable indices/four representations, stale rejection and production undo/redo.
- test_rig_view_ipc.py: --multi-selection, --batch-rest require --character. --batch-rest
  explicitly mutates a disposable fixture. Existing inventory check preserves full
  selection rather than collapsing it to old active; Rest pose source accepted.
- rt_test_rig_roundtrip.py: run_rest(new_output.rtp,character) validates actual native
  hierarchy/bone snapshot, actor/rest transforms, representations and cleared selection.

User grouped acceptance after normal application build:
1. Disposable owned chain3 or humanoid, Scene mode: Ctrl select/toggle in viewport and
   tree, Shift range. Active orange/others green; get_selection and inspector agree.
2. Enter Rig Edit, Shift-drag blank space box, Ctrl+Shift adds; Escape cancels box.
   Select parent+child together, Move with gizmo/G mode: both move once, pivot at active
   or center. Rotate/R around each pivot; child cannot receive the delta twice.
3. During a drag Escape restores original rest and does not add undo. Release gives one
   undo/redo step for entire selection. Changing selection/pivot/revision cancels drag.
4. Save a new .rtp and reopen: all rest changes/actor placement and four rig representations
   survive; transient selection empty. Existing weighted/bound rig allows selection/map
   inspection but rest edit stays rejected. Other rigs/clips remain independent.
5. Optional embedded existing tests run_multiselection("OwnedRig"), then
   run_batch_rest("OwnedRig") on a fresh disposable clip-free rig, and optional run_rest
   to new native output. IPC alternative on same suitable disposable fixture:
   python scripts/test/test_rig_view_ipc.py --character OwnedRig --multi-selection --batch-rest

Next Phase 2B: paired mirror rest edits/creation. It is planned, not delivered by 2A.

Final non-build checks passed: existing Python test files parse, descriptor JSON
parses, all six new source/header project registrations exist with no-PCH settings,
and the IPC capability audit validates all 489 dispatched methods against the
security mirror with generated descriptors current. C++ compilation and runtime
acceptance remain pending the user's build and grouped checklist above.

### Phase 2A user check and fit mesh coordinate correction (2026-09-13)

User compiled and confirmed multi-selection works without problems. This confirmation
covers multi-selection; grouped batch rest, pivot, undo/redo and native round-trip
checks are still awaiting user runtime confirmation.

User reported a correctly placed mesh appeared X-rotated by 90 degrees in the
Front/Side fitting panel; rotating the viewport mesh by 90 degrees compensated it.
Source inspection found fitting and preflight transformed current P by object base,
while static viewport geometry uses local P_orig and final object placement. P may
already be world-baked, so the old path could transform it twice.

RigMeshSpace now selects static local P_orig/N_orig (fallback P/N) from flat DNA.
Fit samples, bounds, multipart assembly and stale-content token use the same source
and final object placement once. Initial binding uses those same local positions
and normals with its existing static-transform guard. UI/Python/IPC continue through
the common services. No import rotation or skeleton rest correction was added.

Existing rig_pose_preview_test.cpp includes a 90-degree X rotation plus translation
fixture with baked/stale P distinct from P_orig, primitive fallback and skinned P
selection. It has not been compiled or executed by Codex. After user rebuild: restore
the mesh to its correct viewport placement, press Prepare alignment again and compare
Front (X/Y) and Side (Z/Y). Also check a translated/rotated multipart target and ensure
applying alignment/binding preserves viewport placement.

### Alignment window source delivery (2026-09-13)

User requested a separate resizable alignment panel with richer navigation.
The right Rig dock now contains Open Alignment and short status; full fitting
canvases live in Skeleton Alignment. The window renders from the UI frame so
switching right-dock contexts does not hide an open draft. It is transient,
with no permanent viewport shelf or project data added.

Front (X/Y) and Side (Z/Y) sit side by side in resizable columns; optional Top
(X/Z) edits the third projection. Each camera has independent cursor-anchored
wheel zoom (5%-10000%) and middle-button pan. Fit to View / Reset fits mesh bounds
and all draft landmarks and resets all cameras. Window resize retains camera
center/zoom. Mesh samples, skeleton and selected-joint labels have visibility
controls; mesh opacity is adjustable. The mesh remains the bounded 8192-point
preview from the common setup service, rather than a surface renderer.

Left-drag edits the two displayed world coordinate axes and leaves the third
unchanged. All views share one draft landmark map. The joint dropdown selects
overlapping landmarks; numeric world XYZ edits reject nonfinite values. Escape
restores the dragged landmark's original coordinates; closing or Cancel discards
unapplied draft state. Prepare regenerates coarse alignment and resets views and
axes confirmation. Successful Apply preserves the window/status, clears its
consumed draft and creates the existing single undo step. Reload closes the panel.
Opening a different rig/mesh replaces the previous transient draft.

Canonical authoring operations remain delivered together through getRigFitSetup,
previewRigFit and commitRigFit: UI calls these same services as Python
rt.rig.get_fit_setup / preview_fit / commit_fit and matching IPC methods. Camera,
visibility, opacity, window placement and drag state are presentation state; they
do not mutate scene geometry or add scripting/IPC business logic. Prepare and
Preview are read-only; only Apply publishes rig rest. Before preview, the UI
compares current setup revision/content token with the displayed draft and asks
for Prepare if it is stale. Commit retains the shared core stale-preview checks.
No automatic anatomical fit, weight painting, skeleton mirror or new bind logic
is included in this panel delivery.

Build/runtime acceptance remains user-run. Validate the grouped panel checklist
in RIG_VIEW_BUILD_CHECKS.md, including zoom/pan isolation, Top's unchanged Y axis,
Escape/Cancel immutability, stale preview rejection and existing Apply undo/redo.

Final panel source checks passed: balanced C++ delimiters, focused file size,
one frame registration, calls to all three common authoring services, project
XML/descriptor JSON and generated descriptor freshness. Projection directions
match viewport Front, Right (Side: -Z right) and Top (-Z up); picking, dragging,
zoom and pan invert those same signed projections. No build or application
launch was performed.

Viewport rest-gizmo initiation now respects UI mouse capture, and its G/R/S
shortcuts respect keyboard capture. Alignment pan also refuses to start while
a viewport gizmo is using the mouse. Final delimiter/source checks passed after
these input guards; runtime behavior remains user-run.

### Alignment Apply usability and requested next steps (2026-09-13)

User confirmed the expanded Alignment window looks good, but Apply stayed disabled
and edits seemed panel-only. Source gates required axes confirmation and a fresh
explicit Preview with every joint inside bounds; editing invalidated that Preview.
Apply now validates the current draft itself before the common commit, so a separate
Preview click is optional. Axes confirmation still gates both actions. Panel explains
draft vs scene rest, lists/selects outside-bounds joints and colors them red; bounds,
stale request, finite positions, nonzero segments and owned/unweighted/clip-free
core validation remain unchanged. After Apply, the existing Bind mesh to skeleton
section is accessible in the Alignment window too, using its existing service/API.

Initial automatic skin weights are already source-delivered: nearest outgoing
parent-owned segment distance weights, strongest four normalized, actual flat
multipart geometry, common actor bind space, saved ownership and one undo/redo.
This is a starting bind, with deformation quality and manual corrections still
user-validated; it is not a heat/voxel/geodesic or learned weight solver. User runtime
verification of this complete alignment-to-bind path remains pending.

Requested follow-up scope, planned (not implemented by this UI correction):
- Phase 2B: Alignment draft mirror using existing anatomy symmetry pairs, a
  selectable/offset symmetry plane, live opposite-landmark editing and explicit
  left-to-right/right-to-left commands. Shared service, Python/IPC validation and
  existing tests must accompany delivery; do not rely solely on bone name guessing.
- Fitting phase: anatomical auto-landmark proposals by rig family. Start with
  humanoid T/A-rest geometry and roles/chains, landmark confidence, unsupported-pose
  rejection and manual correction before Apply. Quadruped/avian/insect families
  need separate rules and validation; coarse height scaling is not auto-detection.
- Binding/weights phase: compare initial distance bind deformation and improve
  surface-aware weighting as needed, followed by existing sculpt-based weight
  painting/normalization and weight mirror.

Mixamo-like joint proposals are product direction, not a claimed current capability.

Alignment layout compacted following user feedback: reduced local padding,
two-row toolbar/footer, help tooltip, outside-joint dropdown and measured
footer/status reservation give canvas space back to the projections. No
authoring service or scripting/IPC operation changed. User visual/runtime
verification remains pending; non-build source checks passed.

### User-validated bind path and animation priority (2026-09-13)

User compiled and confirmed Alignment Apply updates scene rest, binding works,
and initial weights appear correctly assigned. This is positive acceptance of
the first alignment-to-bind path; it does not establish full deformation quality,
bind undo/redo, native reopen, all anatomy families or every asset. Earlier
pending notes for these same basic actions are superseded by this checkpoint.

Current execution priority requested by the user:
1. Phase 2B: mirror Edit/rest operations and symmetric bone creation, plus
   paired Alignment draft edits using anatomy pair metadata. Do not unlock
   bound weighted rest edits without explicit rebind/clip migration support.
2. Phase 7A + 8A: bound-rig Pose mode, FK interaction and bone-channel key authoring
   with Auto Key. This is the highest animation priority after 2B; automatic
   landmark inference and advanced heat weighting need not block this milestone.
3. Phase 7B: limb IK, pole controls, foot planting and contacts supporting movement.
4. Phase 9A: quick footprint placement/editing and a timed contact recipe.
5. Phase 9B: spline-guided walk/run with root motion and generated contacts.
6. Phase 9C: spline-guided flight with orientation/banking and anatomy-aware
   wing cycles; airborne motion has its own recipe rather than foot contacts.

7A/8A bounded first delivery contract (planned):
- Distinct Pose mode operates on evaluated local pose offsets or canonical
  editable bone channels. Rest hierarchy, inverse bind offsets, actor bind
  placement and flat stored weights remain unchanged. Multiselection/active
  control/pivot follow the existing selection contract.
- Pause playback for pose editing; one evaluator owns the authoring result
  before producing joint globals and skin matrices. Viewport, CPU picking and
  GPU deformation must agree. Resolve controller/ozz/graph evaluation ownership
  explicitly; Rest/Animated display switch alone is not a Pose editor.
- Create/select an editable target clip and timeline time, with explicit bone
  key insertion plus optional Auto Key. Existing object transform keys are
  infrastructure to inspect, not a substitute for skeleton-local channels.
- Auto Key commits affected channels at the current time when a pose gesture
  finishes; update an existing key at the same time, quaternion rotation
  continuity and canonical time units required. Escape/cancel creates no key.
  One gesture yields one undo step covering pose/clip changes; unkeyed pose
  preview has an explicit reset/exit policy. Imported clips remain separate
  unless deliberately made editable.
- Shared core operations, Python and IPC, named validation errors, native
  persistence and meaningful existing regression coverage delivered together.
  Accept with two keyed poses, interpolation, selected-channel Auto Key,
  cancel/undo/redo, CPU/GPU pose agreement and native clip reopen.

Movement recipe contracts (planned):
- Footprints store position/orientation, side/limb role and contact timing.
  Step length, speed, stride/cadence and stance/swing adjust the recipe; IK
  checks reachability and ground samples read canonical flat scene geometry.
- Reuse existing editable spline data where suitable. Arc-length sampling
  controls physical travel speed; path tangents guide actor/root heading.
  Walk/run blends gait and contacts without double-applying root movement.
  Degenerate/unsupported paths and unreachable steps return explicit errors.
- Flight needs a stable path frame, turn banking limits, speed/altitude and
  optional anatomy-aware flapping controls. No claim of aerodynamic simulation
  or a universal humanoid/avian/insect gait solver.
- Live preview and bake use the same evaluator. Recipe, spline references,
  timing, editable outputs and rebuild policy persist; baking produces a
  normal editable clip for existing timeline/curve tools.

All items above are next work, not delivered Pose/Auto Key/IK/motion features.
No application launch or build was performed for this planning checkpoint.

### Phase 2B source delivery: paired rest, creation and Alignment mirror (2026-09-13)

RigMirror is the common plane/pair/reflection core. UI, Python and IPC share
its services; no scene Triangle facade geometry or weight mutation was added.
Plane axis x/y/z and finite offset are canonical rig-space values. Reflecting
a source global matrix on both spatial and basis sides keeps proper handedness
instead of leaving negative scale. Parent/child target locals are solved from
immutable source globals against updated target parents, so targets mirror once.
Untargeted descendants inherit updated parents. Original source data is staged
and remains unchanged on validation failure.

Mirror rest / creation in the Rig dock:
- Mirror selected copies each selected paired source to its opposite. Center
  unpaired keys reject explicitly; selecting both sides of one pair is ambiguous.
- All Left -> Right / All Right -> Left copies every metadata pair in the chosen
  direction. Axis and plane offset are editable, default rig X=0.
- Create opposite bone takes exactly one unpaired non-root source, explicit
  new name and explicit source side. It creates one bone, not a subtree. The
  source parent maps to its paired opposite when available, otherwise remains
  shared. A symmetry pair is added; roles/chains remain explicitly configured.
- Existing slots remain stable; creation adds one slot. Shared RigEdit staging
  refreshes all representations/runtime and revision; one production history
  command preserves/restores model, anatomy and full selection. Weighted/bound
  rest edits, imported non-owned rigs and rigs with clips remain blocked.

Alignment panel:
- Compact Mirror checkbox enables paired live landmark editing; Mirror... opens
  axis/offset controls and directional copy commands. Offset initially uses the
  mesh bounds center converted into rig space, so translated targets do not
  accidentally mirror around world origin.
- All views/numeric inputs share the same world-space draft. Live edits reflect
  through actor placement * rig plane reflection * inverse(actor placement).
  Unpaired center landmarks edit independently. Invalid mirrored requests leave
  both sides unchanged; Escape restores the entire pre-drag landmark map.
- Mirror only changes the draft; existing Apply validates and commits one scene
  rest operation. Prepare resets mirror settings/metadata; Cancel/reload discards.

Delivered shared surfaces (Python names match IPC):
- rig.mirror_rest(character,bones,rig_revision,direction="selected",axis="x",offset=0)
- rig.create_mirrored_bone(character,bone,name,source_side,rig_revision,axis="x",offset=0)
- rig.get_mirrored_landmarks(character,landmarks,bones,rig_revision,
  direction="selected",axis="x",offset=0) -> complete new landmark map

get_mirrored_landmarks is Read and does not mutate scene/history/selection.
The two edit operations use existing rig SceneWrite policy and one shared command
service. Directional bones=[] means all pairs; selected requires nonempty unique
sources. Strict named errors cover unknown/unpaired/duplicate/ambiguous sources,
invalid plane/side/direction, stale revision, no pairs/no change, invalid matrices
and fit landmark shape/finite data, plus common anatomy/rig gates. Descriptor
overlay and generated method metadata accompany implementation.

Existing test files extended, no new test files and no compilation/live execution:
- rig_pose_preview_test.cpp: offset plane, paired parent+child positions and proper
  orientation, source immutability, invalid plane/ambiguous/unpaired/duplicate,
  world landmark reflection under rotated/translated/tiny actor and nonfinite
  rejection, opposite-parent creation and already-paired rejection. Existing
  standalone build now also links RigMirror.cpp and RigAnatomy.cpp alongside its
  prior RigPosePreview/RigBatchRest/RigBindMath dependencies.
- rig_view_contract.py: run_mirror(character) on a disposable owned meshless
  clip-free rig checks optional creation/pairs/representations/undo/redo, read-only
  mirrored landmarks, rest mirror position agreement, stable indices, stale
  rejection and full selection restoration. Leaves edits committed on that fixture.
- test_rig_view_ipc.py: --mirror --character requires that same disposable fixture.
- Existing run_rest native round-trip check can be used after mirror/creation to
  compare canonical hierarchy and derived bone snapshots on new .rtp output.

Source-delivered only; user build/runtime acceptance still pending. Next priority
is Phase 7A/8A bound-rig Pose/FK and bone keys/Auto Key. This 2B delivery does not
provide pose mirror, weighted rebind, auto-landmark inference, IK or locomotion.

Final 2B non-build checks passed: focused module sizes, balanced delimiters and
local includes, existing Python fixture syntax, descriptor JSON, all six new
header/source registrations and no-PCH settings. IPC audit passed for 492
methods/34 prefixes, with security mirror and generated descriptors current.
Python mirror landmark parsing maps invalid/nonfinite JSON text to the named
rig_fit_invalid_landmark error before core dispatch. No compile/live test
results are claimed.

### Phase 2B user acceptance (2026-09-13)

User compiled and tested the 2B delivery and reported it works without problems.
The paired Edit/rest mirror, opposite-bone creation and Alignment mirror delivery
is user-confirmed. This supersedes the preceding pending build/runtime status
for this delivery; no individual unreported regression or native round-trip test
is claimed. Weighted rest rebind and Pose mirror remain separate follow-ups.

Next active implementation priority: Phase 7A + 8A, bound-rig Pose/FK and explicit
bone-channel keys/Auto Key. Preserve rest hierarchy and inverse bind weights,
resolve pose evaluator ownership, and deliver common core/UI/Python/IPC, history
and clip persistence together according to the animation-priority contract above.


### Phase 7A + 8A source delivery (2026-09-13)

Bound-rig Pose/FK and explicit bone-channel keys/Auto Key are now source-delivered.
User build and representative runtime deformation acceptance are pending.

- Shared focused core: RigPoseAuthoringState, RigPoseAuthoringMath and
  RigPoseAuthoring. UI, Python and IPC call RtApiRigPoseAuthoring; no UI-only
  key insertion or pose mutation. This is a deterministic local service, with
  no AI subscription, network service or model inference dependency. CPU/GPU
  pose sampling and skinning still have runtime cost.
- Evaluation order: canonical rest hierarchy -> selected native authored clip
  -> committed per-frame absolute local pose -> replaceable preview -> skin.
  This service owns an active Pose rig before graph/controller/ozz evaluation.
  Selected parent and child world deltas apply once through common batch math.
  Authoring is initially owned bound rigs only. Edit and sculpt/paint/mesh-edit
  modes are mutually exclusive; exit Pose before rest editing or pose-view change.
- Pose panel in existing Rig dock: Enter Pose, editable clip creation/selection,
  frame, Auto Key, active local position/rotation, key selected/all, coverage and
  selected-bone weight map. G/R viewport gizmos support existing multi-selection
  and pivot; scale is unavailable for this rigid FK authoring milestone.
- Preview is transient and cancellable. Release/apply creates one history step.
  Auto Key off keeps the committed local pose only for the current frame;
  frame changes/mode exit clear unkeyed overrides. Auto Key on inserts/replaces
  affected local position/quaternion channels only when applying a changed pose.
  Explicit key insertion uses committed pose and rejects outstanding previews.
- Native authored clips use existing AnimationData position/rotation channels
  and a serialized rigAuthoring marker. Clips start with rest keys at tick zero;
  duration includes one tick beyond the final key. Scene FPS converts frames to
  seconds; clip FPS controls ticks per second independently. Quaternion signs
  are made continuous, duplicate times replace rather than append.
- Canonical node localBind, inverse bind matrices, skin weights and P_orig remain
  unchanged. Evaluated P/N use the flat TriangleMesh/DNA path, including CPU
  picking/weight overlays while GPU rendering is active. Exit to Rest restores
  CPU geometry as well as skin matrices. Empty weight rows stay at bind position;
  coverage inspection reports them without inventing weights or repairing bind.
- Clip/key history is copy-on-write and preserves interaction mode on undo/redo.
  Transient unkeyed pose history does not mark the project modified. Native save
  stores clips/keys, not preview, Auto Key or selected-clip session settings.
- Named validation includes owned/bound gates, empty/duplicate/unknown bones,
  rigid finite matrices, render-job lock, clip conflicts, frame/tick limits and
  stale rig revisions. Queries return explicit character, revision, local matrix
  map, frame/FPS and clip metadata for agents; no mouse coordinates are required.

Python/IPC operations (all shared-service backed): get_pose_state,
get_pose_coverage, create_pose_clip, select_pose_clip, set_pose_auto_key,
set_pose_frame, preview_pose_transform, preview_pose_locals, apply_pose_preview,
cancel_pose_preview, insert_pose_keys. Existing set_mode now supports pose.
IPC discovery descriptors explain ordering, matrix units, timing and errors.

Example embedded Python workflow for an already bound owned rig:

```python
rt.rig.set_mode("pose", "CharacterRig")
rt.rig.create_pose_clip("CharacterRig", "PoseTest", fps=24)
rt.rig.set_pose_frame(24)
state = rt.rig.get_pose_state("CharacterRig")
bone = next(b["name"] for b in rt.rig.list_bones("CharacterRig") if b["weighted"])
local = list(state["local_transforms"][bone])
local[3] += 0.05
rt.rig.set_pose_auto_key(True)
rt.rig.preview_pose_locals("CharacterRig", {bone: local}, state["rig_revision"])
rt.rig.apply_pose_preview("CharacterRig")
```

Extended existing regression fixtures: rig_pose_preview_test.cpp (link new
RigPoseAuthoringMath.cpp), rig_view_contract.py run_pose / IPC --pose, and
rt_test_rig_roundtrip.py run_pose for a NEW .rtp output. These tests are not
executed by Codex. Source-only IPC audit passes: 503 methods, 34 prefixes,
485 documented; generated descriptors and security classification agree.

This initial evaluation stack is not yet a multi-clip blending editor. Pose
mirror, weighted rest/rebind edits, IK/contact, footprints, spline walk/run and
flight remain the next roadmap milestones. Proceed to 7B/9A after user accepts
this bound deformation/key milestone, prioritizing any binding gaps revealed.

Final Pose non-build checks: focused module sizes/includes/delimiters, project
and filter XML/registrations, existing Python regression AST, descriptor JSON
and generated descriptor freshness passed. No build or live test was run.


### Edit/Pose gizmo hover and idle refresh correction (2026-09-13)

User compiled/tested the Pose milestone and confirmed it works, then reported
jitter on mouse hover/click without dragging in Edit/Pose and intermittent drag
startup. Source inspection found ImGuizmo's hover path requests next-frame
WantCaptureMouse, while RigViewportUI treated that same flag as an external UI
capture and skipped Manipulate. This alternated gizmo drawing and blocked clicks.

RigViewportUI now uses real hovered ImGui windows to protect authoring panels;
ImGuizmo's NoInputs overlay and the dock central passthrough do not block the
viewport gizmo. An inherited Scale tool falls back to Translate on Edit/Pose,
so entering these modes cannot silently leave a rigid-only gizmo unavailable.

Pose evaluation now uses transient dirty/frame acknowledgement separate from
the gesture serial. Enter, pose/key/clip mutation, cancel and undo/redo invalidate
evaluation; completed skin evaluation acknowledges it. Frame changes request a
new evaluation. An unchanged preview during a stationary drag is a no-op and
does not reset accumulation. Active-rig clips no longer force idle Pose evaluation;
other characters retain their existing animation evaluation behavior. Shared
UI/Python/IPC pose operations all invalidate via the same API wake path.

Existing rig_pose_preview_test.cpp covers dirty acknowledgement, frame changes
and evaluation invalidation preserving the gesture serial. Its authored-key
fixture now declares its own poseError before use. No build/live tests by Codex;
user rebuild and hover/click/drag verification remain pending for this fix.

User checks after rebuild:
1. Edit and Pose: hover translation arrows/rotation rings for several seconds,
   then click without dragging, then drag. Gizmo must remain visible and responsive.
2. Drag selected parent/child, release, Escape, undo/redo; no dropped gesture.
3. Move mouse into Rig/Alignment panels: viewport gizmo must not steal panel input.
4. Pose idle, stationary mouse-held drag, frame scrub/playback, numeric preview,
   Auto Key and clip selection: redraw only when required; deformation stays current.
5. Enter Edit/Pose after selecting Scale in Scene: Translate is available immediately.


### Gizmo correction user acceptance (2026-09-13)

User rebuilt and tested the correction, confirming that Edit/Pose hover/click
jitter and intermittent manipulation failure are resolved. This supersedes the
pending user verification status above for these reported issues. No separate
execution of every optional regression or native round-trip test is claimed.

### Phase 7A rest-relative Pose mirror (2026-09-14)

Source-delivered; user build/runtime checks pending. Completes the open paired
Pose mirror before Phase 7B IK. The existing Rig dock exposes Mirror selected pose,
Apply pose preview and Cancel. Select one side of each anatomy pair; selecting
both sides fails explicitly. No pair-name guessing or new hierarchy is used.

RigMirror::mirrorPose copies local rest-relative motion through each joint's
rest orientation basis and an actor-local axis reflection. Target rest origins,
orientations and lengths remain its own, including asymmetric manual fitting.
Selected parent/child local motions copy once; source and unpaired locals remain
unchanged. Rest, offsets, weights and original flat geometry are not mutated.
The common preview evaluator applies target joint rules and reports limit_hits.

UI, embedded Python and IPC call mirrorRigPose. `rig.mirror_pose(character,
bones, rig_revision, direction="selected", axis="x")` stages a transient preview.
Call apply_pose_preview to commit one history step; Auto Key writes only changed
target channels. Cancel, frame changes and mode exit discard preview. An existing
preview, stale revision, unpaired/unknown/duplicate source, ambiguous pair or
invalid axis/direction fails without mutation. Directional copies accept empty
bones for all pairs. Axis is actor-local x/y/z; plane offsets cancel for relative
displacements and are not a parameter. Keyed results use native authored clips.

Example for an already bound rig in Pose with an edited source joint:

```python
state = rt.rig.get_pose_state("CharacterRig")
pair = rt.rig.get_anatomy("CharacterRig")["symmetry"][0]
rt.rig.mirror_pose("CharacterRig", [pair["left"]], state["rig_revision"])
rt.rig.apply_pose_preview("CharacterRig")  # Auto Key follows current session setting
```

Existing regression files extended, no new per-step test files: pure C++ pose
mirror checks cover asymmetric rest, differing joint bases, parent/child motion,
round-trip and invalid pair/axis cases. rig_view_contract.run_pose_mirror and
test_rig_view_ipc.py --pose-mirror cover preview/cancel, stale/ambiguous rejection,
Auto Key, undo/redo, channel sampling and unchanged rest/flat weights. These live
fixtures require a disposable owned bound rig with an unrestricted anatomy pair.
Existing rt_test_rig_roundtrip.run_pose can verify keyed mirrored native reopen.

User build/runtime checklist:
1. Build normally. Bound humanoid -> Pose -> create clip -> frame 24 -> Auto Key.
2. Rotate one arm and forearm; select those left joints -> Mirror selected pose.
   Right motion mirrors; its rest lengths stay unchanged. Apply -> undo -> redo.
3. Mirror -> Cancel; source/target committed pose returns exactly. Select both
   sides of one pair; named ambiguity error and no pose mutation are expected.
4. Scrub frames 0/12/24, check interpolation and CPU/Vulkan/OptiX deformation,
   overlay and picking. With target joint rules enabled inspect limit_hits.
5. Save to a NEW .rtp and reopen; select the clip/frame and verify mirrored keys.
   Python: rig_view_contract.run_pose_mirror(character). IPC: python
   scripts/test/test_rig_view_ipc.py --character CharacterRig --pose-mirror.

No build, C++ regression execution or application launch by Codex. Next bounded
authoring milestone remains Phase 7B two-bone limb IK/pole target and contact
foundation, sharing this Pose preview/commit/key service; Phase 9A follows.

### Phase 7B two-bone control and position-contact foundation (2026-09-14)

Source-delivered; user build/runtime acceptance pending. No new test scripts or
regression code are written for this batch, at the user's request. Build and
application launch remain user-owned; Codex performs source/descriptor/XML checks.

- RigIK is the common analytical two-bone solver. Controls contain stable full
  root/mid/tip keys and a name, separate from deform joints. Consecutive rigid
  nonzero-length chains are required; overlapping driven controls are rejected.
  Anatomy v3 stores definitions, retaining v1/v2 native compatibility. Rename and
  deletion-reference guards include controls. Explicit custom definitions are
  supported; supported _arm/_leg anatomy chains can generate the initial set.
- Pose evaluation is canonical rest -> authored clip -> FK locals -> one IK blend
  -> joint rules -> skin. Handle previews store FK input and runtime controls,
  never feed an already blended result back into the next blend. Independent
  limbs solve parent-first. Root/mid rotations change; local translations,
  rest/bind offsets, segment lengths, flat weights and P_orig remain unchanged.
  Existing flat TriangleMesh/DNA skinning handles CPU picking and GPU rendering.
- World-space target and pole positions are converted through actor placement.
  Overlay, IK evaluator and handle queries share rigScenePlacement, including
  the live bound flat mesh transform rather than a stale saved actor placement.
  Placement changes request IK reevaluation once; unchanged placement stays idle.
  Law-of-cosines reach is clamped for unreachable goals. Collinear/degenerate
  poles fall back to the current bend and then a deterministic perpendicular.
  Nonfinite/unstable targets and invalid placement fail before publishing preview.
  get_controls reports achieved world tip and target_error_world after limits;
  get_pose_state retains joint-rule limit_hits. A residual does not imply success
  at an unreachable target or a guaranteed contact after joint limits.
- Enabling IK matches current target/pole. Returning blend to zero bakes achieved
  limb rotations into FK input, preserving the current pose. Direct FK edits of
  IK-driven limb joints are guarded, while pelvis/root and other un-driven bones
  stay editable so planted contacts can oppose body motion. Pose mirror currently
  requires switching active limbs to FK first.
- Position contacts capture the achieved tip, enable full IK and hold the world
  target across frame scrub. Regular unkeyed targets reset on frame change;
  contacts survive while Pose remains active. Mode exit or clip selection clears
  runtime controls. This is position-only, not foot orientation/ground inference,
  timed contact intervals, animated target tracks or an IK export format.
- The existing Rig/Pose dock adds limb selection, IK enable/blend, Target/Pole
  viewport handles, world numeric positions and Pin position in world. IK viewport
  handles use Translate, protected panel hover and stale-gesture checks. Release
  commits one history step; Escape/Cancel discard preview. Auto Key writes changed
  solved bone channels through the existing clip service. Explicit Key selected /
  Key all also consume the solved pose. Native persistence stores definitions and
  authored bone keys; live IK targets, contact pins and handle selection are transient.

Common UI/Python/IPC operations: get_controls, create_controls, select_control,
set_ik_target, set_ik_fk and set_ik_contact. Runtime setters stage preview; use
apply_pose_preview/cancel_pose_preview. create_controls updates persistent
definitions/revision in one history step; re-query revision after setup. Null or
omitted definitions derive supported anatomy limbs; an explicit array replaces
them, [] clears them. Definition changes, stale revisions, render locks and
unknown control names have shared named validation errors. Control selection is
transient and is exposed to automation too.

Focused modules: Animation/RigIK, Api/RtApiRigIK, RtIpcRigIK, RtPythonRigIK,
UI/RigIKUI and RigIKViewportUI. Pose evaluator/history remain in the existing
focused modules; no new implementation enters files over 2000 lines. Project and
filter registrations are delivered with source. New test scripts are deferred
until the core authoring/control/movement structure is established.

User build/runtime checklist:
1. Build normally. Owned bound humanoid -> Pose -> Create limb controls -> select
   an arm or leg. Enable IK: current endpoint must not snap. Drag Target/Pole in
   viewport; Apply/release, Escape, undo/redo and numeric input must agree.
2. Move target out of reach: no stretch or NaN; residual remains visible. Test a
   straight limb and collinear pole. Blend 0/0.5/1; disable IK and check FK pose
   preservation. With joint rules enabled, inspect limit_hits and target residual.
3. Pin both feet, select Bone FK gizmo, move pelvis/root and scrub frames: feet
   hold position within reach/limits. Release contact, then disable IK to return
   to FK. Tip orientation is currently free to follow the limb.
4. Create/select authored clip and enable Auto Key. Move target, release, change
   frames and inspect interpolation. Save to a NEW .rtp, reopen: control definitions
   and solved bone keys survive; live targets/contact pins intentionally reset.
5. Compare CPU/Vulkan/OptiX deformation, joint overlay and picking with camera fixed.

Next: tip orientation and timed contact/control channels plus bake, then Phase 9A
footprints. Spline walk/run and jump recipes must use the same evaluator and
bone-key authoring service rather than a second motion/skinning pipeline.

### Fast motion authoring direction: recipes plus optional 2D views (2026-09-14)

User asks whether a 2D editor would accelerate posing or whether footprints and
spline recipes can generate walk/run/jump/lie-down variations. Product direction:
build rig-aware motion recipes over controls/contacts/keys first; add compact 2D
views as editing surfaces for those canonical operations, not a separate rig.

- Top-down 2D edits footprint spacing, heading, spline/path and left/right contact
  sequence. Side 2D edits root height, jump arc and timing/contact transitions.
  These fit the existing bottom editor/right contextual dock; 3D viewport remains
  the deformation/pose inspection and final correction surface.
- Walk/run variants derive from speed, stride, cadence, contact duration, body
  bounce/lean, arm swing and deterministic variation seed. Spline arc length sets
  travel, gait phase sets contacts; root motion must apply once. Validated limb
  roles/chain lengths are prerequisites, not automatic anatomical understanding.
- Jump requires takeoff/root trajectory, flight pose and landing contacts; lying
  down also requires authored pelvis/spine/body pose transitions and support
  contacts. Footprints alone cannot supply these poses. Base pose/clip recipes
  plus control-aware transitions provide variation, with manual 3D correction.

This section is design only. No 2D motion editor, gait/jump/lie-down generator,
timed contact recipe, spline-follow bake or variation system is implemented here.

### Hand/foot IK user acceptance and selected-joint limit editing (2026-09-14)

User compiled the preceding package and confirmed working hand/foot IK. This
acceptance covers the reported limb behavior/build, not an unreported exhaustive
IK/FK/contact/undo/native regression run. User requests the proposed anatomical
IK progression and viewport joint visualization to be implemented incrementally.

New source delivery: RigJointLimits captures the selected joint's evaluated world
transform and its neutral limit frame from the current parent global plus canonical
NodeHierarchy local rest. The origin follows the actual joint. Axes and diagrams
never use inverse-offset skin matrices as joint coordinates. Angular measurements
follow the same rest-relative swing/twist convention as JointRule. Common queries
return frame matrices, current angles, enabled/type/range and active IK/FK state;
missing hierarchy/parent or unstable transforms fail explicitly.

The active joint alone shows current XYZ axes, IK/FK and rule status. Hinge/ball
rules show signed twist arcs, neutral/current-angle spokes and boundaries. Ball
rules also show the swing cone rim/generators. Disabled proposed rules use the original dim gray boundary without label backing;
an enforced boundary or observed outside-limit state has an explicit label/color.
This is a motion-limit diagram, not a weight influence envelope.

Joint Motion adds Show joint axes / limits and Edit limits in viewport. The
edit mode pauses playback and gives min/max/swing handles priority over FK/IK/actor
gizmos. Blue minimum and orange maximum handles use distinct radii to avoid
-180/+180 overlap; ball swing has a green handle. Ray/plane angle dragging uses
the captured rest-relative frame, with a horizontal screen fallback when edge-on.
Bounds preview visually without changing the canonical rule on every pointer
frame. Release calls the common set_joint_limits/profile service once, giving one
undo step. Escape, selection/frame/revision/pose/frame-transform changes and mode
exit discard the draft. Floating UI windows retain mouse ownership.

set_joint_limits changes an existing hinge/ball rule's minimum/maximum/swing,
preserving axis, enabled and position lock. Ownership, finite ranges, neutral
in-range contract, stale revision, render lock, pose preview and no-change errors
are shared with the existing profile service. Limits persist in native anatomy;
undo/redo invalidates the evaluator. Rest/bind offsets, topology, original flat
geometry and weights remain unchanged. Axis direction and rule activation remain
editable through the existing Joint Motion inspector/shared profile API.

UI, Python and IPC surfaces delivered together: get_joint_limit_view,
set_joint_limits, get_joint_limit_overlay and set_joint_limit_overlay. Display/edit
flags are transient active-joint session settings, separate from enforcing a rule.
New focused Animation/RigJointLimits, Api/RtApiRigJointLimits and
UI/RigJointLimitsUI modules are registered in project/filter XML. No implementation
is added to a file over 2000 lines. No new test script/regression code, build or
application launch by Codex; source/descriptor/XML checks only.

User build/runtime checks:
1. Build normally. Select a joint -> Joint Motion -> add/review suggestions or set
   hinge/ball. Show axes/limits: diagrams stay at the actual joint with camera,
   actor placement, Rest/Animated/Pose and limb IK changes.
2. Enable viewport editing. Drag blue min/orange max/green swing, including an
   edge-on camera. Release -> undo/redo; Escape and selection changes cancel.
   Disabled suggestions remain disabled after changing their bounds.
3. Enable the rule and pose/IK the limb: current angle/limit label and residual
   agree with the evaluator. Imported rigs remain inspectable/display-only.
4. Save to a NEW .rtp and reopen: canonical limits survive; overlay edit mode
   resets. Joint Motion numeric/profile edits and Python/IPC updates must agree.

### Remaining anatomical control and envelope sequence (2026-09-14)

IK/FK will expand by anatomical capability, not by assigning two-bone IK to every
bone. Detailed selected-joint views remain contextual to preserve viewport clarity.

| Order | Structure | Current status / next work |
|---|---|---|
| 1 | Joint axes/limits and viewport bound authoring | User confirmed overlay/limit editing works; contrast experiment reverted to original gray/plain text by request; viewport axis-direction handles remain a follow-up |
| 2 | Limb end orientation and timed contact/control channels | World orientation, timed IK keys/contact intervals and new-clip bone bake source-delivered; runtime acceptance pending; richer track editor and solver expansion remain open |
| 3 | Anatomical chain solvers | 4..64-bone FABRIK, two-point cubic spline shape and role-based one-bone aim controls source-delivered; runtime acceptance pending |
| 4 | Fingers and grouped FK | FK curl/spread controls and authored pose presets remain planned |
| 5 | Influence envelopes | Category capsule reweight preview/apply plus pose-following viewport boundaries are source-delivered through common UI/API/Python/IPC; native settings, revision checks and undo cover weight application | User runtime acceptance, per-bone radii and viewport radius handles remain |
| 6 | Fast movement authoring | Humanoid in-place Walk Recipe v1 runtime-confirmed; body-motion refinement awaits rebuilt runtime acceptance. Footprints, root-motion spline walk/run, jump/body transitions and compact top/side 2D views remain planned |

Envelope editing will describe weight-production influence volumes, independently
of joint motion limits. Changing a capsule will not silently overwrite skin weights:
an explicit preview/apply/recompute operation must write the flat TriangleMesh/DNA
rows through the strongest-four normalization and common geometry transaction.
Weighted editing/rebind still needs its own safety/undo/CPU-GPU invalidation gates.
Automatic anatomy or deformation quality is not established by drawing envelopes.

### Overlay acceptance and limb tip orientation (2026-09-14)

User confirmed the joint overlay additions work. Passive gray boundaries were
hard to read against the viewport. Disabled rules now use opaque pale blue;
limit segments have dark under-strokes and status/angle labels have dark backing.

Hand/foot IK now optionally holds a world-space tip orientation. Select Orientation
handle in Limb IK / Contacts to rotate the hand/foot; Hold tip orientation enables
or releases the constraint. World quaternion (wxyz) is also exposed numerically.
The shared set_ik_orientation(character, control, orientation_world, enabled,
rig_revision) operation is delivered in Python and IPC. It accepts only finite unit
quaternions (norm-squared tolerance 0.001), and shares Pose ownership, revision,
render-lock, preview/apply/cancel, history and Auto Key semantics with IK targets.
get_controls reports orientation_enabled, orientation_world, achieved
tip_orientation_world and angular residual. UI activation matches the achieved
orientation to avoid a snap.

Position solving and local IK/FK blending run first. The tip rotation then blends
once toward the world target using the same blend, preserving tip translation and
chain lengths. Actor rotation is extracted through TRS decomposition so uniform
placement scale does not contaminate orientation. Joint rules run afterwards and
may leave an angular residual. Contact capture matches achieved orientation;
position contacts retain the optional orientation across scrubbing. Live targets
remain transient; solved bone keys use the existing native clip persistence.
Timed orientation/contact channels and batch baking are still pending.

No new test scripts, build or app launch. User runtime checklist:
1. Build; compare inactive limits against bright and dark viewport backgrounds.
2. Enable limb IK, select Orientation handle, rotate hand/foot; target position
   and bone lengths stay fixed. Apply/release, undo/redo and Escape must agree.
3. Try blend 0/0.5/1, enabled joint limits and a rotated/uniformly scaled actor.
   Compare the angular residual; position contacts retain orientation on scrub.
4. Auto Key a rotated tip, save a NEW .rtp and reopen; solved tip keys survive.
   Python/IPC set_ik_orientation uses [w,x,y,z]; invalid/non-unit values fail
   without changing pose. Timed tracks and full-body solvers remain future work.

### Timed IK channels, contact intervals and bone bake (2026-09-14)

Source delivered; user build/runtime acceptance pending. AnimationData now owns
versioned per-control IK channels. Keys contain world target/pole, unit quaternion
[w,x,y,z], orientation activation, IK activation and blend. Times are seconds,
created from frame/Pose fps independently of clip ticks per second. Keys replace
at the same time; before the first key the limb uses FK; after the last it holds.
Position/pole/blend interpolate linearly and orientation uses slerp. Positive blend
ramps IK between enabled endpoints, including FK-zero to IK-one transitions;
orientation activation steps at its key. Evaluation follows the existing clip loop.

Contact intervals capture the CURRENT achieved tip/pole and optional orientation
for [start,end), use full IK, override saved keys while active, and release to
keys/FK at end. Exact interval replacement recaptures its pose; overlaps reject
without publishing, adjacent intervals are valid. Live controls remain a separate
Pose override layer; editing a timed contact does not accidentally turn every
saved interval into a permanent scrub pin. Explicit live contacts still override
saved channels until released. Keying/capturing/clearing one limb clears that
limb's live override. Native loading validates schema, limits, control references
and times against clip duration; old projects without channels load unchanged.

The shared pose evaluator now samples authored bone channels -> saved IK -> live
Pose overrides -> one IK blend -> joint rules -> canonical flat CPU/GPU skin.
Saved IK also evaluates outside Pose in authored playback. Existing FK guards and
selected-joint inspection use effective controls. Control-definition replacement
rejects dangling saved channel references; clear the relevant channels first.

Auto Key on IK edits now stores IK control keys rather than solved bones underneath
an active IK layer. Undriven FK edits still write FK bone keys. Without timed IK,
explicit solved-pose bone insertion retains its existing behavior; with timed IK,
requesting a driven bone fails rig_ik_use_control_keys_or_bake. Auto Key captures
handles, not a contact lifespan; use an explicit interval for a timed plant.

Bake samples the SAVED source clip/IK plus joint rules, writes all bones each frame
in an inclusive range, and creates/selects a NEW named bone clip without IK
channels. Source clip remains editable and unchanged. Existing bone channels
outside the bake range are copied; IK outside that range is not baked. Transient
live/FK overrides are excluded. Actor placement is fixed at the current scene
placement for sampling; animated actor transforms and automatic contact transition
matching remain follow-ups. Operation caps: 1001 frames, 100000 bone samples,
frame/tick range 0..1000000; channels cap at 256 controls and 10000 keys+intervals.

UI: Limb IK / Contacts -> IK clip channels. Key IK control here, Capture contact
interval, Clear this limb's IK channels, and Bake all IK to new bone clip use the
same canonical services as Python/IPC: get_ik_channels, insert_ik_key,
set_ik_contact_interval, clear_ik_channels, bake_ik_channels. Mutations require
owned bound Pose, selected editable clip, matching revision, no render job and
no pending preview. Each publish is one undo step and marks the project modified;
errors publish nothing. Focused RigIKChannels/RigIKTimeline/RigIKTimelineUI modules
are project/filter registered; the large ProjectManager receives serialization
calls only. No new test script, build or application launch by Codex.

User runtime checklist:
1. Create/select a Pose clip; at frame 0 set IK blend 0 and key the control, then
   at frame 20 set target/pole/orientation and full blend and key again. Scrub
   between keys; verify interpolation, bone lengths, and joint-limit residuals.
2. Capture a contact at the current pose for frames [5,15). Scrub 4/5/14/15;
   confirm pin activation and release. Adjacent interval succeeds; overlap fails.
   Change orientation/root FK, release live pins, then undo/redo channel edits.
3. Auto Key an IK drag: an IK key appears, without solved IK bone keys below it.
   Auto Key an undriven pelvis/root edit: its FK keys still appear.
4. Bake frames 0..20 to a new name; compare original saved-channel playback with
   baked playback on these frames. Undo restores source selection and removes the
   baked clip; redo restores it. Duplicate name/out-of-range bake changes nothing.
5. Save a NEW .rtp, reopen and explicitly select each source/baked clip; keys,
   intervals and baked bones survive. Python/IPC operations must match the UI.

Next anatomical structure: spine/neck/tail multi-joint or spline IK and contextual
head/wing aim. Timeline drag/key deletion, contact ramps and animated placement
support can extend this service later; fingers, envelopes and fast gait recipes
remain in the subsequent roadmap sequence.

### Overlay rollback and multi-joint chain IK (2026-09-14)

User rejected the contrast experiment: the pale blue and dark text backing made
the overlay worse. Restored original inactive gray (155,155,165,140), plain text
without backing rectangles, and original single-stroke lines. Enabled/outside
colors and existing limit-edit behavior remain unchanged.

Multi-joint IK foundation source-delivered; user runtime checks pending.
Animation/RigChainIK implements bounded FABRIK for 4..64 consecutive rigid bones.
The chain root stays fixed and local translations/segment lengths are preserved.
Unreachable targets straighten to maximum reach; reachable endpoints iterate up
to 32 passes with a relative endpoint tolerance. The pole seeds a bend while
solving a changed endpoint; it is not a direct curvature handle at a matched tip.
The existing curve is retained at activation when the tip already matches.
Spline curve shaping, twist distribution and per-iteration limit projection are
future work. Shared joint limits apply after IK and may leave target residual.

IKControl optionally carries the complete chain alongside root/mid/tip. Legacy
three-bone controls keep the analytic solver and four-field serialization.
Chain controls use persistent anatomy version 4; versions 1/2/3 still load.
Definition validation, overlap detection, driven-bone guards, rename/reference
checks and IK->FK matching cover EVERY chain bone. get_controls reports solver,
bones, full actor chain length and the usual position/orientation residuals.

UI: IK Controls / Contacts -> Add chain IK -> choose a canonical anatomy chain,
then select its IK control. The common create_chain_control(character, chain,
rig_revision) service appends rather than replaces existing limb controls and
uses create_controls validation/history/revision/ownership logic. Python and IPC
are registered and described together. Explicit create_controls rows can supply
chain:[root,...,tip], with mid equal to the second bone. Duplicate names, shared
driven bones, disconnected/zero-length/scaled chains or stale revisions reject.

Targets, world orientation, blend, position contacts, timed IK keys/contact
intervals, Auto Key, undo/apply/cancel, native persistence and bone bake reuse the
existing common IK/Pose evaluator. No UI-only solver, new test scripts, builds or
app launch. Focused RigChainIK.h/.cpp are project/filter registered. C++ modules
touched in this step were formatted with 4-space indentation and a 100-column
line-length target, respecting the user's readability preference.

User runtime checklist:
1. Build; verify inactive overlay is gray again and text has no backing rectangle.
2. Use an existing anatomy spine/neck/tail chain with 4..64 bones. Add chain IK,
   re-query revision, select and enable it: matched endpoint should not jump.
3. Drag its target in several directions and outside reach. Check fixed anchor,
   preserved bone lengths, residual, and existing limb IK under the moving body.
   Test optional tip orientation and blend 0/0.5/1; returning to FK preserves all
   chain locals. Enabled joint rules may limit the achieved endpoint.
4. Apply/undo/redo and Escape; key controls/contact intervals, scrub and bake to a
   new bone clip. Save a NEW .rtp and reopen: full chain definitions and channels
   survive; source versus baked playback agree in the sampled range.
5. Python/IPC create_chain_control must match UI behavior. Overlap, duplicate
   name, disconnected chain, zero segment and stale revision change nothing.

Next: spline curve/shape controls for these chains and contextual head/wing aim;
grouped finger FK, influence envelopes and fast gait authoring remain subsequent.

### IK-to-bone selection handoff correction (2026-09-14)

User built and tried the preceding additions; reported that after manipulating
an IK control, other bones could not be manipulated until leaving/re-entering
Scene/Pose. Root cause in the selection flow: successful bone selection did not
release pose.control, so drawRigIKGizmo kept owning the viewport instead of
letting the selected bone's FK gizmo draw. Overlay picking also tested global
WantCaptureMouse, which ImGuizmo itself sets for hover/drag.

Fixed in the shared RigSelection service: successful bone selection/clear releases
only the selected IK handle, resets its handle kind, invalidates the interaction
serial and discards an uncommitted IK preview. Committed IK targets, live contacts,
saved channels and FK pose remain intact. Invalid selection leaves all state
unchanged. Hierarchy, viewport, Python and IPC selection use this same core path.
Overlay picking now respects actual floating UI windows/popups and current gizmo
hover/drag ownership rather than the global capture flag. No mode toggle needed
for undriven bones. Bones belonging to an active IK chain remain solver-driven;
return that control's blend to FK before directly rotating its driven bones.

Source/static correction delivered; runtime acceptance pending. No build, app
launch or new test scripts. User check: manipulate a hand/foot/long-chain target,
release the mouse, select an undriven pelvis/root/other joint from viewport and
hierarchy and manipulate it without changing mode. Re-select another IK control;
its target must work and prior contacts remain active. Selection during an IK
preview discards only the draft; invalid script/IPC selection discards nothing.
Check menus/windows still block viewport clicks and active gizmo drags do not
select bones behind their handles.

Current position: hand/foot IK, orientation, timed channels/contact intervals,
bone bake and multi-joint FABRIK core delivered. Selection handoff corrected before
expanding interaction complexity. Next feature remains spline curve/shape controls
for spine/neck/tail, followed by contextual head/wing aim, grouped fingers,
influence envelopes and fast gait recipes. User build/runtime checks establish
specific accepted behavior; full numerical/deformation coverage remains deferred.

### Cubic spline shape guide for multi-joint IK (2026-09-14)

Source delivered; user runtime acceptance pending. Long chain controls now accept
an optional cubic Bezier guide: live chain root -> two world-space interior points
-> existing world target. IK Controls / Contacts -> selected chain -> Spline shape
exposes Use spline shape, Shape point 1/2 viewport selection and world numeric
inputs. The selected guide and S1/S2 markers are drawn without label backing.
Existing passive joint-limit gray/plain text remains unchanged.

Canonical RigSplineIK math validates points, samples the curve at 257 stations,
converts through the actual flat-mesh actor placement and initializes chain joints
by normalized arc-length fractions derived from bone lengths. RigChainIK then
runs bounded FABRIK with a fixed root and unchanged local translations, followed
by the existing single IK/FK blend and common joint rules. A straight compressed
guide receives a deterministic bend seed. The curve guides shape; it does not
stretch the bones or force exact curve adherence. Unreachable endpoints straighten
to reach. Joint rules and iteration bounds may leave a residual; get_controls
reports the achieved tip error and spline_fabrik solver kind. Collapsed curves
reject when solved. Default points approximate current first/last bone tangents;
activation may refit a curved pose and is not guaranteed to reproduce arbitrary
FK curvature exactly. Curve fitting, more points and twist distribution remain
follow-ups.

Canonical set_ik_spline(character, control, points_world, enabled, rig_revision)
is delivered in C++, Python and IPC. Exactly two finite [x,y,z] world points are
required; [] is allowed only when disabling/clearing the spline. Limb controls
reject rig_ik_spline_requires_chain. Shared Pose ownership/revision/render-lock,
preview/apply/cancel, history and Auto Key semantics apply. Moving a viewport
shape point enables the guide. Selecting another bone releases shape gizmo
ownership through the preceding shared selection handoff without dropping saved
IK or contacts. No mode exit/re-entry is required for undriven bones.

IKPose holds splineEnabled/splineWorld. Runtime equality and validation include
both. get_controls exposes spline_enabled, spline_world and sampled
spline_guide_world. Legacy active chain keys without shape points obtain tangent
defaults for inspection/editing while preserving their actual target/pole/rotation.
IK key insertion and Auto Key record shape state. Points interpolate linearly
when both keys carry them; activation steps. Contacts capture/retain enabled
shape points and timed bake uses the same curve/chain solver. Native IK channels
use version 2 when shape points are stored, otherwise version 1. Version 1 files
still load with the spline disabled; malformed shapes and spline keys on limb
controls reject. Persistent chain definitions remain anatomy version 4.

Focused Animation/RigSplineIK and UI/RigSplineIKUI modules are project/filter
registered. Existing focused API/bindings and viewport modules have only relevant
integration; large source files have no new implementation. All C++ touched here
uses 4-space indentation and a 100-column target. No new test scripts, builds or
app launch by Codex; static source/XML/descriptor coverage checks only.

User build/runtime checklist:
1. Build, select a 4..64-bone chain IK, open Spline shape and enable it. Select
   S1/S2 and drag each in the viewport; review the curve and achieved skeleton.
   Root and bone lengths stay fixed; shape changes are one undo step per release.
2. Test Apply/Cancel, Escape, undo/redo, target movement, reach limits, blend
   0/0.5/1 and joint rules. Select an undriven bone after a shape drag and edit it
   without changing mode; reselect the chain and verify shape points survived.
3. Auto Key or key the IK control at two frames with different shape points.
   Scrub between them, create a contact interval and bake to a new bone clip;
   compare saved-channel playback with baked playback over sampled frames.
4. Save a NEW .rtp/reopen; shape keys/contacts survive. A pre-spline version 1
   chain clip still plays and obtains editable default shape handles. Test a
   rotated/uniformly scaled actor, malformed points, collapsed curve and stale
   revision through Python/IPC; rejected edits do not publish pose/history.

Next: contextual head/look/wing aim and grouped finger FK. More spline points,
shape-fitting and distributed twist are extensions of this guide, followed by
influence envelopes and gait/footprint/spline movement authoring.

### Spline handle visibility correction (2026-09-15)

User runtime report: spline controls appeared in the system UI and viewport and
could be selected, but the translation gizmo did not appear. This was an IK
viewport rendering ownership bug, not required user workflow. drawRigIKGizmo
returned before BeginFrame/Manipulate whenever any ImGui window was hovered, so
selecting S1/S2 from the dock hid the newly selected handle while the pointer
remained over that dock.

Corrected the separation between rendering and input ownership. Selected target,
pole, orientation and spline handles remain drawn while a dock/popup is hovered.
A UI-owned press is rejected before an IK drag can begin, with ImGuizmo state
reset and the press blocked until release. Moving the pointer into the viewport
then allows normal manipulation; active drags remain uninterrupted. Pose preview,
history, Auto Key, channels and solver data are unchanged. No mode change should
be needed. Source/static fix only; user runtime acceptance pending. No build,
application launch or new test script by Codex.

User check: select Shape point 1/2 with the mouse still over the IK panel and
confirm the viewport translation gizmo is already visible. Clicking/dragging the
panel must not move the shape point. Move into the viewport and drag the gizmo;
release applies one history step, Escape cancels, and selecting a bone hands off
to FK as documented. Also recheck target, pole and orientation handles because
they share the corrected renderer.

### Contextual role-based aim controls (2026-09-15)

Source delivered; user build/runtime acceptance pending. IK Controls / Contacts ->
Add aim control now creates a one-bone control from any explicit anatomy role.
This covers head/look, beak, wing-segment and custom directional controls without
applying two-bone IK across the whole skeleton. The stable control name is
`<role>.aim`. The local aim axis is derived from the first nonzero direct child,
with local +Y as the leaf fallback; a perpendicular roll-up axis is stored with
the definition. The Target handle controls direction and Roll up controls twist.

Focused Animation/RigAimIK preserves the bone's local translation and computes a
world target swing plus roll stabilization, then blends once in local quaternion
space. Existing joint rules run after the common IK evaluator and may leave an
angular residual. get_controls reports solver `aim`, the driven bone and
aim_error_degrees. Aim definitions are persistent anatomy version 5; earlier
versions 1..4 continue loading unchanged. Definitions reject unknown solver/role,
non-unit or non-orthogonal axes, duplicate names and driven-bone overlap.

Canonical create_aim_control(character, role, rig_revision) is exposed through
C++, Python and IPC and delegates publication to the existing create_controls
history/revision service. Existing set_ik_target, IK/FK blend, preview/apply/cancel,
Auto Key, explicit IK keys and bone bake use the same evaluator. Position contacts,
spline shaping and independent tip-orientation channels do not describe a
one-bone aim constraint and are rejected; the UI hides those controls for aim.
The timeline UI now labels generic controls correctly and its previously compressed
C++ layout was reformatted to the agreed 4-space/100-column style.

No project build, application launch or new test script was run by Codex. Static
source, XML registration and generated descriptor audits are the acceptance used
for this source delivery.

User runtime checklist:
1. Build, enter Pose, choose Add aim control -> head. Select `head.aim`; enabling
   at its matched target must not jump. Target and Roll up translation gizmos must
   remain visible after selection from the dock.
2. Move Target around the character and move Roll up around the look direction.
   Confirm the head points at the target, roll is stable, local position does not
   move, IK/FK blend 0/0.5/1 works, and enabled head limits report residual rather
   than changing the target.
3. Repeat with a wing role that has a direct child. Confirm its stored axis follows
   that child direction. For a leaf role, review the documented +Y fallback.
4. Apply/Cancel, Escape, undo/redo, key two target/roll states and scrub between
   them; bake to a bone clip. Save to a NEW .rtp/reopen and verify anatomy version
   5 plus aim keys survive. Python/IPC behavior must match the UI.
5. Verify duplicate role aim, overlap with another active definition, stale
   revision, target exactly at the bone origin, contact, spline and orientation
   requests reject without publishing history or pose changes.

Next: grouped finger FK curl/spread and pose presets, then canonical influence
envelopes. Fast footprint/contact and spline-guided gait authoring follows those
foundations; spline twist distribution remains a chain-solver extension.

### Animator control layer v1: semantic limb shapes and explicit FK/IK matching (2026-09-19)

Runtime build and first interaction accepted by the user. Pose mode now draws all
limb targets as constant-screen-size viewport controls instead of requiring the
IK combo first. The target shape comes from canonical anatomy role metadata:
hands use a compact box, feet use a foot plate, aim controls use a triangle and
unclassified/custom controls use a ring. Left/right colors, selected orange,
contact green, inactive-FK dimming, selected pole diamond and the active IK/FK
label make ownership visible without changing deform bones or skin data. Target
and pole shapes select the existing canonical control/handle and hand off to the
same ImGuizmo editing path. In FK, root/mid joint rings select the corresponding
bone and its existing rotation gizmo, so switching control modes does not create
a second FK evaluator. `get_controls` reports these semantic display hints and
world-space FK handles to UI, Python and IPC consumers. The selected two-bone
target also carries a compact `IK`/`FK` badge; clicking it invokes the same
pose-matched operation as the dock buttons and commits through the shared service.

Two explicit two-bone operations clarify the existing pose-preserving contract.
`match_ik_to_fk` captures the visible tip, bend plane and tip orientation, clears
contact and switches to full IK. `match_fk_to_ik` copies the achieved limb locals
into FK and disables IK/contact. Both stage the common pose preview; the dock
buttons apply it through the existing single history transaction. Python and IPC
expose the same services. Chain/spline and aim controls reject these bounded limb
operations with `rig_ik_match_requires_two_bone`; their future matching requires
solver-specific rules rather than pretending two-bone semantics are universal.

User runtime checklist:
1. Build, enter Pose and create limb controls. All four hand/foot targets should
   appear in the viewport before choosing the IK combo; click each target and its
   selected pole diamond.
2. Pose an arm in FK, press `Match IK <- FK`; the hand and elbow must not pop.
   Move the target/pole, then press `Match FK <- IK`; the visible pose must stay
   fixed and the normal bone FK gizmo must become editable again.
3. Repeat on both feet with actor translation/rotation and blend 0/.5/1. Verify
   selected/contact/FK colors and undo/redo. Python/IPC match calls must agree
   with the dock and reject a spine/aim control without changing the pose.

### Rig authoring UI ownership checkpoint (2026-09-19)

The existing right contextual dock remains the owner; no standalone permanent rig
studio or wide shelf was introduced. Pose now presents clip/mode and IK controls
before selected-bone FK, keys and secondary tools. Quick Motion is collapsed by
default and follows manual controls; deformation diagnostics and skin binding are
collapsed secondary sections in Pose. Rig setup keeps template creation, mesh
preparation, skeleton source/topology and alignment/deformation as explicit visual
stages while the large Alignment canvas remains a temporary focused window.

This is an ordering/ownership correction only. UI continues calling canonical
services, control/deform separation is unchanged, and Python/IPC contracts do not
gain duplicate operations. Before Control Layer v2, run the grouped transition,
undo/redo and new-project save/reopen matrix. Next feature order is grouped finger
FK and pose presets, then heel/ball/toe foot pivots and foot roll; gait refinement
follows those control foundations.

### Visual control language foundation: display contract v1/v2 (2026-09-19/21)

The visual layer now has a focused shared `Animation/RigControlDisplay` contract
instead of deriving hand/foot colors and shapes inside the API adapter. Solver
definitions remain authoritative for behavior; the display descriptor is a
deterministic presentation projection of the solver control plus canonical anatomy
roles. This deliberately avoids anatomy schema churn while the visual vocabulary is
still being established.

`rig.get_controls` now returns top-level `display_contract` version 2 and a nested
`display` object for every control. Version 2 replaces purely constant-screen
visuals with `hybrid_anatomical_clamped`: rows expose placement-scaled
`length_world`, while the contract retains `screen_constant_minimum` hit regions.
It also names Primary/Secondary/Deform levels, the built-in shape and color-role
vocabularies and the contact-marker convention. Each control describes semantic
kind, side/color role and target/pole/FK shape, level, scale and allowed-channel
hints. UI, Python and IPC consume the nested descriptor directly; the superseded
flat presentation fields were removed because this control layer has no
legacy-client compatibility target.

The viewport consumes the nested contract. Side color remains stable when a control
is pinned; contact is now a separate green inner marker, selection remains orange,
and built-in root and finger-arc shapes are available for the next control families.
The current slice does not yet add root/COG, grouped finger or foot-roll solver
controls, nor does it persist custom display overrides. Next: expose transient
Primary/Secondary/Deform visibility, then add body controls against this contract.

### Pose key ownership correction (2026-09-19)

Bone channels already supported atomic position/quaternion insertion, same-frame
replacement and Auto Key, but the normal Timeline `I` shortcut still created a
generic character marker and there was no exact-frame bone-key removal operation.
The control layer now owns this interaction while Pose is active: `I` inserts or
updates the selected IK control, otherwise all selected FK bones. Timeline insertion
at another frame first moves the Pose playhead to that frame. Timeline Delete/X on a
selected character key routes to the same selected control/bones instead of mutating
the unrelated object-transform track.

`remove_pose_keys` and `remove_ik_key` are shared undoable core/API operations exposed
to UI, Python and IPC. They remove only exact current-frame keys, preserve other
frames/contact intervals and fail `rig_edit_no_change` when nothing matches. Pose
state reports `keyed_bones_at_frame`; the dock shows how many selected bones are
keyed and offers explicit insert/update and removal buttons. The standalone rig pose
fixture covers exact removal, missing-key no-op atomicity and empty IK-channel cleanup.
This closes the keying foundation required before grouped finger controls.

### Authored clip runtime ownership correction (2026-09-19)

An authored rig clip could appear in the Anim Graph clip picker yet fail to play,
or continue playing an older key snapshot. The picker read the canonical
`SceneData::animationDataList`, while graph evaluation sampled the per-character
`AnimationController` registry (and could retain an earlier Ozz scaffold). Pose and
walk authoring replace immutable `AnimationData` snapshots for undo/redo, but those
runtime registries were not refreshed by the exchange commands.

`Animation/RigClipRuntimeSync` is now the shared bridge from canonical scene clips
to the selected character runtime. Pose-key and generated-motion exchanges call it
after every execute/undo swap, rebuilding the character-filtered controller and Ozz
set, clearing stale evaluated globals and preserving the scene-wide controller for
non-character consumers. Authored clips therefore become immediately playable in
Anim Graph without the manual **Sync Scene** action, and key edits/undo/redo expose
the same current clip snapshot to Timeline and graph evaluation.

The existing Graph Editor remains the intended curve UI; no second rig curve editor
will be introduced. Its current imported-animation projection is still one-way and
uses generic Timeline transform tracks, so it must not be treated as the owner of
rig keys. The next integration slice is a focused adapter that projects only the
active authored clip and commits graph edits through the same undoable rig-key API;
quaternion rotation and curve interpolation metadata must have one explicit,
serializable canonical contract before tangent editing is enabled for bone tracks.

### Skeleton-first Pose capability and reversible Unbind (2026-09-19)

Pose authoring no longer treats a weighted mesh as evidence that a skeleton is
editable. An authoring-owned `NodeHierarchy` plus its skeleton representation is
the authoring capability; a valid skin binding is a separate deformation-preview
capability. Meshless rigs can enter Pose, create/select clips, author FK/IK and key
`AnimationData`. Bound rigs additionally validate skin offsets and deform their flat
`TriangleMesh` consumers. Pose coverage now reports `skeleton_authoring` and
`deformation_preview` independently.

Initial bind no longer routes through the rest-edit guard that rejects clips.
Existing authored clips survive a later bind and are republished to the character
controller and Ozz runtime. This establishes the intended order-independent flow:
skeleton -> animation -> optional mesh binding.

`rig.unbind_mesh(character)` is the shared undoable inverse exposed through UI,
Python and IPC. It restores present mesh parts to canonical bind `P_orig/N_orig`,
clears skin influences and explicit rig membership, and preserves the authored
skeleton, anatomy, controls, placement and all clips. Missing registered parts are
removed from the registry without manufacturing geometry. Undo restores geometry,
weights, membership and runtime clip bindings as one heavy transaction.

### Canonical bone-curve adapter v1: bidirectional value/time editing (2026-09-19)

The existing Graph Editor remains the only curve UI. Its generic Timeline tracks
are now a projection, never a second owner. The adapter follows these rules:

1. `AnimationData` position/quaternion keys remain the canonical key values and
   times; the adapter projects only the active authored clip and selected bone.
2. Graph value/time edits publish one atomic rig-key command, then the projection is
   rebuilt from the committed clip. Timeline track mutation alone is never accepted
   as a rig edit. Translation component drags publish the complete XYZ value so a
   three-component key cannot be torn into mismatched times.
3. UI, Python `rt.rig.edit_pose_key` and IPC `rig.edit_pose_key` call the same service
   with identical validation, undo and runtime-refresh behavior. Missing source keys
   and occupied destination times reject without partially changing the clip. Bones
   driven by timed IK at either endpoint direct the author to control keys or baking
   instead of accepting an FK edit hidden by the runtime solver.
4. Future translation interpolation may use constant/linear/Bezier per scalar
   channel only after that metadata lives beside the canonical keys and is sampled
   and serialized. Quaternion rotation keeps the existing runtime interpolation;
   Euler display is an editing projection, not a second stored rotation. Free Euler
   value tangents stay disabled until a declared rotation representation can
   reproduce them exactly at runtime and export.
5. Clip/character/bone identity is explicit in every operation, avoiding collisions
   between multiple clips that animate the same prefixed bone name.

This first adapter delivery covers bidirectional position value/time operations,
quaternion-key time moves, selection refresh and undo/redo. Rotation is displayed as
Euler projection for inspection, but vertical Euler editing and rig tangent handles
remain disabled: they would otherwise create a visually editable path that the
runtime quaternion sampler does not reproduce. Interpolation/tangent publication
follows only with matching sampler and native serialization support.

Rig key dragging now previews directly in the disposable Graph Editor projection,
matching the established node-animation interaction instead of leaving the key at
its source until mouse release. Position and rotation families are split visually
when they originally share a frame, so dragging one never appears to move the other.
The canonical clip is still published only on release as one undoable command; a
collision or rejected edit discards the preview and rebuilds it from `AnimationData`.

### Selected-bone Timeline projection and authored hold sampling (2026-09-19)

The old `syncFromAnimationData` path imported every skeletal channel once into
generic Timeline tracks. New pose keys therefore did not appear, while every bone
permanently polluted the track list. Skeletal clips are now excluded from that
one-way import. `RigTimelineProjection` builds disposable tracks only for the active
authored clip and current bone selection, removes stale all-bone projections from
older sessions, and refreshes when the immutable clip snapshot or selection changes.
Pressing `I` remains routed to the canonical rig-key API; the projected diamond
appears on the next UI frame without becoming a second key owner.

Pose authoring sampling is now non-looping: before the first key it holds the first,
between keys it interpolates, and after the final key it holds the final value.
Previously `calculateAnimationTransform` always wrapped by clip duration; because a
new authored key extends duration by one tick, the final pose existed for one frame
and immediately blended/wrapped to the first key. General animation callers retain
looping by default, while Pose explicitly requests clamped sampling. Anim Graph loop
behavior remains the clip node's playback decision. The standalone pose fixture now
checks final-key hold behavior.

### Aim/chain creator visibility correction (2026-09-15)

User runtime report showed only Create limb controls and Bone FK gizmo in the IK
dock. The four entries in IK control were the limb definitions created by the
first button; no aim definition had been created. Root cause: Add chain IK and
Add aim control used ImGui BeginMenu outside a menu bar/menu, so this dock layout
did not expose them as usable selectors.

Both creators are now explicit labeled combos in the dock: Add chain IK -> Choose
anatomy chain and Add aim control -> Choose anatomy role. Existing definitions
remain visible but disabled in each creator, preventing duplicate publication.
Bone FK gizmo stays beside Create limb controls. Spline UI eligibility now checks
the reported `fabrik`/`spline_fabrik` solver kind as well as the 4-bone minimum;
two-bone limb and one-bone aim controls cannot display Spline shape accidentally.
No core solver, persistence or binding contract changed. Source/static correction
only; user build/runtime acceptance pending, with no Codex build or new test script.

### Human Walk Recipe v1: editable in-place clip (2026-09-15)

Baseline generation accepted at runtime; stronger body-motion/update workflow is
source-delivered with user rebuild acceptance pending. This is the first usable
Quick Motion slice and intentionally precedes grouped fingers/envelopes. The Pose
dock now exposes Quick Motion -> Human Walk (in place), with clip name, cadence,
cycle count, stride, foot lift, body bounce, arm swing and body motion. Motion distances are
fractions of measured character height, so the defaults scale with the rig rather
than requiring scene-unit knowledge. Check recipe reports frames, duration and
resolved scene-unit distances. Generate / Update publishes and selects one
ordinary authored bone clip in one undo step. Reusing its name replaces the same
editable rig clip; imported clips remain protected from overwrite.

The focused RigWalkRecipe core accepts only explicit humanoid anatomy and requires
pelvis, left/right ankle roles and the canonical left/right arm/leg chains. It
derives temporary two-bone controls without changing persistent control setup.
Forward comes from the ankle-to-toe rest direction with +Z fallback. Each output
frame moves the feet through deterministic stance/swing trajectories, lowers
T-rest hands beside the pelvis with stable elbow poles, offsets them with body
bounce and counter-swings them against the legs. Runtime inspection of the first
generated clip then added planted foot orientation with swing-phase toe pitch,
lateral pelvis weight transfer, pelvis tilt/yaw and distributed lower/upper-spine
counter rotation. The body-motion factor scales these additions. Arm targets now
follow a vertical arc and their elbow poles travel with the swing instead of
remaining fixed. Post-build IPC measurement showed that motion magnitude was
already substantial while chest and head still rotated as one block. The recipe
therefore counter-rotates a mapped head to stabilize gaze and adds small mapped
clavicle rotations so the shoulder girdle participates in arm swing. The existing
analytic IK and joint-limit evaluator produces the final hierarchy, then the
shared pose-key writer stores rigid local transforms.

The resulting v1 clip is loopable, in-place and actor-placement independent. It
contains normal bone keys rather than persistent world-space targets, so it can
play after the actor is moved and can be corrected through existing Pose/FK tools.
It does not yet generate root motion, terrain-aware footprints, timed contact
channels, path following or a top/side 2D editor. Those build on this recipe and
must keep root travel single-sourced when added.

Canonical preview_human_walk and create_human_walk_clip operations are exposed
through C++, Python and IPC. Validation bounds cadence to 20..300 steps/min,
cycles 1..8, fps 1..120, stride .05..0.80 character height, step height 0..0.25,
body bounce 0..0.15 and arm swing/body motion 0..1. Generation is limited to 1000
frames and 100000 frame-joint samples. A same-character rig-authored name replaces
that clip atomically; an imported name conflict, stale revision, pending preview,
missing humanoid roles/chains or invalid dimensions publishes nothing. Native
clip persistence, selection, undo/redo and renderer invalidation reuse authoring rules.

Live IPC inspection before implementation confirmed the open scene is a valid
target: character `Rig`, humanoid anatomy version 4, Pose mode at 24 fps, revision
15, selected `PoseClip`, canonical pelvis/toe/limb roles, four two-bone limb
controls and a six-bone spine FABRIK control. No IPC mutation was made.

User runtime checklist:
1. Build, stay in Pose and open Quick Motion. Keep defaults, press Check recipe;
   expect roughly 58 frames at the live 24 fps/100 cadence/two-cycle settings.
   The playback wrap must continue at the same motion step without a visible pop.
2. Generate `Walk_Loop`; it should become the selected editable clip. Play/scrub
   the full range and confirm alternating feet, opposite arm swing, body bounce,
   no NaN/stretch and a visually continuous loop boundary.
3. Change stride, foot lift, bounce, arm swing and body motion, then use the same
   `Walk_Loop` name. It must update one clip instead of adding a duplicate.
   Joint rules may reduce reach but bone lengths/local translations must remain
   valid. A conflicting imported clip name and invalid values must publish nothing.
4. Correct a generated frame using normal FK/Pose tools, undo/redo generation,
   save to a NEW .rtp and reopen. Move the actor and replay: the in-place motion
   should follow the actor because the generated clip contains local bone keys.
5. Compare UI with Python/IPC preview_human_walk/create_human_walk_clip using the
   same values and rig revision. Results and validation codes must agree.

Next fast-motion slice: explicit left/right footprint/contact events plus root
motion, then arc-length spline following and the compact top/side 2D editor. Run
and motion variations reuse the same recipe instead of adding a second evaluator.

Natural gait must remain a skeleton problem before muscle/weight polish. The next
walk refinement is a phase model with contact, down, passing and up landmarks,
heel strike, toe-off and a center-of-mass target over the support foot. Muscle,
volume preservation and improved skin weights can refine the silhouette later;
they cannot correct rigid head/shoulder timing or missing foot-contact mechanics.

### Animation key wrap correction found by Human Walk (2026-09-15)

Runtime accepted: the user confirmed the final-frame foot positions are corrected.
Read-only IPC sampling
of the generated `Walk_Loop` exposed a shared sampler defect at its exact last key:
the old position/rotation/scale search replaced the last key index with zero, then
evaluated the last tick as an extrapolation from key zero. At frame 57 this moved
the inspected foot more than one character unit away instead of returning key 57.
`AnimationData::calculateAnimationTransform` now retains the actual key at or
before the sample, wraps only its next index, computes elapsed time across the
duration boundary and clamps interpolation to 0..1. The same segment calculation
is shared by position, rotation and scale channels. This corrects all clips using
the common sampler; it does not change stored keys or clip timing.

### Bounded anatomical capsule weights baseline (2026-09-15)

The user observed that arm motion pulls torso vertices and that the automatic
weight diameter is too broad. Read-only IPC inspection confirmed a distribution
problem rather than a normalization failure: all 11,433 vertices were fully
weighted, normalized and limited to four influences, but LeftClavicle affected
2,507 vertices above .05 and overlapped Chest on 2,363; RightClavicle affected
3,134 and overlapped Chest on 3,020. Left/RightUpperArm also had significant
Chest, Spine and Pelvis overlap. The existing nearest-segment inverse-distance
baseline has no influence boundary, so a numerically valid row can still be
anatomically wrong.

The first envelope slice is now source-delivered for authored bound rigs. Each
parent-owned rest segment receives a height-relative capsule radius selected from
canonical anatomy: torso/head/neck/spine, ordinary limbs, or hand/foot/toe/paw.
Scores are zero beyond the capsule, use configurable falloff inside it, then pass
through the shared deterministic strongest-four normalization. A vertex outside
every capsule falls back to its nearest segment and is counted in the preview,
preserving the fully-weighted contract without hiding coverage gaps. The service
reads and replaces actual flat `TriangleMesh`/DNA `P_orig` and `skin_weights`;
Triangle facade geometry is not used.

`rig.preview_envelope_weights` computes vertex, segment, fallback and per-bone
influence counts without changing the scene. `rig.apply_envelope_weights`
recomputes from canonical inputs, requires the preview/current rig revision and
publishes all registered parts atomically as one heavy undo step. The common C++
service is used by Bind UI, Python and IPC. Apply invalidates CPU, raster, Vulkan
and OptiX animation geometry. Algorithm name and torso/limb/extremity radii plus
falloff persist in native projects and are returned by `rig.get_binding`; load
rejects unknown algorithms or nonfinite/out-of-range settings.

This baseline exposes category-level radii so the current leakage can be tested
before adding more controls. Per-bone stored radius/falloff, selected capsule
boundaries, viewport radius handles, surface visibility/interior constraints and
heat diffusion remain open. Imported skins outside the authored binding registry
are not rewritten.

User runtime checklist:
1. Rebuild, select the bound `Rig`, open Bind mesh to skeleton -> Automatic
   influence envelopes and preview the defaults. Record `fallback_vertices`.
2. Apply, enable selected-bone weight tint and compare both clavicles/upper arms
   with Chest and Spine. Torso vertices distant from the shoulder capsules should
   stop following the arms while the shoulder seam retains a smooth blend.
3. Play `Walk_Loop`, then adjust limb radius and falloff, preview and apply again.
   Favor the smallest limb radius that preserves the shoulder/elbow transition.
4. Verify one-step undo/redo, then save to a NEW `.rtp`, reopen and confirm
   `rig.get_binding` reports `anatomical_capsule_v1` and the same settings.

### Binding/weight panel visibility correction (2026-09-16)

The envelope controls were unreachable from Edit Bone while a character was in
Pose mode because RigEditing returned immediately after drawing pose authoring.
The same capability was also hidden behind the old pre-bind label and a second
closed header. Edit Bone now exposes a default-open `Skin binding and weights`
section for the active Pose character as well as Scene-mode owned-rig selection;
the nested `Automatic influence envelopes` section is default-open. Binding and
weight services, validation and scene mutation remain unchanged.

### Influence capsule viewport preview (2026-09-16)

The first envelope Preview returned only counts; it did not publish any drawable
view state, so the user correctly saw no viewport change. Edit Bone now enables a
separate transient capsule overlay after a successful weight preview. Capsules
follow the currently displayed skeleton pose while their radii remain based on
canonical rest height. Ordinary boundaries are faint cyan and the active bone is
amber, with low-alpha fill so the mesh stays readable. Slider changes refresh a
visible overlay immediately; `Show capsule overlay` controls it independently of
weight application.

The view operation is shared as `rig.get_envelope_overlay` and
`rig.set_envelope_overlay` in C++, Python and IPC. It validates the same authored
binding and radius ranges, mutates no skin rows, creates no history entry and is
not persisted. The actual Preview remains read-only through IPC; only Edit Bone
chooses to enable the display after Preview succeeds. Per-bone stored radii and
interactive viewport radius handles remain the next envelope authoring slice.

The first user build reported `LNK2001` from RigEnvelopeWeights to
`RigAuthoring::listBones`. The envelope core no longer depends on the higher-level
RigView listing service. It now derives displayed segment transforms directly
from canonical hierarchy plus current Pose evaluation or captured animation
globals, then applies the registered bound-mesh placement. This removes the
linker edge and keeps weight generation independent from inspector/view records.

### Focused 2D Envelope Editor and tapered profiles (2026-09-16)

To avoid expanding the Edit Bone dock into a long property shelf, the binding
section now keeps one compact `Open Envelope Editor` action. The separate floating
editor follows viewport bone selection and presents a dark 2D bone profile canvas.
Its two upper handles edit proximal/distal radius vertically and axial extension
horizontally; precise numeric controls and falloff remain below the canvas.

Per-bone overrides store two height-relative radii, two bone-length-relative
extensions and falloff. Apply evaluates the tapered capsule against every other
category/default or overridden capsule, regenerates all registered flat DNA skin
rows, retains strongest-four normalization and publishes geometry plus the profile
as one heavy undo step. The selected-bone weight tint is enabled after Apply, so
the color display always represents actual committed skin rows rather than a UI
approximation. Profiles persist in native projects and participate in undo/redo.

The shared operations are `rig.get_bone_envelope` and
`rig.apply_bone_envelope` in C++, Python and IPC. The editor is the first focused
2D envelope authoring slice. Direct viewport radius handles, symmetry-linked
editing and a noncommitting proposed-weight heatmap remain follow-ups; the current
viewport overlay immediately reflects committed tapered profiles.

### Data-driven scalar control rig and grouped finger FK (2026-09-21)

The first remaining control-layer gap after IK/FK matching is now implemented for
the generated `humanoid_detailed` template. Pose mode exposes relative Curl,
Spread and Thumb curl/opposition gestures directly around the authored viewport
anchor. A drag always starts from the committed pose, so live preview events do
not accumulate. Releasing the control commits through the existing pose preview
command, undo and Auto Key path, then returns the gesture value to zero.

The original template-specific evaluator has been replaced before further
controls were added. Anatomy v7 persists generic scalar control definitions:
stable ID/label/group, display anchor/shape/side, value range/default and one or
more explicit local-axis bone rotation drivers. The evaluator knows neither
humanoids nor fingers. `humanoid_detailed` merely authors six definitions for its
two hand groups; an owned custom rig can author the same capability with its own
bone keys and axes. Imported-rig authoring still follows the existing ownership
rules rather than bypassing them in the evaluator.

The shared operations are `rig.get_driven_controls` and
`rig.preview_control_values` in C++, Python and IPC. Sparse values are validated
against persisted definitions, evaluated atomically from the committed pose and
then committed through the existing pose preview, undo and Auto Key path. The old
template-specific `preview_finger_pose` API was removed rather than retained as a
compatibility alias. Canonical animated scalar-control curves and bake-to-bones
remain the next control-data slice.

### Viewport hand controls (2026-09-21)

Grouped scalar controls are now represented in the same hybrid viewport layer as
the existing IK controls. Every driven-control group is collected from
its authored group and anchor; no template ID is inspected. The anchor icon is
offset slightly from the bone and connected by a subtle line so it does not hide
an IK target at the same joint. Icons follow the displayed pose and use a hybrid
anatomical scale: the control's world-space chain length is projected at its
current depth, then clamped to a usable visual range. Thus controls shrink with
the rig in distant views and grow with it in close views without becoming
unusable. Hit regions retain a DPI-scaled minimum independent of visual size,
and selected controls turn amber.

Selecting the anchor opens translucent satellite handles around it. Curl and
Thumb use vertical drags, Spread uses a horizontal drag, Shift enables fine
control and Escape cancels. Hover shows the authored label; active drag also
shows the scalar value. Releasing submits the scalar ID through the shared
`preview_control_values` operation and the existing pose Apply/Auto Key/undo
path. The old permanent Grouped Finger FK slider block is no longer drawn, which
keeps the Pose panel independent of the number of authored control families.
Unknown scalar semantics receive a generic labelled handle, so non-humanoid rigs
still remain operable while richer authored glyph metadata is a later display
contract extension.

IK target, pole and FK handle sizes use the same projected-chain hybrid scale.
Two-bone controls measure their root-mid-tip world length; scalar groups measure
the longest authored anchor-to-driven-bone chain. Visual opacity remains low
while idle and strengthens on hover/selection, keeping dense rigs readable
without returning to a permanent slider shelf.
