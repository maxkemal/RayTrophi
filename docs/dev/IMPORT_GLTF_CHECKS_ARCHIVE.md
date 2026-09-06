# glTF migration acceptance history (through batch 9)

> **Durum:** ARŞİV — preserved before the ufbx increment. 2026-09-05.

> User confirmed: sections 7, 8 and 9(a) passed. Section 9(b) is duplicate geometry storage, not an incorrect triangle count. Seven shared meshes account for about 4.39M copied triangles; explicit instancing remains open.


**Bu parti bir REFAKTÖR DEĞİL, bir HATA DÜZELTMESİ.** Faz 3'ün ilk adımı olarak
`loadModelToTriangles`'ın çağıranlarını saydım ve 5. partinin "glTF artık
Assimp'e uğramıyor" iddiasının **eksik** olduğunu ölçtüm: iddia tek çağırma
noktasında doğrulanmıştı, oysa üç tane var.

| Çağıran | glTF dalı var mıydı |
|---|---|
| `Renderer.cpp` (`create_scene`) | ✔ vardı |
| `FoliageAssetLibrary.cpp` | ✘ **yoktu** |
| `scene_ui.cpp` (animasyon klibi) | ✘ **yoktu** |

Ve **bütün bitki kütüphanesi `.glb`** (`assets/vegetation/**`: her ağaç, çim,
çiçek). Yani 5. partiden sonra aynı dosya sahneye nasıl girdiğine göre iki farklı
sonuç veriyordu: model olarak sürüklenince doğru, bitki olarak dikilince
**Assimp'ten, ters V ile.** İki okuyucu, iki konvansiyon, tek sahne, hata yok.

Manuel test; script yok.

---

## §0 — Derleme

★ **Yeni dosya var, `.vcxproj` güncellendi:** `source/src/Import/ModelImport.cpp`
(+ `include/Import/ModelImport.h`). Filters dosyasına dokunulmadı — `GltfDirectReader`
de orada listeli değil, aynı düzeni korudum.

Yeniden adlandırma: `rtimport::GltfReadOptions` → **`rtimport::ImportOptions`**,
ve struct `GltfDirectReader.h`'den `Import/ImportedModel.h`'ye taşındı. İçindeki
hiçbir alan glTF'e özgü değildi; artık üç format için ortak olduğuna göre eski ad
yanlış olurdu (kural 5: anlamı genişleyen alanın adı da değişir). Altı yer.

`external/ufbx/` eklendi (ufbx.h + ufbx.c + LICENSE, MIT). **Derlemeye DAHİL
DEĞİL** — `.vcxproj`'a okuyucu yazıldığında girecek. Şu an sadece bir sonraki
partinin beklememesi için duruyor.

---

## §1 — ★★★ BİTKİ DOKULARI DÜZELDİ Mİ (bağımsız, en hızlı, partinin ASIL İDDİASI)

**Yap:** araziye bir ağaç/çim katmanı serp (`assets/vegetation/**` içinden
herhangi biri — hepsi `.glb`). Yakınlaş ve **yaprak/kabuk dokusuna** bak.

**Ne görmen gerek:** doku artık **düz**. Ayırt edici kıyas: **aynı `.glb`'yi
ayrıca sahneye model olarak sürükle** ve yan yana koy — ikisi **birebir aynı**
görünmeli.

**Bozuksa ne demek:**
- **İkisi hâlâ farklı görünüyorsa** → dispatch çalışmıyor, `FoliageAssetLibrary`
  bir şekilde Assimp dalına düşüyor.
- **Şimdi İKİSİ BİRDEN ters olduysa** → doğrudan okuyucudaki V çevirmesi yanlış
  yönde, ve 4. partide bunu kaçırmışız demektir (o tur simetrik olduğu için
  round-trip ile ölçülemiyordu — bkz. aynı adlı bellek notu).

★ **Bu partinin en sinsi riski burada değil, aşağıda (§2).** Bu madde gürültülü
bozulur; §2 sessizce bozulur.

## §2 — ★★★ Bitkiler HÂLÂ GÖRÜNÜYOR ve ÇOK MATERYALLİ mi (sessiz bozulma riski)

**Yap:** aynı serpme sahnesi, Vulkan RT ve raster.

**Ne görmen gerek:** orman **duruyor**, ve ağaçlar **birden fazla materyal**
gösteriyor (kabuk ayrı, iğne/yaprak kartı ayrı).

**Bozuksa ne demek:**
- **Orman kayboldu** → okuyucuya `emitSingleFacadePerMesh = false` geçmeyi
  kaçırmışım demektir. ★ Sebep: `InstanceManager` merkezlenmiş kopyaları
  `source.triangles`'ı **tek tek gezerek** üretiyor; temsilci facade verilirse
  bütün ağaç **tek üçgene** düşer.
- **Ağaçlar var ama hepsi TEK materyalde** → biri kaynağa `flat_meshes`
  vermiştir. ★ Bu **kasıtlı olarak yapılmadı**: kütüphane prototipi sahnede bir
  dünya nesnesi DEĞİL, ve `flat_meshes` BLAS'ını dünya-nesnesi geçişinin
  kaydından alıyor (`m_meshRegistry`). Kayıt yoksa backend `continue` ediyor —
  yani flat vermek burada **görünmez orman** demek olurdu; 5. partide tam olarak
  bu bedel ödendi. Kütüphane bitkisi facade şeklinde kalır; bunun bilinen
  bedeli çok materyalli kaynak ve ayrı bir iş (backend'in dünya nesnesi olmayan
  mesh için de BLAS kaydetmesi).

## §3 — Varlık tarayıcısından animasyon klibi

**Yap:** asset browser'dan bir **`.glb`** animasyon klibi sahneye ekle.

**Ne görmen gerek:** klip listeye giriyor ve oynatınca karakter **doğru** poz
alıyor.

**Bozuksa ne demek:** eskiden bu yol da Assimp'e gidiyordu, yani klip **eski skin
uzayında** geliyordu — Faz 1'in düzelttiği hatanın ta kendisi. Poz hâlâ bozuksa
klip okuma değil, klibin hedef iskeletle eşleşmesi (`source_prefix`) sorunudur:
artık `clip_model.importName` okunuyor, eskiden `loader->currentImportName`.

## §3b — ★★★ ÇOK MATERYALLİ OBJE TEK PARÇA MI (bu turda bildirilen gerileme)

**Yap:** çok materyalli bir `.glb` aç (ağaç tipik: kabuk + iğne + kart = 9
primitif). Outliner'a bak, birini seç, sonra ondan bir scatter kaynağı yap.

**Ne görmen gerek:** **TEK obje**. Seçim bütün ağacı alıyor, scatter bütün ağacı
serpiyor.

**Bozuksa ne demek:** materyal sayısı kadar obje görüyorsan `nodeName` yine
ayrılmış demektir.

★★★ **Bu bir tasarım tercihi değil, dokümante edilmiş bir SÖZLEŞME, ve bu turda
kırılmıştı:** glTF bir mesh'i **materyal başına bir primitif**e böler; okuyucu
her primitife `ad + "_prim"N` veriyordu, yani her materyal AYRI bir obje
oluyordu. Motor bunun tersine yazılmış, üç ayrı yerde:

| Yer | Ne diyor |
|---|---|
| `AssimpLoader::processNodeToTriangles` | bir düğümün bütün mesh'lerine **aynı** `uniqueNodeName`'i verir, gerekçesi satır içinde yazılı |
| `scene_ui_scatter.cpp::gatherScatterSource` | `nodeName` eşleşen **HER** TriangleMesh'i toplar — tek kardeşten kaynak kurmak daha önce *"object bütünlüğü yok"* diye raporlanmış |
| `scene_ui_materials.cpp` | çok materyalli importun **birden fazla** TriangleMesh ürettiğini ve `direct_mesh_nodes`'un yalnızca sonuncusunu tuttuğunu açıkça yazıyor |

★ Benzersizlik hâlâ olması gereken yerde: **düğümler ARASINDA**
(`resolveUniqueMeshNodeName`). Mesh başına kimlik gerektiren her şey **işaretçiyle**
anahtarlıyor (`"[DirectMesh]-<ad>-<ptr>"` BLAS anahtarı, raster `meshKey`) —
paylaşılan adın güvenli olmasının sebebi tam olarak bu.

★★ **Bu partiden ÖNCE kaydedilmiş proje dosyaları `_prim` adlarını taşıyor.**
Geriye uyum katmanı yazılmadı (kural 5); etkilenen varlığı yeniden import et.

## §4 — Gerileme: normal model importu değişmedi mi

**Yap:** bir `.glb` ve bir `.fbx` modeli her zamanki gibi aç.

**Ne görmen gerek:** ikisi de 5. partideki gibi. `.glb` için
`[glTF] direct reader: ...` satırı, `.fbx` için Assimp satırları.

**Bozuksa ne demek:** `create_scene` bu partide **sadece** `ImportOptions` adını
aldı, davranışı değişmedi. Burada bir fark görüyorsan yeniden adlandırma yanlış
bir yeri yakalamıştır.

★ `create_scene` bilerek `readGltf`'i **doğrudan** çağırmaya devam ediyor: ışık,
kamera, instance grubu ve import istatistiklerini tüketiyor, seam bunları
taşımıyor. Bu ikinci bir okuyucu değil, aynı okuyucunun **daraltılmış**
sarmalayıcısı — kural 5'in gerçekten önemsediği ayrım bu.

## §5 — ★★★ IMPORT HIZI — ✅ **DOĞRULANDI (2026-09-05)**

> Kullanıcı: *"import işlemi Assimp paralel yapıdan bile çok hızlı oldu."*
> Yani kaybedilen prefetch geri alınmakla kalmadı, üstüne yerinde unpack ve
> toplu indeks okuma da geldi.


**Yap:** çok dokulu bir `.glb` aç (bitki varlıkları tipik: ~20 atlas). Log'a bak.

**Ne görmen gerek:** İKİ yeni satır —
```
[glTF] texture prefetch: 20/20 image(s) on N worker(s) in T s
[glTF]   parse … s | materials+textures … s | geometry … s | animation … s
```
ve `materials+textures` süresinin **belirgin şekilde düşmüş** olması.

**Bozuksa ne demek:**
- **Prefetch satırı yoksa** → dosyada 2'den az benzersiz doku var (o zaman zaten
  iş yok) ya da `loadMaterials` kapalı.
- **`20/20` yerine daha az** → çözülemeyen ya da GPU'ya yüklenemeyen doku var;
  bunlar hata değil, `textureFor` onları seri yolda tekrar deniyor.
- **Dokular yanlış/karışık geldiyse** → prefetch anahtarı `(image, TextureType)`
  ikilisi; aynı görüntünün albedo ve roughness kopyası AYRI girdilerdir.

★★★ **Bu bir hızlandırma değil, KAYBEDİLEN bir hızlandırmanın geri alınması.**
`AssimpLoader::prefetchTextures` bu paralel geçişi uzun süredir yapıyordu;
doğrudan okuyucuya geçerken **taşınmadı**. Yani aynı dosya, aynı pikseller, N
çekirdek yerine 1. Hiçbir şey bozulmadı — import sadece yavaşladı, ve
**kaybedilen bir optimizasyon hiçbir şey olarak raporlanır.**

★ Faz süreleri zaten ÖLÇÜLÜYORDU (`ImportStats::seconds_parse/materials/
geometry/animation`), ama yalnızca toplam basılıyordu. Artık dördü de basılıyor:
"import yavaş" şikâyeti bundan sonra yeniden enstrümantasyon gerektirmiyor.

**Geometri tarafında yapılanlar** (aynı §, ayrı ölçüm gerekmez — üçgen sayısı
yüksek bir dosyada `geometry` süresi düşmeli):
- **Yerinde unpack.** Pozisyon/normal/UV artık doğrudan SoA dizisine açılıyor;
  geçici vektör + eleman eleman kopyalama gitti (`Vec3`/`Vec2` sıkı paketli,
  `static_assert` ile korunuyor). 356k köşelik bir ağaçta mesh başına ~12 MB
  ayırma+kopyalama iptal.
- **Toplu indeks okuma.** `cgltf_accessor_read_index()` indeks BAŞINA bir
  çağrıydı — 356k üçgen = 1M+ çağrı. `cgltf_accessor_unpack_indices` uint32
  kaynakta memcpy, uint16 kaynakta (çoğu exporter) tek sıkı genişletme döngüsü.
  ★ Sparse/buffer view'sız accessor'da 0 döndürüyor; bu hata değil, genel yola
  düşme sinyali — eski döngü o yüzden **duruyor**, "ulaşılmaz" diye silinmedi.

★ `cgltf_accessor_unpack_floats`'ı elle hızlandırmaya ÇALIŞMADIM: cgltf'in
kendisinde zaten float32 + doğal stride için memcpy hızlı yolu var (cgltf.h:2441).
Ölçmeden "generic görünüyor, elle yazayım" demek burada net zarar olurdu.

---

## §6 — ★★★ EXPORT → REIMPORT TURU — ✅ **DOĞRULANDI (2026-09-05)**

> Kullanıcı: *"doğru export import scatter işlemi oldu, layer sayısı doğru
> geldi, objeler doğru yerde ve boyutta."* — yani tur artık **kapanıyor.**


**Yap:** scatter içeren bir sahneyi `.glb` olarak dışa aktar, **yeni bir sahnede aç.**

**Ne görmen gerek:**
- Çok materyalli ağaç **TEK obje** (outliner'da bir satır, materyal sayısı kadar değil).
- **TEK foliage katmanı** (9 materyalli ağaç için 9 katman değil).
- Ormanın yerleşimi ve ölçeği kaynak sahneyle aynı.

**Neydi:** yazıcı her `TriangleMesh` için **ayrı bir glTF mesh + ayrı node**
yazıyordu. Çok materyalli obje N kardeş mesh olduğuna göre dosyaya **aynı adlı N
node** gidiyordu; geri açılışta bu adlar benzersizleştirilmek zorunda kalıyor ve
obje N parçaya bölünüyordu. Scatter tarafında daha da kötüydü: `planScatter`
kardeş **başına bir `EXT_mesh_gpu_instancing` node'u** yazıyordu — yani bir orman
katmanı dokuz katmana dönüşüyor, her biri ağacın bir parçasını taşıyor ve **kendi
bounding box'ına göre yeniden merkezleniyordu.** Dosya her seferinde geçerliydi.

**Ne yapıldı:** `planGeometry` artık `flat_` girdilerini **(nodeName, transform
işaretçisi)** ile grupluyor ve grubu **tek mesh + tek node** olarak yazıyor;
`planScatter` da kaynak başına **tek emission** üretiyor.
★ Transform işaretçisi anahtara bilerek girdi: yalnızca adı aynı olan ama başka
yerde duran iki obje **birleşmemeli**.
★★ Vertex verisi **iki kez yazılmıyor**: `emitFlatMeshGroup` her üyeyi
`flatMeshIndex_`'e kaydediyor, yani hem sahne objesi hem scatter kaynağı olan bir
mesh ikinci kopyayı değil aynı glTF mesh'ini paylaşıyor (`BinPlan`'da dedup yok —
her `add()` dosyaya ekler, bu yüzden önemli).

**Bozuksa ne demek:**
- **Obje hâlâ bölünüyorsa** → gruplama anahtarı tutmamış; iki kardeşin
  `transform` işaretçisi farklıysa (ayrı `Transform` nesneleri) ayrı gruplara
  düşerler.
- **Dosya boyutu belirgin büyüdüyse** → `flatMeshIndex_` cache'i ıskalanmış,
  prototipin vertex verisi hem sahne objesi hem scatter kaynağı olarak iki kez
  yazılmış demektir.

★★ **ÇÖZÜMLEMİŞTİM, ÜRETMEDİ — ama silmiyorum (aşağıdaki nota bak):**
orman **bütün** geldiği hâlde hâlâ **kaymış ya da yanlış ölçekte** duruyorsa,
sebebi bölünme değil **terimin iki kez uygulanması**dır. Yazıcı yerleşimlere
`T_i * translation(-mesh_center) * sourceWorld` gömüyor (dosya kendi kendine
yeterli olsun diye — doğru olan bu). Ama geri okurken prototip mesh'leri sahne
objesi olarak da geliyor ve **kendi node transform'larını taşıyorlar**, sonra
backend `translation(-mesh_center) * sourceWorld` terimini **bir kez daha**
uyguluyor. Prototip başlangıç noktasında ve birim ölçekliyse fark görünmez
(önceki testin muhtemelen böyleydi); başka bir yerde ya da ölçekliyse hem
**aralık** hem **boyut** kayar — tam senin tarifin.

✅ **Bu testın çıktısı: kayma YOK.** Yani terim iki kez uygulanmıyor — en olası
sebep prototipin origin'de ve birim ölçekli olması, ki scatter prototipi için
tipik yazarlık budur. ★ Risk **ölmüş değil, tetiklenmemiş**: prototipi taşınmış
veya ölçeklenmiş bir sahnede tekrar bak. Görülürse çözümü şu — `ScatterSource`'a
"yerleşimler ZATEN nihai" bayrağı koyup (a) `InstanceManager::buildOneSource`'un
içe aktarılan `mesh_center = 0`'ı **ezmesini** engellemek, (b)
`VulkanBackend` / `VulkanBackend_Raster` / `VulkanViewportBackend` / `OptixWrapper`
içinde `sourceToScatter`'ı birim almak. Dört+ dokunuş, ayrı ve test edilebilir bir
artış olmalı. **Bu maddede kayma görürsen söyle, sıradaki iş o olur.**

---

## §7 — ★★★ SPEC DIŞI AMA OKUNABİLİR DOSYALAR — ✅ DOĞRULANDI (2026-09-05)

> Kullanıcı `village_house_and_barn_glb.glb` için: "açıldı test ettim sorunsuz geçti".
> Bu varlığın açılış testi geçti; ayrı UV seti, prefetch sayacı ve log değerleri
> ayrıca raporlanmadığından bu sonuç onların bağımsız doğrulaması sayılmaz.

**Yap:** `village_house_and_barn_glb.glb`'yi (627 MB, Unreal Engine 5.2.1) aç.

**Ne görmen gerek:** dosya **açılıyor**, ve log'da bir UYARI:
```
[glTF] file is not spec-conformant (cgltf_validate result N) but is safe to read;
continuing. Generator: Unreal Engine 5.2.1. Empty primitives (no vertices,
skipped): 566 of 1141.
```

**Neydi:** `cgltf_validate` dosyayı reddediyordu ve okuyucu bunu **kesin ret**
saydığı için sahne hiç yüklenmiyordu. Ölçtüm: dosyada **1141 primitiften 566'sı
BOŞ** — mesh başına ikinci bir primitif, POSITION/NORMAL/TANGENT/TEXCOORD
accessor'larının hepsi `"count": 0` ve indeksi yok. Unreal her materyal slotu için
bir primitif yazıyor, LOD bölümünde geometri olmasa bile.

Şartname `accessor.count >= 1` diyor, yani **cgltf haklı** ve dosya gerçekten spec
dışı. Ama dosya **okunabilir**: `emitPrimitive` zaten sıfır sayılı POSITION'da erken
dönüyor, o 566 primitif hiçbir şey üretmiyor. Assimp yıllarca bu dosyayı **daha
müsamahakâr olduğu için** açıyordu; fallback kalkınca dosya tamamen açılamaz oldu.

★★★ **Bu, 5. partinin §1'inde yazılı riskin ta kendisi:** *"bir dosya eskiden
Assimp'in örttüğü için çalışıyor görünüyorduysa, artık AÇILMAZ."* Öngörülmüştü, ve
gerçekleşti.

**Ne yapıldı:** `cgltf_validate` artık **kapı değil, hızlı yol**. Reddederse
`validateReadSafety()` çalışıyor ve **yalnızca belleği koruyan** kuralları
denetliyor:
- accessor aralığı kendi buffer view'ına sığıyor mu (`cgltf_accessor_unpack_*`
  kendi sınır denetimini YAPMIYOR),
- buffer view aralığı kendi buffer'ına sığıyor mu,
- indeks accessor'ı skaler ve u8/u16/u32 mu (`cgltf_accessor_read_index` buna
  göre dallanıyor).

Bunlar geçerse dosya **uyarıyla** okunuyor; geçmezse ret, ama artık **sebebini
söyleyerek** ("accessor 12 reads 400 bytes from a 96-byte view" gibi) — eskisi
sadece "rejected the file" diyordu, ki bu teşhis edilemez bir mesajdır.

**Bozuksa ne demek:**
- **Dosya hâlâ açılmıyorsa** → artık log sebebini yazıyor; o cümleyi bana gönder.
  Gerçekten bozuk bir tampon varsa reddetmek **doğru** davranış.
- **Açılıyor ama geometri eksikse** → uyarıdaki "empty primitives" sayısına bak.
  566/1141 bekleniyor; çok daha yüksekse gerçek geometri düşüyor demektir.
- ★ **Çökme olursa** güvenlik denetimi eksik kalmış demektir — bu maddeyi hemen
  bildir, çünkü artık spec dışı dosyaları okumaya çalışıyoruz.

★ Bu dosya aynı zamanda §2'yi (ekstra UV setleri) ve §5'i (745 görüntü → paralel
doku prefetch'i) gerçek ölçekte sınar: `TEXCOORD_1` taşıyor ve 1140 materyali var.

---

## §8 — ★★★ MATERYALİN SAMPLE ETTİĞİ UV SETİ (`texCoord`) — 8. parti

**Yap:** aynı Unreal varlığını (`village_house_and_barn_glb.glb`) aç. Önceden
**dokusu gelmemiş gibi görünen bitkilere** bak.

**Ne görmen gerek:** dokuları geliyor. Materyal panelinde o objelerin
**UV Set = 1** gösteriyor olması gerekir (0 değil).

**Neydi:** glTF'te her `textureInfo` bir `texCoord: N` alanı taşır — "TEXCOORD_N'i
örnekle" demek. Düz şartname, uzantı değil. Okuyucu bunu **hiç okumuyordu**,
her zaman TEXCOORD_0'dan örnekliyordu.

Ölçüm (dosyayı doğrudan okuyarak):

| | |
|---|---|
| Boş olmayan primitiflerin kullandığı materyal | 574 |
| Yalnızca boş primitiflerin kullandığı (ölü) materyal | 566 |
| Dokusu olan canlı materyal | 248 |
| **Bunlardan `texCoord: 1` kullanan** | **241** |
| `texCoord: 0` kullanan | 7 |

Yani o dosyadaki dokulu nesnelerin neredeyse hepsi **yanlış UV setinden**
örnekleniyordu. Bu bir hata gibi görünmüyor — "doku gelmedi" gibi görünüyor.

★ **AssimpLoader bunu HEP okuyordu** (`AI_MATKEY_UVWSRC` → `selected_uv_set`);
doğrudan okuyucuya taşınmamıştı. Paralel doku prefetch'iyle **aynı aile**:
hiçbir şey bozulmadı, dosya sadece yanlış geldi.

**Ne yapıldı:** `preferredUvSet()` yardımcısı hem materyali hem geometriyi
besliyor, yani ikisi **ayrışamaz**:
- `materialIdFor` → `pbr->selected_uv_set`
- `emitPrimitive` → birincil `"uv"` özniteliğine **o** seti yazıyor (motorun
  konvansiyonu bu: `applyUVSet(n)` seti `"uv"`e kopyalar, `selected_uv_set`
  hangisi olduğunu kaydeder).

★ glTF UV setini **slot başına**, motor **materyal başına** modelliyor. Assimp
gibi "ilk dokulu slot kazanır" alındı, ama slotlar ayrışırsa artık **uyarı
basılıyor** — sessizce birini seçmek bu deponun klasik hatası olurdu.

**Bozuksa ne demek:**
- **Hâlâ düz renk / boş görünüyorsa** → log'da *"material wants TEXCOORD_n but
  the primitive only has TEXCOORD_0"* uyarısı var mı bak. Varsa dosya tutarsız,
  set 0'a düşülüyor (doğru davranış).
- **Bazı haritalar doğru, bazıları kaymışsa** → *"addresses DIFFERENT UV sets per
  texture slot"* uyarısına bak. Motor materyal başına tek set tuttuğu için bu
  dosya için tam karşılık yok; uyarı tam da bunu görünür kılmak için var.

### VRAM uyarısı — hata DEĞİL, ölçüldü

`~9181 MB texture / 12329 MB` = **%74.5**. `VulkanBackend` %70'te **uyarıyor**,
%85'te `checkAndTrimVRAMThreshold()` ile **kırpıyor** — yani sistem tasarlandığı
gibi çalışıyor, henüz kırpma eşiğinin altında.

★ Boşa giden doku **yok**: ölçtüm, 744 görüntünün **hepsi** canlı materyaller
tarafından isteniyor (ölü materyallere ait 0 görüntü). Yani "boş primitiflerin
dokularını atlayarak yer kazanırız" diye bir kazanç yok — varlık gerçekten ağır
(4K atlaslar, adında da yazıyor: `..._Billboard_4K_...`).

Gerçek bir azaltma isteniyorsa bu bir **ÖZELLİK** olur, hata düzeltmesi değil:
import'ta bütçe üstündeki dokuları küçültmek (mip düşürme), ya da gri tonlu
haritaları tek kanallı yüklemek (`Texture::is_gray_scale` zaten ölçülüyor —
RGBA8 yerine R8 4 kat yer kazandırır). ★ **Kullanıcı kararı olmadan yapılmadı:**
ikisi de görüntü kalitesini değiştirir.

---

## §9 — ÖLÇÜM: dokusuz gelen objeler + düğüm başına geometri çoğaltması

### (a) "Dokusu gelmeyen bitkiler" — ✅ KAPANDI: dosyada doku yok, hata bizde değil

> **Doğrulandı (2026-09-05):** kullanıcı baktı, objeler **pembe**. Bu
> `baseColorFactor = [1, 0, 1, 1]`'in birebir karşılığı, yani okuyucu dosyayı
> DOĞRU okuyor. Importer hatası değil — varlığın kendisi eksik.

Ölçüm (`village_house_and_barn_glb.glb`, dosya doğrudan okunarak):

| | |
|---|---|
| Boş olmayan primitiflerin kullandığı materyal | 574 |
| Bunlardan dokusu OLAN | 248 |
| **Bunlardan HİÇ dokusu olmayan** | **326** |

O 326 materyalin `baseColorFactor` değeri **`[1, 0, 1, 1]`** — yani **macenta**,
Unreal'in "materyal eksik" yer tutucusu. Adları: `MI_Lemongrass_..._lod5`,
`_lod6`, `_lod7`, `_lod8`... UE bu LOD'ları dokusuz, yer tutucu materyalle
export etmiş.

★ Ayırt edici gözlem şuydu ve **pembe** çıktı: macenta = dosya böyle;
gri/beyaz olsaydı `baseColorFactor` uygulanmıyor demek olurdu, yani ayrı bir hata.
Bu ayrımı önceden yazmak, "doku gelmedi" şikâyetini bir tahmin turuna değil tek
bir bakışa indirdi.

★★ **İSTEĞE BAĞLI, YAPILMADI: uzak LOD'ları hiç almamak.** Bu varlıkta her bitki
lod0–lod8 olarak AYRI obje geliyor; pembe olanlar lod5+ ve zaten yakın LOD'ların
üstünde duruyorlar. "Yalnızca LOD0'ı içe al" seçeneği hem bu kalabalığı hem §9(b)
çoğaltmasının bir kısmını çözer. ★ Ama bu bir **ad sezgisi** (`_lodN` son eki),
ve glTF'te standart LOD yok (`MSFT_lod` bu dosyada kullanılmamış). Sessiz bir
sezgiyle geometri düşürmek bu deponun sevmediği şeydir; açık bir import
seçeneği olarak, varsayılanı KAPALI yapılmalı. Kullanıcı kararı.

★ Bu, §8'in (texCoord) yerine geçmez: o düzeltme **dokusu OLAN** 248 materyali
ilgilendiriyor (241'i `texCoord: 1`). Düzeltmenin canlı olduğunu doğrulamak için
dokulu bir Lemongrass (lod0–lod4) seç ve materyal panelinde **UV Set = 1**
yazıyor mu bak. Yeni `.cpp` artımsal derlemede alınır; **tam rebuild gerekmez.**

### (b) ★★ AÇIK: 7 mesh tekrar tekrar yerleştiriliyor, geometri paylaşılmıyor

★ **Önce bir düzeltme:** bu madde ilk yazılışında "977 mesh / 8.8M üçgen log'u
şişik" diye çerçevelenmişti. **Yanlıştı** — 977 sayısı benzersiz mesh sayısıyla
değil **düğüm sayısıyla** kıyaslanmalıydı. Kullanıcı itiraz etti, yeniden ölçüldü:

| | |
|---|---|
| Benzersiz mesh / üçgen | 574 · 4.385.711 |
| **Düğüm yerleşimi / üçgen** | **975 · 8.778.982** |
| Motor logu | 977 mesh · 8.778.982 üçgen |

Düğüm başına primitif toplamı **977**, üçgen toplamı **8.778.982** — log ile
**birebir aynı, fark 0**. Yani **çift kayıt YOK**; okuyucu dosyanın düğüm
grafiğinde ne yazıyorsa onu kuruyor ve sayım doğru.

**Gerçek bulgu daha dar ve daha ilginç:** birden fazla düğümün paylaştığı mesh
**yalnızca 7 tane**, ama tekrar sayıları yüksek:

```
S_Old_Wooden_Beam_lod0_Var1        315 düğüm × 9.996 üçgen
S_Mossy_Mounds_lod0                 50 düğüm × 8.549
S_Mossy_Stone_Wall_lod0_Var1        20 düğüm × 21.046
S_Medieval_Modular_Door_lod0_Var1   10 düğüm × 9.428
S_Framed_Wooden_Window_lod0          9 düğüm × 39.227
S_Traditional_Lantern_lod0           2 düğüm × 37.004
Box_98A56D3E                         2 düğüm × 80
```

Fazladan tutulan geometri: **4.393.271 üçgen** — üçgen sayımız doğru, ama o
üçgenlerin yarısı aynı verinin kopyası. Bu RAM'e, VRAM'e ve BLAS sayısına
yazılıyor.

**Yapılacak iş (ölçüldü, yapılmadı):** okuyucu "aynı `cgltf_mesh`'i gösteren N
düğüm" desenini tanıyıp bunları `EXT_mesh_gpu_instancing` varmış gibi bir
`InstanceGroup`'a çevirebilir. 315 kopya tam olarak instancing'in var olma
sebebi; dosya sadece o uzantıyı kullanmamış, düz tekrar eden düğüm yazmış.

★ Neden otomatik yapılmadı: motor mesh başına tek `transform` tutuyor, yani
geometriyi paylaşmak **gerçek instancing** ister, sahne objesi kopyası değil. Ve
bir eşik gerekiyor — 2 düğümlük bir mesh için InstanceGroup kurmak kazançtan çok
karmaşıklık getirir. Eşiği ölçüye dayandır (bu dosyada 315/50/20 açık ara, 2'ler
sınırda), ve dönüşümü seçimin/outliner'ın gördüğü şeyi değiştirdiği için **açık
bir import seçeneği** yap.

---

## Bu partide KAPANMAYAN

- **Faz 3'ün asıl işi: FBX okuyucusu.** `external/ufbx/` yerinde ama tek satır
  kod yazılmadı. Sıradaki parti: `UfbxReader` → `ImportedModel`, ve
  `ModelImport.cpp`'deki Assimp dalının yerini alması. Ondan sonra OBJ, en son
  `assimp-vc143-mt.lib` düşer (`AssimpLoader.h` + `Texture.h`'daki `aiTexture`
  kurucusu + `AssetRegistry.cpp`'nin FBX/OBJ metadata dalı onunla ölür).
- **`AssetRegistry` hâlâ FBX/OBJ için `Assimp::Importer` kuruyor.** glTF dalı 5.
  partide `probeGltf`'e geçti; kalan dal ufbx gelince kapanır.
- ★ **Kural 1 borcu: import istatistiklerinin IPC yüzeyi yok.** `ImportStats`
  ölçüyor ama `scene.import_model` hiçbir şey döndürmüyor. Devam ediyor.
- ★★ **`KHR_texture_transform` OKUNMUYOR.** Blender'da Mapping node kullanan her
  varlık bu uzantıyla gelir; offset/scale/rotation sessizce yok sayılıyor.
  Yazarken dikkat: motorun `applyMaterialUVTransform`'u **(0.5,0.5) merkezli**,
  glTF'inki **(0,0) çıpalı**, üstüne import'taki V çevirmesi var.
- **UV düzenlemesi Vulkan RT'de tam rebuild** (raster'da sorun yok, ERTELENDİ).
- ★ **Yazıcının skin sözleşmesi.** Kendi export ettiğimiz skinli glTF standart
  semantiği sağlamıyor; okuyucu `generator == "RayTrophi Studio"` sniff'i ile
  telafi ediyor. Düzelmedikçe dosyalarımız Blender'da yanlış açılır.
- **Ekstra UV setleri (5. parti §2) ve varlık tarayıcısı (5. parti §4) hâlâ
  gözle doğrulanmadı.** Çok UV'li bir `.glb`'de `UV Set 1` listeleniyor mu, ve
  `.glb`'lerin tarayıcıdaki üçgen/mesh sayısı + önizleme çerçevelemesi doğru mu.
- **Seçim gizmosu bazı objelerde çizilmiyor.**
- **Oynatım hızı düzeltmesinin AĞIR sahnede doğrulanması.**
