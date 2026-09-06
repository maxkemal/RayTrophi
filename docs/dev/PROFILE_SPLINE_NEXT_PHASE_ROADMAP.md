# Profile Spline Next Phase Roadmap

> **Durum:** AKTİF — Faz 1/2 açık maddeler, Faz 3.6 uygulamada, Faz 3.7 (ortak yüzey çizimi + River'ın tüketiciye dönüşü) yeni eklendi, Faz 4 native UI smoke doğrulaması bekliyor

> Fixed-topology animation bake is implemented as the source-independent Geometry
> Cache contract documented in `GEOMETRY_CACHE_DEFORMATION.md`. New spline modifiers
> should preserve stable topology where possible so they can use this shared bake path.

Bu roadmap yeni, River'dan bağımsız 2D Spline authoring çekirdeği içindir. River
sistemi bu çekirdeğin sahibi değildir; ileride ortak spline verisini tüketen ayrı
bir kullanıcı olarak bağlanabilir.

## 2026-08-22 oturum checkpoint'i

Bugünkü Release doğrulaması başarılıdır. `scripts/test/rt_test_spline_curve_ipc.ps1`
gerçek açık uygulama üzerinde aşağıdaki zinciri uçtan uca çalıştırdı ve geçici test
objelerini sonunda temizledi:

`Spline Object -> Curve Taper -> Curve Twist -> Curve Wave + Noise -> Curve to Mesh -> Output`

Doğrulanan sonuçlar:

- Spline animation/self-test: `PASS`; point konumu, radius ve object transform interpolasyonu.
- Animated spline'a point eklendikten sonra iki key'in de üç point topolojisine taşınması.
- Sweep/evaluation: `PASS`; 25 vertex, 24 triangle, outward normal, point radius,
  seam-safe UV, B-Spline knot insertion ve closest-hit.
- Live deformed skin: 234 vertex, 384 triangle.
- Geometry Cache: 6 sample, 16.848 byte; interpolation, binary round-trip,
  topology guard, enable/disable/clear.
- IPC capability/descriptor audit: 358 method, 346 documented; security mirror ve
  descriptor güncelliği `PASS`.

Spline sağ dock ana akışı artık `Shape & Controls -> Skin & Deform -> Animation`.
Point ayrıntıları yalnız Edit Mode'da görünür; Taper/Twist/Wave progressive disclosure
altındadır; doğrudan Sweep/Revolve/Loft varsayılan kapalı `Advanced Surface Tools`
bölümündedir.

### Yarın buradan devam

1. Release UI'yi gerçek kullanım sahnesinde görsel olarak değerlendir: bilgi yoğunluğu,
   hizalama, boşluklar, seçili/disabled durumlar ve küçük dock genişliği.
2. Viewport-first eksikleri tamamla: güvenilir spline pivot taşıma/gösterimi, point-handle
   ortak selection kimliği, sürükle-onaylı endpoint Extrude ve numeric value bubble.
3. Sonraki üretim node'u olarak `Instance Along Curve` tasarla; sabit-topoloji olmayan
   instance çıktısını Geometry Cache sözleşmesinden açıkça ayır.
4. [Tamamlandı] Quick Skin'i aynı hostu güncelleyen persistent spline display yap;
   custom profile, Convert to Mesh ve Remove Display akışlarını ekle.
5. Modifier/node parametre animasyonu için ortak property-track kararı ver. Spline point
   ve object animasyonu hazırdır; node slider keyleme ayrı kapsamdır.
6. Project save/reload ile spline, graph node parametreleri ve Geometry Cache sidecar
   birleşik manuel smoke testi yap.

## Mevcut temel

- [x] Mesh-free `SplineObject`, XZ / Y-up authoring düzlemi ve temel primitive'ler.
- [x] Viewport point overlay, seçim ve tek sahipli point gizmo transform akışı.
- [x] Edit mode, selected-point dock kontrolleri ve point index overlay'i.
- [x] Bezier point handle verisi ve otomatik tangent yenileme.
- [x] Bezier adıyla sınırlı olmayan curve-type ayrımı (`Linear`, `Bezier`, `BSpline`);
      eski `BezierSpline` sınıf adı River uyumluluğu için geçici olarak korunuyor.

## Faz 1 — Ortak 2D Spline modeli ve edit servisi

- [x] UI mantığından bağımsız `SplineEditService` başlangıç modülü.
- [x] Bezier segmentine hit parametresinde control point ekleme altyapısı.
- [x] De Casteljau tabanlı Bezier segment insert/subdivide altyapısı.
- [x] Açık spline endpoint extrude altyapısı.
- [x] Eğri değerlendirme tipini point tipinden ayır: `Linear`, `Bezier`, `BSpline`.
- [x] Linear evaluation ve linear insert/subdivide.
- [x] Uniform cubic B-spline evaluation ve minimum dört control-point doğrulaması.
- [x] Açık uniform cubic knot modeli ve şekli koruyan B-Spline insert/subdivide politikası.
- [ ] Periodic knot modeli ve kullanıcı tarafından seçilebilir degree doğrulaması.
- [ ] Anchor, `InHandle`, `OutHandle` seçim kimliklerini ortaklaştır.

## Faz 2 — Viewport authoring araçları

- [x] UI araç seçimi: `Select`, `Insert Point`, `Subdivide`, `Extrude`.
- [x] Insert Point aracında spline üzerinde mouse hover hit testi ve preview marker.
- [x] Sol tıkla hit parametresine yeni nokta ekleme; eklenen noktayı seçili bırakma.
- [x] Insert Point aracı başarılı eklemeden sonra aktif kalır; ardışık segment
      eklemeleri için Select moduna zorla dönmez.
- [x] Subdivide paneli: cut sayısı ve seçili segment hedefi.
- [x] Sabit yükseklikte scroll’lu control-point listesi.
- [x] Ctrl ile çoklu point selection ve viewport/list selection senkronu.
- [x] Aktif point gizmo translation delta'sı seçili tüm control point'lere uygulanır.
- [x] Tek seçili segment veya çoklu seçili segment batch subdivide.
- [x] Edit mode dışında seçili segment subdivide operasyonu.
- [x] Bezier handle ve B-spline control-point düzenleme panelleri.
- [ ] Extrude yalnızca açık spline’ın geçerli ilk/son anchor’ında etkin olsun.
- [ ] Extrude sonrası yeni anchor’ın viewport mouse hareketiyle taşınması ve sol tıkla onayı.
- [ ] Closed spline için endpoint extrude’ı kapat; insert/subdivide yine çalışsın.
- [x] Hover, preview, invalid-state, API undo/redo transaction ve dirty işareti.

## Faz 3 — Değerlendirme ve üretim tüketicileri

- [x] Ortak `SplineEvaluationService`: position, tangent, normal/frame, arc-length ve
      3D closest-hit; Linear, Bezier ve uniform cubic B-spline aynı saf core yolu kullanır.
- [x] Screw/Revolve, Sweep, Loft ve profile viewport overlay bu servisi kullanır;
      UI içinde geometri üretimi yoktur.
- [x] Contextual spline dock içinde seçili authoring kaynağıyla Sweep,
      Revolve/Screw ve iki kesitli Loft mesh üretim kontrolleri.
- [x] Profile operation preview oturumu kaynak spline'ları seçilebilir/editable
      tutar; linked source veya settings değişince wireframe geometry otomatik
      rebake edilir, `Apply as Mesh` ve `Cancel Preview` açık transaction sınırıdır.
- [x] Başarısız preview diagnostic code/message dock'ta kalıcıdır; başarılı
      preview vertex/triangle sayısını gösterir ve seçim Apply'a kadar değişmez.
- [x] Sweep profile object final scale'i ve ek Profile Scale değerini birlikte
      kullanır; Revolve source transformunu preview ve committed mesh'e taşır.
- [x] Revolve Axis Radius Offset, merkez etrafındaki Circle gibi negatif-X'e
      geçen kapalı profilleri torus-style dönüş için eksenden uzaklaştırır.
- [x] Revolve/Screw açık veya kapalı radial profile, X/Y/Z ekseni ve partial/full
      açı aralığını UI, Python ve IPC üzerinden ortak settings ile destekler.
- [x] Sweep çift normal terslemesi kaldırıldı; Loft winding'i section seçim
      sırasından bağımsız dışa yönlendirilir ve self-testler outward normal doğrular.
- [x] Sweep/Revolve/Loft evaluated sonucunu canonical flat `TriangleMesh` / DNA SoA
      publish yoluna verir; scene commit mevcut ortak publisher/command katmanındadır.
- [x] Plane, closed state, curve type ve handles güvenli serialization; seçim durumu
      transient kabul edilip yüklemede güvenli biçimde sıfırlanır.
- [x] Deterministic outward normals, seam-safe side UV, planar cap UV, material
      inheritance ve canonical flat modifier output policy.

## Faz 3.5 — Geometry Nodes curve data flow

- [x] `Curve` socket payload ve scene-authoritative immutable spline snapshot.
- [x] `Spline Object`, `Resample Curve` ve `Curve to Mesh` node'ları.
- [x] Opsiyonel kapalı profile; profile bağlanmazsa circle tabanlı Tube/Cable üretimi.
- [x] Control-point `userData1` değerini path radius multiplier olarak sweep halkalarına taşı.
- [x] Source point, radius, transform veya node parametresi değişince live graph preview rebake.
- [x] Graph sonucu canonical flat `TriangleMesh` / DNA SoA olarak gerçek viewport geometrisine yayınlanır.
- [x] Profile operation transient preview wireframe yerine triangle tabanlı shaded surface gösterir.
- [x] Linear/Bezier/B-Spline point, handle, radius/user data ve spline object TRS keyframe sistemi.
- [x] Timeline frame evaluation sırasında live Geometry Nodes curve graph rebake.
- [x] Curve Deform: fixed-topology Taper, Twist ve deterministic Wave + Noise.
- [ ] Instance Along Curve.

## Faz 3.6 — Genel amaçlı 3D curve: terrain/scatter/mesh yüzeyi tüketicileri

> Bu faz, 2026-08-23 tasarım tartışmasının çıktısıdır. Amaç: profile authoring'e
> özgü `SplinePlane` düzlem kilidini genel amaçlı 3D curve kullanımından (yol
> güzergahı, force field yönü, hair guide, mesh yüzeyi sınır çizgisi) ayırmak.
> River sistemi bu çekirdeğin sahibi değildir ve ayrı kalır (bkz. dosya başı not);
> bu faz River'ı taşımaz, yalnızca River'ın bugün terrain'e kilitli olan
> surface-hit point-ekleme desenini genel çekirdeğe açar.
>
> Road üst katmanının kapsamı ve kabul kriterleri
> [TERRAIN_ROAD_NETWORK_ROADMAP.md](TERRAIN_ROAD_NETWORK_ROADMAP.md) içindedir.
> Curve geometrisinin otoritesi `SplineObject` olarak kalır; road katmanı ayrı
> kontrol noktası, gizmo, undo veya `set_points` API'si kurmaz.

- [x] `SplinePlane::Free` eklendi: point-edit gizmo'da (`ProfileSplinePointGizmo.cpp`)
      hiçbir eksen sıfırlanmaz; profile-tüketen `canonicalProfile()` artık Free'yi
      sessizce YZ gibi yanlış yorumlamıyor (profileScale identity'ye düşer).
- [x] Sweep ve Revolve giriş noktaları (`ProfileSplineEditor.cpp`) `SplinePlane::Free`
      kaynak profili `non_planar_profile` diagnostic code'uyla reddediyor — mevcut
      preview diagnostic deseni (`Faz 3`) kullanılarak, sessiz yanlış sonuç yerine.
      Loft zaten `worldSpline()` (3D, plan-bağımsız) kullandığı için dokunulmadı.
      DERLENMEDİ.
- [ ] `Curve to Mask` node (genel amaçlı, road'a özel değil): `Curve` girdisini
      (mevcut `DataType::Curve` pin tipi, terrain graph'ıyla paylaşılıyor) arc-length
      örneklemesiyle `Image2D`/`Mask` çıktısına rasterize eder; genişlik `userData1`
      radius multiplier deseninden gelir (Curve Deform ile aynı desen). Açık spline
      için width-based stamp, kapalı spline için inside-test (point-in-polygon) modu.
- [ ] `Road Carve` terrain node: `Curve to Mask` çıktısını + `Max Grade` kısıtını
      kullanarak cut-and-fill height adjust yapar. `RiverBedCarveNode`'un
      non-destructive carving deseninden ilham alır ama ondan farklı olarak arazi
      eğimini pasif takip etmez, aktif olarak sınırlar.
      Solver Faz A'da bir kez çalışır ve graded height ile ölçüm alanlarını aynı
      immutable/revision-tagged sonuçta tutar; Faz B publisher bu sonucu yeniden
      çözmeden yayınlar. `analysisFields.clear()` bu snapshot'ı geçersiz kılamaz.
- [x] Extrude, River'ın "Add" modeliyle çalışacak şekilde yeniden tasarlandı
      (`ProfileSplineOverlay.cpp`, `surfaceSnapPosition()`): tetikleme artık
      küçük endpoint ikonuna ekran-uzayı proximity ile isabet etmeyi
      gerektirmiyor (ilk sürüm buydu ve "kontrolsüz" hissettiriyordu — tıklanan
      ikon ile ray'in çarptığı 3D derinlik farklı olabiliyordu). Bunun yerine:
      Extrude aracı açık spline'ın geçerli bir uç noktasında (`selected_point`
      0 veya son index) aktifken her tıklama, imlecin hedeflediği yere yeni
      nokta ekliyor ve o nokta bir sonraki tıklama için uç nokta oluyor —
      River'da `addControlPoint`'in her klikte zincirlemesiyle birebir aynı
      akış. Hedef pozisyon sırasıyla: genel scene BVH + linear fallback (mesh,
      `scene_ui_selection.cpp` deseni) → `TerrainManager::intersectRay`
      (terrain, `scene_ui_river.hpp:705` deseni) → River'ın Y=0 zemin fallback'i
      (hiçbiri hit vermezse). Ayrı bir IPC parametresi gerekmedi —
      `SplineEditService::extrudeEndpoint` zaten açık Vec3 hedef alıyor,
      script/IPC çağrıları snap'i hiç görmeden kendi pozisyonunu veriyor
      (rt.ui istisnasındaki gibi: mutasyon her zaman aynı açık-pozisyon core
      çağrısından geçiyor). DERLENMEDİ.
- [ ] Insert Point aracına da aynı Surface Snap uygulanmalı (şu an yalnızca
      var olan eğri üzerinde hit-test yapıyor, boş uzayda/yüzeyde yeni nokta
      başlatmıyor — bu hâlâ açık).
- [ ] Extrude'un canlı mouse-follow + sol tık onay akışı (Faz 2'nin açık maddesi)
      Surface Snap'i her frame çağırıp önizleme göstermeli; şu an yalnızca tek
      tıklamada anlık snap oluyor, sürükleme sırasında canlı takip yok.

## Faz 3.7 — Yüzey üzerine çizim ORTAK bir servis; River bir TÜKETİCİ olur

> 2026-08-30 tasarım kararı. Tetikleyen gözlem: arazi üzerine Water panelinden
> mouse ile spline nehir çiziliyor, ama o eğri genel spline sisteminde
> **görünmüyor** — `spline.*` IPC'si yok, keyframe alamıyor, Curve Deform veya
> `Curve to Mask` tüketemiyor. Sorun "nehir eksik" değil; **üç ayrı katman tek
> bir struct'a kaynamış** durumda.

### Bugünkü üç katman ve nerede çakışıyorlar

| Katman | Ne yapar | Bugün |
|---|---|---|
| **Yerleştirme** | imleç altındaki yüzeyde dünya noktası bulur | İKİ kopya: `surfaceSnapPosition` (`ProfileSplineOverlay.cpp`, **anonim namespace** — dışarıdan erişilemez; mesh BVH + terrain + Y=0) ve `scene_ui_river.hpp`'deki satır içi kopya (**yalnız terrain** + Y=0, mesh'i hiç görmez) |
| **Eğri** | yazılan kontrol noktaları | `SplineObject::spline` ve `RiverSpline::spline` — **ikisi de aynı tip: `BezierSpline`** |
| **Tüketici** | eğrinin ne anlama geldiği | `RiverSpline`'a kaynak: hidrolik veri, su parametreleri, üretilmiş mesh, `waterSurfaceId` hepsi aynı struct'ta |

İkinci satır kararı verir: nehrin eğrisi ile spline object'in eğrisi **zaten
aynı tiptir**. Yani bu bir yeniden yazım değil, bir **ayrıştırma**.

### Hedef yapı: bir kez çiz, çok kez ilişkilendir

```text
Surface Draw tool  ->  SplineObject  (tek yazarlık otoritesi)
                            |
        +-------------------+-------------------+
        |                   |                   |
   RiverAttachment    RoadAssignment      (hair guide, scatter path, ...)
```

Eğri kimin olduğunu bilmez; **tüketici eğriye bağlanır**. Aynı eğri hem yol hem
de nehir geçişi referansı olabilir.

- [x] **3.7a — `SplineSurfaceAuthoring` servisi.** YAZILDI, DERLENMEDİ. `surfaceSnapPosition` anonim
      namespace'ten çıkar, kendi `.h/.cpp` çiftine taşınır ve sonuç bir
      **değer** döndürür: pozisyon + normal + neyin vurulduğu
      (`Mesh | Terrain | GroundPlane`) + vurulan nesne kimliği. Yalnız pozisyon
      döndürmek, çağıranın "ne vurdum" bilgisini varsayıma çevirir.
- [x] **3.7b — `SurfaceFilter` (SESSİZ DAVRANIŞ DEĞİŞİMİNİ ÖNLER).** YAZILDI, DERLENMEDİ. River'ın
      kopyası mesh'i görmüyor; ortak servise geçince nehir çizerken bir kayanın
      veya köprünün üstüne snap olmaya **başlar**. Bu bir hata olarak
      görünmez — makul görünen yanlış sonuçtur. O yüzden servis
      `TerrainOnly | MeshAndTerrain | GroundPlane` filtresi alır ve River
      `TerrainOnly` seçer: davranış kasıtla korunur, tesadüfen değil.
- [ ] **3.7c — River eğri sahipliğini bırakır.** `RiverSpline` içindeki
      `BezierSpline spline` alanı kaldırılır; yerine `spline_object_id`. Kalan
      alanlar (hidrolik dizisi, mesh ayarları, `WaterWaveParams`,
      `waterSurfaceId`) `RiverAttachment` olur — road katmanının
      `RoadAssignment`'ıyla **aynı şekil**. İki bağımsız tüketicinin aynı şekle
      yakınsaması, şeklin doğru olduğunun kanıtıdır.
- [ ] **3.7d — Nokta başına kanal ADLANDIRILIR.** `BezierControlPoint` bugün
      `userData1/2/3 + userColor` taşıyor; River 1'i genişlik, 2'yi derinlik
      olarak kullanıyor, Road da genişlik isteyecek. **Yeni `userDataN`
      eklenmez** — anonim slot, tüketiciye göre anlam değiştiren alan demektir
      ve bu deponun tekrar eden arıza sınıfıdır. Bunun yerine attachment bir
      **kanal haritası** beyan eder (`width -> userData1`), UI slider'ı o adı
      yazar ve `spline.set` kendini açıklar hale gelir. Dördüncü kanal ihtiyacı
      doğarsa slot eklemek yerine adlandırılmış attribute'a geçilir.
- [x] **3.7e — Tek çizim aracı.** YAZILDI, DERLENMEDİ. "Nehir ekle" bir mod olmaktan çıkar; viewport
      aracı `Draw Curve on Surface` bir `SplineObject` üretir/uzatır, tüketici
      sonradan iliştirilir. Extrude'un zaten River'ın "her tıkta zincirle"
      modeline taşınmış olması (Faz 3.6) bu aracın hazır davranışıdır.
- [ ] **3.7f — Göç, sessiz değil.** Proje yüklenirken her `RiverSpline`,
      `SplineObject` + `RiverAttachment`'a çevrilir ve **serileştirme anahtarı
      adı değişir** (`rivers` -> `river_attachments`). Kural 5: ölü yolu sök,
      ama anlamı sessizce değiştirme — eski anahtar adı korunursa yeni kod eski
      dosyayı yarım okur ve bunun hiçbir belirtisi olmaz. Dönüşüm loglanır.
- [~] **3.7g — River script/IPC'ye açılır (kural ★★★1 açığı).** Eğri tarafı YAZILDI (`scene.raycast`, `spline.append_point`, `spline.create` primitive=`empty` plane=`free`); `river.*` attachment yüzeyi 3.7c/3.7f göçüne bağlı ve AÇIK. Bugün
      `river.*` diye **hiçbir IPC metodu yok**; nehir sistemi panel-only, yani
      test edilemez. Göçten sonra eğri tarafı `spline.*`'tan bedava gelir;
      geriye yalnız attachment kalır: `river.attach`, `river.detach`,
      `river.list`, `river.set_params`. Beş dokunuş + overlay satırı.

### Bu partide ayrıca çıkan üç şey (2026-08-30, YAZILDI/DERLENMEDİ)

- **Canlı imleç takibi.** Extrude kodda "her tıkta zincirle" modelindeydi ama
  snap yalnızca tık anında oluyordu; ekranda hiçbir şey noktaın nereye
  düşeceğini söylemediği için araç "otomatik bir noktaya uzatıyor" gibi
  hissettiriyordu. Artık her frame snap çağrılıp hedef nokta ve son noktadan
  çizgi çiziliyor. **İşaretçi rengi ne vurulduğunu söylüyor** (mavi=terrain,
  turuncu=mesh, gri=zemin düzlemi) — kayaya düşecek bir nokta tıktan ÖNCE
  görülüyor.
- **★ Extrude dünya koordinatını yerel diziye yazıyordu.** Kontrol noktaları
  eğrinin yerel uzayında tutuluyor, snap sonucu ise dünya pozisyonu. Taşınmış
  veya döndürülmüş bir eğriyi extrude etmek noktaı ofset kadar yanlış yere
  koyuyordu; identity transform'da görünmüyordu. `transform.inverse()` eklendi.
- **★★ River'ın ilk mesh'i hiçbir zaman otomatik üretilemiyordu.** Kare
  döngüsündeki kapı `river.needsRebuild && river.flatMesh` istiyordu, ama
  `flatMesh` yalnızca `generateMesh()` içinde doğuyor — yani yeni çizilen bir
  nehir "Rebuild Mesh" düğmesi bulunana kadar meshsiz kalıyordu.
  `updateAllRivers` de sadece proje yüklenirken çağrılıyor. Kapı
  `pointCount() >= 2` oldu (parametre düzenleme yollarında da aynısı).
  **Carve elle kalmaya devam ediyor**: mesh türetilmiş ve ucuz, carve araziye
  yıkıcı.

Ölçüm: `scripts/probe_surface_curve_authoring.py`

### ★★★ Açık kök: dördüncü bir carve uygulaması var

`terrain.carve_river` (`RtApiTerrain.cpp:1456`) bir nehir spline'ını örnekleyip
`TerrainManager::carveRiverBed` / `carveRiverBedNatural` ile **doğrudan
heightmap'e yazıyor** — imperatif, yıkıcı, `TerrainSnapshot` ile geri alınıyor.

Yani "spline -> yükseklik" yolu bu depoda zaten var; olmayan şey
**non-destructive, graph'ta değerlendirilen, alan yayınlayan** yol. Bu ayrım
önemli çünkü `Road Carve` devreye girdiğinde aynı işin iki uygulaması olacak ve
ikisi kaçınılmaz olarak ayrışacak. Karar Faz 3.6 bitmeden verilmeli:

1. `carveRiverBed*` `Road Carve`'ın bir profili haline gelir (tercih edilen), veya
2. açıkça "yıkıcı tek seferlik authoring aracı" diye etiketlenir ve graph
   yolundan ayrı tutulur, veya
3. sökülür (kural 5).

Karar verilmemesi üçüncü sessiz yol demektir.

## Faz 4 — Scripting, IPC ve doğrulama

- [x] Insert, subdivide, extrude ve serialization için scripting API ve IPC.
- [x] UI, scripting ve IPC’nin aynı edit core yolunu kullanması.
- [x] Validation/error semantiği, capability audit kaydı ve IPC descriptor dokümantasyonu.
- [x] Spline animation insert/remove/list/self-test için ortak API, Python ve IPC yüzeyi.
- [x] Cubic B-Spline knot insertion ve modifier evaluation API.
- [x] Curve-type evaluation için pure core self-test (`runSplineEvaluationSelfTest`).
- [x] Release IPC regression ve cleanup doğrulaması.
- [ ] Native UI görsel smoke ve project save/reload manuel doğrulaması.

## Tasarım kararları

`CurveType` ile `PointRole` aynı enum değildir. Bezier anchor noktası `Anchor`, tangent
kolları `InHandle` ve `OutHandle` rolündedir. B-spline kontrol noktası eğri üzerinde
bulunmak zorunda olmadığından Bezier handle alanları B-spline verisine zorunlu eklenmez.

Subdivide şekli koruyan bir geometri operasyonudur: Bezier için De Casteljau, linear için
segment bölme, B-spline için knot insertion kullanılacaktır. Mouse hit testi evaluation
service'in segment ve `t` sonucuna dayanmalı; ekran örneklemesi yalnızca ilk hover kabuğudur.

## Manual validation gate

- `mesh.profile.sweep.self_test` sonucunda iç içe evaluation raporunun `PASS`,
  linear midpoint/length ve B-spline midpoint değerlerinin deterministik olduğunu doğrula.
- Insert Point hover/sol tık, subdivide cut sayısı ve açık uç Extrude akışını doğrula.
- Closed spline’da endpoint Extrude’ın engellendiğini doğrula.
- Multi-select transform, handle transform ve undo/redo sınırlarını doğrula.
- Save/reload sonrası curve type, handles, knots ve seçim güvenliğini doğrula.
- Bir mesh host graph'ında `Spline Object -> Resample Curve -> Curve to Mesh -> Output`
  bağla; spline point ve Curve Radius değişikliklerinin Live Curve Preview ile mesh'i yenilediğini doğrula.
- `spline.animation.self_test` ile frame 0/10 arasındaki object transform, point ve radius midpoint sonucunu doğrula.
- `spline.skin.create` ile Taper/Twist/Wave node kimliklerini ve üretilen flat mesh'i doğrula.
- Aynı host üzerinde `geometry_cache.bake/status/set_enabled/clear` ve topology guard'ı doğrula.
