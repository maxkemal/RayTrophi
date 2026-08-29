# 2D Spline Authoring Guide

Bu belge Profile Spline çekirdeğinin UI, Python, IPC ve proje dosyası arasındaki
ortak sözleşmesini anlatır. River sistemi bu API’nin sahibi değildir; River,
ortak spline verisini tüketen ayrı bir kullanıcıdır.

## Curve type

- `linear`: Kontrol noktaları doğru parçalarının uçlarıdır.
- `bezier`: Anchor noktaları ve relative `tangent_in` / `tangent_out` handle’ları kullanır.
- `bspline`: En az dört kontrol noktasıyla uniform cubic B-spline değerlendirmesi yapar.

Curve type ile point role aynı kavram değildir. B-spline kontrol noktaları handle
değildir; seçilen kontrol noktasının pozisyonu doğrudan düzenlenir. B-spline
Insert ve Subdivide işlemleri cubic Boehm knot insertion kullanır; knot vektörü
payload ile saklanır ve ekleme sırasında mevcut eğrinin şekli korunur.

## Python

```python
import rt

print(rt.spline.list())
payload = rt.spline.get("OpenLineSpline")
rt.spline.insert_point("OpenLineSpline", segment=0, t=0.5)
rt.spline.subdivide("OpenLineSpline", segments=[0, 1], cuts=2)
rt.spline.extrude("OpenLineSpline", endpoint=1, position=(4.0, 0.0, 2.0))
```

`rt.spline.get()` versioned JSON metni döndürür. Değiştirilmiş payload,
`rt.spline.set(name, payload)` ile doğrulanarak uygulanabilir. `pivot_offset`
üç bileşenli lokal pivot alanıdır; viewport `P` pivot düzenlemesi, proje kaydı,
Python ve IPC bu tek canonical alanı kullanır.

## IPC

Her çağrı JSON-RPC gövdesi kullanır:

```json
{"id": 1, "method": "spline.subdivide",
 "params": {"name": "OpenLineSpline", "segments": [0, 1], "cuts": 2}}
```

Temel yöntemler `spline.list`, `spline.get`, `spline.set`, `spline.insert_point`,
`spline.subdivide` ve `spline.extrude`’dur. Başarısız doğrulamalar `__error`
alanıyla döner; UI, Python ve IPC aynı `SplineEditService` yolunu kullanır.
Mutasyonlar `SceneHistory` snapshot komutuna kaydedilir; bu nedenle `rt.undo()` ve
`rt.redo()` Python/IPC çağrılarından sonra da aynı spline durumunu geri getirir.

## Serialization

Proje JSON’unda `spline_objects` dizisi altında `rt.spline.v1` payload’ları
saklanır. Payload; isim, plane, curve type, closed state, transform, pivot_offset, anchor
pozisyonları, Bezier handle’ları ve user data’yı içerir. Seçim ve geçici hover
durumu kalıcı veri değildir; yüklemede temizlenir.

## Modifier hazırlığı

Spline authoring payload’ı modifier’lara doğrudan source olarak bağlanır:

1. `SplineEvaluationService` Linear, Bezier ve uniform cubic B-spline için
   position/tangent/normal-frame/arc-length/closest-hit üretir. Viewport overlay,
   Sweep, Revolve ve Loft aynı saf core servisini tüketir.
2. Screw/Revolve profili açısal örnekler.
3. Sweep path frame’i taşır.
4. Loft aynı parametre domain’inde birden fazla spline’ı eşler.
5. Skin çoklu spline kesitlerini ortak ring topology’ye yayınlar.
6. Tüm sonuçlar canonical flat `TriangleMesh` / DNA SoA yoluna gider.

Revolve/Screw seçili authoring profilini radius/height düzleminde yorumlar. Profil
açık bir yan kesit veya kapalı bir hacim kesiti olabilir; radius koordinatı negatif
olamaz. UI'da X/Y/Z dönüş ekseni ile derece cinsinden başlangıç/bitiş açıları seçilir.
Python ve IPC aynı ayarları `axis`, `start_angle`, `end_angle` ve eksen pivotu
alanlarıyla kullanır (`pivot_x/y/z` veya IPC `axis_pivot`);
API açı birimi radyandır. Tam `2*pi` tur açısal halkayı kapatır, kısmi tur seam'leri
açık bırakır. `radius_offset`, profile X değerine dönüşten önce eklenir; merkezli
Circle profilini torus gibi döndürmek için offset profil yarıçapından büyük seçilir.

Viewport authoring akışında X/Y/Z seçimi dünya eksenidir; profil objesinin
rotasyonu seçilen ekseni gizlice yeniden yönlendirmez. Eksen varsayılan olarak
spline obje pivotundan geçer. Object mode'da `G` spline ve pivotu birlikte taşır;
`P` pivot edit modunda turuncu artı işaretiyle gösterilen pivot, spline geometrisi
yerinde kalacak biçimde taşınır. `Axis Pivot Offset` bilinçli ek eksen ofsetidir.
Profil objesinin scale değeri profile bake edilir.

Revolve preview geometrisi spline pivotuna göre lokal üretilir. `Apply as Mesh`
sonucu spline'ın dünya pivotunu mesh transformu olarak devralır; vertex'ler dünya
koordinatına bake edilmez. Bu nedenle üretilen mesh'in origin/gizmosu world sıfırında
değil, kullanıcının gördüğü Screw eksenindedir.

Bir kapalı kesiti açık bir omurga boyunca taşımak `Sweep` işlemidir. `Loft`, iki
veya daha fazla kapalı kesiti birbirine bağlar; açık omurgayı Loft kesiti olarak
kabul etmez. UI, Loft sırasında seçilen açık spline'ı Sweep omurgası olarak
kullanmak için doğrudan geçiş sunar.

UI operation akışı source-first'tür: `Start Preview` kaynak spline seçimini korur,
profile/path/loft section point veya transform değişiklikleri viewport wireframe
preview'u otomatik günceller. Hatalar diagnostic code ve mesajla dock'ta kalır.
`Apply as Mesh` sonucu canonical flat mesh olarak ve undo kaydıyla yayınlar;
`Cancel Preview` yalnız transient preview'u temizler.

Modifier UI, Python ve IPC geometri matematiğini tekrar etmez; aynı evaluation ve
publish servislerini çağırır.

## 2D düzlem ve ortografik görünüş

Yeni spline oluştururken görünüşe karşılık gelen authoring düzlemi seçilir:
Front için XY, Top için XZ, Side için YZ. Nokta gizmosu hareket deltasını obje
lokalindeki bu düzleme kilitler; düzlem normali koordinatına ekran veya dünya ekseni
kayması yazılmaz. Revolve/Screw tek bir seçili kontrol noktasını değil, spline'ın
tüm segmentlerini `Profile Samples` sayısında değerlendirir.
Eski veya yanlış düzlemde oluşturulmuş kaynaklar `Authoring Plane` ile yeniden
yorumlanabilir; düzlem değiştirildiğinde kontrol noktaları ve handle'ların 2D
lateral/height koordinatları korunarak yeni düzleme remap edilir.

Right/Left görünüşünde YZ kullanılır ve Revolve profili `radius=Z, height=Y`
olarak yorumlanır. XY için eşleme `radius=X, height=Y`, XZ için
`radius=X, height=Z` olur. Revolve winding'i kontrol noktalarının ileri veya ters
çizilmesinden bağımsız olarak dışa normalize edilir.

`Open Line` tek eksenli dikey/spine primitive’idir ve canonical 2D height ekseni
Y üzerinde kurulur. Düzlem remap sonucu Front XY ve Side YZ için dünya Y,
Top XZ için dünya Z doğrultusunda oluşur. Circle, Rectangle ve Open Arc iki
ekseni birlikte kullandığından seçilen authoring düzlemini tamamen doldurur.

Spline kaynakları normal scene object yaşam döngüsünü kullanır: viewport veya
hiyerarşi seçiminden Delete/X, Object > Delete ve scripting/IPC object delete aynı
undo destekli `SplineObjectLifecycle` servisine gider.

## Geometry Nodes curve workflow

Bir mesh output objesinin Geometry Graph'ında şu zincir kullanılır:

`Spline Object -> Resample Curve -> Curve to Mesh -> Output`

`Curve to Mesh` profile inputu boşken circle profile ile Tube/Cable üretir. İkinci bir
kapalı `Spline Object` profile inputuna bağlanırsa aynı node custom profile sweep yapar.
Seçili spline point'indeki `Curve Radius`, serialized `userData1` alanıdır ve node'da
`Use Point Radius` açıkken path halkalarını çarpar. UI, `spline.set`, generic node graph
Python API'si ve IPC aynı source/core yolunu kullanır.

`Live Curve Preview`, source point/radius/transform veya node parametresi değiştiğinde
graph'ı yeniden değerlendirir ve sonucu canonical flat mesh olarak viewport'a yayınlar.
Spline seçimi korunur; dolayısıyla source edit edilirken üretim yüzeyi eşzamanlı değişir.
Profile operation preview ise scene/hierarchy'yi kirletmeden aynı generated triangle'ları
shaded transient surface olarak gösterir.

## Spline animation

Spline Animation panelinden mevcut frame'e object transform, tüm kontrol noktaları veya
ikisi birlikte keylenebilir. Point payload; Linear/Bezier/B-Spline konumlarını, Bezier
handle'larını, handle modunu, auto tangent durumunu, radius/user data kanallarını ve rengi
taşır. Aynı track üzerindeki point key'leri curve tipi, open/closed durumu, point sayısı ve
B-Spline knot topolojisini korumalıdır; uyumsuz key açık bir validation hatası üretir.

Timeline değerlendirmesi spline'ı deform ederken bağlı live Geometry Nodes graph'ını aynı
frame'de yeniden üretir. Böylece `Curve to Mesh` sonucu tel kafes bir vekil değil, kıvrılan
kablo/hortumun gerçek canonical flat mesh preview'udur. Python `rt.spline.insert_keyframe`
ve IPC `spline.keyframe.insert/remove/list` UI ile aynı `SplineAnimation` servisini kullanır;
`spline.animation.self_test` object, point ve radius interpolasyonunu deterministik test eder.
`rt.spline.create` / `spline.create` ise UI ile aynı undo destekli `SplineObjectService`
yolundan circle, rectangle, open line veya open arc kaynağı üretir.

Animated spline üzerinde insert, subdivide, extrude veya point delete yapılınca aynı
topoloji işlemi mevcut bütün point key'lerine uygulanır. Böylece key'ler korunur ve yeni
kontrol, her frame anahtarındaki eğri biçiminden türetilir; ham `spline.set` ile animasyon
topolojisini belirsiz biçimde değiştirmek validation hatasıdır.

`Curve to Mesh` yan yüzeyleri profile seam'inde çift vertex kullanarak U=0/U=1 ayrımını
korur. Kapakların ayrı rim vertexleri, planar UV'leri ve hard normalleri vardır. Node'daki
`Material ID=-1` host mesh materyalini devralır; açık bir ID verilirse canlı üretilen mesh
o materyalle render edilir. Animated spline kullanan graph'ta klasik Apply devre dışıdır:
mevcut engine'de mesh morph/vertex-cache track bulunmadığı için Apply canlı bağı yok ederdi.
Viewport/IPR'da materyalli current-frame render graph live tutularak yapılır. Final
animation sequence render'ının her frame'de aynı graph runtime'ını çağırması henüz
tamamlanmadı. İlerideki Vertex Cache Bake ayrı bir veri modeli, renderer playback,
serialization ve Python/IPC yüzeyi olarak eklenmelidir.

## Quick Skin / Bevel

The streamlined spline workspace follows `Shape & Controls -> Skin & Deform -> Animation`.
The legacy direct Sweep/Revolve/Loft controls remain available under the collapsed
`Advanced Surface Tools` section instead of occupying the main authoring flow.

`Create Live Deformed Skin` creates this editable Geometry Graph chain:

`Spline Object -> Curve Taper -> Curve Twist -> Curve Wave + Noise -> Curve to Mesh -> Output`

Taper multiplies the animatable point-radius field. Twist rotates the transported sweep
frame. Wave + Noise offsets curve controls deterministically. All three operate in the
Curve domain before mesh generation, so spline keys, material preview and fixed-topology
Geometry Cache baking remain compatible. Python `rt.spline.create_skin` and IPC
`spline.skin.create` expose the same parameters through `rtapi::createSplineSkinAdvanced`.

Skin is a persistent non-destructive display owned by the spline, not a repeated mesh
creation command. Enabling it creates one linked preview host; changing radius,
resolution, caps, Taper/Twist/Wave or the source controls updates that same host.
`Profile=Circular` uses the generated ring. Selecting another closed spline adds a
second `Spline Object` node and uses it as the live custom cross-section.

`Convert to Mesh` / `rt.spline.finalize_skin` / `spline.skin.finalize` evaluates the
current frame, removes the graph link and leaves the existing preview host as an ordinary
canonical flat mesh. `Remove Skin Display` / `clear_skin` / `spline.skin.clear` removes
only the linked preview; it never deletes the editable source spline. Display settings
and the host link are part of the spline payload, while temporary UI status is not.

`Bevel Radius` ana kalınlıktır. `Use Point Radius` açıkken sonuç yarıçapı, spline
point'lerindeki keylenebilir `Curve Radius` değerleriyle çarpılır. Path resolution ve
radial segments sabit kaldığı sürece topology değişmez ve çıktı doğrudan Geometry Cache
ile bake edilebilir.
