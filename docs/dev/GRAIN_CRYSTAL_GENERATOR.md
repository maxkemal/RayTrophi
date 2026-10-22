# Tane ve kristal kütüphanesi üreticisi

2026-10-22 — kaynak uygulaması tamamlandı; kullanıcı derlemesi ve canlı kabul bekliyor.

## Kullanım

**Add Mesh → Procedural Generator → Type: Grains / Crystals**.
Mesh Grains içindeki **Create Grain / Crystal Library…** aynı pencereyi açar.
Bu yoldan açıldığında hedef domain görünür; **Use Library in Mesh Grains** açıksa
üretilen objeler örnek listesine bağlanır ve kendi materyalleri kullanılır.
Normal Add Mesh yolunda domain bağlamak zorunlu değildir.

Üretici her örneği ayrı, düzenlenebilir bir flat `TriangleMesh` olarak sahneye
ekler. Örnekler Origin etrafında aralıklı bir XZ ızgarasında yerleşir. Mesh'ler,
UV'ler, transform'lar ve materyaller normal proje geometri kaydında korunur;
ayrı bir asset dosyası veya facade koleksiyonu gerekmez. Oluşturulmuş mesh'ler
normal mesh araçlarıyla düzenlenir; bu sürüm parametrik modifier stack eklemez.

Bir Generate işlemi **tek Undo/Redo adımıdır**. Domain bağlandıysa Undo önceki
örnek listesini/shape/materyal kullanımını da geri getirir. Tekrar üretmek önceki
kütüphaneyi silmez; aynı isim varsa benzersiz son ek kullanılır.

## Şekiller ve kontroller

| Şekil | Kontroller |
|---|---|
| Rock / Sand | Cube-sphere çözünürlüğü 2–16, yuvarlaklık 0–1, Perlin düzensizliği 0–0.5 |
| Voronoi Fragments | 1–64 örnek isteği, seed, kenar yuvarlaklığı, Edge Detail 2–16; çok küçük hücreler elenebilir |
| Snow Crystal | 3–12 kol (varsayılan 6), her yanda 0–5 dal, dal uzunluğu, plaka kalınlığı, kol düzensizliği |
| Mineral Crystal | 3–12 prizma kenarı, uç yüksekliği 0–0.9; Quartz / Ice, Salt, Sugar biçim başlangıçları |

Ortak kontroller: örnek sayısı 1–64, seed 0–2147483647, XYZ aspect 0.05–5,
minimum/maksimum çap 0.00001–100 m, uniform/log-uniform boyut dağılımı, origin,
tek materyal veya 1–32 girdili ağırlıklı materyal paleti. Palet her örneğe bir
materyal seçer; ağırlıklar pozitif olmalıdır. Materyaller mevcut sahne
materyallerinden seçilir. Kar geometrisi tek kapalı ekstrüde konturdan oluşur;
üst üste binen saydam dal kutuları kullanılmaz. Dal uzunluğu, konturun kendini
kesmemesi için dal aralığıyla sınırlandırılır.

Çap, **üretilen sahne örneğinin en büyük eksen boyudur**. Mesh Grains örneği
carrier hacmine eşlemeye devam eder; örneğin sahne çapını değiştirmek DEM
yarıçapı/kütlesini değiştirmez. Fiziksel tane boyut dağılımı ayrı iştir.

## Ortak API

UI, Python ve IPC `rtapi::generateGrains` servisini kullanır. Geometri
`GrainGenerator.cpp` çekirdeğinde üretilir; sahne, materyal ve history işlemleri
`RtApiGrainGenerator.cpp` içindedir. Hatalı tür, aralık, bilinmeyen alan, olmayan
materyal/domain veya bozuk palet sahneye obje eklemeden reddedilir.

Python:

```python
result = rt.scene.generate_grains(
    type="snow", name="SnowLibrary", count=12, seed=42,
    arms=6, branches=3, branch_length=0.12, thickness=0.02,
    aspect=[1, 1, 1], diameter_min=0.001, diameter_max=0.003,
    size_distribution="log_uniform", origin=[0, 2, 0])
print(result["objects"])
```

Ayrı terminalden IPC:

```python
from rt_ipc import RtIpc

ipc = RtIpc()
try:
    result = ipc.call("scene.generate_grains",
        type="voronoi", name="SandLibrary", count=16, seed=42,
        roundness=0.15, diameter_min=0.02, diameter_max=0.04,
        origin=[0, 2, 0], domain="SandDomain")
finally:
    ipc.close()
```

`domain` verilirse aktif parçacık sisteminde mevcut Liquid/Matter domain olmalı
ve `geometry=icosphere` kullanmalıdır. Servis `near_mesh_shape=scene`, yeni
`near_mesh_sources`, `near_mesh_source_materials=true`, `near_mesh_enabled=true`
yazar. Render route, solver, near distance ve bütçe değişmez.

Palet örneği: `material_weights=[{"material": "Quartz", "weight": 8},
{"material": "DarkMineral", "weight": 2}]`. `material` ve `material_weights`
aynı istekte verilmez. `origin` üç sonlu sayı ve ±1e6 m içinde olmalıdır.
İsim 1–128 karakterdir; dahili `[` başlangıcı ve `_inst_` foliage işareti reddedilir.

Sonuç: `objects`, `count`, `requested_count`, `type`, `seed`, `domain` ve her
örnek için `object`, `diameter`, `material_id`, `triangles` alanları.

## Uyumluluk ve kapsam

Mesh Grains'teki sabit Rock/Voronoi dropdown'ı kaldırıldı. Eski sahne ve script
ayarları okunmaya devam eder; legacy üretim de aynı ortak çekirdeği çağırır.
Eski kaynakları değiştirmek için **Use Selected Scene Objects** veya yeni
üreticiyi kullanmak gerekir. Scene object listesinin dar paneldeki görsel
düzenlemesi bu partinin dışındadır ve değiştirilmedi.

## Kabul

Derlemeyi ve uygulamayı kullanıcı başlatır. Ajan proje veya shader derlemesi yapmadı.

1. Yeni Type ve dört şekil formunu aç; seed/count/biçim ayarlarıyla üret.
2. `python scripts/test/rt_grain_generator_ipc.py`: dört tip, flat mesh/bounds,
   yanlış girdide atomiklik, IPC sağkalımı ve tek adım Undo/Redo → PASS.
3. `--keep` ile örnekleri tut; materyal paletini ve UV'leri görsel kontrol et.
   Bir kopya proje kaydet/aç; objeler, konumlar, geometri ve materyaller korunmalı.
4. Mesh Grains kısayolundan domain'e bağla. Tek Undo objeleri kaldırıp eski
   kaynak listesini geri getirmeli; Redo yeni listeyi ve objeleri geri getirmeli.
5. DEM/MPM/fluid splat sahnelerinde üretilen örnekleri kullan; distance/bütçe,
   kaynak materyalleri ve yakın/uzak ayrımı önceki sözleşmeyi korumalı.
6. Kullanıcı C++ kaynak testlerinde `grain_generator_test.cpp` ve
   `grain_mesh_library_test.cpp` çalıştırılmalı. Determinizm, kapalı yüzey alan
   dengesi, pozitif hacim, kar sınır durumları ve materyal aktarımı denetlenir.

Ajan kontrolleri: Python AST, JSON/XML, proje kayıtları, descriptor `--check`
ve kaynak whitespace. Bunlar C++/canlı render kabulünün yerine geçmez.

## Voronoi rounded edges (2026-10-22)

A request for one example now selects a cut cell from at least 32 seeded cells,
instead of returning the uncut source cube. Count specifies output examples.
Roundness=0 preserves sharp facets. Positive roundness smooths the intersection
of supporting planes and adds real surface samples with analytic smooth normals.
Resolution (UI: Edge Detail, 2..16) subdivides each original triangle into
resolution squared triangles. This same control works in UI, Python and IPC.
Start with roundness=0.6 and resolution=6. Higher detail and example counts cost
more generation time. Regenerate existing meshes to apply the change.
