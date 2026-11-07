# UV-only shading güncellemesi

2026-11-05. Kaynak teslimi; yeni kullanıcı build'i ve canlı GPU kabulü bekleniyor.
Ajan build/compile çalıştırmadı, uygulama başlatmadı.

## Sorun ve ayrım

UV-only EditCommand genel `scheduleSceneMutationRebuilds(ctx, false)` çağrısıyla
geometry generation'ını artırıyor, raster/RT rebuild bayraklarını kaldırıyordu.
CPU BVH rebuild'i istenmese bile Solid ve RT tüm geometriyi yeniden yüklüyordu.

Materyal uniform/texture güncellemesi tek başına UV authoring için yeterli değildir:
UV'ler materyal parametre buffer'ında değil, geometriye bağlı ayrı shading
attribute buffer'larındadır. Yeni yol bu UV buffer'larını yerinde günceller;
pozisyon/indeks/BLAS/TLAS değişmez. RT/CPU accumulation yeniden başlar.

Mouse drag zaten detached preview üretir; mouse bırakılınca tek commit olur.
Bu ayrım transform/normalize/flip/swap/project ve topolojiyi değiştirmeyen corner/
atlas sonuçları ile undo/redo için aynı EditCommand üzerinden çalışır. No-op
yayın yapmaz. UI, script ve IPC aynı core işlemi çağırmaya devam eder.

## Resident backend yolları

- Vulkan RT: exact TriangleMesh kimliğine bağlı solo indexed BLAS'in packed
  allocation'ındaki UV aralığına upload yapılır. Alias handle'ın sıfır ofsetine
  yazılmaz; UV address - geometry base address ofseti kullanılır. AS refit yoktur.
- Solid/Material Preview: aynı mesh pointer'ıyla kurulan raster key bulunur;
  indexed/de-indexed ayrımı indexCount ile yapılır. Yalnız UV buffer ve cpuUVs
  aynası güncellenir. Seçim overlay'i, pozisyon/normal/index/instance buffer'ları
  yeniden kurulmaz. VulkanViewportBackend bu base adapter işlemini kullanır.
- OptiX: exact source_hittable kimliğiyle material bucket'lar bulunur. Gather,
  mevcut flat upload'ın face/material sırasını korur. Weld corner map varsa her
  GPU vertex'ine gelen yeni UV'lerin aynı olduğu doğrulanır; mevcut d_uvs yerinde
  güncellenir. SBT pointer'ı ve AS değişmez. Bu mevcut UV davranışının maintenance
  düzeltmesidir; OptiX'e özel yeni authoring yeteneği eklenmedi.
- CPU Embree: hit interpolasyonu canlı flat UV kanalından okur; BVH güncellenmez.

GPU in-flight okumaları, mevcut frame/stream senkronizasyonuyla upload öncesi
tamamlanır. Buffer sync maliyeti tamamen yok olmaz; tam geometri/AS kurulumundan
ayrıdır. Her UV commit'i resident buffer başına hedefli upload yapar.

## Geçerlilik ve fallback

Yeni IBackend::updateFlatMeshUVs sözleşmesi desteklenmeyen layout'ta false döner.
Legacy/merged, borrowed external geometry, eşleşmeyen count/kimlik, OptiX tangent
buffer veya UV'leri artık ayrışan GPU-welded vertex durumlarında yalnız başarısız
backend'in normal rebuild bayrağı kullanılır. Başarılı viewport bu yüzden yeniden
invalid edilmez. Başka işlemlerin önceden koyduğu pending bayrakları temizlenmez.
Henüz resident olmayan mesh sonraki lazy upload'da yeni flat UV'leri okur.

CPU topolojisini değiştiren set_corners/unwrap/pack vertex split sonucu mevcut
heavy geometry transaction'ında kalır; bu durumda geometri rebuild'i doğrudur.
GPU weld ayrışması da CPU indeksleri aynı olsa bile resident GPU layout'unu
değiştirmeyi gerektirir; böyle bir fallback, sıradan UV offset'iyle aynı şey değildir.

UV-only commit geometry generation'ını artırmaz. Panel refresh/undo görünürlüğü
için ayrı attribute revision sayacı, UV::sceneRevision bildirimine eklenir.
Surface/CPU UI mesh cache invalidasyonu korunur; bu, AS rebuild talebi değildir.

## Doğrulama

10/10 non-build kontrat kontrolü geçti. Yeni backend modüllerinin VS proje ve
filters kayıtları doğrulandı; CMake ilgili klasörleri zaten glob ile alır.
Kaynak sınırı kontrolü UV-only komutta genel scheduler olmadığını ve yeni backend
UV modüllerinde AS build/refit çağrısı bulunmadığını denetler. Bunlar canlı GPU
görsel doğrulamanın yerine geçmez.

Yeni kullanıcı build'inden sonra uygulama açık, sahne sakin/paused iken ayrı terminalde:

```powershell
python scripts/test/rt_test_uv_attribute_updates_ipc.py
python scripts/test/rt_test_uv_corner_ipc.py
python scripts/test/rt_test_uv_authoring_ipc.py
```

İlk probe geçici plane üzerinde UV transform/undo/redo ölçer: üç attribute publish,
sıfır CPU BVH/raster/RT build/refit ve sıfır fallback bekler. Fixture kurma/silmenin
meşru rebuild'leri ölçüm dışında tutulur. Global perf sayaçları ve scene history
etkilenir. Probe capability yoksa eski binary'de yazmaya başlamaz.

İlk probe Solid/Material Preview ve Rendered modlarında ayrı ayrı çalıştırılmalı;
backend değiştiriliyorsa OptiX de denenmelidir. Texture/normal map'li nesnede UV
hareketi, undo/redo ve mod değiştirip geri dönme görsel kontrolü yapılmalıdır.
Perf scope'ları `uv.attribute_publish`, `uv.vulkan.attribute_upload`,
`uv.optix.attribute_upload`, `uv.attribute_rebuild_fallback` ile takip edilir.
