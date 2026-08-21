# Agent Mesh Modeling Cookbook

Bu tarifler ajanların mesh üretim hedefini RayTrophi içinde tamamlaması için başlangıç akışlarıdır.
Gerçek method parametreleri `mesh.tools.describe` ve IPC descriptor kayıtlarından alınmalıdır.

## Profile sweep

1. Profile workspace'te kapalı bir 2D profile oluştur.
2. Winding, self-intersection, hole/nesting ve cap politikasını validate et.
3. Curve workspace'te path oluştur; arc-length, frame, twist ve closed-loop seçeneklerini ayarla.
4. `preview` ile sweep yoğunluğu ve seam'i kontrol et.
5. `commit`, sonra normals/UV/material diagnostics çalıştır.

## Edit topology

1. Edit workspace ve doğru selection domain'i seç.
2. Edge loop/ring veya face region seçimini doğrula.
3. Extrude/inset/bevel gibi operasyonu preview et.
4. Yeni edge/face/poly sayısını ve boundary/non-manifold uyarılarını kontrol et.
5. Commit et; undo group ve revision değerini kaydet.

## SDF boolean

1. İki kapalı veya açık operandı isimleriyle keşfet.
2. Boolean workspace'te union/intersection/difference seç.
3. SDF backend, voxel size, narrow band ve smoothing değerlerini belirle.
4. Preview kalitesini kullan; self-intersection ve open-surface warnings'i kontrol et.
5. Commit sonrası remesh, normals, material transfer ve validation çalıştır.

## Sonuç doğrulama

Her üretim zincirinin sonunda şu kontroller yapılır:

- geometry revision beklenen şekilde arttı mı?
- changed vertex/edge/face/poly sayıları makul mü?
- zero-area veya flipped face oluştu mu?
- boundary/non-manifold durumu beklenen mi?
- UV/material/corner attribute transfer edildi mi?
- undo ve save/load sonrası aynı topology korunuyor mu?
- render ve edit overlay aynı mesh revision'ını gösteriyor mu?
