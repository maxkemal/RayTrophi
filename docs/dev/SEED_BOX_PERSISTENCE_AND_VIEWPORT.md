# Seed kutusu: kalıcılık düzeltmesi ve viewport yönü

## Düzeltilen yol

Seed min/max, mod, mesh bağı ve reset tarifi hem ProjectManager hem SceneSerializer
tarafından zaten yazılıp okunuyordu. Sorun, runtime domain hareket ankrajları
yüklemede henüz kurulmadığı için ilk senkronizasyonun yeni domain varsayılan kutusu
üretmesi ve yüklenen sınırları ezmesiydi.

Runtime `fluid_seed_box_initialized` bilgisi eklendi. İki yükleyici de iki sınır
alanı mevcutsa bunu işaretler. Mevcut ortak seed yazarı da API/script/IPC ile
verilen kutuyu işaretler. İlk senkronizasyon yalnızca yeni, henüz tanımlanmamış
tarife varsayılan kutu üretir. Domain hareket ankrajları yine ilk senkronizasyonda
kurulur; kayıtlı kutuya yükleme sırasında hayali hareket uygulanmaz.

Bu runtime işareti ayrı proje alanı olarak kaydedilmez; var olan min/max
alanlarından yeniden türetilir. Mevcut domain taşıma ve boyutlandırma davranışı
korunur. Bu düzeltme canlı parçacık önbelleği kaydetme özelliği değildir.

## Viewport için önerilen sonraki kapsam — henüz uygulanmadı

- Domain altında seçilebilir “Seed Region” alt öğesi; ayrıca render mesh oluşturmaz.
- Seed Box modunda kutuyu seçip dünya eksenlerinde taşıma ve boyutlandırma.
  Mevcut veri AABB olduğu için döndürme sunulmaz; dönük şekil için Mesh Volume.
- Seed kutusu ve domain seçimi ayrı olmalı; kutuyu taşımak domain'i veya mevcut
  parçacıkları taşımamalı. Panelde merkez/boyut ve min/max aynı veriyi düzenlemeli.
- Sürükleme sırasında yalnızca tarif/önizleme değişmeli. Yeniden üretim, mevcut
  auto-reseed tercihiyle ancak işlem sonunda yapılmalı; her mouse hareketinde değil.
- Ortak seed-region düzenleme servisi, script ve IPC yüzeyleri, doğrulama,
  undo/redo ve gizmo aynı kapsamda teslim edilmeli. Mevcut fluid.seed doğrudan
  parçacık üretimi istediği için salt tarif düzenleme işlemi yerine kullanılmamalı.
- Küp/mesh, karmaşık ve dönük başlangıç hacimleri için mevcut seçenek olarak kalır.

## Kullanıcı derlemesi sonrası kontrol

1. Seed Box min/max değerlerini belirgin asimetrik bir kutuya ayarla.
2. Kaydet, yeniden aç; ilk senkronizasyondan sonra min/max ve cyan kutu korunmalı.
3. Ana proje açma ve SceneSerializer kullanan sahne yolunda tekrarla.
4. Yeni domain'de önceki varsayılan kutu üretilmeli.
5. Domain taşındığında seed beraber taşınmalı; domain boyutlandırıldığında seed
   bağımsız kalmalı. Reset tarifi ve canlı cache davranışını ayrı kontrol et.

Statik kontrol: `python scripts/test/check_fluid_seed_box_contracts.py`.
`fluid_seed_box_test.cpp` kullanıcı tarafından derlenecek helper regresyonudur;
canlı save/open testi yerine geçmez. Codex uygulamayı başlatmadı veya derlemedi.
