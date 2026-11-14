# Player Embree sınırı — 2026-11-13

Embree SDK include sınırı sıfırlandı. Bu başlık/backend kaynak ayrımıdır;
Player hedefi/link testi, C++ proje derlemesi ve canlı kabul yapılmadı.

## Değişiklikler

- `EmbreeHandles.h`, Embree 4'ün opaque device/scene ve callback argüman
  bildirimlerini sağlar. `EmbreeBVH.h` ve `HairSystem.h` SDK başlığı taşımaz.
  Veri üyeleri, alan sırası ve sınıf yerleşimi değiştirilmedi.
- `EmbreeBVH.cpp` SDK include'unu kendisi alır ve gerçek offline implementasyon
  olduğu için Player aday manifestinden dışlanır.
- Saç CPU BVH oluşturma, ray/occlusion ve volumetric brush sorguları
  `HairEmbree.cpp` içine taşındı. Destructor/clear aynı private release yordamını
  çağırır. Saç üretimi, interpolasyon, dynamics ve GPU buffer hazırlığı ortak
  `HairSystem.cpp` içinde aynı kalır.
- Studio `HairEmbree.cpp` derler. `HairQueryHost.cpp` Studio projesinde yalnız
  `None`; gelecekteki Player bununla linklenmeli. İki implementasyon aynı hedefte
  linklenmez. Host Embree sahnesi oluşturmaz; offline CPU ray/brush sorguları hit
  üretmez. Vulkan saç çizimi ortak strand buffer'larını kullanmaya devam eder.
  Player capability/authoring sınırı bu offline sorguları destekleniyor saymamalı.
- `ParticleRenderBridge.cpp` başka çalışma satırında olduğundan düzenlenmedi;
  başlık ayrımıyla SDK bağımlılığı kalktı ve iki derleme dalında denetlendi.

## Sayım düzeltmesi

Denetleyici UTF-8 BOM ile başlayan dosyanın ilk include satırını atlıyordu.
`utf-8-sig` okuma ve iki regresyon testi eklendi. Bu hata `EmbreeBVH.cpp`'yi
gizliyordu: önceki 6 sayısı düzeltilmiş okuyucuyla **7** idi. CUDA/OptiX, OIDN ve
Python sayıları bu düzeltmede değişmedi. Baseline yükseltilmedi; bağımlılıklar
ayrıldıktan sonra Embree=0 olarak sıkılaştırıldı.

Sonuç: **552 aday / 536 temiz**, CUDA/OptiX **0**, Embree **0**, OIDN **0**,
Python **16**. Aday sayısı sabit: bir Host eklendi, bir offline kaynak dışlandı.
Önceki raporlanan temiz sayısı 530; BOM düzeltmesinden sonraki gerçek başlangıç
529 idi. Son ölçümde yasaklı başlığa ulaşan yalnız 16 Python adayı kaldı.

## Doğrulama

- **14/14 MSVC `/Zs` PASS:** material API, ortak HairSystem, ParticleRenderBridge,
  Renderer ve InstanceManager SDK'lı/SDK'sız; EmbreeBVH, HairEmbree ve Main Studio;
  HairQueryHost Player. Altı Player include dökümünde CUDA/OptiX/Embree SDK yok;
  bu SDK include yolları kontrol komutlarından çıkarıldı.
- AssimpLoader.cpp aday include grafiğinde temizlendi; aktif vcxproj/tlog kaydı
  bulunmadığından `/Zs` kontrolüne dahil edilmedi.
- Dört taşınan CPU gövdesi ve sahne release gövdesi kaynak eşitliği PASS.
  Kalan ortak HairSystem gövdesi yalnız include/release çağrısı değişimiyle birebir
  aynı; yorum/boşluk/string literal biçimlendirmesi normalize edildi.
- İki BOM regresyon testi, XML/proje/filter tek kayıt ve karşılıklı kaynak seçimi,
  include audit ve `git diff --check` PASS. Shader veya shader ABI değişikliği yok;
  derlenmesi/dağıtılması gereken shader yok. IPC/script imzaları değişmedi.

## Sonraki kullanıcı kabulü

Önceki partilerle birlikte Studio TAM build/link ve proje reload; CPU Embree
render, material remap, instance/particle bridge, saç lazy BVH, brush picking,
shadow occlusion, clear/proje yeniden açma/kapanış. Vulkan saç/groom animasyonu ve
GPU upload dirty davranışı da kontrol edilmeli. Bu canlı/sayısal kabul koşulmadı.

SDK'sız başlıklar ortak kodun Embree sınıf metotlarını çağırmasını teknik olarak
engellemez. Player runtime/loader/renderer ayrımında bu offline çağrıların link
sınırı ayrıca çözülmeli; include=0 gerçek Embree kütüphanesiz link kanıtı değildir.
Sonraki parti 16 Python girişinin ortak servis/IPC yolundan ayrılmasıdır.
