# Player Python include sınırı — 2026-11-13

Python bağlamaları ortak servis ve IPC kaynaklarından ayrıldı. Son ölçümde
**547 adayın 547'si temiz**: CUDA/OptiX, Embree, OIDN ve Python include sayıları
**0 / 0 / 0 / 0**. Bu, SDK'sız linklenmiş Player kanıtı değildir.

## Kaynak sınırı

On karma dosyada Python fonksiyonları yeni Studio-only kaynaklara taşındı.
Ortak işlemler/IPC/JSON ve descriptor kayıtları solda kalır; Python aynı servisleri
sağdaki bağlama üzerinden çağırır. İki taraf birbirinin alternatifi değildir:
Studio ikisini birlikte derler; Player yalnız ortak tarafı alır.

| Ortak servis/IPC kaynağı | Studio Python kaynağı |
|---|---|
| `RtAudioBindings.cpp` | `RtPythonAudioBindings.cpp` |
| `RtFluidLabelBindings.cpp` | `RtPythonFluidLabelBindings.cpp` |
| `RtIpcLensWeather.cpp` | `RtPythonLensWeatherBindings.cpp` |
| `RtIpcObjectSubstance.cpp` | `RtPythonObjectSubstanceBindings.cpp` |
| `RtIpcPostExposure.cpp` | `RtPythonPostExposureBindings.cpp` |
| `RtIpcViewportCutout.cpp` | `RtPythonViewportCutoutBindings.cpp` |
| `RtUvBindings.cpp` | `RtPythonUvBindings.cpp` |
| `RtVolumeMaterialBindings.cpp` | `RtPythonVolumeMaterialBindings.cpp` |
| `RtRayGpuTiming.cpp` | `RtPythonRayGpuTimingBindings.cpp` |
| `RtWeatherEffectsBindings.cpp` | `RtPythonWeatherEffectsBindings.cpp` |

Lens weather request doğrulaması tek ortak invoke yordamındadır. Weather hata
dönüşümü ve UV işlem listesi de tek kaynaklıdır. Fluid label JSON sözlüğü ortak
kalır. `allow_nan=false`, boolean kontrolü, hata mesajları ve IPC enqueue yolları
korundu; yeni işlem veya bağımsız iş mantığı eklenmedi.

Lens/object-substance/weather bağlama başlıkları artık pybind SDK include etmez;
yalnız `module_` ileri bildirimi taşır. Böylece `RtIpc.cpp` transitif sızıntısı da
kalktı. `RtIpc.cpp` dispatch gövdeleri bu partide değiştirilmedi.

Beş gerçek Python sahibi de Player manifestinden dışlandı: `RtAddons.cpp`,
`RtPyScene.cpp`, `RtUi.cpp`, `RtRealtimeDenoiserBindings.cpp` ve
`RtUpscaleBindings.cpp`. Bunlar interpreter/addon UI veya yalnız Python bağlama
kodudur; genel ImGui, ses, SVGF, FSR ve sahne servisleri dışlanmadı. Yeni on dosya
mevcut `RtPython*` offline kuralına uyar ve Studio vcxproj/filters'a kayıtlıdır.

552 → 547 aday değişimi beş gerçek Python sahibinin sınıflandırılmasından gelir;
536 → 547 temiz artışı on karma kaynak ve transitif IPC başlığının temizlenmesidir.
Baseline dört kategori için sıfıra sıkılaştırıldı.

## Doğrulama

- MSVC `/Zs` **34/34 PASS**: on ortak kaynak iki dalda, on yeni Python modülü,
  `RtIpc.cpp` iki dalda, Studio `RtPython.cpp` ve `RtPyScene.cpp`.
  On bir Player include dökümünde CUDA/OptiX/Embree/Python/pybind SDK başlığı yok.
  Python include yolu Player komutundan çıkarıldı. Studio için gerçek proje
  Python 3.11/vcpkg include yolu kullanıldı.
- **32 kaynak eşitliği kontrolü PASS**: on taşınan Python gövdesi, IPC/ortak
  operasyonlar, hata/JSON yardımcıları ve UV işlem sözlüğü. Yorum ve biçimlendirme
  normalize edildi; bu çalışma canlı/nümerik test değildir.
- On yeni modül proje ve filters içinde tek `ClCompile` kaydına sahip. Include
  audit ve `git diff --check` PASS. Shader, struct alanı/yerleşimi veya IPC/script
  imzası değişmedi; shader derleme/dağıtım ihtiyacı yok.
- IPC capability audit PASS: **757 dispatch**, 39 namespace; descriptor kontrolü
  güncel: **743 metot**, 724 belgeli, 583 parametreli. Generator çıktısı değiştirilmedi.
- Dokuz kaynak sözleşme suite'inde **86/89** kontrol PASS. Taşınan Python kodunu
  okuyan fog/mixture/temperature testleri yeni dosyaya yöneltildi; operation ve
  servis assertion'ları korunur, çağrı boşlukları regex ile tolere edilir.

## Önceden var olan test hataları

Üç Python kaynak testi parti öncesi API snapshot'larıyla da aynı assertion'da
başarısız oldu; bu partide beklentileri veya ilgili üretim kodu değiştirilmedi:

| Test | Açık beklenti |
|---|---|
| GPU timing contracts | `beginRayGpuTiming(cmd, w, h)` arıyor; mevcut yol `traceW/traceH` kullanıyor. |
| Gas source provenance | `SimCache.h` içinde cache v18 arıyor; güncel cache sürümü farklı. |
| Volume emission mask | Hacim alanlarını hâlâ `params.h` içinde arıyor; ortak hacim türleri ayrı başlıkta. |

Ayrı `node scripts/check_post_v2.mjs` denetimi de dokunulmayan OptixWrapper.cpp'de
`launchOptixDisplayPost(` metnini bulamadığı için FAIL. Başarılı test diye sayılmadı;
ilgili post render kaynakları bu partide düzenlenmedi.

## Sonraki kullanıcı kabulü ve kalan link işi

Kullanıcı önceki partilerle birlikte Studio TAM build/link almalı; yeni Python
kaynakları için proje reload gerekli olabilir. Python `rt` importu, on taşınan
bağlamanın çağrıları ve karşılık gelen IPC işlemlerinde başarı/hata yanıtları;
yanlış boolean, eksik kamera/context ve NaN/Inf girdi reddi kontrol edilmeli.
Ses, UV, lens hava, object substance, post exposure, cutout, volume material,
particle labels, weather ve GPU timing canlı kabulü yapılmadı.

Player projesi henüz yok. Ortak `RtIpc.cpp` içindeki `script.run_file` ve addon
işlemleri gibi interpreter isteyen yolların hedef/capability sınırı, ayrıca
`RtApi.cpp` script çağrıları ve addon UI tick'leri gerçek runtime/link ayrımında
çözülmeli. Include=0 bu sembolleri otomatik silmez. Eski su, offline renderer ve
Embree sembol sınırları da aynı kabulün parçasıdır.

Sıradaki mimari adım `Main.cpp` (8043 satır) içinden ortak `AppRuntime` ile
`StudioShell` sınırını kurmak; pencere/cihaz/kare döngüsü/sim tick/IPC pompası ortak,
paneller/Python/addon/authoring Studio tarafında kalmalıdır. Ardından ikinci VS
hedefi ve gerçek SDK kütüphanesiz Player link testi gelir.
