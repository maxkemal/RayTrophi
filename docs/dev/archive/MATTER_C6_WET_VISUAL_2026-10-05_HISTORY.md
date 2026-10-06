# Matter C5/C6 — kısa devir, 2026-10-05

## Durum
- Kullanıcı derledi; açık boş sahne dış IPC ile kuruldu. Codex build/launch yapmadı.
- C5 yoğunluk + drenaj batching kullanıcı derlemesiyle canlı ölçüldü:
  kaynaklar durduktan sonra 180 scheduler adımında su 44.865001821 ->
  44.865001111 kg; fark -0.71 mg. Kuru Sand 84.000001252 kg sabit.
  Drenaj, mixed GPU contact ve identity/transfer probları PASS.
- C6 ilk canlı kabul: 25 + 180 manuel scheduler adımı, dt=1/60; ikinci
  pencerede su 44.865001631 -> 44.865001949 kg (+0.32 mg), kuru Sand
  84.000001252 kg sabit. Wet/pore drainage/mixed contact probları PASS.
  205 adımda 1269 parçacık; son adım 2 birth + 124 refill, 365 wet carrier.
  Pore pressure max 135.785 Pa, capillary cohesion max 233.769 Pa.
  Güncel kontroller: GPU/pore/transfer/stages/wet source audits + batch math PASS.
  C++ regression kaynakları yazıldı; çalıştırılmadı. Commit/push yapılmadı.

## Toplu derlemedeki değişiklikler
- Kuru yoğunlukla ilk kütle; kesin transfer kütleleri korunur. Drenaj hücre
  bazında birleşir/refill yapar; slot/alıcı yokken su gözenekte kalır.
- C6: saturation = pore_mass/capacity. Sürtünme/dilatancy azalır; capillary
  cohesion/tensile 4*S*(1-S)*peak; effective compression=max(p-u,0).
  u=scale*rho_water*g*h*S^2: ampirik hücre-head yaklaşımıdır, pressure PDE değil.
- Transport kütlesi dry+pore; gerilme kuvveti yalnız canonical dry mass/density
  hacmini kullanır. Yeni wet-response ve dry-volume GPU buffer'ları var.
- Aynı canonical saturation, granular splat görünümünde 8 kuru/ıslak materyal
  bandını seçer. Türetilen materyaller kuru scene material'dan kopyalanır;
  original değişmez, registry generation/save-load sonrası yeniden türetilir.
  Uniform raster sphere shortcut wet görünümde bypass edilir.
- Pore Water UI + mevcut rt.fluid.set_pore_exchange / IPC ortak servisinde
  wet_response_enabled, wet_appearance_enabled ve 6 coefficient eklendi.
  Transactional validation, serializer ve cache fingerprint bağlı; defaults kapalı.
- Matter Output: Remove output binding düğmesi mevcut ortak API'yi çağırır.
  Eski C4 küçük harf water binding canlı API ile kaldırıldı.

## Açık sahne
C5_WaterSand_Acceptance: kanonik Sand/Water, finite emitterler, sağda kuru kum
kontrolü. Kum sıcak renk, Water mavi Splat, nötr zemin, yakın kamera. Zemin
collider değil; Closed domain fizik sınırıdır. Sahne kullanıcı tarafından kaydedilir.
C6 flags açık, türetilen 7 wet materyal mevcut; Output yalnız Sand/Water.
Sağ kontrol başlangıçta kuru; uzun koşuda yayılan su buraya da ulaşır.
Ölçüm pencereleri frame 0'da paused idi. Sonraki kullanıcı timeline hareketi
frame 250'ye geçti; bu durum ayrı, kapalı 180-adım bilançosuna dahil değil.

## Son hızlı ölçüm
Yeni rapor canlı PASS. Yüklü sahnede wet flags kapalıydı; açılıp ayrı reset
koşusunda 25 + 60 manuel adım ölçüldü. Su 44.865001476 -> 44.865000735 kg
(-0.74 mg); en büyük örnek sapma 0.86 mg. Kuru Sand 84.000001252 kg sabit;
360 wet + 60 tam kuru carrier. Snapshot raporu PASS; gerçek save/load/cache
restore karşılaştırması yapılmadı. Sahne paused frame 0, wet flags açık bırakıldı.
Ham kayıt: matter_c6_quick_wet_2026-10-05.json.

## Kullanıcı build / kabul
Son kaynak partisi C++ derlemesini bekler, shader aynı: UI/Python/IPC ortak
wet_appearance_full_saturation [0.001,1], default .05. Görünüm A=clamp(S/.05)
ile 8 bant seçer; S=%4.75 artık band 7/full wet, %1 band 2. Fizik hâlâ gerçek S.
Serializer/cache hash ve spatial rapor bağlı; policy v2, eski snapshot yeniden alınır.
Kullanıcı Sand1'i elle koyulaştırınca doğru RT sonucu gördü: sorun transfer değil,
full-pore doluluğuna bağlı zayıf görünüm tepkisiydi. C++ regression kaynakları
hassasiyet, pure fizik, invalid patch/roundtrip/hash ve band parity'yi kapsar.
RT düşük-S düzeltmesi kullanıcı derlemesinde doğrulandı; shader aynı. Kullanıcının
Physics Domain 1 sahnesinde Smax=%4.75, 2292 wet carrier fakat round(7*S) ile
8333 kumun tümü kuru band 0'da kaldı. Ortak band seçimi ceil(7*S) oldu;
S=0 kuru, S>0 ilk wet band. Renderer ve spatial rapor aynı helper'ı kullanır.
Yeni binary: 2924 wet band 1 + 6909 dry band 0; Smax=%5.10. RT'de yalnız
türetilmiş band 1 geçici magenta yapıldı: temas çevresindeki ayrı wet küreler
magenta, kuru yığın aynı kaldı. Renk finally ile geri yüklendi, control sabit.
Per-sphere RT routing PASS; normal ilk band yalnız %6.43 koyulaşır ve wet
kürelerin çoğu alttadır. Görsel: matter_wet_rt_magenta_probe.jpg (diagnostic).
Üç C++ regression kaynağı düşük-S hatasını kapsar; C++ test target çalıştırılmadı.
Yeni küçük kaynak partisi: canonical granular shape/8-band spatial ölçümleri
`fluid.matter_models` / Python raporuna eklendi. C++ regression ve read-only
snapshot/cache karşılaştırma probu hazır. Kullanıcı derledi; canlı canonical
shape/band ve wet probları PASS. Shader aynı.
Fixture --dt/--voxel/--water-limit/--balance destekler; timeline değişimini reddeder.
Kesin kalan kapılar ve sonraki profile/emitter/domain/UI veri sahipliği:
[MATTER_ACCEPTANCE_AND_AUTHORING.md](MATTER_ACCEPTANCE_AND_AUTHORING.md).
1. Canlı ilk parti geçti. Scene kullanıcı tarafından kaydedilecek.
2. Fixture artık boş sahnede materyal/zemin/ışık/kamera kurar. Tekrar:
   python scripts/test/rt_c5_acceptance_scene_ipc.py --setup --wet --steps 25
   --log docs/dev/matter_c6_acceptance_live_2026-10-05.json; sonra --steps 180
   aynı --log ile. --setup sahneyi değiştirir/reset eder.
3. Ayrı reset koşularında kuru/nemli/doygun çökme ve angle-of-repose;
   spatial wet görünüm/dry control; iki dt/çözünürlük; save/load+bake/scrub.
4. C6 tam kabul, tam pore-pressure/drag/buoyancy, uzun havuz-dolum ve C7 açık.

## Referanslar
- Model/ayar/test ayrıntısı: MATTER_WET_RESPONSE.md.
- Ham ölçüm: matter_c5_acceptance_live_2026-10-05.json (reset koşuları ayrıdır).
- Yeni C6 ölçümü: matter_c6_acceptance_live_2026-10-05.json; görsel:
  c6_acceptance_visual.jpg. Sonraki salt okunur problar frame 250'de çalıştı.
- Geçmiş devir: archive/MATTER_C5_HANDOFF_2026-10-05_HISTORY.md.
- C++ test kaynakları: matter_physical_mass_test.cpp, matter_wet_response_test.cpp,
  matter_wet_palette_test.cpp.
