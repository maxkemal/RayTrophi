# Derleme sonrası aynı sahne: buhar evresi / RAM incelemesi

2026-10-10. Açık uygulamaya dış `scripts/test/rt_ipc.py` ile bağlanıldı.
Uygulama başlatılmadı, build yapılmadı, authored ayarlar değiştirilmedi.
İlk sekiz fluid.step sadece solver probe'uydu; timeline/cache ilerlemedi.
Ardından timeline kare 1..40 ölçüldü. Kullanıcı sahneyi 54. kareye ilerletip
buhar oluştuğunu bildirdi; son ölçüm 54. karede durmuş sahnede alındı.
Sahne ölçüm sonunda 54. karede bırakıldı; reset/cache clear yapılmadı.

## Buhar oluşmuş evre: kare 54

128³, 23.833 sıvı parçacığı, Water, Vulkan, closed, thermal açık.

| Ölçüm | Değer |
|---|---:|
| Liquid solver total | 116,21 ms |
| Gas solver total | 102,57 ms |
| İki solver toplamı; cache/render hariç | 218,78 ms |
| Gas pressure host stage | 34,14 ms |
| Gas scalar advection host stage | 14,43 ms |
| Gas tail host sync | 7,33 ms |
| Gas voxelization | 18,29 ms |
| Surface heat/evaporation host stage | 1,54 ms |
| Inventory advection enqueue stage | 0,0389 ms |
| Inventory fizik full-grid upload/download | 0 / 0 byte |
| Liquid stage upload/download | 89.236.560 / 86.287.348 byte |
| Inventory snapshot cumulative | 905.969.664 byte = 54 × 16 MiB |
| Process private memory | 8.748.732.416 byte |
| Process working set | 7.256.944.640 byte |
| Cache raporlanan RAM | 2.814.595.353 byte, 55 kare |
| Driver VRAM usage | 1.371.709.440 byte |

Yeni resident yol çalışıyor; eski CPU kg/J advection artık bu ölçümde
çalışmıyor. 0,0389 ms **GPU kernel süresi değildir**, submit/enqueue ölçüsüdür.
Timestamp probe kare 14..16'da toplam GPU compute 24,7..24,9 ms;
gas SOR 80 dispatch yaklaşık 16,3 ms. Inventory Semi + Mac + conservation
kernel toplamı yaklaşık 1,86 ms (bu değer kare 54 ölçümü değildir).

Eski 713/823 ms ölçümü kare 118'de ve daha çok parçacıkla alınmıştı. Yeni
kare 54 ile eş kare/eş durum hızlanma oranı hesaplanamaz. Solver toplamı
218,78 ms de viewport FPS veya uçtan uca kare süresi değildir.
`timeline.set_frame` bazen yalnız scrub isteğinin yanıtını verir; 17..19 ms
RPC yanıtı gerçek simülasyon süresi olarak kullanılmadı.

## RAM büyümesi ve kalan aktarım

Kare 1..40 process private memory artışı ortalama **104.796.580 byte/kare**
(~99,94 MiB). Cache sayacı ortalama **51.655.306 byte/kare** (~49,26 MiB).
Bu, VRAM taşmasından farklı bir RAM/kare snapshot problemidir; VRAM ölçümleri
1,37 GB civarında kaldı. Process memory farkının tümü otomatik olarak cache
sızıntısı sayılmaz; kaynak incelemesi aşağıdaki somut eksikleri doğruladı:

- `SceneData::compressGridDomainStates` bütün State'i `c.meta = st` ile kopyalıyor.
  Gaz `meta.grid` bulk scalar/velocity alanları sonradan temizleniyor.
- Matter'ın ikinci `matter_liquid_grid` alanı metadata içinde tam olarak kalıyor;
  aynı işlemin bulk-array temizliği bu ikinci gride uygulanmıyor.
- `SimFrameCacheMemory::metaAllocationBytes` yalnız `state.grid` ve parçacıkları
  sayıyor; `matter_liquid_grid` hesaba girmiyor. Bu yüzden cache bütçesi gerçek
  Matter snapshot RAM'ini eksik görüyor. Process/cache farkının tamamının sayısal
  dağılımı henüz alan bazında ölçülmedi.
- Kg/J snapshot kare başına 16 MiB indiriliyor ve cache metadata'sında kayıpsız
  float array olarak tutuluyor. Fizik roundtrip 0 olsa da otomatik timeline cache
  bu readback'i her kare yeniden yapıyor.
- Liquid aşamasında ~175,5 MB/adım çift yönlü IO devam ediyor. Bu sayacın alan
  bazında attribution'ı henüz yok; tamamı termal transferdir veya tamamı cache'dir
  denemez. Yeni inventory çalışması bu eski particle/grid tüketicilerini kaldırmadı.

Bu nedenle kg/J CPU advection'ı taşımak genel maliyeti/belleği tek başına çözmedi.
Önce Matter cache ikinci grid saklama/yeniden kurma sözleşmesi ve doğru byte
hesabı; ardından otomatik snapshot cadence/residency ve liquid IO tüketicileri
hedeflenmeli. Cache'i silmek veya fizik envanterini silmek çözüm olarak uygulanmadı.

## Fizik kabulünü durduran tutarsızlık

Kare 54 sabitken iki kez doğrulanan veriler:

- Son adım vaporization ledger: **0,128675996 kg** Water aktarıldı;
  source/target mass ve latent eşit.
- API gas physical inventory toplamı: **0,000934690 kg**, **3570,296 J**.
- Gas tracer görünür: 35.930 aktif density hücresi, max density 0,979.
- `thermal_evaporated_kg` 0 raporluyor, fakat ledger'da vaporization var.

Closed domain için tek adımın aktardığı kütleden daha küçük toplam envanter
ciddi tutarsızlıktır. Henüz kaybın deposit/advection/cache/API reduction yollarından
hangisinde olduğu doğrulanmadı. Ledger dengesi tek başına grid kütle korunumunu
kanıtlamaz. Bu yeni GPU yolunun fizik kabulü **PASS değildir**.
Surface helper'ın sıvı `thermal_stats.evaporated_kg` alanını kendi sonucu ile
ataması boiling telemetry'sini ezme adayıdır; ayrı source/live audit gerekir.

Kullanıcı havada buhar oluşmadığını, yere temas sonrası oluştuğunu bildirdi.
Temas öncesinde parçacıklar yaklaşık 373,15 K idi; 32. karede son-adım evaporation
0 idi. 54. karede contact_cooled_particles=1964 ve vaporization ledger oluştu.
Bu gözlem ayrı bir termal/faz kontrolü olarak açık; yalnız bu ölçümle fiziksel
nedeni veya hatanın tüm kapsamı doğrulanmış sayılmaz.

## Veriler

- `gas_residency_probe_2026_10_10.json`: doğrudan solver, cache ilerlemeyen probe.
- `gas_residency_timeline_probe_2026_10_10.json`: timeline 1..40, process/cache slope,
  ayrı 13..16 GPU timestamp örnekleri. Sonradan eklenen observed_step_and_queries_ms
  IPC/sorgu beklemelerini de içerir; render FPS değildir.
- `gas_residency_vapor_probe_2026_10_10.json`: kullanıcı buhar evresi, sabit kare 54,
  tekrar toplam/ledger okuması ve process memory.

Yeni source değişikliği veya derleme yapılmadı; yalnız harici ölçüm aracı ve rapor
oluşturuldu. Daha uzun tek parça ilerlemeler otomatik risk denetiminde reddedildi;
RAM headroom kontrolüyle sekiz karelik probe kabul edildi. Son buhar evresine
kullanıcı ilerletti; bu nedenle ölçüm için ek ilerleme gerekmedi.
