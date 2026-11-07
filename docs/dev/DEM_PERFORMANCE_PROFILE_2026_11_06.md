# DEM performans ölçümü — 2026-11-06

## Koşullar ve yöntem

Kullanıcı revision 36 düzeltmesinden sonra sahnenin çalıştığını doğruladı.
Açık sahne kare 52'de duraklatılmıştı: 74 439 DEM tanesi, 1770 collider yüzü,
sabit collider, kuru kum, 25 mm fiziksel yarıçap. Ayarlar değiştirilmedi;
reset/seed yapılmadı. Mevcut uyku sistemi açık (hız eşiği 0,002 m/s,
bekleme 0,2 s); contact probe ve audio telemetry kapalı.

Dış `scripts/test/rt_ipc.py` istemcisiyle altı `fluid.step(dt=1/24)` çağrısı
yapıldı. Her adımdan sonra GPU timestamp sonuçları okundu ve sıfırlandı.
Her ölçümde yeni 186 `sim_matter_grain_step` çağrısı var: cache oynatımı
ölçülmedi. GPU kernel profiling yalnız bu pencere için açılıp tekrar kapatıldı.
Bu ölçüm oturmuş/hareketi azalmış mevcut durum içindir; serbest düşüş veya
ilk çarpışmayla kontrollü A/B karşılaştırması değildir. UI/render ile GPU
paylaşımı da duvar süresi değişkenliğine katkı verebilir; izole benchmark değildir.

## Sonuçlar

| Kare | Toplam adım ms | GPU toplam ms | DEM temas/integrasyon ms | Uyuyan tane |
| --- | ---: | ---: | ---: | ---: |
| 53 | 477,02 | 435,22 | 408,12 | 91 |
| 54 | 428,88 | 391,28 | 367,98 | 111 |
| 55 | 307,02 | 266,30 | 255,43 | 116 |
| 56 | 431,97 | 391,29 | 378,72 | 219 |
| 57 | 449,82 | 411,42 | 396,61 | 192 |
| 58 | 339,37 | 302,00 | 290,28 | 217 |

- Medyan: toplam 430,43 ms, GPU 391,28 ms, temas/integrasyon 373,35 ms.
- Altı adımın GPU süresinin toplamda %95,43'ü `sim_matter_grain_step` içinde.
  Bu kernel komşu temasını, geçmiş/sürtünme/rolling işlemlerini, collider BVH
  taramasını, sleep denetimini ve integrasyonu birlikte içerir. Alt maliyetleri
  bu timestamp verisiyle ayrıca ayırmak mümkün değildir.
- Her adım 186 DEM alt adımı, 744 DEM dispatch. Sınırlayıcı `damping`,
  stiffness adaptation yok, max_substeps 512 tavanına ulaşılmıyor.
- Liste kurma GPU süresi 9,09–24,70 ms. Gerçek liste kurma sayısı 5–16;
  shader kernel'i her alt adımda gönderilse de çoğunda erken dönüyor.
  Dispatch azaltımı tek başına ana GPU yükünü ortadan kaldırmaz.
- State resident; yalnız 112 byte upload, yaklaşık 4,47 MB readback.
  Readback host süresi 1,15–1,27 ms. Host ordering yaklaşık 11,5–13,7 ms;
  aktarım/CPU ordering bu pencerede baskın maliyet değil.
- Son alt adımda yaklaşık 352 bin temas, 341 bin grain-pair, 652 bin aday,
  1,07 milyon history probe, 170 bin collider BVH node ziyareti.
  Bunlar son alt adım sayaçlarıdır, kare toplamı değildir.

## Mevcut uyku sistemi

Uyuyan pay yalnız %0,12–0,29. Ortalama taşıyıcı hızları 0,00331–0,00456 m/s;
son karede hızlı uç taneler 0,31–0,77 m/s. Bu nedenle yığını tamamen hareketsiz
kabul edip uyumayanları hata sayamayız. Ortalama hız tek başına kaç tanenin
eşiğin altında olduğunu da söylemez.

Mevcut sleep hem doğrusal hızı, spin*radius hızını, bekleme süresini hem de
kuvvet/tork dengesini denetliyor. Komşu darbesi audit/uyanma isteği verebiliyor.
Bir sonraki inceleme düşük sleep oranının hız, spin, kuvvet dengesi veya
komşu audit koşullarından hangisinden geldiğini ayırmalı. Hız eşiğini yalnız
daha yüksek sleep sayısı elde etmek için artırmak fizik doğruluğu kanıtı değildir.

## Öncelik

1. `sim_matter_grain_step` içindeki collider, history/contact ve sleep audit
   maliyetlerini ayrı ölç; mevcut uykuya giriş/uyanma nedenlerini say.
2. Daha uzun dinlenmiş yığında aynı sayaçları gözle; uykuya geçişi korunmuş
   başlangıç durumlarından kontrollü test et. Bu pencere henüz tam dinlenmiş değil.
3. Sonra temas kernel'i ve mevcut sleep davranışını optimize et. Alt adım
   sayısını azaltmak damping/stability sınırını değiştirdiği için ayrı fizik
   doğrulaması gerektirir. Yalnız max_substeps artırmak bu pencereyi hızlandırmaz.

Sahne kare 58'de duraklatılmış ve manuel script sürücüsünde bırakıldı.
Profil kapalı, fizik/uyku ayarları aynı; altı gerçek adım sonucu korunuyor.

## Revision 37 — Uyku nedenleri için opt-in sayaçlar (canlı doğrulama bekliyor)

Mevcut `contact_probe_enabled` anahtarıyla açılır; yeni fizik ayarı yok.
Python ve IPC'nin ortak `fluid.matter_models` çıktısında
`grain_diagnostics.contact_probe.sleep` altında okunur. Probe kapalıyken
bu sayaçlar için atomik işlem yapılmaz. Kuvvet/sleep eşikleri ve audit
politikasının fiziksel kararları değiştirilmedi.

Kare olayları (aynı tane birden çok alt adımda sayılır):

- `frame_contact_skips`: sleeping hızlı dönüşüyle atlanan temas çözümleri.
- `frame_contact_solves`: ağır temas yoluna giren tane-alt adım sayısı.
- `frame_periodic_audits`: sleeping iken periyodik audit zamanına girenler.
- `frame_neighbour_audits_consumed`: okunan/temizlenen komşu audit bitleri.
- `frame_new_neighbour_audit_requests`: önceden clear olan AUDIT bitini set
  eden istekler. Aynı pending bite tekrarlanan istekler ayrıca sayılmaz.

Son alt adım snapshot'ı (tane başına bir örnek, koşullar birbirini dışlamaz):

- `final_linear_speed_blocked`, `final_spin_speed_blocked`: çıkış doğrusal
  hızı veya spin*radius mevcut sleep hız sınırını karşılamıyor.
- `final_force_balance_blocked`, `final_torque_balance_blocked`: mevcut
  kuvvet/tork denge testi sağlanmıyor. Aynı iki karşılaştırma hem fiziksel
  sleep kararında hem tanılamada kullanılır; farklı tolerans hesabı yok.
- `final_coupling_blocked`: drag/lift nedeniyle sleep yasak.
- `final_fresh_history`, `final_context_reset`: history/context durumu;
  bunlar o andaki eligibility'yi mutlaka engellemez ama rest clock'un
  yeniden başlamasını açıklayabilir.
- `final_rest_eligible`: hız/dönme/denge/coupling koşullarını sağlayanlar;
  uyuyanlar da bu gruba dahildir. `final_rest_time_pending`: bunlardan
  henüz gerekli kesintisiz rest süresini doldurmayanlar.
- `final_sampled_grains`, `final_sleep_disabled`: örnek sayısı ve kapalı
  sleep politikası. Final alt adım zaten zorunlu temas auditi olduğu için
  sleeping hızlı dönüşü final snapshot'ını atlamaz.

`counts_valid=false` uint32 sayaç overflow'u demektir; sayıları oran olarak
kullanma. `final_snapshot_complete=false` son alt adımın bütün taneler için
ölçülmediğini (örneğin hata nedeniyle erken çıkıldığını) gösterir. Held
attempt'lerde kısmi sayaçlar korunur; tam kare istatistiği diye sunulmaz.
Başarılı normal adımda `skips + solves = grains * runtime.substeps` olmalı.
Bu sayaçlar ek atomik maliyet getirdiği için açık probe süreleri baseline
performansıyla doğrudan kıyaslanmamalı.

### Kullanıcı build ve doğrulaması

1. C++ + dört DEM shader'ını birlikte derle: revision **37**, ortak
   diagnostics/audio header **45 word**. Shader include bağımlılığına yeni
   `sim_matter_grain_sleep_probe.glsl` eklendi. Uygulamanın okuduğu SPIR-V
   dizini de güncel olmalı. Ajan build/shader compilation çalıştırmadı.
2. `matter_grain_params_test.cpp` testindeki schema/overflow aktarım kontrollerini
   çalıştır. Kaynak denetimleri (`check_matter_grain_contracts.py`,
   `check_matter_grain_sleep_contracts.py`, `check_matter_grain_sleep_transfer.py`)
   ajan tarafından geçti; bunlar GPU fizik doğrulaması değildir.
3. Kumlu sahnenin yığını yerleşmeye başladıktan sonra duraklat. Dış terminalden
   `python scripts/test/probe_dem_collider_ipc.py --steps 4 --require-success`.
   Script geçerli sayaç, tam final snapshot ve solves/skips korunumu da denetler;
   sonunda eski probe anahtarını geri yükler. Reset/seed yapmaz.
4. Hız/dönme ile kuvvet/tork nedenlerini ve rest-time pending sayısını karşılaştır;
   sonra mevcut sleep sistemine yönelik değişiklik seç. Eşikleri ölçmeden artırma.

## Revision 37 canlı uyku nedenleri ölçümü

Kullanıcı shader include sırası düzeltmesinden sonra uygulamayı derledi.
Açık sahnede ilk dört gerçek adım kare 23..26, ardından 32 + 16 gerçek adımla
kare 75'e kadar ilerleme ölçüldü. Reset/seed ve fizik ayarı değişikliği yapılmadı.
Kare numarası API'nin timeline zaman eşlemesidir; süre ve adım sayısı için
doğrudan `fluid.step(dt=1/24)` çağrıları kullanıldı (toplam 52 çağrı).

Tüm başarılı adımlarda sayaçlar geçerli, final snapshot tam ve
`contact_skips + contact_solves = 74439 * substeps` kontrolü geçti.
Hata/overflow gözlenmedi. Aşağıdaki nedenler birbirini dışlamaz:

| Kare | Uyuyan | Hız engeli | Spin engeli | Kuvvet dengesi engeli | Tork dengesi engeli | Rest süresi bekleyen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | 0 | 74345 | 74347 | 74359 | 65580 | 77 |
| 26 | 1 | 74320 | 74282 | 74308 | 66239 | 88 |
| 39 | 14 | 69594 | 48855 | 62084 | 30874 | 2248 |
| 47 | 19 | 61416 | 14682 | 33487 | 7392 | 10418 |
| 59 | 320 | 37197 | 2596 | 14867 | 1445 | 33179 |
| 67 | 952 | 33266 | 1120 | 6797 | 301 | 37733 |
| 71 | 2586 | 17699 | 747 | 6908 | 240 | 50463 |
| 75 | 5020 | 27218 | 689 | 3554 | 133 | 40231 |

23/26 sleeping değeri `final_rest_eligible - final_rest_time_pending` ile
rest hedefini dolduran tane sayısıdır; son alt adımda yeni uyuyanlar nedeniyle
runtime `sleeping_grains` ile birebir aynı olmayabilir. Diğer tablodaki sleeping
değerleri runtime sayacıdır. Snapshot'ta diğer anlık kriterler sabit tutulmaz;
bir tanenin her alt adımda aynı koşulu sağlayıp sağlamadığı ayrıca ölçülmedi.

- Context reset, fresh history ve coupling blocked sayaçları her snapshot'ta 0.
  Gizli bir context/history resetinin uykuyu engellediğine dair veri yok.
- Erken 23..25 karelerinde bütün yaklaşık 14,89 milyon tane-alt adım teması
  hesaplandı, sleep skip/audit sayıları 0. Bu aşama hâlâ aktif hareket.
- Kare 75: uyuyan 5020/74439 (%6,74), ortalama hız 0,0018335 m/s.
  27 218 tane hâlâ 0,002 m/s doğrusal hız eşiğini aşmış; yalnız 689 tane
  spin, 133 tane tork dengesi koşulunu sağlayamamış.
- Aynı karede o anda uygun 45 262 tanenin 40 231'i 0,2 saniye rest süresini
  tamamlayamamış. Bu, sayacın arızalı olduğunu kanıtlamaz: eligibility
  anlık, fizik politikası ise kesintisiz uygunluk gerektiriyor.
- Kare 75: 1 032 563 temas hesabı atlandı, 12 813 091 hesaplandı. Skip payı
  %7,46; son andaki sleeping oranından farklı bir kare toplamı ölçümüdür.
  18 980 periyodik sleeping audit, 21 100 tüketilen komşu audit ve 24 560
  yeni AUDIT bit isteği var. Bunlar farklı olaylar/tekrarlar; toplamları
  tane sayısı diye yorumlanmaz. Erken pencerede ana engel komşu audit değildi.

Sonuç: mevcut uyku sistemi çalışıyor fakat bu yığında yavaş devreye giriyor.
İlk optimizasyon hipotezi, kalan düşük doğrusal hareket/temas salınımı ve
kesintisiz eligibility koşuludur. Hareketin fiziksel yeniden yerleşme mi yoksa
sayısal salınım mı olduğu bu sayaçlarla tek başına ayrılmadı. Eşikleri veya
rest politikasını hemen gevşetmek yerine aynı dinlenmiş başlangıçtan kontrollü
karşılaştırma gerekir. Collider/history alt maliyetleri hâlâ ayrıca ayrıştırılmadı.

Probe açık olduğu için bu pencerenin adım süreleri önceki probe-kapalı baseline
ile hız kazanımı olarak kıyaslanmadı. Son durumda sahne kare 75'te duraklatıldı;
probe anahtarı önceki kapalı değerine geri yüklendi, fizik ayarları korundu.
