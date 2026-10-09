# DEM uyku aktarımı — revizyon 24 (kaynak hazır, canlı kabul bekliyor)

2026-10-09. Kullanıcı zayıf üst katman hareketinin sıkışmış çekirdeği uyanık
bırakmasını düzeltmeyi istedi. Render geçişi değişiklikleri kullanıcı isteğiyle
geri alınmış durumda; bu faz render yaşam döngüsünü değiştirmiyor.

## Uygulanan E1

1. `g_dynamic_contact` göreli hız veto'su kaldırıldı. Gerçek net kuvvet/tork dengesi
   ayrı `force_balanced`; düşük hızlı destek bildirimi sadece gerçek denge testi
   başarısızsa komşuya gider. Hareketli temas ikinci halkayı kendiliğinden tetiklemez.
2. Komşu bit'i **AUDIT**: hedef kendi tam temas/duvar/collider/köprü çözümünü çalıştırır.
   Dengeli ve yavaş hedef dinlenme yaşını korur; bildirimin kendisi sayacı sıfırlamaz.
   Desteksiz düşüş, host destek kaybı ve dış kuvvet korumaları korunur.
3. Yeni `sim_matter_grain_sleep_transfer.glsl`: sönüm ve Coulomb/rolling/twist sonrası
   reaksiyon değişiminin itki ve açısal itkisi hedef kütle/ataletine uygulanır.
   Statik elastik ön yük değişmemişse yeni aktarım sayılmaz. Karar mevcut doğrusal
   ve yüzey açısal hız deadband'inin per-DOF kinetik eşdeğeridir; eşikler değişmedi.
   Kaynağın kendi hız eşiği filtre değil: yavaş ağır tane hafif hedefi etkileyebilir.
4. Güçlü yeni teması denetim aralığı boyunca kaçırmamak için finite-mass normal
   çarpışma üst sınırı ve Coulomb/rolling/twist limitli öngörü kullanılır. Bu
   koruyucu üst sınır gerçek sönümlü enerji ölçümü değildir; sadece denetim talebidir.
   Kararı hedefin tam net kuvvet/tork çözümü verir. Islak köprü bildirimi korunur.
5. Dinlenme **fiziksel saniye** olarak ilerler. CFL dt, sleep context'ten çıkarıldı;
   yeni hafif tane/adaptif dt tüm yığının yaşını silmez. Uzun süre + çok küçük dt
   için kompansasyonlu float toplama kullanılır. Gravity/kuvvet yasası/domain/uyku
   ayarı/collider/support değişiklikleri hâlâ uyandırır. Allocation veya gerçek
   contact-budget retry tarihçeyi hâlâ sıfırlayabilir; bu nedenler kaldırılmadı.
6. Owner-only ek metadata sözcüğü: uykuda atlanan solve süresi, uyanık adayda saat
   yuvarlama kompansasyonu. Persist eden yaylar audit'te geçen süreyle ilerler;
   yeni temaslar tek dt alır. Bu, sabit göreli hızın integralini düzeltir. Değişken
   hareket için endpoint hızla catch-up hâlâ bir yaklaşımdır; tam trajectory replay
   veya destek sertifikası değildir. Yavaş yük/tekrarlı darbe kabulü ayrıca gerekli.

## ABI / maliyet

17 SSBO / 128 B push / `4*substeps` aynı. History record 7 word x 24 = 672 B/tane
aynı. Metadata map + owner/mask/rest/clock: 4 → **5 uint/capacity**, +4 B/ayrılmış
slot (capacity=2^21 için +8 MiB). Host working-byte ledger ve gerçek allocation
birlikte güncellendi. Shader descriptor length guard'ı eski host metadata'sını
indekslemeden reddeder; revision=24 host/shader birlikte deploy edilmeli.
Yeni copy süresi sözcüğü her skipped step'te owner tarafından yazılır; etkisi canlı
native GPU ölçümünde doğrulanacak. Daha yüksek uyku oranı tek başına kabul değil.

Yeni authored parametre yok. Mevcut `sleep`, `sleep_speed_m_s`, `sleep_time_s`
UI/Python/IPC aynı param/core yolunda; UI yardım metni yeni davranışı anlatır.

## Doğrulama ve kullanıcı sırası

Ajan build, shader compile veya uygulama açılışı yapmadı. Source/spec kontrolleri
compiler/GPU kabulünün yerine geçmez.

Üç grain/sleep/transfer source-spec kontrolü PASS; geniş denetim 8/10 PASS.
İki eski transfer/pore kontrolünün literal copy assertion'ları ortak sütun
iterator refactor'unu tanımıyor. Alanlar kaynakta mevcut; kontrol güncellemesi
ve kalan assertion'ların yeniden koşulması S1b devir notunda kayıtlı.

1. Kullanıcı eşleşen C++ ve dört grain SPV'yi derleyip/deploy edip uygulamayı açar.
   `compile_sim_shaders.bat`'taki mevcut dört grain wrapper yeni ortak include'u
   otomatik içerir. Scalar/pressure kernel ABI'lerine bu faz dokunmadı.
2. Boş, durmuş sahne; aktif collider/force-field yok. Dış terminalden:
   `python scripts/test/rt_h1_grain_suite.py --only sleep sleep_transfer`
   `sleep_transfer`: 27 taneli küçük bed + önceden planlanmış tek projectile;
   yoğunluk oranı 1/1000, hız .1 m/s zayıf kol; eşit yoğunluk, 1 m/s güçlü kol.
   Açık/kapalı dört koşu. Birth sırasında authoring edit/reset yok. Weak core en az
   23/27 sleeping, güçlü darbe en az bir hedefi uyandırmalı. Mass ratio/kütle,
   population, COM (1 mm) ve iki yönlü enerji (1e-5 J + %25) kapıları var.
   Yeni fixture **canlı koşulmadı**; ilk kabulde fixture gate hatası ile model
   hatası ayrılmalı. Kendi source/domain/material'ini temizler, frame0 paused bırakır.
   Log `docs/dev/grain_sleep_transfer_live.json`.
3. `--only history static settle wet motion coexist` regresyonları; destek kaybı ve
   yavaş yük/köprü değişimi. Mevcut kısa probun agresif unsupported fall kapısı kalır.
4. Dört kollu repose A/B ve 1.3M 180-adım A/B tekrar. Rev23 referansı
   `grain_sleep_ab_rev23/REPORT.md`: dört açı paritesi PASS ama açık size 4.04849°
   > 4° FAIL, wall 5.75 / 6.47 s, sleeping %37.7. Bu eski ölçümler rev24 kazancı değil.

## Sonraki E2 — aktif hesap alanı

E1 fizik kapısı geçmeden başlamaz. Aktif grain listesi + denetim cephesiyle gerçek
contact işinin ve mümkünse dispatch alanının daraltılması; destek kaybı, köprü,
komşu-listesi kapsamı, CFL, canonical SoA ownership ve sleeping audit korunacak.
Bütün yığını tek ada yapıp üst katmanla birlikte uyandırmak hedef değil.
Bu faz **henüz uygulanmadı**; SIMD copy/metadata maliyeti E1'de devam ediyor.

Diğer ajan için bağımsız iş: [HANDOFF_S1B_CELL_FIELDS.md](HANDOFF_S1B_CELL_FIELDS.md).
