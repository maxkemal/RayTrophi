# Birleşik madde — kalan yol haritası (2026-10-19)

> **Durum:** AKTİF. Kod partiler halinde ilerler. Test yalnız birkaç partiden sonra, tek bir toplu duman koşusuyla yapılır (`scripts/test/rt_matter_smoke.py`). Ayrıntılı tarih [BIRLESIK_MADDE_DOMAIN_TASARIMI.md](BIRLESIK_MADDE_DOMAIN_TASARIMI.md) dosyasında; bu dosya yalnız sırayı ve kuralı tutar.

## Çalışma kuralı (2026-10-19 kararı)

- **Kod önce, test en sonda.** P1–P4 arka arkaya yazılır. Her parti kendi
  devir satırını bırakır. Kullanıcı tek derleme alır, ardından tek bir duman
  koşusu yapılır.
- **Duman koşusu özet basar.** Test başına tek satır (`7/7`, ya da FAIL olan
  kapıların adları). Uzun çıktı diske, `%TEMP%`'e gider. Ajan yalnız FAIL
  olanın logunu açar.
- **Test sırasında ayıklama yok.** FAIL nota yazılır, düzeltmesi bir sonraki
  kod partisine girer. Tek istisna, testin kendisini koşulamaz yapan bir hata
  (2026-10-19'daki silinmiş-ad tuzağı gibi). O zaman en fazla bir prob yazılır.
- **Her yeni yetenek için en fazla bir kısa test.** 3–7 sayısal kapı, uzun
  süpürme yok. Görüntü ancak sayılar karar veremiyorsa, tek mozaik.
- H1 paketi gibi uzun süitler yalnız ilgili kod değiştiyse koşulur.

## ★ Kapanış kapsamı (2026-10-20, kullanıcı onayı) — DONDURULDU, 4/4 CANLI

Plan bu dört maddeyle kapanır. Başka hiçbir şey eklenmez. Yeni bulgular
aşağıdaki "Sonra" listesine yazılır, plana girmez.

1. ✔ YAZILDI (9. build). 10× tane testi yavaşlaması: kodda değil, uzun oturum
   birikimi (taze uygulamada 120 adım 13 s); koşucu temizliği + `--restart-app`.
2. ✔ CANLI: `fluid.run` (T0 onunla ölçüldü).
3. ✔ CANLI: donmuş domain'de tane adımı koşuyor.
4. ✔ CANLI: T0 EQUAL (gaz 1.06×, sıvı 0.97×, sonuçlar eşit). Karar 1'in kapısı geçti; T6 kullanıcı kararı.

Test kuralı: her partide en fazla 10 dakikalık hızlı duman. Uzun süitler
plan sonunda bir kez koşulur.

## Sonra (plan dışı, sırasız)

- FLIP oranı kararı (P6c; ölçüm tablosu NEXT_BUILD_CHECKS 6. build).
- İnce ızgarada sıvı hacim sıkışması (P6d; `rt_test_liquid_column_ipc.py` hazır).
- sleep_transfer strong A/B enerji kapısı sınırda.
- P9 gözenek ısısı kolu (zaman aşımı yüzünden koşmadı) ve buz çarpışmasının
  canlı kanıtı (madde 3 düzelince).
- CPU PCG son kalıntıyı raporlamıyor.
- P7 sis/yağmur render bağlantısı (bulut ajanıyla).
- DEM yatağının oturtmadan sonra 1.5 mm/kare titremesi.
- Taneler buzun içinden geçiyor (868 buz yüzü var, kum zemine iniyor); aday:
  statik collider önbelleği parmak izi buz yüzlerini görmüyor.
- `fluid.run`'dan hemen sonraki ilk `fluid.matter_models` okuması 0 kg dönüyor.
- Uzun oturumda simülasyon adımının ~10× yavaşlaması (taze uygulamada yok;
  sızıntı/birikim kökü ölçülmedi).
- Yalnız panelde kalan ayarlar için IPC denetimi (sıvı basınç iterasyonu
  böyle çıktı).

## Partiler (sıra)

| # | Parti | Neden şimdi | Durum |
|---|---|---|---|
| P0 | 2026-10-19'un dört düzeltmesi: `temperature_fresh`, `add_primitive` silinmiş ad, tane duvar filmi, readiness koruması | Yazıldı | CANLI (duman 10/10) |
| P1 | **Kapasitesiz ısı yutaklarının sökümü.** Sıvının kapalı duvar "temas" soğutması (`coolThermalLiquid`, ortam termostatı) ve MSF ortam gevşemesi (`cooling_rate`) film + kapasite modeline geçer. Termal alan "yerel ortam"a film üzerinden ısı veren bir kaynak olur. | Tane testi bu yolun %97 enerji yuttuğunu gösterdi. Sıvı ve kabuk aynı sınıf. Kural 5 gereği `cooling_rate` emekliye ayrılır. | CANLI, duman 10/10 (alan dış kaynak olarak kaldı, hızı alanın `thermal_rate_per_s`'i) |
| P2 | **Buhar hücresinde gerçek hava sıcaklığı.** Yüzey filmleri (MSF gather, tane↔hava) Tv yerine T okur: buhar envanteri bağlanır, `airKelvinFromVirtual`. | FG1'de aynı sınıf düzeltildi; kaynayan suyun üstündeki film sıcak çalışıyor. | CANLI, duman 10/10 (envanter host'taysa tane↔hava kuru varsayar) |
| P3 | **Tane domain'inde donma.** Donmuş sıvı için bir sahip (statik engel taşıyıcı) ya da donan suyun Ice tanesine dönmesi. `freeze_held_by_grains` sıfıra iner. Ice tane erimesinin canlı kanıtı aynı testte. | Tane + soğuk = bugün aşırı soğuma; "tek domain" iddiasının son termal deliği. | CANLI, duman 10/10: donmuş sıvı iki şeridin dışında tutulan buz; taneler buza çarpmıyor (açık) |
| P4 | **Tane ısı geçişinin ölçeği.** Hücre başına OpenMP; 1e5 tanede `thermal_grain_ms` hedefi < 5 ms. Gözenek suyu tane kapasitesine katılır, bütün geçişlerde tutarlı olarak. | Yoğun sahneler (H1 dense 16k) termal açılınca seri döngüye takılır. | CANLI, duman 10/10 (iki faz paralel; gözenek suyu kapasitede) |
| P5 | **`fluid_extinguishing` söküldü** (ajan kararı, 2026-10-19). Bayrakla birlikte `cooling_power` ve `oxygen_dilution` da ölüydü (FG1 eski söndürmeyi sökmüştü; buhar O2'yi fiziksel seyreltiyor). Madde alanı, domain aynaları, proje/sahne serileştirme, panel ve hash'ten çıktı; `substance.set` emekli gerekçesiyle reddeder. `fluid.set_combustion` artık yakıt olmayan sıvıyı tutuşturamaz (`enabled && flammable`). | Kural 5 | CANLI (madde 1/1, FG1 8/8) |
| P6 | **Bayat test temizliği.** 11 statik `check_matter_*` PASS (revizyon ve önbellek sürüm pinleri "en az" kuralına çevrildi, grain kilidi yokluğu, frame cache `std::move`). `g2_dry_wet` Soil'e (MPM) geçti. `sleep_transfer`: "projectile not born" kökü yatak çöküşü (COM y 0.030, beklenen ~0.075), doğum yeri dolu; test artık bu sebebi adıyla raporlar. Çöküşün regresyon olup olmadığı açık. | H1 süitini yeniden güvenilir yapmak | CANLI: statik 11/11, g2_dry_wet PASS; sleep_transfer yatak çöküşüyle FAIL (açık: kararlı yatak mı, regresyon mu) |
| P6b | **Seed oturtma.** Seed alt-kafesi tane yarıçapını ve temas yükünü bilmiyor; bu yüzden ilk karede çöküş ve çalkantı oluyor. Çözüm dinamik gevşeme (`MatterSeedSettle.inl`): alanın kendi adımları zaman, kaynak, ısı ve gaz dururken koşar, her adımdan sonra hız ve spin sıfırlanır. API: `fluid.settle`, panel ve `settle_*` okuması. sleep_transfer tek katlı yatağa ve `fluid.settle`'a geçti. | Kullanıcı gözlemi (Fill Domain çalkantısı); sleep_transfer aynı sınıfta | CANLI, kısmi: sleep_transfer 5/5; seed_settle 2/5, durma ölçütü (hız) çözücü tabanında takılıyor, sıvıda 0.2 m/s dinlenme tabanı şüpheli. Yer değiştirme ölçütü + düzlük algısı + kare başı konum okuması YAZILDI (5. build). Tank tabanı kökü: FLIP 0.90 gürültüsü (FLIP 0 → 0.004 m/s) + boyutsuz tam sayı yoğunluk düzeltmesi; çözüm kararı açık (P6c) |
| P6c | **Sıvı dinlenme tabanı (karar).** Dinlenik tankta 0.16–0.21 m/s FLIP gürültüsü. Seçenekler: suyun FLIP oranını düşürmek / FLIP oranını h ve dt'ye bağlamak / yoğunluk düzeltmesini boyutlu gevşeme süresi + yumuşak yoğunlukla yazmak. | Kullanıcının gördüğü "su durgun başlamıyor"un asıl kökü | Yoğunluk kısmı CANLI (sıvı testleri PASS). Su konumla da p95 4.35 mm/kare çalkalanıyor (FLIP payı). DEM yatağı oturtmadan sonra da 1.5 mm/kare titriyor (kök açık). Oturtma eşleşme kusuru düzeltildi (6. build). FLIP ölçümü: 0.9→0.3 hareketi 5–6× azaltıyor (1.8→0.33 mm/kare ortalama); ince ızgara azaltmıyor; FLIP 0.3'te 0.02 m/s taban kalıyor (FLIP dışı kaynak, sınır adayı). Dökülme: 0.3 tepe hızları korur, duvara tırmanma/sallanmanın %13–17'sini kaybeder, geç evrede %45 daha sakin. Karar: kullanıcı. FLIP kısmı: 5. build sonrası ölçümle (0.9/0.3 × 5/2.5 cm) |
| P6d | **Sıvı hacim sıkışması ince ızgarada.** 1.25 cm'de −15%, 1 cm'de −26/−32% (duran ve düşen levha); 5 ve 2.5 cm temiz. Baş aday basınç iterasyon tavanı 24. İterasyon/kalıntı IPC'ye açıldı + seed kesilme uyarısı/okuması YAZILDI (7. build); ölçüm `rt_test_liquid_column_ipc.py` 24/100/400. Kullanıcı hipotezi (2026-10-20): seed hücrelere eşit dağıtıyor, sim başlayınca parçacıklar fiziksel konumlarına çökerken yükseklik düşüyor; aynı test bunu iterasyon etkisinden ayırır. PARK EDİLDİ (kullanıcı kararı: ana plana devam). | Kullanıcı gözlemi; 5. build yoğunluk düzeltmesi maskeyi kaldırdı | açık |
| P9 | **Birleşik maddenin delikleri.** Taneler buza çarpıyor (donmuş hücre yüzleri statik üçgen). Ice tane erimesinin canlı testi. Cihazdaki buhar envanteri tane↔hava filmine indiriliyor. Emilen gözenek suyu ısısını taşıyor. | P8 ölçümünden önce Matter eksiksiz olmalı | CANLI kısmi: Ice erimesi PASS; buz zemini FAIL (donmuş domain'de tane adımı koşmuyor); gözenek kolu koşmadı (zaman aşımı); grain_heat ~10× yavaşladı (kök açık) |
| P7 | Sis/yağmur render bağlantısı, yağmur doğum yeri ölçümü | Bulut ajanıyla ortak karar | ertelendi |
| P8 | T0 eşitlik ölçümü, sonra uzman Gas/Liquid domain'lerinin geleceği | Plan kuralı: ölçülmeden kaldırılmaz | ertelendi |

Test turu: P0–P4 yazıldıktan sonra **tek derleme ve tek duman koşusu**. P6
araya, kod gerektirmeyen bir anda girebilir.

## Duman koşusu (en sonda)

`python scripts/test/rt_matter_smoke.py` koşuyu yapar. Varsayılan kollar hızlı
olanlardır: madde, gaz↔tane, Ek 11, yüzey suyu, FG1, tane ısısı.
`--with-h1` uzun H1 süitini de ekler.
