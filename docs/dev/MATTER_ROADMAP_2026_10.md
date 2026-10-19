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
| P6b | **Seed oturtma.** Seed alt-kafesi tane yarıçapını ve temas yükünü bilmiyor; bu yüzden ilk karede çöküş ve çalkantı oluyor. Çözüm dinamik gevşeme (`MatterSeedSettle.inl`): alanın kendi adımları zaman, kaynak, ısı ve gaz dururken koşar, her adımdan sonra hız ve spin sıfırlanır. API: `fluid.settle`, panel ve `settle_*` okuması. sleep_transfer tek katlı yatağa ve `fluid.settle`'a geçti. | Kullanıcı gözlemi (Fill Domain çalkantısı); sleep_transfer aynı sınıfta | CANLI, kısmi: sleep_transfer 5/5; seed_settle 2/5, durma ölçütü (hız) çözücü tabanında takılıyor, sıvıda 0.2 m/s dinlenme tabanı şüpheli. Sıradaki: yer değiştirme ölçütü + tank tabanı kökü |
| P7 | Sis/yağmur render bağlantısı, yağmur doğum yeri ölçümü | Bulut ajanıyla ortak karar | ertelendi |
| P8 | T0 eşitlik ölçümü, sonra uzman Gas/Liquid domain'lerinin geleceği | Plan kuralı: ölçülmeden kaldırılmaz | ertelendi |

Test turu: P0–P4 yazıldıktan sonra **tek derleme ve tek duman koşusu**. P6
araya, kod gerektirmeyen bir anda girebilir.

## Duman koşusu (en sonda)

`python scripts/test/rt_matter_smoke.py` koşuyu yapar. Varsayılan kollar hızlı
olanlardır: madde, gaz↔tane, Ek 11, yüzey suyu, FG1, tane ısısı.
`--with-h1` uzun H1 süitini de ekler.
