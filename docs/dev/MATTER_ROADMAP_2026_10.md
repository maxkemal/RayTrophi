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
| P5 | **Kullanıcı kararı: `fluid_extinguishing`.** Bayrak artık hiçbir söndürmeyi sürmüyor; ya sökülür (~15 dosya + serileştirici) ya da `fluid_flammable` ile birleşir. | Karar bekliyor | KARAR |
| P6 | **Bayat test temizliği.** Grain sözleşmeleri (REVISION 24→27), `g2_dry_wet` (Sand artık DEM), `sleep_transfer` "projectile not born" kökü. | H1 süitini yeniden güvenilir yapmak | açık |
| P7 | Sis/yağmur render bağlantısı, yağmur doğum yeri ölçümü | Bulut ajanıyla ortak karar | ertelendi |
| P8 | T0 eşitlik ölçümü, sonra uzman Gas/Liquid domain'lerinin geleceği | Plan kuralı: ölçülmeden kaldırılmaz | ertelendi |

Test turu: P0–P4 yazıldıktan sonra **tek derleme ve tek duman koşusu**. P6
araya, kod gerektirmeyen bir anda girebilir.

## Duman koşusu (en sonda)

`python scripts/test/rt_matter_smoke.py` koşuyu yapar. Varsayılan kollar hızlı
olanlardır: madde, gaz↔tane, Ek 11, yüzey suyu, FG1, tane ısısı.
`--with-h1` uzun H1 süitini de ekler.
