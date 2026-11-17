# Player boya başlangıcı — canlı IPC incelemesi, 2026-11-15

Kullanıcı açık Player'da aynı sahneyi çalıştırıp ilk takılmayı araştırmayı istedi.
Uygulama başlatılmadı, C++ build yapılmadı, ürün kodu değiştirilmedi.
Ölçüm dış süreçte `scripts/test/rt_ipc.py` ile yapıldı; sandbox hata 5 sonrası
named-pipe erişimi yükseltilmiş yürütmeyle sağlandı.

## Sonuç ve sınır

Bu fixture'da **SDF pişirme yok**: dört sistemin her birinde üç `ObjectAABB` ve
bir `PlaneY` çarpıştırıcı bulunuyor. Patlama fixture'ı değil, dört sürekli sprey
çalıştırıldı. Başka bir sahnedeki patlama için SDF sonucu genellenemez.

**Doğrulanan boya takılması, oyun sırasında yapılan senkron hazırlık ve her yüzey
eklemesinde bütün atlasın yeniden kurulmasıdır.** Tek bir >1 saniyelik işlem bu
koşuda doğrulanmadı. İlk proje açma RPC'si 532.12 ms sürdü. İlk display penceresi
1105.55 ms maksimum present aralığı bildirdi, fakat pencere sahne öncesi boş/idle
süreyi de içeriyor; bunu tek başına işlem takılması olarak kullanmak yanlış olur.
İlk render örneğinde backend 96.00 ms; CPU kayıt 6.30 ms, upload 1.67 ms idi.
Bunlar ayrı zaman aralıkları/örtüşen ölçülerdir; toplanarak kesin yükleme süresi üretilemez.

## Temiz çözünürlük karşılaştırması

Her çözünürlükten önce özgün proje yeniden açıldı. Dört yüzeyin her birinde gerçek
`valid_texels` kontrolü yapıldı: 512 için 262144, 1024 için 1048576.
RPC süreleri ana iş parçacığına ulaşma beklemesini de kapsar; saf atlas bake süresi değildir.
İlk tick değerleri hemen sonraki durum sorgusunun gördüğü örneklerdir; tüm karelerin
tepe değerleri yakalandığı iddia edilmez.

| İşlem | 512² | 1024² |
|---|---:|---:|
| Yüzey ekleme RPC, zemin | 16.04 ms | 60.57 ms |
| Yüzey ekleme RPC, arka duvar | 19.45 ms | 63.31 ms |
| Yüzey ekleme RPC, sol duvar | 20.13 ms | 63.96 ms |
| Yüzey ekleme RPC, sağ duvar | 20.04 ms | 62.70 ms |
| Ekleme sonrası tick, 1 yüzey | 7.59 ms | 19.06 ms |
| Ekleme sonrası tick, 2 yüzey | 13.95 ms | 39.14 ms |
| Ekleme sonrası tick, 3 yüzey | 18.86 ms | 60.86 ms |
| Ekleme sonrası tick, 4 yüzey | 22.53 ms | 72.12 ms |
| Yerleşmiş durumdan tek tick örneği | 0.60 ms | 1.41 ms |

`WetPaintGpu::addSurface` UV→konum/normal rasterizasyonunu, gutter ve tile hazırlığını
senkron yapıyor. `layout_dirty_` sonraki `tick` içinde `rebuildDevice` çağırıyor.
Bu fonksiyon bütün eski tamponları bırakıp tüm yüzeyleri yeniden ayırıyor, yüklüyor
ve boya durumunu sıfırlıyor. Böylece dört ayrı ekleme 1+2+3+4 yüzeylik hazırlık yapıyor.
1024'te texel başına sekiz GPU alanı toplam 80 bayt: dört yüzey için yaklaşık
320 MiB, art arda dört yeniden kurulumda yaklaşık 800 MiB texel alanı hazırlığı.
Tile/istatistik tamponları ve sürücü/staging ek maliyeti bu hesaba dahil değil.
Bu, yalnız performans sorunu da değil: yeni yüzey eklenince mevcut boya durumu sıfırlanıyor.

## Partikülsüz kontrol ve pipeline adayı

Aynı odanın geçici, yalnız partikül sistemleri çıkarılmış kopyasıyla A/B yapıldı.
Geometri ve sidecar dosyaları aynı tutuldu; geçici kopyalar ölçüm sonunda silindi.
Sıcak proje açma RPC'leri: partikülsüz 296.85 / 280.11 ms; partiküllü
340.57 / 336.71 ms. Bu toplam açma süreleri tek bir alt sisteme atfedilemez.

Kaynakta önemli bir ilk kullanım maliyeti bulundu: `scene_data.h` partiküller
Vulkan istediğinde `createVulkanSimulationComputeBackend` çağırıyor. Backend
kurucusu `loadPipelines()` ile kayıtlı **273 compute kernel** için pipeline
oluşturmayı deniyor; yalnız partikül/boya kernel'leriyle sınırlamıyor.
`vkCreateComputePipelines` uygulama `VkPipelineCache` nesnesi olmadan çağrılıyor
(sürücünün kendi önbelleği yine bulunabilir). Bu, ilk efekt anındaki takılma için
güçlü kaynak adayıdır; mevcut exe'de kernel başına süre yok, katkısı canlı ölçümle
ayrı olarak kanıtlanmış değildir. İlk render pipeline hazırlığı da bu koşuda ayrıştırılmadı.

## Ölçüm tuzağı

`paint.wet_clear_surfaces` materyaldeki boya dokularını silmiyor.
512 dokusu varken yeniden 1024 istemek metadata'da 1024 görünmesine rağmen
262144 geçerli texel ve 256 tile döndü. Temiz proje açılınca gerçek 1024
1048576 texel ve 1024 tile döndü. Bundan sonraki karşılaştırmalar resolution
alanına tek başına güvenmemeli; gerçek texel/tile sayısını doğrulamalı.
Önceki ajanın sayıları bu bulgu nedeniyle otomatik olarak geçersiz sayılmadı;
o koşunun gerçek texel sayıları ayrıca kontrol edilmeli.

## Düzeltme yönü

1. Sabit odanın yüzeylerini tek hazırlık işlemiyle yükleme aşamasında kurmak;
   oynanış sırasında tekrar atlas kurmamak, mevcut boya durumunu korumak.
2. Pipeline hazırlığını yükleme aşamasında ölçmek ve gerekli kernel'lerle sınırlamak;
   ilk efektte derlemeye bırakmamak. Kesin dağılım için backend init/loadPipelines,
   UV bake, tampon ayırma/yükleme ve ilk render süreleri ayrı kaydedilmeli.
3. Sonra sürekli kare maliyetindeki istatistik beklemesi ve ayrı texture kopyaları.

Tekrar üretim: açık Player ile `python scripts/test/rt_player_paint_startup_probe.py`.
Betik sahneyi değiştirir, sonunda özgün odayı gerçek 1024 boya yüzeyleriyle bırakır.
Kısa ham sonuçlar `PLAYER_PAINT_STARTUP_PROBE_2026_11_15.json` içinde; ayrıntılı
geçici RPC kayıtları `docs/dev/.captures/player-paintroom/` altında tutuldu.
Ekran görüntüsü alınmadı. Bu inceleme, optimizasyonun uygulanması veya soğuk
proses patlama testinin kabulü anlamına gelmez.
