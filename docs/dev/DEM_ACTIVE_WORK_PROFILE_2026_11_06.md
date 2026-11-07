# Revision 38 aktif iş listesi — canlı ilk ölçüm

2026-11-06. Kullanıcı C++/shader eşleşmesini düzeltti ve sahne çalıştı.
Bu ölçüm dış IPC ile yapıldı; reset/seed/cache clear yok. Başlangıç kare 0,
74 439 kuru Sand DEM tanesi, radius 0,025 m; compact_sleep_work=true,
contact probe kapalı. Aktif hesap yüzünden GPU elapsed/duvar süreleri ölçüldü,
cache oynatımı kullanılmadı.

## Yerleşme

97 gerçek adımda %80 uyku hedefi geçildi. Örnekler:

| Kare | Uyuyan tane | Mod | DEM alt adımı | Grain host toplam ms, profil kapalı |
| --- | ---: | --- | ---: | ---: |
| 1 | 0 | dense | 174 | 77,14 |
| 16 | 0 | dense | 248 | 257,52 |
| 48 | 26 | dense | 200 | 213,97 |
| 64 | 1080 | dense | 200 | 204,88 |
| 80 | 5979 | dense | 200 | 201,92 |
| 96 | 57819 | compact | 200 | 111,42 |
| 97 | 60839 | compact | 200 | 107,57 |

Bu satırlar farklı fiziksel durumlar olduğundan tek başına bir hızlanma
karşılaştırması değildir. Hiçbir adım held/nonfinite olmadı.

## GPU timestamp incelemesi

İkişer dense/compact blok, her biri 3 gerçek adım. Son bloklar %99 civarı
uykuda, her kare 200 alt adım ve bir gerçek Verlet liste kurma ile çalıştı.

| Medyan, son 3'lük blok | Dense, 104..106 | Compact, 107..109 |
| --- | ---: | ---: |
| Toplam adım ms | 290,96 | 304,48 |
| GPU toplam ms | 251,90 | 265,96 |
| Heavy step kernel ms | 224,27 | 119,52 |
| Active classifier ms | yok | 115,64 |
| DEM dispatch/frame | 800 | 1000 |

Heavy step yaklaşık %46,7 ucuzladı. Ancak classifier kazancın çoğunu tüketti;
bu profilli kısa blokta tam adım kazanımı yok. İlk compact blok daha da dalgalıydı:
classifier 44,34–205,31 ms, full step 200,57–409,95 ms. GPU timestamp ölçümü
her dispatch'e ek yük getirdiğinden normal kullanımı ayrı ölçtük.

## GPU timestamp kapalı kısa karşılaştırma

Dense/compact/dense/compact sırası, her blok 3 gerçek adım; contact probe kapalı.
Kare 110..121, substeps=200, uyuyan yaklaşık %99,1–99,3. Yalnız scheduler
anahtarı değiştirildi; fizik/sleep parametreleri aynı kaldı, rewind olmadı.

| Mod | Altı tam adım örneği, ms | Medyan ms |
| --- | --- | ---: |
| Dense | 291,87; 86,76; 230,95; 348,43; 81,89; 204,92 | **217,93** |
| Compact | 322,82; 82,25; 195,46; 221,49; 83,37; 196,35 | **195,90** |

Bu pencerede medyan yaklaşık %10,1 daha düşük; yoğun yolun örnek aralığı
81,89–348,43 ms, compact'ınki 82,25–322,82 ms. Örnekler küçük ve değişkenlik
büyük. Aynı checkpoint'ten iki kol veya izole viewport/GPU koşulları kullanılmadı;
kesin/stabil hızlanma oranı, fizik paritesi ya da geniş performans kabulü değildir.

## Sonuç ve sonraki hedef

- Aktif yol gerçekten çalıştı; yüksek uykuda ağır temas işini azalttı.
- Sleep classifier/carry ve ekstra dispatch maliyeti artık önemli. Sonraki
  optimizasyon classifier'ın her alt adımda okuduğu/taşıdığı state ve GPU grup
  iş planlaması olmalı. Uyku/uyanma güvenliği korunmalı.
- Substep sayısı ve eşikler bu ölçümde değiştirilmedi. Görülen yaklaşık %10
  medyan fark üzerinden %90 uyku = %90 hızlanma beklentisi kurulamaz.
- Son durum: kare 121, paused/manual script driver; uyuyan 73 936 (%99,324).
  compact_sleep_work=true geri yüklendi; contact probe ve GPU profiling kapalı.
  Şu ana kadar fizik guard'ı/held/rewind yok; bütün adımlar başarılı.

Uygulama ve kapsam: [DEM_ACTIVE_WORK_QUEUE.md](DEM_ACTIVE_WORK_QUEUE.md).
