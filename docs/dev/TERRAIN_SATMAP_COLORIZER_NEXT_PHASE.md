# Terrain SatMap Colorizer — Next Phase

Bu ek plan, mevcut `TERRAIN_SATMAP_COLORIZER_ROADMAP.md` için uygulama sırasını
netleştirir: önce maskelerin çoklu renk üretimi, sonra heightmap/mask hizalama
denetimi.

## S1 — Mask başına çoklu renk rampası

SatMap tek bir ortak gradient yerine alan başına mini rampalar kullanır:

- Height: alçak kot → grass/soil, orta kot → rock, yüksek kot → alpine/snowline
- Slope: yumuşak → toprak/ot, orta → karışık kaya, dik → sert kaya
- Flow: kuru → toprak, orta → koyu/ıslak toprak, yüksek → yosun/su yakın ton
- Soil/Hardness: yumuşak zemin → sıcak toprak, sert zemin → taş/kaya

Her rampada en az dört anlamlı renk bölgesi bulunur. Preset seçimi rampaları ve
mask karışım ağırlıklarını birlikte getirir; kullanıcı stop'ları elle kurmak
zorunda kalmaz.

## S2 — Snow korumalı özel overlay

Snow, Height/Slope/Flow rampalarından biri değildir. `Snow Layer` çıktıları ayrı
ve yüksek öncelikli bir overlay olarak okunur:

- `Snow`: temiz kar kapsaması
- `Ice`: soğuk mavi/beyaz buz tonu
- `Meltwater` ve `Meltwater Depth`: ıslak/kirlenmiş/erimeye yakın kar tonu
- `Avalanche`: taşınmış veya bozulmuş kar kapsaması

Renk sırası:

```text
Base SatMap → slope/flow/soil variation → snow/ice protected overlay
```

Snow maskesi doluysa SatMap onu ezemez. Eriyen bölgelerde geçiş temiz kar →
ıslak kar → kirli kar → alttaki soil/rock şeklinde kademeli olur. Dik yan
yüzeylerde slope retention kar miktarını azaltır; yatay yüzeylerde korur.

## S3 — Preset semantic binding

Presetler node ID'sine değil semantic role'e bağlanır:

```text
Height, Slope, Flow, Soil/Hardness, Snow, Ice, Meltwater, Avalanche
```

Graph'te alan varsa otomatik bağlanır. Alan yoksa preset mevcut alanlarla
çalışır ve height tabanlı fallback kullanır. Snow presetleri, Snow Climate
presetleriyle uyumlu olabilir ancak fiziksel Snow Layer çıktılarının yerine
geçmez.

## S4 — Heightmap/mask hizalama denetimi

Bu konu çoklu renk fazından sonra ayrı ele alınır:

1. Height, Slope, Flow ve Snow maskeleri tek renk debug map olarak export edilir.
2. Aynı UV noktaları arasında karşılaştırma yapılır.
3. Row-0 yönü, Y flip, bilinear koordinat merkezi ve aspect oranı kontrol edilir.
4. Field → paint ve paint → shader UV zinciri ayrı ayrı test edilir.
5. Kayma çözülmeden preset renkleriyle ilgili hata raporlanmaz.

## S5 — Paint-resolution gerçek değerlendirme

Height, erosion, flow ve snow kaynakları field çözünürlüğünde kalır. SatMap renk
kararları, mini rampalar, threshold'lar, noise ve snow overlay kararları paint
resolution'da hesaplanır. Kaynak maskeler paint grid'e bilinear/bicubic örneklenir;
procedural kararlar ise paint grid'de yeniden üretilir.

Bu sıra 1024 field + 4096 paint senaryosunda gerçek yüksek çözünürlüklü renk ve
maske ayrıntısı üretir; yalnızca sonradan upscale yapılmış bir görüntü üretmez.
