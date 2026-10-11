# Kullanıcı son derlemesi — açık sahne maliyet ölçümü

Dış IPC ile 0→6 kare, GPU timestamp açık; timestamp ayarı sonunda geri
yüklendi. Sahne kare 6'da durmuş bırakıldı, authored parametreleri değişmedi.
Build/uygulama başlatma yapılmadı. Ham son altı kayıt:
[gas_whitewater_latest_cost_2026_10_10.json](gas_whitewater_latest_cost_2026_10_10.json).

## Bulgular

- İlk-kare hold kalkmış: kare 6 `gpu_resident`, `inventory_step_held=false`.
  2,500 sıvı parçacığı, gaz envanteri 0.001728534 kg / 6602.596 J.
  Son adım thermal evaporated 0.000575746 kg. Bu sayılar tek başına tam
  fizik/ledger kabulü değildir.
- WW kapalı, alive/bin/input/gen/advect sayaçları sıfır; bu ölçümde WW maliyeti yok.
- Altı adım medyanı: liquid 66.93 ms, gas 105.56 ms (ayrı medyan toplamı
  172.49 ms). GPU timing raporları 21.91–25.12 ms. Bunlar ilgili compute
  context'in ölçülen kernel toplamlarıdır; viewport ve raporda olmayan işi
  içermez. Bu farkın tamamı CPU hesap süresi diye yorumlanmamalıdır.
- Liquid transfer medyanı upload 33,944,328 B, download 84,586,848 B:
  toplam 113.04 MiB/adım. 16 transfer-batch sonu, batch süre toplamı medyan
  16.14 ms. Fizik kg/J tam-grid upload/download sayaçları sıfır. Kalan
  sıvı/grid consumer transferleri hâlâ büyük.
- Gaz pressure wall medyan 27.13 ms; GPU `sim_grid_sor` 80 dispatch/adım,
  16.36–17.19 ms. Ölçülen GPU süresinde başlıca iş basınç çözümü.
- Surface exchange wall medyan 10.50 ms; gas host-sync 7.41 ms. Inventory
  scalar transport wall sayaçları kuyruklama süresini de içerir; saf yürütme
  maliyeti olarak kullanılmaz.
- RAM cache 2,227,600 B / 1 kareden 206,542,024 B / 7 kareye çıktı:
  yaklaşık 32.48 MiB/ek kare. Süreç private memory kare 1'de 3.827 GiB,
  kare 6'da 3.993 GiB; working set 1.786→1.954 GiB. RAM büyümesinde
  cache somut ve büyük bir bileşen. Bunu tek başına leak olarak adlandırmayız.
- Başlangıç VRAM kullanım 2,116,243,456 B; render/viewport tracked
  659,032,544 B, untracked 1,457,210,912 B. API untracked kısmı sim buffer,
  driver veya başka kategorilere ayıramıyor; tamamı buhar belleği sayılamaz.

## Durmuş viewport

Kare 6'da yaklaşık 3.017 s boyunca `display_frames` 2167'de sabit kaldı.
Raster frames submitted 9034→9131, consumed 9033→9130: yaklaşık 32.15
çizim/s. Resource drains 309'da sabit. Frame telemetry 1.34–1.37 ms/frame,
tek draw call, 12 mesh üçgeni raporluyor; bu ölçüm boş proje değil, durmuş
test sahnesidir. GPU yüzdesi IPC tarafından ölçülmüyor; kullanıcının %20/%25
gözlemi bu sayaçlardan doğrulanmış bir yüzdelik sayı değildir.

Kaynakta Main'in background/dormant sunum kapısı, viewport GPU render
işinden sonra geliyor. Sunumu durdurmak daha önce kaydedilen GPU işini
durdurmuyor. `needsViewportRender()` dirty/probe/presentationPending
durumları da render isteğini tekrar açabiliyor. Idle çizimin kapatılması
bu aşamadaki gereksiz çalışmaya odaklanmalı; kullanıcı etkileşimi, timeline,
IPC edit, cache restore ve async slot tüketimi korunmalı.

## Öncelik

1. Background/dormant raster redraw nedenini ön render kapısında kesmek.
2. Sıvı consumer/mirror ihtiyacını adım bazında izleyip kalan 113 MiB
   transferi azaltmak; CPU'nun gerçekten okuduğu grid'i atlamamak.
3. Timeline snapshot büyümesini lossless sabit-değer/payload paylaşımıyla
   azaltmak; rewind doğruluğu ve budget hesabı korunmalı.
4. Gas pressure için sabit 80 SOR dispatch yerine doğrulanmış residual/
   çözüm maliyeti çalışması. Fizik kabulü olmadan iterasyon sayısı düşürülmez.

Bu turn ölçüm ve kaynak audit'idir; yukarıdaki yeni optimizasyonlar uygulanmış
veya hız kazancı ölçülmüş sayılmaz. Önceki readback/logger düzeltmeleri ayrı
partidir. Timeline RPC dönüş süresi solver süresi değildir: scrub ertelenerek
işlenebilir; bu rapor solver sayaçlarını ve GPU timestamp'lerini kullanır.
