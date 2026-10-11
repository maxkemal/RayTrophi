# Solid fokus dışı GPU çizimi

Kullanıcı canlı ayrımı: boşta yüksek GPU tüketimi Solid modda; RT ve
RayFusion'da görülmüyor. Dış IPC, durmuş kare 63'te iki saniyede 85 raster
submission, sıfır display-frame ilerlemesi ve beş resource drain gösterdi.
Cache 64 kare / 2,309,496,477 bayt (2.15 GiB). Bellek sorunu bu çizim kapısıyla
çözülmüş sayılmaz; cache ayrı optimizasyon gerektirir.

Kaynak düzeltmesi: Main'in GPU render bloğundan önce küçük
ViewportGpuWorkPolicy çağrısı. Yalnız shading_mode=0 (Solid), fokus dışı veya
minimize durumda GPU viewport işini erteler; start_render isteği korunur.
Fokus/görünürlük geri geldiğinde son sahne çizilir. RT, RayFusion ve Matcap
bu yeni kapıya takılmaz. Sequence output ve explicit capture kapıyı aşar.

Simülasyon adımına fokus koşulu eklenmedi. Background event beklemesi Play,
anim graph playback veya sequence output sırasında 1 ms; sim cache üretimi
uzun 500 ms idle beklemesine takılmaz. Duran sahnede mevcut uzun bekleme
korunur. Simulation GPU işinin background Play sırasında devam etmesi
beklenen davranıştır; yalnız boşa Solid çizimi ertelenir.

Source/policy/XML kontrolü PASS, native build yapılmadı. Kullanıcı C++
derlemesi sonrası kontrol: Solid paused + alt-tab sırasında submitted sayacı
durmalı; Solid Play + alt-tab sırasında frame/cache sayacı ilerlemeli;
fokus dönüşünde güncel kare görünmeli. RT/RayFusion, capture ve sequence
output regresyonları kontrol edilir. Shader değişmedi. Bu kapı idle redraw'un
kaynağındaki dirty/presentation tekrarını tamamen çözmüş sayılmaz; live
resource-drain devam ederse ayrı producer audit'i gerekir.
