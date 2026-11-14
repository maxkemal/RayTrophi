# Player canlı smoke — 2026-11-13

Kullanıcının Faz 1k sonrası tam build/link yaptığı Studio çalıştırıldı. Kullanıcı
canlı testleri devraldı; uygulama açık bırakıldı, sonraki test/resize/kapanış
ajan tarafından tamamlanmış sayılmadı. Yeni Faz 2a kaynakları bu exe'de yoktur.

## Doğrulanan kapsam

- Vulkan → OptiX → CPU/Embree → Vulkan geçişi ve 128 sample kare üretimi.
  OptiX diagnostic `first_cuda_error=0`; Embree flat mesh axis hit 6/6.
- Gerçek Vulkan OIDN GPU yolu 17/17 başarılı çağrı; CPU fallback 0.
- Python transport ayrımı sonrası 17/17 kısa binding kontrolü geçti.
- UV authoring, volume material, audio telemetry, upscale validation,
  lens/weather ve realtime denoiser API probları geçti. Bunlar tüm GPU
  efektlerinin görüntü kalitesi kabulü değildir; SVGF available=false idi.
- Kaydet/yeniden aç: iki boş fluid/gas domaini, 16 hair guide ve hacim
  malzemesi korundu; timeline 1/2/3/0 seek doğrulandı.
- Volume testinin finally bloğu, get çıktısındaki ham ve türetilmiş smoke
  alanlarını birlikte geri gönderiyordu. Test betiği yalnız ham alanları
  geri yükleyecek şekilde düzeltildi; yeniden koşu ve eşitlik kontrolü geçti.
  Üretim servisinde değişiklik yapılmadı. İlk başarısız geçici log korunur.

## Devralınan test durumu / sınırlar

Uygulama geçici fixture.rtp sahnesinde, Vulkan Rendered ve viewport OIDN açık
bırakıldı. Başlangıç viewport OIDN ayarı kapalıydı. Başlangıç sahnesi ve ayarlar
`docs/dev/.captures/player-live-smoke/start.rtp` ve `denoiser-original.json`
içinde. HD720p preset alanı seçildi ancak Apply Resolution uygulanmadı;
son ölçülen gerçek render hedefi 1680×945. Native pencere boyutu değişti,
gerçek render-target/AOV resize kabulü ve temiz kapanış henüz yok.

Gaz/fluid boş fixture kabulüdür; uzun solver, VDB sequence, gerçek yağış,
saç ray-hit/shadow, FSR dispatch, SVGF görüntü kalitesi veya sayısal backend
paritesi kanıtlanmadı. Legacy log yolunda `There are no triangles!` satırı
vardı; çalışan flat mesh renderı bu satırı temiz log kabulüne dönüştürmez.
Gerçek Player exe/link kabulü henüz yok. İki ajan doğrulama JPG'si silindi;
geçici log ve sahneler ignored `.captures` altında tutulur.
