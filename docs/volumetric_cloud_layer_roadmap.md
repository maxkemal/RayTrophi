# Volumetric Cloud Layer Roadmap

Bu not, world cloud sisteminin eski flat parametrelerden gercek katmanli procedural volume mimarisine tasinmasi icin yasayan yol haritasidir.

## Guncel Durum

- `NishitaSkyParams` artik iki adet `CloudLayerParams layers[2]` tasiyor.
- Eski `cloud_*` ve `cloud2_*` flat world alanlari kaldirildi.
- Scene JSON semasi yeni formata gecti:
  - `nishita.clouds.enabled`
  - `nishita.clouds.quality`
  - `nishita.clouds.base_steps`
  - `nishita.clouds.use_fft`
  - `nishita.clouds.layers[]`
- Eski cloud scene anahtarlari okunmuyor. Geriye uyumluluk bilerek drop edildi.
- Internal sky cloud volume sistemi layer basina ayri procedural VDB uretiyor:
  - `__RayTrophi_Internal_SkyCloudVolume_L1`
  - `__RayTrophi_Internal_SkyCloudVolume_L2`
- CPU sampler artik layer secimi yapmiyor; procedural cloud density icin sadece `local_p + VDBVolume` okuyor.
- GPU/Vulkan volume instance yolu layer-aware; her layer ayri volume instance olarak senkronize ediliyor.
- World UI cloud bolumu tek layer-editor dongusune tasindi.
- World keyframe sistemi layer bazli `cloud_layers[2]` geometry/lighting kaydi kullaniyor.
- `shape_type` artik CPU, OptiX/CUDA ve Vulkan procedural density formulunu etkiliyor.
- OptiX/CUDA tarafinda procedural cloud icin density veya coverage sifirken erken cikis var; bu sayede slider 0 -> yukari oynarken TLAS topology korunur ama bos cloud volume march edilmez.
- OptiX procedural cloud shadow/stochastic occlusion adimlari buyuk arazi sahnelerinde sinirlandi; normal NanoVDB volume yolu ayni davranista tutuldu.
- Internal sky cloud layerlari artik `scene.world.objects` / TLAS listesine eklenmiyor; sadece global volume buffer uzerinden render ediliyor. Boylece cloud toggle buyuk sahnede OptiX geometry sync tetiklemiyor.
- OptiX generic VDB shadow/occlusion probe'lari procedural sky cloud'u atliyor; sky cloud ana kamera ray'inde render ediliyor, mevcut sahne VDB'leri ise kendi normal occlusion yolunda kaliyor.

## Ana Hedef

Iki mevcut world cloud layerini sadece iki kopya noise olarak degil, farkli karakterlere sahip kaliteli volumetric cloud katmanlari olarak render etmek:

- Layer basina cloud type/shape.
- Layer basina yukseklik profili ve erosion davranisi.
- Layer basina scatter/absorption/shadow/phase.
- CPU, OptiX/CUDA ve Vulkan RT taraflarinda ayni anlamlara sahip parametreler.
- Cumulus + cirrus, overcast + low fog gibi kombinasyonlarin dogal gorunmesi.

## Tamamlanan Isler

1. `CloudLayerParams` struct eklendi.
2. `NishitaSkyParams` flat cloud alanlarindan layer dizisine tasindi.
3. Scene save/load yeni `clouds.layers[]` semasina gecti.
4. Internal cloud volume uretimi layer basina ayrildi.
5. L1/L2 max-density secim mantigi kaldirildi.
6. CPU procedural sampler sade imzaya indi.
7. UI layer editor dongusune tasindi.
8. Keyframe sistemi layer bazli geometry/lighting modeline gecti.
9. Cloud shape type CPU/CUDA/Vulkan instance ve shader yoluna tasindi.

## Siradaki Teknik Isler

### 1. Cloud Shape Modeli

Ilk aktif shape modeli eklendi. Simdilik tek procedural weather/noise kaynagindan farkli profiller uretiliyor.

Tipler:

- `0 Generic/Cumulus`: puffy, kopuk, orta yukseklik.
- `1 Cirrus`: ince, yassi, yuksek, dusuk density.
- `2 Stratocumulus`: genis patch'ler, orta puff.
- `3 Stratus/Overcast`: genis kapali tabaka, yumusak detay.
- `4 Fog/Low`: yere yakin, soft ve dusuk kontrast.

Backend paritesi:

- CPU: `rt_sample_procedural_cloud_cpu`
- OptiX/CUDA: `sample_procedural_cloud_density`
- Vulkan: `proceduralCloudDensity`

Kalan iyilestirme:

- Cirrus icin daha iyi streak/anvil stretch.
- Stratocumulus icin patch mask.
- Overcast icin alt taban yumusakligi ve horizon breakup.
- Fog icin terrain/camera yuksekligine daha iyi baglama.

### 2. Layer Kalite Parametreleri

Mevcut global `cloud_quality` ve `cloud_base_steps` korunuyor. Sonraki adim layer basina kontrol:

- `max_steps`
- `shadow_steps`
- `step_scale`
- noise octave preset

### 3. Scatter / Absorption Iyilestirme

Mevcut shader volume yolu zaten scattering/absorption/multi-scatter tasiyor. Cloud layer UI ve sync tarafinda fiziksel anlamlar netlestirilecek:

- `sigma_s`
- `sigma_a`
- `albedo`
- dual HG phase
- powder effect
- multi-scatter approximation

### 4. Atmosphere Entegrasyonu

Bulut rengi emission ile degil atmosphere + sun transmittance ile tasinmali:

- Sun -> cloud transmittance.
- Camera -> cloud transmittance.
- Sky ambient tint.
- Horizon/aerial perspective uyumu.

### 5. Weather Map

Coverage tek threshold olmaktan cikacak:

- Low-frequency weather mask.
- Coverage/density/type mask.
- Wind offset ile stabil animasyon.
- Ileride texture veya cached procedural tile.

## Kisa Vadeli Uygulama Sirasi

1. Presetlerin shape type ve lighting degerlerini gercek layerlara daha artist-friendly yazmasini iyilestirmek.
2. Shape profillerini weather mask ile zenginlestirmek.
3. Shadow marching'i shape/layer tipine gore ucuz/pahali ayarlamak.
4. Layer basina kalite/max step parametrelerini eklemek.
5. Weather map icin dusuk frekansli maskeyi density formulune katmak.

## Notlar

- Derleme kullanici tarafinda yapilacak; bu akista build komutu calistirilmez.
- Coverage araligi simdilik `0.0 - 0.6`.
- Density araligi simdilik `0.0 - 1.5`.
- Scale araligi simdilik `0.1 - 1.0`.
- `shape_type` UI/JSON ve backend density formulunde aktif girdi.
