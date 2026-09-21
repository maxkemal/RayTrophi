# Değişim kapısına alan eklemek İKİ yeri değiştirir

> **Durum:** ARŞİV — 2026-09-17'de ölçülerek bulundu ve düzeltildi.

## Belirti

OptiX'te render başlıyor, örnek sayacı ilerliyor, ama görüntü birikmiyor:
pencerede hep son gürültülü örnek duruyor, önceki örneklemeler kayboluyor.

## Ölçüm

`render.optix_accum_status` (bu arıza için yazıldı):

| Alan | Değer |
|---|---|
| `accumulated_samples` | 1 → 35 (tırmanıyor) |
| `buffer_w_mean/max/center` | **1,00** (tamponda piksel başına tam 1 örnek) |
| `prev_zero_pixels` | **1.587.600** = 1680×945 → her launch'ta HER piksel |
| `read_w/h` vs `image_w/h` | 1680×945 = 1680×945 → alet hizalı |
| `reset_buffers_calls` | kare başına +1 |
| `set_render_params_reason` | `realtime_taa_samples,realtime_dof_max_coc,realtime_dof_max_taps` |

İki ekran görüntüsü, aynı kamerada, samples=1 ve samples=35: komşu-piksel farkı
**54,07** ve **54,65**. Birikim olsaydı ~54/√35 ≈ 9 olmalıydı.

## Kök

`Main.cpp`'de viewport render ayarlarını backend'e iletmeden önce bir **değişim
karşılaştırması** var: hiçbir şey değişmediyse `setRenderParams` çağrılmaz.

2026-09-14'te bu karşılaştırmaya beş alan eklendi (`realtime_taa`,
`realtime_taa_samples`, `realtime_depth_of_field`, `realtime_dof_max_coc`,
`realtime_dof_max_taps`) — çünkü TAA checkbox'ı kapatmak backend'e
ulaşmıyordu. **Ama aynı alanların `last_*` kopyalarına yazan satırlar
eklenmedi.** `last_taa_samples = -1` gibi başlangıç değerleri sonsuza kadar
öyle kaldı, yani karşılaştırma her karede "değişti" dedi.

Zinciri: kapı her kare açıldı → `OptixBackend::setRenderParams` her kare →
`OptixWrapper::resetBuffers()` her kare → içindeki
`cudaMemset(d_accumulation_float4, 0, ...)` **resize koşulunun dışında** olduğu
için birikim tamponu her kare sıfırlandı.

## İkinci kusur: alet yalan söylüyordu

`resetBuffers()` pikselleri sıfırlıyor ama `accumulated_samples = 0` satırı
resize koşulunun **içinde**. Yani tampon sıfırlanırken sayaç sayıyordu. Bu
asimetri olmasaydı arıza ilk gün görünürdü: sayaç 1'de takılır kalırdı.

## Ders

> **Bir değişim karşılaştırmasına alan eklemek İKİ yeri değiştirir:
> karşılaştırma ve hafıza.** Yalnızca birini yapmak hata vermez. Karşılaştırmayı
> unutursan kapı hiç açılmaz (2026-09-14'te düzeltilen hata); hafızayı
> unutursan kapı hiç kapanmaz (bu hata). İkisi de sessizdir, ve ikincisi
> "gereksiz iş" olarak değil **veri kaybı** olarak ortaya çıkar.

★ Kapı backend'den bağımsız: belirti OptiX'te görüldü çünkü orada sonuç silinen
bir tampondu, ama her kare gereksiz `setRenderParams` bütün backend'lerde
yapılıyordu.

★★ Aletin kör noktası: ilk eklediğim `wipe_count` sayacı yalnızca **realloc**
yolunu sayıyordu, bu memset'i değil. `wipe_count: 1` okumak "tampon
silinmiyor" diye yorumlandı ve beni iki tur yanlış yere götürdü. Kökü bulan
şey, host tarafındaki her okumanın ölçtüğü belleğin aynısını kullandığını kabul
edip **çekirdeğe kendi gördüğünü sorduran** `prev_zero_pixels` sayacı oldu.

## İkinci belirti: Solid → OptiX geçişinde çökme (aynı kök)

`Solid`'den doğrudan OptiX'e geçiş `CUDA error: unspecified launch failure
(719)` ile çöküyordu; Vulkan RT üzerinden geçiş çökmüyordu.

**Mekanizma (proje sahibinin teşhisi, kodla doğrulandı):**
`OptixWrapper::resetBuffers()` çözünürlük değiştiğinde `d_framebuffer` dahil
**bütün device tamponlarını** serbest bırakıyor — ve bunu **hiçbir stream
senkronizasyonu yapmadan**. O tamponları o anda uçuştaki bir `optixLaunch` ve
ona bağlı asenkron kopya kullanıyor olabilir. Senkronizasyonsuz `cudaFree`,
cihaz tarafında use-after-free'dir; belirtisi tam olarak 719'dur.

Vulkan RT üzerinden geçişte çözünürlük zaten eşitti, yani bu dal hiç
çalışmıyordu — iki yolun farkı buydu.

**★★★★ Dikkat: kapı düzeltilince çökme "geçti" gibi göründü, ama tehlike
değişmedi — yalnızca tetikleyici seyreldi.** Gerçek bir çözünürlük değişimi
render uçuştayken onu geri getirirdi. Bu yüzden serbest bırakmadan önce
`cudaStreamSynchronize(stream)` eklendi; yalnızca gerçekten yeniden tahsis
edilirken çalışır, kare başına maliyeti yoktur.

★ Aynı ders bu depoda TLAS için zaten ödenmişti: uçuştaki bir trace sırasında
hızlandırma yapısını yıkmak device-lost üretiyordu
(`feedback_vulkan_tlas_destroy_inflight_trace`). Kural sınıfı aynı:
**uçuştaki işin kullandığı device belleğini, beklemeden bırakma.**

★★ "Belirti geçti" ≠ "arıza gitti". Bir düzeltme arızanın TETİKLEYİCİSİNİ
seyreltiyorsa, arızanın kendisi hâlâ oradadır ve daha nadir, daha açıklanamaz
biçimde geri döner.

### Düzeltme yetersizdi: DÖRT serbest bırakma yeri vardı

İlk turda yalnızca `resetBuffers()` senkronize edildi ve çökme "geçti" sanıldı.
Sonra RayFusion → OptiX geçişinde 719 tekrar geldi (o derlemede düzeltme zaten
yoktu). Aramayı genişletince aynı desenin **dört** yerde olduğu görüldü:

| Yer | Ne bırakıyor | Neden tehlikeli |
|---|---|---|
| `resetBuffers()` | tüm device tamponları | uçuştaki launch kullanıyor olabilir |
| progressive, `resolution_changed` dalı | `d_framebuffer` + sabitlenmiş host tamponları | **ve** `host_output_copy_pending = false` ile bekleyen kopyayı olayı beklemeden "tamam" ilan ediyordu |
| progressive, pinned yeniden tahsis | sabitlenmiş host tamponları | asenkron kopyanın hedefi |
| `partialCleanup()` | **geometri** (`d_vertices`/`d_indices`), framebuffer, pinned | AS'in ve uçuştaki launch'ın referans verdiği bellek; sahne rebuild ve backend geçişinde çağrılıyor |

★★★★ İkinci satırdaki `host_output_copy_pending[slot] = false;` özellikle
öğreticidir: bir bayrağı false yapmak DMA'yı durdurmaz, yalnızca bizi kör eder.
Sonra o belleği bırakmak, hâlâ yazılmakta olan sabitlenmiş belleği geri
vermektir.

★★★ `partialCleanup()`'taki `std::lock_guard` yanıltıcıydı: mutex **host**
tarafını korur, GPU'da devam eden işi durdurmaz. "Kilit var, güvenli" okuması
bu sınıf hatanın tipik kaynağıdır.

★★ Hata bir sonraki senkron kontrolde raporlandığı için **çağrı yığını
yanıltır**: 719 `optixLaunch`/`launchOptixDisplayPost` satırında görünür, oysa
orada doğmaz. Satır numarasını kök sanmak bu turda iki kez yanlış yere
götürebilirdi.
