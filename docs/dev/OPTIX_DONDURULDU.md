# OptiX: DONDURULDU

> **Durum:** REFERANS — OptiX yolu **geliştirilmiyor**, ama derleniyor ve
> çalışıyor. 2026-09-16'da donduruldu.

## Karar

OptiX/CUDA yolu **bakım modunda**. Yeni özellik almaz, mevcut davranışı korur.
Bir yetenek yalnızca OptiX'te çalışıyorsa, o yetenek **eksiktir** — Vulkan
birincil GPU yolu (bkz. `CLAUDE.md` kural 6).

**Silinmedi ve silinmeyecek**, çünkü gerçek bir işi var: bkz. aşağıdaki TDR.

## Neden dondu

Tek kişilik bir ekip için asıl maliyet kodun durması değil, **her çekirdek
değişikliğinin ikinci bir limanı olmasıdır.** `params.h` ABI'sine dokunan her
değişiklik `.cu` çekirdeklerine, `OptixWrapper`'a, `OptixBackend`'e ve
`IBackend`'e ayrı ayrı taşınmak zorunda. 2026-09-16'da seçim tamponunu sökerken
bu vergi beş dosyada ödendi.

## TDR: OptiX'in gerçek işi

★★★★ **TDR OptiX'i öldürmez.** Gözlem (2026-09-16, proje sahibi): TDR'de
yalnızca **Vulkan sürücüsü** sıfırlanıyor; raster ve Vulkan RT çöküyor, CUDA/
OptiX bağlamı ayakta kalıyor. Bu yüzden OptiX bu üründe gerçek bir cankurtaran
— "aynı GPU, o da ölür" varsayımı **yanlıştır** ve bir ajanın buraya bakmadan
yapacağı ilk tahmin tam olarak budur.

Sonuç: OptiX derlemeden **çıkarılamaz**, varsayılan-kapalı bir derleme
anahtarının arkasına da konamaz. Dondurma yalnızca "yeni iş almaz" demektir.

## Canlandırmak isteyen nereden başlar

1. `include/params.h` — device params ABI'si. Vulkan tarafında karşılığı
   değiştiyse önce burası hizalanır.
2. `src/Device/ray_color.cuh` — çekirdek.
3. `src/Render/OptixWrapper.cpp` — host tarafı; `OptixBackend` yalnızca
   yönlendirici.

## Sökülenler (geri istenirse git geçmişinde)

- GPU seçim tamponu (2026-09-16): `params.pick_buffer` / `pick_depth_buffer`,
  `ensurePickBuffers`, `getPickedObjectId`, `getPickedObjectName` ve çekirdek
  yazmaları. Seçim artık Vulkan raster ID geçişiyle yapılıyor
  (`scene.pick_gpu`). Bu tampon **okuyanı olmadan** her karede yazılıyordu.
