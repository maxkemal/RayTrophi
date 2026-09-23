# Izgara hacimsel verinin önünde kalıyor

> **Durum:** AKTİF — 2026-09-24'te teşhis edildi, **düzeltilmedi** (bilerek).

## Belirti

Realtime viewport'ta ana ızgara katı geometride doğru tıkanıyor, ama **hacimsel
verinin (duman/gaz) önünde kalıyor** — yoğun dumanın içinden ızgara görünüyor.

## Neden böyle

Çizim sırası doğru: hacim sahne geçişinde, ızgara ondan sonraki LDR geçişinde.
Sorun derinlikte:

| Boru hattı | depthTest | depthWrite |
|---|---|---|
| `MaterialPreviewSdfSurface` | ✔ | ✔ → ızgarayı **tıkıyor** |
| `MaterialPreviewVolume::previewPipeline` (HDR/material) | ✘ | ✘ |
| `MaterialPreviewVolume::solidPipeline` (solid) | ✔ | ✘ |

Hacim **hiçbir varyantta derinlik yazmıyor**, o yüzden sonradan çizilen ızgara
(derinlik testi açık, `LESS_OR_EQUAL`) hiçbir zaman reddedilmiyor.

★ Bu, yarı saydam bir hacim için genel olarak *doğru* davranıştır — derinlik
yazan saydam yüzey sıralama sorunları üretir. Belirti yine de meşru: yoğun
duman ızgarayı gizlemeli.

## Neden "tek satır" değil

`material_preview_volume.frag` **zaten** `gl_FragDepth` yazıyor (~336):

```glsl
vec4 firstClip = pc.viewProj * vec4(ro + rd*firstContributionT, 1.0);
float firstDepth = firstClip.z / firstClip.w;
gl_FragDepth = clamp(firstDepth, 0.0, 1.0);
```

Ama bu `firstContributionT` — hacmin **ilk katkı** noktası, opaklaştığı yer
değil. Boru hattında yazmayı açmak yeterli değil, çünkü:

1. En ince duman bile (alpha ≈ 0,001) ızgarayı **tamamen** gizler. Opaklık
   eşiği gerekir: `alpha >= T` ise `firstDepth`, değilse uzak değer.
2. "Uzak değer yaz" ancak derinlik **testi** açıkken güvenlidir. Oysa
   `previewPipeline`'da test **KAPALI** — o hâlde eşiğin altındaki fragmanlar
   sahne derinliğini 1.0 ile ezer ve arkadaki her şey öne çıkar.
3. Yani `previewPipeline`'da derinlik testini de açmak gerekir, ve bu
   hacmin önündeki geometriyle ilişkisini değiştirir.

★★★ Üçü birlikte, bu deponun en çok sessiz arıza üretmiş alanında
(çakışık hacim kutuları, TLAS slotu, publish kapısı, solid-probe kapısı) bir
davranış değişikliğidir. "Kolay düzeltme" görünüp değilken yapılan tam olarak
bu sınıftır.

## Yapılacak (ayrı parti)

1. Shader: `alpha` eşiğine bağlı `gl_FragDepth` (eşik altında uzak değer).
2. `previewPipeline`: `depthTestEnable = VK_TRUE`, `depthWriteEnable = VK_TRUE`.
3. `solidPipeline`: `depthWriteEnable = VK_TRUE` (testi zaten açık).
4. Eşiği **kadran yap ve IPC'ye aç** — "ne kadar yoğun duman tıkar" sanatsal bir
   karardır ve sabit bir sayı olarak gömülürse kalibrasyon turuna gömülür.

★ Kontrol: hacmin İÇİNDEKİ partikül/hair overlay'lerinin kaybolmaması. Derinlik
yazan bir hacim onları da tıkar, ve bu istenmeyen bir yan etki olabilir —
düzeltmeyi kabul etmeden önce o kareye bak.

★★ Ayrıca `compile_shaders.bat` çalıştırılmalı: shader değişiyor.
