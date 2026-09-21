# Raster: "fusion" maliyeti ÖLÇÜLDÜ — fragment gölgelendirme × overdraw

> **Durum:** AKTİF — canlı IPC ölçümü (2026-09-08, GPU culling düzeltmesi
> derlendikten SONRA). **Ölçümler geçerli; §3'ün YORUMU düzeltildi** —
> bkz. [RASTER_MICROTRIANGLE_WALL.md](RASTER_MICROTRIANGLE_WALL.md).
> Düzeltme yazılmadı.

## Sahne ve ortam

1000 foliage + terrain + river. GPU culling artık açık
(`gpu_culling=true`, `cull_mesh_count=35`, full instance 722→433). 1680×945.
Ölçüm yöntemi: IPC'den kamerayı 16-20 kez oynat, `frames_submitted` deltasına
böl. Sayılar IPC gidiş-dönüşünü de içerir, **ama her durum için aynı** — anlamlı
olan farklardır. Tekrarlanabilirlik: aynı durum iki kez 270,9 / 271,4 ms.

## Ölçüm

| Durum | ms/kare | üçgen | ms / Müçgen |
|---|---|---|---|
| **SOLID shading** | **98,9** | 45,0M | **2,2** |
| Material + three_point (3 ışık) | 95,4 | 38,4M | 2,5 |
| Material + scene, quality preset | 316,9 | 44,3M | **7,2** |
| Material + scene, balanced | 257,8 | 35,6M | 7,2 |
| Material + scene, performance | 206,2 | 27,4M | 7,5 |
| Material + scene + traced probe | 270,9 / 271,4 | 38,6M | — |
| Material + scene + sky_bake (spec vis KAPALI) | 243,3 | 38,4M | — |

## Üç sonuç, üçü de doğrudan okunuyor

**1. Işık döngüsü DEĞİL.** Sahnede `scene_light_count = 0`. Scene modunda
fragment başına ışık döngüsü 1 iterasyon (Physical Sky güneşi), three_point'te
**3**. Yani daha az ışık iterasyonu olan yol 3 kat pahalı.

**2. Geometri DEĞİL — gölgelendirme.** Solid, **45,0M** üçgenle 98,9 ms;
scene/quality **44,3M** üçgenle 316,9 ms. Aynı geometri, aynı draw call sayısı,
**3,2 kat** fark. Aradaki ~218 ms tamamen fragment gölgelendirmesi.

**3. Maliyet üçgen sayısıyla DOĞRUSAL, piksel sayısıyla değil.** Scene modunda
üç kalite kademesinde de **7,2-7,5 ms/Müçgen**; solid ve three_point'te
**2,2-2,5**. Ekran 1,59M piksel, gönderilen 45M üçgen.

⚠★★★★ **DÜZELTME (aynı gün, kaynak okunduktan sonra).** Burada önce
"→ dolayısıyla overdraw" yazılmıştı. **O çıkarım eksik belirlenmiş.** Aynı
doğrusallığı üç mekanizma üretir ve ikisi overdraw değildir: **quad
kuantalama** (piksel-altı üçgen yine tam bir 2×2 quad gölgelendirir) ve
**primitive rate** (üçgen başına sabit setup maliyeti). 45M üçgen / 1,59M
piksel = piksel başına 28 üçgen, yani sahne **koşulsuz** mikro-üçgen
rejiminde. Ayrım önemli çünkü depth prepass yalnız gerçek overdraw'ı
düzeltir, diğer ikisini düzeltmez. Tam analiz:
[RASTER_MICROTRIANGLE_WALL.md](RASTER_MICROTRIANGLE_WALL.md).

★★★★ Kullanıcının "fusion seçince birden aşırı maliyet" gözlemi doğru, ama
sebep RayFusion'ın kendisi değil: scene lighting fragment yolu (1562 satırlık
`material_preview_frag.frag`) her örtüşen katmanda yeniden koşuyor.

### RayFusion'ın payı ayrıca ölçüldü

Traced ↔ sky_bake geçişi speküler görünürlük tüketicisini (`spacing.y`
kapısı) açıp kapatıyor: **270,9 → 243,3 ms, yani ~28 ms/kare.** Bu, scene
maliyetinin ~%16'sı. `rayfusion_specular_visibility.glsl` dokümanında
"GPU maliyeti ölçülmedi" yazıyordu; **artık ölçüldü.**

Difüz probe tüketicisinin (`probe_field.glsl`, 8 komşu × 2 texel) payı
**ölçülemedi**: tüketiciyi kapatan bir kol yok. Bu bir ölçüm boşluğu.

### Üretici tarafı temiz

`bounce_prepare_ms = 0,142`, `trace_ms = 0,247`, `signature_ms = 0,071`.
Önceki partinin CPU düzeltmeleri tutuyor; bu partinin maliyeti tamamen GPU
fragment tarafında.

## Aday: DEPTH PREPASS — ★ SONRADAN GERİ PLANA ALINDI

Depo'da raster viewport için depth prepass **yok** (aramadaki "prepass"
sonuçlarının hepsi hair BLAS prepass'i).

Aritmetik: 45M üçgen, 1,59M piksel. Overdraw büyük ve maliyet ona doğrusal.
Depth-only bir ön geçiş (foliage için alpha test dahil) + `EQUAL` derinlik
testiyle gölgelendirme geçişi, her pikseli **bir kez** gölgelendirir.
Beklenen: ~218 ms'lik gölgelendirme terimi overdraw katsayısına bölünür.

⚠ Bu bir tahmindir, ölçüm değil. Alpha-test'li foliage prepass'te de doku
okur, yani prepass bedava değil; ve şeffaf/blend geçişleri prepass'in dışında
kalmalı.

⚠★★★★ **Ve tahminin kendisi fazla iyimserdi.** Prepass geometriyi **bir kez
daha gönderir**; mikro-üçgen rejiminde geometri gönderimi zaten tabanın
kendisidir (solid: 455M üçgen/sn). Gerçekçi hesap 316 → ~200-220 ms, yani
**1,4-1,5x** — taban ~200 ms'de çakılı kalır. Prepass doğru bir iş ama baş
aday değil; üçgen sayısı düştükten sonra ucuz bir ek olarak girer.

## İkinci eksen: LOD/proxy

`scatter_triangle_target = 29,6M` ama gönderilen 38-45M. 1033 instance'ın
yalnızca **28**'i proxy'ye düştü (2 688 üçgen). Kullanıcı ayrıca proxy
kalitesinin düşük olduğunu bildirdi. Yani LOD hem **devreye girmiyor** hem de
girdiğinde **kabul edilebilir görünmüyor** — iki ayrı iş.

★ LOD'u iyileştirmek iki terimi birden keser: hem 2,2 ms/Müçgenlik raster
tabanını hem de onunla doğrusal olan gölgelendirme terimini.

★★★★ **NEDEN devreye girmediği bulundu:** `raster_cull.comp` her mesh'in LOD
eşiğini **tüm sahnenin** üçgen hedefine karşı çözüyor, yani 35 cull mesh'in
her biri 29,6M'in tamamını kendisinin sanıyor. Efektif tavan 35 katı; geri
besleme oranı her kare 1,5'e yapışıyor ve eşik clamp'e kaçıyor. Ölçülen
28/1033 tam olarak bu. Ayrıntı ve önerilen düzeltme:
[RASTER_MICROTRIANGLE_WALL.md](RASTER_MICROTRIANGLE_WALL.md) §4.

## Ölçüm boşlukları (kapatılmalı)

- Difüz probe tüketicisini kapatan kol yok → payı bilinmiyor.
- Overdraw doğrudan ölçülmüyor; üçgen sayısıyla doğrusallıktan **çıkarsandı**.
- Gölge atlası geçişinin geometriyi kaç kez yeniden gönderdiği bakılmadı.
- `resource_drains` hâlâ scene modunda kare başına ~1, solid'de 0. GPU culling
  düzeltmesi instance yükleme drenajını kaldırdı; kalan drenajın kaynağı
  bulunmadı.

## İlgili

- [RASTER_GPU_CULLING_NEVER_ENABLED.md](RASTER_GPU_CULLING_NEVER_ENABLED.md)
- [RAYFUSION_FOLIAGE_FRAME_COST.md](RAYFUSION_FOLIAGE_FRAME_COST.md)
- [RAYFUSION_SPECULAR_VISIBILITY.md](RAYFUSION_SPECULAR_VISIBILITY.md) — "ölçülmedi" notu bu ölçümle kapandı
