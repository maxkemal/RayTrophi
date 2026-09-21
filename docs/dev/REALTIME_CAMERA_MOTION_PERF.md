# Genel Realtime — kamera hareketinde düşük FPS

> **Durum:** AKTİF — **ÖLÇÜLDÜ (2026-09-07): karenin ~%60'ı Physical Sky'ın ambient koni integrali.** Kök neden bulundu, kod henüz değiştirilmedi; düzeltme sırada.

Kullanıcı hair malzeme eksikliğini ve genel Realtime performansını ayrı
bildirdi; performans belirtisini **kamera hareketinde düşük FPS** olarak
netleştirdi. Hair üretimi bu şikâyetin ölçülmüş nedeni değildir.

---

## ÖLÇÜM (2026-09-07) — karenin ~%60'ı Physical Sky'ın AMBIENT konisi

Sahne: 1680×945, 695 735 üçgen, 67 draw, 1 nokta ışık, Nishita + HDRI overlay,
cam malzeme, DoF açık, `quality` preset. Yöntem: IPC'den `camera.set_position`
uykusuz döngüde sürülüp `viewport.frame_telemetry`'deki `frames_submitted`
artışı başına duvar saati.

| konfigürasyon | ms/kare | fark |
|---|---:|---|
| Material, hepsi açık | **102.0** (5 koşuda 100.2–104.2) | — |
| − DoF | 100.3 | DoF post = **1.7** |
| − güneş cascade'leri (`sun_intensity=0`) | 98.9 | 3 tam sahne gölge geçişi = **1.3** |
| − Physical Sky (`world.set_mode solid`) | **38.6** | **gökyüzü = ~60 ms** |
| − kalite preset (gökyüzü kapalıyken) | 38.3 | 0.3 |
| Solid shading (material shader YOK) | 41.0 | ≈ material-eksi-gökyüzü |
| Gökyüzü **açık** + koni 8→4 tap | **79.1** | −21 |

### Yukarıdaki aday listesinden ELENENLER

Bu dosyanın altındaki "maliyet adayları" bölümü ölçümle **yanlış ağırlıklı**
çıktı; kısaca:

- **Gölge atlasının her kare yeniden çizilmesi**: güneşin 3 cascade'i = **1.3 ms**.
  Cache eklemenin değeri, bu sahnede, ölçülen bütçenin %1.3'ü.
- **Transmission replay + doku fetch'leri + cam/dispersion + 1608 satırlık PBR
  shader**: hepsi birlikte **~3 ms**. Material-eksi-gökyüzü (38.6) Solid'den
  (41.0) daha ucuz ölçüldü — gürültü içinde.
- **Readback / present / DoF gather**: DoF = 1.7 ms; `host_read_ms` 0.0.

### KÖK NEDEN

`samplePhysicalSkyFiltered` **yalnızca ambient bloğunda** çağrılıyor
(`shaders/material_preview_frag.frag` ~1291/1297: `envDiffuse(N)` +
`envReflection(R)`) — yani doğrudan ışık terimi değil, **doğrudan
aydınlatılmayan** terim. Her çağrı 1+8 = 9 kez tam `sampleCanonicalWorld`:
SkyView LUT + güneş diski (acos/asin/smoothstep) + transmittance LUT +
overlay açıkken **ayrıca ham equirect fetch** (atan/acos/fract). İki lob =
fragment başına ~18 tam gökyüzü değerlendirmesi.

Prefiltered kademe **`worldMode == 1u`'e kapalı** (~1261):

```glsl
if (worldMode == 1u && (sceneFlags & 32u) != 0u) {   // 1u = HDRI
    irradiance  = texture(worldIrradiance, ...);      // 3 ucuz fetch
    prefiltered = textureLod(worldPrefiltered, ...);
    brdf        = texture(worldBrdfLut, ...);
} else {
    envDiffuse    = samplePhysicalSkyFiltered(N, 1.0);   // 9 tap
    envReflection = samplePhysicalSkyFiltered(R, rough); // 9 tap
}
```

Shader'ın kendi yorumu bunu söylüyor: *"HDRI has its prefiltered maps below."*
Nishita'nın prefiltered karşılığı hiç üretilmemiş.

★★★ **En kötü bileşim: HDRI overlay + Physical Sky.** HDRI var ama prefiltered
kademesi erişilemiyor; ham equirect, koninin **içinde** 9 kez örnekleniyor.
HDRI'nin bedeli iki kez ödeniyor, faydası hiç alınmıyor.

★★ **Ve maliyet gölgelenen FRAGMENT sayısıyla ölçekleniyor.** Ana geçiş
depth-prepass'siz forward ve mesh'ler `unordered_map` sırasında çiziliyor
(`VulkanViewportBackend.cpp` ~3760), yani derinlik karmaşıklığı arttıkça aynı
piksel bu integrali defalarca ödüyor. "Geometri kalabalıkça pahalılaşıyor"
belirtisinin mekanizması budur.

### YAPILACAK (sıradaki parti)

1. Nishita için **prefiltered kademe** üret: SkyView LUT'undan irradiance
   haritası + GGX prefilter mip zinciri (HDRI'nin `MaterialPreviewIbl.cpp`
   yolunun karşılığı). Kare başına bir kez; LUT zaten ~15 Hz'e kısılı.
2. Shader kapısını `worldMode == 1u`'dan **"prefiltered hazır mı"**ya çevir.
   Overlay açıkken overlay de prefiltere KATILMALI, yoksa koni geri gelir.
3. Koni integralini fallback olarak bırak (prefilter hazır değilken).

★ Tap sayısını kısmak palyatif, çözüm değil: 8→4 yalnızca 21 ms getirdi, çünkü
merkez eval + güneş diski + overlay fetch her hâlükârda ödeniyor.

### KABUL TESTİ

Aynı prob, aynı sahne, `quality` preset:

- **Maliyet:** 102 ms → hedef ~45 ms/kare.
- **★★★ GÖRÜNTÜ DEĞİŞMEMELİ:** `render.probe` ile min/mean/max al; ambient
  terimin ortalaması korunmalı. Yalnızca hıza bakmak, ambient'i karartan bozuk
  bir uygulamayı da "GEÇTİ" sayardı — prefilter enerjiyi KORUMALI.
- Nishita **ve** HDRI-overlay açık halde ayrı ayrı ölç.

### ★★★★ ÖLÇÜ ALETİ UYARISI — `viewport.render_frames` KULLANILAMAZ

`rtapi::renderViewportFrames` (`src/Api/RtApiViewport.cpp:248`)
`g_ctx->backend_ptr` üzerinden `render_progressive_pass` çağırıyor: **path
tracer'ı** sürüyor, realtime raster'ı değil. Bu tuzağa 2026-09-07'de düşüldü;
verdiği sayılar 1558/1027/1031 ms'ti ve **shading moduna da kalite preset'ine
de duyarsızdı**. Teşhis anahtarı tam olarak budur: *raster'ı ölçtüğünü
sandığın bir sayı `viewport.set_shading solid` ile değişmiyorsa, o sayı
raster'ı ölçmüyordur.*

★★ `loop.viewport_render` de bu işi **görmez**: material 2.06 ms / solid
2.19 ms okurken gerçek fark 103 vs 41 ms'ti (asenkron kare halkası GPU
maliyetini o bölümün dışına düşürüyor).

★ Uygulama boştayken `loop.frame` ~507 ms okur ve `loop.present` /
`throttle_sleep` **hiç koşmaz** — bu dormant tier'ın `SDL_WaitEventTimeout`'u,
arıza değil. `frames_submitted` deltası 0 ise ölçtüğün şey yok.

**Çalışan prob:** `camera.set_position` (parametre adı `position`, Vec3 dizisi)
uykusuz döngüde + `frames_submitted` artışı başına duvar saati. IPC'den kamera
hareketi kare ÜRETİYOR (20 hareket → 20 submitted/consumed doğrulandı).

### Yan bulgu (ayrı, küçük)

`VulkanViewportBackend::buildRasterGeometry` (~5446) taban sınıfın
`rebuildRasterInstanceLayout()` çağrısını düşürmüş → global instance buffer ve
**GPU culling realtime viewport'ta hiç açılmıyor** (`gpu_culling=false`,
`cull_mesh_count=0`; telemetri doğruladı). Proxy/LOD yolu da bu yüzden ölü.
Ama korkulan drenaj fırtınası OLUŞMUYOR: 80 karede `resource_drains` deltası
material'da 79 (=1/kare), solid'de 0. Düzeltmek doğru, kare bütçesinde yeri yok.

---

## (Aşağısı 2026-09-06 kaynak denetimi — artık ölçümle ELENMİŞTİR, bağlam için korundu)

## Kaynakta görülen maliyet adayları

- `VulkanViewportBackend::renderProgressive` temiz sahne/sabit kamerada eski
  görüntüyü sunabilir. Kamera hash'i değişince normal raster geçişleri yeniden
  çalışır. Sabit görünümün hızlı olması hareketli karelerin ucuz olduğunu göstermez.
- `MaterialPreviewShadow.cpp::recordMaterialPreviewShadowPass` çizilen karede
  seçili shadow view'larının mesh gölgelerini ve varsa layered volume shadow
  compute işini yeniden yapar. Kareler arası shadow revision cache'i yoktur.
  Point ışık altı yüz, directional/güneş birden fazla kademe kullanır.
- `MaterialPreviewTransmission.cpp` opaque snapshot sonrası mesh listesini
  tekrar dolaşıp transmission pass'i için tekrar draw gönderir. Shader'ın
  opaque fragment'i elemesi vertex/draw maliyetini sıfırlamaz. Materyal
  sınıflandırmasıyla yalnız gereken mesh'leri gönderme fırsatı incelenmeli.
- Hacim/SDF ve HDR post ekran kaplamasına göre maliyet üretir; etkin DoF,
  bulanık piksellerde ek gather örnekleri kullanır. Bunlar bağımsız GPU geçiş
  zamanları olmadan tek toplamdan ayrılamaz.
- Son görüntü GPU→CPU okunup sunumda SDL dokusuna aktarılır. Frame ring
  gecikmeyi örter, bu kopyaları ortadan kaldırmaz. Çözünürlükle birlikte
  `host_read_ms`, `present_ms`, `display_texture_upload_ms`, slot waits ve
  tüketilen/gönderilen kare farkı ölçülmeli.

`frame_ms`/`cpu_record_ms` GPU pass timestamp değildir. Mevcut mesh draw ve
üçgen sayaçları bütün shadow/transmission/fullscreen işlerinin toplamını
göstermez. İyi görünen ana geometri sayımı, ucuz bir kare kanıtı değildir.

## Cache eklemeden önce doğruluk sınırı

Point/spot ışık dönüşümü değişmese bile mevcut projeksiyon far mesafesi
kamera merkezine göre hesaplanabilir; directional kademeleri zaten kameraya
bağlıdır. Dahası mevcut mesh gölge draw'ı camera-culling sonucunda compact
edilmiş instance listesini kullanabiliyor. Kamera dönüşü caster listesini
değiştirebilir; yalnız ışık/sahne dirty bayrağıyla cache yapmak eski/eksik
gölge bırakabilir.

Cache anahtarı gerçek shadow matrisi, caster görünürlüğü/dönüşüm/geometri,
alpha/material/texture ve hacim density/transform/optical revision'larını
kapsamalı. Işığa göre caster seçimi ve ekran dışından gölge düşmesi de
doğrulanmalı. Kalite veya hareketli gaz değişiminde cache geçersizleşmeli.

## Önerilen ilk uygulama

1. Frame slot'larına bağlı, sonuç hazırken okunabilen GPU timestamp'leri:
   mesh shadow, volume shadow compute, opaque, transmission/volume/SDF,
   post ve readback. Ölçüm için her kare GPU bekletilmez. UI/script/IPC aynı
   sonuçları okur; ölçülmeyen değer sıfır diye raporlanmaz.
2. Aynı kamera hareketi ve aynı çözünürlükte Performance/Balanced/Quality;
   DoF açık/kapalı ve hacimli/hacimsiz karşılaştırma. Kamera sabit idle
   karelerini hareketli render maliyetiyle aynı ortalamaya karıştırma.
3. En büyük ölçülen kalemi optimize et: doğru invalidation ile gölge cache'i,
   gereksiz transmission draw'larının elenmesi veya görüntü aktarımı.

İlk aday gölge tekrarlarıdır; **ölçülmüş kök neden değildir**. Bu incelemede
uygulama Rendered modundaydı, bu yüzden oradan okunan eski raster telemetry
değerleri hareketli Realtime benchmark'ı olarak kullanılmadı.
