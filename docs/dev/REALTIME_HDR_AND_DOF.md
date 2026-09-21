# Realtime raster: HDR ara hedef ve alan derinliği

> **Durum:** AKTİF — 2026-09-06'da yazıldı, **DERLENMEDİ**. Kabul testi
> `scripts/ipc/Probe-RealtimeDof.ps1`. Bu not, raster viewport'un tek geçişten
> iki geçişe taşınmasını ve realtime DoF'u anlatır.

## Neden HDR ara hedef şart oldu

Kullanıcının isteği tek cümleydi: *"realtime için de DoF ekleyelim."* Kod
okununca iş DoF değil, **onun ön koşuluydu**:

> **Bokeh, parlak bir noktanın DAİRE olarak açılmasıdır.** Bu ancak tonemap'ten
> ÖNCE, kırpılmamış HDR değerlerde olur. Raster viewport ise görüntüleme
> dönüşümünü **her shader'ın içinde** uyguluyordu (`material_preview_frag`,
> `material_preview_sky`, `material_preview_volume`,
> `material_preview_sdf_surface`), yani bulanıklaştırılacak HDR bir ara hedef
> **yoktu** — elde olan 8-bit, display-encoded bir kareydi.

★★★★ Onu bulanıklaştırmak bokeh değil **gri leke** üretir: parlak nokta zaten
1.0'a kırpılmıştır, blur onu geri getiremez. Sırası yanlış olan bir DoF
"çalışır ama ucuz durur" — ve o, kimsenin bug diye raporlamadığı türden bir
arızadır. Bu yüzden batch'in büyük kısmı DoF değil **hedef ayrımıdır**.

## Yeni akış

| Sıra | Geçiş | İçerik |
|---|---|---|
| 1 | `hdrRenderPass` (RGBA16F + D32, CLEAR) | gökyüzü, material-preview mesh'leri, hacim, SDF yüzeyi |
| 2 | transmission replay (RGBA16F, LOAD) | kırılma/cam — HDR anlık görüntüyü örnekler |
| 3 | `raster_post.comp` (compute) | **DoF + post zinciri** → 8-bit hedefe yazar |
| 4 | `renderPassLoad` (R8 + D32, LOAD) | ızgara, gizmo, saç, partikül, edit overlay |
| 5 | seçim maskesi + kompozit | değişmedi |

Solid/Matcap modunda 1–3 **hiç koşmaz**: o modun içeriği zaten
display-referred'dır ve tonemap'ten geçirmek yazılı renklerini bozardı. O modda
tek geçiş (`renderPass`, CLEAR) eskisi gibi çalışır.

★★★ Load/store op'ları render-pass **uyumluluğunu etkilemez**, bu yüzden LDR'nin
CLEAR ve LOAD varyantları aynı pipeline'ları paylaşır. Format uyumluluğun
parçasıdır, bu yüzden HDR geçişine bağlanan pipeline'lar ayrıca yaratılır
(`solidPipelineHdr`).

## CoC formülü path tracer ile AYNI

```
CoC_px = h * lensRadius * |zf - t| / (zf * t * 2 * tan(fovY/2))
```
Türetme: lens noktası `d` kadar kayar ve ışın odak düzlemindeki noktaya nişan
alır; `t` derinliğindeki bir nokta için dünya sapması `|d| * |zf - t| / zf`,
açı `sapma / t`, piksel karşılığı bölü `2*tan(fov/2)` çarpı yükseklik.

★★★ `lensRadius = aperture * 0.5` — yani `Camera::lens_radius`'un ta kendisi.
**Ayrı bir "realtime blur" kadranı bilerek KONULMADI:** o kadran, Rendered ile
Realtime'ın ayrışabileceği ilk yer olurdu ve belirtisi "önizleme yalan
söylüyor" olurdu. Panel/IPC'deki iki sayı (`max_coc_pixels`, `max_taps`)
**maliyet tavanlarıdır**, şiddet değil.

★ Varsayılan kamerada `aperture = 0` → CoC 0 → geçiş düz bir tonemap'e erken
çıkar. DoF'u açık bırakmanın bedeli yok; "açtım ama hiçbir şey olmadı"nın
cevabı neredeyse her zaman budur.

## Toplama (gather) kuralı

Arama yarıçapı **yalnızca merkezin CoC'u olamaz**: önde duran bulanık bir nesne
arkasındaki keskin pikselin üzerine taşmalıdır. Kaba bir 8-örneklemli halka ile
komşuluğun en büyük CoC'una bakılır; bakılmazsa bulanık siluetin kenarı jilet
gibi kalır.

★★★ **Scatter-as-gather:** bir örneklem merkeze ancak *kendi dairesi merkezi
örtüyorsa* katkı verir (`influence = clamp(ct - radius + 1, 0, 1)`). Bu kural
olmadan keskin arka plan, önündeki bulanık nesnenin üzerine sızar ve belirti
"arka plan hayalet bırakıyor" olur — yanlış tarafa bakarsın.

## Dört kapı, ve neden hepsi RAPORLANIYOR

`viewport.get_depth_of_field` yalnızca `active` döndürmez, kapalıysa
`inactive_reason` da döner:

1. ayar kapalı, 2. shading modu Material değil, 3. kamera ortografik,
4. `aperture = 0`.

★★★ Çıplak bir `false`, çağıran tarafta **"ölçtüm, sıfır"** ile
**"ölçemedim"**i aynılaştırır — bu deponun adı konmuş hata sınıfı. Betik de
(`Probe-RealtimeDof.ps1`) aynı ayrımı korur: ölçüm bölgesinde kenar yoksa
"OLCULEMEDI" der, "GECTI" demez.

## Ölçüm nasıl yapılır (kabul testinin mantığı)

`render.probe` bir bölgenin min/mean/max parlaklığını verir. **Bulanıklık
kontrastı (max−min) düşürür ama ortalamayı korur.** İkisini birlikte ölçmek
şart:

- yalnızca kontrast düşüşüne bakmak, **karartmayı** da bulanıklık sayardı;
- yalnızca ortalamaya bakmak hiçbir şey söylemezdi.

★★ Ve DoF "her şeyi bulanıklaştırmak" değildir: odaktaki nesne **keskin**
kalmalı. Betiğin 4. kapısı tam olarak bunu ölçer — bu kapı olmadan ekranı
komple bulanıklaştıran bozuk bir uygulama da "GEÇTİ" derdi.

## Bilinçli olarak yapılmayanlar

- **Bloom.** Artık mümkün (HDR hedef var) ama ayrı bir iş; bu batch'in kapsamı
  DoF'un ön koşuluydu.
- **Bıçak sayısına (blade count) göre poligonal bokeh.** Toplama diski dairesel;
  `cam.blade_count` okunmuyor. Cinema lens kusurlarıyla birlikte ayrı adım.
- **Yarı çözünürlüklü DoF.** Maliyet `max_coc_pixels` ve `max_taps` ile
  sınırlanıyor; yarım çözünürlük ayrı bir hedef + upsample demek.
- **Temel adaptörün (RT backend) kendi viewport kopyası HDR'ye taşınmadı.**
  Orada material preview artık **kapalı** (Solid/Matcap çizer), çünkü dört sahne
  shader'ı scene-linear yazıyor ve o yolda post geçişi yok. Açık bırakmak
  "viewport aşırı parlak" üretirdi. İki kopyanın birleştirilmesi ayrı iş.

## ★ En sinsi başarısızlıklar (bunlara ayrıca bak)

1. **Izgara/gizmo renkleri kaymış görünür.** O çizimler post'tan SONRAKİ geçişe
   taşındı; kaymışlarsa hâlâ HDR geçişindedirler.
2. **Boş sahnede material modunda gökyüzü kaybolur.** `hdrPassActive` ile
   `useMaterialPreview` **iki ayrı sorudur** (hedef vs. çizilecek şey var mı);
   aynı sanılırsa gökyüzü pipeline'ı uyumsuz geçişe bağlanır.
3. **Cam/kırılma içindeki parlak arka plan düz beyaza yapışır.** Transmission
   replay post'tan SONRA koşuyordur; kırpılmış görüntüden kırılma hesaplanır.
4. **Yeniden boyutlandırmada bozulma.** `hdrFramebuffer` boyuta bağlıdır ve
   `keepPipeline` korumasının DIŞINDA yok edilmelidir.
