# Mesh Paint - Fill Layer ve Smart Mask Mimari Raporu

Tarih: 2026-05-01
Durum: Tasarim raporu / uygulama yol haritasi

## Ozet

RayTrophi Mesh Paint mevcut durumda cok kanalli, katmanli ve GPU'ya hizli guncellenen bir texture paint sistemi. Bir sonraki dogru buyume adimi, raster boya katmanlarinin yanina **non-destructive Fill Layer** ve bu fill layer'lari kontrol eden **duzenlenebilir mask** sistemini eklemek.

Bu raporun ana karari:

- Fill Layer, tum texture'i raster olarak boyayan basit bir kisayol olmamali.
- Fill Layer procedural kalmali: kanal degeri, kanal secimi ve mask ayrica tutulmali.
- Curvature/AO gibi generator'lar dogrudan fill layer mask'i uretebilmeli.
- Kullanici generator sonucunu daha sonra manuel fircayla duzeltebilmeli.

Boylece sistem Substance kadar karmasik olmadan, Blender Texture Paint'ten daha guclu bir "renderer icinde hizli PBR boya" akisi sunar.

## Mevcut Yapinin Kisa Durumu

Mevcut Mesh Paint mimarisi su parcalardan olusuyor:

- `MeshPaintAdapter`: Secili mesh/material slot hedefini yonetir.
- `PaintTextureSet`: Base Color, Normal, Roughness, Metallic, Emission, Mask, Transmission ve Opacity texture'larini tutar.
- `PaintLayerStack`: Katmanlari kanal bazli pixel buffer'larla saklar ve renderer'in kullanacagi flat texture'lara composite eder.
- `PaintLayerData`: Her katman icin metadata ve kanal pixel verisini tutar.
- `PaintTextureCommand` / `PaintLayerCommand`: Stroke bazli undo/redo baglantisini saglar.
- Project save/load: Mesh paint layer stack'leri `mesh_paint_layers` altinda saklar.

Bu temel saglam. Fill Layer ve Smart Mask, mevcut stack'in ustune eklenmeli; mevcut raster paint davranisi bozulmamali.

## Problem

Raster paint layer, kullanicinin fircayla dogrudan pixel boyadigi katmandir. Bu karakter yuz boyama, leke, stamp, elle rutus gibi isler icin dogru modeldir.

Fakat su workflow'larda raster layer tek basina yetersiz kalir:

- Tum yuzeye tek materyal degeri vermek ama sadece maske ile gorunurluk kontrol etmek
- Edge wear / cavity dirt / AO shadow gibi otomatik maskeler uretmek
- Uretilen maskeyi sonradan elle duzeltmek
- Farkli PBR kanallarinda ayni maskeyi kullanmak
- Fill degerini sonradan degistirip tum sonucu aninda guncellemek

Eger Fill Layer, "tum texture'i raster pixel olarak doldur" seklinde uygulanirsa:

- Deger degisince tum texture yeniden yazilmak zorunda kalir.
- Mask/generator non-destructive olmaz.
- Katman tipi anlamsizlasir.
- Smart mask tekrar uretildiginde manuel duzeltmeler kaybolabilir.

Bu yuzden Fill Layer ayri bir layer tipi olmalidir.

## Hedef Workflow

### Karakter Yuz Boyama Ornegi

1. Kullanici hazir skin texture'li insan modelini acar.
2. Head mesh/material slot secilir.
3. `Add Fill Layer` ile `Face Paint Red` katmani eklenir.
4. Kanal: Base Color.
5. Fill degeri: kirmizi/oksit pigment.
6. Mask baslangici: siyah, yani layer gorunmez.
7. Kullanici mask'i fircayla yuze boyar.
8. Ikinci Fill Layer: beyaz detay cizgileri.
9. Roughness kanalinda ayni maskeye bagli daha mat pigment degeri verilir.
10. Layer visibility ac/kapat ile once/sonra gosterilir.

### Edge Wear / Dirt Ornegi

1. `Add Fill Layer`.
2. Kanal: Base Color + Roughness.
3. Fill: acik renk / toz veya koyu kir.
4. Mask Source: Curvature Convex veya Cavity.
5. `Generate Mask`.
6. Kullanici mask'i elle temizler veya guclendirir.
7. Opacity/blend mode ile sonucu ayarlar.

Bu workflow Substance hissi verir ama daha dogrudan ve daha az karmasiktir.

## Veri Modeli

Mevcut `PaintLayerData` genisletilmeli. Onerilen temel model:

```cpp
enum class PaintLayerKind : uint8_t {
    Raster = 0,
    Fill
};

enum class PaintMaskMode : uint8_t {
    None = 0,
    Manual,
    Generated,
    GeneratedPlusManual
};

struct PaintLayerMask {
    int width = 0;
    int height = 0;
    PaintMaskMode mode = PaintMaskMode::Manual;
    std::vector<uint8_t> pixels; // 0..255 alpha mask

    // Optional: generator metadata for regeneration.
    std::string generator_type;
    nlohmann::json generator_params;
};

struct PaintFillChannel {
    bool enabled = false;
    CompactVec4 value = CompactVec4(255, 255, 255, 255);
    std::shared_ptr<Texture> texture; // optional future support
};

struct PaintLayerData {
    PaintLayer meta;
    PaintLayerKind kind = PaintLayerKind::Raster;
    uint32_t id = 0;
    int width = 0;
    int height = 0;

    // Raster layer content.
    std::array<std::vector<CompactVec4>, kPaintChannelCount> channel_pixels;

    // Fill layer content.
    std::array<PaintFillChannel, kPaintChannelCount> fill_channels;

    // Shared layer mask.
    PaintLayerMask mask;
};
```

### Neden Shared Mask?

Ilk surumde katman basina tek mask daha anlasilir:

- Kullanici layer'in nerede gorunecegini boyar.
- Ayni mask Base Color, Roughness, Metallic gibi secili kanallari birlikte kontrol eder.
- UI daha sade kalir.

Gelecekte kanal basina mask eklenebilir; fakat ilk uygulamada karmasiklik artar.

## Composite Davranisi

`PaintLayerStack::compositeChannel()` su mantiga genisletilmeli:

1. Katman kapaliysa atla.
2. Kanal katmanda etkilenmiyorsa atla.
3. Katman `Raster` ise mevcut `channel_pixels[channel]` kullan.
4. Katman `Fill` ise aktif kanal icin `fill_channels[channel].value` veya texture sample kullan.
5. Katman mask'i varsa source alpha ile mask alpha carp.
6. Layer opacity uygula.
7. Blend mode uygula.

Pseudo:

```cpp
for (const PaintLayerData& layer : layers_) {
    if (!layer.meta.visible) continue;
    if (!layerAffectsChannel(layer, channel)) continue;

    for each pixel:
        CompactVec4 src;
        if (layer.kind == Raster) {
            src = layer.channel_pixels[channel][i];
        } else {
            src = layer.fill_channels[channel].value;
            src.a = applyMask(src.a, layer.mask.pixels[i]);
        }

        blendPixel(dst[i], src, layer.meta.opacity, layer.meta.blend_mode);
}
```

## Mask Boyama

Fill Layer aktifken firca iki modda calismali:

- `Paint Content`: Raster layer icin mevcut davranis.
- `Paint Mask`: Fill layer icin varsayilan davranis.

Fill Layer seciliyken Paint/Erase fircasi:

- Paint: mask alpha'yi artirir.
- Erase: mask alpha'yi azaltir.
- Soften: mask alpha'yi blur/relax eder.
- Fill: mask'i 0 veya 255 doldurur.

Bu sayede kullanici fill degerini degistirmeden sadece gorunurlugu boyar.

## Smart Mask Generator'lari

Smart Mask sistemi ilk surumde uc temel mask uretmelidir:

1. **Convex / Edge Mask**
   - Disa bakan keskin kenarlar.
   - Edge wear, kuru boya, asinan metal icin.

2. **Concave / Cavity Mask**
   - Iceri bakan cukur ve kir birikim alanlari.
   - Dirt, grime, pigment birikimi icin.

3. **Ambient Occlusion Mask**
   - Daha genis kapanma/golge alanlari.
   - Toz, kir, temas golgesi ve cavity shading icin.

Curvature tek bir gri texture olarak dusunulmemeli. Convex ve concave ayrimi kullaniciya daha temiz kontrol verir.

## Curvature Hesabi

Ilk pragmatik yontem:

- Mesh adjacency bilgisi editable mesh cache veya yeni helper ile kurulur.
- Her edge icin iki komsu face normal'i bulunur.
- Dihedral angle hesaplanir.
- Isaretli sonuc convex/concave olarak ayrilir.
- Vertex'lere agirlikli dagitilir.
- Sonuc UV texture'a bake edilir.

Yaklasik sinyal:

```cpp
float angle = atan2(length(cross(n0, n1)), dot(n0, n1));
float sign = dot(edgeDirection, cross(n0, n1)) >= 0 ? 1.0f : -1.0f;
float signedCurvature = angle * sign;
```

Not: Isaretin tutarli olmasi icin triangle winding ve edge yonu sabitlenmeli. Bu kisim test edilmeden production kabul edilmemeli.

### Kritik Noktalar

- Smooth normal yerine geometric face normal ile baslamak daha guvenilir.
- Hard edge ve UV seam ayrimi karistirilmamali.
- Degenerate triangle ve sifir alan UV atlas atlanmali.
- Sonuc texture'a bake edilirken island padding/dilation uygulanmali.
- Overlap UV varsa generator sonucu kullaniciya uyarili sunulmali.

## AO Mask Hesabi

AO generator iki yoldan biriyle baslayabilir:

### CPU Raycast AO

- Her sample texel veya vertex icin hemisphere ray atilir.
- Scene BVH veya selected mesh BVH kullanilir.
- 16/32/64 sample kalite secenekleri olur.
- Ilk surumde selected mesh local AO yeterlidir.

Avantaj:

- Kolay dogrulanir.
- Render backend'den bagimsizdir.

Dezavantaj:

- Yuksek cozunurlukte yavas olabilir.

### GPU / Vulkan-OptiX AO

- Daha hizli olabilir.
- Fakat backend entegrasyonu ve readback karmasiktir.

Oneri: Ilk surum CPU/local AO, sonra GPU hizlandirma.

## UV Bake ve Padding

Smart mask generator sonucunun paint layer mask'ine yazilmasi icin UV bake gerekir.

Gerekli adimlar:

1. Hedef mesh/material slot triangle'lari toplanir.
2. Aktif UV set belirlenir.
3. Her triangle UV uzayinda rasterize edilir.
4. Her covered texel icin barycentric koordinat hesaplanir.
5. Vertex/face curvature veya AO sonucu sample edilir.
6. Mask pixel'e yazilir.
7. Island disina dilation/padding uygulanir.

Padding zorunlu. Aksi halde mipmap ve bilinear sample seam kenarlarinda kirli goruntu uretir.

## UI Onerisi

Paint Layer paneline:

- `+ Raster Layer`
- `+ Fill Layer`
- Layer type badge: `Raster` / `Fill`
- Fill Layer seciliyken:
  - Channels: Base Color, Roughness, Metallic, Normal, Opacity...
  - Value editor: renk veya scalar slider
  - Mask controls:
    - Clear Mask
    - Invert Mask
    - Fill Mask
    - Generate Mask
  - Generator:
    - Convex Edge
    - Cavity
    - AO
    - Curvature Mix
  - Generator params:
    - Strength
    - Contrast
    - Blur
    - Min/Max threshold
    - Dilation

Brush Dock:

- Fill Layer aktifse varsayilan mode `Paint Mask`.
- UI'da net yazmali: `Editing: Layer Mask`.
- Mask preview overlay opsiyonel olmali.

## Serialization

`mesh_paint_layers` formatina yeni alanlar eklenmeli:

```json
{
  "version": 2,
  "layers": [
    {
      "id": 3,
      "kind": "fill",
      "name": "Edge Wear",
      "visible": true,
      "opacity": 0.8,
      "blend_mode": "overlay",
      "fill_channels": {
        "base_color": { "enabled": true, "value": [220, 210, 190, 255] },
        "roughness": { "enabled": true, "value": [230, 230, 230, 255] }
      },
      "mask": {
        "width": 2048,
        "height": 2048,
        "mode": "generated_plus_manual",
        "generator_type": "convex_edge",
        "generator_params": {
          "strength": 1.0,
          "contrast": 0.6,
          "dilation": 8
        },
        "binary_png": { "offset": 1234, "size": 5678 }
      }
    }
  ]
}
```

Backward compatibility:

- Eski layer'lar `kind = Raster` varsayilmali.
- `mask` yoksa tam opak kabul edilmeli.
- `fill_channels` yoksa raster davranisi bozulmamali.

## Undo/Redo

Yeni komut tipleri:

- `PaintLayerMaskCommand`
  - before mask pixels
  - after mask pixels
  - layer stack key
  - layer id

- `PaintFillLayerPropertyCommand`
  - channel enabled flags
  - fill values
  - opacity/blend/mask params degisiklikleri

Generator kullanimi:

- Generate Mask tek command olmali.
- Eger generated mask uzerine manuel boya yapiliyorsa manuel stroke ayri command olmali.

## Export / Bake

Export davranisi mevcut flat texture cikisina uyumlu olmali:

- Export edilen texture set, raster + fill + mask sonucunu flatten eder.
- Kullanici isterse fill layer mask'lerini ayrica PNG olarak export edebilir.
- Layered project dosyasi non-destructive kalir.

Bu ayrim onemli:

- Render/export icin flatten texture.
- Proje duzenleme icin non-destructive layer data.

## Performans ve Bellek

Fill Layer avantajlari:

- Tum kanal pixel buffer'i yerine tek fill degeri + mask tutabilir.
- 4K fill layer memory maliyeti RGBA channel buffer yerine 1 kanal mask olabilir.

Yaklasik:

- 4K RGBA raster channel: 4096 x 4096 x 4 = 64 MB
- 4K mask: 4096 x 4096 x 1 = 16 MB

Eger fill layer 3 kanali etkiliyorsa raster olarak 192 MB gerekecekken, fill+mask modeli 16 MB + kucuk metadata ile calisir.

Dirty update:

- Mask boyama dirty rect uretmeli.
- Composite sadece etkilenen region ve etkilenen kanallar icin calismali.
- Generator tum mask'i degistirdiginde full channel composite kabul edilebilir.

## Riskler

### UV kaynakli riskler

- Overlap UV generator sonucunu bozar.
- Stretch, curvature/AO mask'in texture dagilimini kotu gosterir.
- UV health warning ile birlikte gelmesi onemli.

### Curvature isaret riski

- Convex/concave ayrimi mesh winding, normal ve edge yonune hassastir.
- Test mesh seti olmadan dogru kabul edilmemeli.

### UI karmasasi

- Fill Layer, mask, generator, kanal secimi ayni anda fazla gelebilir.
- Ilk UI sade olmali; advanced parametreler collapsible section'da olmali.

### Serialization riski

- Eski project dosyalari bozulmamali.
- Version alanlari net tutulmali.

## Test Plani

### Unit / data tests

- Eski `PaintLayerStack` deserialize -> raster layer olarak yuklenir.
- Fill layer mask yoksa tam opak composite eder.
- Mask 0 ise hic etki etmez.
- Mask 255 ise fill value tam uygulanir.
- Opacity ve blend mode raster ile ayni davranir.

### Golden image / visual tests

- Plane uzerinde Base Color fill layer.
- Sphere uzerinde mask paint.
- Hard-edge cube uzerinde convex edge mask.
- Concave test mesh uzerinde cavity mask.
- UV seam olan mesh uzerinde dilation testi.

### Workflow tests

- Add Fill Layer -> Generate Mask -> Paint Mask -> Undo -> Redo.
- Fill value degistir -> composite guncellenir.
- Save project -> reopen -> layer ve mask ayni gelir.
- Export textures -> flatten sonuc dogru cikar.

## Uygulama Asamalari

### Asama 1 - Veri modeli ve composite

- `PaintLayerKind` ekle.
- `PaintLayerMask` ve `PaintFillChannel` ekle.
- `PaintLayerStack::compositeChannel()` fill layer desteklesin.
- Backward compatible serialize/deserialize.

Deger: Fill layer altyapisi gelir.

### Asama 2 - UI ve manuel mask boyama

- `+ Fill Layer` butonu.
- Fill layer property UI.
- Fill layer aktifken brush mask boyasin.
- `PaintLayerMaskCommand` ile undo/redo.

Deger: Non-destructive fill + manual mask kullanilabilir.

### Asama 3 - Curvature generator

- Mesh adjacency helper.
- Convex/cavity curvature bake.
- UV rasterize + padding.
- Generator UI.

Deger: Edge wear ve cavity dirt workflow baslar.

### Asama 4 - AO generator

- CPU local AO bake.
- Sample count / radius / contrast parametreleri.
- Mask'e yazma ve undo.

Deger: Daha dogal kir/golge maskeleri.

### Asama 5 - Polish ve pipeline

- Mask preview overlay.
- Mask export.
- Generator presetleri.
- UV health warning ile baglanti.
- Triplanar fallback planina uyum noktalarinin belirlenmesi.

Deger: Production-ready hissi.

## Karar Ozeti

1. Fill Layer ayri layer tipi olacak.
2. Fill Layer raster pixel degil, kanal degeri + mask olarak saklanacak.
3. Ilk surumde katman basina tek shared mask olacak.
4. Curvature generator convex ve concave'i ayri uretmeli.
5. AO generator ilk etapta CPU/local AO olabilir.
6. Generated mask, manuel fircayla duzeltilebilir olmali.
7. Export flatten texture verir; project dosyasi non-destructive kalir.

Bu mimari, Mesh Paint sistemini sadece "fircayla texture boyama" seviyesinden alip "akilli, katmanli PBR material painting" seviyesine tasir.
