# `scene.delete` SİLMEZ, GİZLER — ve anlık görüntü alan tüketici bunu kaçırır

> **Durum:** AKTİF — kök neden ölçüldü ve **düzeltildi** (2026-09-07);
> RayFusion tarafı kapandı, genel semantik sorusu açık.

★ Bu notun ilk sürümündeki kök neden teşhisi **yanlıştı** ve ikinci ölçümde
düzeldi. Yanlış teşhis burada bırakıldı, çünkü asıl ders o farkta.

## Belirti

`scene.delete` ile silinen bir nesne RayFusion'ın hızlandırma yapısında
**kalmaya devam etti**. Görüntüde hiçbir belirti yok: nesne çizilmiyor, sahne
listesinde yok, hata da yok.

## Ölçüm

Geçici bir küp eklenip silindi (iki ayrı sahnede, aynı desen):

| aşama | `list_objects` | raster `draw_calls` | raster `total_instances` | AS `blas_count` |
|---|---:|---:|---:|---:|
| önce | 60 | 67 | 67 | 67 |
| ekleyince | 61 | 68 | 68 | 68 |
| **silince** | **60** | **67** | **68** | **68** |

## Kök neden: silme bir GİZLEMEDİR

`DeleteObjectCommand::execute` hiçbir şeyi yok etmiyor. Yaptığı:

- nesneyi `markObjectPendingDelete` ile işaretlemek,
- `setSceneObjectVisibility(..., false)` ile **görünürlüğü kapatmak**.

`world.objects` girdiyi tutmaya devam ediyor; gerçek silme ancak
`compactPendingDeletedObjects()` çalıştığında oluyor. Backend tarafında bunun
karşılığı `setVisibilityByNodeName` → **`RasterInstance::mask = 0`**.

Bu yüzden:

- `g_scene_geometry_generation` **artmıyor** — ve bu **doğru**. Geometri
  değişmedi, yalnızca görünürlük değişti. Undo'nun anında olmasının bedeli bu.
- `m_rasterMeshes` mesh'i **tutuyor** (vertex tamponu duruyor).
- `m_rasterInstances` girdiyi **tutuyor**, `mask = 0` ile.
- Çizim döngüsü her yerde `if (ri.mask == 0) continue` diyor → `draw_calls`
  düşüyor, `total_instances` düşmüyor. İkisi de doğru; **farklı şey sayıyorlar.**

★★★ İlk teşhisimde "`m_rasterMeshes` mesh'i düşürüyor, `m_rasterInstances`
sarkan girdi tutuyor" yazmıştım. Sarkan girdi diye bir şey yok — girdi
**kasıtlı olarak** duruyor ve maskeli. Belirti aynıydı, mekanizma değil; ve
yanlış mekanizmadan türetilen "kuşağı silme de artırsın" önerisi undo'yu
tam yeniden kuruluma çevirirdi.

## Kapatılan

**Kapı.** `ensureRayFusionSceneAS` `m_rasterBuiltGeometryGeneration`'a bağlıydı.
Silme o sayacı haklı olarak artırmadığı için kapı **maske değişimini hiç
görmüyordu** — AS silinen nesneyi `mask = 0xFF` ile tutmaya devam ediyordu, yani
ışın onu **gerçekten görecekti**. Kapı artık iki içerik imzasına bağlı:

- `geometry_signature` — mesh kümesi. Değişirse bütün BLAS'lar yeniden kurulur.
- `instance_signature` — yerleşimler **ve maske**. Yalnız o değişirse **TLAS tek
  başına** tazelenir (`tlas_only_refreshes`).

**Sayaç.** Maskesi 0 olan instance TLAS'a hiç girmiyor ve `instances_hidden`
olarak raporlanıyor. Işın için maskeli bir slot ile hiç olmayan slot aynı şey;
ama `instance_count` için değil — izlenmeyen bir sahneyi raporlayan sayaç, bu
deponun en pahalı hata sınıfıdır.

**Politika, açıkça yazıldı.** `blas_count` bir **ikametgah** sayısıdır, sahne
içeriği değil: gizli mesh raster vertex tamponunu koruduğu için BLAS'ını da
korur. BLAS'ı görünürlüğe bağlamak, yerleşim-tazelemesinin artık var olmayan
bir BLAS'a atıf yapmasına yol açardı. Yani **silmede `instance_count` düşer,
`blas_count` düşmez** — ve bu bir arıza değil.

## Doğrulama (2026-09-07, aynı gün, düzeltme sonrası)

| aşama | `blas_count` | `instance_count` | `instances_hidden` | `builds` | `tlas_only` |
|---|---:|---:|---:|---:|---:|
| önce | 3 | 1 | 2 | 3 | 151 |
| ekleyince | **2** | 2 | 0 | 4 | 151 |
| silince | 2 | **1** | **1** | 4 | **152** |

Silmede `instance_count` düşüyor, `builds` artmıyor, yalnız TLAS tazeleniyor.

★ **Eklemede `blas_count` 3'ten 2'ye DÜŞTÜ.** Sebep silme değil: ekleme gerçek bir
geometri değişimi olduğu için tam raster yeniden kurulumu tetikliyor
(`destroyAllRasterMeshes` → yalnız görünür nesnelerden yeniden kurulum), ve o
kurulum **önceki iki gizli mesh'i tasfiye ediyor**. Yani gizli BLAS sonsuza kadar
durmuyor; ilk gerçek geometri değişiminde geri alınıyor. Bu, aşağıdaki 1. sorunun
da cevabı.

Yerleşim testi ayrıca geçti: bir nesneyi taşıyınca `builds` 4 → 4 sabit,
`tlas_only_refreshes` 152 → 153. Kapının kendi maliyeti `signature_ms` = 0,0009 ms
(küçük sahne; büyük scatter sahnesinde tekrar bakılmalı). `device_lost` = 0.

## Açık kalan: genel yol

1. ~~Maskeli girdi ne zaman temizleniyor?~~ **Cevaplandı:** bir sonraki tam raster
   yeniden kurulumunda (herhangi bir gerçek geometri değişimi). O ana kadar
   `total_instances` performans teşhisinde **gizlileri de sayıyor**.
   ★ Yan sonuç: araya bir ekleme girdikten sonra undo artık anında değil — mesh
   tasfiye edilmiş olur. Bu raster'ın önceden beri var olan davranışı.
2. `hasValidRasterCache` aynı kuşağa bakıyor. Silmeden sonra "önbellek geçerli"
   demeye devam eder — bu **doğru**, çünkü geometri gerçekten değişmedi; ama
   aynı sayaca bakan başka bir tüketici maskeye kör olursa aynı hatayı yapar.
3. Bu deste dışında maskeye kör bir tüketici var mı? Aranacak desen: geometri
   kuşağını okuyup `mask` okumayan her yer.

★ **Test dersi:** bu hata yalnız "ekle" ile test edilseydi **geçerdi** — ekleme
kuşağı artırır, silme artırmaz. Simetrik bir işlemi tek yönde test etmek kapıyı
değil yalnızca çalışan yönü doğrular.

İlgili: `RAYFUSION_RENDERER.md` (Adım 2), `NEXT_BUILD_CHECKS.md` madde 4.
