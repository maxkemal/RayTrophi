# Islak boya tuvali döşemeli duvarda (paint_base_tex)

> **Durum:** CANLI DOĞRULANDI — 2026-11-24 gece: Studio + Player tam/artımlı derleme ve kayıt/yükleme + `@base` kademeli açılma + RayFusion viewport doku değişimi doğrulandı.

### Kayıt (güncellendi 2026-11-24 gece)

- `WetPaintGpu::cleanBaseTextureForSave(material_id)`: `MaterialManager::serialize` ve
  `ProjectManager::serializeTextures` overlay, composite ve `@base` yüzeylerinde geçici şeffaf veya
  beyazlatılmış tuval yerine temiz taban dokusunu (varsa gerçek kaynak dokuyu, yoksa `base_original`/`base`
  temiz kopyasını) yazar.
- `MeshPaintAdapter::assignTextureToChannel`: `isGeneratedPaintTexture` yalnız tek renk / tamamen şeffaf
  (`isUniformOrEmptyPaintTexture`) tuvalleri sıfırlar; kayıtlı projelerdeki gerçek duvar/normal dokuları
  korunur.
- `WetPaintGpu::updateCanvasModes`: Materyal panelinden albedo dokusu değiştirilirse yeni dokuyu anında
  `s.base_texture`/`s.base`/`@base` içine alır; `VulkanBackend::uploadMaterials` yeni `vulkan_dirty`
  dokuları `compactMaterialTexturesLocked()` ile RayFusion 2D-dizi sayfalarına ekler.

## Sorun

GPU ıslak boya (`Paint/WetPaintGpu.cpp`) boyayı malzemenin **albedo dokusunun yerine** bağlanan bir
tuval sayfasına yazar; sayfa = taban + gizli resim + boya (`paint_wet.comp` `surfaceComposite`).
Renderer o sayfayı malzemenin UV dönüşümüyle (scale/offset/rotation/tiling) örnekler.

- `MeshPaintAdapter::bindTextureSetToMaterial` tuval bağlanınca dönüşümü kimliğe sıfırlar ve duvar
  kağıdını tuvale **bir kez gerilmiş** kopyalar (`cloneTextureForPaint`). Duvar kağıdı dev görünür.
- Kullanıcı panelden döşemeyi 12'ye çekince dönüşüm artık **tuvale** uygulanır: boya, gizli resim
  ve gerilmiş duvar kağıdı birlikte 12×12 tekrarlanır. Ölçüm (2026-11-24): resmin 209 265
  boyanabilir texel'inin duvarda ~1/141'i görünüyordu.
- ★ Sinsi yanı: hiçbir şey hata vermez; boya "küçük ve çok" görünür, resim "küçük" görünür.

Geçici çözüm döşemesiz duvar dokusuydu (`wallfull.png`). Bu not kalıcı çözüm.

## Karar: tuval örtüsü + ayrı taban yuvası

Malzemenin sıcak çekirdeğindeki (`Material`, binding 2, 160 B) boş `_core_pad0` sözcüğü
`paint_base_tex` oldu — **struct boyutu ve ofsetler aynı**, eski host 0 yazdığı için yeni shader
eski exe ile de aynı çizer.

| `paint_base_tex` | Anlam |
|---|---|
| 0 | Sıradan malzeme. Bugünkü yol **bit bit aynı**. |
| ≠ 0 | Tuval malzemesi. Bütün doku yuvaları (albedo, normal, roughness, height, opacity…) tuval sayfası, **ham mesh UV**'siyle örneklenir. UV dönüşümü **yalnız** `paint_base_tex`'e uygulanır. `albedo = taban.rgb × (1 − tuval.a) + tuval.rgb` (tuval önceden çarpılmış). |

Neden bu ve "bake" değil: tabanı döşenmiş haliyle tuvale pişirmek (CPU `PaintLayerStack`'in
yaptığı) renderer'a dokunmaz ama 1024'lük tuvalde 12× döşeme = tekrar başına 85 px; duvar kağıdı
bulanıklaşır ve her döşeme değişiminde yeniden pişirmek gerekir. Örtüde duvar kağıdı kendi
çözünürlüğünde kalır.

### Mod seçimi (otomatik, malzemeden)

`WetPaintGpu::updateCanvasModes` her tick, `addSurface` ekleme anında:

- **overlay** ⇔ yüzeyin altında bir **kaynak taban dokusu** var (`PaintTextureSet` kaynak dokusu,
  kendi `generated/` tuvalimiz değil) **ve** malzemenin dönüşümü kimlik değil **ve**
  `reveal_picture != "@base"`.
- Ekleme sırasında adaptör dönüşümü sıfırlamadan önce okunur ve overlay'de geri yazılır; overlay
  malzemede `bindTextureSetToMaterial` artık dönüşümü sıfırlamaz (`paintBaseTexture` varken).
- Panelden/IPC'den döşeme sonradan verilirse (asıl arıza senaryosu) yüzey overlay'e geçer; döşeme
  1'e dönerse composite'e döner. Geçiş o yüzeyin boyasını sıfırlar (taban rengi düzenlemesinin yolu:
  taban dilimi + `MODE_RESET`).
- Taban dokusu **olmayan** (düz renk) tuvalde dönüşüm yalnız boyayı döşeyebilir → kimliğe çekilir,
  `[WetPaint] ... reset to identity` uyarısı.
- `@base` gizli resim modunda taban zaten resmin kendisi, döşenecek taban yok → composite.

### Çekirdek tarafı (`paint_wet.comp`)

Yüzey bayrağı bit 5 (`32u`, `SURFACE_OVERLAY`): çıkış sayfası `overlayComposite` — resim sonra
boya, doğrusal ışıkta önceden çarpılmış, alfa = kapsama. Opak bir taban `b` üzerinde
`b(1−a) + rgb` tam olarak `surfaceComposite`'in `mix(mix(b, resim, ap), boya, a)`'sıdır.
Overlay'de taban düzlemi yalnız boya toplama (pickup) rengini taşır (taban dokusunun ortalaması,
alfa 0); ilk çıkış sıfır (şeffaf). Binding sayısı 17 ve push 64 B AYNI.

### Renderer tarafı

| Yol | Değişiklik |
|---|---|
| Raster `material_preview_frag` (+ `_covered`) | `uv` overlay'de ham, `paintBaseUV` dönüşümlü (tile-break tabana); albedo örtüsü; tuval alfası opaklık DEĞİL (taban alfası). |
| Vulkan RT `closesthit.rchit` | aynı kural, `materialUV`/`paintBaseUV`. |
| Opaklık: `material_preview_shadow_frag`, `rayfusion_rt_shadow.comp`, `shadow_anyhit.rahit` | overlay'de opaklık tuvali ham UV (`materialSlotUV`). |
| RayFusion prob GI (`RayFusionBounce.cpp`) | overlay'de albedo yuvası = taban (dönüşümlü); boya GI'a girmez (önceden çarpılmış tuval boyasız yerde siyah olurdu). |
| Host | `PrincipledBSDF::paintBaseTexture` (çalışma zamanı, kaydedilmez) → `MaterialData::paintBaseTexture` → `VkGpuMaterial` blok 26 (400→416 B, yalnız CPU) → çekirdek `paint_base_tex`; doku yükleme kuyruğu + `syncExistingTexture`. |

**Kapsam dışı (bilerek):** CPU render ve OptiX (dondurulmuş) tuvali hâlâ malzeme dönüşümüyle
örnekler → overlay duvarda boya orada döşenir. RayFusion GI'da opaklık tuvali dönüşümlü okunur
(döşemeli + `reveal_opacity` birlikte nadir). Kanal tuvalleri (roughness/height/normal) taban
kanal dokularını hâlâ bir kez gerilmiş tohumlar — döşemeli bir normal/roughness haritası overlay'de
dev görünür; ileride aynı desenle `paint_base_*` yuvaları gerekebilir.

### Kayıt

Overlay'de dönüşüm malzemede kalır, kayıtla birlikte yazılır; açılışta `addSurface` dönüşümü
bağlamadan önce okur → yine overlay. Composite'te davranış eskisi gibi.

## IPC / panel

- `paint.wet_status` → `surfaces[].canvas_mode` (`composite|overlay`), `surfaces[].base_texture`
  (Python aynı). Mod ayarlanmaz, malzemeden okunur: `material.set uv_scale_x/uv_scale_y` ile sürülür.
- Studio Game Design yüzey ayarlarında overlay iken salt-okur "Canvas over tiled base: …" satırı.

## Doğrulama

Sıralı kontrol listesi: `NEXT_BUILD_CHECKS.md` "Döşemeli duvarda tuval" bölümü.
