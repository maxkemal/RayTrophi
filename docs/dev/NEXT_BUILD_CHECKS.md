# 2026-10-13 (gece): sphere_closesthit, closesthit ile tek modül + şimşeğin ince yapısı + sahneye çarpma

Önceki derlemede doğrulananlar:
- bulut taşıması → `cloud_callable.rcall`: RT derlemesi 651 sn → ~3 dk;
  görüntü aynı; iki sahnede pass süresi aynı;
- yatay çizgili bant (kabuk kesişimi) — görüntü sorunsuz bildirildi.

## Bu derlemede
- **`sphere_closesthit.spv` artık yok.** Köpük/sıçrama küreleri
  (`sphere_intersection.rint`, binding 18) kendi hit grubunda
  `closesthit.spv`'yi kullanıyor.
  - Önce: `closesthit.rchit` `-DSPHERE_HIT=1` ile ikinci kez derleniyordu
    (3.0 MB SPIR-V, pipeline'da ikinci bir tam closest-hit).
  - Şimdi: iki grubu çalışma anında `gl_HitKindEXT` ayırıyor. Küre,
    `FOAM_SPHERE_HIT_KIND = 1` raporluyor; üçgenler 0xFE/0xFF.
  - Küre normali artık hit attribute'tan değil merkezden hesaplanıyor
    (kesişimle aynı aritmetik). Kesişim shader'ı attribute'u üçgen
    düzeninde (vec2) bırakıyor.
  - Malzeme VM'i tek çağrı: üçgende `objectRandom = -1`, yani
    `evalMaterialProgram`'ın önceden yaptığının aynısı.
  - Ortak tanım: `foam_sphere.glsl` (struct, binding 18, hit kind).
- `compile_shaders.ps1` eski `sphere_closesthit.spv`'yi siliyor (stale
  listesi).
- **Şimşeğin ince yapısı** (`cloud_lightning.glsl`, `CloudBoltGeometry.h`).
  - Sorun (canlı görüntü): yerden buluta 8–10 km'lik bir kanal 32 eşit
    parçaydı. Parça başına ~300 m düz çizgi düşüyordu; bulut tabanının
    altında 4–5 kırık kalıyordu ve hiç dal görünmüyordu.
  - Shader: kameranın büyük gördüğü parça, orta nokta kaydırmasıyla
    32'ye kadar alt parçaya bölünüyor (`cloudBoltRayFine`). Kırılma
    açısı her ölçekte sabit. Şekil parça kimliğinden üretiliyor, kareden
    kareye değişmiyor. Uzak şimşek eskisiyle aynı.
  - CPU: yere inen şimşekte dal köklerinin 3/4'ü bulut tabanının altındaki
    açık havadan seçiliyor (`clearAirBelowY`).
- **Şimşek sahneye çarpıyor** (`Atmosphere/LightningStrike*`,
  `lightning.attach_to_scene`, varsayılan açık).
  - Sorun (canlı): yere inen şimşek `ground_altitude_m`'ye (y = 0) kadar
    iniyordu. Dağa çarpan bir kanal kayanın içinden geçip devam ediyordu;
    dallar o tabana yassılaştığı için hepsi aynı noktanın çevresine
    "çarpıyordu".
  - Şimdi: "yuvarlanan küre" kuralı. Çarpma mesafesi (olaydan olaya
    45–140 m, akımdan) içindeki arazi, mesh ve scatter ağaç tepeleri
    taranıyor. Alçalan kürenin İLK dokunduğu nokta hedef oluyor.
  - Hedef olay başına bir kez seçiliyor, ömrü boyunca sabit kalıyor ve
    rüzgârla kaymıyor.
  - Dallar hedefin biraz üstünde (düşüşün %4'ü, ≥ 15 m) havada bitiyor.
  - Sahnede altında hiçbir şey yoksa `ground_altitude_m` kullanılıyor
    (panelde "Fallback strike altitude").
  - Ölçüm: `world.cloud_stats` → `lightning[i].strike` {kind, object,
    position, striking_distance_m, contact_height_m, candidates}.

---

## Kontrol listesi (sırayla)

1. **Shader derlemesi.**
   - Görmen gereken: hata yok; `x64/Release/shaders/sphere_closesthit.spv`
     yok.
   - Bozuksa: "foamSpheres redeclared" çıkarsa `foam_sphere.glsl` iki kez
     include ediliyor (koruma makrosu).

2. **RT pipeline derleme süresi.** Rendered'a geç.
   - Ölçüm: `render.rt_pipeline_status` → `compile_seconds` ve
     `stage_seconds`.
   - Görmen gereken: ~3 dk'dan biraz daha az; `stage_seconds` listesinde
     `sphere_closesthit` yok.
   - Not: toplam süreyi en büyük aşama (büyük ihtimalle
     volume_closesthit) belirlediği için düşüş küçük olabilir. Bu
     derlemede `stage_seconds`'ı oku; sıradaki adayı o seçecek.

3. **Köpük/sıçrama küreleri (göz).** Beyaz su / sıvı splat'ı olan bir
   sahne, Vulkan RT.
   - Görmen gereken: küreler öncekiyle aynı. Malzeme rengi, yuvarlak
     gölgelenme ve Object Info → Random farkı korunmalı.
   - Bozuksa:
     - küreler düz renk ya da hep yukarı bakan normalle aydınlanıyorsa
       hit kind eşleşmiyor (`FOAM_SPHERE_HIT_KIND` iki tarafta aynı mı);
     - küreler yanlış malzemeyle çıkıyorsa küre isabeti üçgen yoluna
       düşüyor ve `instances.i[gl_InstanceID]` okunuyor.
   - ★ Sinsi olan: küre isabeti üçgen sayılırsa da bir şey çizilir, çünkü
     instance/geometri okuması sıfır adreslerde "veri yok" yoluna düşer.
     Yani görüntü bozuk değil ama yanlış malzemeli olur. Malzeme rengine
     bak.

4. **Üçgen sahneler (göz).** Herhangi bir normal sahne.
   - Görmen gereken: değişiklik yok. Prosedürel malzemelerde Object Info
     → Random aynı.

5. **Şimşek, yakın plan (göz).** Fırtına sahnesi, kamera yerde, şimşeğe
   1–3 km. Zaman çubuğuyla bir çakmanın üstünde dur.
   - Görmen gereken:
     - kanal metre ölçeğinde kırık; uzun düz cetvel parçaları yok;
     - yere inen şimşekte bulut tabanının altında dallar var;
     - zamanı ileri/geri oynatınca şekil aynı kalıyor;
     - büyüyen uç, uzayan bir çizgi değil kırık yol boyunca ilerliyor.
   - Bozuksa:
     - kanal titriyor ya da her karede şekil değişiyorsa tohum hareket
       eden bir değere bağlı (parça konumu rüzgârla kayıyor; tohum
       birth/completion'dan gelmeli);
     - kırıklar kanaldan kopuk duruyorsa düğüm uçları eşleşmiyor
       (j = n son noktası parçanın ucu olmalı);
     - uzaktaki şimşek değiştiyse seviye seçimi hep > 0 (açısal boyut eşiği).
   - ★ Sinsi olan: maliyet. Yalnızca kanalın yakınındaki pikseller ince
     yolu öder. Çakma anında kare süresi belirgin sıçrıyorsa zarf testi
     (`envelope + width*30`) fazla geniş.
   - Notlar:
     - Bu sahnede `upward_fraction = 1`, yani bütün şimşekler yerden
       yukarı. Yukarı giden şimşeğin dalları gerçekte de üstte, bulutun
       içinde olur. Dal kontrolü için `upward_fraction`'ı 0'a çek.
     - Her iki yol da (Vulkan RT raygen ve RayFusion `cloud_scene.comp`)
       aynı fonksiyonu çağırıyor.

6. **Şimşek sahneye çarpıyor (dağ/ağaç).** Aynı fırtına sahnesi.
   - Hazırlık: `upward_fraction` 0, `ground_strike_fraction` ~0.3.
     Bu sahnede ikisi de 1'de: her çakma yerden yukarı ve hepsi yere iniyor.
   - Ölçüm: çakma anında `world.cloud_stats` → `strike.kind` (terrain |
     mesh | scatter) ve `position.y` arazinin o noktadaki yüksekliği
     (`scene.raycast` ile karşılaştır).
   - Görmen gereken:
     - kanal dağın yüzeyinde bitiyor, içine girmiyor;
     - dallar havada sönüyor, aynı noktanın çevresine toplanmıyor;
     - ağaçlı yamaçta bazen bir ağaç tepesine çarpıyor
       (`kind = scatter`).
   - Bozuksa:
     - `strike` hep null: çözücü kurulmamış (Main.cpp,
       `installLightningStrikeResolver`) ya da şimşek yere inmiyor;
     - kanal havada bitiyorsa hedef bir mesh'in sınır kutusu ya da yanlış
       scatter yüksekliği (`local_bbox.max.y - mesh_center.y` × ölçek);
     - hedef her karede zıplıyorsa önbellek anahtarı değişiyor
       (geometri nesli her karede farklı).
   - Takılma (canlı, 2026-10-14, düzeltildi): ilk sürüm bazı karelerde
     saniyelerce donuyordu. Sorgu, BVH ıskalayınca bütün üçgenleri tarayan
     seçim yolunu kullanıyordu. Şimdi yalnız BVH + TerrainManager.
     Görmen gereken: yeni şimşek doğarken kare takılmıyor.
     Bozuksa: takılma sürüyorsa scatter taraması (çok büyük grup) ya da
     her karede değişen geometri nesli (önbellek hiç tutmuyor).
   - ★ Sinsi olan: dağın yamacında, zirve çarpma mesafesinin dışındaysa
     şimşek yamaca çarpar. Bu doğru davranış; her şimşek zirveye gitmez.

## Bilinen
- Sıradaki derleme yükü adayı: volume_closesthit
  (`docs/dev/RT_SHADER_COMPILE_LOAD.md`).
- Sis/yağış görsel kontrolleri hâlâ bekliyor (önceki listedeki madde 6).
- Ayrı hatalar (görev kartlarında):
  - kapanışta `rtipc_transport` atexit `std::terminate`;
  - `viewport.render_frames` CPU renderer'ını sürüyor.
- `audit_ipc_capabilities.py` `audio.*` uyarıları paralel ajana ait.
