# 2026-10-11 parti 6: yağış perdeleri — zemin tabanı, bulut çerçevesinde eğim, hücre ayak izi, salkım, arazi önü

Önceki parti (gizmo derinlik testi) commit'li (`c23169d`). Bu parti iki turda
yazıldı, ikisi de DERLENMEDİ.

**Belirti (kullanıcı, canlı sahne):** RT ve RayFusion'da kamera y < 0'a inince
güneş yönünden bağımsız, çapraz gelen "god ray" benzeri şeritler; y > 0'da
neredeyse yok; yağış açılınca artıyor. Yağış perdeleri de kaba.

**Ölçüm (IPC, düzeltme öncesi):** kamera y = -3693, katman 0 tabanı 1200 m,
rüzgâr 33.9 m/s, yağış 0.5 mm/h. `rate_mm_h = 0` yapınca ufkun altındaki
şeritler tamamen kayboldu → kaynak yağış perdesi march'ı.

**Kök nedenler:**
1. Gök ışını için zemin tabanı yoktu (yalnız deniz seviyesi küresi). Zemin altındaki
   kamera aşağı bakınca 40 km kesintisiz yağmur görüyordu.
2. Eğim = rüzgâr / düşüş hızı ≈ 5.2 m/m → perde ~80° yatık. Hava haritası zaten
   rüzgârla sürükleniyor; düzgün rüzgârda perde bulut çerçevesinde DİKEY olmalı.
3. Ayak izi 12 km'lik hava lekesi, çizgiler 1 km × 18 km → kaba levhalar.

**Değişiklik:**
- `cloud_common.glsl` `cloudPrecipSample`:
  - düşüş çizgisi bulut çerçevesinde geriye izlenir (karesel, sürüklenen ayak);
  - ayak izi = katman 0'ın taban şekli, curl ile aşağı doğru açılıp saçaklanır;
  - salkım: sütun başına kademeli kopma derinliği (60–450 m) ve virga ucu
    (gf × 0.55–1.3);
  - çizgiler düşüş çizgisi boyunca iki oktav (420 / 110 m, ortalama ~1).
- `cloudSegmentMayContain`: optik derinliği 1e-3'e ulaşamayacak kısa segmentler
  (iç mekân sekmeleri) yağış march'ını atlar.
- `cloud_rt.glsl` `cloudPrecipMarch`:
  - yüzeye çarpan ışında perde yüzeyde biter (y < 0 vadiler dahil); y = 0 tabanı
    yalnız hiçbir şeye çarpmayan ışına uygulanır;
  - çizgi detayı adım boyuna göre söner (120–600 m).
- `CloudParams.h`: `precip[2..3]` = rüzgâra karşı gecikme / m düşüş, zeminde =
  0.4·U / düşüş hızı, ≤ 1.5.
- Arazi önündeki perde zaten iki yolda çağrılıyordu (`cloud_hit_segment.glsl`,
  `cloud_scene.comp`); "yalnız gök ışını" yorumu bayattı, düzeltildi.

Shader değişti → **`compile_shaders.bat` + C++ derlemesi** (CloudParams.h).

## Kontrol listesi (sırayla)

1. **Shader derleniyor mu** — `compile_shaders.bat` hatasız; özellikle `cloud_raster`,
   `cloud_scene`, `cloud_shadow`, `cloud_sample`, `miss`, `raygen`, `closesthit`.
   Bozuksa: hata satırını bildir.

2. **Zemin altı, yağış açık (ana belirti)** — kamera y ≈ -3700, sahnede arazi yok,
   RT ve RayFusion. Ufkun ALTINDA şerit olmamalı.
   Bozuksa: önce exe/spv zaman damgası, sonra `world.get_atmosphere` altitude.

3. **Zemin altı, yağış KAPALI** — ufkun altında güneş yönünde hafif ışık kalırsa
   AYRI konu (aerial/froxel). Bu partide dokunulmadı; bana söyle.

4. **Arazi önü (asıl istek)** — dağlı sahne, yağış 3–5 mm/h, kamera dağlara bakıyor;
   RayFusion VE RT final render. Görmen gereken:
   - perde dağın ÖNÜNDE asılı, yamaca değdiği yerde bitiyor;
   - perdenin arkasındaki yamaç soluklaşıyor;
   - tabandan sarkan tabanı saçaklı salkımlar;
   - neredeyse dik, aşağı doğru genişleyen perdeler, ince lifler.
   Bozuksa: perde yalnız gökte, dağın önünde yoksa RayFusion'da `cloud_scene`
   geçişinin çalıştığını doğrula (bulutlar dağa çöküyorsa çalışıyor) ve
   `rate_mm_h` > 0 olsun.

5. **Virga salkımı** — `precipitation.ground_fraction = 0.5`. Perdeler yarı yolda,
   FARKLI yüksekliklerde bitmeli (püsküllü uç). Hepsi aynı düz çizgide bitiyorsa
   salkım gürültüsü etkisiz.

6. **Kar** — `snow:true`, 1 mm/h. Daha eğik (≤ 56°) ama yere paralel değil.

7. **★ Sessizce makul görünen sonuç:** rüzgâr yönünü 180° çevir (`world.set_climate`,
   wind_direction [-1,0,0]). Ayak yine rüzgâra karşı tarafta kalmalı, yani yönle
   birlikte dönmeli. Aynı tarafta kalıyorsa işaret ters: perde yine "makul" görünür
   ama ayak bulutun önünde koşar.

8. **Perf** — yağışlı/yağışsız RT ve RayFusion kare süresi. İç mekân + yağış açık
   sahnede fark ~0 olmalı (kısa segment atlama). Açık arazide belirgin kayıp
   (> %15) varsa bildir. Adım başına maliyet arttı: taban şekli + salkım + 2 çizgi
   fetch'i.
