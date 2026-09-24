# Sıradaki derlemede kontrol edilecekler

## ★ SSS partisi (2026-09-24) — domain maddelerinden BAĞIMSIZ, önce bunlar

Shader'lar değişti: `bsdf_scatter.glsl`, `shadow_anyhit.rahit`,
`hair_shadow_anyhit.rahit`. **`.spv`'ler yeniden üretilmeli** — üretilmezse
aşağıdaki her madde "hiçbir şey değişmedi" diye FAIL eder ve bu eski shader'dır,
yeni kodun hatası değil.

1. **API round-trip** (render gerekmez) — `Probe-SssResponse.ps1 -Object <mesh>`
   ilk satırı `PASS roundtrip`. FAIL → `parseMaterialParam`/`writeMaterialValue`
   eşlemesi yanlış ya da eski exe.
2. **Enerji** — `energy sss/diffuse` ≈ 0,7–1,1. ~0,3 → çift kararma sürüyor
   (eski spv?). >1,3 → ağırlık pdf'e bölünmüyor, firefly beklenir.
3. **Radius tonu** — `R/G red-radius > neutral`. FAIL = radius hâlâ ton
   üretmiyor: kanal ağırlıkları çalışmıyor.
4. **IOR** — `IOR delta` > 0,5. ★ Sinsi olanı: küçük ama sıfır olmayan bir
   fark gürültüdür; iki render'ı gözle de karşılaştır.
5. **Görsel** — ince bölgeler (kulak/burun) arkadan ışıkta kırmızımsı
   geçirgen olmalı; yüzeyde **siyah benek** veya **parlak lekeler** olmamalı.
   Siyah benek → çıkış normali / nearest-hit any-hit hatası; parlak leke →
   rulet veya pdf bölmesi.
6. **Preset'ler** — Skin/Wax/Milk/Jade yeniden kalibre edildi (amount=1, SSS
   Color = yerleşik renk, IOR da ayarlanıyor). Rendered'da Skin ten rengi,
   Jade yeşil yarı saydam okunmalı. Aşırı açık/soluk → remap veya preset.
7. **RayFusion rengi** (`material_preview_frag.spv` yeniden üretilmeli) —
   Skin preset'inde realtime ve Rendered aynı genel tonda olmalı. Eskiden
   raster SSS Color'ı ~%5 gösteriyordu. Hâlâ base color görünüyorsa → eski spv.
8. **RayFusion eğrilik sarması** — kulak/burun/parmak gibi kıvrımlı yerlerde
   terminatörde kırmızımsı geçiş; **düz duvarda HİÇ** olmamalı (radius dünya
   biriminde). ★ Sinsi olanı: düz-shade (flat normal) mesh'te üçgen içinde
   eğrilik 0 → SSS görünmez ve bu "SSS kapalı" gibi okunur, hata değildir.
   Üçgen kenarlarında parlak çizgi → fwidth(normal) kenar sıçraması; bildir.
9. **SSS Method + Walk Max Steps** (Vulkan'a açıldı; eski `useRandomWalkSSS`
   / `sssMaxSteps` adları söküldü) — probe'un 5. ve 6. satırları:
   `64 vs 256 gap` < %3 ve `fast/walk` ≈ 0,8–1,2. Panelde Method=Fast iken
   slider gri olmalı. ★ Sinsi olanı: Max Steps'i 8'e çekmek hata vermez,
   SSS'i **sessizce karartır** — bu tasarım gereği (kesilen yürüyüş enerjisini
   kaybeder) ve yardım metni bunu söylüyor. Kaydet → aç sonrası iki değer
   korunmalı (`sssMethod`, `sssWalkMaxSteps` JSON anahtarları yeni).
   OptiX de artık aynı alanı okuyor: varsayılan tavanı 6 → 64 (daha yavaş,
   daha açık) — OptiX'te SSS farklı görünürse sebep bu.
10. **SDF/sıvı yüzeyde SSS** (yeni: `volume_closesthit` çıkış kancası) — süt
    veya bal sıvısına SSS'li materyal bağla (`material.set_param` materyal
    adıyla aynı SSS anahtarlarını alır). Rendered'da sıvı `subsurface_color`
    tonunda, ince uçlarda/dalga tepelerinde arkadan ışıkta parlamalı.
    **Kapkara / ölü mat yüzey** → çıkış kancası çalışmıyor (yürüyüşler rulete
    kadar gidip ölüyor, düzeltmeden önceki durum). **Işık giriş noktasında
    toplanıyor, yayılma yok** → `g_sssExited` ile yeniden oturtma devreye
    girmiyor. Method=Fast ile karşılaştır: renk aynı ailede, Fast'ta sızma yok.
    RayFusion'da SDF yüzey artık SSS rengini gösteriyor ama **ışık sızması yok**
    (bilerek; notta yazıyor) — bu bir hata değil.
    ★ Sinsi olanı: sıvı içinde duran bir kaşık/cisim — yürüyüş hangisi yakınsa
    oradan çıkar; kaşığın yüzeyinden çıkan ışık doğru, bu bir sızıntı değil.
11. **Performans** — SSS'li sahnede kare süresi eskisinden çok uzunsa: probe
   artık her adayı gezip en yakını tutuyor (TerminateOnFirstHit yok) ve
   adım tavanı 32→64. TDR varsa önce `MAX_STEPS`'i düşür.

---

> **Durum:** CANLI — 2026-09-24. Bu parti: **domain taşıma artık bake'i
> öldürmüyor.**
>
> Önceki partinin kamera maddeleri kullanıcı tarafından doğrulandı ve
> kapatıldı (`KAMERA_ODAK_KILIDI.md` arşiv notu). Bu liste yalnızca yeni
> davranışı kovalar.
>
> **Ne değişti:**
> 1. `rebaseRestoredGridDomainStates()` — cache'ten geri kurulan kare, domain'in
>    **canlı** kutusuna yeniden oturtuluyor. RAM cache'i ve disk bake'i aynı
>    çağrıdan (`setGridDomainStates`) geçtiği için tek yerde.
> 2. `acceptSimConfigAsBaked()` — taşıma yeni kurulumu "bake'in ait olduğu
>    kurulum" ilan ediyor, böylece kare döngüsünün auto-invalidate'i bir tik
>    sonra cache'i düşürmüyor.
> 3. Üç taşıma yolu da (gizmo / panel / `sim.move_domain`) artık **saf
>    ötelemede** cache'i koruyor, **yeniden boyutlandırmada** eskisi gibi
>    düşürüyor.
>
> Yeni `.cpp` **yok**, yeni IPC metodu **yok**, yeni shader yok.
> Probe script'i iki yere de kopyalandı.

---

## 0. ★★★★★ ASIL KAPI: taşıma bake'i öldürmüyor mu

Bu partinin tek sebebi bu. **Önce bir bake olmalı** — boş cache ile bu madde
hiçbir şey ölçmez.

1. Gaz domain'ini bir süre oynat (RAM cache dolsun).
2. `Invoke-RtIpc sim_cache.status @{}` → `ram_frames` sıfırdan büyük olmalı.
3. Sonra:

```powershell
.\scripts\ipc\Probe-DomainMoveCarriesSources.ps1
```

- **Ne görmen gerek:** `TUM KAPILAR GECTI`. Kapı 3 iki şey söyler: RAM kare
  sayısı **bir tik sonra da** aynı, ve baked imza yeni kutuya göre tazelenmiş.
- **`[ATLANDI] cache bostu`:** bu bir geçiş değil. Önce oynat, sonra tekrar
  çalıştır — yoksa partinin asıl kapısı hiç ölçülmemiş olur.
- **Bozuksa ne demek:** `ram_frames` sıfırlanıyorsa taşıma yolu hâlâ
  `clearSimFrameCache()` çağırıyordur (üç yer: `RtApiFluid.cpp`,
  `scene_ui_gizmos.cpp`, `scene_ui_simulation_domains.cpp`).
- **★★★ EN SİNSİ HÂLİ:** kare sayısının **hemen sonra** durup **bir saniye
  sonra** sıfırlanması. Cache'i düşüren şey taşıma kodu değil, kare döngüsünün
  kendi auto-invalidate'i — ve o bir sonraki tikte koşar. Probe bu yüzden
  ölçmeden önce bekliyor; elle bakarken sen de bekle.

---

## 1. ★★★★★ Gözle: taşınan domain eski yerine ZIPLAMIYOR mu

Asıl şikâyet buydu, ve script bunu göremez (ImGui overlay'i ekran görüntüsüne
girmiyor).

1. Bake'li bir gaz domain'ini gizmo ile kaydır.
2. Timeline'ı **cache'li aralığın içine** sürükle (scrub).

- **Ne görmen gerek:** Kutu, hacim ve duman **yeni** konumda. Tek bir mavi kutu
  var; geride kalan ikinci bir kutu **yok**.
- **Bozuksa ne demek:** Scrub'da eski yere zıplıyorsa `rebaseRestoredGridDomainStates()`
  ya çağrılmıyor ya da saf-öteleme kapısından düşüyor (extent 1e-3'ten fazla
  değişmiş olabilir — gizmo sürüklemesi aynı anda ölçeklemiş olabilir).
- **★★★ EN SİNSİ HÂLİ:** kutunun taşınıp **dumanın** taşınmaması. O zaman
  `bounds`/`origin` kaydırılmış ama `particles.position` kaydırılmamıştır —
  yani sıvı kolu eksiktir. Gaz domain'inde görünmez (gaz indeks uzayında),
  **sıvı domain'de dene.**

---

## 2. ★★★★ Yeniden boyutlandırma HÂLÂ cache'i düşürüyor mu — regresyon kapısı

Bu kapı olmadan madde 0 tehlikelidir: taşımayı korurken boyutlandırmayı da
korumak, bayat bir bake'i hiçbir belirti vermeden sonsuza kadar oynatır.

Bake'li bir domain'in gizmo tutamağından **ölçeğini** değiştir (taşıma değil).

- **Ne görmen gerek:** `sim_cache.status` → `ram_frames` **0'a düşmeli**.
- **Bozuksa ne demek:** `pure_translation` testi yanlış tarafa düşüyor. Eşik
  1e-4; gizmo ölçek matrisinden gelen extent gürültüsü bundan büyük olmalı.
- **★★★ EN SİNSİ HÂLİ:** cache'in durup **eski çözünürlükte** oynamaya devam
  etmesi. Duman makul görünür, sadece artık o kutuya ait değildir.

---

## 3. ★★★ Panel ve gizmo aynı şeyi yapıyor mu

Aynı domain'i bir kez gizmo ile, bir kez panelden `Domain Minimum/Maximum
Bounds` yazarak taşı.

- **Ne görmen gerek:** İki yolda da emitterler geliyor **ve** cache duruyor.
- **Bozuksa ne demek:** İkisi ayrışıyorsa panel yolunda `acceptSimConfigAsBaked()`
  çağrılmıyordur (`scene_ui_simulation_domains.cpp`, bounds_settled bloğu).
- ★ Panelde bounds'u **asimetrik** yaz (yalnızca `bounds_min`) — bu bir taşıma
  değil boyutlandırmadır, cache düşmeli. Panel bunu ayırt edebilmeli.

---

## 4. Yetki aynası ve descriptor tablosu — bağımsız, saniyeler

Bu partide yeni IPC metodu yok, yani bu bir **regresyon** kapısı.

```powershell
python scripts/audit_ipc_capabilities.py
```

- **Ne görmen gerek:** `OK - every dispatched method is classified, mirror
  agrees with RtIpcSecurity.cpp, no dead prefixes, descriptors current.`
- **Bozuksa ne demek:** `sim.` namespace'i geçen partide eklendi; burada
  patlarsa tablo bayattır → `python scripts/gen_ipc_descriptors.py`.

---

## 5. Sıvı domain'i taşı — en son, çünkü en pahalısı

Sıvı kolu gaz kolundan **ayrı kod**: partikül, foam ve UVW dizileri elle
kaydırılıyor.

- **Ne görmen gerek:** Taşınan sıvı domain'inde partiküller kutuyla birlikte
  gelmeli, yüzey dokusu **kaymamalı**.
- **★★★ EN SİNSİ HÂLİ:** partiküllerin gelip **dokunun kayması.** O zaman
  `uvw`/`uvw_b` kaydırılmamıştır. Malzeme koordinatı dünya çerçevesinde
  adreslenir (`uvw == position` dinlenen maddede); geride kalırsa doku sıvının
  üzerinde taşıma mesafesi kadar kayar — hata gibi değil, **sanat yönü gibi**
  görünür.

---

## Devralınan, bu partide DOĞRULANMADI

- **Izgara hacmin önünde** (`IZGARA_HACMIN_ONUNDE.md`) — ertelendi, dokunulmadı.
- **Domain döndürme/ölçekleme çapaları taşımıyor** — yalnızca öteleme ele
  alındı, bilerek.
- **Domain'e özgü cache invalidasyonu** — açık iş, gerekçesi
  `DOMAIN_TASIMA_EMITTERLERI_TASIMIYORDU.md` sonunda.
- 5 audit script'i içerik kaymasından kalıyor (`audit_material_coverage`,
  `audit_raster_material_visibility`, `audit_rayfusion_bounce`,
  `audit_rayfusion_probe_grid`, `audit_screen_gi`) — hepsi bu partiden önce de
  kalıyordu; en az biri yanlış alarm çıktı, kalanı da şüpheli.
