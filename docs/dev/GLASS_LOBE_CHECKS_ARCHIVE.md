# Cam lobu — önceki build kontrol listesi

> **Durum:** ARŞİV — 2026-09-06 kontrol listesinin korunmuş kopyası; testlerin geçtiği anlamına gelmez. Aşağıdaki derleme/ölçüm beyanları önceki partiden aynen korunmuştur.

---

# Bir sonraki build — Cam: specular yansıma lobu (Vulkan RT)

> **Durum:** CANLI — yazıldı, **shader'lar glslc ile sözdizimi doğrulandı**,
> çalıştırılarak doğrulanmadı (2026-09-06). Bu parti tek bir belirtiden çıktı:
> "Vulkan RT'de cam malzemede specular bir yansıma lobu gözleyemedim."

Kök neden ve gerekçe:
[BUG_GLASS_LOBE_CANNOT_SEE_ANALYTIC_LIGHTS.md](BUG_GLASS_LOBE_CANNOT_SEE_ANALYTIC_LIGHTS.md).

**Bir cümlede:** cam lobu NEE'yi atlıyordu (doğru) ve sahne ışıklarının TLAS'ta
geometrisi yok (doğru) — ikisi birlikte camın ayna lobunu **yapısal olarak
ulaşılamaz** kılıyordu. Zayıf bir lob değil, **hiç olmayan** bir lob.

## Ne yapıldı

| İş | Dosya |
|---|---|
| Su'nun özel estimator'ı paylaşılan servise dönüştü: `addWaterV3DirectLighting` → `addDielectricDirectLighting` (`foamCoverage` → genel `diffuseAlbedo`+`diffuseWeight`, `originPush` eklendi) | `source/shaders/bsdf_scatter.glsl` |
| Üçgen cam lobu artık doğrudan ışığı örnekliyor (ön yüz, `scatterGlass`'tan önce) | `source/shaders/closesthit.rchit` |
| Sıvı izoyüzeyine bağlı transmissive materyal için aynısı (bandın kendi çıkış itmesiyle) | `source/shaders/volume_closesthit.rchit` |
| ★ OptiX'te ayrı bir hata: `ggx_glass_effective_normal` **yön** döndürüp normal olarak kullanılıyordu → pürüzlü cam iki kez yansıyordu. Artık yarı vektör (`ggx_glass_micro_normal`) | `source/src/Device/material_scatter.cuh` |
| Ölçü aleti: siyah dünya + tek lamba + **metal kontrol küresi** | `scripts/probe_glass_specular_lobe.py` (+ `x64/Release/scripts/` kopyası) |

Yeni `.cpp` yok, vcxproj'a dokunulmadı. Yeni IPC metodu yok — yeni bir parametre
açılmadı, mevcut `material.*` + `render.probe` yüzeyi bu işi ölçmeye yetiyor.

---

## §0 — ★★★ Shader'ları derle (bu partinin EN OLASI hata sınıfı)

**Yap:** `compile_shaders.bat`. Sonra CUDA/OptiX için `compile_ptx.bat`.

**Ne görmen gerek:** hepsi OK. Burada glslc ile `closesthit`, `volume_closesthit`,
`sphere_closesthit`, `raygen`, `photon`, `hair_closesthit` temiz derlendi — ama
**geçici bir çıktı klasörüne**, yani depodaki `.spv` dosyaları HÂLÂ ESKİ.

★ **Bozuksa ne demek:** `addDielectricDirectLighting` üç dosyaya birden
derleniyor (`closesthit`, `volume_closesthit` ve dolaylı olarak diğerleri) ve
imzası bu partide değişti. Eksik/fazla argüman hatası varsa bir çağıran
güncellenmemiş demektir.

★★ **En sinsi hâli burada değil, §1'de:** `.spv` yeniden üretilmezse uygulama
**eski shader'ı yükler ve hiçbir şey şikâyet etmez** — düzeltmeyi "işe yaramadı"
diye raporlarsın. Belirti aynen düzeltmeden önceki gibi görünür.

## §1 — ★★★ En hızlı bağımsız kontrol: aletin kendisi

**Yap:** uygulamayı aç, sonra:

```powershell
.\scripts\ipc\Start-RayTrophi.ps1        # "HAZIR" bekle
python scripts/probe_glass_specular_lobe.py
```

**Ne görmen gerek:** `RESULT: PASS`, ve üç ölçüm satırında:
- `metal` satırı parlak (`max` belirgin biçimde > 0),
- `glass` satırı **artık siyah değil** (`max` > 0.01),
- `glass/metal` oranı **1.0'ın altında**.

**Bozuksa ne demek — belirtiden sebebe:**

| Belirti | Sebep |
|---|---|
| `FAIL(setup): ... render.probe reports no captured frame` | Ölçüm hiç koşmadı. `viewport.capture` açık ama kare gelmedi — sahne/backend sorunu, cam ile ilgisi yok |
| **`control (metal) is lit` FAIL** | ★ Aşağıdaki hiçbir satır anlam taşımaz. Lamba, kamera ya da viewport hatalı; camı suçlama |
| `glass shows a specular lobe` FAIL, kontrol parlak | Düzeltme ulaşmadı. **Önce §0'a dön** — en olası sebep eski `.spv`. Sonra: `takeGlassLobe` dalına mı girmiyor (transmission gerçekten 1 mi), yoksa `surfaceFrontFace` mi false |
| `lobe is dielectric-plausible` FAIL (oran > 1) | Estimator iki kez ekleniyor ya da ağırlığı yanlış. Cam, aynadan parlak olamaz |
| `nan_fraction` > 0 | Yeni yol geçersiz piksel üretiyor; büyük ihtimalle sıfıra bölme değil, çok yakın bir noktasal ışıkta `1/d²` patlaması |

## §2 — Gözle: pürüzlülük süpürmesi

**Yap:** cam bir küre + bir nokta ışık. Roughness'i 0.0 → 0.1 → 0.3 → 0.6 gezdir.

**Ne görmen gerek:** parlak nokta roughness ile **genişleyip sönümleniyor**.
0.0'da küçük ve sert (estimator tabanı 0.02'ye kırpar, yani delta lob yerine
çok dar bir lob görürsün — bilinçli).

★★ **Sinsi başarısızlık — kimse bunu bug diye raporlamaz:** parlak nokta var ama
**roughness'ten etkilenmiyor**. O zaman estimator koşuyor, ama GGX terimi
yanlış normalle besleniyor demektir; görüntü "makul" görünür ve öyle kalır.

## §3 — ★★ Çift sayım / enerji: cam aynadan parlak olamaz

**Yap:** yan yana iki küre — biri metal (roughness aynı), biri cam. Aynı lamba.

**Ne görmen gerek:** camın parlak noktası metalinkinden **belirgin biçimde
sönük**. IOR 1.5 dik geliş açısında ~%4 yansıtır.

**Bozuksa ne demek:** ağırlık hatası. Cam dalına `transmission` olasılığıyla
giriliyor ve `1/p` telafisi **uygulanmıyor**; estimator o yüzden ağırlık 1 ile
çağrılıyor. İkinci bir yerden daha çağrılıyorsa (ör. inclusion dalı) iki katına
çıkar.

## §4 — Su regresyonu (paylaşılan servis bu partide değişti)

**Yap:** Water V3 yüzeyi olan bir sahne aç, lambalı. Köpüklü bir bölge varsa
onu da gör.

**Ne görmen gerek:** su **eskisiyle aynı**. Parlaklık, köpük görünümü, hepsi.

★ **Neden burada:** camın kullandığı estimator su'nunkinin ta kendisi;
imzası ve foam parametresi genelleştirildi. Su bozulduysa hata camda değil,
`diffuseAlbedo`/`diffuseWeight` çevirisindedir — eski kod foam rengini
fonksiyonun İÇİNDE kuruyordu, artık çağıran veriyor.

★★ **Sinsi hâli:** köpük *biraz* farklı görünür ve "zaten stokastik" diye
geçersin. `mix(vec3(0.72,0.76,0.75), vec3(0.98), foam)` ifadesi bire bir aynı
olmalı — farklıysa çağıran tarafa taşırken bozulmuştur.

## §5 — Sıvı izoyüzeyi (aynı boşluk oradaydı)

**Yap:** bir fluid domain'e **transmission'ı 1 olan bir materyal bağla**
(Water V3 yolu değil — scene materyali). Lambalı sahne.

**Ne görmen gerek:** sıvı yüzeyinde de parlak nokta var, ve yüzey
**kendi kendini gölgelemiyor**.

**Bozuksa ne demek:** gölge ışını level-set bandının içinden başlıyor.
Buradaki çağrı `originPush = exitPush` veriyor ve `seatOutsideBand`'in dışa
giden dalıyla aynı şeyi yapıyor (`hitPos + L*push`) — sıvı üzerinde lekeli/
noktalı karartma görürsen bu eşleşme bozulmuştur.

## §6 — OptiX: pürüzlü cam (mikrofaset normali)

**Yap:** OptiX backend'iyle roughness ≈ 0.2 bir cam objeyi render et.

**Ne görmen gerek:** yansıma **Vulkan ile aynı yöne** gidiyor. Eskiden ışın iki
kez yansıtılıyordu, yani yansıma kabaca geldiği yöne dönüyordu.

★ **Bozuksa ne demek:** bu bir *görünüm değişikliği*, regresyon değil. Eski
görüntüye alışıksan yeni hâli yanlış gelebilir; ölçüt Vulkan ile eşleşmesidir.
★★ **Bu maddede cam parlak noktası BEKLEME:** OptiX'te NEE boşluğu bilerek açık
bırakıldı (kural 6, Vulkan birincil). Lambalı bir sahnede cam Vulkan'da parlar,
OptiX'te parlamaz — **bilinen ve kabul edilmiş fark**, bug olarak raporlama.

## §7 — Maliyet

**Yap:** camlı bir sahnede `rt.perf` / viewport süresine bak, camsız haline göre.

**Ne görmen gerek:** cam ön yüz vuruşu başına **bir ek gölge ışını** kadar fark.
Cam ağırlıklı bir sahnede ölçülebilir, kabul edilebilir olmalı.

**Bozuksa ne demek:** iç yüzlerde de koşuyor. Çağrı `surfaceFrontFace` ile
kapılı; kapı düşmüşse cam gövdesinin içindeki her sekme boşa bir gölge ışını
harcar (ve o ışın zaten kabuğa çarpıp bloke olur).

---

## Devralınan, HÂLÂ doğrulanmamış (önceki parti)

Assimp sökme partisinin import yolu kullanıcı tarafından doğrulandı; **import
yolu dışındaki maddeler açık kaldı** ve o dosya bu partide üzerine yazıldı.
Kapandı saymayın — ayrıntı için `git show HEAD:docs/dev/NEXT_BUILD_CHECKS.md`:

- **§3** Varlık tarayıcısı: FBX/OBJ metadata artık `probeModel` ile; sayılar dolu
  ve tarama belirgin biçimde hızlı olmalı.
- **§4** OptiX render: `buildOptixMaterialTables` **boş üçgen listesiyle de**
  çağrılmak zorunda; çağrılmazsa CUDA 700.
- **§5** Gömülü dokulu GLB/FBX: dokular geliyor mu (harici doku bunu ölçmez).
- **§6** `assimp-vc143-mt.dll` gerçekten kopyalanmıyor mu.
- **§7** Arazi `MacroColorMap` + `TerrainSemanticMap`: `Texture pointer null,
  skip` uyarısı kalkmış olmalı.

Kalıcı borçlar zaten [IMPORT_EXPORT_OPEN_DEBTS.md](IMPORT_EXPORT_OPEN_DEBTS.md)
içinde; yukarıdakiler onlar değil, **yapılmamış doğrulamalar**.

## Bu partide KAPANMAYAN

- **OptiX NEE boşluğu.** Transmission lobu `is_specular = true` yazıyor ve
  `ray_color.cuh`'daki NEE `!is_specular` ile kapılı. Vulkan birincil olduğu için
  estimator portu ertelendi — ama sonucu iki backend arasında **görünür bir
  ayrışma**, sessiz değil (§6).
- **Kısmi transmission karışım sözleşmesi.** `transmission < 1`'de taban katmanı
  `1/(1-t)` telafisiyle **tam ağırlık** alır, cam lobu ise `t` ağırlığında kalır;
  yani toplam specular `spec_taban + t·spec_cam`. Bu depoda hâlihazırda böyle
  (OptiX de aynısını yapıyor) ve bu partide değiştirilmedi. Denetlenmedi.
