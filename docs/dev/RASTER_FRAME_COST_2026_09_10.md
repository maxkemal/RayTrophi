# Raster kare maliyeti — 2026-09-10 ölçümü ve sıradaki hedefler

> **Durum:** AKTİF — aşağıdaki sayılar değişiklik öncesi canlı IPC ölçümleridir.
> İlk kaynak partisi: [Raster kalite temeli](RASTER_QUALITY_FOUNDATION_2026_09_10.md)
> (foliage bounce/coverage ve ölçü aleti düzeltmeleri). Yeni derleme, görüntü kabulü
> ve süre ölçümü bekliyor; AO/yumuşak gölge/LOD henüz uygulanmadı.
> Öncesi: [HANDOFF_2026_09_09_RASTER_FRAME_COST.md](HANDOFF_2026_09_09_RASTER_FRAME_COST.md) (ARŞİV),
> [MATERIAL_COVERAGE_PIPELINE_2026_09_10.md](MATERIAL_COVERAGE_PIPELINE_2026_09_10.md).

---

## 0. Tek cümlelik durum

**Hız tarafı:** kare **324 ms'ten ~81 ms'e** indi ve bunu yapan şey cutout'un
kendisi değil, cutout'un **derinlik ön geçişinin önünü açması**: ön geçiş bugün
19,4 ms'e mal olup **162 ms kazandırıyor** (2026-09-08'de "net negatif,
varsayılan kapalı" diye kaydedilmişti — o kayıt artık yanlış, §3). Kalan 81 ms
**iki eşit yarıdır**: üçgen debisi ~39 ms ve piksel başına gölgelendirme ~38 ms.

**Görüntü tarafı:** raster ile RT arasındaki açık ölçüldü — yakın/uzak parlaklık
oranı **1,18'e karşı 2,02**, zeminde ara ton payı **%1,8'e karşı %33,5** (§8).
İkisinin de tek sebebi var: ambient **sabit bir terim** (kapanmayı görmüyor) ve
gölge **1 ışın, ikili, filtresiz**. Probe GI tüketicisi **bağlı ve çalışıyor** —
iki runtime ayarı yüzünden sessizdi, düzeltilince görüntü 369 kat gürültü üstü
değişti ve **bedavaya** (§7). Ama kapanmayı tek başına açmak zemini **siyaha**
sürüyor: kapanma ile geri konan ışık aynı partide gitmeli (§7d).

**Bütçe var:** gölge terimi 2,3 ms, RT ise örnek başına 61 ms — yani bu sahnede
pahalı olan yol raster, ışın değil (§8b).

---

## 1. Ölçüm yöntemi

`viewport.reset_frame_timings` → kamerayı N kez yaz → `viewport.frame_timings`.
Raster viewport **yalnızca kirliyken** çizer, o yüzden kare sayısı = kamera
yazması sayısı. `viewport.render_frames` bu iş için **kullanılamaz**: path
tracer'ı sürer.

Sahne: 1680x945 (1,587 Mpx), **30,1 M görünür üçgen**, 10 030 instance,
34 draw call, 1 directional ışık, hacim yok. Preset **balanced**, shading
**material**, prepass açık, RT gölge açık.

★ İki tabloyu karşılaştırırken `applied.visible_triangles` eşit değilse
karşılaştırma geçersizdir. Tekrarlanabilirlik ölçüldü: aynı ayarla arka arkaya
iki pencere **82,00 / 81,39 ms** (±%0,8). Bunun altındaki farklar gürültüdür.

---

## 2. Ölçülen kare

| aşama | GPU ms | pay |
|---|---|---|
| **main_pass** | **56,6** | %70 |
| **depth_prepass** | **19,6** | %24 |
| rt_shadow | 2,3 | %2,8 |
| transmission | 1,6 | %2,0 |
| sky / gpu_cull / post / transmission_prep | ~0,3 | %0,4 |
| overlay | 0,002 | ~0 |
| **shadow_atlas** | — | **SKIPPED** (6 cascade devredildi) |
| volume_sdf | — | SKIPPED |

CPU **0,63 ms/kare**. Tamamen GPU-bound.

---

## 3. ★★★★★ Derinlik ön geçişi: "net negatif" kaydı ARTIK YANLIŞ

[RAYFUSION_RT_SHADOWS_AND_TRANSPARENCY.md](RAYFUSION_RT_SHADOWS_AND_TRANSPARENCY.md)
§6c ön geçişi **~22 ms ekliyor, hiçbir şey kazandırmıyor** diye ölçmüş ve
varsayılanını kapatmıştı. Aynı not iki hipotezi ayıramadığını da yazmıştı:
(1) elenecek gizli fragman yok, (2) `discard` var ama `early_fragment_tests`
beyanı yok, sürücü erken-Z **reddini** kapatmış olabilir. Ve ayıracak deneyi de
o not önermişti.

Deney koşuldu (`material_preview_frag.frag` artık `layout(early_fragment_tests) in;`
beyan ediyor, `material_preview_covered.spv` EQUAL + depth-write OFF varyantı
kuruluyor). Sonuç, RT gölge KAPALI iken iki kollu A/B (tek değişen ön geçiş):

| | depth_prepass | main_pass | kare GPU |
|---|---|---|---|
| prepass **AÇIK** | 19,4 | **57,9** | 154,5 |
| prepass **KAPALI** | — | **219,8** | 299,8 |

**19,4 ms'e mal olup 162,0 ms kazandırıyor.** Hipotez 2 doğru çıktı: overdraw
vardı (~4-5x), onu gizleyen şey erken-Z reddinin kapalı olmasıydı. §6c'nin
"overdraw azaltma bir strateji olarak elenir" sonucu **geçersizdir** ve o notta
düzeltildi.

⚠ **Bunu bir daha sökmeyin.** Ön geçiş bugün bu karedeki en büyük tek
optimizasyondur ve ayrıca RT gölge maskesinin ışın başlangıcı için ön koşuldur.

### 3b. `viewport.set_raster_depth_prepass(false)` SESSİZCE yutuluyor

[VulkanViewportBackend.cpp:3948](../../RayTrophiStudio/source/src/Backend/VulkanViewportBackend.cpp#L3948):

```cpp
const bool depthPrepassActive =
    useMaterialPreview && (m_rasterDepthPrepassAllowed || rtShadowFrame) && ...
```

RT gölge açıkken ön geçiş **zorunludur**. Canlı doğrulandı: `set_raster_depth_prepass false`
sonrası `viewport.raster_depth_prepass` → `enabled:false`, ama
`frame_timings.applied.depth_prepass` → `true` ve aşama 48 kare boyunca 25,8 ms
harcamaya devam etti.

Telemetri dürüst, **setter yalancı**. Yapılacak: `viewport.raster_depth_prepass`
`{enabled, forced_by_rt_shadow, effective}` döndürmeli. Bugünkü haliyle bu kol
ile yapılacak her A/B, RT gölge açık unutulduğunda **iki özdeş kolu** ölçer ve
"fark yok" der.

---

## 4. Kare nereye gidiyor — üçgen sayısına göre doğrusal ayrıştırma

Kalite preset'i ile üçgen sayısı değiştirilip üç nokta alındı (aynı kadraj):

| preset | üçgen | depth_prepass | main_pass |
|---|---|---|---|
| performance | 21,29 M | 15,64 | 49,90 |
| balanced | 30,11 M | 19,56 | 56,63 |
| full (proxy yok) | 49,84 M | 29,89 | 72,60 |

Uç noktalardan uydurulan doğru, **orta noktayı ölçmeden tahmin ediyor**:

```
main_pass     = 32,97 ms + 0,795 ms x Mtri     (30,11 M -> tahmin 56,91 / ölçülen 56,63)
depth_prepass =  5,00 ms + 0,499 ms x Mtri     (30,11 M -> tahmin 20,03 / ölçülen 19,56)
```

Dördüncü nokta bağımsız olarak eğimi doğruluyor: **solid** shading (materyal yok,
ön geçiş yok, gölge yok) **66,89 M üçgeni 31,29 ms'te** çiziyor →
**0,468 ms/Mtri**. Ön geçiş 0,499 ile bu tavanın %6 üstünde, yani
**ön geçiş zaten donanım debisinde koşuyor** — oradan shader ile alınacak bir
şey yok, yalnızca daha az üçgen ile.

### 30 M üçgende karenin gerçek bölünmesi

| kalem | ms | pay |
|---|---|---|
| **Geometri debisi** (iki geçiş, 1,294 ms/Mtri) | **38,9** | %48 |
| **Sabit gölgelendirme** (main 33,0 + prepass 5,0) | **38,0** | %47 |
| RT gölge | 2,3 | %3 |
| transmission + diğer | 1,9 | %2 |

Sabit terim, üçgen sayısından bağımsız olan kısımdır: bir kez gölgelendirilen
1,587 M piksel → **piksel başına ~21 ns**. Overdraw ön geçişle zaten kesildiği
için bu, gerçek tek-geçişlik gölgelendirme maliyetidir.

### 4b. Bağımsız ikinci tur

Aynı ayrıştırma `scripts/ipc/Probe-GeometrySlope.ps1` ile, **farklı kadrajda ve
farklı kare sayısıyla** tekrarlandı (21,15 / 29,52 / 50,63 M üçgen):

```
main      sabit = 33,68 ms   egim = 0,879 ms/Mtri   (orta nokta -2,4%)
prepass   sabit =  4,84 ms   egim = 0,478 ms/Mtri   (orta nokta +0,6%)
geometri 40,1 ms (%48) · sabit gölgelendirme 39,8 ms (%48) · RT gölge 1,6 ms (%2)
```

İki turun sabitleri %2 içinde (33,0 / 33,7 ve 5,0 / 4,8). **Kararı veren sayı
tekrarlanabilir.** Eğimlerdeki %10'luk fark kadrajdandır; oran (yarı yarıya
bölünme) her iki turda da aynı çıkıyor.

---

## 5. Otomatik cutout: bu sahnede ÖLÇÜSÜ SIFIR — ve sebebi öğretici

`viewport.set_automatic_cutout` üç kollu A/B:

| | kare GPU | prepass | main_pass |
|---|---|---|---|
| AÇIK | 83,19 | 23,29 | 55,74 |
| KAPALI | 79,56 | 19,55 | 55,90 |
| AÇIK (geri) | 81,25 | 19,94 | 57,06 |

Yayılım tekrar gürültüsünün (±%0,8) içinde. **Sinyal yok.**

Sebep, [AutomaticCutout.h](../../RayTrophiStudio/source/include/Viewport/AutomaticCutout.h)
kendi yorumunda yazıyor: *"Applied to GPU copies only. Authored material flags
and offline cutout stay intact."* Otomatik anahtar **bit 26**'yı yalnızca GPU
materyal kopyasına basar; `MATERIAL_FLAGS_PREVIEW_CUTOUT` ise **bit25 | bit26**
olduğu için, yazarlanmış bit25 zaten açıkken bit26'nın eklediği bir şey yoktur.

★★★★ Ve 2026-09-10'da ölçülen **324 → 82 ms**'i yapan şey bit26 değil,
**bit25**'ti: dört iğne materyalinin `alpha_cutout` alanı canlı olarak açıldı ve
`transmission` 121,8 → 1,5 ms'e düştü. Yani kazancın büyük kısmı gölgelendirmeden
değil, **transmission replay'in o materyalleri şeffaf sanmayı bırakmasından**
geldi. Bu, otomatik cutout'un neden ölçülmediğini de açıklıyor: transmission
replay uygunluğu **CPU tarafında yazarlanmış materyalden** karar veriliyor,
GPU kopyasındaki bit26'yı hiç görmüyor.

> **Kural olarak:** yalnızca GPU kopyasına basılan bir bayrak, CPU'da karar veren
> hiçbir tüketiciye ulaşmaz. İki tarafın da görmesi gereken bir sınıflandırma,
> yazarlanmış materyalde yaşamalıdır.

⚠ **CANLI DURUM UYARISI:** o dört materyalin `alpha_cutout=1` değeri
**kaydedilmedi**. `.rtp.shared` içinde alan yok, eski varsayılan `false` yükleniyor.
**Proje yeniden açılırsa 240 ms geri gelir** ve bu belgedeki bütün sayılar
geçersizleşir. Ölçüm alan bir sonraki ajan önce
`material.get` / `Probe-FoliageCutout.ps1` ile bu dördünü doğrulasın.

---

## 6. Gölgeler — depth shadow neden yavaş, ve yumuşak gölge için bütçe

A/B (tek değişen `viewport.set_rt_shadow`):

| | shadow_atlas | rt_shadow | kare GPU |
|---|---|---|---|
| RT gölge **KAPALI** | **74,3** | — | 153,4 |
| RT gölge **AÇIK** | **SKIPPED** | **2,3** | 87,4 |

**33x.** Cascade atlasının yavaşlığında gizem yok: 3 cascade x 30 M üçgen =
sahneyi üç kez daha çizmek, 0,86 ms/Mtri — §4'teki aynı geometri tavanı. RT
maskesi 6 cascade devralıyor (3 cascade x 2 senkron ışık: sahne directional +
dünya güneşi), **1 587 600 ışın = piksel başına tam 1**.

### 6a. Bugünkü maskenin sınırları (ölçülen, tahmin değil)

[rayfusion_rt_shadow.comp](../../RayTrophiStudio/source/shaders/rayfusion_rt_shadow.comp):

- piksel başına **1 ışın**, tek `pc.sunDirection`, sonuç **ikili 0/1**;
- **filtre yok** — keskin gürültü/aliasing doğrudan tüketiciye gidiyor;
- ışığın `radius` / `width` / `height` alanları **zaten var** (bu sahnenin
  güneşinde `radius: 0.1`) ama dispatch onları **okumuyor**;
- maske **tek yönlü bir ışık kümesi** için geçerli; point/spot hâlâ atlastan
  geçer (bu sahnede tek ışık olduğu için görünmüyor).

### 6b. Önerilen sıra — hepsi 2,3 ms'lik bütçenin içinde

1. **Koni örneklemesi.** Yönü ışığın açısal yarıçapı içinde çevir (directional
   için güneş 0,53°; point/spot için yarı açı `atan(radius / mesafe)`). Işın
   sayısı değişmez → **maliyet değişmez**. ⚠ Tek başına bugünkünden **kötü**
   görünür: aliasing yerine gürültü. Filtre aynı partide gitmeli.
2. **Derinliğe duyarlı (bilateral) ayrılabilir filtre.** Trilinear/mip **yanlış
   araç**: maske bir doku değil, ekran uzayında derinlik süreksizlikleri olan bir
   alan; trilinear ağaç sınırından zemine sızar. Buffer zaten piksel başına
   `vec2(görünürlük, derinlik)` tutuyor, yani bilateral ağırlık **bedava**.
   5x5 iki geçiş ≈ 0,3-0,5 ms. Örnek yönü piksel başına döndürülürse
   (mavi gürültü / Halton) 1 ışın + 5x5 ≈ efektif 25 örnek.
3. **Gerçek penumbra (PCSS).** Sabit yarıçaplı bulanıklık alan ışığı değil,
   yumuşatılmış sert gölge verir; penumbra **engelleyici mesafesiyle** büyür.
   `rayQueryGetIntersectionTEXT` zaten elde — engelleyici mesafesi de yazılırsa
   (`vec2` → `vec4`) filtre yarıçapı piksel başına ondan türer.
4. **Işık başına maske.** Point/spot'u da atlastan almak için maske çoğullanmalı
   (ışık başına dizi veya kanal paketleme). Atlas tamamen düşebilir.

### 6c. ⚠⚠ TÜKETİCİ TUZAĞI — filtreyi yazmadan önce oku

[material_preview_rt_shadow.glsl:30](../../RayTrophiStudio/source/shaders/material_preview_rt_shadow.glsl#L30):

```glsl
if (sampleValue.y >= 1.0 || abs(sampleValue.y - gl_FragCoord.z) > 2e-7) return false;
```

Tüketici, okuduğu pikselin **kendi derinliğini** doğruluyor. Filtre:

- `.y`'ye **dokunmamalı**;
- **yerinde (in-place) yazmamalı** — komşudan okurken kendi çıktısını okumaya
  başlarsa kapı her fragman için sessizce `false` döner ve **cascade'ini
  devretmiş ışık tamamen aydınlık kalır**. Tek belirti "gölgeler kayboldu"
  olur, filtre kodunda hiçbir hata görünmez. **Ping-pong buffer şart.**

---

## 7. RayFusion 2b — tüketici BAĞLI, iki RUNTIME ayarı yüzünden sessizdi

Kullanıcı gözlemi: "2b'nin getirdiği trace probe rays ve single diffuse bounce
hiçbir şeyi değiştirmiyor ve maliyet üzerinde de bir etkisi yok." Gözlem doğru,
**ama sebebi yapısal değil; iki ayar.** İkisi de canlı düzeltildi ve görüntü
değişti.

### 7a. ★★★★★ `core_status.gi_active` YALAN SÖYLÜYOR

```
gi_active         : false
inactive_reason   : "GPU tracing and raster GI composition are not connected"
renderer_available: false
```

**Bu satır yanlış.** Tüketici shader'da bağlı ve çalışıyor:
[material_preview_frag.frag:1183](../../RayTrophiStudio/source/shaders/material_preview_frag.frag#L1183)
`rfSampleProbeField(vWorldPos, N, probeIrradiance)` ambient okumasını probe
alanından alıyor. Aşağıdaki A/B bu yoldan geçen **369 kat gürültü üstü** bir
görüntü değişimi ölçtü. Bu alana bakıp "GI bağlı değil" sonucuna varmak
(ilk okumamda ben vardım) **yanlış teşhis üretir**; `gi_active` ya gerçek
tüketiciye bağlanmalı ya da kaldırılmalıdır.

### 7b. Gerçek iki sebep

**(1) Probe penceresi görünen hiçbir yüzeyi kapsamıyordu.** Varsayılan pencere
build sabiti: 4x2x4 hücre, spacing 3 → **12x6x12 m**. Sahne yüzlerce metre.
[probe_field.glsl:92](../../RayTrophiStudio/source/shaders/probe_field.glsl#L92)
pencere dışındaki her piksel için `false` döner ve global gökyüzü okumasına
düşer — **sessizce, hatasız**. `hit_fraction: 0.0` bunun göstergesiydi.

**(2) İzlenen üretici hiç İSTENMEMİŞTİ.** `producer: sky_bake`,
`producer_traced_requested: false`. Gökyüzü bake'i tasarım gereği fallback ile
**aynı değeri** üretir ("AYNI değer, FARKLI boru"), yani kapsama düzelse bile
tek başına hiçbir şeyi değiştiremezdi.

### 7c. Canlı A/B — kamera sabit, tek değişen üretici

Pencere 10x4x10 hücre / spacing 12 → **120x48x120 m**, 400 probe:

| kol | kare GPU | producer | hit_fraction | trace_ms |
|---|---|---|---|---|
| traced | 84,2 | traced | **0,741** | 0,19 |
| sky_bake | 85,4 | sky_bake | 0,000 | 0,19 |
| traced (tekrar) | 86,0 | traced | 0,741 | 0,28 |

```
taban gurultusu |traced - traced2| = 0,00012      <-- tekrar kolu
sinyal          |traced - skybake| = 0,04474
sinyal / gurultu = 369x
```

Fark **gökyüzünde tam sıfır**, sahnede yoğunlaşıyor (karenin %60-70 şeridinde
tepe 0,107) — yani doğru imza: kapanmaya bağlı bir terim. Sahne ortalaması
0,479 → 0,414 (%14 koyulma), kontrast 0,219 → 0,264 (%21 artış).
`bounce_shaded_hits` 1 → **76**, `hit_fraction` 0 → **0,741**.

**Maliyet: yok.** 84,2 / 85,4 / 86,0 ms — üç kol da gürültü içinde, `trace_ms`
0,19. GI'yı bağlamak bu karede bedava.

### 7d. Yön DOĞRU, ama en koyu çekirdekler AŞIYOR

Aynı A/B'nin görüntü ölçütleri (`compare_images.py`, kamera sabit):

| ölçüt | sky_bake | traced | RT referansı |
|---|---|---|---|
| zemin ara ton payı | 0,165 | **0,336** | 0,335 |
| sahne kontrastı (std) | 0,219 | **0,264** | — |
| sahne ort. luma | 0,479 | 0,414 | — |

★★★ **Ara ton payı RT'nin değerine oturuyor (0,336 / 0,335).** Yani izlenen
probe alanı zemine gerçek bir gradyan getiriyor — düz aydınlatmanın en somut
açığını kapatan şey bu, ve bedava.

⚠ **Ama gözle bakınca gölge çekirdekleri neredeyse saf siyah**, sky_bake'in orta
tonlu yeşiline karşı; RT'de aynı çekirdekler orta gri-yeşil. Yani dağılımın
şekli doğru, **en koyu ucu aşıyor**.

Sebep sayaçlarda: `bounce_active: true` ama **160 materyalin 70'i reddedilmiş**
(layered 69, transparent 69, flagged 59) ve kapının kendi gerekçesi
*"Opaque untextured diffuse/emission ... only"*. Bir ormanda **reddedilen
materyaller sahnenin kendisidir**: kanopi altındaki bir probe gökyüzünü
göremiyor, ve göreceği yüzeylerin hiçbiri ışık geri katkısı yapmıyor →
irradiance ≈ 0 → siyah.

> **Kural:** kapanma (occlusion) ile geri konan ışık (indirect) **aynı partide**
> açılmalıdır. Yalnızca kapanma bağlanırsa gradyan doğru gelir ama en koyu uç
> tabansız kalır; sevk edilebilir hâle gelmesi için ikisi birlikte gerekir.

Bu yüzden 2b'nin sıradaki işi alan borusunu bağlamak DEĞİL, **bounce kapısının
foliage materyallerini kabul etmesi** (69 "layered"/"transparent") ya da
ambient'e bir taban konması, ki kapanma hiçbir zaman sıfıra sürmesin.

### 7e. ★ Okuma tuzağı: `probe_field.minimum` METRE DEĞİL

`rayfusion.probe_field`'ın `minimum` alanı **hücre indeksi** taşıyor.
`[-66, 15, 172]`, `spacing 3.0` ile çarpılınca `(-198, 45, 516)` eder ve kamera
`(-187, 49, 524)` konumundayken pencere tam üstündedir — takip **çalışıyor**.
Metre sanılırsa "pencere 350 m geride" gibi bir kök neden uydurulur; kamera
48 m taşındığında alanın 16 birim kayması da tam olarak bunun imzasıdır
(48/spacing). Alan `minimum_cell` diye adlandırılmalı ya da yanına
`minimum_world` eklenmeli.

### 7f. ★★★★ ÖLÇÜM DERSİ: bu bölümde iki kez YANLIŞ teşhis kurdum

Kayda geçiyor, çünkü ikisi de bu depoda tekrar eden sınıflar:

1. **`gi_active: false`'a inandım** ve "tüketici bağlı değil, yapısal olarak
   değişemez" yazdım. Alan yalan söylüyordu (§7a). **Bir durum alanı, ölçümle
   çakışıyorsa ölçüm kazanır.**
2. **`set_probe_grid` yutuluyor sandım**: istek `applied: true` dönüyor ama 200
   kare boyunca uygulanmıyordu. Sebep istekte değildi — **RT render viewport'u
   `rendered` moduna almıştı ve raster SIFIR kare üretiyordu.** Aleti kontrol
   edince (`frame_timings.frames == 0`) aynı istek anında uygulandı.
   ★ `render.start` shading modunu DEĞİŞTİRİR; ondan sonra raster ölçen her şey
   önce `viewport.shading`'e bakmalı.

★ **Okuma tuzağı (bir kez yanlış okudum, kaydediyorum):** `rayfusion.probe_field`
içindeki `minimum` alanı **metre değil, HÜCRE İNDEKSİ**. `[-66, 15, 172]`,
`spacing 3.0` ile çarpılınca `(-198, 45, 516)` eder ve kamera `(-187, 49, 524)`
konumundayken pencere tam üstündedir — takip **çalışıyor**. Metre sanılırsa
"pencere 350 m geride" gibi bir kök neden uydurulur (kamera 48 m taşındığında
alanın 16 birim kayması da tam olarak bunun imzası: 48/spacing). Alanın adı
`minimum_cell` olmalı, ya da yanına `minimum_world` eklenmeli.

**Karar önerisi:** 2b'ye görüntü kalitesi tartışması açmadan önce tek soru şu —
GI kompozisyonu bağlanacak mı? Bağlanmayacaksa üretici tarafına daha fazla iş
yapmak **ölçülemez**; bağlanacaksa ilk iş `bounce_shaded_hits`'i bu sahnede
anlamlı bir sayıya çıkarmak (70 reddedilen materyalin neredeyse tamamı
"layered"/"transparent").

---

## 8. Raster ile RT'nin ÖLÇÜLEN görüntü farkı — ve RT'nin gerçek fiyatı

Aynı kameradan iki görüntü: raster viewport (`viewport.capture` +
`get_screenshot`) ve yol izleyici (`render.start`, 64 spp). Ölçütler
`scripts/ipc/compare_images.py` ile:

| ölçüt | raster | RT | yorum |
|---|---|---|---|
| sahne ort. luma | 0,518 | 0,395 | raster **%31 parlak** |
| sahne std (kontrast) | 0,219 | 0,187 | — |
| **uzak şerit** | **0,476** | **0,242** | raster uzağı **2 kat** parlak veriyor |
| **YAKIN/UZAK oranı** | **1,18** | **2,02** | ★ derinlik hissi burada ölçülüyor |
| **zeminde ara ton payı** | **0,018** | **0,335** | ★ penumbra: **19 kat** fark |

★★★★★ **"Derinlik hissi az" bir izlenim değil, 1,18'e karşı 2,02'dir.** RT'de
uzak orman yakın zeminin yarısı kadar parlak; raster'da neredeyse aynı. Sebep
tek: raster ambient'i **sabit bir terim**, yani sık kanopinin altındaki bir
piksel ile açık zemindeki bir piksel **aynı gökyüzü ışığını** alıyor. Kapanmayı
görmeyen bir ambient, mesafeyle artan kapanmayı da göremez.

★★★★ **"Gölge sert" de bir izlenim değil: zemin piksellerinin yalnızca %1,8'i
ara tonda.** Bu, "1 ışın / piksel, ikili sonuç, filtre yok"un doğrudan imzası
(§6a) ve §6b'nin sayısal gerekçesi.

⚠ Ölçüt notu: kanopide yerel std raster'da 0,128, RT'de 0,064 — **raster daha
"detaylı" değil, daha GÜRÜLTÜLÜ** (alfa-test aliasing + sert gölge benekleri;
ayrıca raster JPEG, RT PNG). Bu ölçüt hacmi değil gürültüyü ölçüyor, öyle
okunmalı.

### 8b. RT'nin fiyatı — "ray yolunda çözülebilir mi" sorusunun cevabı

`render.start` üç spp ile ölçüldü (1680x945, aynı sahne):

| spp | süre |
|---|---|
| 4 | 1,69 s |
| 16 | 2,27 s |
| 64 | 5,34 s |

→ **örnek başına ~61 ms**, sabit yük ~1,45 s.

★★★★★ **Tam bir yol-izlenmiş örnek (birincil + gölge + sıçrama) 61 ms; raster
karesi 81 ms.** Yani bu sahnede pahalı olan yol RT değil, RASTER. RT'yi
interaktif yapmayan şey örnek maliyeti değil, temiz görüntü için gereken
**örnek sayısı** (16-64 spp → 2-5 s).

Bunun pratik sonucu: **ışın bütçesi bol.** RT gölge maskesi 1 587 600 ışını
2,3 ms'te atıyor → **~690 Mışın/s**. Bu bütçeyle:

| iş | ışın/piksel | tahmini maliyet |
|---|---|---|
| bugünkü sert gölge | 1 | 2,3 ms |
| koni örneklemeli yumuşak gölge | 1 | 2,3 ms (değişmez) |
| 4 örnekli gölge | 4 | ~9 ms |
| **gökyüzü görünürlüğü (AO) terimi** | 1 | **~2-3 ms** |

★★★★ **AO terimi, probe alanının çözmeye çalıştığı işin iyi huylu yarısıdır**
ve §7d'deki siyah çukur arızasına **düşmez**: AO, ambient'i sıfırlayan bir
*yerine koyma* değil, sıfırdan farklı kalan bir ambient'in üzerine binen bir
*çarpandır*. Yukarıdaki 1,18 → 2,02 açığının büyük kısmı buradan kapanır ve
karenin ~%3'üne mal olur.

## 9. Sıradaki hedefler

★ Sıralama artık **iki ölçüte** göre: hız (§4) ve **görüntü açığı** (§8). İkisi
aynı listede, çünkü bütçeyi ikisi paylaşıyor.

| # | hedef | maliyet | kazanç | dayanak |
|---|---|---|---|---|
| 1 | **Gökyüzü görünürlüğü (AO) terimi ışın yolunda** | +2-3 ms | 1,18 → 2,02 açığının büyük kısmı | §8, §8b; siyah çukura düşmez |
| 2 | **Bounce kapısının foliage'ı kabul etmesi** (69 layered/transparent) | 0 ms | probe GI'yı KULLANILABİLİR yapar | §7d; bu olmadan GI = siyah |
| 3 | **Yumuşak gölge** (koni + bilateral + PCSS) | +0,3-0,5 ms | ara ton payı %1,8 → RT'de %33,5 | §6b |
| 4 | **Üçgen debisi** — çok kademeli LOD / impostor | **−38,9 ms'e kadar** | tek en büyük hız kalemi | §9a |
| 5 | **main_pass'ın 33 ms sabiti** — önce ÖLÇ | −? | %40'lık kör nokta | §9c |
| 6 | Opak/cutout ön geçiş ayrımı | −3-5 ms | | §9b |
| 7 | **Ölçü aleti borçları**: `raster_depth_prepass` setter'ı (§3b), `core_status.gi_active` (§7a), `probe_field.minimum` birimi (§7e) | 0 ms | üçü de YANLIŞ TEŞHİS ürettirdi | — |

★★★ 1 ve 2 **aynı partide** gitmeli (§7d kuralı): kapanmayı ışığı geri koymadan
bağlamak görüntüyü bozar.

### 9a. Neden LOD hâlâ listede

Piksel-altı üçgen tam fiyatına rasterize edilir. Bugün iki kademe var (tam mesh /
scatter proxy) ve [GENERAL_MESH_LOD_DESIGN.md](GENERAL_MESH_LOD_DESIGN.md)
uçurumu zaten ölçmüş: **98 500 üçgen vs 96**, arada hiçbir şey yok. §4'teki
0,468 ms/Mtri donanım tavanı, bu kalemin **yalnızca daha az üçgenle** ineceğini
söylüyor. GPU culling 0,08 ms'te instance eliyor; meshlet/cluster seviyesine
inilmedikçe piksel-altı üçgene dokunmaz.

### 9b. Ön geçişin opak kolu

Ön geçiş bugün **her şeyi** alfa-testli fragment shader ile çiziyor
([material_preview_shadow_frag.frag](../../RayTrophiStudio/source/shaders/material_preview_shadow_frag.frag)):
materyal fetch + doku örneği + `discard`. Opak arazi ve gövdeler için tamamı
boşa, ve `discard` erken-Z ile derinlik-yalnız çift hızlı rasterizasyonu kapatır.
Mesh başına "cutout materyali var mı" bayrağı build'de hesaplanıp opak kova
**fragment shader'sız** bir pipeline'a verilirse 5 ms'lik sabit terim ve opak
payın bir kısmı gider.

⚠ Materyal ID **köşe başına** (`matIdBuffer`), yani bir mesh karışık materyalli
olabilir — ayrım mesh başına **konservatif** yapılmalı: tek bir cutout materyali
varsa mesh alfa-testli kovaya gider.

### 9c. main_pass sabiti için önce ölçü aleti

33 ms'lik sabit terim `main_pass` içinde tek bir sayı olarak duruyor. Bu, bu
depoda daha önce **%60'ı tek bir terimde bulunan** (Physical Sky ambient konisi)
duruma birebir benziyor. Materyal shader'ına özellik bazlı anahtar/sayaç
konmadan yazılacak her optimizasyon **tahmindir**.

---

## 10. Araçlar

```powershell
Import-Module .\scripts\ipc\RtIpc.psm1 -Force
.\scripts\ipc\Probe-FrameStages.ps1 -Frames 60            # aşama tablosu, A/B kolları
.\scripts\ipc\Probe-GeometrySlope.ps1                     # §4'teki doğrusal uydurma
Invoke-RtIpc viewport.frame_timings                       # applied + stages
Invoke-RtIpc rayfusion.probe_field                        # §7 sayaçları

.\scripts\ipc\Probe-ProbeGi.ps1                           # §7c traced/sky_bake A/B
python .\scripts\ipc\compare_images.py A.jpg B.jpg [tekrar.jpg]   # §8 görüntü ölçütleri
```

⚠ `rayfusion.core_status`'un `gi_active` alanını ÖLÇÜ OLARAK KULLANMAYIN (§7a):
`false` diyor ve yanlış. Alanın canlı olup olmadığını `probe_field.hit_fraction`
ve `producer` söyler.
