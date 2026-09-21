# RayFusion — speküler yansıma ve emissive yakın alan

> **Durum:** AKTİF — Dilim A+B+C **YAZILDI, DERLENMEDİ** (2026-09-13).
> Sıralı kontrol listesi: `docs/dev/NEXT_BUILD_CHECKS.md`.

Amaç: realtime viewport'ta **düşük roughness yüzeylerin yansıtıcılığını** ve
**emissive yüzeylerin yakın alanı aydınlatmasını** Vulkan RT'ye yaklaştırmak.
İki ayrı eksik, iki ayrı dilim; ortak altyapıyı paylaşırlar ama birbirinin
önkoşulu değildirler.

---

## 0. Bugünkü durum (2026-09-13 kaynak okuması)

**Opak bir metal hiç yerel yansıma almıyor.** SSR çağrısı
`material_preview_frag.frag:1449`'da `drawPhase == PREVIEW_PHASE_TRANSMISSION`
kapısının içinde. Opak faz yalnızca env/sky lookup'ı görüyor (`:1195`,
prefiltered + BRDF LUT) ve üstüne probe momentlerinden bir görünürlük maskesi
çarpılıyor (`:1495`). Odanın içindeki hiçbir şey o metalde yok. RT ile aradaki
görsel farkın büyük kısmının bu olduğu **tahmin ediliyor — piksel ölçümü yok.**

O kapıyı "sadece açmak" bir çözüm değil: SSR `previewOpaqueColor` /
`previewOpaqueDepth`'i okuyor, o snapshot ise opak geçişten **sonra**
kopyalanıyor (`MaterialPreviewTransmission.cpp:263`, `hdrColorImage` →
`sceneColor`). Opak faz sırasında o görüntü henüz yoktur. Ucuz SSR kazancı diye
bir şey yok; RT rotası bu yüzden gerçekten rota.

★ Kontrol edilecek bayat yorum: `:1485` "the opaque snapshot is already display
referred" diyor, ama snapshot artık `hdrColorImage`'dan kopyalanıyor — yani
scene-linear. Doğruysa mevcut SSR kompozisyonu yanlış uzayda yapılıyor.

---

## 1. Probe alanına (1b-β) NEDEN eklenmez

Probe texel'i **grid hücresi merkezinde bir yarım küre ortalamasıdır.** Ayna
yansıması yüzey noktasından parallax ister. `RAYFUSION_PROBE_BOUNCE.md` bunu
zaten yazmış: roughness kasıtlı okunmuyor, çünkü o dilimde yalnız diffuse lob
var.

Oraya bir specular lobu takmak makul görünen ama her aynada yanlış olan bir
leke üretir. Bu deponun en pahalı hata sınıfı (*ölçü/panel yalan söylüyor*) tam
bu şekle benziyor: kimse "yansıma yanlış" demez, "yansıma biraz tuhaf" der ve
kalibrasyon turuna gömülür.

---

## 2. Sert engel: RT geçişlerinin elinde yalnızca DEPTH var

`screen_gi_trace.comp` şekil olarak doğru yapı: piksel başına, **yüzey
orijinli**, hardware ray query, gerçek malzeme çözümü (`rfResolveHit`), gerçek
gölge ışını + env fallback, filtre geçişi, ve tüketicide confidence sözleşmesi.
`rfBounceRadiance` yazılmış ve canlı doğrulanmış — isabet gölgelendirmesi
yeniden yazılmayacak.

Ama specular lobu doğrudan eklenemez:

`giSurface()` normali depth türevinden geri kuruyor (`screen_gi_surface.glsl`),
yani **geometrik** normal; normal map yok. Tüketici bunu zaten biliyor ve
`dot(normal, s.surface.xyz) < 0.5` ile kaçıyor
(`material_preview_screen_gi.glsl:15`). Cosine yarım küre için birkaç derece
hata görünmez. Yansımada **yön hatası normal hatasının 2 katıdır**, ve
roughness 0,05'te lob ~3° geniş: depth türevi normali lobdan daha çok sapar.

Üstüne geçiş roughness/F0/metallic'i bilmiyor. Bunlar hem BRDF **hem de
kapıdır** — pikselin büyük kısmını atlamak istiyoruz.

Frame sırası engeli pekiştiriyor: depth prepass → render pass KAPANIR → RT
shadow + screen GI compute → LOAD ile açılır → ana gölgelendirme
(`MaterialPreviewRtShadow.cpp:388`). Screen GI'ın koştuğu yerde normal yazan
hiçbir şey henüz koşmamıştır. Prepass da yazamaz: shadow vertex shader'ını
devralıyor ve **normal attribute'unu okumuyor**
(`VulkanViewportBackend.cpp:1894`).

---

## 3. Seçilen mimari: ince G-buffer + ayrı reflection geçişi

Ana gölgelendirme geçişine **ikinci bir renk eklentisi** (RGBA16F):

| Kanal | İçerik |
|---|---|
| .xy | oktahedral kodlanmış **shading** normal (normal map dahil) |
| .z | `reflectionRoughness` (specular AA uygulanmış hali, `:924`) |
| .w | split-sum ağırlığı `F0·brdf.x + brdf.y`'nin skaler paketi |

Sonra: geçiş kapandıktan sonra `reflection_trace.comp` koşar, depth + bu
buffer'ı okur, R etrafında GGX-VNDF örnekler, isabeti mevcut
`rfBounceRadiance` makinesiyle gölgelendirir, ve **HDR hedefine DoF'tan önce**
kompozit eder.

Neden bu, fragment içi inline ray query yerine:

- **Üretici = tüketici.** Normal ve roughness'ı gölgelendirmeyi yapan shader
  yazar; normal map dahil bire bir aynı değer. Screen GI'ın `dot<0.5` kaçışı o
  ayrışmanın faturasıdır — aynı fatura ikinci kez ödenmesin.
- **Ölçülebilir ve kapılanabilir.** Ayrı `RasterStage`, ayrı timer, çözünürlük
  ölçeklenebilir, ve uygun pikseller compact edilip **indirect dispatch**
  yapılabilir: parlak yüzeyi olmayan sahne ~sıfıra yakın maliyet. Kapısız
  geçişin bedeli transmission replay'de bir kez ödendi.
- HDR içinde kompozit, bugünkü "post'tan sonra" yolundan doğru uzayda.

Reddedilen alternatif — fragment shader'ında inline ray query: N/roughness/F0
zaten register'da, G-buffer yok, attachment cerrahisi yok. Ama geçiş olarak
**ölçülemez**, maliyet overdraw ile büyür (foliage'da early-Z dersi tam buradan
geçiyor) ve TLAS + hit/material tablolarını raster descriptor set'ine sokmak
gerekir.

**Bedeli dürüstçe:** render pass + framebuffer + o geçişi kullanan **her**
pipeline'ın blend state'i attachment sayısıyla büyür. Prepass'in
`attachmentCount = 1 KALMALI` notu bu bağlantıyı zaten işaretliyor
(`VulkanViewportBackend.cpp:1936`).

---

## 4. Metalik ile sınırlı DEĞİL — çerçeveleme bunu bedava yapar

Bugün:

    envSpecular = prefiltered * (F0 * brdf.x + brdf.y)

`prefiltered`, R yönündeki **ortam radyansı lookup'ıdır.** Reflection geçişi
metalik bir terim *eklemez* — o lookup'ın **yerine geçer**, split-sum terimi
aynen kalır.

Sonuç: dielektrikler bugün kullandıkları Fresnel'in tam aynısıyla bunu
devralır — verniklenmiş ahşap, boyalı zemin, seramik, plastik, hiç ek kod
olmadan. Ve enerji, yerine geçtiği yolla tutarlı kalır.

**Kapı loba kurulur, malzeme sınıfına değil:**

    reflectionRoughness < t   VE   (F0·brdf.x + brdf.y) > w

O ağırlık grazing açıda Fresnel ile yükselir, yani kaba-ish bir dielektrik
zemin tam görünür olduğu yerde kapıdan geçer, tepeden bakışta atlanır.

★ `metallic > x` kapısı **yanlış ölçü aleti** olurdu: sorunun sorulduğu
yüzeyleri tam olarak o eler. Bu, kapının neden burada yazıldığının gerekçesi.

### Üç tuzak — hepsi "makul görünen yanlış sonuç" sınıfından

1. **Çift oklüzyon.** `:1495`'teki `rfSpecularSkyVisibility`, ışının artık *tam
   olarak ölçtüğü* oklüzyonun proxy'sidir. Traced verinin olduğu yerde
   uygulanmamalı. İkisi birlikte kalırsa görüntü sadece kararır ve kimse buna
   bug demez.
2. **Dikiş.** Işın ıskaladığında **aynı** env lookup'ına **aynı** filtrelemeyle
   düşmeli; yoksa aynada traced/untraced bölge arasında görünür sınır çıkar.
3. **Clearcoat ikinci bir lob**, kendi roughness'ıyla (`:1214`). Tek ışın
   ikisine hizmet edemez. Karar: roughness'lar yakınsa taban ışını coat için de
   kullanılır, değilse coat env lookup'ında **kalır** — ve bu yazılır, sessizce
   yaklaştırılmaz.

---

## 5. Emissive yakın alan — AYRI ve PARALEL dilim

Bu yapısal olarak **zaten var:** `rfBounceRadiance` dönüşü
`emission + diffuse * incoming` (`rayfusion_probe_bounce.glsl:261`). Screen GI
ışını emissive bir üçgene çarptığında emission gelir.

Eksik olan bir yetenek değil, **kestirici**: emissive geometri örneklenebilir
bir ışık değil. Işık tablosu yalnızca point + directional, en fazla 64
(`RayFusionBounce.cpp:174`); `Spot` ve `Area` `unsupportedLights`'a yazılıyor.
Yani bir lamba ancak yarım küre ışınının **şansına** bulunuyor,
`samples ∈ {1,2,4}` ile.

★ **Beklenecek belirti şimdiden yazılsın, çünkü kimse buna bug demez:** seed
`rfHash(index+1u)` ve kasıtlı olarak kare numarasından bağımsız — yani titreme
**değil**, sabit bir **noktalı/benekli desen**. Üstüne 5×5 kenar duyarlı blur
(`screen_gi_filter.comp`) yayar ve o filtrede **hiç luminance clamp yok**.
Küçük parlak bir emissive = birkaç pikselde bulunmuş, blurla yayılmış leke.

**Yapılacak iş: emissive üçgen NEE.** Aynı hit tablolarından alan ağırlıklı bir
emissive üçgen listesi kurulur (emission × strength bir eşiğin üstünde) ve
mevcut `rfLights` döngüsünün yanında örneklenir; katkı seçim olasılığına
bölünür — zaten oradaki desen bu.

İki sınır baştan yazılıyor:

- **Transparan malzemeler bounce kümesinden eleniyor** (canlı ölçümde 40'ta 8).
  Pratikte lamba abajurları transparandır, yani hiç katkı vermezler.
- **Spot/Area ışıklar bounce tablosunda hiç yok**, dolayısıyla ne sıçramada ne
  de yansıma gölgelendirmesinde görünürler.

---

## 6. Parity OLMAYACAK kısım — baştan yazılıyor

- **Tek sıçrama.** Aynada ayna yok. İsabet ettiği yüzeyin kendi speküleri yok
  (diffuse + emission + 1 doğrudan ışık). Metal odayı doğru yansıtır, iki krom
  küre birbirini yansıtmaz.
- Refraktif transmission, hacimler, terrain katmanları bounce kümesinde yok.
- Reflection ışını ekran dışı geometriyi görür (SSR'ın yapamadığı şey), ama
  `maxDistance` ötesini env ile yaklaştırır.

---

## 7. Dilimler ve her birinin ÖLÇÜ ALETİ

`bounce_shaded_hits` presedansı: her dilim, *ne olabilir* değil *ne oldu*
sorusunu GPU'da ölçen bir sayaçla gelir.

**Dilim A — G-buffer.** Ana geçişe attachment, oct normal + roughness +
split-sum ağırlığı. Tüketicisi yok. Ölçü aleti: ayrı `RasterStage` zamanı ve
`gbuffer_written_pixels`. Kabul: görüntü **değişmemeli**; maliyet farkı
ölçülmeli.

**Dilim B — reflection trace + kompozit.** GGX-VNDF, 1 ışın/piksel, lob kapısı,
indirect dispatch. Ölçü aletleri:
`reflection_pixels_gated` / `reflection_rays` / `reflection_shaded_hits` /
`reflection_sky_misses`.
★ `rays > 0 && shaded_hits == 0` görüntünün **hiç değişmediğini** söyler — kapı
sayısı bunu asla söyleyemez.
Aynı dilimde `rfSpecularSkyVisibility` çift oklüzyonu kapatılır, yoksa ölçüm
kararmayla karışır.

**Dilim C — emissive üçgen NEE.** B'den bağımsız, önce de yapılabilir. Ölçü
aletleri: `emissive_triangles`, `emissive_samples`, `emissive_shadowed`,
`emissive_rejected_transparent`.

---

## 8. Kural 1 — dört dokunuş + tarif

Her dilim için:

| Katman | Dosya |
|---|---|
| Çekirdek API | `include/Api/RtApi.h` + `src/Api/RtApiRayFusion.cpp` |
| IPC dispatch | `src/Api/RtIpcRayFusion.cpp` |
| Python binding | `src/Api/RtPythonRayFusion.cpp` |
| Yetki | `src/Api/RtIpcSecurity.cpp` (Render capability) |
| Ajan tarifi | `python scripts/gen_ipc_descriptors.py` + overlay JSON |

Kontrol yüzeyi: `rayfusion.set_reflections {"enabled":bool}`,
`rayfusion.set_emissive_bounce {"enabled":bool}`, ve sayaçlar mevcut
`rayfusion.probe_field` yanında `rayfusion.reflection_status` altında.

★ IPC yazısı ile ölçüsü arasına bir kare koyulmalı; toggle uygulanır ama sayaç
ÖNCEKİ partiyi ölçer.

Yeni `.cpp` dosyaları `.vcxproj`'a, yeni shader'lar `compile_shaders.bat`'a
eklenir. Script'ler `scripts/` **ve** `x64/Release/scripts/` altına kopyalanır.
