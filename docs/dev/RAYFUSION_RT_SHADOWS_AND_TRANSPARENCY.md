# RayFusion adım 2 — RT ile gölge ve şeffaflık

> **Durum:** AKTİF — yön kararı ve ölçümler (2026-09-08).
> ★★★★★ **§6c'nin "ön geçiş net negatif" hükmü 2026-09-10'da ÇÜRÜTÜLDÜ**: aynı
> notun önerdiği ayırıcı deney koşuldu, hipotez 2 (`early_fragment_tests`
> eksikliği) doğru çıktı ve ön geçiş **19,4 ms'e mal olup 162 ms kazandırıyor**.
> Ön geçiş artık bu karedeki en büyük tek optimizasyondur; sökmeyin. Yeni ölçüm:
> [RASTER_FRAME_COST_2026_09_10.md](RASTER_FRAME_COST_2026_09_10.md) §3.
> Ölçüm tabanı: [RASTER_MICROTRIANGLE_WALL.md](RASTER_MICROTRIANGLE_WALL.md) §7.

## 1. Hedef — ve iki kez kaçırdığım nokta

RayFusion **bir render yolu**. Amacı raster'ı RT ile *değiştirmek* değil,
raster'ın **en pahalı terimlerini** RT ile ucuzlatmak. Raster'ın koruduğu şey
pazarlık konusu değil:

- **Her karede aynı kalite.** Akümülasyon beklenmez.
- **Hacimler fusion'da zaten hızlı.**
- Düzenleme sırasında kararlı, gürültüsüz görüntü.

★★★★ Bu notun iki önceki turunda ben iki kez yanlış sonuca vardım ve ikisi de
aynı hatanın türeviydi: **terimleri ölçtüm, mimariyi unuttum.** Önce
"cluster LOD yazalım" (üçgeni hedefliyordu — oysa üçgen 0,26 ms/Müçgen),
sonra "RT'ye geçelim" (fusion'ın var oluş sebebini siliyordu). Doğru okuma,
ölçülen terimlerin **hangisinin RT'ye devredilebilir** olduğuna bakmaktı.

## 2. Ölçülen terimler ve devredilebilirlikleri

~29M üçgen, 1680×945, orman sahnesi. `display_loop_period_ms`:

| | ms/kare | üçgen |
|---|---|---|
| raster solid (alfa testi YOK, ayrı pipeline) | **17,5** | 66,3M |
| RT (rendered) | 67,1 | 44,1M |
| **raster scene/fusion** | **427,6** | 29,1M |

★★★★★ **10 MİLYARLIK doğrulama.** Kullanıcı sahneyi 100 000 instance /
**9 905 631 295 üçgene** çıkardı ve solid **hâlâ akıcı**:

```
visible_triangles : 104.581.218      (scatter_triangle_target 96.000.000)
full 1.018 inst   /  proxy 54.075 inst
solid             : 56,8 ms  ->  0,54 ms/Mucgen
```

Bu iki şeyi birden kanıtlıyor: (a) raster geometri yolu 10 milyarlık sahnede de
bağlayıcı kısıt değil, (b) bu partinin mesh başına bütçe payı düzeltmesi
100 000 instance'ta da tutuyor — 54 bin instance proxy'ye düşüyor. Düzeltme
olmadan geri besleme oranı yine 1,5'e yapışır, hiçbir şey proxy'ye düşmez ve
raster 9,9 milyar üçgen göndermeye kalkardı.

Terim ayrışması (ışık A/B'si ve preset süpürmesi; ikisi de yavaş kare, yöntem
orada geçerli):

| Terim | ms | RT'ye devir? |
|---|---|---|
| geometri | ~8 | **hayır** — raster ~3,8 milyar üçgen/sn, zaten en iyi yol |
| **gölge (1 yönlü ışık, 3 kaskad)** | **113** | ★★★★ **evet, en net aday** |
| scene ambient (sky + probe) | ~160 | kısmen — RayFusion probe'u zaten bu |
| **materyal / opacity / şeffaf geçişler** | ~139 | ★★★ **evet, kalite kazancıyla** |

## 3. ★★★★ Gölge: neden en net aday

`material_preview_shadow_data.glsl` `rtPreviewShadow()`, **fragment başına**:

- kaskadı bulmak için **6'ya kadar matris çarpımı** (döngü, ilk kapsayanda durur)
- **9 PCF tap** (`radius=1`), quality preset'inde **25** (`radius=2`)
- üstüne `rtDeepShadow()`
- ayrıca atlasın kendi geometri geçişi (kaskad/yüz başına)

~116M fragment çağrısıyla çarpınca **~1 milyar texture tap**. Ölçülen: **113 ms.**

RT karşılığı **piksel başına bir ışın**: 1,6M ışın, ve zaten kurulu bir AS
üzerinde (`as_bytes 39,3 MB`, `blas_count 32`, `instance_count 10 030`,
`hardware_rt true`, `builds 1`, `tlas_only_refreshes 0`).

★ Bedavaya gelenler — hiçbiri kaskadla iyi yapılamaz:

| Kaskad bugün | RT ışını |
|---|---|
| `params.x` derinlik bias + `params.y` normal bias (ikisi de ayar hilesi) | bias yok, peter-panning yok |
| sert gölge; yumuşaklık PCF yarıçapı taklidi | `sun_size 0.545°` zaten dünya parametresinde → **fiziksel yumuşak gölge** |
| kaskad dikişi, atlas çözünürlük tavanı | ikisi de yok |
| alfa-test'li iğne gölgesi atlas çözünürlüğüyle sınırlı | any-hit ile **doğru** |

## 4. Nereye oturur: MEVCUT RayFusion desenine

RayFusion'ın kurduğu şekil zaten bu: **üretici ışın atar, tüketici doku örnekler.**
Probe alanı böyle çalışıyor (`trace_ms 0,2035` bir compute geçişi, fragment
shader sonucu örnekliyor).

Gölge maskesi **aynı şekil**:

```
[depth prepass] -> [RT gölge geçişi: piksel basina 1 isin] -> [golge maskesi]
                                                                    |
                                          material_preview_frag ----+
                                          rtPreviewShadow() IMZASI AYNI
```

★★★ `rtPreviewShadow()` **imzası değişmez**; içi atlas yerine maskeyi okur.
Bu, deponun kendi kararının aynısı: *tüketici sabit, üretici değişir*
(RayFusion dikişi = ambient okuması).

## 5. Depth prepass — önceki değerlendirmemin düzeltmesi

Daha önce "prepass'in kolu kısa" demiştim. **Yanlıştı**, ve sebebi bozuk bir
geometri sayısıydı (IPC ile şişmiş 97,9 ms; gerçeği 17,5).

Prepass burada **iki işi birden** yapıyor:

1. Overdraw gölgelendirmesini keser (fusion'ın 427,6 ms'inin %98'i fragment).
2. **Gölge ışınının çıkış noktasını üretir** — piksel başına dünya konumu.

⚠ Alternatifi olan "fragment içinde `rayQuery`" **daha kötüdür**: ışını piksel
başına değil **fragment çağrısı** başına atarsın — 1,6M yerine 116M ışın.
Yani prepass burada bir optimizasyon değil, **ön koşul**.

⚠ Prepass foliage için **alfa testi yapmak zorunda**. Solid modun ucuz 17,5 ms'i
alfa testini İÇERMEZ (ayrı pipeline, `SolidPushConstants`), yani prepass o
sayıdan pahalı olacak. Tahmin değil, ölçülecek.

## 6. Sıra

| # | İş | Neden bu sırada |
|---|---|---|
| 1 | ★★★★★ **Depth prepass** — 2026-09-10'da **+162 ms KAZANÇ** ölçüldü, RT gölge açıkken zaten ZORUNLU | §6c'nin "overdraw yokmuş" hükmü çürüdü: overdraw vardı, `early_fragment_tests` eksikti |
| 2a | ✅ **RT gölge maskesi ÜRETİCİSİ** — yazıldı, derlenmedi | Maliyeti, görüntüyü değiştirmeden ölçmek |
| 2b | **Tüketiciyi çevir** (kaskad → maske) | 113 ms'lik terim; önce §8'deki iki engel |
| 3 | **RT şeffaf geçişler** | Bugün `transparency: unsorted_alpha` (sıralama YANLIŞ) ve `transmission: screen_space_thickness` (yaklaşım). Işınla ikisi de doğru olur |
| 4 | Kalan ambient terimi | RayFusion probe'u zaten burada; 1-3 sonrası yeniden ölç |
| 5 | LOD kademeleri | Artık *düzenleme yolunu* ölçekte tutmak için, fusion'ı kurtarmak için değil |

★★ Her adım IPC'ye açılır (CLAUDE.md kural 1) ve **kapatılabilir** olur —
kapatılamayan bir düzeltme kendisini yargılayacak ölçümü de öldürür
(`viewport.set_raster_gpu_instancing` precedent'i).

## 6b. Adım 1 nasıl yazıldı

★★★ **Yeni shader yok.** Ön geçiş, gölge atlasının zaten derlenmiş ikilisini
devralıyor: `material_preview_shadow.vert` ana pipeline'la **birebir aynı**
vertex attribute konumlarını kullanıyor (0=pos, 2=matId, 3-6=model, 7=uv) ve
`pc.viewProj` okuyor — yani kamera geçişinde de doğru matrisi alıyor.
`material_preview_shadow_frag.frag` ise yalnızca alfa testi yapıp `discard`
ediyor. Descriptor düzeni de uyuyor (binding 0 = materyal SSBO, 1 = doku dizisi).

★★★★ **Döngü KOPYALANMADI.** Ön geçiş ile asıl geçiş aynı mesh'lerden aynı
kapılarla geçmek zorunda, o yüzden gövde paylaşılıyor: dış `rasterPass` döngüsü
+ `depthOnlyPass` bayrağı. Bu dosyada aynı sınıfın kuyruğu bir kez kopyalanmış
ve kopya bir çağrıyı düşürmüştü (GPU culling hiç açılmadı, görsel belirtisi
yoktu). `audit_raster_depth_prepass.py` ikinci bir materyal-önizleme döngüsünü
reddediyor.

Pipeline, ana pipeline'ın durum yapılarını **yeniden kullanıyor** (`mpVertInput`,
`mpIA`, `mpRast`, `mpMS`, `mpDyn`), yalnızca derinlik/renk durumunu değiştiriyor:
derinlik YAZAR (`LESS_OR_EQUAL`), `colorWriteMask = 0`, harmanlama kapalı.
Böylece vertex girdi düzeni ikisi arasında **yapısal olarak kayamaz**.

★★ Asıl geçiş **değiştirilmedi** — hâlâ `LESS_OR_EQUAL` + derinlik yazımı açık
+ harmanlama açık. `EQUAL`'e çevirmek şeffaf yüzeyleri kırardı (onlar ön geçişte
derinlik yazmıyor). Gizli fragment'ler `LESS_OR_EQUAL` ile zaten donanım erken-Z
reddine takılıyor; `discard` erken derinlik **yazımını** engeller, **testini**
engellemez.

⚠ Kalan sınırlama: ön geçiş impostor'ları atlıyor ama **kısmi şeffaf** yüzeyleri
atlamıyor. Cam/su ön geçişte derinlik yazarsa arkası elenir.

## 6c. ~~ÖLÇÜLDÜ: ön geçiş tek başına NET NEGATİF~~ — ÇÜRÜTÜLDÜ 2026-09-10

> ★★★★★ **BU BÖLÜMÜN HÜKMÜ GEÇERSİZ.** Aşağıdaki ölçüm doğruydu ama iki
> hipotezi ayıramıyordu ve notun kendisi bunu yazmıştı. Ayırıcı deney (§6c
> sonundaki "erken derinlik testini garantile") koşuldu:
> `material_preview_frag.frag` artık `layout(early_fragment_tests) in;` beyan
> ediyor. 2026-09-10 A/B'si, RT gölge kapalı, tek değişen ön geçiş:
> **main_pass 219,8 → 57,9 ms.** Yani **hipotez 2 doğruydu** — gizli fragman
> vardı, onu gizleyen şey sürücünün erken-Z **reddini** kapatmasıydı.
> Aşağıdaki "overdraw azaltma bir strateji olarak elenir" sonucu **yanlıştır**.
> Ölçüm: [RASTER_FRAME_COST_2026_09_10.md](RASTER_FRAME_COST_2026_09_10.md) §3.
> ★ Ders: bir A/B'nin **sonucu** doğru olabilirken **hükmü** yanlış olabilir;
> ayrılamayan iki hipotez varsa hüküm, ayırıcı deney koşana kadar askıdadır.

### Özgün 2026-09-08 ölçümü (kayıt için korunuyor)

Orman sahnesi, Material + scene, ~21,5M üçgen. `viewport.set_raster_depth_prepass`
ile dört tur A/B:

| | ms/kare |
|---|---|
| **KAPALI** | 323,5 · 323,7 · 324,4 · 324,1 → **ort. 324** |
| **AÇIK** | 339,0 · 366,8 · 337,4 · 340,9 → **ort. 346** |

KAPALI tarafın yayılımı **0,9 ms** — gürültü değil. Ön geçiş **~22 ms (%7)
ekliyor ve ölçülebilir hiçbir şey kazandırmıyor.**

★★★★ **Bu, §1'deki tablonun hangi satırının geçerli olduğunu ÖLÇÜMLE seçiyor.**
Üç mekanizma da üçgenle doğrusal maliyet üretir ama derinlik ön geçişi yalnızca
**klasik oklüzyon overdraw'ını** düzeltir. Ön geçiş hiçbir şey kazandırmadıysa,
elenecek gizli fragment **yok** demektir: foliage kartları önden arkaya
yığılmış değil, **uzamsal olarak yayılmış mikro-üçgenler**. Her biri piksel-altı
kalıp yine de tam bir 2×2 quad çalıştırıyor, ve o fragment'lerin hepsi
**görünür**. Derinlik testinin eleyeceği bir şey yok.

⚠ **Ayrılamayan ikinci ihtimal, dürüstlük gereği:** ana fragment shader'da
`discard` var ve `layout(early_fragment_tests) in;` beyanı **yok**. Sürücü
erken-Z **reddini** tamamen kapatmış olabilir (spec `discard` ile erken derinlik
*yazımını* yasaklar, *testini* yasaklamaz — ama sürücüler muhafazakâr
davranabilir). İki hipotez de aynı ölçümü üretir.

**Ayıracak deney (ucuz, sırada):** ön geçiş koştuğunda ana geçişi derinlik
yazımı KAPALI + `early_fragment_tests` AÇIK bir varyantla çiz. Ön geçiş
derinliği zaten doğru yazdığı için `discard`'ın derinliğe etkisi kalmaz ve
reddin garanti olması sağlanır. Hâlâ kazanç yoksa hipotez 1 doğrulanır ve
**overdraw azaltma bir strateji olarak tamamen elenir.**

★ Ne olursa olsun ön geçiş **boşa gitmedi**: RT gölge maskesi piksel başına
derinlik istiyor ve 113 ms'lik gölge terimine karşı 22 ms iyi bir takas.

## 7. Kabul ölçütü

- Gölge terimi 113 ms → hedef **< 15 ms**, ve görüntü **daha iyi** (yumuşak,
  bias'sız, dikişsiz).
- ★★★ **En sinsi başarısızlık:** maske doğru görünür ama bir kare gecikmelidir.
  Kamera hareket ederken gölge geometriye yapışmaz, kayar — ve durunca düzelir,
  yani ekran görüntüsünde fark edilmez. Hareket hâlinde bakılmalı.
- Her karede aynı kalite korunmalı: akümülasyon beklemesi **girmemeli**.
  Girerse fusion'ın var oluş sebebi gitmiştir.

## 8. Adım 2a nasıl yazıldı — ve 2b'den ÖNCE kapanması gereken iki şey

**Işın bütçesi ÖLÇÜLDÜ** (bu sahnenin kendi TLAS'ında, probe izleyicisiyle):

| probe | ışın | trace_ms | Mışın/sn |
|---|---|---|---|
| 32 | 2 048 | 0,271 | 7,5 |
| 1024 | 65 536 | 0,196 | **334** |

`trace_ms` 2 binden 65 bin ışına kadar **düz** — ölçüm hâlâ sabit dispatch
maliyetiyle sınırlı, yani ışın maliyeti bunun *altında*. **1,59M ışın ≈ 5 ms**,
karşısında ölçülmüş **113 ms** kaskad terimi.

Yazılanlar: `rayfusion_rt_shadow.comp` (ekran uzayı, piksel başına tek ışın),
`MaterialPreviewRtShadow.cpp` (maske görüntüsü + compute pipeline + bariyerler),
`viewport.set_rt_shadow` / `viewport.rt_shadow` kolları, `audit_rt_shadow_pass.py`.

★ `rasterImageBarrier` anonim namespace'ten `Viewport/RasterImageBarrier.h`'ye
taşındı — ikinci bir çeviri birimi ona ulaşamıyordu ve kopyalamak bu deponun
tekrar eden arıza sınıfıydı. Tek gövde, iki kullanıcı.

### ⚠⚠ Engel 1: maske BİR KARE GERİDE

Geçiş şu an **ana geçişten sonra** koşuyor. Sebep: derinliği compute'tan
örneklemek için ana geçişten önce bitirmek şart, o da render pass'i bölmeyi
gerektiriyor. Bu yerleşimde **ölçüm doğru** (dispatch, ışın sayısı ve TLAS
erişim örüntüsü nihai hâliyle aynı) ama **tüketilemez**: belirti "kamera
hareket ederken gölge kayıyor, durunca düzeliyor" olur — ekran görüntüsünde
fark edilmeyen arıza.

**2b'nin ilk işi:** `createViewportRenderPass` fabrikasıyla bir
`hdrRenderPassLoad` varyantı (renk LOAD/GENERAL, derinlik LOAD/ATTACHMENT) ve
kaydın bölünmesi. ★ Load/store op'ları pipeline uyumluluğunu **etkilemediği**
için mevcut pipeline'lar yeni geçişte aynen çalışır — yeni pipeline gerekmez.

### ⚠⚠ Engel 2: alfa testi YOK

Shader `gl_RayFlagsOpaqueEXT` kullanıyor, yani yaprak/iğne kartı ışığı
**kartın tamamıyla** kesiyor. Belirti "gölge yok" değil, **orman fazla
karanlık** — makul görünen ve kimsenin bug diye raporlamayacağı sonuç.

Kapatmak için adayı non-opaque yapıp ray query döngüsünde UV + materyal +
opacity dokusunu örnekleyerek `rayQueryConfirmIntersectionEXT` çağırmak gerek.
Bu, shader'a bindless doku dizisi + materyal buffer'ı bağlamak demek
(`closesthit.rchit`'in zaten yaptığı iş).

★★★ Bu iki engel **bilerek** açık bırakıldı: adım 2a'nın tanımı üretici-only
olmak. `audit_rt_shadow_pass.py` tüketicinin bu partide değişmediğini de
denetliyor — ikisi aynı anda değişirse "kazanç mı, kalite mi değişti" sorusu
ölçülemez olur.

## İlgili

- [RASTER_MICROTRIANGLE_WALL.md](RASTER_MICROTRIANGLE_WALL.md) — terim ölçümleri
- [GENERAL_MESH_LOD_DESIGN.md](GENERAL_MESH_LOD_DESIGN.md) — LOD, artık sıra 5
- [RASTER_SCENE_LIGHTING_FRAGMENT_COST.md](RASTER_SCENE_LIGHTING_FRAGMENT_COST.md)
