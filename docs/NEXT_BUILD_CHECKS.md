# Sıradaki build kontrolleri

Sıra bağlayıcı: üsttekiler bağımsız ve hızlı, alttakiler üsttekinin sonucunu
maskeleyebilir. Bir madde bozuksa altındakilerin "geçmesi" bir şey kanıtlamaz.

---

## PARTİ 6 — prosedürel gözeneklilik (mayalanmış hamur)

Shader + C++ birlikte: `volume_closesthit.rchit` derlenmeli **ve** build alınmalı.

### P6.1 — Kapalıyken hiçbir şey değişmemeli (REGRESYON, önce bu)

`Pore Amount` = 0 ile mevcut bir sıvı sahnesi eskisiyle **birebir** aynı olmalı.
`isoPoreOffset` 0'da erken dönüyor, yani ekstra maliyet de yok.

**Bozuksa ne demek:** ortak `sampleIsoField` sarmalayıcısı yanlış çağrı noktasına
uygulanmış. Gaz marşının 8 çağrısı **ham** `sampleDensityAcc` kalmalıydı; iso
yolunun 13'ü sarmalayıcıya geçti.

### P6.2 — Gözenekler GERÇEK GEOMETRİ mi (asıl kontrol)

`Pore Amount` ≈ 0.25, `Bubble Size` ≈ 0.03, `Size Variation` ≈ 0.6.

**Görmen gereken:** gözeneklerin **kenarları ışık alıyor** — kendi gölgesi,
kendi kırılması var, kesitte hamur gibi görünüyor.

**★ EN SİNSİ BAŞARISIZLIK: gözenekler var ama DÜZ görünüyor** — delik gibi değil,
boyanmış leke gibi. O zaman gradyan gözeneği görmüyordur, yani 6 merkezi-fark
örneğinden biri (veya hepsi) hâlâ ham yoğunluğu okuyordur. Bu "biraz sönük
duruyor" diye raporlanır, bug diye değil. Bakılacak yer: gradyanın altı örneği
de `sampleIsoField` olmalı.

### P6.3 — Gaz devri aynı yüzeye göre kesiliyor mu

Aynı domainde duman/gaz varsa: gözenekli sıvının **içinden** bakıldığında gaz,
çizilen yüzeyle aynı yerde kesilmeli.

**★ Bu maddenin varlık sebebi:** `ISO` eşiği İKİ yerde değerlendiriliyor —
gölgeleyen marş ve `nearestSurfaceSDFCrossing` (gaz→sıvı devrini seçen hakem).
İkisi ortak alandan okumazsa gaz, **render edilmeyen** bir yüzeye göre kesilir ve
hiçbir belirti vermez. Gözenek parametrelerinin materyalde değil **domain'de**
durmasının sebebi de bu: hakem başka hacimler için koşuyor ve bu domain'in
materyaline erişemiyor.

**Bozuksa ne demek:** gözeneklerin içinde duman kayboluyor ya da gözenek
olmayan yerde kesiliyorsa iki site ayrışmış.

### P6.4 — Çözünürlükten bağımsızlık

`Bubble Size` **dünya birimi** (metre). Domain çözünürlüğünü/voksel boyutunu
değiştir: kabarcık boyutu **aynı kalmalı**, sadece kenar netliği değişmeli.
Voksel-göreli olsaydı her çözünürlük değişikliği ekmeği yeniden pişirirdi.

### P6.4b — EMISSION ve TRANSMISSION artık izoyüzeye ulaşıyor

**Tek kök, üç şikâyet.** `SurfaceSample` bir **yansıma** parametre kümesi —
emission, transmission ve opacity için slotu yok. Bu yüzden izoyüzey materyal
dalı bu üçünü hiç okumuyordu. Veri yanlış değildi; **okuyacak kod yoktu.**

**Emission:** "dün çalışıyordu, şimdi çalışmıyor" tam olarak bu. Materyal
bağlamak gölgelemeyi bu dala taşıyor, bu dalda da emission yoktu — yani parlama
materyal bağlandığı anda kayboluyordu. Artık `emission` + `emission_strength`
okunuyor, emission dokusu varsa tri-planar örnekleniyor.
**Görmen gereken:** emissive bir materyal bağlı sıvı kendi kendine parlıyor.
**★ Thin-shell bunun ÜSTÜNDE return ediyor, bilerek** — üçgen dalı da öyle;
emissive bir kabarcık filmi yıkardı.

**Transmission:** `scatterGlass` — mesh'in kullandığının **aynısı**, paylaşılan
`bsdf_scatter`'dan. IOR, roughness, dispersion ve resin-tintli iç hacim mesh'le
birebir aynı davranmalı.

**★ AYIRT EDİCİ TEST (saydamlık DEĞİL, KAYMA):** sıvının arkasına şaşırtıcı bir
desen koy (damalı zemin, ince çubuk).
- Materyali kaldır → yerleşik dielektrik kırıyor, desen **kayıyor**.
- `transmission=1, ior=1.5` materyal bağla → **aynı kayma** olmalı.
- Eskiden bağlayınca kayma **kaybolurdu**; materyal bağlamak yüzeyi bu konuda
  daha yeteneksiz yapıyordu. Kontrol edilen şey bu.
- Thin-shell'de kayma **hiç olmaz** — geçiş lobu ışını bükmez. Bu ikisini
  karıştırmak kolay: "ışık geçiyor" ≠ "kırılıyor".

**★ `frontFace` SDF'den miras.** `!startInside` — marş hangi taraftan
başladığımızı zaten ölçmüş. Mesh bunu sarım sırasından okur; burada bir
**ölçüm**, ve kendi tutarlı yönelimi olmayan bir yüzeyde lobun güvenilir
olmasının sebebi bu.

**Bozuksa ne demek:** kırılma ters yönde bükülüyorsa `frontFace` ters; sıvı
girişte kararıyorsa `exitPush` ile bandın dışına çıkılmıyor (scatterGlass da
`offset_ray` ile çıkıyor, ULP ölçeğinde).

### P6.4c — BİRİKİNTİNİN YAN KENARLARI (ince duvar) — yalnızca shader

Bir havuz/birikinti yap ve **kenardan** bak.

**Görmen gereken:** yan kenar da tıpkı üst yüzey gibi arkasındaki katıyı doğru
gösteriyor; kenarda kaybolan/atlanan yüzey yok.

**★ KÖK: çıkış itmesi ince duvarı ve arkasındakini birlikte atlıyordu.**
`exitPush` bir voksel (üretimde santimetreler). İçeri giden bir lob için bu,
sıvıyı VE hemen arkasındakini tek adımda geçebilir. Üstten bakışta gövde kalın
olduğu için belli olmaz; **yan kenarda bir voksel tüm duvarı kapsar**.

★ Bunu ben ürettim: thin-shell için prob-önce-it düzeltmesini yaparken gerekçem
"yalnızca geçiş lobu düz ilerliyor, ötekiler dışarı gidiyor" idi. Sonra **cam
lobunu ekledim**, onun kırılan yönü içeri gidiyor ve o gerekçe sessizce
kapsamayı bıraktı. Kural artık tek yerde: `seatOutsideBand()` — dışarı giden
lobda hemen dönüyor, içeri gidende itme mesafesi kadar prob atıyor. **Dört
çağrı noktasının hepsi** (thin-shell, cam, resin kaplama, Principled) ondan
geçiyor.

**★ AYRI BİR OLASILIK, bu bir hata DEĞİL:** birikinti kenarı yeterince inceyse
alan ISO=0.5'e hiç ulaşmaz, dolayısıyla orada **yüzey yoktur** — sıvı değil
doğrudan zemin görünür. Bu kalınlık `Level Set Kernel Radius` ve
`Particle Voxel Radius` ile yönetilir, density ile değil. Kenarların "eriyip
kaybolduğunu" görüyorsan önce bu ikisini büyüt; hâlâ varsa yukarıdaki köktür.

### P6.5 — Tri-planar DOKU izoyüzeye ulaşıyor mu (yalnızca shader derlemesi)

İzoyüzeyin UV'si yok ve olamaz — ortada açılacak bir mesh yok, yüzey her karede
alandan yeniden kuruluyor. Şimdiye kadar sıvıya yalnızca materyalin **skaler**
değerleri ulaşıyordu; doku atamak sessizce hiçbir şey yapmıyordu.

Bağlı materyale bir **albedo dokusu** ata.

**Görmen gereken:** doku sıvı yüzeyinde görünüyor, dik yüzeylerde de yatayda da
— üç dünya düzleminden örneklenip normale göre harmanlanıyor.

**Tiling:** materyalin kendi `uv_scale`'i **dünya birimi/tile** anlamına geliyor
(yeni alan eklemedim; mesh'te mesh UV'sini çarpıyor, burada dünya konumunu).
`uv_scale` çok büyükse doku gürültüye döner, çok küçükse tek renge.

**★ Roughness/metallic dokusu ORM kanal politikasını mesh ile PAYLAŞIYOR**
(`pbr_texture_policy.glsl` artık bu shader'a da dahil). Aynı doku mesh'te ve
sıvıda **aynı kanaldan** okunmalı. Farklı okunuyorsa politika kopyalanmış
demektir — kopyalamamak için include edildi.

**★ BEKLENEN SINIR (bug değil): akan sıvı dokunun İÇİNDEN kayar.** Projeksiyon
dünya uzayına çapalı. Hamur/durgun havuz için doğru; dökülen sıvı için değil —
gerçek çözüm advected UVW, henüz yok. Aynı sınır resin iç yapısında da var.

**★ Normal haritası BİLEREK bağlanmadı.** Tri-planar normal mapping düzlem
başına tanjant çerçevesi + whiteout harmanı ister; yarım yapılırsa aydınlatma
hatası gibi görünür, eksik özellik gibi değil.

### P6.6 — Script + kalıcılık

```powershell
Invoke-RtIpc fluid.set_param @{ domain='<ad>'; pore_amount=0.25; pore_scale=0.03; pore_detail=0.6 }
Invoke-RtIpc fluid.get @{ domain='<ad>' }     # pore_* geri okunmalı
```
Sonra projeyi kaydet → kapat → aç: değerler korunmalı (`.rtp` + sahne
serileştiricisinin ikisine de yazıldı).

---

## PARTİ 5 — thin-shell + resin izoyüzeye taşındı

Parti 4'te izoyüzey tam Principled BSDF'e bağlandı, ama üç yapı hâlâ dışarıda
kalıyordu: bunlar üçgen yolunda `scatterPrincipled`'dan **önce**, ayrı tip
dalları olarak dağıtılıyor. Bu partide ikisi taşındı (thin-shell + resin).
Alpha/gözeneklilik taşınmadı — o **iso seviyesi modülasyonu** olarak ayrı
parti (alpha-cutout'un gözenek kenarında normali yok).

**İki ayrı adım gerekiyor, karıştırma:**
- Maddeler 3–7: shader → **`.spv` derlenip `x64/Release/shaders/`'a
  kopyalanmalı**, sonra yeniden başlat.
- Maddeler 0–2: C++ (`RtApi.cpp`, `RtApi.h`, `ParticleSimulation.h`,
  `scene_ui_simulation_domains.cpp`) → **build gerekiyor.**

**★ `rt_payload.glsl` değişti — onu dahil eden HER shader yeniden derlenmeli:**
`photon.rgen`, `raygen.rgen`, `closesthit.rchit`, `hair_closesthit.rchit`,
`sphere_closesthit.rchit`, `volume_closesthit.rchit`, `miss.rmiss`,
`shadow_miss.rmiss`. Yalnızca `volume_closesthit`'i derlersen `MAT_FLAG_BUBBLE`
eski yerinden gelmeye devam eder ve sessizce eski `.spv`'lerle karışık bir
pipeline kalır.

Neden değişti: `photon.rgen` hem `rt_payload.glsl`'i dahil edip hem de
`BOUNCE_*` sabitlerini yerel olarak yeniden tanımlıyordu (payload ABI tek
kaynağa taşınırken atlanmış) — redefinition hatası oradan. Aynı kontrolde
`MAT_FLAG_BUBBLE`'ın yalnızca `closesthit.rchit`'te yerel tanımlı olduğu, yani
thin-shell dalının izoyüzeyde **derlenmeyeceği** de çıktı; sabit
`rt_payload.glsl`'e taşındı ve yerel kopya kaldırıldı.

**★ Bu tam olarak "hepsini derle" talimatının işe yaradığı yer.** Yıllardır
yeniden derlenmemiş shader'larda aynı sınıftan iki latent kopya daha çıktı ve
temizlendi:
- `hair_closesthit.rchit` → yerel `INV_PI` (rt_payload'daki ile aynı değer)
- `sphere_closesthit.rchit` → yerel `RAY_OFFSET` + `BOUNCE_SPECULAR/DIFFUSE/
  TRANSMISSION` (hepsi payload ABI'nin birebir kopyası)

Tüm stage'leri include'larına karşı taradım; kalan iki eşleşme **zararsız**:
`material_program.glsl`'deki `TWO_PI` fonksiyon-içi (yasal gölgeleme) ve
`PNanoVDB.h`'deki `pnanovdb_buf_t` preprocessor'la kapalı bir dalda
(`PNANOVDB_BUF_GLSL`, shader'lar `PNANOVDB_BUF_CUSTOM` tanımlıyor). İkisi de
bugün derleniyor.

Build almadan shader'ı derlersen dallar çalışır ama scriptten süremezsin —
madde 2 hata verir. Bu ara durum beklenen.

---

### 0. UI: yeni fluid domain'de "Liquid Visualization" ALTI dolu mu

Ayrı bir hata, shader'la ilgisi yok — **sadece build gerekiyor**, en hızlı kontrol.

Yeni bir sahne aç → **Add Domain** → **Fluid (Liquid)** → doğrudan
**Shading & Rendering** sekmesi. Hiçbir şeye dokunma, hiç seed etme.

**Görmen gereken:** "Liquid Visualization & Shading" açık ve altında
"Visualization Mode" combo'su + **Level Set Kernel Radius / Particle Voxel
Radius / SDF Narrow Band…** parametreleri var.

**Bozuksa ne demek (eski davranış):** combo "Smooth Glassy Surface" yazıyor ama
**altı boş**. Kök `fluid_render_mode`'un struct varsayılanının `Volume` olması —
sıvı için **ölü konfig**. Panel onu bilerek onarmıyor (doğrusu bu: panel çizimi
sahne verisini tamir etmemeli), ama parametre blokları ham enum'a bakıyordu, o
yüzden combo bir mod gösterip altındaki hiçbir bloğa uymuyordu.

**★ Neden "seed box'a bir kez basınca düzeliyor" gibi görünüyordu:** seed
kontrollerine dokunmak sim'i koşturuyor, normalizasyon da orada
(`syncSimulationRenderVolumes`) yapılıyor. Yani belirti seed moduna bağlı
sanılıyordu; değildi — **hiç koşmamış bir sime** bağlıydı.

Ek olarak: Whitewater bölümünde "Volume foam needs Fluid Render = Surface SDF"
uyarısı, combo zaten Surface SDF gösterirken çıkmamalı.

**Eski projeler:** `.rtp`/sahne dosyaları bu alanı **her zaman açıkça yazıyor**,
yani varsayılan değişikliği yüklenen hiçbir projeyi etkilemez. Yine de bir eski
sıvı sahnesi açıp render modunun aynı kaldığını doğrula — tüketicideki
normalizasyon duruyor, literal `Volume` kaydetmiş eski dosyalar için hâlâ gerekli.

### 1. Yeni materyal parametreleri `material.set`'i geçiyor mu

En hızlı ve en bağımsız kontrol; **render gerektirmiyor.** Uygulama açıkken:

```powershell
Import-Module .\scripts\ipc\RtIpc.psm1 -Force
Invoke-RtIpc material.set @{ object_name = 'MatSwatch'; param = 'is_bubble';      value = 1.0 }
Invoke-RtIpc material.set @{ object_name = 'MatSwatch'; param = 'resin_density';  value = 0.6 }
Invoke-RtIpc material.set @{ object_name = 'MatSwatch'; param = 'resin_color';    value = @(0.92, 0.58, 0.20) }
Invoke-RtIpc material.get @{ object_name = 'MatSwatch'; param = 'resin_density' }
```

**Görmen gereken:** dördü de hatasız; son çağrı `0.6` döndürüyor.

**★ Bu maddeyi ATLAMA.** Bozuksa `material.set` bilinmeyen parametre için hata
döner ve test rig'i o case'i **varsayılan değerlerle** koşturur — yani aşağıdaki
her render "dal çalışmıyor" gibi görünür, oysa parametre hiç ulaşmamıştır.
Bu, altındaki altı maddenin hepsini maskeleyen tek madde.

**Bozuksa ne demek:** `parseMaterialParam` / `readMaterialValue` /
`writeMaterialValue` üçünden biri eksik kaldı (üçü de aynı enum'u kapsamalı;
`switch` eksik `case` verirse derleyici uyarır).

### 2. Geri okuma gerçekten materyalden mi geliyor

```powershell
Invoke-RtIpc material.set @{ object_name = 'MatSwatch'; param = 'is_bubble'; value = 0.0 }
Invoke-RtIpc material.get @{ object_name = 'MatSwatch'; param = 'is_bubble' }
```

**Görmen gereken:** `0`. **★ Sinsi hâli:** `get` her zaman `0` dönüyorsa ve
`set` hata vermiyorsa, getter `Material.h`'deki **sanal varsayılana** düşüyordur
(PrincipledBSDF override'ı çağrılmıyor) — bu "sıfır ölçtüm" gibi görünür, oysa
hiç ölçülmemiştir. Undo/redo da bu okumaya dayandığı için sessizce bozulur.

### 3. REGRESYON: `none` ve mevcut sıvı sahneleri değişmemeli

Rig'in ilk case'i. Materyal bağlı değilken sıvı **eskisiyle aynı** olmalı:
berrak, kırıcı su.

**★ Bu partide asıl regresyon riski başka bir yerde:** derinlik soğurması ve
köpük artık blok başına **bir kez**, en üstte uygulanıyor (üç dal da paylaşıyor).
Yanlış giderse sıvı ya **iki kat koyu** ya da **hiç soğurmuyor** görünür.

**Bozuksa ne demek:** `startInside` soğurma bloğu hoist edilirken ya eski
kopyası kaldı (çift sayım) ya da Principled dalının kendi kopyası tümden silindi.

### 4. thin-shell — önce KÜPTE (kontrol grubu)

`thin_shell` case'i. `closesthit.rchit`'e **dokunulmadı**, yani küpteki görünüm
bu partiden önceki hâliyle birebir aynı olmalı.

**Görmen gereken:** küp saydam, kenarlarda parlak gümüş rim, film kalınlığından
gelen renk kayması.
**Bozuksa ne demek:** sorun bu partide değil; ortak `bsdf_scatter.glsl`'de veya
materyal yükleme yolunda. Alttaki maddeye geçme, önce burayı çöz.

### 5. thin-shell — SIVIDA

**Görmen gereken:** sıvı da saydam + parlak rim. Küple aynı madde okunmalı
(birebir değil — sıvı eğri ve kenarda ince).

**✔ İLK TURDA ÇALIŞTI** (2026-08-12): SDF yüzeyi ince köpük gibi render etti.
Ama kabuğun üzerinde **siyah lekeler** çıktı ve düzeltildi — aşağıdaki kontrol
artık o düzeltmenin doğrulaması:

**Görmen gereken:** kabuğun üstünde siyah/boş leke YOK; kürenin arkasındaki
zemin her yerden görünüyor.

**★ KÖK (düzeltildi): düz geçiş bir SIÇRAMA HARCIYORDU.** `isTransparentPass`
([raygen.rgen](../RayTrophiStudio/source/shaders/raygen.rgen)) bir geçişi ancak
yön değişmemişse **ve** (attenuation tam 1.0 **veya** etiket
`BOUNCE_TRANSPARENT`) ise bedava sayıyor. Film her zaman hafifçe renklendiriyor
(0.85..1.0), yani attenuation testi asla geçemez — iki lob da `BOUNCE_SPECULAR`
etiketliyken **her kesişim tam bir GI sıçraması yiyordu.** Mesh'te kabuk iki
kesişim, hayatta kalıyor; izoyüzeyde level-set bandı defalarca kesiliyor, bütçe
ışın arkadaki zemine varmadan bitiyor ve raygen döngüyü kırıyor. Lekelerin
kabuğun EN KALIN olduğu yerlerde toplanması bu yüzden.

Artık geçiş lobu `BOUNCE_TRANSPARENT`, yansıma lobu `BOUNCE_SPECULAR`.
**★ Aynı hata üçgen yolunda da vardı** (`closesthit.rchit`) ve orada da
düzeltildi — mesh'te ölümcül değildi ama üst üste binen kabarcıklar (şampanya,
sabun köpüğü) bedelini ödüyordu. Yani bu düzeltme mesh render'larını da etkiler.

**★★ İKİNCİ KÖK (2. turda çıktı): çıkış itmesi ARKADAKİ YÜZEYİ ATLIYORDU.**
Sıçrama bütçesi düzeldikten sonra "thin shell hâlâ arkadaki yüzeyi atlayabiliyor"
kaldı. Geçiş lobu `hitPos + rayDir * max(3mm, voxel_size)` ile ilerliyor —
5 cm'lik vokselde **5 cm ileri sıçramak** demek. Filmin hemen arkasındaki katı
yüzey o aralıktaysa temiz atlanıyor, ışın onun ötesinde yeniden başlıyor ve o
yüzey hiç render edilmiyor.

★ Bu **yalnızca geçiş lobuna** özgü: tek düz ilerleyen lob o. Yansıma lobu ve
Principled'ın saçılma lobları yüzeyden dışarı gidiyor.

Çözüm: itmeden önce **sadece itme mesafesi kadar** bir prob atılıyor
(`TerminateOnFirstHit`, birkaç cm, marşın maliyetinin yanında hiç). İçinde katı
varsa ışın, giriş tarafındaki solid probe'un yaptığının aynısıyla devrediliyor:
`skipAABBs` ile hit noktasından devam, üçgen closest-hit gerçek mesafe ve gerçek
yönüyle ateşleniyor.

**Görmen gereken:** filmin arkasındaki küre/zemin her açıdan görünüyor; ince
köpük saydam olduğu yerlerde arkasındakini gösteriyor.
**Bozuksa ne demek:** prob maskesi (`0xF1`) yanlış geometriyi eliyor olabilir —
giriş tarafındaki probe ile aynı maske, ikisi ayrışırsa biri diğerinin görmediği
yüzeyi görür.

**★ İkinci sinsi hâl:** köpük (whitewater) kayboluyorsa, üçgen dalındaki
`payload.radiance = vec3(0.0)` satırı yanlışlıkla kopyalanmıştır — bu shader
radyansı **biriktiriyor** (`+=`), sıfırlamıyor.

### 6. resin — berrak kaplama (`resin_clear`)

**✔ ÇÖZÜLDÜ (2026-08-12): resin çalışıyor. Kod değil, KAPI kapalıydı.**

Yeni bir derleme yapılmadan çalıştı — çünkü resin dalı zaten derliydi
(`volume_closesthit.spv` thin-shell turunda derlenmişti ve resin kodu o dosyada
ilk partiden beri duruyordu). Değişen tek şey materyal ayarı oldu:
`Interior Depth` = `transmission_density` ve kapı `> 1e-4`.

★ Üç bağımsız kaynak baştan aynı şeyi söylüyordu: panelde `Interior Depth 0,00`,
her `[IsoMat]` satırında `resin_density=0,000000`, ve ilk satırda `bubble=1`
(thin-shell resin'den ÖNCE return ediyor). Ölçüm doğruydu; ben itiraz üzerine
onu geçersiz ilan ettim. **İtiraz veridir, ama ölçüm de veridir — uzlaştır,
birini atma.** Doğru hamle "hangi materyalde, hangi değerlerle?" diye sormaktı.

Aşağıdaki "ölçüm geçersiz" gerekçesi **yanlış çıktı**, ama taşıdığı enstrüman
yine de duruyor ve işe yarar — kaydı için:

---

**~~ÖLÇÜM GEÇERSİZ~~ (bu gerekçe yanlıştı, bkz. yukarısı)**

`[IsoMat]` satırı **hacim paketi yeniden kurulurken** çalışıyor. Gözlemlenen iki
yeniden basma da bunu doğruluyor: `bubble` ve `transmission` değişimleri
BLAS/opacity senkronu tetikliyor ("Material sync flipped BLAS opacity
override(s)"), o yüzden log koştu. **Interior Depth ise ne opacity'yi ne BLAS'ı
etkiliyor** → paket kurulmuyor → log koşmuyor. Yani `resin_density=0` satırı
**bayat**; ondan "resin sıfır" sonucu çıkarılamaz.

★ Bu tam olarak "tripwire'ın susması yokluğu kanıtlamaz; enstrüman ölçtüğü şeyle
çakışmamalı" dersinin tekrarı. Ölçüm **tüketiciye** taşındı: `VulkanBackend.cpp`
materyal yükleme döngüsü — GPU'ya giden son host kodu. Konsolda `[MatUpload]`
ara, `[IsoMat]`'in verdiği **indeksle** eşleştir:

```
[MatUpload] idx 14 -> GPU gets  bubble=0 transmission_density=2.000000 ...
```

- **`transmission_density` burada > 0 ise** → GPU doğru veriyi aldı, sorun
  shader dalının içinde; sıradaki adım dala debug view koymak.
- **idx 14 hiç görünmüyorsa veya density 0 ise** → değer GPU'ya hiç ulaşmıyor;
  sorun host tarafında, shader'da değil.

Eski (geçersiz) ölçüm, kaydı için:

```
[IsoMat] 'Obj_2_Material' (id 14) -> plain Principled (no resin: Interior Depth is 0)
         [bubble=0 resin_density=0,000000 transmission=0,000000]
```

İzoyüzeye bağlı materyalin `resin_density`'si **0**. Kapı
(`transmission_density > 1e-4`) hiç açılmamış, yani resin kodu hiç koşmamış —
"etkisiz" gözlemi doğruydu, sebebi shader değildi.

**★ SIRADAKİ AYIRT EDİCİ ADIM.** `Obj_2_Material`'ın **Interior Depth**'ini
0'dan büyük yap (0.6 iyi) ve konsolu izle:
- `[IsoMat]` satırı **yeniden basıyor ve `resin_density=0.600000` diyorsa** →
  değer ulaştı, artık gerçekten resin'e bakıyoruz.
- Satır **yeniden basmıyorsa** → ★ bu ayrı bir hata: materyal düzenlemesi
  izoyüzey bağlamasını yeniden yayınlatmıyor demektir. Log bağlama anında
  çalışıyor; materyal sonradan değişince yeniden çalışmıyorsa **log bayat** bir
  anlık görüntü basıyor olabilir. O durumda önce bunu düzeltmek gerek, yoksa her
  ölçüm yanıltır.

Aşağıdakiler zaten elendi (sorun bunlar değil):

Kod okuyarak bulunamadı. Elenenler (yani sorun bunlar DEĞİL):
- Host→GPU yazımı var (`VulkanBackend.cpp:12187/12682` → `makeMaterialExt`)
- `MaterialExt` host/GLSL layout'u alan alan **birebir** uyuşuyor
- Panelin "Interior Depth"i doğrudan `setTransmissionDensity` yazıyor, kapı yok
- Kapı mantığı ve etki büyüklüğü doğru: amber tint + Interior Depth 2.0'da taban
  albedo'su ≈`exp(-8·ext)` ile çarpılır — bu güçlü bir etki, "belli belirsiz"
  olamaz. Yani dal ya hiç girilmiyor ya da girip görünmüyor.
- Thin-film iridescence çalıştığına göre `MaterialExt` shader'a ULAŞIYOR —
  ★ ama bu yalnızca struct'ın **ilk iki float'ını** kanıtlar (`bubble_ior`,
  `bubble_film`); resin alanları çok daha ileride. Layout'u bu yüzden ayrıca
  karşılaştırdım.

**Bu turda ölçüm için üretici kapısına log kondu** (`VolumetricRenderer.cpp`,
`logIsoMaterialBinding`). Konsolda `[IsoMat]` satırını ara — sıvı yüzeyine
materyal bağlandığı anda, DEĞİŞİMDE bir kez basar:

```
[IsoMat] 'AmberCoat' (id 7) -> RESIN coat  [bubble=0 resin_density=0.600000 transmission=0.000000]
```

Bu satır soruyu ikiye böler:
- **Satır hiç yoksa** → izoyüzeye materyal hiç bağlanmıyor; sorun shader'da
  değil, bağlama zincirinde (`fluid_surface_material_id` → … →
  `iso_material_index`).
- **Satır `-> RESIN coat` diyorsa ama görüntü değişmiyorsa** → değerler doğru,
  sorun shader dalının İÇİNDE. O zaman bir sonraki adım dalın içine bir debug
  view koymak.
- **`-> plain Principled (Interior Depth 0)` veya `-> THIN SHELL` diyorsa** →
  o gate kapalı; hangisi olduğunu satır zaten söylüyor.

Ayrıca script tarafından da doğrulanabilir (bu partide açıldı):
`Invoke-RtIpc material.get @{ object_name='<obj>'; param='resin_density' }`

**Görmen gereken:** sıvı geçirmeyi bırakıyor, üstünde parlak amber bir **deri**
var; renkli cam gibi değil, kaplanmış gibi.

**★ Beklenen ama bug sanılacak şey:** taban aynı spp'de küpten **daha gürültülü.**
İzoyüzey dalında NEE yok, taban yalnızca dolaylı sıçramayla aydınlanıyor. spp
artınca temizleniyorsa normal. **Düz siyah kalıyorsa** normal değil — o zaman
`ss.albedo` soğurmada sıfıra gitmiş demektir (`resin_color` çok koyu değilse
`ext` hesabına bak).

**★ Ayrıca doğrula:** `molten_glass` case'i (transmission=1.0) **değişmemiş**
olmalı. Cam/resin ayrımı üçgendeki gibi stokastik: hem transmission hem
`resin_density` taşıyan materyal amber gövdedir, kaplama değil. Bozulursa
mevcut sahnelerdeki her amber/jade materyali sessizce kaplamaya döner.

### 7. resin — iç yapılar ve ÇAPA (`resin_inclusions`)

Önce durağan kare: küple sıvıdaki toz/zerre yapısı **aynı ölçekte ve renkte**
olmalı.

Sonra **sim'i adımla ve zerrelere bak.** Mesh iç yapıyı objeye çapalar; sıvı
yalnızca **domain'e** çapalayabilir — yani akışkan, yerinde duran bir desenin
içinden akar.

**Görmen gereken:** sıvı akarken zerreler domain'e göre sabit.
**Bu bir kusur değil, belgelenmiş sınır** (gerçek taşıma advected UVW ister,
henüz yok). **Ama küpte de sabitlerse** `resin_object_space` mesh yolunda
çalışmayı bırakmış demektir — o bir regresyon.

---

## Rig

```
python scripts\test\rt_test_fluid_surface_material.py
```

Uygulamanın **dışından**, ayrı terminalden. `scripts/` ve
`x64/Release/scripts/` senkronlandı.

Yeni case'ler: `thin_shell`, `resin_clear`, `resin_inclusions`.
Çıktı: `renders/fluid_surface_material/`.

---

## Bu partide TAŞINMAYAN (bilerek)

- **Alpha / gözeneklilik.** Sıradaki parti: `ISO_THRESH`'i prosedürel 3B alanla
  modüle etmek. ★ Kritik: `ISO` sabiti **iki yerde** —
  `nearestSurfaceSDFCrossing` (gaz↔sıvı devrini seçen hakem) ve ana march.
  Yalnızca ikincisi modüle edilirse gaz, gözeneklerin olmadığı bir yüzeye göre
  kesilir ve hiçbir belirti vermez.
- **Node graph izoyüzeyde.** `evalMaterialProgram` bu shader'da zaten var ama
  `MP_REGISTER_COUNT` **12** (üçgende 32). 12'yi aşan Principled graph
  `prog.active=false` ile **sessizce** devre dışı kalır.
- **Cam kırılması izoyüzeyde.** `SurfaceSample`'da transmission lobu yok.
- **Gölgede gözenek/film.** Hacim gölge yolu yaklaşık (`volume_shadow_strength`).
