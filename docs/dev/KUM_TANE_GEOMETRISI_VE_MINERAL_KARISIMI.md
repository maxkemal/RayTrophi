# Taneli madde: gerçek tane geometrisi ve bileşen karışımı

> **Durum:** AKTİF 2026-10-22 (rev. 6: §8 K0 + gerçek shard + mineral karışımı CANLI ölçüldü; materyal derleyici düzeltmesi derleme bekliyor) — §6b KARAR: şekil mevcut generator + üçgen instancing (T0 hibrit), `rint` çokyüzlüsü yedekte; K3 tane içi yol geçerli.

**Devir notu (2026-10-22):** K0 = yakın-kamera tane mesh'leri + ölçüm. Kod:
`Fluid/GrainMeshNear.{h,cpp}` (yeni), `ParticleRenderBridge.cpp` (tek parçacık
yürüyüşünde küre/mesh ayrımı, ikinci InstanceGroup), domain alanları
`fluid_grain_mesh_near_*`, `fluid.set_splat_geometry near_mesh_*`, `fluid.get
grain_mesh_near`, yeni `render.instance_stats`, panel "Near Grain Meshes".
**K0 ÖLÇÜLDÜ (§8):** hibrit çalışıyor (yakın %17 tane 12 üçgenli mesh → +%9),
ama üçgen sayısı bedava değil (1280 üçgen aynı instance'ta 4.8× ek maliyet) ve
prosedürel küre tane başına en ucuz. **Rev. 6 (§8 sonu):** gerçek Voronoi
shard'ları (~238 üçgen/tane) + mineral karışımı materyali canlıda: opak kumda
shard'lar **bedava** (hepsi mesh ±%0), asıl maliyet kuvars transmission'ı
(2.7×). Materyal programı derleyicisinde bağlantısız renk pini hatası bulundu ve
düzeltildi (derleme bekliyor). Sıradaki: havuz boşluğu düzeltmesini (yazıldı)
derleyip ölç → transmission maliyeti (K3 tane içi yol) → K1b. Kapsam **bütün taneli maddeler** (kum yalnız ilk örnek). Arka uç önceliği: **Vulkan RT → RayFusion (bu sahne tipi için önerilmez) → CPU referans**. RT render maliyeti başka ajanda optimize ediliyor; `closesthit.rchit`'e dokunmadan önce onunla eşlenmeli.

Soru: birleşik madde domain'inin splat'i bugün her taneyi **prosedürel küre**
olarak çiziyor. Gerçek bir tane hangi geometride olmalı, ve bir taneli maddenin
**çoklu bileşeni** (kumda camsı kuvars + feldispat + kabuk + ağır mineral; çakılda
farklı taşlar; karda buz kristali + hava; tuzda küp kristal) render'a nasıl
girmeli?

**Rev. 2 düzeltmeleri (kullanıcı):** yapı sahil kumuyla sınırlı değil; ıslaklık
fiziği ve ıslaklığa göre materyal ataması **zaten var**; tek toplu BLAS zaten
var; **CPU'da ışın ön-hesabına gerek yok** — her şey Vulkan RT'de; material
node'da tane başına **Random** varyasyonu zaten var.

---

## 1. Bugün elimizde ne var (kodu okuyarak)

**Geometri** — `VulkanBackend_Spheres.cpp::gatherSphereCloudUnchecked`:
- Her `point_sphere_mode` grubunun her instance'ı için `point_sphere_visual_children`
  kadar çocuk küre; bütün bulut **tek AABB BLAS / tek TLAS instance**.
- `sphere_intersection.rint` analitik ışın–küre; `closesthit.rchit` normali
  merkezden hesaplar (`FOAM_SPHERE_HIT_KIND`).
- Kayıt `FoamSphereGPU` = 32 bayt: `centerRadius`, `matId`, `objectRandom`,
  **`_p1`, `_p2` boş** → tane başına 64 bit veri kayıt büyümeden eklenebilir.

**Islaklık → materyal (zaten çalışan zincir):**
1. Parçacık başına gözenek suyu: `pore_water_mass_kg` / `pore_capacity_kg`
   (tek sahip, defter bir kez sayar) → `matterParticleSaturation`.
2. Fizik tarafı: `evaluateMatterWetResponse` → sürtünme, dilatans, kılcal
   kohezyon (s·(1−s) tepeli), gözenek basıncı.
3. Görünüm: `matterWetAppearanceBand` doygunluğu **1–7 bandına** böler
   (`wet_appearance_full_saturation`'a göre).
4. `MatterWetPalette` (`MatterWetAppearance.cpp`) her taneli madde için kuru
   materyalden **7 `[MatterWet]` varyantı** türetir (`wet_color_scale`,
   `wet_roughness_scale`).
5. `ParticleRenderBridge` parçacığın `source_index`'ini
   `wet_palette.sourceIndex(...)` ile seçer → küre kaydının `matId`'si.

Yani **materyal seçimi zaten parçacık başına, madde × ıslaklık bandı**.

**Tane başına varyasyon (zaten çalışan):** material node'da `Object Info →
Random` (`MaterialNodesV2.h` `ObjectInfoNode`); splat'te değeri kayıttaki sabit
`objectRandom` verir (düşen tane rengini değiştirmez). Materyal programı
(`material_program.glsl`) albedo, roughness, **transmission ve IOR** slotlarını
sürebildiği için bir madde materyali bugün bile Random → ramp ile taneden
taneye renk, opak↔yarı saydam, IOR farkı üretebilir.

Sonuç: **görünüm amaçlı bileşen karışımı için yeni kod gerekmez** — bir
materyal grafiği yeter. Kod gerektiren tek şey bileşenin **simülasyondan
gelmesi** (yoğunluk, ayrışma) ve şekil.

## 2. Referans: Piper ve sonrası

Kaynak: renderman.pixar.com/stories/piper (fxguide'daki metin aynısı).

| Konu | Piper |
|---|---|
| Neden gerçek tane | Displacement yakın planda normal çözünürlüğün **~36 katını** istedi → instancing |
| Tane | Houdini'de prosedürel, **~5000 poligon**, bump/displacement yok |
| Örnek sayısı | "yalnızca **12** kum tanesi" + primvar kontrolü |
| Yerleşim | Poisson dağılımı, **ince yüzey katmanı**; frustum + bakış açısı + mesafe ayıklaması |
| Malzeme | Gerçek sahilden albedo paleti; **kabuk, taş, kırıcı cam** shader'ları, her biri ıslak/kuru |
| Islaklık | Islak / Nemli / Kuru; su çizgisinde **dinamik IOR geçişi** |
| Maliyet | Bazı sahnelerde **ray depth 128** |

Annecy 2017 (3dvf): 50, 500 ve 5000 poligonlu tane render süresini neredeyse
değiştirmedi. **Maliyet poligonda değil, kırıcı tanenin içinde dolaşan yolun
derinliğinde.** Bu notun bütün tasarımı bu tek gözleme dayanıyor.

*Elio* (SIGGRAPH 2025, "Fine Tuning Every Grain on the Beach"): kum varlığı
tamamen **noktalardan** — bizim "nokta → prosedürel tane" yapımıza Piper'ın
mesh instancing'inden yakın. Meng ve ark. (SIGGRAPH 2015): yakında açık tane,
uzakta türetilmiş hacimsel ortam; 2 milyar tanede 12× hız.

## 3. Genel tane modeli (her taneli madde)

Bir taneli madde **bileşenlerden** oluşur. Her bileşenin dört özelliği var;
kum, kar, çakıl, tuz, şeker, kahve, toprak hep bu tabloya düşer:

| Özellik | Ne | Neyi sürer |
|---|---|---|
| Pay | hacim/kütle oranı | parçacığa atanma (önce görünüm, sonra sim) |
| Şekil | yuvarlaklık, küresellik, form (eşboyutlu / pul / plaka / küp / dendrit) | kesişim |
| Optik | opak difüz / pürüzlü dielektrik / metalik parlak; IOR, iç soğurma rengi | closesthit |
| Yoğunluk | kg/m³ | sim (ayrışma) |

Örnekler:

| Madde | Bileşenler | Şekil | Optik notu |
|---|---|---|---|
| Kuvars kumu | kuvars, feldispat, kabuk, mika, ağır mineral | yarı köşeli, pul (mika), plaka (kabuk) | kuvars **kırıcı dielektrik**, n≈1.55 |
| Karbonat (tropik) kum | kabuk/mercan kırığı | plaka, dallı | yarı saydam, sedef |
| Siyah kum | volkanik cam, bazalt | köşeli konkoidal | camsı, koyu iç soğurma |
| Kar / buz tanesi | buz (n≈1.31) | yaşa göre: dendrit → yuvarlak → köşeli kristal | kırıcı, **çok düşük soğurma** — en derin yollar burada |
| Tuz | halit (n≈1.54) | **küp** | kırıcı, keskin fasetler |
| Şeker | sakaroz (n≈1.54) | prizmatik | kırıcı |
| Çakıl | farklı taşlar | yuvarlak | opak; yüzey dokusu |
| Toprak | mineral + organik | düzensiz | opak, ıslakken çok koyulaşır |

★ Kırıcı bileşen nerede varsa (kuvars, buz, tuz, şeker, cam) "ray depth 128"
problemi oradadır. Opak bileşenler (çakıl, toprak, ağır mineral) bugünkü
küreyle zaten ucuzdur, onlara yalnızca şekil kazandırır.

### Kum için mineral tablosu (ilk örnek)

"Camsı silikon parçalar" = **kuvars (SiO₂, silisyum dioksit)**; kristal ama
optik olarak cama çok yakın.

| Bileşen | Tipik pay | Optik | n | ρ g/cm³ |
|---|---|---|---|---|
| Kuvars | ılıman kıta sahilinde çoğu kez %60–90+ | pürüzlü dielektrik, sütlü/bal tonu | 1.544/1.553 | 2.65 |
| Feldispat | %5–25 | yarı saydam–opak, krem/pembe | 1.52–1.58 | 2.56–2.76 |
| Taş kırıntısı | değişken | opak difüz | — | ~2.7 |
| Kabuk (aragonit/kalsit) | ılımanda az, tropikte ~%100 | opak–yarı saydam | 1.486/1.658 | 2.7–2.95 |
| Mika | <%1–birkaç | ince pul, ayna parıltısı | 1.56–1.60 | 2.8–3.1 |
| Ağır mineral (manyetit, ilmenit, granat) | <%1–5 | opak metalik / kırmızı yarı saydam | granat 1.72–1.89 | **4.5–5.2** |

Boyut: kum 0.0625–2 mm, sahil tipik 0.25–0.5 mm, iyi boylanmış (σφ ≈ 0.35–0.5).
Değerler standart mineraloji tablolarından, bellekten — uygulamada doğrulanmalı.

★ Yoğunluk sütunu simülasyon içindir: ağır mineral kuvarsın ~2 katı, dalga onu
**koyu çizgiler** halinde ayrıştırır. Bileşen parçacıkta sim'de taşınırsa bu
kendiliğinden çıkar; render'da rastgele sayıdan türetilen bileşen bunu asla
yapamaz.

## 4. Öneri

### 4.1 Şekil: yuvarlatılmış dışbükey çokyüzlü (aynı AABB BLAS)

- Tane = **küre ∩ K yarı-uzay** (K = 6–14). Kesişim slab testinin
  genellemesi: düzlem giriş t'lerinin maksimumu, çıkış t'lerinin minimumu,
  küreyle kırp. K adımlı dallanmasız döngü, **tam** (sphere tracing yok).
- Yuvarlaklık = düzlem uzaklığı / küre yarıçapı. Küresellik = düzlem kümesine
  anizotropik ölçek. Pul/plaka (mika, kabuk) = iki yakın paralel düzlem.
  Küp (tuz) = 6 düzlem, yuvarlaklık ≈ köşe kırpma.
- Düzlem normalleri bileşen başına **12–32 örnekten oluşan sabit bir tablodan**
  (Piper'ın 12'si gibi), GPU'da bir kez; yönelim `objectRandom`'dan.
- `_p1` = örnek şekil id + bileşen id + yuvarlaklık sapması; `_p2` = yönelim
  tohumu / ileride sim kimliği. **Kayıt 32 bayt, BLAS düzeni aynı, AABB =
  çevreleyen küre** (değişmez, refit gerekmez).
- Islaklık bandı zaten `matId`'nin içinde — `_p1`'e ayrıca yazılmaz; **tek
  otorite** palet kalır.

### 4.2 Işık taşınımı: tane içi yol closesthit'te analitik

CPU ön-hesabı **yok**, tablo da yok. Tane dışbükey ve analitik olduğu için
içindeki yol yeni `traceRayEXT` gerektirmez:

1. Işın taneye girer (Fresnel ile yansıma/kırılma seçimi — mevcut dielektrik
   lob).
2. Kırılan ışın **aynı K düzleme** karşı içeride yürütülür: dışbükey cisimde
   içeriden atılan ışın tam olarak **bir** çıkış yüzüne çarpar → K adımlık ALU.
3. Çıkışta Fresnel: çıkar ya da toplam iç yansıma ile döngüye döner; iç
   soğurma Beer–Lambert ile (bileşenin soğurma rengi × iç yol uzunluğu).
4. Sınır: N iç olay (ör. 6–8); sınırda kalan enerji soğurulmuş sayılır ve
   sayaçla raporlanır (sessizce kaybolmasın).
5. Çıkan ışın **sahneye tek bir yeni ışın** olarak devam eder.

Sonuç: kırıcı tane başına yol derinliği **1** artar (Piper'da tane başına 2+
iç sekme ve her biri ayrı ışın). Maliyet ışın sayısından ALU'ya taşınır — RT
maliyet çalışmasının tam istediği yön. Bu, Meng 2015'in "tane içini ayrı süreç
olarak modelle" fikrinin ön-hesapsız hali.

**Islaklık bu yolda bedava fizik olur:** tanenin dışındaki ortamın kırılma
indisi = karışım(hava 1.0, su 1.33; bant doygunluğu). Kuvars/su göreli indisi
≈ 1.165'e düşünce yüzey yansıması azalır, ışık derine gider, soğurulur → koyu
(Lekner & Dorf 1988; Piper'ın "dinamik IOR geçişi"). Bugünkü `[MatterWet]`
varyantları albedo/pürüzlülüğü **ölçekliyor**; kırıcı bileşende band
varyantının taşıdığı şey ölçek yerine **dış ortam IOR'u** olabilir. Opak
bileşenlerde mevcut ölçek yolu kalır. (Karar açık — bkz. §6.)

### 4.3 Mesafe katmanları

| Katman | Ne zaman | Ne |
|---|---|---|
| T0 açık tane | tane ekranda ≳ 2 px | 4.1 şekli + 4.2 tane içi yol; **yalnız yüzey katmanı** ve frustum içi |
| T1 splat | tane < 2 px | bugünkü küre; bileşen materyali paletten |
| T2 yüzey | uzak | yüzey + toplu BSDF (bileşen karışımının ortalaması, karşıt-etki tepesi) |

T2, T0/T1 materyallerinden **türetilir**, ayrıca ayarlanmaz; yoksa uzak ve
yakın aynı madde farklı renk olur ("panel yalan söylüyor" sınıfı).

### 4.4 Arka uçlar

| Arka uç | Rolü |
|---|---|
| **Vulkan RT** | Hedef. 4.1 + 4.2 burada yazılır, burada doğrulanır. |
| RayFusion | Bu sahne tipi için önerilmez. T1 davranışı (küre + bileşen materyali) yeterli; tane içi yol taşınmaz. |
| CPU referans | En son. 4.2 saf bir fonksiyon (düzlemler + IOR + soğurma → çıkış ışını); aynı matematik referansa sonra taşınır, öncelik değil. |
| OptiX | Dondurulmuş — yeni özellik almaz. |

## 5. Veri sahipliği (CLAUDE.md §1)

- Bileşen listesi (pay, şekil, optik, yoğunluk) **maddenin** alanıdır
  (`MaterialStateField` `grain_*` yanında); panel yalnız düzenler, kopya tutmaz.
- Bileşen **ayrı materyal değil, materyal grafiğine bir giriş**: tane başına
  bileşen id'si (`_p1`'den) bir `Attribute` / Object Info çıkışı olarak okunur;
  bileşen içi renk/ton varyasyonu mevcut Random'dan gelir. Böylece palet
  (madde × ıslaklık bandı) **dokunulmadan** kalır, materyal sayısı patlamaz
  (bileşen × 8 varyant yerine madde başına yine 8).
- IPC: `matter.*` altında bileşen listesi oku/yaz; tane sayaçları
  (bileşen başına görünür tane, T0'da iç olay sınırına takılan pay).
  Capability + descriptor overlay satırı unutulmamalı.

## 6. Önerilen sıra

1. **K0 — ölçüm, kod yok:** diğer ajanın RT perf ölçümlerinde splat'in payı;
   kesişim mi gölgeleme mi derinlik mi?
2. **K1 — görünüm, KOD YOK:** kum materyal grafiği: `Object Info → Random` →
   eşik/ramp ile kuvars (transmission, IOR 1.55, pürüzlü) / feldispat / kabuk /
   ağır mineral payları. Islak bantlar bu grafiğin kopyalarını zaten türetiyor.
   Amaç: hangi bileşen görünümünün ışık bütçesini zorladığını K3'ten önce
   görmek (transmission'lı taneler derinliği hemen gösterir).
   **K1b — kod:** bileşen id'si tane kaydında (`_p1`) + materyal grafiğinde
   okunabilir çıkış; atama şimdilik paylarla, K4'te sim'den.
3. **K2 — şekil:** yuvarlatılmış çokyüzlü kesişimi + örnek tablo, T0'da.
4. **K3 — tane içi analitik yol (closesthit).** K2 ile birlikte ölç: yol
   derinliği ve ms sayıyla düşmeli.
5. **K4 — bileşen sim'de:** yoğunluk parçacıkta; ağır mineral ayrışması
   kendiliğinden çıkmalı. Madde planı kapanış partileri bitmeden açılmaz.
6. **K5 — T2 toplu BSDF.**

Açık karar: kırıcı bileşende ıslaklık bandı **dış ortam IOR'u** mu taşısın,
mevcut albedo/pürüzlülük ölçeği mi kalsın? IOR yolu fiziksel ve Piper'ın
yöntemi; ama ıslaklığı iki farklı mekanizmayla gösteren iki bileşen tipi demek.

★ **Sinsi başarısızlık uyarısı:** K1'deki Random tabanlı bileşen her şeyi makul
gösterir. K4'te bileşen sim'e geçtiğinde materyal grafiği hâlâ Random'dan
bileşen seçiyorsa render ile sim **iki ayrı otorite** olur ve kimse fark etmez
— ağır mineral ekranda rastgele yerde, sim'de çizgide. K1b'nin bileşen çıkışı
bu yüzden var: K4'ten sonra bileşen **yalnız** o çıkıştan okunmalı, Random
yalnız bileşen *içi* tonu sürmeli.

## 6b. KARAR (rev. 4): şekil = mevcut generator + üçgen instancing, `rint` çokyüzlüsü DEĞİL

Kullanıcı sorusu: önerilen analitik çokyüzlü mü, yoksa mevcut prosedürel
generator'ı doğru yöne çevirip gerçek geometri mi? Cevap: **gerçek geometri.**

**Neden §7.2 yanlış yöne bakıyordu:**
- Üçgen kesişimi RT donanımında; `rint` her AABB adayında **SM'de yazılım**
  olarak koşar. §7.0'da kendimiz yazdık: bu shader kamera + sekme + gölge +
  hacim probe ışınlarının hepsinde çağrılır. 14 düzlemli döngü, 50–200 üçgenlik
  bir tane BLAS'ından büyük olasılıkla **daha pahalıdır**.
- Piper'ın ölçümü aynı şeyi söylüyor: 50/500/5000 poligon render süresini
  değiştirmedi. Üçgen sayısı yük değil.

**Mesh yolunun asıl maliyeti üçgen değil, INSTANCE SAYISIDIR:**
- Her tane bir TLAS instance'ı (instance kaydı + `VkInstanceData` + hareket
  eden tanede her kare TLAS build/refit). Kodun kendi yorumu
  (`ParticleRenderBridge.cpp`): "parçacık başına bir instance milyonlarda hâlâ
  ağır"; tek tavan cihazın `maxInstanceCount`'u. Küre bulutu tam bu yüzden tek
  BLAS'a toplandı.
- Bu yüzden karar **hibrit**: T0 (yakın, yüzey katmanı, frustum içi) taneler
  mesh instance; geri kalanı bugünkü küre bulutunda. T0 instance bütçesi
  ölçülen ve IPC'den okunan bir sayı.

**Altyapı zaten var:**
- `InstanceGroup.sources[]` + instance başına `source_index`; Vulkan,
  `point_sphere_mode` olmayan grupları kaynak mesh'in BLAS'ını paylaşan TLAS
  instance'ları olarak çiziyor (`VulkanBackend.cpp` scatter yolu).
- Instance başına materyal: `VkInstanceData.materialIndex` closesthit'te zaten
  okunuyor. Yani **şekil = kaynak BLAS, materyal (bileşen × ıslaklık bandı) =
  instance materyali** — ikisi ayrı eksen, kombinasyon patlaması yok. (Bugün
  küre yolunda ıslaklık bandı `source_index` ile seçiliyor; mesh yolunda bu
  eksen `materialIndex`'e taşınmalı, yoksa şekil × bant kaynak çarpımı olur.)

**Örnek tane kütüphanesi mevcut araçlarla:**
- **Voronoi fracture** (`FractureGenerator`, CONVEX mod) bir küreyi N hücreye
  böler → N **dışbükey, köşeli** tane: kuvars/obsidyen/tuz kırığı. Bonus:
  her hücre zaten yarı-uzayların kesişimi → K3 tane içi yolunun düzlem listesi
  **bedavaya** gelir.
- **Rock/Stone generator** (gürültülü küre, X/Y/Z ölçek) → yuvarlak tane,
  küresellik ölçekle.
- Geometry Nodes **noise displace** → yüzey dokusu (frosted) ve kabuk/pul için
  ölçekli düz formlar.
- Hedef çözünürlük: 40–200 üçgen/örnek, bileşen başına 12–32 örnek.

**K3 bu kararla değişmez:** tane içi analitik yol (§7.4) closesthit'te,
örnek dışbükeyse ve düzlem listesi varsa (Voronoi hücresi) çalışır; dışbükey
olmayan (gürültü ile bükülmüş) örnekte bugünkü izlenen `scatterGlass`'a düşer.
Düzlem tablosu `rint` için değil, yalnız walk için.

**Hâlâ geçerli tuzaklar:** hacim düzeltmesi (§7.1, örnek mesh'in hacmi
parçacığın temsil ettiği hacme eşitlenir), `offset_ray` (§7.4), float
hassasiyeti (§7.2 — mesh'te de aynı, instance transform'u dünyada), çift
koyulaşma (§7.4).

**Ölçüm (K0 yeniden tanımı):** aynı parçacık seti, aynı kare/kamera/spp:
(a) küre bulutu, (b) T0 mesh instance + küre. Kaydedilecek: ms/sample (doğal
Vulkan örnek sayacı), TLAS instance sayısı, TLAS build ms, bellek. Beklenti:
(b)'nin maliyeti instance sayısıyla ölçeklenir, üçgen sayısıyla değil — bunu
örnek mesh'i 50 ↔ 200 üçgen arasında değiştirerek **ayrıca** doğrula.

`rint` çokyüzlüsü (§7.1–7.3) yalnızca ölçüm "instance sayısı tavan" derse geri
dönülecek yedek olarak duruyor.

## 7. K2 + K3 ayrıntılı tasarım (Vulkan RT) — §7.1–7.3 rev. 4 ile YEDEĞE alındı, §7.4–7.6 geçerli

### 7.0 Diğer ajanın RT perf durumu (2026-10-22, `RT_GAS_OPTICAL_PERFORMANCE.md`)

Bu tasarımı doğrudan etkileyen dört gerçek:

1. **Sphere hit group artık any-hit taşıyor.** Önceden any-hit slotu boştu;
   nearest-solid probe düzeltmesi için `shadow_anyhit` bağlandı ve sphere
   hit-kind'ı yeni `sphere_shadow_probe.glsl`'ye gidiyor (kaynakta, commit'lenmemiş;
   kullanıcı görünürlüğü doğruladı). Orada **normal küre merkezinden** hesaplanıyor
   (SSS sentinel'i için oct normal). K2 şekli değiştirince bu ikinci tüketici de
   değişmek zorunda — yoksa probe küre normali, closesthit çokyüzlü normali okur.
2. **Hacim solid probe'u küreleri de arar** (maske 0xF5, sphere biti 0x04 dahil);
   ölçülen sahnede ~1.04 probe/yol. Yani `rint`'e eklenen her ALU kamera
   ışınlarıyla değil, **kamera + sekme + gölge + hacim probe** ışınlarıyla çarpılır.
3. **Küre kesişim sayısı ve sphere BLAS GPU süresi ölçülmüyor.** Diğer ajan bunu
   açıkça "sunamayız" diye yazmış; önceliği gaz saçılması ve ışık gölgeleri. K0
   ölçümü hâlâ açık — bu tasarım onu kendi A/B anahtarıyla çözer (§7.4).
4. Geçerli ölçüm yöntemi: `scripts/test/rt_probe_gas_cost_ipc.py` desenindeki
   doğal Vulkan örnek sayacı (8→40) üzerinden wall ms/sample;
   `viewport.render_frames` **CPU renderer'ı** çağırır, RT ölçümü için geçersiz.

Çakışma noktası: `closesthit.rchit` ve `shadow_anyhit.rahit` + `sphere_shadow_probe.glsl`
diğer ajanın açık değişikliklerinde. K2/K3 bu dosyalara **o parti commit'lendikten
sonra** girer; ortak dosyada taze oku, `git add -A` yok.

### 7.1 K2 — veri

**Örnek tane tablosu** (yeni SSBO, salt okunur, bir kez yüklenir; binding numarası
`VulkanBackend` descriptor layout'undan seçilecek):

```glsl
struct GrainExemplarGPU {
    vec4  planes[14];   // xyz = unit normal (local), w = distance from centre (unit ball)
    uint  planeCount;   // 0..14; 0 = pure sphere
    float volumeFraction; // V(exemplar) / V(unit ball), host-side radius correction
    uint  _p0, _p1;
};
```

- Örnekler **birim küreye sığar** (her düzlem w ≤ 1, kırpan küre = birim küre).
  Bu sayede AABB bugünkü gibi küre yarıçapından gelir ve **refit/ABI değişmez**.
- Tablo host'ta deterministik üretilir (bileşen şekil parametreleri + tohum →
  jitterlı küresel Fibonacci yönleri + yuvarlaklıktan düzlem uzaklıkları). Bu
  ışın izleme değil, veri üretimi; birkaç KB. Bileşen başına 12–32 örnek,
  toplam ≤ 1024.

**Tane kaydı** (`FoamSphereGPU`, 32 bayt, stride aynı):

| Alan | Bitler | İçerik |
|---|---|---|
| `_p1` | 0–9 | örnek indeksi (0x3FF = **örnek yok → bugünkü küre yolu**, bit-bit aynı) |
| `_p1` | 10–13 | bileşen id (K1b) |
| `_p1` | 14–31 | ayrılmış |
| `_p2` | 32 | yönelim: `packSnorm4x8(quaternion)` |

- Köpük/sprey/kabarcık ve şekil parametresi olmayan her madde 0x3FF yazar →
  mevcut davranış hiç değişmez. Bu, K2'nin geri dönüşsüz bir sürüm değil, madde
  bazında açılan bir yol olmasını sağlar.
- Yönelim: DEM tane dönüşü taşıyorsa ondan (tane yuvarlanırken şekil de döner);
  yoksa `objectRandom` kimliğinden sabit hash. 8 bit/bileşen quaternion rastgele
  yönelim için fazlasıyla yeterli; `rint` `normalize(unpackSnorm4x8)` yapar.

**Hacim düzeltmesi (sinsi):** bir çokyüzlü kendini çevreleyen küreden küçüktür
(tipik `volumeFraction` 0.55–0.8). Yarıçap düzeltilmezse tane yatağı aniden
**seyrek ve içi görünür** olur, kütle aynıyken görsel hacim %20–45 düşer — kimse
bunu bug diye raporlamaz, "kum biraz ince" der. Host görsel yarıçapı
`r / cbrt(volumeFraction)` ile büyütür (AABB de onunla). Bu düzeltme
`fluidRenderProxyChildRadiusScale` çıkışına uygulanır, ayrı bir yerde değil.

### 7.2 K2 — `sphere_intersection.rint`

Yerel uzaya çevirme, ölçekle birlikte (t değişmez):

```
q   = unpack(_p2)                     // grain orientation
o'  = rotateInv(q, O - C) / r         // unit-ball space
d'  = rotateInv(q, D) / r
```

Algoritma (küre ∩ yarı-uzaylar, dışbükey slab):

```
1. sphere test on (o', d')             -> miss: return   // SAME cost as today on misses
   tIn = t0, tOut = t1, inFace = SPHERE, outFace = SPHERE
2. if exemplar == NONE: report as today (face = SPHERE)
3. for i < planeCount:
       den  = dot(n_i, d');  dist = w_i - dot(n_i, o')
       if |den| < eps: if dist < 0 -> return (parallel and outside)
       else t = dist / den
            den < 0 (entering): if t > tIn  { tIn  = t; inFace  = i }
            den > 0 (leaving) : if t < tOut { tOut = t; outFace = i }
       if tIn > tOut -> return          // early out: ray misses the polytope
4. t = tIn in [tmin,tmax] ? tIn : (tOut in [tmin,tmax] ? tOut : miss)
   face = (t == tIn) ? inFace : outFace
5. world normal: face == SPHERE ? normalize(o' + t d') : n_face ; rotate(q, .)
   (unit-ball space -> world: rotation only, scale is uniform)
6. hit attribute vec2: x = uintBitsToFloat(face | backFace << 8)
                       y = uintBitsToFloat(packSnorm2x16(oct(worldNormal)))
   reportIntersectionEXT(t, FOAM_SPHERE_HIT_KIND)
```

- Hit attribute düzeni (vec2) korunur; closesthit küre dalında `baryCoord`'u
  zaten okumuyor (`bary = (1,0,0)`). Normali `rint`'te bir kez hesaplamak,
  closesthit'in **ve** `sphere_shadow_probe`'un aynı normali okumasını sağlar —
  normal için **tek otorite** `rint` olur, iki merkez-tabanlı kopya söker.
- Maliyet: ıskalayan ışın için bugünle aynı (adım 1). İsabet eden ışın için
  K ≤ 14 nokta çarpımı + bir dönüş. Dallanma yalnızca erken çıkış.
- **Her ışın tipi aynı şekli görür.** "Gölge/probe için küre yeter" kısayolu
  YOK: tane ölçeğinde temas gölgesi ve AO yanlış olur, görünen geometri ile
  gölgeyi atan geometri ayrışır — mikro ölçekte "panel yalan söylüyor".

**LOD (T0/T1 sınırı) `rint` içinde, kamera mesafesiyle:**
`r / (|C − camPos| · pixelAngle) < T0_px` ise adım 3 atlanır (küre). Işın
mesafesi değil **kamera mesafesi** kullanılır: böylece bir tane için kamera,
sekme, gölge ve probe ışınları **aynı** şekli görür; sekme ışınının yakın
başlangıcı şekli değiştiremez. `T0_px` render ayarı (IPC). Eşik ~1–2 px'te
geçiş görünmez.

**Hassasiyet tuzağı (sinsi):** `O − C` float'ta hesaplanıyor. Dünya
orijininden 100 m'de float adımı ~7.6 µm; 0.25 mm'lik tane çapı ~33 adım →
yüzey kenarları %3 titrer; 1 km'de %30 → şekil çöker, ama "gürültülü kum" gibi
görünür. T0 ya orijine yakın sahnelerle sınırlanır ya da tane merkezleri
kameraya göre (camera-relative) yazılır. K2 kabul testinde **orijinden uzak
bir sahne** maddesi bu yüzden var.

### 7.3 K2 — tüketiciler

- `closesthit.rchit` küre dalı: `geomNormalRaw = decodeOct(floatBitsToUint(baryCoord.y))`;
  `face == SPHERE && exemplar == NONE` iken bugünkü merkez formülü (bit-bit aynı
  kalmalı — kabul maddesi).
- `sphere_shadow_probe.glsl`: SSS sentinel normali aynı attribute'tan.
- Object Info Location = tane merkezi (değişmez), Random = `objectRandom`
  (değişmez). K1b bileşen çıkışı `_p1` 10–13'ten.
- CPU referans / RayFusion: K2'yi görmez (§4.4); RayFusion küre çizer. Bu bilinçli
  ve notta yazılı bir fark, sessiz değil.

### 7.4 K3 — tane içi analitik yol (`closesthit.rchit`)

Nereye: transmission dalında, `scatterGlass(...)` çağrısının hemen öncesinde:

```
if (foamSphereHit && transmissionChosen && surfaceFrontFace) {
    grainDielectricWalk(...);   // writes scatterOrigin/Dir, attenuation, bounceType
    return;                     // like scatterGlass: no NEE on this event
}
```

Arka yüz isabeti (ışın zaten tanenin içinden başlamış — ör. tane içinde başlayan
bir ışın) bugünkü `scatterGlass` yoluna düşer; walk yalnızca **dışarıdan giriş**
olayını tekelinde tutar.

`grainDielectricWalk` (ayrı include, `grain_walk.glsl`; `rint` ile aynı yerel
uzay ve aynı düzlem tablosu):

```
nIn   = material ior (component)
nOut  = matx.medium_outside_ior            // 1.0 dry; set by MatterWetPalette (below)
entry: Fresnel/TIR with macro normal, GGX microfacet if rough   (= scatterGlass rules)
       reflect -> scatterOrigin = offset_ray(P, +N), dir = reflect, return
       refract -> d inside, p = P (unit-ball space)
for k < GRAIN_WALK_MAX (8):
    exit (tOut, face) of convex from p along d     // same slab loop, origin inside
    segment = tOut * r                              // world metres
    throughput *= exp(-sigma_a * segment)           // Beer-Lambert, component absorption
    p += tOut * d; N = exit normal
    Fresnel(nIn -> nOut) with microfacet sample
    TIR or reflect -> d = reflect(d, N); continue
    refract -> world exit point/dir; scatterOrigin = offset_ray(Pw, +Nw); return
exhausted -> attenuation = 0; counter grain_walk_truncated++
```

- Tane başına **tek** `traceRayEXT` (çıkıştan sonrası). Bugün kırıcı bir tane
  giriş + her iç sekme + çıkış için ayrı ayrı izleniyor (rint içeriden t1'i
  raporluyor). Maliyet ışından ALU'ya taşınır.
- Sekme muhasebesi: olay `BOUNCE_TRANSMISSION` olarak **bir** sekme sayar.
- **Öteleme:** `offset_ray` (ULP tabanlı) kullanılmalı. `RAY_OFFSET = 1e-3` (1 mm)
  **0.25 mm'lik tanenin 4 katı** — kullanılırsa çıkan ışın komşu taneleri atlar,
  yatak olduğundan saydam görünür. Kabarcık yolunun kullandığı sabit ofset
  burada kopyalanmamalı.
- K3 **K2'ye bağımlı değil:** `planeCount = 0` iken walk saf küre üzerinde
  çalışır (çıkış = küre t1). Yani K3 bugünkü transmissive küre splat'ine de
  uygulanabilir; sıra K3 → K2 de mümkündür ve ölçümü ayırır.

**Islaklık — tek otorite:** `MatterWetPalette` bant varyantını üretirken
kırıcı bileşenlerde `medium_outside_ior = mix(1.0, n_su, bant/7)` yazar;
walk bunu okur. Islak koyulaşma fizikten çıkar (kuvars/su ≈ 1.165).
★ **Çift koyulaşma tuzağı:** aynı varyantta `wet_color_scale` (0.55) de
uygulanırsa kırıcı tane iki kez koyulaşır ve ayarla "düzeltilir" — yanlış
kalibrasyon turu. Kırıcı bileşenlerde palet renk ölçeğini **1** tutar, yalnız
IOR yazar; opak bileşenler bugünkü ölçek yolunda kalır. `medium_outside_ior`
yeni bir materyal alanıdır: `matx` genişletilmiş materyal bloğunda boş alan
aranır, yoksa eklenir (CPU/OptiX struct parite'si korunarak — OptiX dondurulmuş,
alan yalnız derlensin diye eklenir, okunmaz).

**Gölge ışınları:** `sphereShadowProbe` taneyi opak kabul edip sonlandırıyor;
korunur. Tek tane içinden geçen doğrudan ışık (kostik) kaybolur — kum yatağında
zaten birkaç tane geçmeden sönen bir katkı, ve `scatterGlass` da NEE yapmıyor.
Bilinçli fark olarak yazıldı.

### 7.5 IPC / ölçüm

| Yetenek | Katman |
|---|---|
| `render.grain_shape_mode` = auto / force_sphere | A/B anahtarı; K0 ölçümü bununla yapılır |
| `render.grain_t0_pixels` | LOD eşiği |
| `render.grain_counters` → polytope isabet, walk olayı, ortalama iç sekme, `grain_walk_truncated` | sayaçlar; mevcut `render.volume_counters` deseniyle aynı aç/kapa |
| bileşen şekil parametreleri (yuvarlaklık, küresellik, form, örnek sayısı) | `matter.*`, maddenin alanı |

Dört dokunuş (RtApi, RtIpc, RtPython, RtIpcSecurity) + descriptor üretici +
overlay satırı. Sayaçlar atomik olduğundan yalnız açıkken çalışır.

### 7.6 Kabul (build sonrası, sıralı)

1. **Bit-bit eşitlik:** şekil parametresi olmayan bir madde ve köpük, K2 öncesi
   ve sonrası aynı kare/kamera/spp'de **aynı** görüntü. Değilse 0x3FF yolu
   bugünkü küre yolunu bozmuş demektir — diğer her sonucu maskeler, ilk bakılır.
2. `force_sphere` ↔ `auto` A/B: ms/sample (doğal Vulkan örnek sayacıyla) ve
   polytope isabet sayacı. T0 dışında kalan sahnede fark ≈ 0 olmalı.
3. Yakın plan: fasetler görünür, gölge sınırı ile görünen kenar aynı yerde,
   SSS sentinel normali yüzeyle tutarlı (probe ↔ closesthit aynı attribute).
4. Hacim düzeltmesi: aynı parçacık setinde kum yatağının ekrandaki doluluk
   oranı küre ile ±%5 içinde. ★ Düşükse §7.1 düzeltmesi uygulanmamıştır ve
   görüntü "makul ama ince" görünür.
5. K3: kuvars bileşenli yatakta ortalama yol derinliği ve ms/sample düşüşü
   sayıyla; `grain_walk_truncated / walk olayı` < %1.
6. Islak/kuru: doygunluk 0 → 1 arasında kırıcı bileşenin luminansı tekdüze
   düşmeli; renk ölçeği 1 olduğu sayaçla değil materyal okumasıyla doğrulanır.
7. Orijinden uzak (≥ 500 m) sahne: titreyen kenar var mı — varsa §7.2
   hassasiyet tuzağı.

## 8. K0 — uygulama (2026-10-22, kaynakta, derlenmedi)

**Ne yapıldı:**
- Domain alanları (render-only, kayıt/yükleme dahil): `fluid_grain_mesh_near_enabled`,
  `_sources` (sahne düğümleri), `_distance` (m, aktif kameradan), `_budget`
  (mesh instance). Üst sınır yok (CLAUDE/hafıza kuralı: tavan panelin sürükleme
  aralığında); API yalnız mesafe > 0 ve kaynakların canlı olmasını ister.
- `ParticleRenderBridge::syncDomainFluidParticleInstances`: aynı parçacık
  yürüyüşünde granüler + splat + kameraya `distance`'tan yakın + bütçe içi
  parçacık **mesh havuzuna**, kalanı küre bulutuna. Her parçacık bir kez çizilir.
  Mesh havuzu ikinci bir transient `InstanceGroup` (`[FluidDomGrainNear] …`),
  kaynakları (malzeme yuvası × örnek): malzeme ekseni (madde bağlaması, ıslak
  bant) ana havuzla aynı, şekil ekseni örnekten.
- Örnek seçimi ve yönelim `particle_id` hash'inden (yeniden tohumlamada şekil
  değişmez). Hacim düzeltmesi: örnek mesh'in diverjans hacmi → küre hacmine
  eşitleyen ölçek (0.5–2 sıkıştırılmış; açık mesh'te uygulanmaz ve sayılır).
- Kamera hareketi duraklatılmış simülasyonu da yeniden böler (`near_camera_key`,
  adım = mesafenin %5'i).
- Ön koşullar, karşılanmazsa status'ta yazılı: geometri `icosphere`, parçacık
  başına tek görsel küre (fiziksel taşıyıcı / grain domain), kaynak listesi,
  aktif kamera.
- Ana havuz gizlenince mesh havuzu silinmez, gizlenir (geri sarmada yapısal
  yeniden kurma = bilinen TDR tetikleyicisi); near kapatılınca/başarısız olunca silinir.
- `gatherSceneMeshSource` (Scene Object modu) ile örnek toplayıcı tek
  implementasyona indirildi (`gatherNormalizedSceneFaces`).
- **`render.instance_stats`** (RtApi/IPC/Python + descriptor): TLAS instance
  sayısı, son build/update yolu, CPU hazırlık ms, küre bulutunun kayıt sayısı.
  Bu sayılar şimdiye kadar yalnız Volume Performance panelindeydi — §1
  ihlaliydi ve K0'ın tam ihtiyacı olan sayıydı.

**Ölçüm:** `scripts/test/rt_k0_grain_mesh_near_ipc.py` (release kopyası da var).
Vakalar: `spheres` → `shards_D` → `shards_4D` → `spheremesh_D` → `spheres_repeat`.
Örnek kütüphanesi sahnenin altına (y = −500) kurulur: Voronoi fracture (convex,
12 hücre) ve aynı sayıda küre primitifi (aynı instance, çok daha fazla üçgen);
iş bitince silinir.

**Bilinen eksikler (K0 kapsamı dışında, bilinçli):**
- Mesh tanelerde Object Info Random instance orijininden → düşen tane rengi
  değişebilir (küre yolunda sabit kimlik var). K1b'de düzeltilecek.
- Yönelim simülasyondan değil hash'ten (DEM dönüşü bağlanmadı).
- Sanal çocuklu (virtual_grains > 1) domain'ler desteklenmiyor; status söylüyor.
- Yakın/uzak sınırında şekil geçişi ani (LOD yumuşatma yok).

**Sonuç — ara ölçüm 1 (2026-10-22, kullanıcı build'i, canlı):**

Sahne: `K0_Sand` Matter domain, grain yarıçapı 2.5 cm, 4614 tane yere
oturmuş (timeline kare 90, duraklatılmış), kamera kumdan ~2 m, Rendered,
RT pipeline hazır. Süre: doğal Vulkan örnek ilerlemesi 8→40, UI dahil wall ms.
Örnek kütüphanesi: 8 küre primitifi = **1280 üçgen/tane** (shard'lar bu
build'de toplanamadı, aşağıya bak).

| Vaka | Mesh tane | Küre tane | TLAS instance | ms/örnek | küreye göre |
|---|---:|---:|---:|---:|---:|
| spheres | 0 | 4614 | 11 | 26.32 | 1.00× |
| küre-mesh, 1.5 m | 783 | 3831 | 2059 | 36.51 | 1.39× |
| küre-mesh, 6 m (hepsi) | 4614 | 0 | 8203 | 59.35 | 2.25× |
| spheres (tekrar) | 0 | 4614 | 11 | 26.38 | 1.00× (sapma %0.2) |

Okuma (henüz karar değil):
- Ölçüm penceresinde hiç accel yeniden kurma yok (`perf.list accel.*` boş):
  fark saf iz sürme maliyeti.
- 1280 üçgenli taneler bu sayıda **prosedürel küreden pahalı**: tane başına
  +7.2 ms / 1000 tane (hepsi mesh). Bu, §6b'deki "üçgen ucuz" varsayımını
  1280 üçgende çürütüyor; asıl soru (30–60 üçgenli shard) bir sonraki build'de.
- **Karıştırıcı 1 — havuz boşluğu:** 4614 canlı mesh için TLAS 8192 slot
  taşıyor (8 kaynak × 4'lük havuz büyümesi); ~3578 sıfır ölçekli instance
  orijinde, yani kum yığınının tam ortasında duruyor. Etkisi ayrı ölçülmeli.
- **Karıştırıcı 2 — gölgelendirme yolu:** mesh isabeti üçgen closesthit'inin
  tamamını (vertex/normal/UV okuma) koşuyor, küre isabeti kısa yolu.
- Hacim ölçümü çalışıyor: küre primitifi için `mean_volume_scale` 1.004.

**Sonuç — tam K0 (2026-10-22, TLAS sayısı + fracture göçü build'i, canlı):**

Aynı fixture (4591 tane, kare 90, kamera ~2 m), `--library boxes`: 12 flat
kutu, tane başına **12 üçgen**, en-boy ve eğim şekle işlenmiş. Ham sonuç:
`docs/dev/k0_grain_mesh_near_20261022.json`.

| Vaka | Mesh | Küre | Üçgen/tane | TLAS slot | ms/örnek | küreye göre |
|---|---:|---:|---:|---:|---:|---:|
| spheres | 0 | 4591 | — | 23 | 25.75 | 1.00× |
| kutu, 1.5 m | 775 | 3816 | 12 | 3095 | 28.07 | 1.09× |
| kutu, 6 m (hepsi) | 4591 | 0 | 12 | 12311 | 31.95 | 1.24× |
| küre-mesh, 1.5 m | 775 | 3816 | 1280 | 2071 | 36.86 | 1.43× |
| spheres (tekrar) | 0 | 4591 | — | 23 | 25.81 | 1.00× (sapma %0.2) |

Görsel kontrol (ekran görüntüsü alındı, silindi): 1.5 m içindeki şerit köşeli
kutu, ötesi küre; ayrım temiz, mesh taneler gerçekten çiziliyor.
`mean_volume_scale` 1.31 (kutu, küre hacmine eşitlenmiş).

**Okuma:**
1. **Üçgen sayısı bedava DEĞİL.** Aynı 775 instance'ta 12 üçgen +2.3 ms,
   1280 üçgen +11.1 ms: ~107× üçgen, ~4.8× ek maliyet. Piper'ın "50/500/5000
   poligon fark etmedi" gözlemi (RenderMan, CPU) bizim Vulkan RT'ye bu ölçekte
   taşınmıyor. §6b'nin "üçgen ucuz" gerekçesi düzeltildi: **düşük poligon
   şart** (≲ 50 üçgen/tane hedefi).
2. **Prosedürel küre tane başına hâlâ en ucuz.** Hepsi 12 üçgenli mesh olunca
   +6.2 ms (≈ +1.35 ms / 1000 mesh tane, +%24).
3. **Hibrit uygulanabilir:** tanelerin %17'si yakın mesh iken toplam maliyet
   +%9. Bu, §6b kararını (yakın mesh, uzak küre) destekliyor, ama yalnız düşük
   poligonla.
4. **★ Karıştırıcı hâlâ duruyor:** TLAS slotlarının çoğu boş. 775 canlı mesh
   için 3072 slot (24 kaynak = 12 örnek × 2 malzeme yuvası, kaynak başına en az
   256), 4591 canlı için 12288 slot (%63 sıfır ölçekli, orijinde). Kutu
   sonuçlarının bir kısmı bu boşluğun maliyeti olabilir; sayılar bu yüzden mesh
   yolunun **üst sınırı**.

**Sıradaki:** (a) havuz boşluğunu azalt — YAZILDI, derlenmedi: yakın havuz
kendi kuralını kullanıyor (`nearPoolCapacityFor`: canlı + %25, 64'lük adım,
boş kaynağa 0, canlı yarıya inince küçülür); 4591 tane için beklenen ~6.1 bin
slot (önce 12311); aynı ölçüm tekrarlanacak; (b) `--library fracture` (göç build'iyle mümkün) ile gerçek
Voronoi shard'larını ölç; (c) K1b: mesh tanelerde Object Info Random için sabit
kimlik.

**Bu ölçümün ortaya çıkardığı iki alet hatası (düzeltildi, build bekliyor):**
1. **Fracture shard'ları "missing source".** Shard'lar sahneye `Triangle`
   facade olarak ekleniyor; toplayıcı yalnız flat `TriangleMesh` okuyor.
   Facade okuma kolu eklenip **geri alındı** (kullanıcı: facade bellek/işlem
   maliyeti yüzünden terk edildi). Doğru düzeltme fracture'ın flat SoA üretmesi;
   bu ayrı bir göç: shard üretimi (`scene_ui_menu.hpp` ~1681), shard silme
   (`fractureEraseNodes_`, yalnız facade siler), rigid body poz yazımı
   (`scene_data.h` 1331/1361/4946, `getRigidBodySourcePivot` yalnız facade),
   termal kırılma köprüsü, proje kaydet/yükle (`ProjectManager` 3955/4172).
   K0 o zamana kadar flat kutu örnekleriyle (`--library boxes`, 12 üçgen)
   ölçülür.
2. **TLAS instance sayısı bayat (0).** CPU yedek yoluyla kurulan TLAS'ta
   `instanceCount` yazılmıyordu; panel de `render.instance_stats` de 0
   gösteriyordu. Artık her kurulumda yazılıyor. (Bu ölçümdeki TLAS sayıları
   refit yolundan geldiği için doğru.)

Ayrıca: `fluid.step` ile ilerletilen durumu kare döngüsü kare 0'a geri aldı
(bilinen ders); fixture `timeline.set_frame` ile kuruldu.

### §8b Gerçek geometri + mineral karışımı (2026-10-22, 13:30 build, canlı)

Derleme beklemeden mevcut exe ile koşuldu (havuz boşluğu düzeltmesi bu
build'de YOK — TLAS slotları hâlâ eski kural). Fixture aynı (4591 tane,
kare 127, kamera [1.3, 0.75, 1.5]).

**Geometri:** `--library fracture`: 0.5'lik küre 12 Voronoi hücresine kırıldı
(`exact_surface=false`), shard'lar flat `TriangleMesh` (göç build'i).
12 örnek toplam 2852 üçgen (**~238 üçgen/tane** — küre yüzeyini taşıyan dış
yüzler yüzünden 30–60 hedefinin üstünde), `mean_volume_scale` 1.43.

**Materyal:** `scripts/test/rt_grain_mineral_material.py --material <ad>`
(yeni). Tane başına tek `Object Info → Random`, eşiklerle bileşen:
kuvars < 0.70 (bal↔süt tonu, transmission 0.55, IOR 1.55), feldispat
0.70–0.88 (krem/pembe, 0.12), taş kırıntısı 0.88–0.97 (gri-kahve, opak,
pürüzlü), ağır mineral ≥ 0.97 (siyah). Renkler kanal kanal Math →
Combine Color ile kurulur (aşağıdaki derleyici hatası yüzünden). Aynı grafik
iki yolda da çalışıyor: küre bulutu Random'u küre kaydından, mesh taneler
instance orijininden alıyor. Görsel kontrol (alındı, silindi): yakında köşeli
shard, ötede küre; dört bileşen ayırt ediliyor, kuvars yarı saydam.

Ham sonuç: `k0_grain_mesh_near_fracture_20261022.json` (transmission'lı),
`k0_grain_mesh_near_fracture_opaque_20261022.json` (`--transmission-scale 0`).

| Vaka | Mesh | Üçgen/tane | TLAS | opak ms | × | kuvars trans. ms | × |
|---|---:|---:|---:|---:|---:|---:|---:|
| spheres | 0 | — | 33 | 37.67 | 1.00 | 102.22 | 1.00 |
| shard, 1.5 m | 780 | ~238 | 3105 | 38.51 | 1.02 | 114.27 | 1.12 |
| shard, 6 m (hepsi) | 4591 | ~238 | 12321 | 37.52 | 1.00 | 135.94 | 1.33 |
| küre-mesh, 1.5 m | 780 | 1280 | 2081 | 54.96 | 1.46 | 122.93 | 1.20 |
| spheres (tekrar) | 0 | — | 33 | 37.78 | 1.00 | 112.13 | 1.10 |

**Okuma:**
1. **Opak kumda gerçek shard geometrisi bedava** — hepsi mesh olsa bile
   (12 bin TLAS slotu, %63'ü boş). Kutu ölçümündeki +%24 burada yok; materyal
   programı tane başına isabeti pahalılaştırınca geometri payı eriyor.
   Üçgen sayısı yine **sınırsız değil**: 1280 üçgenli küre-mesh +%46.
2. **★ Maliyetin sahibi kuvars transmission'ı:** aynı sahne opak 37.7 →
   yarı saydam 102.2 ms (**2.7×**). Her kuvars tanesi kırılan ışını yığının
   içine yolluyor (transmission_bounces 8). Mesh/küre farkı bunun yanında
   küçük; transmission'lı sütunda %10'luk tekrar sapması var, o sütunun
   geometri oranları güvenilir değil.
3. Materyal programı (47 node, piksel başı) sade preset'e göre ~+%46
   (25.75 → 37.67; kare 90 → 127 farkı da var, kaba).
4. Sonuç: geometri kararı (§6b hibrit) doğru, ama **sıradaki optimizasyon
   hedefi şekil değil ışık taşınımı** — tam olarak §7.4 K3 (tane içi analitik
   yol, sekmeleri yığına dağıtmadan). Shard'ların düz kesik yüzleri K3'ün
   düzlem listesini zaten veriyor.

**Bu turda bulunan hatalar:**
1. **★ Materyal programı derleyicisi bağlantısız pin değerini atıyordu
   (DÜZELTİLDİ, derleme bekliyor).** `MaterialNodesV2.h` derleyicisi, değeri
   node özelliğinde değil **pinde** yaşayan girişleri (Mix Color'ın iki rengi,
   Hue/Sat, Combine/Separate Color, Invert, Gamma, Bright/Contrast, AO rengi…
   25 giriş) sabit literal olarak derliyordu. Panelde ya da `nodes.set_param`
   ile girilen renk katlanmış materyalde doğru görünüyor, ama CPU piksel başı
   gölgelendirme ve Vulkan RT 0.5 gri çiziyordu. Klasik "panel yalan söylüyor":
   materyal bilgisi bej, render gri. Düzeltme: `inPin` yardımcısı bağlantısız
   pinin `defaultValue`'sunu derler (`compute()`'un `getInputValue` ile okuduğu
   değer). Özellik destekli girişler (`fr->ior`, `mt->defaultA`, Noise renkleri)
   `inOr`'da kaldı.
2. **`nodes.remove_graph` derlenmiş programı materyalde bırakıyor (DÜZELTİLDİ; build/canlı doğrulama bekliyor).**
   Grafiği silip `base_color`'a kırmızı yazınca render yine gri kaldı; aynı
   materyale yeni grafik kurmak düzeltti. `removeNodeGraph` artık materyalin
   `proceduralProgram` kaydını temizleyip backend programlarını eşitliyor.
3. **Color Ramp durak listesi scriptlenemiyor (DÜZELTİLDİ; build/canlı doğrulama bekliyor).** `NodeParamValue::Rows`, JSON/Python satır dizileri ve
   `setNodeRowsProperty` üzerinden duraklar ortak core yoluna ulaşıyor.
4. **K0 scripti geri yüklemede `sources`'u atlıyordu (DÜZELTİLDİ).** Kullanıcının
   paneldeki örnek listesi (`Default_Cube`, `K0_Ground`) ölçüm kütüphanesiyle
   değişti; ayrıca `--keep-library` ile saklanan kütüphaneyi sonraki koşunun
   temizliği siliyordu (aynı tohum adı). Şimdi kaynaklar da geri yükleniyor,
   tohum adı koşu başına benzersiz.
5. **Doğrulanmadı:** fixture'daki `Sand` → `K0_Dry_Sand` bağlaması tanelere
   ulaşıyor mu? Parçacıklar `T: K0_Sand Sand grains` etiketini taşıyor;
   ölçüm `fluid.set_splat_material` ile domain materyali verilerek yapıldı.

## Kaynaklar

- Pixar RenderMan, "Piper" — renderman.pixar.com/stories/piper
- fxguide, "The tech of Pixar part 1: Piper" (aynı metin)
- 3dvf, Annecy 2017 RenderMan raporu (5000 poligon / render süresi)
- SIGGRAPH 2025 talk, "Fine Tuning Every Grain on the Beach" (Elio) — yalnız özet okundu
- Meng, Papas, Habel, Dachsbacher, Marschner, Gross, Jarosz, *Multi-Scale Modeling and Rendering of Granular Materials*, SIGGRAPH 2015
- Lekner & Dorf, "Why some things are darker when wet", Applied Optics 1988


### 2026-10-22 devam: prosedürel kütüphane, kaynak materyalleri, bütün çözücüler

Kaynakta tamamlandı; derleme ve canlı render kabulü kullanıcıda.

- `fluid.set_splat_geometry near_mesh_shape="scene"|"rock"|"voronoi"`.
  Rock: sabit seed 1337'den türeyen 16 düşük çözünürlüklü Perlin/cube-sphere
  örneği. Voronoi: mevcut `generateFracture` convex çekirdeğinden en çok 16
  hücre. İkisi de sahneye obje/collider eklemeden doğrudan render kütüphanesi
  üretir. UI: **Near Grain Meshes → Grain Shape**.
- Scene modu seçilen bütün flat `TriangleMesh`/DNA SoA yüzlerini, UV'leri ve
  yüz materyal kimliklerini okur. `near_mesh_source_materials=true`
  (**Use Exemplar Face Materials**) her örneğin kendi yüz materyalini korur;
  yakın mesh'lerde domain/substance/ıslak bant override'larının yerine geçer.
  Uzak küreler kendi materyallerinde kalır. Varsayılan false eski projelerin
  görünümünü korur. Üretilen rock/Voronoi domain/substance materyalini kullanır.
- Seçili sahne kaynaklarının SoA verisi, UV, materyal ve transform içeriği
  fingerprint'e girer. Kaynakta yüz materyali/geometri değişirse duraklatılmış
  sahnede bile kütüphane yenilenir; simülasyon resetlenmez. Hash maliyeti seçili
  kaynak kütüphanesiyle ölçeklenir, parçacık sayısıyla değil.
- Mesh görünümü artık **DEM, MPM ve fluid** için bütün Splat-routed ana
  parçacıklara uygulanır. Açıkken bir ana carrier = bir mesh/küre; kullanıcı
  çoklu virtual-children değerini silmeden geçici olarak primary=1 seçilir.
  Kapatınca önceki çoklu küre ayarı geri gelir; whitewater ayrı kalır.
  DEM kendi fiziksel yarıçapını, diğer carrier'lar mevcut voxel/splat çapını
  kullanır; aynı domain'deki sıvılar DEM yarıçapına zorlanmaz. Örnek geometri
  hacmi küreye eşlenir (mevcut 0.5–2 scale sınırı dahil).
- Raster'ın bütün resident sıvı pozisyonlarını tekrar küre olarak çizen kısa
  yolu mesh görünümünde atlanır; yakın carrier hem mesh hem küre çizilmez.
- UI, Python ve IPC aynı `setFluidSplatGeometry` setter'ını kullanır.
  `fluid.get.grain_mesh_near` içindeki `shape`/`source_materials` ayarları
  hem ProjectManager hem SceneSerializer kayıt/açma yolunda saklanır.
  Geçersiz shape patch'in hiçbir alanını değiştirmez. Descriptor yeniden üretildi.
- IPC kaynak denetimi: dış `processRequest`, ana-thread enqueue yardımcıları
  ve batch alt çağrıları `std::exception` yakalıyor; dış sınır ayrıca
  `json::exception` yakalıyor. Eski build'deki kapanmanın yeni executable'da
  giderildiğini `rt_material_node_parity_ipc.py` ve yeni library probe doğrular.
- **Kullanıcı düzeltmesi:** önce bildirilen MPM/fluid emitter durması sahne
  kurulumundan kaynaklanıyordu. Çözücü arızası olarak takip edilmiyor. Yapılan
  genişletme çözücüyü değil parçacığın mesh görünümünü değiştiriyor.
- K0 scripti yeni iki seçeneği de geri yükler. Yeni
  `rt_grain_library_options_ipc.py` atomik hata reddini/IPC sağkalımını ve
  isteğe bağlı rock/Voronoi render split'ini denetler.
  `grain_mesh_library_test.cpp` deterministik kapalı geometri, hacim,
  indexed flat kaynak materyalleri, materyal override ve primary-count
  sözleşmesinin kaynak testidir; ajan derlemedi/çalıştırmadı.
- **Açık, sonraki fizik işi:** fiziksel tane boyut dağılımı. DEM temas, atalet,
  doğum aralığı, CFL/Verlet, sıvı-gaz/MPM coupling ve ıslak kapasite tek
  `radius_m` varsayıyor. Bu partide yalnız render jitter'ı eklenmedi;
  gerçek fiziksel boyut çeşitliliği tamamlandı iddiası yok.


### 2026-10-22: üretici Add Mesh'e taşındı

Güncel yön: `GRAIN_CRYSTAL_GENERATOR.md`. Mesh Grains'teki sabit prosedürel
shape dropdown'ı yerine ortak Grains / Crystals üreticisini açan kısayol var.
Rock/Voronoi yanında dallı kalınlıklı kar ve prizmatik mineral kristali eklendi.
Gerçek flat sahne objeleri, parametreli üretim, materyal paleti ve tek Undo/Redo;
Python/IPC `scene.generate_grains` aynı servisi çağırır. Eski shape ayarları
uyumluluk için ortak çekirdek üzerinden hâlâ okunur. Dar paneldeki scene object
listesi düzenlemesi kullanıcı isteğiyle sonraya bırakıldı. Build/canlı kabul bekliyor.
