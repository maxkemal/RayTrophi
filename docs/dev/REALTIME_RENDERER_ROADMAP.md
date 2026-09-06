# RayTrophi Realtime Renderer Roadmap

> **Durum:** TASLAK — ayri bir Realtime render yolu olarak **PARK EDILDI**
> (2026-08-31). Asagidaki urun sozlesmesi, frame graph ve kalite profilleri
> **gecerli analiz** olarak duruyor; yanlis olan **sirasiydi**. Faz 0.5a'da
> yazilan frame ring / telemetri / instancing isi park EDILMEDI — o raster
> yolunun kendisine aitti ve canli. **2026-08-31: GPU culling + LOD ayrimi
> yazildi (derlenmedi)** — bkz. asagidaki bolum. **2026-09-01: LOD gecis bandi,
> proxy materyali ve `Full` preset'i yazildi (derlenmedi)** — ★★★★ o bolum
> "kademeli LOD mu, sahne isigi mi" sorusunun olculmus cevabini tasiyor:
> pathtrace hizi yanlis hedef, ve "cok kaba proxy"nin baskin sebebi poligon
> sayisi degil proxy'nin materyal hattini hic gormemesiydi.
> **2026-09-01: Faz 1 dilim 1 — material preview'da GERCEK SAHNE ISIKLARI
> yazildi. Faz 1 dilim 2/3 — tum isik tipleri icin bounded shadow atlas ve
> gercek world/HDRI ambient yazildi (derlenmedi).**
> **2026-09-02: atmosfer LUT hatti etkilesimli viewport'ta hic kurulmuyordu —
> duzeltildi ve CANLI DOGRULANDI** (bkz. `NEXT_BUILD_CHECKS.md`); ayrica
> **Katlanabilir Realtime kalite panelinin ilk dilimi, preset-bagimli shadow
> tile/light/PCF butcesi ve Scene GGX enerji kapatmasi yazildi (derlenmedi).**
> FPS telemetrisi tasarimi henuz yazilmadi.
> **2026-09-04: realtime volume/gaz gecisi (`MaterialPreviewVolume.cpp`,
> `material_preview_volume.frag`) baska bir ajan tarafindan yazildi ama SADECE
> mesh'li sahnelerde gorunuyordu — yalniz gaz domain'i olan (baska mesh'i
> olmayan) bir sahnede hic cizilmiyordu. Kok neden `VulkanViewportBackend.cpp`
> `useMaterialPreview` bayragindaki `!m_rasterInstances.empty()` sartiydi:
> pipeline/descriptor/ABI/shader zinciri dogruydu, ama bu on-kosul volume'u
> mesh'e bagimli kiliyordu. DUZELTILDI (derlenmedi) — bkz. `NEXT_BUILD_CHECKS.md`.

## 2026-08-31 karari — neden ayri yol park edildi

Ayri `Realtime` viewport modu (enum, deferred G-buffer pass, iki pipeline, iki
shader cifti) **sokuldu**. Bu bir fikir degisikligi degil, olcum sonucu.

**Yazilan yol calismiyordu, ve calismadigini soyleyemiyordu:**

- `ensureRealtimeViewportResourcesImpl` tek bir `vkCreate*` donusunu kontrol
  etmiyor, sonunda **kosulsuz `return true`** yapiyordu. Kural 6'ya gore
  desteklenmeyen bir yol sessizce yanlis gorunmemeliydi; burada "kaynak hazir"
  bir olcum bile degildi.
- Deferred subpass `depthImage`'i **input attachment** olarak okuyordu, ama o
  image `INPUT_ATTACHMENT` usage biti olmadan olusturuluyor. `vkCreateFramebuffer`
  bunu reddeder; framebuffer NULL kalir; fonksiyon yine `true` doner.
- Solid / hair / particle / edit overlay pipeline'larinin hepsi
  `m_interactiveViewport.renderPass` (2 ek, 1 color) ile derlenmis; realtime
  pass 5 ek / 3 color. **Render pass uyumsuz** — o pipeline'lari realtime
  pass'in icine bind etmek gecersiz.
- `destroyInteractiveViewportResourcesImpl` realtime kaynaklarina hic
  dokunmuyordu; `ensureRealtime`'in erken cikis kapisi
  (`width == width && gbufferAlbedo.image`) resize'da eski framebuffer'i
  koruyordu -> ilk yeniden boyutlandirmada use-after-free.
- `VulkanBackendAdapter::supportsViewportMode(Realtime)` zaten **false**
  donuyordu. Backend bu modu hic sahiplenmemis; UI o kontrolu
  `g_viewport_backend != nullptr` ile atliyordu.

**Ve calissaydi bile bir sey eklemiyordu:** deferred descriptor set'inde isik
buffer'i yoktu; aydinlatma push constant'a **elle yazilmis uc sabit isikti** —
yani Faz 1'in kendi kabul olcutu olan "sabit uc preview isigi sizmaz"
kosulunu, yeni yol ayni sabit uc isikla yeniden uretmisti. Vertex layout'u da
material preview'dan birebir kopyaydi.

### Duzeltilen sira

Faz 1'in teslimati **gercek sahne isiklaridir**, deferred degil. Deferred *cok
isik* icin bir optimizasyondur; DCC viewport'unda ilk kazanc oradan gelmiyor.

Yogun sahne olcumu bunu daha da netlestirdi:

| | Maliyet neyle olcekleniyor |
|---|---|
| Vulkan RT | Sahne karmasikliginda logaritmik, **piksel** sayisinda dogrusal |
| Raster | **Gonderilen ucgen** sayisinda dogrusal |

Milyar ucgenli instance sahnelerinde raster, piksel basina yuzlerce ucgen
gonderir; proxy'siz RT'ye yaklasamaz ve bu asimptotik bir fark, uygulama
kalitesi meselesi degil. **Eevee de bunu proxy'siz yapmiyor**; Nanite yapiyor,
ama Nanite surekli, cluster seviyesinde bir proxy sistemidir. Yani soru "proxy
mi degil mi" degil: **proxy gorunur bir kalite ucurumu mu, yoksa fark
edilmeyen surekli bir LOD mu?**

Bugunku proxy yolunun olculen sorunlari:

- Butce tavani 48M ucgen; 1080p'de kalite olcegiyle ~20-35M'ye iniyor.
- Takas **instance basina ikili** (en yakin N tam, gerisi proxy) — tanimi
  geregi pop uretir.
- Secim **her kare CPU'da**: worldBBox merkezleri, mesafe kareleri,
  `nth_element` + `sort`. Kamera hareket edince `m_rasterFrustumRevision`
  degisir, onbellek duser, butun liste yeniden gezilir.
- ★★★ **Culling ile production instancing birbirini disliyor.**
  `VulkanBackend_Raster.cpp`'de `m_rasterUseGlobalInstBuffer` acikken
  `uploadVisibleRasterInstances` proxy/culling mantigina **varmadan** erken
  donuyor. Yani bugun ya CPU taramali proxy'li yol var, ya CPU'suz ama
  culling'siz yol. Ikisini birden veren bir yol **yok**.

### Yeni sira

1. ~~Realtime'i sok~~ ✔ 2026-08-31.
2. **Sunum kopyalarini azalt.** Olu `SDL_UpdateTexture` kaldirildi (kare basina
   8.3 MB). Kalan: `original_surface` memcpy'si ve Main'in `surface` kopyasi.
   ★ Not: gercek sifir-kopya sunum (eski 0.5b) **ucuz degil** — UI'in tamami
   `ImGui_ImplSDLRenderer2` uzerinde kosuyor, SDL_Renderer2'ye Vulkan image
   verilemez. O adim butun ImGui katmaninin `ImGui_ImplVulkan`'a tasinmasi
   demektir ve ayri bir projedir.
3. ~~**GPU-driven culling + indirect draw**~~ ✔ **yazildi 2026-08-31, derlenmedi.**
   Ayrintisi asagida.
4. **Surekli LOD.** Ikili instance takasi yerine mesafeye gore kademe +
   dithered gecis. GPU culling ile birlikte LOD karari artik bir MESAFE
   ESIGIdir (asagi bak), yani kademe eklemek icin dogru yer hazir.
5. **Gercek sahne isigi/golgesi — material preview yoluna bir SECENEK olarak.**
   `m_lightBuffer` zaten mevcut (`VulkanBackend.h`); `material_preview_frag.frag`
   (631 satir, GGX + Charlie sheen + clearcoat + SSS tint + env specular + terrain
   blend) yeni bir `lightingPreset = 3` (scene) dali alir. Ayri render pass
   kurulmaz; studio preset'leri bozulmaz.

## GPU culling + LOD ayrimi (2026-08-31, yazildi/derlenmedi)

### Duzeltilen ariza

Production instancing (global instance buffer) devreye girdiginde **frustum
culling ve scatter proxy tamamen devre disi kaliyordu.** Iki satir:

- `VulkanBackend_Raster.cpp` -> `uploadVisibleRasterInstances` global yolda
  proxy/butce mantigina **varmadan** erken donuyordu;
- `writeRasterInstanceTransformsToGlobal` `mesh.instanceCount = <tum instance
  sayisi>` yaziyordu.

Sonuc: her mesh'in butun instance'lari, kamera nereye bakarsa baksin, her kare
gonderiliyordu. Gizli nesneler bile bedava degildi -- `mask == 0` sifir matris
yaziyor, dejenere ucgenler yine vertex shader'dan geciyordu.

★★★ Bu arizanin ekranda **hicbir belirtisi yoktur**: goruntu tamamen dogrudur,
yalnizca pahalidir. `HEAD`'de bu yol yok, yani eski binary'de culling calisiyordu;
regresyon commit'lenmemis calismada duruyordu.

### Cozum

`shaders/raster_cull.comp` + `Viewport/RasterGpuCull.{h,cpp}`. Iki dispatch:

| pass | granulerlik | is |
|---|---|---|
| 0 | mesh basina bir dispatch | frustum testi + LOD ayrimi; hayatta kalanlarin matrisi **sikistirilmis** cikti bolgesine kopyalanir |
| 1 | tek dispatch, mesh basina bir is parcacigi | indirect draw komutlari yazilir, LOD mesafe esigi guncellenir |

★ **Raster vertex shader'larinda hicbir degisiklik yok.** Instance matrisi zaten
per-instance vertex attribute olarak geliyordu; yalnizca hangi buffer'in
baglandigi ve kac instance cizildigi degisti. Cizim `vkCmdDraw[Indexed]Indirect`.

Proxy mesh'in kendi instance'i yoktur; komutunu **sahibi** (scatter grubu) yazar.
Tam ve proxy **ayri cikti bolgelerine, ikisi de onden** doldurulur.

★★ Neden ayri bolge: `vkCmdDraw*Indirect`'te `firstInstance != 0` kullanmak
`drawIndirectFirstInstance` cihaz ozelligini gerektirir ve **bu projede o ozellik
etkin degil**. Ayri bolgelerle iki tabanin ikisi de CPU'da bilinir; cizim instance
vertex buffer'ini bolge tabanindan **bayt ofsetiyle** baglar ve komuttaki
`firstInstance` her zaman 0 kalir. Yani cihaz ozelligine hicbir bagimlilik yok.
Bedeli, yalnizca LOD ayrimi olan mesh'ler icin fazladan bir bolge.

### ★★★ Anlam degisikligi: butce -> hedef

Eski CPU yolu butceyi **"kameraya en yakin N instance tam, gerisi proxy"** diye
uyguluyordu (`nth_element` ile sira secimi). GPU'da tek gecisde tam siralama
yapilamaz. Yerine bir **mesafe esigi** tutuluyor ve her kare butceye gore
cozuluyor:

    yeniLodDistSq = min(lodDistSq, gozlenenEnUzakDistSq) * (izinVerilen / gorulen)

Tek adimda yakinsar cunku scatter bir **yuzeye** dagilir: D mesafesi icindeki
instance sayisi ~ D^2, ve `lodDistSq` zaten D^2'dir. `gozlenenEnUzak` kirpmasi
sart: onsuz esik 1e18 tohumundan butceye ~47 karede inerdi, yani sahne
yuklenirken saniyelik bir **tam detay takilmasi**.

Iki gozlenebilir sonuc, ikisi de kasitli:

1. Butce artik **sert tavan degil, HEDEF**. Hizli kamera hareketinde bir-iki
   kare asilabilir. Alan adi bu yuzden `m_rasterScatterTriangleTarget` oldu --
   kural 5: sessizce anlam degistirme, adi da degistir.
2. Buna karsilik **pop azalir**: sira tabanli secimde iki instance'in sirasi yer
   degistirince biri aniden proxy olurdu; mesafe esigi temporal olarak duzgun.

### Olcum yuzeyi

`viewport.frame_telemetry` (IPC + Python + panel) su alanlari kazandi:
`global_instance_buffer`, `gpu_culling`, `total_instances`, `cull_mesh_count`,
`draw_calls`, `visible_triangles`, `full_triangles`, `proxy_triangles`,
`full_instances`, `proxy_instances`, `scatter_triangle_target`.

★★ **`gpu_culling` false + `global_instance_buffer` true = sahne culling'siz ve
proxy'siz ciziliyor.** Bu kombinasyon bir arizadir ve ekranda dogru gorunur.
Bir raster zamanlamasina guvenmeden once bu iki alana bak.

★ GPU culling acikken sayilar GPU'dan okunur ve **bir kare geridir**: compute
kareyi yazar, CPU sonraki karede okur. Sabit kamerada fark yoktur.

### Ilk canli testin buldugu iki kusur (ayni partide duzeltildi)

**1. Butce yalnizca proxy'si olan gruplara uygulaniyordu.** Ilk hal
`FLAG_LOD_SPLIT`'i proxy mesh'in varligina bagliyordu. Eski CPU yolu boyle
degildi: ucgen kirpmasi HER scatter grubunda kosuyordu, yalnizca proxy
YUKLEMESI kosulluydu -- proxy yoksa butce disi instance'lar **cizilmiyordu**.
Ikisini birlestirmek, proxy'si olmayan her grupta butceyi **tamamen etkisiz**
birakti. Artik iki ayri bayrak: `kFlagLodSplit` (butce uygulanir) ve
`kFlagHasProxy` (demote edilenler cizilir).

**2. ★★★ Flat SoA scatter kaynaklarinin proxy'si HIC uretilmiyordu.**
`ensureScatterProxyMesh` imzasi `const std::vector<std::shared_ptr<Triangle>>*`
idi; flat dal `isScatterGroup = true` yapip `continue` ediyordu. Proje
sidecar'indan gelen her sey flat oldugu icin **yogun modern scatter'in proxy'si
yapisal olarak yoktu.** Bu, CLAUDE.md'nin adiyla uyardigi tuzagin ta kendisi:
facade tarayan kod flat mesh'leri "yok" sayar ve bunun hicbir belirtisi olmaz.

Duzeltme kaynak konumlari raster mesh'in kendi `cpuPositions`'indan okuyor.
Facade yolu icin sonuc BIREBIR ayni -- `ensureRasterMeshForTriangles` o diziyi
zaten `getOriginalVertexPosition`'dan dolduruyor -- ve flat dal da artik
proxy uretiyor.

**3. LOD alt siniri 25 m^2 -> 1 m^2.** Eski deger, instance'lari kameradan 5 m
icinde yogunlasan bir sahnede esigi tabana dayayip hedefi uygulanamaz kiliyordu.

★ **Bilinen kalan sinir:** tekduze yogun bir KUMEDE (butun instance'lar yaklasik
ayni mesafede) bir mesafe esigi ayirt edemez -- esik o mesafeyi gecince hepsi
birden doner. Belirtisi `full_instances`'in `total` ile 0 arasinda salinmasidir.
Cozumu mesafeye stokastik bir ayirici eklemektir (`hash01(i) * distSq <= esik`);
once belirtinin gerceklestigi olculmeli.

### Ayni ailedeki ikinci ariza (ayni partide duzeltildi)

`renderSelectionOutlineMaskReadback` per-mesh `instanceBuffer`'a ve
`instanceCount`'a bakiyordu; global yolda ikisi de bos/sifir oldugu icin maske
pass'i **hicbir sey cizmiyordu** ve secim anahatti sessizce kayboluyordu. Artik
sikistirilmamis global buffer'i `firstInstance` ile okuyor (bu gecis pass'inin
kendi kamerasi var, culling istemez).

## LOD gecis bandi + proxy materyali + Full preset (2026-09-01, yazildi/derlenmedi)

### Sorulan soru ve olcumun cevabi

Kullanicinin sorusu: *kademeli LOD ile goruntuyu bozmadan pathtrace hizina
ulasilabilir mi, yoksa sahne isigi/golge/PBR'ye mi gecelim.*

**"Pathtrace hizina ulasmak" yanlis hedef, ve karsilastirma yapisal olarak
asimetrik.** RT'de TLAS instancing sayesinde bir bitkinin bir milyon kopyasi bir
BLAS + bir milyon transformdur; isin maliyeti sahne karmasikliginda ~O(log n) ve
gorunen instance sayisindan neredeyse bagimsizdir. Raster'in maliyeti
**gonderilen ucgende dogrusaldir**, ustune yogun bitkide alpha-test edilmis
yapraklarin overdraw'i biner. Yani tam detayda raster o sahnelerde pathtrace'ten
daha hizli **degildir**, daha yavastir; tek kolu daha az cizmektir -- ki LOD tam
olarak budur. "LOD ile pathtrace hizina ulasalim" dairesel bir cumledir: LOD
hedefe giden bir taviz degil, **mekanizmanin kendisidir**.

★★★ Raster'in RT'yi gercekten yendigi tek yer: **accel yapisi yeniden
kurulmuyor.** Slider cekerken, mesh duzenlerken, sim scrub ederken RT her kare
BLAS/TLAS refit/rebuild oder; raster odemez. Preview raster'in durust isi budur
-- kararli haldeki kalite/ms degil, **geometri DEGISIRKEN interaktif kalmak** --
ve optimize edilecek sey de odur.

### Neden "kademeli LOD kademeleri" bu partide YAPILMADI

Bu depoda **genel amacli mesh decimator yok**. Arandi: terrain'deki "decimated"
izgara yeniden ornekleme (heightfield'e ozel), yoldaki polyline seyreltme --
ikisi de bu ise yaramaz. Proxy bir indirgenmis mesh degil, kaynaktan siluet
profili cikarilmis **96 ucgenlik bir impostor**: 4 dusey duzlem (X, Z, iki
capraz) x 6 dilim x cift yuzlu serit.

Yani yol haritasindaki "surekli LOD kademeleri" adimi kucuk bir adim degil, bir
**decimator projesine bagli**. Ondan once olculmesi gereken sey vardi:

### ★★★★ "Cok kaba"nin baskin sebebi poligon sayisi DEGILDI

`drawWithGBuffer` kosulu `!rmb.isScatterProxy` iceriyordu. Proxy'nin UV'si ve
materyal ID'si de yoktu, yani kosul zaten hicbir zaman gecmiyordu -- ama ayrica
**acikca disliyordu**. Sonuc: Material Preview modunda sahnenin uzak yarisi
dokulu PBR bitkilerden **dokusuz matcap mukavvaya** donuyordu. Poligon eklemeden
once duzeltilecek sey buydu.

### Bu partide yapilanlar

1. **Stokastik gecis bandi.** Esik instance basina sabit bir carpanla dagitiliyor
   (`jitter = 1 + 0.35 * (hash01(src)*2 - 1)`). Gecis suprulen bir cizgi yerine
   **cozunme** olur, ve gorunen sayi esigin surekli/monoton bir fonksiyonu
   haline geldigi icin **tekduze yogun kumede oran cozumu de yakinsar** -- onceki
   partinin acik kayitli belirtisi. Hash girdisi kaynak indekstir, kare sayaci
   degil: degisseydi pop yerine **titreme** olurdu, ki daha kotudur.

2. **Proxy materyale ulasiyor.** `cpuUVs` golge kopyasi eklendi; her yukseklik
   dilimi kaynagin UV **ortalamasini** ve baskin materyalini devraliyor; proxy
   `uvBuffer` + `matIdBuffer` uretiyor; `!rmb.isScatterProxy` kaldirildi.

   ★★★ UV siluet vertex'inden **degil** ortalamadan aliniyor. Ilk yazimda siluet
   vertex'i secilmisti ve bu alfa-test edilen yaprakta **ters teper**: yaprak
   kartinin geometrik silueti kartin kosesidir, atlasta o kose neredeyse her
   zaman seffaf marjdadir.

3. ★★★★★ **Impostor OPAK.** `material_preview_frag` opaklik < 0.1'de `discard`
   ediyor; proxy dilim basina tek UV tasidigi icin o UV seffafa denk gelseydi
   **seridin tamami silinirdi** -- sahnenin uzak yarisi "kabalasmaz", **bosalir**.
   Materyal ID'sinin 31. biti IMPOSTOR bayragi. Push constant'a alan
   **eklenmedi**: o blogun duzeni alti yerde yasiyor ve buyutmek cihaz esigini
   208 -> 224 bayta cikarirdi; ayrica impostor olmak cizimin degil
   **geometrinin** ozelligi, yani vertex basina tasinmasi zaten daha dogru.

4. **`Full` kalite preset'i.** Quality'nin "biraz dahasi" degil: Quality yalnizca
   ucgen hedefini buyutur ve uzak instance'lar hala proxy'ye duser; Full ayrimi
   **tamamen kapatir**. Frustum culling kapanmaz -- kamera arkasini cizmemek
   kalite kaybi degildir. Karar tek yerde:
   `VulkanBackendAdapter::rasterScatterLodSplitEnabled()`, ve GPU ile CPU yollari
   **ayni fonksiyondan** okur.

5. **Preset script/IPC'ye acildi** (`viewport.quality`, `viewport.set_quality`).
   ★★★ Gerekce: LOD sayilarinin **paydasi yoktu**. `full_instances = 4000` tek
   basina hicbir sey soylemez; tasarruf oldugunu bilmek icin ayrimi kapatip
   referansi gorebilmek gerekir. Bir orani, referansini gozleyemeden olcmek
   olcmek degildir. Probe: `scripts/probe_scatter_lod_reference.py`.

6. **Denetim boslugu:** `audit_shader_struct_layout.py` artik `MeshParam` <->
   `MeshBinding` ve `Globals` <-> `GlobalsGPU` aynalarini da kontrol ediyor.
   Bu cift denetim disindaydi ve bu partide `Globals` bir alan buyudu.

### Siradaki lever, bu partinin olcumune bagli

Impostor duzeltmesi olculdukten **sonra** siradakiler, en ucuzdan:
kameraya donen (ya da oktahedral harmanlanan) impostor -- ayni 96 ucgenle,
cunku suanki dort duzlem nesne uzayinda sabittir ve belirli acilardan bicak gibi
ince gorunur; sonra dilim sayisini yukseltmek; ve **ancak o zaman** bir
decimator + gercek LOD kademeleri, cunku bunlarin hicbiri yetmediyse maliyeti
gerekcelendiren bir olcum eldedir.

Aydinlatma tarafi bilerek **sonraya** birakildi ve sira bir tercih degil
**maliyet carpimi**: golge haritasi geometriyi isik/cascade basina yeniden
cizer, yani kotu bir LOD'un ustune aydinlatma kurmak o kotu LOD'u cascade sayisi
kadar carpar. Sirasi geldiginde tavsiye tam cok-isikli PBR degil **tek cascade
gunes golgesi**: mekansal okunabilirligin cogunu o verir, gerisi RT'nin zaten
daha iyi yaptigi isi kopyalamaktir.

---

## Sahne isiklari onizlemede — Faz 1 dilim 1 (2026-09-01, yazildi/derlenmedi)

### Neden bu adim, neden simdi

Onceki partinin LOD'u one alan gerekcesi **tukenmisti**: acik kusuru kapatmak,
olculebilirlik, ve golge gecislerinin kotu LOD'u carpmasi -- ucu de karsilandi.
Devam etmek icin impostor'in darbogaz oldugunu soyleyen bir olcum gerekirdi ve
yoktu. ★★★★ Ayrica test maliyeti acisindan asimetri var: **isik gorunur, LOD
gorunmezdi.** LOD probe scripti gerektirdi cunku basarisizlik bicimi *dogru
gorunen bir goruntuydu*; eksik golge ya da ters yon ekran goruntusunde gorulur.
Bu parti icin yeni probe scripti yazilmadi ve gerekmiyor.

### Yapilan

Material preview'a `Scene` aydinlatma preset'i eklendi. Fragment shader RT
hattinin okudugu **ayni** isik buffer'ini (`VulkanDevice::m_lightBuffer`,
`VkGpuLight[]`) binding 5'ten okuyor.

★★★ Ayri bir onizleme isik listesi **cikarilmadi**, bilerek: iki liste iki
dogruluk kaynagi demektir. Ayni buffer okundugu icin onizleme ile Rendered
arasinda "hangi isiklar var" sorusunda **ayrisma mumkun degil**; ayrisabilecek
tek sey isigin nasil degerlendirildigi, ve o da acikca yaklasiklik olarak
yazili.

Yon/dusum/koni semantigi `bsdf_scatter.glsl` ile birebir. ★ Yonlu isikte
`direction.xyz` **isiga dogru** bakar (yukleme tarafi zaten negatifliyor); bir
kez daha negatiflemek sahneyi ters taraftan aydinlatirdi.

### Yaklasikliklar, yazili olarak

- **Golge bounded atlas yaklasimidir.** `viewport.preview_lighting`, `shadows`
  ve `shadowed_light_count` degerlerini dondurur; atlas disinda kalan isik
  aydinlatmaya devam eder ama golge vermez.
- **Alan isigi merkezinden** ornekleniyor (RT rastgele nokta + MIS).
- **Dunya/HDRI ambient kanonik world verisinden gelir.** Studio/Outdoor baked
  env haritalari Scene'e sizmaz; yalniz descriptor fallback'i olarak bagli
  kalir ve `sceneFlags` kapaliyken shader tarafindan okunmaz.
- **Ust sinir 32 isik**, ve sessiz degil: bir kez log + `scene_light_count` ile
  `scene_light_total` ayri raporlaniyor.

### Yol boyunca bulunan yapisal kusurlar

1. ★★★ **Material preview hattinin IKI descriptor kurulumu var**
   (`VulkanBackendAdapter` ve `VulkanViewportBackend`). Yalnizca birini 10
   binding'e cikarmak, otekinde shader'in **statik olarak kullandigi** iki
   binding'in layout'ta hic bulunmadigi bir pipeline uretirdi. Guncelleme
   fonksiyonu bu yuzden **taban sinifta**.

2. ★★★ **Butun isiklar silinince isik sayisi sifirlanmiyordu.** `setLights`
   yalnizca `!gpuLights.empty()` iken yukluyordu; sahnedeki butun isiklar
   silindiginde `m_lightCount` eski degerinde kaliyor ve sahne **silinmis
   isiklarla** aydinlatilmaya devam ediyordu -- **RT dahil**. Kimse bunu bug
   diye raporlamaz: "isigi sildim ama sahne hala aydinlik" bir aydinlatma
   tercihi gibi okunur.

3. **Preview lighting preset'i viewport cache anahtarinda yoktu (canli test,
   2026-09-01; duzeltildi/derlenmedi).** UI/Python/IPC preset degerini ve
   `start_render` durumunu dogru degistiriyordu, ancak raster backend sabit
   kamerada temiz sahnenin onceki karesini yeniden yayimliyor; yeni push
   constant ancak kamera hareketi camera hash'ini degistirince ciziliyordu.
   `material_preview_lighting_preset` her iki Vulkan material-preview yolunun
   cache anahtarina eklendi. Mesh/pipeline yeniden kurulumu gerekmiyor.

Ayrica `VkGpuLight`'in **alti** GLSL kopyasi vardi ve hicbiri denetlenmiyordu;
bu parti altincisini ekledi ve altisi da denetime kaydedildi.

### Faz 1 dilim 2/3 -- tum isik golgeleri + world/HDRI ambient (2026-09-01, yazildi/derlenmedi)

- Ortak 4096x4096 D32 shadow atlas; 512x512 tile ve 64 tile kapasitesi.
- Directional: kamera merkezli, texel-snapped tek cascade.
- Spot: gercek cone acisindan perspective shadow view.
- Point: +X/-X/+Y/-Y/+Z/-Z alti yuz; cubemap yerine ayni 2D atlasin alti tile'i.
- Area: isik merkezinden/normalinden bounded perspective shadow ve daha genis PCF.
- Atlas tahsisi gorunur GPU light sirasinda deterministik. Point 6, diger tipler
  1 tile tuketir. Atlas dolunca isik aydinlatmaya devam eder ama golgesiz kalir;
  `shadowed_light_count` bunu mevcut Python/IPC durum yuzeyinde raporlar.
- Caster pass kamera raster yolunun ayni compact instance buffer ve indirect
  komutlarini kullanir. Opacity texture/material semantigiyle alpha cutout;
  materyalsiz geometri icin ayri opaque depth pipeline vardir.
- Scene ambient artik sabit Studio/Outdoor rig'inden gelmez. Color world dogrudan,
  HDRI/overlay kanonik yuklenmis environment image + rotation/intensity ile,
  Nishita overlay'in Mix/Multiply/Add/Replace modu korunarak; Nishita sky ise
  Vulkan SkyView LUT'u (LUT henuz hazir degilse bounded analytic fallback) ile
  diffuse ve specular katkisi verir.
- Scene world artik yalniz ikincil aydinlatma degildir: fullscreen background
  pass Color/HDRI veya Vulkan SkyView/Transmittance LUT Physical Sky'i cizer.
  Nishita sun ayri direct-light iterasyonu ve rezerve directional atlas tile'i
  ile yuzey aydinlatmasi/golge uretir.
- Physical Sky slider maliyeti LUT hesabindan cok eski yasam dongusuydu: her
  tikta `waitIdle`, uc image destroy/create ve descriptor rewrite yapiliyordu.
  Sabit boyutlu LUT image/sampler/descriptor kimligi artik korunur; etkilesimli
  guncelleme yalniz gercek LUT girdileri dirty oldugunda params upload + compute
  dispatch + layout barrier yapar. Sun intensity/size ve overlay gibi LUT disi
  world duzenlemeleri LUT'u yeniden uretmez.
- TDR takibi ikinci bir maliyeti ortaya cikardi: SkyView'in 64 adiminin her biri
  40 adimlik transmittance integralini yeniden cagiriyor, tek slider tikini
  yaklasik 84 milyon agir ornege cikariyordu. Transmittance artik phase 0'da bir
  kez uretilir; compute-to-compute image barrier'inin ardindan SkyView ve
  multi-scatter bu LUT'u bilinear okur. Boylece tek invocation'daki ic ice
  integral kalkar ve Physical Sky parametre surukleme Windows watchdog butcesi
  icinde bounded kalir.
- Canli testte raster frame-slot acquisition uyarisinin hemen ardindan TDR
  goruldu. Ring artik hata turunu `VkResult` olarak korur: device lost durumunda
  ayni kayip queue'ya synchronous fallback submit yapilmaz; son tamamlanmis kare
  korunup mevcut backend recovery tetiklenir. Diger gecici ring hatalari tum
  oturumu sync moda kilitlemez, bounded backoff sonrasi async ring yeniden kurulur.
- World UI dirty sahipligi ayrildi: sky/fog degisimi gas/VDB payload upload etmez;
  yalniz gercek sun direction/intensity degisimi light buffer ve shadow atlas'i
  dirty yapar. Realtime SkyView LUT 24 view-step kullanir ve surukleme sirasinda
  en fazla yaklasik 15 Hz yenilenir; hafif world-buffer parametreleri gecikmeden
  guncellenir.
- Fog density metre basina extinction olarak tanimlandi. Eski disabled `0.1`
  varsayilani `0.0001` degerine migrate edilir ve fog kapaliyken density aerial
  perspective hesabina sizmaz; boylece Enable Fog sonrasi kapatma goruntuyu geri
  getirir.
- Kamera-bagimli shadow kayitlari mapped SSBO'yu CPU'dan ezmez; ayni grafik
  komut tamponunda `vkCmdUpdateBuffer` ve iki yonlu buffer barrier ile guncellenir.
  Boylece frame-ring her kamera karesinde CPU drain'e dusmez.

Acik yaklasikliklar: area shadow merkez ornegidir; directional tek cascade
oldugu icin cok buyuk dis mekanlarda uzak shadow kapsami sinirlidir.

### Faz 1 dilim 3/3 -- gercek HDRI IBL (2026-09-01, yazildi/derlenmedi)

- Kanonik world HDRI degistiginde compute tarafinda 64x32 cosine-weighted diffuse
  irradiance, 256x128 dokuz seviyeli GGX prefilter ve 256x256 split-sum BRDF LUT
  uretilir.
- Rotation/intensity shader zamaninda uygulanir; kamera, rotation veya intensity
  degisimi convolution'i yeniden baslatmaz. Yalniz HDRI texture kimligi degisince
  irradiance/prefilter yenilenir; BRDF LUT backend omru boyunca bir kez uretilir.
- Scene Principled ambient diffuse, rough metallic/dielectric specular ve
  clearcoat bu haritalari tuketir. Pipeline/SPV veya guncel kaynak yoksa onceki
  bounded raw-environment yaklasimi korunur ve Python/IPC bunu
  `world_ibl_fallback=true` ile acikca raporlar.

---

Asagidaki bolumler bu sirayla yeniden ele alinacak nihai hedefi tarif ediyor.
Faz numaralari **eski haliyle** birakildi ki yukaridaki karar hangi metni
yeniden sirladigini soyleyebilsin.

---

## FPS gostergesi + katlanabilir Realtime kalite paneli (2026-09-02, DILIM 1 YAZILDI/DERLENMEDI)

Kullanici istegi: **raster modlarinda hiz FPS/sn cinsinden gorulsun**, ve
**realtime preview kalite ayarlari (golge cozunurlugu ve digerleri) UI'da uygun
bir yerde simge durumuna kuculebilen / acilabilen bir alanda toplansin.**

Ikisi de yeni bir render yolu acmiyor — mevcut raster yolunun uzerine
**okuma** (FPS) ve **yazma** (kalite kadranlari) yuzeyi koyuyor. O yuzden Faz 0'in
"UI, Python ve IPC parity" maddesinin altinda duruyorlar, ayri bir faz degil.

### A. FPS: turetilir, OLCULMEZ

Bugun HUD `frame_ms` ve asamalari yaziyor (`scene_ui_viewport.cpp`), ve
**ayni struct'i** `rt.viewport.frame_telemetry` donduruyor — ikinci bir otorite
yok. Eklenmesi gereken tek sey birim.

- ★★★ **Ikinci bir sayac KURULMAZ.** FPS `frame_ms`ten turetilir. Panelin kendi
  sayacini tutmasi, bu deponun en pahali hata sinifidir (panelin yalan soylemesi).
- ★★★ **Ama duzlestirme NEREDE yapilirsa otorite orada olur.** Anlik `1000/frame_ms`
  zipzip oynar; panel kendi kayan ortalamasini tutarsa panel ile script AYRISIR.
  ⇒ kayan pencere `RasterFrameTelemetry` icinde hesaplanir ve
  `fps_avg` / `frame_ms_avg` / `fps_window_frames` olarak **alan** halinde
  yayilir. Panel de script de ayni alani okur.
- ★★★★ **"Kare gonderilmedi" ≠ "0 FPS".** Uygulama bosta ya da pencere odakta
  degilken raster kare gondermiyor — bu oturumda olculdu: 2 saniyede
  `frames_submitted` delta **0**. Oraya `0 FPS` basmak "performans sifir" diye
  okunur. ⇒ ayri bir durum: son N ms'de gonderim yoksa HUD sayi degil
  **`— (idle)`** yazar, ve telemetri `fps_idle = true` doner.
  (Varsayilan bir olcum degildir; yokluk sifir degildir.)
- ★★★ **"FPS" TEK bir buyukluk degil, ve adlari ayrilmali.** Asenkron halkada iki
  ayri hiz var: host gecisinin hizi (`1000/frame_ms`) ve **gercekten sunulan**
  kare hizi (`frames_consumed` turevi). Halka bir kare geriden yayin yaptigi
  icin bunlar ozdes degildir. HUD'un birincil sayisi **sunum hizi** olmali;
  `frame_ms` yaninda kalir. Tek bir "FPS" etiketi altinda birlestirmek, bu
  oturumda uc sayaci yanlis okutan hatanin aynisidir — bir sayinin ADI ne
  olctugunu soylemez.
- Yalnizca raster modlarinda (Solid/Matcap/MaterialPreview). Rendered modda
  halka aktif yol degil; orada anlamli birim **sample/sn**'dir ve
  `ms_per_sample` zaten var. Ikisini ayni etikete koyma.

### B. Katlanabilir kalite alani

Yer: viewport uzerinde overlay, simge durumuna kuculur, tiklaninca acilir.

- ★★★ **Katlanma durumu UI-YEREL kalir ve IPC'ye ACILMAZ.** CLAUDE.md'deki
  ayrimin dogru tarafi budur: "panel acik mi" bir **cizim karari**dir. Ama
  icindeki her kadran bir **deger**dir ve IPC'ye acilir. Kural, paneli
  scriptlemeyi degil, panelin tuttugu DURUMU scriptlemeyi zorunlu kiliyor.
- ★★ Widget seviyesinde surus yasak: "Quality yazan dugmeye bas" degil,
  `viewport.set_quality`. Panel de o cagriyi yapar.

**Icerik — birinci dilim (mevcut olani toparlar, yenisini eklemez):**

| kadran | bugun | durum |
|---|---|---|
| Quality preset (Auto/Performance/Balanced/Quality/Full) | `viewport.set_quality` | ✅ var |
| Preview lighting preset (scene / three_point) | `viewport.set_preview_lighting` | ✅ var |
| Scatter ucgen hedefi | preset icinde dolayli | ⚠ dogrudan kadrani yok |
| **Shadow atlas + tile cozunurlugu** | preset ile 256/512/1024 tile; atlas 4096 | ✅ dilim 1 |
| Shadowed light butcesi | preset ile 4/8/16; direct light dongusu 32 | ✅ dilim 1 |

**Dilim 1 karari:** atlas 4096 olarak sabit tutuldu; preset yalniz tile
cozunurlugunu, caster-light butcesini ve receiver PCF ornek sayisini degistirir.
Boylece kalite degisimi ucustaki karenin ornekledigi image'i yok etmez ve
degisim basina bile device drain gerekmez. Esleme:

| preset | tile | scene shadow light | PCF | Scene BRDF |
|---|---:|---:|---:|---|
| Performance | 256 | 4 | 9 | GGX |
| Auto / Balanced | 512 | 8 | 9 | GGX |
| Quality / Full | 1024 | 16 | 25 | GGX |

Directional lights and Physical Sky sun use camera-centred cascades inside the
same atlas record: Performance uses 2 (middle/far), all other presets use 3
(near/middle/far). Cascade radii are texel-snapped and consume 2/3 tiles per
directional source. Receiver selection chooses the smallest cascade containing
the point; local point/spot/area shadow paths are unchanged.

Full ayrica mevcut sozlesmedeki gibi scatter proxy ayrimini kapatir. Physical
Sky sun preset'e gore iki/uc atlas tile'i ayirir; point light alti tile tuketir. Panel kapasite,
kullanilan ve butceyi ayri gosterir. Ayni degerler `viewport.quality` ile Python
ve IPC'den okunur. Panelin acik/kapali hali UI-yereldir.

Scene PBR de bu dilimde kapatildi: kalite profili Scene'i Blinn-Phong'a dusurmez;
direct diffuse Vulkan RT referansiyla ayni ortalama Fresnel enerji payini dusurur
ve clearcoat additive glow yerine base loblardan enerji alan ust katmandir.

**Material Graph / PBR parite dilimi (2026-09-02):** Realtime artik Vulkan RT'nin
flatten edilmis `MaterialProgram` bytecode'unu ve ayni `material_program.glsl`
evaluator'unu binding 16'dan tuketir. Base color, metallic, roughness, specular,
normal/bump, masked opacity ve emission color/strength graph ciktilari piksel
basina calisir. Program buffer buyumesi, ucustaki raster descriptor'unu degistirmeden
once viewport'u drain eder; RT descriptor duzeni ve binding 23 degistirilmedi.

Bu **tum materyaller tamdir** iddiasi degildir. `viewport.quality` ve Q paneli
parite durumunu deger olarak raporlar: opaque core `rt_aligned`; graph surface
`bounded` (pointiness, named attribute, traced AO ve time girdileri henuz neutral);
clearcoat `iridescent_lobe`; subsurface `radius_profile_approx`;
translucency `thin_surface_approx`; surface anisotropy
`unsupported_abi_conflict`; transparency `unsorted_alpha`;
transmission `screen_space_thickness`; resin interior
`procedural_interior_approx`. Transmission yolu donmus opak renk/depth kopyasi,
ayri transmissive replay, kapali mesh front/back kalinligi, IOR Fresnel,
bounded screen-space reflection, Beer-Lambert, rough refraction ve dispersion kullanir. Base Color texture RT'deki
gibi authoritative'dir ve depthless camda hacim tint'ine donusur. Ekran disi ve
birbiriyle ortusen saydam yuzeyler environment fallback ile bounded kalir.
Vulkan RT referansi degismeden kalir.

**Physical Sky reflection paritesi (2026-09-02):** Raster Scene yolu Vulkan
RT miss shader'inin LUT sonrasi multi-scatter, horizon sun broadening, limb
falloff ve kamera irtifasina bagli transmittance semantigini kullanir. Rough
metal/dielectric yansimalari tek bir normale bukulmus sky lookup yerine kalite
presetine bagli 4/8 deterministik cone ornegiyle filtrelenir; boylece zenit
mavisinin genis GGX lobunda yapay olarak baskinlasmasi engellenir. Bu yalniz
raster Material Preview yoludur; Vulkan RT pipeline/binding/traversal degismedi.

Opaque PBR ikinci diliminde clearcoat, Vulkan RT'deki ayni ince-film
`OPD / cos(theta)` renk modelini direct ve environment loblarinda tuketir.
SSS'nin radius/scale/color alanlari artik wrapped-diffuse profilini kanal bazinda
etkiler; `translucent` on yuz difuz enerjisini azaltip arka-isik lobuna aktarir.
Bunlar traversal/random-walk degil, acikca bounded yaklasimdir. `anisotropic` ve
`sheen` alanlari mevcut ABI'da water wave speed/strength ile ayni slotlari
kullandigi icin ayristirilmadan surface anisotropy acilmayacak.

**★★★ Atlas cozunurlugu bugun bir sabit**, tile ise preset degeridir:
`MaterialPreviewShadow.cpp` → atlas 4096; tile 256/512/1024
⇒ 256/64/16 tile kapasitesi, ve direct-light ust siniri 32 (+1 gunes tile'i).

Bunu runtime yapmanin iki tuzagi var, ikisi de bu depoda bir kez pahaliya
ogrenildi:

1. ★★★ **Atlasi yeniden ayirmak = ucustaki karenin ornekledigi image'i yok
   etmek.** Bu, atmosfer LUT'unda cihazi kaybettiren sinifin ta kendisi
   (2026-09-02). ⇒ yeniden ayirma **degisim basina** drenaj ister, kare basina
   degil; ve descriptor yeniden yazimi drenajdan SONRA gelir.
2. ★★ **Tile cozunurlugu isik butcesini BELIRLER.** Kapasite = (atlas/tile)^2.
   Kullanici tile'i 512→1024 yaparsa kapasite 64→16'ya duser ve golgeli isik
   sayisi **sessizce** kirpilir — belirtisi "bazi isiklar golge vermiyor" olur,
   yani hicbir hata. ⇒ panel kapasiteyi ve kullanilani **deger olarak**
   gostermeli, tipki `viewport.preview_lighting`'in `scene_light_count` ile
   `scene_light_total`'i ayri ayri raporlamasi gibi. Ayni sayilar IPC'den de
   okunmali.

Her kadran sozlesmedeki gibi `supported` / `active` / `approximate` /
`fallback_reason` raporlar; manuel degisiklik preset'i `Custom` yapar; deger
proje ile saklanir (bkz. yukaridaki *Realtime kalite ayarlari sozlesmesi*).

### Dokunuslar (kural 1)

| katman | is |
|---|---|
| Cekirdek | `RtApi.h` + `RtApiViewport.cpp`: `getViewportQualitySettings` / `updateViewportQualitySettings` (kismi guncelleme, sinir disi REDDEDILIR) |
| IPC | `RtIpc.cpp`: `viewport.quality_settings` / `viewport.set_quality_settings` |
| Python | `RtPython.cpp`: `rt.viewport.quality_settings()` / `set_quality_settings(**kw)` |
| Yetki | `RtIpcSecurity.cpp`: `viewport.` prefix'i **zaten var** — ek is yok |
| Tarif | `gen_ipc_descriptors.py` + overlay: tile→kapasite iliskisini ve drenaj maliyetini **notes**'a yaz |
| Telemetri | `RasterFrameTelemetry`: `fps_avg`, `frame_ms_avg`, `fps_window_frames`, `fps_idle` |

### Kabul testi (yazilmadan once tarif edilir)

1. Balanced→Quality yapilir: `viewport.quality` tile'i 512→1024, kapasiteyi
   64→16, PCF'yi 9→25 raporlar; panel ayni sayilari gosterir.
2. Ayni degisiklik sirasinda shadow atlas image kimligi korunur; resource drain
   artmaz ve `backend` `vulkan` kalir (cihaz kaybi yok).
3. Bosta duran uygulamada FPS alani `0` degil `idle` raporlar.
4. ★ En sinsi: atlas kucultuldukten sonra sahne **makul gorunur** ama uzaktaki
   isiklarin golgesi sessizce kaybolur. Testin bakmasi gereken sey ekran degil,
   kapasite/kullanilan cifti.

---

## ✅ KAPANDI: realtime ile Rendered arasinda parlaklik/kontrast farki (2026-09-02)

> Bu bolum TEShIS kaydidir; duzeltme asagidaki "tek goruntuleme donusumu"
> bolumunde ve olcumle dogrulandi (preset yayilimi 0.084 → 0.0004).

Kullanici bildirimi: **realtime/material preview, Rendered'dan daha parlak ve
daha kontrasli.** Ilk hipotez "Rendered'in coklu sicrama destegi".

**★★★ Coklu sicrama bu farkin YANLIS ilk suphelisi.** GI enerji **ekler**:
acikken Rendered'in daha PARLAK ve golgeleri DOLMUS (daha az kontrasli) olmasi
beklenir. Yani "daha kontrasli" GI yoklugu ile tutarli, ama **"daha parlak"
sicrama ile aciklanamaz** — ters yone gider. Parlaklik baska bir yerden.

Kod okunarak bulunan iki bagimsiz aday (**hicbiri olculmedi**):

**1. ★★★★ Dogrudan difuz teriminde 1/π yok.**
`shaders/material_preview_frag.frag:819` →
`diffuseLit += diffuseAlbedo * radiance * NdotL;`
Lambert BRDF `albedo/π`'dir; yol izleyici boluyor, preview bolmuyor
⇒ **π ≈ 3.14x sicak**.

★★★ Telafi **tek dalda**: satir 922
`float diffuseWeight = (qualityMode <= 1u) ? 0.35 : 1.0;`
0.35, 1/π = 0.318'e fazlasiyla yakin — eksik fiziksel terimin yerine konmus bir
**sihirli katsayi**, ve yalnizca dusuk kalite dalinda gecerli.

★★★ **SINANABILIR TAHMIN.** `VulkanViewportBackend.cpp:3412`:
`Performance → qualityMode 1`, `Balanced/Auto → 2`, `Quality/Full → 3`.
Dolayisiyla `viewport.set_quality` ile **Performance ↔ Balanced** arasinda gidip
gelmek, isikta hicbir sey degismeden dogrudan difuzu **~2.86x** degistirmeli.
Varsayilan `Auto` sicak dala dusuyor. Bu gorulurse kok kesinlesir.

**2. ★★★ Preview `post.*` ayarlarini hic okumuyor.**
Ayni shader'in sonu sabit `acesTonemap` + `linearToSRGB`. Rendered gercek post
hattindan geciyor (exposure / gamma / tone_mapping / saturation). **Exposure
preview'a hic uygulanmiyor** ⇒ kullanici exposure'i degistirdiginde iki mod
arasindaki fark da degisir. Parlaklik+kontrast farkinin ikinci, bagimsiz kaynagi.

### Olcum plani (sirali, ucuz olan once)

1. Kamera sabit, `viewport.set_quality` Performance ↔ Balanced, `render.probe`
   `mean_luminance`. ~2.9x fark ⇒ aday 1 dogrulandi.
2. `post.set_exposure` degistir, iki modda da `render.probe`. Preview'in
   ortalamasi kimildamiyorsa ⇒ aday 2 dogrulandi.
3. Ancak bundan SONRA GI'ye bak — ve GI'nin isareti parlaklik degil,
   **histogramin karanlik ucu** (golge dolgusu) olmali.

★ **Kontrollu sahne sart:** tek yonlu isik + tek difuz duzlem, world kapali.
Karisik sahnede uc etki birbirini maskeler ve hangi hipotezin dogrulandigi
soylenemez.

★★ Duzeltirken: 1/π'yi eklemek ve 0.35'i kaldirmak **ayni degisiklikte**
yapilmali, yoksa dusuk kalite dali bu kez 1/π kadar KARANLIK olur. Ve iki
kalite dalinin ayni parlakligi vermesi bir **kabul kriteri**dir: kalite preset'i
BRDF kademesini degistirir, pozlamayi degil.

---

### ✅ OLCULDU (2026-09-02, kullanicinin sahnesi, 1680x945, `render.probe`)

Kamera ve isik sabit; yalnizca belirtilen kadran degisti.

**1. ✅ Aday 1 DOGRULANDI — kalite preset'i POZLAMAYI degistiriyor.**

| preset | qualityMode | diffuseWeight | mean_luminance |
|---|---:|---:|---:|
| Performance | 1 | 0.35 | **0.4055** |
| Balanced / Quality / Full / Auto | 2-3 | 1.00 | **0.4893** |

Ayni sahne, ayni isik, ayni kamera — **yalnizca BRDF kademesi degisti ve goruntu
%21 parlaklasti**. Ekran uzayindaki bu oran, lineer ~2.86x'in ACES + sRGB
tarafindan sikistirilmis halidir (min_luminance 0.0801→0.1003, max 0.9725→1.0000
kirpma). ★★★ Kabul kriteri ihlali: **kalite preset'i BRDF kademesini
degistirmeli, pozlamayi DEGIL.**

**2. ❌ Aday 2 YANLIS CIKTI — preview `post.*`'i OKUYOR.**
Tahminim "preview exposure'i hic uygulamiyor" idi; **olcum eledi**:

| exposure | 0.25 | 1.0 | 2.0 | 4.0 |
|---|---:|---:|---:|---:|
| preview mean | 0.1204 | 0.4893 | 0.7592 | 0.9622 |

Sebep: exposure shader'da degil, **CPU yuzey post gecisinde** uygulaniyor
(`Main.cpp` → `applyToneMappingToSurfaceWithCamera`), ve o gecis her iki modun
da uzerinden geciyor. ★★ Kod okuyarak kurulan makul hipotez, tek bir kadrani
supurmekle elendi.

**3. ⚠ Ama yerine BASKA bir tutarsizlik cikti: IKI FARKLI TONEMAP OPERATORU.**
Preview fragment shader'i **kendi icinde** `acesTonemap` + `linearToSRGB`
uyguluyor; projenin `post.tone_mapping` ayari ise **`none`**. Yani Rendered
duz gidiyor, preview ACES'ten geciyor — ve ustune CPU post gecisi ikisine de
exposure/gamma uyguluyor, yani preview'da **cift kodlama** var (zaten sRGB'ye
kodlanmis 8-bit degerlere exposure uygulanmasi lineer degildir).

**4. Parity acigi — iki mod, ayni sahne:**

| | mean | min | max | ust kova (7/8) |
|---|---:|---:|---:|---:|
| MaterialPreview | **0.4893** | 0.1003 | **1.0000** (kirpiyor) | 47 652 px |
| Rendered (24 spp) | **0.3376** | 0.0000 | 0.8510 | **0** px |

- Preview ortalamasi Rendered'in **1.45 kati**.
- Preview **kirpiyor**, Rendered kirpmiyor.
- ★★★ Preview'in **siyahi yok**: taban 0.1003. Rendered 0.0'a iniyor.
  Kaynak: shader'daki sabit ambient
  (`ambient = envDiffuse * diffuseColor * kD * 0.36`).

★★★★ **Kullanicinin "coklu sicrama" hipotezi olcumle ELENDI.** Rendered
GI'siyle birlikte **DAHA KARANLIK** (0.3376 < 0.4893). GI daha fazla isik
eklemis olsaydi ters yonde olurdu. Ve golgeleri dolduran taraf Rendered degil,
**preview** — ama gercek GI ile degil, hicbir zaman sifira inmeyen **sabit bir
ambient** ile. Yani preview'in karakteri: **kaldirilmis siyahlar + patlamis
parlaklar**; ikisi de fizikten degil, iki ayri sahte terimden geliyor.

**Kalan kok siralamasi (olculmus, tahmin degil):**
1. Difuzde eksik `1/π` (≈ 2.86x) — dogrulandi, en buyuk pay.
2. Sabit ambient tabani — siyahi kaldiriyor, kontrasti sahte gosteriyor.
3. Preview-ici ACES ile projenin `tone_mapping=none` ayarinin celismesi.
4. GI — bu listede **en son**, ve isareti parlaklik degil histogramin karanlik ucu.

★ Not: Performance preset'i (0.35 agirligi) Rendered'a **daha yakin** (0.4055 vs
0.3376). Yani sihirli katsayi kazara kabaca dogru; sicak dal varsayilan olan
`Auto`'nun dustugu daldir.

### Post-processing realtime'da CPU'da — ve bu, yukaridaki parlaklik farkiyla AYNI IS

Kullanici sorusu: *realtime de post'u Rendered gibi GPU'da yapmali, CPU-GPU
transfer maliyeti olmamali — mevcut yapiyor mu?*

**Hayir. Bugunku realtime zinciri:**

```
raster ciz (GPU)
  -> vkCmdCopyImageToBuffer            (GPU -> host-visible buffer)
  -> invalidate + memcpy               (consumeNewestReady)
  -> memcpy                            (presentCachedRasterFrame -> SDL surface)
  -> applyToneMappingToSurfaceWithCamera   ← ★ TAM KARE, CPU'DA
  -> SDL_UpdateTexture
```

Olculebilir kanit: `VulkanViewportBackend.cpp` icinde `m_tonemappedImage` /
`hasTonemapPipeline` gecen **0** yer var; `VulkanBackend.cpp` (RT yolu) icinde
**18** yer var. Yani **GPU tonemap hatti mevcut ama realtime onu hic kullanmiyor.**

★★★ **Shader'in kendi icindeki `acesTonemap` + `linearToSRGB` tam olarak bu
bosluga konmus bir yamadir** — ve yukaridaki 3. bulgunun sebebi odur: projenin
`post.tone_mapping` ayari `none` iken preview ACES uyguluyor, ustune CPU post
gecisi zaten sRGB'ye kodlanmis 8-bit degerlere exposure/gamma uyguluyor.
**Cift kodlama.** Yani "post'u GPU'ya tasi" ile "iki mod ayni parlakligi versin"
AYNI ISIN iki yuzu: post GPU'ya tasinip shader kendi tonemap'ini birakinca iki
mod TEK operator zincirini paylasir.

**Iki ayri is, cok farkli boyutta — karistirma:**

**(a) Post'u GPU'ya tasi.** Readback'ten ONCE bir compute gecisi; shader'daki
ACES/sRGB sokulur; `post.*` tek yerden okunur. Orta boy, kapsami belli, ve
parlaklik parity'sini de kapatir. ★ Kabul: `post.set_exposure` her iki modda
ayni orani uretmeli; `tone_mapping=none` iken preview de duz olmali.

**(b) CPU-GPU transferini TAMAMEN kaldir.** Bu **yapisal** ve (a)'dan cok daha
buyuk: uygulama viewport'u SDL_Renderer ile sunuyor
(`ImGui_ImplSDLRenderer2_Init`, `Main.cpp`), yani **Rendered dahil her yol**
pikselleri CPU'ya indirmek zorunda. Readback'i kaldirmak Vulkan swapchain'e
gecmek ve ImGui'yi Vulkan backend'ine tasimak demek — ayri bir proje.

★★ Yani "hic transfer olmamali" hedefi dogru ama (a) yapilmadan (b)'ye
gecmenin anlami yok: (a) transferi kaldirmaz, **CPU'daki tam-kare post
maliyetini** kaldirir ve dogruluk hatasini duzeltir. (b) transferi kaldirir.

### ✅ ÖLÇÜLDÜ (2026-09-02, 3840x2160, material, Vulkan viewport)

Kullanicinin sorusu: *cizim SDL2/OpenGL yuzeyi kullaniyorsa pikseller hic CPU'ya
inmek zorunda degil demektir, bu buyuk maliyetten kurtarir mi?*

**Premis yarim dogru, sonuc degismiyor.** `Main.cpp` penceresini
`SDL_WINDOW_OPENGL` veya `SDL_WINDOW_VULKAN` bayragi OLMADAN aciyor ve
`SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED)` cagiriyor; Windows'ta
SDL'in ilk hizlandirilmis surucusu **direct3d11**. Yani pikseller GPU'da, ama
**SDL'in D3D11 baglaminda**, bizim `VkDevice`'imizde degil. Bugunku yol
GPU(Vulkan) → CPU → GPU(SDL/D3D11); ara adim tembellik degil **API siniri**.

**Kac ms oldugu artik tahmin degil.** `Probe-PresentCost.ps1`, 20 gecerli ornek,
kamera hareketiyle surulen dogal kare dongusu, hepsinde `dStale=0`:

| alan | ort (ms) | medyan | min | max |
|---|---:|---:|---:|---:|
| frame_ms | 10.565 | 10.729 | 9.223 | 12.957 |
| cpu_record_ms | 4.484 | 4.302 | 3.128 | 6.734 |
| submit_ms | 0.021 | 0.019 | 0.017 | 0.035 |
| slot_wait_ms | 0.003 | 0.003 | 0.003 | 0.004 |
| **host_read_ms** | **3.056** | 2.976 | 2.767 | 3.897 |
| **present_ms** | **2.785** | 2.746 | 2.613 | 2.981 |

`host_read + present = 5.84 ms`, yani **backend karesinin %55.3'u**. Efektif
~10.6 GB/s ile kare basina 2 x 31.64 MB kopya. Bu iki gecis tamamen kalksa
10.57 ms → 4.72 ms (94.6 → 211.7 FPS — backend yarisi icin).

★★★ **Ve bu bir ALT SINIR: zincirde iki kopya daha var ve ikisi de
olculmuyordu.** Backend telemetrisi `renderProgressive` donunce biter; ana dongu
sonra `original_surface → surface` (post gecisi veya `gpu_noop_post` dalinda duz
kopya) ve `surface → SDL_Texture` (`SDL_UpdateTexture`) yapiyor. 4K'da kare
basina **dort** tam-kare 31.6 MB gecisi. Bu partide ikisi de olcume baglandi
(`display_post_ms`, `display_texture_upload_ms`, `display_loop_period_ms` —
`viewport.frame_telemetry`), sayilar bir sonraki build'de okunacak.

★★ **Olcum tuzagi, bir kez kuruldu:** `viewport.render_frames`
`render_progressive_pass(nullptr, nullptr, ...)` cagiriyor, yani SDL surface
NULL ve `presentCachedRasterFrame` memcpy'si hic calismiyor. Onunla olculen
`present_ms` ~0.00 cikar ve "sunum bedava" diye okunur. Sunum yalnizca **dogal**
kare dongusunden olculur.

**Karar sirasi (sayilarla):**
1. **Post'u GPU'ya tasi.** Sunum mimarisinden BAGIMSIZ, ve tek basina dogru:
   CPU'daki tam-kare post kalkar, shader'daki ACES sokulur, `post.*` tek yerden
   okunur ⇒ parlaklik parity'si de ayni degisiklikte kapanir.
2. **Sonra sunum portu.** Üç yol var: (i) Vulkan external-memory ile SDL'in
   D3D11 dokusuna interop — proje OIDN icin zaten bellek disa aktariyor, ama
   `SDL_RenderGetD3D11Device()` ile SDL_Renderer'in altina inmek gerekir,
   kirilgan; (ii) viewport'u Vulkan swapchain'den sunup ImGui'yi SDL'de birakmak
   — **olmaz**, iki API ayni pencere yuzeyine sahip olamaz; (iii) tum UI'yi
   Vulkan'a tasi (`SDL_WINDOW_VULKAN` + swapchain + `ImGui_ImplVulkan`) — dogru
   son durum, sifir readback, ve butun frame-ring/stale makinesini de ortadan
   kaldirir; ayri bir proje.
3. ★ Karar esigi cozunurluge bagli: kopya piksel sayisiyla olcekleniyor, yani
   1080p'de bu sayinin ~1/4'u. **4K hedefse (i)/(iii) gerekcelidir; 1080p'de
   ayni is ~1.5 ms'dir ve once (1) yapilmalidir.**


⚠ (a)'nin maliyet olcumu **YAPILAMADI**: olcum sirasinda ana dongu periyodu
~285 ms'ye cikmisti (bir Rendered ↔ MaterialPreview gidis-donusunden sonra,
`samples` 0'da sabit olmasina ragmen), ve CPU post'un birkac ms'lik farki bu
gurultunun altinda kaldi. ★ O yavaslamanin kendisi ayrica bakilmayi hak ediyor —
ayni oturumda mod degisiminden ONCE ayni sahnede dongu ~5 ms idi.




---

---

## ✅ YAPILDI: tek goruntuleme donusumu (2026-09-02, DERLENDI ve OLCUMLE DOGRULANDI)

Yukaridaki 1. maddedir. Kapsam bilerek "post'u ayri bir GPU gecisine tasi"
degil, **operator birligi** olarak alindi; sebebi asagida.

### Ne vardi

Raster viewport'ta **uc** ayri goruntuleme donusumu bulundu (biri raporda
yoktu, kod okunurken cikti):

| yer | operator | kodlama |
|---|---|---|
| `tonemap.comp` (Rendered) | sabit Reinhard | sRGB (analitik) |
| `material_preview_frag` (nesneler) | sabit ACES | `pow(1/2.2)` |
| `material_preview_sky` (gokyuzu) | sabit ACES | `pow(1/2.2)` |

Yani **ayni sahnede gokyuzu ile nesne ayni operatorden geciyordu ama Rendered
baskasindan**, ve `post.*` ayarlari GPU'ya HIC ulasmiyordu: CPU tarafi onlari
sonradan, zaten sRGB'ye kodlanmis 8-bit degerlere uyguluyordu.

★ Ayrica onizlemenin `linearToSRGB` adli fonksiyonu **sRGB degildi**, duz
`pow(1/2.2)` idi — adi yaptigi isi yanlis soyluyordu.

### Ne yapildi

1. **`shaders/post_chain.glsl` (yeni)** — projenin TEK goruntuleme donusumu.
   `ColorProcessor::processColor` ile ayni sira: exposure → tonemap → renk
   sicakligi → doygunluk → gama → clamp → vignette → sRGB. AGX/ACES/Uncharted/
   Filmic operatorleri C++ karsiliklariyla birebir.
   ★★★ `None` dali **Reinhard**'dir. Bu bir adlandirma borcu ve bilerek
   korundu: bu projede `None`, "HDR'i 0-1'e indirmeyi varsayilan operator
   yapsin" anlaminda kullanilmis. Anlamini degistirmeden adini degistirmek
   ayri bir is; **sessizce anlam degistirmek yasak**.
2. `tonemap.comp`, `material_preview_frag.frag`, `material_preview_sky.frag`
   kendi tonemap'lerini birakti ve bu dosyayi include ediyor.
3. **`g_display_post` aynasi** (`globals.h`): backend'ler `g_ctx`'i gormedigi
   icin post ayarlari onlara ulasamiyordu. Tek yonlu ayna; sahibi hala
   `ColorProcessor`, Main her karede yansitiyor.
4. **★★★★ Difuz teriminde `1/PI` eklendi ve `diffuseWeight = 0.35` /
   `specularWeight = 0.15` sihirli katsayilari SILINDI.** Ucuz Blinn-Phong
   lobu artik `(n+2)/(8*PI)` ile normalize ediliyor — yani 0.15 bir katsayi
   degil, eksik bir normalizasyondu, ve uslu terime bagli olmadigi icin
   parlaklik puruzlulukle yanlis yone kayiyordu.
   `qualityMode` artik yalnizca lobun **bicimini** secer, siddetini degil.
5. **CPU tam-kare post gecisi kaldirildi** (`Main.cpp`): kosul
   `hasNoOpColorProcessing` istiyordu, yani post varsayilan DEGILSE tam-kare
   CPU gecisi kosuyordu — ve yanlis uzayda. Artik GPU her zaman uyguluyor,
   CPU gecisi duz kopyaya indi. `hasNoOpColorProcessing` olu kaldigi icin
   **sokuldu** (kural 5).
6. **Ayna IPC'den gorulebilir:** `viewport.preview_lighting` artik
   `display_tone_mapping` / `display_exposure` / ... doner. Bunlar bir AYAR
   degil, **shader'a giden degerlerdir**. `post.get` ayarin ne oldugunu
   soyler; bunlar onizlemenin onu gorup gormedigini. Ikisi ayrisirsa onizleme
   yalan soyluyor demektir.

### Kabul testi

`scripts/ipc/Probe-DisplayParity.ps1` (+ `x64/Release/` kopyasi). Kapilar:

1. ayna == `post.get`;
2. ★★★ **kalite preset'i `mean_luminance`'i degistirmemeli** (yayilim < 0.01).
   Duzeltmeden once: Performance 0.4055 vs digerleri 0.4893, yani %21;
3. exposure iki modda ayni orani uretmeli (fark < 0.10);
4. mutlak parity — **kapi degil bilgi**. 1.00 beklenmiyor: onizleme hala
   yaklasiklik (alan isigi merkezden orneklenir, GI yok). Onceki oran 1.449.

★★★ **En sinsi basarisizlik:** Rendered'in gorunumunun degismesi. Varsayilan
ayarlarda cikti eskisiyle birebir ayni olmali (`None` → Reinhard). Sayi makul
ama farkli cikarsa (0.30 / 0.37 gibi) kimse bunu bug diye raporlamaz.

### ★★ Neden "post'u ayri bir GPU gecisine tasi" YAPILMADI

Onceki raporda bu is "orta boy" denmisti; kodu okuyunca **degil**. Ayri bir
post gecisi HDR bir render hedefi gerektiriyor (RGBA16F), o da ayni render
pass'e cizen **yedi** shader'in cikti uzayini degistirmek demek —
`solid_frag`, matcap, hair, partikul, edit overlay dahil. Bunlar
display-referred cizimler; tonemap'ten gecirmek onlari bozar, gecirmemek icin
render pass'i **ikiye ayirmak** ve pipeline'lari yeniden dagitmak gerekir.

Yani dogru siralama: **once operator birligi (bu parti), sonra HDR hedefi**.
HDR hedefinin kazanci parlaklik parity'si degil (o simdi kapandi); bloom,
gercek HDR grading ve post'un piksel-basina degil fragment-basina kosmamasi.

★ Bu parti CPU'daki tam-kare post gecisini kaldirir; **GPU→CPU transferini
kaldirmaz**. O ayri ve daha buyuk is (yukaridaki olcum bolumu).


## Nihai ürün sözleşmesi

RayTrophi Realtime, mevcut raster preview'a birkaç efekt ekleyen geçici bir yol
değildir. Vulkan üzerinde çalışan, sahnenin kanonik verisini kullanan ve aynı
materyal/ışık/hacim anlamlarını Path Traced görünümle paylaşan ayrı bir gerçek
zamanlı render yoludur.

Viewport ürün modeli:

| Mod | Amaç |
|---|---|
| Solid | En düşük gecikmeli modelleme ve yerleşim |
| Matcap | Form, normal ve sculpt kontrolü |
| Material Preview | Sahne ışıklarından bağımsız stüdyo/HDRI materyal kontrolü |
| Realtime | Gerçek sahne ışıkları, gölgeler, transparanlık, hacimler ve yaklaşık GI |
| Path Traced | Nihai fiziksel referans ve kalite doğrulaması |

Realtime'ın nihai kapsamı şunların tümüdür:

- Principled yüzey modeli, Material Graph ve texture/procedural girdileri;
- opacity/cutout, blend, emission, normal, bump/height, metallic, roughness,
  anisotropy, sheen, clearcoat, iridescence ve yaklaşık SSS;
- fiziksel IOR, Fresnel, ince ve kalın transmission, absorption, dispersion ve
  kapalı mesh iç ortamları;
- directional, point, spot ve area light ile cache'li yumuşak gölgeler;
- HDRI/world, atmosphere ve emissive katkılar;
- hair, particles, foliage, terrain, water/liquid ve fluid surface;
- VDB, üretim simulation domain gas/smoke/fire/cloud ve mesh volume boundary;
- raster taban çizgisi üzerinde isteğe bağlı Vulkan ray-query kalite katmanı.

### Varsayilan viewport ve iki modlu isik authoring gecisi (2026-09-01)

Vulkan bulunan kurulumlarda baslangic artik mevcut `shading_mode=1` raster
yoludur ve bunun varsayilan aydinlatmasi `Scene`'dir. UI bu gecis doneminde
yalniz iki anlamli secenek gosterir:

| UI secenegi | Kanonik API/IPC adi | Anlam |
|---|---|---|
| Scene (Realtime) | `scene` | Gercek sahne isiklari, shadow atlas ve kanonik world/HDRI/Physical Sky |
| 3 Point (Material Preview) | `three_point` | Sahne isiklarindan bagimsiz materyal inceleme rig'i |

Eski proje enum degerleri korunur; `Classic`, `Studio` ve `Outdoor` girdileri
yuklemede/otomasyonda `three_point` davranisina normalize edilir. Boylece eski
dosyalar kirilmaz ama UI dort farkli isik gercekligi sunmaz.

Bu gecis **nihai ayri Realtime viewport kimligi tamamlandi** demek degildir.
Mevcut material raster yolu bugunden Scene ile varsayilan olur; Faz 1'in gercek
`ViewportMode::Realtime` frame graph'i capability, kalite ve fallback kapilarini
tamamladiginda ayni urun rolunu devralir. Desteklenmeyen makinede Realtime adi
altinda sessiz Material Preview fallback'i yapilmaz; Vulkan yoksa mevcut
Rendered fallback'i acikca raporlanir.

Eski `GasVolume/GasSimulator` üretim fallback'i değildir. Realtime gas yolu,
`VULKAN_PRODUCTION_VOLUMETRICS_ROADMAP.md` sözleşmesindeki GPU-resident simulation
field'larını tüketir. Bir ikinci gas veri modeli kurulmayacaktır.

## Değişmez mimari kuralları

1. Sahne geometrisinin tek otoritesi flat `TriangleMesh` / DNA SoA'dır. Raster,
   shadow ve ray-query yolları per-face `Triangle` facade koleksiyonundan sahne
   üretmez.
2. UI, embedded Python ve IPC aynı `RealtimeRenderSettingsService` ve aynı
   renderer core operasyonlarını çağırır. Panel-only özellik kabul edilmez.
3. Realtime farklı bir materyal sistemi kurmaz. Ortak GPU material ABI ve ortak
   Material Graph derleme çıktıları kullanılır; yalnız realtime'a özgü bounded
   lowering/caching eklenebilir.
4. `VulkanViewportBackend.cpp` yalnız entegrasyon kablosu alır. Frame graph,
   pass'ler ve kaynak yönetimi yeni, odaklı modüllerde yaşar.
5. Her pass açık GPU zaman, VRAM ve örnek bütçesine sahiptir. Quality preset'i
   gizli ve sınırsız iş başlatmaz.
6. Desteklenmeyen bir özellik sessizce yanlış görünmez. Capability/status
   yüzeyi `supported`, `approximate`, `fallback` ve nedeni raporlar.
7. Temporal geçmiş; kamera kesmesi, topology/material/light revision değişimi,
   resolution/quality değişimi ve simulation discontinuity ile deterministik
   olarak geçersiz kılınır.
8. Instance sayısı CPU draw-call sayısı değildir. Aynı mesh/material kullanan
   instance'lar persistent GPU instance verisi ve instanced/indirect draw ile
   yürür. `Quality` ve `Cinematic` görünür instance düşürmez veya proxy ile
   değiştirmez; yalnız doğru frustum/occlusion culling uygulayabilir.

## Mevcut durum — 2026-08-31 statik denetimi

| Alan | Bugünkü durum | Realtime açığı |
|---|---|---|
| Backend ayrımı | Ayrı `VulkanViewportBackend` var | Render graph/pass sahipliği yok |
| Geometri | Flat SoA raster build/refit ve instance sync var | Shadow/G-buffer revision takibi yok |
| Materyal | Ortak Material/MaterialExt ABI; GGX, textures, terrain, masked opacity, normal, emission, clearcoat, sheen, yaklaşık SSS ve bounded surface Material Graph var | Transmission/resin/interior, graph pointiness/named attribute/traced AO/time ve özel pass ayrımı eksik |
| Işıklar | Sahne ışıkları viewport backend'e gönderiliyor; GPU light ABI point/directional/area/spot taşıyor | Preview shader gerçek buffer yerine sabit üç ışık kullanıyor |
| Gölgeler | Realtime raster shadow sistemi yok | Atlas, CSM, caster culling ve cache gerekli |
| Environment | Küçük baked studio/outdoor environment var | Gerçek world HDRI convolution ve exposure parity gerekli |
| Transparency | Opacity değeri ve alpha texture okunuyor | Sıralı/OIT blend, refraction ve temporal composite eksik |
| Transmission | GPU material ABI transmission/IOR/absorption verisini taşıyor | Raster preview bunu çözmüyor |
| Hair/particles | Ayrı viewport pipeline'ları var | Gerçek ışık, shadow receive/cast ve compositing parity eksik |
| Terrain/foliage | Raster mesh ve terrain material blend desteği var | Clustered light, shadow, wind temporal velocity eksik |
| Water/fluid | Raster geometri/particle temsil yolları mevcut | Thickness, refraction, absorption, foam ve shadow entegrasyonu eksik |
| VDB/gas/cloud/fire | Vulkan RT production volume yolu gelişmiş | Raster realtime volume raymarch/composite yolu yok |
| Material Graph | RT bytecode/evaluator Realtime surface tarafından da tüketiliyor | Realtime volume programı ve eksik raster graph girdileri gerekli |
| Otomasyon | Viewport shading/capture/probe Python ve IPC'de var; **sunum telemetrisi eklendi** (`rt.viewport.frame_telemetry` / `viewport.frame_telemetry`) | Realtime ayar ve capability yüzeyi yok; pass telemetrisi yalnız sunum yarısını kapsıyor |
| Frame sunumu | Raster draw instanced; mesh başına tek `vkCmdDraw*(instanceCount)` var | ~~Her değişen kare senkron submit -> tam GPU readback -> CPU kopya -> `SDL_UpdateTexture`~~ **Faz 0.5a ile kapatıldı** (iki slotlu ring, tek submit, bloke etmeyen tüketim). Kalan açık: hâlâ GPU -> CPU -> SDL yolu var; sıfır-kopya sunum 0.5b |
| Instance güncelleme | Mesh paylaşımı ve instance transform buffer var | ~~GPU_ONLY buffer güncellemesi geçici staging + blocking fence kullanıyor~~ **kaldırıldı**; kapasite geometrik büyüyor. Kalan açık: static/dynamic segment, dirty range, GPU culling + indirect draw, stable instance ID |

## Hedef frame graph

```text
Scene snapshot / revision gates
  -> animation, deformation and simulation sync
  -> shadow caster updates
  -> depth + velocity prepass
  -> G-buffer opaque/masked surfaces
  -> depth pyramid + clustered light lists
  -> directional/spot/point/area shadow passes
  -> deferred direct lighting + world IBL
  -> GTAO / screen-space indirect diffuse
  -> forward special surfaces
       -> hair and foliage
       -> transmission/refraction
       -> water/fluid surfaces
       -> particles
  -> realtime volume integration
       -> VDB / gas / cloud / fire
       -> geometry/volume mutual transmittance
  -> SSR / optional ray-query reflections
  -> temporal resolve
  -> bloom / atmosphere / tone map / color management
  -> overlays, selection and gizmos
```

Ana opaque/masked yüzeyler deferred; transmission, hair, particles, water ve
hacimler forward/composite çalışır. Bu hibrit ayrım hem çok ışık ölçeklenmesini
hem de özel materyal semantiklerini korur.

## Kalite ve maliyet profilleri

| Profil | Hedef | Shadow atlas | CSM | Shadowed lights | Screen-space | Volumes |
|---|---:|---:|---:|---:|---|---|
| Performance | 1080p / 60+ FPS | 2048 | 2 | 4 | half-res | half-res, düşük adım bütçesi |
| Balanced | 1080p / 60 FPS | 4096 | 4 | 8 | half-res + temporal | half-res + temporal |
| Quality | 1440p / 30-60 FPS | 4096 | 4 | 16 | full-res kritik pass'ler | adaptive, daha yüksek bütçe |
| Cinematic | viewport still/final preview | 8192 | 4 | 32 | yüksek kalite | yüksek kalite + ray query |
| Auto | ölçülen GPU zamanına göre profil/bütçe seçer | bounded | bounded | bounded | bounded | bounded |
| Custom | uzman kontrolü | doğrulanmış sınırlar | 1-4 | 0-64 | açık seçim | açık seçim |

Auto yalnız önceden doğrulanmış profil sınırları arasında hareket eder. Tek tek
ayarları gizlice sınırsızlaştırmaz. Manuel değişiklik profili `Custom` yapar.

### Realtime kalite ayarlari sozlesmesi

Profil tablosu yalniz isim degildir. Asagidaki kanonik ayarlar
`RealtimeRenderSettingsService` tarafindan dogrulanir, proje ile saklanir ve UI,
Python ile IPC'ye ayni isim/hata semantigiyle acilir:

- **Shadow:** atlas resolution, directional CSM cascade sayisi ve mesafeleri,
  cascade/tile resolution, shadowed-light butcesi, PCF/PCSS filtre kalitesi,
  constant/normal bias, alpha-cutout caster, static/dynamic cache yenileme ve
  istege bagli contact/ray-query shadow.
- **Raster:** internal resolution scale, dinamik resolution ve hedef frame
  suresi, MSAA/TAA, anisotropic filtering, texture/mesh LOD, frustum/occlusion
  culling, clustered-light butcesi ve transparency/OIT kalitesi.
- **Lighting/material:** world IBL convolution resolution/sample butcesi,
  reflection kalitesi, SSS sample/blur butcesi, thin/thick transmission,
  refraction ve absorption sinirlari.
- **Environment/volume/post:** Physical Sky LUT kalitesi ve guncelleme butcesi,
  AO/GI/reflection kalitesi, volume resolution/step butcesi, exposure/color
  management, temporal resolve ve bloom.

Her ayar `supported`, `active`, `approximate` ve `fallback_reason` durumunu
raporlar. Profil degisimi ilgili history/cache'i deterministik gecersiz kilar;
tek bir shadow veya atmosphere slider'i tum device'i `waitIdle` ile durdurmaz.
`Auto` hedef GPU frame suresine gore yalniz dogrulanmis bounded seviyeler
arasinda hareket eder. `Cinematic` pahali olabilir fakat Windows TDR sinirini
asabilecek tek-dispatch is uretmez; uzun compute isi parcalanir veya karelere
yayilir.

## Faz 0 — sözleşme, ayarlar ve ölçüm temeli

Durum: **in progress**

- [x] Mevcut raster, light, surface material ve volume yollarını haritala.
- [x] Realtime ürün ve frame-graph sözleşmesini sabitle.
- [x] Kalite profilleri ile bounded shadow/light/GI/transmission/volume ayar
      modelini bağımsız çekirdek serviste tanımla.
- [ ] GPU capability sorgusu: descriptor indexing, multiview, timestamps,
      subgroup, ray query ve memory budget.
- [~] Frame/pass telemetry veri modeli. **Sunum yarısı kuruldu ve script'e
      açıldı**: `Backend::RasterFrameTelemetry` + `rt.viewport.frame_telemetry`
      + `viewport.frame_telemetry` IPC + HUD satırı. Host zamanlamaları
      (record / slot wait / submit / host read / present), ring sayaçları ve
      `stale_presents` canlı. **Eksik kalan**: GPU timestamp query'leri,
      allocated bytes, active lights, shadow cache hits, volume steps ve
      history rejection — bunlar ölçtükleri pass'lerle birlikte gelir.
- [~] UI, Python ve IPC parity. **Telemetry üç yüzeyde de var** (panel HUD
      satırı `rt.viewport.frame_telemetry` ile *aynı* değerleri okur, ikinci
      bir otorite değildir). Settings ve capabilities parity'si gerçek render
      dikey dilimiyle, Faz 1'de bağlanır.
- [ ] Kaynak ömrü/fence sözleşmesi ve render-thread ownership belgesi.
- [ ] **FPS göstergesi** (raster modları): `RasterFrameTelemetry` içinde kayan
      pencere; panel kendi sayacını TUTMAZ; "kare gönderilmedi" `idle` olarak
      raporlanır, `0` olarak DEĞİL. Tasarım: *FPS göstergesi + katlanabilir
      Realtime kalite paneli* bölümü.
- [ ] **Katlanabilir kalite alanı** (viewport overlay): katlanma durumu UI-yerel,
      içindeki her kadran IPC'ye açık. Gölge atlas/tile çözünürlüğü bugün
      derleme-zamanı sabiti; runtime yapmak drenaj ister ve ışık bütçesini
      değiştirir. Aynı bölüm.

Kabul:

- Geçersiz cascade, atlas veya ışık bütçesi renderer'a ulaşmadan reddedilir.
- Aynı ayar isteği UI, Python ve IPC'de aynı sonuç/hata mesajını üretir.
- Realtime desteklenmeyen makinede sessizce Material Preview'a dönüşmez.

## Faz 1 — gerçek sahne direct lighting dikey dilimi

- Yeni `Realtime` viewport kimliği; Material Preview stüdyo modundan ayrılır.
- Ortak GPU light snapshot'ı; visible point/directional/spot/area ışıkları.
- Cook-Torrance GGX, energy conservation ve fiziksel attenuation.
- Depth/normal/albedo/material-id hedefleri ve ilk deferred lighting pass.
- Opaque flat mesh, instances, skinned/deformed mesh ve terrain.
- Light/material/transform revision invalidation.
- UI + `rt.viewport.realtime.*` + IPC tam parity.

Kabul:

- Işık ekleme, silme, taşıma, renk/intensity/type değişimi sonraki realtime
  frame'de görünür; Material Preview görünümü değişmez.
- Aynı sahnenin direct-light diffuse/specular eğilimi Vulkan RT referansıyla
  ölçülebilir tolerans içinde kalır.
- Sıfır ışık güvenli world/emission sonucu verir; sabit üç preview ışığı sızmaz.

## Faz 0.5 — asenkron viewport sunumu ve production instancing kapısı

Bu faz Faz 1'den önce kapanır. Bugünkü raster draw instancing kullanmasına rağmen
frame sonu bütünüyle seridir:

`record -> blocking submit/fence -> image-to-buffer -> blocking fence -> CPU copy -> SDL texture upload`

Bu yol basit instance sahnelerini dahi GPU/CPU paralelliğinden mahrum bırakır ve
Realtime pass sayısı arttığında daha kötü ölçeklenir.

### 0.5a — güvenli kısa vadeli köprü

- Yeni scheduler semantiği icat edilmez. Vulkan RT'nin mevcut
  `kFrameSlotCount=2`, kalıcı command buffer/fence, `submitSlot/consumeSlot` ve
  aynı slotu yeniden kullanmadan önce bekleme sözleşmesi raster/Realtime için
  referans implementasyondur. RT'ye özgü trace/tonemap kaydı paylaşılmaz; ortak
  frame-slot sahipliği odaklı bir modüle ayrılır.
- Frame başına command buffer allocate/free yerine 2-3 kalıcı frame context.
- Per-frame command buffer, fence, readback buffer ve persistently mapped pointer.
- N karesini submit ederken yalnız hazır N-1/N-2 slotunu SDL'ye yayınla; aktif
  frame fence'ini hemen bekleme.
- Render ve image-to-buffer copy aynı command buffer/submission içinde.
- Viewport capture kapalıyken ikinci bir ölçüm/readback kopyası oluşturma.
- Resize/mode switch sırasında yalnız ilgili frame-context fence'lerini drain et.
- Ayrı CPU record, GPU execute, readback wait, memcpy ve SDL upload telemetry.

Bu köprü GPU -> CPU -> SDL upload maliyetini tamamen kaldırmaz fakat seri stall'ı
kaldırır ve native sunuma geçerken doğruluk/fallback sağlar.

#### 0.5a durumu — YAZILDI, DERLENMEDİ (2026-08-31)

| Madde | Durum | Nerede |
|---|---|---|
| Kalıcı frame context (2 slot), command buffer + fence + readback + mapped ptr | ✔ | `Viewport/RasterViewportFrameRing.{h,cpp}` |
| RT `kFrameSlotCount=2` / `submitSlot`+`consumeSlot` sözleşmesinin referans alınması | ✔ | aynı dosya; RT trace/tonemap kaydı **paylaşılmadı** |
| Render + image-to-buffer copy tek command buffer, tek submit | ✔ | `submitFrame()` |
| N submit edilirken N-1/N-2 yayınlanır, aktif fence beklenmez | ✔ | `consumeNewestReady()` |
| Capture kapalıyken ikinci readback kopyası yok | ✔ | eski `copyImageToBuffer` yalnız fallback dalında |
| Resize/mode switch'te yalnız ilgili fence'ler drain edilir | ✔ | `ensure()` yeniden kurarken, `destroy...Impl()` reset ederken |
| Ayrı CPU record / GPU submit / readback wait / memcpy / present telemetry | ✔ | `RasterFrameTelemetry`, üç yüzeyde okunur |
| Sürücü kalıcı kaynakları reddederse deterministik eski yola düşme | ✔ | mandallanır (`m_rasterFrameRingUnavailable`), `async_present=false` diye **raporlanır** |
| Instance buffer'ın her düzenlemede destroy/recreate edilmesi | ✔ kaldırıldı | `VulkanBackend_Raster.cpp`: kapasite geometrik büyür, `CPU_TO_GPU` ile staging+blocking transfer gitti |

★★★ **Bu fazın en sinsi arıza modu ölçülür hale getirildi.** Halka bilerek bir
kare geriden yayın yapar. Eğer host *hiçbir zaman* tamamlanmış slot bulamazsa
görüntü kalıcı olarak eskir — ve bu durum ekran görüntüsünde **tamamen doğru
görünür**, statik sahnede hissedilmez, yalnızca kamera sürüklenirken bulanır.
Kimse bunu bug diye raporlamaz. `stale_presents` tam olarak bu sayıdır ve kare
sayısıyla birlikte büyümesi arızanın tanımıdır.

★★ **Asenkron sunum ile probe DOĞRUDAN ÇELİŞİR.** Bir script `render.probe`
çağırdığında halka kendi sahne düzenlemesinden ÖNCE kaydedilmiş bir kareyi
verebilir ve görüntüde bunu söyleyen hiçbir şey olmaz. Bu yüzden
`rt.viewport.capture(True)` sunumu senkronlar: gecikme belirliliğe takas edilir,
ve `synchronous_present` alanı hangi rejimde ölçüm yapıldığını söyler. Capture
açıkken alınan zamanlamalar **interaktif zamanlamalar değildir**.

★ Ölçüm scripti: `scripts/probe_viewport_frame_presentation.py`
(+ `x64/Release/scripts/`). Hız iddiası ölçmez — IPC her kare için frame
loop'a döndüğünden interaktif kadansı üretemez; köprünün **yapısal olarak**
asenkron, bloke etmeyen ve tüketen durumda olduğunu ölçer.

### 0.5b — nihai sıfır-kopya sunum

- Vulkan swapchain veya UI tarafından doğrudan örneklenebilir Vulkan viewport image.
- Raster/Realtime color hedefini CPU'ya indirmeden ekrana sun.
- Screenshot/probe/export readback'i yalnız açık istekle asenkron staging'e kopyala.
- UI overlay, selection ve gizmo compositing için tek GPU frame graph/senkronizasyon.
- SDL CPU texture yolu yalnız capability fallback olur ve durum yüzeyinde raporlanır.

### Production instancing

- Bir mesh'in vertex/index/UV/material verisi bir kez upload edilir.
- Transform, previous-transform, visibility, object/material override ve stable
  instance ID ayrı SoA/structured GPU buffer'da tutulur.
- Static ve dynamic instance segmentleri ayrılır; yalnız dirty range güncellenir.
- GPU culling + compact visible list + indirect draw; CPU her frame bütün instance
  matrislerini dolaşıp yeniden paketlemez.
- Shadow/G-buffer/selection/transmission pass'leri aynı stable instance ID ve
  visible-list sözleşmesini kullanır.
- Instance ekleme/silme kapasite geometrik büyür; tek değişiklik bütün buffer'ı
  destroy/recreate etmez. **(✔ yazıldı 2026-08-31)** `uploadRasterInstanceBuffer`
  artık buffer'ı yok edip yeniden kurmuyor; kapasite yetiyorsa yerinde yazıyor,
  yetmiyorsa 1.5× büyüyor. Bellek `GPU_ONLY`'den `CPU_TO_GPU`'ya alındı, çünkü
  `GPU_ONLY` her transform düzenlemesinde geçici staging buffer + bloke eden
  fence demekti. **Kalan**: static/dynamic segment ayrımı, dirty range, GPU
  culling + indirect draw, stable instance ID.
- Quality/Cinematic: frustum dışında olmayan hiçbir instance count/triangle budget
  nedeniyle düşürülmez ve proxy olmaz. Performance/Balanced açıkça raporlu LOD
  kullanabilir; nesne kimliği, transform ve materyal yine korunur.

Kabul:

- 100, 1K, 10K ve 100K aynı-mesh instance benchmark'ında CPU draw-call sayısı
  instance sayısıyla doğrusal büyümez.
- Statik kamera/sahnede instance upload byte sayısı sıfırdır.
- Tek instance hareketinde upload bütün grup kapasitesi değil yalnız bounded dirty
  range/ring segmentidir.
- Quality profilinde `visible canonical instances == submitted full instances +
  correctly culled instances`; budget-dropped/proxy instance sayısı sıfırdır.
- Normal görüntüleme sırasında senkron tam-frame GPU readback yoktur.

## Faz 2 — üretim gölgeleri

- Directional için stable 4-cascade CSM ve texel snapping.
- Spot 2D ve point cubemap shadow.
- Tek atlas allocator, önem tabanlı shadow budget ve cache.
- Alpha-cutout caster; double-sided ve normal-bias semantiği.
- Dynamic/static caster revision ayrımı.
- Area light için önce bounded PCSS yaklaşımı; ray-query varsa kaliteli seçenek.

Kabul:

- Kamera hareketinde cascade yüzmesi belirlenen piksel toleransını aşmaz.
- Deformasyon/simulation caster'ları stale shadow bırakmaz.
- Atlas doluluğu bir ışığı rastgele düşürmez; seçim deterministik ve raporludur.

## Faz 3 — environment, AO, temporal ve post

- HDRI diffuse irradiance ve prefiltered specular IBL.
- BRDF integration LUT ve world rotation/intensity/exposure parity.
- GTAO, bent normal ve bilateral upsample.
- Motion vectors, TAA, disocclusion/history rejection.
- Bloom, atmosphere composite, tone mapping ve mevcut color processing parity.
- Dynamic resolution ve ölçülen GPU-time Auto profili.

## Faz 4 — bütün surface material yapısı

- Material Graph realtime surface programı: constant folding, bounded register ve
  instruction budget; unsupported node diagnostics.
- Normal/bump/height, anisotropy, sheen, clearcoat ve iridescence.
- Screen-space/diffusion-profile SSS; random-walk yalnız path-traced referans olur.
- Emissive surface ışık katkısı: önce screen-space/probe injection, sonra ray query.
- Terrain layer ve semantic overlays; foliage two-sided/translucency.
- Hair BSDF, receive/cast shadow ve motion vectors.

Her node/closure `exact`, `bounded approximation` veya `unsupported` olarak
capability çıktısında sınıflandırılır. Yanlış ama sessiz fallback kabul edilmez.

## Faz 5 — opacity ve transmission

- Masked alpha depth/G-buffer/shadow yolu.
- Blend materyaller için weighted blended OIT; gerektiğinde sıralı özel pass.
- Thin transmission: screen-space refraction + thickness approximation.
- Closed-mesh thick transmission: front/back depth thickness, Beer-Lambert
  absorption, IOR/Fresnel ve rough refraction.
- Screen dışında kalan ışınlar için probe/environment fallback.
- Opsiyonel ray-query ile off-screen refraction/reflection ve kaliteli gölge.
- Bubble/thin-film, resin/interior inclusions ve dispersion için bounded kalite
  katmanları; path-traced referansla açıkça ayrılmış approximation etiketi.

2026-09-03 surface-contract durumu: texture/Material Graph transmission ve IOR,
dielectric Fresnel/TIR, Beer extinction, RT ile aynı spectral dispersion spread'i,
rough transmission cone integrasyonu, bubble film ve resin dust/shard stil
semantiği Material Preview shader'ına ulaşmıştır. Opak sahne renk/depth snapshot'i,
ayrı transmissive replay ve atomik back-depth ile kapalı mesh kalınlığı tamamlanmış,
capability `screen_space_thickness` olarak açılmıştır. Base Color/texture depthless
camın Beer tint'idir; Interior Depth açıldığında RT'deki gibi Interior Color devralır.
Off-screen continuation ve örtüşen çoklu cam katmanları bounded fallback'tir.

2026-09-03 glass reflection kalibrasyonu: opak snapshot depth'i uzerinde bounded
quadratic ray march, gorunen sahne ayrintisini cam yansimasina tasir. Hit yoksa
Physical Sky/HDRI environment reflection devam eder. Depthless cam Base Color
soğurumu, RT'nin giriş+cikis 0.65 yoluna karsilik tek replay'de minimum 1.30
optik yol kullanir; kapali mesh daha kalinsa olculen kalinlik devralir.

2026-09-03 distant-glass stabilization: yansima lobu ekran-uzayi normal
turevlerinden ucuz bir specular-AA footprint alir. Authored Roughness 0 yakin
planda keskin kalir, fakat kamera uzaklastiginda GGX tepesi piksel altina dusup
kaybolmaz. Interior Depth yuzeyi IOR Fresnel + Coat Glossy roughness kullanir;
ayri resin environment terimi kaldirilarak coat yansimasi iki kez eklenmez.

Kabul:

- Transmission materyal opak görünmez ve eksik ekran verisinde siyaha düşmez.
- Kapalı mesh kalınlığı ölçekle tutarlı absorption üretir.
- Transparent nesne selection/object-id ve temporal resolve'u bozmaz.

## Faz 6 — water, liquid, particles ve özel yüzeyler

- FFT water normal/displacement/foam realtime parity.
- Fluid SDF surface: thickness, refraction, absorption, foam ve velocity.
- Particle spheres/billboards, emissive particles ve soft-depth intersection.
- Mutual shadowing: geometry -> liquid/particle ve desteklenen ters yön.
- Simulation render bridge revision ve temporal discontinuity sinyalleri.

## Faz 7 — VDB, gas, smoke, fire, cloud ve mesh volumes

- Üretim simulation domain GPU field'larını kopyasız tüketen realtime volume pass.
- VDB/NanoVDB sparse traversal veya ekran-uzayı froxel entegrasyonu.
- Density, temperature, flame/fuel, soot, emission, anisotropy ve velocity.
- Directional/local-light volume shadowing ve deep sun transmittance cache.
- Geometry-volume mutual transmittance ve camera-inside doğruluğu.
- Volume Material Graph'ın aynı compiled program/field semantiğini tüketmesi.
- Half-resolution temporal volume resolve; velocity-aware reprojection.
- Mesh volume boundary ve SDF surface+volume ortak materyal desteği.

Realtime volume kalite bütçesi, production RT volume `max_steps` değerini körlemesine
kopyalamaz. Görüntü çözünürlüğü, voxel footprint, optical depth ve GPU-time
bütçesinden bounded bir realtime örnekleme planı üretir.

Kabul:

- Fire emission temperature/flame alanı varken density fallback kullanmaz.
- Smoke geometry arkasında/önünde doğru compositing ve shadow alır/verir.
- Boş sparse bölgelerin maliyeti dolu voxel örnek sayısıyla ölçeklenmez.
- Camera-inside ve anisotropic transform referans sahneleri siyah bant üretmez.

## Faz 8 — Lumen yönü: indirect lighting ve reflections

- Depth pyramid tabanlı SSR ve temporal denoise.
- Screen-space diffuse GI (SSGI) ve emissive injection.
- Ekran dışı kapsama için world-space probe grid/DDGI.
- Probe relocation/classification, scrolling clipmap ve bounded update budget.
- Static/dynamic sky visibility.
- Vulkan ray query destekli hybrid reflection, shadow ve probe validation.
- Donanım RT yoksa deterministic screen-space + probe fallback.

Bu fazın hedefi Lumen adını taklit etmek değil; dinamik sahnede ölçülebilir,
bütçeli ve fallback'i açık bir indirect-lighting sistemi üretmektir.

## Faz 9 — üretim kapatma ve parity

- RenderDoc/GPU marker, timestamp ve VRAM instrumentation.
- Shader/pipeline cache ve background-safe rebuild.
- Device lost, resize, mode switch ve project reload ömür testleri.
- Realtime/path-traced A/B referans sahneleri ve image metrics.
- Serialization/version migration.
- UI, Python, IPC dokümantasyonu ve regression test matrisi.
- Performance, Balanced, Quality ve Cinematic kapılarının ayrı onayı.

## Referans sahne matrisi

- Cornell direct-light: dört ışık türü, hard/soft shadow.
- Metallic/roughness/clearcoat/sheens grid.
- Alpha foliage ve layered terrain.
- Thin glass, closed thick glass, colored absorption, bubble ve rough glass.
- Skin/wax SSS, anisotropic metal ve hair cards/strands.
- FFT ocean ve SDF liquid pouring scene.
- Emissive particles ve dense particle occlusion.
- Sparse smoke plume, dense fireball, cloud with sun/local lights.
- Camera inside VDB ve closed mesh volume.
- Fast animated/deformed caster, camera cut ve long temporal playback.
- Screen-space edge/off-screen fallback scene.

Her sahne en az şu değerleri raporlar: frame GPU ms, pass GPU ms, VRAM, visible
ve shadowed light sayısı, shadow-cache hit oranı, cluster overflow, transparent
fragment yükü, volume samples/early-outs, history rejection ve fallback nedenleri.

## “Bitti” tanımı

Realtime renderer yalnız opaque mesh ışıklandırınca bitmiş sayılmaz. Nihai kabul:

1. Desteklenen bütün sahne öğeleri görünür, doğru depth/composite alır ve seçim/
   gizmo iş akışını bozmaz.
2. Surface, transmission ve volume material semantiği Path Traced referansla aynı
   girdileri kullanır; yaklaşım yapılan yerler açıkça raporludur.
3. UI, Python ve IPC aynı operasyon/ayar/capability/telemetry yüzeyine sahiptir.
4. Performance ve kalite bütçeleri gerçek ölçümle doğrulanmıştır.
5. Mod değişimi, timeline playback, simulation, project reload ve resize kaynak
   ömrü hatası, stale frame veya kontrolsüz stall üretmez.
