# Iki foliage asset'i ayni dokuyu paylasiyordu

> **Durum:** REFERANS — 2026-08-30. Kok neden bulundu ve duzeltildi; canli
> dogrulama build sonrasi `docs/dev/NEXT_BUILD_CHECKS.md` uzerinden.

## Belirti (kullanici raporu)

- Sahnede kati obje yokken terrain eklenip **Asset Library**'den cam gibi
  **cok materyalli** bir agac foliage layer'a eklenince **materyal siralamasi
  bozuluyor**.
- Bir cam **ve** bir bauhinia eklenince **cam etkileniyor**; bauhinia
  eklenmezse sorun yok.
- Terrain'den **once** sahnede bir geometri varsa gene sorun yok.
- Vulkan RT, **OptiX ve CPU** — ucunde de ayni.

Son madde teshisi tek basina bitiriyor: ★★★ **uc backend'in ortak noktasi
backend degil, import.** Belirti "render yanlis" diye okunuyordu ve bir parti
Vulkan'da (geometries SSBO, BLAS meshKey ABA, material buffer buyumesi,
`refreshVulkanGeometryDataBinding`'in bos `m_blasList` dali) bosa arandi.
Hepsi temiz cikti — orada degildi.

## Kok neden

`FoliageAssets::loadScatterSource` **butun kutuphane icin tek bir import adi**
kullaniyordu:

```cpp
loader.loadModelToTriangles(fullPath.string(), nullptr, "foliage_asset", true);
```

`AssimpLoader` bu addan iki **kuresel** anahtar tureti:

| Anahtar | Nerede |
|---|---|
| Materyal kayit adi | `currentImportName + "_" + materyal adi` → `MaterialManager` |
| Gomulu doku anahtari | `"embedded_" + currentImportName + "_" + <aiTexture*> + "_" + tip` → **process genelinde `TextureCache`** |

Ikincisi olumcul. `Texture`'in gomulu ctor'u bu anahtarla `TextureCache`'e
bakar ve **isabet halinde decode'u tamamen atlar**, onbellekteki metadata'yi
alip `m_is_loaded = false` ile doner. Yani carpisma "yanlis renk" degil,
**digerinin pikselleri** demek.

Anahtarin tek ayirt edici parcasi ham `aiTexture*` adresi. O adres, ilk
asset'in `aiScene`'i yikilinca serbest kalir ve **bir sonraki import'a ayni
adres geri verilir**. Kutuphanenin tamami ayni on eki paylastigi icin ikinci
agacin gomulu dokulari birincinin anahtarlarina oturuyordu.

★★★ Kodun kendi yorumu bu tuzagi zaten yaziyordu — *"Adding currentImportName
is CRITICAL to prevent collisions if memory addresses are reused across imports
(ABA Problem)"*. Foliage yolu tam da o korumayi, sabit bir literal gecerek
**etkisiz birakmisti.** Koruma vardi; kapsami disinda kullanilmisti.

Kullanicinin uc gozlemi de bundan cikiyor: tek asset → carpisacak kayit yok;
iki asset → adres yeniden kullanimi; **once baska bir mesh import edilirse**
heap durumu degisir, adresler ust uste gelmez, belirti kaybolur. "Sinsi"
olmasinin sebebi bu — arizanin gorunurlugu **allocator'a** bagli.

## Ikinci kok: onbellege alinmis geometri, olu materyal ID'leri

`FoliageAssets` yuklenen ucgenleri dosya yoluna gore onbellege aliyor. Ucgen
materyali **ID** ile tutar. `MaterialManager::clear()` — New Project, ve
`deserialize()` uzerinden **her proje acilisi** — butun ID'leri emekliye ayirir
ama bu onbellek hayatta kalir. Sonraki isabet, artik o slotlari kim isgal
ediyorsa (genelde terrain katmanlari) ona isaret eden geometri veriyordu.
★★ Bayat ID hala **canli bir tabloyu** indeksledigi icin hicbir belirti yok.

`MaterialManager::generation()` eklendi (`clear()` artirir); onbellek kaydi
hangi kusakta cozuldugunu tutar, uyusmazlikta yeniden yukler.

## Ucuncu kok (ayri ariza): Assets listesi kayboluyor

"Diger layer bosaltilinca assets tekrar listelenmiyor, sadece scene objeleri
eklenebiliyor."

`captureFoliageGroupSettings` her karede `layer.useAssetLibrary`'yi **gruptaki
kaynaklardan olcerek** yeniden yaziyordu. Yani bir **olcum** ("bu grup su anda
asset kaynagi tutuyor mu"), bir **kullanici kararini** ("bu layer'i Asset
Library'den yaz") eziyordu. Layer bosalinca mod bir sonraki karede Scene
Layer'a donuyor ve asset secicisini de goturuyordu — asset'i geri koymanin tek
yolu o seciciydi. **Kilitlenme.**

Cikarim artik yalnizca **benimseme** aninda yapiliyor (`adoptSourceMode`):
node bir gruba ilk kez baglanirken. Kare basina aynalama modu ellemiyor.

## Dorduncu kok: "cache" piksel tutmuyordu (ilk duzeltmenin ACIGA CIKARDIGI)

Ilk duzeltmeden sonra ilk olusturma temizdi, ama **ayni oturumda New Project
yapip ayni agaci tekrar koyunca foliage default materyali aliyordu.**

`TextureCache` bir onbellek degildi. Yalnizca metadata tutuyordu
(width/height/has_alpha/is_gray_scale) — **piksel yok**. Gomulu ctor isabet
alinca:

```cpp
m_is_loaded = false;   // "zaten cache'de, decode gerekmez"
return;                // pixels BOS
```

Cagiran taraf ise `if (!texture || !texture->is_loaded()) continue;` diyor.
Yani "isabet" = **dokusuz materyal** = default gorunum.

★★★ Bu yol yillarca calisti cunku anahtari bir heap adresi tasiyordu ve
**neredeyse hic isabet etmiyordu.** Anahtari deterministik yapmak (asset basina
on ek + aiScene indeksi) onu ilk kez guvenilir sekilde isabet ettirdi ve
kabugu goz onune cikardi. ★★ Duzeltme bir arizayi uretmedi; **var olan bir
arizayi gorunur kildi** — eski davranis dogru degildi, sadece sansliydi.

Karsilastirma ayni dosyada duruyor: **dosya** dokusu icin cache isabeti hala
pikselleri yukluyor (`IMG_Load`), onbellegi yalnizca bir probe'u atlamak icin
kullaniyor. O gercek bir optimizasyon. Gomulu olan ise isi yalnizca **sonucu
atarak** atliyordu.

Metadata kisa devresi ve arkasindaki `TextureCache` sinifi **kaldirildi**
(kural 5: dogrulanip olu oldugu anlasilan yol sokulur). `FileTextureCache`
yerinde.

★★ **Ders:** "cache hit" yazan bir yol, isabette **cagiranin bekledigi seyi**
uretmek zorundadir. Metadata donup veriyi atlayan bir kisa devre, bir
optimizasyon degil sessiz bir veri kaybidir.

## Olcum

`scripts/probe_foliage_asset_material_identity.py` (+ `x64/Release/scripts/`).
Iki asset yukler ve **hicbir doku kimliginin iki asset'e birden ait olmadigini**
dogrular. Bunun olculebilmesi icin `material.textures` genisletildi: artik
`{slot, texture}` donuyor — sadece "bu slot dolu" demek bu ariza sinifina
**yapisal olarak kordu**.

## Degisen dosyalar

- `Scene/FoliageAssetLibrary.cpp` — asset basina import on eki; kusak kontrollu
  geometri onbellegi
- `AssimpLoader.h` — `makeEmbeddedTextureKey()`: adres degil **aiScene indeksi**;
  iki anahtar sitesi de buradan gecer; olu `TextureCache` temizligi kaldirildi
- `Texture.h` — ★★★ gomulu doku ctor'undaki **piksel tutmayan metadata kisa
  devresi** ve arkasindaki `TextureCache` sinifi kaldirildi
- `MaterialManager.{h,cpp}` — `generation()`, `clear()` artirir
- `Physics/TerrainNodesV2.cpp` — `adoptSourceMode`; mod artik demote edilmiyor
- `Api/RtApi{.h,Material.cpp}`, `Api/RtIpc.cpp`, `Api/RtPyScene.cpp` —
  `material.textures` doku kimligi donuyor
- `Api/RtApi.cpp` — script'ten eklenen scatter kaynagi artik asset adini aliyor
  ("API_Library_Asset" hepsinde ayniydi)
