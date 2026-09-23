# Kamera odak kilidi: pivot ile lens odak düzleminin ayrılması

> **Durum:** AKTİF — 2026-09-23. Derlendi ve elde denendi: **çekirdek ayrım çalışıyor**
> (Numpad `.` merkeze alıyor, kilitli orbit doğru merkez etrafında dönüyor, pan çalışıyor).
> **İKİ AÇIK ARIZA VAR** — bkz. en alttaki "Açık arızalar" bölümü. 2026-09-24'te ele alınacak.

## Belirti

İki ayrı şikâyet, tek kök:

1. Numpad `.` ile obje merkeze alınıyor, ama ilk orta-tık sürüklemesinde obje
   ekrandan kayıyordu — çerçeveleme "tutmuyor" gibi hissettiriyordu.
2. **Bazen pan neredeyse imkânsızlaşıyordu.** Kamera hareket ediyor ama sanki
   hiç ilerlemiyordu.

İkincisinin hata mesajı, log satırı ve ekran görüntüsünde izi yok. Kimse bunu
bug diye raporlamaz; "kamera bir tuhaf" denir ve geçilir.

## Kök neden

Orbit pivotu diye bir alan yoktu. `Camera::lookat` o rolü oynuyordu ve **üç
yazarı** vardı, birbirlerinden habersiz:

| Yazar | Ne yazıyordu |
|---|---|
| `frame_selected()` (Numpad .) | seçimin bbox merkezi |
| orta-tık ışın taraması | imlecin altındaki yüzey |
| `Camera::setLookDirection()` | `lookfrom + dir * focus_dist` |

Üçüncüsü pahalı olanı:

```cpp
// ÖNCE
lookat = lookfrom + direction_normalized * focus_dist;
```

`focus_dist` bir **lens** özelliğidir — otofokus ve odak halkası yazar. Bu satır
yüzünden **her fare rotasyonu** navigasyon hedefini lensin odak düzlemine
taşıyordu. Odağı 0,5 m'ye çekmek orbit yarıçapını 0,5 m'ye düşürüyordu; pan hızı
(`pan_per_pixel ∝ nav_depth`) ve dolly adımı (`d * exp(x)`) bu yarıçapla
orantılı olduğu için viewport cevap vermez hâle geliyordu.

★ Sistemin yarısı zaten doğru çalışıyordu: tekerlek ve Ctrl-sürükleme
`lookat`'a doğru dolly yapıyordu, numpad 4/6/8/2 gerçek orbit'ti. **Kilitli
olmayan tek hareket fare rotasyonuydu** — ve o, her basışta diğerlerinin
dayandığı çapayı siliyordu.

İkinci kuplaj: dolly her adımda `focus_dist = new_distance` yazıyordu, yani
zoom sinematik odağı sessizce sürüklüyordu.

## Karar

**Pivot ile odak düzlemi ayrı alanlar.** `Camera::orbit_pivot` + `pivot_mode`
(`Free` | `Selection`) + `navDistance()`. Navigasyon yolunda **hiçbir yer artık
`focus_dist` yazmıyor**; otofokus onun tek yazarı.

### ★★★★★ Seçim modunda pivot SAKLANAN değil TÜRETİLEN bir değerdir

Her hareketin başında seçimden yeniden hesaplanır
(`rtapi::refreshPivotFromSelection`, `RtApiCameraNav.cpp`). Gerekçe bu partinin
en önemli cümlesi:

> Saklanan pivot bayatlar. Bayat pivot, bu işin kaldırdığı arızanın ta kendisi.

Türetilmiş pivot **yapısal olarak** bayatlayamaz. Yan kazanç: animasyonlu
objeyi bedavaya takip eder.

Seçim yokken pivot **düşürülür ama mod korunur** — "varsayılan bir ölçüm
değildir": son objenin merkezini tutmak sessizce bayat bir çapa üretirdi.

### Hareketlerin yeni sözleşmesi

| Hareket | Çapa | `focus_dist` yazar mı |
|---|---|---|
| orbit (fare/numpad/IPC) | pivot, `lookat` pivota nişan alır | hayır |
| dolly (tekerlek/Ctrl/IPC) | pivot; ortografikte `ortho_height` | **hayır** (eskiden yazardı) |
| pan | pivot birlikte kayar | hayır |
| Frame Selected | pivotu kurar + kilidi armalar | **hayır** (eskiden yazardı) |
| orta-tık ışını | serbest modda `lookat`, kilitli modda hiçbir şey | AF açıksa evet, **yalnız o** |

Kilitli modda fare rotasyonu **orbit eder**, fly-look yapmaz. Serbest modda
fly-look korunur ama artık `navDistance()` kullanır, `focus_dist` değil.

## Ölçü aleti

★★★ Bu arızanın tek görünür hâli bir **sayı**dır. `camera.get_pivot`
`nav_distance` ve `focus_distance`'ı **yan yana** döndürür; kuplajın geri
gelmediğinin kanıtı budur. Panel de ikisini birlikte yazar
(`Nav radius … | Focus …`).

Doğrulama: `scripts/ipc/Probe-CameraPivotLock.ps1` — kapı 2 bu betiğin kalbi.

## Dokunulan dosyalar

- `source/include/Camera.h`, `source/src/Render/Camera.cpp` — pivot alanları,
  `navDistance`, `orbitAroundPivot`, `dollyToPivot`, `panWorld`; `setLookDirection`
  artık `focus_dist` okumuyor.
- `source/src/Api/RtApiCameraNav.cpp` **(yeni)** — tek gövde: pivot, orbit, pan,
  dolly, Frame Selected.
- `source/include/Api/RtApi.h`, `RtIpc.cpp`, `RtPython.cpp` — `camera.get_pivot`,
  `set_pivot_mode`, `orbit`, `dolly`, `pan`, `frame_selected`.
- `source/src/Core/Main.cpp` — fare/numpad yolları bu gövdeye bağlandı; Rodrigues
  kopyası kaldırıldı.
- `source/src/UI/scene_ui_hierarchy.cpp` — panel kontrolü + iki mesafenin okuması.

★ Panel kontrolü `scene_ui_camera.cpp`'ye **konmadı**: o dosyanın gövdesi
(26–1064) komple yorum bloğu içinde, canlı kamera paneli
`scene_ui_hierarchy.cpp`'de. Bkz. `project_scene_ui_camera_cpp_is_commented_out`.

## Açık kalan

- `pivot_mode` projeye **serialize edilmiyor** (oturum tercihi). Panel bu yüzden
  `markModified` çağırmıyor.
- Kilitli orbit'in yaw/pitch **işaret yönü** zevk meselesi; runtime'da elde
  denenmeli (kontrol listesi madde 6).
- ViewCube'ün kendi serbest-orbit gövdesi (`scene_ui_viewport.cpp` ~754) hâlâ
  ayrı — `orbitAroundPivot`'a taşınabilir, bu partide dokunulmadı.

---

## Açık arızalar (2026-09-23 el testinden)

### ✔ 1. ÇÖZÜLDÜ (karar): pan seçim kilidini BIRAKIR

**Belirtiydi:** Numpad `.` → pan → orta tuşla döndür → kamera objeye geri
ışınlanıyor.

**Kök neden:** seçim modunda pivot *türetilmiş* bir değer;
`refreshPivotFromSelection()` her hareketin başında onu objeye geri çakıyordu,
yani `panWorld`'ün taşıdığı offset her seferinde çöpe gidiyordu. Ardından
`orbitAroundPivot`'un `lookat = pivot` satırı bakışı objeye ışınlıyordu.

**Karar (kullanıcı, 2026-09-23):** pan kilidi **kırsın** — Blender de böyle
davranıyor. İlk tasarımda "birikimli pan offset'i" planlanmıştı; o yol terk
edildi.

★★★ Gerekçesi muhasebeden daha derin: **bir objeden UZAĞA pan yapmak ile ona
KİLİTLİ kalmak çelişkili iki niyettir.** Offset defteri tutmak, çelişkiyi
çözmek yerine etrafından dolaşmak olurdu.

`Camera::panWorld` artık `pivot_mode = Free` + `clearOrbitPivot()` yapıyor.
★★ Çapa kaybolmuyor: `effectivePivot()` `lookat`'e düşüyor ve o da pan ile
birlikte taşındı — orbit/dolly pan edilen noktanın etrafında çalışmaya devam
ediyor, aritmetik birebir aynı. Kesilen tek şey **seçime olan bağ**.

★ Çapanın *armed* bırakılıp yalnızca modun `Free`ye düşürülmesi denendi ve
**reddedildi**: panel "Free" yazarken imleç çapalaması sessizce ölü kalıyordu.
Panelin motorda olmayan bir durumu raporlaması bu deponun en pahalı hata sınıfı.

⚠ **Elde denenmedi.** Ayrıca bir yan etki var ve bilinçli: kilit bırakıldığı için
pan'dan sonra orta tuş rotasyonu **fly-look'a dönüyor** (serbest modun bu
uygulamadaki ezelî davranışı), orbit'e değil. Blender'da pan'dan sonra da orbit
edilir. Bu his yanlış gelirse düzeltme tek satır — bkz. kontrol listesi.

### ★★★★★ 2. ÇÖZÜLDÜ: bayrağı SERVİS ETMEYEN blok onu TEMİZLİYORDU

**Belirti:** realtime raster (solid) modunda fareyle pan ve zoom viewport'u
yenilemiyor; **orta tuşla rotate yeniliyor**; RT modda sorun yok; solid modda
yapılan pan RT'ye geçince **doğru konumda** görünüyor.

#### ★★★★ Bu bir REGRESYON ve sebebi bu partide yazdığım koddu

`Camera::panWorld`, `dollyToPivot`, `orbitAroundPivot` — üçünü de ben yazdım ve
üçü de `markDirty()` çağırıyor. Öncesinde pan/dolly yalnızca
`update_camera_vectors()` çağırıyordu, `markDirty()` **çağırmıyordu**.

#### Kök

`Main.cpp:4535` civarındaki blok:

```cpp
bool is_dirty = scene.camera->checkDirty();        // Camera::is_dirty'yi TÜKETİR
if (is_dirty || is_shaking || is_af_c) {
    ...
    ui_ctx.renderer.syncCameraToBackend(*scene.camera);  // yalnız RENDER backend
    ray_renderer.resetCPUAccumulation();
    g_camera_dirty = false;                        // ★ BAŞKASININ BAYRAĞI
}
```

Blok `Camera::is_dirty`'yi tüketiyor ve yalnızca **render** backend'ini
senkronluyor. Ama kapanışta `g_camera_dirty`'yi temizliyor — oysa o bayrağın
tüketicisi çok daha aşağıdaki (`~5197`) **viewport** senkronu. Raster viewport
kendi cihazında yaşıyor ve o senkronu hiç alamıyor.

#### ★★★ Rotate'in neden çalıştığı — teşhisin tamamı buydu

| Gestür | Gövde | `markDirty()` | Sonuç |
|---|---|---|---|
| serbest rotate | `setLookDirection` | **hayır** | 4535 bloğuna girmez → bayrak 5197'ye ulaşır → **çalışır** |
| pan | `panWorld` | evet | blok bayrağı yutar → viewport senkronu yok → **donuk** |
| dolly/zoom | `dollyToPivot` | evet | aynı | **donuk** |

"Yalnızca rotate çalışıyor" cümlesi teşhisin kendisiydi ve kamerayı değil,
**bayrağın kimin elinde öldüğünü** gösteriyordu.

#### Düzeltme

`g_camera_dirty = false` satırı o bloktan kaldırıldı.

> **Bir tüketici yalnızca KENDİ servis ettiği bayrağı temizleyebilir.**
> Başkasının bayrağını temizlemek, işi yapmaktan ayırt edilemez ve sessizce
> başarısız olur. Bırakmanın maliyeti en fazla bir gereksiz senkron.

★ `markDirty()` çağrıları **yerinde bırakıldı**: kamera gerçekten değişiyor ve
bunu bildirmek doğru. Yanlış olan bildirimi değil, bildirimi yutan taraftı.

#### ★★★★ Aynı kökün ESKİ ikizi: panelin FOV slider'ı (ve 16 kardeşi)

**Belirti (kullanıcının "eski sorun" dediği):** kamera panelindeki FOV slider'ı
raster modda sahneyi güncellemiyor, RT modda çalışıyor.

**Kök — aynı devrin ters yönü.** Panel kamerayı düzenlerken yalnızca
`cam.markDirty()` çağırıyor, `g_camera_dirty`'ye **hiç dokunmuyor**
(`scene_ui_hierarchy.cpp`'de `g_camera_dirty` geçmiyor, 0 kez). Yani:

- 4535 bloğu `checkDirty()` ile uyanır → **render** backend senkronlanır → RT ✔
- `g_camera_dirty` hiç set edilmediği için ~5197 koşmaz → **viewport** ✘

**Ölçüm (düzeltmeden önceki binary, solid mod, `viewport.capture` AÇIK):**
IPC `camera.set_fov` raster'ı **günceller** (çünkü `cameraChanged` bayrağı yazar),
panel slider'ı güncellemez. Aynı değer, iki yol, iki sonuç — fark tam olarak
bayrak.

**Düzeltme:** 4535 bloğu artık `g_camera_dirty = false` yerine
`g_camera_dirty = true` yazıyor: *"render backend'ini servis ettim, viewport
hâlâ bekliyor."* Panelin ~17 çağrı yerinin hepsi tek satırla kapanıyor —
her birine ayrı bayrak yazmak, on sekizincisinde unutulacak kopyadır.

★ Gereksiz iş değil: `checkDirty()` kendini temizler, yani gerçek değişim başına
bir kez ateşler. Shake/AF-C açıkken her kare ateşler ve orada kamera gerçekten
her kare değişir.

#### ⚠ Ölçü aletiyle ilgili ders (bu oturumda İKİ kez)

1. Vec3'ler IPC'den **dizi** döner; `.x` ile okumak sessizce boş üretir ve
   "hiç kımıldamadı" raporlar.
2. `viewport.get_screenshot` **`viewport.capture` açık değilse** önbellekteki
   kareyi verir → hash hiç değişmez → **her ölçüm False okunur.** İki ölçümü
   böyle kaybettim ve neredeyse doğru teşhisi çürütecektim.

> **Ölçmeye başlamadan aletin AÇIK olduğunu doğrula.** Susan bir alet
> "değişiklik yok" der, ve bu bir ölçüm gibi görünür.

#### Elenen hipotezler (dördü de ölçümle)

1. Tek bayrak / iki VkDevice + nesil sayacı — *şekil* doğruydu ama yanlış
   temizleyiciye bakıyordum (2292 sanmıştım, 4550'ymiş).
2. `scene.initialized` render döngüsünü kapatıyor — File→New ile çürüdü.
3. Fare yolunda eksik senkron — `camera.set_position` (viewport'a dokunmaz)
   solid modda kareyi güncelledi.
4. Default kamera kayıtlı değil — `camera.list`: `active_is_orphan=false`,
   `count=1`, indeks geçerli. **Ölçü aleti soruyu tek çağrıda kapattı.**

★★ Dördünün de ortak hatası: ayırt edici koşulu (rotate vs pan) açıklayamayan
bir hipotezi kabul etmek. **Bir hipotez, belirtinin NEDEN seçici olduğunu
açıklamıyorsa daha bakılmamıştır.**

### ★★★★★ 3. Orta tuş BAZEN alâkasız bir noktaya zıplatıyor — kök neden OKUMAYLA BULUNDU

**Belirti:** Orta tuşa basınca kamera arada bir hiç ilgisiz bir yere çapalanıyor.
**Aralıklı** — ve "bazen" kelimesi burada teşhisin kendisi: rastgelelik işareti.

**Kök neden (tahmin değil, iki dosya karşılaştırmasıyla):** `Main.cpp`'deki
navigasyon ışını hâlâ **`get_ray`** kullanıyor:

```cpp
Ray r = scene.camera->get_ray(u, v);     // Main.cpp ~3583
```

`get_ray` **render** ışınıdır: lens distorsiyonunu uygular **ve diyafram diskini
RASTGELE örnekler** —

```cpp
rd = random_in_unit_polygon(blade_count) * lens_r;
Vec3 offset = u * rd.x + v * rd.y;       // ışının ORİJİNİ kayıyor
```

Yani DoF açıkken her orta tıkta ışın **rastgele kaydırılmış bir noktadan**
çıkıyor ve başka bir yüzeye çarpıyor. Aralıklı olmasının sebebi bu; diyafram ne
kadar açıksa sapma o kadar büyük. DoF kapalıyken bile barrel/pincushion
distorsiyonu kare kenarlarında sistematik bir kayma bırakıyor.

★★★ **Bu bir kaçak değil, tek istisna:** depodaki diğer BÜTÜN seçici
`get_viewport_ray` kullanıyor (`scene_ui.cpp` ×2, `scene_ui_selection.cpp`,
`RtApi.cpp`). `scene_ui_selection.cpp:746` gerektiği yorumu zaten yazmış:

> "Raster draws a deterministic pinhole/ortho projection. Do not let render-only
> lens distortion or a random aperture sample move the selection ray away from
> the visible pixel."

Navigasyon ışını o dersi hiç devralmamış. u,v hesabı doğru (aynı tam-pencere
konvansiyonu), fark **yalnızca hangi ışın fonksiyonu** çağrıldığı.

★ Bu arıza **bu partiden önce de vardı** — eski kod da `get_ray` ile `lookat`
yazıyordu. Pivot ayrımı onu ortaya çıkardı, üretmedi.

**Düzeltme:** tek satır — `get_ray` → `get_viewport_ray`. Ama dokunmadan önce
kapı: DoF'u aç, diyaframı sonuna kadar aç, aynı piksele arka arkaya 5 kez orta
tıkla ve `camera.get_pivot` `nav_distance`'ını oku. **Beşi de aynı çıkmalı.**
Önce bu ölçümü al, yoksa düzelttiğini neye dayanarak söyleyeceksin.

