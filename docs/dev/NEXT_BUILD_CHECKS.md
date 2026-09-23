# Sıradaki derlemede kontrol edilecekler

> **Durum:** CANLI — 2026-09-24. Bu parti **ölçü aleti + bir kesin düzeltme**.
>
> **1. `camera.list` eklendi (ÖLÇÜ ALETİ).** Açılıştaki default sahnenin
> kamerasının `scene.cameras`'a kayıtlı olup olmadığını söyler. `active_is_orphan`
> alanı bu partinin tek sebebi.
> **2. C DÜZELTİLDİ:** navigasyon ışını `get_ray` → `get_viewport_ray`
> (Main.cpp ~3583). Kökü kesindi: `get_ray` diyafram diskini rastgele örnekler.
> **3. A hâlâ DENENMEDİ:** pan artık seçim kilidini bırakıyor.
>
> **B. ✔ ÇÖZÜLDÜ ve ÖLÇÜLEREK DOĞRULANDI.** `Main.cpp` ~4550'deki
> `g_camera_dirty = false` kaldırıldı. O blok `Camera::is_dirty`'yi tüketip
> yalnızca **render** backend'ini senkronluyor, ama **viewport** senkronunun
> (~5197) bayrağını temizliyordu — raster bayat kamerayla çiziyordu.
> ★★★ **REGRESYONDU, sebebi bu partide yazdığım koddu:** `panWorld` /
> `dollyToPivot` / `orbitAroundPivot` `markDirty()` çağırıyor; eski pan/dolly
> çağırmıyordu. `setLookDirection` (serbest rotate) hâlâ çağırmıyor —
> "yalnızca rotate çalışıyor" tam olarak bu yüzdendi.
> **Doğrulama (düzeltmeden ÖNCEKİ binary'de):** kullanıcı fareyle pan yaptı,
> sonra kameranın KENDİ konumu kendisine yazıldı (değer değişmedi, `markDirty`
> yok) → **GPU karesi değişti**, yani bayat kare vardı ve `markDirty`'siz
> senkron onu düzeltti.
> `markDirty()` çağrıları yerinde: yanlış olan bildirim değil, yutan taraftı.
> Ayrıntı: `docs/dev/KAMERA_ODAK_KILIDI.md`.
>
> Yeni `.cpp` **yok** (RtApiCameraNav.cpp geçen partide eklendi). Yeni shader yok.
> Test script'leri iki yere de kopyalandı.

---

## 0. ★★★★★ B düzeldi mi — solid modda pan ve zoom

Realtime **solid** modda, arızanın göründüğü default sahnede:

1. Orta tuş sürükle (rotate) → dönmeli. *(Zaten çalışıyordu — regresyon kapısı.)*
2. **Shift + orta tuş (pan)** → kaymalı.
3. **Tekerlek (zoom)** → yakınlaşmalı.

- **Bozuksa ne demek:** hâlâ donuksa `g_camera_dirty`'yi temizleyen ikinci bir
  yer daha vardır — kalanlar 2292, 5208, 5782, 5822, 6653. Her birine sor:
  bu blok o bayrağı gerçekten **servis ediyor mu**? Etmiyorsa temizlememeli.
- **★★★ EN SİNSİ HÂLİ:** pan'ın çalışıp zoom'un çalışmaması (ya da tersi). İkisi
  de `markDirty()` çağırıyor, yani ayrışmaları **başka** bir kapı demektir —
  "biri düzeldi" diye kapatma.
- ★ RT/Rendered modda da dene: orada zaten çalışıyordu, bozulmamalı.

---

## 0a. ★★★★ Panel FOV slider'ı raster'ı güncelliyor mu (ESKİ sorun)

Solid modda kamera panelinden **FOV slider'ını** oynat.

- **Ne görmen gerek:** viewport anında değişmeli. RT modda da bozulmamalı.
- **Bozuksa ne demek:** 4535 bloğu `g_camera_dirty = true` yazmıyordur, ya da
  panel `markDirty()` bile çağırmıyordur (o zaman hiçbir backend duymaz).
- ★ Aynı kapıyı paylaşan diğer panel kadranlarını da dene: odak mesafesi,
  diyafram, sensör/lens. Hepsi `cam.markDirty()` deseninde.
- **★★★ EN SİNSİ HÂLİ:** FOV'un düzelip odak mesafesinin düzelmemesi. İkisi de
  aynı desende, ayrışıyorlarsa o kadran `markDirty()` çağırmıyordur.

---

## 0c. `camera.list` — kayıt sorusunu kapattı (ARŞİV: orphan=false, count=1)

Bu partide eklendi. Mod geçişi gözlemi kamera bağlanma hipotezini zayıflattı,
ama soruyu kesin kapatmak ucuz:

```powershell
Invoke-RtIpc camera.list @{} | ConvertTo-Json -Depth 3
```

- **`active_is_orphan = true` veya `count = 0`:** aktif kamera kayıt defterinde
  yok. Düzeltme: default kamerayı yaratan yer de `SceneData::setActiveCamera`'dan
  geçmeli (kaydı yapan üç yol: `newProject` 1192-1200, `openProject` 2877/2883,
  `create_scene` 4493-4526 — kullanıcının "çalışıyor" dediği üç durum da bunlar).
- **Hepsi false:** kamera düzgün bağlı, hipotez kapandı. Madde 0'a dön.

⚠ **Önceki IPC ölçüm tablosu şüpheli:** alınırken uygulamanın hangi shading
modunda olduğu kaydedilmedi. RT modundaysa raster hakkında hiçbir şey söylemez.
Tekrarlarken **önce `viewport.shading` oku** ve sonuca modu da yaz.

---

## 0d. ★★★★ C düzeltildi mi — orta tuş artık sıçramamalı

`get_ray` → `get_viewport_ray` yapıldı. **Kapalı anahtarla test etme:** diyafram
kapalıyken `lens_r == 0` olduğu için sapma zaten sıfırdır, ve o test ölçmediği
şeyi doğruladı sanır.

```powershell
Invoke-RtIpc camera.set_depth_of_field @{ enabled = $true }
Invoke-RtIpc camera.set_aperture @{ aperture = 0.5 }
Invoke-RtIpc camera.set_pivot_mode @{ mode = 'free' }
```

Sonra **aynı piksele** arka arkaya 5 kez orta tıkla, her seferinde:

```powershell
(Invoke-RtIpc camera.get_pivot @{}).nav_distance
```

- **Ne görmen gerek:** beş sayı da **aynı**.
- **Farklı çıkıyorsa:** düzeltme uygulanmamış ya da ikinci bir `get_ray`
  çağrısı daha var — `grep -n "get_ray(" Main.cpp` ile bak.

---

## 1. Derleme ve yetki aynası — bağımsız, saniyeler

Yeni dosya derlenmeli, yeni metotlar dispatch'e ulaşmalı.

```powershell
python scripts/audit_ipc_capabilities.py
```

- **Ne görmen gerek:** `OK - every dispatched method is classified, mirror
  agrees with RtIpcSecurity.cpp, no dead prefixes, descriptors current.`
  (Yazarken 583 metot, 564 belgeli geçti.)
- **Bozuksa ne demek:** `camera.*` namespace'i `RtIpcSecurity.cpp`'de zaten
  vardı, yani burada patlarsa sorun descriptor tablosunun bayatlığıdır —
  `python scripts/gen_ipc_descriptors.py` çalıştır.

---

## 2. ★★★★★ ASIL KAPI: odak navigasyonu sürüklemiyor

Bu partinin tek sebebi bu. Uygulamayı aç, bir obje seç.

```powershell
.\scripts\ipc\Start-RayTrophi.ps1
Import-Module .\scripts\ipc\RtIpc.psm1 -Force
.\scripts\ipc\Probe-CameraPivotLock.ps1
```

- **Ne görmen gerek:** `TUM KAPILAR GECTI` — özellikle kapı 2:
  `camera.set_focus_distance 0.5` yazıldıktan sonra `nav_distance`
  **kımıldamamalı**.
- **Bozuksa ne demek:** `focus_dist` navigasyon yarıçapına yeniden bağlanmış.
  İlk bakılacak yer `Camera::setLookDirection` — orada `focus_dist` görürsen
  kök odur.
- **★ Sahnede obje yoksa betik `exit 2` verir**, bu bir başarısızlık değil;
  önce bir obje yükle.

---

## 3. Panel ile çekirdek aynı şeyi söylüyor mu — hızlı, göz kontrolü

Kamera panelinde (hierarchy paneli, Nav Scale'in altı) yeni satır:
`Nav radius X.XXm  |  Focus Y.YYm`.

- **Ne görmen gerek:** Odak halkasını viewport'ta sürükle. **Focus değişmeli,
  Nav radius sabit kalmalı.** Tekerlekle zoom yap: **Nav radius değişmeli,
  Focus sabit kalmalı.**
- **Bozuksa ne demek:** İkisi birlikte hareket ediyorsa madde 2 zaten kalmıştır;
  ikisi de kımıldamıyorsa panel `rtapi::getCameraPivot`'u çağırmıyor olabilir.
- **★★★ EN SİNSİ HÂLİ:** iki sayı da **makul görünüp** birlikte hareket etmek.
  Kimse bunu bug diye raporlamaz — tam olarak bir yıl boyunca olan buydu.

---

## 4. Frame Selected "tutuyor" mu — el ile, 10 saniye

Bir obje seç → Numpad `.` → sonra orta-tık ile sürükle.

- **Ne görmen gerek:** Obje **ekranın ortasında kalmalı**; kamera onun etrafında
  dönmeli. Panelde `Locked to: <obje adı>` yazmalı.
- **Bozuksa ne demek:** Obje kayıyorsa rotasyon hâlâ fly-look yolundan geçiyor —
  `Main.cpp`'de `else if (scene.camera->pivot_valid)` dalına girilmiyordur.
- Sonra Numpad 4/6/8/2 ile de dene: aynı merkez etrafında dönmeli.

---

## 5. Pan gerçekten rahatladı mı — el ile, asıl şikâyet

Kilitli modda, kameradan çok uzakta ve çok yakında birer obje ile dene.
Shift + orta-tık sürükle.

- **Ne görmen gerek:** Pan hızı objenin mesafesiyle orantılı ve **sürükleme
  boyunca sabit**. Önce odağı 0,5 m'ye çekip tekrar dene — pan **aynı hızda**
  olmalı.
- **Bozuksa ne demek:** Odak çekince pan yavaşlıyorsa madde 2 kalmıştır.

---

## 5b. Pan kilidi BIRAKIYOR mu — yeni davranış, ilk kez deneniyor

Obje seç → Numpad `.` (panelde `Locked to: …`) → Shift + orta tuşla pan.

- **Ne görmen gerek:** Panelde `Orbit Pivot` artık **Free**, `Locked to:` satırı
  kaybolmuş. Sonraki orta tuş rotasyonu objeye **geri ışınlanmamalı** (A arızası).
- **⚠ Bilinçli yan etki:** kilit bırakıldığı için pan'dan sonra rotasyon
  **fly-look** oluyor. Blender'da pan'dan sonra da orbit edilir. His yanlış
  gelirse: `panWorld`'deki `clearOrbitPivot()`'u kaldır — çapa armed kalır,
  rotasyon orbit'te kalır. **Ama o zaman panel "Free" derken imleç çapalaması
  ölü kalır**; takası bilerek yap, gerekçesi `KAMERA_ODAK_KILIDI.md`'de.

---

## 6. Kilitli orbit'in YÖNÜ — zevk meselesi, karar senin

`Main.cpp`'de kilitli rotasyon `orbitAroundPivot(-dx * rot_speed, dy * rot_speed)`
diyor. İşaretleri serbest fly-look hissine göre seçtim ama **elde denenmedi**.

- **Ne görmen gerek:** Fareyi sağa çekince kameranın sağa dönmesi (objenin sola
  kayması) — Blender alışkanlığı.
- **Ters geliyorsa:** `-dx` → `dx` ve/veya `dy` → `-dy`. Bu bir arıza değil,
  ayar; not olarak `KAMERA_ODAK_KILIDI.md`'nin "Açık kalan" bölümünde duruyor.

---

## 7. Serbest mod eski davranışı koruyor mu — regresyon kapısı

`camera.set_pivot_mode {mode='free'}` ya da panelden "Free".

- **Ne görmen gerek:** Orta-tık sürükleme yine **fly-look** (kamera yerinde
  dönüyor), pan imlecin altındaki yüzeye göre ekran-doğru.
- **Bozuksa ne demek:** Serbest modda da orbit ediyorsa `pivot_valid` serbest
  modda temizlenmiyordur (`setCameraPivotMode("free")` → `clearOrbitPivot`).

---

## 8. Ortografik dolly — en son, çünkü diğerlerini maskeler

Numpad 5 ile ortografiğe geç, tekerlekle zoom.

- **Ne görmen gerek:** Görüntü ölçeği değişmeli (`ortho_height`), kamera
  pozisyonu **değişmemeli**.
- **Bozuksa ne demek:** Kamera ilerliyorsa `dollyToPivot`'un ortografik erken
  dönüşü atlanıyordur. ★ Ortografikte hiçbir şey olmuyormuş gibi görünmesi de
  aynı arızanın diğer yüzü — `camera.get_pivot` `orthographic` alanını bu yüzden
  döndürüyor.

---

## Devralınan, bu partide DOĞRULANMADI

Bir önceki listedeki gaz/fluid maddeleri bu partide **çalıştırılmadı** —
dokunulan kod ayrı (kamera navigasyonu), ama doğrulanmamış olmaları bu yüzden
ortadan kalkmıyor. Bkz. git geçmişinde bir önceki `NEXT_BUILD_CHECKS.md`.
