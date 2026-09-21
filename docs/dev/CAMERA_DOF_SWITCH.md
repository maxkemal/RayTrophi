# Alan derinliği anahtarı: `aperture == 0` neden bir kapı olamazdı

> **Durum:** AKTİF — 2026-09-06 (II). Yazıldı, **DERLENMEDİ**. Kabul testleri
> `scripts/ipc/Probe-CameraDofSwitch.ps1` ve `scripts/ipc/Probe-RealtimeDof.ps1`.
> Önceki parti: [REALTIME_HDR_AND_DOF.md](REALTIME_HDR_AND_DOF.md).

## Belirti

> *"f-stop değeri UI'de değişirse DoF aktif olup bir daha çıkmıyor; önceden
> f-stop sıfır olunca çıkıyordu."*

## Kök neden: DEĞER ile SENTINEL karıştırılmıştı

Alan derinliğinin tek kapalı-anahtarı `aperture == 0` idi. O bir **değer değil,
bir sentinel**: fiziksel bir açıklık sıfır olmaz, ve **hiçbir f-sayısı sıfır
açıklık üretmez**. F-stop kadranı `aperture`ı yazmaya başlar başlamaz kapının
kapanabileceği tek nokta **erişilemez** hale geldi.

★★★★ Genel ders: **bir alanın "yok" durumunu kendi değer kümesinde kodlarsan,
o alana bağladığın ilk sürekli kadran kapıyı yok eder.** Ve bu sessizce olur:
hiçbir yerde hata yoktur, yalnızca geri dönüşü olmayan bir durum vardır.

## Yeni sözleşme

| Alan | Anlamı |
|---|---|
| `Camera::aperture` | **her zaman** fiziksel açıklık (f-sayısının ikizi) |
| `Camera::depth_of_field` | lens diski **örneklensin mi** |
| `Camera::effectiveLensRadius()` | ★ tüketicilerin gördüğü tek büyüklük |

★★★ Kapatmak değeri **yok etmez**: tekrar açmak eski bulanıklığı aynen geri
getirir. Bir anahtar aynı zamanda sıfırlayıcıysa, kullanıcı her açışında
kadranları yeniden ayarlar — kimse buna bug demez, yalnızca "sinir bozucu" der.

★★ Ve kapı tüketicilerde tek tek kurulmaz. Aynı `&&`i CPU renderer'a, OptiX'e,
Vulkan RT'ye ve raster post geçişine ayrı ayrı yazmak, birinin unutulduğu günü
garanti ederdi. Kapı **iki taşıma üreticisinde** uygulanır
(`VulkanBackendAdapter::syncCamera` ve `Renderer`ın kendi `CameraParams`
kopyası — ★ ikincisi kolayca gözden kaçar) ve CPU/OptiX yolunda accessor'dan
okunur.

## Yol boyunca çıkan ikinci arıza: f-sayısının ÜÇ farklı tanımı

Aynı kamerada üç ayrı dönüşüm vardı ve üçü de farklı sayı veriyordu:

| Yazan/okuyan | Formül | f/2.8 için |
|---|---|---|
| Preset combo'su | `FSTOP_PRESETS[i].aperture_value` | 1.20 |
| F-Stop slider'ı | `(focal_mm / f) * 0.01` | 0.18 |
| Panel gösterimi | `focal_mm / aperture` | **f/280** (üst sınıra yapışır) |
| Pozlama geri düşüşü | `0.8 / aperture` | — |

Yani combo'dan f/2.8 seçip slider'a dokunmak bulanıklığı **7 kat**
değiştiriyordu, panelin gösterdiği f-sayısı ise üst sınıra yapışıyordu.

★★★ Otorite **preset tablosudur**: elle ayarlanmış sanatsal bir ölçek (1/f
değil). Ara değerler tabloyu **log-log interpole eder**, iki yön birbirinin
tersidir, ve preset noktalarında değerler **birebir korunur** — tabloyu bir
formülle değiştirmek her mevcut sahnenin bulanıklığını sessizce değiştirirdi
(CLAUDE.md kural 5).

★ Aynı düzeltme HUD üçgeni ile paneli de barıştırdı: üçgeni çevirince panelin
f-sayısı **kımıldamıyordu**, çünkü biri indeksi diğeri açıklığı yazıyordu.
`camera.set_fstop_preset`teki "kayıtlı borç" da bu partide ödendi — panel,
HUD ve script artık aynı sayıyı üretir.

★★ Yan etki, ve **bilinçlidir**: Custom f-stop'lu (indeks 0) sahnelerde pozlama
değişir, çünkü pozlamanın f-sayısı artık panelin gösterdiğiyle aynı. Eskisi
1.6x sapmalı bir geri düşüştü.

## Yolda bulunan üçüncü şey: CPU lens örneklemesi ölçeksizdi

`Camera::get_ray` içinde `random_in_unit_polygon()` **birim** diski örnekler ama
yarıçapla **çarpılmıyordu**. Yani CPU yolunda sapma, açıklık ne olursa olsun
1 dünya birimine kadar çıkıyordu. Belirtisi "CPU render'da DoF aşırı"dır ve
kadranla ölçeklenmediği için kalibrasyon turlarına gömülür.

★ Dördüncüsü: `scene_ui_animation.cpp` keyframe uygularken **bayrağı** değere
yazıyordu (`aperture = has_aperture`), yani anahtarlanmış her karede açıklık
1.0 oluyordu. `TimelineWidget` aynı işi doğru yapıyordu — **iki kopya, biri
yanlış**.

## Script yüzeyi

- `camera.set_depth_of_field {enabled}` — anahtar. Açıklığa dokunmaz; açılırken
  açıklık hiç yazılmamışsa f-sayısından türetir (yoksa "açtım ama hiçbir şey
  olmadı" üretirdi).
- `camera.get` → `depth_of_field`, **`effective_lens_radius`** (uygulanan
  büyüklük), ve artık Custom kameralarda da doğru olan `f_number`.
- `viewport.get_depth_of_field` → **beşinci kapı**: `camera depth of field is
  off`. Burada "aperture is 0" demek **yanlış teşhis** olurdu ve kullanıcıyı
  hiçbir şeyi değiştirmeyen bir eyleme (f-stop çevirmeye) gönderirdi.

## Eski projeler

Anahtar dosyada yoksa varsayılan **`aperture > 0.001`** — yani dosyanın kendi
niyeti. Sabit bir `true`, DoF'u kapatmış her eski sahneyi sessizce
bulanıklaştırırdı. İki serileştirici de (`ProjectManager`, `SceneSerializer`)
aynı geri düşüşü kullanır.

## Bilinçli olarak yapılmayanlar

- **Anahtar animasyona açılmadı.** Keyframe'ler açıklığı sürer; "lensi bir
  karede kapat" senaryosu için talep yok, ve bir bool'u interpole etmek yeni
  bir belirsizlik olurdu.
- **`aperture` yeniden ADLANDIRILMADI.** Anlamı değişmedi (hâlâ fiziksel
  açıklık); değişen, UI'nin 0'ı kontrol amaçlı kullanmayı bırakmasıdır. Eski
  veri doğru okunuyor, bu yüzden rule 5'in gerektirdiği rename gerekmiyor.
- **Preset tablosu düzeltilmedi.** Ölçek fiziksel değil sanatsal; onu
  düzeltmek görünüm değişikliğidir ve ayrı bir karardır.

---

# Ek: HUD ve kamera paneli düzeni (aynı gün, ikinci tur)

## HUD'da ne olmalı — ve ne olmamalı

★★★ Ayrım: **HUD ölçü aletidir, panel ayarın sahibidir.** Üçgen bir istisna
değil, bir *kadran*: fotoğrafçının vizörden çevirdiği üç şey. Ona dördüncü bir
şey eklemek HUD'u ikinci bir ayar paneline çevirir ve aynı değer iki yerde
tutulmaya başlar — bu deponun tekrar tekrar ödediği bedel.

Bu yüzden DoF **anahtarı** HUD'a girdi ama **açıklık/odak kadranları girmedi**.
Anahtarın girme gerekçesi tek ve somut: DoF kapanınca **fokus halkası da
kayboluyor**, yani HUD'un kendi odak aracı yok oluyor ve HUD'dan geri dönmenin
yolu kalmıyordu. Bir aracın kendini kapatıp geri açamaması, kullanıcının
şikâyet ettiği arızanın ta kendisiydi — ters yönü.

★ Rozet ayrıca AP kadranını dürüst yapar: DoF kapalıyken f-stop çevirmek
pozlamayı değiştirir ama bulanıklığı değiştirmez, ve bunun sebebi ekranda yazar.

## Yerleşim: ölçü, çizilen en alt pikseldir

Bilgi satırları üçgenin **geometrik tabanından** (`cy + h*0.5`) başlıyordu; alt
köşelerin değer yazıları ise o tabanın **altına** taşıyor. h = 85 × 0.866 = 73.6
için taban `cy+36.8`, SH/AP yazıları `cy+45..59` — yani ilk bilgi satırı **her
zaman** onların üzerine biniyordu.

★★ Ders: bir **şekle** göre hizalanan yerleşim, şeklin dışına taşan
**etiketleri** görmez.

## "Pro Camera Features" gerçekte neydi

Kullanıcı "AF dışında pek kullanışlı değil" dedi. Koda bakınca sebep beğeni
değil **körlük** çıktı: histogram, focus peaking ve zebra
`ctx.renderer.getFrameBuffer()` okuyor — yani **CPU renderer'ın** tamponunu,
viewport'u değil. Material/Solid raster viewport'unda o tampon boştur; zebra ve
peaking sessizce `return` ediyor.

> ★★★★ Yani üç onay kutusu **hiçbir şey yapmıyordu ve bunun ekranda hiçbir
> işareti yoktu.** Bu, "panelin yalan söylemesi" sınıfının tam örneği.

Karar: **silmek yerine körlüğü görünür yapmak.** Kaynak yoksa kutular kapalı ve
sebebi yazılı. Bunları viewport karesine bağlamak ayrı bir iştir (readback yolu
`render.probe`de zaten var) — ama yapılana kadar yalan söylememeliler.

## AF neden Camera paneline taşındı

AF bir izleme overlay'i **değildir**: AF-C modunda seçili nokta her karede
sahneyi ölçüp `cam.focus_dist`i **ezer** ve backend'i yeniden senkronlar.
Kamera durumunu değiştiren bir şey, kamera panelinde yaşar.

★★ Ve kural 1 ihlali buradaydı: AF'nin **hiç** script yüzeyi yoktu.
`viewport.set_af` / `viewport.get_af` eklendi; `get` nokta sayısını alan
modundan **türetir** (Zone21 = 5×5) ve kapalıysa sebebi adıyla söyler (HUD
kapalı / BVH yok / sekans render'ı sürüyor). Kabul testi:
`scripts/ipc/Probe-ViewportAf.ps1` — 4. kapı AF-C'nin odağı gerçekten ezdiğini
ölçer, ki "focus_distance ayarladım ama tutmuyor" sorusunun cevabı bir yerde
dursun.

★ Ölü kod: `drawProCameraPanel` boş bir gövdeydi (kural 5) — söküldü.

