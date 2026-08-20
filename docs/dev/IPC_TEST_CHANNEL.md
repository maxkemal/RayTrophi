# IPC test kanalı — kare döngüsünü GÖREN test yolu

> **Durum:** AKTİF — 2026-08-19'da kuruldu. Sim kontrol sözleşmesi
> **ölçüldü ve geçti**; normal Play panelden doğrulandi. Açık kalan iki arıza
> aşağıda (timeline yolu duruyor; ikinci domain parçacıkları siliyor). İstemci `scripts/test/rt_ipc.py`,
> süit `scripts/test/rt_test_physics_ipc.py`, sonuç
> `scripts/test/_physics_ipc_result.txt`.

## Neden ikinci bir kanal

`script.run_file` ile koşan bir test uygulamanın **ana thread'ini tutar**, yani
kare döngüsü o script çalışırken **hiç dönmez**. Bu bir yavaşlık değil, bir
**körlük**: döngünün yaptığı hiçbir şey o testten görünmez.

Ölçüldü (2026-08-19): birebir aynı çağrı dizisi

| | tohum | ikinci domain sonrası |
|---|---|---|
| script içinden | 6760 | 6760 |
| IPC'den | 22932 | **0** |

Aynı motor, aynı metotlar, zıt cevaplar. **Hiçbiri "gerçek" olan değil** —
farklı yarıları ölçüyorlar:

- **script içi** — çekirdek mantık, hızlı, döngüye kör
- **IPC** — uygulamayı gerçekten sürer, döngü arızalarını görür

Tek kişilik bir projede bu ikisi olmadan *"uygulamada çalışıyor ama testte
çalışmıyor"* yanlışlanamaz bir cümle olarak kalır.

## ★★★ İlk koşunun bulduğu: `physics.step` GERİ ALINIYOR

IPC'den 240 × `physics.step` sonrası gövde `y = 50.00000`, `simulated = false`.
Açık okuma "çözücü koşmuyor". **Yanlış.**

Okumayı adımlarla **aynı batch'e** koyunca (aradan kare geçmiyor):

```
read INSIDE the batch : y=49.70980  (58 adımda 0.29020 m, 0.242 s)
read AFTER  the batch : y=50.00000  simulated=false
```

0.29 m / 0.242 s tam olarak yerçekimi. **Adım gerçekten çalışıyor; kare döngüsü
onu %100 geri alıyor.** Ve çağrı `true` dönüyor.

★★★ Bu, "yok sayılmak"tan **kötüdür**: ajan gerçek iş yapar, başarı raporu alır,
hiçbir şey ölçemez, ve çözücüyü suçlar.

## ★★ Timeline yolu hayatta kalıyor — ama DURUYOR

`timeline.set_frame` ile sürülünce poz bir sonraki çağrıda hâlâ orada
(`simulated = true`). Ama ilerlemiyor:

```
frame   6 -> y=49.67790
frame  12 -> y=49.43570
frame  24 -> y=49.43524
frame  48 -> y=49.43570
```

Yalnızca son kareyi okuyan bir çağıran, **havada durmuş** bir gövde görür —
makul görünen, tamamen yanlış bir gözlem.

★ `scene_data.h` bunun bir yarısını zaten yazıyor: re-sim **UI tick başına
`max_steps` ile sınırlı**, yani rigid timeline gösterilen kareyi geriden takip
eder. Ama gözlenen şey gecikme değil **durma**; kök henüz bulunmadı.

## ★★★ Ve asıl yapısal sebep: `setFrame` SAYIYI taşıyor, SİMÜLASYONU değil

`rtapi::setFrame` dört ayrı kare değişkeni yazıyor —
`ui.timeline`, `scene.timeline.current_frame`,
`render_settings.animation_current_frame`, `animation_playback_frame` — ve
`start_render` bayrağını kaldırıyor. **Çözücüyü ilerleten yolu
(`syncRigidToFrame` / `restoreSimFrame`) hiç çağırmıyor.**

Panelden scrub yapınca çözücü ilerliyor çünkü o yol yakalama adımını da
tetikliyor. IPC'den kare kurunca **sayı gidiyor, sahne gitmiyor.**

Kullanıcı tarafından bağımsız olarak gözlendi (2026-08-19): *"IPC script ile
sürülünce UI timeline işlenen kareye gitmiyor, ya da play yapılıyor ama
viewport timeline'ı takip etmiyor."* Aynı kök.

## Karar ve sözleşme (2026-08-19)

Kullanıcının kararı: **otorite yarışı yok.** Timeline kullanıcınındır — her an
scrub, play, pause, stop yapabilir. Script de sürebilir; eksik olan şey
script'in **haberdar edilmesiydi.**

★★★ Ama ölçüm bunu bir yerde aşıyordu: **testte timeline'a kimse dokunmadı.**
Döngü kendi başına geri alıyordu. Yani "koşmadan önce durumu öğren" tek başına
yetmez — script 240 kez "geri alındın" bilgisi alır ve yine hiçbir şey ölçemez.

Üç parça birlikte uygulandı:

1. **`physics.step` playhead'i de ilerletir.** Anlaşmazlığın kaynağı buydu:
   script `t = 0.24 s`'de, döngü frame 0'da, ve döngü dünyayı kendine
   uydurdu. **Anlaşamayan iki saatin hakeme değil, tek saat olmaya ihtiyacı
   var.** Kullanıcı bunun görünen yarısını bağımsız olarak bildirmişti:
   *"script sürerken playhead gitmiyor."*
2. **Döngü, script adım atarken rigid'i yeniden sürmez** — ama `tl_frame`
   script'in koyduğu kareden ayrıldığı an (kullanıcı scrub yaptı) ya da play
   başladığı an talep **derhal** düşer. Kullanıcı hiç engellenmez.
3. **`sim.control_state`** — `epoch` (çözücüler her yeniden konumlandığında
   artar), `driver` (`user`/`playback`/`script`), `frame`, `playing`.
   ★★ `epoch` taşıyıcı alan: ölçümün **etrafında** okunur. Değiştiyse ölçüm
   geçersizdir ve çağıran **bunu bilir**. Bugüne kadar geri alınmış bir poz ile
   hiç hareket etmemiş bir gövde **birbirinden ayırt edilemiyordu** — pahalı
   olan kısım buydu.

★ `timeline.set_frame` de artık talebi düşürüyor: script'ten kare kurmak
kullanıcı-eşdeğeri bir eylemdir.

## Eski hâli: doğru davranış ne olurdu

Bir ajan için **tek** anlamlı ilkel şu: *"simülasyonu ilerlet, sonra bak."*
Bugün iki yol var ve ikisi de bunu vermiyor — biri geri alınıyor, öteki
duruyor. Yön:

`setFrame` kareyi kurarken çözücü senkronunu da yapmalı, ve `physics.step`
döngünün otoritesiyle **çakışmayan** tek bir yola indirilmeli. Bu depo iki kod
yolunu "her ihtimale karşı" yaşatmanın bedelini defalarca ödedi; burada iki yol
zaten birbirini eziyor.

## Ölçülen sonuç (2026-08-19, sözleşme sonrası)

```
1.  batch içinde y=49.70980 → batch sonrası y=49.70980  simulated=True   OK
1b. frame 0 → 4, driver=script, epoch 383 → 431                        OK
    scrub sonrası: driver=user, script_driving=False, epoch=432         OK
3.  ret mesajı: region (-0.400 50.000 -0.400)                          OK
4.  bölgesiz tohum → 14440 parçacık                                    OK
```

Kullanıcı panelden doğruladı: **timeline doğru çalışıyor**, normal Play
bozulmamış. ★ Script sürerken playhead'in ilerlemesi artık beklenen davranış.

★★ **Ergonomi notu:** adımlar 1/240 s'lik paketler hâlinde çok hızlı aktığı için
kullanıcının araya girecek pencere bulması zor. Kontrol scrub anında derhal
kullanıcıya geçiyor (1b bunu ölçüyor), ama *pratikte* araya girebilmek ayrı bir
iş — açık.

## AÇIK kalan iki arıza

1. **Timeline ile sürülen düşüş duruyor:** `49.678 → 49.436 → 49.436 → 49.436`.
   Bir koşuda frame 24'te `y=50.0, simulated=False` döndü — yani sadece
   durmuyor, **arada rest'e sıfırlanıyor**. `physics.step` yolundan bağımsız;
   timeline'ın kendi yakalama/cache yolunda.
2. **İkinci domain birincinin 22932 parçacığının TAMAMINI siliyor**, iki okuyucu
   da hemfikir. Script kanalı bunu göremıyor.

## Süitin kendi dersleri

★★★ **İki kontrolüm yanlış nicelikle ölçtüğü için YEŞİL geçti**, ve bu bir test
dosyasındaki en kötü hata:

1. `abs(after - inside) < 0.01 * y` — gövde 0.29 m geri alınmıştı, ama 49.7 m
   irtifanın %1'i 0.497 m. **Toleransı yanlış niceliğe ölçekledi.** Doğrusu
   hareketin bir kesri.
2. `seen[i] > seen[i+1] - 1e-4` — üç birebir aynı okumada geçti. **Eşitliğe
   izin veren bir tolerans, plato'yu göremez.** Doğrusu kesin azalma.

★ 0. vaka **döngünün gerçekten döndüğünü ÖLÇÜYOR** (`viewport.samples`
`rendered` modda ilerler, `solid`'de 0'a çakılıdır) ve dönmüyorsa **koşuyu
durduruyor**. Yoksa bu dosya sessizce in-process rig'in yavaş bir kopyasına
dönüşür ve yeşili göründüğünün tam tersini anlatır.

★ `scripts/test/rt_ipc.py` içinde `method` **positional-only**: `agent.describe`
gibi metotların `method` adlı bir parametresi var, ve `/` olmadan çağrı kendi
ilk argümanıyla çakışıyor. API'nin bir kısmına ulaşamayan istemci, aletin kör
noktasıdır.

★ Bu dosya `x64/Release/scripts` altına **kopyalanmaz** — orası uygulamanın
YÜKLEDİĞİ scriptler için; bu uygulamayı dışarıdan sürer.
