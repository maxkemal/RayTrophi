# Balon tutma ve taşıma

2026-11-20: kaynak hazır, C++ Studio/Player build ve canlı kabul bekleniyor.
Bu tur C++ build, Player/Studio başlatma veya canlı sahnede mutasyon yapılmadı.
Shader değişmedi. Gizli objelerle ana salonu yenileme sonraki çalışmadır.

## Kullanım

Mesh balonlarda varsayılan ve normal arayüzdeki tek fare kontrolü:

- Tıkla ve bırak: balon olduğu yerde patlar.
- Sol basılı tut: fareyle sağ/sol/yukarı/aşağı taşı; tekerlekle ileri/geri taşı.
  Pozitif tekerlek kameradan uzaklaştırır (adım başına 0,35 metre).
- Bırak: son kabul edilmiş dünya konumunda patlat.
- Escape, odak kaybı veya menü/debug kamera açma: patlatmadan iptal et.

Player ve Studio formundaki mod seçiciler kaldırıldı; açıklama metni kaldı.
Alan verilmemiş eski/yeni mesh proje config'leri carry kullanır. Açıkça kaydedilmiş
`interaction_mode:"pop"` eski proje, yerel profil ve scriptlerde korunur; API'den
hala seçilebilir. Particle fallback'in varsayılanı pop'tur ve carry desteklemez.

Apply oturumda, Save bu Player'da bu oyun için `player_game_settings.json`
içinde saklar. Yalnız kontrol modunu değiştirmek havuzları yeniden kurmaz,
balonları yeniden doğurmaz ve boyayı temizlemez. Başka oyun ayarlarıyla birlikte
değiştirilirse onların mevcut yeniden yapılandırma kuralları geçerlidir.
Studio Game Design formunda aynı `interaction_mode` proje ayarı vardır.
Taşıma yalnız flat mesh balonlarda desteklenir; particle fallback'te seçenek
kapalıdır ve API açık hata döndürür.

## Tek çekirdek ve sözleşme

`Paint/BalloonInteraction.*` pointer yaşam döngüsünü ve çarpışmayı yönetir.
`BalloonPaintDemo.cpp` balon konumu/anchor ve mevcut instance-transform yolunu
günceller; patlama mevcut `popBalloons` yolunda gerçekleşir. Geometri rebake veya
her kare GPU konum indirmesi yoktur. Native `Core/PlayerBalloonInput.*` yalnız
SDL girişini toplar, ardışık hareketleri frame başına birleştirir ve ortak
`rtapi::wetPaintDemoCommand` hizmetine gönderir.

Mevcut Python/IPC yüzeyleri aynıdır:

```python
# Python: rt.paint.wet_demo(operation, config)
# IPC: client.call("paint.wet_demo", operation=operation, config=config)
demo("interaction", {"mode": "carry"})  # pop|carry; havuzları yeniden kurmaz
demo("pointer", {"action": "down", "screen": [0.5, 0.5]})
demo("pointer", {"action": "move", "screen": [0.6, 0.5]})
demo("pointer", {"action": "wheel", "wheel": 2})
demo("pointer", {"action": "up"})
demo("pointer", {"action": "cancel"})
```

`screen` üst-sol başlangıçlı normalize [x,y], 0..1, sonlu sayılar. `down` için
zorunlu; move/up/wheel verilmezse son pointer noktası kullanılır. `wheel` yalnız
wheel eyleminde, zorunlu sonlu sayı -100..100. `owner` isteğe bağlı string,
1..64 karakter, varsayılan `mouse`. Bir owner'ın tuttuğu balonu başka owner
devralamaz, hareket ettiremez, bırakamaz veya iptal edemez (başarılı no-op).
Tek anda bir tutma vardır. Gelecekte iskelet el girişleri aynı servise örneğin
`owner="body:7:right"` ile bağlanabilir; bu tur iskelet jest algılayıcısı eklenmedi.

`pointer` sonucu kompakt `{held, owner, depth, moved, popped}` nesnesidir.
-1 balon bulunmadığı/işlem yapılmadığı anlamına gelir. `depth` kamera ileri
eksenindeki metre mesafesidir; duvara yaslandıktan sonra tekerlek ters yönde
hemen yanıt verebilsin diye kabul edilen konuma uyarlanır. Diğer demo işlemleri
tam status biçimini korur. Status `interaction` ve `balloons[].held` ekler.
Configure/proje kaydında `interaction_mode: "pop"|"carry"`; mesh varsayılanı carry,
particle fallback varsayılanı pop'tur.

Yanlış action, anahtar, tür, ekran sınırı veya wheel değeri tutmayı değiştirmeden
reddedilir. Hazırlanmamış demo down/move/up/wheel için hata döndürür; cancel
hazırlanmamış demo üzerinde de güvenlidir. `interaction` yalnız hazırlanmış
demo ve tek `mode` alanını kabul eder. Doğrudan mevcut `pop` işlemi korunur;
tutulan balona dışarıdan açıkça pop verilirse tutma da bırakılır.

## Hareket ve sınırlar

Tutulan balon bob/drift güncellemesinden ve rastgele burst seçiminden çıkarılır.
Tutma aktif oyuncu etkinliğidir; idle reset tetiklemez. Finale kuyruğu tutulan
balonu atlar; finale reset tutma bitene kadar bekler. Reset/reconfigure/project
replacement tutmayı temizler. İptal edilen balon son konumundan normal hareketine
devam eder; patlama ve eski konuma geri sıçrama yapılmaz.

Çarpışma, görünür flat TriangleMesh'lerin DNA SoA yerel kaynak pozisyonları ve
ortak dünya dönüşümleri üzerinden alınan statik üçgen snapshot'ını kullanır.
Balon visual node'ları, transient ve skinli mesh'ler dışarıda kalır. Opacity'si
sıfır olan gizli obje `visible=true` kaldığından engeldir; mesh visibility kapalı
objeler engel değildir. Küre yarıçapı pick yarıçapının tam boyut ×1,15 zarfıdır;
doğuş animasyonu sırasında da küçülmez. 1 cm güvenlik payı, mesafe tabanlı
conservative advancement ve teğetsel kayma büyük wheel sıçramalarında duvar
geçişini engeller. İterasyon bütçesi dolarsa son güvenli noktada kalır.

Snapshot yalnız tutma başlangıcında alınır. Hareketli karakterler, hareketli
engeller, ayrı terrain/instance yolları ve balon-balona fizik çarpışması bu
sürümde hedef değildir. Büyük statik sahnelerde pickup/sweep maliyeti ayrıca
ölçülmelidir; oda sahnesinde çarpışma verisi her pointer olayında yeniden kurulmaz.

## Build ve kabul

Studio + Player artımlı build. Yeni iki cpp iki vcxproj'de `/EHsc` ile kayıtlı;
CMake mevcut recursive glob'u kullanır. Proje, shader binding/push ABI'si değişmez.

Build sonrası disposable sahnede, ayrı terminalden:

```powershell
python scripts/test/rt_balloon_carry_ipc.py --player --exercise-disposable
# Studio ortak çekirdeği:
python scripts/test/rt_balloon_carry_ipc.py --exercise-disposable
```

Test başlangıç config'ini sonunda geri kurar; geçici balon/boya durumunu geri
almaz ve kayıt yapmaz. Kontroller: down patlatmaz; taşıma/depth gerçekten taşır;
tutulan balon drift etmez; yanlış girdiler atomik reddedilir; başka owner
devralamaz; cancel patlatmaz; rastgele burst tutulan balonu atlar; wheel arka
duvarı geçmez; release patlaması taşınmış dünya konumunda olur; mode değişimi
tutmayı temizler; proje design çıktısı modu içerir; eski touch/pop çalışır.

Elle: sol basılı sürükle, wheel iki yön, bırak; duvara yasla ve yana kaydır;
Escape/Alt-Tab/F2/F5/F9 ile tutmanın patlatmadan iptalini, yeniden açılan projede
ve Player yerel Save/yeniden başlatma sonrası modun korunmasını kontrol et.
Şu an yalnız XML/kayıt, Python syntax, config alanları, EN/TR ve diff statik
kontrolleri yapıldı; C++ ve canlı taşıma kabulü ÇALIŞTIRILMADI.

2026-11-20 kullanıcı kabulü: taşıma ve kısa tıklamayla patlatma Player
üzerinde denendi ve başarılı bulundu. Sonraki değişiklik yalnız varsayılanı
carry yapmak ve normal UI mod seçicilerini kaldırmaktır; bu son UI/default
değişikliği yeni C++ build bekler. Otomatik dış test henüz çalıştırılmadı.
