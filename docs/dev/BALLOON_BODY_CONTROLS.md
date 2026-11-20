# Kamera olmadan iki elle balon testi

2026-11-20: C++ çekirdeği ve sanal iskelet kabul betiği hazır. C++ build ve
çekirdeğin canlı kabulü yapılmadı. Kamera gerekmez; yeni binary gerekir.
Önceki fare taşıması kullanıcı tarafından Player'da kabul edildi.

## Davranış

- Tek el teması: 0,24 saniyelik karar penceresinden sonra patlama. El bu sırada
  geri çekilmiş olsa da temas kenarı korunur; güvenilir yeni el paketleri
  gelmeye devam etmelidir. İkinci el aynı balonu çevrelerse tutma önceliklidir.
- İki el: balonun karşılıklı iki yanında, el küresi dahil temas menzilinde
  0,12 saniye kalınca tutulur. Başlangıç el açıklığı kaydedilir.
- Taşıma: iki bileğin dünya koordinatlarındaki orta noktası, 0,06 saniyelik
  yumuşatmayla izlenir. İlk tutuşun balon/orta nokta farkı korunur; ilk anda
  konum atlaması yoktur. Oyuncunun yürümesi de aynı dünya konumu hareketidir.
- Bırakma: iki el de güvenilirken açıklık başlangıcın 1,65 katını veya
  başlangıç + 0,25 metreyi aşar ve 0,15 saniye sürerse balon son kabul edilen
  konumda patlar. Karar sırasında taşınmış konum dondurulur.
- Takip kaybı: bir el/kişi confidence <0,6 veya yeni paket yoksa hareket donar.
  Son iki güvenilir el gözleminden 0,8 saniye sonra patlatmadan iptal edilir.
  Kısa kayıp aynı kişiyle sürdürülür. Başka kimlik tutmayı devralamaz.
- Patlama/iptal sonrasında görülen eller eski temas alanından çıkmadan yeni
  etkileşim açılmaz. Bir el görünüyorsa onun çıkması yeterlidir; iki elin
  kaybolması yeni bir temas sayılmaz.

Tek anda mevcut ortak çekirdekte bir balon tutulabilir. Fare veya başka kişi
tutuyorsa iskelet girişi onu devralmaz. Fare ekran hareketi ve iskelet dünya
hareketi aynı çarpışma/instance-transform/patlama yolunu kullanır.

## Giriş ve API

Gerçek kamera köprüsü, avatar, kamera kartı ve sanal testin ortak kaynağı:
`input.set_body_poses` / Python `rt.input.set_body_poses`.
Mevcut 15 eklemli paket değişmedi. `l_wrist` ve `r_wrist` dünya XYZ/metre ve
confidence kullanılır; sender zamanı monoton olmalıdır. Karar ve kayıp süreleri
receiver steady-clock ile ölçülür. Tutulan/atlanmış eski pozun tekrarı yeni
gözlem sayılmaz: `BodyPoseRuntime::receivedAt(id)` alım zamanını verir.
Yeni paket gelmediyse poz map'i tekrar kopyalanmaz; kayıp saatleri yine yürür.

Varsayılan `body_controls=false`; kamera/poz gelmesi tek başına oyunu değiştirmez.
Player Game settings > İki elle iskelet kontrolü > Apply/Save ile açılır.
Studio Game Design formunda aynı seçenek vardır. Yerel Player profiline ve
proje `game_design.balloon_demo.body_controls` alanına kaydolur. Yalnız bu
seçeneği değiştirmek balon/havuzları yeniden kurmaz.

```python
# Ortak Python yüzeyi:
rt.paint.wet_demo("body", {"enabled": True})
rt.paint.wet_demo("body", {})  # status
# Aynı IPC:
ipc.call("paint.wet_demo", operation="body", config={"enabled": True})
```

Yalnız opsiyonel boolean `enabled` kabul edilir. Yanlış tür, bilinmeyen anahtar
ve mesh olmayan demoda etkinleştirme mutasyondan önce reddedilir. Configure
aynı alanı `body_controls` boolean olarak alır. Status `body_interaction`
enabled/active_person/persons[].phase ve grabs/moves/releases/taps/
tracking_cancels/error sayaçlarını içerir.

Ortak pointer API dünya uzayıyla genişletildi:

```python
demo("pointer", {"action": "down", "position": [x, y, z],
                 "balloon": 0, "owner": "body:9001:both"})
demo("pointer", {"action": "move", "position": [x, y, z], "owner": "body:9001:both"})
demo("pointer", {"action": "up", "owner": "body:9001:both"})
```

`screen` ve `position` birlikte kullanılamaz. Dünya position sonlu XYZ,
±10000 metre; opsiyonel balloon yalnız world-down'da nonnegative int32'dir.
Seçilen balonun temas menzilinde olunmalıdır; uzaktan id ile tutma yoktur.
World-down mesh taşımasıdır; legacy ekran `pop` modu bunu değiştirmez. Sonraki
hareket aynı koordinat uzayını kullanır. Dünya tutmasında wheel yoktur.
Konumsuz up son gerçek konumda bırakır; tracking loss kendi owner'ını iptal eder.

## Sanal test

```powershell
# Şimdi de çalışır: offline 113 paket, 15 eklem, şema kontrolü; uygulamaya bağlanmaz.
python scripts/test/rt_balloon_body_ipc.py --plan

# Studio + Player C++ build sonrası; disposable sahne açık, kamera/sender kapalı:
python scripts/test/rt_balloon_body_ipc.py --run --exercise-disposable
# Studio Playtest canlıyken aynı çekirdek:
python scripts/test/rt_balloon_body_ipc.py --run --studio --exercise-disposable
```

Canlı betik yalnız `input.set_body_poses` ile 30 Hz sentetik iskelet gönderir;
doğrudan pointer/pop/burst komutu göndermez. Hazırlıkta demo configure ve
reset_balloons kullanır. Denemeler: ilk elin hemen patlatmaması, iki elle tutma,
sanal yürüme/orta nokta ile taşıma, ayrılınca son konumda patlama, kısa tek el
temasının geri çekilmeden sonra da patlaması, confidence kaybında donma,
aynı kimliğin geri gelmesi ve tamamen kaybolunca patlatmadan iptal.

Neutral test pozlarında bilekler kasıtlı olarak odadan uzağa çekilir: kalabalık
balonlarda tesadüfi temas üretmeden jesti sıfırlayan, anatomik demo olmayan test
pozudur. Kamera aktifse, mevcut başka kişi/sender veya tutulan balon varsa test
başlamaz. Oyun session'ı yapılandırılmamış veya attract/auto_start=false/rules=[]
olmalıdır; başka kurallar test patlamalarını üretmemelidir.

Betik sonunda pozları temizler ve başlangıç demo config'ini geri kurar. Oturum
boyası/balon konumları/sayaçları geri alınmaz; proje veya yerel profil kaydedilmez.
Embedded script workspace'den çalıştırılmaz; harici terminal/named-pipe kullanılır.

Bu tur offline şema 113 paket PASS; proje kayıtları/EHsc, Python AST, EN/TR,
config alanları ve diff statik kontrol edildi. C++ derleme, tanıyıcının gerçek
çalışması, Player çizimi/çarpışma maliyeti ve gerçek kamera jest doğruluğu henüz
doğrulanmadı. Shader değişmedi. Yeni `BalloonBodyInteraction.cpp` iki hedefe kayıtlı;
BodyPoseRuntime header/metodu ve BalloonInteraction world yolu da derlenmelidir.


2026-11-20 kullanıcı Player canlı kabul sonucu: iki-el grip / sanal yürüme orta
noktası / ayrılınca bırakma PASS; doğrudan pop/pointer göndermeden tek-el touch
PASS; confidence freeze / aynı kimlik resume / eksik paket cancel PASS;
validation ve persisted design PASS. Sayaçlar: grabs=2, moves=49, releases=1,
taps=1, tracking_cancels=1. Kamera donanımı testi yapılmadı. Sonraki gizli obje
geçiş politikası değişiklikleri bu kabulden sonradır ve yeni derleme/test bekler.
