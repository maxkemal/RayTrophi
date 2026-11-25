# Balon boyama odası

2026-11-15 — Kullanıcı Player build'i sonrası dış IPC kabulü PASS. İlk yükleme
hazırlığı ve tekli/üçlü patlamalar ölçüldü; aşağıdaki canlı kabul bölümüne bakın.

## Sahne ve davranış

8×8 m oda, zemin ve üç ayrı materyalli boyanabilir duvar. Odanın içinde rastgele
konumlarda 12 renkli balon bulunur; her balon kırmızı, mavi, sarı veya yeşilin
yalnız birini taşır. Balon görünümü mevcut sphere particle renderer ile çizilir;
gerilen/yırtılan bir balon zarı veya sıvı hacim çözücüsü eklenmedi.

2.5–5 oyun saniyesi aralıkla rastgele balon seçilir. Olayların %60'ı tekli,
%30'u ikili, %10'u üçlü patlama ister. Seçilen balonlar aynı simülasyon tick'inde
patlar. Balon başına 1800 damla, eşit katı açılı küresel dağılımla 4.5–9 m/s
başlangıç hızı alır. Mevcut partikül fiziği yerçekimi ve sürüklemeyi uygular;
damlalar AABB/düzlem çarpışmasında boya bırakıp ölür. Yüzeye ulaşmayanlar
2.8 saniyede biter. Sürekli damla emisyonu yoktur. Patlayan balonlar 3.5 saniye
sonra yeni konum/renkle geri gelir. İlk tekrar üretim tohumu 417'dir.

Bu bir balistik boya damlası demosudur. Bilinen `spread` derece/radyan sorunu
kullanılmadı: başlangıç hızları çekirdekte doğrudan oluşturulur. Analitik
çarpıştırıcılar kullanılır, SDF pişirme yapılmaz.

## Yükleme hazırlığı

`balloons.rtp.balloons.json` yan dosyası projeyi bu davranışa açıkça dahil eder.
Player, projeyi yüklerken ortak `Paint::balloonPaintCommand` hizmetini çağırır.
Yan dosya yoksa demo kapalı kalır ve önceki projenin ıslak boya yüzeyleri temizlenir.
Yan dosya hatalıysa proje açma çağrısı açıklayıcı hata döndürür; kısmi demo başlamaz.

Dört yüzey aynı ana iş parçacığı işlemi içinde eklenir. CPU UV→dünya atlasları
hazırlanır, ardından tek GPU layout kurulumu yapılır. Sonraki patlamalar yüzey
eklemez ve atlası yeniden kurmaz. Atlas geometrisi doğrudan `TriangleMesh` /
DNA konum, normal, UV ve materyal dizilerinden okunur. Bir tek uyumluluk `Triangle`
nesnesi mevcut materyal/doku adapter'ına verilir; yüz koleksiyonu geometri kaynağı
olarak kullanılmaz. Akma hesabı ortak `SurfaceFlowField::computePixelFlow` içindedir.

CPU/GPU damla havuzları önceden ayrılır; en büyük aynı renkli üçlü patlama
kadar geçici damla ile ballistic/collision/slot-record yolu ısıtılır ve damlalar
oyun başlamadan kaldırılır. GPURequired hazırlığı başarısızsa sessiz CPU geçişi
yerine yükleme hatası alınır. Boş havuzdan sonraki ilk doğumun mevcut tam SoA
yüklemesi hâlâ olabilir; bu çalışma GPU yayın/istatistik beklemelerini değiştirmez.
Balonların grafik pipeline'ı ilk balonlar çizilirken devreye girer; render tarafındaki
ilk kare maliyetinin tamamen kaldırıldığı iddia edilmez.

Hazırlık yükleme ekranının maliyetidir; bütün hazırlığın süresi `prepare_ms`,
atlas bölümünün süresi `atlas_ms` olarak ayrı raporlanır. Bunlar oynanış kare
maliyetine dahil edilerek karşılaştırılmamalıdır. Shader/bağlama/push constant
değişmedi; shader derlemesi gerekmiyor.

## Script ve IPC

İki yüz aynı API ve çekirdeği çağırır:

```python
# Studio script: bilgi okumak veya hazırlanmış demoyu elle tetiklemek
rt.paint.wet_demo(operation="status")
rt.paint.wet_demo(operation="stop")  # yalnız otomatik patlama takvimini durdurur
rt.paint.wet_demo(operation="burst", config={"count": 3})
rt.paint.wet_demo(operation="start")
```

IPC eşdeğeri: `paint.wet_demo`, parametreler `operation` ve `config`.
`configure` config'i fixture yan dosyasıyla aynı nesnedir ve boyayı/havuzları
yeniden hazırlar. `status`, `start`, `stop` boş config alır. `burst` yalnız integer
`count` 1..3 kabul eder; hazır balon sayısı daha azsa mevcutları patlatır.
Hazırlık öncesinde start/burst reddedilir; yanlış işlem, alan, tip ve aralıklar
aynı hata metniyle script/IPC'ye döner. Bu komut uç noktası SceneWrite ister;
`status` işlemi GPU'dan veri indirmez veya GPU beklemez.

Config: dört farklı flat mesh adı `surfaces`; `resolution` 32..2048;
`seed` 0..2147483647; `balloons` 3..24; `particles_per_balloon` 32..8192;
bu ikisinin çarpımı en çok 131072; `2 <= interval_min <= interval_max <= 30`
saniye; boolean `autostart`. Patlama şekli (2026-11-18): `burst_clumpiness`
0..1 (vars. 0.7; 0 = eşit ince sprey) ve `burst_jets_max` 3..24 (vars. 10) —
her patlama yırtılma yönü, eşitsiz (gamma) kopma bağı payları, 1..8'lik damla
kümeleri, 1..3 ağır blob ve sprey payını yeniden çeker (`Paint::balloonBurstPattern`).
Islak boya iz yarıçapı sabit olduğu için "iri damla" birlikte uçan kümedir. Sahne dört `BalloonPaintRed/Blue/Yellow/Green`
sistemini, her birinde dört analitik çarpıştırıcı ve sıfır domain ile taşımalıdır.
Yerleşim aralığı bu 8×8 m fixture'a aittir; genel oda sınırı editörü değildir.
Mevcut boya dokusunun boyutu istenen çözünürlükten farklıysa configure açıkça
reddedilir; temiz projeyi yeniden açmak gerekir.

## Maliyet sıçraması ölçümü

`events` son 128 patlama grubunu tutar. Her olayda id, oyun zamanı, konum/renkler,
eşzamanlı balon sayısı, doğan damla sayısı ve `spawn_ms` vardır. İzleyen üç oyun
saniyesinin **her simülasyon tick'inde** `simulation_ms_max`, `paint_ms_max`,
`frame_interval_ms_max`, `frames` ve `sources_dropped` güncellenir. Yakın olayların
pencereleri örtüşebilir; maliyetin yalnız bir balona ait olduğu iddia edilmez.

`simulation_ms_max` ortak simülasyon saatinin CPU duvar süresidir. Player'ın sonraki
doku yayını, render ve present işlemlerini içermez. `frame_interval_ms_max` saate
verilen dt'dir; gerçek GPU süresi veya kesin present aralığı değildir. Manuel IPC
patlamasının doğum maliyeti ayrı `spawn_ms` alanındadır; otomatik patlamada aynı
doğum ayrıca simülasyon saatinin içindedir. Genel kare maliyeti için
`player.display` de rapora alınır; bu ortalama en kötü tek kare yerine geçmez.

`pools` CPU/resident kapasitesi, canlı sayısı ve son GPU durumunu gösterir.
Yoğun patlamada mevcut 64-kaynak/tile tavanı hâlâ boya düşürebilir; sayaç raporlanır,
düzeltildiği iddia edilmez. `publish_host_downloads` sıfır olmalıdır.

## Build ve canlı kabul

1. Kullanıcı **RayTrophiPlayer Release x64 artımlı build** alır. Yeni kaynak iki
   VS projesinde kayıtlıdır. Studio da aynı çekirdek ve Python bağlamasını derler;
   ortak yaygın struct veya shader ABI değişmedi.
2. Yeni Player açılır. File/Open ile
   `docs/dev/fixtures/balloon-paint-room/balloons.rtp` açılırsa yan dosya otomatik
   hazırlanır ve balonlar patlamaya başlar.
3. Ayrı terminalde:

   ```powershell
   python scripts/test/rt_player_balloon_paint_ipc.py --seconds 30
   ```

   Betik eski exe'yi sahneyi değiştirmeden reddeder. Yeni exe'de yükleme hazırlığını,
   gerçek 1024² texel sayılarını, tekli/üçlü doğumu, 4 saniyede damlaların bitmesini,
   sadece 12 balonun kalmasını, geçersiz isteklerin reddini, rastgele olayları,
   GPU boya kapsamasını ve host'suz yayını kontrol eder. Sonunda otomatik demo
   çalışır kalır; başlangıç ekran pacing/continuous ayarı geri getirilir.
4. Geçici JSON kabul raporu `docs/dev/.captures/balloon-paint-room/acceptance.json`.
   Ekran görüntüsü üretilmez. Gerçek görsel kabulde tek renkli balon, patladığı
   anda kaybolma, fiziksel saçılma, leke ve yeniden doğum kullanıcıyla doğrulanır.

Fixture intentional test asset'tir; orijinal sprey odasının `.bin` / `.shared`
flat geometrisi byte-identical kopyalanmıştır. Yeniden üretmek için mevcut kaynak
fixture ile `python scripts/test/rt_player_balloon_paint_ipc.py --make-fixture`.

Statik doğrulama: MSVC `/Zs` ile değişen kaynakların sözdizimi; IPC descriptor /
capability denetimi; Player proje generator kontrolü ve include sınırı; Python
sözdizimi, fixture collider/emisyon/sidecar hash kontrolleri. Bunlar uygulama
build'i, link veya yeni demonun canlı kabulünün yerini tutmaz.

## Canlı kabul — kullanıcı build'i, 2026-11-15

`rt_player_balloon_paint_ipc.py` kabul raporu oluştu: hazır atlas, dört gerçek
1024² yüzey, 12 tek renkli balon durumu, 1800/5400 damlalı tekli/üçlü patlama,
4 saniyede damlaların bitip 12 balon kalması, yanlış count tip/aralıklarının
reddi, rastgele patlamalar, kapsama ve GPU içi yayın kontrolleri PASS.
Görsel görünüm bu turda ayrıca ekran görüntüsüyle incelenmedi; bu sonuç IPC kabulüdür.

Yükleme RPC 959.71 ms; demo hazırlığı 352.04 ms, onun atlas bölümü 317.21 ms.
Bu süreler birbirinin alt bölümleri: toplanmaz. Başlangıç maksimum dt'sindeki
959.67 ms de aynı yükleme aralığıdır, patlama penceresi değildir.

| Kontrollü olay | Doğan damla | CPU doğum | En kötü boya tick | En kötü simülasyon saati | En büyük giriş dt |
|---|---:|---:|---:|---:|---:|
| Tek balon | 1800 | 0.51 ms | 3.46 ms | 4.85 ms | 10.81 ms |
| Üç balon (üçü de yeşil) | 5400 | 3.47 ms | 3.92 ms | 4.14 ms | 10.27 ms |

Bunlar farklı üç saniyelik pencerelerin maksimumları; üçlü patlamanın her zaman
tekliden ucuz olduğu sonucu çıkarılamaz. İlk kabulün sonunda 10 olay, %41.19
atlas kapsaması, 159.57 rendered fps, düşen boya 0, host'a indirilen boya 0 B.

Uzun koşu snapshot'ında 170.82 oyun saniyesi / 46 olay, hata ve device loss yok;
havuz kapasiteleri CPU 21601 / GPU 32768 olarak kaldı. Bu snapshot alınırken
Player idle, render sayaçları 11.3 saniye eskiydi. O dönemi de kapsayan olaylarda
64–72 ms giriş dt ve en fazla 48.15 ms sim / 46.26 ms boya sıçraması var.
Bunları gizleyip bütün koşuya 10 ms sınırı atfetmek veya sadece balon sayısına
bağlamak doğru olmaz. Zamanlayıcı/GPU beklemeleri ayrıca profil gerektirir.

Ardından pacing 0 ile aktif render sırasında 18 saniyelik ek pencere alındı:
beş tamamlanmış tekli/ikili olayda en büyük giriş dt 9.18 ms, sim 4.63 ms,
boya 3.40 ms; son render penceresi 167.98 fps. Görülen olaylarda kaynak kaybı 0.
continuous=true istendi, fakat snapshot'ta continuous=false; uygulanan pacing=0,
idle=false ve render sayaçları günceldi. Ayarlar ölçüm sonunda geri getirildi.

Sonuç: bu kabulde bir saniyelik hazırlık patlamalarda tekrarlanmadı; yüklemeye
alındı. Oynanışın her koşulda sıçramasız olduğu veya mevcut GPU beklemelerinin
kaldırıldığı iddia edilmiyor. Ürün kaynaklarında bu kabul turunda değişiklik yok.
Kısa/uzun ham ölçüler: `PLAYER_BALLOON_PAINT_ACCEPTANCE_2026_11_15.json`.

## Jolt Soft Body prototipi

Mevcut mesh balon demo ayarlarına `"soft_body": true` ekleyin. Alan isteğe bağlı
ve bool türündedir. Hem sahnenin demo ayarında hem de aynı servise bağlanan
Python `rt.paint.wet_demo("configure", config)` ve IPC `paint.wet_demo`
çağrılarında kullanılabilir. `true` için `visual_nodes` gerekir.

Taşıma sırasında duvar basıncı oluştuğunda tek bir kapalı Jolt kabuğu, temas
eksenine göre iki plaka arasında 60 Hz fizik adımıyla sıkışır. Kabuğun boyutları
görünür flat mesh'in instance ölçeğini sürer. Bu prototipte dünya temasını
ve patlama konumunu mevcut taşıma kodu belirler. Balon eldeyken temas kesilirse
ölçek kademeli toparlanır. Yerel yüzey deformasyonu sonraki aşamadır.

Durumda `soft_body_axial_scale` anlık ölçeği, `soft_body_min_axial_scale`
yapılandırmadan beri en küçük değeri, `soft_body_ms_max` ise kabuk güncellemesinin
en uzun CPU süresini verir. Toplam kare maliyeti için aynı sahneyi özellik kapalı
ve açık olarak karşılaştırın.

Windows Player derlemesinden sonra, `PaintBack` içeren geçici balon sahnesinde:

```powershell
python scripts/test/rt_balloon_carry_ipc.py --player --exercise-disposable
python scripts/test/rt_balloon_carry_ipc.py --player --exercise-disposable --soft-body
```

Android Player portu derlenebilir olduğunda aynı sahnede fizik maliyetini ve
kare süresini gerçek bir `arm64-v8a` cihazda ölçün.

## Tüm balonlarla Jolt hareket testi

Mesh balon demolarında Jolt hareketi varsayılan olarak kapalıdır.
`"jolt_motion": true` küresel gövdeleri ve varsayılan çarpışma kabuklarını açar. Python
`rt.paint.wet_demo` ve IPC `paint.wet_demo` aynı ayar ve durum alanlarını
kullanır. `soft_body` açıkken eldeki balonun duvar sıkışması da birlikte çalışır.

Her şişmiş balon Jolt dünyasında küresel bir gövde alır. Gövdeler birbirleriyle
ve hazırlıkta alınan statik flat mesh oda geometrisiyle çarpışır. Serbest
balonlar farklı zamanlarda düşer ve kaldırma kuvvetiyle yeniden yükselir.
Etkin yerçekimi düşük, kaldırma kuvveti kademeli devreye girer ve hafif hava
akımı balonları yana sürükler. Jolt gövdesinin dönüşü sahnedeki balon modeline
uygulanır. Hava akışı serbest balonlara düşük bir dönme momenti verir.

`jolt_motion` açıkken her balon için 18 köşeli basınçlı bir soft-body kabuğu
ayrılır. Zemin, oda ve balon temasları kendi kabuğunu yükler. Kabuğun temas
eksenindeki daralması ve yanlara genişlemesi görünür modelin şeklini sürer.
Kabuğun boyutları instance dönüşümüne uygulanır. Oda ve balon çarpışmalarında
birbirini iten gövdeler küreseldir. `"impact_soft_body": false` ile kabukları
kapatıp önceki görsel sıkışmayla aynı sahnede maliyeti karşılaştırabilirsiniz.
Player'da balonu patlatmadan düşürmek için sol tuşu **Shift basılıyken** bırakın.
Dış istemcide bunun karşılığı mevcut `pointer` işleminin `cancel` eylemidir.
Normal sol tuş bırakma boya patlamasını tetikler.

`status.jolt_motion` aktif balon gövdesi sayısını, fizik adımlarını, yeni
balon ve oda temaslarını ve Jolt hareket güncellemesinin son/en uzun CPU süresini
verir. `motion_ms_max` görsel dönüşüm gönderimini, `soft_body_ms_max` eldeki
balonun sıkışma simülasyonunu ayrı ölçer. Kare süresini özellik kapalı ve açıkken
aynı sahnede karşılaştırın.
`squashes` başlatılan esnemeleri, `peak_squash` en yüksek hedef sıkışma
oranını gösterir.
`soft_shells` ayrılan kabukları, `peak_shell_squash` ölçülen daralmayı,
`shell_step_ms_max` kabuk maliyetinin bir fizik adımındaki tepesini ve
`rotating_bodies` anlık dönen gövdeleri bildirir.

`wind_strength` 0–1 aralığındadır ve varsayılanı 0,25'tir. `wind` işlemi
`{"wind_strength": 0.5}` gibi bir ayarı mevcut balonları yenilemeden uygular.
Player Oyun Kontrolleri aynı ayarı kullanır. Balon Fiziği bölümünde Jolt ve
çarpışma kabukları ayrı ayrı açılıp kapatılabilir.

Yeni demo yapılandırmasında `balloons` varsayılanı 5'tir. Masaüstü ve Android
Player, kaydedilmiş yerel oyun ayarları yoksa projeyi en fazla 5 aktif balonla
ve Jolt kapalı açar. `visual_nodes` aktif sayıyı karşılamalıdır ve 24'e kadar mesh
içerebilir. Artan mesh'ler ayarlardan balon sayısı yükseltilene kadar gizlidir.

Mesh balonlar Studio, Windows Player ve Android Player'da opak hazırlanır.
Oyun, balon malzemesinin opaklık ve transmission dokularını, kabarcık ve reçine
içini ve Material Graph'ın bu iki çıkışını devre dışı bırakır.
Eski proje ve `paint.wet_demo` yapılandırmalarındaki `opaque_balloons` ve
`transmission` alanları okunabilir, ancak uygulanmaz ve yeni yapılandırmaya
yazılmaz. Python ve IPC aynı hazırlık yolunu kullanır.

Üç duvarlı odalarda ön sınır yan duvarların bitiminden 25 cm öteye kurulur.
Oda geometrisinden hesaplandığı için kamera hareket ettiğinde yer değiştirmez.
Sınıra yaklaşan serbest balonlar içeri yönelir. Taşıma da aynı sınırda durur ve
sanal yüzeyde beklemek balonu patlatmaz. `status.jolt_motion.front_barrier`
etkin düzlemin dünya koordinatını ve içeri yönünü verir.

```powershell
python scripts/test/rt_balloon_jolt_motion_ipc.py --exercise-disposable --expect-front-barrier
```

Dağıtımdaki Pastel Paint Workshop açıkken testi çalıştırın. Görsel kontrolde
bir balonu diğerinin üstünde bırakıp teması ve savrulmayı da gözleyin.

