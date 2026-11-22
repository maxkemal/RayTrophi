# Gizli oyun objeleri — 2026-11-20

Player > Oyun ayarları > Gizli objeler: sahnedeki boya ile beliren statik objeler
adlarıyla listelenir; toplam ve etkin sayı gösterilir. `reveal_opacity=true` olan
wet paint objeleri otomatik bulunur. En fazla 64 farklı obje yönetilir.

Her obje için görünürlük:

- `paint`: başlangıçta saydam, boya geldikçe belirir.
- `shown`: opacity reveal kapalı; mevcut materyali ile sürekli görünür.
- `hidden`: obje gözden ve turdan çıkar; rastgele yerleştirmeye katılmaz.

`randomize_on_reset` varsayılan true. Reset, seçili etkin objeleri varsayılan olarak **zemine oturtup duvarlara
yakın** taşır. `room.placement=walls_floor` varsayılandır; önceki serbest XYZ
yerleşimi `free` ile isteğe bağlıdır. Duvar bandı `wall_band` (0.1..2 m, default
0.65), kameraya obje kutusundan en az uzaklık `camera_clearance` (0..10 m,
default 3) panelden/script/IPC ayarlanır. Böylece Y kendiliğinden yükselmez ve
obje kameranın içine gelmez. Objenin bütün AABB köşeleri sınır içinde kalır.
Y dönüşü başlangıç tasarım matrisine göre uygulanır; birikmez. Varsayılan
-15..15 derece; izin verilen düzenli aralık -45..45. Ölçek ve mevcut eğim korunur.
Kapalı taşıma seçeneği objeyi olduğu yerde tutar. Gizli mod kayıtlı objeyi listeden silmez.

Düzenlemeler Apply/Uygula veya Save/Bu oyun için kaydet ile ortak servise gider.
Save, Player'ın oyun bazlı `player_game_settings.json` profiline yazar; .rtp'yi
değiştirmez. Studio'da script/IPC ile yapılan tasarım, normal proje kaydında
`game_design.hidden_objects` altında saklanır. Anlık reset sayacı kaydedilmez.

## Geçiş ve boya birbirinden bağımsız

**Tasarım kararı:** gizli objeler oyuncunun avatarına veya kamera kartına engel
olmamalı. `blocks_player=false` ve `blocks_balloon=false` varsayılandır.
Böylece hem oyuncu hem taşınan balon içlerinden geçer; boya damlasının mevcut
PaintCollider'ı objeyi boyamaya devam eder. Bu iki izin boyama collider'ını kapatmaz.

İstenirse obje bazında `blocks_player=true`: düz dünya XZ hareketi, obje dünya
AABB'si etrafında 25 cm yarıçaplı dik oyuncu proxy'siyle sınırlanır. Kesintisiz
slab sweep büyük koordinat atlamalarında geçişi önler, iki sınırlı eksen kayması
kenarda yürümeyi sağlar. BodyPoseRuntime bütün eklemleri birlikte kaydırır;
avatar ve el jestleri aynı düzeltilmiş dünya pozunu kullanır. Kamera kartının
çizim ankrajı aynı servise bağlıdır; 40-byte shared-memory protokolü değişmedi.
Bu, isteğe bağlı basit AABB engel politikasıdır; genel fizik veya adım/tırmanma sistemi değildir.

`blocks_balloon=true`: mevcut üçgen tabanlı taşıma çarpışmasına katılır.
Yönetilmeyen salon geometrisinin eski balon çarpışması korunur. Gizli/tur dışı
objeler iki izin true olsa da engel olmaz. İzin değişince mevcut taşıma güvenle
iptal edilir; yeni tutuş yeni engel listesini alır. El takibi kapatılmaz.

## Güvenli reset ve maliyet

Otomatik salon tanımı flat mesh adlarından gelir: `PaintBack`, `PaintLeft`,
`PaintRight`, veya `Room_Wall`, `Room_Floor`, `Room_Ceiling` ile biten adlar.
Gerekirse panelde manuel min/max XYZ belirlenir. Tanınan duvarların iç yarı
uzayları manuel kutuda da korunur. Dışa taşma payı 2.5 cm'dir.

Yerleşim bütün objeler için önceden planlanır: obje başına en fazla 64 deneme;
görünür sabit eşya ve diğer yerleştirilen objelerin AABB'leriyle çakışma reddedilir.
Engel izni açık objelerde mevcut takip edilen oyuncu/balon örtüşmesi de reddedilir.
Yer bulunamazsa bütün konumlar korunur; panel/status hata gösterir. Normal oda
resetinin boya/balon temizliği yine gerçekleşir. Henüz pişen boya collider'ı varsa
konum değişimi reddedilir. İçbükey/karmaşık salonlarda yarı uzaylar fazla tutucu
olabilir; otomatik yerleşim yerine sabit konum veya uygun manuel sınır kullanılabilir.

Yalnız statik, transient olmayan, skin içermeyen flat mesh desteklenir. Animasyon
veya simülasyon dünya pozu authored matristen farklıysa taşıma reddedilir.
Kaynak geometri DNA SoA/original positions + indekslerdir. Per-face Triangle
koleksiyonu kullanılmaz. Reset dönüşümleri balon hareketi gibi flat transform handle + backend instance
yolundan geçer; her reset CPU vertekslerini yeniden pişirmez veya undo kaydı
üretmez. Normal authoring dönüşümü API'si değişmedi.

Reset sonrası mevcut boya collider türü ve yerel cooked SDF korunur. Yalnız
konum/Y dönüşü değişince SDF'nin l2w/w2l matrisleri mevcut kaynak çözümleyiciden
yenilenir; collider silinip dört sistemde tekrar pişirilmez. Ölçek değişirse
eski akışla recook gerekir. Sonraki wet paint tick'i geometriyi hemen kontrol eder;
atlas konumlarının eski yerde yarım saniye kalması önlenir. Bu maliyet reset anındadır.
Yeni shader veya shader ABI değişikliği yoktur. Çalışma anı maliyeti canlı test bekliyor.

## Ortak script / IPC işlemleri

Python örneği:

```python
rt.paint.wet_demo("hidden", {"operation": "status"})
rt.paint.wet_demo("hidden", {"operation": "update", "object": "HiddenRabbit",
    "visibility": "paint", "randomize_on_reset": True,
    "yaw_min": -15, "yaw_max": 15,
    "blocks_player": False, "blocks_balloon": False})
rt.paint.wet_demo("hidden", {"operation": "bounds", "mode": "manual",
    "min": [-3, 0, -3], "max": [3, 3, 3]})
rt.paint.wet_demo("hidden", {"operation": "randomize"})
```

IPC: `paint.wet_demo` operation=`hidden`, config=aynı sözlük.
Player operator: `game.hidden.status|update|bounds|configure|randomize|player_anchor`
aynı çekirdeğe gider. `game.settings.values.hidden_objects` tam tasarım snapshot'ıdır;
`game.settings.apply/save` nested snapshot'ı doğrular ve uygular. UI ayrı mantık içermez.

`configure`: `{operation:configure, config:{version:1, room:{mode:auto}, objects:[...]}}`.
Objelerde yukarıdaki alanlara ek olarak 16 sayılık row-major `base_transform`,
`paint_collider_mode:none|aabb|obb|mesh_sdf`, `sdf_resolution_mode:0|1|2` bulunur.
Update yalnız belirtilen obje alanlarını değiştirir. Snapshot tamamen değiştirilir;
reveal yüzeyleri otomatik keşfedildiği için kaydı silmek yerine `visibility:hidden` kullanılır.

Bilinmeyen alan, eksik obje/yüzey, yinelenen ad, bool yerine sayı, sonlu olmayan
koordinat/matris, singular matris, ters/sınır dışı yaw veya sırasız oda kutusu hata
verir. Giriş önce bütünüyle doğrulanır; hatalı patch tasarımı değiştirmez.
`player_anchor` test ve alternatif giriş adaptörleri için aynı geçiş servisini açar:
owner (1..64 karakter), position XYZ, radius (.05..2), height (.2..5); kabul edilen
position döner. En fazla 32 geçici ankraj geçmişi tutulur; proje/izin/reset temizler.

## Derleme sonrası kabul

Kullanıcı Studio ve Player artımlı C++ derlemesini yapar. Altı yeni cpp her iki
vcxproj'ye EHsc ile kayıtlı; CMake mevcut recursive glob üzerinden bulur.
Bu görevde C++ build, uygulama açma veya canlı sahne değişikliği çalıştırılmadı.
Python AST, EN/TR JSON/tekrarlanan anahtarlar, vcxproj XML/kayıtları ve diff kontrolü PASS.

Hazır sahnenin deneme kopyasında, harici terminalden:

```powershell
python scripts/test/rt_hidden_game_objects_ipc.py --player
python scripts/test/rt_hidden_game_objects_ipc.py --player --exercise-disposable
```

İlk komut salt okunur. İkincisi görünürlük, doğrulama atomikliği, geçiş/engel,
büyük hareket sweep'i, sabit objeler, oda/obje sınırı, birikmeyen yaw, farklı reset
konumu ve imkânsız yerleşimde pozların korunmasını sınar. Metadata finally'de geri kurulur. Studio'da matrisler de geri alınır; Player IPC
yazma politikasını genişletmemek için Player'da test konumları kalır. Boya, undo
geçmişi ve sayaçlar geri alınmaz.
Kamera/sender kapalı, kimse balon tutmuyor ve oyun kuralları çalışmıyor olmalı.
Gerçek shader görünümü, avatar/kart hareketi, SDF cook süresi ve yeniden açılışta
Player kaydını ayrıca gözle doğrula. Named-pipe erişim hata 5 ise sandbox yetkisiyle
aynı harici istemciyi yeniden çalıştır; embedded script workspace kullanma.


## Canlı hata incelemesi ve düzeltme — 2026-11-20

Kullanıcının ekran görüntüsündeki kırmızı/mavi havada asılı kümeler, yalnız
"iç yüzey boyası" olarak yorumlanmamalı. Açık Player IPC ölçümünde bir boya
havuzu 151639 alive / `host_consumer_collider_sdf_not_cooked`, diğer üçü zero /
gpu_resident bildirdi. Üç gizli objenin mesh_sdf'si hazır değildi. Bu gözlem
renderdeki kümelerle uyumludur; yeni binary ile parçacık hareket/ömür testi gerekir.
Mevcut set_collider recook denemesinde de üçü ready=false kaldı. Tek obje
yeniden bağlanınca canlı boya havuzu sıfıra düştü; olay yeniden oluşumu test edilmeli.

Duvar/obje boya atlası ayrı ölçüldü: damlaların ömrü beklendikten sonraki reset
ile iki ardışık coverage ölçümü 0/0, sekiz yüzeyin tamamı zero bulundu. İlk
resetten hemen sonraki küçük yeni izler, kullanıcı da patlatmaya devam ederken
gözlendi. Bu, tek başına reset shader'ının bozuk olduğunun kanıtı değildir.

Kaynak düzeltmeleri:

- Hazır olmayan SDF, CPU'nun da kullanamadığı geçici bir collider'dır. Device
  packing zaten bu SDF'yi atlar; eligibility artık bütün havuzu sırf bu sebeple
  CPU yoluna indirmez. Grid yayınlanınca aynı packing otomatik ekler. Shader
  ve descriptor ABI değişmedi.
- Room reset / `paint.wet_reset` mevcut demo damlalarını, ghost'ları ve ortak
  trail havuzunu temizler; atlas tekrar eski damlalarla boyanmaz. Reset tick
  eski collision event'lerini stamp etmeden tüketir.
- Gizli paint objesinin genişletilmiş dünya AABB'sinde başlayacak damlalar
  atlanır. Balonun merkezi kutunun içinde olsa da kutu dışında başlayan
  damlalar üretilir. Ses/balon patlaması çalışır. Bu AABB koruması açık/içbükey
  objelerde bilinçli olarak fazla tutucudur; per-particle üçgen sorgusu veya
  GPU download yoktur. `hidden.paint_origin` operation (position XYZ) aynı
  servisten `{allowed:bool}` döndürür. Gerçek doğan damla sayısı pop events
  `drops` alanında raporlanır.

Dış testler (yeni Studio/Player C++ derlemesi sonrası, disposable sahnede):

```powershell
python scripts/test/rt_hidden_game_objects_ipc.py --player --exercise-disposable
python scripts/test/rt_paint_drop_cleanup_ipc.py --player --exercise-disposable
```

İlk test zemin/duvar bandı, kamera uzaklığı, rigid reset sonrası hazır SDF
korunumu ve iç/dış boya doğuş sınamasını da içerir. İkinci test gerçek pop
sonrası damlaların ömründe bitmesini ve havada damla varken clear ile
damla/ghost/atlas'ın sıfırlanıp sıfır kalmasını sınar. Yeni C++ kodu bu tur
derlenmedi; yalnız mevcut binary üzerindeki teşhis IPC ölçümleri yapıldı.


## F9 yer değiştirmeme düzeltmesi — 2026-11-20

Canlı Player status: round=0, randomize_on_reset=true olan üç obje; hata
`No safe placement found within 64 attempts: 3_Toys_0.001`. F9 çağrısı doğru
yere gidiyor; plan başarısız olduğu için bütün pozlar korunuyordu.

Kaydedilmiş flat RTP8 geometrisi salt okunur incelendi: Cube_1 bütün zemini
kaplıyor ve üstü y=0.033; Cube_2 bütün tavanı kaplıyor, altı y=3.370. Eski
planner duvarların altını (0) zemin kabul edip objenin altını 0.030
yapıyordu. Zemin aynı zamanda mobilya engeli sayılınca 64 adayın tamamı
zemine çarpıyordu.

Yeni hesap oda XZ alanının en az yüzde 90'ını kaplayan, oda yüksekliğinin
yüzde 15'inden ince, alt/üst sınır bandındaki statik slab'ları sınır olarak
tanır. Zemin üstü ve tavan altı kullanılabilir Y sınırlarını daraltır; bu
slab'lar mobilya engel listesine girmez. Kullanıcının manuel kutusu genişletilmez.
Duvar/kamera ve gerçek mobilya kontrolleri korunur. Tanınan adlar ortak
status.room.boundary_slabs alanında UI/script/IPC için görülebilir. PaintFloor
ve PaintCeiling de isim üzerinden oda tanımına dahil edilir.

Dış kabul testi artık doğrudan F9'un çağırdığı `paint.wet_demo reset`
operasyonunu iki kez çalıştırır: round artışı, hata olmaması, farklı pozlar,
zemin üstü, duvar bandı, kamera ve hazır SDF korunumu sınanır. Workshop
fixture'ında Cube_1/Cube_2 slab tanımı ayrıca doğrulanır. C++ derlemesi yok;
yeni planner canlı çalıştırılmadı. Ek IPC obje-listesi sorgusu otomatik onay
incelemesinin kullanım sınırı nedeniyle çalıştırılamadı; izin reddi değildi.


## Hareketten sonra boya yakalamama — 2026-11-20

Kaynak incelemesinde somut uyumsuzluk bulundu: HiddenObjectMotion yalnız
instance transform'unu güncelliyor, WetPaintGpu::paintFaces ve geometryKey
ise eski dünya `P/N` dizilerini okuyordu. checkGeometryNextTick çağrısı eski
P değişmediği için rebake üretmiyordu. Render ve SDF kaynak çözümleyicisi
yeni konumu kullanırken boya atlası eski yerde kalabiliyordu. Bu bir UV
değişikliği değildir; mevcut UV'nin dünya konumu/normal eşlemesidir.

Yönetilen statik gizli objelerin boya bake'i artık canonical local P_orig ve
N_orig + mevcut final transform kullanır. Normaller inverse-transpose ile
taşınır. Unmanaged yüzeylerin mevcut P/N yolu korunur; skin/deform yolu
bu değişiklikle yeniden tanımlanmaz. CPU-only geometri fingerprint'ine
16 matris bileşeni eklendi (8->24 double); simetrik yaw'da AABB/toplam
değişmese de atlas invalidasyonu çalışır. GPU struct/descriptor ve shader
değişmedi. Hafif instance hareketi ve yerel cooked SDF korunumu devam eder.

Dış hidden-object testi artık her F9 eşdeğeri reset sonrası geometry_rebakes
artışını bekler ve yeni obje dünya kutusunu kapsayan facing=false dab
ile ilgili objenin valid texel coverage'ının yüzde 90 üstüne çıkmasını
sınar. Böylece SDF'den bağımsız olarak atlasın yeni konumu yakaladığı
ölçülür; SDF ready kontrolleri ayrıca sürer. Statik kontroller/Python AST
PASS. C++ derleme ve bu yeni canlı test çalıştırılmadı; mevcut otomatik
onay inceleme kullanım sınırı nedeniyle ek canlı IPC denenmedi.


## Boyalı/görünür objeden sonra F9: Vulkan aygıt kaybı — 2026-11-20

Kullanıcı atlas pose düzeltmesini başarılı doğruladı; görünür yapıp boyadıktan
sonraki F9'da device loss bildirdi. x64/Release/SceneLog.txt içinde GPU
`write_invalid`, adres 0x00000002f2600000, takip edilen live/recent buffer
owner bulunamadı ve raster submit device loss var. Bu rapor tek başına
hangi shader/kaynağın hatalı olduğunu kanıtlamaz; canlı crash tekrar edilmedi.

Kaynak incelemesi F9 geometri rebake'inin bütün atlası release/rebuild ettiğini
gösterdi. Sabit boyut/tile sayısında bu artık yapılmaz: yalnız taşınan
yüzeyin pos_valid, flow_normal, tile AABB ve source UV-metric planes4/5
sayfaları mevcut tamponlara yüklenir. Tile surface indeksleri ve byte offset'ler
korunur; renk/output/state/channel/texture handle'ları değiştirilmez. Kaynak
normal metrikleri güncellenir. Gerçek atlas boyutu/yüzey listesi değişikliği
önceki structural rebuild yolunu kullanır. Başarılı geometri patch'inden sonra
mevcut yüzey-aralığı reset/publish yolu kullanılır; normal frame'lerde ek bekleme yoktur.

Reset hareketi öncesinde compute synchronize ve aktif viewport/render
waitForCompletion sınırı eklendi. Bu yalnız oda yerleşimi resetinde çalışır.
Geometri patch öncesinde compute işi de tamamlanır; bütün atlas tamponlarının
tekrar tahsisi ve büyük kanal upload'ları rigid hareket yolundan çıkarıldı.

Ortak stats API/IPC/Python'a `device_rebuilds` ve `geometry_uploads` eklendi:
`paint.wet_status` / `rt.paint.wet_status()`. İlki başarılı yapısal atlas
kurulumlarını, ikincisi başarılı yüzey geometri upload'larını sayar. CPU
WetPaintStats ve WetPaintStatusInfo değişti; GPU descriptor/struct/shader aynı.

Dış hidden regression şimdi shown/paint/shown/paint olmak üzere dört döngüde
önce mevcut konumu boyar, F9 ile aynı reset çağrısını yapar, device_rebuilds'in
sabit ve geometry_uploads'ın artmış olmasını; telemetry device_lost=false
olmasını ve yeni konumda boya yakalanmasını sınar. Python AST, stat alanlarının
core/API/script/IPC eşleşmesi, in-place patch'te buffer create/destroy olmaması,
<2000 satır kuralı ve diff kontrolü PASS. C++ derlemesi, shader derlemesi veya
uygulama başlatma yapılmadı; shader değişmedi. Yeni kodla device-loss kabulü
kullanıcı derlemesinden sonra yapılmalı; bu hata kesin çözüldü denmiyor.


2026-11-20 kullanıcı son in-place atlas/reset düzeltmesinden sonra çok sayıda
Player testi yaptığını ve şimdilik sorun oluşmadığını bildirdi. Bu kullanıcı
kabulüdür; Codex ayrı GPU crash stres testi çalıştırmadı. Son çalışan Release
çıktısı Gizli Renkler tek sahne kamera saha testi paketine alındı.
