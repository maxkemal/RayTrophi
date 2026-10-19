# Sentetik ses üreticisi: bütün fiziksel olay aileleri

> **Durum:** AKTİF — 2026-10-19. Mevcut sekiz ailede v2 DSP, kaynak ayarları ve
> UI/Python/IPC ilk kod partisi teslim edildi; derleme/dinleme kabulü bekliyor.
> Henüz uygulanmamış olay/veri/modül önerileri aşağıda geleceğe dönük kapsamdır.

Kaynak kontrol noktası ve gerçek API adları:
[Physical generator v2](PHYSICAL_AUDIO_GENERATION.md#2026-10-19-v2-source-delivery-and-source-inspector).
Bu parti S1/S3a'nın bir bölümüdür; yeni solver event stream, kabarcık/sıçrama,
atmosfer/termal generator, tam akustik export veya bütün ailelerde kalite kabulü değildir.

## Ana planla ilişki

Önce [ana audio planının](AUDIO_MASTER_PLAN.md) A0–A2 ortak kullanım/kayıt temeli,
sonra bu belgenin S0–S6 partileri uygulanır. [Kalite kapanış kapsamı](PHYSICAL_AUDIO_QUALITY_CLOSURE.md)
referanstır. Sıvı, gaz, katı, deformasyon/kırılma, atmosfer ve termal olaylar kapsamda;
tek birini seçip diğerlerini ertelemiş saymıyoruz. Uygulama yine test edilebilir
partilere bölünür. Bütün olaylarda tek filtreden geçirilmiş gürültü yeterli kabul edilmez.

## S0: mevcut temel ve uyumluluk

Mevcut `generator.version=1`; liquid_flow, gas_flow, friction, rolling, impact,
contact_activity, deformation ve fracture_activity modellerini doğrular.
PhysicalAudioDsp native data source, scheduled replay ve dry bake tarafından ortak
kullanılır. PhysicalAudioHistory örneklenmiş kontrol/olay geçmişini saklar;
PhysicalAudioTimeline/Schedule sample konumlarını değerlendirir. Bunlar korunur.

İlk uygulama S0'da mevcut sekiz model için golden fixture ve işlem kapsamını
kaydeder. Yeni doğal karakter `version=2` gibi açık bir recipe sürümünde gelir;
v1 proje/seed/bake davranışı korunur. Eski tarifler otomatik çevrilmez. Yeni sürümün
alanları v1'de yanlışlıkla kabul edilmez; sürüm değişimi history uyumluluğunu da denetler.
Kesin sürüm ve alan adları implementasyonla şema/discovery'ye birlikte girer.

## S1: ortak üretici omurgası

Üretici yalnız kaynağın kuru excitation/rezonans sesini üretir. Mesafe, çevre,
iletim ve yansımalar mevcut ortak acoustic/mixer yolunda uygulanır. Aynı etki iki
kez uygulanmaz. Mono noktasal kaynak ile büyük alanın uzamsal kaynak dağılımı ayrı
model kararlarıdır; geniş yağmur/akış alanı tek yanlış merkezde toplanmaz.

Gerekli olduğu ailelerde üç bölüm birlikte çalışır:

| Bölüm | Görev | Giriş |
|---|---|---|
| Sürekli doku | Akış, türbülans, sürtünme, düşük frekanslı gövde | Zamanlı ve yumuşatılmış fizik kontrolü |
| Geçici olay | Damla/temas/darbe/kırılmanın başlangıç ve kısa ayrıntısı | Kimlikli, örnek zamanlı olay veya açık artistik olay üreticisi |
| Rezonans/sönüm | Yüzey/nesne/kap karakteri ve kuyruğu | Uygun fizik parametreleri veya açıkça authored timbre profili |

Bu bölümler ayrı playback sistemi veya sınırsız yeni voice yaratmaz; kaynak içinde
sabit bütçeli DSP durumudur. Gürültü/olay çeşitliliği seed ile deterministiktir;
frame sayısından, UI refresh'ten veya callback blok sınırından yeni rastgelelik türemez.
Sıfır aktivitede yeni excitation biter, mevcut kuyruk belirlenmiş sınırda drain olur.

Önerilen odaklı modüller: `PhysicalAudioRecipeV2`, `PhysicalAudioControlFrame`,
`PhysicalAudioEventStream`, `PhysicalAudioMaterialResponse` ve aile DSP modülleri.
Mevcut `PhysicalAudioDsp` sürüm seçen ince facade olabilir. Adlar mevcut dosya/API
değildir. Validation ve coefficient hazırlama render döngüsünün dışındadır.
Callback'te JSON, allocation, lock, dosya/scene/geometry erişimi veya telemetry
tarama yoktur; sabit kapasiteli durum ve sınırlı sayıda olay işlenir.

## S2: domain ile sürümlü veri sözleşmesi

Audio tarafı diğer ajanın birleşik domain solver/shader/buffer/test koduna müdahale
etmez. Yeni telemetry ihtiyacı bu sözleşme üzerinden domain sahibine entegrasyon
gereksinimi olarak aktarılabilir; şu anda gönderilmiş bir ajan mesajı veya kabul
edilmiş GPU ABI değildir. Mevcut yayın okunur; eksikler `unavailable` kalır.

Önerilen sürekli yayın zarfı:

- `schema_version`, scene/domain/source kimliği ve `epoch`/generation.
- Monoton yayın sequence, simülasyon başlangıç zamanı ve kapsanan süre; saniye.
- Ölçü adı/birimi, değer, ölçülmüş/tahmini/authored provenansı, validity ve nedeni.
- Varsa konum, yön/hız, temas alanı ve Substance kimlikleri; koordinat sistemi ve
  metre/scene-unit dönüşümü açık. Eksik alan fiziksel sıfırla aynı şey değildir.

Önerilen olay zarfı:

- `epoch`, kaynak/domain kimliği ve stabil event identity/sequence.
- Solver'ın olay zamanı; normalleştirmede scene sample konumu. Geliş zamanı farklıdır.
- Event kind, konum, katılan nesne/malzeme kimlikleri ve yalnız gerçekten mevcut
  nicelikler: impuls N·s, dissipated energy J, temas süresi s, hız m/s gibi.
- Enerji, impuls ve güç ayrı alan/anlam taşır. Birbirine örtük çevrilmez; integral
  gerekiyorsa zaman aralığı/provenans gereklidir. Kinetik aktivite akustik watt değildir.
- Sıralama, gecikme, duplicate/overflow sayaçları ve eksik veri politikası.

Sample mapping mutlak solver/scene zamanından yapılır; dt parçalanması aynı olayı
çoğaltmaz. Pause, rewind, scene reset ve seek yeni epoch veya açık yeniden kurulum
ister. Geç olay sessizce “şimdi oldu” diye çalınmaz; tolerans/drop durumu raporlanır.
Kapasiteler ve işlem bütçeleri API'de bildirilir. Fizik enerjisi keyfi şekilde
birleştirilmeden önce coalescing kuralı tanımlanır; kayıp olay miktarı görünür olur.

Fixture/artistik giriş aynı typed validator'u kullanabilir ancak gerçek solver
verisi etiketi taşımaz. Aktivite türevinden otomatik splash/collision/kırılma
olayı, parçacık sayısından kabarcık yarıçapı veya sıcaklıktan yanma olayı uydurulmaz.

## S3: aile modelleri ve gerekli veri

| Aile | Sesin yapısı | Ölçülmüş sürüş için gereken veri / yokluk davranışı |
|---|---|---|
| Sıvı akışı | Değişen türbülans, yüzey/gövde dokusu, uygun kap rezonansı | Akış aktivitesi/hız ve ölçek; yalnız toplam aktivite varsa artistik flow texture |
| Damla/sıçrama | Kısa temas başlangıcı ve farklı yüzey cevabı | Kimlikli sıvı-temas olayı, zaman/konum ve uygun impuls/enerji bilgisi |
| Kabarcık/kaynama | Olay bazlı sönümlü bileşen; akışa ayrı katman | Uygun boyut/yarıçap veya açık authored dağılım; verisizken ölçülmüş bubble modeli sunulmaz |
| Gaz/rüzgâr | Türbülans katmanları, hamleler; uygun durumda jet/islık | Akış/hız/yön ve ölçek/geometri; hız tek başına belirli bir islık frekansı iddiası değildir |
| Darbe/temas | Temas transient'i, malzeme/nesne rezonansı ve kuyruk | Zamanlı temas, çift malzeme, boyut/geometri ve excitation bilgisi; aggregate güç tekil impact değildir |
| Sürtünme | Kayma dokusu, pürüzlü değişim ve gövde cevabı | Slip hızı, temas/normal kuvvet veya dissipated power; eksikte açık artistik doku |
| Yuvarlanma | Yavaş gövde ve mikrotemaslar; kayma sesinden ayrı | Yuvarlanma hızı/ölçek/pürüzlülük ve ilgili fizik kontrolü |
| Deformasyon/kırılma | Sürekli strain dokusu ve ayrı kırılma transient'leri | Mevcut aktivite kontrolü korunur; tekil fracture için güvenilir event/time/provenans |
| Yağış | Yüzey türüne göre damla/çarpma topluluğu ve arka doku | Yerel yağış/yüzey bilgisi; oranla üretilen seeded olaylar artistik/statistical diye etiketlenir |
| Yıldırım/gök gürültüsü | Kanal/uzaklıkla zamanlı darbe ve düşük gövde | Gerçek lightning event/zaman/geometri; mevcut sample yolu sentetik generator diye sunulmaz |
| Termal/yanma/buhar | Uygun süreçte gürültü, patırtı ve akış bileşenleri | Faz/debi/yanma etkinliği ve varsa olaylar; termal veri hazır değilse model unavailable/artistic |

İlk v2 partisi mevcut sekiz modelde ortak doku/transient/rezonans mimarisini ve
uyumlu alanları getirir. Yeni damla, kabarcık, atmosfer ve termal modeller ayrı
aile partileriyle eklenir; bir genel “nature” preset'iyle tamamlanmış sayılmaz.
Hiçbir aile dinleme kabulünden çıkarılmaz. Eksik solver yayını yalnız ilgili
ölçülmüş modu bekletir; fixture/artistik model ve diğer aileler ilerleyebilir.

## Malzeme ve seviye

Kullanıcıya sade malzeme aileleri sunulur; demir/bakır gibi uzun test listesi
zorunlu akış olmaz. Substance profil/override kimliği tek otoritedir. Akustik yolun
absorption/transmission katsayılarıyla kaynağın excitation/rezonans parametreleri
aynı nicelik değildir; biri diğerine doğrudan kopyalanmaz.

Boyut, sertlik/pürüzlülük, sönüm ve uygun malzeme cevabı fiziksel veri veya açık
authored preset olabilir. Varsayılanlar başlangıç karakteridir; ölçülmüş katalog
ya da solved eigenmode diye sunulmaz. PBR metalness/renk sesin fizik otoritesi değildir.

Input calibration, generator output gain, kaynak gain, mesafe ve Master gain
birbirinden ayrılır. Enerji/güç girişini dBFS/SPL diye etiketlemeyiz. Clipping
önleme açık headroom ve ölçümlerle yapılır; her sonucu otomatik normalize ederek
olaylar arasındaki şiddet farkı yok edilmez. Köşe koşulları sınırlanır ve raporlanır.

## S4: zaman, replay ve export

Mevcut history/schedule/bake genişletilir. Yeni event kind ve kontrol alanları
version'lı timeline'da doğrulanır; eski impact-only `energy_j` listesi sessizce
farklı olay sözleşmesine dönüştürülmez. Aynı tarif/seed/geçmiş için platformdaki
DSP sonucu callback blok bölünmesinden bağımsızdır. Floating point platformları
arasında bit-identical çıktı ancak ayrıca kanıtlanırsa vaat edilir.

Geçmiş kaydı; sample zamanı, epoch ve kimlik taşır. UI-sampled take ile substep
solver kaydı ayrı kapsamlardır. Rewind/cache replay aynı kayıtları kullanır;
veri yoksa eksik replay raporlanır. Bounded preroll/checkpoint ile seek worker'da
hazırlanır; stale sonuç yeni kaynağa uygulanmaz. Pause DSP zamanını doğru tutar.

Native ve bake aynı aile DSP'sini çağırır. Kaynak pitch, pozisyon ve acoustic
response zaman serileri dry bake'den ayrı eklenir. Kapsamı desteklemeyen export
sessizce kuru dosya üretip “işlenmiş” demez. Başarıda yeni asset oluşturulur;
kullanıcının kaynak dosyası korunur; iptal/hata geçici çıktıyı bırakmaz.

## S5: UI, Python ve IPC aynı işlemler

Mevcut source create/update/get/list, transport, physical history/replay/bake
operasyonları tercih edilir; sırf UI için ikinci generator registry oluşturulmaz.
Yeni yetenekler core'un model/kontrol/event türlerini ve kapasitelerini bildiren
bir catalogue/discovery sözleşmesiyle açılır; kesin operasyon adları uygulamada
mevcut servis/descriptors ile birlikte kararlaştırılır.

Her yeni model/alan için aynı partide:

1. Core doğrulama, sürüm/limitler, atomik hata ve migration.
2. Ortak servis işlemi, script binding, IPC dispatch/schema ve capability.
3. Runtime durumları: supported/unavailable, input age/provenans, epoch/sequence,
   queue/drop, draining/active, peak ve kullanılan playback/export kapsamı.
4. Mevcut inspector/Designer bağlamında üretici ayarları; kaynak adı/kimliği,
   measured/artistic ayrımı, temel kontroller ve katlanır ayrıntılar.
5. Kabul probe'u, documentation/discovery ve runtime test script mirror'u.

UI'da solver etkinleştirme veya telemetry toplama örtük yapılmaz. Ses oynatmak,
fizik simülasyonunu başlatmak ve veri kaydı ayrı kullanıcı işlemleridir.

## S6: kabul matrisi ve teslim partileri

| Parti | Teslim | Kabul kapısı |
|---|---|---|
| S0 | v1 envanter/golden fixture ve v2 sınırları | Eski proje/recipe/seed sonucu korunur |
| S1 | Ortak v2 recipe, typed kontrol/olay, bounded DSP omurgası | Şema/validation, deterministik block split, allocation/ownership incelemesi |
| S2 | Salt okunur domain adapter ve epoch/zaman/provenans | Sahte veri yok; missing/stale/dedup/overflow/reset davranışı |
| S3a | Mevcut sekiz modelde v2 doğal karakter | Her modelde düşük/orta/yüksek, başlangıç/duruş, malzeme/boyut dinlemesi |
| S3b | Damla/sıçrama/kabarcık, atmosfer ve termal aileler | Her yeni model için gerekli veri ve ayrı sayısal/işitsel sahne |
| S4 | Yeni geçmiş ve native/replay/bake kapsamı | Aynı inputla PCM tutarlılığı; seek/pause/reset ve export sınırları |
| S5 | Tam UI/Python/IPC ve kalıcılık | Aynı alan/hata davranışı, save/load/undo/duplicate, tarif geçişi |
| S6 | Bütünleşik kalite ve performans | Bütün aileler, 1/8/16/32 kaynak, dry/akustik yollar, uzun kullanım |

S1–S5 yatay teslim koşullarıdır; her model partisi kendi API/UI/test ayağıyla gelir,
script/IPC'yi sona bırakmaz. Veri hazır değilse neden açık görünür ve o kabul kapısı
açık kalır. S6'da bütün aileler tekrar birlikte kontrol edilir.

Gerekli sahneler: kapta/serbest akış ve damla yüzeyleri; gaz/rüzgâr değişimi;
farklı malzemelerde darbe/kayma/yuvarlanma; deformasyon/kırılma; yağış/yıldırım;
uygun termal olaylar. Her sahnede yakın/uzak, düşük/orta/yüksek, örtüşen olaylar,
sıfır aktivite, stale/missing input ve değişen malzeme test edilir.

Sayısal kabul; finite ve bounded PCM, sessizlik/kuyruk, olay onset/dedup,
partition invariance, enerji/seviye davranışı ve history kapsamını kapsar. İşitsel
kabul; tekrar hissi, yapay ton/klik, olayla zaman/timbre uyumu ve malzeme farkını
kapsar. Kabul edilmiş bir aile bütün sistem kalitesinin yerine geçmez.

Codex derleme veya Studio başlatma yapmaz. Kullanıcı build'inden sonra dış IPC
fixture'ları ve kullanıcının dinlemesi birlikte kaydedilir. Diğer domain ajanının
aktif sahne/testlerine bu plan için müdahale edilmez.
