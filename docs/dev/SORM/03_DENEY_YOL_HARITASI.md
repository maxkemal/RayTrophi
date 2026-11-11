# SORM — Deney yol haritası v0.3 — 11.11.2026

**Protokol:** Her koşuda RNG tohumu, model sürümü, giriş dizisi, parametreler, zaman adımı, ham yanıt, ölçüm ve başarısızlık saklanır. Pilot için ≥10 tohum; sonra güç analizi. Plastisite kapalı ve eşleştirilmemiş kontroller zorunlu.

| Aşama | Model | Deney | Geçiş kapısı |
|---|---|---|---|
| 0 — Tanım | Önce M0; sonraki modeller kendi kapısında | Denklemler, boyut ve ilgili enerji hesabı | Uygulanacak model için tutarlı sözleşme |
| 1 — İlk uyarı | M0, 1k | A/B, sıfır giriş, sönüm; eşzamanlı sahne gözlemi | Kararlı bazal yanıt ve gözlem açık/kapalı eşdeğerliği |
| 2 — İz | M1 vs aynı M1'in donmuş-iz kontrolü; ek M0 | A/B tekrar, aralıklı öğrenme, unutma | Hızlı durum eşitlendikten sonra geçmiş etkisi kontrolün üzerinde |
| 3 — Seçicilik | M1/M2 | Tepki yayılımı, gecikme, maliyet | Ayırt etme iyileşmesi; maliyet neyse rapor |
| 4 — Tamamlama | M1/M2 | Bozuk/kısmi A, görülmemiş bozunum | A referansına kontrole göre yaklaşma |
| 5 — Çağrışım | M1/M2 | A+B eşleştirme, sonra A-only | B etkisi eşleştirilmemiş kontrolün üzerinde |
| 6 — Gruplar | M1/M3 | Topoloji, hasar, yeni görev | İşlev sürekliliği/yeniden örgütlenme |
| 7 — Ölçek | Seçilen | 1k→10k→100k, CPU/GPU | Kalite ve maliyet eğrileri |
| 8 — Duyular | Seçilen | RayTrophi görüntü/ses/fizik olayları | Çoklu modal çağrışım ve kontrol |

## Her aşamanın uygulama döngüsü
1. Bazal koşuyu çalıştır; ham telemetriyi kaydet.
2. Aynı uyarıları öğrenme açık/kapalı koşullarda uygula.
3. Tekrar sayısını, beklemeyi, enerji bütçesini ve gürültüyü değiştir.
4. Görülmemiş bozulmuş uyarıları dene.
5. Ablasyon yap: iz, gevşeme, gürültü, uzak etkileşim ve topoloji.
6. Çoklu tohum istatistiği, başarısızlık ve kararsızlığı raporla.

## Durdurma koşulları
A/B ayrışmıyorsa, ağ tümüyle senkron kilitleniyorsa, öğrenme yalnızca harici okuma katmanında oluşuyorsa, enerji bilançosu bozuksa veya sonuç sabit sensör girişinden açıklanabiliyorsa sonraki aşamaya geçilmez. 1k/10k doğrulama olmadan 100k optimizasyonu yapılmaz.

M0'da giriş geometrisinin açıklayabildiği A/B farkı bazal ölçüm olarak kaydedilir.
Öğrenme aşamasında durdurma nedeni, iddia edilen ek öğrenme etkisinin bu bazal
farktan veya hızlı durum kalıntısından ayrılamamasıdır. Enerji bilançosu kapısı
M2 içindir; M0/M1'e tanımlanmamış fiziksel enerji ölçüsü uygulanmaz.

## K0/K1: İlk 1.000 parçacıklı protokol

**Kesin uygulama referansı:** [M0 v1 sözleşmesi](07_M0_SOZLESMESI.md).
Aşağıdaki adımlar genel özettir; sayısal seçimler ve tam kol takvimi 07'dedir.

Bu teslim düzeneği doğrular; plastisite veya öğrenme başarısı iddia etmez.

1. `02_MATEMATIK_MODELLER.md` içindeki M0 seçimi: periyodik 10×10×10 sabit kafes,
   altı komşu, ortak parametreler ve sıfır etkinlik. Komşuluk öğrenilmiş ilişki
   değil etkileşim olanağıdır.
2. A/B için eşit süre, genlik dağılımı ve toplam giriş bütçeli iki farklı uzamsal
   alan belirle; alanları kaydet. Çekirdek yalnız sayısal giriş alır, sembol
   etiketleri deney düzenleyicisinde kalır.
3. Başlangıç gevşemesi → sıfır/A/B probları → tekrar dizisi → girişsiz bekleme →
   aynı problar. Sunum sırasını dengele. Karşılaştırılabilir probları aynı
   checkpoint'ten dallandır; sıralı probların etkisini öğrenme sayma. Süreler τ_x
   biriminde, ilerleme tam adım sayısıyla tanımlanır.
4. CPU çift duyarlık, Heun ve gürültüsüz koşuyla başla. Δt=τ_x/100 adayını aynı
   süreli Δt/2 ve Δt/4 koşularıyla doğrula. Gürültü ayrı sürümlü deneydir.
5. Ham yanıtı sabit aralıkla kaydet. Pencere, uzaklık, eşik ve sıfır yanıt
   davranışını önceden sabitle. Tüm birimler ile doğrudan giriş alanı dışında
   kalan birimlerin sonuçlarını ayrı raporla.
6. A-A/A-B uzaklığı, N_eff, doygunluk, etkinlik integrali ve hesaplama süresini
   kaydet. M0'daki ayrışma öğrenme değildir.

M0 katsayıları, uyarı alanları, süreler, örnekleme aralığı ve sayısal toleranslar
07'de kesinleştirildi. Gerçek manifest/uygulama K1'de oluşturulacak; henüz koşu yok.

### K1 kabul kapısı

- Sonlu ve sınırlı durum; sıfır girişte modelin beklenen gevşemesi.
- Aynı ortam/model/manifest/RNG ile deterministik tekrar; platformlar arası bit
  eşitliği ayrıca doğrulanmadan vaat edilmez.
- Eşit süre ve girişle adım küçültmede önceden tanımlanan normlarda yakınsama.
- Kesintisiz koşu ile checkpoint'ten devam eşdeğerliği.
- Batch CLI, Python ve dış IPC'de aynı çekirdeğin aynı sonuçları üretmesi.
- Geçersiz istekte durumun değişmemesi; tamamlanmış adımda iptal/checkpoint;
  başarısız veya eksik çıktının başarıdan ayrılması.

## K2: Geçmiş izinin nedensel kontrolleri

Tek bir M1 adayı önce matematiksel olarak tamamlanır. Aynı başlangıç ve aynı
uyarı dizisinden eşleştirilmiş koşullar üretilir:

| Koşul | Ayırdığı etki |
|---|---|
| Plastisite açık / aynı hızlı denklemde h donmuş | Plastisitenin ek katkısı |
| Eğitimli / eğitimsiz | Maruziyet geçmişi |
| Eğitimli h korunmuş, x ve hızlı geçici durumlar eşitlenmiş | Yavaş iz / hızlı kalıntı |
| Eğitimli sistemde h başlangıca döndürülmüş | Etkinin izde taşınması |
| h dağılımı korunarak düğümler arasında karıştırılmış | Toplam iz miktarı / ilişkisel yerleşim |

Karıştırılmış iz nedensel analiz müdahalesidir; fiziksel süreç iddiası değildir ve
geometriye bağlı dağılımı da bozabileceği belirtilir. Problarda plastisitenin
açık/donmuş olduğu kaydedilir. Donmuş prob yeni öğrenmeyi engeller; sürekli
plastisite ayrı deney olabilir.

Çağrışımda eşleştirilmiş/eşleştirilmemiş koşulların maruziyeti, giriş bütçesi ve
bekleme süreleri eşitlenir. B referansı bağımsız B-only problarından sabit yöntemle
çıkarılır. Tamamlama testleri görülmemiş bozulmalar kullanır; bozulum miktarı ve
kalan giriş bütçesi kontrol edilir.

## Kayıt ve istatistik sözleşmesi

- Manifest/model/çıktı şeması sürümü, kaynak revizyonu ve dirty ağaç bilgisi;
  gerektiğinde ilgili kaynak hash'leri veya patch'i arşivlenir.
- Geometri, giriş sırası ve dinamik gürültü için ayrı RNG akışları; algoritma,
  tohum, gerçekleşen giriş dizisi ve checkpoint'te RNG durumu saklanır.
- Parametre, gerçek adım sayısı/süresi, ham x ve kullanılan yavaş durum,
  komşuluk, kontrol/ablasyon, ölçüm, hata ve durdurma nedeni kaydedilir.
- Pilot ≥10 bağımsız tohumdur. Aynı deterministik kafeste kullanılmayan tohumu
  değiştirmek bağımsız tekrar üretmez; değişen rastgele etken belirtilir.
  Gürültüsüz deterministik kabul koşuları ayrı değerlendirilir.
- Tek tohumdaki problar bağımsız örnek sayılmaz. Kontrol farkları tohum içinde
  eşleştirilir; etki büyüklüğü, belirsizlik ve başarısızlıklar raporlanır.
- Geliştirme tohumları son değerlendirmeden ayrılır. Pilot sonrası güç analizi
  yapılır; metrikler ve başarı eşiği doğrulama öncesinde dondurulur. Çoklu
  aday/parametre denemeleri gizlenmez.
- Düşük etkinlik/maliyet ancak korunmuş görev başarısıyla yorumlanır. GPU çalışma
  süresi, CPU enqueue süresi ve aktarım maliyeti ayrılır.


## K1-G: CPU M0 ile erken görsel laboratuvar

K1 sayısal çekirdeği çalışır çalışmaz aynı teslim içinde sahne gözlemi eklenir;
GPU compute veya K7 beklenmez. İlk görünüm 1.000 sabit konumlu birim, işaretli
etkinlik renkleri, giriş alanı ve açılıp kapanabilen sabit komşuluk çizgileridir.
Uyarı başlangıcı, simülasyon zamanı/adımı, deney evresi ve lejant görünürdür.

Duraklatma, tek sayısal adım, sunum hızını değiştirme ve kaydedilmiş durumları
zaman içinde inceleme desteklenir. Sunum hızı Δt'yi değiştirmez. Geçmiş kaydı
izlemek canlı çekirdeği geri sarmaz; yeniden deney için açık checkpoint işlemi
kullanılır. Tekrar sayısı, A/B sunumu ve sayısal kontrol aynı servis üzerinden
UI/Python/IPC'de erişilebilir; görsel ayarlar bilimsel parametrelerden ayrılır.

K1'de enerji/öğrenilmiş bağ/uzmanlaşma varmış gibi görünüm üretilmez. M1'de iz ve
etkili ilişki katmanları; M2'de enerji/akı; M3'te bağ oluşum/kopma eklenir. Küme
analizi sayısal tanımı kesinleşince görünür olur. Eğitim öncesi/sonrası ve kontrol
karşılaştırmaları aynı ölçek, kamera ve prob zamanında yapılır.

Ek kabul kapısı: aynı manifestle görünüm açık/kapalı, farklı kamera, renk ölçeği,
çizgi filtresi ve çizim hızlarında aynı adımlardaki canonical durum/sonuç eşittir.
Snapshot adımı ile telemetri adımı eşleşir; atlanan görsel kare varsa gösterilir.
Yavaş çizim yalnız sunumu yavaşlatabilir, bilimsel adımları düşüremez. GPU çözücü
olmaması GPU raster çizimi kullanmayı engellemez. Canlı görsel kabul, kullanıcı
uygulamayı o turda açıkça çalıştırmayı istediğinde yapılır.
