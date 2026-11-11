# SORM — Yöntem kararları ve kodlama öncesi sözleşme

**Tarih:** 11.11.2026. **Durum:** Öneriler ve karar kapıları; deney sonucu değildir.

Bu belge `01`–`04` belgelerini tamamlar. Aşağıdaki öneriler, ilgili karar kaydıyla
kesinleştirilmeden uygulanmış veya kullanıcı tarafından tek tek seçilmiş sayılmaz.
Kullanıcı bu turda dokümantasyonu istemiştir; kodlama başlamamıştır.

## Kararların durumu

- **Kesin ilke:** Başlangıçta bilgi kimliği, A/B uzmanlığı veya hedef grup yok.
  Yerel yasalar bütün birimler için aynıdır; geometrik giriş farklılığı serbesttir.
- **Planlanan mimari:** C++ tek hesaplama çekirdeği ve ortak servis; Python deney
  tarifleri, tohum/kontrol orkestrasyonu ve sonuç analizi. Python ve IPC aynı
  servisi çağırır; denklemler Python'da ikinci kez uygulanmaz.
- **Planlanan ilk kapsam:** CPU M0, 1.000 birim, ham telemetri ve K1-G sahne
  gözlemi. GPU compute zorunlu değildir; mevcut GPU çizimi kullanılabilir.
- **Öneri:** Önce kesin M0, ardından tek ve açık M1 adayı. M2 enerji, M3 değişen
  topoloji ve hiyerarşi daha sonraki karar kapılarında ele alınır.
- **Açık karar:** M1 temsil biçimi/denklemi, kesin parametreler, giriş geometrisi,
  zamanlar, metrikler ve sayısal/işlevsel kabul eşikleri.

**K0 güncellemesi — 11.11.2026:** M0'a ait denklem, bütün sayısal değerler,
giriş geometrisi, protokol, metrikler ve sayısal eşikler
[07_M0_SOZLESMESI.md](07_M0_SOZLESMESI.md) içinde kesinleştirildi. Yukarıdaki açık
kararlar M1 ve sonrası içindir. Bu bir tanım kararıdır; sayısal kabul sonucu yok.
İlk iki ayrıntılandırma adımının M0 kapsamı tamamlandı; M1 kapsamı açık kalır.

## Önerilen altı ayrıntılandırma adımı

| Sıra | Üretilecek sözleşme | Tamamlanma koşulu |
|---|---|---|
| 1 | Hipotez → ölçülebilir davranış | Geçmiş izi, seçicilik, tamamlama ve çağrışım ayrı tanımlı; her biri için yanlış pozitif ve kontrol var. |
| 2 | Kesin M0 | Tek denklem, tüm katsayılar, başlangıç/sınır koşulları, giriş alanı, integratör ve toleranslar sayısal olarak tanımlı. |
| 3 | Tek ilk M1 | Yerel iz üretimi, gevşeme, etkileşime geri etkisi, sınırlar ve donmuş-iz kontrolü eksiksiz. |
| 4 | Deney protokolü | A/B sunum/bekleme/tekrar dizisi, hızlı durum eşitleme, kontrol ve prob davranışı belirli. |
| 5 | Ölçüm/karar kuralları | Yanıt penceresi, uzaklık, sıfır yanıt, tohum birimi, başarı ve durdurma eşikleri belirli. |
| 6 | Gözlem sözleşmesi | Renk/çizgi/ölçek matematiksel alanla eşleşiyor; snapshot ve telemetri aynı adımda; gözlem sonucu değiştirmiyor. |

Her sayısal değer birim/boyutsuzlaştırma, gerekçe, geçerli aralık ve model sürümüyle
kaydedilir. Pilotla seçilen değerler pilot olarak işaretlenir; doğrulama verisi
üzerinde ayar yapılmaz. Parametre seçimindeki başarısız denemeler de saklanır.

## M1 için temsil önerisi

**Öneri:** İlk adayda tek düğüm izini başlangıç seçeneği olarak incele; ilişkiye
özel hatırlama gereksinimi karşılanamıyorsa, gerekçesi kaydedilmiş ayrı bir
kenar-izi adayıyla karşılaştır. Bu henüz nihai model kararı değildir.

| Seçenek | Anlamı | Sınır ve karşılaştırma gereği |
|---|---|---|
| Düğüm izi h_i | Birimin yerel geçmişini özetler. | Basit ve küçük durum; hangi komşuyla geçmiş yaşandığını ayrı tutamaz. |
| Kenar izi h_ij | Belirli komşu etkileşiminin geçmişini taşır. | İlişkisel bellek doğrudan temsil edilir; simetri/yön, başlangıç ve kapasite açıkça tanımlanmalıdır. |

Kenar izine geçiş M3 değildir: sabit komşulukta sürekli iz taşımak ile yeni
kenar oluşturmak farklıdır. Bütün izlerin ortak nötr başlangıcı olmalıdır.
Bir adayın başarısızlığını gizlemek için düğüm/kenar temsili sessizce değiştirilmez.

İlk adayda birlikte etkinlik gibi basit yerel bir G ile başlamak önerilir;
gecikmeli korelasyon, inhibisyon ve ek izler aynı anda eklenmez. Basit kuralın
yalnız genel iletkenlik artışı veya senkron kilitlenme üretebileceği kabul edilir.
Kesin G seçilmeden öğrenme beklenmez; bu metin nihai denklem tanımlamaz.

## Matematikte kapatılacak sorular

1. x işaretli mi, negatif x'in yorumlanması ne? f ve eşik hangi aralıkta?
2. İz etkinlik, birlikte etkinlik, gecikme veya akıdan hangisini biriktiriyor?
   G yalnız gerçekten yerel bilgi kullanıyor mu?
3. İz güncellemesi etkileşimi nasıl değiştiriyor? Değişen nicelik k, b veya başka
   bir katsayı mı? Boyut, simetri ve işaret koşulları ne?
4. Doygunluk/sınırlılık denklemden mi geliyor? Clamp gerekiyorsa hangi modele
   karşılık geliyor? Yapay kararlılık öğrenme sonucu diye sunuluyor mu?
5. τ_h/τ_x oranı ne; gevşeme hangi koşulda gerçekleşiyor? α=0 ile bütün h'yi
   dondurmak aynı değil: gevşeme ayrı ablasyon olmalı.
6. Komşu toplamı nasıl normalize ediliyor? Derece/yarıçap değişiminde kazanç
   değişimiyle topoloji etkisi ayrılabiliyor mu?
7. Heun doğrulaması hangi norm ve toleransla yapılıyor? Gürültü varsa süreç,
   adım ölçeklemesi ve RNG eşleştirmesi tanımlı mı?

## Deney ve ölçüm önerileri

Önce tam x(t) kaydı korunur. Birincil metrik için sabit prob penceresindeki yanıt
vektörlerinin normalize Öklid uzaklığı adaydır; boyut normalizasyonu ve zaman
örneklemesi tanımlanmalıdır. Kosinüs benzerliği ikincil olabilir; sıfıra yakın
vektörler için açık davranış gerekir. Metrik henüz kesin seçilmemiştir.

Genlik değişimiyle örüntü değişimi ayrı raporlanır. Benzerlik iyileşmesi yalnız
yanıtın sıfıra çökmesinden kaynaklanamaz. Doğrudan giriş alanı dışındaki ölçüm,
giriş geometrisi kontrolü ve h korunurken hızlı durum eşitleme zorunludur.

Başarı için yalnız h'nin değişmesi yeterli değildir. Seçilen davranışın donmuş
kontrolü aşması, bağımsız tohumlarda tekrarlanması ve önceden dondurulmuş eşiklerle
değerlendirilmesi gerekir. İstatistik birimi bağımsız koşu/tohumdur; aynı koşudaki
parçacıklar veya zaman örnekleri bağımsız deney sayılmaz.

## Diğer ajanların çalışması ve zorunlu kayıt

Başka ajanlar bu araştırmayı sürdürebilir. Bu ifade mevcut turda otomatik ajan
başlatma veya kodlama yetkisi değildir. İş alanları önce
`../AJAN_KOORDINASYON.md` üzerinde ayrılır; çakışan dosya düzenlenmez.

**Her ajan, her anlamlı ilerlemeyi ve her sonucu mutlaka belgelere işler.**
Olumlu, olumsuz, kararsız, iptal edilmiş ve çalıştırılamamış sonuçlar dahildir.
Sohbet mesajı, koordinasyon panosu veya ham çıktı tek başına araştırma kaydı yerine
geçmez. İşin tamamlanma koşulu ilgili bilimsel/teknik belge ile
`06_ARASTIRMA_GUNLUGU.md` kaydının birlikte güncellenmesidir.

Çalışma uzun sürerse her tamamlanan karar, deney veya iş paketi sonunda kayıt
yapılır; yalnız en son teslim beklenmez. Ani kesintiden sonra ilk devam adımı
eksik kaydı tamamlamaktır. Gözlem, yorum ve öneri açıkça ayrılır; çalıştırılmamış
kontrol PASS yazılamaz. Sonuç sonrası model/protokol değişirse yeni sürüm ve neden
kaydedilir; eski başarısızlık silinmez.
