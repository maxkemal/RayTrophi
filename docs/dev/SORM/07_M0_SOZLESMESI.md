# SORM M0 v1 — Sayısal model ve bazal deney sözleşmesi

**Tarih:** 11.11.2026. **Model:** `sorm.m0.v1`. **Protokol:** `sorm.baseline.v1`.
**Durum:** Kodlama öncesi tanım tamamlandı; uygulama ve deney kabulü yapılmadı.

Bu belge M0 için önceki genel adayları kesinleştirir. Seçilen değerler ilk bazal
referansın mühendislik seçimleridir; öğrenmeye uygunlukları deneyle doğrulanmış
değildir. Değer değiştirmek yeni manifest/koşu, denklem veya protokol anlamını
değiştirmek yeni model/protokol sürümü gerektirir. M1–M3 kararları açık kalır.

## 1. Amaç ve kapsam

1.000 özdeş yasaya tabi sabit birimde girişin yayılması/sönmesi, deterministik
adımlama, ham kayıt ve salt-okunur sahne gözlemini doğrulamak. M0 öğrenmez; iz,
öğrenilmiş bağ, fiziksel enerji veya başlangıç uzmanlığı içermez. Bu protokolde
A/B ayrışması fiziksel giriş geometrisinden gelir; öğrenme başarısı değildir.

## 2. Durum, geometri ve indisler

- Kafes koordinatları a,b,c ∈ {0,…,9}; indeks i=a+10b+100c, N=1000.
- Sabit model konumu r_i=(a,b,c); kafes aralığı 1 boyutsuz uzunluk.
- Her eksende periyodik sınır; altı komşu sırası +a,−a,+b,−b,+c,−c.
- Öz kenar yok; 6.000 yönlü kayıt, 3.000 benzersiz yönsüz kenar.
- Tek dinamik değişken x_i ∈ R; başlangıç x_i=0. ID yalnız indisleme içindir.
- Her birimin aynı parametreleri vardır. h, e, hız ve fiziksel kütle M0 durumu
  değildir; arayüzde sıfır değerli ölçülmüş nicelikler gibi gösterilmez.

## 3. Kesin denklem ve katsayılar

Boyutsuz zaman s=t/τ_x ile:

```text
dx_i/ds = -x_i + tanh(I_i(s) + (1/6) Σ[j∈K_i](x_j-x_i))
τ_x = 1 model zaman birimi
θ = 0
σ = 0
c_ij = 1/6  (komşu olmayan çiftte 0)
```

Toplam komşuluk kazancı 1'dir; 1/6 normalizasyonu dereceyi gizlice kazanca
çevirmemek için seçildi. Burada derece sabittir; farklı topoloji ayrı protokoldür.
İşaretli x etkinliktir, enerji değildir. Bu girişlerle negatif etkinlik beklenmez;
işaretli sözleşme daha sonraki negatif giriş denemelerini peşinen yasaklamaz.

Sürekli modelde |x|≤1 bölgesi değişmezdir: x=1 sınırında türev pozitif, x=−1
sınırında negatif olamaz. Sıfır girişte mutlak maksimumda komşu farkının işareti
geri çağırıcıdır; dolayısıyla ||x(s+w)||∞≤exp(−w)||x(s)||∞. Bu analitik özellik
ayrık çözücünün testi yerine geçmez. Model korunumlu enerji difüzyonu veya
salınımlı fiziksel dalga denklemi değildir; viewport'ta “etkinlik yayılımı” denir.

## 4. Giriş alanları

```text
S_A = {a∈{1,2}, b∈{4,5}, c∈{4,5}}
S_B = {a∈{6,7}, b∈{4,5}, c∈{4,5}}
I_i(s) = 1, eğer i aktif alanın içindeyse ve darbe açıksa; diğer durumda 0
darbe süresi = 2
```

Her alan 8 birimdir, örtüşmez; B, A'nın a yönünde 5 hücre periyodik ötelemesidir.
Bu seçim şekil/derece farkı olmadan konum etkisini sınar. Bir darbenin
∫Σ|I_i|ds bütçesi 16, ∫ΣI_i²ds bütçesi de 16'dır; bunlar joule değildir.
A/B isimleri düzenleyici metaverisidir; çekirdek yalnız N bileşenli sayısal alan
alır. Bu, başlangıçta sembol uzmanlığı atamak değil sabit sensör geometrisidir.
K0'da eşzamanlı A+B, bozulmuş giriş ve gürültü kullanılmaz.

## 5. Ayrık adımlama

CPU binary64; başlangıç Δs=0,01. Yakınsama koşuları 0,005 ve 0,0025 kullanır.
Güncellemeler eşzamanlıdır; yerinde Gauss–Seidel tarzı yazım yasaktır.

```text
F(x,I) = denklem sağ tarafı
k1 = F(x_n,I_n)
y  = x_n + Δs k1
k2 = F(y,I_n)
x_(n+1) = x_n + (Δs/2)(k1+k2)
```

Giriş sabit yarı-açık aralıklarda tanımlanır. Bir adım [s_n,s_(n+1)) içindeki
aynı I_n ile iki aşama hesaplanır; k2 darbe sınırında sonraki girişe bakmaz.
Bütün olaylar adım sınırındadır; süre→adım dönüşümünde tam bölünebilirlik
doğrulanır. Olay zamanları kayan noktalı eşitlikle değil tamsayı sayacıyla izlenir.
Clamp, adaptif Δs veya başarısız adımı sessizce atlama yoktur.

## 6. Bazal koşu ailesi

Önce x=0 ile 10 birim sıfır giriş uygulanır ve ortak C0 checkpoint'i alınır.
Bu tam sıfır başlangıçta analitik olarak sıfır kalır. Her aşağıdaki kol C0'dan
bağımsız başlar; probun bitiş durumu başka kola taşınmaz.

Tablodaki zamanlar kol başlangıcına göre yereldir. Checkpoint'in global zamanı
korunur; kayıt hem global adımı hem kol-yerel ofseti içerir. Dallandırma gizlice
simülasyon zamanını sıfırlamaz. Tek adım ve snapshot kimliği bu ayrımı korur.

| Kol | Sunum | Süre / amaç |
|---|---|---|
| Z | Yalnız sıfır giriş | 22; sıfır koruma |
| P_A, P_B | [0,2) darbe, [2,22) sıfır | Ön referans probları |
| T_A, T_B | 100 × (2 darbe + 20 sıfır), ardından 20 sıfır | 2220; tekrar sonrası gevşeme |
| T_AB | 100 slot; tek slotta A, çift slotta B; slot 2+20, ardından 20 sıfır | 2220; dengeli sıra kontrolü |
| T_BA | T_AB'nin A/B sırası ters | 2220; sıra kontrolü |

Her T kolunun son checkpoint'i üç kola dallanır: P_A, P_B ve Z; her biri yine
22 sürer. Ayrıca yalnız analiz müdahalesi olarak x=0 yapılıp aynı üç prob
tekrarlanır; bu doğal unutma değil hızlı durum eşitlemedir. M0'da başka dinamik
durum olmadığı için bu eşitlenmiş problar ön referansları yeniden üretmelidir.
Son slotta zaten 20 sıfır vardır; tablodaki ek 20 ayrıca uygulanır (son darbeden
sonra toplam 40). Bu ayrıntı uzun beklemeyi ve sıralamayı belirsiz bırakmaz.

M0 v1 deterministiktir, RNG tüketmez. Manifestte seed=0 ve rng=none yazılır;
aynı koşuyu 10 farklı kullanılmayan tohumla çalıştırmak bağımsız deney değildir.
Çoklu bağımsız tohum kapısı stokastik/pilot model sürümünde uygulanır.

## 7. Örnekleme ve ölçümler

Ham x her 0,1 birimde, başlangıç ve son dahil kaydedilir. Topoloji ve giriş
takvimi bir kez, olaylar zaman damgasıyla saklanır; anlık I takvimden tam üretilir.
Kabul koşularında aşağıdaki maksimum hatalar her integrasyon adımında da izlenir.
Çizim için kare atlanabilir; bilimsel kayıt aralığı değiştirilemez.
Olay sınırındaki kayıt x(s)'yi ve sağdaki aralıkta kullanılacak I(s)'yi taşır;
hemen önceki adımı hangi girişin ilerlettiği olay takviminden ayrıca bellidir.

Bir probun birincil yanıtı 0≤s≤6 aralığında 0,1 aralıkla 61 örnektir:

```text
R_Q[m,i] = x_i(0,1 m), m=0,…,60
d_Q(R,S) = sqrt( Σ_m Σ_(i∈Q) (R[m,i]-S[m,i])² / (61 |Q|) )
```

Q için hem bütün 1.000 birim hem Q_out=V\(S_A∪S_B), yani 984 birim raporlanır.
Uzaklık yanıt normuna bölünmez; sıfır yanıt için de tanımlıdır. Genlik farkı
ayrıca RMS normlarıyla verilir; bu metrik tek başına örüntü öğrenmesi göstermez.

Her kayıtta max|x|, RMS(x), Σ|x|, etkin birim sayısı (|x|≥0,01), doygun birim
sayısı (|x|≥0,99) ve N_eff kaydedilir. Σx²=0 ise N_eff=0; diğer durumda
(Σ|x|)²/Σx². Etkinlik integrali kayıt aralığında trapezle hesaplanır; yaklaşık
integral ve kayıt aralığı olduğu belirtilir, fiziksel enerji diye sunulmaz.

Gevşeme süresi darbe kapandıktan sonra ||x||∞≤0,001 olup 1 birim boyunca bu
sınırda kaldığı ilk örnek zamanıdır; bulunmazsa “ulaşmadı”dır. Bu, sabit girişte
kararlı dengeye ulaşma veya işlem gecikmesi ölçüsü değildir.

## 8. K1 sayısal kabul eşikleri

Eşikler ilk uygulama için önceden seçilen kabul değerleridir; PASS oldukları
henüz gözlenmedi. Başarısızlıkta neden incelenir, sonuç kurtarmak için eski kayıt
değiştirilmez. Değişen eşik yeni protokol sürümü alır.

| Kontrol | Kabul |
|---|---|
| Boyut/topoloji | 1000 durum; her düğümde 6 farklı karşılıklı komşu, öz kenar yok; 3000 benzersiz kenar |
| Sonluluk/sınır | Her adımda sonlu; |x|≤1+1e−12; bu pozitif girişlerde x≥−1e−12 |
| Sıfır başlangıç, sıfır giriş | Tüm koşu boyunca max|x|≤1e−14 |
| Homojen analitik fikstür | Ayrı kabul kolunda bütün i için I=1 ve x(0)=0: x_i(s)=tanh(1)(1−exp(−s)); 0≤s≤2 boyunca maksimum hata ≤2e−5; ardından I=0 ile x_i(2+w)=x_i(2)exp(−w) |
| Adım yakınsaması | Ortak 0,1 örneklerinde, her kol için E_01_005≤2e−5 ve E_005_0025≤5e−6; E maksimum bileşen mutlak farkıdır |
| Sıfır-giriş gevşeme | Darbe sonrası her örnekte M(w)≤M(0) exp(−w)+2e−5; 20 sonunda M≤1e−7 |
| Hızlı durum eşitleme | Sonraki prob ile C0 probunda maksimum bileşen farkı ≤1e−12 |
| Öteleme simetrisi | P_A'nın 5 hücre ötelenmiş yanıtı ile P_B arasında maksimum fark ≤1e−12 |
| Tekrar/checkpoint/paket | Aynı binary ve ortamda aynı adımlarda binary64 x bitleri eşit; CLI/Python/IPC ve farklı paket boyları dahil |
| Görünüm bağımsızlığı | Gözlem açık/kapalı ve kamera/filtre/hız değişiminde aynı adım canonical durum bitleri eşit |

Yakınsama fark oranı da raporlanır; yaklaşık 4 beklenir, yuvarlama tabanında oran
yorumlanmaz ve tek başına kabul kapısı değildir. Sıfır-giriş sınırı ayrık hata
payıyla test edilir. Platformlar arası tanh/derleyici farkında bit eşitliği vaat
edilmez; ortamlar arası fark ayrıca ≤2e−5 sınırıyla raporlanır.

A/B uzaklığı, Q_out yayılımı ve tekrar öncesi/sonrası uzaklıklar raporlanır;
bunlara öğrenme başarı eşiği atanmaz. Sıfır veya beklenmedik ayrışma bulgusu
gizlenmez. M1'e geçiş için mekanik/sayısal kabul tamamlanmalı; M1'in işlevsel
eşikleri onun protokolünde ayrıca belirlenmelidir.

## 9. K1-G görünüm sözleşmesi

- Model→sahne dönüşümü yalnız çizim içindir: merkez (a−4,5,b−4,5,c−4,5), ölçek 1
  sahne birimi, görsel yarıçap 0,12. Bu ölçek fiziksel metre ilan edilmez.
- Sabit diverjan renk lejantı: x=−1 mavi, x=0 koyu gri, x=+1 turuncu; iki aralıkta
  doğrusal RGB enterpolasyonu. Otomatik ölçek ve fiziksel emission varsayılan kapalı.
- Giriş aktif alanı ayrı çerçeve/işaretle gösterilir; A/B rengi x'i değiştirmez.
- Kenarlar nötr gri, varsayılan kapalı; sabit c=1/6, eşit kalınlık. Periyodik
  sarmalayan kenarlar farklı kesikli gösterimle ve açık lejantla belirtilir;
  düz dünya mesafesi uzun fiziksel etkileşim gibi sunulmaz.
- Adım, s, kol, sunum indeksi, model sürümü, görünen/toplam kenar ve snapshot
  zamanı görünür. Ham x/komşu/giriş seçilerek okunabilir.
- Normal sunum 1 model zamanı / duvar saniyesi, 0,1× ve 10× seçenekleri;
  yetişemiyorsa yavaşlar, sayısal adım atlamaz. Tek adım tam Δs ilerletir.
- Enerji, iz ve öğrenilmiş bağ katmanları M0'da “tanımsız”dır. Görsel yakınlık
  veya renk kümeleri uzmanlık olarak etiketlenmez.

## 10. Manifest, kayıt ve uygulama sınırı

İlk manifest model/protokol kimlikleri, N/kafes/komşu sırası, Δs, τ_x, c, θ,
σ, başlangıç, giriş alanları/takvimi, kol planı, kayıt aralığı, ölçüm/eşik sürümü,
rng=none, seed=0 ve çıktı hedefini içerir. Bilinmeyen veya çelişen alanlar reddedilir.
K0 bu anlamsal sözleşmeyi kapatır; gerçek JSON şeması, IPC alan/hata kodları ve
çıktı codec'i K1 uygulamasında bu belgeyi değiştirmeden adlandırılıp belgelenir.

Binary64 ham dizi ve tip/şekil/endian bilgili sürümlü metaveri önerilir; CSV
analiz özeti/isteğe bağlı dışa aktarım olabilir. Uzun bir kolun 0,1 örnekli x
verisi yaklaşık 178 MB'dır; bellekte sınırlı oynatma tamponu, diske akış gerekir.
Kayıt kaybı/hatası görünür başarısızlıktır. Geçici ekran görüntüleri kalıcı kanıt
değildir; ham sayısal çıktı ayrı deney dizininde saklanır.

Bu dosya kodlama, build veya uygulama başlatma yetkisi değildir. K0'ın M0
matematik/protokol tanımı tamamlanmıştır; K1 kabulü ve hipotez sonucu açık kalır.
Her ilerleme/sonuç ilgili belgeye ve `06_ARASTIRMA_GUNLUGU.md` dosyasına işlenir.
