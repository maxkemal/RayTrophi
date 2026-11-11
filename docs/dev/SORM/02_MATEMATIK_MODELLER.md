# SORM — Matematiksel aday modeller v0.3

**Durum (11.11.2026):** Aşağıdaki genel denklemler model ailesidir. M1–M3 henüz
tek anlamlı yürütülebilir sözleşmeler değildir. İlk M0 seçimi ve açık kararlar
aşağıda ayrılır; alternatifler başarılı sonuca göre sessizce değiştirilemez.

## Durum ve giriş
N özdeş yasaya tabi birim; konum r_i (ilk sürüm sabit), hızlı etkinlik x_i, kullanılabilir enerji e_i≥0, yavaş yapısal iz h_i, yerel komşular K_i. Gerekirse faz φ_i. Başlangıçta sembole özel ağırlık yok. A/B, öğrenmeyen sabit sensör dönüştürücüsünden geçen farklı u(r,t) alanlarıdır; toplam giriş bütçesi dengelenir.

## M0: Plastisitesiz bazal referans
τ dx_i/dt = -x_i + f(I_i + Σ_{j∈K_i} c_ij(x_j-x_i) - θ) + σξ_i.
c_ij yalnız yerel geometriden doğar; h değişmez. Bu modelde geçmişten öğrenme iddiası yoktur.

## M1: Etkileşim izi modeli
τ_x dx_i/dt = -x_i + f(I_i + Σ_j J_ij(h_i,h_j,x_i,x_j) - θ) + σ_x ξ_i.
τ_h dh_i/dt = -∂V(h_i)/∂h_i + αG(x_i,{x_j}) + σ_h ζ_i; τ_h >> τ_x.
Aday J_ij = k_ij(h_i,h_j)(x_j-x_i) + b_ij(h_i,h_j)tanh(x_j).
b_ij negatif olabilir (baskılama). G için yerel birlikte değişim, gecikmeli korelasyon veya akı değişimi alternatifleri test edilir. Hiçbiri A etiketi veya küresel doğru yanıt kullanmaz.

### M1 karşılaştırmasının zorunlu eşitliği

M1'in asıl kontrolü aynı hızlı denklemin, aynı başlangıç h ile plastisitesi
dondurulmuş sürümüdür. Yukarıdaki M0 tek başına yeterli kontrol değildir: b terimi
veya başka hızlı dinamik farkları başarıyı plastisiteden bağımsız değiştirebilir.
Donmuş kontrol ile iz silme/karıştırma ablasyonları ayrı deneylerdir.

K2 öncesinde f, V, G, k, b; işaret ve simetri koşulları; tüm değişkenlerin
aralıkları; başlangıç değerleri; zaman sabitleri ve komşu toplamının
normalizasyonu kesinleştirilmelidir. Geometri değiştikçe derece değişimi etkin
kazancı da değiştirebilir. Her aday ayrı model sürümü alır.

Düğüm izi h_i ilişkiye özel h_ij değildir: bir düğümün farklı komşularla geçmişini
tek değere sıkıştırır. İlk aday tek düğüm iziyle sınırlanabilir; başarısızlık bu
kapasite sınırıyla yorumlanır. Kenar izi ayrı karşılaştırmadır; başlangıçta
öğrenilmiş kenar veya sembole özel ağırlık atanmaz.

## M2: Enerji aktarımı açık model
F_ij=-F_ji, örnek F_ij=κ_ij(h)(e_j-e_i).
de_i/dt=P_i-D_i+Σ_j F_ij, P_i≥0 dış besleme, D_i≥0 kayıp.
d(Σe_i)/dt=ΣP_i-ΣD_i (kapalı sınırda).
x_i=ψ(e_i,h_i,komşu etkileri). Yapısal değişim maliyeti varsa D_i'ye eklenir. e_i≥0 ve akı antisimetri korunur. M1 etkinliği fiziksel enerji sayılmaz.

Örnekte κ_ij=κ_ji≥0 gerekir. Antisimetri toplam muhasebeyi sağlar, ayrık adımda
e_i≥0 koşulunu tek başına sağlamaz. Aktarım, kayıp ve yapı maliyeti birlikte
pozitifliği korumalıdır; negatif enerjiyi sonradan clamp etmek muhasebe hatasını
gizleyemez. ψ, P, D ve yapısal maliyet tanımlanmadan M2 uygulanmaz.

## M3: Açık bağ karşılaştırması
a_ij∈{0,1}; oluşum olasılığı P=1-exp(-k_ijΔt), k_ij≥0; kopma hızı d_ij≥0. Yerel iz ve sınırlı komşuluk üzerinden hesaplanır. Başlangıç varsayımı değil, M1'e rakip deneydir.

Bu olasılık aralık boyunca sabit hız varsayar; değişen hız için entegre hazard
veya yakınsaması doğrulanmış ayrık yaklaşım gerekir. Kapasite dolduğunda seçim,
karşılıklı kenar simetrisi, eşzamanlı oluşum/kopma ve güncelleme sırası açıkça
tanımlanmalıdır; indeks sırası belirli grupları kayıramaz.

## Unutma
Örnek: dh/dt=α·yerel_etkileşim−λh+σζ. λ gevşeme oranıdır, termodinamik entropi ölçüsü değildir. Hızlı/yavaş iki iz ancak tek iz yetersizse eklenir.

## Ölçümler
- Yanıt R_s(t)=[x_1,...,x_N], sabit analiz penceresi.
- İç tekrar tutarlılığı ve A-B ayrıştırma farkı; metrik deney öncesi sabitlenir.
- Etkin yayılım: eşik üstü birim sayısı; katılım N_eff=(Σ|x_i|)^2/(Σx_i²), sıfır paydada 0.
- Gecikme: kararlı yanıta kadar süre.
- M2 maliyeti: ∫ΣD_i dt; yapı maliyeti D içinde yer alıyorsa ikinci kez eklenmez.
  Ayrı raporlanırsa toplamın aynı terimleri bir kez saydığı gösterilir.
  M1 için ∫Σ|x_i|dt yalnızca etkinlik vekilidir; joule değildir.
- Bellek ömrü: bekleme aralığına karşı ayırt edilebilirlik.
- Çağrışım: A-only sonrasında B referansına yakınlık, eşleştirilmemiş kontrole göre.
- Kolektif yapı: topluluk modülerliği, işlev sürekliliği, hasar/ablasyon dayanıklılığı.

## Sayısal kontroller
Δt küçültme, tohum tekrarı, sınırlılık, sıfır giriş, enerji muhasebesi, akı antisimetri, doygunluk, gürültü ve etkileşim yarıçapı taraması. Tek başarılı tohum yeterli değildir.

### Gürültü sözleşmesi

ξ ve ζ ifadeleri uygulamadan önce SDE veya sonlu korelasyonlu süreç olarak
tanımlanır. Örneğin `dx=a dt+sqrt(2D_x) dW` seçilirse rastgele artım
`sqrt(2D_x Δt) Z`, Z~N(0,1) olur. τ içeren yazımla dönüşüm açıkça yapılır.
Her adımda sabit büyüklükte rastgele artım eklemek Δt değişince başka model üretir.
İlk doğrulama gürültüsüzdür. Gürültülü yakınsama kıyasında ortak Brown hareketi
artımlarının eşleştirilmesi veya dağılımsal yakınsama yöntemi belirtilir.

### İlk 1.000 birim için M0 sözleşmesi

**Güncelleme:** M0'ın kesin katsayı/giriş/zaman/eşik tanımı
[M0 v1 sözleşmesinde](07_M0_SOZLESMESI.md) tamamlandı. Aşağıdaki liste özettir;
henüz doğrulanmış sayısal sonuç değildir.

- Boyutsuz etkinlik/giriş/katsayılar; zaman τ_x biriminde raporlanır.
- Sabit periyodik 10×10×10 kafes, altı komşu; c_ij=c_ji=c≥0, x_i(0)=0.
- f=tanh, θ=0; ilk koşuda σ=0. Denklem yukarıdaki M0'dır.
- CPU çift duyarlık, sabit adımlı Heun; başlangıç adayı Δt=τ_x/100.
- Aynı süre için Δt/2 ve Δt/4 kıyası zorunlu; aday adım kararlılık garantisi değildir.
- c=1/6, giriş genliği 1, darbe 2, slot 22, kayıt aralığı 0,1; ayrıntılı
  alanlar, kol süreleri ve toleranslar 07'de sabittir. Gerçek manifest K1'de yazılır.

### Ölçüm tanımlarındaki açık kararlar

Bu açık kararlar M1 ve sonraki aşamalar içindir. M0'ın penceresi, RMS uzaklığı,
sıfır yanıt davranışı ve sayısal kabul eşikleri 07'de kesinleştirilmiştir.

Yanıt penceresi ve uzaklık metriği, etkinlik eşiği, kararlı yanıt için tolerans ve
bekleme süresi, B referansı oluşturma yöntemi, hasar sonrası görev başarısı ve
grafik çıkarma/topluluk yöntemi önceden tanımlanır. Sıfır yanıtın benzerlik
metriklerinde ele alınması belirtilir. M0'daki kararlı ayrışma öğrenme sayılmaz.


### Gözlem alanlarının matematiksel anlamı

M0'da gösterilebilir alanlar r_i, x_i, I_i ve sabit c_ij/K_i'dir. Sabit komşuluk
çizgileri öğrenilmiş bağ olarak adlandırılmaz. M1'de h_i ve tanımlı k_ij/b_ij
ayrı katmanlardır; anlık J_ij, kalıcı ilişki gücüyle aynı büyüklük değildir.
M2'de e_i ve antisimetri sözleşmesine sahip F_ij enerji/akı görünümü sağlar.
M3'te a_ij oluşum/kopma görünümü sağlar. Modelde olmayan alan uydurulmaz.

Renk dönüşümü ve görsel eşik yalnız görünüm parametresidir. Karşılaştırmalarda
ortak sabit ölçek ve lejant kullanılır; otomatik ölçek açıksa aralık görünür ve
kayıtlıdır. Negatif etkinliğin işareti korunur. Küme renklendirmesi kullanılırsa
analiz penceresi, yöntem ve parametreleri belirtilir; etiketler dinamike girmez.
