# Elastic MPM sıcaklık yumuşaması — 2026-11-08

> Bu belge önceki bağlantı düzeltmesini ve o zamanki s0.2 tarifini anlatır.
> Yeni termal parametreler, sakız s0.02 tarifi ve değişen GPU ABI için
> [ELASTIC_THERMAL_RESPONSE.md](ELASTIC_THERMAL_RESPONSE.md) geçerlidir.

## Bulgu ve düzeltme

Chewing Gum, Elastic MPM yasası kullanır. Presetteki softening başlangıcı310K,
genişlik15K ve artık dayanım0.2 değerleri FluidDomainSubstance üzerinden
çözücüye zaten taşınıyordu. CPU ve GLSL elastik gerilme yasası da Young modülünü
ve Kelvin–Voigt viskoz gerilmesini aynı `soft` katsayısıyla zaten çarpıyordu.

Kopukluk FluidDomainStep'teki güncelleme kapısıydı: softening yalnızca
`granular_enabled` açıkken türetiliyordu. Elastic profil bu bayrağı kullanmaz;
karma Matter koordinatörü de tüm domain'i granular gibi çalıştırmamak için
bayrağı kapatır. Bu yollarda softening dizisi1'de veya eski değerinde kalabiliyordu.

Yeni `updateMatterMpmSoftening` gerçek parçacık modeli/taşıyıcısını seçer.
Granular ve Elastic MPM birlikte kapsanır; yalnız sıvı ve DEM taşıyıcılarının
mevcut scalar değerleri değiştirilmez. Mevcut `updateSoftening` eğrisi ortak
kalır; isteğe bağlı maske eklendi. Ayrı bir sıcaklık yasası çoğaltılmadı.

Çağrı, ısı alanı/soğuma ve parçacık iletiminden sonra, alt adımlardan ve CPU/GPU
gerilme hesabından önce bir kez yapılır. GPU aynı host softening dizisini mevcut
Granular::uploadState yoluyla alır. Shader formülü veya push-constant düzeni
değişmedi. Domain'in ortak MPM malzeme parametresi sınırı korunur; bu düzeltme
bağımsız çok malzemeli MPM yasası eklemez.

## Sakızda beklenen etki

- 302.5K ve altında katsayı1; 310K'de0.6; 317.5K ve üstünde0.2.
- Baz E200000Pa → sıcak durumda etkin E40000Pa.
- Baz Kelvin–Voigt η30Pa·s → sıcak durumda η6Pa·s; uygulanmış kararlılık tavanı
  daha düşükse gerçek değer bundan da düşük olabilir.
- Profilde görülen η30 başlangıç katsayısıdır; sıcaklık her kare profil verisini
  yeniden yazmaz. Etkin gerilmeye scalar olarak uygulanır.
- Maxwell gevşeme süresi1.5s ve akma gerinimi0.5 sabit kalır. Yeni η(T) veya
  gevşeme zamanı(T) fit'i eklenmedi. Residual0.2 nedeniyle sıcak sakız da sonlu
  direnç taşıyan Elastic MPM'dir; sıvı çözücüye dönüşmez.

Bu sayılar mevcut presetin başlangıç yaklaşımını açıklar; gerçek sakız için
ölçülmüş sıcaklık-rheology eğrisi iddiası değildir.

## Doğrulama

- Codex non-build kaynağa dayalı kontrolleri:
  `python scripts/test/check_matter_mpm_softening_contracts.py`.
- Kullanıcı C++ fizik test hedefi: `matter_mpm_softening_test.cpp`.
  İlgili MatterMpmSoftening/MatterSubstanceState/fizik-kütüphane bağımlılıklarıyla
  bağlanmalı. Test gerçek elasticStressUpdateParticle hesabında viskoz katkıyı
  η0 bazını çıkararak ayırır; sıcak/soğuk oranını ve maske davranışını kontrol eder.
- Derleme sonrası aynı sakız örneğini 293.15K ve en az320K **parçacık sıcaklığında**
  karşılaştır. Isı alanının hedef sıcaklığına ulaşmak zaman alabilir; yalnız alan
  değerini ayarlamak parçacıkların hemen o değerde olduğunu göstermez.
- Active Matter / Material resolution tablosunda ortalama parçacık sıcaklığını
  kontrol et. Aynı yük altında sıcak örneğin daha fazla deformasyonu beklenir.
  Isı alanı, sıcak/soğuk kalma süresi, h ve dt karşılaştırmada aynı tutulmalı.
- Soğukken yumuşama olmamalı; ısı alanı kapalı ve softening_temperature0 iken
  nötr katsayı1 geri gelmeli. Sıvı/DEM içeren karma sahnede taşıyıcılar korunmalı.

Codex derleme, shader compilation veya canlı uygulama testi yapmadı. Bu teslim
var olan madde alanlarının UI/script/IPC üzerinden çözücüye ulaşma hatasını
düzeltir; yeni bir authoring işlemi veya ayar eklemez.
