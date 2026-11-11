# Madde preset envanteri ve sadeleştirme önerisi

> Tarihsel envanter: 2026-11-08 düzeltmeleri sonrasında katalog 26 somut profildir.
> Güncel farklar ve açık eksikler: [BUILTIN_PRESET_CORRECTIONS.md](BUILTIN_PRESET_CORRECTIONS.md).

Tarih: 2026-11-07. Çalışma ağacının statik kaynak incelemesidir; derleme, uygulama
çalıştırma veya yeni fizik doğrulaması yapılmadı. Bu belge presetleri değiştirmez.
Değerler ölçülmüş malzeme sertifikası değil, kodda bulunan başlangıç değerleridir.

## Kapsam ve gerçek seçim yolu

- `RayTrophiStudio/source/src/Physics/MaterialStateField.cpp`: 26 adlandırılmış profil.
  Mud ve Mud Slurry, `Physics/Fluid/MudMaterial.cpp` fabrikalarından gelir.
- `Physics/Fluid/MatterSubstanceState.cpp::configureBuiltinMatterTransport`:
  Sand, Gravel ve Ice DEM'e, kalanlar MPM taşıyıcı tercihine ayarlanır.
  Bu tercih yalnızca ilgili granular model seçildiğinde anlamlıdır; Water MPM
  katısı haline gelmez.
- `Physics/SubstanceCatalogue.cpp::appendFoundationSubstances`: altı temsilciyi
  kopyalayarak toplamı 32'ye çıkarır. Proje tarafından türetilmiş maddeler hariçtir.
- `Physics/Fluid/FluidDomainSubstance.cpp`: madde modelini ve etkin sıvı
  viskozitesini çözücü parametrelerine aktarır.
- `include/MaterialStateField.h`: açıkça atanmayan alanların varsayılanları.

`category` menü gruplamasıdır; fizik modeli değildir. `Fuel` de ayrı bir sıvı
mekanik modeli değildir. Isıl yüzey maddesinin varsayılan `Fluid` alanı bulunması,
sahnedeki ahşap veya çeliğin otomatik olarak sıvı parçacıkla çözüldüğü anlamına gelmez.
Backend seçimi (CPU/Vulkan/CUDA) aşağıdaki malzeme modelinden ayrı bir boyuttur.

## 26 adlandırılmış profil

Yoğunluk kg/m³, kinematik viskozite ν m²/s, dinamik viskozite μ Pa·s.
μ değerleri yalnızca aynı fazdaki sıvı yoğunluğu ile μ=ρν hesabıdır.

| Profil | Mevcut mekanik/temel rol | Önemli mevcut değer | Karar önerisi |
|---|---|---|---|
| Water | Fluid, basınç projeksiyonlu sıvı; ısıl faz verileri | ρ997, ν1e-6 | Temel Newtonyen referansı olarak tut |
| Oil | Fluid + yanabilir sıvı kimyası | ρ850, ν1e-4, μ0.085 | Tut; yağ türü ve referans sıcaklığını belirt |
| Honey | Fluid, sabit viskozite | ρ1420, ν0.007, μ9.94 | Tut; sıcaklık/bileşim aralığı belli Newtonyen yaklaşım olarak etiketle |
| Chocolate | Fluid, sabit viskozite; erimiş çikolata yorumu | ρ1270, ν0.004, μ5.08 | Öncelikli model düzeltmesi: akma gerilmeli, kayma hızına bağlı sıvı |
| Mud Slurry | Fluid, homojen yoğun süspansiyon yaklaşımı | ρ1700, ν0.002, μ3.4 | Mud'un sulu tarifi olarak sun; çok fazlı çamur diye tanıtma |
| Mud | Elastic MPM, elastoplastik + Kelvin–Voigt | E80kPa, akma gerinimi0.025, η150 | Tut; mevcut EVP yaklaşımı, ölçülmüş zemin yasası değil |
| Sand | Granular + DEM | ρ1600, sürtünme35°, tane yarıçapı0.25mm | Tut; kuru taneli referans |
| Gravel | Granular + DEM | ρ1750, sürtünme43°, yarıçap4mm | Sand ile aynı model ailesinde ayrı tane tarifi; doğrudan silme |
| Soil | Granular MPM, kohezyonlu süreklilik | ρ1450, sürtünme30°, kohezyon800Pa | Tut; kuru/nemli tarifleri ileride su içeriğiyle ayır |
| Snow | Granular MPM, kar plastisitesi/sertleşmesi | ρ400, E140kPa; basınç eşiği0.025, çekme0.0075 | Ayrı model davranışı; koru |
| Ice | Granular + DEM, gevşek buz parçaları; erime | ρ917, sürtünme2.3°, E9.3GPa | “Ice chunks” tarifi olarak açıkla; sürekli buz katısıyla eşitleme |
| Gelatin | Elastic MPM, sönümlü elastik ve kopma; erime | E5kPa, η8, kopma gerinimi0.8 | Koru; elastik referans |
| Dough | Elastic MPM, elastoplastik | E30kPa, akma0.10, η50 | EVP altında örnek tarif; Mud ile parametre benzerliği silme nedeni değil |
| Chewing Gum | Elastic MPM, gevşeme + plastisite | E200kPa, akma0.5, gevşeme1.5s, η30 | Koru; viskoelastik referans; ısıl yumuşamanın tüketici yolunu ayrıca doğrula |
| Gasoline | Fluid + buharlaşma/yanma | ρ740, ν6e-7 | Kimyasal örnek; termokimyasal değerlerin kaynağını doğrula |
| Alcohol | Fluid + buharlaşma/yanma | ρ789, ν1.5e-6; mol kütlesi etanol yönünde | Bileşimi açık “Ethanol” örneğine dönüştürmeden önce değerleri doğrula |
| Wood (Oak) | Isıl yüzey, gözeneklilik, yanma/kömürleşme | ρ750, c1700, k0.17 | Koru; otomatik ahşap mekanik kırılma modeli değildir |
| Paper | Isıl yüzey, emme, yanma | ρ800, c1340, k0.05 | Lifli malzeme örneği; Wood ile aynı değil |
| Cloth | Isıl yüzey, emme, yanma | ρ400, c1300, k0.06 | Lif bileşimini belirt; kumaş mekanik çözücüsü vaadi oluşturma |
| Iron | Isıl yüzey/erime; sıvıda Fluid | ρ7874, k80.4, sıvı ν1e-6 | Metal örneği olarak tut; katı metal plastisitesi tanımlı sayma |
| Steel | Isıl yüzey/erime; sıvıda Fluid | ρ7850, k45, sıvı ν1e-6 | Iron'a yakın yoğunluk fakat farklı iletkenlik; tek alaşım ailesinde örnek |
| Copper | Isıl yüzey/erime; sıvıda Fluid | ρ8960, k401, sıvı ν5e-7 | Koru; metal ısı iletimi karşılaştırması için farklı |
| Plastic (PE) | Isıl yüzey/erime/yanma; eriyik sabit viskoziteli | ρ950, erime403K, sıvı ν0.30 | Polimer eriyiği örneği; sıcaklık ve kayma bağımlılığı eksik |
| Wax | Isıl yüzey/erime/yanma; sıvıda Fluid | ρ900, erime330K, sıvı ν5e-6 | Koru; faz tarifi kimlikten ayrı olmalı |
| Stone | Isıl yüzey/erime | ρ2600, erime1473K; eski melt_viscosity0.9 | Öncelikli denetim: sıvı ν açık atanmamış; varsayılan1e-6 ile legacy Lava0.5 çelişiyor |
| Flesh | Isıl yüzey/yanma örneği | ρ1050, c3500, k0.5 | Test/ileri örneklerde tut; biyomekanik malzeme diye sunma |

## Altı birebir kopya ve eski adaptörler

| Foundation | Kopyaladığı profil |
|---|---|
| Generic metal | Steel |
| Generic mineral | Stone |
| Generic polymer | Plastic (PE) |
| Generic fibrous solid | Wood (Oak) |
| Generic liquid | Water |
| Generic mineral grains | Sand |

Bunlar fiziksel olarak genel metal/mineral/polimer değildir; temsilcinin tüm
sayısal değerlerini kopyalarlar. En temiz sadeleştirme: aileyi bir sınıflandırma
olarak koruyup “bu aileden yeni madde” işlemi sunmak; bu altı kaydı ayrı malzeme
seçeneği olmaktan çıkarmak. Hedef: 32 yerine 26 somut başlangıç örneği.
Mevcut `based_on`/proje maddesi sistemi bunun için kullanılabilir.

`FluidDomainSubstance.cpp` içindeki Lava, Wet Sand, Cohesive Soil ve Molten Plastic
legacy uyarlamaları bu 32'ye ek canlı katalog girdisi değildir. Bunları menüden
preset silmekle karıştırmamak gerekir. Özellikle legacy Molten Plastic'in cohesive
granular yolu ile PE'nin Fluid eriyik yolu aynı fizik varsayımı değildir.
Yeni katalog bu çelişkiyi devralmamalı. Eski sahne desteği gerekmiyorsa adaptörler
ayrı temizlikte kaldırılabilir; aktif çağrı/serialization yolları önce taranmalı.

## Görünüşü etkileyen iki önemli teknik bulgu

1. `FluidDomainSubstance.h` sabitleri ν·(1/24)/h² < 0.001 olduğunda,
   ısıl sıvı yolu etkin değilse viskoziteyi sıfıra düşürüyor. Bu gerçek zaman
   adımı ölçümü değil sabit referans dt kullanıyor. h=0.10m için Oil değeri
   0.000417 olduğundan viskoz çözüm atlanıyor; Honey0.0292, Chocolate0.0167,
   Mud Slurry0.00833 olduğundan bu üçü eşik üstünde kalıyor. Oil için geçiş
   h≈0.0645m. Bu nedenle yağın su gibi görünmesi yalnızca yanlış ν girişi değildir.
   Çözüm: etkin/istenen ν ve atlama nedenini telemetry'de göstermek; eşiği
   doğruluk politikasıyla değerlendirmek. Her sıvıyı MPM'e taşımak bunu çözmez.
2. `sim_fluid_g2p.glsl` içindeki internal_friction, hızı exp(-k·dt) ile çarpar.
   Honey/Chocolate0.2 ve Oil0.15 gibi değerler iç kayma yasası veya akma gerilmesi
   değil genel hız sönümüdür. Serbest hareketi de etkileyebilir. Fiziksel rheology
   kalibrasyonunda bunu ayrı sayısal/ortamsal sönüm olarak ayırmak gerekir.

Eski `melt_viscosity`, SI birimli `liquid_kinematic_viscosity` ile aynı şeymiş gibi
yorumlanmamalı. Özellikle Stone/Lava için tüketici zinciri ve birimler ayrı
izlenmeli. Bu inceleme tüm CPU/Vulkan/CUDA aşamalarının sayısal eşitliğini kanıtlamaz.

## Önerilen ürün ve fizik yapısı

Tek bir “endüstriyel standart preset listesi” yok. Yaygın ve savunulabilir ayrım:

1. **Madde kimliği:** bileşim, yoğunluk, ısı kapasitesi, iletkenlik, kimya ve
   kaynak/provenans. Referans sıcaklık, basınç ve ölçüm aralığı açık olmalı.
2. **Durum/tarif:** sıcaklık, katı/sıvı oranı, nem, tane boyutu/dağılımı,
   paketlenme ve kar sertliği. Wet Sand yeni bir kimyasal madde olmamalı.
3. **Malzeme yasası:** Newtonyen; akma gerilmeli/shear-thinning sıvı;
   elastik; viskoelastoplastik; sürtünmeli/kohezyonlu granüler; kar plastisitesi.
   İlk etapta mevcut modeller korunur, eksik yasa açıkça “yaklaşım” diye belirtilir.
4. **Taşıyıcı ve backend:** projeksiyonlu sıvı, MPM süreklilik veya DEM taneler;
   ardından CPU/Vulkan/CUDA yetenek kontrolü. Desteklenmeyen yasa sessizce başka
   davranışa düşmemeli; nedenini UI/script/IPC aynı çekirdekten bildirmeli.
5. **Temas çifti:** duvar/collider sürtünmesi, adhesion Pa, temas mesafesi ve
   ayrılma; iç kohezyondan ayrı. Bir çamur tarifi bütün yüzeylere aynı şekilde
   yapışmamalı; malzeme–yüzey çifti ve sıcaklık/nem etkisi açık olmalı.
6. **Görünüm:** render shader'ı ve yüzey/hacim/tane temsili; fizik yasasından ayrı.

Ana menüde model aileleri ve birer referans gösterilebilir; 26 somut madde
aranabilir örnek kütüphanesinde kalır. Böylece yakın örnekleri veri kaybetmeden
seyreltiriz. Iron/Steel/Copper veya Water/Ice/Snow yalnızca benzer isim/yoğunluk
nedeniyle birleştirilmemeli. Mud/Dough/Gum aynı MPM taşıyıcısında olsa da aynı
akma ve gevşeme davranışına sahip değildir.

## Uygulama sırası ve doğrulama

- P0: Oil viskozite atlama görünürlüğü, Stone/Lava etkin sıvı parametreleri,
  eski sönüm ve viskozite birimlerinin ayrılması. Fizik ayarını gizlice değiştirme.
- P1: altı kopyayı kaldırıp aileden türetme akışını mevcut madde servisinde
  birleştir; UI, scripting, IPC, şema ve katalog testlerini birlikte güncelle.
- P2: Chocolate için düzenlileştirilmiş Herschel–Bulkley/Bingham gibi açık akma
  gerilmeli model; çikolata ölçümlerine göre Casson karşılaştırması da yapılabilir.
  Mevcut MPM EVP'yi eşdeğer sıvı modeli gibi yeniden adlandırma.
- P3: Honey için μ(T), PE için μ(T,γ̇); nem/katı oranlı çamur tarifleri daha sonra.

Kabul ölçümleri: Couette/Poiseuille viskozite geri çıkarımı; akma eşiği altında
hareketsizlik ve üstünde akış; üç h ve dt ile yakınsama; elastik geri dönüş/
gevşeme; granüler yığılma; ısıtma-soğutmada enerji/kütle dengesi; yüzeye yapışma
ve ayrılmada kütle korunumu. Backend'ler destekledikleri aynı yasa üzerinde
karşılaştırılmalı. Bunlar sonraki uygulamanın test planıdır; bu tur çalıştırılmadı.

## Dış teknik dayanaklar

- COMSOL: [Newtonyen olmayan akış modelleri](https://doc.comsol.com/6.3/doc/com.comsol.help.cfd/cfd_ug_fluidflow_single.06.082.html).
- Deneysel çalışma: [Dark chocolate viscosity model comparison](https://academic.oup.com/ijfst/article/44/1/162/7864833).
  Çikolata için sabit viskozitenin ötesindeki model ihtiyacını destekler; bizim
  preset değerlerini doğrulamaz.
- SideFX: [MPM Source](https://www.sidefx.com/docs/houdini/nodes/sop/mpmsource.html),
  [MPM Solver](https://www.sidefx.com/docs/houdini/nodes/sop/mpmsolver.html).
  Malzeme davranışı ile çözücü ayrımına örnektir; evrensel sertifika değildir.
