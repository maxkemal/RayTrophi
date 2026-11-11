# Birleşik madde: kimlik, durum ve mekanik davranış

## Kabul edilen yön

Isıl/kimyasal madde kütüphanesi korunur. Demir, kâğıt, ahşap, bakır ve benzeri
profillerin değeri yalnızca parçacık mekaniğiyle değerlendirilmez. Isı iletimi,
nem, yanma, kömürleşme, kül ve faz dönüşümü bağımsız kullanım alanlarıdır.
Bu belge hedef mimaridir. İlk salt-okunur çözümleme raporu uygulanmıştır;
ayrıntılar `MATTER_MATERIAL_RESOLUTION.md` içindedir. Duruma özel yeni mekanik
şema ve yeni fizik yasaları henüz uygulanmış değildir. Katalogdaki altı birebir
Generic kopyası kaldırıldı; 26 somut örnek ve 12 gerçek başlangıç profili kaldı.
Preset fizik düzeltmeleri `BUILTIN_PRESET_CORRECTIONS.md` içinde açıklanır.
Mevcut doğum/fusion model seçimi de `MatterConstitutiveState` çekirdeğinde
birleştirilmiştir. Raporda katı/sıvı mekanik kapsamı, pinned taşıyıcı ile
deformasyon çözen katı ayrımı ve model/durum uyuşmazlık sayıları bulunur.

Madde kimliği, anlık durum, mekanik yasa, sayısal çözüm, temas ve görünüm
ayrı sorumluluklardır. UI, scripting ve IPC bunları aynı çekirdek üzerinden okur
ve düzenler. Ayrı bir ikinci madde kütüphanesi kurulmaz.

Elastik viskozite, akma ve gevşeme için malzemeye ait softening güçleri eklendi;
varsayılanlar önceki yasayı korur. Sakız başlangıç tarifi ayrı güncellendi.
Güncel kapsam ve kalan işler: `ELASTIC_THERMAL_RESPONSE.md`.

## Mevcut koddan yararlanılacak temel

| Sorumluluk | Mevcut konum | Gerekli ayrım |
|---|---|---|
| Madde ve türetme | SubstanceProfile / SubstanceLibrary | Isıl/kimyasal verileri koru; mekanik varsayılanı açık yorumla |
| Maddeyi nesneye bağlama | RtApiSubstance / nesne madde bağları | Isıl kullanım için mekanik çözücü zorunlu olmasın |
| Parçacık kimliği | substance_tag | Faz değişiminde kimliği yalnızca taşıyıcı değişti diye değiştirme |
| Doğum ve faza göre model | MatterSubstanceState | Yeni durum çözümlemesini burada merkezileştir |
| Domain fizik parametreleri | resolveDomainSubstancePhysics | İstenen özellik ile gerçekten uygulanan özelliği ayır |
| Başlangıç sayısal ayarları | applySubstanceSolverHints | Fizik yasasıyla karıştırma; madde düzenlenince kullanıcı ayarını ezme |
| DEM sahipliği | MatterGrainOwnership | Mevcut engel/neden raporlamasını ortak çözümleme yaklaşımına örnek al |

Önemli sınırlama: resolveParticleGranularPhysics ilk uygun parçacığın profilinden
ortak MPM parametreleri seçiyor. Bu, aynı domain içinde her maddenin bağımsız
mekanik yasayla çözüldüğünün kanıtı değildir. Çok malzemeli hedef genişletilmeden
önce domain-geneli parametrelerle parçacığa özgü parametreler ayrılmalı.

## İlk uygulanacak dikey kapsam

Önce davranışı değiştirmeyen, okunabilir bir durum/mekanik çözümleme raporu:

- Kimlik ve tarif: profil adı, varsa türediği profil.
- Temsil: nesne yüzeyi, süreklilik parçacığı veya DEM tanesi.
- Durum girdileri: mevcut sıcaklık ve faz oranları; bilinmiyorsa açıkça bilinmiyor.
- İstenen ve etkin malzeme yasası, taşıyıcı ve backend.
- İstenen ve etkin viskozite; atlama veya desteklenmeme nedeni.
- Alanların kapsamı: profil/parçacık/domain; ortak domain parametresi seçilmişse
  hangi maddenin belirleyici olduğu.

Bu raporun çekirdeği odaklı yeni modülde olmalı. UI, script ve IPC aynı sonucu
kullanmalı; UI içinde ikinci bir karar tablosu bulunmamalı. Hatalı profil adı,
sonlu olmayan sıcaklık ve geçersiz durum girdileri ortak doğrulama hatası vermeli.
Tanımlanmamış mekanik model, mevcut nesnenin ısıl kullanımını engellememeli.
Yeni API adları mevcut madde API şemasına bakılarak uygulama sırasında seçilmeli;
bu belgedeki kavramlar mevcut endpoint iddiası değildir.

## Sonraki değişiklikler

1. Katı/sıvı durumuna ait mekanik tanımları açıklaştır. “Katı davranış tanımsız”
   ile “Fluid” aynı anlamda olmasın. Mevcut Surface/MSF nesnelerini otomatik MPM'e
   çevirme. Tanım yoksa istenen parçacık üretimi anlaşılır hata vermeli.
2. Mevcut doğum, erime ve donma tüketicilerini aynı çözümleyiciye bağla.
   Faz oranı/histerezis ve enerji hesabını yeniden, bağımsız bir yerde kurma.
3. Oil viskozite atlama politikasını ölçümlerle değerlendir; Chocolate için açık
   akma gerilmeli sıvı yasasını ayrı kapsamda uygula. MPM EVP eşdeğer sayılmasın.
4. Yapıldı: gerçek başlangıç örnekleri ile ayrıntılı örnekler menüde ayrılır;
   birebir Generic kopyaları kaldırıldı. UI/API aynı katalog metadata'sını kullanır.

## Korunacak fizik anlamları

- Su, kar ve buz kimyasal olarak ilişkili olsa da mevcut tarifler farklı yoğunluk,
  mikro yapı ve taşıyıcı içerir. İsimlerini birleştirmek termodinamik tutarlılık
  sağlamaz; mevcut profiller bu aşamada birleştirilmez.
- Erime kimyasal kimliği koruyabilir. Yanma ise reaktanlardan ürünlere kütle ve
  enerji aktarımıdır; “aynı madde etiketi sonsuza kadar korunur” kuralı konmaz.
- İç kohezyon ile yüzeye adhesion ayrı tutulur. Temas çifti, mesafe ve durum
  bağımlılığı gelecekte ortak temas servisi içinde ele alınır.
- Kaynak/provenans ve geçerli sıcaklık aralığı olmayan sayılar başlangıç değeri
  olarak gösterilir; doğrulanmış endüstriyel veri iddiası yapılmaz.

## Tamamlanma koşulları

Ortak çekirdek, UI gösterimi, scripting ve IPC yüzeyleri, ortak hata anlamları ve
belgeler birlikte teslim edilir. Salt-okunur rapor için saf çözümleme testleri;
fizik değişikliği için ilgili viskozite/akma/faz/enerji ölçümleri gerekir.
Derleme ve uygulama doğrulamasını kullanıcı yapar. Bu karar belgesi fizik
değişikliği, tamamlanmış API veya çalıştırılmış test olarak raporlanmamalı.
