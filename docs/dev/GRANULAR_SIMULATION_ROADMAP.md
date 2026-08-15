# Granular Simulation Roadmap

> **Durum:** AKTIF — Granuler/MPM calismasinin ana yol haritasi.

## Amaç

Mevcut APIC fluid altyapısını kuru kum ve benzeri granüler malzemeleri taşıyabilecek
bir MPM/APIC constitutive solver'a dönüştürmek. Hedef; parçacıkların birbiri üzerinde
kayması, doğal repose angle oluşturması, darbe ile kopup dağılması ve durduğunda
statik yığın oluşturmasıdır. Vulkan Compute üretim yolu, CPU yolu ise aynı
matematiğin referans ve doğrulama yolu olacaktır.

Her faz, yalnızca kod derlenebilir hale geldiğinde değil, belirlenmiş sahne/test
sonuçları gözlenebilir olduğunda tamamlanmış sayılır.

## Ortak kurallar

- Granüler state doğrudan `FluidParticles` SoA verisinde tutulur; `Triangle` facade
  veya görsel parçacık verisi authoritative state değildir.
- CPU ve Vulkan aynı parametre adlarını, clamp kurallarını, state geçişlerini ve
  hata semantiğini kullanır.
- UI, scripting API ve IPC aynı material/solver service katmanını çağırır.
- Mevcut sıvı presetleri geriye dönük korunur; `sand` yeni solver'a açıkça bağlanır.
- Her faz için kütle, momentum, enerji, parçacık sayısı ve NaN/Inf kontrolleri
  otomatik testlere eklenir.
- Proje build'i Codex tarafından çalıştırılmaz; kullanıcıya faz sonunda kesin
  build/test komutları verilir.

## Faz 0 — Referans ve teşhis kapısı

**Amaç:** Yeni model başlamadan mevcut sıvı ve mevcut `sand` davranışını ölçmek.

**Çıktı:** Tekrarlanabilir CPU/Vulkan karşılaştırma sahneleri ve baseline raporu.

**Test edilebilir sonuç:**

- Serbest düşüşte water/honey/sand mesafesi ölçülür.
- Kum yığını yüksekliği, yayılma yarıçapı ve durma süresi kaydedilir.
- P2G/G2P, pressure, viscosity ve damping süreleri ayrı raporlanır.
- Aynı başlangıç durumunda CPU ve Vulkan farkları tolerans içinde raporlanır.

Bu fazda mevcut fluid iyileştirmeleri de ayrıştırılır: bütün gövdeyi frenleyen
`internal_friction`, yanlışlıkla sıvı kalınlığı yerine kullanılan damping ve
FLIP snapshot sıralaması ayrı regression testleriyle sabitlenir.

## Faz 1 — Granüler state ve CPU constitutive referansı

**Amaç:** Parçacık başına elastik/plastik state eklemek ve CPU'da güvenilir referans
malzeme güncellemesi kurmak.

**Yeni state:** deformation gradient veya eşdeğer küçük state, plastic volume/
hardening, material id ve detach flag. State SoA ve serialization sürümleriyle
uyumlu olmalıdır.

**Model:** Drucker–Prager başlangıç modeli; basınçtan bağımsız çekme ayrılması,
return mapping, iç sürtünme açısı, kohezyon, dilatancy ve sertlik parametreleri.

**Test edilebilir sonuç:**

- Tek parçacık uniaxial compression/shear testinde yield öncesi elastik,
  yield sonrası sınırlı plastik gerilme görülür.
- Sıfır kohezyonlu kum tensile durumda bağ kurmaz.
- Malzeme güncellemesi NaN üretmez ve sabit timestep altında kararlıdır.
- CPU golden testleri aynı girişte bit seviyesinde veya tanımlı toleransta tekrar
  edilebilir.

## Faz 2 — CPU granüler MPM/APIC yolu

**Amaç:** Constitutive state'i gerçek P2G/grid/G2P akışına bağlamak.

**Test edilebilir sonuç:**

- Eğik olukta kum üst tabakası alt tabaka üzerinde kayar.
- Dökme kutusu sahnesinde doğal repose angle oluşur ve yığın durur.
- Düşen kum bloğu çarpışmada parçalanır; parçacıklar yeniden tek sıvı kütlesi
  gibi birleşmez.
- Kütle korunur; parçacık sayısı yalnızca açık detach/reseed politikasına göre
  değişir.

Bu fazda `sand` presetindeki yapay `internal_friction`, yüksek wall damping ve
APIC ağırlığı kaldırılır veya granüler solver tarafından bypass edilir.

## Faz 3 — Vulkan Compute parity

**Amaç:** CPU constitutive ve transport matematiğini Vulkan Compute'a taşımak.

**Test edilebilir sonuç:**

- CPU ve Vulkan aynı üç sahnede pozisyon, hız, toplam kütle ve momentum açısından
  toleranslı eşleşir.
- Shader debug/readback ile yield, detach ve plastic state sayaçları doğrulanır.
- GPU yolu CPU'dan anlamlı biçimde hızlıdır; atomik/indirgeme darboğazları ölçülür.

Deterministik olmayan GPU sıralaması için exact equality yerine tanımlı fiziksel
toleranslar kullanılır; solver algoritması CPU'dan farklılaştırılmaz.

## Faz 4 — Kopma, dağılma ve temas kalitesi

**Amaç:** Granüler görünümün fiziksel olarak anlamlı hale gelmesi.

**Kapsam:** düşük komşulukta detach, tensile cut-off, temas normal/teğetsel
sürtünmesi, statik-dinamik sürtünme ayrımı, yüzey yapışması opsiyonu ve kontrollü
reseed. Görsel parçacık renderer'ı yalnızca state'i gösterir.

**Test edilebilir sonuç:**

- Kum duvara çarpınca akış yönünde parçalanır, yüzeye yapışıp sıvı filmi oluşturmaz.
- Keskin kenardan dökülen malzeme kopuk akış ve ayrı taneler üretir.
- Yatay yüzeyde açı kritik değerin altına indiğinde hareket durur.
- Aynı sahne farklı parçacık çözünürlüğünde benzer repose angle verir.

- Temassız serbest düşüşte hasar tam olarak sıfır kalır; geri kazanılabilir
  grid-crossing gerinimi parçacık sayısı ve voxel çözünürlüğüyle ölçülür.
  Tepe değer kırılma başlangıcına yaklaşırsa transfer yolu MLS-MPM/APIC affine
  reproduksiyon kapısından geçmeden üretim kalitesinde sayılmaz.

### Faz 4B — Kalıcı parça kimliği ve multi-field temas

**Neden temel fiziktir:** Kopmuş veya bağımsız iki küme aynı grid düğümüne
geldiğinde tek hız alanında ortalanamaz. Her temas dalı kendi kütle, momentum ve
hız alanını korumalı; ardından normal itme ve Coulomb teğetsel sürtünme ile
çözülmelidir. Aksi halde bloklar birbirinin içine geçer, yayılır veya yapay olarak
yeniden tek kütleye dönüşür.

**Çıktı:** Parçacık başına kalıcı `fragment_id/contact_field`, grid düğümü başına
sınırlı sayıda malzeme alanı, alanlar arası non-penetration ve statik/dinamik
sürtünme. Hasar bağı kopardığında yeni parça kimliği oluşur; rebonding yalnız
gerçek temas ve iyileşme koşulları sağlandığında kimlikleri yeniden birleştirebilir.

**Test edilebilir sonuç:**

- IPC sahnesinde ardışık bırakılan iki kohezyonlu blok çarpışır, momentum aktarır
  ve birbirinin içinden geçmez.
- Kopmuş iki parça yakın grid düğümlerini paylaşsa bile hızları tek alanda
  ortalanmaz; ayrıldıktan sonra tekrar yapay olarak yapışmaz.
- Temas sonrası toplam kütle korunur, normal bağıl hız penetrasyon üretmez ve
  teğetsel impuls Coulomb sınırını aşmaz.
- Tek blok kum/yığın regresyonu multi-field açıldığında repose angle ve settling
  açısından bozulmaz.

## Faz 5 — Sıvı solver iyileştirmeleri ve ortak altyapı

Granüler model stabil hale geldikten sonra sıvı tarafında yalnızca ölçümle doğrulanmış
iyileştirmeler alınır:

- Fiziksel kinematik viskozitenin grid çözünürlüğü ve timestep ile doğru ölçeklenmesi.
- FLIP snapshot'ın viskozite/pressure sırasından önce alınması.
- Density correction'ın sıvı ve granüler modlarda ayrı yorumlanması.
- Wall slip, collider friction ve no-slip davranışlarının tek bir temas sözleşmesine
  bağlanması.
- Reseed'in momentum ve kütle muhasebesini bozmayacak şekilde sınırlandırılması.
- Water/oil/honey/mud regression sahnelerinde serbest düşüş, shear ve settling
  davranışlarının korunması.

**Test edilebilir sonuç:** Eski sıvı sahneleri bozulmadan kalır; honey serbest
düşüşte water ile aynı yerçekimi ivmesini korur, shear altında daha yavaş akar;
kum ise ayrı constitutive yol ile repose angle ve kayma davranışı gösterir.

**2026-08-15 kütle-korunum kapısı:** Dinamik reseed artık aynı stepte kalabalık
hücrelerden çıkardığından fazla parçacık ekleyemez; mevcut birim-kütleli P2G
yolunda net pozitif reseed yasaktır. Küçük flow-source packing sınırının
biriktirdiği emission borcu da tek yerel pakete sınırlandı. UI, Python ve IPC
`reseed_added_particles` / `reseed_removed_particles` sayaçlarını raporlar.
Kabul testi: `scripts/test/rt_test_fluid_reseed_conservation.py`; emittersiz kapalı
domain 180 step boyunca reset anındaki parçacık sayısını aşmamalıdır.

## Faz 6 — API, IPC, preset ve üretim kapısı

**Amaç:** Özellikleri kullanıcıya güvenilir şekilde açmak.

**Yeni yüzeyler:** granüler material preset/get/set, friction angle, cohesion,
dilatancy, plasticity, detach ve solver backend seçimi. UI, Python scripting ve
IPC aynı service fonksiyonlarını kullanır; hatalı aralıklar aynı hata kodunu döner.

**Üretim kabul testleri:**

- CPU ve Vulkan backend seçimi aynı sahneyi çalıştırır.
- Sand, wet sand, gravel ve cohesive soil presetleri serialize/deserialize edilir.
- IPC smoke testleri parametre doğrulamasını ve state istatistiklerini kapsar.
- Uzun süreli settling testinde parçacık/enerji patlaması ve NaN görülmez.

## Faz 7 — Uzama ve kırılmaya duyarlı SurfaceSDF

Bu faz temel granüler fizik ve Faz 4B multi-field temas tamamlandıktan sonra
başlar. Render sistemi kendi parça tahminini üretmez; fiziğin kalıcı
`fragment_id` ve bağ durumunu authoritative kaynak olarak kullanır.

**Kapsam:**

- Bağları sağlam parçacıklar hamur/plastik gibi tek ve uzayabilen bir SDF
  komşuluğunda kalır.
- Kırılmış farklı fragment kimlikleri aynı Zhu–Bridson ağırlık ortalamasına
  katılamaz; uzamsal olarak yakın olsalar bile yüzey onları yeniden kaynatmaz.
- Hasara bağlı crack-face yarıçapı ve yönlü kernel, ince boyun kopmasını
  korurken sağlam bölgelerde pürüzsüz yüzey üretir.
- `Liquid`, `Cohesive Stretch` ve `Fracture Aware` yüzey profilleri aynı temel
  servis üzerinden UI, scripting API ve IPC'ye açılır; sahne kaydında aynı
  doğrulama/clamp sözleşmesi kullanılır.

**Test edilebilir sonuç:** Sağlam hamur şeridi uzarken yüzeyi kesintisiz kalır;
fizik bağı koptuğu karede iki dal ayrı SDF yüzeylerine dönüşür ve tekrar
yaklaşsalar bile rebonding olmadan görsel olarak birleşmez. SurfaceSDF ile splat
görünümü aynı `fragment_id` ayrımını gösterir.

## Önerilen ilk uygulama sırası

Önce Faz 0 baseline sahneleri ve Faz 1 CPU constitutive referansı hazırlanmalı.
Bu iki fazın çıktısı görülmeden Vulkan shader'ına veya preset tuning'e geçilmemeli;
aksi halde mevcut yapay damping ile yeni granüler fiziği ayırmak mümkün olmaz.
