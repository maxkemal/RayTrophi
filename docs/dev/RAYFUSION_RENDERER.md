# RayFusion — raster ve RT ile melez render

> **Durum:** AKTİF — 0.1 probe kontrol çekirdeği ve UI/Python/IPC denetimi yazıldı (2026-09-07); DERLENMEDİ. GPU üretimi ve görüntü birleşimi henüz yok, çalışan RayFusion modu teslim edilmedi.

## Uygulama takibi — kod, build ve doğrulama ayrı

| Dilim | Kod durumu | Derleme / doğrulama |
|---|---|---|
| 0.1 Probe kontrol çekirdeği | Yazıldı: dünya hücresi cache'i, sınırlı iş planlama, revision/epoch biletleri, paket doğrulama, temporal blend, moment görünürlüğü | Kaynak/IPC denetimi geçti; C++ derlemesi ve 34 native kontrol bekliyor |
| 0.1 Denetim yüzeyleri | UI, `rt.rayfusion.*`, `rayfusion.*` IPC ve capability/descriptor bağlantıları yazıldı | Runtime bağlantısı kullanıcı build'inden sonra doğrulanacak |
| 0.1 Probe penceresi (ızgara) | Yazıldı: hücre sayısı/aralık/yerleşim derleme sabiti olmaktan çıktı, UI + Python + IPC'den ayarlanır, tampon tavana göre bir kez ayrılır ([RAYFUSION_PROBE_GRID.md](RAYFUSION_PROBE_GRID.md)) | Kaynak denetimi geçti; varsayılan değişmedi, kapsam ölçümü kullanıcı build'inden sonra |
| 0.2 RayFusionScene ve GPU sahibi | Başlamadı: flat geometri/instance kimliği, aynı cihaz kaynakları, refit/rebuild ve retirement | Bekliyor |
| 0.3 GPU probe üretimi | Başlamadı: ışın üretimi, hit/material/ışık değerlendirmesi, yönlü texel projeksiyonu, relocation/classification, GPU zamanları | Bekliyor |
| 1 GI + sky birleşimi | Başlamadı: raster tüketicisi, eski diffuse ambient'in kaldırılması, ayrı speküler sky prefilter | Görüntü/FPS testi bekliyor |
| 2 Kararlılık ve bütçe | Başlamadı: GPU geçişleri, yakınsama pump'ı, ışık değişimi, kamera kesmesi, ekran düzeltmesi | Bekliyor |
| 3 Hair | GI'den sonra | Bekliyor |

**Dilim 0.1, GPU kaynak temelinin tamamlandığı anlamına gelmez.** Cihaz ve
sahne epoch'ları bu çekirdeğe dışarıdan verilir; gerçek kaynak sahibi 0.2'de
bağlanacak. `ProbeField` şu an canlı renderer cache'i olarak çalışmıyor;
native testler izole örnekler kullanıyor. Görüntüde GI/sky/FPS değişimi bu
partinin kabul şartı değildir. Ayrıntı ve sıralı test:
[RAYFUSION_PROBE_CORE.md](RAYFUSION_PROBE_CORE.md).

Tamamlandı etiketi kullanıcı derlemesi ve ilgili kabul kapıları geçmeden
konmaz. Planlanan bütçeler, dispatch edilmiş ışın veya ölçülmüş GPU süresi
diye raporlanmaz. Tam bir ikinci path tracer sonraki ihtiyaçlara göre ele
alınır; ortak geometri/materyal otoritesi kopyalanmaz.

## Faz tablosu — dikiş sırası ve durum

Sıralamanın ölçütü: **her adım bir sonrakinin ön koşulunu ödesin.** Dikiş
ambient okumasıdır — tüketici sabit kalır, üretici değişir. Böylece sürpriz
maliyet çıkarsa geri dönüş bedeli sıfırdır.

Durum etiketleri: `YAZILDI` (kod var, derlenmedi) · `DERLENDİ` ·
`ÇALIŞTI` (uygulamada koştu) · `ÖLÇÜLDÜ` (sayı var) · `DOĞRULANDI` (kabul
testi geçti). **Bir faz ancak ÖLÇÜLDÜ'den sonra bitmiş sayılır** — "çalışıyor
gibi görünüyor" bu tabloda bir durum değildir.

| # | Faz | Durum | Ne ödüyor |
|---|---|---|---|
| P | Probe önbelleği + zamanlayıcı (CPU kontrol düzlemi) | **DERLENDİ + ÇALIŞTI** (2026-09-07) | Adım 1'in dünya hücresi/bilet/red sözleşmesi |
| 0 | Prefiltered ambient: Physical Sky → equirect bake → mevcut IBL zinciri | **★ ÖLÇÜLDÜ** (2026-09-07) | Ölçülen 60 ms'lik koni integrali; **ve Adım 1a'nın tüketici arayüzü** |
| 0b | Ana geçişe depth prepass | AÇIK | Overdraw'ın ambient'i tekrar tekrar ödemesi |
| 1a | Probe SSBO + shader'da **konuma bağlı** ambient okuması; ilk üretici sky bake | **★ DOĞRULANDI** (2026-09-07) | Boru hattı: yükleme, GLSL ABI, arama, invalidate — **ışınsız doğrulandı** |
| 2 | Viewport'ta AS ikametgahı + ray query | **★DOĞRULANDI** (2026-09-07) | **GI'nin ÖN KOŞULU** — ışın yoksa dolaylı ışık da yok |
| 1b-α | Probe üreticisi ray query ile **görünürlük** izler | **YAZILDI** (2026-09-07) | Görünürlüğe bağlı ambient; `kUnoccludedDistance` sabiti ölür |
| 1b-β | İsabet noktasında albedo + sıçrama | **İLK DİLİM YAZILDI** (derlenmedi) | Dokusuz opak diffuse/emission + ortam/point/directional; geniş materyal kapsamı ve GPU kabulü açık |
| 1c | Probe tüketicisinde mesafe görünürlüğü + 8-probe harmanlama | **YAZILDI** (2026-09-08, derlenmedi) | 1b-β sonrası mavi ışık kaçağı; [uygulama ve kabul](RAYFUSION_PROBE_SAMPLING.md) |
| 1d | Speküler sky için yönlü probe görünürlük tahmini | **YAZILDI** (2026-09-08, derlenmedi) | Kıvrımlarda kalan sky tabakası; [sınırlar ve kabul](RAYFUSION_SPECULAR_VISIBILITY.md) |
| 1e | Derinlikli probe overlay + bütçeli kamera takibi | **YAZILDI** (2026-09-08, derlenmedi) | [Kontroller, ilk otomatik yerleşim sınırı ve kabul](RAYFUSION_PROBE_OVERLAY.md) |
| 3 | Hacim yolu (gas/VDB/SDF froxel marşı) ve iki yolun yan yana ölçümü | ERTELENDİ — kullanıcı kararı | Mevcut yol korunur; ölçülmüş ihtiyaç olmadan yeni yol/seçici eklenmez |
| 4 | Hair'in bu aydınlatmaya bağlanması | AÇIK | — |

★★★ **Sıra düzeltildi (2026-09-07).** Önceki tabloda Adım 1 (probe alanı),
Adım 2'den (AS) önce geliyordu. Bu yanlıştı: bir probe'un ışığı **görünürlüğe**
bağlı, görünürlük de ışına. AS olmadan probe alanı gökyüzünü tekrar üretmekten
başka bir şey yapamaz — yani GI değil, sky'ın kopyası olur. Bu yüzden üretici
tarafı (1b) AS'nin arkasına alındı; tüketici tarafı (1a) ışın gerektirmediği
için önde kalabilir ve AS geldiği gün doğrulanabilir hale gelir.

### P — Probe önbelleği (YAZILDI, derlenmedi)

`RayFusion::ProbeField` (CPU): dünya hücresi + generation + tek kullanımlık
serial ile eşleşen bilet; geç gelen eski sonuç geri çevrilir; toroidal scroll
yalnız yeni hücreleri sıfırlar; bütçe kalite preset'inden gelir. Beş dokunuş
tamam: `RtApi` · `RtIpc` · `RtPython` · capability · üretilmiş descriptor +
overlay. `rayfusion.core_status` bilerek **`gi_active=false`** raporluyor.

Bu fazda **bilinerek eksik** olanlar — sonraki fazın işi, eksiklik değil:
scheduler yaş/belirsizlik sırası tutmuyor (round-robin kürsör), ve
`targetUpdates`'e ulaşan probe `invalidate` gelmeden bir daha planlanmıyor.
Yani ışık yavaşça değişip revizyonu büyütmüyorsa alan tazelenmez.

### Adım 0 — Prefiltered ambient (★ ÖLÇÜLDÜ, 2026-09-07)

Ölçüm: 1680x945 / 695k üçgen karenin **102 ms'inin ~60 ms'i** Physical Sky'ın
ambient koni integraliydi (lob başına 9 tap, fragment başına iki lob, depth
prepass'siz forward geçişte). Doku + cam + gölge + DoF hepsi birlikte ~3 ms.

Yapılan: sky bir **yön fonksiyonu**, o yüzden yönlü önbelleğe taşındı.

- `canonical_world.glsl` — `sampleCanonicalWorldEx(d, includeSunDisc)` tek
  üretici olarak ayrıldı; fragment de bake de **aynı** fonksiyonu çağırıyor.
- `world_sky_capture.comp` — kanonik dünyayı 256x128 equirect'e basıyor;
  tüketicinin `worldDirToUV` rotasyonu bake'te geri ekleniyor.
- Mevcut `material_preview_ibl.comp` zinciri (irradiance + 9 GGX mip + BRDF)
  **değişmeden** bu bake'i tüketiyor: yeni konvolüsyon kodu yok.
- `packPreviewWorldUniforms` — dünya uniform'larını paketleyen **tek** yer;
  önceki kopya `MaterialPreviewShadow.cpp`'den söküldü.
- Shader kapısı `worldMode == 1u && (sceneFlags & 32u)` → `(sceneFlags & 32u)`.

★★ **Güneş diski bake'e girmiyor.** Güneş raster yoluna kendi gölge atlası
tile'ı olan analitik yönlü ışık olarak ulaşıyor; diski irradiance'a da gömmek
her difüz yüzeyi **iki kez** aydınlatırdı — bugünkü koni integrali tam olarak
bunu yapıyor. Bu yüzden Adım 0 aynı zamanda bir **düzeltme**: parlaklık
değişecek, ve bu beklenen davranıştır. Ayna yüzeyler (roughness <= 0.025)
güneşi görmeye devam etsin diye tek doğrudan örnek alıyor — dokuz taplık koni
değil.

#### Ölçüldü (2026-09-07, çalışan uygulamada)

Aynı sahne, aynı çözünürlük: 1680x945, 695 735 üçgen, 67 draw, kalite `full`,
shading `material`, Nishita + HDR + DoF açık. `viewport.preview_lighting` →
`world_ibl_source = "sky"`, `world_ibl_ready = true`.

**Nedensellik için orijinal ayırıcı test** (gökyüzünü kapat/aç), sürekli
rejimde kamera sürülerek:

| konfigürasyon | backend frame_ms (medyan) |
|---|---:|
| Physical Sky AÇIK | **1,121** |
| `world.set_mode solid` (gökyüzü KAPALI) | **1,099** |
| fark = gökyüzünün karedeki payı | **~0,02 ms** |

2026-09-07 sabahındaki aynı ayırıcı test **102,0 → 38,6 ms** vermişti, yani
gökyüzü ~60 ms'ti. **Ambient koni integrali karede artık yok.**

★★ Dürüstlük kaydı: karenin tamamı 102 ms'ten ~1,1 ms'e inmedi *yalnızca* bu
değişiklikle. Aynı build başka perf işleri de taşıyor. Nedensel olarak
atfedilebilen tek şey **gökyüzü terimi**: 60 ms → 0,02 ms. Geri kalanı bu
belge sahiplenmez.

★ Aynı ölçümde çıkan ve **bu işle ilgisi olmayan** açık bulgu: sürekli rejimde
`resource_drains = 1,00/kare` ve `stale_presents = 1,00/kare`. Yani kamerayı
oynatmak her karede GPU'nun okuduğu bir kaynağı mutasyona uğratıyor. Probe bunu
bağımsız bir açık olarak işaretliyor; IBL bağlama yolu değil (`materialPreviewDescSet`
tek ve sabit, ilk bind'dan sonra yeniden bağlanmıyor).

★ Ölçü aleti: `viewport.preview_lighting` artık `world_ibl_source`
(`hdri` | `sky` | `none`) ve `world_sky_capture_supported` döndürüyor.
`world_ibl_ready` tek başına hangi üreticinin çalıştığını söyleyemezdi — ve iki
yolun kare maliyeti aynı değil.

## Ürün hedefi ve öncelik

Kullanıcının son düzeltmesi: hedef yalnız Realtime'a GI eklemek değil,
**RayFusion adlı melez bir render modu** kurmaktır. Raster ana görünürlük ve
uygun doğrudan ışık işini; RT sahne görünürlüğüne bağlı dolaylı ışık ve gerekli
yansıma işini üstlenir. İlk görsel teslim **doğru dolaylı aydınlatma ağı**,
ardından **hair malzemesinin bu aydınlatmaya bağlanmasıdır**.

RayFusion bir GPU/backend markası değil, görüntü üretim yöntemidir. İlk
uygulama Vulkan raster + Vulkan RT üzerinde, aynı VkDevice ve aynı sahne
kaynaklarıyla tasarlanır. Mevcut modları yalnız yeniden adlandırmak bu hedefi
karşılamaz. UI'da hazır bir mod gibi listelenmeden önce gerçek pass zinciri,
capability kapısı, hata durumu ve ölçüm yüzeyleri tamamlanır.

Yaklaşımın özü, ekran uzayındaki ucuz veriyi dünya uzayında tutulan görünürlük
ve ışık önbelleğiyle birleştirmektir: ekran izleri hızlıdır ama yalnız görünür
olanı bilir; dünya uzayındaki önbellek kadraj dışını da taşır. Bu, bir sınıf
yöntemin ortak fikridir — burada kurulan ışık ağı başka bir uygulamanın
mimarisini, kalite ya da maliyet davranışını devralmaz. Geçerli olan tek
sözleşme bu belgede tanımlanan ve ölçülen davranıştır.

## Mevcut kodun getirdiği sınırlar

- `VulkanViewportBackend` ve render backend'i ayrı Vulkan kaynak sahipleridir.
  Başka VkDevice'da oluşturulmuş buffer, descriptor veya AS handle'ı ödünç
  alınamaz. Ortak CPU otoritesi, ortak GPU kaynak sahipliği anlamına gelmez.
- `ViewportMode` şu an Rendered/Solid/MaterialPreview/Matcap içeriyor. API ve
  UI shading eşlemesi de bu dört mod üzerinden çalışıyor; yalnız enum eklemek
  bütün çağıranları RayFusion'a taşımış olmaz.
- Rasterda HDR color ve depth mevcut. Ayrı GI normal/material/velocity ve
  history hedefleri ile geçerlilik sözleşmesi henüz kurulmuş değil.
- Vulkan cihaz kurulumu AS + RT pipeline özelliklerini ele alıyor. Ray-query
  kullanılacaksa extension/feature desteği ayrıca sorgulanıp etkinleştirilmeli;
  RT pipeline desteği ray-query'nin hazır olduğunu kanıtlamaz. İlk tracing
  adaptörü mevcut RT pipeline kabiliyetine dayanabilir.
- [Eski Realtime yol haritası](REALTIME_RENDERER_ROADMAP.md), sökülen ayrı
  modda framebuffer usage, render-pass uyumu, resize ve başarı raporlaması
  hatalarını kaydeder. Yeni kullanıcı kararı o modun tekrar kopyalanması değil;
  bu başarısızlık kapıları RayFusion kabul testlerine taşınır.

## Görüntü üretim zinciri

| İş | Sorumlu yol | Sözleşme |
|---|---|---|
| Ana görünürlük | Raster depth + yüzey verisi | Dünya normal'i, materyal bilgisi, hareket/reprojection verisi; alpha/graph görünürlüğü tutarlı |
| Doğrudan aydınlatma | Raster shading + mevcut uygun gölgeler | Aynı ışık ve materyal otoritesi; görünür pikselin doğrudan ışığı bir kez |
| Dünya uzayında dolaylı ışık | Bütçeli RT probe güncellemeleri | Gelen ışık ve yönlü mesafe/görünürlük; ekran dışındaki geometri dahil |
| Yakın ayrıntı | İkinci aşamada ekran uzayı düzeltmesi | Probe sonucu ile güven ağırlıklı birleşim; iki kez ışık ekleme yok |
| Speküler yansıma | Roughness/bütçe seçimiyle environment, ekran izi ve gerektiğinde RT | Diffuse irradiance, ayna yansıması yerine kullanılamaz |
| Hacim (gas, VDB, SDF) | Froxel/ekran uzayı ray-march (raster-compute) | Opak depth'e karşı marş; aynı scene-linear HDR; GI ile çift sayım yok |
| Hair | GI tesliminden sonra HDR hair geçişi | Aynı ışık/material kaynakları, tel yönüne bağlı saç BSDF'i ve coverage |
| Birleştirme | Scene-linear HDR | Emission + doğrudan + dolaylı; DoF/pozlama/tonemap en sonda |

Sky/HDRI dış ortam ışığı kaynağıdır. GI aktifken mevcut görünürlükten bağımsız
diffuse sky terimi ayrıca eklenmez. Sky prefilter tek başına GI değildir;
duvarın arkasında gökyüzü görülmediği sürece ışık içeri ulaşamaz. Speküler
environment katkısı kendi enerji payı ve görünürlük sözleşmesiyle ayrıdır.

## Dünya uzayında ışık ağı

Izgaraya yerleşen probe'lar iki şey saklar: yönlü irradiance ve aynı yönlerde
mesafe momentleri (ortalama ve ortalama kare). Güncelleme, probe'dan bütçeli bir
ışın demeti atıp gelen ışığı ve isabet mesafesini bu iki haritaya projelemektir.
Yüzey yakın probe'lardan örnek alırken ağırlık, mesafe momentlerinden türeyen
görünürlük tahminiyle çarpılır — arada duvar varsa o probe'un payı düşer.
Duvar içinde kalan probe'lar sınıflandırılır, gerektiğinde taşınır. İnce
duvarlar ve küçük odalar kaçak testidir; sırf yumuşak görünen renk alanı
başarı değildir.

İlk teslim doğrudan aydınlanan yüzeylerden bir diffuse bounce ve görünürlüğü
hesaplanan çevre ışığıdır. Sonraki bounce'lar yalnız önceki tamamlanmış ışık
alanından, enerji kontrollü güncellemeyle eklenir. Post-process görmüş son
kareyi ışık kaynağı olarak geri beslemek yasaktır. Küçük emissive yüzeyler ve
dar pencere açıklıkları için örnekleme eksikliği ayrı ölçülür.

Sahne büyükse kameranın çevresinde kademeli probe alanları kullanılır.
Kamera dönüşü geometri rebuild sebebi değildir. Alan kaydırıldığında yalnız
yeni hücreler başlatılır; eski hücrenin ışığı başka dünya konumuna taşınmaz.
Tüm probe'lar her kare güncellenmez: değişen ışık/geometri, yeni açılan alan,
probe yaşı ve belirsizliği sırayı belirler. Sabit kamerada da yakınsama işi
bitene kadar frame pump çalışmalı; mevcut idle kapısı işi erken kesmemeli.

## Yolu iş sınıfı seçer, sabit rol değil

Melez olmanın anlamı iki motoru yan yana koymak değil, **her işi ucuz olduğu
yola vermektir.** İki yolun maliyet eğrisi aynı yerde kırılmaz:

- **Yoğun opak geometri:** raster maliyeti çizilen instance/üçgen sayısı ve
  overdraw ile doğrusal büyür. Işın izleme hızlanma yapısı üzerinde logaritmiğe
  yakın davranır ve görünmeyen yüzeyi hiç ödemez; instance sayısı arttıkça
  birincil görünürlükte bile rekabetçi olabilir. Ama bu hesaba **AS kurma/refit
  borcu dahildir**: deforme olan veya topolojisi değişen geometride kırılma
  noktası geri kayar. Statik kalabalık ile animasyonlu kalabalık aynı soru
  değildir.
- **Katılımcı ortam (gas, VDB, SDF):** iş yüzey bulmak değil, ışın boyunca
  integral almaktır. Froxel ızgarasında veya ekran uzayında yürüyen
  raster/compute yolu örnekleri komşu pikseller arasında paylaşır, düşük
  çözünürlükte hesaplayıp yukarı örnekleyebilir ve adım sayısını doğrudan
  bütçeler. Aynı integrali piksel başına bağımsız RT ışınıyla almak, gölge ve
  ikincil ışın eklendiğinde hızla pahalılaşır.

Bu yüzden belgenin başındaki "raster görünürlük + RT dolaylı ışık" bölüşümü
**ilk uygulama tercihidir, mimari kural değil.** Sözleşme şudur:

> Bir iş sınıfı için birden çok yol olabilir; hangisinin koştuğunu **ölçüm ve
> bütçe** seçer — panel değil, varsayılan değil.

Seçicinin (router) uyması gerekenler:

1. **Girdiler sayılabilir olmalı:** görünür instance/üçgen sayısı, overdraw
   tahmini, bekleyen AS güncelleme borcu, hacim domain sayısı ve voxel
   yoğunluğu, hacmin ekranda kapladığı alan, kalite preset bütçesi. Sayılamayan
   bir girdiye dayanan seçim, sabitlenmiş bir tahmindir.
2. **Seçim raporlanmalı:** iş sınıfı başına `chosen_path`, `reason` ve o yola
   giden GPU ms. Ölçülmeyen zaman `available=false`. Hangi yolun koştuğunu
   göstermeyen bir melez sistemde performans regresyonu **görünmez** — yol
   sessizce değişir, kullanıcı yalnız "bazen yavaş" der.
3. **Geçiş histerezisli olmalı:** eşiğin etrafında salınan seçici, ortalama
   maliyeti korusa bile görüntüde ve kare süresinde titreme üretir. Her eşik
   çifttir (aç/kapa) ve geçişin kendi maliyeti bütçeye dahildir.
4. **İki yol aynı sonucu üretmek zorunda:** aynı ışık/materyal otoritesi, aynı
   scene-linear HDR uzayı. Yol değiştiğinde parlaklık zıplıyorsa bu bir
   ayar meselesi değil, yollardan birinin **yanlış** olduğunun kanıtıdır.
   Kabul testi: aynı sahneyi iki yolla da üret, farkı ölç; eşik üstü fark FAIL.
5. **Karma aynı karede olur:** opak geometri RT ile, hacim raster-marş ile aynı
   karede koşabilir. Bu durumda derinlik/görünürlük el değiştirmesi **tek
   yerde** tanımlanır: depth'i kim yazar, hacim hangi depth'e karşı marş eder,
   birleştirme sırası nedir. Bu tanım ikiye kopyalanırsa iki yol bir süre
   sonra ayrışır.

Sıra önemli: **seçici ölçümden sonra gelir.** Önce her iş sınıfı için iki yol da
tek tek koşturulabilir ve ayrı ayrı ölçülebilir olmalı (elle zorlanabilen bir
override ile), sonra otomatik seçim yazılır. Tersi yapılırsa router'ın kararı
doğrulanamaz; yalnız inanılır.

### Adım 1a — Probe boru hattı (★ DOĞRULANDI, 2026-09-07)

Amaç **görüntüyü değiştirmek değil**, ambient değerinin geçtiği boruyu
değiştirmek. Değer hâlâ aynı gökyüzü bake'inden geliyor; değişen tek şey, ışın
geldiği gün yalnızca **üreticinin** değişecek olması.

- `probe_field.glsl` — 32 baytlık `ProbeTexel` ABI'si (`RayFusion::ProbeTexel`
  ile birebir), binding 21 texel SSBO'su, binding 22 ızgara parametreleri.
  `rfSampleProbeField(worldPos, N)` kapsıyorsa `true` döner; kapsamıyorsa çağıran
  mevcut global okumada kalır — **kapsanmayan piksel sessizce kararmaz.**
- `MaterialPreviewProbeField.cpp` — CPU üreticisi. IBL zinciri irradiance'ı
  pişirdiği anda o görüntü (64x32 RGBA32F) hâlâ `GENERAL` layout'tayken staging'e
  kopyalanıyor, CPU'ya iniyor, ve her probe'un 64 oktahedral texel'i o haritadan
  örnekleniyor. **Gökyüzü ikinci kez hesaplanmıyor** — ikinci bir üretici, iki
  ayrı sonuç demek olurdu.
- Zamanlama gerçek: `budgetForQuality` bütçesi kadar bilet, `publish` ile
  yayın, kabul edilen slot varsa GPU tamponu yazılır. `performance` preset'inde
  32 probe dört karede dolar — sayaçların **hareket etmesi** boruyu görmenin
  tek yolu, çünkü görüntünün değişmemesi kasıt.

Bilerek eksik olanlar (sıra, eksiklik değil):
- **Kaydırma yok.** Pencere sabit (`counts 4x2x4`, `spacing 3 m`, minimum
  `-2,-1,-2`). Kaydırma torusal slot eşlemesini değiştirir, ve kamerayla birlikte
  değişen bir üretici henüz yok.
- **Görünürlük yok.** Mesafe momentleri "kapanmamış" yazılıyor. Küçük bir mesafe
  uydurmak, hiç yapmadığımız bir ölçümü rapor etmek olurdu.
- **İçerik konumdan bağımsız.** Alan bugün "her hücre açık gökyüzünü görüyor"
  diyor — ve bu bugün **doğru**.

★ Ölçü aleti: `rayfusion.probe_field` → `producer`, `budget_preset`,
`total/valid/pending/in_flight`, `accepted/rejected`, ızgara. `gi_active` hâlâ
`false`: boru gerçek, ışık henüz yeni değil.

#### Doğrulandı (2026-09-07, çalışan uygulamada, iç mekan sahnesi)

★★★ Bu fazın kabul ölçütü terstir: **görüntünün AYNI kalması** başarıdır.
O yüzden "gözle fark yok" tek başına bir kanıt değil — boru ölüyken de aynı
görünür. Ayıran şey sayaçlar ve kapsama:

| ölçüm | sonuç |
|---|---|
| `rayfusion.probe_field` | `supported/configured/uploaded/bound` hepsi `true` |
| alan doluluğu | `valid = 32 / total = 32`, `pending = 0` |
| geç/bayat sonuç | `rejected = 0` |
| üretici | `producer = "sky_bake"` |
| boşta churn | 3 sn boşta `accepted` **sabit** (1600 → 1600), imza sabit |
| tüketici kapsaması | 60 nesnenin **58'i** probe penceresinin içinde |
| görüntü | kutu sınırında dikiş yok, davranış değişmedi (kullanıcı testi) |

★★ `accepted = 1600` = 50 tam dolum; kare başına değil, oturum boyunca yapılan
~50 dünya değişikliğinden (mod geçişleri, güneş düzenlemeleri, ölçüm A/B'si).
Boşta sabit kalması bunu ayırıyor — kare başına yeniden dolsaydı sayaç akardı.

★ Kapsama neden önemli: probe penceresi dünya merkezinde 12x6x12 m. Sahne
tamamen dışında kalsaydı her piksel fallback'e düşerdi ve "fark yok" **hiçbir
şey** söylemezdi. Sahne pencereyi kesiyor (`default_Cube` x=11 ve
`1_Floor.001` y=3,5 dışarıda), yani hem probe yolu hem fallback aynı karede
koştu ve aralarında görünür fark çıkmadı.

★ Kalan boşluk, dürüstlük kaydı: bu ölçüm oktahedral eşlemenin **bit düzeyinde**
doğru olduğunu kanıtlamaz; ayrışmış bir eşleme kutu sınırında fark üretirdi ve
üretmedi — bu güçlü ama nihai değil. Kesin kanıt, probe içeriğini bilerek
farklı bir değere çevirip görüntünün değişmesini görmek olurdu (Adım 1b'de
ışınlar geldiğinde bu zaten kendiliğinden olacak).

### ★★★★ Adım 2 sanıldığından KÜÇÜK: cihaz zaten hazır

Doğrulandı (2026-09-07): `VulkanViewportBackend` `initialize()`'ı **override
etmiyor**, yani raster viewport'un VkDevice'ı `VulkanBackendAdapter::initialize()`
üzerinden `m_device->initialize(true, ...)` ile kuruluyor — `preferHardwareRT =
true`. Donanım destekliyorken `VK_KHR_acceleration_structure`,
`VK_KHR_ray_tracing_pipeline` ve `VK_KHR_ray_query` o cihazda **zaten etkin**.

Eksik olan yetenek değil, **ikametgah**: `VulkanViewportBackend.cpp` içinde tek
satır AS kodu yok (`AccelerationStructure|rayQuery|buildTlas` → 0 eşleşme).
Yani Adım 2 "raster cihazında RT'yi pazarlık et" değil, "zaten RT yapabilen bir
cihazda BLAS/TLAS kur ve compute'ta ray query kullan" işi.

Bu, maliyeti ortadan kaldırmaz — AS belleği ve deformasyon/skinning/scatter için
refit borcu duruyor, ve o borç GI kullanılmasa bile her karede ödenir. Ama
kırılma noktasını öne çeker: fatura kesildiği anda yoğun geometride RT
görünürlük yeni bir maliyet değil, marjinal bir soru olur.

### Adım 2 — AS ikametgahı (★DOĞRULANDI)

Raster viewport artık kendi cihazında BLAS/TLAS tutuyor. Cihaz zaten
`preferHardwareRT = true` ile kurulduğu için eksik olan yetenek değildi,
**ikametgahtı**.

- `createTriangleBLAS_Device` — BLAS'ı **zaten cihazda olan** raster pozisyon
  tamponundan kuruyor. Mevcut `createBLAS` CPU işaretçisi alıyor, yani her
  vertex'i ikinci kez yüklerdi ve bu adımın ölçmek istediği fatura iki katı
  görünürdü. Geometri flat SoA: indeks yok, üçgenler ardışık üçlüler.
- Raster pozisyon tamponuna `BufferUsage::ACCELERATION` eklendi. Bayrak bellek
  maliyeti getirmez; **olmadan** tampon AS kurulumuna girdi olarak okunamaz ve
  tek belirti build anında bir validation hatası olurdu.
- BLAS `externalGeometry = true` ile işaretli: vertex tamponu **ödünç**.
  Yıkımda onu da silmek, çizilen geometriyi AS ile birlikte götürürdü.
- Kapı **iki içerik imzası**: `geometry_signature` (mesh kümesi) ve
  `instance_signature` (yerleşimler **ve maske**). İlki değişirse bütün BLAS'lar
  yeniden kurulur; yalnız ikincisi değişirse **TLAS tek başına** tazelenir. AS
  yıkımından önce `drainInteractiveViewportInFlight()` — uçuştaki bir karenin
  izlediği AS'yi yıkmak arıza değil **cihaz kaybıdır**.

  ★★★★ İlk sürüm kapıyı `g_scene_geometry_generation`'a bağlamıştı ve **canlı
  testte yakalandı**: AS **silinmiş bir küpü** izlenen sahnede tuttu. Sebep
  ölçüldüğünde ilk teshisimden farktı çıktı: **`scene.delete` silmez, gizler**
  (`mask = 0`), mesh de instance da undo için yerinde kalır, dolayısıyla kuşağın
  artmaması **doğrudur**. Maskesi 0 olan instance artık TLAS'a hiç girmiyor ve
  `instances_hidden` olarak raporlanıyor.

  ★★ **`blas_count` bir ikametgah sayısıdır, sahne içeriği değil** — gizli mesh
  raster vertex tamponunu koruduğu için BLAS'ını da korur, ve silmede düşmez.
  BLAS'ı görünürlüğe bağlamak, yerleşim-tazelemesinin artık var olmayan bir
  BLAS'a atıf yapması demekti. İzlenen sahneyi anlatan sayı `instance_count`.
  Ekleme sayacı artırdığı için yalnız "ekle" ile test edilseydi geçerdi.
  Kök neden ve açık kalanlar:
  [BUG_DELETE_LEAVES_STALE_GEOMETRY_SNAPSHOT.md](BUG_DELETE_LEAVES_STALE_GEOMETRY_SNAPSHOT.md).
  Kapının kendi maliyeti `signature_ms` olarak raporlanıyor — bir kapının
  koruduğu şeyden ucuz olması yetmez, **kaçırmaması** da gerekir.
★ **Ölçülen (2026-09-07):** 67 mesh'lik sahnede 67 BLAS / 67 instance / 12,2 MB,
`builds = 1`, `last_build_ms ≈ 126`, hiçbir mesh veya instance atlanmadı. 12,2 MB
başlı başına bir doğrulama: pozisyon tamponları tek başına 25 MB, yani BLAS
**ödünç tampondan** kuruldu, kopyadan değil. Kamera 10 sn gezdirildi, `builds`
sabit. Ekle/sil/taşı turlarının hepsi geçti, `device_lost = 0`, kapının kendi
maliyeti `signature_ms` = 0,0009 ms. Adım 0/1a regresyon görmedi: `world_ibl_source`
= `sky`, probe alanı 32/32 geçerli, `rejected = 0`, `accepted` boşta sabit.

- Sahiplik sınırı: `VulkanBackendAdapter` hem raster viewport'un hem render
  backend'inin ortak tabanı. `m_blasList` render backend'inde RT sahnesidir, ve
  oradan yeniden kurmak renderer'ın geometrisini silerdi. Bu yüzden yol
  **yalnızca tamamen kendisine ait** bir listeye dokunuyor; değilse gerekçesini
  raporlayıp çekiliyor.

Bilerek dışarıda bırakılanlar, sessizce değil **raporlanarak**:
- Scatter impostor proxy'leri AS'ye girmiyor (kameraya bakan bir pano, ışının
  gördüğü dünyaya konulamaz).
- Instance tavanı 65536; aşan sayı `instances_skipped` olarak raporlanıyor.
  Raster görüntüde olup AS'de olmayan instance, ışının **boşluk** olarak
  okuyacağı bir deliktir — sessiz kalamaz.
- Refit yok: geometri deforme oluyorsa şu an tam yeniden kurulum. Refit borcu
  Adım 1b ile ölçülecek.

★★★ **Donanım yoksa UI pasif ve gerekçeli.** `rayfusion.scene_as.hardware_rt`
false ise panel bunu turuncu bir satırla ve **çekirdekten gelen** gerekçeyle
söylüyor. "Yapamaz" ile "henüz yapmadı" aynı gösterilemez; kontrolü aktif
bırakıp sessizce çalışmamasına izin vermek bu deponun en pahalı hata sınıfıdır.
Işınla sürülen ilk kontrol eklendiğinde (1b) o kontrol `ImGui::BeginDisabled`
ile aynı gerekçeye sarılmalı.

★ Ölçü aleti: `rayfusion.scene_as` → `hardware_rt`, `ready`, `blas_count`,
`instance_count`, `instances_skipped`, `meshes_skipped`, `as_bytes`,
`last_build_ms`, `builds`, `inactive_reason`. **`builds` yalnız kamera
oynarken artıyorsa bu bir hatadır** — kapı görünüme değil geometri kuşağına
bağlı.

### Adım 1b-α — üretici ART IK IŞIN İZLİYOR (YAZILDI, derlenmedi)

★★★ **Tüketici bu partide tek satır değişmedi.** `probe_field.glsl` ve
`material_preview_frag.frag` aynen duruyor. Bu bir üslup tercihi değil
ÖLÇÜMÜN ŞARTI: tüketici sabitken görüntüde çıkan fark yalnızca üreticiden
gelebilir. İki taraf aynı anda değişseydi "GI geldi mi" sorusu cevapsız kalırdı.

**Ne izleniyor:** her workgroup bir probe; 64 çağrı önce 64 ışın izler
(deterministik Fibonacci küresi, jitter YOK — gürültü olsaydı süzmek gerekirdi ve
süzme ölçümü bulanıklaştırırdı), sonra AYNI 64 çağrı bu paylaşılan ışın kümesinden
64 oktahedral texel'i kosinüs ağırlığıyla toplar. Texel başına ayrı ışın atmak
aynı bilgiyi 64 kez satın almak olurdu.

**★★ Birim tüketiciye ait.** `rfSampleProbeField` normal yönünde TEK texel okur
ve doğrudan albedo ile çarpar, yani bir texel "normali bu olan yüzeyin gördüğü
kosinüs ağırlıklı ORTALAMA ışıma"dır (irradiance/PI). Bu yüzden kaynak,
ön-integre edilmiş irradiance haritası DEĞİL prefiltered ortamın mip 0'ı (roughness
0 = ham ışıma). Ön-integre haritayı kullanmak kosinüsü iki kez uygulamak olurdu.
Mip 0 aynı zamanda HDRI ve Physical Sky için AYNI dokudur, yani üretici hangi
dünyanın açık olduğunu bilmek zorunda kalmıyor.

**İsabet = karanlık, şimdilik.** Yüzeyin kendi albedosu ve sıçrayan ışık 1b-β'nin
işi. Buraya bir sabit albedo koymak ÖLÇÜLMEMİŞ bir sayıyı ölçüm gibi göstermek
olurdu; sıfır ise dürüst bir alt sınır ve farkı tam olarak GÖRÜNÜRLÜK kadar yapar.
Pratik sonucu: kapalı alanlar tam bir GI çözümünden **daha karanlık** okunur.

**Geometrinin içinde doğan probe.** Sabit ızgara bir probe'u duvarın içine
koyabilir; o probe'un bütün ışınları çarpar ve cevabı "siyah" çıkar. Ama doğru
cevap siyah değil **"ölçülemedi"**dir. Arka yüz oranı %35'i geçen probe alfa 0 ile
yayınlanıyor: slot GEÇERLİ sayılıyor (yoksa her kare yeniden izlenirdi) ama
`rfSampleProbeField` onun için false dönüyor, yani piksel 1a'dan beri var olan
geri düşme yoluna gidiyor. Bu kapı zaten tasarlanmıştı; 1b onu ilk kez kullandı.

**Revizyon artık geometriyi de taşıyor.** İzlenen üretici gökyüzüne olduğu kadar
GEOMETRİYE de bağlı. Yalnız gökyüzü imzasıyla, bir duvarı taşımak her probe'u
artık orada olmayan bir duvarın gölgesiyle bırakırdı — geçerli, değişmemiş ve
yanlış, hiçbir hata mesajı olmadan.

★★★ **Kol kapatılabilir, ve bu zorunlu.** `rayfusion.set_probe_producer`
`traced=false` ile 1a'nın sky bake üreticisine geri dönüyör. Yerini aldığı
üreticiye karşı AYNI sahnede karşılaştırılamayan bir üretici ölçülemez; bu depo
"arızalı yolu kapatan düzeltme ölçümü de öldürür" dersini bir kez ödedi.
Panel kontrolü, donanım yoksa `ImGui::BeginDisabled` ile ve **çekirdekten gelen**
gerekçeyle kapalı — 1a'da verilen söz buydu.

★ **Bu partinin en sinsi arızası:** `hit_fraction = 0`. Işınlar sahneyi hiç
görmüyorsa her yön gökyüzü olur ve görüntü 1a ile **BİREBİR aynı** çıkar — yani
doğru görünen bir hiçlik. Kimse bunu bug diye raporlamaz. Ölçü aleti bu yüzden
`rayfusion.probe_field.hit_fraction` olarak dışarı veriliyor ve panelde yüzde
olarak yazılıyor.

★ **Bilinen maliyet, tahmin edilip çözülmedi:** gönderim SENKRON (bekle + geri
oku). Alan dolduğunda `schedule()` boş döndüğü için bu bir kerelik bir bedel; ama
bir nesne gizmo ile sürüklenirken instance imzası her kare değişir ve durak her
kare ödenir. Asenkron okuma bunu kaldırır, ama önce `trace_ms` ile ÖLÇÜLMELİ.

★ Ölçü aleti: `rayfusion.probe_field` → `producer` (ÇALIŞAN, istenen değil),
`producer_traced_requested`, `producer_reason`, `hit_fraction`, `rejected_inside`,
`trace_ms`, `traced_publishes`. Kol: `rayfusion.set_probe_producer {traced}`.

### 1b-α kararlılık kapısı — ikinci Add sonrası TDR (2026-09-07)

**Kullanıcı yeniden testi:** TDR tekrarlanmadı. Bu sonuç ikinci Add düzeltmesini
destekler; tüm stres kabul dizisinin tamamlandığı anlamına gelmez.
Sonraki canlı incelemede gas/fluid örtüşmesinde Vulkan RT'ye özgü görüntü farkı
doğrulandı; kaynak düzeltmesi ve build kapısı
[VULKAN_RT_GAS_FLUID_OVERLAP.md](VULKAN_RT_GAS_FLUID_OVERLAP.md) belgesinde.

Kullanıcı bildirimi: ilk obje ekleniyor, ikinci Add sırasında sürücü TDR ile
çöküyor. Kaynak incelemesinde `createTLAS` eski yapıyı yok edip yeniden
oluştururken probe descriptor güncellemesinin yalnız sayısal handle değişimine
bağlı olduğu görüldü. Sürücü aynı handle değerini yeniden kullanabilir; eski
descriptor yeni AS tahsisine kendiliğinden bağlanmaz. Bu, belirtiyle uyumlu
somut bir yaşam döngüsü hatasıdır; runtime kök neden doğrulaması henüz yok.

**Düzeltme YAZILDI; derleme ve runtime testi bekliyor:**

- Her senkron probe partisinden önce descriptor güncellenir; TLAS ve ortam
  kaynaklarında handle yeniden kullanımı artık güncellemeyi atlatamaz.
- BLAS yazımı → TLAS build okuması ve AS yazımı → compute ray-query okuması
  için açık bellek bariyerleri eklendi.
- Boş raster sahnede ve BLAS yenilenirken hazır durumu kapatılır; probe çağrısı
  hazır olmayan AS'yi reddeder. TLAS null handle sonucu başarı sayılmaz.

Sözleşme: [Vulkan kaynak descriptor'ları](https://docs.vulkan.org/spec/latest/chapters/descriptors.html)
ve [AS senkronizasyonu](https://docs.vulkan.org/spec/latest/chapters/accelstructures.html).
Ortak servis düzeltmesidir; mevcut UI/Python/IPC yüzeyleri aynı yolu kullanır.
Yeni komut veya geometri otoritesi eklenmedi.

**Kullanıcı build/test sırası (1b-β öncesinde):**

1. C++ projesini derle; bu düzeltmede shader kaynağı değişmedi.
2. Boş sahnede traced açıkken Add ile küp, ardından küre ekle; farklı mesh ve
   aynı mesh tekrarlarını toplam 20 eklemeye kadar dene.
3. Taşı, gizle/göster, sil/undo ve tüm objeleri sil → yeniden ekle geçişlerini
   dene. Aynı diziyi `rayfusion.set_probe_producer {traced:false}` ile karşılaştır.
4. `rayfusion.scene_as` ve `rayfusion.probe_field` durumlarını kaydet:
   instance sayısı görünür sahneyle uyumlu, producer traced, yayın sayısı artıyor
   olmalı; boş sahnede AS hazır olmamalı. Validation açıksa AS/descriptor ve
   senkronizasyon hatası olmamalı; TDR veya device loss kabul edilmez.
5. Bu kapı geçince 1b-α görünürlük sahneleri ve `trace_ms` ölçümü; ardından
   1b-β flat geometri/materyal tablosu ve tek sıçrama. TDR testi geçmeden
   1b-α doğrulandı sayılmaz.

Sınır: ortak `createTLAS` / `endSingleTimeCommands` halen `void` dönüyor;
null olmayan handle, GPU build/submit başarısını tek başına kanıtlamaz.
Bu hata aktarımının güçlendirilmesi ayrı ortak backend işi olarak açık.

## Kaynak sahipliği ve maliyet

1b-β ilk dilim uygulama, kontrol yüzeyleri, açık sınırlar ve kullanıcı build
listesi: [RAYFUSION_PROBE_BOUNCE.md](RAYFUSION_PROBE_BOUNCE.md).
Gas geçişi kullanıcının son tekrarında sorunsuz; geçici bildirim doğrulanmış
kalıcı hata sayılmadı. Physical Sky yön farkı kullanıcının kararıyla ertelendi.

Tek `RayFusionScene` kaynağı flat TriangleMesh/DNA verisinden mesh ve instance
kimliklerini üretir. Materyal, texture ve instance kaynakları raster ve RT
tarafında aynı GPU sahibi altında kullanılır. Paylaşılan mesh tek geometri
olarak kalır; instance dönüşümleri ayrı tutulur. nodeName kimlik değildir.

Geometri, transform, material/texture, ışık/world ve scene/device epoch ayrı
revision'lara sahiptir. Kamera hareketi yalnız görünürlük ve ekran geçmişini
etkiler. Statik geometri kare başına tekrar taranmaz/yüklenmez; deformasyon ve
topoloji değişiminin refit/rebuild maliyeti ayrıca raporlanır.

Frame slot'ları kaynakların ömrünü taşır. GPU'nun kullandığı descriptor,
buffer veya AS değiştirilmeden önce doğru bağımlılık/fence uygulanır; sırf
ölçüm almak için queue/device idle eklenmez. Proje değişiminde eski scene
epoch'una ait tamamlanan iş yeni sahneye yayınlanamaz. Bu tasarım, diğer
ajanın açılış/TDR düzeltmesini değiştirme yetkisi olarak yorumlanmaz.

Viewport Quality; ray bütçesi, probe güncelleme sayısı, alan yoğunluğu,
ekran çözünürlüğü ve temporal filtre maliyetini kontrol eder. Işık şiddeti
kalite preset'ine göre değiştirilmez. Full da sınırsız dispatch anlamına
gelmez. Mevcut Full geometri davranışıyla GI bütçesi birbirinden ayrılır.

Başlangıç GPU süresi hedefi ölçümle belirlenecek; sahne/çözünürlük/GPU
tanımlanmadan FPS veya milisaniye garantisi yok. Ray tracing, instance sayısı
ve AS güncelleme maliyetinden bağımsız değildir.

## Teslim sırası ve gözlenebilirlik

1. **Ortak sahne ve kaynak temeli:** same-device sahipliği, capability raporu,
   revision/epoch ve GPU pass zamanları. Raster + RT sahne kimlikleri aynı
   görünür geometriyi göstermeli; kamera hareketinde rebuild sayacı artmamalı.
   Aynı partide her iş sınıfı için **yol zorlama (override)** ve yol başına GPU
   ms ölçümü açılır; otomatik seçici bu ölçümden sonra yazılır.
2. **RayFusion ilk görüntüsü:** raster doğrudan ışık + görünürlük ağırlıklı
   RT probe diffuse GI + HDR birleşim. Eski diffuse sky boyaması devreden
   çıkar. İlk sürümün desteklemediği materyal/geometri açıkça raporlanır.
3. **Kararlılık ve maliyet:** probe güncelleme zamanlaması, ışık açma/kapama
   tepkisi, kamera kesmesi ve disocclusion; ardından ekran uzayı düzeltmesi.
4. **Hair:** canonical renk/melanin, longitudinal/azimuthal roughness ve
   transmission; tangent/radius/root UV/material ID; HDR, gölge ve GI tüketimi.
   Hair performansı, genel Realtime performansının nedeni sayılmaz.
5. **İleri RT efektleri:** gerekli yüzeylerde yansıma/kırılma kalitesini artırma.
   Bu, diffuse GI veya hair teslimini belirsiz süre erteleme gerekçesi değildir.
6. **Hacim yolu ve seçici:** gas/VDB/SDF için froxel/ekran uzayı marşı ile RT
   marşı aynı sahnede yan yana ölçülür. Kazanan sabitlenmez; ölçülen kırılma
   noktası histerezisli seçiciye bağlanır ve seçim raporlanır.

Her dilim UI + scripting + IPC aynı servise bağlı teslim edilir. Önerilen
gözlem alanları: requested/active mode, inactive reason, scene epoch, probe
total/valid/updated, rays dispatched, pending updates, build/refit sayıları,
geometri ve cache bytes, ayrı GPU pass ms. Ölçülmeyen zaman `available=false`
olmalı. Warm-up ve desteklenmeyen yol, başarılı GI diye raporlanamaz.
Metot adları henüz runtime'a eklenmedi; dispatch/capability/descriptor üretimi
ve kullanıcı ayarlarının proje kaydı uygulama partilerinin parçasıdır.

## Görsel kabul sahneleri

- Kapalı oda: dışarıdaki parlak sky içeriyi tek renge boyamamalı.
- Pencereli oda: ışık açıklıktan girmeli; pencere kapanınca dolaylı ışık sönmeli.
- Kırmızı duvar + nötr zemin: görünür çevrede sınırlı renk yansıması; aradaki
  engelin öte tarafına kaçak olmamalı.
- Kamera dışındaki ışıklı yüzey: kamera dönünce katkı aniden kaybolmamalı.
- Işık kapama, hareketli kapı, kamera kesmesi: eski ışık izi kalmamalı;
  yakınsama gecikmesi ölçülmeli, gizlenmemeli.
- Solid world, HDRI, Physical Sky ve HDRI overlay: kaynak değişiminde cache
  yenilenmeli; eski diffuse ambient veya overlay ikinci kez eklenmemeli.
- Shared mesh, transform, sculpt, skinning ve proje değişimi: eski AS/cache
  yayınlanmamalı; GUI/script/IPC aynı aktif durumu okumalı.
- Aynı gaz/VDB/SDF sahnesi iki yolla: yoğunluk, gölge ve HDR birleşim farkı
  eşik altında kalmalı; yol değiştiğinde parlaklık zıplamamalı, hacim opak
  geometriyle aynı depth'e karşı çözülmeli.
- Yoğun instance sahnesi (orman/scatter) iki yolla: kırılma noktası ölçülmeli,
  AS refit borcu maliyete dahil edilmeli; seçicinin kararı `reason` ile okunmalı.
- Hair aşamasında arkadan aydınlatılan saç ve koyu/açık melanin: yönlü parlama,
  geçirgenlik, kalınlık ve DoF birlikte doğrulanmalı.

GI için cam, alpha foliage, SDF ve VDB katılımı ayrı capability/kabul
maddeleridir. Opak triangle testinin geçmesi bu alanlarda görünürlüğün doğru
olduğunu kanıtlamaz. Path-traced referansla eş pozlama karşılaştırılır;
yalnız ortalama parlaklık değil bölgesel ışık dağılımı ve kaçaklar değerlendirilir.
