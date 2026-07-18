# RayTrophi Studio

<div align="center">

![Durum](https://img.shields.io/badge/durum-aktif%20geli%C5%9Ftirme-orange.svg)
![C++](https://img.shields.io/badge/C++-20-00599C.svg?logo=c%2B%2B)
![Platform](https://img.shields.io/badge/platform-Windows%20x64-0078D6.svg?logo=windows)
![Backend](https://img.shields.io/badge/render-CPU%20%7C%20OptiX%20%7C%20Vulkan%20RT-76B900.svg)
![Lisans](https://img.shields.io/badge/lisans-MIT-green.svg)

**Hibrit CPU/GPU path tracer çekirdeği üzerine kurulu, açık kaynaklı bir 3B içerik üretim (DCC) uygulaması.**

Modelle, sculpt yap, boya, tüy/saç grooming, simüle et, ışıklandır, animasyonla, render al — tek uygulamada.

[![RayTrophi Tanıtım](https://img.youtube.com/vi/-xRiPhc-p6k/maxresdefault.jpg)](https://www.youtube.com/watch?v=-xRiPhc-p6k)
**[▶️ YouTube'da tanıtımı izle](https://www.youtube.com/watch?v=-xRiPhc-p6k)**

[Nedir](#-nedir) • [Çalışma Alanları](#-çalışma-alanları) • [Render](#%EF%B8%8F-render--backendler) • [Simülasyon](#-fizik--simülasyon-paketi) • [Hızlı Başlangıç](#-hızlı-başlangıç) • [Mimari](#%EF%B8%8F-mimari) • [Galeri](#-galeri)

</div>

---

## 📖 Nedir

**RayTrophi Studio**, bir path-tracing render motoru olarak başladı ve tam teşekküllü bir **dijital içerik üretim (DCC) uygulamasına** dönüştü. Sahneyi sıfırdan kurabileceğiniz tek bir masaüstü programı: poligon modelleme ve sculpt, doku boyama, saç/tüy grooming, arazi ve bitki örtüsü, sıvı/gaz/whitewater simülasyonu, okyanus ve nehirler — ve tümü, üç değiştirilebilir backend üzerinde çalışan fiziksel temelli bir path tracer ile render edilir (CPU, NVIDIA OptiX ve Vulkan Ray Tracing).

Bu bir render-farm eklentisi ya da kütüphane değil. Modern dock'lu bir arayüzü, animasyon zaman çizelgesi, her araçta geri/ileri al, proje kaydet/yükle ve render sonucunun üzerine uygulanan tahribatsız bir sanat-yönetimi (Stylize) katmanı olan interaktif bir editördür.

### Tasarım hedefleri

- **Tek uygulama, tam pipeline.** Geometri oluşturma, look-dev, FX, animasyon ve final render aynı sahnede, aynı `.rtp`/`.rts` projesinde, aynı geri-al yığınında yaşar.
- **Üç render backend'i, tek özellik seti.** CPU (Embree), OptiX ve Vulkan RT arasında sahneyi değiştirmeden geçiş yap. Vulkan, önerilen interaktif backend; OptiX ve CPU birinci sınıf kalır.
- **Fiziksel temelli ama sanat-yönetilebilir.** Principled BSDF + spektral saç + volumetrik + DCC seviyesi sıvılar; üstüne, fiziği bozmadan görüntüyü yağlıboya/mürekkep/toon görünümlere boyayan bir Stylize katmanı.
- **Durumu konusunda dürüst.** Bu aktif, tek kişilik bir proje. Bir alt sistem deneyselse veya geliştirme aşamasındaysa, bunu açıkça belirtir.

> **Durum:** aktif geliştirme. Henüz sürüm etiketi yok; `main` dalı güncel yapıdır.

---

## 📊 Bir bakışta proje
<!-- STATS_START -->
| Metrik | Değer |
| :--- | :--- |
| **Proje kod / shader satırı** | ~259.000 |
| **Proje kod / shader dosyası** | 360+ |
| **GPU çekirdek & shader dosyası** | 56 (CUDA, OptiX PTX, Vulkan GLSL/RT, compute) |
| **UI kontrol noktası** | 1.278+ |
| **Render backendleri** | CPU (Embree) · NVIDIA OptiX · Vulkan RT |
| **Düğüm sistemleri** | Arazi (66), Animasyon (14+), Materyal (11+) |
| **Son doğrulama** | 2026-07-18 |
<!-- STATS_END -->

Sayımlar `raytrac_sdl2/source` kapsamındadır ve tek dosyalık dış kütüphaneleri (`simdjson`, `stb`, `json.hpp`, `tinyexr`) hariç tutar.

Tam teknik rapor: **[ARCHITECTURE_TR.md](ARCHITECTURE_TR.md)** · English: **[README.md](README.md)**

---

## 🧭 Çalışma alanları

RayTrophi Studio, hepsi aynı canlı sahne üzerinde çalışan, göreve odaklı çalışma alanlarına bölünmüştür:

| Çalışma alanı | Orada ne yaparsın |
|---------------|-------------------|
| **Layout / Sahne** | Asset içe aktar, obje/ışık/kamera yerleştir ve dönüştür, hiyerarşi kur, kutu-seçim, gizmo |
| **Modelleme** | Poligon düzenleme (extrude, inset, bevel, loop cut, weld, merge, UV unwrap), modifier yığını |
| **Sculpt** | Mesh ve arazide fırça tabanlı yüzey heykeltıraşlığı (PBVH hızlandırmalı) |
| **Boyama** | Mesh üzerine doğrudan katmanlı PBR doku boyama (çok kanallı, blend modları) |
| **Arazi** | Sculpt + tahribatsız arazi/biome grafiği, erozyon, hidroloji, kar/buzul, adlandırılmış alanlar, heightmap I/O |
| **Bitki / Scatter** | Biome duyarlı node katmanları, Asset Library + sahne kaynakları, kural tabanlı ve elle boyanan GPU instancing |
| **Saç** | Tüy/saç groom, tara, kes/uzat, simüle et ve render et |
| **Simülasyon** | Sıvı (APIC/FLIP), gaz/duman/ateş, whitewater, rigid + soft body & cloth (Jolt), mesh/primitif collider, kuvvet alanları, emitter |
| **Animasyon** | Çok-track zaman çizelgesi, kanal bazlı keyframe, iskelet animasyonu, animasyon grafiği |
| **Render / Look-dev** | Backend seç, örnekleme/kalite ayarla, denoise, tonemap ve sonucu Stylize'la |

---

## 🖥️ Render & backendler

Tek bir fiziksel temelli path tracer, üç hızlandırma backend'ini besler. Sahne, materyaller ve ışıklar üçünde de aynıdır — ana göre uygun olanı seçersin (başsız/GPU'suz için CPU, NVIDIA eğri donanımı için OptiX, hızlı interaktif look-dev için Vulkan RT).

### Materyaller & gölgeleme
- **Principled BSDF** (Disney tarzı uber-shader): albedo, roughness, metallic, specular, clearcoat, sheen, anisotropy, transmission/IOR
- **Lambertian, Metal, Dielectric** klasik modeller
- **Yüzey-altı saçılım (SSS)**
- **Interior Volume** (resin / cam-bilye içi): opak taban üstünde veya şeffaf camın içinde Beer-Lambert derinlik emilimi (üç backend'de de), artı prosedürel iç sistem (Vulkan RT): deterministik DDA ile gezilen kapanımlar — keskin kir benekleri, hava kabarcıkları ve renkli cam kırıkları (yuvarlak çip veya obje döndükçe parlayan uzamış faset kristaller) — ve stilli toz bulutları (nebula, iki renkli billow, lifli çizgiler, domain-warp'lı "suda mürekkep" boya girdabı); obje/dünya çapalama, küratörlü iç-hacim preset'leri, ekstra sahne ışını yok. Kırık renkleri foton caustic'e taşınır — vitray desenli caustic. Teknik not: [docs/INTERIOR_VOLUME.md](docs/INTERIOR_VOLUME.md)
- **Spektral / melanin tabanlı saç BSDF**
- NanoVDB seyrek hacimler ve prosedürel gürültü yoğunluğuyla **volumetrik render**
- Tam doku desteği (albedo, roughness, metallic, normal, emission, transmission, opacity), sRGB/linear yönetimi

### Işıklandırma & gökyüzü
- Nokta, yönlü, spot ve **mesh tabanlı alan ışıkları**; emissive materyaller
- **HDR/EXR ortam haritaları** (equirectangular)
- Gündüz/gece döngülü **Nishita fiziksel gökyüzü**: prosedürel yıldız ve ay (evreler, ufuk büyütme, atmosferik sönümleme), güneş halesi, otomatik güneş↔yönlü-ışık senkronu
- **Küresel volumetrik bulutlar** (Henyey-Greenstein saçılım, adaptif ray marching, kapsama/yoğunluk/yükseklik/rüzgâr kontrolleri, yumuşak ufuk geçişi) — HDRI, düz renk veya Nishita gökyüzü üzerinde çalışır
- Çoklu önem örneklemeli (MIS) yumuşak gölgeler

### Foton caustic & volumetrik ışık huzmeleri
- **Aşamalı (progressive) photon-map caustic** (Vulkan RT): ışık tarafındaki foton geçişi, kamera yolunun kullandığı RT pipeline'ını ve BSDF'leri aynen paylaşır; kırılan (LS⁺D) enerjiyi dünyaya sabit bir hash grid'e biriktirir — cam odak lekeleri, halka caustic'leri ve renkli cam desenleri gibi düz path tracing'in pratikte asla yakınsayamadığı efektler
- **Spektral dispersiyon caustic'e bedava taşınır**: kamera ışınlarının kullandığı stokastik hero-dalga boyu taşıması fotonlarla birlikte yol alır — hem yüzey deseninde hem huzmelerde gökkuşağı saçakları
- **Volume objesi gerektirmeyen hacimsel ışık huzmeleri**: fotonlar uçuş segmentleri boyunca ikinci, daha kaba bir dünya grid'ine de enerji bırakır; sınırlandırılmış bir kamera march'ı bunu görünür in-scatter'a çevirir. Cam bir obje ve bir ışık yeterli — sis veya katılımcı ortam kurulumu gerekmez. Saçılım şiddeti (sanal toz yoğunluğu), isteğe bağlı 3D türbülans modülasyonu ve ışık→cam bacağının da parladığı direct-shaft modu (point ışıklar karışım örneklemeli emisyona geçer, her yöne huzme verir)
- Kameradan bağımsız dünya-uzayı hedefleme (fotonlar geçirgen objelerin canlı union sınırlarına nişan alır, her frame yeniden değerlendirilir), ölçek-duyarlı grid boyutlandırma, yumuşak koni-kernel ve trilinear yoğunluk okumaları
- Dürüst sınırlar: şimdilik yalnız Vulkan RT (OptiX/CPU portları planda), huzmelerde izotropik faz, keskinlik grid çözünürlüğüyle sınırlı

### Örnekleme & post
- Aşamalı **birikimsel path tracing** + **adaptif örnekleme** (örnekleri gürültülü bölgelere yoğunlaştırır)
- Alan derinliği, hareket bulanıklığı
- **Intel Open Image Denoise (OIDN)** — CPU ve CUDA hızlandırmalı yollar, viewport ve final
- Tone mapping ve son-işleme

### Backend karşılaştırması

<details>
<summary>⚡ <b>Özellik eşitliği: OptiX vs Vulkan RT</b> (genişlet)</summary>

| Özellik | OptiX | Vulkan RT | Not |
|---------|:-----:|:---------:|-----|
| Principled BSDF | ✅ | ✅ | Tam eşitlik |
| Lambertian / Metal / Dielectric | ✅ | ✅ | Tam eşitlik |
| Yüzey-altı saçılım (SSS) | ✅ | ✅ | Küçük renk tonu farkı |
| Clearcoat & Anisotropic | ✅ | ✅ | Tam eşitlik |
| Volumetrik render (NanoVDB) | ✅ | ✅ | Kalıcı leaf-cache accessor; interaktifte OptiX'e eşit veya daha hızlı |
| **Saç sistemi** | ✅ | ✅ | Analitik LSS kesişim + LSS-sıkı AABB; burada OptiX donanım eğrilerini geçer |
| HDR / EXR ortam | ✅ | ✅ | Tam eşitlik |
| Nishita gökyüzü & gündüz/gece | ✅ | ✅ | Tam eşitlik |
| Volumetrik bulutlar | ✅ | 🧪 | Küçük saçılım farkları |
| Su / Okyanus (FFT) | ✅ | 🧪 | Dalga yansıma farkları |
| İskelet animasyonu (GPU skinning) | ✅ | ✅ | Vulkan compute shader |
| Alan derinliği / hareket bulanıklığı | ✅ | ✅ | Tam eşitlik |
| Yumuşak gölge (MIS) / alan ışıkları | ✅ | ✅ | Tam eşitlik |
| Tone mapping & post-FX | ✅ | ✅ | Vulkan'da GPU compute tonemap, trace komut tamponuna kaynaşık |
| OIDN denoise | ✅ | ✅ | OptiX'in CUDA-interop yolu daha sıkı |
| Adaptif / aşamalı render | ✅ | ✅ | Vulkan daha hızlı yakınsar (kare başı daha az ek yük) |
| Stylize katmanı | ✅ | ✅ | CPU / Vulkan / OptiX eşleşen çıktı üretir |
| Foton caustic + volumetrik ışık huzmeleri | ❌ | ✅ | Şimdilik yalnız Vulkan; foton geçişi kamera RT pipeline'ını paylaşır |

> **Açıklama:** ✅ tam destek &nbsp;|&nbsp; 🧪 destekli, küçük çıktı farkları olabilir

</details>

<details>
<summary>📈 <b>İnteraktif ölçümler</b> (genişlet)</summary>

Aynı sahne, aynı ayarlar, aynı donanım, kamera hareket halinde. Bunlar interaktif viewport kare hızlarıdır, final-kare değil. Statik sahnelerde adaptif örnekleme, pikseller yakınsadıkça her iki backend'i de 500 fps üzerine taşır.

| Sahne | Vulkan RT | OptiX | Oran |
|---|:---:|:---:|:---:|
| Mesh-yoğun + Nishita atmosfer | 600 fps | 50 fps | 12.0× |
| Saç-yoğun (kübik B-spline, LSS kesişim) | 300 fps | 70 fps | 4.3× |
| Hacim / VDB bulut (Fast preset) | 300 fps | 200 fps | 1.5× |
| Hacim / VDB bulut (Balanced preset) | benzer | benzer | ≈1.0× |
| Hacim / VDB bulut (Exact, kamera hareketli) | 16 fps | 23 fps | 0.7× |

**Vulkan interaktifte neden önde:** asenkron fence tabanlı ping-pong kare pipeline'ı (kare başı `vkQueueWaitIdle` yok), küçük RGBA8 staging'e GPU compute tonemap, analitik Linear-Swept-Sphere saç kesişimi, march adımları boyunca kalıcı NanoVDB read-accessor ve sıcak yolda piksel başı birikim atomic'i olmayan yalın çekirdek.

**OptiX'in hâlâ kazandığı yer:** kamera hareketindeyken Exact hacim preset'i (donanımsal CUDA NanoVDB doku yolu), CUDA-yerel sıfır-kopya OIDN interop ve özellikle NVIDIA eğri donanım primitifleri gereken final kareler.

</details>

---

## 🌀 Fizik & simülasyon paketi

CUDA ve CPU backend'li, çok iş parçacıklı grid ve parçacık tabanlı bir FX paketi; doğrudan path-traced render pipeline'ına entegre. Birden çok simülasyon domaini, emitter, collider, rigid body ve kuvvet alanı tek çalışma alanında bir arada bulunur ve projeyle birlikte kaydedilir.

### Sıvı — APIC / FLIP çözücü
- Açısal momentumu koruyan, sayısal dağılımı en aza indiren ayarlanabilir karışımlı **APIC/FLIP** çözücü
- PCG + MIC(0) ön koşullu basınç çözümü (CPU) ve GPU'da Jacobi-PCG / çok-ızgaralı (MGPCG) basınç çözümü ile **MAC kademeli ızgara**
- **Varyasyonel (cut-cell) katı eşleşmesi** (Batty/Bridson): kesirli MAC-yüzey ağırlıkları analitik primitiflere karşı alt-ızgara doğruluğunda çarpışma sağlar; hareketli collider'lar basınç çözümü üzerinden gerçek momentum/sıçrama aktarır
- **Ghost-fluid 2. derece serbest yüzey** (Gibou/Enright): alt-hücre level set, sıvı yüzeyindeki voksel "merdivenleşmesini" giderir
- Keyframe ile animasyonlu collider'lar her alt-adımda yeniden konumlanır, böylece sıvı hareketli geometriyi takip eder
- Adaptif çözünürlük, açık/kapalı sınır modları, sızıntıyı önleyen dinamik parçacık yeniden tohumlama, sıvı materyal preset'leri (Su, Yağ, Özel)

### Gaz, duman & ateş
- Sıcaklık, is ve yakıt yoğunluğu için çok iş parçacıklı yoğun-ızgara çözücü
- Yanma dinamikleri (tutuşma, ısı salımı, alev sönümlenmesi) + prosedürel FBM curl-noise türbülans
- Verimli büyük domainler için seyrek-VDB aktif-voksel Poisson çözümü

### Whitewater (Ihmsen ve ark. 2012)
Hapsolmuş hava ve dalga tepesi potansiyellerinden üretilen ikincil **sprey** (havada), **köpük** (yüzeyde) ve **kabarcık** (su altı); collider tepkisiyle çözücü içinde taşınır:
- **Dinamik PBR materyal yönlendirme** — sprey için geçirgen damlacıklar, köpük için saçılan mat-beyaz PBR, gümüşi yarı-geçirgen kabarcıklar — herhangi bir sahne materyalini bağlayan *Özel Materyal Geçersiz Kılma* ile
- Su-altı kabarcıklarda toplam-iç-yansıma (TIR) kaynaklı koyu-halka kusurlarını azaltan **TIR düzeltmesi**
- Yüzey köpüğünü yumuşatılmış level-set su mesh'ine projelendirip dalgalı suda yüzen köpüğü gideren **Newton-Raphson dalga oturtma**
- Deterministik hash tabanlı boyut varyasyonu ve ömür sonunda yumuşak çözünme
- Yakın çekim detayı için ayarlanabilir icosphere alt bölümü (0–3)

### Gövdeler — Jolt Physics: rigid, soft & cloth + iki-yönlü sıvı eşleşmesi
- Paylaşılan simülasyon zaman çizelgesinde **Jolt Physics** tabanlı gövde çözücü: herhangi bir sahne objesini **Static, Dynamic veya Kinematic** olarak işaretle; sınırlarına oturtulan **box / sphere / capsule / yönlü-kutu** primitifleri ya da objenin gerçek geometrisini kullanan bir **mesh collider** — static gövdeler için tam üçgen mesh, hareketli gövdeler için convex hull; böylece SDF/mesh kaynaklı bir collider OBB yerine gerçek şekille çarpışır
- **Soft body & cloth.** Bir mesh'i deforme olabilen **soft body** veya **cloth** olarak işaretle (Jolt soft-body çözücü): gövde başına sertlik/compliance, basınç (kapalı-hacim şişirme), sönümleme, iterasyon, vertex çarpışma kalınlığı ve **vertex pinleme** (rest vertex'leri sabit tutarak kumaşı köşe/kenardan asma); deforme mesh doğrudan render'a geri yazılır
- Gövde başına **kütle veya yoğunluktan-otomatik-kütle, doğrusal & açısal sönümleme, sürtünme, geri tepme (restitution), yerçekimi ölçeği, başlangıç doğrusal/açısal hızı, uyku ve eksen bazlı öteleme/dönüş kilitleri**
- **Kuvvet alanları her gövde türünü sürer** — rigid (COM'da kuvvet), soft & cloth (vertex bazlı hız itmesi, pinli vertex'ler hariç)
- **İki-yönlü sıvı eşleşmesi.** Gövde, varyasyonel cut-cell yolu üzerinden sıvı/gaz ızgarasına hareketli bir katı olarak voksellenir, böylece sıvıyı iter ve sıçratır; karşılığında sıvı level set'inden örneklenen **kaldırma kuvveti (buoyancy) ve doğrusal/açısal sürükleme** gövdeye geri etki eder — render'ın okuduğu aynı alandan yüzme, batma ve sallanma davranışı
- **Kinematic** gövdeler keyframe ile sürülür (sıvıyı karıştıran animasyonlu collider'lar); **Dynamic** gövdelerin sahibi çözücüdür, böylece zaman çizelgesi simüle edilen pozla çakışmaz
- **Seçici yeniden-pişirme.** Bir gövdeyi düzenlemek veya taşımak, (pahalı) sıvı bake'ini yalnızca o gövde gerçekten bir sıvı domainiyle etkileşiyorsa düşürür — alakasız bir static prop kendi başına ucuzca yeniden simüle edilir, sıvı önbelleği korunur

### Yüzeyler, önbellek & serileştirme
- Render zamanı sıvı mesh'i için Laplacian yumuşatmalı **Yu-Turk anizotropik yüzey rekonstrüksiyonu**; yüzey çözünürlüğü sim ızgarasından bağımsız
- **SimCache disk pişirme** — ağır sıvı/köpük/gaz karelerini proje yanına ikili `.simcache` dosyalarına pişir, yeniden simüle etmeden zaman çizelgesini gerçek zamanlı tara
- Simülasyon durumu, domain ayarları, özel materyaller, timeline önbellekleri ve preset'lerin `.rtp` / `.rts` içine tam serileştirilmesi

> GPU MGPCG basınç yolu canlı; varyasyonel katılar + ghost-fluid'in GPU portu (Faz 2) geliştirme aşamasında — tam DCC eşitliği için yüzey gerilimi, örtük viskozite ve dar-bant/seyrek performans çalışmaları da öyle.

---

## 🛠️ Prosedürel & oluşturma araçları

### 🏔️ Arazi, biome & hidroloji grafiği

- Gerçek zamanlı sculpt fırçaları (yükselt, alçalt, yumuşat, düzleştir, stamp) ve World Machine / Gaea iş akışları için 16-bit heightmap içe/dışa aktarımı
- **Terrain Nodes V2** — 66 kayıtlı node, canlı property düzenleme, önizleme, gruplama, tekrar kullanılabilir kurulumlar ve Apply ile sahneye çıktı sunan, serileştirilen tahribatsız grafik
- **Üretim & şekillendirme** — heightmap/hardness girdileri; prosedürel noise, fault, mesa, shear, terrace, smooth, normalize, resample, remap, blend, math, clamp, overlay ve screen işlemleri
- **Erozyon & jeoloji** — hidrolik, termal, fluvial ve rüzgâr erozyonu; sediman birikimi, alüvyon yelpazesi, delta, ıslaklık, toprak derinliği, litoloji ve katman sentezi
- **Arazi analizi & adlandırılmış alanlar** — eğim, yükseklik, eğrilik, akış, exposure, watershed, concavity, convexity, valley ve wetness verileri bir kez üretilip yüzey, biome ve foliage node'larında yeniden kullanılabilir
- **Biome Composer** — Temperate Mixed, Lush Valleys, Alpine Tundra, Arid Highlands ve Boreal Mountains preset'leri normalize Forest / Grass / Rock / Alpine maskeleri ile paketlenmiş biome splat haritası üretir
- **Hidroloji** — watershed analizi, nehir ağları ve hidroliği, nehir yatağı oyma, spline çıktısı, göl havzası algılama, göl yüzeyleri, kanal, kıyı ve köpük maskeleri
- **İklim & kar** — climate, snowfall, settling, melt/freeze, bağıl kar çizgisi, buzul akışı ve kar/yüzey birleştirme node'ları
- **Hızlı kurulum işlemleri** — düzenli biome-field ve biome-foliage grafik kollarını otomatik kurar; oluşturulan tüm node'lar düzenlenebilir kalır

### 🌿 Node tabanlı biome foliage & scatter

- **Foliage Layer → Foliage Set / Biome → Foliage Output** her bitki sınıfını bağımsız bir kural olarak korur, set düzeyinde tahribatsız density/seed kontrolü ekler ve arazi grafiği Apply edildiğinde bağlı katmanların tamamını dağıtabilir
- Yerleştirme kuralları hedef adet, seed, minimum aralık, eğim ve yükseklik aralıklarının yanında; ayarlanabilir eşik/etkiye sahip adlandırılmış density, exclusion ve scale alanlarını kapsar
- Her katman **Recommended**, **Asset Library** veya **Scene Objects** kaynaklarından ağırlıklı birden fazla obje alabilir; arama, biome/tip filtresi, tekrar önleme, thumbnail, hover önizleme ve kompakt kaynak düzenleme hem Terrain UI hem node properties içinde ortaktır
- Asset Library önce model metadata'sını tarar, geometriyi ihtiyaç anında ortak cache'e yükler. `RayTrophiStudio/assets` altına eklenen kullanıcı bitki ve kaya asset'leri aynı kataloğa ve öneri akışına otomatik katılır
- Kütüphane asset'leri taşınabilir göreli referanslar kullanır, eksik dosyaları bildirir, mesh tabanından araziye oturur ve kaynak bounding box'ından hesaplanan varyasyonla gerçek-dünya hedef yüksekliğine ölçeklenebilir
- Bitkiler varsayılan olarak **World Y-Up** kullanır; çim, kaya veya özellikle araziye hizalanması istenen asset'lerde kaynak başına **Follow Slope** ve normal influence kullanılabilir
- Terrain UI ile foliage node'ları aynı `InstanceGroup` verisinin senkron görünümleridir: ekleme, silme, ağırlık, scale/yükseklik, yönelim ve katman-kuralı değişiklikleri iki yöne yayılır; silinen kaynaklar eski grafik verisinden geri gelmez
- Mevcut scatter iş akışları korunur: GPU-instanced çim/ağaç/kaya, çakışma duyarlı prosedürel yerleşim, elle detay boyama ve küresel dinamik rüzgâr (şiddet, yön, gust)

### 💇 Saç & tüy
- GPU simüle ve render; Vulkan'da analitik LSS kesişim
- Grooming fırçaları: tara, kes/uzat, yumuşat
- Fizik: teller mesh'lerle çarpışır, yerçekimi/kuvvetlere tepki verir
- Melanin tabanlı saç BSDF

### 🌊 Okyanus & 🏞️ nehirler
- Köpük üretimi, kostik ve derinliğe bağlı su-altı volumetrikleriyle **FFT okyanus**
- Araziye otomatik oyulan, akış haritalı ve akışa göre obje sürüklemeli **spline/bezier nehirler**

### 🗿 Modelleme, sculpt & boyama
- **Edit Mesh modu** — extrude, inset, bevel, loop cut, sil/birleştir/weld/split, normal çevir, akıllı yeniden-üçgenleme, UV otomatik-unwrap/akıllı paketleyici
- **Sculpt modu** — Grab, Draw, Inflate, Layer, Clay, Clay Strips, Pinch, Smooth, Flatten, Scrape, Crease; Shift→Smooth, Ctrl→ters; X/Y/Z ayna; yoğun mesh'ler için PBVH budama; modifier-yığını alt bölümü; ortak mesh/arazi sculpt yolu
- **Mesh Paint** — katmanlı PBR boyama (Base Color, Normal, Roughness, Metallic, Emission, Mask, Transmission, Opacity); boya/sil/yumuşat/stamp/fill/clone/spray fırçaları; Normal/Add/Multiply/Screen/Overlay blend modlu katman yığını; height→normal pişirme; dirty-region GPU güncellemeleri; projeye PNG blob olarak serileşir
- Tüm düzenleme modlarında, isteğe bağlı adım gruplamalı tam **geri/ileri al**; mesh düzenlemeleri CPU/GPU tamponlarına yayılır ve modifier uygulanmış GLB olarak dışa aktarılır

### 🎨 Stylize — tahribatsız sanat yönetimi
Yakınsama sonrası, path-traced sonucu + AOV tamponlarını okuyup; sahne geometrisini, materyalleri veya ışıkları değiştirmeden görüntüyü yeniden stilize eden bir katman. Domain-maskeli kompozit, gök/materyal/dış-çizgi/dünya efektlerini ayrı tutar.
- **Gök katmanı** — görüş-ışınına kilitli stilize gradyanlar, bulut kümeleri ve güneş (Painterly Clouds, Cartoon Cel, Sunset Bands, Ink Wash, Clear Gradient)
- **Painterly materyal katmanı** — yüzeye kilitli fırça alanları (ekran-uzayı kayması yok), palet etkisi, kenar saygısı, pigment kalınlığı ve Wet Oil modeli (Body/Load/Pickup/Deposit/Buildup)
- **Dış-çizgi katmanı** — derinlik/normal/materyal süreksizliği kenarları; Ink, Oil, Pencil, Dry Brush, Pressure çizgi türleri
- **Profiller** — Painterly Oil, Gouache, Ink + Wash, Graphic Toon, Clay/Maquette, Dreamy Sunset
- **Backend eşitliği** — CPU, Vulkan compute (`stylize.comp`) ve OptiX CUDA (`StylizeKernel.cu`) eşleşen çıktı üretir; birikimi sıfırlamadan yeniden uygular

### 🖥️ Viewport gölgeleme
- Hızlı sculpt/paint geri bildirimi için Vulkan raster **Solid + Matcap** modu (matcap'leri `raytrac_sdl2/assets/matcaps/` içine bırak)
- Herhangi bir backend'de ışın izlemeli interaktif önizleme; gizmo sırasında idle-preview

---

## 🎞️ Animasyon & UI

- Grup hiyerarşili (Obje / Işık / Kamera / Dünya) **çok-track zaman çizelgesi & Graph Editor**, bağımsız Bezier eğrisi kanal animasyonu (Konum, Dönüş, Ölçek, Işık ayarları, Kamera parametreleri ve PBR Materyal özellikleri), yeniden boyutlandırılabilir splitter panel, toplu görünürlük seçimli katlanabilir grup başlıkları ve kısayol destekli keyframe/tangent düzenleme (sürükleme, silme/sığdırma kısayolları).
- Quaternion interpolasyonlu **iskelet animasyonu** + GPU compute skinning; durum makineleri ve IK blend uzayları için **animasyon grafiği** (14+ düğüm)
- **Toplu / sekans render** — animasyonu görüntü dizisine (materyal keyframe'leriyle) aktar, render ortasında iptal edilebilir, kare başına simülasyon sürümlü
- Modern **ImGui** dock'lu koyu UI, render kalite preset'leri (Low/Medium/High/Ultra), dinamik çözünürlük ölçekleme, sahne hiyerarşisi, materyal editörü, performans metrikleri
- **Seçim** — kutu seçim (sağ-sürükle), karışık ışık+obje seçimi, Ctrl+tık ekle/çıkar, hepsini/hiçbirini seç, çoklu-obje dönüşüm
- Dönüşüm, silme, çoğaltma, ışıklar için **geri/ileri al** — Ctrl+Z / Ctrl+Y

### 📦 Asset tarayıcı & kütüphane
- `model`, `anim_clip`, `vdb` ve `vdb_sequence` için metadata tabanlı keşif
- Yerleşik proje `assets` kökü + kullanıcı eklemeli yerel kütüphaneler
- Önizleme/thumbnail önbellekli asset kartları, favoriler, etiketler, kayıtlı koleksiyonlar ve akıllı klasörler
- Viewport hayalet önizlemeli ve otomatik seçimli sürükle-bırak yerleştirme
- Düzen, kütüphane listesi ve filtreler için proje-kapsamlı UI kalıcılığı

---

## 🚦 Hızlı başlangıç

### Önkoşullar

**Gerekli**
- **Visual Studio 2022** (MSVC v143) — önerilen yapı sistemi
- Windows 10/11 (x64)
- CMake 3.20+ (opsiyonel; VS2022 tercih edilir)

**Opsiyonel (GPU render)**
- NVIDIA GPU (SM 5.0+): GTX 9xx/10xx/16xx veya RTX serisi
- CUDA Toolkit 12.0+, OptiX 7.x/8.x SDK
- Vulkan SDK 1.3+ (Vulkan RT yolu için)

| GPU Serisi | Mimari | Mod | Performans |
|------------|--------|-----|------------|
| RTX 40xx | Ada Lovelace | Donanım RT | ⚡ En hızlı |
| RTX 30xx | Ampere | Donanım RT | ⚡ Çok hızlı |
| RTX 20xx | Turing | Donanım RT | ⚡ Hızlı |
| GTX 16xx | Turing | Compute | 🔶 İyi |
| GTX 10xx | Pascal | Compute | 🔶 Orta |
| GTX 9xx | Maxwell | Compute | 🔶 Daha yavaş |

### Ortam değişkenleri

Proje bağımlılıkları sistem ortam değişkenleriyle çözer. Derlemeden önce bunları yerel kurulum yollarına ayarla:

| Değişken | Açıklama | Örnek |
|----------|----------|-------|
| `SDL2_ROOT` | SDL2 kökü | `E:\...\SDL2-2.30.4` |
| `OPTIX_ROOT` | OptiX SDK | `C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0` |
| `EMBREE_ROOT` | Embree kökü | `E:\...\embree-4.4.0.x64.windows` |
| `OIDN_ROOT` | Intel OIDN kökü | `E:\...\oidn-2.3.0.x64.windows` |
| `ASSIMP_ROOT` | Assimp kökü | `E:\...\Assimp` |
| `CUDA_PATH` | CUDA Toolkit | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x` (genelde otomatik) |
| `VULKAN_SDK` | Vulkan SDK | `C:\VulkanSDK\1.3.xxx.0` |

Yönetilen bağımlılıklar: SDL2, Embree 4.x, Assimp 5.x, ImGui, OpenMP, stb_image, TinyEXR, Intel OIDN, NanoVDB ve CUDA/OptiX (opsiyonel).

### Derleme

**Visual Studio 2022 (önerilen)**
```bash
git clone https://github.com/maxkemal/RayTrophi.git
cd RayTrophi/raytrac_sdl2
# raytrac_sdl2.vcxproj dosyasını Visual Studio 2022'de aç
# Release | x64 seç, sonra Build > Build Solution (Ctrl+Shift+B)
# Çıktı: x64/Release/raytracing_render_code.exe
```
Tüm bağımlılıklar (DLL, PTX, shader, kaynaklar) çıktı dizinine otomatik kopyalanır.

**CMake**
```bash
cmake -S raytrac_sdl2 -B raytrac_sdl2/build -G "Visual Studio 17 2022" -A x64
cmake --build raytrac_sdl2/build --config Release -j 12
# Çıktı: raytrac_sdl2/build/bin/RELEASE/"RayTrophi Studio.exe"
```
CMake; çalıştırılabilir, PTX, Vulkan shader ve runtime DLL'lerini `build/bin/<CONFIG>` altında izole tutar, VS2022 `x64` çıktısının üzerine asla yazmaz.

### Çalıştırma
Çalıştırılabiliri başlat; dock'lu UI açılır. **File → Load Scene** ile model içe aktar (GLTF önerilir; Assimp ile 40+ format).

---

## 🏗️ Mimari

```
RayTrophi/
└── raytrac_sdl2/
    └── source/
        ├── src/
        │   ├── Core/        # Giriş noktası (Main.cpp), proje yönetimi
        │   ├── Render/      # Renderer, OptiX wrapper, Embree/Parallel BVH, kamera, doku
        │   ├── Backend/      # Vulkan RT, OptiX, viewport backendleri, sahne doku yöneticisi
        │   ├── Scene/        # Objeler, ışıklar, materyaller, instancing, mesh, BSDF'ler
        │   ├── Physics/      # Sıvı (APIC/FLIP), gaz, whitewater, arazi, okyanus, nehir, sim dünyası
        │   ├── Device/       # CUDA çekirdekleri (.cu/.cuh), OptiX cihaz kodu, Vulkan compute
        │   ├── Hair/         # Saç sistemi, teller, skinning, saç BSDF
        │   ├── Paint/        # Mesh & arazi boyama adaptörleri, katman yığını
        │   ├── Stylize/      # Stylize CPU/CUDA çekirdekleri ve durumu
        │   ├── Animation/    # Animasyon denetleyici, düğümler, Ozz runtime
        │   ├── Viewport/     # Viewport sahne senkronu
        │   ├── Math/         # Vec/Matrix/Quaternion
        │   ├── UI/           # ImGui panelleri, zaman çizelgesi, gizmolar, editörler
        │   └── Utils/        # Yükleyiciler, serileştirme, yardımcılar
        └── include/          # Header'lar (Backend, Core, Fluid, Hair, NodeSystem, Paint, Stylize, Viewport, Utils)
```

**Render backendleri**
- **EmbreeBVH** (`Render/EmbreeBVH.cpp`) — Intel CPU çekirdekleri
- **ParallelBVHNode** (`Render/ParallelBVHNode.cpp`) — özel SAH BVH, OpenMP-paralel yapı
- **OptixWrapper** (`Render/OptixWrapper.cpp`, `Device/*.cu`) — CUDA/OptiX, SBT + doku-objesi önbelleği
- **VulkanBackend** (`Backend/VulkanBackend.cpp`) — `VK_KHR_ray_tracing_pipeline`, TLAS/BLAS refit, compute skinning, asenkron ping-pong kare pipeline'ı, GPU tonemap

**Düğüm sistemleri** (`include/NodeSystem/`) — Arazi, Animasyon ve Materyal editörlerinin paylaştığı grafik çekirdeği.

---

## 🎨 Galeri

[![RayTrophi Tanıtım](https://img.youtube.com/vi/-xRiPhc-p6k/maxresdefault.jpg)](https://www.youtube.com/watch?v=-xRiPhc-p6k)
**[▶️ Tam demo reel'i izle](https://www.youtube.com/watch?v=-xRiPhc-p6k)**

<div align="center">

<img src="render_samples/1.png" width="800" alt="Karmaşık mimari sahne"><br>
<i>Karmaşık mimari sahne — 3,3M üçgen, Embree BVH</i>

<img src="render_samples/indoor2.png" width="800" alt="İç mekan tasarımı"><br>
<i>Volumetrik ışıklandırma ve yüzey-altı saçılımlı iç mekan</i>

<img src="render_samples/output1.png" width="800" alt="OptiX GPU render"><br>
<i>OptiX ile GPU path tracing</i>

<img src="render_samples/stylesed_winter_dragon1.png" width="800" alt="Stilize ejderha"><br>
<i>Tahribatsız Stylize katmanıyla stilize render</i>

<img src="render_samples/RayTrophi_cpu1.png" width="800" alt="CPU render"><br>
<i>Aşamalı iyileştirmeli saf CPU path tracing</i>

<img src="render_samples/yelken.png" width="800" alt="Açık hava sahnesi"><br>
<i>Nishita fiziksel gökyüzüyle açık hava ortamı</i>

</div>

---

## 🗺️ Yol haritası

**Yakın zamanda eklenenler**
- ✅ Vulkan RT backend (interaktif birincil) — GPU skinning, asenkron ping-pong pipeline, analitik LSS saç
- ✅ Fizik & parçacık simülasyon paketi (APIC/FLIP sıvı, gaz/ateş, whitewater)
- ✅ Rigid, soft-body & cloth dinamiği (Jolt Physics) — primitif/mesh collider, vertex pinleme, kuvvet-alanı eşleşmesi, iki-yönlü sıvı eşleşmesi (katı voksellemesi + kaldırma/sürükleme), seçici sıvı yeniden-pişirme
- ✅ GPU MGPCG sıvı basınç çözümü (CUDA)
- ✅ Varyasyonel cut-cell katı eşleşmesi + ghost-fluid 2. derece serbest yüzey (CPU)
- ✅ Çok-materyalli whitewater PBR yönlendirme + Newton-Raphson dalga oturtma
- ✅ SimCache disk kare pişirme + tam simülasyon serileştirme
- ✅ CPU / Vulkan / OptiX eşitlikli Stylize katmanı
- ✅ Sculpt modu (mesh + arazi) ve katmanlı mesh boyama
- ✅ Aşamalı photon-map caustic + spektral dispersiyon + volumetrik ışık huzmeleri (Vulkan RT)

**Planlanan / devam eden**
- [ ] OptiX / CPU'da caustic; huzmeler için anizotropik faz ve gerçek yoğunluk alanları (VDB)
- [ ] Varyasyonel katılar + ghost-fluid yüzeyin GPU portu (Faz 2)
- [ ] Sıvı yüzey gerilimi, örtük viskozite, dar-bant/seyrek performans
- [ ] Binned SAH / index tabanlı BVH / SBVH uzamsal bölme
- [ ] USD format desteği
- [ ] Ağ / dağıtık render
- [ ] Işık-yolu görselleştirme & hata ayıklama
- [ ] Linux / macOS desteği (şu an yalnızca Windows: SDL2 + Windows bağımlılıkları)

---

## 🐛 Bilinen sınırlamalar

- Bugün **yalnızca Windows** (SDL2 + Windows bağımlılıkları); Linux/macOS port gerektirir.
- **OptiX**, NVIDIA GPU gerektirir (SM 5.0+); RTX donanım RT çekirdeği, GTX compute kullanır (daha yavaş).
- Çok büyük sahneler (>10M üçgen) belleği zorlayabilir.
- CMake ve VS2022 **ayrı çıktı klasörleri** kullanır — eski PTX/DLL karışmaması için ayrı tut.
- Vulkan volumetrik bulutlar ve FFT okyanus, OptiX'e kıyasla küçük çıktı farkları gösterir (eşitlik tablosuna bak).
- GPU sıvı basıncı canlı, ancak varyasyonel katılar + ghost-fluid yüzey şimdilik yalnızca CPU.

---

## 🤝 Katkı

Katkılar memnuniyetle karşılanır — performans çalışmaları, yeni materyal/FX modelleri, format desteği, hata düzeltmeleri ve dokümantasyon.

1. Repoyu fork'la
2. Özellik dalı oluştur (`git checkout -b feature/ozelligin`)
3. Değişikliklerini commit'le
4. Push'la ve bir Pull Request aç

---

## 📝 Lisans

MIT Lisansı — bkz. [LICENSE.txt](LICENSE.txt).

Üçüncü taraf kütüphane ve SDK'lar kendi lisansları altında kalır. Jolt Physics, Assimp, Dear ImGui, ozz-animation, Intel OIDN, Embree, OptiX/CUDA, Vulkan, SDL2, JSON kütüphaneleri, stb, TinyEXR, NanoVDB/OpenVDB, miniz ve ilgili notlar için [LICENSE.txt](LICENSE.txt) içindeki **Third-party components** bölümüne bakın.

## 🙏 Teşekkürler

**Embree** (Intel CPU ışın izleme) · **OptiX** (NVIDIA GPU ışın izleme) · **Vulkan** · **Jolt Physics** (rigid-body fizik) · **Assimp** (asset içe aktarma) · **ImGui** (UI) · **SDL2** · **Intel OIDN** (denoise) · **NanoVDB** (seyrek hacimler) · **Ozz-animation** (iskelet animasyonu) · **stb** · **TinyEXR**

## 👤 Yazar

**Kemal Demirtaş** — [@maxkemal](https://github.com/maxkemal)

- **Sorunlar:** [GitHub Issues](https://github.com/maxkemal/RayTrophi/issues)
- **Tartışmalar:** [GitHub Discussions](https://github.com/maxkemal/RayTrophi/discussions)

---

<div align="center">

**⭐ RayTrophi Studio işine yarıyorsa repoya yıldız ver.**

❤️ ve bolca ☕ ile yapıldı.

</div>
