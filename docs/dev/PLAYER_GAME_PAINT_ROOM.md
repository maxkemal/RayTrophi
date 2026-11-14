# Player ilk oyun — Boyama Odası (AVM etkileşimli oyun alanı)

> **Durum:** TASLAK — 2026-11-14: Player'ın ilk ürün demosu. Hedef pazar AVM'lerdeki kapalı, 6–10 yaş etkileşimli oyun alanları. Kod yazılmadı. Kullanıcı yönü: mevcut yetenekler yeniden yazılmaz, genel (özel donanımsız) kamera, gerçek zamanlı GPU partikül + çok yüzeyli ıslak boya.

Üst plan: [RAYTROPHI_PLAYER_PLAN.md](RAYTROPHI_PLAYER_PLAN.md). Partikül olay tasarımı:
[PARTICLE_SYSTEM_GPU_ROADMAP.md](PARTICLE_SYSTEM_GPU_ROADMAP.md) §9. İlk kare kabulü:
[PLAYER_STATIC_VIEWER_ACCEPTANCE.md](PLAYER_STATIC_VIEWER_ACCEPTANCE.md) (2026-11-14 PASS).

## 0. Ürün bağlamı — bu bir PC oyunu değil

İlk oyunlar AVM'deki kapalı oyun alanlarında koşacak: bir oda, duvara/zemine yansıtma
ya da büyük ekran, aynı anda birkaç çocuk, gün boyu gözetimsiz çalışma. Bu bağlam
demodan çok **motorun** gereksinimlerini belirler:

| Bağlam | Motor gereksinimi |
|---|---|
| Aynı anda 4–10 çocuk | N kişilik poz girişi, kalıcı kişi kimliği, örtüşmede (occlusion) kişiyi kaybetmeme |
| 6–10 yaş, okuma yok | Talimatsız oynanış: ilk hareket anında görsel + ses cevabı; başarısızlık durumu yok |
| Oturum (3–5 dk) ve sıra | Oturum durum makinesi: bekleme (attract) → oyun → final → sıfırlama |
| Gün boyu gözetimsiz | Cihaz kaybında kendini kurtarma (Player'da OptiX cankurtaranı yok), watchdog, otomatik başlatma, kayıt |
| Birden çok mekân | Uzaktan izleme: mevcut TLS IPC (`RtIpcSecurity`) zaten var; sağlık/oturum sayaçları oradan okunur |
| Yansıtma / ekran | Çoklu çıkış, projeksiyon bükme (warp) + kenar karıştırma; kamera ↔ oda ↔ projektör kalibrasyonu |
| Çocuk görüntüsü | Varsayılan: görüntü yalnız bellekte, kayıt/ekran görüntüsü yok; ailelere verilen çıktı yalnız **eser** |
| Operatör (oyun alanı görevlisi) | Başlat/durdur/ses/zorluk/içerik seçimi — IPC'den, üstüne basit operatör ekranı |

### 0.1 İki gösterim modu (aynı çekirdek)

- **Ayna/pencere modu:** büyük ekran ya da tek duvar yansıtması. Çocuk kendini maskeli
  video kartı olarak sanal odada görür. Kullanıcının ilk tarifindeki mod.
- **Oda modu (projeksiyon eşleme):** gerçek odanın duvarları ve zemini tuvaldir. Sanal
  oda fiziksel odanın kalibre kopyası; video kartı gerekmez, çocuğun gerçek bedeni oradadır.
  Boya gerçek duvarda akar. Mevcut oyunlardan ayrışmanın en güçlü yolu budur.

Ortak olan her şey (poz, kurallar, partikül, boya, ses) iki modda aynı; fark yalnız
kamera/çıkış eşlemesi ve video kartının açık olması.

## 1. Rakiplerden ayrışma — ne "daha etkileşimli" demek

Mevcut etkileşimli zemin/duvar sistemleri çoğunlukla 2B: dokunma noktası → önceden
çizilmiş animasyon. Bu motorun farkı **fiziğin gerçekten çalışması**:

- Boya bir sprite değil: akar, damlar, renkler ıslakken karışır, emilir, kurur
  (mevcut ıslak boya modeli). Aynı hareket iki kez aynı sonucu vermez.
- Patlama bir animasyon değil: partiküller sahneye çarpar, nereye çarptıysa iz bırakır.
- Çocuğun bütün bedeni etkileşir (iskelet + vekiller), yalnız temas noktası değil.
- Oda birikir: oturum boyunca çocukların ortak eseri büyür ve finalde canlanır.

## 2. Demo v1 — oturum akışı

1. **Bekleme (attract):** oda sakin, birkaç balon süzülür. Çocuk alana girer girmez
   bedeninin çevresinde kendi renginde hafif bir iz/parıltı belirir — talimatsız ilk cevap.
   Her çocuğa bir renk atanır.
2. **Oyun:**
   - Balon patlat → balonun renginde partikül patlaması → çarptığı yerde ıslak leke.
   - **El boyası:** duvara yakın el (oda modunda gerçek duvar) kendi renginde damla bırakır.
   - **Ayak boyası:** zemine sert basış (aşağı hız eşiği) zeminde sıçrama bırakır.
   - **Islakken karışım:** kırmızı lekeye mavi gelirse mor — ıslak modelin pickup rezervuarı.
   - **Özel balonlar:** dev balon yalnız 2–3 çocuk birlikte dokununca patlar (iş birliği);
     gökkuşağı balonu çok renkli; **su balonu** boyayı yıkar/akıtır (ıslak modelin runoff'u).
3. **Ortak hedef:** odanın boyanmış oranı (atlas üzerinde GPU indirgeme) bir çubukla
   gösterilir; eşiklerde oda tepki verir (müzik katmanı eklenir, ışık değişir).
4. **Final:** eşik ya da süre dolunca boyalı bölgeler **canlanır** — mavi lekelerden
   balıklar, yeşilden çiçekler (mevcut scatter/partikül/animasyon; boyanmış maskeyi
   yoğunluk kaynağı olarak kullanır). Konfeti. Odanın eseri (çocuk görüntüsü olmadan)
   çıkış ekranında / QR ile gösterilir.
5. **Sıfırlama:** boya kurur ve solar, oda bekleme durumuna döner.

## 3. Platform ilkelleri — demo değil, sonraki oyunların da kullanacağı katman

Demoya özel kod yazmak yerine her oyunun kullanacağı ilkeller kurulur. Python Player'da
yok; oyun mantığı **veri olarak kural**dır ve CLAUDE.md §1 gereği tamamı IPC'den
okunur/yazılır, Studio'da panelden düzenlenir.

| İlkel | Örnek |
|---|---|
| **Tetik** | temas (vekil × nesne), bölge giriş/çıkış, hareket (basış, kollar yukarı), grup (N kişi aynı nesnede), eşik (kapsama %), zamanlayıcı, partikül olayı |
| **Eylem** | emitter tetikle (renk/yön parametreli), boya damlası, ses çal, animasyon/state tetikle, nesne göster/gizle/doğur, durum değiştir |
| **Durum** | oturum durum makinesi (bekleme/oyun/final/sıfırlama), sayaçlar, kişi başı renk |
| **Ölçüm** | kapsama oranı, kişi sayısı, olay sayaçları, kare süresi — hepsi IPC'de |

İlk oyun ve sonrakiler (balık tutma, top havuzu, ritim zemini…) bu tabloyla kurulur.

## 4. Bugünün gerçeği (2026-11-14 ölçümü, kodu okuyarak)

| Konu | Durum | Kanıt |
|---|---|---|
| Player sahne + görüntü | **PASS** (EchoRoomsLab, 26 mesh, device-loss yok) | kabul belgesi |
| GPU partikül yolu | Yalnız **çarpıştırıcısız**, self-collision'sız, domain'siz sistem device-resident. Çarpıştırıcılı sistem CPU referansında. | ROADMAP §2; `sim_particle_ballistic.comp` 47 satır, çarpışma bağlaması yok |
| GPU collider | Var ama **ızgara sorgusu**: `Fluid::ColliderGpuQuery` mesh BVH'ini cihazda tutar, kafes noktalarına uzaklık/SDF döndürür. | `include/Fluid/ColliderGpuQuery.h` |
| Partikül olayları | `OnCollision` / sub-emitter / surface splat **tasarımda var, kodda yok**. | ROADMAP §9 |
| Jolt temasları | Var: `RigidBodySystem::contactEvents()` | `include/RigidBodySystem.h:397` |
| Islak boya modeli | Var, CPU: `noteWetDab` + `tickWetPaint` (difüzyon, akma, emilme, damla başı, uç birikimi, pickup rezervuarı). | `src/Paint/MeshPaintAdapter.cpp:1909` |
| Islak boya tick'i | **Panelde** (`scene_ui_modifiers.cpp:6202`, ImGui DeltaTime). Player'da ilerlemez. | — |
| Islak boya ölçeği | Adaptör başına **tek** `active_region`; Studio tek nesne boyar; adaptör `Triangle` facade'dan kuruluyor (`RtApiPaint.cpp:85`). | `MeshPaintAdapter.h:122` |
| Boya API'si | Katman işlemleri var; **dünya noktasına damla** yok. | `RtApiPaint.cpp` |
| Rig sürüşü | Klip/state machine/graf parametresi var; **dış poz girişi yok**. `RigChainIK` var. | `RtApi.h:3151–3247` |
| Görüntü kanalı | IPC JSON; paylaşımlı bellek yok. | — |
| Çoklu çıkış / warp / kalibrasyon | Yok (tek SDL penceresi). | — |
| Cihaz kaybı kurtarma | Player'da yok (plan §4 riski). | — |

## 5. Mimari

```text
 [Algı köprüsü — harici süreç; herhangi bir kamera (1..N) + herhangi bir kütüphane]
     │ IPC (değer): input.set_body_poses {kişi id, eklem[], güven, t}  (N kişi)
     │ Paylaşımlı bellek (görüntü, yalnız ayna modu): RGBA, alpha = kişi maskesi
     ▼
 [Player — ortak kare döngüsü, sabit adım, oturum durum makinesi]
   girdi → anim.set_external_pose → rig/IK → el/ayak/gövde kinematik vekilleri
   kural motoru: tetikler (Jolt temas, bölge, hareket, grup, kapsama, partikül olayı)
                 → eylemler (emitter, boya, ses, anim, göster/gizle, durum)
   GPU partikül: balistik + statik sahne SDF + analitik kapsüller → olay buffer'ı (cihazda)
   GPU ıslak boya: olay + el/ayak damlaları → çok yüzeyli atlas → ıslak tick → materyal
   render: RayFusion (sabit bütçe); çıkış eşleme (ekran | projektör warp/blend)
```

### 5.1 Genel kamera girişi (karar 2026-11-14)

- Özel donanım istenmez. Kamera ve kütüphane (ör. MediaPipe Pose + segmentasyon; varsa
  RGB-D SDK'sı) **motorun dışında**. Motor kamerayı tanımaz. Birden çok kamera varsa
  füzyon ve kişi kimliği köprünün işidir; motor N iskelet alır.
- **D'siz konum:** kalibre zemin düzlemi + ayak piksel konumu → zemindeki X/Z.
  Derinlik kamerası kaliteyi artırır, şart değildir.
- **İskelet → mevcut rig:** `anim.set_external_pose` dış eklemleri mevcut rig'e/IK'ya
  verir. Rig görünmez olabilir; görevi vekilleri taşımak.
- **Görüntü → video kartı (ayna modu):** kişinin derinliğinde kameraya bakan kart, doku
  mevcut yükleme yoluyla (`IBackend::uploadTexture2D`) her kare güncellenir.
- **Neden iki kanal:** 640×480 RGBA @30 fps ≈ 37 MB/s; JSON/base64 ile pipe'tan geçmez.
- **Gizlilik:** çocuk görüntüsü yalnız bellekte; `viewport.get_screenshot` /
  `render.start` varsayılan olarak video kartını **gizler**; eser çıktısında hiç yok.
- **Test kazancı:** ajan sahte N kişi pozu + sahte maskeli kare ile bütün oturumu
  kamerasız sürer (sahte çocuk koşucuları). Bu oyunun QA altyapısıdır.

### 5.2 GPU partikül çarpışması

Device-resident balistik yol genişletilir; CPU referansı Studio'da doğrulama için kalır.

- **Statik sahne:** oda sahne yüklenince **bir kez** SDF'e pişirilir (`ColliderGpuQuery`
  kafes sorgusu, signed_distance). Kernel SDF'i üç doğrusal örnekler, normal = gradyan.
- **Dinamik:** vekiller ve balonlar **analitik** kapsül/küre (kare başına küçük buffer).
- **Olay:** çarpan partikül ölür ve sınırlı append buffer'a `{konum, normal, renk,
  enerji, yüzey kimliği}` yazar. Taşma sayılır, IPC'den okunur. Olaylar CPU'ya inmez.

### 5.3 GPU ıslak boya (çok yüzey, maliyet ıslak alanla)

- **Aynı model GPU'ya:** `tickWetPaint` denklemleri ve `BrushSettings` ıslak parametreleri
  compute shader'a taşınır; yeni boya modeli yok.
- **Çok yüzey atlası** + statik yüzeyler için bir kez **UV→dünya konumu/normal atlası**.
- **Döşeme aktif listesi** (32×32): damga yalnız olayın dünya yarıçapındaki döşemeleri
  açar; tick yalnız aktif döşemelerde; kuruyan döşeme düşer.
- **Damga:** aktif texel, karenin olaylarını kendi dünya konumuna göre tartar — üçgen
  arama yok, dikişler kendiliğinden doğru.
- **Kapsama ölçümü:** atlas üzerinde GPU indirgeme → oran + renk başına oran (IPC).
- **Kuruma/solma** sıfırlama için zamanla parametrelenir (oturum sonu temiz oda).
- **Studio:** fırça da aynı GPU yoluna geçer (CLAUDE.md §5); adaptör o partide flat SoA'ya.

### 5.4 Çıkış eşleme ve kalibrasyon

- Sanal kamera(lar) fiziksel projektör(ler)in iç/dış parametreleriyle kurulur (oda modu).
- Çıkış başına warp ızgarası + kenar karıştırma maskesi (son işlem geçişi).
- Kalibrasyon: projektöre desen yansıt, kameradan oku — köprü süreci yapar, sonucu
  IPC'den yazar. İlk sürüm: elle köşe eşleme (operatör ekranı + IPC).

### 5.5 Kiosk sağlamlığı

- Vulkan cihaz kaybı → cihazı ve sahne GPU kaynaklarını yeniden kur, oturum durumunu koru.
- Watchdog (harici): IPC nabzı kesilirse Player'ı yeniden başlat; çökme logu korunur
  (atexit dersi: FATAL logu silinmemeli).
- RT pipeline önbelleği kurulumda ısıtılır (plan §4: ilk derleme 168 sn).
- Sağlık sayaçları (kare süresi, kişi sayısı, oturum sayısı, hata) IPC + TLS uzaktan.

## 6. Sıra

| # | İş | Kabul |
|---|---|---|
| 0 | ✔ Player ilk kare | 2026-11-14 PASS |
| 1 | Ortak kare döngüsü: partikül + Jolt + anim tick tek yerde, sabit adım | Studio ve Player aynı tick; IPC `runtime` adım sayacı |
| 2 | GPU partikül: statik SDF + analitik kapsül + olay buffer'ı | 50k partikül ≤1.5 ms GPU (hedef); taşma sayacı IPC'de |
| 3 | GPU ıslak boya: atlas, döşeme listesi, damga, tick, kapsama | 4+ yüzeyde eşzamanlı leke; maliyet ıslak döşemeyle orantılı; ≤2 ms (hedef) |
| 4 | `paint.wet_dab` + kapsama okuması (3'ün IPC yüzü) | IPC ile damla → atlas/kapsama değişimi okunur |
| 5 | Kural motoru (tetik/eylem/durum) + oturum durum makinesi | balon patlat → emitter; IPC'den kural yaz/oku |
| 6 | `input.set_body_poses` (N kişi) + `anim.set_external_pose` + vekiller + hareket tetikleri | sahte 4 çocuk koşucusu bütün oturumu oynar |
| 7 | Demo içeriği: oda, balon türleri, final canlanması, ses | sahte oturum: kapsama eşikleri → final |
| 8 | Ayna modu: paylaşımlı bellek görüntü + video kartı + gizlilik | sahte maskeli kare kartta; ekran görüntüsünde yok |
| 9 | Oda modu: çıkış eşleme, warp/blend, kalibrasyon | tek projektörle köşe eşleme |
| 10 | Kiosk: cihaz kaybı kurtarma, watchdog, pipeline ısıtma, sağlık sayaçları | zorlanmış device-lost sonrası oturum sürer |
| 11 | Algı köprüsü (webcam + MediaPipe; RGB-D isteğe bağlı; çoklu kişi) | gerçek kamera ile 7'nin testi |

1–7 kamerasız ilerler; gerçek kamera ve donanım en sonda.

## 7. Açık kararlar (kullanıcı)

- ✔ **Gösterim modu (karar 2026-11-14):** ilk demo **ayna modu, büyük ekran**
  (75–86" TV ya da LED duvar + ekranın üstünde tek webcam). Gerekçe: müşterinin
  projeksiyon kalitesi bilinmiyor; AVM ortam ışığı projeksiyon kontrastını düşürür;
  kalibrasyon gerektirmez, demo her yere taşınır; çocuk kendini gördüğü için "vay"
  anı en güçlü. Oda modu mimaride kalır (çıkış eşleme bir ayar, adım 9); 8 ve 9 yer
  değiştirmez ama 9 demo için şart değildir. Ayna modunda sanal duvarlar ekran
  düzleminin arkasında durur; el boyası "ekrana uzanma" (derinlik eşiği) ile tetiklenir.
- **Hedef donanım:** AVM kurulumunda GPU sınıfı (kare bütçesi buna göre).
- **Aynı anda azami çocuk sayısı** ve oturum süresi.

## 8. Riskler

- **Paralel ajan:** Codex `ParticleSimulation.*` ve partikül render köprülerinde
  (AJAN_KOORDINASYON §1). Adım 2 aynı dosyalara dokunur → başlamadan iş bölümü.
- Webcam tabanlı çoklu kişi takibinde örtüşme ve kimlik değişimi — köprü sorunu, ama
  oyun kuralları kişi kaybını tolere etmeli (renk yeniden atanabilir).
- SDF çözünürlüğü ↔ ince eşyalar; atlas boyutu × yüzey sayısı bellek bütçesi.
- Oda modunda projektör gölgesi (çocuk ışığı keser) — tasarım bunu oyun öğesi yapabilir.
- Video kartının sahne ışığıyla ilişkisi (unlit mi, ortamla çarpılmış mı) — görsel karar.
- `material_count=1` şüphesi (kabul belgesi) — renkli materyalli sahnede doğrulanmalı.
