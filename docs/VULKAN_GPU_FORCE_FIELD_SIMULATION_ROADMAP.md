# Vulkan GPU Force-Field Simulation Roadmap

## Amaç

Particle, gas ve APIC fluid sistemlerindeki force-field değerlendirmesini ve
ona bağlı temel timestep aşamalarını Vulkan Compute üzerinde kalıcı GPU
buffer'larıyla çalıştırmak.

Üretim yolu:

`Scene Force Fields -> Packed GPU Snapshot -> Vulkan Compute Solvers -> Vulkan RT`

CPU yolu kaldırılmayacak. CPU;

- doğruluk referansı,
- cihaz/dispatch hatasında güvenli fallback,
- deterministik inceleme ve test

olarak korunacak. Üretim hedefi Vulkan Compute'tur. CUDA, Vulkan yolu kararlı
ve ölçülmüş hale geldikten sonra aynı backend-bağımsız sözleşmeyi izler.

Bu belge şu genel planların odaklanmış yürütme planıdır:

- `docs/simulation_physics_foundation_plan.md`
- `docs/VULKAN_PRODUCTION_VOLUMETRICS_ROADMAP.md`

## Değişmez kurallar

1. GPU buffer adresi, onu üreten compute backend'den uzun yaşayamaz.
2. RT hiçbir zaman serbest bırakılmış veya yeniden boyutlandırılmakta olan
   simulation buffer'ını okuyamaz.
3. Backend değişiminden önce bütün ödünç dense device adresleri geçersizlenir.
4. Bir faz, CPU/GPU karşılaştırma testi ve Play/Pause/backend-switch testi
   geçmeden tamamlanmış sayılmaz.
5. GPU dispatch başarısızlığı yalnız ilgili domain/sistemi fallback'e indirir;
   uygulamayı veya diğer sistemleri durdurmaz.
6. Normal playback sırasında tam parçacık veya tam grid readback yapılmaz.
7. Kuvvet bir timestep içinde yalnız bir kez uygulanır.
8. Sistem ve domain sahipliği korunur:
   `(particle_system_id, domain_index)` birlikte kimliktir.

## Mevcut durum

Tamamlanan altyapı:

- [x] `SimulationWorld` ve sıralı sistem yürütme
- [x] CPU `SimulationForceFieldSnapshot`
- [x] `PackedForceField` dizisi
- [x] Ortak compute storage buffer upload'u
- [x] Gas/fluid/particle affect maskeleri
- [x] Vulkan Compute buffer ve dispatch soyutlaması
- [x] GPU gas advection, combustion, buoyancy, vorticity, turbulence ve pressure
- [x] GPU APIC P2G, pressure/MGPCG, G2P ve density aşamalarının önemli bölümü
- [x] Vulkan Compute -> Vulkan RT canlı dense-field paylaşımı
- [x] Backend değişiminde canlı dense device-address invalidation

Kalan temel işler:

- Gas/fluid/particle kuvvet değerlendirmesi Vulkan Compute yoluna taşındı.
- Fluid Wind surface-drag XZ kolon yüzeyi GPU'da kuruluyor.
- APIC advection tail GPU'da; parçacık sayısını değiştiren reseed/compaction ve
  bazı boundary yayınları CPU sözleşmesinde kalıyor.
- Gas için active AABB/brick dispatch ve force-field broadphase henüz üretim
  optimizasyonu olarak açık.
- Ölçüm, uzun süreli VRAM kararlılık testi ve CPU/GPU sayısal karşılaştırma
  matrisi Faz 8 üretim kapısının parçası olarak açık.

---

## Faz 0 — Ölçüm, güvenlik ve referans sahneleri

Amaç: optimizasyondan önce darboğazı ve doğru sonucu ölçülebilir hale getirmek.

### İşler

- [ ] Force-field CPU süresini ayrı ölç:
  - particle force
  - gas grid force
  - APIC particle force
  - fluid surface-column build
- [ ] Force öncesi/sonrası upload, download ve synchronize sürelerini ölç.
- [ ] Domain başına aktif force sayısını ve değerlendirilen eleman sayısını göster.
- [ ] GPU buffer generation/owner-backend kimliği ekle.
- [ ] Eski backend'e ait handle/device-address kullanımını assertion ile reddet.
- [ ] Aşağıdaki referans sahnelerini kaydet:
  - 100k discrete particle + Wind
  - gas plume + Wind + Vortex
  - APIC pool + surface-drag Wind
  - iki gas + bir fluid + çoklu force field
  - çoklu preset + collider + emitter
- [ ] CPU referans çıktıları için velocity/density/temperature özet hash'leri al.

### Tamamlanma ölçütü

- Profil ekranı force-field eklenince oluşan CPU ve transfer maliyetini ayrı gösterir.
- Backend switch, Play/Pause ve cache replay sırasında validation hatası oluşmaz.
- Referans sahneler tekrar üretilebilir ve karşılaştırılabilir.

---

## Faz 1 — Ortak GPU force evaluator

Amaç: bütün solver shader'larının aynı force matematiğini kullanması.

### İşler

- [ ] `PackedForceField` CPU/GLSL binary layout'u için `static_assert` ve shader
  layout testi ekle.
- [x] Ortak shader include oluştur:
  `shaders/include/sim_force_fields.glsl`
- [x] Şu alanları CPU ile aynı sırada uygula:
  - directional
  - point/attractor
  - wind
  - vortex
  - turbulence/noise
  - drag
- [x] Shape ve falloff:
  - sphere
  - box/OBB
  - infinite/global
  - inner radius
- [x] Affect mask ve enabled kontrolü ekle.
- [x] NaN/Inf, sıfır yön ve negatif radius korumaları ekle.
- [x] Force buffer boşken sıfır maliyetli early-out sağla.
- [ ] Aynı evaluator için küçük CPU test fonksiyonu ve GPU probe dispatch yaz.

### Tamamlanma ölçütü

- Tek nokta örneklerinde CPU/GPU ivme farkı kabul edilen epsilon içinde kalır.
- Birden fazla force sırasız veya eksik uygulanmaz.
- GPU evaluator hiçbir renderer/Vulkan RT kaynağına bağımlı değildir.

---

## Faz 2 — Discrete Particle tamamen GPU force yolu

Amaç: particle emitter'lardan çıkan bütün parçacıkların force integration'ını
GPU'da tutmak.

### İşler

- [x] Particle force/integrate compute shader'ına ortak evaluator'ı bağla.
- [ ] Parçacık başına:
  - [x] gravity
  - [x] force fields
  - [x] linear drag
  - [ ] lifetime/age
  - [ ] velocity clamp (mevcut discrete CPU yolunda ayrıca sınır yok)
  - [ ] position integration
- [ ] Object/force-field emitter spawn sonucunu doğrudan GPU spawn kuyruğuna taşı.
- [ ] CPU spawn descriptor'ları için toplu upload ring buffer kullan.
- [ ] Dead/alive slot compaction veya GPU free-list tasarla.
- [ ] Particle collider broadphase'i GPU'ya uygun ayrı aşama yap.
- [ ] CPU force uygulanmış işaretini kaldır; force yalnız GPU dispatch'te çalışsın.
- [ ] Render bridge için her kare SoA readback yerine GPU instance-transform
  üretim yolu planla/uygula.

### Tamamlanma ölçütü

- Wind açık/kapalı arasında tam particle-buffer upload/download oluşmaz.
- 100k particle + dört force field normal playback'te CPU force döngüsüne girmez.
- CPU/GPU trajectory karşılaştırması sabit timestep'te kabul edilen sapma içindedir.

---

## Faz 3 — Gas force tamamen GPU

Amaç: gas grid force değerlendirmesini CPU `GridFluid::addForceFields` yolundan
çıkarmak.

### İşler

- [x] `sim_gas_force_evaluate.comp` ve `sim_gas_force_gather.comp` oluştur.
- [x] MAC velocity yüzlerine force katkısını yarışsız biçimde uygula.
- [x] Dispatch sırasını CPU referansıyla aynı tut:
  transport -> combustion/buoyancy -> external force -> projection.
- [x] Domain affect maskini shader'a geçir.
- [x] İlk sürümde tam grid dispatch; doğruluk sonrasında aktif AABB/brick dispatch.
- [ ] Force uygulandığında CPU grid velocity'nin tekrar GPU'ya yüklenmesini kaldır.
- [x] GPU force başarılıysa CPU `addForceFields` aşamasını açıkça skip et.
- [x] Force başarısızsa aynı frame içinde kontrollü CPU fallback uygula; çift
  kuvvet uygulanmasını engelle.

İlerleme notu: buoyancy -> external force -> vorticity -> turbulence artık tek
device-resident velocity zincirinde çalışır. Dört ayrı tam MAC upload/readback
yerine zincir başında bir upload ve sonunda bir doğruluk yayını kalmıştır.
CPU boundary yayını sonrasında dissipation -> pressure da resident zincirdir;
dissipation sonrası eski tam-field readback/re-upload kaldırılmıştır. Kalan
geçici yayın, ilk velocity-advection zinciri ve CPU boundary aşaması GPU'ya
alındığında tamamen kaldırılacaktır.

### Tamamlanma ölçütü

- Wind/Vortex bulunan Vulkan gas domain CPU force döngüsüne girmez.
- Normal playback tam-grid velocity transferi yapmaz.
- Plume yönü, momentum ve pressure sonucu CPU referansına yakın kalır.
- Çoklu gas domain aynı packed force buffer'ını güvenle paylaşır.

---

## Faz 4 — APIC fluid body force tamamen GPU

Amaç: fluid parçacık kuvvetlerini GPU P2G öncesinde doğrudan GPU buffer'ında
uygulamak.

### İşler

- [x] `sim_fluid_particle_forces.comp` ortak evaluator yolunu tamamla.
- [x] Gravity, genel force field, drag ve container-motion katkısını birleştir.
- [x] GPU velocity clamp ekle.
- [ ] Force dispatch sonucunu doğrudan P2G girdisi yap.
- [x] Vulkan yolunda `force_fields_require_cpu` nedeniyle oluşan CPU force
  değerlendirmesini kaldır (P2G öncesi geçici velocity readback/upload halen
  Faz 6 resident-chain işidir).
- [ ] GPU başarısızlığında CPU uygulanmış/uygulanmamış durumunu tek state flag ile
  güvenli yönet.
- [ ] Çoklu fluid domain ve çoklu force için buffer offset/count doğrula.

### Tamamlanma ölçütü

- Force field eklemek APIC parçacık velocity upload'u üretmez.
- P2G/pressure/G2P zinciri force açıkken de GPU yolunda kalır.
- Particle sayısı büyüdükçe CPU force süresi artmaz.

---

## Faz 5 — Fluid Wind surface-drag GPU yolu

Amaç: sıvının tamamını uçuran body force yerine mevcut fiziksel yüzey rüzgârı
davranışını GPU'da korumak.

### İşler

- [x] Domain başına XZ surface-height buffer ayır.
- [x] Her frame buffer'ı compute ile temizle.
- [x] Particle -> XZ column height için pozitif grid-space float atomic max kullan.
- [x] İkinci dispatch'te parçacığın yüzeye uzaklığını hesapla.
- [x] `fluid_surface_depth`, coupling, curl detail ve relative wind velocity'yi uygula.
- [x] Birden fazla Wind alanını ortak evaluator üzerinden topla.
- [x] Surface-drag kapalı Wind için normal body-force yolunu kullan.
- [x] Vulkan yolunda CPU column buffer ve CPU particle pass'ini devreden çıkar.

### Tamamlanma ölçütü

- Derin sıvı sakin kalırken yüzey rüzgârla taşınır.
- CPU ve GPU yüzey-bandı sınıflandırması referans sahnede uyuşur.
- Wind eklendiğinde CPU column-build süresi sıfıra iner.

---

## Faz 6 — Collider ve emitter GPU sürekliliği

Amaç: force yolu GPU'ya geçtiğinde collider/emitter yüzünden tekrar CPU
round-trip oluşmasını engellemek.

### İşler

- [ ] Point/Object/Mesh Surface emitter verisini GPU-friendly descriptor'a paketle.
- [ ] Object transform/keyframe sonucunu frame başına tek küçük upload ile yayınla.
- [ ] Collider descriptor ve dirty-region bilgisini ortak GPU buffer'da tut.
- [ ] Particle collider broadphase ve primitive temaslarını GPU'ya taşı.
- [ ] Gas/fluid mevcut GPU collider mask/SDF yoluyla birleştir.
- [ ] Çoklu sistemde descriptor aralıklarını system/domain offset tablosuyla ayır.
- [ ] Preset ekleme/silme sırasında stable slot veya generation kontrollü rebuild kullan.

### Tamamlanma ölçütü

- Çoklu emitter/collider bulunan sahnede yanlış sisteme çapraz etki oluşmaz.
- Hareketli/keyframe'li obje emitter ve collider tam konumu takip eder.
- Emitter/collider eklemek RT'ye stale pointer/customIndex taşımaz.

---

## Faz 7 — Kalıcı GPU timestep ve transfer temizliği

Amaç: normal playback'i tek veya az sayıda command submission ile tamamlamak.

### İşler

- [ ] Particle/gas/fluid field buffer'larını timestep boyunca GPU'da tut.
- [ ] Dispatch'leri tek command-buffer zincirinde sırala.
- [ ] Stage arası yalnız gerekli compute barrier'ları kullan.
- [ ] Per-stage `synchronize()` çağrılarını kaldır.
- [ ] Readback'i yalnız şu durumlarla sınırla:
  - kompakt istatistik
  - kullanıcı cache/export talebi
  - validation/reference modu
- [ ] Vulkan Compute -> Vulkan RT erişim bariyeri ve sahiplik sözleşmesini belge.
- [ ] Buffer resize için retire queue/fence kullan; RT okurken destroy etme.
- [ ] Backend switch sırasında:
  synchronize -> invalidate borrowed addresses -> destroy compute -> rebuild.

### Tamamlanma ölçütü

- Normal playback tam parçacık/grid readback yapmaz.
- Pause, rewind, cache replay ve backend switch TDR üretmez.
- VRAM kullanımı uzun testte sabit plato oluşturur.

---

## Faz 8 — Optimizasyon ve üretim kapısı

### İşler

- [ ] Gas active AABB/brick dispatch.
- [ ] Force alanı AABB broadphase: her eleman bütün force'ları değerlendirmesin.
- [ ] Force'ları global/local ve sistem maskesine göre GPU'da compact et.
- [ ] Çok kullanılan Wind/Directional alanlar için hızlı uniform yol.
- [ ] Timestamp query ile domain/stage GPU süreleri.
- [ ] VRAM/live-buffer/retired-buffer sayaçları.
- [ ] NaN, clamp, invalid handle ve fallback sayaçları.
- [ ] Interactive/Preview/Final kalite bütçeleri.

### Tamamlanma ölçütü

- Dört force field, iki gas, bir fluid ve 100k particle sahnesi bütçeyi aşmaz.
- Boş gas alanı full-grid force maliyeti ödemez.
- 10 dakikalık playback'te VRAM artışı ve driver validation hatası görülmez.

---

## Zorunlu test matrisi

Her faz aşağıdaki matrisin ilgili kısmını geçmelidir.

| Test | Vulkan Compute + Vulkan RT | Vulkan Compute + OptiX | CUDA/CPU fallback |
|---|---:|---:|---:|
| Play/Pause x20 | zorunlu | zorunlu | zorunlu |
| Timeline rewind/cache replay | zorunlu | zorunlu | zorunlu |
| Backend A -> B -> A x10 | zorunlu | zorunlu | zorunlu |
| İki gas preset | zorunlu | zorunlu | referans |
| Gas + fluid + particles | zorunlu | zorunlu | referans |
| Wind + vortex + turbulence | zorunlu | zorunlu | referans |
| Çoklu collider/emitter | zorunlu | zorunlu | referans |
| Animasyonlu object emitter | zorunlu | zorunlu | referans |
| 10 dakika VRAM stabilitesi | zorunlu | zorunlu | gözlem |

## Faz kapatma kaydı

Bir faz tamamlandığında bu şablon belgeye eklenir:

```text
Faz:
Tarih:
Değişen dosyalar:
CPU referans sonucu:
GPU sonucu:
Performans önce/sonra:
VRAM önce/sonra:
Fallback testi:
Play/Pause testi:
Backend-switch testi:
Bilinen açıklar:
```

## Uygulama sırası

Kesin sıra:

1. Faz 0 — ölçüm ve güvenlik
2. Faz 1 — ortak evaluator
3. Faz 2 — discrete particles
4. Faz 3 — gas
5. Faz 4 — APIC body force
6. Faz 5 — fluid surface Wind
7. Faz 6 — collider/emitter sürekliliği
8. Faz 7 — kalıcı timestep ve transfer temizliği
9. Faz 8 — optimizasyon ve üretim kapısı

Yeni özellik ekleme bu sıra tamamlanana kadar ikinci önceliktedir. Bir fazda
driver loss, stale handle, kontrolsüz fallback veya sürekli VRAM artışı görülürse
sonraki faza geçilmez.

## 2026-07-30 APIC resident-tail ilerleme notu

- Vulkan APIC tail için `sim_fluid_advect_tail.comp` eklendi.
- GPU G2P sonrasında spray air-drag, velocity damping, RK2 particle advection,
  closed/open/periodic domain sınırları ve voxel-collider axis-slide aynı
  kalıcı particle/grid buffer'larında çalışıyor.
- GPU tail başarılı olduğunda ikinci CPU `Fluid::step` advection ve boundary
  döngülerini tekrar çalıştırmıyor.
- Açık domain outflow kompaksiyonu ve parçacık sayısını değiştiren reseed
  sahiplik aşaması mevcut CPU render/cache SoA sözleşmesi nedeniyle CPU'da
  tutuldu; ağır hücre/parçacık advection hesabı artık değildir.
- Domain silme iki fazlıdır: RT volume/TLAS ayrılır, ardından solver compute
  buffer'ları bırakılır. Hiyerarşi ve viewport silme aynı güvenli yolu kullanır.

## 2026-07-30 doğrulama özeti

- Wind, vortex ve turbulence alanları gas domain üzerinde Vulkan Compute ile,
  gözle görülür ek CPU force maliyeti olmadan doğrulandı.
- Fluid force yolu Vulkan Compute olarak raporlandı; vortex ile SDF yüzeyde
  hızlı girdap oluşumu doğrulandı.
- GPU collider-source yolu, hareketli collider çevresinde akışın sarılması,
  alev tutunması ve materyal kontrollü otomatik tutuşma ile doğrulandı.
- Çoklu gas preset, domain, emitter, collider ve force-field sahnede birbirini
  silmeden çalışacak şekilde ayrıştırıldı.
- Gas cache replay, Play/Pause, backend geçişi ve domain silme yollarındaki
  stale-handle/TDR sınıfı hatalar için kaynak yaşam süresi korumaları eklendi.
- Uzun süreli testte gas compute buffer sayısı karelerle büyümek yerine sabit
  plato oluşturdu; kare başına oluşturulan domain buffer kaçağı giderildi.
- Bilinen açıklar: Vulkan RT'ye Solid moddan geçişte Fluid Surface SDF yayını
  her zaman kalıcı değil; kullanıcı render modu otomatik değiştirilmeden sorun
  ayrı bir yayın/senkronizasyon işi olarak bırakıldı. Cache verisi olmayan boş
  volume karesi Vulkan RT'de siyah domain gösterebilir.
