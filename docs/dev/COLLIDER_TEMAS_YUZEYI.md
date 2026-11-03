# Collider temas yüzeyi — sıvı, DEM ve MPM için tek tanım

> **Durum:** AKTİF — MPM collider teması CANLI 8/8 (yükselen/kayan/yapışık plaka, germe ve yırtılma), sub-voxel φ 12/12, duman testi 14/14 (2026-11-04). DEM ve sıvı tarafı TASLAK; kar seed + collider elle doğrulanmadı.

## Neden

Bir collider'ın yüzeyi bugün her çözücüde ayrı bir yerden geliyor, bir kısmında da
hiç gelmiyor:

| Çözücü | Bugün | Sorun |
|---|---|---|
| Parçacık sistemi, kinematik proxy | `friction` 0.45, `restitution` collider'da | Yalnız orada |
| DEM (tane) | `grain_friction` / `grain_restitution` MADDEDE | Collider'ın yüzeyi hesaba girmiyor |
| Sıvı | `viscosity_wall_slip` MADDEDE | Collider'ın yüzeyi hesaba girmiyor |
| MPM (kum, kar, elastik) | Hiçbir şey: duvara bitişik yüz **sıfırlanıyor** | Hareketli collider = duran duvar; normal yönde yapışkan |

**Tetikleyen olay (2026-11-02):** kullanıcı sahnesinde, yükselen bir collider'ın üzerindeki
sakız (yırtılma eşiği 1, akma eşiği 2) anında koptu. Aynı düzen IPC ile yeniden üretildi
(`rt_test_mpm_collider_ipc.py`, eski exe):
- Plaka 10 cm yükseldi, sakızın tepesi 0.2196 m'de hiç kıpırdamadı.
- Plakanın hücreleri voksel voksel ilerledi, alttaki parçacıklar yalnız konumdan dışarı
  itildi.
- 34. karede bütün blok tek bir katmana ezildi (min = max = 0.2375 m). Gözle görülen
  "kopma" bu ezilme.

## İlke

- **Optik ≠ mekanik.** Roughness haritası temas yüzeyini BELİRLEMEZ. Cilalı buz parlak
  ve kaygan, cilalı kauçuk parlak ve tutucu, ıslak taş parlak ama kayganlığı başka
  sebepten. Render ayarı fiziği sessizce değiştirmemeli. En fazla bir kez "roughness'tan
  doldur" önerisi olabilir; değer kopyalanır, sonra fiziğe ait olur.
- **Önce collider başına tek değer**, sistem doğrulanır. Yüzeyde değişen sürtünme
  (çamurlu yama, buzlu şerit) sonra, ayrı bir sürtünme haritasıyla gelir.
- Collider'ın yüzeyi ve maddenin yüzeyi **çift olarak** birleşir: sürtünmede
  geometrik ortalama, yapışmada küçük olan. "Buz üstünde bal" ile "beton üstünde bal"
  farklı davranmalı.
- Dört katman kuralı: API, IPC, Python ve panel. Script'ten yazılan her alan collider
  panelinde düzenlenebilir.

## Temas yüzeyi alanları (Adım 2)

| Alan | Birim | Sıvı | DEM | MPM |
|---|---|---|---|---|
| `friction` μ | — | Duvarda kayma oranı (0 = serbest kayma) | Coulomb | Izgara yüzünde teğet Coulomb |
| `adhesion` | Pa | Islatma / temas açısı (sonra) | Kopma kuvveti | Bu çekme gerilmesinin altında yüzeyden ayrılmaz |
| `restitution` | — | — | Var | Normal yönde sınırlı geri sekme |

## Adımlar

1. **MPM'de hareketli ve ayrılabilir duvar** (YAZILDI). `sim_matter_mpm_boundary.comp`:
   katıya bitişik yüz, gövde katıya doğru (katıya GÖRE) giderse katının hızını alır;
   ayrılıyorsa kendi hızını korur. Alan duvarı durağandır. Teğet yüzler değişmedi
   (serbest kayma, eskisi gibi). CPU ikizi `APICFluidSolver.cpp`
   `enforceSolidBoundaries` (granüler dal). Hem karma MPM şeridinde hem tek modelli
   granüler GPU zincirinde kullanılıyor. Sıvı şeridi değişmedi.
2. **Temas yüzeyi verisi:** collider'da μ / yapışma / sekme; IPC + panel; maddeyle çift
   birleştirme. MPM ve DEM önce bağlanır.
3. **MPM sürtünmesi ve yapışma:** yalnız ayrılmayı değil, teğet Coulomb'u ve
   `adhesion`'ı da ızgara yüzüne koymak. MAC'te teğet bileşen normal bileşenle aynı
   yüzde durmadığı için normal itkiyi komşu yüzden okumak gerekiyor; ayrıntı o adımda.
   Sakızın iki yüzey arasında **uzaması** bu adımla gelir: yapışma olmadan yükselen
   collider sakızı yalnız taşır.
4. **Sıvı:** duvarda kayma ve ıslatma, yüzey gerilimi işiyle birlikte.

## Adım 2–3 nasıl yazıldı (2026-11-02, derlenmedi)

- **Alanlar:** `contact_friction` (Coulomb μ, varsayılan 0.5) ve `contact_adhesion_pa`
  (varsayılan 0). Yeri: `SimulationColliderInfo`, `ParticleColliderDesc`, IPC
  (`collider.create/update/get`), Python (`rt.collider`), proje kaydı (ProjectManager),
  sahne serileştirici, panel (collider → Material → "Contact Surface (MPM)", yapışma
  logaritmik 0–1e6 Pa).
- ★ Mevcut `friction` alanı DOKUNULMADI. O, parçacık sistemlerinin çarpma başına teğet
  hız kaybı (0..1); Coulomb μ farklı bir birim. Aynı ada iki anlam yüklenmedi.
- **Hücre başına damga:** voxelizer her collider'ın μ ve yapışmasını kendi hücrelerine
  yazıyor (`grid.solid_contact_friction/adhesion`). Değerler önbellek imzasında ve kare
  önbelleğinde (`MatterGridFrameCache`) yer alıyor. Kinematik proxy hücreleri 0 okuyor:
  sürtünmesiz, yapışmasız.
- **MPM duvarı, iki geçiş** (`sim_matter_mpm_boundary.comp`, CPU `enforceMpmContact`):
  - Geçiş 0, teğet yüzler: katıya göre kayma, μ·Δv_n kadar azalır. Δv_n, normal
    kısıtın komşu normal yüzden kaldırmak üzere olduğu hız (normal itki).
  - Geçiş 1, normal yüzler: gövde katıya giremez. Bir yüzü tutmak için gereken gerilme
    ρ·h·Δv/dt; yapışmanın altında yüz katıya bağlı kalır, üstünde ayrılma
    yapışma·dt/(ρ·h) kadar yavaşlar.
  - Alan duvarları: durağan, sürtünmesiz, yapışmasız. Bugünkü davranış korunuyor.
- **Henüz yok:**
  - Madde tarafıyla çift birleştirme. Şimdilik μ ve yapışma yalnız collider'dan.
  - DEM bağlantısı. Tane çarpışma üçgenleri tek listede birleşiyor, hangi collider'a
    ait oldukları tutulmuyor; üçgen başına yüzey dizisi gerekiyor.
  - Sekme (`restitution`) MPM'de kullanılmıyor.

## Voksel collider ile iki tuzak (2026-11-02 canlı, düzeltme yazıldı)

- **Katın içindeki yüzler:** G2P'nin karesel şablonu duvarın bir hücre içine uzanır.
  Oradaki yüzler katının hızını taşıdığında μ 0 bile sakızı sürükledi. Artık oradaki
  değer, yanındaki açık yüzün sürtünmeden geçmiş değeri (hayalet).
- **Voksel adımı boşluğu:** plaka her h ilerlemesinde bir hücre zıplar. Arada gövde
  ile duvar arasında boş bir hücre kalır ve yapışma tutamaz. Yapışma bandı, bu boş
  hücrenin gövde tarafındaki yüzü yapışma sınırına kadar duvarın hızına tutar.
  Gerçek çözüm alt-voksel collider (SDF tabanlı ızgara teması); o, bu notun sonraki
  adımı olabilir.

## Canlı (2026-11-02, exe 22:27)

- Hayalet yüzlerden sonra sürtünme gerçekten μ'ye bağlı: μ 0 → −0.003 m, μ 0.8 → 0.0998 m
  (plaka 0.1 m).
- Yapışma bandı sakızı bir hücre boyunca taşıyor (tepe 0.200 → 0.225 m). Sonra sakız tam
  hücre sınırında takılıyor: plakanın voksel görüntüsü bir sonraki hücreyi kapladıkça
  advect'in katı çözümü parçacığı içeri sokmuyor. Hücre boşalınca boşluk bir hücreyi
  aşmış oluyor.
- **Sonraki adım: alt-voksel temas.** Collider'ın SDF'i ile parçacığın ya da yüzün gerçek
  uzaklığı okunacak. Hücre maskesi yalnız "içeride mi" söyleyebiliyor.

## Plan: alt-voksel temas (tek derleme, ONAY BEKLİYOR)

**Sorun:** MPM şeridi collider'ı yalnız hücre maskesiyle görüyor; maske "içeride mi" der,
"ne kadar uzakta" demez. Plaka bütün hücre adımlarıyla ilerler. Sakızın tepesi bir hücre
sınırında takılıyor ve yapışma bir sonraki hücreye geçemiyor (canlı: 0.200 → 0.225 m'de
durdu). Aynı sebeple yükselen plakada 2 cm boşluk var.

**Fikir:** standart MPM'deki gibi collider'ın işaretli mesafesini (φ) kullanmak. Temas
hücre maskesinden değil φ'den okunur; düzlem, küre ve kutuda tam, SDF'te SDF
çözünürlüğünde doğru olur.

| # | İş | Nerede |
|---|---|---|
| S1 | `grid.solid_phi`: hücre merkezinde collider'a işaretli uzaklık (m; içeride negatif). Voxelizer her modda damgalar: düzlem/küre/kapsül/AABB/OBB analitik; mesh SDF **üç doğrusal** örnekleme (bugün en yakın örnek); BVH en yakın üçgen + işaret. Ayak izinin 2 hücre dışına kadar yazılır, diğer yerler +∞. Voxel önbellek imzası ve kare önbelleğinde. | `ParticleSimulation.cpp` voxelizer, `FluidGrid.h`, `MatterGridFrameCache.h` |
| S2 | Cihaza yükleme (`var_solid_phi`), MPM şeridi için. | `FluidGpuTransferStages.inl`, `MatterGpuStep.inl` |
| S3 | **Parçacık düzeyinde temas** (MPM ilerletme kuyruğu, granüler mod): hücre maskesiyle bloklama yerine φ(p) üç doğrusal okunur, normal ∇φ'den. φ < 0 ise parçacık yüzeye itilir; katıya göre hız: içe doğru normal bileşen kesilir, teğette Coulomb (μ·|Δv_n|). Ayrılırken φ < h/2 bandında yapışma tutar (ρ·h·v_n/dt ≤ yapışma). Hücre maskesi parçacığı artık durdurmaz. | `sim_fluid_advect_tail.glsl` (MATTER + granular), CPU advect ikizi |
| S4 | Izgara duvarı (mevcut 3 geçiş) kalır ama "katı" yalnız φ_merkez < −h/2 olan DERİN hücre. Kısmen örtülü hücreler açık sayılır: gövde orayı doldurabilir, sınırı S3 tutar. Yapışma bandı (geçiş 2) S3 gelince kaldırılır (geriye uyum yükü yok). | `sim_matter_mpm_boundary.comp`, CPU ikizi |
| S5 | Test: `rt_test_mpm_collider_ipc.py` aynı 5 kol. Hedef 8/8; yükselmede boşluk < h/4. Ek kol: düzlem collider'a eğik düşen jöle, gerçek yüzeyde durmalı (boşluk < 3 mm). | test |

**Riskler:**
- φ damgası BVH modunda pahalı olabilir (hücre başına en yakın üçgen). Yalnız ayak izi + 2
  hücrede yapılacak ve önbellek imzasıyla korunacak.
- Kar ve kum da S3–S4'ten geçer: duvar dibindeki yığın davranışı değişebilir. Duman
  testinde `fast_fall`, `p9_floor`, `mpm_dem_clock`.
- Sıvı şeridi DEĞİŞMEZ; φ ileride sıvının duvar kaymasına da açılabilir.

**Collider damga maliyeti ölçüldü (2026-11-02, exe 22:27, derlemesiz).** Kullanıcı
arşivinden iki kaya, her kare 2 mm kaydırılan collider, 1.2 m'lik alan.
`sim.fluid.voxelize_colliders`, 10 karenin medyanı:

| Mesh | Mod | h 2.5 cm (110 k hücre) | h 1.25 cm (885 k hücre) |
|---|---|---|---|
| kaya 9.7 k üçgen | obb | 0.5 ms | 4.2 ms |
| | **mesh_bvh** | **57 ms** | **341 ms** |
| | mesh_sdf | 0.7 ms | 5.2 ms |
| slate 352 k üçgen | obb | 0.6 ms | 5.0 ms |
| | **mesh_bvh** | **174 ms** | **1240 ms** |
| | mesh_sdf | 0.7 ms | 5.4 ms |

- **Asıl yük mesh_bvh:** hücre sayısı × log(üçgen) ile ölçekleniyor. 352 k üçgende ince
  ızgarada kare başına 1.2 s.
- **mesh_sdf üçgen sayısından bağımsız:** OBB kadar ucuz (~5 ms / 885 k hücre). Pişirme
  0.5–1 s, asenkron ve bir kez (katı cisim).
- ★ **Ama SDF kaba:** 64³ en yakın örnek, 2.7 m'lik bir cisimde ~5.5 cm hücre demek
  (simülasyon h'si 1.25 cm). Ucuz ama alt-voksel temas için yetersiz.

**Sıra buna göre:**
1. **P1 (tek derleme, hemen):** φ'yi SDF'ten ÜÇ DOĞRUSAL ve analitik şekillerden CPU'da
   damgala (ölçüldü: ~5 ms) + S3/S4. Fizik doğru olur, maliyet bugünkü SDF yolu kadar.
2. **P2 — GPU SDF pişirme:** çözünürlük simülasyonun h'sine bağlı (SDF hücresi ≤ h/2).
   Büyük cisimde doğruluk bunu ister, CPU pişirmesi o çözünürlükte pahalı. φ'nin GPU'da
   okunması da 5 ms'yi ve yüklemeyi kaldırır.
3. **P3 — deforme mesh (iskeletli karakter): GPU BVH refit + en yakın üçgen.** Bugünkü
   mesh_bvh yolunun (1.2 s) yerine geçer. DEM'in GPU BVH taraması ortak kullanılabilir;
   kurulum CPU'dan GPU refit'e taşınır.

**FLIP "rampası" (derlemesiz ölçüldü, exe 22:27):** artık rampa değil, bir düzlük. 3 s'lik
dinlenik tankta hız FLIP 0.97'de ~0.06, 0.9'da ~0.035, 0.5 ve 0'da ~0.03 m/s'de sabit.
Yoğunluk düzeltmesi kapalıyken 0.053. Saf PIC'te bile ~0.03'lük bir taban kalıyor: sınırlı
bir gürültü düzeyi (4 parçacık/hücre, h = 5 cm, ~0.6 mm/kare). Düşük öncelik.

## Alt-voksel temas: nasıl yazıldı (2026-11-02 gece, derlenmedi)

- **Ortak çekirdek:** `sim_collider_bvh_query.comp` + `ColliderGpuQuery`. Mevcut 32 baytlık
  düğüm biçimiyle düz BVH (`buildMatterGrainColliderBvh`; dejenere üçgenler önce atılıyor),
  mesh sürümü başına bir kez yükleniyor. Sorgu noktaları q = M·(idx + taban + 0.5): ızgara
  kökeni, voksel ve dünya→yerel dönüşüm tek 3×4 matriste. Mod 0 işaretsiz (kabuk), mod 1
  işaretli (pişirme; CPU ile aynı 3 ışın oylaması).
- **P1:** `grid.solid_phi` voxelizer'da yazılıyor. Üç doğrusal SDF örnekleyici, pişirmenin
  kendi örnek noktalarını kullanıyor (hücre merkezleri; eski t·(n−1) yarım hücre
  kayıktı). Ölçekli nesnede dünya uzaklığı d / |S⁻¹n| ile.
- **P2:** pişirme işi publish tik'inde GPU'da, başarısızsa CPU işçisinde. Auto
  çözünürlük eksen başına; ince bir plaka uzun kenarının adımıyla pişmez.
- **P3:** mesh_bvh kabuğu her kare GPU'da (1.2 s'lik CPU döngüsü yalnız geri dönüş).
- **S3 / S4:** MPM şeridinde parçacık teması ve derin maske (kernel başlığı neden ve nasıl
  olduğunu anlatıyor). Yapışma bandı söküldü.
- **Eksik:** deforme mesh'in GPU refit'i, madde–collider çift birleştirmesi, DEM, sıvı. CPU
  referansı S4'ü uygulamıyor (ortak hücre maskesi).

## Canlı (2026-11-02, exe 23:32): S4 neden yanlış

MPM'de çarpışmayı taşıyan ızgaradır. Parçacık başına temas (S3) gerilme yaymaz: banda
giren parçacığın yaklaşma hızını keser, üstündeki parçacıklar düşmeye devam eder ve gövde
bandın kalınlığına (h/2) yığılır. S4 kısmen örtülü hücreleri ızgaradan çıkarınca, iki
hücre kalınlığındaki plakada kayan noktada HİÇBİR hücre "derin" kalmadı ve ızgara desteği
yok oldu. Jöle 8 cm'den 1.15 cm'ye ezildi; yırtılma yok, genişlik aynı.

Doğrusu standart MPM grid-SDF collision: temas ızgara yüzünde, yüzün φ'si ve ∇φ ile.
Izgara taşır, yüzey hücreden ince olur.

## Izgara yüzünde φ teması (2026-11-03, yazıldı, derlenmedi)

`sim_matter_mpm_boundary` artık φ'li collider'larda standart grid-SDF collision yapıyor:
- yüz başına φ_yüz ve ∇φ; φ_yüz < 0 ise yüzün tam hızı (komşu yüzlerden) katıya göre
  ayrıştırılıyor;
- yaklaşma kesiliyor, Coulomb, ayrılırken yapışma.

Destek ızgarada, eşik φ'nin kendisi, yani hücreden ince. S3 yalnız güvenlik itmesi. S4
kaldı ama zararsız, çünkü destek artık maskeden gelmiyor.

## Kurtarma ışınlaması ve ince kabuk (2026-11-03, yazıldı, derlenmedi)

- **Katıdan kurtarma MPM'i ezer.** Maske hücreyi merkezine göre katı sayar. Eğri ya da
  hücre yüzüne denk gelmeyen bir yüzeyde dinlenen parçacık "katı" hücrede kalır ve her
  kare ışınlanıp ¼ hıza iner. R = 20 m kubbede jöle 10 → 0.6 cm. Bu, düzlemde hiç görünmez:
  yüzey hücre yüzüne denk geliyor.
  - Kural: φ'li hücredeki MPM parçacığı ışınlanmaz; yüzey itmesi alır
    (`applyMpmParticleContact`, kare başına da).
- **Sırt.** Kalınlığı ~2h'den ince bir kabukta φ_yüz < 0 olan tek yüz orta düzlemdir ve
  orada ∇φ = 0. Bu yüz artık akışı keser.
- **Kapandı (yazıldı):** pişmiş SDF ve kabuk yüzeyi ~3 cm aşağıdaydı. Kökü
  `Matrix4x4::determinant` (işaret iki kez): eğik OBB'nin tersi yanlış ölçekliydi.
- **Yapışma bandı:** 0 ≤ φ_yüz < h'de yalnızca ayrılan yüz tutulur; içeride kalan
  yapışma, plaka yarım yüz uzaklaşınca bırakıyordu.

## Ölçüm

`scripts/test/rt_test_mpm_collider_ipc.py`: sakız plakada oturur, plaka 0.1 m/s ile
1 s yükselir. Kapılar: `gum_rides` (merkez, yükselmenin ≥ %70'i kadar çıkar),
`gum_keeps_height` (yükseklik ≥ %60), `gum_intact`, `gum_on_plate`, `finite`.
Eski exe: `gum_rides` FAIL (0.063 m / 0.10 m), blok tek katmana ezildi.
Yeni exe (16:51): 5/5; sakız 0.102 m yükseldi, yükseklik 0.090 → 0.072 m, kopuk yok.
Adım 3 kolları: `slide` (plaka +x 0.1 m/s, μ 0.8 → sakız ≥ %60 izler), `slip` (μ 0 →
≤ %25), `stretch` (iki plaka, yapışma 1e6 Pa, üst plaka kalkar → yükseklik ≥ %30 artar ve
iki yüze de değer).
