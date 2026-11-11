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
- **Yapışma = kohezyon (2026-11-08, CANLI — kullanıcı sahnesinde düzeldi):** yapışma yalnız normali
  tutuyordu, sürtünme yalnız yaklaşan yüze uygulanıyordu. Kürenin altında her yüz
  ayrılır → yapışkan yüzey **sürtünmesiz bir ray** oldu: yan taraftaki kar yüzey boyunca
  serbest kaydı, yapışma onu alt yüzde bükerek teğet doğrultuda ~√(2gR) hızla yana
  fırlattı (kullanıcı sahnesi: μ 20, yapışma 1 MPa, h 0.1 m, kare ~110). Yeni yasa
  τ ≤ yapışma + μ·basınç: yapışma bütçesi gerilme ve kayma arasında paylaşılır
  (hız biriminde hold = yapışma·dt/(ρ·h); çekmede kayma bütçesi hold − min(v_n, hold),
  baskıda μ·|v_n| + hold); bantta (dışarıda) yaklaşan yüz normalini korur, yalnız
  kayması tutulur. Yapışmasız collider'da davranış değişmedi.

## Ölçüm

`scripts/test/rt_test_mpm_collider_ipc.py`: sakız plakada oturur, plaka 0.1 m/s ile
1 s yükselir. Kapılar: `gum_rides` (merkez, yükselmenin ≥ %70'i kadar çıkar),
`gum_keeps_height` (yükseklik ≥ %60), `gum_intact`, `gum_on_plate`, `finite`.
Eski exe: `gum_rides` FAIL (0.063 m / 0.10 m), blok tek katmana ezildi.
Yeni exe (16:51): 5/5; sakız 0.102 m yükseldi, yükseklik 0.090 → 0.072 m, kopuk yok.
Adım 3 kolları: `slide` (plaka +x 0.1 m/s, μ 0.8 → sakız ≥ %60 izler), `slip` (μ 0 →
≤ %25), `stretch` (iki plaka, yapışma 1e6 Pa, üst plaka kalkar → yükseklik ≥ %30 artar ve
iki yüze de değer).

## Sıvı şeridi: yükselen collider sıvıyı taşımıyor (2026-11-08, ÖLÇÜLDÜ; ayrılan duvar YAZILDI, derlenmedi)

Kullanıcı sahnesi (dem1.rtp, h 0.1 m, küre R 1 m, μ 20, yapışma 1 MPa), çikolata
(ν 0.004 m²/s, duvar kayması 0): küre 84→108. karelerde −1.29'dan 1.13 m'ye ~1.25 m/s ile
çıkarken sıvının en üst noktası hep kürenin ALTINDA (ör. 96. kare küre altı −1.00, sıvı
max_y −0.90; 108. kare küre altı 0.13, max_y −1.14). Kaldırılan kütle ≈ 0.

Kod sırası (`MatterGpuStep.inl`, sıvı şeridi): P2G → `sim_fluid_zero_solid_faces` (katı
yüzü = **0**) → FLIP tabanı → viskozite → basınç (kapalı yüz = katı hızı) → temas →
yine **sıfırlama** → G2P. Yani G2P hareketli kürenin yüzünü durağan duvar olarak okur;
küre ancak alttaki açık yüzün emmesiyle yarım hızda çeker, bir kare sonra arada boş hücre
açılır ve bağ kopar. Ayrıca `contact_adhesion_pa`/`contact_friction` sıvı şeridinde HİÇ
okunmuyor (adım 4 açıktı). Basınç çözümü ise ayrılmaya izin vermiyor (sınırsız çekme):
yalnız sıfırlamayı kaldırmak her sıvıyı (su dahil) kürelere yapıştırır.

Plan: katı yüzü öngörülen (P2G) hıza göre karar verir — yaklaşan/tutulan yüz = katı hızı,
ayrılan yüz yapışma bütçesini aşarsa (ρ·h·v_n/dt > yapışma) kendi hızını korur; basınç
çekirdekleri kapalı yüzde sv yerine bu öngörülen yüz değerini okur; G2P öncesi sıfırlama
kalkar. Basınç çekirdekleri tüm sıvı domain'leriyle ortak → bağımsız sıvı yolu da aynı
duvar çekirdeğine geçmeli.

Yan bulgu: küre tabana inerken (alt boşluk 9 cm < h) çikolata 7 m/s ile tavana (2.76 m)
sıçradı; ν 0.004 bu çözünürlükte etkisiz (ν·dt/h² ≈ 0.017/kare, viskoz uzunluk ölçeği
alt-voksel).

### Bal/çikolata MPM mi sıvı mı? — ölçüm ve fizik (2026-11-08)

Aynı sahne, aynı tohum (4×1.2×4 m, 4 ppc, 87k parsel). Kaldırma ölçüsü = parsellerin y p95'i
(tohum ≈ −1.5 m), küre 1.16 m'de durduktan sonra (112–128. kare):

| Koşu | p95_y | Not |
|---|---|---|
| Sıvı şeridi, Chocolate, yapışma 1 MPa | −1.62 | yapışma okunmuyor, G2P küreyi durağan görüyor |
| MPM (elastik yasa), bal: η 10 Pa·s KV, E 77 kPa, ν 0.49, τ 1e-4 s; yapışma 1 MPa | **+0.31** | kütlenin >%5'i kürede asılı, akmıyor |
| Aynı MPM bal, yapışma 0 | −1.55 | yalnız viskozite: kaldırma görünmüyor |

Fizik: yükselen gövdenin viskoz sıvıdan çektiği film Landau–Levich (viskoz–yerçekimi)
rejiminde h ≈ √(μU/ρg) = √(10·1.25/(1420·9.81)) ≈ 3 cm → kütlenin ~%2'si, sonra süzülür.
h = 10 cm'de bu film alt-voksel; hiçbir çözücü göstermez. 1 MPa yapışma sıvı için fiziksel
değil (ıslatma yapışması ~0.1 J/m² iş, gerilme değil); MPM'deki büyük kaldırma bu sayıdan.
Çikolata: Casson akma ~10 Pa → ρgH < τ0 ancak H < 1 mm'de tutar; bu ölçekte ~3 Pa·s Newton
sıvısı gibi davranır. Endüstri: Newton viskoz sıvılar (su, yağ, bal, erimiş çikolata) FLIP +
örtük viskozite (Batty–Bridson; bal kıvrılması FLIP gösterisidir); akma gerilmeli/viskoelastik
olanlar (diş macunu, hamur, sakız, çamur) MPM. **Karar önerisi:** bal ve çikolata SIVI
şeridinde kalır; eksik olan sıvı duvar teması (yukarıdaki plan). Yapışma sıvıda gerilme
sınırı olarak değil, ayrılmayı geciktiren bir ıslatma terimi olarak modellenmeli.

Yan bulgu (MPM bal koşuları): dinlenik karede bile live_max_speed 15–45 m/s (ortalama
0.2–0.7 m/s) — birkaç kaçak parsel; küre tabana inerken (boşluk 9 cm < h) tavana sıçrama
iki şeritte de var. Ayrı iş.

### Ayrılan sıvı duvarı — yazıldı (2026-11-08)

`sim_matter_liquid_wall.glsl` (yoğun + seyrek). Bridson'un eşitsizlik sınırının
ön-karar hali: collider yüzünde r = s·(u − v_s) (s = katı alt hücreyse +1), hold =
yapışma·dt/(ρ·h). r ≤ hold → kapalı (weight 0, v_s); r > hold → AÇIK (weight 1,
u = v_s + s·(r − hold)): varyasyonel matris açık yüzü sıvı-olmayan hücreye karşı p = 0
hayalet sayar, yani katının bıraktığı boşluk havadır. Collider yüzleri host'un ikili
geçişinden hep 0 ağırlıkla gelir, her alt adım 0/1 yazmak kesin ve geri yükleme istemez.
mod 0 (P2G sonrası, G2P öncesi): alan duvarları 0, iki tarafı katı yüz katı hızı, collider
yüzü dokunulmaz (FLIP tabanı sıvının kendi hızını tutar, duvar itkisi post − pre ile gelir).
mod 1 basınç içinde, ağırlık yüklemesinden sonra (seyrek sayfalar her çağrıda yükleniyor).
Host: `enforceSolidBoundaries` (liquid_collider_wall) ve `enforceGridSolidFaceBoundaries(…,
keep_collider_faces)` yalnız iç yüzleri katı hızına çeker. Yoğunluk projeksiyonu duvarı
kapatır (konum düzeltmesi, momentum değil). Açık: CPU-yalnız basınç ikizi; Coulomb
sürtünmesi sıvıda yok (teğet tutunma viskozitenin kaymazlık koşulundan).

### Canlı (exe 14:00, 2026-11-08): duvar ÇALIŞIYOR ama ikili vokselde bağ kopuyor

Kullanıcı sahnesi, eşdeğer tohum (85k), yapışma 0 ve 1 MPa: durum hash'leri FARKLI (duvar
devrede), ama 1 MPa'da da kaldırılan kütle ≈ 0 (p95 farkı < 2 cm). Kök MPM'de 2026-11-02'de
görülenle aynı: collider ikili voksel (h 0.1 m), küre 5 cm/kare yükseliyor. Tutulan yüz
sıvıyı bir alt adım çeker, ama boşalan hücre kare boyunca hâlâ katı (advect çözümü
parçacığı geri iter); hücre boşalınca araya bir HAVA hücresi girer ve yüz artık katıya
değil havaya bakar → yapışma tutamaz. Ayrıca kürenin yan yüzlerinde (x/z) dikey hareket
yüz normaline dik: yalnız viskozitenin kaymazlık koşulu taşır, o da h 0.1'de zayıf
(ν·dt/h² ≈ 0.017/kare). Çözüm MPM'deki gibi: φ tabanlı alt-voksel temas + 0 ≤ φ < h
yapışma bandı (sıvı şeridi için). Yeni not açılmadı; bu bölüm plan.

Viskozite ölçek notu: çikolata ν 0.004, bal 0.007 m²/s fiziksel. 1 m / 1 m/s'de Re ≈ 250 /
140 → bu ölçekte "koyu" görünüm cm-altı özelliklerden (ince perde, iplik, kaplama) gelir;
h 0.1'de çözülmez. Viskozite uygulanıyor (viscosity_sparse_used true).

### Sıvı yapışma bandı — CANLI (2026-11-08; test 7/7, smoke quick 13/13; > 1 voksel/kare collider bandı geçer)

`sim_matter_liquid_wall.glsl` mod 3/2. Basınç içinde, mod 1'den önce: hava hücresi (maske
0), solid_phi < h, sıvıya komşu; ∇φ'nin tersinde yarım hücre ötedeki hücre gerçek collider
(maske −1) ve yapışması > 0 ise hücre −2 işaretlenir, svx/svy/svz ve yapışma o hücreden
kopyalanır. Mod 1 bant hücresini katı sayar; yalnız ayrılan yüzü tutar (r ≤ 0 → açık).
Her çıkışta (RAII) mod 2 geri alır ve karma şeritte `pressure_statics_uploaded = false`
(bandın yazdığı ağırlıklar ve svx yeniden yüklenir). `gridHasGluedCollider` yapışkan
collider yoksa bandı hiç çalıştırmaz. Gerçek alt-voksel φ yüz teması (MPM'deki gibi
φ_yüz ile karar) hâlâ yok: bant kaba ama ikili voksel boşluğunu kapatıyor.

## Yapışma bir ÇİFT özelliğidir: yüzey tavanı × malzeme kapasitesi (2026-11-11, TASARIM — yazılmadı)

**Gözlem (kullanıcı):** su gibi maddelerde yapışma fazla baskın. **Kök:** yapışma yalnız yüzeyin değeri —
collider `contact_adhesion_pa`, domain duvarı `wall_adhesion_pa`. Tüketiciler aynı sayıyı her malzemeye
uyguluyor: sıvı duvarı (`sim_matter_liquid_wall`: tut ⇔ r ≤ yapışma·dt/(ρh)), domain duvar ızgarası
(`sim_domain_wall_grid`), akıtma kuyruğu (`wall_adhesion_acceleration`), MPM sınırı (`sim_matter_mpm_boundary`),
DEM duvarı (`sim_matter_grain.glsl` adhesion_limit) ve CPU ikizleri (`APICFluidSolver.cpp` glue). Su, bal,
çikolata aynı Pa ile aynı tutuluyor; tek fark ρ.

**Fizik:** sıvının duvara tutunması iki kaynaktan gelir, ikisi de MALZEMENİN:
1. **Islatma (kılcal):** ~γ(1+cosθ)/h. Su γ 0.072 N/m, h 2.78 cm → **~5 Pa** (θ = 0 en kötü durum). Yani 1 kPa
   collider değeriyle su tutkal gibi; fiziksel olarak su bırakır.
2. **Viskoz film (Stefan yapışması):** ayrılmaya karşı gerilme ~ μ·r/h, hıza bağlı. Bal (μ ~10) su (1e-3)'nun
   10⁴ katı; çikolata/bal "yapışkanlığı" budur ve voksel altı film çözülmediği için model gerekir.

**Kural (tek fonksiyon, bütün tüketiciler):**
  A_eff = min(A_yüzey, A_malzeme(r))
  sıvı:  A_malzeme = 2γ/h + μ·|r|/h       (γ: `liquid_surface_tension_n_m`, μ = ρ·ν, maddeden)
  tut ⇔ ρ·h·r/dt ≤ A_eff;  bırakınca fazla hız r − A_eff·dt/(ρh)
Yüzey değeri ANLAMINI korur ("bu yüzey en fazla bu kadar tutar"), malzeme kapasiteyi verir → alan adı
değişmez, eski projeler aynı okunur. Sonuç: su kendiliğinden bırakır, bal/çikolata viskozitesiyle tutar.
Domain duvarı aynı fonksiyonu çağırır; ayrı iş değil.

**Kapsam (ilk parti yalnız SIVI):** MPM (kar/hamur) ve DEM şimdilik aynı kalır — elastik sakız testi 1e6 Pa
yapışmayla geçiyor ve malzeme kapasitesi (çekme dayanımı/kohezyon) her madde için tanımlı değil; ayrı karar.

**Dokunulacaklar:** `sim_matter_liquid_wall.glsl` (push +γ, +μ; kendi kernelim), `sim_domain_wall_grid.glsl` +
`DomainWallGpuConstants.h` (header → geniş derleme), `FluidGpuTransferStages.inl` (kuyruk ivmesi), CPU
`APICFluidSolver.cpp`. Kabul: `rt_test_liquid_wall_ipc.py` — su yapışık yüzeyden bırakmalı (yeni kol), bal
`glued_follows` geçmeye devam etmeli; `rt_matter_smoke.py --quick`.
