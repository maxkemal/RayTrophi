# NEXT BUILD CHECKS

## ★ TAM REBUILD — dinlenik kar yüzeyindeki kıpırdama teşhisi — 2026-11-04 gece (YAZILDI, derlenmedi)

**Durum:** ajanın başlattığı tam Rebuild yarıda durduruldu. Rebuild önce temizlediği için
`x64\Release` nesne dosyaları ve exe eksik ya da eski olabilir: **tam Rebuild** al.
`RtApi.h`'de `FluidParticleProbe` büyüdü (yapı düzeni değişti).

**Sorun (canlı, kullanıcının sahnesinin kopyası):**
- Havadan düşüp kırılan kar 150 kare sonra %99.6 dinleniyor, ama ~300 parçacık hiç durmuyor.
- Ortalama hız 0.0007 m/s, yani bu ~300 parçacık ~0.2 m/s civarında. Domain uykusu "ya
  hep ya hiç" olduğu için kare ~130 ms'de kalıyor.
- 11 000 parçacık (%12.6) kopmuş.
- Hipotez (doğrulanmadı): kütlenin üstündeki tek tek tanelerin basıncı ~0. Settle yalnız
  basınç altındakini sönümlüyor; kar FLIP karışımı kullandığı için seyrek bölgedeki tane
  ızgara gürültüsüyle sarsılıyor.

**Yazılan:** `fluid.particle_extremes` artık `fastest` listesi veriyor. Her sonda: `speed`,
`granular_sleeping`, `granular_detached`, `granular_damage`, `granular_pressure`,
`support_accel`.

**Derledikten sonra (ajan koşturur):** teşhis sondası en hızlı 32 parçacığı basar.
- `granular_pressure` ~0, `support_accel` < 2.45 (destekli) ve `granular_sleeping` 0 ise
  hipotez doğru. Düzeltme: settle'ın sönümü desteğe bağlansın, basınca değil.
- `support_accel` büyükse tane gerçekten havada (dökülme); doğal.

## MPM domain uykusu + P2G gradyan atlama + kümelenmiş P2G — 2026-11-04 (AJAN DERLEDİ, CANLI)

Kullanıcı onayıyla ajan derledi: tam Rebuild ×2, artımlı ×5; shader'lar glslc ile
`source/shaders` + `x64/Release/shaders`. Ayrıntı ve ölçümler:
[MPM_UYUYAN_KAROLAR.md](MPM_UYUYAN_KAROLAR.md).

**Canlı:**
- `rt_test_mpm_sleep_ipc.py`: dinlenik 83 k kar **123 → 16.7 ms/kare (7.4×)**, şekil farkı
  0.06 mm; ince katman 8.4×; havadaki blok hiç uyumadı.
- Uyku kapalıyken P2G 84 → 49 ms/kare (sıvı yokken kütle gradyanı atlanıyor).
- Kısmi uyku (hücre bazlı) ölçüldü: %5–7 YAVAŞ, söküldü.
- Uyuyan karede istatistik bloğu boş yayımlanıyordu (`granular_sleeping` 0, plastik J 1):
  sayaçlar artık parçacıklardan yeniden sayılıyor (`snow_support` `floor_sleeps` bunu yakaladı).
- `rt_matter_smoke --quick` **16/16** (`snow_support`, `mpm_sleep` dahil).

**Senin kontrolün için (elle):**
1. Kar sahnende Settle on Seed açık, kar dinlenince `fluid.step_stats` `gpu_status`
   `matter_asleep` olmalı ve kare hızlanmalı. Bir collider'ı hareket ettirince ya da
   kartopu atınca (parçacık sayısı değişir) hemen uyanmalı.
2. ★ **En sinsi:** uyuyan bir yığının yanına yavaşça yaklaşan bir collider. Hareketli
   collider bütün domain'i uyandırır, yani doğru. Ama collider'ı DURDURUP yığına yaslarsan
   yığın 3 kare sonra yeniden uyur. Collider o anda kara itme uyguluyorsa bu yanlıştır.
   Bağlam özeti collider setini (imzayı) görür, temas kuvvetini görmez.
3. Kuvvet alanı (rüzgâr vb.) varken domain hiç uyumamalı.

## ★ TAM REBUILD — MPM settle desteğe bakıyor (A aşaması) — 2026-11-04 (CANLI)

**Canlı sonuç (tam Rebuild sonrası):**
- `snow_support` **5/5**:
  - havadaki blok 0.25 s'de 0.343 m düştü (serbest düşüş 0.307), hasar 0;
  - zemindeki kar hızı 6e-5 m/s, %93'ü uyuyor;
  - J_P büyümesi 7e-5, mandal etkisi yok.
- `seed_settle` 6/6. `rt_matter_smoke --quick` **15/15**.
- Kullanıcı kendi kar sahnesinde doğruladı: kar artık asılı kalmıyor.
- Not: hava düşüşünün serbest düşüşten fazla çıkması büyük olasılıkla ölçümün tohum anındaki
  (oturtma öncesi) üst sınırdan başlaması; doğrulanmadı, kapı alt sınır olduğu için sonucu
  etkilemiyor.


**★ Tam Rebuild gerekli:** `FluidParticles` iki yeni dizi aldı (`granular_support_prev_vg`,
`granular_support_accel`), `Granular::GpuBuffers` iki tampon, `SettleConstants` 32 → 48
bayt. Artımlı derleme eski `.obj` bırakırsa alakasız bir yerde erişim ihlali görürsün.
**Derlenecek:** C++ ve iki shader: `sim_fluid_granular_settle` ve `sim_matter_settle`
(ikisi de `sim_fluid_granular_settle.glsl`'yi içeriyor). Madde denetimleri temiz.

**Kök (canlı ölçüm, kullanıcının sahnesi):**
- Havada tohumlanan kar, oturtma açıkken üstten asılı kaldı: üst sınır 24 kare boyunca
  2.18'de, alt kısım koptu, hasar > 0.5 olan 18 850 parçacık. Oturtma kapalıyken blok tek
  parça serbest düştü.
- Settle, basıncı olan her yavaş parçacığı sönümleyip uyutuyordu. Alt adım başına
  yerçekimi ~0.025 m/s, uyku eşiği 0.03, yani her alt adımda sıfırlanıyordu.
- Basınç destek değildir.

**Düzeltme:**
- Sönüm ve uyku yalnızca desteklenen parçacıkta: yerçekimi yönündeki net ivmenin 0.2 s
  ortalaması 0.25·g altında.
- İvme sönümden önceki hızla ölçülür; titreşim ortalamada söner.
- Yeni parçacık desteksiz başlar.
- Uyanma kendiliğinden (destek giderse ~0.06 s).
- GPU (iki yol aynı fonksiyondan) ve CPU ikizi.
- B aşaması (gerçek iş atlayan uyku) için TASLAK: [MPM_UYUYAN_KAROLAR.md](MPM_UYUYAN_KAROLAR.md).

**Sırayla** (boş ve duraklatılmış sahnede):
1. `python scripts/test/rt_test_snow_support_ipc.py`: **5/5** hedef (yeni test; eşikler
   canlıda ayarlanmadı).
   - `air_falls` FAIL: settle hâlâ havadaki parçacığı tutuyor. `granular_support_accel`
     yüklenmiyor olabilir (`uploadState` boyut kontrolü) ya da yerçekimi 0 geliyor.
   - ★ **En sinsi:** `floor_sleeps` FAIL ama `floor_rests` PASS. Kar dinleniyor ama hiç
     uyumuyor demektir: 0.25·g toleransı titreşime göre dar. Ortalamanın değeri tolerans
     ayarı için kritik.
   - `floor_no_ratchet` FAIL: titreşim sönümü kayboldu ve J_P mandal gibi büyüyor
     (eski gerileme, `SettleParams::damping_max_speed`).
2. `python scripts/test/rt_test_seed_settle_ipc.py`: 6/6 kalmalı. Oturtma zemindeki tohumda
   biraz daha yavaş yakınsayabilir; destek 0.2 s içinde öğreniliyor.
3. `python scripts/test/rt_matter_smoke.py --quick` (arka planda): 15/15
   (`snow_support` eklendi).
4. Elle: senin kar sahnen, Settle on Seed açık. Kar tek parça düşmeli.

## SDF pişirme payı + kayma kolu testi — 2026-11-04 (CANLI, tam Rebuild sonrası)

**Canlı sonuç:**
- `collider_phi` 12/12.
- `mpm_collider` **8/8**:
  - germe 7.5 → 16.7 cm, iki plakaya tutunuyor (üst boşluk 1.7 mm);
  - yükseliş 9.7 cm;
  - kayma 0.100 m.
- `rt_matter_smoke --quick` **14/14**; kayma kolu tam pakette de 0.100 m.
- ★ İlk derlemede `substance.remove` erişim ihlali verdi. Kod hatası DEĞİLDİ: artımlı derleme
  `RtApiSubstance.obj` ve dört büyük `.obj`'yi `ParticleColliderDesc` düzen değişikliğinden
  sonra yeniden derlememişti. Tam Rebuild çözdü. Yaygın bir struct'ın düzeni değişince
  listenin başına "tam Rebuild" yazılır.


**Derlenecek:** yalnız C++ (`scene_data.h`, `ParticleSimulation.h/.cpp`). Shader değişmedi.
Madde denetimleri temiz.

**Sondalar (canlı, önceki exe):**
1. **Yapışık germe kopuşu = SDF kutusu plakanın dışında çok dar.**
   - Pişirme her eksende kendi boyunun %15'i kadar pay ekliyordu. 5 cm plakada bu y'de
     7.5 mm; h 2.5 cm.
   - `fluid.sample_solid_phi`: plakanın altında φ 7.5 mm sonra "uzak"a (2h) atlıyor. Sıfır
     geçişi gerçek alt yüzün 0–1 cm üstünde; plaka bir hücre (6 kare) ilerledikçe döngüsel.
   - Yapışma bandı (φ < h) bu plakada pratikte yok. Sakız ~1.9 kat uzayıp hücre hizası
     kötü düştüğü karede bıraktı. `step_stats` kopuşun çevresinde hiçbir sayaçta değişmedi.
   - **Düzeltme:**
     - pay eşyönlü ve en uzun kenardan: 0.15 · max(boyut); Auto'da en az 3 voxel;
     - pişen mesh boyu artık collider'da `sdf_mesh_size` olarak saklanıyor;
     - sekiz okuyucu `sdf_extents / 1.3` varsayımını bıraktı.
2. **Kayma kolu kararsızlığı = test pişirmeyi beklemiyordu (büyük olasılık).**
   - İkiye bölme: `elastic` + `mpm_collider` geçti; hızlı yarı + `mpm_collider` geçti;
     yavaş yarı + `mpm_collider` geçti. Yalnız tam paket (iki kez) 0.00005 m verdi.
   - Pişirme süresi bu oturumda 97–342 ms arası değişti. Yük altında geç pişen plakadan
     sakız zemine düşer ve kaymaz.
   - **Test düzeltildi:** `plate()` artık `sdf_ready`'yi bekliyor (iki kopya da güncel).

**Sırayla:**
1. `python scripts/test/rt_test_collider_phi_ipc.py`: 12/12 kalmalı. Küre payı değişmedi
   (0.15 · 1.8 m).
2. `python scripts/test/rt_test_mpm_collider_ipc.py`: **8/8** hedef.
   - `gum_stretches` FAIL ise germe kolunu kur ve her karede plaka alt yüzünün 2 cm altından
     1 cm içine kadar eksende `fluid.sample_solid_phi` oku. φ artık 2 cm boyunca doğrusal
     (−0.0 … +0.02) olmalı.
     - Doğrusalsa kopuşun kökü başka.
     - Hâlâ 7.5 mm'de 0.05'e atlıyorsa pişirme eski payla kalmış (`sdf_dims` y'de büyümeli:
       0.0575 / 0.0125 ≈ 5 → ~11).
   - ★ **En sinsi:** `rise` ve `slide` sayıları değişirse, ölçek (`scale = current /
     sdf_mesh_size`) yanlış okunuyor demektir. 1.0 olmalı.
3. `python scripts/test/rt_matter_smoke.py --quick` (arka planda): **14/14**, kayma kolu
   tam pakette de 0.100.

## Proje açılışında zaman çizelgesi yarışı — 2026-11-03 gece (CANLI: saç projesi çökmeden açıldı)

- **Belirti:** saç sistemli proje açılırken read access violation. Yığın: `SDL_main` →
  `std::map` iteratörü `++`. Yığındaki `json` / `VectorKey` adları bağlayıcının özdeş
  fonksiyon birleştirmesi (COMDAT folding); gerçek döngü `scene.timeline.tracks`
  (`Main.cpp`).
- **Kök:** proje yüklemesi kapları iş parçacığında yeniden kurarken ana döngü her kare
  zaman çizelgesi izlerini, model bağlamlarını ve gaz/VDB listelerini geziyordu. Rüzgâr,
  materyal ve `animationDataList` döngüleri zaten `sceneLoadActive` ile korunuyordu; bu
  dördü korumasızdı.
- **Düzeltme:** `sceneLoadActive` yukarı taşındı. Yükleme sürerken bu dört döngüye hiç
  girilmez ve önbelleğe alınmış bayraklar kullanılır.
- **Kontrol:** aynı saç projesini 3–4 kez aç/kapat; çökme olmamalı. Açıldıktan sonra
  zaman çizelgesi anahtarları çalmalı (önbellek yükleme bitince tazelenir).
  - Yine çökerse: yükleyici `scene_loading` bayrağını kabı değiştirdikten SONRA mı
    kaldırıyor? Ona bak.

## Canlı sonuç — 2026-11-03 gece (determinant + yapışma bandı derlendi)

- `collider_phi` **12/12**:
  - SDF sıfır geçişi 0.0986 (tepe 0.100);
  - SDF boşluğu −4.9 mm, kabuk −4.6 mm, analitik küre −4.9 mm: üç yol aynı yerde;
  - kurtarma sayacı 0.
- `mpm_collider` **7/8**, üç koşuda da bit düzeyinde aynı:
  - kayma kararsızlığı KAPANMADI. Tek başına 3 koşuda 0.1000 m, ama duman paketinde
    (`rt_matter_smoke --quick`, 13/14) 0.00005 m.
    - Aday: önceki testlerden kalan durum (OBB memo, yüzey önbelleği, aynı ad), ya da plaka
      SDF'si pişmeden kaymanın başlaması.
    - Sına: duman paketini yalnız `elastic` + `mpm_collider` ile koştur;
      `collider.get sdf_ready` kayma kolu başında yazdır.
  - `drawn_gum_tears` ilk kez geçti (172 / 1152 koptu).
- `elastic` 11/11.
- **AÇIK: `gum_stretches`.**
  - Sakız üst plakaya yapışık 7.9 → ~15.5 cm uzuyor, 32. karede ucu bırakıp eski boyuna
    (p95 0.197) geri sıçrıyor (ortalama hız 0.49).
  - Kırılma gerinimi 1 ve 5 iken bit düzeyinde aynı: yırtılma değil.
  - Fizik değil: E = 2e5 Pa, Hencky ≈ 0.67, gerilme ~1e5 Pa; yapışma 1e6 Pa, tutma eşiği
    ~95 m/s.
  - Kopuştan önce plaka-sakız boşluğu ~1 mm, yani bant kaybolmuyor.
  - Sonraki aday: kopuş karesinde `fluid.step_stats` (alt adım sayısı, pushed) ve
    `fluid.sample_solid_phi` ile plaka altı. Gerilmiş ince boyunda (necking) ızgara kütlesinin
    seyrelmesi ya da alt adım sayısının değişmesi.

## ★ Matrix4x4::determinant düzeltildi + yapışma bandı — 2026-11-03 akşam (YAZILDI, derlenmedi)

**Derlenecek:** C++ (`src/Math/Matrix4x4.cpp` — GENİŞ ETKİ, aşağıya bak) ve
`sim_matter_mpm_boundary` shader'ı. Yeni `.cpp` yok. Madde denetimleri temiz.

**Önceki partinin canlı sonucu (kurtarma + sırt + φ sondası):**
- `collider_phi` 9/12:
  - küre 1 → 8.2 cm, SDF 8.0 cm, kurtarma sayacı her karede 0;
  - FAIL `sdf_phi_surface`, `sdf_matches_analytic`, `shell_on_surface`.
- `mpm_collider` 4/8 → 6/8: yükseklik, sürtünmeyle taşıma 0.100 / 0.100 m, plaka üstünde;
  FAIL `gum_stretches`, `drawn_gum_tears`.
- `elastic` 11/11.

**Kökler (`fluid.sample_solid_phi` ile ölçüldü):**
1. **★ `Matrix4x4::determinant()` kofaktör işaretini İKİ KEZ uyguluyordu** (2025-12-10'dan
   beri).
   - z ekseni etrafında θ dönmede det = cos 2θ: 10°'de 0.94, 30°'de 0.5, 45°'de 0 (ters
     birim matris dönüyor!), 90°'de −1.
   - `inverse()` adjugate / det olduğu için eğik her çerçeve yanlış ölçekle tersleniyordu.
   - Collider OBB'si ilk üçgenden kurulur; kürede eğik. Ölçüm: SDF ve kabuk φ'si ikisi de
     eğim 1.03, yüzey 3 cm aşağıda; analitik küre 0.2 mm içinde.
   - Küp plakalar neredeyse sağlamdı: eksene hizalı ya da yer değiştirmiş çerçevede hata
     yalnız işaret, kutu da merkezine göre simetrik.
2. **Yapışma yalnız plakanın İÇİNDEKİ yüzlerde etkiliydi.**
   - Plaka kalkıp boşluk yarım yüzü geçince bırakıyordu: sakız ~1 cm izleyip geri düştü.
   - Şimdi 0 ≤ φ_yüz < h bandında, yalnız AYRILAN yüzlerde aynı tutma kuralı. Yapışma ve
     katı hızı yüzeyin öbür yanındaki hücreden okunur. Destek ve sürtünme yine yalnız
     φ_yüz < 0'da.

**Sırayla:**
1. `python scripts/test/rt_test_collider_phi_ipc.py`: **12/12** hedef.
   - `sdf_phi_zero_y` ≈ 0.100 ve kabukta tepede φ ≈ 0 olmalı.
   - Hâlâ ~0.07 ise çağıran yol başka bir ters fonksiyon kullanıyor (`inverse_affine` vb. ara).
   - `shell_on_surface` hâlâ FAIL ise (boşluk ~−2 cm): sırt yüzü orta düzlemde tutuyor, S3
     itmesi 0.12'ye kaldırmıyor.
2. `python scripts/test/rt_test_mpm_collider_ipc.py`: **8/8** hedef.
   - `gum_stretches` FAIL ve `stretch_upper_gap_m` ≈ 0.10: bant yüzleri kütle taşımıyor; bant
     h'den geniş olmalı (1.5h dene).
   - ★ **En sinsi:** `slide` / `slip` değerleri değişirse bant yapışma dışında bir şeye
     dokunuyor demektir. Glue 0 iken bant hiçbir şey yapmamalı.
   - **Kararsız kol (bu exe):** `slide_gum_x_m` 3 koşuda 0.1000 / 0.00005 / 0.1000 çıktı
     (iki uçlu, diğer kollar bit düzeyinde aynı).
     - Aday: OBB çerçevesi ilk üçgenden kurulur. Üçgen sırası koşudan koşuya değişirse
       çerçeve kimi zaman yer değiştirmiş olur; hatalı det −1 ise ters matrisin tamamı
       negatifleniyordu.
     - Determinant düzeltmesinden sonra testi **3 kez** koştur; üçünde de ≈ 0.100 olmalı.
       Değilse kök başka: kayma kolunun SDF pişirme zamanlaması.
3. `python scripts/test/rt_test_elastic_mpm_ipc.py`: 11/11.
4. `python scripts/test/rt_matter_smoke.py --quick`: 14/14 (10 dakikayı geçebilir; arka
   planda koştur).
5. **★ Determinant düzeltmesinin geniş etkisi — elle gözle bak.** `Matrix4x4::inverse`'ü
   137 yer çağırıyor. Eğik her şey (45°'ye yakın olanlar en çok) değişecek:
   - döndürülmüş objede gizmo ve seçim;
   - saç groom'u, rig, sculpt;
   - kinematik collider, kuvvet alanı, gaz/VDB hacmi;
   - terrain eğrisi, scatter.
   Beklenen: daha DOĞRU. Ama eskiyi telafi etmek için yazılmış bir kod varsa şimdi bozulur.
   Örnek ölçüt: 45° dönmüş bir küp collider sıvıyı doğru yerde durdurmalı. Eskiden ters
   birim matris çıkıyordu, yani collider dünya orijininde sanılıyordu.

## MPM için kurtarma ışınlaması kapandı + sırt teması + φ sondası — 2026-11-03 (YAZILDI, derlenmedi)

**Derlenecek:** C++ ve bir shader (`sim_matter_mpm_boundary`, sırt dalı). Yeni `.cpp` yok.
Madde denetimleri, IPC yetki denetimi ve tarif üreticisi temiz.

**Önceki partinin canlı sonucu (bu düzeltmelerden ÖNCE):**
- `collider_phi` 6/8:
  - düzlem tam yüzeyde (0.01 mm) ve 7.3 cm boyunda; S4'ün ezmesi kapandı;
  - FAIL `sdf_matches_analytic` (−42.6 mm) ve `shell_on_surface` (−56.6 mm).
- `mpm_collider` 4/8. `elastic` 11/11. Duman testi 12/14.

**Bulunan kökler (sondayla ölçüldü):**
1. **Katıdan kurtarma MPM parçacıklarını ışınlıyordu.**
   - Maske hücreyi merkezine göre katı sayar. Eğri ya da hücre yüzüne denk gelmeyen bir
     yüzeyde duran parçacık "katı" hücrededir.
   - `recoverParticlesFromSolidCells` onu her kare üstteki hücre merkezine taşıyıp hızını
     ¼'e indiriyordu.
   - Ölçüm: R = 20 m kubbede bile μ 1 ile jöle 10 cm → 0.6 cm; düzlemde 7.3 cm.
     "Dinlenik" hız 0.16–0.22 m/s.
   - ★ Küre kolu boşluk kapısını GEÇİYORDU ama 1 cm'e ezilmişti: sessiz, makul görünen sonuç.
   - Kar "collider'a yapışık" raporunun da kökü.
   - **Düzeltme:** φ taşıyan hücredeki MPM parçacığı (Granular, Elastic, Auto + MPM) ışınlanmaz.
     Aynı karede `applyMpmParticleContact` koşar: φ < 0 ise yüzeye konur, yalnız içe doğru hızı
     kesilir. Bu itme eski granüler zincirde de (saf kar) koşar; orada S3 yoktu. İtilen
     parçacık varsa GPU parçacık tamponu tazelenir.
2. **İnce kabukta sırt.**
   - 2 cm kalınlık, h 2.5 cm: hücre merkezinde φ satır satır +1.75 / −0.75 / −0.75 / +1.75 cm.
   - φ_yüz < 0 olan tek yüz plakanın orta düzlemi; orada ∇φ = 0, kod teması atlıyordu.
     Jöle kabuğun içinden düşüyordu.
   - **Düzeltme:** φ_yüz < 0 ve eğim yok ise yüz katının hızını alır (akış kesilir).
     CPU ikizi de aynı.
3. **AÇIK — pişmiş SDF yüzeyi ~3 cm aşağıda gösteriyor** (jöle p05 0.07, gerçek tepe 0.10).
   - Önbellek değil: zorla yeniden damgalama aynı sonucu verdi.
   - Kafes kuralı değil: pişirme ve örnekleyici hücre merkezinde.
   - Mesh tepesi 0.0999'da (1280 üçgen).
   - İpucu: Auto boyutları 187/186/187. 1.8 m kürede ceil(2.34 / 0.0125) = 188 olmalıydı,
     yani pişirilen üçgenlerin kutusu ~1.785–1.79 m.
   - Ölçmek için yeni `fluid.sample_solid_phi`.

**Yeni IPC:**
- `fluid.sample_solid_phi {domain, points}`: nokta başına hücre, maske, hücre φ'si ve üç
  doğrusal φ.
- `fluid.step_stats` iki yeni alan taşıyor: `recovered_solid_particles`,
  `mpm_surface_pushed_particles`.

**Sırayla:**
1. `python scripts/test/rt_test_collider_phi_ipc.py`: 12 kapı.
   - Yeni kapılar: `sphere_keeps_height` / `sdf_keeps_height` (≥ 6 cm), `no_mpm_teleport`
     (kurtarma sayacı her karede 0), `sdf_phi_surface` (SDF'nin sıfır geçişi tepeye h/4 içinde).
   - `sphere_keeps_height` FAIL ve `recovered_max` > 0: atlama koşulu tutmuyor. Parçacığın
     `constitutive_model`'i Auto ve `auto_is_mpm` yanlış olabilir.
   - `sphere_keeps_height` FAIL ve `recovered_max` = 0: ezilme başka yerden geliyor. Kubbede
     ızgara desteğine bak; düzlem tutuyorsa eğimli n'nin x/z yüzleri.
   - `shell_on_surface` FAIL, boşluk ≈ −2 cm: sırt kesmesi çalışıyor ama jöle orta düzlemde
     oturuyor. S3 onu 0.12'de tutmalıydı; `pushed_max`'a bak.
   - `sdf_phi_surface` FAIL: `sdf_phi_zero_y` kaymanın kendisi. 0.07 civarı ise sorun
     pişirmede ya da örneklemede, ızgarada değil. `phi_column`'u analitik
     `|p − c| − R` ile karşılaştır.
2. `python scripts/test/rt_test_mpm_collider_ipc.py`: önceki 4/8 (yükseklik, sürtünme, germe,
   yırtılma).
   - Plakalar `mesh_sdf`; 1. kök hepsini bozuyordu (¼ hız kesmesi taşımayı öldürür).
   - 3. kök (SDF kayması) açık kaldıkça tam 8/8 beklenmez. `rise_height` > 6 mm ve
     `slide_gum_x_m` > 6 cm olmalı.
3. `python scripts/test/rt_test_elastic_mpm_ipc.py`: 11/11 kalmalı (collider yok).
4. `python scripts/test/rt_matter_smoke.py --quick`: 14/14 hedef.
   - Sıvı testleri (`seed_settle`, `fire_steam`) değişmemeli: sıvı parçacıkları hâlâ
     ışınlanıyor, değişiklik yalnız MPM'de.
5. Elle: snow seed + collider sahnesi. Kar tepeye yapışmadan oturmalı.
   `fluid.step_stats` `recovered_solid_particles` 0 olmalı.

## Izgara yüzünde φ teması + GPU sorgu sağlamlığı — 2026-11-03 (YAZILDI, derlenmedi)

**Derlenecek:** C++ ve iki shader: `sim_matter_mpm_boundary` (yeniden yazıldı; 13 tampon) ve
`sim_matter_mpm_particle_contact` (sadeleşti). Yeni `.cpp` yok. glslc OK; madde
denetimleri ve IPC yetki denetimi PASS.

**İçerik:**
- **Temas ızgarada** (MPM grid-SDF collision), yüz başına:
  - φ_yüz iki hücrenin ortalaması, temas φ_yüz < 0 iken; n = ∇φ (yüz çevresinde merkezi fark);
  - yüzün tam hızı: kendi bileşeni + öteki bileşenlerin 4 komşu yüz ortalaması (anlık
    kopyadan, yarışsız);
  - katıya göre: yaklaşma kesilir, kayma μ·|v_n| kadar azalır; ayrılırken yapışma tutar.
  - φ'siz duvarlar (alan duvarı, kinematik proxy) eski taşı/bırak kuralında. Eski iki
    geçiş (sürtünme + hayalet) söküldü: φ'li içteki boş yüz zaten v = 0 okuyup kayıcı
    plakada sürükleme yapmıyor.
- **Anlık kopya tamponları** `mpm_contact_snapshot_*` (ayrı, çünkü `scratch_vel_*` MPM
  şeridinin FLIP tabanı).
- **S3 yalnız güvenlik itmesi:** φ < 0'daki parçacık yüzeye konur ve içe doğru hızını
  kaybeder. Sürtünme ve yapışma ızgarada.
- **S4 (derin maske) kaldı.** Artık zararsız: ızgara desteği maskeden değil φ'den geliyor.
  Maske yalnız ilerletmenin parçacığı kısmen örtülü hücreye sokmasına izin veriyor.
- **CPU ikizi** (`enforceMpmContact`) aynı φ yüz kuralına çevrildi. CPU'da S4 yok (ortak maske).
- **GPU sorgu sağlamlığı:** ölü tutamak denetimi (`getBufferSize == 0` ya da arka uç
  değişti ise BVH ve çıktı yeniden kurulur). Hata mesajı dispatch / indirme / boşaltma
  diye ayrılıyor ve nokta, üçgen, boyut bilgisini taşıyor. Başarısız BVH önbellekten
  atılıyor.
- **Auto çözünürlük:** ölçek düzeltmesi YAPILMADI, gerek yokmuş. OBB'nin yerel çerçevesi
  ölçeği zaten taşıyor (187³ = 1.3 × 1.8 / 0.0125, doğru). Önceki "düzeltilecek" notu yanlıştı.
- `rt_test_collider_phi_ipc.py`: yeni kapı `plane_keeps_height` (10 cm küp ≥ 6 cm kalmalı).

**Sırayla:**
1. `python scripts/test/rt_test_collider_phi_ipc.py`: **8/8** hedef.
   - `plane_keeps_height` FAIL: ızgara hâlâ desteklemiyor. φ yüklenmiyor olabilir
     (`var_solid_phi`); ya da φ_yüz < 0 eşiği iki hücre kalınlığındaki plakada da tutmuyor.
   - `plane_on_surface` FAIL, boşluk ≈ +h/2: temas φ_yüz < 0'da açılıyor ve yüzey hücre
     merkezinden yarım hücre kayık okunuyor. Eşiği yüzeyin tam yerine taşımak gerekebilir.
   - `sdf_cooked_on_gpu` / `shell_on_gpu` FAIL: `collider_query_error` artık hangi adımın
     düştüğünü söylüyor. "dispatch failed" ise pipeline ya da descriptor; "download" ise
     transfer.
2. `python scripts/test/rt_test_mpm_collider_ipc.py`: **8/8** hedef.
   - ★ **En sinsi:** `frictionless_slips` FAIL (μ 0'da sakız yine sürükleniyor). İçteki boş
     yüz mantığı doğru, ama gövdenin plakaya taşan B-spline kütlesi o yüzlere plakanın
     hızını taşıyor olabilir. `slip_gum_x_m`'e bak.
3. `python scripts/test/rt_test_elastic_mpm_ipc.py`: 11/11 kalmalı.
4. `python scripts/test/rt_matter_smoke.py --quick`: **14/14** hedef.
5. Performans: arşiv kayalarıyla mesh_bvh GPU tablosu yeniden; slate 352 k, h 2.5 cm
   satırı bu kez GPU'da kalmalı.

## Alt-voksel temas — CANLI 2026-11-02 (exe 23:32, .spv 23:20)

**Çalışan:**
- **P3 GPU kabuk sorgusu.** mesh_bvh damga süresi, ilk turdaki tabloyla aynı düzende:

  | Mesh | h | CPU (önce) | GPU |
  |---|---|---|---|
  | kaya 9.7 k | 2.5 cm | 57 ms | **6.7 ms** |
  | kaya 9.7 k | 1.25 cm | 341 ms | **27.6 ms** |
  | slate 352 k | 1.25 cm | 1240 ms | **19 ms** (65×) |
  | slate 352 k | 2.5 cm | 174 ms | 156 ms: GPU `dispatch/download failed`, CPU'ya düştü (AÇIK) |

- **P2 GPU pişirme.** Auto modda küre mesh'i 187³, 90 ms. (Not: Auto, dünya h'sini yerel
  birimlere çevirmiyor. Ölçekli nesnede hücre sayısı ölçekle sapıyor; düzeltilecek.)
- **P1 + S3 analitik küre:** jöle yüzeye −1.2 mm ile oturuyor (eskiden bir hücre mertebesi).
- Elastik test zeminde 11/11.

**Bozuk — kök ölçüldü: S4 (derin maske) tasarımı yanlış.**
- Jöle analitik OBB plakada yüzeye tam oturuyor (0.1250), yırtılmıyor (kopuk 0), genişliği
  aynı; ama yüksekliği 8 cm'den **1.15 cm'ye** iniyor. Bu, S3 bandının kalınlığı (h/2).
  μ 0'da da aynı.
- 2 hücre kalınlığındaki plakanın hücre merkezleri φ ≈ −h/2 ve kayan noktada eşiği (≤ −h/2)
  kaçırıyor. MPM ızgarasında plakanın hiçbir hücresi katı değil, yani **ızgara gövdeyi
  taşımıyor**. Parçacık başına S3 gerilme yaymaz; banda giren parçacığın yaklaşma hızını
  keser. Bütün gövde bandın içine yığılıyor. Küre kolu derin hücreler kaldığı için geçti.
- Aynı aileden: düzlem +17.8 mm, SDF küre −38.6 mm, kabuk −34.7 mm; collider testinde sakız
  8 mm'ye ezildi, sürtünme taşımıyor (`rt_test_collider_phi_ipc` 3/7,
  `rt_test_mpm_collider_ipc` 4/8).

**Duman 12/14:** yalnız `mpm_collider` (4/8) ve `collider_phi` (2/7). Diğer 12 kol PASS,
regresyon yok. ★ Dumandaki `collider_phi` koşusunda `sdf_cooked_on_gpu` ve `shell_on_gpu`
da FAIL oldu (ilk koşuda PASS). Bir probe'da da pişirme CPU'da yapıldı (14 ms). Slate
352 k'daki `dispatch/download failed` ile birlikte: **GPU sorgusu aralıklı başarısız
oluyor, geri dönüş sessizce CPU'ya geçiyor.** Raporlanıyor (`collider_query_error`), ama
kökü bilinmiyor; sonraki partide ölçülecek.

**Sonraki parti (plan):** teması **ızgara yüzünde φ ile** yapmak (standart MPM grid-SDF
collision). Yüzün φ'si (komşu hücrelerden) < 0 ise normal = ∇φ; bileşen-bileşen
yaklaşma kesilir, Coulomb ve yapışma ızgarada uygulanır. Izgara gövdeyi taşır, yüzey
hücreden incedir. S4 sökülür (ortak maske kalır, ama MPM ilerletmesinde hücre bloklaması
yerine φ). S3 yalnız güvenlik itmesi olarak kalır (normal/sürtünme ızgarada). Ayrıca: Auto
çözünürlükte h'nin yerel birimlere çevrilmesi ve 352 k'daki GPU sorgu hatası.

## TEK DERLEME: alt-voksel collider teması P1+P2+P3 — 2026-11-02 gece (YAZILDI, derlenmedi)

Plan ve ölçüm: [COLLIDER_TEMAS_YUZEYI.md](COLLIDER_TEMAS_YUZEYI.md).

**Derlenecek:**
- C++, **yeni `.cpp`:** `source/src/Physics/Fluid/ColliderGpuQuery.cpp` (+ `.h`); vcxproj ve
  filters'a eklendi.
- **Yeni shader'lar:** `sim_collider_bvh_query`, `sim_matter_mpm_particle_contact`.
- **Değişen shader:** `sim_matter_mpm_boundary`.
- Hepsi `compile_sim_shaders.bat` listesinde ve glslc OK.
- Madde denetimleri (36 ızgara alanı) ve IPC yetki denetimi PASS; tanımlayıcılar üretildi.

**İçerik:**
- **P1 — `grid.solid_phi`:** hücre merkezinde collider'a işaretli uzaklık (m). Voxelizer her
  modda yazıyor:
  - düzlem, küre, kapsül, AABB ve OBB analitik (OBB eksen ölçekli, ortalama ölçek değil);
  - mesh SDF **üç doğrusal** (eskiden en yakın örnek, yarım hücre kayık);
  - mesh_bvh kabuk: mesafe − kalınlık;
  - φ'si olmayan katı hücre (convex): −h/2.
  Kare önbelleğinde.
- **P2 — GPU SDF pişirme:** `rebuildSDFColliderAsync` işi kuyruğa koyuyor; publish anında
  `sim_collider_bvh_query` işaretli modda (3 ışın parite oylaması, CPU ile aynı) çalışıyor.
  Başarısız olursa aynı iş CPU işçisine gidiyor. Yeni çözünürlük modları: 3 = 256³,
  4 = **Auto** (eksen başına pişirme hücresi ≤ en ince domain vokselinin yarısı).
  `collider.get` → `sdf_dims`, `sdf_cooked_on_gpu`, `sdf_cook_ms`.
- **P3 — mesh_bvh kabuğu GPU'da:** BVH mesh sürümü başına bir kez yükleniyor; her kare
  hücre kutusu tek dispatch ve tek okuma. Başarısızsa CPU döngüsü çalışıyor.
  `fluid.step_stats` → `collider_query_gpu`, `collider_query_ms`, `collider_query_error`.
- **S3 — parçacık teması** (`sim_matter_mpm_particle_contact`, MPM şeridinde her alt adımda
  ilerletmeden sonra; CPU ikizi `applyMpmParticleContact`): φ < 0 ise yüzeye itme; içe
  giderken normal bileşen kesilir, kaymaya Coulomb uygulanır; ayrılırken h/2 bandında yapışma.
- **S4 — derin maske:** MPM şeridinin ızgarası yalnız φ < −h/2 hücreleri katı sayıyor. φ'siz
  (kinematik) hücreler katı kalıyor. Yalnız GPU'da; CPU referansı ortak maskeyi kullanıyor.
- Önceki partinin **yapışma bandı (geçiş 2) söküldü**; işi S3 yapıyor. Duvar kernel'i 2 geçiş.

**Sırayla:**
1. `python scripts/test/rt_test_collider_phi_ipc.py`: **7/7** hedef (h = 2.5 cm, h/4 = 6 mm).
   - `plane_on_surface` FAIL, boşluk ≈ +h/2 ya da +h: S3 çalışmıyor; jöle eskisi gibi voksel
     maskesinde duruyor. `fluid.step_stats` ve log'da `MPM parcel collider contact failed`
     ara; yoksa `var_solid_phi` yüklenmiyor.
   - `plane_on_surface` FAIL, boşluk negatif (yüzeyin altında): S3 itmesi işaret hatası ya
     da gradyan yönü ters.
   - `sdf_cooked_on_gpu` FAIL: `sdf_cook_ms` var ve GPU değilse kuyruk CPU'ya düşmüş, log'da
     `collider GPU query` hatası. Pişirme hiç bitmediyse publish tik'i GPU kuyruğunu çağırmıyor.
   - `sdf_matches_analytic` FAIL, `sphere_on_surface` PASS: üç doğrusal örnekleme ya da
     d_dünya = d_yerel / |S⁻¹n| dönüşümü yanlış. Küre mesh'inin facet'i en fazla ~4 mm sapar.
   - `shell_on_gpu` FAIL: `collider_query_error` alanına bak (BVH kurulumu, tampon sınırı).
2. `python scripts/test/rt_test_mpm_collider_ipc.py`: 8/8 hedef. Önce germe takılıyordu.
   - ★ **En sinsi:** `gum_stretches` PASS ama `friction_carries` FAIL. S4 kısmen örtülü
     hücreleri açınca ızgara sürtünmesi (geçiş 0) zayıflar. Kayan plakayı artık S3'ün
     Coulomb'u taşımalı; taşımıyorsa S3'te sürtünme normal itkiye bağlı, durgun temasta
     itki çok küçük kalıyor olabilir.
   - `gum_on_plate` yükselmede boşluk h/4 altında olmalı (eskiden 2 cm).
3. `python scripts/test/rt_test_elastic_mpm_ipc.py`: 11/11 kalmalı; zemin alan duvarı, S3'e
   girmiyor.
4. `python scripts/test/rt_matter_smoke.py --quick`: **14/14** hedef (yeni `collider_phi`).
   Collider'lı kum/kar kolları S3/S4'ten geçer.
5. **Performans karşılaştırması** (ajan, arşiv mesh'leri `E:d model_arsiv`): ölçüm
   tablosu aynı düzenle tekrarlanır. mesh_bvh 352 k üçgende 1240 ms'den milisaniyeler
   mertebesine düşmeli; `collider_query_ms` ile bak. mesh_sdf Auto'da pişirme süresi
   `sdf_cook_ms`.
6. Gözle: senin sakız sahnen; alt ve üst collider'a yapışma 1e6 verince sakız uzar. Collider
   panelinde SDF Resolution listesinde 256³ ve Auto var.

**Bu partide YOK:** deforme mesh'te GPU refit. Mesh sürümü değişirse BVH CPU'da yeniden
kurulup yükleniyor; sorgunun kendisi GPU'da. Ayrıca madde–collider çift birleştirmesi ve
DEM'e collider başına sürtünme.

## TEK DERLEME partisi — CANLI 2026-11-02 (exe 22:27, .spv 22:15–22:16)

| Test | Sonuç |
|---|---|
| Madde profilleri | PASS |
| Elastik | **11/11** (ilk kez tam): hamur 0.096 m / jöle 0.110 m, hamur hızı 0.018, jöle 0.022 |
| Collider | **6/8** |
| Hızlı duman | **12/13**; tek FAIL `mpm_collider` germe kolları. `elastic` dumanda ilk kez tam; regresyon yok |

Collider:
- PASS: rise (0.103/0.100 m), `friction_carries` (μ 0.8: 0.0998 m) ve **`frictionless_slips`**
  (μ 0: −0.003 m). Hayalet yüzler sürüklemeyi bitirdi; sürtünme artık gerçekten μ'ye bağlı.
- FAIL `gum_stretches` + `drawn_gum_tears`: sakız hiç gerilmiyor, yırtılması da bu yüzden yok.
  Mekanizma ölçüldü (`stretch` örnekleri, kare 13–24):
  - Sakızın tepesi plakayı 0.200 → 0.225 m'ye kadar 0.1 m/s ile izliyor (bant çalışıyor).
  - Sonra TAM hücre sınırında (0.225) duruyor, hız kayıtta 0.1 m/s kalıyor.
  - Plakanın voksel görüntüsü o hücreyi doldurduğu sürece advect'in katı çözümü
    parçacığı hücreye sokmuyor. Hücre boşaldığında aradaki boşluk bir hücreyi aşmış oluyor.
  - Hücre tabanlı yamalarla kapanmaz. **Çözüm: alt-voksel collider teması** (SDF ile
    yüz/parçacık uzaklığı); ayrı bir parti.

## TEK DERLEME partisi — 2026-11-02 gece (YAZILDI, derlenmedi)

Kullanıcı isteği: derleme döngüsü çok sık. Kalan işler tek derlemeye toplandı.

**Derlenecek:** C++ ve üç shader: `sim_matter_mpm_boundary`, `sim_fluid_granular_stress_update`,
`sim_matter_stress_update`. Yeni `.cpp` yok. Üç shader glslc OK. Madde denetimleri (9 dosya)
ve IPC yetki denetimi PASS.

**İçerik:**
1. **Hayalet yüzler** (MPM duvarı, geçiş 1): katının içindeki yüz, yanındaki açık yüzün
   sürtünmeden geçmiş değerini alıyor. G2P artık plakanın hızını içeriden okuyup μ 0'daki
   sakızı sürüklemiyor.
2. **Yapışma bandı** (geçiş 2): voksel plaka bir hücre zıpladığında açılan boş hücrenin
   gövde tarafındaki yüz, gövde ayrılırken yapışma sınırına kadar duvarın hızına tutuluyor.
   Yaklaşmayı durdurmuyor. CPU ikizi doluluğu parçacıklardan sayıyor.
3. **Sakız toplam uzamayla yırtılıyor:** yırtılma gerinimi = en büyük elastik asal gerinim +
   sünme ve akmanın bıraktığı biriken çekme akışı (`fracture_history` artık bu birikimi
   taşıyor, her adım sıfırlanmıyor). Jöle etkilenmez (sünme/akma yok). Eskiden sakız
   yavaş çekilince gerçekçi bir eşikte hiç kopmuyor, çok küçük bir eşikte ise anında
   dağılıyordu.
4. **Hamur akma eşiği** 0.15 → 0.10 (`dough_keeps_dent`).
5. **Madde editörü:** yırtılma, hasar hızı, akma ve gevşeme alanlarında "0 = kapalı" ve
   değer anlamı tooltip'te.
6. Collider testi 5 kola çıktı: yeni `draw` kolu (eşik 0.5, ~2.5× germe → yırtılmalı).
   Testin temizlik hatası düzeltildi (sahne nesneyi ".001" diye yeniden adlandırınca).

Adım 2–3'ün canlı sonucu (exe 22:00, `rt_test_mpm_collider_ipc.py` 5/7):
- PASS: rise (sakız 0.103 m / 0.100), slide (μ 0.8: 0.0998 m), sağlık.
- FAIL `frictionless_slips`: μ 0'da da 0.0998 m. Sürtünme değil, G2P'nin karesel şablonu
  plakanın İÇİNDEKİ yüzlere uzanıp oradaki 0.1 m/s'yi okuyordu. O değeri Adım 1'de
  "iki tarafı katı yüz = katının hızı" kuralı yazmıştı; sayısal bir sürükleme.
- FAIL `gum_stretches`: sakız üst plakayı 4 kare izledi (tepe 0.200 → 0.210), sonra
  bıraktı. Voksel plaka her 6 karede bir bütün hücre zıplıyor; arada sakızla plaka
  arasında BOŞ bir hücre kalıyor ve hiçbir yüz duvara değmiyor. Yükselme kolundaki
  2 cm'lik boşluk da aynı şey.
- Duman (exe 22:00) 11/13: yalnız bilinen `dough_keeps_dent` ve bu iki `mpm_collider` kapısı.
  Varsayılan μ 0.5 hiçbir kolu bozmadı.
- Ayrıca test hatası: sahne ikinci plakayı ".001" diye yeniden adlandırınca temizlik
  collider'ı yanlış adla silmeye çalıştı (düzeltildi).


**Sırayla** (önce bağımsız ve hızlı olanlar):
1. `python scripts/test/rt_test_substance_profiles_ipc.py`: PASS (Dough 0.10 okunuyor).
2. `python scripts/test/rt_test_mpm_collider_ipc.py`: **8/8** hedef.
   - `frictionless_slips` FAIL: sürükleme hayalet yüzlerden değil, başka bir yoldan
     (advect'in katı çözümü ya da P2G). `slip_gum_x_m` 0.0998'den düştüyse ama hâlâ
     0.025'ten büyükse kısmi; kaynağı ayrı ölçülmeli.
   - ★ `friction_carries` YENİ FAIL: önceki PASS aslında sürüklemeymiş, Coulomb yolu sakıza
     ulaşmıyor. Durgun boşluk (`p05 − lower_top`) bir hücreden küçük olmalı.
   - `gum_stretches` FAIL ama üst boşluk küçük: bant çalışıyor, sakız iç gerilmesiyle
     bırakıyor; yükseklik artışına bak.
   - `drawn_gum_tears` FAIL ama `gum_stretches` de FAIL: sakız hiç gerilmediği için
     yırtılmadı; önce germeyi çöz.
   - ★ **En sinsi:** `intact_finite` FAIL (eşik 1'deki `stretch` kolu da yırtılıyor):
     biriken akış ölçüsü fazla ölçüyor. Örneğin basıyla yayılma da "uzama" diye
     sayılıyor olabilir.
3. `python scripts/test/rt_test_elastic_mpm_ipc.py`: **11/11** hedef.
   - `dough_keeps_dent` artık PASS olmalı; `dough_holds_body` FAIL olursa 0.10 fazla
     akıtıyor demektir.
   - `gelatin_intact` / `tear_tears` aynı kalmalı (jölede akış yok).
4. `python scripts/test/rt_matter_smoke.py --quick`: **13/13** hedef.
5. Gözle: madde editöründe Elastic alanlarının tooltip'leri; collider panelinde Contact
   Surface; kendi sakız sahnende alt ve üst collider'a yapışma 1e6 verilince sakız uzar,
   yırtılma eşiği 1'de kopmaz, 0.5'te uzarken kopar.

**Bu partide yok, sonraki için:** FLIP 0.97'nin yavaş doğrusal rampası (2 s'de 0.06 m/s,
kök bilinmiyor; derlemesiz ölçülebilir); DEM'e collider başına sürtünme (üçgenlerde
collider kimliği yok); madde–collider çift birleştirmesi; alt-voksel (SDF) ızgara teması.

## Collider temas yüzeyi Adım 2–3: μ + yapışma, MPM Coulomb — 2026-11-02 (YAZILDI, derlenmedi)

**C++ + shader** (`sim_matter_mpm_boundary` yeniden; push 32 bayt, 9 tampon). Yeni `.cpp` yok.
glslc OK. Madde denetimleri (9 dosya; kare önbelleği 35 alan) ve IPC yetki denetimi PASS;
tanımlayıcılar yeniden üretildi.

**Ne değişti** (ayrıntı [COLLIDER_TEMAS_YUZEYI.md](COLLIDER_TEMAS_YUZEYI.md)):
- Collider'da `contact_friction` (μ, 0.5) ve `contact_adhesion_pa` (0): API, IPC, Python,
  iki kayıt yolu ve panel (Material sekmesi → "Contact Surface (MPM)").
- Hücre başına damgalanıyor; kare önbelleğinde ve voxel önbellek imzasında.
- MPM duvarı iki geçiş: teğet Coulomb (normal itkiyle sınırlı) + normal kısıt ve
  yapışma. CPU ikizi aynı. Alan duvarları değişmedi (sürtünmesiz, yapışmasız).
- ★ Varsayılan μ 0.5: collider üzerindeki MPM gövdeleri (kum, kar, jöle) artık kaymıyor.
  Eskiden collider yüzeyi teğette serbest kaymaydı.

Sırayla:
1. `python scripts/test/rt_test_mpm_collider_ipc.py`: **7/7** hedef (4 kol).
   - `friction_carries` FAIL, `frictionless_slips` PASS: geçiş 0 hiç çalışmıyor ya da μ
     cihaza 0 geliyor. `collider.get`'te `contact_friction` 0.8 mi? Değilse IPC/patch;
     öyleyse yükleme (`uploadGpuSolidContact`) ya da damga.
   - `gum_stretches` FAIL ve üst boşluk büyük: sakız üst plakadan ayrıldı. 1e6 Pa
     yetmiyor, ya da sakızın iç gerilmesi (E 2e5, gevşeme 1.5 s) bu çekme hızında 1e6'yı
     geçiyor. JSON'daki `stretch` örneklerinde ayrılmanın ilk karesine bak.
   - ★ **En sinsi:** `gum_stretches` PASS ama `intact_finite` FAIL. Sakız uzarken
     yırtılıyor (eşik 1 = 2.7× gerilme); iki plakaya da değiyor görünür. Yırtılma yapışmanın
     değil sakızın sınırı; gerekirse test hızı/eşiği ayarı.
2. `python scripts/test/rt_matter_smoke.py --quick`: 12/13 kalmalı (`dough_keeps_dent`).
   `mpm_dem_clock` ve `fast_fall` collider kullanıyorsa sürtünme varsayılanı etkileyebilir.
3. Gözle: collider panelinde iki yeni alan ve tooltip'ler; projeyi kaydet/aç, değerler
   korunuyor. Senin sakız sahnende alt ve üst collider'a yapışma (ör. 1e6) verince sakız
   aralarında uzamalı.

## MPM hareketli duvar — CANLI 2026-11-02 (exe 16:51, .spv 16:47)

`rt_test_mpm_collider_ipc.py` **5/5**:
- Plaka 0.100 m yükseldi, sakız 0.102 m. Hız plakayla aynı (0.09–0.10 m/s).
- Yükseklik 0.090 → 0.072 m (önce tek katmana ezilme). Kopuk parçacık yok.
- Sakızın tabanı plaka üstünden 2 cm (`gum_on_plate`, sınır 5 cm).

Hızlı duman **12/13**. Tek FAIL bilinen `dough_keeps_dent`. Kum, kar, tane, ısı, düşüş kolları
regresyonsuz: zeminden ayrılabilme hiçbir kolu bozmadı.

Açık (Adım 3): sakız taşınıyor ama iki yüzey arasında uzamıyor; bunun için yapışma
(`adhesion`) gerekiyor. Plaka–sakız arası 2 cm boşluk, voksel adımlı collider maskesinden
(h = 2.5 cm) geliyor.

## MPM hareketli + ayrılabilir duvar (Collider temas yüzeyi, Adım 1) — 2026-11-02 (YAZILDI, derlenmedi)

**C++ + yeni shader.** Yeni dosya `sim_matter_mpm_boundary.comp`; `compile_sim_shaders.bat`
listesine ve Vulkan kernel tablosuna (7 tampon, 16 bayt) eklendi, glslc OK. Yeni `.cpp` yok.
Madde denetimleri (8 dosya) + IPC yetki denetimi PASS. Tasarım:
[COLLIDER_TEMAS_YUZEYI.md](COLLIDER_TEMAS_YUZEYI.md).

**Ne değişti:** MPM şeridi katıya bitişik ızgara yüzlerini sıfırlıyordu. Bu yüzden
hareketli bir collider ona duran bir duvar gibi görünüyordu ve normal yönde yapışkandı.
Şimdi:
- Gövde katıya (katıya GÖRE) doğru giderse yüz katının hızını alır; ayrılıyorsa kendi
  hızını korur.
- Alan duvarı durağan. Teğet yüzler eskisi gibi serbest kayma.
- GPU: karma MPM şeridi + tek modelli granüler zincir. CPU: `enforceSolidBoundaries`
  granüler dalı. Sıvı şeridi değişmedi.

**Eski exe ölçümü (yeni test, 16:20 exe):** plaka 10 cm yükseldi, sakız 6.3 cm. Sakızın
tepesi hiç kıpırdamadı, alttan ezildi ve 34. karede tek bir katmana yapıştı
(min = max = 0.2375). Kullanıcının gördüğü "kopma" bu.

Sırayla:
1. `python scripts/test/rt_test_mpm_collider_ipc.py`: 5/5 hedef (`gum_rides`,
   `gum_keeps_height`, `gum_intact`, `gum_on_plate`, `finite`).
   - `gum_rides` FAIL, yükseklik korunuyor, sakız yerinde kalıyor: collider'ın hücre hızı
     (`grid.solid_vel`) `scene.set_transform` ile hareket eden mesh_sdf collider için
     DOLMUYOR olabilir. Yeni kural doğru, ama okuduğu hız 0. Sıradaki ölçüm bu.
   - ★ **En sinsi:** `gum_rides` PASS ama `gum_on_plate` FAIL (sakız plakanın üstünde
     uçuyor). Ayrılma kuralı plakanın ivmesinden fazlasını taşıyıp sakızı fırlatıyor
     olabilir. JSON'daki `gap_m` serisine bak: büyüyorsa fırlatma.
2. `python scripts/test/rt_test_elastic_mpm_ipc.py`: 10/11 kalmalı. Jöle artık zeminden
   gerçekten sekebilir (normal yönde yapışkanlık kalktı); `gelatin_calms` kayarsa sekme
   enerjisi bu partiden.
3. `python scripts/test/rt_matter_smoke.py --quick`: 13 kol (yeni `mpm_collider`).
   **Kum ve kar zeminden ayrılabiliyor artık**: `fast_fall`, `p9_floor`, `mpm_dem_clock`,
   `seed_settle` (kum dinlenmesi) dikkatle. Kum yığını zeminde zıplıyorsa çarpmanın normal
   hızı artık sıfırlanmıyor, yalnız katıya girişi engelleniyor; bu doğru fizik ama yeni.
4. Gözle (Play): senin sakız sahnen (eşik 1, akma 2). Sakız plakayla birlikte yükselir,
   ezilmez. **Uzamaz:** uzama için iki yüzeye yapışma gerekir (Adım 3, `adhesion`).

## Çadır yoğunluk düzeltmesi — CANLI 2026-11-02 (exe 16:20, .spv 16:15)

`rt_test_seed_settle_ipc.py` **6/6**, yeni `liquid_long_rest` dahil: dinlenik suyun hızı
1 s'de 0.047, 2 s'de **0.062 m/s** (önce 0.27, üstel). Kalan yavaş doğrusal rampa, FLIP 0.97'nin
bilinen davranışı (gevşeme kapalıyken de 0.053). Kum değişmedi (p95 0.1 mm, merkez kayması
0.16 mm). Duman testinin ilk 6 kolu PASS (substance, thermal_films, phase3_freeze,
fire_steam 8/8, grain_readiness, seed_settle 6/6). Bu koşuda uygulama ~5× yavaştı:
derleme sonrası VS ServiceHub 6.5 GB. Kalan kollar ayrı koşturuldu: p9_melt 4/4, p9_floor 2/2,
mpm_dem_clock 5/5, run_readback 1/1, fast_fall 2/2, elastic 10/11 (bilinen `dough_keeps_dent`).
**Toplam 11/12, regresyon yok.** Elastik alan Vulkan'da (`Mixed Vulkan: granular-only`, ~10 ms/adım);
testteki yüksek CPU çözücüden değil, IPC turları + kare başı host işi + VS indekslemesinden (ölçülmedi).

## Yoğunluk düzeltmesi: kutu ortalaması → çadır — 2026-11-02 (YAZILDI, derlenmedi)

**C++ + CUDA + shader.** `sim_fluid_density_relaxation.glsl`'yi içeren her shader yeniden
derlenmeli: `sim_fluid_cg_residual_init`, `sim_fluid_divergence_porous`,
`sim_sparse_mac_divergence_porous` ve bütün `sim_sparse_pressure_*` (12 dosya). Hepsi
glslc'den hatasız geçti. Tane/GPU madde denetimleri PASS (çadır ağırlığı için yeni iddia).
`check_domain_panel_fields` FAIL eskiden beri (HEAD'de de var), bu partiyle ilgisi yok.

**Kök (derlemesiz ölçüm + doğrusal model):** dinlenik tankın üstel rampasını yoğunluk
düzeltmesinin 27 hücrelik KUTU ortalaması yapıyordu.
- Kutu filtresinin Fourier dönüşümü yüksek frekansta negatif (eksen başına
  1/3 + 2/3·cos π = −1/3). Bu yüzden düzeltme, ızgara ölçeğindeki parçacık sayısı
  modlarında ters işaretle çalışıp onları büyütüyordu.
- Ölçümlerle tutarlı: büyüme 1/gevşeme ile hızlanıyor, kare süresinden bağımsız,
  CPU = Vulkan, FLIP 0'da da var.
- Model (e ← e − dt/max(τ,dt)·S(e), 12³): kutu 2 s'de 6.6e3× (τ 0.05) ile 4.5e4× (τ 0.0125)
  arası büyüyor. Çadırda (eksen başına 1-2-1; dönüşümü (1+cos k)/2 ≥ 0) her τ ve fps için
  ~0.5–0.6'ya sönüyor.
- **Düzeltme:** ortalama üç uygulamada (GLSL, CPU `APICFluidSolver.cpp`, CUDA) çadır
  ağırlıklı. İç/yüzey testi ve iki yönlülük aynı.

Sırayla:
1. `python scripts/test/rt_test_seed_settle_ipc.py`: 6/6. Yeni kapı **`liquid_long_rest`**:
   2 s / 24 fps'de dinlenik su, 2 s'deki hız < 0.08 ve < 2 × 1 s'deki. Eskiden 0.27'ye
   çıkıyordu; şimdi ~0.05 bekleniyor (FLIP 0.97'nin bilinen yavaş doğrusal rampası, ayrı iş).
   - Hâlâ üstel ise: shader'lar derlenmemiş olabilir (Vulkan), ya da kutu dışında ikinci
     bir negatif lob var. CPU arka ucuyla tekrarla; CPU geçip Vulkan kalıyorsa .spv eski.
   - ★ **En sinsi:** `liquid_at_rest` PASS, ama kütle merkezi yavaşça iniyor ya da
     yükseliyor. Çadırın ızgara ölçeğindeki modu (k = π) hiç düzeltmez, ama söndürür de:
     o mod artık büyümemeli. Yanlış yöne giden bir sapma ayrı bir sızıntıdır.
2. `python scripts/test/rt_matter_smoke.py --quick`: 12'de 11 beklenir (`elastic`'in
   `dough_keeps_dent` ayarı açık). `seed_settle` 6/6, `p9_melt` / `fire_steam` /
   `thermal_films` değişmemeli. Islak tane (gözenekli diverjans aynı terimi kullanıyor):
   `grain_readiness`.
3. Ritter barajı (varsa): çadır gürültüyü kutudan biraz az bastırır (ağırlık kare toplamı
   0.053'e karşı 0.037), yani yüzey biraz daha kıpırtılı olabilir. Ama artık rampa yok.

## FLIP rampası — kare dt testi, derlemesiz (2026-11-02, exe 14:40)

**Hipotez ("saf sıvıda da kare başına yerçekimi çekici") REDDEDİLDİ.** Dinlenik su tankı
(`rt_test_seed_settle_ipc` düzeni: 0.6×0.5×0.6 m, voxel 5 cm, dolu 0.3, settle açık), 2 s.
Değerler ortalama hız (m/s):

| Süre | Fluid 24 fps | Fluid 96 fps | Matter 24 fps |
|---|---|---|---|
| 0.25 s | 0.0145 | 0.0126 | 0.0145 |
| 0.50 s | 0.026 | 0.017 | 0.026 |
| 0.75 s | 0.042 | 0.025 | 0.042 |
| 1.00 s | 0.095 | 0.071 | 0.083 |
| 1.50 s | **0.268** | 0.158 | 0.205 |
| 2.00 s | 0.230 | 0.174 | 0.268 |

- Kare süresini 4 kat küçültmek rampayı yalnız biraz geciktiriyor; hatanın büyük kısmı kare
  başına itki değil.
- ★ **Rampa doğrusal değil, ÜSTEL:** fiziksel zamanda ~0.25 s'de ikiye katlanıyor, 1–1.5 s'de
  0.2 m/s'ye ulaşıyor. 30 karelik testler (0.5 s) bunu "yavaş rampa, 0.0006 m/s/kare" diye
  görüyordu. Su tankı 2 s'de durgun DEĞİL. Sinsi tür: kısa kabul testi geçiyor.
- 96 fps'de kütle merkezi 0.100 → 0.084 m düşüyor (1.6 cm), yani sıkışma ya da hacim kaybı
  da var. 24 fps'de bu düşüş 0.004 m.
- Fluid ve Matter domain aynı davranıyor: ortak yol (FLIP aktarımı, basınç, yoğunluk
  düzeltmesi).

### FLIP / yoğunluk taraması (2026-11-02, aynı tank, 2 s, 24 fps; Water: FLIP 0.97, gevşeme 0.05 s)

Değerler ortalama hız (m/s), 0.25 s aralıklarla; `cy` = 2 s'deki kütle merkezi yüksekliği (seed 0.100).

| Kol | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | cy |
|---|---|---|---|---|---|---|---|---|---|
| varsayılan | .015 | .026 | .042 | .083 | .162 | .205 | .188 | **.268** | .0944 |
| FLIP 0.5 | .008 | .010 | .016 | .032 | .051 | .065 | .151 | .245 | .0915 |
| FLIP 0 | .007 | .008 | .075 | .128 | .180 | .193 | .172 | .223 | .0920 |
| **gevşeme 0** | .014 | .025 | .033 | .040 | .046 | .050 | .052 | **.053** | **.1001** |
| FLIP 0.5 + gevşeme 0 | .008 | .010 | .009 | .010 | .009 | .011 | .011 | **.011** | .1001 |
| gevşeme 0.2 s | .014 | .026 | .034 | .042 | .048 | .052 | .055 | .066 | .0990 |
| gevşeme 0.0125 s | .015 | .026 | .082 | .149 | .232 | .224 | .358 | **.338** | .0866 |
| gevşeme 0.05, 96 fps | .013 | .017 | .025 | .071 | .151 | .172 | .159 | .196 | .0845 |

★ **Kök: yoğunluk KONUM düzeltmesi (`projectLiquidDensityPositions`), FLIP değil.**
- Düzeltme kapalıyken patlama yok: hız 2 s'de 0.053 (FLIP 0.97) ya da 0.011 (FLIP 0.5)
  m/s'de duruyor ve kütle merkezi kıpırdamıyor (0.1001).
- Düzeltme açıkken FLIP 0 bile 0.22 m/s'ye çıkıyor. "Yalnız FLIP 0 kaldırıyordu" bulgusu
  30 karelik (0.5 s) pencerenin yanılsamasıydı.
- Büyüme düzeltmenin KAZANCIYLA ölçekleniyor: gevşeme 0.0125 s → 0.34 m/s, 0.05 s → 0.27,
  0.2 s ≈ kapalı. Kare süresine bağlı değil (96 fps de patlıyor), yani sürekli zamanlı bir geri
  besleme; ayrık adım taşması değil.
- Bu tür bir düzeltme yoğunluk fazlasını SÖNDÜRMELİ, ama burada kütle merkezini 1–1.5 cm
  aşağı çekiyor ve hareketi büyütüyor. Kuşkulu yer fazla yoğunluk tahmincisi (iki yönlü
  terim + 27 hücre ortalaması + yalnız hava yüzü iç test, D2.8 partisi) ya da serbest yüzey
  satırında bir yön/işaret hatası. Gerçek bir gevşeme kazancı artınca daha hızlı söner, daha
  hızlı büyümez: bu, sistemin tekil/yanlış işaretli bir bileşenini düzeltmenin beslediğini
  gösteriyor (CLAUDE.md'deki "daha çok yakınsadıkça daha sert" dersi).
- Sıradaki (derlemesiz): düzeltmenin kaydırma alanını ve fazla yoğunluk okumasını katmanlara
  (taban / iç / yüzey) göre ölçmek. IPC'de yoğunluk kalıntısı istatistiği varsa onu oku.
  Yoksa küçük bir ölçüm yüzeyi gerekir (derleme).

## Yerçekimi alt adımda — CANLI sonuç 2026-11-02 (exe 14:40)

Elastik **10/11**, hızlı duman **11/12** (tek FAIL `elastic`). Kar/kum/tane kolları
(`fast_fall`, `p9_floor`, `mpm_dem_clock`, `seed_settle`) PASS: alt adım yerçekimi regresyon yapmadı.

| Ölçüm | 14:16 | 14:40 |
|---|---|---|
| Hamur son hızı | 0.171 m/s | **0.023** (dough_stops PASS) |
| Jöle son hızı | 0.146 | **0.036** (tepeden %2) |
| Jöle yüksekliği | 0.107 (FAIL) | **0.110** (PASS) |
| Normal jöle kopuk parçacık | 123 | **1** (gelatin_intact PASS) |
| Yırtılan jöle | 3584/3584 | 3584/3584 |

Kalan tek FAIL **`dough_keeps_dent`**: hamur 0.1104 m, jöle 0.1100 m, fark 0.4 mm. Hamurun
akma eşiği 0.15, 20 cm'lik düşüşte neredeyse hiç plastik akma vermiyor; jöle ile aynı yere
oturuyor. Kod hatası değil, ayar ya da kapı sorusu. Seçenekler: eşik 0.15 → 0.08–0.10 (önceki
0.05 bloğu 5 cm'ye akıtıyordu), ya da kapıyı "hamur ilk temasta kalıcı iz tutar" ölçüsüyle
(yükseklik/seed oranı) yeniden tanımlamak.

## MPM yerçekimi alt adımda + jöle eşiği — 2026-11-02 (YAZILDI, derlenmedi)

**Yalnız C++.** Shader yok, yeni `.cpp` yok. Madde GPU/tane/gözenek denetimleri (iki bayat
iddia önceki partinin koduna göre güncellendi; yeni bir iddia eklendi), IPC yetki denetimi PASS.

Derlemesiz ölçümle bulunan kök (exe 14:16):
- **Hamurun "hayalet" hızı bir YERÇEKİMİ ÇEKİCİ.** Karma madde adımında yerçekimi, adımdan
  önce kare başına bir kez, kare süresiyle parçacık hızına ekleniyordu (g·1/24 = 0.41 m/s).
  Durağan bir katı her kare zemine bu darbeyi yiyor, taban kenarı çınlıyor.
  - Kanıt 1: saklanan hız kenarda içe doğru 0.36 m/s, gerçek yer değiştirme 0.01 m/s.
    40 × 1 ms'lik adımla hız işaret değiştiriyor, yani gerçek bir salınım.
  - Kanıt 2: aynı alt adım dt'siyle kararlı hız kare süresiyle orantılı: 0.168 / 0.077 /
    0.044 m/s (kare dt 1/24, 1/48, 1/96), yani ≈ 0.4 · g · kare_dt.
  - Önceki partideki "açık viskozite salınımı" teşhisi yanlıştı (alt adımlar 24 → 36'ya çıkınca
    hız da 0.11 → 0.17'ye ÇIKMIŞTI).
- **Düzeltme:** MPM parçacığı (granüler ya da elastik) taşıyan karma Vulkan adımında
  yerçekimi ve paketli kuvvet alanları HER ALT ADIMDA kendi dt'siyle uygulanıyor
  (`MatterSubstepForces`; tane alanının ortak saati bunu zaten böyle yapıyordu). Kare
  başında yalnız kap hareketi kalıyor. Yalnız CPU'da çalışan alanlar (gürültü, rüzgâr
  yüzey sürüklemesi) eski kare yolunda.
- Gelatin yırtılma eşiği 0.6 → 0.8 (Hencky; 2.2× gerilme). Test kapısı `gelatin_intact`
  artık "kopuk parçacık ≤ %0.5" diyor (en büyük hasar < 0.5 yerine): tek bir köşe çatlağı
  kapıyı düşürmesin, %3'lük kenar çatlağı düşürsün.

Sırayla:
1. `python scripts/test/rt_test_elastic_mpm_ipc.py`: 11/11 hedef.
   - `dough_stops` hâlâ FAIL: JSON'da `dough_end_speed`'e bak. 0.17 aynı kaldıysa yeni yol
     çalışmıyor (alanda MPM parçacığı algılanmadı ya da eski exe). Düştü ama < 0.03 değilse
     ikinci, daha küçük bir kaynak var; kare dt'yi yarıya indirip yeniden ölç: orantılı
     düşüyorsa hâlâ kare başına bir itki var.
   - `gelatin_calms` / `gelatin_keeps_shape`: jölenin 3 Hz titremesi aynı çekiçle
     besleniyordu, ikisi birlikte düzelmeli.
   - ★ **En sinsi:** `dough_holds_body` FAIL, yani hamur birden akıyor. Yerçekimi artık alt
     adımlarda düzgün etki ediyor; çekiç gerilmeyi kısa darbelerle yüklüyordu. Akma eşiği
     0.15 sürekli yüke yetmeyebilir. Bu kodun bozulduğu anlamına gelmez, ayar sorusu.
2. `python scripts/test/rt_matter_smoke.py --quick`: 12/12 hedef. **Kar ve kum MPM
   alanları da artık yerçekimini alt adımda alıyor** (tane/DEM alanları değişmedi).
   `fast_fall`, `p9_floor`, `mpm_dem_clock` dikkatle: düşüş hızı ya da yığın şekli değiştiyse
   bu partiden.
3. Gözle (Play): hamur yere oturur ve kıpırdamaz; jöle zıplar, ~1 s'de durulur, kenarı
   çatlamaz. Kar yığını eskisi gibi.

Açık hipotez, test edilmedi: saf sıvı alanlarda da yerçekimi kare başına bir kez (aynı GPU
kuvvet geçişi, FluidDomainStep). **FLIP dinlenme rampası** bundan olabilir. Derlemesiz test:
dinlenik tankı kare dt 1/24 ve 1/96 ile aynı süre koştur; rampa kare dt ile ölçekleniyorsa kök bu.

## Elastik E1 düzeltmeleri + E2 canlı sonuç — 2026-11-02 (exe 14:16, .spv 14:03)

Madde testi PASS (31 hazır). Elastik **7/11**. Hızlı duman **11/12**, tek FAIL `elastic`;
`seed_settle` 5/5 (`liquid_settle_harmless` dahil), kar/kum/termal regresyon yok.

| Kapı | Sonuç | Sayı |
|---|---|---|
| lane_runs_elastic, gelatin_lands, gelatin_calms, dough_holds_body, water_spreads, tear_tears, finite | PASS | yırtılan jöle 3584/3584 kopuk |
| gelatin_keeps_shape | FAIL (kıl payı) | yükseklik 0.107 < 0.110 (0.7 × 0.157); genişlik 0.21 OK |
| gelatin_intact | FAIL | normal jöle 20 cm'den: 8. karede 15, 48. karede 123 parçacık kopuk (~%3, kenarlar) |
| dough_keeps_dent | FAIL | hamur 0.112, jöle 0.107 (jöle yırtılıp titrediği için alçak) |
| dough_stops | FAIL | **0.17 m/s, 36 alt adımda; şekil 8. kareden sonra 1 mm içinde sabit** |

Bulgular (sonraki partiye, burada ayıklanmadı):
- ★ **Hamurun hayalet hızı viskozite salınımı DEĞİLMİŞ.** Alt adım 24 → 36'ya çıktı, hız
  0.11 → 0.17'ye ÇIKTI ve 20. kareden sonra sabit (0.167–0.171). Gerçek kuvvet alt adımla
  küçülürdü; bu sabit bir bileşen. Aday: zemin/duvar hücresinde saklanan hız, konum
  kelepçelenirken sıfırlanmıyor (eski E1 hipotezi geri döndü). Ölçüm: hızı yüksekliğe göre
  dilimle; yalnız alt katman taşıyorsa duvar.
- Jöle 3 Hz civarı titremeyi sürdürüyor: hız kare kare 0.13 / 0.25 arası, yükseklik 0.094 –
  0.139. Kalıcı kopuk kenarlar bu salınımı besliyor olabilir.
- Yırtılma eşiği 0.6 çarpmada kenarı koparıyor. Ya eşik yükselmeli ya da kapı en büyük hasar
  yerine kopuk ORANINA bakmalı (%3 kenar çatlağı gerçekçi olabilir).

## Elastik E1 düzeltmeleri + E2 yırtılma — 2026-11-02 (YAZILDI, derlenmedi)

**C++ + shader.** Shader: `sim_fluid_granular_stress_update` ve `sim_matter_stress_update`.
Push bloğu **112 bayt kaldı** (116 indeksli): yırtılma eşiği eski pad yuvasına girdi, hasar
hızı DP'nin `damage_rate` alanını kullanıyor (iki yasada da saniyede hasar). İki shader
glslc'den geçti; madde GPU/wet/lane denetimleri, IPC yetki denetimi PASS, tanımlayıcılar
güncel. Yeni `.cpp` yok.

Derlemesiz ölçümle bulunan kökler (E1 canlı, eski exe):
- **Hamurun hareketsiz 0.11 m/s'si = açık viskozitenin alt adım salınımı.** Saklanan hız
  gerçek hareketin 6.7 katı (η=50); η=0'da bile 2.1 katı, çünkü dalga adımı tam sınırdaydı
  (c·dt/h = 0.35). Hız η ile ölçekleniyor: 0.112 / 0.065 / 0.051 (η 50 / 10 / 0).
  Düzeltme: elastik yasa alt adımları viskoziteye göre de boyutlandırıyor
  (η·dt/(ρh²) ≤ 1/12) ve dalga için %50 pay bırakıyor (modül × 2.25). Çekirdekteki tavan
  1/6'dan 1/12'ye indi; artık yalnız alt adım tavanı dolarsa devreye giriyor.
- **Jöle kaymıyor, durulmuyor:** kütle merkezi x/z ≈ 0, dikey titreşim 2 s sonra hâlâ
  ~0.2 m/s. η 2 → 8 Pa·s. Hamurun akma eşiği 0.05 → 0.15 (viskozitesiz 15 cm'lik blok
  5 cm'ye akıyordu).
- **Genişlik ölçüsü güvenilmezdi:** min/max sınırı kaçak parçacıkları izliyor, aynı jöle
  0.35 ya da 0.64 m okunuyordu. `fluid.state_digest` artık `bounds_p05/bounds_p95` veriyor;
  test bunları kullanıyor.
- **E2 yırtılma:** en büyük asal Hencky gerinimi `elastic_fracture_strain`'i geçince hasar
  `elastic_damage_rate` hızıyla birikir, iyileşmez. Hasar yalnız ÇEKME gerilmesini ve
  viskoziteyi siler; basınç kalır, yani kopan parçalar birbirine çarpar. d ≥ 0.99
  `granular_detached` olarak sayılır. Gelatin: eşik 0.6, hız 30.

Sırayla:
1. `python scripts/test/rt_test_substance_profiles_ipc.py`: PASS, yeni alanlar okunuyor.
2. `python scripts/test/rt_test_elastic_mpm_ipc.py`: 11 kapı, çıktı `%TEMP%/rt_elastic_mpm.json`.
   - `dough_stops` FAIL (hız ≥ 0.03, şekil sabit): alt adım boyutlandırması yüklenmemiş
     (eski exe) ya da 1/12 de yetmiyor. JSON'daki `substeps` 24'ten büyük olmalı.
   - `gelatin_calms` FAIL: η=8 yetmiyor ya da yeni bir pompa var. Hız büyüyorsa pompa,
     yavaş sönüyorsa ayar.
   - `tear_tears` FAIL ama `tear_max_damage` > 0: hasar birikiyor, 0.99'a ulaşmıyor.
     `tear_max_damage` = 0: eşik ya da hız shader'a ulaşmıyor (push alanı).
   - ★ **En sinsi:** `gelatin_intact` FAIL, yani normal jöle 20 cm'den düşüp yırtılıyor.
     Sayılar "jöle sağlam" görünür, ama Play'de parça parça olur. Eşiği yükselt; kodu bozuk
     sayma.
   - `gelatin_keeps_shape` artık yüzdeliklerden okunuyor. FAIL ederse JSON'daki `p05`/`p95`
     serisine bak: cisim yayılıyor mu, yoksa tek bir kenar mı akıyor?
3. Hızlı duman (`rt_matter_smoke.py --quick`): 12/12 hedef. Kar ve kumda alt adım sayısı
   DEĞİŞMEMELİ, yeni boyutlandırma yalnız elastik yasaya ait.
4. Gözle (Play): jöle zıplar, titrer ve ~1 s'de durulur; hamur basık kalır ve kıpırdamaz.
   Düşük eşikli jöle çarpınca parçalara ayrılır.

Hâlâ açık, bu partide yok: **FLIP rampası.** CPU'da da var (60 karede 0.032 m/s), yani
GPU'ya özgü değil. Yoğunluk projeksiyonu açıkken ~50. karede 0.022'den 0.051'e bir sıçrama
var; kapalıyken yalnız yavaş rampa (0.025). Kök bulunmadı.

## Büyük parti canlı sonuç — 2026-11-02 (exe 12:12, .spv 12:00)

Hızlı duman **11/12**. Regresyon yok; `seed_settle` dumanda ilk kez 5/5.

| Ölçüm | Önce | Sonra |
|---|---|---|
| Su p95 tepesi | 0.89 mm/kare (20. karede sıçrama) | **0.45** (sıçrama yok) |
| Kütle merkezi, 30 kare | +1.4 mm | **+0.1 mm** (şişme kapandı) |
| Madde testi | — | PASS, 31 hazır madde |
| Elastik test | — | 6/8 |

Elastik: `mpm_law = elastic`, 11 alt adım (Dough 24). Jöle iniyor ve **gerçekten zıplayıp
titriyor**: yükseklik 0.109 ile 0.213 m arasında salınıyor, sonda 0.148 (tohum 0.175). Hamur
8. karede 0.10 m'ye basıklaşıp kalıyor. Su 0.03 m'ye yayılıyor. İki FAIL:
1. **gelatin_keeps_shape:** genişlik her 4 karede ~3 cm artıyor (0.20 → 0.64 m), yükseklik
   sabitken. Cisim yassılaşmıyor; tabandan sızan parçacıklar sınır kutusunu şişiriyor
   olmalı (ölçülmedi).
2. **gelatin_settles:** sonda hız 0.20 m/s. ★ Hamur 8. kareden sonra hiç kıpırdamıyor, ama
   ortalama hızı sabit **0.11 m/s**: konum değişmiyor, hız değişmiyor. Aday: zemin
   hücresindeki MPM parçacığının hızı her adım duvarın içine bakıyor, konumu da kırpılıyor.
   Jöledeki dip hız ve tabandaki sızma aynı kökten gelebilir. İlk ölçüm: en alttaki
   parçacıkların hız vektörü (`particle_extremes` ilk 64 parçacığı zeminden okur).

Hâlâ açık: FLIP'teki yavaş rampa (30 karede 0.45 mm, eşik 0.5). Daha uzun koşuda geçmez.

## Büyük parti: elastik MPM (E1) + yoğunluk şişmesi + su kapısı — 2026-11-02 (DERLENDİ, sonuç yukarıda)

**C++ + CUDA + shader.** Shader paketini derle. Değişenler: `sim_fluid_granular_stress_update`,
`sim_matter_stress_update` (push 96 → 112 / 100 → 116), `sim_matter_partition`,
`sim_fluid_density_relaxation.glsl` içerenler (`sim_fluid_cg_residual_init`,
`sim_sparse_pressure_*`, `*_divergence_porous`). Yeni `.cpp` yok. Statik: shader'lar
glslc'den geçti; `check_matter_gpu_contracts` (13 ABI), wet/lane denetimleri, IPC yetki
denetimi PASS; tanımlayıcılar güncel. Kalan `check_*` FAIL'leri bu partiden önce de vardı
(bulut/gaz/hacim/panel). Tasarım: [ELASTIK_MPM.md](ELASTIK_MPM.md).

Sırayla, hızlıdan yavaşa:
1. **ABI.** Uygulama açılır, gaz ya da madde adımı "push constant" hatası vermez. Verirse
   eski `.spv` ile yeni exe eşleşmemiştir: stres shader'ı 112 bayt bekliyor.
2. `python scripts/test/rt_test_substance_profiles_ipc.py`: Gelatin/Dough/Chewing Gum
   `elastic`, alanlar okunuyor. FAIL → madde kaydı (SubstanceLibrary alan tablosu).
3. `python scripts/test/rt_test_seed_settle_ipc.py`: **liquid_at_rest PASS** beklenir (önce
   0.89 mm/kare, tek karelik tepe). `liquid_settle_harmless` yeni kapı. Hâlâ ~0.9 ise duvar
   katmanı düzeltmesi yüklenmemiş (eski `.spv`). Su seviyesi düşüyorsa (kütle merkezi
   −1 mm'den fazla) duvar hücreleri sistematik olarak eksik doluymuş demektir; o zaman
   "katı komşu iç hücre sayılır" kuralı yalnız alan duvarına daraltılmalı.
4. `python scripts/test/rt_test_elastic_mpm_ipc.py`: 8 kapı. Sıra önemli:
   - `lane_runs_elastic` FAIL → `mpm_law` "none" ya da "drucker_prager": yasa seçilmedi
     (resolveParticleGranularPhysics) ya da parçacık MPM şeridine girmedi (partition).
   - ★ **En sinsi:** jöle hiç düşmüyor ya da her karede seed yerinde duruyor. Bu, sonuç
     birleştirmenin elastik parçacığı atladığını gösterir (MatterGpuStep `mpm_parcel`).
     Sayılar "şekli korudu" der; `gelatin_lands` bunu yakalar.
   - `gelatin_keeps_shape` FAIL ve su gibi yayılıyorsa: gerilme P2G'ye ulaşmıyor.
   - `gelatin_settles` FAIL (hız büyüyor): enerji pompası. Önce viskoziteyi 0 yapıp dene;
     düzelirse açık Kelvin-Voigt tavanı yetmiyor.
   - `dough_keeps_dent` FAIL: akma çalışmıyor ya da jöle de kalıcı ezildi.
   - `water_spreads` FAIL ise test anlamsız: kontrol kolu bozuk.
   Çıktı: `%TEMP%/rt_elastic_mpm.json`. Alt adım sayısını oradan yaz.
5. Hızlı duman: `python scripts/test/rt_matter_smoke.py --quick` (artık `elastic` içeriyor).
   Kar, kum ve su değişmemeli. `mpm_dem_clock` ve `p9_floor` farklıysa MPM yük ölçümü
   (MatterGranularLoad) veya `has_fluid` değişikliği bir şeyi kaydırmıştır.
6. Gözle (Play): Gelatin küpü düşer, yaylanır, titreşir ve durulur. Dough basıklaşıp kalır.
   Chewing Gum yavaşça sarkar. Kar alanında `mpm_law` = snow, kum alanında drucker_prager.

Bu partideki diğer iki düzeltme:
- **Yoğunluk şişmesinin kökü:** iç hücre testi duvar ve katı komşuyu "yüzey" sayıyordu.
  Zemin ve duvar katmanları tek yönlü kalıyordu: fazla dolu itiliyor, eksik dolu
  çekilmiyordu. Gürültü bu yüzden şişmeye dönüşüyordu (30 karede +1.4 mm). Artık yalnız hava
  komşusu iç hücreyi bozuyor. Formülün dört kopyası aynı: CPU, ortak GLSL, CUDA.
- **Elastik kütle:** katı model `density` ile tartılır (önce `liquid_density`).

Ölçülmeyen: FLIP'teki yavaş rampa (~0.0006 m/s/kare) bu partide ele alınmadı.

## D2.8 canlı sonuç — 2026-11-02 (exe 11:25, dört g2p .spv yeni)

Kök doğrulandı, kapı henüz geçmedi.

| Ölçüm | Önce | Sonra |
|---|---|---|
| Su p95 tepesi (seed_settle) | 4.31 mm/kare | 0.89 mm/kare (eşik 0.5) |
| Ortalama hız, 30. kare | 0.165 m/s | 0.023 m/s |
| İlk adım hatası | 0.0092 m/s | 0.0009 m/s (FLIP 0 ile aynı) |
| Ritter FLIP 0.3 | 1.11 m/s, %54 | 1.86 m/s, %23 |
| Ritter FLIP 0.9 | 1.95 m/s, %20 | 1.97 m/s, %19 |
| Hızlı duman | — | 10/11, yalnız seed_settle FAIL |

Ritter'de 0.3'ün gerilemesi kapandı (no_regression PASS). Doğruluk kapıları (≤%15) hâlâ FAIL.
Ön ayar değiştirilmedi.

Kalan, sonraki kod partisine (test sırasında ayıklanmadı):
1. ★ **0.89'luk tepe ve şişme = konum tabanlı yoğunluk projeksiyonu.** Yalnız tek bir kare
   (~20.) 0.89 mm sıçrıyor, komşuları ~0.45. Kütle merkezi 30 karede +1.4 mm (FLIP 0'da da
   aynı). `solver_density_relaxation_s=0` ile ikisi de yok: p95 ≤ 0.35 mm, kütle merkezi
   +0.05 mm. Yani projeksiyon dinlenik ppc 4 tohumu fazla dolu sanıp genişletiyor.
2. FLIP'te yavaş rampa kalıyor: kare başına ~0.0006 m/s, 30 karede 0.02 m/s (FLIP 0'da 0.006).
   30 kare kapının altında, ama daha uzun koşuda büyür. İlk aday duvar sınırlayıcısı
   (`use_solid_flip_limiter` 0.65) ve P2G'nin duvardaki kesmesi.
3. `liquid_control_moves` artık anlamsız: su için oturtma 1 adımda ve sıfır yer değiştirmeyle
   bitiyor, açık/kapalı kollar birebir aynı (0.8928 mm). Bu kapı önceden yalnız gürültü farkıyla
   geçiyordu. Su kolunda kapıyı "kontrol hareket ediyor" yerine "ikisi de dinlenik" yap.

## D2.8 dinlenik su: G2P duvar kesmesi — 2026-11-02 (DERLENDİ, sonuç yukarıda)

**C++ + CUDA + shader.** Shader'ları derle: `sim_fluid_g2p`, `sim_sparse_mac_g2p`,
`sim_sparse_mac_matter_g2p`, `sim_matter_g2p` (dördü de aynı `sim_fluid_g2p.glsl`).
ABI ve push sabitleri değişmedi.

Kök (derlemesiz ölçüldü): G2P, alanın dışına düşen yüzlerin ağırlığını atıyordu ve kalan
ağırlıkları yeniden normalize etmiyordu. Duvara yarım hücreden yakın her parçacıkta
G2P(düzgün alan) kısa kalıyor. Her adım parçacığa verilen yerçekiminin FLIP deltası ile
geri alınması eksik kalıyordu; artan aşağı itiş duvar boyunca birikiyordu.
Sonuç: duvarda aşağı, ortada yukarı bir konveksiyon hücresi. Kanıt: tek parçacıkta beklenen
`eksik ağırlık 0.536 × g·dt × 0.63` = −0.0552 m/s, ölçülen −0.0552 m/s. Elenenler (hepsi
aynı rampa): CG 2000 iterasyon (kalıntı 1e-6), yoğunluk düzeltmesi kapalı, APIC 0, sürükleme
ve sönüm kapalı. Hız dt'den bağımsız olarak ~0.15 m/s'ye doyuyor, kütle merkezi 0.5 s'de
+4.5 mm yükseliyor (şişme). Düzeltme: alan dışındaki yüz en yakın yüzün değerini alır.
Vulkan GLSL, CPU `APICFluidSolver.cpp` ve CUDA `fluid_g2p_gather_kernel` aynı.
P2G dağıtımlarına dokunulmadı; orada yüz değeri normalize edilmiş ortalama.

Sırayla:
1. `python scripts/test/rt_test_seed_settle_ipc.py`: liquid_at_rest PASS (su p95 < 0.5 mm/kare;
   önce 4.31). Ortalama hız büyümemeli, ~0.005 m/s'de kalmalı (önceki FLIP 0 kontrolüyle aynı).
   Değişmediyse: eski `.spv` yüklü (dört g2p dosyasının zaman damgasına bak). Yarıya indiyse:
   kalan kaynak duvar sınırlayıcısı (`use_solid_flip_limiter`, 0.65).
2. Kütle merkezi 30 karede ±1 mm içinde (önce +4.5 mm). Hâlâ yükseliyorsa şişme ayrı bir kök.
3. `python scripts/test/rt_test_ritter_flip_ipc.py`: cephe hızı artmalı. Eski kesme, zemindeki
   parçacığın x-hızının PIC kısmını %12'ye varan oranda düşürüyordu; 0.3'ün gerilemesi büyük ölçüde
   buradan olabilir. Ön ayarı bu ölçümle **değiştirme**; yalnız sayıları yaz.
4. Hızlı duman (`rt_matter_smoke.py --quick`): bütün sıvı G2P'si değişti. fast_fall, p9,
   mpm_dem_clock, run_readback aynı kalmalı.
5. ★ Sinsi olan: duvar artık gerçekten kaygan (köşede önceden %50'ye varan teğet sürtünme vardı).
   Köşelerde sıvının duvara tırmanması ya da alan dışına sızma görülürse, normal bileşenin
   sabit uzatması yanlış demektir (yansıtma gerekir). Sayılar bunu göstermez, Play'de bak.

## Madde regresyonu, 10-29 partisi — 2026-11-02 (KOŞULDU, derleme gerekmedi)

Exe 11-01 14:54, 10-29 madde kaynaklarından yeni. Boş sahne, dış IPC.
`rt_matter_smoke.py --with-legacy-project --only <14 kol>`: **13/14 PASS**.

- PASS: matter_lanes 7/7, simulation_transport 8/8, substance, thermal_films 5/5,
  phase3_freeze, fire_steam 8/8, grain_readiness, p9_melt 4/4, p9_floor 2/2,
  run_readback, fast_fall 2/2, mpm_dem_clock 5/5 (40 continuum × 2 DEM = 80, grid 40),
  legacy_project (Gas+Fluid → Matter, kapalı kaynak fazı, gaz shader'ı, canonical yeniden açma).
- FAIL: seed_settle liquid_at_rest. Su p95 4.311 mm/kare, settle kapalıyken 4.327; Sand 0.106.
  10-29 ile aynı, D2.8 açık.
- Düzeltilen iki **test** hatası (kod değil, scripts + x64/Release kopyası):
  1. `mixed_clock_schedule` her zaman FAIL ediyordu. Test ve `rt_probe_grain_clock_ipc.py`
     `runtime.substep_limit` değerini tavan diye geçiriyordu. Oysa bu değer sınırlayan
     etkenin adı (`"damping"`). Yazılan tavan `grain_diagnostics.cfl_terms.max_substeps`'te.
  2. legacy_project `KeyError: 'shader'`. Yeni gaz domain'inin malzemesi yok
     (`has_material:false`), yani kayıtta taşınacak shader da yok. Test artık önce
     `volume.set_material` ile malzeme yazıyor.
- ★ Sinsi olan: 1. hata "saat tutarsız" gibi okunuyordu, ama kapı hiçbir durumda geçemezdi.
- Koşulmayanlar: Ritter, 4 emitter'lı hız probu, cache preflight bozuk dosya senaryosu
  (C++ testi var, derlenmedi), gerçek tarihsel proje dosyası.

## Katı yoklamaları rayQuery REDDEDİLDİ — 2026-11-01 (en önce)

★ Kod geri alındı ama kurulu `.spv` hâlâ reddedilen denemedir. Shader paketini yeniden
derle, C++ gerekmez. Derlemezsen gaz yaklaşık 3,5 kat yavaş çalışır ve hata vermez.

1. 69. kare, burn malzemesi, iç ışık kapalı: ~105 ms. 300 ms üstüyse eski `.spv` yüklü.
2. Aynı karede iç ışık 8 adım: ~120 ms. 250 ms civarıysa `ca9ac44b` öncesi bir shader.

Neden yavaşladığı bulunmadı; handoff notunun başında.

## İç ışık rayQuery denemesi — 2026-11-01

★ KOŞULDU, madde 1–3 tamam: 1 adım 245 → 110 ms, 32 adım ~290 → 136 ms, görüntü
sayıları aynı, test PASS. Kalan yalnız madde 4 (elle).

1. Pipeline kurulmalı. Kurulmazsa cihazda rayQuery kapalıdır; hacim hiç çizilmez.
2. 69. kare, burn malzemesi, iç ışık 1 adım: kapalı hale yakın.
3. `python scripts/test/rt_test_pyro_internal_ipc.py --domain "Nuclear Gas" --out <scratchpad>`
   PASS ve p50 değerleri aynı (.636 civarı).
4. Gaz içine katı koy: ateş ışığı katıdan geçmemeli. Bu sahnede yalnız zemin var, ölçülmedi.

Adım için 8 kullan: ışık 32 ile aynı, maliyet eşit ya da düşük. 1–2 adım ateşe ulaşmaz.

## Eklemeli flame, iç ışık + burnout, panel — 2026-11-01

★ KOŞULDU: kullanıcı derledi, ajan ölçtü. Madde 5 ve 6 tamam, Smoke eşlemesi script
tarafında PASS. Panel (1, 2, 4), senin kendi malzemenle 3. madde ve RayFusion gözle
bakılmadı. Sayılar handoff notunun başında. `flame_emission` başlangıcı 0.05–0.5
YANLIŞ çıktı: 0.1 ölçülemiyor, görünür etki 0.5–2. Etkisi başlıkta değil, zemin
halkası ve gövdede; 0 bırak. Düzeltilmedi: adım sayısı enerjiyi değiştiriyor, 32
kullan, adımı düşürmek maliyet kazandırmaz. Aşağısı ilk liste olarak duruyor.

    python scripts/test/rt_test_pyro_flame_ipc.py --domain "Nuclear Gas" --out <scratchpad>

Sırayla bak, hızlıdan yavaşa:

1. Derleme temiz. Sık kırılma adayı: `VolumeMaterialUI.cpp` yeniden bölündü,
   `scene_ui.cpp` içinde iç içe `TreeNode`/`TreePop` çiftleri taşındı.
2. Panel: Smoke, Lighting, Fire, Quality bölümleri görünmeli; gaz domain'inde Data da.
   Bir ayar kaybolduysa veya ImGui stack hatası varsa TreePop dengesi bozulmuştur.
3. Senin mevcut malzemen (independent kapalı, internal 32/32): görüntü önceki gibi
   kalmalı, `flame_emission` 0. Değiştiyse legacy iç ışık integrali bozulmuştur.
   Yalnız burnout açıkken yanan hücrelerde çok hafif fark beklenen: düzeltilen hata.
4. Duman: Color değiştir, siluet aynı kalmalı. Opacity değiştir, renk aynı kalmalı.
   İlk düzenlemede görüntü sıçrarsa `applySmokeAppearance` eşlemesi yanlıştır.
5. Independent aç, internal 32 kalsın, burnout .9 (.2–.6). Önce internal 0 ile kıyasla:
   duman ateşten ışık almalı. Almıyorsa `sampleDensityChannels` kanalı boş dönüyor.
6. Test PASS ve 69. karede `add2`, `noflame`dan farklı olmalı. Aynıysa hücreler hâlâ
   sıfır üretiyor: `volumeFlameRadiance` veya `flame` adresi. Süre `noflame` ile yakın
   olmalı; flame kapalıyken ek okuma yok.
7. `flame_emission` değeri: yoğunluk şiddeti taşır, 2 çok güçlü olabilir. 0.05–0.5 ile
   başla. Beyaza patlıyorsa bu ayardır, hata değil.
8. Koyu noktalar: independent modda 12–30. karede bak. Legacy'de görülmemesi onları
   temize çıkarmaz.
9. RayFusion: iç ışık, burnout ve flame aynı yönde görünmeli. Görünmüyorsa frag'daki
   668 ofsetine bak. M=0 elle bakış ve eski proje açılışı önceki listeden açık.

★ EN SİNSİ SONUÇ: Bağımsız modda iç ışık artık açılıyor; daha önce sessizce sıfırdı.
İç ışık adımı kayıtlı eski bir independent sahne, hatasız şekilde daha parlak ve daha
pahalı açılır. Bug gibi raporlanmaz; `internal_light_steps` değerine bak.

## Pyro flame partisi — 2026-11-01

★ Bu partideki `flame_boost` artık `flame_emission`; üstteki liste geçerli.

Ajan derledi (önceki yetkiyle), `rt_test_pyro_flame_ipc.py` PASS, malzeme geri
yüklendi, proje kaydedilmedi. Ölçümden SONRA tek kaynak değişikliği yapıldı
(`FluidDomainFogVolume.cpp` ölü blok söküldü), bu nedenle C++ yeniden derlenmeli.
Shader değişmedi. Sayılar: handoff notunun başı.

1. Derleme temiz bitmeli. Hata verirse `registerOrUpdateLiveVolume` çağrısına bak.
2. Yeniden test: `python scripts/test/rt_test_pyro_flame_ipc.py --domain "Nuclear Gas" --out <scratchpad>`.
   PASS ve 12. karede `boost4` p95 > `noflame` p95 görülmeli. Değilse cihaz yolundaki
   flame adresi kopmuştur.
3. 69. karede `boost4` ≈ `noflame` BEKLENEN sonuç: yanan hücrelerin taban emisyonu
   sıfıra yakın. Hata sayma; `boost1` zeminde farklı çıkarsa örnek sayısı farkıdır.
4. Önbellek oynatımı: gazı cache'ten çal, `flame_boost` 0 ve 4 karşılaştır, 12.
   kare civarında fark görülmeli. Fark yoksa yeni NanoVDB yolu çalışmıyor; bu yol
   hiç ölçülmedi.
5. Koyu noktalar: 12–30. karede ilk koşuda var, geri sarınca yok. Hangi koşulda
   çıktığı bilinmiyor; kök bulunmadan kaliteyle ilgili kesin karar verme.
6. RayFusion: 3. madde ve aşağıdaki listedeki gözle bakışlar hâlâ açık.

★ EN SİNSİ SONUÇ: Flame boost slider'ı geç karelerde hiçbir şey yapmaz ve hata
vermez. Kullanıcı bunu "bozuk" sanabilir, oysa çarpan sıfırın üstünde çalışıyor.
Geç karede parlaklık için `soot_burnout` kullan, boost'u yükseltme.

## Pyro tepki partileri — 2026-11-01

Bu parti ajan tarafından kullanıcı talimatıyla derlendi (Release + shader) ve
`rt_test_pyro_response_ipc.py` nuke.rtp kare 69'da PASS verdi. Kalan gözle bakışlar:

1. RayFusion: `temperature_gamma`, `soot_burnout` ve yumuşak sınırlayıcı aynı yönde
   görünmeli. Görünmüyorsa frag'daki bayt ofsetleri (560..572) kaymıştır.
2. Panel: Physical volume bölümünde "Measure field" sayıları göstermeli; "Fit
   temperature to field" sonrası EV düşürmek gerekebilir. Fit beyazlatıyorsa bu
   beklenen sonuç, hata değil.
3. Eski proje: `fire_scatter_*` taşıyan bir sahne hatasız açılmalı. Hata verirse
   okuyucu eski anahtarı reddediyordur.
4. M=0 yayıcı: zemin ışığı kaybolmamalı. Kaybolursa `vrcProvenEmpty` işaretine bak.
5. C++ testi `volume_material_authoring_test.cpp` düzenlendi, koşulmadı.

★ EN SİNSİ SONUÇ: Random walk artık gerçekten çalışıyor, açık kalan eski sahneler
kat kat yavaşlar ve duman gövdesi açılır. Sonuç hatasız ama farklıdır. Gerekçe
handoff notunda: 64 spp'de kıvılcım duruyor, varsayılan yapılmadı.

★ KOŞULDU: test PASS (süre %40–50 düştü, başlık aynı, zemin +0,8 ve +2,1/255).
Madde 6 (M=0 elle bakış) ve RayFusion gözle kontrolü hâlâ açık.

Derleme: yalnızca shader paketi; C++ değişmedi. Soğuk RT pipeline bekle, hazır
olmadan testi başlatma. Tek test, Rendered modda:

    python scripts/test/rt_test_pyro_bounce_cache_ipc.py --target "gas:Nuclear Gas" --out <scratchpad>

Sırayla bak:

1. `RESTORED True` ve sonda PASS. Değilse malzeme bozuk kalmıştır, kaydetme.
2. `counters`: `radiance_cache_marches` sıfırdan büyük, `secondary_density_samples`
   önbelleksiz haline göre çok düşük. Sıfırsa önbellek hiç devreye girmiyor;
   süre kazancı başka nedendendir.
3. `tuned`: `cache_ms`, `full_ms`in belirgin altında (sekme 32 denemesi 117 ms'ye
   inmişti). Düşmediyse süre önbellek dolumuna gidiyor olabilir, 2. maddeye dön.
4. `cap_cache` ≈ `cap_full`. Başlık kamera ışını, önbellek ona dokunmamalı.
   Farklıysa sekme dışı bir yol etkilenmiştir.
5. `ground_cache` ≈ `ground_full`, iki malzemede de (eşik 5/255). `bright` asıl
   sınav: 2026-10-25'te zemin burada +8/255 kaymıştı.
6. Burnout 1'de ve M=0'da zemin ışığı kaybolmamalı: vakum dalı ve sıfır yoğunluklu
   yayıcı. Scriptte yok, yalnız M=0 için elle bakılacak.

★ EN SİNSİ SONUÇ: `vrcProvenEmpty` yoğunluğa bakar. Yoğunluğu sıfır bir yayıcı
sessizce atlanırsa ışık hatasız, makul biçimde azalır. 1e-12 işareti bunu
önlemek için; zemin karanlıksa önce buna bak.

Bilinen, bu partiden gelmeyen: `check_gas_overlap_cache_contracts`,
`check_volume_optical_block_contracts`, `check_volume_shared_walk` birer eski
metne bağlı FAIL veriyor (ör. `term.emission = source * term.extinction`).

## Soot burnout + RT kamera step sınırı — 2026-11-01

★ KOŞULDU: test PASS; madde 1, 2, 3, 5 tamam. Madde 4 tutmadı: 64 sınırı %17
kazandırdı, kalan maliyet sekme ışınlarında (ayrıntı handoff notunun başında).
Madde 7 (RayFusion) hâlâ gözle bakılacak. Aşağısı ilk liste olarak duruyor.

Derleme: C++ VE tam shader paketi. `params.h` değişti, CUDA/OptiX da yeniden
derlenmeli. VkVolumeInstance 688 bayt kaldı. Shader hash değiştiği için ilk
açılışta soğuk RT pipeline bekle (son ölçüm 398,7 sn); bu ayrı bir bulgu değil.
Pipeline hazır olmadan test başlatma.

Tek test, açık sahnede, Rendered modda:

    python scripts/test/rt_test_pyro_soot_steps_ipc.py --target "gas:Nuclear Gas" --out <scratchpad>

Sırayla bak; sonuncular öncekilere bağlı:

1. `contract` satırı hepsi true olmalı. False ise: IPC/doğrulama eksik; ölçümlere
   geçme, sonuçlar yanlış ayarla alınmıştır.
2. `RESTORED True` ve sonda PASS. False ise malzeme bozuk kalmıştır, kaydetme.
3. `legacy_gate` ≈ `legacy_b0` (fark ≤ .003). Değilse burnout source-function
   emisyonuna sızmıştır: alev kararır.
4. `cap64_thin` süresi `cap0_thin`in yaklaşık yarısı, ortalama ±%2. Süre aynıysa
   sınır shader'a ulaşmıyor: paket/decode kaçağı. Önceki referans 120 / 230 ms.
5. `thin_b90` / `thin_b90_low` montajı: sıcak çekirdek içeriden parlamalı, dış
   koyu kabuk durmalı. İkisi de `thin_b0` ile aynıysa aralık sahnedeki sıcaklığa
   denk gelmiyor; önce `soot_burnout_start/end` değerlerini düşür, kodu suçlama.
6. Beyaz patlama: `soot_burnout` 1'e yakınken sönüm sıfırlanır, emisyon j·dt'ye
   geçer ve ölçek küçük kalmaz. Patlıyorsa EV düşür; bu ayar sonucudur.
7. RayFusion'a geç, aynı malzeme: aynı yön görünmeli, step 128'de bloklaşma
   olmamalı. Bu adım scriptte yok, gözle bakılacak.
8. Yeni malzemede `max_steps` 128 gelmeli. Kayıtlı sahne kendi değerini (121) korur.

★ EN SİNSİ SONUÇ: gölgeler burnout'u görmez. Sıcak gazın gölgesi eski yoğunlukta
kalır ve bu, hatasız "biraz karanlık çekirdek" gibi görünür. Bilinçli sınır;
kusur sayılıp görünmeden burnout gücünü yükseltme.

Bilinen, bu partiden gelmeyen: `check_volume_march_lod_contracts.py` iki maddede
FAIL (`volumeShadowDue` 4 yerine 3). Çalışma ağacında bu partiden önce
değişmişti; ayrıca bakılmalı.

## Latest build rejected: noisy fire / 398.7 s cold RT pipeline

PYRO_SOURCE_CHANNEL_HANDOFF_20261101.md başındaki güncel bölümü önce oku.
Kullanıcı derledi: fire ışığı aşırı gürültülü ve görsel kabul başarısız.
SceneLog: shader cache mismatch, RT pipeline ready in 398,7 s. Yeni ayarlar
rt_pyro_noise_startup_20261101.json içinde. Bu süre cold pipeline toplamı;
sampler'ın tek başına payı veya warm restart süresi ölçülmedi.
Runtime fire off driver compile yükünü kaldırmaz. Aday sayısını artırmayı
çözüm sayma; bounded ayrı compute lighting/cache yaklaşımı öneriliyor,
henüz uygulanmadı. Önceki 103 test GPU/derleyici performans kabulü değildir.

## Source density correction / clean handoff — 2026-11-01 (en önce)

PYRO_SOURCE_CHANNEL_HANDOFF_20261101.md: açık sahne 128 spp incelendi; ayarlar
rt_pyro_handoff_live_20261101.json içinde. Canlı eğri [0,1,1,1,1,1,1,1] emisyon
maskesini de sürüp yaklaşık raw density 0.0886 üstünü kaynaktan siliyordu.

Kaynakta düzeltildi: independent_emission açıkken emission/mask pre-curve;
extinction aynı smoke eğrisinde. Tek alan okumasından iki kanal, ek VDB lookup
yok. RT/NanoVDB/mixture/fire/preview ve boş alan kanıtları güncellendi.
103 non-build kontrol geçti; C++ fog bounds regression eklendi, derlenmedi.

1. Kullanıcı C++ ve tam shader paketini derlemeli; ABI hâlâ 688/172.
2. Aynı kare/ayarlar ile kaynak farkını kontrol et; açılan kaynak parlaklığı için
   emission'ı yeniden ayarlamak gerekebilir. Smoke eğrisini değiştirmek j/mask
   girişini düzleştirmemeli. Independent kapalı legacy aynı kalmalı.
3. Sıfır smoke eğrisi / multiplier, farklı maskeler, ordinary/mixture/preview
   kontrolü. Shader build ve GPU kabulü agent tarafından yapılmadı.
4. Handoff notundaki kalan limiter, yüksek extinction, ilk saçılma sınırı ve
   erken tükenen aday bütçesini izle. Yeni geniş performans matrisi tekrarlama.

## Bağımsız emisyon + alıcıya göre kaynak seçimi — 2026-11-01 (en önce)

PYRO_INDEPENDENT_EMISSION_20261101.md. Kaynakta hazır; agent build/uygulama
açılışı/canlı sahne düzenlemesi veya yeni performans benchmark'ı yapmadı.

1. Kullanıcı C++/CUDA ve tam shader paketini derlemeli:
   ./RayTrophiStudio/compile_shaders.bat -VolumeOptimization Off
   ABI hâlâ 688 byte/172 word. Response[0]: bit 0=response, bit 1=bağımsız j.
   Eski SPV flags 2/3'ü yanlış yorumlar; VolumeOnly ile yetinme.
2. Harici terminal: python scripts/test/rt_test_volume_fire_scatter_ipc.py --target "gas:Nuclear Gas"
   Yeni boolean, aday limitleri/atomiklik/restore. C++ scalar, material authoring
   ve fog bounds testleri güncellendi; agent derlemedi.
3. Panelde **Emission independent of smoke multiplier** aç. Katsayı parlaklığı
   eski yoğunlukla büyüyen S ile aynı kalmak zorunda değil: emission exposure/
   intensity'yi bir kez ayarla. Sonra sabit tutup smoke multiplier 1/10/30/0 dene:
   j büyümemeli; görüntü soğurmaya göre değişmeli. Sıfır extinction'da emisyon
   kaybolmamalı. Kayıt/aç ve preset korunmasını kontrol et.
4. **Fire lights smoke (Vulkan RT)** açık, random walk kapalı, 8 adayla başla.
   Yerel seçim en fazla dörder aday tüketir; toplam ışın bütçesi authored 1..32.
   32 aday her kamera adımında yeniden üretilmez. Katı/duman gölgesini ve iki
   farklı kaynağı kontrol et; aday sayısı enerji çarpanı gibi davranmamalı.
5. Independent j yayıcıları eski S radiance cache'ini ve aligned shared walk'ı
   bypass eder; maliyet artışı mümkün. RT/RayFusion default'ları değiştirilmedi.
   Legacy internal integral yeni j kipinde çalışmaz.

97 non-build test + light-cache ABI kontrolü geçti. GPU ve derleyici kabulü bekliyor.

## İlk saçılma build sonrası — 2026-11-01

`PYRO_FIRST_SCATTER_LIVE_20261101.md`: kullanıcı derledi; yeni kontrol ve
yaklaşık 32 ek kaynak density okuması çalışıyor. Aynı kamera/kare/post ile
off/new/legacy karşılaştırması yapıldı, bütün ayarlar geri alındı. Performans
benchmark'ı tekrarlanmadı. İstenen koyu kabuk/iç parıltı henüz kabul edilmedi.
Doğrudan emisyon hâlâ smoke multiplier ile büyüyor. Sonraki kaynak işi ortak
bağımsız j semantiği ve alıcıya göre kaynak seçimi; sadece aday artırma değil.
RT~32 ve RayFusion>=128 kullanıcı gözlemi; paylaşılan default 32'ye çevrilmedi.

## Yeni bulgu — 2026-11-01

`PYRO_MARCH_DENSITY_FINDINGS_20261101.md`: kullanıcı aynı patlamada 30/256
raymarch kalitesi arasında görünür fark olmadan yüksek maliyet farkı buldu.
Yeni benchmark yapılmadı. Sabit/adaptive baseStep bütçeye bağlı; emisyon guard'ı
mixture'da coarsening'i ayrıca sınırlar. Mevcut açık binary'de yeni fire kontrolü
yok. Density multiplier emisyon response'undaki density_power'a da giriyor;
density artınca ışığın artması yeni ilk-saçılma kodundan önce mevcut.
İlk-saçılma kabulünde bu ayarları sabit tut; 256'yı zorunlu kalite sayma.

## Alevden dumana ilk saçılma — 2026-10-31 (en önce)

`PYRO_FIRST_SCATTER.md`. Yeni source-space RIS yolu kaynakta, default **kapalı**.
Tek hacim ve genel mixture kamera march'ı; Vulkan RT. Random walk kapalı olmalı.
Agent build, shader compile, uygulama açılışı veya yeni performans ölçümü yapmadı.

1. C++ ve tam shader paketini kullanıcı derlemeli:
   `.\RayTrophiStudio\compile_shaders.bat -VolumeOptimization Off`.
   ABI hâlâ **688 byte / 172 word**; density_curve[11] artık aday sayısı.
2. Harici terminal: `python scripts/test/rt_test_volume_fire_scatter_ipc.py --target "gas:Nuclear Gas"`.
   İki ayarı restore eder; sahne kaydetmez. C++ `volume_material_authoring_test.cpp`
   yeni kontrollerin kayıt/legacy/default/atomiklik testlerini de içerir; agent derlemedi.
3. Patlama, aynı kare/kamera/post: random walk off, internal_light_steps=0.
   **Fire lights smoke (Vulkan RT)** off/on, 8 adayla karşılaştır. Doğrudan
   emisyon ve extinction tanımları korunurken çevre dumanda ışık katkısı aranır.
4. Sahne ışıkları olmadan da çalışmalı; opak katı/duman katkıyı kesmeli;
   kaynak emisyonu veya receiver scatter=0 olduğunda ilgili katkı gitmeli.
   Emisyon maskesi ve kare değişimi aynı kaynağı izlemeli. 1/8/32 aday
   enerji çarpanı gibi davranmamalı. Noise ve kalite henüz GPU'da doğrulanmadı.
5. Volume Graph olan gas sahnelerinde bu ilk sürüm katkı üretmez. OptiX/RayFusion
   uygulaması yok. Kalıcı cache veya tam multiscatter çözümü eklenmedi.
   Yeni yol açıkken legacy iç integral aynı kaynakta çalışmaz; kayıt alanları tutuldu.

## Density emission mask — 2026-10-31 (en önce)

`PYRO_DENSITY_EMISSION_MASK.md`. Maske kaynakta hazır, default **kapalı**.
Agent build veya uygulama açılışı çalıştırmadı; uygulama kapalı olduğu için canlı kabul yok.
ABI değişti: VkVolumeInstance **688 byte / 172 word**. Önceki 672 bilgisi tarihsel.

1. Studio kapalıyken C++/CUDA ve **tam** shader paketi derle:
   `.\RayTrophiStudio\compile_shaders.bat -VolumeOptimization Off`.
   VolumeOnly yeterli değil: raygen, closesthit, intersection ve preview/SDF raw
   okuyucuları da yeni stride'ı kullanmalı. Eski binary/SPV karışımı çalıştırma.
2. Harici terminal: `python scripts/test/rt_test_volume_emission_mask_ipc.py --target "gas:Nuclear Gas"`.
   Dört alan round-trip, yanlış tip/aralık reddi, atomiklik ve eski malzemenin restore'u.
   Scalar C++ testi `volume_emission_mask_test.cpp` kullanıcı tarafından build/run edilmeli.
3. Aynı patlama/kare/kamera/128 spp: önce mask kapalı görüntüyü al. Sonra panelde
   **Density emission mask** aç (Independent emission response açılması gerekmiyor).
   Center 0.2, width 0.1, strength 1 başlangıç denemesi; doğru eşik sahneye bağlı.
   Yoğun loblar kararmalı, aralıklarda sıcak alev kalmalı. Alev kaybolursa center yükselt.
4. Duman silueti ve geçirgenliği aynı kalmalı. Maskeyi açınca duman kayboluyorsa
   yanlışlıkla extinction maskeleniyordur. Kamera/zemin aydınlatması aynı kaynağı izlemeli;
   maskelenen lob çevreyi hâlâ eski güçte aydınlatıyorsa cache veya NEE bayattır.
5. Multiplier değiştirilince maskenin kendi density eşiği kaymamalı; genel optik
   kalınlık/parlaklık yine değişir. Mask center/width güncellemesi eski cache göstermemeli.
6. İki uyumlu Matter gazında maskeyi aç: ayrı slotlar ve nedeni görünmeli. Kapat/strength=0:
   gizli mask ayarları farklı olsa da diğer optikler uyumluysa birleşme geri gelmeli.
   Kaydet/aç ve preset değişiminde mask ayarları korunmalı. RT/OptiX/viewport/CPU kontrolü yap.
7. Vulkan maliyeti: adaptifi geçici kapatıp harici terminalden
   `python scripts/test/rt_probe_pyro_thermal_cost_ipc.py --target "gas:Nuclear Gas"`.
   Probe termal ve constant sıfır emisyonu ABBA karşılaştırır; sonra ayarları geri yükler.
   Çıktı termal yolun toplam fiyatıdır, tek fetch maliyeti değil. Testten sonra adaptifi
   önceki değerine geri koy. Yeni maskenin 2x backend farkını çözdüğü henüz ölçülmedi.

61 statik test ve light-cache sözleşme kontrolü geçti; C++/GPU yürütme doğrulanmadı.

## Pyro optik veri yolu 1: sıcaklık birimi — 2026-10-31 (en önce)

Plan: `PYRO_OPTICAL_PIPELINE_PLAN.md`. Bu parti yalnız ilk aşamadır. C++/CUDA
ve shader derlemesi gerekir; agent derleme veya yeni canlı kabul çalıştırmadı.
VkVolumeInstance stride değişmedi (672); density_curve[10] artık sıcaklık birimi.
Eski binary ile yeni shader veya tersi kullanılmamalı.

1. Studio kapalıyken C++/CUDA build ve tam shader paketi:
   `.\RayTrophiStudio\compile_shaders.bat -VolumeOptimization Off`.
   `-VolumeOnly` bu parti için yeterli değil: material_preview_volume.frag da değişti.
2. Harici terminalden:
   `python scripts/test/rt_test_volume_temperature_units_ipc.py --target "gas:Nuclear Gas"`.
   Hedef farklıysa `volume.list_materials` çıktısını kullan. Test ayarları geri yükler;
   geçerli 0/1/2 round-trip ve geçersiz değerlerin atomik reddini doğrular.
3. Açık sahne frame 69, aynı kamera/128 spp: soğuk pus ve kırmızı/beyaz yanlış kenarlar
   azalmalı; sıcak çekirdek kalmalı. Tam alev derinliği/HDR iyileşmesi henüz beklenmez:
   ramp endpoint ve 64 limiti sonraki aşamalardır. NEE artık kamera ile aynı emisyon
   değerlendiricisini ve limiti kullanır; çevre aydınlığında fark olabilir.
4. VDB paneli Temperature units: Source default (0), Kelvin (1), Normalized range (2).
   Live sim source default Kelvin'dir; ithal birimi bilinmeyen VDB legacy kalır.
   Normalize 0..1 ithal VDB'de 2 seç; kaydet/aç ve preset sonrası değer korunmalı.
   Kelvin/Normalized modunda eksik veya sıfır sıcaklık yoğunluk üzerinden parlamamalı.
5. RT / viewport / CPU / CUDA yollarında aynı birim anlamını kontrol et. Kelvin 20 ve
   20.01 sıcaklıkları artık farklı birim dallarına düşmemeli. Eski importer/malzemeler
   için legacy 0 kontrolü de yapılmalı. Son sıcak pikselin kaybolması yanlış metadata
   veya yanlış kaynak birimi belirtisi olabilir; cutoff yükselterek gizleme.
6. Birleşik Matter akışlarında birim ayarını değiştir: uyumsuz akış ayrılmalı; iki
   kaynak tekrar uyumlu yapılınca birleşmeli. Yeni birimle cache bayat kalmamalı.

Standalone scalar test: `scripts/test/volume_temperature_mapping_test.cpp` kullanıcı
tarafından C++17 ile build/run edilmeli. NaN/Inf, eksik/soğuk alan, normalize ve
Kelvin sürekliliği/monotonluğu kapsanır. Agent bunu derlemedi.
Non-build: yeni temperature sözleşmesi 7, material 10, optical bounds 5 kontrolü ve
light-cache sözleşme kontrolü geçti.

## VDB panelinin domain malzemesine bağı — 2026-10-31 (en önce)

C++ build gerekir; agent build veya uygulama çalıştırmadı. Shader değişmedi.
`getShader` düzeltmesine rağmen panel `getOrCreateShader` ile nötr render kopyasına
gidiyordu. Artık ikisi de yazılı kaynak malzemesini verir; canlı malzeme ataması
domain'in pointer bağını korur. Kırpma anahtarı saçılma/soğurma/emisyon türünü de izler.

1. Sahneyi duraklatılmış aç; hiçbir gizle/göster veya cutoff düzenlemesi yapmadan
   slot sayısı, kutu ve GPU süresini kaydet. İlk yayın da ince blok kırpmasından geçer.
2. Birleşik VDB panelindeki multiplier/cutoff ile aynı kaynağın
   `volume.get_material` (`matter:<domain>:<field>`) değerlerini karşılaştır.
   Panel düzenlemesi iki hedefte de aynı görünmeli; kare değişince ve kaydet/aç sonrası kalmalı.
3. Cutoff sürüklemesini bırakınca kutu yenilenmeli. Saçılma/soğurma ve emisyon None
   geçişlerinde de kırpma yenilenmeli. Yalnız bir kaynak optiği değişirse birleşme ayrılabilir.
4. Birleşik hacmin paneli yalnız adında gösterilen kaynağı düzenler; açıklama görünmeli.
   Diğer kaynak ayarını ayrıca doğrula. Görsel fark ve ince taban buharı kaybını kontrol et.

Statik kontroller: `python scripts/test/check_fog_optical_bounds_contracts.py` (5),
`python scripts/test/check_volume_material_contracts.py` (10) geçti.

## İnce buhar, Solid çökmesi, birleşme durumu — 2026-10-31 (önce bu)

★ **Ek:** panelde cutoff değişince kutu gizle/göster olmadan hemen küçülmeli. Küçülmüyorsa yayın isteği çalışmıyor.
Sürüklerken yayın olmaz: slider ve eğri kesilmemeli, kutu fare bırakılınca güncellenir. Bırakınca güncellenmiyorsa istek kaçıyor.

C++ derlemesi gerekir, shader değişmedi. Önceki iki partinin (panel, yayın isteği) kontrolleri de bununla gelir.

1. **Solid + Play.** Uygulama kapanmamalı. Gaz durursa `inventory_error` dolu demektir: reset ister, doğru davranış.
   Kapanırsa `StartupCrash.log` son `[FATAL]` satırına bak.
2. **Durum satırı.** Gaz paneli "Gas render: ..." göstermeli; emisyon farklıyken sebebi yazmalı.
3. **İnce buhar.** `Physics Domain 1` frame 52: kutu üstü 4.19'dan belirgin inmeli, süre ~210 ms'den düşmeli.
4. **★ Sessiz hata:** tabandaki ince buhar katmanı kaybolursa eşik fazladır (`kInvisibleBlockOpticalDepth`).
   Hata vermez, kimse bug diye raporlamaz.
5. **Etkisizlik.** Süre değişmediyse tutamlar eşikten kalındır ya da gaz emissive'dir; kırpma atlanır.

## Birleşik gazda malzeme sıfırlanması — 2026-10-31 (önce bu)

★ **Ek:** duraklatılmışken panel düzenlemesi artık birleşmeyi yeniler (`matterFogMergeChanged`). Uyumsuz
malzemede slot ikiye çıkmalı; açılışta uyumlu akışlar kare beklemeden birleşmeli. `scene_data.h` değişti,
tam derleme bekle. Yayın her karede tekrarlanıyorsa (FPS düşer) istek kilidi çalışmıyordur.

C++ derlemesi gerekir, `VDBVolume.h` değişti, geniş derleme bekle. Shader değişmedi.

1. **Panel doğru değeri gösteriyor.** Birleşik sahnede gazı seç: çarpan gerçek değerde olmalı, 1.0 değil.
   1.0 görünüyorsa panel hâlâ nötr kopyayı okuyor.
2. **Ayar kalıcı.** Çarpanı değiştir, başka kareye geç, geri dön: değer kalmalı, görüntü de değişmiş olmalı.
3. **Renk değişimi.** Yalnız bir akışın rengini değiştir: gaz slotu ikiye çıkmalı, maliyet artar. Bu doğru.
4. **★ Sessiz hata:** panelde değer kalıyor ama görüntü değişmiyorsa birleşme anahtarı yenilenmiyordur.

## Gaz akışı birleştirme aşama 1 — 2026-10-30 (önce bu)

★ **Canlı sonuç:** 1–3 ve 5 tamam: 262 → ~135 ms. **Madde 4 (görüntü) gözle yapılmadı**; JPEG farkı p99 19.
Açılışta birleşme ilk yeniden yayına kadar gecikiyor.

C++ derlemesi gerekir, shader değişmedi. Not: `RT_GAS_STREAM_MERGE_20261030.md`.

1. **Sahne bozulmamalı.** İki gazlı sahneyi aç: A ve B materyalleri farklıysa `render.volume_slots`
   hâlâ iki gaz göstermeli. Biri kayıpsa birleştirme yanlış tetiklenmiştir.
2. **Birleşme.** B'ye A'nın emisyon ayarlarını ver (`volume.set_material`, `ramp_enabled`,
   `color_ramp`). Slotlarda tek gaz kalmalı, `volume.get_material` iki hedefi de açmalı.
3. **Süre.** Küplü sahnede yaklaşık tek akış bandı (~131 ms) beklenir; iki akış 240–280 ms.
4. **★ Sessiz hata:** görüntü A+B ile aynı kalmalı. Gaz belirgin koyulaşır veya açılırsa eğri iki kez
   işleniyor ya da hiç işlenmiyordur; hata vermez.
5. **Görünürlük.** B'yi gizle: birleşme bozulup tek A kalmalı, gazın B payı kaybolmalı. Geri aç.

## Sekme ışını: radyans önbelleği yürüyüşü — 2026-10-30 (önce bu)

★ **Son ölçüm:** A+B 271 → 239 ms (adım 2), 224 ms (adım 4), notun §10.3'ü. Madde 3 hâlâ yapılmadı.

★ **Bu derlemede yeni:** yürüyüş `secondary_step_scale`'e uyuyor (notun §10.2'si). Aynı süreçte
`secondary_step_scale` 1 ve 2 ile A+B ölç: fark sekme yürüyüşünün payıdır. ★ Görsel: küpteki gaz
yansıması koyulaşırsa veya bantlanırsa kaba adım fazla gelmiştir, kimse bug diye raporlamaz.

★ **Canlı sonuç:** 1–2 tamam: A −%11, B −%27, A+B 252–323 ms; sekme hâlâ ~130 ms (notun §10.1'i).
Madde 3 (görüntü) yapılmadı.

Yalnız shader. `.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off`,
aynı küplü sahne, frame 99, duraklat. Notun §10'u.

1. **Sayaçlar.** `radiance_cache_marches` ~0.14, kamera ~12.5: değişmemeli. `radiance_cache_fallbacks` 0.
2. **Süre.** A, B, A+B ve `max_bounces=1`. Aynı oturumda oranla oku. A+B 300'ün belirgin
   altına inmeli. İnmediyse kök boş adımlar değil, dolu adımların 32 okumasıdır.
3. **★ Sessiz hata:** küp üstündeki gaz yansıması koyulaşır veya parlarsa `vrcProvenEmpty`
   yayınlanmamış köşeyi boş saymıştır. Görüntü önceki gibi kalmalı.

## İki gaz adım maliyeti — 2026-10-30 (önce bu)

★ **Canlı sonuç:** 1–3 tamam: oran ~1.8× (başlangıç 3.2×), notun §9.1'i. Süreleri bantlar
arasında kıyaslama, oran kullan. **Madde 4 (renk) ve önceki partinin görsel maddeleri yapılmadı.**

Yalnız shader. `.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off`,
sonra iki gazlı sahneyi aç (legacy, frame 99, duraklat, Rendered). Notun §9'u.

1. **Pipeline hazır, SHA aynı.** Değilse sonraki sayılar eski shader'ındır.
2. **Sayaçlar.** `camera_density_samples` ve `shadow_density_samples` önceki düzeyde kalmalı
   (~10–11 ve ~4.3), `majorant_queries` 3.5–3.9'dan belirgin düşmeli. Düşmediyse yaprak
   önbelleği çalışmıyor; ortak yürüyüş açılmıyor olabilir.
3. **A, B, A+B matrisi, kare dizileriyle.** A ve B değişmemeli, A+B ~80 ms'nin altına inmeli.
   Yalnız medyana bakma, iki bant varsa tur sayısını artır.
4. **★ Sessiz hata:** görüntü aynı kalmalı. Birim terim yanlış alana bağlanırsa iki gazın rengi
   yer değiştirir, hata vermez. A ve B farklı malzemeyle bir kez bak.

## Tek gaz okuma maliyeti — 2026-10-30 (önce bu)

★ **Canlı sonuç:** madde 1 tamam (sayaçlar aynı). Madde 2 hükümsüz: süre örnek sırasına bağlı,
sample 8–23 medyanı bu sahnede sağlam ölçü değil (notun §8.3'ü). Madde 3 ve 4 yapılmadı.

Yalnız shader değişti. `.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off`
ardından `nuke.rtp` aç, frame 94'te duraklat, Rendered. Hash'ler açılışla değişebilir
ama bu sahne kendini iyi tekrarlıyordu (412–415 ms).

1. **Sayaçlar kıpırdamamalı.** `camera_density_samples` ~36.6, `light_cache_hits` ~10.3,
   `light_cache_misses` ~0.07. Hit düştüyse tutulan okuma kaydı kırılmış; miss arttıysa
   anahtar yanlış. Bu değişiklik iş sayısını değil, iş fiyatını düşürür.
2. **GPU süresi.** Taban 412–415 ms. Düştüyse okuma payı doğrulanır. Düşmediyse okuma
   fiyatı sanıldığı gibi bellek değildir; `shadow_steps=0` ile yeniden ayır.
3. **Görüntü aynı kalmalı.** ★ Sessiz hata: sıcaklık ışını kayarsa ateş topu biraz başka
   renkte ama makul görünür. Eski ekran görüntüsü yok; ramp sınırlarını ve gaz kenarını
   önceki izleniminle karşılaştır, şüphede `sampleTemperature` ile A/B yap.
4. **İki gazlı sahnede tekrar.** A, B, A+B. A+B ~75–80 ms bandında veya altında olmalı.

## nuke.rtp + Play çökmesi — 2026-10-30 (önce bu)

Kök bulundu: `Gas source provenance exceeds 8 emitters / 256 MiB budget`. 166x256x166
= 7.05 M hücre, tek emitter için bile sabit bütçenin üstünde; istisnayı yakalayan kimse
yoktu. Tavan söküldü, reddedilen kaynak artık domain'i hold eder. `Main.cpp` log
koruması da kaldı. C++ derlemesi gerekir (shader'lar daha önce derlendiyse yeterli).

1. `nuke.rtp` aç, Play. Görmen gereken: gaz kaynağı çalışır, uygulama kapanmaz.
   Bozuksa: `StartupCrash.log` son `[FATAL]` satırı artık gerçek hatayı söyler.
2. `render.volume_stats` ile gerçekten gaz yürüdüğünü doğrula: yoğunluk artıyor.
3. ★ Sessiz başarısızlık: uygulama açık kalır ama gaz hiç büyümez. Bu hold'dur;
   domain için `gas_stats.inventory_error` dolu demektir (reset ister).
4. Düşük VRAM'de `GasSourceCurrent` ayırma hatası hâlâ istisna atar ve uygulamayı
   kapatır: yalnız kaynak bırakma noktası korundu, adım içi `require` korunmadı.

## İki gaz çakışma maliyeti — 2026-10-30

★ **Canlı sonuç (tau bütçesi öncesi):** 1–3 tamam. A+B ~77 ms (önce ~112), A ve B
değişmedi. **4–7 görsel maddeleri henüz yapılmadı.**

★ **Tau bütçesi geri alındı:** A+B ~80 ms, gölge örneği −%33, süre kazancı gürültüden
ayrılamadı (notun §7.1'i). Kaynaktan söküldü; kurulu shader hâlâ onu içeriyor.
Aşağıdaki komutla yeniden derle, sonra 1–3'ü koş. Beklenen: gölge örneği yeniden
~4.3/yol, A+B ~75–80 ms bandı. Gölge örneği 2.9 civarında kaldıysa eski `.spv`
ölçülüyordur. ★ Görsel maddeler (4–7) bu derlemeyle yapılmalı.

Ayrıntı: `RT_GAS_OVERLAP_COST_20261030.md`. Yalnız shader değişti, C++ derlemesi
gerekmez. Studio kapalıyken:

```powershell
.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off
```

Aynı sahneyi legacy modda aç. Gaz hash'leri açılıştan açılışa değişebilir; bu
yüzden ölçümler kendi içinde karşılaştırılır, önceki 112 ms ile değil. Sıra:

1. **Pipeline kurulu mu.** `render.rt_pipeline_status` `ready`, `volume_closesthit`
   mevcut, kaynak ve `x64/Release/shaders/volume_closesthit.spv` SHA'sı aynı.
   Değilse: eski shader ölçülüyor, aşağıdaki hiçbir sayı geçerli değil.
2. **Sayaçlar çalışıyor mu (GPU süresinden bağımsız, hızlı).** Tek geçiş A+B:
   `camera_density_samples` 13.8'den belirgin düşük (beklenti ~8–10/yol),
   `shadow_density_samples` 6.9'dan düşük (~2–3/yol), `step_budget_exhausted`
   0.113'ten düşük, `light_cache_misses` warm 0. Kamera örneği değişmediyse
   `mfRefinedSpan`/`mfCellCurveSlope` etkin değil; gaz karışım yoluna girmiyor olabilir.
3. **A+B/A/B/gizli matrisi.** `rt_probe_volume_split_cost.py` ile A+B alıp onu
   `rt_probe_two_gas_cost_matrix.py --initial-both` ile ver. Görmen gereken: A ve B
   tek başına yaklaşık aynı (shader tek gaz yolunu değiştirmedi; değiştiyse bir
   include sızmıştır), A+B çok daha düşük. Hedef çoğu 45–60 ms, referans 34.5 ms.
   80 ms üstü: izler veya kapılar beklenen yolda değil, 2. adıma dön.
4. **Dumanın ışığı (görsel).** Yarı saydam ince saçak, özellikle yoğun çekirdeğin
   uzak tarafındakiler. Gökyüzü kapısı burada görünürlüğü tutar; saçak parlıyorsa
   kapı fazla erken kapıyor.
5. **★ En sinsi: katı gölgesi kaybolmuş ama gaz makul görünüyor.** Çakışma bölgesinde
   gaz artık katının gölgesini almaz; bu bilinçli (aynı gazın tek başına davranışı),
   ama bug gibi rapor edilmez. Gaz içine/önüne katı koy, A+B ile A tek başına
   karşılaştır: ikisi aynı davranmalı ve çakışma sınırında dikiş olmamalı. Kabul
   edilmezse tek satır: `mfGeometryBlocks` içindeki `!mfWalking` erken dönüşü.
6. **Yoğun gaz arkası.** Opak gaz gövdesinin arkasından parlak emisyon/arka plan
   sızmamalı. 1e-3 eşiği 0.001 geçirgenlik bırakır, tek gazla aynı.
7. **Random walk, üç katkıcı, gazın hizalı olmadığı durum.** Görüntü önceki gibi
   olmalı: iz ve `sampleTemperature` geri düşüşü bu yollarda duruyor. Fark varsa
   `mfWalking` kapısı veya ortak yürüyüş bayrağı yanlış yere bağlanmıştır.

## Hacimsiz pipeline A/B canlı sonucu — 2026-10-30

Kullanıcı build'i ardından iki gerçek stage listesi IPC'den doğrulandı. Aynı
kamera/hedef, frame, render ayarları, 1680x945 ve sample 8–39 ile medyan:
volume dahil 99.145424 ms, volume hit grubu hariç 98.818768 ms (%0.329 fark).
Bu sahnede büyük genel RT maliyetinin salt volume grubunun bağlanmasından
kaynaklandığı hipotezi desteklenmedi. `-O0` karşılaştırması veya iki-gaz sorunu
çözülmüş değildir. Geçici kamera hedefi ve sayaçlar geri yüklendi. Normal
kullanım için test environment değişkeni kaldırılarak uygulama yeniden açılmalı.

## Hacimsiz sahnede volume pipeline A/B — 2026-10-30

C++ kaynakları hazır: `RT_VK_AB_NO_VOLUME=1` process-start test modu volume hit
grubunu pipeline hash'inden önce çıkarır. Diğer shaderlar aynı kalır; published
volume veya TLAS volume slotu varsa RT çizimi reddedilir. Normalde değişken yok/0
olmalı. Shader derlemesi gerekmez, yeni C++ build gerekir. Altı non-build test
geçti; canlı GPU kabulü yapılmadı. Aynı kaydedilmiş hacimsiz sahne ve aynı binary
ile iki ayrı açılışta ölçüm alınmalı; mevcut gaz sahnesinde uygulanmamalı.
Komutlar: [RT_VOLUME_PIPELINE_AB.md](RT_VOLUME_PIPELINE_AB.md).

## Son compiler denemesi ve scalar validation düzeltmesi — 2026-10-29

Kullanıcı `-O` için uzun tek çekirdek çalışması ardından -1073741819 /
0xC0000005 (access violation) bildirdi. `-O0` derlemesi ardından gelen
`VkWorldDataExtended` offset 120 hatası, doğrulayıcıya scalar layout özelliğinin
iletilmemesiydi. Shader GL_EXT_scalar_block_layout gerektiriyor; backend de
scalarBlockLayout özelliğini açıyor. Validator artık `--scalar-block-layout`
kullanır; buffer ABI değiştirilmedi. İncelemede runtime'da eski teşhis anahtarının
scalarBlockLayout özelliğini false tuttuğu da görüldü; artık cihazın desteklediği
özellik isteniyor ve mevcut pNext zincirine ekleniyor. Bu düzeltme C++ derlemesi
gerektirir; pipeline oluşturma süresi ve görüntü ayrıca kontrol edilmeli.
Mevcut SPV scalar seçeneğiyle gerçek spirv-val salt okunur kontrolünden geçti.
Normal varsayılan `Off` / `-O0` oldu.
`-VolumeOnly` artık uzun `-O` denemesine girmez; Auto/Full yalnız açık seçimdir.
Optimizasyon denemesi performans sorununu çözmedi; GPU kabulü hâlâ açık.

## Hacim shader derleme optimizasyonu — 2026-10-29

`compile_shaders.bat -VolumeOnly` artık yalnız `volume_closesthit` için `-O`
dener, SPIR-V'yi `spirv-val` ile doğrular ve başarılıysa yayımlar. `Auto`
modunda compiler/validation hatası bir kez `-O0` fallback yapar; yayınlama
hatası fallback ile gizlenmez. Her iki derleme başarısızsa eski SPV korunur.
`-VolumeOptimization Full` fallback yapmaz; `Off` eski `-O0` referansıdır.
Log'daki son `OK ... [-O ...]` veya `[-O0 ...]` gerçek yayımlanan seçimi gösterir.
VolumeOnly diğer shaderları yeniden derlemez, dağıtmaz veya silmez. Agent gerçek
derleme çalıştırmadı; mock hata yolu ve salt kaynak planı kontrolleri yapıldı.
Yeni shader uygulamada yüklenmeden GPU karşılaştırması geçerli değildir.

## Gas RT ortak hücre yürüyüşü — son kaynak partisi, 2026-10-29

Tam hizalı iki Matter gas alanı için ortak kamera hücre yürüyüşü eklendi.
Kaynakların yoğunluk eğrileri, sıcaklığı ve optik katkıları ayrı korunuyor.
Genel ve hızlı yol aynı optik yanıt modülünü kullanıyor. Üçüncü alan, hizasız
alanlar, grafik/noise, random walk veya adaptif kapalı durumda genel yol sürer.
Bu parti RT shader derlemesi gerektirir; GPU hız/görüntü kabulü henüz yapılmadı.
43 birim testi ve iki ek kaynak denetimi geçti. Ayrıntılar ve kabul listesi:
[GAS_RT_SHARED_CELL_WALK.md](GAS_RT_SHARED_CELL_WALK.md).

## Gas RT maliyeti — 2026-10-29

İki görünür gas alanının adaptif mixture yolunda aynı voxel için tekrarlanan
NanoVDB köşe/sınır hesapları ışın başına önbelleğe alındı. Kalite ayarları
azaltılmadı. Kullanıcı RT shader derlemesinden sonra mevcut frame 89 / Pause
sahnesinde GPU süre ve görüntü eşliği kabulü gerekli; henüz hızlanma ölçülmedi.
Ayrıntı: [GAS_RT_MIXTURE_CELL_CACHE.md](GAS_RT_MIXTURE_CELL_CACHE.md).

## En son ana kaynak değişiklikleri — D2 / T6 E2/E6, 2026-10-29

Son başarılı kullanıcı build'inden sonra RAM/disk cache frame preflight
eklendi. Yeni C++ derlemesi gerekir; shader revision hâlâ 33. Kullanıcının
son yönlendirmesi: ana plan kaynakları önce, hassas fizik ölçüm/tuning en son.
Toplu kabulde iki sistemli cache'de ikinci domain'in lane/count veya frame
dosyası bozuk olduğunda ilk domain/MSF yarım restore edilmemeli, disk cache
valid=false ve rebake log'u görülmeli. Eşleşen RAM/disk frame restore ve
legacy project roundtrip de aynı turda yeniden kontrol edilmeli.
İki loader'ın ortak saved-type parser'ı da bu partide: Gas/Fluid/Matter ve
0/1/2 korunmalı; bilinmeyen tip artık Gas'a sessiz dönüşmemeli. Eksik eski
type alanının Gas varsayılanı değişmedi. Karşılaştırmalı C++ test kaynakları
eklendi; canlı dosya kabulü son toplu tura bırakıldı.
Loader ilk synchronize çağrısı artık bütün flow source kayıtlarından sonra;
aksi halde legacy type erken kayboluyor, kaynak phase göçü atlanıyordu.
Legacy roundtrip'in iki kapalı/yanlış-faz kaynağı yeni build'de doğru faza
dönmeli; ikinci save/open'da da değişmemeli.
D2 kuvvet maliyeti düzeltmesi de C++ partisinde: yüzey Wind/atmosfer rüzgârı
yoksa kolon clear/build ve kolon okuması atlanır. Son kabulde body Gravity/
Vortex alanı, surface-drag Wind ve atmosfer rüzgârı davranışları ayrı kontrol
edilmeli; süre karşılaştırması aynı sahne/sayılarda yapılmalı. Shader ABI ve
revision bu değişiklikte değişmedi. `fluid_force_columns_test.cpp` ilgili
bağımsız C++ kontrat kaynağıdır (ajan derlemedi).
Nested-clock yuvarlama artık authored DEM limit'i aşamaz: örnek 512 istek,
40 continuum, limit512 →64 continuum x8 DEM. Normal sınır içi zamanlama aynı.
Son mixed_clock/4-emitter probunda DEM=continuum*per yanında DEM<=limit de
doğrulanır. `matter_common_clock_test.cpp` bağımsız exhaustive oracle içerir;
Python aritmetik oracle uygulamasız geçti, C++ derlenmedi. Yeni shader yok.

## Yeni kullanıcı derlemesi canlı kabul — 2026-10-29

Kullanıcı "derledim açık" dedi. Başlangıç: boş domain/source, timeline mode,
Pause/frame0. Dokuz seçili smoke kolu sırayla dış IPC ile koşuldu: **7/9 PASS**.

- matter_lanes 7/7, simulation_transport 8/8, substance, P9 melt 4/4,
  P9 floor 2/2, MPM+DEM clock 5/5 ve run_readback geçti.
- Snow-only artık 300 parsel, COM y=0.18331 m; mixed 600 parsel ve inferred
  Snow y~0.185 m. Ölçülen schedule 40 continuum x2 DEM=80, grid=40.
- Floor min-y=0.209966 m, 960 buz yüzü, step_held=false. Status temiz:
  önceki `density position projection failed` artık yok.
- Açık: seed_settle liquid_at_rest (4/5; Water p95=4.311 mm/kare,
  Sand=0.1056 mm/kare). Ritter 0.3 hız=1.1093 m/s, hata=%54.28;
  0.9 hız=1.9494 m/s, hata=%19.65; ideal referans=2.4261 m/s.
  İki doğruluk kapısı ve 0.3 nonregression geçmedi. **0.3'e geçiş reddedilir**;
  0.9 da bu deneyde kabul doğruluğunu sağlamadı. Built-in değiştirilmedi.
- Son durum: Pause/frame0, timeline mode; domain/source/forcefield boş.
  Loglar `C:/Users/maxkemal/AppData/Local/Temp/rt_matter_smoke/`.
  Bu tur full suite, legacy save/open veya 4-emitter performans kabulü değildir.

## Son testten sonraki kaynak partisi — 2026-10-29

**Bu parti C++ yanında simülasyon shader derlemesi de ister (kullanıcıda).**
DEM host/shader revision 33: history_blocks kapasite koruması 8 kelimelik
yerleşime düzeltildi. `sim_matter_grain_list_clear/hash/list_build/step`
çıktıları aynı güncel GLSL'den gelmeli; eski revision 32 SPIR-V kabul edilmez.

Kullanıcı derlemesi sonrası boş sahne/Pause üzerinde dış IPC ile sırayla:

Önce `python scripts/test/rt_test_simulation_transport_ipc.py`: gerçek kare
döngüsünde Play ilerlemeli, Pause kareyi tutmalı, Live'da playhead sabitken
kaynak parçacık üretmeli. Test fluid.step/run kullanmaz; gelen mode'u geri
yükler. Bu test full smoke'da `simulation_transport` adıyla kayıtlıdır.

1. `python scripts/test/rt_test_run_readback_ipc.py` — son sample/dış okuma
   eşitliği, pozitif kütle, manual claim ve ilerleyen timeline frame.
2. `python scripts/test/rt_test_mpm_dem_shared_clock_ipc.py` — Snow/Sand/mixed
   sayılar ve düşüş; run sample ile dış digest uyuşmazsa fizik yorumu yapılmaz.
3. `python scripts/test/rt_test_substance_profiles_ipc.py` — Sand kapasitesi 0.1.
4. `python scripts/test/rt_test_p9_matter_ipc.py --arm floor` — buz teması ve
   status'ta density projection failure olmaması ayrıca kontrol edilmeli.
5. `python scripts/test/rt_test_p9_matter_ipc.py --arm melt` — ilk kütle sıfır
   olmamalı ve final conservation kapısı geçmeli.
6. `python scripts/test/rt_test_ritter_flip_ipc.py` — türetilmiş Water 0.3/0.9
   karşılaştırması, 0.4 s kuru yatak salımı. Her kolun son cephe hızının
   `2*sqrt(9.81*0.15)` referansına bağıl hatası <=%15; 0.3 hata payı 0.9'dan
   en çok 5 yüzde puan kötü olabilir. Çıktı sistem TEMP/rt_ritter_flip.json.
   Bu sonlu derinlik/çözünürlük deneyi tek başına genel solver doğrulaması
   değildir; rest/column kapılarıyla birlikte preset kararı verilir.
7. `python scripts/test/rt_test_matter_legacy_roundtrip_ipc.py` — yalnız boş,
   harcanabilir projede: TEMP dosyalarına save/open yapar, aktif proje yolu
   değişir. Synthetic legacy Gas+Fluid göçü, kapalı kaynak fazı, tek grid, gaz
   shader korunumu, canonical save/open. Gerçek tarihsel dosya kabulü ayrıdır.

Toplu son tur (boş proje, Pause): `python scripts/test/rt_matter_smoke.py`.
Yeni matter_lanes, grain_sleep ve ritter_flip kolları full turda kayıtlıdır;
quick tur kapsamı aynı kalır. Legacy proje turu yalnız açıkça
`--with-legacy-project` eklenirse çalışır; harcanabilir proje gerekir. Tekil
örnek: `python scripts/test/rt_matter_smoke.py --with-legacy-project --only legacy_project`.
Phase-grid read-only probe ise authored domain adı ister, boş-scene otomatik
suite'e eklenmedi. 4-emitter maliyet ölçümü kendi sahnesinde ayrı koşulur:
`python scripts/test/rt_probe_grain_clock_ipc.py --steps 204 --every 12 --require-clock`.
Clock kabulü: DEM=continuum*per, 0<grid_steps<=continuum ve hiçbir held kare
yok. Prob JSON'u tam host_ms/clock/contactless/transfer verisini, ayrı
`TEMP/rt_probe_grain_clock_summary.json` ise örneklenmiş medyan/p95 dökümünü
verir. İç içe süreleri toplama; IPC wall-clock ile solver süresini karşılaştırma.

Bu parti fluid.step/run için physics.step ile ortak manual timeline claim,
sıvısız density no-op GPU restore düzeltmesi ve Sand test beklentisini içerir.
Henüz derlenmedi/canlı kabul edilmedi. Son teşhis pipe error 2: bağlantı yok;
ajan uygulama açmadı. Liquid-at-rest, kalan D2 ve geniş T6 kabulü hâlâ açık.

## Kullanıcı derlemesi hızlı kabul — 2026-10-29

Boş sahne/Pause, dış named-pipe istemcisi. Build ajan tarafından yapılmadı.

- `rt_test_matter_lanes_ipc.py`: **7/7 PASS** (canonical identity, presets,
  absent grid, invalid/no-mutation, source guard, lane switch, auto phase).
- `rt_matter_smoke.py --quick --timeout 360`: **7/11 test PASS**.
  Thermal films 5/5, phase3 freeze, fire/steam 8/8, grain readiness 1/1,
  P9 floor 2/2, run readback 1/1 ve fast fall 2/2 geçti.
- Açık: substance Sand `grain_water_capacity_fraction == 0` beklentisi;
  seed_settle liquid_at_rest (su p95 4.31 mm/kare; tane kapıları geçti);
  p9_melt mass kapısı (`mass_after_pour_kg=0`, son 140.325079 kg; bu okuma
  fiziksel kütle kaybını kanıtlamaz); mpm_dem_clock 1/4 (snow-only digest 0,
  sand 300, mixed 600; boş Snow artık yanlış PASS sayılmıyor).
- Düz buz levhası: 4608 frozen parcel, 960 yüz, kum min-y 0.2099565 m,
  COM y 0.2120414 m; step_held=false. **Ek açık:** status'ta
  `density position projection failed` var. Floor gate geçti diye density
  pass kabulü verilmez; bu hata sonraki kaynak düzeltmesine girer.
- Son salt-okuma: Pause, frame 0, driver none; domain/source/forcefield listeleri
  boş. Loglar `%LOCALAPPDATA%/Temp/rt_matter_smoke/`.

Bu hızlı tur T0, tam legacy save/open ve uzun hacim/Ritter kabulü değildir.
Tur sırasında kaynak düzeltmesi veya uzun ayıklama koşusu yapılmadı.

## Rig: Human Walk v2 (W1–W4) + Rig Studio kalıntı sökümü — 2026-10-28 (yalnız C++, derlenmedi; madde partisiyle birlikte derlenecek)

### R1 otomatik işaret + R3 geodezik ağırlık (2026-10-29, derlenmedi; derleme planın sonunda)

**R1 otomatik işaret (2026-10-29, derlenmedi):** `Animation/RigAutoLandmarks`,
`rig.auto_landmarks` (salt okunur) + Python + Alignment penceresinde **"Auto place"**.
`get_fit_setup` ile aynı uzay ve biçim → doğrudan `preview_fit` / `commit_fit`'e girer.
Şablonun sınır kutusuna ölçeklenmiş halinden başlar. Humanoid: baş tepesi en yüksek
noktalardan, ayak ucu/top/bilek her taraftaki en alçak banttan, el ucu kalçanın
üstündeki en yanal noktadan (T ve A poz), dirsek/bilek omuz→el çizgisinde, diz kalça→
bilek çizgisinde; her eklem kendi kesitinin ortasına çekilir ve kesit yoğunluğu + ne
kadar kaydığıyla puanlanır. Parmaklar tespit edilen kol yönüne ve uzunluğuna taşınır,
bilerek .45 (eşiğin altı) — eller hep bir bakış ister. Diğer aileler: her eklem ebeveyn
kemiğine dik kesitte ortalanır. Mesh −Z'ye bakıyorsa (`forward_check`) tüm güvenler
.3'e kırpılır. Panelde düşük güvenliler **amber**; elle taşınan eklem listeden düşer.

**R3 geodezik voksel ağırlık (2026-10-29, derlenmedi):** `Animation/RigGeodesicWeights`
(Dionne & de Lasa 2013). Bağlı mesh'ler bind pozisyonlarında birlikte vokselleşir
(çözünürlük 32..256, varsayılan 96): üçgenler yüzey, dış kenardan taşma doldurma = dışarı,
kalan = iç. Her deforme eden kemik (envelope ile aynı segmentler, root hariç) yalnız
**katı** vokseller içinde 26 komşulu Dijkstra ile mesafe büyütür (boyun yarısına kadar);
vertex en yakın 4 kemiği tutar, ağırlık = max(d, ½ voksel)^-falloff, sonra ortak en
güçlü-4 sözleşmesi. Açık mesh taşar → yalnız yüzey kabuğu mesafe taşır
(`closed_volume` false), yine kullanılır. `rig.preview_geodesic_weights` (salt okunur) /
`rig.apply_geodesic_weights` (envelope ile **aynı** ağır geri alma komutu; algoritma
`geodesic_voxel_v1` projeye yazılır, yükleyici kabul eder; envelope yarıçapları ve
profiller korunur). Panel: Skin binding → envelope bölümünün altında "Geodesic voxel
weights". Ölçüm aracı: `scripts/test/rt_probe_rig_auto_ipc.py [--apply]`.

Sırayla:
1. `python scripts/test/rt_probe_rig_auto_ipc.py Rig` (sahneyi değiştirmez) →
   auto_landmarks satırı + geodezik önizleme. Bağlı rig'de fit hedefi zaten skinli olduğu
   için auto_landmarks **reddedilir** (tasarım: `target_already_skinned`) — R1'i test
   etmek için yeni bir humanoid mesh içe al, şablon rig oluştur, bağlamadan önce koş.
2. Yeni mesh'te: Alignment → Prepare → **Auto place** → amber eklemler az olmalı (T poz
   insan: çoğunlukla yalnız parmaklar). Önden ve yandan bak: bilek/dirsek/diz kol ve
   bacak içinde. Preview → `outside_bounds` 0. Bozuksa: el ucu kalçaya düşüyor → kol
   gövdeye yapışık (uyarı satırı söyler); her şey .3 → mesh −Z'ye bakıyor.
3. Geodezik: önizlemede `fallback_vertices` 0, `closed_volume` true (kapalı mesh),
   `bones_outside_mesh` küçük (parmak ucu kemikleri olabilir).
4. `--apply` ile koş → ROM önce/sonra satırları; `folded` ve `pinched` azalmalı ya da
   eşit. Uygulamada Ctrl+Z → eski ağırlıklar ve `rig.get_binding` eski algoritma.
   Kaydet → yeni .rtp → aç → `geodesic_voxel_v1` yükleniyor (yükleyici reddetmemeli).
★ Sinsi: çözünürlük düşükse (≤48) parmaklar aynı voksellere düşer ve birbirinin
ağırlığını alır — ROM'da parmak hareketleri "katlanma" gösterir, gövde temiz görünür.
Sayı kötüyse önce çözünürlüğü artır, falloff'a sonra dokun.
★ Süre: 96'da birkaç saniye UI donar (UI iş parçacığında koşar) — hata değil.

### Rig gizmosu + R2 ROM testi (2026-10-29, derlenmedi; derleme planın sonunda)

**R2 ROM testi (2026-10-29, derlenmedi):** yeni çekirdek `Animation/RigRomTest`
(vcxproj'a eklendi). Her uzuv zinciri için kök swing ±60 / spread ±45 / twist ±45, orta
eklem 110° bükülme (bükülme ipucundan **uzağa** katlanır: diz öne gider, baldır geriye),
uç eklem twist ±60 / flex ±45; her spine/neck/head rolü flex ±30, yan ±25, twist ±35.
Eksenler kemikten + bükülme ipucundan + ayaklardan ölçülen karakter çerçevesinden
kurulur, sabit eksen yok; yazılmış eklem limitleri her pozu kırpar (`limited`). Deri
CPU'da `pose · rest⁻¹` ile bind pozisyonları ve kayıtlı flat ağırlık satırlarından
hesaplanır; her geçerli üçgen için alan oranı: < .5 **sıkışma** (çökme, şeker ambalajı),
> 2 **gerilme**; normali baskın kemiğinin taşıyacağı yöne ters dönen üçgen **katlanma**.
`rig.preview_rom_test` (salt okunur, her modda) satır başına hareket, kemik, açı, en
kötü mesh/vertex/kemik ve klipteki tepe karesini verir; `rig.create_rom_clip` aynı
hareketleri dinlenim/tepe/dinlenim anahtarlarıyla klibe yazar (tek geri alma adımı).
Panel: Pose → Quick Motion → "Range of motion test" (kontrol / klip üret / en kötü 10
tablosu). Yürüyüşün klip yayınlama gövdesi ortak `publishAuthoredClip`'e taşındı.
Ölçüm aracı: `scripts/test/rt_probe_rig_rom_ipc.py`.

**Gizmo düzeltmesi (2026-10-29, derlenmedi):** kemik/rig gizmosu fare bir panelin
üstüne geçince **kayboluyordu** — `RigViewportUI` panel hover'ında çizmeden dönüyordu.
Artık IK gizmosu gibi: her zaman çizilir, panel fareyi tuttuğunda yalnız sürükleme
başlatılamaz.

Sırayla (bağımsız ve hızlı önce):
1. **Gizmo (gözle, 10 sn):** Edit veya Pose'da bir kemik seç, fareyi viewport'tan sağ
   dock'a ve geri gezdir → gizmo hiç kaybolmamalı. Panelin üstündeyken gizmonun altına
   denk gelen bir yere tıkla → panel tepki verir, kemik **hareket etmez**. Bozuksa:
   kaybolma sürüyor → derleme eski; panel tıklaması kemiği oynatıyor → sürükleme
   engeli çalışmıyor.
2. `python scripts/test/rt_probe_rig_rom_ipc.py Rig` → 4/4 PASS; tablo en kötü 10
   hareketi listeler. Bozuksa: `unknown_influences` > 0 → deri satırları bu rig'in
   indekslerini göstermiyor (binding/indeks remap sorunu, ROM değil); motions 0 →
   anatomide zincir/rol yok.
3. Panelde "Check deformation" → aynı sayılar; "Generate ROM clip" → `ROM_Test` seçili
   klip, oynatınca her eklem sırayla bükülüp dinlenime döner. En kötü satırın
   `frame_peak` karesine git → sıkışma/katlanma tabloda adı geçen kemikte görünmeli.
4. Kendi kontrolü: Envelope Editor'da kol kapsülünü daralt/genişlet → `min_area_ratio`
   değişmeli (ROM ağırlığa duyarlı).
★ Sinsi: diz `bend` hareketi ipucu olmayan eski rig'de karakter ileri yönünden
türetilir; dirsek de öne katlanır ve "kol yanlış bükülüyor" görünür. Önce
`rig.derive_limb_hints` (yukarıdaki v8 bölümü) — ROM raporu bunu kendisi söylemez.


**Şablon v4 / anatomi v8 (2026-10-29, derlenmedi):** iki kalıcı anatomi alanı —
`bend_hints` (zincir başına rig uzayında bükülme yönü) ve `foot_pivots` (bacak
zincirinin üçüncü kemiğine, bileğe göre topuk/top/uç, bileğin dinlenim çerçevesinde).
`IKControl.bendHint` → `matchIKPose` polü dinlenim bükülmesi yerine ipucundan kurar
(K4 artık **Pose IK'da da** kapalı, kontroller yeniden oluşturulunca). Yürüyüş kayıtlı
pivotları okur; kayıt yoksa aynı ortak `deriveFootPivots` ile türetir; kayıt parmaklarla
.02h'den fazla ayrışırsa `foot_pivots_differ_from_toes` raporlanır (refit sonrası bayat
kayıt sessiz kalmasın). Şablonlar: humanoid/detaylı → diz +Z, dirsek −Z, ayak pivotları
(boyla ölçekli); kuş → bacak +Z. Eski rig'ler için `rig.derive_limb_hints`
(UI düğmesi + Python + IPC): ileri yönü **ayaklardan** ölçer (bilek→parmak), sabit
eksen yok. Panel: Rig anatomy → "Bend hints and foot pivots" (doldur / yeniden türet /
Enter ile düzenle / sil); zincir silinince ona bağlı ipucu ve pivotlar da silinir.
Eklem yönelim çerçeveleri bu partide **yok** — parmak curl eksenleri, limit eksenleri
ve bind matrisleri yerel eksene bağlı; ayrı parti.

Prob artık önce `rig.derive_limb_hints` çağırır ve 9 eşik koşar (+ pivotlar anatomiden
ve parmaklarla uyumlu, + diz polü anatomi ipucundan). Ek elle kontroller, sırayla:
1. Rapor: `forward` ≈ (0,0,1), `forward_assumed` false, 4 ipucu, 2 pivot seti,
   `pivots_estimated` boş. Bozuksa: forward ters → ayak bilek/parmak sırası ters
   okunuyor; ipucu 0 → aile `humanoid` değil.
2. `rig.get_anatomy` → `version` 8, `bend_hints` + `foot_pivots` var. Kaydet → yeni
   .rtp → aç → aynı.
3. Pose'da IK kontrollerini yeniden oluştur → `rig.get_controls` satırlarında
   `bend_hint`; bacak hedefini yukarı çek → diz **öne** bükülür, dirsek geriye.
4. Yeni humanoid şablon rig (`rig.create`) → anatomi doğrudan v8, pivotlar boyla ölçekli
   (1,8 m'de topuk z ≈ −0,051, top z ≈ 0,171).
5. Panelde bir pivotu değiştir (Enter) → `foot_pivots_differ_from_toes` true olur;
   "Re-derive all" → false. Zincir sil → kayıt reddedilmeden ipucu/pivot da gider.
★ Sinsi: IK kontrolleri v8'den ÖNCE oluşturulduysa ipucu taşımaz ve diz eski gibi
dinlenim bükülmesinden davranır — hata yok, sadece eski davranış. 3. madde bu yüzden
"yeniden oluştur" diyor.


**W1b canlı (2026-10-29, ikinci derleme):** prob 6/6 PASS verdi ama **yanlış**: diz
f0–f13 boyunca 11,5°'de sabitti (= `leg_reach` .995 tavanı). Testin pencere hatası
itiş öncesi kareleri de sayıyordu. Kök: diz profili bacak boyunu **milimetre**
değiştiriyor (18° ile 11,5° arası ~6 mm), `body_bounce` .01h ise **1,8 cm** ekliyordu —
zıplama pelvisi tavana itip profili eziyordu. Derlemesiz doğrulandı (IPC'den
`body_bounce=0`, `leg_reach=.998`): diz temasta 7,3° → yükte 17,8° → itiş öncesi 7,3° →
salınım tepesi 70°, penetration 1e-7, hata 1e-7. Aynı tabloda yeni bulgu: pelvis
temas anında **tek karede 3–3,4 cm düşüyor** (her adımda "küt" oturma) — öndeki
bacak tavana son anda yetişip pelvisi çekiyor.

**W1c (2026-10-29, derlenmedi):** pelvis düşüşü iki geçişte: önce kare başına ham
düşüş, sonra döngüsel **genişlet (max, r) + kutu bulanıklığı (r)**, r = döngünün
%8'i. Sonuç hiçbir karede ham değerin altında değil → her erişim tavanı hâlâ tutar.
Varsayılanlar: `body_bounce` **0**, `leg_reach` **.998**. Prob düzeltildi: diz değişimi
yalnız temas→topuk kalkışı penceresinde ölçülür, yeni eşik **pelvis adımı ≤ .006h/kare**.

Beklenen (W1c sonrası): 7/7 PASS; tabloda pelvisY kareden kareye ≤ ~1 cm değişmeli,
temas karesinde (f0/f29) sıçrama yok. Diz eğrisi yukarıdaki derlemesiz koşuyla aynı
biçimde (temas ~7°, yük ~18°, salınım ~70°), yalnızca birkaç derece daha bükük olabilir
(yumuşatma düşüşü yalnız artırır). Bozuksa: pelvis adımı hâlâ büyük → yumuşatma
uygulanmıyor; diz yine sabit → `body_bounce` 0 değil ya da tavan her karede bağlıyor.


`python scripts/test/rt_probe_human_walk_ipc.py Rig` — karakter Pose'da. `Walk_Loop`'u
varsayılanlarla yeniden üretir, sol bacağın bir döngüsünü kare kare basar, 6 eşik:
ayak hedef hatası ≤ .005h, zemine girme ≤ .002h, erişilemeyen kare 0, **basma
fazında diz en az 8° değişmeli**, temasta diz ≤ 12°, salınım dizi tepesi 50–75°.
Bozuksa: diz yine sabit → pelvis profili uygulanmıyor (ya da `leg_reach` tavanı
her karede bağlıyor, `max_reach_ratio` .995'e yapışık); penetration sürüyor → parmak
bükülmesi yanlış işaretli (parmak aşağı kıvrılıyor) ya da rigde toe rolü yok
(`toe_bend` false). Sonra gözle: yandan bak, basan dizin temasta neredeyse düz,
yüklenirken hafif bükük, itişte yine düz olması gerekir.
★ Sinsi: dikey zıplama fazla gelirse `body_bounce`'u 0 yap — artık asıl dikey hareketi
bacak geometrisi üretiyor, parametre yalnızca üstüne ekler.

**İlk canlı ölçüm (2026-10-29, W1–W4 derlemesi, `humanoid_detailed`, 1,8 m, varsayılanlar):**
`max_reach_ratio` .970, `foot_target_error` 1,9e-7 m, `unreachable_frames` 0,
pelvis düşüşü 2,8–7,2 cm → W1 sayısal olarak PASS. **İki FAIL, ikisi de gözle görülen
"robotik" his:** (1) basan bacakta diz bütün basma fazı boyunca **28,1°'de sabit** —
pelvis çözümü bacağı her karede tam `leg_reach`=.97 sınırına çiviliyordu (.97 = 28°
diz, kosinüs kuralı), bacak bükük bir değnek gibi pergel yürüyüşü yapıyordu; kalça hiç
geriye açılmıyordu (min +5°). (2) `ground_penetration_max` 3,99 cm: topuk top
etrafında kalkarken parmaklar bükülmediği için uç zemine giriyordu. Salınım dizi tepesi
74°, dirsek 25–57°, ayak yuvarlanması (topuk 15°, kalkış 30°) beklendiği gibi.

**W1b düzeltmesi (2026-10-29, derlenmedi):** pelvis yüksekliğini artık basan bacağın
**diz bükülme profili** belirler: temasta ~5°, `stance_knee_deg` (18°) yükleme
bükülmesi, topuk kalkmadan önce ~6°; bacaklar destek ağırlığıyla harmanlanır.
`leg_reach` yalnız tavan (varsayılan .97 → **.995**), `body_bounce` bunun üstüne eklenir
(varsayılan .02 → **.01**, çünkü dikey hareketi artık geometri üretiyor). Parmak kemiği
topuk kalkışında ayak pitch'ine karşı bükülür, salınımın ilk üçte birinde düzelir
(`toe_bend`); uç ölçümü parmak kemiğinden yapılır. Göğüs ve kollar pelvisi döngünün
%5'i kadar geriden izler (`upper_body_lag`). Yeni alan `stance_knee_deg` 0..35 her
katmanda. Ölçüm aracı: `scripts/test/rt_probe_human_walk_ipc.py` (6 eşik).

Kaynak: `RigWalkRecipe.cpp` yeniden yazıldı (W1 pelvis yüksekliği, W2 ayak
yuvarlanması, W3 gövde dönüşleri karakter çerçevesinde, W4 diz polü karakterin
baktığı yönden); yeni tarif alanları `leg_reach`,
`heel_strike_deg`, `toe_off_deg` UI/Python/IPC'de. Plan ve teşhis:
`RIG_AUTHORING_ROADMAP.md` §11. 20 ölü Rig Studio dosyası silindi (vcxproj'da
zaten yoktu; `RigFingerFK*` canlı, dokunulmadı). Önden arkaya, bağımsız ve hızlı
olan önce:

1. **Derleme.** Hatasız bitmeli. Bozuksa: silinen 20 dosyadan birini include eden
   bir yer kalmış demektir (hata satırı dosyayı söyler) — ya da yeni `RigWalkRecipe.cpp`
   tip hatası.
2. **Check recipe, varsayılanlarla** (Pose → Quick Motion; ya da IPC
   `rig.preview_human_walk`). Paneldeki yeni iki satır ve `metrics`:
   - `max_reach_ratio` ≈ .97 (≤ .98), `unreachable_frames` = 0
   - `foot_target_error_fraction` < .005 (panelde "%0.5" altı)
   - `pelvis_drop_min` ≈ 0, `pelvis_drop_max` > 0 (beklenen ~.02–.05 × boy)
   - `ground_penetration_max` ≈ 0 (≤ .002 × boy)
   Bozuksa: reach > .98 → pelvis çözümü uygulanmıyor; reach doğru ama foot error
   büyük → ayağı hedeften koparan şey IK sonrası **eklem limitleri** ya da diz polü
   (teşhis K4/K5); penetration büyük → pivot türetimi yanlış (`foot_pivots`
   alanındaki heel/ball/tip noktalarına bak, `foot_pivots_estimated`).
3. **Karşılaştırma:** aynı rig'de `leg_reach=1` ile tekrar → `max_reach_ratio` ≈ 1 ve
   `pelvis_drop_max` 2'dekinden küçük olmalı. Bu, sayının pelvis çözümünden
   geldiğinin kanıtı.
4. **Generate `Walk_Loop`, YANDAN izle.** Ayak artık yerden havada kalmamalı, diz hiç
   kilitlenmemeli; topuk vuruşunda parmak yukarıda, kalkıştan önce topuk ball
   etrafında yükselmeli; duruş sonu → salınım başı geçişinde ve döngü sınırında
   ayak sıçramamalı. Bozuksa: salınım başında sıçrama → `moveFoot` uç noktaları
   duruş eğrisiyle eşleşmiyor.
5. **`heel_strike_deg=0`, `toe_off_deg=0`** → yuvarlanma kaybolur, düz taban; pelvis
   çözümü sürer (metrikler 2'deki gibi kalmalı).
6. **Python paritesi:** `rt.rig.preview_human_walk(char, leg_reach=.95, toe_off_deg=40)`
   ile aynı değerleri IPC'ye ver → `metrics` birebir aynı. Geçersiz değer
   (`leg_reach=.5`) iki yoldan da `rig_walk_invalid_recipe`.
7. **W3/W4 — önizleme alanları:** `body_frame` = `character`, `knee_pole` =
   `character_forward`. Şablon rig'de yürüyüş öncekiyle **gözle aynı** görünmeli
   (şablonun eksenleri zaten dünyaya hizalı). Fark görünürse W3'ün çevrimi yanlış.
8. **W3/W4 — asıl test:** içe alınmış, yönlendirilmiş kemikli bir humanoid
   (ör. Mixamo FBX, humanoid anatomisi eşlenmiş) üzerinde Generate. Gövde öne
   eğilmeli, kalça yürüyüş yönünde dönmeli, dizler HER karede öne bükülmeli.
   Bozuksa: gövde yana yatıyor/ters dönüyor → `context.forward/left` yanlış
   türetilmiş (ayak→parmak yönü ve omuz farkı, önizlemede görünmüyor); diz yana
   bükülüyor → pol hesabı.

★ **Sinsi, makul görünen sonuç:** büyük stride'da karakter belirgin şekilde
**çömelerek** yürür. Bu hata değil, bacak boyunun dürüst sonucu (eski hali aynı
stride'da ayağı 5 cm havada bırakıyordu). Çömelme fazlaysa stride'ı düşür.
★ İkinci sinsi durum: `ground_penetration_max` ≈ 0 ama `foot_target_error`
büyük → ayak hedefin **üstünde** kalıyor; önden bakınca görünmez.

Yapılmadı: pivotlar ve bükülme yönü kalıcı/düzenlenebilir değil (şablon v4);
Pose modundaki IK kontrolleri hâlâ dinlenim bükülmesinden pol alıyor (W4 yalnız
yürüyüş tarifinde). Önizleme artık kare başına ~9 poz örneklemesi yapar; çok
kemikli rig'de "Check recipe" bir an takılabilir — hata değil.

## Birleşik madde canlı kabul — 2026-10-28 (kullanıcı derlemesi)

**4-emitter sahnesi devam kabulü (aynı gün):** ayarlar değiştirilmeden
`rt_probe_emission_ceiling_ipc.py` ile 204 adım, dt 1/24 (8.5 s simülasyon,
199 s duvar süresi). Domain tavanı 100000; son okuma 99146 parsel,
`step_held=false`. Tavan kısılması kaynaklarda ve domain'de birebir aynı:
89.571863067 kg (Water 51.040814146, Snow 2.053671730, Sand 36.477377191).
Son adımda da kısılan kütle pozitif. Son parçacık sayısının tavanın altında
olması, o adımda emisyon kapısının bağlanmadığı anlamına gelmez.
Durum çubuğundaki sarı **EMITTERS THROTTLED: Physics Domain 1** uyarısı
gözle doğrulandı. Tek seferlik HUD mesajı gözlenmedi; pill tıklama testi
yapılmadı. Sahne resetlenmeden ölçülen durumda bırakıldı; ayarlar kaydedilmedi.
Rapor: `%LOCALAPPDATA%/Temp/rt_probe_emission_ceiling.json`.

Boş sahne, Pause; dış `scripts/test/rt_ipc.py` istemcisi kullanıldı.
Sandbox named-pipe hata 5 sonrası testler gerekli escalation ile koşuldu.

- `rt_test_emission_budget_ipc.py`: **6/6 PASS**. 2 kg sınırında
  2.009578 kg / 129 parsel (parsel 0.015578 kg; bir parsel toleransında).
  400 parçacık tavanında kaynak/domain kısılan kütlesi aynı: 87.237503 kg;
  `emission_withheld` defter kaydı mevcut, eski sayı anahtarı reddediliyor.
- `rt_test_matter_exchange_ledger_ipc.py`: **PASS**, kütle hatası 0 kg,
  enerji hatası 0 J.
- `rt_matter_smoke.py --quick`: **7/11 test PASS**. Geçenler: thermal_films,
  phase3_freeze, fire_steam (8/8), grain_readiness, p9_melt (4/4),
  run_readback, fast_fall.
- Açık: substance testi Snow `granular_compaction_hardening > 0` assertion;
  seed_settle 4/5 (`liquid_at_rest`, su p95 4.53 mm/kare);
  p9_floor 1/2 (`sand_rests_on_ice`, 864 buz yüzüne rağmen kum min y 0.00999 m);
  mpm_dem_clock 3/4 (`equal_counts`: snow-only okuması 0, sand-only 300,
  mixed 600; kök bu turda ayıklanmadı).

Loglar: `%LOCALAPPDATA%/Temp/rt_matter_smoke/`; emisyon raporu:
`%LOCALAPPDATA%/Temp/rt_test_emission_budget.json`. Son salt-okuma kontrolü:
Pause, frame 0; domain, kaynak ve forcefield listeleri boş. UI pill/HUD ve
4-emitter sahnesinde uzun koşu kabulü henüz yapılmadı. Bu kayıt tüm D2/D3
işlerini veya önceki tarihli build bekleyen maddeleri kapatmaz.

## Gaz §24: doğal GPU ölçümü (build gerekmiyor)

İki ana prob düzeltildi: viewport.render_frames artık kullanılmaz. GPU süreleri
sayaçlar kapalıyken doğal display akışından, counters ayrı pass'ten alınır.
Aynı voxel/kamera kimliğiyle son256/16/256=611/572/606ms; örnekler32.88/10.31/32.88.
MaxSteps çalışıyor ama kalan per-entry maliyet yüksek. Yeni shader/C++ değişmedi.

- `python scripts/test/rt_probe_gas_max_steps_ipc.py` yeni sampling_method etiketini
  ve aynı scene_signature'ı göstermeli; orijinal ayarlar restore edilmeli.
- `python scripts/test/rt_probe_gas_overlap_cost_ipc.py --frames 4` aynı doğal
  helper'ı kullanır. Eski forced-pass ms sonuçlarını yeni sayılarla karıştırma.
- Sabit/per-entry maliyet için GPU profiling/ayrıntılı maliyet ayrımı gerekiyor;
  yeni bir spekülatif march algoritması eklenmedi.

Canlı ölçüm ve Python sampling/snapshot testleri geçti; yeni build gerekmiyor.


## Gaz §23: snapshot kimliği / tekrar yükleme tutarlılığı (shader değişmedi)

Yeni GPU optimizasyonu yok. Yeni Python probu şimdi kullanılabilir; eski
build'de summary_only kimlik verir. Tüm voxel hash'leri için C++ build gerekir
(FogFieldDigest.h + RtApiMatterFogInspection), shader build gerekmez.

1. Yeni C++ build sonrası `fluid.inspect_fog_fields` her field'da density_hash /
   temperature_hash döndürmeli. UI/IPC/Python aynı read-only API'yi kullanır.
2. Aynı oturumda `python scripts/test/rt_probe_gas_max_steps_ipc.py`:
   başta scene_signature/fingerprint_strength, sonra her koşunun kimliği ve
   çözünürlüğü yazılır. Kimlik değişirse sonuç INVALID; eski raporla oran kurma.
   Orijinal cap ve instrumentation finally ile geri alınmalı.
3. Reset/step yapmadan aynı karede iki read-only snapshot aynı olmalı. Kaydet/
   yeniden yükle veya timeline cache restore sonrası aynı alan bekleniyorsa
   field hash ve optical_bounds karşılaştır. Float hash değişirse sim alanı
   aynı değildir; yalnız AABB farkı üzerinden renderer regresyonu ilan etme.
4. Son kullanıcı256/16/256-repeat:625.6/488.7/603.8 ms; iki256'nın sayaçları
   birebir aynı. Son reload buhar kutusu%6.06, önceki yavaş snapshot%45.08 ve
   kamera hedefi farklı. Girdi farkının kaynağı hâlâ açık.

Statik snapshot kimliği testleri geçti. Ajan uygulamayı açmadı/derlemedi.


## Gaz §22: genel karışım Max Steps cap'i (yalnız shader, DERLENMEDİ)

Kaynakta bug doğrulandı: genel karışım max_steps okumuyordu. Yeni cap,
segmentteki ilgili üyelerin en düşük değeridir; kaba adım kalan yolu kaplar,
gaz kuyruğu kesilmez. Boş atlama örnek bütçesi yemez. Random-walk/cache değişmez.

1. Kullanıcı `compile_shaders.bat` çalıştırır. Yeni C++ build GEREKMEZ.
2. Aynı kare/kamera/çözünürlükte dış terminal:
   `python scripts/test/rt_probe_gas_max_steps_ipc.py`.
   256/16/256-repeat ölçülür, orijinal ayarlar sonunda geri alınır. 16'da
   uzun kamera segmentleri daha az örneklenmeli; step_budget_exhausted yerel
   adaptif hedefin work cap yüzünden kabalaştığı ışınlarda artabilir.
3. Çok düşük bütçede görüntü kabalaşabilir; gaz sonu kaybolmamalı. İnce buhar,
   emissif pikler ve SDF/solid önündeki sınır görsel kontrol edilsin.
4. UI, Python ve IPC max_steps aynı sonucu vermeli. İki çakışan üyeden biri
   16 diğeri256 iken ortak cap16; off-ray bir16-lı slot etkilememeli.
5. Pipeline süre/boyut kaydedilsin. §21 son GPU tabanı185/84/280 ms,
   kamera density9.482/yol; residency dört NanoVDB grid için1. Sekme density
   ve radiance cache fallback0 korunmalı. Sonuç GPU'da henüz ölçülmedi.

Statik/nümerik örnek tavanı ve Beer–Lambert tam segment coverage kontrolleri geçti.


## Gaz §21: NanoVDB yoğunluk/sıcaklık device-local (yalnız C++, DERLENMEDİ)

Somut açık: CPU_TO_GPU buffer isteği host-visible/coherent ister, VRAM istemez.
RTX 3060 memory type sıralamasında ilk eşleşen type 3 sistem heap'indedir;
host-visible VRAM type 5 sonra gelir. Yeni allocation GPU_ONLY ile
DEVICE_LOCAL zorunlu. Shader ve örnek politikası değişmez; kazanç ölçülmedi.

1. C++ build al, uygulamayı yeni executable ile yeniden aç. Shader build GEREKMEZ.
   Yeni module VulkanVolumeResidency.cpp vcxproj/filter listelerine eklendi.
2. Aynı scene/kare 50/kamera/1680×945, Vulkan Rendered:
   `python scripts/test/rt_probe_fog_bounds_ipc.py`.
   Her iki Matter NanoVDB slotunda density_device_local=1 ve
   temperature_device_local=1 olmalı. 0=host heap (FAIL), -1=bilinmiyor;
   -1'i VRAM'de sanma. Dar kutular korunmalı.
3. `python scripts/test/rt_probe_gas_overlap_cost_ipc.py --frames 4`.
   Son aynı oturum authored birlikte GPU 537 ms; önceki 50. kare koşusu 573 ms.
   GPU süreleri, kamera density, majorant/probe sayıları ve görüntüyü karşılaştır.
   Örnek sayısı değişmeden süre azalması bu partinin hedefidir. İkincil density
   ve radiance_cache_fallbacks sıfır korunmalı. Süre düşmediyse RAM hipotezini
   tek neden sayma; yeni residency kanıtını ve GPU profiling'i kullan.
4. Kamera sabitken görüntü aynı olmalı. Sim playback'te first upload/host
   beklemeleri ayrıca ölçülsün: GPU_ONLY staging/copy ekleyebilir. Content
   update, timeline seek, backend switch ve kaynak görünürlüğü çalışmalı.
5. OptiX ~110 ms kullanıcı referansı kaydedildi ama optik formüller farklı;
   eşit kaliteli/eşit GPU-dispatch kapsamlı benchmark varsayılmamalı.

Statik residency/Fog bounds/material/march kontrolleri geçti. Gerçek GPU
bellek yerleşimi ve performans kullanıcı build'i sonrası doğrulanacak.


## Gaz §20.1: blok adımı geri alındı (CANLI §20.2, yeni build gerekmiyor)

Geri alma sonrası güncel canlı ölçüm: kare 50, buhar kutusu %51.5, GPU
446/78/573 ms. Önceki 329 ms snapshot kutusu %11.27 idi; sabit kare/kamera
olmadan doğrudan geri dönüş oranı çıkarılamaz. Mevcut shader yeniden
derlenmek zorunda değil. Aşağıdaki liste tarihsel geri alma kontrolüdür.

§20 kullanıcı ölçümünde FAIL: birlikte 766 ms (önce 329), kamera örneği
24.794/yol (önce 10.745), majorant sorgusu 11.542/yol (önce 4.463).
İkincil density/fallback hâlâ sıfır; sekme cache'i sorun değil.

1. Kullanıcı `compile_shaders.bat` çalıştırır. Bu geri almanın performans
   düzeltmesi için C++ gerekmez; eski tooltip'e dönmek için C++ build isteğe bağlı.
   Yeni optical-block include silindi; tek hacim yönlendirmesi de kaldırıldı.
2. Aynı kare/kamera/çözünürlükte dış terminal:
   `python scripts/test/rt_probe_gas_overlap_cost_ipc.py --frames 4`.
   §19 tabanı buhar/duman/birlikte 261/84/329 ms. Geri dönüş henüz ölçülmedi;
   kamera density ve majorant sorguları da tabana yaklaşmalı.
3. `python scripts/test/rt_probe_fog_bounds_ipc.py`: dar kutu hâlâ korunmalı
   (son kullanıcı ölçümü buhar %11.27, duman %2.66). Görünür buhar değişmemeli.
4. İkincil density/cache fallback sıfır kalmalı. Pipeline süresini kaydet.

Statik geri alma, adaptive, optical, cache ve Fog bounds kontrolleri geçti.
Yeni deney eklenmedi. Sonraki değişiklik öncesi kalan GPU işinin profillenmesi
veya daha ayrıntılı sayaçlarla ayrılması gerekiyor.

## Gaz §19: materyale göre daralan Fog kutusu (CANLI hız/kutu geçti)

Canlı sonuç: buhar kutusu %11.27, duman %2.66; GPU buhar/duman/birlikte
261/84/329 ms. İkincil density ve cache fallback sıfır. Yeni build gerekmiyor;
aşağıdaki listede görsel ve materyal değişimi kontrolleri açık. Ayrıntı §19.1.

Bu parti shader değiştirmez. Önceki §18.1 kullanıcı tarafından derlendi,
anlamlı iyileşme gözlenmedi. Yeni kutu kodu görünür katkısı olmayan ham
yoğunluk kalıntılarını güvenli biçimde dışarıda bırakır. Kazanç ölçülmedi.

1. C++ build al. Header proje listelerine eklendi; yeni .cpp modülü yok.
   Gerçek çekirdek regresyonu istenirse VS Developer terminalinde:
   `cl /nologo /std:c++17 /EHsc /I RayTrophiStudio/source/include scripts/test/fog_optical_bounds_test.cpp RayTrophiStudio/source/src/Math/Vec3.cpp "/Fe$env:TEMP\fog_optical_bounds_test.exe" "/Fo$env:TEMP\"`
   ardından `& "$env:TEMP\fog_optical_bounds_test.exe"` çalıştır. Ajan derlemedi.
2. Aynı sahne/kare/kamera (matter4emiter, kare 38–39). Dış terminal:
   `python scripts/test/rt_probe_fog_bounds_ipc.py`.
   Buharın eski AABB oranı %93.02 idi. `render_box_fraction` ve gerçek
   `render.volume_slots` kutusunu kontrol et. fallback=false olmalı;
   kutu gerçekten daralmadıysa performans kazancı varsayma.
3. `python scripts/test/rt_probe_gas_overlap_cost_ipc.py --frames 4`.
   Önceki aynı koşu 771 / 185 / 1052 ms (buhar/duman/birlikte), sonraki
   baseline 1153 ms; çözünürlük 1680×945. İkincil density/fallback=0 korunmalı.
4. Cutoff'u azalt, eğriyi değiştir, yoğunluk multiplier'ını 0 yapıp geri al:
   aynı timeline karesinde kutu büyümeli/daralmalı, gizlenen akış geri gelmeli.
   UI, `volume.set_material` IPC ve Python `rt.volume.set_material` aynı
   davranmalı. Değerleri geri yükle. Graph/noise'da fallback tam grid olmalı.
5. İnce buhar kenarları, zemine düşen gölge ve sıcak emisyon önceki görüntüyle
   karşılaştırılsın; görünür veri kesilmemeli. Hareketli plume kutusu büyürken
   yeni yoğunluk görünür olmalı. Fog → SDF/Splat kaydı temizliği korunmalı.

Statik kontroller geçti; C++ build, C++ nümerik test ve canlı hız/görüntü
kabulü kullanıcı build'inden sonra yapılacak.


## Gaz §18.1: boş örnek erken dönüşü / kamera dönüşümü reuse (yalnız shader)

§17 CANLI: ikincil density=0 ve fallback=0, fakat birlikte 1052 ms,
üstüne yalnız buhar 771 ms. Cache bypass artık kök değil. Sonraki küçük
parti `volume_mixture.glsl` ve `volume_mixture_adaptive.glsl` içindeki
tekrarlı işleri azaltır; örnek çözünürlüğünü değiştirmez. Kazanç bilinmiyor.

1. Kullanıcı `compile_shaders.bat` çalıştırır; bu yeni parti için C++ gerekmez.
2. Aynı kamera, 1680×945, kare 38–39. Pipeline compile_seconds taban 219.734 sn;
   %10 artış sınırını ve SPV boyutunu kaydet.
3. Dış terminal: `python scripts/test/rt_probe_gas_overlap_cost_ipc.py --frames 4`.
   Kamera örnekleri, sekme örnekleri ve görüntü aynı kalmalı; GPU sürelerini
   771 / 185 / 1052 ms ile karşılaştır. Cache fallback=0 korunmalı.
4. Graph'ın boş kaynakta yoğunluk üretmesi ve random-walk regresyonu.
   Değişim hızlandırmazsa yoğunluk sayısını tekrar azaltmaya çalışmak yerine
   GPU profiler ile NanoVDB erişimi / register spill / ray tracing işini ayır.

Canlı A/B ayarları geri yüklendi. Başka build veya uygulama başlatma yapılmadı.


## Genel gaz karışımı cache + Fog kaydı temizliği — 2026-10-28 (C++ ve shader)

Bu parti §16b'deki genel karışım bypass'ını düzeltir. Aşağıdaki eski
"yalnız shader" partisi önceki denemedir; bu parti onun devamıdır.

1. C++ build ve `compile_shaders.bat` çalıştır. Yeni shader include:
   `volume_cached_segment.glsl`; değişenler: `volume_radiance_cache.glsl`,
   `volume_mixture.glsl`, closest-hit bağlantısı. C++: MatterFogPublisher ve VDB UI.
2. `E:\rt_k\matter4emiter.rtp`, kare 38–39, kamera sabit, Vulkan Rendered.
   `render.rt_pipeline_status` derleme süresini ve SPV boyutunu kaydet.
   Taban 206 sn / 982,912 B; %10'dan fazla artış kabul dışı.
3. Dış terminalden `python scripts/test/rt_probe_gas_overlap_cost_ipc.py`.
   Isınma sonrası both: `secondary_density_samples` yaklaşık 0,
   `radiance_cache_marches` artmalı; GPU ms ayrı hacimlerin toplamına yaklaşmalı.
   Son ölçüm 598 + 112 → 1157 ms idi. Yeni kazanç henüz ölçülmedi.
4. Cache kapalı/açık karşılaştır: zemin/sıvı dolaylı ışığı ve sıcak emissif
   çakışmalar; kamera görüntüsü değişmemeli. Tek hacim, programlı hacim,
   random-walk ve SDF önündeki fog sınırı regresyonlarını kontrol et.
5. Bir emitter'in katı/sıvı çıktısını Fog yap, VDB materyalini düzenle;
   SDF ve Splat'a geçir. Eski carrier Fog kaydı listeden ve seçili özelliklerden
   kalkmalı; gerçek gaz/vapor akışları kalmalı. Fog'a dönünce materyal geri gelmeli.
   Aynı değişimi Python/IPC çıktılarıyla ve undo/redo ile tekrarla.

Statik kontroller geçti; ajan build çalıştırmadı ve uygulamayı açmadı.


## Gaz RT: çakışan iki gaz hacminde radyans önbelleği — 2026-10-28 (yalnız shader)

Emitter bütçesi partisi CANLI geçti (kullanıcı: hepsi PASS). Bu parti yalnız
`volume_radiance_cache.glsl` ve `volume_closesthit.rchit`'e dokunuyor:
`compile_shaders.bat` yeter, C++ derlemesi gerekmez. Sahne:
`E:
t_k\matter4emiter.rtp`, kare 39, kamera değişmeden.

1. **Derleme süresi (ilk iş, kural):** shader'ı derle, uygulamayı aç, Rendered'e
   geç ve ilk RT görüntüsüne kadar bekle. Sonra `render.rt_pipeline_status` →
   `stage_seconds` toplamını ve `x64/Release/shaders/volume_closesthit.spv`
   boyutunu yaz. Taban: 977,104 B; geçmiş ölçümler 314 sn (normal) ve 495 sn
   (geri alınan deneme). Belirgin artış (> %10) varsa tek çağrı yeri kuralı
   tutmamış demektir.
2. **Hız (asıl kabul):** iki gaz görünürken bir RT örneği önceden 964 ms
   (yalnız buhar 366, yalnız duman 86). Aynı ölçüm, kamera sabit:
   `python scripts/test/rt_probe_gas_overlap_cost_ipc.py`.
   Görmen gereken: ikisi birlikteyken ikincil yoğunluk örneği ~0 (önce 3.3/yol),
   GPU ms ~ ayrı ayrı toplamına (~450) yakın. Hâlâ 900+ ise önbellek çakışmada
   kaçırılıyor (`radiance_cache_fallbacks` sayacına bak).
3. **Görüntü:** iki gazın çakıştığı yerde sekme ışığı (zemine ve sıvıya düşen
   dolaylı ışık) öncekinden belirgin parlak ya da karanlık olmamalı. Hafif fark
   beklenir: her ortamın J'si yalnız kendi ortamıyla gölgeleniyor.
   ★ Sinsi olan: buhar ve dumanın İKİSİ de blackbody (emission_mode 2). Kamera
   march'ı companion'ın emisyonunu atlıyor, önbellek atlamıyor: düşük sıcaklıkta
   fark görünmez, ama sıcak bir duman companion olursa sekme ışınlarında
   kameradan fazla parlar ve hiçbir şey hata vermez.
4. **Regresyon:** tek gazlı sahnede (yalnız buhar ya da yalnız duman) süre ve
   görüntü değişmemeli; tek ortam yolu aynı formüle iner (J = J₁).

Statik: `check_volume_march_lod_contracts` OK, gaz optik / boş eleme / ışık
önbelleği / malzeme / karışım kontrolleri OK. `check_rt_gpu_timing_contracts`
FAIL, bu partiden bağımsız: kontrol `beginRayGpuTiming(cmd, w, h)` arıyor, kod
FSR partisinde `traceW, traceH`'ye geçmiş (diğer ajanın alanı).

## Emitter bütçesi kg + Max Particles kısma uyarısı — 2026-10-28 (yalnız C++)

Ortak saat partisinin 2. derlemesi CANLI geçti (128'de GRAINS SOFTENED ile
devam; 512'de bölünme 24 × 8). Bu parti shader'a dokunmuyor.
Kaynak sınırı artık kg: `use_mass_limit` + `max_emitted_mass_kg` (eski
`use_particle_limit`/`max_emitted_particles` IPC ve Python'da gerekçeli
reddediliyor, proje yüklemesinde uyarıyla düşüyor). Sıvı parseli doğumda
kütlesini alıyor. Tavan bağlayınca kısılan kütle sayılıyor, defterde
`emission_withheld`, durum çubuğunda sarı EMITTERS THROTTLED.

1. **Kabul testi (en hızlı, her şeyi kapsar):**
   `python scripts/test/rt_test_emission_budget_ipc.py` (boş sahne, Pause).
   Beklenen 6/6: eski anahtar reddi, 2 kg'da durma (bir parsel içinde),
   parsel sayısı = kütle / parcel_mass_kg, tavan kısıyor, domain = kaynak
   toplamı, defterde emission_withheld.
   Bozuksa: `budget_stops_at_mass` FAIL ve kütle 2 kg'ı çok aşıyorsa döngü
   içindeki kg kesmesi çalışmıyor; `ceiling_withholds` FAIL ise tavan kapısı
   artık kısılanı saymıyor.
2. **Panel:** bir sıvı kaynağında "Limit Emitted Mass" + "Max Emitted Mass
   (kg)" ve ilerleme çubuğu kg cinsinden. Kapalıyken "Emitted: x kg (n
   particles)". Domain Max Particles'ı küçük yap (ör. 400): kaynağın altında
   "Max Particles withheld ... kg" satırı, durum çubuğunda sarı **EMITTERS
   THROTTLED** pill'i ve bir kez HUD mesajı. Tıklayınca domain açılmalı.
3. **Senin sahnende (E:
t_k\matter4emiter.rtp) uzun koşu:** su 10k/s,
   tavan 100k. Tavana ulaşınca sarı pill çıkmalı; `fluid.list_domains` →
   `emission_withheld_kg_step` > 0. Buharlaşma sürerken pill kalır, bu doğru:
   kaynak artık kaybı dolduruyor.
4. **Test istemcisi:** test betikleri eski sayı sınırı yerine `parcels=N`
   (ve `limit_parcels=`) geçiyor; `rt_ipc.py` bunu kaynağın
   `parcel_mass_kg`'sıyla kg'a çeviriyor ve tane yarıçapı gibi ayarlar sonradan
   değişirse bir sonraki reset/adımda yeniden boyutluyor. Kontrol:
   `rt_test_snow_heat_ipc.py` (kaynaktan SONRA tane yarıçapını değiştiriyor)
   önceki gibi 11/11 ve aynı tane sayıları (sand 1500, snow 3000).
   Sayılar farklıysa yeniden boyutlama tetiklenmiyor.
5. **Regresyon:** `rt_test_matter_exchange_ledger_ipc.py` (yeni türler izinli),
   `rt_matter_smoke.py --quick`.
6. ★ **Sinsi olan:** sıvı parseli artık doğumda kütlesini alıyor (önceden 0
   doğup bir sonraki adımda dolduruluyordu). Değer aynı formül
   (voksel³/PPC × yoğunluk) ama hacim `state.voxel_size`'tan; bir domain'in
   sıvı ızgarası farklı bir voksel kullanıyorsa ilk adımdaki kütle önceki
   davranıştan sapar ve hiçbir şey çökmez. Kontrol: smoke'taki kütle
   korunumu kapıları (`p9_melt`, `snow_heat` ledger) değişmemeli.
7. `matter.exchanges` özetine (source/target/mass_error) emisyon kayıtları
   girmiyor, yalnız `exchanges` listesinde görünüyor; özet sayıları önceki
   gibi kalmalı.

## Ortak saat: sıvı yol minimumu söküldü — 2026-10-27 (2. derleme, yalnız C++)

1. derlemenin canlı sonucu (kullanıcı sahnesi, probe, limit 512): 120 adımda
**hiç durmadı**; köprü hacmi tabanda (6.54e-9 m³), sınırlayan yay `tangential`
(köprü değil), varsayılan temas 12–14 (eskiden 24). Ama bölünme `154 x 2`:
sürekli ortamı sıvının yol minimumu (her yenilemede < 2.5 mm) 154 adıma
zorluyordu, ve limit 128'de "Matter common clock needs 132 substeps" ile
domain'i durduruyordu (DEM ise 0.5 yumuşatmayla 126'ya sığmıştı).
Bu derleme o minimumu söküyor; shader değişmedi.

1. Limit **128**'de (sahnede şu an 128) Play: durmamalı. Sarı GRAINS
   SOFTENED pill'i görünebilir (DEM 128'e sığmak için yumuşar), kırmızı
   SIM STOPPED görünmemeli. Kırmızı çıkarsa ve mesaj "liquid/MPM continuum
   needs N" diyorsa sürekli ortamın KENDİ CFL'i (sıvı hızı / kar sertliği)
   limiti aşıyor demektir; o gerçek bir sınır.
2. Probe (limit 512): `python scripts/test/rt_probe_grain_clock_ipc.py --every 12`.
   Bölünme `C x K`'da C artık ~24 civarı (kar MPM dalga sınırı 23), K ~13;
   toplam yine ~300. `grain_step_ms` 1. derlemede 142 → 361 ms (parçacık
   arttıkça) idi; aynı adımlarda belirgin düşmeli. Düşmüyorsa maliyet sürekli
   ortamda değil DEM'in kendisinde demektir (o zaman D2-2'nin ikinci yarısı:
   yalnız temastaki taneler).
3. ★ Sinsi olan: tanelerin gördüğü sıvı alanı artık daha seyrek tazeleniyor
   (sürekli adım başına bir kez). Hızlı akan suda taneler sürüklenmeyi bir
   adım geç görür; çökme yok, yalnız taneler akıntıyı biraz "geç" takip eder.
   Kontrol: aşağıdaki 6. madde (`momentum_residual_n_s` ~0) ve gözle kum
   suya düşerken önceki derlemeye göre farklı savruluyor mu.
Aşağıdaki 1. derleme listesi hâlâ geçerli (madde 1–2 CANLI geçti).

## Tane alt adımı: köprü tabanı, sertlik uyarlama, ayrık ortak saat — 2026-10-27

C++ **ve shader** (`sim_matter_grain.glsl` → `compile_shaders.bat`; revizyon
31 → 32). Sahne: `E:
t_k\matter4emiter.rtp` (su, kar, 600 K kum, gaz).
Önceki ölçüm: ortak saat 322 alt adım, ıslanmayla 32. adımda 527 > 512 → durma.
Durma uyarısı (önceki parti) kullanıcı tarafından CANLI doğrulandı.

1. **Derleme + shader.** Exe ve shader'lar birlikte. Shader eskiyse ilk karede
   "grain step shader revision mismatch; rebuild simulation shaders" görürsün:
   bu, derlemenin değil shader'ın eksik olduğunu söyler.
2. **Probe (en hızlı, her şeyi gösterir):**
   `python scripts/test/rt_probe_grain_clock_ipc.py` (sahne açık, Pause).
   Görmen gereken:
   - `never held in 120 steps`.
   - Bölünme `sub N = C x K`: C sürekli ortamın kendi sayısı (kar MPM dalga
     sınırı ~23-24 civarı), K DEM'in her birindeki iç adımı. Eskiden C = 322.
   - Islanmadan sonra (adım ~22+) kaynak `contact`, `bridge` DEĞİL;
     `Vb` ≥ ~6.5e-9 m³ (2.5 cm tanede 1e-4 × tane hacmi).
   - `contacts a/b`: b (varsayılan) 12–10 civarı, 24 değil.
   - `grain_step_ms` öncekinden düşük (önceki tabanı ölçmedik; sayıyı yaz).
   Bozuksa: kaynak hâlâ `bridge` → shader eski (revizyon) ya da taban hesabı
   yanlış; C hâlâ yüzlerce → `.inl`'deki `clock.minimum_substeps` değişikliği
   derlenmemiş.
3. **Sertlik uyarlama:** Physics → Advanced solver → Contact substep limit'i
   64'e indir, Play. Sim durmamalı; durum çubuğunda sarı
   "GRAINS SOFTENED x0.xx" pill'i, DEM panelinde "Contact springs softened"
   satırı. Taneler patlamamalı, daha fazla iç içe geçebilir. Limit'i geri al:
   pill kaybolmalı. Bozuksa (yine kırmızı SIM STOPPED): mesaj "even with the
   contact springs softened 1000x" diyorsa sınırı köprü ya da çok küçük tane
   koyuyor, ayrıntı `cfl_terms`'te.
4. **Islak kum hâlâ yapışkan mı:** `rt_g2_dry_wet_compare_ipc.py` ve H1
   `--wet-only`. Köprü tabanı yalnız çok az sulu taneyi etkilemeli; ıslak kum
   davranışı değişmemeli. Islak taraf kuruya yaklaştıysa taban çok yüksek.
5. **Ortak saat regresyonu:** `rt_test_mpm_dem_shared_clock_ipc.py` (4/4
   bekleniyor), `rt_test_snow_heat_ipc.py`, `rt_matter_smoke.py --quick`.
6. ★ **Sinsi olan — sıvı sürüklemesi/kaldırma ölçeği:** DEM artık bir sürekli
   adımda K kez dönüyor; sürükleme dürtüsü K adım boyunca birikip sıvıya bir
   kez dönüyor, kaldırma kuvveti sürekli adımın dt'siyle. Yanlışsa hiçbir şey
   çökmez: taneler suda biraz fazla yüzer ya da batar ve "fizik böyle" gibi
   görünür. Kontrol: `fluid.matter_models` → `grain_diagnostics.liquid`
   `momentum_residual_n_s` önceki gibi ~0 kalmalı ve `buoyancy_impulse_n_s`
   ≈ Σ(ρ_su·g·V_batık)·t. Artık büyüdüyse tepki K ile ölçeklenmiş demektir.
7. **Temas bütçesi aşımı:** yoğun bir yığında (`rt_h1_dense_grain_scene_ipc.py`
   ortak saatte) `runtime.cfl_contacts_exceeded` arada bir true olabilir; bu
   bir sonraki kareyi büyütür. Sürekli true ve yığın titriyorsa margin (+4)
   yetmiyor.

Statik kontroller: 11 `check_matter_*`'tan 7'si PASS. Dördü **bu partiden
önce de** eski pinlerle düşüyor: `check_matter_grain_contracts`
(`const uint SLOTS = 24u` arıyor; revizyon 30'dan beri kapasite çalışma
zamanında), iki sleep kontrolü (`5 * capacity * sizeof(uint32_t)`),
`check_matter_gpu_contracts` (`stress_update` push 100 ≠ 68). Sonra listesine.

## SVGF ilk à-trous geçmişi — 2026-10-27

Kaynak yazıldı; shader/C++ derlenmedi, canlı kalite ve maliyet kabulü bekliyor.

1. C++ ve iki shader varyantını (`realtime_svgf.spv`, `realtime_svgf32.spv`)
   kullanıcı derlemeli. Yeni historyFlags biti 32 iki tarafta eşleşir; eski shader
   ile yeni C++ bu iyileştirmeyi uygulamaz.
2. Önceki derlemeyle aynı iç mekan, kamera hareketi, çözünürlük, sample ve SVGF
   ayarlarında karşılaştır. İlk à-trous rengi yalnız hareket karelerinde geçmişe
   gider; son geniş filtre sonucu, momentler ve history length değişmez.
3. FSR kapalıyken duvar/tavan lekesi ve kareler arası oynama azalmış mı ölç.
   İnce pervaz, tel, doku, yazı ve parlak yansımada zamanla yumuşama olmamalı.
   Aynı kontrolleri FSR Quality/Balanced ve timeline/simülasyon hareketinde yap.
4. Hızlı kamera hareketi, yeni açılan yüzey, hareketli nesne ve ışık değişiminde
   hayalet izi olmamalı. Reactive > 0.5 pikseller uzaysal filtreden önceki rengi
   saklamayı sürdürür; geçmiş kabul/reddetme testleri değiştirilmedi.
5. Kamera durunca donmuş hareket geçmişi tekrar filtrelenmemeli. OIDN açık/kapalı
   duruş geçişini, resize, sahne değişimi ve SVGF kapat/aç işlemlerini kontrol et.
   Ham birikim/final render sonucu aynı kalmalı.
6. `render.realtime_denoiser_stats` ile aynı render çözünürlüğünde temporal,
   variance, atrous ve total GPU ms karşılaştır. Yeni dispatch veya tampon yok;
   ilk geçişte piksel başına bir history okuması/yazması var. Kazanç ve maliyet
   canlı ölçülmeden PASS sayılmaz.

## FSR proje kaydı — 2026-10-27

Kullanıcı derleyip kayıt/yükleme işleminin başarılı olduğunu doğruladı.
Shader değişmedi. Aşağıdaki ayrıntılı regresyon maddelerinin tamamı için ayrı
kabul kaydı henüz yok.

1. C++ derlemesinden sonra FSR modunu Auto, kaliteyi Balanced, sharpness'ı 0.35,
   rejection_reactive'i false yapıp projeyi kaydet. Uygulamayı yeniden açıp projeyi
   yükle; panel ve IPC `render.upscale_stats` / Python `render.upscale_stats()`
   üzerinden dört tercihin korunduğunu kontrol et.
2. Off ve Fsr modlarını, diğer kalite seçeneklerini, sharpness 0/1 ve reactive true
   değerlerini de kaydet/yükle ile kontrol et. UI ve `render.set_upscale` ile yapılan
   değişiklikler aynı `RenderSettings::upscale` kaydını kullanır.
3. `render_settings.upscale` içermeyen eski proje, önceki açık projenin FSR tercihini
   taşımadan Off / Quality / 0.2 / true varsayılanlarına dönmeli.
4. Geçersiz mode/quality, sayısal olmayan veya aralık dışı sharpness ve yanlış tipte
   reactive değeri yükleme hatası vermeli. Kayıtta GPU status/history bulunmamalı.
5. FSR kullanılamayan backend/donanımda kayıtlı tercih korunmalı; mevcut uygunluk
   kontrolü ve tam çözünürlük fallback'i çalışmalı. Kaydet/yükle işlemini IPC ve
   Python proje API'leri üzerinden de doğrula.

> **B4 Auto geçişi çökme — 2026-10-27:** Kullanıcı ffxCreateContext içinde
> DLL → null function çağrısı bildirdi. SDK 1.1.4 kaynak denetimi, backend'in
> vkGetBufferMemoryRequirements2KHR adresini yüklediğini doğruladı; uygulama bu
> extension'ı açmıyor (Vulkan core sürümünü kullanıyor). FSR'ye verilen resolver,
> KHR adresi yoksa ABI eşdeğeri core vkGetBufferMemoryRequirements2'ye dönüyor.
> DLL çağrısından önce SDK'nin gerekli device komutları kontrol ediliyor;
> eksik komut durumunda adıyla tam çözünürlük fallback yapılıyor.
> Kök neden canlı debugger ile henüz doğrulanmadı. C++ Release yeniden derlenip
> uygulama yeniden açılarak Auto/Pro hareket testi yapılmalı; shader değişmedi.


> **B4 ilk canlı kontrol — 2026-10-27:** Kullanıcı derledi. 1920×1080,
> Auto / Ultra Performance ile durağan durumda giriş=çıkış (beklenen).
> Harici IPC kamera hareket turunda da active=false, dll_loaded=false,
> dispatches=0; genel uygunluk engeline takıldı. Kamera tur sonunda geri yüklendi.
> Kaynak denetimi: camera_mode==0 koşulu varsayılan Pro (1) modunu da dışlıyordu.
> Auto+Pro desteği ve engelleyen koşulu ayrı ayrı bildiren inactive_reason eklendi.
> Bu düzeltme için kullanıcı C++ Release derlemesi gerekir; FSR GPU kabulü henüz PASS değil.


> **Durum:** DERLEME BEKLİYOR — 2026-10-27 B4d (lens distortion için hız vektörleri, FSR distortion kapısı kaldırıldı). B4c ve öncesi CANLI, `e44153d9`. Ayrıntı: `RT_REALTIME_DENOISE_UPSCALE_PLAN.md` en üstteki devir notu.

## B4d — distortion'lı kamerada doğru hız vektörü + FSR (2026-10-27, derlenmedi; RAYGEN + C++)

**Kullanıcı:** `compile_shaders.bat` (`raygen.rgen` değişti) + C++ x64 Release. Push/binding
değişmedi. **Statik:** glslc `--target-env=vulkan1.3 --target-spv=spv1.4` çıktısız temiz.

1. **Kapı kalktı** — distortion'lı eski bir projeyi aç, FSR balanced, kamerayı hareket ettir.
   `render.upscale_stats`. Görmen gereken: `active=true`, render ~988×555, `inactive_reason`
   artık "camera lens distortion is active" DEĞİL. Bozuksa: eski exe.

2. **Vektör doğru mu (distortion'sız regresyon önce)** — distortion 0'lı sahnede
   `render.set_settings debug_view=16` (motion vectors) + küçük yörünge: B4'teki gibi.
   Sonra distortion'lı sahnede aynısı. Görmen gereken: düz bir duvarda vektör renkleri
   merkezden kenara **pürüzsüz** değişir, kenar bandında ani sıçrama yok.
   Bozuksa: kenarda dairesel bir halka/sıçrama → Newton tersi yanlış işaretli veya aspect
   ölçeği hatalı (`motionLensPx`).

3. **SVGF geçmişi kenarlarda** — distortion'lı sahne, `debug_history=true`, yavaş yörünge.
   Görmen gereken: kenarlar da yeşil (uzun geçmiş). Önceden kenar bandı sarı/kırmızıya
   kayıyordu (yanlış piksele bakıyordu). Bozuksa: kenarlar hâlâ kırmızı → vektör hâlâ pinhole.

4. **★ Sessiz arıza: FSR ile kenarda kayma** — distortion'lı sahnede FSR açık, ince
   çizgili/desenli bir yüzeyi ekran kenarına getirip yatay yörünge. Görmen gereken:
   desen kenarda da keskin ve yerinde. Bozuksa: kenarlarda "biraz yüzen/bulanık" desen —
   kimse hata diye raporlamaz; vektör ölçeği kenarda yanlış demektir.

5. **Güçlü barrel** — distortion'ı panelde en negatif değere çek. Görmen gereken: çökme yok,
   aşırı kenar bölgesi en kötü ihtimalle reddedilir (debug_history kırmızı, iz yok).

**B4d+ — gizli distortion durumu (aynı derleme, yalnız C++):** araştırma: model CPU/OptiX/
Vulkan RT'de aynı ve doğru; RayFusion uygulamıyor. Sorun durumdu: canlı panel YOKTU
(`scene_ui_camera.cpp` `drawCameraContent` yorum satırında), değer projede kayıtlı/yüklü,
iki çelişen otomatik formül (`Camera.cpp` her modda FOV'dan, `Camera.h` Cinema'da).
Düzeltme: proje anahtarı `lens_distortion` (eski `distortion` OKUNMAZ → eski projeler 0 açılır),
`Camera.cpp` formülü söküldü (tek sahip `calculateLensCharacteristics`, yalnız Cinema),
FOV altına her modda "Lens Distortion" slider'ı, IPC `camera.set_lens_distortion` +
`camera.get` `lens_distortion`/`auto_lens`, Python `rt.camera.set(lens_distortion=..)`.

6. **Eski proje sıfır açılır** — önceden FSR'ı "distortion" yüzünden kapatan eski projeyi aç.
   `camera.get`. Görmen gereken: `lens_distortion=0`, panelde slider 0. Kaydet, yeniden aç:
   hâlâ 0. Bozuksa: eski anahtar hâlâ okunuyor.
7. **Yeni değer kalıcı** — slider'ı −0.1 yap, kaydet, yeniden aç → −0.1. IPC ile 0.6 dene →
   ret (kırpma yok). Görüntü kenarları bükülmeli, RayFusion'da bükülmemeli (beklenen).
8. **Auto lens sahipliği** — Cinema + Auto Lens Characteristics aç: slider kilitli ve değeri
   gösterir, `camera.set_lens_distortion` reddedilir. Pro'ya geç: slider açılır ve değer artık
   FOV değiştikçe KENDİLİĞİNDEN DEĞİŞMEZ. **★ Sessiz arıza:** Pro'da FOV oynatınca kenar bükümü
   değişiyorsa eski `Camera.cpp` formülü hâlâ çalışıyor (eski exe).

## B4c+ ek (derlendi, CANLI) — son kare, parçacık hayaleti, merkez okuma

Kullanıcı derleyip doğruladı: son kare artık ham gürültüye dönmüyor (duruş SVGF'si
yakınsamış karede de gürültüsü hedefin üstündeki piksellerde filtreli), FSR'da akışkan
parçacık hayaleti yok (`rejection_reactive`: kümelenmiş raygen reddi → reactive).
Ölçülen: skin sahnesi SVGF 10.5 ms (à-trous merkez okuma düzenlemesi yalnız %4).
Bilinen uç durum: yakınsamadan sonra yalnız tonemap'i yenileyen işlem (pozlama
değişikliği) SVGF'siz ham görüntüyü gösterebilir — gözlenirse raporla.

## B4c — sahne animasyonu SVGF ve FSR için hareket sayılır (2026-10-27, derlenmedi; YALNIZ C++)

**Kullanıcı:** yalnız C++ x64 Release; shader değişmedi. Yeni dosya yok.

**Kök (canlı, skin animasyonlu sahne, kamera duruyor):** `moving=false`, `static_active=true`,
`sample=1`, saniyede 38.7 SVGF çalışması ve **38.7 geçmiş sıfırlaması**, ortalama geçmiş **1.0**,
SVGF **11.8 ms/kare** (à-trous 7.6) tam çözünürlükte, FSR kapalı. Nesne hareketi
algılaması (TLAS seri ↔ "görülen" kopya) aynı karenin render'ında ikisi birden
ilerlediği için policy anında hep eşitti: yapısal olarak hiç tetiklenmiyordu.
Yeni: önceki trace'e bakılır; kamera dururken art arda ≥2 trace birikimi sıfırladıysa
`motion_source="scene"` ve SVGF ile FSR hareket modunda çalışır. Eski seri karşılaştırması söküldü.

**Statik kontrol:** IPC audit PASS, descriptor üretildi, probe AST temiz ve Release aynasında.

1. **API probe'u** — `python scripts/test/rt_probe_realtime_denoiser_ipc.py`.
   Görmen gereken: PASS, `scene_motion` bool değilse reddediliyor, stats'ta `motion_source`
   ve `scene_reset_streak` var. Bozuksa: alan IPC listesinde yok (eski exe?).

2. **Skin sahnesi, animasyon oynuyor, kamera duruyor** — `render.realtime_denoiser_stats`
   (synchronize=true) 2 sn arayla iki kez.
   Görmen gereken: `motion_source="scene"`, `moving=true`, `static_active=false`,
   `history_resets` artışı ≈ 0, `history_length_mean` 1.0'dan belirgin yüksek (arka plan 24'e
   yaklaşır). FSR açıksa `upscale_stats.active=true` ve render ~988×555, SVGF `total_ms`
   11.8'den belirgin düşük.
   Bozuksa: `scene_reset_streak` 0–1'de kalıyorsa animasyon her trace'te birikimi sıfırlamıyor
   (ya da kamera hareketli sanılıyor); streak ≥2 ama source "none" → `scene_motion` kapalı.

3. **Animasyon durunca normale dönüş** — oynatmayı durdur.
   Görmen gereken: birkaç kare içinde `motion_source="none"`, FSR tam çözünürlüğe döner (bir kez
   reset), birikim yükselir, duruş SVGF'si (B4b) devreye girer.
   **★ Sessiz arıza:** streak kendini besler ve source "scene"de takılı kalırsa görüntü hep
   düşük çözünürlüklü ve 1 spp kalır ama "hareket ediyor" gibi makul görünür. Kontrol:
   durduktan sonra `sample` artıyor mu?

4. **Tek değişiklik animasyon değildir** — kamera dururken tek bir `scene.set_transform` ya da
   bir slider değişikliği yap. Görmen gereken: `scene_reset_streak` 1'e çıkar ve sıfırlanır,
   `motion_source` "none" kalır, FSR devreye girmez. Bozuksa: eşik 1'e düşmüş ya da her
   düzenlemede gereksiz çözünürlük geçişi oluyor (titreme).

5. **Görüntü: skin karakteri hareket ederken** — `viewport.get_screenshot`, `debug_history=true`.
   Görmen gereken: duvar ve zemin yeşil (uzun geçmiş), deforme olan karakter kırmızı/sarı
   (tutarlılık testi reddediyor, beklenen). Karakterin arkasında iz varsa → rijit hız
   yaklaşımı deformasyonu yanlış reprojekte ediyor (ayrı iş, hacim ajanının hız sözleşmesine).

6. **Kapatma A/B'si** — `render.set_realtime_denoiser scene_motion=false`.
   Görmen gereken: eski ölçümler geri gelir (geçmiş 1.0, ~11.8 ms). Kanıt, bu partinin farkın
   kaynağı olduğunu gösterir.

## B4b — duruş sönmesi ölçülen gürültüyle + FSR'ı SVGF'den ayırma (2026-10-27, derlenmedi; SHADER + C++)

**Kullanıcı:** `compile_shaders.bat` (yalnız `realtime_svgf.comp` değişti → `realtime_svgf.spv`
ve `realtime_svgf32.spv`), ardından C++ x64 Release. Yeni C++: `Backend/VulkanRawGuard.cpp`
(vcxproj/filters'ta). SVGF push hâlâ **64 bayt**; yeni bindingler 15 (r32f Welford M2)
ve 16 (slot başına sayaç SSBO). **Eski .spv + yeni exe = descriptor uyumsuzluğu**: shader
derlenmeden C++ çalıştırılırsa SVGF pipeline'ı hata verir (reason'da görünür).

**Statik kontrol:** glslangValidator iki varyant (HDR16 / RAW32) temiz; IPC audit PASS
(709 metot), descriptor üretildi; iki probe `py_compile` temiz ve Release aynasında.
Derleme ve canlı GPU doğrulaması DEĞİLDİR.

Sıra: bağımsız ve hızlı olanlar önce; görüntü sonuçları en sonda.

1. **API/IPC probe'ları** — `python scripts/test/rt_probe_realtime_denoiser_ipc.py`,
   ardından `python scripts/test/rt_probe_upscale_ipc.py`.
   Görmen gereken: ikisi de `"api_validation": "PASS"`. Upscale probe'u artık SVGF kapalıyken
   FSR isteğinin **korunduğunu** ve `mode="auto"`'nun SVGF'siz **kabul edildiğini** doğrular;
   denoiser probe'u `still_noise` aralığını ve eski `static_samples` adının **reddini** doğrular.
   Bozuksa: "FSR refused without SVGF" → `RtApiUpscale.cpp` kapısı hâlâ duruyor (eski exe?);
   `static_samples` kabul edildiyse IPC alan listesi eski.

2. **Hareket AOV'u FSR'ı korur** — FSR açıkken, SVGF kapalıyken
   `render.set_motion enabled=false`. Görmen gereken: "disable upscale before disabling the
   motion AOV" hatası. Bozuksa: FSR hız vektörsüz çalışmaya çalışır (eskiden bunu yalnız SVGF
   kuralı dolaylı engelliyordu).

3. **Temiz sahnede duruş SVGF'si kendiliğinden durur** — az gürültülü bir dış sahne, OIDN
   kapalı, kamera dur. `render.realtime_denoiser_stats` birkaç saniye arayla.
   Görmen gereken: `static_active=true` ile başlar, `still_filtered_fraction` 1.0'dan düşer,
   kısa sürede `still_complete=true` olur ve `dispatches` artmayı keser.
   Bozuksa: fraction hep -1 → sayaç okunmuyor (slot etiketi / eşleme); hep 1.0 → shader M2'yi
   okumuyor (binding 15 eski .spv) ya da `still_noise` 0.

4. **★ Gürültülü iç mekanda duruş: örnek arttıkça DAHA TEMİZ** — yatak odası, OIDN kapalı,
   hedef 128. 8, 32, 64, 100 ve 127. örneklerde `viewport.get_screenshot` + aynı bölgede
   `render.probe` (duvar/tavan).
   Görmen gereken: gürültü monoton azalır; eskiden 32→64 arasında artıyordu (σ√N/64).
   `still_filtered_fraction` 64'te hâlâ büyük olabilir (doğru: oda 2% RSE'ye inmiyor);
   son çeyrekte (96→127) ham görüntüye yumuşak geçiş, 127'de **sıçrama yok**.
   Bozuksa: 8'den önce lekeli ham noktalar → güven eşiği çalışmıyor; 127'de sert geçiş →
   ev sahibi rampası (rawBlend) gelmiyor.
   **★ Sessiz arıza:** her şey "makul" görünür ama yakınsamış kare filtreli kalır. Kontrol:
   hedefe ulaşınca SVGF kapalıyken alınan 128 spp karesi ile **bit-bit aynı** olmalı
   (B3c ölçütü: meanabs 0).

5. **SVGF kapalı + FSR açık** — `render.set_realtime_denoiser mode=off`, FSR balanced,
   kamera hareket. `render.upscale_stats`. Görmen gereken: `active=true`,
   `input_provider="none"`, `svgf_filter_ms=0`; görüntü gürültülü ama doğru konumda, durunca
   tam çözünürlüğe döner ve birikim temiz. Sonra SVGF'yi aç: `input_provider="svgf"`.
   Bozuksa: hareket karesi siyah/eski → guard.save çalışmıyor (raw yedeği yok); durunca
   kalıcı bulanıklık veya FSR izi → guard.restore atlanıyor (birikim kirlendi — ciddi).

6. **SVGF + FSR regresyonu** — varsayılanlar (SVGF açık, FSR balanced), aynı oda, 14°
   yörünge. Görmen gereken: B4'teki gibi active 988×555, FSR ~1.1 ms, parlaklık durağanla
   aynı (bu oturumda 0.470 / 0.462). Bozuksa: guard sırası SVGF'nin yedek okumasını bozdu
   (SVGF artık backup'ı guard'dan alıyor).

## B4 — FSR 3.1 Vulkan, tek derleme ve parti sonunda tek IPC kabul turu

**Kullanıcı:** tüm shader'lar için `RayTrophiStudio/compile_shaders.bat`, ardından
C++ x64 Release. Bu tek partidir; ajan SDK/shader/C++ derlemesi çalıştırmaz.
Yeni shader: `upscale_inputs.comp` → `upscale_inputs.spv` (HDR16),
`upscale_inputs32.spv` (`RAW32`, HDR32). Push **64 bayt**, 8 storage image.
RT camera push **212**, B2 motion record **112**, SVGF push **64 bayt** değişmedi.
Yeni kaynaklar proje/filter'da. `FidelityFX.targets` DLL+MIT lisansı `$(OutDir)`'e
kopyalar; **x64/Release/amd_fidelityfx_vk.dll** şu anda hazır. SDK derlenmez,
import lib bağlanmaz. Uygulama yeniden açıldığında yeni Vulkan mantıksal cihaz
özellikleri de kullanılacak. Varsayılan upscale off; SDK varlığı hız PASS değildir.

**Statik kontrol:** IPC audit PASS (709 metot), descriptor `--check` PASS;
proje/filter/targets XML, 8 yeni C++ modül kaydı, probe AST ve Release aynaları
PASS. Bunlar C++/shader derlemesi veya canlı GPU doğrulaması değildir.

Ön koşul: Vulkan Rendered, standart perspektif/lens, SVGf ve motion açık;
OIDN viewport seçeneği ayrıca korunur. İç oda 1680x945, aynı bounce/pozlama,
motion profile kapalı; ardından bulut/gaz sahnesi. Dış named pipe
`scripts/test/rt_ipc.py`; gömülü script içinde bekleme döngüsü YOK.
Windows error 5 olursa aynı probe gerekli sandbox yükseltmesiyle yeniden koşulur.

1. **İLK: soğuk DLL-yok kapısı.** Yeni açılan uygulama henüz upscale off iken,
   FSR DLL'ini geçici olarak Release dışında tut; ilk mode=auto ve kamera
   hareketinde uygulama çalışmalı, available/active false + açıklayıcı reason,
   tam çözünürlük. Sonra DLL'i geri koy, mode off/on ve tekrar hareket: dll_loaded,
   available, active true; ffx_version provider'dan dolu. Zaten yüklenmiş DLL'i
   sonradan kaldırmak bu testi yapmaz. Başka DLL indirme/taşıma yok; orijinali koru.
2. **API atomikliği/bağımlılıklar.** Dış terminal:
   `python scripts/test/rt_probe_upscale_ipc.py`. 11 geçersiz güncelleme reddi,
   kısmi yazım yok, aktif upscale'nin SVGF/motion bağımlılığı korunmalı; başlangıç
   ayarları geri yüklenir. Python eş yüzeyleri `rt.render.set_upscale(...)`,
   `rt.render.upscale_stats(synchronize=False)`; yanlış bool/enum/range reddi.
3. **Boyut ve jitter.** Rendered'da
   `python scripts/test/rt_probe_upscale_ipc.py --motion-seconds 4`.
   balanced: render_w/h display'den küçük, active/dispatches artar; jitter phase
   SDK'dan (>8), authored Halton tercihi değişmez. Durunca active false ve
   render=display. ★ Sessiz arıza: mode fsr iken active false / dispatches sabit
   ise FPS karşılaştırmasını kazanım sayma; reason/inactive_reason oku.
4. **İşaret/ölçek.** Aynı 1° sağ dönüş: native RT mean_dx yaklaşık +28 px;
   balanced RT vektörü yaklaşık +28 * render_w/display_w. FSR ekranında kayma
   yine yaklaşık28 display px. Ters işaret, ölçek katlanması, yüzme/bulanıklık
   → motionVectorScale/jitter/depth sözleşmesi bozuk. Duran kamera ~0 px.
   Motion sample'ı hareket çağrısının hemen ardından al, sonraki dispatch üstüne yazar.
5. **Hız.** Mode off ve balanced için ayrı `set_gpu_timing enabled=true`
   pencereleri; RT ms/örnek medyanı ve ölçülen piksel boyutu. Balanced RT
   **<=0.5x** native hedefi. `upscale_stats synchronize=true`: GPU timing
   available true, gpu_ms + svgf_filter_ms OIDN fast ~44 ms'nin altında hedef.
   İlk context/pipeline kurulumu hariç. `realtime_denoiser_stats.total_ms` FSR'ı
   da içerdiğinden iki kez toplama; gerçek hareket FPS'ini ayrıca sayıyla yaz.
6. **Duruş/tek örnek.** OIDN açık/kapalı iki durumda ani duruşta eski poz,
   çift kontur veya kararma/gürültü sıçraması olmamalı. Bridge yalnız doğru
   poz/epoch/yüzeyde; slider düzenlemesinden sonra eski FSR görüntüsü karışmamalı.
   Target=1'de durunca eski düşük çözünürlüklü karede kalmamalı; son tonemap
   pompalanır. Çalışmazsa epoch/slot/reset/bridge kimliği hatalıdır.
7. **Referans.** Sabit pozlama/seed/zamanla native off ve hareketten sonra full
   128 örnek: ham radyans ve son ekran bit bit aynı hedef. Birikim sayısının
   ÖNCE sıfırlanmasını bekle; eski completed bayrağıyla erken çıkma. Filtreli veya
   upscaled radyansın Welford/OIDN girdisine sızması en ağır sessiz arızadır.
8. **Kalite.** Pervaz/avize/hasır/saç yörüngede titrememeli; hareketli nesne
   arkasında iz <4 kare; gaz/nükleer/bulut reactive bölgede hayaletsiz.
   Gürültü, parlaklık ve hayalet sayısını görüntü ölçümüyle raporla; yalnız FPS
   artışı kalite kabulü değildir. B3 §7 açık işlerini yeni B4 başarısı sayma.
9. **Yaşam döngüsü.** Hareket sırasında quality değiştirme, tek/çift viewport
   boyutlarıyla resize, Material/Solid/Rendered geçişi; siyah kenar/eski slot/
   hatalı boyut yok. Raster/durağan/final/debug ve desteklenmeyen lenslerde
   active false, tam çözünürlük; raster boşta gereksiz sürekli redraw olmamalı.
   Backend değişimi/device recovery eski DLL context/descriptors taşımamalı.
10. **Hata ve temizlik.** SDK/feature/context/shader hatasında full-res reason;
    başarısız FSR slot'u yayımlanmaz. Kamera ve tüm seçenekleri başlangıca al,
    kullanıcı projesini kaydetme. Geçici resimleri yalnız `docs/dev/.captures/`
    altında tut ve sonunda sil. Sonuçları buranın başına tek rapor olarak yaz;
    yeni bulgular plan §7'ye. Kabulden sonra kullanıcı auto'yu açar.

## B4 — SDK hazırlığı (2026-10-26) — kaynak/dağıtım doğrulandı

1. **PASS:** resmî SDK v1.1.4 arşivi (470,858,939 bayt), SHA-256 manifestte.
   `amd_fidelityfx_vk.dll`: x64, AMD Authenticode Valid, 5 FFX API export'u mevcut.
2. **PASS:** `x64/Release/amd_fidelityfx_vk.dll` + `FidelityFX-LICENSE.txt` hazır;
   mevcut `x64/Debug` kopyaları da aynı hash/lisans. Beş API başlığı third_party'de.
3. **PASS:** `scripts/setup_fidelityfx.ps1` yerel pinli arşivle çalıştı; PowerShell
   parser, dosya hash'leri, proje/filter/targets XML ve import/kopya bağlantıları
   doğrulandı. `FidelityFX.targets` sonraki normal x64 derlemelerinde `$(OutDir)`
   DLL+lisans dağıtımını yapar; SDK kaynak derlemesi veya import lib bağımlılığı yok.
4. **HENÜZ YOK:** FSR loader/context, düşük çözünürlük dispatch, UI/API ve canlı
   GPU kabulü. DLL indirildi/kopyalandı diye `available=true`/FPS kazanımı sayılmaz.
   Uygulama ve SDK DLL çalıştırılmadı. SDK hazırlığı için ayrı uygulama derlemesi
   gerekmiyor; §6 somut planı onaylandıktan sonra tüm B4 tek derleme partisi,
   canlı kabul listesi o parti sonunda güncellenecek.

## B3f SONUÇLARI (2026-10-26, canlı, iç oda, OIDN kapalı) — PASS; B3 ailesi KAPANDI

- (1) PASS durma anı: yakınsamış ham 0.470; hareket 0.475–0.481 (+%1–2); durunca 4. örnek
  0.469, 16. örnek 0.472 → düşüş ~%1, hareket karelerinin kendi oynamasından küçük
  (B3e: %4.5). Gürültü 0.026 → 0.033 (B3e: 0.023 → 0.039).
- (4) PASS 128 örnekte ekran bit bit aynı (static 128/256 vs 0).
- (2) eski poz / çift kontur ve (3) slider sıfırlaması: koşulmadı (kullanıcı gözü yeterli görüldü).

B3 ailesinden Sonra'ya kalanlar (plan §7): sağ dikey pervazda hafif noktalanma; hareket
karesi yakınsamıştan +%1–2 parlak; à-trous 6.8 ms (hedef ≤ 6); hayalet (<4 kare), hacim
reactive, bulutlu sahne siluetleri, timeline oynatımında SVGF tetiklenmemesi ölçülmedi.

## B3f — durma geçişi: dondurulmuş hareket geçmişi (2026-10-26, derlenmedi; YALNIZ realtime_svgf shader'ı)

Durağan karede SVGF artık hareket sırasında biriken geçmişi atmıyor: geçmiş DONDURULUR
(değişmeden ileri taşınır, birikime girmez) ve birikimle örnek sayısına göre harmanlanır,
`(geçmiş·n_h + birikim·N) / (n_h + N)`; geçmişin payı örnek geldikçe kendiliğinden söner.
Geçmiş yalnız SVGF geçmişi ardışık geçerliyse kullanılır (kameranın durmasıyla gelen
geçiş; slider düzenlemesi gibi hareketsiz sıfırlamada kullanılmaz).

1. **★ Durma anı.** İç oda, OIDN kapalı, static_samples 64: hareketten sonra her yeni
   örnekte ekran ortalaması ve gürültüsü. B3e: son hareket 0.487 / 0.023 → 4. örnek
   **0.465** / 0.039 → 16. örnek 0.479. Hedef: 1–8. örnek parlaklığı hareket karesinin
   %1 içinde, gürültü sıçraması yok.
2. **★ Sessiz arıza — eski poz:** dondurulan geçmiş bir önceki pozdan kalırsa durunca
   bir an çift görüntü / kayma olur. Hızlı dönüşten ani durmada kenarlarda çift kontur
   olmamalı.
3. Slider düzenlemesi (kamera sabit, örn. bounce değiştir): durağan kare geçmiş
   kullanmamalı (eski ayarın görüntüsü karışmamalı) — ilk örneklerde yeni ayar görünür.
4. 128 örnekte ekran bit bit aynı (B3d PASS korunmalı).

## B3e SONUÇLARI (2026-10-26, canlı) — ince hatlar düzeldi; durma anı FAIL

- ★ İnce hatlar: hareketli karede pervaz/avize kolu çizgileri belirgin şekilde sürekli;
  sağ dikey pervazda hafif noktalanma kaldı (B3d'den çok az).
- Hareket gürültüsü 0.031 → **0.023** (ön-filtre filtreyi güçlendirdi), parlaklık 0.487.
- ★ FAIL (kullanıcı gördü: "durunca kararma ve gürültü, hızla temizleniyor"): son hareket
  0.487/0.023 → durağan 4. örnek 0.465/0.039, 16. örnekte 0.479. Kök: durağan mod ~20
  karelik hareket geçmişini atıp 1–4 örnekli birikimin yalnız uzaysal filtresine geçiyordu
  (çarpık dağılım + kenar durdurucu → koyu, az örnek → gürültülü). → B3f.

## B3e — à-trous varyans ön-filtresi (2026-10-26, derlenmedi; YALNIZ realtime_svgf shader'ı)

C++ değişmedi; yalnız `realtime_svgf.comp` (iki varyant). Luminans kenar durdurucusunun
eşiği (`4·√varyans`) artık 3×3 Gauss ile ön-filtrelenmiş varyanstan (Schied 2017).

1. **★ İnce hatlar.** İç oda, durağan 8 örnek (static_samples 64, OIDN kapalı) ve hareket:
   pervaz olukları / avize kolu kırpıntısı yakınsamış kareyle — B3d'deki noktalı koyu
   çizgiler sürekli ve yakınsamış kadar soluk olmalı.
2. **Gürültü / parlaklık.** Hareket parlaklığı 0.482 civarı kalmalı; gürültü B3b'nin 0.031'inden
   kötüleşmemeli (ön-filtre filtreyi biraz güçlendirir, gürültü düşebilir).
3. **★ Sessiz arıza — fazla yumuşama:** doku/kontak gölgesi detayı yakınsamış kareden belirgin
   silik → ön-filtre eşiği fazla genişletti; à-trous ms aynı kalmalı (+9 okuma, ~0.3 ms).
4. Durağan eğri (B3c tablosu): 8 örnekte fark ≤ 12.9.

## B3d SONUÇLARI (2026-10-26, canlı, iç oda) — 3/3 PASS, kök yeni yerde

- (1) PASS sızıntı: static_samples 128 ve 256, max 128 → yakınsamış ekran static 0 ile bit bit
  aynı, static_active false.
- (2) PASS varsayılan 64 (yeni başlatmada).
- (3) ★ Geçmiş debug: hareket sırasında piksellerin %98'i yeşil (uzun geçmiş), sarı %1,
  kırmızı ~0; kırmızı/sarı yalnız kenar şeritlerinde serpiştirilmiş. Koyu noktaların çoğu
  YEŞİL piksellerde → kök geçmiş değil FİLTRE. Kullanıcı gözlemi: "ince hatları kırıyor".
  Durağan 8 örnekte de pervaz oluklarının 1 px koyu çizgileri yakınsamıştan koyu ve
  NOKTALI. Kök: à-trous luminans eşiği piksel başına ham varyanstan → komşu hat pikselleri
  rastgele farklı eşik alıyor. Makaledeki 3×3 varyans ön-filtresi eksikti → B3e.
- Not: B3c'deki kamera piksel izi geçmiş bulmayı iyileştirdi (kırmızı neredeyse yok) ama
  görünen koyu noktalar filtrenin işiydi; iki adım birlikte gerekliydi.

## B3d — durağan limit düzeltmesi + varsayılan 64 + SVGF geçmiş debug (2026-10-26, derlenmedi)

**Derleme:** `realtime_svgf.comp` (iki varyant) + C++; RT pipeline derlenmez.
- Durağan limit artık `hedef − 1` ile kırpılıyor (B3c'de hedefe eşit limit son ekrandaki
  kareye sızdı, aşağıda). `static_active` son durağan kareden sonra false'a döner.
- `static_samples` varsayılanı **64**.
- Ölçü aleti: `render.set_realtime_denoiser {debug_history:true}` (panel: "SVGF History
  Debug") → SVGF karelerinde piksel başına geçmiş uzunluğu: kırmızı = yok, sarı = < 4,
  yeşil rampa = uzun. IPC/Python/overlay/panel hazır, audit OK.

1. **Sızıntı yok.** static_samples 128 ve 256, max samples 128: yakınsamış ekran static 0 ile
   bit bit aynı (B3c: meanabs 0.84, max 27). `static_active` 128'de false.
2. **Varsayılan.** Yeni başlatmada static_samples 64; OIDN kapalı, kamera durunca
   `static_active` true, 63'ten sonra false.
3. **★ Koyu noktaların kökü (önce bu ölçülür, düzeltme sonra).** iç oda, hareket sırasında
   `debug_history:true` + ekran görüntüsü: hasır delikleri / pervaz kenarı / avize kolu
   kırmızı mı (geçmiş bulunamıyor) yoksa yeşil mi (geçmiş var, koyuluk başka yerden: albedo
   demodülasyonu / alfa / à-trous)? Sonuca göre düzeltme seçilir; tahminle yazılmaz.

## B3c SONUÇLARI (2026-10-26, canlı, iç oda) — durağan mod PASS (sızıntı düzeltmesi B3d), koyu noktalar SÜRÜYOR

- Panel: kullanıcı doğruladı (bounce + denoise örnek sayılarının altında).
- Durağan mod varsayılan 0 olduğu için kullanıcıda devreye girmedi (tasarım gereği, ama
  yanlış varsayılan → B3d 64). 64 ile: örnek 2–63 her dispatch static_active, 64'te durur.
- ★ Kalite eğrisi (yakınsamış 128 spp ham kareye ortalama mutlak fark, 8-bit; iki koşu ort.):

  | örnek | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
  |---|---|---|---|---|---|---|---|---|
  | static 0 | 26.7 | 33.6 | 30.2 | 30.6 | 23.1 | 16.6 | 13.3 | 11.1 |
  | static 64 | 26.6 | 24.4 | **14.1** | **12.9** | 11.9 | 11.4 | 11.1 | 11.1 |

  8 filtreli örnek ≈ 48 ham örnek. Maliyet: limit altında ~12.5 örnek/sn, üstünde ~14.4 (~%13).
- ★ FAIL sızıntı: limit 128 = hedef 128 → yakınsamış ekran meanabs 0.84 / max 27 farklı
  (256 de aynı: 128'e kırpılıyor); limit 100 → bit bit aynı. Kök: asenkron ping-pong ekranda
  bir önceki submit'i gösterir ve birikim bitince son submit tüketilmez → ekrandaki kare
  127. örnek, o da hâlâ filtreli. Düzeltme B3d (limit ≤ hedef − 1).
- Koyu tekil noktalar (hasır delikleri, pervaz, avize kolu) kamera piksel izinden sonra da
  SÜRÜYOR → tahmin yanlış/eksik; B3d'de geçmiş debug görünümüyle ölçülecek.

## B3c — izole piksel geçmişi (2026-10-26, derlenmedi; yalnız realtime_svgf + C++)

B3b canlı testinde parlak kenar hattı gitti ama ince/izole özelliklerde (hasır
delikleri, pervaz kenarı, avize kolu) **koyu tekil noktalar** kaldı. Kök: piksel izi
yalnız aynı malzemeli komşudan ölçülüyordu; komşusu olmayan pikselde 0 → tutarlılık
testi reddediyor, à-trous malzeme sınırını geçmiyor → piksel 1 spp kalıyor (parlakları
ateş böceği kırpması alıyor, koyular kalıyor). Düzeltme: aynı malzemeli komşu yoksa iz
= kamera uzaklığı × piksel açısı / max(|cos|, 0.2). `VulkanMotionHistory::cameraFootprint`
(prepareDispatch'e verilen vektörlerden), SVGF push 32 → **48 bayt** (`camera` vec4).

1. Rendered açılıyor, `realtime_denoiser_stats` available true (push ABI tuttu).
2. **★ Koyu noktalar:** iç oda, hareket sırasında avize/pervaz/hasır kapak kırpıntısı
   aynı pozdaki yakınsamış kareyle: tekil koyu noktalar büyük ölçüde gitmeli.
   history_length_mean B3b'deki 20.3'ten yükselmeli.
3. **★ Sessiz arıza — hayalet:** izole pikseller artık geniş toleransla geçmiş kabul
   ediyor; hızlı 3° dönüşte ince özelliklerin arkasında iz/çift görüntü olmamalı
   (iz ≤ 4 kare). Varsa fallback çarpanı (0.2 alt sınır / 3 iz) fazla gevşek.
4. Parlaklık B3b ile aynı kalmalı (hareket 0.482 / yakınsamış 0.484).
5. Bulutlu sahne siluet noktaları (B3b madde 5, henüz koşulmadı).

**B3c'ye eklenenler (aynı derleme):**
- **Durağan mod** `static_samples` (0 = kapalı, varsayılan 0; 0..1024): kamera duruyor ve
  viewport OIDN KAPALI iken SVGF yalnız uzaysal kısmını birikmiş görüntü üzerinde
  çalıştırır (temporal yok: birikim zaten yansız temporal filtre), varyans = birikimin
  uzaysal yayılımı (yakınsadıkça küçülür), görüntü `örnek / limit` ile ham birikime söner.
  Limit hedef örnek sayısıyla kırpılır → yakınsamış kare hiçbir zaman filtrelenmez.
  OIDN açıkken devreye girmez (OIDN durağan kareye sahip). Push 48 → **64 bayt**
  (`sampleCount`). IPC/Python `static_samples`, istatistikte `static_active`. Panel:
  "SVGF On Still Frames" (0..256).
- **Panel düzeni:** iki render panelinde de Light Paths (bounce + RR + hareket profili +
  Motion Vectors) ve Denoise (SVGF, sonra OIDN + takvim) bölümleri örnek sayılarının
  hemen altında (`DrawBounceControls`, `DrawDenoiseControls`, ortak). Eski "Light
  Transport & Denoise" / "Light Paths" bölümleri "Light Transport" oldu (kostik, ışık
  şaftları, debug görünümü); ayrı "Denoising" bölümü kaldırıldı. OIDN kutusu
  "Enable Denoiser (OIDN)".

6. **Panel.** İki panelde de Sampling bölümünde Max Samples'ın altında Light Paths ve
   Denoise; Light Transport'ta bounce/denoiser tekrarı YOK; OIDN kapalıyken SVGF
   ayarları görünür. (Panel IPC'den sürülemez — kullanıcı gözü.)
7. **API.** Probe script `static_samples` geçersizlerini (−1, 2000, 1.5) reddetmeli,
   rejections 15.
8. **★ Durağan mod kalite eğrisi (iç oda, OIDN KAPALI).** `static_samples` 0 ve 64 ile
   kamerayı 0.0001 kıpırdatıp örnek 1, 2, 4, 8, 16, 32, 64'te ekran görüntüsü; her
   birinin 128 örnekli ham yakınsamış kareye ortalama mutlak farkı. Beklenen: SVGF'li
   eğri düşük örnekte belirgin altta, 64'te ikisi eşit (fade = 1). `static_active`
   1..63'te true, 64+ false. Kesişme noktası (ham birikimin SVGF'yi yakaladığı örnek)
   varsayılan limit adayıdır — sayıyı yaz.
9. **★ Sessiz arıza — durağan mod yakınsamış kareye sızıyor:** `static_samples` 256,
   max samples 128: 128'de görüntü ham kareyle bit bit aynı olmalı (limit hedefe
   kırpılıyor). Değilse son kare filtreli kalıyor ve kimse fark etmez.
10. **Maliyet.** Durağan karede örnek/sn `static_samples` 0 vs 64 (SVGF ~10 ms/örnek,
    RT ~95 ms → ~%10 beklenir), yalnız limit altında.

## B3b SONUÇLARI (2026-10-26, canlı, iç oda) — 6 PASS, 1 kısmi, 2 koşulmadı

- (1) PASS varsayılan açık: yeni başlatmada IPC yazmadan mode svgf, firefly_clip 3.
- (2) PASS probe: api_validation PASS, 12 ret; `variance_clip` "unknown field" ile reddedildi.
- (3) PASS (API yolu): SVGF açıkken `set_motion enabled=false` reddedildi, motion açık
  kaldı. Panel tıklaması IPC'den sürülemez — kullanıcı gözüyle doğrulanacak.
- (4) ★ PASS parlaklık (profil kapalı): hareket SVGF **0.482** vs yakınsamış OIDN 0.484
  (**−%0.5**; B3: −%11). firefly_clip 0.5/3/10 → 0.478/0.482/0.484 (etkisiz),
  à-trous 1 iterasyon 0.479. Gürültü 0.031 (B3 0.030), OIDN her kare 0.012.
- (5) KISMİ siluet: iç odada parlak kenar hattı yok; ince/izole özelliklerde koyu
  tekil noktalar kaldı (kullanıcı da gördü) → B3c. Bulutlu sahne koşulmadı.
- (6) Hayalet: koşulmadı (B3c ile birlikte).
- (7) Maliyet: total 10.3 ms (temporal 2.37, à-trous 6.82) — temporal'ın 9 tap +
  kayıt okuması ölçülebilir maliyet eklemedi; history_length_mean 20.3.
- (8) PASS referans: 128 örnekte on/off/on bit bit aynı.

## B3b — SVGF geçmiş düzeltmesi + panel + varsayılan açık (2026-10-26, derlenmedi)

**Derleme:** yalnız `realtime_svgf.comp` (iki varyant: `realtime_svgf.spv` +
`realtime_svgf32.spv`, compile_shaders.bat ikisini üretir) + C++. **Raygen
DEĞİŞMEDİ → RT pipeline yeniden derlenmez.** ABI: SVGF descriptor 13 → 15 binding
(13/14 = raygen motionHistory0/1, rgba32ui), push yine 32 bayt (alan adları
`historyFlags`, `fireflyClip`). IPC/Python alanı `variance_clip` → **`firefly_clip`**
(anlamı değişti: artık geçmişi değil yalnız geçmişsiz pikseli kırpar); eski ad
"unknown field" ile reddedilir. Descriptor üretildi, audit OK.

Ne değişti:
- Temporal geçiş geçmişi artık KIRPMIYOR (B3: −%11 karartmanın kökü).
- Geçmiş araması: önce 2×2 bilineer ayak izi, her tap için tutarlılık (aynı
  malzeme, normal > 0.8, teğet düzlem mesafesi < 0.5 piksel izi, uzaklık < 3 iz;
  piksel izi = aynı malzemeli komşu kayıtların en kısa adımı); hiçbiri tutmazsa
  3×3 arama. Raygen'in kendi testinin kabul ettiği tap her zaman geçerli
  (hareketli rijit nesneler bugünkü davranışı korur).
- Geçmişi < 4 olan pikselin O ANKİ örneği ateş böceği kırpmasından geçer
  (aynı malzemeli 3×3 komşu, merkez hariç, ortalama + k·σ, k = firefly_clip 3).
- Varsayılan AÇIK (`RealtimeDenoiserSettings::enabled = true`; proje dosyasına yazılmaz).
- Panel: "Denoise While Moving (SVGF)" + ayarları Light Paths maliyet bloğunda,
  Motion Vectors altında; OIDN anahtarına bağlı değil. "Motion Vectors" kutusu
  artık `rtapi::setMotion` ile yazıyor — SVGF açıkken kapatma hata mesajıyla reddedilir.

Kontrol (sıra: bağımsız ve hızlı önce):
1. **Varsayılan açık.** Uygulamayı yeni başlat, sahne aç, hiçbir IPC yazmadan
   `render.realtime_denoiser_stats {synchronize:false}`: mode `svgf`, firefly_clip 3,
   available true (ilk hareketten sonra). Panelde kontrol viewport OIDN KAPALIYKEN
   de görünüyor. mode off → globals varsayılanı tutmadı.
2. **API.** `python scripts/test/rt_probe_realtime_denoiser_ipc.py` → api_validation
   PASS, 12 ret. Ek: `render.set_realtime_denoiser {variance_clip:3}` reddedilmeli.
3. **Panel bağımlılığı.** SVGF açıkken panelde Motion Vectors'ı kapat → kutunun
   altında "disable SVGF before disabling the motion AOV", `motion_stats.enabled`
   true kalır. Sessizce kapanıyorsa panel API'yi atlıyor.
4. **★ Parlaklık (iç oda, hareket profili KAPALI).** B3 yöntemi: hareket
   sırasında ekran ortalama luma vs yakınsamış OIDN (0.491). B3: SVGF 0.435
   (−%11). Hedef: **yakınsamışın %3 içinde**, gürültü ≤ 0.035 (B3 0.030).
   Hâlâ −%5'ten koyu → kalan kayıp tonemap Jensen'i ya da à-trous; iterasyon
   1 vs 4 ile ayır (B3'te à-trous etkisizdi).
5. **★ Siluet noktaları (bulutlu dış sahne).** Hareketli SVGF karesinde dağ sırtı
   ve uzak arazi kırpıntısı: B3'teki beyaz nokta hattı kaybolmalı. Debug 17'deki
   kırmızı piksellerle aynı yerlerde kalan nokta varsa 3×3 arama tutmuyor
   (piksel izi 0 mı: ince/izole özellik).
6. **★ Sessiz arıza — yanlış kabul edilen geçmiş:** kenarlarda iz/hayalet (hareket
   bulanıklığı gibi görünür, kimse hata demez). İç odada nesneyi +0.05 y taşıyıp
   ve hızlı 3° dönüşte arkada kalan kenarda iz 4 kareden kısa olmalı. Uzun iz →
   tutarlılık eşiği gevşek (düzlem 0.5 iz / uzaklık 3 iz).
7. **Maliyet.** `realtime_denoiser_stats {synchronize:true}` hareket medyanı:
   B3 total 10.1 (temporal 2.4, à-trous 6.7). Temporal 9 tap + kayıt okuması
   ekledi; ~3–4 ms beklenir. history_length_mean B3 20.6'dan yüksek olmalı
   (siluetler artık geçmiş biriktiriyor).
8. **Referans.** 128 örnekte SVGF on/off ham kare bit bit aynı (B3 PASS) — değişmemeli.
9. **Temizlik.** Kamera/nesne/ayarlar başlangıca; görüntüler silinir.

## B3 SONUÇLARI (2026-10-26, canlı, iç oda 1680×945, RTX 3060) — 8 PASS, 1 FAIL (parlaklık), 3 ölçülmedi

- (1) PASS raster kamera hızı: 1° dönüş dx **+28.47** (RT 28.46), dy −0.01, max 34.6,
  ret %2.2 (≈ giren şerit %1.7 + kenar).
- (2) PASS TAA boşta birikiyor: 2 → 4 → 6 → 8 (0.7 sn arayla), artık 1'de kalmıyor.
  Yavaş (~3 kare/sn boşta) — not, hata değil.
- (3) PASS raster nesne hızı: +0.05 y → dy +0.049, max 6.26 (RT 6.3), durunca 0.001;
  raster_objects_available true; Solid'de raster_available false + doğru açıklama.
  Çoklu instance / culling açık tekrarı YAPILMADI.
- (4) PASS `rt_probe_realtime_denoiser_ipc.py`: api_validation PASS, 12 ret,
  motion bağımlılığı reddi doğru.
- (5) PASS politika: SVGF açıkken 4 sn harekette SVGF +70 dispatch, viewport OIDN
  2 çağrı (ilk kare); kapalıyken OIDN 51 çağrı. moving/presenting true.
  ★ SVGF viewport OIDN anahtarından BAĞIMSIZ çalışıyor (OIDN kapalı: +57 dispatch).
- (6) ÜSTÜNDE: total medyan **10.1 ms** (hedef ≤6): à-trous 6.7 (4 iter), temporal
  2.4, variance 0.6. OIDN fast 44 ms'ye göre 4.3× ucuz. history_length_mean 20.6.
- (7) Gürültü (ekran JPEG, komşu piksel farkı — P0 metriğiyle AYNI DEĞİL, yalnız
  göreli): hareket ham 0.19–0.25, SVGF 0.030, OIDN her kare 0.012, yakınsamış 0.014.
- (8) Hayalet/ayrılma: ÖLÇÜLMEDİ.
- (9) PASS referans: 128 örnekte SVGF on/off/on ham kareleri **bit bit aynı**
  (meanabs 0, max 0) — filtre birikime sızmıyor.
- (10) Hacim sahnesi + final: ÖLÇÜLMEDİ (sahne açılmadı).
- (11) B2 hız maliyeti tekrar: +4.0 / +0.6 ms → ortalama ~2 ms, gürültü bandında.
- ★★ **FAIL — hareket halinde karanlık (kullanıcı fark etti: "durunca OIDN gelince
  sahne birden aydınlanıyor").** Ekran ortalama parlaklığı (yakınsamış OIDN 0.491):
  SVGF + hareket profili 0.379 (**−23%**), SVGF profil kapalı 0.435 (**−11%**),
  OIDN her kare 0.489 (0). İki ayrı neden:
  a. Hareket profili 2 bounce → iç mekânda dolaylı ışık kesiliyor (~12 puan).
     Fiziksel, SVGF'nin hatası değil; profil bir hız/kalite takası.
  b. SVGF temporal geçmiş KIRPMASI enerji kaybettiriyor: variance_clip 0.5 →
     0.412, 3 → 0.437, 10 → 0.461; à-trous 1..5 iterasyon parlaklığı DEĞİŞTİRMİYOR
     (0.433–0.437). 1 spp yol izleme örnekleri çarpık dağılır; 3×3 komşuluğun
     ortalama ± kσ penceresi çoğu pikselde gerçek ortalamanın ALTINDA kalır ve
     doğru geçmişi aşağı kırpar. Kırpma TAA'dan gelme; Schied 2017 SVGF geçmişi
     kırpmaz (yalnız geometri testiyle reddeder). Kalan ~−6% (clip 10) muhtemelen
     tonemap Jensen'i (artık gürültü konkav eğride karartır) — düzeltmeden sonra ölç.
- ★ UI: SVGF kontrolü `DrawDenoiserScheduleControl` içinde, yani yalnız viewport
  OIDN açıkken görünüyor; ama SVGF ondan bağımsız çalışıyor → OIDN kapalıyken SVGF
  açık ve düğmesi GİZLİ. Panel çekirdeği göstermiyor (CLAUDE.md "panel yalan").

- ★★ **İKİNCİ BULGU (bulutlu dış sahne, kullanıcı fark etti: "obje sınırlarında ışıklı
  hat"):** hareketli SVGF karesinde siluetler (dağ sırtı, yatık uzak arazi) boyunca
  beyaz noktalar; köprü ve yakınsamış OIDN karesinde yok. Debug view 17 ile birebir
  çakışıyor: o pikseller HER KARE geçmiş reddi alıyor. Kök: B2 geçmiş testi tek
  pikselli ve birincil ışın rastgele alt-piksel jitter'lı → siluette vuruş her kare
  nesne/arka plan arasında değişir, malzeme testi reddeder, piksel hiç geçmiş
  biriktirmez (1 spp kalır). À-trous luminans kenar durdurucusu parlak aykırı merkezi
  komşulardan ayırdığı için bu 1 spp pikseller PARLAK nokta olarak kalır (çarpık
  dağılım: karanlıklar ortalanır, parlaklar korunur). Düzeltme yönü (B3b): SVGF
  makalesindeki gibi 2×2 bilineer ayak izinde tap başına tutarlılık + 3×3 arama
  yedeği (biri tutarsa kabul), reddedilen/kısa geçmişli piksellere yalnız ORADA
  ateş böceği kırpması. B2 RT ret oranı bu sahnede durgun %0.7.
- **Bulutlu sahne hız:** SVGF devrede (12 sürükleme karesinde SVGF 11, OIDN 2–3),
  ama kare süresi aynı (~600 ms/iter her iki modda): hareketin her ilk örneği RT
  ~470 ms (durgunda 160–330 ms'ye iner), OIDN eşzamanlı çalışıp RT'nin gölgesinde
  kalıyor (duvar 560–760 ms, GPU paylaşımı). Bu sahnede denoiser seçimi FPS'i
  değiştirmez; kaldıraç RT (ilk örnek maliyeti, B4 ölçek). Bulutlar gökyüzü yolunda
  (reactive 0); timeline oynatımı SVGF'yi tetiklemiyor (yalnız kamera + TLAS seri).

**Sonraki küçük parti (B3b) önerisi:** (1) geçmiş kırpmasını kaldır ya da
yalnız reactive/ayrılma bölgesinde uygula; kabul: hareket parlaklığı profil kapalı
yakınsamışın %3 içinde, gürültü ≤ 0.035. (1b) siluet geçmişi: 2×2 ayak izi + 3×3 arama, ret piksellerinde ateş böceği kırpması.
(2) SVGF kontrolünü OIDN bloğundan
çıkar, "Fewer Bounces While Moving" yanına koy. (3) Ardından varsayılan AÇIK
kararı (aşağıda). (4) à-trous 6.7 ms: fp16 ara görüntü / paylaşılan bellek — ölç.

**Varsayılan açık kararı (kullanıcı sordu):** sayılar destekliyor — hareket
karesi OIDN'in ~¼ maliyeti, referans bit bit korunuyor, durgun karede hiçbir şey
değişmiyor. ŞART: (1) parlaklık düzeltmesi, (2) UI yerleşimi. Hareket profili
varsayılan kapalı kalmalı (açılırsa karartma beklenen takas, yardım metninde yazmalı).

## B3 — tek derleme, ardından tek kabul turu (2026-10-26)

Önce kullanıcı `RayTrophiStudio/compile_shaders.bat` ile **tüm shader'ları**,
sonra aynı partide C++ derler. Ajan derleme çalıştırmaz / uygulamayı başlatmaz.
Yeni shader: `realtime_svgf.comp` → `realtime_svgf.spv` (HDR16),
`realtime_svgf32.spv` (`RAW32`, HDR32); sürücü ikisini de üretir.
Değişenler: `material_preview.vert`, `material_preview_frag.frag` ve covered
varyantı, `raster_cull.comp`, `raster_taa.comp`.

ABI: raster HDR 4 → 5 attachment (4 renk + derinlik), material vertex 6 binding /
12 attribute; cull 8 → 10 storage binding; TAA 5 → 6 binding, push yine **80 bayt**.
SVGF push **32 bayt**, 12 storage image + 1 storage buffer. Yeni `.cpp` dosyaları
vcxproj ve filters'a kayıtlı. OptiX değişmedi.

**Kaynak son kontrolü (2026-10-26):** IPC capability audit PASS (707 metot),
üretici `--check` PASS; proje/filter XML ve 7 yeni modül kaydı, probe AST +
Release kopyası, shader binding/push kaynak incelemesi ve `git diff --check`
PASS. Bunlar derleme veya canlı GPU kabul testi değildir.

**Başlangıç:** iç oda, 1680×945, sabit pozlama, aynı kamera ve bounce ayarları.
SVGF varsayılan **off**; 4 iterasyon / 24 geçmiş / variance_clip 3. Ayarlar
B2 hız denetimleri gibi oturumluk, proje dosyasına yazılmaz.
IPC dış terminalde `scripts/test/rt_ipc.py` üzerinden; gömülü script alanından
çağrılmaz. Named pipe error 5 olursa probe aynı transportla yükseltilmiş çalışır.

1. **İLK: B2 raster kamera düzeltmesi.** Material/RayFusion, TAA açık,
   bir kare geçmiş hazırken 1° sağa dönüş ve hemen
   `render.motion_stats {sample:'raster'}`: mean_dx ≈ **+28 px**, dy ≈ 0.
   Ret: giren şerit + kenarlar/disocclusion. Hâlâ %100 ret / sıfır hız:
   eski SPV, `historyFlags` ayrımı veya önceki kamera hattı bozuk.
   B3 derinlik reddi yeni: eski beklenti olan yalnız giren şeride ek ret
   olabilir; nesne kenarlarıyla ilişkili olduğunu sayısal olarak not et.
2. **Boşta TAA 16'ya gidiyor.** Kamerayı bırak, `viewport.taa` raporunu
   birkaç saniye aralıkla oku: accumulated_samples 1 → target_samples (16),
   converged true; ardından CPU/GPU gereksiz çizimle meşgul kalmamalı.
   1'de kalırsa dış `needsViewportRender()` kapısı; her kare 1 olursa
   yakınsama isteği yanlışlıkla sahne dirty'sine çevrilmiş demektir.
3. **Raster nesne hızı + culling eşleşmesi.** Material modunda sabit kamerada
   bir nesneyi +0.05 y taşı ve hemen raster motion oku: nesnede dy pozitif,
   arka plan ~0; durunca hız ~0. `raster_objects_available` true olmalı.
   GPU culling açıkken birden fazla instance/proxy ile tekrarla: başka
   nesnede vektör oluşması önceki matrislerin compaction sırasının bozulmasıdır.
   Görünürlük/topoloji değişiminde ilk yeni nesne kareleri geçmişi reddetmeli.
   Material per-mesh fallback kamera hızı taşır; `raster_objects_available=false`
   bunu bildirir. Solid/Matcap mevcut tasarımda TAA geçişini çalıştırmaz;
   raster_available=false ve açıklama beklenir, eski Material AOV okunmamalı.
   Material → Solid geçişinde TAA yüzünden sonsuz yeniden çizim olmamalı.
4. **Yüzey ve atomik doğrulama.** Dış terminalde
   `python scripts/test/rt_probe_realtime_denoiser_ipc.py` çalıştır.
   12 geçersiz ayar reddi, kısmi yazım yok, açık SVGF'nin motion bağımlılığı
   korunuyor ve başlangıç ayarları geri yükleniyor → API validation PASS.
   Python eş yüzeyleri: `rt.render.set_realtime_denoiser(...)`,
   `rt.render.realtime_denoiser_stats(synchronize=False)`.
   Embedded Python yalnız ayar/okuma için; kare bekleyen kabul testi IPC'den.
5. **SVGF hazır ve politika doğru.** `render.set_motion {enabled:true}`;
   `render.set_realtime_denoiser {mode:'svgf'}`. Rendered'da kamera hareketi:
   available/moving/presenting true, dispatches artıyor; viewport OIDN calls
   hareket süresince artmamalı (ilk kaynak kurulumu/slot gecikmesi sınırı hariç).
   Mode='off' anında ham/OIDN yoluna dönmeli. available=false ise reason'ı oku;
   off ve debug/final yolda SVGF dispatches sabit kalmalı. **Sessiz arıza:**
   mode='svgf' görünürken dispatches artmıyor; bunu performans kazanımı sayma.
6. **GPU maliyeti.** Hareketten sonra `render.realtime_denoiser_stats
   {synchronize:true}`: gpu_timing_available true, temporal_ms, variance_ms,
   atrous_ms ve total_ms son tamamlanan GPU sorgularından gelir. CPU kayıt
   süresi değildir. history_length_mean 1..history_max aralığında olmalı.
   total_ms **≤6 ms** hedefi; toplam backup/compose/tonemap/restore'u da
   içerir (geçiş sürelerinin toplamından büyük olması beklenir). İlk shader
   kurulumu hariç, birkaç hareket karesinin medyanını yaz. Aşılıyorsa hangi
   geçişin baskın olduğunu raporla; partinin içine yeni optimizasyon sokma.
7. **Gürültü.** Aynı pozlama/kamera hareketi dizisinde ham 1 spp, OIDN fast,
   SVGF için komşu piksel farkını ölç. P0: ham **0.065**, OIDN **0.012**;
   SVGF OIDN'e yakın olmalı. Sadece FPS artışını kalite PASS sayma.
   Doğrulama görüntüleri `docs/dev/.captures/` altında geçici; sayıları yazıp sil.
8. **Hayalet ve ayrılma.** Hareketli nesne arkası, yeni açılan duvar,
   ekran kenarından giren yüzey: iz **4 kareden kısa** olmalı. Nesne durunca
   vektörün sıfırlanması ve yeni bölgede history length düşmesi birlikte
   doğrulansın. Kamera ileri/geri ve saf dönüş ikisini de kullan.
9. **Hareketten durgunluğa geçiş + referans.** Durunca ilk 8 örnekte SVGF
   görüntü köprüsü ham birikime söner, OIDN B1 takvimine döner. SVGF geçmişi
   Welford tahminine yazılmaz. 128 örnekte mode off/on ham HDR ve son OIDN
   görüntüsü aynı olmalı (sabit pozlama, aynı seed). Görünmez ama ağır hata:
   filtrelenmiş görüntünün progressive/OIDN girdisine sızması; 128 örnek
   karşılaştırması bunu yakalar. Ayar değiştirme, viewport resize ve backend
   değişiminde eski çözünürlük/epoch görüntüsü görünmemeli.
10. **Hacim reactive + final.** Nükleer/gaz sahnesinde hareket: reactive
    bölgede geçmiş ağırlığı azalır, yüksek reactive'de à-trous atlanır;
    hacim detayları yüzey normaliyle yıkanmamalı. Sonra final render:
    SVGF dispatches değişmez, OIDN mevcut yüksek kalite yolunda çalışır.
11. **B2 maliyet takibi.** SVGF off iken `render.set_motion` açık/kapalı,
    aynı oda/ayar ve `render.set_gpu_timing {enabled:true}` ile ayrı pencere:
    RT ms/örnek karşılaştır. B2 farkı ~3.5 ms (%3.5). SVGF on için RT ve
    SVGF maliyetlerini ayrı yaz. Geçmiş yazımı / önceki konum / register
    paylaştırması henüz GPU profiler ile ölçülmedi; tahmini bulgu diye yazma.
12. **Temizlik ve tek rapor.** Kamera/nesne dönüşümü/denoiser/motion/debug
    ayarlarını başlangıca al; kullanıcı projesini kaydetme. Sayısal B3
    sonuçlarını buranın en üstüne tek seferde yaz; yeni bulgular plan §7'ye.
    Geçici doğrulama ekran görüntülerini sil. B1'in RR gürültü ve geçici
    proje roundtrip işleri ayrı Sonra maddeleri olarak kalır.

## B2 — denoiser kopya etiketi + hız AOV + RayFusion hızı + jitter (2026-10-26) — CANLI 11/12, raster düzeltmesi derleme bekliyor

**Sonuçlar (iç oda, 1680×945, RTX 3060):**
- (1) pipeline ready (ilk derleme 213 sn), görüntü normal.
- (2) ★ converged_only eski-kare hatası DÜZELDİ: 0.25 birim kaydırıp yakınsatınca
  denoise karesi yeni ham kareyle faz korelasyonu 0 px (fark 9.9), eski pozla
  fark 23.7 (B1: 6 px kayma).
- (3) every_sample 8.61 örnek/sn, 128 çağrı, 44.7 ms/çağrı — B1 ile aynı;
  senkron kopya yolu her çağrıda tetiklenmiyor.
- (4) render cihazı: rt_available/enabled/history_valid true, dispatch artıyor.
- (5) Duran kamera mean 0.0001 px. **rejected %4.1** (beklenti <%1 yanlıştı):
  debug view 17'de yalnız geometri kenarları (hasır kapaklar, avize, kitaplar) —
  rastgele alt-piksel jitter kenarda birincil vuruşu başka örneğe düşürüyor,
  doğru davranış.
- (6) Saf 1° sağa dönüş: mean_dx **+28.5 px** (işaret doğru), dy −0.01, max 34.6
  (merkez 25.9 × sec² kenarı ✓).
- (7) Dönüşte rejected %6.4 (≈ %4.1 kenar + %1.7 giren şerit), durunca %4.2.
- (8) Nesne +0.05 y: dy +0.044 ortalama, max 6.3 px, object_motion_dispatches her
  taşımada +1.
- (9) Debug 16 düz gri, 17 koyu mavi + kırmızı kenarlar.
- (10) ★ **FAIL — raster:** material modunda `sample='raster'` hep %100 rejected,
  hareket 0. Kök: TAA kamera hareketinde renk geçmişini sıfırlıyor
  (`m_taaFrameIndex = 0`) ve hız yazımı aynı `historyValid` bayrağına
  bağlıydı → hareketli her kare "geçmiş yok". Düzeltme yazıldı: push alanı
  `historyFlags` (bit 0 renk geçmişi, bit 1 önceki kamera), hız bit 1'e bağlı.
  Ayrıca: raster viewport boştayken yeniden çizmiyor gibi
  (`viewport.taa accumulated_samples` 1'de kaldı) — B3'te bak.
- (11) Halton: jitter 8 fazlı (0.125,0.278) (0.375,0.056) (0,−0.167) (0.25,−0.389) …,
  duran kamerada hız yine 0.0001.
- (12) Maliyet: hız açık 94.5 / 99.5 ms, kapalı 91.4 / 95.3 ms → **+~3.5 ms (%3.5)**,
  eşik %2'nin üstünde. Kapalıyken de kod shader'da (yalnız bayrakla atlanıyor),
  B1 tabanının (96–100) üstüne çıkmadı. B3'te ölç: hangi kısım (geçmiş
  yazımı / önceki konum tersi / canlı register).

**Raster düzeltmesinin kontrolü (B3 derlemesinde):** material modunda 1° dönüş →
`sample='raster'` mean_dx ≈ +28 px, rejected ≈ giren şerit; hareketten sonra
ilk karede w=0.

Ayrıntılı madde listesi (B2 derlemesi için yazılan):


Plan: RT_REALTIME_DENOISE_UPSCALE_PLAN.md §4. **Önce `compile_shaders` — TÜMÜ:**
`rt_payload.glsl` değişti (payload'a `primaryInst`), yani onu içeren her RT
shader'ı (raygen, closesthit, hair/volume closesthit, miss, photon …) ve
`raster_taa.comp`. RT pipeline önbelleği geçersizleşir: ilk Rendered açılış
~3 dk derler (`render.rt_pipeline_status` "compiling" → "ready"), bu hata değil.
Push constant 200 → 212 bayt; RT descriptor seti 38 → 41 binding; raster TAA
seti 4 → 5. Yeni dosyalar: Backend/VulkanMotionHistory.*, Backend/VulkanMotionAov.cpp,
Api/RtApiMotion.* (vcxproj + filters'a eklendi). Yeni IPC: `render.motion_stats`,
`render.set_motion`; `debug_view` 16 (Motion Vectors), 17 (Motion History).
Panel: Light Paths altında "Motion Vectors" + "Halton Jitter". Audit OK.

Sözleşme (RT ve raster aynı): xy = noktanın önceki karedeki izdüşümü − şimdiki
izdüşümü (piksel, y aşağı), z = reactive, w = 1 geçmiş reddedildi.
Raster: yalnız KAMERA hareketi, yalnız viewport TAA açıkken.

1. **Shader ABI / görüntü aynı.** Rendered açılıyor, pipeline "ready", güzel
   görüntü B1 ile aynı (hız AOV ışıltıyı değiştirmez). Siyah ekran / pipeline
   hatası → eski .spv (payload veya push constant ayrışık; static_assert 212).
2. **★ Denoiser eski kare düzeltmesi.** B1'deki test aynen: denoiser açık,
   `schedule=converged_only`, kamerayı 0.25 birim kaydır, yakınsat, ekran
   görüntüsü al; denoiser'ı kapat, ham kareyi al → faz korelasyonu 0 px
   (B1'de 6 px idi). Hâlâ kaymışsa etiket karşılaştırması tutmuyor.
3. **every_sample hızı düşmedi.** every_sample'da 128 örneğe örnek/sn ≈ 8.6
   (B1 ölçümü). Belirgin düşüş → her çağrı senkron kopya yoluna giriyor
   (etiket yanlış: `m_currentSamples - m_denoiserCopySamples` hep > 1).
4. **motion_stats temel.** `render.motion_stats`: render cihazı satırında
   `rt_available` true, `rt_enabled` true, `dispatches` artıyor,
   `rt_history_valid` true. rt_available false → görüntüler ayrılmadı (log'da
   "Motion AOV allocation failed").
5. **Duran kamera = 0.** DURAN kamerada `render.motion_stats @{ sample='rt' }`:
   `mean_motion_px` < 0.05, `rejected_fraction` küçük (< %1; ince kenarlar).
   ★ Sessiz arıza: mean_motion ~0.3–0.7 px → jitter sadeleşmiyor; "hafif
   bulanık" diye görülür, kimse bug demez.
6. **★ İşaret ve ölçek: saf dönüş.** Kamerayı yalnız HEDEFİ kaydırarak sağa
   küçük bir açıyla (θ ≈ 1°) döndür ve HEMEN ardından `sample='rt'`.
   Beklenen: `mean_dx` POZİTİF (içerik sola kayar, nokta önceden sağdaydı) ve
   ≈ W·θ/fov_x (1680 px, fov_y 35.3°, en-boy 1.778 → fov_x ≈ 58.3° → 1° ≈
   28.8 px), `mean_dy` ≈ 0. Gökyüzü de aynı kaymayı verir (dönüş derinlikten
   bağımsız). İşaret ters → sözleşme ters; ölçek %10'dan fazla sapıyor →
   izdüşüm ya da genişlik yanlış. Örnek bir sonraki dispatch'e kaldıysa
   mean_motion 0 çıkar — tekrarla.
7. **Ayrılma.** Aynı hareket sırasında `rejected_fraction` ≈ ekrana yeni giren
   kenar şeridi (≈ |mean_dx|/W) + kenarlar; kamera durunca bir dispatch içinde
   ~0'a döner. Hep ~1 → geçmiş slotları ters (okuma = yazma).
8. **Nesne hareketi.** Bir nesneyi IPC transform ile kaydır: render cihazında
   `object_motion_dispatches` artmalı; Debug View 16'da yalnız nesne renkli,
   arka plan gri. Artmıyor → TLAS önceki dizi yakalanmıyor (CPU yolu: aynı
   sayı + aynı yol şartı; GPU yolu: kopya).
9. **Debug görünümleri.** `render.set_settings @{ debug_view=16 }` duran kamerada
   düz gri; 17 duran kamerada koyu mavi (geçmiş kullanıldı), hacim pikselleri
   yeşilimsi. Sonra `debug_view=0`.
10. **RayFusion / raster.** Material/RayFusion moduna geç, `viewport.taa` açık;
    madde 6'daki dönüşü yap, `sample='raster'`: mean_dx RT ile aynı işaret ve
    ~aynı büyüklük. `raster_available` false ise `raster_reason`.
11. **Halton.** `render.set_motion @{ halton_jitter=$true }` → birikim baştan,
    `motion_stats` jitter 8 fazda değişen (|x|,|y| ≤ 0.5) değerler; madde 5
    yine ~0. Sonra kapat (varsayılan kapalı).
12. **Maliyet.** RT dispatch ms, `render.set_motion @{ enabled=$false }` ile
    karşılaştır (iç oda ~96–100 ms). Fark %2'den büyükse ölç ve not et.

## B1 — OIDN takvimi + hareket bounce profili + RR düğmesi (2026-10-26) — CANLI 8/9, 1 HATA

**Sonuçlar (iç oda, 1680×945, denoiser fast):**
- (1) shader ABI OK, pipeline ready. (2) RR=3 → 100 ms (taban 96, gürültü içi).
- (3) RR 2 → 82.9 ms (−%17), RR 1 → 75.3 ms (−%25); 0 reddedildi.
  Gürültü karşılaştırması YAPILMADI — varsayılan 3 kalıyor.
- (4)(6) Takvim, 128 örneğe kadar: every_sample 8.6 örnek/sn (124 çağrı),
  progressive 10.3/sn (5 çağrı, 119 önbellek), converged_only 10.5/sn
  (1 çağrı). Denoiser kapalıyla aynı hız bandı (+%20 every_sample'a göre).
- (5) 8 kamera hareketi → 18 çağrı: her harekette yeni çağrı var.
- (7) Hareket profili: hareket sırasında active=true, effective 2, authored 10
  (değişmedi), RT 34.7 ms; durunca active=false, effective 10, RT 98 ms.
- (8) Kapalıyken effective = authored. Geçersiz settle_ms reddedildi.
- (9) Proje kaydı denenmedi (kullanıcının projesine yazmamak için).

**★ HATA (kullanıcı fark etti: "converged'de son denoise karesinde kamera
hafif sağa dönmüş gibi"):** ölçüldü — kamera 0.25 birim kaydırılıp
converged_only ile yakınsatıldığında denoise karesi YENİ ham kareye göre
6 px kaymış ve ESKİ konumun ham karesine daha yakın (ortalama fark 10.2 /
16.9); every_sample doğru (fark 10.2, kayma 0). Kök neden:
`VulkanBackendAdapter::getDenoiserFrameGPU` "fence-deferred ping-pong" —
her çağrı BİR ÖNCEKİ çağrıda kopyalanan kareyi döndürür, yenisini arka
planda başlatır. Her örnekte çağrılınca gecikme 1 örnek (görünmez); takvim
çağrıları seyrekleştirince "önceki çağrı" önceki kamera konumu olabilir.
progressive'de etkisi: sıfırlamadan sonraki ilk denoise eski pozu gösterir,
bir sonraki çağrıda düzelir. converged_only'de bir sonraki birikime kadar
kalır. Düzeltme B2 partisinde (plan §4): kopyayı (epoch, örnek) ile etiketle,
eşleşmiyorsa senkron kopya + bekleme. O zamana kadar güvenli seçim
`every_sample`.

Ayrıntılı madde listesi (B1 derlemesi için yazılan):

Plan: RT_REALTIME_DENOISE_UPSCALE_PLAN.md §3. **Önce `compile_shaders` (raygen
değişti: push constant'a `rrStartBounce` eklendi, 196 → 200 bayt), sonra C++.**
Yeni/değişen IPC: `render.set_denoiser schedule`, `render.denoiser_stats`
(`skipped_by_schedule`, `cache_presents`), `render.set_settings rr_start_bounce`,
`render.get_settings` (`effective_*`, `motion_profile_active`),
`render.get_motion_profile` / `render.set_motion_profile`. Panel: Light Paths
altında "Roulette From Bounce" + "Fewer Bounces While Moving"; Denoiser altında
"Viewport Schedule". Üç ayar da projeye kaydediliyor. Audit OK.
Varsayılanlar: schedule = progressive (YENİ davranış), RR = 3 (bugünkü),
hareket profili KAPALI.

Ölçüm tabanı (iç oda, 1680×945): RT 95.7 ms/örnek, denoiser fast açıkken
örnek/sn ~9.8 (her örnek denoise), kapalıyken ~11.2.

1. **Shader ABI.** Rendered mod açılıyor, görüntü bugünküyle aynı, log'da
   push constant / pipeline hatası yok. Siyah ekran ya da pipeline hatası →
   raygen.spv eski (compile_shaders koşmadı) ya da C++ ile ABI ayrışık
   (static_assert 200).
2. **RR varsayılanı birebir.** `render.get_settings` → `rr_start_bounce` 3;
   RT dispatch bugünkü gibi ~96 ms. Farklıysa RR yolu varsayılanı değiştirdi.
3. **RR düğmesi çalışıyor.** `rr_start_bounce=2` → RT ms belirgin düşmeli,
   `samples` sıfırlanmalı (birikim baştan). Düşmüyorsa shader eski alanı okuyor.
   Gürültüyü ölç (komşu piksel farkı, 128 örnekte): 3'e göre ne kadar arttı —
   varsayılan bu sayıya bakılarak değişir, ölçmeden değil. Sonra 3'e geri al.
4. **Takvim: progressive.** Denoiser açık, kamera kıpırdat, birikim bitene kadar
   bekle: `calls` ≈ log2(128)+1 = 8 civarı (birden çok örnek aynı karede
   düşerse daha az), `skipped_by_schedule` ≈ 120, `cache_presents` =
   `skipped_by_schedule`. Örnek/sn ≥ 10.8 (bugün 9.8). Son kare
   `every_sample` ile aynı görünmeli (ikisi de 128'de denoise eder).
   ★ Sessiz arıza: ekran birikim boyunca ESKİ bir denoise karesinde donar ve
   "denoiser çok yumuşak" diye okunur — `cache_presents` artarken `calls` hiç
   artmıyorsa takvim hiç tetiklenmiyor.
5. **★ Hareket halinde eski kare YOK.** Denoiser açık, progressive, kamerayı
   birkaç kez peş peşe kıpırdat (`camera.set_position` adımları): her
   kıpırdatmada `calls` artmalı (yeni epoch → önbellek geçersiz). Artmıyor ve
   `cache_presents` artıyorsa ekran ÖNCEKİ kamera konumunun görüntüsünü
   gösteriyor — en kötü hata, kimse fark etmeden "gecikme" sanır.
6. **converged_only / every_sample.** `converged_only`: birikim boyunca
   `calls` 0, `cache_presents` 0 (ham görüntü), tamamlanınca 1 çağrı.
   `every_sample`: eski davranış, `skipped_by_schedule` 0.
7. **Hareket profili.** `render.set_motion_profile @{ enabled=$true }`
   (2/2/2, 150 ms). Kamerayı ~1 sn boyunca sık kıpırdat ve bu sırada
   `render.get_settings`: `motion_profile_active` true, `effective_max_bounces` 2,
   `max_bounces` 10 (yazılan değer DEĞİŞMEMELİ). RT dispatch ~31 ms.
   Durduktan >150 ms sonra: active false, effective 10, birikim baştan,
   RT ~96 ms. Active hiç true olmuyorsa poz tespiti `scene.camera`'yı değil
   başka kamerayı izliyor.
8. **Profil kapalıyken etkisiz.** enabled=false: kamera hareketinde
   effective = authored, RT ~96 ms. Sonra profili kapalı bırak.
9. **Proje kaydı.** Ayarları değiştir, projeyi kaydet/aç: schedule, RR ve
   profil geri gelmeli; `motion_profile_active` açılışta false.

## Birikim bitmiş kareye bounce ve denoiser düzenlemesi ulaşsın (2026-10-26 parti 2) — CANLI PASS

Sonuç: (1) bounce 9 → 1 sn'de samples 10, aktif; (3) denoiser aç → 1 çağrı, converged 1, samples 128'de kaldı; (5) boşta 5 sn 0 çağrı; (4) komşu piksel farkı kapalı/açık/kapalı 0.065 / 0.012 / 0.065 — kapatınca ham kareye dönüyor. (2) panel sürükleme elle denenmedi.

İkisi de canlı ölçüldü (aşağıda), ikisi de "çekirdek değişti, ekran değişmedi"
sınıfı. Render bloğu `start_render` kapısının arkasında; birikim bitince kimse
kapıyı açmıyordu, açılsa da backend yeni kare basmıyordu.
Değişenler: `rtapi::setRenderBudget` (her bounce yazımı `start_render`),
scene_ui iki "Light Paths" bloğu (DragInt değişince `start_render`),
`IBackend::requestPresentRefresh()` (Vulkan: `m_tonemapRefreshPending`),
Main.cpp'de denoiser imzası tespiti (değişince `start_render` + yenileme).

1. **Bounce, IPC.** Birikim bitmiş (`viewport.status` complete=true), sonra
   `render.set_settings @{ max_bounces=9 }`: 1 sn içinde `samples` küçülmeli,
   `rendering_active` true. Öncesi: 3 sn boyunca 128'de kaldı. Geri 10 yap.
2. **Bounce, panel.** Aynı şey Total Bounces sürükleyerek. Render yeniden
   başlamıyorsa iki panel bloğundan biri atlanmış.
3. **Denoiser AÇ, bitmiş kare.** Denoiser kapalı, birikim bitmiş,
   `render.reset_denoiser_stats`, `render.set_denoiser @{ viewport_enabled=$true }`,
   2 sn: `calls >= 1`, `converged_calls >= 1`, ekran temiz. Öncesi: 0 çağrı.
   Birikim YENİDEN BAŞLAMAMALI (`samples` 128'de kalır) — başlıyorsa yenileme
   yolu yanlışlıkla reset tetikliyor.
4. **★ Denoiser KAPAT, bitmiş kare.** `viewport_enabled=$false`: ekran
   gürültülü ham kareye dönmeli. Sessiz arıza hâli: ekran temiz KALIR ve
   "kapattım, hiçbir şey değişmedi" diye okunur — `requestPresentRefresh`
   g_backend'e ulaşmıyor demektir (g_backend Vulkan adapter değil mi bak).
   `viewport.get_screenshot` ile iki kare karşılaştırılabilir (scratchpad'e).
5. **Boşta maliyet yok.** Birikim bittikten sonra hiçbir şeye dokunmadan 5 sn:
   `denoiser_stats.calls` 0 kalmalı. Artıyorsa imza her karede değişiyor.

## Denoiser IPC'ye açıldı + maliyet sayacı (2026-10-26 parti 1) — CANLI PASS

Yeni: `render.get_denoiser`, `render.set_denoiser`, `render.denoiser_stats`,
`render.reset_denoiser_stats` (+ `rt.render.*` Python), perf `loop.denoise`,
dosyalar `Api/RtApiDenoiser.h/.cpp`. Canlı sonuç (iç oda, 1680×945, RTX 2503):

| Durum | RT dispatch ms/örnek | denoise ms/çağrı | örnek/sn |
|---|---|---|---|
| Denoiser kapalı | 95.7 | — | ~11.2 |
| Açık, fast | 110.7 | 43.0 | ~9.8 |
| Açık, balanced | 119.9 | 63.3 | ~9.0 |

- Yol `gpu` (CUDA sıfır kopya). Denoise RT ile ÖRTÜŞÜYOR: 43 ms'lik çağrı
  örnek hızını yalnız ~%13 düşürüyor, ama aynı GPU'yu paylaştığı için RT
  dispatch'ini +15 ms (fast) / +24 ms (balanced) şişiriyor.
- `repeat_sample_calls` = 0, birikim bitince `calls` = 0: kod okumasından
  çıkan "OIDN boşta her turda koşuyor" şüphesi YANLIŞ çıktı.
- Geçersiz `quality` reddedildi, hiçbir alan değişmedi.

## Gaz march'ı: büyüyen adım söküldü (2026-10-25, derlenmedi; SHADER + C++)

RT_GAS_SECONDARY_MARCH.md §14. Radyans önbelleği canlı doğrulandı (−%53,
geri düşme 0); büyüyen adım örnek azaltmadı ve başlığı kararttı, söküldü.
Değişen shader: volume_closesthit. C++: VulkanBackend(_Volumes), RtApi, IPC,
Python, panel, descriptor. Kaynak: `check_volume_march_lod_contracts.py` → OK.

1. **Açılış değeri.** `render.volume_march_lod` beş alan döner:
   1/4/2/128 ve `secondary_radiance_cache` true. `*_transmittance_growth` yok.
   Panelde büyüme kaydırıcıları yok. Yedi alan dönüyorsa eski exe.
2. **Referans birebir.** 1/1/1/0, önbellek kapalı: önceki referansla aynı
   (nükleer sahne, düzlem açık ~500 ms). Farklıysa geri alma eksik
   (`grep growStride shaders` boş olmalı).
3. **Önbellek hâlâ çalışıyor.** Varsayılanda `secondary_density_samples`
   ≈ 0, `radiance_cache_fallbacks` ≈ 0; başlık referansla aynı tonda (büyüme
   onu −1.8/255 kararttıyordu).

## Gaz: sekme ışınları için radyans önbelleği + geçirgenliğe göre büyüyen adım (2026-10-25, derlenmedi; SHADER + C++)

RT_GAS_SECONDARY_MARCH.md §13. Yeni shader include:
`volume_radiance_cache.glsl` (vcxproj'a eklendi). Değişen shader'lar:
volume_closesthit, volume_light_cache.glsl (kanal 6 → 10),
volume_instrumentation.glsl (+2 sayaç, struct 248 → 256 B). C++: ışık önbelleği
kanal sayısı, VulkanBackend(_Volumes), RtApi, IPC, Python, panel, descriptor.
RT pipeline baştan derlenir. Kaynak kontrolleri:
`check_volume_march_lod_contracts.py` ve `check_volume_light_cache_contracts.py`
→ OK.

1. **Ayar ve ABI.** `render.volume_march_lod` yedi alan döner:
   1/4/2/128, `secondary_radiance_cache` true, büyüme 3/3. `render.volume_stats`
   içinde `radiance_cache_marches` / `radiance_cache_fallbacks` var. Sayaçlar
   saçmaysa shader ile C++ struct'ı hizasız. `render.volume_light_cache`
   `channels: 10` döner.
2. **Referans birebir.** 1/1/1/0, önbellek kapalı, büyüme 1/1: maliyet ve
   görüntü önceki referansla aynı. Farklıysa büyüme ya da önbellek kapalıyken
   de çalışıyor.
3. **Önbellek doluyor.** Varsayılanda, statik karede birkaç örnekten sonra
   `radiance_cache_fallbacks`, `radiance_cache_marches`'ın yanında küçük
   kalmalı; `secondary_density_samples` neredeyse sıfıra inmeli. Geri düşmeler
   yüksek kalıyorsa radyans etiketi her karede değişiyor (ışık durumu hash'i
   kararsız) ya da sahiplik bırakılmıyor.
4. **Işık önbelleği bozulmadı.** `light_cache_hits` / `light_cache_misses`
   eskisi gibi. Sahnede 5'ten fazla ışık varken 6.+ ışığın gölgesi march'tan
   gelmeli. ★ Kanal eşlemesi toplam kanal sayısını izlerse ışık 5–8 radyans
   kanallarına yazar: yanlış gölge, hata yok.
5. **Görüntü.** Nükleer sahne, düzlem açık; referans ile varsayılan aynı
   oturumda, aynı karede:
   - zemindeki dolaylı ateş ışığı;
   - dumanın gölgeli tarafı;
   - yoğun başlığın arkasındaki kenarlar.
   Hafif yumuşama beklenir; zeminin belirgin kararması ya da açılması
   beklenmez.
6. ★ **Sinsi olan: donmuş önbellek.** Işık ya da güneş hareket ettirildiğinde
   sekme ışınlarının gördüğü gaz eski aydınlatmada kalırsa, ışık durumu
   hash'i değişmeyi kaçırıyor demektir. Görüntü inandırıcı görünür ama yanlış.
   Kontrol: güneşi döndür; zemindeki dolaylı ışık birkaç örnek içinde yeni
   yöne uymalı.

## Gaz march'ı: uyarlanan adım söküldü, varsayılan 1/4/2/128 (2026-10-25, derlenmedi; SHADER + C++)

RT_GAS_SECONDARY_MARCH.md §12. Önceki derlemedeki uyarlanan adım canlı
ölçüldü: daha çok örnek aldı ve görüntüyü açtı, söküldü. Değişen shader:
volume_closesthit. C++: VulkanBackend(_Volumes), RtApi, IPC, Python, panel,
descriptor. Kaynak: `python scripts/test/check_volume_march_lod_contracts.py` → OK.

1. **Açılış değeri.** `render.volume_march_lod` dört alan döner:
   1/4/2/128, `adaptive_*` alanı yok. Panelde uyarlanan adım kaydırıcıları
   yok. Yedi alan dönüyorsa eski exe.
2. **Referans birebir.** 1/1/1/0'da maliyet ve görüntü uyarlanan adımdan
   önceki referansla aynı. Fark varsa geri alma eksik
   (`grep segStride shaders` boş olmalı).
3. **Varsayılan.** Nükleer sahne, düzlem açık: referansa göre ~−%30, zemin
   referansla aynı ton (ortalama fark ≈ +1/255). Zemin belirgin açıksa sınır
   hâlâ 32.

## Obje görünürlüğü çekirdekte: `scene.set_visible` / `scene.get_visible` (2026-10-25, derlenmedi)

Outliner gözü artık `rtapi::setObjectVisible` çağırıyor (yeni dosya
`RtApiObjectVisibility.cpp`, vcxproj'a eklendi). IPC `scene.set_visible`
(SceneWrite) + `scene.get_visible` (Read, backend maske okuması dahil),
`scene.object_info.visible`, Python `rt.scene.set_visible/get_visible`.
Test: `python scripts/test/rt_object_visibility_ipc.py --object Plane_1`.

1. **Derleme + `agent.discover`**: `scene.set_visible` ve `scene.get_visible`
   listede, ikisi de documented. Yoksa eski exe ya da üretilmiş tablo bayat
   (`python scripts/audit_ipc_capabilities.py` yeşil olmalı).
2. **Bilinmeyen ad reddedilir**: `scene.set_visible {name:"yok", visible:false}`
   hata döner. Başarı dönüyorsa `forEachEntry` boş eşleşmeyi saymıyor —
   script'ler yanlış adla "gizledim" sanır.
3. **Panel gözü hâlâ çalışıyor** (tek obje, grup, Characters/Models): göz
   kapanınca obje viewport'tan kalkar, açınca geri gelir. Çalışmıyorsa UI →
   rtapi yönlendirmesi kırık (`setHierarchyObjectVisibility` artık yalnızca
   `rtapi::setObjectVisible` çağırıyor; `rtapi::g_ctx` bağlı mı?).
   Characters/Models gözü eskiden yalnız `member->visible` çeviriyordu ve
   backend maskelerine hiç ulaşmıyordu — artık ulaşmalı.
4. **Test script'i PASS** (Plane_1, Vulkan RT açık): `hidden:` satırlarında
   `viewport_raster`, `viewport_rt`, `render_rt` maskeleri 0/N, `pick_gpu`
   objeyi döndürmez; `restored:` satırlarında N/N. `SKIP` = o backend'de
   read-back yok (-1) ya da obje hiç kurulmamış (0 instance) — PASS değil.
5. **★ EN SİNSİ: viewport'ta gizli, RT render'da görünür (ya da tersi).**
   Sessizce makul görünür çünkü baktığın pencere doğru davranır. Ölç:
   `scene.get_visible` → `viewport_raster_visible` ile `render_rt_visible`
   gizlenmiş objede İKİSİ DE 0 olmalı. Gözle: objeyi gizle, final render al
   (`render.start`), objenin gölgesi/yansıması da kalkmış olmalı. Biri 0
   diğeri değilse iki rol (`g_viewport_backend` / `g_backend`) tek backend
   sanılıp ikincisine `setVisibilityByNodeName` gitmemiştir
   (`render_is_viewport` alanına bak).
6. **★★ Alan raporu (2026-10-25, Nuclear Detonation / Plane_1): "göster"
   ilk seferde RT'ye ulaşmıyordu.** Kök: Vulkan `updateGeometry`, raster
   rebuild ve OptiX snapshot GİZLİ objeyi tamamen ATLAR — gizliyken herhangi
   bir rebuild olduysa objenin instance'ı yoktur ve `setVisibilityByNodeName
   (true)` maskeleyecek bir şey bulamayıp sessizce hiçbir şey yapmaz; obje
   alakasız bir sonraki rebuild'e kadar yoktur. Düzeltme: `setObjectVisible`
   gösterirken backend read-back'te ad için 0 instance (backend'de başka
   instance varken) görürse rebuild planlar; OptiX'te read-back yok, her
   göstermede OptiX rebuild planlanır. Ölç: test script'inin `rebuilt:` ve
   `restored:` fazları (`--rt-counters` ile RT sonucu da: yüzey sekmesi/yol
   gizlide değişmeli, restore'da baseline'a dönmeli). `restored: ... 0
   instances, baseline had N` FAIL'i tam bu hatadır. Log'da
   `[Visibility] ... geometry rebuild scheduled` görülmeli.
   Aynı testte gizliyken outliner gözü kapalı kalmalı: `rebuildMeshCache`
   rep facade'a `tmesh->visible` kopyalıyor; göz açık görünüyorsa o satır
   çalışmıyor. **H tuşu (hepsini göster) aynı hataya açık** — `showAllInstances`
   da düşmüş instance'ı geri getiremez; bu partide dokunulmadı.
7. **Pending-delete**: silinmiş (undo bekleyen) objeye `visible:true` → hata,
   obje gizli kalır.
8. **Bilinen sınır**: `scene.raycast` görünürlüğe bakmaz (snap yüzeyi
   sorgusu, eskisi gibi); gizli objeye hâlâ isabet etmesi bug değil. Proje
   kaydında görünürlük yazılıyor mu bu partide değişmedi.

## Kamera ışık adımı varsayılanı 1 (2026-10-25, derlenmedi; yalnız C++ varsayılanları)

1. **Açılışta `render.volume_march_lod` 1/4/4/32 döner**, panelde Camera Light
   Stride 1. 2 dönüyorsa eski exe. Kutunun gölgeli alt yüzünde yatay bant
   yok. Bant varsa kamera adımı hâlâ 2, ya da başka bir şey bant çiziyor
   (ayar 1/1/1/0 ile kontrol et).

## Gaz march LOD'u: ışık adımı + ikincil adım (2026-10-25, CANLI: splat+gaz −%47, tam sahne −%39; kamera varsayılanı 1 derleme bekliyor)

RT_GAS_SECONDARY_MARCH.md §8. Değişen shader'lar: volume_closesthit (march),
closesthit, hair_closesthit, raygen, rt_sky.glsl (yalnız dünya aynası
yeniden adlandırıldı). C++: vulkan_world_data.h, VulkanBackend(_Volumes),
RtApi, IPC, Python, yetki, panel, descriptor. RT pipeline baştan derlenir.
Kaynak: `python scripts/test/check_volume_march_lod_contracts.py` → OK.

1. **Ayar okunuyor.** `render.volume_march_lod` 2/4/4/32 döner; panelde dört
   kaydırıcı aynı değerleri gösterir. `render.set_volume_march_lod
   camera_light_stride=17` reddedilir. Okunmuyorsa ya da 0'lar dönüyorsa
   IPC/yetki zinciri eksik.
2. **Referans birebir.** 1/1/1/0'da görüntü ve maliyet bu partiden önceki
   gibi: splat_gas ~286, gas_only ~93 ms. Farklıysa kelime yanlış çözülüyor
   (0 = eski march olmalı) ya da `volumeShadowDue` k = 1'de eski koşul değil.
3. **Kazanç.** `python scripts/test/rt_volume_march_lod_ab_ipc.py`: splat_gas
   belirgin düşer (hedef ~150–180 ms); gas_only bir miktar düşer; splat_only
   aynı kalır. splat_gas'ta `secondary_density_samples` ~1/3'e iner,
   `camera_density_samples` değişmez.
4. **Görüntü.** Referans ile varsayılanı aynı kamerada karşılaştır:
   - doğrudan görünen gazın yumuşak gölgeleri ve ışık konisi kenarı;
   - gaz içindeki splat'ların dolaylı aydınlanması.
   Kamera tarafında hafif yumuşama beklenir; bant ya da basamak beklenmez.
5. ★ **Sinsi olan: donmuş gölge.** Gazın içinde, ışığa göre gölgede kalan
   bölge 2/4/4/32'de referanstan belirgin AYDINLIK ya da bütün ışın boyunca
   tek tonda görünüyorsa, önbelleklenen gölge ilk değerinde kalmış demektir
   (`volumeShadowDue` yenilemiyor). Bu "LOD biraz yumuşattı" gibi okunur ama
   yanlış ışıktır. Kontrol: `camera_light_stride` 1 ile 4 arasında değiştir;
   gölgeli bölge yalnız yumuşamalı, kaybolmamalı.
6. **Panel.** Kaydırıcı bırakılınca uygulanır ve birikim sıfırlanır; sürükleme
   sırasında her karede yeniden yükleme olmamalı.

## Gaz march'ı: kamera / ikincil sayaç ayrımı (2026-10-25, CANLI: toplam tam, ikincil pay %45.5 → ölçüt kaldı)

RT_GAS_SECONDARY_MARCH.md §4 adım 0. Yalnız ölçü aleti; görüntü ve maliyet
değişmemeli. Değişen shader'lar: volume_instrumentation.glsl,
volume_closesthit, closesthit (küre varyantı dahil).
C++: VulkanVolumeInstrumentation.h (224 → 248 B), RtApi, IPC, Python, panel,
descriptor.

1. **ABI tutuyor.** `render.volume_counters enabled=true`, sonra
   `render.volume_stats`. Eski alanlar önceki ölçümle aynı mertebede:
   splat + gazda yol başına `density_samples` ~33. Hepsi sıfır ya da saçma
   büyük sayılarsa shader ile C++ struct'ı hizasız; ikisi birlikte derlenmedi.
2. **Sınıflar toplamı kapatıyor.** `camera_density_samples +
   secondary_density_samples == density_samples`; aynısı rays ve shadow için.
   Eksik kalıyorsa sınıflanmamış örnek var: kayıt çağıran bir shader main()'de
   `volumeInstrumentationSetSegment` çağırmıyor.
3. **Yalnız gazda kamera ağır basıyor.** gas_only'de kamera payı büyük olmalı.
   Kamera bitinin anlamı doğru mu, buna bakılır.
4. ★ **Sinsi olan: ikincil sıfır.** splat_gas'ta `secondary_density_samples`
   0 ise toplam yine tutar (madde 2 geçer), ama kamera biti sekmelerde
   temizlenmiyor demektir. Bu durumda "ikincil pay yok, LOD işe yaramaz" diye
   yanlış karar çıkar. Splat'lardan seken ışınlar gaza giriyor: ikincil sayı
   sıfır olamaz.
5. **Karar ölçütü.** splat_gas'ta ikincil örnek payı > %60 ise LOD (adım 1)
   doğru hedefte.

## Gaz içindeki splat'lar: iç içe katı izi GERİ ALINDI (2026-10-25, CANLI DOĞRULANDI: 27 / 97 / 295 ms)

RT_GAS_OPTICAL_PERFORMANCE.md §"Seçenek A ölçüldü: gerileme, geri alındı".
Geri dönen shader'lar: volume_closesthit, closesthit (küre varyantı dahil),
hair_closesthit, miss, rt_payload.glsl. C++: özyineleme derinliği yeniden 2.
RT pipeline baştan derlenir.

1. **Maliyet eski yerine döndü.** Aynı sahne, kare 28, `fluid.set_phase_output`:
   gas_only ~94 ms, splat_gas ~311 ms, splat_only ~25 ms. A'lı derlemede
   265 / 406 / 25 ms ölçüldü. gas_only hâlâ ~265 ise sebep A değil; aynı ağaçtaki
   başka bir değişiklik (paralel ajan) ya da eski .spv. Önce .spv zaman damgasına bak.
2. **Görüntü.** Gaz içindeki splat'lar görünür, sonraki sekmenin gökyüzü yerinde.
   Bu yol A'dan önce canlı doğrulanmıştı; bozuksa geri alma eksik kalmış demektir
   (`grep PL_NESTED_SOLID shaders` boş olmalı).

## Hacim kutuları dolu hücrelere daralır: sıvı SurfaceSDF + host gaz (2026-10-25, derlenmedi; yalnız C++)

Matter Fog akışı daraltması önceki build'de canlı doğrulandı (kutular dar geldi).
Bu parti aynısını sıvı SurfaceSDF / host gaz yayınına ekler.
RT_GAS_OPTICAL_PERFORMANCE.md §"Devam: ilk sahne". Ölçüm: aynı sahne, kare 28.

1. **SDF kutusu daraldı.** Katı splat, gaz hidden, sıvı sdf iken
   `render.volume_slots`: "[Fluid NanoVDB]" slotu artık [0,5]³ değil, su ve
   köpüğün etrafında küçük bir kutu. Daralmadıysa: yükleme bu karede olmadı ve
   kutu hiç hesaplanmadı (bir sim adımı ilerlet), ya da canlı GPU yolu.
2. **Maliyet.** Aynı rota (splat / sdf / hidden) 135 ms'den ~26 ms'ye (boş
   sahne) yaklaşır; tam inmez, çünkü suyun kendisi gerçek bir yüzey.
3. **Su görünümü aynı.** Kenarda kesilmiş su yüzeyi ya da köpük = kutu fazla dar.
4. ★ **Sinsi olan:** sahne kaydedilip yeniden açılınca, ya da sim geri sarılınca
   su kutusu yeniden hesaplanmadan eski dar kutu kalırsa, yeni su o kutunun
   dışında GÖRÜNMEZ ve hiçbir hata vermez. Geri sar → oynat → suyun yeni
   konumunda göründüğünü kontrol et.
5. **Gaz (host yolu).** Host'tan yüklenen gaz domain'lerinde de kutu daralır;
   canlı GPU gazında değişmez.

## Hacim ışık önbelleği — A/B anahtarı söküldü (2026-10-25, derlenmedi; yalnız C++)

V1 canlı kabul edildi (docs/dev/VOLUME_LIGHT_CACHE.md §11–12). Bu parti yalnız
`enabled`'ı söker; shader değişmedi, RT pipeline yeniden derlenmez.
Kaynak kontrolü: `python scripts/test/check_volume_light_cache_contracts.py` → OK.

1. **Derleme temiz.** `VolumeLightCacheConfig` hiçbir yerde kalmamış olmalı;
   kalmışsa bağlayıcı/derleyici hatası verir.
2. **Panel.** Performance'ta onay kutusu yok, "Light Cache Cells" var; değiştirince
   `render.volume_light_cache.max_cells` aynı değeri gösterir.
3. **IPC.** `render.set_volume_light_cache {max_cells: 48}` çalışır,
   `{enabled: false}` "max_cells" eksik diye reddedilir. `render.volume_light_cache`
   çıktısında `enabled` alanı yok.
4. ★ **Sinsi olan:** önbellek gerçekten açık mı? Counters açıkken duran karede
   `light_cache_hits` > 0 ve misses ≈ 0. Anahtar sökülürken `wanted` koşulu
   yanlış kalsaydı hiçbir hata olmaz, yalnızca eski hıza dönülürdü.

## Islak kum 4. parti — tane hash kovası 16 → 32 (2026-10-23, derlenmedi; C++ + shader)

Canlı (teğet seed build'i): 50 000 ıslak Sand tanesi (ıslaklık 1.0) düşerken
"grain neighbour bucket overflow (more than 16 grains hashed to one bucket)"
ile tutuldu — havada, küp collider'ın 35 cm üstünde (çarpma değil; sertlik ×4
de aynı). Sebep: ıslak hash hücresi 3r, yoğun seed kafes adımı 1.474r → kötü
hizada hücrede ~14 merkez (meşru); 16'da tek hash çakışması taşırıyor.
Değişiklik: `kMatterGrainBucketCapacity` 32, shader `BUCKET` 32, revizyon 31
(32-bit indeks 2^31'e sığar; 50k tanede kova belleği ~18 → ~35 MB).

1. **Shader derlendi:** `sim_matter_grain` revizyon 31; eski .spv ile adım
   "shader revision mismatch" der.
2. **Aynı sahne:** 50k ıslak Sand bloğu düşüp yere çarpar, tutulma yok.
3. ★ **Sinsi olan:** "neighbour list full" (bit 8) — 32'lik Verlet listesi
   yoğun ıslak paketlemede dolabilir (3r içinde FCC ~18 komşu, iç içe
   geçmede daha fazla); retry var ama taşarsa yine tutar.

## Islak kum 3. parti — teğet seed (2026-10-23, derlenmedi; yalnız C++)

FCC aralığı 2.2002R → 2.0851R (jitter 0.07R → 0.03R): boşluk 0.2R → 0.085R,
doluluk 0.556 → 0.653. Not: ISLAK_KUM_DOGRULAMA.md 3. parti + DEVİR bölümü.

1. **C++ testi** `matter_grain_birth_test.cpp`: 172 tane, ≥ 2R ayrılık geçmeli.
2. **Seed sayısı:** 0.3×0.5×0.3 m Sand, r 12.5 mm → ~3300 (önce 2813).
3. **İlk kare sakin:** seed sonrası 5 adımda ortalama hız 0.52 m/s değil, küçük
   (blok bulut gibi düşmüyor); `fluid.get settle_converged` true olmalı.
   ★ **En sinsi sonuç:** kuru blok ayakta kalıyor — FCC kilitlenmesi; gerçek
   kuru kum çöker. Öyleyse jitter'ı ölçerek büyüt (d formülüyle birlikte).
4. **Islak kısa kolon** (0.2 m, ıslaklık 0.3, Sand r 0.25 mm): önceki 0.029'dan
   yüksek olmalı; hâlâ çöküyorsa yapışma ölçeği (DEVİR maddesi) sırada.

## Islak kum 2. parti — FCC seed + köşeli kum (2026-10-23, derlenmedi; yalnız C++)

Canlı (18:09): seed ıslaklığı çalışıyor (wet_grains, 8834 köprü), ama seed'li
blok kuru da ıslak da tek tane kalınlığına yayıldı. Kökler (ISLAK_KUM notu):
Sand μ_r 0.02 (kuru döküm ~8°; 0.2 ile 30°, ıslak 47°) ve basit kübik seed
(0.39, kayma direnci yok). Değişiklik: seed FCC (0.556), Sand μ_r 0.2 + su
kapasitesi 0.1, Gravel μ_r 0.25; `matter_grain_birth_test.cpp` güncellendi.

Sıra:

1. **C++ testi:** `scripts/test/matter_grain_birth_test.cpp` geçmeli (172 tane,
   ≥2R ayrılık, voxel bağımsızlığı, bütçe öneki).
2. **Seed yoğunluğu:** Sand seed 0.3×0.5×0.3 m, r 12.5 mm → ~2600–2900 tane
   (önce 1800). *Bozuksa:* hâlâ 1800 = eski exe.
3. **Kolon:** `.\scripts\ipc\Probe-WetSandBlock.ps1 -Case column -Wetness 0,0.3`
   — artık türetme gerekmez ama probe WetSandProbe kullanır (Sand + kapasite
   0.1; μ_r Sand'den 0.2 gelir). Kuru: yığına çöker (merkez ~0.06–0.08);
   ıslak 0.3: belirgin yüksek (ayakta ya da dik). ★ **En sinsi sonuç:** kuru
   kolon da ayakta kalır → FCC kristal kilitlenmesi (gerçek kumda yok); o zaman
   jitter'ı ölçerek büyütmek gerekir.
4. **Kaya:** `-Case rock` — ıslak blok topak halinde çatlamalı.
5. **Regresyon:** kuru Sand dökümü artık ~30° (önce düz yayılıyordu); Sand +
   Water sahnelerinde taneler ıslanır (köprü) — tasarım gereği.

## Islak kum 1. parti — seed'e başlangıç ıslaklığı (2026-10-23, derlenmedi; yalnız C++)

★ Önce: kullanıcı "derledim" dedi ama `x64/Release/RayTrophiStudio.exe` ve
`RayTrophiStudio.log` 16:10 — kar 6. parti (tek Snow, substep paneli) bu
exe'de YOK (madde listesinde Packed Snow duruyor). Bu build ikisini birden getirir.

Bulgular (canlı, 16:10 exe):
- Sand = DEM (`granular_transport dem`), `grain_water_capacity_fraction = 0` →
  Sand hiç ıslanamaz; ıslak kum için türetip kapasite vermek gerekiyor.
- Islak tane yalnız akış kaynağından (`grain_birth_saturation`) doğuyordu;
  `fluid.seed` hep KURU → ıslak, sıkıştırılmış blok yazılamıyordu.
- Kalıp testi: 3600 ıslak tane döküldü (wet_grains true, merkez 0.277 m), ama
  kalıp collider'larını kapatmak simülasyonu 0. kareye sardı → 0 parçacık.
  Sim sürerken collider/kaynak düzenlemek = geri sarma; test tek kurulum olmalı.
- Köprü kuvveti ölçeklemesi doğru: F = 2πγcosθ·R·(R/r) (2026-10-21).

Değişiklik: domain seed tarifine `fluid_seed_grain_saturation` (0..1 kapasite);
seed DEM tanelerine `initMatterGrainBirthWater` ile su verir; ownership bunu
ıslak doğum sayar. Kayıt (proje + sahne), imza hash'i, `fluid.seed
grain_saturation`, `rt.fluid.seed(grain_saturation=)`, `fluid.get
seed_grain_saturation`, panelde "Seed Grain Wetness" (yalnız tane domain'inde).

Sıra:

1. **Exe yeni mi?** `RayTrophiStudio.exe` tarihi > 16:10, madde listesinde
   Packed Snow YOK. Değilse aşağıdakilerin hiçbiri koşamaz.
2. **Kolon testi:** `.\scripts\ipc\Probe-WetSandBlock.ps1 -Case column`
   (kuru 0 ve ıslak 0.3). Kuru: merkez 0.25'ten belirgin düşer (yayılır);
   ıslak: ~0.25'te kalır (ayakta). `wet_grains` ıslakta True, kuruda False.
   ★ **En sinsi sonuç:** ikisi de aynı düşüyor ve wet_grains True — seed suyu
   veriliyor ama köprü kuvveti tutmuyor (yapışma bu ölçekte zayıf); ikisi de
   ayakta → kuru kum da yapışıyor (sürtünme/uyku tutuyor), o da yanlış.
3. **Kaya testi:** `-Case rock`: ıslak blok kayanın üstünde topak halinde
   kalmalı/çatlamalı (merkez ~0.45), kuru blok kayanın etrafına akmalı.
4. **Panel:** tane domain'inde seed bölümünde "Seed Grain Wetness" kaydırıcısı;
   değiştirip bırakınca yeniden seed eder; proje kaydedip açınca değer kalır.

## Kar 6. parti — tek Snow, substep yalanı (2026-10-23, derlenmedi; yalnız C++)

Kullanıcı: "iki kar ön ayarı kafa karıştırır" ve "UI alt adımı 12 gösterse de
200 çıkıyor". Ölçüm: panelde görünen tek sayı **Max Advection Substeps = 12**
(`APICSolverParams::max_substeps`) — Matter granular yolunda hiçbir şeyi
sınırlamıyor (her ortak alt adımın içindeki taşımayı sınırlar, orada zaten 1).
Gerçek sayıyı elastik dalga CFL'si veriyor: 1/24 s ÷ (0.35 × 0.025 m ÷
√(1.4e5·5 / 400) = 41.8 m/s) = 200. Panel yalnız "adaptive" diyordu.

Değişiklik: `Packed Snow` kaldırıldı (kimse türetmemişti; ad hash etiketi ⇒
kayıtlı projeye etkisi yok). `Granular::planGranularSubsteps` (planlayıcıyla
aynı aritmetik) → panel "Granular Skeleton" başlığında: dinlenmede substep,
son kare, formül (kare dt / (0.35·voxel / dalga hızı)), dalga hızının
bileşenleri. Advection kaydırıcısının altında granular için açıklama. IPC/Python:
`granular_predicted_substeps`, `granular_wave_speed_m_s`,
`granular_sizing_young_modulus`, `granular_frame_dt`.

Sıra:

1. **Madde listesi:** yalnız `Snow`; `Packed Snow` yok.
2. **Panel = koşu.** Snow domain'i (voxel 0.025, 24 fps): "Solver substeps per
   frame: 200 at rest | last frame: 200". `fluid.get granular_predicted_substeps`
   200. *Bozuksa:* tahmin ≠ son kare (dinlenmede) ⇒ planlayıcı ile UI farklı
   formül kullanıyor (ikisi de `planGranularSubsteps`/`elasticStepInfo` olmalı).
3. **Advection kaydırıcısı** altında "Granular material: the solver substep
   count is set by its stiffness…" satırı (yalnız granular domain'de).
4. ★ **Sinsi olan:** çarpma anında "last frame" tahminden BÜYÜK olabilir
   (gerinme hızı terimi) — bu doğru; dinlenmede eşit olmalı.

## Kar 5. parti — sertleşme tabanı (2026-10-23, derlenmedi)

Build: **C++ VE `compile_shaders.bat`** (stress_update; push constant boyutu aynı,
son pad → `snow_min_hardening`). Yeni madde alanı `granular_snow_min_hardening`
(Snow 0.1). `detached` artık "tabana inmiş" demek (eskiden %99 kayıp).

4. parti canlı (15:43 exe): topak yapısı atandı (J_P 0.952..1.105), kaya sonucu
DEĞİŞMEDİ. Hız taraması 2.3 / 3.3 / 4.6 m/s → hepsinde ~%70 yumuşama, %26 toz
(enerjiden bağımsız). Kayaya ÇARPMADAN konan blok 1.5 s'de dağıldı (varsayılan
E); E 7e5 ile aynı blok sağlam durdu. Görsel (ekran görüntüsü, silindi): kaya
bloğu DELİP içine girmiş, blok kayayı sarıp zemine inmiş, dışarıdan sağlam küp
gibi; sayaçlardaki "toz" kayanın çevresindeki kesme bölgesiydi. Kütle merkezi
y 0.305 (kaya üstünde dururken beklenen ~0.39).

Kök: exp(ξ(1−J_P)) bütün modülleri (sıkışma dahil) çarpıyor ve tabanı yoktu;
gevşeyen kar basınçsız toz oldu, engelin etrafından aktı.

Sıra:

1. **Shader derlendi** (`.spv` > `.glsl` tarihi).
2. **Delip geçme bitti.** `Probe-SnowImpact.ps1 -Target rock`: `centroid y`
   0.305'ten belirgin yüksek (0.35+). `powder` sütunu artık tabandaki
   parçacıklar; çatlak yüzeyleri kadar olmalı, bloğun yarısı değil.
   ★ **En sinsi sonuç:** merkez yükselir ama blok tek parça kayanın üstünde
   asılı kalır (çatlamaz). Görsele bak: kenarlar aşağı kopmalı.
3. **Durgun kaya testi** (varsayılan E): kayanın üstüne konan blok dağılmamalı
   ya da yalnız sarkan kenarları kopmalı.
4. **Zemin regresyonu:** `-Target floor` önceki gibi (ortalama J_P ~0.68).
5. **Packed Snow (karar verildi: Snow toz kar kalır).** Yeni yerleşik madde
   `Packed Snow`: aynı yasa, ρ 550, E 1e6, iletkenlik 0.45; katalogda Snow'un
   yanında, görünümü Snow ile aynı. Test: `fluid.set_param SnowTest
   default_substance="Packed Snow"` → `Probe-SnowImpact.ps1 -Target rock`:
   kaya üstünde konan blok sağlam durmalı (E 7e5 ölçümü gibi), 4.6 m/s'de
   kenarlar kopmalı, `centroid y` ~0.39'a yakın. Substep Snow'un ~2.3 katı
   (voxel 0.025: ~180).
   *Bozuksa:* madde listede yoksa eski exe; Snow ile aynı davranıyorsa E
   okunmuyor (`fluid.get granular_young_modulus` 1e6 olmalı).

## Kar 4. parti — Voronoi topak yapısı (2026-10-23, derlenmedi; yalnız C++)

Build: yalnız C++ (shader değişmedi). Madde alanı yeni: `granular_snow_clump_size`
(m, varsayılan 0.12); Snow dağılımı 0.05 → 0.1.

3. parti canlı (15:14 exe, 15:11 .spv; `Probe-SnowImpact.ps1`, voxel 0.025):
buz tabanı PASS, `hcap` 0. Zemin: 5.6 → 0.5 m/s iki adımda (FLIP'siz 0.17),
ortalama J_P 0.686. Kaya: %65 parçacık %50+ yumuşamış, %20–45 toz — FLIP öncesiyle
aynı. Mandal yalnız ~12 basınçsız yüzey parçacığında (sönüm basınç kapılı).
Homojenlik testi: bağımsız (beyaz) dağılım 0.05 → 0.3 hiçbir şeyi değiştirmedi →
sorun genlik değil İLİŞKİ UZUNLUĞU (P2G ~50 parçacığı ortalıyor).

Sıra:

1. **Yapı atandı.** `Probe-SnowImpact.ps1 -Target floor` ilk satırında J_P aralığı
   ≈ e^±0.1 (0.905 .. 1.105) ve ortalaması 1'den biraz büyük (sınırlar gevşek).
   *Bozuksa:* 0.951..1.051 = eski exe (beyaz gürültü, 0.05).
2. **Kaya: topaklar kalıyor (asıl hedef).** `-Target rock`: `powder` ve `d90`
   3. partiden (yaklaşık 8 000 / 10 000) belirgin düşük, `d50` ise yüksek kalabilir
   (çatlak yüzeyleri). ★ **En sinsi sonuç:** sayılar düşer ama kar hâlâ tek
   kütle — çatlak açılmamış, yalnız kırılganlık azalmış. Ayırt etmek için
   görsel bak: sınırlar boyunca ayrılan 10–15 cm'lik parçalar olmalı.
3. **Zemin:** blok yine sıkışır (beklenen); kenarlardan topak kopması varsa artı.
4. **Topak boyutu duyarlılığı:** SnowProbe gibi türetilmiş bir maddede
   `granular_snow_clump_size` 0.06 / 0.2 dene; parça boyu onunla değişmeli.
   0.1 m voxelde 0.12 → 0.3 m'ye (3 voxel) yükseltilir; bu bilinçli.

## Kar 3. parti — FLIP + yavaş parçacık sönümü (2026-10-23, derlenmedi)

Build: **C++ VE `compile_shaders.bat`**. ★ Önce şunu bil: 15:03 exe'si kar
2. partinin C++'ını içeriyor ama `.spv`'ler 13:15 tarihli — tavan ve buz tabanı
kernel'de YOK. Canlıda J_P 0.303'e indi (taban 0.436) ve durgun blokta %76
parçacık `hcap` bayrağı taşıdı. Bu partinin değişen shader'ları:
`sim_fluid_granular_stress_update.glsl` (2. parti), `sim_fluid_granular_settle.glsl`
(push constant boyutu aynı, pad0 → `damping_max_speed`).

Canlı ölçüm (`scripts/ipc/Probe-SnowImpact.ps1`, voxel 0.025, 17 576 parçacık,
0.3 m blok ~2 m'den, ~6 m/s; adım 49 ms, 80 substep):
düz zemin → ortalama J_P 0.68, 13 parçacık %50+ sertlik kaybı, dinlenmede J_P max
1.06 → 1.17 tırmanıyor (titreşim mandalı); küre kaya → ortalama J_P 1.73, %67
parçacık %50+ kayıp, %49 toz. Model kırılıyor (geometri ayrımı PASS), ama düz
zeminde ölü durma ve kayada topaksız toza dönme.

Değişiklik: Matter yolunda kar için FLIP (Snow `solver_flip_blend` 0.95; P2G
gerilme impulsundan ÖNCE taban kopyası, aynı ağırlıklarla normalize;
`granularFlipSnapshotWanted`), settle sönümü kar için yalnız < 0.5 m/s parçacıklara.

Sıra:

1. **Shader'lar derlendi.** `.spv` tarihleri `.glsl`'lerden yeni. Bozuksa her
   şeyi maskeler.
2. **Kar 2. parti kernel'i.** `Probe-SnowImpact.ps1 -Target floor`: J_P floor
   PASS (≥ 0.436), `hcap` ~0.
3. **Titreşim mandalı yok.** Aynı koşuda "rest ratchet" PASS: dinlenmede J_P max
   sabit. *Bozuksa:* settle `damping_max_speed` okunmuyor (eski settle .spv).
4. **FLIP etkin.** Kaya vakasında çarpmadan sonra `v_max` 3. partiden yüksek
   kalmalı (parçalar uçuyor); düz zeminde blok "ölü durmamalı" (çarpma sonrası
   ilk örnekte v_mean önceki 0.17'den belirgin büyük).
   ★ **En sinsi sonuç:** FLIP açık görünür ama taban alanı boş/sıfırsa G2P
   v_p + v_new − 0 yapar → kar ivmelenip patlar ya da `invalid` > 0. Patlama
   görürsen önce scratch alanın dolup dolmadığına bak.
5. **Kayada topak kalıyor mu.** `d90`/`powder` 3. partiden (8 662 / 17 576)
   belirgin düşük olmalı; hâlâ yarısı tozsa θs (0.0075) bu çözünürlük için
   fazla kırılgan — ayarı ölçerek değiştir.
6. **Kum regresyonu.** Sand/Gravel (DP) domain'inde FLIP KAPALI kalmalı
   (`flip_blend` yalnız snow'da okunur): yığın davranışı değişmemeli.

## Ayrı parti — ölçeklenmiş küp + collider takibi (2026-10-23, derlenmedi)

Build: yalnız C++ (shader yok, yeni .cpp yok). Dosyalar: `scene_data.h`,
`SceneSelection.h/.cpp`, `scene_ui_hierarchy.cpp`, `RtApi.h`, `RtApiFluid.cpp`,
`RtIpc.cpp`, `RtPython.cpp`.

Kökler:
- **Küp eski boyuta dönüyor** = inspector'ın Pos/Rot/Scl alanları. Herhangi bir
  alan değişince `fromTRS(pos, rot, scale)` TÜMÜYLE handle'a yazılıyor; bu üçlü
  seçim anında alınmış bir KOPYA (`sel.selected`). IPC `scene.set_transform`,
  undo, script onu hiç tazelemiyor; gizmo da bir kare geriden tazeliyor
  (`getPivotMatrix()` yeni matris uygulanmadan ÖNCE okunuyor). Sonra Pos'a
  dokunmak eski ölçeği geri yazıyordu. IPC yolu kendisi temiz: canlıda
  ölçek 2 → taşı → ölçek 2 kaldı. Düzeltme: alanlar çizilmeden önce
  `syncObjectTransformFromHandle()` canlı handle'dan tazeler (önbellek hâlâ
  aynı matrisi veriyorsa Euler korunur).
- **Collider eski pozda kalıyor** = `getSurfaceMeshCacheForObject` epoch memo'su.
  Epoch yalnız geometri jenerasyonu ya da keyframe re-pose ile bitiyordu; düz
  transform düzenlemesi (inspector, gizmo, `TransformCommand`=IPC/undo, script)
  bitirmiyordu. Tek istisna seçili obje + duraklatılmış timeline (gizmo
  çiziminde). Düzeltme: memo, handle'ın build anındaki matrisiyle eşleştikçe
  geçerli (tüketici tazeliği kontrol eder; tüm yollar kapsandı).
- **IPC körlüğü**: `collider.list/get` yalnız yazılmış `bounds_min/max`'ı
  (obb modunda KULLANILMAYAN -1..1) gösteriyordu. Yeni salt-okunur alanlar:
  `resolved_valid`, `resolved_world_min/max`, `resolved_size` — çözücülerin
  kullandığı resolver'dan.

Sıralı kontrol (bağımsız/hızlı önce):

1. **IPC readback var mı** — `Invoke-RtIpc collider.list @{}` →
   `Default_Cube Collider` satırında `resolved_valid: true`, `resolved_size`
   küpün ölçeğine eşit (ölçek 1 küp için ~2,2,2).
   Yoksa: eski exe (zaman damgasına bak).
2. **Collider IPC ile takip ediyor** — `scene.set_transform @{name='Default_Cube';
   scale=@(2,1,1)}` → hemen `collider.list`: `resolved_size` ≈ (4,2,2);
   ardından `translation=@(0.5,…)` → `resolved_world_min/max` x'te +0.5 kaydı,
   boyut aynı. Timeline OYNARKEN de aynı sonuç gelmeli.
   Eski boyut/konum dönüyorsa: memo kontrolü devreye girmiyor (matris
   karşılaştırması); `resolveObjectTransformForSimulation(…, false)` bulamıyor olabilir.
3. **Panel: ölçek korunuyor** — küpü seç, IPC ile ölçekle (2,1,1), sonra
   inspector'da Pos'u sürükle. Ölçek 2 KALMALI, Scl alanı 2 göstermeli.
   Aynısını gizmo ile ölçekleyip inspector'dan taşı. Bozuksa: sync çağrısı
   alanlardan önce çalışmıyor.
4. **Panel: döndürülmüş obje Rot sürüklemesi** — Rot.y'yi 0→120 sürükle;
   değer sıçramamalı/titrememeli. Titriyorsa: sync her kare tetikleniyor
   (karşılaştırma toleransı / konvansiyon uyumsuzluğu).
5. **Undo** — ölçek + taşıma sonrası Ctrl+Z: obje ve collider birlikte döner.
6. ★ **Sinsi olan** — kar MPM sim'i (reset + oynat) ölçeklenmiş/taşınmış küpe
   çarpmalı. Collider overlay doğru görünüp sim eski yere çarpıyorsa: sim kare
   önbelleği eski kareleri oynatıyor (bu parti önbelleği temizlemiyor —
   reset gerekir). Bu sessizce makul görünür; kar havada eski küpe oturur.
7. **Perf** — `rt.perf`'te `sim.collider.surface_cache_rebuild` statik sahnede
   kare başına 0 kalmalı (memo hâlâ tutuyor). Artıyorsa: handle matrisi her
   kare yeniden yazılıyor (bit düzeyinde değişen bir yazıcı var).

Bilinen sınır: inspector Pos/Rot/Scl düzenlemesi hâlâ undo kaydı ATMIYOR ve
`fromTRS`'in dönüş inşası `getPivotMatrix`'ten (Rz·Ry·Rx) farklı görünüyor —
doğrulanmadı; döndürülmüş objede ayrı iş, bu partide dokunulmadı.

## Kar 2. parti — sertleşme tavanı, buz tabanı, yoğunluk dağılımı (C++ derlendi, SHADER DERLENMEDİ)

Build: **C++ VE shader** (`compile_shaders.bat`; push constant boyutu değişmedi, iki pad alan oldu).
Not: `docs/dev/KAR_STOMAKHIN_MPM.md` §6.

Canlı ölçüm (1. parti build'i, 87 552 parçacık, 0.1 m voxel, ~3 m düşüş, 7 m/s çarpma):
serbest düşüşte J_P = 1.000 sabit (kayma yok), 9 substep / 92 ms. Çarpma karesinde
7646 parçacığın modülü dalga sınırına kırpıldı (substep bir kare geç kaldı), blok
eşit sıkıştı (ortalama J_P 0.74, min 0.395 = buzdan yoğun), hiçbir parçacık
gevşemedi (max J_P 0.88), sonra 184 substep / 430 ms'de KALICI kaldı.

Ek ölçüm (1. parti exe, 13:26): çarpmada adım 131 → 528/420 ms (dt 1/60);
GPU kernel toplamı substep başına ~2.3 ms, bunun ~1.7 ms'si `sim_matter_p2g`.
Substep başına maliyet makul; patlayan şey substep SAYISI (9 → 184). 24 fps
zaman çizelgesinde dt 2.5× → "kare başına saniyeler". Domain DEM yolundan
geçiyor: Sand → Surface madde bağı Sand'in `dem` taşımasıyla tane çözücüsünü
açıyor (0 tane varken) ve backend'i Vulkan'a kilitliyor.

Sıra: bağımsız ve hızlı görülen önce.

1. **Shader yüklendi.** Log'da stress_update push-constant hatası yok.
   *Bozuksa:* .spv eski; aşağıdakilerin hepsini maskeler.
2. **Maliyet sabit.** `Probe-SnowStomakhin.ps1` çıktısında `sub` serbest düşüşte
   VE çarpmadan sonra aynı (~9·√5 ≈ 20). Adım süresi ~200 ms civarı, 430 değil.
   *Bozuksa:* çarpmadan sonra artıyorsa boyutlandırma hâlâ ölçülen J_P'ye bakıyor
   (eski exe). Daha ucuz istersen Snow'da `granular_snow_max_hardening` = 1 →
   ~9 substep (sıkışınca sertleşme yok, gevşeme/kırılma yine çalışır).
3. **Kırpma sıfıra yakın.** `hcap` (hardening_capped) çarpma karesinde ~0.
   *Bozuksa:* yüzlerce = substep tavana göre boyutlanmıyor.
4. **J_P buz tabanı.** Probe "J_P floor respected" PASS (≥ 0.436).
5. **Başlangıç dağılımı.** Seed sonrası ilk satırda J_P ≈ 0.951 .. 1.051.
   *Bozuksa:* 1.000/1.000 = `assignSnowHeterogeneity` çağrılmıyor.
6. **Gevşeme (asıl hedef).** Probe "loosening" PASS: J_P max, başlangıç
   dağılımının üstüne çıkıyor. ★ **En sinsi sonuç:** blok yine eşit sıkışıp
   esniyor, J_P max başlangıçtaki 1.05'te kalıyor. Bu, düz levhanın düz zemine
   çarpmasının geometrisi olabilir (yanal kuşatma, çekme yok) — 7. madde ayırır.
7. **Geometri ayrımı (kullanıcı/ajan):** aynı karı küre olarak ya da eğik bir
   collider'a düşür. Kürede/eğimde kırılıp levhada kırılmıyorsa model doğru,
   levha testi çekme üretmiyor. İkisinde de kırılmıyorsa granular hatta FLIP
   eksikliği (KAR notu §6) sıradaki iş.
8. **Görsel:** kenarlardan topak kopması, açık kalan çatlaklar.
9. **Backend kilidinin sebebi görünüyor.** Domain sekmesinde kilitli backend
   altında "DEM is on because Sand (substance binding) uses granular transport
   'dem'…" yazmalı; `fluid.matter_models` → `grain_ownership.origin` aynı şeyi
   döner. Sand bağını kaldırınca (ya da reset sonrası) CPU seçilebilmeli.
   *Bozuksa:* yalnız "Grains need the Vulkan backend." = eski exe.

---

## 1. parti (2026-10-23) — Stomakhin dalı + sayaç körlüğü + gerilme yoğunluğu

Canlı sonuç: sayaçlar ölçülüyor (measured true, counted = n), snow yasası seçili,
serbest düşüşte J_P sabit — 1–3. maddeler PASS sayılabilir. 4–7 yukarıdaki partiye
devredildi.

## Önceki parti (2026-10-22, 3. parti) — henüz doğrulanmadı

Önceki partinin açık deneyi (RT gaz phase-cache var/yok iki build) bu partiyle
çakışmıyor; ayrıntısı `RT_GAS_OPTICAL_PERFORMANCE.md` sonunda.

Build: C++ yeterli (shader değişmedi). Değişen: `MaterialNodesV2.h` (derleyicide
`inPin`: değeri pinde yaşayan 25 giriş artık pin değerini derler),
`ParticleRenderBridge.cpp` (yakın mesh havuzu kapasite kuralı
`nearPoolCapacityFor`). Gerçek shard + mineral ölçümü bu build'den önce canlı
koşuldu (KUM_TANE…md §8b).

Sıra: bağımsız ve hızlı görülen önce.

1. **Mix Color rengi render'a ulaşıyor.** Yeni principled materyal → node
   grafiği: Mix Color (Fac 0, Color1 kırmızı, bağlantısız) → Base Color;
   bir küpe ata, Rendered (Vulkan RT). Küp **kırmızı** olmalı.
   Ajan yolu: `rt_grain_mineral_material.py` yerine aynı grafiği
   `nodes.set_param` ile kurup `render.probe`/ekran görüntüsü.
   *Bozuksa:* gri (0.5) = derleyici hâlâ literal basıyor (eski exe — zaman damgası).
2. **Eski sahneler.** Mix Color / Hue-Sat / Combine Color kullanan kayıtlı bir
   materyali aç. ★ **En sinsi sonuç:** render'ı DEĞİŞEBİLİR — bu düzeltmenin
   kendisi: artık panelde görünen pin değeri çiziliyor, eskiden 0.5/1.0 literal.
   Değişen sahne bug değil; değişmeyen ama panelde farklı renk gösteren sahne bug.
   *Bozuksa:* bağlı girişli bir node'un rengi değiştiyse `inPin` bağlantıyı
   atlıyor (olmamalı — ilk satırı `inOr` ile aynı).
3. **Küre yolu değişmedi.** Near-mesh kapalı bir kum sahnesinde
   `render.instance_stats` ve ms/örnek önceki build'le aynı.
   *Bozuksa:* değişiklik ana havuza sızmış (yalnız yakın havuz kuralı değişti).
4. **Boşluk azaldı.** Ajan fixture'ı kurar ve tam K0'ı koşar:
   `python scripts	est
t_k0_grain_mesh_near_ipc.py --domain K0_Sand --apply --distance 1.5 --library fracture`.
   Beklenen hepsi-mesh vakası (4591): `tlas_instances` ≈ 6.1 bin (önce 12321);
   1.5 m (780): ≈ 1.0–1.6 bin (önce 3105). Script artık kullanıcının near
   `sources` listesini de geri yükler (`near_restored` true).
   *Bozuksa:* hâlâ ~12 bin = eski exe ya da kapasite yolu eski kuralı kullanıyor.
5. **Kamera hareketi.** Fixture'da kamerayı birkaç kez az oynat (aynı mesafe
   bandında): her harekette yapısal yeniden kurma olmamalı (`perf.list`
   `accel.vulkan_rt.rebuild` sayısı hareket başına artmamalı, yalnız kapasite
   aşılınca artar).
   ★ ms/örnek boşluk azaldıktan sonra hiç değişmezse "boşluk bedava" diye
   okunur — opak shard ölçümü (§8b) zaten bunu söylüyor; ama yalnızca
   `tlas_instances` gerçekten düştüyse geçerli. Önce slot sayısına bak.

6. **Grafik silme + Color Ramp durakları + pin değeri (tek script).**
   `python scripts	est
t_material_node_parity_ipc.py` → `PASS`
   (pin_value, remove, ramp_stops). Kendi küpünü 60 m yukarıda kurar, `render.probe`
   ile ölçer, kamerayı geri yükler.
   ★ **ESKİ exe'de bu script uygulamayı ÇÖKERTTİ** (2026-10-22): `nodes.set_property`
   iç içe dizi değeri (`[[...],...]`) alınca pipe koptu, süreç kapandı. Eski
   `nodeParamFromJson` 3 elemanlı diziyi Vector3 sanıp `get<float>()`'u bir dizi
   üzerinde çağırıyordu; atılan `json::type_error` yakalanmıyor gibi. Yeni build'de
   bu yol `Rows` olarak ayrıştırılıyor. Yine de kontrol et: IPC dispatch'i
   `std::exception` (yalnız `runtime_error` değil) yakalıyor mu? Yakalamıyorsa
   herhangi bir metoda bozuk tipli değer göndermek uygulamayı düşürür.

Doğrulama (ajan): değişiklikler derlenmedi. Derleyici hatası canlıda bulundu
(kırmızı Mix Color gri çizildi), düzeltme sonrası koşulmadı.

---

## Ayrı parti (aynı build): fracture shard'ları flat SoA

Kaynakta, derlenmedi. Değişen: `scene_ui_menu.hpp` (shard üretimi + silme),
`scene_data.h` (`nodeWorldCenter`, `getRigidBodySourcePivot`, kayıt öncesi
rest snapshot'ı, yeni `forgetBodyMeshCaches`), `ThermalFractureBridge.cpp`
(`nodeMeshVolume` artık `indices` üzerinden). Yeni .cpp yok, IPC yüzeyi değişmedi.
Her shard artık **tek** `TriangleMesh` (ardışık üçlü, `indices[i]==i`); facade üretilmiyor.

Sıra: bağımsız ve hızlı görülen önce.

1. **Shard tek flat mesh.** `scene.add_primitive cube` → `physics.fracture_object`
   (site_count 24) → bir `shard_objects` adına `scene.object_info`.
   Görmen gereken: `meshes: 1`, `triangles` > 0, `has_bounds: true`.
   *Bozuksa:* `meshes: 0` = hâlâ facade üretiliyor (eski exe — zaman damgasına bak).
2. **Görünüm/malzeme aynı.** Viewport'ta preview gap'li shard'lar; exact modda dış
   yüz kaynak dokusunu taşır, kesik yüzler `<obje>_Interior` malzemesinde.
   *Bozuksa:* hepsi tek renk/doku (0,0) = materialID/uv özniteliği yazılmamış.
3. **Seçim/picking.** Bir shard'a viewport'ta tıkla → hiyerarşide o shard seçilir;
   Rigid Body listesinde tıklamak da viewport'ta seçer.
   *Bozuksa:* `rebuildMeshCache` flat shard'ı `direct_mesh_nodes`'a almıyor.
4. **Re-fracture + unfracture.** Aynı objeyi farklı seed'le ikinci kez kır, sonra
   `physics.unfracture_object`. Beklenen: ikinci kesimde eski shard'lardan **hiçbiri**
   kalmaz (obje sayısı = yeni shard sayısı + diğerleri); unfracture orijinali geri
   koyar, `__shard_` adlı obje kalmaz.
   *Bozuksa:* çift shard = `fractureEraseNodes_` flat'i silmiyor.
5. **Make Breakable → kır → simüle et.** `physics.make_fracture_group` (veya panel),
   `physics.break_fracture_group`, birkaç kare oynat. Shard'lar ayrılıp düşmeli,
   grup kütlesi hacme göre (24 shard'lık 2 m küp ≈ 8 m³ × yoğunluk), **24 kg değil**.
   Patlama merkezden dışa doğru olmalı.
   *Bozuksa:* hepsi aynı yöne/orijinden uçuyorsa `nodeWorldCenter` flat'i okumuyor;
   kütle = shard sayısı kg ise `nodeMeshVolume` 0 dönüyor.
6. **Re-fracture SONRASI simülasyon.** 5'i yap, sonra objeyi re-fracture et ve tekrar
   kır/oynat. Çökme veya eski kesimin pozunda donmuş shard olmamalı.
   *Bozuksa:* rigid bake cache'i ölü `TriangleMesh*` tutuyor (`forgetBodyMeshCaches`
   çağrılmamış) — bu UB, bazen "çalışır" görünür.
7. **Kayıt/açma.** `scripts/material_phase_tests/phase_10_fracture_roundtrip.py`
   (script değişmedi, Release kopyası aynı). PASS + yeniden açılışta shard adları ve
   cluster'lar aynı.
8. **★ Simülasyon ortasında kayıt (en sinsi).** 5'teki sahnede shard'lar havadayken
   (kare ~30) `project.save`, sonra `project.open`. Açılışta shard'lar **dinlenik
   (kesilmiş) pozda** olmalı, havadaki pozda değil; kayıttan sonra da canlı sahne
   havadaki pozda kalmalı.
   *Bozuksa:* açılışta shard'lar dağınık duruyorsa kayıt öncesi rest snapshot'ı flat
   mesh'i atlıyor — hata vermez, "kaydettiğim yer" gibi makul görünür. Bu
   add_primitive ile eklenen her flat rigid body için de geçerliydi (eski hata).
9. **Termal kırılma.** Yanmış bir objede `pattern=2` ile kır; panel mesajında
   "no recorded burn damage" yalnız gerçekten hasar yoksa çıkmalı; kırılma bölgesi
   yanık bölgeye yoğunlaşmalı.
10. **K0 tane örneği olarak shard.** (Diğer oturumun scripti; o oturum facade reddini
    kaldırınca) `rt_k0_grain_mesh_near_ipc.py --library fracture`: `exemplars` = shard
    sayısı, `exemplar_triangles` > 0, `status: active`.
    *Bozuksa:* `missing source` = `gatherNormalizedSceneFaces` shard'ı bulamıyor.


## 2026-10-22: prosedürel tane kütüphanesi ve DEM/MPM/fluid görünümü

Derlemeyi kullanıcı yapar; ajan build veya uygulama başlatmadı. Yeni
`UI/GrainMeshLibraryUI.cpp` proje/filters listelerine kaydedildi.

1. Üstteki materyal/K0 maddeleri 1–6'yı yeni executable ile tamamla.
2. Domain'i Splat/Particles görünümünde, geometry=icosphere olarak hazırla;
   birkaç carrier doğduktan sonra timeline'ı duraklat, kamerayı Near Distance
   içine getir. Ayrı terminalde:
   `python scripts/test/rt_grain_library_options_ipc.py --domain K0_Sand --render`
   Beklenen PASS: rock ve voronoi için near_instances>0; yanlış string/sayı/
   iç içe dizi sonrası IPC hâlâ cevap verir. Probe ayarları geri yükler.
3. 2'yi küçük **DEM, MPM ve fluid** sahnelerinde ayrı çalıştır. Virtual Grains
   8 iken mesh açık → primary_visual_children=1; kapalı → eski ayar geri
   gelmeli (MPM Physical Carriers açıksa primary=1 kalması doğrudur).
   Whitewater sayısı etkilenmemeli. Rendered RT ve Solid/Matcap görünümünde
   aynı carrier hem küre hem mesh olarak çift çizilmemeli.
4. Near Grain Meshes → Grain Shape: Procedural Rock / Rounded ve Procedural
   Voronoi / Angular kaynak obje seçmeden çalışmalı. Scene obje sayısı sabit
   kalmalı. Near Distance/bütçe dışında kalan carrier'lar küre kalmalı.
5. Scene objects seç; iki farklı materyalli kaynak (mümkünse bir kaynakta iki
   yüz materyali) ekle. Use Exemplar Face Materials açıkken yüz materyalleri
   ve UV'ler korunmalı; kapalıyken domain/substance/ıslak bantlar kullanılmalı.
   Kaynakta materyal atamasını değiştir: paused sahnede yakın mesh güncellenmeli.
   Uzak küreler bu seçenekten etkilenmez; yakın/uzak renk farkı bu modda beklenir.
6. Kaydet/aç: shape ve source_materials aynı kalmalı. Eski projede scene/false.
   K0 A/B ölç: near_restored=true; benchmark kendi shape/materyal seçimlerini
   geri yüklemeli. K0 fixture önce yeniden kurulmalı, önceki sahne kaydedilmemişti.
7. `scripts/test/grain_mesh_library_test.cpp` kaynak testi kullanıcı test
   derlemesinde çalıştırılmalı; ajan proje kuralı gereği derlemedi.
8. Fiziksel boyut dağılımı sonraki iştir. Önceki emitter durması kullanıcı
   tarafından sahne kurulumuna bağlandı; çözücü bug'ı olarak açık tutulmuyor.

Ajanın build gerektirmeyen kontrolleri: Python AST, JSON/XML okuma,
IPC descriptor üretimi/--check, ilgili dosyalarda diff whitespace kontrolü.
Canlı render, kayıt round-trip ve C++ kabul testleri çalıştırılmış sayılmaz.


## 2026-10-22: Add Mesh / Grains–Crystals üreticisi

Ayrıntılı sözleşme: `docs/dev/GRAIN_CRYSTAL_GENERATOR.md`.

1. Kullanıcı derlemesinde yeni `Scene/GrainGenerator.cpp`, `UI/GrainGeneratorUI.cpp`
   ve `Api/Rt{Api,Ipc,Python}GrainGenerator.cpp` modülleri alınmalı (vcxproj/filters
   kaydedildi; CMake source glob'ları kapsıyor). Ajan derlemedi/uygulamayı başlatmadı.
2. Add Mesh → Procedural Generator → Grains / Crystals: Rock/Sand, Voronoi,
   Snow Crystal ve Mineral Crystal. Seed, adet, biçim, çap aralığı ve ağırlıklı
   materyaller kullanıcı kontrolünde olmalı. Yeni kütüphane gerçek flat objeler olmalı.
3. Ayrı terminal: `python scripts/test/rt_grain_generator_ipc.py` → PASS.
   `--keep` geometri/materyal/kayıt incelemesi için örnekleri bırakır.
4. Mesh Grains → Create Grain / Crystal Library… aynı pencereyi doğru hedefle
   açmalı; Generate seçili domain'e kaynakları ve kendi materyallerini bağlamalı.
   Tek Undo/Redo objeleri ve önceki domain kaynak ayarlarını birlikte geri getirmeli.
5. Karı alt/üst/yandan incele: kapalı, kalınlıklı dallı plaka; uç/kenar delikleri
   ve üst üste iç dal yüzleri olmamalı. Salt: 4 kenar/0 uç; Quartz/Ice: 6 kenarlı
   prizma; Sugar: seçilebilir uzama/uç. Aspect ve seed sonuçta görülmeli.
6. Üretilen örnekleri içeren proje kopyasını kaydet/aç. İsimler, geometri,
   yerleşim, UV ve materyaller korunmalı. Küçük çaplı kar örnekleri de Mesh Grains
   kaynağı olmalı (SoA yüz okumasında mutlak minik-alan elemesi kaldırıldı).
7. Eski near_mesh_shape=rock/voronoi sahneleri açılmalı; yeni UI'de ayrı sabit
   üretici dropdown'ı görünmemeli. Scene object listesi tasarımı ertelendi.
8. Kullanıcı kaynak testlerinde `grain_generator_test.cpp` + mevcut
   `grain_mesh_library_test.cpp` çalıştırılmalı. Fiziksel boyut dağılımı hâlâ ayrı iş.

## Voronoi edge rounding acceptance (2026-10-22)

- User build: GrainGenerator.cpp and GrainGeneratorUI.cpp integration.
- Generate Voronoi with Examples=1: an irregular cut cell, not an uncut cube.
- Change seed: different geometry; repeat seed: deterministic geometry.
- Compare Roundness=0/0.6 with Edge Detail=6 under lighting and wireframe.
- Compare Edge Detail=2/8: real triangle density must change when rounded.
- Aspect=(0.7,1.6,1): verify smooth normals and closed surface.
- Repeat via Python and scene.generate_grains IPC with identical parameters.
- User compile/run scripts/test/grain_generator_test.cpp: new single-cell,
  seed, closed-surface, smooth-normal and detail regressions.
- Recheck Snow Crystal, materials, Undo/Redo and save/load.

Codex did not build or launch the application; visual acceptance remains pending.

## DEM physical size distribution (2026-10-22, user build/live checks pending)

- Build C++ and DEM / grain-fluid / grain-MPM shaders together (DEM revision 28).
- Build/run scripts/test/matter_grain_size_test.cpp and existing birth/stages tests.
- In a paused empty scene, external terminal: python scripts/test/rt_grain_size_ipc.py.
- UI: min radius .01 m, max .02 m, Sand + large stones, large fraction .10.
  Verify Sphere and Near Mesh agree, larger stone resting heights, impacts,
  sleep/wake, source spacing, reset replay, project save/load and cache scrub.
- Recheck uniform/log-uniform, max=0 legacy fixed size, invalid patches unchanged,
  and all five size fields locked while particles are live.
- Recheck wet/liquid/gas/thermal/MPM momentum/heat regressions for mixed sizes.
- Dense 1:2 and 1:4 piles: inspect explicit bucket/list/contact capacity holds;
  existing 16 bucket / 32 list / 24 history limits remain, no silent truncation.
- See docs/dev/DEM_SIZE_DISTRIBUTION.md for settings and physical semantics.

No project/shader compilation or application/live IPC verification performed by Codex.

## DEM size controls and visible refusal feedback follow-up

- Pending user build: size edits are staged together, applied atomically with
  Apply size settings; invalid ranges are shown beside controls; reset required
  blocks applying sizes while preserving the draft. Discard restores saved sizes.
- Large stone share now uses percent with small/large count labels and estimated
  volume/mass dominance; optional 95% sand / 5% stones preset. Physics RNG unchanged.
- Rejected settings and held solver steps display prominent orange/red wrapped text.
- Verify two radius edits do not apply individually; Apply/Discard; external IPC
  settings refresh the draft; pending size edits do not alter other solver changes.
- Live IPC diagnosis in Physics Domain 1: damping bound requested 529 substeps,
  cap 512. Raised cap to 1024; steps dt=1/60 and dt=1/24 succeeded, held=false.
  Same 1143 grains retained; center Y .828615 -> .808913 -> .747721 m.
  Timeline remained paused at frame 22; sizes and source unchanged, no reset/save.
- This verifies the current executable; pending UI source changes need a user build.

## DEM mixed-size contact pile / runtime neighbour lists (revision 29)

Live reproduction at paused frame 17, Physics Domain 1: minimum radius .026 m,
maximum .083 m, bimodal large fraction .07, seed 1360, max_substeps 1024.
One external dt=1/24 step held with "grain neighbour list full (>32 ...)" at
4289 grains. This is a separate capacity failure from the previous CFL limit.
No reset, size/source edits or save were performed during this reproduction.

Pending user build:
- Rebuild C++ and all sim_matter_grain shader variants together (revision 29).
- Fixed-size domains retain 32-list baseline; mixed sizes start with 128 slots.
- Standalone DEM retries list-only overflow with a doubled budgeted buffer from
  unchanged host state; shared MPM/liquid clocks refuse replay to avoid double steps.
- Allocation includes old+new peak memory and checks shader index width.
- IPC diagnostics: runtime.neighbour_list_capacity / neighbour_list_retries.
- With the failing pile open and paused, external terminal:
  python scripts/test/rt_grain_neighbour_capacity_ipc.py --steps 8
- Check actual floor contact, dense mixed pile, fixed-size baseline, storage/working
  budget failures, and shared-clock mixed cases. Actual contact history remains 24;
  this change removes the 32-candidate ceiling, not all other solver budgets.

Codex performed source checks only for revision 29; no compilation or live run of
new code. The open executable still has revision 28 and cannot use this fix yet.

## DEM contact history overflow at floor impact (revision 30)

Live read-only diagnosis: paused frame 29, Physics Domain 1, 94,915 grains,
minimum radius .011 m, maximum .036 m, large count 995 (1% authored), seed 1360,
max_substeps 2048. Status: contact count/history-in-flight exceeds legacy 24.
No reset, setting mutation, extra tick or save performed during this diagnosis.

Pending user build and acceptance:
- Rebuild C++ plus all sim_matter_grain shader variants together, revision 30.
- Mixed-size history starts at 64 records, fixed-size remains 24. Four 32-bit
  mask words support records across boundaries 31/32, 63/64 and 95/96 up to 127.
- Block metadata is now owner + four masks + rest + compensation (7 words),
  following the capacity-sized identity->block map. Sleep/audit offsets changed.
- Standalone DEM history overflow retries from unchanged canonical host state
  with capacity doubled to at most 128, charged to actual/peak memory budget.
- Shared MPM/liquid clocks cannot replay and continue to refuse overflow.
- Adaptive CFL retries use observed maximum contacts + 4, bounded by allocated
  history; shared clocks use the full allocated history bound.
- runtime.contact_history_capacity / contact_history_retries expose allocation.
- External terminal with the failed pile paused:
  python scripts/test/rt_grain_neighbour_capacity_ipc.py --steps 8
- Verify floor impact and sustained dense packing beyond 24 contacts, sleep/wake,
  changing neighbour identities, population reorder, GPU-resident fixed-size baseline,
  friction/rolling persistence, conservative memory refusal, and mixed-owner safety.
- This raises mixed-size memory use (64 records for every grain initially);
  it is not sparse per-grain contact storage and does not establish million-grain
  cinematic performance. At 128 records a million-grain history alone is ~3.34 GiB,
  potentially above a device's single-buffer limit. The checked budget must refuse it.

Revision-30 work received source checks only. No build/shader compilation or live
physics acceptance of the new code was performed by Codex.

## Size-aware contact CFL bound (host-only follow-up, shader ABI stays revision 30)

Live diagnosis at frame 26: .011/.036 m bimodal 1%, 86,462 grains, cap 2048,
held at damping bound 2081. Existing bound combined globally smallest mass with
largest radius stiffness. This pair does not describe a physical grain.

Source change: for each inertial mass mi and radius ri, bound incident stiffness
by alpha*max(ri,2*ri*Rmax/(ri+Rmax)); reduce maximum ki/mi across particles.
Then omega^2 <= 2*contact_budget*max(ki/mi), retaining conservative damping and
accuracy algebra. Wet bridge bound remains conservative. Mixed-size rolling
uses the conservative unequal-radius rotational factor. Equal dry sizes retain
the old normal bound. No stiffness, restitution, size or substep cap is modified.

Validation:
- Analytical wall/pair inequalities checked for .011/.016/.026/.036 m; new
  normal-rate bound is .468085 of the old fictitious min-mass/max-k bound.
- Added C++ regression for unequal pairs, walls and equal-size parity in
  scripts/test/matter_grain_size_test.cpp (not compiled/run by Codex).
- Live revision-30 executable, unchanged authored settings: 12 external ticks
  at dt=1/48 succeeded, 36..42 peak contacts, history 64, neighbour list 128,
  no hold. This validates shorter dt under old bound, NOT the new host code.
- User rebuild C++, no shader ABI change. Repeat original dt=1/24 dense impact
  and sustained pile, inspect cfl_stiffness_mass_rate_s2 and substeps.
- Repeat fixed-size convergence, high rolling friction and wet/shared-clock
  cases. Lower substep count is accepted only with stable/converged physics.
- No reset, material/size edits, substep cap changes or scene save by Codex;
  12 manual half ticks advanced the live simulation, timeline left paused.

## Explicit reset clears stale DEM CFL hint

Live check confirmed contact_resolution=8 was applied; a NEW dt=1/24 step still
held on damping (2129 > 2048), not accuracy. Lowering accuracy cannot remove a
damping limit. The new size-aware bound remains pending a user C++ build.

Source audit additionally found resetGridDomainStates cleared host particles but
only invalidated generic field copies; retained matter_runtime->grain kept the
previous dense pile's cfl_contact_hint, publication identities and sleep context.
Added invalidateMatterGrainAfterReset in the focused DEM module and minimal calls
in both explicit reset buffer loops. Reusable allocations/growth requests remain;
CFL hint resets to baseline, history/residency/sleep/motion publication is cold.
Do NOT call this on per-step invalidateDeviceCopies (would erase valid history).

User build and regression:
- C++ rebuild only, shader ABI stays revision 30.
- After a dense pile/error, reset to sparse births and verify first-step
  cfl_contacts_per_grain starts at baseline instead of the old peak.
- Test fixed/mixed sizes, empty reset, repeated reset, and births restarting IDs.
- C++ reset-state regression added to matter_grain_size_test.cpp, not executed.
- Source checks passed. No reset or authored setting changes made in the live
  scene by Codex in this follow-up; one failed probe tick was attempted.

## Live close-out 2026-10-22: DEM continuation and MPM/DEM fixture

User-built executable: existing 100,000-grain DEM scene completed six dt=1/24
steps without holds, 1162..1260 substeps, max contacts 33, mean step_total 1358 ms.
This closes the reported short continuation failure, not long-run/scale acceptance.

Created MPM Sand + DEM Stones in the same scene. Previous domain/source retained
but disabled; editing setup reset previous live particles. New fixture completed
86 dt=1/120 ticks with 12,288 MPM carriers and 10 DEM stones, nonzero bidirectional
contact events and sampled momentum residual ~1e-8 N*s. It remains paused at frame 0.
Actual penetration/support/convergence acceptance remains next work.

Details: docs/dev/DEM_MPM_ACCEPTANCE_20261022.md
Raw measurements and previous configuration: docs/dev/dem_mpm_acceptance_20261022.json
No compilation or application launch by Codex; live tests used external named pipe.

## Seed material UX follow-up

- User C++ build: existing MatterModelControls.cpp now provides the seed panel's
  shared domain-default substance picker and granular MPM/DEM/fluid model label.
- Minimal integration call in scene_ui_simulation_domains.cpp; no feature body
  added to the over-2000-line panel file. No shader changes.
- Verify Seed/Matter pickers remain synchronized; liquid/MPM/DEM labels match
  substance.get; errors are visible; reset+seed and save/load preserve material.
- UI, Python and IPC all use existing updateFluidDomain; no UI-only material state.
- Current live bed is granular MPM, NOT fluid (0 fluid / 12288 MPM / 10 DEM).
  Fluid-like appearance is not dismissed; pile/slope/penetration acceptance remains.
- Source checks passed, no compilation and no live acceptance of the new UI yet.


### DEM seed ownership and physical packing (2026-10-22)

- First Sand seed regression: resolve substance ownership and material before
  birth, after replacement clears old particles. Previously a first seed could
  become MPM and lock that owner until reset.
- UI, Python `rt.fluid.seed`, IPC `fluid.seed` share this core. DEM seeds use
  sphere-volume mass and a maximum-radius lattice independent of voxel/PPC.
  Variable radii keep deterministic IDs; smaller grains begin with gaps and settle.
  This is conservative initial packing, not dense equilibrium packing.
- DEM requested count is a lower bound when truncated (emitted + 1). Candidate
  work is bounded; capacity/scan cutoff has a separate DEM warning.
- User: C++ build only, no shader changes. Codex did not compile or run C++ tests.
- Run updated `scripts/test/matter_grain_birth_test.cpp`: physical spacing/mass,
  granular model, voxel independence, append collision rejection, budget, small box.
- Live: empty domain, builtin Sand, seed replace=true BEFORE any step.
  `fluid.matter_models`: DEM > 0, MPM = 0, reset_pending=false. Repeat after
  reset/reseed. Replace existing incorrectly born MPM particles.
- Identical box/radius with different voxel/PPC must give identical DEM count.
  Custom granular_transport=mpm and Water retain voxel/PPC sampling and owners.
  MPM default + DEM source must seed MPM, even with domain grains enabled.
- Drop Sand above floor to test impact scattering. A zero-velocity resting seed
  is not a bounce test. Restitution and friction were not changed.


### DEM seed column symmetry regression (2026-10-22)

Live read-only evidence after the ownership fix: Physics Domain 1 has 34,512
DEM grains, zero MPM/fluid owners, reset_pending=false, dry grains, zero liquid
bridges, restitution=0.5. Horizontal bounds remain the seed footprint; horizontal
momentum and spin energy are zero, maximum contacts per grain=2. This matches
perfectly aligned columns compressing/rebounding without lateral excitation,
not an adhesive material or MPM ownership fallback.

- Seed now uses deterministic bounded position jitter (10% of maximum radius
  per coordinate) with spacing 2.2002 R. The worst adjacent jitter extremes
  retain at least 2.0002 R separation along their separating axis. Bounds include
  the jitter margin. No velocity kicks or changes to material forces.
- Existing size_seed controls reproducibility through the same UI/script/IPC
  core. Seed positions remain independent of voxel/PPC and particle budget.
- Initial count is lower than the old perfectly touching lattice; this remains
  conservative packing, not a dense packing generator. The box needs room for
  the maximum diameter plus its jitter margin.
- Updated C++ regression assertions cover pair separation, containment, layer
  misalignment, seed repeatability/change, append rejection, budget-prefix stability.
  Codex did not compile/run C++ tests or launch the application.
- User build: ordinary C++, no shader change. Clear/replace and seed again: old
  live positions cannot acquire the new birth jitter. Run with settle-on-seed
  disabled for an observable drop. Verify horizontal spreading and nonzero spin
  after impact, no coherent column bounce, no held step or capacity overflow.
  Repeat with settle enabled and with radius variation; check a final stable pile.
