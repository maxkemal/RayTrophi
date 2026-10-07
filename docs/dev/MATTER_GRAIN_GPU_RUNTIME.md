# H1 dry grain GPU runtime candidate — 2026-10-05

> **Durum:** AKTİF — H1 dry grain GPU adayı; 2026-10-06 fused substep/statik sürtünme/tane kütlesi CANLI PASS (G2 dahil); B2–B8 + B9a build bekliyor (en alttaki bölümler). Parti sırası: [MATTER_H1_GRAIN_ROADMAP.md](MATTER_H1_GRAIN_ROADMAP.md).

2026-10-06 birth-fix kullanıcı build sonrası tam dış IPC PASS:
g60/120=9.80958/9.80961 m/s², floor min y .02133 m, spin1.504 rad/s.
64-grain .6 s min y .01872 m, dry mass drift0; dt-half final COM/RMS farkı0.
İlk adım COM ve dispatch1469/737 farklı: test iki farklı outer dt çalıştırdı.
Önceki birth-overlap RED arşivli, gate değişmedi. Tam log:
`matter_h1_grain_birth_fix_full_live_2026-10-06.json`.
H1-G0/G1 veya üretim çözücü kararı tamamlandı sayılmaz.

- Aynı Matter domain/emitter/particle identity ve kanonik SoA kullanılır.
  Opt-in grain sahibi MPM P2G/stress/G2P/advection yolunu çalıştırmaz.
- GPU: hash-clear → linked cell hash → sphere contact → force/torque integration.
  Hash çakışmaları gerçek hücre koordinatıyla filtrelenir; komşular kesilmez.
  40'tan fazla tane teması CFL bütçesini aşarsa host publication reddedilir.
- Grain doğumları fizik çapından küçük aralıkla kabul edilmez. Mevcut ve aynı
  kaynaktan yeni kabul edilen taneler sparse birth hash'te aynı anda kontrol edilir;
  domain duvarından en az radius kadar içeride doğarlar. 64 attempt/request,
  source/frame en fazla262144 deneme; dolu bölgenin kalan isteği accumulator'da
  ertelenir. Kimlik/kütle yalnız kabul edilen doğumda üretilir. MPM davranışı aynı.
- Yerçekimi grain alt adımında bir kez uygulanır. Fizik yarıçapı ayardır;
  render çocuk sayısı 1, görsel yarıçap fizik yarıçapıdır. Kütle emitter'ın
  mevcut `rest_mass_kg * mass_fraction` değeridir; yarıçap kütleyi değiştirmez.
- Dönme mevcut affine SoA'nın antisymmetric/skew kısmında taşınır. Temas torku
  ortak temas noktasını kullanır; mevcut affine cache/serializer bu state'i saklar.
- Kayma: hız sönümü + Coulomb sınırı; bounded rolling direnci. Kalıcı tangential
  displacement/statik sürtünme geçmişi yok. Disipasyon henüz termal enerjiye aktarılmaz.
- Temas alt adımları arasında full-state CPU transferi yok. Frame sonunda host
  publication ve mevcut render/cache köprüsü sürer; uçtan uca GPU residency yok.
- Closed Vulkan, explicit dry granular carriers; gravity + force fields (2026-10-07), moving/skinned colliders and bone proxies with vertex velocities. Pore/wet/thermal/
  solid phase kapalı; sıvı/frozen/wet taşıyıcı, hareketli/kinematic collider veya
  force field step'i tutar. Static PlaneY ve doğrudan TriangleMesh/DNA SoA'dan
  flat mesh desteklenir; eski per-face facade resolver kullanılmaz.
- Yeni collider paketi: world-space flat BVH + connected coplanar patch IDs.
  Dört bağımsız temas, sekiz patch ve stack64; taşma tanı bayrağı publication'ı
  reddeder. Bitişik düz üçgenler aynı spring'i iki kez uygulamaz; ortak kenar/
  vertex destekleri deduplicate edilir. Curved facet zincirleri plane testinden
  geçmeden tek patch olmaz. En fazla4096 yüz/100000 taşıyıcı sınırı korunur.
- Geometry fingerprint değişmedikçe BVH/triangle/patch GPU upload yapılmaz.
  Cache/save fizik state'i değiştirmez; BVH derived scratch olarak yeniden kurulur.
- Initial clear/hash → contact → clear/integrate-hash → … → contact/integrate.
  Fusion yalnız kendi updated position'ını hash'ler; sonraki contact backend'in
  compute barrier'ından sonra bütün yazımları okur. Dispatch4N yerine3N+1.
  Shader ABI11 buffer/80 byte; contact revision2 küçük GPU tanı alanıyla
  doğrulanır. Eksik/eski SPIR-V fizik publish yerine açık build hatası verir.
- UI: Matter > Dry Grain Solver (candidate). Python/IPC:
  `fluid.grain_settings(domain)` / `fluid.set_grain_settings(domain, **patch)`.
  Aynı API/core validation, render lock, transactional patch ve save/load doğrulaması.
  Ayar değişikliğinden önce parçacıkları resetle; disabled MPM varsayılanı korunur.
- `fluid.matter_models` içinde grain settings, spin energy/max spin ve angular
  momentum sorgulanır. Kullanıcıya ait sahne maddeleri/presetleri değiştirilmez.

## Tek build sonrası

Kullanıcı C++ ve `compile_sim_shaders.bat` derler; yeni dört grain `.comp` ortak
`sim_matter_grain.glsl` içerir. Codex build çalıştırmaz.

1. Boş, frame0 paused sahnede dış terminal:
   önce `python scripts/test/rt_h1_grain_runtime_ipc.py --pile-only`, ardından tam koşu.
2. Gerçek emitter/runtime free-fall60/120 Hz, floor hit/spin, 64-grain pile
   dt-half/mass/control ve invalid settings ölçülür. Replay kullanılmaz.
3. Flat ramp/sekme, yüksek yoğunluk/maliyet, repose/uzun yerleşme ve save/load/
   bake/scrub/resume canlı matrisi hâlâ açık. Bu script tek başına G1'i kapatmaz.
4. Mevcut ortak Matter regresyonu:
   `python scripts/test/rt_g2_dry_wet_compare_ipc.py --expect-empty-fluid-skipped`

Statik sözleşme: `python scripts/test/check_matter_grain_contracts.py`.
C++ test kaynakları `matter_grain_params_test.cpp` ve lease testi yazıldı;
Codex tarafından derlenmedi/çalıştırılmadı.

## 2026-10-06 extended baseline ve sonraki build

Mevcut derlemede flat ramp ve 256/1024 tane .6 s kabul PASS, dry drift0.
kn100 kN/m; min y .01952/.02075 m. Son adım89.24/95.08 ms,3273 dispatch;
hız oranı veya üretim performansı kabulü değildir. Tam kanıt
`matter_h1_grain_extended_before_bvh_2026-10-06.json`.
İki flat destek köşesi eski nearest-face yolunda RED: 5. adım x=.017162 m,
gate .0175 m. `matter_h1_grain_corner_before_manifold_2026-10-06.json` arşivli.

BVH/manifold/fusion kaynak ve statik sözleşme hazır; C++ + tüm grain shader build
bekliyor (yeni `sim_matter_grain_integrate_hash.comp` dahil). Sonra dış IPC:
`python scripts/test/rt_h1_grain_runtime_ipc.py --corner-only`, `--extended-only`,
ardından tam script. Köşe gate'i aynı. Geometry/memory/CFL overflow ve shader
revision reddi host state'i yayımlamaz. C++ BVH/stage test kaynakları yazıldı,
Codex derlemedi/çalıştırmadı. Static history/repose/uzun settle/cache kabulü,
wet/heat coupling ve tam GPU render residency hâlâ açık; G1 tamamlanmadı.

## BVH/manifold/fusion canlı sonuç — 2026-10-06

Kullanıcı C++/shader build sonrası köşe aynı .0175 m gate ile PASS: min destek
mesafesi .019195 m (eski nearest-face RED .017162 m). Contact shader revision2
publish guard geçti. Flat ramp ve 256/1024 tane .6 s kabul/mass/control PASS;
dispatch3273→2457, son ölçülen adım39.27/41.33 ms (önce89.24/95.08).
Bu tek koşu son-adım kıyasıdır; genel hız yüzdesi/üretim C7 kabulü değildir.
Tam temel regresyon da PASS: g9.80958/9.80961; floor min y .021424 m,
peak spin1.410 rad/s;64-grain final dt-half COM/RMS fark0, dry drift0.

Kanıtlar: `matter_h1_grain_corner_after_manifold_2026-10-06.json`,
`matter_h1_grain_extended_after_bvh_2026-10-06.json`,
`matter_h1_grain_full_after_bvh_2026-10-06.json`.
Ara tam koşu Windows log dosyası kilidinde kesildi; kayıt retry ile düzeltildi,
son tam koşu tamamlandı. C++/shader değişmedi. Frame0 paused, test authoring
kaynakları kapalı, save yok. Paket rebuild beklemez; kalıcı static friction,
uzun settle/repose, cache kabulü, wet/heat ve tam GPU render residency açık.

## Normal-axis twisting resistance — source pending build

8 s /256 grain baseline: dry drift0,6–8 s COM/RMS range13/14 µm;
translation energy reaches1.33e-8 J but spin remains .017655 J. Existing rolling
resistance intentionally projects out normal-axis spin, so ideal point spheres
retain this mode. Archive: `matter_h1_grain_settle_before_twist_2026-10-06.json`.

New optional `twisting_friction`0..1 (default0 for compatibility) opposes relative
normal-axis spin by contact torque. Approximate contact radius
`min(R_eff,sqrt(R_eff*overlap))`; torque limit coefficient*Fn*contact_radius,
bounded by the48-contact angular impulse budget. Sphere pair torques are equal
and opposite; no force, mass change or uniform angular damping is introduced.
This is kinetic twisting resistance, not persistent static friction history.
Heat coupling is still pending. UI exposes the existing shared setting operation;
Python/IPC `fluid.set_grain_settings` and save/load use the same validation/core.

C++ and grain shader rebuild required, contact revision3 (11 buffers/80 bytes;
twist coefficient bits replace unused triangle-count metadata). External test
`rt_h1_grain_runtime_ipc.py --settle-only` authors twist .1; same energy/settle
gates retained. Then full base and extended tests. No new live PASS claimed.

## Twisting canlı kabul — 2026-10-06

Contact revision3 kullanıcı build ile doğrulandı. Twist .1 /Cn4:8 s spin enerjisi
6.28e-21 J;6 s doğrusal KE nedeniyle6–8 s enerji gate RED (1.313e-5 J/tane).
Yalnız normal damping4→8 Ns/m kontrollü profil değişikliği:8 s PASS; mass drift0,
6–8 s COM/RMS aralığı0; total kinetic energy/tane6.09e-11 J. Koşul sınırları
korundu. Default twist0/Cn4 ve global presetler değiştirilmedi.
`matter_h1_grain_settle_twist_cn4_2026-10-06.json` ve
`matter_h1_grain_settle_twist_cn8_2026-10-06.json` arşivli. Bu test kinetic twist
ve bu profilin yerleşmesini doğrular; static history/repose/wet/heat/cache veya
production GPU residency kabulünü kapatmaz.


## Fused substep + statik sürtünme + tane kütlesi — 2026-10-06 (kaynak, build bekliyor)

Bağlam: [BIRLESIK_MADDE_DOMAIN_TASARIMI.md](BIRLESIK_MADDE_DOMAIN_TASARIMI.md) H1-G0 öncesi maliyet
(DEM ile PBD/XPBD kıyası adil olsun diye) + H1-G1'in repose ve çözünürlük ön şartları.

**Teşhis (ölçümden):** yoğun tabloda tane 16× artınca süre yalnız 1.6× arttı; dispatch
her kolda 3471. Maliyet tane işi değil, alt adım sayısı × dispatch ek yükü. 3471 =
3×1157 alt adım; sınır `contact_dt = .1·sqrt(m/(48k))` = 0.0144·sqrt(m/k) idi. Bu hem 48
temas (eşit yarıçapta en fazla ~12+destek) hem ayrıca .1 güvenlik payı alıyordu.

**Değişen:**
- *Tek dispatch/alt adım.* Durum bank 0 (kanonik pos/vel/affine) ile bank 1 (grain scratch)
  arasında ping-pong; alt adım k yalnız bank k&1'i okur, kendi tanesini bank (k+1)&1'e yazar.
  Temas+entegrasyon+sonraki hash tek kernel (`sim_matter_grain_step`). Hash başları iki
  tablo, girdi = (nesil<<17)|index; bayat nesil zinciri bitirir, alt adım başına clear yok.
  Alt adım sayısı çift (kare kanonik bankta biter). Kare: clear + hash + S step = S+2 dispatch.
- *CFL.* Temas bütçesi 24 (grain+duvar+mesh patch, aşımı yayımlamayı reddeder; eski kapı 40).
  stability = 1/sqrt(2·24·k_eff/m) (Gershgorin, symplectic sınırın yarısı),
  k_eff = max(k, 3.5·kt, 2.81·μr²·k); accuracy = π·sqrt(m/2k)/`contact_resolution` (vars. 24);
  damping = .5·m/(24·(2cn+7cs)); travel = .1r/vmax. Hangisi bağladıysa
  `grain_diagnostics.runtime.substep_limit` söyler.
- *Statik sürtünme.* Cundall–Strack teğetsel yay, kt = `tangential_stiffness_ratio`·k
  (vars. 2/7: teğetsel ve normal temas periyodu eşit). Yay temas düzlemine taşınır
  (büyüklük korunur), Coulomb konisinde kayarken yay koniyle tutarlı yeniden yazılır (LAMMPS
  geleneği). ratio 0 = eski yalnız-kinetik kayma.
- *Yuvarlanma yayı (EPSD2, Ai ve ark. 2011).* Eski yuvarlanma direnci sıfır spinde sıfır
  tork veriyordu ve toplam-sınır 1/48 ile çarpılıyordu: eğimdeki tane yuvarlanarak sürünürdü,
  statik sürtünme tek başına yığını tutamazdı. Şimdi k_r = 2.25·μr²·k·R², sönüm kritiğin .3'ü,
  tork sınırı μr·Fn·R. Bu tek yuvarlanma modeli; eski kinetik yol söküldü.
- *Temas geçmişi.* Tane başına 24 slot × {anahtar+teğetsel yay, yuvarlanma yayı}, iki bank,
  yalnız sahibi okur/yazar. Anahtar: ortak tanenin kararlı kimliği (31 bit), duvar
  0x80000000|eksen, mesh 0xC0000000|patch. Sahip kimliği uyuşmazsa (yeniden sıralama) geçmiş
  boş sayılır. Tampon büyürken (1.5× geometrik) ve reddedilen adımdan sonra geçmiş sıfırlanır —
  `runtime.history_reset_this_step` raporlar. Bellek ~1.5 KB/tane (16k: 25 MB, 100k: 150 MB).
- *Tane kütlesi.* Doğumda kütle = madde yığın yoğunluğu / `packing_fraction` (vars. .6) ×
  (4/3)πr³; voxel/PPC parsel kütlesinden bağımsız. Sand r=.025: 0.1745 kg (eski .10 voxel'de
  0.2, .15 voxel'de 0.646). Yarıçap değişince parçacık reset şart (eski kural).
- *Backend.* Vulkan compute aynı recording içinde aynı buffer setine aynı descriptor set'i
  yeniden kullanır; submit sınırı hâlâ 512 *dispatch* (eskisi gibi). Tüm Vulkan çözücüleri etkilenir.

**Ölçülmedi / açık:** yoğun maliyet (beklenen ~204 dispatch/kare, eski 3471), repose açısı
ölçümü, CPU referans (H1-R) hâlâ eski kinetik yuvarlanmayı kullanıyor — GPU parity iddiası
yok. Frame sonu tam-state host publication sürüyor (H1-C7 kuralına aykırı, açık madde).
Kabul sırası: [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) en üst bölüm.


## Fused partisi canlı sonuç — 2026-10-06 (kullanıcı build, dış IPC)

| Test | Sonuç | Kanıt |
|---|---|---|
| Temel (serbest düşüş/zemin/pile 60-120 Hz) | PASS. g 9.8100/9.8101; 60 Hz 166 alt adım/168 dispatch, `damping` bağlı (öngörü birebir). 64 tane 11.1701 kg = 64×0.17453. Pile dt farkı COM %0.59, RMS %0.18 | `matter_h1_grain_fused_base_2026-10-06.json` |
| `--static-only` (20° düz mesh eğim) | PASS. Yay 2/7: ilk 5.6 mm oturma, sonra son 1 s kayma **0**, 1 yapışık temas. Yay 0: sabit 0.1464 m/s sürünme = analitik m·g·sin20°/c_s = 0.1464 m/s. Kütle 0.174533 kg | `matter_h1_grain_static_hold_2026-10-06.json` |
| `--convergence-only` | PASS. Cn=Cs=1 ile `accuracy` bağlı; 62→122 alt adım, COM %3.07, RMS %2.12 (gate %5). 0.6 s neredeyse elastik geçici rejim: kaba yakınsama, hassas ölçüm değil | `matter_h1_grain_convergence_2026-10-06.json` |
| `--extended-only` (rampa/köşe/256/1024) | PASS. Köşe min destek .01953 m > .0175; dispatch 2457→173 | `matter_h1_grain_fused_extended_2026-10-06.json` |
| `--settle-only` Cn8 8 s | PASS. Son 2 s enerji/tane **1.06e-12 J** (önceki 6.09e-11), COM aralığı 0, kütle sapması 0, 271/271 temas yapışık. Taneler tek katman yayıldı (max 3 temas/tane): repose ölçümü DEĞİL | `matter_h1_grain_fused_settle_cn8_2026-10-06.json` |
| Yoğun maliyet (aynı profil) | 1024/4096/16384 medyan **25.5/24.4/41.1 ms** (eski 173.6/213.8/272.2; 6.8×/8.8×/6.6×), dispatch 209 (eski 3471), kütle sapması 0 | `matter_h1_dense_grain_fused_2026-10-06.json` |
| Kernel profili (timestamp açık, şişkin) | `sim_matter_grain_step` 1024: 243 µs/çağrı, 4096: 289 µs. 4× tane ≈ aynı süre → **gecikme sınırlı**: bağlı listeli hash yürüyüşü bağımlı global yükleme zinciri. Sonraki maliyet kaldıracı dispatch değil, hücreye sıralı parçacık düzeni (adım başına counting sort) | `matter_h1_grain_kernel_profile_2026-10-06.json` |
| Diğer Vulkan çözücüleri (G2 dry/wet, MPM+pore GPU) | PASS (tüm assert'ler, 4 kol). Aynı build iki koşu: kuru kol fark 1e-6–1e-5 m (float atomik gürültüsü), ıslak kol ~1 mm'ye büyüyor. 10-05 build'ine göre her kolda adım 25'te sabit ~6.6 mm COM farkı: kaynak, emitter tohumunun **kaynak nesnenin bellek adresinden** gelmesi (`MatterDomainSources.inl` `&source >> 4`) — her oturum farklı doğum konumu; 360 tanenin r=.26 küredeki ortalama saçılımı ≈6 mm ile uyumlu. Kütle/parçacık sayısı birebir. Yanlış bağlama çöp fizik verirdi, sabit ofset değil; ama oturumlar arası birebir A/B bu tohum yüzünden **mümkün değil** | `matter_g2_dry_wet_after_descriptor_reuse{,_run2}_2026-10-06.json` |

Test düzeltmeleri (kod değil fikstür): statik kolda tane 5 cm'den bırakılınca Cn=4'te
(e≈0.9) 2 s boyunca sekiyor ve uçuşta eğimden aşağı kayıyordu — ölçülen fikstürdü; tane
artık eğime oturmuş doğuyor. Yakınsama kolu ilk koşuda 0.0 fark verdi çünkü `damping`
bağlıydı ve iki kol aynı alt adımı koştu (boş ölçüm); kol artık `accuracy` bağlamasını şart koşuyor.

Açık: repose açısı (gerçek yığın), emitter tohumunun oturumdan bağımsız yapılması (build'ler arası A/B için), H1-G0 DEM–PBD/XPBD kıyası, hücre sıralı
komşuluk, parçacık başına sahiplik, kare sonu host publication (H1-C7).

## Kova komşuluğu + yığılma açısı — 2026-10-06, 2. parti (kaynak, build bekliyor)

Profil ölçümü: step kernel 4× tanede ≈ aynı µs/çağrı → gecikme sınırlı. Bağlı listeli hash
her komşu için bağımlı bir global yükleme zinciri yürüyordu. Yerine üç dönen tablo × kova
(sayaç + 16 indeks; kova sayısı ≥ 4N). Alt adım k tablo k%3'ü okur, kendi tanesini (k+1)%3'e
ekler, (k+2)%3'ü temizler: o tablo en son k−1'de okundu ve ilk k+1'de yazılacak, iki yönü de
backend bariyeri sıralar. Step dispatch'i max(N, kova) iş parçacığı; her biri bir kova temizler.
Kova taşması yayımlamayı reddeder (komşu düşürmek temas silmek olurdu). Fizik değişmedi.

`grain_diagnostics.pile`: yatay ağırlık merkezi çevresinde tane çapı genişliğinde halkalar,
halka yüzeyi = en yüksek tane tepesi − en alçak tane tabanı; zirvenin %20–80'i arasındaki
halkalara en küçük kareler eğimi → `repose_angle_deg`. Sentetik 20/30/38° koni ile Python
aynası 19.5/29.1/37.7° okudu (ayrık istifleme ~0.5° düşük okur). Duvara dayalı/çoklu yığın
ölçüm değildir. Kabul: `--repose-only` (sıra NEXT_BUILD_CHECKS en üst).

## Parçacık başına sahiplik + su–tane bağlama (B3) — 2026-10-06, 3. parti (kaynak, build bekliyor)

Arıza: aynı domain'de su + dry grain → parçacıklar emitter'da doğup kalıyordu (tane adımı
akışkan taşıyıcıyı görünce bütün adımı reddediyordu). Artık taşıma sahibi taşıyıcı başına.

- `runMatterGrainStep` (MatterGrainStep.inl) koordinatör: `partitionMatterGrainOwners` →
  sıvı alt kümesi `state.particles` ile takas edilip `runGpuFluidParticleIntegrateForces` +
  `runMatterGpuStep` (karma Vulkan sıvı şeridi) → `buildMatterGrainLiquidField` +
  `prepareMatterGrainCoupling` → `stepMatterGrainGpu` → `applyMatterGrainLiquidReaction` →
  `mergeMatterGrainOwners`. Kanonik dizi ancak hepsi başarılıysa yazılır: `[sıvı…, tane…]`.
- Tane çözücüsü artık **kendi** bank-0 tamponlarına sahip (`grain_positions/velocities/
  affines`); domain grid tamponlarına ve `fluid_uploaded_particle_count`'a bağımlılık yok.
  Tane alt kümesi kimliğe göre sıralı: geçmiş yuvaları başka taşıyıcı silinince kaymaz.
- Shader revision 7, 15 buffer (binding 14 `Coupling`: tane başına 3 vec4). Bağlama yoksa
  satırlar sıfır → shader'da kesin no-op. Örtük çift: göreli hız 1/(1+dt β(1/m+1/M)) ile
  söner, `m v + M u` korunur; birikmiş sürüklenme impulsu geri okunur.
- Sürükleme: Di Felice (1994) gözeneklilikli; kaldırma: hidrostatik Arşimet, batma oranı
  s = clamp(α_sıvı/ε). Topak: hücre sıvısının tanenin hacim ağırlığı kadar payı (hücrede bir
  tane hacminden fazla tane varsa paylar hücre kütlesine eşitlenir, aşamaz).
- Doğum: tane doğum filtresi yalnız granüler taşıyıcılara karşı aralık tutar; sıvı kaynak
  tane domain'inde voxel/PPC parsel doğumunu korur.
- Render: tane domain'inde sıvı parsel kendi yarıçapıyla çizilir.
- Ayarlar `fluid_coupling`, `drag_viscosity_pa_s`; tanı `grain_diagnostics.liquid`.
- Bulutta koşturulan: `matter_grain_coupling_test.cpp` (g++, kütle yoğunluğu stub'lı) PASS —
  kaldırma 3.6788 m/s² = analitik, sıvı/tane momentum artığı 0. Statik sözleşme PASS.
- Bilerek dışarıda: hacim dışlama, −V∇p, dönme sürüklenmesi, ıslanma (yol haritası B5/B6).

## B4–B9a — 2026-10-06, 2. bulut turu (kaynak, build bekliyor)

| Parti | Çekirdek değişiklik | Dosyalar |
|---|---|---|
| B9a | Cihaz geçmişi yalnız yayımlandığı durum için geçerli; CPU referansı EPSD2 | MatterGrainGpu.cpp, GranularContact.cpp, GranularReference.cpp |
| B5 | Gözenek ağırlıkları + tane hızı → `sim_fluid_divergence_porous`; basınç kuvveti −Vρ∇p | MatterGrainCoupling.cpp, FluidGpuPressure.inl, MatterGrainStep.inl |
| B6 | Tane suyu (emilim/kuruma/doğum), Willett köprüsü, Bond ölçeği | MatterGrainCoupling.cpp, sim_matter_grain.glsl, MatterDomainSources.inl |
| B7 | `solver_kind = xpbd` küçük adımlı aday — **2026-10-06 söküldü** (H1-G0, roadmap B7) | — |
| B8 | Morton hücre sırası; geçmiş kimlikle GPU'da taşınır | sim_matter_grain_permute{,_copy}.comp, MatterGrainCoupling.cpp |
| B4 | Yerleşik bank 0 yeniden kullanımı; transfer sayaçları | MatterGrainGpu.cpp |

ABI: shader revision 10, push constant 112 bayt (`wet`: hücre boyu, kılcal ön çarpan, kopma
sınırı, çözücü türü), 15 buffer; bağlama satırı 2'nin w bileşeni köprü suyu hacmi.
Yeni ayarlar: `volume_exclusion`, `wet_grains`, `water_capacity_fraction`,
`absorption_rate_per_s`, `drying_rate_per_s`, `surface_tension_n_m`, `contact_angle_deg`,
`represented_grain_radius_m`, `birth_saturation` (`solver_kind`, `xpbd_substeps` XPBD ile söküldü) — panel,
IPC, Python, kayıt, descriptor. Yeni test kolları: `--history-only`, `--porous-only`,
`--wet-only`. Kabul sırası NEXT_BUILD_CHECKS en üst.
