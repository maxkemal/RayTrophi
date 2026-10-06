# Matter — kısa devir, 2026-10-06

- Build/shader kullanıcıda; Codex app launch/build yapmaz. Canlı test dış IPC.
- Hedef: tek Matter domain, ortak kanonik GPU SoA/kimlik/local state,
  alt adımda tek transport owner. UI düzeni temel çözücüden sonra.
- Dry grain adayı: emitter → GPU hash/contact/force/spin → mevcut render/cache.
  Birth non-overlap + bounded retries/backlog; kütle/kimlik yalnız kabulde doğar.
- **Canlı temel PASS:** g60/120=9.80958/9.80961; floor/spin; 64-grain .6 s
  dt-half COM/RMS son fark0, dry drift0. Önceki birth108 J RED arşivli.
- **Canlı extended PASS (BVH öncesi):** flat ramp, 256/1024 tane dry drift0;
  min y .01952/.02075 m, son adım89.24/95.08 ms, dispatch3273. kn100 kN/m.
  Kayıt: `matter_h1_grain_extended_before_bvh_2026-10-06.json`.
- **BVH/manifold/fusion kullanıcı build + canlı PASS:** köşe min destek mesafesi
  .019195 m > .0175 gate (eski RED x=.017162). Flat ramp/256/1024 ve tam temel
  regresyon geçti; shader revision2 doğrulandı. Dispatch3273→2457.
  256/1024 son adım39.27/41.33 ms (önce89.24/95.08); tek koşu, genel hız oranı değil.
- Connected coplanar patch +4 destek/8 patch/stack64; taşma step'i tutar.
  Geometry değişmedikçe BVH upload yok. Fused integrate/hash4N→3N+1.
- Önceki BVH paketi için rebuild beklemiyor. Test kaynakları/domain kapalı, frame0 paused;
  save yok. Loglar `matter_h1_grain_{corner,extended,full}_after_*_2026-10-06.json`.
- **Twisting revision3 kullanıcı build + canlı:** spin modu giderildi.
  Cn4 profili6 s doğrusal KE yüzünden enerji gate'inde RED (1.313e-5 J/tane);
  Cn8 + twist .1 kontrollü8 s PASS: dry drift0,6–8 s COM/RMS aralığı0,
  toplam enerji/tane6.09e-11 J. Gate değişmedi; eski/ideal default twist0 korunur.
  Kayıtlar `matter_h1_grain_settle_twist_cn{4,8}_2026-10-06.json`.
  Twist0 eski-profil tam temel regresyon da PASS; yeni build beklemiyor.
  Bu kinetic burulma direncidir; kalıcı statik contact history değildir.
- **2026-10-06 2. parti (build bekliyor): kova komşuluğu + yığılma açısı.** Bağlı liste → 3 dönen
  sabit kovalı tablo (gecikme sınırlı kernel için); `grain_diagnostics.pile` repose ölçümü;
  `--repose-only` (μr duyarlılığı, makullük, tane boyutu yakınsaması). Sıra NEXT_BUILD_CHECKS üstte.
- **2026-10-06 fused grain partisi — CANLI PASS (G2 dahil):** yoğun medyan 25.5/24.4/41.1 ms (eski 173.6/213.8/272.2), statik eğim tutunma, settle enerji 1.06e-12 J/tane. Kaynak notu: Alt adım başına 1 dispatch
  (ping-pong + nesil damgalı hash), 24-temas Gershgorin + doğruluk CFL (yoğun profilde
  beklenen ~204 dispatch/kare, eski 3471), Cundall–Strack + EPSD2 yuvarlanma yayı geçmişi,
  tane kütlesi = yığın yoğunluğu/packing × küre. Vulkan descriptor set reuse ortak backend'de.
  Kabul sırası `NEXT_BUILD_CHECKS.md` en üst; ayrıntı `MATTER_GRAIN_GPU_RUNTIME.md` son bölüm.
  Aşağıdaki "statik history açık" maddesi bu partiyle kaynakta kapandı, canlıda değil.
- **Temel solver açık:** kalıcı static friction/contact history, uzun yerleşme/
  repose, cache/scrub/resume kabulü, solver aday kıyası. G1 tamamlandı değil.
- Wet/thermal/heat coupling ve tam GPU solver→render residency açık; frame sonunda
  host publication sürer. CPU H1-R21/21 PASS üretim GPU parity anlamına gelmez.
- Legacy MPM G2 kuru8 s settle4/4 RED; dt RMS%27.28 / COM%19.08 fark sürer.
  C5 uzun havuz, C6 pressure PDE/drag/buoyancy ve production C7 ayrıca açık.
- Fog/render ajanın mevcut çalışması korunur. Yeni UI/profil göçü bu pakette yok.

Plan: [BIRLESIK_MADDE_DOMAIN_TASARIMI.md](BIRLESIK_MADDE_DOMAIN_TASARIMI.md).
Runtime/kabul: [MATTER_GRAIN_GPU_RUNTIME.md](MATTER_GRAIN_GPU_RUNTIME.md).
CPU referans: [MATTER_H1_GRAIN_CONTACT.md](MATTER_H1_GRAIN_CONTACT.md).

Yoğun göz testi hazır: `H1_Dense_Grain_Lab`/`H1_Dense_Sand_Stream`, frame0 paused,
source açık,16384 limit. Gerçek akışta16384/4.5 s görüldü; save yok.
Voxel.10/.2 kg/r.025/kn200k aynı-profile medyan1024/4096/16384=173.6/213.8/272.2 ms;
0.5 s25 örnek, render FPS değil. [Kurulum ve tablo](MATTER_DENSE_GRAIN_SETUP.md).
