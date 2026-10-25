# Kar: Stomakhin 2013 kurucu modeli (MPM)

> **Durum:** AKTİF — 2026-10-23; 1. parti derlendi ve canlı ölçüldü; 2. parti (§6) kaynakta, derlenmedi.

## Devir notu (2026-10-23, 2. parti)

- 1. parti canlı: sayaçlar ölçülüyor, serbest düşüşte J_P sabit. Ama 7 m/s
  çarpmada blok eşit sıkıştı (ortalama J_P 0.74, min 0.395 > buz), hiç
  gevşemedi, esnedi; substep 9 → 184 (430 ms/adım) ve kalıcı.
- 2. parti (§6): sertleşme tavanı (`granular_snow_max_hardening`, varsayılan 5),
  J_P tabanı = ρ/ρ_buz, substep tavana göre SABİT, başlangıç yoğunluk dağılımı
  (`granular_snow_heterogeneity` 0.05). Derleme: C++ + shader.
- Açık soru: düz levha + düz zemin çekme üretmeyebilir; NEXT_BUILD_CHECKS 7.
  madde küre/eğim ile ayırır. Sonraki aday: granular hatta FLIP (§5).

## Devir notu (2026-10-23, 1. parti)

- **Ne yazıldı:** Snow artık Drucker-Prager'a giydirilmiş Stomakhin sayıları
  değil, Stomakhin et al. 2013'ün kendi yasası: tekil değer kırpma
  [1-θc, 1+θs], F_E/F_P ayrımı (yalnız J_P tutulur), her modül
  exp(ξ(1-J_P)) ile ölçeklenir, gerilme Kirchhoff τ. CPU referansı
  `GranularConstitutive.h::snowStressUpdateParticle`, port
  `sim_fluid_granular_stress_update.glsl::snowStressUpdate`. Madde alanı
  `granular_snow_plasticity` (+ `_critical_compression`, `_critical_stretch`,
  `_hardening`); Snow ön ayarı açık.
- **Aynı partide iki bağımsız kök düzeltildi** (ikisi de kar için ölçülen
  "erimiş çikolata"ya katkı veriyordu, aşağıda).
- **Derlemede gereken:** C++ ve `compile_shaders.bat` (stress_update push
  constant 64→96 bayt; eski .spv ile çalışırsa `matter_lane` pad'den okunur).
- **Sıradaki:** `scripts/ipc/Probe-SnowStomakhin.ps1` canlı; sonra kartopu,
  sıkıştırma, yamaç bırakma sahneleri (aşağıda §4).

## 1. Neden ayrı dal

Ölçüm (2026-10-23, canlı sahne: 2.25 m kutu, ~1 m kar, 50 bin parçacık, küp
collider): DP sınırı q ≤ c + p·tanφ, c = 500 Pa. Serbest kolonda σ(1 − tan30°/3)
> 500 Pa → **h > ~16 cm** akar. 1 m blok her yerde akma sınırında, ve akış
mükemmel plastik: kesme/çekme malzemeyi zayıflatmıyor (dilatancy 0, yumuşama
yalnız sıkışmada), dolayısıyla deformasyon lokalize olmuyor; çatlaklar
rebonding (0.5/s) ile basınç altında kapanıyordu.

Stomakhin'de topaklanmayı ve kırılmayı yapan tek şey asimetri:
sıkışan bölge J_P < 1 → sertleşir (topak); gerilen/kesilen bölge J_P > 1 →
yumuşar (çatlak lokalize olur, parçalar ayrılır). Bu bir yasa farkı, sayı farkı
değil.

## 2. Durum dizilerinin bu daldaki anlamı

| Dizi | Snow dalında |
|---|---|
| `deformation_col*` | tam F_E (DP'deki projekte küçük gerinme değil) |
| `plastic_volume` | 1/J_P (DP sıkıştırmayla AYNI yön: > 1 sıkışmış) |
| `stress_*` | Kirchhoff τ; P2G dinlenme hacmiyle çarpar → V0·τ |
| `damage` | 1 − min(exp(ξ(1−J_P)), 1): gevşemeyle kaybedilen sertlik |
| `fracture_history` | max(J_P − 1, 0) |
| `hardening` | Σ|ln(σ/σc)| birikmiş plastik gerinme |
| bayrak 2 (detached) | sertliğin ≥ %99'u kaybedildi (toz) |
| bayrak 64 (YENİ) | sertleşmiş E substep dalga sınırına kırpıldı |

SVD: F^T F üzerinde sabit 6 süpürmelik döngüsel Jacobi (NR 11.1); U = F V / σ,
Gram-Schmidt, üçüncü sütun çapraz çarpım (det F > 0 ve det V = +1). F_E en
fazla %25 adım + kırpılmış tekil değerlerden geldiği için σ ≳ 0.7: koşul iyi.

Substep: kar en sert parçacığa göre boyutlanır (E·max exp(ξ(1−J_P)), maddenin
kendi yoğunluğuyla). Çekirdek yine E'yi ρ(0.35h/dt)² ile kırpar ve bayrak 64'ü
kaldırır — sessizce değil.

Settle: karda basınç kapılı temas sönümü (8/s) KAPALI; çarpma enerjisini yiyip
kartopunun dağılmasını engelliyordu. Uyku açık.

## 3. Aynı partide düzeltilen iki kök

1. **Matter yolu sayaç körlüğü.** `runMatterGpuStep` cihaz durumunu her kare
   indiriyor ama hiç saymıyordu; sayım yalnız eski sıvı yolunda vardı
   (`FluidDomainStep.inl`, karışık dal `break` ile oraya ulaşmıyor). Canlıda
   4 m/s çarpma 120 adım boyunca 0 akma / 0 hasar okudu. Üç kopya sayım tek
   `Granular::countMaterialState`'e (`GranularCounters.h`) toplandı; Matter
   yolunda yalnız granular parselleri sayar. Yeni: `granular_material_measured`
   (false = ölçülmedi, sıfır değil), `granular_counted`,
   `granular_{min,mean,max}_plastic_jacobian`, `granular_hardening_capped`.
2. **Gerilme kuvveti yoğunluğu 1600'e sabitti.** Etiketsiz (seed'lenmiş =
   varsayılan madde) parselin kütlesi Snow yoğunluğuyla (400) hesaplanıyor,
   gerilme hacmi ise `matterParticleDryVolume` içinde 1600 (Sand) yedeğiyle →
   hacim 1/4, gerilme kuvveti 1/4. Tekil GPU ve CPU yolları da 1600 sabitini
   kullanıyordu. Şimdi üçü de varsayılan maddenin yoğunluğu
   (`APICSolverParams::granular_density`). ★ Sand 1600 olduğu için değişmez;
   Gravel (1750) ve Soil (1450) ~%10 farklı sertlik hisseder — bu düzeltmenin
   kendisi.

## 4. Kabul sahneleri (henüz koşulmadı)

- **Düşürme** (`Probe-SnowStomakhin.ps1`): J_P her iki yöne ayrılmalı
  (min < 0.99 VE max > 1.01). Yalnız aşağı giden J_P = sıkışıyor ama
  çatlamıyor = eski semptom.
- **Kartopu fırlatma** duvara: parçalanmalı, parçalar topak kalmalı.
- **Sıkıştırma** (piston/collider): J_P < 1 ve sertleşme; yük kalkınca geri
  esneme küçük olmalı.
- **Yamaç bırakma:** levha halinde kopma, akış değil.

## 5. Sonra listesi

- `MatterGrainMpmContact::prepare` hâlâ `matterParticleDryVolume(continuum, i)`
  çağırıyor (domain maddesi yok): etiketsiz kar parselinin temas yarıçapı
  ∛(1/4) ≈ 0.63× küçük. Yalnız DEM tane + MPM karışık domain'i etkiler.
- DP substep planlayıcısı hâlâ `kGranularDensity` ile boyutlanıyor (yalnız kar
  dalı gerçek yoğunluğu kullanıyor).
- Matter granular hattında `flip_blend` sabit 0 (saf APIC). Stomakhin %95 FLIP
  kullanır; APIC açısal momentumu koruduğu için ilk tahmin yeterli, canlıda
  dağılma enerjisi az görünürse buraya bak.

## 6. 2. parti: maliyet ve dağılma (2026-10-23)

Canlı ölçüm (87 552 parçacık, 0.1 m, ~3 m düşüş → 7 m/s):

| adım | J_P min / ort / max | substep | not |
|---|---|---|---|
| serbest düşüş | 1.000 / 1.000 / 1.000 | 9 | 92 ms/adım, kayma yok |
| çarpma karesi | 0.409 / 0.98 / 1.000 | 9 | 7646 parçacığın E'si kırpıldı |
| +1..+4 | 0.395 / 0.93→0.74 / 1.0→0.94 | 171→184 | blok baştan sona eziliyor |
| durgun | 0.395 / 0.743 / 0.880 | 184 | 430 ms/adım, kalıcı |

Okuma: dinamik basınç ρv² ≈ 19.6 kPa, bakir karın sıkışma sınırı ~E·θc ≈ 3.5 kPa
→ sıkışma dalgası bütün bloğu geçiyor; bu, modelin bu hızda doğru cevabı.
Yanlış olanlar: (a) J_P 0.395 buzdan yoğun (ρ/0.395 = 1013 kg/m³); (b) substep
geçen karenin en sert parçacığına göre boyutlanıyordu, çarpma karesi bakir
sayıyla koştu ve kırpılan E aşırı sıkışmayı büyüttü; (c) sertleşme sınırsızdı
(e^6 ≈ 400× → √400 = 20× substep, kalıcı); (d) tekdüze seed'de çatlağın
başlayacağı zayıf nokta yok.

Değişiklik:
- J_P ≥ ρ/917 (`snowMinPlasticJacobian`), push constant'taki pad'den.
- exp(ξ(1−J_P)) ≤ `granular_snow_max_hardening` (yalnız sertleşme tarafı;
  yumuşama sınırsız). Yaygın açık-kaynak MPM kar örnekleri de bu çarpanı
  kırpar; makalede yok — bu bir maliyet düğmesi.
- Substep tavana göre sabit (`snowSizingYoungModulus`); `maxSnowHardening`
  söküldü.
- Bakir parsel (pv == 1, birikmiş plastik 0) ilk görüldüğünde particle_id
  hash'inden J_P = e^{s·u} çeker (`assignSnowHeterogeneity`, FluidDomainStep
  yumuşama bloğu; her backend sonra host'tan yükler).

Not: J_P yalnız NET hacmi söyler. Ezilen kar yana yayılırken eksenel sıkışma
(σ < 1−θc) ve yanal çekme (σ > 1+θs) aynı adımda kırpılır; çarpım net
sıkışmaysa J_P düşer. Düz levha düz zemine çarptığında yanal kuşatma yüzünden
net gevşeme hiç oluşmayabilir — bu modelin özelliği, bug değil. Ayırt etmek
için küre/eğim testi (NEXT_BUILD_CHECKS 7).

### 6b. Maliyetin dökümü (1. parti exe, aynı gün)

Çarpmada adım 131 → ~420 ms (dt 1/60). `perf.gpu_kernel_timings`: 566 substep,
1299 ms → substep başına ~2.3 ms; `sim_matter_p2g` 940 ms (%72, atomik
scatter, 87k parçacık), `sim_matter_stress_p2g` 168 ms. Yani maliyet substep
SAYISINDA; tavan (§6) onu ~9·√5'e sabitler. P2G'nin kendisini ucuzlatmak ayrı
iş (Sonra).

Backend kilidi: `matterGrainOwnership` domain'in madde-materyal bağlarını da
"tane istiyor" diye sayar. Bu sahnede Sand yalnız görünüm için bağlıydı
(`dem` taşıma) → DEM açık, backend Vulkan'a kilitli, 0 tane. Kural değişmedi;
sebebi artık `grain_ownership.origin` ve panelde görünüyor.

## 7. 3. parti: ince voxel darbe testi ve FLIP (2026-10-23)

Ölçüm `scripts/ipc/Probe-SnowImpact.ps1` (voxel 0.025, 0.3 m blok, ~6 m/s;
çalışan exe 2. partinin C++'ı + 1. partinin .spv'si — tavan/buz tabanı kernel'de
yoktu, J_P 0.303'e indi):

| hedef | J_P ort | %50+ kayıp | toz | not |
|---|---|---|---|---|
| düz zemin | 0.68 | 13 | 0 | ölü durma (v 5.6 → 0.17), dinlenmede J_P max 1.06→1.17 |
| küre kaya | 1.73 | %67 | %49 | kırılıyor ama topak kalmıyor |

Okuma: yasa çekmede kırıyor (geometri ayrımı doğrulandı). İki kusur:
(a) dinlenme titreşimi sönmüyordu (temas sönümü kapalıydı) → θs her salınımda
aşılıp J_P mandal gibi yükseliyor; (b) PIC/APIC hız farklarını siliyor →
zeminde ölü durma, kayada parça yerine toz.

Değişiklik: settle sönümü kar için yalnız < 0.5 m/s parçacıklara
(`SettleParams::damping_max_speed`, `kSnowSettleDampingMaxSpeed`); Matter yolunda
kar için FLIP — taban P2G'de scatter ile gerilme impulsu ARASINDA kopyalanıp aynı
ağırlıkla normalize edilir (`granularFlipSnapshotWanted`, `FluidGpuP2G.inl`);
Snow `solver_flip_blend` 0.95. DP kumları PIC/APIC kalır. Tek-model granular yol
FLIP'siz kalır (Sonra).

Açık: kayada %49 toz, θs = 0.0075'in bu çözünürlükte fazla kırılgan olup
olmadığı FLIP'li ölçümden SONRA karar verilecek (parametre değil, önce dinamik).

## 8. 4. parti: homojenlik → Voronoi topakları (2026-10-23)

3. parti canlı (FLIP + yavaş sönüm derlendi): zemin 5.6 → 0.5 m/s (FLIP'siz 0.17),
kaya sonucu değişmedi (%65 parçacık %50+ yumuşamış, toz). Kullanıcının tahmini
"taneler çok homojen" test edildi: bağımsız parçacık dağılımı 0.05 ve 0.3 aynı
sonucu verdi. Sebep: P2G düğüm başına ~50 parçacığı ortalar ve bir hücredeki
bütün parçacıklar aynı ızgara hızıyla deforme olur; tek zayıf parçacık
yerelleşemez. Zayıf bölge hücreleri kapsamalı.

Değişiklik: `assignSnowHeterogeneity` artık parçacığın doğum konumunda
`snowClumpField` okur — Worley hücreleri (`granular_snow_clump_size`, ≥ 3 voxel),
hücre içi rastgele sertlik, sınır şeridi (≥ 1 voxel) gevşek. Snow dağılımı 0.1.
Beklenen: çatlak sınırlar boyunca yürür, hücreler topak kalır. Ölçülmedi.

Kullanılan test maddesi: `SnowProbe` (Snow'dan türetilmiş, heterogeneity 0.3) —
kullanıcı sahnesine kaydedilmedi; probe sahnesi `SnowTest` + `SnowTest Rock`.

## 9. 5. parti: gevşek kar basınca direnmiyordu (2026-10-23)

4. parti canlı: Voronoi yapısı atandı ama kaya sonucu aynı; hız taraması
(2.3–4.6 m/s) aynı sonuç → enerji değil. Çarpmasız durgun test: varsayılan
E'de kaya üstündeki blok 1.5 s'de dağıldı; E 7e5'te sağlam kaldı. Konsol
hesabı: 15 cm sarkma, 0.3 m derinlik → üst yüzde ~0.9 kPa çekme ≈ θs·E = 1.05 kPa
(sınırlarda ×0.37). Referans parametreler taze toz kar.

Görsel (ekran görüntüsü alındı, silindi): kaya bloğu delip içine girmiş, blok
zemine inmiş, dışı sağlam küp. Kütle merkezi 0.305 (beklenen ~0.39). Sayaçların
"toz"u ayrılmış parça değil, kayanın çevresindeki basınçsız kesme bölgesi.

Kök: sertleşme çarpanı sıkışma modülünü de çarpıyor, tabanı yoktu (J_P 3–4 →
e^-20..-30). Değişiklik: `granular_snow_min_hardening` (0.1; yaygın açık-kaynak
MPM kar örneklerindeki [0.1, 5] kırpması), `detached` = tabanda. Probe kütle
merkezini raporluyor. Ölçülmedi.

Karar bekleyen: varsayılan Snow taze toz mu (E 1.4e5) kalsın, yoksa ayrı bir
"Packed Snow" (E 7e5–1.4e6) maddesi mi eklensin.

Karar (kullanıcı, 2026-10-23): Snow taze toz kar olarak kalır (E 1.4e5, ρ 400).
Yeni yerleşik `Packed Snow`: aynı yasa ve parametreler, ρ 550, E 1e6, iletkenlik
0.45 (kartopu / rüzgâr kabuğu; sıkışmış kar 0.5–5 MPa). Maliyet: dalga hızı
√(E/ρ) Snow'un ~2.3 katı → substep ~2.3×. Etiket ad hash'i (FNV), sıra değil;
araya eklemek kayıtlı projeleri kaydırmaz. Ada bağlı iki yer güncellendi:
`MatterMaterialPreset.cpp` (görünüm), `SubstanceCatalogue.cpp` (aile).

## 10. 5. parti canlı: taban + Packed Snow (2026-10-23, exe 16:10, .spv 15:59)

`Probe-SnowImpact.ps1 -Target rock` (voxel 0.025, ~4.6 m/s; probe artık
`-Substance` alıyor — önceden domain'i her koşuda Snow'a geri çeviriyordu):

| madde | substep / ms | J_P min | kütle merkezi y | görsel |
|---|---|---|---|---|
| Snow (toz) | 80 / 54 | 0.436 | 0.165 | kaya etrafından zemine iniyor |
| Packed Snow | 182 / 104 | 0.600 | 0.270 | sağlam küp, kayaya ~12 cm oturmuş, çatlak yok |

Durgun (çarpmasız) kaya testi: Packed Snow DURUYOR (y 0.375, d90 %0);
Snow çöküyor (y 0.310, d90 %39) — taze toz için makul.

Okuma: taban delip geçmeyi kesmedi ama Packed Snow'da derinliği yarıya indirdi;
darbe temas noktasında buz yoğunluğuna kadar ezme (J_P = ρ/ρ_buz) olarak
emiliyor, ayrılma yok. Test düzeneği sınırlayıcı: düz yüzlü küp, yalnız
yerçekimi (≤ ~7 m/s), kendinden dar bir küre. Makaledeki sahne (fırlatılan küre
duvara) IPC'de kurulamıyor: seed'in şekil ve başlangıç hızı yok. Sıradaki iş bu
(seed tarifine `shape` = box|sphere ve `velocity`; panel + IPC + Python +
kayıt).

## 11. 6. parti: tek ön ayar ve panelin substep yalanı (2026-10-23)

Kullanıcı heterojenliği 0.44'e çıkarıp Fog çıkışıyla "daha doğru" buldu (sınırlar
baştan tabanda = önceden çatlamış 12 cm topaklar). Kararlar: tek `Snow`
(`Packed Snow` kaldırıldı; sıkışmış kar = türet + E/ρ yükselt). Panel yalnız
"Max Advection Substeps 12" gösteriyordu, koşu 200'dü (dalga CFL'si; 24 fps,
voxel 0.025, tavan 5). Panel ve IPC artık `planGranularSubsteps` ile tahmini,
son kareyi ve formülü gösteriyor. Maliyet kaldıraçları sırasıyla: voxel (~h⁻⁴),
tavan (√), E/ρ (√), fps (doğrusal).
