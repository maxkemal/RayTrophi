# Elastik MPM katısı — jöle, hamur, sakız

> **Durum:** AKTİF — E1+E2 CANLI 10/11 (2026-11-02 14:40): yerçekimi çekici düzeltildi (hamur 0.023, jöle 0.036 m/s, jöle sağlam); açık: hamurun ezilme izi jöleden ayırt edilmiyor (akma 0.15 ayarı).

## Neden şimdi, neden böyle

`MatterConstitutiveModel::Elastic` aylardır vardı ama yalnız bir etiketti. Karma
çözücü onu "not implemented" diye reddediyordu, bölümleme shader'ı da onu hiçbir
şeride koymadan sayıyordu. Kar dalı (Stomakhin 2013) ise tam bir sabit-korotasyonlu
MPM'di: F_E, SVD, Kirchhoff gerilmesi; hem CPU'da hem GLSL'de. Elastik katı da
aynı gerilme, ama kırpma ve sertleşme yok. Bu yüzden **yeni şerit değil, MPM
şeridinin üçüncü yasası** yazıldı.

## Model (E1)

Sabit-korotasyonlu hiperelastisite. Opsiyonel üç etkinin hepsi yalnız
**deviatorik** Hencky gerinimine (e = ln σ − tr/3) etki eder, bu yüzden hiçbiri
hacmi değiştiremez:

| Alan | Etki | Örnek |
|---|---|---|
| `elastic_young_modulus`, `elastic_poisson_ratio` | Sertlik | jöle 5 kPa, sakız 200 kPa |
| `elastic_relaxation_s` | Maxwell sünmesi: e ← e·exp(−dt/τ) | sakız sarkar, uzar |
| `elastic_yield_strain` | Hencky von Mises akma, mükemmel plastik: \|e\| ≤ ε_y | hamur çukuru tutar |
| `elastic_viscosity_pa_s` | Kelvin-Voigt: τ += η·J·(C + Cᵀ) | jölenin titremesi söner |

Hepsi 0 → tam elastik cisim. Isıyla yumuşama madde üzerindeki
`granular_softening_*` alanlarını kullanır (sakız 310 K).

Hazır maddeler: **Gelatin** (elastik + sönüm, 308 K'de erir), **Dough**
(elastoplastik), **Chewing Gum** (viskoelastik, akar). Değerler mertebe olarak
seçildi, ölçülmüş uyumlar değil.

## Nasıl bağlandı

- Madde → şerit: `resolveParticleGranularPhysics`, ilk MPM parçacığının maddesi
  Elastic ise `applyElasticSubstancePhysics` uygular. E, ν ve yoğunluk
  `granular_*` alanlarına gider, çünkü şeridin alt adımlarını boyutlandıran ve
  gerilmeyi bölen alanlar onlar. `elastic_solid` bayrağı yasayı seçer.
- GPU: `StressUpdateConstants.snow_plasticity` alanı **`mpm_law`** oldu (0 DP,
  1 kar, 2 elastik); blok 96 → 112 bayt, indeksli varyant 100 → 116. Şeritte
  `elasticStressUpdate` çalışır; oturtma (settle) atlanır, çünkü bir jölenin
  titremesini öldürür.
- Bölümleme: model 3 artık MPM şeridine (granular_index) gider.
- Karma GPU adımı: Elastic reddi kalktı; `has_granular` = MPM parçacığı (granüler
  veya elastik); alt adım boyutlandırması elastik yoğunlukla, sertleşme
  tavanı olmadan. ★ Sonuç birleştirme de MPM parçacığının hepsini kapsıyor.
  Yalnız Granular kalsaydı elastik cisim her adım hareket edip sessizce geri
  dönerdi.
- FLIP: elastik şerit kendi maddesinin karışımını alır (`elastic_flip_blend`).
  Alanın `flip_blend` değeri Default Substance'a (çoğu zaman suya) aittir.
- Kütle: katı model (granüler veya elastik) `density` ile tartılır,
  `liquid_density` ile değil.
- Viskozite açık (explicit) yazıldı: η ≤ ρh²/(6dt) ile tavanlanır.

## E1 canlı ölçümleri ve E2 (2026-11-02, yazıldı, derlenmedi)

- Açık Kelvin-Voigt salınımı: hamurda saklanan hız, gerçek hareketin 6.7 katı.
  Elastik yasa artık alt adımları viskoziteye göre (η·dt/(ρh²) ≤ 1/12) ve %50 dalga
  payıyla boyutlandırıyor.
- Ayar: Gelatin η 2 → 8 Pa·s, Dough akma 0.05 → 0.15.
- E2 yırtılma: `elastic_fracture_strain` (en büyük asal Hencky gerinimi) ve
  `elastic_damage_rate`. Hasar çekme gerilmesini ve viskoziteyi siler; basınç kalır.
  GPU'da hız DP'nin `damage_rate` push alanından okunuyor, eşik eski pad yuvasında;
  blok 112 baytta kaldı.
- Ölçü: `fluid.state_digest` → `bounds_p05/bounds_p95`.

## E2 canlı (2026-11-02, exe 14:16)

- Yırtılma çalışıyor: düşük eşikli jöle 45 cm'den 3584/3584 kopuk.
- Normal jöle (eşik 0.6) 20 cm'den kenarda ~%3 kopuyor; jöle hâlâ ~3 Hz titriyor.
- ★ Hamurun sabit 0.17 m/s'si alt adım 24 → 36 olunca büyüdü: viskozite salınımı değil,
  sabit bir bileşen (zemin hücresi adayı). Önceki partideki teşhis yanlıştı.

## Yerçekimi çekici (2026-11-02, yazıldı, derlenmedi)

Karma madde adımı yerçekimini kare başına bir kez, kare süresiyle ekliyordu. Durağan bir
elastik cisim her kare 0.41 m/s'lik bir darbe yiyor ve ≈ 0.4·g·kare_dt genlikle çınlıyordu
(kare dt'yi yarıya indirince hız da yarıya iniyor, alt adım dt'si aynıyken). MPM parçacığı
taşıyan karma Vulkan adımı artık yerçekimini her alt adımda alıyor (`MatterSubstepForces`).
Gelatin yırtılma eşiği 0.8.

CANLI (exe 14:40): hamur hızı 0.171 → 0.023, jöle 0.146 → 0.036 m/s, normal jöle kopuğu 123 → 1.
Kar/kum dumanı regresyonsuz.

## Toplam uzamayla yırtılma (2026-11-02, yazıldı, derlenmedi)

Yırtılma eskiden yalnız elastik gerinimi okuyordu; sünme ve akma onu küçük tutar. Sonuç:
yavaş çekilen sakız gerçekçi bir eşikte hiç kopmuyordu (akma 0.5'te elastik tavan
~0.41), çok küçük bir eşikte ise durağan gerinimiyle anında dağılıyordu. Artık yırtılma
gerinimi = en büyük elastik asal gerinim + `fracture_history`. Bu alan sünme ve akmanın
her adım bıraktığı en büyük asal çekme akışının birikimi. Jöle değişmez. Dough akma
eşiği 0.10.

## Sınırlar (E1)

- **Alan başına bir MPM malzemesi.** Elastik ile granüler MPM aynı alanda
  adımı tutar ("one MPM material per domain"); sessizce yanlış yasa koşmaz.
- DEM (tane) alanında elastik parçacık reddedilir.
- Yırtılma E2'de yazıldı; iyileşme (yeniden yapışma) yok. İnce yapı XPBD işi ayrı.
- Panelde elastik bölüm yalnız `default_constitutive_model = elastic` maddede
  görünür; yumuşama alanları "Granular" grubunda kalır ("Show unused").

## IPC

- Madde alanları genel tablodan geçer: `substance.get/set` ve kayıt otomatik.
- `fluid.list_domains` → `mpm_law`: `drucker_prager` / `snow` / `elastic` / `none`.
  Domain panelinde de aynı satır var.
- Test: `scripts/test/rt_test_elastic_mpm_ipc.py`, `rt_matter_smoke.py` içinde `elastic`.
