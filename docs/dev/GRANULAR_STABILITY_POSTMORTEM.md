# Granüler kararlılık — kök neden postmortem'i (2026-08-16)

> **Durum:** REFERANS — Dört kök neden ve iki backend arasındaki aşama sözleşmesi.
> Granüler çözücüye dokunmadan önce okunmalı.

Bildirilen belirti: **Young modülü ~1000'in altındayken parçacıklar bir süre
normal davranıyor, sonra biriken enerji tek noktada boşalıp onları savuruyor.**
Bu notta dördü de ölçülmüş dört ayrı kök neden var; hiçbiri diğerinin tekrarı
değil ve üçü "makul görünen" kategorisindeydi.

---

## 1. ★★★ Alt-adım yalnızca dalga CFL'ine bakıyordu

`elasticStepInfo` substep sayısını tek bir ölçütten türetiyordu:

```
stable_young = rho * (0.35 * h / dt)^2
substeps     = ceil(sqrt(E / min(E, stable_young)))
```

Bu ölçüt E **arttıkça** daha çok substep ister. Sonuç: E ≤ `rho*(0.35h/dt)²`
olan her değer **tek substep** alır — h=0.1, 24 fps için eşik **≈1129 Pa**.
Bildirilen "1000'in altı" bir sezgi değil, bu formülün kendisiydi.

Ama gerilme çekirdeği deformasyon gradyanını `F ← (I + dt·C)·F` ile integre
ediyor ve bu **birinci mertebe üstel harita**, kendi CFL'i var: `dt·‖C‖ ≪ 1`.
Bu ölçütte E **hiç geçmiyor**. Yumuşak malzeme tam kare dt'siyle, `dt·‖C‖ ~ 1`
ile integre ediliyordu. Hata çarpımsal olduğu için hemen patlamıyor: kare kare
birikiyor, sonra boşalıyor.

**Çare:** ikinci bir limb — gerinim-hızı CFL'i, `measureLoad()` ile bir önceki
karenin affine alanından ölçülen `‖C‖` üzerinden. `wave_substeps` ve
`strain_substeps` ayrı raporlanır ki hangisinin istediği tahmin edilmesin.

★ Host ölçümü **tanımı gereği bir kare geç** — C ancak G2P'den sonra host'ta
var. Bu yüzden shader kendi clamp'ini taşır (bkz. 2). İkisi de tek başına
yetersiz.

## 2. ★★★★★ `det(F)` reset'i bir "pop" makinesiydi

Canlı IPC ölçümü (E=1000, 240 adım): **22 örnekte** `det(F)` reset olayı, tek
karede 1167 parçacığa kadar. Kontrol koşusu E=2e5: **sıfır**.

Sebep sanılanın aksine biriken sıkışma değil, çekirdeğin **sonundaki
gerilme→gerinim geri-projeksiyonuydu**. Saklanacak gerinim `stress/E`; E
düştükçe sınırsız büyür. Ölçülen: 1 kPa yığın kendi 26 kPa overburden'ını
taşıyor, K=667 Pa ⇒ istenen hacimsel gerinim **≈ -39**. `I+eps` negatif
determinantlı çıkıyor, *sonraki* adımın kinematik kontrolü parçacığı geçersiz
sayıp gerilmesini **tek adımda sıfıra** döküyor. Basamak fonksiyonu boşalması —
gözle "kabarcık patlaması".

**Çare:** bir deformasyon gradyanı yalnızca **geri kazanılabilir** gerinim
taşıyabilir. Fazlası kalıcıdır (= sıkışma) ve zaten parçacık konumlarında yaşar.
Saklanan gerinim Frobenius `kGranularMaxStoredStrain = 0.5`'te kapanır, fazlası
plastik olur. `S` zaten return-map'lenmiş halde **dokunulmadan** yazılır, yani
boşaltılacak bir şey kalmaz. 0.5, `det(I+eps) ≥ 0.125` garantiler.

★★ Teşhisin ayırt edici sorusu **zamanda çakışmaydı, büyüklük değil**: patlama
anında `invalid` mi `strain_limited` mi arttı, yoksa ikisi de düz mü kaldı.

## 3. ★★★★★ CPU'da FLIP granülde kapatılmamıştı

Belirti: aynı değerlerde GPU esneyip hacmini korurken CPU hacim kaybedip
çöküyor, **"Young hiç çalışmıyor" gibi**.

FLIP `v_yeni = v_eski + (grid_post − grid_pre)` kurar. Snapshot P2G'den hemen
sonra alınır — yani elastik gerilme diverjansı **zaten `pre`'nin içinde**.
Granülde basınç projeksiyonu da yok, dolayısıyla `post ≈ pre` ve fark gerilmeyi
**tam olarak geri çıkarır**. Sand'in `flip_blend = 0.92`'siyle gerilme yalnızca
PIC payıyla, **0.08 ağırlıkla** ulaşıyordu — efektif olarak E/12.5.

GPU bunu baştan beri kapatıyordu (`has_flip = !granular_enabled`). CPU portu
gerilme çekirdeklerini birebir kopyaladı ve **aşamayı** kaçırdı.

★★★ **Bu maddenin dersi bütün notun en genellenebilir olanı:** iki backend
yalnızca her aşamanın *matematiği* üzerinde değil, **hangi aşamaların var
olduğu** üzerinde de anlaşmak zorunda. Matematiği doğru kopyalamak yetmez —
nitekim burada gerilme baştan sona **doğru hesaplanıyordu**, fark etmesi zor
olan yarısı buydu.

Aynı aileden ikinci kırık: Sand preseti Drucker–Prager öncesinden kalma
`internal_friction = 4.0`'ı hâlâ taşıyor; GPU G2P granülde sıfırlıyor, CPU
uyguluyordu.

## 4. ★★ Kararlılık ≠ geçerlilik

Bir malzeme mükemmel kararlı olup **kendi ağırlığını taşıyamayacak kadar
yumuşak** olabilir. Bunun çaresi substep değildir; hiçbir zaman adım küçültmekle
düzelmez. `measureLoad()` gerçek parçacık kolonundan (domain kutusundan değil —
ince tabaka tutan yüksek domain ince tabakanın yükünü taşır) `rho·g·h` ölçer ve
`kGranularStiffnessLoadRatio = 10` ile karşılaştırır.

★ Eşik **gerinim cinsinden** tanımlıdır (%10), Pa cinsinden değil. Pa olsaydı
her sahnede yeniden ayarlanması gerekir ve sessizce anlamını yitirirdi.

---

## İki backend arasındaki aşama sözleşmesi

Granülde GPU'nun özel davrandığı **her** nokta ve CPU karşılığı. Yeni bir aşama
eklenirse bu tablo büyümeli.

| Aşama | Granülde davranış | CPU |
|---|---|---|
| Gerilme diverjansı | P2G'ye, **normalizasyondan önce** | `APICFluidSolver.cpp:664` |
| `internal_friction` | **0** | `:834` |
| Advection | **Lagrange** (parçacığın kendi hızı) | `:2246` |
| Reseed | **kapalı** (kütle + constitutive geçmişi yok eder) | `:2569` |
| FLIP | **kapalı** (bkz. 3) | `:3119` |
| Basınç projeksiyonu | **atlanır** (malzeme sıkıştırılabilir) | `:3210` |
| Constitutive + settle | G2P **sonrası**, cihaz yapmadıysa | `:3256` |

`GranularConstitutive.h` **referanstır**, `sim_fluid_granular_stress_update.comp`
**porttur**. Ayrıldıklarında shader yanlıştır. Push constant bloğu ile
`StressUpdateParams` bilerek alan-alan aynıdır.

## Ölçülebilir hale gelen şeyler

Sessiz mekanizma bu turun asıl düşmanıydı; hepsi artık UI + IPC + Python'da:

- `granular_wave_substeps` / `granular_strain_substeps` — hangi limit istedi
- `granular_strain_limited` — `dt·C` clamp'lendi: adım **hayatta, doğru değil**
- `granular_compaction_capped` — kalıcı sıkışma. **Hata satırı değil**; yumuşak
  malzemede sürekli görünmesi beklenir. `det(F)` reset'inin yok ettiği ayrım
  buydu: "yumuşak ama geçerli" ile "ıraksıyor".
- `granular_overburden_pressure` / `granular_young_modulus_for_load` /
  `granular_stiffness_below_load`

## Yan bulgu: substep tavanı

Ölçüm: h=0.05 + 24 fps'de E=2e5 **27** substep istiyor. Eski tavan 16, teslim
edilen `E_eff ≈ 72 kPa` — yazılan değer çalışmıyordu. Varsayılan **32**.

## Açık

`fluid.set_param` ile `granular_enabled=true` grid domain'e inmiyor gibi
görünüyor (iki kez geri okuma `False`). Eski binary ihtimali var, rebuild
sonrası doğrulanmalı. `fluid.list_domains`'in rheology'yi eski editör aynasından
okuması ayrı bir hataydı ve düzeltildi.
