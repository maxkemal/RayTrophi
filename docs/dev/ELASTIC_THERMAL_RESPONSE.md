# Elastik MPM: malzemeye ait sıcaklık tepkisi

## Ortak yasa ve authoring

Mevcut sıcaklık/yanma yumuşama scalar'ı s∈[0,1] korunur. E_eff=max(1,E*s).
Üç yeni madde alanı, diğer tepkileri ayrı ayrı kontrol eder:

| Alan | Aralık | Varsayılan | Etki |
|---|---|---|---|
| elastic_thermal_viscosity_power | 0..4 | 1 | η_eff=η*s^p |
| elastic_thermal_yield_power | 0..4 | 0 | ε_y,eff=ε_y*s^p |
| elastic_thermal_relaxation_power | 0..4 | 0 | τ_eff=τ*s^p |

p0 katsayıyı sabit tutar, p1 mevcut s'yi izler; daha büyük p sıcak durumda daha
güçlü azalma verir. Başlangıç akma gerinimi veya τ0 ise ilgili terim kapalı kalır.
Etkin açık akma/gevşeme katsayıları 1e-6 tabanına sahiptir; sıcak durumda sıfıra
inmeleri yanlışlıkla terimi kapatmaz. Sayısal viskozite ve dalga kararlılık
tavanları korunur; alt adımlar soğuk referansla muhafazakâr olarak hesaplanır.

Varsayılanlar önceki davranışı korur: yalnız E ve Kelvin–Voigt katkı s ile
azalır; akma ve Maxwell süresi sabittir. Softening kapalıysa s1 ve tüm güçler
nötrdür. Bu bir Arrhenius/WLF fit'i veya yeni sıvı yasası değildir.

Alanlar SubstanceLibrary'nin ortak Float şemasındadır. UI madde editörü,
`rt.substance` ve `substance.set` aynı doğrulama/kalıtım/transactional patch yolunu
kullanır. Builtin doğrudan değişmez; yeni tarif için türetilmiş madde kullanılır.
Alanlar proje override'larında normal diğer madde alanları gibi saklanır.

## Sakız başlangıç tarifi

Chewing Gum için artık dayanım0.2→0.02; üç güç de1. Soğuk değerler aynı kalır.
310K merkez ve15K aralıkta, 317.5K üzerinde hedef s0.02:

- E200000→4000Pa;
- Kelvin–Voigt η30→0.6Pa·s, kararlılık sınırından önce;
- akma gerinimi0.5→0.01;
- Maxwell gevşeme süresi1.5→0.03s.

Bu, ölçülmüş sakız verisi değil, daha hızlı sıcak akış gösteren authoring
başlangıcıdır. Jelatin, hamur ve çamurun varsayılan güçleri değiştirilmedi;
aynı sıcaklık tepkisi hepsine zorlanmadı. Jelatinin mevcut erime/fusion yolu
korunur. Sakız hâlâ viskoelastoplastik MPM'dir; sıvıya faz dönüşümü veya yeni
damage/kopma tarifi eklenmedi. Mevcut sakız fracture eşiği kapalı kalır.

## Telemetry

`fluid.matter_models.material_resolution.carriers[].elastic_thermal_response`
senkronize softening örnek sayısı ve min/max scalar'daki katsayıları raporlar.
Kaynak ortak seçilmiş MPM malzemesidir; her madde bağımsız çözülüyormuş gibi
raporlanmaz. Değerler stability cap ve damage öncesidir; GPU gerilme ölçümü
değildir. UI'de model/transport hücresinin tooltip'inde görünür.

Eski `granular_effective_young_modulus` alanı karma yolda baz/sizing değerini
taşır; sıcaklık sonrası E ölçümü gibi kullanılmamalıdır. Yeni termal yanıt
alanı bu ayrımı açıklar; eski alanın anlamı sessizce değiştirilmedi.

## Kullanıcı derlemesi ve kabul

**Uygulama ve stress shader'ları birlikte yeniden derlenmeli.** Host stress
sabitleri112→124byte, indexed Matter116→128byte. C++ scalar struct'ta alignas16
yoktur; tail padding lane'i132byte'a taşımamalı. Offset/size static_assert'ları
ve Vulkan pipeline boyutları birlikte güncellendi.

En az `sim_fluid_granular_stress_update.spv` ve
`sim_matter_stress_update.spv` güncel olmalı. Kullanıcının mevcut
`RayTrophiStudio/source/shaders/compile_sim_shaders.bat` akışı kullanılabilir.
Codex bu komutu, uygulama derlemesini veya başka shader compilation çalıştırmadı.

- C++ saf regresyon: `scripts/test/elastic_thermal_response_test.cpp`,
  Math/Vec3.cpp ile kullanıcı tarafından derlenip çalıştırılır.
- Statik: `check_elastic_thermal_response_contracts.py`,
  `check_matter_gpu_contracts.py`, `check_matter_mpm_softening_contracts.py`.
- Harici IPC: `rt_test_substance_profiles_ipc.py`,
  `rt_test_material_resolution_ipc.py "Domain name"`.
- Aynı geometri/yük/temas ayarıyla 293K ve320K parçacıkları karşılaştır.
  Sakız sıcakken daha hızlı kalıcı deformasyon/gevşeme göstermeli; soğuk kontrol
  aynı kalmalı. Enerji, kütle, h/dt yakınsaması ve CPU/GPU sayısal karşılaştırma
  bu kaynağa dayalı kontrollerden ayrı kabul ölçümleridir.

Önceki canlı ölçüm eski tarifi doğrulamıştı: 66240 parçacık293→504K, s0.2.
Yeni s0.02 ve güç1 tarifi o eski ölçümle doğrulanmış sayılmaz.

## Birleşik Matter planının durumu

Tamamlanan kaynak işleri: ortak birth/fusion seçimi, malzeme raporu,
Elastic/mixed softening bağlantısı, bu parametreli termal tepki, preset
tutarsızlıkları/kopya temizliği ve seed kutusu yükleme düzeltmesi.

Uygulaması kalan ayrı işler: Seed Region viewport gizmosu ve ortak tarif
düzenleme/undo servisi; çikolata için akma gerilmeli sıvı yasası; sıvılarda
ölçümlü μ(T); Oil çözünürlük atlamasının doğruluk politikası; tanımsız durumun
üretiminde ortak ret semantiği. Bağımsız çok malzemeli MPM de mevcut domain
parametresi sınırının ötesinde ayrı çözücü işidir. Bunlar bu termal değişiklikle
tamamlanmış sayılmaz. Canlı kabul/derleme doğrulaması da beklemektedir.
