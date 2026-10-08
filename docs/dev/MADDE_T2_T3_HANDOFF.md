# Madde T1/T3 kabulü ve T2 devam noktası

> **Durum:** AKTİF — 2026-10-08. Kullanıcı build'i başarılı; T1/T3 hızlı dış IPC kabulü PASS. T2 kısmi, tam T3 kabulü açık.

Kanonik madde planı: [MADDE_TIPLERI_TASARIMI.md](MADDE_TIPLERI_TASARIMI.md).
Ana domain planı: [BIRLESIK_MADDE_DOMAIN_TASARIMI.md](BIRLESIK_MADDE_DOMAIN_TASARIMI.md).
Kanıt: [madde_t1_t3_live_2026-10-08.json](madde_t1_t3_live_2026-10-08.json).

## Bu oturumda doğrulanan

Kullanıcı C++ build aldığını ve uygulamanın açık olduğunu bildirdi. Ajan build
almadı veya uygulama başlatmadı. Testler ayrı Python process'inde
`scripts/test/rt_ipc.py` ile çalıştı; sandbox'ta Windows error 5 sonrası
named-pipe erişimi için escalation kullanıldı.

| Kabul | Sonuç | Kapsam |
|---|---|---|
| `rt_test_substance_profiles_ipc.py` | PASS | 21 yerleşik, kategori, türetme, override, kalıtım, Revert, hatalı yamanın reddi, referans varken silmenin reddi; Wax `5e-6`, granular solver hints kontrolü |
| `rt_test_domain_substance_ipc.py` | PASS | Water/Honey/Chocolate/Mud/Sand/Gravel/Soil/Wax seçimi, eski girdilerin mutasyonsuz reddi, madde editinden sonra hemen fizik readback'i, numerical tuning korunumu, yeniden seçimde hints, referans koruması |
| Son temizlik sorgusu | PASS | Geçici T1/T3 domain, madde ve collider kalmadı |

Simülasyon adımlanmadı, proje dosyası kaydedilmedi. Authoring işlemleri mevcut
mekanizma gereği simülasyon cache'ini geçersiz kılabilir; bu test fiziksel
hareket, cache replay veya görsel kalite kabulü değildir.

## Kapanan kaynak partisi

- `SubstanceLibrary` ve `FluidDomainSubstance` proje/filters kayıtları tamam.
- FluidPreset/FluidChemistryPreset yerine domain `default_substance` kullanır.
  Fizik madde tablosundan, sayısal ayarlar seçimde uygulanan solver hints'ten gelir.
- UI/Python/IPC aynı core yoluna bağlı; eski fizik setter'ları açık hata verir.
- Node testleri domain fizik alanı yerine `viscosity_wall_slip` kullanır.
- Eski granular preset testi kaldırıldı; madde sözleşmesine taşındı.
- Descriptor/audit ve panel alan denetimi yeni sahipliği doğrular; kaynak
  kontrolleri geçti. Scriptlerin Release kopyaları eşitlendi.
- Göç toleransı küçük ν farklarını yutmaz. Madde editinden sonra fizik/kimya
  aynaları hemen yenilenir; numerical hints tekrar uygulanmaz.

## Tam kapanıştan önce kalan kabul

Sıra [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md)'de. Hızlı testleri tekrar
istemek gerekmez; yeni kaynak değişikliği veya regresyon varsa tekrar çalışır.

1. Türetilmiş maddeli proje save/open ve yeni projede kütüphane izolasyonu.
2. Eski Honey/Lava/Wet Sand/Cohesive Soil/Molten Plastic ve özel ayarlı domain
   göçü; aynı fizik, numerical tuning ve tekrar yüklemede tek madde.
3. Panel görsel kontrolü; node opt-in/Clear Overrides geri alma kabulü.
4. Wax/MSF `5e-3` → `5e-6` viskozite değişimi, termal zincir ve rheology görseli.
5. Önceki H1 hareketli collider/suite regresyonları gerektiğinde.

## Kodda tam kaldığımız yer: T2 sonraki kaynak partisi

T3'ü yeniden yazma. Tasarım §6a kalan T2'yi ayrı parti olarak tanımlar.

- Kaynağın `initial_constitutive_model` authoring alanı ve binding
  `SubstancePhase`/Constitutive seçimi hâlâ var. Başlıca girişler:
  `ParticleSimulation.h`, `MatterDomainSources.inl`, `RtApiFluid.cpp`,
  `MatterModelControls.cpp`, domain paneli, iki serializer ve API bindings.
- ✔ T2a (2026-10-08, kaynak): `FluidThermalLiquid.cpp` `updateThermalFreeze` donma/erime
  eşiğini parçacığın maddesinden okur (eritilebilir → `melt_kelvin`, eritilemez → donmaz,
  etiketsiz → domain değeri). Karışık Water/Wax domain'i artık ayrılır. Kabul:
  `NEXT_BUILD_CHECKS.md` 2–5. `buildThermalViscosityField` ν(T) eğrisi hâlâ domain
  eşiğini kullanıyor (açık).
- ★ Karar bekleniyor: 263 K'de "doğuşta katı" kuralı, desteksiz donmuş parçacığın
  asılı kalmasını gerektirir (bkz. `NEXT_BUILD_CHECKS.md` açık T2).
- ✔ Karar (2026-10-08, kullanıcı): **Ayrı `Ice` maddesi.** Ice, Water ile erime eşiği
  (273.15 K) ve gizli ısı üzerinden bağlanır. Mevcut Ice preset'i (`MaterialStateField.cpp`)
  korunur; katı davranış parametreleri (sürtünme, E/ν, kırılma) henüz YOK ve kaynaksız
  uydurulmayacak. Açık: katı buz granül (Drucker-Prager) mi, rijit gövde mi olarak
  çözülecek — bu karar kendi ölçümüyle verilir.
- ✔ Karar (2026-10-08, kullanıcı): **Phase türetilir.** Kullanıcı seçimi olmaktan çıkar;
  faz = f(sıcaklık, melt_kelvin). Binding Phase combo'su ve IPC/Python `phase` parametresi kalkar.
- ★ Bulgu: binding `phase == Solid` bugün `FluidDomainStep.inl:284` içinde ızgara katı maskesine
  çevriliyor (akışı engelleyen İKİNCİ katı yolu) ve `MatterGrainParams.cpp:262`'de granül
  taşıyıcı seçiminde kullanılıyor. Yani "Phase kalkar" bir silme değil: katı maske tek yola
  (donmuş parçacıklar) indirilmeli. Açık: eski sahnelerdeki katı bağlamaların göçü
  (doğuşta donmuş mu, sıvıya mı çevrilsin, yoksa reddedilsin?). Karar bekleniyor.
- ★ Söküm kapsamı: `initial_constitutive_model` ~16 yerde (RtApi.h, RtApiFluid, RtIpc, RtPython,
  descriptor'lar, SceneSerializer, ProjectManager, UI, scene_data hash, ParticleSimulation.h,
  MatterDomainSources.inl). Binding `constitutive_model` ayrıca ~12 yerde. Derlemeden
  doğrulanamayacağı için tek partide yapılacak; descriptor'lar `gen_ipc_descriptors.py` ile yenilenir.
- ✔ Söküm partisi (2026-10-08, kaynak): Initial Model ve bağlama `constitutive_model` kalktı; doğum modeli
  yalnızca maddeden. Bağlama `phase` = statik katı blok işareti (sıvı fazı sıcaklıktan türer). Hata B
  düzeltildi (eritilemeyen madde donmaz). Kabul adımları: `NEXT_BUILD_CHECKS.md`.
  Eski katı bağlama göçü: statik blok tag yolunda kalır (donmuş bayrağına bağlanmaz — termal zincir
  kapalıyken bayrak silinir ve eritilemeyen parça ilk adımda erir; kontrol sonucu).
- ★ Faz yazıcısı: `frozen` bayrağı zaten tek yazıcı (`updateThermalFreeze`). `constitutive_model`
  ise doğumda birkaç yerden yazılıyor (`MatterDomainSources.inl`, `MatterDomainSynchronization.inl`).
  Bunların tek çözücüde toplanması, faz routing'i değiştiği için T2b-3 ile birlikte yapılır.
- ★ Maliyet: `MatterGpuPartition.cpp` her dispatch'te tüm model dizisini (4 B/parçacık) yükler.
  Revizyon sayacı ve yalnız değişimde yükleme ölçümden sonra eklenecek (T2b-2).
- Frozen bayrağı ile binding solid mask iki ayrı yol; tek `madde + sıcaklık →
  hal/model` core çözümü ve tek bayrak yazarı henüz uygulanmadı.
- Ice→Water yükleme göçü, soğuk başlangıç sıcaklığının korunması ve ilgili
  testler açık. T2 kabulü: aynı madde 263 K'de katı doğar, ısınınca sıvıya geçer;
  UI/Python/IPC/serializer aynı hizmet ve hata semantiğini paylaşır.
- `fluid_flammable`, `fluid_extinguishing` vb. domain kimya aynaları hâlâ
  saklanıyor; seçme/edit sırasında eşitleniyor. Sökmek gerekiyorsa explicit
  combustion enable kontrollerini madde fiziğiyle karıştırmadan ele al.
- Elastik MPM karma yol, tam gaz/granül bağlantısı ve uzman Gas/Liquid
  domain'lerini kaldırma bu hızlı kabul ile tamamlanmış sayılmaz (T4/T5/T6).

Yeni feature logic 2000 satır üstü dosyalara eklenmez; odaklı yeni core modülü,
bu dosyalarda yalnız entegrasyon çağrıları. Build kullanıcıya aittir.
