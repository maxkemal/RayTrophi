# Tek domain — simülasyon etiketleri, ilk çekirdek partisi

2026-09-28. Kullanıcı derledi; dış IPC ile ilk canlı kontroller geçti.
100.000 parçacıkta 11,5–12,7 ms CPU etiketleme maliyeti ölçüldü; optimizasyon
açık. Tam sonuçlar ve henüz çalıştırılmayan kontroller `NEXT_BUILD_CHECKS.md`
15. partide.
Ana plan: `BIRLESIK_MADDE_DOMAIN_TASARIMI.md`, Faz 2.

## Sözleşme

`FluidParticleLabels` simülasyon sonunda, whitewater adımından sonra çalışır.
Etiket `FluidParticles::flags` bit 8..11'de tutulur. Madde kimliği
`substance_tag`, termal donma bit 3, outflow bit 1 olarak ayrı kalır.
Mevcut emit / removeSwap / compact / bellek snapshot yaşam döngüsü tüm flags
sözcüğünü taşıdığı için etiket parçacıktan ayrılmaz. Yeni parçacık `unknown`
doğar; sınıflandırma tüketicilerde değil simülasyonda yapılır.

Güncel sınıflandırıcı `mass+neighborhood_v2`:

- Termal frozen biti: `frozen`.
- Diğer sıvı parçacıkları: 1,5 simülasyon vokseli yarıçapında, kendisi hariç
  en fazla 6 komşu sayılır. 0–2 komşu `spray`, 6 ve üzeri `body`.
- 3–5 komşuda önceki body/spray durumu korunur; önceki durum bilinmiyorsa body.
- Granüler domain ve solid madde bağlamaları sıvı spreyine dönüştürülmez;
  frozen olmayanları şimdilik `unknown`.
- Geçersiz koordinat veya voksel boyutu `unknown`; float→int dönüşümü sonlu
  ve int aralığı içinde olduğu doğrulandıktan sonra yapılır.
- Kalan `mass_fraction` değeri `0 < m <= 0,15` olan sıvı parseli `mist` olur.
  Mist sıvı komşuluk binlerine girmez, fog yoğunluğuna kalan kütlesi kadar
  katkı verir ve örtüşen gaz alanının hızına kütle ölçekli tepki süresiyle
  yaklaşır. Granüler/solid parseller mist olmaz; frozen önceliğini korur.

Bu, ilk önerilen komşuluk kriterinin uygulanmasıdır; fiziksel kalibrasyon
tamamlanmadı. Parçacık/voksel oranı ve reseed yoğunluğu sayımı etkiler.
Histerezis kararlılık sağlar ama eksik fiziksel kalibrasyonun yerini tutmaz.
Body/spray/frozen etiketleri çözücü davranışını değiştirmez. Mist etiketi
24. partiden itibaren yalnız örtüşen gaz hızına alt-ızgara sürüklemeyi açar.
**Etiketler görüntüyü de değiştirir (17. parti):**
görünüm anahtarı (madde, durum etiketi). Madde bir görünüm seçer: bağlama
`representation`, yoksa domain varsayılanı. Etiketin yönlendirmesi
(`fluid_label_routes`) bunu parçacık başına değiştirebilir.

- **Varsayılan tablo** (kullanıcı kararı, tasarım tablosu §4.2): spray, foam,
  bubble → splat; mist → fog; body, frozen, unknown → maddeyi izler.
  Aynı spray/foam/bubble satırları whitewater'ı da yönlendirir (20. parti).
- **`unknown` yalnızca izler.** Çözücüde ve setter'da zorunlu; ölçülmemiş bir
  parçacık başka bir şey gibi sunulamaz.
- **Tüketiciler tek kuralı paylaşır.** Level set, materyal koordinatı,
  kompozisyon, fog yoğunluğu ve splat köprüsü aynı parçacık kararıyla filtreler
  (`FluidViewPlan::viewForParticle`, `FluidViewSelection`).
- **Yüzey rebuild imzası etiket bitlerini ve tabloyu hash'ler.** Konumu değişmeden
  body'den spray'e geçen parçacık, duraklatılmış karede de yüzeyden çıkar.
- **Kaynak bayrakları (18. parti).** Etiket yönlendirmesinin hedefi KALICI
  kaynaktır. Tüketiciler `anyLiveIn(view)` false iken işi atlar: splat havuzu
  gizlenir, fog slotu gizlenir. İlk sürüm kaynağı canlıya bağlamıştı; ölçüldü:
  RT rebuild tam olarak spray 0↔>0 karelerinde tetikleniyordu.
- **Erişim:**
  - IPC `fluid.set_label_views {domain, routes:{label:route}, reset}`, Python
    `rt.fluid.set_label_views`.
  - `fluid.get` şunları döndürür: `label_routes`, `views[].labels`,
    `views[].particles`, `hidden_particles`.
  - Panel: Output → Liquid Display → Particle State Views.

Seyrek hash komşuluk binleri domain hacmi yerine parçacık sayısıyla büyür.
Bin kurulumu CPU'dadır; sınıflandırma OpenMP varsa paraleldir. Son geçişin
süresi, işlenen parçacık sayısı ve etiketi değişen sayısı raporlanır.
Vulkan simülasyon yolu da mevcut host parçacık verisini kullanır; yeni GPU
readback veya shader eklenmedi. Ek CPU maliyeti canlı ölçülmelidir.

## Whitewater ve korunum

Mevcut `FoamType` → ortak `ParticleLabel` eşlemesi tek çekirdektedir.
Whitewater'ın spray/foam/bubble değerlerini zaten simülasyon yazar; rapor
yeniden sınıflandırmaz. Bu küme hâlâ ikincil ve kütlesizdir.
APIC kütlesine eklenmez; ana parçacık sayısıyla toplanıp fiziksel sıvı
miktarı olarak sunulmaz.

**20. parti (Faz 2-W / W1):** Whitewater'ın nerede çizileceğine aynı
`fluid_label_routes` karar verir (`FluidViewPlan::viewForWhitewater`); ayrı
`FoamRenderMode` söküldü. Whitewater'ın maddesi yok, Follow untagged girişi
izler. sdf = yüzey hacminde beyaz ortam, fog = fog yoğunluğuna ekleme.
**Depo birleşmesi İPTAL** (kullanıcı kararı): whitewater çözülemeyen ölçeğin
yer tutucusu, fizik değil; aynı depoya girerse "spray" etiketi bazen kütleli
bazen kütlesiz olur. Gerçek karşılıklar: kabarcık için iki fazlı akış
(birleşik madde domain'i), spray için çözünürlük. Ayrıntı: tasarım notu §8c.

## UI, Python ve IPC

- UI: Physics Domain → Measure → Simulation & Collision Statistics →
  Particle State Labels.
- Python: `rt.fluid.get(domain)["particle_labels"]` ve
  `rt.fluid.list_domains()` girdilerindeki aynı alan.
- IPC: `fluid.get {domain: "..."}` ve `fluid.list_domains` girdilerindeki
  `particle_labels`.

Tümü `inspectParticleLabels` sonucunu okur. Python ve IPC JSON alanları da
ortak dönüştürücüden gelir. Salt okunur çıktıdır; etiket atama setter'ı yoktur.
Domain bulunamazsa mevcut API hatası döner. Gaz veya canlı durum yoksa
`available:false`; sıfır sayaçlar ölçüm yapıldığı anlamına gelmez.

`primary` ve `secondary` her zaman şu anahtarları içerir:
`unknown`, `body`, `spray`, `foam`, `bubble`, `mist`, `frozen`.

```text
sum(primary)   == primary_particles == fluid.get.particle_count
sum(secondary) == secondary_particles
primary_complete == available && primary_particles > 0 && primary.unknown == 0
secondary_affects_mass == false
mist_generation == true
mist_mass_fraction_max == 0.15
classifier == "mass+neighborhood_v2"
render_routing == "substance+label"
```

`last_step` son sınıflandırmanın ölçümüdür; sonradan emitter eklenmesi/silinmesi
veya cache okunması halinde güncel parçacık sayısıyla aynı olmak zorunda değildir.
Güncel sayaçlar her sorguda mevcut diziden okunur, eski sayım döndürülmez.
Disk cache v6 (2026-09-28, 16. parti) etiket bitlerini ve frozen bitini
saklar. v5 ve öncesi reddedilir; ölçüldü: v5 diskten oynatmada 5.324/5.324
`unknown`. Bellek snapshot'ı flags'i zaten taşır. Test:
`rt_test_fluid_labels_cache_ipc.py` (RAM'i `sim_cache.clear ram_only` ile atıp
diskten okur). Bilinmeyen etiketi body gibi sunmak yasaktır.

## Kullanıcının çalıştıracağı kontroller

1. Normal C++ uygulama derlemesi. Dört yeni `.cpp` Visual Studio projesine ve
   filtrelerine eklendi; CMake mevcut recursive source glob'uyla bulur.
   Bu parti shader değiştirmez.
2. Yeni sıvı simülasyonunu birkaç kare oynatıp durdur. Eski disk cache'ini
   oynatmak sınıflandırıcıyı çalıştırmaz. Ayrı terminal:

   ```powershell
   python scripts/test/rt_test_fluid_labels_ipc.py "Physics Domain 1"
   ```

   Beklenen: PASS, primary toplamı particle_count, dolu gövdede body,
   kopuk damlalarda spray. Panel ve Python aynı sayıları göstermeli.
3. Yeniden seed/reset veya eski cache playback: `--allow-unclassified` ile
   aynı probu çalıştır. Unknown beklenebilir, hiçbir parçacık sayımdan düşmez.
4. Donan mum: simülasyon adımı sonrasında frozen sayısı artmalı; yeniden erime
   sonrasında etiket frozen kalmamalı. Kütle/hareket önceki davranışı korumalı.
5. Whitewater açık: secondary spray/foam/bubble ayrı sayılır. Whitewater
   açıp kapatmak primary toplamını ikincil parçacık sayısı kadar artırmamalı.
6. Aynı sahnenin büyük parçacık sayısında `last_step.milliseconds` değerini
   kaydet. Bu ölçüm CPU etiketleme maliyetidir; solver GPU süresi değildir.

2026-09-29 GPU kabul sonucu: 100.000 parçacıkta sıcak-kare ortalaması
**2,034 ms** (`on_gpu=true`); önceki aynı Vulkan iş yükü 4,304 ms, yaklaşık
2,1× hızlanma. Bin taşmasında CPU fallback doğrulandı. Ayrıntı ve ham raporlar:
NEXT_BUILD_CHECKS 23. parti.

Bağımsız C++ çekirdek regresyonu (kullanıcı, x64 Native Tools terminalinde,
repo kökünden; `NDEBUG` tanımlanmamalı):

```powershell
New-Item -ItemType Directory -Force .tmp/fluid_labels_test
Push-Location .tmp/fluid_labels_test
cl /nologo /std:c++17 /EHsc /I../../RayTrophiStudio/source/include ../../scripts/test/fluid_particle_labels_test.cpp ../../RayTrophiStudio/source/src/Physics/Fluid/FluidParticleLabels.cpp ../../RayTrophiStudio/source/src/Math/Vec3.cpp /Fe:fluid_particle_labels_test.exe
./fluid_particle_labels_test.exe
Pop-Location
```

Test: izolasyon/gövdeye katılma, histerezis, donma/çözülme, swap/compaction,
yeni parçacık, NaN/aşırı koordinat, geçersiz voksel, solid/granül ayrımı,
whitewater sayımı ve kütle/hız/konumun etiketleme tarafından değiştirilmemesi.
Codex bağımsız C++ test derlemesini yapmadı ve uygulamayı başlatmadı.
Kullanıcının açtığı derleme üzerinde dış IPC testleri çalıştırıldı.
