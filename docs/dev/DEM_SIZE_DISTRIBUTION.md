# DEM fiziksel boyut dağılımı

2026-10-22: kaynak değişikliği hazır; kullanıcı derlemesi ve canlı kabul bekliyor.

Domain → Solvers → DEM bölümünde minimum yarıçap, maksimum yarıçap, dağılım,
boyut seed'i ve iki boyutlu karışımda iri taşların sayı oranı ayarlanır.
Maksimum yarıçap `0` ise önceki sabit boyut davranışı korunur. Diğer durumda
minimum ile minimumun dört katı arasında ve en fazla 1 metre olabilir.

- `uniform`: yarıçaplar minimum–maksimum arasında eşit olasılıklı.
- `log_uniform`: yarıçapın logaritması eşit olasılıklı.
- `bimodal` / **Sand + large stones**: küçük taneler minimum, büyük taşlar
  maksimum yarıçapta; `large_grain_fraction` iri taşların **sayı oranıdır**.

Örnek: minimum 0.01 m, maksimum 0.02 m, bimodal, iri taş oranı 0.10.
Aynı yoğunluk ve doluluk oranında büyük taşın kuru kütlesi küçük tanenin
sekiz katıdır. Her ikisinin yerçekimi ivmesi aynıdır; darbe, dönme ve sürüklenme
tepkileri kütle, atalet ve boyut nedeniyle farklıdır. Bu çalışma farklı taş
yoğunluğu veya tane başına ayrı fiziksel malzeme eklemez; domain'in mevcut DEM
malzemesi kullanılır. Görsel kaynak mesh materyalleri ayrı sistemdir.

## Ortak UI / script / IPC

UI mevcut `matterGrainSettings` servisini çağırır. Python ve IPC aynı patch
doğrulaması, hata ve doğum kilidini kullanır. Örnek Python:

```python
rt.fluid.set_grain_settings(
    domain="Matter", radius_m=0.01, radius_max_m=0.02,
    size_distribution="bimodal", large_grain_fraction=0.10, size_seed=1337)
```

IPC: `fluid.set_grain_settings`, aynı parametreler. Okuma: `fluid.grain_settings`.
`radius_m`, `radius_max_m`, `size_distribution`, `size_seed` ve
`large_grain_fraction` için dolu domain'de değişiklik reddedilir; önce parçacıklar
resetlenmelidir. Hatalı patch hiçbir ayarı değiştirmez. Alanlar sahne kaydında
mevcut grain JSON serileştirmesi üzerinden saklanır; eski kayıt sabit boyutla açılır.

Yarıçap, kalıcı parçacık kimliği ve kayıtlı dağılım ayarlarından deterministik
hesaplanır. Parçacık sıralama, domain altkümesine ayırma ve kimlikleri koruyan
cache dönüşü yeniden boyut seçmez. Doğumda `r³` kuru kütle, `r³` su kapasitesi
ve iki tanenin yarıçap toplamına göre çakışma kontrolü uygulanır.

GPU kütle tamponu: ilk N float kütle, sonraki N float yarıçap; iki kuyruk float'ı
gerçek temsil yarıçapı ve en büyük simülasyon yarıçapıdır. DEM shader revision
**28** gerekir. Eski SPIR-V ile yeni C++ kullanılmamalıdır.

Temas yarıçap toplamını, atalet `0.4*m*r²` değerini kullanır. Temas yay sertliği
çiftin etkin yarıçapıyla ölçeklenir; eşit boyutta eski davranış korunur. Geniş
aşama küçük tanelerin hücre ölçeğini koruyup büyük taneler için arama erişimini
genişletir. Verlet listesi her çiftin erişimine göre oluşturulur. Zaman adımı
en küçük kütle, en küçük yarıçap ve en büyük temas sertliğiyle güvenli tarafta seçilir.
Duvar/mesh temasları, uyku uyandırması, CPU/GPU sıvı etkileşimi, MPM teması,
gözeneklilik, su kapasitesi, hava sürüklemesi, ısı yüzeyi ve Sphere/Near Mesh
çizimi aynı fiziksel yarıçapı kullanır.

## Sınırlar ve kabul

Mevcut 24 temas geçmişi, 32 komşu listesi ve hash bucket kapasitesi korunur.
Çok yoğun büyük/küçük karışımlarda bu sınırlar aşılırsa adım açık hata ile tutulur;
temaslar sessizce düşürülmez. Bu sürüm sınırsız polidispers paketleme sağlamaz.
Geniş dağılım ve küçük minimum yarıçap daha fazla arama/substep gerektirebilir.
İlk kabulü 1:2 yarıçap oranıyla yapın; 1:4 yoğun yığın ayrıca kapasite testi ister.

`fluid.matter_models` → `grain_diagnostics.size_population`: ölçülen minimum ve
maksimum yarıçap, toplam küre hacmi ve aralığın orta noktasının iki tarafındaki
tane sayıları. `bimodal` için bunlar doğrudan küçük/iri popülasyondur.

Kullanıcı doğrulaması:

1. C++ ve ilgili DEM/sıvı/MPM shader'larını birlikte derleyin.
2. `scripts/test/matter_grain_size_test.cpp` ve mevcut doğum/stages testlerini
   derleyip çalıştırın: eski sabit boyut, dağılım, kütle/su oranı, doğum aralığı,
   sıvıya verilen hacim ve sıralama determinismi.
3. Boş, duraklatılmış sahnede ayrı terminalden
   `python scripts/test/rt_grain_size_ipc.py`: gerçek dağılım, kübik kütle,
   reset/tekrar ve canlı ayar kilidi. Uygulamanın gömülü script alanından çalıştırmayın.
4. UI'de 0.01/0.02 m ve %10 iri taş dökün. Sphere ve Near Mesh görünümünü,
   zeminde oturma yüksekliğini, iri taş–kum darbelerini ve yığındaki kapasiteyi inceleyin.
5. Aynı seed ile reset, sahne kaydet/yükle ve cache ileri/geri karşılaştırın.
6. Mevcut sleep/transfer, gaz sürüklemesi, wet-grain, ısı ve MPM/sıvı momentum
   regresyonlarını değişken ve sabit boyutta tekrarlayın.

Codex derleme, shader derleme, uygulama başlatma veya canlı IPC testi çalıştırmadı.

## Revision 29: variable neighbour capacity

Mixed-size populations now start with 128 neighbour candidates per grain;
fixed-size populations retain 32. A standalone DEM list-only overflow retries
with doubled capacity, subject to real device/working-set budgets and shader
index width. Shared MPM/liquid clocks cannot safely replay external owners and
still report overflow. The GPU mass-buffer tail has one extra float for list
stride; rebuild C++ and every DEM shader variant together. Actual contact history
remains 24 contacts; neighbour candidates and touching contacts are different budgets.
Runtime diagnostics expose neighbour_list_capacity and neighbour_list_retries.

## Revision 30: mixed-size contact history

The floor-impact refusal at 24 records is addressed separately from neighbour
candidates. Mixed-size history starts at 64, fixed-size at 24; standalone DEM can
retry history overflow at 128. Four mask words replace the old single mask,
including sleep metadata offset updates. Memory is charged for actual allocated
history and transactional growth; shared external clocks are never replayed.
Adaptive CFL uses measured contact counts plus margin instead of assuming the
whole newly allocated capacity on every standalone retry. Diagnostics report
contact_history_capacity and contact_history_retries. Rebuild C++ and all DEM
shader variants together (revision 30). Live verification remains pending.

This increases per-population history allocation; sparse per-grain storage is
still future architecture work. A one-million-grain, 128-record history alone
uses about 3.34 GiB and can exceed a device single-buffer limit.
