# Gaz / sıvı yüzeyi ve whitewater — 2026-10-10 kaynak partisi

Durum: kaynak ve matematik kontrolleri PASS. C++/shader derlemesi çalıştırılmadı.
Bu partinin canlı fizik ve performans kabulü henüz yapılmadı. Önceki canlı
raporlar eski binary ölçümleridir; bu değişikliklerin hızlanma kanıtı değildir.

## Fizik

- Yüzey alışverişi global sütun tepesinden çıkarıldı. Kanonik sıvı SoA'sından
  gaz hücresinin altı açık yüzü bulunur; katı ve dolu komşular açık yüz sayılmaz.
  Jetin yan yüzleri zemine ulaşmadan ısı ve buhar alışverişi yapabilir.
- Yüz başına ayrık kütle/ısı kapasitesi payları kullanılır. Tek parçacığın
  bütçesi birden fazla yüz tarafından tekrar harcanmaz. CPU ve GPU aynı planı
  ve sonuç doğrulamasını kullanır; sonuçlar fiziksel parçacıkta yeniden birleşir.
- Kaynama buharı en yakın erişilebilir sıvı/hava yüzeyine çıkar. Tam kapalı
  bölgede çıkış bulunamazsa kütle ve aşırı ısı korunur; uzaktaki sütun tepesine
  yapay aktarım yapılmaz.
- Buharlaşmanın soğuması başlangıçtaki toplam kapasite üzerinden hesaplanır.
  Kalan sıvı ve çıkan buharın duyulur enerjisi ile gizli enerji birlikte korunur.
  Kaynamada gizli enerjiden arta kalan aşırı ısı çıkan buhara taşınır.
- Faz değiştiren kütlenin kaynak hızı, kütle ağırlıklı MAC yüz hızına aktarılır.
  GPU inventory yolunda kompakt kayıt yüklemesi ve yeni momentum shader'ı
  kullanılır; tam gaz-grid indirmesi eklenmez. Duvar yüzlerini mevcut sınır
  çözümü yönetir. Bu, mevcut taşıyıcı gaz modelinin düzeltmesidir; tam sıkışabilir
  çok bileşenli gaz çözümü değildir.

## Whitewater

- Kapalıyken criterion/bin/neighbour hesapları çalışmaz. İkincil parçacıklar,
  sayaçlar ve WW compute tamponları bırakılır. Kaynama bubble görsel üretimi de
  kapı arkasındadır; fiziksel kaynama, gizli ısı ve void-fraction devam eder.
- Render'ın hâlâ okuyabileceği ortak sphere tahsisi hemen yok edilmez; boş
  görünüm yayımlanır ve mevcut render yaşam döngüsüne bırakılır.
- Açıkken önceki adımın potansiyelleri veya komşu sayıları kullanılmaz.
  Mevcut sıvı verisinden criterion ve mevcut WW parçacıklarından komşular
  tek transfer batch'inde alınır. Ek ölçüm amaçlı fence kaldırıldı.
- Donmuş, tükenmiş ve sıvı olmayan parçacıklar WW kaynaklarına dahil edilmez.
- Bin penceresi sıvı ve mevcut WW çevresine daraltılır; orijinal dünya hücre
  kuantizasyonu ve arama yarıçapı korunur. Host görünümünün güvenilir olmadığı
  device/mixed-model yolunda tam-domain fallback korunur.
- Bir GPU kovası taşarsa eksik komşu listesi fizik hesabına verilmez; criterion
  ve sınıflandırma eksiksiz CPU hesabına döner. Bu yoğun bölgede maliyeti
  artırabilir, fakat yanlış düşük yoğunluk/spray sınıflandırması üretmez.
- Native, script ve IPC stats: `inputs_current`, `overflow_fallback`,
  `bin_cells`, `bin_bytes`, `input_ms`. Sonuncusu batch/transfer/bekleme dahil
  duvar süresidir, saf GPU kernel süresi değildir.

## Maliyet sınırları

Katısız ve granular olmayan, geçerli GPU hızını koruyan sıvı yolunda viscosity
ve pressure başlangıcındaki yinelenen MAC upload'ları atlanır. Collider veya
CPU'nun değiştirdiği hız varsa eski güvenli upload devam eder. Gerekli CPU
consumer indirmeleri ve timeline/cache snapshot transferleri kaldırılmadı.
Bu parti tüm solver'ı GPU-resident yapmış sayılmaz. WW dar pencere kazanımı
sahneye bağlıdır. Hız, RAM ve VRAM kazancı eşleşen build'de ölçülmelidir.

## Tek kullanıcı derlemesi ve kabul

1. C++ projeyi ve simulation shader'larını aynı kaynak partisinden derle.
   Yeni `sim_inventory_momentum.comp`, değişen
   `sim_inventory_surface_exchange.comp` ve dört `sim_foam_*` shader'ının
   SPV'leri runtime'a birlikte ulaşmalı. WW buffer/push-constant ABI değişti;
   eski shader ile yeni C++ karıştırılmamalı. Uygulamayı yeniden başlat.
2. Cache kapalı, timeline durmuş test sahnesinde eşit adım/dt ile WW kapalı ve
   açık ölç: liquid/gas/WW süreleri, transfer byte, private RAM ve VRAM. Kapalı
   WW sayısı/WW hesap stats sıfır olmalı; açıkta `inputs_current` true olmalı.
   Taşma varsa `overflow_fallback` true ve GPU sonuçları reddedilmiş olmalı.
3. Aynı sıcak su jetini zemine temas öncesinde ve sonrasında izle. Yan yüzeyde
   uygun sıcaklık/nem koşullarında buhar üretimi, yerel çıkış, kaynak hızının
   taşınması ve sonra gazın yükselmesi kontrol edilir. Her hava durumunda görünür
   kaynama zorunlu değildir: ısı/nem ve kaynama bütçeleri korunur.
4. Dış terminalden, durmuş disposable test sahnesinde sırayla çalıştır:

   ```powershell
   python scripts/test/rt_test_gas_inventory_gpu_ipc.py
   python scripts/test/rt_test_contact_heat_ipc.py
   python scripts/test/rt_probe_whitewater_determinism_ipc.py --backend vulkan
   python scripts/test/rt_test_whitewater_label_routes_ipc.py
   ```

   Bu testler geçici domain açar; mevcut domainleri geçici durdurabilir.
   Primary WW A/B sonucu deterministik olmayan GPU yolunda A/A gürültü
   tabanıyla değerlendirilir. Kütle/enerji ledger-grid tutarlılığı ve cache
   rewind ayrıca kontrol edilir. CPU backend aynı fizik senaryosuyla karşılaştırılır.

## Yapılan kaynak kontrolleri

Dokuz test PASS: GPU inventory contracts/math, altı-yüz/yerel çıkış/enerji/
momentum matematiği, WW pencere/off/current/overflow/API contracts, matter phase,
solver stages, kayıpsız frame cache, GPU dispatch ve domain panel fields.
IPC descriptor `--check` PASS. Bunlar C++ veya GPU çalıştırmaz.
