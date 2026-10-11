# Gaz kg/J GPU sahipliği — 2026-10-10

Kaynak hazır; kullanıcı C++/simulation shader derlemesi ve canlı kabul bekliyor.
Ölçülen termal-açık sahnede gas 823 ms, CPU kg/J advection 713 ms (%86.6).
Bu maliyetin taşınım/korunum hesabı GPU'ya alındı; henüz hızlanma ölçülmedi.

## Fizik adımı

Kg/J alanlarının kanonik sahibi GPU buffer bankasıdır. Semi-Lagrange veya
limited MacCormack final device MAC velocity alanını okur. Double reduction ve
normalizasyon cihazda çalışır, iki alan başarılı dispatch sonrası handle swap
ile birlikte yayımlanır. Fizik adımında tam kg/J grid upload/download **0 byte**.
Görsel scalar ile ortak sampling kullanılır; CPU passive-scalar referansına
uygun biçimde envanter solid hücrelerde sıfırlanmaz. Closed/periodic toplam
korunur; open boundary gerçek outflow'u geri eklemez.

Boiling/mist/combustion mevcut CPU üreticileri, eski grid aynasına yazmak yerine
benzersiz hücreler halinde küçük deposit kayıtları gönderir. Surface heat ve
below-boiling evaporation GPU'daki güncel vapour mass/humidity üzerinden
hesaplanır. Önce geçici sütun sonuçları doğrulanır, sonra gas buffer'larına
commit edilir. Sıvının mevcut host SoA tüketicileri için yüzey parçacık kayıtları
upload ve sonuç/ledger kayıtları download edilir; bütün parçacık fiziğinin CPU
ile ilişkisi kaldırılmış değildir. Conduction/freeze ve diğer mevcut termal
sistemlerin tam device residency kapsamı bu değişiklikten ayrıdır.

Boş gaz domaininde altı büyük kg/J alanı ayrılmaz; ilk gerçek kg/J katkısında
oluşur. Isı alışverişi ama envanter yoksa tek 4-byte sıfır buffer okunur.
128³ için advection alanları ve conservation scratch yaklaşık 48.25 MiB;
summary ve compact surface scratch buna eklenir. 2D dispatch büyük gridlerde
65535 grup sınırını aşmayı sağlar.

## Tüketiciler ve hatalar

`fluid.get`/list ve script/IPC toplam/centroid bilgisi aynı core reduction'a
bağlıdır: değişen revision için 64-byte sonuç, aynı revision sorgusunda cache.
Timeline kare önbelleği ve disk bake **tam snapshot tüketicileridir**; mevcut
CPU cache formatı nedeniyle istendiklerinde iki grid okunur. Timeline her kare
cache alıyorsa bu maliyet her kare devam eder. Bu nedenle uygulamanın tüm CPU
aktarımları sıfırlandı denmez. GPU cache snapshot residency henüz uygulanmadı.
Rewind/restore veya backend geçişinde açık ownership sınırı vardır; restore
bir kez upload eder. Cache kopyası canlı cihazın değişken revision'ını paylaşmaz.
Disk formatındaki kg/J float alanları kayıpsız ve sürüm 12 ile aynı kalır.

Allocation/dispatch/readback veya carrier hatası CPU fiziğine sessiz fallback
yapmaz: inventory_step_held/error yayımlanır, playback durur; reset gerekir.
Bozuk kare cache'e yazılmaz. CPU backend mevcut referans yolunu kullanır.

UI, script ve IPC aynı runtime ölçümlerini kullanır:

| Alan | Anlam |
|---|---|
| inventory_transport | idle / cpu / gpu_resident_idle / gpu_resident / gpu_failed |
| inventory_advection_ms | Kg/J taşınım/korunum host duvar süresi; kernel timestamp değil |
| gpu_surface_exchange_ms | Surface heat/evaporation, compact parcel IO dahil |
| inventory_grid_upload_bytes/download_bytes | Son fizik adımındaki tam grid transferi |
| inventory_snapshot_download_bytes | Ownership ömrü boyunca snapshot okumaları |
| inventory_query_download_bytes | Ownership ömrü boyunca küçük summary okumaları |
| inventory_step_held / inventory_error | Hata ve reset gerektiren durma |

`measured` zamanlardan önce okunmalı. Sparse deposit, compact parcel ve mevcut
gas scalar/render transferleri grid byte sayaçlarının kapsamına girmez.

## Tek derleme ve kabul

Ajan uygulama veya shader derlemedi ve canlı yeni kaynak kabulü çalıştırmadı.
Kaynak ABI/kayıt/consumer kontrolü, film matematik oracle'ı ve mevcut Matter
phase/solver-stage kontrolleri PASS. Bunlar C++/GLSL derleyici veya GPU testi değildir.

1. Kullanıcı eşleşen C++ ve `source/shaders/compile_sim_shaders.bat` derlemesini
   tek partide alır. Sekiz yeni inventory SPV yanında ortak scalar include'a
   taşınan mevcut grid advect/MacCormack SPV'leri de yenilenmeli. Runtime shader
   klasörüne eşleşen sonuçlar konur ve uygulama kullanıcı tarafından yeniden açılır.
2. Durmuş disposable sahnede dış terminal:
   `python scripts/test/rt_test_gas_inventory_gpu_ipc.py`.
   Boiling ve heat-exchange fixture'larının mevcut ledger/kütle/faz kapılarını
   korur; measured adımlarda resident yol, sıfır fizik grid transferi ve tekrar
   sorgusunda summary cache ister. Fixture reset yapar; asıl canlı state'i korumaz.
3. `python scripts/test/rt_test_contact_heat_ipc.py` ile temas regresyonu.
   CPU/Vulkan closed/periodic/open, Semi/MacCormack spatial parity, reset/backend
   swap ve cache/rewind/bake kg/J roundtrip hâlâ canlı doğrulama kapılarıdır.
   Allocation/dispatch/readback failure kapıları da kaynak testleriyle kapanmaz.
4. Aynı 128³ / kare 118 sahnede eski 713/823 ms ile inventory, surface, total,
   snapshot ve query byte sayaçlarını birlikte karşılaştır. Cache açık/kapalı
   ölçümü ayrı kaydet. Son-adım evaporation=0, toplam envanter=0 anlamına gelmez.
5. Yeni proje açarken timeline 0, playback durmuş olmalı; önceki projenin
   kaydedilen karesine replay yapılmamalı. Project load/reset kaynak değişikliği
   bu aynı derleme partisine dahildir.

Tam birleşik domain/H1 maliyet kabulü, GPU cache residency ve tüm termal
consumer migration kapıları açık kalır; yalnız kaynak hazır olmasıyla kapanmaz.
