# İlk kare hold ve kapanış logger yaşam sırası — 2026-10-10

Kullanıcının yeni derlemesi dış named-pipe üzerinden incelendi. 0→1 karede
416 sıvı parçacığı oluşuyor; gaz `inventory_step_held=true`, hata
`gas surface exchange failed; reset required`. UI bu hata nedeniyle Play'i
duraklatıyor. GPU timestamp tekrarında `sim_inventory_surface_exchange`
çalıştı (0.132 ms); surface commit ve sonraki gaz adımları çalışmadı.
Bu buhar üretimi/fizik performans kabulü değildir: gaz adımı başarısızdır.

Runtime SPV'nin parcel alan offset'leri 0..48, parcel/result stride 64 ve
result alan offset'leri 0/32/48 host yapılarıyla uyuşuyor. Özel host alignas
kaldırılması buffer byte layout'unu değiştirmiyor.

Somut kaynak kusuru: surface dispatch kaydedildikten sonra sonuç tekil
`downloadBuffer` ile okunuyordu. Vulkan backend'in batch dışı indirmesi ayrı
copy command buffer'ını submit eder; bekleyen compute recording'i submit
etmez. Dolayısıyla doğrulama shader'ın güncel sonucu yerine henüz yazılmamış
veya eski buffer içeriğini okuyabilir. Surface readback ve aynı kusuru taşıyan
inventory summary readback şimdi transfer batch'ine kaydediliyor. Batch sonu
aynı komut akışındaki dispatch→barrier→copy→fence sırasını tamamlıyor; ekstra
tam-grid transferi eklenmiyor. Başarısız aşama/parcel ayrıntıları inventory
hatasında korunuyor; generic üst hata ilk ayrıntıyı ezmiyor.

Kapanış stack'i `sceneLogOnChangeStates()` yerel-static unordered_map'ine
global renderer destructor'ından erişim içeriyor. Bu map, global renderer'dan
önce yok edilebilir. Map ve mutex artık tek yaşam nesnesinde tutuluyor; bu
nesnenin destructor'ı üyeleri yok edilmeden logging'i kapatıyor. UILogger
destructor'ı da logging'i kapatıyor. Log makroları ve on-change/reset yolları
yaşam kapısını kontrol ediyor. Geç renderer destructor'ları yok edilmiş log
nesnesi, mutex veya map'e erişmiyor. Tam process shutdown canlı kabulü bekliyor.

İnceleme sırasında `x64/Release/SceneLog.previous.txt` 130,533,783 bayttı
(yaklaşık 124.5 MiB), tarihi 2026-10-01; aktif log yaklaşık 12 KiB idi.
Önceki log bir rotasyon arşividir, simülasyon girdisi olarak okunmaz. Dosya
boyutu bu ilk-kare hold'unun açıklaması değildir. Codex log silmedi.

Build yapılmadı. Üç gaz/WW kaynak-matematik kontrolü ve logger lifetime
contracts kontrolü çalıştırılır. Sonraki kullanıcı C++ derlemesinde (bu düzeltme
shader değiştirmez), uygulamayı yeniden başlatıp aynı sahnede Play'in birkaç
kare kesintisiz ilerlemesi, inventory hold'unun kalkması, buhar/kgJ ledger
kontrolleri ve normal kapanış doğrulanmalı. İnceleme sahnesi kare 1'de durmuş
bırakıldı, authored parametreleri değiştirilmedi; GPU timing ayarı geri yüklendi.
