# Cache birikmeyen açık sahne: GPU ve havadaki buhar

Kullanıcı derlemesi sonrası dış IPC ile 40 manuel fizik adımı (dt=1/24).
Timeline başlangıç/son frame 0; oynatma kapalı. Reset veya authored ayar değişimi
uygulanmadı. GPU timestamp açılıp başlangıçtaki kapalı duruma geri getirildi.
Cache probe boyunca 1 kare / 2.227.568 byte'da sabit kaldı. Build yapılmadı.

## Maliyet

Erken sıcak örnekler (4/8/12. manuel adım): solver IPC wall 158,8 / 186,4 /
224,0 ms; GPU compute kernel toplamı 22,5 / 25,6 / 42,1 ms. Liquid total
56,3 / 70,4 / 81,5 ms; gas total 86,3 / 99,9 / 126,2 ms.
Liquid upload+download yaklaşık 169..170 MB/adım. Liquid 15..16 transfer batch
kapatışı ve 4 synchronize çağrısı; batch_end toplam 15..21 ms.

Daha sonraki GPU toplamları 25,5..195,7 ms arasında değişti. Baskın gerçek
compute kernel `sim_grid_sor`: 80 dispatch, örneğe göre 16,6..131,5 ms.
Aynı kernel workload'unun değişen timestamp süresi için GPU clock/contention/
preemption ayrıştırması yapılmadı; bütün fark host fence'e yüklenemez.

Manuel IPC wall geç örneklerde 615..981 ms. Perf `loop.frame` ~505 ms ve
Main.cpp paused/idle branch SDL_WaitEventTimeout(...,500) içeriyor. Bu idle
bekleme RPC yanıtını geciktirebilir; 500 ms fark solver veya GPU wait maliyeti
olarak sayılmaz. Bu çalışma normal PLAY modunun FPS/occupancy ölçümü değildir.
Kullanıcının %25 GPU gözlemi tek başına compute limit veya wait kanıtı değildir.
Host IO ve compute arasında dur-kalk hâlâ vardır, pressure kernel de gerçek yük.

## Havadaki buhar

Örneklerde contact_cooled_particles=0. 8. adım evaporation ~0,0364 gram,
12. adım ~0,0394 gram; bu görünür yoğun bir plume olmak zorunda değildir.
40. adım physical inventory ~0,741 gram ve gas max density ~0,0213.
Sıcaklık çoğunlukla 373,15 K, max ~373,1648 K. Su tam boiling noktasında
başlıyor; superheat tarafından ödenen latent üretimi doğal olarak küçüktür.
Evaporation/boiling ayrı kapıları kaynakta korunmalı, sebepsiz buhar eklenmemeli.

Kaynakta somut geometri kısıtları:

1. GasSurfaceExchangeGpu sadece (x,z) sütununun en yüksek y'sini seçiyor.
   Düşen jetin yan yüzeyleri surface heat/evaporation'a katılmıyor. Alt parçacık
   üzerindeki boş hava yerine yukarıdaki jet parçası bütün sütunu örtüyor.
2. Boiling release_y=max(column_top, parcel_y)+1. Kaynamış parçacığın yakındaki
   açık hava yüzeyi yerine bütün sütunun üstüne release ediliyor. Zemindeki
   yayılmış düşük sütunlarla yüksek jet sütunu farklı davranır.
3. Phase transfer momentum ledger'a kaydediliyor; boiling snippet'i gas MAC
   velocity'ye bu momentumun yerel deposit'ini uygulamıyor. Gazın moving-liquid
   boundary/buoyancy tepkisi bu doğrudan faz momentum transferinin yerine geçmez.

Buhar tümüyle sabit değil: physical mass centroid y 12. adımda 1,706,
40. adımda 1,337. Bu yeni üretim + advection'ın birleşik centroid'idir; tek
parcel izleyicisi veya gas velocity yönünün doğrudan ölçümü değildir. Gazın
max speed'i 40. adımda 3,42 m/s; moving liquid boundary mean y ~-9,98 m/s.
Jet yakınında sürüklenme ile buoyancy birbirine karşı çalışabilir. Gazın yerel
hızını örnekleyip aynı katkının trajektorisini izleme kabulü hâlâ gerekir.

8. adımda evaporation pozitifken summary mass 0 göründü; daha sonra inventory
pozitif. Önceki korunum fix'i tüm deposit/summary zincirinin doğruluğunu kapatmaz.
Closed ledger/grid parity yeni build'de ayrı acceptance ile tekrar doğrulanmalı.

## Önerilen sonraki kaynak kapsamı

- Gas surface için 3D occupied-cell tablosu, altı açık yüzey ve komşu air-cell
  temelinde gruplama. Yüzey alanı parcel sayısına bölünmeli; çift latent/heat
  ödeme ve collider hücresine vapor deposit yapılmamalı.
- Boiling için gerçek parcel çevresinde açık gas komşusu, local mass/energy ve
  momentum deposit. Cold/closed/no-air fallback ve ledger bütünlüğü açık tanımlı.
- Liquid P2G/viscosity/pressure/G2P arası gereksiz dense MAC host publication'ı
  kaldırmak; CPU boundary/thermal consumers için ownership güncel tutulmalı.
- SOR'u körlemesine azaltmadan gas pressure ölçümü; convergence/divergence ve
  plume davranışı aynı toleransla karşılaştırılmalı.

Kaynak bu probe'da değiştirilmedi. Ham örnekler:
`gas_optimized_probe_2026_10_10.json`, `gas_optimized_step_probe_2026_10_10.json`,
`gas_optimized_perf_sections_2026_10_10.json`.
