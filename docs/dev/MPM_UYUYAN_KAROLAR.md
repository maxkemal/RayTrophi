# MPM uyuyan karolar

> **Durum:** TASLAK — 2026-11-04. A aşaması (desteğe bakan settle) CANLI: snow_support 5/5, duman 15/15. Bu not B
> aşamasının (gerçekten iş atlayan uyku) tasarımı. Kod yok. Önkoşul: A canlıda PASS ve
> dinlenik MPM'in kare maliyetinin ölçümü.

## Neden

MPM'de (kar, granüler, elastik) dinlenen bir yığın her alt adımda P2G, gerilme
güncellemesi, ızgara ve G2P'yi tam olarak ödüyor. Bugünkü "uyku" bayrağı (`flags & 8`,
`granular_sleeping`) **hiçbir işi atlamıyor**; yalnızca hızı sıfırlıyor. DEM'de benzeri
ölçülmüştü: yığının büyük kısmı durgunken GPU süresinin çoğu sıfıra yakın hızları entegre
etmeye gidiyordu.

## A aşaması (CANLI, 2026-11-04) — destek ölçütü

`GranularConstitutive.h` `settleParticle` / `updateSupport`, GPU
`sim_fluid_granular_settle.glsl`:

- Sönüm ve uyku yalnızca **desteklenen** parçacığa: yerçekimi yönündeki net ivmenin
  0.2 s'lik ortalaması 0.25·g'nin altında.
- İvme, sönümden ÖNCEKİ hızdan ölçülür; yoksa sönümlenen bir düşüş limit hızda ivmesiz
  görünür.
- Yeni parçacık desteksiz başlar (`kGranularSupportFresh`).
- Uyanma kendiliğinden: desteği giden parçacık bir alt adımda g·dt kazanır, ortalama
  ~0.06 s'de eşiği geçer.

Kök: basınç destek değildir. Havada oturtulmuş kar bloğu üstten asılı kaldı.

## B aşaması — tasarım

### Birim: karo, parçacık değil

DEM'de uyuyan tane komşusuna sabit bir cisim gibi görünür. MPM'de gerilmeyi **ızgara**
taşır. Tek bir parçacığı uyutup P2G'den çıkarmak komşu hücrenin kütlesini ve gerilmesini
eksiltir; destek kaybolur. Uyuyan birim **ızgara karosu** olmalı: 4³ hücre, sıkıştırılmış
sayfa düzeni ile aynı boyut.

### Uyku

Bir karo uyur, eğer kesintisiz `sleep_time` (varsayılan 0.2 s) boyunca:

- içindeki her MPM parçacığı A ölçütüne göre desteklenmiş ve hızı `sleep_speed`'in altında;
- karonun ızgara yüzlerinde net hız değişimi (G2P öncesi / sonrası) toleransın altında;
- karoda sıvı, gaz, DEM teması ya da hareketli collider yok.

### Uyuyan karo

- Parçacıkları P2G, gerilme, G2P ve ilerletme yapmaz; F, J_P ve hasar donar.
- Karonun hücreleri ızgarada **duran bir katı** gibi davranır: collider maskesine benzer
  bir "uyku maskesi". Uyanık komşu, MPM sınır kuralından (`sim_matter_mpm_boundary`)
  geçer, yani sürtünme ve destek aynı yerden gelir.
- Sayaçlar ve görünüm donmuş durumu okur. DEM ders 5: muhasebe kaybolmaz; son alt adımda
  sayılır.

### Uyanma (DEM'in çözdüğü sorunlar)

| DEM'de ne oldu | MPM karosu için kural |
|---|---|
| Hız tek başına denge değildi (rev 22) | Uyku için A desteği ve yüz ivmesi; yalnız hız yetmez |
| Desteksiz tam-g düşüş donmamalı | A ölçütü bunu tanım gereği dışlar |
| Uyanık komşuya da WAKE yazmak atomik çığı ve sayaç sıfırlamayı üretti (rev 23) | WAKE yalnızca **uyuyan** komşu karoya ve yalnızca sınır yüzünde göreli hız > `sleep_speed` ise |
| Bağlam değişince herkes uyanmalı | Yerçekimi, kuvvet alanı, collider parmak izi, domain sınırı, madde düzenlemesi, tohum → tüm karolar uyanır |
| Dayanak kaybolunca havada kalma | Uyuyan karonun yerçekimi yönündeki komşusu boşalır ya da uyanıp uzaklaşırsa uyanır |
| Zincirleme uyanma tüm yığını uyandırır | Uyanan karo komşusunu ancak kendisi gerçekten hareket ederse uyandırır (yüz hızı), "uyandım" bilgisiyle değil |

### Ölçüm (yazmadan önce)

1. Dinlenik kar yığınında (örnek: kullanıcının 87 k parçacıklı sahnesi) kare süresinin
   ne kadarının P2G, gerilme ve G2P'ye gittiği (`gpu_kernel_timings`).
2. Karoların yüzde kaçının A ölçütüyle uyku adayı olduğu.
3. Beklenen kazanç: ölçülmeden iddia yok. DEM'de 1.125–1.25× kaldı, çünkü host'taki
   birleştirme, sıralama ve yükleme işleri uykudan etkilenmiyordu.

### Kabul kapıları (taslak)

- Uyku açık / kapalı: dinlenik yığın şekli ve kütle merkezi farkı < 1 mm.
- Uyuyan yığına düşen kartopu: yalnızca çarpma bölgesi uyanır; uyanık karo sayısı sınırlı.
- Desteği çekilen yığın, kapalı olanla aynı ±1 karede düşer.
- Kare süresi kazancı ölçülür ve raporlanır.
