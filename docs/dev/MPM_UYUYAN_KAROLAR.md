# MPM uyuyan karolar

> **Durum:** CANLI — 2026-11-04. A (desteğe bakan settle) ve B'nin domain uykusu canlı: dinlenik 83 k kar 123 → 16.7 ms/kare (7.4×), şekil farkı 0.06 mm. Kısmi (karo/hücre) uyku yazıldı, ölçüldü (%5–7 YAVAŞ), söküldü. Sıradaki kaldıraç: alt adım dispatch zinciri.
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


## B — canlı sonuç ve ne söküldü (2026-11-04)

Bu turu ajan derledi ve ölçtü (kullanıcı onayıyla). Sahne: kar, 5 m kapalı domain,
h 0.1 m, kullanıcının sahnesinin ölçüsü.

### 1. Ölçülen gerçek: dinlenik MPM karesi zincire bağlı, parçacık işine değil

- `perf.gpu_kernel_timings`: 384 parçacıkla GPU **94.5 ms/kare**, 100 k parçacıkla
  **102.6 ms/kare**.
- Küçük sahnede ızgara çekirdekleri çağrı başına 3–6× yavaş: seri zincir GPU'yu boşta
  bırakınca GPU saatini düşürüyor.
- Kare = ~50 alt adım (karın dalga hızı, hareket değil) × ~36 dispatch. Her dispatch'in
  arkasında tam compute→compute bariyeri var.
- `dispatch_call_ms` (~100 ms) dispatch başına ana makine maliyeti DEĞİL. 512 dispatch'te
  bir `synchronize()` dispatch'in içinden çağrılıyor; süre büyük ölçüde GPU'yu beklemek.

### 2. Kısmi uyku: yazıldı, ölçüldü, söküldü

Donmuş hücrelerin parçacıkları iki şeritten de çıkarıldı (bölümleme kodu 4); hücreler
MPM şeridinin katı maskesine "duran duvar" olarak eklendi. Üç ders, sırasıyla:
1. **Duvar destek verir, yük vermez.** İlk kural (alt hücrede kütle varsa destekli) uyanık
   karı donmuş bir tepenin altında yüksüz bıraktı. Tepe uyanınca yük birden bindi, plastik
   kar ezildi ve yığın uyku kapalıya göre **0.39 m alçakta** bitti. Düzeltme: sakin
   26-komşuluk ve zeminden yukarı yalnız donmuş hücre üzerinden destek.
2. **Yerçekimi donmuş parçacığa da işliyordu.** Kanonik kuvvet aşaması bütün parçacıklara
   uygulanıyor, G2P ise geri almıyordu. Sonraki karede parçacık g·Δt = 0.41 m/s ile uyanıp
   yığını sarsıyordu; donma hiç yerleşmedi. Düzeltme: geri yazımda hız sıfırlanır.
3. **Sonuç:** şekil 3 mm içinde doğru, ama kare **%5–7 YAVAŞ** (üç A/B). %74 donmuşken
   P2G/G2P/gerilme süreleri değişmedi; işi azaltmak yalnızca GPU saatini düşürdü.
   CLAUDE.md "ölü yolu sök" gereği söküldü. Kurallar `MatterMpmSleep.h` başlığında duruyor:
   zincir kısalırsa kısmi uyku yeniden anlam kazanabilir.

### 3. Kalan: domain uykusu

- Domain'in bütün parçacıkları 3 kare dinlenirse adım hiç alt adım koşmaz
  (`gpu_status = matter_asleep`).
- Dinlenme ölçütü parçacık düzeyinde: 0.1 m/s altı VE (uyuyor YA DA desteklenen ve
  0.03 m/s altı).
- Hücre kurallarına (sakin komşuluk, destek zinciri) gerek yok; onlar kısmi uykunun yük
  sorunlarıydı. Hücre kuralları açıkken bütün yığının %65–74'ü "dinleniyor" görünüyordu;
  parçacık ölçütüyle %100. Önceki "yığın sürünüyor" tahminim yanlıştı: tavan, kısmi
  dondurmanın kendi sarsıntılarıydı.
- Uyumaz: anahtar kapalı (`fluid.set_grain_settings sleep`, DEM ile ortak), kuvvet alanı,
  sıvı/elastik parçacık, hareketli collider, bağlam değişimi (yerçekimi, ızgara, collider
  seti, parçacık sayısı).
- Canlı (`rt_test_mpm_sleep_ipc.py`):

  | | Uyku açık | Uyku kapalı | Hız | Şekil farkı |
  |---|---|---|---|---|
  | Kalın yığın (83 k) | 16.7 ms | 123.4 ms | 7.4× | 0.06 mm |
  | İnce katman (14 k) | 4.0 ms | 33.8 ms | 8.4× | 1.02 mm |

  Havadaki blok hiç uyumadı.

### 4. Ayrıca bu turda (gerçek iş kesintisi; GPU'yu doyuran sahnelerde işe yarar)

- **Sıvı yoksa P2G kütle gradyanını atlıyor.** Tek okuyucusu sıvı/granüler temas. Gradyan
  atomikleri P2G'nin 5 atomiğinden 3'üydü: P2G **84 → 49 ms/kare**.
- **Kümelenmiş alt grup P2G:** aynı taban hücredeki 8'li küme önce toplar, tek lane yazar.
  Yalnız `VK_SUBGROUP_FEATURE_CLUSTERED_BIT` olan cihazda yüklenir; ölçülen kazanç küçük
  (~%5–10).

### Sıradaki kaldıraç (yazılmadı)

Dinlenmeyen (hareketli ya da kısmen hareketli) MPM sahnesinin karesini zincir belirliyor:
- x/y/z bileşen dispatch'lerini birleştirmek (P2G, gerilme P2G, temizleme, normalize);
- bağımsız dispatch'ler arasındaki bariyerleri kaldırmak.

Paylaşılan sıvı koduna dokunur; önce küçük-parçacık kontrollü ölçümle zincir payını
kesinleştir.
