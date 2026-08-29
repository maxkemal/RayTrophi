# Çökelme modeli — neden LEM tutuldu, ne eklendi

> **Durum:** AKTİF — 2026-08-26. Avulsiyon ve alluvial yayılma yazıldı,
> runtime doğrulanmadı (bkz. `NEXT_BUILD_CHECKS.md`).

★ 2026-08-27 eki: aşağıdaki "Ölçü aleti" bölümünün dayandığı yol ölçüldü ve
**iki arıza** çıktı — `terrain.erode` GPU yolunda istatistikler hiç dolmuyordu
(probe sıfırı yeşil sayıyordu), ve `deepest_deposit_meters` bu yolda brüt
defter, node yolunda net yükselmedir. Ayrıca `maxDepositionMeters`'ın yalnızca
route geçişini bağladığı, talus'un sınırın dışından yükselttiği ve doymadığı
ölçüldü. Tümü: `TERRAIN_BUILD_CAP_SCOPE.md`.

## Kararın kendisi

Üç sistem yaklaşık aynı işi yapıyordu — kanal kazmak: **mature river channel**
(`terrain_hydraulic_channel_evolve.comp`), graph'taki **Fluvial node**
(`fluvialErosionGPU` → `terrain_fluvial_runoff/apply/talus`), ve LEM'in
**fluvial cycle**'ı. Fazlalık gerçekti ve sökülmesi planlanıyordu.

Sökümden önce üçünün **sediment davranışı** tek tek okundu, ve tablo kararı
tersine çevirdi:

| | Sediment | Taşıma menzili |
|---|---|---|
| `channel_evolve` | `sedimentOut[i] = sed + erode - deposit`, **aynı hücrede** | **Yok** — hücreler arası hiç hareket yok |
| `fluvial_runoff` | parselin taşıdığı yük | Parsel ömrü kadar (~64 adım) |
| LEM `route` | MFD ağırlıkları boyunca aşağı advekasyon + `fdep` | **Sınırsız**, host `eroded == deposited + exported + carried` denetliyor |

İstenen davranış — *taşınan toprağın düzlüklere yayılması, yamaç diplerinde
birikmesi, birikimin içinde yeni kanal açması* — bir tanenin sırttan ovaya
gitmesini gerektirir. Bu, ilk ikisinde **tanım gereği** olamaz.

★★★ Yani silinmeye hazırlanan şey, istenen davranışın tek temeliydi. Fazlalık
kanal kazıcılarda, taşıyıcıda değil. Söküm sırası buna göre değişti: taşıyıcı
tutulup tamamlandı, kazıcı fazlalığı sonraya bırakıldı.

## Eksik olan neydi

Taşıyıcı vardı ama **çökelme yarısı** dört yerde eksikti. İkisi bu partide
kapatıldı:

### 1. ★★★ Avulsiyon yoktu — yapısal engel

`solveDrainage()` `drainageRefreshInterval` (6) iterasyonda bir koşuyor,
`route` ise iterasyon başına `sedimentRouteSteps` (96) adım. Yani **96 adım
boyunca yatak yükselirken ağırlıklar donuktu.**

Yelpazeyi yapan şey çökelme değil, çökelmenin **akış yolunu bozması**: yatak
yükselir → kanal artık en alçak yol değildir → yol değiştirir → terk edilmiş
lob kalır → tekrar. Donuk ağırlıkla akış, kendi gömdüğü kanalı kullanmaya
devam ediyordu. Sonuç: koni değil **tek bir sırt**, ve "birikim içinde tekrar
kanal oluşumu" imkânsız.

**Çözüm:** yönlendirme yüzeyini `filled` yerine `max(filled, bed)` yap ve route
döngüsünün içinde `avulsionInterval` adımda bir yeniden türet.

İki özellik bunu tahmin değil, güvenli kılıyor:

- Çökelme olmamış her yerde `filled >= bed`, yani yüzey **bit-birebir aynı**.
  Daha önce reddedilen düz-hücre yedeğinin ürettiği baskılı-devre deseni bu
  yolla üretilemez.
- Bir hücre **hiçbir zaman** bütün komşularının üstüne çıkıp terminal sink
  olamaz, çünkü `route` bir geçişteki çökelmeyi `depositionSafety * (en alçak
  komşuya düşüş)` ile sınırlıyor, safety < 1. ★ Yani geçen partide eklenen
  **aşağı-akış sınırı bu geçişin ön koşulu** — o olmadan avulsiyon hücreyi
  mühürleyip yükünü sonsuza kadar hapsedebilirdi.

Debi (`areaA`) bilerek donuk bırakıldı: kanalda büyüyen bir bar **suyun nereye
gittiğini** değiştirir, **ne kadar olduğunu** değil. Ayrıca route adımı başına
multigrid çözümü demek olurdu.

### 2. ★★★ Taze çökel yayılmıyordu

`route` yükü tam suyun olduğu yere bırakıyordu, o yüzden yelpaze dar bir sırt,
yamaç dibi eteği ise hiç oluşmuyordu. Gerçekte vadiden çıkan akarsu
sınırlanmasını kaybeder, çökel kanal-ölçeğinde bir eğimi tutamaz ve 1–5°'lik
koniye gevşer.

**Çözüm:** `terrain_lem_alluvium.comp` — talus'un iki değişiklikle tekrarı:

1. Duraylı açı **alluvial** açı (varsayılan 2°), kaya şev açısı değil.
2. ★★★ Yalnızca **gevşek** malzeme hareket edebilir; kaynak hücrenin alluvium
   kalınlığıyla clamp'li. Bu clamp olmadan geçiş "bütün araziyi 2°'ye gevşet"
   demektir — biraz yanlış görünmez, **dağları eritir**. Clamp bu geçişi bir
   sediment süreci yapan şeydir, düzleştirme filtresi değil.

Alluvium alanı çökelme defterinden **ayrı** tutuldu. Aynı tampon olsaydı
yayılma kendini ikinci kez deftere yazar ve kapanış kontrolü tam yelpazenin
büyüklüğü kadar sahte bir kaçak raporlardı.

`consolidation` hareketliliği azaltır, **yüksekliği değil** — o olmadan
alluvium yalnızca büyür ve geç iterasyonlar erken iterasyonlardan daha
agresif yayar.

### 3. Tane boyu tek — AÇIK

Bir `settlingVelocity` ⇒ çakıl, kum ve silt aynı yerde düşüyor ⇒ **dereceli**
yelpaze (tepede çakıl, uçta silt) yok. Kütle defteri fraksiyon başına kapanır,
mimari değişmez; maliyet route'u fraksiyon sayısıyla çarpar.

### 4. Yanal / taşkın bileşeni yok — AÇIK

`route` yükü yalnızca aşağı ağırlıklar boyunca taşıyor ⇒ doğal levee ve taşkın
ovası yok. Yayılma geçişi bunun bir kısmını üstleniyor ama **eğim-güdümlü**,
akış-güdümlü değil.

## ★★★ Ölçü aleti

Kütle defteri bu partiyi **ölçemez**: aynı hacimli bir sırt ile bir yelpaze
defterde birebir aynı görünür. Bu yüzden `HydraulicErosionStats`'a şekil
sayıları eklendi (`depositedCells`, `depositedAreaFraction`,
`deepestDepositMeters`, `meanDepositMeters`) ve **birlikte** okunmalılar:

- **Sırt**: dar alan, `peak/mean` 10+
- **Yelpaze**: geniş alan, `peak/mean` 2–4

★★★ Sinsi okuma: **yüksek alan oranı + düşük oran**. Bu yelpaze değil, yayılma
geçişinin gevşek-malzeme sınırından kaçıp anakayayı düzlemesidir — ve manzara
bundan daha yumuşak, daha hoş çıkar. Kimse bunu bug diye bildirmez.

Tek değişkenli ayrım: `scripts/ipc/Probe-DepositShape.ps1`. `avulsion_interval
= 0` **birebir** eski davranışa döner, yani A/B gerçek bir kontrol grubudur.
