# DEM uyuyan taneler

> **Durum:** AKTİF — 2026-10-09. Yazıldı, derlenmedi (shader revizyonu 21). Kabul: NEXT_BUILD_CHECKS Parti 10.

## Neden

Durgun bir yığında her tane her substep'te tüm temaslarını yeniden hesaplıyor ve
sıfıra yakın bir hızı entegre ediyor. 1.3M tanelik dökmenin sonunda yığının büyük
kısmı hareketsiz; GPU süresi (kare başına ~0.95 s) bu tanelere harcanıyor. Uyuyan
tane ne temas hesaplar ne entegre olur, yalnız durumunu öbür banka kopyalar.

## Kural

- **Uyku:** tanenin hızı ve `|ω|·r` değeri `sleep_speed_m_s`'nin altında,
  `sleep_time_s` boyunca (substep sayısına çevrilir: `ceil(sleep_time_s / substep_dt)`)
  kesintisiz kalırsa uyur. Eşik **mutlak birimde** (m/s): bir hareketin özelliği,
  çözücü ayarının oranı değil.
- **Uyuyan tane:** konumunu tutar, hızı ve dönmesi tam sıfırlanır. Uyanık komşuları
  onu duran bir cisim olarak görür (temas kuvveti tek taraflı hesaplanır). Temas
  geçmişi dondurulur (maske ve kayıtlar değişmez).
- **Uyanma:** hızlı (eşiğin üstündeki) bir tane, uyuyan bir taneye temas ettiği
  substep'te onun sayaç kelimesine `WAKE` biti koyar; uyuyan tane bir sonraki
  substep'te uyanır. Bir substep gecikme: o substep'te uyuyan tane sabit duvar gibi
  davranır (substep ~0.1 ms).
- **Hiç uyumaz:** sıvı sürüklemesi (`drag.w > 0`) ya da kuvvet alanı/kaldırma
  (`lift` satırı sıfır değil) altındaki tane — bunlar kareden kareye değişir.
- **Karede kapalı:** hareketli çarpıştırıcı yüzü varsa (uyuyan tane yüzün içine
  gömülürdü), MPM teması etkinse (impulslar adımlar arasında hızlara yazılıyor,
  uyuyan tane onları silerdi), ortak saatte.
- **Herkes uyanır:** yayımlanmış bir tane kaybolduysa (emme/silme) — bir şeyin
  dayanağı olabilir; uyuyan tane havada kalırdı.

## Depolama

Sayaç tanenin temas geçmişi bloğunda: blok başına {sahip, maske, dinlenme}. Blok
tanenin kimliğini izlediği için sayaç sıralama değişikliklerinde tanenin peşinden
gider; geçmiş sıfırlanınca (FRESH) sayaç da sıfırlanır.

Yarış: komşu yalnız `WAKE` bitini `atomicOr` ile koyar. Sahip tane substep başında
`atomicAnd(~WAKE)` ile okuyup temizler, sonunda `atomicAnd(WAKE)` + `atomicOr(sayaç)`
ile yazar — arada konan bir `WAKE` korunur ve bir sonraki substep'te okunur.

## Bilinen sınırlar

- Eşikten yavaş sürünen bir eğim donar. Eşik 2 mm/s ve süre 0.2 s ile bunun
  gerektirdiği ivme ~0.01 m/s²; statik test kolu (0.146 m/s sürünme) uyanık kalır.
- Uyuyan tane yerçekimi değişimini görmez; ağır yük altında yavaşça artan basıncı da
  görmez (yükü taşıyan tane yavaşsa uyandırmaz). Yığın büyürken yeni gelen taneler
  hızlıdır ve temas ettiklerini uyandırır.
- CFL temas ipucu (`diagnostics[2]`) uyuyan taneleri saymaz; uyananlar ölçülenden
  fazla temas getirirse mevcut yeniden koşma mekanizması devreye girer.

## Ayarlar (script + panel)

`fluid.set_grain_settings(sleep=..., sleep_speed_m_s=..., sleep_time_s=...)`;
panelde Solvers → Advanced solver. `sleep=false` A/B referansı.
Ölçü: `grain_diagnostics.runtime.sleeping_grains` (son substep).
