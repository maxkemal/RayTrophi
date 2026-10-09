# DEM uyuyan taneler

> **Durum:** AKTİF — 2026-10-09. Revizyon 21 canlıda static/wet sayaçlarını kaybetti.
> Revizyon 22 kullanıcı build'inde **sleep/static/settle/wet 4/4 PASS**.
> Revizyon 23 canlı: kısa sleep PASS; dört repose A/B açı paritesi PASS, açık
> tane boyutu kapısı 4.04849° > 4° FAIL. 1.3M ölçek PASS; duvar hızlanması 1.125x.
> Tam kabul açık. Son rapor: [grain_sleep_ab_rev23/REPORT.md](grain_sleep_ab_rev23/REPORT.md).
> Revizyon 24 kaynakta hazır; alıcı kütlesi/ataleti ve aktarılan itkiye göre yerel
> audit uygulanıyor. Build ve canlı kabul bekliyor; tam destek sertifikası ve
> aktif hesap alanı henüz uygulanmadı. [Güncel plan](DEM_SLEEP_TRANSFER_IMPLEMENTATION.md).
> Aşağıdaki ilk tasarım revizyon 21'i anlatır; düzeltmeler sondaki bölümlerdedir.

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

## Revizyon 22 — sayaç, denge ve uyandırma düzeltmesi

**Canlı kanıt (revizyon 21, diğer ajan):** uyku açık base/history/settle/dense PASS;
static son alt adım `sticking_contacts=0`, wet `liquid_bridges=0` yüzünden FAIL.
Uyku kapalı static/wet PASS. Static tanesi yerini tutuyor: ilk kırılma kesinlikle
ölçümün kaybolmasıdır. Repose uyku açıkken 590 s'de yalnız bir kolu bitirdi;
uyku kapalı iki kol bitti. Bu süre farkının tam nedeni henüz ayrıştırılmadı;
yalnız sayaç düzeltmesiyle hız sorununun çözüldüğü iddia edilmiyor.

Kaynakta doğrulanan hatalar ve yeni davranış:

1. **Erken return bütün muhasebeyi atlıyordu.** Uyuyan tane son substep'teki temas,
   yapışma ve köprü sayaçlarına, ayrıca sonraki kare CFL temas ipucuna girmiyordu.
   Yeni yol son alt adımda gerçek contact/bridge hesabını çalıştırır. Diğer alt
   adımlarda hızlı kopya yolu kalır. Uzun karelerde de en fazla 20 ms aralıkla
   denetim yapılır (alt adım daha büyükse her alt adım). Denetim hizası taneye
   göre dağıtılmaz; uyuyan GPU wave'lerinin topluca erken dönmesi korunur.
2. **Hız tek başına mekanik denge değildi.** Uykuya giriş artık net kuvvet ve
   tork artığını, hareketli temas/bridge bağını ve güncel coupling satırlarını
   da kontrol eder. Uyuyan taneye bu karede drag/lift veya dışarıdan hız/dönme
   gelirse audit beklemeden uyanır. Denetimde denge bozulduysa entegre edilir;
   düşük hızlı yeni dengesizlik yakın temas/bridge komşularına WAKE yayar.
3. **Yüksek N'de tam sıfır kuvvet istenmez.** Temel tolerans `sleep_speed/time`.
   Float konum hassasiyetinin normal yay kuvveti belirsizliği için
   `2 * position_ULP * stiffness * inverse_mass * contacts` sınırı kullanılır;
   yüzey açısal ivme belirsizliği küre ataleti nedeniyle bunun 2.5 katıdır.
   Lineer tolerans en fazla `0.25*|g|`, açısal yüzey toleransı `0.625*|g|` olur
   (`g>0` ise). Böylece küçük/sert tanelerin sayısal artığı bütün uykuyu
   engellemez, gevşek hız/süre ayarı da desteksiz tam `g` düşüşünü donduramaz.
   **Bu toleransın repose ve büyük-N davranışı canlı kabulde ölçülecek.**
4. **Uyandırma yalnız üst üste binen hızlı komşuyla sınırlıydı.** Aktif sıvı
   köprüsü artık yüzler ayrıkken de WAKE taşır. Host, yerçekimi/kuvvet yasası,
   domain sınırları, uyku ayarı, substep süresi veya collider fingerprint'i
   değişince bütün dinlenme sayaçlarını uyandırır. Önceki konteks yalnız başarılı
   yayınla ilerletilir; tane başına yeni host taraması/GPU bankası yok.
5. **Rest yazımı iki atomik işlem arasında 0 gösteriyordu.** Eski komşu kontrolü
   bu aralıkta "henüz uyumuyor" deyip WAKE bırakmayabiliyordu. Yeni komşu doğrudan
   `atomicOr(WAKE)` yapar. Sahip tek CAS döngüsüyle sayacı değiştirir; arada
   bırakılan WAKE korunur. Uyku kapalıysa sayaç tek atomicExchange ile temizlenir.

`sim_matter_grain_sleep.glsl` ortak shader policy modülüdür. Descriptor **17**,
push constant **128 B**, history bankası **672 B/tane + aynı block metadata** ve
`4*substeps` dispatch sayısı değişmedi. GPU/CPU script/IPC/UI aynı mevcut core
uyku ayarını ve istatistikleri kullanır. Varsayılan uyku açık kaldı; kabul kapanmadı.

**Maliyet okuması:** 384 substep / 16.7 ms karede yerleşmiş tane bir kez contact
denetimi yapar, kalan 383 adımda kopya yolu sürer. Bu teorik hesap, ölçülmüş hızlanma
değildir. `cost_last_substep` artık bütün tanelerin gerçek audit temaslarını içerir;
bu sayaçların azalmasını bekleme. `gpu_wait`, sleeping fraction ve toplam duvar
süresini A/B karşılaştır. Host merge/order/upload/download maliyetleri bu partide
değişmedi; GPU kazancı toplam kare süresine birebir yansımayabilir.

Doğrulama: `check_matter_grain_contracts.py` ve
`check_matter_grain_sleep_contracts.py` kaynak/spec kontrolleri PASS; CAS yarış
sıraları, audit aralığı, float ULP ve desteksiz/rotasyonel denge örnekleri denetlendi.
Bunlar **GLSL derlemesi/GPU sayısal kabulü değildir**. Yeni dış IPC probu
`rt_test_grain_sleep_ipc.py` kaynakta (koşulmadı); sleep açık/kapalı serbest düşüş,
gerçek uyku ve son alt adım temas/yapışma/CFL muhasebesini zorlar.

### Revizyon 22 canlı sonuç — kullanıcı koşusu (2026-10-09)

`rt_h1_grain_suite.py --only sleep static wet settle`: **4/4 PASS**.
Sleep 33 s, static 250 s, settle 12 s, wet 64 s. Yeni prob desteksiz düşüşü,
gerçek uyumayı ve uyuyan tanenin temas/yapışma/CFL muhasebesini doğruladı.
Static kayan kol son 1 s'de 0.146423115 m kaydı; discriminator korunuyor.
Settle: kütle/COM/RMS drift 0, enerji 6.690211889e-15 J/tane, resident tail 5 örnek.
Wet: cohesion RMS 0.204118119 m, köprü 2680; grain water 0.837685364 kg,
toplam su drift 5.488811610e-7 kg. Önceki static/wet kırılmaları bu kapıda giderildi.
Repose'nin süre/açı paritesi ve 1.3M hız kazancı henüz bu sonuçla kanıtlanmadı.


### Revizyon 23 — gereksiz komşu wake/atomik maliyeti (kaynak hazır; build bekliyor)

Revizyon 22 canlı A/B kayıtları `docs/dev/grain_sleep_ab_rev22/` altında.
Repose uyku açık `mu_r_.05` saçılma 0.317333 > 0.30: FAIL. `mu_r_.3` açıları
30.03675° açık / 32.83456° kapalı, fark 2.79781° > 2°: FAIL. Kapalı iki kolun
kendi fizik kapıları PASS. Bu yüzden dört hızlı PASS, tam uyku kabulü değildir.

1.3M açık koşuda 90. kare sleeping=30, KE=155.0077 J; 180. kare sleeping=420072
(%32.31), KE=0.0627299 J, mass=7434.8235 kg. Son enerji öteleme RMS hızını
4.108 mm/s yapar; görünürde sabit olması 2 mm/s + yüzey dönme kapısını geçtiği
anlamına gelmez. Eşik veya repose kabul sınırları gevşetilmedi.

Kaynakta bulunan somut sorun: hızlı/dengesiz tane zaten uyanık her komşuya WAKE
bırakıyor; bu adayın dinlenme sayacını tekrar başlatıyor ve aktif yoğun yığında
çok fazla atomik yazı üretiyordu. Revizyon 23 coherent rest okumasından sonra
sadece uyuyan komşuya atomik WAKE bırakır. Uyanık aday zaten her alt adımda gerçek
relative motion/kuvvet/tork uygunluğunu hesaplar. Pozitif rest CAS yazımı WAKE'i
korur; uyuyan sahibin sayacı geçici 0 göstermez. Sıfır rest yazımı zaten uyanık
sahipte gereksiz atomik işlemi atlar. Wake okuması yalnız bit varsa atomicAnd
kullanır; okumadan sonra gelen wake sonraki alt adıma korunur. Uyku kapalıyken
rest atomikleri/denge hesabı atlanır; yeniden açılınca host context wake geçerlidir.
Descriptor/banka/dispatch/push ABI değişmedi. Shader ve host revision=23.

Kaynak/spec kontrolleri PASS; bu sonuç derleme veya GPU kabulü değildir. Çalışan
uygulama revizyon 22 kalır; revizyon 23 için kullanıcı shader + C++ build'i gerekir.
Sonrasında sleep/static/wet/settle, motion/coexist, dört kollu repose A/B ve aynı
1.3M 180-kare maliyet/enerji/bounds A/B yeniden koşulmalı. Repose hatasının bu
maliyet düzeltmesiyle giderildiği henüz iddia edilmiyor.

DEM + granular MPM aynı Matter domain'de çalışıyor: Sand DEM + Soil MPM dış IPC
`rt_test_matter_transport_ipc.py --owners-only` PASS; 17 temas, impulse=0.13090269
Ns, residual=1.16415e-10 Ns, ortak saat 130 transport / 7 continuum grid adımı.
Bu elastik MPM veya üç-sahipli sıvı kabulü değildir: varsayılan tam probda 12
örnekte liquid support hiç aktif olmadı ve prob FAIL. Ayrı teşhis açık kalır.


**Tamamlanan 1.3M revision 22 A/B:** 180 adım; iki koşuda ölçek kapıları PASS.
Son 20 medyan duvar 5.225 / 6.530 s,
GPU wait 4660.9 / 5451.4 ms (açık/kapalı).
Açık sleeping 420072/1.3M; duvar hızlanması 1.250x.
Repose FAIL nedeniyle tam uyku kabulü kapanmadı.
Rapor: [grain_sleep_ab_rev22/REPORT.md](grain_sleep_ab_rev22/REPORT.md).


**Revizyon 23 kısa canlı uyku kapısı — 2026-10-09: PASS.**
`python scripts/test/rt_test_grain_sleep_ipc.py`, dış IPC; shader revision=23.
Gevşek uyku ayarında desteksiz düşüş açık/kapalı paritesi PASS, sleeping=0.
Yerleşmiş tek tane: kapalı sleeping=0, açık sleeping=1; her ikisinde son-substep
contacts=1, sticking=1, max_contacts_per_grain=1 ve contactless=0. Prob kendi
source/domain'ini temizledi. Wet/motion/repose/large-N revizyon 23 kapıları bu kısa
koşuda çalıştırılmadı. Log: `grain_sleep_ab_rev22/sleep_rev23_quick.log`.


**Revizyon 23 tam repose + 1.3M A/B (2026-10-09):** Dört açık/kapalı açı paritesi
PASS; açık tam repose tane boyutu kapısı 4.048489° > 4° FAIL,
kapalı 3.323032° PASS. İki koşunun kendi kol kapıları PASS.
1.3M 180 adımlık ölçek kapıları iki koşuda PASS; son 20 medyan duvar
5.750 / 6.470 s (açık/kapalı),
hızlanma 1.125x; açık sleeping 490007/1.3M.
Tam kabul kapanmadı. Kaynakta hareket veto'su `balanced` üzerinden kuvvet dengesizliği
gibi wake yayıyor; enerji/itki/destek ayrımı henüz uygulanmadı. Bu tur core kaynak
şartları değiştirilmedi. Rapor: [grain_sleep_ab_rev23/REPORT.md](grain_sleep_ab_rev23/REPORT.md).


### DEM uyku aktarımı — revizyon 24 (kaynak; build/canlı kabul bekliyor)

Hareket veto'su/gerçek kuvvet dengesi ayrıldı. Komşu bildirimi audit talebi;
dengeli hedef yaşını korur. Post-damping/Coulomb reaksiyon değişimi + bounded
forecast hedef kütle/ataletine göre süzülür. Rest fiziksel saniye, CFL dt context
wake nedeni değil; skipped history süresi tutulur. Metadata +4 B/capacity;
17/128 ve 4 dispatch/substep aynı. Aktif frontier E2 henüz yazılmadı.
Sıra ve sınırlar: [DEM_SLEEP_TRANSFER_IMPLEMENTATION.md](DEM_SLEEP_TRANSFER_IMPLEMENTATION.md).
İlk kullanıcı kapısı: `python scripts/test/rt_h1_grain_suite.py --only sleep sleep_transfer`.
Rev23 canlı rakamları rev24 kabulü değildir. Bağımsız diğer ajan görevi:
[HANDOFF_S1B_CELL_FIELDS.md](HANDOFF_S1B_CELL_FIELDS.md).
