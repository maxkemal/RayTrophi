# Sıradaki kullanıcı build (C++ + shader): tane için force field ve hareketli collider

Diğer çözücüler gibi tane de artık force field'ları ve hareket eden collider'ları hissediyor.
Öncesinde grain domain'i bunlardan biri varken adımı tutuyordu ("static colliders, gravity only").

**Ne değişti:**
- **Force field:** her tane için kare başına `force_snapshot->evaluateAt(..., Fluid)` (sıvıyla aynı
  alanlar ve affects_fluid maskesi); DEM alt adımları boyunca sabit ivme (lift satırına eklenir).
  Sıvı şeridi: paketlenebilir alanlar GPU force pass'ine (önceden `nullptr` geçiliyordu), CPU-only
  olanlar (noise, wind surface drag) host'ta + upload — tane'siz domain ile aynı yol.
- **Hareketli collider:** her collider'ın yüzleri ayrı aralık; köşe hızı = (şimdiki − önceki kare
  köşesi)/dt → katı, dönen ve iskeletli (bone anim) mesh aynı yoldan. Önceki köşe yalnız zamanda
  bir kare geriyse geçerli (scrub/reset sıçraması hız sayılmaz; o zaman collider'ın kendi doğrusal hızı).
- **Kemik proxy'leri** (ayak/el kapsülleri, `physics.collider.proxy_set.*`, Granular tüketicisi):
  8×4 küre/kapsül ya da dönük kutu olarak üçgenlenir, domain'e ulaşanlar alınır; hız = linear +
  angular × (x − merkez).
- **Shader (rev 15):** yüz, alt adımda `son − v·(kalan süre)` konumunda (kare içinde süpürür, kare
  başı sıçramaz); temas göreli hızı `v_tane − v_duvar` (barisentrik) → ayak taneyi iter, sürtünme
  sürükler. BVH düğümleri süpürmeyi kapsar. Travel CFL'i collider hızını da sayar.
- Hâlâ tutulan tek şey: **hareketli domain**.
- Rapor: `grain_diagnostics.runtime.collider_faces`, `collider_speed_max_m_s`,
  `field_acceleration_max_m_s2`.

Sıra:
1. `python scripts/test/check_matter_grain_contracts.py` → PASS (bulutta PASS).
   `matter_grain_collider_bvh_test.cpp` bulutta PASS (süpürme sınırları + hız sırası).
2. `python scripts/test/rt_grain_motion_ipc.py` (yeni; x64'e de kopyala) → `PASS grain force fields + moving collider`.
   RESULT satırını getir: wind kolu COM x > rest + .05, rüzgâr ivmesi ≈ 4, küre hızı ≈ 1 m/s, sweep yığını iter.
   ★ Sinsi: `collider_speed_max` ≈ 1 ama COM kıpırdamıyorsa göreli hız shader'a ulaşmıyor (rev 15
   yüklenmemiş = eski SPIR-V; exe/shader zaman damgası).
3. Kendi sahnen: rüzgâr force field + kum → savrulmalı; bone anim ayak proxy'li karakter kumda
   yürüyünce kum itilmeli/iz kalmalı. Proxy sayısı fazlaysa "4096 faces" hatası (o zaman proxy azalt).
4. Suite regresyonu: `python scripts/test/rt_h1_grain_suite.py --quick` → hepsi PASS (statik
   sahnelerde hız yok: `triangle_velocity` boş, eski yol birebir).

---

# Sıradaki kullanıcı build (YALNIZ C++): kum–su basınç geri beslemesi söküldü

**Ölçüm (`rt_grain_float_probe.py --compare --frames 120`, 24 fps, ikisi de 0. kareden):**

| | toplam ms | sıvı alt adım | maks m/s | p99 m/s |
|---|---|---|---|---|
| etkileşim açık | 420.99 | 68 | 50.44 | 7.51 |
| etkileşim kapalı | 146.73 | 12 | 7.92 | 7.08 |
| **düzeltme sonrası** açık | 164.12 | 11 | 7.73 | 7.02 |
| düzeltme sonrası kapalı | 140.55 | 12 | 7.90 | 7.07 |

✅ Madde 2 canlı PASS (2026-10-07): uç hızlar kalktı, etkileşimin ek maliyeti ~24 ms (%17).

**Sebep:** volume exclusion açıkken tanenin basınç kuvveti sıvının ölçülen ivmesinden geliyor
(ρV(Du/Dt − g)) ve tepkisi sıvıya darbe olarak da veriliyordu. Ama taneler poröz projeksiyonda
zaten engel; projeksiyon darbeyi geri alıyor, sonraki kare bunu sıvı ivmesi diye ölçüyor ve kuvvet
hücredeki tane/sıvı hacim oranıyla büyüyor. Kum hücrenin sıvıdan fazlasını doldurunca (paketleme .6
> gözenek .4) sınırsız: 50 m/s uç parseller → 68 alt adım → 3× maliyet. Aynı döngü taneye büyük
yukarı kuvvet verir: **dalga üzerinde kalan kum topaklarının da muhtemel sebebi.**
**Düzeltme:** volume exclusion açıkken sıvıya yalnız sürükleme darbesi döner; basınç tepkisi
projeksiyonda (çift sayım yok). Hidrostatik kipte (exclusion kapalı) değişiklik yok.

**Ek (kullanıcı bildirimi): "grain buffers exceed the domain…" ile adım tutuluyor, taneler donuyor.**
Sebep aygıt değil, domain kaynak bütçesi: Interactive 512 MiB, Preview 1024 MiB; sıvı şeridi önce
alır, tane kalanı kullanır (tane başına ~3 KB, büyürken eski+yeni ~2.5×). Bütçe panelde yalnız
Quality Profile ile değişiyordu (script'ten yazılabilir, panelden değil — kural ihlali). Şimdi
Quality Profile altında "Enforce Resource Budget" + "Resource Budget (MiB)" (değiştirince profil
Custom olur). Hata metni hangi sınır olduğunu, tane sayısını, gereken ve kalan MiB'i söylüyor;
aygıt sınırı ayrı mesaj. Not: ayrıca sabit 100000 tane üst sınırı var (ayrı mesaj).

**Ek (kullanıcı isteği): ıslanan tanenin koyulaşması.** MPM'in ıslak görünüm sistemi (8 doygunluk
bandı, Principled kopyaları) parçacık başına `pore_water_mass_kg / pore_capacity_kg` okur; Wet grains
bu ikisini tanelerde zaten yazıyor. Eksik olan paneldi: grain açıkken Pore Water bölümü tümden
gizleniyordu. Şimdi Matter sekmesinde "Wet Look": "Saturation darkens…" + renk/pürüzlülük çarpanı +
tam ıslak doygunluk. MPM fiziği (emme, Darcy, ıslak dayanım) grain'de kapalı kalır.
Görmen gereken: Wet grains + bu kutu açık → suya giren kum koyulaşır, kuruyunca açılır.
★ Hiç koyulaşmıyorsa: Wet grains kapalı (film suyu 0) ya da tanenin malzemesi Principled değil.

Sıra:
1. `python scripts/test/check_matter_grain_contracts.py` → PASS.
1b. Domain paneli: Quality Profile altında iki yeni alan; bütçeyi büyütünce tutulan adım bir sonraki karede yürümeli.
2. Aynı sahne: `python scripts/test/rt_grain_float_probe.py --compare --frames 120`.
   Görmen gereken: coupled satırında maks m/s ≈ p99'un ~1–2 katı (≈8–15), sıvı alt adım 68 → ~12–20,
   toplam ms uncoupled'a yakın (~150–200). Hâlâ 50 m/s ise sebep başka (sürükleme tepkisi) — getir.
3. Gözle: kum suya düşünce batmalı/çökelmeli, dalga sırtında topak kalmamalı.
   ★ Sinsi: kum hiç batmıyor ama "sakin" görünüyorsa sürükleme baskın — `rt_grain_float_probe.py`
   (parametresiz) üç kolunu koş.
4. `python scripts/test/rt_h1_grain_suite.py --only coexist coexist_hydrostatic porous wet` → PASS.
   coexist ivmesi eklenen kütle kapsamı düşebilir (gate [4.90, 6.31] içinde kalmalı); FAIL ise getir.

---

# Sıradaki kullanıcı build (YALNIZ C++): tane adımının CPU maliyeti — önce ölç

Kullanıcı gözlemi: GPU tane adımı sırasında CPU kullanımı periyodik inip çıkıyor. Kodda CPU her
kare: sahip ayırma + Morton sıralaması (N log N) + alt kümelerin TAM kopyası (sıvı ve tane için
ayrı `FluidParticles` kopyası + compact), kimlik eşlemesi için `unordered_map` (tane başına
düğüm ayırma), tüm durumu GPU'ya yükleme (sıra her kare değiştiği için `state_resident` false),
GPU'yu bekleyen senkron indirme (16k'da ~1 MB), yayın kopyaları ve geri birleştirme yapıyor.
Bu parti **ölçer** ve en bariz israfı söker:
- `fluid.matter_models → grain_diagnostics.runtime.host_ms` {order, prepare, gpu_wait, publish,
  merge}; dense tablosunda medyanları.
- Yayında tüm `FluidParticles`'ın kare başına kopyası (`auto result = p`) kaldırıldı: yalnız
  GPU'nun sahip olduğu 3 dizi indirilir ve taşınır.

**Ek (kullanıcı bildirimi):** yarıçap küçültülünce taneler doğduğu yerde kalıyor. Sebep: kütle
r³ ile düşer, sertlik N/m sabit kalır → alt adım ~r^1.5 kısalır (r .025→.01: ~196→~770 alt adım
> max_substeps 512) → adım tutulur, taneler donar; sebep yalnız gri bir satırdaydı. Şimdi:
Solvers'ta "Step held - grains do not move: …" normal renkte; CFL hata metni gereken alt adımı ve
sığan sertliği sayıyla verir ("Lower Contact stiffness to <= X N/m"); yarıçap tooltip'i ilişkiyi
anlatır. Kalıcı çözüm açık iş: örtüşmeden türetilen sertlik (yarıçap değişince kendiliğinden doğru).

**Ek 2 (kullanıcı bildirimi):** yeni Matter domain + Sand point source + Granular → "Enable discrete
grains" reddediliyor, altında iki kez "Solid phase cannot run with grains yet". Sebep: katı faz anahtarı
varsayılan AÇIK ve kural maddeye bakmadan engelliyordu; anahtar yalnız Solid bağlı bir madde varsa
iş yapar (FluidDomainStep aynı şekilde okur). Kural artık yalnız Solid bağlı madde varsa engeller ve
nereden kaldırılacağını söyler; reddedilen enable'ın hata metni engel satırlarını tekrar basmıyor.

**Ek 3 (kullanıcı senaryosu):** kum + su emitter çalışırken collider eklenince iki emitterin
parçacıkları donuyor. Sebep: tane adımı yalnız PlaneY ve mesh collider kabul ediyordu, diğerleri
adımı tutuyordu. Şimdi Sphere/Capsule üçgenlenir (24×12), ObjectOBB/ConvexDecomp nesnenin yüzeyi,
ObjectAABB o yüzeyin dünya kutusu. Ayrıca dolu domain'de tane ayarlarının HEPSİ reset istiyordu
(ıslak tane seçimi bu yüzden reddediliyordu); artık yalnız doğumda sabitlenenler (enabled,
radius_m, packing_fraction, wet_grains kapatma) reset ister, gerisi sonraki adımda uygulanır.

**Ek 4:** (a) Proje kaydı tane ayarlarını yazmıyordu (yalnız SceneSerializer yazıyordu;
ProjectManager domain'leri ayrıca yazar) → açılışta "Enable discrete grains" kapalı, kum MPM gibi.
Şimdi `matter_grain` proje dosyasında; kontrat bunu kontrol ediyor. (b) Dalga üstünde kalan kum
topakları için tanı: `python scripts/test/rt_grain_float_probe.py` — açık sahnede 3 kol (olduğu gibi /
volume_exclusion kapalı / fluid_coupling kapalı), ayarları sonunda geri koyar; tabloyu getir.

**Ek 5 (maliyet):** kum+su sahnesinde CPU sürekli ~%8 meşgul, GPU dolu çalışmıyor. `host_ms`'e
`coupling` eklendi (sıvı alanı, sürüklenme/basınç hazırlığı, tepki, su alışverişi). Ölçüm:
`python scripts/test/rt_grain_float_probe.py --cost --frames 120` → kare süresinin sıvı aşamaları,
tane host aşamaları, aktarım ve senkron noktaları (medyan). Tabloyu getir; GPU'nun boş kalması
genelde CPU'nun bu senkron noktaları arasında bekletmesidir.

**Ek 6 (canlı maliyet, kum+su 5832 parçacık):** kare 99.7 ms: `batch_end` 45.3 ms (33 senkron),
yükleme 26.75 MB/kare, tane GPU 12.6 ms, bağlama host 10.3 ms. Bağlamanın dört fonksiyonu
(sıvı alanı, gözeneklilik, basınç kuvveti, su alışverişi) her kare ızgara boyutunda diziler ayırıp
sıfırlıyor ve tüm ızgarayı dolaşıyordu; gözeneklilik dört ızgara dizisini tamamen yedekleyip geri
yazıyordu. Şimdi: kalıcı tamponlar + yalnız yazılan hücrelerin temizliği, yalnız tanelerin
değdiği yüzler/hücreler değişir ve yedeklenir (sonuç aynı: C++ testi + seyrek geri yüklemenin
bit-bit eşitliği PASS). Beklenen: bağlama 10 ms → ~1 ms. 26.75 MB yükleme ve 33 senkron sıvı
şeridinin ızgara yükleme yolunda — sıradaki iş (bu ölçümle hangisinin kaldığı görülecek).

**Ek 7 (kernel ölçümü):** bağlama 10.3 → 3.4 ms (canlı). GPU 57 ms/kare: sıvı basınç CG ~36 ms
(~10 basınç çözümü/kare × ~19 iterasyon), tane DEM 12 ms. Sıvı şeridi kare başına ~10 alt adım
koşuyor: CFL en hızlı tek parselden. Şimdi `grain_diagnostics.liquid` {speed_max_m_s,
speed_p99_m_s, liquid_substeps} ve `--cost` bunları basar. max ≫ p99 ise birkaç uç parsel
(muhtemelen tane tepki itkisi az parselli hücrelerde) tüm karenin basınç çözümlerini çoğaltıyor.

**Build:** yalnız C++.

00000. Aynı sahne, `python scripts/test/rt_grain_float_probe.py --cost --frames 120` → `coupling`
       ~1 ms olmalı; tabloyu getir (önce: 99.7 ms toplam).
0000. Kum+su sahnesini kaydet, kapat, aç → Solvers'ta grains açık kalmalı.
000. Kum + su + Sphere/Box collider: taneler ve su collider'a çarpıp akmalı; Solvers "Ready.".
     Oynarken Wet grains aç → kabul edilmeli, sudaki taneler su emmeli (Measure: grain water).
00. Yeni Matter domain + point source (Sand, Granular) → Solvers'ta enable kabul edilmeli, "Ready.".
0. Yarıçapı .01'e indir, reset, oynat: Solvers'ta held satırı ve önerilen sertlik görünmeli; o
   sertliği gir → taneler düşmeli.
1. `python scripts/test/rt_h1_grain_suite.py --only dense` → tablonun yeni sütunlarını getir.
   Okuma: `gpu_wait` büyükse GPU (ya da senkron bekleme) baskın; `order`/`merge`/`prepare`
   büyükse CPU kopyaları/sıralama — sıradaki adım onları söker (sıralamayı GPU'ya taşımak,
   alt küme kopyası yerine indeks, durumu GPU'da tutmak).
2. `--quick` (regresyon): 13/13 PASS kalmalı.

---

# Sıradaki kullanıcı build (YALNIZ C++): DEM birleşik kararlılık sınırı + coexist eklenmiş kütle

**SONUÇ (2026-10-07): `--quick` 13/13 PASS, dense PASS.** Alt adım 234 → **196** (tahmin
tuttu), dispatch 243 → 205. 16384: medyan 60.1 → **53.4 ms** (−%11), p95 90.4 → **73.1**
(−%19). 4096: 20.4 → 27.3 ms medyan ama p95 aynı (35.5 → 35.7) — o boyutta kare süresini
ek yük/gürültü belirliyor (önceki iki koşu 19.9 ve 20.4). Sıradaki maliyet adımı: 24 temas
bütçesi (ölçülen en fazla 9).

1. **DEM maliyeti.** Sertlik ve sönüm sınırları ayrı ayrı uygulanıyordu; sönüm sınırı .5/oran =
   1/(4ζω), büyük ζ'de birleşik limitin yarısı ve her sönümlü sahnenin alt adımını o belirliyordu
   (16k'da 234/kare). Şimdi tek sınır: sembolik Euler x'' = −ω²x − 2ζωx' için kesin kararlılık
   h < (2/ω)(√(1+ζ²) − ζ), yarısı (ζ = 0'da eski 1/ω, güvenlik payı aynı). Etiket ζ > .1 ise
   `damping`, değilse `stability`. Beklenen: yoğun sahnede ~234 → ~196 alt adım (~%16 hız).
   24 temas bütçesi değişmedi (ölçülen en fazla 10; asıl kazanç orada, ayrı ve riskli iş).
2. **Coexist "aşımı" fizikmiş.** Durgun suda bırakılan küre yer değiştirdiği suyu da ivmelendirir
   (eklenmiş kütle, C_m = .5): gerçek ilk ivme g(m − ρV)/(m + .5ρV) = **5.16**, test yalnız
   kaldırmayla 6.13 bekliyordu. Hidrostatik kol (suyun tepkisini göremez) 6.10 ✓; basınç kolu
   5.49–5.74 = eklenmiş kütlenin bir kısmını yakalıyor. Kapı artık [.95·5.16, 1.03·6.13] ve
   yakalanan pay basılır. Tam eklenmiş kütle terimi ayrı tasarım işi (çözünürlüğü olmayan
   CFD-DEM'de açık C_m terimi + tanenin kendi sürüklediği akışı çift saymama).

Bulutta: değişen C++ sözdizimi PASS, kontrat PASS, descriptor/IPC denetimi PASS.

**Build:** yalnız C++ (shader değişmedi, rev 14).

1. `python scripts/test/rt_h1_grain_suite.py --quick` (yavaşları atlar) → hepsi PASS.
   - `settle`, `base`, `convergence` alt adım sayıları değişebilir; kapılar aynı kalmalı.
   - `coexist`: `coexist added mass: … fraction of added mass captured …` satırı.
2. `python scripts/test/rt_h1_grain_suite.py --only dense` → tabloda substeps ~196 ve limit `damping`,
   ms düşmüş olmalı (önce 20.4 / 60.1).
   ★ Sinsi: dense geçer ama `settle` KE kuyruğu büyür ya da `spin_grows: True` → yeni sınır
   kararlılığın sınırında; sayıları getir.

---

# Sıradaki koşu (BUILD YOK, yalnız script): convergence kum malzemesiyle

**SONUÇ (2026-10-07): PASS.** Sekme e .541 / .539 (100 / 200; Δ .002 → yakınsak), hedef .5:
+%8, eşlemenin sistematik payı — doğrusal yay-sönüm `fn = max(0, kδ − c v)` çekme kuvvetini
kestiği için temas erken biter, analitik ζ(e)'den az enerji kaybeder (bilinen etki). Gerekirse
eşleme sayısal tersine çevrilir; şimdilik ±.05 içinde. Yığın COM %0.18, rms %1.6.
Açık kalan (öneri sırası): coexist basınç itkisi aşımı (%10–25), DEM maliyeti (sönüm CFL).

**Repose PASS (2026-10-07, e .5):** μr .05/.1/.3 → 18.2/23.1/34.8° (monoton), r .0175 → 24.4°
(boydan bağımsız, Δ1.3°), saçılan ≤ .22. Kayıt: roadmap B2.

**Convergence teşhisi:** eski kol e ≈ .96 kullanıyordu (accuracy sınırı bağlasın diye); 64 tane
.6 s sonunda hâlâ zıplıyor, COM anlık değeri kaotik → %6.7 bir yakınsama ölçüsü değil. Yeni kol:
kum (e .5, kayma sönümü 0), çözünürlük 100/200 (ikisi de accuracy sınırlı: 6.6e-5 / 3.3e-5 s;
sönüm sınırı 7.3e-5 s). İki ölçü: (1) **tek tane sekmesi** → gerçek restitution; alt adım yarıya
inince |Δe| ≤ .02 ve e = .5 ± .05 (yeni restitution eşlemesinin canlı doğrulaması — duvar
temasında m_eff = m); (2) 64 tanelik yığın 1 s sonra durmuşken COM/rms ≤ %5.

1. `git pull`, `python scripts/test/rt_h1_grain_suite.py --only convergence` → `convergence rebound e …`
   ve `PASS convergence` satırları.
   ★ e ≈ .35 çıkarsa: duvar sönümü eşlemede m yerine m/2 kullanılmış gibi (√2 fazla sönüm);
   e ≈ .5 ama |Δe| büyükse: açık Euler sönüm terimi alt adıma duyarlı.

---

# Sıradaki kullanıcı build (YALNIZ C++): repose teşhisi

Dünkü FAIL: μr .05 kolunda "yığın" duvarlara kadar yayılmış tabaka; ölçüm `measured=false`
ve test ilk kolda durdu (diğer kolların sayıları yok). İki ayrı sorun ayrıldı:

1. **Ölçüm kırılgandı.** Profil her halkanın EN YÜKSEK tanesini alıyordu; zemine tek kat saçılmış
   taneler ve duvara yığılan küme eğimi düz/pozitif yapıyordu, merkez de dağılmış tanelerle
   kayıyordu. Yeni `grain_diagnostics.pile`: eksen = ilk katın üstündeki tanelerin merkezi
   (duvar kümesi ikinci geçişte dışlanır); yığın = eksenden itibaren 1.5 kattan kalın halkalar,
   `base_radius_m` eteği; dışarıdakiler `scattered_grains/scattered_fraction`; ölçülemezse
   `reason` (no_pile | too_few_slope_rings | slope_not_decreasing), `max_extent_m`. Bulutta
   sentetik koni 20/30/38° → 19.5/29.1/39.4°, 400 saçılmış tane + duvar kümesi eklenince aynı,
   düz tabaka → no_pile.
2. **Test parametresi çok sekken.** Tarihî kollar `restitution_of(8 Ns/m)` = **e .87** (kum ~.5)
   kullanıyor. Yeni kol `mu_r_.1_e_.5`: aynı yığın e .5 ile. Yalnız o yığın kurarsa yayılma
   sekme saçılmasıdır (fizik doğru, test malzemesi yanlış); o da yayılırsa yuvarlanma/sürtünme
   yığında etkisiz → çekirdek hatası.

Test artık bütün kolları koşturur, her kol için tek satır basar (`angle peak base scattered
extent KE/grain reason`), hükmü sonda verir.

**Build:** yalnız C++ (shader değişmedi, revision 14).

1. `python scripts/test/rt_h1_grain_suite.py --only repose` → özetteki 5 `repose …` satırını
   ve `repose angles` satırını getir. FAIL olsa da tablo tam çıkmalı.
   - e .5 kolunda açı 15–35° ve scattered küçük (< .2) → sorun test malzemesi; tarihî kolları
     e .5'e çekeriz.
   - e .5 kolu da `no_pile` → çekirdek; sıradaki iş yuvarlanma yolunun yığın içi davranışı.
   ★ Sinsi: açı ölçülür ama `scattered` > .3 → yığının üçte biri saçılmış; açı yalnız kalan
   çekirdeğin, malzemenin değil.
2. İstersen tam suite (`--quick` yeter); diğer kollar değişmemeli.

Not: `scripts/test/matter_grain_params_test.cpp` zaten derlenmiyordu (`resizeAll` private,
önceki bir değişiklik); pile kısmı bulutta ayrı düzenekte koşturuldu. Ayrı iş olarak düzeltilecek.

---

# ★ YARIN BURADAN DEVAM (2026-10-06 akşamı, rev 14 canlı suite sonucu)

`rt_h1_grain_suite.py` tam koşu: **14/16 PASS**, FAIL: `convergence`, `repose`.
Tam loglar kullanıcının makinesinde `docs/dev/grain_suite/` (git'e girmiyor; gerekirse iste).

**Geçenler ve sayıları:**
- readiness: 6 bozucu düzenleme sebebiyle reddedildi (`solver_kind` dahil). base: g 9.8101 / 9.8100.
- settle: KE/tane 4.2e-13, `spin_grows: False`, resident 5 — yeni kapı doğru çalışıyor.
- **porous: fark .1033 m, beklenen .1023 (+%1)** — geçen tur +%29'du; ε·ppc düzeltmesi oturdu.
- coexist_hydrostatic 6.10 (beklenen 6.13, −%0.5). **coexist (basınç kuvveti) 5.49 (−%10.5)**:
  kapı ±%15 içinde ama geçen tur 5.74'tü, kayıyor. İz satırlarında basınç itkisi Arşimet'in
  %10–25 üstünde dalgalanıyor (.0053–.0067 vs .00535) → tane fazla frenleniyor. İzlenecek.
- wet: rms .205, 2706 köprü, su sapması 1.9e-6 kg. g2_dry_wet PASS.
- dense (DEM tek): 4096 → 20.42 ms (p95 35.5), 16384 → 60.11 ms (p95 90.4), 234 alt adım,
  `damping` sınırlı, en fazla 8/10 temas/tane. XPBD öncesiyle aynı → söküm DEM'e dokunmadı.
  **Maliyet işinin taban çizgisi bu.**

**FAIL 1 — convergence:** alt adım yarıya inince yığın COM'u %6.7 değişiyor (kapı %5; rms %1.4
iyi). Eski kayıt (`matter_h1_grain_convergence_2026-10-06.json`, restitution öncesi) %3.1'di.
Şüpheli: restitution geçişi duvar/zemin sönümünü √2 artırdı (aynı geri sekme, farklı c) — 256
tane küçük yığında COM tek tanelerin zıplamasına duyarlı. Yarın ilk iş: kol iki kez koşturulup
gürültü mü (aynı ayarda iki koşu farkı) yoksa sistematik mi ayır; sistematikse sönüm teriminin
alt adım bağımlılığına (açık Euler'de c·v, dt ile değişen enerji kaybı) bak.

**FAIL 2 — repose (μr .05 kolu):** `measured=false`, peak .090 m, `base_radius_m` 0 (hiçbir
halka 4r = .1 m yüksekliğe ulaşmadı), 22 halka uydu ama eğim ≥ 0 → yığın değil, **duvarlara
(1.2 m) kadar yayılmış tabaka**; kenarda birikince profil dışa doğru yükseliyor. Bu kolun
canlıda daha önce koştuğuna dair kayıt yok (B2 "kaynak, build yok" kalmıştı) — ilk gerçek
koşu olabilir. Yarın: (a) test sırası — `measured` kontrolünden önce duvara ulaşma raporlansın
ki sebep okunur olsun; (b) μr .05 + μ .5 küre için literatür ~20°; yayılma sürtünme/yuvarlanma
yolunun yığında etkisiz kaldığını gösterebilir (tek tane eğim testi geçiyor, kolektif yığın
geçmiyor) → diğer kollar (μr .3/.1) koşmadan durdu, onların sayıları da gerekli:
`--only repose` ilk kolda durmasın diye test tüm kolları koşturup sonra assert etmeli.

**Sıradaki iş sırası (öneri):** 1) repose testini teşhis edilebilir yap + tüm kolları koştur,
2) convergence gürültü/sistematik ayrımı, 3) coexist basınç itkisi aşımı, 4) DEM maliyeti:
sönüm CFL sınırı (24 temas varsayımı → ölçülen/örtük sönüm), 5) madde başına tane malzemesi.

---

# Sıradaki kullanıcı build (C++ + SHADER): XPBD söküldü (H1-G0 kapandı) + settle kapısı

Bu partide (2026-10-06): 16k tablosu anlaşılan ölçütü geçemedi (hiçbir XPBD ayarı DEM'e %5
içinde değil; en yakını .949 @ 80 alt adım, maliyet .53) → **XPBD söküldü**, DEM tek tane
çözücüsü. Shader'dan XPBD dalları, CFL'den XPBD sınırı, ayarlardan `solver_kind` /
`xpbd_substeps` (script'ten gelirse sebebiyle reddedilir, eski sahnede yüklenirken atılır),
panelden "Solver (comparison)" ve "XPBD substeps", testten `--xpbd-compare` ve CPU kopyası
çıktı. Gerekçe + tablo: [MATTER_H1_GRAIN_ROADMAP.md](MATTER_H1_GRAIN_ROADMAP.md) B7.

Settle teşhisi: geçen koşuda ötelenme enerjisi 1e-10'a indi ama dönme enerjisi 4 s boyunca
yavaşça söndü (kapı %12 payla geçti). √E zamanla doğrusal düşüyor = sabit tork = tek bir
tanenin temas normali etrafında dönmesi, burulma sürtünmesiyle (μt·Fn·√(Rδ), ≈2.5 rad/s²)
yavaşlaması. Fizik doğru (masadaki topaç); kapı yanlış ölçüyordu. Kapı artık: ötelenme KE/tane
≤ 1e-5 ve dönme enerjisi kuyrukta **hiç artmıyor** (enerji kaynağı yok). Geçince kapı değerleri
basılır.

Bulutta: grain shader'ları glslangValidator + spirv-val PASS; değişen C++ g++ sözdizimi PASS
(bilinen `sprintf_s` düzenek yan etkisi hariç); kontrat + panel envanteri (254 → 252, iki
bilinçli kaldırma) + IPC denetimi PASS.

**Build:** C++ + `compile_shaders.bat`. Grain shader revision **14**.

**Tek komut (yeni):** `python scripts/test/rt_h1_grain_suite.py` aşağıdaki 0–9'un hepsini
sırayla koşturur (`--quick` yavaşları atlar, `--only settle dense` seçer), tam çıktıyı ve her
kolun JSON'unu `docs/dev/grain_suite/`'e yazar, sonda kısa özet basar. **Yalnız
"==== H1 grain suite summary" sonrasını getir.** Panel (madde 8) gözle kalır.

0. **Statik:** `python scripts/test/check_matter_grain_contracts.py`,
   `python scripts/test/check_domain_panel_fields.py` → PASS (252 now).
1. **Revizyon:** tane koşusunda `shader revision mismatch` yok (varsa shader derlenmedi).
2. **Kilitler:** `rt_h1_grain_runtime_ipc.py --readiness-only` → `PASS grain readiness`;
   artık `solver_kind='xpbd'` de "was removed" ile `rejected` basılmalı.
3. **Temel regresyon:** `rt_h1_grain_runtime_ipc.py`, `--history-only`, `--static-only`,
   `--convergence-only`, `--extended-only`. DEM yolu değişmedi (yalnız XPBD dalları çıktı);
   değerler önceki turla aynı olmalı. Fark = söküm DEM'e dokunmuş, hemen getir.
4. **Yerleşme:** `--settle-only --settle-normal-damping 8` → `PASS settle {...}`;
   `spin_grows: False`, `tail_energy_per_grain_j` ≈ 4e-13 (geçen tur 1.06e-10/256).
   ★ Sinsi: `spin_grows: True` = bir temas tanelere enerji pompalıyor; kapı doğru tutar ama
   sayıları getir.
5. **Su + tane:** `--coexist-only`, `--coexist-only --coexist-hydrostatic`, `--porous-only`
   (önceki tur: 5.74, hidrostatik 9.81, porous fark .132).
6. **Islak:** `--wet-only`.
7. **Maliyet (DEM tek):** `python scripts/test/rt_h1_dense_grain_scene_ipc.py --counts 4096 16384 --steps 120`
   → sonda `count median_ms p95_ms substeps limit contacts max/grain` tablosu. Beklenen geçen
   turun DEM satırları (19.9 / 60.4 ms, 234 alt adım, `damping`). Bu tablo sıradaki maliyet
   işinin (sönüm CFL sınırı) taban çizgisi; getir.
8. **Panel (gözle, 1 dk):** Solvers → Granular Solver → Advanced solver'da "Solver
   (comparison)" yok; Measure → Grain Step "DEM: N substeps … limited by …".
9. **Uzun:** `--repose-only`; `rt_g2_dry_wet_compare_ipc.py --expect-empty-fluid-skipped`.

Önceki parti (B9b) canlıda: **tüm kollar hatasız geçti** (kullanıcı hepsini koştu); panel gözle
temiz (taşan Max Auto Resolution / Boundary Padding satırı düzeltildi); `--xpbd-compare` ve 16k
tablo → XPBD kararı.

---

# Sıradaki kullanıcı build (C++ + SHADER): tane fiziksel parametreleri + domain paneli + kilitler

Bu partide (2026-10-06, tek build): tane küre yarıçapı düzeltmesi; çekirdekte tek "tane hazır
mı" kuralı (`matterGrainBlockers`) ve onu bozan API düzenlemelerinin reddi; sönüm yerine
**restitution** (çarpışma geri sekmesi, temas başına etkin kütle), sürüklenme viskozitesi sıvı
maddeden; domain paneli 6 sekme kararına göre (tane malzemesi Matter, çözücü Solvers, rapor
Measure), kilitler sebep satırıyla, standart satır + zorunlu tooltip, "Collapse all";
16k DEM/XPBD maliyet testi. Gerekçe ve kararlar: [MATTER_H1_GRAIN_ROADMAP.md](MATTER_H1_GRAIN_ROADMAP.md).

Bulutta yapılan: grain shader'ları glslangValidator + spirv-val; değişen C++ dosyaları Linux g++
sözdizimi geçişi (MSVC'ye özgü hatalar görünmez; `scene_ui_forcefield.hpp:1721 sprintf_s`
düzeneğin bilinen yan etkisi); `matter_grain_coupling_test.cpp` PASS; CPU XPBD kopyası PASS;
kontrat + panel alan envanteri + IPC denetimi PASS. Canlı hiçbir şey koşmadı.

**Build:** C++ + `compile_shaders.bat`. Grain shader revision **13**. Yeni başlık
`source/src/UI/DomainPanelWidgets.h` (vcxproj'ta); silinen kullanılmayan dosyalar
`scene_ui_simulation_domains.inl`, `scene_ui_fluid_billboards.hpp` (build "bulunamadı" derse
bir yerde unutulmuş include var — hata metnini getir).

**Ara düzeltme (gözle kontrol sonrası):** Max Auto Resolution + Boundary Padding aynı satırdaydı,
ikisi de standart genişlikte → panel taşıyordu. Padding kendi satırında; alan envanteri artık
`SameLine` sonrası standart genişlikte alanı FAIL eder. **Build:** yalnız C++.

0. **Statik (uygulama gerekmez):** `python scripts/test/check_matter_grain_contracts.py`,
   `python scripts/test/check_domain_panel_fields.py` (254 kontrol: hiçbiri kaybolmadı, yeni
   yardımcı dosyalarda tooltip'siz kontrol yok), `python scripts/test/xpbd_grain_friction_reference.py`.
1. **Revizyon:** herhangi bir tane koşusunda `shader revision mismatch` yok.
2. **Kilitler (IPC):** `rt_h1_grain_runtime_ipc.py --readiness-only` → `PASS grain readiness`:
   backend=cpu, boundary=open, pore exchange açma, eski `normal_damping_n_s_m` ve
   `drag_viscosity_pa_s` anahtarları reddedilir, domain değişmeden kalır. Bozuksa: bir yazıcı
   kontrolü atlıyor (hangisi `ACCEPTED` diye basılır).
3. **Temel regresyon:** `rt_h1_grain_runtime_ipc.py` (serbest düşüş), `--history-only`,
   `--static-only`, `--convergence-only`, `--extended-only`. Testler eski sönüm değerlerinin
   restitution karşılığıyla koşar (`grain_units.restitution_of`); tane-tane fiziği aynı,
   **duvar teması √2 daha sönümlü** (aynı geri sekme). Değerler önceki turlarla yakın olmalı.
4. **Yerleşme:** `--settle-only --settle-normal-damping 8` → enerji/COM kapısı + `state_resident`.
   ★ Sinsi: geçer ama yığın daha "ölü" görünür → duvar sönümü beklenenden fazla; sayıları getir.
5. **Su + tane:** `--coexist-only` (≈6.13 ±%15, batma ≈1), `--coexist-only --coexist-hydrostatic`,
   `--porous-only` (fark .051–.153 m). Sürüklenme artık Water maddesinin viskozitesinden
   (1e-3 Pa s, önceki sabitle aynı): sonuçlar önceki turla aynı olmalı. Belirgin fark = madde
   profilinden yanlış viskozite okunuyor.
6. **Islak:** `--wet-only` (kuru rms > ıslak rms / .8, köprü > 0, su korunumu).
7. **XPBD:** `--xpbd-compare` (eğim 0, tablo).
8. **Ölçekte maliyet (H1-G0 kararı):**
   `python scripts/test/rt_h1_dense_grain_scene_ipc.py --counts 4096 16384 --solvers dem xpbd:20 xpbd:40 xpbd:80 --steps 120`
   → sonda tablo (median ms, COM, rms, ms/dem, rms/dem). Ölçüt: 16k'da rms/dem .95–1.05 olan
   ilk XPBD satırı ms/dem ≤ .33 değilse XPBD sökülür. Tabloyu getir.
9. **Panel (gözle, ~5 dk):** tane açık Matter domain'i seç.
   - **Solvers** en üstte "Granular Solver (grains)": "Ready." ya da sebep satırları; Accuracy
     Draft/Production/Reference; Advanced altında XPBD (deneysel); Liquid coupling.
   - **Matter**: "Granular Material (grains)" (gerçek tane yarıçapı, μ, μr, restitution, su);
     legacy "Granular Material" yerine tek satır; Pore Water yerine tek satır.
   - **Domain**: backend ve sınır gri, altında sebep; **Environment/Solvers**: Domain Motion
     Coupling ve Reseed gri.
   - **Output**: "Grain spheres — Diameter …"; viewport'ta küreler tane boyutunda (bildirdiğin
     hata). ★ Küreler hâlâ varsayılan boyuttaysa GPU yolu hâlâ seçiliyor.
   - **Measure**: "Grain Step" raporu. Her sekmede "Collapse all". Gri kontrolde tooltip görünür.
   - Kayıtlı tane sahnen varsa aç: yüklenir, restitution sönümden çevrilmiş olur.
10. **Uzun / diğer:** `--repose-only`; `rt_g2_dry_wet_compare_ipc.py --expect-empty-fluid-skipped`.

Bilinen açık iş (bu partide yok): madde başına tane malzemesi (SubstanceProfile; GPU'da tane
başına malzeme tablosu), örtüşmeden türetilen sertlik, sıvı parselleri için tane domain'inde
ayrı render grubu (sanal çocuklar), gaz ortam havası referansının Environment'a taşınması.

---

# Sıradaki kullanıcı build (C++ + SHADER): H1 tane B2–B8 + B9a — tek uzun test geçişi

Bulutta yazılan sekiz parti tek build'de: B2 kova komşuluğu + yığılma açısı, B3 su + tane
aynı domain'de (sürüklenme/kaldırma), B5 hacim dışlama + basınç kuvveti, B6 ıslak tane
(emilim/kuruma, sıvı köprüsü), B7 XPBD adayı, B8 hücre sırası + kimlikle taşınan temas
geçmişi, B4 yerleşik durum + transfer sayaçları, B9a geçmiş geçersiz kılma + CPU referansı
EPSD2. Ayrıntı ve gerekçe: [MATTER_H1_GRAIN_ROADMAP.md](MATTER_H1_GRAIN_ROADMAP.md).

Bulutta yapılabilenler yapıldı: tüm grain shader'ları + `sim_fluid_divergence_porous` +
`sim_matter_grain_permute{,_copy}` glslangValidator + spirv-val ile derlendi; değişen
C++ dosyaları (ParticleSimulation.cpp ve .inl'leri dahil) Linux g++ sözdizimi geçişinden
geçti (Windows/CUDA/SDL başlıkları taklit edilerek — MSVC'ye özgü hatalar burada görünmez);
`matter_grain_coupling_test.cpp`, `matter_grain_reference_slope_test.cpp`,
`matter_granular_contact_test.cpp` g++ ile derlenip PASS koştu. Canlı hiçbir şey koşmadı.

**Build:** C++ + shader. Yeni dosyalar vcxproj/filters'ta: `MatterGrainCoupling.cpp/.h`,
`sim_fluid_divergence_porous.comp`, `sim_matter_grain_permute.comp`,
`sim_matter_grain_permute_copy.comp`. `compile_shaders.bat` (hepsini derler) ya da
`compile_sim_shaders.bat` (yenileri listeye eklendi). Grain shader revision **12**, push
constant **112 bayt**, 15 buffer.

**★ Ara düzeltme (2026-10-06, canlı turlardan sonra):** `--coexist-only` gömülü tane
ivmesi 9.43, sonra 9.08 çıktı (beklenen 6.13); `--coexist-hydrostatic` kolu 6.50 ile geçti.
Teşhis satırları: porous kolda tanenin basınç itkisi Arşimet'in ≈ %10'u ve işareti kare
kare dönüyordu → hata basınç **okumasında**. GPU basınç tamponu okuması tamamen söküldü
(önceki ara denemenin `sim_matter_accumulate` / `frame_pressure`'ı da gitti — release
yolundaki sızıntı önerisi bu yüzden geçersiz). Kuvvet artık sıvının ölçülen ivmesinden:
F = batma · ρV (Du/Dt − g) (`applyMatterGrainLiquidAccelerationForce`; C++ testinde durgun
su = tam Arşimet, yukarı g ivmelenen su = 2×). Tanesiz liquid subset granular elastik alt
adım sayısını ödemiyor (yalnız CFL). **Build:** C++ + `compile_sim_shaders.bat` (silinen
kernel listeden çıktı; eski `sim_matter_accumulate.spv` kalırsa zararsız).
Canlı sonuçlar: 0, 2, 3, settle PASS; hidrostatik kol + pour PASS; 6'dan devam.
**İkinci tur (ivme kuvveti canlıda):** gömülü tane 6.96 (±%15 içinde), itki gürültüsüz;
ama `submerged` .69 (25 cm derindeki tane!) ve porous farkı yalnız .023 m (beklenen .102).
Düzeltmeler: (a) batma artık doluluk — hücre, parselleri gözeneklerinin yarısını
doldurunca sıvı sayılır (seyrek FLIP havuzu hacmin .65–.85'ini tutar); (b)
`sim_fluid_divergence_porous` yoğunluk düzeltmesini gözenek hacmine (ε·ppc) hedefler —
eskiden tam ppc'ye çekip gözenekleri yeniden dolduruyordu (dışlamayı zamanla geri
alıyordu); (c) `--porous-only` kapısı artık iki kolun **farkı** (havuzun kendi oturma
kayması +.041 m iki kolda ortak). **Build:** C++ + shader.
**Üçüncü tur: 6 ve 7 PASS.** Gömülü tane 5.74 (beklenen 6.13, −%6), batma .97–1.0, itki
Arşimet'in ~%8 üstünde (dışlamanın dinamik basıncı; makul). Kapalı kol 9.81, pour PASS.
Porous fark .132 m (beklenen .102, +%29). Sıradaki: 8 (`--wet-only`), 9 (`--xpbd-compare`).
**Dördüncü tur:** `--wet-only` kuru kolda 512 tane 1/30 s'de .25 m küreye sığmadı (doğum
filtresi çakışmayı reddeder, ~380 sığar) → test doğumu 1 s'ye yaydı, sayı emisyondan sonra
kontrol ediliyor. `--xpbd-compare`: XPBD eğimde 3.3 mm/s kaydı (DEM 2e-8) — statik sürtünme
düzeltmesinin 2.5/3.5'i dönmeye gidiyor, yuvarlanma sınırı dönmeyi öldürüp merkezin kaymasını
bırakıyordu. Artık tanθ ≤ min(μ, μr) iken temas tamamen ötelemeyle tutuluyor. Grain shader
revision **11** (`shader revision mismatch` görürsen shader derlenmedi). **Build:** C++ + shader.
**Beşinci tur:** `--wet-only` PASS (kuru rms .324, ıslak .197, 2736 köprü; su sapması 1e-6 kg).
XPBD kayması aynı kaldı (3.25 mm) — ilk düzeltme yalnız hareketsiz başlangıçta tutuyordu.
CPU kopyası (`scripts/test/xpbd_grain_friction_reference.py`) kök nedeni gösterdi: hız bir kez
oluşunca 3.5/m yolu her alt adımda temas noktasını durdurup dönmeyi silmeye devam ediyor,
merkez 2.5·h·g·sinθ'da sabit kayıyor. Kural: öteleme düzeltmesi F = min(tam durdurma, μλ);
F ≤ μr·λ ise yalnız öteleme (tutunma ya da dönmeden kayma), değilse yuvarlanma yolu. CPU
kopyasında eğim tutunması 20/40/80 alt adımda 1e-17, yuvarlanma ivmesi 1.739 (analitik 1.738),
kayma ivmesi .657 (analitik .657). Revision **12**. Rampa collider gizmosu döndürülmüş
plane'de yanıltıcı görünüyor; tane geometrisi doğru (DEM rampa testi analitik düzleme göre).
**Altıncı tur:** XPBD eğim tutunması 0.0 (düzeltme çalıştı). `--xpbd-compare` yığın kolu:
XPBD malzemesi alt adımla değişiyor (rms .363/.386/.435 @20/40/80, DEM .467; COM kayması %5.4 >
%5). Karar AÇIK: test maliyet sütunu (adım ms, tane GPU ms) kazandı, alt adım duyarlılığı artık
bulgu olarak kaydediliyor; `--xpbd-compare` tekrar koşulunca tablo sonuna kadar çıkar (roadmap B7).

0. **Statik sözleşme** `python scripts/test/check_matter_grain_contracts.py` → PASS.
1. **Açılış/revizyon.** Herhangi bir tane koşusunda `shader revision mismatch` yok. Varsa
   shader'lar derlenmedi (ya da yalnız eski bat çalıştı).
2. **Temel regresyon** `python scripts/test/rt_h1_grain_runtime_ipc.py` → g ±%5, pile dt
   ≤ %10. Yalnız-tane domain'i artık kendi tamponlarında, hücre sırasında ve kimlik eşlemeli;
   burada kırılan her şey B3/B8 tane yolu. `bucket overflow` = 16 kapasite yetmedi.
3. **Temas geçmişi** `--history-only` (B8/B9a): reset sonrası ilk adım
   `history_reset_this_step=true`, sonraki adımlarda reset yok, en az bir adımda
   `history_remapped=true` (hücre sırası değişti, geçmiş GPU'da taşındı). Bozuksa: her adım
   reset → host bir aşama tane konumunu değiştiriyor (imza) ya da permute kernel'i yanlış;
   hiç remap yok → sıralama devrede değil.
4. **Statik + yakınsama + extended + settle** (`--static-only`, `--convergence-only`,
   `--extended-only`, `--settle-only --settle-normal-damping 8`) → önceki değerler.
   `--settle-only` artık B4 kapısı da içerir: yerleşmiş yığında `state_resident` en az bir
   örnekte true (yoksa bir host aşaması her kare taneye dokunuyor).
   ★ Statik kol 3. maddeden sonra: geçmiş taşınmıyorsa tutunma burada da kaybolur.
5. **Maliyet (B2 + B8)** `python scripts/test/rt_h1_grain_kernel_profile_ipc.py --counts 1024 4096 --stiffness 200000`
   → `sim_matter_grain_step` µs/çağrı 243/289'dan düşmeli; `python scripts/test/rt_h1_dense_grain_scene_ipc.py`
   → medyan 25.5/24.4/41.1 ms ile kıyas. `runtime.upload_bytes/download_bytes/transfer_batches`
   kaydedilir (B4 ölçümü: sonraki yerleşiklik adımı bunlara göre seçilecek).
6. **Su + tane (B3, B5 varsayılan açık)** `--coexist-only`: gömülü tane ilk ivmesi açıkken
   ≈ **6.13 m/s²** (±%15; artık hidrostatik değil çözücünün basınç gradyanı), kapalıyken 9.81;
   yığına su dökme: adım tutulmaz, su düşer, kütleler sabit, momentum artığı ≤ 1e-3 × alışveriş.
7. **Hacim dışlama (B5)** `--porous-only`: 1000 tane havuza dökülünce su yüzeyi (p95)
   açıkken N·V/A ≈ **0.102 m**'nin %50–150'si yükselir, kapalıyken ≤ %25. Bozuksa: açık kolda
   yükselme yok → ağırlıklar/porous kernel devrede değil (`liquid.porous_cells` 0 mı?);
   kapalı kolda yükselme → tane sıvıyı başka bir yoldan itiyor.
8. **Islak tane (B6)** `--wet-only`: kohezyon A/B (512 tane kolon çöküşü, doğumda yarı
   doygun, temsil edilen tane .5 mm): ıslak yatay RMS ≤ 0.8 × kuru, köprü sayısı > 0 (kuru 0),
   tutulan su sabit; havuzda emilim: tane suyu > 0, sıvı + tane suyu sabit (≤ 1e-4 göreli),
   `water_balance_error_kg` ≤ 1e-5.
9. **DEM vs XPBD (B7, H1-G0)** `--xpbd-compare`: iki çözücü de serbest düşüş ±%5 ve 20°
   eğimde tutunma; XPBD yığını 20→80 alt adımda COM ≤ %5, taban ≤ .2 r değişir (iterasyonla
   sertleşmiyor). Çıktı tablosu (COM/RMS/alt adım/ms) karar girdisi — kazanan ilan etme,
   bana getir.
10. **Yığılma açısı (B2)** `--repose-only` (uzun): μr .05 vs .3 ≥ 3°, μr .1 15–45°,
    r .025 vs .0175 ≤ 4°.
11. **Diğer çözücüler** `python scripts/test/rt_g2_dry_wet_compare_ipc.py --expect-empty-fluid-skipped`
    → PASS (porous kernel yalnız tane domain'inde; variational yol değişmedi). Bir sıvı
    sahnesi (variational açık collider'lı) hızlı göz kontrolü.
12. **Senin sahnen (görsel).** Su + kuru tane aynı domain: su akar, taneler suya girince
    yavaşlar, dibe çöker, su seviyesi yükselir; sıvı parsel yarıçapıyla, taneler fizik
    yarıçapıyla çizilir. Yüksek çözünürlükte tek tane emitter'da kalmamalı; kalırsa
    `fluid.matter_models` → `mixed_execution.status` metnini getir. İstersen
    `wet_grains` + `wet_appearance_enabled` ile ıslanan taneleri gör.

★ En sinsi: **3. madde geçip 4. madde (statik tutunma) çoklu tanede kayması.** Tek tane
remap'e girmez; geçmiş taşıma hatası yalnız çok taneli yığında yavaş sürünme olarak görünür
ve "biraz yumuşak kum" gibi okunur. `--settle-only` COM aralığı bunu yakalar.
★ İkinci: 6. maddede açık kolun 6.13 değil ~2.45 vermesi → basınç kuvveti ve hidrostatik
kaldırma birlikte uygulanıyor (çift sayım: 9.81 − 2 × 3.68).
★ Üçüncü: 7. maddede yükselme var ama su kütlesi düşüyor → yükselme dışlamadan değil
kayıptan.


# Sıradaki kullanıcı build (C++ + SHADER): dry grain — birleşik alt adım, statik sürtünme, tane kütlesi

> **Canlı sonuç (2026-10-06):** 1–7 PASS. Statik tutunma: yay 2/7 kayma 0, yay 0 0.1464 m/s (= analitik). Yakınsama (accuracy bağlı 62→122 alt adım) COM %3.07 / RMS %2.12. Yoğun medyan 25.5/24.4/41.1 ms (eski 173.6/213.8/272.2), dispatch 209 (eski 3471). Step kernel gecikme sınırlı (4× tane ≈ aynı süre). 8. madde (G2) PASS: aynı build iki koşu farkı 1e-5 m düzeyi; 10-05'e göre sabit ~6.6 mm fark emitter tohumunun bellek adresinden gelmesinden (oturumlar arası A/B imkânsız — ayrı iş). Ayrıntı: MATTER_GRAIN_GPU_RUNTIME.md "Fused partisi canlı sonuç".

Değişiklik (2026-10-06): alt adım başına 3 dispatch → 1 (ping-pong durum + nesil damgalı
hash), CFL 48-temas/0.1 yerine 24-temas Gershgorin + "çarpışma başına N alt adım" doğruluk
sınırı, Cundall–Strack teğetsel yay + EPSD2 yuvarlanma yayı (tane başına 24 slot temas
geçmişi), tane kütlesi = yığın yoğunluğu / packing × küre hacmi, Vulkan backend'de aynı
recording içinde descriptor set yeniden kullanımı (TÜM Vulkan compute çözücülerini etkiler).
Yeni shader: `sim_matter_grain_step.comp`; `_contact/_integrate/_integrate_hash` silindi.
`compile_sim_shaders.bat` gerekli. Ayrıntı: [MATTER_GRAIN_GPU_RUNTIME.md](MATTER_GRAIN_GPU_RUNTIME.md) §2026-10-06 fused.

1. **Statik sözleşme** (build gerekmez, saniyeler): `python scripts/test/check_matter_grain_contracts.py`
   → PASS. Bozuksa: kaynak/ABI uyumsuz, build'e geçme.
2. **Temel regresyon** (boş, frame0 paused sahne, dış terminal):
   `python scripts/test/rt_h1_grain_runtime_ipc.py` → serbest düşüş g ±%5, zemin/spin,
   64-tane pile 60/120 Hz ≤%10. Her örnekte `grain_diagnostics.runtime.dispatches == substeps+2`
   ve substeps çift. Beklenen: varsayılan profilde (k 20 kN/m, Cn/Cs 4) `substep_limit`
   = `damping`, 60 Hz'de ~166 alt adım / 168 dispatch (eski formül ~366 alt adım / ~1100).
   Bozuksa: `shader revision mismatch` = shader derlenmedi; `contact count exceeds the 24` =
   bütçe gerçek yığında yetmiyor (ölçüm olarak kaydet, gevşetme); pile dt farkı büyüdüyse
   yeni CFL fazla gevşek → 4. madde ile doğrula.
3. **Statik tutunma** `--static-only` → 20° eğimde tek tane: ratio 2/7 son 1 s kayma
   ≤ 0.5 mm, `sticking_contacts_last_substep ≥ 1`, kütle 0.17453 kg (±%0.01);
   ratio 0 kolu ≥ 1 cm kaymalı (ölçü aletinin ayırt ettiğinin kanıtı).
   Bozuksa: ratio 2/7 de kayıyorsa geçmiş her karede sıfırlanıyor olabilir
   (`history_reset_this_step` ilk kareden sonra false olmalı) veya yuvarlanma yayı çalışmıyor.
   Kütle 0.2 kg çıkarsa doğum yolu eski voxel/PPC kütlesini yazıyor.
4. **Gerçek yakınsama** `--convergence-only` → aynı 64-tane pile, contact_resolution 24 vs 48
   (alt adım gerçekten yarıya iner); son COM/RMS göreli fark ≤ %5. Eski 60/120 Hz testi aynı
   alt-adım dt'sini karşılaştırıyordu, bu yüzden "fark 0" ölçüm değildi.
5. **Köşe + rampa + yoğunluk** `--corner-only`, sonra `--extended-only` → eski gate'ler aynı.
6. **Uzun yerleşme** `--settle-only --settle-normal-damping 8` → 8 s enerji/kütle/COM gate'i.
   Yeni teğetsel yay enerji POMPALIYORSA burada görünür (son 2 s enerji/tane > 1e-5 J).
7. **Maliyet** `python scripts/test/rt_h1_dense_grain_scene_ipc.py` → 1024/4096/16384 medyan
   ms eski 173.6/213.8/272.2 ile kıyas; `median_dispatch` ~204 olmalı (eski 3471),
   `last_runtime.substep_limit` raporlanır. Bozuksa: dispatch düştü ama süre düşmediyse
   maliyet dispatch ek yükü değil kernel'in kendisi (history taraması) — sonraki parti oraya.
8. **Diğer Vulkan çözücüleri** (descriptor set yeniden kullanımı ortak backend'de):
   `python scripts/test/rt_g2_dry_wet_compare_ipc.py --expect-empty-fluid-skipped` + gaz/sıvı
   herhangi bir hızlı sahne. Beklenen: aynı sonuç, biraz daha az CPU kayıt süresi.

★ En sinsi başarısızlık: **8. madde.** Descriptor set yeniden kullanımı yanlış bağlamaya yol
açarsa hata vermez; bir kernel önceki dispatch'in tamponunu okur ve "biraz farklı"
simülasyon üretir. Anahtar buffer id'lerinin tamamı, cache her `synchronize()`'da
temizleniyor ve destroy/resize senkronize ediyor — ama gaz/sıvı sonuçlarının önceki
build ile sayısal kıyası tek gerçek kanıt.
★ İkinci sinsi: 3. maddede ratio 0 kolu da tutunursa test hiçbir şey ölçmüyordur.


# Sıradaki kullanıcı build (yalnız SHADER): yürüyüş uçuşu index uzayında

> **Canlı sonuç (2026-10-06, aynı kamera):** 16 spp 5.34 -> 4.99 s (-%6.5), 64 spp 22.9 -> 21.5 s; density 385M -> 353M, walk_events aynı (50.6M); görüntü ort. parlaklık 109.66 vs 109.05, |fark| 5/255 = 64 spp gürültüsü. Sonuç: süre olay sayısıyla ölçekleniyor (~57 ns/olay), sayılan hiçbir kalemle değil (gölge örneği -%56, prob -%64 süreyi değiştirmedi). Kalan: warp sapması / dev closest-hit yazmaç baskısı — büyük yeniden yapılanma ister. Kod tarafı kapatıldı; kalan düğmeler: max_events 8-16, step 1-1.5, daha az spp + denoiser.

> Önceki partinin canlı sonucu: ertelenmiş prob izlemeyi 49.0M -> 17.7M (olay başına
> 0.99 -> 0.35) düşürdü ama 16 spp süre 5.24 -> 5.34 s: izleme darboğaz DEĞİL ("70 ns/izleme"
> çıkarımı yanlıştı — iki bilinmeyenli farkı başka ölçümün birim maliyetiyle bölmüştüm).
> Temas bölgesinde sızıntı görülmedi. Kalan aday: uçuş adımı başına SAYILMAYAN iş — her
> adımda boş-döşeme sorgusu, yaprak majorantı ve trilineer örnek dünya->yerel->index
> dönüşümünü üç kez yeniden yapıyordu (step çarpanı 0.5->1 süreyi %26 düşürmüştü).

Değişiklik: NanoVDB fog için uçuş ve gölge kirişi index uzayında: idx = o + d*t uçuş
başına bir kez; adım başına bir ağaç sorgusu + bir trilineer okuma. Yoğun gaz ve materyal
gürültülü hacimler eski dünya-uzayı döngüsünde. rwLeafMajorant söküldü (tek tüketici gitti).

1. **Görüntü birebir.** Aynı kare/kamera/IŞIK ile önceki build görüntüsüyle ortalama fark
   gürültü düzeyinde (< ~2/255). Sistematik kayma = index yoğunluğu sampleDensityAcc ile
   uyuşmuyor (remap/cutoff/pivot) — en sinsi sonuç: "biraz daha açık/koyu kar".
2. **Maliyet.** Aynı koşulda 16 spp, max_events 51: süre 5.3 s'den düşmeli; density_samples
   ve walk_events ~aynı kalmalı (aynı iş, daha ucuz adım). Örnek sayısı değişirse yol farklı.
3. **Petek yok.** Spread büyük/küçük iki değerde yaprak kenarı deseni görünmemeli.


# H1 twisting revision3 - 8 s canlı PASS

- Kullanıcı C++/shader build sonrası twist .1 /Cn8 Ns/m8 s PASS; mass drift0,
  son2 s COM/RMS aralığı0, enerji/tane6.09e-11 J. Gate korunur.
- Cn4 koşulu6 s doğrusal KE nedeniyle RED; spin direnci her iki koşulda çalıştı.
- Default twist0/Cn4 değişmedi; bu paket yeni build beklemez.
- Kalıcı static contact history ve repose/cache/wet/residency kabulü açık.


# Sıradaki kullanıcı build (yalnız SHADER): yürüyüşte ertelenmiş katı probu

> Sayaç ölçümü (kar sahnesi, kare 203, 16 spp, max_events 51 vs 8): 5.24 s vs 3.06 s.
> İzleme ~70 ns, voksel örneği ~1.3 ns (shadow_steps 8/16 kıyasından). 51 olayda: katı probu
> 49M izleme = olay başına 0.99 = ~3.4 s (~%65); gölge ışını 9.1M (%82 atlanıyor) ~0.6 s;
> örnekler 506M ~0.7 s. Yürüyüş ortalama 10.1 olay, %4.5 bütçeye takılıyor.

Değişiklik: prob artık yalnız yürüyüş son probdan probeEps = max(2 voxel, 5 mm) uzaklaşınca,
son prob noktasından bugünkü noktaya TEK doğru parçası; çıkış uçuşu her zaman problu.

1. **Maliyet.** Aynı kare/kamera/ayarlar, 16 spp: walk_probe_traces/walk_events 0.99'dan
   belirgin düşmeli (~0.1-0.3); süre 5.2 s'den ~2.5-3 s'ye. Düşmezse olaylar arası yer
   değiştirme zaten probeEps'i aşıyor (seyrek ortam) — probeEps'i ölç.
2. **★ Sinsi sonuç: zemin/küre temasında ışık sızıntısı.** Yol bir yüzeye probeEps kadar
   girip dönebilir. Kar-zemin ve kar-küre temasında yeni ışıklı/koyu bant, ya da yüzeyin
   ARKASINDA kar ışığı görünürse sızıyor. Önceki görüntüyle temas bölgesini kıyasla.
3. **Görüntü eşdeğerliği.** Genel parlaklık önceki build ile ~aynı (fark < birkaç/255).


# Sıradaki kullanıcı build (C++ + SHADER): yürüyüş maliyet sayaçları

> Önceki partinin canlı sonucu: shadow_steps 8 ile gölge örnekleri 347M -> 154M (-%56) ama
> 16 spp süre 6.1 -> 5.9 s; 64 spp 22.0 -> 19.8 s (-%10); görüntü farkı 2/255. Örnek SAYISI
> süreyi taşımıyor -> olay başına sabit maliyet (izleme?) adayı. Bunu ayırmak için sayaç.

Yeni render.volume_stats alanları: walk_paths, walk_events, walk_probe_traces,
walk_shadow_traces, walk_shadow_skipped, walk_event_capped. VolumePerformanceStats
160 -> 184 bayt (46 sözcük); tek tanım volume_instrumentation.glsl, 3 shader include eder.

1. **ABI.** Panel Volume Performance ve render.volume_stats eski alanları AYNI değerlerle
   göstermeli (önceki ölçümle karşılaştır). Kaymış/saçma değerler = glsl ile C++ struct
   uyuşmuyor.
2. **Tutarlılık.** walk_probe_traces ≈ walk_events (olay başına bir prob);
   walk_shadow_traces + walk_shadow_skipped ≈ walk_events (ışık varken). Değilse sayaç
   yanlış yerde.
3. **Karar ölçümü.** Kar sahnesi kare 203, 16 spp: max_events 8 / 51 iki koşu. Süre farkını
   (walk_probe_traces + walk_shadow_traces) farkına ve density+shadow örnek farkına böl:
   hangisinin birimi süreyi açıklıyorsa hedef odur.


# Sıradaki kullanıcı build (yalnız SHADER): yürüyüş gölge kirişi = shadow_steps

Taban ölçüm (kar sahnesi, kare 203, kullanıcı kamerası, 16 spp; step 1.575, res 4x, 612k parçacık):
yürüyüş 6.1 s — hacme giriş 8.49M, yoğunluk 508M, GÖLGE 347M örnek; yoğunluk≈0 ile 1.1 s
(salt kutuya giriş+geçiş yükü ≤ %18); march 0.5 s. Gölge kirişi sabit 16 örnekti ve
shadow_steps (kullanıcıda 8) yok sayılıyordu. Şimdi: örnek = clamp(shadow_steps, 2, 64),
erken bırakma tau > 6 (önce 12).

1. **Maliyet.** Aynı kare/kamera/ayarlar, 16 spp: shadow_density_samples 347M'den belirgin
   düşmeli (≈ yarı veya daha az), süre 6.1 s altına. Düşmüyorsa kirişler zaten erken
   bitiyordu ve maliyet olay başına izleme (iki trace) tarafında.
2. **★ Sinsi sonuç: kar ışık sızdırıyor.** Az örnekli kiriş ince ama yoğun bir katmanı
   atlayabilir: gölge tarafı/oyuklar AÇILIR, "biraz daha yumuşak ışık" gibi görünür.
   shadow_steps 8 vs 16 kıyasla; oyuklarda fark varsa 8 bu sahne için az.
3. **Açık (ayrı konu):** fluid.set_param visible=false fog hacmini GİZLEMEDİ (sayaçlar
   march ile birebir). Domain görünmezken fog'un çizilmesi muhtemel hata.

# H1 BVH/manifold/fusion - 2026-10-06 canlı PASS

- Kullanıcı C++/shader build sonrası köşe/ramp/256/1024 ve tam temel grain regresyon PASS.
- Köşe .019195 m > .0175 gate; dry drift0; shader revision2 doğrulandı.
- Dispatch3273→2457; 256/1024 son ölçüm39.27/41.33 ms. Genel hız oranı değil.
- Bu paket rebuild beklemiyor; frame0 paused, test kaynakları kapalı.
- Kalıcı static friction, uzun settle/repose, cache ve production residency kabulü açık.


# Sonraki derleme kontrolleri — RT pipeline asenkron derleme + disk cache (2026-10-05)

> **Canlı (2026-10-05):** ilk Rendered geçişinde ERİŞİM İHLALİ — nvoglv64.dll, renderInteractiveViewportImpl sonundaki endSingleTimeCommands (raster stand-in, derleme sürerken). Geçici karar: kRasterWhileRTPipelineCompiles = false (VulkanBackend.cpp); derleme sürerken son kare yeniden gösterilir, HUD mesajı ve arka plan derleme aynen kalır. Açık: raster stand-in, Rendered-mod geometri/TLAS yeniden kurulumuyla aynı karede validation layer ile doğrulanmalı.


> **Durum:** CANLI — bu bölüm en üstte; aşağıdaki eski bölümler duruyor.

Shader değişmedi, yalnızca C++ derlemesi. Yeni dosya: `source/src/Backend/VulkanRTPipelineBuild.cpp`
(vcxproj + filters'a eklendi). Eski senkron `VulkanDevice::createRTPipeline` silindi.
Sürmek için: `Start-RayTrophi.ps1`, `Import-Module .\scripts\ipc\RtIpc.psm1 -Force`,
`Invoke-RtIpc render.rt_pipeline_status`.

★ Önce bilmen gereken: disk cache **shader değiştikten sonraki ilk derlemeyi hızlandırmaz** —
yeni SPIR-V cache'te yoktur. O ilk derlemeyi taşınabilir yapan (1) worker thread, (2) deferred
operation. Cache'in işi ikinci açılış.

1. **Derleme + açılış (en hızlı).** Log'da `RT pipeline cache loaded (... KB)` (ilk açılışta yok,
   normal) ya da `RT pipeline disk cache disabled` YOK olmalı. Bozuksa: `loadRTPipelineCache`
   `initialize()`'da `loadRayTracingFunctions()`'tan sonra çağrılmıyor ya da LOCALAPPDATA çözülmedi.
2. **UI donmuyor (asıl şikâyet).** `compile_shaders.bat` çalıştır (shader'ları değiştir ya da
   sadece yeniden üret — sürücü cache'ini geçersiz kılmak için en az bir .spv içeriği değişmeli),
   uygulamayı aç, Solid'den Rendered'a geç. Görmen gereken: viewport **Solid çizmeye devam eder**,
   kamera dönebilir, paneller tepki verir; HUD'da turuncu satır
   `Compiling ray tracing shaders (first use after a shader update)... 0:42 · showing raster meanwhile`
   ve saniye sayacı akar. Bitince satır kaybolur, örnek sayacı 1/N'den başlar.
   Bozuksa: (a) hâlâ donuyorsa `requestRTPipelineBuild` yerine bir şey derlemeyi render
   thread'inde bekliyor — `pollRTPipelineBuild(block)` `tex == nullptr` ile mi çağrılıyor bak
   (Main viewport'u `raytrace_texture` geçiyor, null olmamalı); (b) viewport siyah/donuk kare
   ama UI canlı → `drawRasterWhileRTPipelineUnavailable` false döndü (graphics queue yok ya da
   raster yolu tekrar renderProgressive'e düştü; log'da "Interactive viewport mode selected").
3. **IPC durum okuması (2 ile aynı anda).** Derleme sürerken
   `Invoke-RtIpc render.rt_pipeline_status` → `rt_pipeline_state: "compiling"`, `compile_seconds`
   artıyor; bitince `"ready"`, `build_count: 1`, `deferred_threads` ≥ 2 (NVIDIA'da beklenen:
   çekirdek−1'e kadar), `cache_hit: false` ya da `null`.
   ★ Sinsi: `deferred_threads: 1` ve süre eskisiyle aynı → sürücü işi ertelemedi
   (`VK_OPERATION_NOT_DEFERRED_KHR`) ya da `maxConcurrency` 1 döndü; asenkron yine çalışır ama
   paralel derleme kazancı YOK. Bunu hata diye kimse raporlamaz, sayıya bak.
4. **Paralel kazanç ölçümü (A/B).** Aynı değişmiş shader setiyle iki kez ölç, her seferinde
   `.spv`'yi yeniden üretip (sürücü cache'i ısınmasın diye bir byte değiştir) ve
   `RT_VK_PIPELINE_CACHE=0` ile:
   `RT_VK_RT_COMPILE_THREADS=0` (senkron, deferred yok) vs ayarsız (otomatik). `compile_seconds`
   oranını yaz. Beklenti: anlamlı düşüş; volume_closesthit tek başına en uzun aşamaysa kazanç
   aşama sayısıyla sınırlı kalır (sürücü genelde aşama başına paralelleştirir).
   Bozuksa (fark yok): 3'teki `deferred_threads`'e bak — 1 ise sürücü paralel vermiyor demektir.
5. **İkinci açılış (cache).** Uygulamayı kapat/aç, Rendered'a geç. Görmen gereken: HUD satırı
   ya hiç görünmez ya bir saniye; `render.rt_pipeline_status` → `cache_loaded: true`,
   `cache_hit: true` (ya da sürücü feedback vermiyorsa `null`), `compile_seconds` < ~2.
   Dosya: `%LOCALAPPDATA%\RayTrophiStudio\vk_pipeline_cache\rt_pipeline_<vendor>_<device>.bin`.
   Bozuksa: `cache_loaded: false` + `cache_reject_reason` doluysa başlık uyuşmadı (sürücü
   güncellemesi sonrası beklenen: "written by another driver version", dosya silinip yeniden
   yazılır). ★ Sinsi: `cache_hit: false` ama süre yine kısa → hızlandıran NVIDIA'nın kendi
   disk cache'i, bizimki değil; ayırt etmek için `RT_VK_PIPELINE_CACHE=0` ile karşılaştır.
6. **Bozuk cache dayanıklılığı.** Uygulama kapalıyken `.bin`'in ilk 16 byte'ını sıfırla, aç.
   Log'da `RT pipeline cache discarded (corrupt header size 0)` / benzeri, çökme yok, derleme
   normal sürede biter ve dosya yeniden yazılır. Bozuksa: `validateCacheBlob` atlanıyor.
7. **Hair / sphere / volume sonrası adımlar.** Saç + köpük (prosedürel küre) + hacim içeren bir
   sahnede derleme bitince: saç doğru hit shader'la görünmeli (log: `Hair SBT offset corrected`
   gerekiyorsa), küreler görünmeli. Bozuksa: `onRTPipelineInstalled()` (eski createRTPipeline
   sonrası adımlar) kurulumdan sonra çalışmadı. ★ Sinsi: saç "makul" ama hacim shader'ıyla
   gölgelenmiş görünür = hair TLAS offset'i eski kaldı.
8. **Derleme sırasında mod değiştirme.** Derleme sürerken Rendered → Solid → Rendered. Çökme yok;
   Solid'de HUD'da ek satır olarak sayaç görünür; derleme Solid'deyken biterse satır
   `Ray tracing shaders compiled (m:ss), switching to Rendered...` olur ve Rendered'a geçince
   hemen kurulur (`awaiting_install: true` → `ready`).
9. **render.start derleme sırasında.** Derleme sürerken `render.start` → iş `rendering`'de 0
   örnekle bekler, derleme bitince örnekler akar ve PNG yazılır. Bozuksa: iş raster görüntüyü
   kaydettiyse `isAccumulationComplete` derleme sırasında true döndü (`m_rtPipelineInstallPending`
   kontrolü). ★ Sinsi: çıktı PNG Solid görüntüsü — hata vermez, düzgün görünür.
10. **Animasyon/sekans render (tex == nullptr).** Shader değiştikten sonra ilk iş bir sekans
    render olsun: ilk kare derleme bitene kadar bekler (bloklu yol), kare boş/raster OLMAMALI.
11. **Kapanış derleme sırasında.** Derleme sürerken uygulamayı kapat: log'da
    `Shutdown is waiting for the RT pipeline compile to finish.`, sonra temiz çıkış
    (std::terminate / çökme yok). Derleme iptal edilemez; kapanış o kadar sürer.
12. **Hata yolu.** Bir RT shader'ını bilerek boz (ör. `raygen.spv`'yi kes): HUD kırmızı
    `Ray tracing pipeline failed: ...`, viewport Solid çizmeye devam eder,
    `render.rt_pipeline_status` → `failed` + `error`. Aynı shader'larla tekrar denemez
    (dakikalar sürerdi); shader'ı düzeltip yeniden üretince yeni derleme başlar.

Davranış değişiklikleri (bilerek):
- Aynı shader'larla tekrar girişte (`rebuildAccelerationStructure` init bayrağını sıfırlar)
  pipeline artık **yeniden derlenmiyor**; eskiden her seferinde derleniyor ve eski pipeline,
  layout'lar ve SBT sızıyordu. Descriptor set/pipeline layout cihaz başına bir kez kuruluyor.
- "Any-hit olmadan tekrar dene" yolu aslında yalnızca **saç gölge any-hit'ini** düşürüyordu
  (üçgen any-hit'i geri ekliyordu); artık adı buna göre ve yalnızca o aşama varsa deneniyor.
- Yeni IPC/Python: `render.rt_pipeline_status` / `rt.render.rt_pipeline_status()` (yetki: Read).

---

# Sıradaki kullanıcı build (yalnız SHADER): yürüyüş adımı = shader Step çarpanı

Ölçüm matrisi (kare 203, 16 spp, kullanıcı sahnesi): şu an 15.3 s (111 yoğunluk +
60 gölge örneği/ışın); olay 8: 8.6 s; fog 1x: 7.6 s; march 0.5 s (ama march
max_steps 16 + shadow_steps 0 = çok kaba, adil kıyas değil). Petek düzeltmesi
canlı doğrulandı (boşluklar gitti). Yürüyüşün uçuş/gölge adımı artık
max(step_size, 0.25 voxel) — step_size = voxel_step_multiplier x fog voxel.

1. **Varsayılan aynı.** voxel_step_multiplier 0.5'te görüntü ve süre önceki
   build ile aynı olmalı.
2. **Hız düğmesi.** 0.5 -> 1 -> 2: süre kabaca yarı/çeyrek; kar opaklığı
   korunmalı, yalnız en ince tane detayı yumuşar. Kar İNCELİYOR/delikleniyorsa
   adım ince yapıları atlıyor — çarpanı düşür.

# H1 tane referansı — kullanıcı build/canlı 21/21 PASS

- 27-tane .4 s dt-half max konum farkı .285 mm; kütle/momentum/spin/sekme/eğim/yerel S PASS.
- Sekme fixture k 10→20 kN/m kontrollü dt/k ölçümü; gate aynı, solver değişmedi.
- Granül için yeni build yok. Fog paketinin aşağıdaki build notları ayrı korunur.
- Dış IPC: `python scripts/test/rt_h1_grain_reference_ipc.py --preview`.
- Preview yeni ayrı materyalli 27 grain+zemin/camera, static snapshot; timeline DEM değil.
- Kanıt/sınırlar: [MATTER_H1_GRAIN_CONTACT.md](MATTER_H1_GRAIN_CONTACT.md).

# Sıradaki kullanıcı build (C++ + SHADER): yaprak-majorant uçuş + mesafe tabanlı yüzey erozyonu

> **Canlı sonuç + düzeltme (shader-only):** ilk sürüm HATALI — kullanıcı: maliyet düşmedi, spread artınca bloklar boş. Ölçüm (kare 203, 16 spp): 145 yoğunluk/ışın (önce 116), 12.7 s; spread 2 renderı PETEK deseni: kar yalnız yaprakların son voxel diliminde. Kök: 4x fog + spread 0.18 parçacıkları sivri tepeye çeviriyor, yaprak max'ı çok yüksek (sigmaBar ~1e4/m), delta döngüsü 512 örnek sınırında yaprağın KALANINI ATLIYORDU. Düzeltme: (1) delta yalnız sigmaBar*adım < 1 (seyrek) yapraklarda, (2) bütçe biterse kalan düzenli adımla, asla atlama. Aynı kontrol: kare 203, 16 spp — petek YOK, yoğunluk/ışın 145 altına. Ek ölçüm: max events 51 -> 8: 12.7 s -> 8.1 s.

Kullanıcı sahnesi (Physics Domain 1, 5 m, sim voxel 7.35 cm, fog 4x = 1.84 cm,
max_density 10.3, random walk) TABAN ÖLÇÜM, 16 spp: 9.2 s; hacim ışını başına
116 yoğunluk + 41 gölge örneği. Maliyet olay sayısı değil: uçuşlar aktif ama
seyrek yapraklarda yarım voxel adımla yürüyordu.

1. **Maliyet (asıl hedef).** Aynı sahne, aynı kare, 16 spp + volume_counters:
   density_samples/volume_rays 116'dan belirgin düşmeli, süre 9.2 s altına.
   Düşmüyorsa: grid istatistiksiz kuruluyor (leaf max 0 => sigmaBar 0 =>
   yaprak BOŞ sayılıp ATLANIR — bu durumda kar KAYBOLUR, aşağıya bak) ya da
   örnekler son voxel diliminde (regular adım) harcanıyor.
2. **★ Sinsi sonuç: kar delikli/saydam.** Yaprak maksimumu gerçek değerin
   altındaysa (istatistik kapalı/yanlış) delta tracking çarpışmaları kaçırır:
   kar sessizce incelir, hata vermez. Karşılaştır: random_walk açık, aynı
   kare, önceki build görüntüsüyle opaklık aynı olmalı.
3. **Erozyon bandı artık mesafe tabanlı.** depth 1.0 ile ince kar katmanı
   SİLİNMEMELİ (önceki blur tabanlı sürüm gövde bant yarıçapından inceyse her
   şeyi siliyordu). depth = 2-4 sim voxel, strength 0.8: yüzey topaklanır,
   en fazla strength*depth derine keser; daha derin hücrelere dokunulmaz.
4. **Erozyon CPU maliyeti.** 4x gridde chamfer dönüşümü (2 geçiş, 26 komşu)
   tek iş parçacığı: kare başına ~0.5-1 s olabilir. Oynatımda takılma = bu.

# Sıradaki kullanıcı build (C++ + SHADER): random walk similarity geçişi

> Kullanıcı: max events 8 ile 128 arasında görsel fark yok -> varsayılan 128 -> 32 (kayıtlı sahneler kendi değerini korur). Sonuç: ince katmanda maliyet olay SAYISINDAN değil olay başına işten (gölge marşı + gölge ışını + katı probu + 4x gridde 3 mm adım); madde 3 similarity kazancını küçük gösterebilir — sayaçlarla ayrıştır.

Yürüyüş ilk random_walk_exact_events (varsayılan 4, 1..512) olayı gerçek faz ve
sigma ile yürür, sonra similarity ortamına geçer: sigma_s*(1-g), izotropik faz;
g = çift lobun ortalama kosinüsü, yalnız g > 0.05. Kameranın ilk uçuşu hep
tam. Işığa geçirgenlik artık yürüyüşün kendi rwTransmittance'ı (o anki ortamın
sigma'sıyla); lightMarchAcc yürüyüşte kullanılmıyor. GPU: _retired_cloud'dan bir
float daha (random_walk_exact_events), boyut aynı, 4 ayna.

1. **Geri okuma.** fluid.get_fog_shader / gas.get_shader random_walk_exact_events
   (4); 0 -> 1, 999 -> 512 kırpılır.
2. **Görünüm eşdeğerliği.** Kullanıcının sahnesi (g 0.99, saçılma 3.51):
   exact_events 512 (tam) vs 4: genel parlaklık ve gölge dağılımı yakın olmalı;
   yüzeye yakın ileri-saçılma parıltısı biraz farklı olabilir. Belirgin KARARMA
   ya da AÇILMA varsa indirgenmiş albedo/sigma yanlış.
3. **Maliyet.** Aynı sahne 64 spp: 65.7 s (önceki ölçüm, exact) -> belirgin
   düşüş bekleniyor (g 0.99'da olay sayısı ~1/(1-g)). Düşmüyorsa yürüyüşler zaten
   olay sınırına değil gölge/izleme maliyetine takılıyor demektir.
4. **★ Sinsi sonuç: g 0.99'da indirgenmiş ortam neredeyse saydam.** sigma_s'
   = 3.51*0.01*100 = 3.5/m -> 28 cm serbest yol, 12 cm kar katmanını ışık
   neredeyse düz geçer ve zemini aydınlatır. Bu similarity'nin hatası değil,
   g 0.99'un fiziği (tam yürüyüş de aynı yöne gider). Gerçek kar g ~0.8-0.9.

# Sıradaki kullanıcı build (yalnız C++): fog çözünürlük çarpanı + yüzey bandı erozyonu

> **Canlı sonuç (2026-10-05):** 2 PASS (res 5 / depth 2 reddedildi, geri okuma doğru). 3 PASS: 1x→4x gövde parlaklığı korunuyor, kenar keskinleşti (seed karesi okunuyor). 5 PASS: depth 0.05 üst yüzeyi topaklandırdı, gövde delinmedi (etki ince). 6: 64 spp A 1x 28.7 s (ilk render ek yükü dahil), B 4x 47.9 s, C 4x+bant 49.5 s.

Shader değişmedi. Yeni alanlar: `fluid.set_fog resolution_multiplier` (1..4) ve
`erosion_depth` (0..1 dünya birimi); panelde Fog View → "Fog Resolution",
"Surface Band". `splatFogDensityForSelection(int ppc)` →
`splatFogDensityWeighted(float parcel_density)` olarak yeniden adlandırıldı
(int sayıyı ağırlık diye geçirmek sessizce yanlış olurdu).

Ölçüm notu (bu partiden önce): random walk kalıcı maliyeti march'ın ~3x'i
(16 spp: march 1.1 s, walk g0.3 2.1 s, g0.99 3.3 s). Ayar değişikliğinden
sonraki İLK final render ~15 s fazladan sürüyor (aynı ayarla art arda 31 s →
15 s) — yürüyüş değil; RT viewport da final render ile GPU paylaşıyor.

1. **Varsayılanlar birebir eski görüntü.** resolution 1, depth 0: SnowTest
   aynı kalmalı (res 1 + tüm parseller fog iken hâlâ solver grid.density
   kullanılır). Fark varsa yeni dal res 1'de de devreye giriyor.
2. **Geri okuma / red.** `fluid.get fog_resolution_multiplier`,
   `fog_erosion_depth`; resolution 5 ve depth 2 reddedilmeli.
3. **Yoğunluk birimi korunuyor.** res 1 → 2 → 4: aynı spread ile karın genel
   parlaklığı/opaklığı AYNI kalmalı, yalnız kenarlar keskinleşmeli. Res
   arttıkça kar inceliyor/koyulaşıyorsa ağırlık (res³/ppc) yanlış.
4. **★ Spread solver voxel'inde.** res 4'te spread 5.6 = 22 ince voxel sigma:
   CPU blur ~135 tap × 3 geçiş × 6M hücre → oynatımda saniyeler. Yüksek
   çözünürlükte spread'i 0.5–1'e indir; bu bir hata değil ama "uygulama
   dondu" gibi görünür.
5. **Yüzey bandı.** depth 0.03–0.06, strength 0.8, topak 0.04–0.08: düz kar
   levhasının ÜST yüzeyi topaklanmalı, içi delinmemeli. Seyrek serpinti bu
   modda büyük ölçüde silinir — beklenen.
6. **Maliyet.** res 2/4'te march ve yürüyüş ince voxel'de adım atar: render
   süresi kabaca res ile doğrusal artar; bellek res³.

# Sıradaki kullanıcı build (C++ + SHADER): path-traced çoklu saçılma (random walk)

> **Düzeltme canlı (2026-10-05):** ince geniş katman + küp, 64 spp: march 8 s, walk 15 s (~1.9x; düzeltme öncesi ~3x). Madde 6 PASS: küpün gölgesi karda mavimsi (yalnız gökyüzü). Küpün saydam görünmesi malzemeydi (kullanıcı doğruladı), yürüyüş değil.
> **Düzeltme partisi (shader-only, aynı gün):** kullanıcı: gölge var, güneşli taraf fazla saçıyor, çok pahalı. Kök: Nishita modunda yürüyüş güneşi İKİ kez sayıyordu (Directional sahne ışığı + worldData güneş NEE; yüzey shader ikincisini bilerek kapatmış). Nishita NEE söküldü; geometri gölge ışını yalnız hacim geçirgenliği >1e-3 iken; katı probu yalnız uçulan mesafede. Bekleneni: güneşli taraf belirgin sönük (yaklaşık yarı doğrudan ışık), süre 20 s civarından aşağı. Gölge tarafındaki parlamalar büyük olasılıkla güneş DİSKİNE kaçan ışınların firefly'ı (miss 80000x) — gerçek glint değil.

> **Canlı sonuç (2026-10-05):** 2 PASS (random_walk true/128 geri okundu). 3 PASS: kar orta bölge RGB 196/199/204 (march) -> 243/245/248 (walk), zemin 226/238/247 her ikisinde AYNI (kapalı yol değişmedi). 64 spp 7 s -> 20 s (~3x). 5/6/7 henüz bakılmadı; kar düz/az formlu, yere gölgesi bu açıda okunmuyor.

Shader değişti: `volume_closesthit.rchit` (yeni dal), `closesthit.rchit` ve
`volume_intersection.rint` (yalnız struct alan adı). VkVolumeInstance BOYUTU
DEĞİŞMEDİ: emekli `_retired_cloud[8]`'in ilk iki float'ı
`random_walk_enabled / random_walk_max_events` oldu. `params.h` GpuVDBVolume'a
iki int eklendi → CUDA da yeniden derlenir; OptiX bu alanları OKUMAZ.
Açma: `fluid.set_fog_shader {random_walk:true}` / `gas.set_shader` /
panel Edit Fog Medium → Advanced Scattering → "Path-Traced Multiple Scattering".

1. **Kapalıyken birebir eski görüntü.** Varsayılan kapalı; mevcut fog/gaz
   sahnesi piksel piksel aynı olmalı. Fark varsa: bir struct aynası kaydı
   (ilk instance doğru, sonrakiler bozuk) — dört bildirimi karşılaştır.
2. **Geri okuma.** `fluid.get_fog_shader` → `random_walk true`,
   `random_walk_max_events` 8..512'ye kırpılmış. `gas.get_shader` aynı.
3. **Kar testi (SnowTest sahnesi, güneş açık).** random_walk açıkken kar,
   aydınlık zeminle aynı parlaklık ailesinde BEYAZ olmalı; gölge tarafı
   gökyüzünden mavimsi. Hâlâ griyse: (a) `shadow_steps` 0 mı (öz-gölge yok
   ama bu PARLATIR, griliği açıklamaz), (b) yürüyüş dalı hiç çalışmıyor —
   emission_mode ≠ 0, Volume Graph bağlı ya da layered SDF var mı bak.
4. **Gürültü / hız.** Aynı spp'de belirgin daha gürültülü ve yavaş olması
   BEKLENEN. Kare süresi 10x'ten fazla uzarsa olay sayısını 32'ye indirip
   kıyasla; fark büyükse maliyet olay başına iki trace + iki lightMarch'tır.
5. **★ Zemin teması.** Karın oturduğu zemin karın altında GÖRÜNMEMELİ ve kar
   zemine gölge düşürmeli. Kar zeminin içinden "sızıyorsa" yürüyüşün katı
   probu (0xF5) zemini görmüyor demektir.
6. **Nesne gölgesi.** Kar üstüne bir küp koy: küpün gölgesi karda görünmeli
   (yürüyüş NEE'si geometri gölgesi çeker; eski march çekmez).
7. **★ Sinsi sonuç: olay bütçesi.** max_events 32 vs 256: kalın karda 32
   belirgin KOYU ise yürüyüşler bütçeye takılıyor ve kalan enerji siliniyor —
   hata gibi değil "biraz koyu kar" gibi görünür. Bütçeyi artır, albedoyu
   değil.
8. **Renkli absorpsiyon (bilinen yaklaşım).** absorption_color renkliyken
   march'a göre biraz daha koyu/az renkli: uçuş en büyük kanal sönümünde
   örnekleniyor. Gri ortamda (kar) birebir.

RayFusion ve OptiX random walk'u uygulamaz (march/heuristik kalır).

# Sıradaki kullanıcı C++ build: fog erozyonu (kar/pamuk görünümü) + fog medium IPC

> **Canlı sonuç (2026-10-05, build sonrası):** 2 PASS (geri okuma, 3 red), 4 PASS (kenar topaklandı, gövde delinmedi; wet_sand 100k, voxel 2.5 cm, max_density 1.16), 8 geri okuma PASS. ★ AÇIK: güneş altında hacim, aydınlık zeminden GRİ — tek saçılma + multi_scatter heuristiği yüksek albedoyu taşımıyor. Erozyon değil integratör meselesi.

Shader/ABI/.spv değişmedi; yalnız C++. Yeni `.cpp` yok. Erozyon GPU'da değil,
yüklemeden önce grid'de (`Fluid::erodeFogDensity`, FluidDomainFogVolume.cpp) —
Vulkan RT, RayFusion ve CPU aynı NanoVDB'yi okur; OptiX'e dokunulmadı.
Sahne: fog view'u olan bir sıvı/granüler domain (`fluid.set_label_views` body→fog
ya da render_mode fog), timeline duraklatılmış.

1. **Derleme + descriptor.** `fluid.set_fog` parametrelerinde `erosion_strength/
   size/detail/seed`, `fluid.set_fog_shader`'da `anisotropy_back/lobe_mix/
   multi_scatter` görünmeli (`agent.discover`). Yoksa: eski exe ya da üretici
   çalışmadan derlendi.
2. **Geri okuma (bağımsız, hızlı).** `fluid.set_fog {erosion_strength:0.8,
   erosion_size:0.15}` sonra `fluid.get` → `fog_erosion_strength 0.8`,
   `fog_erosion_size 0.15`. `erosion_strength 1.5` ve `erosion_detail 0` RED
   dönmeli (sıkıştırılmamalı). Kabul edip değeri kırpıyorsa: doğrulama atlanmış.
3. **Strength 0 = birebir eski görüntü.** Varsayılan 0; mevcut fog sahneleri
   piksel piksel aynı kalmalı. Fark varsa erozyon kapalıyken de buffer
   kopyalanıp değiştiriliyor demektir.
4. **Görsel: kenar topaklanır, gövde DELİNMEZ.** strength 0.7–1, size 0.1–0.3 m,
   spread 2–3: seyrek kenar pamuk topaklarına bölünmeli, dolu gövde (hücre ≈1)
   bütün kalmalı. ★ Sinsi sonuç: gövde de beneklenmişse hata gibi görünmez,
   "biraz fazla gürültü" gibi görünür. Anlamı: gövde yoğunluğu 1'in altında
   (granülerde sıkışma ya da düşük ppc) — remap 1.0'ı "dolu" sayıyor. O zaman
   bir "full level" parametresi gerekiyor, gürültü ayarı değil.
5. **Ölçek dünya biriminde.** Domain boyutunu/voxel'i değiştir, size sabit:
   topak boyutu metrede aynı kalmalı. Domain'le büyüyorsa noise AABB'ye
   normalize ediliyor demektir (eski material density noise yolu karışmış).
6. **Kare kare titreme yok.** Duraklatılmış karede tekrar sync (spread'i oynat,
   geri al) → aynı desen. Değişiyorsa hash deterministik değil.
7. **Hareketli malzeme (bilinen sınır, hata değil).** Düşen karda topaklar
   malzemeyle gitmez, dünyada sabit durur ("kaynama"). UVW ile taşımak ayrı iş.
8. **multi_scatter / lobe IPC.** `fluid.set_fog_shader {multi_scatter:0.9,
   scattering_color:[1,1,1], absorption_coefficient:0}` → Vulkan RT'de beyaz,
   daha az kontrastlı gölge; `fluid.get_fog_shader` değeri geri vermeli; panelde
   Edit Fog Medium → Advanced Scattering aynı değeri göstermeli.
   ★ RayFusion'da bu üçü ETKİSİZ (shader okumuyor) — erozyon görünür, parlaklık
   değişmez. Bu beklenen; RayFusion pariteleri ayrı iş.
9. **Kaydet/aç.** Erozyon değerleri .rtp'de `fluid_fog_erosion_*` olarak kalıcı.

Not: overlay JSON'da bir `Â§` bozulması (ATMOSPHERE_WEATHER satırı) düzeltildi.
Audit'te `fluid.matter_models` yetki aynası uyuşmazlığı bu partiden önce de vardı.

# Son kullanıcı build — boş sıvı lane canlı kontrol tamamlandı

- Yeni build gerekmiyor. Freefall 4/4 ve dry→water/wet→dry dönüşü PASS.
- 8 s kuru matris kütle/CFL/pressure-off PASS; yerleşme 4/4 RED,
  dt RMS %27.28 / COM %19.08: G2 açık.
- Ortak Matter dispatch 1810→453; kontrollü C7 hız kıyası değildir.
  Pore OFF legacy pure-granular sahne bu optimizasyon kapsamına girmez.
- Tekrar için dış terminalde üç probe `--expect-empty-fluid-skipped` ile:
  `rt_g2_free_fall_ipc.py`, `rt_g2_dry_wet_compare_ipc.py`, `rt_g2_dry_matrix_ipc.py`
  (`scripts/test/` altında). Test runtime resetler, authoring geri yükler; save yapmaz.
- Güncel sonuç: `matter_g2_dry_matrix_empty_lane_skipped_2026-10-05.json`.

Aşağıdaki bölümler önceki build partilerinin tarihsel kontrol notlarıdır.

# Sıradaki kullanıcı C++ build: granül sönümü eşit fiziksel zaman

- timeScaledSubstepDamping: multiplier^(frame_dt/(1/60)/substeps).
  Saf/Matter granular ortak; liquid eski outer-step semantiği. 1/60 referans korunur.
- Shader/ABI/schema yok. GPU/wet/pore/stages ve float32 retention PASS;
  C++ regression source güncel, test target derlenmedi.
- Açık/duraklatılmış uygulama, dış terminal:
  `python scripts/test/rt_g2_dry_matrix_ipc.py --log docs/dev/matter_g2_dry_matrix_time_scaled.json`
- Baseline: matter_g2_dry_matrix_before_time_scaling_2026-10-05.json.
  Son build dry kütle/CFL PASS, yerleşme4/4 RED, dt RMS farkı%26.67.
- Test authoring enabled/visible durumunu geri yükler; runtime reset/frame0;
  test domain/source disabled/hidden. Proje kaydedilmez.
- Yeni time scaling henüz canlı ölçülmedi; sleep/force/contact/packing ayrı açık.

# Sıradaki kullanıcı C++ build: ortak alt adım sönümü

- Shader/ABI değişmedi. GranularStepPolicy ortak multiplier kökü; saf ve iki
  Matter lane outer-step velocity/affine damping ürününü korur.
- Statik GPU/wet/pore/stages + float32 ürün referansı PASS; C++ regression
  matter_granular_damping_test.cpp kaynak hazır, test target çalıştırılmadı.
- Uygulama açık/paused; dış terminal:
  `python scripts/test/rt_g2_dry_wet_compare_ipc.py`
  `python scripts/test/rt_g2_dry_matrix_ipc.py`
- Son eski-binary kuru matris: kütle/CFL PASS; 4/4 yerleşme RED; dt RMS farkı
  %24.32. Sabit baseline matter_g2_dry_matrix_before_damping_2026-10-05.json.
- Testler authoring ayarlarını geri yükler; runtime reset/frame0,
  test domain/source disabled/hidden. Proje kaydedilmez.
- Yeni sönüm henüz canlı ölçülmedi; değişen fizik için eski bake tekrar alınır.

# Kuru/ıslak solver tutarlılığı — kullanıcı build ve canlı tekrar PASS

- Kullanıcı normal C++ derler; shader değişmedi.
- Açık uygulama duraklatılmışken dış terminal:
  `python scripts/test/rt_g2_dry_wet_compare_ipc.py`.
- Üç sonlu koşu: kuru / su+wet kapalı / su+wet açık. Su öncesi COM/RMS
  farkı <=1 mm, kaynak sonrası Sand/su kütle kapıları geçmeli.
- Test authoring ayarlarını geri yükler; runtime resetlenir, frame0 kalır.
  Test domain/sources disabled/hidden kalır; proje kaydedilmez.
- Yeni binary: nötr kuru COM farkı .103 mm, RMS .0093 mm PASS; Sand72 kg
  sabit, su max sapma .726 mg. Hareketli yığın için denge/repose açık.
- Pore/wet/GPU statik kontrolleri PASS; G2/repose/DEM kabulü hâlâ açık.

# Sonraki derleme kontrolleri — prosedürel küre (splat/whitewater) düzeltmeleri

**Yeni G2 kaynak partisi:** mixed granular-only load/CFL ölçümü ve yayın açığı
düzeltildi; normal C++ build, shader aynı. Açık sert sahne ve dış probe:
[MATTER_G2_MECHANICS.md](MATTER_G2_MECHANICS.md). Mekanik kabul henüz açık.

**Güncel Matter partisi — 2026-10-05:** Kullanıcı C6'yı derledi; boş sahne kuruldu.
Wet/drainage/mixed GPU canlı probları PASS. Emisyon sonrası 180 manuel adımda
su farkı +0.32 mg, kuru Sand sabit. C6 tam fiziksel kabul matrisi açık.
Kısa kalan liste: [MATTER_C5_HANDOFF.md](MATTER_C5_HANDOFF.md).
Shader değişmedi. Test komutları:
Kabul ölçümleri ve düşük-S band düzeltmesi kullanıcı derlemesinde canlı PASS.
RT geçici magenta A/B ayrı wet küre materyallerini doğruladı; renk geri yüklendi.
Son kaynak partisi henüz derlenmedi: full-wet görünüm doluluğu .05 default,
UI/Python/IPC/JSON/cache hash/spatial ölçümler birlikte. Normal C++ build;
shader aynı. Pore Water'da Full wet look at pore saturation=0.05, wet appearance
açık; reset/replay sonrası koyulaşma ve roughness aynı wet kürelerde kontrol edilir.
[MATTER_ACCEPTANCE_AND_AUTHORING.md](MATTER_ACCEPTANCE_AND_AUTHORING.md).
Aşağıdaki eski feature kontrolleri tarihsel ve ilgili regresyonlar için korunur.

**Granular GPU occupancy (2026-10-02):** Yeni `sim_fluid_occupancy.comp` i?in ?nce
shader derlemesi/deploy, sonra C++ derlemesi. Bo? ve duraklat?lm?? sahnede d??
terminalden `python scripts/test/rt_test_granular_stiffness_residency_ipc.py` ?al??t?r.
Konum geri okumalar? UVW yenileme/kare sonuyla s?n?rl?; GPU maske ve 512 descriptor
s?n?r?nda g?nderim do?rulanmal?. ?l??m ve fallback kontrol listesi:
[GRANULAR_GPU_OCCUPANCY.md](GRANULAR_GPU_OCCUPANCY.md). S?re kazanc? hen?z ?l??lmedi.

> **Durum:** CANLI — her partide üzerine yazılır

Önce `compile_shaders.bat` (sphere_closesthit.spv artık `closesthit.rchit -DSPHERE_HIT=1`
ile üretiliyor; eski `sphere_closesthit.rchit` silindi), sonra C++ derlemesi.

00. **Sim frame cache: dinamik bütçe + afin sadeleştirme (YENİ, en üstte).**
   `sim_cache.status`: `budget_bytes` artık 4 GiB sabit değil = min(RAM%70, RAM-2GiB) (alt sınır 1 GiB).
   Kontrol: sayı makinenin RAM'iyle uyuşuyor mu. Yakalama ayrıca boş RAM < 2 GiB iken durur.
   Parçacık cache'inde ara karelerde APIC `affine` atılıyor: aynı sahnede 251 kare cache'le, `ram_bytes`
   eskiden 2.74 GB idi (≈10.9 MB/kare) — düşmeli (~%15-30). Bozuksa: scrub'da parçacıklar bozuk görünür
   ya da scrub sonrası oynatma sıçrar (resume anahtar kareden başlamıyor demektir).
   + Parçacık kuantizasyonu (ara kareler): konum/hız/uvw 16-bit, düzgün skalerler (kütle, sıcaklık, bayrak,
   etiket...) tek değere. Aynı sahnede (1M parçacık, 251 kare) `ram_bytes` 24.3 GB idi → ~9-12 GB beklenir.
   Görsel: ara karede scrub'la, anahtar karedeki (25'in katı) görüntüyle yan yana — fark gözle görünmemeli.
   Bozuksa: ara karede parçacık "titreşiyor"/gruplar atlıyor = uvw ya da konum aralığı yanlış.
   ★ Sinsi: `fluid.get` kütle/sıcaklık okumaları ara kareye scrub'ta anahtar kareyle AYNI olmalı
   (lossless sözleşmesi); farklıysa bir skaler yanlışlıkla kuantize olmuş demektir.
   + GRANÜLER (991k parçacık sahnesi 17.4 GB/250 kare = 71 MB/kare idi): ara karelerde deformasyon gradyanı, stres,
   plastik hacim/artış, kırılma geçmişi atılıyor (çözücü-içi; okuyan yok); damage/softening/hardening/yield/bond/
   flags kayıpsız (üniformsa tek değer). Beklenen ≈ 71 → ~25-35 MB/kare. Bozuksa: ara kareye scrub'ta
   `granular_damaged/detached` sayıları anahtar kareyle farklı çıkar (okunan alan kayıp). ★ Sinsi: ara kareden
   oynatınca yığın "yumuşak/akışkan" davranır = ensureGranularStateSize varsayılanı (kimlik deformasyon) ile devam
   ediyor, anahtar kareden değil — `simLiveVelocityValid` false olmalıydı.
   ★ Sinsi: ara kareye scrub edip oynatınca sim "makul" ama sıvı yavaş/ağır davranır = affine sıfır kaldı,
   `simLiveVelocityValid` false olmalıydı. İlk 25 karelik anahtar kareden devam ettiğini doğrula.

0. **A/B ölçüm anahtarı (YENİ).** `fluid.set_splat` (RtApiFluid) `geometry`: `icosphere` = prosedürel küre bulutu,
   `icosphere_mesh` = eski üçgen ikosfer (panelde de 3. seçenek). Aynı sahnede iki değer arasında
   geçip kare süresi + VRAM'i karşılaştır; `fluid.get splat_triangles` prosedürelde 0, meshte 20<<2s.
   Bozuksa: geçişte parçacıklar kayboluyorsa point_sphere_mode bayrağı yapısal değişimi tetiklemiyor
   (ParticleRenderBridge ~1321). ★ Sinsi: iki mod aynı görünür ama ikisi de aynı yolu çalıştırıyordur —
   `splat_triangles` ve VRAM farkı yoksa anahtar etkisiz.

1. **Shader derleniyor mu?** `compile_shaders.bat` "sphere_closesthit" satırında OK.
   Bozuksa: SPHERE_HIT dalındaki `VkGeometryData(...)` yapıcısı veya `foamSpheres`
   tanımı (closesthit.rchit ~706, ~1175). Hata mesajı satırı yeterli.
2. **Görünürlük (en hızlı görülen).** Solid'de sim 0. karede → Rendered'a geç: partiküller
   ilk karede görünmeli. Oynat: 12. kare civarında kaybolma/geri gelme OLMAMALI.
   Bozuksa: TLAS refit (updateInstanceTransforms, "sphere" bloğu) çalışmıyor ya da
   `g_vulkan_rebuild_pending` yolu tetikleniyor (log: capacity). ★ Sinsi hâli:
   partiküller "çoğu zaman" görünür, yalnızca bazı karelerde boş kalır — seyrek
   atlama da bug'dır.
3. **RayFusion/Solid ara sıra çizim atlama.** Aynı sim, RayFusion'da 100 kare oynat; atlama
   olmamalı. Bozuksa ayrı bir neden (raster tarafı: rayFusionExcluded=point_sphere_mode).
4. **Materyal özellikleri.** Splat materyaline SSS / bubble / interior depth ver, Rendered'da
   bak: ikon küreyle aynı görünmeli. Bozuksa: matx/binding 24 okuması sphere varyantında
   farklı davranıyor; ★ sinsi: "düz beyaz/saydam küre" makul görünür ama özellik yok.
5. **Performans/bellek.** Sim oynarken kare süresi ve VRAM'i önceki (ikosfer) yolla karşılaştır.
   Bu parti: tarama başına scratch create/destroy kalktı, buffer kapasiteyle ayrılıyor.
   Hâlâ yüksekse: waitIdle + tek-seferlik komut + fence her karede (aşağıdaki not).

---

## Önceki parti (korunuyor — hâlâ derlenmemiş olabilir)

## Cihaz özellikleri + AS usage bitleri (2026-10-01) — YAZILDI, DERLENMEDİ

`VulkanBackend.cpp`: cihazda `geometryShader`, `fragmentStoresAndAtomics`, `independentBlend`
(destekleniyorsa) açıldı; AS tamponlarına `ACCELERATION_STRUCTURE_STORAGE` biti, HW-RT'de
VERTEX/INDEX tamponlarına `BUILD_INPUT_READ_ONLY` biti eklendi. Derleme: yalnız msbuild.

1. ★ ÖNCE: uygulama açılıyor mu, Solid/MaterialPreview/RayFusion/Rendered görüntüsü eskisi gibi mi?
   Özellik açmak sürücü derlemesini değiştirebilir (scalarBlockLayout'ta bir takılma vardı, sebep
   sürücü güncellemesi çıktı). Bozuksa: üç özelliği tek tek geri kapat.
2. Blend/outline/seçim konturu değişirse (independentBlend artık gerçekten uygulanıyor): söyle.
3. `RT_VK_VALIDATION=1` ile 03614/03673/00605/06340/00704 sayıları sıfıra yaklaşmalı.
4. AÇIK kaldı: 03047 kalan ~15 descriptor yazıcısı (RT set'i + material-preview) — ayrı iş.

## RT descriptor set: uçuştaki trace'e yazma (2026-10-01) — YAZILDI, DERLENMEDİ

`vkUpdateDescriptorSets-None-03047` ×43736: `bindRTDescriptors` her karede tüm RT set'ini
(28+ binding) yazıyordu, önceki karenin trace'i hâlâ uçuştayken. Düzeltme (`VulkanBackend.cpp`):
yazım grubunun hash'i; yalnızca bir tutamaç GERÇEKTEN değiştiyse (resize, sahne yeniden kurulumu)
frame-slot fence'leri beklenir (2 sn sınırlı). Değişmeyen karede yalnızca hash maliyeti, bekleme yok.
Derleme: yalnız msbuild. (Header değişti: `m_rtDescWriteHash` — tam yeniden derleme.)

1. RT'de bir süre izle. FPS öncekiyle aynı (hash ~30 yazma, ihmal edilebilir).
2. `RT_VK_VALIDATION=1` ile: `03047` sayısı belirgin düşmeli. ★ Sıfır olmayabilir: set'e
   yazan başka ~15 yer var (doku/bulut/hacim), onlara dokunulmadı. Kalan sayı hangi binding'in
   yazıldığını söyler.
3. Pencere boyutunu değiştir: kısa bir takılma olabilir (fence bekleme), çökme olmamalı.
4. ★ Sessiz: sahne yeniden kurulunca görüntü bir kare eski kalırsa söyle.

## Device-lost avı kapandı (2026-10-01) — 4 geçişte çökme yok; validation bayrağı kapatıldı

Kök neden: yeniden boyutlandırmada transmission image/sampler drain'siz yıkılıyordu
(`VulkanViewportBackend.cpp`). "Takılma" ise sürücü güncellemesi sonrası shader derlemesiydi.
Derleme: yalnız msbuild (bayrak false).

1. Aynı senaryo, validation kapalı: RayFusion → Rendered çökmüyor, FPS normale döndü.
2. ★ AÇIK (çökertmiyor, log'u 42 MB şişirdi): `vkUpdateDescriptorSets-None-03047` ×43736 —
   material-preview descriptor set'i uçuştaki komut tamponu varken yazılıyor
   (`updateMaterialPreviewTextureDescriptor`, drenajsız; bkz. BUG_VIEWPORT_DEVICE_LOST_ON_PROJECT_OPEN.md).
   Ayrıca `scalarBlockLayout` kapalı bırakıldı (kEnableScalarBlockLayout), binding 8/24 stageFlags düzeltildi.
3. Diğer bilinen validation hataları: AS tampon usage bitleri (03614/03673), derinlik layout'u (01197).

## RT pipeline layout + scalar layout (2026-10-01, 2. tur) — YAZILDI, DERLENMEDİ

Drain düzeltmesi işe yaradı: yeni log'da `vkDestroyImage/Sampler in use` ve `Device lost`
satırı YOK. Log artık RT pipeline kurulumunda bitiyor. Validation'ın o noktadaki hataları:
- `RayTracingPipeline-layout-07988`: binding 8 (atmosfer LUT) closesthit'te kullanılıyor ama
  stageFlags yalnız RAYGEN|MISS; binding 24 (MaterialExt) shadow_anyhit'te kullanılıyor ama
  yalnız CLOSEST_HIT. İkisi de eklendi (`VulkanBackend.cpp` bindings[8], [24]).
- spirv-val "improperly straddling vector": shader'lar scalar block layout ister, cihazda
  `scalarBlockLayout` hiç açılmamıştı. Özellik artık destekleniyorsa açılıyor.
Derleme: yalnız msbuild.

1. Aynı senaryo. Gör: RayFusion → Rendered çökmeden RT görüntüsü geliyor.
   Çökerse: SceneLog'un SON `[VulkanValidation]` satırlarını at (RT pipeline'dan sonrası).
2. SceneLog'da `07988` ve `08737` (spirv-val) satırları kalmamalı. Kalıyorsa hangi binding yaz.
3. ★ Sessiz: RT gölge/atmosfer görüntüsü değişebilir (any-hit artık doğru MaterialExt'i okuyor).

## RayFusion → Rendered device-lost: drain eksikti (2026-10-01) — YAZILDI, DERLENMEDİ

Validation, ilk `Device lost`'tan hemen önce: `vkDestroyImage-01000` + `vkDestroySampler-01082`
(transmission image/sampler, uçuştaki frame-ring komut tamponu kullanıyor). Kaynak: yeniden
boyutlandırma yolu (`VulkanViewportBackend.cpp`, `ensureInteractiveViewportResourcesImpl`)
drain etmeden yıkıyordu. Düzeltme: yıkımdan önce `drainInteractiveViewportInFlight()`.
Derleme: yalnız msbuild.

1. Aynı senaryo (terrain, RayFusion → Rendered). Gör: çökmüyor, SceneLog'da
   `vkDestroyImage-image-01000` / `vkDestroySampler-sampler-01082` yok.
   Hâlâ çöküyorsa: aynı iki VUID hâlâ geliyor mu? Geliyorsa başka bir yıkım noktası
   (`VulkanBackend.cpp:16377` adapter yolu, `setViewportMode`); gelmiyorsa yeni log'u at.
2. ★ Sessiz: validation açık kaldığı için FPS düşük — normal. İş bitince
   `kForceValidationForDeviceLostHunt`'ı false yap.
3. Ayrı, çökertmeyen validation hataları (sonra): `VkAccelerationStructureCreateInfoKHR-buffer-03614`
   (AS tamponunda ACCELERATION_STRUCTURE_STORAGE biti yok), `vkCmdBuildAccelerationStructuresKHR-geometry-03673`
   (vertexData BUILD_INPUT biti yok), derinlik görüntüsü oldLayout uyumsuzluğu (118×),
   `vkMapMemory` device-local belleğe (372×).

## RayFusion → Rendered TDR (terrain) teşhisi (2026-10-01) — YAZILDI, DERLENMEDİ

Gözlem: sahne doğrudan RT'de kurulunca çökmüyor; RayFusion'dan RT'ye dönünce çöküyor.
Log'da ilk `Device lost ... endSingleTimeCommands/vkQueueSubmit` satırı yield/rebuild
satırlarından ÖNCE; yani submit çöken iş değil, onu fark eden ilk çağrı. Eklenen:
`RT_VK_VALIDATION=1` ortam değişkeni (Khronos validation, mesajlar SceneLog'a
`[VulkanValidation]` olarak yazılır). Vulkan SDK kurulu olmalı; yoksa instance açılmaz.
Derleme: yalnız msbuild.

1. `set RT_VK_VALIDATION=1` ile başlat, terrain + RayFusion → Rendered. SceneLog'da
   `[VulkanValidation]` satırlarına bak; çöküşten önceki ilk hata kaynaktır.
2. Aynısını terrain yerine küpte dene (terrain'e özgü mü?).
3. ★ Validation sessizse ve yine çöküyorsa: gerçek GPU zaman aşımı (TDR 4101) —
   söyle, viewport AS yield sırası ve RT'nin ilk AS build'i üzerinde duracağız.

## Bulut: texture sınırında düz kesik hat (2026-10-01) — YAZILDI, DERLENMEDİ

Hipotez (kanıtlanmadı): majorant haritası başlangıç tile'ının 3x3 komşuluğunu sınırlar,
ama rüzgâr kesmesi (`cloudShear`) arama noktasını yükseklikle kaydırıyor (≤5.12·|shear| m/m).
Dik bakan ışın bir segmentte komşuluğun dışına çıkıyor → majorant 0 → bulut tile kenarında
düz kesiliyor. "Bazen" olması (rüzgâr + bakış açısı bağımlı) bununla uyumlu.
Düzeltme: `cloud_rt.glsl` `cloudSegmentLen()` — segment boyu ışının dikeyliğine ve kesmeye
göre kısalır; delta/ratio tracking ve boş-tile atlaması bunu kullanır.
Derleme: yalnız `compile_shaders.bat` + msbuild.

1. **Görsel, kesiğin görüldüğü sahne/kamera:** düz hat kayboldu. Bozuksa: hat hâlâ varsa
   sebep bu değil — hattın yönünü (dünya eksenine paralel mi?) ve kaç m aralıklı tekrar
   ettiğini söyle (tile = extent/256, 150 km'de ~590 m; 2048 doku sınırıysa extent'in kendisi).
2. **Kontrol:** rüzgârı 0 yap, hat değişiyor mu? Değişmiyorsa kesme değil, başka kaynak.
3. ★ **Sessiz:** maliyet — dik bakışta segment kısaldı, majorant fetch'i artar. fps'e bak.

## Bulut aydınlatma + çözünürlük (2026-10-01) — YAZILDI, DERLENMEDİ

Görüntüyle teşhis (RayFusion, katman 2: 12.9 km Cb, sönüm 1.47/m): büyük yapıların altı
**doygun lacivert**. Üç sebep: güneş ışığı yalnız exp(-τ) (Cb'de sıfır), ortam örtülmesi
exp(-0.5τ) (sıfır), ortam rengi zenitten (koyu mavi). Düzeltme (cloud_rt.glsl `cloudMarch`):
difüzyon kuyruğu `Es·pIso·0.5·(1-e^{-τ/8})/(1+0.1τ)`; ortam örtülmesi 1/(1+0.2τ);
ortam = zenit+ufuk ortalaması; zemin sekmesi gölgede ≥%35. Çözünürlük: taban gürültüsü
128³→192³, detay 32³→64³; RayFusion adım 96→128 (yeni projelerde; açık sahnede panelden).
Derleme: `compile_shaders.bat` + msbuild.

1. **Görsel, aynı sahne:** Cb'nin altı **gri** (mavi değil), içi gölgeli ama siyah değil;
   kule yüzeylerinde daha keskin detay. Bozuksa: hâlâ lacivert → cloud_raster.spv eski.
2. ★ **Sessiz başarısızlık:** ince bulutlar (açık gökte cumulus kenarı) öncekinden belirgin
   PARLAKSA difüzyon terimi ince bulutu da besliyor (olmamalı, (1-e^{-τ/8}) kapısı var).
3. **Açılış süresi:** gürültü üretimi ~3.4× texel (tek sefer). Açılışta takılma/TDR olursa
   söyle — üretimi dilimlere böleriz.
4. **Maliyet:** fps'i 96 vs 128 adımla karşılaştır (Clouds → Quality → Realtime steps).

## Faz 3e düzeltme: bulut yüksekliği + kesme (2026-10-01) — YAZILDI, DERLENMEDİ

Ölçüm (önceki derleme, 'scattered', 1800 m katman): en yüksek hücre katmanın %46'sı,
medyan %25 → 2 km genişliğinde ~400 m boyunda "krep". Sebep: yükseklik kısan
çarpanlar üst üste (sqrt(cov) × gelişme × profil × daralma). Her birine taban kondu.
Kesme: düz doğrusal eğim → yükseklikle karesel (taban sabit, tepe eğilir) + konuma
göre şiddet 0.4–1.6 ve yön ±25° (curl dokusu); kazanç 0.025 → 0.015.
Derleme: yalnız `compile_shaders.bat` + msbuild (CloudParams.h yorumu/sabit).

1. **`Probe-CloudVertical.ps1`.** Gör: en yüksek ≥ 17/24, medyan belirgin altında;
   kesme > 0. FAIL "max < 17" → hâlâ krep, söyle. (Precip betiği düzeltildi ve bu
   derlemeyle zaten geçti: 136/441 yağışlı.)
2. **Görsel:** tepeler hücreden hücreye farklı miktarda eğik, tabanlar dik.
   Bozuksa: hepsi aynı eğik → curl dokusu okunmuyor.
3. **Maliyet:** yoğunlukta +1 doku okuması (curl). RayFusion fps'ine bak.
4. **Erozyon (kullanıcı gözlemi: "hücre başına yapılıyor"):** doğruydu — erozyon örtüyle
   ölçeklenmiş değere (≤ cov, ~0.4) uygulanıyordu, her iç bölge eşiğe yakındı ve detay
   her hücreyi her yerde yiyordu. Artık 0..1 şekil üzerinde (çekirdek ~1) aşındırıp sonra
   örtüyle ölçekliyor. Gör: iç kısım dolu/yumuşak, yırtılma yalnız dış kenarda.
   ★ Sessiz yan etki: bulutlar biraz daha yoğun/parlak görünebilir (daha az iç kayıp) — beklenen.

## Faz 3e: bulut dikey yapısı (2026-10-01) — YAZILDI, DERLENMEDİ (4b-1 ile tek test)

Derleme: `compile_shaders.bat` + msbuild (4b-1 ile aynı derleme). Değişenler:
weather map A = hücre gelişme yüksekliği; rüzgâr kesmesi (`precip2.zw`, 0.025 m/m
per m/s, ≤0.5); konvektif gürültü dikeyde ×1/0.6 esnek; hücre başı taban ±%4;
ışık march'ı 40 m'den ikiye katlanan 6 örnek + 3 km uzak örnek; powder terimi;
bulut içinde adım ×0.6 (≥25 m).

1. **`Probe-CloudVertical.ps1`.** Gör: ALL PASS. Kontrol 1 FAIL (max/medyan ≈ 1) →
   A kanalı okunmuyor (weather map eski: bulut tohumunu değiştir, yeniden üretir).
   Kontrol 2 FAIL → kesme yönü ters (`precip2.zw` işareti).
2. **Görsel:** 'scattered' preset, rüzgâr 15 m/s. Gör: çoğu hücre alçak, aralarında
   birkaç kule; kuleler rüzgâr yönüne hafif eğik; kuleler arasında koyu girintiler,
   güneşe bakan kenarlarda ayrışma. Güneşe karşı bakınca kenar parlaması (silver
   lining) **kaybolmamalı**.
3. **Maliyet:** RayFusion fps'ini öncekiyle kıyasla. Bulut içi adım ×0.6 → ~%30–60
   pahalı beklenir. Çok düştüyse söyle (adımı 0.8'e çekeriz).
4. ★ **Sessiz başarısızlık:** bulutlar genel olarak **koyulaştıysa** (sadece girintiler
   değil) powder çok güçlü; bu "daha dramatik" diye kabul edilebilir ama enerji kaybıdır.
   Öğle güneşinde açık gökteki cumulus parlak beyaz kalmalı.

## Faz 4b-1: yağış perdeleri (2026-10-01) — YAZILDI, DERLENMEDİ

Derleme: `compile_shaders.bat` (cloud_common/cloud_rt/cloud_sample → miss, cloud_raster,
cloud_shadow, cloud_sample) + msbuild. `CloudParamsGPU` 208 → 240 B: shader derlenmeden
exe çalışırsa bulutlar çöp okur — **ikisi birlikte**.

1. **`Probe-Weather.ps1` sonra `Probe-Precip.ps1`.** Gör: ALL PASS. Kontrol 2'de "0/441
   wet" → B kanalı/kapsam kapısı sıfır (weather map yeniden üretilmemiş olabilir,
   bulut tohumunu değiştir); "441/441" → kapı yok, her yerde yağıyor.
2. **Görsel, iki mod:** nem 0.93, kararsızlık 0.9, kamera zeminde, ufka bak. Gör:
   küme çekirdeklerinin altında gri, aşağı inen perdeler; rüzgâr 15 m/s'de rüzgâr
   yönüne **eğik**. RT ve RayFusion'da aynı yerde. Bozuksa: RayFusion'da yok,
   RT'de var → cloud_raster.spv eski.
3. **Virga:** nem 0.6'ya indir (türetme açık) → perdeler zemine inmeden incelip biter.
   Panel → Clouds → Precipitation → "Reaches ground" < 1 görünmeli.
4. **Kar:** sıcaklık 271 K → perdeler daha yoğun, beyazımsı, rüzgârda çok daha
   eğik (düşüş 1 m/s).
5. ★ **Sessiz başarısızlık:** perdeler yalnız **gökyüzüne karşı** çizilir (miss /
   RayFusion gök katmanı). Önünde arazi olan perde görünmez — bu bilinen sınır,
   bug değil (4b-2'de yakın parçacıklar + arazi önü ele alınacak).

## Bulut play düzeltmeleri (2026-10-01) — YAZILDI, DERLENMEDİ

Derleme: yalnız msbuild (shader değişmedi).

1. **RT play:** animasyon verisi olmayan sahnede oynatma "hızlı yol"a girip
   `start_render` kurmuyordu; birikim sıfırlanıyor ama yeni kare çizilmiyordu
   (oynatırken otomatik birikim kapalı). Artık RT bulut açıkken her kare çizilir.
   Gör: rüzgâr 15 m/s, play, RT modda bulutlar kayar. Bozuksa: hâlâ yalnız elle
   kare değişiminde kayıyorsa sahnede anahtar kare var demektir → farklı dal, haber ver.
2. **RayFusion çözünürlüğü:** varsayılan bölen 2 → 1 (1/2 rüzgârla sarsıyordu).
   ★ Açık projede kayıtlı değer 2 kalır: panelde *Realtime res divisor = 1* yap.
   Sessiz başarısızlık: 1/1'de hâlâ sarsıntı varsa sorun çözünürlük değil, geçmiş
   (TAA) yeniden izdüşümü — ertelenen titreme maddesiyle aynı.

## Faz 4a: iklimden türetilmiş hava + animasyon kök düzeltmesi (2026-10-01) — YAZILDI, DERLENMEDİ

Derleme: `compile_shaders.bat` (`cloud_noise.comp` B kanalı) + msbuild (yeni .cpp yok).
**Uyumluluk YOK (kullanıcı kararı):** eski `nishita.cloud_*` içe aktarımı silindi;
kayıtlı projelerde `derive_from_climate` anahtarı yoksa AÇIK gelir → katman 0 iklimden.

1. **Animasyon (kök neden):** bulut zamanı TimelineWidget::draw'da yazılıyordu —
   yalnız timeline paneli çizilirken çalışır (IPC set_frame ile ölçüldü: kare 50,
   bulut zamanı 0). Artık ana döngü `scene.timeline.current_frame / fps`'ten türetir
   ve zaman/revizyon değişince iki adapter'a iter. Rüzgâr 15 m/s, play: iki modda kayar.
   Bulut kapatınca `world.cloud_stats` → `rt_rendering=false` olmalı (önce bayat kalıyordu).
2. **`Probe-Weather.ps1`**: 3 kontrol, hepsi OK.
3. Panel → Climate: "Instability" kaydırıcısı; Clouds: "Derive from climate" + türetilmiş
   değerler satırı; açıkken Layer 1'in taban/kalınlık/örtü/tip alanları gri.
4. Nem 0.9, kararsızlık 0.9 → gökte kümülonimbüs kuleleri; nem 0.4 → açık gök.
   Preset uygula → "Derive" kapanır, preset görünümü kalır.
★ Sinsi: türetme açıkken katman 2/3 katman 0 ile çakışırsa **sessizce kapatılır**
(doğrulama çakışmayı yasaklıyor). Orta katman kaybolursa sebep bu — panelde görünür.


## RayFusion siyah gök + oynatmada donuk bulut (2026-10-01) — yalnız msbuild

1. **Siyah gök:** sahneye gas/fluid domain ekle → RayFusion gökyüzü normal kalmalı;
   domain'i sil / yeni proje aç → yine normal. Sebep: viewport pipeline yeniden
   kurulunca material-preview seti YENİDEN AYRILIYOR, binding 25–27 yeni sette
   yazılmamıştı (gök yazılmamış descriptor okuyup siyahtı, oturum boyu). Artık
   bağlanan set izleniyor (`m_cloudBoundPreviewSet`), değişince yeniden yazılır;
   sky/gölge bayrakları da set eşleşmezse kapalı.
2. **Oynatma:** rüzgâr 15 m/s, play — RT ve RayFusion'da bulutlar her karede kayar.
   Sebep: dünya senkron kapıları oynatmanın her karesinde çalışmıyor. Artık ana
   döngü bulut zamanı değişince iki Vulkan adapter'ına `setCloudState` iter
   (Main.cpp, yükleme kapısından sonra); o da birikimi sıfırlar → RayFusion yeniden çizilir.
★ Sinsi: RT'de play sırasında her kare gürültülü (her karede birikim baştan) — beklenen.

## Rüzgârda donmuş bulutlar (2026-09-30) — shader + C++

Kullanıcı: "play'de ve realtime'da bulutlar hareket etmiyor; tek tek karede ediyor".
İki sebep: (1) RT: bulut zamanı WorldData'da değil → oynatmada birikim sıfırlanmıyor,
eski gök kalıyor. Artık `setCloudState` zaman/sürüklenme değişince `resetAccumulation`.
(2) RayFusion: geçmiş, bulutun rüzgârla kaymasını bilmeden izdüşürülüyordu (α=0.06 →
donuk). Artık kare arası sürüklenme (`camUp.w`, `prevUp.w`) çıkarılıyor ve zaman
akarken α ≥ 0.2.
1. Rüzgâr 15 m/s, play: iki modda bulutlar akıcı kayar. Gölgeler de kayar (RayFusion
   haritası her kare, RT doğrudan).
★ Sinsi: RT'de oynatma artık her karede yeniden birikiyor → play sırasında RT
gürültülü (beklenen; durunca temizlenir).

## Bulut şekli + cirrus (2026-09-30) — YAZILDI, yalnız shader derlemesi

`cloud_common.glsl`: 4 cins profil (stratus / stratokümülüs / kümülüs / Cb), iki
ölçekli kabarık yapı (3× büyük hücre modülasyonu), kümülüs tepeye daralır, Cb'de
örs yayılması, tabanda daha güçlü kıvrım (curl) + erozyon 0.35→0.45.
`cloud_rt.glsl`: **cirrus** artık çiziliyor (iki modda): rüzgâr yönünde uzamış
ince buz tabakası, tek saçılma, HG g=0.75.
1. Preset'leri sırayla dene: `fair_weather_cumulus` (ayrık kubbeler, katmanlı
   tümsekli tepeler), `storm` (yüksek kuleler, tepede örs), `overcast_stratus`
   (düz tabaka), `high_cirrus` (ince, çizgili, rüzgâr yönünde).
2. RT ile RayFusion aynı şekilleri göstermeli (tek fonksiyon).
★ Sinsi: bulutlar ortalama olarak **seyreldi/küçüldü** — iki ölçekli modülasyon
(`big = mix(0.55,1,nL.r)`) örtüyü düşürür; preset'lerin örtüsü artık fazla az
görünüyorsa 0.55 → 0.7.

## Faz 3c-2: RayFusion titreme + bulut gölgesi (2026-09-30) — CANLI (gölgeler doğru)

Derleme: `compile_shaders.bat` (yeni `cloud_shadow.comp`; değişen `cloud_raster.comp`,
`material_preview_frag.frag`) + msbuild (preview set 26→28 binding).
1. Log'da "cloud_shadow pipeline unavailable" YOK.
2. Kamera hareketi: titreme gitmeli (geçmiş artık komşuluk varyansına kırpılıyor,
   hareket ağırlığı 0.6→0.15). Hareket bitince ~1 sn'de temizlenir. Hafif bulanıklık
   normal; iz (ghost) kalıyorsa `cloud_raster.comp` clip katsayısı 1.25 → 1.0.
3. RayFusion zemininde bulut gölgesi: RT ile aynı yerde. Güneşi alçalt: gölgeler
   uzar ve kayar. Kamera 16 km'den uzaklaşınca harita kamerayla gelir (kenarda gölge yok).
★ Sinsi: gölgeler RT'ye göre **ayna/kaymış** → harita uv ekseni (x↔z) ya da
izdüşüm işareti (`previewCloudShadow`). Aynı kareyi iki modda kıyasla.

## Faz 3c: RayFusion bulutları (2026-09-30) — CANLI (kullanıcı: "çok güzel çalıştı")

Derleme: `compile_shaders.bat` (yeni `cloud_raster.comp`; değişen `cloud_rt.glsl`,
`material_preview_sky.frag`) + msbuild (yeni .cpp yok; preview set 25→26 binding).
Nasıl çalışır: aynı `cloudMarch` compute'ta ¼ çözünürlükte (`realtime_resolution_divisor`),
`realtime_steps` (96) adım, kare başına jitter + geçmişe yeniden izdüşüm; sky pass
binding 25'ten `sky*a + rgb` birleştirir.

1. Log: "Cloud pipelines ready" ve **"cloud_raster pipeline unavailable" YOK**.
2. RayFusion + Nishita + preset: gökte bulut görünür; RT ile aynı yerde, benzer
   parlaklıkta (aynı fonksiyon). Kamera durunca ~1 sn'de gürültü temizlenir.
3. Kamerayı çevir: kısa süreli hafif iz (ghost) normal; kalıcı iz/titreme = yeniden
   izdüşüm hatası (ekran y yönü ters olabilir: `cloud_raster.comp` pn.y işareti).
4. Bulut ayarı değiştir: eski bulut hemen kaybolmalı (geçmiş anahtarı).
5. Sahne boşken/HDRI modunda: bulut yok, hata yok.
★ Sinsi: bulutlar RayFusion'da RT'den **ayna görüntüsü/kaymış** — yön kuralı
(`vNdc` ↔ uv) tutmuyor. İki modda aynı kareyi yan yana kıyasla.
Bilinen eksik: RayFusion'da bulut GÖLGESİ yok (Beer gölge haritası, sıradaki iş).


## Bulut ambient örtme (2026-09-30) — shader + ★C++ BİRLİKTE derlenmeli

1. C++ derlenmemişse `rt_steps` 16'da kalır (kaba, düz). İkisini derle.
2. Kalın bulut: üst yüzey parlak, iç/alt kısım gri-koyu gradyan olmalı. Hâlâ tek
   beyazsa: `ATMOSPHERE_CLOUDS.md` DEVİR NOTU madde 1 (oktav kalibrasyonu).


## Atmosfer Faz 3b: RT yol izlemeli bulutlar (2026-09-30, atmosfer oturumu) — YAZILDI, DERLENMEDİ

★ **GÜNCELLEME (aynı gün, kullanıcı: "kalitesiz ve aşırı pahalı"):** varsayılan RT
bulutu artık **Nubis/Hillaire marcher** (`cloudMarch`, `cloud_rt.glsl`): örnek başına
tek jitter'lı march (`quality.rt_steps`, vars. 128), 6 örnekli güneş taraması, 3
oktav çoklu saçılma, enerji korunumlu adım. Yol izleyici yalnız
`quality.rt_reference_path_trace=true` ile (kalibrasyon). Yer gölgesi 24 adımlı
optik derinlik. Beklenen: bulutlu kare maliyeti eskisinin ~1/20–1/50'si, birkaç
düzine örnekte gürültüsüz. Sadece shader derlemesi + msbuild (yeni .cpp yok).
★ Sinsi: oktav sabitleri (0.5/0.5/0.5) kalibre edilmedi — referansla yan yana
bakınca marcher belirgin koyu/açık ise bildir; kalibrasyon 3c başında.

**Derleme:** `compile_shaders.bat` — yeni include `cloud_rt.glsl`; değişen
`miss.rmiss`, `closesthit.rchit`, `raygen.rgen`, `photon.rgen`, `hair_closesthit`,
`volume_closesthit`, `volume_intersection.rint`, `cloud_noise.comp`,
`cloud_common.glsl`, `rt_payload.glsl` → **bütün RT shader'ları** yeniden derlenmeli
(world struct'tan bulut bloğu çıktı; eski .spv ile yeni exe = bozuk gökyüzü). msbuild: yeni .cpp yok.

Ne değişti:
- **Söküm:** Vulkan artık eski prosedürel bulut hacmini TLAS'a koymuyor;
  `proceduralCloudDensity`, `volume_type 3` dalları, world struct bulut bloğu
  silindi, `VkVolumeInstance` bulut alanları `_retired_cloud[8]` dolgusu (boyut aynı).
  Eski hacim **yalnız OptiX/CPU** için yaşıyor (dondurulmuş, karar a).
- **RT bulut:** miss shader'ında delta tracking + ratio tracking + güneş NEE +
  kaçan yolda gökyüzü; faz Jendersie–d'Eon (örnekleme HG karışımı). Boş gök
  256² **majorant haritasıyla** atlanır (weather map'ten, 3×3 döşeme maksimum).
  Kamera ışınları `rt_max_bounces`'a kadar; ikincil ışınlar 3 sekmede kesilir
  (panorama 3c'de; `rt_secondary_full_march` kaldırır).
- **Yer gölgesi:** closesthit'te directional ışık NEE'si bulut geçirgenliğiyle çarpılır.
- Bulut güneşi = `sun_intensity` × bulut yüksekliğindeki atmosfer geçirgenliği
  (directional "Sun" ile aynı ölçek).
- `world.cloud_stats`: backend satırında `rt_rendering`; `renderer` artık
  `{vulkan_rt: path_traced, optix: legacy_volume_mapping}`.

Sırayla:

1. **Açık gök, bulut kapalı.** Nishita sahnesi, bulut yok: gökyüzü/güneş öncekiyle
   aynı. Bozuksa (siyah/renkli gök, NaN): world struct ile shader'lar eşleşmiyor —
   bir .spv eski kalmış.
2. **Preset `fair_weather_cumulus` (RT).** `world.cloud_stats` → render satırı
   `rt_rendering=true`. Gökte ayrık kümülüsler; güneş tarafı parlak kenar, alt
   taban daha koyu; ufka doğru pus içinde. Birkaç yüz örnekte gürültü temizlenir.
   Bozuksa: `rt_rendering=false` → doku üretilmemiş (log "Cloud pipelines ready"?);
   bulut yok ama true → majorant 0 (mode 4 dispatch'i) ya da aralık hesabı.
3. **Yer gölgesi.** Güneşi 30–40° yüksekliğe al: zeminde bulut gölgeleri
   yumuşak kenarlı lekeler. Rüzgâr/zaman değişince gölgeler bulutlarla birlikte kayar.
4. **`overcast_stratus`.** Gök gri ve düzgün, zemin gölgesiz ama belirgin karanlık
   (güneş ışığı ~kesilir). Güneş diski görünmez.
5. **Performans (sayı yeter).** `rt.perf` / örnek süresi: bulut açıkken kaç kat
   yavaşladığını not et — 3d bütçesinin girdisi. Kabaca 2–4× beklenir.
6. **OptiX.** OptiX'e geç: eski hacim bulutları hâlâ görünür (paketten).
7. **Probe-Clouds.ps1** (3a'nınki, değişmedi): hepsi OK.

★ **Sessizce makul görünen:** bulutlar var ve güzel ama güneşin tersindeki taraf
**fazla aydınlık/düz** — kaçan yolun gökyüzü katkısı iki kez sayılıyor olabilir
(güneş diski dahil gök → `cloudSkyAmbient` güneşsiz olmalı). Güneşe bakan
kenardaki parlama ile ters taraf arasında belirgin kontrast yoksa bildir.
★ İkincisi: yerdeki gölge ile gökteki bulut **aynı yerde değil** — gölgeler
kaymış görünürse `cloudSphere` hassasiyeti ya da drift işareti.

Bilinen eksikler (bilerek): hacim (sis/gaz) ve saç NEE'si bulut gölgesi almaz;
bulutun içinde/arkasında kalan dağ (geometri) buluttan önce çizilir — bulut
yalnız sahneden kaçan ışınlarda. Işık huzmeleri 3d'de.

---

## Atmosfer Faz 3a: bulut otoritesi + GPU bulut alanı (2026-09-30, atmosfer oturumu) — YAZILDI, DERLENMEDİ

Tasarım: `ATMOSPHERE_CLOUDS.md` §10 (3a). ★ Bu partide **yeni resim yok**:
Vulkan RT hâlâ eski bulut hacmini çiziyor, ama artık yeni otoriteden türetilen
paketle. Fiziksel bulut renderer'ı 3b'de gelir; eski hacim orada sökülür
(3a'da söksem iki derleme boyunca Vulkan'da bulut olmazdı).

**Derleme:** `compile_shaders.bat` — yeni `cloud_noise.comp`, `cloud_sample.comp`
(glob'la derlenir; `cloud_common.glsl` include). msbuild — **yeni .cpp:**
`Atmosphere/AtmosphereClouds.cpp`, `Api/RtApiClouds.cpp`,
`Backend/VulkanDeviceClouds.cpp`; yeni başlıklar `Atmosphere/AtmosphereClouds.h`,
`Backend/CloudParams.h` (hepsi vcxproj + filters'ta).

Ne değişti:
- **`atmosphere::CloudState` tek otorite** (World'de): 3 hacimsel katman (taban,
  kalınlık m; örtü; tip 0 stratus–1 cumulonimbus; sönüm 1/m; damlacık µm;
  erozyon; hücre/detay boyu m), cirrus, weather map, kalite. Sanatsal düğmeler
  (silver/shadow/ambient/absorption/anisotropy/emissive) **YOK**.
- `nishita.cloud_*` artık **paket**: yalnız `World::syncCloudPacket` yazar
  (OptiX / Stylize / CPU / eski Vulkan hacmi okur). Projede **kaydedilmez**;
  proje `atmosphere.clouds` taşır. Eski projeler **bir kez içe aktarılır**
  (etkin/yükseklik/örtü/yoğunluk→sönüm/ölçek→hücre boyu; ışık düğmeleri atılır).
- Rüzgâr bulutta değil **iklimde**: sürüklenme = iklim rüzgârı × timeline
  zamanı (kareye bağlı → deterministik). `World::setCloudTime` TimelineWidget'tan.
- Anahtar kareleri: tek `has_clouds` grubu (`fclds` + `clds`); eski
  `fcd/fcc/.../fcll` ve `cd/cc/...` anahtarları **okunmaz**.
- Panel "Clouds" baştan yazıldı: preset + katmanlar + cirrus + weather + kalite.
- GPU (iki cihazda ayrı, aynı shader): 128³ base + 32³ detail noise, 128² curl,
  2048² weather map (compute ile üretilir, tile'lanır). **Tembel:** hiçbir katman
  açık değilse hiçbir şey üretilmez.
- `cloud_common.glsl`: yoğunluk fonksiyonu + Jendersie–d'Eon faz — 3b/3c'nin
  kullanacağı TEK tanım; bugün yalnız ölçüm uç noktası kullanıyor.
- IPC: `world.get_clouds`, `world.set_clouds` (kısmi yama, katmanlar eleman
  bazında), `world.apply_cloud_preset`, `world.cloud_stats`,
  `world.sample_clouds` (GPU'da yoğunluk / geçirgenlik ölçümü). Python aynı adlar.

Sırayla:

1. **Açılış + log.** Görmen gereken: `[Vulkan] Cloud pipelines ready` iki kez
   (render + viewport). Bozuksa: `.spv` yok (compile_shaders) ya da pipeline
   kurulamadı (log hata der).
2. **`Probe-Clouds.ps1`.** Görmen gereken: hepsi OK (9 bölüm). ★ 7. madde
   (render/viewport paritesi) FAIL ise iki cihaz FARKLI bulut üretiyor —
   3c'de "biraz farklı görünen" bulutların sebebi olur, kimse hata demez.
3. **Eski proje.** Bulutlu kaydedilmiş eski bir sahne aç. Log: "Imported legacy
   cloud settings". Bulutlar hâlâ görünmeli (aynı yükseklik, benzer örtü). Işık
   görünümü biraz değişebilir (sanatsal düğmeler atıldı) — beklenen.
4. **Panel.** Preset "fair_weather_cumulus" → Apply: RT'de bulut görünür (eski
   hacim). Katman 2'yi katman 1 ile çakışacak şekilde aç → panelde turuncu hata,
   değişiklik uygulanmaz.
5. **Anahtar.** Kare 0 ve 100'de "Clouds" anahtarı (örtü 0.2 → 0.6), oynat:
   örtü akıcı değişir. Kaydet-aç: anahtarlar geri gelir.
6. **Tembellik.** Temiz projede `world.cloud_stats`: `noise_ready=false`,
   `weather_ready=false`. Preset uygulayınca ikisi true, `weather_generations`
   1. Sadece rüzgâr/zaman değişince **sabit** kalır.
7. **OptiX (dondurulmuş).** OptiX'e geçip bulutlu sahneyi render et: eski
   bulut kodu paketten beslendiği için bulut görünmeli.

★ **Sessizce makul görünen:** weather map "çalışıyor" ama örtü yüzdesi preset'le
tutmuyor (fBm dağılımı düzleşmemiş) — gökyüzü "biraz fazla/az bulutlu" görünür.
Probe 5. madde dolu kesri ölçer; aralık bilerek geniş (0,03–0,6), 3b'de
görsel olarak tekrar bakılacak.

---

## Atmosfer Faz 2: iklim -> fizik (2026-09-30, atmosfer oturumu) — YAZILDI, DERLENMEDİ

Plan: `ATMOSPHERE_SYSTEM.md` §3 ve §7 Faz 2. Kural: atmosfer simülasyona **ortam +
sınır koşulu** verir, geri yazılmaz (tek yön). Her tüketicinin kendi değeri kalır,
yanında `inherit_atmosphere` kapısı vardır ve API **etkin değeri + kaynağını**
raporlar (`wind_source` / `ambient_source`: `atmosphere` | `local`).

**Derleme:** önce `compile_shaders.bat` — değişen: `sim_msf_ambient.comp`,
`sim_fluid_advect_tail.comp`, `sim_particle_ballistic.comp`,
`sim_fluid_particle_forces.comp`. Sonra msbuild: **yeni .cpp**
`Api/RtApiAtmosphere.cpp` (vcxproj + filters'a eklendi). Sökülenler:
`InstanceGroup::updateWind` (ölüydü), `CloudManager.h` + `_Unused/CloudManager.cpp`
(ölüydü; sabit 20 m/s'in kaynağı).

**Push-constant ABI değişti (üçü de kernel tablosuyla birlikte):**
`sim_fluid_advect_tail` 64→80 B, `sim_particle_ballistic` 40→56 B,
`sim_msf_ambient` boyut aynı (48 B, `params2.w` artık dünya ortamı).
`sim_fluid_particle_forces` 96 B aynı (dört pad alanı yeniden adlandırıldı).
Tablo ile shader ayrışırsa pipeline hatasız kurulur ve çöp okur — bu yüzden
ilk bakılacak şey partikül/sıvı sahnesinin ÇÖKMEDEN adım atması.

Ne bağlandı (varsayılan → sakin dünyada eski davranış):

| Tüketici | Kapı | Varsayılan | Etki |
|---|---|---|---|
| MSF / dünya ısısı | `world.set_thermal inherit_atmosphere` | **kapalı** | ortam = iklim T; kuruma × (1−RH). `reference_kelvin` (kalibrasyon sıfırı) ortamdan AYRILDI |
| Gaz (GridFluid) ortam T | aynı kapı | kapalı | `ambient_temperature` normalize ortam |
| Gaz stratification | `gas.set_settings inherit_atmosphere` | **kapalı** | (g/cp − L) / kelvin_per_unit — FİZİKSEL ve minik (~1e-5/m) |
| Foliage | `scatter.set_wind inherit_atmosphere` | **kapalı** | yön = iklim; hız × v/5, bükülme × (v/5)² |
| Okyanus/göl | `water.set_wind inherit_atmosphere` | **kapalı** | GPU malzemesine iklim rüzgarı (nehir hariç) |
| APIC sıvı | `fluid` domain `inherit_atmosphere` | **açık** | sprey sürüklemesi rüzgâra göreli; yüzey %3 × rüzgâra sürüklenir |
| Partikül | `particle.set_physics inherit_atmosphere` | **açık** | sürükleme rüzgâra doğru: v = W + (v+a·dt−W)·drag |

Sırayla (bağımsız ve hızlı görülen önce):

1. **Açılış.** Partikül veya sıvı içeren bir sahneyi aç, oynat. Görmen gereken:
   çökme yok, adım sayacı ilerliyor. Bozuksa: push-constant ABI ayrışması
   (yukarıdaki üç boyut) — en olası kırılma noktası bu parti.
2. **`Probe-ClimateCoupling.ps1`** (parametresiz): termal yüzey, inherit
   aç/kapa, reference ayrı, red, partikül rüzgârı. Görmen gereken: hepsi OK.
   Foliage/okyanus/gaz için `-ScatterGroup/-WaterSurface/-GasDomain` ver.
3. **★ Sakin dünya değişmez (kabulün kilidi).** Rüzgar 0 iken bilinen bir
   partikül/sıvı test sahnesinin sayısal çıktısı (alive/mean_speed/hash) Faz 2
   öncesiyle AYNI olmalı. Farklıysa varsayılan-açık iki tüketiciden (APIC,
   partikül) biri sakin havada bile bir şey ekliyor demek.
4. **Eski proje.** Faz 2 öncesi kaydedilmiş bir sahne aç. Görmen gereken:
   `world.get_thermal`: `inherit_atmosphere=false`, `reference_kelvin ==
   ambient_kelvin`; sahne aynı ateş/erime davranışında.
5. **Kayıt/yükleme, İKİ serileştirici.** Proje kaydet-aç VE sahne serileştirici
   (`SceneSerializer`): her kapının değeri geri gelmeli (foliage, su, partikül,
   sıvı, gaz, thermal). Biri unutulursa kapı sessizce varsayılana döner.
6. **MSF ortam.** `inherit_atmosphere=true`, iklim T=270. Kızgın bir nesneyi
   (yanan kütük) bir kare oynat: 270 K'e doğru soğur, "293'e"e değil.
   `effective_ambient_kelvin` 270 raporlar.
7. **Foliage.** Rüzgarlı bir foliage grubunda `inherit_atmosphere=true`,
   iklim rüzgârı 10 m/s +Z: yaprak +Z'ye yatar, sallanma hızlanır; rüzgar 0'a
   çekilince DURUR. Görsel + `scatter.get_wind effective_*` (2.0 / 0.4 / +Z).
8. **Okyanus.** Su yüzeyinde inherit aç, iklim rüzgârını 90° döndür: mikro
   dalgalar/Gerstner yönü döner. Anahtarla (timeline) oynatınca DEĞİŞMELİ —
   CUDA FFT emekli olduğundan rüzgar yalnız GPU malzemesinden geçer ve her kare
   yenilenir (WaterManager::update).
9. **APIC sprey.** Sıvı domain'inde bir sıçrama; rüzgar 8 m/s: damlacıklar
   rüzgâr yönüne sürüklenir, durmaz. Yüzeyde çok yavaş kayma.
10. **Partikül.** `linear_drag > 0` bir sistem, rüzgar 6 m/s: hız rüzgâra
    yakınsar. `linear_drag = 0`: rüzgâr İTMEZ (fizik bu; hata değil).
11. **Gaz.** `gas.get_settings`: `effective_ambient_stratification` ~9e-6
    civarı. Açılınca duman sütununun ÜST TAVANI YOK — bu beklenen (fiziksel
    atmosfer 100 m'de tabakalaşmaz); preset'in mantar şapkası için kapalı kal.

★ **Sessizce makul görünen başarısızlıklar** (kimse bug demez):
- APIC/partikül sakin dünyada eskisinden farklıyken "fizik böyleymiş" sanmak
  (madde 3 bunun içindir).
- `inherit_atmosphere` açık ama tüketici HAM alanı okuyor: panel iklimi
  gösterir, simülasyon yerel değerle koşar. Madde 6-10 oynatarak bakar,
  yalnız `get` ile değil.
- Gaz stratification açıkken plume'un tavanının kaybolması "bozuldu" gibi
  görünür; değildir (madde 11).
- Anahtarlanmış iklim + simülasyon önbelleği: imza iklimi yalnız imza anında
  örnekler; iklim anahtarlı bir sahnede önbellek pişirildiği andaki ortamı
  tekrarlar.

Bu partide YAPILMAYAN (bilerek): gazın açık sınırda **rüzgâr girişi** (gaz
çözücüsünün rüzgâr girdisi yalnız Wind force field'lardan geliyor; sınır koşulu
işi ayrı), `sim.world_thermal` düğümünün inherit alanı, FFT okyanus CUDA yolu
(emekli), rüzgâr kayması (shear) ve esinti (§2 — `ClimateState` tek zemin
rüzgârı taşıyor), OptiX/CUDA (dondurulmuş; CUDA leg'i yeni push alanlarını
görmez).

---

## Güneş = directional ışık, iki backend'de de (2026-09-30, atmosfer oturumu) — YAZILDI, DERLENMEDİ

Kullanıcı gözlemi: directional ışık yokken RayFusion sahneyi güneşle aydınlatıyor,
RT yalnız gökyüzüyle. Kodda bulunan: RT dünya güneşini bilerek ışık saymıyor
(`closesthit.rchit` ~2700, `if (false && ...)`); RayFusion ise onu **beş yerde**
sahne ışıklarına EK ışık olarak katıyordu — directional var mı bakmadan. Sun sync
açıkken directional = güneş ⇒ **RayFusion güneşi iki kez sayıyordu.** Karar
(kullanıcı): güneş yalnız directional ışıktır; Nishita'ya geçişte directional
yoksa otomatik "Sun" eklenir.

**Derleme:** `compile_shaders.bat` (değişen: `material_preview_frag.frag`,
`material_preview_sdf_surface.frag`, `material_preview_volume.frag`,
`material_preview_rt_shadow.glsl` include'u) + msbuild (yeni .cpp YOK).

Ne değişti:
- Yeni `rtapi::ensureWorldSunLight` / IPC `world.ensure_sun_light` / Python
  `rt.world.ensure_sun_light()`: sahnede hiç directional yoksa (gizli olan da
  sayılır) dünya güneşine hizalı, geri alınabilir bir "Sun" ekler — yön, şiddet
  = `sun_intensity`, beyaz, disk yarıçapı = tan(güneş açısal yarıçapı).
- `world.set_mode nishita` ve paneldeki Sky Model combo'su (artık aynı rtapi
  çağrısı) bunu **yalnız geçişte** çağırır. Sonradan silinen Sun geri gelmez.
- Panel → Light Sync bölümü: directional yoksa **"Add Sun Light"** düğmesi (zaten
  Nishita'da açılan eski projeler için; geçiş olmadığından otomatik eklenmez).
- RayFusion'dan SÖKÜLDÜ: yüzey shader'ındaki ek güneş ışığı, SDF yüzey ve hacim
  shader'larındaki güneş terimi, gölge atlasındaki güneş kaskadı (bayrak bit4),
  RT gölge maskesinin güneş kapsamı, bounce ışık tablosundaki güneş (+
  transmittance satır kopyası). IPC `rayfusion.*` çıktısından
  `bounce_sun_included` / `bounce_sun_tint_from_lut` anahtarları KALKTI.

Sırayla:

1. **Directional'lı Nishita sahnesi, RayFusion parlaklığı.** Aynı kamera,
   Material Preview'da `render.probe` ortalaması. Görmen gereken: öncekinden
   belirgin DÜŞÜK ve Rendered'e yakın (önceki ölçüm 0,597 vs 0,414, ×1,44).
   Oran ~1,0'a inerse açık "realtime ×1,45 parlak" hatası da buydu → hafıza
   kaydını kapat. Hâlâ ×1,4 ise çift sayım o hatanın sebebi DEĞİLDİ, başka yere bak.
2. **Mod geçişi.** Directional'sız sahne, Solid → Nishita. Görmen gereken:
   ışık listesinde "Sun" belirir, güneş diski ile gölge yönü aynı, Ctrl+Z onu
   kaldırır. Bozuksa: gölge ters yönde → `direction` işareti; ışık yok →
   combo rtapi'ye gitmedi.
3. **Silinen Sun geri gelmez.** Sun'ı sil, güneş açısını oynat, panelde gez.
   Görmen gereken: geri gelmez, panelde "Add Sun Light" düğmesi görünür.
4. **Eski proje (zaten Nishita, directional yok).** Açınca ışık eklenmemeli;
   düğmeye basınca eklenmeli. RayFusion artık bu sahnede güneşsiz — RT ile aynı.
5. **IPC.** `world.ensure_sun_light` iki kez: ilki `created=true,name=Sun`,
   ikincisi `created=false`. `world.set_mode nishita` directional'sız sahnede
   `lights.list`'e bir directional ekler.
6. **Gölge maliyeti.** RayFusion + directional: `material_preview` gölge
   istatistiklerinde güneş için ikinci bir kaskad seti OLMAMALI (eskiden
   directional + güneş = iki set).
7. **★ Sinsi olan: bounce rengi.** Directional'lı iç mekânda RayFusion bounce
   AÇIK. Tavan sıcak kalmalı (güneş artık bounce'a directional üzerinden
   giriyor). Tavan maviye döndüyse bounce directional'ı taşımıyor demektir —
   kimse "bug" demez, "GI biraz soğuk" der.
8. **Gün batımı tonu (bilinen kayıp, bug değil).** Güneş alçakken RayFusion
   yüzey ışığı artık LUT transmittance ile turuncuya DÖNMÜYOR (eskiden dönen
   ek güneşti); RT de hiç dönmüyordu. İki backend eşit. Directional'ı
   transmittance'la tint'lemek ayrı iş.

Açık (bu partide yapılmadı): RT `miss.rmiss` güneş diskini (80000×) ikincil
ışınlara da veriyor — directional varken RT'de de dolaylı çift güneş olabilir
(15 Eylül'de ölçülen "RT tavanı sıcak" muhtemelen buydu). Kamera ışını dışında
diski kapatmak ayrı, ölçülerek yapılacak iş.

---

## Atmosfer Faz 1b: aerial froxel + yükseklik sisi (2026-09-30, atmosfer oturumu) — YAZILDI, DERLENMEDİ

Plan: `ATMOSPHERE_SYSTEM.md` §7 Faz 1b ve "Kaldığımız yer".

**2026-09-30 canlı (derleme sonrası): 1, 2 (Probe-Climate 10/10), 3 (Probe-Aerial
1–4), 5 ve 6 (gökyüzü) GEÇTİ.** Gökyüzü şeritleri iki backend'de BİREBİR (sis
kapalı 0,7458/0,7444, açık 0,9075/0,9075; sis deltası ufukta +0,0781/+0,0781)
→ froxel paritesi ve RayFusion v yönü doğru. Yüzeyde delta metriği
kullanılamaz: sis kapalı taban Rendered 0,414 / Material 0,597 (×1,44 — ayrı,
önceden açık "realtime parlak" hatası); sis baskınken 0,857/0,861. Probe artık
taban farkı >%3 ise BELİRSİZ der. Hava gökyüzünde bilerek 0. Açık: 4, 7–13 (göz
+ eski proje + animasyon + OptiX).

**Derleme:** önce `RayTrophiStudio\compile_shaders.bat` — yeni
`atmosphere_aerial_froxel.comp` (glob'la otomatik derlenir) ve değişenler:
`atmosphere_lut.comp` (ortak başlığa taşındı), `raygen.rgen`, `raster_post.comp`,
`closesthit.rchit`, `hair_closesthit.rchit`, `miss.rmiss`, `volume_closesthit.rchit`
(yalnız dünya struct'ında alan adı; ofset aynı). Yeni include'lar:
`atmosphere_common.glsl`, `aerial_froxel.glsl`. Sonra msbuild (yeni başlık
`Backend/AerialFroxelParams.h` vcxproj'da; yeni .cpp YOK).

Ne değişti:
- Vulkan RT ve RayFusion aynı compute shader'ı (`atmosphere_aerial_froxel.comp`)
  kendi cihazlarında koşar: 32×32 ekran hücresi × 32 görüş-derinliği dilimi
  (0–64 km, karesel), hava + yükseklik sisi tek ortam. RT raygen binding 8
  slot [3]'ten, RayFusion `raster_post.comp` binding 3'ten okur.
- raygen'deki sanatsal aerial formülü (`aerial_min/max_distance`, `aerial_density`,
  `pow(T, distFactor)`) ve ayrı analitik sis karışımı **SÖKÜLDÜ**.
- Sis artık aydınlatılan bir ortam: `fog_color` → `fog_albedo`,
  `fog_sun_scatter` → `fog_anisotropy` (HG g). `fog_height` artık GERÇEKTEN
  kullanılıyor (önceden ölüydü: yoğunluk `origin.y`'ye göreydi).
- Eski projelerde/keyframe'lerde `fog_color`, `fog_sun_scatter`, `aerial_*`
  okunmaz → varsayılana döner (bilinçli, ilke 6).
- IPC/Python: `world.get_aerial`, `world.set_aerial`, `world.atmosphere_stats`.
  Sis ve aerial daha önce IPC'de HİÇ yoktu.
- `Renderer.cpp` animasyon render'ı sis/aerial keyframe'lerini hiç uygulamıyordu;
  artık uygular.
- OptiX ve CPU renderer: eski formül, sökülen alanlar yerine eski varsayılanlar
  (1000 m / 10000 m / 1.0) — donmuş eşleme, davranış değişmedi.

Derlemeden sonra, SIRAYLA:

1. **Log:** `[Vulkan] Aerial froxel pipeline ready (...)` İKİ kez (RT cihazı +
   viewport cihazı). Görmen gereken: iki satır. Bozuksa: `...froxel.spv not found`
   → shader derlenmedi/kopyalanmadı; `Failed to create` → SceneLog'da Vulkan
   doğrulama hatasına bak (descriptor düzeni).
2. **LUT refactor'ü hiçbir şeyi oynatmadı mı** (bağımsız, hızlı):
   `scripts\ipc\Probe-Climate.ps1` → 10/10. Sonra Faz 1a tarifi (aerial KAPALI,
   sis KAPALI): Rendered'da RH 0,1 → ortalama parlaklık **0,8065**, RH 0,9 →
   **0,8272** (09-30 ölçümüyle aynı sahne/kamera). Bozuksa: `atmosphere_common.glsl`'e
   taşıma LUT matematiğini değiştirdi. ★ Sinsi: fark küçük olur (±0,002) ve
   "gökyüzü biraz farklı" diye kimse raporlamaz — sayıyla karşılaştır.
3. **`scripts\ipc\Probe-Aerial.ps1`** (bölgesiz): adım 1–4 GEÇMELİ.
   ★★ En sinsi başarısızlık adım 4'te: kamera SABİTKEN `froxel_dispatches`
   artıyorsa froxel her kare yeniden kuruluyor — görüntü doğru, maliyet gizli.
4. **Göz — RayFusion'da pus VAR mı:** uzak zemini olan bir sahne, Material
   modu. Daha önce RayFusion'da aerial HİÇ yoktu; şimdi uzak tepeler gökyüzü
   rengine doğru solmalı. "Enable Aerial Perspective" kapat/aç → fark görünmeli.
   Bozuksa (hiç fark yok): `world.atmosphere_stats` viewport `froxel_active`
   false mu? True ise binding 3 yer tutucuya bağlı kalmış olabilir
   (`postFroxelBound`).
5. **Yön — sis tabakası ALTTA mı:** sisi aç (density 0,002, height 50, falloff
   0,01), Material modunda sis ekranın ALTINDA (zeminde) yoğun olmalı.
   Üstte/gökyüzünde yoğunsa RayFusion'un v ekseni ters (`raster_post.comp`
   `fuv = 1 - uv.y`). RT'de aynı görüntü doğruysa ve RayFusion tersse sebep budur.
6. **Parite:** `Probe-Aerial.ps1 -Region x,y,w,h` (uzak zemini kapsayan bölge,
   gökyüzü girmesin). Sis ve hava için `|dRT-dRF|/max ≤ %5`. İkisi de ~0 ise
   froxel kuruluyor ama okunmuyor (★ sayaç bunu göstermez).
7. **`fog_height` artık etkili:** height 50 → 500 arası sürükle; sis tabakasının
   üst sınırı yükselmeli. Önceden bu kaydırıcı hiçbir şey yapmıyordu.
8. **Gökyüzü + sis:** ufka bak, sis açık: ufuk bandı sislenmeli, iki backend'de
   de. Sis kapalıyken gökyüzü aerial yüzünden ÇİFT puslanmamalı (gökyüzü
   pikselleri yalnız sis bloğunu okur).
9. **DoF + sis (RayFusion):** açıklık aç, uzak sırtı bulanıklaştır: bulanık sırt
   odaktaki halinden DAHA NET/koyu görünmemeli (her DoF örneği kendi
   derinliğiyle puslanıyor).
10. **HDRI / Color dünya modu:** sis açık → RT'de eskisi gibi düz renk
    (fog_albedo radyans olarak), RayFusion'da da artık görünür. Bozuksa
    (RT'de sis kayboldu): LUT yokken froxel kurulmuyor — `froxel_active`'e bak.
11. **Eski proje aç:** çökme yok; sis rengi/güneş saçılımı varsayılana döndü
    (bilinçli); aerial min/max kaydırıcıları panelde yok.
12. **Sequence render (viewport yolu):** sis yoğunluğunu kare 0 ve 50'de keyle,
    sequence render al: sis değişmeli. (2026-09-30 ek: ölü `render_Animation`
    worker'ı ve `updateAnimationState` içindeki ikinci dünya keyframe
    uygulayıcısı SÖKÜLDÜ; tek uygulayıcı `TimelineWidget::draw`.) Ayrıca
    dosya animasyonlu (FBX/glTF) bir sahnede güneş/sis keyleri oynatmada
    doğru mu — eskiden o durumda iki uygulayıcı üst üste yazıyordu.
    Derleme hatası çıkarsa: `render_Animation`/`pending_anim_transform_updates`
    başka bir çeviri biriminde kullanılıyordu demektir (grep temizdi).
13. **OptiX (CUDA varsa):** sis/aerial eskisi gibi görünür (donmuş eşleme).
    Bozuksa: `ray_color.cuh` derlenmemiş (`compile_ptx.bat`).

14. **Güneş Mie çekirdeği (2026-09-30 ek):** kullanıcı gözü — RT'de güneşin
    hemen çevresinde keskin Mie çekirdeği vardı, RayFusion'da yalnız geniş hale.
    Sebep: LUT Mie fazını 2,0'de kırpıyor, kırpılan kısmı yalnız `miss.rmiss`
    geri ekliyordu. Artık ortak `sky_sun_corona.glsl`, iki shader'da aynı terim.
    Derle (`miss.rmiss`, `material_preview_sky.frag`); Material ve Rendered'da
    güneş çevresi aynı görünmeli, RT'nin görünümü DEĞİŞMEMELİ (terim birebir
    taşındı). Bozuksa (RT'de çekirdek kayboldu): `mu` kapsamı ya da include.

Bilinen risk: RayFusion derinliği D32 standart-Z; uzak mesafede (> ~5 km)
çözünürlük kaba → çok uzak zeminde pus BANTLANABİLİR. Görülürse not et, reversed-Z
ayrı iş.

## Atmosfer Faz 1a: iklim otoritesi + ölü nem canlandı (2026-09-30, paralel ajan — atmosfer)

**✔ 2026-09-30 CANLI DOĞRULANDI: 1–5 GEÇTİ** (Probe-Climate 10/10; nem LUT maliyeti
kontrolle eşit; RH 0,1→0,9 iki backend'de +0,021 parlaklık, geri dönüşlü, NaN 0;
kullanıcı gözle onayladı). **Açık: 6 (keyframe), 7 (kaydet/aç), 9 (OptiX).**
Ayrıntı: `ATMOSPHERE_SYSTEM.md` §7 ve "Kaldığımız yer".

Plan: `ATMOSPHERE_SYSTEM.md`. Shader: yalnız
`atmosphere_lut.comp` değişti — **`.spv` DERLENMEDİ**, `RayTrophiStudio\compile_shaders.bat`
ile derle (x64\Release\shaders'a kendisi kopyalar).

- Yeni `atmosphere::ClimateState` (`Atmosphere/AtmosphereClimate.h/.cpp`, vcxproj'da)
  sıcaklık (K), lapse rate, bağıl nem, basınç ve rüzgarın TEK sahibi. `World` tutar.
- `nishita.humidity/temperature` ve `weather.wind_*` artık `derived_*` adlı
  **paket** alanları; yalnız `World::syncClimatePacket()` yazar.
- ★ Nem hiçbir renderer'da okunmuyordu (Vulkan `weather.x`'e yazıp shader
  okumuyordu; CPU/OptiX LUT hiç bakmıyordu). Artık higroskopik Mie büyümesi:
  `(1-RH)^-0.5`, iki LUT'ta aynı formül.
- ★ Render sırasındaki keyframe uygulayıcısı (`Renderer.cpp`) nem/sıcaklık
  anahtarlarını HİÇ uygulamıyordu. Artık iklim bloğu olarak uygular.
- IPC/Python: `world.get_climate`, `world.set_climate`, `world.sample_climate`.
  `world.set_atmosphere` eski `humidity/temperature` anahtarlarını **reddeder**.
- `makeAtmosphereLUTParamsGPU` iki kopyadan tek başlığa (`Backend/AtmosphereLutParams.h`).
- Panel: yeni **Climate** bölümü (her dünya modunda). Atmosphere'den nem/sıcaklık,
  Weather'dan rüzgar kaydırıcıları kalktı. Keyframe: tek `Climate` grubu.

Derlemeden sonra, SIRAYLA:

1. **Derleme.** Hata `humidity`, `temperature`, `wind_speed`, `wind_direction`,
   `weather_wind_*`, `has_humidity` içeriyorsa: taşınmayan bir çağıran kaldı —
   bana dosya:satır ver. `AtmosphereLUTParamsGPU` "redefinition" hatası: eski
   kopyalardan biri geri gelmiş (paralel düzenleme).
2. **`.\scripts\ipc\Probe-Climate.ps1` → PASS.** Bağımsız ve sayısal; ilk bu.
   - 1. adım FAIL (applied_* uyuşmuyor) = ayna kopuk, panel yalan söylüyor.
   - 2. adım FAIL = ISA formülü yanlış (p(1000 m) ≈ 89875 Pa, ρ₀ ≈ 1,225 beklenir).
   - 3. adım FAIL = eski anahtar sessizce kabul ediliyor; eski scriptler "çalışıp"
     hiçbir şey yapmaz.
3. **Panel.** World → **Climate** bölümü HDRI modunda da görünmeli.
   Atmosphere'de Humidity/Temperature, Weather'da Wind kaydırıcısı **olmamalı**.
4. **Nem görünür mü (asıl kazanç).** Nishita, gün ortası. Climate → Rel. Humidity
   %10 → %90: ufuk pusu belirgin artmalı; **hem Rendered (Vulkan RT) hem Realtime
   (RayFusion)** aynı yönde değişmeli (ikisi aynı LUT shader'ını koşar).
   - ★ **En sinsi başarısızlık:** `world.get_climate` → `applied_mie_humidity_scale`
     değişiyor (ör. 3,16) ama görüntü değişmiyor. Bu kod hatası değil, **eski
     shader**: `compile_shaders.bat` koşmadıysa `x64/Release/shaders/atmosphere_lut.spv`
     hâlâ 2026-09-29 tarihli (12368 bayt) ve `weather.x`'i okumuyor. Kimse bunu
     bug diye raporlamaz, "nem az etkili" sanılır. (OptiX/CPU yolu shader'sız
     olduğu için orada nem yine görünür — iki backend'in ayrışması bu ipucudur.)
5. **`Probe-AtmosphereCost.ps1 -Field surface_relative_humidity` → OK.** Nem
   düzenlemesi LUT'u kirletir; kontrolden 2 kat pahalıysa CPU LUT'a düşülüyor.
6. **Keyframe.** Kare 0'da Climate'i anahtarla (15 °C, %10), kare 50'de (−10 °C, %80).
   Sürükleyince panel değerleri takip etmeli. **Animasyon render'ında da**
   değişmeli (önceden render yolu bu anahtarları yok sayıyordu).
7. **Kaydet/aç.** İklim değerleri korunmalı (proje JSON'unda `atmosphere.climate`).
   Eski proje açılınca iklim varsayılana döner (15 °C, %10, 0 m/s) ve eski
   nem/sıcaklık/rüzgar anahtarları kaybolur — **beklenen**, geriye uyum yok.
8. **Beklenen küçük fark:** varsayılan sahneler ~%5 daha puslu (RH 0,1'de Mie
   ×1,054; önceden nem hiç uygulanmıyordu). Belirgin fark varsa ölçek hatasıdır.
9. OptiX (CUDA varsa): nem OptiX gökyüzünü de değiştirmeli (CPU LUT yolu).

## Hareketli sıvı→gaz sınırı + termal faz ledger — 27D (2026-09-30)

**DERLENDİ; CANLI IPC KABULÜ PASS. Faz 3 kapandı. Shader değişmedi.**

- Yeni `FluidGasMovingBoundary`, gaz domain'iyle örtüşen bütün canlı sıvı
  parsellerini gaz çözünürlüğünde fiziksel hacim doluluğuna çevirir. Hücre %25
  dolulukta sınır olur; önceki sınır hücresi %15'e kadar tutulur. Hız, parsel
  kütlesiyle ağırlıklandırılır; `mist` dışarıda bırakılır çünkü 27C onu gerçek
  gaz envanterine aktarır.
- Üretilen hücreler yeni bir sınır sistemi kurmaz: mevcut
  `clear/applySubstanceSolidOverlay` ve yüz-ağırlığı geri yükleme yolu kullanılır.
  Böylece collider önbelleği delinmez, hayalet hücre kalmaz, CPU ve Vulkan aynı
  `solid`, `solid_vel` ve kapalı MAC yüzlerini tüketir.
- `gas.step_stats` hem IPC hem Python'da `liquid_boundary_cells` ve
  `liquid_boundary_mean_velocity` verir. `sim_graph.couplings` gerçek çalışma
  olduğunda `liquid_moving_boundary` kaydı üretir.
- Yeni `FluidThermalPhaseExchange`, `updateThermalFreeze` öncesi/sonrası fazı
  karşılaştırır. Donan ve çözülen her madde için fiziksel
  `rest_mass_kg * mass_fraction`, duyulur enerji, füzyon gizli ısısı ve momentum
  aynı `MatterExchangeLedger` adımına dengeli `freezing`/`melting` olayı olarak
  yazılır. Termal zinciri kapatıp frozen bayraklarını temizlemek de gerçek bir
  çözülme olayı olarak görünür.
- Yeni dosyalar VS proje/filters içinde kayıtlı. Proje XML'i, Python sözdizimi,
  iki test kopyasının byte eşitliği ve IPC descriptor/security denetimi PASS:
  644 yöntem, 623 belgeli, 508 parametreli.

Derlemeden sonra boş sahne ve açık uygulamada:

1. `python scripts/test/rt_test_fluid_phase3_boundary_freeze_ipc.py`
2. Beklenen: `liquid_boundary_cells > 0`, üç bileşenli ortalama sınır hızı,
   `liquid_moving_boundary` coupling izi ve en az bir `freezing` olayı.
3. Ledger satırı ve toplamı için kütle farkı `0`, enerji farkı en fazla `1e-5`.
4. Test kendi gas/fluid domain'lerini ve gas kaynağını temizlemeli; sahne yine
   boş kalmalı.
5. PASS sonrası Faz 3 kapanır; sıradaki çekirdek iş Faz 4 gerçek domain
   birleşmesidir.

Canlı kabul sonucu (2026-09-30): 1. adımda 168, 2. adımda 335 parsel dondu;
gaz sınırı 2. adımda 85 hücreye ulaşıp kararlı kaldı. Dengeli `freezing` olayı
0,1900000125 kg taşıdı; toplam kütle ve enerji farkı 0. Coupling izi görüldü,
test PASS verdi ve geçici sahneyi temizledi.

## Gerçek mist→gaz faz aktarımı + fiziksel gaz taşınımı — 27C (2026-09-30)

**DERLENDİ ve canlı IPC'de doğrulandı. Shader değişmedi.**

- Yeni `FluidMistPhaseExchange`, daha önce yalnız fog olarak çizilen ve gaz
  sürüklemesi alan `mist` parselinin kalan
  `rest_mass_kg * mass_fraction` değerini APIC’ten tek seferde borçlanır.
  Parçacık kaldırılır; aynı kg, duyulur enerji, gizli ısı ve momentum
  `mist_to_gas` defter olayına ve gazın fiziksel kg/J hücre alanına yazılır.
- Gazın boyutsuz density/fuel/temperature kanalları görünür çözücü tracer’ı
  almaya devam eder. Yanıcı mist yakıt olur; bütün mist gaz yoğunluğunda görünür
  kalır. Fiziksel muhasebenin sahibi tracer değil, kg/J yan alanlarıdır.
- Bir sıvı domain’i adım başına yalnız ilk geçerli örtüşen gaz domain’ine
  aktarılır. 27B yanma kaybı ile 27C artık mist tüketimi aynı eşleştirme kapısını
  ve aynı `MatterExchangeLedger` adımını kullanır. CPU ve Vulkan gaz yolları
  aynı host çekirdeğini çağırır.
- Fiziksel `gas_phase_mass_kg` / `gas_phase_energy_j` hücreleri artık çözülmüş
  gaz hızıyla, domain’in Semi-Lagrange/MacCormack ve open/closed/periodic sınır
  kuralıyla taşınır. Görsel dissipation fiziksel kütleyi silmez. Kapalı/periodic
  sınırda toplam yeniden normalize edilerek korunur; açık sınır yalnız kayba
  izin verir, sayısal kütle üretimi engellenir.
- Ortak `FluidDomainInfo` iki yeni ölçüm taşır: `gas_phase_active_cells` ve
  `gas_phase_mass_centroid`. `fluid.get/list_domains` IPC ve Python aynı çekirdek
  değerleri verir; böylece uzamsal taşınım yalnız toplam kg ile varsayılmaz,
  doğrudan ölçülür.
- `rt_test_fluid_mist_ipc.py` artık `mist_to_gas` olayı ve coupling izini,
  APIC parçacık azalmasını, aktif fiziksel gaz hücrelerini ve +X taşıyıcı hız
  altında kütle merkezinin en az 0,005 m ilerlemesini ister. Kaynak ve dağıtılmış
  script kopyaları eşit.
- Yeni `.h/.cpp` VS proje ve filters dosyasında kayıtlı. Proje XML’i, Python
  AST ve script hash eşitliği PASS. Güncel descriptor toplamı 27D ile 644 yöntem.

Derlemeden sonra, boş sahnede ve uygulama açıkken sırayla:

1. `python scripts/test/rt_test_fluid_mist_ipc.py` — beklenen: önceki mist/fog/
   drag/vaporization kabullerine ek olarak `mist_exchange_seen=true`,
   `mist_coupling_seen=true`, parçacık sayısında düşüş ve
   `max_mist_centroid_x > first_mist_centroid_x + 0.005`; cleanup temiz.
2. `python scripts/test/rt_test_matter_exchange_ledger_ipc.py --step` — boş
   aktarım adımı, kütle/enerji farkları 0.
3. `python scripts/test/rt_test_fluid_labels_cache_ipc.py` — SimCache v8 yaz/oku
   ve etiket regresyonu PASS, unknown 0, cleanup temiz.
4. Bu kabul tamamlandı. Hareketli sıvı sınırı ve donma defter kayıtları 27D'de
   kodlandı; 27D canlı kabulünden sonra Faz 4 gerçek domain birleşmesine geçilecek.

Canlı kabul sonucu (2026-09-30):

- Genişletilmiş mist testi PASS. 5.184 parçacığın 10. adımında 2.554 mist ve
  canlı fog oluştu; 11. adımda 2.554 mist parseli APIC’ten kaldırıldı ve
  7,04276854 kg için dengeli `mist_to_gas` olayı yazıldı. Sonraki adımda yeni
  2.181 mist de aktarıldı; parçacık sayısı 449’a indi.
- Fiziksel gaz envanteri aktif kaldı ve +X taşıyıcı hızını izledi: olay sonrası
  kütle merkezi x değeri `11,1317377 → 11,1563578 m` ilerledi (0,0246201 m;
  kabul eşiği 0,005 m). Son ölçülen gaz envanteri 948,087259 kg ve
  1.397,904272 MJ. Mist drag, fog, vaporization ve `mist_to_gas` coupling
  izlerinin tamamı görüldü; ledger kütle/enerji farkları 0.
- İlk koşu ürün hatası olmadan testte takıldı: olay görülür görülmez çağrılan
  `fluid.set_combustion(enabled=false)` simülasyon durumunu yeniden kurup gaz
  yan alanlarını sıfırlıyordu. Mid-probe ayar mutasyonu kaldırıldı; yeniden
  derleme gerektirmeden ikinci koşu geçti. Test içinde sim ayarı değiştirme.
- Ledger regresyonu PASS: `step=53`, `events=0`, kütle ve enerji farkı 0.
- SimCache v8/etiket regresyonu PASS: disk frame 5’te 5.324 body, unknown 0.
  Bütün geçici kaynak/domainler temizlendi; cleanup hatası ve kalan domain yok.
- Raporlar: `.tmp/fluid_mist_live.json` ve `.tmp/fluid_labels_cache.json`.

## APIC→gaz tek kaynaklı fiziksel aktarım — 27B (2026-09-30)

**DERLENDİ ve canlı IPC'de doğrulandı. Shader değişmedi.**

- Yeni `FluidCombustionExchange` APIC `mass_fraction` kaybını tek kaynak
  otoritesi yapar. Aynı `lost_fraction * rest_mass_kg` değeri gaz hedefinin
  fiziksel kg/J yan alanına ve `MatterExchangeLedger` vaporization olayına
  yazılır.
- Boyutsuz gaz `fuel/density/temperature/interaction` alanları yalnız solver ve
  render tracer'ıdır. kg→tracer dönüşümü tek yerde
  `lost_kg / (liquid_density * fluid_voxel_volume)` olarak yapılır.
- Eski `sim_fluid_surface_combustion` shader'ının yanıcı sıvı yatırımı artık
  çağrılmaz. Shader yalnız extinguishing yüzey quench yolunda çalışır; böylece
  hücre başına yakıt yatırımı ile parsel başına kütle kaybının çift sayımı
  kapandı.
- Bir sıvı domain adım başına yalnız ilk geçerli örtüşen gaz domain'ine aktarım
  yapar. Örtüşen iki gaz kutusu aynı parseli iki kez harcayamaz.
- Gaz durumu `gas_phase_mass_kg` ve `gas_phase_energy_j` hücre yan alanlarını
  taşır. `fluid.get` / `fluid.list_domains` IPC ve `rt.fluid.get/list_domains`
  Python aynı toplamları ortak `rtapi` üzerinden raporlar.
- SimCache v8 iki fiziksel gaz alanını da saklar. v7 cache yeniden bake ister.
- `rt_test_fluid_mist_ipc.py` artık pozitif `vaporization` olayı, olay ve özet
  korunum farkları, ayrıca gaz fiziksel envanterinin hedef kg/J'yi kapsadığını
  doğrular. İki script kopyası eşit; Python AST, proje XML ve 635 yöntemlik IPC
  yetenek denetimi PASS.

Derlemeden sonra:

1. `python scripts/test/rt_test_fluid_mist_ipc.py` çalıştır. Beklenen: önceki
   5.184 parçacık/mist/fog/drag kabulüne ek olarak `exchange_seen=true`, pozitif
   `last_exchange.target_mass_kg`, sıfıra yakın mass/energy error ve pozitif
   `gas_inventory.mass_kg/energy_j`.
2. `python scripts/test/rt_test_matter_exchange_ledger_ipc.py --step` PASS
   olmalı; boş aktarım adımında `step>0`, farklar 0.
3. `python scripts/test/rt_test_fluid_labels_cache_ipc.py` SimCache v8 ile PASS
   olmalı ve cleanup hatası vermemeli.
4. Kabul sonrası 27C gerçek artık mist→gaz transferini aynı fiziksel alan ve
   deftere bağlayacak. Gaz kg/J alanlarının uzamsal adveksiyonu bu sonraki
   taşıma partisinde ele alınacak; 27B kaynak/hedef yatırımı ve toplam envanteri
   kurar.

Canlı kabul sonucu (2026-09-30):

- Mist/yanma kabulü PASS: 5.184 parçacığın 10. adımında 2.629 body, 1 spray ve
  2.554 mist; fog görünümü canlı, `mist_gas_drag` izi var. Son vaporization
  olayı 95,529923 kg ve 59,590214 MJ'yi kaynak/hedefte eşit kaydetti; olay ve
  defter kütle/enerji farkları sıfır. Gaz fiziksel envanteri 803,468031 kg ve
  1.174,584914 MJ; cleanup hatası ve kalan domain yok.
- İlk tanı koşusunda 20,7407572628 kg olay toplamı ile hücre başına `float`
  envanter toplamı arasında 3,65e-7 kg yuvarlama farkı görüldü. Ürün aktarımı
  değişmedi; kabul eşiği, uzamsal `float` alan için ölçeğe bağlı 2e-6 bağıl
  paya çevrildi ve sonraki koşu geçti.
- Zorunlu adımlı ledger regresyonu PASS: `step=13`, `events=0`, kütle ve enerji
  farkı 0.
- SimCache v8 etiket/cache regresyonu PASS: disk frame 5'te 5.324 body,
  unknown 0; cleanup hatası yok.
- Raporlar: `.tmp/fluid_mist_live.json` ve `.tmp/fluid_labels_cache.json`.

## Fiziksel APIC parsel kütlesi — 27A temel partisi (2026-09-29)

**DERLENDİ ve canlı IPC'de doğrulandı. Shader değişmedi.**

- `FluidParticles::rest_mass_kg`, `mass_fraction == 1` iken parselin temsil
  ettiği fiziksel kg değeridir. Taşıma, swap-remove ve compaction bu yan alanı
  parçacıkla birlikte korur.
- `FluidPhysicalMass` yalnız eksik/geçersiz değerleri başlatır. Etiketli
  parçacık kanonik `SubstanceProfile::liquid_density` değerini; etiketsiz
  parçacık domain kimya preset'ini kullanır. Nominal değer
  `density * voxel_size^3 / particles_per_cell` formülüdür. Madde etiketi
  araması sabit hash tablosudur; her kare madde listesi taranmaz.
- MSF→APIC eriyik aktarımı baştan yaklaşık değer üretmez: başarıyla borçlanacak
  `spawn_mass / spawn_count` değeri doğrudan her yeni parsele yazılır ve
  rollback ile birlikte geri alınır.
- SimCache v7 hem `mass_fraction` hem `rest_mass_kg` saklar. Eski cache güvenli
  biçimde reddedilip yeniden bake edilir; cache'den canlı devam fiziksel kg
  envanterini değiştirmez.
- Yeni mantık odaklı `FluidPhysicalMass.h/.cpp` modülündedir. Büyük
  `ParticleSimulation.cpp` yalnız bir başlatma çağrısı aldı. VS proje/filters
  XML kayıtları ve XML doğrulaması PASS; `diff --check` yeni dosyalarda PASS.

Derlemeden sonra:

1. Çözüm derlenmeli; özellikle yeni `FluidPhysicalMass.cpp` projeye girmiş
   olmalı. Bu partide SPV üretimi gerekmez.
2. Uygulama açıkken
   `python scripts/test/rt_test_matter_exchange_ledger_ipc.py` çalıştır.
   Beklenen: PASS, `step=1`, sonlu toplamlar ve sıfıra yakın kütle/enerji farkı.
3. `python scripts/test/rt_test_fluid_mist_ipc.py` çalıştır. Beklenen: önceki
   kabul gibi mist üretimi, fog ve `mist_gas_drag` PASS. Bu parti görünür
   davranışı değiştirmemeli.
4. Varsa mevcut bir sim cache'i açmayı dene: v6 cache sessizce okunmamalı;
   sürüm uyuşmazlığıyla yeniden bake istemeli. Yeni v7 bake yazılıp yeniden
   okunabilmeli.
5. Kabul sonrası 27B: parçacık kaynak kaybını tek otorite yapıp aynı kayıp kg/J
   değerini gaz fiziksel yan alanına ve `MatterExchangeLedger` olayına yatır;
   bağımsız hücre shader borcunu kaldırarak çift sayımı kapat.

Canlı kabul sonucu:

- Zorunlu adımlı ledger testi PASS: `step=1`, `events=0`, kütle ve enerji
  farkı 0; geçici domain temizlendi.
- Mist regresyonu PASS: 5.184 parçacığın 10. adımında 1.081 mist, canlı fog ve
  `mist_gas_drag`; cleanup hatası yok.
- SimCache yaz/oku regresyonu PASS: disk frame 5'te 5.324 body, unknown 0;
  cleanup hatası yok. Yeni exe SimCache v7 ile yazıp okuyabildi.
- Raporlar: `.tmp/fluid_mist_live.json` ve `.tmp/fluid_labels_cache.json`.

## Vulkan RT: boş sahne artık GPU'da izleniyor — ayrı parti (2026-09-29, paralel ajan)

**✔ DOĞRULANDI (2026-09-29, kullanıcı derledi): boş sahne RT artık maliyetsiz, sorun yok. Shader değişmedi. Yalnız `VulkanBackend.cpp` +
tek satırlık bayrak silmeleri (`VulkanBackend.h`, `VulkanViewportBackend.cpp`,
`VulkanBackend_Raster.cpp`).**

Ölçülen hata (IPC, `perf.list`): boş sahne + Rendered → `loop.viewport_render`
**161 ms/kare**, `viewport.status` samples **0'da takılı**, birikim hiç bitmiyor.
Tek plane ekleyince 2.5 ms, 128 örnekte duruyor. Sebep atmosfer LUT'u DEĞİL
(world=solid'de de aynı): boş sahnede RT kapatılıyor, `presentBackgroundOnly()`
her pikselde `World::evaluate`'i tek iplikte CPU'da çağırıyor ve erken döndüğü
için `m_currentSamples` hiç artmıyordu.

Değişiklik: `createTLAS` sıfır instance ile de TLAS kurar (her ışın miss → GPU
gökyüzü); boş sahne dallarında `m_rtPipelineReady` kapatılmaz;
`presentBackgroundOnly` ve artık yalnızca yazılan `m_hasPresentedRenderedFrame`
söküldü. Boş sahnede geometri/instance descriptor'ları (binding 4/5) yok edilmiş
tampona değil material tamponuna bağlanır; instance fallback'i `range=0`
(spec ihlali) yerine `VK_WHOLE_SIZE`. Yan düzeltme: geçersiz BLAS indeksli
instance atlanınca build range hâlâ İSTENEN sayıyı kullanıp tamponun sonunu
okuyordu; artık yüklenen sayı.

Derlemeden sonra (sırayla):

1. **Boş sahne, Rendered, Vulkan.** `viewport.status` → `samples` artmalı ve
   hedefte `accumulation_complete=true` olmalı; `perf.list` `loop.viewport_render`
   ortalaması plane'li sahneyle aynı düzeyde (birkaç ms). Gökyüzü görünmeli.
   *Bozuksa:* samples 0'da kalıyorsa hâlâ `!isRTReady()||!hasTLAS()` dalı →
   `m_rtPipeline` o anda yok ya da boş TLAS kurulamadı (log: "Cannot bind RT
   descriptors" / "RT pipeline not ready").
2. **Aynı boş sahnede world=sky (Nishita) ve HDRI.** Gökyüzü GPU miss shader'ından
   gelmeli, plane'li sahnedeki gökyüzüyle birebir aynı ton.
   ★ **Sinsi olan:** ekran SİYAH ama samples artıyor ve hızlı → trace çalışıyor,
   miss shader'ın world buffer'ı (binding 7) boş sahnede hiç yüklenmemiş. Hızlı
   ve "yakınsamış" görünür, kimse bug diye raporlamaz. Plane'li sahneyle tonu
   karşılaştır.
3. **Boş → plane ekle → sil → plane ekle.** Çökme / device-lost yok, her geçişte
   birikim sıfırlanıp yeniden yakınsıyor. *Bozuksa:* binding 4/5 hâlâ yok edilmiş
   tamponu gösteriyor.
4. **Boş sahnede hair/groom ekle-kaldır** (`clearHairGeometry` ve
   `uploadHairStrands` boş dalları da değişti). Çökme yok, gökyüzü kalıyor.
5. **Proje aç (dolu → boş → dolu), backend geçişi Solid↔Rendered.** İlk kare
   kısa süre siyah olabilir (artık CPU gökyüzü yok, son kare/siyah gösterilir);
   kalıcı siyah ya da donmuş eski kare OLMAMALI.
6. **Validation layer açıksa:** boş sahnede `VUID-VkDescriptorBufferInfo-range`
   veya null buffer uyarısı çıkmamalı.

## Faz 3 dönüşüm defteri çekirdeği — 26. parti (2026-09-29)

**DERLENDİ ve canlı IPC'de doğrulandı. Shader değişmedi.**

- Yeni `MatterExchangeLedger` her simülasyon adımında gerçekten çalışan faz
  aktarımlarını olay kimliğiyle kaydeder. Birimler sabittir: kg, J ve kg·m/s.
  Özet kaynak/hedef kütlesi ile enerji/gizli-ısı hesabından korunum farklarını
  yeniden türetir; raporlanmış bir “PASS” bayrağına güvenmez.
- İlk üretici mevcut MSF→APIC eriyik aktarımıdır. Başarılı rezervuar borcundan
  sonra kaynak/hedef kütlesi ve momentum birebir kaydedilir. MSF gather zaten
  gizli erime ısısını entalpi bütçesiyle ödediği için `latent_required_j` ile
  `latent_accounted_j` aynı olayda görünür.
- `matter.exchanges` IPC ve `rt.matter.exchanges()` Python aynı `rtapi` raporunu
  okur. `traced=false` runtime olmadığını; `traced=true` + boş liste son adımda
  aktarım olmadığını ifade eder.
- Büyük `ParticleSimulation.cpp` yalnız adım başı defter sıfırlama çağrısını
  aldı; çekirdek ve API ayrı modüllerde. VS proje/filters kayıtları eklendi.
- `rt_test_matter_exchange_ledger_ipc.py` iki kopyada eşit. IPC yetenek denetimi
  635 yöntem için PASS; descriptor ve yetki aynası güncel.

Derlemeden sonra:

1. Uygulama açıkken ayrı terminalde
   `python scripts/test/rt_test_matter_exchange_ledger_ipc.py`.
   Boş sahnede `traced` runtime durumunu dürüstçe vermeli; bütün toplamlar sonlu,
   hata değerleri kaynak/hedef toplamlarından yeniden hesaplanabilir olmalı.
2. Mevcut `phase_06_mass_transfer_scene.py` ile bir eriyik aktarımı çalıştırıp
   aktarımın hemen arkasından `rt.matter.exchanges()` çağır. Tek `melting`
   satırında kaynak/hedef kg eşit, `mass_error_kg` sıfıra yakın ve gizli ısı
   gerekli/ödenmiş değerleri eşit olmalı.
3. Sonraki 27. parti aynı deftere gerçek mist→gaz aktarımını bağlayacak. Gaz
   yoğunluğunun birimsiz render tracer'ı fiziksel kütle sahibi sayılmayacak;
   gaz fazı kg yan alanı ve kg→tracer dönüşümü tek modülde kurulacak.

Canlı kabul sonucu: ilk salt-okunur sorgu `step=0/events=0` ile sözleşmeyi
geçti. Boş domain listesinde `fluid.step` çözücüye girmediği için test geçici
parçacıksız domain denedi; bu da doğal erken dönüş nedeniyle adım üretmedi.
Son sürüm küçük geçici sıvı tohumu kurdu: `step=1`, `events=0`, kütle ve enerji
hatası 0; test PASS ve geçici domain kaldırıldı. İki script aynası güncel.

27. parti öncesi kod okuması önemli bir tuzağı doğruladı: mevcut GPU yüzey
yanması gazı **hücre başına** `surface_state` birimiyle beslerken APIC tarafında
kütleyi **parçacık başına** `mass_fraction` ile azaltıyor. Bunlar aynı olayın
iki bağımsız hesabı; misti ayrıca gaza yatırmak çift sayım yaratabilir. Bu yüzden
27 önce shader/hedef yatırımı ile parçacık kaynak kaybını fiziksel kg cinsinden
aynı olayda eşleştirecek; sonra kalan mist faz aktarımı yapılacak.

## Ortak fiziksel madde tablosu K1 — 25. parti (2026-09-29)

**DERLENDİ ve canlı IPC'de doğrulandı. Shader değişmedi.**

- Mevcut `SubstanceProfile` tek fiziksel otorite olarak genişletildi: sıvı
  yoğunluğu ve kinematik viskozitesi, flash/autoignition Kelvin, buharlaşma
  gizli ısısı, buharlaşma/soğutma/oksijen/flame katsayıları ile granüler
  sürtünme ve kohezyon aynı kayıtta.
- Water, Gasoline, Alcohol ve Oil kanonik madde kayıtları eklendi. APIC'in
  Water/Gasoline/Alcohol/Oil/Plastic/Wax kimya preset'leri artık bu tablodan
  türetiliyor. Kelvin dönüşümü aktif `WorldThermalState` ölçeğini kullanıyor;
  varsayılan ölçekte eski normalize değerler korunuyor.
- `MoltenMassTransfer` içindeki Plastic/Wax/Iron/Steel/Ice ad kontrolleri
  kaldırıldı. Eriyik viskozitesi ve kimyası doğrudan aynı profile bakıyor.
- `msf.substance {name}` ve `rt.msf.substance(name)` fiziksel değerleri aynı
  API çekirdeğinden okuyor. Bilinmeyen ad sorguda hata; eski sahne yükleme
  yolu olan `findSubstance` geriye uyumluluk için Wood varsayımını koruyor.
- Yeni `RtApiSubstance.cpp` büyük API dosyasına iş mantığı eklemeden ortak
  sorguyu sağlıyor. VS proje/filters kaydı yapıldı. IPC descriptor üreticisi
  çalıştırıldı; yetenek denetimi 634 yöntem için PASS.
- `rt_test_substance_profiles_ipc.py` iki kopyada eşit; Python sözdizimi ve
  descriptor JSON denetimi PASS.

Derlemeden sonra uygulama açıkken ayrı terminalde:

1. `python scripts/test/rt_test_substance_profiles_ipc.py`
   Beklenen: `PASS`, benzersiz madde adları, fiziksel alanlar sonlu, Water
   extinguishing, beş yakıt flammable, eski eriyik viskoziteleri aynı ve
   bilinmeyen madde reddedilmiş.
2. Mevcut mist kabulünü kısa regresyon olarak çalıştır:
   `python scripts/test/rt_test_fluid_mist_ipc.py`. Benzin kimyası artık ortak
   tablodan geldiği için mist üretimi, fog ve `mist_gas_drag` yine PASS olmalı.
3. K1 kabulünden sonra sıradaki iş Faz 3 dönüşüm defteri: liquid→gas/mist
   kütlesini ve enerjisini kaynak/hedef tarafında aynı işlem kimliğiyle ölçmek.

İlk canlı koşu sonucu: listedeki bütün maddeler ve fiziksel değer kontrolleri
geçti; test yalnız bilinmeyen madde dalında durdu. Kök fizik tablosu değil,
handler'ın IPC'nin beklediği `{"__error": ...}` yerine `{"ok": false, ...}`
döndürmesiydi. Kaynakta `__error` zarfına düzeltildi. Aynı ilk derlemede mist regresyonu PASS: 5.184 parçacığın
10. adımında 1.081 mist, canlı fog ve `mist_gas_drag`; cleanup hatası yok.

İkinci derleme canlı sonucu: `rt_test_substance_profiles_ipc.py` PASS;
**15 kanonik madde** eksiksiz ve sorgulanabilir, bilinmeyen ad IPC hatasıyla
reddediliyor. K1 kapandı. Sıradaki çekirdek iş Faz 3 dönüşüm defteri ve
liquid→gas/mist kütle-enerji korunumu.

## Gerçek düşük kütleli mist — 24. parti (2026-09-29)

**DERLENDİ ve canlı IPC'de doğrulandı.**

- `mass_fraction` değeri `0 < m <= 0,15` olan sıvı parseli CPU ve Vulkan
  sınıflandırıcılarında `mist`; frozen öncelikli, granular/solid hariç.
- Mist liquid komşuluk binine katılmaz. Fog seçimi yoğunluğu sabit bir tam
  parsel gibi değil, kalan kütlesiyle splat eder.
- Yeni `FluidMistCoupling` mist hızını örtüşen gaz MAC alanına kütle ölçekli
  tepkiyle yaklaştırır. Kütle sahibi APIC parselidir; bu adım kütle üretmez veya
  silmez. Mevcut yüzey buharlaşmasının gaza verdiği kütle ayrı kalır.
- IPC/Python: `mist_generation=true`, `mist_mass_fraction_max=0.15`,
  `classifier=mass+neighborhood_v2`.
- GPU shader ABI: clear/scatter/classify push constant 80 bayt; scatter 5,
  classify 6 buffer. Label çağrısı mass buffer'ını aynı karede yükler.

Canlı kabul sonuçları, boş sahnede:

1. CPU tam label regresyonu PASS; dense/body, isolated/spray, frozen önceliği,
   görünümden bağımsızlık, clear ve cleanup.
2. Vulkan tam label regresyonu PASS; yeni mass buffer + 80 bayt shader ABI ile
   `on_gpu=true`. 5.324 parselde sıcak adımlar yaklaşık 0,42–0,81 ms.
3. 32.000 yoğun parsel taşma testi PASS: `on_gpu=false`, CPU fallback,
   32.000 body / unknown 0.
4. Yeni `rt_test_fluid_mist_ipc.py` PASS: sıcak/hareketli gazla örtüşen 5.184
   yanıcı sıvı parselinden 10. adımda **1.081 mist**; fog görünümü canlı ve
   `mist_gas_drag` gerçek çözücü izinde. Rapor `.tmp/fluid_mist_live.json`.
5. SDF/splat/fog rota regresyonu PASS: 6.120 body SDF, 2 spray splat; mist
   varsayılanı fog, hidden/reset/geçersiz rota reddi doğru.
6. İlk koşuda yalnız testteki `0.15` tam float eşitliği kırıldı
   (`0.15000000596`); fizik sonucu değildi. Test `abs_tol=1e-6` ile düzeltildi,
   iki script aynası eşit.
7. Bütün geçici domain ve flow source kayıtları temizlendi; sahne yeniden boş.

Sıradaki çekirdek iş Faz K ortak madde özellik tablosudur. Faz 3
dönüşüm defteri, buharlaşan/mist kütlesinin gaz alanına girişini ayrıca ölçüp
kütle ve enerjiyi iki tarafta eşleştirecek.

## GPU parçacık etiketleme prototipi — 23. parti (2026-09-29)

**DERLENDİ ve canlı IPC'de doğrulandı. Üç compute shader eklendi.** Performans
kapısı geçti; GPU yolu tutuluyor.

- `sim_fluid_label_bin_clear/scatter/classify`: 1,5 voksel yarıçaplı yoğun
  bin, tamsayı atomik üyelik ve parçacık başına komşu sayımı. Body/spray
  histerezisi, frozen önceliği ve flags bit koruması CPU sözleşmesiyle aynı.
- Sonuç aynı karede host `flags` dizisine iner; API, IPC, UI, cache ve görünüm
  çözücüsü tek otoriteyi okumaya devam eder. `last_step.on_gpu` IPC/Python ve
  panelde hangi yolun gerçekten çalıştığını gösterir.
- Bin başına 64 aday sınırı var. Tek bir taşma bile GPU sonucunu reddeder ve
  değiştirilmemiş host flags üzerinden tam CPU sınıflandırıcısı çalışır.
  Granüler domain, solid-substance tag veya Vulkan dışı backend de CPU'ya döner.
- Buffer boyutu cihazın `max_storage_buffer_bytes` sınırını aşarsa güvenli CPU
  dönüşü var. Domain buffer yaşam döngüsü yeni dört buffer'ı da serbest bırakır.
- Beklenen risk: güncel CPU yolu zaten 4,17–4,99 ms. Pozisyon+flags upload ve
  aynı-kare flags readback'i yeni fence maliyeti yaratabilir; shader hızlı olsa
  bile toplam adım yavaşlayabilir. Bu nedenle sonuç ölçülmeden “optimizasyon”
  sayılmayacak.

Canlı kabul sonuçları:
1. Vulkan 100.000: `on_gpu=true`, dört karede 100.000 body / unknown 0.
   Süreler **4,164 / 1,969 / 1,797 / 2,336 ms**; sıcak-kare ortalaması
   **2,034 ms**. Önceki derlemede aynı Vulkan benchmarkının sıcak ortalaması
   4,304 ms idi: uçtan uca label süresinde yaklaşık **2,1× hızlanma**.
2. CPU backend karşılaştırması: `on_gpu=false`, 100.000 body / unknown 0;
   9,454 / 8,155 / 7,383 / 7,720 ms. Solver backend'i de değiştiği için bu
   3,8× oran doğrudan aynı-fizik A/B'si değil; kabul hesabı önceki Vulkan
   exe tabanına göre yapıldı.
3. İşlev regresyonu PASS: dense body, isolated spray, frozen 968,
   thermal-off sonrası frozen 0, display particles/fog/surface bağımsızlığı,
   clear ve cleanup. GPU yolu raporlandı.
4. Görünüm regresyonu PASS: 6.120 body SDF, 2 spray splat; spray hidden,
   reset ve geçersiz rota reddetmeleri doğru.
5. Yeni `rt_test_fluid_label_gpu_fallback_ipc.py`: 32.000 yoğun parçacıkta
   64'lük bin kapasitesi bilerek aşıldı; `on_gpu=false` CPU fallback,
   32.000 body / unknown 0, frame ve cleanup PASS.
6. Raporlar: `.tmp/fluid_labels_benchmark_vulkan.json`,
   `.tmp/fluid_labels_benchmark_cpu.json`, `.tmp/fluid_labels_live.json`,
   `.tmp/fluid_label_views.json`, `.tmp/fluid_label_gpu_fallback.json`.
7. Test sonunda frame 0 ve `fluid.list_domains=[]`.

Statik kontroller de PASS: Python AST, VS proje/filters XML kayıtları, shader
dosya/kayıt eşleşmesi ve dört test-script aynası. Sıradaki ana iş gerçek mist.

## Çözücü determinizmi ve etiket maliyeti — 22. parti (2026-09-29)

**Canlı tanı tamamlandı; ürün kodu/shader değişmedi.** Determinizm probuna
`--backend cpu|vulkan` eklendi ve iki script kopyası eşitlendi.

- Aynı 27.300 parçacıklı dam-break, 40 kare, üç koşu:
  - CPU: A==B bit bit aynı; whitewater açık C de ana sıvıda A ile bit bit aynı.
    A–B ve A–C centroid farkı 0. Whitewater 46.555 parçacıkla gerçekten çalıştı.
  - Vulkan: A≠B; centroid gürültü tabanı **0,00892 m**. Whitewater açık C'nin
    A'dan farkı **0,00610 m**, doğal A–B farkının altında; ana sıvıya whitewater
    sızıntısı görülmedi. Whitewater 27.884 parçacıkla çalıştı.
  - Raporlar: `.tmp/whitewater_determinism_cpu.json` ve
    `.tmp/whitewater_determinism_vulkan.json`.
- Kaynak ayrıldı: `sim_fluid_p2g_scatter.comp` aynı MAC yüzüne paralel
  `atomicAdd(float)` yapıyor. Toplama sırası Vulkan'da tanımlı değil ve float
  toplama birleşmeli olmadığından sonuç koşudan koşuya ayrışıyor. CPU'nun aynı
  sahnede bit eşit olması, ortak çözücü/seed/reseed yarışını eliyor.
- Bu bir veri yarışı veya whitewater sızıntısı olarak değerlendirilmedi.
  Bit tekrarlanabilir çıktı gerektiğinde CPU referans yolu kullanılmalı.
  Vulkan'ı bit deterministik yapmak sıralı gather/sort veya doğrulanmış
  fixed-point P2G ister; mevcut shader notuna göre eski fixed-point denemesi
  ölçek yüzünden akışı durdurmuş. Bu mimari değişiklik tek-domain akışını
  bloke etmiyor; şimdilik ölçülmüş toleranslı GPU yolu olarak tutuluyor.

**Güncel etiket tabanı, 100.000 parçacık / Vulkan sim / dört kare:**

| Kare | Toplam (ms) | bin (ms) | classify (ms) |
|---|---:|---:|---:|
| 1 | 4,9874 | 3,1628 | 1,8246 |
| 2 | 4,4950 | 2,4041 | 2,0909 |
| 3 | 4,1727 | 2,4663 | 1,7064 |
| 4 | 4,2447 | 2,5137 | 1,7310 |

- Eski 7,8–10,4 ms ölçümü artık güncel değil. Tüm karelerde 100.000 body,
  unknown 0; geçici domain kaldırıldı ve timeline kare 0'a döndü.
- Rapor: `.tmp/fluid_labels_optimized_benchmark.json`.
- GPU etiketleme hâlâ olası, fakat aynı karede host `flags` isteyen mevcut
  sözleşme yeni readback/fence doğurur. Bu ölçümde bütün sim adımı zaten üç
  synchronize çağrısında 6,42 ms harcadı. Port ancak mevcut fence'e eklenerek
  veya etiket tüketicileri cihazda tutulup readback kaldırılarak yapılmalı;
  salt shader portu otomatik hız kazancı sayılmayacak.
- Sıradaki geliştirme kapısı: GPU prototipi end-to-end adımı 4 ms'den anlamlı
  düşürüyorsa alınır; düşürmüyorsa CPU yolu korunup gerçek mist üretimine geçilir.

## Kapalı tankta parçacık kaybı — 21. parti (2026-09-28)

**DERLENDİ ve canlı IPC'de doğrulandı (2026-09-29). Shader değişmedi.**

- Eski exe ile Vulkan IPC probu: 10.816 → 10.369 parçacık, 80 adım.
  İlk kayıp adım 39: reseed −202/+12, net −190. Toplam −880/+433,
  net −447. 80/80 adımın sayı farkı reseed sayaçlarıyla eşleşti.
  Kanıt: `.tmp/closed_tank_conservation_before.json`.
- Bu koşuda timeline da ilerledi; mevcut sahne/frame korunumu FAIL.
  Geçici domain kaldırıldı, önceden etkin domainler tekrar etkinleştirildi.
  Dolayısıyla bu koşu kontrollü son kabul testi değildir. Prob artık boş
  sahne ve sabit timeline karesi ister; mevcut domainleri kapatıp çalışmaz.
- Kök: `redistributeParticles` fazlalıkları önce siliyor, yalnız uygun
  hücrelere geri ekliyordu; kullanılmayan bütçe kayboluyordu. Eski test yalnız
  büyümeyi yasaklıyor, kaybı PASS sayıyordu.
- Yeni `FluidParticleRedistribution` modülü: silme/emit/compact yok. Seyrek
  iç hücre yalnız yüz komşusunun fazlalığından, aynı substance tag ile mevcut
  parçacık alır. Uygun yer yoksa fazlalık kalır. Hız, affine, iki UVW nesli,
  sıcaklık, kütle oranı ve tüm yan diziler değişmez. Granüler/frozen/solid
  madde parçacıkları ve yüzey hücreleri taşıma dışında.
- Bu yerel örnek dağılımı düzeltmesidir; basınç çözümünün yerini almaz.
  Taşınan parçacığın konumu değiştiği için açısal momentum veya yüzey
  hacminin birebir korunması iddia edilmiyor. Yoğun hücreler uygun alıcı
  yokken max_per üstünde kalabilir. `reseed_added/removed` artık 0 beklenir.
- CPU ve Vulkan son-adveksiyon yolu aynı çekirdeği çağırır; mevcut UI,
  Python ve IPC işlemleri bu ortak yola ulaşır. Yeni API eklenmedi.

Canlı kabul sonucu:
1. Vulkan, 120 adım: **10.816 → 10.816**, tüm adımlarda sabit;
   `reseed_added/removed` 0/0, timeline kare 0'da kaldı, temizlik hatası yok.
   Rapor: `.tmp/closed_tank_conservation_vulkan_after.json`.
2. CPU, 120 adım: **10.816 → 10.816**, tüm adımlarda sabit;
   `reseed_added/removed` 0/0, timeline kare 0'da kaldı, temizlik hatası yok.
   Rapor: `.tmp/closed_tank_conservation_cpu_after.json`.
3. Test sonunda `fluid.list_domains=[]`, frame 0: geçici domain kalmadı.
4. Assertions açıkken `scripts/test/fluid_particle_redistribution_test.cpp`
   ile `source/src/Physics/Fluid/FluidParticleRedistribution.cpp` ve
   `source/src/Math/Vec3.cpp` dosyalarını,
   `source/include` include yolu ile derle/çalıştır. Sayı, yan dizi kimliği,
   aynı seed sonucu, alıcısız havuz, farklı madde, frozen/granüler/solid,
   düşük max_particles ve geçersiz konum vakaları var. Ayrı standalone
   executable üretilmedi; uygulama derlemesi yeni çekirdeği başarıyla bağladı.
5. Kapalı tankta görsel hacim/dalga davranışını kontrol et; parçacık sayısı
   korunumu tek başına hacim veya basınç doğruluğunu kanıtlamaz.

Statik kontroller: Python AST, VS XML kayıtları ve test scriptlerinin
`scripts/test` / `x64/Release/scripts/test` eşitliği PASS. C++/shader build
PASS. RT bloklaşması ertelenmiş durumda. Sıradaki çekirdek işi çözücünün
run-to-run determinizm ayrımıydı; 22. partide tamamlandı.

## Güncel öncelik (2026-09-29)

- Kullanıcı RT bloklaşmasını erteledi; çoğunlukla SDF su materyalinde
  gördüğünü belirtti. Bu araştırmada shader kodu değiştirilmedi.
- Koddan aday: `FluidLevelSet.cpp` her hücrede baskın materyali A, ikincisini
  B olarak saklıyor. `volume_closesthit.rchit::sampleComposition` en yakın
  hücrenin A/B kimliğini alıp sekiz köşenin ham B ağırlığını interpolasyonla
  birleştiriyor. Komşularda A/B tersse aynı fiziksel materyalin ağırlığı
  süreksizleşebilir. Kimlik bazlı ağırlık toplama düzeltmesi uygulanmadı;
  gözlenen lekelerin nedeni olduğu canlı testle kanıtlanmadı.
- Kapalı tank kaybı 21. partide kapandı; CPU ve Vulkan 120 adım korunum geçti.
- Determinizm 22. partide ayrıldı: CPU bit aynı, Vulkan float-atomik P2G
  toleranslı. Whitewater ana sıvıyı ölçülen gürültü tabanının üstünde etkilemedi.
- Ana sıra: GPU etiketleme performans kapısı, gerçek mist, ortak madde
  özellikleri, dönüşüm defteri ve domain birleşmesi.


> **Durum:** CANLI — her partide üzerine yazılır. Önceki sürüm (particle
> authoring + SSS) git geçmişinde: `git show 220bed8:docs/dev/NEXT_BUILD_CHECKS.md`.
>
> En üstteki bölümler güncel devir sırasıdır; altındakiler önceki partilerin
> doğrulama geçmişidir.

## Faz 2-W / W1: whitewater'ı da label routes çiziyor, FoamRenderMode söküldü (2026-09-28, 20. parti)

Karar (kullanıcı, 2026-09-28): hedef tam akışkan fiziği; whitewater çözülemeyen
ölçeğin kütlesiz yer tutucusu. Birleşme yalnız görünümde, depo birleşmesi (W2)
iptal. Plan: `BIRLESIK_MADDE_DOMAIN_TASARIMI.md` §8c.

Ne değişti:
- Particle State Views'taki spray/foam/bubble satırları artık whitewater'ı da
  yönlendiriyor (splat = küre, sdf = yüzey hacminde beyaz ortam, fog = fog
  hacmine eklenir, hidden). Whitewater panelindeki "Foam Render" combo'su yok.
- Splat havuzu yalnız splat'e yönlenen whitewater'ı tutuyor. Eskiden 100k köpük
  "Volume" iken grup silinirdi; şimdi yönlendirilmeyen tip havuza hiç girmez.
- Ölü metaball köpük yüzeyi söküldü: `FoamSurface.cpp/.h` (vcxproj'dan da),
  `surface_*` alanları. Derleme bunu ilk yakalar.
- ProjectManager whitewater görünümünü (`volume_color/opacity/…_strength`)
  artık kaydediyor.
- ★ **Görünür değişiklik:** varsayılan tablo spray/foam/bubble → splat. Eskiden
  whitewater varsayılanı "Volume" idi, yani eski sahnelerde whitewater artık
  küre olarak çizilir. Eski görünüm için foam ve bubble → Isosurface.

✔ **Derlendi, IPC testleri geçti (2026-09-28).** İlk derlemede `RtApi.h`
`FoamParams`'ı görmüyordu (W0 eksiği): ileri bildirim + RtIpc/RtPython'a
`FluidFoam.h`. Sonra:
- `rt_test_whitewater_label_routes_ipc.py` 11/11 PASS. 76.228 whitewater
  (spray 5.653 / foam 20.931 / bubble 49.644): defaults hepsi splat; hepsi
  hidden; karışıkta sdf 20.931 = foam, fog 49.644 = bubble, hidden 5.653 =
  spray (simülasyonun kendi sayımıyla birebir); fog hacmi yayınlandı
  (vdb_id 2); render_mode reddedildi.
- İlk turda göz kontrolü (3) ve rebuild regresyonu (4) açık kaldı.

**Yeniden başlatma sonrası devam testi (2026-09-28):**
- Dış IPC, geçici dam break, Vulkan RT, kare 0–40: yönlendirme **11/11 PASS**.
  76.999 whitewater: spray 6.080 / foam 21.711 / bubble 49.208.
  Karışık görünümde sdf=21.711, fog=49.208, hidden=6.080, splat=0;
  fog hacmi yayınlandı. Canlı içerik sıfır olmadığı için anlamlı test.
- **Madde 4 PASS:** ayrı sabit-kapasite turunda RAM cache kare 0/8 üzerinden
  0 → 4.326 → 0 → 4.326 whitewater. `accel.vulkan_rt.rebuild` **12'de**,
  `accel.vulkan_rt.update_geometry` **11'de** sabit. İlk havuz kurulumu test
  penceresinden önce. 40 karelik büyüme turundaki kapasite artışlarıyla
  boş/dolu geçiş regresyonu birbirine karıştırılmadı.
- **Madde 3 kısmi görsel:** tek RT capture alındı; beyaz yüzey katkıları ve
  hacim görünür. `render.probe`: available=true, nan_fraction=0,
  black_fraction=0. Bu, Wave Tank'taki kullanıcı panel/göz kontrolünü ve
  bloklaşma teşhisini tamamlamaz; RT bloklaşması sıradaki ana iştir.
- Bu iki turda TDR olmadı; aralıklı TDR'nin çözüldüğü anlamına gelmez.
- Geçici domain'ler temizlendi; kamera, shading/capture ve kare geri yüklendi.
  Başlangıçta domain yoktu; mevcut küp değiştirilmedi.
- Kanıt: `.tmp/whitewater_label_routes_full.json`,
  `.tmp/whitewater_rt_regression.json`, `.tmp/whitewater_rt_zero_crossing.json`,
  `.tmp/whitewater_mixed_rt.jpg`. Son küçük turun sayım sonucu:
  `.tmp/whitewater_label_routes.json` (11/11 PASS).

**Açık bulgu (kullanıcı, 2026-09-28): RT'ye geçiş düğmesinde aralıklı TDR.**
Kullanıcıya göre W1'den önce de ara ara oluyordu; bu partiyle ilgisi yok.
- Log: kayıp, "SWITCHING to Vulkan RT" satırından ÖNCE, bir
  `endSingleTimeCommands/vkQueueSubmit` gönderiminde gözlendi. Hemen önce
  MaterialPreview iki hacim (yüzey + fog) çiziyordu. Kurtarma üç kez
  `vkCreateDevice -3` hatasıyla durdu; uygulamanın yeniden başlaması gerekti.
- Log'da hangi VkDevice'ın (viewport mu, render backend mi) ve hangi işin
  asıldığı YOK. `reportVulkanDeviceFailure` yalnız işlem adını yazıyor.
  Sıradaki adım ölçü aleti: cihaz rolü + son başarılı fence'ten beri geçen
  süre + son gönderilen işin etiketi; daha iyisi `VK_EXT_device_fault`.

Kontroller (W0'ın 19. parti adımlarıyla birlikte; derleme tek):
1. **Derleme.** Hata çıkarsa ilk bakılacak yer: kaldırılan `render_mode` /
   `FoamSurface` referansı kalmış mı.
2. **`rt_test_whitewater_label_routes_ipc.py`, ben koşarım.**
   - defaults: tüm whitewater splat'te, `fluid.get views[splat].whitewater` = alive.
   - hepsi hidden: hidden = alive, hiçbir görünüm whitewater raporlamıyor.
   - karışık (spray hidden / foam sdf / bubble fog): sayılar alive'a eşit
     toplanıyor, splat 0; bubble varsa fog hacmi yayınlanmış (vdb_id ≥ 0).
   - `set_whitewater render_mode` reddediliyor.
   - ★ Sinsi olan: alive = 0 ise bütün eşitlikler boşuna tutar. Test bunu
     FAIL eder.
3. **Göz kontrolü (sen):** Wave Tank, whitewater açık.
   - Panel: whitewater bölümünde "Spray: splat, foam: splat, bubbles: splat"
     satırı; Now drawing'de "Splat: … (N particles + M whitewater)".
   - Particle State Views'ta foam → Isosurface: köpük yüzeyde beyaz ortam
     olarak görünmeli (eski Volume görünümü), küreler kaybolmalı.
   - bubble → Fog: fog hacmi açılmalı ve kabarcıklar fog olarak görünmeli.
   - Bozuksa: köpük iki kez çiziliyorsa (hem küre hem ortam) bir tüketici
     tabloyu okumuyor demektir.
4. **RT rebuild regresyonu:** splat'e yönlü whitewater ile oynatmada
   `accel.vulkan_rt.rebuild` foam 0↔>0 geçişlerinde artmamalı. Grup ROUTE'a
   bağlı, canlı içeriğe değil (18. partinin dersi).

## Faz 2-W / W0: whitewater ve parçacık bütçesi IPC'de + determinizm probu (2026-09-28, 19. parti)

Plan: `BIRLESIK_MADDE_DOMAIN_TASARIMI.md` §8c. W0 fizik değiştirmez; yalnızca
kural 1 ihlallerini kapatır ve W2'nin kapısını ölçülebilir yapar.

Ne yazıldı:
- **`fluid.get_whitewater` / `fluid.set_whitewater`:** FoamParams'ın canlı
  alanları ve canlı sayaçlar (alive/spray/foam/bubble, gen/advect ms, GPU
  bayrakları). Ölü metaball `surface_*` alanları dışarıda. Değer bütün olarak
  doğrulanıyor. Tip hatası istekte düşer, asla UI kuyruğunda değil.
  - Üretim/dinamik değişirse RAM cache düşer.
  - Görünüm alanı değişirse yalnızca repaint olur.
- **`fluid.set_param max_particles`:** Parçacık bütçesi, [1000, 10M].
  `fluid.get` artık `max_particles` döndürüyor.
- **`fluid.state_digest`:** Canlı parçacıkların bit bit FNV hash'i
  (konum/hız), ağırlık merkezi, ortalama hız, whitewater sayısı ve hash'i.
- Python karşılıkları, overlay; descriptor'lar 633 metot, audit OK.
- **Yeni prob `rt_probe_whitewater_determinism_ipc.py`:**
  - A: whitewater kapalı.
  - B: tekrar kapalı; çözücü koşudan koşuya deterministik mi?
  - C: açık; birincil parçacıklar değişmiyor mu?

Derleme: yalnız C++, shader yok.

1. **Derleme.**
2. **Prob, ben koşarım.** Olası sonuçlar:
   - **A==B:** W2 kapısı bit bit eşitlik olur.
   - **A≠B:** Çözücü zaten deterministik değil (ör. float atomik P2G). Kapı,
     A–B ağırlık merkezi gürültü tabanına göre konur.
   - **C≠A ve A==B:** Whitewater BUGÜN birincil sıvıyı etkiliyor demektir.
     Birleşmeden önce bulunması gereken bir sızıntı.
   - ★ Sinsi olan: C'de whitewater sayısı 0. O zaman "etkilemiyor" yalnızca
     "hiç çalışmadı" demektir. Prob bunu FAIL eder.
3. **Whitewater IPC turu:** `set_whitewater` ile geçersiz aralık (ta_min ≥
   ta_max) reddedilmeli ve hiçbir alan değişmemeli.

✔ **Sonuçlar (2026-09-28, dam break 27.300 parçacık, 40 kare, Vulkan):**
- **A≠B: çözücü koşudan koşuya bit bit deterministik DEĞİL.** Aynı sahne,
  aynı kare, farklı hash; ağırlık merkezi farkı 0,0179 m (gürültü tabanı).
  Muhtemel kaynak float atomik P2G / paralel sıra; ayrıca aranmadı.
- **A–C farkı 0,0128 m, gürültü tabanının ALTINDA** (C'de 22.864 whitewater
  çalıştı). Whitewater'ın birincil suyu etkilediğine dair iz yok. Bu ölçümle
  kanıtlanamaz da; bit eşitliği olmadan ancak "taban içinde" denebilir.
- Parçacık sayısı üç koşuda da 27.300: whitewater bütçeyi yemiyor.
- W2 iptal olduğu için kapı artık gerekmiyor. Determinizm eksikliği ayrı bir
  bulgu: tekrar üretilebilir sim/cache karşılaştırmaları için önemli.
- IPC turu: geçersiz aralık, eşik sırası, tip hatası, negatif max_foam,
  render_mode reddedildi; alanlar değişmedi; geçerli düzenleme uygulandı;
  `max_particles 50` → 1000'e sıkıştırıldı.

**Açık bulgu (kullanıcı, 2026-09-28): RT'de SDF yüzeyinde blok şeklinde lekeler.**
Wave Tank, kare 113'te ekran görüntüsü alındı. Yüzeyde eksene hizalı, keskin
kenarlı, açık gri dikdörtgenler var; arka plan rengine yakınlar.
- Elenenler (kod okuması): yüzey köpüğü (`fluid_surface_foam` = 0) ve iso
  yürüyüşünde blok atlama (yok).
- Sıradaki ayrım: lekeler yüzeyde DELİK mi (ışın arkaya geçiyor), yoksa
  parlak yüzey mi? RayFusion aynı karede gösteriyor mu? `render.probe` ya da
  tek bir IOR/absorpsiyon A/B'si ayırır.
- Ekran görüntüsü:
  `scratchpad/rt_blocky.png` (oturum geçici dizini).

## Splat havuzu spray 0↔>0 geçişinde RT rebuild ediyordu (2026-09-28, 18. parti)

✔ **Derlendi ve ölçüldü.**
- Aynı dam break, Vulkan RT (`rendered`), kare 20–60: spray 0↔>0 arasında
  birçok kez geçti (23, 24, 26, 51, 53, 60…). `accel.vulkan_rt.rebuild` ve
  `update_geometry` **0**; önce aynı aralıkta 6 idi.
  Rapor: `.tmp/splat_rebuild_churn_after.json`.
- **Pozitif kontrol:** `render_mode` particles→surface geçişi sayaçları
  1→2 artırdı. Sayaç ölüyken sıfır okunmuş değil.
- 40. karede splat görünümü canlı: 69 spray. `rt_test_fluid_label_views_ipc.py`
  yeniden PASS.
- Kalan: kullanıcının RT'de gözle takılma kontrolü.

17. parti ✔:
- `rt_test_fluid_label_views_ipc.py` PASS: 6.120 body sdf'de, 2 spray splat'ta.
  Hidden, reset ve reddetme kontrolleri de geçti.
- Etiket probu ve cache testi regresyonu geçti.
- Kullanıcı gözle doğruladı: spray küre olarak çiziliyor.

**Kullanıcı bulgusu:** Spray splat olunca, az parçacıkta bile RT modunda
(cache'ten oynatmada da) bazı kareler çok pahalı. RayFusion ve raster'da sorun yok.

**Ölçüldü** (dam break, Vulkan RT, kare 20–60, `perf.list` artışları):
- `accel.vulkan_rt.rebuild` + `update_geometry` **yalnızca** spray 0↔>0
  geçişinde tetiklendi: 23, 24, 26, 53, 54 ve 58. kareler.
- Spray 2–71 arasında değişirken (27–52) hiç tetiklenmedi.
- Rapor: `.tmp/splat_rebuild_churn.json`.

**Kök (benim 17. partideki kararım):** Etiket yönlendirmeleri kaynağı yalnızca
canlı anahtar varken işaretliyordu. Son damla gövdeye katılınca `plan.splat`
false oldu, splat havuzu silindi (yapısal değişiklik = tam RT rebuild), ilk
yeni damlada yeniden kuruldu. Fog yönlendirmesinde daha kötüsü olurdu:
`plan.fog` false olunca fog slotu SİLİNİYOR. Bu, slot kimliği churn'ü, yani
siyah bant sınıfı.

**Düzeltme:**
- **Çözücü:** Etiket yönlendirmesinin hedef görünümü artık kalıcı kaynak.
  Unknown hariç; unknown her zaman maddeyi izler.
- **Splat köprüsü:** Splat görünümünde canlı parçacık yoksa mevcut havuz
  korunur ve gizlenir (refit), parçacık taraması yapılmaz. Hiç kurulmamış havuz
  ilk damlaya kadar kurulmaz.
- **Fog:** Fog'da canlı parçacık yoksa slot gizlenir, yoğunluk geçişi atlanır.
- **Panel "Now drawing":** Yönlendirilen etiketleri tablodan adlandırıyor.

Yan etki: `fluid.get` artık her sıvıda splat ve fog görünümünü "allocated"
olarak raporlar (`live false`, `particles 0`). Doğru; kaynak kalıcı.

1. **Derleme** (yalnız C++).
2. **Aynı ölçüm, ben koşarım.** "Label Splash Test" sahnesinde RT modu, kare
   20–60. Görmen gereken: ilk damlada **tek** yapısal olay (havuz kurulumu),
   sonraki 0↔>0 geçişlerinde rebuild **yok**.
3. **`rt_test_fluid_label_views_ipc.py`** yeniden. Hâlâ PASS olmalı.
4. **Senin için, gözle:** RT'de sıçrama boyunca takılma olmamalı.

**Açık, kullanıcıya ait:** Splat materyali bağlanabiliyor (Output → Splat
Material combo, `fluid.set_splat_material`) ama DÜZENLENEMİYOR: nesneye
atanmamış bir BSDF Material Properties'te açılamıyor. Bu, ertelenen Faz 1-UI
2. partisinin (`SelectableType::Material`) kapsamı.

## Faz 2: görünüm anahtarı (madde, durum etiketi) — spray splat olur (2026-09-28, 17. parti)

Kullanıcı kararı: tasarım tablosu varsayılan olarak açık. Spray, foam ve bubble
→ splat; mist → fog; body, frozen ve unknown → maddeyi izler (unknown
değiştirilemez). **SDF ve fog modundaki mevcut sahnelerde kopan damlalar artık
küre olarak çizilir.** Bu bilinçli bir görüntü değişikliği.

Ne değişti:
- **Çözücü** (`FluidViewResolver`): `viewFor(tag, label)`, `viewForParticle`,
  `FluidViewSelection`, `countParticlesPerView`, `distinctViewKeys`.
  `liveTagsIn` / `liveTagsNotIn` / `distinctSubstanceTags` söküldü.
- **Gather'lar:** level set, materyal koordinatı ve kompozisyon (`FluidLevelSet`),
  fog yoğunluğu ve sıcaklığı dışlama listesi yerine seçim nesnesi alıyor. Splat
  köprüsü parçacık başına karar veriyor.
- **Yüzey rebuild imzası** etiket bitlerini ve tabloyu içeriyor.
- **Domain alanı** `fluid_label_routes` hem proje hem sahne serileştiricisinde,
  etiket adı → yönlendirme adı sözlüğü olarak.
- **IPC/Python:** `fluid.set_label_views`. `fluid.get` artık `label_routes`,
  `hidden_particles`, `views[].labels` ve `views[].particles` döndürüyor.
  Etiket raporunda `render_routing` = `"substance+label"`. Descriptor'lar
  yeniden üretildi (630), audit OK.
- **Panel:** Output → Liquid Display → "Particle State Views" (6 combo +
  Reset). "Now drawing" satırları etiketleri ve parçacık sayılarını gösteriyor.

Derleme: yalnız C++, shader yok, yeni .cpp yok.

1. **Derleme.** En olası hata yeri include zinciri: `ParticleSimulation.h` →
   `FluidViewResolver.h` → `FluidParticleLabels.h`.
2. **`python scripts/test/rt_test_fluid_label_views_ipc.py`.** Ben koşarım.
   Görmen gereken: PASS. Varsayılan tabloda sdf parçacıkları = body + frozen,
   splat parçacıkları = spray. spray→hidden spray'i her görünümden çıkarır.
   Reset tabloyu geri getirir. Geçersiz adlar reddedilir.
3. **`python scripts/test/rt_test_fluid_labels_ipc.py <domain>`** (yeni
   `render_routing`) ve 16. partinin cache testi regresyon için.
4. **Görsel, açık bir SDF su sahnesi:** sıçrayan tek damlalar küre olarak
   görünmeli, gövde yüzey kalmalı. ★ Sinsi olan: sayım doğru ama damlalar
   ekranda yok. Bu, splat havuzu kuruluyor ama malzeme ya da boyut yanlış
   demektir (spray, maddenin materyaliyle splat olur).
5. **Maliyet notu:** her görünüm kararı parçacık başına bir tablo bakışı ve
   `distinctViewKeys` her karede birkaç kez tam tarama yapıyor. 100.000
   parçacıkta ölçülmedi.

## Faz 2: etiketler disk bake'inde kalıcı — SimCache v6 (2026-09-28, 16. parti)

✔ **Derlendi ve doğrulandı.**
- `rt_test_fluid_labels_cache_ipc.py` PASS: diskten okunan 5. karede 5.324 body,
  `unknown 0`. Derlemeden önce aynı okuma 5.324/5.324 `unknown` veriyordu.
- `ram_only` disk bağlamasını korudu (`valid` true, `ram_frames` 0).
- Test düzeltmesi: `check_report(require_classified=True)` sınıflandırıcının
  koşmasını şart koşuyor. Oynatmada sınıflandırıcı koşmaz, etiket dosyadan gelir.
  Test artık `primary_complete`'i kendisi doğruluyor.
- Açık: frozen bitinin diskten dönüşü ayrıca test edilmedi (aynı maske).

Görünüm yönlendirmesi etikete geçmeden önce kapatılması gereken kapı
(`FLUID_PARTICLE_LABELS.md`: bilinmeyen etiketi body gibi sunmak yasak).

**Hata derlemeden ÖNCE ölçüldü.** Eski derlemede geçici bir sıvı domain'i
0..6 kare bake edildi ve 5. kare okundu: 5.324 parçacığın **5.324'ü `unknown`**.
Bake sonrası RAM'de kare yok (`ram_frames 0`), yani okuma diskten yapıldı. Kök:
yazıcı `flags` yazmıyordu, okuyucu sıfırlıyordu.

Düzeltme:
- **`SimCache.h` kVersion 5 → 6.** Parçacık başına `flags & (etiket bitleri
  8..11 | frozen biti 3)` yazılıyor ve okunuyor. Outflow biti (1) o adımlık bir
  işaret, dışarıda kaldı.
- **Eski (v5) cache'ler reddedilir** ve yeniden simülasyona düşer. Dosyadaki
  önceki sürüm kararlarıyla aynı: dosya veriyi taşımıyorsa oynatmak dürüst değil.
- **Yeni IPC seçeneği `sim_cache.clear {ram_only:true}`.** RAM karelerini atar,
  disk bake'ini bağlı tutar. Oynatma RAM'i tercih ediyor ve RAM `flags`'i zaten
  taşıyor; bu seçenek olmadan bir disk testi kırık dosyada da geçebilirdi.
  Python: `rt.sim_cache.clear(ram_only=True)`.

Derleme: C++ (`SimCache.h/.cpp`, `RtApi.h`, `RtApiSimNodes.cpp`, `RtIpc.cpp`,
`RtPython.cpp`). Shader yok, yeni .cpp yok.

1. **Derleme.**
2. **`python scripts/test/rt_test_fluid_labels_cache_ipc.py`.** Ben koşarım.
   Görmen gereken: `PASS disk frame 5` ve `unknown 0`, `body > 0`.
   Eski derlemede bu script `ram_only clear unbound the disk bake` ile düşüyordu
   (seçenek yoktu); bu da beklenen negatif koldu.
3. **★ Sinsi olan: eski bake'ler.** v5 ile bake edilmiş bir sahne artık diskten
   oynamaz, yeniden simüle eder. Bir sahne "bake'im kayboldu" gibi görünürse
   sebep budur; yeniden bake gerekir.

## Tek domain / Faz 2: simülasyon etiketleri ilk çekirdek partisi (2026-09-28, 15. parti)

**Kullanıcı önceliği:** Solid/splat ve materyal paneli işi ertelendi. Doğrudan
simülasyon çekirdeği. Bu parti Faz 2'nin tamamı değildir: parçacık etiketini
ve ölçüm sözleşmesini kurar; render yönlendirmesi, whitewater depolama
birleşmesi, mist ve dönüşüm defteri sonraki işlerdir.

`FluidParticleLabels` simülasyon sonunda body/spray/frozen yazar. Komşuluk
kriteri ilk sürümdür (1,5 voksel yarıçap, 2/6 komşu histerezisi). Etiket
flags bit 8..11'de parçacıkla taşınır; görünüm/fizik ayarı tarafından yazılmaz.
Whitewater etiketleri aynı sözlükte, ayrı ve kütlesiz kaynak olarak raporlanır.
UI Measure, Python ve IPC `particle_labels` aynı çekirdek raporunu okur.

1. **C++ derleme, kullanıcıda.** Dört yeni `.cpp` proje/filtre kayıtlarında.
   Shader değişmedi. Codex derleme veya uygulama çalıştırmadı.
2. **Yeni simülasyon adımı, sonra pause:** ayrı terminalde
   `python scripts/test/rt_test_fluid_labels_ipc.py "Physics Domain 1"`.
   Beklenen: PASS; primary toplamı particle_count. Panel Measure ve
   `rt.fluid.get(domain)["particle_labels"]` aynı değerleri göstermeli.
3. **Eski disk cache / seed / granül:** `--allow-unclassified`.
   Unknown beklenebilir; olmayan ölçüm başarılı sınıflandırma sayılmamalı.
4. **Fizik regresyonu:** donma/erime frozen'ı değiştirir; whitewater
   secondary'de kalır; etiketleme konum/hız/kütleyi değiştirmez.
5. **Maliyet:** `particle_labels.last_step.milliseconds`; büyük sahnede
   kaydet. Yeni CPU geçişinin maliyeti henüz canlı ölçülmedi.

Ayrıntılı sözleşme, sınırlar ve bağımsız çekirdek test komutu:
[FLUID_PARTICLE_LABELS.md](FLUID_PARTICLE_LABELS.md).

**Statik doğrulama:** IPC capability/descriptor audit PASS (629 metot);
proje XML'i ve dört yeni `.cpp` kaydı PASS; Python probunun sözdizimi ve
ortak rapor bağlantıları PASS. Bunlar C++ derlemesi veya canlı simülasyon
testi değildir.

**2026-09-28 kullanıcı derlemesi sonrası canlı IPC:**

- **Derleme ✔** kullanıcı derledi; çalışan uygulama yeni alanları döndürüyor.
- **Etiket/sayım ✔** geçici sıvı domain'i: yeni seed 5.324 unknown → ilk
  adımda 5.324 body, sonraki 7 adımda sayım tam. Tek parçacık unknown → spray.
  `fluid.get` ve `fluid.list_domains` sayım sözleşmeleri geçti.
- **Görünümden bağımsızlık ✔** surface → particles → fog → surface
  geçişlerinde mevcut etiket sayıları değişmedi.
- **Frozen ✔** tabana destekli 968 soğuk parçacık frozen; termal sayaçla
  birebir. Termal zincir kapatılıp adımlanınca 968 body, frozen 0.
  Bu, termal kapatma ile serbest bırakmayı doğrular; sıcaklık yükselterek
  gerçek erime ayrıca test edilmedi.
- **Clear ✔** canlı sayımlar sıfırlandı; `last_step` son ölçüm olarak kaldı
  (sözleşmeye uygun).
- **Timeline/Vulkan ✔** 40³ grid, 100.000 parçacık, kare 1–4: her karede
  100.000 body, unknown 0. `fluid.step_stats` GPU P2G/pressure/G2P/density
  yolunu doğruladı; etiketleme hâlâ CPU'da.
- **Maliyet ⚠** 5.324 parçacık: 0,3634–0,4103 ms (medyan 0,3864).
  100.000 parçacık: **11,4582–12,7358 ms/adım**. İşlev geçti, büyük sahne
  maliyeti kabul edilmiş sayılmamalı; etiketleme optimizasyonu açık iş.
- **Python/IPC ✔** `rt.fluid.get` ve dış IPC raporları gaz domain'inde aynı;
  `available:false` doğru. Var olmayan domain açık hata döndürdü.
- **Sahne geri alındı ✔** test domain'leri kaldırıldı; özgün
  `Physics Domain 1` gaz domain'i etkin, timeline kare 0. Küp değiştirilmedi.

**2026-09-28 optimizasyon derlemesi sonrası ölçüm** (hash düğümleri yeniden
kullanılıyor, önce kendi hücre, küçük kümede seri; süre bin/classify diye ayrık):

- **İşlev ✔** `rt_probe_fluid_labels_live_ipc.py` tam geçti. Kontrol edilenler:
  dense body, görünümden bağımsızlık, izole spray, soğuk-destekli frozen 968 =
  termal sayaç, termal kapalıyken frozen 0, clear. Temizlik hatasız.
- **5.324 parçacık:** 0,23–0,30 ms (önce 0,36–0,41).
- **100.000 parçacık** (`--benchmark`, 4 timeline karesi):

  | Kare | Toplam (ms) | bin (ms) | classify (ms) |
  |---|---|---|---|
  | 1 | 10,36 | 4,16 | 6,19 |
  | 2 | 7,94 | 1,99 | 5,95 |
  | 3 | 7,79 | 2,41 | 5,38 |
  | 4 | 7,97 | 2,58 | 5,39 |

  - Önceki ölçüm 11,46–12,74 ms idi; kararlı durum yaklaşık **%35 hızlı**.
  - Tüm karelerde 100.000 body, unknown 0.
  - 1. karedeki yüksek bin süresi ilk ayırmadan geliyor; sonraki karelerde yarıya
    iniyor, yani yeniden kullanım çalışıyor.
  - ⚠ **Kalan maliyet classify'da:** 5,4–6,2 ms. Parçacıkların %93–97'si kendi
    hücresinde çözülüyor (`center_resolved`), yine de parçacık başına ~55 ns.
    Sıradaki hedef bu, ya da etiketlemeyi zaten cihazda olan P2G/G2P'nin yanına
    GPU'ya taşımak.
  - Rapor: `.tmp/fluid_labels_optimized_benchmark.json`.

Tekrarlanabilir geçici-domain probu:
`python scripts/test/rt_probe_fluid_labels_live_ipc.py` (timeline durmuşken).
Ham raporlar: `.tmp/fluid_labels_live.json`,
`.tmp/fluid_labels_timeline_scale.json`, `.tmp/fluid_labels_python.json`.
**Açık:** nonzero whitewater sayımı, eski disk cache playback, görsel panel
eşitliği, gerçek ısıtarak erime ve bağımsız C++ histerezis/compaction testi.

## Solid/Matcap: SDF yüzey gazın önüne çiziliyordu (2026-09-28, 14. parti)

13. parti ✔: kullanıcı turunda üç düzeltme de doğrulandı.

Kök: Solid/Matcap dalında sıra "hacim → SDF yüzey" idi. Hacim geçişi derinlik
YAZMIYOR (yalnız kendi ilk katkı derinliğiyle test ediyor), bu yüzden sonra
çizilen SDF yüzey önündeki gazın üstüne boyadı. Sıra çevrildi: önce yüzey
(derinlik yazar), sonra hacim. Değişen dosyalar `VulkanViewportBackend.cpp` ve
`VulkanBackend.cpp`, yalnız Solid/Matcap dalı. RayFusion/Material Preview ve RT
aynı kaldı. Shader değişikliği yok.

1. **Solid modda SDF sıvı + gaz.** Görmen gereken: yüzeyin önündeki gaz
   yüzeyin üstünde, arkasındaki gaz gizli.
2. **Solid modda splat + gaz: ⚠ ERTELENDİ (2026-09-28, kullanıcı kararı).**
   Öncelik tek domain sisteminin tamamlanması. Bu sorun ana çalışmayı bloke
   etmez; sonraki render sorunlarıyla birlikte ele alınacak. Aşağıdaki inceleme
   notları korunuyor, bu aşamada splat düzeltmesi veya tanı deneyi yapılmayacak.
   Kullanıcı teyidi: SDF artık
   doğru; splat geometrisi gazın önüne overlay gibi çıkıyor.
   Elenenler (kod okuması, ölçüm yok):
   - **Çizim sırası değil.** İki yolda da (`VulkanViewportBackend.cpp` ~4299
     instanced batch, `VulkanBackend.cpp` ~17760) splat havuzu instanced raster
     batch'te, hacim geçişinden ÖNCE çiziliyor (`VulkanBackend_Raster.cpp:2069`
     splat havuzu muafiyeti).
   - **Transmission replay değil.** Solid'de `hdrPassActive` false, replay koşmuyor.
   - **Debug dot overlay değil.** `fluid_debug_overlay` ayrı bir anahtar
     (`scene_ui_gizmos.cpp:591`).

   Hacim Solid'de şöyle çalışıyor (`material_preview_volume.frag`):
   - Snapshot yok (`lightDir0.w=0`); `gl_FragDepth` = ilk yoğunluk örneğinin
     derinliği (`firstContributionT`).
   - Solid pipeline'da depthTest açık, depthWrite kapalı.
   - Mantıken önde duran gaz splat'ın üstüne karışmalı.

   Sıradaki ajan için sorular, **ölçerek**:
   - (a) Splat batch'i Solid'de depthWrite ile mi çiziliyor? Instanced pipeline'ın
     depth state'ine bak.
   - (b) Hacim fragmanları splat piksellerinde derinlik testinden mi düşüyor,
     yoksa geçip çok ince mi karışıyor? Deney: `gl_FragDepth`'i geçici olarak 0
     yap. Gaz splat'ın üstüne çıkıyorsa sorun (b) derinliği.
   - (c) Splat'ın Solid'de hacimden sonra çalışan başka bir çağrısı var mı?
     RenderDoc veya `markRasterStage` sırası.
   SDF için yapılan sıra değişikliği (yüzey → hacim) splat'ı etkilemedi.
3. **Bilinen yaklaşım:** Solid'de hacim, yüzeyin arkasındaki yoğunluğu da
   biriktiriyor (opak snapshot yok). Yüzeyin önündeki ince gaz biraz koyu
   görünebilir; bu önizleme için kabul edildi.

## RT: sisin içindeki gaz + Liquid Body + fog düğmesi VDB'ye (2026-09-28, 13. parti)

12. parti derlendi ve çalıştı. Kullanıcının turundan üç bulgu çıktı:

- **RT'de sis alanının içindeki gaz çizilmiyor, dışındaki çiziliyor.** RayFusion'da
  doğru. Kök: gaz closest-hit'i tek hacim yürüyor. Sahnede sis (0..1,8 m) ve gaz
  (0..5 m) kutuları aynı köşeden başlıyor. Traversal'ı kazanan kutunun tamamı
  yürünüyor, ışın onun çıkışından devam ediyor ve öbür hacmin o kutunun içindeki
  kısmı atlanıyor.
  Düzeltme (`volume_overlap_selection.glsl` `resolveGasOverlapSegment` +
  `volume_closesthit.rchit`):
  - Işın her gaz/sis giriş-çıkışında segmente kesiliyor.
  - Örtüşme segmentinde iki hacim TEK ortam olarak yürünüyor. Emisyonlu ya da
    programlı olan birincil oluyor. Öbürü "companion": yoğunluğu sönüme ve
    saçılmaya ekleniyor, kendi ışık march'ı her gölge terimine çarpan olarak
    giriyor (T_a·T_b toplam ortam için tam).
  - Seçim traversal sırasına değil tam aday kümesine bakıyor, yani deterministik.
  - Sınırlar: aynı anda üçüncü bir hacim yok sayılır; companion'ın emisyonu ve
    materyal programı çalışmaz.
- **Fog seçiliyken materyal düzenleme VDB paneline geçmeli.** "Edit Fog
  Medium..." hacmi seçiyordu ama Properties Simulation sekmesinde kalıyordu.
  Artık `tab_to_focus = "VDB"`.
- **SDF yüzeyde hacim materyali açılıyor ama etkisiz.** Iso yürüyüşü o
  shader'dan yalnız üç alan okuyor. Tam gaz shader editörü yerine "Liquid Body"
  bloğu geldi (Absorption Color, Absorption, Refraction Tint). Tint, Surface
  Material bağlıyken gri; soğurma materyal altında da geçerli. Yeni IPC:
  `fluid.get_surface_interior` / `fluid.set_surface_interior`. Panel artık bu
  shader'ı kendisi oluşturmuyor: oluşturduğu gaz preset'i ayarlanmamış kalır, 16
  adım yürür ve gövdeyi siyah çizer.

Derleme: **shader değişti** (`volume_closesthit.rchit`, `volume_overlap_selection.glsl`).
Yeni .cpp yok.

1. **Derleme + shader.** Görmen gereken: hatasız. `volume_closesthit.spv` zaman
   damgası yeni olmalı. Eskiyse 2. madde eski davranışı gösterir, düzeltme
   işe yaramamış gibi görünür.
2. **Açık sahne, Vulkan RT (en önemli).** Görmen gereken: sisin içindeki gaz
   görünür ve RayFusion'a benzer. Bozuksa:
   - Gaz hâlâ yoksa companion seçilmiyor (`gasOverlapCandidate` filtresi,
     `source_type`).
   - ★ **Sinsi olan:** gaz görünür ama sis gaz kutusunun sınırında renk ya da
     parlaklık değiştiriyor (gaz kutusu sisin içinde "kutu" gibi seçiliyor).
     Bu, companion'ın saçılma ya da gölge karışımının sis tek başınayken
     yapılan hesapla uyuşmadığını gösterir. Ekran görüntüsü yeter.
   - Sisin gazdan bağımsız alt kısmı değişmemeli (tek üyeli segment = eski yol).
   - Sorun yalnız sıvı **fog** olarak çizilince vardı (kullanıcı teyidi); SDF ve
     splat modunda gaz zaten doğruydu. Bu iki modda görüntü **değişmemeli**:
     SDF hacmi (source_type 4) ve splat geometrisi companion adayı değil. Değişirse
     `gasOverlapCandidate` fazla geniş demektir.
3. **IPC: `fluid.get_surface_interior {domain:"Burning Fuel Liquid"}`.** Ben
   koşarım. Görmen gereken: `ok`, `tint_active` Surface Material'a göre. Fog
   modunda `created:false` olabilir, bu normal. `set_surface_interior`,
   yüzey hiç çizilmediyse açık bir hata döner.
4. **Panel.** Görmen gereken:
   - Output > Fog View > "Edit Fog Medium..." Properties'i VDB sekmesine geçirir.
   - SDF modunda "Volumetric Absorption & Density" yok, yerinde 3 kontrollü
     "Liquid Body" var.
   - Surface Material bağlayınca Refraction Tint gri olur.

## Faz 1-UI / 1. parti: Physics Domain paneli altı sekme (2026-09-28, 12. parti)

✔ Derlendi, kullanıcı turunda çalıştı. Turda çıkan eksikler 13. partide.

> `BIRLESIK_MADDE_DOMAIN_TASARIMI.md` §8b. Üç sekme (Setup & Grid · Solver &
> Physics · Shading & Rendering) ve renkli Gas/Fluid düğmeleri yerine
> **Domain · Matter · Environment · Solvers · Output · Measure**. YALNIZ
> TAŞIMA: widget kodu değişmedi, script ile blok blok yeniden sıralandı (her
> sınır metne karşı doğrulandı, parantez dengesi sekme başına kontrol edildi).
> - Domain: "Contents" combo (Gas / Liquid; eski renkli düğmelerin mantığı
>   aynen), cihaz, kalite profili, çözünürlük, sınırlar, sıvı tohumlama, Flow
>   Sources.
> - Matter: yanma (gaz); Liquid Material (preset + reoloji + termal sıvı),
>   granül, yanıcı sıvı/gaz bağlaşımı, madde bağlamalarının FİZİK yarısı.
> - Environment: dünya ortamı (salt okunur), domain override'ı (artık SIVI
>   için de — çözücü zaten okuyordu, panel yalnız gazda gösteriyordu),
>   yerçekimi (sıvı).
> - Solvers: gaz kanalları/kaldırma/türbülans; APIC/FLIP (karışım,
>   dissipation, coupling, basınç), reseed.
> - Output: gaz hacim shader'ı, Liquid Display (mod, Now drawing, debug),
>   madde bağlamalarının GÖRÜNÜM yarısı ("Substance Look"), splat, SDF,
>   whitewater, fog view, yüzey hacim shader'ı.
> - Measure: istatistikler, adım süreleri, VDB export.
> - Sekme dışında kalanlar (her karede çalışmalı): sim bake kontrolleri,
>   otomatik reseed, Remove Domain.
> - Söküldü: `if (false && …)` ile kapalı eski FluidObject bake bloğu (185
>   satır, ölü). Reset düğmesinin rengi (tema kuralı).
> - **Yeni IPC (kural 1):** `fluid.get_environment` / `fluid.set_environment`
>   (gaz ve sıvı): ortam override'ı yalnız paneldeydi.
> - Parti 2'de: `SelectableType::Material` + gömülü editörlerin sökülmesi
>   (gaz "Unified Volume Shader", sıvı "Volumetric Absorption & Density",
>   köpük `drawInlineMatEditor`); whitewater üretim eşikleri → Solvers.

Sıra: 1–2 hızlı; 3 elle tur.

1. **Derleme.** Yalnız UI + API; shader yok, yeni dosya yok. Hata beklenen
   yer: taşınan bloklarda kapsam (`fp`, `fp_edited` sekme başına yeniden
   tanımlandı). Kullanılmayan `mats` uyarısı (Matter döngüsü) zararsız.
2. **IPC:** `fluid.set_environment {domain, override_enabled: true,
   ambient_kelvin: 350}` → `fluid.get_environment` `effective_ambient_kelvin
   350`; panel Environment sekmesinde aynı değer.
3. **Elle tur, gaz ve sıvı domain'inde:** her sekmeyi aç. *Görmen gereken:*
   hiçbir bölüm iki sekmede düzenlenmiyor; gaz domain'inde sıvı bölümü yok
   ve tersi. ★ SİNSİ: bir bölüm HİÇ görünmüyorsa (yanlış faz dalına düşmüş)
   kimse fark etmez — eski panelde olup yeni panelde bulamadığın bir ayar
   varsa söyle.

## Sis segmenti 1 mm'ye kırpılıyordu + düz ortam geçişi bounce yemez (2026-09-28, 11. parti)

> 10. partinin aleti ÖLÇTÜ (3. derleme, exe 11:31, bölge öz-denetimi geçti).
> Koyu tabaka, 10 bounce: yolların **%58'i** bounce tavanında ölüyor; yol
> başına 7,92 bounce'un **6,44'ü sis segmenti**. İçeriden hakem aramalarının
> yalnız %0,04'ü çıkışı buluyor. Sıvısız sis bile yol başına 2,66 segment.
> **Kök:** `volume_intersection.rint` sis aralığını `min(tFar, gl_RayTmaxEXT)`
> ile kırpıyordu. `gl_RayTmaxEXT` traversal SIRASINA bağlı; sıvı kutusu
> girişini 1 mm geride raporladığı için, BVH onu önce ziyaret edince sis 1 mm
> yürüyüp ~1 cm atlıyor, tekrar giriyordu. Her atlama bir bounce. Kullanıcının
> "her bounce çok kısa, adımdan bağımsız" gözlemi buydu. Hakem de çıkışı o
> 1 mm'de arıyordu.
> **Düzeltme (derlenmedi):** (1) gaz/sis kutusu gerçek çıkışı raporlar (sky
> cloud ve sıvı kırpmayı korur). (2) Yeni `BOUNCE_MEDIUM_PASS`: yön değiştirmeyen
> sis devamı bounce bütçesinden düşmez, yalnız geçiş tavanından düşer. Sayaç
> `gas_segments_charged` → **`medium_passes`** (anlam değişti, ad da değişti).
> Shader: `volume_intersection.rint`, `volume_closesthit.rchit`, `raygen.rgen`,
> `rt_payload.glsl`, `volume_instrumentation.glsl`. Scratch'te glslc temiz.

> **✔ 1. derleme (exe 11:49, spv 11:50–11:51), ölçüldü:**
> - **1 ✔**. İlk koşu yine "main thread dispatch timeout" verdi (açılıştan
>   sonraki ilk Rendered geçişi; 10. partide de görüldü), tekrarı temiz.
> - **2 ✔** Koyu tabaka, 10 bounce: `capped` %58,2 → **%0,0**; sis segmenti/yol
>   6,44 ücretli → 0,63 ücretsiz; içeriden hakem çıkışı %0,04 → **%99,5**.
>   Yalnız-sis bölgesinde de `capped` %15,6 → %0. 10 ve 51 bounce birebir aynı:
>   artık hiçbir yol tavana ulaşmıyor.
> - **3 ✔** Kararma yok. ★ Görünüm değişti: sis artık gerçek kalınlığı kadar
>   birikiyor, yani çok daha yoğun (önceki "üstten saydam" görünüm hatanın
>   parçasıydı). Jet sütlü gri-beyaz, çünkü kırılan ışın artık alttaki sisin
>   tamamından geçiyor.
> - **Yeni açık madde:** varsayılan sis tarifi (`makeLiquidFogShader`) kahverengi-bej
>   veriyor. Yorumu "emilim rengi suyu mavi gösterir" diyor, ama MAVİ emilim
>   maviyi yutar. Tarif ya da yorum ters; ayrı iş.
> - Script'in otomatik launch-y kalibrasyonu söküldü: düzeltme hacim ışını
>   oranını tersine çevirdi ve bölgeleri kolona taşıdı. Yön sabit (top-down,
>   ölçüldü).
> - **4 ✔** (kullanıcı): gaz + katı sorunsuz.
> - **5 ~** (kullanıcı): sky cloud çalışıyor; kamera-bulut-içi/dağ testi yapılmadı.
> - **⚠ Yeni açık, bu partiden bağımsız (kullanıcı gözlemi, sonra bakılacak):**
>   1. **Sky cloud parametresi RT modda cihaza aktarılmıyor.** Parametre
>      değişince görüntü değişmez, ancak mod değişimiyle güncellenir. Tanıdık
>      sınıf: yazan yolun yeniden yayın/resync istememesi (bkz. IPC
>      `render_mode` resync hatası, gas shader `republishGasLookAndRepaint`).
>   2. **Solid modda sıvı yüzeyi gaz yüzeyinin ÖNÜNDE çiziliyor** — ikisi
>      arasında derinlik hesabı yok. RayFusion ve RT doğru. Raster önizleme
>      yolunda hacim/yüzey derinlik sıralaması eksik.

Sıra: 1 önce (spv); 2–3 ölçüm; 4–5 regresyon, gözle.

1. **Shader + derleme.** `compile_shaders.bat`: rint + iki rchit + raygen +
   photon (rt_payload'ı include ediyor). *Eski rint spv:* ölçüm hiç değişmez.
2. **Ölçüm:** `python scripts\test\rt_probe_fog_bounce_budget_ipc.py`.
   *Görmen gereken:* bounces10 `capped` %58 → birkaç %; `medium/path` ~1–3
   (eskiden 6+); `in/found` oranı belirgin artmalı; fog-only `capped` ~0.
   *capped düştü ama `pass%` yükseldi:* ping-pong sürüyor, artık geçiş
   tavanına takılıyor — bana getir.
3. **Görsel, aynı sahne, 10 bounce:** sise gömülü yüzey artık koyu değil
   (51 bounce'taki görüntüye yakın). Tek mozaik yeter.
4. **Regresyon — gaz + katı:** yerde duran bir obje üstünde duman domain'i
   (ateş/duman sahnesi). Obje görünür, duman objenin arkasına geçmez.
   *★ SİNSİ:* obje dumanın içinde "yarı saydam" görünüyorsa katı probu artık
   tek başına yetmiyor (eskiden kırpma da yardım ediyordu).
5. **Regresyon — sky cloud / Nishita:** kamera bulut içindeyken önündeki dağ
   görünür. Bu yola dokunulmadı, ama intersection dosyası değişti.

## Bounce bütçesi ölçü aleti + render ayarları ve sis shader'ı IPC'de (2026-09-28, 10. parti)

> 9. partinin 4. maddesinden doğdu: sise gömülü sdf sıvısı siyah, total
> bounce 51 yapınca aydınlanıyor. Kullanıcının Bounce Count gözlemi (10'da
> içi sarı = tavan) bütçe tükenmesini gösterdi, ama HANGİ olayın harcadığı
> kaynaktan okunamadı. Bu parti onu ölçülebilir yapar; düzeltme DEĞİL.
> - **Sayaçlar** (`render.volume_stats`): `paths_traced`, `paths_bounce_capped`,
>   `paths_pass_capped`, `charged_specular/diffuse/transmission/other`,
>   `free_passes`, `gas_segments_charged`, `arbiter_started_inside/_inside_found`.
>   Hepsi (eskiler dahil) `render.volume_counters {enabled, region:[x0,y0,x1,y1]}`
>   ile bir piksel bölgesine sınırlanabilir. ABI 25 → 40 kelime (C++
>   static_assert + GLSL aynı sırada).
> - **Kural 1 açıkları kapandı:** `render.get_settings` / `render.set_settings`
>   (max/diffuse/transmission bounce, debug_view), `fluid.get_fog_shader` /
>   `fluid.set_fog_shader`. Python: `rt.render.get_settings/set_settings`,
>   `rt.render.volume_counters(enabled, region)`, `rt.fluid.get_fog_shader/set_fog_shader`.
> - `makeLiquidFogShader` artık paylaşılan (`Fluid/FluidFogDensity.h`).
> - Panel metrik raporuna (Performance > Volumetrics) yeni sayaçlar eklendi.
> Yeni .cpp YOK. Shader değişti: `raygen.rgen`, `volume_closesthit.rchit`,
> `volume_instrumentation.glsl` (closesthit.rchit da include ediyor).

> **1. derleme (exe 10:53, spv 10:44–10:45), canlı IPC:**
> - **1 ✔** spv'ler taze. Scratch'te `spirv-dis` ile doğrulandı: raygen'deki
>   instrumentation struct'ı 40 üyeli, bölge kapısı 25–28'i okuyor.
> - **2 ✔** kimlik testi solid PASS, rendered PASS (SKIP yok). 7. madde artık
>   ORİJİNAL kombinasyonla (yüzey varsayılanı + steam→`fog`) geçiyor, yani
>   9. partinin `fog` ayrıştırıcı düzeltmesi doğrulandı. Rendered'ın ilk
>   denemesi `main thread dispatch timeout` verdi, tekrarında temiz geçti.
>   Açılıştan sonraki ilk Rendered geçişi olabilir.
> - **3 ✔** `render.get/set_settings` (20/6 yazıldı ve geri okundu, 99 ve ters
>   bölge reddedildi).
> - **4 ✗ ÖLÇÜ ALETİ KIRIK, sayılar geçersiz.** Her bölge tüm görüntüyü saydı:
>   üst yarı = alt yarı = 2% köşe = 1680×945×spp. Kök: sayaç çağrısı tek bir
>   cihaz seçiyordu (`g_ctx->backend_ptr`; ana döngü onu her karede shading'e
>   göre değiştiriyor), `render.start` ise render cihazında iz sürüyor. Kanıt:
>   `enabled=false` + render → `enabled=true`; yarım bölge + render → geri
>   okuma `0,0,1,1`. **Düzeltildi, derlenmedi:** `RtApi.cpp` sıfırlamayı TÜM
>   Vulkan cihazlarına yazıyor, okurken hepsini topluyor. Script'e öz-denetim
>   eklendi: 2% köşe tam görüntünün %1'inden fazlasını sayarsa ölçümü reddeder.
>   ★ Aynı kusur 9. partideki elle ölçümü de etkiledi (`B_nofog` kolunun
>   tamamen 0 çıkması); o sayılara güvenme.
> - Panel (Performance > Volumetrics) hâlâ yalnız aktif cihazı sıfırlıyor ve
>   gösteriyor; tek cihazlı panel kullanımı için doğru, IPC'ye dokunmuyor.
>
> **2. derleme (exe 11:18):** öz-denetim ölçümü DOĞRU olarak reddetti (2%
> köşe = tam görüntü). Cihaz düzeltmesi tek başına yetmedi. **Asıl kök:**
> `rtapi::renderStart` her `render.start`'ta `setVolumeInstrumentation(true)`
> çağırıyordu: sayaçlar bölgesiz sıfırlanıp açılıyordu ("probe 0 raporlamasın"
> diye eklenmişti). Kanıt: kapat + render → `enabled=true, paths=0`; köşe +
> render → bölge `0,0,1,1`. **Söküldü.** Artık render sayaçlara dokunmuyor;
> çağıran kendisi açar. Ayrıca cihaz listesi artık `g_backend`'i açıkça
> içeriyor (Solid'de `backend_ptr` viewport'tur) ve `render.volume_stats`
> cihaz başına `devices[]` satırı döndürüyor (rol, enabled, bölge, yol).
>
> **Sıradaki derlemede:** yalnız 4. madde (C++ değişti, shader DEĞİŞMEDİ).
> Script önce öz-denetimi koşar; geçemezse `devices[]` hangi cihazın bölgeyi
> kaybettiğini söyler.

Sıra: 1–3 bağımsız ve hızlı; 4 hepsinin sonucunu kullanır.

1. **Derleme + shader.** `compile_shaders.bat` ÜÇ spv'yi yenilemeli (raygen,
   volume_closesthit, closesthit). *★ SİNSİ:* eski spv + yeni exe = GLSL 25
   kelime görür, C++ 40 yazar. Çökmez, ama `enabled` sonrası her şey kayar:
   yeni sayaçlar hep 0 ya da saçma çıkar. Önce spv zaman damgasına bak.
2. **Eski davranış bozulmadı.** `python scripts\test\rt_test_volume_slot_identity_ipc.py`
   ve `... rendered`: PASS (9. partinin `fog` ayrıştırıcı düzeltmesi de bununla
   doğrulanır). Performance > Volumetrics panelinde eski sayaçlar eskisi gibi.
3. **Yeni IPC tek tek.** `render.get_settings` → `max_bounces 10`;
   `render.set_settings {max_bounces: 20}` panelde "Total Bounces 20" ve
   accumulation sıfırlanır; `debug_view: 6` panelde "Bounce Count".
   `fluid.set_fog_shader {density_multiplier: 0}` duraklatılmış karede sisi
   hemen söndürür. *Kare değişince geri gelirse:* sync shader'ı yeniden kuruyor.
4. **Ölçüm (bana bırak):** `python scripts\test\rt_probe_fog_bounce_budget_ipc.py`.
   Koyu tabakada 10 / 51 bounce ve sis kapalı kolları + sadece-sis referansı.
   Beklenen: 10'da `capped%` yüksek, 51'de düşük; `gas/path` ile
   `spec/path` hangi olayın bütçeyi yediğini söyler. *`paths 0`:* bölge
   izdüşümü ıskaladı (satırdaki region'a bak).

## Faz 1 / 1. parti: tek görünüm çözücüsü + aynı domain'de yüzey VE fog (2026-09-28, 9. parti)

> `BIRLESIK_MADDE_DOMAIN_TASARIMI.md` Faz 1. "Ne çiziliyor" artık TEK yerde:
> `Fluid/FluidViewResolver` (yeni). Hacim rotası, splat köprüsü, `fluid.get`
> ve paneldeki "Now drawing" onu okuyor. Bir sıvı domain'i artık iki hacim
> taşıyabilir: yüzey birincil slotta, fog kendi slotunda ve kendi shader'ıyla
> (`fluid_fog_shader`, yeni `FluidDomainFogVolume.cpp`). Madde görünümüne
> `fog` eklendi. Ölü köpük slotu (hiç ayrılmamıştı) fog slotu oldu.
> Yeni dosyalar vcxproj'da: `FluidViewResolver.cpp`, `FluidDomainFogVolume.cpp`.
> "Physics Domain" adı (8. parti) de bu derlemeye giriyor.

> ✔ 1. derleme (exe 2026-09-28 00:16), canlı IPC:
> - **1 ✔** derlendi.
> - **2:** 0–6. maddeler solid'de PASS. 7. madde IPC'de kırıldı:
>   `fluid.set_substance_material` `fog`u reddediyordu. Enum, çözücü, panel ve
>   kayıt `Fog`u biliyordu, yalnız `RtApiFluid.cpp`'deki ayrıştırıcı
>   bilmiyordu. **Düzeltildi, derlenmedi.**
> - Aynı 7. madde **ters kombinasyonla** (domain `fog` + steam→`sdf`) koşuldu:
>   solid ve rendered'da **tam PASS**, iki backend'de de SKIP yok. Bir domain'de
>   iki ayrı hacim var (sdf ve fog `vdb_id`'leri farklı), ikisi de aktif; fog
>   slotu `source=nanovdb`, `fluid.get views[]` ile `render.volume_slots` aynı
>   kimlikleri gösteriyor. Yani çözücü + fog slotu çalışıyor; eksik olan yalnız
>   IPC yazıcısıydı.
> - **Test kusuru, düzeltildi:** `flow_source.create` domain'i sıfırlıyor
>   (`particle_count` 20736'dan 0'a), setup'ta tohumlanan etiketsiz su gidiyor.
>   Eski 7. madde yalnız `untagged` beyanına bakıyordu, `live`a bakmıyordu.
>   Artık kaynak eklendikten sonra yeniden tohumluyor ve `live`ı da istiyor
>   (iki kopya da güncel).
>
> - **4 — ⚠ SİNSİ MADDE ÇIKTI (2026-09-28, aynı exe, IPC + path tracer 64 spp).**
>   Domain `fog` + steam→`sdf`, dolu havuza düşen jet. İkisi aynı anda çiziliyor
>   (özellik ✔). Ama **sdf yüzeyinin sise gömülü kısmı siyaha yakın.**
>   A/B, her seferde tek değişken:
>   - Tohum yok, sis canlı değil: aynı jet tabanda açık mavi ve temiz. Yani
>     kararmanın sebebi zemin teması değil, sis.
>   - Sise yarı gömülü katı kutu: kutunun sisteki kısmı yalnız hafifçe griye
>     dönüyor. Yani sis gölgesi bu mertebede, siyah değil. Bu **hata, fizik değil.**
>   - Sayaçlar: `density_samples/volume_rays` 0,82 (sağlıklı),
>     `layered_handoffs` 3.086. Işın hapsi (çakışık kutu dersi) **yok**. Sis
>     segmenti yüzeyde kesilince `skipGasVolumes` kalkıyor, yani bounce da
>     sayılmıyor.
>   - ✔ **KÖK (kullanıcı buldu):** total bounce 64 yapınca sis aydınlandı; her
>     bounce ışığı bir adım daha ileri taşıdı. Sisteki her SAÇILMA olayı gerçek
>     bir bounce sayılıyor (`didScatter` dalı bounceType'ı varsayılan bırakıyor
>     → `bounce++`, bounce ≥3'ten itibaren RR). Katı kutu NEE ile doğrudan ışık
>     alıyor. Sıvının cam lobunda NEE yok, yani ışığı yalnız sisten rastgele
>     yürüyüşle çıkması gereken devam yollarından alıyor, ve bütçe o yürüyüş
>     bitmeden tükeniyor → siyah. "Bounce sayılmıyor" diye elediğim şey yalnız
>     kesilen segmentti, saçılma değildi.
>   - **Ölü alan:** `VkVolumeInstance::max_volume_bounces`
>     (`vulkan_volume_types.h:239`) ne yazılıyor ne okunuyor. Ayrı bir hacim
>     bounce bütçesi niyet edilmiş, hiç bağlanmamış.
>   - Yan bulgu: sıvı fog shader'ı (`fluid_fog_shader`) **yalnız panelden**
>     düzenlenebiliyor, IPC yolu yok (kural 1). Bu yüzden A/B sis yoğunluğunu
>     sıfırlayarak yapılamadı.
>
> **Sıradaki derlemede yalnız:** `python scripts\test\rt_test_volume_slot_identity_ipc.py`
> (+ `rendered`), orijinal kombinasyonla PASS. Sonra 3–7 (elle/görsel).

Sıra: 1–3 bağımsız ve hızlı; 4–6 görsel, 1–3 geçmeden bakma.

1. **Derleme.** `ParticleSimulation.h`, `scene_data.h`, `VulkanBackend.h`,
   `RtApi.h` değişti → uzun derleme. *Link hatası "FluidViewResolver" /
   "syncDomainFogVolume":* vcxproj girdisi.
2. **Tek komut, iki kez** (uygulama açık, timeline duraklatılmış):
   `python scripts\test\rt_test_volume_slot_identity_ipc.py` ve aynısı
   `rendered` argümanıyla. *Görmen gereken:* `PASS` (+ solid'de `[SKIP] render`).
   Yeni 7. madde bir "steam" Flow Source'u fog'a bağlar. *Bozuksa:*
   - "sdf and fog views" FAIL → çözücü iki görünümü vermiyor (bağlama
     okunmadı ya da parçacık etiketi yazılmadı; `fluid.get` `substances`'a bak).
   - "separate fog volume" FAIL, views doğru → fog senkronu çalışmıyor
     (`syncDomainFogVolume` çağrılmıyor ya da `domain_fog_*` boyutsuz).
   - "fog slot active" FAIL, domain satırı doğru → hacim backend'e ulaşmıyor.
3. **Eski sahneler.** Yüzey (SDF) ve Particles modunda kaydedilmiş bir proje
   aynı görünmeli. *Değiştiyse:* çözücünün varsayılanı eski mod eşlemesiyle
   uyuşmuyor (`defaultViewForMode`).
4. **Görsel — asıl özellik.** Fog varsayılanlı domain + bir maddeye `sdf`
   override (ya da yüzey varsayılanı + bir maddeye `fog`): Rendered'da ikisi
   AYNI ANDA. Önceden override sisi yutuyordu. Tek mozaik yeter.
   *★ SİNSİ:* kesişim bölgesinde siyah bant → gaz→yüzey hakemi fog'dan
   yüzeye el değiştirmiyor (`nearestSurfaceSDFCrossing` kapısı).
5. **VDB/hacim paneli.** Fog hacmini seç (domain panelindeki "Edit Fog
   Medium..." düğmesi seçer): "Fog view of liquid domain ..." yazar, mod
   combo'su YOK; shader değişikliği sisi değiştirir ve kare değişince KALIR.
   Yüzey hacminde IOR/roughness hâlâ var. *Kare değişince geri dönüyorsa:*
   panel hacme yazıyor, domain'e değil.
6. **Eski fog-mod sahnesi** (27 Eylül derlemesiyle kaydedilen): sisin rengi
   ve yoğunluğu korunmalı (yüklemede `shader` → `fluid_fog_shader`).
   *★ SİNSİ:* sis mavi "Liquid Fog" preset'iyle geliyorsa göç çalışmadı —
   hata gibi değil "biraz farklı" görünür.
7. **Timeline önbellek oynatımı.** Fog'lu domain'i bake et, geri sar, oynat:
   sis her karede güncel. *Donuk ya da bir önceki kare:* fog slotu
   `invalidateSimulationRenderBindings`'de bağ çözülmüyor.
8. **★ SİNSİ: karışık domain'de yüzey küçülebilir.** SDF artık fog/splat
   etiketli parçacıkları İÇERMİYOR (önceden fog-varsayılanlı + SDF override
   domain'de etiketsiz fog parçacıkları da yüzeye giriyordu). Yüzeyin
   ötekilerden ayrılması DOĞRU davranış.
9. **Bilinen, bu partide değil:** eski bir fog-mod sahnesine sonradan bir SDF
   maddesi eklenirse yüzey, yüklenmiş (fog'a ayarlı) `shader` ile boyanır —
   bir kez mod değiştirmek preset'i yeniden uygular. İki katılımcı ortam
   (sıvı fog + ayrı gaz domain'i) aynı kutuda: eskiden de vardı, yeni değil.

## "Grid Domain" → "Physics Domain" (2026-09-27, 8. parti)

> Yalnız görünen ad ve YENİ domain'lerin varsayılan adı. Kayıtlı sahnelerdeki
> adlar (ör. "Grid Domain 1") kullanıcı verisidir, değişmez. Test script'lerinin
> varsayılan argümanı "Physics Domain 1" oldu (iki kopya da güncellendi).

1. **Derleme.** `ParticleSimulation.h` / `SceneSelection.h` / `scene_data.h`
   değişti → uzun derleme.
2. **Panel.** Simulation panelinde düğme "+ Add Domain", liste başlığı "Physics
   Domains"; yeni domain "Physics Domain 1". *Eski ad görünüyorsa:* exe eski.
3. **★ SİNSİ: eski sahnede test script'i domain'i bulamaz.** Eski projede domain
   hâlâ "Grid Domain 1"dir; script'lere adı argüman olarak ver
   (`... rt_test_fluid_fog_mode_ipc.py "Grid Domain 1"`). "domain not found"
   bir regresyon DEĞİL.

## `render.volume_slots`: hacim başına satır + kimlik testi (2026-09-27, 7. parti)

> ✔ 2. derleme (2026-09-27): `solid` PASS (viewport, packet order) ve `rendered` PASS
> (iki backend, SKIP yok — TLAS/stable_key yolu dahil).
> 1. derleme: 10 FAIL — araç TLAS'sız viewport'a kördü.
> Faz 0'ın ilk kod işi (`BIRLESIK_MADDE_DOMAIN_TASARIMI.md` §8b madde 3). Yeni IPC/
> Python metodu; `VulkanDevice::updateVolumeBuffer` artık son yüklemenin CPU
> gölgesini ve bir `upload_serial` tutuyor. ⚠ `VulkanBackend.h` değişti →
> uzun derleme. Render davranışı DEĞİŞMEZ (yalnız okuma).

1. **Derleme.** Yeni dosya yok; vcxproj değişmedi.
2. **Tek komut, hepsini koşar** (uygulama açık, timeline duraklatılmış, ~30 s):
   `python scripts	est
t_test_volume_slot_identity_ipc.py`
   *Görmen gereken:* `PASS`. Kendi domain'ini (`SlotProbe`) kurar, siler.
   *Bozuksa, hangi madde:*
   - **0 FAIL** (domain hacim sahibi değil): üretici/rota sorunu, araç değil.
   - **1 FAIL** (stable_key değişti): hiçbir şey değişmezken kimlik churn'ü —
     siyah bant sınıfı; çıktıyı bana getir.
   - **6 FAIL** (vdb_id uyuşmuyor): sahne ile backend farklı hacmi gösteriyor.
   - **5 FAIL** (has_temperature true): köpük kapalıyken SDF'ye sıcaklık yükleniyor.
   - **4 FAIL**: alakasız TLAS rebuild SDF'yi düşürüyor (bilinen tarihî arıza).
   - **2 FAIL**: particles modunda sıvı hacmi hâlâ aktif çiziliyor.
3. **★ SİNSİ: `[SKIP] render` normaldir, `[SKIP] viewport` değildir.** Render
   backend'i Solid'de hacim tutmaz (Rendered'a girince dolar). Raster viewport
   TLAS kurmaz; slotları `packet_order: true` ile gelir (ad/stable_key yok,
   `vdb_id` ile eşlenir). İlk derlemede araç bu yolu hiç görmüyordu — 10 FAIL'in
   kökü buydu, uygulama değil. Viewport SKIP görürsen araç yine kör demektir.

## IPC `fluid.set_param render_mode` artık render senkronu istiyor (2026-09-27, 6. parti)

> ✔ 2026-09-27 (exe 22:21): probe -ForceResync OLMADAN koştu — particles→surface→fog
> sırasında viewport hacmi 0→1→1, üç mod Solid'de doğru temsil. Madde 3 (panel) elle bakılmadı.
>
> Faz 0 canlı matrisinde bulundu (`BIRLESIK_MADDE_DOMAIN_TASARIMI.md` §8b 2. tur).
> Panel combo'su mod değişince `requestSimulationTimelineRenderResync()` +
> `start_render` çağırıyordu, IPC çağırmıyordu. Duraklatılmış timeline'da
> Solid/Material bir ÖNCEKİ modun temsilini çiziyordu; Rendered her karede kendi
> senkronladığı için doğruydu. Tek değişiklik: `RtApiFluid.cpp` render_mode dalı.

1. **Derleme.** Tek satırlık ekleme; hata beklenmez.
2. **Probe, -ForceResync OLMADAN.** Sıvısı olan bir domain'le (timeline duraklatılmış):
   `.\scripts\ipc\Probe-FluidRenderMatrix.ps1 -Domain "<ad>" -OutDir C:\temp\m`
   *Görmen gereken:* viewport sütununda particles satırları surface/fog
   satırlarından **bir eksik** hacim gösterir; `particles_solid.jpg` yalnız
   küreler, `surface_solid.jpg` yüzey, `fog_solid.jpg` açık mavi sis önizlemesi.
   *Bozuksa:* surface_solid boş ya da fog_solid keskin yüzey → istek hâlâ
   ulaşmıyor (exe eski mi, önce zaman damgasına bak).
3. **Panel davranışı değişmemeli.** Panelden modu değiştir: eskisi gibi anında.
4. **★ SİNSİ: Rendered'da hiçbir fark görmezsin.** Rendered zaten her karede
   senkronluyordu; "değişiklik bir şey yapmadı" sanma — fark yalnız Solid ve
   Material'da, duraklatılmış timeline'da.
5. **Bilinen, bu partide düzeltilmedi:** `fluid.step` de render senkronunu
   tetiklemiyor (yalnız `timeline.set_frame` ve Rendered tetikliyor). Script'le
   adım atıp raster'ı ölçen testler hâlâ bir önceki kareyi görebilir.

## Fog Spread + sisin blackbody'si parçacık sıcaklığından (2026-09-27, 5. parti)

> ✔ 2026-09-27: kullanıcı derledi ve test etti — sorun yok, sıcaklıkla blackbody doğru davranıyor.

> Ham splat parçacık başına yalnız 8 hücreye değiyor; seyrek sprey tek tek
> lekelere dönüşüyordu. Eski "geniş bulut" görüntüsü gerçek yoğunluk değil,
> SDF'nin yüzey yardımcısıydı (siyah küp de ondan). Yeni alan
> `fluid_fog_spread_voxels` (varsayılan 1.5, 0..6): sis rotası grid yerine
> Gauss ile yayılmış KOPYAYI yükler (`Fluid::spreadFogDensity`, yeni
> `FluidFogDensity.cpp` — vcxproj'a eklendi). Çözücü grid'i ve
> `active_density_cells` DEĞİŞMEZ. IPC/Python: `fluid.set_fog spread_voxels`,
> okuma `fluid.get fog_spread_voxels`.

1. **Derleme.** Yeni `.cpp` projede; link hatası varsa vcxproj girdisi.
2. **Script.** `python scripts	est
t_test_fluid_fog_mode_ipc.py "Grid Domain 1"`
   — 4. adım: 2.5 geri okunur, 7 reddedilir.
3. **Görsel, fog modunda.** Panelde "Volume Material (Fog)" altında
   "Fog Spread (voxels)". 0 → bugünkü lekeler; 1.5 → sürekli bulut; 3+ →
   yumuşak, geniş. Kaydırıcı DURAKLATILMIŞ karede de anında etki etmeli.
   *Etki etmiyorsa:* köprü yeniden yüklemiyor (resync isteği yutuluyor).
4. **Zemin kenarı.** Havuz domain duvarına kadar sönükleşmeden gitmeli (çekirdek
   duvarda yeniden normalize ediliyor). *Kenarda koyu şerit:* normalize çalışmıyor.
5. **Maliyet.** Büyük domain'de (≥200³) spread 6 ile kare süresine bak;
   ayrılabilir 3 geçiş + OpenMP. *Belirgin takılma:* yayma her karede değil
   yalnız yükleme karesinde çalışmalı — sık çalışıyorsa bana söyle.
6. **★ SİNSİ: yoğunluk tepe değeri düşer.** Yayma kütleyi korur ama tepeyi
   indirir; ince sprey daha soluk görünür. Bu hata değil — Density çarpanıyla
   dengelenir. Aynı spread'de "önceki kadar yoğun değil" normal.
7. **★★ SİNSİ: bugün parlayan sıvı sönebilir.** Blackbody/ChannelDriven emisyonlu
   sis artık PARÇACIK sıcaklığını (Kelvin) okuyor. Önceden sıvıda sıcaklık kanalı
   yoktu ve shader `temperature <= 0` görünce YOĞUNLUĞU sıcaklık sayıyordu —
   "sıcak akışkan" görüntüsü o yedekten geliyordu. Oda sıcaklığındaki sıvı
   (≈293 K) şimdi neredeyse hiç parlamaz (radyans ~ (T/Tmax)⁴). Bu DOĞRU
   davranış; hata sanma. Sıcak sıvı için Flow Source'ta sıcaklık override'ı ver.
8. **Sıcaklık verisi.** `fluid.get`: `particle_kelvin_measured: true`,
   `particle_min_kelvin` / `particle_max_kelvin`. Script'in 4. adımı bunu
   kontrol eder. *false:* parçacıklar 0 K (yazılmamış) doğuyor — emitter yolu.
9. **Görsel: lav.** Flow Source sıcaklık override 1400 K, sis modu, Blackbody:
   kaynaktan çıkan sıvı turuncu-sarı parlar. Termal zinciri aç: soğudukça
   koyu kırmızıya döner ve söner. *Tüm bulut tek renk:* sıcaklık kanalı
   yüklenmiyor (köprüde `splatFogTemperatureKelvin` false dönüyor).
10. **Kenar rengi.** Bulutun ince kenarı çekirdekle aynı sıcaklık rengini
    taşımalı (kütle ağırlıklı yayma). *Kenarlar kırmızı, çekirdek beyaz ve
    hepsi aynı sıcaklıktaysa:* sıcaklık yoğunlukla birlikte bulanıklaşıyor.

## ★ Fog düzeltmesi: sıvının yoğunluk sayacı her adımda sıfırlanıyordu (2026-09-27, 4. parti)

> İlk fog derlemesinde sis hiçbir modda çizilmedi; VDB panelindeki Fog seçimi de
> artık çizmiyordu (o seçim şimdi aynı domain moduna yazıyor). CANLI ÖLÇÜLDÜ:
> 104k parçacık, splat her adımda çalışıyor (`[FluidGPU DensitySplat]` logu),
> ama `active_density_cells = 0`. Kök: `stepGridDomains` sonundaki analiz
> geçişi sayacı HER domain için sıfırlayıp yoğunluğu yalnız gazda yeniden
> sayıyordu ("fluid domains don't touch grid.density"). Fog kapısı
> `active_density_cells > 0` istediği için kendini kapattı. Sınırlar
> (`active_density_min/max`) da sıvıda hiç dolmuyordu. Düzeltme: tarama sıvıya
> da açıldı (`ParticleSimulation.cpp`, analiz geçişi). Shader değişikliği YOK.

1. **Sayaç.** Fog modunda bir kare ilerlet: `fluid.get` →
   `active_density_cells > 0`, `max_density` ≈ 1 civarı (hücre başına
   parçacık/ppc). *Hâlâ 0 ise:* exe eski, ya da sim o karede adım atmadı
   (önbellekten geldi) — Play ile birkaç adım at.
2. **Log.** SceneLog'da `[VolumeGate 0] ... route=VolumeFog` satırı artık
   `RENDERABLE` ve `active_cells>0` demeli. *NOT renderable + active_cells>0:*
   kapıda başka bir koşul var, bana log'u getir.
3. Sonra aşağıdaki bölümün 1–7. maddeleri (hepsi bu sayaca bağlıydı).

## Sıvı için Volumetric Fog modu + VDB paneli domain'e yazıyor (2026-09-27, 3. parti)

> Kök neden: VDB panelindeki "Fog ↔ Refractive" combo'su domain'in her karede
> yeniden ürettiği hacme yazıyordu; köprü onu domain'den geri yazıyordu
> (`render_as_isosurface = fluid_surface_route`). Sıvı yoğunluk üreticisi zaten
> her adımda çalışıyor (`sim_fluid_density_splat`), yani Ağustos'taki "sıvı
> yoğunluk splat'lemez" gerekçesi geçersizdi. Yeni kayıt değeri `VolumeFog = 4`;
> kayıtlı `Volume = 0` anlamını KORUYOR (sıvıda → SDF), eski sahneler değişmez.
> Otomatik test: `python scripts\test\rt_test_fluid_fog_mode_ipc.py "Grid Domain 1"`.

1. **Script.** `fluid.set_param render_mode=fog` → `fluid.get` `"fog"`;
   `active_density_cells > 0`. *0 ise:* üretici sorunu, render değil.
2. **★ Asıl hata: kare değişince düşme.** Script'in 3. adımı: üç kare boyunca
   mod `fog` kalır. Elle: VDB panelinde domain hacmini seç — "Owned by liquid
   domain …" satırı görünmeli; Fog seç, timeline'ı sür: sis kalmalı.
   *Düşüyorsa:* panel hâlâ hacme yazıyor (`fluidDomainOwningVolume` null
   dönüyor) ya da başka bir yol domain modunu geri yazıyor.
3. **Panel.** Liquid Display'de üçüncü seçenek "Volumetric Fog / Gas". Seçince
   "Volume Material (Fog)" bölümü açılır; slider'lar sisi değiştirir.
   "Now drawing: Fog volume: untagged".
4. **Görsel: Material ve Rendered.** Sıvı sis olarak görünür, domain shader'ı
   (yoğunluk/saçılma/emilim) ile. İlk geçişte `Liquid NanoVDB Preview` preset'i
   uygulanır (yoğunluk ×50, mavi emilim).
5. **Eski sahne.** Sıvı için `Volume` (0) kaydedilmiş eski bir .rtp: SDF açılmalı,
   sis DEĞİL. *Sis açılıyorsa:* dekoder 0'ı yanlış çözüyor.
6. **★ SİNSİ: SDF override + fog.** Bir maddeye SurfaceSDF override ver, domain
   fog'da kalsın: sis ÇİZİLMEZ (tek domain hacmi yüzeye gider). Panel bunu
   turuncu uyarıyla söylemeli; `fluid.get` `effective_representation` "sdf".
   Uyarı yoksa kullanıcı "sis bozuk" sanır.
7. **VDB IOR/roughness/foam.** Domain hacminde bu üçü artık domain alanlarını
   yazar; kare değişince korunur.

## Splat havuzu raster'da, foliage LOD'undan muaf; Virtual Particles söküldü (2026-09-27)

> Kök nedenler canlı sahnede ölçüldü (88k su, granüler kapalı), düzeltmeler
> derlenmedi. Özet: `docs/dev/GRANULAR_HEIGHTFIELD_GORUNUM.md` "Neden söküldü".
> Otomatik test: `python scripts	est
t_test_fluid_splat_raster_ipc.py "Grid Domain 1"`
> (domain'de parçacık olmalı; mod/subdiv/shading'i geri yükler).

Sıra: bağımsız ve hızlı olanlar önce; 5–7 birbirini maskeler, 1–4 geçmeden bakma.

1. **Derleme.** Beş `GranularVirtual*` dosyası silindi ve vcxproj/filters'tan
   çıkarıldı. *Bozuksa:* "cannot open GranularVirtual…" = eski bir include
   kalmış; bana satırı söyle.
2. **Eski proje açılışı.** `virtual_particles` ile kaydedilmiş bir projeyi
   aç (şu anki sahne öyle). `fluid.get` → `render_mode: "particles"`.
   *Bozuksa:* `"volume"`/`"surface"` = değer 3, `fluidRenderModeFromStored`
   üzerinden okunmuyor (ProjectManager / SceneSerializer).
3. **Script reddi.** `fluid.set_param render_mode=virtual_particles` hata
   döner, mod değişmez. `fluid.set_splat_geometry subdivisions=4` reddedilir.
   *Bozuksa:* kabul ediyorsa eski exe (zaman damgasına bak).
4. **Panel.** Liquid Display combo'sunda iki seçenek var. Splat Spheres →
   "Splat Geometry & Preview" açılır; Smooth Surface → "Surface SDF Settings".
   Yeni oluşturulan domain'de Sphere Subdivision Detail = 0.
   *Bozuksa:* Splat seçiliyken SDF ayarları açılıyorsa `current_mode_idx`
   eşlemesi kaymış (eskiden Virtual seçilince tam olarak bu oluyordu).
5. **Solid: impostor gerçekten çiziyor mu.** Test script'inin 3. adımı.
   `viewport.frame_telemetry` → `raster_sphere_groups ≥ 1`,
   `sphere_impostors_uploaded > 0`, `sphere_impostors_drawn == uploaded`.
   *Bozuksa:* uploaded>0 ama drawn=0 → çizim kapısı kapalı (pipeline/buffer/
   matcapDescSet/mod), `recordParticleBillboards`. groups≥1 ama uploaded=0
   → havuzda görünür parçacık yok ya da builder yüklemiyor.
   ✔ **Sayaçlar kökü buldu (2026-09-27, 2. derleme):** uploaded=56666,
   drawn=0, ready=true, Solid ve Matcap'te aynı. Kapalı kapı `matcapDescSet`:
   teardown onu `keepPipeline`'dan bağımsız yok ediyordu, yani **ilk panel/
   pencere yeniden boyutlandırmasından sonra** set kalıcı NULL; yalnız
   pipeline sıfırdan kurulurken yeniden yaratılıyordu. Solid geometri set'i
   `!= NULL` korumasıyla bağlayıp çizmeye devam ettiği için hiçbir şey bozuk
   görünmüyordu. Düzeltme her iki backend'de (viewport + base) teardown'ı
   `!keepPipeline` arkasına aldı. **Doğrulama:** Solid'de splat'ler görünür;
   sonra bir paneli sürükleyip viewport'u YENİDEN BOYUTLANDIR — hâlâ
   görünür olmalı ve `sphere_impostors_drawn == uploaded`. Yan etki olarak
   kullanıcının yüklediği matcap dokusu da resize sonrası artık kaybolmamalı.
6. **Material / RayFusion: havuz raster listesinde.** Script'in 4. adımı:
   `full_instances > 1`, `proxy_instances == 0`. Sonra sim'i oynat.
   ★ `total_instances`'a BAKMA: boş havuz slotları artık maskeli olarak listede
   duruyor, yani o sayı slot sayar, parçacık değil — boş havuzda da büyük çıkar.
   *Görmen gereken:* küreler hep görünür, kart/düz şerit yok, oynatırken
   titreme yok. *Bozuksa:* full_instances=1 → havuz çizilmiyor;
   log'da `buildRasterGeometry early-out` + `stampedBy` satırına bak.
7. **★ SİNSİ: yeni doğan parçacıklar.** Boş domain'den emitter ile başlat,
   Material modunda oynat. Akan su **kesintisiz** görünmeli. Eski kusur
   tam olarak "seyrek, delikli akış" gibi görünüyordu — kimse bunu bug diye
   raporlamaz, emisyon ayarı sanılır. Havuz kademesi büyümeden (aynı kapasitede)
   dolan slotlar artık `mask` ile açılıyor; delik görürsen bu madde bozuk.
8. **Foliage regresyonu.** Yoğun foliage sahnesi, Auto preset:
   `proxy_instances > 0` hâlâ olmalı. *Bozuksa:* muafiyet foliage'a da
   sızmış (`group.transient` yanlış true) — kare maliyeti patlar.
9. **Boş havuz maliyeti.** Cache temizle (0 parçacık, havuz dolu kalır),
   Material modunda `frame_ms` ve `visible_triangles`. Havuz slotları artık
   raster'da duruyor ama `mask=0` → GPU cull atlar. Eski ölçüm: 1M ölü slot
   = 101,9 ms. *Görmen gereken:* visible_triangles ≈ sahnenin kendisi,
   frame_ms normal. *Bozuksa:* ölü slotlar çiziliyor → cull bounds `w<0`
   yazılmıyor (`writeRasterInstanceBound`).
10. **Vulkan RT (Rendered) değişmedi.** Aynı sahnede RT'ye geç: splat'ler
    görünür, hız önceki gibi. subdiv 0'ın (20 üçgen) görünümü yakın planda
    kabul edilebilir mi — değilse Splat Geometry'den 1'e çek; bu yalnız
    default değişikliği.
11. **Flat kaynaklı scatter senkronu.** Flat SoA kaynaklı bir foliage grubu
    olan sahnede bir instance'ı gizmo ile taşı: yerinde kalmalı. Senkron
    artık `scatterSourceTransform`'u uyguluyor; eskiden ilk senkronda
    merkezleme ofseti kadar kayıyordu. *Bozuksa:* obje taşıma anında zıplar.
12. **1M boş slot maliyeti (ÖLÇÜLDÜ, düzeltme derlenmedi).** Canlı ~520k sabitken
    havuz 524k → 1M olunca Material karesi (sim oynarken) **49 → 91 ms** çıktı.
    Boş slotlar ücretsiz değilmiş. Düzeltme: transform senkronu boş kalan
    slotu matris kurmadan atlıyor. *Görmen gereken:* aynı deneyde fark
    ≲5 ms. *Hâlâ ~40 ms ise:* maliyet CPU senkronu değil GPU/yükleme tarafı
    (`uploadRasterInstanceBuffer` 1M matris yazıyor) — o zaman dirty aralık
    yüklemesi gerekir. Deney: domain `surface`→`particles` (havuzu sıfırlar),
    ~507k tohum, Material, timeline'ı kare kare ilerlet, `frame_ms` izle.
13. **TDR (591k splat, Material→RT) TEKRARLANAMADI.** IPC ile denenenler, hepsi
    temiz: RT 65k/135k/270k/540k; Material→RT 540k; RT'de ve Material'da
    kare kare oynatma; Material'da havuz büyümesi (524k→1M) + hemen RT.
    Sürücü kaydı: `nvlddmkm` olay 153 (TDR zaman aşımı). Log'da kayıp,
    RT "yield" satırından ÖNCE gözlendi. Denenemeyen tek koşul **sürekli
    Play** (IPC'de play yok; set_frame kare kare adım atıyor). *Tekrar
    ederse:* SceneLog'u koru ve hangi modda Play'e bastığını not et.

## Vulkan RT: çok splat varken oynatmada yavaşlama — TLAS refit + orijindeki ölü slotlar (2026-09-26)

> ✔ **Kullanıcı doğruladı (derlendi):** RT oynatma yavaşlaması kayboldu. Havuz
> kademesi büyürken tam kurulum kısa bir takılma yapıyor (1M'de kabul edilebilir).
> ⚠ Aynı derlemede 1M splat ile Solid→Rendered'da **yeni bir TDR** — sessiz
> render cihazı ölümü, bkz. `GRANULAR_HEIGHTFIELD_GORUNUM.md` "AÇIK" bölümü.

> Kullanıcı gözlemi: RT'de play sırasında (havuz büyüyüp instance sayısı
> değiştikten sonra) render çok yavaşlıyor, Solid→Rendered yapınca hemen
> hızlanıyor. İki kök, ikisi de derlenmedi:
>
> 1. **TLAS hiç yeniden kurulmuyordu, yalnız refit ediliyordu** (`createTLAS`
>    + GPU `recordGpuTLASUpdate`, sayı değişmedikçe `MODE_UPDATE`). Refit ilk
>    kurulumun hiyerarşisini tutar, kutuları büyütür. Havuz kademesi büyüyünce
>    tam kurulum olur, o anda slotların çoğu ölü; sonraki karelerde onlar domain'de
>    doğdukça ağaç bozulur. Solid→Rendered tek taze kurulumdu. Artık iki yol da
>    **aynı nesneye yerinde `MODE_BUILD`** yapıyor: handle/adres/descriptor
>    değişmez, destroy yok (TDR riski eklemez).
> 2. **Ölü slotlar `identity + mask 0` idi** = orijinde birim küre, hepsi üst
>    üste, domain de orijinde. Mask gölgelemeyi eler ama BVH özdeş kutuları
>    ayıramaz: orijinden geçen her ışın o yaprakları tek tek ziyaret eder. Artık
>    `parkedTLASInstanceTransform()` (1 mm, y=-10 km) — CPU iki yol + GPU
>    `instance_prepare.comp`. Kostik hedef kutusu mask 0'ı atlıyor (eskiden
>    ölü slotları orijinde sayıyordu).
>
> ★ Kare atlamalı IPC A/B (build@20 → refit@143) fark göstermedi (~31 ms/örnek
> her iki kolda) — çünkü atlama havuz kapasitesini değiştirip zaten tam
> kurulum yaptırıyor. Kullanıcının koşulu **canlı play**; o IPC'den sürülemiyor.
> Ayrıca: `viewport.status.ms_per_sample` her durumda 0 dönüyor (ölü alan).

1. **`instance_prepare.comp` → `.spv` derlendi mi.** Shader değişti. Eski spv
   ile GPU yolu ölü slotları hâlâ orijine koyar; CPU yolu park eder. Bozuksa:
   yavaşlama yalnızca GPU scatter güncellemesi olan karelerde sürer.
2. **Aynı senaryo: RT'de play, havuz birkaç kademe büyüsün.** Beklenen: hız
   oynatma boyunca sabit kalır, Solid→Rendered artık fark yaratmaz. Bozuksa
   (hâlâ Solid'e geçip dönünce hızlanıyor): üçüncü bir kaynak var — BLAS
   (splat icosphere değil, foam/sphere GAS) veya volume tarafı.
3. **Görsel: splat'lar, köpük, cam/su kostikleri doğru yerde.** Bozuksa
   (orijinde ya da -10 km yönünde bir artefakt): park dönüşümü mask 0 olmadan
   bir yere sızıyor.
4. **Maliyet: sahne hareketsizken (pause) örnek hızı eskisi kadar.** Her kare
   BUILD, refit'ten pahalı; ama hareketsiz karede TLAS hiç çağrılmamalı.
   ★ Sessiz başarısızlık: pause'da da kare süresi birkaç ms yüksekse biri
   TLAS'ı değişiklik olmadan her kare güncelliyor demektir — bu düzeltmeden
   önce de öyleydi, sadece refit ucuz olduğu için görünmüyordu.
5. **TDR yok.** RT'de play + pause + Solid↔Rendered birkaç tur. Yerinde
   BUILD da bir yazma; önceki refit ile aynı senkronizasyonu kullanıyor.

> OptiX IAS'ında aynı refit deseni duruyor (`OptixAccelManager.cpp` ~1850);
> OptiX dondurulduğu için dokunulmadı.
>
> Ölçüm: `render.fluid.splat_instances` / `render.fluid.foam_instances` perf
> bölümleri eklendi (splat köprüsünün CPU maliyeti, granüler gösterim kararı
> için).

## Fluid Particles modu: çift çizim (mavi diskler) söküldü (2026-09-26)

> Particles modundaki fluid domain'leri hem render bridge'in küre
> instance'larıyla hem de `ParticleBillboardBuilder::appendGridDomainParticles`
> ile (domain rengi, yani mavi, billboard) iki kez çiziliyordu. Billboard'lar
> canlı konumu okuyor, instance'lar bir adım geriden geliyordu. Oynatırken her
> kürenin önüne mavi bir disk çıkıyor, duraklatınca disk kürenin içinde
> kayboluyordu. IPC ekran görüntüsüyle yeniden üretildi: duraklatılmışta yok,
> adım sırasında var. Billboard kopyası söküldü.

1. Particles modunda bir su sütunu oynat. Mavi halka/disk **olmamalı**,
   küreler beyaz/materyal renginde kalmalı. Hâlâ varsa başka bir yol daha
   çiziyordur (overlay'in debug noktaları: `particle_display_mode`).
2. Emitter'lı bir particle sistemi (fluid'siz) billboard olarak görünmeye
   devam etmeli; o yol değişmedi.

## Sim compute: kernel tablosu binding sayısı ≠ shader (validation ile bulundu)

> `RAYTROPHI_VK_VALIDATION=1` açılışta `VUID-VkComputePipelineCreateInfo-layout-07988`
> bastı. Uyuşmayan iki kernel:
> - `sim_fluid_granular_stress_update`: tablo 14, shader ve dispatch 15
>   (`bond_scale`);
> - `sim_fluid_surface_combustion`: tablo 6, shader ve dispatch 9.
>
> Pipeline layout eksik binding'le kurulup üstüne daha büyük bir descriptor
> set bağlanıyordu. Bu tanımsız davranış; NVIDIA pratikte tolere ettiği için
> kum testi çalışıyordu. Tablo düzeltildi. `dispatch` artık sayı uyuşmazlığında
> reddediyor ve kernel adını `[SimCompute] ... dispatch refused` diye bir kez
> log'a yazıyor.
> Kalıcı denetim: `python scripts/audit_sim_kernel_bindings.py` (0 uyuşmazlık).

1. Validation açıkken başlat: `07988` artık **hiç** görünmemeli.
2. Kum testi (granüler) ve yanan sıvı (yüzey yanması) önceki gibi çalışmalı.
   Log'da `dispatch refused` satırı varsa başka bir kernel'in çağrı yeri
   tablodan ayrışmış demektir; satır kernel'in adını söyler.

## TDR: sim çalışırken RT'ye geçiş — KÖK BULUNDU, düzeltme derlenmedi (2026-09-26)

> Yeniden üretim: sim canlı moddayken, parçacık havuzu büyürken RT'ye geçiş.
> Validation açıkken yakalandı; device lost'tan önce **hiç validation hatası
> yok**. Yani sorun API kullanımı değil, zamanlama.
>
> Üç çöküşteki ortak dizi:
> `RayFusion ... yielded` → `SWITCHING to Vulkan RT` → `RayFusion scene AS built … 11 KB`
> → device lost. Viewport'un RayFusion build'leri hep 4 KB; 11 KB'lık olanlar
> başka bir cihazın, yani **render backend'in** build'i.
>
> Kök: `scene_ui_procamera.cpp` ve `scene_ui_selection.cpp`'deki "AS ısıtıcıları"
> `ensureRayFusionSceneAS`'i `ctx.backend_ptr` (render backend) üzerinde de
> çağırıyordu. RayFusion TLAS'ı cihazın TEK TLAS slotuna kurar
> (`m_device->createTLAS`). Render backend'de bu slot path tracer'ın TLAS'ıdır.
> - Parçacık havuzu büyüyünce instance imzası değişiyor ve yeniden kurulum
>   tetikleniyor.
> - Bu RT'ye geçiş karesine denk gelirse path tracer'ın TLAS'ı, uçuştaki
>   trace'ler onu kullanırken 65.536 instance'lık, farklı sıralı bir TLAS ile
>   değiştiriliyor.
> - `drainInteractiveViewportInFlight` yalnız viewport karelerini bekliyor,
>   trace slotlarını beklemiyor.
>
> Düzeltme:
> - Isıtıcılar yalnız `g_viewport_backend`'i ısıtıyor.
> - Build log'u artık `device=` yazıyor.
>
> Ek olarak yan bulgu: iki sim kernel'inin binding sayısı, ayrı bölümde.

1. **Tanıyı doğrula:** RT'ye her geçişte log'da `[RayFusion] scene AS built`
   satırı varsa `device=` değeri viewport cihazınınki olmalı. Render cihazında
   bir build görünüyorsa başka bir çağıran kalmış demektir.
2. **Tekrar dene:** canlı mod, çok parçacık, havuz büyürken RT'ye birkaç kez
   geç (validation açık kalabilir). Device lost **olmamalı**.
3. ★ Sinsi başarısızlık: TDR biter ama Solid/MaterialPreview'da gölge ya da
   RayFusion etkisi kaybolur. O zaman viewport ısıtması başka bir sebeple
   render backend'e bağımlıymış; bunu `render.probe` ile ölç.
4. Açık kalan yapısal risk: RayFusion ile path tracer'ın aynı cihazda aynı
   TLAS slotunu paylaşması. Tek backend'li kurulumda yalnız mod kapısı
   (`m_viewportMode != Rendered`) koruyor.

## Animasyon: timeline'a bağlı graph zamanı + döngüyü açan root motion (2026-09-26)

> Teşhis (yürüyen Mixamo karakteri, 34 karelik döngü):
> - Graph klipleri timeline'ı DEĞİL biriken delta'yı izliyordu
>   (`graph_follows_timeline=true` iken bile). Kare atlayınca poz donuyordu,
>   geri sarınca aynı kare farklı poz veriyordu.
> - Root motion, kök kemiğin ötelemesini TAMAMEN sıfırlıyordu; kalça
>   yüksekliği de sıfırlandığı için karakter yere gömülüyordu. Yatay hareket
>   de her karede nesnenin transform'una `position += delta` diye kalıcı
>   yazılıyordu, bu yüzden geri sarınca ve oynatma yolu değişince karakter
>   son kaldığı konumdan başlıyordu.
>
> Yeni model (`Animation/RootMotionUnroll`):
> - Klip zamanı = (kare − başlangıç) / fps; graph timeline'ı izliyorsa bu
>   mutlak zaman kullanılır.
> - Root motion = kök kemiğe `tamamlanan döngü × döngü başına yol` eklenir.
>   Yol = son konum anahtarı − ilk konum anahtarı; kemiğin ebeveyn uzayında
>   eklendiği için iskelet ölçeği hiyerarşide kendiliğinden uygulanır.
> - Kemik sıfırlanmaz, transform'a yazılmaz, nesnenin konum anahtarları
>   uygulanır.
>
> Sökülenler: `RootMotionDelta`, iki çıkarım yolu, iki transform itme bloğu,
> blend düğümlerindeki root-motion lerp'leri, UI'nin "root motion açıksa konum
> anahtarını atla" kuralı.
> Yeni IPC: `anim.set_root_motion {character, enabled, bone?}`. `anim.character`
> artık `root_motion_resolved_bone`, `root_motion_cycle_travel` ve
> `root_motion_travel_valid` döndürüyor.
> Yeni dosyalar: `RootMotionUnroll.h/.cpp` (vcxproj + filters'a eklendi).

1. **Derleme.** `AnimationController::getAnimatedGlobalTransform` imzası
   değişti (döngü sayaçları). `RootMotionDelta`'yı kullanan başka bir yer
   kalmışsa derleyici söyler; grep temiz çıktı.
2. **Root motion kapalıyken** (varsayılan) karakter eskisi gibi yerinde
   döngü atmalı, döngü sonunda geri sıçramalı. Tek fark: graph timeline'ı
   izliyorsa artık kare atlamaları doğru poza gidiyor. Kapalıyken bir şey
   değiştiyse, zaman kaynağı sırası bozulmuştur.
3. **Probe:** test sahnesini aç, sonra
   `python scripts/test/rt_probe_root_motion_timeline_ipc.py 1 "1 Kinematic Preview" 110`.
   Root motion'ı AÇIK bırakır. Görmen gereken:
   - `bone:` boş olmayan bir kemik, `cycle travel` sıfır olmayan bir vektör ve
     `valid: True`;
   - `scrub` satırında her kare tek bir konum;
   - `walk` satırında ≥ 3 m yatay yol, en büyük kare adımı ~0,05–0,15 m
     (eski döngü sıçraması 1,6 m idi);
   - `rewind offset ≈ 0`, sonunda PASS.
   `valid: False` ise otomatik seçilen kemiğin (Armature/RootNode) konum
   anahtarı yok: `anim.set_root_motion character=1 enabled=true
   bone=<Hips'in tam adı>` ile sabitle.
4. **Görsel:** oynat. Karakter domain boyunca kesintisiz yürümeli ve y'de
   gömülmemeli. Durdurup 0'a sarınca başa dönmeli; sahnenin kayıtlı
   transform'u değişmemeli. ★ Sinsi başarısızlık: karakter yürüyor ama
   bir noktada viewport'tan kayboluyor. Nesnenin sınır kutusu hâlâ başlangıç
   yerinde kaldığı için culling ediliyor olabilir. Bunu yalnız göz görür.
5. **Kum/su ile:** `rt_probe_kinematic_foot_stamps_ipc.py "1 Kinematic Preview" 110`.
   `max` solver speed artık 36–40 m/s sıçrama göstermemeli (döngü ışınlanması
   bitti); ayak başına sıfır hücreli kare olmamalı.
6. **Kalan bilinen eksik:** state machine ve blend geçişleri hâlâ delta ile
   ilerliyor; timeline'a bağlı değiller. Klip değişince (`play`) döngü sayacı
   sıfırlanır ve karakter o klibin başlangıç konumuna döner.

## Kinematic collider: ayak kutusu mesh'ten + çözücü damga telemetrisi (2026-09-26)

> Teşhis (yürüyen karakter, ~1 voxel su, 5,9 cm voxel): ayak kutusu **bileğe**
> ortalanmış sabit 3,75×2,5×6,25 cm idi; basan ayakta alt yüzü y≈0,07'de,
> suyun ancak üstüne değiyordu. Ayaklar karede 1–2 hücre damgalıyordu.
> `autoFit` içindeki "bilek→parmak kutusu" dalı **ölü koddu**: `isBodyAnchor`
> foot/toe/head'i daha önce yakalıyor. Değişiklikler:
> - `collectKinematicJointPoses` her joint için baskın olduğu (ağırlık ≥ 0,5)
>   rest-pose skin köşelerinin kemik-yerel sınırını ölçer; `fitLeaf` foot/toe
>   kutusunu bu sınırdan kurar (topuk+taban+parmak). Köşesi olmayan joint eski
>   sabit kutuya düşer. Ölü foot ve head dalları söküldü.
> - Yeni `physics.collider.proxy_set.solver_stamps` (+ Python
>   `proxy_set.solver_stamps()`): son grid adımında proxy×domain başına
>   `stamped_cells` ve çözücünün `solid_vel`'e yazdığı hız. Sayım zaten dönen
>   damga döngüsünde bir artırım; ek geçiş yok.
> - `proxy_set.sample` notuna "hız OKUMA anındaki geçmişe göre, çözücününki
>   değil" uyarısı eklendi.

1. **Derleme.** Yeni dosya yok; `KinematicColliderVoxelizer.h` artık
   `ParticleSimulation.h`'ten include ediliyor. `FluidGrid` class/struct ileri
   bildirimi C4099 uyarısı verirse zararsız (önceden de vardı).
2. **Telemetri tek başına** (refit ETMEDEN, önce eski kutuyla ölç):
   `python scripts/test/rt_probe_kinematic_foot_stamps_ipc.py "1 Kinematic Preview" 24`.
   Görmen gereken: her karede dört ayak satırı, `frames stepped 24/24`, ayak
   hücreleri ortalama ~1–2,5 (önceki elle ölçümle aynı). Bu, sayacın doğru
   saydığının kanıtı. `steps` artmıyorsa log hiç dolmuyor — çağrı noktası
   bağlanmamış.
3. **Refit** — aynı komut `--refit` ile (set'in proxy'lerini YENİDEN ÜRETİR).
   Görmen gereken: foot/toe kutularının `local_position`'ı sıfır değil (Mixamo'da
   kemik ekseninde ileri ve aşağı), `half_extents` ayağın gerçek boyu kadar
   (ayak ~10+ cm uzunluk). Ardından basan ayakta **hücre sayısı belirgin artar**
   ve `zero-frames` düşer. Hâlâ ~1–2 ise: kutu yanlış uzayda (placement /
   globalInverse zinciri) — `local_position`'ı viewport overlay'de gözle kontrol et.
4. **Viewport overlay** (kinematic preview açık): ayak kutuları topuktan
   parmak ucuna uzanmalı ve tabana oturmalı. ★ Sinsi başarısızlık: kutu doğru
   BOYUTTA ama bilekte ya da ayağın üstünde — telemetri hücre sayısını artmış
   gösterir ama su yine yalnız üstten ezilir. Bunu yalnız overlay gösterir.
5. **Görsel**: splat ve granüler testinde ezilme/dağılma. Su 1 voxel
   derinliğindeyse etki yine sınırlı kalır — domain'i yürüme alanına indirip
   voxel'i küçültmek (sahne ayarı) ayrı bir kol.

## Particle Faz 1.5 Batch B: cihazda kalan balistik adım + vertex pulling (2026-09-26)

> Parçacık ajanının partisi. Tasarım: `PARTICLE_SYSTEM_GPU_ROADMAP.md` Faz 1.5
> "Batch B design". Faz 0'ın "kısmi GPU" yolu (her adım tüm SoA upload +
> sync + hız indirme) **silindi**; sistem ya tamamen cihazda ya tamamen CPU
> referansında koşar. `particle.stats` alanları yeniden adlandırıldı:
> `gpu_force_status`→`gpu_status` (`gpu_resident`), `forces_on_gpu`→
> `device_resident`, `force_*`/`mirror_*`/`upload_ms` yok; yeni: `residency`,
> `step_*`, `snapshot_*` (kümülatif), `nonfinite_measured`.
> Yeni dosyalar: `ParticleDeviceResidency.cpp`, `sim_particle_ballistic.comp`,
> `sim_particle_spawn.comp`, `particle_viewport_pull.vert`,
> `include/particle_appearance_lut.glsl`. Silinen: `sim_particle_force_integrate.*`.

1. **Shader + log.** `compile_shaders.bat` (12:07'de zaten koşmuş, üç yeni
   `.spv` var). Açılışta konsolda `particle_viewport_pull.spv missing`
   **olmamalı**. Varsa: pull pipeline yok, cihazdaki her sistem viewport'ta
   görünmez (sim çalışır).
2. **IPC testi** `python scripts/ipc_test_client.py` — `phase1.5B:` satırları.
   Kritik olan: `CPU and Auto trajectories agree` (|dp| ≤ 1e-4 m, 16 adım).
   Büyükse ballistic kernel ile CPU yolunun işlem sırası ayrışmış.
   `Auto did not go resident (...)` basıyorsa sebep parantezde — makinede
   Vulkan compute yoksa beklenen, varsa hata.
3. **`rt_api_smoke_test.py`** (Python yüzeyi aynı alanlar).
4. **Panel**: Physics sekmesi → `Last step: GPU (gpu_resident), Vulkan ...,
   state on device`. Collider ekle → `CPU (host_consumer_colliders)`, kaldır →
   tekrar GPU. **Parçacıklar geçişte sıçramamalı** (cihaz durumu `cpu_path`
   snapshot'ıyla eve geliyor).
5. **Viewport görsel (vertex pulling)**: Campfire/kıvılcım, Auto. Billboard'lar
   görünür ve **hareket eder**; tek sistemde additive alev + alpha duman
   ikisi de çizilir (row lookup'taki blend biti).
   ★ **En sinsi hal: parçacıklar görünür ama doğdukları yerde DONUK.** Bu,
   builder'ın CPU quad yolundan bayat host pozisyonlarını çizdiği anlamına
   gelir (pull edilemedi ve snapshot da alınmadı). `particle.stats` →
   `snapshot_last_reason` = `foreign_device` ve her karede artıyorsa: sim ile
   viewport **farklı VkDevice** (`render.volume_tables` ile teyit) — o zaman
   donukluk bir hatadır; artmıyorsa `residentDrawBuffers` false dönüyor.
6. **Zaman çizelgesi oynat + geri sar**: oynarken `snapshot_mirrored_steps`
   her karede artmalı, `snapshot_sync_count` yalnızca ilk karede bir kez
   artmalı (sonrasında önbellek talebi adımın kendi fence'ine biniyor). Her
   karede artıyorsa talep→ayna mekanizması çalışmıyor: oynatma eski kısmi
   yol kadar pahalı. Geri sarınca doğru parçacıklar görünmeli
   (restore → `residency host` → tam upload).
7. **Politika geçişi oynarken**: Auto → CPU → Auto. Sıçrama yok. `step_blocked`
   yalnızca GPU Required + collider/self collision'da (sebep `gpu_status`'ta).
8. **Render in Raytrace açık** sistem: RT instance'ları parçacıkların GÜNCEL
   yerinde (`snapshot_last_reason` `raytrace_instances`). Debug display
   (nokta) modu: noktalar hareket eder (`debug_dots`).
9. **Kapı ölçümü** `.\scripts\ipc\Probe-ParticleBaseline.ps1 -Scenarios
   ballistic -Counts 1024,8192,32768` (timeline durmuş, sahnede başka
   render yükü yokken). Auto satırları `step[gpu_resident]`, `down 0 B`.
   **Faz 5 kapısı: 32k resident < 1,18 ms (Faz 0 CPU).** `down` > 0 ise
   betik uyarır: bir tüketici her kare host durumu çekiyor (madde 5/8).
   plane/self_collision satırları bilerek `host_consumer_*` (Faz 6).

## Particle Faz 1.5 Batch A: görünüm profili + GPU LUT, eski start/end yolu söküldü (2026-09-26)

> Parçacık ajanının partisi; kinematic collider bölümü (dosyanın sonunda)
> diğer ajanın, ikisi aynı derlemeye girer ve birbirinden bağımsızdır.
> Plan ve sözleşme değişiklikleri: `PARTICLE_SYSTEM_GPU_ROADMAP.md` → Phase 1.5.

**Ne değişti:** parçacık başına renk/boyut/opaklık artık SoA'da tutulmuyor ve
CPU her adımda lerp yapmıyor. Emitter bir `appearance_profile_id` taşıyor.
Profil 64 örneklik bir LUT'a pişiriliyor ve raster shader, RT instance boyutu,
gaz deposit ağırlığı ile debug noktaları AYNI LUT'u okuyor. Billboard blend'i
sistemden profile taşındı. Eski projeler yüklenirken iki anahtarlı profile
çevriliyor.
Yeni dosyalar (vcxproj'a eklendi): `ParticleAppearanceProfile.h/.cpp`,
`ParticleSimulationAppearance.cpp`, `UI/ParticleBillboardBuilder.h/.cpp`,
`UI/ParticleAppearanceUI.h/.cpp`, `Viewport/ParticleBillboardData.h`,
`Backend/VulkanViewportParticles.cpp`. Silinen: `scene_ui_fluid_billboards.hpp`.
Shader: `particle_viewport.vert` yeniden yazıldı (frag aynı).

> ✔ 2026-09-26 (11:06 exe + 11:08 spv, canlı IPC ve viewport ekran görüntüsü):
> - **1 ✔**
> - **2 ✔:** 277/277 PASS, 26 `phase1.5:` satırının hepsi OK.
> - **3 ✔:** varsayılan profil sarı→kırmızı, küçülüyor.
> - **4 ✔ (IPC ile):** additive alev ve alpha duman aynı sistemde. Duman profili
>   canlı olarak maviye + emisyon 1→3 çevrildi; LUT yeniden yüklendi ve ekrana yansıdı.
> - **5 ✔:** kaydet→aç→kaydet'te id'ler aynı ve profil çoğalmıyor; dosyada legacy
>   anahtar 0. Sentetik v2 projesinde açık değerler birebir geçti, anahtarsız
>   emitter eski varsayılanları aldı, sistem `blend_mode=1` → iki profil alpha.
> - **Not:** `render_in_raytrace` açıkken RT küreleri viewport'ta opak çizilip
>   billboard'ları örtüyor. Bu eskiden de böyleydi, bu partinin hatası değil.
> - **Kalan:** 4'ün panel tarafı (Duplicate/önizleme şeridi), 6, 7, 8, 10.

1. **Önce shader'ları derle** (`compile_shaders.bat`), sonra exe.
   `particle_viewport.spv` eski kalırsa ne olur: pipeline 8 float'lık vertex +
   128 baytlık push constant bekler, eski shader ise 9 float + 144 bayt bekler.
   Parçacıklar hiç görünmez ya da dev ve anlamsız üçgenler olarak çizilir.
   **Bozuksa:** önce `x64/Release/shaders/particle_viewport.spv`'nin zaman
   damgasına bak.
2. **IPC testi** — `python scripts/ipc_test_client.py`. Bütün `phase1.5:`
   satırları OK olmalı. Asıl sinyaller:
   - `set_emitter(start_size) → error` ve `set_system(blend_mode) → error`.
     *Bozuksa:* anahtarlar sessizce yutuluyor; eski bir script görünümü
     değiştirdiğini sanıp hiçbir şey yapmaz.
   - `profiles landed in B only`. *Bozuksa:* `system_id` etkin sisteme
     düşüyor demektir.
   Python tarafı: `rt_api_smoke_test.py` →
   `[rt-smoke] rt.particle phase-1.5 appearance profiles: OK`.
3. **Görsel eşdeğerlik (en hızlı göz testi).** Boş bir sisteme "Add Point
   Emitter" ekle, render ayarlarında emitter_only'yi kapat, oynat. Görünüm eski
   varsayılanın aynısı olmalı: sarımsı başlar, kırmızıya döner, küçülür ve söner.
   Explosion presetini de dene: çekirdek büyük ve parlak, kıvılcımlar küçük.
   **Bozuksa:**
   - Her şey beyazsa ve boyutu 1 m ise parçacıklar fallback satırına (satır 0)
     düşüyor, yani `lut_row` eşleşmiyor.
   - Boyut doğru ama renk sabitse shader LUT'u yanlış indeksliyor.
4. **Tek sistemde iki blend** (bu partinin gerekçesi). Aynı sisteme ikinci bir
   emitter ekle. Emitter > Spawning Appearance Dynamics > **Duplicate** bas,
   Blend'i Alpha yap, rengi koyu gri ver. Parlak (additive) ve koyu (alpha)
   parçacıklar aynı anda görünmeli. Profil önizleme şeridi viewport'la aynı
   renkleri göstermeli.
5. **Eski proje migrasyonu.** Bu partiden önce kaydedilmiş, parçacık emitter'lı
   bir `.rtp` aç. Görünüm aynı olmalı. Panelde her emitter için
   "<ad> Appearance" profili görünmeli. Eski projede sistem Alpha idiyse
   profiller de alpha olmalı. Sonra kaydet (yeni ada), yeniden aç: aynı
   görünüm, aynı profil id'leri. Kaydedilen `.rtp`'de `start_size` geçmemeli
   (`findstr /c:"start_size" dosya.rtp` boş dönmeli).
   *Bozuksa* (ikinci açılışta profiller çoğaldıysa): migrasyon idempotent
   değil demektir, yani emitter `appearance_profile_id` olmadan yazılmış.
6. **Viewport boyutunu değiştir** (paneli sürükle). Parçacıklar kaybolmamalı.
   Boyut değişince LUT tamponu atılıp yeniden yükleniyor. **Bozuksa:**
   descriptor stale kalmış; `writeParticleLutDescriptor` çağrılmıyor.
7. **Fluid domain, Particles render modu** (bir sıvı domain'inde
   `fluid_render_mode = Particles`). Billboard'lar domain renginde, eski
   boyutta ve alpha ile çizilmeli. Bu yol da artık aynı LUT'tan geçiyor.
8. **RT render (Vulkan RT) parçacık instance'ları.** Parçacıklar ömürleri
   boyunca küçülmeli (boyut LUT'tan geliyor). "Inherit color" açıkken malzeme
   rengi = ilk emitter profilinin doğum rengi. Profilin rengini değiştirince
   RT malzemesi yeniden kurulmalı.
9. **Bilerek yapılan davranış değişiklikleri:**
   - Ash debris artık kendi "Ash Debris" profiliyle ve **alpha** ile çiziliyor.
     Eskiden hedef sistemin blend'ini alıyordu; koyu kül additive'de
     görünmezdi.
   - Profil düzenlemesi sim cache'ini **yalnızca opaklık eğrisi değişince**
     temizler. Renk sürüklerken sim sıfırlanmamalı.
10. **★ SİNSİ OLAN — gaz kuplajı.** Kamp ateşi / ignited fuel jet sahnesinde
    oynat; `particle.stats` → `grid_deposit_landed` ve duman sütunu önceki
    derlemeyle aynı büyüklükte olmalı. Deposit ağırlığı artık opaklığı
    lerp'lenmiş SoA'dan değil LUT'tan okuyor. İki anahtarlı doğrusal profilde
    sonuç birebir aynı olmalı. Duman hafifçe zayıflamışsa kimse bunu bug diye
    raporlamaz, "kalibrasyon" sanılır. Zayıflama varsa ilk şüpheli
    `sampleAppearance`'ın gördüğü profil id'si: 0 ise fallback'in 1→0
    opaklığı kullanılıyor demektir.

## Particle Faz 0 düzeltmesi + Faz 1 ilk parti: sistem adresleme, render IPC (2026-09-25)

> Parçacık ajanının partisi. Aşağıdaki "donma/autosave" bölümü diğer ajanın,
> ikisi aynı derlemeye girer ve birbirinden bağımsızdır.

**Ölçülen (13:09 exe, canlı IPC):** Faz 0 kontrol 1 ✔ (208/208), kontrol 5 ✔
(GPURequired adımı reddetti, parçacık y=5'te kaldı). **Kontrol 6 en sinsi
sonucu verdi:** bütün Auto satırları `backend_not_vulkan` — GPU yolu hiç
koşmamıştı; tablo "1.07x hızlanma" gösteriyordu, ikisi de CPU'ydu.
Kök: `syncSimulationWorld()` compute backend'ini yalnızca **grid domain**
backend'lerinden seçiyordu; parçacık politikası seçime hiç katılmıyordu, IPC
`particle.step` de seçimi hiç çalıştırmıyordu. CPU referansı:
`docs/dev/particle_baseline_2026-09-25_cpu.json`.

Değişen: `scene_data.h` (backend seçimi + sistem id sayacı),
`ParticleSimulation.cpp` (addEmitter uid tekilliği), `RtApi.h`,
`RtApiParticle.cpp`, `RtIpc.cpp` / `RtPython.cpp` (parçacık blokları
**çıkarıldı**), **yeni** `RtIpcParticle.cpp/.h`, `RtPythonParticle.cpp/.h`
(vcxproj'a eklendi), `scene_ui_forcefield.hpp` (sistem adı alanı), üretilmiş
descriptor'lar, overlay, `ipc_test_client.py`, `rt_api_smoke_test.py`,
`Probe-ParticleBaseline.ps1`. Audit derlemeden önce geçti (602 metot).

1. **Derleme / bağlama.** *Bozuksa:* `dispatchParticleIpc` ya da
   `registerParticleBindings` çözümlenemedi → iki yeni `.cpp` vcxproj'da yok.
   `rt.particle` modülü Python'da "has no attribute" → `RtPython.cpp`'deki
   kayıt çağrısı çalışmıyor.
> ✔ 2026-09-26 (23:21 exe, canlı): 2 ✔ (bütün `phase1:` satırları OK; 4 FAIL =
> 3'ü 23:40 sonrası eklenen deposit sayaçları, 1'i sahnede Default_Cube yok),
> 3 ✔ (`force[gpu]`, down = 12×capacity), 4 ✔ (baseline alındı, ballistic
> Δ ≤ 2.4e-7 m, plane Δ 0 — ama ilk koşunun plane satırları GEÇERSİZDİ:
> bulut düzleme hiç ulaşmıyordu, probe düzeltildi ve yeniden ölçüldü).
> Kalan: 1 (derlendi ✔), 5, 6, 7, 7b, 7c, 8.

2. **IPC testi** — `python scripts/ipc_test_client.py`. Bütün `phase1:`
   satırları OK olmalı. Asıl sinyal **`phase1: writes landed in B only`**:
   *Bozuksa* (`A=2 B=0`) açık bir `system_id` sessizce aktif sisteme düşüyor —
   bu partinin düzelttiği şeyin ta kendisi. `uid survives index shift` FAIL →
   uid çözümü index'e bakıyor.
3. **★ Faz 0 kontrol 6'yı tekrarla** —
   `.\scripts\ipc\Probe-ParticleBaseline.ps1 -Counts 2048 -Scenarios ballistic -Samples 10`
   Auto satırı artık **`force[gpu]`** demeli ve `down` ≈ `12 × capacity` bayt.
   *Bozuksa:* hâlâ `backend_not_vulkan` → `syncSimulationWorld`'ün parçacık
   dalı çalışmadı (Vulkan compute oluşturulamadı ve kilit düştü — SceneLog'a
   bak). Sahnede panelden bilerek "GPU (CUDA)" seçilmiş bir domain varsa bu
   değer DOĞRUDUR: kuvvet çekirdeği yalnız Vulkan'da; probe'u boş sahnede koş.
   `no_dispatch_support` → Vulkan compute bağlamı var ama dispatch yok.
   Probe artık bu durumda sarı uyarı basar ve `speedup`'ı boş bırakır.
4. **Tam baseline** — `-OutputPath .\particle_baseline.json`, JSON'u bana
   gönder. ★ **En sinsi sonuç:** `force[gpu]` ama `ballistic`
   `max_position_delta` büyük (>1e-4 m) → GPU kuvvet çekirdeği CPU'dan farklı
   fizik hesaplıyor; hız farkı "GPU hızlı" diye okunur, oysa sonuç yanlıştır.
5. **Python smoke** — uygulama içinden `rt_api_smoke_test.py`; son satırlardan
   biri `[rt-smoke] rt.particle phase-1 system addressing + render + uid: OK`.
6. **Panel ↔ çekirdek** — Simulation > Particles > System sekmesinde combo'nun
   altında **ad alanı**. Adı değiştir, Enter → hiyerarşide yeni ad. Başka bir
   sistemin adını yaz → kırmızı "already in use" ve alan eski ada döner.
   Tersini de dene: `particle.set_system {"system_id": N, "name": "X"}` →
   panel X göstermeli. `particle.set_render {"shape":"cube"}` → panelde
   "Ray Trace Shape" Cube olmalı.
7. **Davranış değişikliği (bilerek):** domain'siz bir parçacık sahnesi Auto'da
   artık Vulkan compute kullanır. Bir kıvılcım emitter'ı ile timeline'ı 3–4
   kez Oynat/Duraklat → TDR yok (`VULKAN_PARTICLE_PRESET_PAUSE_TDR.md`'deki
   çit düzeltmesi bu yolu da kapsıyor olmalı). Küçük sayılarda kare süresi
   hafif artabilir: kısmi GPU yolu her adımda hızı indiriyor, Faz 5'e kadar
   beklenen maliyet; `particle.stats.gpu_force_ms` ile görünür.
7b. **Domain backend combo'su** (Fluid/Gas domain paneli): sıra artık CPU →
   "GPU (Vulkan Compute - Recommended)" → "GPU (CUDA - Alternative, NVIDIA)".
   Eski bir projede CUDA seçili domain hâlâ CUDA göstermeli, Vulkan seçili olan
   Vulkan. *Bozuksa* (seçim kaymış görünüyorsa): combo sırası `backend_values`
   ile eşleşmiyor — kayıtlı değer değişmez, yalnızca etiket yanlış olur.
7c. **Deposit sayaçları IPC'de** — kamp ateşi sahnesinde oynatırken
   `particle.stats` → `grid_deposit_landed` > 0 olmalı. Zemin collider'ı
   olmayan spark emitter'da `grid_deposit_dropped_no_domain` büyük çıkar:
   parçacıklar y<0'a düşüp gaz kutusundan çıkıyor (2026-09-25 canlı
   sahnede alt sınır y=-3.06 ölçüldü). *Bozuksa* (üçü de 0, oranlar > 0):
   stats son adımı değil başka bir sistemi okuyor.
8. **Kaydet/aç** — bir emitter'ın `uid`'ini oku, kaydet, projeyi yeniden aç,
   `particle.get_emitter {"emitter_uid": <uid>}` aynı emitter'ı bulmalı;
   `list_systems` id'leri aynı kalmalı.
   *Beklenen, hata değil:* `clear_systems` sonrası yeni sistemler artık
   "Particle System 1"den değil kaldığı sayıdan devam eder (id'ler tekrar
   kullanılmaz — eski id'yi tutan script yeni sisteme yazmasın diye).

## "Bazen donup kalıyor" = otomatik kayıt 122 s ana iş parçacığında (2026-09-25)

**Ölçülen (canlı, sahneyi IPC ile kare kare sürerek):** 175 karede sim en
fazla ~170 ms/kare (ilk kare 1.3 s: collider ağırlıklarının ilk kurulumu).
Donma anında `cdb` ile ana iş parçacığı yığını: `autosave::tick → writeNow →
saveProject → serializeTextures → fwrite`. `project.autosave_status`:
`last_write_ms 122793`. SceneLog: `embedded: 64, png re-encoded: 64,
previous-embed-reused: 0, bin-bytes: 1224050340`, `serializeTextures 122586 ms`.
Sim DEĞİL — her 300 sn'de bir, CPU tek çekirdekte 64 dokuyu baştan kodluyordu.
Basit sahnede görünmemesinin nedeni az doku.

**Kök (iki kat):**
1. Önceki kaydın doku baytlarını aynen kopyalayan yol, manifesti ana `.rtp`
   JSON'undan okuyordu; format 3.0'dan beri dokular `.rtp.shared`'da → indeks
   HER ZAMAN boş → her kayıt (Ctrl+S dahil) her dokuyu yeniden kodluyordu.
2. `Texture::upload_to_gpu()` yükleme bekleyen dokuyu `markVulkanDirtyFull()`
   ile işaretliyordu, o da `save_dirty` kuruyordu → açılışta hiç
   dokunulmamış dokular "kaydedilmemiş değişiklik" sayılıyordu.

**Düzeltme:** manifest `.shared`'dan okunur (yoksa eski düzen); otomatik kayıt
/ "Farklı kaydet" kaynak projenin `.bin`'ini de indekse ekler; yeniden
kullanılan blob bayt bayt kopyalanır (JPG'ye yeniden sıkıştırma denemesi —
PNG decode — kaldırıldı); `markVulkanUploadPending()` yalnız GPU bayrağını
kurar. **Otomatik kayıt artık `save_dirty`'ye DOKUNMAZ** (aşağıda 3) ve
**oynatma sırasında ertelenir**, durdurunca yazar.
Değişen: `ProjectManager.cpp/.h`, `Autosave.cpp`, `Texture.h`, `Main.cpp`.
Yeni araçlar: `scripts/test/drive_fluid_frame_timing.py` (timeline'ı IPC ile
kare kare sürer, her karenin aşama dökümü), `probe_fluid_frame_spikes.py`.

**Ek (aynı parti, kullanıcı isteği):** otomatik kayıt artık **arka planda**
(Ctrl+S gibi) ve **`ProjectManager::saveProjectCopy`** ile yazar — proje yolu,
adı, son projeler, `is_modified`, doku `save_dirty` bayrakları HİÇ değişmez
(eski "yaz sonra geri al" ana iş parçacığında doğruydu, arka planda Ctrl+S'i
autosave.rtp'ye yazdırırdı). Bütün kayıtlar tek mutex'le sıraya girer. Ayar
açılıp kapatılabilir: **File > Project Save Options > Autosave** (+ aralık),
Template Hub'da da; IPC `project.autosave_set {enabled, interval_sec}`,
Python `rt.project.autosave_set`. HUD: "Autosaving..." → "Autosaved (x s)" /
"Autosave failed: ...". Ek değişen: `Autosave.h`, `RtApi*`, `RtIpc*`,
`RtPython.cpp`, `scene_ui.cpp`, `scene_ui_menu.hpp`, `TemplateHubUI.cpp`.

0. **Arka plan + HUD:** `project.autosave_set {"interval_sec": 60}`, bir şeyi
   değiştir, 1 dk bekle (oynatma KAPALI) → HUD "Autosaving..." sonra
   "Autosaved"; o sırada viewport dönmeye devam etmeli. `enabled:false` →
   `write_count` artmamalı; menüdeki onay kutusu da kapalı görünmeli.
   *Bozuksa:* HUD yoksa `progress().finished` kenarı kaçıyor.
0b. **★ SİNSİ OLAN — kimlik:** otomatik kayıt yazarken Ctrl+S → proje KENDİ
   dosyasına kaydedilmeli (`project.path` hâlâ senin .rtp'n), son projeler
   listesinde autosave.rtp görünmemeli.
1. **Hızlı:** projeyi aç, dokunmadan Ctrl+S. SceneLog (`x64/Release/SceneLog.txt`)
   `previous-embed-reused: 64`, `png re-encoded: 0` demeli; `serializeTextures`
   saniyeler (1.2 GB kopya), 122 s değil. *Bozuksa:* reused 0 ise indeks hâlâ
   boş ("Loaded N previous embedded texture entries" satırına bak); reused
   birkaç, re-encoded çoğunluk ise dokular açılışta hâlâ `save_dirty`
   (başka bir `markVulkanDirtyFull` yükleme yolu var).
2. **Otomatik kayıt:** IPC `project.autosave_now` → `last_write_ms` saniyeler.
   Oynatırken `project.autosave_status.seconds_until_next` negatife düşer ama
   yazma olmaz; durdurunca bir kez yazar.
3. **★ SİNSİ OLAN — boyama kaybı:** bir dokuyu boya, `project.autosave_now`
   (veya 5 dk bekle), sonra Ctrl+S, projeyi yeniden aç → boyama DURMALI. Hata
   vermeden kaybolursa: otomatik kayıt `save_dirty`'yi temizlemiş ve Ctrl+S
   projenin eski `.bin`'indeki boyanmamış baytları kopyalamıştır.
4. **Aynı dosyaya aynı dosyadan kopya:** Ctrl+S, projenin kendi `.bin`'inden
   okuyup geçici dosyaya yazar, sonra yerine koyar. İki kez art arda Ctrl+S →
   ikinci de reused 64, proje açılıyor ve dokular doğru.
5. **Uçtan uca:** `python scripts/test/drive_fluid_frame_timing.py 240 250`
   (mum sahnesi açık) → ilk kare dışında 1 s'yi aşan kare olmamalı ve betik
   `dispatch timeout` ile kesilmemeli (bu partide 175. karede kesildi = kayıt).

## Ara ara donma: sim zamanlaması + collider OBB önbelleği (2026-09-25)

**Şikâyet:** önbellek düzeltmesinden sonra çok hızlandı ama "bazen donup
kalıyor, CPU bekliyor". **Ölçülen (canlı, derleme öncesi):**
`sim.fluid.solid_face_weights` genelde ~0 ms, en fazla 840 ms (önbellek ara
ara yeniden kuruluyor); `loop.ui_draw` en fazla 7.4 s, `loop.frame_tail` en
fazla 3.1 s — ama sim adımının kendisinde zamanlayıcı olmadığı için HANGİ
aşama olduğu okunamadı. Collider'lar statik; SDF oynatmada yeniden
pişirilmiyor; timeline tick başına en fazla 1 kare ilerliyor (8 adımlık
yakalama döngüsü aday değil).
**Bulunan sürekli israf:** `resolveObjectOBBForSimulation` her çağrıda mesh'in
BÜTÜN üçgenlerini kopyalayıp iki kez tarıyordu — collider başına, adımda
birkaç kez. Voxel önbelleği isabet ederken bile `voxelize_colliders` ~8 ms
buydu. **Düzeltme:** OBB, yüzey önbelleği girdisi yeniden kurulana/silinene
kadar ezberleniyor (`sim_obb_memo_`). **Yeni zamanlayıcılar:**
`sim.timeline.update` (ebeveyn), `.config_sig`, `.source_poses`, `.step`,
`.capture_frame`, `.render_sync`, `.restore_frame`, `.restore_frame_disk`;
`sim.collider.obb_resolve`, `sim.collider.surface_cache_rebuild`.
Değişen: `scene_data.h`. Yeni: `scripts/test/probe_fluid_frame_spikes.py`.

1. ✔ (ölçüldü: <0.5 ms) **Hızlı:** oynat, `sim.fluid.voxelize_colliders` artık ~0-1 ms olmalı
   (önceden ~8). *Bozuksa:* `sim.collider.surface_cache_rebuild` her adımda
   sayıyorsa yüzey önbelleği her kare yeniden kuruluyor (geometri kuşağı her
   kare artıyor demek) — OBB ezberi de onunla birlikte her kare düşer.
2. ✔ (IPC sürüşüyle ölçüldü: sim ≤170 ms/kare, donma otomatik kayıttı — üstteki bölüm) **Asıl ölçüm:** terminalden
   `python scripts/test/probe_fluid_frame_spikes.py 300 250` çalıştır, sonra
   uygulamada oynat ve donmayı bekle. Her donma aralığı için hangi scope'un
   büyüdüğünü basar. Çıktıyı bana ver. *Okuma:* `sim.timeline.step` büyük ama
   `sim.fluid.*` küçükse çözücünün zamanlanmamış kısmı (basınç/GPU bekleme);
   `render_sync` büyükse yüzey (SDF) inşası; `capture_frame` büyükse kare
   önbelleğine kopyalama; hiçbiri büyük değil ama `loop.frame` büyükse sim
   dışı bir yol (untimed satırı).
3. **★ SİNSİ OLAN:** OBB ezberi, collider'ı hareket ettirince ESKİ kutuyu
   verirse hata vermez — sıvı objenin eski yerine çarpar. Kontrol: oynatma
   durmuşken collider objesini gizmo ile kaydır, tekrar oynat → sıvı yeni
   konuma çarpmalı; collider gizmo kutusu da objeyle birlikte gitmeli.

## Donmuş mum collider önbelleğini her karede öldürüyordu (2026-09-25)

**Şikâyet:** mum sahnesi collider'a değince CPU patlıyor. **Ölçülen (canlı):**
`loop.frame` 2586 ms; çözücü aşamalarının toplamı ~60 ms (pressure 44, P2G 4,
G2P 2.4). Sahnede iki `mesh_sdf` collider: makine + masa örtüsü (domain tabanını
kaplıyor). 992/1000 parçacık donmuş, 242 katı hücre.

**Kök:** katı örtü (donmuş mum) yüzlerini collider yüz ağırlıklarının ÜSTÜNE
yazıyordu; bu yüzden örtü varken her karede `collider_weights_init = false`
yapılıyordu → her karede tam reset + her mesh collider'ın analitik
süper-örneklemesi. Donma temasla başladığı için "collider'a değince" göründü.
**Düzeltme:** örtü kapattığı yüzleri eski değerleriyle kaydediyor
(`FluidGrid::overlay_weight_restore_*`), sonraki adımın başında geri yüklüyor;
collider önbelleği hep saf collider ağırlıklarını görüyor. Ayrıca sıvı adımına
`RTPERF_FRAME_SCOPE`: `sim.fluid.voxelize_colliders`, `.thermal_cool_freeze`,
`.solid_overlay`, `.solid_face_weights`, `.thermal_viscosity_field`.
Değişen: `FluidGrid.h`, `ParticleSimulation.cpp`. Yeni dosya yok.

1. ✔ (kullanıcı: "çok hızlandı") **Aynı sahne, aynı an** — `perf.list`: donmuş mum varken
   `sim.fluid.solid_face_weights` birkaç ms olmalı (önbellek isabeti), saniye
   değil; `loop.frame` yüzlerce ms'ye inmeli. *Bozuksa:* `solid_face_weights`
   hâlâ büyükse başka bir yol önbelleği bozuyor (collider imzası her kare
   değişiyor mu — `sim.fluid.voxelize_colliders` de büyükse evet). Başka bir
   scope büyükse darboğaz oradadır; ismini bana yaz.
2. **★ SİNSİ OLAN: geri yükleme yanlışsa** — iki yönde de hatasız görünür:
   (a) collider yüzleri açık kalır → sıvı makineden/örtüden hafifçe SIZAR;
   (b) örtü yüzleri kapalı kalır → mum eridikten sonra havada görünmez bir
   duvar kalır, sıvı "hiçbir şeye" çarpar. Kontrol: mumu dök, sonra termal
   zinciri kapat (donmuşlar serbest kalır) → sıvı makineden sızmamalı ve
   eski donmuş bölgeden serbestçe akmalı.

## Termal sıvı (mum) + yüzey ayarları IPC'de (2026-09-25)

**Amaç:** fotoğraf makinesinin üstüne erimiş mum dökmek. Mum, viskoz olduğu
için değil **soğuyup donduğu** için mum gibi görünür; motorda bu yön yoktu
(ısı zinciri yalnızca erime yönündeydi, sıvı parçacıklar gaz grid'i dışında
hiç soğumuyordu, emitter parçacıkları **0 K** doğuruyordu).

Kurulan:
- `APICSolverParams::thermal_*` + yeni `Fluid/FluidThermalLiquid.h/.cpp`:
  yüzeyde havaya, collider/kapalı duvara temasta daha hızlı soğuma (Newton);
  donma noktasının üstünde log-ölçekli ν(T) rampası (substance viskozite alanı
  kanalından, CPU ve GPU aynı alanı görür); donma noktasının altında
  **desteğe değen** parçacık donar (`kParticleFlagFrozen`), sabitlenir (v=0)
  ve mevcut katı-faz örtüsüne girer → üstüne dökülen mum katman katman birikir.
  Destek şartının gerekçesi başlıkta: havada donan parçacık askıda kalırdı.
- **Wax** preset'i (sıcakta ν=5e-6, 330 K'de donar, 25 K kalınlaşma bandı).
  ★ Wax dışındaki her preset zinciri KAPATIR (Water 293 K'de doğar, 330 K'nin
  altındadır — kapanmasa su donardı).
- Flow source **Pour Temperature (K)** (anahtar: `fluid_temperature_override`
  + `fluid_temperature_kelvin`). Kapalıyken artık domain ortamında doğar;
  `seedBox` de ortamda doğurur (ikisi de 0 K yazıyordu).
- Yüzey ayarları IPC/Python'da: `surface_resolution_multiplier`,
  `kernel_radius_voxels`, `particle_radius_voxels`, `narrow_band_voxels`,
  `smoothing_iterations`, `anisotropy_*`, `position_smoothing` →
  `fluid.set_param` (aralık dışı REDDEDİLİR) ve `fluid.get` (+ ölçülen
  `surface_grid_dim` / `surface_build_ms`).
- Panel: "Thermal Liquid" bölümü + canlı ölçüm; **Heat Conduction** satırı
  (bu alan script'ten yazılabiliyordu, panelde hiç yoktu); preset combo'suna
  **Molten Plastic** (eksikti — o preset'teki domain "Custom" görünüyordu) ve
  **Wax**.
- ★ Okurken bulunan hata: `fluid.set_substance_material` GAZ lookup'ı
  kullanıyordu → sıvı domain'de yalnızca "gas domain not found" dönebilirdi.
  Sıvı lookup'ına çevrildi. **Canlıda ölçülmedi** (uygulama kapalıydı) — 1.
  maddenin 6. fazı bunu ölçer.

Değişen: `APICFluidSolver.h/.cpp`, `FluidParticles.h`, `FluidLevelSet.h/.cpp`,
`ParticleSimulation.h/.cpp`, `RtApi.h`, `RtApiFluid.cpp`, `RtApiSimNodes.cpp`,
`SimulationNodes.h`, `RtIpc.cpp`, `RtPython.cpp`, `ProjectManager.cpp`,
`SceneSerializer.cpp`, `scene_data.h` (bake imzası), `scene_ui_forcefield.hpp`,
`scene_ui_simulation_domains.cpp`, üretilmiş `RtIpcMethodDescriptors.cpp`,
overlay JSON. **Yeni `.cpp`: `Physics/Fluid/FluidThermalLiquid.cpp`** —
`.vcxproj` + `.filters`'a eklendi. Shader değişmedi. Yeni IPC metodu yok
(var olanlara anahtar eklendi) → yetki tablosu değişmedi; audit geçti.

> ✔ 1 GEÇTİ (canlı, dışarıdan): 353 K doğuyor, ortam altı yok, 30. karede 241 /
> 240. karede 4660 donmuş, ν 1.1e-5..5e-2, kapatınca 0; set_substance_material
> artık "material not found" diyor (gaz lookup hatası canlıda doğrulandı).
> ★ IPC test script'leri uygulamanın script workspace'inden ÇALIŞTIRILMAZ —
> ana thread'i tutup kendi isteğini bekler; rt_ipc artık bunu açıkça reddediyor.

1. **Otomatik test** — boş/az dolu bir sahnede
   `python scripts/test/rt_test_thermal_wax_ipc.py`. Kendi rig'ini kurar
   ("WaxProbe"), sonunda PASS/FAIL. Fazlar bağımsız sırayla:
   - 1 yüzey anahtarları gidip geliyor, 5 reddediliyor. *Bozuksa:* anahtar
     dispatch'e ya da `fluid.get`'e ulaşmadı.
   - 2 wax→water zinciri kapatıyor. *Bozuksa:* `applyPreset` sıfırlaması.
   - 3 döküm sıcaklığı gidip geliyor.
   - 4 **dök–soğut–dondur**: sıcak doğuyor (≈353 K), ortam altına inen yok,
     ortalama düşüyor, ν alanı ≥10× aralık, 4 s sonunda `frozen > 0`.
     *Bozuksa:* "colder than ambient" → hâlâ 0 K doğuran bir yol var;
     "nothing froze" + `cold_unsupported > 0` → destek testi; `frozen=0` ve
     `cold_unsupported=0` → hiç soğumuyor.
   - 5 kapatınca donmuş parçacık kalmıyor.
   - 6 `set_substance_material` sıvı domain'i kabul ediyor.

2. **Panel** — Simulation > domain > APIC Solver: preset listesinde
   Molten Plastic ve Wax var. Thermal Environment gerçek etkin ambient'i;
   Thermal Liquid bölümü freeze/release eşiklerini, `tau=1/rate`, kaynakların
   ambient/custom dağılımını ve canlı sıcaklık farkını gösteriyor. Flow source
   `Initial Temperature = Domain Ambient | Custom` kullanıyor ve her durumda
   **Effective Birth Temperature** yazıyor. Şunları ayrıca doğrula:
   - birth = ambient iken “soğuyacak sıcaklık farkı yok” uyarısı görünür;
   - freeze > ambient ve kaynak ambient iken “hemen donabilir” uyarısı görünür;
   - custom 353 K, ambient 293 K, freeze 303 K iken tahmini contact/air eşik
     süreleri görünür ve sıcaklık telemetrisi zamanla düşer;
   - art arda eklenen point/object flow source'lar benzersiz ad alır; IPC'de
     `flow_source.get/update` ile ayrı ayrı hedeflenebilir.
   *Bozuksa:* eski exe (zaman damgası) veya yeni
   `scene_ui_fluid_thermal.cpp` proje girdisi eksik.

> ✔ 3 kullanıcı manuel doğruladı (2026-09-25): donma oluyor, sıcaklıkla değişiyor.
> Katman yığılması ve yan yüzeyde sabitleme ayrıca raporlanmadı.

3. **Asıl sahne: makineye mum** — domain'i makinenin üstünü saran dar kutu
   yap, preset **Wax**, makineyi **mesh collider** olarak ekle, flow source
   Pour Temperature ≈ 350 K, oynat. Görmen gereken: dökülen mum önce akar,
   yavaşlar, makinenin üstünde ve yan yüzünde **durur**; sonraki döküm
   öncekinin üstüne yığılır. Panelde Frozen artmalı.
   ★ **SİNSİ OLAN:** Frozen sayısı artıyor ama mum yine de akıp gidiyorsa
   (`fluid.get` → `solid_phase_cells` 0 kalıyor), donmuş katman bir hücreyi
   dolduracak kadar kalın değil — örtü eşiği (`solid_phase_fill`) seed
   yoğunluğuna bağlı. Görüntü "mum yapışmıyor" der, sayılar "dondu" der; bug
   raporu olarak gelmez. Çare: `solid_phase_fill` düşür ya da çözünürlük.
   İkinci sinsi olan: yan yüzeyde donmuş katman aşağı kayıyorsa sabitleme
   (pin) GPU yolunda kaybolmuş demektir — `Fluid::step`'teki kuvvet sonrası
   sıfırlama ya da G2P geri yüklemesi.

4. **Yüzey inceliği** — aynı sahnede Surface Detail 2–3 + Anisotropic Kernel.
   `fluid.get`: `surface_grid_dim` = sim grid × çarpan, `surface_build_ms`.
   Sim maliyeti (`fluid.step_stats`) değişmemeli; yalnızca yüzey kurulumu artar.

5. **Kaydet/aç** — Wax preset'li domain + döküm sıcaklığı olan kaynak →
   kaydet, aç. `fluid.get` thermal_* ve `flow_source.get` pour alanları aynı.
   *Bozuksa, ÖNCE bak:* ProjectManager'daki "parseDomain ... fluid block"
   tripwire log satırı. O satır varsayılanları basıyorsa bu, 2026-08-16'dan
   beri açık olan `.rtp` fluid_params kaybıdır — bu partinin hatası değil.

6. **Maliyet** — donmuş parçacık varken GPU advect tail atlanır (katı
   parçacık istisnası host'ta), sıvı başına kare başına bir grid gidiş-dönüşü
   geri gelir. `fluid.step_stats` baytlarını not et; anlamlıysa sıradaki iş.

Bilinen sınırlar: cache playback sıcaklığı/donmayı taşımaz (render için
önemsiz, cache'ten *devam* etmek donmuş katmanı sıfırlar). Donmuş parçacık
hareketli collider'la birlikte gitmez (v=0 sabitlenir).

## Granüler grid hızı cihazda kalıyor (2026-09-25)

> ✔ 1–3 doğrulandı (canlı ölçüm): upload 1330→200 MB, download 582→25 MB,
> P2G 131→1.6 ms, G2P 163→25 ms, alt adım başına 6.25 MB; kullanıcı tek
> çekirdek beklemesini artık görmüyor. **4 ve 5 açık.**

**Şikâyet:** fluid domain + 2 collider, sim play ile cache'lenirken bazı
karelerde tek çekirdek meşgul, uzun bekleme.
**Ölçülen (derlemeden önce, canlı uygulama):** 114³ granüler domain,
**1000 parçacık**, bir karede **1.33 GB upload + 582 MB download**.
Her elastik alt adımda (32'ye kadar; collider yığını zorladıkça artıyor =
"bazı kareler") P2G alanı indiriliyor, solid-yüz kıskacı host'ta koşuyor,
G2P için geri yükleniyordu; kare boyunca sabit olan solid hızı da her alt
adımda 3 alan olarak yeniden yükleniyordu. Hepsi grid boyutuyla ölçekli,
tek iş parçacıklı memcpy.

Değişen: `ParticleSimulation.cpp`, `SimulationComputeVulkan.cpp` (kayıt
satırı), **yeni shader** `shaders/sim_fluid_zero_solid_faces.comp`
(4 buffer, 36 bayt push constant). Yeni `.cpp` yok.
Yeni prob: `scripts/test/probe_granular_transfer.py`.

1. **Shader derlendi mi** — `compile_shaders.bat` sonrası
   `sim_fluid_zero_solid_faces.spv` exe'nin shader klasöründe olmalı.
   *Bozuksa:* Scene Log'da bir kez `sim_fluid_zero_solid_faces unavailable`
   uyarısı çıkar ve eski yol koşar (2. maddedeki sayılar değişmez). Çökme
   OLMAMALI.

2. **Transfer baytları** — aynı sahneyi aç, timeline'ı oynat,
   `python scripts/test/probe_granular_transfer.py "Grid Domain 1" 20`.
   Görmen gereken (114³, 32 alt adım): **download ≈ 18 MB/kare** (3 yüz
   alanı, kare sonunda bir kez) + parçacık boyutlu; **upload ≈ 32 × 5.9 MB
   (fluid mask) + 18 MB (solid hız, bir kez) ≈ 200 MB/kare**. Önce 1330 / 582.
   `up/substep MB` sütunu ~6 civarında olmalı, ~41 değil.
   *Bozuksa:* hâlâ ~1.3 GB → yerleşik yol devreye girmedi. Kapı koşulları:
   viskozite > 0, katı-faz madde etiketi (solid substance), Periodic sınır,
   CUDA backend, GPU kuvvet entegrasyonu kapalı. Bunlardan biri doğruysa eski
   yol BEKLENEN davranıştır.

3. **Kare süresi** — collider temasında (alt adım sayısı 32'ye çıkarken)
   tek çekirdek beklemesi belirgin kısalmalı. `perf.get loop.frame`
   `last_ms` / `max_ms`; önce temas karelerinde yüzlerce ms.
   *Bozuksa:* 2 geçip 3 geçmiyorsa darboğaz transfer değilmiş — sıradaki
   aday alt adım başına `buildFluidMaskFromParticles` (tam grid fill) ve
   parçacık indirmeleri.

4. **★ SİNSİ OLAN: davranış aynı mı** — kum yığını collider'ın üstünde
   durmalı, içinden geçmemeli; domain duvarları tutmalı. Cihaz kıskacı
   uygulanmazsa sonuç **hatasız ve makul** görünür: kum biraz "daha akışkan",
   collider'a biraz gömülüyor. Kimse bunu bug diye raporlamaz.
   A/B: `sim_fluid_zero_solid_faces.spv`'yi geçici olarak yeniden adlandır
   (eski yol zorlanır), aynı kareye kadar oynat, `fluid.get`'ten
   `granular_sleeping`, `granular_yielded`, `granular_mean_accumulated_plastic`
   al; .spv'yi geri koy, tekrarla. Aynı mertebede olmalı. Bit-bit beklenmez:
   iki kıskaç aynı yüzleri sıfırlıyor ve float gidiş-dönüşü kayıpsız, ama
   P2G'nin float atomic'leri AYNI yolun iki koşusunda bile toplama sırasını
   değiştirir. Referans için önce eski yolu iki kez koş: aradaki fark gürültü
   tabanıdır.
   *Bozuksa:* kıskaç cihazda maskeyi yanlış okuyor (maske yüklemesi ile
   dispatch sırası) ya da indeks düzeni `FluidGrid::vel*Index` ile uyuşmuyor.

5. **Sıvı domain değişmedi** — collider'lı bir SIVI (granüler değil) domain
   eskisi gibi davranmalı; bu partide onun yolu değişmedi (`runGpuFluidG2P`
   içindeki "solid yok" kapısı çağırana taşındı, çağıran zaten veriyordu).
   *Bozuksa:* sıvı collider'dan sızıyorsa G2P artık host kıskacını görmeyen
   cihaz alanını örnekliyor → çağıran taraftaki `hasAnySolid()` kapısına bak.

## Particle Faz 0 — ölçüm katmanı + collider IPC (2026-09-24)

> 13:09 derlemesinde ölçüldü: 1 ✔, 5 ✔, 6 ✗ (`backend_not_vulkan` — kökü ve
> düzeltmesi en üstteki parçacık bölümünde; 6–8 bir sonraki derlemede tekrar).
> 2 (smoke), 3, 4 kullanıcıda.

Değişen: `ParticleSimulation.h/.cpp`, `RtApi.h`, `RtApiParticle.cpp`,
`RtIpc.cpp`, `RtIpcSecurity.cpp`, `RtPython.cpp`, `ProjectManager.cpp`,
`scene_ui_forcefield.hpp`, üretilmiş `RtIpcMethodDescriptors.cpp`.
Yeni `.cpp` yok, shader değişmedi. Audit derlemeden önce geçti.

Sıra: bağımsız ve hızlı olanlar önce; 6–8 ancak 1–5 temizse anlamlı.

1. **Yeni alanlar var mı** — `python scripts/ipc_test_client.py`.
   `particle.stats(phase0 fields)` satırında "missing" FAIL olmamalı,
   `collider.*` altı test OK olmalı.
   *Bozuksa:* "missing" → eski exe; `collider.create` "unknown method" → eski
   exe ya da dispatch; "not authorized" → `collider.` namespace'i yetkiye
   ulaşmadı.

2. **Smoke test** — `python scripts/rt_api_smoke_test.py` sonuna kadar geçmeli.
   Yeni assert'ler: `stage_backends` beş anahtar; `forces_on_gpu` ile
   `gpu_force_status == "gpu"` birebir; CPU politikasında
   `gpu_force_status == "cpu_policy"` ve `force_download_bytes == 0`.
   *Bozuksa:* CPU politikasında indirme byte'ı > 0 → politika kapısı GPU
   bloğunu atlamıyor.

3. **Panel ↔ çekirdek** — Simulation > Particles > Physics sekmesinde
   **Execution** combo'su ve altında gri "Last step: forces CPU/GPU (durum),
   backend adı" satırı. Combo'yu CPU yap → timeline'ı birkaç kare oynat →
   satır `cpu_policy` demeli; IPC'den `particle.get_physics` aynı değeri
   (`cpu`) dönmeli. Tersini de dene: `particle.set_physics
   {"execution_policy":"auto"}` → combo Auto'ya dönmeli.
   *Bozuksa:* panel ile IPC farklıysa panel başka bir runtime'ı okuyor.

4. **Kaydet/aç** — politikayı CPU yap, projeyi kaydet, kapat, aç →
   combo hâlâ CPU. *Bozuksa:* `ProjectManager` `physics.execution_policy`
   yazmıyor/okumuyor; alan açılışta sessizce Auto'ya düşer.

5. **GPU Required gerçekten reddediyor mu** — sim compute backend'i Vulkan
   değilken (ya da Vulkan'da iken `particle.stats.compute_backend` neyse)
   `set_physics {"execution_policy":"gpu_required"}`, bir `particle.spawn`,
   `particle.step`, `particle.stats`:
   Vulkan değilse `step_blocked: true`, `gpu_force_status:
   "gpu_required_blocked"`, `stage_backends.forces: "blocked"` ve parçacık
   **hareket etmemiş** olmalı (`get_state_sample` pozisyonu değişmez).
   *Bozuksa:* parçacık düştüyse GPURequired sessizce CPU'ya düşüyor —
   yol haritasının açıkça yasakladığı şey.

6. **★ Auto satırı gerçekten GPU mu** — boş sahnede, timeline durmuşken:
   `.\scripts\ipc\Probe-ParticleBaseline.ps1 -Counts 2048 -Scenarios ballistic -Samples 10`
   Auto satırındaki `force[...]` **`gpu`** olmalı.
   ★ **En sinsi sonuç:** `backend_not_vulkan` / `no_dispatch_support` /
   `buffers_not_ready`. Script hata vermez, Auto satırı CPU satırıyla aynı
   süreyi gösterir ve "GPU hızlandırması yok" diye okunur — oysa GPU yolu
   **hiç koşmamıştır**. Bu durumda karşılaştırma tablosu anlamsızdır; önce
   nedeni (status değeri) bana yaz.

7. **Transfer maliyeti görünür mü** — aynı koşuda Auto satırında
   `down` (force_download_bytes/adım) ≈ `3 × 4 × capacity` byte olmalı
   (üç hız bileşeni, **kapasite** boyutunda — alive değil), `force_sync_calls`
   ≈ 1. `mirror` her iki politikada ≈ `(9×4 + 1) × capacity` byte.
   *Bozuksa:* 0 → sonda (probe scope) transferleri görmüyor; beklenenden çok
   büyükse uploadToCompute adım başına iki kez tam yükleniyor (emit
   `data_version_`'ı artırdığında beklenen, ama sayıyı not et).

8. **Tam baseline** — `.\scripts\ipc\Probe-ParticleBaseline.ps1 -OutputPath
   .\particle_baseline.json`. 3 senaryo × 3 sayım × 2 politika. Sonda
   "CPU vs other policy" tablosu çıkar.
   Beklenen: `ballistic` için `max_position_delta` çok küçük (~1e-4 m ve
   altı). `plane` ve `self_collision`'da fark büyüyebilir — çarpışma küçük
   float farklarını büyütür, bu tek başına hata değildir; ama `ballistic`
   büyükse GPU kuvvet çekirdeği ile CPU yolu **farklı fizik** hesaplıyor.
   `nonfinite_max` her satırda 0 olmalı.
   JSON'u bana gönder: Faz 5'in kabul toleransları ve hız kapıları bu
   sayılardan türetilecek.

### Bu partide bulunan ve ölçülmesi gereken şeyler

- GPU kuvvet bloğu (upload + dispatch + sync + hız indirme)
  `integrate_start`'tan **önce** çalışıyordu; hiçbir aşama zamanlayıcısı
  onu görmüyordu. Artık `gpu_force_ms`.
- İndirme doğrudan `buffers_`'a yapılıyordu: y bileşeni başarısız olursa x
  zaten GPU'da entegre edilmiş kalıyor ve CPU yolu kuvveti x'e **ikinci kez**
  uyguluyordu. Artık üçü de gelirse takas ediliyor, yoksa host dokunulmamış.
- Transferler **kapasite** boyutunda; ölü slotlar da taşınıyor.
- `spawn()` her parçacıkta `findDeadSlot()` ile baştan tarıyor (O(kapasite)).
  Büyük burst'lerde `emit_ms`'e bak.
- Çarpıştırıcılar yalnızca Python'daydı; `collider.*` artık IPC'de.
## Kinematic Collider Sources K1/K2 — build/live checks (2026-09-26)

> First K1 build live result: IPC CRUD/validation PASS. Character `1` exposed
> 80 bones; old auto-fit produced 64/64 resolved and moving proxies, but its
> arbitrary input order kept finger chains before the right lower leg/foot.
> Body-first + detail-opt-in is a post-build source fix and is the first check
> for the next build. A cleaned 22-proxy preview set was left in the live scene.
> The 2026-09-26 scale probe then found a second concrete unit bug: the 1%
> Mixamo rig turned the nominal 0.025-0.20 m auto-fit limits into 0.0008-0.002 m
> sampled radii. Auto-fit now evaluates limits and bone lengths in world metres
> and stores the converted bone-local dimensions; this change needs a rebuild.

1. Build once after the K1 source additions. Open a scene with one evaluated
   rig visible in `rig.list_characters`.
2. Open Simulation > Colliders > Kinematic Collider Sources. Select the rig,
   add a set, then run Auto Fit Skeleton. Confirm the list contains bounded
   sphere/capsule/box proxies, includes both feet, excludes finger/eye/skirt/
   twist/end detail bones by default, and no unresolved proxy appears at the
   world origin. Run `rt_probe_kinematic_rig_ipc.py 1`; expected PASS.
3. With viewport gizmos enabled, confirm cyan analytic outlines follow the
   animated bones. Toggle Show in Viewport and confirm the overlay disappears
   without disabling solver participation. The overlay already projects world
   dimensions; after rebuilding, rerun
   `python scripts/test/rt_setup_kinematic_preview_ipc.py 1`. It reuses and
   refits the exact `1 Kinematic Preview` set instead of accumulating copies.
4. From a separate terminal run
   `python scripts/test/rt_test_kinematic_collider_ipc.py`. Expected: `PASS`.
   In the Codex sandbox the named pipe may require an escalated run; Windows
   error 5 is not evidence that the app listener is absent.
5. For the real rig, call `physics.collider.proxy_set.sample` twice across two
   animation poses. Confirm `resolved=true`, centers follow their bones, and
   the second sample reports finite velocity. Scrub backwards, then later K2
   must reset authoritative history before any solver consumes it.
6. Save to a disposable `.rtp`, reopen it, and verify set/proxy IDs, shapes,
   local values, consumer mask, contact values and `viewport_visible` survive.
   A malformed or duplicate ID must reject the kinematic section instead of
   partially loading it.
7. Run `python scripts/test/rt_inspect_kinematic_scale_ipc.py` and compare one
   thigh plus each foot with the viewport. Sampled capsule segment/radius must
   be in world units. On the live 1% rig, a thigh/leg radius should be on the
   order of centimetres (roughly 0.09 m for a 0.52 m segment), not the measured
   0.002 m. Foot boxes must also remain proportional after the refit.
8. Put a low/medium-resolution Fluid domain around the legs, seed water, keep
   Fluid + Granular consumers enabled, and play the walk. The leg/foot cells
   must block liquid and transfer local limb velocity; a moving proxy must not
   leave solid ghost cells behind it. Scrub backwards and replay: no velocity
   explosion on the first resumed step. Shortcut:
   `python scripts/test/rt_setup_kinematic_water_ipc.py`. For an existing
   scene, run `rt_inspect_kinematic_fluid_scene_ipc.py` first. Do not infer the
   visible free-surface height from the domain AABB: `fluid.get` currently
   reports domain bounds and aggregate particle counts, not the particles'
   world-space height distribution. Confirm water depth in the viewport (or a
   future particle-bounds diagnostic) before judging foot contact.
   **2026-09-26 live partial PASS:** on the Vulkan water scene, an IPC-driven
   frame 0 -> 6 produced 6/6 collider-voxelization calls (6.981 ms total),
   retained 57,800 particles, and followed 0.26-0.34 m foot motion. The user
   confirmed visible foot gizmos and local water interaction. Rewind then
   reported `dropped_seeds=['Grid Domain 1']`; that run is invalid for
   ghost-cell acceptance. Refill using a persistent FillLevel/reseed recipe,
   then rerun `rt_probe_kinematic_rewind_ipc.py`. Remaining: moving-stamp ghost
   check and rewind/replay velocity.
   **Authoring bug found:** `Seed Fluid Now` always calls the SeedBox service;
   that service changes the runtime mode to `SeedBox`, even when the panel says
   `Fill Domain`. With `Recreate Seed on Reset` off, rewind then drops the tank.
   Until the shared core/API/IPC fix lands, use Fill Domain with Auto Reseed on
   Edit and do not press Seed Fluid Now, or explicitly arm Recreate Seed on
   Reset before testing rewind.
9. Repeat with a smoke-filled Gas domain and Gas enabled. CPU gas and Vulkan
   gas must both part smoke locally around the moving limbs; Vulkan uses the
   same host-stamped `solid`/`solid_vel` upload contract. If the whole column
   follows the character centroid, the old rigid-collider path is being used.
   Shortcut: `python scripts/test/rt_setup_kinematic_smoke_ipc.py`.
10. Disable Fluid or Gas in the set consumer mask while leaving Show in
    Viewport enabled. The outline must stay visible but the disabled solver
    must ignore the proxies. Re-enable it without recreating the set.
# 28. parti — Faz 4 tek Matter domain (2026-09-30)

**Durum:** DERLENDİ, CANLI IPC PASS.

- `Matter=2`, eski `Gas=0` / `Fluid=1` sayılarını korur.
- Tek domain iki ayrı fizik alanı taşır: gaz ana grid, APIC `matter_liquid_grid`.
- UI/API/IPC/Python: `type=matter`, `phases=[gas, liquid|granular]`.
- Flow source `phase=gas|liquid`; panel ve proje serileştirmesi aynı alanı kullanır.
- Render: ana hacim liquid SDF; ikinci hacim gaz ile liquid fog/whitewater'ı tek
  katılımcı ortamda birleştirir.
- Burning Fuel Spill ve Ignited Fuel Jet artık tek Matter domain oluşturur.
- Statik kontroller: descriptor üretimi/check PASS (651/632/513), JSON ve Python
  syntax PASS. Proje derlemesi ajan tarafından çalıştırılmadı.

Canlı kabul (boş sahne):

```powershell
python scripts/test/rt_test_matter_domain_ipc.py
```

Kapı: tek domain listelenmeli; `phases` gaz+sıvı olmalı; aynı domain'deki
iki faz-kaynak sonrası hem `particle_count > 0` hem `active_density_cells > 0`
ve `gas.step_stats.measured == true` olmalı. Test kaynak/domain temizliğini
`finally` ile yapar.

Canlı sonuç:

- `PASS: one Matter domain advanced gas and liquid phases`
- `particle_count = 120`
- `active_density_cells = 67`
- Faz 4 kapısı kapandı.

# 29. parti — gerçek Matter preset sertleştirmesi (2026-09-30)

**Durum:** KAYNAK DÜZELTİLDİ, YENİDEN DERLEME/CANLI KABUL BEKLİYOR.

Kullanıcının açık Burning Fuel Spill sahnesinde eski ikili-domain beklentilerini
görünür kılan ilk gerçek preset ölçümü:

- IPC yalnız **bir** `Burning Fuel Matter` domain'i raporladı; panelde görülen
  gaz/sıvı bölümleri aynı domain'in iki fazıdır.
- 65.664 sıvı parçacığı ve 41.703 etkin SDF hücresi vardı; gaz yoğunluğu,
  gaz kütlesi ve dönüşüm defteri sıfırdı. Bu nedenle ikinci gaz hacmi haklı
  olarak oluşmuyordu.
- Tek adımda gaz tarafı yaklaşık 139,38 ms harcadı: basınç 54,03 ms, scalar
  advection 20,44 ms, host sync 7,61 ms. Boş gaz fazını sırf Matter sıvı
  parçacığı var diye çalıştıran kaba kapı bunun ana nedeniydi.
- Ana SDF hacminin VDB kimliği aynı anda viewport'a `live_gas` olarak
  yayınlanıyordu. Backend değişiminin yüzeyi bir kare geri getirip sonraki
  karede yeniden kaybetmesi bu yanlış kimlik sahipliğiyle uyumluydu.

Kaynak düzeltmeleri:

1. Matter ana hacmi yalnız sıvı SDF'dir; canlı gaz GPU alanları bu kimliğe
   bağlanmaz. Gaz/fog ikincil hacim slotunun sahibidir.
2. `fluid_auto_ignite`, boş gaz fazında ilk yakıt/ısı/duman aktarımını gerçekten
   başlatır. Önce sıcak gaz gerektiren kilitlenme kaldırıldı.
3. Sırf Matter sıvı parçacıkları var diye gaz çözücüsü çalışmaz. Gaz kaynağı,
   mevcut gaz içeriği veya gerçekten auto-ignite olabilen yanıcı sıvı yoksa
   gaz fazı `Idle` kalır. Gaz içeriği testi yoğunluğun yanında fuel/flame/ısıyı
   da kapsar.
4. Public API örneği `production_gas_presets.add_burning_fuel_spill()` ve
   `scripts/test_burning_fuel_spill.py` de eski iki-domain kurulumundan tek
   `type="matter"` domain'e geçirildi; eski adları tekrar çalıştırmada temizler.

Derlemeden sonra tek kabul turu:

1. Burning Fuel Spill'i boş sahneye ekle, Play'i en az 10 kare yürüt. SDF her
   kare görünür kalmalı; backend değiştirmek gerekmemeli.
2. `fluid.get` içinde `active_density_cells > 0`, `matter.exchange_stats`
   içinde en az bir aktarım ve `render.volume_slots` içinde birbirinden farklı
   ana SDF ile Matter Gas kimlikleri görülmeli.
3. Ardından mevcut `scripts/test/rt_test_matter_domain_ipc.py` çalışmalı.
4. Yanıcı olmayan, yalnız sıvı kaynaklı bir Matter domain'de bir adım ölç:
   `gas.step_stats.measured=false` ve durum `Idle` olmalı. Böylece boş gaz
   basınç maliyetinin geri gelmediği kanıtlanır.

Bu tur shader değiştirmedi. Ortak bounds nedeniyle Burning Fuel Spill'in APIC
alanı eski 1,8 m yüksek sıvı kutusu yerine 5 m yüksek Matter kutusunu kullanır;
daha serbest/çalkantılı hareket ve APIC grid maliyetinin bir bölümü bunun doğal
sonucudur. Bunu geri almak aktif alt-grid/sparse sıvı çalışması ister ve Faz 5
performans işi olarak ölçülmelidir.

İlk yeniden derleme sonrası canlı bulgu:

- Gaz görünür oldu ve iki render slotu doğru ayrıldı (`SDF=888`, `Matter Gas=887`).
- Kare 46'da 65.664 sıvı parçacığı ve 38.155 SDF hücresi vardı; kare 47'de
  parçacık sayısı bir anda sıfıra, SDF kimliği `-1`'e düştü. Gaz fazı 9.302 kg
  raporlayıp yaşamaya devam etti. Bu bir render kaybı değil, gerçek toplu kütle
  tüketimiydi.
- Kök neden: CPU faz aktarımı `auto_ignite` etkisini bütün APIC parçacıklarına
  her kare tam oranla uyguluyordu. Aynı başlangıç kütle payına sahip bütün
  parçacıklar aynı karede kaldırma eşiğine ulaşıyordu.
- Kaynak düzeltildi: yanıcı sıvı aktarımı artık APIC yoğunluğundan serbest yüzeyi
  (üst ve dört yan komşu) bulur ve yalnız bu yüzey parçacıklarını tüketir. Alt
  yüzey zeminden buhar kaynağı sayılmaz; davranış mevcut GPU yüzey köprüsüyle
  aynıdır. Yeniden derleme/canlı kabul bekliyor.

Sonraki kabulde kare 46→47 arasında `particle_count` sıfıra düşmemeli. Gaz ve
SDF ayrı slotlarda kalmalı; gaz kütlesi artarken sıvı kütlesi yüzeyden kademeli
azalmalıdır.

# 30. parti — Matter fazlarına ayrı kalıcı GPU buffer (2026-10-01)

**Durum:** KAYNAK TAMAM, DERLEME/CANLI A/B BEKLİYOR.

Tek domain'in iki fizik çözmesi tek başına hata değildir; iki aktif faz iki
farklı denklem takımı gerektirir. Yapısal sorun, iki fazın aynı
`SimulationGridDomainComputeBuffers` nesnesini sırayla kullanmasıydı: APIC
çekirdekleri gazın cihaz alanlarının üstüne yazıyordu. Gaz host-kaynak yayını
ayrıca bugün bütün alan defterini kaba biçimde invalid ediyor; bu sonraki ayrı
optimizasyondur.

Değişiklik:

- Matter domain başına `matter_liquid_compute_buffers_` eklendi.
- Sıvı grid'i APIC için öne alınırken sıvı GPU buffer takımı da öne alınır;
  adım sonunda grid ve buffer birlikte geri değiştirilir.
- Gaz buffer'ı APIC boyunca yerinde kalır. Reset, cache restore,
  domain silme ve compute-resource release iki takımı da yönetir.
- GPU whitewater render buffer'ı Matter için sıvı takımından okunur.
- Fizik, shader ve kullanıcıya açık ayarlar değişmedi. Karşılığı daha yüksek
  kalıcı VRAM'dir. Kaba gas-host invalidation hâlâ upload doğurabileceği için
  hız kazancı ölçülmeden varsayılmaz.

Derleme sonrası A/B kabulü:

1. Ignited Fuel Jet'i 20–30 kare çalıştır; `fluid.step_stats` ve
   `gas.step_stats` oku.
2. Önceki canlı referans: 80×55×40 grid, gaz `total_ms≈80,69`,
   `gpu_source_upload_ms≈1,08`, `gpu_host_sync_ms≈1,03`. Burning Fuel Spill'in
   eski kötü karesi: 50³ grid, `total_ms≈139,38`, source upload 1,84 ve host
   sync 7,61 ms idi. Sahne/frame birebir aynı değilse yalnız aşama dağılımını
   karşılaştır, yüzde hızlanma iddia etme.
3. Gaz ve SDF görünümü, sıvı parçacık sayısı, `matter.exchanges` kütle/enerji
   hataları değişmemeli. `render.volume_slots` iki ayrı kimlik göstermeli.
4. `perf.get_gpu_memory` ile ek kalıcı VRAM'i kaydet. Bu artış beklenir; fazların
   cihaz verisini her kare yeniden kurmamanın bedelidir.

Sonraki katman: APIC'nin bugün tam `grid.getCellCount()` üzerinde dispatch ettiği
clear/normalize/density/MGPCG geçişlerini, parçacık AABB + güvenlik bandından
üretilen etkin hücre bölgesine indirmek. Bu, ortak Matter kutusunun boş üst
bölgesini çözmez ve eski ayrık sıvı kutusuna yakın maliyeti geri getirir. Güçlü
iki-fazlı tek basınç denklemi değildir; iki çözücünün fizik ayrımını korur.
## 31. parti — az splat için RT havuzu ve ortak-grid teşhisi (2026-10-01)

Kullanıcı bulgusu: yalnız birkaç splat görünürken bile maliyet yüksek ve RT moda
geçiş aralıklı TDR üretiyor. Kök statik incelemede doğrulandı:
`fluidPoolCapacityFor` ilk canlı splatta havuzu doğrudan **16.384** instance'a
yuvarlıyordu. Görünmeyen slotlar kararlı indeks/refit için maskelense de TLAS
instance kaydı, ilk yapısal kurulum ve geçiş maliyetini taşıyor.

Kaynak değişikliği (derlenmedi): küçük havuz kademesi artık 256'dan başlıyor ve
256→1024→4096→16.384 şeklinde büyüyor. 16.384 üstündeki geometrik büyüme ve 1M güvenlik
tavanı değişmedi. Böylece 1–256 splat için ilk RT havuzu **64 kat** küçüldü.
Sabit havuz/refit sözleşmesi korunuyor; her kare create/delete geri gelmedi.

Ortak-grid teşhisi: Matter bugün gaz `state.grid` ve sıvı
`matter_liquid_grid` için aynı çözünürlük, voxel ve tüm-domain sınırını kullanıyor.
Bu, tipik birleşik presette sıvıyı gazın geniş kutusunda çözdürüyor. Ayrıca kaynak
bütçesi Matter için 128+224=352 byte/hücre saydığı için aynı MB sınırında eski
Gas domain'e göre çözünürlüğü daha erken düşürebilir; bu durumda gazın girdap,
alev kalınlığı ve yükselme ayrıntısı değişebilir. Hedef mimari tek domain kimliği
altında faza özgü fiziksel grid kapsamı/çözünürlüğüdür. Yeni ayrı GPU buffer
takımları bunun ön koşuludur. CPU serbest-yüzey basınç çözücüsü zaten parçacık
AABB'sini bir hücre genişletip o bölgede çalışıyor; kalan büyük kazanç GPU APIC
P2G normalize/CG dispatch'lerini ve fiziksel sıvı gridini etkin bölgeye indirmek.

Derleme sonrası kapı:

1. Az splatlı sahnede ilk yapısal kurulumun 256-slot sınıfında olduğunu log/perf
   ile doğrula; `render.fluid.splat_instances` ve RT geçiş süresini kaydet.
2. Sim oynarken Material/Solid→Rendered geçişini birkaç kez tekrarla; TDR ve
   `device lost` olmamalı.
3. Splat sayısını 256 ve 1024 sınırlarından geçir; yalnız sınır geçişinde bir
   yapısal rebuild, aradaki karelerde refit görülmeli.
4. Büyük havuz regresyonu: 16k üstünde önceki ikiye katlama ve 1M tavan sürmeli.

**Canlı sonuç — PASS (2026-10-01):**

- Burning Fuel Spill, Vulkan Matter, 50³ ortak grid: 12 kare sonunda 65.664
  sıvı parçacığı, 9.693 aktif gaz hücresi ve 6.670 yanan hücre; dropped seed yok.
- Isınmış kareler yaklaşık 245–265 ms. Son karede sıvı P2G 3,71 ms,
  basınç 10,05 ms, G2P 1,15 ms; fakat aktarım hâlâ 16.545.616 byte upload +
  10.868.616 byte download/kare. Ayrı buffer doğruluk kapısını geçti, büyük
  bant genişliği kazancı üretmedi.
- Gaz son adım: toplam 98,45 ms; source upload 2,67 ms, host sync 3,09 ms,
  basınç 12,83 ms, scalar advect 3,89 ms, analysis 5,03 ms, voxelize 5,05 ms.
  Raporlanan alt aşamalar toplam süreyi açıklamıyor; tam-grid dispatch/batch ve
  ölçülmeyen senkronizasyon sonraki profiling hedefi.
- Kontrollü geçici sahne: 6.120 body + **2 spray**. Material telemetrisi tam
  **256 total_instances**, 2 full_instances ve 40 görünür üçgen raporladı.
  İki Material→Rendered geçişi tamamlandı; `device_lost=false`, TDR yok.
  RT delta: 5 rebuild / 37,07 ms ve 5 update_geometry / 51,72 ms.
- Test kendi domainini kaldırdı, Burning Fuel Matter'ı yeniden etkinleştirdi ve
  görünümü başlangıçtaki Solid moda döndürdü. Son durum tek domain, kare 12.

# 32. parti — tek fazı da yavaşlatan fiziksel envanter regresyonu (2026-10-01)

**Durum:** KAYNAK TAMAM, DERLEME/CANLI A/B BEKLİYOR.

Kullanıcı, maliyetin yalnız iki faz aynı anda çalışırken değil, tek gaz veya tek
sıvıda da yüksek olduğunu bildirdi. Burning Fuel Spill gaz ölçümünde raporlanan
alt aşamaların `total_ms` değerini açıklamaması üzerine ortak gaz kuyruğundaki
ölçülmeyen işler incelendi.

Bulgu ve düzeltme:

1. Her gaz ve Matter grid yeniden boyutlandığında fiziksel faz kütlesi ile enerji
   sidecar'ları tam hücre sayısında ayrılıyor, sonra içerikleri tamamen sıfır olsa
   bile her kare CPU'da iki MacCormack scalar advection geçiriyordu. Düz gaz
   domain'i de Matter faz aktarımı kullanmadığı halde bu bedeli ödüyordu.
2. Sidecar'lar artık ilk gerçek sıvı→gaz aktarımına kadar ayrılmıyor. Bu, düz gaz
   ve henüz aktarım yapmamış Matter için iki tam-grid CPU geçişini kaldırır.
3. Envanter mevcutsa pasif scalar taşınması pozitif hücrelerin AABB'si ile iki
   yönlü trace ve trilinear örnekleme için güvenli bandı kapsar. Periodic sınır
   tam grid kullanır. Kapalı sınırda mevcut toplam-kütle/enerji renormalizasyonu
   korunur; geçersiz ve negatif değerler sıfırlanır.
4. Tekrarlanan büyük geçici vektör allocation'ları thread-local scratch ile
   kaldırıldı.
5. Eksik ölçüm yüzeyleri tamamlandı: `gas.step_stats` IPC/Python artık
   `inventory_advection_ms`, `fluid.step_stats` ise tam sıvı-domain `total_ms`
   değerini döndürür. Çekirdek API, IPC, Python ve descriptor metinleri birlikte
   değişti.

İlk canlı tek-sıvı ölçümü (100.000 parçacık, 40³ grid, Vulkan, dört kare):

- P2G 8,59 ms, basınç 19,76 ms, G2P 5,00 ms, advect 4,25 ms ve density
  1,64 ms idi. GPU etiketleme son karede 10,17 ms sürdü.
- Kare başına 15.443.200 byte upload, 10.630.584 byte download; 14 upload,
  17 download, 174 dispatch, 13 batch sonlandırma ve 3 synchronize görüldü.
  Yalnız batch sonlandırmalarında 48,26 ms, synchronize çağrılarında 7,67 ms
  host beklemesi ölçüldü.
- İlk `fluid.step_stats.total_ms` 0,0745 ms döndü; bu gerçek toplam değildi.
  Vulkan APIC yolu `Fluid::step`i ikiye böldüğü için sayaç yalnız ikinci küçük
  tail çağrısını ölçüyordu. Kaynak düzeltildi: `total_ms` artık APIC, etiketleme,
  whitewater ve density köprüsünü kapsayan gerçek sıvı-domain duvar süresidir.
- GPU timestamp desteği bu compute queue'da yok; kernel süre tablosu alınamadı.
  Mevcut kanıt sıvı maliyetinin gazdaki boş CPU geçişiyle aynı kökten gelmediğini,
  gerçek APIC basınç/parçacık işi ve çok sayıdaki submit/readback sınırında
  toplandığını gösteriyor.

Temiz sayaçla tekrar: dört `sim.timeline.step` toplam 234,50 ms, son kare
56,54 ms, en yavaş kare 70,33 ms. Son kare P2G 7,57 ms, basınç 13,64 ms,
G2P 3,03 ms, advect 3,26 ms, density 1,74 ms ve GPU label 6,02 ms idi.
13 batch sonu 34,46 ms, üç explicit synchronize 5,81 ms sürdü. Dolayısıyla
tek-sıvı referans yükünün gerçek sim maliyeti yaklaşık 56–70 ms/kare; viewport
ve 16 ms throttle bunun dışında. En yüksek sonraki kazanç, G2P'nin velocity+
affine readback'i ile hemen arkasındaki device advect-tail position+velocity
readback'ini tek teslim/readback sınırında birleştirmek; ardından aktif sıvı
grid kapsamıdır. Bu, başarısız device-tail yolunda doğru host fallback'i
koruyacak ayrı bir residency değişikliği olarak yapılmalıdır.

Derleme sonrası kısa kabul:

1. Burning Fuel Spill'i aynı voxel/kaliteyle 12 kare sür. Son üç karenin frame,
   `fluid.step_stats.total_ms`, `gas.step_stats.total_ms` ve
   `inventory_advection_ms` değerlerini kaydet. Gaz envanteri varken yeni kalem
   sıfır olmamalı; önceki 98,45 ms gaz toplamıyla yalnız aynı sahne/frame ise
   karşılaştır.
2. Tek gaz presetini en az 8 kare sür. Faz aktarımı olmayan düz gazda
   `inventory_advection_ms` yaklaşık sıfır olmalı ve grid-domain bilgisinde
   fiziksel gaz kütlesi sıfır kalmalı.
3. Yanıcı olmayan tek sıvılı Matter sahnesini sür. Gaz adımı `measured=false`
   kalmalı; sıvı `total_ms` gerçek APIC toplamını vermeli.
4. Kapalı Burning sahnesinde kütle/enerji ledger hatası ve SDF/gaz görünümü
   değişmemeli. Aktif-bölge taşıması performans değişikliğidir; fiziksel toplamı
   değiştirmemelidir.

Sonraki maliyet katmanı değişmedi: sıvının ortak gaz kutusunda tam-grid GPU
P2G/normalize/MGPCG ve büyük transferler çalışması. Part 32 bunu çözmez; tek-gaz
regresyonunu kaldırır ve tek-sıvı APIC toplamını ölçülebilir yapar.

# 33. parti — tek-domain çekirdeği kapanış sırası (2026-10-01)

**Durum:** PERFORMANS SIRASI KORUNDU; tam su+kum fiziği C4–C6 olarak eklendi.

Bağlayıcı ayrıntı ve bitiş ölçütleri
`BIRLESIK_MADDE_DOMAIN_TASARIMI.md §8d` içindedir. Kalan sıra:

1. **C0:** Part 32'nin build/live kabulü.
2. **C1:** G2P + device advect-tail geri okumalarını tek teslimde birleştir.
3. **C2:** GPU APIC için parçacık AABB + halo etkin hücre penceresi.
4. **C3:** Tek Matter kimliği altında faza özgü bounds ve voxel; UI/API/IPC,
   serializer/cache ve preset göçü birlikte.
5. **C4:** Parçacık başına liquid/granular constitutive model ve çok malzemeli
   P2G/contact; domain-geneli `granular_enabled` fizik otoritesi olmaktan çıkar.
6. **C5:** Korunumlu serbest su ↔ granül gözenek suyu emilim/drenajı.
7. **C6:** Doygunluktan kumun efektif gerilme, sürtünme, kohezyon, dilatasyon ve
   aynı otoriteden kuru/ıslak materyal görünümü.
8. **C7:** Eşdeğer kaliteyle performans A/B matrisi ve su jeti/kum yatağı,
   ıslanma cephesi, drenaj, kuru-nemli-doygun yığın kabulü. Ledger, kapalı tank,
   determinizm, cache ve render regresyonları geçince plan `TAMAMLANDI` yapılır.

Gaz-sıvı güçlü ortak projeksiyon ayrı kalır. Granül-sıvı contact, emilim,
gözenek basıncı ve doygunluk görünümü kapanışın içindedir. RT SDF bloklaşması,
splat materyal editörü, Solid-splat sırası ve OptiX kapsamı ayrıdır.

# 34. parti — C0 canlı ölçüm ve liquid-only boş gaz düzeltmesi (2026-10-01)

**Durum:** CANLI IPC PASS; C0 KAPANDI.

Son derlemede C0 ölçümleri:

- 100.000 parçacık / 40³ Vulkan sıvı: gerçek `fluid.step_stats.total_ms`
  18,29 ms; son `sim.timeline.step` 23,17 ms. Yeni toplam sayaç kabul edildi.
- Düz gaz 40³, on adım: fiziksel kg/J envanteri yokken
  `inventory_advection_ms=0,0009`; lazy sidecar kapısı kabul edildi.
- Liquid-only Matter 40³, 100.000 su parçacığı: sıvı 52,21 ms iken boş gaz
  yine 27,48 ms çalıştı. Gaz density/fuel/temperature sıfırdı; buna rağmen
  `liquid_boundary_cells=27081` idi.

Kök neden ve kaynak düzeltmesi:

1. Sıvı SDF splat'i ortak `state.active_density_cells/max_density` sayaçlarını
   dolduruyordu. Gaz idle kapısı bunları gaz içeriği sanıp basınç, scalar,
   analiz ve voxel geçişlerini çalıştırıyordu.
2. Gaz idle kararı artık analizden alınan gaz-fazına ait
   `gas_stats.active_density_cells/max_density` snapshot'ını okur. Public
   `gas.step_stats` de aynı kaynağı raporlar.
3. İlk karede Matter mist aktarımını uyutmamak için gerçek `Mist` etiketi ayrı
   uyanma nedenidir. Böylece boş sıvı SDF gazı uyandırmaz, aktarılacak mist uyandırır.

Derleme sonrası tek kabul:

1. Yanıcı olmayan liquid-only Matter, 40³ ve 100.000 suyla en az altı doğrudan
   adım ilerletilir.
2. `gas.step_stats.measured=false`, gas compute status `Idle` olmalı;
   `fluid.step_stats.total_ms` geçerli kalmalı.
3. Mist içeren Matter'da ilk adım gas solver'ı uyandırmalı ve mist→gas ledger
   kaydı üretmeli.
4. Bu iki kapı geçince C0 kapanır; sıradaki kod partisi C1 residency'dir.

2026-10-01 canlı tekrar: geçici liquid-only Matter 40³ domain'e tam 100.000
parçacık tohumlandı ve altı Vulkan adımı ilerletildi. Parçacık sayısı
100.000 → 100.000 kaldı, `fluid.step_stats.total_ms=50,1854` ve
`gas.step_stats.measured=false` döndü. Sıvı SDF sayaçları boş gaz çözücüsünü
artık uyandırmıyor. Geçici domain kaldırıldı.

# 35. parti — Substance → parçacık constitutive kimliği temeli (2026-10-01)

**Durum:** DERLENDİ; CANLI API/VERİ YOLU PASS, ÇOKLU CONSTITUTIVE SOLVER C4'TE. Bu parti
çok malzemeli solver/contact değildir; C4'ün kaybolmayan veri sözleşmesidir.

- Yeni `MatterConstitutiveModel`: `auto|fluid|granular|elastic`. Termodinamik
  `phase` ve render `representation` eksenlerinden ayrıdır.
- Flow Source, Substance profili ve domain Substance override'ı modeli taşır.
  Çözüm sırası: emitter başlangıç override'ı → domain Substance override'ı →
  merkezi Substance varsayılanı → eski `granular_enabled` uyumluluk fallback'i.
- `FluidParticles::constitutive_model` emit, reserve, swap-remove, compact,
  cache ve bellek telemetrisinde parçacıkla birlikte yaşar. SimCache v9 oldu;
  eski disk cache yeniden bake ister.
- Merkezi kütüphaneye `Sand`, `Gravel`, `Soil` granular varsayılanıyla eklendi.
  `Water` ve mevcut akışkan profilleri fluid varsayılanındadır.
- Emitter paneli Substance picker, custom id, merkezi kimya özeti ve başlangıç
  modeli gösterir. Ayrıntılı reaksiyon sabitleri emitter'a kopyalanmaz; merkezi
  Substance/Reaction Rules tasarımında kalır.
- UI, Python ve IPC aynı core setter/parsing yolunu kullanır. Proje/scene
  serializer, simulation signature ve generated IPC descriptor güncellendi.
- `fluid.get`/`fluid.list_domains` artık `fluid_model_particles`,
  `granular_model_particles`, `elastic_model_particles` ve
  `unresolved_model_particles` raporlar.

Statik kabul:

- `python scripts/gen_ipc_descriptors.py --check` PASS: 652/633/513.
- JSON/XML parse ve `rt_test_fluid_substances.py` Python syntax PASS.
- Proje derlemesi ajan tarafından çalıştırılmadı.

Derleme sonrası:

1. `python scripts/test/rt_test_fluid_substances.py` çalıştır. Granular binding
   readback'i, geçersiz model reddi ve en az bir `granular_model_particles`
   üretimi PASS olmalı.
2. Matter emitter'da Substance=`Sand`, Initial Model=`From Substance`; birkaç
   kare sonra granular sayaç artmalı. Substance=`Water` emitter'ı aynı domain'de
   fluid sayaç üretmeli.
3. Bu tur yalnız kimliğin uçtan uca taşındığını kanıtlar. İki popülasyon henüz
   ayrı constitutive çekirdeklere dispatch edilmediği için fiziksel su+kum
   sonucu kabul edilmeyecek. Sonraki C4 partisi GPU/CPU sınıflandırılmış
   P2G/constitutive dispatch ve contact alanıdır.

2026-10-01 canlı kabul:

- Flow Source `initial_constitutive_model=auto` ve Substance kimliği doğru
  okundu. Granular override round-trip, geçersiz model reddi ve parçacık SoA
  sayacı geçti. Solid faz ölçümünde 2.666 işaretli parçacık / yaklaşık 690
  bloklayan hücre görüldü; master switch kapalıyken hücre sayısı sıfırlandı.
- İki sıvı kaynağı bağımsız okumada toplam 5.332 parçacık olarak, iki ayrı
  Substance etiketiyle yayınlandı. Uzun ve kesintisiz `fluid.step` IPC dizisinin
  hemen sonundaki aynı-oturum sorgusu sıfır görüyor; oturum serbest kalınca
  sonuç frame loop tarafından topluca yayınlanıyor. Bu harness zamanlaması
  fizik sonucunu boş saymamalı; cache kabulü timeline kareleriyle ayrıştırılacak.
- Görsel kontrolde fluid emitter birkaç kare tutup sonra parçacıkları topluca
  saçabiliyor; granular yolunda aynı belirti görülmedi. C4 çoklu constitutive
  dispatch kabulüne ilk 20 kare için parçacık sayısı, tepe yoğunluğu ve ortalama
  hız serisi eklenecek. Amaç render yayın paketlenmesiyle gerçek doğum bölgesi
  basınç darbesini ayrı ölçmek.

# 36. parti — C1 birleşik G2P + advect-tail readback (2026-10-01)

**Durum:** CANLI IPC PASS; C1 KAPANDI.

- Vulkan liquid yolunda G2P artık velocity ve affine'i hemen host'a indirmiyor.
  Advect-tail aynı kayıt zincirinde bu device sonucunu tüketiyor ve son
  position/velocity/affine tek transfer batch'inde yayınlanıyor.
- Böylece 100.000 parçacıkta G2P sonrası yinelenen velocity indirmesi olan
  `100000 × sizeof(Vec3) = 1.200.000` byte/kare kaldırıldı; G2P batch fence'i de
  advect-tail fence'iyle birleşti.
- Kapsam bilinçli olarak Vulkan liquid + canlı solid parcel bulunmayan yol.
  Granular state sidecar'ları ve solid parcel host restore sözleşmesi eski güvenli
  yolu koruyor; C4 dispatch bu ayrımı parçacık modeline göre yeniden kuracak.
- Device advect-tail dispatch edilemezse G2P velocity/affine önce host'a geri
  alınır ve Call 2 host tail doğru güncel hızla çalışır. Eski kare hızını okuyup
  sessizce yanlış advection yapmasına izin verilmez.

Derleme sonrası kabul:

1. Uygulama açık, boş sahne ve timeline duraklatılmışken
   `python scripts/test/rt_test_fluid_c1_residency_ipc.py` çalıştır. Script 40³
   Vulkan liquid domain'e 100.000 parçacık tohumlar ve iki eş dört-adım koşusu yapar.
   `fluid.step_stats`: particle count 100.000 ve G2P GPU olmalı.
2. Transfer probunda aynı referansa göre download en az 1.200.000 byte/kare ve
   batch end en az bir adet azalmalı (önce 10.630.584 byte / 13 batch end).
3. Aynı başlangıçtan önce/sonra `fluid.state_digest` count/position/velocity
   hashleri eşit olmalı. Kapalı tank count testi PASS kalmalı.
4. Open boundary ve canlı solid Substance yolu host fallback ile çalışmalı;
   kayıp/NaN veya bir kare eski hız görülmemeli. Bu kapılar geçince C1 kapanır
   ve C2 aktif sıvı çalışma penceresine geçilir.

2026-10-01 canlı sonuç: 100.000 parçacıkta download tam olarak
10.630.584 → 9.430.584 byte/kare, batch end 13 → 12 oldu. G2P GPU kaldı;
ilk ölçümde `total_ms=18,524`. İki eş dört-adım koşusunda count 100.000 kaldı;
centroid en büyük farkı yaklaşık `1,04e-10 m`, mean-speed farkı `5,39e-10 m/s`.
GPU bit hash'i eşit değil; fiziksel fark 1e-6 kabul toleransının çok altında.
Geçici domain kaldırıldı. Sonraki iş C2 aktif sıvı çalışma penceresi.


## Granular authored stiffness and particle residency (2026-10-02)

- Build the application; no shader source changed in this follow-up.
- Empty, paused disposable scene: `python scripts/test/rt_test_granular_stiffness_residency_ipc.py`.
  Checks >64 required substeps, legacy budgets 1/32/64, CPU planner, Vulkan
  closed/open motion/material-coordinate agreement and transfer reduction.
- Run existing granular CPU/Vulkan parity and soft-stability gates on a
  disposable scene, and `rt_test_fluid_c1_residency_ipc.py` on an empty scene.
- Supplied 991666-particle scene: rebuild old cache from an initial/full state;
  expect 48 needed/run substeps and effective/requested Young ~=381300 Pa.
  While fresh simulation advances, run externally:
  `python scripts/test/probe_granular_transfer.py "Grid Domain 1" 15`.
- Compare total time and bytes, not G2P alone: its execution wait may move to Advect.
  Check fallback kernels and cache scrub/resume separately. Details:
  [GRANULAR_SUBSTEP_RESIDENCY.md](GRANULAR_SUBSTEP_RESIDENCY.md).
- Not run by Codex: project builds, live new-code simulation, shader failure
  injection and existing scene cache reset. The open scene was not mutated.

## G2 particle sphere volume floor (2026-10-03)

- Uygulamayı derle; bu turda shader kaynağı değişmedi.
- Başlangıç yoğunluğu kontrolü artık `Seed Particles Per Voxel` olarak görünür.
  Yalnız `Seed Fluid Now` tarifini değiştirir. Point/Object Bounds/Mesh Surface
  flow source yoğunluğu `Injected Particles / Sec` ile yazılır; PPC bu kaynakların
  hızını çarpmaz.
- `granul_test1.rtp` açıldıktan sonra önce **Sand** presetini yeniden seç. Ardından
  yalnız istenen override'ları uygula: friction `52.7°`, Young `840500 Pa`.
  `Custom` etiketi beklenir; cohesion `0`, tensile cutoff `0`, hardening `0`,
  Poisson `0.25`, dilatancy `5`, fracture strain `0.02`, damage rate `8` kalmalı.
- Splat Geometry bölümünde mevcut `radius_factor=0.09`, `size_multiplier=2.61`
  korunabilir. 8 PPC için panel `Effective radius: 0.310 vx (particle volume
  floor)` göstermeli; eski etkin değer `0.235 vx` idi.
- Reset + Seed sonrası durgun yığında ve collider çevresinde büyük karanlık iç
  boşluklar kapanmalı. Taneler bağımsız küre kalmalı; Surface SDF'ye geçmemeli.
- PPC `4` yapıldığında etkin taban yaklaşık `0.391 vx`, PPC `8` yapıldığında
  yaklaşık `0.310 vx` olmalı. Daha büyük elle yazılmış küre boyutu aynen kalmalı.
  Etkin yarıçap canlı çözücünün malzeme-noktası hedefini okumalı; emitter-only
  domainde kullanılmamış seed slider'ını okumamalı. `Scene Object / Mesh Group`
  bu fiziksel sphere tabanından etkilenmemeli.
- Aynı PPC, radius factor ve size multiplier ile Granular anahtarı açılıp
  kapatıldığında etkin procedural-sphere yarıçapı değişmemeli. Hacim tabanı
  liquid ve granular particle görünümlerine ortak uygulanır; panel etiketi
  `particle volume floor` olmalıdır.
- Malzeme davranışı için cohesion/tensile sayaçları kuru Sand'de sıfır kalmalı;
  `stiffness_below_load=false` ve effective/requested Young yakın olmalı.
- Codex derleme veya yeni kodla canlı uygulama doğrulaması çalıştırmadı.

## G3 fluid virtual particle spheres — Vulkan RT + RayFusion (2026-10-03)

- Önce `RayTrophiStudio/compile_shaders.bat` ile
  `fluid_sphere_proxy.vert -> fluid_sphere_proxy.spv` üret, sonra uygulamayı
  derle. Codex talimat gereği shader veya proje derlemesi çalıştırmadı.
- `granul_test1.rtp` içinde Sand presetini yeniden seçip Reset + Seed yap.
  Solid/Scene (RayFusion) görünümünde her fizik taşıyıcısı panelde istenen sayıda
  küçük tanecik olarak görünmeli; hareket sırasında çocuk kümeleri taşıyıcıyla
  kararlı hareket etmeli, titreşen/rastgele yeniden dağılım olmamalı.
- Çocuk yerleşimi artık her taşıyıcıda eksene hizalı aynı küpü tekrarlamaz.
  Taşıyıcı/çift kimliğinden türetilen kararlı 3B yönler ve radyal yayılım
  kullanır. Karşılıklı çiftler küme merkezini taşıyıcının üzerinde tutar.
- Splat Geometry panelindeki `Virtual Grains per Particle` 1..32 aralığındadır;
  `Grain Size Variation` 0..0,75 aralığında deterministik çap farkı verir.
  Karşılıklı çiftlerin küresel hacmi normalize edilir; bu ayarlar solver
  parçacık sayısı, kütle veya yoğunluğu değiştirmez.
- Granular domainlerde `Physical Granular Carriers` varsayılan olarak açıktır.
  Bu modda ana kum gövdesi için `primary_visual_children=1` olmalı ve her
  görünen sphere gerçek bir MPM taşıyıcısını izlemelidir. Kapalı olduğunda sanal
  çocuklar düşük çözünürlük boşluk doldurma önizlemesi olarak yeniden kullanılır.
  Whitewater çocukları bu ana gövde seçiminden bağımsızdır.
- 298³ canlı sahne için eski binary ölçümü: 191.666 fizik taşıyıcısı, 3 etkin
  çocuk ve 574.998 görünür sphere. Yeni build'de aynı sahneyi Reset + Reseed
  ettikten sonra `granular_physical_carriers=true`,
  `primary_visual_children=1` ve ana gövde görünür sayısı 191.666 olmalı.
  Sphere'ler taşıyıcı çevresinde üçlü bağlı kümeler oluşturmamalı. Bu kabul
  gerçek tane-tane yuvarlanma iddiası değildir; MPM taşıyıcılarında orientasyon,
  açısal hız veya DEM temas çözümü yoktur.
- Aynı sahnede `granular_young_modulus=200000`, yük için gereken değer 446.058 Pa
  ve `granular_stiffness_below_load=true` ölçüldü. Görsel A/B ile fizik A/B'yi
  karıştırma. Yük davranışı testinde önce paneldeki gereken değerin üstüne çıkıp
  uyarının kapanmasını doğrula; aynı voxel/dt için yaklaşık 500 kPa değeri wave
  substep sayısını kabaca 80'den 127'ye çıkaracağından toplam süreyi de kaydet.
- `Virtual Grain Budget (millions)` kullanıcı kontrolündedir ve domain başına
  1..32 milyon görünür küre aralığını kabul eder. Etkin çocuk sayısı canlı
  parçacık sayısından değil `Max Particles + Max Foam` kapasitesi ve bu bütçeden
  hesaplanır; simülasyon ilerlerken 8 -> 7 gibi sessiz topoloji değişimi
  olmamalıdır.
- `fluid.get` ve `fluid.list_domains` içindeki `virtual_grains_requested`,
  `virtual_grains_effective`, `primary_visual_children`,
  `whitewater_visual_children`, `granular_physical_carriers`,
  `virtual_grain_count`, `virtual_grain_budget`,
  `virtual_grain_rt_estimated_bytes` ve `grain_size_variation` panelle aynı
  kanonik ayar/istatistiği raporlamalı. RT tahmini hızlandırma yapısı scratch
  belleğini içermez.
- `viewport.frame_telemetry` içindeki `sphere_impostors_drawn` yaklaşık
  `virtual_grain_count` olmalı. Canlı parçacık sayısı artarken
  `virtual_grains_effective` değişmemeli; yalnız kullanıcı çocuk sayısını,
  bütçeyi veya `Max Particles` kapasitesini değiştirdiğinde yeniden hesaplanmalı.
- 2026-10-03 ilk canlı okumada 100.000 parçacık için
  `sphere_impostors_uploaded=100000` ve `sphere_impostors_drawn=100000` görüldü;
  bu doğrudan GPU proxy yolunun değil CPU ebeveyn fallback'inin çalıştığını
  gösterir. Kaynak `fluid_sphere_proxy.vert`, mevcut `.spv` ikilisinden daha
  yeniydi. Shader derlemesinden sonra bu sahnede upload sayısı 100.000 olmamalı,
  draw sayısı yaklaşık 800.000 olmalı.
- Rendered/Vulkan RT görünümünde aynı tane yarıçapı ve aynı sekizli yerleşim
  görünmeli. Tek birleşik procedural-sphere BLAS kullanılmalı; TLAS instance
  sayısı parçacık veya çocuk sayısıyla büyümemeli.
- Bu görsel çoğaltma artık yalnız granular presetine bağlı değildir; prosedürel
  sphere kullanan bütün fluid domainlerinde çalışır. Parçacık hacim tabanı liquid
  ve granular procedural sphere yollarında ortaktır. Çocuk yayılımı yaklaşık
  0,72 voxel hedefler. Her çocuk, en çok 1,8 voxel içindeki kararlı yakın taşıyıcı
  adaylarının uzaklık ve yön ağırlıklı ortalamasına çekilir; aday bulunmazsa
  deterministik serbest yayılım kullanılır.
- Spray, foam ve bubble grupları aynı çocuk sayısı, varyasyon, bütçe ve komşuluk
  sözleşmesini hem Vulkan RT hem RayFusion'da kullanmalıdır. `virtual_grain_count`
  splat'e yönlenmiş whitewater örneklerini de içermelidir.
- Bu tur tam hücre hash'i veya APIC grid hızından yeni örnek üretmez. Sonraki
  kalite aşaması, kararlı örnek kimliğiyle gerçek hücre komşuluk buffer'ını ve
  grid ağırlıklı hız taşınmasını kullanacaktır.
- RayFusion sırasında `fluid_positions` için yeni kare başı CPU upload'u
  oluşmamalı. Granüler `fluid.step_stats` fizik parçacık sayısı, solve transfer
  byte'ları ve G2P/P2G çağrıları bu görsel çarpandan etkilenmemeli.
- Splat/Surface/Fog karışık label sahnesinde yanlış parcel görünürse doğrudan
  pull kabul edilmemiş demektir; CPU filtreli fallback doğru görünümü korumalı.
- OptiX sonucu bu kabulün parçası değildir. OptiX uygulamasından önce CPU
  yerleşim referansı ve Vulkan sonuçlarıyla sayısal merkez/radius karşılaştırması
  eklenecek.
- Ayar ve istatistik sözleşmesi için açık uygulamada dış terminalden
  `python scripts/test/rt_test_granular_render_proxy_ipc.py "Grid Domain 1"`
  çalıştır; test geçersiz aralıkları, get/list read-back'ini ve bütçe sonrası
  toplam görsel tane hesabını doğrular, önceki ayarları geri yükler.
- Kum materyalinde `Object Info > Random` çıkışını bir Color Ramp veya Hue
  zinciri üzerinden Principled Base Color'a bağla. Rendered/Vulkan RT'de her
  procedural sphere farklı bir değer almalı; taşıyıcı ve sanal çocuklar hareket
  ederken kendi renkleri kareler arasında değişmemeli.
- Material Preview/RayFusion'da üçgen-sphere fallback içindeki her taşıyıcı da
  farklı ve kararlı renk almalı. Normal mesh/scatter nesnelerinin eski dünya
  origin'i tabanlı Object Random davranışı korunmalı. Renk özelliği sphere
  kaydını büyütmez; önceden padding olan alanı kullanır ve solver transferlerine
  yeni buffer veya kare başı upload eklemez.
- RayFusion üçgen-sphere fallback'i `Virtual Grains per Particle` sayısını RT
  ile aynı uygulamalı. Çocuk merkezleri/radius değerleri ilk kurulumda ve her
  transform sync sırasında ortak komşuluk çekirdeğinden çözülmeli; collider
  teması veya akış sırasında ana taşıyıcıya çökme ve çocuk kaybı olmamalı.

## 2026-10-04 kabul güncellemesi ve sonraki kapsam

**C2b build kullanıcı tarafından başarılı doğrulandı.** Canlı occupancy/basınç
ölçümü toplu kabulde yapılacak. Yeni kaynak partisi C3a ortak faz erişimi;
kontrol listesi [MATTER_PHASE_GRID.md](MATTER_PHASE_GRID.md).

**Güncel toplu build: C3b + splat taşıma düzeltmesi.** Bağımsız gaz/sıvı
bounds/voxel, ortak UI/Python/IPC authoring, phase budget/allocation,
serializer/cache ve karma preset göçü kaynakta tamamlandı. Tek derleme sonrası
Matter phase probe, pause/taşıma, cache/scrub, save/load ve iki karma preset
aynı kabul turunda kontrol edilecek. C3b C++/canlı kabul henüz yapılmadı.

**C2b occupancy + basınç penceresi kabulü.** Kaynak kod hazır;
kullanıcı test aralıklarını uzatmamızı istedi. Yeni 17 `_window.spv` varyantı
ana shader batch'ine bağlı. Ayrı ayrı mikro kabul yerine
`FLUID_ACTIVE_WINDOW.md` C2b checklist'ini tek partide çalıştır. Yeni alanlar:
`occupancy_on_gpu`, `pressure_window_used`, `pressure_window_cells`.
Önceki C2a canlı PASS bu yeni binary için kabul sayılmaz.

Sonraki kod partisi artık kaynakta: **C2a Vulkan P2G normalize penceresi**.
2026-10-04: kullanıcı derledi; dış IPC canlı matris PASS. Yerel geçici sıvıda
1.200/34.848 hücre, periodic fallback'te pencere kapalı; geçici domain kaldırıldı.
Probe varsayılan domain/hata kontrolü düzeltildi ve Release kopyası güncellendi.
Sayısal eski-build A/B ve C2'nin occupancy/MGPCG kısmı hâlâ açık.
Yeni `sim_fluid_normalize_window.comp` shader'ını ve uygulamayı kullanıcı
derlemeli. Önceki kabul bu yeni partiyi kapsamaz. Kesin checklist:
[FLUID_ACTIVE_WINDOW.md](FLUID_ACTIVE_WINDOW.md). Clear/occupancy/MGPCG henüz
daraltılmadı; C2 açık kalır.

Kullanıcı önceki değişiklikleri derlediğini ve sorunsuz çalıştığını bildirdi.
Önceki partilerdeki derleme bekliyor ifadeleri tarihsel kayıttır. Bu bildirim
sayısal korunum/performans, kolon/repose veya frame-cache resume testlerini
kendiliğinden PASS yapmaz; bu ölçümler ayrı kabul kaydı olarak açık kalır.

İlk plan güncellemesi yalnız belgeleri değiştirmişti; devamında C2a kodu eklendi
ve yukarıdaki yeni shader/C++ derlemesi gerekli oldu. C4 ortak transfer ve
contact sonrasında H1 hibrit yüzey taneleri uygulanır. Kimlik, tek fizik sahibi,
çift impuls ve açısal momentum geçiş kapıları ana planın
`C4–H1 veri ve momentum sözleşmesi (2026-10-04)` bölümünde tanımlıdır.

## C4a büyük altyapı partisi — 2026-10-04

Shader değişikliği yok. C++ yeni kimlik/model transfer modülleri ve UI/API/IPC
kayıtları derlenecek. Kesin kabul checklist'i: [MATTER_TRANSFER_CORE.md](MATTER_TRANSFER_CORE.md).
Disk sim-cache sürümü v10 oldu; eski bake dosyaları yeniden üretilmeli.
C4b karma su-kum solver henüz uygulanmadı; `mixed_transport_ready=false`.
Son kalite notları: gas+SDF Liquid Body parametre tutarlılığı, paused domain
taşıma/splat/cache, eski Liquid Display panel sırası; [MATTER_PHASE_GRID.md](MATTER_PHASE_GRID.md).

C4b ek kontrol: `scripts/test/matter_model_batch_test.cpp` ayrı CPU test hedefinde
çalıştırılmalı; temas hatasında kanonik grid/parçacık rollback ve lane sidecar
kimlik eşlemesi doğrulanır. Bu kaynak test Codex tarafından derlenmedi.

C4b GPU partition kaynak kontrolü: `python scripts/test/check_matter_gpu_contracts.py`.
Tam karma GPU entegrasyonu bekleniyor; şu aşamada yeni derleme istenmiyor.
Son derlemede `sim_matter_partition.comp` shader'ı da SPIR-V'a derlenmeli;
Codex shader veya uygulama derlemesi çalıştırmadı.

CPU referans kabul kaynağı: `scripts/test/matter_mac_contact_test.cpp` fiziksel
yüz-kütlesi momentumu/enerjisi, bozuk layout rollback ve bütçe reddini sınar.
C++ testi henüz derlenmedi veya çalıştırılmadı.


2026-10-04 güncel C4b test noktası: `MATTER_MIXED_GPU_TEST_POINT.md`.
Karma GPU canlı bağlantısı, ortak alt adım, fiziksel P2G kütlesi ve GPU temas
kaynakta eklendi; önceki “bağlı değil” kayıtları tarihsel ilerleme notlarıdır.
Derleme/shader/sahne kabulü henüz kullanıcı tarafından yapılmadı. Sonraki adım
tek derleme ile karma GPU kabulü; C4 tamamlandı etiketi henüz verilmedi.


### 2026-10-05 — C5 büyük kaynak partisi, kabul bekliyor
GPU emilim/drenaj, kanonik pore mass/capacity/porosity/thermal energy sidecar'ları,
ıslak taşıyıcı transport kütlesi, korunum kapılı yayın ve ledger olayları yazıldı.
UI, Python ve IPC ortak authoring servisine bağlı; serializer ve cache v11 hazır.
Bu bir kaynak teslimidir: C++/shader derlenmedi, canlı C5 kabulü yapılmadı.
C5 fiziksel kabulü açık; hücre-local ilk adımın dt/çözünürlük, havuz doluluğu ve
cache round-trip ölçümleri gerekir. C6 wet friction/cohesion/pore-pressure ve
ıslak görünüm henüz uygulanmadı; C7 kalite kabulü açık. İlk scope Closed Vulkan
Water + Sand/Gravel/Soil. Eski cache v10 yeniden bake edilir.
Kesin devir, dosyalar, sınırlamalar ve sonraki kabul adımları:
[MATTER_C5_HANDOFF.md](MATTER_C5_HANDOFF.md).

Kullanıcı shader derlemesinde sim_matter_pores.spv üretmeli; dış probe:
`python scripts/test/rt_test_matter_pores_ipc.py "DOMAIN" --expect absorption`
Sonra drenaj sahnesinde `--expect drainage`. Codex bu turda derleme/probe yapmadı.


2026-10-05 canlı C5 testinde emilim/drenaj gözlendi fakat kabul açık: granül mass initialization liquid_density kullanıyor; tiny drainage births 50k havuzu doldurdu. C4 tam drag/buoyancy ve C5 birth batching öncelikli. Ayrıntı MATTER_C5_HANDOFF.md son bölümünde. Bu turda fizik kaynak değişmedi, yeniden derleme gerekmez.


### 2026-10-05 C6 ilk kaynak partisi (kullanıcı toplu derlemesini bekliyor)
Canonical pore saturation -> wet strength/local head ve 8-band granular splat
appearance; shared UI/Python/IPC authoring, serializer/cache hash ve test kaynakları
bağlı. C6 fiziksel kabul kapanmadı; local head pressure PDE değildir. Toplu shader
ABI: stress_update 18/68, stress_p2g 9/52. Güncel kısa durum/test listesi:
[MATTER_C5_HANDOFF.md](MATTER_C5_HANDOFF.md); model sınırları:
[MATTER_WET_RESPONSE.md](MATTER_WET_RESPONSE.md).
