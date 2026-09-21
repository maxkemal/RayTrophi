# Gaz adımının gerçek maliyeti: taşıma, hesap değil

> **Durum:** AKTİF — 2026-09-20'de ölçüldü, 2026-09-21'de ilk yarısı uygulandı.
> İKİ ÖNEMLİ DÜZELTME için aşağıdaki "2026-09-21 düzeltmeleri" bölümüne bak:
> round-trip'lerin hepsi israf DEĞİL, ve o bayrak `true` YAPILIYOR.

## Ölçüm

160×247×160 (6,32M hücre) nükleer gaz domain'i, adım başına **525,99 ms**:

| | ms | % |
|---|---|---|
| GPU aşamaları (9 kalem) | 312,94 | 59,5 |
| Host çözücü (`GridFluid::step`, çoğu `boundaries+solids` 40,46) | 51,55 | 9,8 |
| **`field analysis scan` (host)** | **115,52** | **22,0** |
| Hiçbir satıra yazılmamış | ~46 | ~8,7 |

## ★★★★★ Kök neden: aşama başına host gidiş-dönüşü

Yedi GPU aşamasının **hepsi** aynı kalıpta:

```
uploadBuffer(girdi...)  →  dispatch  →  synchronize()  →  downloadBuffer(çıktı...)
```

Sayım: adım başına **35 upload + 20 download**. Bu domain'de her grid alanı
~25 MB. Yani **1 GB'ın üzerinde PCIe trafiği ve yedi tam boru hattı durması**,
her biri birkaç milisaniyelik çekirdekleri çalıştırmak için.

Ve art arda gelen aşamalar, bir öncekinin az önce indirdiğini **tekrar
yüklüyor**. GPU compute'un %20'de, host'un %30'da olmasının sebebi bu: host
hesaplamıyor, **kopyalıyor**.

## ★★★★★ Ve mekanizma ZATEN KURULMUŞ — hiç açılmamış

`SimulationGridDomainComputeBuffers::gpu_resident_fields_valid`:

- tanımı `= false`
- **15 yerde `false` atanıyor**
- 5 yerde okunuyor (`runGpuScalarAdvection`'daki yükleme atlama, 6994'teki
  yayınlama indirmesi, …)
- ve **hiçbir yerde `true` yapılmıyor**

Yani hızlı yolların hiçbiri hayatında bir kez bile çalışmadı. Bu, bu deponun
tekrar eden hata sınıfı: *üretici ≠ tüketici*, ve **varsayılan bir ölçüm
değildir** — sadece `false` olabilen bir bayrak, "yerleşik değil" diye
raporluyor ve kimse sorgulamıyor.

## Neden bu partide uygulanmadı

Denendi ve **geri alındı.** Velocity advection'ın indirmesi kaldırılıp handle
takasına çevrildiğinde ağaç bozuluyor: `runGpuScalarAdvection` hâlâ host'tan
velocity yüklüyor (ölü bayrağa baktığı için), yani cihazdaki taze advect
sonucunun üzerine bayat host kopyasını yazıyor. Çökmez — **sessizce yanlış
fizik** üretir.

★ Doğru uygulama, alan başına üç durumlu bir defter ister (yalnız-host /
ikisi-aynı / yalnız-cihaz) **ve** host'un grid'e yazdığı her yerin
işaretlenmesini. Host yazan yerler: `GridFluid::step` (boundaries, dissipation),
gaz enjeksiyonu, akış kaynakları, reset/resize. Bunlar işaretlenmeden defter
yalan söyler.

★★ Muhafazakâr ve güvenli kural: her **host aşaması çağrısından sonra** defteri
tamamen geçersiz kıl. Beş civarı çağrı yeri, düzinelerce yazma noktası değil;
fazladan geçersiz kılar (kazancın bir kısmını kaybeder) ama asla bayat okumaz.

## Sıralama önemli: önce taşıma, sonra alt adımlama

CFL 3,10 ölçüldü ve detay tavanı orada (bkz. `NUKLEER_PRESET.md`). Ama
advection'ı alt adımlamak, **mevcut halde çekirdekleri değil taşımayı
çoğaltır** — 4 alt adım, 4 kat upload/download demek. Yerleşiklik girmeden
alt adımlama ölçülürse yanlış şey ölçülmüş olur.

Sıra: **1)** yerleşik alanlar → **2)** tekrar ölç → **3)** advection alt adımlama.

## Bu partide yapılan

`field analysis scan` paralelleştirildi ve div/mod kaldırıldı:

- z dilimi başına yerel toplam, dilim başına bir OpenMP iterasyonu. MSVC'nin
  OpenMP 2.0'ında `max`/`min` reduction yok, dolayısıyla dilim-parçalı biçim
  hem taşınabilir hem hızlı olanı.
- x/y/z artık döngü sayaçları — aktif hücre başına tam sayı bölme/modulo yoktu.
- Hız maksimumu da parçalı paralel; `max` sıradan bağımsız olduğu için sonuç
  **bit-aynı**.
- ⚠ `total_density` ve `total_fuel` toplama SIRASI değişti. 6,3M float'ın
  toplamı olduğu için eski tek-akümülatör sırasıyla bit-aynı DEĞİL; parçalı
  toplam daha doğru olanı. Panelde son hanelerin oynaması beklenir ve normaldir.


---

# 2026-09-21 düzeltmeleri — iki iddiam yanlıştı

## ★ 1. Round-trip'lerin hepsi gereksiz değil

Adım sırası çıkarıldı ve şunu gösteriyor:

```
... GPU: velocity advect -> scalar advect -> combustion -> body forces
    HOST: GridFluid::step  (boundaries + solids, 40 ms)      <-- BURADA
    GPU: dissipation -> pressure -> publish -> majorant
```

`GridFluid::step` **her adımda, GPU zincirinin ortasında** koşuyor ve grid'e
yazıyor. Yani o noktadaki bir download/upload çifti **yapısal**, israf değil.

İsraf olan, *ardışık GPU aşamaları arasındaki* tekrar. "Yedi round-trip"in
hepsini kaldırmak, önce `boundaries + solids`'i GPU'ya taşımayı gerektirir —
ve o portun var olduğu ama devrede olmadığı zaten not edilmişti.

## ★★ 2. `gpu_resident_fields_valid` HİÇ `true` YAPILMIYOR demiştim — yanlış

Yapılıyor: satır ~10691, adım **sonunda**, RT köprüsü için skaler alanlar
yayınlandıktan sonra `= fields_published`. `= true` diye aradığım için kaçırdım.

Gerçek arıza daha ince ve daha öğretici: bayrak adım **sonunda** set ediliyor,
adım **başında** temizleniyor. Adım-sonrası tüketiciler (snapshot/download
yolları) onu doğru görüyor. Ama `runGpuScalarAdvection`'ın hızlı yolu tam
ortada çalışıyor, yani **her domain'in her adımında false okuyor**.

> Ders, bayrağın sahte olması değil — **set edildiği yer ile okunduğu yer
> arasındaki boşluk.**

## Bu partide uygulanan: yükleme defteri

Alan başına tek bit, anlamı "cihaz kopyası güncel". Upload sonrası set,
download sonrası set (iki kopya artık aynı), host yazınca temizlenir.
**İndirmeler kasten yerinde bırakıldı**: host kopyası her zaman tazelendiği
için kaçırılmış bir geçersiz kılma, sessiz bayat okuma değil **fazladan bir
upload** maliyetine dönüşür.

Uygulananlar:

- `runGpuVelocityAdvection`: yüklemeler deftere bağlandı; sonuç scratch'te
  olduğu için **handle takası** yapılıyor (cihazlar arası kopyalama API'si yok,
  takas bedava). Artık cihaz ve host aynı fikirde, sonraki aşama upload atlıyor.
- `runGpuScalarAdvection`: ölü kapı yerine gerçek defter. Bu tek değişiklik
  adım başına üç yüz arayüzü yüklemesini kaldırıyor.
- `GridFluid::step` / `stepSparseVDB` sonrası **tam geçersiz kılma**. Defterin
  bağlı olduğu tek kritik nokta bu; kaçırılırsa duvarlar sessizce çalışmaz.
- Adım başı ve her reset/resize noktasında geçersiz kılma (13 yer eşlendi).

⚠ Yalnızca VelX/VelY/VelZ dönüştürüldü. Skaler alanların yüklemeleri hâlâ
koşulsuz — ve bu **kasıtlı**: bir aşama cihaz tamponunu yazıp indirmiyorsa,
defterin o alan için bir şey iddia etmemesi gerekir. Ölçüm geldikten sonra
alan alan genişletilecek.

## Ölçüm aleti

`gas.step_stats` açıldı (çekirdek API + IPC + Python + yetki + descriptor).
Panelin bastığı bütün aşama satırları artık script'ten okunabiliyor, yani
optimizasyonun A/B'si ekran görüntüsü yapıştırmadan yapılabilir.

★ `measured` alanı yük taşıyor: her süre hem "bu aşama bedava" hem "adım
koşmadı" için 0.0 döner, ve bunu atlayan bir çağıran boşta duran bir domain'i
0 ms'lik bir adım olarak kaydeder — bir A/B'de bu **tam kazanç** gibi okunur.

---

# 2026-09-21 — donanım mı, metodoloji mi? Ölçüldü: metodoloji

Soru kullanıcıdan geldi ve doğru soruydu: *"bu değerler donanım sınırına mı
çarpıyor yoksa metodolojik bir yanlışlık mı? Bunu çözmeden optimizasyon boşuna
emek olur."*

## 1. Sabit gecikme değil, iş

Çözünürlük taraması, aynı sahne:

| voxel | hücre | total ms | ns/hücre |
|---|---|---|---|
| 0.17 | 3 380 000 | 204.5 | 60.5 |
| 0.24 | 1 201 888 | 94.2 | 78.4 |
| 0.34 | 422 500 | 35.1 | 83.1 |
| 0.48 | 150 236 | 12.7 | 84.4 |

Hücre 22.5 kat azalırken süre 16 kat azalıyor. Sabit senkronizasyon maliyeti
hâkim olsaydı küçük gridde ns/hücre **patlardı** — patlamıyor. Yani gecikme
sınırında değiliz.

## 2. Hesap değil, taşıma — ve ölçü aleti zaten elimizdeydi

| kernel | veri | süre | efektif bant |
|---|---|---|---|
| `sim_gas_majorant` — **cihazdan çıkmaz** | 25.8 MB | 0.202 ms | **125 GB/s** |
| combustion | ~103 MB | 13.8 ms | 7.3 GB/s |
| velocity advect | ~129 MB | 18.8 ms | 6.7 GB/s |
| scalar advect | ~103 MB | 21.4 ms | 4.7 GB/s |

Aynı GPU, aynı grid, aynı adım. **17–27 kat.** 5–7 GB/s aralığı VRAM değil,
PCIe bile değil — host memcpy + staging aralığı.

★★★ **Metodolojik hata bendeydi.** `GasStepStats`'ın kendi başlığı uyarıyor:
*"it is NOT isolated kernel time, and reading it as such will make a
transfer-bound stage look compute-bound."* Ben tam olarak bunu yaptım: aşama
sürelerine bakıp hedef sıraladım ve **seyrek dispatch**'e yöneldim. Seyrek
dispatch kernel işini azaltır; kernel işi toplamın ~%3'ü. 13 ms kovalıyordum,
17 kat dururken.

## 3. Seyrek dispatch ÖLÇÜLDÜ ve elendi

Faz A (majorant kernel'ine aktif blok listesi) kuruldu ve ölçtü:

- hücre doluluğu **%43.1**
- **blok doluluğu %89.7** (6482 / 7225)

8³ blokla kazanç tavanı %10. Transferler kalktıktan sonra bile %10. **8³ ile
bu iş her iki senaryoda da ölü.** Faz A boşa gitmedi: bir dalı ölçümle kapattı
ve `active_blocks` sayacı bedava kaldı.

## 4. Defter neden kazanç vermemişti

★★★ **On beş `invalidateDeviceCopies()` çağrısının dokuzu doğrudan bir
indirmenin ardından geliyordu.** İki mekanizma aynı sanılmıştı:

```
gpu_resident_fields_valid = "cihaz kopyası OTORİTER"
defter biti               = "cihaz kopyası host ile EŞİT"
```

Bir indirmeden sonra birincisi false, **ikincisi true**'dur. Defter her
aşamada siliniyordu, yani bir sonraki aşama önceki aşamanın az önce geri
verdiği alanı yeniden yüklüyordu.

★★★★ Ve ikinci bir örnek: `runGpuScalarAdvection`'ın `scalar_inputs_resident`
kapısı **okunduğu anda her zaman false**'tu — 10614'te tanımlanıyor, **10730'da
okunuyor**, 10735'te atanıyor. Üstelik beslendiği değer başka bir aşamanın
bayrağıydı ve tek boolean üç alan için cevap veriyordu.

> İkisi de aynı şekil: **bayrağın SET edildiği yer ile OKUNDUĞU yer arasındaki
> boşluk.** Aynı dosyada iki kez.

## 5. Bu partide yapılan ve YAPILMAYAN

Yapılan: dokuz indirme yerinde defter artık doğru işaretleniyor (hata
yollarında geçersiz kılma korunarak), combustion ve scalar advection defteri
okuyor, ve advection sonrası `swap` bir **host yazması** olarak kaydediliyor.

⚠ Bu sonuncusu olmasaydı defter tek tehlikeli yönde yalan söylerdi: bir sonraki
aşama yüklemeyi atlar ve **advection öncesi** alanı ilerletirdi. Çökme yok,
uyarı yok — her yerde bir adım geride bir alan.

YAPILMAYAN: **indirmeler duruyor.** Scalar advection sonucu hâlâ host'a inip
swap edildiği için combustion üç alanı yeniden yüklemek zorunda. Ödülün tamamı
GPU alt zincirinin cihazda kalmasını ve host sınır aşamasından önce **tek bir
indirme** yapılmasını gerektiriyor. Sıradaki parti.


---

## 2026-09-21 — Cihaz yerleşik alt zincir: DOĞRULANDI

### 1. Fizik birebir aynı — ölçüldü

`gas.reset` + 120 kare, tek değişken `device_resident_chain`:

| kare | hücre | fill | tepe | merkez | peakT | meanT |
|---|---|---|---|---|---|---|
| 40 | 211 742 | 0.06265 | 16.150 | 11.292 | 8.0034 | 3.41141 |
| 80 | 721 307 | 0.21340 | 31.790 | 22.530 | 5.5467 | 2.27425 |
| 120 | 959 364 | 0.28384 | 34.000 | 28.940 | 3.8441 | 1.29586 |

Açık ve kapalı değerler **son haneye kadar aynı**. Kullanıcı ayrıca görsel
olarak takip etti: bozulma yok, kuvvet alanlarının etkileri doğru.

★ Bu testin önemi, bir önceki turda **başarısız olmasıyla** kanıtlandı:
`runGpuTurbulence` combustion'ın cihazdaki sonucunu bayat host kopyasıyla
eziyordu ve f120'de 959 364 yerine 96 826 hücre çıkıyordu. Anahtarın bake
imzasına **kasten dahil edilmemesi** bu yüzden doğru karardı.

### 2. Kazanç

| | açık | kapalı |
|---|---|---|
| `gpu_combustion_ms` | **6.79** | 14.09 |
| `total_ms` | **212.83** | 218.15 |
| `cpu_total_ms` | 31.01 | 32.26 |

Combustion **yarıya indi** (−4 indirme × 12.89 MB). Adımın tamamında kazanç
yalnızca ~5.3 ms (%2.4), ve muhasebesi tam olarak beklendiği gibi:

* combustion'dan 4 indirme kalktı (−51.6 MB),
* buoyancy 2 yüklemeyi atladı (−25.8 MB),
* `gasSyncGridToHost` aynı 4 indirmeyi **geri ekledi** (+51.6 MB),
* net −25.8 MB ≈ 3.7 ms — ölçülen 5.3 ms.

> **Yani yapısal geri okuma durdukça ödül küçük kalır.** `boundaries + solids`
> host'ta ve zincirin ORTASINDA olduğu sürece bu indirme ödenmek zorunda.
> Sıradaki adım bu yüzden hızlandırma değil, o aşamanın GPU'ya taşınmasıdır.

### 3. ⚠ Ölçü aletinde kör nokta

`gasSyncGridToHost` iki `gas_gpu_mark` arasında değil, **hiçbir satıra
yazılmayan bir boşlukta** çalışıyor: `cpu_total_ms` büyümedi, ama `total_ms`
onu taşıyor. Bir sonraki optimizasyon turundan ÖNCE buna kendi satırı verilmeli,
yoksa yapısal geri okumanın kaldırılması **ölçülemez.**

### 4. Blok doluluğu (Faz A sayacı, ilk kez canlı)

`active_blocks / total_blocks = 6472 / 7225 = %89.6` — daha önce hesaplanan
%89.7 ile aynı. **Seyrek dispatch kararlı şekilde elendi** (8³ blokla kazanç
tavanı %10). Sayacın kendisi duruyor; blok boyutu değişirse karar yeniden
verilebilir.

### 5. Anahtar söküldü (aynı gün, doğrulamadan sonra)

`device_resident_chain` görevini tamamladı: bir regresyonu yakaladı
(`runGpuTurbulence`) ve ardından eşitliği kanıtladı. Kural 5 gereği ikinci yol
söküldü — combustion'ın indirme dalı artık yok, alan API/IPC/Python/proje
dosyasından kaldırıldı.

⚠ Eski projelerde `gas_device_resident_chain` yazılı kalabilir; okuyucu
söküldüğü için **sessizce yok sayılır** ve yerleşiklik her zaman açıktır.

★ Karşı görüş kayıt altına: bu anahtar aynı zamanda bir **ölçü aletiydi** ve
gerçek bir hatayı o yakaladı. Benzer bir veri yolu değişikliği yapılırsa
anahtarı **yeniden koymak**, kalıcı bırakmaktan daha doğrudur: geçici bir
karşılaştırma aleti, kalıcı bir ikinci kod yolu değil.

### 6. Ölçü aletindeki kör nokta kapatıldı

`gpu_host_sync_ms` eklendi (header → API → IPC → Python → panel). Panelin
`gpu_sum` toplamı da bu satırı içeriyor, yani `phase_sum` ile `total_ms`
arasındaki açıklanmayan fark kapanıyor.

---

## 2026-09-21 (akşam) — Host taramaları kaldırıldı: 213 → 163 ms

Cihaz yerleşikliği bittikten sonra kalan host kalıntısına bakıldı ve **teşhis
ilk bakışta yanlıştı**: "host'ta `boundaries + solids` kaldı" diyordum. Ölçünce
ikisi de bu sahnede bedavaydı — `setWallBcs` ilk satırında dönüyor (boundary
"open") ve `enforceSolidBoundaries` boş katı listesinde erken çıkıyor.

23 ms başka bir şeydi: **`clampVelocity`**, `vel_x/y/z` üzerinde ~10.3M float'lık
seri bir tarama. Ve ikinci bir tanesi daha vardı: `sanitizeProjectedVelocity`,
GPU projeksiyonundan sonra host'ta, aynı büyüklükte.

★ **İkisi de adı başka şey olan satırlara yazılıyordu** — birincisi
`cpu_boundary_ms`'e, ikincisi `gpu_publish_ms`'e. Yani adımın en pahalı iki host
işi, ikisi de kendilerini açıklamayan etiketlerin altında duruyordu. Optimizasyon
hedefini zamanlama tablosundan seçerken bu ders ikinci kez geldi.

**Yapılan:** ikinci tarama projeksiyonun **indirmesinden önce** cihazda bir
dispatch'e dönüştü (`sim_grid_velocity_dissipate_clamp`, `factor = 1.0` ile saf
clamp+scrub) — veri zaten oradaydı ve zaten inecekti, yani **transfer maliyeti
sıfır**. Birincisi `skip_pressure_projection` açıkken atlandı: gerekçesi
"projeksiyon sıçraması" ve orada projeksiyon hiç çalışmıyor.

| satır | önce | sonra |
|---|---|---|
| `cpu_boundary_ms` | 23.07 | **0.00** |
| `gpu_publish_ms` | 26.20 | **4.53** |
| `cpu_total_ms` | 31.01 | **6.42** |
| `gpu_host_sync_ms` | (yoktu) | **7.06** |
| `total_ms` | 212.83 | **163.30** |

Fizik f40/f80/f120'de birebir aynı kaldı.

### Kalan dağılım (163.30 ms)

| aşama | ms |
|---|---|
| pressure projection | 46.52 |
| velocity advection | 26.97 |
| scalar advection | 19.44 |
| body forces | 14.51 |
| host readback (`gpu_host_sync_ms`) | 7.06 |
| host solver kalıntısı (`cpu_total_ms`) | 6.42 |
| combustion | 5.46 |
| velocity dissipation | 5.86 |
| source upload | 4.72 |
| publish | 4.53 |
| analysis scan | 4.61 |

Sıradaki: host kalıntısı artık yalnız surface dust (~3.2) + skaler dissipation
(~2.9). İkisi GPU'ya geçerse `GridFluid::step` gaz yolunda tamamen atlanabilir ve
7.06 ms'lik yapısal geri okuma da kalkar. Ondan sonrası pressure projection'dır
ve **transferler kalktığına göre artık yeniden ölçülmesi gerekir.**

### ★★★ Ve bir regresyon: sıvı tutuşmuyordu

MSF pyrolysis ile sıvı yüzey yanması gaz alanlarını **doğrudan cihaz
tamponlarına** yazıp geriye hiçbir şey vermiyor. Defter bu üreticileri
tanımıyordu, dolayısıyla skaler advection "yükle" cevabını alıp **bayat host
kopyasını depozitin üstüne yazıyordu.** Sıvı buharlaşan yakıtı ve ısıyı, hiçbir
şeyin yakamadığı bir tampona bırakıyordu — tutuşma eşiği hiç kapı değildi, o
yüzden 0'a çekmek işe yaramadı.

Bu, kaldırılan `scalar_inputs_resident` parametresinin tam olarak engellediği
şeydi. Kaldırmak doğruydu (tek boolean üç alan için cevap veremez) ama **defter
ancak ÜRETİCİ de bağlıysa çalışır** ve bu ikisi bağlanmamıştı.

★ Yan bulgu: depozitler `interaction` (alev) kanalını okuyup yazıyor ama
öncesindeki yükleme bloğu onu hiç yüklemiyordu — alev bayat bir cihaz kopyasının
üstüne ekleniyordu. O bloğun yorumu da "(fuel, temp, density, **vel**)" diyordu ve
velocity'yi hiç yüklemiyordu.

Doğrulandı: `burning_cells` 3037/adım, preset'ler tutuşuyor, `total_ms` 163.30 —
`interaction` yüklemesinin bedeli ölçüm gürültüsünün içinde kaldı.

