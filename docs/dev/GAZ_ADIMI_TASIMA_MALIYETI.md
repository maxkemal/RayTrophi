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
