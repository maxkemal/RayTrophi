# 719 "unspecified launch failure" uydurma bir koddu

> **Durum:** ARŞİV — kök bulundu, düzeltildi ve çalışma anında DOĞRULANDI (2026-09-17). Ders kalıcı.

## Belirti

Solid / RayFusion viewport modundan **doğrudan** OptiX'e geçişte uygulama
anında ölüyordu:

```
[ERROR] CUDA error: unspecified launch failure (719) at ...OptixWrapper.cpp:1879
[FATAL] std::terminate called without active exception
```

Vulkan RT → OptiX geçişi **çökmüyordu.** Satır numarası her oturumda
değişiyordu: 1879 → 1930 → 1977 → 2033 → 2034.

## Üç ayrı yanılgı, tek kök

### 1. Satır numarası hiç kaymadı

Numaranın değişmesi "hata geziniyor" diye okundu ve *asenkron, yapışkan bir
CUDA hatası* varsayımına götürdü. Gerçekte **hep aynı satırdı**; dosya
düzenlendikçe numarası kaydı. Teşhis aracı olarak bakılan şey, teşhis edenin
kendi düzenlemelerini ölçüyordu.

### 2. 719 bir ölçüm değil, bir atamaydı

```cpp
CUDA_CHECK(launchOptixDisplayPost(...) ? cudaSuccess : cudaErrorLaunchFailure);
```

`cudaErrorLaunchFailure` **sabit 719**'dur ve metni "unspecified launch
failure"dır. Yani `launchOptixDisplayPost` hangi sebeple `false` dönerse dönsün,
log bir **cihaz arızası** bildiriyordu. Fonksiyonun ilk satırı ise:

```cpp
if(!hdr || !display || width<=0 || height<=0) return false;
```

Bu yalnızca bir **argüman kontrolü** — ortada CUDA hatası yok. Solid → OptiX
doğrudan geçişinde tamponlar henüz tahsis edilmemiş oluyor, koruma devreye
giriyor, ve zararsız bir "bu kareyi atla" durumu cihaz çöküşü olarak
raporlanıyordu.

> **Ders:** bir `bool`u hata koduna çevirirken **var olan** bir kod seçme.
> `false` → `cudaErrorLaunchFailure` eşlemesi, bilgi kaybını bilgi gibi
> gösterdi. Bu, deponun *"Varsayılan bir ölçüm değildir"* dersinin tam
> kardeşi.

### 3. `CUDA_CHECK` `throw` etmiyor, `std::terminate()` çağırıyor

Bu yüzden `terminate` ile 719 **ayrı iki arıza sanıldı.** Terminate işleyicisi
`std::current_exception()` boş bulup şunu bastı:

```
[STATE] no CUDA error latched at terminate -> terminate is an INDEPENDENT fault
        (typical cause: a joinable std::thread destroyed without join/detach).
```

İki kat yanlıştı:

- Ortada **exception yoktu** çünkü makro fırlatmıyor, doğrudan öldürüyor.
- İlk-hata mandalı **boştu** çünkü `CUDA_CHECK` mandala hiç uğramıyordu —
  mandal yalnızca `rtNoteCudaError()` çağrılan yerlerde doluyor.

Sonuç: enstrüman, kendi ölçemediği bir arıza için **olumlu teşhis** bastı ve
joinable-thread avına yolladı.

> **Ders (deponun kendi kuralı):** *"Tripwire'ın susması yokluğu kanıtlamaz."*
> Susan bir tripwire'ın "bağımsız arıza" gibi **pozitif bir sonuç** yazması ise
> düpedüz zarardır.

## Bulunan ikinci kusur: serbest bırakılmış ama null olmayan işaretçi

Aynı yolda, kontrolsüz tahsis:

```cpp
if (d_framebuffer) cudaFree(d_framebuffer);
cudaMalloc(&d_framebuffer, ...);      // dönüş değeri OKUNMUYOR
```

`cudaMalloc` başarısız olursa `d_framebuffer` **az önce serbest bıraktığımız
adresi tutmaya devam eder** — null değil, dolayısıyla sonraki her kontrole göre
"canlı", ve doğrudan `optixLaunch`a veriliyor. `d_accumulation_float4` için de
aynısı. Bu, bu oturumda TDR kurtarma yolunda görülen desenin aynısı:
**non-null, canlı demek değildir.**

## Düzeltme

| Yer | Değişiklik |
|---|---|
| `src/Device/post_exposure.cuh` | `launchOptixDisplayPost` → **`runOptixDisplayPost`**, dönüş `bool` → `cudaError_t`. Argüman hatası artık `cudaErrorInvalidValue`; gerçek hata `cudaGetLastError()`ten aynen geçer |
| `include/oidn_blend_cuda.h` | Bildirim güncellendi. **Ad değişti**, çünkü `bool`→`cudaError_t` sessiz bir anlam tersine çevirmesi olurdu (`cudaSuccess == 0`, yani eski `if(...)` çağrısı tam ters çalışırdı) |
| `src/Render/OptixWrapper.cpp` | İki çağrı yeri artık ölmüyor: `reportDisplayPostFailure()` oranı sınırlı log basar, gerçek cihaz arızasını mandala yazar, kareyi atlar. Sunum adımı süreç değerinde değildir |
| `src/Render/OptixWrapper.cpp` | `d_framebuffer` ve `d_accumulation_float4` tahsisleri kontrol ediliyor; işaretçi malloc'tan **önce** null'lanıyor; başarısızlıkta kare temiz atlanıyor |
| `src/Render/OptixWrapper.cpp` | `CUDA_CHECK` artık ölmeden **önce** `rtLatchCudaError(err, "CUDA_CHECK <dosya>:<satır>")` çağırıyor. Yeni `rtLatchCudaError` açık bir kod alır; `rtNoteCudaError` onun `cudaGetLastError()` saran hali |
| `src/Core/Main.cpp` | `[STATE]` mesajı artık teşhis uydurmuyor: boş mandalın hiçbir şey kanıtlamadığını söylüyor ve `[STACK]` karelerine yönlendiriyor |

## Kalıcı ders

Bu arızanın üç katmanı da **ölçü aletinin kendisindeydi**, ölçtüğü sistemde
değil:

1. Satır numarası ölçenin düzenlemelerini ölçüyordu.
2. Hata kodu ölçülmemişti, atanmıştı.
3. Mandal, kapsamadığı bir yol için olumlu teşhis basıyordu.

Bir ölçüm sistemde bir şey bulmuyorsa, ilk soru **"aletin kapsamı bu yolu
içeriyor mu"** olmalı.

## Ek (aynı gün, ilk çalıştırma): mandalın kendisi de yanlış yeri suçladı

Düzeltmeden sonraki ilk turda **çökme olmadı**, ama şu düştü:

```
[CUDA] FIRST error surfaced here: partialCleanup: sync before releasing
       -> invalid argument (1).
```

İki ayrı kusur daha çıktı, ikisi de yine **alette**:

**a) Site ölçmüyordu.** Kalıp şuydu:

```cpp
cudaStreamSynchronize(stream);                        // dönüş OKUNMUYOR
rtNoteCudaError("partialCleanup: sync before releasing");
```

`rtNoteCudaError` içeride `cudaGetLastError()` çağırır — bu, senkronizasyonun
sonucu değil, **thread'in hata durumudur**: daha önce herhangi bir yerde
bırakılmış bir hata. Yani alakasız, zararsız bir `cudaErrorInvalidValue` bu
siteye **atfedildi.** Dört senkronizasyon sitesi de artık kendi dönüş değerini
mandala veriyor, ve kodu elinde olan diğer siteler açık kod alan
`rtNoteCudaError(cudaError_t, const char*)` aşırı yüklemesini kullanıyor.

**b) Yanlış atama gerçek arızayı kalıcı olarak maskeliyordu.** Mandal
"ilk kazanır" mantığındaydı: zararsız bir hata yerleştiği anda, aradığımız
**gerçek 719 bir daha hiç yazılamazdı.** Artık yapışkan bir hata, yerleşmiş
yapışkan-olmayanı bir kez yerinden edebiliyor ve logda bunu söylüyor.

**c) Metin fazla iddialıydı.** "Every CUDA call AFTER this returns the same
error" yalnızca **yapışkan** hatalar (719, 700, 702, ECC…) için doğrudur.
`cudaErrorInvalidValue` yapışkan değildir — bağlam kullanılabilir durumdadır.
Mandal artık `rtCudaErrorIsSticky()` ile ayırıyor ve yapışkan olmayan için
açıkça *"bu bir çökmeyi AÇIKLAMAZ"* diyor.

> **Ders:** aynı gün, aynı alet, aynı sınıf hata **dört kez**. Bir teşhis
> mesajı yazarken sorulacak soru "doğru mu" değil, **"yanlış olduğunda okuyanı
> nereye yollar"**dır. Bu mandal sırasıyla: olmayan bir thread arızasına,
> masum bir temizlik fonksiyonuna, ve neredeyse gerçek arızanın tamamen
> gizlenmesine yolladı.

## Doğrulama ve kapanış (2026-09-17)

Kullanıcı derledi ve çalıştırdı: **loglar sustu, hatalı log yazılmıyor,
çökme yok.** Konu kapandı.

Neyin kanıtlandığı konusunda dürüst olmak gerekir:

- **Kanıtlandı:** uydurma 719 artık üretilmiyor; `partialCleanup`a yapılan
  yanlış atfetme ortadan kalktı; Solid/RayFusion → OptiX geçişi süreci
  öldürmüyor.
- **Kanıtlanmadı:** sessizlik, ortada hiç CUDA hatası olmadığını göstermez —
  yalnızca mandalın artık *olmayan* bir hatayı uydurmadığını gösterir. Gerçek
  bir yapışkan hata doğarsa mandal onu adıyla yazacak, ve gerekirse yerleşmiş
  yapışkan-olmayanı yerinden edip bunu logda söyleyecek.

Bu ayrım, bu arızanın bütün dersini özetliyor: **susan bir alet ile doğru
ölçen bir alet aynı çıktıyı verebilir.** Farkı yalnızca aletin kapsamını
bilerek okuyabilirsin.
