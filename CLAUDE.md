# RayTrophi Studio — ajanlar için çalışma kuralları

Bu dosya bu depoda çalışan her ajan tarafından otomatik okunur. Kurallar
tercih değil; her biri bir kez pahalıya öğrenildi.

---

## ★★★ 1. Her yeni yetenek script/IPC tarafına da açılır

**Bu projenin en önemli kuralı budur.** Bir özellik yalnızca panelden
erişilebiliyorsa, o özellik **test edilemez** sayılır.

Gerekçesi ölçekte: ~1 milyon satır kod, binlerce parametre, ve projeyi
sürdüren **tek bir kişi**. Manuel test ancak çok uzun aralıklarla yapılabilir.
IPC/script katmanı bir kolaylık değil, bu projenin **QA altyapısıdır** —
ajanların uygulamayı dışarıdan sürebilmesi için kuruldu.

Her yeni yetenek için gereken **dört** dokunuş:

| Katman | Dosya |
|---|---|
| Çekirdek API | `source/include/Api/RtApi.h` + `source/src/Api/RtApi*.cpp` |
| IPC dispatch | `source/src/Api/RtIpc*.cpp` |
| Python binding | `source/src/Api/RtPython*.cpp` |
| Yetki (capability) | `source/src/Api/RtIpcSecurity.cpp` |

Üçü yapılıp biri unutulursa hata almazsın: `authorize()` **fail-closed**
çalışır, yani metot sessizce reddedilir. `scripts/audit_ipc_capabilities.py`
bu ikisini karşılaştırır; yeni bir namespace eklediysen çalıştır.

**Tek istisna:** `rt.ui` — panel çizimi scriptlenmez.

★ Bir yeteneği yalnızca UI'da bırakmak, zincirin ortasına **manuel bir adım**
koymaktır. Fracture üretimi tam olarak böyleydi: script bir objeyi yakabiliyor
ve shard'ları kırılabilir yapabiliyordu, ama kesme işlemi yalnızca bir düğmeydi
— yani her otomatik test yarıda kesilip insana devrediliyordu.

---

## 2. Build'i KULLANICI alır

`msbuild`, `cmake`, `dotnet build` **çalıştırma**. Derleme kullanıcının
makinesinde ve onun kontrolünde. Kodu yaz, ne test edileceğini söyle, bırak.

Yeni bir `.cpp` eklediysen `.vcxproj`'a da eklemeyi unutma.

## 3. Her iş partisini SIRALI kontrol listesiyle bitir

`docs/dev/NEXT_BUILD_CHECKS.md`'ye yaz (her partide üzerine yazılır). Sıralama
ölçütü: **bağımsız ve hızlı görülen önce**, diğerlerinin sonucunu maskeleyen
sonra. Her madde için hem "ne görmen gerek" hem "bozuksa ne demek" yaz.

★ En sinsi başarısızlığı ayrıca işaretle: sessizce makul görünen sonuç. Onu
kimse bug diye raporlamaz.

## 3b. Mühendislik notları `docs/dev/` altında ve DURUM ETİKETLİdir

`docs/` kullanıcıya bakan HTML kılavuz; yol haritası, mimari kararı, kabul testi
ve postmortem `docs/dev/` altında yaşar. Yeni bir not açtıysan:

1. İlk başlığın hemen altına tek satır durum koy —
   `> **Durum:** AKTİF | REFERANS | ARŞİV | TASLAK | CANLI — kısa açıklama`
2. `docs/dev/README.md` indeksine bir satır ekle (doğru durum tablosuna).

Biten bir notu **silme**, `ARŞİV`e çevir; postmortem'ler bu deponun en pahalı
öğrenilmiş bilgisidir. Dil serbest — çoğu Türkçe, indeks bunu söylüyor.

## 4. Test scriptleri İKİ yere kopyalanır

`scripts/...` **ve** `x64/Release/scripts/...`. Uygulama ikincisinden okur;
yalnızca ilkini güncellemek eski scripti koşturur.

## 5. Geriye uyum yükü yok

Bir yol doğrulanıp ölü olduğu anlaşıldıysa **sök**. İki kod yolunu "her ihtimale
karşı" yaşatmak bu kod tabanında tekrar tekrar sessiz arızaya dönüştü. Ama
sessizce **anlam** değiştirme: alanın adı da değişmeli ki eski veri yanlış
okunmasın.

## 6. Vulkan birincil GPU yolu

CUDA/OptiX hızlı ama ikincil. Bir şey yalnızca birinde çalışıyorsa Vulkan'da
çalışmalı.

## 7. Tek branch: `main`

---

## Uygulamayı sürmek

```powershell
.\scripts\ipc\Start-RayTrophi.ps1        # başlatır, IPC hazır olunca "HAZIR" der
Import-Module .\scripts\ipc\RtIpc.psm1 -Force
Invoke-RtIpc physics.fracture_object @{ object = 'Beam'; site_count = 40; pattern = 2 }
```

Yerel pipe (`\\.\pipe\RayTrophiStudio`) **token istemez** — kimlik doğrulaması
yalnızca uzak (TLS) istekler için yapılır, yerelde güvenlik sınırı pipe ACL'i.
Uygulama açılışında pipe kendiliğinden kalkar.

`render.start` bir görüntü dosyası üretir, ve görüntüler okunabilir — yani
**görsel doğrulama da otomatikleştirilebilir.**

---

## Teşhiste tekrar eden dersler

Bu kod tabanında aynı hata sınıfları dönüp duruyor. Yeni bir arıza ararken
önce bunlara bak:

- **Üretici ≠ tüketici.** Bir olayı kuyruğa koyan yer ile tüketen yer ayrı
  döngülerde olabilir. "Uygulamada çalışıyor ama testte çalışmıyor" (veya
  tersi) neredeyse her zaman budur.
- **Tüketicinin BİRİMİNİ oku.** Bir sayıyı başka bir alana verirken. Impulse'ı
  hız alanına yazmak çökmez, "çok güçlü" gibi görünür ve kalibrasyon turuna
  gömülür.
- **★ Bir çözücüde "daha çok yakınsa, daha sert" bir belirti FİZİK DEĞİLDİR.**
  Gerçek bir kuvvet yakınsadıkça küçülür. İterasyon sayısıyla *büyüyen* şey
  neredeyse her zaman **tekil bir sistemin boş uzay bileşenidir** (referanssız
  bir bölge). Bu tek özellik, makul görünen bütün fiziksel adayları eler.
- **Varsayılan bir ölçüm değildir.** `false` dönen bir okuma çağıran tarafta
  yutulursa, bilgi eksikliği sessizce "sıfır ölçtüm"e dönüşür.
- **"Yok" ≠ "silinmiş".** Kaydediciler bulamadıklarını silinmiş sanabilir.
- **Tripwire'ın susması yokluğu kanıtlamaz.** Enstrümanın anahtarı, ölçtüğü
  şeyle çakışmamalı.
- **★ İNDEKS TAMPONU YOK, ve bu bir tercih.** `DNA::GeometryDetail` bir
  **attribute sistemi**: köşe başına vertex tutar, üçgenler ardışık üçlülerdir.
  `get_indices()` / `get_index_count()` diye bir şey yok — indeksli model
  geçmişte kaldı. Weld/unique eşlemesi gerekiyorsa ayrı bir önbellek olarak
  kurulur (bkz. erime yolundaki `flat_soa_to_unique`), geometriye gömülmez.
- **★ Geometri her zaman FLAT SoA'dır** (`TriangleMesh` + `DNA::GeometryDetail`).
  `Triangle` facade eski yol; hâlâ karşına çıkar ama **yeni kod flat SoA'yı esas
  almalı**. Sahnede nesne sayan/arayan kod yalnızca facade tararsa flat mesh'leri
  "yok" sayar ve bunun hiçbir belirtisi olmaz — proje sidecar'ından
  (`.rtp.bin`) gelen her şey flat'tir. `rtapi::listObjects()` ikisini de ele
  alıyor, örnek al. Yeni geometri ÜRETİRKEN de facade yığmak yerine flat SoA
  hedefle. **Temizlik fırsatçıdır:** ayrı bir göç projesi açma, dokunduğun
  dosyadaki facade referanslarını temizle ve öteye gitme.
- **Eski binary.** Beklenen bir UI öğesi ekranda yoksa, geometriyi ayıklamadan
  önce exe'nin zaman damgasına bak.
