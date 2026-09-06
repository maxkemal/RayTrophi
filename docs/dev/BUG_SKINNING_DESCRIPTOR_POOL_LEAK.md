# GPU skinning, sahneye obje ekleyince CPU'ya düşüyordu

> **Durum:** REFERANS — kök neden bulundu ve düzeltildi (2026-09-05), RUNTIME
> DOĞRULANMADI. Doğrulama adımları `NEXT_BUILD_CHECKS.md` §1'de.

## Belirti (kullanıcı raporu, 2026-09-05)

> "raster mod skin animde gpu compute ile vertexleri doğru hareket ettirirken
> sahneye sonradan bir obje eklenince cpu fallback'e düşüyor, bu skin anim
> hesaplamaları vulkan rt'de olmuyor"

İki ayrı şikâyet gibi görünüyor. **Tek kök neden.**

---

## ★★★★ Kök neden: descriptor set'i SERBEST BIRAKMADAN düşürmek

`m_skinningDescPool` **sabit sayıda** descriptor set tutan tek bir havuz, ve bu
havuzu **iki tüketici paylaşıyor**:

| Tüketici | Nerede | Ne için |
|---|---|---|
| Raster viewport | `RasterMeshBuffer::skinningDescSet` | Solid/Matcap/Realtime |
| Vulkan RT | `BLAS::skinningDescSet` | Rendered |

`destroyRasterMesh()` her tamponu yok ediyordu ama `skinningDescSet`'i
**yok saymıyordu — sadece yapıyı sıfırlıyordu** (`mesh = RasterMeshBuffer{}`).
Havuz `FREE_DESCRIPTOR_SET_BIT` olmadan kurulmuştu ve hiç reset edilmiyordu, yani
o slot **kalıcı olarak** tükenmiş oluyordu.

Sahneye herhangi bir obje eklemek raster mesh'lerini yeniden kurar
(`destroyAllRasterMeshes` → yeniden inşa). Yani:

```
yeniden kurulum 1 : N skinned mesh  →  N set ayrıldı
yeniden kurulum 2 : N set daha ayrıldı (önceki N sızdı)
...
toplam ayrılan > havuz kapasitesi  →  vkAllocateDescriptorSets BAŞARISIZ
                                   →  dispatchSkinningToBuffers false döner
                                   →  CPU skinning'e sessiz düşüş, KALICI
```

Kapasite 64'tü. Bir karakter tipik olarak bir düzine skinned alt-mesh olarak
gelir (gövde, saç, kıyafet), yani **iki-üç yeniden kurulum havuzu bitiriyordu.**

### İkinci belirti neden aynı kök

BLAS yolu **aynı havuzdan** ayırıyor. Raster tarafı havuzu boşalttığı anda
`dispatchSkinning()` da ayıramıyor ve `return` ediyor — Vulkan RT'de deformasyon
hiç uygulanmıyor. "RT'de olmuyor" ile "raster CPU'ya düştü" **aynı olayın iki
yüzü.**

### Neden kimse bunu bug diye raporlamadı

Başarısızlık `return false`. Çağıran taraf sessizce CPU skinning'e geçiyor,
görüntü **doğru** kalıyor — sadece yavaş. Tek iz, süreç başına bir kez basılan
bir `VK_INFO` satırıydı. Bu deponun en pahalı hata sınıfı: *makul görünen sonuç.*

★ Yorumun kendisi hatayı savunuyordu:

```
// Pool is never reset; sets are allocated once per BLAS and reused every
// frame (FREE_DESCRIPTOR_SET_BIT not needed).
```

"Bir kez ayrılır" **yanlıştı**: her sahne yeniden kurulumu yeni bir set ayırıyor.
Yorum ölçüm değil varsayımdı, ve varsayım ölçüm yerine geçtiği için 64'lük
kapasite yeterli sanıldı.

---

## Düzeltme

| Dosya | Değişiklik |
|---|---|
| `VulkanDevicePipelines.cpp` | Havuza `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT`; kapasite 64 → 512 |
| `VulkanBackend.h` | `freeSkinningDescriptorSet()` + `m_skinningDescPoolCapacity` / `m_skinningDescSetsLive` sayaçları |
| `VulkanBackend.cpp` | Serbest bırakma uygulaması; iki BLAS yıkım yolunda çağrı; tükenme artık `SCENE_LOG_WARN` ile SAYIYLA raporlanıyor |
| `VulkanBackend_Raster.cpp` | `destroyRasterMesh()` artık set'i iade ediyor |

Ayrıca yol boyunca kapatılan yan sızıntılar:

- `blas.persistentBoneMatsBuffer` **hiçbir** BLAS yıkım yolunda yok edilmiyordu
  (düz bellek sızıntısı).
- Bone matris tamponu büyüdüğünde `skinningDescSet = VK_NULL_HANDLE` atanıyordu
  — yine iade etmeden düşürme.

★ Serbest bırakmanın güvenliği: her iki çağıran da işi bitmiş GPU işinden sonra
çalışıyor. `dispatchSkinning*` `endSingleTimeCommands` ile submit ediyor ve o
fence bekliyor; BLAS yıkımı `vkDeviceWaitIdle` sonrasında.

---

## ★★★ RT neden yine de daha yavaş — ve bu ayrı bir konu

Kullanıcının üçüncü sorusu: *"rt ile skin anim gpu compute aynı mı, sanki rt'de
daha yavaş hesaplanıyor gibi"*.

**Compute aynı.** Aynı `skinning.spv`, aynı pipeline, aynı havuz. Fark
skinning matematiğinde değil, **sonrasında**:

| | Raster (`dispatchSkinningToBuffers`) | RT (`dispatchSkinning`) |
|---|---|---|
| Skin | ✔ | ✔ |
| BLAS refit | — | ✔ (aynı komut tamponunda) |
| Submit + fence | mesh başına | BLAS başına |
| Kare başına ek | — | `drainInFlightTraces()` + `updateTLAS` |

Yani RT **tasarım gereği** daha pahalı ve maliyet **hızlandırma yapısı işi**,
skinning değil.

★ Başlıkta bir yalan vardı: header'da `dispatchSkinningAll(...)` bildirilmişti,
yorumu `batch: 1 submit for all BLASes`. **Hiç tanımlanmamış ve hiç
çağrılmamıştı.** Var olmayan bir optimizasyonu ilan eden bir bildirim, hiç
bildirim olmamasından kötüdür: "RT neden yavaş?" sorusuna yanlış cevap verir.
Söküldü (kural 5). Per-BLAS submit'leri tek komut tamponunda toplamak hâlâ
buradaki gerçek kazanç — ayrı bir iş.

---

## Ölçü aleti

Tükenme artık sayı olarak raporlanıyor:

```
[Vulkan] GPU skinning disabled: descriptor pool exhausted (512/512 sets live).
Skinning falls back to the CPU from here.
```

`live == capacity` görüyorsan bir yerde hâlâ iade edilmeyen bir set var. Sayı
kapasitenin altındayken başarısızlık geliyorsa sebep havuz değildir.

Bağlantılı: [[feedback_vulkan_compute_descriptor_pool_batching]] — aynı sınıf,
farklı havuz.
