# ⚠ AÇIK: sky değişimi bazen ekrana ulaşmıyor (yayınlama askıda)

> **Durum:** AÇIK — kullanıcı bildirdi, ölçüm KISMEN yapıldı, **park edildi**
> (2026-09-08). Aralıklı (intermittent). Bir somut bulgu var (§2) ve o
> bulgu bağımsız olarak doğrulanabilir.

## 1. Belirti

Projelerde sky değişimi bazen tepkisiz kalıyor; sahne proje kaydedilmiş hâliyle
duruyor. RT yolu **doğru tepki veriyor**. Kullanıcının son gözlemi: panelden de
değişiyor, ama **bazen yayınlama askıda kalıyor ya da render güncellenmiyor**.
Kamera hareketi de render'ı tetiklediği için maskelenebiliyor.

★ Bu şekliyle bir **sunum/geçersizleştirme** arızası, değer arızası değil.

## 2. ★★★ ÖLÇÜLEN: `solid` ile `hdri` AYNI üretici imzasına hash'leniyor

Canlı IPC, `world.set_mode` döngüsü, her adımdan sonra `rayfusion.probe_field`:

| set_mode | mode | producer_signature |
|---|---|---|
| nishita | nishita | 17653716898287885454 |
| **solid** | solid | **10267504932950959680** |
| nishita | nishita | 12230242487235155960 |
| **hdri** | hdri | **10267504932950959680** |
| nishita | nishita | 3466029052444085419 |

`solid` ve `hdri` **birebir aynı** — üstelik iki farklı anda, farklı
`hit_fraction` (0,1531 ve 0,2937) ile. Yani imza bu iki modu **ayırt etmiyor**.

★ Bu, "solid moda / HDRI'ya geçmek değiştirmiyor" belirtisinin makul mekanizması.
İmza değişmezse `state->appliedRevision` eşit kalır ve probe alanı yeniden
uygulanmaz (`MaterialPreviewProbeField.cpp:307` civarı).

⚠ Ama **tek başına açıklamıyor**: `traced_publishes` yine de artıyor (69 → 80),
yani yayın oluyor. Ve `world.set_sun_elevation` DOĞRU yayılıyor
(imza 17632… → 768511… → 17653…). Yani kırık olan bütün sky yolu değil,
**mod kimliğinin imzaya girmemesi**.

## 3. Elenenler

- **IPC yolu sağlam:** `world.set_mode` ve `world.set_sun_elevation` çekirdeğe
  ulaşıyor, `world.get` doğru raporluyor, kullanıcı ekranda da doğruladı.
- **Panel yolu da çekirdeğe ulaşıyor** (kullanıcı doğruladı) — ilk hipotez olan
  "panel `worldChanged()` çağırmıyor" **yanlış çıktı**: panel `g_world_dirty`,
  `resetCPUAccumulation()` ve `backend->resetAccumulation()` yapıyor
  (`scene_ui_world.cpp:1678`), yalnızca `setWorld()` çağrısını bilerek
  Main loop'a erteliyor (LUT flush'tan sonra tek transfer için).

## 4. Sıradaki adım (park; buradan devam edilecek)

1. **Ucuz ve bağımsız:** mod kimliğini üretici imzasına kat. `solid`/`hdri`
   ayrımı olmadan probe alanı bu iki mod arasında hiç yenilenmez.
2. Ertelenen `setWorld()`'ün Main loop'ta **her zaman** servis edilip
   edilmediğine bak. Bu deponun tekrar eden sınıfı:
   [rebuild bayrağı işten SONRA temizlendi] ve
   [mod geçişi servis etmediği rebuild İSTEĞİNİ düşürdü].
   Aralıklı olması tam bu şekle uyuyor.
3. ★ Kamera hareketi render'ı tetiklediği için arıza **kendini gizliyor**.
   Tekrar üretirken kamerayı kesinlikle sabit tut.

## İlgili

- `MaterialPreviewProbeField.cpp` — `producerSignature` / `appliedRevision`
- `RtApi.cpp:2007` `worldChanged()`, `RtApi.cpp:2083` `setWorldMode`
- `scene_ui_world.cpp:573` mod combo'su, `:1678` uygulama bloğu
