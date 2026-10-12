# 2026-10-12 parti 12: otomatik fırtına hücrelerini düzenlenebilir yapma (dondurma)

Parti 7–11 derlendi ve `ce1c7a6` olarak commitlendi. Parti 11'in ölçülmemiş
maddeleri (basınç, kuleler, yukarı yönlü yıldırım, kare süresi) aşağıda,
bu partinin maddelerinden sonra.

## Neden
Kullanıcı "sahnedeki 2 hücreye UI'dan ulaşılamıyor" dedi. IPC ile bakıldı:
`storms.cells` BOŞ. Görünen hücreler otomatikti (`max_cells 15`, alan
1000 m, türetme kapalı). Otomatik hücreler şeritlerin saf fonksiyonu
olduğu için saklanan bir halleri yok, tek tek düzenlenemezler.

## Özet
- `atmosphere::freezeStormCells(state, t, wind, first)`: o an canlı
  otomatik hücreleri elle konmuş hücreye çevirir. Doğum, doğum konumu,
  yarıçap, tepe değeri ve ömür aynen korunur, yani gök değişmez. En fazla
  8 hücre çevrilir. Ardından `max_cells 0` ve `derive_from_climate false`
  yapılır. `StormCellState.peak` alanı eklendi.
- **IPC:** `world.set_clouds {storms:{freeze_automatic:true}}`. Bu bir eylem;
  yamanın geri kalanı uygulandıktan sonra çalışır.
- **Panel:**
  - "Convert live automatic cells to authored" düğmesi (aynı IPC yolunu
    kullanır).
  - "Authored cells (N)" artık varsayılan olarak açık.
- **Gizmo:** Otomatik hücrelerin de tutamağı var. Tıklayınca hepsi
  dondurulur ve tıklanan seçili kalır, hemen sürüklenebilir.

## Kontrol listesi (sırayla)

1. **IPC ile dondurma (görüntüden önce).**
   - Yap: `world.cloud_stats`, sonra
     `world.set_clouds {storms:{freeze_automatic:true}}`, sonra yine
     `world.cloud_stats` ve `world.get_clouds`.
   - Görmen gereken:
     - Hücre konumları, yarıçapları ve `life` değerleri AYNI kalır, ama
       `manual: true` olur ve `manual_index` 0..n−1 sıralanır.
     - `get_clouds.storms.cells` dolar; `max_cells` 0 olur.
   - Bozuksa: konumlar kaydıysa doğum konumundan rüzgâr kayması yanlış
     çıkarılıyor (`c.age_s` ile `wind × age`).
   - ★ Sinsi olan: 8'den fazla canlı otomatik hücre varsa fazlası kaybolur.
     Bu beklenen davranış (elle konan hücre sınırı), ama gökte görünür bir
     değişiklik yaratır.

2. **Panel.** Storms bölümü.
   - Görmen gereken: düğme dondurur, ardından "Authored cells (N)" listesi
     dolar. "Remove" bir hücreyi siler; konum ve yarıçap değiştirilebilir.

3. **Gizmo.** Önce dondurma yapmamış bir sahne aç.
   - Görmen gereken: "auto cell (…)" tutamağına tıklayınca liste dondurulur,
     tıklanan hücre seçili kalır ve XZ okları çıkar. Sürükleyince hücre
     kayar.
   - Bozuksa: tıklayınca seçim hemen kalkıyorsa, aynı tıklamada boş alan
     seçimi kaldırıyor demektir.

4. **Parti 11 kalanları:**
   - basınç (`synoptic_lift`)
   - hücre kuleleri ve girdap (`turrets 0` / `rotation 0` karşılaştırması)
   - yukarı yönlü yıldırım (`kind: "upward"`, çatallar yukarı açılıyor mu)
   - kare süresi

## Bilinen
- `scripts/audit_ipc_capabilities.py`, `audio.*` metotlarında FAIL veriyor.
  Bunlar paralel ajanın ses işi. Tarif tablosu yeniden üretildi.
