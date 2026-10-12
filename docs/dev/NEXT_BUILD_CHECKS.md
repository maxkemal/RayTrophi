# 2026-10-12 parti 16b: fırtına kabuğu parti 14'e geri alındı

Parti 16 canlıda kötü çıktı: kule dağ gibiydi, detay kaybolmuştu ve RT
maliyeti arttı. Parti 14 kabuğu geri geldi. Gerekçe:
`docs/dev/ATMOSPHERE_CELL_CLOUDS.md` §8.1. Parti 13–16b henüz commit
edilmedi. Bu derleme parti 14 görünümünü verirse hepsi birlikte
commitlenecek.

## Özet
- `cloud_cells.comp`: parti 14 hali (bükme yok, uydu halkası yok).
- `CloudCellsGPU`: 32 B başlık (`frame` yok).
- Kabuk: 13c yüzey modeli (kubbe, billow ve duvar itmesi, kabarcık
  oyması), tepe bükme ile birlikte.
- Katman üreticisi: sütun modu ve nem halkası yok.
  - `cloud_rt.glsl` katman majorantı yine `column.a` okumuyor.
- `storms2.z`: yeniden duvar bulutu payı (300 m × wall_cloud).
- Korunan: 15b döşeme bükmesi (yalnız şekil okumaları, ±900 m) ve
  döndürülmüş kabarcık oktavı (320 m); yağış tipleri; graupel payı.

---

## Kontrol listesi (sırayla)

1. **Derleme ve shader'lar.**
   - Görmen gereken: hata yok.
   - Bozuksa:
     - `CloudCellsGPU std430 layout` static_assert patlıyorsa bir yerde
       hâlâ `frame` yazılıyor.
     - Shader'da `cloudShellShear` tanımsız diyorsa sıra bozulmuş.
       Tanımı `cloudShellDensity`'den önce olmalı.

2. **Kare süresi (önce bu, bağımsız).** RT modda, yol izleyici kapalı,
   aynı sahne ve açı.
   - Görmen gereken: parti 14 derlemesindeki süre, yani parti 16'dan
     belirgin hızlı.
   - Bozuksa: hâlâ yavaşsa sebep kabuk değil. O zaman
     `rt.perf` / `gpu_kernel_timings` ile ölçülür.

3. **Ana kule (asıl istek).** Kullanıcının "baya güzel oldu" dediği
   parti 14 görünümü:
   - karnabahar yüzeyli kule ve dik duvarlar
   - basamaklı tepeler
   - örs kama şeklinde
   - Bozuksa: kule düz bir koni ya da dağ gibiyse yüzey modeli
     çalışmıyor demektir. Bak: `cloudShellDensity` → `inside` /
     `kTowerBillowScaleM`.
   - ★ Sinsi olan: parti 14'e çok benzer ama kabarcıklar biraz kaymış ve
     dönmüş görünür. Bu beklenen; döşeme düzeltmesi korundu. Sıra halinde
     tekrar eden desen OLMAMALI.

4. **Hücre dışındaki katman.**
   - Görmen gereken: Fair weather cumulus parti 14'teki gibi. Hücrenin
     içinde katman kenara çekilmiş, çevrede kendi örtüsünde.
   - Bozuksa: hücre çevresinde kare kare kesik varsa majorant
     tutarsızdır. 15b'nin `column.a` satırı tamamen silinmeli.

5. **Graupel payı (IPC, parti 15'ten bekliyor).**
   - Yap: `world.set_climate {surface_temperature_k: 268.15}` ve
     `world.sample_clouds {mode:"precipitation_form", backend:"viewport"}`.
   - Görmen gereken: olgun çekirdek altında `snow`.
   - Yap: inversiyon (270.15 K, lapse −0.004).
   - Görmen gereken: yere yakın `freezing_rain`.

## Bilinen
- Devin altındaki küçük bulutlar, üst katmana bağlantı ve ince stratus
  hâlâ yok. Kabuğun yüzeyine dokunmadan, açılıp kapatılabilir ayrı
  adımlarla yapılacak.
- `scripts/audit_ipc_capabilities.py`, `audio.*` metotlarında FAIL veriyor
  (paralel ajan).
