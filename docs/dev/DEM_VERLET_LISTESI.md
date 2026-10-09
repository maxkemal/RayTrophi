# DEM komşu listesi (Verlet) — kuru tane çözücüsü

> **Durum:** AKTİF — 2026-10-08. Derlendi; 1.3M maliyet testi PASS (2026-10-09). Fizik alt kümesi (base/history/static/settle/dense/wet) 6/6 PASS (2026-10-09; ayrıntı NEXT_BUILD_CHECKS Parti 7).

## Neden

`grain_diagnostics.runtime.cost_last_substep` sayaçları (1.3M dökülen tane, r=8 mm,
2026-10-08): son substep'te 15.04M bucket slotu okundu, 15 064 tane-tane teması
bulundu → **temas başına ~1000 aday okuması**; tanelerin %98.9'u temassız. Her aday
okuması rastgele bir konum okuması (`position(j,bank)`). Bellekte de üç dönen
bucket tablosu tane başına 816–1632 B tutuyordu (temas geçmişinden sonra en büyük kalem).

## Tasarım

- **Tek bucket tablosu**, yalnız liste yeniden kurulurken temizlenir ve doldurulur.
  Eski düzen her substep'te bir tabloyu temizleyip bir sonrakine ekliyordu.
- **Tane başına komşu listesi**: `kMatterGrainListCapacity` (32) slot + sayaç.
  Kesme = hücre boyu = `2r + köprü payı + skin`, `skin = 0.5 r`.
- **Yeniden kurulum tetikleyicisi GPU'da**: adım kernel'i, tane son kurulumdan beri
  `skin/2`'den fazla yer değiştirdiyse bir sonraki substep'in bayrağını kaldırır.
  Üç bayrak kelimesi dönüşümlü (`diagnostics[12 + k%3]`); substep k'nin liste
  kernel'leri k'nin bayrağını okur, adım kernel'i k+1'inkini yazar, k'nin
  `list_clear`'ı k+2'ninkini sıfırlar (en son k-1'de okundu). Host geri okuması yok.
- **Her karenin başında zorunlu kurulum** (host `diagnostics[12] = 1` yükler):
  taneler her kare hücre sırasına göre yeniden dizilir, doğum/emme olur.
- **Doğruluk**: iki tane kurulumdan beri en fazla `skin/2` hareket ettiyse
  aralarındaki mesafe en fazla `skin` değişti; adımda `2r + köprü payı` içindeki her
  çift kurulumda kesme içindeydi. Bucket taşması ve liste taşması yayımlamayı reddeder.
- Aşamalar: substep başına `list_clear → hash → list_build → step` (kare başı
  `clear` revizyon 20'de söküldü: geçmiş artık yerinde, sıfırlama FRESH bayrağıyla). Bayrak kapalıyken ilk üçü erken döner.
- **Temas geçmişi bu partide değişmedi** (anahtar araması temas başına ~1 okuma
  ölçüldü); sonraki partide 1536 → 672 B/tane (bkz. aşağı).

## Beklenen (ölçülecek)

| | Önce | Hedef |
|---|---|---|
| Aday okuması / substep (dökme) | 15.04M | **1.90M** (ölçüldü, 15 050 temas) |
| Bucket belleği / tane | 816–1632 B | 272–544 B (+132 B liste, +12 B kurulum konumu) |
| Kare (1.3M, 384 substep) | ~9.4 s | **~2.5–3.3 s** (ölçüldü) |
| Liste kurulumu / kare | — | **8** (384 substep'te) |

Sayaçlar: `cost_last_substep.neighbour_candidates` (adımda okunan liste girdisi),
`list_rebuilds` (karedeki kurulum sayısı).

## Sonraki

1. ~~Temas geçmişini listedeki konuma bağlamak ve çift başına tek kayıt (1536 → ~384 B).~~
   Yerine (2026-10-09, revizyon 20, derleme bekliyor): tek bank, yerinde güncelleme, host'un
   tuttuğu tane→blok haritası → **1536 → 672 B** (+12 B harita/sahip/maske), permute
   kernel'leri söküldü. Çift başına tek kayıt tek bankta yarış (j, i yazarken okur);
   listeye bağlamak 32 > 24 slot. Ayrıntı: NEXT_BUILD_CHECKS Parti 8.
2. Uyuyan kümeler.
3. Host: GPU sıralaması, kare başına tam indirmeyi kaldırmak (B4).
   Ölçüldü (2026-10-09, 1.3M): kare ~3.3 s'nin ~1.7 s'i tane host aşamaları, GPU ~0.95 s →
   **uyuyan kümelerden önce gelir.** İlk adım (gereksiz kopya/sıralama/hash) yazıldı,
   NEXT_BUILD_CHECKS Parti 9. Kalan: kare başı konum/hız/affine yükleme + indirme (~88 MB×2).
