# Stylize Layer — Çalışma Mantığı (Notlar)

Stylize, path-traced render'ı **değiştirmeyen**, onun çıktısının üzerine uygulanan bir
**ekran-uzayı post-process** katmanıdır. Sonucun kalitesi AOV (geometri/malzeme) verisinden
gelir; renk filtresi değildir.

## 1. Veri kaynağı: AOV'ler
Her piksel için gereken girdiler:
- `albedo` — yüzey rengi (path-noise değil, malzeme albedo'su)
- `normal` — dünya-uzayı normal
- `world_position` — birincil isabetin dünya konumu (yüzeye-kilitli fırça için)
- `depth` — kameradan uzaklık (`|world_position − cameraOrigin|`)
- `material_id` — gerçek malzeme indeksi (outline malzeme sınırları için)
- `edge` — CPU'da komşu piksellerden türetilir (depth + normal + material_id süreksizliği)

## 2. AOV'ler nereden gelir
- **CPU render:** doğrudan `cpu_*_accumulation_buffer`'lardan (render sırasında dolar).
- **GPU render (Vulkan/OptiX):** `Renderer::fillStylizeAOVFromBackend` backend'in primary-hit
  AOV'lerini `getDenoiserFrame(useAuxiliary=true, includeColor=false)` ile CPU buffer'lara çeker.
  - Vulkan: raygen `denoiserPositionImage` (set 0, binding 17, rgba32f) yazar.
  - OptiX: kernel `stylize_position` buffer'ına yazar (`ray.origin + dir*t` ile world pos).
  - Ortak encoding (`.w` kanalı): `0` = miss, `1` = isabet/bilinmeyen malzeme (saç/hacim),
    `≥2` → `material_id = w − 2`. `depth` CPU'da world pos + kamera origin'den hesaplanır.
  - Yön: GPU buffer'ları bottom-up (CPU buffer düzeniyle aynı), flip gerekmez.

## 3. Piksel işleme — `StylizePostProcess::applyPostProcess`
Sırayla: palette ramp → malzeme tonu koruma → renk sadeleştirme (edge/material guard'lı) →
stroke field (fbm + yüzeye-kilitli koordinat) → opsiyonel wet-oil modeli → pigment kalınlığı →
outline (edge tabanlı, çizgi tipi/renk modu). Sonuç `global_strength` ile orijinalle harmanlanır.

## 4. Paralellik
- Post pass (`applyStylizeToSurface` / tonemap yolu) satır bazında `std::execution::par_unseq`.
- AOV decode döngüsü (`fillStylizeAOVFromBackend`) de `par_unseq` (piksel bağımsız).
- Geriye kalan seri/senkron maliyet sadece GPU→CPU `getDenoiserFrame` readback'i.

## 5. AOV cache (GPU yolu maliyet optimizasyonu)
- Cache, **kamera hash**'i (lookfrom+lookat+vup+vfov) + sample-reset ile anahtarlanır.
- Kamera hareket ederken her kare yeniden çekilir (gecikme/hayalet yok); sabitken cache'ten
  okunur (readback atlanır). Çözünürlük değişimi de geçersiz kılar.
- Sekans/animasyon render'ında `forceRefresh=true` (kamera sabit + obje hareketi durumu).

## 6. Parametre değişimi davranışı (önemli)
- Stylize parametreleri **accumulation reset ETMEZ** — post-process'tir, render'ı/AOV'leri
  bozmamalı. `markStylizeChanged` yalnızca `stylize_redisplay = true` set eder.
- `stylize_redisplay` post'u mevcut yakınsamış görüntü üzerinde yeniden çalıştırır ve
  **kullanıcının tonemap aç/kapa ayarına saygı duyar** (`apply_tonemap` gibi tonemap'i zorlamaz).
- Display, surface her yeniden kurulduğunda (`surface_rebuilt`: tonemap-apply VEYA render VEYA
  redisplay) stylize'ı tekrar uygular.

## 7. Backend pariteleri / derleme
- Vulkan ve OptiX aynı CPU stylize kodunu besler (backend-agnostik).
- RT payload location-0 yapısı 6 shader'da paylaşılır; alan eklenince hepsi yeniden derlenir
  (raygen/closesthit/hair_closesthit/volume_closesthit/miss/shadow_miss + .spv) + CUDA/OptiX.
