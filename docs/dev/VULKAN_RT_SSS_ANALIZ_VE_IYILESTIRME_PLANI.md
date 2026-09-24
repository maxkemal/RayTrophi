# Vulkan RT (RayFusion) SSS (Subsurface Scattering) Analiz & İyileştirme Planı

> **Durum:** REFERANS — 2026-09-24. SDF/sıvı SSS eki RUNTIME DOĞRULANMADI. Vulkan RT random walk + RayFusion paritesi + yöntem/adım kontrolleri yapıldı; kullanıcı Vulkan RT'de görsel olarak doğruladı. RayFusion Aşama 1-3 (§4) hâlâ açık.

## 0. 2026-09-24 denetimi: vaat ≠ kod

Bu notun §3'ü yazıldığında kod şunu yapıyordu — ve bunu yakalayan hiçbir test
yoktu, çünkü SSS parametreleri **yalnızca panelden** yazılabiliyordu (kural 1):

| Vaat | Koddaki gerçek | Düzeltme |
|---|---|---|
| Çift Beer-Lambert kaldırıldı | Çıkışta `exp(-σ_t d)·A`, sonra bir kez daha `·sssColor` | Ağırlıklar tek-örnek MIS; çıkışta ek çarpan yok |
| RGB yarıçap farkı | Her kanal aynı `A` ağırlığını alıyordu → radius TON ÜRETEMEZDİ | `w_c = σ_s,c T_c / Σ p_k σ_t,k T_k` |
| Christensen-Burley | `sssColor` doğrudan tek-saçılma albedosu (her olayda bileşik → koyu/doygun) | Cycles remap: çoklu→tek saçılma albedosu + yarıçap fit'i |
| IOR ile kırılma | `-N`, `N` boyunca kırılıyordu = hep `-N`; slider ölü | Gelen ışın (`rayDir`) kırılıyor |
| Exact çıkış | `TerminateOnFirstHit` any-hit'te EN YAKINI değil rastgele bir yüzeyi döndürür | Any-hit tüm adayları yok sayıp en yakını tutuyor |
| — | Çıkış normali İÇERİ çevriliyordu | `dot(n, dir) > 0` = dışarı |
| — | Rulet öldürünce iç noktadan giriş normaliyle yeniden yayıyordu (sızıntı) | Yol sonlanıyor |
| — | `freePath` 4×yarıçapa kırpılıyordu (bias) | Kaldırıldı |
| API/IPC `subsurface_ior` | `setMaterialParam` hiçbir SSS anahtarı tanımıyordu | `subsurface`, `_color`, `_radius`, `_scale`, `_anisotropy`, `_ior` |

Ölçüm: `scripts/ipc/Probe-SssResponse.ps1 -Object <mesh>`.

**2026-09-24 ek: yöntem + adım tavanı Vulkan'a açıldı.** `useRandomWalkSSS`
(bool) → `sssMethod` (0=Random Walk, 1=Fast); `sssMaxSteps` (vars. 6, 1–32) →
`sssWalkMaxSteps` (vars. 64, 8–256). Ad değişti çünkü anlam değişti: remap
sonrası tek-saçılma albedosu ~0,99, 6 adımlık tavan enerjinin çoğunu öldürür.
Vulkan ABI boyutu değişmedi (`_ext_pad0/1` ve `_volume_pad0/1` kullanıldı).
Fast = SSS payı `subsurface_color` ile Lambert, probe yok — RayFusion ile aynı
renk. IPC: `subsurface_method`, `subsurface_max_steps`. Alanlar artık proje
JSON'una da yazılıyor (önceden hiç kaydedilmiyordu).
Anlam değişikliği: **SSS Color artık hedef (çoklu saçılma) rengidir**; eski
kodla aynı değer daha açık ve daha az doygun görünür. Preset'ler eski koyu
modele göre ayarlanmıştı, yeniden kalibre edilmeli.

## 1. Giriş ve Amaç

RayTrophi Studio bünyesindeki Vulkan RT (RayTracing / Path Tracing) ve RayFusion motorunda kullanılan Subsurface Scattering (SSS - Yüzey Altı Saçılım) altyapısının endüstri standardına (Blender Cycles / Disney Principled BSDF) ulaştırılması, performans maliyetinin düşürülmesi ve RayFusion (Raster) için gelecek SSS entegrasyon adımlarının planlanması bu dokümanın temel amacını oluşturur.

---

## 2. Mevcut Durum & Problem Analizi (Legacy SSS vs Disney Random Walk)

### 2.1 Eski Sabit Adım (Fixed 6-Step Blind March) Yaklaşımının Sorunları
* **Kör Raymarch (Blind March)**: Eski `scatterSSS` fonksiyonu, yüzey normallerini ve gerçek geometri kalınlığını gözetmeksizin sabit 6 adımlık bir uzay yürüyüşü yapmaktaydı.
* **Katlama & Kararma (Double Beer-Lambert Darkening)**: Üstel sampling yapıldığı halde ekstra Beer-Lambert sönümleme çarpanı uygulandığı için materyal iç kısımları aşırı kararıyor, canlı insan cildi, balmumu veya yeşim taşı görüntüsü elde edilemiyordu.
* **Yönelim ve Kırılma Eksikliği**: Işık yüzeyden içeri girerken nesnenin kırılma indisini (IOR) dikkate almıyor, yüzey normali boyunca düz yönleniyordu.
* **Anisotropy (Yönlü Saçılım) Yokluğu**: İleri/geri saçılım (Henyey-Greenstein faz fonksiyonu) desteklenmediği için ışık ortam içinde izotropik dağılıyordu.
* **Maliyet/Performans Dengesizliği**: Her adımda kör turlamalar GPU üzerinde yüksek TDR riskine ve firefly (aşırı parlak gürültü piksellerine) sebep oluyordu.

---

## 3. Gerçekleştirilen İyileştirmeler (Disney / Cycles Random Walk Entegrasyonu)

Vulkan RT shader katmanında (`bsdf_scatter.glsl`, `closesthit.rchit`, `volume_closesthit.rchit`, `shadow_anyhit.rahit`) yapılan teknik güncellemeler:

### 3.1 Christensen-Burley & Hero-Wavelength Sampling
- **Sönümme Katsayıları**: SSS yarıçapına göre katsayılar $\sigma_t = 1 / (\text{sssRadius} \times \text{sssScale})$ olarak tanımlandı.
- **Random Walk Adımları**: Maksimum adım sayısı 32'ye çıkarıldı ve 4. adımdan itibaren Roulette (Russian Roulette) sönümleme mekanizması eklendi. Böylece ince yüzeylerde ışınlar hızla sonlanarak performans korundu.

### 3.2 Kırılmalı Giriş Yönü (Snell's Law / `subsurfaceIor`)
- Işının nesneye girdiği noktadaki kırılma açısı Snell Kanunu (`refractLikeOptix`) ile hesaplandı. `subsurfaceIor` parametresi ile ışının ortam içine doğru bükülmesi sağlandı.

### 3.3 Henyey-Greenstein Anisotropy Faz Fonksiyonu
- `sampleHG` faz fonksiyonu eklenerek `subsurfaceAnisotropy` ($-0.99$ ile $+0.99$ arası) parametresiyle ışığın nesne içinde ileriye veya geriye doğru saçılması sağlandı.

### 3.4 AnyHit Sentinel ve Yüzey Çıkış Tespiti
- `shadow_anyhit.rahit` ve `hair_shadow_anyhit.rahit` shader'larında `SSS_SUBSURFACE_PROBE` (sentinel mask `0x555B0DEDu`) mantığı iyileştirildi. Işın nesneden çıkarken exact mesafe ($T$) ve çıkış normali octahedral olarak paketlenip geri döndürüldü.

### 3.5 Mimari ve API/IPC Katmanı Entegrasyonu
- `SurfaceSample` yapısına `subsurfaceIor` eklendi.
- `closesthit.rchit` (poligon yüzeyler) ve `volume_closesthit.rchit` (akışkan isosurface yüzeyleri) `matx.subsurface_ior` ve `imx.subsurface_ior` değerlerini `SurfaceSample`'a aktaracak şekilde güncellendi.
- C++ Core (`scene_data.h`), Scripting API (`RtApiFluid.cpp`, `RtPython.cpp`) ve IPC (`RtIpc.cpp`, `RtIpcMethodDescriptors.cpp`) katmanlarında `subsurface_ior` tamamen temsil edildi.

---

**2026-09-24 ek 2: SDF/sıvı izoyüzeyinde SSS.** Yürüyüşün çıkış probe'u
yalnızca üçgen görüyordu (mask 0x01); izoyüzey üçgen değil, yani içeriden hiç
bulunamıyor, her yürüyüş rulete kadar gidip ölüyordu → süt/bal kararıyordu, hata
yok. Ayrıca volume_closesthit saçılmadan sonra origin'i `hitPos`'a (GİRİŞE)
yeniden oturtuyordu. Çözüm: `bsdf_scatter.glsl`'de `SSS_CUSTOM_EXIT` kancası
(`sssCustomExit`), volume_closesthit izo alanda içeri→dışarı geçişi arıyor
(yalnızca önce içeride olunduysa; ilk nokta bandın dış yarısında olabilir),
yakın olan sınır (üçgen ya da alan) kazanır; `g_sssExited/Pos/N` ile çıkıştan
oturtuluyor. RayFusion SDF raster: yalnızca renk paritesi, sızma yok.

## 4. RayFusion (Rasterizer) SSS İyileştirme Planı

> **2026-09-24 — Aşama 0 yapıldı (RT paritesi):** `material_preview_frag.frag`
> SSS lobunu RT ile aynı enerji paylaşımıyla kuruyor: diffuse'un `amount` payı
> `subsurface_color`'a gider (base color'a değil). Işık sızması kanal başına,
> **dünya birimli** mfp (`radius·scale`) × ekran türevlerinden eğrilik;
> ince bölgelerde `exp(-2/κ / mfp)` arka ışık. Yeteneğin adı
> `radius_profile_approx` → `curvature_wrap_rt_albedo`. Aşama 1-3 hâlâ açık.

Path Tracing tarafında Random Walk SSS standarda çekildikten sonra, hibrit/rasterizer motoru olan **RayFusion** için aşağıdaki adımlar planlanmıştır:

```
[RayFusion Raster SSS Planı]
   ├── Aşama 1: Depth & Normal Buffer tabanlı Screen-Space SSS (SSSS / Separable Bilateral Blur)
   ├── Aşama 2: Light Translucent Pass (Thin-Surface Backlight SSS)
   └── Aşama 3: Irradiance Cache / Texture-Space BSSRDF (Yüksek Kaliteli Nesne Bazlı Saçılım)
```

### 4.1 Aşama 1: Screen-Space Subsurface Scattering (SSSS)
- **Yöntem**: G-Buffer'dan alınan `Subsurface Amount` ve `Subsurface Profile` maskeleriyle post-process aşamasında 2-pass Separable Gaussian/Bilateral Blur uygulaması.
- **Hedef**: Realtime 60+ FPS modunda insan cildi ve balmumu objeler için çok düşük GPU maliyetli SSS çıktısı üretmek.

### 4.2 Aşama 2: Thin-Surface Backlight Transmission
- **Yöntem**: Kulak, yaprak veya ince plastik gibi nesnelerde arkadan gelen ışığın nesne kalınlığına (Shadow Depth Difference) göre geçiş sağlaması.

---

## 5. Doğrulama ve Test Protokolü

Yeni SSS mimarisinin doğrulanması için aşağıdaki adımlar takip edilmelidir:

1. **Shader Compilation**:
   - `bsdf_scatter.glsl`, `closesthit.rchit`, `volume_closesthit.rchit`, `shadow_anyhit.rahit`, `hair_shadow_anyhit.rahit` shader'ları spir-v olarak derlenmeli (`.spv`).
2. **Görsel Kalite Audit (Cycles Karşılaştırması)**:
   - SSS Radius: RGB bazlı sönümleme farklarının (örneğin kırmızının daha derine işlemesi) doğrulanması.
   - SSS Anisotropy & IOR: Işık kaynağı nesnenin arkasındayken saçılım halkasının biçimi ve kenar yumuşaklığının kontrolü.
3. **Performans / TDR Audit**:
   - 4K viewport render altında Russian Roulette kesintilerinin adımları erken bitirip bitirmediği ve GPU frame süresinin doğrulanması.
