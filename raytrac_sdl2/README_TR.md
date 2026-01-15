# RayTrophi - Gelişmiş OptiX & Hibrit Ray Tracing Motoru

RayTrophi, **NVIDIA OptiX 7**, **SDL2**, **ImGui** ve **OpenVDB (NanoVDB)** ile geliştirilmiş, yüksek performanslı ve modüler bir ray tracing motorudur. Gerçek zamanlı önizleme ile offline path tracing arasındaki boşluğu doldurur; hacimsel efektler (volumetrics), node tabanlı arazi üretim sistemi ve tam kapsamlı animasyon zaman çizgisi gibi gelişmiş özellikler sunar.

![RayTrophi](RayTrophi_image.png)

## 🚀 Temel Özellikler

### 🔥 Hacimsel Render (VDB) (YENİ)
- **OpenVDB / NanoVDB Desteği:** Standart `.vdb` dosyalarını ve sequence'ları (dizi) içe aktarın.
- **Sequence Oynatma:** Patlama, duman ve ateş gibi hacimsel animasyonları gerçek zamanlı oynatın.
- **GPU Path Tracing:** NVIDIA GPU hızlandırmalı tam hacimsel render.
- **Blackbody Işıması:** Fiziksel tabanlı, sıcaklığa (Temperature) göre ateş/patlama ışıklandırması.
- **Hibrit Destek:** GPU yoksa veya yetersizse otomatik CPU render moduna geçiş.

### 🎬 Animasyon Sistemi
- **Timeline & Keyframe:** Objeler, Işıklar, Kameralar ve Dünya (World) özellikleri için animasyon.
- **Graph Editor:** Node (düğüm) tabanlı animasyon kontrolü.
- **Animasyon Render Modu:** `render_Animation` döngüsü ile kare kare animasyon çıktısı (Image Sequence).
- **Skinning:** Temel karakter animasyonu desteği (CPU skinning -> GPU upload).

### 🌍 Arazi & Çevre (Terrain & Environment)
- **Terrain Node Sistemi (V2):** 
  - Grafik tabanlı arazi üretimi (Perlin, Erozyon, Hidrolik Aşınma).
  - **AutoSplat:** Eğim ve Yüksekliğe göre otomatik kaplama/doku.
  - **Splat Haritaları:** Maskeleri PNG olarak dışa aktarma.
- **Su Sistemi:**
  - **FFT Okyanus:** Gerçek zamanlı derin okyanus simülasyonu.
  - **Gerstner Dalgaları:** Kıyı ve göl dalga efektleri.
  - **Nehir Editörü:** Bezier eğrileri (spline) ile nehir yatağı çizim aracı.
- **Atmosfer:**
  - Nishita Gökyüzü Modeli (Spektral Gece/Gündüz Döngüsü).
  - Hacimsel Sis (Fog) ve God Ray efektleri.
  - Yüksekliğe göre yoğunlaşan Çift Katmanlı (Dual-Lobe) Bulutlar.

### 🖌️ Sahne Editörü & Araçlar
- **Scatter Fırçası:** Çimen, ağaç ve diğer objeleri doğrudan arazi üzerine boyayarak yerleştirin.
- **Terrain Fırçası:** Araziyi (Heightmap) gerçek zamanlı olarak şekillendirin ve boyayın.
- **Gizmolar:** Blender tarzı Translasyon, Rotasyon ve Ölçekleme araçları.
- **Undo/Redo (Geri/İleri Al):** Tüm sahne işlemleri için gelişmiş komut geçmişi.
- **Varlık Yönetimi:** GLTF/GLB modellerini materyalleriyle birlikte içeri aktarın.

### 🎨 Render Çekirdeği
- **Hibrit Motor:** 
  - **GPU:** OptiX 7 (RTX Hızlandırmalı) Path Tracing.
  - **CPU:** Intel Embree / Paralel BVH Fallback (Yedek).
- **Materyaller:** Principled BSDF (Disney), Cam, Metal, Emisyon, Volumetric.
- **Denoiser:** Entegre OIDN (Open Image Denoise) ile temiz önizlemeler.

## 🎮 Kontroller

### Viewport Gezinme
- **Yörünge (Orbit):** Orta Fare Tuşu Sürükle
- **Kaydırma (Pan):** Shift + Orta Fare Tuşu Sürükle
- **Yakınlaşma (Zoom):** Fare Tekerleği veya Ctrl + Orta Fare Tuşu
- **Odaklanma:** `F` (Seçili objeye odaklan)

### Araçlar & Düzenleme
- **Seçim:** Sol Tık
- **Gizmo Modları:** `G` (Taşı), `R` (Döndür), `S` (Ölçekle)
- **Kopyala (Duplicate):** `Shift + Sürükle`
- **Sil:** `Del` veya `X`
- **Geri/İleri Al:** `Ctrl+Z` / `Ctrl+Y`
- **Animasyonu Oynat:** `Boşluk (Space)`

### Render
- **Final Render:** `F12`
- **Animasyon Render:** (Render Panelinden Başlatılır)

## 🔧 Derleme Talimatları (Build)
1. **Gereksinimler:**
   - Visual Studio 2022
   - NVIDIA Sürücüleri (Güncel)
   - CUDA Toolkit 11.x veya 12.x
   - OptiX 7.x SDK (Ortam Değişkeni: `OPTIX7_PATH`)
2. **Kurulum:**
   - `raytrac_sdl2.sln` dosyasını açın.
   - `vcpkg` bağımlılıklarının kurulu olduğundan emin olun (SDL2, ImGui, Assimp, OIDN, OpenVDB/NanoVDB).
3. **Derleme:**
   - `Release` modunu seçin.
   - Çözümü Derle (`Ctrl+Shift+B`).
4. **Çalıştırma:**
   - `raytracing_render_code.exe` uygulamasını başlatın.

## 📜 Lisans
Geliştirici: **Kemal DEMİRTAŞ**.
Bu proje eğitim ve portfolyo amaçlıdır.
