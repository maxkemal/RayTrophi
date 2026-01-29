# RayTrophi - Gelişmiş OptiX & Hibrit Ray Tracing Motoru

RayTrophi, **NVIDIA OptiX 7**, **SDL2**, **ImGui** ve **OpenVDB (NanoVDB)** ile geliştirilmiş, yüksek performanslı ve modüler bir ray tracing motorudur. Gerçek zamanlı önizleme ile offline path tracing arasındaki boşluğu doldurur; hacimsel efektler (volumetrics), node tabanlı arazi üretim sistemi ve tam kapsamlı animasyon zaman çizgisi gibi gelişmiş özellikler sunar.

![RayTrophi](RayTrophi_image.png)

## 🚀 Temel Özellikler

### 🌪️ Gaz & Akışkan Simülasyonu (YENİ)
- **Fiziksel Gaz Çözücü:** GPU hızlandırmalı sıvı/duman simülasyonu, gerçek zamanlı geri bildirim.
- **Kuvvet Alanları (Force Fields):** Point, Directional, Vortex, Turbulence ve Drag kuvvetleri ile simülasyonu yönlendirin.
- **Hacimsel Render (Volumetric):** Gerçekçi ateş ve patlamalar için Dual-Lobe Phase, Multi-Scattering ve Blackbody ışınımı.
- **OpenVDB / NanoVDB Desteği:** Karmaşık hacimsel veriler için standart `.vdb` dosyalarını yükleyin.

### 🎬 Gelişmiş Animasyon Sistemi
- **Animasyon Grafiği (AnimGraph):** Karakter mantığı için Durum Makinesi (State Machine) (Idle -> Walk -> Run).
- **Blend Spaces:** Parametrelere göre animasyonlar arası pürüzsüz geçiş (Örn: Hız, Yön).
- **Timeline & Keyframe:** Objeler, Işıklar, Kameralar ve Dünya özellikleri (Gökyüzü, Bulut Yoğunluğu vb.) için tam animasyon.
- **Skinning Desteği:** Karakter meshleri için GPU hızlandırmalı skinning.

### 🌍 Arazi & Bitki Örtüsü (Foliage)
- **Foliage Boyama:** Milyonlarca ağaç, çim ve kaya örneğini GPU optimizasyonu ile araziye boyayın. Fırça yarıçapı, yoğunluk ve yüzey hizalama kontrolleri içerir.
- **Terrain Node Sistemi (V2):** 
  - Hidrolik Erozyon simülasyonu.
  - Prosedürel gürültü nodları (Perlin, Worley).
- **Su Sistemi:**
  - **FFT Okyanus:** Beyaz köpük (foam) efektli gerçek zamanlı derin okyanus.
  - **Nehir Aracı:** Akış haritaları ve türbülans efektleri ile Bezier eğrileri üzerinden nehir oluşturma.
- **Atmosfer:**
  - Nishita Gökyüzü Modeli (Spektral Gece/Gündüz Döngüsü).
  - Hacimsel Sis ve Hüzme Işıkları (God Rays).

### 🖌️ Sahne Editörü & Araçlar
- **Modern Arayüz (UI):** World, Terrain, Water ve Animation panelleri için modernize edilmiş, karanlık temalı ve düzenli arayüz.
- **Etkileşimli Gizmolar:** Taşıma, Döndürme ve Ölçekleme için Blender tarzı 3D manipülatörler.
- **Varlık Yönetimi:** Güçlü GLTF/GLB içe aktarma desteği.
- **Dokümantasyon:** Modern web arayüzüne sahip entegre çevrimdışı dokümantasyon.

### 🎨 Render Çekirdeği
- **Hibrit Motor:** 
  - **GPU:** OptiX 7 (RTX Hızlandırmalı) Path Tracing ve Instancing desteği.
  - **CPU:** Intel Embree / Paralel BVH Fallback.
- **Materyaller:** Principled BSDF (Disney), Cam, Metal, Emisyon, Volumetric.
- **Denoiser:** Temiz önizlemeler için Intel OIDN entegrasyonu.

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
