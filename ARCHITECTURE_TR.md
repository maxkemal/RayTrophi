# 📊 RayTrophi Studio: Teknik Özellikler ve Mimari Raporu
**Rapor Tarihi:** 11 Ekim 2026
**Durum:** Aktif Geliştirme / Çoklu Backend GPU Stylize Aşaması
**Yazar:** Kemal Demirtaş (maxkemal)

---

## 🛠️ Proje Genel Bakış
RayTrophi Studio, profesyoneller için tasarlanmış yüksek performanslı, modüler bir **Path Tracing** motorudur. **NVIDIA OptiX (CUDA)** ve **Vulkan Ray Tracing**backendlerini destekleyen hibrit bir mimariye sahiptir. Gelişmiş arazi erozyon fiziği, iskeletsel animasyon sistemleri ve profesyonel düzeyde düğüm tabanlı (node-based) bir editörü bir araya getirir.

---

## 📉 1. Kod Tabanı İstatistikleri (Doğrulanmış)
*11 Ekim 2026 tarihinde depodan doğrulanarak ölçülmüştür. Üçüncü parti harici kütüphaneler (`simdjson`, `nlohmann/json`, `miniz`, `stb`, `tinyexr`, `imgui`, `JoltPhysics`, `ozz-animation`, `nanovdb`) ve çalışma zamanı veri/asset dosyaları (JSON şablonları, presetler, 3D modeller) birinci parti kodlardan net olarak ayrıştırılmıştır.*

### 🛠️ Kendi Yazdığımız Kodlar (Motor ve Ekosistem)

| Katman / Bileşen | Dosya Türleri | Dosya Sayısı | Satır Sayısı | Net Kod Satırı (Boşluksuz) | Açıklama |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **C++ Motor Çekirdeği & UI** | `.h`, `.hpp`, `.inl`, `.cpp` | 1.180 | 541.459 | 492.365 | Motor mimarisi, sahne grafları, Vulkan/OptiX hatları, UI & editörler |
| **CUDA Çekirdekleri** | `.cu`, `.cuh` | 31 | 18.135 | 15.648 | GPU donanım hızlandırmalı fizik, MPM ve simülasyon kernelleri |
| **Shaderlar (Vulkan / RT / Compute)** | `.comp`, `.glsl`, `.vert`, `.frag`, `.rchit` vb. | 364 | 43.216 | 39.858 | Path tracing shaderları, compute simülasyon adımları, post-process |
| **RayTrophiAgent** | Python (`.py`) | 17 | 1.699 | 1.474 | Otonom AI ajan çekirdeği, IPC araçları ve çoklu model orkestrasyonu |
| **Geliştirme, IPC & Test Betikleri** | `scripts/` (`.py`, `.ps1`) | 364 | 57.027 | 51.665 | Harici named-pipe IPC test probları, doğrulama araçları, derleme betikleri |
| **Kök Dizin Yardımcıları** | `.hpp`, `.py` | 3 | 202 | 185 | Kök yardımcıları ve IPC doğrulama betikleri |
| **TOPLAM (Kendi Yazdığımız)** | — | **1.959** | **661.738** | **601.195** | **Motora ve projeye ait birinci parti kaynak kodlar** |

### 📦 Harici ve Üçüncü Parti Kütüphaneler (Proje Kodundan Ayrılanlar)

| Kütüphane | Konum | Dosya Sayısı | Satır Sayısı | Açıklama |
| :--- | :--- | :---: | :---: | :--- |
| **simdjson** | `source/include/simdjson.h`, `src/simdjson.cpp` | 2 | 248.846 | Yüksek performanslı JSON ayrıştırıcı |
| **nlohmann/json** | `source/include/json.hpp` | 2 | 49.533 | Modern C++ JSON tek-başlıklı (single-header) kütüphane |
| **miniz / miniz_zip** | `external/miniz*` | 10 | 9.021 | Deflate / Zip arşiv sıkıştırma kütüphanesi |
| **stb_image & write** | `source/include/stb_image*` | 2 | 9.598 | Tek-başlıklı görsel okuma & yazma kütüphanesi |
| **tinyexr** | `source/include/tinyexr.h` | 1 | 9.405 | Tek-başlıklı OpenEXR yükleme/kaydetme |
| **ImGuizmo & UI Araçları** | `source/include/ImGuizmo*` | 2 | 3.486 | 3D transform gizmo aracı |
| **PNanoVDB** | `source/shaders/PNanoVDB.*` | 3 | 7.236 | NVIDIA NanoVDB shader entegrasyonu |
| **IconsMaterialDesign** | `source/include/IconsMaterialDesignIcons.h` | 1 | 7.461 | Oluşturulmuş ikon kod noktası haritalaması |
| **ImGui & ImSequencer** | `external/imgui/` | 18 | 67.153 | Hafif ve etkileşimli grafik kullanıcı arayüzü |
| **JoltPhysics** | `external/JoltPhysics/` | 444 | 103.636 | Katı cisim (rigid body) ve çarpışma fizik motoru |
| **ozz-animation** | `external/ozz-animation/` | 523 | 732.125 | İskeletsel animasyon ve harmanlama çalışma zamanı |
| **nanovdb** | `libs/nanovdb/` | 191 | 63.869 | Seyreltik hacimsel veri yapıları |
| **ufbx & cgltf** | `external/ufbx/`, `external/cgltf/` | 4 | 46.556 | FBX ve glTF 2.0 içe aktarma ayrıştırıcıları |
| **TOPLAM (Harici Kütüphaneler)** | — | **1.203** | **1.358.325** | *Harici bağımlılık ayak izi* |

### 📂 Yapısal Kırılım (Motor Kaynak Kodu Odaklı)
*   **UI & Editör Sistemleri (~109,2k Satır):** Modern docking arayüzü, düğüm editörleri, timeline bileşenleri, viewport kaplamaları, mesh-paint ve denetçi panelleri.
*   **Fizik, Simülasyon & Parçacıklar (~100,0k Satır):** Hidrolik/termal LEM erozyonu, granüler MPM çözücü, akışkanlar mekaniği, gaz/duman, okyanus su sistemleri ve VDB hacimleri.
*   **API, IPC & Betik Katmanı (~71,6k Satır):** Kanonik Python betik katmanı, harici named-pipe IPC sunucusu ve dinamik düğüm serileştirme.
*   **Backend & Donanım Renderı (~46,2k Satır):** Vulkan RT boru hattı, rasterizer önizleme, OptiX backend, bellek yığınları ve AI denoising (OIDN) entegrasyonu.
*   **Shaderlar & Hesaplama Aşamaları (~43,3k Satır):** SPIR-V / GLSL path tracer, akışkan/parçacık simülasyonları için compute shaderlar ve post-process kernelleri.
*   **Render Çekirdeği & Sahne Temsili (~34,2k Satır):** Düz TriangleMesh DNA SoA geometrisi, Embree BVH hızlandırması, ışık örnekleme ve kamera orkestrasyonu.
*   **Aygıt Soyutlaması & Donanım Mimarisi (~19,3k Satır):** Aygıt limitleri, kuyruk yönetimi, tanımlayıcı (descriptor) bağlama ve çit (fence) senkronizasyonu.

---

## 🕹️ 2. Kullanıcı Etkileşimi ve Karmaşıklık
*Son kullanıcının motor üzerindeki kontrol gücünün nicelleştirilmesi.*

### ⌨️ Kullanıcı Arayüzü Kontrol Noktaları
*   **Sayısal Girişler (Slider/Drag):** 512 (Işık, materyal, terrain, fizik ve stylize profil ince ayarları).
*   **Eylem Butonları:** 218 (Komut çalıştırma, araçlar, dosya işlemleri, bake/apply iş akışları).
*   **Seçiciler (Checkbox/Combo/Menu/List):** 514 (Özellik aç/kapat, algoritma, mod ve preset seçimleri).
*   **Metin/Değer Girişleri:** 57.
*   **Renk Kontrolleri:** 30.
*   **Tree/Collapsing Header Noktaları:** 110.
*   **Toplam Etkileşim Elemanı:** **1.278+ Aktif Kontrol Noktası.**

### 🧩 Dinamik Graf (Düğüm) Sistemleri
Motor, görsel programlama için **61'den fazla benzersiz düğüm tipi** sunar:
*   **Terrain Graph (36+ düğüm):** Gelişmiş erozyon iş akışları ve maske oluşturma.
*   **Animation Graph (14+ düğüm):** Durum makineleri (State machines), IK blend spaceler.
*   **Material Graph (11+ düğüm):** Prosedürel PBR shader kurgusu.

---

## 🏗️ 3. Teknoloji Yığını
Endüstri standardı çerçeveler ve donanım hızlandırmalı API'ler üzerine inşa edilmiştir:

*   **[NVIDIA OptiX](https://developer.nvidia.com/optix):** Donanım hızlandırmalı Ray Tracing.
*   **[Vulkan SDK](https://www.vulkan.org/):** Modern platformlar arası grafik ve hesaplama.
*   **[Intel Embree](https://github.com/embree/embree):** Yüksek performanslı CPU ışın izleme çekirdekleri.
*   **[cgltf](https://github.com/jkuhlmann/cgltf):** glTF 2.0 / GLB ayrıştırma; hem içe hem dışa aktarmada kullanılır.
*   **[ufbx](https://github.com/ufbx/ufbx):** FBX içe aktarma; skinning ve baked transform animasyonu dahil.
*   **[NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb):** GPU dostu seyreltik hacim gösterimi.
*   **[Intel OIDN](https://www.openimagedenoise.org/):** Profesyonel düzeyde yapay zeka tabanlı "denoising" (gürültü giderme). Çift mod desteği ile entegre edilmiştir:
    *   **CPU Modu:** Intel TBB ve ISPC kernellerini kullanarak yüksek başarımlı gürültü giderme.
    *   **CUDA Modu:** NVIDIA donanımlarında gerçek zamanlı, yüksek kaliteli önizleme için GPU hızlandırmalı gürültü giderme.
*   **[ImGui](https://github.com/ocornut/imgui):** Hafif ve etkileşimli grafik kullanıcı arayüzü.

---

## 🎮 Render Cihazları ve İş Yükü Dağılımı
Geliştiriciler ve yapay zeka ajanlarının doğru yönlendirilmesi için render mekanizmaları şu şekilde kurgulanmıştır:
*   **Vulkan RT (Varsayılan Render Aygıtı):** GPU tarafındaki birincil ve varsayılan path tracing / render motorumuzdur.
*   **Vulkan Raster (Önizleme ve Edit Modu):** Temel düzenleme (edit), sculpt, boyama (paint) gibi etkileşimli işlemler varsayılan olarak raster modunda çalıştırılır.
*   **OptiX (İkincil Render Aygıtı):** Şu an için GPU tarafında ikinci/alternatif render motoru olarak konumlandırılmıştır.
*   **Intel Embree (CPU Çekirdeği):** CPU tabanlı ray tracing ve temel sahne BVH yapısı için motorun çekirdek yapısını oluşturur.

---

## 🚀 Bu Rapor Neden Önemli?
Geliştiriciler ve işverenler için bu rapor şunları göstermektedir:
1.  **Ölçeklenebilirlik:** 600k+ satırlık motora ait kod/shader yüzeyine sahip büyük, çoklu backend grafik kod tabanını yönetme deneyimi.
2.  **Full-Stack Grafik:** Hem düşük seviyeli GPU çekirdeklerinde hem de yüksek seviyeli UI/UX mantığında yetkinlik.
3.  **Karmaşık Mühendislik:** Fizik, matematik ve profesyonel yazılım mimarisinin derin entegrasyonu.

---
**Not:** Otomatik kod tabanı analizi ve mimari tarama yoluyla oluşturulmuştur.
