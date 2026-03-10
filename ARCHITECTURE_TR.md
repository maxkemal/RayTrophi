# 📊 RayTrophi Studio: Teknik Özellikler ve Mimari Raporu
**Rapor Tarihi:** 10 Mart 2026  
**Durum:** Aktif Geliştirme / Modüler Aşama  
**Yazar:** Kemal Demirtaş (maxkemal)

---

## 🛠️ Proje Genel Bakış
RayTrophi Studio, profesyoneller için tasarlanmış yüksek performanslı, modüler bir **Path Tracing** motorudur. **NVIDIA OptiX (CUDA)** ve **Vulkan Ray Tracing**backendlerini destekleyen hibrit bir mimariye sahiptir. Gelişmiş arazi erozyon fiziği, iskeletsel animasyon sistemleri ve profesyonel düzeyde düğüm tabanlı (node-based) bir editörü bir araya getirir.

---

## 📉 1. Kod Tabanı İstatistikleri (Doğrulanmış)
*Mühendislik eforunun dökümü; özgün mantık, dış bağımlılıklardan titizlikle ayrılmıştır.*

| Katman | Dosya Sayısı | Satır Sayısı | Açıklama |
| :--- | :---: | :---: | :--- |
| **C++ Core (Özgün)** | 212 | 119.700 | Motor mantığı, UI iskeleti ve sistem yönetimi. |
| **GPU Kernels (CUDA/Shader)** | 37 | 16.484 | Işın takibi (path tracing), kesişim ve hesaplama kernelleri. |
| **Dahili Kütüphaneler** | ~180 | ~362.000 | ImGui, NanoVDB, Simdjson, STB. |
| **Vcpkg Bağımlılıkları** | ~6.000 | ~1.103.800 | OpenVDB, Boost, OpenEXR, TBB. |
| **GENEL TOPLAM** | **~6.430** | **~1.602.000** | **Proje tarafından yönetilen toplam kod satırı.** |

### 📂 Yapısal Kırılım (Kaynak Kod Odaklı)
*   **Fizik & Düğümler (17k Satır):** Hidrolik/Termal erozyon, VDB işleme ve sıvı simülasyonları.
*   **UI & Editör (21k Satır):** Modern arayüz, zaman çizelgesi (timeline) ve özellik editörleri.
*   **Render Çekirdeği (13k Satır):** Embree BVH entegrasyonu ve ışık örnekleme mantığı.

---

## 🕹️ 2. Kullanıcı Etkileşimi ve Karmaşıklık
*Son kullanıcının motor üzerindeki kontrol gücünün nicelleştirilmesi.*

### ⌨️ Kullanıcı Arayüzü Kontrol Noktaları
*   **Sayısal Girişler (Slider/Drag):** 407 (Işık, malzeme ve fizik ince ayarları).
*   **Eylem Butonları:** 189 (Komut çalıştırma, araçlar, dosya işlemleri).
*   **Seçiciler (Checkbox/Combo):** 210 (Özellik aç/kapat, algoritma seçimleri).
*   **Toplam Etkileşim Elemanı:** **1.000+ Aktif Kontrol Noktası.**

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
*   **[Assimp](https://github.com/assimp/assimp):** Güçlü 3D model içe aktarma (40+ format).
*   **[NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb):** GPU dostu seyreltik hacim gösterimi.
*   **[Intel OIDN](https://www.openimagedenoise.org/):** Profesyonel düzeyde yapay zeka tabanlı "denoising" (gürültü giderme). Çift mod desteği ile entegre edilmiştir:
    *   **CPU Modu:** Intel TBB ve ISPC kernellerini kullanarak yüksek başarımlı gürültü giderme.
    *   **CUDA Modu:** NVIDIA donanımlarında gerçek zamanlı, yüksek kaliteli önizleme için GPU hızlandırmalı gürültü giderme.
*   **[ImGui](https://github.com/ocornut/imgui):** Hafif ve etkileşimli grafik kullanıcı arayüzü.

---

## 🚀 Bu Rapor Neden Önemli?
Geliştiriciler ve işverenler için bu rapor şunları göstermektedir:
1.  **Ölçeklenebilirlik:** Milyon satırı aşan bir kod tabanını yönetme deneyimi.
2.  **Full-Stack Grafik:** Hem düşük seviyeli GPU çekirdeklerinde hem de yüksek seviyeli UI/UX mantığında yetkinlik.
3.  **Karmaşık Mühendislik:** Fizik, matematik ve profesyonel yazılım mimarisinin derin entegrasyonu.

---
**Not:** Otomatik kod tabanı analizi ve mimari tarama yoluyla oluşturulmuştur.
