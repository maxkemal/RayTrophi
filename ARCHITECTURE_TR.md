# 📊 RayTrophi Studio: Teknik Özellikler ve Mimari Raporu
**Rapor Tarihi:** 23 Mayıs 2026
**Durum:** Aktif Geliştirme / Çoklu Backend GPU Stylize Aşaması
**Yazar:** Kemal Demirtaş (maxkemal)

---

## 🛠️ Proje Genel Bakış
RayTrophi Studio, profesyoneller için tasarlanmış yüksek performanslı, modüler bir **Path Tracing** motorudur. **NVIDIA OptiX (CUDA)** ve **Vulkan Ray Tracing**backendlerini destekleyen hibrit bir mimariye sahiptir. Gelişmiş arazi erozyon fiziği, iskeletsel animasyon sistemleri ve profesyonel düzeyde düğüm tabanlı (node-based) bir editörü bir araya getirir.

---

## 📉 1. Kod Tabanı İstatistikleri (Doğrulanmış)
*23 Mayıs 2026 tarihinde repodan ölçülmüştür. Sayımlar `raytrac_sdl2/source` ağacını kapsar; proje sayımları vendored tek-dosya kütüphaneleri (`simdjson`, `stb`, `json.hpp`, `tinyexr`) hariç tutar.*

| Katman | Dosya Sayısı | Satır Sayısı | Açıklama |
| :--- | :---: | :---: | :--- |
| **Proje Kodu + Shaderlar** | 327 | 207.957 | Motora ait C++, CUDA, Vulkan GLSL ve OptiX shader kaynakları. |
| **GPU Kernel + Shader Dosyaları** | 52 | 21.125 | CUDA kernelleri, OptiX PTX kaynakları, Vulkan RT shaderları ve compute shaderlar. |
| **Tüm Source Tree** | 333 | 467.165 | Proje kodu ile `raytrac_sdl2/source` altındaki embedded tek-dosya kütüphaneler. |
| **Vendored External Ağaçları** | Harici | Bu sayımda yok | vcpkg, Ozz ve NanoVDB gibi büyük üçüncü parti ağaçlar doğrulanmış proje LoC sayımına dahil edilmemiştir. |

### 📂 Yapısal Kırılım (Kaynak Kod Odaklı)
*   **UI & Editör (53,6k Satır):** Modern arayüz, timeline, hiyerarşi, materyal, terrain, mesh-paint ve özellik editörleri.
*   **Backend Katmanı (19,9k Satır):** Vulkan RT, OptiX, backend soyutlaması ve viewport backend entegrasyonu.
*   **Fizik & Düğümler (19,0k Satır):** Hidrolik/Termal erozyon, terrain nodes, VDB işleme, gaz, sıvı ve okyanus sistemleri.
*   **Render Çekirdeği (16,3k Satır):** Embree BVH entegrasyonu, renderer orkestrasyonu, OptiX wrapper, ışık örnekleme ve acceleration manager yapıları.

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
*   **[Assimp](https://github.com/assimp/assimp):** Güçlü 3D model içe aktarma (40+ format).
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
1.  **Ölçeklenebilirlik:** 200k+ satırlık motora ait kod/shader yüzeyine sahip büyük, çoklu backend grafik kod tabanını yönetme deneyimi.
2.  **Full-Stack Grafik:** Hem düşük seviyeli GPU çekirdeklerinde hem de yüksek seviyeli UI/UX mantığında yetkinlik.
3.  **Karmaşık Mühendislik:** Fizik, matematik ve profesyonel yazılım mimarisinin derin entegrasyonu.

---
**Not:** Otomatik kod tabanı analizi ve mimari tarama yoluyla oluşturulmuştur.
