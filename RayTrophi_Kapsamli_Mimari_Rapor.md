# 🚀 RayTrophi Studio: Kapsamlı Mimari ve Kod Raporu
**Üretim Tarihi:** 10 Mart 2026, 23:18  
**Proje Durumu:** Aktif / Modüler Faz  
**Geliştirici:** Kemal Demirtaş (maxkemal)

---

## 💎 1. Proje Vizyonu ve Mimari Katmanlar
RayTrophi Studio, hibrit bir **Path Tracing** motoru olarak tasarlanmış olup, hem **NVIDIA OptiX (CUDA)** hem de **Vulkan Ray Tracing** backend'lerini destekleyen modüler bir mimariye sahiptir. Sadece bir render motoru değil, arazi üretimi, fiziksel simülasyon ve animasyon yönetimi sunan kapsamlı bir **DCC (Digital Content Creation)** aracıdır.

---

## 📊 2. Kod Hacmi ve Dosya İstatistikleri
*Projenin tüm bağımlılıkları ve özgün kodlarının detaylı analizi.*

### 📂 Dosya Dağılımı
| Kategori | Dosya Sayısı | Satır Sayısı | Açıklama |
| :--- | :---: | :---: | :--- |
| **C++ Core (Custom)** | 212 | 119.700 | Motor çekirdeği, UI ve Sistem yönetimi. |
| **GPU Kernels (CUDA/Shader)** | 37 | 16.484 | Raygen, Intersection ve Compute görevleri. |
| **Internal Libs (3. Parti)** | ~180 | ~362.000 | ImGui, NanoVDB, Simdjson, STB. |
| **vcpkg Bağımlılıkları** | ~6.000 | ~1.103.800 | OpenVDB, Boost, OpenEXR, TBB. |
| **GENEL TOPLAM** | **~6.430** | **~1.602.000** | **Ekosistemin toplam boyutu.** |

### 📁 Klasör Yapısı
*   `source/src/Core`: Ana döngü ve olay yönetimi.
*   `source/src/Render`: Embree BVH ve render mantığı.
*   `source/src/Physics`: Erozyon, VDB ve Akışkan simülasyonları.
*   `source/src/Device`: Donanıma özel (Vulkan/OptiX) kernel implementasyonları.
*   `source/src/UI`: Modern editör arayüzü ve widgetlar.
*   `source/include`: Projenin tüm tip tanımları ve API kontratları.

---

## 🕹️ 3. Kullanıcı Etkileşim Kapasitesi (UI/UX)
*Kullanıcının motor üzerinde sahip olduğu kontrol gücü.*

### ⌨️ Girdi ve Etkileşim Alanları
| Tip | Sayı | Görevi |
| :--- | :---: | :--- |
| **Sayısal Girişler (Slider/Drag)** | 407 | Işık, Malzeme ve Fizik parametreleri. |
| **Eylem Butonları** | 189 | Render başlatma, Dosya işlemleri, Araç seçimleri. |
| **Seçim Alanları (Checkbox/Combo)** | 210 | Özellik aç/kapat, Algoritma seçimleri. |
| **Menüler ve İkincil Girdiler** | 200+ | Sağ tık menüleri, Gizmo kontrolleri, Metin girişleri. |
| **TOPLAM ETKİLEŞİM** | **1.015+** | **Aktif Kontrol Noktası.** |

### 🧩 Dinamik Düğüm (Node) Sistemi
Kullanıcı, statik UI dışında **61+ benzersiz düğüm tipi** ile kendi mantığını oluşturabilir:
*   **Terrain Nodes (36+):** Erozyon wizard'larından maske boyamaya kadar.
*   **Animation Nodes (14+):** State machine ve blend space yapıları.
*   **Material Nodes (11+):** Procedural texture ve PBR shader graph.

---

## 🔌 4. Üçüncü Parti Yapılar ve Entegrasyonlar
*Projenin üzerine inşa edildiği teknoloji yığını:*

*   **[NVIDIA OptiX](https://developer.nvidia.com/optix):** Donanım hızlandırmalı Ray Tracing (CUDA tabanlı).
*   **[Vulkan SDK](https://www.vulkan.org/):** Cross-platform modern grafik ve compute API.
*   **[ImGui](https://github.com/ocornut/imgui):** Editör arayüzü ve debug araçları.
*   **[Assimp](https://github.com/assimp/assimp):** 40+ formatta 3D model yükleme desteği.
*   **[NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb):** GPU üzerinde seyreltik hacimsel veri yapıları.
*   **[SDL2](https://www.libsdl.org/):** Pencere yönetimi, input ve multimedya katmanı.
*   **[Intel Embree](https://github.com/embree/embree):** Yüksek performanslı CPU Ray Tracing kernels.

---

## 🛡️ 5. Güvenlik ve Lisans Notu
*   **Custom Code:** Kemal Demirtaş (Proprietary / MIT - Tercihe Bağlı).
*   **Libraries:** MIT, Apache 2.0 ve BSD lisansları altında yasal dağıtım.

---
**Önemli Not:** Bu rapor, projenin mevcudiyetini ve teknik derinliğini belgelemek amacıyla otomatize edilmiş kod analizi ve mimari tarama ile oluşturulmuştur.
