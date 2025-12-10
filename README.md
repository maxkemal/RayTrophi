# Ray Tracing Project

This project includes a ray tracing engine implemented in C++ using SIMD (Single Instruction, Multiple Data) instructions for performance optimization. It allows creating 3D scenes with various materials and light sources. Additionally, it features an OBJ loader for importing models and a simple scene builder.

## Features

- **Ray Tracing:** Realistic shadows and reflections are achieved using ray tracing for each pixel.
- **Materials and Lights:** Supports various materials such as Lambertian, Metal, Dielectric, volumetric, sss material and different light sources including directional lights, point lights, area light etc.
- **OBJ Loader:** Includes a loader function to import models in OBJ format into the scene.
- **SIMD Optimization:** Vector operations are optimized using SIMD (Vec3SIMD) to improve performance.
- **BVH:** Uses an AABB Hierarchical Bounding Volume Hierarchy (BVH) for fast ray-scene intersections.

## Usage

The project compiles with C++20 and runs in an environment that supports SIMD instructions. Follow these steps to compile and run the project:

1. **Requirements:**
   - A compiler that supports C++20 (e.g., Visual Studio 2022).
   - SDL2 library (used for graphics output).

2. **Compilation:**
   - Use the `CMakeLists.txt` file in the project folder to compile the project.
   - Link necessary libraries and files during compilation.

3. **Execution:**
   - After compilation, run the generated executable to render example scenes.
   - Outputs are saved as image files such as `output.png`.

## Example Code

```cpp
// Basic functions for scene creation and rendering

#include "Renderer.h"

int main() {
    // Scene creation
    std::vector<std::shared_ptr<Light>> lights;
    auto [world, bvh] = create_scene(lights, background_color);

    // Rendering
    Renderer renderer;
    renderer.render_image(world, bvh, lights, camera, image_width, image_height, samples_per_pixel);

    return 0;
}

Contributing
Contributions are welcome! If you want to work on or improve the project, submit a pull request. Any feedback, questions, or suggestions can be communicated through the issues section.

Türkçe
# Ray Tracing Projesi

Bu proje, C++ ve SIMD (Yığın İşlemci Komutları) kullanarak ray tracing yöntemini uygulayan bir ışın izleme motoru içerir. Projede 3 boyutlu sahneler oluşturulabilir, farklı malzemeler ve ışık kaynakları kullanılabilir. Ayrıca, OBJ dosyalarını yüklemek için bir yükleyici ve basit bir sahne oluşturucu bulunmaktadır.

## Özellikler

- **Ray İzleme:** Her piksel için ışın izleme yöntemi kullanılarak gerçekçi gölgeler ve yansımalar elde edilir.
- **Malzemeler ve Işıklar:** Lambertian, Metal, Dielektrik gibi farklı malzemeler ve yönlü ışıklar, nokta ışıklar gibi çeşitli ışık kaynakları desteklenir.
- **OBJ Yükleyici:** Sahneye OBJ formatında modeller yüklemek için bir yükleyici işlevi mevcuttur.
- **SIMD Optimizasyon:** Performansı artırmak için SIMD (Vec3SIMD) kullanılarak vektörel işlemler desteklenir.
- **BVH:** Hızlı ışın-sahne etkileşimleri için AABB Hiyerarşik Bölümleme yapısı (BVH) kullanılır.

## Kullanım

Proje, C++20 standardı ile derlenir ve SIMD işlemcileri destekleyen bir ortamda çalışır. Aşağıdaki adımları izleyerek projeyi derleyebilir ve çalıştırabilirsiniz:

1. **Gereksinimler:**
   - C++20 destekleyen bir derleyici (örneğin, Visual Studio 2022).
   - SDL2 kütüphanesi (grafik çıktı için kullanılır).

2. **Derleme:**
   - Proje klasöründe bulunan `CMakeLists.txt` dosyasını kullanarak projeyi derleyin.
   - Derleme sırasında gerekli kütüphaneleri ve dosyaları bağlayın.

3. **Çalıştırma:**
   - Derleme tamamlandıktan sonra oluşan yürütülebilir dosyayı çalıştırarak örnek sahneleri render edin.
   - Çıktılar, görüntü dosyaları şeklinde `output.png` olarak kaydedilir.

## Örnek Kod

```cpp
// Sahne oluşturma ve render işlemi için temel işlevler

#include "Renderer.h"

int main() {
    // Sahne oluşturma
    std::vector<std::shared_ptr<Light>> lights;
    auto [world, bvh] = create_scene(lights, background_color);

    // Render işlemi
    Renderer renderer;
    renderer.render_image(world, bvh, lights, camera, image_width, image_height, samples_per_pixel);

    return 0;
}
Katkıda Bulunma
Katkılarınızı memnuniyetle karşılıyoruz! Proje üzerinde çalışmak veya geliştirmeler yapmak isterseniz bir çekme isteği (pull request) gönderin. Her türlü geri bildirimi, soruyu veya öneriyi issues bölümünden iletebilirsiniz.
