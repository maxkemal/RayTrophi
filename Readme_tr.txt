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