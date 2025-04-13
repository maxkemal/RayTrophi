#include <SDL_main.h> 
#include <fstream>
#include <locale>
#include <SDL_image.h>
#include "Renderer.h"
#include "CPUInfo.h"

std::shared_ptr<ParallelBVHNode> build_bvh(const std::vector<std::shared_ptr<Hittable>>& objects, double time0, double time1) {
    return std::make_shared<ParallelBVHNode>(objects, 0, objects.size(), time0, time1);
}

inline float abs(const Vec3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}



// Nokta dönüþtürme fonksiyonu
Vec3 transform_point(const Matrix4x4& mat, const Vec3& point) {
    double x = mat.m[0][0] * point.x + mat.m[0][1] * point.y + mat.m[0][2] * point.z + mat.m[0][3];
    double y = mat.m[1][0] * point.x + mat.m[1][1] * point.y + mat.m[1][2] * point.z + mat.m[1][3];
    double z = mat.m[2][0] * point.x + mat.m[2][1] * point.y + mat.m[2][2] * point.z + mat.m[2][3];
    double w = mat.m[3][0] * point.x + mat.m[3][1] * point.y + mat.m[3][2] * point.z + mat.m[3][3];

    // Perspektif bölme
    if (w != 1 && w != 0) {
        x /= w;
        y /= w;
        z /= w;
    }

    return Vec3(x, y, z);
}

int main(int argc, char* argv[]) {

    setlocale(LC_ALL, "Turkish");   
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }
   
    SDL_Window* window = SDL_CreateWindow("Ray Tracing", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, image_width, image_height, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (window == nullptr) {
        std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Surface* surface = SDL_GetWindowSurface(window);
    if (surface == nullptr) {
        std::cerr << "SDL_GetWindowSurface Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    CPUInfo::initialize();
    CPUInfo::print_cpu_info();

    if (CPUInfo::has_feature("AVX2")) {
        std::cout << "Rendering started with AVX2 support..." << std::endl;
     
     // Renderer::normalMapBuffer.reserve(image_width* image_height);
        Renderer renderer(image_width, image_height,1, 1);
        renderer.render_image(surface, window, 20,5);
       // renderer.render_Animation(surface, window, 1, 1,30,1);
    }
    else {
        CPUInfo::print_cpu_info();
        std::cerr << "Error: AVX2 support is required for rendering. Rendering aborted!" << std::endl;
        // Render iþlemi baþlatýlmaz
    }
    
    // Render iþlemi bittikten sonra pencereyi açýk tutan döngü
    bool quit = false;
    SDL_Event e;
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = true; // Kapatma isteði geldiðinde döngüden çýk
            }
        }
        // Küçük bir gecikme ekleyerek iþlemciyi fazla meþgul etmemek için bekleyin
        SDL_Delay(100);
    }

    SDL_FreeSurface(surface);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}




