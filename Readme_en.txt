# Ray Tracing Project

This project includes a ray tracing engine implemented in C++ using SIMD (Single Instruction, Multiple Data) instructions for performance optimization. It allows creating 3D scenes with various materials and light sources. Additionally, it features an OBJ loader for importing models and a simple scene builder.

## Features

- **Ray Tracing:** Realistic shadows and reflections are achieved using ray tracing for each pixel.
- **Materials and Lights:** Supports various materials such as Principled BSDF, Metal, Dielectric, Volumetric and light sources including directional lights, point lights, etc.
- **Assimp Loader:** Includes a loader function to import models in few format into the scene.
- **SIMD and AVX256 Optimization:** Vector operations are optimized using SIMD (Vec3SIMD) to improve performance.
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

