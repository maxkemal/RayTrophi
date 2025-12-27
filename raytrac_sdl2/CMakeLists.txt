cmake_minimum_required(VERSION 3.18)
project(RayTrophi VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 50)  # SM 5.0 for broad GPU support

# --- WINDOWS SETTINGS (Match VS2022 Project) ---
if(WIN32)
    add_compile_definitions(
        WIN32
        _WINDOWS
        NOMINMAX           # Prevent Windows min/max macros
        _CRT_SECURE_NO_WARNINGS
        UNICODE
        _UNICODE
    )
    
    # MSVC specific flags
    if(MSVC)
        # Match VS2022 Release configuration
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS")
        
        # Multi-processor compilation
        add_compile_options(/MP)
    endif()
endif()

# --- USER DEFINED PATHS (Overridable via Cache) ---
set(SDL2_ROOT "E:/SDL2-2.30.4" CACHE PATH "Path to SDL2 root directory")
set(SDL2_IMAGE_ROOT "E:/SDL2-2.30.4/SDL2_image-2.8.2" CACHE PATH "Path to SDL2 Image root directory")
set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0" CACHE PATH "Path to OptiX SDK")

# Paths from dis_bagimliliklar
set(EMBREE_ROOT "E:/RayTrophi_projesi/dis_bagimliliklar/embree-4.4.0.x64.windows" CACHE PATH "Path to Embree root")
set(OIDN_ROOT "E:/RayTrophi_projesi/dis_bagimliliklar/oidn-2.3.0.x64.windows" CACHE PATH "Path to OIDN root")
set(ASSIMP_ROOT "E:/RayTrophi_projesi/dis_bagimliliklar/Assimp" CACHE PATH "Path to Assimp root")

# --- SOURCE FILES ---
set(SOURCES_CPP
    "source/src/AABB.cpp"
    "source/src/AreaLight.cpp"
    "source/src/AssimpLoader.cpp"
    "source/src/Camera.cpp"
    "source/src/Dielectric.cpp"
    "source/src/DirectionalLight.cpp"
    "source/src/EmbreeBVH.cpp"
    "source/src/HittableList.cpp"
    "source/src/ImGuizmo.cpp"
    "source/src/Light.cpp"
    "source/src/Main.cpp"
    "source/src/Material.cpp"
    "source/src/MaterialManager.cpp"
    "source/src/Matrix4x4.cpp"
    "source/src/Mesh.cpp"
    "source/src/OptixWrapper.cpp"
    "source/src/OptixAccelManager.cpp"
    "source/src/ParallelBVHNode.cpp"
    "source/src/PointLight.cpp"
    "source/src/PrincipledBSDF.cpp"
    "source/src/ProjectManager.cpp"
    "source/src/Renderer.cpp"
    "source/src/SceneCommand.cpp"
    "source/src/SceneHistory.cpp"
    "source/src/SceneSelection.cpp"
    "source/src/SceneSerializer.cpp"
    "source/src/SequencerAdapter.cpp"
    "source/src/SplashScreen.cpp"
    "source/src/SpotLight.cpp"
    "source/src/Texture.cpp"
    "source/src/ThreadLocalRNG.cpp"
    "source/src/TimelineWidget.cpp"
    "source/src/Triangle.cpp"
    "source/src/Vec2.cpp"
    "source/src/Vec3.cpp"
    "source/src/Vec3SIMD.cpp"
    "source/src/Volumetric.cpp"
    "source/src/World.cpp"
    "source/src/globals.cpp"
    "source/src/scene_ui.cpp"
    "source/src/scene_ui_animation.cpp"
    "source/src/scene_ui_camera.cpp"
    "source/src/scene_ui_gizmos.cpp"
    "source/src/scene_ui_hierarchy.cpp"
    "source/src/scene_ui_lights.cpp"
    "source/src/scene_ui_materials.cpp"
    "source/src/scene_ui_selection.cpp"
    "source/src/scene_ui_viewport.cpp"
    "source/src/scene_ui_world.cpp"
    "source/src/ui_modern.cpp"
    "source/src/unified_converters.cpp"
)
file(GLOB SOURCES_IMGUI "source/external/imgui/*.cpp")
file(GLOB HEADERS "source/include/*.h" "source/external/imgui/*.h")

# miniz compression library (Public Domain)
set(MINIZ_SOURCES
    "source/external/miniz.c"
)



# Gather all sources
# Gather all sources
set(ALL_SOURCES ${SOURCES_CPP} ${SOURCES_IMGUI} ${MINIZ_SOURCES} ${HEADERS})

# --- CUDA / OPTIX ---
find_package(CUDAToolkit REQUIRED)

set(CUDA_SOURCE "source/src/raygen.cu")
set(PTX_OUTPUT "${CMAKE_BINARY_DIR}/raygen.ptx")

# Custom command to generate PTX
# NOTE: PTX depends on these headers too - if they change, PTX must be rebuilt
set(PTX_HEADER_DEPS
    "source/include/material_gpu.h"
    "source/include/params.h"
    "source/include/payload.h"
    "source/include/sbt_data.h"
    "source/src/material_scatter.cuh"
    "source/src/vec3_utils.cuh"
    "source/src/random_utils.cuh"
    "source/src/ray_color.cuh"
    "source/src/trace_ray.cuh"
)

add_custom_command(
    OUTPUT ${PTX_OUTPUT}
    COMMAND ${CMAKE_CUDA_COMPILER}
    -ptx "${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}"
    -o "${PTX_OUTPUT}"
    -I"${OPTIX_ROOT}/include"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/source/include"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/source/src"
    --use_fast_math
    --maxrregcount=64
    -gencode=arch=compute_50,code=compute_50
    -allow-unsupported-compiler
    DEPENDS ${CUDA_SOURCE} ${PTX_HEADER_DEPS}
    COMMENT "Compiling CUDA to PTX: ${CUDA_SOURCE}"
)

# --- EXECUTABLE ---
add_executable(${PROJECT_NAME} ${ALL_SOURCES} ${PTX_OUTPUT})

# --- INCLUDE DIRECTORIES ---
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/source/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/external"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/external/imgui"
    "${OPTIX_ROOT}/include"
    "${SDL2_ROOT}/include"
    "${SDL2_IMAGE_ROOT}/include"
    "${EMBREE_ROOT}/include"
    "${OIDN_ROOT}/include"
    "${ASSIMP_ROOT}/include"
)

# --- LIBRARIES (FULL PATHS) ---
# SDL2
set(LIB_SDL2 "${SDL2_ROOT}/lib/x64/SDL2.lib")
set(LIB_SDL2MAIN "${SDL2_ROOT}/lib/x64/SDL2main.lib")
set(LIB_SDL2IMAGE "${SDL2_IMAGE_ROOT}/lib/x64/SDL2_image.lib")

# Embree
set(LIB_EMBREE "${EMBREE_ROOT}/lib/embree4.lib")
set(LIB_TBB "${EMBREE_ROOT}/lib/tbb12.lib")

# OIDN
set(LIB_OIDN "${OIDN_ROOT}/lib/OpenImageDenoise.lib")
set(LIB_OIDN_CORE "${OIDN_ROOT}/lib/OpenImageDenoise_core.lib")

# Assimp
set(LIB_ASSIMP "${ASSIMP_ROOT}/lib/x64/assimp-vc143-mt.lib")

# Link Libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${LIB_SDL2}
    ${LIB_SDL2MAIN}
    ${LIB_SDL2IMAGE}
    ${LIB_EMBREE}
    ${LIB_TBB}
    ${LIB_OIDN}
    ${LIB_OIDN_CORE}
    ${LIB_ASSIMP}
    CUDA::cudart
    CUDA::cuda_driver
    kernel32 user32 gdi32 winspool comdlg32 advapi32 shell32 ole32 oleaut32 uuid
)

# --- POST BUILD COPY ---
# Copy .ptx to executable directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PTX_OUTPUT}"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/raygen.ptx"
    COMMENT "Copying raygen.ptx to output directory"
)

message(STATUS "Build setup complete for RayTrophi")
