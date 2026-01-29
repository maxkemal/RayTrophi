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

    # OpenMP Support
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        link_libraries(OpenMP::OpenMP_CXX)
    endif()
    
    # MSVC specific flags
    if(MSVC)
        # Debug configuration
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi /RTC1")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /DEBUG /SUBSYSTEM:CONSOLE")
        
        # Release configuration
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /SUBSYSTEM:WINDOWS")
        
        # RelWithDebInfo
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /O2 /Zi /DNDEBUG")
        set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /DEBUG /SUBSYSTEM:CONSOLE")
        
        add_compile_options(/MP)
    endif()
endif()

# --- USER DEFINED PATHS ---
set(SDL2_ROOT "E:/SDL2-2.30.4" CACHE PATH "Path to SDL2 root directory")
set(SDL2_IMAGE_ROOT "E:/SDL2-2.30.4/SDL2_image-2.8.2" CACHE PATH "Path to SDL2 Image root directory")
set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0" CACHE PATH "Path to OptiX SDK")
set(EMBREE_ROOT "E:/RayTrophi_projesi/dis_bagimliliklar/embree-4.4.0.x64.windows" CACHE PATH "Path to Embree root")
set(OIDN_ROOT "E:/RayTrophi_projesi/dis_bagimliliklar/oidn-2.3.0.x64.windows" CACHE PATH "Path to OIDN root")
set(ASSIMP_ROOT "E:/RayTrophi_projesi/dis_bagimliliklar/Assimp" CACHE PATH "Path to Assimp root")

# --- SOURCE FILES (Refactored Structure) ---
# Automatically collect source files from new subdirectories to avoid manual maintenance
file(GLOB_RECURSE SOURCES_CPP 
    "source/src/Core/*.cpp"
    "source/src/Render/*.cpp"
    "source/src/Scene/*.cpp"
    "source/src/Physics/*.cpp"
    "source/src/Math/*.cpp"
    "source/src/UI/*.cpp"
    "source/src/Utils/*.cpp"
    "source/src/Animation/*.cpp"
)

file(GLOB SOURCES_IMGUI "source/external/imgui/*.cpp")
file(GLOB_RECURSE HEADERS "source/include/*.h" "source/external/imgui/*.h") # Recursive header search

# miniz compression library
set(MINIZ_SOURCES
    "source/external/miniz.c"
)

# Gather all sources
set(ALL_SOURCES ${SOURCES_CPP} ${SOURCES_IMGUI} ${MINIZ_SOURCES} ${HEADERS})

# --- CUDA / OPTIX ---
find_package(CUDAToolkit REQUIRED)

# Updated Path for Device Code
set(CUDA_SOURCE "source/src/Device/raygen.cu")
set(PTX_OUTPUT "${CMAKE_BINARY_DIR}/raygen.ptx")

# Updated Paths for PTX dependencies
set(PTX_HEADER_DEPS
    "source/include/material_gpu.h"
    "source/include/params.h"
    "source/include/payload.h"
    "source/include/sbt_data.h"
    "source/src/Device/material_scatter.cuh"
    "source/src/Device/vec3_utils.cuh"
    "source/src/Device/random_utils.cuh"
    "source/src/Device/ray_color.cuh"
    "source/src/Device/trace_ray.cuh"
)

# Additional PTX Files (as seen in compile_ptx.bat)
set(CUDA_KERNEL_gas "source/src/Device/gas_kernels.cu")
set(PTX_OUTPUT_gas "${CMAKE_BINARY_DIR}/gas_kernels.ptx")

set(CUDA_KERNEL_erosion "source/src/Device/erosion_kernels.cu")
set(PTX_OUTPUT_erosion "${CMAKE_BINARY_DIR}/erosion_kernels.ptx")

set(CUDA_KERNEL_foliage "source/src/Device/foliage_deform.cu")
set(PTX_OUTPUT_foliage "${CMAKE_BINARY_DIR}/foliage_deform.ptx")

# FFT Pressure Solver for Gas Simulation (10-50x faster)
set(CUDA_KERNEL_fft "source/src/Device/gas_fft_solver.cu")
set(PTX_OUTPUT_fft "${CMAKE_BINARY_DIR}/gas_fft_solver.ptx")


# Custom command to generate PTX (RayGen)
add_custom_command(
    OUTPUT ${PTX_OUTPUT}
    COMMAND ${CMAKE_CUDA_COMPILER}
    -ptx "${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}"
    -o "${PTX_OUTPUT}"
    -I"${OPTIX_ROOT}/include"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/source/include"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/source/src"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/source/src/Device"
    --use_fast_math
    --maxrregcount=64
    -gencode=arch=compute_50,code=compute_50
    -allow-unsupported-compiler
    DEPENDS ${CUDA_SOURCE} ${PTX_HEADER_DEPS}
    COMMENT "Compiling CUDA to PTX: ${CUDA_SOURCE}"
)

# --- EXECUTABLE ---
# Add only raygen.ptx to executable (others are likely loaded dynamically or need separate rules)
add_executable(${PROJECT_NAME} ${ALL_SOURCES} ${PTX_OUTPUT})

# --- INCLUDE DIRECTORIES ---
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/source/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src"
    # Recursively add all subdirectories in source/src as includes
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Core"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Render"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Scene"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Physics"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Math"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/UI"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Utils"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Animation"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Device"
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
    CUDA::cufft           # cuFFT for FFT-based pressure solver
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

message(STATUS "Build setup complete for RayTrophi (Refactored)")
