cmake_minimum_required(VERSION 3.18)
project(RayTrophi VERSION 1.0 LANGUAGES CXX CUDA)

option(RAYTROPHI_ENABLE_OZZ "Enable future Ozz animation integration hooks" ON)

set(RAYTROPHI_XATLAS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/xatlas" CACHE PATH "Path to vendored xatlas source directory")
set(RAYTROPHI_HAS_XATLAS OFF)
if(NOT (EXISTS "${RAYTROPHI_XATLAS_ROOT}/xatlas.h" AND EXISTS "${RAYTROPHI_XATLAS_ROOT}/xatlas.cpp"))
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/xatlas/source/xatlas/xatlas.h" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/xatlas/source/xatlas/xatlas.cpp")
        set(RAYTROPHI_XATLAS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/xatlas/source/xatlas" CACHE PATH "Path to vendored xatlas source directory" FORCE)
    endif()
endif()
if(EXISTS "${RAYTROPHI_XATLAS_ROOT}/xatlas.h" AND EXISTS "${RAYTROPHI_XATLAS_ROOT}/xatlas.cpp")
    set(RAYTROPHI_HAS_XATLAS ON)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 50)  # SM 5.0 for broad GPU support

# Keep CMake artifacts isolated from the checked-in VS2022 x64 output folders.
# Each generator/config gets a self-contained runtime directory under build/bin.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(CONFIG DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG} "${CMAKE_BINARY_DIR}/bin/${CONFIG}")
endforeach()

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
        
        # Match VS build flags required for optimal CPU performance and proper OpenMP enablement
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj /Zc:__cplusplus /openmp /arch:AVX2 /fp:fast")
    endif()
endif()

# --- USER DEFINED PATHS ---
# Read paths from environment variables. These can be overridden via CMake cache variables (-D).
set(SDL2_ROOT "$ENV{SDL2_ROOT}" CACHE PATH "Path to SDL2 root directory")
set(SDL2_IMAGE_ROOT "$ENV{SDL2_IMAGE_ROOT}" CACHE PATH "Path to SDL2 Image root directory")
set(OPTIX_ROOT "$ENV{OPTIX_ROOT}" CACHE PATH "Path to OptiX SDK")
set(RAYTROPHI_EXTERNAL_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/../../external_dependencies" CACHE PATH "Path to RayTrophi external_dependencies directory")
if(NOT EXISTS "${RAYTROPHI_EXTERNAL_DEPS}")
    set(RAYTROPHI_EXTERNAL_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/../external_dependencies" CACHE PATH "Path to RayTrophi external_dependencies directory" FORCE)
endif()

function(raytrophi_set_path_if_valid VAR VALUE)
    if(NOT "${VALUE}" STREQUAL "" AND EXISTS "${VALUE}")
        set(${VAR} "${VALUE}" CACHE PATH "" FORCE)
    endif()
endfunction()

if(NOT EXISTS "${SDL2_ROOT}/include/SDL.h")
    raytrophi_set_path_if_valid(SDL2_ROOT "${RAYTROPHI_EXTERNAL_DEPS}/SDL2-2.30.4")
endif()
if(NOT EXISTS "${SDL2_IMAGE_ROOT}/include/SDL_image.h")
    raytrophi_set_path_if_valid(SDL2_IMAGE_ROOT "${RAYTROPHI_EXTERNAL_DEPS}/SDL2-2.30.4/SDL2_image-2.8.2")
endif()
if(NOT OPTIX_ROOT AND EXISTS "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")
    # Fallback to standard NVIDIA OptiX installation path if environment variable is not set
    set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0" CACHE PATH "Path to OptiX SDK" FORCE)
endif()
if(NOT EXISTS "${OPTIX_ROOT}/include/optix.h" AND EXISTS "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
    set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0" CACHE PATH "Path to OptiX SDK" FORCE)
endif()
set(EMBREE_ROOT "$ENV{EMBREE_ROOT}" CACHE PATH "Path to Embree root")
set(OIDN_ROOT "$ENV{OIDN_ROOT}" CACHE PATH "Path to OIDN root")
set(ASSIMP_ROOT "$ENV{ASSIMP_ROOT}" CACHE PATH "Path to Assimp root")
set(VULKAN_SDK "$ENV{VULKAN_SDK}" CACHE PATH "Path to Vulkan SDK")
if(NOT EXISTS "${EMBREE_ROOT}/include/embree4/rtcore.h")
    raytrophi_set_path_if_valid(EMBREE_ROOT "${RAYTROPHI_EXTERNAL_DEPS}/embree-4.4.0.x64.windows")
endif()
if(NOT EXISTS "${OIDN_ROOT}/include/OpenImageDenoise/oidn.hpp")
    raytrophi_set_path_if_valid(OIDN_ROOT "${RAYTROPHI_EXTERNAL_DEPS}/oidn-2.3.0.x64.windows")
endif()
if(NOT EXISTS "${ASSIMP_ROOT}/include/assimp/Importer.hpp")
    raytrophi_set_path_if_valid(ASSIMP_ROOT "${RAYTROPHI_EXTERNAL_DEPS}/Assimp")
endif()

# --- SOURCE FILES (Refactored Structure) ---
# Automatically collect source files from new subdirectories to avoid manual maintenance
file(GLOB SOURCES_CPP_ROOT "source/src/*.cpp")
file(GLOB_RECURSE SOURCES_CPP 
    "source/src/Core/*.cpp"
    "source/src/Render/*.cpp"
    "source/src/Scene/*.cpp"
    "source/src/Physics/*.cpp"
    "source/src/Math/*.cpp"
    "source/src/UI/*.cpp"
    "source/src/Utils/*.cpp"
    "source/src/Animation/*.cpp"
    "source/src/Hair/*.cpp"
    "source/src/Backend/*.cpp"
    "source/src/Paint/*.cpp"
    "source/src/Stylize/*.cpp"
    "source/src/Viewport/*.cpp"
)
list(APPEND SOURCES_CPP ${SOURCES_CPP_ROOT})
list(APPEND SOURCES_CPP "source/src/Render/FoliageWindSystem.cpp")
list(APPEND SOURCES_CPP "source/src/Scene/AssetRegistry.cpp")

file(GLOB SOURCES_IMGUI "external/imgui/*.cpp")
file(GLOB_RECURSE HEADERS "source/include/*.h" "external/imgui/*.h") # Recursive header search
list(APPEND HEADERS "source/include/FoliageWindSystem.h")
list(APPEND HEADERS "source/include/AssetRegistry.h")

# miniz compression library
set(MINIZ_SOURCES
    "external/miniz.c"
)

# Jolt Physics (v5.5.0, MIT) - compiled in-project for matching flags/runtime/SIMD
file(GLOB_RECURSE SOURCES_JOLT "external/JoltPhysics/Jolt/*.cpp")

# Host-callable CUDA files (compiled to .obj)
set(CUDA_SOURCES_OBJ
    "source/src/Device/fft_ocean.cu"
    "source/src/Device/foliage_deform.cu"
    "source/src/Device/skinning_kernels.cu"
    "source/src/Device/gas_kernels.cu"
    "source/src/Device/gas_fft_solver.cu"
    "source/src/Device/oidn_blend.cu"
    "source/src/Stylize/StylizeKernel.cu"
)

set(OZZ_RUNTIME_SOURCES
    "external/ozz-animation/src/base/log.cc"
    "external/ozz-animation/src/base/platform.cc"
    "external/ozz-animation/src/base/containers/string_archive.cc"
    "external/ozz-animation/src/base/encode/group_varint.cc"
    "external/ozz-animation/src/base/io/archive.cc"
    "external/ozz-animation/src/base/io/stream.cc"
    "external/ozz-animation/src/base/maths/box.cc"
    "external/ozz-animation/src/base/maths/math_archive.cc"
    "external/ozz-animation/src/base/maths/simd_math.cc"
    "external/ozz-animation/src/base/maths/simd_math_archive.cc"
    "external/ozz-animation/src/base/maths/soa_math_archive.cc"
    "external/ozz-animation/src/base/memory/allocator.cc"
    "external/ozz-animation/src/animation/runtime/animation.cc"
    "external/ozz-animation/src/animation/runtime/animation_utils.cc"
    "external/ozz-animation/src/animation/runtime/blending_job.cc"
    "external/ozz-animation/src/animation/runtime/ik_aim_job.cc"
    "external/ozz-animation/src/animation/runtime/ik_two_bone_job.cc"
    "external/ozz-animation/src/animation/runtime/local_to_model_job.cc"
    "external/ozz-animation/src/animation/runtime/motion_blending_job.cc"
    "external/ozz-animation/src/animation/runtime/sampling_job.cc"
    "external/ozz-animation/src/animation/runtime/skeleton.cc"
    "external/ozz-animation/src/animation/runtime/skeleton_utils.cc"
    "external/ozz-animation/src/animation/runtime/track.cc"
    "external/ozz-animation/src/animation/runtime/track_sampling_job.cc"
    "external/ozz-animation/src/animation/runtime/track_triggering_job.cc"
    "external/ozz-animation/src/animation/offline/raw_animation.cc"
    "external/ozz-animation/src/animation/offline/raw_animation_archive.cc"
    "external/ozz-animation/src/animation/offline/raw_animation_utils.cc"
    "external/ozz-animation/src/animation/offline/additive_animation_builder.cc"
    "external/ozz-animation/src/animation/offline/animation_builder.cc"
    "external/ozz-animation/src/animation/offline/raw_skeleton.cc"
    "external/ozz-animation/src/animation/offline/raw_skeleton_archive.cc"
    "external/ozz-animation/src/animation/offline/skeleton_builder.cc"
)

# Gather all sources
set(ALL_SOURCES ${SOURCES_CPP} ${SOURCES_IMGUI} ${MINIZ_SOURCES} ${SOURCES_JOLT} ${CUDA_SOURCES_OBJ} ${HEADERS})
if(RAYTROPHI_ENABLE_OZZ)
    list(APPEND ALL_SOURCES ${OZZ_RUNTIME_SOURCES})
endif()
if(RAYTROPHI_HAS_XATLAS)
    list(APPEND ALL_SOURCES "${RAYTROPHI_XATLAS_ROOT}/xatlas.cpp")
endif()

# --- CUDA / OPTIX ---
find_package(CUDAToolkit REQUIRED)

# --- VULKAN ---
find_package(Vulkan REQUIRED)

# Updated Path for Device Code
set(CUDA_SOURCE "source/src/Device/raygen.cu")
set(PTX_OUTPUT "${CMAKE_BINARY_DIR}/raygen.ptx")
set(PTX_OUTPUT_miss "${CMAKE_BINARY_DIR}/miss_kernels.ptx")
set(PTX_OUTPUT_hitgroup "${CMAKE_BINARY_DIR}/hitgroup_kernels.ptx")

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

function(raytrophi_add_ptx OUTPUT SOURCE)
    add_custom_command(
        OUTPUT "${OUTPUT}"
        COMMAND ${CMAKE_CUDA_COMPILER}
        -ptx "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}"
        -o "${OUTPUT}"
        -I"${OPTIX_ROOT}/include"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/source/include"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/libs"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/source/src"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/source/src/Device"
        --use_fast_math
        --maxrregcount=64
        -gencode=arch=compute_50,code=compute_50
        -allow-unsupported-compiler
        DEPENDS "${SOURCE}" ${PTX_HEADER_DEPS}
        COMMENT "Compiling CUDA to PTX: ${SOURCE}"
        VERBATIM
    )
endfunction()

raytrophi_add_ptx("${PTX_OUTPUT}" "${CUDA_SOURCE}")
raytrophi_add_ptx("${PTX_OUTPUT_miss}" "source/src/Device/miss_kernels.cu")
raytrophi_add_ptx("${PTX_OUTPUT_hitgroup}" "source/src/Device/hitgroup_kernels.cu")
raytrophi_add_ptx("${PTX_OUTPUT_erosion}" "${CUDA_KERNEL_erosion}")
raytrophi_add_ptx("${PTX_OUTPUT_foliage}" "${CUDA_KERNEL_foliage}")

# --- EXECUTABLE ---
# Runtime-loaded PTX files are part of the target so CMake generates/copies them with the exe.
set(PTX_OUTPUTS
    "${PTX_OUTPUT}"
    "${PTX_OUTPUT_miss}"
    "${PTX_OUTPUT_hitgroup}"
    "${PTX_OUTPUT_erosion}"
    "${PTX_OUTPUT_foliage}"
)
add_executable(${PROJECT_NAME} ${ALL_SOURCES} ${PTX_OUTPUTS})
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "RayTrophi Studio")

if(RAYTROPHI_ENABLE_OZZ)
    target_compile_definitions(${PROJECT_NAME} PRIVATE RAYTROPHI_ENABLE_OZZ=1)
endif()
if(RAYTROPHI_HAS_XATLAS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE RAYTROPHI_HAS_XATLAS=1)
endif()

# --- INCLUDE DIRECTORIES ---
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/source/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/include/Hair"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/include/Backend"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Hair"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Viewport"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Device"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/src/Stylize"
    "${CMAKE_CURRENT_SOURCE_DIR}/external"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/JoltPhysics"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/ozz-animation/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/ozz-animation/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/libs"
    "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/installed/x64-windows/include"
    "${OPTIX_ROOT}/include"
    "${SDL2_ROOT}/include"
    "${SDL2_IMAGE_ROOT}/include"
    "${EMBREE_ROOT}/include"
    "${OIDN_ROOT}/include"
    "${ASSIMP_ROOT}/include"
    "${VULKAN_SDK}/Include"
)

if(RAYTROPHI_HAS_XATLAS)
    target_include_directories(${PROJECT_NAME} PRIVATE "${RAYTROPHI_XATLAS_ROOT}")
    message(STATUS "xatlas backend enabled from: ${RAYTROPHI_XATLAS_ROOT}")
else()
    message(STATUS "xatlas backend disabled: vendored source not found at ${RAYTROPHI_XATLAS_ROOT}")
endif()

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

# VCPKG Libraries
file(GLOB VCPKG_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/installed/x64-windows/lib/*.lib")
set(ADDITIONAL_LIBS
    openvdb.lib Imath-3_1.lib Iex-3_3.lib IlmThread-3_3.lib OpenEXR-3_3.lib OpenEXRCore-3_3.lib blosc.lib zlib.lib lz4.lib zstd.lib DirectXTex.lib
)

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
    Vulkan::Vulkan
    kernel32 user32 gdi32 winspool comdlg32 advapi32 shell32 ole32 oleaut32 uuid
)

target_link_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/installed/x64-windows/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE ${ADDITIONAL_LIBS})

# Use delay loading for Driver API so the application can gracefully fall back to Vulkan/CPU when NVIDIA driver is missing.
if(MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE "/DELAYLOAD:nvcuda.dll" "/DELAYLOAD:vulkan-1.dll" "/DELAYLOAD:cudart64_12.dll" "/DELAYLOAD:cudart64_11.dll" "/DELAYLOAD:cufft64_11.dll")
    # Also need delayimp.lib to hook the delay loading
    target_link_libraries(${PROJECT_NAME} PRIVATE delayimp)
endif()

# --- POST BUILD COPY ---
# Copy runtime-loaded PTX files next to the executable.
foreach(PTX_FILE IN LISTS PTX_OUTPUTS)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PTX_FILE}"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying ${PTX_FILE} to output directory"
    )
endforeach()

# Copy Vulkan shaders to executable directory/shaders
# Compiled .spv files live under source/shaders/ (produced by compile_shaders.bat)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/source/shaders"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMENT "Copying Vulkan shaders to output directory/shaders"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/assets"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets"
    COMMENT "Copying runtime assets to output directory/assets"
)

function(raytrophi_copy_runtime_dll DLL_PATH)
    if(EXISTS "${DLL_PATH}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL_PATH}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            COMMENT "Copying runtime DLL: ${DLL_PATH}"
        )
    endif()
endfunction()

raytrophi_copy_runtime_dll("${SDL2_ROOT}/lib/x64/SDL2.dll")
raytrophi_copy_runtime_dll("${SDL2_IMAGE_ROOT}/lib/x64/SDL2_image.dll")
raytrophi_copy_runtime_dll("${EMBREE_ROOT}/bin/embree4.dll")
raytrophi_copy_runtime_dll("${CMAKE_CURRENT_SOURCE_DIR}/../x64/Release/embree4.dll")
raytrophi_copy_runtime_dll("${EMBREE_ROOT}/bin/tbb12.dll")
raytrophi_copy_runtime_dll("${OIDN_ROOT}/bin/OpenImageDenoise.dll")
raytrophi_copy_runtime_dll("${OIDN_ROOT}/bin/OpenImageDenoise_core.dll")
raytrophi_copy_runtime_dll("${OIDN_ROOT}/bin/OpenImageDenoise_device_cpu.dll")
raytrophi_copy_runtime_dll("${OIDN_ROOT}/bin/OpenImageDenoise_device_cuda.dll")
raytrophi_copy_runtime_dll("${OIDN_ROOT}/bin/OpenImageDenoise_device_hip.dll")
raytrophi_copy_runtime_dll("${ASSIMP_ROOT}/bin/x64/assimp-vc143-mt.dll")

file(GLOB RAYTROPHI_VCPKG_RUNTIME_DLLS "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/installed/x64-windows/bin/*.dll")
foreach(RUNTIME_DLL IN LISTS RAYTROPHI_VCPKG_RUNTIME_DLLS)
    raytrophi_copy_runtime_dll("${RUNTIME_DLL}")
endforeach()

message(STATUS "Build setup complete for RayTrophi (Refactored)")
