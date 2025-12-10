cmake_minimum_required(VERSION 3.18)
project(RayTrophi VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# --- USER DEFINED PATHS (Overridable via Cache) ---
set(SDL2_ROOT "E:/SDL2-2.30.4" CACHE PATH "Path to SDL2 root directory")
set(SDL2_IMAGE_ROOT "E:/SDL2-2.30.4/SDL2_image-2.8.2" CACHE PATH "Path to SDL2 Image root directory")
set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0" CACHE PATH "Path to OptiX SDK")

# Paths from dis_bagimliliklar
set(EMBREE_ROOT "E:/RayTrophi_projesi/dis_bagimliliklar/embree-4.4.0.x64.windows" CACHE PATH "Path to Embree root")
set(OIDN_ROOT "E:/RayTrophi_projesi/dis_bagimliliklar/oidn-2.3.0.x64.windows" CACHE PATH "Path to OIDN root")
set(ASSIMP_ROOT "E:/RayTrophi_projesi/dis_bagimliliklar/Assimp" CACHE PATH "Path to Assimp root")

# --- SOURCE FILES ---
file(GLOB SOURCES_CPP "source/cpp_file/*.cpp")
file(GLOB SOURCES_IMGUI "source/imgui/*.cpp")
file(GLOB HEADERS "source/header/*.h" "source/imgui/*.h")

# Gather all sources
set(ALL_SOURCES ${SOURCES_CPP} ${SOURCES_IMGUI} ${HEADERS})

# --- CUDA / OPTIX ---
find_package(CUDAToolkit REQUIRED)

set(CUDA_SOURCE "source/cpp_file/raygen.cu")
set(PTX_OUTPUT "${CMAKE_BINARY_DIR}/raygen.ptx")

# Custom command to generate PTX
add_custom_command(
    OUTPUT ${PTX_OUTPUT}
    COMMAND ${CMAKE_CUDA_COMPILER}
    -ptx "${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}"
    -o "${PTX_OUTPUT}"
    -I"${OPTIX_ROOT}/include"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/source/header"
    --use_fast_math
    --maxrregcount=64
    -gencode=arch=compute_86,code=compute_86
    -allow-unsupported-compiler
    DEPENDS ${CUDA_SOURCE}
    COMMENT "Compiling CUDA to PTX: ${CUDA_SOURCE}"
)

# --- EXECUTABLE ---
add_executable(${PROJECT_NAME} ${ALL_SOURCES} ${PTX_OUTPUT})

# --- INCLUDE DIRECTORIES ---
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/source/header"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/imgui"
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
