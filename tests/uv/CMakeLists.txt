cmake_minimum_required(VERSION 3.18)
project(RayTrophiUvRemapTests LANGUAGES CXX)

get_filename_component(UV_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(UV_SRC "${UV_ROOT}/RayTrophiStudio/source")
add_executable(uv_corner_remap_tests
    UvCornerRemapTests.cpp
    "${UV_SRC}/src/Math/Vec2.cpp"
    "${UV_SRC}/src/Math/Vec3.cpp"
    "${UV_SRC}/src/UV/UvLayout.cpp"
    "${UV_SRC}/src/UV/UvCornerRemap.cpp"
    "${UV_SRC}/src/UV/UvXatlasBackend.cpp"
    "${UV_ROOT}/RayTrophiStudio/external/xatlas/xatlas.cpp"
    "${UV_SRC}/src/MeshEdit/GeometryRemap.cpp")
target_include_directories(uv_corner_remap_tests PRIVATE "${UV_SRC}/include"
    "${UV_ROOT}/RayTrophiStudio/external/xatlas")
target_compile_definitions(uv_corner_remap_tests PRIVATE RAYTROPHI_HAS_XATLAS=1)
target_compile_features(uv_corner_remap_tests PRIVATE cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(uv_corner_remap_tests PRIVATE Threads::Threads)
enable_testing()
add_test(NAME uv_corner_remap COMMAND uv_corner_remap_tests)
