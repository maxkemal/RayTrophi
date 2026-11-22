# Android Player build. The source list, include dirs and defines are GENERATED into
# player_sources.cmake by scripts/gen_player_android_cmake.py - do not list sources here.
cmake_minimum_required(VERSION 3.22.1)
project(RayTrophiPlayer LANGUAGES C CXX)

get_filename_component(RT_PROJECT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../RayTrophiStudio" ABSOLUTE)
include(${CMAKE_CURRENT_LIST_DIR}/player_sources.cmake)

set(CMAKE_CXX_STANDARD ${RT_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Third-party dependencies (variables documented in android/README.md) -----------------
foreach(dep IN LISTS RT_DEP_LIST)
    if(NOT DEFINED ${dep} OR "${${dep}}" STREQUAL "")
        message(FATAL_ERROR "${dep} is not set. See android/README.md (dependencies).")
    endif()
endforeach()

# SDL2 as a subproject; SDL2_image needs the SDL2 target to exist first.
add_subdirectory("${RT_SDL2_DIR}" SDL2)
add_subdirectory("${RT_SDL2_IMAGE_DIR}" SDL2_image)

# Prebuilt trees: <prefix>/<abi>/{include,lib}.
set(RT_EXTRA_INCLUDES "")
set(RT_EXTRA_LIBS "")
foreach(dep IN LISTS RT_DEP_LIST)
    if("${RT_DEP_${dep}_KIND}" STREQUAL "prebuilt")
        list(APPEND RT_EXTRA_INCLUDES "${${dep}}/${ANDROID_ABI}/include")
        foreach(lib IN LISTS RT_DEP_${dep}_LIBS)
            find_library(RT_LIB_${dep}_${lib} NAMES ${lib}
                         PATHS "${${dep}}/${ANDROID_ABI}/lib" NO_DEFAULT_PATH)
            if(NOT RT_LIB_${dep}_${lib})
                message(FATAL_ERROR "lib${lib} not found under ${${dep}}/${ANDROID_ABI}/lib")
            endif()
            list(APPEND RT_EXTRA_LIBS "${RT_LIB_${dep}_${lib}}")
        endforeach()
    endif()
endforeach()

# --- The Player: SDLActivity loads "libmain.so" and calls SDL_main -----------------------
add_library(main SHARED ${RT_PLAYER_SOURCES})
target_compile_definitions(main PRIVATE ${RT_PLAYER_DEFINES})
target_include_directories(main PRIVATE ${RT_PLAYER_INCLUDE_DIRS} ${RT_EXTRA_INCLUDES})

string(REPLACE "-" "_" RT_ABI_VAR "${ANDROID_ABI}")
target_compile_options(main PRIVATE ${RT_CXX_FLAGS_COMMON} ${RT_CXX_FLAGS_${RT_ABI_VAR}} -fopenmp)

# The vcxproj builds most files without exception handling; files marked Sync keep it.
# Fallback if clang rejects a try/catch in a no-EH file: -DRT_ANDROID_FORCE_EXCEPTIONS=ON.
option(RT_ANDROID_FORCE_EXCEPTIONS "Compile every file with exceptions enabled" OFF)
if(RT_ANDROID_FORCE_EXCEPTIONS)
    target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fexceptions>)
else()
    target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)
    set_source_files_properties(${RT_PLAYER_EXCEPTION_SOURCES} PROPERTIES COMPILE_OPTIONS "-fexceptions")
endif()

# OpenMP: static runtime so no libomp.so has to be packaged into the APK.
target_link_options(main PRIVATE -fopenmp -static-openmp)

if(TARGET SDL2::SDL2)
    target_link_libraries(main PRIVATE SDL2::SDL2)
else()
    target_link_libraries(main PRIVATE SDL2)
endif()
if(TARGET SDL2_image::SDL2_image)
    target_link_libraries(main PRIVATE SDL2_image::SDL2_image)
else()
    target_link_libraries(main PRIVATE SDL2_image)
endif()
target_link_libraries(main PRIVATE ${RT_EXTRA_LIBS} ${RT_SYSTEM_LIBS})
