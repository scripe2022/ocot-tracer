cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(RayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB CPP_SOURCES
    "${CMAKE_SOURCE_DIR}/*.cpp"
)
file(GLOB CU_SOURCES
    "${CMAKE_SOURCE_DIR}/*.cu"
)

set(COMMON_CXX_FLAGS
    -O3
    -std=gnu++20
    -Wall
    -Wextra
    -Wshadow
    -D_GLIBCXX_ASSERTIONS
    -fmax-errors=2
)

set(COMMON_LIBS
    png
)

add_executable(raytracer
    ${CPP_SOURCES}
    ${CU_SOURCES}
)

set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(raytracer PRIVATE ${COMMON_CXX_FLAGS})
target_compile_definitions(raytracer PRIVATE GPU_RENDERING=1)
target_link_libraries(raytracer PRIVATE ${COMMON_LIBS})

add_executable(raytracer-cpu
    ${CPP_SOURCES}
)
target_compile_options(raytracer-cpu PRIVATE ${COMMON_CXX_FLAGS})
# CPU 版本不需要 GPU_RENDERING 宏
target_link_libraries(raytracer-cpu PRIVATE ${COMMON_LIBS})

add_executable(raytracer-gpu
    ${CPP_SOURCES}
    ${CU_SOURCES}
)
set_target_properties(raytracer-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(raytracer-gpu PRIVATE ${COMMON_CXX_FLAGS})
target_compile_definitions(raytracer-gpu PRIVATE GPU_RENDERING=1)
target_link_libraries(raytracer-gpu PRIVATE ${COMMON_LIBS})

add_executable(raytracer-gdb
    ${CPP_SOURCES}
    ${CU_SOURCES}
)
set_target_properties(raytracer-gdb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(raytracer-gdb PRIVATE
    -O0
    -g
    -std=gnu++20
    -Wall
    -Wextra
    -Wshadow
    -D_GLIBCXX_ASSERTIONS
    -fmax-errors=2
)
target_compile_definitions(raytracer-gdb PRIVATE GPU_RENDERING=1)
target_link_libraries(raytracer-gdb PRIVATE ${COMMON_LIBS})

