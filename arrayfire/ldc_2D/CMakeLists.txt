cmake_minimum_required(VERSION 3.5)
enable_language(CUDA)
project(ArrayFire-Lid-Driven-Cavity-2D VERSION 0.1 LANGUAGES CUDA CXX)

find_package(ArrayFire)
find_package(CUDA)

add_executable(ldc2d src/ldc2d_af.cu)

add_library(lbm_lib STATIC src/lbm_lib.cu include/lbm_lib.h)

target_include_directories(ldc2d PUBLIC include PRIVATE ${ArrayFire_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_include_directories(lbm_lib PUBLIC include)
#set_property(TARGET ldc2d PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(ldc2d ${CUDA_LIBRARIES} ArrayFire::afcuda lbm_lib )
