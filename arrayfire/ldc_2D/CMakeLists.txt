cmake_minimum_required(VERSION 3.0)
enable_language(CUDA)
project(ArrayFire-Lid-Driven-Cavity-2D VERSION 0.1 LANGUAGES CXX CUDA)

find_package(ArrayFire)
find_package(CUDA)

add_executable(ldc2d ldc2d.cu)
target_link_libraries(ldc2d ArrayFire::afcuda )


