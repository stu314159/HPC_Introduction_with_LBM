cmake_minimum_required(VERSION 3.0)
project(ArrayFire-Linear-1D-Advection VERSION 0.1 LANGUAGES CXX)

find_package(ArrayFire)

add_executable(lax lax.cpp)
target_link_libraries(lax ArrayFire::afcuda)

add_executable(maccormack maccormack.cpp)
target_link_libraries(maccormack ArrayFire::afcuda)

add_executable(rusanov rusanov.cpp)
target_link_libraries(rusanov ArrayFire::afcuda)

