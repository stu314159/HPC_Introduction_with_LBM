cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(hello LANGUAGES CXX CUDA)
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

add_executable(hello 
	       src/hello.C
	       src/CudaPart.cu
	       include/CudaPart.h
	      )

include_directories(include)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)

target_compile_options(hello PUBLIC -O3)
set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(hello PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(hello 
	             ${MPI_CXX_LIBRARIES}
		     ${CUDA_LIBRARIES}		   
		     )
target_include_directories(hello
                            PUBLIC
		           ${MPI_CXX_INCLUDE_PATH}
		           ${CUDA_INCLUDE_DIRS}
		           )
