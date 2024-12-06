# based on example: https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/

cmake_minimum_required(VERSION 3.29)
project(kaleidoscope)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# default location when installing with homebrew, we have to redefine LLVM_DIR or LLVM_CONFIG.cmake can not be found
set(LLVM_DIR /opt/homebrew/opt/llvm/lib/cmake/llvm)

find_package(LLVM REQUIRED CONFIG)

add_executable(kaleidoscope main.cpp
        include/KaleidoscopeJIT.cpp
        include/KaleidoscopeJIT.h)

include_directories(${LLVM_INCLUDE_DIRS})
# -g -O3 --cxxflags
target_compile_options(kaleidoscope PRIVATE -g -O3 ${LLVM_CXXFLAGS})

#link LLVM libraries
#llvm_map_components_to_libnames(llvm_libs support core irreader)
llvm_map_components_to_libnames(llvm_libs core orcjit native)
# --ldflags --system-libs --libs core
target_link_libraries(kaleidoscope PRIVATE ${llvm_libs} ${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBS} ${LLVM_LIBS})
