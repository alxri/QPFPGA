cmake_minimum_required(VERSION 3.16)
project(qpfpga_solver LANGUAGES CXX)

add_library(qpfpga SHARED
    src/api.cpp
    src/utils.cpp
    src/fpga_utils.cpp
)

target_include_directories(qpfpga
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pmt
        /usr/local/include
)

# Enforce C++17
target_compile_features(qpfpga PUBLIC cxx_std_17)

# Add Optimization and Warning flags
target_compile_options(qpfpga PRIVATE -O3 -Wall)

# Specify the library path for the linker
target_link_directories(qpfpga PRIVATE /usr/local/lib)

# Link the required libraries (cma, m, rt) AND pmt
target_link_libraries(qpfpga PRIVATE cma m rt pmt)

# Tell the linker where to find libcma.so and libpmt.so at runtime (rpath)
set_target_properties(qpfpga PROPERTIES
    INSTALL_RPATH "/usr/local/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)