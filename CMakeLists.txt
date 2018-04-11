cmake_minimum_required (VERSION 3.1)
project (perceptron)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
add_executable (perceptron src/main.cpp
                           src/neuron.cpp
                           src/func.cpp
                           src/net.cpp
)

find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

target_link_libraries (perceptron Boost::program_options)

target_include_directories (perceptron PUBLIC inc)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_BUILD_TYPE DEBUG)
