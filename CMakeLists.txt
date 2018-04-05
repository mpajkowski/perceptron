cmake_minimum_required (VERSION 3.1)
project (perceptron)
set (CMAKE_CXX_STANDARD 17)
add_executable (perceptron src/main.cpp
                           src/neuron.cpp
                           src/propagation.cpp
                           src/func.cpp
)

find_package(Boost 1.36.0)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

target_include_directories (perceptron PUBLIC inc)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
