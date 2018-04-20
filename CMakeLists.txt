cmake_minimum_required (VERSION 3.3)
project (perceptron)
set (CMAKE_CXX_STANDARD 17)

add_executable (perceptron src/main.cpp
                           src/neuron.cpp
                           src/func.cpp
                           src/net.cpp
                           src/application.cpp
                           src/serializer.cpp
                           src/fileLogger.cpp
                           3rdParty/tinyxml2/tinyxml2.cpp
)

find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()
target_link_libraries (perceptron Boost::program_options)

find_package(Threads)
target_link_libraries (perceptron ${CMAKE_THREAD_LIBS_INIT})

target_include_directories (perceptron PUBLIC inc)
target_include_directories (perceptron PUBLIC 3rdParty/fast-cpp-csv-parser)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_BUILD_TYPE DEBUG)

target_include_directories (perceptron PUBLIC inc)
target_include_directories (perceptron PUBLIC 3rdParty/fast-cpp-csv-parser)
target_include_directories (perceptron PUBLIC 3rdParty/tinyxml2)
