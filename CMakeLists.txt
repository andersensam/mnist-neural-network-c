cmake_minimum_required(VERSION 3.5.0)
    
# here we specify that the project is C language only, so the default
# C compiler on the system will be used
project(mnist-neural-network C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_DEBUG "-O0 -Wall -Wextra -Wpedantic -g")

include_directories("../src/include")

add_library(utils ../src/utils.c)
add_library(MNIST_Labels ../src/MNIST_Labels.c)
add_library(MNIST_Images ../src/MNIST_Images.c)
add_library(Neural_Network ../src/Neural_Network.c)
add_library(Neural_Network_Threading ../src/Neural_Network_Threading.c)

link_libraries(MNIST_Labels utils)
link_libraries(MNIST_Images utils)
link_libraries(Neural_Network utils)
link_libraries(Neural_Network_Threading utils)

link_libraries(Neural_Network MNIST_Labels)
link_libraries(Neural_Network MNIST_Images)
link_libraries(Neural_Network Neural_Network_Threading)

add_executable(mnist-neural-network
    ../src/inference.c
    ../src/training.c
    ../src/main.c)