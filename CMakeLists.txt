cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 20)
project(NeuroNet LANGUAGES CXX)

set(EXECUTABLE_NAME neuro_net)
set(NET_LIBRARY net_lib)

add_subdirectory(src)