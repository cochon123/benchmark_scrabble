cmake_minimum_required(VERSION 3.16)
project(scrabble_quackle_certifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED QUACKLE_SOURCE)
  message(FATAL_ERROR "Set QUACKLE_SOURCE to the Quackle source directory")
endif()

find_package(Qt6 REQUIRED COMPONENTS Core)
add_subdirectory("${QUACKLE_SOURCE}" quackle-core)

add_library(quackle_dictionary_tools
  "${QUACKLE_SOURCE}/quackleio/dawgfactory.cpp"
  "${QUACKLE_SOURCE}/quackleio/dict.cpp"
  "${QUACKLE_SOURCE}/quackleio/dictfactory.cpp"
  "${QUACKLE_SOURCE}/quackleio/dictimplementation.cpp"
  "${QUACKLE_SOURCE}/quackleio/flexiblealphabet.cpp"
  "${QUACKLE_SOURCE}/quackleio/gaddagfactory.cpp"
  "${QUACKLE_SOURCE}/quackleio/util.cpp"
)
target_include_directories(quackle_dictionary_tools PUBLIC
  "${QUACKLE_SOURCE}"
  "${QUACKLE_SOURCE}/quackleio"
)
target_link_libraries(quackle_dictionary_tools PUBLIC quackle Qt6::Core)

add_executable(build_dictionary build_dictionary.cpp)
target_link_libraries(build_dictionary PRIVATE quackle_dictionary_tools)

add_executable(certify_moves
  certify_moves.cpp
  "${QUACKLE_SOURCE}/test/trademarkedboards.cpp"
)
target_include_directories(certify_moves PRIVATE
  "${QUACKLE_SOURCE}"
  "${QUACKLE_SOURCE}/test"
)
target_link_libraries(certify_moves PRIVATE quackle)
