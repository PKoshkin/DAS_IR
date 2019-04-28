project(indexer)
cmake_minimum_required(VERSION 3.0)

set(headers)
set(sources)

macro(dir_src dir)
  file(GLOB __headers__ ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.h)
  file(GLOB __sources__ ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.cpp)
  list(APPEND headers ${__headers__})
  list(APPEND sources ${__sources__})
endmacro()

dir_src(".")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

add_executable(${PROJECT_NAME} ${sources} ${headers})
