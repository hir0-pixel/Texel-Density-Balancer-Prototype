cmake_minimum_required(VERSION 3.16)
project(VramGovernorDay5R LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
# Prefer vcpkg CONFIG packages; fallback to FetchContent if missing.
find_package(glfw3 CONFIG QUIET)
find_package(GLEW  CONFIG QUIET)
include(FetchContent)

if (NOT glfw3_FOUND)
  FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.3.9)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(glfw)
endif()

if (NOT GLEW_FOUND)
  FetchContent_Declare(glewsrc GIT_REPOSITORY https://github.com/nigels-com/glew.git GIT_TAG glew-2.2.0)
  FetchContent_MakeAvailable(glewsrc)
  add_subdirectory(${glewsrc_SOURCE_DIR}/build/cmake ${glewsrc_BINARY_DIR}/cmake_build)
endif()

add_executable(VramGovernorDay5R src/main.cpp)

if (glfw3_FOUND)
  target_link_libraries(VramGovernorDay5R PRIVATE glfw)
else()
  target_link_libraries(VramGovernorDay5R PRIVATE glfw)
endif()

if (GLEW_FOUND)
  target_link_libraries(VramGovernorDay5R PRIVATE GLEW::GLEW)
else()
  target_link_libraries(VramGovernorDay5R PRIVATE libglew_static)
  target_compile_definitions(VramGovernorDay5R PRIVATE GLEW_STATIC)
endif()

target_link_libraries(VramGovernorDay5R PRIVATE OpenGL::GL)

if (MSVC)
  target_compile_definitions(VramGovernorDay5R PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
