cmake_minimum_required(VERSION 3.20)
project(VramGovernor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GLFW and GLEW installed via vcpkg
find_package(glfw3 CONFIG REQUIRED)
find_package(glew CONFIG REQUIRED)

add_executable(vram_app
    src/main.cpp
)

# Link OpenGL, GLFW, GLEW
if (WIN32)
  target_link_libraries(vram_app PRIVATE opengl32)
endif()

target_link_libraries(vram_app PRIVATE glfw GLEW::GLEW)

# Nice warnings
if (MSVC)
  target_compile_options(vram_app PRIVATE /W4 /permissive-)
else()
  target_compile_options(vram_app PRIVATE -Wall -Wextra -Wpedantic)
endif()
