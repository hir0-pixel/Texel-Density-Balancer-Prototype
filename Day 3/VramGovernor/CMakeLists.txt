cmake_minimum_required(VERSION 3.20)
project(VramGovernor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# vcpkg packages
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW  CONFIG REQUIRED)

add_executable(vram_app
    src/main.cpp
)

# Link OpenGL + deps
if (WIN32)
  target_link_libraries(vram_app PRIVATE opengl32)
endif()
target_link_libraries(vram_app PRIVATE glfw GLEW::GLEW)

# Include stb_image.h
target_include_directories(vram_app PRIVATE third_party)

# Warnings
if (MSVC)
  target_compile_options(vram_app PRIVATE /W4 /permissive-)
else()
  target_compile_options(vram_app PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Copy asset next to EXE after build (so relative path "assets/checker.png" works)
add_custom_command(TARGET vram_app POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vram_app>/assets"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/assets/checker.png"
          "$<TARGET_FILE_DIR:vram_app>/assets/checker.png")
