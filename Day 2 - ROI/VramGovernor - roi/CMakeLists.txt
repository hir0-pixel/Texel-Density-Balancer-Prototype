cmake_minimum_required(VERSION 3.20)
project(VramGovernorROI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# vcpkg packages
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW  CONFIG REQUIRED)

add_executable(vram_app_roi
    src/main.cpp
)

# Link libraries
if (WIN32)
  target_link_libraries(vram_app_roi PRIVATE opengl32)
endif()
target_link_libraries(vram_app_roi PRIVATE glfw GLEW::GLEW)

# Include stb_image.h
target_include_directories(vram_app_roi PRIVATE third_party)

# Warnings
if (MSVC)
  target_compile_options(vram_app_roi PRIVATE /W4 /permissive-)
else()
  target_compile_options(vram_app_roi PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Copy asset next to EXE after build (so paths just work)
add_custom_command(TARGET vram_app_roi POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vram_app_roi>/assets"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/assets/checker.png"
          "$<TARGET_FILE_DIR:vram_app_roi>/assets/checker.png")
