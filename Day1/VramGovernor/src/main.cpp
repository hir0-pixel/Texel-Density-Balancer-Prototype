#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

static void APIENTRY glDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
    std::cerr << "[GL] " << message << "\n";
}

int main() {
    // 1) Init GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    // Request OpenGL 3.3 Core (fine for our needs)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Day 1 – VramGovernor", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // 2) Load GL functions with GLEW (must be after a context exists)
    glewExperimental = GL_TRUE; // needed for core profiles
    GLenum glewErr = glewInit();
    // GLEW can produce a benign GL_INVALID_ENUM on init; clear it
    glGetError();

    if (glewErr != GLEW_OK) {
        std::cerr << "GLEW init error: " << glewGetErrorString(glewErr) << "\n";
        glfwTerminate();
        return -1;
    }

    // (Optional) Debug output if supported
#ifdef GL_DEBUG_OUTPUT
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(glDebugCallback, nullptr);
#endif

    // 3) Print GPU info
    std::cout << "Vendor  : " << glGetString(GL_VENDOR) << "\n";
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
    std::cout << "Version : " << glGetString(GL_VERSION) << "\n";

    // 4) Main loop – just clear the screen for Day 1
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Esc to quit
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, 1);
        }

        int w, h; glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.07f, 0.10f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
