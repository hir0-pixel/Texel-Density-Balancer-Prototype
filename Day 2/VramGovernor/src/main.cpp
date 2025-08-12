#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static void APIENTRY glDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
    std::cerr << "[GL] " << message << "\n";
}

// Simple shaders
const char* vsSrc = R"(
#version 330 core
layout(location = 0) in vec2 aPos; // position (x, y)
layout(location = 1) in vec2 aUV;  // texture coordinates (u, v)

out vec2 vUV; // sent to fragment shader

void main() {
    vUV = aUV;                        // pass UV to fragment shader
    gl_Position = vec4(aPos, 0.0, 1.0); // set position in clip space
}
)";

const char* fsSrc = R"(
#version 330 core
in vec2 vUV;                  // from vertex shader
out vec4 fragColor;           // final pixel color
uniform sampler2D uTex;       // bound texture

void main() {
    fragColor = texture(uTex, vUV); // lookup texture at UV coordinate
}
)";

GLuint compileShader(GLenum type, const char* src){
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok=0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if(!ok){
        char buf[1024]; glGetShaderInfoLog(sh, 1024, nullptr, buf);
        std::cerr << "Shader error: " << buf << "\n";
    }
    return sh;
}

GLuint createProgram(){
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

int main() {
    // Init GLFW
    if (!glfwInit()) { std::cerr << "Failed to init GLFW\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Day 2 – Textured Quad", nullptr, nullptr);
    if (!window) { std::cerr << "Failed to create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Init GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed\n"; return -1; }
    glGetError();

    // Debug output
#ifdef GL_DEBUG_OUTPUT
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(glDebugCallback, nullptr);
#endif

    std::cout << "Vendor  : " << glGetString(GL_VENDOR) << "\n";
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
    std::cout << "Version : " << glGetString(GL_VERSION) << "\n";

    // Quad geometry
    float verts[] = {
        // pos     // uv
        -1, -1,   0, 0,
         1, -1,   1, 0,
         1,  1,   1, 1,
        -1,  1,   0, 1
    };
    unsigned int idx[] = { 0,1,2,  2,3,0 };

    GLuint vao,vbo,ebo;
    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
    glGenBuffers(1,&ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(idx),idx,GL_STATIC_DRAW);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    // Load image
    int tw,th,channels;
    unsigned char* data = stbi_load("assets/checker.png", &tw, &th, &channels, STBI_rgb_alpha);
    if(!data){ std::cerr << "Failed to load texture\n"; return -1; }

    // Create texture + mipmaps
    GLuint tex;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,tw,th,0,GL_RGBA,GL_UNSIGNED_BYTE,data);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(data);

    // Create sampler
    GLuint samp;
    glGenSamplers(1,&samp);
    glSamplerParameteri(samp, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glSamplerParameteri(samp, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameterf(samp, GL_TEXTURE_MAX_ANISOTROPY, 4.0f);

    // LOD bias var
    float lodBias = 0.0f;

    // Shader program
    GLuint prog = createProgram();
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog,"uTex"),0);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // LOD bias control
        if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) lodBias += 0.01f;
        if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) lodBias -= 0.01f;

        glBindSampler(0, samp);
        glSamplerParameterf(samp, GL_TEXTURE_LOD_BIAS, lodBias);

        glClearColor(0.07f, 0.10f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUseProgram(prog);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
    }

    glDeleteProgram(prog);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
