// Day 2.5 – ROI Blend (center sharp, periphery lower texel density)
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <algorithm> // std::clamp
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static void APIENTRY glDebugCallback(GLenum, GLenum, GLuint, GLenum, GLsizei, const GLchar* message, const void*) {
    std::cerr << "[GL] " << message << "\n";
}

// ---------- Shaders ----------
static const char* kVS = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

// ROI blend in screen space using gl_FragCoord (pixels)
static const char* kFS = R"(
#version 330 core
in vec2 vUV;
out vec4 fragColor;

// Same texture bound twice with different samplers:
uniform sampler2D texHi; // unit 0: sharp sampler (bias = 0)
uniform sampler2D texLo; // unit 1: biased sampler (bias > 0)

// ROI parameters in framebuffer pixels
uniform vec2  roiCenter;   // screen center or mouse pos
uniform float roiRadius;   // fully sharp radius
uniform float roiFeather;  // soft transition thickness

void main() {
    float d = distance(gl_FragCoord.xy, roiCenter);
    // 0.0 inside radius (sharp), 1.0 outside radius+feather (biased)
    float w = smoothstep(roiRadius, roiRadius + roiFeather, d);

    vec4 sharp = texture(texHi, vUV);
    vec4 soft  = texture(texLo, vUV);
    fragColor = mix(sharp, soft, w);
}
)";

// ---------- Shader helpers ----------
static GLuint compileShader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = GL_FALSE;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetShaderInfoLog(sh, sizeof(log), &n, log);
        std::cerr << "Shader compile error:\n" << log << "\n";
    }
    return sh;
}

static GLuint createProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER,   vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = GL_FALSE;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048]; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof(log), &n, log);
        std::cerr << "Program link error:\n" << log << "\n";
    }
    return p;
}

int main() {
    // --- Window/context ---
    if (!glfwInit()) { std::cerr << "Failed to init GLFW\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(1000, 700, "Day 2.5 – ROI Blend", nullptr, nullptr);
    if (!window) { std::cerr << "Failed to create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // --- GLEW ---
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed\n"; return -1; }
    glGetError(); // clear benign flag

#ifdef GL_DEBUG_OUTPUT
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(glDebugCallback, nullptr);
#endif

    std::cout << "Vendor  : "   << glGetString(GL_VENDOR)   << "\n";
    std::cout << "Renderer: "   << glGetString(GL_RENDERER) << "\n";
    std::cout << "Version : "   << glGetString(GL_VERSION)  << "\n";

    // --- Fullscreen quad (clip space) ---
    float verts[] = {
        // pos     // uv
        -1.f, -1.f,  0.f, 0.f,
         1.f, -1.f,  1.f, 0.f,
         1.f,  1.f,  1.f, 1.f,
        -1.f,  1.f,  0.f, 1.f
    };
    unsigned int idx[] = { 0,1,2,  2,3,0 };

    GLuint vao=0, vbo=0, ebo=0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // --- Load texture ---
    int tw=0, th=0, ch=0;
    unsigned char* pixels = stbi_load("assets/checker.png", &tw, &th, &ch, STBI_rgb_alpha);
    if (!pixels) {
        std::cerr << "Failed to load texture: assets/checker.png\n";
        return -1;
    }

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(pixels);

    // --- Create samplers ---
    GLuint sampHi = 0, sampLo = 0;
    glGenSamplers(1, &sampHi);
    glGenSamplers(1, &sampLo);

    // Sharp sampler (no extra bias)
    glSamplerParameteri(sampHi, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glSamplerParameteri(sampHi, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameterf(sampHi, GL_TEXTURE_LOD_BIAS, 0.0f);
    // Optional anisotropy:
    // glSamplerParameterf(sampHi, GL_TEXTURE_MAX_ANISOTROPY, 4.0f);

    // Biased sampler (periphery)
    glSamplerParameteri(sampLo, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glSamplerParameteri(sampLo, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    float lodBias = 1.2f; // start biased (tweak with keys)
    glSamplerParameterf(sampLo, GL_TEXTURE_LOD_BIAS, lodBias);
    glSamplerParameterf(sampLo, GL_TEXTURE_MAX_ANISOTROPY, 1.0f);

    // --- Program and uniforms ---
    GLuint prog = createProgram(kVS, kFS);
    glUseProgram(prog);

    GLint uTexHi = glGetUniformLocation(prog, "texHi");
    GLint uTexLo = glGetUniformLocation(prog, "texLo");
    glUniform1i(uTexHi, 0); // unit 0
    glUniform1i(uTexLo, 1); // unit 1

    GLint uROICenter  = glGetUniformLocation(prog, "roiCenter");
    GLint uROIRadius  = glGetUniformLocation(prog, "roiRadius");
    GLint uROIFeather = glGetUniformLocation(prog, "roiFeather");

    // Initial ROI params (pixels)
    int fbW=0, fbH=0; glfwGetFramebufferSize(window, &fbW, &fbH);
    float roiRadius  = 90.0f;
    float roiFeather = 50.0f;
    float roiX = fbW * 0.5f;
    float roiY = fbH * 0.5f;

    // --- Main loop ---
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Update viewport (in case of resize)
        glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);

        // ROI follows mouse (comment this block if you want static center)
        double mx=0.0, my=0.0;
        glfwGetCursorPos(window, &mx, &my);
        roiX = static_cast<float>(mx);
        roiY = static_cast<float>(fbH - my); // flip Y to match gl_FragCoord

        // Adjust periphery bias with keys: ']' and '['
        if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) lodBias += 0.01f;
        if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET)  == GLFW_PRESS) lodBias -= 0.01f;
        lodBias = std::clamp(lodBias, -0.25f, 3.0f);
        glSamplerParameterf(sampLo, GL_TEXTURE_LOD_BIAS, lodBias);

        // Upload ROI uniforms
        glUseProgram(prog);
        glUniform2f(uROICenter,  roiX, roiY);
        glUniform1f(uROIRadius,  roiRadius);
        glUniform1f(uROIFeather, roiFeather);

        // Clear + draw
        glClearColor(0.07f, 0.10f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Bind same texture on two units; bind different samplers
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindSampler(0, sampHi);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindSampler(1, sampLo);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteSamplers(1, &sampHi);
    glDeleteSamplers(1, &sampLo);
    glDeleteProgram(prog);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
