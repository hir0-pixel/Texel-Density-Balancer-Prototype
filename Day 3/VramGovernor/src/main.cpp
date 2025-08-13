// Day 3 – stable textured cube (no ROI, no FBO), LOD bias on [ and ]
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float frameCount = 0.0f;
float fps = 0.0f;
auto lastTime = std::chrono::high_resolution_clock::now();

static void APIENTRY glDebugCallback(GLenum, GLenum, GLuint, GLenum, GLsizei, const GLchar* msg, const void*) {
    std::cerr << "[GL] " << msg << "\n";
}

// ---------------- Shaders ----------------
static const char* kVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
uniform mat4 uMVP;
out vec2 vUV;
void main(){
    vUV = aUV;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

static const char* kFS = R"(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main(){
    fragColor = texture(uTex, vUV);
}
)";

// --------------- Utils ---------------
static GLuint compile(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ char log[2048]; glGetShaderInfoLog(s,2048,nullptr,log); std::cerr<<"SH error:\n"<<log<<"\n"; }
    return s;
}
static GLuint linkProgram(const char* vs, const char* fs){
    GLuint v=compile(GL_VERTEX_SHADER,vs);
    GLuint f=compile(GL_FRAGMENT_SHADER,fs);
    GLuint p=glCreateProgram();
    glAttachShader(p,v); glAttachShader(p,f);
    glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ char log[2048]; glGetProgramInfoLog(p,2048,nullptr,log); std::cerr<<"PR error:\n"<<log<<"\n"; }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// --------------- Matrices (column-major) ---------------
static void makePerspective(float fovyRad, float aspect, float zn, float zf, float out[16]){
    float f = 1.0f / std::tan(fovyRad * 0.5f);
    out[0]=f/aspect; out[4]=0; out[8]=0;                     out[12]=0;
    out[1]=0;        out[5]=f; out[9]=0;                     out[13]=0;
    out[2]=0;        out[6]=0; out[10]=(zf+zn)/(zn-zf);      out[14]=(2*zf*zn)/(zn-zf);
    out[3]=0;        out[7]=0; out[11]=-1.0f;                out[15]=0;
}
static void normalize3(float v[3]){
    float L=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); if(L>0){ v[0]/=L; v[1]/=L; v[2]/=L; }
}
static void cross3(const float a[3], const float b[3], float out[3]){
    out[0]=a[1]*b[2]-a[2]*b[1]; out[1]=a[2]*b[0]-a[0]*b[2]; out[2]=a[0]*b[1]-a[1]*b[0];
}
static void makeLookAt(float ex,float ey,float ez, float cx,float cy,float cz, float ux,float uy,float uz, float out[16]){
    float f[3]={cx-ex, cy-ey, cz-ez}; normalize3(f);
    float up[3]={ux,uy,uz}; normalize3(up);
    float s[3]; cross3(f,up,s); normalize3(s);
    float u[3]; cross3(s,f,u);
    // column-major
    out[0]=s[0]; out[4]=s[1]; out[8] =s[2]; out[12]=-(s[0]*ex + s[1]*ey + s[2]*ez);
    out[1]=u[0]; out[5]=u[1]; out[9] =u[2]; out[13]=-(u[0]*ex + u[1]*ey + u[2]*ez);
    out[2]=-f[0];out[6]=-f[1];out[10]=-f[2];out[14]= (f[0]*ex + f[1]*ey + f[2]*ez);
    out[3]=0;    out[7]=0;    out[11]=0;    out[15]=1;
}
static void mul44(const float a[16], const float b[16], float out[16]){
    // column-major: out = a * b
    for(int c=0;c<4;++c){
        for(int r=0;r<4;++r){
            out[c*4 + r] =
                a[0*4 + r]*b[c*4 + 0] +
                a[1*4 + r]*b[c*4 + 1] +
                a[2*4 + r]*b[c*4 + 2] +
                a[3*4 + r]*b[c*4 + 3];
        }
    }
}

// --------------- Geometry (full cube: 36 indices) ---------------
struct V { float x,y,z,u,v; };
static const V cubeVerts[] = {
    // Front (+Z)
    {-1,-1, 1, 0,0}, { 1,-1, 1, 1,0}, { 1, 1, 1, 1,1}, {-1, 1, 1, 0,1},
    // Back (-Z)
    { 1,-1,-1, 0,0}, {-1,-1,-1, 1,0}, {-1, 1,-1, 1,1}, { 1, 1,-1, 0,1},
    // Left (-X)
    {-1,-1,-1, 0,0}, {-1,-1, 1, 1,0}, {-1, 1, 1, 1,1}, {-1, 1,-1, 0,1},
    // Right (+X)
    { 1,-1, 1, 0,0}, { 1,-1,-1, 1,0}, { 1, 1,-1, 1,1}, { 1, 1, 1, 0,1},
    // Top (+Y)
    {-1, 1, 1, 0,0}, { 1, 1, 1, 1,0}, { 1, 1,-1, 1,1}, {-1, 1,-1, 0,1},
    // Bottom (-Y)
    {-1,-1,-1, 0,0}, { 1,-1,-1, 1,0}, { 1,-1, 1, 1,1}, {-1,-1, 1, 0,1},
};
static const unsigned cubeIdx[] = {
    // front
    0, 1, 2, 2, 3, 0,
    // back
    4, 5, 6, 6, 7, 4,
    // left
    8, 9,10,10,11, 8,
    // right
   12,13,14,14,15,12,
    // top
   16,17,18,18,19,16,
    // bottom
   20,21,22,22,23,20
};

int main(){
    // --- Window & context ---
    if(!glfwInit()){ std::cerr<<"Failed to init GLFW\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DEPTH_BITS,24); // request a depth buffer

    GLFWwindow* window = glfwCreateWindow(1280,720,"Day 3 – Stable Cube",nullptr,nullptr);
    if(!window){ std::cerr<<"Failed to create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    if(glewInit()!=GLEW_OK){ std::cerr<<"GLEW init failed\n"; return -1; }
    glGetError(); // clear benign flag

#ifdef GL_DEBUG_OUTPUT
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(glDebugCallback,nullptr);
#endif

    // --- Geometry buffers ---
    GLuint vao,vbo,ebo;
    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
    glGenBuffers(1,&ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(cubeVerts),cubeVerts,GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(cubeIdx),cubeIdx,GL_STATIC_DRAW);

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(V),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(V),(void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    // --- Texture ---
    int tw=0, th=0, ch=0;
    unsigned char* pixels = stbi_load("assets/checker.png",&tw,&th,&ch, STBI_rgb_alpha);
    if(!pixels){ std::cerr<<"Failed to load assets/checker.png\n"; return -1; }

    GLuint tex=0;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,tw,th,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(pixels);

    // --- Sampler (LOD bias adjustable) ---
    GLuint samp=0;
    glGenSamplers(1,&samp);
    glSamplerParameteri(samp,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    glSamplerParameteri(samp,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    float lodBias = 0.0f;
    glSamplerParameterf(samp,GL_TEXTURE_LOD_BIAS,lodBias);

    // --- Program ---
    GLuint prog = linkProgram(kVS,kFS);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog,"uTex"),0);
    GLint uMVP = glGetUniformLocation(prog,"uMVP");

    // --- Main loop ---
    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();

        // LOD bias controls
        if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) lodBias += 0.01f;
        if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET)  == GLFW_PRESS) lodBias -= 0.01f;
        lodBias = std::clamp(lodBias, -0.25f, 3.0f);
        glSamplerParameterf(samp,GL_TEXTURE_LOD_BIAS,lodBias);

        int fbW,fbH; glfwGetFramebufferSize(window,&fbW,&fbH);
        glViewport(0,0,fbW,fbH);
        glClearColor(0.07f,0.10f,0.15f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        // Build MVP (column-major)
        float proj[16], view[16], mvp[16];
        makePerspective(60.0f*(float)M_PI/180.0f, (float)fbW/(float)fbH, 0.1f, 100.0f, proj);

        float t = (float)glfwGetTime();
        float eyeX = std::cos(t)*5.0f, eyeY = 2.0f, eyeZ = std::sin(t)*5.0f;
        makeLookAt(eyeX,eyeY,eyeZ, 0,0,0, 0,1,0, view);

        // model = simple spin around Y
        float c = std::cos(t*0.8f), s = std::sin(t*0.8f);
        float model[16] = {
             c, 0, s, 0,
             0, 1, 0, 0,
            -s, 0, c, 0,
             0, 0, 0, 1
        };

        float pv[16];  mul44(proj, view, pv);
        float mvpCM[16]; mul44(pv, model, mvpCM);

        glUseProgram(prog);
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, mvpCM); // column-major upload (no transpose)

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindSampler(0, samp);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, (GLsizei)(sizeof(cubeIdx)/sizeof(unsigned)), GL_UNSIGNED_INT, 0);
		// Count frames
		frameCount++;
		auto currentTime = std::chrono::high_resolution_clock::now();
		float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();

		// Update every second
		if (deltaTime >= 1.0f) {
			fps = frameCount / deltaTime;
			frameCount = 0;
			lastTime = currentTime;
			std::cout << "FPS: " << fps << " | LOD Bias: " << lodBias << "\n";
		}

        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteSamplers(1,&samp);
    glDeleteTextures(1,&tex);
    glDeleteProgram(prog);
    glDeleteBuffers(1,&ebo);
    glDeleteBuffers(1,&vbo);
    glDeleteVertexArrays(1,&vao);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
