// Day 4 – Automatic Texel-Density Governor (VRAM + Density fallback)
// - Core-profile safe extension scan (no GL_INVALID_ENUM spam)
// - If VRAM telemetry exists: keep freeMB >= target headroom
// - Else: keep average screen-space density near target
// - Uses MRT to write color + per-pixel density to an offscreen FBO
// - Averages density by mipmapping the R16F metric texture and reading its 1x1

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void APIENTRY glDebugCallback(GLenum, GLenum type, GLuint, GLenum, GLsizei, const GLchar* msg, const void*) {
    // Keep prints small; you can filter by 'type' if needed
    if (type == GL_DEBUG_TYPE_ERROR)
        std::cerr << "[GL] " << msg << "\n";
}

/* ======================= Shaders ======================= */
// Scene pass writes both color and density to two render targets
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
layout(location=0) out vec4 outColor;   // to colorTex
layout(location=1) out vec4 outMetric;  // to metricTex (R in [0..1] density)

uniform sampler2D uTex;

float computeDensityNorm(sampler2D tex, vec2 uv){
    // Estimate mip level using screen-space UV derivatives
    vec2 texSize0 = vec2(textureSize(tex, 0)); // base level (w,h)
    vec2 dUVdx = dFdx(uv) * texSize0;
    vec2 dUVdy = dFdy(uv) * texSize0;
    float rho = max(length(dUVdx), length(dUVdy)); // texels per pixel
    float lambda = log2(max(rho, 1e-8));           // mip level estimate (magnification -> negative)
    // Normalize by theoretical max mip of the bound texture
    float maxEdge = max(texSize0.x, texSize0.y);
    float maxMip = floor(log2(maxEdge));
    float norm = clamp(lambda / max(1.0, maxMip), 0.0, 1.0);
    return norm;
}

void main(){
    vec4 c = texture(uTex, vUV);
    outColor  = c;

    // Per-pixel density 0..1 into R; pack as vec4 for MRT
    float d = computeDensityNorm(uTex, vUV);
    outMetric = vec4(d, 0.0, 0.0, 1.0);
}
)";

/* ======================= Utils ======================= */
static GLuint compile(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS, &ok); // not all drivers set this; just check status:
    glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ char log[2048]; glGetShaderInfoLog(s,2048,nullptr,log); std::cerr<<"SH error:\n"<<log<<"\n"; }
    return s;
}
static GLuint linkProgram(const char* vs, const char* fs){
    GLuint v=compile(GL_VERTEX_SHADER,vs), f=compile(GL_FRAGMENT_SHADER,fs);
    GLuint p=glCreateProgram();
    glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ char log[2048]; glGetProgramInfoLog(p,2048,nullptr,log); std::cerr<<"PR error:\n"<<log<<"\n"; }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

/* ======================= Matrices (column-major) ======================= */
static void makePerspective(float fovyRad, float aspect, float zn, float zf, float out[16]){
    float f = 1.0f / std::tan(fovyRad * 0.5f);
    out[0]=f/aspect; out[4]=0; out[8]=0;                     out[12]=0;
    out[1]=0;        out[5]=f; out[9]=0;                     out[13]=0;
    out[2]=0;        out[6]=0; out[10]=(zf+zn)/(zn-zf);      out[14]=(2*zf*zn)/(zn-zf);
    out[3]=0;        out[7]=0; out[11]=-1.0f;                out[15]=0;
}
static void normalize3(float v[3]){ float L=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); if(L>0){ v[0]/=L; v[1]/=L; v[2]/=L; } }
static void cross3(const float a[3], const float b[3], float out[3]){
    out[0]=a[1]*b[2]-a[2]*b[1]; out[1]=a[2]*b[0]-a[0]*b[2]; out[2]=a[0]*b[1]-a[1]*b[0];
}
static void makeLookAt(float ex,float ey,float ez, float cx,float cy,float cz, float ux,float uy,float uz, float out[16]){
    float f[3]={cx-ex, cy-ey, cz-ez}; normalize3(f);
    float up[3]={ux,uy,uz}; normalize3(up);
    float s[3]; cross3(f,up,s); normalize3(s);
    float u[3]; cross3(s,f,u);
    out[0]=s[0]; out[4]=s[1]; out[8] =s[2]; out[12]=-(s[0]*ex + s[1]*ey + s[2]*ez);
    out[1]=u[0]; out[5]=u[1]; out[9] =u[2]; out[13]=-(u[0]*ex + u[1]*ey + u[2]*ez);
    out[2]=-f[0];out[6]=-f[1];out[10]=-f[2];out[14]= (f[0]*ex + f[1]*ey + f[2]*ez);
    out[3]=0;    out[7]=0;    out[11]=0;    out[15]=1;
}
static void mul44(const float a[16], const float b[16], float out[16]){
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

/* ======================= Cube geometry ======================= */
struct V { float x,y,z,u,v; };
static const V cubeVerts[] = {
    {-1,-1, 1, 0,0}, { 1,-1, 1, 1,0}, { 1, 1, 1, 1,1}, {-1, 1, 1, 0,1},
    { 1,-1,-1, 0,0}, {-1,-1,-1, 1,0}, {-1, 1,-1, 1,1}, { 1, 1,-1, 0,1},
    {-1,-1,-1, 0,0}, {-1,-1, 1, 1,0}, {-1, 1, 1, 1,1}, {-1, 1,-1, 0,1},
    { 1,-1, 1, 0,0}, { 1,-1,-1, 1,0}, { 1, 1,-1, 1,1}, { 1, 1, 1, 0,1},
    {-1, 1, 1, 0,0}, { 1, 1, 1, 1,0}, { 1, 1,-1, 1,1}, {-1, 1,-1, 0,1},
    {-1,-1,-1, 0,0}, { 1,-1,-1, 1,0}, { 1,-1, 1, 1,1}, {-1,-1, 1, 0,1},
};
static const unsigned cubeIdx[] = {
    0,1,2, 2,3,0,    4,5,6, 6,7,4,
    8,9,10, 10,11,8, 12,13,14, 14,15,12,
    16,17,18, 18,19,16, 20,21,22, 22,23,20
};

/* ======================= VRAM telemetry (Core-safe) ======================= */
static bool gHasNVX = false;
static bool gHasATI = false;

static void scanExtensionsOnce(bool& hasNVX, bool& hasATI) {
    hasNVX = hasATI = false;

#ifdef GLEW_NVX_gpu_memory_info
    if (GLEW_NVX_gpu_memory_info) hasNVX = true;
#endif
#ifdef GLEW_ATI_meminfo
    if (GLEW_ATI_meminfo) hasATI = true;
#endif

    if (!hasNVX || !hasATI) {
        GLint n = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &n);
        for (GLint i = 0; i < n; ++i) {
            const char* ext = (const char*)glGetStringi(GL_EXTENSIONS, i);
            if (!ext) continue;
            if (!hasNVX && std::strcmp(ext, "GL_NVX_gpu_memory_info") == 0) hasNVX = true;
            if (!hasATI && std::strcmp(ext, "GL_ATI_meminfo") == 0)         hasATI = true;
        }
    }
}

#ifndef GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX
#define GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX        0x9047
#define GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX   0x9048
#define GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX 0x9049
#endif
#ifndef GL_TEXTURE_FREE_MEMORY_ATI
#define GL_TEXTURE_FREE_MEMORY_ATI 0x87FC
#endif

// Return (totalMB, freeMB); ok=false if neither extension present
static void queryVRAM_MB(int& totalMB, int& freeMB, bool& ok) {
    ok = false; totalMB = -1; freeMB = -1;

    if (gHasNVX) {
        GLint totalKB=0, availKB=0;
        glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &totalKB);
        glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &availKB);
        totalMB = totalKB / 1024;
        freeMB  = availKB / 1024;
        ok = true; return;
    }
    if (gHasATI) {
        GLint vals[4] = {0,0,0,0};
        glGetIntegerv(GL_TEXTURE_FREE_MEMORY_ATI, vals);
        freeMB  = vals[0] / 1024; // heuristic
        totalMB = -1;
        ok = true; return;
    }
}

/* ======================= Dummy 4K texture harness (to create pressure) ======================= */
static std::vector<GLuint> gDummyTex;

static GLuint makeDummy4KTexture(){
    const int W=4096, H=4096;
    std::vector<unsigned char> tmp(W*H*4);
    for(int y=0;y<H;++y){
        for(int x=0;x<W;++x){
            int i=(y*W+x)*4;
            tmp[i+0]=(unsigned char)((x+y)&255);
            tmp[i+1]=(unsigned char)((x*3)&255);
            tmp[i+2]=(unsigned char)((y*7)&255);
            tmp[i+3]=255;
        }
    }
    GLuint t=0; glGenTextures(1,&t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,tmp.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    return t;
}
static void addDummyBatch(int n=10){
    for(int i=0;i<n;++i) gDummyTex.push_back(makeDummy4KTexture());
    std::cout<<"[load] +"<<n<<" dummy 4K textures (total "<<gDummyTex.size()<<")\n";
}
static void freeDummyBatch(int n=10){
    for(int i=0;i<n && !gDummyTex.empty(); ++i){
        GLuint t = gDummyTex.back(); gDummyTex.pop_back();
        glDeleteTextures(1,&t);
    }
    std::cout<<"[load] -"<<n<<" dummy 4K textures (total "<<gDummyTex.size()<<")\n";
}

/* ======================= Offscreen FBO (color + metric) ======================= */
struct FBO {
    GLuint fbo=0, colorTex=0, metricTex=0, rboDepth=0;
    int w=0, h=0, metricMipCount=1;
};

static int mipCountFor(int w, int h){
    int m=1; while (w>1 || h>1){ w=std::max(1,w/2); h=std::max(1,h/2); ++m; }
    return m;
}

static void destroyFBO(FBO& f){
    if (f.rboDepth) glDeleteRenderbuffers(1,&f.rboDepth);
    if (f.metricTex) glDeleteTextures(1,&f.metricTex);
    if (f.colorTex)  glDeleteTextures(1,&f.colorTex);
    if (f.fbo)       glDeleteFramebuffers(1,&f.fbo);
    f = {};
}

static bool createFBO(FBO& f, int w, int h){
    destroyFBO(f);
    f.w=w; f.h=h;
    f.metricMipCount = mipCountFor(w,h);

    glGenFramebuffers(1,&f.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, f.fbo);

    // Color
    glGenTextures(1,&f.colorTex);
    glBindTexture(GL_TEXTURE_2D, f.colorTex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, f.colorTex, 0);

    // Metric (R16F) with full mip chain for averaging
    glGenTextures(1,&f.metricTex);
    glBindTexture(GL_TEXTURE_2D, f.metricTex);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,f.metricMipCount-1);
    for(int level=0, lw=w, lh=h; level<f.metricMipCount; ++level){
        glTexImage2D(GL_TEXTURE_2D, level, GL_R16F, lw, lh, 0, GL_RED, GL_FLOAT, nullptr);
        lw = std::max(1, lw/2);
        lh = std::max(1, lh/2);
    }
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, f.metricTex, 0);

    // Depth (renderbuffer)
    glGenRenderbuffers(1,&f.rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, f.rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, f.rboDepth);

    GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, bufs);

    bool ok = (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return ok;
}

/* ======================= Main ======================= */
int main(){
    // --- Window / GL ---
    if(!glfwInit()){ std::cerr<<"Failed to init GLFW\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DEPTH_BITS,24);
    GLFWwindow* window = glfwCreateWindow(1280,720,"Day 4 – VRAM + Density Governor",nullptr,nullptr);
    if(!window){ std::cerr<<"Failed to create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    if(glewInit()!=GLEW_OK){ std::cerr<<"GLEW init failed\n"; return -1; }
    glGetError(); // clear benign flag from GLEW

#ifdef GL_DEBUG_OUTPUT
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(glDebugCallback,nullptr);
#endif

    // --- Extension scan (once) ---
    scanExtensionsOnce(gHasNVX, gHasATI);
    if (!gHasNVX && !gHasATI) {
        std::cout << "[warn] No VRAM telemetry extension found (NVX/ATI). "
                     "Governor will use density metric.\n";
    }

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

    // --- Texture + sampler for cube ---
    int tw=0, th=0, ch=0;
    unsigned char* pixels = stbi_load("assets/checker.png",&tw,&th,&ch, STBI_rgb_alpha);
    if(!pixels){ std::cerr<<"Failed to load assets/checker.png\n"; return -1; }
    GLuint tex=0; glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,tw,th,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(pixels);

    GLuint samp=0; glGenSamplers(1,&samp);
    glSamplerParameteri(samp,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    glSamplerParameteri(samp,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    float lodBias = 0.0f;
    glSamplerParameterf(samp,GL_TEXTURE_LOD_BIAS,lodBias);

    // --- Program ---
    GLuint prog = linkProgram(kVS,kFS);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog,"uTex"),0);
    GLint uMVP = glGetUniformLocation(prog,"uMVP");

    // --- Offscreen FBO (color+metric) ---
    int fbW=0, fbH=0; glfwGetFramebufferSize(window,&fbW,&fbH);
    FBO fbo; if (!createFBO(fbo, fbW, fbH)) { std::cerr<<"FBO creation failed\n"; return -1; }

    // --- Governor settings ---
    bool governorOn = true;
    int totalMB=-1, freeMB=-1; bool vramOK=false;
    queryVRAM_MB(totalMB, freeMB, vramOK);

    // VRAM headroom target (if available)
    int targetFreeMB = (vramOK && freeMB>0) ? std::max(256, std::min(freeMB, 1536)) : 1024;
    int bandMB = 128;                 // hysteresis ±128MB
    float kp_vram = 0.0035f;          // P-gain for MB error
    float rate    = 0.04f;            // max |bias delta| per update

    // Density target (used always; primary if no VRAM)
    float targetDensity = 0.35f;
    float bandDensity   = 0.03f;      // deadband around target
    float kp_den        = 0.75f * 0.02f; // P-gain for density error (scaled)

    // --- Main loop ---
    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();

        // Keys: load harness, toggle, targets, manual bias
        if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) addDummyBatch(10);
        if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) freeDummyBatch(10);
        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) governorOn = true;
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) governorOn = false;
        if (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS) targetFreeMB += 256;
        if (glfwGetKey(window, GLFW_KEY_COMMA)  == GLFW_PRESS) targetFreeMB = std::max(128, targetFreeMB - 256);
        if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) lodBias += 0.01f;
        if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET)  == GLFW_PRESS) lodBias -= 0.01f;

        // Handle resize (recreate FBO)
        int ww=0, hh=0; glfwGetFramebufferSize(window,&ww,&hh);
        if (ww != fbo.w || hh != fbo.h){
            createFBO(fbo, ww, hh);
        }

        // Build MVP (column-major)
        glViewport(0,0,fbo.w,fbo.h);
        float proj[16], view[16], model[16], pv[16], mvp[16];
        makePerspective(60.0f*(float)M_PI/180.0f, (float)fbo.w/(float)fbo.h, 0.1f, 100.0f, proj);
        float t = (float)glfwGetTime();
        float eyeX = std::cos(t)*5.0f, eyeY = 2.0f, eyeZ = std::sin(t)*5.0f;
        makeLookAt(eyeX,eyeY,eyeZ, 0,0,0, 0,1,0, view);
        float c = std::cos(t*0.8f), s = std::sin(t*0.8f);
        float mdl[16] = { c,0,s,0,  0,1,0,0,  -s,0,c,0,  0,0,0,1 };
        std::copy_n(mdl,16,model);
        mul44(proj, view, pv);
        mul44(pv, model, mvp);

        // --- Draw scene into FBO with MRT (color + metric) ---
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo);
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.07f,0.10f,0.15f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, mvp);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindSampler(0, samp);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, (GLsizei)(sizeof(cubeIdx)/sizeof(unsigned)), GL_UNSIGNED_INT, 0);

        // --- Compute frame-average density via mipmap on metricTex ---
        glBindTexture(GL_TEXTURE_2D, fbo.metricTex);
        glGenerateMipmap(GL_TEXTURE_2D);
        float avgDensity = 0.0f;
        {
            int lastLevel = fbo.metricMipCount - 1; // 1x1
            glGetTexImage(GL_TEXTURE_2D, lastLevel, GL_RED, GL_FLOAT, &avgDensity);
        }

        // --- VRAM telemetry (if available) ---
        queryVRAM_MB(totalMB, freeMB, vramOK);

        // --- Controller: "best of both" ---------------------------------
        // 1) If VRAM present and below threshold band -> bias up aggressively
        if (governorOn && vramOK && freeMB >= 0){
            int low  = targetFreeMB - bandMB;
            int high = targetFreeMB + bandMB;
            if (freeMB < low){
                float err = float(low - freeMB);
                float step = std::min(rate, kp_vram * err);
                lodBias += step; // more blur
            } else if (freeMB > high){
                float err = float(freeMB - high);
                float step = std::min(rate, kp_vram * err);
                lodBias -= step; // sharper
            }
        }

        // 2) Always run density keeper (small correction) so visual quality stabilizes
        if (governorOn){
            float err = avgDensity - targetDensity;
            if (std::fabs(err) > bandDensity){
                // Small-step correction around target
                float step = std::clamp(kp_den * err, -rate*0.5f, rate*0.5f);
                lodBias += step;
            }
        }

        // Clamp bias to hardware-reasonable range
        lodBias = std::clamp(lodBias, -0.25f, 3.0f);
        glSamplerParameterf(samp, GL_TEXTURE_LOD_BIAS, lodBias);

        // --- Blit color to default framebuffer ---
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo.fbo);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0,0,fbo.w,fbo.h, 0,0,ww,hh, GL_COLOR_BUFFER_BIT, GL_LINEAR);

        // --- Console HUD every ~0.7s ---
        static double t0 = glfwGetTime();
        double now = glfwGetTime();
        if (now - t0 > 0.7){
            std::cout
                << "freeMB=" << (vramOK?freeMB:-1)
                << "  targetFreeMB=" << targetFreeMB
                << "  avgDensity=" << avgDensity
                << "  targetDensity=" << targetDensity
                << "  bias=" << lodBias
                << "  dummyTex=" << gDummyTex.size()
                << "  gov:" << (governorOn ? "on" : "off")
                << "\n";
            t0 = now;
        }

        glfwSwapBuffers(window);
    }

    // Cleanup
    for (GLuint t : gDummyTex) glDeleteTextures(1,&t);
    destroyFBO(fbo);
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
