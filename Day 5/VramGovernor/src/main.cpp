// Day 5R — Priority-aware VRAM Governor with REAL VRAM commitment (Day-4 style)
// Forces driver to commit VRAM for each pad by FBO clear + mipgen.
// Auto-switches from telemetry to fallback if driver counter doesn't move.
// Hotkeys: B (alloc ~256MB), Shift+B (free), [ / ] (global bias nudge), R (reset), C (toggle telemetry manually)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#ifndef GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX
#define GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX         0x9047
#define GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX    0x9048
#define GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX  0x9049
#define GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX            0x904A
#define GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX            0x904B
#endif

#ifndef GL_TEXTURE_MAX_LEVEL
#define GL_TEXTURE_MAX_LEVEL 0x813D
#endif

// ---------- Small GL helpers ----------
static void glCheck(const char* where){
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) std::fprintf(stderr,"[GL] err=0x%X at %s\n",(unsigned)e, where);
}

static GLuint compile(GLenum type, const char* src){
    GLuint s=glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ GLint len=0; glGetShaderiv(s,GL_INFO_LOG_LENGTH,&len); std::string log(len,'\0');
        glGetShaderInfoLog(s,len,nullptr,log.data()); std::fprintf(stderr,"shader err:\n%s\n",log.c_str()); std::exit(1);}
    return s;
}
static GLuint link(GLuint vs, GLuint fs){
    GLuint p=glCreateProgram(); glAttachShader(p,vs); glAttachShader(p,fs); glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ GLint len=0; glGetProgramiv(p,GL_INFO_LOG_LENGTH,&len); std::string log(len,'\0');
        glGetProgramInfoLog(p,len,nullptr,log.data()); std::fprintf(stderr,"link err:\n%s\n",log.c_str()); std::exit(1);}
    return p;
}

// ---------- Fullscreen tri-strip ----------
static const float QUAD[] = {
    -1.f,-1.f, 0.f,0.f,   1.f,-1.f, 1.f,0.f,   1.f, 1.f, 1.f,1.f,
    -1.f,-1.f, 0.f,0.f,   1.f, 1.f, 1.f,1.f,  -1.f, 1.f, 0.f,1.f
};

static const char* VS = R"(#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){ vUV=aUV; gl_Position=vec4(aPos,0.0,1.0); })";

static const char* FS = R"(#version 330 core
in vec2 vUV; out vec4 fragColor;
uniform sampler2D uTex; uniform float uBias;
void main(){ vec3 c = texture(uTex, vUV, uBias).rgb; fragColor=vec4(c,1.0); })";

// ---------- Checker texture ----------
static std::vector<uint8_t> makeChecker(int w,int h,int chk=32){
    std::vector<uint8_t> v((size_t)w*h*4);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){
        int cx=(x/chk)&1, cy=(y/chk)&1; uint8_t t=(cx^cy)?230:30;
        size_t i=((size_t)y*w+x)*4; v[i]=v[i+1]=v[i+2]=t; v[i+3]=255;
    }
    return v;
}
static GLuint makeCheckerTex(int W=2048,int H=2048){
    auto pix = makeChecker(W,H,32);
    GLuint t=0; glGenTextures(1,&t); glBindTexture(GL_TEXTURE_2D,t);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,pix.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,0);
    return t;
}

// ---------- “Pad” textures that COMMIT VRAM (≈256MB each) ----------
// 8192x8192 RGBA8 ≈ 256 MiB at base level; with mipmaps ≈ +33%.
static const int PAD_W = 8192;
static const int PAD_H = 8192;

struct Pad {
    GLuint tex=0, fbo=0;
};
static std::vector<Pad> gPads;

static Pad createCommittedPad(){
    Pad P{};
    glGenTextures(1,&P.tex);
    glBindTexture(GL_TEXTURE_2D, P.tex);

    // Allocate immutable storage + full mip pyramid
    int levels = 1 + (int)std::floor(std::log2(std::max(PAD_W,PAD_H)));
    glTexStorage2D(GL_TEXTURE_2D, levels, GL_RGBA8, PAD_W, PAD_H);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, levels-1);

    // Force physical commitment without huge CPU uploads:
    // 1) Attach level 0 to an FBO and clear → driver must materialize memory
    glGenFramebuffers(1, &P.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, P.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, P.tex, 0);
    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr,"[FBO] incomplete, status=0x%X\n", st);
    }

    // Clear at full size (we don't need to change viewport; we can just clear)
    glClearColor(0.11f, 0.12f, 0.14f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // 2) Generate mips -> further materializes the pyramid
    glBindTexture(GL_TEXTURE_2D, P.tex);
    glGenerateMipmap(GL_TEXTURE_2D);

    // Cleanup binds
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glCheck("createCommittedPad");
    return P;
}

static void destroyPad(Pad& P){
    if (P.fbo) glDeleteFramebuffers(1,&P.fbo);
    if (P.tex) glDeleteTextures(1,&P.tex);
    P.fbo=0; P.tex=0;
}

// ---------- Telemetry / Fallback ----------
enum class TelMode { NVX, ATI, FALLBACK };

struct Telemetry {
    TelMode mode = TelMode::FALLBACK;
    bool nvx=false, ati=false;
    bool useTelemetry=true;          // we will auto-flip this if frozen
    int  fallbackBaseFreeMB = 2048;  // only used in fallback
    int  padBlocks = 0;

    void init(){
        GLint n=0; glGetIntegerv(GL_NUM_EXTENSIONS,&n);
        for(GLint i=0;i<n;++i){
            const char* ext = (const char*)glGetStringi(GL_EXTENSIONS,(GLuint)i);
            if(!ext) continue; std::string s(ext);
            if(s=="GL_NVX_gpu_memory_info") nvx=true;
            if(s=="GL_ATI_meminfo")         ati=true;
        }
        mode = nvx ? TelMode::NVX : (ati ? TelMode::ATI : TelMode::FALLBACK);
        std::printf("[Init] Telemetry NVX=%d ATI=%d -> %s\n",
            nvx?1:0, ati?1:0, mode==TelMode::NVX?"NVX":(mode==TelMode::ATI?"ATI":"FALLBACK"));
    }

    std::pair<bool,int> readFreeMB(){
        if(useTelemetry){
            if(mode==TelMode::NVX){
                GLint kb=0; glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX,&kb);
                if(glGetError()==GL_NO_ERROR && kb>0) return {true, kb/1024};
            } else if(mode==TelMode::ATI){
                GLint kb[4]={0,0,0,0}; glGetIntegerv(0x87FC, kb); // GL_TEXTURE_FREE_MEMORY_ATI
                if(glGetError()==GL_NO_ERROR && kb[0]>0) return {true, kb[0]/1024};
            }
        }
        int freeMB = std::max(0, fallbackBaseFreeMB - padBlocks * 256);
        return {false, freeMB};
    }
};

// Auto-detect frozen telemetry: if two consecutive pad allocations
// don’t move the telemetry by >=128 MB, switch to fallback automatically.
struct TelWatchdog {
    int lastMB = -1;
    int consecutiveNoMoves = 0;
    void onAllocCheck(Telemetry& tel){
        auto [valid, nowMB] = tel.readFreeMB();
        if(!valid) return; // already fallback
        if(lastMB<0){ lastMB = nowMB; return; }
        int delta = lastMB - nowMB; // expected positive if memory decreased
        lastMB = nowMB;
        if (delta < 128) {
            consecutiveNoMoves++;
            if (consecutiveNoMoves >= 2) {
                tel.useTelemetry = false;
                std::printf("[Auto] Telemetry frozen → switching to FALLBACK.\n");
            }
        } else {
            consecutiveNoMoves = 0;
        }
    }
} gWatch;

// ---------- Governor ----------
struct Governor {
    int   targetFreeMB = 1024;
    int   hysteresisMB = 128;
    int   spikeThreshMB = 256;
    float biasLow=0.f, biasNorm=0.f, biasHigh=0.f;
    float biasMin=0.f, biasMax=8.f;
    float stepGradual=0.5f, stepSpike=1.25f;
    float globalNudge=0.f;

    int lastFreeMB=-1;
    double lastEval=0.0, evalDt=0.25;
    double lastPrint=0.0;

    void clamp(){ auto C=[&](float& x){ x=std::clamp(x,biasMin,biasMax); };
        C(biasLow); C(biasNorm); C(biasHigh); }
    void reset(){ biasLow=biasNorm=biasHigh=0.f; globalNudge=0.f; lastFreeMB=-1; }

    void escalate(){ if(biasLow<biasMax) biasLow+=stepGradual;
                     else if(biasNorm<biasMax) biasNorm+=stepGradual;
                     else if(biasHigh<biasMax) biasHigh+=stepGradual; clamp(); }
    void deescalate(){ if(biasHigh>biasMin) biasHigh-=stepGradual;
                       else if(biasNorm>biasMin) biasNorm-=stepGradual;
                       else if(biasLow>biasMin) biasLow-=stepGradual; clamp(); }
    void spike(){ biasLow += stepSpike; clamp(); }
    void nudge(float d){ globalNudge = std::clamp(globalNudge+d, -4.f, 4.f); }

    void evaluate(double now, int freeMB, bool telValid){
        if(lastFreeMB<0){ lastFreeMB=freeMB; lastEval=now; return; }
        if(now-lastEval < evalDt) return;
        lastEval = now;

        int delta = freeMB - lastFreeMB; // negative = drop
        if(delta <= -spikeThreshMB) spike();
        int lo = targetFreeMB - hysteresisMB;
        int hi = targetFreeMB + hysteresisMB;
        if(freeMB < lo) escalate();
        else if(freeMB > hi) deescalate();
        lastFreeMB = freeMB;

        if(now-lastPrint>0.5){
            lastPrint=now;
            std::printf("freeMB=%4d (Δ %+4d) [%s] Bias L/N/H=%.2f/%.2f/%.2f global=%.2f\n",
                freeMB, delta, telValid?"telemetry":"fallback",
                biasLow,biasNorm,biasHigh,globalNudge);
        }
    }
} gGov;

// ---------- GL State ----------
static GLuint gProg=0, gVAO=0, gVBO=0, gSceneTex=0;
static Telemetry gTel;
static bool gRunning=true;

// ---------- Draw three panels ----------
static void drawPanel(int x,int y,int w,int h, float bias, GLuint tex){
    glViewport(x,y,w,h);
    glUseProgram(gProg);
    glUniform1f(glGetUniformLocation(gProg,"uBias"), bias);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(gProg,"uTex"), 0);
    glBindVertexArray(gVAO);
    glDrawArrays(GL_TRIANGLES,0,6);
}

// ---------- Input ----------
static void keyCB(GLFWwindow*, int key,int, int action,int mods){
    if(action!=GLFW_PRESS && action!=GLFW_REPEAT) return;
    switch(key){
        case GLFW_KEY_ESCAPE: gRunning=false; break;
        case GLFW_KEY_B: {
            // Make and commit a pad; check telemetry right after
            Pad P = createCommittedPad();
            gPads.push_back(P);
            gTel.padBlocks = (int)gPads.size();
            glFinish();                 // ensure work is flushed so NVX can update
            gWatch.onAllocCheck(gTel);  // auto-fallback if frozen
            std::printf("[Pad] +%dMB  pads=%d\n", 256, (int)gPads.size());
        } break;
        case GLFW_KEY_LEFT_BRACKET:  gGov.nudge(-0.125f); break;
        case GLFW_KEY_RIGHT_BRACKET: gGov.nudge(+0.125f); break;
        case GLFW_KEY_R: {
            for(auto& P: gPads) destroyPad(P);
            gPads.clear(); gTel.padBlocks=0; gGov.reset();
            std::printf("[Reset] pads cleared, biases reset.\n");
        } break;
        case GLFW_KEY_C: {
            gTel.useTelemetry = !gTel.useTelemetry;
            std::printf("[Toggle] useTelemetry=%s\n", gTel.useTelemetry?"true":"false");
        } break;
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT: break;
        default:
            // Shift+B to free one
            if((mods & GLFW_MOD_SHIFT) && key==GLFW_KEY_B){
                if(!gPads.empty()){ destroyPad(gPads.back()); gPads.pop_back(); gTel.padBlocks=(int)gPads.size();
                    std::printf("[Pad] -%dMB  pads=%d\n", 256, (int)gPads.size()); }
            }
        break;
    }
}

// ---------- Main ----------
int main(){
    if(!glfwInit()){ std::fprintf(stderr,"GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(1200,400,"Day 5R: Real VRAM + Priority Governor",nullptr,nullptr);
    if(!win){ std::fprintf(stderr,"Window create failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    if(glewInit()!=GLEW_OK){ std::fprintf(stderr,"GLEW init failed\n"); return 1; }

    std::printf("Vendor  : %s\n", (const char*)glGetString(GL_VENDOR));
    std::printf("Renderer: %s\n", (const char*)glGetString(GL_RENDERER));
    std::printf("Version : %s\n", (const char*)glGetString(GL_VERSION));

    glfwSetKeyCallback(win, keyCB);

    // Geometry
    glGenBuffers(1,&gVBO); glBindBuffer(GL_ARRAY_BUFFER,gVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(QUAD),QUAD,GL_STATIC_DRAW);
    glGenVertexArrays(1,&gVAO); glBindVertexArray(gVAO);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(float)*4,(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(float)*4,(void*)(sizeof(float)*2));
    glBindVertexArray(0); glBindBuffer(GL_ARRAY_BUFFER,0);

    GLuint vs=compile(GL_VERTEX_SHADER,VS), fs=compile(GL_FRAGMENT_SHADER,FS);
    gProg=link(vs,fs); glDeleteShader(vs); glDeleteShader(fs);

    gSceneTex = makeCheckerTex(2048,2048);

    // Telemetry: start by trusting it; watchdog will switch to fallback if frozen
    gTel.init();
    // Nice baseline if we fall back later; seed from NVX total if available
    GLint kbTotal=0; glGetIntegerv(GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX, &kbTotal);
    if(glGetError()==GL_NO_ERROR && kbTotal>0) gTel.fallbackBaseFreeMB = (kbTotal/1024)*9/10;
    else gTel.fallbackBaseFreeMB = 6000; // harmless default

    std::puts("Hotkeys: B (+256MB), Shift+B (-256MB), [ / ] nudge, R reset, C toggle telemetry");

    while(!glfwWindowShouldClose(win) && gRunning){
        glfwPollEvents();
        int W,H; glfwGetFramebufferSize(win,&W,&H);
        glViewport(0,0,W,H);
        glDisable(GL_DEPTH_TEST);
        glClearColor(0.11f,0.12f,0.14f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        auto [valid, freeMB] = gTel.readFreeMB();
        double t = glfwGetTime();
        gGov.evaluate(t, freeMB, valid);

        // Panels: Left=Low, Center=Normal, Right=High
        int third = std::max(1, W/3);
        drawPanel(0,0, third, H, gGov.biasLow  + gGov.globalNudge, gSceneTex);
        drawPanel(third,0, third, H, gGov.biasNorm + gGov.globalNudge, gSceneTex);
        drawPanel(third*2,0, W - third*2, H, gGov.biasHigh + gGov.globalNudge, gSceneTex);

        // HUD in title
        char title[256];
        std::snprintf(title,sizeof(title),
            "Day5R | freeMB=%d [%s] | Bias L/N/H=%.2f/%.2f/%.2f | pads=%zu",
            freeMB, valid?"telemetry":"fallback", gGov.biasLow,gGov.biasNorm,gGov.biasHigh, gPads.size());
        glfwSetWindowTitle(win, title);

        glfwSwapBuffers(win);
    }

    for(auto& P: gPads) destroyPad(P);
    glDeleteTextures(1,&gSceneTex);
    glDeleteVertexArrays(1,&gVAO);
    glDeleteBuffers(1,&gVBO);
    glDeleteProgram(gProg);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
