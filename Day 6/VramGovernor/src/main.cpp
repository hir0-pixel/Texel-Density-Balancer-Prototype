// Day 6 — N Governed Objects with Priority Buckets (builds on Day-5R)
// - Real VRAM commitment pads (FBO clear + mipmap), auto-fallback telemetry
// - Governor manages an array of objects grouped by priority buckets
// - Escalation: Low -> Normal -> High ; Recovery: High -> Normal -> Low
// - Per-tick step budget to avoid popping; spike "tourniquet" on Low
// Hotkeys: B (+~256MB pad), Shift+B (free pad), [ / ] (global nudge), R (reset), C (toggle telemetry)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

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

// =================== GL helpers ===================
static GLuint compile(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ GLint len=0; glGetShaderiv(s,GL_INFO_LOG_LENGTH,&len);
        std::string log(len,'\0'); glGetShaderInfoLog(s,len,nullptr,log.data());
        std::fprintf(stderr,"[Shader] compile error:\n%s\n", log.c_str()); std::exit(1); }
    return s;
}
static GLuint link(GLuint vs, GLuint fs){
    GLuint p = glCreateProgram();
    glAttachShader(p,vs); glAttachShader(p,fs); glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ GLint len=0; glGetProgramiv(p,GL_INFO_LOG_LENGTH,&len);
        std::string log(len,'\0'); glGetProgramInfoLog(p,len,nullptr,log.data());
        std::fprintf(stderr,"[Program] link error:\n%s\n", log.c_str()); std::exit(1); }
    return p;
}

static const float QUAD[] = {
    -1.f,-1.f, 0.f,0.f,  1.f,-1.f, 1.f,0.f,  1.f, 1.f, 1.f,1.f,
    -1.f,-1.f, 0.f,0.f,  1.f, 1.f, 1.f,1.f, -1.f, 1.f, 0.f,1.f
};

static const char* VS = R"(#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){ vUV=aUV; gl_Position=vec4(aPos,0.0,1.0); })";

static const char* FS = R"(#version 330 core
in vec2 vUV; out vec4 fragColor;
uniform sampler2D uTex;
uniform float uBias;
void main(){
    vec3 c = texture(uTex, vUV, uBias).rgb;
    fragColor = vec4(c,1.0);
})";

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

// =================== Pad allocator (real commit) ===================
struct Pad { GLuint tex=0, fbo=0; };
static const int PAD_W=8192, PAD_H=8192;
static std::vector<Pad> gPads;

static Pad createCommittedPad(){
    Pad P{};
    glGenTextures(1,&P.tex);
    glBindTexture(GL_TEXTURE_2D, P.tex);
    int levels = 1 + (int)std::floor(std::log2(std::max(PAD_W,PAD_H)));
    glTexStorage2D(GL_TEXTURE_2D, levels, GL_RGBA8, PAD_W, PAD_H);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, levels-1);

    glGenFramebuffers(1,&P.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, P.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, P.tex, 0);
    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr,"[FBO] incomplete 0x%X\n", st);
    }
    glClearColor(0.12f,0.13f,0.15f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, P.tex);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glBindTexture(GL_TEXTURE_2D,0);
    return P;
}
static void destroyPad(Pad& P){
    if(P.fbo) glDeleteFramebuffers(1,&P.fbo);
    if(P.tex) glDeleteTextures(1,&P.tex);
    P.fbo=0; P.tex=0;
}

// =================== Telemetry & fallback ===================
enum class TelMode { NVX, ATI, FALLBACK };
struct Telemetry {
    TelMode mode = TelMode::FALLBACK;
    bool nvx=false, ati=false;
    bool useTelemetry=true;
    int  fallbackBaseFreeMB = 2048;
    int  padBlocks=0;

    void init(){
        GLint n=0; glGetIntegerv(GL_NUM_EXTENSIONS,&n);
        for(GLint i=0;i<n;++i){
            const char* e=(const char*)glGetStringi(GL_EXTENSIONS,(GLuint)i);
            if(!e) continue; std::string s(e);
            if(s=="GL_NVX_gpu_memory_info") nvx=true;
            if(s=="GL_ATI_meminfo") ati=true;
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
                GLint kb[4]={0,0,0,0}; glGetIntegerv(0x87FC, kb);
                if(glGetError()==GL_NO_ERROR && kb[0]>0) return {true, kb[0]/1024};
            }
        }
        int freeMB = std::max(0, fallbackBaseFreeMB - padBlocks*256);
        return {false, freeMB};
    }
} gTel;

struct TelWatchdog {
    int lastMB=-1, noMoves=0;
    void onAllocCheck(){
        auto [valid, nowMB] = gTel.readFreeMB();
        if(!valid) return;
        if(lastMB<0){ lastMB=nowMB; return; }
        int delta = lastMB - nowMB;
        lastMB = nowMB;
        if(delta < 128){
            if(++noMoves >= 2){
                gTel.useTelemetry = false;
                std::printf("[Auto] Telemetry frozen -> FALLBACK\n");
            }
        } else noMoves=0;
    }
} gWatch;

// =================== Day 6 Data Model ===================
enum class Priority : int { Low=0, Normal=1, High=2 };

struct GovObject {
    int         id = -1;
    Priority    priority = Priority::Normal;
    float       bias = 0.f;
    float       biasMin = 0.f;
    float       biasMax = 8.f;
    bool        visible = true;
    float       estMB = 64.f; // optional: estimated memory footprint
    // Draw placement (for our grid demo)
    int gridX=0, gridY=0;
};

struct Governor {
    // Global goals
    int   targetFreeMB = 1024;
    int   hysteresisMB = 128;
    int   spikeThreshMB= 256;

    // dynamics
    float stepGradual   = 0.5f;
    float stepSpike     = 1.25f;
    int   stepBudgetPerTick = 4; // number of object-steps per tick

    // time
    int    lastFreeMB=-1;
    double lastEval=0.0;
    double evalDt=0.25;
    double lastPrint=0.0;

    // debug/global
    float globalNudge=0.f;

    // objects
    std::vector<GovObject> objects;

    // scratch buckets (indices into objects)
    std::vector<int> bucketLow, bucketNorm, bucketHigh;

    void rebuildBuckets(){
        bucketLow.clear(); bucketNorm.clear(); bucketHigh.clear();
        for(int i=0;i<(int)objects.size();++i){
            if(!objects[i].visible) continue;
            switch(objects[i].priority){
                case Priority::Low:    bucketLow.push_back(i);  break;
                case Priority::Normal: bucketNorm.push_back(i); break;
                case Priority::High:   bucketHigh.push_back(i); break;
            }
        }
        // Example policy: within each bucket, largest memory first
        auto sortByEstMB = [&](std::vector<int>& b){
            std::sort(b.begin(), b.end(), [&](int a,int b){ return objects[a].estMB > objects[b].estMB; });
        };
        sortByEstMB(bucketLow);
        sortByEstMB(bucketNorm);
        sortByEstMB(bucketHigh);
    }

    static void clampObj(GovObject& o){ o.bias = std::clamp(o.bias, o.biasMin, o.biasMax); }

    void applySteps(std::vector<int>& bucket, float delta, int& budget){
        for(int idx : bucket){
            if(budget<=0) break;
            GovObject& o = objects[idx];
            float old = o.bias;
            o.bias += delta;
            clampObj(o);
            if (o.bias != old) --budget; // count only if it actually changed
        }
    }

    void spikeTourniquet(){
        // Hit the Low bucket first, stronger step; budget-limited
        int budget = stepBudgetPerTick;
        applySteps(bucketLow, +stepSpike, budget);
    }

    void escalate(){
        int budget = stepBudgetPerTick;
        // Low -> Normal -> High
        applySteps(bucketLow,  +stepGradual, budget);
        applySteps(bucketNorm, +stepGradual, budget);
        applySteps(bucketHigh, +stepGradual, budget);
    }

    void deescalate(){
        int budget = stepBudgetPerTick;
        // High -> Normal -> Low
        applySteps(bucketHigh, -stepGradual, budget);
        applySteps(bucketNorm, -stepGradual, budget);
        applySteps(bucketLow,  -stepGradual, budget);
    }

    void nudge(float d){ globalNudge = std::clamp(globalNudge+d, -4.f, 4.f); }

    void evaluate(double now, int freeMB, bool telValid){
        if(lastFreeMB<0){ lastFreeMB=freeMB; lastEval=now; return; }
        if(now-lastEval < evalDt) return;
        lastEval = now;

        int delta = freeMB - lastFreeMB; // negative = drop
        lastFreeMB = freeMB;

        rebuildBuckets();

        if (delta <= -spikeThreshMB) spikeTourniquet();

        int lo = targetFreeMB - hysteresisMB;
        int hi = targetFreeMB + hysteresisMB;

        if      (freeMB < lo) escalate();
        else if (freeMB > hi) deescalate();

        if(now-lastPrint>0.5){
            lastPrint=now;
            std::printf("freeMB=%4d (Δ %+4d) [%s] objs=%zu  L/N/H=%zu/%zu/%zu  nudge=%.2f\n",
                freeMB, delta, telValid?"telemetry":"fallback",
                objects.size(), bucketLow.size(), bucketNorm.size(), bucketHigh.size(), globalNudge);
        }
    }
} gGov;

// =================== GL state & rendering ===================
static GLuint gProg=0, gVAO=0, gVBO=0, gTex=0;
static bool gRunning=true;

static void drawQuadViewport(int x,int y,int w,int h, float bias){
    glViewport(x,y,w,h);
    glUseProgram(gProg);
    GLint locBias = glGetUniformLocation(gProg,"uBias");
    glUniform1f(locBias, bias + gGov.globalNudge);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gTex);
    glUniform1i(glGetUniformLocation(gProg,"uTex"), 0);
    glBindVertexArray(gVAO);
    glDrawArrays(GL_TRIANGLES,0,6);
    glBindVertexArray(0);
}

static void drawObjectsGrid(int fbW,int fbH){
    // Simple 3x2 grid layout for our 6 demo objects, using object.gridX/gridY
    int cols=3, rows=2;
    int cellW = fbW/cols, cellH = fbH/rows;
    for(const auto& o : gGov.objects){
        int vx = o.gridX * cellW;
        int vy = (rows-1 - o.gridY) * cellH; // origin bottom
        drawQuadViewport(vx, vy, cellW, cellH, o.bias);
    }
}

// =================== Input ===================
static void keyCB(GLFWwindow*, int key,int, int action,int mods){
    if(action!=GLFW_PRESS && action!=GLFW_REPEAT) return;
    switch(key){
        case GLFW_KEY_ESCAPE: gRunning=false; break;
        case GLFW_KEY_B: {
            Pad P = createCommittedPad();
            gPads.push_back(P);
            gTel.padBlocks = (int)gPads.size();
            glFinish();
            gWatch.onAllocCheck();
            std::printf("[Pad] +256MB pad=%d\n",(int)gPads.size());
        } break;
        case GLFW_KEY_R: {
            for(auto& P: gPads) destroyPad(P);
            gPads.clear(); gTel.padBlocks=0;
            for(auto& o : gGov.objects) o.bias = 0.f;
            std::printf("[Reset] pads cleared; biases reset.\n");
        } break;
        case GLFW_KEY_C:
            gTel.useTelemetry = !gTel.useTelemetry;
            std::printf("[Toggle] useTelemetry=%s\n", gTel.useTelemetry?"true":"false");
            break;
        case GLFW_KEY_LEFT_BRACKET:  gGov.nudge(-0.125f); break;
        case GLFW_KEY_RIGHT_BRACKET: gGov.nudge(+0.125f); break;
        default:
            if((mods & GLFW_MOD_SHIFT) && key==GLFW_KEY_B){
                if(!gPads.empty()){
                    destroyPad(gPads.back()); gPads.pop_back(); gTel.padBlocks=(int)gPads.size();
                    std::printf("[Pad] -256MB pad=%d\n",(int)gPads.size());
                }
            }
        break;
    }
}

// =================== Main ===================
int main(){
    if(!glfwInit()){ std::fprintf(stderr,"GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(1200,600,"Day 6: N-Object Priority VRAM Governor",nullptr,nullptr);
    if(!win){ std::fprintf(stderr,"Window create failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    if(glewInit()!=GLEW_OK){ std::fprintf(stderr,"GLEW init failed\n"); return 1; }

    std::printf("Vendor  : %s\n", (const char*)glGetString(GL_VENDOR));
    std::printf("Renderer: %s\n", (const char*)glGetString(GL_RENDERER));
    std::printf("Version : %s\n", (const char*)glGetString(GL_VERSION));

    glfwSetKeyCallback(win, keyCB);

    // GL geometry
    glGenBuffers(1,&gVBO); glBindBuffer(GL_ARRAY_BUFFER,gVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(QUAD),QUAD,GL_STATIC_DRAW);
    glGenVertexArrays(1,&gVAO); glBindVertexArray(gVAO);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(float)*4,(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(float)*4,(void*)(sizeof(float)*2));
    glBindVertexArray(0); glBindBuffer(GL_ARRAY_BUFFER,0);

    GLuint vs=compile(GL_VERTEX_SHADER,VS), fs=compile(GL_FRAGMENT_SHADER,FS);
    gProg=link(vs,fs); glDeleteShader(vs); glDeleteShader(fs);

    gTex = makeCheckerTex(2048,2048);

    // Telemetry init + seed fallback baseline
    gTel.init();
    GLint kbTotal=0; glGetIntegerv(GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX, &kbTotal);
    if(glGetError()==GL_NO_ERROR && kbTotal>0) gTel.fallbackBaseFreeMB = (kbTotal/1024)*9/10;
    else gTel.fallbackBaseFreeMB = 6000;

    // --------- Build Day 6 object set (3x2 grid) ---------
    // Two of each priority; different estMB (so largest-first has effect).
    auto addObj = [&](int id, Priority pr, int gx,int gy, float estMB){
        GovObject o; o.id=id; o.priority=pr; o.bias=0.f; o.biasMin=0.f; o.biasMax=8.f;
        o.visible=true; o.estMB=estMB; o.gridX=gx; o.gridY=gy; gGov.objects.push_back(o);
    };
    // Row 0 (bottom): Low, Low, Normal
    addObj(0, Priority::Low,    0,0, 200.f);
    addObj(1, Priority::Low,    1,0, 150.f);
    addObj(2, Priority::Normal, 2,0, 180.f);
    // Row 1 (top): Normal, High, High
    addObj(3, Priority::Normal, 0,1, 120.f);
    addObj(4, Priority::High,   1,1, 220.f); // "main" (largest est)
    addObj(5, Priority::High,   2,1, 100.f);

    std::puts("Hotkeys: B (+256MB), Shift+B (-256MB), [ / ] nudge, R reset, C toggle telemetry");

    while(!glfwWindowShouldClose(win) && gRunning){
        glfwPollEvents();
        int W,H; glfwGetFramebufferSize(win,&W,&H);
        glViewport(0,0,W,H);
        glDisable(GL_DEPTH_TEST);
        glClearColor(0.10f,0.11f,0.13f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        auto [valid, freeMB] = gTel.readFreeMB();
        double t = glfwGetTime();
        gGov.evaluate(t, freeMB, valid);

        drawObjectsGrid(W,H);

        // HUD
        char title[256];
        auto &o0=gGov.objects[0], &o4=gGov.objects[4], &o5=gGov.objects[5];
        std::snprintf(title,sizeof(title),
            "Day6 | freeMB=%d [%s] | objs=%zu | sample biases: L0=%.2f  H4=%.2f  H5=%.2f | pads=%zu",
            freeMB, valid?"telemetry":"fallback",
            gGov.objects.size(), o0.bias, o4.bias, o5.bias, gPads.size());
        glfwSetWindowTitle(win, title);

        glfwSwapBuffers(win);
    }

    for(auto& P: gPads) destroyPad(P);
    glDeleteTextures(1,&gTex);
    glDeleteVertexArrays(1,&gVAO);
    glDeleteBuffers(1,&gVBO);
    glDeleteProgram(gProg);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
