# Texel Density Balancer — VRAM Governor (Prototype)

This contains a prototype system that dynamically adjusts texel density in real time based on GPU VRAM usage. The goal is to maintain rendering stability and visual consistency while preventing memory exhaustion on GPUs, particularly for high-resolution medical visualization and dense 3D scenes.

The project follows a day-by-day learning and development progression. Each day’s folder represents an incremental improvement stage, and files may repeat as concepts build from fundamentals toward a fully adaptive VRAM-aware texture pipeline.

---

## 🧠 Overview

Modern GPU pipelines often allocate textures statically, which can waste memory or crash under extreme load. This prototype introduces a **VRAM governor** that monitors VRAM usage and dynamically scales texture resolution as needed, similar to a power governor, but for memory.

This system ensures:

- Stable rendering under VRAM pressure  
- Real-time texture density adaptation  
- Graceful quality reduction instead of sudden stalls  
- Consistent user experience in memory-constrained environments  

The long-term vision is integration within a clinical 3D visualization system where GPU headroom is critical and medical clarity must be preserved.

---

## ⚙️ How It Works

| Component | Function |
|----------|----------|
| VRAM Monitor | Reads current GPU memory usage periodically |
| Governor Logic | Compares usage against a user-set threshold (0–100%) |
| Texture Policy | Reduces or restores texel density accordingly |
| Real-Time Rebinding | Re-allocates texture data based on available memory |
| Stability Focus | Maintains smooth output without flicker or popping |

This creates an adaptive texture system that reacts in real time to GPU load.

---

## ✅ Current Features

- Real-time VRAM polling
- Adjustable VRAM usage cap (0–100%)
- Dynamic texture down-scaling and re-allocation
- Smooth degradation and recovery behavior
- OpenGL pipeline foundation (GLFW + GLAD)

---

## 🛠️ Planned Enhancements

- Smart texture priority (important textures degrade last)
- Vulkan-based backend for compute-heavy scenes
- Streaming textures & mip-bias control
- Integration into medical DICOM visualization pipeline
- Research-grade performance evaluation tools

---

## 📦 Prerequisites

Before building, install:

- **CMake ≥ 3.20**
- **Visual Studio** (Desktop C++ workload)
- **GLFW** (for windowing + input)
- **GLAD** (OpenGL loader)
- **vcpkg (optional)** — manual vendor libraries also supported

> Clear any previous `build/` folder before rebuilding to avoid stale cache issues.

---

## 🧩 Build Instructions (Windows)

```bash
mkdir build
cd build
cmake ..
cmake --build .
