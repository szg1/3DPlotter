# Integer-Feasible Surface Plotter

This repository contains tools to visualize 3D mathematical surfaces along with integer-feasible points constrained by linear inequalities. It's particularly useful for integer programming or operational research visualization.

## Features

- **3D Surface Rendering:** Visualizes mathematical planes and surfaces in 3D space.
- **Constraint Visualization:** Highlights the integer points that satisfy a given set of linear inequalities.
- **Auto-framing Camera:** Automatically adjusts the camera to keep the relevant points inside the frame.
- **Auto-Rotation:** Continuously rotates the model for better spatial understanding.

## Directory Structure

The project has two implementations:
- `cpp_mac/`: High-performance C++ implementations for macOS, using OpenGL or Metal for rendering.
- `py_deprecated/`: An older, deprecated Python implementation using Matplotlib and SymPy.

---

## 1. C++ macOS Implementations (`cpp_mac/`)

This directory contains two rendering backends for macOS: one based on OpenGL (legacy) and one based on Metal (modern).

The constraints are defined directly inside the C++ code for these backends (e.g., `x >= 0`, `y >= 0`, `x <= 40`, `2x + y <= 100`, `x + y <= 80` with the surface `z = 3x + 2y`).

### OpenGL Version (`sips.cpp`)

Uses the deprecated OpenGL framework in macOS (GLUT).

**Requirements:**
- macOS with Xcode Command Line Tools installed.

**Build and Run:**
```bash
cd cpp_mac
./build.sh
./plot
```
(Alternatively, build manually with `clang++ sips.cpp -o plot -std=c++17 -framework OpenGL -framework GLUT`)

### Metal Version (`sips.mm`)

A modern macOS implementation using Apple's Metal graphics API. This is the recommended backend for macOS as OpenGL is deprecated.

**Requirements:**
- macOS with Xcode Command Line Tools installed.

**Build and Run:**
```bash
cd cpp_mac
./buildmm.sh
./PlotMetal
```
(Alternatively, build manually with `clang++ sips.mm -o PlotMetal -std=c++17 -fobjc-arc -framework Cocoa -framework Metal -framework MetalKit -framework QuartzCore`)

---

## 2. Python Implementation (Deprecated) (`py_deprecated/`)

An older implementation written in Python. It reads the constraints and functions from a file (`fun.plt`) and renders them using `matplotlib`.

**Requirements:**
- Python 3
- Required packages: `numpy`, `sympy`, `matplotlib`

**Setup:**
```bash
cd py_deprecated
pip install -r requirements.txt
```

**Run:**
You need to provide a `fun.plt` file in the same directory (you can copy the one from `cpp_mac/fun.plt` to `py_deprecated/`).

```bash
cp ../cpp_mac/fun.plt .
python main.py
```
