#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads fun.plt (functions & constraints separated by newlines) and plots them in 3D.

File format (examples):
    # Functions
    z = 3*x + 2*y
    z = sin(x) + cos(y)

    # Constraints
    x >= 0
    y >= 0
    x <= 40
    2*x + y <= 100
    x + y <= 80
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import sympy as sp
from pathlib import Path
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from matplotlib import animation

# ---------- config ----------
INPUT_FILE = "fun.plt"
DEFAULT_RANGE = (-10.0, 10.0)
GRID_N = 120
SURFACE_ALPHA = 0.9
BOUNDARY_ALPHA = 0.25
COLORMAP = "viridis"
ROTATION_STEPS = 180   # number of frames for rotation
ROTATION_INTERVAL = 100  # milliseconds between frames
# ----------------------------

x, y, z = sp.symbols('x y z', real=True)

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

def parse_sympy(s: str):
    return parse_expr(
        s,
        local_dict={"x": x, "y": y, "z": z},
        transformations=TRANSFORMS,
        evaluate=True,
    )

ineq_ops = ["<=", ">=", "<", ">"]
ineq_re = re.compile(r"(<=|>=|<|>)")

def clean_line(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    if ":" in line:
        left, right = line.split(":", 1)
        if re.fullmatch(r"[A-Za-z_][\w\- ]*", left.strip()):
            return right.strip()
    return line

def parse_fun_or_constraint(line: str):
    if "=" in line and not ineq_re.search(line):
        lhs, rhs = map(str.strip, line.split("=", 1))
        lhs_expr = parse_sympy(lhs)
        rhs_expr = parse_sympy(rhs)
        if lhs_expr == z:
            return ("func", sp.simplify(rhs_expr))
        elif rhs_expr == z:
            return ("func", sp.simplify(lhs_expr))
        else:
            return ("eq_boundary", (lhs_expr, rhs_expr))
    m = ineq_re.search(line)
    if not m:
        raise ValueError(f"Line not understood: {line}")
    op = m.group(1)
    lhs, rhs = map(str.strip, ineq_re.split(line, maxsplit=1)[0:3:2])
    lhs_expr = parse_sympy(lhs)
    rhs_expr = parse_sympy(rhs)
    return ("ineq", (lhs_expr, op, rhs_expr))

# --- read file ---
path = Path(INPUT_FILE)
if not path.exists():
    raise SystemExit(f"Cannot find '{INPUT_FILE}' in {Path.cwd()}")

raw_lines = path.read_text(encoding="utf-8").splitlines()
entries = []
for raw in raw_lines:
    s = clean_line(raw)
    if not s:
        continue
    entries.append(parse_fun_or_constraint(s))

functions = []
ineqs = []
eq_boundaries = []

for kind, payload in entries:
    if kind == "func":
        functions.append(payload)
    elif kind == "ineq":
        ineqs.append(payload)
    elif kind == "eq_boundary":
        eq_boundaries.append(payload)

# --- derive x/y bounds from simple constraints ---
xmin, xmax = DEFAULT_RANGE
ymin, ymax = DEFAULT_RANGE

def maybe_const(expr):
    free = expr.free_symbols
    return (len(free) == 0, float(sp.N(expr))) if len(free) == 0 else (False, None)

for lhs, op, rhs in ineqs:
    if lhs == x:
        is_c, c = maybe_const(rhs)
        if is_c:
            if op in ("<", "<="):
                xmax = min(xmax, c)
            elif op in (">", ">="):
                xmin = max(xmin, c)
    elif lhs == y:
        is_c, c = maybe_const(rhs)
        if is_c:
            if op in ("<", "<="):
                ymax = min(ymax, c)
            elif op in (">", ">="):
                ymin = max(ymin, c)
    elif rhs == x:
        is_c, c = maybe_const(lhs)
        if is_c:
            if op in (">", ">="):
                xmax = min(xmax, c)
            elif op in ("<", "<="):
                xmin = max(xmin, c)
    elif rhs == y:
        is_c, c = maybe_const(lhs)
        if is_c:
            if op in (">", ">="):
                ymax = min(ymax, c)
            elif op in ("<", "<="):
                ymin = max(ymin, c)

if xmin >= xmax:
    xmin, xmax = DEFAULT_RANGE
if ymin >= ymax:
    ymin, ymax = DEFAULT_RANGE

# --- prepare continuous grid ---
X = np.linspace(float(xmin), float(xmax), GRID_N)
Y = np.linspace(float(ymin), float(ymax), GRID_N)
XX, YY = np.meshgrid(X, Y, indexing='xy')

def lamb(expr):
    return sp.lambdify((x, y), expr, modules=["numpy"])

# Separate XY-only vs z-involving constraints
def xy_only(expr):
    return expr.free_symbols <= {x, y}

ineqs_xy = []
ineqs_z  = []
for lhs, op, rhs in ineqs:
    if xy_only(lhs) and xy_only(rhs):
        ineqs_xy.append((lhs, op, rhs))
    else:
        ineqs_z.append((lhs, op, rhs))

# Build feasible XY mask
feasible_xy = np.ones_like(XX, dtype=bool)
for lhs, op, rhs in ineqs_xy:
    fL = lamb(lhs)
    fR = lamb(rhs)
    L = fL(XX, YY)
    R = fR(XX, YY)
    if op == "<=":
        feasible_xy &= (L <= R)
    elif op == "<":
        feasible_xy &= (L < R)
    elif op == ">=":
        feasible_xy &= (L >= R)
    elif op == ">":
        feasible_xy &= (L > R)

# --- plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Draw boundary helper
def draw_boundary(lhs, rhs):
    try:
        sol = sp.solve(sp.Eq(lhs, rhs), z, dict=True)
        if sol:
            Zb = sp.lambdify((x, y), sol[0][z], modules=["numpy"])(XX, YY)
            ax.plot_surface(XX, YY, Zb, alpha=BOUNDARY_ALPHA, linewidth=0)
            return
    except Exception:
        pass
    return

for lhs, rhs in eq_boundaries:
    draw_boundary(lhs, rhs)

for i, fexpr in enumerate(functions, start=1):
    f_np = lamb(fexpr)
    ZZ = f_np(XX, YY)
    mask = feasible_xy.copy()
    for lhs, op, rhs in ineqs_z:
        L = sp.lambdify((x, y), lhs.subs(z, fexpr), modules=["numpy"])(XX, YY)
        R = sp.lambdify((x, y), rhs.subs(z, fexpr), modules=["numpy"])(XX, YY)
        if op == "<=":
            mask &= (L <= R)
        elif op == "<":
            mask &= (L < R)
        elif op == ">=":
            mask &= (L >= R)
        elif op == ">":
            mask &= (L > R)

    # Continuous surface
    Zmasked = np.ma.array(ZZ, mask=~mask)
    ax.plot_surface(XX, YY, Zmasked, cmap=COLORMAP, alpha=SURFACE_ALPHA, linewidth=0, antialiased=True)

    # --- Integer overlay ---
    xi = np.arange(int(np.floor(xmin)), int(np.ceil(xmax)) + 1)
    yi = np.arange(int(np.floor(ymin)), int(np.ceil(ymax)) + 1)
    XI, YI = np.meshgrid(xi, yi)
    ZI = f_np(XI, YI)

    mask_int = np.ones_like(XI, dtype=bool)
    for lhs, op, rhs in ineqs_z + ineqs_xy:
        L = sp.lambdify((x, y), lhs.subs(z, fexpr), modules=["numpy"])(XI, YI)
        R = sp.lambdify((x, y), rhs.subs(z, fexpr), modules=["numpy"])(XI, YI)
        if op == "<=":
            mask_int &= (L <= R)
        elif op == "<":
            mask_int &= (L < R)
        elif op == ">=":
            mask_int &= (L >= R)
        elif op == ">":
            mask_int &= (L > R)

    # scatter integer feasible points
    xs, ys, zs = XI[mask_int], YI[mask_int], ZI[mask_int]
    ax.scatter(xs, ys, zs, c="red", s=20, label="integer points" if i == 1 else "")

    # add labels for highest and lowest integer z
    if xs.size > 0:
        max_idx = np.argmax(zs)
        min_idx = np.argmin(zs)

        x_max, y_max, z_max = xs[max_idx], ys[max_idx], zs[max_idx]
        x_min, y_min, z_min = xs[min_idx], ys[min_idx], zs[min_idx]

        ax.text(x_max, y_max, z_max,
                f"max ({x_max}, {y_max}, {z_max:.2f})", color="black")
        ax.text(x_min, y_min, z_min,
                f"min ({x_min}, {y_min}, {z_min:.2f})", color="black")

ax.set_title("Functions with Constraints (from fun.plt)")
ax.legend()

# --- Rotation animation ---
def rotate(angle):
    ax.view_init(elev=30, azim=angle)

ani = animation.FuncAnimation(
    fig, rotate, frames=ROTATION_STEPS, interval=ROTATION_INTERVAL, blit=False
)

plt.show()

