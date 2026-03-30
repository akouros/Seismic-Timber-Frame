"""
Nonlinear incremental pushover analysis of a pin-pin timber frame.

Braces are modelled as nonlinear truss elements whose axial stiffness is
updated each step based on the dowel backbone curve:

    k_brace = n_dowels * spline.deriv(delta_brace / 2) / 2

This replaces EA/L in the standard truss stiffness matrix — no separate
elastic brace member or nodal spring nodes are needed.

Compression brace buckling: when delta_brace < 0 the brace stiffness is
set to a small residual (1 lbf/in) so only the tension brace carries load.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ---------------------------------------------------------------------------
# Frame geometry
# ---------------------------------------------------------------------------
NODES = np.array([
    [0,   0],   # 0  base-left  (pin)
    [96,  0],   # 1  base-right (pin)
    [0,  72],   # 2  left column  @ brace
    [96, 72],   # 3  right column @ brace
    [24, 96],   # 4  left beam    @ brace
    [72, 96],   # 5  right beam   @ brace
    [0,  96],   # 6  top-left  beam-column joint
    [96, 96],   # 7  top-right beam-column joint
], dtype=float)

N_DOF = len(NODES) * 3          # 24 (u, v, theta per node)

# PSL frame member properties
E_FRAME = 2_000_000.0           # psi
A_FRAME = 5.25 * 9.5            # 49.875 in^2
I_FRAME = 5.25 * 9.5**3 / 12   # 375.8  in^4

# Frame elements only — braces are handled separately as nonlinear trusses
FRAME_ELEMS = [
    (0, 2),   # left col  lower
    (2, 6),   # left col  upper
    (1, 3),   # right col lower
    (3, 7),   # right col upper
    (6, 4),   # beam left
    (4, 5),   # beam middle
    (5, 7),   # beam right
]

# Brace node pairs
BRACE_PAIRS = [(2, 4), (3, 5)]

# Boundary conditions: pin at nodes 0 and 1 (fix u, v; theta free)
FIXED_DOFS = [0, 1, 3, 4]
FREE_DOFS  = [i for i in range(N_DOF) if i not in FIXED_DOFS]


# ---------------------------------------------------------------------------
# Euler-Bernoulli frame element stiffness (local 6x6, global assembly)
# ---------------------------------------------------------------------------
def _frame_local_k(E, A, I, L):
    EAL = E * A / L
    EI  = E * I
    k   = np.zeros((6, 6))
    # axial
    k[0, 0] =  EAL;  k[0, 3] = -EAL
    k[3, 0] = -EAL;  k[3, 3] =  EAL
    # bending
    k[1, 1] =  12*EI/L**3;  k[1, 2] =  6*EI/L**2
    k[1, 4] = -12*EI/L**3;  k[1, 5] =  6*EI/L**2
    k[2, 1] =   6*EI/L**2;  k[2, 2] =  4*EI/L
    k[2, 4] =  -6*EI/L**2;  k[2, 5] =  2*EI/L
    k[4, 1] = -12*EI/L**3;  k[4, 2] = -6*EI/L**2
    k[4, 4] =  12*EI/L**3;  k[4, 5] = -6*EI/L**2
    k[5, 1] =   6*EI/L**2;  k[5, 2] =  2*EI/L
    k[5, 4] =  -6*EI/L**2;  k[5, 5] =  4*EI/L
    return k


def _T6(c, s):
    T = np.zeros((6, 6))
    T[0, 0] =  c;  T[0, 1] = s
    T[1, 0] = -s;  T[1, 1] = c
    T[2, 2] =  1.0
    T[3, 3] =  c;  T[3, 4] = s
    T[4, 3] = -s;  T[4, 4] = c
    T[5, 5] =  1.0
    return T


def assemble_frame_k():
    """Assemble global stiffness from frame elements (columns + beam) only."""
    K = np.zeros((N_DOF, N_DOF))
    for (ni, nj) in FRAME_ELEMS:
        dx = NODES[nj][0] - NODES[ni][0]
        dy = NODES[nj][1] - NODES[ni][1]
        L  = np.hypot(dx, dy)
        c  = dx / L;  s = dy / L
        dofs = [3*ni, 3*ni+1, 3*ni+2, 3*nj, 3*nj+1, 3*nj+2]
        T    = _T6(c, s)
        k_g  = T.T @ _frame_local_k(E_FRAME, A_FRAME, I_FRAME, L) @ T
        for ii, gi in enumerate(dofs):
            for jj, gj in enumerate(dofs):
                K[gi, gj] += k_g[ii, jj]
    return K


# ---------------------------------------------------------------------------
# Nonlinear brace truss elements
# ---------------------------------------------------------------------------
def assemble_nonlinear_braces(k_brace_left, k_brace_right):
    """
    Build the 24x24 contribution from both nonlinear brace truss elements.
    k_brace replaces EA/L in the standard truss stiffness matrix:

        K_global = k_brace * [[c^2,  cs, -c^2, -cs ],
                               [cs,  s^2, -cs,  -s^2],
                               [-c^2,-cs,  c^2,  cs ],
                               [-cs, -s^2, cs,   s^2]]
    """
    K = np.zeros((N_DOF, N_DOF))
    for k_br, (ni, nj) in zip([k_brace_left, k_brace_right], BRACE_PAIRS):
        dx = NODES[nj][0] - NODES[ni][0]
        dy = NODES[nj][1] - NODES[ni][1]
        L  = np.hypot(dx, dy)
        c  = dx / L;  s = dy / L
        dofs = [3*ni, 3*ni+1, 3*nj, 3*nj+1]
        kg = k_br * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s],
        ])
        for ii, gi in enumerate(dofs):
            for jj, gj in enumerate(dofs):
                K[gi, gj] += kg[ii, jj]
    return K


# ---------------------------------------------------------------------------
# Brace axial deformation recovery
# ---------------------------------------------------------------------------
def recover_delta_brace(U, ni, nj):
    """
    Signed axial deformation of the brace along its axis.
    Positive = elongation (tension), negative = shortening (compression).
    """
    dx = NODES[nj][0] - NODES[ni][0]
    dy = NODES[nj][1] - NODES[ni][1]
    L  = np.hypot(dx, dy)
    c  = dx / L;  s = dy / L
    return (U[3*nj] - U[3*ni])*c + (U[3*nj+1] - U[3*ni+1])*s


# ---------------------------------------------------------------------------
# Dowel backbone spline (Fritsch-Carlson via PchipInterpolator)
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_fit_spline(csv_path, force_jump_limit=300.0):
    """
    Load CSV, filter jumps > force_jump_limit, fit PchipInterpolator.
    Returns raw data arrays, spline, derivative spline, and x_max (0.607 in).
    """
    df    = pd.read_csv(csv_path, header=0, names=["disp", "force"])
    disp  = df["disp"].values.astype(float)
    force = df["force"].values.astype(float)

    # Filter points where force jumps more than the limit
    keep = [True]
    for i in range(1, len(force)):
        keep.append(abs(force[i] - force[i-1]) <= force_jump_limit)
    disp  = disp[keep]
    force = force[keep]

    spline    = PchipInterpolator(disp, force)
    spline_d1 = spline.derivative()

    x_max = float(disp[-1])       # tested range limit (0.607 in)
    f_max = float(spline(x_max))

    return disp, force, spline, spline_d1, x_max, f_max


def get_k_dowel(delta_dowel, spline_d1, x_max, k_extrap):
    """
    Tangent stiffness of a single dowel at displacement delta_dowel (>= 0).
    Extrapolates linearly beyond x_max.
    """
    delta_dowel = abs(delta_dowel)
    if delta_dowel <= x_max:
        return max(float(spline_d1(delta_dowel)), 0.0)
    return float(k_extrap)


def compute_k_brace(delta_brace, n_dowels, spline_d1, x_max, k_extrap):
    """
    Nonlinear brace axial stiffness.

    Two end-connections in series, each with n_dowels dowels in parallel:
        k_end  = n_dowels * k_dowel(delta_brace / 2)
        k_brace = k_end / 2   (two k_end springs in series)

    Compression (delta_brace < 0): brace assumed buckled, residual = 1 lbf/in.
    """
    if delta_brace < 0.0:
        return 1.0                      # buckled — residual only
    delta_dowel = delta_brace / 2.0
    k_d = get_k_dowel(delta_dowel, spline_d1, x_max, k_extrap)
    return n_dowels * k_d / 2.0


# ---------------------------------------------------------------------------
# Pushover analysis
# ---------------------------------------------------------------------------
def run_pushover(delta_P, n_steps, k_extrap, n_dowels, spline_d1, x_max):
    """
    Incremental tangent stiffness pushover.
    Returns arrays: drift (in), base_shear (lbf), delta_dowel_history (in).
    Prints step-1 diagnostics to console.
    """
    U          = np.zeros(N_DOF)
    drift_hist = [0.0]
    shear_hist = [0.0]
    ddow_hist  = [0.0]

    # Initial brace stiffnesses (delta_brace = 0 at start)
    k_bl = compute_k_brace(0.0, n_dowels, spline_d1, x_max, k_extrap)
    k_br = compute_k_brace(0.0, n_dowels, spline_d1, x_max, k_extrap)

    # Incremental load vector: dP/2 at node 6, dP/2 at node 7
    dP_vec          = np.zeros(N_DOF)
    dP_vec[3*6]     = delta_P / 2.0
    dP_vec[3*7]     = delta_P / 2.0
    dP_free         = dP_vec[FREE_DOFS]

    K_frame = assemble_frame_k()    # constant — assemble once

    for step in range(n_steps):

        # --- Assemble tangent stiffness ---
        K_t    = K_frame + assemble_nonlinear_braces(k_bl, k_br)
        K_free = K_t[np.ix_(FREE_DOFS, FREE_DOFS)]

        # --- Solve ---
        try:
            dU_free = np.linalg.solve(K_free, dP_free)
        except np.linalg.LinAlgError:
            break

        dU = np.zeros(N_DOF)
        for i, dof in enumerate(FREE_DOFS):
            dU[dof] = dU_free[i]
        U += dU

        # --- Recover brace deformations ---
        db_left  = recover_delta_brace(U, *BRACE_PAIRS[0])
        db_right = recover_delta_brace(U, *BRACE_PAIRS[1])

        # delta_dowel (positive, for backbone plot)
        dd_left  = abs(db_left)  / 2.0
        dd_right = abs(db_right) / 2.0
        dd_max   = max(dd_left, dd_right)

        # --- Update k_brace for next step ---
        k_bl = compute_k_brace(db_left,  n_dowels, spline_d1, x_max, k_extrap)
        k_br = compute_k_brace(db_right, n_dowels, spline_d1, x_max, k_extrap)

        total_P = delta_P * (step + 1)
        drift   = float(U[3*6])

        # --- Step 1 console diagnostic ---
        if step == 0:
            k_lat = total_P / drift if abs(drift) > 1e-14 else float("inf")
            print("\n=== STEP 1 DIAGNOSTIC ===")
            print("delta_brace  left  (2->4) : %+.6f in" % db_left)
            print("delta_brace  right (3->5) : %+.6f in" % db_right)
            print("delta_dowel  left         : %.6f in  (= |delta_brace| / 2)" % dd_left)
            print("delta_dowel  right        : %.6f in" % dd_right)
            print("k_brace      left         : %.4f lbf/in  (used next step)" % k_bl)
            print("k_brace      right        : %.4f lbf/in  (buckled=1)" % k_br)
            print("Drift u_node6             : %.6f in" % drift)
            print("System K_lateral (dP/drift): %.2f lbf/in" % k_lat)
            print("=========================\n")

        drift_hist.append(drift)
        shear_hist.append(total_P)
        ddow_hist.append(dd_max)

    return np.array(drift_hist), np.array(shear_hist), np.array(ddow_hist)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Timber Frame Pushover", layout="wide")
    st.title("Nonlinear Pushover Analysis — Pin-Pin Timber Frame")

    # --- Sidebar ---
    st.sidebar.header("Analysis Parameters")
    delta_P  = st.sidebar.slider("dP load increment (lbf)",          10,  200,   50, step=10)
    n_steps  = st.sidebar.slider("Number of steps",                   10,  150,   80, step=5)
    k_extrap = st.sidebar.slider("k_extrap beyond 0.607 in (lbf/in)", 0, 2000,  200, step=50)
    n_dowels = st.sidebar.slider("Dowels per brace end",               1,    6,    1, step=1)

    # --- Load backbone data and fit spline ---
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "SSTubeandWood1.csv")
    disp_raw, force_raw, spline, spline_d1, x_max, f_max = \
        load_and_fit_spline(csv_path)

    # --- Run pushover ---
    drift, base_shear, dd_hist = run_pushover(
        delta_P, n_steps, k_extrap, n_dowels, spline_d1, x_max
    )

    # --- Metrics ---
    peak_shear    = float(np.max(base_shear))
    peak_idx      = int(np.argmax(base_shear))
    drift_at_peak = float(drift[peak_idx])
    K_init = (base_shear[1] / drift[1]) if (len(drift) > 1 and abs(drift[1]) > 1e-14) else 0.0
    max_dd = float(np.max(dd_hist))

    # Spline curve for plot
    x_fit = np.linspace(0, x_max, 300)
    y_fit = spline(x_fit)
    x_ext = np.array([x_max, x_max + 0.35])
    y_ext = np.array([f_max, f_max + k_extrap * 0.35])

    # Test envelope
    env_d = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5]
    env_v = [0,   500, 750, 1150, 1550, 1850, 2000]

    # --- Build Plotly figure ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pushover Curve", "Dowel Backbone"),
        horizontal_spacing=0.10,
    )

    # Left: pushover curve
    fig.add_trace(go.Scatter(
        x=drift, y=base_shear,
        mode="lines", name="Model",
        line=dict(color="royalblue", width=2.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=env_d, y=env_v,
        mode="lines+markers", name="Test envelope",
        line=dict(color="darkorange", dash="dash", width=2),
        marker=dict(size=7, symbol="circle-open"),
    ), row=1, col=1)

    fig.add_annotation(
        x=drift_at_peak, y=peak_shear,
        text="Peak: %.0f lbf<br>Drift: %.3f in" % (peak_shear, drift_at_peak),
        showarrow=True, arrowhead=2, arrowcolor="royalblue",
        bgcolor="white", bordercolor="royalblue", font=dict(size=11),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Horizontal drift at node 6 (in)", row=1, col=1)
    fig.update_yaxes(title_text="Base shear (lbf)",                row=1, col=1)

    # Right: dowel backbone
    fig.add_trace(go.Scatter(
        x=disp_raw, y=force_raw,
        mode="markers", name="CSV data",
        marker=dict(color="gray", size=6, opacity=0.7),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode="lines", name="Spline fit",
        line=dict(color="crimson", width=2.5),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x_ext, y=y_ext,
        mode="lines", name="Extrapolation",
        line=dict(color="darkorange", dash="dash", width=2),
    ), row=1, col=2)

    fig.add_vline(
        x=max_dd,
        line=dict(color="navy", dash="dot", width=1.5),
        annotation_text="max dd = %.3f in" % max_dd,
        annotation_position="top right",
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Dowel displacement (in)", row=1, col=2)
    fig.update_yaxes(title_text="Dowel force (lbf)",       row=1, col=2)

    fig.update_layout(
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Metrics row ---
    warn = "  *** exceeds tested range" if max_dd > x_max else ""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak base shear",       "%.0f lbf"   % peak_shear)
    c2.metric("Drift at peak",         "%.3f in"    % drift_at_peak)
    c3.metric("Initial stiffness K_T", "%.0f lbf/in" % K_init)
    c4.metric("Max delta_dowel" + warn, "%.4f in"   % max_dd)


if __name__ == "__main__":
    main()
