"""
Nonlinear incremental pushover analysis of a pin-pin timber frame.
Frame: 96x96 in, PSL columns/beams + PSL truss braces with dowel connections.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ---------------------------------------------------------------------------
# Constants – frame geometry and material
# ---------------------------------------------------------------------------
NODES = np.array([
    [0,   0],   # 0 base-left  (pin)
    [96,  0],   # 1 base-right (pin)
    [0,  72],   # 2 left column @ brace
    [96, 72],   # 3 right column @ brace
    [24, 96],   # 4 left beam @ brace
    [72, 96],   # 5 right beam @ brace
    [0,  96],   # 6 top-left beam-column joint
    [96, 96],   # 7 top-right beam-column joint
], dtype=float)

N_NODES = len(NODES)
N_DOF   = N_NODES * 3           # 24 total DOFs (u, v, θ per node)

# PSL frame member properties
E_FRAME = 2_000_000.0           # psi
b_col   = 5.25;  d_col = 9.5
A_FRAME = b_col * d_col         # 49.875 in²
I_FRAME = b_col * d_col**3 / 12 # 375.8 in⁴

# PSL truss brace properties
E_TRUSS = 2_000_000.0
b_br    = 2.0;   d_br = 8.0
A_TRUSS = b_br * d_br           # 16 in²

# Frame elements: list of (nodeI, nodeJ, E, A, I, is_truss)
FRAME_ELEMS = [
    (0, 2, E_FRAME, A_FRAME, I_FRAME, False),   # left col lower
    (2, 6, E_FRAME, A_FRAME, I_FRAME, False),   # left col upper
    (1, 3, E_FRAME, A_FRAME, I_FRAME, False),   # right col lower
    (3, 7, E_FRAME, A_FRAME, I_FRAME, False),   # right col upper
    (6, 4, E_FRAME, A_FRAME, I_FRAME, False),   # beam left
    (4, 5, E_FRAME, A_FRAME, I_FRAME, False),   # beam middle
    (5, 7, E_FRAME, A_FRAME, I_FRAME, False),   # beam right
    (2, 4, E_TRUSS, A_TRUSS, 0.0,    True ),    # left brace
    (3, 5, E_TRUSS, A_TRUSS, 0.0,    True ),    # right brace
]

# Brace end-nodes for δ_brace recovery (i, j pairs)
BRACE_PAIRS = [(2, 4), (3, 5)]

# Nodes where dowel springs are added
DOWEL_NODES = [2, 3, 4, 5]

# Left brace angle: from node 2(0,72) to node 4(24,96)
# dx=24, dy=24 → 45°; right brace: node 3(96,72) to node 5(72,96) → dx=-24, dy=24
# Spring oriented along brace axis for all four nodes
def brace_angle(ni, nj):
    dx = NODES[nj][0] - NODES[ni][0]
    dy = NODES[nj][1] - NODES[ni][1]
    L  = np.hypot(dx, dy)
    return dx / L, dy / L          # cos, sin

# Fixed DOF indices: pin at nodes 0 and 1 → fix u,v (DOFs 0,1,3,4)
FIXED_DOFS = [0, 1, 3, 4]
FREE_DOFS  = [i for i in range(N_DOF) if i not in FIXED_DOFS]


# ---------------------------------------------------------------------------
# Euler-Bernoulli frame element stiffness (local 6x6)
# ---------------------------------------------------------------------------
def frame_local_k(E, A, I, L):
    EAL  = E * A / L
    EI   = E * I
    k = np.zeros((6, 6))
    k[0, 0] =  EAL;  k[0, 3] = -EAL
    k[3, 0] = -EAL;  k[3, 3] =  EAL
    k[1, 1] =  12*EI/L**3;  k[1, 2] =  6*EI/L**2
    k[1, 4] = -12*EI/L**3;  k[1, 5] =  6*EI/L**2
    k[2, 1] =   6*EI/L**2;  k[2, 2] =  4*EI/L
    k[2, 4] =  -6*EI/L**2;  k[2, 5] =  2*EI/L
    k[4, 1] = -12*EI/L**3;  k[4, 2] = -6*EI/L**2
    k[4, 4] =  12*EI/L**3;  k[4, 5] = -6*EI/L**2
    k[5, 1] =   6*EI/L**2;  k[5, 2] =  2*EI/L
    k[5, 4] =  -6*EI/L**2;  k[5, 5] =  4*EI/L
    return k


def rotation_matrix_6dof(c, s):
    T = np.zeros((6, 6))
    T[0, 0] =  c;  T[0, 1] = s
    T[1, 0] = -s;  T[1, 1] = c
    T[2, 2] =  1.0
    T[3, 3] =  c;  T[3, 4] = s
    T[4, 3] = -s;  T[4, 4] = c
    T[5, 5] =  1.0
    return T


def rotation_matrix_4dof(c, s):
    """Rotation for truss element (u,v at each end, 4 DOFs)."""
    T = np.zeros((4, 4))
    T[0, 0] =  c;  T[0, 1] = s
    T[1, 0] = -s;  T[1, 1] = c
    T[2, 2] =  c;  T[2, 3] = s
    T[3, 2] = -s;  T[3, 3] = c
    return T


def truss_local_k(E, A, L):
    k = E * A / L * np.array([[ 1, 0, -1, 0],
                               [ 0, 0,  0, 0],
                               [-1, 0,  1, 0],
                               [ 0, 0,  0, 0]], dtype=float)
    return k


def elem_dof_indices(ni, nj, is_truss=False):
    """Global DOF indices for element nodes."""
    if is_truss:
        return [3*ni, 3*ni+1, 3*nj, 3*nj+1]
    else:
        return [3*ni, 3*ni+1, 3*ni+2, 3*nj, 3*nj+1, 3*nj+2]


def assemble_frame_k():
    """Assemble structural stiffness without dowel springs."""
    K = np.zeros((N_DOF, N_DOF))
    for (ni, nj, E, A, I, is_truss) in FRAME_ELEMS:
        xi, yi = NODES[ni]
        xj, yj = NODES[nj]
        dx = xj - xi;  dy = yj - yi
        L  = np.hypot(dx, dy)
        c  = dx / L;   s  = dy / L
        dofs = elem_dof_indices(ni, nj, is_truss)
        if is_truss:
            k_loc = truss_local_k(E, A, L)
            T     = rotation_matrix_4dof(c, s)
            k_glob = T.T @ k_loc @ T
        else:
            k_loc  = frame_local_k(E, A, I, L)
            T      = rotation_matrix_6dof(c, s)
            k_glob = T.T @ k_loc @ T
        for ii, gi in enumerate(dofs):
            for jj, gj in enumerate(dofs):
                K[gi, gj] += k_glob[ii, jj]
    return K


# ---------------------------------------------------------------------------
# Dowel spring assembly
# ---------------------------------------------------------------------------
def add_dowel_springs(K, k_dowel_left, k_dowel_right, n_dowels):
    """
    Add nodal dowel springs at DOWEL_NODES.
    Left brace (2→4): springs at nodes 2 and 4.
    Right brace (3→5): springs at nodes 3 and 5.
    Spring axes along respective brace directions.
    k_eff = n_dowels * k_dowel (springs in parallel, one per brace end)
    """
    K = K.copy()
    brace_data = [
        (BRACE_PAIRS[0], k_dowel_left,  [2, 4]),
        (BRACE_PAIRS[1], k_dowel_right, [3, 5]),
    ]
    for (ni, nj), k_d, spring_nodes in brace_data:
        c, s = brace_angle(ni, nj)
        k_eff = n_dowels * k_d
        # 2x2 spring stiffness contribution in global coords
        ks = k_eff * np.array([[c*c, c*s],
                                [c*s, s*s]])
        for node in spring_nodes:
            u_dof = 3 * node
            v_dof = 3 * node + 1
            dofs  = [u_dof, v_dof]
            for ii, gi in enumerate(dofs):
                for jj, gj in enumerate(dofs):
                    K[gi, gj] += ks[ii, jj]
    return K


# ---------------------------------------------------------------------------
# Spline fitting (Fritsch-Carlson monotone cubic)
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_fit_spline(csv_path, k_extrap, force_jump_limit=300.0):
    df = pd.read_csv(csv_path, header=0, names=["disp", "force"])
    disp  = df["disp"].values.astype(float)
    force = df["force"].values.astype(float)

    # Filter out large force jumps between consecutive points
    keep = [True]
    for i in range(1, len(force)):
        keep.append(abs(force[i] - force[i-1]) <= force_jump_limit)
    disp  = disp[keep]
    force = force[keep]

    # PchipInterpolator implements Fritsch-Carlson monotone cubic spline
    spline    = PchipInterpolator(disp, force)
    spline_d1 = spline.derivative()          # tangent stiffness function

    x_max     = disp[-1]                     # 0.607 in
    f_max     = float(spline(x_max))
    k_last    = float(spline_d1(x_max))

    return disp, force, spline, spline_d1, x_max, f_max, k_last, k_extrap


def get_tangent_stiffness(delta, spline_d1, x_max, k_extrap):
    """Return tangent stiffness at given displacement."""
    delta = abs(delta)
    if delta <= x_max:
        return max(float(spline_d1(delta)), 0.0)
    else:
        return k_extrap


# ---------------------------------------------------------------------------
# δ_brace recovery
# ---------------------------------------------------------------------------
def recover_delta_brace(U, ni, nj):
    """Axial relative displacement between brace end nodes."""
    xi, yi = NODES[ni]
    xj, yj = NODES[nj]
    dx = xj - xi;  dy = yj - yi
    L  = np.hypot(dx, dy)
    c  = dx / L;   s  = dy / L
    ui = U[3*ni];  vi = U[3*ni+1]
    uj = U[3*nj];  vj = U[3*nj+1]
    return (uj - ui)*c + (vj - vi)*s


# ---------------------------------------------------------------------------
# Pushover analysis
# ---------------------------------------------------------------------------
def run_pushover(delta_P, n_steps, k_extrap, n_dowels, spline_d1, x_max):
    """
    Incremental tangent stiffness pushover.
    Returns arrays: drift (in), base_shear (lbf), delta_dowel_history.
    """
    U           = np.zeros(N_DOF)
    drift_hist  = [0.0]
    shear_hist  = [0.0]
    ddow_hist   = [0.0]

    # Initial dowel stiffness (at zero displacement)
    k_d_left  = get_tangent_stiffness(0.0, spline_d1, x_max, k_extrap)
    k_d_right = get_tangent_stiffness(0.0, spline_d1, x_max, k_extrap)

    # Load vector: ΔP/2 horizontal at nodes 6 and 7
    dP_vec = np.zeros(N_DOF)
    dP_vec[3*6]   = delta_P / 2.0
    dP_vec[3*7]   = delta_P / 2.0

    K_struct = assemble_frame_k()

    for _ in range(n_steps):
        # Assemble tangent stiffness with current dowel springs
        K_t = add_dowel_springs(K_struct, k_d_left, k_d_right, n_dowels)

        # Partition to free DOFs
        K_free = K_t[np.ix_(FREE_DOFS, FREE_DOFS)]
        dP_free = dP_vec[FREE_DOFS]

        # Solve
        try:
            dU_free = np.linalg.solve(K_free, dP_free)
        except np.linalg.LinAlgError:
            break

        # Update full displacement vector
        dU = np.zeros(N_DOF)
        for i, dof in enumerate(FREE_DOFS):
            dU[dof] = dU_free[i]
        U += dU

        # Recover brace deformations
        db_left  = recover_delta_brace(U, *BRACE_PAIRS[0])
        db_right = recover_delta_brace(U, *BRACE_PAIRS[1])

        # δ_dowel = δ_brace / 2
        dd_left  = abs(db_left)  / 2.0
        dd_right = abs(db_right) / 2.0
        dd_max   = max(dd_left, dd_right)

        # Update dowel tangent stiffness for next step
        k_d_left  = get_tangent_stiffness(dd_left,  spline_d1, x_max, k_extrap)
        k_d_right = get_tangent_stiffness(dd_right, spline_d1, x_max, k_extrap)

        total_P   = delta_P * (_ + 1)
        drift     = U[3*6]           # horizontal displacement at node 6

        drift_hist.append(drift)
        shear_hist.append(total_P)
        ddow_hist.append(dd_max)

    return np.array(drift_hist), np.array(shear_hist), np.array(ddow_hist)


# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Timber Frame Pushover", layout="wide")
    st.title("Nonlinear Pushover Analysis — Pin-Pin Timber Frame")

    # Sidebar
    st.sidebar.header("Analysis Parameters")
    delta_P  = st.sidebar.slider("ΔP load increment (lbf)", 10,  200, 50,  step=10)
    n_steps  = st.sidebar.slider("Number of steps",         10,  100, 60,  step=5)
    k_extrap = st.sidebar.slider("k_extrap beyond 0.607 in (lbf/in)", 0, 2000, 200, step=50)
    n_dowels = st.sidebar.slider("Dowels per brace end",     1,    6,   1,  step=1)

    # Load CSV and fit spline
    csv_path = os.path.join(os.path.dirname(__file__), "SSTubeandWood1.csv")
    disp_raw, force_raw, spline, spline_d1, x_max, f_max, k_last, _ = \
        load_and_fit_spline(csv_path, k_extrap)

    # Re-fit with current k_extrap (k_extrap is a slider that changes)
    # (cache key includes k_extrap so this is handled by cache)

    # Run pushover
    drift, base_shear, dd_hist = run_pushover(
        delta_P, n_steps, k_extrap, n_dowels, spline_d1, x_max
    )

    # Metrics
    peak_shear   = float(np.max(base_shear))
    peak_idx     = int(np.argmax(base_shear))
    drift_at_peak = float(drift[peak_idx])
    # Initial stiffness: slope of first nonzero step
    if len(drift) > 1 and drift[1] != 0:
        K_init = base_shear[1] / drift[1]
    else:
        K_init = 0.0
    max_dd       = float(np.max(dd_hist))

    # Spline evaluation for plot
    x_fit   = np.linspace(0, x_max, 300)
    y_fit   = spline(x_fit)
    x_ext   = np.array([x_max, x_max + 0.3])
    y_ext   = np.array([f_max, f_max + k_extrap * 0.3])

    # Test envelope
    env_disp  = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5]
    env_shear = [0,   500, 750, 1150, 1550, 1850, 2000]

    # ---- Build figure ----
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pushover Curve", "Dowel Backbone"),
        horizontal_spacing=0.10
    )

    # --- Left: Pushover curve ---
    fig.add_trace(go.Scatter(
        x=drift, y=base_shear,
        mode="lines", name="Model",
        line=dict(color="royalblue", width=2.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=env_disp, y=env_shear,
        mode="lines+markers", name="Test envelope",
        line=dict(color="darkorange", dash="dash", width=2),
        marker=dict(size=7, symbol="circle-open")
    ), row=1, col=1)

    # Peak annotation
    fig.add_annotation(
        x=drift_at_peak, y=peak_shear,
        text=f"Peak: {peak_shear:.0f} lbf<br>Drift: {drift_at_peak:.3f} in",
        showarrow=True, arrowhead=2, arrowcolor="royalblue",
        bgcolor="white", bordercolor="royalblue", font=dict(size=11),
        row=1, col=1
    )

    fig.update_xaxes(title_text="Horizontal drift at node 6 (in)", row=1, col=1)
    fig.update_yaxes(title_text="Base shear (lbf)", row=1, col=1)

    # --- Right: Dowel backbone ---
    fig.add_trace(go.Scatter(
        x=disp_raw, y=force_raw,
        mode="markers", name="CSV data",
        marker=dict(color="gray", size=6, opacity=0.7)
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode="lines", name="Spline fit",
        line=dict(color="crimson", width=2.5)
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x_ext, y=y_ext,
        mode="lines", name="Extrapolation",
        line=dict(color="darkorange", dash="dash", width=2)
    ), row=1, col=2)

    # Vertical line: max δ_dowel reached
    fig.add_vline(
        x=max_dd,
        line=dict(color="navy", dash="dot", width=1.5),
        annotation_text=f"max δ_d = {max_dd:.3f} in",
        annotation_position="top right",
        row=1, col=2
    )

    fig.update_xaxes(title_text="Dowel displacement (in)", row=1, col=2)
    fig.update_yaxes(title_text="Dowel force (lbf)", row=1, col=2)

    fig.update_layout(
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics row
    warn = " ⚠️ exceeds tested range" if max_dd > x_max else ""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Peak base shear",       f"{peak_shear:.0f} lbf")
    col2.metric("Drift at peak",         f"{drift_at_peak:.3f} in")
    col3.metric("Initial stiffness K_T", f"{K_init:.0f} lbf/in")
    col4.metric("Max δ_dowel" + warn,    f"{max_dd:.4f} in")


if __name__ == "__main__":
    main()
