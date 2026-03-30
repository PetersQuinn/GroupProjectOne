import numpy as np
from matplotlib import pyplot as plt
from types import SimpleNamespace
from multivarious.opt import sqp
from multivarious.rvs import lognormal, beta
from multivarious.utl import opt_options, plot_cvg_hst, format_plot, plot_ECDF_ci

def H(X):
    """
    Heaviside function applied element-wise to matrix X.
    Returns 1 where X > 0, 0 otherwise.
    """
    HX = (X > 0).astype(float)
    return HX

def power_grid_analysis(v, C):
    """
    Analysis function for the 12-node power grid with transmission losses.

    The 24 design variables v are the directed power flows P_{i,j} on each
    branch. The 14 inequality constraints enforce:
      - Generator output <= capacity  (rows 0-1)
      - Power delivered >= demand     (rows 2-13, negated so all are <=)

    Transmission losses: power arriving at the destination node is
      P_{i,j} * [1 - epsilon * H(P_{i,j})]
    i.e. the loss is incurred by whichever end is the *receiving* end.

    In matrix form: A * diag[1 - eps * H(-L)] * v <= b
    where L = A @ diag(v), and H(-L) flags outgoing flows (A entry = +1 side).

    Parameters
    ----------
    v : array-like (24,)
        Power flows [PA1, PA2, PA3, PB7, PB8, PB9,
                     P12, P1_11, P1_12, P23, P2_12,
                     P34, P45, P46, P56, P57, P67, P68,
                     P78, P89, P9_10, P10_11, P10_12, P11_12]
    C : SimpleNamespace
        Constants: C.A (14x24), C.b (14,), C.c (24,), C.loss (scalar epsilon)

    Returns
    -------
    f : float
        Total generation cost  f = c^T v
    g : array (14,)
        Constraint residuals  g = A*[1-eps*H(-L)]*v - b  (feasible when <= 0)
    """
    A      = C.A      # 14x24 constraint matrix (negated demand rows, entries in {-1,0,+1})
    A_orig = C.A_orig # 14x24 un-negated matrix — physical signs for loss computation
    b      = C.b      # 14-vector of RHS values
    c      = C.c      # 24-vector of cost coefficients
    loss   = C.loss   # scalar transmission loss factor epsilon
    A_orig = C.A_orig  # 14x24 un-negated matrix — physical signs for loss computation
    L = A_orig @ np.diag(v)                        # 14x24, use original signs for loss
    A_scaled = C.A * (1 - loss * H(-L))            # apply scaled A (with negated demand rows)
    f = c @ v
    g = A_scaled @ v - b

    return f, g

# ==============================================================================
# where i actually solve the problem and do the Monte Carlo risk analysis
# ==============================================================================

# --- Problem constants --------------------------------------------------------
cA   = 20                                      # cost rate at generator A, $/MW
cB   = 12                                      # cost rate at generator B, $/MW
D    = np.array([10, 15, 20, 10, 20, 50,
                 80, 30, 10,  8, 12, 11])      # nominal demand at nodes 1-12, MW
G    = np.array([110, 300])                    # max generation capacity [PA, PB], MW
T    = 90                                      # max transmission capacity per line, MW
loss = 0.05                                    # transmission loss factor epsilon

C = SimpleNamespace()

# 14x24 constraint matrix A (entries: 0, +1, -1)
# Rows:  0=NodeA, 1=NodeB, 2=Node1 ... 13=Node12
# Cols:  0=PA1, 1=PA2, 2=PA3, 3=PB7, 4=PB8, 5=PB9,
#        6=P12, 7=P1_11, 8=P1_12, 9=P23, 10=P2_12,
#        11=P34, 12=P45, 13=P46, 14=P56, 15=P57,
#        16=P67, 17=P68, 18=P78, 19=P89,
#        20=P9_10, 21=P10_11, 22=P10_12, 23=P11_12
#
# Sign convention:
#   +1 at a node means that variable's power LEAVES that node
#   -1 at a node means that variable's power ARRIVES at that node
C.A = np.array([
    # Row 0: Node A — all flows leave A, total <= G_A
    [  1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    # Row 1: Node B — all flows leave B, total <= G_B
    [  0,   0,   0,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    # Row 2: Node 1 — negate demand constraint
    # net in = PA1_arrival - P12_leaving - P1_11_leaving - P1_12_leaving >= D1
    # negated: -PA1_arrival + P12_leaving + P1_11_leaving + P1_12_leaving <= -D1
    [ -1,   0,   0,   0,   0,   0,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    # Row 3: Node 2 — connects to: A(arrives PA2), 1(P12 arrives), 3(P23 leaves), 12(P2_12 leaves)
    # net in = PA2 + P12 - P23 - P2_12 >= D2
    # negated: -PA2 - P12 + P23 + P2_12 <= -D2
    [  0,  -1,   0,   0,   0,   0,  -1,   0,   0,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    # Row 4: Node 3 — connects to: A(arrives PA3), 2(P23 arrives), 4(P34 leaves)
    # net in = PA3 + P23 - P34 >= D3
    # negated: -PA3 - P23 + P34 <= -D3
    [  0,   0,  -1,   0,   0,   0,   0,   0,   0,  -1,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    # Row 5: Node 4 — connects to: 3(P34 arrives), 5(P45 leaves), 6(P46 leaves)
    # net in = P34 - P45 - P46 >= D4
    # negated: -P34 + P45 + P46 <= -D4
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    # Row 6: Node 5 — connects to: 4(P45 arrives), 6(P56 leaves), 7(P57 leaves)
    # net in = P45 - P56 - P57 >= D5
    # negated: -P45 + P56 + P57 <= -D5
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,   0,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0],
    # Row 7: Node 6 — connects to: 4(P46 arrives), 5(P56 arrives), 7(P67 leaves), 8(P68 leaves)
    # net in = P46 + P56 - P67 - P68 >= D6
    # negated: -P46 - P56 + P67 + P68 <= -D6
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,   0,   1,   1,   0,   0,   0,   0,   0,   0],
    # Row 8: Node 7 — connects to: B(arrives PB7), 5(P57 arrives), 6(P67 arrives), 8(P78 leaves)
    # net in = PB7 + P57 + P67 - P78 >= D7
    # negated: -PB7 - P57 - P67 + P78 <= -D7
    [  0,   0,   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,   0,   1,   0,   0,   0,   0,   0],
    # Row 9: Node 8 — connects to: B(arrives PB8), 6(P68 arrives), 7(P78 arrives), 9(P89 leaves)
    # net in = PB8 + P68 + P78 - P89 >= D8
    # negated: -PB8 - P68 - P78 + P89 <= -D8
    [  0,   0,   0,   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1,   1,   0,   0,   0,   0],
    # Row 10: Node 9 — connects to: B(arrives PB9), 8(P89 arrives), 10(P9_10 leaves)
    # net in = PB9 + P89 - P9_10 >= D9
    # negated: -PB9 - P89 + P9_10 <= -D9
    [  0,   0,   0,   0,   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,   1,   0,   0,   0],
    # Row 11: Node 10 — connects to: 9(P9_10 arrives), 11(P10_11 leaves), 12(P10_12 leaves)
    # net in = P9_10 - P10_11 - P10_12 >= D10
    # negated: -P9_10 + P10_11 + P10_12 <= -D10
    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,   1,   1,   0],
    # Row 12: Node 11 — connects to: 1(P1_11 arrives), 10(P10_11 arrives), 12(P11_12 leaves)
    # net in = P1_11 + P10_11 - P11_12 >= D11
    # negated: -P1_11 - P10_11 + P11_12 <= -D11
    [  0,   0,   0,   0,   0,   0,   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,   0,   1],
    # Row 13: Node 12 — connects to: 1(P1_12 arrives), 2(P2_12 arrives), 10(P10_12 arrives), 11(P11_12 arrives)
    # net in = P1_12 + P2_12 + P10_12 + P11_12 >= D12
    # negated: -P1_12 - P2_12 - P10_12 - P11_12 <= -D12
    [  0,   0,   0,   0,   0,   0,   0,   0,  -1,   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1,  -1],
], dtype=float)

# RHS vector b (14,):
#   rows 0-1: generator capacities (positive)
#   rows 2-13: negated demands (negative)
C.b = np.block([G, -D])                        # [G_A, G_B, -D1, ..., -D12]

# Cost coefficient vector c (24,):
#   cA for PA1, PA2, PA3; cB for PB7, PB8, PB9; 0 for all transmission flows
C.c = np.block([cA*np.ones(3), cB*np.ones(3), np.zeros(18)])

# Transmission loss factor
C.loss = loss

# A_orig: un-negated constraint matrix — used to compute loss direction correctly.
C.A_orig = C.A.copy()
C.A_orig[2:, :] *= -1   # un-negate the demand rows (rows 2-13)

m, n = C.A.shape   # m=14 constraints, n=24 design variables

# Sanity check: each column of A should sum to 0
# (every branch flow leaves one node and arrives at another, so net = 0)
sumA = np.sum(C.A, axis=0)
print("Column sums of A (should all be 0):", sumA)

# --- Bounds on design variables ----------------------------------------------
# Generator-connected branches: flow must be 0 <= P <= T (can only leave generator)
# All other branches: -T <= P <= T (bidirectional)

v_lb      = -T * np.ones(n)          # all flows lower bound = -T
v_lb[0:6] = 0.0                      # generator branches: PA1,PA2,PA3,PB7,PB8,PB9 >= 0
v_ub      = T * np.ones(n)           # all flows upper bound = T

# Physics-informed initial guess: push power through the network from each
# generator toward its local demand nodes. Total demand = 276 MW.
# B (cheap, cB=12) is loaded heavily; A (expensive, cA=20) covers nodes 1-3.

# Initial guess for power flows v (24,):
v_init = np.array([
    15.0, 20.0, 25.0, 85.0, 85.0, 85.0,  # generator flows
     5.0,  5.0,  5.0,  5.0,  5.0, 20.0,  # upper-network transmission
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # mid-network transmission
    10.0, 50.0, 50.0,  5.0,  5.0,  5.0,  # lower-network transmission
])

# --- Optimization options [msg, tol_v, tol_f, tol_g, max_evals] -------------
opts = [1, 1e-3, 1e-4, 1e-3, 5e3]

# --- Solve in two steps: first without losses (easier), then with losses -----
# Step 1: loss = 0 to get a good initial guess
C.loss = 0
v_opt, f_opt, g_opt, cvg_hst, lbda_opt, _ = sqp(
    power_grid_analysis, v_init, v_lb, v_ub, opts, C)

# Step 2: use lossless solution as warm start, now solve with losses
v_init = v_opt
C.loss = loss
v_opt, f_opt, g_opt, cvg_hst, lbda_opt, _ = sqp(
    power_grid_analysis, v_init, v_lb, v_ub, opts, C)

# --- Plot convergence history ------------------------------------------------
format_plot(font_size=15, line_width=3, marker_size=7)
plot_cvg_hst(cvg_hst, v_opt, opts, save_plots=True)
plt.show()

# --- Post-processing: net generation, supply, demand, shortfall --------------
# Net generation: total power leaving generators
net_generation = np.sum(v_opt[0:6])

# Recompute L at the optimal solution using original (un-negated) A signs
L_opt      = C.A_orig @ np.diag(v_opt)
A_scaled_opt = C.A * (1 - loss * H(-L_opt))   # 14x24
supply       = -(A_scaled_opt @ v_opt)          # un-negate demand rows to get actual arrivals
net_supply   = np.sum(supply[2:])               # sum over demand nodes 1-12

# Net demand
net_demand = np.sum(D)

# Net shortfall
net_shortfall = net_demand - net_supply

# --- Display results ---------------------------------------------------------
formatted_supply = np.char.mod('%6.2f', supply[2:])
print(f"supply : {' '.join(formatted_supply)}")

formatted_demand = np.char.mod('%6.2f', D)
print(f"demand : {' '.join(formatted_demand)}")

print("f_opt (total generation cost $):", np.round(f_opt * 100) / 100)
print("net generation (MW):", np.round(net_generation * 100) / 100)
print("net demand     (MW):", np.round(net_demand * 100) / 100)
print("net supply     (MW):", np.round(net_supply * 100) / 100)
print("net shortfall  (net demand - net generation, MW):",
      np.round((net_demand - net_generation) * 1e3) / 1e3)
print("net shortfall  (net demand - net supply,     MW):",
      np.round(net_shortfall * 1e3) / 1e3)

print("\nv_opt:")
labels = ['PA1','PA2','PA3','PB7','PB8','PB9',
          'P12','P1_11','P1_12','P23','P2_12',
          'P34','P45','P46','P56','P57',
          'P67','P68','P78','P89',
          'P9_10','P10_11','P10_12','P11_12']
for lbl, val in zip(labels, v_opt):
    print(f"  {lbl:>8s} = {val:8.3f} MW")

print("\nConstraint check A*[1-eps*H(-L)]*v - b <= 0:")
print(" ", np.round(g_opt * 1e4) / 1e4)
print("  All satisfied:", np.all(g_opt <= 1e-4))

print("\nBounds check -T <= v <= T:")
print("  Lower bound ok:", np.all(v_opt >= v_lb - 1e-4))
print("  Upper bound ok:", np.all(v_opt <= v_ub + 1e-4))


# ==============================================================================
# Monte Carlo Risk Analysis stuff
# ==============================================================================
N = 100

# Lognormal demands: median = D, CoV = 0.1 for all nodes
medD = D.copy()                           # median demand vector, MW
nd   = len(D)                             # number of demand nodes (12)
covD = 0.10 * np.ones((nd, 1))           # coefficient of variation
RD   = np.eye(nd)                         # correlation matrix: uncorrelated
Drand = lognormal.rnd(medD, covD, N, RD) # nd-by-N matrix of random demands

# Beta-distributed line capacities: between 70 and 99 MW, q=4, p=2
aT    = 70.0 * np.ones((n, 1))           # lower bound on capacity, MW
bT    = 99.0 * np.ones((n, 1))           # upper bound on capacity, MW
qT    =  4.0 * np.ones((n, 1))           # lower exponent of Beta distribution
pT    =  2.0 * np.ones((n, 1))           # upper exponent of Beta distribution
RT    = np.eye(n)                         # correlation matrix: uncorrelated
Trand = beta.rnd(aT, bT, qT, pT, N, RT) # n-by-N matrix of random capacities

# Storage for Monte Carlo results
mc7_net_demand   = np.zeros(N)
mc7_net_supply   = np.zeros(N)
mc7_net_shortfall = np.zeros(N)

for k in range(N):
    Dk = Drand[:, k]      # random demand vector for this sample, MW
    Tk = Trand[:, k]      # random capacity vector for this sample, MW

    C.b = np.block([G, -Dk])

    v_lb_k        = np.zeros(n)
    v_lb_k[0:6]   = 0.0             # generator branches non-negative
    v_lb_k[6:]    = -Tk[6:]         # bidirectional branches
    v_ub_k        = Tk.copy()

    # Warm-start from nominal optimal solution
    v_init_k = np.clip(v_opt, v_lb_k, v_ub_k)

    # Solve (with losses)
    C.loss = loss
    try:
        v_k, f_k, g_k, _, _, _ = sqp(
            power_grid_analysis, v_init_k, v_lb_k, v_ub_k, opts, C)
    except Exception:
        # If sqp fails to converge, record what it has
        v_k = v_init_k

    # Compute supply delivered to demand nodes for this sample
    L_k        = C.A_orig @ np.diag(v_k)
    A_scaled_k = C.A * (1 - loss * H(-L_k))
    supply_k   = -(A_scaled_k @ v_k)

    mc7_net_demand[k]    = np.sum(Dk)
    mc7_net_supply[k]    = np.sum(supply_k[2:])
    mc7_net_shortfall[k] = mc7_net_demand[k] - mc7_net_supply[k]

# Reset C.b to nominal before Task 8
C.b = np.block([G, -D])

#ECDF of one demand sample 
fig7c = plt.figure()
plot_ECDF_ci(Drand[0, :], 95, fig7c.number, x_label='Demand at Node 1 (MW)', save_plots=True)
plt.title('ECDF of Demand at Node 1 (lognormal, median=10 MW)')
plt.savefig('task7c_ecdf_demands.png', dpi=150)
plt.show()

# ECDF of one line capacity sample
fig7d = plt.figure()
plot_ECDF_ci(Trand[0, :], 95, fig7d.number, x_label='Line Capacity, Branch 0 (MW)', save_plots=True)
plt.title('ECDF of Line Capacity (Beta, 70-99 MW, q=4, p=2)')
plt.savefig('task7d_ecdf_capacities.png', dpi=150)
plt.show()

# scatter plot of two node demands (uncorrelated)
fig, ax = plt.subplots()
ax.scatter(Drand[0, :], Drand[1, :], alpha=0.6)
ax.set_xlabel('Demand at Node 1 (MW)')
ax.set_ylabel('Demand at Node 2 (MW)')
ax.set_title('Demand Node 1 vs Node 2 (should appear uncorrelated)')
format_plot(font_size=14, line_width=2, marker_size=5)
plt.tight_layout()
plt.savefig('task7e_scatter_demands.png', dpi=150)
plt.show()

# scatter plot of two line capacities (uncorrelated)
fig, ax = plt.subplots()
ax.scatter(Trand[0, :], Trand[1, :], alpha=0.6)
ax.set_xlabel('Capacity, Branch 0 (MW)')
ax.set_ylabel('Capacity, Branch 1 (MW)')
ax.set_title('Line Capacity Branch 0 vs Branch 1 (should appear uncorrelated)')
format_plot(font_size=14, line_width=2, marker_size=5)
plt.tight_layout()
plt.savefig('task7f_scatter_capacities.png', dpi=150)
plt.show()

# --- Task 7(g): ECDF of net demand, net supply, net shortfall ----------------
fig7g = plt.figure()
plot_ECDF_ci(mc7_net_demand,    95, fig7g.number, x_label='Net Demand (MW)',    save_plots=False)
plot_ECDF_ci(mc7_net_supply,    95, fig7g.number, x_label='Net Supply (MW)',    save_plots=False)
plot_ECDF_ci(mc7_net_shortfall, 95, fig7g.number, x_label='Net Shortfall (MW)', save_plots=True)
plt.title('ECDF of Net Demand, Net Supply, Net Shortfall (uncorrelated)')
plt.show()

# --- Task 7(h): fraction of samples with shortfall > 0.1 MW -----------------
frac7 = np.mean(mc7_net_shortfall > 0.1)
print(f"Fraction of samples with net shortfall > 0.1 MW: {frac7:.3f}")


# 8: correlated demands and line capacities 

# Correlated demand: pairwise correlation = 0.8
RD_corr = 0.8 * np.ones((nd, nd)) + 0.2 * np.eye(nd)
Drand_corr = lognormal.rnd(medD, covD, N, RD_corr)

# Correlated line capacity: pairwise correlation = 0.7
RT_corr = 0.7 * np.ones((n, n)) + 0.3 * np.eye(n)
Trand_corr = beta.rnd(aT, bT, qT, pT, N, RT_corr)

# Storage
mc8_net_demand    = np.zeros(N)
mc8_net_supply    = np.zeros(N)
mc8_net_shortfall = np.zeros(N)

for k in range(N):
    Dk = Drand_corr[:, k]
    Tk = Trand_corr[:, k]

    C.b = np.block([G, -Dk])

    v_lb_k       = np.zeros(n)
    v_lb_k[0:6]  = 0.0
    v_lb_k[6:]   = -Tk[6:]
    v_ub_k       = Tk.copy()

    v_init_k = np.clip(v_opt, v_lb_k, v_ub_k)

    C.loss = loss
    try:
        v_k, f_k, g_k, _, _, _ = sqp(
            power_grid_analysis, v_init_k, v_lb_k, v_ub_k, opts, C)
    except Exception:
        v_k = v_init_k

    L_k        = C.A_orig @ np.diag(v_k)
    A_scaled_k = C.A * (1 - loss * H(-L_k))
    supply_k   = -(A_scaled_k @ v_k)

    mc8_net_demand[k]    = np.sum(Dk)
    mc8_net_supply[k]    = np.sum(supply_k[2:])
    mc8_net_shortfall[k] = mc8_net_demand[k] - mc8_net_supply[k]

# Reset C.b
C.b = np.block([G, -D])

# 8(c): ECDF of one demand sample (correlated)
fig8c = plt.figure()
plot_ECDF_ci(Drand_corr[0, :], 95, fig8c.number, x_label='Demand at Node 1 (MW)', save_plots=True)
plt.title('ECDF of Demand at Node 1 (correlated, rho=0.8)')
plt.savefig('task8c_ecdf_demands.png', dpi=150)
plt.show()

# 8(d): ECDF of one line capacity (correlated) 
fig8d = plt.figure()
plot_ECDF_ci(Trand_corr[0, :], 95, fig8d.number, x_label='Line Capacity, Branch 0 (MW)', save_plots=True)
plt.title('ECDF of Line Capacity (correlated, rho=0.7)')
plt.savefig('task8d_scatter_capacities.png', dpi=150)
plt.show()

# 8(e): scatter plot two node demands (correlated) 
fig, ax = plt.subplots()
ax.scatter(Drand_corr[0, :], Drand_corr[1, :], alpha=0.6, color='orange')
ax.set_xlabel('Demand at Node 1 (MW)')
ax.set_ylabel('Demand at Node 2 (MW)')
ax.set_title('Demand Node 1 vs Node 2 (should appear correlated, rho=0.8)')
format_plot(font_size=14, line_width=2, marker_size=5)
plt.tight_layout()
plt.savefig('task8e_scatter_demands.png', dpi=150)
plt.show()

# 8(f): scatter plot two line capacities (correlated) 
fig, ax = plt.subplots()
ax.scatter(Trand_corr[0, :], Trand_corr[1, :], alpha=0.6, color='orange')
ax.set_xlabel('Capacity, Branch 0 (MW)')
ax.set_ylabel('Capacity, Branch 1 (MW)')
ax.set_title('Line Capacity Branch 0 vs Branch 1 (correlated, rho=0.7)')
format_plot(font_size=14, line_width=2, marker_size=5)
plt.tight_layout()
plt.savefig('task8f_scatter_capacities.png', dpi=150)
plt.show()

# 8(g): ECDF of net demand, net supply, net shortfall (correlated) 
fig8g = plt.figure()
plot_ECDF_ci(mc8_net_demand,    95, fig8g.number, x_label='Net Demand (MW)',    save_plots=True)
plot_ECDF_ci(mc8_net_supply,    95, fig8g.number, x_label='Net Supply (MW)',    save_plots=True)
plot_ECDF_ci(mc8_net_shortfall, 95, fig8g.number, x_label='Net Shortfall (MW)', save_plots=True)
plt.title('ECDF of Net Demand, Net Supply, Net Shortfall (correlated)')
plt.show()

# 8(h): fraction with shortfall > 0.1 MW 
frac8 = np.mean(mc8_net_shortfall > 0.1)
print(f"Fraction of samples with net shortfall > 0.1 MW: {frac8:.3f}")

print("\nDone.")