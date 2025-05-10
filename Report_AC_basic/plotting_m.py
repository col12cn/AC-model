import math
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["GRB_LICENSE_FILE"] = "/nfs/home/colinn/gurobi.lic"
import gurobipy as gp
from gurobipy import GRB

print("Gurobi version:", gp.gurobi.version())

# -------------------------
# Parameter Setup
# -------------------------
N = 10             # Number of time intervals
X = 1e3            # Total shares to sell
T = 10             # Trading horizon
tau = T / N        # Length of each interval

s = 0.0625         # Spread cost per share
eta = 2.5e-03      # Temporary impact coefficient
gamma = 2.5e-04    # Permanent impact parameter
sigma = 0.95       # Volatility

M = 1e6            # Big-M constant
C_max = 2563       # Threshold for implementation shortfall

delta = 1
num_runs = 30      # Monte Carlo replications per δ


# To store results
delta_results = {}
delta_list = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]

# -------------------------
# Run optimization for each delta
# -------------------------
for m in delta_list:
    print(f"Processing particles: {m}")

    # per‐run storage
    inventory_plots = []
    trade_plots     = []
    obj_vals        = []
    solve_times     = []
    tail_probs      = []

    for run in range(num_runs):
        # simulate random shocks
        xi = np.random.normal(0, 1, (m, N))
        gamma_samples = np.random.normal(gamma, 0, m)

        # build model
        model = gp.Model("FixedGammaOpt")
        model.setParam('OutputFlag', 0)

        n_vars = model.addVars(N, lb=0, name="n")
        b_vars = model.addVars(m, vtype=GRB.BINARY, name="b")
        model.addConstr(gp.quicksum(n_vars[k] for k in range(N)) == X, "TotalShares")

        IS_expr = []
        for p in range(m):
            perm_term = (
                0.5 * gamma_samples[p] * X**2
                - 0.5 * gamma_samples[p] * gp.quicksum(n_vars[k] * n_vars[k] for k in range(N))
            )
            spread_term = s * X
            temp_term   = eta * gp.quicksum((n_vars[k] * n_vars[k]) / tau for k in range(N))

            stoch = gp.LinExpr()
            for k in range(N):
                inv_expr_k = X - gp.quicksum(n_vars[i] for i in range(k+1))
                stoch += inv_expr_k * xi[p, k]
            stoch = sigma * math.sqrt(tau) * stoch

            IS_p = perm_term + spread_term + temp_term + stoch
            IS_expr.append(IS_p)
            model.addQConstr(IS_p <= C_max + M * b_vars[p], name=f"IS_Constraint_{p}")

        model.addConstr(gp.quicksum(b_vars[p] for p in range(m)) <= delta * m, "ChanceConstraint")
        model.setObjective((1.0 / m) * gp.quicksum(IS_expr[p] for p in range(m)), GRB.MINIMIZE)

        model.optimize()

        # --- NEW: extract per‐run metrics ---
        if model.status == GRB.OPTIMAL:
            obj_vals.append(model.ObjVal)
            solve_times.append(model.Runtime)
            # tail‐probability
            b_arr = np.array([b_vars[p].X for p in range(m)])
            tail_probs.append(b_arr.mean())
        else:
            # in case of infeasible, record NaN
            obj_vals.append(np.nan)
            solve_times.append(np.nan)
            tail_probs.append(np.nan)
        # --- END NEW ---

        # get trades & inventory
        trade_list = [n_vars[k].X for k in range(N)] if model.status == GRB.OPTIMAL else [0]*N
        inv_list = [X]
        inv = X
        for t in trade_list:
            inv -= t
            inv_list.append(inv)

        trade_plots.append(trade_list)
        inventory_plots.append(inv_list)

    # --- NEW: aggregate results ---
    mean_traj = np.array(inventory_plots).mean(axis=0)
    std_traj  = np.array(inventory_plots).std(axis=0)

    mean_obj  = np.nanmean(obj_vals)
    std_obj   = np.nanstd(obj_vals)

    mean_time = np.nanmean(solve_times)
    std_time  = np.nanstd(solve_times)

    mean_tail = np.nanmean(tail_probs)
    std_tail  = np.nanstd(tail_probs)
    # --- END NEW ---

    # store everything
    delta_results[m] = {
        "mean_inventory": mean_traj,
        "std_inventory":  std_traj,
        "inventory_runs": inventory_plots,
        "trade_runs":     trade_plots,
        "obj_vals":       obj_vals,
        "mean_obj":       mean_obj,
        "std_obj":        std_obj,
        "solve_times":    solve_times,
        "mean_time":      mean_time,
        "std_time":       std_time,
        "tail_probs":     tail_probs,
        "mean_tail":      mean_tail,
        "std_tail":       std_tail,
    }

# (Your existing plotting code can then pull from delta_results[...] as needed)
for m in delta_list:
    print(f"Mean_IS {m} : " , (delta_results[m]["mean_obj"]))
    print(f"Std_IS_{m} : " , (delta_results[m]["std_obj"]))
    print(f"Mean_Time_{m} : " , (delta_results[m]["mean_time"]))
    print(f"StdTime_{m} : " , (delta_results[m]["std_time"]))
# --- Create an output directory with the current time stamp ---
import pickle
results_dir = f"particle count"
# os.makedirs(results_dir, exist_ok=True)
pickle_file = os.path.join(results_dir, "particles count_results_delta1.pkl")
with open(pickle_file, "wb") as f:
    pickle.dump(delta_results, f)