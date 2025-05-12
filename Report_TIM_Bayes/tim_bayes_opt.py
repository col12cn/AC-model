import math
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import os
os.environ["GRB_LICENSE_FILE"] = "/nfs/home/colinn/gurobi.lic"

import gurobipy as gp
print("Gurobi version:", gp.gurobi.version())
from gurobipy import GRB

import pickle
import time
import datetime

# -------------------------
# List of NetCDF trace files
# -------------------------
trace_files = ["/nfs/home/colinn/Report_AC/Report_TIM_Bayes/20250512_180619/trace_250_gap_1_seed_300.nc", "/nfs/home/colinn/Report_AC/Report_TIM_Bayes/20250512_180619/trace_2000_gap_1_seed_300.nc"]

# -------------------------
# Parameter Setup
# -------------------------
N = 10               # Number of time intervals
m = 1500             # Number of scenarios
X = 1e3              # Total shares to sell
T = 1                # Trading horizon
tau = T / N         # Interval length
s = 0                # Spread cost per share
eta = 1e-03 * tau    # Temporary impact coefficient
sigma = 0.45         # Volatility

delta = 0.1          # Fixed chance constraint threshold
M = 1e4              # Big-M constant
C_max = 1000         # IS threshold
num_runs = 10        # Number of Monte Carlo runs per trace file

all_results = {}

# -------------------------
# Loop over each trace file
# -------------------------
for trace_file in trace_files:
    print(f"\nProcessing trace file: {trace_file} ")
    # Load posterior samples via ArviZ
    idata = az.from_netcdf(trace_file)
    # Flatten posterior draws for joint sampling
    kappa_vals = idata.posterior['kappa'].values.reshape(-1)
    gamma_vals = idata.posterior['gamma'].values.reshape(-1)
    beta_vals  = idata.posterior['beta'].values.reshape(-1)
    n_samples = kappa_vals.size

    # Prepare containers
    inventory_plots = []
    trade_plots = []
    obj_vals = []
    solve_times = []
    tail_probs = []
    errors = {}

    # Monte Carlo sampling using joint posterior draws
    for run in range(num_runs):
        print(f" Run Number: {run}")
        xi = np.random.normal(0, 1, (m, N+1))
        # Sample joint indices into posterior draws
        idx = np.random.randint(0, n_samples, size=m)
        kappa_samples = kappa_vals[idx]
        gamma_samples = gamma_vals[idx]
        rho_samples   = beta_vals[idx]

        model = gp.Model("TIM_MCMC_Opt")
        model.Params.OutputFlag = 0
        model.Params.NonConvex = 2
        model.Params.MIPFocus = 2
        model.Params.Cuts = 2
        model.Params.Presolve = 2
        model.Params.OBBT = 2
        model.Params.Threads = 86
        model.Params.MIPGap = 2e-3

        # Decision variables
        n = model.addVars(N+1, lb=0, name="n")
        b = model.addVars(m, vtype=GRB.BINARY, name="b")
        model.addConstr(gp.quicksum(n[k] for k in range(N+1)) == X, "TotalShares")

        # Build IS expressions
        IS_exprs = []
        for p in range(m):
            κ = kappa_samples[p]
            γ = gamma_samples[p]
            ρ = rho_samples[p]

            perm = 0.5 * γ * X**2 - 0.5 * γ * gp.quicksum(n[k]*n[k] for k in range(N+1))

            spread = s * X

            temp = (eta / tau) * gp.quicksum(n[k]*n[k] for k in range(N+1))

            sto = 0.0
            for k in range(N+1):
                inv_expr_k = X - gp.quicksum(n[i] for i in range(k+1))
                sto += inv_expr_k * xi[p, k]
            sto = sigma * math.sqrt(tau) * sto

            #Transient Impact
            decay = κ * gp.quicksum(n[k] * (gp.quicksum(
                n[i] * math.exp(-ρ * tau * (k - i)) for i in range(k)))
                for k in range(N+1))


            IS_p = perm + spread + temp - sto + decay
            IS_exprs.append(IS_p)
            model.addQConstr(IS_p <= C_max + M * b[p], name=f"IS_{p}")

        model.addConstr(gp.quicksum(b[p] for p in range(m)) <= delta * m, "Chance")
        model.setObjective((1.0/m) * gp.quicksum(IS_exprs), GRB.MINIMIZE)
        model.optimize()

        if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            trades = [n[k].X for k in range(N+1)]
            b_arr = np.array([b[p].X for p in range(m)])
            obj_vals.append(model.ObjVal)
            solve_times.append(model.Runtime)
            tail_probs.append(b_arr.mean())
        else:
            errors[f"run_{run}"] = model.status
            obj_vals.append(np.nan)
            solve_times.append(np.nan)
            tail_probs.append(np.nan)

        inv = X
        inv_traj = [X]
        for q in trades:
            inv -= q
            inv_traj.append(inv)
        inventory_plots.append(inv_traj)
        trade_plots.append(trades)

    # Store results per trace
    all_results[os.path.basename(trace_file)] = {
        "inventory": inventory_plots,
        "trades": trade_plots,
        "obj": obj_vals,
        "time": solve_times,
        "tail": tail_probs,
        "errors": errors
    }

# Save aggregated outputs
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"TIM_opt_results_{timestamp}"
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "all_results.pkl"), "wb") as f:
    pickle.dump(all_results, f)
print(f"Optimization results saved to: {outdir}")
