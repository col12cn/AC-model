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
M = 5e3              # Big-M constant
C_max = 1001         # IS threshold
num_runs = 10        # Number of Monte Carlo runs per trace file

all_results = {}

# -------------------------
# Loop over each trace file
# -------------------------
theta_sigma_list = [8e-06, 2e-06, 0]
alpha_a_list = [5, 8, 1e6]

for i in range(3):
    theta_sigma = theta_sigma_list[i]
    alpha_a = alpha_a_list[i]
    # Prepare containers
    inventory_plots = []
    trade_plots = []
    obj_vals = []
    solve_times = []
    tail_probs = []
    errors = {}
    kappa_list = []
    gamma_list = []

    # Monte Carlo sampling using joint posterior draws
    for run in range(num_runs):
        print(f" Run Number: {run}")
        xi = np.random.normal(0, 1, (m, N+1))
        # Sample joint indices into posterior draws
        theta_samples = np.random.normal(2e-5, theta_sigma, m)
        if i != 2:
            alpha_samples = np.random.beta(alpha_a, alpha_a, m)
        else:
            alpha_samples = np.array([0.5] * m)

        # Transformations to get model parameters
        kappa_samples = theta_samples * alpha_samples * 100     # scale as before
        gamma_samples = theta_samples * (1 - alpha_samples) * 100

        print(f"Mean of Kappa: {np.mean(kappa_samples)}", f"STD : {np.std(kappa_samples)}")
        print(f"Mean of Gamma: {np.mean(gamma_samples)}", f"STD : {np.std(gamma_samples)}")

        model = gp.Model("TIM_MCMC_Opt")
        model.Params.OutputFlag = 0
        model.Params.NonConvex = 2
        model.Params.Cuts = 2
        model.Params.Presolve = 2
        model.Params.MIPFocus = 2
        model.Params.OBBT = 2
        model.Params.Threads = 86
        model.Params.MIPGap = 0.002
        if i ==0:
            model.Params.TimeLimit = 6000
        else:
            model.Params.TimeLimit = 4800
        # Decision variables
        n = model.addVars(N+1, lb=0, name="n")
        b = model.addVars(m, vtype=GRB.BINARY, name="b")
        model.addConstr(gp.quicksum(n[k] for k in range(N+1)) == X, "TotalShares")

        # Build IS expressions
        IS_exprs = []
        for p in range(m):
            κ = kappa_samples[p]
            γ = gamma_samples[p]
            # ρ = rho_samples[p]
            ρ = 2.231
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

        if model.status in (2, 9, 13):
            trades = [n[k].X for k in range(N+1)]
            b_arr = np.array([b[p].X for p in range(m)])
            obj_vals.append(model.ObjVal)
            solve_times.append(model.Runtime)
            tail_probs.append(b_arr.mean())
            print("Model Status =", model.status)
            print("Gap =", model.MIPGap)
        else:
            trades = [[np.nan]*(N+1)]
            errors[f"run_{run}"] = model.status
            obj_vals.append(np.nan)
            solve_times.append(np.nan)
            tail_probs.append(np.nan)

        inv = X
        inv_traj = [X]
        if model.status in (2, 9, 13):
            for q in trades:
                inv -= q
                inv_traj.append(inv)
        inventory_plots.append(inv_traj)
        trade_plots.append(trades)

        kappa_list.append(kappa_samples)
        gamma_list.append(gamma_samples)
    # Store results per trace
    all_results[i] = {
        "inventory": inventory_plots,
        "trades": trade_plots,
        "obj": obj_vals,
        "time": solve_times,
        "tail": tail_probs,
        "errors": errors,
        "kappa_samp": kappa_list,
        "gamma_samp": gamma_list,
    }

# Save aggregated outputs
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"TIM_opt_fixed_mean_results_{timestamp}"
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "all_results.pkl"), "wb") as f:
    pickle.dump(all_results, f)
print(f"Optimization results saved to: {outdir}")
