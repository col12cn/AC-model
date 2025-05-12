
import math
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["GRB_LICENSE_FILE"] = "/nfs/home/colinn/gurobi.lic"

import gurobipy as gp

print("Gurobi version:", gp.gurobi.version())

from gurobipy import GRB

import pickle
import time
import datetime

start_time = time.time()

# -------------------------
# Parameter Setup
# -------------------------
N = 10             # Number of time intervals
m = 2000            # Number of scenarios (particles)
X = 1e3            # Total shares to sell
T = 1             # Trading horizon
tau = T / N        # Length of each interval

s = 0             # Spread cost per share
eta =  1e-03*tau         # Temporary impact coefficient
gamma = 1e-03    # Permanent impact parameter
#sigma = 0        # Volatility
kappa = 1e-03
rho = 2.231
sigma= 0.45

# sigma_list = [0, 1e-04, 1e-03, 1e-02]
delta_list = [1, 0.2, 0.1, 0.075, 0.05]


M = 1e4              # Big-M constant
C_max = 1000     # Threshold for implementation shortfall


# Number of optimization runs for each delta (to compute average trajectories)
num_runs = 15

# To store average and std trajectories for each delta
delta_results = {}
print(f"BigM = {M}, particle = {m}")

for delta in delta_list:
    print(f"Processing delta: {delta}")
    lap_start = time.time()

    inventory_plots = []  # To store inventory trajectories for each run
    trade_plots =[]
    b_plots = []
    obj_vals        = []
    solve_times     = []
    tail_probs      = []
    error_list ={}
    for run in range(num_runs):
        print(f"Run Number: {run}")
        # Generate Brownian noise for each scenario (xi ~ N(0,1))
        xi = np.random.normal(0, 1, (m, N+1))
        # Generate gamma samples (here standard deviation is 0, so each is gamma)
        gamma_samples = np.random.normal(gamma, 0, m)
        
        # Create a new Gurobi model for this run
        model = gp.Model("TIMOpt_Speed")
        model.Params.OutputFlag   = 1
        model.Params.NonConvex    = 2
        model.Params.MIPFocus     = 2   # 1 focus on finding feasible solution, 2 focus on proving optimality
        model.Params.Cuts         = 2  
        model.Params.Presolve     = 2
        model.Params.OBBT = 2
        model.Params.Threads = 86
        # Python API
        model.Params.MIPGap = 2e-3
        
        # Decision variables: trades for each interval and binary variables for scenarios
        n_vars = model.addVars(N+1, lb=0, name="n")
        b_vars = model.addVars(m, vtype=GRB.BINARY, name="b")
        
        # Total shares traded must equal X
        model.addConstr(gp.quicksum(n_vars[k] for k in range(N+1)) == X, "TotalShares")
        
        # Build implementation shortfall (IS) expressions for each scenario
        IS_expr = []
        for p in range(m):
            gamma_p = gamma_samples[p]
            # Permanent impact: 0.5 * gamma_p * X^2 - 0.5 * gamma_p * sum(n_k^2)
            perm_term = 0.5 * gamma_p * (X**2) - 0.5 * gamma_p * gp.quicksum(n_vars[k]*n_vars[k] for k in range(N+1))
            
            # Spread cost: s * X
            spread_term = s * X
            
            # Temporary impact: eta * sum(n_k^2 / tau)
            temp_term = (eta /tau)* gp.quicksum((n_vars[k]*n_vars[k]) for k in range(N+1))
            
            # Stochastic term: sigma * sqrt(tau) * sum((X - sum_{i=0}^k n_i) * xi[p,k])
            stochastic_term = 0.0
            for k in range(N+1):
                inv_expr_k = X - gp.quicksum(n_vars[i] for i in range(k+1))
                stochastic_term += inv_expr_k * xi[p, k]
            stochastic_term = sigma * math.sqrt(tau) * stochastic_term
            
            #Transient Impact
            decay_term = kappa * gp.quicksum(n_vars[k] * (gp.quicksum(
                n_vars[i] * math.exp(-rho * tau * (k - i)) for i in range(k)))
                for k in range(N+1))


            # Total IS for scenario p
            IS_p = perm_term + spread_term + temp_term - stochastic_term + decay_term
            IS_expr.append(IS_p)

            # Big-M constraint: if b_p=0 then IS_p <= C_max, otherwise constraint is relaxed
            model.addQConstr(IS_p <= C_max + M * b_vars[p], name=f"IS_Constraint_{p}")
        
        # Chance constraint: At most delta*m scenarios can exceed the threshold
        model.addConstr(gp.quicksum(b_vars[p] for p in range(m)) <= delta * m, "ChanceConstraint")

        # Objective: minimize average implementation shortfall
        obj = (1.0 / m) * gp.quicksum(IS_expr[p] for p in range(m))
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Optimize the model
        model.optimize()
        
        allowed = (GRB.OPTIMAL, GRB.SUBOPTIMAL)   # keep only these
        trade_list = []
        b_list =[]
        if model.status in allowed:                # safe: variable values exist
            for k in range(N+1):
                trade_list.append(n_vars[k].X)

            for p in range(m):
                b_list.append(b_vars[p].X)

            run_info   = {"status": model.status,
                        "objval": model.ObjVal}
            print(run_info)
            obj_vals.append(model.ObjVal)
            solve_times.append(model.Runtime)
            # tail‐probability
            b_arr = np.array([b_vars[p].X for p in range(m)])
            tail_probs.append(b_arr.mean())

        else:
            print(f"Run sigma={sigma}, #{run}: discarded (status={model.status}).")
            error_list[f"sigma_{sigma}_run_{run}"] = model.status
            obj_vals.append(np.nan)
            solve_times.append(np.nan)
            tail_probs.append(np.nan)
            
        # Compute the inventory trajectory (starting at X and subtracting trades)
        inventory_list = [X]
        inv = X

        for trade in trade_list:
            inv -= trade
            inventory_list.append(inv)
            #plt.plot([x for x in range(len(inventory_list))], inventory_list)
        inventory_plots.append(inventory_list)
        trade_plots.append(trade_list)
        b_plots.append(b_list)

    lap_end = time.time()
    lap_time = lap_end - lap_start
    print(f"Lap time: {lap_time:.6f} seconds for sigma: {sigma}")

    # store everything
    delta_results[delta] = {
        "inventory_runs": inventory_plots,
        "trade_runs":     trade_plots,
        "obj_vals":       obj_vals,
        "solve_times":    solve_times,
        "tail_probs":     tail_probs,
    }


# --- Create an output directory with the current time stamp ---
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"TIM_delta_results_{current_time}"
os.makedirs(results_dir, exist_ok=True)

pickle_file = os.path.join(results_dir, "aggregated_results.pkl")
with open(pickle_file, "wb") as f:
    pickle.dump(delta_results, f)

print("Results saved in:", results_dir)
print(error_list)

end_time = time.time()
run_time = end_time - start_time
print(f"Elapsed time: {run_time:.6f} seconds")


# In[ ]:




