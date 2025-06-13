# %% Import and build directory
import numpy as np
import matplotlib.pyplot as plt
from data_syn import *
import os
from datetime import datetime
import time
start_time = time.time()

# Build a timestamped directory
now = datetime.now().strftime("%Y%m%d_%H%M%S")    
outdir = os.path.join("TIM_MCMC_Results", now)
os.makedirs(outdir, exist_ok=True)
#%% loop over different number of trade data points
trade_list = [50, 100, 250, 500, 1000]
for trades in trade_list:
    # True model parameters (to be recovered by inference)
    kappa = 1e-5   # transient impact scale
    beta  = 2.231   # decay rate
    gamma = 1e-5   # permanent impact scale
    sigma = 0.45   # volatiliy
    tau =0.01       
    min_gap = 10     # minimum gap between each trade
    seed = 300
    N = int(1.5* min_gap*(trades) + trades)
    print(f"Gap ={min_gap}, seed ={seed}")
    S, v_sim = data_syn(kappa, beta, gamma, sigma, tau, trades, N, min_gap, seed)
    print(f"Working on trades ={trades}")
    # %% MCMC 
    # %% imports
    import pytensor
    # compile all functions with Numba instead of the C/BLAS backend
    pytensor.config.mode = "NUMBA"
    # disable any C/C++ compiler calls
    pytensor.config.cxx = ""
    import pymc as pm
    import pytensor.tensor as at
    from pytensor.scan import scan
    import arviz as az
    #%% formatting data
    v_data = v_sim         # shape (T,)
    S_data = S             # shape (T+1,)
    τ      = tau           # time‐step
    dS_data = S_data[1:] - S_data[:-1]  # increments shape (T,)
    sigma_true = 0.5


    k_mu, k_sigma = lognormal_params(3e-5, 2e-5)  # for θ
    a_alpha, b_alpha = 4,4                        # for α∈[0,1]
    b_mu, b_sigma = lognormal_params(3, 2)        # for β
    #%% Building Prior and Likelihood
    with pm.Model() as model:
        # — Priors on reparameterized space — 
        θ     = pm.LogNormal("theta", mu=k_mu, sigma=k_sigma)
        α     = pm.Beta("alpha", alpha=a_alpha, beta=b_alpha)
        β     = pm.LogNormal("beta",  mu=b_mu, sigma=b_sigma)

        # — Deterministic transforms back to κ, γ — 
        κ = pm.Deterministic("kappa", θ * α)
        γ = pm.Deterministic("gamma", θ * (1 - α))

        # — Data inputs — 
        v     = at.constant(v_data)    # (T,)
        S_obs = at.constant(dS_data)   # (T,)

        # — Exponential‐impact state via scan — 
        def step(v_k, i_prev, κ, β, τ):
            # di = (κ·v_k - β·i_prev)·τ
            return i_prev + (κ * v_k - β * i_prev) * τ

        i_seq, _ = scan(
            fn=step,
            sequences=[v],
            outputs_info=[at.zeros(())],
            non_sequences=[κ, β, τ]
        )  # shape (T,)

        # shift to get i_{k-1}
        i_prev = at.concatenate([at.zeros((1,)), i_seq[:-1]])  # (T,)

        # — Build the drift on price increments — 
        drift_perm  = γ * τ * v       # permanent part
        drift_trans = (i_seq - i_prev)  # transient part
        mu_inc      = -drift_perm - drift_trans

        # — Likelihood — 
        σ_obs = sigma_true * at.sqrt(τ)
        pm.Normal("dS", mu=mu_inc, sigma=σ_obs, observed=S_obs)

        #%%  Posterior Inference
        trace = pm.sample(
            draws=1000,
            tune=4000,
            cores = 10,
            chains = 10,
            target_accept=0.99, 
            return_inferencedata=True
        )
    #%% store data
    trace.to_netcdf(os.path.join(outdir, f"trace_{trades}_gap_{min_gap}_seed_{seed}.nc"))

end_time = time.time()
run_time = end_time - start_time
print(f"Elapsed time: {run_time:.6f} seconds")




