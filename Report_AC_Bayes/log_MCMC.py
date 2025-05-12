import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pymc as pm
from scipy.stats import gaussian_kde
import arviz as az
import pandas as pd
from AC_class import AlmgrenChrissEnv
from data_gen_func import *

# --- Parameters ---
sample_sizes = [1, 10,25, 50, 100, 250, 500, 1000, 2000]
seeds = [50]

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = f"results_random_seed_1.2/run_{timestamp}"
os.makedirs(base_dir, exist_ok=True)

import time
start_time = time.time()

df = pd.read_csv('/nfs/home/colinn/Report_AC_Bayes/AC_model_data1.2.csv')

# --- Prior Setup ---
a, b = 2e-03, 2.5e-07
theta_sigma = np.sqrt(np.log(1 + b / (a * a)))
theta_mu = np.log(a) - 0.5 * np.log((b / a * a) + 1)

# Store posterior samples
dict_of_traces = {}
posterior_gammas = {seed: {} for seed in seeds}

# --- Run Inference ---
for seed1 in seeds:
    for n in sample_sizes:
        print(f"[INFO] Sampling n = {n}, seed = {seed1}")
        x1, y1 = price_change(extract(df, n, seed = seed1))

        with pm.Model() as model:
            theta = pm.Normal("theta", mu=theta_mu, sigma=theta_sigma)
            gamma = pm.Deterministic("gamma", pm.math.exp(theta))
            likelihood = pm.Normal("y", mu=gamma * x1, sigma=0.95, observed=y1)
            trace = pm.sample(1500, tune=4000, target_accept=0.99, return_inferencedata=True,cores =8, chains = 8)

        # Save full trace
        trace_file = os.path.join(base_dir, f"trace_n{n}_seed{seed1}.nc")
        az.to_netcdf(trace, trace_file)

        # Store posterior gamma samples
        posterior_gamma = trace.posterior["gamma"].values.flatten()
        posterior_gammas[seed1][n] = posterior_gamma

        print(f"[SAVED] trace_n{n}_seed{seed1}.nc")

# --- Plot per-seed comparisons of sample sizes ---
for seed1 in seeds:
    plt.figure(figsize=(10, 6))
    for n in sample_sizes:
        samples = posterior_gammas[seed1][n]
        kde = gaussian_kde(samples)
        x_plot = np.linspace(*np.percentile(samples, [0.5, 99.5]), 1000)
        plt.plot(x_plot, kde(x_plot), label=f"n = {n}")

plt.axvline(1.2e-03, linestyle="--", color="black", label="True γ")
plt.grid(True)
plt.xlabel("γ ($10^{-5}$)", fontsize=n-2)
plt.ylabel('Density', fontsize = n-2)
plt.legend(loc = "best", fontsize = n-8)
# plt.ticklabel_format(style='plain', axis='x')
plt.tick_params(axis="both", which="both",labelleft =False, labelsize=n-4)
plt.grid(True)
ax = plt.gca()
ax.get_xaxis().get_offset_text().set_visible(False)
plt.tight_layout()
plot_path = os.path.join(base_dir, f'gamma_density_posterior.pdf')
plt.savefig(plot_path, bbox_inches = 'tight', pad_inches = 0.05)
plt.close()
print(f"[PLOT SAVED] {plot_path}")

end_time = time.time()
print("Total runtime (seconds):", end_time - start_time)