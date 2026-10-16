import arviz as az
from ode_code import load, calc_basic_rep, plot_mean_results, mdict

run_name = "2026_10_14_16_50_06_776462_0"
runs = load(run_name)
# print(az.summary(az.from_numpyro(runs)))
calc_basic_rep(run_name)
plot_mean_results(run_name, mdict)
