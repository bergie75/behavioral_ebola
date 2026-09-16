import arviz as az
from ode_code import load
from numpyro import sample
from numpyro import deterministic

runs = load("behav_5000x4_0")
print(az.summary(az.from_numpyro(runs)))
