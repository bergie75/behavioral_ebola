import jax
import jax.numpy as jnp
from jax.numpy import array as arr
from jax import lax, random
from jax.experimental.ode import odeint
from jax.scipy.special import logsumexp
from jax.scipy.special import expit as logistic
from jax.scipy.special import logit
from jax.scipy.special import gammaln
from jax.scipy import stats

import numpyro as pn
from numpyro import sample
from numpyro import deterministic
import numpyro.distributions as dist
from numpyro.distributions import Normal as Norm
from numpyro.distributions import Exponential as Ex
from numpyro.distributions import Poisson as Pois
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import print_summary, hpdi

from scipy.interpolate import BSpline
import numpy as np ##scipy needs actual numpy

import warnings
import os
import pandas as pd
import dill

# config information, taken directly from Sasha's code at the moment
# set numpyro platform to cpu because I don't have the right kind of gpu
pn.set_platform("cpu")

# tell numpyro to use mult cpu cores (this many chains can run in parallel)
pn.set_host_device_count(4)

# jax wants float32's by default, but sometimes it helps to use higher-precision floats:
HIPREC = True

if HIPREC: 
    pn.enable_x64()
    fl = np.float64
    toint = np.int64
else:
    fl = np.float32
    toint = np.int32

mdict = {"beta": 1,
    "inc_rate": 1,
    "hosp_rate_C": 1,
    "hosp_rate_I": 1,
    "rec_rate_C": 1,
    "rec_rate_I": 1,
    "rec_rate_H": 1,
    "safe_bury_rate": 1,
    "hosp_death_rate": 1,
    "death_rate_C": 1,
    "death_rate_I": 1,
    "initInfected": 1.0,
    "psize": 117233362.0}

# improt case data
cwd = os.getcwd()
case_file = os.path.join(cwd, "behavioral_ebola", "data", "insp_sitrep__cumulative_confirmed_cases__daily.csv")
DRC_cases = pd.read_csv(case_file)

# replacing NA (technically ND?) values with most recent valid value at that province
DRC_cases["cumulative_confirmed_cases"] = DRC_cases["cumulative_confirmed_cases"].replace("ND", np.nan).astype(float)
DRC_cases["date"] = pd.to_datetime(DRC_cases["date"])
DRC_cases = DRC_cases.sort_values(["nom", "date"])
DRC_cases["cumulative_confirmed_cases"] = DRC_cases.groupby("nom")["cumulative_confirmed_cases"].ffill()
sum_provinces = DRC_cases.groupby("date").sum(numeric_only=True).reset_index()

# add missing dates, interpolating the missing values
full_dates = pd.date_range(start=sum_provinces["date"].min(), end=sum_provinces["date"].max(), freq="D")

# separate out whcih dates are added
original_dates = set(sum_provinces["date"])
date_was_present = full_dates.isin(original_dates)

# finish adding missing dates
sum_provinces = sum_provinces.set_index("date").reindex(full_dates)
sum_provinces.index.name = "date"
sum_provinces["cumulative_confirmed_cases"] = sum_provinces["cumulative_confirmed_cases"].interpolate(method="time", limit_area="inside")
sum_provinces = sum_provinces.reset_index()

# number of days we must simulate with our model and all true case counts
cum_case_values = np.array(sum_provinces["cumulative_confirmed_cases"][date_was_present])
num_days = len(sum_provinces)
total_obs_cases = cum_case_values[-1]
scaled_cum_cases = cum_case_values/total_obs_cases

# # program flow controls and parameter setting
# run_models = True
# runs_behav = []
# dat = {'mdict': mdict, 'data_ok': DRC_ok}

## for jax's RNG
def key_gen(seed = random.PRNGKey(8927)):
    def key():
        nonlocal seed
        seed, new_key = random.split(seed)
        return new_key
    return key

key = key_gen()

def store(obj, name):
    cwd = os.getcwd()
    with open(os.path.join(cwd, "behavioral_ebola", "output", f'{name}.dill'), 'wb') as f:
        dill.dump(obj, f)

def load(name):
    cwd = os.getcwd()
    with open(os.path.join(cwd, "behavioral_ebola", "output", f'{name}.dill'), 'rb') as f:
        return dill.load(f)

def pdz(series):
    return (series - series.mean()) / series.std()

## resizes an array, repeating the last element if needed
def resize(x, new_size):
    return np.concatenate([x,np.repeat(x[-1], max(0,new_size-len(x)))])[:new_size]

def ebola_rhs_scalar(y,t,p):
    # unpack
    S, E, C, I, H, F, R, sigma, CumC, CumI = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]
    beta, exp_alpha, hosp_alpha, dead_alpha, death_thresh, sigma_rate, inc_rate, death_rate_C, death_rate_I, hosp_rate_C, hosp_rate_I, death_rate_H, rec_rate_C, rec_rate_I, rec_rate_H, safe_bury_rate, conf_rate = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16]

    # important intermediate quantitities
    N = S + E + C + I + H + F + R
    force_of_infection = (1-sigma)*beta*((exp_alpha*E+C+I+hosp_alpha*H+dead_alpha*F)/N)

    # compute derivatives
    S_dot = -force_of_infection*S
    E_dot = force_of_infection*S - inc_rate*E
    C_dot = inc_rate*conf_rate*E - rec_rate_C*C - hosp_rate_C*C - death_rate_C*C
    I_dot = inc_rate*(1-conf_rate)*E - rec_rate_I*I - hosp_rate_I*I - death_rate_I*I
    H_dot = hosp_rate_C*C + hosp_rate_I*I - rec_rate_H*H - death_rate_H*H
    F_dot = death_rate_H*H + death_rate_C*C + death_rate_I*I - safe_bury_rate*F
    R_dot = rec_rate_C*C + rec_rate_I*I + rec_rate_H*H + safe_bury_rate*F
    sigma_dot = sigma_rate*(F/N-death_thresh)*sigma*(1-sigma)
    CumC_dot = E*conf_rate*inc_rate
    CumI_dot = E*(1-conf_rate)*inc_rate
    return np.stack([S_dot, E_dot, C_dot, I_dot, H_dot, F_dot, R_dot, sigma_dot, CumC_dot, CumI_dot])

def model_behav_scalar(mdict):
    # all quantities determined in advance
    beta = deterministic("beta", mdict["beta"])
    inc_rate = deterministic("inc_rate", mdict["inc_rate"])

    # hospitalization parameters
    hosp_rate_C = deterministic("hosp_rate_C", mdict["hosp_rate_C"])
    hosp_rate_I = deterministic("hosp_rate_I", mdict["hosp_rate_I"])

    # recovery rates
    rec_rate_C = deterministic("rec_rate_C", mdict["rec_rate_C"])
    rec_rate_I = deterministic("rec_rate_I", mdict["rec_rate_I"])
    rec_rate_H = deterministic("rec_rate_H", mdict["rec_rate_H"])
    
    # death rates
    death_rate_C = deterministic("death_rate_C", mdict["death_rate_C"])
    death_rate_I = deterministic("death_rate_I", mdict["death_rate_I"])
    death_rate_H = deterministic("hosp_death_rate", mdict["hosp_death_rate"])
    
    # burial parameters
    safe_bury_rate = deterministic("safe_bury_rate", mdict["safe_bury_rate"])

    # to be fit
    # behavioral parameters
    conf_rate = sample("conf_rate", dist.Beta(1,8))
    sigma_rate = sample("sigma_rate", dist.Uniform(0,10))
    death_thresh = sample("death_thresh", dist.Uniform(0,1))

    # infectiousness of other states
    exp_alpha = sample("exp_alpha", dist.Beta(0.5,9.5))
    hosp_alpha = sample("hosp_alpha", dist.Beta(4,1))
    dead_alpha = sample("dead_alpha", dist.Beta(1,1))

    # construct initial condition for simulation
    i0 = deterministic("i0", mdict['initInfected'])
    e0 = deterministic("e0", 0.0)
    h0 = deterministic("h0", 0.0)
    f0 = deterministic("f0", 0.0)
    r0 = deterministic("r0", 0.0)
    log_init = sample("log_init", Norm(4.0, 4.0)) ##via Sasha
    c0 = deterministic("c0", np.exp(log_init))
    psize = mdict['Population']
    y0 = np.stack([psize - (e0 + i0 + c0 + h0 + f0 + r0), e0, c0, i0, h0, f0, r0, c0, i0])

    p = [beta, exp_alpha, hosp_alpha, dead_alpha, death_thresh, sigma_rate, inc_rate, death_rate_C, death_rate_I, hosp_rate_C, hosp_rate_I, death_rate_H, rec_rate_C, rec_rate_I, rec_rate_H, safe_bury_rate, conf_rate]
    timepoints = jnp.array(range(0, num_days))
    pred_cum = odeint(ebola_rhs_scalar, y0, timepoints, p)[:,8]
    pred_at_known_days = pred_cum[date_was_present]
    scaled_pred_known_days = pred_at_known_days/total_obs_cases

    nu = sample("nu", dist.Gamma(4.0,1.0)) ## df should be < 10 for robustness
    sigma = sample("sigma", dist.Exponential(1.0))
    eps = 1.0 / psize
    sample("daily", dist.StudentT(nu, np.log(eps+scaled_pred_known_days), sigma), obs=np.log(eps+scaled_cum_cases))

if __name__ == "__main__":
    runs_behav = []
    runs_behav.append( 
        MCMC(NUTS(model_behav_scalar, 
        target_accept_prob=0.9, dense_mass=True, init_strategy=pn.infer.init_to_sample), 
        num_warmup=2500, num_samples=5000, num_chains=4)
    )
    runs_behav[-1].run(key(), mdict)
    store(runs_behav[-1],'behav_5000x4_'+str(len(runs_behav)-1))