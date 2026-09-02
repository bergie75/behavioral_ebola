import jax
import jax.numpy as np
from jax.numpy import array as arr
from jax import lax, random
from jax.experimental.ode import odeint
from jax.scipy.special import logsumexp, tanh
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
import numpy as num ##scipy needs actual numpy

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

mdict = {"conn_mat" : 1,
    "exp_alpha": 1,
    "hosp_alpha": 1,
    "dead_alpha": 1,
    "inc_rate": 1,
    "inf_death_rate": 1,
    "hosp_rate": 1,
    "rec_rate_I": 1,
    "rec_rate_H": 1,
    "safe_bury_rate": 1}

# load case data
cwd = os.getcwd()
case_file = os.path.join(cwd, "behavioral_ebola", "data", "insp_sitrep__cumulative_confirmed_cases__daily.csv")
DRC_cases = pd.read_csv(case_file)
DRC_cases["daily"] = np.diff(DRC_cases["cumulative_confirmed_cases"].values, prepend=0)

# find and replace any differencing errors in the data, from Sasha's
bad_DRC = DRC_cases["daily"] < 0.0
DRC_cases.loc[bad_DRC,"daily"] = np.nan
DRC_ok = np.logical_not(bad_DRC.values)

# program flow controls and parameter setting
run_models = True
runs_behav = []
dat = {'mdict': mdict, 'data_ok': DRC_ok}

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

def ebola_rhs(y,t,p):
    # unpack
    S, E, C, I, H, F, R, sigma, CumC, CumI = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]
    beta, conn_mat, exp_alpha, hosp_alpha, dead_alpha, death_thresh, sigma_rate, inc_rate, death_rate_C, death_rate_I, hosp_rate_C, hosp_rate_I, hosp_death_rate, rec_rate_C, rec_rate_I, rec_rate_H, safe_bury_rate, conf_rate = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17]

    # important intermediate quantitities
    N = S + E + C + I + H + F + R
    force_of_infection = (1-sigma)*beta*np.matmul(conn_mat, (exp_alpha*E+C+I+hosp_alpha*H+dead_alpha*F)/N)

    # compute derivatives
    S_dot = -force_of_infection*S
    E_dot = force_of_infection*S - inc_rate*E
    C_dot = inc_rate*conf_rate*E - rec_rate_C*C - hosp_rate_C*C - death_rate_C*C
    I_dot = inc_rate*(1-conf_rate)*E - rec_rate_I*I - hosp_rate_I*I - death_rate_I*I
    H_dot = hosp_rate_C*C + hosp_rate_I*I - rec_rate_H*H - hosp_death_rate*H
    F_dot = hosp_death_rate*H + death_rate_C*C + death_rate_I*I - safe_bury_rate*F
    R_dot = rec_rate_C*C + rec_rate_I*I + rec_rate_H*H + safe_bury_rate*F
    sigma_dot = sigma_rate*(np.matmul(conn_mat,F)-death_thresh)*sigma*(1-sigma)
    CumC_dot = E*conf_rate*inc_rate
    CumI_dot = E*(1-conf_rate)*inc_rate
    return np.stack([S_dot, E_dot, C_dot, I_dot, H_dot, F_dot, R_dot, sigma_dot, CumC_dot, CumI_dot])

def model_behav(mdict, data_ok, beta_prior_exp = 0.2):
    # all quantities determined in advance
    conn_mat = deterministic("conn_mat", mdict["conn_mat"])
    exp_alpha = deterministic("exp_alpha", mdict["exp_alpha"])
    hosp_alpha = deterministic("hosp_alpha", mdict["hosp_alpha"])
    dead_alpha = deterministic("dead_alpha", mdict["dead_alpha"])
    inc_rate = deterministic("inc_rate", mdict["inc_rate"])
    inf_death_rate = deterministic("inf_death_rate", mdict["inf_death_rate"])
    hosp_rate = deterministic("hosp_rate", mdict["hosp_rate"])
    rec_rate_I = deterministic("rec_rate_I", mdict["rec_rate_I"])
    rec_rate_H = deterministic("rec_rate_H", mdict["rec_rate_H"])
    safe_bury_rate = deterministic("safe_bury_rate", mdict["safe_bury_rate"])

    # to be fit
    # beta, death_thresh, sigma_rate

    p = [beta, conn_mat, exp_alpha, hosp_alpha, dead_alpha, death_thresh, sigma_rate, inc_rate, death_rate_C, death_rate_I, hosp_rate_C, hosp_rate_I, hosp_death_rate, rec_rate_C, rec_rate_I, rec_rate_H, safe_bury_rate, conf_rate]
    pred_vals = jax.lax.scan(onestep, y0, p_t)[1] #don't need scan, call solver directly
    pred_daily = np.diff(pred_vals[:,6], prepend=0)
    pred_ok = pred_daily[data_ok] / mdict['totalI']

if __name__ == "__main__":
    if run_models:
        # chain =MCMC()
        # chain.run
        # store(chain)
        runs_behav.append( 
            MCMC(NUTS(model_behav, 
            target_accept_prob=0.9, dense_mass=True, init_strategy=pn.infer.init_to_sample), 
            num_warmup=2500, num_samples=5000, num_chains=4)
        )
        runs_behav[-1].run(key(), **dat)
        store(runs_behav[-1],'behav_5000x4_'+str(len(runs_behav)-1))
        # comment