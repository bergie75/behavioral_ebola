import numpyro

numpyro.set_host_device_count(4)  # before JAX is initialized

import jax
from numpyro.infer import MCMC, NUTS

print(jax.local_device_count())  # should print 4