import sympy as sp
import dill
import os

# transmission
beta = sp.symbols("beta")
alpha_E, alpha_H, alpha_F = sp.symbols("alpha_E alpha_H alpha_F")
sigma = sp.symbols("sigma")

F = beta*(1-sigma)*sp.Matrix([
    [alpha_E,              1,                  1,             alpha_H,          alpha_F],
    [0,       0,             0,          0, 0],
    [0,       0,             0,          0, 0],
    [0,       0,             0,          0, 0],
    [0,       0,             0,          0, 0],
])

# Define the transport terms
gamma, theta = sp.symbols("gamma theta")
rho_C, eta_C, d_C = sp.symbols("rho_C eta_C d_C")
rho_I, eta_I, d_I = sp.symbols("rho_I eta_I d_I")
rho_H, d_H = sp.symbols("rho_H d_H")
omega = sp.symbols("omega")

# Jacobian of transport
V = sp.Matrix([
    [gamma,              0,                  0,             0,          0],
    [-gamma*theta,       rho_C + eta_C + d_C, 0,             0,          0],
    [-gamma*(1 - theta), 0,                  rho_I + eta_I + d_I, 0,     0],
    [0,                   -eta_C,             -eta_I,        rho_H + d_H, 0],
    [0,                   -d_C,               -d_I,          -d_H,       omega]
])

# Compute and simplify the symbolic inverse
V_inv = sp.simplify(V.inv())
sp.pprint(V_inv)

# print matrix for basic reproduction number
res = sp.simplify(F*V_inv)
#print(res[1:5,:])  # if this is all zeros, our basic reproduction number is just the 1,1 entry

basic_rep_num = sp.simplify(res[0,0])
print(basic_rep_num)

save_rep = os.path.join(os.getcwd(), "behavioral_ebola", "output", "basic_rep.dill")
with open(save_rep, "wb") as file:
    dill.dump(basic_rep_num, file)