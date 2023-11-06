# block modules
import numpy as np
from numpy.random import default_rng
from scipy.stats import rv_discrete
from icecream import ic  # simple printing

rng = default_rng(3)  # set the seed
# block modules

# block rvs
xk = range(6)
pk = [1 / 6, 1 / 5, 1 / 4, 1 / 8, 11 / 120, 1 / 6]
demand = rv_discrete(values=(xk, pk))
# block rvs

# block data2
L = 2
h = 40 * 0.5 / 30  # daily holding cost
b = 100 * 0.2  # daily backlog cost
K = 50

s, S = 3, 20  # The policy parameters
# block data2

# block simulation
N = 100
D = demand.rvs(size=N, random_state=rng)
Pp = np.zeros(N)
Qp = np.zeros(N)
Ip = np.zeros(N)
Pp[0] = Ip[0] = S

for t in range(1, N):
    Qp[t] = (S - Pp[t - 1]) * (Pp[t - 1] <= s)
    Pp[t] = Pp[t - 1] + Qp[t] - D[t]

# Mind the leadtime L
for t in range(1, min(L, N)):
    Ip[t] = Ip[t - 1] - D[t]
for t in range(min(L, N), N):
    Ip[t] = Ip[t - 1] + Qp[t - L] - D[t]
# block simulation

# block results
ic(D.mean(), Pp.mean(), Ip.mean(), Qp.mean())

Iplus = np.maximum(Ip, 0)
Imin = np.maximum(-Ip, 0)

ic(Iplus.mean(), Imin.mean())

cost = K * (Qp > 0).mean() + h * Iplus.mean() + b * Imin.mean()
ic(cost)

alpha = (Ip >= 0).mean()
beta = 1 - np.minimum(D, Imin).sum() / D.sum()
alpha_c = ((Qp[:-L] > 0) * (Ip[L - 1 : -1] >= 0)).sum() / (Qp[:-L] > 0).sum()
ic(alpha, beta, alpha_c)
# block results
