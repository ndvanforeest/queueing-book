# block modules
import numpy as np
from numpy.random import default_rng
from icecream import ic  # simple printing

import random_variable as rv

# block modules


# block data2
demand = rv.RV({1: 1 / 6, 2: 1 / 5, 3: 1 / 4, 4: 1 / 8, 5: 11 / 120, 6: 1 / 6})

L = 2
h = 40 * 0.5 / 30  # daily holding cost
b = 100 * 0.2  # daily backlog cost
K = 50

s, S = 3, 20  # The policy parameters
# block data2

# block simulation
N = 100
rng = default_rng(3)  # set the seed
D = demand.rvs(size=N, random_state=rng)
P = np.zeros(N)
Q = np.zeros(N)
I = np.zeros(N)
P[0] = I[0] = S

for t in range(1, N):
    Pprime = P[t - 1] - D[t - 1]
    Q[t] = (S - Pprime) * (Pprime <= s)
    P[t] = Pprime + Q[t]

# Mind the leadtime L
for t in range(1, min(L, N)):
    I[t] = I[t - 1] - D[t]
for t in range(min(L, N), N):
    I[t] = I[t - 1] - D[t - 1] + Q[t - L]
# block simulation

# block results
ic(D.mean(), P.mean(), I.mean(), Q.mean())

Iplus = np.maximum(I, 0)
Imin = np.maximum(-I, 0)

ic(Iplus.mean(), Imin.mean())

cost = K * (Q > 0).mean() + h * Iplus.mean() + b * Imin.mean()
ic(cost)

alpha = (I >= 0).mean()
beta = np.minimum(D, Imax).sum() / D.sum()
alpha_c = ((Q[:-L] > 0) * (I[L - 1 : -1] >= 0)).sum() / (Q[:-L] > 0).sum()
ic(alpha, beta, alpha_c)
# block results
