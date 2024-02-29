# block modules
import numpy as np
import matplotlib.pyplot as plt
from pytictoc import TicToc

import random_variable as rv


# block modules

# block seaborn
from latex_figures import fig_in_latex_format

# block seaborn


# block tictoc
def Plus(x):
    return max(x, 0)


tic = TicToc()
tic.tic()
# block tictoc

# block rvs
X = rv.RV({4: 1 / 3, 5: 1 / 3, 3: 1 / 3})
S = rv.RV({4: 1 / 3, 5: 1 / 3, 7: 1 / 3})
# block rvs

# block computedist
horizon = 30
A = rv.RV({0: 1})  # Arrival time A_0
A += X  # Arrival time A_1
W = rv.RV({0: 1})  # Waiting time W_1
# Mind that we start at 2 instead of 1
for i in range(2, horizon):
    A += X
    W = rv.apply_function(Plus, W + S - X)

D = A + W + S

tic.toc()
# block computedist

# block simulation
num_runs = 1000
nth_waiting_time = np.zeros(num_runs)
nth_departure = np.zeros(num_runs)
rng = np.random.default_rng(3)
for n in range(num_runs):
    x = X.rvs(horizon, rng)
    s = S.rvs(horizon, rng)
    x[0] = s[0] = 0

    a = x.cumsum()
    w = 0
    for i in range(1, horizon):
        w = Plus(w + s[i - 1] - x[i])
    nth_waiting_time[n] = w
    nth_departure[n] = a[i] + w + s[i]

# check KPIs
print(D.mean(), nth_departure.mean())
print(D.var(), nth_departure.var())

tic.toc()
# block simulation

# block figdeparturetimes
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey=True)
ax1.set_title("Departure times")
ax1.set_xlabel("Time")
ax1.hist(
    nth_departure,
    bins=30,
    density=True,
    cumulative=True,
    histtype='step',
    align="right",
    color='k',
    lw=0.5,
    label="sim",
)
ax1.plot(
    D.support(), [D.cdf(k) for k in D.support()], c='k', lw=0.75, label="exact"
)
ax1.legend()
# block figdeparturetimes

# block figwaitingtimes
ax2.set_title("Waiting times")
ax2.set_xlabel("Time")
ax2.hist(
    nth_waiting_time,
    bins=30,
    density=True,
    cumulative=True,
    histtype='step',
    align="right",
    color='k',
    lw=0.5,
    label="sim",
)
ax2.plot(
    W.support(), [W.cdf(k) for k in W.support()], c='k', lw=0.75, label="exact"
)
# block figwaitingtimes

# block figsave
fig.tight_layout()
fig.savefig("../figures/waiting_time_distribution.pdf")
# block figsave


"""
Plot the densities
"""

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey=True)
ax1.set_title("Departure times")
ax1.set_xlabel("Time")
ax1.hist(
    nth_departure,
    bins=30,
    density=True,
    histtype='step',
    align="right",
    color='k',
    lw=0.7,
    label="sim",
)
x = D.support()
y = [D.pmf(k) for k in D.support()]
ax1.plot(x, y, 'ko', ms=1)
ax1.plot(
    x,
    y,
    color='k',
    lw=0.75,
    label="exact",
)
ax1.legend()

ax2.set_title("Waiting times")
ax2.set_xlabel("Time")
ax2.hist(
    nth_waiting_time,
    bins=30,
    density=True,
    histtype='step',
    align="right",
    color='k',
    lw=0.7,
    label="sim",
)
x = W.support()
y = [W.pmf(k) for k in W.support()]
ax2.plot(x, y, 'ko', ms=1)
ax2.plot(
    x,
    y,
    color='k',
    lw=0.75,
    label="exact",
)

fig.tight_layout()
fig.savefig("../figures/waiting_time_distribution_pmf.pdf")

"""
Case with 4.1
"""
X = rv.RV({4.1: 1 / 3, 5: 1 / 3, 3: 1 / 3})
S = rv.RV({4: 1 / 3, 5: 1 / 3, 7: 1 / 3})

horizon = 30
A = rv.RV({0: 1})  # arrival time
W = rv.RV({0: 1})  # waiting time
for i in range(1, horizon):
    A += X
    W = rv.apply_function(Plus, W + S - X)

D = A + W + S


num_runs = 1000
nth_waiting_time = []
nth_departure = []
rng = np.random.default_rng(3)
for _ in range(num_runs):
    x = X.rvs(horizon, rng)
    s = S.rvs(horizon, rng)
    x[0] = s[0] = 0

    a = x.cumsum()
    w = 0
    for i in range(1, horizon):
        w = Plus(w + s[i - 1] - x[i])
    nth_waiting_time.append(w)
    nth_departure.append(a[i] + w + s[i])


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey=True)
ax1.set_title("Departure times")
ax1.set_xlabel("Time")
ax1.hist(
    nth_departure,
    bins=30,
    density=True,
    histtype='step',
    align="right",
    color='k',
    lw=0.7,
    label="sim",
)
x = D.support()
y = [D.pmf(k) for k in D.support()]
ax1.plot(x, y, 'ko', ms=1)
ax1.plot(
    x,
    y,
    color='k',
    lw=0.75,
    label="exact",
)
ax1.legend()

ax2.set_title("Waiting times")
ax2.set_xlabel("Time")
ax2.hist(
    nth_waiting_time,
    bins=30,
    density=True,
    histtype='step',
    align="right",
    color='k',
    lw=0.7,
    label="sim",
)
x = W.support()
y = [W.pmf(k) for k in W.support()]
ax2.plot(x, y, 'ko', ms=1)
ax2.plot(
    x,
    y,
    color='k',
    lw=0.75,
    label="exact",
)

fig.tight_layout()
fig.savefig("../figures/waiting_time_distribution_pmf_4.1.pdf")
