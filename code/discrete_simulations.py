# block modules
import numpy as np
import matplotlib.pyplot as plt

# block modules

# block seaborn
import seaborn as sns

sns.set_style("whitegrid")
tex_fonts = {
    "text.usetex": True,
    "font.family": "fourier",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

plt.rcParams.update(tex_fonts)
# block seaborn


# block queuelength
def compute_queue_length(a, c, L0=0):
    """Compute queue length L for  arrivals a,  capacities c
    and starting level L0"""
    L = np.zeros_like(a)
    L[0] = L0
    for i in range(1, len(a)):
        d = min(c[i], L[i - 1] + a[i])
        L[i] = L[i - 1] + a[i] - d
    return L


# block queuelength

# block setup
labda, mu = 6, 7  # arrival and service rates
L0 = 40  # starting level of queue
num = 50  # run length
x = np.arange(0, num)  # x values for the plot
# block setup

# block stability
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey=True)
ax1.set_xlabel("time")
ax1.set_ylabel("Queue length")
ax1.set_title("Stable system")

for seed in range(10):
    rng = np.random.default_rng(seed)
    a = rng.integers(labda - 1, labda + 2, size=num)
    c = mu * np.ones_like(a)
    L = compute_queue_length(a, c, L0)
    ax1.plot(x, L, linewidth=0.75)
# block stability

# block unstable
mu = 5
ax2.set_xlabel("time")
ax2.set_title("Unstable system")
for seed in range(10):
    rng = np.random.default_rng(seed)
    a = rng.integers(labda - 1, labda + 2, size=num)
    c = mu * np.ones_like(a)
    L = compute_queue_length(a, c, L0)
    ax2.plot(x, L, linewidth=0.75)


fig.tight_layout()
fig.savefig('../figures/queue-discrete-time-stability.pdf')
# block unstable


# block meanvar
labda, mu, L0 = 6, 6.1, 0
num = 5000

a = rng.integers(labda - 1, labda + 2, size=num)
c = int(mu) * np.ones_like(a)
p = mu - int(mu)  # fractional part of mu
c += rng.binomial(1, p, size=len(c))
print(c.mean())  # just a check
L = compute_queue_length(a, c, L0)
print(L.mean(), L.var())
# block meanvar


# block makecsv
def make_vector_mean_scv(mu, scv, num):
    """Make a vector of length num with a given mean mu
    and square coefficient of variation scv"""
    p = 1 / (1 + scv)
    b = mu / p
    vec = b * rng.binomial(1, p, size=num)
    return vec


# block makecsv


# block constant
num = 5000
x = np.arange(0, num)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey=True)

ax1.set_title("Constant arrivals")
ax1.set_xlabel("time")
ax1.set_ylabel("Queue length")
a = make_vector_mean_scv(mu=6, scv=0, num=num)  # constant arrivals
for scv in (2, 1, 0.5):
    c = make_vector_mean_scv(mu=6.5, scv=scv, num=num)
    L = compute_queue_length(a, c, L0=0)
    ax1.plot(x, L, linewidth=0.5, label=f"$c^2 = {scv}$")
ax1.legend()
# block constant

# block variable
ax2.set_xlabel("time")
ax2.set_title("Both variable")
for scv in (2, 1, 0.5):
    a = make_vector_mean_scv(mu=6, scv=scv, num=num)
    c = make_vector_mean_scv(mu=6.5, scv=scv, num=num)
    L = compute_queue_length(a, c, L0=0)
    ax2.plot(x, L, linewidth=0.5, label=f"$c^2 = {scv}$")

fig.tight_layout()
fig.savefig('../figures/queue-discrete-time-scv.pdf')
# block variable


# block tandem
def compute_queue_length(a, c, L0=0):
    """Compute departures d and queue length L for
    arrivals a,  capacities c and starting level L0"""
    L = np.zeros_like(a)
    d = np.zeros_like(a)
    L[0] = L0
    for i in range(1, len(a)):
        d[i] = min(c[i], L[i - 1] + a[i])
        L[i] = L[i - 1] + a[i] - d[i]
    return d, L  # return d too


# block tandem


# block simtandem
labda, L0 = 6, 0
num = 5000
x = np.arange(num)

rng = np.random.default_rng(3)
a = rng.integers(labda - 1, labda + 2, size=num)
c1 = rng.poisson(labda + 1, num)
c2 = rng.poisson(labda + 0.1, num)
c3 = rng.poisson(labda + 1.5, num)
c4 = rng.poisson(labda + 1.0, num)
# block simtandem

# block figtandem
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlabel("time")
ax.set_ylabel("Queue length")

d1, L1 = compute_queue_length(a, c1)
ax.plot(x, L1, linewidth=0.5, label="Q1")

d2, L2 = compute_queue_length(d1, c2)
ax.plot(x, L2, linewidth=0.5, label="Q2")

d3, L3 = compute_queue_length(d2, c3)
ax.plot(x, L3, linewidth=0.5, label="Q3")

d4, L4 = compute_queue_length(d3, c4)
ax.plot(x, L4, linewidth=0.5, label="Q4")

plt.legend()
fig.tight_layout()
fig.savefig('../figures/queue-discrete-network.pdf')
# block figtandem
