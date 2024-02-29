# block modules
import numpy as np
import matplotlib.pyplot as plt

# block modules


# block seaborn
def initialize_plots():
    import seaborn as sns

    rc = {
        "text.usetex": True,
        "font.family": "fourier",
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    sns.set(style="whitegrid", rc=rc)


initialize_plots()
# block seaborn


# block expdata
num_periods = 10
num_patients = 50
rng = np.random.default_rng(2)
# block expdata

# block interarrivals
X = rng.uniform(low=0.9, high=1.1, size=(num_patients, num_periods + 1))
X[:, 0] = 0  # set first column of X to 0
A = X.cumsum(axis=1)
# block interarrivals


# block expfiguresetup
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 3))
ax1, ax2, ax3, ax4 = axes.flatten()
# block expfiguresetup

# block plotuniforms
ax1.set_xlim(0, 11)
for i in range(num_patients):
    x = A[i, 1:]
    y = i * np.ones(num_periods)
    ax1.plot(x, y, ls='None', marker="o", ms=0.2, c="k")
# block plotuniforms

# block plotjitter
for i in range(num_patients):
    x = A[i, 1:]
    y = rng.uniform(-0.5, 0.5, size=num_periods)  # add jitter
    ax1.plot(x, y, ls='None', marker="o", ms=0.2, c="k")
# block plotjitter


# block allarrivals
A = np.sort(A.flatten())
X = A[1:] - A[:-1]
# block allarrivals

# block compare-with-exp
bins = np.linspace(0, 5 / num_patients, 50)
exp_density = num_patients * np.exp(-num_patients * bins)
ax2.hist(X, bins, density=True, color="k")
ax2.plot(bins, exp_density, ":", c="k", lw=0.75, label="pdf")
ax2.legend()
# block compare-with-exp


X = rng.uniform(low=0.9, high=1.1, size=(num_patients, num_periods + 1))
# block uniformize
X[:, 0] = rng.uniform(low=0, high=1, size=num_patients)
# block uniformize
A = X.cumsum(axis=1)

ax3.set_xlim(0, 11)

for i in range(num_patients):
    x = A[i, 1:]
    y = i * np.ones(num_periods)
    ax3.plot(x, y, ls='None', marker="o", ms=0.2, c="k")

for i in range(num_patients):
    x = A[i, 1:]
    y = rng.uniform(-0.5, 0.5, size=num_periods)  # add jitter
    ax3.plot(x, y, ls='None', marker="o", ms=0.2, c="k")

A = np.sort(A.flatten())
X = A[1:] - A[:-1]
bins = np.linspace(0, 5 / num_patients, 50)
ax4.hist(X, bins, density=True, color="k")
exp_density = num_patients * np.exp(-num_patients * bins)
ax4.plot(bins, exp_density, ":", c="k", lw=0.75, label="pdf")
ax4.legend()

# block export-exp-figure
fig.tight_layout()
fig.savefig("../figures/IBD_exponential.pdf")
# block export-exp-figure

quit()

half_year = 100  # days
num_periods = 6
delta = 1
num_days = num_periods * half_year  # simulation duration
latest_day = num_days + half_year + delta
a = np.zeros(latest_day, dtype=int)
a[: half_year + 1] = num_patients

L = a[0]  # initial number of people in queue
c = num_patients
for n in range(1, num_days):
    d = min(L, c)
    L += a[n] - d
    idx = n + half_year + rng.integers(-delta, delta + 1, d)
    np.add.at(a, idx, 1)

pois = [poisson(num_patients).pmf(i) for i in range(2 * num_patients)]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 3))
axes = axes.flatten()

for n in range(num_periods):
    ax = axes[n]
    y = a[n * half_year : (n + 1) * half_year]
    ax.hist(y, density=True, label=f"{n}")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.2)
    ax.set_title(f"Half year {n}")
    ax.plot(pois, ":", c="k")

fig.tight_layout()
fig.savefig("../figures/IBD_poisson.pdf")
