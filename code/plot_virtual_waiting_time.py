"""Generate the figure in the file for the simulation of continuous time queueing processes.
"""

import heapq

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
tex_fonts = {
    "text.usetex": True,
    "font.family": "fourier",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
plt.rcParams.update(tex_fonts)

rng = np.random.default_rng(3)
labda, mu = 3, 4
num = 10
X = rng.exponential(scale=1 / labda, size=num)
S = rng.exponential(scale=1 / mu, size=num)
X[0] = S[0] = 0
D = np.zeros_like(X)

A = X.cumsum()
for i in range(1, num):
    D[i] = max(A[i], D[i - 1]) + S[i]

J = D - A
W = J - S
# print(J.mean(), J.var())
# print(J.sum() / D[-1])


def At(t, side='right'):
    """
    The right or left continuous version of A(t), depending on SIDE, ,
    A(t) = max {k : A_k \geq t}
    A(t-) = A(t) -1.
    """
    return max(np.searchsorted(A, t, side=side) - 1, 0)


def virtual(t, side='right'):
    """
    The right or left continuous virtual waiting time at t, depending on SIDE
    """
    k = At(t, side)
    return max(J[k] - (t - A[k]), 0)


fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 3), sharex=True)
ax1.set_ylabel("Virtual time $V$")
ax2.set_ylabel("Length $L$")
ax2.set_xlabel("time")
ax2.set_yticks([0, 1, 2, 3, 4, 5])

for i in range(1, num):
    ax1.fill([A[i], A[i], D[i]], [0, J[i], 0], c='green', alpha=0.2)
    ax1.fill([A[i], A[i], A[i] + W[i]], [0, W[i], 0], c='blue', alpha=0.2)


heap = list(heapq.merge(([t, 1] for t in A[1:]), ([t, -1] for t in D[1:])))
heapq.heapify(heap)

length, past = 0, 0
while heap:
    now, delta = heapq.heappop(heap)
    ax1.plot([past, now], [virtual(past), virtual(now, 'left')], c='k', lw=0.75)
    ax1.plot([now, now], [virtual(now, 'left'), virtual(now)], ":", c='k', lw=0.75)
    ax2.plot([past, now], [length, length], c='k', lw=0.75)
    ax2.plot([now, now], [length, length + delta], ":", c='k', lw=0.75)
    length += delta
    past = now


fig.tight_layout()
fig.savefig('../figures/queue-continuous-length.pdf')
