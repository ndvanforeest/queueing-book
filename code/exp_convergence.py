import matplotlib.pyplot as plt
from latex_figures import fig_in_latex_format

import random_variable as rv


# block modfunction
def mod(a, b):
    """Compute a modulo b where a can also be a RV."""
    match a:
        case int():
            return a % b
        case float():
            return a % b
        case rv.RV():
            return rv.apply_function(lambda x: x % b, a)
        case _:
            raise ValueError("Unknown type passed to mod")


# block modfunction


# block minsmaxs
step = rv.RV({-1: 1 / 3, 0: 1 / 3, 1: 1 / 3})


def minmax(cycle, num):
    position = rv.RV({0: 1})
    mins, maxs = [], []
    for i in range(num):
        position = mod(position + step, cycle)
        pmf = [position.pmf(k) for k in range(cycle)]
        mins.append(min(pmf))
        maxs.append(max(pmf))
    return mins, maxs


# block minsmaxs


fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(6, 3))

num = 30
xx = range(num)


# block onefig
cycle = 5
mins, maxs = minmax(cycle, num)
ax1.plot(xx, maxs, "x", ms=2, ls=":", lw=1, label="Maxs")
ax1.plot(xx, mins, "o", ms=2, ls=":", lw=1, label="Mins")
ax1.set_title("Cycle 5")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Probability")
ax1.legend()
# block onefig

cycle = 10
mins, maxs = minmax(cycle, num)
ax2.plot(xx, maxs, "x", ms=2, ls=":", lw=1, label="Maxs")
ax2.plot(xx, mins, "o", ms=2, ls=":", lw=1, label="Mins")
ax2.set_title("Cycle 10")
ax2.set_xlabel("Iteration")
ax2.legend()


fig.tight_layout()
fig.savefig("../figures/exp_convergence.pdf")
