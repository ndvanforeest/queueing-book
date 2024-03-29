import numpy as np

from random_variable import RV

labda, mu = 1, 3
rho = labda / mu
f = {1: 1, 2: 1, 3: 1}
S = RV(f)


def norm_pi(pi):
    norm = sum(pi[n] for n in pi.keys())
    return {n: pi[n] / norm for n in pi.keys()}


def complete_rejection(K):
    pi, n = {}, 0
    pi[0] = 1
    while n < K:
        pi[n + 1] = sum(
            pi[m] * (S.sf(n - m) - S.sf(K - m)) for m in range(n + 1)
        )
        pi[n + 1] *= rho
        n += 1
    return RV(pi)


def partial_acceptance(K):
    pi, n = {}, 0
    pi[0] = 1
    while n < K:
        pi[n + 1] = sum(pi[m] * S.sf(n - m) for m in range(n + 1))
        pi[n + 1] *= rho
        n += 1
    return RV(pi)


def complete_acceptance(K):
    pi, n = {}, 0
    pi[0] = 1
    while S.sf(n - K) > 0:
        pi[n + 1] = sum(pi[m] * S.sf(n - m) for m in range(min(n, K) + 1))
        pi[n + 1] *= rho
        n += 1
    return RV(pi)


pi = complete_rejection(5)
print(pi.mean())
pi = partial_acceptance(5)
print(pi.mean())
pi = complete_acceptance(5)
print(pi.mean())
