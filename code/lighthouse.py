# block modules
from functools import cache
import numpy as np

import random_variable as rv

# block modules


# block basestock
class Basestock:
    def __init__(self, D, L, h, b):
        self.D = D
        self.L = L
        self.h = h
        self.b = b
        self.X = sum(self.D for i in range(self.L))

    @cache
    def IP(self, S):
        return S

    @cache
    def IL(self, S):
        return self.IP(S) - self.X

    @cache
    def Imin(self, S):
        return rv.apply_function(lambda x: max(0, -x), self.IL(S))

    @cache
    def Iplus(self, S):
        return rv.apply_function(lambda x: max(0, x), self.IL(S))

    def cost(self, S):
        return self.b * self.Imin(S).mean() + self.h * self.Iplus(S).mean()

    def alpha(self, S):
        return (self.IL(S) - self.D).sf(-1)

    def beta(self, S):
        m = rv.compose_function(lambda x, y: min(x, y), self.D, self.Iplus(S))
        return m.mean() / self.D.mean()


# block basestock


# block parameters
# Daily demand
D = rv.RV({1: 1 / 6, 2: 1 / 5, 3: 1 / 4, 4: 1 / 8, 5: 11 / 120, 6: 1 / 6})
L = 2
c = 100  # buying price
b = 0.1 * c  # backlog
h = 0.5 * c / 30  # holding
S = 5

base = Basestock(D, L, h, b)
# block parameters

# block basetests
theta = L * D.mean()
assert np.isclose(base.IL(S).mean(), S - theta)
assert np.isclose(
    base.Imin(S).mean(), theta - sum(base.X.sf(j) for j in range(0, S))
)
assert np.isclose(base.alpha(S), (base.X + D).cdf(S))
# block basetests

# block runit
for S in range(-2, 8):
    print(base.Imin(S).mean(), base.Iplus(S).mean(), base.cost(S))
    print(base.alpha(S), base.beta(S))
# block runit


# block qr
class Qr:
    def __init__(self, D, L, h, b):
        self.D = D
        self.L = L
        self.h = h
        self.b = b
        self.X = sum(self.D for i in range(self.L))

    @cache
    def IP(self, Q, r):
        return rv.RV({i: 1 / Q for i in range(r + 1, r + Q + 1)})

    @cache
    def IL(self, Q, r):
        return self.IP(Q, r) - self.X

    @cache
    def Imin(self, Q, r):
        return rv.apply_function(lambda x: max(0, -x), self.IL(Q, r))

    @cache
    def Iplus(self, Q, r):
        return rv.apply_function(lambda x: max(0, x), self.IL(Q, r))

    def cost(self, Q, r):
        return (
            self.b * self.Imin(Q, r).mean() + self.h * self.Iplus(Q, r).mean()
        )

    def alpha(self, Q, r):
        return (self.IL(Q, r) - self.D).sf(-1)

    def beta(self, Q, r):
        m = rv.compose_function(
            lambda x, y: min(x, y), self.D, self.Iplus(Q, r)
        )
        return m.mean() / self.D.mean()


# block qr


# block runqr
qr = Qr(D, L, h, b)
Q, r = 3, 10
print(qr.IL(Q, r).mean())
print((Q + 1) / 2 + r - qr.X.mean())
# block runqr
