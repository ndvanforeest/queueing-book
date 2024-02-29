"""M/M/c queue.  """


import math
from dataclasses import dataclass


@dataclass
class MMC:
    labda: float
    mu: float
    c: int

    @property
    def rho(self):
        return self.labda / self.mu / self.c

    @property
    def ES(self):
        return 1 / self.mu

    @property
    def G(self):
        c, rho = self.c, self.rho
        res = sum((c * rho) ** n / math.factorial(n) for n in range(c))
        res += (c * rho) ** c / (1 - rho) / math.factorial(c)
        return res

    @property
    def EQ(self):
        res = (self.c * self.rho) ** self.c / math.factorial(self.c) / self.G
        res *= self.rho / (1 - self.rho) ** 2
        return res

    @property
    def ELs(self):
        return self.labda / self.mu

    @property
    def EL(self):
        return self.ELs + self.EQ

    @property
    def EW(self):
        return self.EQ / self.labda

    @property
    def EJ(self):
        return self.EL / self.labda


def test():
    """Compare with the results for the MM1 queue"""
    labda, mu = 3, 4
    rho = labda / mu
    mm1 = MMC(labda, mu, c=1)

    assert math.isclose(mm1.rho, rho)

    mm1_mean_sojourn_time = 1 / mu / (1 - rho)
    assert math.isclose(mm1.EJ, mm1_mean_sojourn_time)

    mm1_mean_waiting_time = rho / mu / (1 - rho)
    assert math.isclose(mm1.EW, mm1_mean_waiting_time)

    mm1_mean_queue_length = labda * mm1_mean_waiting_time
    assert math.isclose(mm1.EQ, mm1_mean_queue_length)

    mm1_mean_system_length = labda * mm1_mean_sojourn_time
    assert math.isclose(mm1.EL, mm1_mean_system_length)


if __name__ == '__main__':
    test()
