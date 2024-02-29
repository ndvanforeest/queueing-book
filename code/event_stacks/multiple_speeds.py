"""Simulate a multi-server queue with different speeds and compare
to an M/M/C queue."""
from collections import Counter

import numpy as np

import mmc
from events import Events
from job import Job
from queues import Queue
from server import Server
from servers import Servers
from simulator import Simulation
from stats import Statistics


def multiple_speeds():
    num = 10000
    labda, mu = 3, 3.1

    rng = np.random.default_rng(3)
    X = rng.exponential(scale=1 / labda, size=num)
    X[0] = 0
    A = X.cumsum()
    loads = rng.exponential(scale=1 / mu, size=num)
    jobs = [Job(id=i, arrival_time=A[i], load=loads[i]) for i in range(1, num)]

    servers = Servers()
    for i, rate in enumerate([1, 0.1, 0.01]):
        servers.push(Server(id=i, rate=rate))

    sim = Simulation(Events(), Queue(), Statistics(), servers)
    sim.initialize_jobs(jobs)
    sim.run()

    mm1 = mmc.MMC(labda, mu, c=1)
    print("W: ", sim.stats.mean_waiting_time(), mm1.EW)
    print("J: ", sim.stats.mean_sojourn_time(), mm1.EJ)
    print("Q: ", sim.stats.mean_queue_length(), mm1.EQ)

    count = Counter([j.free_servers for j in sim.stats])
    print(count)


multiple_speeds()
