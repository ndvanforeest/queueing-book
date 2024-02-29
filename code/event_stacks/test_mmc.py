"""This compares the result of the simulation of a multi-server queue
 to the exact results of an M/M/c queue.
"""

import numpy as np

import mmc
from events import Events
from job import Job
from queues import Queue
from server import Server
from servers import Servers
from simulator import Simulation
from stats import Statistics


def test_mmc():
    num = 10000
    labda, mu = 3, 4

    rng = np.random.default_rng(3)
    X = rng.exponential(scale=1 / labda, size=num)
    X[0] = 0
    A = X.cumsum()
    loads = rng.exponential(scale=1 / mu, size=num)
    jobs = [Job(id=i, arrival_time=A[i], load=loads[i]) for i in range(1, num)]

    servers = Servers()
    for i, rate in enumerate([1, 1]):
        servers.push(Server(id=i, rate=rate))

    sim = Simulation(Events(), Queue(), Statistics(), servers)
    sim.initialize_jobs(jobs)
    sim.run()

    mm2 = mmc.MMC(labda, mu, c=2)
    print("W: ", sim.stats.mean_waiting_time(), mm2.EW)
    print("J: ", sim.stats.mean_sojourn_time(), mm2.EJ)
    print("Q: ", sim.stats.mean_queue_length(), mm2.EQ)


test_mmc()
