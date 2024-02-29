"""The effect of priority service in multi-server queues."""
from collections import defaultdict

import numpy as np

import mmc
from events import Events
from priority_job import PriorityJob
from queues import Queue
from server import Server
from servers import Servers
from simulator import Simulation
from stats import Statistics


def priority_jobs():
    num = 100000
    labda, mu = 3, 3.2

    rng = np.random.default_rng(3)
    X = rng.exponential(scale=1 / labda, size=num)
    X[0] = 0
    A = X.cumsum()
    loads = rng.exponential(scale=1 / mu, size=num)
    priorities = rng.binomial(1, 0.1, size=num)
    jobs = []
    for i in range(1, num):
        job = PriorityJob(id=i, arrival_time=A[i], load=loads[i])
        job.priority = priorities[i]
        jobs.append(job)

    servers = Servers()
    for i, rate in enumerate([1]):
        servers.push(Server(id=i, rate=rate))

    sim = Simulation(Events(), Queue(), Statistics(), servers)
    sim.initialize_jobs(jobs)
    sim.run()

    mm1 = mmc.MMC(labda, mu, c=1)
    print("W: ", sim.stats.mean_waiting_time(), mm1.EW)
    print("J: ", sim.stats.mean_sojourn_time(), mm1.EJ)
    print("Q: ", sim.stats.mean_queue_length(), mm1.EQ)

    wait = defaultdict(float)
    nums = defaultdict(float)
    for j in sim.stats:
        nums[j.priority] += 1
        wait[j.priority] += j.waiting_time
    for k in sorted(nums.keys()):
        print(f"Priority: {k}, EW: {wait[k] / nums[k]:.2f}")


priority_jobs()
