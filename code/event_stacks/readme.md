* Queueing simulations with event stacks

Event stacks are the soul of discrete event simulations.
In this project we demonstrate how to use this concept to a simulate of a multi server queue and a  queue that serves jobs based on priority.

- Events.py contains the different types of events and the event stack that is implemented as heapq.
- job.py implements the regular job class and a priority job
- myQueue.py has two different queue types. One is a regular deque, the other derives from heapq to enable priority queueing
- servers.py has two different server types: one derives from deque, the other from heapq to be able to select servers based on some specific rule
- simulator.py contains the simulation class
- stats.py is for the statistics computations as the end of the simulation


The following files contains tests and experiments, and show how to run the simulations
- test_mmc.py compares the result of the simulation to the exact results that we compute for M/M/c queue
- multiple_speeds.py does experiments with servers with different speeds
- priority_jobs.py does experiments for a single server queue but with jobs with different priorities
