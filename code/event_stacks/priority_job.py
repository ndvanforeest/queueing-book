from dataclasses import dataclass

from job import Job
from server import Server


@dataclass(eq=False)
class PriorityJob(Job):
    priority: int = 0

    def __lt__(self, other):
        if self.priority == other.priority:
            return self.arrival_time < other.arrival_time
        else:
            return self.priority < other.priority
