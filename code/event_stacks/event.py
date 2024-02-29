from dataclasses import dataclass

from job import Job


@dataclass
class Event:
    time: float
    job: Job

    def __lt__(self, other):
        return self.time < other.time


class ArrivalEvent(Event):
    pass


class DepartureEvent(Event):
    pass
