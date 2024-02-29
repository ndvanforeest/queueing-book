from dataclasses import dataclass

from server import Server


@dataclass
class Job:
    id: int
    arrival_time: float
    load: float
    service_time: float = 0
    departure_time: float = 0
    queue_length: int = 0
    free_servers: int = 0
    server: Server = None

    @property
    def sojourn_time(self):
        return self.departure_time - self.arrival_time

    @property
    def waiting_time(self):
        return self.sojourn_time - self.service_time

    def __repr__(self):
        return (
            f"{self.id},  {self.load}, {self.service_time},"
            f"{self.arrival_time}, {self.departure_time}"
        )

    def __hash__(self):
        return self.id

    def __lt__(self, other):
        return self.arrival_time < other.arrival_time
