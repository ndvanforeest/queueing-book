from dataclasses import dataclass


@dataclass
class Server:
    id: int
    rate: float

    def __lt__(self, other):
        return self.rate > other.rate
