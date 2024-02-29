from heapq import heappop, heappush


class Events(list):
    def push(self, event):
        heappush(self, event)

    def pop(self):
        return heappop(self)

    def is_empty(self):
        return len(self) == 0
