from heapq import heappop, heappush


class Queue(list):
    def pop(self):
        return heappop(self)

    def push(self, job):
        heappush(self, job)

    def length(self):
        return len(self)

    def is_empty(self):
        return self.length() == 0
