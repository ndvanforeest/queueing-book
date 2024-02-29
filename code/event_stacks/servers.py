from heapq import heappop, heappush


class Servers(list):
    def push(self, server):
        heappush(self, server)

    def pop(self):
        return heappop(self)

    def num_free(self):
        return len(self)

    def is_server_available(self):
        return self.num_free() > 0
