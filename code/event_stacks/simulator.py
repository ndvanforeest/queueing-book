from event import ArrivalEvent, DepartureEvent


class Simulation:
    def __init__(self, events, queue, statistics, servers):
        self.events = events
        self.queue = queue
        self.stats = statistics
        self.servers = servers
        self.now = 0

    def initialize_jobs(self, jobs):
        for job in jobs:
            self.events.push(ArrivalEvent(job.arrival_time, job))

    def serve_job(self, job):
        server = self.servers.pop()
        job.server = server
        job.service_time = job.load / server.rate
        job.departure_time = self.now + job.service_time
        job.queue_length = self.queue.length()
        job.free_servers = self.servers.num_free()
        self.events.push(DepartureEvent(job.departure_time, job))

    def run(self):
        while not self.events.is_empty():
            event = self.events.pop()
            self.now, job = event.time, event.job
            if isinstance(event, ArrivalEvent):
                if self.servers.is_server_available():
                    self.serve_job(job)
                else:
                    self.queue.push(job)
            elif isinstance(event, DepartureEvent):
                self.servers.push(job.server)
                self.stats.push(job)
                if not self.queue.is_empty():
                    job = self.queue.pop()
                    self.serve_job(job)
            else:
                raise ValueError("Unknown event")
