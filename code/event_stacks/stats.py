class Statistics(list):
    def push(self, job):
        self.append(job)

    def num_jobs(self):
        return len(self)

    def mean_waiting_time(self):
        tot = sum(job.waiting_time for job in self)
        return tot / self.num_jobs()

    def mean_sojourn_time(self):
        tot = sum(job.sojourn_time for job in self)
        return tot / self.num_jobs()

    def mean_queue_length(self):
        tot = sum(job.queue_length for job in self)
        return tot / self.num_jobs()

    def mean_servers_free(self):
        tot = sum(job.free_servers for job in self)
        return tot / self.num_jobs()
