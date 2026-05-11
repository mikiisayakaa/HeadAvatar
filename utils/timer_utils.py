import torch
from contextlib import contextmanager

class CudaTimer:
    def __init__(self):
        self.times = {}

    @contextmanager
    def record(self, name):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        self.times[name] = self.times.get(name, 0.0) + elapsed

    def reset(self):
        self.times.clear()

    def summary(self):
        total = sum(self.times.values())
        print("|".join([f"{k}: {v:.2f}ms\n" for k, v in self.times.items()]) + f" | Total: {total:.2f}ms" + "\n")