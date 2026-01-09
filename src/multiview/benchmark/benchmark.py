from typing import List
# from omegaconf import DictConfig
# from multiview.benchmark.task import Task


class Benchmark:
    def __init__(self, tasks):
        self.tasks = tasks

    def evaluate(self, method_configs: List[dict]):
        return
