#!/usr/bin/env python
import psutil
import time
from util.args import parse_args
from util.config import Config

from data_loader.default_data_loader import DefaultDataLoader

# 1D meta pretext tasks
from core.MetaCPC import MetaCPCLearner
from core.MetaSimCLR1D import MetaSimCLR1DLearner
from core.MetaTPN import MetaTPNLearner


class Experiment:
    def __init__(self, config):
        self.cfg = config

    def run(self):
        # Model creation
        learner = None

        if self.cfg.pretext == "metacpc":
            learner = MetaCPCLearner(self.cfg)
        elif self.cfg.pretext in ["metasimclr", "setsimclr"]:
            learner = MetaSimCLR1DLearner(self.cfg)
        elif self.cfg.pretext == "metatpn":
            learner = MetaTPNLearner(self.cfg)

        default_data_loader = DefaultDataLoader(self.cfg)
        test_dataset = default_data_loader.get_datasets()
        learner.run(test_dataset)


if __name__ == "__main__":
    for i in range(10):
        prefix = "Idle"
        cpu_usage = psutil.cpu_percent()  # CPU usage in %
        ram_usage = psutil.virtual_memory().used / 1e6  # RAM usage in MB
        print(f"{prefix} CPU Usage: {cpu_usage}%")
        print(f"{prefix} RAM Usage: {ram_usage:.2f} MB")
        time.sleep(0.5)
    args = parse_args()
    cfg = Config(args.config)
    exp = Experiment(cfg)
    exp.run()
