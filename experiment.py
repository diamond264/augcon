#!/usr/bin/env python
import psutil
import yaml
import time
import argparse

from data_loader.default_data_loader import DefaultDataLoader

# 1D meta pretext tasks
from core.MetaCPC import MetaCPCLearner
from core.MetaSimCLR1D import MetaSimCLR1DLearner
from core.MetaTPN import MetaTPNLearner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()
    return args


class Config:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        for key, value in config_dict.items():
            self.set_config(key, value)

    def set_config(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                getattr(self, key).set_config(k, v)
        else:
            setattr(self, key, value)


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
        ram_usage += psutil.swap_memory().used / 1e6  # Swap usage in MB
        print(f"{prefix} CPU Usage: {cpu_usage}%")
        print(f"{prefix} RAM Usage: {ram_usage:.2f} MB")
        time.sleep(0.5)
    args = parse_args()
    cfg = Config(args.config)
    exp = Experiment(cfg)
    exp.run()
