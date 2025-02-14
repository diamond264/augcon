import pickle
from data_loader.CPCDataset import CPCDataset
from data_loader.SimCLRDataset import SimCLRDataset
from data_loader.TPNDataset import TPNDataset


class DefaultDataLoader: # data loader uses load_dataset method 
    def __init__(self, cfg):
        self.cfg = cfg
        self.test_dataset = self.load_dataset()

    def load_dataset(self):
        with open(self.cfg.test_dataset_path, "rb") as f:
            test_dataset = pickle.load(f)
        return test_dataset

    def get_datasets(self):
        reduce_augs = True
        test_dataset = None
        if self.cfg.pretext == "tpn" or self.cfg.pretext == "metatpn":
            test_dataset = TPNDataset(self.test_dataset, reduce_augs=reduce_augs)
        if (
            self.cfg.pretext == "cpc"
            or self.cfg.pretext == "metacpc"
            or self.cfg.pretext == "autoencoder"
            or self.cfg.pretext == "metaautoencoder"
        ):
            test_dataset = CPCDataset(self.test_dataset)
        if (
            self.cfg.pretext == "simclr"
            or self.cfg.pretext == "metasimclr"
            or self.cfg.pretext == "simsiam"
            or self.cfg.pretext == "metasimsiam"
            or self.cfg.pretext == "setsimclr"
        ):
            test_dataset = SimCLRDataset(self.test_dataset, reduce_augs=reduce_augs)

        return test_dataset
