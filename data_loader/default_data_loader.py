import pickle
from data_loader.CPCDataset import CPCDataset
from data_loader.SimCLRDataset import SimCLRDataset
from data_loader.TPNDataset import TPNDataset


class DefaultDataLoader:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.load_dataset()

    def load_dataset(self):
        with open(self.cfg.train_dataset_path, "rb") as f:
            self.train_dataset = pickle.load(f)
            # if len(self.train_dataset) > 15000:
            #     indices = random.sample(range(len(self.train_dataset)), 15000)
            #     self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)
        with open(self.cfg.test_dataset_path, "rb") as f:
            self.test_dataset = pickle.load(f)
        with open(self.cfg.val_dataset_path, "rb") as f:
            self.val_dataset = pickle.load(f)

    def get_datasets(self):
        # if self.cfg.dataset_name == "wesad" or self.cfg.dataset_name == "ninaprodb5" or self.cfg.dataset_name == "opportunity":
        #     reduce_augs = True
        # else:
        #     reduce_augs = False
        reduce_augs = True
        if self.cfg.pretext == "tpn" or self.cfg.pretext == "metatpn":
            train_dataset = TPNDataset(self.train_dataset, reduce_augs=reduce_augs)
            val_dataset = TPNDataset(self.val_dataset, reduce_augs=reduce_augs)
            test_dataset = TPNDataset(self.test_dataset, reduce_augs=reduce_augs)
        if (
            self.cfg.pretext == "cpc"
            or self.cfg.pretext == "metacpc"
            or self.cfg.pretext == "autoencoder"
            or self.cfg.pretext == "metaautoencoder"
        ):
            train_dataset = CPCDataset(self.train_dataset)
            val_dataset = CPCDataset(self.val_dataset)
            test_dataset = CPCDataset(self.test_dataset)
        if (
            self.cfg.pretext == "simclr"
            or self.cfg.pretext == "metasimclr"
            or self.cfg.pretext == "simsiam"
            or self.cfg.pretext == "metasimsiam"
            or self.cfg.pretext == "setsimclr"
        ):
            train_dataset = SimCLRDataset(self.train_dataset, reduce_augs=reduce_augs)
            val_dataset = SimCLRDataset(self.val_dataset, reduce_augs=reduce_augs)
            test_dataset = SimCLRDataset(self.test_dataset, reduce_augs=reduce_augs)

        return train_dataset, val_dataset, test_dataset
