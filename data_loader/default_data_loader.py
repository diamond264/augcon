import torch
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
        if self.cfg.dataset_name == "wesad" or self.cfg.dataset_name == "ninaprodb5" or self.cfg.dataset_name == "opportunity":
            reduce_augs = True
        else:
            reduce_augs = False
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
            # ##### For Shepherd - Noisy in Domain Label Experiment ###########
            if self.cfg.noisy_level != 0:
                self.add_noise_on_domain_label()

            train_dataset = SimCLRDataset(self.train_dataset, reduce_augs=reduce_augs)
            val_dataset = SimCLRDataset(self.val_dataset, reduce_augs=reduce_augs)
            test_dataset = SimCLRDataset(self.test_dataset, reduce_augs=reduce_augs)

        return train_dataset, val_dataset, test_dataset

    def add_noise_on_domain_label(self):
        # Extract unique domain values
        unique_domains = sorted({domain.item() for _, _, domain in self.train_dataset})
        domain_map = {d: unique_domains[(i + 1) % len(unique_domains)] for i, d in enumerate(unique_domains)}
        num_samples = int(len(self.train_dataset) * self.cfg.noisy_level)

        # Convert dataset to a list of tuples for modification
        new_data = [tuple(data) for data in self.train_dataset]  # Unpack dataset

        # Modify only the first `num_samples` domains
        for i in range(num_samples):
            _, _, domain = new_data[i]  # Unpack data point
            domain = domain.item()  # Convert tensor to int

            # Shift domain
            domain = domain_map[domain]
            domain = torch.tensor(domain)  # Convert back to tensor

            # Update dataset
            new_data[i] = (new_data[i][0], new_data[i][1], domain)

        # Convert back to a TensorDataset
        self.train_dataset = torch.utils.data.TensorDataset(*map(torch.stack, zip(*new_data)))
    ##################################################################
