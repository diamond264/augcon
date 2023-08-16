import os
import time
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from PIL import ImageFile
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, ConcatDataset

from core.image_loader.SimCLRLoader import SimCLRLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

class DomainNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, logger, file_path):
        st = time.time()
        self.cfg = cfg
        self.logger = logger
        self.file_path = file_path

        self.pre_transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.dataset = []
        self.domains = []
        self.domain_labels = []
        self.indices_by_domain = defaultdict(list)
        self.logger.info(f"Loading dataset from {file_path}")
        self.loader = self.get_loader()
        self.preprocessing()
        self.logger.info(f"Preprocessing took {time.time() - st} seconds")

    def get_loader(self):
        loader = transforms.Compose([
            self.pre_transform,
            self.post_transform
        ])
        if self.cfg.mode == 'pretrain':
            if self.cfg.pretext == 'simclr' or \
               self.cfg.pretext == 'simsiam' or \
               self.cfg.pretext == 'metasimsiam':
                loader = SimCLRLoader(pre_transform=self.pre_transform,
                                    post_transform=self.post_transform)
        return loader

    def preprocessing(self):
        cnt = 0
        for i, domain in enumerate(self.cfg.domains):
            dataset = ImageFolder(os.path.join(self.file_path, domain), self.loader)
            
            # ##### FOR TESTING PURPOSES #####
            # limit = 15000
            # if len(dataset) > limit and self.cfg.mode == 'pretrain':
            #     dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), limit, replace=False))
            
            self.dataset = ConcatDataset([self.dataset, dataset])
            self.domain_labels.extend([i] * len(dataset))
            self.domains.append(i)
            self.indices_by_domain[i] = np.arange(len(dataset))+cnt
            cnt += len(dataset)
        
        self.domain_labels = np.array(self.domain_labels)
        self.domain_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.domain_labels))

    def get_indices_by_domain(self):
        return self.indices_by_domain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feature, class_label = self.dataset[idx]
        domain_label = self.domain_labels[idx][0]
        return feature, class_label, domain_label