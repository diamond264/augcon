import os
import time
import torch
import numpy as np

from tqdm import tqdm
from PIL import ImageFile
from torchvision import transforms
from torchvision.datasets import ImageFolder

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

        self.features = []
        self.class_labels = []
        self.domain_labels = []
        self.logger.info(f"Loading dataset from {file_path}")
        self.preprocessing(100)
        self.logger.info(f"Preprocessing took {time.time() - st} seconds")

    def preprocessing(self, limit_per_domain=None):
        for i, domain in enumerate(self.cfg.domains):
            if self.cfg.pretext == 'simclr':
                loader = SimCLRLoader(pre_transform=self.pre_transform,
                                      augmentations=self.cfg.augmentations,
                                      post_transform=self.post_transform)
            else:
                loader = transforms.Compose([
                    self.pre_transform,
                    self.post_transform
                ])
            
            path = os.path.join(self.file_path, domain)
            dataset = ImageFolder(path, transform=loader)

            self.logger.info(f"- Loading {domain} dataset ({i}/{len(self.cfg.domains)})")
            cnt = 0
            for (feature, classidx) in tqdm(dataset):
                self.features.append(feature)
                self.class_labels.append(int(classidx))
                self.domain_labels.append(i)
                cnt += 1
                if limit_per_domain is not None:
                    if cnt >= limit_per_domain:
                        break

        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)
        self.class_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.class_labels))
        self.domain_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.domain_labels))

    def __len__(self):
        return len(self.features)

    def get_num_domains(self):
        return len(self.domains)

    def __getitem__(self, idx):
        return self.features[idx], self.class_labels[idx][0], self.domain_labels[idx][0]
