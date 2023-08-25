import os
import time
import torch
import random
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from PIL import ImageFile
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, ConcatDataset

from core.image_loader.SimCLRLoader import SimCLRLoader
from core.image_loader.MetaSimCLRLoader import MetaSimCLRLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

class DomainNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, logger, file_path, type='train', label_dict=None):
        st = time.time()
        self.cfg = cfg
        self.logger = logger
        self.file_path = file_path
        self.type = type
        self.label_dict = label_dict

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
        elif self.cfg.mode == 'finetune' and not self.type == 'test' :
            if self.cfg.pretext == 'metasimsiam':
                loader = MetaSimCLRLoader(pre_transform=self.pre_transform,
                                          post_transform=self.post_transform)
        return loader
    
    def get_label_dict(self):
        return self.label_dict

    def preprocessing(self):
        cnt = 0
        if self.cfg.mode == 'finetune' and len(self.cfg.domains) > 1:
            print("Finetuning on multiple domains is not supported yet")
            assert(0)
        
        for i, domain in enumerate(self.cfg.domains):
            dataset = ImageFolder(os.path.join(self.file_path, domain), self.loader)
            
            # ##### FOR TESTING PURPOSES #####
            # limit = 15000
            # if len(dataset) > limit and self.cfg.mode == 'pretrain':
            #     dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), limit, replace=False))
            # ################################
            if self.cfg.mode == 'finetune':
                dataset = self.filter_nway_kshot(dataset, self.cfg.n_way, self.cfg.k_shot)
            
            self.dataset = ConcatDataset([self.dataset, dataset])
            self.domain_labels.extend([i] * len(dataset))
            self.domains.append(i)
            self.indices_by_domain[i] = np.arange(len(dataset))+cnt
            cnt += len(dataset)
        
        self.domain_labels = np.array(self.domain_labels)
        self.domain_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.domain_labels))

    def filter_nway_kshot(self, dataset, n_way, k_shot):
        if self.label_dict == None:
            all_labels = np.array(dataset.classes)
            filtered_labels = np.random.choice(all_labels, n_way, replace=False)
            # For debugging purposes
            # filtered_labels = all_labels[-n_way:]
            self.label_dict = {label: i for i, label in enumerate(filtered_labels)}
        print(self.label_dict.keys())
        filtered_dataset = []
        for label in self.label_dict.keys():
            label_to_num = {label: i for i, label in enumerate(dataset.classes)}
            num_to_label = {i: label for i, label in enumerate(dataset.classes)}
            label_num = label_to_num[label]
            indices = np.array(dataset.targets)
            indices = np.where(indices == label_num)[0]
            if self.type == 'train':
                if len(indices) < k_shot:
                    continue
            else:
                k_shot = len(indices)
            indices = np.random.choice(indices, k_shot, replace=False)
            # For debugging purposes
            # indices = indices[:k_shot]
            new_dataset = torch.utils.data.Subset(dataset, indices)
            new_dataset = [(feature, self.label_dict[num_to_label[label]]) for feature, label in new_dataset]
            filtered_dataset = ConcatDataset([filtered_dataset, new_dataset])
            
        return filtered_dataset

    def get_indices_by_domain(self):
        return self.indices_by_domain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feature, class_label = self.dataset[idx]
        domain_label = self.domain_labels[idx][0]
        return feature, class_label, domain_label