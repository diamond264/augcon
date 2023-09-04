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
from torch.utils.data import Dataset, ConcatDataset, Subset

from core.image_loader.PosPairLoader import PosPairLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
_DIGIT5_RESIZE = (32, 32)
_DIGIT5_MEAN = [0.5, 0.5, 0.5]
_DIGIT5_STDDEV = [0.5, 0.5, 0.5]

_TRAIN_NUM = int(25000 * 0.7)
_TEST_NUM = 9000

class DigitFiveDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, logger, type='train', label_dict=None):
        st = time.time()
        self.cfg = cfg
        self.logger = logger
        self.type = type
        self.label_dict = label_dict
        
        if self.type == 'train':
            self.file_path = self.cfg.train_dataset_path
        elif self.type == 'test':
            self.file_path = self.cfg.test_dataset_path
        elif self.type == 'val':
            self.file_path = self.cfg.val_dataset_path
        else: assert(0)

        self.pre_transform = transforms.Compose([
            transforms.Resize(_DIGIT5_RESIZE)
        ])
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_DIGIT5_MEAN,
                                 std=_DIGIT5_STDDEV)
        ])

        self.dataset = []
        self.domains = []
        self.domain_labels = []
        self.indices_by_domain = defaultdict(list)
        self.logger.info(f"Loading dataset from {self.file_path}")
        self.loader = self.get_loader()
        self.preprocessing()
        self.logger.info(f"Preprocessing took {time.time() - st} seconds")

    def get_loader(self):
        # return x
        loader = transforms.Compose([
            self.pre_transform,
            self.post_transform
        ])
        # return k, q
        if self.cfg.mode == 'pretrain':
            loader = PosPairLoader(pre_transform=self.pre_transform,
                                      post_transform=self.post_transform)
        # return x, k, q
        elif self.cfg.mode == 'finetune' and not self.type == 'test':
            if 'meta' in self.cfg.pretext:
                loader = PosPairLoader(pre_transform=self.pre_transform, 
                                       post_transform=self.post_transform, 
                                       return_original=True)
        return loader

    def get_label_dict(self):
        return self.label_dict

    def preprocessing(self):
        if self.cfg.mode == 'finetune' and len(self.cfg.domains) > 1:
            print("Finetuning on multiple domains is not supported yet")
            assert (0)

        cnt = 0
        for i, domain in enumerate(self.cfg.domains):
            dataset = ImageFolder(os.path.join(self.file_path, domain), self.loader)

            if self.cfg.down_sample:
                if self.type == "train" and len(dataset) > _TRAIN_NUM:
                    # For downsampling
                    target_len = _TRAIN_NUM
                    random_indices = torch.randperm(len(dataset))[:target_len]
                    dataset = torch.utils.data.Subset(dataset, random_indices)

                elif self.type == "test" and len(dataset) > _TEST_NUM:
                    # For downsampling
                    target_len = _TEST_NUM
                    random_indices = torch.randperm(len(dataset))[:target_len]
                    dataset = torch.utils.data.Subset(dataset, random_indices)

            if self.cfg.mode == 'finetune':
                dataset = self.filter_nway_kshot(dataset, self.cfg.n_way, self.cfg.k_shot)

            self.dataset = ConcatDataset([self.dataset, dataset])
            self.domain_labels.extend([i] * len(dataset))
            self.domains.append(i)
            self.indices_by_domain[i] = np.arange(len(dataset)) + cnt
            cnt += len(dataset)
        
        self.domain_labels = np.array(self.domain_labels)
        self.domain_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.domain_labels))

    def filter_nway_kshot(self, dataset, n_way, k_shot):
        if self.label_dict == None:
            all_labels = np.array(dataset.classes)
            filtered_labels = np.random.choice(all_labels, n_way, replace=False)
            self.label_dict = {label: i for i, label in enumerate(filtered_labels)}
        print(f'selected labels: {self.label_dict.keys()}')
        
        filtered_dataset = []
        for label in self.label_dict.keys():
            label_to_num = {l: i for i, l in enumerate(dataset.classes)}
            num_to_label = {i: l for i, l in enumerate(dataset.classes)}
            label_num = label_to_num[label]
            
            indices = np.array(dataset.targets)
            indices = np.where(indices == label_num)[0]
            if self.type == 'train':
                if len(indices) < k_shot: continue
            else: k_shot = len(indices)
            indices = np.random.choice(indices, k_shot, replace=False)
            
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