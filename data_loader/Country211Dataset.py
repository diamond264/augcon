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

from core.image_loader.PosPairLoader import PosPairLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
_PACS_SIZE = (224, 224)
_PACS_MEAN = [0.485, 0.456, 0.406]
_PACS_STDDEV = [0.229, 0.224, 0.225]


class Country211Dataset(torch.utils.data.Dataset):
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
        else:
            assert (0)

        self.pre_transform = transforms.Compose([
            transforms.Resize(_PACS_SIZE)
        ])
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_PACS_MEAN,
                                 std=_PACS_STDDEV)
        ])

        self.dataset = []
        self.domains = []
        self.domain_labels = []
        self.indices_by_domain = defaultdict(list)
        self.logger.info(f"Loading dataset from {self.file_path}")
        self.preprocessing()
        self.logger.info(f"Preprocessing took {time.time() - st} seconds")

    def get_loader(self, rand_aug=False):
        # return x
        loader = transforms.Compose([
            self.pre_transform,
            self.post_transform
        ])
        # return k, q
        if self.cfg.mode == 'pretrain':
            loader = PosPairLoader(pre_transform=self.pre_transform,
                                   post_transform=self.post_transform,
                                   rand_aug=rand_aug)
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

        loader = self.get_loader(self.cfg.rand_aug)
        dataset = ImageFolder(self.file_path, loader)

        if self.cfg.mode == 'finetune':
            dataset = self.filter_nway_kshot(dataset, self.cfg.n_way, self.cfg.k_shot)

        self.dataset = ConcatDataset([self.dataset, dataset])
        self.domain_labels.extend([0] * len(dataset))
        self.domains.append(0)
        self.indices_by_domain[0] = np.arange(len(dataset)) + cnt
        cnt += len(dataset)

        self.domain_labels = np.array(self.domain_labels)
        self.domain_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.domain_labels))

    def filter_nway_kshot(self, dataset, n_way, k_shot):
        if self.label_dict == None:
            all_labels = np.array(dataset.classes)
            filtered_labels = np.random.choice(all_labels, n_way, replace=False)
            self.label_dict = {label: i for i, label in enumerate(filtered_labels)}
        print(f'selected labels: {self.label_dict}')

        filtered_dataset = []
        for label in self.label_dict.keys():
            label_to_num = {l: i for i, l in enumerate(dataset.classes)}
            num_to_label = {i: l for i, l in enumerate(dataset.classes)}
            label_num = label_to_num[label]


            indices = np.array(dataset.targets)
            indices = np.where(indices == label_num)[0]
            if self.type == 'train':
                if len(indices) < k_shot: continue
            else:
                k_shot = len(indices)
            indices = np.random.choice(indices, k_shot, replace=False)

            new_dataset = torch.utils.data.Subset(dataset, indices)
            new_label = self.label_dict[label]

            if self.cfg.supervised_adaptation and self.type != 'test':
                xs, qs, ks = [], [], []
                for feature, _ in new_dataset:
                    xs.append(feature[0])
                    qs.append(feature[1])
                    ks.append(feature[2])

                random.shuffle(ks)
                new_dataset = []
                new_dataset = [([xs[i], qs[i], ks[i]], new_label) for i in range(len(xs))]
            else:
                new_dataset = [(feature, new_label) for feature, _ in new_dataset]

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