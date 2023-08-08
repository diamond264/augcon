import torch.utils.data
import pandas as pd
import os
import time
import pickle
import numpy as np
from collections import defaultdict
# import itertools

class HHARDataset(torch.utils.data.Dataset):
    # load static files
    def __init__(self, file, class_type, domain_type, transform=None,
                 model=None, device=None, user=None, gt=None,
                 complementary=False, seq_len=256,
                 load_cache=False, save_cache=False, cache_path=None,
                 split_ratio=0.8, save_opposite="", fixed_data_path=""):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            gt: condition on action
            user: condition on user
            model: condition on model
            device: condition on device instance
            complementary: is it complementary dataset for given conditions? (used for "multi" case)
        """
        self.seq_len = seq_len
        self.OVERLAPPING_WIN_LEN = self.seq_len // 2
        self.WIN_LEN = self.seq_len
        
        st = time.time()
        
        self.metadata = {
            'users': ['a', 'b', 'c', 'd', 'e', 'f'],
            'models': ['nexus4', 's3', 's3mini', 'lgwatch'],
            'devices': ['lgwatch_1', 'lgwatch_2',
                        'gear_1', 'gear_2'
                        'nexus4_1', 'nexus4_2',
                        's3_1', 's3_2', 's3mini_1', 's3mini_2'],
            'gts': ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown']
        }
        
        self.class_type = class_type
        self.domain_type = domain_type
        self.user = user
        self.model = model
        self.device = device
        self.gt = gt
        self.complementary = complementary
        self.split_ratio = split_ratio
        self.save_opposite = save_opposite
        self.fixed_data_path = fixed_data_path
        
        if os.path.exists(self.fixed_data_path): return
        
        self.df = pd.read_csv(file)
        self.transform = transform
        
        ppt = time.time()
        self.dataset = None
        self.preprocess()

        if load_cache:
            # Load the Tensordataset from the .pkl file
            with open(cache_path, 'rb') as f:
                self.features, self.class_labels, self.domain_labels = pickle.load(f)
                self.dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(self.features).float(),
                    torch.from_numpy(self.class_labels),
                    torch.from_numpy(self.domain_labels))
        else:
            self.df = pd.read_csv(file)
            if complementary:  # for multi domain
                if user: self.df = self.df[self.df['User'] != user]
                if model: self.df = self.df[self.df['Model'] != model]
                if device: self.df = self.df[self.df['Device'] != device]
                if gt: self.df = self.df[self.df['gt'] != gt]
            else:
                if user: self.df = self.df[self.df['User'] == user]
                if model: self.df = self.df[self.df['Model'] == model]
                if device: self.df = self.df[self.df['Device'] == device]
                if gt: self.df = self.df[self.df['gt'] == gt]

            self.transform = transform
            ppt = time.time()

            self.dataset = None
            self.preprocessing()
            
            print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.
                  format(len(self.df.index), time.time() - ppt, time.time() - st))
        
        if save_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump((self.features, self.class_labels, self.domain_labels), f)

    def preprocessing(self):
        self.num_domains = 0
        self.features = []
        self.class_labels = []
        self.domain_labels = []
        
        # TODO: TBI
        # self.kshot_datasets = []  # list of dataset per each domain

        if self.complementary:
            users = set(self.metadata['users'])
            if self.user: users = users - set(self.user)
            models = set(self.metadata['models'])
            if self.model: models = models - set(self.model)
            devices = set(self.metadata['devices'])
            if self.device: devices = devices - set(self.device)
            gts = set(self.metadata['gts'])
            if self.gt: gts = gts - set(self.gt)
        else:
            users = set(self.metadata['users'])
            if self.user: users = set([self.user])
            models = set(self.metadata['models'])
            if self.model: models = set([self.model])
            devices = set(self.metadata['devices'])
            if self.device: devices = set([self.device])
            gts = set(self.metadata['gts'])
            if self.gt: gts = set([self.gt])

        # domain_superset = list(itertools.product(models, users, devices, gts))
        domain_superset = list(set(self.metadata[f'{self.domain_type}s']))
        valid_domains = []

        for idx in range(max(len(self.df) // self.OVERLAPPING_WIN_LEN - 1, 0)):
            user = self.df.iloc[idx * self.OVERLAPPING_WIN_LEN, 9]
            model = self.df.iloc[idx * self.OVERLAPPING_WIN_LEN, 10]
            device = self.df.iloc[idx * self.OVERLAPPING_WIN_LEN, 11]
            gt = self.df.iloc[idx * self.OVERLAPPING_WIN_LEN, 12]
            domain_label = -1
            
            if self.domain_type == 'user': domain = user
            elif self.domain_type == 'model': domain = model
            elif self.domain_type == 'device': domain = device
            elif self.domain_type == 'gt': domain = gt
            
            if self.class_type == 'user': class_label = user
            elif self.class_type == 'model': class_label = model
            elif self.class_type == 'device': class_label = device
            elif self.class_type == 'gt': class_label = gt

            for i in range(len(domain_superset)):
                if domain_superset[i] == domain and domain_superset[i] not in valid_domains:
                    valid_domains.append(domain_superset[i])
                    break

            if domain in valid_domains:
                # domain_label = valid_domains.index(domain)
                domain_label = self.domain_to_number(domain)
            else:
                continue

            feature = self.df.iloc[idx * self.OVERLAPPING_WIN_LEN:(idx + 2) * self.OVERLAPPING_WIN_LEN, 3:6].values
            feature = feature.T

            self.features.append(feature)
            self.class_labels.append(self.class_to_number(class_label))
            self.domain_labels.append(domain_label)

        self.num_domains = len(valid_domains)
        self.features = np.array(self.features, dtype=np.float)
        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)
        
        if self.split_ratio != 0:
            splitted = self.split_pretrain_data(self.features, self.class_labels, self.domain_labels, self.split_ratio, self.save_opposite)
            self.features = splitted[0]
            self.class_labels = splitted[1]
            self.domain_labels = splitted[2]
        
        # TODO: composing this should be out of this function
        self.dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.features).float(),
            torch.from_numpy(self.class_labels),
            torch.from_numpy(self.domain_labels))

        # self.datasets = []  # list of dataset per each domain
        # # append dataset for each domain
        # for domain_idx in range(self.num_domains):
        #     indices = np.where(self.domain_labels == domain_idx)[0]
        #     self.datasets.append(torch.utils.data.TensorDataset(torch.from_numpy(self.features[indices]).float(),
        #                                                         torch.from_numpy(self.class_labels[indices]),
        #                                                         torch.from_numpy(self.domain_labels[indices])))
        #     kshot_dataset= KSHOTTensorDataset(len(np.unique(self.class_labels)),
        #                                                   self.features[indices],
        #                                                   self.class_labels[indices],
        #                                                   self.domain_labels[indices])
        #     self.kshot_datasets.append(kshot_dataset)
        # # concated dataset
        # self.dataset = torch.utils.data.ConcatDataset(self.datasets)
        print('Valid domains:' + str(valid_domains))

    def split_pretrain_data(self, features, class_labels, domain_labels, split_ratio=0.8, save_opposite=""):
        # Create a dictionary to store the indices of each domain
        domain_indices = defaultdict(list)
        for i, domain in enumerate(domain_labels):
            domain_indices[domain].append(i)

        # Split the data by domain
        final_indices = []
        final_opposite_indices = []
        for domain, indices in domain_indices.items():
            split_len = int(len(indices)*split_ratio)
            sampled_indices = np.random.choice(len(indices), size=split_len, replace=False)
            splitted_indices = np.array(indices)[sampled_indices]
            final_indices.extend(splitted_indices)
            
            if save_opposite:
                opposite_indices = np.setdiff1d(np.arange(len(indices)), sampled_indices)
                splitted_opposite_indices = np.array(indices)[opposite_indices]
                final_opposite_indices.extend(splitted_opposite_indices)

        # Split the data into the training and validation sets
        splitted_features = features[final_indices]
        splitted_class_labels = class_labels[final_indices]
        splitted_domain_labels = domain_labels[final_indices]
        
        if save_opposite:
            splitted_opposite_features = features[final_opposite_indices]
            splitted_opposite_class_labels = class_labels[final_opposite_indices]
            splitted_opposite_domain_labels = domain_labels[final_opposite_indices]
            
            opposite_data = (splitted_opposite_features,
                             splitted_opposite_class_labels,
                             splitted_opposite_domain_labels)
            with open(save_opposite, 'wb') as f:
                pickle.dump(opposite_data, f)

        return (splitted_features, splitted_class_labels, splitted_domain_labels)

    def filter_domain(self, user=None, model=None, device=None, gt=None):
        if os.path.exists(self.fixed_data_path): return
        
        # Create a dictionary to store the indices of each domain
        if self.domain_type == 'user':
            filtered_domain = self.domain_to_number(user)
            print(f'filtered domain: {user}')
        if self.domain_type == 'model':
            filtered_domain = self.domain_to_number(model)
            print(f'filtered domain: {model}')
        if self.domain_type == 'device':
            filtered_domain = self.domain_to_number(device)
            print(f'filtered domain: {device}')
        if self.domain_type == 'gt':
            filtered_domain = self.domain_to_number(gt)
            print(f'filtered domain: {gt}')
        
        indices = []
        for i, domain in enumerate(self.domain_labels):
            if domain == filtered_domain:
                indices.append(i)

        # Split the data into the training and validation sets
        self.features = self.features[indices]
        self.class_labels = self.class_labels[indices]
        self.domain_labels = self.domain_labels[indices]
        
        self.dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(self.features).float(),
                    torch.from_numpy(self.class_labels),
                    torch.from_numpy(self.domain_labels))
    
    def split_kshot_dataset(self, shot_num, test_size, val_size):
        if os.path.exists(self.fixed_data_path):
            return self.load_fixed_dataset_if_exists()
        
        class_indices = defaultdict(list)
        for i, classs_lab in enumerate(self.class_labels):
            class_indices[classs_lab].append(i)
        
        train_indices = []
        test_val_indices = []
        for class_lab, indices in class_indices.items():
            sampled_indices = np.random.choice(len(indices), size=shot_num, replace=False)
            splitted_indices = np.array(indices)[sampled_indices]
            train_indices.extend(splitted_indices)
            
            opposite_indices = np.setdiff1d(np.arange(len(indices)), sampled_indices)
            splitted_opposite_indices = np.array(indices)[opposite_indices]
            test_val_indices.extend(splitted_opposite_indices)

        # Split the data into the training and validation sets
        train_features = self.features[train_indices]
        train_class_labels = self.class_labels[train_indices]
        train_domain_labels = self.domain_labels[train_indices]
        
        sampled_indices = np.random.choice(len(test_val_indices), size=test_size, replace=False)
        test_indices = np.array(test_val_indices)[sampled_indices]
        opposite_indices = np.setdiff1d(np.arange(len(test_val_indices)), sampled_indices)
        sampled_indices = np.random.choice(len(opposite_indices), size=val_size, replace=False)
        val_indices = np.array(opposite_indices)[sampled_indices]
        
        test_features = self.features[test_indices]
        test_class_labels = self.class_labels[test_indices]
        test_domain_labels = self.domain_labels[test_indices]
        
        val_features = self.features[val_indices]
        val_class_labels = self.class_labels[val_indices]
        val_domain_labels = self.domain_labels[val_indices]
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_features).float(),
            torch.from_numpy(train_class_labels),
            torch.from_numpy(train_domain_labels))
        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_features).float(),
            torch.from_numpy(test_class_labels),
            torch.from_numpy(test_domain_labels))
        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(val_features).float(),
            torch.from_numpy(val_class_labels),
            torch.from_numpy(val_domain_labels))
        
        if self.fixed_data_path and not os.path.exists(self.fixed_data_path):
             with open(self.fixed_data_path, 'wb') as f:
                pickle.dump((train_dataset, test_dataset, val_dataset), f)
        return train_dataset, test_dataset, val_dataset

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        domains = self.metadata[f'{self.domain_type}s']
        return len(domains)

    # TODO: TBI
    # def get_datasets_per_domain(self):
    #     return self.kshot_datasets
    
    def domain_to_number(self, label):
        domains = self.metadata[f'{self.domain_type}s']
        dic = {v: i for i, v in enumerate(domains)}
        if label in dic.keys():
            return dic[label]
        else:
            assert 0, f'no such label in class info'
            return -1

    def class_to_number(self, label):
        classes = self.metadata[f'{self.class_type}s']
        dic = {v: i for i, v in enumerate(classes)}
        if label in dic.keys():
            return dic[label]
        else:
            assert 0, f'no such label in class info'
            return -1

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        return self.dataset[idx]
    
    def load_fixed_dataset_if_exists(self):
        if os.path.exists(self.fixed_data_path):
            print("Loading fixed dataset...")
            with open(self.fixed_data_path, 'rb') as f:
                train_dataset, test_dataset, val_dataset = pickle.load(f)
                return train_dataset, test_dataset, val_dataset


# TODO: TBI
# class KSHOTTensorDataset(torch.utils.data.Dataset):
#     def __init__(self, num_classes, features, classes, domains):
#         assert (features.shape[0] == classes.shape[0] == domains.shape[0])

#         self.num_classes = num_classes
#         self.features_per_class = []
#         self.classes_per_class = []
#         self.domains_per_class = []

#         for class_idx in range(self.num_classes):
#             indices = np.where(classes == class_idx)
#             self.features_per_class.append(np.random.permutation(features[indices]))
#             self.classes_per_class.append(np.random.permutation(classes[indices]))
#             self.domains_per_class.append(np.random.permutation(domains[indices]))

#         self.data_num = min(
#             [len(feature_per_class) for feature_per_class in self.features_per_class])  # get min number of classes

#         for i in range(self.num_classes):
#             self.features_per_class[i] = torch.from_numpy(self.features_per_class[i][:self.data_num]).float()
#             self.classes_per_class[i] = torch.from_numpy(self.classes_per_class[i][:self.data_num])
#             self.domains_per_class[i] = torch.from_numpy(self.domains_per_class[i][:self.data_num])

#     def __getitem__(self, index):

#         features = torch.FloatTensor(self.num_classes, *(self.features_per_class[0][0].shape)) # make FloatTensor with shape num_classes x F-dim1 x F-dim2...
#         classes = torch.LongTensor(self.num_classes)
#         domains = torch.LongTensor(self.num_classes)

#         rand_indices = [i for i in range(self.num_classes)]
#         np.random.shuffle(rand_indices)

#         for i in range(self.num_classes):
#             features[i] = self.features_per_class[rand_indices[i]][index]
#             classes[i] = self.classes_per_class[rand_indices[i]][index]
#             domains[i] = self.domains_per_class[rand_indices[i]][index]

#         return (features, classes, domains)

#     def __len__(self):
#         return self.data_num
