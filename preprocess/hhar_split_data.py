import torch.utils.data
import pandas as pd
import os
import pickle
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
# import itertools

class ProcessHHAR():
    def __init__(self, file, class_type, seq_len=256,
                 split_ratio=0.8, drop_size_threshold=500,
                 model=None, user=None, gt=None,
                 shots=[10, 5, 2, 1],
                 pretrain_dir='', finetune_dir='',
                 finetune_test_size=300, finetune_val_size=100):
        self.metadata = {
            'user': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
            'model': ['nexus4', 's3', 's3mini', 'lgwatch', 'gear'],
            'device': ['lgwatch_1', 'lgwatch_2',
                        'gear_1', 'gear_2'
                        'nexus4_1', 'nexus4_2',
                        's3_1', 's3_2', 's3mini_1', 's3mini_2'], # *** Devices are not considered
            'gt': ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown']
        }
        self.pretrain_dir = pretrain_dir
        self.finetune_dir = finetune_dir
        
        self.finetune_test_size = finetune_test_size
        self.finetune_val_size = finetune_val_size
        
        self.seq_len = seq_len
        self.OVERLAPPING_WIN_LEN = self.seq_len // 2
        self.WIN_LEN = self.seq_len
        
        self.domain_types = ['user', 'model', 'gt']
        self.class_type = class_type
        self.domain_types.remove(self.class_type)
        
        self.user = user
        self.model = model
        self.gt = gt
        
        self.split_ratio = split_ratio
        self.shots = shots
        self.shots.sort(reverse=True)
        
        print(f'Loading data from file...')
        self.df = pd.read_csv(file)
        self.df.columns = self.df.columns.str.lower()
        self.features, self.class_labels, self.domain_labels = self.split_window(self.df)
        self.valid_domains = self.drop_minorities(drop_size_threshold)
        print(f'Valid domains (size {len(self.valid_domains)}):')
        print(self.valid_domains)
        
        domain_dic = {
            'user': self.class_to_number('user', self.user) if self.user else None,
            'model': self.class_to_number('model', self.model) if self.model else None,
            'gt': self.class_to_number('gt', self.gt) if self.gt else None
        }
        self.domain = tuple([domain_dic[dom] for dom in self.domain_types])
        print(f'Target domain:')
        print(self.domain)
        
        if not self.domain in self.valid_domains:
            assert 0, 'not a valid domain'
    
    def drop_minorities(self, drop_size_threshold):
        domains = []
        for dom_1 in self.metadata[self.domain_types[0]]:
            for dom_2 in self.metadata[self.domain_types[1]]:
                domains.append((dom_1, dom_2))
        
        print(f'searching domains with too small data... (<{drop_size_threshold})')
        idx_per_domain = {}
        size_per_domain = {}
        for idx, domain_label in tqdm(enumerate(self.domain_labels)):
            if not domain_label in idx_per_domain.keys():
                idx_per_domain[domain_label] = [idx]
                size_per_domain[domain_label] = 1
            else:
                idx_per_domain[domain_label].append(idx)
                size_per_domain[domain_label] += 1
        
        valid_domains = []
        invalid_domains = []
        valid_idxs = []
        for domain, size in size_per_domain.items():
            if size < drop_size_threshold:
                domain_1, domain_2 = domain
                d1_sum_size = 0
                d2_sum_size = 0
                for d, s in size_per_domain.items():
                    if d[0] == domain_1:
                        d1_sum_size += s
                    if d[1] == domain_2:
                        d2_sum_size += s
                if d1_sum_size > d2_sum_size:
                    invalid_domain = domain_2
                else:
                    invalid_domain = domain_1
                invalid_domains.append(invalid_domain)
        
        for domain, idxs in idx_per_domain.items():
            if domain[0] in invalid_domains or domain[1] in invalid_domains:
                continue
            valid_domains.append(domain)
            valid_idxs.extend(idxs)
        
        print('dropping domains with too small data...')
        valid_features = [self.features[i] for i in valid_idxs]
        valid_class_labels = [self.class_labels[i] for i in valid_idxs]
        valid_domain_labels = [self.domain_labels[i] for i in valid_idxs]
        
        self.features = valid_features
        self.class_labels = valid_class_labels
        self.domain_labels = valid_domain_labels
        
        return valid_domains
    
    def split_source_target(self):
        source_idxs = []
        target_idxs = []
        print('splitting source-domain data and target-domain data')
        for idx, domain in tqdm(enumerate(self.domain_labels)):
            if domain == self.domain:
                target_idxs.append(idx)
            elif domain[0] != self.domain[0] and domain[1] != self.domain[1]:
                source_idxs.append(idx)
        
        source_data = ([self.features[i] for i in source_idxs],
                   [self.class_labels[i] for i in source_idxs],
                   [self.domain_labels[i] for i in source_idxs])
        target_data = ([self.features[i] for i in target_idxs],
                   [self.class_labels[i] for i in target_idxs],
                   [self.domain_labels[i] for i in target_idxs])
        return source_data, target_data
    
    def process(self):
        self.source_data, self.target_data = self.split_source_target()
        print(f'Loaded source domain data({len(self.source_data[0])}) and target domain data({len(self.target_data[0])})')
        pt_data, source_ft_data = self.split_pt_ft(self.source_data)
        print(f'Splitted source domain data into pre-training data({len(pt_data[0])}) and fine-tuning data({len(source_ft_data[0])})')
        
        pt_features, pt_class_labels, pt_domain_labels = pt_data
        pt_idxs = list(range(len(pt_features)))
        random.shuffle(pt_idxs)
        train_idxs = pt_idxs[:int(0.8*len(pt_idxs))]
        test_idxs = pt_idxs[int(0.8*len(pt_idxs)):int(0.9*len(pt_idxs))]
        val_idxs = pt_idxs[int(0.9*len(pt_idxs)):]
        
        features = [pt_features[i] for i in train_idxs]
        class_labels = [pt_class_labels[i] for i in train_idxs]
        domain_labels = [pt_domain_labels[i] for i in train_idxs]
        pretrain_train_path = os.path.join(self.pretrain_dir, 'train.pkl')
        self.save(features, class_labels, domain_labels, pretrain_train_path)
        
        features = [pt_features[i] for i in test_idxs]
        class_labels = [pt_class_labels[i] for i in test_idxs]
        domain_labels = [pt_domain_labels[i] for i in test_idxs]
        pretrain_test_path = os.path.join(self.pretrain_dir, 'test.pkl')
        self.save(features, class_labels, domain_labels, pretrain_test_path)
        
        features = [pt_features[i] for i in val_idxs]
        class_labels = [pt_class_labels[i] for i in val_idxs]
        domain_labels = [pt_domain_labels[i] for i in val_idxs]
        pretrain_val_path = os.path.join(self.pretrain_dir, 'val.pkl')
        self.save(features, class_labels, domain_labels, pretrain_val_path)
        print(f'Saved pre-training data of source domain')
        
        source_features, source_class_labels, source_domain_labels = source_ft_data
        idx_per_source_classes = {}
        for idx, class_label in enumerate(source_class_labels):
            if class_label in idx_per_source_classes.keys():
                idx_per_source_classes[class_label].append(idx)
            else: idx_per_source_classes[class_label] = [idx]
        
        for shot in self.shots:
            train_idxs = []
            remaining_idxs = []
            for class_label, idxs in idx_per_source_classes.items():
                random.shuffle(idxs)
                train_idxs.extend(idxs[:shot])
                remaining_idxs.extend(idxs[shot:])
            random.shuffle(remaining_idxs)
            test_idxs = remaining_idxs[:self.finetune_test_size]
            val_idxs = remaining_idxs[self.finetune_test_size:self.finetune_test_size+self.finetune_val_size]
                
            features = [source_features[i] for i in train_idxs]
            class_labels = [source_class_labels[i] for i in train_idxs]
            domain_labels = [source_domain_labels[i] for i in train_idxs]
            finetune_source_train_path = os.path.join(self.finetune_dir, f'{shot}shot', 'source', 'train.pkl')
            self.save(features, class_labels, domain_labels, finetune_source_train_path)
            
            features = [source_features[i] for i in test_idxs]
            class_labels = [source_class_labels[i] for i in test_idxs]
            domain_labels = [source_domain_labels[i] for i in test_idxs]
            finetune_source_test_path = os.path.join(self.finetune_dir, f'{shot}shot', 'source', 'test.pkl')
            self.save(features, class_labels, domain_labels, finetune_source_test_path)
            
            features = [source_features[i] for i in val_idxs]
            class_labels = [source_class_labels[i] for i in val_idxs]
            domain_labels = [source_domain_labels[i] for i in val_idxs]
            finetune_source_val_path = os.path.join(self.finetune_dir, f'{shot}shot', 'source', 'val.pkl')
            self.save(features, class_labels, domain_labels, finetune_source_val_path)
        print(f'Saved fine-tuning data of target domain')
        
        target_features, target_class_labels, target_domain_labels = self.target_data
        idx_per_target_classes = {}
        for idx, class_label in enumerate(target_class_labels):
            if class_label in idx_per_target_classes.keys():
                idx_per_target_classes[class_label].append(idx)
            else: idx_per_target_classes[class_label] = [idx]
        
        for shot in self.shots:
            train_idxs = []
            remaining_idxs = []
            for class_label, idxs in idx_per_target_classes.items():
                random.shuffle(idxs)
                train_idxs.extend(idxs[:shot])
                remaining_idxs.extend(idxs[shot:])
            random.shuffle(remaining_idxs)
            test_idxs = remaining_idxs[:self.finetune_test_size]
            val_idxs = remaining_idxs[self.finetune_test_size:self.finetune_test_size+self.finetune_val_size]
            
            features = [target_features[i] for i in train_idxs]
            class_labels = [target_class_labels[i] for i in train_idxs]
            domain_labels = [target_domain_labels[i] for i in train_idxs]
            finetune_target_train_path = os.path.join(self.finetune_dir, f'{shot}shot', 'target', 'train.pkl')
            self.save(features, class_labels, domain_labels, finetune_target_train_path)
            
            features = [target_features[i] for i in test_idxs]
            class_labels = [target_class_labels[i] for i in test_idxs]
            domain_labels = [target_domain_labels[i] for i in test_idxs]
            finetune_target_test_path = os.path.join(self.finetune_dir, f'{shot}shot', 'target', 'test.pkl')
            self.save(features, class_labels, domain_labels, finetune_target_test_path)
            
            features = [target_features[i] for i in val_idxs]
            class_labels = [target_class_labels[i] for i in val_idxs]
            domain_labels = [target_domain_labels[i] for i in val_idxs]
            finetune_target_val_path = os.path.join(self.finetune_dir, f'{shot}shot', 'target', 'val.pkl')
            self.save(features, class_labels, domain_labels, finetune_target_val_path)
        print(f'Saved fine-tuning data of target domain')
    
    def save(self, features, class_labels, domain_labels, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        np_features = np.array(features)
        np_class_labels = np.array(class_labels)
        np_domain_labels = np.array([self.domain_set_to_number(self.domain_types, d) for d in domain_labels])
        
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(np_features).float(),
            torch.from_numpy(np_class_labels),
            torch.from_numpy(np_domain_labels))
        
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f'Saved {len(np_features)} instances in {path}')
        
    def split_pt_ft(self, data):
        print('splitting pre-training data and fine-tuning data')
        features, class_labels, domain_labels = data
        idxs = list(range(len(features)))
        random.shuffle(idxs)
        idxs_1 = idxs[:int(len(idxs)*self.split_ratio)]
        idxs_2 = idxs[int(len(idxs)*self.split_ratio):]
        
        data_1 = (
            [features[i] for i in idxs_1],
            [class_labels[i] for i in idxs_1],
            [domain_labels[i] for i in idxs_1]
        )
        data_2 = (
            [features[i] for i in idxs_2],
            [class_labels[i] for i in idxs_2],
            [domain_labels[i] for i in idxs_2]
        )
        return data_1, data_2

    def split_window(self, df):
        features = []
        class_labels = []
        domain_labels = []

        print('splitting windows...')
        for idx in tqdm(range(max(len(df) // self.OVERLAPPING_WIN_LEN - 1, 0))):
            user = df.iloc[idx * self.OVERLAPPING_WIN_LEN, 9]
            model = df.iloc[idx * self.OVERLAPPING_WIN_LEN, 10]
            gt = df.iloc[idx * self.OVERLAPPING_WIN_LEN, 12]
            user = self.class_to_number('user', user)
            model = self.class_to_number('model', model)
            gt = self.class_to_number('gt', gt)
            
            if self.class_type == 'user':
                class_label = user
                domain = (model, gt)
            elif self.class_type == 'model':
                class_label = model
                domain = (user, gt)
            elif self.class_type == 'gt':
                class_label = gt
                domain = (user, model)

            feature = df.iloc[idx * self.OVERLAPPING_WIN_LEN:(idx + 2) * self.OVERLAPPING_WIN_LEN, 3:6].values
            feature = feature.T

            features.append(feature)
            class_labels.append(class_label)
            domain_labels.append(domain)
        
        return (features, class_labels, domain_labels)

    def class_to_number(self, class_type, label):
        classes = self.metadata[class_type]
        dic = {v: i for i, v in enumerate(classes)}
        if label in dic.keys():
            return dic[label]
        else:
            print(label)
            assert 0, f'no such label in class info'
            return -1
    
    def domain_set_to_number(self, domain_types, domain):
        domain_1 = self.metadata[domain_types[0]]
        domain_2 = self.metadata[domain_types[1]]
        domains = []
        for d1 in domain_1:
            d1_num = self.class_to_number(domain_types[0], d1)
            for d2 in domain_2:
                d2_num = self.class_to_number(domain_types[1], d2)
                domains.append((d1_num, d2_num))
        dic = {v: i for i, v in enumerate(domains)}
        if domain in dic.keys():
            return dic[domain]
        else:
            print(domain)
            assert 0, f'no such domain in domain info'
            return -1


if __name__ == '__main__':
    file_path = '/mnt/sting/hjyoon/projects/cross/HHAR/ActivityRecognitionExp/hhar_minmax_scaling_all.csv'
    base_out_dir = '/mnt/sting/hjyoon/projects/cross/HHAR/augcon'
    
    class_type = 'gt'
    users = ['a', 'b', 'c', 'd', 'f']
    models = ['nexus4', 's3', 's3mini', 'lgwatch']
    for user in users:
        for model in models:
            pretrain_dir = os.path.join(base_out_dir, f'target_user_{user}_model_{model}', 'pretrain')
            finetune_dir = os.path.join(base_out_dir, f'target_user_{user}_model_{model}', 'finetune')
            dataset = ProcessHHAR(file_path, 'gt', user=user, model=model, pretrain_dir=pretrain_dir, finetune_dir=finetune_dir)
            dataset.process()