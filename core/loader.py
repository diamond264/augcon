# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import random
import torchvision.datasets as datasets
import numpy as np
import torch

class AugConTransform:
    def __init__(self, pre_process, post_process, base_transforms):
        self.pre_process = pre_process
        self.post_process = post_process
        self.base_transforms = base_transforms

    def __call__(self, x, params=None):
        q = self.pre_process(x)
        q = self.post_process(q)

        k = self.pre_process(x)
        applied_params = []
        for i, (transform, range_, prob) in enumerate(self.base_transforms):
            if not params is None:
                if params[i][0] == True: apply = 1
                else: apply = 0
                k, new_params, applied = transform(k, range_, apply, params[i][1])
            else:
                k, new_params, applied = transform(k, range_, prob, None)
            applied_params.append((applied, new_params))
        k = self.post_process(k)
        return ([q, k], applied_params)


class AugConDatasetFolder(datasets.vision.VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        extensions = datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        print(classes,class_to_idx)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        samples2 = samples.copy()
        random.shuffle(samples2)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.samples2 = samples2
        self.targets = [s[1] for s in samples]
        self.targets2 = [s[1] for s in samples2]
        self.imgs = self.samples

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        return datasets.folder.make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return datasets.folder.find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        path2, target2 = self.samples2[index]
        sample = self.loader(path)
        sample2 = self.loader(path2)
        if self.transform is not None:
            sample, params = self.transform(sample)
            sample2, params2 = self.transform(sample2, params)
            # sample2, params2 = self.transform(sample2)
        if self.target_transform is not None:
            target = self.target_transform(target)
            target2 = self.target_transform(target2)

        return sample, sample2, target, target2, params, params2

    def __len__(self) -> int:
        return len(self.samples)
class Train_cls_loader(datasets.vision.VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        extensions = datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None
        super().__init__(root, transform=transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = AugConDatasetFolder.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        sample_idx=[[samples[i] for i in range(len(samples)) if samples[i][1]== idx]for idx in range(len(classes))]
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.sample_idx= sample_idx

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return datasets.folder.find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        #print(target)
        #print(self.class_to_idx[target])
        foo = np.random.choice(len(self.sample_idx[target]),1, replace=True)[0]
        foo =0
        #print(self.sample_idx[target][foo])
        positive= self.loader(self.sample_idx[target][foo][0])
 
        negative_list= [i for i in range(len(self.classes)) if i!=target]
        negative_idx=np.random.choice(negative_list,1,replace=True)[0]
        #negative= self.loader(self.sample_idx[negative_idx][np.random.choice(len(self.sample_idx[negative_idx]),1, replace=True)[0]][0])
        negative= self.loader(self.sample_idx[negative_idx][0][0])
        negative_target= self.classes[negative_idx]
        if self.transform is not None:
            sample = self.transform(sample)
            positive = self.transform(positive)
            negative= self.transform(negative)
        pos_lab=1
        neg_lab=0
        #print(sample)
        return sample, positive, pos_lab, negative, neg_lab, target, negative_target

    def __len__(self) -> int:
        return len(self.samples)
    
class Val_cls_loader(datasets.vision.VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        extensions = datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None
        super().__init__(root, transform=transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = AugConDatasetFolder.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        sample_idx=[[samples[i] for i in range(len(samples)) if samples[i][1]== idx]for idx in range(len(classes))]
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.sample_idx= sample_idx

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return datasets.folder.find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        #print(target)
        #print(self.class_to_idx[target])
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

def support_set(       
        root: str,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        ):
        extensions = datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None
        classes, class_to_idx = datasets.folder.find_classes(root)
        samples = AugConDatasetFolder.make_dataset(root, class_to_idx, extensions, is_valid_file)
        print(len(samples))
        sample_idx=np.array([[samples[i] for i in range(len(samples)) if samples[i][1]== idx]for idx in range(len(classes))])
        support_set= []
        print('sample',sample_idx.shape)
        for sample in sample_idx[:,0]:
            print(sample)
            sample = transform(loader(sample[0]))
            support_set.append(sample)
        
        support_set= torch.stack(support_set)
        return support_set
        
