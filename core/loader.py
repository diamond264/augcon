# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import random
import torchvision.datasets as datasets


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
        for i, (transform, range_) in enumerate(self.base_transforms):
            if not params is None:
                k, p = transform(k, params[i], range_)
            else:
                k, p = transform(k, None, range_)
            applied_params.append(p)
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
        if self.target_transform is not None:
            target = self.target_transform(target)
            target2 = self.target_transform(target2)

        return sample, sample2, target, target2, params, params2

    def __len__(self) -> int:
        return len(self.samples)