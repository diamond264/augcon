from glob import glob
import numpy as np
import pickle
import torch
import torch.utils

paths = glob("/mnt/sting/hjyoon/projects/cross/WESAD/augcon/*/finetune/*/target/*.pkl")
paths += glob(
    "/mnt/sting/hjyoon/projects/cross/NinaproDB5/augcon/*/finetune/*/target/*.pkl"
)

for path in paths:
    with open(path, "rb") as f:
        data = pickle.load(f)
        features = []
        labels = []
        domains = []
        for d in data:
            features.append(d[0].numpy())
            lab = d[1] - 1
            lab = lab.numpy()
            labels.append(lab)
            domains.append(d[2].numpy())
        features = np.array(features)
        labels = np.array(labels)
        domains = np.array(domains)

        new_dataset = torch.utils.data.TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(domains, dtype=torch.long),
        )

        with open(path, "wb") as f:
            pickle.dump(new_dataset, f)
