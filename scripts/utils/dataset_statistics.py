from glob import glob
from tqdm import tqdm
import pickle

training_data = glob('/mnt/sting/hjyoon/projects/cross/HHAR/augcon/*/pretrain/train.pkl')

total_len = 0
num = 0
for t in tqdm(training_data):
    with open(t, 'rb') as f:
        data = pickle.load(f)
        total_len += len(data)
        num += 1

print(total_len/num)