from collections import Counter

import numpy as np
import torch

from tqdm import tqdm

from typing import List

from diora.data.treeencoding import CharBatchMap, BatchMap


def choose_negative_samples(negative_sampler, k_neg, device):
    neg_samples = negative_sampler.sample(k_neg, device)
    #neg_samples = torch.from_numpy(neg_samples)
    return neg_samples


def calculate_freq_dist(data, vocab_size):
    
    # TODO: This becomes really slow on large datasets.
    counter = Counter()
    for i in range(vocab_size):
        counter[i] = 0
    for x in tqdm(data, desc='freq_dist'):
        if isinstance(x, torch.Tensor):
            counter.update(x.tolist())
        else:
            counter.update(x)
    freq_dist = [v for k, v in sorted(counter.items(), key=lambda x: x[0])]
    freq_dist = np.asarray(freq_dist, dtype=np.float32)
    
    return freq_dist


class NegativeSampler:
    def __init__(self, freq_dist, dist_power, word2idx, char2idx, epsilon=10**-2):
        self.dist = freq_dist ** dist_power + epsilon * (1/len(freq_dist))
        self.dist = self.dist / sum(self.dist)      # Final distribution should be normalized
        self.rng = np.random.RandomState()
        self.i2word = {i : w for w, i in word2idx.items()}
        self.char2idx = char2idx

        self.len = len(word2idx)

    def set_seed(self, seed):
        self.rng.seed(seed)

    def sample(self, num_samples, device):
        ids : np.ndarray = self.rng.choice(len(self.dist), num_samples, p=self.dist, replace=False)
        ids = torch.from_numpy(ids).to(device)

        #ids = torch.arange(self.len)
        
        if self.char2idx is not None:
            chars : List[List[int]] = self.char2idx.chars[[self.i2word[i] for i in ids]]


            return CharBatchMap(input_ids = ids.view(1, -1),
                                char_ids = [[torch.tensor(word, dtype = torch.int).to(device) for word in chars]],
                                batch_size = 1,
                                length = len(ids))

        else:
            return BatchMap(input_ids = ids.view(1, -1),
                            batch_size = 1,
                            length = len(ids))
