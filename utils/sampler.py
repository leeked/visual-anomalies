import torch
from torch.utils.data.sampler import Sampler
import numpy as np

class BalancedSampler(Sampler):
    def __init__(self, dataset, num_samples=None, replacement=True):
        self.indices = list(range(len(dataset)))
        self.num_samples = num_samples if num_samples is not None else len(self.indices)
        self.replacement = replacement

        # Get labels for all samples
        label_list = []
        for idx in self.indices:
            _, target = dataset[idx]
            labels = target['labels'].numpy()
            label_list.extend(labels)

        label_counts = np.bincount(label_list)
        weights = 1. / (label_counts + 1e-6)  # Avoid division by zero
        min_sample_weight = weights.min() if len(weights) > 0 else 1.0
        self.weights = np.zeros(len(self.indices))
        for idx in self.indices:
            _, target = dataset[idx]
            labels = target['labels'].numpy()
            if len(labels) > 0:
                sample_weight = np.mean([weights[label] for label in labels])
            else:
                sample_weight = min_sample_weight  # Assign minimum weight to samples with no labels
            self.weights[idx] = sample_weight

        # Normalize weights
        total_weight = self.weights.sum()
        if total_weight > 0:
            self.weights = torch.DoubleTensor(self.weights / total_weight)
        else:
            self.weights = torch.ones(len(self.indices)) / len(self.indices)

    def __iter__(self):
        sample_indices = torch.multinomial(self.weights, self.num_samples, self.replacement)
        return iter(sample_indices.tolist())

    def __len__(self):
        return self.num_samples
