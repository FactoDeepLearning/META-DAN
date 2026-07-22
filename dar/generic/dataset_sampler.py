from torch.utils.data import RandomSampler
import torch
from dar.generic.utils import randint
import copy
from random import shuffle

class MultiDatasetSampler(RandomSampler):
    """
    Sampler that handles train dataset made up of several subdatasets.
    A sample from each dataset has the same choice to be picked no matter the unbalance.
    """
    def __init__(self, data_source, generator=None):
        super().__init__(data_source, replacement=False, num_samples=None, generator=generator)
        
        self.datasets_indices = {}
        for i, s in enumerate(data_source.samples):
            dataset_name = s["dataset"]
            if dataset_name not in self.datasets_indices:
                self.datasets_indices[dataset_name] = [i, ]
            else:
                self.datasets_indices[dataset_name].append(i)
        self.num_datasets = len(self.datasets_indices)

        self.current_indices_per_dataset = copy.deepcopy(self.datasets_indices)
        for index_list in self.current_indices_per_dataset.values():
            shuffle(index_list)


    def get_dataset_index(self, dataset_name):
        current_indices = self.current_indices_per_dataset[dataset_name]
        if len(current_indices) == 0:
            self.current_indices_per_dataset[dataset_name] = copy.deepcopy(self.datasets_indices[dataset_name])
            current_indices = self.current_indices_per_dataset[dataset_name]
            shuffle(current_indices)
        return current_indices.pop()
    
    def get_random_index(self):
        dataset_name = list(self.datasets_indices.keys())[randint(0, self.num_datasets)]
        return self.get_dataset_index(dataset_name)
    
    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        num_samples_per_dataset = self.num_samples // self.num_datasets
        indices = []
        for dataset_name in self.datasets_indices.keys():
            dataset_indices = []
            while len(dataset_indices) != num_samples_per_dataset:
                dataset_indices.append(self.get_dataset_index(dataset_name))
            indices.extend(dataset_indices)

        remaining_samples = n - len(indices)
        for _ in range(remaining_samples):
            indices.append(self.get_random_index())

        shuffle(indices)
        yield from indices
