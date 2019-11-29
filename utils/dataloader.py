'''
    task loader.
    encapsulation of DataLoader and TensorDataset by adding keys.
'''

from typing import Dict, List
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import utils

class DictDataLoader():
    '''
        [DESCRIPTION]
          because TensorDataset lacks key index function.
          we create this class.
        [PARAMS]
          `tensors_dict`: a group of tensors with key. first dim of tensor must same(for sampling).
          `dl_config`: dataloader config, see Dataloader for detail.
    '''

    # tensors key list
    key_list: List = None

    # dataloader
    dataloader = None

    def __init__(self, tensors_dict: Dict[str, torch.Tensor], dl_config: dict()):
        # generate kv list.
        self.key_list, tensors_list = utils.dict_to_kvlist(tensors_dict)
        # load tensors.
        datasets = TensorDataset(*tensors_list)
        self.dataloader = DataLoader(datasets, **dl_config)

    def __iter__(self):
        return DictDataLoaderIter(self)

    def __len__(self):
        return len(self.dataloader)

class DictDataLoaderIter():
    '''
        [DESCRIPTION]
          iterator for task loader.
        [PARAM]
          `loader`: task loader, for fetching saved data.
    '''

    # parent task loader.
    loader: DictDataLoader = None

    def __init__(self, loader: DictDataLoader):
        self.loader = loader
        self.iter = iter(loader.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        '''
            sample data from self.dataloader, then combine all sub tensors
            into a dict. stop when dataloder raise exception
        '''
        batch: List[torch.Tensor] = next(self.iter)
        return utils.kvlist_to_dict(self.loader.key_list, batch)
