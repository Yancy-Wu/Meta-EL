'''
    Adaptar interface file.
'''

from typing import Tuple, Dict
import torch
from META_EL.datasets import Task

class Adaptar():
    '''
        convert an Task object to support Tensors and query Tensors.
        implementation depend on model and source task content
    '''

    def generate_support_tensors(self, task: Task) -> Dict[str, torch.Tensor]:
        '''
            convert all Ways in an Task object to tensor dict.
            `task`: a meta-learning task define struct.
            `return`: example for bert using EasyAdaptar, it will generate:
             {'input_ids': tensor[ways_num, shots_num, fixed_len],
             'att_mask': tensor[ways_num, shots_num, fixed_len]}
        '''
        raise NotImplementedError

    def generate_query_tensors(self, task: Task) -> Dict[str, torch.Tensor]:
        '''
            convert all test examples in an Task object to Tensors.
            `task`: a meta-learning task define struct.
            `return`: example for bert using EasyAdaptar, it will generate:
             {'input_ids': tensor[query_num, fixed_len],
             'att_mask': tensor[query_num, fixed_len],
             'y': tensor[query_num]}
        '''
        raise NotImplementedError
