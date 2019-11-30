'''
    base adapter class, for zel and mel.
'''

from typing import Dict, List, Any
from tqdm import tqdm
import torch
from .tensorizer import Tensorizer

class Adapter():
    '''
        an adapter converting a group of specific input, such as example for zel
        and task for mel, to large tensors by using tensorizer.
    '''

    @staticmethod
    def _raw_list_to_tensors(tensorizer: Tensorizer, sources: List[Any], desc: str):
        '''
            convert origin raw data list to tensor dict.
            desc is tqdm description.
        '''
        tensor_map = dict()
        records = tqdm(sources, desc)
        # append all tensors
        for record in records:
            for name, tensor in tensorizer.encode(record).items():
                tensor = torch.unsqueeze(tensor, 0)
                tensor_list = tensor_map.setdefault(name, [])
                tensor_list.append(tensor)
        # close tqdm to avoid print empty line.
        records.close()
        # cat tensors
        for name, tensor_list in tensor_map.items():
            tensor = torch.cat(tensor_list)
            tensor_map[name] = tensor
        return tensor_map

    def generate_tensors(self, objs: List[object]) -> Dict[str, torch.Tensor]:
        '''
            [DESCRIPTION]
              convert an obj list to large tensors.
            [PARAMS]
              `objs`: a type of data struct.
        '''
        raise NotImplementedError
