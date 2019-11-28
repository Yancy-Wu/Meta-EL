'''
    a kind of implementation of adapter.
'''

from typing import Dict, List, Any
import torch
from tqdm import tqdm
from tensorizer import Tensorizer
from .datasets import Example

class Adapter():
    '''
        an easy implementation for zel adapter.
    '''

    def __init__(self, query_tensorizer: Tensorizer, candidate_tensorizer: Tensorizer):
        self.query_tensorizer = query_tensorizer
        self.candidate_tensorizer = candidate_tensorizer

    @staticmethod
    def _generate_tensors(tensorizer, records: List[Any])-> Dict[str, torch.Tensor]:
        '''
            convert origin data list to tensor dict.
            prefix will add to dict key names.
        '''
        tensor_map = dict()
        records = tqdm(records)
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

    def generate_candidate_tensors(self, candidates: List[Any]) -> Dict[str, torch.Tensor]:
        '''
            [DESCRIPTION]
              only convert candidates to tensors, will not modify dict name
            [PARAMS]
              `candidates`: type should be same as `Example.candidate`
        '''
        return self._generate_tensors(self.candidate_tensorizer, candidates)

    def generate_query_tensors(self, queries: List[Any]) -> Dict[str, torch.Tensor]:
        '''
            [DESCRIPTION]
              only convert queries to tensors, will not modify dict name
            [PARAMS]
              `queries`: type should be same as `Example.query`
        '''
        return self._generate_tensors(self.query_tensorizer, queries)

    def generate_example_tensors(self, examples: List[Example]) -> Dict[str, torch.Tensor]:
        '''
            [DESCRIPTION]
              convert all examples to tensors
            [PARAMS]
              `examples`: example list.
              `return`: a dict of tensors whose shape is [example_num, **]
              ** are tensorizer tensor size.
            [EXAMPLES]
              if using BERT tensorizer, it will generate:
              {'query': {'input_ids': tensor[example_num, sequence_length], 'att_mask': ...},
               'candidate': {'input_ids': tensor[example_num, sequence_length], 'att_mask': ...},
               'y': tensor[example_num]}
        '''
        # generate inputs list.
        queries = [example.query for example in examples]
        candidates = [example.candidate for example in examples]
        y = [example.y for example in examples]
        # generate tensors
        query_tensor_map = self._generate_tensors(self.query_tensorizer, queries)
        candidate_tensor_map = self._generate_tensors(self.candidate_tensorizer, candidates)
        y_tensors = torch.LongTensor(y)
        # append all tensors to dict
        return {
            'query_tensor_map': query_tensor_map,
            'candidate_tensor_map': candidate_tensor_map,
            'y': y_tensors
        }
