'''
    a kind of implementation of adapter.
'''

from typing import Dict, List, Any
import torch
from base.adapter import Adapter
from base.tensorizer import Tensorizer
from .datasets import TrainExample

class ZelAdapter(Adapter):
    '''
        an easy implementation for zel adapter.
    '''

    def __init__(self, query_tensorizer: Tensorizer, candidate_tensorizer: Tensorizer):
        self.query_tensorizer = query_tensorizer
        self.candidate_tensorizer = candidate_tensorizer

    def generate_candidate_tensors(self, candidates: List[Any]) -> Dict[str, torch.Tensor]:
        '''
            using candidate tensorizer to convert candidates to tensors.
        '''
        return self._raw_list_to_tensors(self.candidate_tensorizer, candidates, 'candidate')

    def generate_query_tensors(self, queries: List[Any]) -> Dict[str, torch.Tensor]:
        '''
            using query tensorizer to convert queries to tensors.
        '''
        return self._raw_list_to_tensors(self.query_tensorizer, queries, 'query')

    # pylint: disable=arguments-differ
    def generate_tensors(self, examples: List[TrainExample]) -> Dict[str, torch.Tensor]:
        '''
            [DESCRIPTION]
              convert all examples to tensors
            [PARAMS]
              `examples`: TrainExample list.
              `return`: a dict of tensors whose shape is [example_num, **]
              ** are tensorizer tensor size.
        '''
        # generate inputs list.
        queries = [example.query for example in examples]
        candidates = [example.candidate for example in examples]
        y = [example.y for example in examples]
        # generate tensors
        candidate_tensor_map = self.generate_candidate_tensors(candidates)
        query_tensor_map = self.generate_query_tensors(queries)
        y_tensor = torch.LongTensor(y)
        # append all tensors to dict
        return {
            'query_tensor_map': query_tensor_map,
            'candidate_tensor_map': candidate_tensor_map,
            'y': y_tensor
        }
