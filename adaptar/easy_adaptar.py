'''
    a kind of implementation of adaptar.
'''

from typing import Dict
import torch
from . import Adaptar
from ..tokenizer import Tokenizer
from ..datasets import Task

class EasyAdaptar(Adaptar):
    '''
        an easy implementation adaptar.
        using no external knowledge, suitable for dataset with only text X and label Y.
    '''

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def generate_support_tensors(self, task: Task) -> Dict[str, torch.Tensor]:
        '''
            `return`: a dict of tensors.
            the name of tensor depend on tokenizer key name.
            the tensor shape are [ways_num, shots_num, **]
            ** are tokenizer tensor size.
        '''
        ways_num = len(task.ways)
        shots_num = len(task.ways[0].shots)
        tensor_map = dict()
        # append all tensors, result list shape: [total_shots_num, ???]
        for way in task.ways:
            for shot in way.shots:
                for name, tensor in self.tokenizer.encode(shot.x):
                    tensor.unsequeeze(0)
                    shots_embs = tensor_map.setdefault(name, [])
                    shots_embs.append(tensor)
        # cat and reshape tensors -> [ways_num, shots_num, ???]
        for tensor in tensor_map.values():
            tensor = torch.cat(tensor)
            tensor = tensor.view([ways_num, shots_num, *tensor.size()[1:]])
        return tensor_map


    def generate_query_tensors(self, task: Task) -> Dict[str, torch.Tensor]:
        '''
            `return`: a dict of tensors, always include an tensor named 'y'(label)
            the name of other tensors depend on tokenizer key names.
            the tensor shape are [query_num, **]
            y tensor shape are [query_num]
        '''
        label_str = [way.y for way in task.ways]
        tensor_map = dict()
        labels = []
        # append all tensors.
        for example in task.test:
            for name, tensor in self.tokenizer.encode(example.shot.x):
                tensor.unsequeeze(0)
                example_embs = tensor_map.setdefault(name, [])
                example_embs.append(tensor)
            labels.append(label_str.index(example.y))
        # cat tensors -> [query_num, ???]
        for tensor in tensor_map.values():
            tensor = torch.cat(tensor)
        # update y tensor -> [query_num]
        tensor_map.update({'y': torch.LongTensor(labels)})
        return tensor_map
