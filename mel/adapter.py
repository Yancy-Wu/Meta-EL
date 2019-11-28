'''
    a kind of implementation of adapter.
'''

from typing import Dict, List
import torch
from tensorizer import Tensorizer
from .datasets import Task

class Adapter():
    '''
        an easy implementation adapter.
        convert a list of task struct to tensors.
    '''

    def __init__(self, tensorizer: Tensorizer):
        self.tensorizer = tensorizer

    # aux function for add prefix per key in an dict, except y
    @staticmethod
    def _add_prefix(d: Dict, prefix: str):
        for key in [*d.keys()]:
            if key != 'y':
                d.update({prefix + key: d.pop(key)})
        return d

    def generate_support_tensors(self, tasks: List[Task]) -> Dict[str, torch.Tensor]:
        '''
            [DESCRIPTION]
              convert all Ways in an Task object to tensor dict.
            [PARAMS]
              `task`: a meta-learning task define struct.
              `return`: a dict of tensors whose shape is [task_num, ways_num, shots_num, **]
              (** are tensorizer tensor size) and whose name are query_ prefix.
            [EXAMPLES]
              if using BERT tensorizer, it will generate:
              {'support_input_ids': tensor[tasks_num, ways_num, shots_num, sequence_length],
              'support_att_mask': tensor[tasks_num, ways_num, shots_num, sequence_length]}
        '''
        tasks_num = len(tasks)
        ways_num = len(tasks[0].support)
        shots_num = len(tasks[0].support[0].shots)
        tensor_map = dict()
        # append all tensors, result list shape: [total_shots_num, ???]
        for task in tasks:
            for way in task.support:
                for shot in way.shots:
                    for name, tensor in self.tensorizer.encode(shot.x).items():
                        tensor = torch.unsqueeze(tensor, 0)
                        shots_embs = tensor_map.setdefault(name, [])
                        shots_embs.append(tensor)
        # cat and reshape tensors -> [task_num, ways_num, shots_num, ???]
        for name, tensor_list in tensor_map.items():
            tensor = torch.cat(tensor_list)
            tensor = tensor.view([tasks_num, ways_num, shots_num, *tensor.size()[1:]])
            tensor_map[name] = tensor
        return self._add_prefix(tensor_map, 'support_')

    def generate_query_tensors(self, tasks: List[Task]) -> Dict[str, torch.Tensor]:
        '''
            [DESCRIPTION]
              convert all test examples in an Task object to Tensors.
            [PARAMS]
              `return`: a dict of tensors, always include an tensor named 'y'(label)
              the name of other tensors depend on tensorizer key names.
              the tensor shape are [tasks_num, query_num, **]
              y tensor shape are [tasks_num, query_num]
            [EXAMPLES]
              if using BERT tensorizer, it will generate:
              {'query_input_ids': tensor[tasks_num, query_num, sequence_length],
              'query_att_mask': tensor[tasks_num, query_num, sequence_length],
              'y': tensor[tasks_num, query_num]}
        '''
        tasks_num = len(tasks)
        query_num = len(tasks[0].query)
        tensor_map = dict()
        labels = []
        # append all tensors.
        for task in tasks:
            label_str = [way.y for way in task.support]
            for example in task.query:
                for name, tensor in self.tensorizer.encode(example.shot.x).items():
                    tensor = torch.unsqueeze(tensor, 0)
                    example_embs = tensor_map.setdefault(name, [])
                    example_embs.append(tensor)
                labels.append(label_str.index(example.y))
        # cat tensors -> [task_num, query_num, ???]
        for name, tensor_list in tensor_map.items():
            tensor = torch.cat(tensor_list)
            tensor = tensor.view([tasks_num, query_num, *tensor.size()[1:]])
            tensor_map[name] = tensor
        # update y tensor -> [query_num]
        tensor_map.update({'y': torch.LongTensor(labels).view([tasks_num, query_num])})
        return self._add_prefix(tensor_map, 'query_')
