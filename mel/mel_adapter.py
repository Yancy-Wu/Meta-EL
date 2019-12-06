'''
    a kind of implementation of adapter.
'''

from typing import Dict, List
import torch
from base.tensorizer import Tensorizer
from base.adapter import Adapter
from utils import deep_apply_dict
from .datasets import Task

class MelAdapter(Adapter):
    '''
        an easy implementation adapter.
        convert a list of task struct to tensors.
    '''

    def __init__(self, support_tensorizer: Tensorizer, query_tensorizer):
        self.support_tensorizer = support_tensorizer
        self.query_tensorizer = query_tensorizer

    # pylint: disable=arguments-differ
    def generate_tensors(self, tasks: List[Task]) -> Dict[str, torch.Tensor]:
        '''
            [DESCRIPTION]
              convert all Ways in an Task object to tensor dict.
            [PARAMS]
              `task`: a meta-learning task define struct.
              `return`: a dict of tensors whose shape is [task_num, ways_num, shots_num, **]
              (** are tensorizer tensor size) and whose name are query_ prefix.
        '''
        # task info, and tensor shape
        task_num = len(tasks)
        way_num = len(tasks[0].support)
        shot_num = len(tasks[0].support[0].shots)

        # flatten task, prepare data.
        supports = []
        queries = []
        y = []

        # data done.
        for task in tasks:
            label_str = [way.y for way in task.support]
            supports += sum([way.shots for way in task.support], [])
            queries.append(task.query.x)
            y.append(label_str.index(task.query.y))

        # tensor generate.
        support_tensor_map = self._raw_list_to_tensors(self.support_tensorizer, supports, 'support')
        query_tensor_map = self._raw_list_to_tensors(self.query_tensorizer, queries, 'query')
        y_tensor = torch.LongTensor(y)

        # reshape tensors.
        deep_apply_dict(support_tensor_map, lambda _, v: v.view(task_num, way_num, shot_num, -1))
        deep_apply_dict(query_tensor_map, lambda _, v: v.view(task_num, -1))

        return {
            'support_tensor_map': support_tensor_map,
            'query_tensor_map': query_tensor_map,
            'y': y_tensor
        }
