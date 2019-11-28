'''
    a pipeline for pluggable meta learning modules.
'''

from typing import List, Dict
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as functional
from pytorch_transformers import AdamW
from base.config import Config
from base.dataloader import DictDataLoader
from tensorizer import Tensorizer
from .datasets import Datasets
from .adapter import Adapter
from .datasets import Task

class Mel(Config):
    '''
        [DESCRIPTION]
          free-combination of model, datasets, tensorizer and other component.
          of course maybe some component only support specfic other component.
          see module for detail.
        [PARAMS]
          set all class members.
    '''

    # all component here. please set me.
    dataset: Datasets = None
    tensorizer: Tensorizer = None
    model: nn.Module = None

    # useful member.
    adapter: Adapter = None
    train_loader: DictDataLoader = None
    valid_loader: DictDataLoader = None
    device = torch.device('cpu')

    def __init__(self, components: dict, device=torch.device('cuda:1'), batch_size=1):
        super().__init__(components)
        self.device = device
        self.adapter = Adapter(self.tensorizer)
        self.model.to(self.device)

        # generating train tensors
        print('generating train tasks:')
        train_task_num = self.dataset.TRAIN_TASKS_NUM
        train_tasks = [x for x, _ in zip(self.dataset.train_tasks(), trange(0, train_task_num))]
        self.train_loader = self._tasklist_to_loader(train_tasks, {'batch_size':batch_size})

        # generating validation tensors
        print('generating val tasks:')
        valid_task_num = self.dataset.VALID_TASKS_NUM
        valid_tasks = [x for x, _ in zip(self.dataset.valid_tasks(), trange(0, valid_task_num))]
        self.valid_loader = self._tasklist_to_loader(valid_tasks, {'batch_size':batch_size})

    def _tasklist_to_loader(self, tasks: List[Task], loader_kargs: Dict) -> DictDataLoader:
        '''
            convert tasks to a DictDataLoader for inputing to model.
        '''
        # create new task loader
        support_tensors_dict = self.adapter.generate_support_tensors(tasks)
        query_tensors_dict = self.adapter.generate_query_tensors(tasks)
        tensors_dict = dict(support_tensors_dict, **query_tensors_dict)
        return DictDataLoader(tensors_dict, **loader_kargs)

    def _dict_to_device(self, d: Dict):
        '''
            send a group of tensors to device.
        '''
        for key, tensor in d.items():
            d[key] = tensor.to(self.device)

    def train(self):
        '''
            train start here.
        '''
        optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-5)
        for step, batch in enumerate(self.train_loader):
            self._dict_to_device(batch)
            # input size: [task_num(sampled), ways_num, shots_num, seq_length]
            # label size: [tasks_num(sampled), query_num]
            print(f'now {step} begin:')
            y = batch.pop('y').view(-1)
            res = self.model.forward(**batch)
            # res size: [task_num(sampled), query_num, ways_num], flatten it.
            res = res.view(-1, res.size(-1))
            loss = functional.cross_entropy(res, y)
            print(loss)
            optimizer.zero_grad()
            loss.backward()  # grad += sth
            optimizer.step()  # data += grad
            self.valid()

    def valid(self):
        '''
            validation start here.
        '''
        overall_num = 0
        overall_correct_num = 0
        for batch in self.valid_loader:
            self._dict_to_device(batch)
            y = batch.pop('y')
            res = self.model.forward(**batch)
            predict_y = torch.argmax(res, dim=-1)
            is_right = torch.eq(y, predict_y).view(-1)
            overall_num += torch.numel(is_right)
            overall_correct_num += torch.sum(is_right, 0)
        print('top1 ACC:', overall_correct_num / float(overall_num))
