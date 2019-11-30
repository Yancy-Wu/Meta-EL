'''
    a pipeline for pluggable zel and mel trainer modules.
'''

import torch
import torch.nn
import torch.nn.functional as F
from pytorch_transformers import AdamW
from base.config import Config
from base.dataset import Dataset
from base.adapter import Adapter
from utils.dataloader import DictDataLoader
from utils import deep_apply_dict

class Trainer(Config):
    '''
        [DESCRIPTION]
          free-combination of model, datasets, adapter and other component.
          of course maybe some component only support specfic other component.
          see module for detail.
        [PARAMS]
          set all class members.
    '''

    # all component here. please set me.
    dataset: Dataset = None
    adapter: Adapter = None
    model: torch.nn.Module = None

    # useful member.
    train_loader: DictDataLoader = None
    valid_loader: DictDataLoader = None

    # super parameters. tensor device
    DEVICE = torch.device('cpu')

    # how many examples per batch.
    TRAIN_BATCH_SIZE = 100

    # how many examples per valid.
    VALID_BATCH_SIZE = 100

    # how many round we train all examples.
    ROUND = 5

    def __init__(self, components_and_config: dict):
        super().__init__(components_and_config)
        self.model.to(self.DEVICE)

        # generating train tensors
        print('[TRAINER]: generating train loader')
        train_data = self.dataset.train_data()
        print('[TRAINER]: converting train data to tensors:')
        train_tensors_map = self.adapter.generate_tensors(train_data)
        self.train_loader = DictDataLoader(train_tensors_map, {
            'batch_size': self.TRAIN_BATCH_SIZE,
            'shuffle': True
        })

        # generating validation tensors
        print('\n[TRAINER]: generating val loader:')
        valid_data = self.dataset.train_data()
        print('[TRAINER]: converting val data to tensors:')
        valid_tensors_map = self.adapter.generate_tensors(valid_data)
        self.valid_loader = DictDataLoader(valid_tensors_map, {
            'batch_size': self.VALID_BATCH_SIZE,
            'shuffle': True
        })

    def _valid(self):
        # validation start here.
        overall_num = 0
        overall_correct_num = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                deep_apply_dict(batch, lambda _, v: v.to(self.DEVICE))
                y = batch.pop('y')
                res = self.model.forward(**batch)
                predict_y = torch.argmax(res, dim=-1)
                is_right = torch.eq(y, predict_y).view(-1)
                overall_num += torch.numel(is_right)
                overall_correct_num += torch.sum(is_right, 0)
        print('ACC:', overall_correct_num / float(overall_num))

    def train(self):
        '''
            train start here.
            when train ended, please create Predication object to predict.
        '''
        optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-5)
        for round_num in range(0, self.ROUND):
            for step, batch in enumerate(self.train_loader):
                deep_apply_dict(batch, lambda _, v: v.to(self.DEVICE))
                y = batch.pop('y').view(-1)
                res = self.model.forward(**batch)
                res = res.view(-1, res.size(-1))
                loss = F.cross_entropy(res, y)
                print(f'[round: {round_num}]: {step}/{len(self.train_loader)} end. loss: {loss}')
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # do valid per round
            print(f'**** now round {round_num} valid begin:')
            self._valid()
