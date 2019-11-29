'''
    a pipeline for pluggable zel trainer modules.
'''

import logging
import torch
import torch.nn.functional as F
from pytorch_transformers import AdamW
from base.config import Config
from base.dataloader import DictDataLoader
from base.utils import dict_to_device
from .datasets import Datasets
from .adapter import Adapter
from .similar_net import SimilarNet

LOGGER = logging.getLogger(__name__)

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
    dataset: Datasets = None
    adapter: Adapter = None
    model: SimilarNet = None

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
        train_examples = self.dataset.train_examples()
        print('[TRAINER]: converting train examples to tensors:')
        train_tensors_map = self.adapter.generate_example_tensors(train_examples)
        self.train_loader = DictDataLoader(train_tensors_map, {
            'batch_size': self.TRAIN_BATCH_SIZE,
            'shuffle': True
        })

        # generating validation tensors
        print('\n[TRAINER]: generating val loader:')
        valid_examples = self.dataset.valid_examples()
        print('[TRAINER]: converting val examples to tensors:')
        valid_tensors_map = self.adapter.generate_example_tensors(valid_examples)
        self.valid_loader = DictDataLoader(valid_tensors_map, {
            'batch_size': self.VALID_BATCH_SIZE,
            'shuffle': True
        })

    def _valid(self):
        # validation start here.
        overall_num = 0
        overall_correct_num = 0
        for batch in self.valid_loader:
            dict_to_device(batch, self.DEVICE)
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
                dict_to_device(batch, self.DEVICE)
                # input size: [batch_size, seq_length], label size: [batch_size]
                y = batch.pop('y')
                res = self.model.forward(**batch)
                # res size: [batch_size, 2]
                loss = F.cross_entropy(res, y)
                print(f'[round: {round_num}]: {step}/{len(self.train_loader)} end. loss: {loss}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # do valid per round
            print(f'**** now round {round_num} valid begin:')
            self._valid()
