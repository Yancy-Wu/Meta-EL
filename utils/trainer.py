'''
    a pipeline for pluggable zel and mel trainer modules.
'''

import torch
import torch.nn
import torch.nn.functional as F
from tqdm import trange
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

    def _load(self, what: str, batch_size: int) -> DictDataLoader:
        # generating train tensors
        print(f'[TRAINER]: generating {what} loader')
        data = getattr(self.dataset, f'{what}_data')()
        print(f'[TRAINER]: converting {what} data to tensors:')
        tensors_map = self.adapter.generate_tensors(data)
        return DictDataLoader(tensors_map, {
            'batch_size': batch_size,
            'shuffle': True
        })

    def __init__(self, components_and_config: dict):
        super().__init__(components_and_config)
        self.model.to(self.DEVICE)
        # load data
        self.train_loader = self._load('train', self.TRAIN_BATCH_SIZE)
        self.valid_loader = self._load('valid', self.VALID_BATCH_SIZE)

    def _eval(self, loader: DictDataLoader):
        # validation start here.
        overall_num = 0
        overall_correct_num = 0
        # eval model and no grad.
        self.model.eval()
        with torch.no_grad():
            with trange(0, len(loader)) as progress:
                for batch, _ in zip(loader, progress):
                    deep_apply_dict(batch, lambda _, v: v.to(self.DEVICE))
                    y = batch.pop('y')
                    res = self.model.forward(**batch)
                    predict_y = torch.argmax(res, dim=-1)
                    is_right = torch.eq(y, predict_y).view(-1)
                    overall_num += torch.numel(is_right)
                    overall_correct_num += torch.sum(is_right, 0)
        # print finally eval results.
        print('ACC:', overall_correct_num / float(overall_num))

    def train(self):
        '''
            train start here.
        '''
        optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-5)
        for round_num in range(0, self.ROUND):
            # do valid per round
            print(f'**** now round {round_num} valid begin:')
            self._eval(self.valid_loader)
            # do train
            for step, batch in enumerate(self.train_loader):
                self.model.train()
                deep_apply_dict(batch, lambda _, v: v.to(self.DEVICE))
                y = batch.pop('y').view(-1)
                res = self.model.forward(**batch)
                res = res.view(-1, res.size(-1))
                loss = F.cross_entropy(res, y)
                print(f'[round: {round_num}]: {step}/{len(self.train_loader)} end. loss: {loss}')
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def test(self, test_data: list):
        '''
            test start here.
        '''
        print(f'[TRAINER]: converting test data to tensors:')
        tensors_map = self.adapter.generate_tensors(test_data)
        loader = DictDataLoader(tensors_map, {
            'batch_size': self.VALID_BATCH_SIZE,
            'shuffle': True
        })
        print('now we begin to test:')
        self._eval(loader)
