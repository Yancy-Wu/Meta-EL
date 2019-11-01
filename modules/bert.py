'''
    easy bert modules.
'''

import torch
from torch import nn
from pytorch_transformers import BertModel
from META_EL.base.config import Config

class Bert(nn.Module, Config):
    '''
        support different pooling method
    '''

    # pooling method, avail value: cls, avg
    POOLING_METHOD = 'avg'

    def __init__(self, config=None):
        nn.Module.__init__(self)
        Config.__init__(self, config)

    @classmethod
    def from_pretrained(cls, model_dir, config=None):
        '''
            load bert pre-train model from disk
            `model_dir`: model location
        '''
        if not config:
            config = dict()
        config.update({'bert': BertModel.from_pretrained(model_dir)})
        return cls(config)

    # pylint: disable=arguments-differ
    def forward(self, input_ids: torch.LongTensor, att_mask: torch.LongTensor):
        '''
            ** stand for any dim.
            `inputs_ids shape`: (**, sequence_length)
            `att_mask shape`: (**, sequence_length)
            `return shape`: (**, hidden_size)
        '''
        saved_size = input_ids.size()[:-1]
        # re-size to 2-dim: (count, sequence_length)
        seq_length = input_ids.size()[-1]
        input_ids = input_ids.view(-1, seq_length)
        att_mask = att_mask.view(-1, seq_length)
        # retrieve bert embeddings: (count, sequence_length)
        embs = self.bert(input_ids, att_mask)[0]
        # use python reflection to call pooling
        pooling_func = getattr(self, f'_{self.POOLING_METHOD}_pooling')
        # re-size to origin
        return pooling_func(embs, att_mask).view(*saved_size, -1)

    @staticmethod
    def _avg_pooling(embs: torch.FloatTensor, att_mask: torch.LongTensor) -> torch.FloatTensor:
        '''
            `embeddings shape`: (**, sequence_length, hidden_size)
            `att_mask shape`: (**, sequence_length)
            `return shape`: (**, hidden_size)
            return average token embeddings except [PAD]
        '''
        # avail_mask_count: (**, 1)
        avail_mask_count = att_mask.sum(-1, keepdim=True)
        # att_mask: (**, sequence_length, 1)
        att_mask = att_mask.unsqueeze(-1)
        # output: (**, hidden_size)
        return (att_mask * embs).sum(-2) / avail_mask_count

    @staticmethod
    def _cls_pooling(embs: torch.FloatTensor, _: torch.LongTensor) -> torch.FloatTensor:
        '''
            `embeddings shape`: (**, sequence_length, hidden_size)
            `return shape`: (**, hidden_size)
            return cls token embeddings.
        '''
        # select sequence_length dim first index(cls embs).
        return torch.index_select(embs, -2, 0).squeeze(-2)
