'''
    easy bert modules.
'''

import torch
from torch import nn
from pytorch_transformers import BertModel, BertConfig
from base.config import Config

class Bert(nn.Module, Config):
    '''
        support different pooling method
    '''

    # pooling method, avail value: cls, avg
    POOLING_METHOD = 'avg'

    # fine-tune layers
    FINETUNE_LAYER_RANGE = '1:12'

    # bert config
    config: BertConfig = None

    def __init__(self, config=None):
        nn.Module.__init__(self)
        Config.__init__(self, config)
        self._unfreeze(self.FINETUNE_LAYER_RANGE)

    @classmethod
    def from_pretrained(cls, model_dir, config=None):
        '''
            load bert pre-train model from disk
            `model_dir`: model location
        '''
        config = config if config else dict()
        bert = BertModel.from_pretrained(model_dir)
        config.update({'bert': bert, 'config': bert.config})
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
        embs = self.bert.forward(input_ids, att_mask)[0]
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

    def _unfreeze(self, layer_range: str):
        """
            set which layers can be fine-tuned
            `layer_range`: to fine tune layer, such as 9:10
        """
        s = layer_range.split(':')
        assert len(s) == 2
        i = int(s[0])
        j = int(s[1])

        for p in self.bert.parameters():
            p.requires_grad = False  # self.bert.pooler is off here

        # embeddings
        if i == 0:
            # pylint: disable=no-member
            for name, p in self.bert.embeddings.named_parameters():
                p.requires_grad = True

        # layers
        for l in range(max(i - 1, 0), j):
            # pylint: disable=no-member
            for name, p in self.bert.encoder.named_parameters():
                if str(l) in name:
                    p.requires_grad = True
