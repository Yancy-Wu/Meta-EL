'''
    prototypical network implementation.
'''

import torch
from torch import nn
from META_EL.base.config import Config

class Protonet(nn.Module, Config):
    '''
        PROTOTYPICAL NETWORKS FOR FEW-SHOT LEARNING
        ICLR 2017.
    '''
    class Prototype(nn.Module):
        '''
            get prototype of a group of examples embeddings
        '''
        # pylint: disable=arguments-differ
        def forward(self, embs: torch.FloatTensor):
            '''
                `embs shape`: (**, ways_num, shots_num, example_emb_size)
                `return shape`: (**, ways_num, example_emb_size)
            '''
            return embs.mean(-2)

    def __init__(self, config=None):
        nn.Module.__init__(self)
        Config.__init__(self, config)
        self.proto = Protonet.Prototype()

    # pylint: disable=arguments-differ
    def forward(self, embs: torch.FloatTensor, query_embs: torch.FloatTensor):
        '''
            ** stand for any dim size
            `embs` shape: (**, ways_num, shots_num, example_emb_size)
            `query_embs` shape: (**, query_num, example_emb_size)
            `return shape`: (**, query_num, ways_num), only E distance, no softmax
        '''
        # fetch ways prototype embeddings.
        proto_embs: torch.FloatTensor = self.proto(embs)
        # shape: (**, ways_num, example_emb_size) -> (**, 1, ways_num, example_emb_size)
        proto_embs = proto_embs.unsqueeze(-3)
        # shape: (**, query_num, example_emb_size) -> (**, query_num, 1, example_emb_size)
        query_embs = query_embs.unsqueeze(-2)
        return ((proto_embs - query_embs) ** 2).sum(-1) ** 0.5
