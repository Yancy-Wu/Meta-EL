'''
    REFERENCE: PROTOTYPICAL NETWORKS FOR FEW-SHOT LEARNING
    ICLR 2017.
'''

from typing import Dict
import torch
from torch import nn
from modules.protonet import Protonet

class PrototypicalNetwork(nn.Module):
    '''
        using bert as encoder, conbine with prototypical network.
        `pretrain_bert_dir`: bert pretrained model location.
        `config`: config['bert'] will be sent to bert model config.
        see META_EL.modules.bert for detail config parameters.
    '''

    def __init__(self, support_model: nn.Module, query_model: nn.Module):
        super().__init__()
        self.support_model = support_model
        self.query_model = query_model
        self.protonet = Protonet()

    # pylint: disable=arguments-differ
    def forward(self, support_tensor_map: Dict[str, torch.LongTensor],
                query_tensor_map: Dict[str, torch.LongTensor]):
        '''
            [DESCRIPTION]
              for each query item, return its p2 distance in terms of support proto.
            [PARAMS]
              `support_tensor_map item shape`: (tasks_num, ways_num, shots_num, support_seq_length)
              `query_tensor_map item shape`: (tasks_num, query_num, query_seq_length)
              `return shape`: (tasks_num, query_num, ways_num), no softmax.
        '''
        # generate all embeddings
        support_embs = self.support_model.forward(**support_tensor_map)
        query_embs = self.query_model.forward(**query_tensor_map)
        # return protonet distance
        return self.protonet.forward(support_embs, query_embs)
