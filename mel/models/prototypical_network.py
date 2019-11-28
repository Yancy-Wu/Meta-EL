'''
    REFERENCE: PROTOTYPICAL NETWORKS FOR FEW-SHOT LEARNING
    ICLR 2017.
'''

import torch
from torch import nn
from modules import bert, protonet

class PrototypicalNetwork(nn.Module):
    '''
        using bert as encoder, conbine with prototypical network.
        `pretrain_bert_dir`: bert pretrained model location.
        `config`: config['bert'] will be sent to bert model config.
        see META_EL.modules.bert for detail config parameters.
    '''

    def __init__(self, pretrain_bert_dir, config=None):
        super().__init__()
        conf = config if config else dict()
        self.bert = bert.Bert.from_pretrained(pretrain_bert_dir, conf.get('bert', None))
        self.protonet = protonet.Protonet(conf.get('protonet', None))

    # pylint: disable=arguments-differ
    def forward(self, support_input_ids: torch.LongTensor, support_att_mask: torch.LongTensor,
                query_input_ids: torch.LongTensor, query_att_mask: torch.LongTensor):
        '''
            `support_input_ids shape`: (tasks_num, ways_num, shots_num, sequence_length)
            `support_att_masks shape`: (tasks_num, ways_num, shots_num, sequence_length)
            `query_input_ids shape`: (tasks_num, query_num, sequence_length)
            `query_att_masks shape`: (tasks_num, query_num, sequence_length)
            `return shape`: (tasks_num, query_num, ways_num), no softmax.
        '''
        # support_embs: (tasks_num, ways_num, shots_num, hidden_size)
        # query_embs: (tasks_num, query_num, hidden_size)
        support_embs = self.bert.forward(support_input_ids, support_att_mask)
        query_embs = self.bert.forward(query_input_ids, query_att_mask)
        # return protonet distance
        return self.protonet.forward(support_embs, query_embs)
