'''
    a little difficult. very slow when running on CPU
'''

import torch
from tokenizer.bert_tokenizer import EasyBertTokenizer
from mel.datasets.context_zel import ContextZel
from mel.models.prototypical_network import PrototypicalNetwork
from mel.pipeline import MetaPipeline

def main():
    ''' entry point '''
    MetaPipeline({
        'dataset': ContextZel('./datasets/context_zel', {
            'TRAIN_TASKS_NUM': 600,
            'VALID_TASKS_NUM': 25,
            'WAYS_NUM_PRE_TASK': 5,
            'SHOTS_NUM_PRE_WAYS': 2,
            'QUERY_NUM_PRE_WAY': 0.8
        }),
        'tokenizer': EasyBertTokenizer.from_pretrained('./pretrain/multi_cased_L-12_H-768_A-12', {
            'FIXED_LEN': 200
        }),
        'model': PrototypicalNetwork('./pretrain/multi_cased_L-12_H-768_A-12', {
            'bert':{
                'POOLING_METHOD': 'avg'
            },
            'protonet':{}
        })
    }, torch.device('cuda:1'), 1).train()

if __name__ == '__main__':
    main()
