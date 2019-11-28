'''
    so easy and fast task!
    just run on CPU!
'''

import torch
from tokenizer.bert_tokenizer import EasyBertTokenizer
from mel.pipeline import MetaPipeline
from mel.datasets.zeshel import Zeshel
from mel.models.prototypical_network import PrototypicalNetwork

def main():
    ''' entry point '''
    MetaPipeline({
        'dataset': Zeshel('./datasets/zeshel/', {
            'TRAIN_TASKS_NUM': 1000,
            'VALID_TASKS_NUM': 20,
            'WAYS_NUM_PRE_TASK': 5,
            'SHOTS_NUM_PRE_WAYS': 2,
            'QUERY_NUM_PRE_WAY': 1
        }),
        'tokenizer': EasyBertTokenizer.from_pretrained('./pretrain/uncased_L-12_H-768_A-12', {
            'FIXED_LEN': 64
        }),
        'model': PrototypicalNetwork('./pretrain/uncased_L-12_H-768_A-12', {
            'bert':{
                'POOLING_METHOD': 'avg'
            },
            'protonet':{}
        })
    }, torch.device('cpu'), 20).train()

if __name__ == '__main__':
    main()
