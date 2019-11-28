'''
    as simple as xel!
    just run on CPU!
    ACC is 100%
'''

import torch
from tokenizer.bert_tokenizer import EasyBertTokenizer
from mel.pipeline import MetaPipeline
from mel.datasets.single_xel import SingleXel
from mel.models.prototypical_network import PrototypicalNetwork

def main():
    ''' entry point '''
    MetaPipeline({
        'dataset': SingleXel('./datasets/pivot-based-el-data/', 'bn', {
            'TRAIN_TASKS_NUM': 1000,
            'VALID_TASKS_NUM': 20,
            'WAYS_NUM_PRE_TASK': 4,
            'SHOTS_NUM_PRE_WAYS': 1,
            'QUERY_NUM_PRE_WAY': 0.5
        }),
        'tokenizer': EasyBertTokenizer.from_pretrained('./pretrain/multi_cased_L-12_H-768_A-12', {
            'FIXED_LEN': 32
        }),
        'model': PrototypicalNetwork('./pretrain/multi_cased_L-12_H-768_A-12', {
            'bert':{
                'POOLING_METHOD': 'avg'
            },
            'protonet':{}
        })
    }, torch.device('cpu'), 20).train()

if __name__ == '__main__':
    main()
