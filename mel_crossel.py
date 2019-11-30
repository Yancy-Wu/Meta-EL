'''
    as simple as xel!
    just run on CPU!
    ACC is 100%
'''

import torch
from tensorizer.bert_tokenizer import EasyBertTokenizer
from modules.bert import Bert
from utils.trainer import Trainer
from mel.datasets.crossel.single import SingleCrosselDataset
from mel.models.prototypical_network import PrototypicalNetwork
from mel.mel_adapter import MelAdapter

def main():
    ''' entry point '''
    # create datasets. provide train and eval data.
    dataset = SingleCrosselDataset('../datasets/cross_el/', 'bn', {
        'TRAIN_TASKS_NUM': 1000,
        'VALID_TASKS_NUM': 50,
        'WAYS_NUM_PRE_TASK': 6,
        'SHOTS_NUM_PRE_WAYS': 1,
        'QUERY_NUM_PRE_WAY': 0.5
    })

    # tensorizer. convert an example to tensors.
    tensorizer = EasyBertTokenizer.from_pretrained('../pretrain/uncased_L-12_H-768_A-12', {
        'FIXED_LEN': 32,
        'DO_LOWER_CASE': True
    })

    # adapter. call tensorizer, convert a batch of examples to big tensors.
    adapter = MelAdapter(tensorizer, tensorizer)

    # embedding model. for predication.
    bert = Bert.from_pretrained('../pretrain/uncased_L-12_H-768_A-12', {
        'POOLING_METHOD': 'avg',
        'FINETUNE_LAYER_RANGE': '1:12'
    })

    # prototypical network for training.
    model = PrototypicalNetwork(bert, bert)

    # trainer. to train siamese bert.
    trainer = Trainer({
        'dataset': dataset,
        'adapter': adapter,
        'model': model,
        'DEVICE': torch.device('cuda:2'),
        'TRAIN_BATCH_SIZE': 20,
        'VALID_BATCH_SIZE': 20,
        'ROUND': 20
    })

    # train start here.
    trainer.train()

if __name__ == '__main__':
    main()
