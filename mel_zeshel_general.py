'''
    zeshel general datasets.
'''

import os
import torch
import sys
from tensorizer.bert_tokenizer import EasyBertTokenizer
from modules.bert import Bert
from utils.trainer import Trainer
from mel.datasets.zeshel.general import GeneralZeshelDataset
from mel.models.prototypical_network import PrototypicalNetwork
from mel.mel_adapter import MelAdapter
from zel.zel_predictor import ZelPredictor

def main(config: dict):
    ''' entry point '''
    # print tag.
    print(10 * '*', __file__, config)
    if not os.path.exists(config['saved_dir']):
        os.makedirs(config['saved_dir'])

    # set zel predictor
    predictor = ZelPredictor.load(config['predictor'], {
        'TEST_BATCH_SIZE': 200,
        'DEVICE': torch.device(config['device'])
    })

    # create datasets. provide train and eval data.
    dataset = GeneralZeshelDataset('../datasets/zeshel/', predictor, {
        'TRAIN_TASKS_NUM': 4000,
        'VALID_TASKS_NUM': 300,
        'WAYS_NUM_PRE_TASK': config['way_num'][0],
        'QUERY_USING_CONTEXT': config['context'],
    })

    # tensorizer. convert an example to tensors.
    tensorizer = EasyBertTokenizer.from_pretrained('../pretrain/uncased_L-12_H-768_A-12', {
        'FIXED_LEN': 72,
        'DO_LOWER_CASE': True
    })

    # adapter. call tensorizer, convert a batch of examples to big tensors.
    adapter = MelAdapter(tensorizer, tensorizer)

    # embedding model. for predication.
    bert = Bert.from_pretrained('../pretrain/uncased_L-12_H-768_A-12', {
        'POOLING_METHOD': 'avg',
        'FINETUNE_LAYER_RANGE': '9:12'
    })

    # prototypical network for training.
    model = PrototypicalNetwork(bert, bert)

    # trainer. to train siamese bert.
    trainer = Trainer({
        'dataset': dataset,
        'adapter': adapter,
        'model': model,
        'DEVICE': torch.device(config['device']),
        'TRAIN_BATCH_SIZE': 2,
        'VALID_BATCH_SIZE': 5,
        'ROUND': 3
    })

    # train start here.
    trainer.train()

    # test here.
    for way_num in config['way_num']:
        test_dataset = GeneralZeshelDataset('../datasets/zeshel/', predictor, {
            'TRAIN_TASKS_NUM': 0,
            'VALID_TASKS_NUM': 0,
            'WAYS_NUM_PRE_TASK': way_num,
            'QUERY_USING_CONTEXT': config['context']
        })
        trainer.test(test_dataset.test_data())
        sys.stdout.flush()

    # save model
    bert.save_pretrained(config['saved_dir'])

if __name__ == '__main__':
    for using_context in [1, 0]:
        main({
            'way_num': [50, 20, 10, 5, 2],
            'context': bool(using_context),
            'device': 'cuda:0',
            'saved_dir': f'./saved/zeshel_general/context_{using_context}',
            'predictor': './saved/zeshel_zel/context_0.bin'
        })
