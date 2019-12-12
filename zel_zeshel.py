'''
    test zel on zeshel datasets.
'''

import os
import torch
import sys
from utils.trainer import Trainer
from zel.models.similar_net import SimilarNet
from zel.zel_predictor import ZelPredictor
from zel.zel_adapter import ZelAdapter
from zel.datasets.zeshel import Zeshel
from tensorizer.bert_tokenizer import EasyBertTokenizer
from modules.bert import Bert

def main(config: dict):
    '''
        zel pipeline example.
    '''
    # print tag.
    print(10 * '*', __file__, config)
    saved_dir = os.path.dirname(config['saved_file'])
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # create datasets. provide train and eval data.
    dataset = Zeshel('../datasets/zeshel', {
        'TRAIN_WAY_PORTION': 0.9,
        'QUERY_USING_CONTEXT': config['context'],
        'CANDIDATE_USING_TEXT': config['context']
    })

    # tensorizer. convert an example to tensors.
    tensorizer = EasyBertTokenizer.from_pretrained('../pretrain/uncased_L-12_H-768_A-12', {
        'FIXED_LEN': config['fixed_len'],
        'DO_LOWER_CASE': True
    })

    # adapter. call tensorizer, convert a batch of examples to big tensors.
    adapter = ZelAdapter(tensorizer, tensorizer)

    # embedding model. for predication.
    bert = Bert.from_pretrained('../pretrain/uncased_L-12_H-768_A-12', {
        'POOLING_METHOD': 'avg',
        'FINETUNE_LAYER_RANGE': '8:12'
    })

    # siamese bert for training.
    model = SimilarNet(bert, bert, bert.config.hidden_size, {
        'DROP_OUT_PROB': 0.1,
        'ACT_NAME': 'relu',
        'USE_BIAS': False
    })

    # trainer. to train siamese bert.
    trainer = Trainer({
        'dataset': dataset,
        'adapter': adapter,
        'model': model,
        'DEVICE': torch.device(config['device']),
        'TRAIN_BATCH_SIZE': 50,
        'VALID_BATCH_SIZE': 500,
        'ROUND': 2
    })
    # train start here.
    trainer.train()

    # train done, fetch bert model to prediction.
    tester = ZelPredictor(model, adapter, {
        'TEST_BATCH_SIZE': 100,
        'EMB_BATCH_SIZE': 1000,
        'DEVICE': torch.device(config['device'])
    })
    # add candidates.
    tester.set_candidates(dataset.all_candidates())
    tester.save(config['saved_file'])

    # we start test here.
    test_data = dataset.test_data()
    for i in config['top_what']:
        print(f'we test zeshel here: top-{i}')
        tester.test(test_data, i)
        sys.stdout.flush()

if __name__ == '__main__':
    for context, fixed_len in zip([1, 0], [72, 16]):
        main({
            'context': bool(context),
            'fixed_len': fixed_len,
            'device': 'cuda:1',
            'top_what': [100, 50, 20, 10, 5, 2, 1],
            'saved_file': f'./saved/zeshel_zel/context_{context}.bin'
        })
