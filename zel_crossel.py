'''
    test xel on crossel datasets.
'''

import os
import torch
from utils.trainer import Trainer
from zel.models.similar_net import SimilarNet
from zel.zel_predictor import ZelPredictor
from zel.zel_adapter import ZelAdapter
from zel.datasets.crossel import Crossel
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
    dataset = Crossel('../datasets/crossel', {
        'TRAIN_WAY_PORTION': 0.9,
        'MATCH_KEY': config['match_key'],
        'TEST_LANGUAGE': 'uk' # it doesn't matter.
    })

    # tensorizer. convert an example to tensors.
    tensorizer = EasyBertTokenizer.from_pretrained('../pretrain/multi_cased_L-12_H-768_A-12', {
        'FIXED_LEN': config['fixed_len'],
        'DO_LOWER_CASE': True
    })

    # adapter. call tensorizer, convert a batch of examples to big tensors.
    adapter = ZelAdapter(tensorizer, tensorizer)

    # embedding model. for predication.
    bert = Bert.from_pretrained('../pretrain/multi_cased_L-12_H-768_A-12', {
        'POOLING_METHOD': 'avg',
        'FINETUNE_LAYER_RANGE': '9:12'
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
        'TRAIN_BATCH_SIZE': 300,
        'VALID_BATCH_SIZE': 1000,
        'ROUND': 2
    })

    # train start here.
    trainer.train()

    # train done, fetch bert model to prediction.
    tester = ZelPredictor(model, adapter, {
        'TEST_BATCH_SIZE': 200,
        'EMB_BATCH_SIZE': 1000,
        'DEVICE': torch.device(config['device'])
    })
    # add candidates.
    tester.set_candidates(dataset.all_candidates())
    tester.save(config['saved_file'])

    # we start test here.
    for lan in config['lan']:
        print(f'we are testing lan: {lan}')
        test_dataset = Crossel('../datasets/crossel', {'TEST_LANGUAGE': lan})
        test_data = test_dataset.test_data()
        for i in config['top_what']:
            print(f'now top-{i} ACC:')
            tester.test(test_data, i)

if __name__ == '__main__':
    for match_key, fixed_len in zip(['TEXT', 'TITLE'], [32, 16]):
        main({
            'match_key': match_key,
            'fixed_len': fixed_len,
            'device': 'cuda:0',
            'lan': ['bn', 'jv', 'mr', 'pa', 'te', 'uk'],
            'top_what': [64, 32, 16, 8, 4, 2, 1],
            'saved_file': f'./saved/crossel_zel/{match_key}.bin'
        })
