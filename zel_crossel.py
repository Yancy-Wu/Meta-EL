'''
    test xel on crossel datasets.
'''

import os
import torch
from utils.trainer import Trainer
from zel.models.similar_net import SimilarNet
from zel.tester import Tester
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
    if not os.path.exists(config['saved_dir']):
        os.makedirs(config['saved_dir'])

    # create datasets. provide train and eval data.
    dataset = Crossel('../datasets/crossel', {
        'TRAIN_WAY_PORTION': 0.9,
        'TEST_LANGUAGE': 'uk', # it doesn't matter
        'MATCH_KEY': config['match_key']
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
        'TRAIN_BATCH_SIZE': 1000,
        'VALID_BATCH_SIZE': 1000,
        'ROUND': 2
    })

    # train start here.
    trainer.train()

    # train done, fetch bert model to prediction.
    tester = Tester(model, adapter, {
        'TEST_BATCH_SIZE': 100,
        'EMB_BATCH_SIZE': 1000,
        'DEVICE': torch.device(config['device'])
    })
    # add candidates.
    tester.set_candidates(dataset.all_candidates())
    tester.save(config['saved_dir'])

    # we start test here.
    for lan in config['lan']:
        test_dataset = Crossel('../datasets/crossel', {'TEST_LANGUAGE': lan})
        for i in config['top_what']:
            print(f'we test crossel here: lan-{lan}, top-{i}')
            tester.test(test_dataset.test_data(), i)

if __name__ == '__main__':
    for match_key, fixed_len in zip(['TITLE', 'TEXT'], [16, 32]):
        for top in [64, 32, 16, 8, 4, 2, 1]:
            main({
                'match_key': match_key,
                'fixed_len': fixed_len,
                'device': 'cpu',
                'saved_dir': f'./saved/crossel/zel_{match_key}.pkl',
                'lan': ['bn', 'jv', 'mr', 'pa', 'te', 'uk']
            })
