'''
    test zel on zeshel datasets.
'''

import os
import torch
from zel.str_predictor import StrPredictor
from zel.datasets.zeshel import Zeshel

def main():
    '''
        zel pipeline example.
    '''

    # create datasets. provide train and eval data.
    dataset = Zeshel('../datasets/zeshel')

    # train done, fetch bert model to prediction.
    tester = StrPredictor()
    # add candidates.
    tester.set_candidates(dataset.all_candidates())

    # we start test here.
    test_data = dataset.test_data()
    for i in config['top_what']:
        print(f'we test zeshel here: top-{i}')
        tester.test(test_data, i)

if __name__ == '__main__':
    main()
