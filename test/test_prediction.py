'''
    test prediction load function.
'''

import sys
import torch
sys.path.append('./')

from zel.prediction import Prediction

FN = '../trained/zel_zeshel_bert_35388.pkl'

print('[PREDICTION]: loading...')
prediction = Prediction.load(FN, torch.device('cuda:0'))

print(prediction.predict(['apple', 'hello']))
