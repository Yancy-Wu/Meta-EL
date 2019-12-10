'''
    predict candidate for given queries.
'''

from typing import List
from .config import Config

class Predictor(Config):
    '''
        a predictor, predict most similar candidates for given queries.
        candidates are initialized beforehand.
    '''
    def set_candidates(self, candidates: list):
        '''
            load candidates.
        '''
        raise NotImplementedError
    
    def predict(self, queries: list, topk: int = 1) -> List[list]:
        '''
            for queries, predict its candidates.
        '''
        raise NotImplementedError
