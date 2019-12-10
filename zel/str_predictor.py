'''
    str predictor. no training
'''

from typing import List, Any, Dict, Tuple
from base.predictor import Predictor
from .datasets import TestExample, Candidate
import Levenshtein
import numpy

class StrPredictor(Predictor):
    '''
        using string similarity compare.
        Levenshtein Distance algorithm. as base base baseline.
    '''

    # save candiate ids and vals
    candidate_ids: List[str] = None
    candidate_vals: List[str] = None

    def set_candidates(self, candidates: List[Candidate]):
        '''
            load candidates.
            first ID, second string value.
        '''
        self.candidate_ids = [t.x for t in candidates]
        self.candidate_vals = [t.y for t in candidates]
    
    def predict(self, queries: list, topk: int = 1) -> List[list]:
        '''
            for queries, predict its candidates.
        '''
        topk_candidate = []
        for query in queries:
            score = [Levenshtein.distance(query, candidate) for candidate in self.candidate_vals]
            score = numpy.array(score)
            rank = numpy.argsort(-score)[:topk]
            topk_candidate.append(self.candidate_ids[rank])
        return topk_candidate

    def test(self, test_data: List[TestExample], topk: int = 1):
        '''
            test a group of examples, calulate top k ACC.
        '''
        print('[TESTER]: now begin to test:')
        correct_num = 0
        xs = [example.x for example in test_data]
        ys = [example.y for example in test_data]
        predict_kys = self.predict(xs, topk=topk)
        for y, predict_ky in zip(ys, predict_kys):
            correct_num += 1 if y in predict_ky else 0
        # calulate ACC here.
        print('final ACC: (%.4f)' % (float(correct_num) / len(test_data)))
