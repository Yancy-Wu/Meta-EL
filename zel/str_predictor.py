'''
    str predictor. no training
    no fucking use. ACC all 0.
'''

from typing import List, Any, Dict, Tuple
from base.predictor import Predictor
from .datasets import TestExample, Candidate
import numpy
import tqdm

class StrPredictor(Predictor):
    '''
        using string similarity compare. as base base baseline.
    '''

    # save candiate ids and vals
    candidate_ids: List[str] = None
    candidate_vals: List[str] = None

    @staticmethod
    def _lcs(str_a, str_b):
        '''
            longest common substring of str_a and str_b, with O(n) space complexity
        '''
        if len(str_a) == 0 or len(str_b) == 0:
            return 0
        max_len = 0
        dp = [0 for _ in range(len(str_b) + 1)]
        for i in range(1, len(str_a) + 1):
            left_up = 0
            for j in range(1, len(str_b) + 1):
                up = dp[j]
            if str_a[i-1] == str_b[j-1]:
                dp[j] = left_up + 1
                max_len = max([max_len, dp[j]])
            else:
                dp[j] = 0
            left_up = up
        return max_len

    def set_candidates(self, candidates: List[Candidate]):
        '''
            load candidates.
            first ID, second string value.
        '''
        self.candidate_ids = [t.y for t in candidates]
        self.candidate_vals = [t.x for t in candidates]
    
    def predict(self, queries: list, topk: int = 1) -> List[list]:
        '''
            for queries, predict its candidates.
        '''
        topk_candidate = []
        for query in tqdm.tqdm(queries):
            score = [self._lcs(query, candidate) for candidate in self.candidate_vals]
            score = numpy.array(score)
            rank = numpy.argsort(-score)[:topk].tolist()
            ids = [self.candidate_ids[i] for i in rank]
            topk_candidate.append(ids)
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
