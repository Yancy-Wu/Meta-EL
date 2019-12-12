'''
    test and prediction class.
    for retriving most similar candidate according to input query.
'''

from typing import List, Any, Dict
from tqdm import trange
import numpy
import torch
import torch.nn.functional as F
from base.config import Config
from base.predictor import Predictor
from utils import deep_apply_dict
from utils.dataloader import DictDataLoader
from .models.similar_net import SimilarNet
from .datasets import TestExample, Candidate
from .zel_adapter import ZelAdapter

class ZelPredictor(Predictor):
    '''
        [DESCRIPTION]
          initilize me by *trained* model and adapter.
          add candidates, then you can call predict to find most similar top K candidate
          support embedding cache and support multiple similarity calculate algorithm.
        [PARAMS]
          `similar_net`: *trained* model, for calc candidate and query embs.
          `adapter`: for tesorizing candidate and query.(two tensorizer)
          `conf`: set parameters.
    '''

    # batch size when generating
    EMB_BATCH_SIZE = 1000

    # test batch size when testing
    TEST_BATCH_SIZE = 100

    # device, where to save model and embs
    DEVICE = torch.device('cpu')

    # model and adaptar(adaptar for generating tensors, model for generating embs)
    similar_net: SimilarNet = None
    adapter: ZelAdapter = None

    # saved candidate embs.
    # ids shape: [candidate_num], embs shape: [saved_candidate_num, hidden_size]
    candidate_labels: List[str] = None
    candidate_embs: torch.Tensor = None

    def __init__(self, similar_net: SimilarNet, adapter: ZelAdapter, conf: Dict = None):
        Predictor.__init__(self, conf)
        self.similar_net = similar_net.to(self.DEVICE)
        self.similar_net.eval()
        self.adapter = adapter

    def set_candidates(self, candidates: List[Candidate]):
        '''
            candidate type is same as `Example.y`.
            it will pre-calculate candidate embeddings and save them.
            for further predicting.
        '''
        # save label
        self.candidate_labels = [candidate.y for candidate in candidates]
        self.candidate_embs = []

        # load tensors.
        candidate_val = [candidate.x for candidate in candidates]
        tensors_map = self.adapter.generate_candidate_tensors(candidate_val)
        dataloader = DictDataLoader(tensors_map, {'batch_size': self.EMB_BATCH_SIZE})

        # calculate here.
        with trange(0, len(dataloader)) as progress:
            with torch.no_grad():
                for batch, _ in zip(dataloader, progress):
                    # emb shape: (EMB_BATCH_SIZE, hidden_size)
                    deep_apply_dict(batch, lambda _, v: v.to(self.DEVICE))
                    embs = F.normalize(self.similar_net.candidate_model(**batch), dim=-1)
                    self.candidate_embs.append(embs)

        # concat candidate embs
        self.candidate_embs = torch.cat(self.candidate_embs)

    def predict(self, queries: List[Any], topk: int = 1) -> List[List[Any]]:
        '''
            predict candidates with which are most similar.
            `return`: for each query, return a list of TOP K candidates id
        '''
        # ready to calculate.
        topk_candidate = []
        tensors_map = self.adapter.generate_query_tensors(queries)
        dataloader = DictDataLoader(tensors_map, {'batch_size': self.TEST_BATCH_SIZE})

        # do predict.
        with trange(0, len(dataloader)) as progress:
            with torch.no_grad():
                for batch, _ in zip(dataloader, progress):
                    # embs shape: [TEST_BATCH_SIZE, hidden_size]
                    deep_apply_dict(batch, lambda _, v: v.to(self.DEVICE))
                    embs = F.normalize(self.similar_net.query_model(**batch), dim=-1)
                    # score shape: [TEST_BATCH_SIZE, candidate_num]
                    score = torch.matmul(embs, self.candidate_embs.transpose(0, 1))
                    # topk_indices shape: [TEST_BATCH_SIZE, topK_candidate]
                    _, topk_indices = torch.topk(score, topk)
                    topk_candidate.append(topk_indices)

        # return all candidate
        topk_candidate = torch.cat(topk_candidate)
        to_str = numpy.frompyfunc(lambda i: self.candidate_labels[i], 1, 1)
        return to_str(topk_candidate.cpu().numpy())

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

    def save(self, fn: str):
        '''
            save prediction model for re-using.
            `fn`: saved file name.
        '''
        import warnings
        warnings.filterwarnings('ignore')
        torch.save(self, fn)
        warnings.resetwarnings()

    @classmethod
    def load(cls, fn: str, conf: dict):
        '''
            load prediction model from filename
        '''
        device = conf.get('DEVICE')
        self = torch.load(fn, map_location=device)
        Config.__init__(self, conf)
        return self
