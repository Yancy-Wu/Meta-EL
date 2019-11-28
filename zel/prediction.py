'''
    prediction class.
    for retriving most similar candidate according to input query.
'''

from typing import List, Any, Dict
import numpy
import torch
import torch.nn.functional as F
from base.config import Config
from .adapter import Adapter
from .similar_net import SimilarNet

class Prediction(Config):
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

    # maximum cached embs num.
    CACHED_EMBS_NUM = 100000

    # return top K candidate
    TOP_K = 32

    # num of candidate per batch checking(only work when embs are not full cached)
    CANDIDATE_BATCH_NUM = 1000

    # num of query predict batch num
    QUERY_BATCH_NUM = 100

    # device, prefer another GPU
    DEVICE = torch.device('cpu')

    # model and adaptar(adaptar for generating tensors, model for generating embs)
    similar_net: SimilarNet = None
    adapter: Adapter = None

    # saved candidate embs.
    # ids shape: [candidate_num], embs shape: [saved_candidate_num, hidden_size]
    candidate_ids: List[str] = None
    candidate_val: List[Any] = None
    saved_candidate_embs: torch.Tensor = None

    def __init__(self, similar_net: SimilarNet, adapter: Adapter, conf: Dict = None):
        Config.__init__(self, conf)
        self.similar_net = similar_net.to(self.DEVICE)
        self.adapter = adapter

    def _generate_embs(self, inputs: List[Any], what: str):
        '''
            generate embeddings from inputs -> [input_num, hidden_size]
            what must be 'query' or 'candidate'
        '''
        # using python reflection.
        tensors_map = getattr(self.adapter, f'generate_{what}_tensors')(inputs).to(self.DEVICE)
        embs = getattr(self.similar_net, f'{what}_model')(**tensors_map)
        # pre normalize, or other operation for easy compute similarity.
        return F.normalize(embs)

    def add_candidate(self, candidate_ids: List[str], candidate_val: List[Any]):
        '''
            candidate type is same as `Example.y`.
            it will pre-calculate candidate embeddings and save them.
            for further predicting. [current you cannot run it twice]
        '''
        self.candidate_ids = candidate_ids
        self.candidate_val = candidate_val
        saved_candidate = candidate_val[:self.CACHED_EMBS_NUM]
        self.saved_candidate_embs = self._generate_embs(saved_candidate, 'candidate')

    def predict(self, queries: List[Any]) -> List[List[Any]]:
        '''
            predict candidates with which are most similar.
            `return`: for each query, return a list of TOP K candidates id
        '''
        score = []
        # unsaved candidates.
        candidates = self.candidate_val[self.CACHED_EMBS_NUM:]

        # queries: [query_num]
        # query_embs shape: [query_batch_num, hidden_size]
        for i in range(0, len(queries), self.QUERY_BATCH_NUM):
            query_batch = queries[i:i + self.QUERY_BATCH_NUM]
            query_embs = self._generate_embs(query_batch, 'query')
            batch_score = [torch.matmul(query_embs, self.saved_candidate_embs.transpose())]

            # rank in unsaved candidates
            # candidate_embs shape: [candidate_batch_num, hidden_size]
            for j in range(self.CACHED_EMBS_NUM, len(self.candidate_val), self.CANDIDATE_BATCH_NUM):
                candidate_batch = candidates[j:j + self.CANDIDATE_BATCH_NUM]
                candidate_embs = self._generate_embs(candidate_batch, 'candidate')
                batch_score.append(torch.matmul(query_embs, candidate_embs.transpose()))

            # cat shape: [query_batch_num, candidate_num]
            score.append(torch.cat(batch_score))

        # score shape: [query_num, candidate_num]
        # topk shape: [query_num, k] -> [query_num * k]
        score = torch.cat(score)
        _, topk_indices = torch.topk(score, self.TOP_K)
        to_str = numpy.frompyfunc(lambda i: self.candidate_val[i], 1, 1)
        return to_str(topk_indices.numpy())
