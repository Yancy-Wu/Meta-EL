'''
    prediction class.
    for retriving most similar candidate according to input query.
'''

from typing import List, Any, Dict
from tqdm import trange
import numpy
import torch
import torch.nn.functional as F
from base.config import Config
from base.adapter import Adapter
from utils import deep_apply_dict
from utils.dataloader import DictDataLoader
from .models.similar_net import SimilarNet

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

    # batch size when generating
    EMB_BATCH_SIZE = 1000

    # device, where to save model and embs
    DEVICE = torch.device('cpu')

    # model and adaptar(adaptar for generating tensors, model for generating embs)
    similar_net: SimilarNet = None
    adapter: Adapter = None

    # saved candidate embs.
    # ids shape: [candidate_num], embs shape: [saved_candidate_num, hidden_size]
    candidate_ids: List[str] = None
    candidate_val: List[Any] = None
    candidate_embs: torch.Tensor = None

    def __init__(self, similar_net: SimilarNet, adapter: Adapter, conf: Dict = None):
        Config.__init__(self, conf)
        self.similar_net = similar_net.to(self.DEVICE)
        self.similar_net.eval()
        self.adapter = adapter

    def _generate_embs(self, inputs: List[Any], what: str):
        '''
            generate embeddings from inputs -> [input_num, hidden_size]
            what must be 'query' or 'candidate'
        '''

        print(f'\n[PREDICTION]: generating {what} embs.')
        # using python reflection to generating tensors.
        tensors_map = getattr(self.adapter, f'generate_{what}_tensors')(inputs)
        dataloader = DictDataLoader(tensors_map, {'batch_size': self.EMB_BATCH_SIZE})

        # start generating embs.
        embs_list = []
        progress = trange(0, len(dataloader), desc='generating')
        with torch.no_grad():
            for batch, _ in zip(dataloader, progress):
                # send to device, call model.
                deep_apply_dict(batch, lambda _, v: v.to(self.DEVICE))
                embs = getattr(self.similar_net, f'{what}_model')(**batch)
                embs_list.append(F.normalize(embs))

        progress.close()
        return torch.cat(embs_list)

    def add_candidate(self, candidate_ids: List[str], candidate_val: List[Any]):
        '''
            candidate type is same as `Example.y`.
            it will pre-calculate candidate embeddings and save them.
            for further predicting. [current you cannot run it twice]
        '''
        self.candidate_ids = candidate_ids
        self.candidate_val = candidate_val
        self.candidate_embs = self._generate_embs(self.candidate_val, 'candidate')

    def predict(self, queries: List[Any], topk: int = 1) -> List[List[Any]]:
        '''
            predict candidates with which are most similar.
            `return`: for each query, return a list of TOP K candidates id
        '''

        # query to embeddings.
        query_embs = self._generate_embs(queries, 'query')
        # query_embs shape: [query_num, hidden_size]
        # candidate_embs shape: [candidate_num, hidden_size]
        score = torch.matmul(query_embs, self.candidate_embs.transpose(0, 1))

        # fetch top k, convert to original candidate value.
        _, topk_indices = torch.topk(score, topk)
        to_str = numpy.frompyfunc(lambda i: self.candidate_val[i], 1, 1)
        return to_str(topk_indices.cpu().numpy())

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
    def load(cls, fn: str, device: torch.device):
        '''
            load prediction model from filename
        '''
        obj = torch.load(fn, map_location=device)
        obj.DEVICE = device
        return obj
