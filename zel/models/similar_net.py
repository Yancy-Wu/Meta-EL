'''
    similar net.
    for calculating query and candidate similarity.
'''
from typing import Dict
import torch
from torch import nn
from modules.activations import GeLU, Swish, GeLU_OpenAI
from base.config import Config

activation = {
    "gelu": GeLU,
    "relu": nn.ReLU,
    "swish": Swish,
    "gelu2": GeLU_OpenAI
}

class SimilarNet(nn.Module, Config):
    '''
        [DESCRIPTION]
          A net for calc similarity.
          output is a probability, for two classify task.
        [PARAMS]
          `query_model`: model who recepts query and generate output.
          `candidate_model`: model who recepts candidate and generate output.
          `hidden_size`: query_model and candidate_model output hidden_size.
          `config`: config['bert'] will be sent to bert model config.
    '''

    # drop out probability.
    DROP_OUT_PROB = 0.1

    # activition function name
    ACT_NAME = 'relu'

    # whether update bias item for self.proj
    USE_BIAS = False

    def __init__(self, query_model: nn.Module, candidate_model: nn.Module, hidden_size, conf=None):
        nn.Module.__init__(self)
        Config.__init__(self, conf)
        # save model.
        self.query_model = query_model
        self.candidate_model = candidate_model
        # projection layer.
        self.proj = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size, bias=self.USE_BIAS),
            activation[self.ACT_NAME](),
            nn.Linear(hidden_size, 2, bias=self.USE_BIAS)
        )

    # pylint: disable=arguments-differ
    def forward(self, query_tensor_map: Dict[str, torch.LongTensor],
                candidate_tensor_map: Dict[str, torch.LongTensor]):
        '''
            [DESCRIPTION]
              input query tensors(will be parsed by query_model), and candidate tensors.
              output probability where query and candidate pair is same.
            [PARAMS]
              `query_tensor_map` item shape: [batch_num, query_seq_len]
              `candidate_tensor_map` item shape`: [batch_num, candidate_seq_len]
              query_model and candidate_model return shape: [batch_num, hidden_size]
              `return shape`: [batch_num]
        '''
        # generate all embeddings
        query_emb = self.query_model.forward(**query_tensor_map)
        candidate_embs = self.candidate_model.forward(**candidate_tensor_map)
        differ_embs = (query_emb - candidate_embs).abs()
        # concat all embs and output probability
        features = torch.cat([query_emb, candidate_embs, differ_embs], dim=-1)
        return self.proj(features)
