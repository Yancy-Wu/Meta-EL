'''
    base tensorizer class, for zel and mel.
'''

from typing import Dict, Any
import torch
from .config import Config

class Tensorizer(Config):
    '''
        a Tensorizer define.
    '''

    def encode(self, source: Any) -> Dict[str, torch.Tensor]:
        '''
            encode an input.
            return a dict of tensors
            `source`: input source.
        '''
        raise NotImplementedError
