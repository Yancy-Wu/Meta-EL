'''
    tokenizer interface.
    convert a str to tensor.
'''

from typing import Dict, Any
import torch
from base.config import Config

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
