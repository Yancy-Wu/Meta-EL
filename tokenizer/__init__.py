'''
    tokenizer interface.
    convert a str to tensor.
'''

from typing import Dict
import torch
from META_EL.base.config import Config

class Tokenizer(Config):
    '''
        a tokenizer.
        ``
    '''

    # pad or clip to maintain input length.
    FIXED_LEN = 32

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        '''
            encode an text.
            return a dict of tensors
            `text`: input text.
        '''
        raise NotImplementedError
