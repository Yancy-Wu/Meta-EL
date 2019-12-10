'''
    bert tokenizer
'''

from typing import Dict
from pytorch_transformers import BertTokenizer
import torch
from base.tensorizer import Tensorizer

class EasyBertTokenizer(Tensorizer):
    '''
        convert text to bert format tensors.
        include an input_ids, att_mask and token_type.
    '''

    # pad or clip to maintain input length.
    FIXED_LEN = 32

    # whether do lower case
    DO_LOWER_CASE = True

    # bert tokenizer
    tokenizer = None

    @classmethod
    def from_pretrained(cls, model_dir, config=None):
        '''
            load tokenizer from pre-trained bert model.
            `model_dir`: model place.
            `config`: super-parameters config.
        '''
        self = cls(config)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=self.DO_LOWER_CASE)
        return self

    # pylint: disable=arguments-differ
    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        '''
            generate bert model inputs, `return`:
            input_ids: text tokenization ids.
            att_mask: mask [PAD] tokens
        '''
        text = ' '.join(text.split()[:2 * self.FIXED_LEN])
        pad_token_id = self.tokenizer.pad_token_id
        ids = self.tokenizer.encode(text)
        # padding or clip to fixed length
        ids = (ids + [pad_token_id] * max(self.FIXED_LEN - len(ids), 0))[:self.FIXED_LEN]
        # add [CLS] and [SEP], then generate other tensor
        ids = self.tokenizer.add_special_tokens_single_sentence(ids)
        att_mask = [0 if x == pad_token_id else 1 for x in ids]
        return {
            'input_ids': torch.LongTensor(ids),
            'att_mask': torch.LongTensor(att_mask)
        }
