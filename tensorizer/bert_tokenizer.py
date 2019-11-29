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

    # bert tokenizer
    bert_tokenizer = None

    @classmethod
    def from_pretrained(cls, model_dir, config=None):
        '''
            load tokenizer from pre-trained bert model.
            `model_dir`: model place.
            `config`: super-parameters config.
        '''
        if not config:
            config = dict()
        config.update({'bert_tokenizer': BertTokenizer.from_pretrained(model_dir)})
        return cls(config)

    # pylint: disable=arguments-differ
    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        '''
            generate bert model inputs, `return`:
            input_ids: text tokenization ids.
            att_mask: mask [PAD] tokens
        '''
        pad_token_id = self.bert_tokenizer.pad_token_id
        ids = self.bert_tokenizer.encode(text)
        # padding or clip to fixed length
        ids = (ids + [pad_token_id] * max(self.FIXED_LEN - len(ids), 0))[:self.FIXED_LEN]
        # add [CLS] and [SEP], then generate other tensor
        ids = self.bert_tokenizer.add_special_tokens_single_sentence(ids)
        att_mask = [0 if x == pad_token_id else 1 for x in ids]
        return {
            'input_ids': torch.LongTensor(ids),
            'att_mask': torch.LongTensor(att_mask)
        }
