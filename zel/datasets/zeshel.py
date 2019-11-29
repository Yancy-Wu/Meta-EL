'''
    context_zel csv file format: [~ as sep]
    documents/all.json: title ~ text ~ document_id
    mentions.json: category ~ text ~ end_index ~ context_document_id ~
                    label_document_id ~ mention_id ~ corpus ~ start_index
'''
from typing import List, Tuple, Any
import random
from tqdm import trange
import pandas
from . import ZelDataset

class Zeshel(ZelDataset):
    '''
        [DESCRIPTION]
          step 1: sample positive from all.json
          step 2: random sample negative. concat all as train.
        [PARAMS]
          `root`: zel datasets dir location.
    '''

    # using what worlds as datasets.
    WORLDS = ['final_fantasy', 'muppets']

    # document DataFrame: document_id - title - text
    # train DataFrame and valid DataFrame: mention_id - label_document_id - text
    _doc: pandas.DataFrame = None
    _train: pandas.DataFrame = None
    _valid: pandas.DataFrame = None

    def __init__(self, root, conf=None):
        super().__init__(conf)
        kargs = {'orient': 'records', 'lines': True}
        # generate documents
        ds = [pandas.read_json(f'{root}/documents/{world}.json', **kargs) for world in self.WORLDS]
        self._doc = pandas.concat(ds).set_index('document_id', drop=False)
        # generate train and val
        self._train = pandas.read_json(f'{root}/mentions/train.json', **kargs)
        self._valid = pandas.read_json(f'{root}/mentions/val.json', **kargs)
        self._train = self._train[self._train['corpus'].isin(self.WORLDS)]
        self._valid = self._valid[self._valid['corpus'].isin(self.WORLDS)]

    def all_candidates(self) -> Tuple[List[str], List[Any]]:
        id_list = self._doc['document_id'].tolist()
        val_list = self._doc['title'].tolist()
        return id_list, val_list

    def _generate(self, mention_record: pandas.Series, doc_record: pandas.Series) -> Example:
        '''
            given a mention record and doc record, generate an example.
        '''
        # query: mention, candidate: doc title
        query = mention_record['text']
        candidate = doc_record['title']
        y = mention_record['label_document_id'] == doc_record['document_id']
        return Example(query, candidate, y)

    def _negative_doc(self, mention_record: pandas.Series) -> pandas.Series:
        '''
            generate doc record whose id is not same as mention label document id.
              in fact. this function has very small probability to sample a positive example.
            `return`: a doc record.
        '''
        _ = mention_record
        # pandas.sample is too slow, i dont know why.
        sample_ix = random.sample(range(0, len(self._doc)), 1)[0]
        return self._doc.iloc[sample_ix]

    def _positive_doc(self, mention_record: pandas.Series) -> pandas.Series:
        '''
            generate doc record whose id is same as mention label document id.
            `return`: a doc record.
        '''
        doc_id = mention_record['label_document_id']
        return self._doc.loc[doc_id]

    def train_data(self) -> List[Example]:
        examples = []
        # generate examples 1:1 pos and neg
        progress = trange(0, len(self._train))
        for [_, mention_record], _ in zip(self._train.iterrows(), progress):
            positive_doc = self._positive_doc(mention_record)
            negative_doc = self._negative_doc(mention_record)
            examples.append(self._generate(mention_record, positive_doc))
            examples.append(self._generate(mention_record, negative_doc))
        progress.close()
        return examples

    def valid_data(self) -> List[Example]:
        examples = []
        # all positive
        progress = trange(0, len(self._valid))
        for [_, mention_record], _ in zip(self._valid.iterrows(), progress):
            positive_doc = self._positive_doc(mention_record)
            examples.append(self._generate(mention_record, positive_doc))
        progress.close()
        return examples

    def test_data(self) -> List[Example]:
        pass
