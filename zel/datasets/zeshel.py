'''
    docs.json:
      TITLE ~ TEXT ~ ID ~ CORPUS
    train/test.json:
      ID ~ LCONTEXT ~ MENTION ~ RCONTEXT ~ CORPUS
'''
from typing import List
import random
from tqdm import trange
import pandas as pd
from . import ZelDataset, TrainExample, TestExample, Candidate

class Zeshel(ZelDataset):
    '''
        [DESCRIPTION]
          step 1: sample positive from docs.json
          step 2: random sample negative. concat all as train.
        [PARAMS]
          `root`: zel datasets dir location.
    '''
    # train way portion
    TRAIN_WAY_PORTION = 0.9

    # match key
    MATCH_KEY = 'TITLE'

    # document DataFrame: document_id - title - text
    # train DataFrame and valid DataFrame: mention_id - label_document_id - text
    _docs: pd.DataFrame = None
    _train: pd.DataFrame = None
    _valid: pd.DataFrame = None
    _test: pd.DataFrame = None

    def __init__(self, root, conf=None):
        super().__init__(conf)
        kargs = {'orient': 'records', 'lines': True}

        # generate documents
        examples = pd.read_json(f'{root}/train.json', **kargs)
        train_way_num = int(self.TRAIN_WAY_PORTION * len(examples))
        self._docs = pd.read_json(f'{root}/docs.json', **kargs).set_index('ID', drop=False)
        self._train = examples[:train_way_num]
        self._valid = examples[train_way_num:]
        self._test = pd.read_json(f'{root}/test.json', **kargs)

    def all_candidates(self) -> List[Candidate]:
        '''
            return all candidates, contain id and value.
        '''
        id_list = self._docs['ID'].tolist()
        val_list = self._docs[self.MATCH_KEY].tolist()
        return [Candidate(_val, _id) for (_val, _id) in zip(val_list, id_list)]

    def _generate(self, mention_record: pd.Series, doc_record: pd.Series) -> TrainExample:
        # query: mention, candidate: doc title or other
        query = mention_record['MENTION']
        candidate = doc_record[self.MATCH_KEY]
        y = mention_record['ID'] == doc_record['ID']
        return TrainExample(query, candidate, y)

    def _negative_doc(self, mention_record: pd.Series) -> pd.Series:
        '''
            generate doc record whose id is not same as mention label document id.
              in fact. this function has very small probability to sample a positive TrainExample.
            `return`: a doc record.
        '''
        _ = mention_record
        # pd.sample is too slow, i dont know why.
        sample_ix = random.sample(range(0, len(self._docs)), 1)[0]
        return self._docs.iloc[sample_ix]

    def _positive_doc(self, mention_record: pd.Series) -> pd.Series:
        '''
            generate doc record whose id is same as mention label document id.
            `return`: a doc record.
        '''
        doc_id = mention_record['ID']
        return self._docs.loc[doc_id]

    def _create_examples(self, df: pd.DataFrame) -> List[TrainExample]:
        # generate examples 1:1 pos and neg
        examples = []
        progress = trange(0, len(df))
        for [_, mention_record], _ in zip(df.iterrows(), progress):
            positive_doc = self._positive_doc(mention_record)
            negative_doc = self._negative_doc(mention_record)
            examples.append(self._generate(mention_record, positive_doc))
            examples.append(self._generate(mention_record, negative_doc))
        progress.close()
        random.shuffle(examples)
        return examples

    def train_data(self) -> List[TrainExample]:
        return self._create_examples(self._train)

    def valid_data(self) -> List[TrainExample]:
        return self._create_examples(self._valid)

    # test data return TestExamples
    def test_data(self) -> List[TestExample]:
        examples = []
        progress = trange(0, len(self._test))
        for [_, record], _ in zip(self._test.iterrows(), progress):
            examples.append(TestExample(record['MENTION'], record['ID']))
        progress.close()
        return examples
