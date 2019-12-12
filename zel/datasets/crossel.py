'''
    kb.json:
      TITLE ~ TEXT ~ ID
    train/test.json:
      ID ~ MENTION ~ LAN
'''
from typing import List
import random
from tqdm import trange
import pandas as pd
from . import ZelDataset, TrainExample, TestExample, Candidate

class Crossel(ZelDataset):
    '''
        [DESCRIPTION]
          step 1: sample positive from kb.json
          step 2: random sample negative. concat all as train.
        [PARAMS]
          `root`: xel datasets dir location.
    '''
    # test langugage
    TEST_LANGUAGE = None

    # train way portion
    TRAIN_WAY_PORTION = 0.9

    # whether using context
    CANDIDATE_USING_TEXT = False

    # document DataFrame: document_id - title - text
    # train DataFrame and valid DataFrame: mention_id - label_document_id - text
    _kb: pd.DataFrame = None
    _train: pd.DataFrame = None
    _valid: pd.DataFrame = None
    _test: pd.DataFrame = None

    @staticmethod
    def _query_mention(q: pd.Series):
        return q['MENTION']

    @staticmethod
    def _candidate_title(c: pd.Series):
        return c['TITLE']

    @staticmethod
    def _candidate_desc(c: pd.Series):
        return c['TITLE'] + ' [SEP] ' + c['TEXT']

    def __init__(self, root, conf=None):
        super().__init__(conf)
        kargs = {'orient': 'records', 'lines': True}

        # set function
        self._query = self._query_mention
        self._candidate = self._candidate_desc if self.CANDIDATE_USING_TEXT else self._candidate_title

        # generate documents
        examples = pd.read_json(f'{root}/train.json', **kargs)
        train_way_num = int(self.TRAIN_WAY_PORTION * len(examples))
        self._kb = pd.read_json(f'{root}/kb.json', **kargs).set_index('ID', drop=False)
        self._train = examples[:train_way_num]
        self._valid = examples[train_way_num:]
        self._test = pd.read_json(f'{root}/test.json', **kargs)
        self._test = self._test[self._test['LAN'] == self.TEST_LANGUAGE]

    def all_candidates(self) -> List[Candidate]:
        '''
            return all candidates, contain id and value.
        '''
        id_list = self._kb['ID'].tolist()
        val_list = self._kb.apply(lambda c: self._candidate(c), axis=1).tolist()
        return [Candidate(_val, _id) for (_val, _id) in zip(val_list, id_list)]

    def _generate(self, mention_record: pd.Series, doc_record: pd.Series) -> TrainExample:
        # query: mention, candidate: doc title or other
        query = self._query(mention_record)
        candidate = self._candidate(doc_record)
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
        sample_ix = random.sample(range(0, len(self._kb)), 1)[0]
        return self._kb.iloc[sample_ix]

    def _positive_doc(self, mention_record: pd.Series) -> pd.Series:
        '''
            generate doc record whose id is same as mention label document id.
            `return`: a doc record.
        '''
        doc_id = mention_record['ID']
        return self._kb.loc[doc_id]

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
            examples.append(TestExample(self._query(record), record['ID']))
        progress.close()
        return examples
