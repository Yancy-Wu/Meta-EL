'''
    eval support set overlap with train support set slightly.
    documents.json: ID ~ TEXT ~ TITLE
    train/val/test.json: LCONTEXT ~ MENTION ~ RCONTEXT ~ ID
'''
import random
import warnings
from typing import List
from tqdm import trange
import pandas
from zel.tester import Tester
from . import ZeshelDataset
from .. import Way, Task

class GeneralZeshelDataset(ZeshelDataset):
    '''
        using zel prediction model to generate train and val task.
        `root`: GeneralZeshelDataset dir location.
        `prediction`: Prediction model object.
    '''

    # general mel must be 2 shots and 1 fixed query
    SHOTS_NUM_PRE_WAYS = None
    QUERY_NUM_PRE_WAY = None

    # document DataFrame: ID - TEXT - TITLE
    _doc: pandas.DataFrame = None
    _train: pandas.DataFrame = None
    _valid: pandas.DataFrame = None
    _test: pandas.DataFrame = None

    # zel prediction model
    tester: Tester = None

    def __init__(self, root: str, tester: Tester, conf=None):
        super().__init__(root, conf)
        self.tester = tester

        # check parameters.
        if self.SHOTS_NUM_PRE_WAYS or self.QUERY_NUM_PRE_WAY:
            warnings.warn('shots and query num will not work in general mel setting')

        # read data. set index.
        kargs = {'orient': 'records', 'lines': True}
        self._doc = pandas.read_json(f'{root}/documents.json', **kargs).set_index('ID')
        self._train = pandas.read_json(f'{root}/train.json', **kargs)
        self._valid = pandas.read_json(f'{root}/valid.json', **kargs)
        self._test = pandas.read_json(f'{root}/test.json', **kargs)

    # pylint: disable=arguments-differ
    def _generate_way(self, doc: pandas.Series) -> Way:
        # using text and title as shots.
        return Way([doc['TEXT'], doc['TITLE']], doc['ID'])

    def _create_task(self, pd: pandas.DataFrame, retain=False) -> List[Task]:
        # given a mention record, create an task. set retain to True when eval and test.
        tasks = []
        progress = trange(0, len(pd))
        predicted_doc_ids = self.tester.predict(pd['MENTION'].tolist(), self.WAYS_NUM_PRE_TASK)

        # if retain origin candidates, maybe there exists non-correct shot in support set.
        for i in progress:
            query, candidate_doc_ids = pd.iloc[i], predicted_doc_ids[i]

            # if not retain, check whether label is on candidate, otherwise replace randomly.
            if not retain and not query['ID'] in candidate_doc_ids:
                i = random.randint(0, self.WAYS_NUM_PRE_TASK - 1)
                candidate_doc_ids[i] = query['ID']

            # generate task
            candidate_docs = self._doc.loc[candidate_doc_ids]
            tasks.append(self._generate_task(query, candidate_docs))

        progress.close()
        return tasks

    def train_data(self) -> List[Task]:
        return self._create_task(self._train[:self.TRAIN_TASKS_NUM], False)

    def valid_data(self) -> List[Task]:
        return self._create_task(self._valid[:self.VALID_TASKS_NUM], True)

    def test_data(self) -> List[Task]:
        return self._create_task(self._test, True)
