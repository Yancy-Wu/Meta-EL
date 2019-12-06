'''
    eval support set overlap with train support set slightly.
    train/test_doc.json: ID ~ TEXT ~ TITLE
    train.json: LCONTEXT ~ MENTION ~ RCONTEXT ~ ID
    test.json: LCONTEXT ~ MENTION ~ RCONTEXT ~ ID ~ CANDIDATE
'''
import warnings
from typing import List
from tqdm import trange
import pandas as pd
from . import ZeshelDataset
from .. import Way, Task

class OverallZeshelDataset(ZeshelDataset):
    '''
        [DESCRIPTION]
          step 1: random sample mention examples from 'mentions.json' as query set
          step 2: fetch corresponding label_document text and title
                and random sample negative document as support set
        [PARAMS]
          `root`: OverallZeshelDataset dir location.
    '''

    # train way portion
    TRAIN_WAY_PORTION = 0.9

    # general mel must be 2 shots
    SHOTS_NUM_PRE_WAYS = None

    # document DataFrame: ID - TEXT - TITLE
    _train_doc: pd.DataFrame = None
    _test_doc: pd.DataFrame = None
    _train: pd.DataFrame = None
    _valid: pd.DataFrame = None
    _test: pd.DataFrame = None

    def __init__(self, root, conf=None):
        super().__init__(conf)

        # check parameters.
        if self.SHOTS_NUM_PRE_WAYS:
            warnings.warn('shots num will not work in overall mel setting')

        # read data. set index.
        kargs = {'orient': 'records', 'lines': True}
        self._train_doc = pd.read_json(f'{root}/train_doc.json', **kargs).set_index('ID', drop=False)
        self._test_doc = pd.read_json(f'{root}/test_doc.json', **kargs).set_index('ID', drop=False)
        self._train = pd.read_json(f'{root}/train.json', **kargs)
        self._test = pd.read_json(f'{root}/test_{self.WAYS_NUM_PRE_TASK}_ways.json', **kargs)

        # divide data.
        train_num = int(len(self._train) * self.TRAIN_WAY_PORTION)
        self._valid = self._train[train_num:]
        self._train = self._train[:train_num]

    # pylint: disable=arguments-differ
    def _generate_way(self, doc: pd.Series) -> Way:
        # using text and title as shots.
        return Way([doc['TEXT'], doc['TITLE']], doc['ID'])

    def _sample_task(self, df: pd.DataFrame) -> Task:
        # sample a meta-learning task from a DataFrame
        query = df.sample().iloc[0]
        noise_doc = self._train_doc.drop(index=query['ID']).sample(self.WAYS_NUM_PRE_TASK - 1)
        support = noise_doc.append(self._train_doc.loc[query['ID']])
        return self._generate_task(query, support)

    def train_data(self) -> List[Task]:
        # sample train task.
        with trange(0, self.TRAIN_TASKS_NUM) as progress:
            return [self._sample_task(self._train) for _ in progress]

    def valid_data(self) -> List[Task]:
        # sample valid task.
        with trange(0, self.VALID_TASKS_NUM) as progress:
            return [self._sample_task(self._valid) for _ in progress]

    def test_data(self) -> List[Task]:
        tasks = []
        # generate task per line
        with trange(0, len(self._test)) as progress:
            for (_, query), _ in zip(self._test.iterrows(), progress):
                support = self._test_doc.loc[query['CANDIDATE']]
                tasks.append(self._generate_task(query, support))
        # pad pad pad pad...
        return tasks
