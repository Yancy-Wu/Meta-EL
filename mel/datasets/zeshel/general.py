'''
    eval support set overlap with train support set slightly.
    docs.json: ID ~ TEXT ~ TITLE
    train/test.json: LCONTEXT ~ MENTION ~ RCONTEXT ~ ID
'''
import random
import warnings
from typing import List
from tqdm import trange
import pandas
from base.predictor import Predictor
from . import ZeshelDataset
from .. import Way, Task

class GeneralZeshelDataset(ZeshelDataset):
    '''
        using zel prediction model to generate train and val task.
        `root`: GeneralZeshelDataset dir location.
        `prediction`: Prediction model object.
    '''
    # train way portion
    TRAIN_WAY_PORTION = 0.9

    # general mel must be 2 shots
    SHOTS_NUM_PRE_WAYS = None

    # document DataFrame: ID - TEXT - TITLE
    _doc: pandas.DataFrame = None
    _train: pandas.DataFrame = None
    _valid: pandas.DataFrame = None
    _test: pandas.DataFrame = None

    # zel prediction model
    predictor: Predictor = None

    def __init__(self, root: str, predictor: Predictor, conf=None):
        super().__init__(conf)
        self.predictor = predictor

        # check parameters.
        if self.SHOTS_NUM_PRE_WAYS:
            warnings.warn('shots and query num will not work in general mel setting')

        # read data. set index.
        kargs = {'orient': 'records', 'lines': True}
        self._doc = pandas.read_json(f'{root}/docs.json', **kargs).set_index('ID', drop=False)
        self._train = pandas.read_json(f'{root}/train.json', **kargs)
        self._test = pandas.read_json(f'{root}/test.json', **kargs)

        # divide data.
        train_num = int(len(self._train) * self.TRAIN_WAY_PORTION)
        self._valid = self._train[train_num:]
        self._train = self._train[:train_num]

    # pylint: disable=arguments-differ
    def _generate_way(self, doc: pandas.Series) -> Way:
        # using text and title as shots.
        return Way([doc['TEXT'], doc['TITLE']], doc['ID'])

    def _create_task(self, pd: pandas.DataFrame, retain=False) -> List[Task]:
        # given a mention record, create an task. set retain to True when eval and test.
        tasks = []
        progress = trange(0, len(pd))
        predicted_doc_ids = self.predictor.predict(pd['MENTION'].tolist(), self.WAYS_NUM_PRE_TASK)

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
        return self._create_task(self._train.sample(self.TRAIN_TASKS_NUM), False)

    def valid_data(self) -> List[Task]:
        return self._create_task(self._valid.sample(self.VALID_TASKS_NUM), True)

    def test_data(self) -> List[Task]:
        return self._create_task(self._test, True)
