'''
    kb.json: ID ~ TEXT ~ TITLE
    train.json: LAN ~ MENTION ~ ID
    test.json: LAN ~ MENTION ~ ID ~ CANDIDATE
'''
import warnings
from typing import List
from tqdm import trange
import pandas as pd
from .. import Way, Task
from . import CrosselDataset

class OverallCrosselDataset(CrosselDataset):
    '''
        `root`: OverallCrosselDataset dir location.
    '''

    # train way portion
    TRAIN_WAY_PORTION = 0.9

    # general mel must be 2 shots
    SHOTS_NUM_PRE_WAYS = None

    # document DataFrame: ID - TEXT - TITLE
    _kg: pd.DataFrame = None
    _train: pd.DataFrame = None
    _valid: pd.DataFrame = None
    _test: pd.DataFrame = None

    def __init__(self, root, conf=None):
        super().__init__(conf)

        # check parameters.
        if self.SHOTS_NUM_PRE_WAYS:
            warnings.warn('shots num will not work in overall crossel mel setting')

        # read data. set index.
        kargs = {'orient': 'records', 'lines': True}
        examples = pd.read_json(f'{root}/train.json', **kargs)
        train_example_num = int(self.TRAIN_WAY_PORTION * len(examples))
        self._kg = pd.read_json(f'{root}/kb.json', **kargs).set_index('ID', drop=False)
        self._train = examples[:train_example_num]
        self._valid = examples[train_example_num:]
        self._test = pd.read_json(f'{root}/test_{self.TEST_LANGUAGE}.json', **kargs)

    # pylint: disable=arguments-differ
    def _generate_way(self, doc: pd.Series) -> Way:
        # using text and title as shots.
        return Way([doc['TEXT'], doc['TITLE']], doc['ID'])

    def _sample_task(self, df: pd.DataFrame) -> Task:
        # sample a meta-learning task from a DataFrame
        query = df.sample().iloc[0]
        noise_doc = self._kg.drop(index=query['ID']).sample(self.WAYS_NUM_PRE_TASK - 1)
        support = noise_doc.append(self._kg.loc[query['ID']])
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
                key = f'CANDIDATE_{self.WAYS_NUM_PRE_TASK}_WAYS'
                support = self._kg.loc[query[key]]
                tasks.append(self._generate_task(query, support))
        # pad pad pad pad...
        return tasks
