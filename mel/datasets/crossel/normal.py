'''
    support.json: ID - SHOTS{LAN - MENTION}
    test_{lan}.json: ID - LAN - MENTION - CANDIDATE_50_WAYS - ... - CANDIDATE_2_WAYS
'''
import random
from typing import List
from tqdm import trange
import pandas as pd
from .. import Way, Task
from . import CrosselDataset

class NormalCrosselDataset(CrosselDataset):
    '''
        `root`: xel datasets dir location. should exist [en-kb] file
        `lauguage`: training language.
    '''

    # train way portion
    TRAIN_WAY_PORTION = 0.9

    # ID - SHOTS{LAN - TITLE}
    _support: pd.DataFrame = None
    _train_tasks: pd.DataFrame = None
    _valid_tasks: pd.DataFrame = None
    _test: pd.DataFrame = None

    def __init__(self, root, conf=None):
        super().__init__(conf)

        # check parameters.
        assert self.SHOTS_NUM_PRE_WAYS < 10, 'No so much shots!'

        # read data.
        kargs = {'orient': 'values', 'lines': True}
        support = pd.read_json(f'{root}/support.json', **kargs).set_index('ID', drop=False)
        train_way_num = int(self.TRAIN_WAY_PORTION * len(support))
        self._support = support
        self._train_tasks = support[:train_way_num]
        self._valid_tasks = support[train_way_num:]
        self._test = pd.read_json(f'{root}/test_{self.TEST_LANGUAGE}.json', **kargs)

    # pylint: disable=arguments-differ
    def _generate_way(self, way: pd.Series) -> Way:
        # random sample shots.
        shots = random.sample(way['SHOTS'], self.SHOTS_NUM_PRE_WAYS)
        return Way([self._x(shot) for shot in shots], way['ID'])

    def _sample_task(self, df: pd.DataFrame) -> Task:
        df = df.sample(self.WAYS_NUM_PRE_TASK)
        way = df.sample().iloc[0]
        # remove such shot, and remains are used to generate support.
        shots = way['SHOTS']
        random.shuffle(shots)
        df.at[way.name, 'SHOTS'] = shots[:-1]
        query: dict = shots[-1]
        query.update({'ID': way['ID']})
        # generate query and support and task.
        task = self._generate_task(query, df)
        return task

    # sample an task from train ways
    def train_data(self) -> List[Task]:
        # sample train task.
        with trange(0, self.TRAIN_TASKS_NUM) as progress:
            return [self._sample_task(self._train_tasks) for _ in progress]

    def valid_data(self) -> List[Task]:
        # sample valid task.
        with trange(0, self.VALID_TASKS_NUM) as progress:
            return [self._sample_task(self._valid_tasks) for _ in progress]

    def test_data(self) -> List[Task]:
        tasks = []
        # generate task per line
        with trange(0, len(self._test)) as progress:
            for (_, query), _ in zip(self._test.iterrows(), progress):
                key = f'CANDIDATE_{self.WAYS_NUM_PRE_TASK}_WAYS'
                support = self._support.loc[query[key]]
                tasks.append(self._generate_task(query, support))
        # pad pad pad pad...
        return tasks
