'''
    there is no cross between train support set and eval support set.
    zeshel train/test_tasks.json file format:
        {ID:"", SHOTS:[{ MENTION: "", LCONTEXT:"", RCONTEXT:""}, ...]}
    zehsel test.json: LCONTEXT ~ MENTION ~ RCONTEXT ~ ID ~ CANDIDATE
'''
import random
from typing import List
import pandas as pd
from tqdm import trange
from . import ZeshelDataset
from .. import Way, Task

class NormalZeshelDataset(ZeshelDataset):
    '''
        zeshel datasets implementation.
        `root`: zeshel datasets file location.
    '''

    # train way portion
    TRAIN_WAY_PORTION = 0.9

    # whether support shots using context or not.
    SUPPORT_USING_CONTEXT = True

    # document DataFrame: ID - TEXT - TITLE
    _test_tasks: pd.DataFrame = None
    _train_tasks: pd.DataFrame = None
    _valid_tasks: pd.DataFrame = None
    _test: pd.DataFrame = None

    def __init__(self, root, conf=None):
        super().__init__(conf)

        # set function.
        self._support_x = self._x_context if self.SUPPORT_USING_CONTEXT else self._x_normal

        # filter
        kargs = {'orient': 'values', 'lines': True}
        func = lambda way: len(way['SHOTS']) > self.SHOTS_NUM_PRE_WAYS
        tasks = pd.read_json(f'{root}/train_tasks.json', **kargs)
        tasks = tasks.loc[tasks.apply(func, axis=1)]
        train_way_num = int(self.TRAIN_WAY_PORTION * len(tasks))
        self._train_tasks = tasks.iloc[:train_way_num]
        self._valid_tasks = tasks.iloc[train_way_num:]
        self._test_tasks = pd.read_json(f'{root}/test_tasks.json', **kargs)
        self._test_tasks = self._test_tasks.set_index('ID', drop=False)
        self._test = pd.read_json(f'{root}/test_{self.WAYS_NUM_PRE_TASK}_ways.json', **kargs)

        # check parameters.
        if len(self._train_tasks) < self.WAYS_NUM_PRE_TASK:
            raise AssertionError('no enough ways to generate an task')

    # pylint: disable=arguments-differ
    def _generate_way(self, way: pd.Series) -> Way:
        # random sample shots.
        shots = random.sample(way['SHOTS'], self.SHOTS_NUM_PRE_WAYS)
        return Way([self._support_x(shot) for shot in shots], way['ID'])

    def _sample_task(self, df: pd.DataFrame) -> Task:
        df = df.sample(self.WAYS_NUM_PRE_TASK)
        way = df.sample().iloc[0]
        # remove such shot, and remains are used to generate support.
        shots_backup = way['SHOTS']
        random.shuffle(shots_backup)
        way['SHOTS'] = shots_backup[:-1]
        query: dict = shots_backup[-1]
        query.update({'ID': way['ID']})
        # generate query and support and task.
        task = self._generate_task(query, df)
        way['SHOTS'] = shots_backup
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
                support = self._test_tasks.loc[query['CANDIDATE']]
                tasks.append(self._generate_task(query, support))
        # pad pad pad pad...
        return tasks
