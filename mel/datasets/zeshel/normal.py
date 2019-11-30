'''
    [SETTTING]
      using context or no context.
      multi-shots.
      there is no cross between train support set and eval support set.
    [FORMAT]
      zeshel json file format:
      {
          id:"",
          shots:[{ mention: "", left_context:"", right_context:""}, ...]
      }
'''
import random
import json
import logging
from typing import List
from tqdm import trange
from .. import MelDataset, Way, Task, Example

LOGGER = logging.getLogger(__name__)

class NormalZeshelDataset(MelDataset):
    '''
        [NOTE]: we do not differentiate WAYS and CLASSES.
        zeshel datasets implementation.
        `root`: zeshel datasets file location which contain train.json, test.json.
    '''

    # portion of ways for generating training tasks
    TRAIN_WAYS_PORTION = 0.9

    # train ways, valid ways and test tasks
    _train_ways: List[Way] = None
    _valid_ways: List[Way] = None
    _test_tasks: List[Task] = None

    def __init__(self, root, conf=None):
        super().__init__(conf)

        # not support fraction query num.
        self.QUERY_NUM_PRE_WAY = int(self.QUERY_NUM_PRE_WAY)
        if self.QUERY_NUM_PRE_WAY < 1:
            raise AssertionError('not support fraction query num per way')

        all_ways = self._generate_ways_from_file(f'{root}/train.json')
        train_ways_num = int(len(all_ways) * self.TRAIN_WAYS_PORTION)
        self._train_ways = all_ways[:train_ways_num]
        self._valid_ways = all_ways[train_ways_num:]

        # if no enough ways, raise error.
        if len(self._train_ways) < self.WAYS_NUM_PRE_TASK:
            raise AssertionError('no enough ways to generate an task')

        # if available shots are too few, log a warning
        train_shots_num = sum([len(way.shots) for way in self._train_ways])
        if train_shots_num < self.WAYS_NUM_PRE_TASK * self.TRAIN_TASKS_NUM:
            LOGGER.warning(f'{train_shots_num} examples maybe too few for generating tasks')

    def _generate_ways_from_file(self, fn: str) -> List[Way]:
        '''
            load all ways and shots from file.
            `return`: list of all ways(shots in it).
        '''
        avail_ways: List[Way] = []

        for line in open(fn):
            _way = json.loads(line)
            _y = _way['entity_id']
            _shots = _way['items']

            # filter all ways whose shots count less than SHOTS_NUM
            if len(_shots) < self.SHOTS_NUM_PRE_WAYS + self.QUERY_NUM_PRE_WAY:
                continue

            # generate new way
            avail_ways.append(Way([_shot['mention'] for _shot in _shots], _y))

        return avail_ways

    def _sample_task(self, ways: List[Way]) -> Task:
        '''
            generate an meta-learning task from a list of ways
            `return`: a Task object.
        '''
        task = Task([], [])

        # generate ways and tests. [pylint say error, i dont know why]
        # pylint: disable=unsubscriptable-object
        way: Way = None
        for way in random.sample(ways, self.WAYS_NUM_PRE_TASK):
            shots = random.sample(way.shots, self.SHOTS_NUM_PRE_WAYS + self.QUERY_NUM_PRE_WAY)
            train_shots = shots[:self.SHOTS_NUM_PRE_WAYS]
            test_shots = shots[self.SHOTS_NUM_PRE_WAYS:]
            task.support.append(Way(train_shots, way.y))
            task.query += [Example(shot, way.y) for shot in test_shots]
        # got it
        return task

    # sample an task from train ways
    def train_data(self) -> List[Task]:
        progress = trange(0, self.TRAIN_TASKS_NUM)
        tasks = [self._sample_task(self._train_ways) for _ in progress]
        progress.close()
        return tasks

    # sample an task from valid ways
    def valid_data(self) -> List[Task]:
        progress = trange(0, self.VALID_TASKS_NUM)
        tasks = [self._sample_task(self._valid_ways) for _ in progress]
        progress.close()
        return tasks

    def test_data(self) -> List[Task]:
        return self._test_tasks
