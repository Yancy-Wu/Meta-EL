'''
    zeshel  json file format:
    {
        entity_id:"",
        items:[{ mention: "", category:"", context:""}, ...]
    }
'''
import random
import json
import logging
from typing import List
from . import Datasets, Way, Task, Example, Shot

LOGGER = logging.getLogger(__name__)

class Zeshel(Datasets):
    '''
        [NOTE]: we do not differentiate WAYS and CLASSES.
        zeshel datasets implementation.
        `root`: zeshel datasets file location which contain train.json, test.json.
    '''

    # portion of ways for generating training tasks
    TRAIN_WAYS_PORTION = 0.9

    # train ways, valid ways and test tasks
    _train_ways: List[Way] = []
    _valid_ways: List[Way] = []
    _test_tasks: List[Task] = []

    # sample an task from train ways
    def train_tasks(self) -> List[Task]:
        for _ in range(0, self.TRAIN_TASKS_NUM):
            yield self._sample_task(self._train_ways)

    # sample an task from valid ways
    def valid_tasks(self) -> List[Task]:
        for _ in range(0, self.VALID_TASKS_NUM):
            yield self._sample_task(self._train_ways)

    def test_tasks(self) -> List[Task]:
        return self._test_tasks

    def __init__(self, root, conf=None):
        super().__init__(conf)
        all_ways = self._generate_ways_from_file(f'{root}/train.json')
        train_ways_num = len(all_ways) * self.TRAIN_WAYS_PORTION
        self._train_ways = all_ways[:train_ways_num]
        self._valid_ways = all_ways[train_ways_num:]

        # if available shots are too few, log a warning
        train_shots_num = sum([len(way.shots) for way in self._train_ways])
        if train_shots_num < self.WAYS_NUM_PRE_TASK * self.TRAIN_TASKS_NUM:
            LOGGER.warning(f'{train_shots_num} examples are too few for generating tasks')

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
            if len(_shots) < self.SHOTS_NUM_PRE_WAYS:
                continue

            # generate new way
            avail_ways.append(Way({
                'y': _y,
                'shots': [Shot({'x': _shot['mention'], 'ext': _shot}) for _shot in _shots]
            }))

        return avail_ways

    def _sample_task(self, ways: List[Way]) -> Task:
        '''
            generate an meta-learning task from a list of ways
            `return`: a Task object.
        '''
        task = Task()

        # generate ways and tests. [pylint say error, i dont know why]
        # pylint: disable=unsubscriptable-object
        way: Way = None
        for way in random.sample(ways, self.WAYS_NUM_PRE_TASK):
            random.shuffle(way.shots)
            train_shots = way.shots[:self.SHOTS_NUM_PRE_WAYS]
            test_shots = way.shots[self.SHOTS_NUM_PRE_WAYS:]
            task.ways.append(Way({'shots': train_shots, 'y': way.y}))
            task.test += [Example({'shot': shot, 'y' : way.y}) for shot in test_shots]

        # got it
        return task
