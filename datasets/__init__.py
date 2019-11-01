'''
    convert datasets varied of source to meta-learning tasks
    all UPPERCASE variable are super parameters.
'''
from typing import List
from ..base import config

class Shot(config.Config):
    ''' Shot struct define in meta-learning task '''

    # shot input
    x = 'a mention or else'

    # external info for a shot
    ext = dict()

class Example(config.Config):
    ''' Example like this: (x, ext_info) -> y '''

    # shot
    shot: Shot = None

    # label
    y = 'maybe entity ID'

class Way(config.Config):
    '''
        Way struct define in meta-learning task:
        [(x0, ext_info0), ..., (xk, ext_infok)] -> y
    '''

    # shots label
    y = 'maybe entity ID'

    # all shots, type should be class Shot()
    shots: List[Shot] = []

class Task(config.Config):
    '''
        Task struct define in meta-learning, a task like this:
            # |-------------------------- N WAYS -------------------------|
            # |-------- K SHOTs --------|   for training
            [([x00, x01, ..., x0k], y0), ..., ([xn0, xn1, ..., xnk], yn)]
            # |------- UNCERTAIN -------|   for testing
            [(x0, y0), ..., (x?, y?)]
        train taskset and test taskset combine to an task
    '''

    # all ways, type should be class Way()[]
    ways: List[Way] = []

    # Example(shots) for testing. type should be (class Shot(), y)
    test: List[Example] = []


class Datasets(config.Config):
    '''
        dataset interface.
    '''

    # meta-learning train tasks num
    TRAIN_TASKS_NUM = 150

    # meta-learning validation tasks num
    VALID_TASKS_NUM = 10

    # meta-learning task ways num
    WAYS_NUM_PRE_TASK = 20

    # meta-learning task shots num
    SHOTS_NUM_PRE_WAYS = 3

    def __init__(self, conf=None):
        super().__init__(conf)

    def train_tasks(self) -> List[Task]:
        '''
            generate tasks for training.
            `return` Task[], length is TRAIN_TASKS_NUM.
        '''
        raise NotImplementedError

    def valid_tasks(self) -> List[Task]:
        '''
            generate tasks for validating.
            `return` Task[], length is VALID_TASKS_NUM.
        '''
        raise NotImplementedError

    def test_tasks(self) -> List[Task]:
        '''
            get tasks for testing.
            `return` Task[], length is uncertain(but not much).
        '''
        raise NotImplementedError
