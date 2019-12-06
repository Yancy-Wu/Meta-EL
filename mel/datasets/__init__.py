'''
    convert datasets varied of source to meta-learning tasks
    all UPPERCASE variable are super parameters.
'''
import sys
from typing import List, Any
from base.dataset import Dataset

class Example():
    ''' Example like this: x -> y '''

    # shot
    x: Any = None

    # label
    y: Any = 'maybe entity ID'

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Way():
    '''
        Way struct define in meta-learning task:
        [x0, x1, ..., xk] -> y
    '''

    # shots label
    y: Any = 'maybe entity ID'

    # all shots
    shots: List[Any] = None

    def __init__(self, shots, y):
        self.y = y
        self.shots = shots

class Task():
    '''
        Task struct define in meta-learning, a task like this:
            # |-------------------------- N WAYS -------------------------|
            # |-------- K SHOTs --------|   for support
            [([x00, x01, ..., x0k], y0), ..., ([xn0, xn1, ..., xnk], yn)]
            # |------- 1 example -------|   for query
            [(x0, y0)]
        support taskset and query taskset combine to an task
    '''

    # all ways, type should be class Way()[]
    support: List[Way] = None

    # Example(shots) for testing. type should be (class Shot(), y)
    query: Example = None

    def __init__(self, support, query):
        self.support = support
        self.query = query

class MelDataset(Dataset):
    '''
        dataset interface.
    '''

    # meta-learning train tasks num, default as much as possible
    TRAIN_TASKS_NUM = sys.maxsize

    # meta-learning validation tasks num, default as much as possible
    VALID_TASKS_NUM = sys.maxsize

    # meta-learning task ways num
    WAYS_NUM_PRE_TASK = 20

    # meta-learning task shots num
    SHOTS_NUM_PRE_WAYS = 2

    def train_data(self) -> List[Task]:
        '''
            generate tasks for training.
            `return` Task[], length is TRAIN_TASKS_NUM.
        '''
        raise NotImplementedError

    def valid_data(self) -> List[Task]:
        '''
            generate tasks for validating.
            `return` Task[], length is VALID_TASKS_NUM.
        '''
        raise NotImplementedError

    def test_data(self) -> List[Task]:
        '''
            get tasks for testing.
            `return` Task[], length is uncertain(but not much).
        '''
        raise NotImplementedError
