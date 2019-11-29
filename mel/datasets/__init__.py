'''
    convert datasets varied of source to meta-learning tasks
    all UPPERCASE variable are super parameters.
'''
from typing import List, Any
from base.dataset import Dataset

class Example():
    ''' Example like this: x -> y '''

    # shot
    x: Any = None

    # label
    y: Any = 'maybe entity ID'

    def __init__(self, query, y):
        self.query = query
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
            # |-------- K SHOTs --------|   for training
            [([x00, x01, ..., x0k], y0), ..., ([xn0, xn1, ..., xnk], yn)]
            # |------- M examples -------|   for testing
            [(x0, y0), ..., (x?, y?)]
        support taskset and query taskset combine to an task
    '''

    # all ways, type should be class Way()[]
    support: List[Way] = None

    # Example(shots) for testing. type should be (class Shot(), y)
    query: List[Example] = None

    def __init__(self, support, query):
        self.support = support
        self.query = query

class MelDataset(Dataset):
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
    SHOTS_NUM_PRE_WAYS = 2

    # meta-learning query examples num per way
    # if less than 1, there will be some ways no query example.
    QUERY_NUM_PRE_WAY: float = 1

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
