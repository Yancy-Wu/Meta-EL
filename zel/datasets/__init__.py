'''
    convert datasets varied of source to classify whether two examples are identical
    all UPPERCASE variable are super parameters.
'''
from typing import List, Any, Tuple
from base.dataset import Dataset

class TrainExample():
    ''' Example like this: (query, candidate) -> y '''

    # two part to compare
    query: Any = None
    candidate: Any = None

    # label, for two classification, be 0 or 1.
    y: int = 0

    def __init__(self, query, candidate, y):
        self.query = query
        self.candidate = candidate
        self.y = y

class TestExample():
    ''' Example like This: (x -> y) '''

    # example x
    x: Any = None

    # example label
    y: Any = None

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Candidate(TestExample):
    ''' now same as test example '''
    pass

class ZelDataset(Dataset):
    '''
        zel dataset interface.
    '''

    def train_data(self) -> List[TrainExample]:
        '''
            generate examples for training.
            `return` TrainExample[], length is as much as possible.
        '''
        raise NotImplementedError

    def valid_data(self) -> List[TrainExample]:
        '''
            generate tasks for validating.
            `return` TrainExample[], length is as much as possible.
        '''
        raise NotImplementedError

    def test_data(self) -> List[TestExample]:
        '''
            get tasks for testing.
            `return` TestExample[], length is as much as possible.
        '''
        raise NotImplementedError

    def all_candidates(self) -> List[Candidate]:
        '''
            get all candidate part of example, or called all entities.
            `return`: candidate obj list.
        '''
        raise NotImplementedError
