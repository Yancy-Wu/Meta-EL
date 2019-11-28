'''
    convert datasets varied of source to classify whether two examples are identical
    all UPPERCASE variable are super parameters.
'''
from typing import List, Any, Tuple
from base.config import Config

class Example():
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

class Datasets(Config):
    '''
        dataset interface.
    '''

    def train_examples(self) -> List[Example]:
        '''
            generate examples for training.
            `return` Examples[], length is as much as possible.
        '''
        raise NotImplementedError

    def valid_examples(self) -> List[Example]:
        '''
            generate tasks for validating.
            `return` Examples[], length is as much as possible.
        '''
        raise NotImplementedError

    def test_examples(self) -> List[Example]:
        '''
            get tasks for testing.
            `return` Examples[], length is as much as possible.
        '''
        raise NotImplementedError

    def all_candidates(self) -> Tuple[List[str], List[Any]]:
        '''
            get all candidate part of example, or called all candidates.
            `return`: candidate id list and candidate value list.
        '''
        raise NotImplementedError
