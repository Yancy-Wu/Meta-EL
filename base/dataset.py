'''
    base dataset class, for zel and mel.
'''

from typing import List
from base import config

class Dataset(config.Config):
    '''
        dataset interface.
    '''

    def train_data(self) -> List[object]:
        '''
            generate train obj data for training.
            `return` object list, feed to adapter for futher tensorlizing.
        '''
        raise NotImplementedError

    def valid_data(self) -> List[object]:
        '''
            generate tasks for validating.
            `return` object list, feed to adapter for futher tensorlizing.
        '''
        raise NotImplementedError

    def test_data(self) -> List[object]:
        '''
            get tasks for testing.
            `return` object list, feed to adapter for futher tensorlizing.
        '''
        raise NotImplementedError
