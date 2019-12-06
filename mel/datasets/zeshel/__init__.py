'''
    zeshel dataset base class.
'''
import random
import warnings
from typing import List
from tqdm import trange
import pandas
from utils import log
from .. import MelDataset, Way, Task, Example

class ZeshelDataset(MelDataset):
    '''
        `root`: ZeshelDataset dir location.
    '''

    # whether query using context or not.
    QUERY_USING_CONTEXT = True

    def __init__(self, conf=None):
        super().__init__(conf)

        # set function
        self._x = self._x_context if self.QUERY_USING_CONTEXT else self._x_normal

    @staticmethod
    def _x_normal(xs: pandas.Series) -> Example:
        # convert pandas series to finally x, no context
        return xs['MENTION']

    @staticmethod
    def _x_context(xs: pandas.Series) -> Example:
        # convert pandas series to finally x, with context
        return xs['LCONTEXT'] + ' [SEP] ' + xs['MENTION'] + ' [SEP] ' + xs['RCONTEXT']

    def _generate_query(self, query: pandas.Series) -> Example:
        # convert pandas series to query.
        return Example(self._x(query), query['ID'])

    def _generate_way(self, way: pandas.Series) -> Way:
        # convert pandas series finally support way
        raise NotImplementedError

    def _generate_task(self, query: pandas.Series, support: pandas.DataFrame) -> Task:
        # conver query and support pandas dataframe to finally task
        support = [self._generate_way(way) for _, way in support.iterrows()]
        return Task(support, self._generate_query(query))
