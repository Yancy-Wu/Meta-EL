'''
    cross-lingual json file format:
    en-kb(ID ||| NAME ||| TYPE):
        53424859 ||| Henriette Aymer de Chevalerie ||| PER
    en-am(ID ||| EN_NAME ||| AM_NAME ||| TYPE):
        3378263 ||| John Michael Talbot ||| ጆን ማይክል ታልበት ||| PER
'''
import random
from typing import List
from tqdm import trange
import pandas
from . import MelDataset, Way, Task, Example

class SingleXel(MelDataset):
    '''
        [NOTE]: we do not differentiate WAYS and CLASSES.
        xel single datasets implementation.
        will train on a selected language and also test on that.
        `root`: xel datasets dir location. should exist [en-kb] file
        `lauguage`: training language.
    '''

    # portion of examples for generating training tasks
    TRAIN_EXAMPLES_PORTION = 0.9

    # xel one-shot setting.
    SHOTS_NUM_PRE_WAYS = 1

    # it will be more difficult
    QUERY_NUM_PRE_WAY: float = 0.5

    # kg DataFrame: ID - EN_NAME - TYPE
    # train DataFrame and valid DataFrame: ID - EN_NAME - LRL_NAME - TYPE
    _kg: pandas.DataFrame = None
    _train: pandas.DataFrame = None
    _valid: pandas.DataFrame = None

    def __init__(self, root, language, conf=None):
        super().__init__(conf)
        # check parameters.
        assert self.SHOTS_NUM_PRE_WAYS == 1, 'XEL only support one shot!'
        assert self.QUERY_NUM_PRE_WAY < 1, 'your asshole!'
        # read data.
        all_examples: pandas.DataFrame = pandas.read_csv(
            f'{root}/links/grapheme/en-{language}_links', engine='python',
            sep='\\|{3}', names=['ID', 'EN_NAME', 'LRL_NAME', 'TYPE'])
        self._kg: pandas.DataFrame = pandas.read_csv(
            f'{root}/kb/en_kb', engine='python',
            sep=r'\|{3}', names=['ID', 'EN_NAME', 'TYPE'])
        # divide data.
        train_examples_num = int(len(all_examples) * self.TRAIN_EXAMPLES_PORTION)
        self._train = all_examples[:train_examples_num]
        self._valid = all_examples[train_examples_num:]

    def _sample_task(self, df: pandas.DataFrame) -> Task:
        '''
            generate an meta-learning task from a DataFrame
            random sample query examples from `df`.
            then send LRL text as query, en text as support.
            at last. padding support set by random sampling from _kg.
            `return`: a Task object.
        '''
        task = Task([], [])

        # generate candidate
        query_num = int(self.WAYS_NUM_PRE_TASK * self.QUERY_NUM_PRE_WAY)
        query_shots = df.sample(query_num)
        candidate_support_ways = self._kg.sample(self.WAYS_NUM_PRE_TASK)

        # remove query shots which have existed in candidate_support_ways
        exists_index = candidate_support_ways['ID'].isin(query_shots['ID'])
        candidate_support_ways = candidate_support_ways[~exists_index]
        support_ways = candidate_support_ways.sample(self.WAYS_NUM_PRE_TASK - query_num)

        # generate query shots and corresponding support ways
        for _, _shot in query_shots.iterrows():
            task.query.append(Example(_shot['LRL_NAME'], _shot['ID']))
            task.support.append(Way([_shot['EN_NAME']], _shot['ID']))

        # pad support ways
        for _, _shot in support_ways.iterrows():
            task.support.append(Way([_shot['EN_NAME']], _shot['ID']))

        # shuffle
        random.shuffle(task.query)
        random.shuffle(task.support)
        return task

    # sample an task from train DataFrame
    def train_data(self) -> List[Task]:
        progress = trange(0, len(self.TRAIN_TASKS_NUM))
        tasks = [self._sample_task(self._train) for _ in progress]
        progress.close()
        return tasks

    # sample an task from valid DataFrame
    def valid_data(self) -> List[Task]:
        progress = trange(0, len(self.VALID_TASKS_NUM))
        tasks = [self._sample_task(self._valid) for _ in progress]
        progress.close()
        return tasks

    def test_data(self) -> List[Task]:
        pass
