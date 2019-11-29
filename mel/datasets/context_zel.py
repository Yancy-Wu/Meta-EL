'''
    context_zel csv file format: [~ as sep]
    documents.json: ID ~ TEXT ~ TITLE
    mentions.json: LCONTEXT ~ MENTION ~ RCONTEXT ~ ID
    output mention and context string:
    '{left_context}[SEP]{mention}[SEP]{right_context}'
'''
import random
from typing import List
from tqdm import trange
import pandas
from . import MelDataset, Way, Task, Example

class ContextZel(MelDataset):
    '''
        [DESCRIPTION]
          step 1: random sample mention examples from 'mentions.csv' as query set
          step 2: fetch corresponding label_document text and title
                and random sample negative document as support set
        [PARAMS]
          `root`: context zel datasets dir location.
    '''

    # portion of examples for generating training tasks
    TRAIN_EXAMPLES_PORTION = 0.9

    # context_zel two-shot setting.
    SHOTS_NUM_PRE_WAYS = 2

    # it will be more difficult
    QUERY_NUM_PRE_WAY: float = 0.5

    # document DataFrame: ID - TEXT - TITLE
    # train DataFrame and valid DataFrame: ID - LEFT_CONTEXT - MENTION - RIGHT_CONTEXT
    _doc: pandas.DataFrame = None
    _train: pandas.DataFrame = None
    _valid: pandas.DataFrame = None

    def __init__(self, root, conf=None):
        super().__init__(conf)
        # check parameters.
        assert self.SHOTS_NUM_PRE_WAYS == 2, 'context XEL only support two shot!'
        assert self.QUERY_NUM_PRE_WAY < 1, 'your asshole!'
        # read data.
        kargs = {'orient': 'records', 'lines': True}
        self._doc: pandas.DataFrame = pandas.read_json(f'{root}/documents.json', **kargs)
        all_examples: pandas.DataFrame = pandas.read_json(f'{root}/mentions.json', **kargs)
        # divide data.
        train_examples_num = int(len(all_examples) * self.TRAIN_EXAMPLES_PORTION)
        self._train = all_examples[:train_examples_num]
        self._valid = all_examples[train_examples_num:]

    def _sample_task(self, df: pandas.DataFrame) -> Task:
        '''
            generate an meta-learning task from a DataFrame
            `return`: a Task object.
        '''
        task = Task([], [])

        # sample query examples.
        query_num = int(self.WAYS_NUM_PRE_TASK * self.QUERY_NUM_PRE_WAY)
        query_shots = df.sample(query_num)

        # generate and sample support ways
        query_ids = query_shots['ID']
        related_doc_index = self._doc['ID'].isin(query_ids)
        noise_doc = self._doc[~related_doc_index].sample(self.WAYS_NUM_PRE_TASK - len(query_ids))
        support_ways = pandas.concat([noise_doc, self._doc[related_doc_index]])

        # generate query shots
        for _, _shot in query_shots.iterrows():
            x = _shot['LCONTEXT'] + '[SEP]' + _shot['MENTION'] + '[SEP]' + _shot['RCONTEXT']
            task.query.append(Example(x, _shot['ID']))

        # generate support ways
        for _, _shot in support_ways.iterrows():
            task.support.append(Way([_shot['TITLE'], _shot['TEXT']], _shot['ID']))

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
