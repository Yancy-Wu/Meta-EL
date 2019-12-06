'''
    utils which cannot be grouped to certain module.
'''

import functools
from typing import List, Tuple, Any

def dict_to_kvlist(d: dict, sep='|') -> Tuple[List[str], List[Any]]:
    '''
        save dict keys and val to list deeply.
        using dfs, return key list and val list pair.
        `sep`: for deep level key split, such as:
          result->code will be 'result|code'
    '''
    keys = []
    vals = []
    def _travel(prefix: str, v):
        # call sub travel
        if isinstance(v, dict):
            for key, val in v.items():
                new_prefix = prefix + sep + key
                _travel(new_prefix, val)
        # add to list
        else:
            keys.append(prefix[1:])
            vals.append(v)

    _travel('', d)
    return keys, vals

def kvlist_to_dict(keys: List[str], vals: List[Any], sep='|') -> dict:
    '''
        from k,v lists recover dict.
        result|code will be result->code
    '''
    res = dict()

    # recover a key
    def _recover(key: List[str], val, d: dict):
        if not key:
            raise AssertionError('key error')
        if len(key) == 1:
            d.update({key[0]: val})
        else:
            next_d = d.setdefault(key[0], dict())
            _recover(key[1:], val, next_d)

    # recover all keys to re-build dict.
    for key, val in zip(keys, vals):
        _recover(key.split(sep), val, res)

    return res

def deep_apply_dict(d: dict(), func: type(lambda k, v: None)):
    '''
        travel dict deeply, apply func to all value.
    '''
    for key, val in d.items():
        if isinstance(val, dict):
            deep_apply_dict(val, func)
        else:
            d[key] = func(key, val)

def log(cls, text):
    '''
        auto log class name and text.
    '''
    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kw):
            print(f'[{cls.__name__}]: {text}')
            return func(*args, **kw)
        return _wrapper
    return _decorator
