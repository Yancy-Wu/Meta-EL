'''
    provide config feature.
'''


class Config:
    '''
        base class.
    '''

    def __init__(self, config=None):
        ''' add attr(s) to current object, override if exists '''
        if not config:
            return
        if isinstance(config, dict):
            for key, val in config.items():
                setattr(self, key, val)
        else:
            raise AssertionError('config type error')
