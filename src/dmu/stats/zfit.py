'''
Module intended to wrap zfit

Needed in order to silence tensorflow messages
'''

import dmu.generic.utilities as gut

with gut.silent_import():
    import tensorflow

import zfit
