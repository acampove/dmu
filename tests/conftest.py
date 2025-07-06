'''
File needed by pytest
'''
import logging

import numpy
import mplhep
import matplotlib.pyplot as plt

from _pytest.config import Config
# Needed to make sure zfit gets imported properly
# before any test runs
from dmu.stats.zfit        import zfit
from dmu.workflow.cache    import Cache as Wcache
from dmu.logging.log_store import LogStore

# ------------------------------
def pytest_configure(config : Config):
    '''
    This will run before any test by pytest
    '''
    _config = config
    numpy.random.seed(42)

    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('git.cmd').setLevel(logging.WARNING)

    LogStore.set_level('dmu:stats:kde_optimizer', 10)

    plt.style.use(mplhep.style.LHCb2)
    Wcache.set_cache_root(root='/tmp/tests/dmu/cache/cache_dir')
# ------------------------------
