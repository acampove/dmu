'''
Module with tests for GofCalculator class
'''
import math
from dataclasses              import dataclass

import zfit
import numpy
import pytest

from zfit                     import Data       as zdata
from dmu.stats.gof_calculator import GofCalculator
from dmu.logging.log_store    import LogStore

log = LogStore.add_logger('dmu:stats:test_gofcalculator')
#---------------------------------------------
@dataclass
class Data:
    '''
    Class used to share attributes
    '''
    minimizer = zfit.minimize.Minuit()
    obs       = zfit.Space('x', limits=(-10, 10))
    obs_bin   = zfit.Space('x', limits=(-10, 10), binning=50)
#---------------------------------------------
@pytest.fixture(scope='session', autouse=True)
def _initialize():
    LogStore.set_level('dmu:stats:gofcalculator', 10)
    numpy.random.seed(42)
#---------------------------------------------
def _get_model():
    mu  = zfit.Parameter('mu', 3, -1, 5)
    sg  = zfit.Parameter('sg', 2,  0, 5)
    pdf = zfit.pdf.Gauss(obs=Data.obs, mu=mu, sigma=sg)

    return pdf
# -------------------------------------------
def _get_data() -> zdata:
    data_np = numpy.random.normal(0, 1, size=10000)
    data_zf = zfit.Data.from_numpy(obs=Data.obs, array=data_np)

    return data_zf
# -------------------------------------------
def _get_nll(binned : bool):
    pdf = _get_model()
    dat = _get_data()

    if binned:
        dat = dat.to_binned(space=Data.obs_bin)
        pdf = pdf.to_binned(space=Data.obs_bin)
        nll = zfit.loss.BinnedNLL(model=pdf, data=dat)
    else:
        nll = zfit.loss.UnbinnedNLL(model=pdf, data=dat)

    return nll
# -------------------------------------------
def test_unbinned():
    '''
    Test GofCalculator with unbinned data
    '''
    nll = _get_nll(binned=False)
    res = Data.minimizer.minimize(nll)
    print(res)

    gcl = GofCalculator(nll, ndof=10)
    gof = gcl.get_gof(kind='pvalue')

    assert math.isclose(gof, 0.965, abs_tol=0.01)
# -------------------------------------------
def test_binned():
    '''
    Test GofCalculator with binned data
    '''
    nll = _get_nll(binned=True)
    res = Data.minimizer.minimize(nll)
    print(res)

    gcl = GofCalculator(nll, ndof=10)
    gof = gcl.get_gof(kind='pvalue')

    assert math.isclose(gof, 0.965, abs_tol=0.01)
# -------------------------------------------
