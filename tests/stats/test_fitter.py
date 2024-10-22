'''
Module containing unit tests for Fitter class
'''

import os
from dataclasses import dataclass
from functools   import cache

import ROOT
import zfit
import numpy
import pytest
import pandas              as pnd

from ROOT                  import RDF, RDataFrame, gInterpreter
from dmu.stats.fitter      import Fitter
from dmu.logging.log_store import LogStore

log = LogStore.add_logger('dmu:logging:test_fitter')
#-------------------------------------
@pytest.fixture
def _initialize():
    LogStore.set_level('dmu:logging:Fitter', 10)
    os.makedirs(Data.plt_dir, exist_ok=True)
#-------------------------------------
@dataclass
class Data:
    '''
    Data class used to store share data
    '''
    nsample = 50000
    plt_dir = 'tests/Fitter/plots'

    pdf     = None
    obs     = zfit.Space('m', limits=(-7.5, +7.5))
    arr     = numpy.random.normal(0, 1, size=100)
    df      = pnd.DataFrame({'x' : arr})
    zf      = zfit.data.from_numpy(obs=obs, array=arr)

    l_arg_simple = [arr, df, zf]
#-------------------------------------
def _get_data(path):
    if os.path.isfile(path):
        rdf=RDataFrame('tree', path)

        return rdf

    gInterpreter.ProcessLine('TRandom3 r(1);')

    d_val      = {}
    d_val['x'] = numpy.random.uniform(-1, 4, size=Data.nsample)
    d_val['y'] = numpy.random.uniform(-1, 4, size=Data.nsample)

    rdf = RDF.FromNumpy(d_val)
    rdf = rdf.Define('m', 'r.Gaus(x, 2 + y/4.)')
    rdf.Snapshot('tree', path)

    return rdf
#-------------------------------------
@cache
def _get_pdf():
    mu  = zfit.Parameter('mu', 1.0, -5, 5)
    sg  = zfit.Parameter('sg', 1.3,  0, 5)
    nev = zfit.Parameter('nev', 100,  0, 10_000_000)

    pdf = zfit.pdf.Gauss(obs=Data.obs, mu=mu, sigma=sg)
    pdf = pdf.create_extended(nev)

    return pdf
#-------------------------------------
def _make_dir_path(path : str) -> str:
    '''
    Takes path to directory, makes directory, returns path
    '''

    os.makedirs(path, exist_ok=True)

    return path
#-------------------------------------
def test_ntries():
    '''
    Test fitting with multiple tries
    '''
    pdf = _get_pdf()
    obj = Fitter(pdf, Data.arr)
    res = obj.fit(ntries=10, pval_threshold=0.99)

    assert res.valid
#-------------------------------------
@pytest.mark.parametrize('dat', Data.l_arg_simple)
def test_simple(dat):
    '''
    Simples fitting test
    '''
    pdf = _get_pdf()
    obj = Fitter(pdf, dat)
    res = obj.fit()

    assert res.valid
#-------------------------------------
def test_constrain():
    '''
    Fits with constraints to parameters
    '''
    pdf = _get_pdf()
    obj=Fitter(pdf, Data.arr)

    res=obj.fit()
    res.hesse()
    assert res.valid

    res=obj.fit(d_const={'mu' : (0, 0.1), 'sg' : (1, 0.01), 'nev' : (100, 0)})
    res.hesse()
    assert res.valid
#-------------------------------------
def test_ranges():
    '''
    Fit data in disjoint ranges
    '''
    obs   = zfit.Space('x', limits=(0, 10))
    lb    = zfit.Parameter("lb", -1,  -2, 0)
    model = zfit.pdf.Exponential(obs=obs, lam=lb)

    nev   = zfit.Parameter('nev', 100, 0, 100000)
    epdf  = model.create_extended(nev)

    data  = numpy.random.exponential(5, size=10000)
    data  = data[(data < 10)]
    data  = data[(data < 2) | ((data > 4) & (data < 6)) |  ((data > 8) & (data < 10)) ]

    obj   = Fitter(epdf, data)
    rng   = [(0,2), (4, 6), (8, 10)]
    res   = obj.fit(ranges=rng)

    assert res.valid
#-------------------------------------
def test_wgt():
    '''
    Test fit to weighted dataset
    '''
    dat_dir = _make_dir_path('tests/Fitter/splitter/')

    rdf = _get_data(f'{dat_dir}/Data.root')
    arr = rdf.AsNumpy(['m'])['m']
    wgt = numpy.random.binomial(1, 0.5, size=arr.size)

    pdf = _get_pdf()
    dat = zfit.data.from_numpy(array=arr, weights=wgt, obs=pdf.space)

    obj=Fitter(pdf, dat)
    res=obj.fit()
    res.hesse(method='minuit_hesse')

    assert res.valid
