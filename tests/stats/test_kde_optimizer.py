'''
This module implements tests for the KDEOptimizer class
'''
import os

import pytest
import pandas            as pnd
import matplotlib.pyplot as plt

from dmu.stats.zfit           import zfit
from dmu.generic              import utilities  as gut
from dmu.stats.kde_optimizer  import KDEOptimizer
from dmu.logging.log_store    import LogStore
from dmu.stats.zfit_plotter   import ZFitPlotter

from zfit.core.interfaces     import ZfitSpace  as zobs
from zfit.core.interfaces     import ZfitData   as zdata
from zfit.core.interfaces     import ZfitPDF    as zpdf

log=LogStore.add_logger('dmu:stats:test_kde_optimizer')
# ---------------------------------------------------------
class Data:
    '''
    Used to share attributes
    '''
    out_dir = '/tmp/tests/dmu/stats/kde_optimizer'

    os.makedirs(out_dir, exist_ok=True)
# ---------------------------------------------------------
def _plot_fit(pdf : zpdf, data : zdata, name : str) -> None:
    obj= ZFitPlotter(data=data, model=pdf)
    obj.plot()

    plt.savefig(f'{Data.out_dir}/{name}.png')
    plt.close()
# ---------------------------------------------------------
def _get_data(kind : str, obs : zobs) -> zdata:
    data = gut.load_data(package='dmu_data', fpath=f'stats/kde_optimizer/{kind}.json')
    df   = pnd.DataFrame(data)

    arr_mass = df['val'].to_numpy()
    arr_wgt  = df['wgt'].to_numpy()

    dat  = zfit.data.from_numpy(obs=obs, array=arr_mass, weights=arr_wgt)

    return dat
# ---------------------------------------------------------
@pytest.mark.parametrize('kind', ['signal', 'control'])
def test_simple(kind : str):
    '''
    Simplest test
    '''
    log.info('')

    obs  = zfit.Space('mass', limits=(4500,7000))
    dat  = _get_data(kind=kind, obs=obs)
    opt  = KDEOptimizer(data=dat, obs=obs)
    pdf  = opt.get_pdf()

    _plot_fit(pdf=pdf, data=dat, name=f'simple_{kind}')
