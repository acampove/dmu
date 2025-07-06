'''
Module holding KDEOptimizer class
'''

from scipy.linalg import bandwidth
from scipy.optimize           import minimize_scalar
from dmu.stats.zfit           import zfit
from dmu.stats.gof_calculator import GofCalculator
from dmu.logging.log_store    import LogStore

from zfit.core.interfaces     import ZfitData   as zdata
from zfit.core.interfaces     import ZfitPDF    as zpdf
from zfit.core.interfaces     import ZfitSpace  as zobs

log=LogStore.add_logger('dmu:stats:kde_optimizer')
# -----------------------------------------------------
class KDEOptimizer:
    '''
    Class meant to wrap KDE1DimFFT in order to pick bandwidth
    that optimizes the goodness of fit
    '''
    # --------------------------
    def __init__(self, data : zdata, obs : zobs):
        '''
        data: zfit data
        obs : observable
        '''
        self._data     = data
        self._obs      = obs
        self._last_pdf : zpdf|None = None
        self._niter    = 0
    # --------------------------
    def _calculate_fom(self, bandwidth : float) -> float:
        pdf = zfit.pdf.KDE1DimFFT(
                obs      = self._obs,
                data     = self._data,
                bandwidth= bandwidth,
                padding  = {'lowermirror' : 0.2, 'uppermirror' : 0.2})

        nll = zfit.loss.UnbinnedNLL(data=self._data, model=pdf)
        gcl = GofCalculator(nll, ndof=6)
        pvl = gcl.get_gof(kind='pvalue')

        self._last_pdf = pdf
        self._niter   += 1

        fom = (pvl - 0.5) ** 2
        log.debug(f'{self._niter:<10}{pvl:<10.3f}{fom:<10.3f}{bandwidth:<10.0f}')

        if fom < 0.1:
            return 0.0

        return fom
    # --------------------------
    def get_pdf(self) -> zpdf:
        '''
        Returns KDE1DimFFT with optimal bandwidth
        '''
        log.debug(40 * '-')
        log.debug(f'{"Iteration":<10}{"Pvalue":<10}{"FOM":<10}{"Bandwidth":<10}')
        log.debug(40 * '-')

        for bwt in range(1, 100, 10):
            fom = self._calculate_fom(bandwidth=bwt)
            if fom < 0.01:
                break

        log.debug(40 * '-')
        pvl = 1 - fom

        if self._last_pdf is None:
            raise ValueError('PDF not found')

        if pvl < 0.2:
            log.warning(f'Took {self._niter} iterations for pvalue/bandwidth: {pvl:.3f}/{bwt}')
        else:
            log.debug(f'Took {self._niter} iterations for pvalue/bandwidth: {pvl:.3f}/{bwt}')

        return self._last_pdf
# -----------------------------------------------------
