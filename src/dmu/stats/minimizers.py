'''
Module containing derived classes from ZFit minimizer
'''
import numpy

import zfit
from zfit.result                   import FitResult
from zfit.core.basepdf             import BasePDF           as zpdf
from zfit.minimizers.baseminimizer import FailMinimizeNaN
from dmu.stats.gof_calculator      import GofCalculator
from dmu.logging.log_store         import LogStore

log = LogStore.add_logger('dmu:ml:minimizers')
# ------------------------
class AnealingMinimizer(zfit.minimize.Minuit):
    '''
    Class meant to minimizer zfit likelihoods by using multiple retries,
    each retry is preceeded by the randomization of the fitting parameters
    '''
    # ------------------------
    def __init__(self, ntries : int, pvalue : float = -1, chi2ndof : float = -1):
        '''
        ntries  : Try this number of times
        pvalue  : Stop tries when this threshold is reached
        chi2ndof: Use this value as a threshold to stop fits
        '''
        self._ntries   = ntries
        self._pvalue   = pvalue
        self._chi2ndof = chi2ndof

        self._check_thresholds()

        super().__init__()
    # ------------------------
    def _check_thresholds(self) -> None:
        good_pvalue  = 0 <= self._pvalue < 1
        good_chi2dof = self._chi2ndof > 0

        if good_pvalue and good_chi2dof:
            raise ValueError('Threshold for both chi2 and pvalue were specified')

        if good_pvalue:
            log.debug(f'Will use threshold on pvalue with value: {self._pvalue}')
            return

        if good_chi2dof:
            log.debug(f'Will use threshold on chi2ndof with value: {self._chi2ndof}')
            return

        raise ValueError('Neither pvalue nor chi2 thresholds are valid')
    # ------------------------
    def _is_good_fit(self, nll) -> bool:
        log.debug('Checking GOF')

        gcl = GofCalculator(nll)
        pvl = gcl.get_gof(kind='pvalue')
        ch2 = gcl.get_gof(kind='chi2/ndof')

        is_good_pval = pvl > self._pvalue   and self._pvalue   > 0
        is_good_chi2 = ch2 < self._chi2ndof and self._chi2ndof > 0
        is_good      = is_good_pval or is_good_chi2

        if is_good_pval:
            log.info(f'Stopping fit, found p-value: {pvl:.3f} > {self._pvalue:.3f}')

        if is_good_chi2:
            log.info(f'Stopping fit, found chi2/ndof: {ch2:.3f} > {self._chi2ndof:.3f}')

        if not is_good:
            log.debug(f'Could not read threshold, pvalue/chi2: {pvl:.3f}/{ch2:.3f}')

        return is_good
    # ------------------------
    def _randomize_parameters(self, nll):
        '''
        Will move floating parameters of PDF according
        to uniform PDF
        '''

        log.debug('Randomizing parameters')
        l_model = nll.model
        if len(l_model) != 1:
            raise ValueError('Not found and and only one model')

        model = l_model[0]
        s_par = model.get_params(floating=True)
        for par in s_par:
            ival = par.value()
            fval = numpy.random.uniform(par.lower, par.upper)
            par.set_value(fval)
            log.debug(f'{par.name:<20}{ival:<15.3f}{"->":<10}{fval:<15.3f}{"in":<5}{par.lower:<15.3e}{par.upper:<15.3e}')
    # ------------------------
    def minimize(self, nll, **kwargs):
        '''
        Will run minimization and return FitResult object
        '''
        for i_try in range(self._ntries):
            log.info(f'try {i_try:02}/{self._ntries:02}')
            try:
                res = super().minimize(nll, **kwargs)
            except (FailMinimizeNaN, ValueError, RuntimeError) as exc:
                log.warning(exc)
                self._randomize_parameters(nll)
                continue

            if self._is_good_fit(nll):
                return res

            self._randomize_parameters(nll)

        return res
# ------------------------
