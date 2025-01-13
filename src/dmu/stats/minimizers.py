'''
Module containing derived classes from ZFit minimizer
'''

import zfit

# ------------------------
class AnealingMinimizer(zfit.minimize.Minuit):
    '''
    Class meant to minimizer zfit likelihoods by using multiple retries,
    each retry is preceeded by the randomization of the fitting parameters
    '''
    # ------------------------
    def __init__(self, ntries : int, pvalue : float):
        self._ntries = ntries
        self._pvalue = pvalue

        super().__init__()
    # ------------------------
    def _is_good_fit(self, nll) -> bool:
        log.debug('Checking GOF')

        gcl = GofCalculator(nll)
        gof = gcl.get_gof(kind='pvalue')

        is_good = gof > self._pvalue

        if is_good:
            log.info(f'Stopping fit, found p-value: {gof:.3f} > {self._pvalue:.3f}')

        log.debug(f'Found p-value: {gof:.3f} <= {self._pvalue:.3f}')

        return is_good
    # ------------------------
    def _randomize_parameters(self, nll):
        '''
        Will move floating parameters of PDF according
        to uniform PDF
        '''

        log.info('Randomizing parameters')
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
            log.info(f'try {i_try}')
            res = super().minimize(nll, **kwargs)
            if self._is_good_fit(nll):
                return res

            self._randomize_parameters(nll)

        return res
# ------------------------
