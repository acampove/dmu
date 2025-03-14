'''
Module containing CVDiagnostics class
'''
import os

import numpy
import matplotlib
import matplotlib.pyplot as plt
import pandas            as pnd

from scipy.stats           import kendalltau
from ROOT                  import RDataFrame
from dmu.ml.cv_classifier  import CVClassifier
from dmu.ml.cv_predict     import CVPredict
from dmu.logging.log_store import LogStore

NPA = numpy.ndarray
Axis= matplotlib.axes._axes.Axes
log = LogStore.add_logger('dmu:ml:cv_diagnostics')
# -------------------------
class CVDiagnostics:
    '''
    Class meant to rundiagnostics on classifier

    Correlations
    ------------------
    Will calculate correlations between features + signal probability and some external target variable specified in the config
    '''
    # -------------------------
    def __init__(self, models : list[CVClassifier], rdf : RDataFrame, cfg : dict):
        self._l_model = models
        self._cfg     = cfg
        self._rdf     = rdf
        self._target  = cfg['correlations']['target']
        self._l_feat  = self._get_features()
    # -------------------------
    def _get_features(self) -> list[str]:
        cfg   = self._l_model[0].cfg
        l_var = cfg['training']['features']

        return l_var
    # -------------------------
    def _add_columns(self, rdf : RDataFrame) -> RDataFrame:
        cfg    = self._l_model[0].cfg
        d_def  = cfg['dataset']['define']
        for var, expr in d_def.items():
            rdf = rdf.Define(var, expr)

        return rdf
    # -------------------------
    def _get_scores(self) -> NPA:
        if 'score_from_rdf' not in self._cfg:
            log.debug('Using score from model')
            prd = CVPredict(models=self._l_model, rdf = self._rdf)

            return prd.predict()

        name = self._cfg['score_from_rdf']
        log.debug(f'Picking up score from dataframe, column: {name}')
        arr_score = self._rdf.AsNumpy([name])[name]

        return arr_score
    # -------------------------
    def _get_arrays(self) -> dict[str, NPA]:
        rdf   = self._add_columns(self._rdf)
        l_col = [ name.c_str() for name in rdf.GetColumnNames() ]

        missing= False
        l_var  = self._l_feat + [self._target]
        for var in l_var:
            if var not in l_col:
                log.error(f'{"Missing":<20}{var}')
                missing=True

        if missing:
            raise ValueError('Columns missing')

        d_var          = rdf.AsNumpy(l_var)
        d_var['score'] = self._get_scores() 

        return d_var
    # -------------------------
    def _run_correlations(self, method : str, ax : Axis) -> Axis:
        d_arr      = self._get_arrays()
        arr_target = d_arr[self._target]

        d_corr= {}
        for name, arr_val in d_arr.items():
            if name == self._target:
                continue

            d_corr[name] = self._calculate_correlations(var=arr_val, target=arr_target, method=method)

        ax = self._plot_correlations(d_corr=d_corr, method=method, ax=ax)

        return ax
    # -------------------------
    def _plot_correlations(self, d_corr : dict[str,float], method : str, ax : Axis) -> Axis:
        df      = pnd.DataFrame.from_dict(d_corr, orient="index", columns=[method])
        figsize = self._cfg['correlations']['figure']['size']
        ax      = df.plot(label=method, figsize=figsize, ax=ax)

        return ax
    # -------------------------
    def _save_plot(self):
        plot_dir = self._cfg['output']
        os.makedirs(plot_dir, exist_ok=True)

        plot_path = f'{plot_dir}/correlations.png'
        log.info(f'Saving to: {plot_path}')

        title = None
        if 'title' in self._cfg['correlations']['figure']:
            title = self._cfg['correlations']['figure']['title']

        rotation=30
        if 'rotate' in self._cfg['correlations']['figure']:
            rotation = self._cfg['correlations']['figure']['rotate']

        plt.ylim(-1, +1)
        plt.title(title)
        plt.ylabel('Correlation')
        plt.grid()
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    # -------------------------
    def _remove_nans(self, var : NPA, tgt : NPA) -> tuple[NPA,NPA]:
        arr_nan_var = numpy.isnan(var)
        arr_nan_tgt = numpy.isnan(tgt)
        arr_is_nan  = numpy.logical_or(arr_nan_var, arr_nan_tgt)
        arr_not_nan = numpy.logical_not(arr_is_nan)

        var         = var[arr_not_nan]
        tgt         = tgt[arr_not_nan]

        return var, tgt
    # -------------------------
    def _calculate_correlations(self, var : NPA, target : NPA, method : str) -> float:
        var, target = self._remove_nans(var, target)

        if method == 'Pearson':
            mat         = numpy.corrcoef(var, target)

            return mat[0,1]

        if method == r'Kendall-$\tau$':
            tau, _ = kendalltau(var, target)

            return tau

        raise NotImplementedError(f'Correlation coefficient {method} not implemented')
    # -------------------------
    def run(self) -> None:
        '''
        Runs diagnostics
        '''
        if 'correlations' in self._cfg:
            ax = None
            for method in self._cfg['correlations']['methods']:
                ax = self._run_correlations(method=method, ax=ax)

            self._save_plot()
# -------------------------
