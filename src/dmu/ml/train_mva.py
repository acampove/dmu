'''
Module with TrainMva class
'''
# pylint: disable = too-many-locals, no-name-in-module
# pylint: disable = too-many-arguments, too-many-positional-arguments
# pylint: disable = too-many-instance-attributes
# pylint: disable = too-many-arguments, too-many-positional-arguments

import os
import copy
import json
import math

import joblib
import pandas as pnd
import numpy
import matplotlib.pyplot as plt

from sklearn.metrics         import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from ROOT import RDataFrame, RDF

import dmu.ml.utilities         as ut
import dmu.pdataframe.utilities as put
import dmu.plotting.utilities   as plu

from dmu.ml.cv_diagnostics   import CVDiagnostics
from dmu.ml.cv_classifier    import CVClassifier as cls
from dmu.plotting.plotter_1d import Plotter1D    as Plotter
from dmu.plotting.matrix     import MatrixPlotter
from dmu.logging.log_store   import LogStore

NPA = numpy.ndarray
log = LogStore.add_logger('dmu:ml:train_mva')
# ---------------------------------------------
class NoFeatureInfo(Exception):
    '''
    Used when information about a feature is missing in the config file
    '''
    def __init__(self, message : str):
        super().__init__(message)
# ---------------------------------------------
class TrainMva:
    '''
    Interface to scikit learn used to train classifier
    '''
    # ---------------------------------------------
    def __init__(self, bkg : RDataFrame, sig : RDataFrame, cfg : dict):
        '''
        bkg (ROOT dataframe): Holds real data
        sig (ROOT dataframe): Holds simulation
        cfg (dict)          : Dictionary storing configuration for training
        '''
        self._cfg       = cfg
        self._auc       = math.nan # This is where the Area Under the ROC curve for the full sample will be saved
        self._l_ft_name = self._cfg['training']['features']

        self._rdf_sig_org = sig
        self._rdf_bkg_org = bkg

        rdf_bkg = self._preprocess_rdf(bkg)
        rdf_sig = self._preprocess_rdf(sig)

        df_ft_sig, l_lab_sig = self._get_sample_inputs(rdf = rdf_sig, label = 1)
        df_ft_bkg, l_lab_bkg = self._get_sample_inputs(rdf = rdf_bkg, label = 0)

        self._df_ft = pnd.concat([df_ft_sig, df_ft_bkg], axis=0)
        self._l_lab = numpy.array(l_lab_sig + l_lab_bkg)

        self._rdf_bkg = self._get_rdf(rdf = rdf_bkg, df_feat=df_ft_bkg)
        self._rdf_sig = self._get_rdf(rdf = rdf_sig, df_feat=df_ft_sig)
    # ---------------------------------------------
    def _get_extra_columns(self, rdf : RDataFrame, df : pnd.DataFrame) -> list[str]:
        d_plot = self._cfg['plotting']['features']['plots']
        l_expr = list(d_plot)
        l_rdf  = [ name.c_str() for name in rdf.GetColumnNames() ]

        l_extr = []
        for expr in l_expr:
            if expr not in l_rdf:
                continue

            if expr in df.columns:
                continue

            l_extr.append(expr)

        return l_extr
    # ---------------------------------------------
    def _get_rdf(self, rdf : RDataFrame, df_feat : pnd.DataFrame) -> RDataFrame:
        '''
        Takes original ROOT dataframe and pre-processed features dataframe
        Adds missing branches to latter and returns expanded ROOT dataframe
        Need to make plots
        '''

        l_extr_col = self._get_extra_columns(rdf, df_feat)
        if len(l_extr_col) > 20:
            for name in l_extr_col:
                log.debug(name)
            raise ValueError('Found more than 20 extra columns')

        d_data = rdf.AsNumpy(l_extr_col)
        log.debug(f'Adding extra-nonfeature columns: {l_extr_col}')
        df_extr = pnd.DataFrame(d_data)

        nmain = len(df_feat.columns)
        nextr = len(df_extr.columns)

        log.debug(f'Main  DF size: {nmain}')
        log.debug(f'Extra DF size: {nextr}')

        df_all = pnd.concat([df_feat, df_extr], axis=1)

        return RDF.FromPandas(df_all)
    # ---------------------------------------------
    def _pre_process_nans(self, df : pnd.DataFrame) -> pnd.DataFrame:
        if 'dataset' not in self._cfg:
            return df

        if 'nan' not in self._cfg['dataset']:
            log.debug('dataset/nan section not found, not pre-processing NaNs')
            return df

        d_name_val = self._cfg['dataset']['nan']
        log.info(70 * '-')
        log.info('Doing NaN replacements')
        log.info(70 * '-')
        for var, val in d_name_val.items():
            nna = df[var].isna().sum()

            log.info(f'{var:<20}{"--->":20}{val:<20.3f}{nna}')
            df[var] = df[var].fillna(val)
        log.info(70 * '-')

        return df
    # ---------------------------------------------
    def _preprocess_rdf(self, rdf : RDataFrame) -> RDataFrame:
        if 'define' not in self._cfg['dataset']:
            log.debug('No definitions found')
            return rdf

        log.debug('Definitions found')
        d_def = self._cfg['dataset']['define']
        for name, expr in d_def.items():
            log.debug(f'{name:<20}{expr}')
            try:
                rdf = rdf.Define(name, expr)
            except TypeError as exc:
                l_col = [ name.c_str() for name in rdf.GetColumnNames() ]
                branch_list = 'found_branches.txt'
                with open(branch_list, 'w', encoding='utf-8') as ifile:
                    json.dump(l_col, ifile, indent=2)

                raise TypeError(f'Branches found were dumped to {branch_list}') from exc

        return rdf
    # ---------------------------------------------
    def _get_sample_inputs(self, rdf : RDataFrame, label : int) -> tuple[pnd.DataFrame, list[int]]:
        d_ft = rdf.AsNumpy(self._l_ft_name)
        df   = pnd.DataFrame(d_ft)
        df   = self._pre_process_nans(df)
        df   = ut.cleanup(df)
        l_lab= len(df) * [label]

        return df, l_lab
    # ---------------------------------------------
    def _get_model(self, arr_index : NPA) -> cls:
        model = cls(cfg = self._cfg)
        df_ft = self._df_ft.iloc[arr_index]
        l_lab = self._l_lab[arr_index]

        log.debug(f'Training feature shape: {df_ft.shape}')
        log.debug(f'Training label size: {len(l_lab)}')

        model.fit(df_ft, l_lab)

        return model
    # ---------------------------------------------
    def _get_models(
            self,
            load_trained : bool,
            workers      :  int) -> list[cls]:
        '''
        Will create models, train them and return them
        '''
        if load_trained:
            log.warning('Not retraining, but loading trained models')
            return self._load_trained_models()

        nfold = self._cfg['training']['nfold']
        rdmst = self._cfg['training']['rdm_stat']
        if nfold < workers:
            log.warning(f'Asked {workers} workers, but only found {nfold} folds, running with {nfold} workers')
            workers = nfold

        kfold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=rdmst)

        l_model=[]
        ifold=0

        l_arr_lab_ts = []
        l_arr_all_ts = []
        l_arr_sig_ts = []
        l_arr_bkg_ts = []

        for arr_itr, arr_its in kfold.split(self._df_ft, self._l_lab):
            log.debug(20 * '-')
            log.info(f'Training fold: {ifold}')
            log.debug(20 * '-')
            model = self._get_model(arr_itr)
            l_model.append(model)

            arr_sig_tr, arr_bkg_tr, arr_all_tr, arr_lab_tr = self._get_scores(model, arr_itr, on_training_ok= True)
            arr_sig_ts, arr_bkg_ts, arr_all_ts, arr_lab_ts = self._get_scores(model, arr_its, on_training_ok=False)

            self._save_feature_importance(model, ifold)
            self._plot_correlations(arr_itr, ifold)
            self._plot_scores(
                    ifold  =     ifold,
                    sig_trn=arr_sig_tr,
                    sig_tst=arr_sig_ts,
                    bkg_trn=arr_bkg_tr,
                    bkg_tst=arr_bkg_ts)

            TrainMva.plot_roc(arr_lab_ts, arr_all_ts, kind='Test' , ifold=ifold)
            xval, yval, _ = TrainMva.plot_roc(arr_lab_tr, arr_all_tr, kind='Train', ifold=ifold)
            self._plot_probabilities(xval, yval, arr_all_tr, arr_lab_tr)
            self._save_roc(xval=xval, yval=yval, ifold=ifold)

            ifold+=1

            l_arr_lab_ts.append(arr_lab_ts)
            l_arr_all_ts.append(arr_all_ts)
            l_arr_sig_ts.append(arr_sig_ts)
            l_arr_bkg_ts.append(arr_bkg_ts)

        arr_lab_ts = numpy.concatenate(l_arr_lab_ts)
        arr_all_ts = numpy.concatenate(l_arr_all_ts)
        arr_sig_ts = numpy.concatenate(l_arr_sig_ts)
        arr_bkg_ts = numpy.concatenate(l_arr_bkg_ts)

        xval, yval, self._auc = TrainMva.plot_roc(
                arr_lab_ts,
                arr_all_ts,
                kind ='Test',
                ifold=-1)

        self._plot_probabilities(xval, yval, arr_all_ts, arr_lab_ts)
        self._save_roc(xval=xval, yval=yval, ifold=-1)
        self._plot_scores(ifold=-1, sig_tst=arr_sig_ts, bkg_tst=arr_bkg_ts)

        return l_model
    # ---------------------------------------------
    def _save_roc(self, xval : NPA, yval : NPA, ifold : int) -> None:
        ifold    = 'all' if ifold == -1 else ifold # -1 represents all the testing datasets combined
        val_dir  = self._cfg['saving']['output']
        val_dir  = f'{val_dir}/fold_{ifold:03}'
        os.makedirs(val_dir, exist_ok=True)
        plt_path = f'{val_dir}/roc.png'
        jsn_path = plt_path.replace('.png', '.json')
        df       = pnd.DataFrame({'x' : xval, 'y' : yval})
        df.to_json(jsn_path, indent=2)

        min_x = 0
        min_y = 0
        if 'min' in self._cfg['plotting']['roc']:
            [min_x, min_y] = self._cfg['plotting']['roc']['min']

        max_x = 1
        max_y = 1
        if 'max' in self._cfg['plotting']['roc']:
            [max_x, max_y] = self._cfg['plotting']['roc']['max']

        plt.xlabel('Signal efficiency')
        plt.ylabel('Background rejection')
        plt.title(f'Fold: {ifold}')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.grid()
        plt.legend()
        plt.savefig(plt_path)
        plt.close()
    # ---------------------------------------------
    def _load_trained_models(self) -> list[cls]:
        out_dir    = self._cfg['saving']['output']
        model_path = f'{out_dir}/model.pkl'
        nfold      = self._cfg['training']['nfold']
        l_model    = []
        for ifold in range(nfold):
            fold_path = model_path.replace('.pkl', f'_{ifold:03}.pkl')

            if not os.path.isfile(fold_path):
                raise FileNotFoundError(f'Missing trained model: {fold_path}')

            log.debug(f'Loading model from: {fold_path}')
            model = joblib.load(fold_path)
            l_model.append(model)

        return l_model
    # ---------------------------------------------
    def _labels_from_varnames(self, l_var_name : list[str]) -> list[str]:
        try:
            d_plot = self._cfg['plotting']['features']['plots']
        except KeyError as exc:
            raise KeyError('Cannot find plotting/features/plots section in config, using dataframe names') from exc

        l_label = []
        for var_name in l_var_name:
            if var_name not in d_plot:
                raise NoFeatureInfo(f'No plot found for feature {var_name}, cannot extract label')

            d_setting = d_plot[var_name]
            if 'labels' not in d_setting:
                raise NoFeatureInfo(f'No no labels present for plot of feature {var_name}, cannot extract label')

            [xlab, _ ]= d_setting['labels']

            l_label.append(xlab)

        return l_label
    # ---------------------------------------------
    def _save_feature_importance(self, model : cls, ifold : int) -> None:
        l_var_name           = self._df_ft.columns.tolist()

        d_data               = {}
        d_data['Variable'  ] = self._labels_from_varnames(l_var_name)
        d_data['Importance'] = 100 * model.feature_importances_

        val_dir  = self._cfg['saving']['output']
        val_dir  = f'{val_dir}/fold_{ifold:03}'
        os.makedirs(val_dir, exist_ok=True)

        df = pnd.DataFrame(d_data)
        df = df.sort_values(by='Importance', ascending=False)

        table_path = f'{val_dir}/importance.tex'
        d_form = {'Variable' : '{}', 'Importance' : '{:.1f}'}
        put.df_to_tex(df, table_path, d_format = d_form)
    # ---------------------------------------------
    def _get_scores(self, model : cls, arr_index : NPA, on_training_ok : bool) -> tuple[NPA, NPA, NPA, NPA]:
        '''
        Returns a tuple of four arrays

        arr_sig : Signal probabilities for signal
        arr_bkg : Signal probabilities for background
        arr_all : Signal probabilities for both
        arr_lab : Labels for both
        '''
        nentries = len(arr_index)
        log.debug(f'Getting {nentries} signal probabilities')

        df_ft    = self._df_ft.iloc[arr_index]
        arr_prob = model.predict_proba(df_ft, on_training_ok=on_training_ok)
        arr_lab  = self._l_lab[arr_index]

        l_all    = [ sig_prob for [_, sig_prob] in arr_prob ]
        arr_all  = numpy.array(l_all)

        arr_sig, arr_bkg= self._split_scores(arr_prob=arr_prob, arr_label=arr_lab)

        return arr_sig, arr_bkg, arr_all, arr_lab
    # ---------------------------------------------
    def _split_scores(self, arr_prob : NPA, arr_label : NPA) -> tuple[NPA, NPA]:
        '''
        Will split the testing scores (predictions) based on the training scores

        tst is a list of lists as [p_bkg, p_sig]
        '''

        l_sig = [ prb[1] for prb, lab in zip(arr_prob, arr_label) if lab == 1]
        l_bkg = [ prb[1] for prb, lab in zip(arr_prob, arr_label) if lab == 0]

        arr_sig = numpy.array(l_sig)
        arr_bkg = numpy.array(l_bkg)

        return arr_sig, arr_bkg
    # ---------------------------------------------
    def _save_model(self, model : cls, ifold : int) -> None:
        '''
        Saves a model, associated to a specific fold
        '''
        out_dir    = self._cfg['saving']['output']
        model_path = f'{out_dir}/model.pkl'

        if os.path.isfile(model_path):
            log.info(f'Model found in {model_path}, not saving')
            return

        dir_name = os.path.dirname(model_path)
        os.makedirs(dir_name, exist_ok=True)

        model_path = model_path.replace('.pkl', f'_{ifold:03}.pkl')

        log.info(f'Saving model to: {model_path}')
        joblib.dump(model, model_path)
    # ---------------------------------------------
    def _get_correlation_cfg(self, df : pnd.DataFrame, ifold : int) -> dict:
        l_var_name = df.columns.tolist()
        l_label    = self._labels_from_varnames(l_var_name)
        cfg = {
                'labels'     : l_label,
                'title'      : f'Fold {ifold}',
                'label_angle': 45,
                'upper'      : True,
                'zrange'     : [-1, +1],
                'size'       : [7, 7],
                'format'     : '{:.3f}',
                'fontsize'   : 12,
                }

        if 'correlation' not in self._cfg['plotting']:
            log.info('Using default correlation plotting configuration')
            return cfg

        log.debug('Updating correlation plotting configuration')
        custom = self._cfg['plotting']['correlation']
        cfg.update(custom)

        return cfg
    # ---------------------------------------------
    def _plot_correlations(self, arr_index : NPA, ifold : int) -> None:
        log.debug('Plotting correlations')

        df_ft = self._df_ft.iloc[arr_index]
        l_lab = self._l_lab[arr_index]

        arr_sig_idx, = numpy.where(l_lab == 1)
        arr_bkg_idx, = numpy.where(l_lab == 0)

        df_ft_sig = df_ft.iloc[arr_sig_idx]
        df_ft_bkg = df_ft.iloc[arr_bkg_idx]

        self._plot_correlation(df_ft=df_ft_sig, ifold=ifold, name='signal'    )
        self._plot_correlation(df_ft=df_ft_bkg, ifold=ifold, name='background')
    # ---------------------------------------------
    def _plot_correlation(
            self,
            df_ft : pnd.DataFrame,
            ifold : int,
            name  : str) -> None:

        log.debug(f'Plotting correlation for {name}/{ifold} fold')

        cfg = self._get_correlation_cfg(df_ft, ifold)
        cov = df_ft.corr()
        mat = cov.to_numpy()

        val_dir  = self._cfg['saving']['output']
        val_dir  = f'{val_dir}/fold_{ifold:03}'
        os.makedirs(val_dir, exist_ok=True)

        obj = MatrixPlotter(mat=mat, cfg=cfg)
        obj.plot()
        plt.savefig(f'{val_dir}/correlation_{name}.png')
        plt.close()
    # ---------------------------------------------
    def _get_nentries(self, arr_val : NPA) -> str:
        size = len(arr_val)
        size = size / 1000.

        return f'{size:.2f}K'
    # ---------------------------------------------
    def _plot_scores(
            self,
            ifold   : int,
            sig_tst : NPA,
            bkg_tst : NPA,
            sig_trn : NPA = None,
            bkg_trn : NPA = None) -> None:
        '''
        Will plot an array of scores, associated to a given fold
        '''
        ifold = 'all' if ifold == -1 else ifold
        log.debug(f'Plotting scores for {ifold} fold')

        val_dir  = self._cfg['saving']['output']
        val_dir  = f'{val_dir}/fold_{ifold:03}'
        os.makedirs(val_dir, exist_ok=True)

        plt.hist(sig_tst, histtype='step', bins=50, range=(0,1), color='b', density=True, label='Signal Test: '     + self._get_nentries(sig_tst))
        plt.hist(bkg_tst, histtype='step', bins=50, range=(0,1), color='r', density=True, label='Background Test: ' + self._get_nentries(bkg_tst))

        if sig_trn is not None and bkg_trn is not None:
            plt.hist(sig_trn, alpha = 0.3, bins=50, range=(0,1), color='b', density=True, label='Signal Train: '    + self._get_nentries(sig_trn))
            plt.hist(bkg_trn, alpha = 0.3, bins=50, range=(0,1), color='r', density=True, label='Background Train: '+ self._get_nentries(bkg_trn))

        plt.legend()
        plt.title(f'Fold: {ifold}')
        plt.xlabel('Signal probability')
        plt.ylabel('Normalized')
        plt.savefig(f'{val_dir}/scores.png')
        plt.close()
    # ---------------------------------------------
    def _plot_probabilities(
            self,
            arr_seff: NPA,
            arr_brej: NPA,
            arr_sprb: NPA,
            arr_labl: NPA) -> None:

        roc_cfg = self._cfg['plotting']['roc']
        if 'annotate' not in roc_cfg:
            log.debug('Annotation section in the ROC curve config not found, skipping annotation')
            return

        l_sprb   = [ sprb for sprb, labl in zip(arr_sprb, arr_labl) if labl == 1 ]
        arr_sprb = numpy.array(l_sprb)

        plt_cfg = roc_cfg['annotate']
        if 'sig_eff' not in plt_cfg:
            l_seff_target = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        else:
            l_seff_target = plt_cfg['sig_eff']
            del plt_cfg['sig_eff']

        arr_seff_target = numpy.array(l_seff_target)
        arr_quantile    = 1 - arr_seff_target

        l_score = numpy.quantile(arr_sprb, arr_quantile)
        l_seff  = []
        l_brej  = []

        log.debug(60 * '-')
        log.debug(f'{"SigEff":20}{"BkgRej":20}{"Score":20}')
        log.debug(60 * '-')
        for seff_target, score in zip(arr_seff_target, l_score):
            arr_diff = numpy.abs(arr_seff - seff_target)
            ind      = numpy.argmin(arr_diff)

            seff     = arr_seff[ind]
            brej     = arr_brej[ind]

            log.debug(f'{seff:<20.3f}{brej:<20.3f}{score:<20.2f}')

            l_seff.append(seff)
            l_brej.append(brej)

        plu.annotate(l_x=l_seff, l_y=l_brej, l_v=l_score, **plt_cfg)
    # ---------------------------------------------
    def _plot_features(self):
        '''
        Will plot the features, based on the settings in the config
        '''
        out_dir         = self._cfg['saving']['output']
        d_cfg           = self._cfg['plotting']['features']
        d_cfg['saving'] = {'plt_dir' : f'{out_dir}/features'}

        ptr   = Plotter(d_rdf = {'Signal' : self._rdf_sig, 'Background' : self._rdf_bkg}, cfg=d_cfg)
        ptr.run()
    # ---------------------------------------------
    def _save_settings_to_tex(self) -> None:
        self._save_nan_conversion()
        self._save_hyperparameters_to_tex()
    # ---------------------------------------------
    def _save_nan_conversion(self) -> None:
        if 'dataset' not in self._cfg:
            return

        if 'nan' not in self._cfg['dataset']:
            log.debug('NaN section not found, not saving it')
            return

        d_nan = self._cfg['dataset']['nan']
        l_var = list(d_nan)
        l_lab = self._labels_from_varnames(l_var)
        l_val = list(d_nan.values())

        d_tex = {'Variable' : l_lab, 'Replacement' : l_val}
        df    = pnd.DataFrame(d_tex)
        val_dir  = self._cfg['saving']['output']
        os.makedirs(val_dir, exist_ok=True)
        put.df_to_tex(df, f'{val_dir}/nan_replacement.tex')
    # ---------------------------------------------
    def _save_hyperparameters_to_tex(self) -> None:
        if 'hyper' not in self._cfg['training']:
            raise ValueError('Cannot find hyper parameters in configuration')

        d_hyper = self._cfg['training']['hyper']
        d_form  = { f'\\verb|{key}|' : f'\\verb|{val}|' for key, val in d_hyper.items() }
        d_latex = { 'Hyperparameter' : list(d_form.keys()), 'Value' : list(d_form.values())}

        df = pnd.DataFrame(d_latex)
        val_dir  = self._cfg['saving']['output']
        os.makedirs(val_dir, exist_ok=True)
        put.df_to_tex(df, f'{val_dir}/hyperparameters.tex')
    # ---------------------------------------------
    def _run_diagnostics(self, models : list[cls], rdf : RDataFrame, name : str) -> None:
        if 'diagnostics' not in self._cfg:
            log.warning('Diagnostics section not found, not running diagnostics')
            return

        cfg_diag = self._cfg['diagnostics']
        out_dir  = cfg_diag['output']
        plt_dir  = None

        if 'overlay' in cfg_diag['correlations']['target']:
            plt_dir  = cfg_diag['correlations']['target']['overlay']['saving']['plt_dir']

        cfg_diag = copy.deepcopy(cfg_diag)
        cfg_diag['output'] = f'{out_dir}/{name}'
        if plt_dir is not None:
            cfg_diag['correlations']['target']['overlay']['saving']['plt_dir'] = f'{plt_dir}/{name}'

        cvd = CVDiagnostics(models=models, rdf=rdf, cfg=cfg_diag)
        cvd.run()
    # ---------------------------------------------
    def run(
            self,
            workers      : int  = 1,
            skip_fit     : bool = False,
            load_trained : bool = False) -> float:
        '''
        Will run the training

        Parameters
        ----------------
        workers     : By default 1, but can be used to run with multiple processes, one per fold
        skip_fit    : By default false, if True, it will only do the plots of features and save tables
        load_trained: If true, it will load the models instead of training, by default false

        Returns
        ----------------
        Area under the ROC curve from evaluating the classifiers
        on samples that were not used in their training. Uses the full sample
        '''
        self._save_settings_to_tex()
        self._plot_features()

        if skip_fit:
            return self._auc

        l_mod = self._get_models(load_trained = load_trained, workers=workers)
        if not load_trained:
            for ifold, mod in enumerate(l_mod):
                self._save_model(mod, ifold)

        self._run_diagnostics(models = l_mod, rdf = self._rdf_sig_org, name='Signal'    )
        self._run_diagnostics(models = l_mod, rdf = self._rdf_bkg_org, name='Background')

        return self._auc
    # ---------------------------------------------
    @staticmethod
    def plot_roc_from_prob(
            arr_sig_prb : NPA,
            arr_bkg_prb : NPA,
            kind        : str,
            ifold       : int,
            color       : str = None) -> tuple[NPA,NPA, float]:
        '''
        Takes arrays of signal and background probabilities
        and plots ROC curve

        Parameters
        --------------------
        arr_bkg/sig_prb : Array with background/signal probabilities
        kind            : String used to label the plot
        ifold           : If no fold makes sense (i.e. this is the full sample), use ifold=-1
        kind            : Used to label the plot
        color           : String with color of curve

        Returns
        --------------------
        Tuple with 3 elements:

        - Array of x coordinates of ROC curve
        - Array of y coordinates of ROC curve
        - Area under the curve
        '''
        arr_sig_lab = numpy.ones_like( arr_sig_prb)
        arr_bkg_lab = numpy.zeros_like(arr_bkg_prb)

        arr_prb     = numpy.concatenate([arr_sig_prb, arr_bkg_prb])
        arr_lab     = numpy.concatenate([arr_sig_lab, arr_bkg_lab])

        res = TrainMva.plot_roc(
                l_lab=arr_lab,
                l_prb=arr_prb,
                color=color,
                kind =kind,
                ifold=ifold)

        return res
    # ---------------------------------------------
    @staticmethod
    def plot_roc(
            l_lab : NPA,
            l_prb : NPA,
            kind  : str,
            ifold : int,
            color : str = None) -> tuple[NPA, NPA, float]:
        '''
        Takes the labels and the probabilities and plots ROC
        curve for given fold

        Parameters
        --------------------
        ifold : If no fold makes sense (i.e. this is the full sample), use ifold=-1
        kind  : Used to label the plot

        Returns
        --------------------
        Tuple with 3 elements:

        - Array of x coordinates of ROC curve
        - Array of y coordinates of ROC curve
        - Area under the curve
        '''
        log.debug(f'Plotting ROC curve for {ifold} fold')

        xval, yval, _ = roc_curve(l_lab, l_prb)
        xval          = 1 - xval
        area          = auc(xval, yval)

        if color is None:
            color='red' if kind == 'Train' else 'blue'

        if ifold == -1:
            label=f'Test sample: {area:.3f}'
        else:
            label=f'{kind}: {area:.3f}'

        plt.plot(xval, yval, color=color, label=label)

        return xval, yval, area
# ---------------------------------------------
