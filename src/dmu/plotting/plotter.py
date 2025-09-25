'''
Module containing plotter class
'''

import os
import json
import math

import numpy
import matplotlib.pyplot as plt
from pathlib import Path

from ROOT                  import RDF # type: ignore
from omegaconf             import DictConfig, OmegaConf
from dmu.logging.log_store import LogStore

log = LogStore.add_logger('dmu:plotting:Plotter')
# --------------------------------------------
class Plotter:
    '''
    Base class of Plotter1D and Plotter2D
    '''
    #-------------------------------------
    def __init__(
        self,
        d_rdf: dict|None            =None,
        cfg  : dict|DictConfig|None =None):
        '''
        Parameters
        --------------
        d_rdf: Dictionary where
            key  : Identifier of dataset
            value: ROOT dataframe representing dataset

        cfg  : Dictionary or DictConfig instance holding configuration
        '''
        if not isinstance(  cfg, (dict,DictConfig)):
            raise ValueError('Config dictionary not passed')

        if not isinstance(d_rdf, dict):
            raise ValueError('Dataframe dictionary not passed')

        if isinstance(cfg, dict):
            self._cfg = OmegaConf.create(obj=cfg)
        else:
            self._cfg = cfg

        self._d_rdf : dict[str, RDF.RNode]            = { name : self._preprocess_rdf(rdf=rdf, name=name) for name, rdf in d_rdf.items()}
        self._d_wgt : dict[str, numpy.ndarray] | None = None

        self._title : str = ''
    #-------------------------------------
    def _check_quantile(self, qnt : float):
        '''
        Will check validity of quantile
        '''

        if 0.5 < qnt <= 1.0:
            return

        raise ValueError(f'Invalid quantile: {qnt:.3e}, value needs to be in (0.5, 1.0] interval')
    #-------------------------------------
    def _find_bounds(self, d_data : dict, qnt : float = 0.98):
        '''
        Will take dictionary between kinds of data and numpy array
        Will return tuple with bounds, where 95% of the data is found
        '''
        self._check_quantile(qnt)

        l_max = []
        l_min = []

        for arr_val in d_data.values():
            minv = numpy.quantile(arr_val, 1 - qnt)
            maxv = numpy.quantile(arr_val,     qnt)

            l_max.append(maxv)
            l_min.append(minv)

        minx = min(l_min)
        maxx = max(l_max)

        if minx >= maxx:
            raise ValueError(f'Could not calculate bounds correctly: [{minx:.3e}, {maxx:.3e}]')

        return minx, maxx
    #-------------------------------------
    def _preprocess_rdf(
        self,
        rdf  : RDF.RNode,
        name : str) -> RDF.RNode:
        '''
        Parameters
        --------------
        rdf  : ROOT dataframe
        name : Name of sample associated to dataframe

        Returns
        --------------
        Preprocessed dataframe
        '''
        log.debug(f'Preprocessing dataframe for {name}')

        rdf = self._define_vars(rdf)
        if 'selection' in self._cfg:
            rdf = self._apply_selection(rdf)
            rdf = self._max_ran_entries(rdf)

        return rdf
    #-------------------------------------
    def _define_vars(self, rdf):
        '''
        Will define extra columns in dataframe and return updated dataframe
        '''
        if 'definitions' not in self._cfg:
            log.debug('No definitions section found, returning same RDF')
            return rdf

        log.info('Defining extra variables')
        for name, expr in self._cfg.definitions.items():
            log.debug(f'{name:<30}{expr:<150}')
            rdf = rdf.Define(name, expr)

        return rdf
    #-------------------------------------
    def _apply_selection(self, rdf):
        '''
        Will take dataframe, apply selection and return dataframe
        '''
        if 'cuts' not in self._cfg.selection:
            log.debug('Cuts not found in selection section, not applying any cuts')
            return rdf

        log.debug('Applying cuts')
        for name, cut in self._cfg.selection.cuts.items():
            log.debug(f'{name:<50}{cut:<150}')
            rdf = rdf.Filter(cut, name)

        return rdf
    #-------------------------------------
    def _max_ran_entries(self, rdf):
        '''
        Will take dataframe and randomly drop events
        '''

        if 'max_ran_entries' not in self._cfg['selection']:
            log.debug('Cuts not found in selection section, not applying any cuts')
            return rdf

        tot_entries = rdf.Count().GetValue()
        max_entries = self._cfg.selection.max_ran_entries

        if tot_entries < max_entries:
            log.debug(f'Not dropping dandom entries: {tot_entries} < {max_entries}')
            return rdf

        prescale = math.floor(tot_entries / max_entries)
        if prescale < 2:
            log.debug(f'Not dropping random entries, prescale is below 2: {tot_entries}/{max_entries}')
            return rdf

        rdf = rdf.Filter(f'rdfentry_ % {prescale} == 0', 'max_ran_entries')

        fnl_entries = rdf.Count().GetValue()

        log.info(f'Dropped entries randomly: {tot_entries} -> {fnl_entries}')

        return rdf
    # --------------------------------------------
    def _print_weights(self, arr_wgt : numpy.ndarray|None, var : str, sample : str) -> None:
        if arr_wgt is None:
            log.debug(f'Not using weights for {sample}:{var}')
            return

        num_wgt = len(arr_wgt)
        sum_wgt = numpy.sum(arr_wgt)

        log.debug(f'Using weights [{num_wgt},{sum_wgt:.0f}] for {var}')
    # --------------------------------------------
    def _get_fig_size(self):
        '''
        Will read size list from config dictionary if found
        other wise will return None
        '''
        if 'general' not in self._cfg:
            return None

        if 'size' not in self._cfg.general:
            return None

        return self._cfg.general.size
    #-------------------------------------
    def _get_weights(self, var : str) -> dict[str, numpy.ndarray] | None:
        '''
        Parameters
        ----------------
        var: Variable name

        Returns
        ----------------
        Dictionary with weights
        '''
        cfg = self._cfg.plots[var]
        if 'weights' not in cfg:
            return None

        # Pick cached weights
        if self._d_wgt is not None: 
            return self._d_wgt

        d_weight    = {sam_name : self._read_weights(name=cfg.weights, rdf=rdf) for sam_name, rdf in self._d_rdf.items()}
        self._d_wgt = d_weight

        return d_weight
    # --------------------------------------------
    def _read_weights(self, name : str, rdf : RDF.RNode) -> numpy.ndarray:
        '''
        Parameters
        -------------------
        name: Name of column holding weights
        rdf : ROOT dataframe with weights

        Returns
        -------------------
        Numpy array with weights or ones if no weight found
        '''
        v_col = rdf.GetColumnNames()
        l_col = [ col.c_str() for col in v_col ]

        if name not in l_col:
            nentries = rdf.Count().GetValue()
            log.debug(f'Weight {name} not found, using ones')

            return numpy.ones(nentries)

        log.debug(f'Weight {name} found')
        arr_wgt = rdf.AsNumpy([name])[name]

        return arr_wgt
    #-------------------------------------
    def _get_plot_name(self, var : str) -> str:
        if 'plots_2d' in self._cfg:
            #For 2D plots the name will always be specified in the config
            return var

        if 'name' not in self._cfg['plots'][var]:
            # For 1D plots the name can be taken from variable name itself or specified
            return var

        return self._cfg['plots'][var]['name']
    #-------------------------------------
    def _save_plot(self, var : str) -> Path:
        '''
        Parameters
        -----------------
        var (str) : Name of variable, needed for plot name

        Returns
        -----------------
        Path to plot
        '''
        d_leg = {}
        if 'style' in self._cfg and 'legend' in self._cfg.style:
            d_leg = self._cfg.style.legend

        plt.legend(**d_leg)

        plt_dir = Path(self._cfg.saving.plt_dir)
        plt_dir.mkdir(parents=True, exist_ok=True)

        name = self._get_plot_name(var)

        plot_path = plt_dir / f'{name}.png'
        log.info(f'Saving to: {plot_path}')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(var)

        return plot_path
    #-------------------------------------
    def _data_to_json(self,
                      data : dict[str,float],
                      name : str) -> None:

        # In case the values are numpy objects, which are not JSON
        # serializable
        data = { key : float(value)  for key, value in data.items() }

        plt_dir = self._cfg['saving']['plt_dir']
        os.makedirs(plt_dir, exist_ok=True)

        name      = name.replace(' ', '_')
        json_path = f'{plt_dir}/{name}.json'
        with open(json_path, 'w', encoding='utf-8') as ofile:
            json.dump(data, ofile, indent=2, sort_keys=True)
# --------------------------------------------
