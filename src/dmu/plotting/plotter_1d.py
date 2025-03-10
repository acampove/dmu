'''
Module containing plotter class
'''

from hist import Hist

import numpy
import matplotlib.pyplot as plt

from dmu.logging.log_store import LogStore
from dmu.plotting.plotter  import Plotter
from dmu.plotting.fwhm     import FWHM

log = LogStore.add_logger('dmu:plotting:Plotter1D')
# --------------------------------------------
class Plotter1D(Plotter):
    '''
    Class used to plot columns in ROOT dataframes
    '''
    # --------------------------------------------
    def __init__(self, d_rdf=None, cfg=None):
        '''
        Parameters:

        d_rdf (dict): Dictionary mapping the kind of sample with the ROOT dataframe
        cfg   (dict): Dictionary with configuration, e.g. binning, ranges, etc
        '''

        super().__init__(d_rdf=d_rdf, cfg=cfg)
    #-------------------------------------
    def _get_labels(self, var : str) -> tuple[str,str]:
        if 'labels' not in self._d_cfg['plots'][var]:
            return var, 'Entries'

        xname, yname = self._d_cfg['plots'][var]['labels' ]

        return xname, yname
    #-------------------------------------
    def _is_normalized(self, var : str) -> bool:
        d_cfg     = self._d_cfg['plots'][var]
        normalized=False
        if 'normalized' in d_cfg:
            normalized = d_cfg['normalized']

        return normalized
    #-------------------------------------
    def _get_binning(self, var : str, d_data : dict[str, numpy.ndarray]) -> tuple[float, float, int]:
        d_cfg  = self._d_cfg['plots'][var]
        minx, maxx, bins = d_cfg['binning']
        if maxx <= minx + 1e-5:
            log.info(f'Bounds not set for {var}, will calculated them')
            minx, maxx = self._find_bounds(d_data = d_data, qnt=minx)
            log.info(f'Using bounds [{minx:.3e}, {maxx:.3e}]')
        else:
            log.debug(f'Using bounds [{minx:.3e}, {maxx:.3e}]')

        return minx, maxx, bins
    #-------------------------------------
    def _run_plugins(self,
                     arr_val : numpy.ndarray,
                     arr_wgt : numpy.ndarray,
                     hst,
                     name    : str) -> None:
        if 'plugin' not in self._d_cfg:
            return

        d_plugin = self._d_cfg['plugin']
        if 'fwhm' in d_plugin:
            arr_bin_cnt = hst.values()
            maxy = numpy.max(arr_bin_cnt)
            cfg  = d_plugin['fwhm']
            obj  = FWHM(cfg=cfg, val=arr_val, wgt=arr_wgt, maxy=maxy)
            fwhm = obj.run()

            form        = cfg['format']
            this_title  = form.format(fwhm)

            if 'add_std' in cfg and cfg['add_std']:
                mu         = numpy.average(arr_val            , weights=arr_wgt)
                avg        = numpy.average((arr_val - mu) ** 2, weights=arr_wgt)
                std        = numpy.sqrt(avg)
                form       = form.replace('FWHM', 'STD')
                this_title+= '; ' + form.format(std)

            self._title+= f'\n{name}: {this_title}'
    #-------------------------------------
    def _plot_var(self, var : str) -> float:
        '''
        Will plot a variable from a dictionary of dataframes
        Parameters
        --------------------
        var   (str)  : name of column

        Return
        --------------------
        Largest bin content among all bins and among all histograms plotted
        '''
        # pylint: disable=too-many-locals

        d_data = {}
        for name, rdf in self._d_rdf.items():
            log.debug(f'Plotting: {var}/{name}')
            d_data[name] = rdf.AsNumpy([var])[var]

        minx, maxx, bins = self._get_binning(var, d_data)
        d_wgt            = self._get_weights(var)

        l_bc_all = []
        for name, arr_val in d_data.items():
            label        = self._label_from_name(name, arr_val)
            arr_wgt      = d_wgt[name] if d_wgt is not None else numpy.ones_like(arr_val)
            arr_wgt      = self._normalize_weights(arr_wgt, var)
            hst          = Hist.new.Reg(bins=bins, start=minx, stop=maxx, name='x').Weight()
            hst.fill(x=arr_val, weight=arr_wgt)
            self._run_plugins(arr_val, arr_wgt, hst, name)
            hst.plot(label=label, histtype='errorbar', marker='.', linestyle='none')
            l_bc_all    += hst.values().tolist()

        max_y = max(l_bc_all)

        return max_y
    # --------------------------------------------
    def _label_from_name(self, name : str, arr_val : numpy.ndarray) -> str:
        if 'stats' not in self._d_cfg:
            return name

        d_stat = self._d_cfg['stats']
        if 'nentries' not in d_stat:
            return name

        form = d_stat['nentries']

        nentries = len(arr_val)
        nentries = form.format(nentries)

        return f'{name}{nentries}'
    # --------------------------------------------
    def _normalize_weights(self, arr_wgt : numpy.ndarray, var : str) -> numpy.ndarray:
        cfg_var = self._d_cfg['plots'][var]
        if 'normalized' not in cfg_var:
            log.debug(f'Not normalizing for variable: {var}')
            return arr_wgt

        if not cfg_var['normalized']:
            log.debug(f'Not normalizing for variable: {var}')
            return arr_wgt

        log.debug(f'Normalizing for variable: {var}')
        total   = numpy.sum(arr_wgt)
        arr_wgt = arr_wgt / total

        return arr_wgt
    # --------------------------------------------
    def _style_plot(self, var : str, max_y : float) -> None:
        d_cfg  = self._d_cfg['plots'][var]
        yscale = d_cfg['yscale' ] if 'yscale' in d_cfg else 'linear'

        xname, yname = self._get_labels(var)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.yscale(yscale)
        if yscale == 'linear':
            plt.ylim(bottom=0)

        title = self._title
        if 'title'      in d_cfg:
            this_title = d_cfg['title']
            title += f'\n {this_title}'

        title = title.lstrip('\n')

        plt.ylim(top=1.2 * max_y)
        plt.legend()
        plt.title(title)
    # --------------------------------------------
    def _plot_lines(self, var : str) -> None:
        '''
        Will plot vertical lines for some variables

        var (str) : name of variable
        '''
        if 'style' in self._d_cfg and 'skip_lines' in self._d_cfg['style'] and self._d_cfg['style']['skip_lines']:
            return

        if var in ['B_const_mass_M', 'B_M']:
            plt.axvline(x=5280, color='r', label=r'$B^+$'   , linestyle=':')
        elif var == 'Jpsi_M':
            plt.axvline(x=3096, color='r', label=r'$J/\psi$', linestyle=':')
    # --------------------------------------------
    def run(self):
        '''
        Will run plotting
        '''

        fig_size = self._get_fig_size()
        for var in self._d_cfg['plots']:
            plt.figure(var, figsize=fig_size)
            max_y = self._plot_var(var)
            self._style_plot(var, max_y)
            self._plot_lines(var)
            self._save_plot(var)
# --------------------------------------------
