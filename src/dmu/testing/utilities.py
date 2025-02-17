'''
Module containing utility functions needed by unit tests
'''
import os
from typing              import Union
from dataclasses         import dataclass
from importlib.resources import files

from ROOT import RDF, TFile, RDataFrame

import pandas as pnd
import numpy
import yaml

from dmu.logging.log_store import LogStore

log = LogStore.add_logger('dmu:testing:utilities')
# -------------------------------
@dataclass
class Data:
    '''
    Class storing shared data
    '''
# -------------------------------
def _double_data(df_1 : pnd.DataFrame) -> pnd.DataFrame:
    df_2   = df_1.copy()
    df     = pnd.concat([df_1, df_2], axis=0)

    return df
# -------------------------------
def _add_nans(df_good : pnd.DataFrame) -> pnd.DataFrame:
    log.debug('Adding NaNs')
    df_bad    = df_good.copy()
    df_bad[:] = numpy.nan

    df        = pnd.concat([df_good, df_bad])
    df        = df.reset_index()

    return df
# -------------------------------
def get_rdf(kind : Union[str,None] = None,
            repeated : bool        = False,
            nentries : int         = 3_000,
            add_nans : bool        = False):
    '''
    Return ROOT dataframe with toy data
    '''
    d_data = {}
    if   kind == 'sig':
        d_data['w'] = numpy.random.normal(0, 1, size=nentries)
        d_data['x'] = numpy.random.normal(0, 1, size=nentries)
        d_data['y'] = numpy.random.normal(0, 1, size=nentries)
        d_data['z'] = numpy.random.normal(0, 1, size=nentries)
    elif kind == 'bkg':
        d_data['w'] = numpy.random.normal(1, 1, size=nentries)
        d_data['x'] = numpy.random.normal(1, 1, size=nentries)
        d_data['y'] = numpy.random.normal(1, 1, size=nentries)
        d_data['z'] = numpy.random.normal(1, 1, size=nentries)
    else:
        log.error(f'Invalid kind: {kind}')
        raise ValueError

    df = pnd.DataFrame(d_data)

    if repeated:
        df = _double_data(df)

    if add_nans:
        df = _add_nans(df)

    rdf = RDF.FromPandas(df)

    return rdf
# -------------------------------
def get_config(name : Union[str,None] = None):
    '''
    Takes path to the YAML config file, after `dmu_data`
    Returns dictionary with config
    '''
    if name is None:
        raise ValueError('Name not pased')

    cfg_path = files('dmu_data').joinpath(name)
    cfg_path = str(cfg_path)
    with open(cfg_path, encoding='utf-8') as ifile:
        d_cfg = yaml.safe_load(ifile)

    return d_cfg
# -------------------------------
def _get_rdf(nentries : int) -> RDataFrame:
    rdf = RDataFrame(nentries)
    rdf = rdf.Define('x', '0')
    rdf = rdf.Define('y', '1')
    rdf = rdf.Define('z', '2')

    return rdf
# -------------------------------
def get_file_with_trees(path : str) -> TFile:
    '''
    Picks full path to toy ROOT file, in the form of /a/b/c/file.root
    returns handle to it
    '''
    dir_name    = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)

    snap        = RDF.RSnapshotOptions()
    snap.fMode  = 'recreate'

    l_tree_name = ['odir/idir/a', 'dir/b', 'c']
    l_nevt      = [    100, 200, 300]

    l_rdf = [ _get_rdf(nevt) for nevt in l_nevt ]
    for rdf, tree_name in zip(l_rdf, l_tree_name):
        rdf.Snapshot(tree_name, path, ['x', 'y', 'z'], snap)
        snap.fMode  = 'update'

    return TFile(path)
