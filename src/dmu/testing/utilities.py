'''
Module containing utility functions needed by unit tests
'''
import os
import math
import glob
from typing              import Union
from dataclasses         import dataclass
from importlib.resources import files

from ROOT import RDF, TFile, RDataFrame

import uproot
import joblib
import pandas as pnd
import numpy
import yaml

from dmu.ml.train_mva      import TrainMva
from dmu.ml.cv_classifier  import CVClassifier
from dmu.logging.log_store import LogStore

log = LogStore.add_logger('dmu:testing:utilities')
# -------------------------------
@dataclass
class Data:
    '''
    Class storing shared data
    '''
    out_dir = '/tmp/tests/dmu/ml/cv_predict'

    d_col   = {
            'main' : ['index', 'a0', 'b0'],
            'frn1' : ['index', 'a1', 'b1'],
            'frn2' : ['index', 'a2', 'b2'],
            'frn3' : ['index', 'a3', 'b3'],
            'frn4' : ['index', 'a4', 'b4'],
            }
# -------------------------------
def _double_data(df_1 : pnd.DataFrame) -> pnd.DataFrame:
    df_2   = df_1.copy()
    df     = pnd.concat([df_1, df_2], axis=0)

    return df
# -------------------------------
def _add_nans(df : pnd.DataFrame, columns : list[str]) -> pnd.DataFrame:
    size = len(df) * 0.2
    size = math.floor(size)

    l_col = df.columns.tolist()
    if columns is None:
        l_col_index = range(len(l_col))
    else:
        l_col_index = [ l_col.index(column) for column in columns ]

    log.debug(f'Replacing randomly with {size} NaNs')
    for _ in range(size):
        irow = numpy.random.randint(0, df.shape[0])      # Random row index
        icol = numpy.random.choice(l_col_index)      # Random column index

        df.iat[irow, icol] = numpy.nan

    return df
# -------------------------------
def get_rdf(
        kind              : Union[str,None] = None,
        repeated          : bool        = False,
        nentries          : int         = 3_000,
        use_preffix       : bool        = False,
        columns_with_nans : list[str]   = None):
    '''
    Return ROOT dataframe with toy data
    '''
    # Needed for a specific test
    xnm = 'preffix.x.suffix' if use_preffix else 'x'

    d_data = {}
    if   kind == 'sig':
        d_data[xnm] = numpy.random.normal(0, 1, size=nentries)
        d_data['w'] = numpy.random.normal(0, 1, size=nentries)
        d_data['y'] = numpy.random.normal(0, 1, size=nentries)
        d_data['z'] = numpy.random.normal(0, 1, size=nentries)
    elif kind == 'bkg':
        d_data[xnm] = numpy.random.normal(1, 1, size=nentries)
        d_data['w'] = numpy.random.normal(1, 1, size=nentries)
        d_data['y'] = numpy.random.normal(1, 1, size=nentries)
        d_data['z'] = numpy.random.normal(1, 1, size=nentries)
    else:
        log.error(f'Invalid kind: {kind}')
        raise ValueError

    df = pnd.DataFrame(d_data)

    if repeated:
        df = _double_data(df)

    if columns_with_nans is not None:
        df = _add_nans(df, columns=columns_with_nans)

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
# -------------------------------
def get_models(rdf_sig : RDataFrame, rdf_bkg : RDataFrame) -> list[CVClassifier]:
    '''
    Will train and return models
    '''

    cfg                     = get_config('ml/tests/train_mva.yaml')
    cfg['saving']['output'] = Data.out_dir

    obj= TrainMva(sig=rdf_sig, bkg=rdf_bkg, cfg=cfg)
    obj.run()

    pkl_wc     = f'{Data.out_dir}/model*.pkl'
    l_pkl_path = glob.glob(pkl_wc)
    l_model    = [ joblib.load(pkl_path) for pkl_path in l_pkl_path ]

    return l_model
# -------------------------------
def _make_file(
        fpath    : str,
        tree     : str,
        nentries : int) -> None:

    fdir       = os.path.dirname(fpath)
    sample     = os.path.basename(fdir)
    l_col_name = Data.d_col[sample]
    data       = {}
    for col_name in l_col_name:
        if col_name == 'index':
            data[col_name] = numpy.arange(nentries)
            continue

        data[col_name] = numpy.random.normal(0, 1, nentries)

    with uproot.recreate(fpath) as ofile:
        log.debug(f'Saving to: {fpath}:{tree}')
        ofile[tree] = data
# -------------------------------
def build_friend_structure(file_name : str, nentries : int) -> None:
    '''
    Will load YAML file with file structure needed to
    test code that relies on friend trees, e.g. DDFGetter

    Parameters:
    -------------------
    file_name (str): Name of YAML file with wanted structure, e.g. friends.yaml
    nentries (int) : Number of entries in file
    '''
    cfg_path = files('dmu_data').joinpath(f'rfile/{file_name}')
    with open(cfg_path, encoding='utf=8') as ifile:
        data = yaml.safe_load(ifile)

    if 'tree' not in data:
        raise ValueError('tree entry missing in: {cfg_path}')

    tree_name = data['tree']

    if 'samples' not in data:
        raise ValueError('Samples section missing in: {cfg_path}')

    if 'files' not in data:
        raise ValueError('Files section missing in: {cfg_path}')

    for fdir in data['samples']:
        for fname in data['files']:
            path = f'{fdir}/{fname}'
            _make_file(fpath=path, tree=tree_name, nentries=nentries)
# ----------------------------------------------
