'''
Unit test for Mva class
'''
import pytest
import mplhep
import matplotlib.pyplot as plt

from   dmu.logging.log_store import LogStore
from   dmu.ml.train_mva      import TrainMva, NoFeatureInfo
import dmu.testing.utilities as ut

log = LogStore.add_logger('dmu:ml:test_train_mva')

# -------------------------------
@pytest.fixture(scope='session', autouse=True)
def _initialize():
    LogStore.set_level('dmu:ml:train_mva', 10)
    plt.style.use(mplhep.style.LHCb2)
# -------------------------------
@pytest.mark.parametrize('nfold', [3, 10])
def test_simple(nfold : int):
    '''
    Test a simple training
    '''
    rdf_sig = ut.get_rdf(kind='sig')
    rdf_bkg = ut.get_rdf(kind='bkg')
    cfg     = ut.get_config('ml/tests/train_mva.yaml')
    cfg['training']['nfold'] = nfold
    path    = cfg['saving']['output']
    cfg['saving']['output'] = path.replace('train_mva', f'train_mva_{nfold:03}')

    obj= TrainMva(sig=rdf_sig, bkg=rdf_bkg, cfg=cfg)
    obj.run()
# -------------------------------
def test_preffix():
    '''
    Tests using prefixes and suffixes in feature names, e.g.:

    preffix.x.suffix
    '''
    rdf_sig = ut.get_rdf(kind='sig', use_preffix=True)
    rdf_bkg = ut.get_rdf(kind='bkg', use_preffix=True)
    cfg     = ut.get_config('ml/tests/train_mva_with_preffix.yaml')

    obj= TrainMva(sig=rdf_sig, bkg=rdf_bkg, cfg=cfg)
    obj.run()
# -------------------------------
def test_missing_feature_plot():
    '''
    Tests that a feature's plot is missing,
    thus the label corresponding to the feature's name cannot be accessed
    '''
    nfold   = 2

    rdf_sig = ut.get_rdf(kind='sig')
    rdf_bkg = ut.get_rdf(kind='bkg')
    cfg     = ut.get_config('ml/tests/train_mva.yaml')

    del cfg['plotting']['features']['plots']['y']

    cfg['training']['nfold'] = nfold
    path    = cfg['saving']['output']
    cfg['saving']['output'] = path.replace('train_mva', 'missing_feature_plot')

    obj= TrainMva(sig=rdf_sig, bkg=rdf_bkg, cfg=cfg)
    with pytest.raises(NoFeatureInfo):
        obj.run()
# -------------------------------
def test_repeated():
    '''
    Tests training when inputs have repeated samples
    '''
    rdf_sig = ut.get_rdf(kind='sig', repeated=True)
    rdf_bkg = ut.get_rdf(kind='bkg')
    cfg     = ut.get_config('ml/tests/train_mva.yaml')

    obj= TrainMva(sig=rdf_sig, bkg=rdf_bkg, cfg=cfg)
    obj.run()
# -------------------------------
def test_nans():
    '''
    Tests training when inputs have NaNs
    '''
    rdf_sig = ut.get_rdf(kind='sig', columns_with_nans=['x', 'y'])
    rdf_bkg = ut.get_rdf(kind='bkg')
    cfg     = ut.get_config('ml/tests/train_mva.yaml')

    obj= TrainMva(sig=rdf_sig, bkg=rdf_bkg, cfg=cfg)
    obj.run()
# -------------------------------
def test_with_diagnostics():
    '''
    Will add diagnostics plots, for correlations, etc
    '''
    rdf_sig = ut.get_rdf(kind='sig')
    rdf_bkg = ut.get_rdf(kind='bkg')
    cfg     = ut.get_config('ml/tests/train_mva_with_diagnostics.yaml')

    obj= TrainMva(sig=rdf_sig, bkg=rdf_bkg, cfg=cfg)
    obj.run()
# -------------------------------
def test_only_diagnostics():
    '''
    Will only run diagnostics with already trained models
    '''
    rdf_sig = ut.get_rdf(kind='sig')
    rdf_bkg = ut.get_rdf(kind='bkg')
    cfg     = ut.get_config('ml/tests/train_mva_with_diagnostics.yaml')

    obj= TrainMva(sig=rdf_sig, bkg=rdf_bkg, cfg=cfg)
    obj.run(load_trained=True)
# -------------------------------
def test_with_workers():
    '''
    Test a simple training
    '''
    nfold = 10
    nwork =  4

    rdf_sig = ut.get_rdf(kind='sig')
    rdf_bkg = ut.get_rdf(kind='bkg')
    cfg     = ut.get_config('ml/tests/train_mva.yaml')
    cfg['training']['nfold'] = nfold
    path    = cfg['saving']['output']
    cfg['saving']['output'] = path.replace('train_mva', f'train_mva_{nfold:03}')

    obj= TrainMva(sig=rdf_sig, bkg=rdf_bkg, cfg=cfg)
    obj.run(workers=nwork)
# -------------------------------
