'''
Unit test for Mva class
'''

from dmu.logging.log_store import LogStore
from dmu.ml.train_mva      import TrainMva

import dmu.testing.utilities as ut

log = LogStore.add_logger('dmu:ml:test_train_mva')
# -------------------------------
def test_train():
    '''
    Test training
    '''
    rdf_sig = ut.get_rdf(kind='sig')
    rdf_bkg = ut.get_rdf(kind='bkg')
    cfg     = ut.get_config('ml/tests/train_mva.yaml')

    obj= TrainMva(sig=rdf_sig, bkg=rdf_bkg, cfg=cfg)
    obj.run()
# -------------------------------
def main():
    '''
    Script starts here
    '''
    LogStore.set_level('data_checks:train_mva', 10)

    test_train()
# -------------------------------
if __name__ == '__main__':
    main()
