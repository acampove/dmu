'''
Module containing tests for version management functions
'''
import os
import shutil

import pytest
from dmu.generic.version_management import get_last_version
from dmu.generic.version_management import get_next_version
from dmu.generic.version_management import get_latest_file
from dmu.logging.log_store          import LogStore

log=LogStore.add_logger('dmu:test_version_management')
#-----------------------------------------------------------
class Data:
    '''
    Class used to share attributes
    '''
    user    = os.environ['USER']
    out_dir = f'/tmp/{user}/tests/version_manager/'
# ----------------------
@pytest.fixture(scope='session', autouse=True)
def initialize():
    '''
    This will run before any test
    '''
    LogStore.set_level('dmu:test_version_management', 10)
    LogStore.set_level('dmu:version_management'     , 10)
#-----------------------------------------------------------
def _create_files(nfiles : int) -> str:
    '''
    Helper function that makes files
    '''
    test_dir = f'{Data.out_dir}/latest_file'
    os.makedirs(test_dir, exist_ok=True)

    for ind in range(nfiles):
        file_path = f'{test_dir}/name_v{ind}.txt'
        with open(file_path, 'w', encoding='utf-8') as ofile:
            ofile.close()

    return test_dir
# ----------------------
def _make_dirs(name : str, versions : list[str]) -> str:
    '''
    Parameters
    -------------
    name    : Test name
    versions: Names of versioned directories

    Returns
    -------------
    path to directory containing versioned directories
    '''
    path = f'{Data.out_dir}/{name}'
    shutil.rmtree(path, ignore_errors=True)

    for version in versions:
        versioned_path = f'{path}/{version}'
        os.makedirs(versioned_path, exist_ok=True)

    return path
#-----------------------------------------------------------
def _get_dir(name : str) -> tuple[str,str]:
    # -------------------------------------------
    # This should fail the test
    # -------------------------------------------
    if   name == 'non_numeric':
        l_ver = ['vx', 'vy', 'vz']
    elif name == 'extra_file':
        l_ver = ['some_dir', 'v7p6', 'v7p7']
    elif name == 'collisions':
        l_ver = ['v7.6', 'v7p6', 'v7p7']
    elif name == 'with_commas':
        l_ver = ['v1,3', 'v7,6', 'v7,7']
    # -------------------------------------------
    # This should not fail the test
    # -------------------------------------------
    elif name == 'with_p':
        l_ver = ['v7', 'v7p6', 'v7p7']
    elif name == 'numeric':
        l_ver = ['v1', 'v2', 'v3']
    elif name == 'numeric_period':
        l_ver = ['v1.1', 'v1.2', 'v1.3']
    else:
        raise ValueError(f'Invalid kind of version: {name}')

    path = _make_dirs(name=name, versions=l_ver)

    return path, l_ver[-1]
#-----------------------------------------------------------
@pytest.mark.parametrize('kind', ['with_p', 'numeric', 'numeric_period'])
def test_versioning_formats(kind : str):
    '''
    Tests getting last version for different versioning formats
    '''
    log.info('')
    dir_path, iversion = _get_dir(name=kind)
    oversion           = get_last_version(dir_path=dir_path, version_only=True)

    log.info(f'{kind:<20}{iversion:<10}{oversion:<10}')

    assert iversion == oversion
#-----------------------------------------------------------
@pytest.mark.parametrize('kind', ['non_numeric', 'extra_file', 'collisions', 'with_commas'])
def test_must_fail(kind : str):
    '''
    Check cases where the version finder will raise
    '''
    log.info('')
    dir_path, _ = _get_dir(name=kind)

    with pytest.raises(ValueError):
        get_last_version(dir_path=dir_path, version_only=True)
#-----------------------------------------------------------
def test_files():
    '''
    Testing getting the latest file, from its version
    '''
    file_dir  = _create_files(5)
    last_file = get_latest_file(dir_path = file_dir, wc='name_*.txt')

    assert last_file == f'{Data.out_dir}/latest_file/name_v4.txt'
#-----------------------------------------------------------
def test_next():
    '''
    Tests getting next version
    '''
    assert get_next_version('v1')           == 'v2'
    assert get_next_version('v1.1')         == 'v2.1'
    assert get_next_version('v10.1')        == 'v11.1'

    assert get_next_version('/a/b/c/v1')    == '/a/b/c/v2'
    assert get_next_version('/a/b/c/v1.1')  == '/a/b/c/v2.1'
    assert get_next_version('/a/b/c/v10.1') == '/a/b/c/v11.1'
#-----------------------------------------------------------
