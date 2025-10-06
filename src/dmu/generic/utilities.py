'''
Module containing generic utility functions
'''
import os
import glob
from pathlib import Path
import time
import json
import pickle
import inspect
from importlib.resources   import files
from importlib.util        import find_spec
from typing                import Callable, Any, cast
from functools             import wraps
from contextlib            import contextmanager

import importlib.util
import yaml
from yaml.constructor      import ConstructorError
from omegaconf.errors      import ConfigKeyError
from omegaconf             import ListConfig, OmegaConf, DictConfig, ValidationError
from dmu.logging.log_store import LogStore

TIMER_ON=False
log = LogStore.add_logger('dmu:generic:utilities')
# --------------------------------
class ConfigValidation:
    '''
    Class used to hold config validation data
    '''
    # If True, it will require:
    # - Existence of schema
    # - That validation passes
    # Or else an exception will be risen
    enforce     = False          
    schema_name = 'ConfigSchema' # When validating OmegaConf configs, this will be the top level class name of the schema
# --------------------------------
class BlockStyleDumper(yaml.SafeDumper):
    '''
    Class needed to specify proper indentation when
    dumping data to YAML files
    '''
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)
# ------------------------------------
def object_to_string(obj : Any, sort_keys=False) -> str:
    '''
    Parameters
    -------------------
    obj: Python object, e.g. dictionary, it can contain the following objects:
    - Path (pathlib) 
    - DictConfig (omegacong)
    sort_keys: If true will sort the keys of all the dictionaries. Not recommended for mixed type keys

    Returns
    -------------------
    JSON string
    '''
    def default_encoder(x):
        if isinstance(x, DictConfig):
            return OmegaConf.to_container(cfg=x, resolve=True)

        if isinstance(x, Path):
            return str(x)

        log.info(x)
        raise TypeError(f"Unserializable type: {type(x)}")

    try:
        json_string = json.dumps(obj, sort_keys=sort_keys, default=default_encoder)
    except Exception as exc:
        raise RuntimeError(f'Cannot covert to JSON string: {obj}') from exc

    return json_string
# ---------------------------------
@contextmanager
def enforce_schema_validation(value : bool):
    '''
    Parameters
    -------------
    value:  If True it will raise ValidationError if validation of config fails because of:
            - Schema file not found
            - Problems with the config
    '''
    old_val = ConfigValidation.enforce 
    ConfigValidation.enforce = value
    try:
        yield
    finally:
        ConfigValidation.enforce =old_val
# ---------------------------------
def load_data(package : str, fpath : str) -> Any:
    '''
    This function will load a YAML or JSON file from a data package

    Parameters
    ---------------------
    package: Data package, e.g. `dmu_data`
    path   : Path to YAML/JSON file, relative to the data package

    Returns
    ---------------------
    Dictionary or whatever structure the file is holding
    '''

    cpath = files(package).joinpath(fpath)
    cpath = str(cpath)
    data  = load_json(cpath)

    return data
# --------------------------------
# --------------------------------
def _load_conf_from_wcard(
    package       : str, 
    fwcard        : str, 
    resolve_paths : bool) -> list[DictConfig]:
    '''
    This function will load a YAML or JSON files from a data package
    and a wildcard

    Parameters
    ---------------------
    package: Data package, e.g. `dmu_data`
    fwcard : Wildcard representing path with respect to package 
    resolve_paths: When the config is too complex, instead of including it
                   in the main YAML file, one includes the path to another
                   YAML file. Those paths will be replaced by the actual config
                   when this flag is True

    Returns
    ---------------------
    List of DictConfig class from the OmegaConf package
    '''
    wcard = files(package).joinpath(fwcard)
    wcard = str(wcard)
    l_path= glob.glob(wcard)

    if not l_path:
        raise ValueError(f'No files found in: {wcard}')

    l_fpath = []
    for path in l_path:
        [_, fpath] = path.split(package)
        l_fpath.append(fpath[1:])

    l_cfg = []
    for fpath in l_fpath: 
        cfg = _load_conf_from_path(package=package, fpath=fpath, resolve_paths=resolve_paths)
        l_cfg.append(cfg)

    return l_cfg
# --------------------------------
def _load_conf_from_path(
    package       : str,
    fpath         : str,
    resolve_paths : bool) -> DictConfig:
    '''
    This function will load a YAML or JSON file from a data package

    Parameters
    ---------------------
    package: Data package, e.g. `dmu_data`
    fpath  : Path to YAML/JSON file, relative to the data package
    resolve_paths: When the config is too complex, instead of including it
                   in the main YAML file, one includes the path to another
                   YAML file. Those paths will be replaced by the actual config
                   when this flag is True

    Returns
    ---------------------
    DictConfig class from the OmegaConf package
    '''
    log.debug(f'Loading: {package}:{fpath}')

    cpath = files(package).joinpath(fpath)
    cpath = cast(str, cpath)

    log.debug(f'Loading configuration from: {cpath}')

    try:
        cfg = OmegaConf.load(cpath)
    except ConstructorError as exc:
        raise ConstructorError(f'Cannot load {cpath}') from exc

    if not isinstance(cfg, DictConfig):
        raise ValueError(f'Object in {cpath} is not a dictionary')

    _validate_schema(cfg=cfg, package=package, fpath=fpath)

    if not resolve_paths:
        return cfg

    return _resolve_sub_configs(cfg=cfg, package=package)
# ----------------------
def _validate_schema(
    cfg     : DictConfig,
    package : str, 
    fpath   : str) -> None:
    '''
    Parameters
    -------------
    cfg    : Config to be validated
    package: Name of data package where config to be validated lives
    fpath  : Relative (to package) path to config file
    '''
    spath   = _get_schema_path(package=package, fpath=fpath)
    if spath is None: 
        log.debug(f'No schema found in: {spath}')
        return

    spec  = importlib.util.spec_from_file_location('placeholder', spath)
    if spec is None:
        raise ValueError(f'Cannot load spec from: {spath}')

    smodule=importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ValueError('Loader not found')

    loader.exec_module(smodule)

    if not hasattr(smodule, ConfigValidation.schema_name):
        raise AttributeError(f'Cannot find {ConfigValidation.schema_name} in module {spath}')

    ConfigSchema = getattr(smodule, ConfigValidation.schema_name)
    schema = OmegaConf.structured(ConfigSchema)

    try:
        cfg_val = OmegaConf.merge(schema, cfg)
        OmegaConf.to_object(cfg_val)
    except (ValidationError, ConfigKeyError) as exc:
        if ConfigValidation.enforce:
            raise RuntimeError(f'Failed to validate {package}/{fpath}') from exc
# ----------------------
def _get_schema_path(package : str, fpath : str) -> str|None:
    '''
    Parameters
    -------------
    package: Name of data package with configs
    fpath  : Path to config relative to `package`

    Returns
    -------------
    Either:

    - Path to schema file, a python module
    - None, if no package was found and schema validation is not enforced 
    '''
    package = package.removesuffix('_data')
    package = f'{package}_schema'
    sname   = os.path.basename(fpath).replace('.yaml', '_config.py')

    if not find_spec(package) and ConfigValidation.enforce:
        raise FileNotFoundError(f'Cannot find schema package: {package}')

    if not find_spec(package):
        return None

    spath = files(package).joinpath(sname)
    spath = str(spath)

    if not os.path.isfile(spath) and ConfigValidation.enforce:
        raise FileNotFoundError('Missing config schema')

    if not os.path.isfile(spath):
        return None

    return spath
# ----------------------
def _resolve_sub_configs(cfg : DictConfig, package : str) -> DictConfig:
    '''
    Parameters
    -------------
    cfg    : Configuration dictionary
    package: Name of data package where secondary configurations will live

    Returns
    -------------
    Input dictionary where yaml paths have been replaced by actual configuration
    '''
    for key, val in cfg.items():
        if isinstance(val, DictConfig):
            _resolve_sub_configs(cfg=val, package=package)
            continue

        if not isinstance(val, str):
            continue

        if not val.endswith(('.yaml','.yml')):
            continue

        log.debug(f'Resolving sub-config: {val}')
        # Do not enforce validation for sub-configs
        # They might not have a schema associated
        # And their validation should be done explicitly
        # I.e. only validate top level configs
        with enforce_schema_validation(value=False):
            cfg[key] = load_conf(package=package, fpath=val)

    return cfg
# --------------------------------
def _get_module_name( fun : Callable) -> str:
    mod = inspect.getmodule(fun)
    if mod is None:
        raise ValueError(f'Cannot determine module name for function: {fun}')

    return mod.__name__
# --------------------------------
def timeit(f):
    '''
    Decorator used to time functions, it is turned off by default, can be turned on with:

    from dmu.generic.utilities import TIMER_ON
    from dmu.generic.utilities import timeit

    TIMER_ON=True

    @timeit
    def fun():
        ...
    '''
    @wraps(f)
    def wrap(*args, **kw):
        if not TIMER_ON:
            result = f(*args, **kw)
            return result

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        mod_nam = _get_module_name(f)
        fun_nam = f.__name__
        log.info(f'{mod_nam}.py:{fun_nam}; Time: {te-ts:.3f}s')

        return result
    return wrap
# --------------------------------
def dump_json(
    data      : dict|str|list|set|tuple|DictConfig|ListConfig,
    path      : str|Path,
    exists_ok : bool = False,
    sort_keys : bool = False) -> None:
    '''
    Saves data as JSON or YAML, depending on the extension, supported .json, .yaml, .yml

    Parameters
    ----------------
    data     : dictionary, list, etc
    path     : Path to output file where to save it
    exists_ok: If False (default) will raise RunTimeError if file already found
    sort_keys: Will set sort_keys argument of json.dump function
    '''
    if isinstance(data, (DictConfig, ListConfig)):
        py_data = OmegaConf.to_container(data)
    else:
        py_data = data

    path = _prepare_path(path=path, exists_ok=exists_ok)
    with open(path, 'w', encoding='utf-8') as ofile:
        # These lines should deal with objects like Path or DictConfig
        # in py_data
        json_string = object_to_string(obj=py_data)
        data        = json.loads(json_string)

        if path.name.endswith('.json'):
            json.dump(data, ofile, indent=4, sort_keys=sort_keys)
            return

        if path.name.endswith('.yaml') or path.name.endswith('.yml'):
            yaml.dump(data, ofile, Dumper=BlockStyleDumper, sort_keys=sort_keys)
            return

        raise NotImplementedError(f'Cannot deduce format from extension in path: {path}')
# --------------------------------
def dump_text(
    lines     : list[str]|ListConfig,
    path      : str|Path,
    exists_ok : bool=False,
    sort_lines: bool=False) -> None:
    '''
    Parameters
    ----------------
    lines     : List or ListConfig with lines to write 
    path      : Path to output file where to save it
    exists_ok : If False (default) will raise RunTimeError if file already found
    sort_lines: Will sort lines before saving 
    '''
    path = _prepare_path(path=path, exists_ok=exists_ok)

    if sort_lines:
        lines = sorted(lines)

    data = '\n'.join(lines)
    with open(path, 'w', encoding='utf-8') as ofile:
        ofile.write(data)
# ----------------------
def _prepare_path(path : Path|str, exists_ok : bool) -> Path:
    '''
    Parameters
    -------------
    path: Path to file to text file to produce
    exists_ok: If False and path exists, will raise exception

    Returns
    -------------
    Path to file to be made
    '''
    if isinstance(path, str):
        path = Path(path)

    if path.exists() and not exists_ok:
        raise FileExistsError(f'Path {path} already found')

    path.parent.mkdir(exist_ok=True, parents=True)

    return path
# --------------------------------
def load_json(path : str|Path):
    '''
    Loads data from JSON or YAML, depending on extension of files, supported .json, .yaml, .yml

    Parameters
    path     : Path to outut file where data is saved
    '''
    if isinstance(path, str):
        path = Path(path)

    if not path.is_file:
        raise FileNotFoundError(f'Cannot find: {path}')

    with open(path, encoding='utf-8') as ofile:
        if path.name.endswith('.json'):
            data = json.load(ofile)
            return data

        if path.name.endswith('.yaml') or path.name.endswith('.yml'):
            data = yaml.safe_load(ofile)
            return data

        raise NotImplementedError(f'Cannot deduce format from extension in path: {path}')
# --------------------------------
def dump_pickle(data, path : str) -> None:
    '''
    Saves data as pickle file

    Parameters
    data     : dictionary, list, etc
    path     : Path to output file where to save it
    '''
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)

    with open(path, 'wb') as ofile:
        pickle.dump(data, ofile)
# --------------------------------
def load_pickle(path : str) -> None:
    '''
    loads data file

    Parameters
    path     : Path to output file where to save it
    '''
    with open(path, 'rb') as ofile:
        data = pickle.load(ofile)

    return data
# --------------------------------
@contextmanager
def silent_import():
    '''
    In charge of suppressing messages
    of imported modules
    '''
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
# --------------------------------
