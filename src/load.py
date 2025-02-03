from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import pickle

try:
    import dask.dataframe as dd
except ImportError:
    dd = None  # Dask is optional

def load_df_to_dataframe(data_path: Union[str, Path], 
                         chunk_size: Optional[int] = None, 
                         use_dask: Optional[bool] = False) -> pd.DataFrame:
    """
    Load a DataFrame from a given data path and return it as a pandas DataFrame.
    Supports .csv, .json, .pkl, .parquet, .feather, .hdf, and .pickle formats.
    
    Args:
        data_path (Union[str, Path]): The path to the data file.
        chunk_size (Optional[int], optional): Chunk size for reading the data. Defaults to None.
        use_dask (Optional[bool], optional): Use Dask for reading data if available. Defaults to False.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If data_path is invalid or unsupported format is used.
        FileNotFoundError: If the specified path does not exist.

    Examples:
        >>> df = load_df_to_dataframe('/path/to/data.csv')
    """
    if not data_path:
        raise ValueError("data_path must be a non-empty string or Path.")
    
    target_path = Path(data_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Path '{target_path}' does not exist.")
    
    if use_dask and dd is None:
        raise ImportError("Dask is not installed. Please install it or set use_dask=False.")
    
    suffix = target_path.suffix
    dask_supported_formats = {".csv", ".json", ".parquet", ".hdf"}
    pandas_supported_formats = {".csv", ".json", ".pkl", ".parquet", ".feather", ".hdf", ".pickle"}
    
    if suffix not in pandas_supported_formats:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    if use_dask and suffix not in dask_supported_formats:
        raise ValueError(f"Dask does not support the {suffix} format.")
    
    load_func = {
        ".csv": pd.read_csv,
        ".json": pd.read_json,
        ".pkl": pd.read_pickle,
        ".parquet": pd.read_parquet,
        ".feather": pd.read_feather,
        ".hdf": pd.read_hdf,
        ".pickle": lambda path: pd.DataFrame(pickle.load(open(path, "rb")))
    }
    
    if use_dask:
        dask_load_func = {
            ".csv": dd.read_csv,
            ".json": dd.read_json,
            ".parquet": dd.read_parquet,
            ".hdf": dd.read_hdf,
        }
        df = dask_load_func[suffix](target_path, blocksize=chunk_size).compute()
    else:
        if suffix in {".pkl", ".feather", ".pickle"} and chunk_size:
            raise ValueError(f"Pandas does not support reading {suffix} files in chunks.")
        df = load_func[suffix](target_path, chunksize=chunk_size) if chunk_size else load_func[suffix](target_path)
    
    return df


def load_df_to_dataset(data_path: Union[str, Path], 
                       chunk_size: Optional[int] = None, 
                       use_dask: Optional[bool] = False) -> Dataset:
    """
    Load a DataFrame from a given data path and return it as a Dataset object.
    This function can load data of type .csv, .json, .pkl, .parquet, .feather, .hdf, and .pickle.
    
    Args:
        data_path (str): The path to the data file.
        chunk_size (Optional[int], optional): The size of the chunks to read the data in. Defaults to None.
        use_dask (Optional[bool], optional): Whether to use Dask for reading the data. Defaults to False.        
    
    Returns:
        Dataset: The loaded dataset.
        
    Raises:
        ValueError: If data_path is not a non-empty string.
        FileNotFoundError: If the specified path does not exist.

    Examples:
    >>> dataset = load_df_to_dataset('/path/to/data.csv')
    >>> print(dataset)
    """
    # Validate the input type >
    if data_path is None:
        raise ValueError("data_path must be a non-empty string")
    elif isinstance(data_path, (str, Path)):
        target_path = Path(data_path)
    else:
        raise ValueError("data_path must be a string or pathlib path. Got: {type(data_path)}")
    
    # Validate that the path exists >
    target_path = Path(data_path) 
    if not target_path.exists():
        raise FileNotFoundError(f"Path '{target_path}' does not exist.")
    
    # Initialise >
    dataset = Dataset()
    
    # Load the dataset >
    if target_path.suffix == '.csv':
        if use_dask:
            dataset.data = dd.read_csv(target_path, blocksize=chunk_size)
        else:
            dataset.data = pd.read_csv(target_path, chunksize=chunk_size)
    elif target_path.suffix == '.json':
        if use_dask:
            dataset.data = dd.read_json(target_path, blocksize=chunk_size)
        else:
            dataset.data = pd.read_json(target_path, chunksize=chunk_size)
    elif target_path.suffix == '.pkl':
        if use_dask:
            raise ValueError("Pickle format is not supported by Dask.")
        else:
            if chunk_size:
                raise ValueError("Pandas does not support reading .pkl files in chunks.")
            else:
                dataset.data = pd.read_pickle(target_path)
    elif target_path.suffix == '.parquet':
        if use_dask:
            dataset.data = dd.read_parquet(target_path, engine='pyarrow', blocksize=chunk_size)
        else:
            if chunk_size:
                raise ValueError("Pandas does not support reading parquet files in chunks.")
            else:
                dataset.data = pd.read_parquet(target_path)
    elif target_path.suffix == '.feather':
        if use_dask:
            raise ValueError("Feather format is not supported by Dask.")
        else:
            if chunk_size:
                raise ValueError("Pandas does not support reading feather files in chunks.")
            else:
                dataset.data = pd.read_feather(target_path)
    elif target_path.suffix == '.hdf':
        if use_dask:
            dataset.data = dd.read_hdf(target_path, blocksize=chunk_size)
        else:
            dataset.data = pd.read_hdf(target_path, chunksize=chunk_size)
    elif target_path.suffix == '.pickle':
        if use_dask:
            raise ValueError("Pickle format is not supported by Dask.")
        else: 
            if chunk_size:
                raise ValueError("Pandas does not support reading .pickle files in chunks.")
            else:                    
                data = pd.DataFrame()
                with open(target_path, 'rb') as f:
                    data = pickle.load(f) 
                dataset.data = data
    else:
        raise ValueError(f"Unsupported file format: {target_path.suffix}")
    
    return dataset

