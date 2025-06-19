import numpy as np
import pandas as pd
import random
import torch as t
from typing import Optional, List, Tuple, Dict, Any

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class TimeSeriesDataset(Dataset):
    """
    Time Series Dataset object for N-BEATSx implementation.
    
    Parameters
    ----------
    Y_df : pd.DataFrame
        DataFrame with target variable. Must contain columns ['unique_id', 'ds', 'y']
    X_df : pd.DataFrame, optional
        DataFrame with temporal exogenous variables. Must contain columns ['unique_id', 'ds']
    S_df : pd.DataFrame, optional
        DataFrame with static exogenous variables. Must contain columns ['unique_id', 'ds']
    f_cols : List[str], optional
        Name of columns which future exogenous variables (eg. forecasts)
    ts_train_mask : List[int], optional
        Must have length equal to longest time series. Specifies train-test split. 1s for train, 0s for test.
    """
    
    def __init__(self,
                 Y_df: pd.DataFrame,
                 X_df: Optional[pd.DataFrame] = None,
                 S_df: Optional[pd.DataFrame] = None,
                 f_cols: Optional[List[str]] = None,
                 ts_train_mask: Optional[List[int]] = None):
        
        # Validate input DataFrames
        if not isinstance(Y_df, pd.DataFrame):
            raise TypeError("Y_df must be a pandas DataFrame")
        
        required_y_cols = ['unique_id', 'ds', 'y']
        if not all(col in Y_df.columns for col in required_y_cols):
            raise ValueError(f"Y_df must contain columns {required_y_cols}")
        
        if X_df is not None:
            if not isinstance(X_df, pd.DataFrame):
                raise TypeError("X_df must be a pandas DataFrame or None")
            required_x_cols = ['unique_id', 'ds']
            if not all(col in X_df.columns for col in required_x_cols):
                raise ValueError(f"X_df must contain columns {required_x_cols}")

        print('Processing dataframes ...')
        # Pandas dataframes to data lists
        ts_data, s_data, self.meta_data, self.t_cols, self.X_cols = self._df_to_lists(
            Y_df=Y_df, S_df=S_df, X_df=X_df
        )

        # Dataset attributes
        self.n_series = len(ts_data)
        self.max_len = max([len(ts['y']) for ts in ts_data])
        self.n_channels = len(self.t_cols)  # y, X_cols, insample_mask and outsample_mask
        self.frequency = pd.infer_freq(Y_df.head()['ds'])
        self.f_cols = f_cols if f_cols is not None else []

        # Number of X and S features
        self.n_x = 0 if X_df is None else len(self.X_cols)
        self.n_s = 0 if S_df is None else S_df.shape[1] - 1  # -1 for unique_id

        print('Creating ts tensor ...')
        # Balances panel and creates
        # numpy s_matrix of shape (n_series, n_s)
        # numpy ts_tensor of shape (n_series, n_channels, max_len) n_channels = y + X_cols + masks
        self.ts_tensor, self.s_matrix, self.len_series = self._create_tensor(ts_data, s_data)
        
        if ts_train_mask is None:
            ts_train_mask = np.ones(self.max_len)
        
        if len(ts_train_mask) != self.max_len:
            raise ValueError(f'ts_train_mask must have length {self.max_len}, got {len(ts_train_mask)}')

        self._declare_outsample_train_mask(ts_train_mask)

    def _df_to_lists(self, Y_df: pd.DataFrame, S_df: Optional[pd.DataFrame], 
                    X_df: Optional[pd.DataFrame]) -> Tuple[List[Dict], List[Dict], List[Dict], List[str], List[str]]:
        """
        Convert DataFrames to lists for internal processing.
        
        Returns
        -------
        Tuple containing ts_data, s_data, meta_data, t_cols, X_cols
        """
        unique_ids = Y_df['unique_id'].unique()

        if X_df is not None:
            X_cols = [col for col in X_df.columns if col not in ['unique_id', 'ds']]
        else:
            X_cols = []

        if S_df is not None:
            S_cols = [col for col in S_df.columns if col not in ['unique_id']]
        else:
            S_cols = []

        ts_data = []
        s_data = []
        meta_data = []
        
        for i, u_id in enumerate(unique_ids):
            # Fix deprecated np.asscalar() calls
            top_row = Y_df['unique_id'].searchsorted(u_id, 'left').item()
            bottom_row = Y_df['unique_id'].searchsorted(u_id, 'right').item()
            
            serie = Y_df[top_row:bottom_row]['y'].values
            last_ds_i = Y_df[top_row:bottom_row]['ds'].max()

            # Y values
            ts_data_i = {'y': serie}

            # X values
            if X_df is not None:
                for X_col in X_cols:
                    serie = X_df[top_row:bottom_row][X_col].values
                    ts_data_i[X_col] = serie
            ts_data.append(ts_data_i)

            # S values
            s_data_i = defaultdict(list)
            if S_df is not None:
                for S_col in S_cols:
                    s_data_i[S_col] = S_df.loc[S_df['unique_id'] == u_id, S_col].values
            s_data.append(s_data_i)

            # Metadata
            meta_data_i = {'unique_id': u_id, 'last_ds': last_ds_i}
            meta_data.append(meta_data_i)

        t_cols = ['y'] + X_cols + ['insample_mask', 'outsample_mask']
        
        return ts_data, s_data, meta_data, t_cols, X_cols

    def _create_tensor(self, ts_data: List[Dict], s_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create tensor representations of the time series data.
        
        Returns
        -------
        Tuple containing ts_tensor, s_matrix, len_series
        """
        s_matrix = np.zeros((self.n_series, self.n_s), dtype=np.float32)
        ts_tensor = np.zeros((self.n_series, self.n_channels, self.max_len), dtype=np.float32)

        len_series = []
        for idx in range(self.n_series):
            ts_idx = np.array(list(ts_data[idx].values()), dtype=np.float32)
            
            # Fill tensor with time series data
            ts_tensor[idx, :self.t_cols.index('insample_mask'), -ts_idx.shape[1]:] = ts_idx
            ts_tensor[idx, self.t_cols.index('insample_mask'), -ts_idx.shape[1]:] = 1.0

            # To avoid sampling windows without inputs available to predict we shift -1
            # outsample_mask will be completed with the train_mask, this ensures available data
            ts_tensor[idx, self.t_cols.index('outsample_mask'), -(ts_idx.shape[1]):] = 1.0
            
            if self.n_s > 0:
                s_matrix[idx, :] = list(s_data[idx].values())
            len_series.append(ts_idx.shape[1])

        return ts_tensor, s_matrix, np.array(len_series, dtype=np.int32)

    def _declare_outsample_train_mask(self, ts_train_mask: List[int]) -> None:
        """Update attribute and ts_tensor with train mask."""
        self.ts_train_mask = np.array(ts_train_mask, dtype=np.float32)

    def get_meta_data_col(self, col: str) -> List[Any]:
        """
        Get specific column from metadata.
        
        Parameters
        ----------
        col : str
            Column name to retrieve
            
        Returns
        -------
        List of values for the specified column
        """
        if col not in self.meta_data[0]:
            raise KeyError(f"Column '{col}' not found in metadata")
        return [x[col] for x in self.meta_data]

    def get_filtered_ts_tensor(self, offset: int, output_size: int, 
                              window_sampling_limit: int, 
                              ts_idxs: Optional[List[int]] = None) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Get filtered tensor for training/validation.
        
        Parameters
        ----------
        offset : int
            Time offset for filtering
        output_size : int
            Size of output horizon
        window_sampling_limit : int
            Maximum window size for sampling
        ts_idxs : List[int], optional
            Specific time series indices to include
            
        Returns
        -------
        Tuple containing filtered_ts_tensor, right_padding, ts_train_mask
        """
        last_outsample_ds = self.max_len - offset + output_size
        first_ds = max(last_outsample_ds - window_sampling_limit - output_size, 0)
        
        if ts_idxs is None:
            filtered_ts_tensor = self.ts_tensor[:, :, first_ds:last_outsample_ds]
        else:
            filtered_ts_tensor = self.ts_tensor[ts_idxs, :, first_ds:last_outsample_ds]
            
        right_padding = max(last_outsample_ds - self.max_len, 0)  # Pad with zeros if needed
        ts_train_mask = self.ts_train_mask[first_ds:last_outsample_ds]

        # Validate tensor
        nan_count = np.sum(np.isnan(filtered_ts_tensor))
        if nan_count > 0:
            raise ValueError(f'Filtered tensor has {nan_count} NaN values')
            
        return filtered_ts_tensor, right_padding, ts_train_mask

    def get_f_idxs(self, cols: List[str]) -> List[int]:
        """
        Get indices for future columns.
        
        Parameters
        ----------
        cols : List[str]
            Column names to get indices for
            
        Returns
        -------
        List of indices for the specified columns
        """
        missing_cols = [col for col in cols if col not in self.f_cols]
        if missing_cols:
            raise ValueError(f'Columns {missing_cols} are not available in f_cols')
        
        f_idxs = [self.X_cols.index(col) for col in cols]
        return f_idxs