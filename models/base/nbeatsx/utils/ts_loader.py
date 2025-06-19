import numpy as np
import pandas as pd
import random
import torch as t
import copy
from typing import Optional, List, Dict, Any, Iterator
from ts_dataset import TimeSeriesDataset
from collections import defaultdict


class TimeSeriesLoader(object):
    """
    Time Series Loader object, used to sample time series from TimeSeriesDataset object.
    
    Parameters
    ----------
    ts_dataset : TimeSeriesDataset
        Time Series Dataset object which contains data in PyTorch tensors optimized for sampling.
    model : str
        Model which will use the loader, affects the way of constructing batches. Currently supports 'nbeats'.
    offset : int
        Equivalent to timestamps in test (data in test will not be sampled). It is used to filter
        the PyTorch tensor containing the time series, to avoid using the future during training.
    window_sampling_limit : int
        Equivalent to calibration window. Length of the history (prior to offset) which will be sampled
    input_size : int
        Size of inputs of each window (only for NBEATS), eg. 7 days
    output_size : int
        Forecasting horizon
    idx_to_sample_freq : int
        Frequency of sampling. Eg: 1 for data_augmentation, 24 for sampling only at 12:00am
    batch_size : int
        Number of batches (windows) to sample
    is_train_loader : bool
        True: will only sample time stamps with 1s in mask, False: will only sample time stamps with 0s in mask
    shuffle : bool
        Indicates if windows should be shuffled. True is used for training and False for predicting.
    """
    
    def __init__(self,
                 ts_dataset: TimeSeriesDataset,
                 model: str,
                 offset: int,
                 window_sampling_limit: int,
                 input_size: int,
                 output_size: int,
                 idx_to_sample_freq: int,
                 batch_size: int,
                 is_train_loader: bool,
                 shuffle: bool):
        
        # Validate inputs
        if not isinstance(ts_dataset, TimeSeriesDataset):
            raise TypeError("ts_dataset must be a TimeSeriesDataset instance")
        
        supported_models = ['nbeats']
        if model not in supported_models:
            raise ValueError(f"Model '{model}' not supported. Supported models: {supported_models}")
        
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")

        # Dataloader attributes
        self.model = model
        self.window_sampling_limit = window_sampling_limit
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.idx_to_sample_freq = idx_to_sample_freq
        self.offset = offset
        self.ts_dataset = ts_dataset
        self.t_cols = self.ts_dataset.t_cols
        self.is_train_loader = is_train_loader  # Boolean variable for train and validation mask
        self.shuffle = shuffle  # Boolean to shuffle data, useful for validation

        # Create rolling window matrix in advance for faster access to data and broadcasted s_matrix
        self._create_train_data()

    def _update_sampling_windows_idxs(self) -> List[int]:
        """
        Update sampling window indices based on available data masks.
        
        Returns
        -------
        List of valid sampling indices
        """
        # Only sample during training windows with at least one active output mask and input mask
        outsample_condition = t.sum(
            self.ts_windows[:, self.t_cols.index('outsample_mask'), -self.output_size:], 
            axis=1
        )
        insample_condition = t.sum(
            self.ts_windows[:, self.t_cols.index('insample_mask'), :self.input_size], 
            axis=1
        )
        
        # Element-wise product to find valid windows
        sampling_idx = t.nonzero(outsample_condition * insample_condition > 0)
        sampling_idx = list(sampling_idx.flatten().numpy())
        
        return sampling_idx

    def _create_windows_tensor(self) -> tuple[t.Tensor, np.ndarray]:
        """
        Create rolling windows tensor for efficient data access.
        
        Returns
        -------
        Tuple containing windows tensor and s_matrix
        """
        # Memory efficiency is gained from keeping across dataloaders common ts_tensor in dataset
        # Filter function is used to define train tensor and validation tensor with the offset
        tensor, right_padding, train_mask = self.ts_dataset.get_filtered_ts_tensor(
            offset=self.offset, 
            output_size=self.output_size,
            window_sampling_limit=self.window_sampling_limit
        )
        
        # Convert to PyTorch tensors with proper dtype
        tensor = t.from_numpy(tensor).float()
        train_mask = t.from_numpy(train_mask).float()

        # Outsample mask checks existence of values in ts, train_mask is used to filter out validation
        # is_train_loader inverts the train_mask in case the dataloader is in validation mode
        mask = train_mask if self.is_train_loader else (1 - train_mask)
        tensor[:, self.t_cols.index('outsample_mask'), :] = (
            tensor[:, self.t_cols.index('outsample_mask'), :] * mask
        )

        # Pad tensor appropriately
        padder = t.nn.ConstantPad1d(padding=(self.input_size, right_padding), value=0.0)
        tensor = padder(tensor)

        # Last output_size outsample_mask and y to 0 (ensure no validation leakage)
        tensor[:, self.t_cols.index('y'), -self.output_size:] = 0.0
        tensor[:, self.t_cols.index('outsample_mask'), -self.output_size:] = 0.0

        # Creating rolling windows and 'flattening' them
        windows = tensor.unfold(
            dimension=-1, 
            size=self.input_size + self.output_size, 
            step=self.idx_to_sample_freq
        )
        
        # Reshape: n_serie, n_channel, n_time, window_size -> n_serie, n_time, n_channel, window_size
        windows = windows.permute(0, 2, 1, 3)
        windows = windows.reshape(-1, self.ts_dataset.n_channels, self.input_size + self.output_size)

        # Broadcast s_matrix: This works because unfold in windows_tensor orders: time, serie
        n_windows = len(windows)
        n_series = self.ts_dataset.n_series
        repeat_factor = n_windows // n_series
        
        s_matrix = np.tile(self.ts_dataset.s_matrix, (repeat_factor, 1))
        
        # Handle remainder if not evenly divisible
        remainder = n_windows % n_series
        if remainder > 0:
            s_matrix = np.vstack([s_matrix, self.ts_dataset.s_matrix[:remainder]])

        return windows, s_matrix

    def __len__(self) -> int:
        """Return number of available sampling windows."""
        return len(self.windows_sampling_idx)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over batches of data."""
        if self.shuffle:
            sample_idxs = np.random.choice(
                a=self.windows_sampling_idx,
                size=len(self.windows_sampling_idx), 
                replace=False
            )
        else:
            sample_idxs = self.windows_sampling_idx

        if len(sample_idxs) == 0:
            raise ValueError('No valid sampling indices found. Check the data and masks.')

        # Calculate number of batches
        n_batches = int(np.ceil(len(sample_idxs) / self.batch_size))

        for idx in range(n_batches):
            start_idx = idx * self.batch_size
            end_idx = (idx + 1) * self.batch_size
            ws_idxs = sample_idxs[start_idx:end_idx]
            batch = self.__get_item__(index=ws_idxs)
            yield batch

    def __get_item__(self, index: List[int]) -> Dict[str, np.ndarray]:
        """Get batch item based on model type."""
        if self.model == 'nbeats':
            return self._nbeats_batch(index)
        else:
            raise NotImplementedError(f"Model '{self.model}' batch creation not implemented")

    def _nbeats_batch(self, index: List[int]) -> Dict[str, np.ndarray]:
        """
        Create batch for N-BEATS model.
        
        Parameters
        ----------
        index : List[int]
            Indices of windows to include in batch
            
        Returns
        -------
        Dictionary containing batch data
        """
        # Access precomputed rolling window matrix (RAM intensive)
        windows = self.ts_windows[index]
        s_matrix = self.s_matrix[index]

        # Extract different components of the batch
        insample_y = windows[:, self.t_cols.index('y'), :self.input_size]
        insample_x = windows[:, 
                            (self.t_cols.index('y') + 1):self.t_cols.index('insample_mask'), 
                            :self.input_size]
        insample_mask = windows[:, self.t_cols.index('insample_mask'), :self.input_size]

        outsample_y = windows[:, self.t_cols.index('y'), self.input_size:]
        outsample_x = windows[:, 
                             (self.t_cols.index('y') + 1):self.t_cols.index('insample_mask'), 
                             self.input_size:]
        outsample_mask = windows[:, self.t_cols.index('outsample_mask'), self.input_size:]

        batch = {
            's_matrix': s_matrix,
            'insample_y': insample_y.numpy(), 
            'insample_x': insample_x.numpy(), 
            'insample_mask': insample_mask.numpy(),
            'outsample_y': outsample_y.numpy(), 
            'outsample_x': outsample_x.numpy(), 
            'outsample_mask': outsample_mask.numpy()
        }
        return batch

    def _create_train_data(self) -> None:
        """Create rolling window matrix for fast information retrieval."""
        self.ts_windows, self.s_matrix = self._create_windows_tensor()
        self.n_windows = len(self.ts_windows)
        self.windows_sampling_idx = self._update_sampling_windows_idxs()

    def update_offset(self, offset: int) -> None:
        """
        Update offset and recreate training data if changed.
        
        Parameters
        ----------
        offset : int
            New offset value
        """
        if offset == self.offset:
            return  # Avoid extra computation
        self.offset = offset
        self._create_train_data()

    def get_meta_data_col(self, col: str) -> List[Any]:
        """Get metadata column from underlying dataset."""
        return self.ts_dataset.get_meta_data_col(col)

    def get_n_variables(self) -> tuple[int, int]:
        """Get number of temporal and static variables."""
        return self.ts_dataset.n_x, self.ts_dataset.n_s

    def get_n_series(self) -> int:
        """Get number of time series."""
        return self.ts_dataset.n_series

    def get_max_len(self) -> int:
        """Get maximum length of time series."""
        return self.ts_dataset.max_len

    def get_n_channels(self) -> int:
        """Get number of channels in tensor."""
        return self.ts_dataset.n_channels

    def get_X_cols(self) -> List[str]:
        """Get list of exogenous variable column names."""
        return self.ts_dataset.X_cols

    def get_frequency(self) -> Optional[str]:
        """Get inferred frequency of time series."""
        return self.ts_dataset.frequency