import math
import numpy as np
import torch as t
import torch.nn as nn
from typing import Tuple, Optional, Dict, List, Any
from tcn import TemporalConvNet


def filter_input_vars(insample_y: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor, 
                     t_cols: List[str], include_var_dict: Dict[str, List[int]]) -> t.Tensor:
    """
    Filter input variables based on include_var_dict specification.
    This function is specific for the EPF task.
    
    Parameters
    ----------
    insample_y : torch.Tensor
        Insample target values
    insample_x_t : torch.Tensor
        Insample exogenous variables
    outsample_x_t : torch.Tensor
        Outsample exogenous variables
    t_cols : List[str]
        List of column names
    include_var_dict : Dict[str, List[int]]
        Dictionary specifying which variables to include
        
    Returns
    -------
    torch.Tensor
        Filtered input tensor
    """
    # Get device from input tensor
    device = insample_x_t.device
    
    outsample_y = t.zeros((insample_y.shape[0], 1, outsample_x_t.shape[2]), device=device)

    insample_y_aux = t.unsqueeze(insample_y, dim=1)

    insample_x_t_aux = t.cat([insample_y_aux, insample_x_t], dim=1)
    outsample_x_t_aux = t.cat([outsample_y, outsample_x_t], dim=1)
    x_t = t.cat([insample_x_t_aux, outsample_x_t_aux], dim=-1)
    batch_size, n_channels, input_size = x_t.shape

    if input_size != 168 + 24:
        raise ValueError(f'Expected input_size to be 192 (168+24), got {input_size}')

    x_t = x_t.reshape(batch_size, n_channels, 8, 24)

    input_vars = []
    for var in include_var_dict.keys():
        if len(include_var_dict[var]) > 0:
            if var not in t_cols:
                raise ValueError(f"Variable '{var}' not found in t_cols")
            
            t_col_idx = t_cols.index(var)
            t_col_filter = include_var_dict[var]
            
            if var != 'week_day':
                input_vars.append(x_t[:, t_col_idx, t_col_filter, :])
            else:
                if t_col_filter != [-1]:
                    raise ValueError(f'Day of week must be of outsample, got {t_col_filter}')
                day_var = x_t[:, t_col_idx, t_col_filter, [0]]
                day_var = day_var.view(batch_size, -1)

    x_t_filter = t.cat(input_vars, dim=1)
    x_t_filter = x_t_filter.view(batch_size, -1)

    if len(include_var_dict['week_day']) > 0:
        x_t_filter = t.cat([x_t_filter, day_var], dim=1)

    return x_t_filter


class _StaticFeaturesEncoder(nn.Module):
    """
    Static features encoder for processing static exogenous variables.
    
    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    """
    
    def __init__(self, in_features: int, out_features: int):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU()
        ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass through static encoder."""
        x = self.encoder(x)
        return x


class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    
    Parameters
    ----------
    x_t_n_inputs : int
        Number of temporal input features
    x_s_n_inputs : int
        Number of static input features  
    x_s_n_hidden : int
        Number of hidden units for static features
    theta_n_dim : int
        Dimension of theta parameters
    basis : nn.Module
        Basis function module
    n_layers : int
        Number of hidden layers
    theta_n_hidden : List[int]
        List of hidden layer sizes
    include_var_dict : Dict[str, List[int]]
        Dictionary of variables to include
    t_cols : List[str]
        List of column names
    batch_normalization : bool
        Whether to use batch normalization
    dropout_prob : float
        Dropout probability
    activation : str
        Activation function name
    """
    
    def __init__(self, x_t_n_inputs: int, x_s_n_inputs: int, x_s_n_hidden: int, 
                 theta_n_dim: int, basis: nn.Module, n_layers: int, 
                 theta_n_hidden: List[int], include_var_dict: Optional[Dict[str, List[int]]], 
                 t_cols: List[str], batch_normalization: bool, dropout_prob: float, 
                 activation: str):
        super().__init__()

        if x_s_n_inputs == 0:
            x_s_n_hidden = 0
        theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden

        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        
        # Define activation functions
        self.activations = {
            'relu': nn.ReLU(),
            'softplus': nn.Softplus(),
            'tanh': nn.Tanh(),
            'selu': nn.SELU(),
            'lrelu': nn.LeakyReLU(),
            'prelu': nn.PReLU(),
            'sigmoid': nn.Sigmoid()
        }
        
        if activation not in self.activations:
            raise ValueError(f"Activation '{activation}' not supported. Available: {list(self.activations.keys())}")

        hidden_layers = []
        for i in range(n_layers):
            # Add linear layer
            hidden_layers.append(
                nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i+1])
            )
            hidden_layers.append(self.activations[activation])

            # Batch norm after activation
            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=theta_n_hidden[i+1]))

            # Dropout
            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        # Output layer
        output_layer = [nn.Linear(in_features=theta_n_hidden[-1], out_features=theta_n_dim)]
        layers = hidden_layers + output_layer

        # Static encoder setup
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(
                in_features=x_s_n_inputs, 
                out_features=x_s_n_hidden
            )
        else:
            self.static_encoder = None
            
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor,
                outsample_x_t: t.Tensor, x_s: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """
        Forward pass through N-BEATS block.
        
        Parameters
        ----------
        insample_y : torch.Tensor
            Insample target values
        insample_x_t : torch.Tensor
            Insample exogenous variables
        outsample_x_t : torch.Tensor
            Outsample exogenous variables
        x_s : torch.Tensor
            Static variables
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Backcast and forecast tensors
        """
        if self.include_var_dict is not None:
            insample_y = filter_input_vars(
                insample_y=insample_y, 
                insample_x_t=insample_x_t, 
                outsample_x_t=outsample_x_t,
                t_cols=self.t_cols, 
                include_var_dict=self.include_var_dict
            )

        # Static exogenous processing
        if self.static_encoder is not None:
            x_s = self.static_encoder(x_s)
            insample_y = t.cat((insample_y, x_s), 1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast


class NBeats(nn.Module):
    """
    N-Beats Model.
    
    Parameters
    ----------
    blocks : nn.ModuleList
        List of N-BEATS blocks
    """
    
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                outsample_x_t: t.Tensor, x_s: t.Tensor, return_decomposition: bool = False) -> t.Tensor:
        """
        Forward pass through N-BEATS model.
        
        Parameters
        ----------
        insample_y : torch.Tensor
            Insample target values
        insample_x_t : torch.Tensor
            Insample exogenous variables
        insample_mask : torch.Tensor
            Insample mask
        outsample_x_t : torch.Tensor
            Outsample exogenous variables
        x_s : torch.Tensor
            Static variables
        return_decomposition : bool
            Whether to return block-wise decomposition
            
        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Forecast, and optionally block forecasts
        """
        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:]  # Level with Naive1
        block_forecasts = []
        
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals, 
                insample_x_t=insample_x_t,
                outsample_x_t=outsample_x_t, 
                x_s=x_s
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        if return_decomposition:
            # (n_batch, n_blocks, n_time)
            block_forecasts = t.stack(block_forecasts)
            block_forecasts = block_forecasts.permute(1, 0, 2)
            return forecast, block_forecasts
        else:
            return forecast


class IdentityBasis(nn.Module):
    """
    Identity basis function for N-BEATS.
    
    Parameters
    ----------
    backcast_size : int
        Size of backcast
    forecast_size : int
        Size of forecast
    """
    
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """Forward pass through identity basis."""
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast


class TrendBasis(nn.Module):
    """
    Trend basis function for N-BEATS.
    
    Parameters
    ----------
    degree_of_polynomial : int
        Degree of polynomial for trend
    backcast_size : int
        Size of backcast
    forecast_size : int
        Size of forecast
    """
    
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        polynomial_size = degree_of_polynomial + 1
        
        # Create backcast basis using modern tensor creation
        backcast_basis_np = np.concatenate([
            np.power(np.arange(backcast_size, dtype=np.float32) / backcast_size, i)[None, :]
            for i in range(polynomial_size)
        ])
        self.backcast_basis = nn.Parameter(
            t.from_numpy(backcast_basis_np).float(), requires_grad=False
        )
        
        # Create forecast basis using modern tensor creation
        forecast_basis_np = np.concatenate([
            np.power(np.arange(forecast_size, dtype=np.float32) / forecast_size, i)[None, :]
            for i in range(polynomial_size)
        ])
        self.forecast_basis = nn.Parameter(
            t.from_numpy(forecast_basis_np).float(), requires_grad=False
        )

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """Forward pass through trend basis."""
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """
    Seasonality basis function for N-BEATS.
    
    Parameters
    ----------
    harmonics : int
        Number of harmonics
    backcast_size : int
        Size of backcast
    forecast_size : int
        Size of forecast
    """
    
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        frequency = np.append(
            np.zeros(1, dtype=np.float32),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=np.float32) / harmonics
        )[None, :]
        
        backcast_grid = -2 * np.pi * (
            np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size
        ) * frequency
        forecast_grid = 2 * np.pi * (
            np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size
        ) * frequency

        # Create backcast templates using modern tensor creation
        backcast_cos_template = t.from_numpy(np.transpose(np.cos(backcast_grid))).float()
        backcast_sin_template = t.from_numpy(np.transpose(np.sin(backcast_grid))).float()
        backcast_template = t.cat([backcast_cos_template, backcast_sin_template], dim=0)

        # Create forecast templates using modern tensor creation
        forecast_cos_template = t.from_numpy(np.transpose(np.cos(forecast_grid))).float()
        forecast_sin_template = t.from_numpy(np.transpose(np.sin(forecast_grid))).float()
        forecast_template = t.cat([forecast_cos_template, forecast_sin_template], dim=0)

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """Forward pass through seasonality basis."""
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


class ExogenousBasisInterpretable(nn.Module):
    """Interpretable exogenous basis function for N-BEATS."""
    
    def __init__(self):
        super().__init__()

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """Forward pass through interpretable exogenous basis."""
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast


class Chomp1d(nn.Module):
    """Chomp1d layer for causal convolutions."""
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Remove last chomp_size elements to maintain causality."""
        return x[:, :, :-self.chomp_size].contiguous()


class ExogenousBasisWavenet(nn.Module):
    """
    WaveNet-based exogenous basis function for N-BEATS.
    
    Parameters
    ----------
    out_features : int
        Number of output features
    in_features : int
        Number of input features
    num_levels : int
        Number of dilation levels
    kernel_size : int
        Kernel size for convolutions
    dropout_prob : float
        Dropout probability
    """
    
    def __init__(self, out_features: int, in_features: int, num_levels: int = 4, 
                 kernel_size: int = 3, dropout_prob: float = 0):
        super().__init__()
        
        # Learnable weight parameter for input scaling
        self.weight = nn.Parameter(t.Tensor(1, in_features, 1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0.5))

        # Build WaveNet layers
        layers = []
        
        # Input layer
        padding = (kernel_size - 1) * (2**0)
        layers.extend([
            nn.Conv1d(in_channels=in_features, out_channels=out_features,
                     kernel_size=kernel_size, padding=padding, dilation=2**0),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        ])
        
        # Dilated convolution layers
        for i in range(1, num_levels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            layers.extend([
                nn.Conv1d(in_channels=out_features, out_channels=out_features,
                         padding=padding, kernel_size=kernel_size, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU()
            ])

        self.wavenet = nn.Sequential(*layers)

    def transform(self, insample_x_t: t.Tensor, 
                  outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """Transform input through WaveNet."""
        input_size = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        # Element-wise multiplication, broadcasted on b and t
        x_t = x_t * self.weight
        x_t = self.wavenet(x_t)

        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """Forward pass through WaveNet exogenous basis."""
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast


class ExogenousBasisTCN(nn.Module):
    """
    TCN-based exogenous basis function for N-BEATS.
    
    Parameters
    ----------
    out_features : int
        Number of output features
    in_features : int
        Number of input features
    num_levels : int
        Number of TCN levels
    kernel_size : int
        Kernel size for convolutions
    dropout_prob : float
        Dropout probability
    """
    
    def __init__(self, out_features: int, in_features: int, num_levels: int = 4, 
                 kernel_size: int = 2, dropout_prob: float = 0):
        super().__init__()
        n_channels = num_levels * [out_features]
        self.tcn = TemporalConvNet(
            num_inputs=in_features, 
            num_channels=n_channels, 
            kernel_size=kernel_size, 
            dropout=dropout_prob
        )

    def transform(self, insample_x_t: t.Tensor, 
                  outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """Transform input through TCN."""
        input_size = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)
        x_t = self.tcn(x_t)
        
        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """Forward pass through TCN exogenous basis."""
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast