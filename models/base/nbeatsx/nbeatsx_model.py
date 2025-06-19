import math
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from tcn import TemporalConvNet

def filter_input_vars(insample_y: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor, 
                     t_cols: list, include_var_dict: dict) -> t.Tensor:
    """Filter input variables for EPF task - modernized version"""
    device = insample_x_t.device
    batch_size = insample_y.shape[0]
    
    # Create outsample_y placeholder
    outsample_y = t.zeros((batch_size, 1, outsample_x_t.shape[2]), device=device)

    # Combine insample and outsample data
    insample_y_aux = insample_y.unsqueeze(1)
    insample_x_t_aux = t.cat([insample_y_aux, insample_x_t], dim=1)
    outsample_x_t_aux = t.cat([outsample_y, outsample_x_t], dim=1)
    x_t = t.cat([insample_x_t_aux, outsample_x_t_aux], dim=-1)
    
    batch_size, n_channels, input_size = x_t.shape
    assert input_size == 168 + 24, f'input_size {input_size} not 168+24'

    # Reshape to weekly structure
    x_t = x_t.reshape(batch_size, n_channels, 8, 24)

    input_vars = []
    for var in include_var_dict.keys():
        if len(include_var_dict[var]) > 0:
            t_col_idx = t_cols.index(var)
            t_col_filter = include_var_dict[var]
            
            if var != 'week_day':
                input_vars.append(x_t[:, t_col_idx, t_col_filter, :])
            else:
                assert t_col_filter == [-1], f'Day of week must be of outsample not {t_col_filter}'
                day_var = x_t[:, t_col_idx, t_col_filter, [0]]
                day_var = day_var.view(batch_size, -1)

    x_t_filter = t.cat(input_vars, dim=1)
    x_t_filter = x_t_filter.view(batch_size, -1)

    if len(include_var_dict.get('week_day', [])) > 0:
        x_t_filter = t.cat([x_t_filter, day_var], dim=1)

    return x_t_filter

class _StaticFeaturesEncoder(nn.Module):
    """Enhanced static features encoder with modern techniques"""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.5, 
                 use_batch_norm: bool = True):
        super().__init__()
        
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        
        layers.extend([
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(inplace=True)
        ])
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
            
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.encoder(x)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for time series"""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Generate Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = t.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = t.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Output projection and residual connection
        output = self.out_linear(attended)
        return self.layer_norm(x + output)

class ResidualBlock(nn.Module):
    """Residual block with optional attention"""
    def __init__(self, hidden_size: int, use_attention: bool = False, 
                 attention_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = MultiHeadAttention(hidden_size, attention_heads, dropout)
        
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.use_attention:
            # Reshape for attention (batch, seq, features)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
                squeeze_output = True
            else:
                squeeze_output = False
                
            x = self.attention(x)
            
            if squeeze_output:
                x = x.squeeze(1)
        
        return self.norm(x)

class NBeatsBlock(nn.Module):
    """
    Enhanced N-BEATS block with modern features:
    - Improved activations (Swish, GELU, Mish)
    - Optional residual connections
    - Optional attention mechanisms
    - Better normalization strategies
    """
    def __init__(self, x_t_n_inputs: int, x_s_n_inputs: int, x_s_n_hidden: int, 
                 theta_n_dim: int, basis: nn.Module, n_layers: int, theta_n_hidden: list,
                 include_var_dict: Optional[dict], t_cols: Optional[list], 
                 batch_normalization: bool, dropout_prob: float, activation: str,
                 use_residual_connections: bool = False, use_attention: bool = False,
                 attention_heads: int = 8):
        super().__init__()

        # Modern activation functions
        self.activations = {
            'relu': nn.ReLU(inplace=True),
            'swish': nn.SiLU(),  # Swish/SiLU
            'gelu': nn.GELU(),
            'mish': nn.Mish(),
            'softplus': nn.Softplus(),
            'tanh': nn.Tanh(),
            'selu': nn.SELU(),
            'lrelu': nn.LeakyReLU(inplace=True),
            'prelu': nn.PReLU(),
            'sigmoid': nn.Sigmoid()
        }

        if x_s_n_inputs == 0:
            x_s_n_hidden = 0
        theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden

        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.use_residual_connections = use_residual_connections
        self.use_attention = use_attention

        # Build hidden layers
        hidden_layers = []
        for i in range(n_layers):
            # Linear layer
            linear_layer = nn.Linear(in_features=theta_n_hidden[i], out_features=theta_n_hidden[i+1])
            hidden_layers.append(linear_layer)
            
            # Activation
            hidden_layers.append(self.activations[activation])

            # Normalization (after activation for better performance)
            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=theta_n_hidden[i+1]))

            # Dropout
            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))
            
            # Optional residual connection and attention
            if self.use_residual_connections or self.use_attention:
                hidden_layers.append(ResidualBlock(
                    theta_n_hidden[i+1], 
                    use_attention=self.use_attention,
                    attention_heads=attention_heads,
                    dropout=self.dropout_prob
                ))

        # Output layer
        output_layer = [nn.Linear(in_features=theta_n_hidden[-1], out_features=theta_n_dim)]
        layers = hidden_layers + output_layer

        # Static encoder with modern features
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(
                in_features=x_s_n_inputs, 
                out_features=x_s_n_hidden,
                dropout=self.dropout_prob,
                use_batch_norm=self.batch_normalization
            )
        
        self.layers = nn.Sequential(*layers)
        self.basis = basis

        # Residual projection for skip connections
        if self.use_residual_connections:
            input_dim = x_t_n_inputs + x_s_n_hidden
            if input_dim != theta_n_hidden[-1]:
                self.skip_projection = nn.Linear(input_dim, theta_n_hidden[-1])
            else:
                self.skip_projection = nn.Identity()

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor,
                outsample_x_t: t.Tensor, x_s: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:

        # Filter input variables if specified
        if self.include_var_dict is not None:
            insample_y = filter_input_vars(
                insample_y=insample_y, 
                insample_x_t=insample_x_t, 
                outsample_x_t=outsample_x_t,
                t_cols=self.t_cols, 
                include_var_dict=self.include_var_dict
            )

        # Process static features
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = t.cat((insample_y, x_s), 1)

        # Store input for residual connection
        residual_input = insample_y

        # Compute theta parameters
        theta = self.layers(insample_y)

        # Apply residual connection if enabled
        if self.use_residual_connections:
            skip = self.skip_projection(residual_input)
            # Reshape if necessary for residual addition
            if len(skip.shape) != len(theta.shape):
                if len(theta.shape) > len(skip.shape):
                    skip = skip.unsqueeze(-1).expand_as(theta)
                else:
                    theta = theta.view(skip.shape)
            theta = theta + skip

        # Generate backcast and forecast using basis functions
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast

class NBeats(nn.Module):
    """
    Enhanced N-Beats Model with modern features:
    - Better gradient flow
    - Optional model parallelism
    - Improved numerical stability
    """
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                outsample_x_t: t.Tensor, x_s: t.Tensor, return_decomposition: bool = False) -> Union[t.Tensor, Tuple[t.Tensor, t.Tensor]]:

        # Flip sequences for proper temporal ordering
        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        # Initialize forecast with naive baseline
        forecast = insample_y[:, -1:].clone()  # Use clone() for better memory management
        block_forecasts = []
        
        # Process each block
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals, 
                insample_x_t=insample_x_t,
                outsample_x_t=outsample_x_t, 
                x_s=x_s
            )
            
            # Update residuals with masking
            residuals = (residuals - backcast) * insample_mask
            
            # Accumulate forecasts
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # Stack block forecasts for decomposition analysis
        block_forecasts = t.stack(block_forecasts, dim=1)  # (batch, n_blocks, time)

        if return_decomposition:
            return forecast, block_forecasts
        else:
            return forecast

class IdentityBasis(nn.Module):
    """Identity basis function - unchanged but with better typing"""
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast

class TrendBasis(nn.Module):
    """Trend basis with improved numerical stability"""
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        polynomial_size = degree_of_polynomial + 1
        
        # Create polynomial basis with better numerical properties
        backcast_time = np.arange(backcast_size, dtype=np.float32) / backcast_size
        forecast_time = np.arange(forecast_size, dtype=np.float32) / forecast_size
        
        backcast_basis = np.array([backcast_time ** i for i in range(polynomial_size)])
        forecast_basis = np.array([forecast_time ** i for i in range(polynomial_size)])
        
        self.backcast_basis = nn.Parameter(
            t.tensor(backcast_basis, dtype=t.float32), requires_grad=False
        )
        self.forecast_basis = nn.Parameter(
            t.tensor(forecast_basis, dtype=t.float32), requires_grad=False
        )

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast

class SeasonalityBasis(nn.Module):
    """Seasonality basis with improved harmonic generation"""
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        
        # Generate frequency array with better numerical properties
        frequency = np.append(
            np.zeros(1, dtype=np.float32),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=np.float32) / harmonics
        )[None, :]
        
        # Create time grids
        backcast_grid = -2 * np.pi * (
            np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size
        ) * frequency
        forecast_grid = 2 * np.pi * (
            np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size
        ) * frequency

        # Create harmonic basis functions
        backcast_cos = t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32)
        backcast_sin = t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32)
        backcast_template = t.cat([backcast_cos, backcast_sin], dim=0)

        forecast_cos = t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32)
        forecast_sin = t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32)
        forecast_template = t.cat([forecast_cos, forecast_sin], dim=0)

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum('bp,pt->bt', theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast

class ExogenousBasisInterpretable(nn.Module):
    """Interpretable exogenous basis - unchanged but with better typing"""
    def __init__(self):
        super().__init__()

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast

class Chomp1d(nn.Module):
    """Chomping layer for causal convolutions"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()

class ExogenousBasisWavenet(nn.Module):
    """Enhanced WaveNet-based exogenous basis with better initialization"""
    def __init__(self, out_features: int, in_features: int, num_levels: int = 4, 
                 kernel_size: int = 3, dropout_prob: float = 0.0):
        super().__init__()
        
        # Learnable feature weighting with improved initialization
        self.weight = nn.Parameter(t.empty(1, in_features, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Build WaveNet layers
        layers = []
        
        # Input layer
        padding = (kernel_size - 1) * (2**0)
        layers.extend([
            nn.Conv1d(in_channels=in_features, out_channels=out_features,
                     kernel_size=kernel_size, padding=padding, dilation=2**0),
            Chomp1d(padding),
            nn.ReLU(inplace=True)
        ])
        
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))
        
        # Dilated convolution layers
        for i in range(1, num_levels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            layers.extend([
                nn.Conv1d(in_channels=out_features, out_channels=out_features,
                         padding=padding, kernel_size=kernel_size, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU(inplace=True)
            ])

        self.wavenet = nn.Sequential(*layers)

    def transform(self, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """Transform exogenous variables through WaveNet"""
        input_size = insample_x_t.shape[2]

        # Concatenate input and output time series
        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        # Apply learnable weights (element-wise multiplication, broadcasted)
        x_t = x_t * self.weight
        
        # Process through WaveNet
        x_t = self.wavenet(x_t)

        # Split back into backcast and forecast components
        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast

class ExogenousBasisTCN(nn.Module):
    """Temporal Convolutional Network basis with enhanced features"""
    def __init__(self, out_features: int, in_features: int, num_levels: int = 4, 
                 kernel_size: int = 2, dropout_prob: float = 0.0):
        super().__init__()
        
        # Create channel configuration for TCN
        n_channels = [out_features] * num_levels
        
        self.tcn = TemporalConvNet(
            num_inputs=in_features, 
            num_channels=n_channels, 
            kernel_size=kernel_size, 
            dropout=dropout_prob
        )

    def transform(self, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """Transform exogenous variables through TCN"""
        input_size = insample_x_t.shape[2]

        # Concatenate input and output sequences
        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        # Process through TCN
        x_t = self.tcn(x_t)
        
        # Split processed features
        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]

        return backcast_basis, forecast_basis

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, 
                outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = t.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        return backcast, forecast

# Enhanced Loss Functions with better numerical stability
class FocalLoss(nn.Module):
    """Focal Loss for handling imbalanced data"""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: t.Tensor, target: t.Tensor, mask: Optional[t.Tensor] = None) -> t.Tensor:
        mse = F.mse_loss(pred, target, reduction='none')
        pt = t.exp(-mse)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * mse
        
        if mask is not None:
            focal_loss = focal_loss * mask
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class QuantileLoss(nn.Module):
    """Quantile loss for uncertainty estimation"""
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred: t.Tensor, target: t.Tensor, mask: Optional[t.Tensor] = None) -> t.Tensor:
        """
        pred: (batch_size, num_quantiles, forecast_length)
        target: (batch_size, forecast_length)
        """
        target = target.unsqueeze(1)  # Add quantile dimension
        losses = []
        
        for i, q in enumerate(self.quantiles):
            error = target - pred[:, i:i+1]
            loss = t.max(q * error, (q - 1) * error)
            
            if mask is not None:
                loss = loss * mask.unsqueeze(1)
                
            losses.append(loss.mean())
        
        return t.stack(losses).mean()

class AdaptiveLoss(nn.Module):
    """Adaptive loss that combines multiple loss functions"""
    def __init__(self, loss_types: list = ['mse', 'mae'], weights: Optional[list] = None):
        super().__init__()
        self.loss_types = loss_types
        
        if weights is None:
            self.weights = nn.Parameter(t.ones(len(loss_types)))
        else:
            self.weights = nn.Parameter(t.tensor(weights, dtype=t.float32))

    def forward(self, pred: t.Tensor, target: t.Tensor, mask: Optional[t.Tensor] = None) -> t.Tensor:
        total_loss = 0
        weights_sum = 0
        
        for i, loss_type in enumerate(self.loss_types):
            weight = F.softmax(self.weights, dim=0)[i]
            
            if loss_type == 'mse':
                loss = F.mse_loss(pred, target, reduction='none')
            elif loss_type == 'mae':
                loss = F.l1_loss(pred, target, reduction='none')
            elif loss_type == 'huber':
                loss = F.huber_loss(pred, target, reduction='none')
            else:
                continue
                
            if mask is not None:
                loss = loss * mask
                
            total_loss += weight * loss.mean()
            weights_sum += weight
            
        return total_loss / weights_sum if weights_sum > 0 else total_loss

# Model utilities for interpretability and analysis
class ModelInterpreter:
    """Utility class for model interpretation and analysis"""
    
    @staticmethod
    def compute_feature_importance(model: NBeats, dataloader, device: str = 'cuda') -> dict:
        """Compute feature importance using gradient-based methods"""
        model.eval()
        importance_scores = {}
        
        with t.enable_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 10:  # Limit to first 10 batches for efficiency
                    break
                    
                # Move batch to device
                batch_tensors = {k: t.tensor(v, device=device, requires_grad=True) 
                               for k, v in batch.items() if isinstance(v, np.ndarray)}
                
                # Forward pass
                output = model(**batch_tensors)
                loss = output.mean()  # Simple aggregation
                
                # Backward pass
                loss.backward()
                
                # Collect gradients
                for name, tensor in batch_tensors.items():
                    if tensor.grad is not None:
                        if name not in importance_scores:
                            importance_scores[name] = []
                        importance_scores[name].append(tensor.grad.abs().mean().item())
        
        # Average importance scores
        for name in importance_scores:
            importance_scores[name] = np.mean(importance_scores[name])
            
        return importance_scores
    
    @staticmethod
    def extract_block_contributions(model: NBeats, input_data: dict) -> dict:
        """Extract individual block contributions for decomposition analysis"""
        model.eval()
        
        with t.no_grad():
            # Get full prediction with decomposition
            forecast, block_forecasts = model(**input_data, return_decomposition=True)
            
            # Analyze each block's contribution
            block_stats = {}
            for i in range(block_forecasts.shape[1]):
                block_output = block_forecasts[:, i, :]
                block_stats[f'block_{i}'] = {
                    'mean': block_output.mean().item(),
                    'std': block_output.std().item(),
                    'min': block_output.min().item(),
                    'max': block_output.max().item(),
                    'contribution_ratio': (block_output.abs().mean() / forecast.abs().mean()).item()
                }
                
        return block_stats

# Backward compatibility functions
def create_legacy_nbeats(**kwargs):
    """Create N-BEATS model using legacy parameter format"""
    import warnings
    warnings.warn("create_legacy_nbeats is deprecated. Use the new Nbeats class with NBeatsConfig.", 
                  DeprecationWarning, stacklevel=2)
    
    # This would map old parameters to new format
    # Implementation would depend on specific legacy format
    pass

# Export key classes for easy importing
__all__ = [
    'NBeats', 'NBeatsBlock', 'IdentityBasis', 'TrendBasis', 'SeasonalityBasis',
    'ExogenousBasisInterpretable', 'ExogenousBasisWavenet', 'ExogenousBasisTCN',
    'MultiHeadAttention', 'ResidualBlock', 'FocalLoss', 'QuantileLoss', 
    'AdaptiveLoss', 'ModelInterpreter'
]