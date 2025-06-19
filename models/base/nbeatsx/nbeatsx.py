import os
import time
import numpy as np
import pandas as pd
import random
import gc
import copy
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass
from collections import defaultdict

import torch as t
from torch import optim, Tensor
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from functools import partial

from nbeatsx_model import NBeats, NBeatsBlock, IdentityBasis, TrendBasis, SeasonalityBasis
from nbeatsx_model import ExogenousBasisInterpretable, ExogenousBasisWavenet, ExogenousBasisTCN
from utils.ts_loader import TimeSeriesLoader
from utils.losses import MAPELoss, MASELoss, SMAPELoss, MSELoss, MAELoss, PinballLoss
from utils.metrics import mae, pinball_loss

@dataclass
class NBeatsConfig:
    """Configuration class for N-BEATS model"""
    input_size_multiplier: int = 2
    output_size: int = 24
    shared_weights: bool = False
    activation: str = 'relu'
    initialization: str = 'he_uniform'
    stack_types: List[str] = None
    n_blocks: List[int] = None
    n_layers: List[int] = None
    n_hidden: List[List[int]] = None
    n_harmonics: List[int] = None
    n_polynomials: List[int] = None
    exogenous_n_channels: int = 32
    batch_normalization: bool = False
    dropout_prob_theta: float = 0.0
    dropout_prob_exogenous: float = 0.0
    x_s_n_hidden: int = 0
    learning_rate: float = 1e-3
    lr_decay: float = 0.5
    n_lr_decay_steps: int = 9
    weight_decay: float = 0.0
    l1_theta: float = 0.0
    n_iterations: int = 1500
    early_stopping: int = 30
    loss: str = 'MAE'
    loss_hypar: float = 1.0
    val_loss: str = 'MAE'
    random_seed: int = 1
    seasonality: int = 1
    use_mixed_precision: bool = True
    scheduler_type: str = 'cosine'  # 'step', 'cosine', 'onecycle'
    gradient_clip_val: float = 1.0
    use_residual_connections: bool = False
    use_attention: bool = False
    attention_heads: int = 8

def init_weights(module: t.nn.Module, initialization: str) -> None:
    """Initialize weights with modern PyTorch methods"""
    if isinstance(module, t.nn.Linear):
        if initialization == 'orthogonal':
            t.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            t.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif initialization == 'he_normal':
            t.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        elif initialization == 'glorot_uniform':
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            t.nn.init.normal_(module.weight, 0.0, (1.0 / module.in_features) ** 0.5)
        else:
            raise ValueError(f'Initialization {initialization} not supported')

class EarlyStopping:
    """Enhanced early stopping with best model restoration"""
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 restore_best_weights: bool = True, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, score: float, model: t.nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: t.nn.Module) -> None:
        self.best_weights = copy.deepcopy(model.state_dict())

class AdaptiveGradientClipper:
    """Adaptive Gradient Clipping for stable training"""
    def __init__(self, clip_factor: float = 0.01, eps: float = 1e-3):
        self.clip_factor = clip_factor
        self.eps = eps
    
    def __call__(self, model: t.nn.Module) -> None:
        for p in model.parameters():
            if p.grad is None:
                continue
            
            p_norm = p.detach().norm().clamp_(min=self.eps)
            g_norm = p.grad.detach().norm()
            
            max_norm = self.clip_factor * p_norm
            if g_norm > max_norm:
                p.grad.detach().mul_(max_norm / (g_norm + self.eps))

class Nbeats:
    """
    Modernized N-BEATS model with enhanced features.
    
    This implementation includes:
    - Mixed precision training
    - Advanced learning rate scheduling
    - Adaptive gradient clipping
    - Enhanced early stopping
    - Residual connections (optional)
    - Attention mechanisms (optional)
    """
    
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    IDENTITY_BLOCK = 'identity'

    def __init__(self, config: Union[NBeatsConfig, Dict], device: Optional[str] = None):
        """Initialize N-BEATS model with configuration"""
        if isinstance(config, dict):
            self.config = NBeatsConfig(**config)
        else:
            self.config = config
            
        # Extract commonly used attributes for backward compatibility
        self.input_size = int(self.config.input_size_multiplier * self.config.output_size)
        self.output_size = self.config.output_size
        self.shared_weights = self.config.shared_weights
        self.activation = self.config.activation
        self.initialization = self.config.initialization
        self.stack_types = self.config.stack_types or ['trend', 'seasonality']
        self.n_blocks = self.config.n_blocks or [3, 3]
        self.n_layers = self.config.n_layers or [4, 4]
        self.n_hidden = self.config.n_hidden or [[512, 512, 512, 512], [512, 512, 512, 512]]
        self.n_harmonics = self.config.n_harmonics or [1, 1]
        self.n_polynomials = self.config.n_polynomials or [2, 2]
        self.exogenous_n_channels = self.config.exogenous_n_channels
        
        # Training parameters
        self.batch_normalization = self.config.batch_normalization
        self.dropout_prob_theta = self.config.dropout_prob_theta
        self.dropout_prob_exogenous = self.config.dropout_prob_exogenous
        self.x_s_n_hidden = self.config.x_s_n_hidden
        self.learning_rate = self.config.learning_rate
        self.lr_decay = self.config.lr_decay
        self.n_lr_decay_steps = self.config.n_lr_decay_steps
        self.weight_decay = self.config.weight_decay
        self.n_iterations = self.config.n_iterations
        self.early_stopping = self.config.early_stopping
        self.loss = self.config.loss
        self.loss_hypar = self.config.loss_hypar
        self.val_loss = self.config.val_loss
        self.l1_theta = self.config.l1_theta
        self.l1_conv = 1e-3
        self.random_seed = self.config.random_seed
        self.seasonality = self.config.seasonality
        
        # Modern features
        self.use_mixed_precision = self.config.use_mixed_precision
        self.scheduler_type = self.config.scheduler_type
        self.gradient_clip_val = self.config.gradient_clip_val
        self.use_residual_connections = self.config.use_residual_connections
        self.use_attention = self.config.use_attention
        self.attention_heads = self.config.attention_heads

        if device is None:
            device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.device = device

        self._is_instantiated = False
        self.scaler = GradScaler() if self.use_mixed_precision else None
        self.gradient_clipper = AdaptiveGradientClipper(self.gradient_clip_val * 0.01)

    def create_stack(self) -> List[t.nn.Module]:
        """Create N-BEATS stack with modern enhancements"""
        if hasattr(self, 'include_var_dict') and self.include_var_dict is not None:
            x_t_n_inputs = self.output_size * int(sum([len(x) for x in self.include_var_dict.values()]))
            
            # Correction because week_day only adds 1 no output_size
            if len(self.include_var_dict.get('week_day', [])) > 0:
                x_t_n_inputs = x_t_n_inputs - self.output_size + 1
        else:
            x_t_n_inputs = self.input_size

        block_list = []
        self.blocks_regularizer = []
        
        for i in range(len(self.stack_types)):
            for block_id in range(self.n_blocks[i]):
                # Batch norm only on first block
                batch_normalization_block = (len(block_list) == 0) and self.batch_normalization
                
                # Dummy regularizer in block. Override with 1 if exogenous_block
                self.blocks_regularizer.append(0)

                # Shared weights
                if self.shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    nbeats_block = self._create_block(
                        stack_type=self.stack_types[i],
                        stack_idx=i,
                        x_t_n_inputs=x_t_n_inputs,
                        batch_normalization_block=batch_normalization_block
                    )
                    
                    # Apply weight initialization
                    init_function = partial(init_weights, initialization=self.initialization)
                    nbeats_block.layers.apply(init_function)
                    
                block_list.append(nbeats_block)
                
        return block_list

    def _create_block(self, stack_type: str, stack_idx: int, x_t_n_inputs: int, 
                     batch_normalization_block: bool) -> t.nn.Module:
        """Create individual N-BEATS block"""
        common_args = {
            'x_t_n_inputs': x_t_n_inputs,
            'x_s_n_inputs': self.n_x_s,
            'x_s_n_hidden': self.x_s_n_hidden,
            'n_layers': self.n_layers[stack_idx],
            'theta_n_hidden': self.n_hidden[stack_idx],
            'include_var_dict': getattr(self, 'include_var_dict', None),
            't_cols': getattr(self, 't_cols', None),
            'batch_normalization': batch_normalization_block,
            'dropout_prob': self.dropout_prob_theta,
            'activation': self.activation,
            'use_residual_connections': self.use_residual_connections,
            'use_attention': self.use_attention,
            'attention_heads': self.attention_heads,
        }
        
        if stack_type == 'seasonality':
            return NBeatsBlock(
                theta_n_dim=4 * int(np.ceil(self.n_harmonics[stack_idx] / 2 * self.output_size) - (self.n_harmonics[stack_idx] - 1)),
                basis=SeasonalityBasis(
                    harmonics=self.n_harmonics[stack_idx],
                    backcast_size=self.input_size,
                    forecast_size=self.output_size
                ),
                **common_args
            )
        elif stack_type == 'trend':
            return NBeatsBlock(
                theta_n_dim=2 * (self.n_polynomials[stack_idx] + 1),
                basis=TrendBasis(
                    degree_of_polynomial=self.n_polynomials[stack_idx],
                    backcast_size=self.input_size,
                    forecast_size=self.output_size
                ),
                **common_args
            )
        elif stack_type == 'identity':
            return NBeatsBlock(
                theta_n_dim=self.input_size + self.output_size,
                basis=IdentityBasis(
                    backcast_size=self.input_size,
                    forecast_size=self.output_size
                ),
                **common_args
            )
        elif stack_type == 'exogenous':
            return NBeatsBlock(
                theta_n_dim=2 * self.n_x_t,
                basis=ExogenousBasisInterpretable(),
                **common_args
            )
        elif stack_type == 'exogenous_tcn':
            return NBeatsBlock(
                theta_n_dim=2 * self.exogenous_n_channels,
                basis=ExogenousBasisTCN(self.exogenous_n_channels, self.n_x_t),
                **common_args
            )
        elif stack_type == 'exogenous_wavenet':
            self.blocks_regularizer[-1] = 1
            return NBeatsBlock(
                theta_n_dim=2 * self.exogenous_n_channels,
                basis=ExogenousBasisWavenet(self.exogenous_n_channels, self.n_x_t),
                **common_args
            )
        else:
            raise ValueError(f'Block type {stack_type} not found!')

    def _get_scheduler(self, optimizer: t.optim.Optimizer, steps_per_epoch: int) -> t.optim.lr_scheduler._LRScheduler:
        """Get learning rate scheduler"""
        if self.scheduler_type == 'step':
            lr_decay_steps = self.n_iterations // self.n_lr_decay_steps
            lr_decay_steps = max(lr_decay_steps, 1)
            return optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=self.lr_decay)
        
        elif self.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=steps_per_epoch * 10, T_mult=1, eta_min=self.learning_rate * 0.01
            )
        
        elif self.scheduler_type == 'onecycle':
            total_steps = self.n_iterations
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.learning_rate * 3, total_steps=total_steps,
                pct_start=0.3, anneal_strategy='cos'
            )
        
        else:
            raise ValueError(f'Scheduler type {self.scheduler_type} not supported')

    def __loss_fn(self, loss_name: str):
        """Loss function factory"""
        def loss(x, loss_hypar, forecast, target, mask):
            base_loss = 0
            if loss_name == 'MAPE':
                base_loss = MAPELoss(y=target, y_hat=forecast, mask=mask)
            elif loss_name == 'MASE':
                base_loss = MASELoss(y=target, y_hat=forecast, y_insample=x, seasonality=loss_hypar, mask=mask)
            elif loss_name == 'SMAPE':
                base_loss = SMAPELoss(y=target, y_hat=forecast, mask=mask)
            elif loss_name == 'MSE':
                base_loss = MSELoss(y=target, y_hat=forecast, mask=mask)
            elif loss_name == 'MAE':
                base_loss = MAELoss(y=target, y_hat=forecast, mask=mask)
            elif loss_name == 'PINBALL':
                base_loss = PinballLoss(y=target, y_hat=forecast, mask=mask, tau=loss_hypar)
            else:
                raise ValueError(f'Unknown loss function: {loss_name}')
            
            return base_loss + self.loss_l1_conv_layers() + self.loss_l1_theta()
        return loss

    def __val_loss_fn(self, loss_name: str = 'MAE'):
        """Validation loss function factory"""
        def loss(forecast, target, weights):
            if loss_name == 'MAPE':
                return mape(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'SMAPE':
                return smape(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'MSE':
                return mse(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'RMSE':
                return rmse(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'MAE':
                return mae(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'PINBALL':
                return pinball_loss(y=target, y_hat=forecast, weights=weights, tau=0.5)
            else:
                raise ValueError(f'Unknown loss function: {loss_name}')
        return loss

    def loss_l1_conv_layers(self) -> Tensor:
        """L1 regularization for convolutional layers"""
        loss_l1 = t.tensor(0.0, device=self.device)
        for i, indicator in enumerate(self.blocks_regularizer):
            if indicator and hasattr(self.model.blocks[i].basis, 'weight'):
                loss_l1 += self.l1_conv * t.sum(t.abs(self.model.blocks[i].basis.weight))
        return loss_l1

    def loss_l1_theta(self) -> Tensor:
        """L1 regularization for theta parameters"""
        loss_l1 = t.tensor(0.0, device=self.device)
        for block in self.model.blocks:
            for layer in block.modules():
                if isinstance(layer, t.nn.Linear):
                    loss_l1 += self.l1_theta * layer.weight.abs().sum()
        return loss_l1

    def to_tensor(self, x: np.ndarray) -> Tensor:
        """Convert numpy array to tensor"""
        return t.tensor(x, dtype=t.float32, device=self.device)

    def fit(self, train_ts_loader: TimeSeriesLoader, val_ts_loader: Optional[TimeSeriesLoader] = None, 
            n_iterations: Optional[int] = None, verbose: bool = True, eval_steps: int = 1) -> None:
        """
        Fit the N-BEATS model with modern training features
        
        Args:
            train_ts_loader: Training data loader
            val_ts_loader: Validation data loader (optional)
            n_iterations: Number of training iterations (optional)
            verbose: Whether to print training progress
            eval_steps: Evaluation frequency
        """
        # Validate input size compatibility
        assert self.input_size == train_ts_loader.input_size, \
            f'Model input_size {self.input_size} != data input_size {train_ts_loader.input_size}'

        # Set random seeds
        t.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Get dataset attributes
        self.n_x_t, self.n_x_s = train_ts_loader.get_n_variables()
        self.include_var_dict = getattr(train_ts_loader, 'include_var_dict', None)
        self.t_cols = getattr(train_ts_loader, 't_cols', None)

        # Instantiate model
        if not self._is_instantiated:
            block_list = self.create_stack()
            self.model = NBeats(t.nn.ModuleList(block_list)).to(self.device)
            self._is_instantiated = True

        # Training setup
        if n_iterations is None:
            n_iterations = self.n_iterations

        # Estimate steps per epoch for scheduler
        steps_per_epoch = max(len(train_ts_loader), 1)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = self._get_scheduler(optimizer, steps_per_epoch)
        
        # Loss functions
        training_loss_fn = self.__loss_fn(self.loss)
        validation_loss_fn = self.__val_loss_fn(self.val_loss)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=self.early_stopping) if self.early_stopping > 0 else None

        print('\n' + '='*30 + ' Start fitting ' + '='*30)
        
        # Training tracking
        start_time = time.time()
        self.trajectories = {'iteration': [], 'train_loss': [], 'val_loss': []}
        self.final_insample_loss = None
        self.final_outsample_loss = None

        # Training loop
        iteration = 0
        epoch = 0
        
        while iteration < n_iterations:
            epoch += 1
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in iter(train_ts_loader):
                iteration += 1
                if iteration > n_iterations:
                    break

                self.model.train()
                
                # Parse batch
                batch_tensors = {
                    'insample_y': self.to_tensor(batch['insample_y']),
                    'insample_x': self.to_tensor(batch['insample_x']),
                    'insample_mask': self.to_tensor(batch['insample_mask']),
                    'outsample_x': self.to_tensor(batch['outsample_x']),
                    'outsample_y': self.to_tensor(batch['outsample_y']),
                    'outsample_mask': self.to_tensor(batch['outsample_mask']),
                    's_matrix': self.to_tensor(batch['s_matrix'])
                }

                # Forward pass with mixed precision
                optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with autocast():
                        forecast = self.model(
                            x_s=batch_tensors['s_matrix'],
                            insample_y=batch_tensors['insample_y'],
                            insample_x_t=batch_tensors['insample_x'],
                            outsample_x_t=batch_tensors['outsample_x'],
                            insample_mask=batch_tensors['insample_mask']
                        )
                        
                        training_loss = training_loss_fn(
                            x=batch_tensors['insample_y'],
                            loss_hypar=self.loss_hypar,
                            forecast=forecast,
                            target=batch_tensors['outsample_y'],
                            mask=batch_tensors['outsample_mask']
                        )
                    
                    # Backward pass with mixed precision
                    if not t.isnan(training_loss):
                        self.scaler.scale(training_loss).backward()
                        self.scaler.unscale_(optimizer)
                        self.gradient_clipper(self.model)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    
                else:
                    forecast = self.model(
                        x_s=batch_tensors['s_matrix'],
                        insample_y=batch_tensors['insample_y'],
                        insample_x_t=batch_tensors['insample_x'],
                        outsample_x_t=batch_tensors['outsample_x'],
                        insample_mask=batch_tensors['insample_mask']
                    )
                    
                    training_loss = training_loss_fn(
                        x=batch_tensors['insample_y'],
                        loss_hypar=self.loss_hypar,
                        forecast=forecast,
                        target=batch_tensors['outsample_y'],
                        mask=batch_tensors['outsample_mask']
                    )
                    
                    # Backward pass
                    if not t.isnan(training_loss):
                        training_loss.backward()
                        self.gradient_clipper(self.model)
                        optimizer.step()

                # Update learning rate
                if self.scheduler_type != 'step':
                    lr_scheduler.step()
                
                epoch_loss += training_loss.item() if not t.isnan(training_loss) else 0
                num_batches += 1

                # Evaluation and logging
                if iteration % eval_steps == 0 and verbose:
                    avg_epoch_loss = epoch_loss / max(num_batches, 1)
                    
                    display_string = f'Step: {iteration}, Time: {time.time()-start_time:.3f}, ' \
                                   f'Insample {self.loss}: {avg_epoch_loss:.5f}'
                    
                    self.trajectories['iteration'].append(iteration)
                    self.trajectories['train_loss'].append(avg_epoch_loss)

                    # Validation evaluation
                    if val_ts_loader is not None:
                        val_loss = self.evaluate_performance(val_ts_loader, validation_loss_fn)
                        display_string += f", Outsample {self.val_loss}: {val_loss:.5f}"
                        self.trajectories['val_loss'].append(val_loss)

                        # Early stopping check
                        if early_stopping and early_stopping(val_loss, self.model):
                            print('\n' + '-'*19 + ' Stopped training by early stopping ' + '-'*19)
                            break

                    print(display_string)

            # Step learning rate for step scheduler
            if self.scheduler_type == 'step':
                lr_scheduler.step()
            
            # Break if early stopping triggered
            if early_stopping and early_stopping.counter >= early_stopping.patience:
                break

        # Final evaluation
        if n_iterations > 0:
            self.final_insample_loss = epoch_loss / max(num_batches, 1)
            final_string = f'Step: {iteration}, Time: {time.time()-start_time:.3f}, ' \
                          f'Insample {self.loss}: {self.final_insample_loss:.5f}'
            
            if val_ts_loader is not None:
                self.final_outsample_loss = self.evaluate_performance(val_ts_loader, validation_loss_fn)
                final_string += f", Outsample {self.val_loss}: {self.final_outsample_loss:.5f}"
            
            print(final_string)
            print('='*30 + '  End fitting  ' + '='*30 + '\n')

    def predict(self, ts_loader: TimeSeriesLoader, return_decomposition: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Generate predictions using the trained model
        
        Args:
            ts_loader: Data loader for prediction
            return_decomposition: Whether to return block-wise decomposition
            
        Returns:
            Tuple of (targets, forecasts, [decomposition], masks)
        """
        self.model.eval()
        assert not ts_loader.shuffle, 'ts_loader must have shuffle as False for prediction.'

        forecasts = []
        block_forecasts = []
        outsample_ys = []
        outsample_masks = []
        
        with t.no_grad():
            for batch in iter(ts_loader):
                # Convert batch to tensors
                batch_tensors = {
                    'insample_y': self.to_tensor(batch['insample_y']),
                    'insample_x': self.to_tensor(batch['insample_x']),
                    'insample_mask': self.to_tensor(batch['insample_mask']),
                    'outsample_x': self.to_tensor(batch['outsample_x']),
                    's_matrix': self.to_tensor(batch['s_matrix'])
                }

                # Generate forecasts
                with autocast(enabled=self.use_mixed_precision):
                    forecast, block_forecast = self.model(
                        insample_y=batch_tensors['insample_y'],
                        insample_x_t=batch_tensors['insample_x'],
                        insample_mask=batch_tensors['insample_mask'],
                        outsample_x_t=batch_tensors['outsample_x'],
                        x_s=batch_tensors['s_matrix'],
                        return_decomposition=True
                    )

                # Store results
                forecasts.append(forecast.cpu().numpy())
                block_forecasts.append(block_forecast.cpu().numpy())
                outsample_ys.append(batch['outsample_y'])
                outsample_masks.append(batch['outsample_mask'])

        # Concatenate results
        forecasts = np.vstack(forecasts)
        block_forecasts = np.vstack(block_forecasts)
        outsample_ys = np.vstack(outsample_ys)
        outsample_masks = np.vstack(outsample_masks)

        self.model.train()
        
        if return_decomposition:
            return outsample_ys, forecasts, block_forecasts, outsample_masks
        else:
            return outsample_ys, forecasts, outsample_masks

    def evaluate_performance(self, ts_loader: TimeSeriesLoader, validation_loss_fn) -> float:
        """Evaluate model performance on given dataset"""
        self.model.eval()
        target, forecast, outsample_mask = self.predict(ts_loader=ts_loader)
        complete_loss = validation_loss_fn(target=target, forecast=forecast, weights=outsample_mask)
        self.model.train()
        return complete_loss

    def save(self, model_dir: str, model_id: str, state_dict: Optional[Dict] = None) -> None:
        """Save model with enhanced metadata"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if state_dict is None:
            state_dict = self.model.state_dict()

        # Save model with config and training info
        save_dict = {
            'model_state_dict': state_dict,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'trajectories': getattr(self, 'trajectories', {}),
            'final_losses': {
                'insample': getattr(self, 'final_insample_loss', None),
                'outsample': getattr(self, 'final_outsample_loss', None)
            }
        }

        model_file = os.path.join(model_dir, f"model_{model_id}.model")
        print(f'Saving model to:\n {model_file}\n')
        t.save(save_dict, model_file)

    def load(self, model_dir: str, model_id: str, load_config: bool = True) -> None:
        """Load model with enhanced metadata"""
        model_file = os.path.join(model_dir, f"model_{model_id}.model")
        path = Path(model_file)

        assert path.is_file(), f'No model_*.model file found in {model_file}!'

        print(f'Loading model from:\n {model_file}\n')

        checkpoint = t.load(model_file, map_location=self.device)
        
        # Load model state
        if not self._is_instantiated:
            # Need to instantiate model first
            if 'config' in checkpoint and load_config:
                old_config = self.config
                self.config = NBeatsConfig(**checkpoint['config'])
                # Restore device and other runtime settings
                self.config.device = old_config.device if hasattr(old_config, 'device') else self.device
                self._update_attributes_from_config()
            
            # Get dataset attributes (these should be set during fit or explicitly)
            if not hasattr(self, 'n_x_t'):
                self.n_x_t = 0
            if not hasattr(self, 'n_x_s'):
                self.n_x_s = 0
                
            block_list = self.create_stack()
            self.model = NBeats(t.nn.ModuleList(block_list)).to(self.device)
            self._is_instantiated = True
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Load training trajectories if available
        if 'trajectories' in checkpoint:
            self.trajectories = checkpoint['trajectories']
        if 'final_losses' in checkpoint:
            self.final_insample_loss = checkpoint['final_losses'].get('insample')
            self.final_outsample_loss = checkpoint['final_losses'].get('outsample')

    def _update_attributes_from_config(self) -> None:
        """Update instance attributes from config"""
        for key, value in self.config.__dict__.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if not self._is_instantiated:
            return {'status': 'Model not instantiated'}
            
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_instantiated': self._is_instantiated,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'device': self.device,
            'mixed_precision': self.use_mixed_precision,
            'stack_info': {
                'types': self.stack_types,
                'n_blocks': self.n_blocks,
                'n_layers': self.n_layers,
                'n_hidden': self.n_hidden
            }
        }
        
        if hasattr(self, 'trajectories'):
            info['training_info'] = {
                'total_iterations': len(self.trajectories.get('iteration', [])),
                'final_train_loss': self.trajectories['train_loss'][-1] if self.trajectories.get('train_loss') else None,
                'final_val_loss': self.trajectories['val_loss'][-1] if self.trajectories.get('val_loss') else None,
                'best_val_loss': min(self.trajectories['val_loss']) if self.trajectories.get('val_loss') else None
            }
        
        return info

# Additional utility functions for backward compatibility
def mape(y: np.ndarray, y_hat: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Mean Absolute Percentage Error"""
    if weights is None:
        weights = np.ones_like(y)
    mask = weights > 0
    if not np.any(mask):
        return 0.0
    return 100 * np.average(np.abs(y[mask] - y_hat[mask]) / np.abs(y[mask]), weights=weights[mask])

def smape(y: np.ndarray, y_hat: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    if weights is None:
        weights = np.ones_like(y)
    mask = weights > 0
    if not np.any(mask):
        return 0.0
    denominator = np.abs(y[mask]) + np.abs(y_hat[mask])
    denominator = np.where(denominator == 0, 1, denominator)  # Avoid division by zero
    return 200 * np.average(np.abs(y[mask] - y_hat[mask]) / denominator, weights=weights[mask])

def mse(y: np.ndarray, y_hat: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Mean Squared Error"""
    if weights is None:
        weights = np.ones_like(y)
    mask = weights > 0
    if not np.any(mask):
        return 0.0
    return np.average((y[mask] - y_hat[mask]) ** 2, weights=weights[mask])

def rmse(y: np.ndarray, y_hat: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mse(y, y_hat, weights))


# Example usage and migration helper
def create_nbeats_from_legacy_params(**kwargs) -> Nbeats:
    """Create N-BEATS model from legacy parameter format"""
    # Map legacy parameters to new config
    config_dict = {}
    
    # Direct mappings
    direct_mappings = [
        'input_size_multiplier', 'output_size', 'shared_weights', 'activation',
        'initialization', 'stack_types', 'n_blocks', 'n_layers', 'n_hidden',
        'n_harmonics', 'n_polynomials', 'exogenous_n_channels', 'batch_normalization',
        'dropout_prob_theta', 'dropout_prob_exogenous', 'x_s_n_hidden',
        'learning_rate', 'lr_decay', 'n_lr_decay_steps', 'weight_decay',
        'l1_theta', 'n_iterations', 'early_stopping', 'loss', 'loss_hypar',
        'val_loss', 'random_seed', 'seasonality'
    ]
    
    for param in direct_mappings:
        if param in kwargs:
            config_dict[param] = kwargs[param]
    
    # Set modern defaults
    config_dict.setdefault('use_mixed_precision', True)
    config_dict.setdefault('scheduler_type', 'cosine')
    config_dict.setdefault('gradient_clip_val', 1.0)
    config_dict.setdefault('use_residual_connections', False)
    config_dict.setdefault('use_attention', False)
    config_dict.setdefault('attention_heads', 8)
    
    # Handle special cases
    if 'include_var_dict' in kwargs:
        # This will be set during fit() when we have access to the data loader
        pass
    if 't_cols' in kwargs:
        # This will be set during fit() when we have access to the data loader
        pass
    
    config = NBeatsConfig(**config_dict)
    device = kwargs.get('device', None)
    
    return Nbeats(config, device)