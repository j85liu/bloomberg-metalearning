import os
import time
import numpy as np
import pandas as pd
import random
import gc
import copy
from typing import Optional, List, Dict, Any, Union, Tuple
from collections import defaultdict

import torch as t
from torch import optim
from pathlib import Path
from functools import partial

from nbeats_model import (
    NBeats, NBeatsBlock, IdentityBasis, TrendBasis, SeasonalityBasis,
    ExogenousBasisInterpretable, ExogenousBasisWavenet, ExogenousBasisTCN
)
from utils.ts_loader import TimeSeriesLoader
from utils.losses import MAPELoss, MASELoss, SMAPELoss, MSELoss, MAELoss, PinballLoss
from utils.metrics import mae, pinball_loss


def init_weights(module: t.nn.Module, initialization: str) -> None:
    """
    Initialize weights of a module.
    
    Parameters
    ----------
    module : torch.nn.Module
        Module to initialize
    initialization : str
        Initialization method name
    """
    if isinstance(module, t.nn.Linear):
        if initialization == 'orthogonal':
            t.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            t.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            t.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass  # Keep default initialization
        else:
            raise ValueError(f'Initialization {initialization} not supported')


class Nbeats(object):
    """
    N-BEATS model implementation with modern PyTorch practices.
    
    Parameters
    ----------
    input_size_multiplier : int
        Multiplier to get insample size. Insample size = input_size_multiplier * output_size
    output_size : int
        Forecast horizon.
    shared_weights : bool
        If True, repeats first block.
    activation : str
        Activation function. Options: ['relu', 'softplus', 'tanh', 'selu', 'lrelu', 'prelu', 'sigmoid'].
    initialization : str
        Initialization function. Options: ['orthogonal', 'he_uniform', 'glorot_uniform', 'glorot_normal', 'lecun_normal'].
    stack_types : List[str]
        List of stack types. Subset from ['seasonality', 'trend', 'identity', 'exogenous', 'exogenous_tcn', 'exogenous_wavenet'].
    n_blocks : List[int]
        Number of blocks for each stack type. Note that len(n_blocks) = len(stack_types).
    n_layers : List[int]
        Number of layers for each stack type. Note that len(n_layers) = len(stack_types).
    n_hidden : List[List[int]]
        Structure of hidden layers for each stack type. Each internal list should contain the number of units of each hidden layer.
        Note that len(n_hidden) = len(stack_types).
    n_harmonics : List[int]
        Number of harmonic terms for each stack type. Note that len(n_harmonics) = len(stack_types).
    n_polynomials : List[int]
        Number of polynomial terms for each stack type. Note that len(n_polynomials) = len(stack_types).
    exogenous_n_channels : int
        Exogenous channels for non-interpretable exogenous basis.
    include_var_dict : Dict[str, List[int]]
        Exogenous terms to add.
    t_cols : List[str]
        Ordered list of ['y'] + X_cols + ['available_mask', 'sample_mask']. Can be taken from the dataset.
    batch_normalization : bool
        Whether perform batch normalization.
    dropout_prob_theta : float
        Float between (0, 1). Dropout for Nbeats basis.
    dropout_prob_exogenous : float
        Float between (0, 1). Dropout for exogenous basis.
    x_s_n_hidden : int
        Number of encoded static features to calculate.
    learning_rate : float
        Learning rate between (0, 1).
    lr_decay : float
        Decreasing multiplier for the learning rate.
    n_lr_decay_steps : int
        Period for each learning rate decay.
    weight_decay : float
        L2 penalty for optimizer.
    l1_theta : float
        L1 regularization for the loss function.
    n_iterations : int
        Number of training steps.
    early_stopping : int
        Early stopping iterations.
    loss : str
        Loss to optimize. Options: ['MAPE', 'MASE', 'SMAPE', 'MSE', 'MAE', 'PINBALL'].
    loss_hypar : Union[float, int]
        Hyperparameter for chosen loss.
    val_loss : str
        Validation loss. Options: ['MAPE', 'MASE', 'SMAPE', 'RMSE', 'MAE', 'PINBALL'].
    random_seed : int
        Random seed for pseudo random pytorch initializer and numpy random generator.
    seasonality : int
        Time series seasonality. Usually 7 for daily data, 12 for monthly data and 4 for weekly data.
    device : Optional[str]
        If None checks 'cuda' availability. Options: ['cuda', 'cpu'].
    """

    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    IDENTITY_BLOCK = 'identity'

    def __init__(self,
                 input_size_multiplier: int,
                 output_size: int,
                 shared_weights: bool,
                 activation: str,
                 initialization: str,
                 stack_types: List[str],
                 n_blocks: List[int],
                 n_layers: List[int],
                 n_hidden: List[List[int]],
                 n_harmonics: List[int],
                 n_polynomials: List[int],
                 exogenous_n_channels: int,
                 include_var_dict: Optional[Dict[str, List[int]]],
                 t_cols: List[str],
                 batch_normalization: bool,
                 dropout_prob_theta: float,
                 dropout_prob_exogenous: float,
                 x_s_n_hidden: int,
                 learning_rate: float,
                 lr_decay: float,
                 n_lr_decay_steps: int,
                 weight_decay: float,
                 l1_theta: float,
                 n_iterations: int,
                 early_stopping: int,
                 loss: str,
                 loss_hypar: Union[float, int],
                 val_loss: str,
                 random_seed: int,
                 seasonality: int,
                 device: Optional[str] = None):
        super(Nbeats, self).__init__()

        # Validate parameters
        self._validate_parameters(
            activation, initialization, stack_types, loss, val_loss,
            n_blocks, n_layers, n_hidden, n_harmonics, n_polynomials
        )

        if activation == 'selu':
            initialization = 'lecun_normal'

        # Architecture parameters
        self.input_size = int(input_size_multiplier * output_size)
        self.output_size = output_size
        self.shared_weights = shared_weights
        self.activation = activation
        self.initialization = initialization
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.exogenous_n_channels = exogenous_n_channels

        # Regularization and optimization parameters
        self.batch_normalization = batch_normalization
        self.dropout_prob_theta = dropout_prob_theta
        self.dropout_prob_exogenous = dropout_prob_exogenous
        self.x_s_n_hidden = x_s_n_hidden
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.n_lr_decay_steps = n_lr_decay_steps
        self.weight_decay = weight_decay
        self.n_iterations = n_iterations
        self.early_stopping = early_stopping
        self.loss = loss
        self.loss_hypar = loss_hypar
        self.val_loss = val_loss
        self.l1_theta = l1_theta
        self.l1_conv = 1e-3  # Not a hyperparameter
        self.random_seed = random_seed

        # Data parameters
        self.seasonality = seasonality
        self.include_var_dict = include_var_dict
        self.t_cols = t_cols

        # Device setup
        if device is None:
            device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.device = device

        self._is_instantiated = False

    def _validate_parameters(self, activation: str, initialization: str, stack_types: List[str],
                           loss: str, val_loss: str, n_blocks: List[int], n_layers: List[int],
                           n_hidden: List[List[int]], n_harmonics: List[int], 
                           n_polynomials: List[int]) -> None:
        """Validate input parameters."""
        valid_activations = ['relu', 'softplus', 'tanh', 'selu', 'lrelu', 'prelu', 'sigmoid']
        if activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")

        valid_initializations = ['orthogonal', 'he_uniform', 'he_normal', 'glorot_uniform', 'glorot_normal', 'lecun_normal']
        if initialization not in valid_initializations:
            raise ValueError(f"initialization must be one of {valid_initializations}")

        valid_stack_types = ['seasonality', 'trend', 'identity', 'exogenous', 'exogenous_tcn', 'exogenous_wavenet']
        for stack_type in stack_types:
            if stack_type not in valid_stack_types:
                raise ValueError(f"stack_type '{stack_type}' not supported. Valid types: {valid_stack_types}")

        valid_losses = ['MAPE', 'MASE', 'SMAPE', 'MSE', 'MAE', 'PINBALL']
        if loss not in valid_losses:
            raise ValueError(f"loss must be one of {valid_losses}")
        if val_loss not in valid_losses + ['RMSE']:
            raise ValueError(f"val_loss must be one of {valid_losses + ['RMSE']}")

        # Validate list lengths match
        if not (len(n_blocks) == len(n_layers) == len(n_hidden) == len(n_harmonics) == len(n_polynomials) == len(stack_types)):
            raise ValueError("All stack configuration lists must have the same length")

    def create_stack(self) -> List[NBeatsBlock]:
        """
        Create the stack of N-BEATS blocks.
        
        Returns
        -------
        List[NBeatsBlock]
            List of configured N-BEATS blocks
        """
        if self.include_var_dict is not None:
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
                        i, x_t_n_inputs, batch_normalization_block
                    )
                    
                    # Apply weight initialization
                    init_function = partial(init_weights, initialization=self.initialization)
                    nbeats_block.layers.apply(init_function)
                    
                block_list.append(nbeats_block)
                
        return block_list

    def _create_block(self, stack_idx: int, x_t_n_inputs: int, 
                     batch_normalization_block: bool) -> NBeatsBlock:
        """
        Create a single N-BEATS block based on stack type.
        
        Parameters
        ----------
        stack_idx : int
            Index of the stack type
        x_t_n_inputs : int
            Number of temporal inputs
        batch_normalization_block : bool
            Whether to use batch normalization
            
        Returns
        -------
        NBeatsBlock
            Configured N-BEATS block
        """
        stack_type = self.stack_types[stack_idx]
        
        if stack_type == 'seasonality':
            return NBeatsBlock(
                x_t_n_inputs=x_t_n_inputs,
                x_s_n_inputs=self.n_x_s,
                x_s_n_hidden=self.x_s_n_hidden,
                theta_n_dim=4 * int(np.ceil(self.n_harmonics[stack_idx] / 2 * self.output_size) - (self.n_harmonics[stack_idx] - 1)),
                basis=SeasonalityBasis(
                    harmonics=self.n_harmonics[stack_idx],
                    backcast_size=self.input_size,
                    forecast_size=self.output_size
                ),
                n_layers=self.n_layers[stack_idx],
                theta_n_hidden=self.n_hidden[stack_idx],
                include_var_dict=self.include_var_dict,
                t_cols=self.t_cols,
                batch_normalization=batch_normalization_block,
                dropout_prob=self.dropout_prob_theta,
                activation=self.activation
            )
        elif stack_type == 'trend':
            return NBeatsBlock(
                x_t_n_inputs=x_t_n_inputs,
                x_s_n_inputs=self.n_x_s,
                x_s_n_hidden=self.x_s_n_hidden,
                theta_n_dim=2 * (self.n_polynomials[stack_idx] + 1),
                basis=TrendBasis(
                    degree_of_polynomial=self.n_polynomials[stack_idx],
                    backcast_size=self.input_size,
                    forecast_size=self.output_size
                ),
                n_layers=self.n_layers[stack_idx],
                theta_n_hidden=self.n_hidden[stack_idx],
                include_var_dict=self.include_var_dict,
                t_cols=self.t_cols,
                batch_normalization=batch_normalization_block,
                dropout_prob=self.dropout_prob_theta,
                activation=self.activation
            )
        elif stack_type == 'identity':
            return NBeatsBlock(
                x_t_n_inputs=x_t_n_inputs,
                x_s_n_inputs=self.n_x_s,
                x_s_n_hidden=self.x_s_n_hidden,
                theta_n_dim=self.input_size + self.output_size,
                basis=IdentityBasis(
                    backcast_size=self.input_size,
                    forecast_size=self.output_size
                ),
                n_layers=self.n_layers[stack_idx],
                theta_n_hidden=self.n_hidden[stack_idx],
                include_var_dict=self.include_var_dict,
                t_cols=self.t_cols,
                batch_normalization=batch_normalization_block,
                dropout_prob=self.dropout_prob_theta,
                activation=self.activation
            )
        elif stack_type == 'exogenous':
            return NBeatsBlock(
                x_t_n_inputs=x_t_n_inputs,
                x_s_n_inputs=self.n_x_s,
                x_s_n_hidden=self.x_s_n_hidden,
                theta_n_dim=2 * self.n_x_t,
                basis=ExogenousBasisInterpretable(),
                n_layers=self.n_layers[stack_idx],
                theta_n_hidden=self.n_hidden[stack_idx],
                include_var_dict=self.include_var_dict,
                t_cols=self.t_cols,
                batch_normalization=batch_normalization_block,
                dropout_prob=self.dropout_prob_theta,
                activation=self.activation
            )
        elif stack_type == 'exogenous_tcn':
            return NBeatsBlock(
                x_t_n_inputs=x_t_n_inputs,
                x_s_n_inputs=self.n_x_s,
                x_s_n_hidden=self.x_s_n_hidden,
                theta_n_dim=2 * self.exogenous_n_channels,
                basis=ExogenousBasisTCN(self.exogenous_n_channels, self.n_x_t),
                n_layers=self.n_layers[stack_idx],
                theta_n_hidden=self.n_hidden[stack_idx],
                include_var_dict=self.include_var_dict,
                t_cols=self.t_cols,
                batch_normalization=batch_normalization_block,
                dropout_prob=self.dropout_prob_theta,
                activation=self.activation
            )
        elif stack_type == 'exogenous_wavenet':
            self.blocks_regularizer[-1] = 1  # Enable regularization for WaveNet
            return NBeatsBlock(
                x_t_n_inputs=x_t_n_inputs,
                x_s_n_inputs=self.n_x_s,
                x_s_n_hidden=self.x_s_n_hidden,
                theta_n_dim=2 * self.exogenous_n_channels,
                basis=ExogenousBasisWavenet(self.exogenous_n_channels, self.n_x_t),
                n_layers=self.n_layers[stack_idx],
                theta_n_hidden=self.n_hidden[stack_idx],
                include_var_dict=self.include_var_dict,
                t_cols=self.t_cols,
                batch_normalization=batch_normalization_block,
                dropout_prob=self.dropout_prob_theta,
                activation=self.activation
            )
        else:
            raise ValueError(f'Block type {stack_type} not supported')

    def __loss_fn(self, loss_name: str):
        """Create loss function based on loss name."""
        def loss(x, loss_hypar, forecast, target, mask):
            if loss_name == 'MAPE':
                return (MAPELoss(y=target, y_hat=forecast, mask=mask) + 
                       self.loss_l1_conv_layers() + self.loss_l1_theta())
            elif loss_name == 'MASE':
                return (MASELoss(y=target, y_hat=forecast, y_insample=x, 
                               seasonality=loss_hypar, mask=mask) + 
                       self.loss_l1_conv_layers() + self.loss_l1_theta())
            elif loss_name == 'SMAPE':
                return (SMAPELoss(y=target, y_hat=forecast, mask=mask) + 
                       self.loss_l1_conv_layers() + self.loss_l1_theta())
            elif loss_name == 'MSE':
                return (MSELoss(y=target, y_hat=forecast, mask=mask) + 
                       self.loss_l1_conv_layers() + self.loss_l1_theta())
            elif loss_name == 'MAE':
                return (MAELoss(y=target, y_hat=forecast, mask=mask) + 
                       self.loss_l1_conv_layers() + self.loss_l1_theta())
            elif loss_name == 'PINBALL':
                return (PinballLoss(y=target, y_hat=forecast, mask=mask, tau=loss_hypar) + 
                       self.loss_l1_conv_layers() + self.loss_l1_theta())
            else:
                raise ValueError(f'Unknown loss function: {loss_name}')
        return loss

    def __val_loss_fn(self, loss_name: str = 'MAE'):
        """Create validation loss function based on loss name."""
        def loss(forecast, target, weights):
            if loss_name == 'MAPE':
                from utils.metrics import mape
                return mape(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'SMAPE':
                from utils.metrics import smape
                return smape(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'MSE':
                from utils.metrics import mse
                return mse(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'RMSE':
                from utils.metrics import rmse
                return rmse(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'MAE':
                return mae(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'PINBALL':
                return pinball_loss(y=target, y_hat=forecast, weights=weights, tau=0.5)
            else:
                raise ValueError(f'Unknown loss function: {loss_name}')
        return loss

    def loss_l1_conv_layers(self) -> t.Tensor:
        """Calculate L1 regularization for convolutional layers."""
        loss_l1 = t.tensor(0.0, device=self.device)
        for i, indicator in enumerate(self.blocks_regularizer):
            if indicator and hasattr(self.model.blocks[i].basis, 'weight'):
                loss_l1 += self.l1_conv * t.sum(t.abs(self.model.blocks[i].basis.weight))
        return loss_l1

    def loss_l1_theta(self) -> t.Tensor:
        """Calculate L1 regularization for theta parameters."""
        loss_l1 = t.tensor(0.0, device=self.device)
        for block in self.model.blocks:
            for layer in block.modules():
                if isinstance(layer, t.nn.Linear):
                    loss_l1 += self.l1_theta * layer.weight.abs().sum()
        return loss_l1

    def to_tensor(self, x: np.ndarray) -> t.Tensor:
        """Convert numpy array to tensor on correct device."""
        return t.from_numpy(x).float().to(self.device)

    def fit(self, train_ts_loader: TimeSeriesLoader, val_ts_loader: Optional[TimeSeriesLoader] = None, 
            n_iterations: Optional[int] = None, verbose: bool = True, eval_steps: int = 1) -> None:
        """
        Fit the N-BEATS model.
        
        Parameters
        ----------
        train_ts_loader : TimeSeriesLoader
            Training data loader
        val_ts_loader : TimeSeriesLoader, optional
            Validation data loader
        n_iterations : int, optional
            Number of training iterations
        verbose : bool
            Whether to print training progress
        eval_steps : int
            Evaluation frequency
        """
        # Validate input size compatibility
        if self.input_size != train_ts_loader.input_size:
            raise ValueError(f'Model input_size {self.input_size} != data input_size {train_ts_loader.input_size}')

        # Set random seeds for reproducibility
        self._set_random_seeds()

        # Get dataset attributes
        self.n_x_t, self.n_x_s = train_ts_loader.get_n_variables()

        # Instantiate model if not already done
        if not self._is_instantiated:
            block_list = self.create_stack()
            self.model = NBeats(t.nn.ModuleList(block_list)).to(self.device)
            self._is_instantiated = True

        # Training setup
        if n_iterations is None:
            n_iterations = self.n_iterations

        lr_decay_steps = max(n_iterations // self.n_lr_decay_steps, 1)

        optimizer = optim.Adam(self.model.parameters(), 
                             lr=self.learning_rate, 
                             weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                               step_size=lr_decay_steps, 
                                               gamma=self.lr_decay)
        
        training_loss_fn = self.__loss_fn(self.loss)
        validation_loss_fn = self.__val_loss_fn(self.val_loss)

        if verbose:
            print('\n' + '='*30 + ' Start fitting ' + '='*30)

        start = time.time()
        self.trajectories = {'iteration': [], 'train_loss': [], 'val_loss': []}
        self.final_insample_loss = None
        self.final_outsample_loss = None

        # Training loop
        early_stopping_counter = 0
        best_val_loss = np.inf
        best_state_dict = copy.deepcopy(self.model.state_dict())
        break_flag = False
        iteration = 0
        epoch = 0

        while (iteration < n_iterations) and (not break_flag):
            epoch += 1
            for batch in iter(train_ts_loader):
                iteration += 1
                if (iteration > n_iterations) or break_flag:
                    continue

                self.model.train()
                
                # Parse batch
                insample_y = self.to_tensor(batch['insample_y'])
                insample_x = self.to_tensor(batch['insample_x'])
                insample_mask = self.to_tensor(batch['insample_mask'])
                outsample_x = self.to_tensor(batch['outsample_x'])
                outsample_y = self.to_tensor(batch['outsample_y'])
                outsample_mask = self.to_tensor(batch['outsample_mask'])
                s_matrix = self.to_tensor(batch['s_matrix'])

                optimizer.zero_grad()
                forecast = self.model(
                    x_s=s_matrix, 
                    insample_y=insample_y,
                    insample_x_t=insample_x, 
                    outsample_x_t=outsample_x,
                    insample_mask=insample_mask
                )

                training_loss = training_loss_fn(
                    x=insample_y, 
                    loss_hypar=self.loss_hypar, 
                    forecast=forecast,
                    target=outsample_y, 
                    mask=outsample_mask
                )

                # Protection against exploding gradients
                if not t.isnan(training_loss).any():
                    training_loss.backward()
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                else:
                    early_stopping_counter = self.early_stopping

                lr_scheduler.step()
                
                if (iteration % eval_steps == 0):
                    train_loss_val = training_loss.cpu().item()
                    display_string = f'Step: {iteration}, Time: {time.time()-start:.3f}, Insample {self.loss}: {train_loss_val:.5f}'
                    
                    self.trajectories['iteration'].append(iteration)
                    self.trajectories['train_loss'].append(train_loss_val)

                    if val_ts_loader is not None:
                        val_loss_val = self.evaluate_performance(
                            ts_loader=val_ts_loader,
                            validation_loss_fn=validation_loss_fn
                        )
                        display_string += f", Outsample {self.val_loss}: {val_loss_val:.5f}"
                        self.trajectories['val_loss'].append(val_loss_val)

                        if self.early_stopping:
                            if val_loss_val < best_val_loss:
                                # Save current model if it improves outsample loss
                                best_state_dict = copy.deepcopy(self.model.state_dict())
                                best_insample_loss = train_loss_val
                                early_stopping_counter = 0
                                best_val_loss = val_loss_val
                            else:
                                early_stopping_counter += 1
                            
                            if early_stopping_counter >= self.early_stopping:
                                break_flag = True

                    if verbose:
                        print(display_string)

                    self.model.train()

                if break_flag:
                    if verbose:
                        print('\n' + 19*'-' + ' Stopped training by early stopping ' + 19*'-')
                    self.model.load_state_dict(best_state_dict)
                    break

        # End of fitting
        if n_iterations > 0:
            if not break_flag:
                self.final_insample_loss = training_loss.cpu().item()
            else:
                self.final_insample_loss = best_insample_loss
                
            string = f'Step: {iteration}, Time: {time.time()-start:.3f}, Insample {self.loss}: {self.final_insample_loss:.5f}'
            
            if val_ts_loader is not None:
                self.final_outsample_loss = self.evaluate_performance(
                    ts_loader=val_ts_loader,
                    validation_loss_fn=validation_loss_fn
                )
                string += f", Outsample {self.val_loss}: {self.final_outsample_loss:.5f}"
            
            if verbose:
                print(string)
                print('='*30 + '  End fitting  ' + '='*30 + '\n')

    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        t.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Set CUDA seeds if available
        if t.cuda.is_available():
            t.cuda.manual_seed(self.random_seed)
            t.cuda.manual_seed_all(self.random_seed)

    def predict(self, ts_loader: TimeSeriesLoader, X_test: Optional[np.ndarray] = None, 
                return_decomposition: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Generate predictions using the fitted model.
        
        Parameters
        ----------
        ts_loader : TimeSeriesLoader
            Data loader for prediction
        X_test : np.ndarray, optional
            Test exogenous variables (unused)
        return_decomposition : bool
            Whether to return block-wise decomposition
            
        Returns
        -------
        Tuple containing outsample_ys, forecasts, and optionally block_forecasts and masks
        """
        self.model.eval()
        
        if ts_loader.shuffle:
            raise ValueError('ts_loader must have shuffle as False for prediction')

        forecasts = []
        block_forecasts = []
        outsample_ys = []
        outsample_masks = []
        
        with t.no_grad():
            for batch in iter(ts_loader):
                insample_y = self.to_tensor(batch['insample_y'])
                insample_x = self.to_tensor(batch['insample_x'])
                insample_mask = self.to_tensor(batch['insample_mask'])
                outsample_x = self.to_tensor(batch['outsample_x'])
                s_matrix = self.to_tensor(batch['s_matrix'])

                forecast, block_forecast = self.model(
                    insample_y=insample_y, 
                    insample_x_t=insample_x,
                    insample_mask=insample_mask, 
                    outsample_x_t=outsample_x,
                    x_s=s_matrix, 
                    return_decomposition=True
                )
                
                forecasts.append(forecast.cpu().numpy())
                block_forecasts.append(block_forecast.cpu().numpy())
                outsample_ys.append(batch['outsample_y'])
                outsample_masks.append(batch['outsample_mask'])

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
        """
        Evaluate model performance on given data loader.
        
        Parameters
        ----------
        ts_loader : TimeSeriesLoader
            Data loader for evaluation
        validation_loss_fn : callable
            Validation loss function
            
        Returns
        -------
        float
            Computed loss value
        """
        self.model.eval()

        target, forecast, outsample_mask = self.predict(ts_loader=ts_loader)
        complete_loss = validation_loss_fn(target=target, forecast=forecast, weights=outsample_mask)

        self.model.train()
        return complete_loss

    def save(self, model_dir: str, model_id: str, state_dict: Optional[Dict] = None) -> None:
        """
        Save model to disk.
        
        Parameters
        ----------
        model_dir : str
            Directory to save model
        model_id : str
            Model identifier
        state_dict : Dict, optional
            State dictionary to save
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if state_dict is None:
            state_dict = self.model.state_dict()

        model_file = os.path.join(model_dir, f"model_{model_id}.model")
        print(f'Saving model to:\n {model_file}\n')
        t.save({'model_state_dict': state_dict}, model_file)

    def load(self, model_dir: str, model_id: str) -> None:
        """
        Load model from disk.
        
        Parameters
        ----------
        model_dir : str
            Directory containing model
        model_id : str
            Model identifier
        """
        model_file = os.path.join(model_dir, f"model_{model_id}.model")
        path = Path(model_file)

        if not path.is_file():
            raise FileNotFoundError(f'No model_*.model file found at {model_file}')

        print(f'Loading model from:\n {model_file}\n')

        checkpoint = t.load(model_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)