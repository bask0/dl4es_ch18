import ConfigSpace as CS


def get_search_space(config_name='default'):
    """Get search space.

    Parameters
    ----------
    config_name (str):
        The configuration name, one of ('default').

    Returns
    ----------
    Hyperparameter search space.

    """

    config_space = CS.ConfigurationSpace()

    if config_name == 'default':
        # Model args.
        config_space.add_hyperparameters([
            # LSTM hidden size.
            CS.UniformIntegerHyperparameter(
                    'hidden_size', lower=20, upper=320, q=20),
            # LSTM number of layers.
            CS.UniformIntegerHyperparameter(
                'num_layers', lower=1, upper=3, q=1),
            # Dropout probability for input.
            CS.UniformFloatHyperparameter(
                'dropout_in', lower=0.0, upper=0.7, q=0.1),
            # Dropout probability for input.
            CS.UniformFloatHyperparameter(
                'dropout_lstm', lower=0.0, upper=0.7, q=0.1),
            # Dropout probability for input.
            CS.UniformFloatHyperparameter(
                'dropout_linear', lower=0.0, upper=0.7, q=0.1)
        ])

        # Optim args.
        config_space.add_hyperparameters([
            # The learning rate.
            CS.CategoricalHyperparameter(
                'learning_rate', choices=[1e-2, 1e-3, 1e-4]),
            # Weight decay (L2 regularization).
            CS.CategoricalHyperparameter(
                'weight_decay', choices=[1e-2, 1e-3, 1e-4])
        ])
    else:
        raise ValueError(
            f'Argument `config_name`: {config_name} not a valid configuration.'
        )

    return config_space


def get_config(config_name):
    """Get hard-coded experiment configuration.

    This configuration involves
    - experiment-independent settings like number of GPUs per trial.
    - configurations for different experiments that are not parameters to
      be tuned, e.g. whether to add noise to the features.

    Parameters
    ----------
    config_name (str):
        The configuration name, one of ('default').

    Returns
    ----------
    Dict of configurations.

    """

    global_config = {
        'experiment_name': 'hydro',
        # Directory for logging etc.
        'store': '/scratch/dl_chapter14/experiments',
        # The name of the metric to MINIMIZE. Make shure this is part of
        # the dict returned by 'YourTrainable._train'.
        'metric': 'loss_eval',
        # The optimizer to use.
        'optimizer': 'Adam',
        # The loss function to use.
        'loss_fn': 'MSE',
        # BOHB hyperband parameters (it is best to not change this):
        # - max_t: Maximum resources (in terms of epochs).
        # - Successive halving factor.
        # - num_samples (https://github.com/ray-project/ray/issues/5775)
        'max_t': 100,
        'halving_factor': 3,
        'num_samples': 81 + 27 + 9 + 6 + 5,
        # Early stopping arguments: This applies to prediction only, where
        # the best configuration from BOHB is used.
        # - grace_period: Minimum number of epochs.
        # - patience: After 'grace_period' number of epochs, training is stopped if
        #   more than 'patience' epochs have worse performance than current best, than
        #   predicitons are made.
        'grace_period': 0,
        'patience': 20,
        # Number of CPUs to use per run.
        'ncpu_per_run': 16,
        # Number of GPUs to use per run (0, 1].
        'ngpu_per_run': 1.0,
        # Number of workers per data loader.
        'num_workers': 10,
        # Batch size is not part of the hyperparaeter search as we use
        # a batch size that optimizes training performance.
        'batch_size': 32,
        # Warmup period in years.
        'warmup': 5,
        # The length of the training sequence, will be used to randomly subset training batch.
        'train_slice_length': 10 * 365,
        # Number of batches per training epoch. This is equivalent to setting a logging frequency.
        'train_sample_size': None,
        # Whether to pin memory; see torch.utils.data.dataloader:Dataloader.
        'pin_memory': True,
        # Data configuration:
        'data_path': '/scratch/dl_chapter14/data/data.zarr/',
        'input_vars': [
            'ccover',
            'lai',
            'lwdown',
            'swdown',
            'psurf',
            'qair',
            'tair',
            'wind',
            'rainf',
            'snowf',
            'et',
            'mrlslfrac',
            'mrro'
        ],
        'input_vars_static': [
            'soil_properties',
            'pft'
        ],
        'target_var': 'et',
        'time': {
            'train': [
                '1950-01-01',
                '2000-12-31'
            ],
            'eval': [
                '2000-01-01',
                '2014-12-31'
            ],
            'warmup': 5,
            'train_seq_length': 2000
        }
    }
    if config_name == 'default':
        experiment_config = {
            'noise_std': 0.0,
            'shuffle': 0
        }
    else:
        raise ValueError(
            f'Argument `config_name`: {config_name} not a valid configuration.'
        )

    return {**global_config, **experiment_config}
