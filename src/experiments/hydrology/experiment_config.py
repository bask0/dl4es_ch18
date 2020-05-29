import ConfigSpace as CS


def decode_config_name(config_name):
    def raise_format_error():
        msg = (
            'expected `config_name` of format '
            '`(w | n)_sm.(w | n)_perm` but got '
            f'`{config_name}`.')
        raise ValueError(msg)

    s = config_name.split('.')
    if len(s) != 2:
        raise_format_error()

    if s[0][0] == 'w':
        with_sm = True
    elif s[0][0] == 'n':
        with_sm = False
    else:
        raise_format_error()

    if s[1][0] == 'w':
        permute = True
    elif s[1][0] == 'n':
        permute = False
    else:
        raise_format_error()

    return with_sm, permute


def get_search_space(config_name):
    """Get search space.

    Parameters
    ----------
    config_name (str):
        The configuration name, one of
        - 'n_sm.n_perm': don't use sm as predictor, don't permute sequences.
        - 'n_sm.w_perm': don't use sm as predictor,       permute sequences.
        - 'w_sm.n_perm':       use sm as predictor, don't permute sequences.
        - 'w_sm.w_perm':       use sm as predictor,       permute sequences.

    Returns
    ----------
    Hyperparameter search space.

    """

    with_sm, permute = decode_config_name(config_name)

    config_space = CS.ConfigurationSpace()

    if permute:
        # Model args, non-temporal case.
        config_space.add_hyperparameters([
            # Dense hidden size.
            CS.UniformIntegerHyperparameter(
                'dense_hidden_size', lower=50, upper=600, q=50),
            # Dense number of layers.
            CS.UniformIntegerHyperparameter(
                'dense_num_layers', lower=2, upper=6, q=1),
            # Dropout probability for input.
            CS.UniformFloatHyperparameter(
                'dropout_in', lower=0.0, upper=0.5, q=0.1),
            # Dropout probability for dense layers.
            CS.UniformFloatHyperparameter(
                'dropout_linear', lower=0.0, upper=0.5, q=0.1),
            # activation function used in dense layers.
            CS.CategoricalHyperparameter(
                'dense_activation', choices=['relu', 'tanh', 'selu'])
        ])
    else:
        # Model args, temporal case.
        config_space.add_hyperparameters([
            # LSTM hidden size.
            CS.UniformIntegerHyperparameter(
                'lstm_hidden_size', lower=50, upper=300, q=50),
            # LSTM number of layers.
            CS.UniformIntegerHyperparameter(
                'lstm_num_layers', lower=1, upper=3, q=1),
            # Dense hidden size.
            CS.UniformIntegerHyperparameter(
                'dense_hidden_size', lower=50, upper=300, q=50),
            # Dense number of layers.
            CS.UniformIntegerHyperparameter(
                'dense_num_layers', lower=2, upper=6, q=1),
            # Dropout probability for input.
            CS.UniformFloatHyperparameter(
                'dropout_in', lower=0.0, upper=0.5, q=0.1),
            # Dropout probability for LSTM.
            CS.UniformFloatHyperparameter(
                'dropout_lstm', lower=0.0, upper=0.5, q=0.1),
            # Dropout probability for dense layers.
            CS.UniformFloatHyperparameter(
                'dropout_linear', lower=0.0, upper=0.5, q=0.1),
            # activation function used in dense layers.
            CS.CategoricalHyperparameter(
                'dense_activation', choices=['relu', 'tanh', 'selu'])
        ])

    # Optim args.
    config_space.add_hyperparameters([
        # The learning rate.
        CS.CategoricalHyperparameter(
            'learning_rate', choices=[1e-1, 1e-2, 1e-3, 1e-4]),
        # Weight decay (L2 regularization).
        CS.CategoricalHyperparameter(
            'weight_decay', choices=[1e-2, 1e-3, 1e-4])
    ])

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
        The configuration name, one of
        - 'n_sm.n_perm': don't use sm as predictor, don't permute sequences.
        - 'n_sm.w_perm': don't use sm as predictor,       permute sequences.
        - 'w_sm.n_perm':       use sm as predictor, don't permute sequences.
        - 'w_sm.w_perm':       use sm as predictor,       permute sequences.

    Returns
    ----------
    Dict of configurations.

    """

    with_sm, permute = decode_config_name(config_name)

    # This is the default config, all other configurations need to build upon this.
    global_config = {
        'experiment_name': config_name,
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
        'max_t': 120,
        'halving_factor': 3,
        'num_samples': 134,  # 81 + 27 + 9 + 6 + 5,
        # Early stopping arguments: This applies to model tuning only, where
        # the best configuration from BOHB is used.
        # - grace_period: Minimum number of epochs.
        # - patience: After 'grace_period' number of epochs, training is stopped if
        #   more than 'patience' epochs have worse performance than current best, than
        #   predicitons are made.
        'grace_period': 50,
        'patience': 20,
        # Number of CPUs to use per run.
        'ncpu_per_run': 10,
        # Number of GPUs to use per run (0, 1].
        'ngpu_per_run': 0.5,
        # Number of workers per data loader.
        'num_workers': 8,
        # Batch size is not part of the hyperparaeter search as we use
        # a batch size that optimizes training performance.
        'batch_size': 64,
        # The length of the training sequence, will be used to randomly subset training batch.
        'train_slice_length': 2000,
        # Number of batches per training epoch. This is equivalent to setting a logging frequency.
        'train_sample_size': 600,
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
            'snowf'
        ],
        'input_vars_static': [
            'soil_properties',
            'pft'
        ],
        'target_var': 'et',
        'time': {
            'train': [
                '1981-01-01',
                '2000-12-31'
            ],
            'eval': [
                '2000-01-01',
                '2013-12-31'
            ],
            'warmup': 5,
            'train_seq_length': 2000
        },
        # If true, the feature and target sequences are permuted randomly in unison.
        'permute': False
    }

    if with_sm:
        # Add soil moisture as predictor.
        global_config['input_vars'] += ['mrlsl_shal', 'mrlsl_deep']

    if permute:
        # Permute sequences.
        global_config['permute'] = True
        global_config['time']['warmup'] = 0

    return global_config
