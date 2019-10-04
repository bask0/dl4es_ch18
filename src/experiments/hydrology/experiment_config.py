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
            CS.UniformIntegerHyperparameter(
                'hidden_size', lower=0, upper=1),
            CS.UniformIntegerHyperparameter(
                'num_layers', lower=0, upper=1),
            CS.UniformIntegerHyperparameter(
                'dropout', lower=4, upper=16, q=2)
        ])

        # Optim args.
        config_space.add_hyperparameters([
            CS.CategoricalHyperparameter(
                'learning_rate', choices=[1e-2, 1e-3, 1e-4])
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
        # The name of the metric to optimize. Make shure this is part of
        # the dict returned by 'YourTrainable._train'.
        # THIS METRIC IS MINIMIZED!
        'metric': 'mse (valid)',
        # BOHB hyperband parameters (it is best to not change this):
        # - max_t: Maximum resources (in terms of epochs).
        # - Successive halving factor.
        # - num_samples (https://github.com/ray-project/ray/issues/5775)
        'max_t': 81,
        'halving_factor': 3,
        'num_samples': 81 + 27 + 9 + 6 + 5,
        # Early stopping arguments: This applies to prediction only, where
        # the best configuration from BOHB is used. After 'grace_period'
        # number of epochs, training is stopped if more than 'patience' epochs
        # have worse performance than current best, then predicitons are made.
        'grace_period': 40,
        'patience': 8,
        # Number of CPUs to use per run.
        'ncpu_per_run': 5,
        # Number of GPUs to use per run (0, 1].
        'ngpu_per_run': 0.5,
        # Number of workers per data loader.
        'num_workers': 2,
        # Batch size is not part of the hyperparaeter search as we use
        # a batch size that optimizes training performance.
        'batch_size': 32,
        # Whether to pin memory; see torch.utils.data.dataloader:Dataloader.
        'pin_memory': False,
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
