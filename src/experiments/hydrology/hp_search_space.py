import ConfigSpace as CS
from typing import Dict


def get_config(experiment: str, name: str) -> Dict:
    """Get a model configuration.

    Parameters
    ----------
    experiment
        Experiment name.
    name
        Configuration name.

    Returns
    ----------
    A tuple of dicts
        (hardcoded configurration, hyperparameter search space)

    """

    config_space = CS.ConfigurationSpace()

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

    return config_space
