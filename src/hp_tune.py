from models.emulator import Emulator
from experiments.hydrology.experiment_config import get_config
import argparse
import ray
import os
import pickle
import shutil
from utils.summarize_runs import summarize_run

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_name',
        '-c',
        type=str,
        help='Configuration name.',
        default='default'
    )

    parser.add_argument(
        '--dl_config_file',
        type=str,
        help='Data loader configuration file.',
        default='../data/data_loader_config.json'
    )

    parser.add_argument(
        '--overwrite',
        '-O',
        help='Flag to overwrite existing runs (all existing runs will be lost!).',
        action='store_true'
    )

    args = parser.parse_args()

    return args


def load_best_config(store):
    best_config = os.path.join(store, 'summary/best_params.pkl')
    if not os.path.isfile(best_config):
        raise ValueError(
            'Tried to load best model config, file does not exist:\n'
            f'{best_config}\nRun `summarize_results.py` to create '
            'such a file.'
        )
    with open(best_config, 'rb') as f:
        config = pickle.load(f)

    return config


def tune(args):

    config = get_config(args.config_name)
    config.update({'is_tune': False})

    cv_store = f'{config["store"]}/{config["experiment_name"]}/{args.config_name}/cv/'
    store = f'{config["store"]}/{config["experiment_name"]}/{args.config_name}/pred/'
    if args.overwrite:
        if os.path.isdir(store):
            shutil.rmtree(store)
    else:
        if os.path.isdir(store):
            raise ValueError(
                f'The directory {store} exists. Set flag "--overwrite" '
                'if you want to overwrite runs - all existing runs will be lost!')
    os.makedirs(store)

    best_config = load_best_config(cv_store)
    best_config.update({'fold': -1})

    best_config.update({'hc_config': config})

    config.update({
        'store': store,
        'is_test': False
    })

    model_path = os.path.join(os.path.dirname(cv_store), 'model.pth')
    print('Restoring model from: ', model_path)
    e = Emulator(best_config)
    e._restore(model_path)
    print('Predicting...')
    predictions = e._predict()
    predictions.to_netcdf(os.path.join(store, 'predictions.nc'))

    summarize_run(store)


if __name__ == '__main__':
    args = parse_args()

    ray.init(include_webui=False, object_store_memory=int(50e9))

    tune(args)

    ray.shutdown()
