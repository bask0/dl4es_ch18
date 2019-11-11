from models.emulator import Emulator
from experiments.hydrology.experiment_config import get_config
import argparse
import os
import ray
import pickle
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


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

    parser.add_argument(
        '--test',
        '-T',
        help='Flag to perform a test run; only a fraction of the data is evaluated in each epoch.',
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
    model_restore_path = f'{cv_store}/{config["experiment_name"]}/model.pth'
    prediction_file = f'{store}predictions.zarr'

    if args.overwrite:
        if os.path.isdir(store):
            shutil.rmtree(store)
    else:
        if os.path.isdir(store):
            raise ValueError(
                f'The directory {store} exists. Set flag "--overwrite" '
                'if you want to overwrite runs - all existing runs will be lost!')
    os.makedirs(store)

    if os.path.isdir(prediction_file):
        shutil.rmtree(prediction_file)

    best_config = load_best_config(cv_store)
    best_config.update({
        'fold': -1,
        'hc_config': config}
    )

    config.update({
        'store': store,
        'is_test': args.test
    })

    # Inference is a single run, we can use more resources.
    best_config['hc_config']['ncpu_per_run'] = 60
    best_config['hc_config']['ngpu_per_run'] = 1
    best_config['hc_config']['num_workers'] = 20
    best_config['hc_config']['batch_size'] = 200

    print('Restoring model from: ', model_restore_path)
    e = Emulator(best_config)
    e._restore(model_restore_path)
    e._predict(prediction_file)
    #predictions.to_netcdf(os.path.join(store, 'predictions.nc'))

    # summarize_run(store)


if __name__ == '__main__':
    args = parse_args()

    ray.init(include_webui=False, object_store_memory=int(50e9))

    tune(args)

    ray.shutdown()
