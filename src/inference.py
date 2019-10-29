import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--summary_dir',
        '-s',
        type=str,
        help='Path to summary dir containing model configuration and checkpoint.',
        default='default'
    )

    parser.add_argument(
        '--overwrite',
        '-O',
        help='Flag to overwrite existing runs (all existng runs will be lost!).',
        action='store_true'
    )

    args = parser.parse_args()

    return args


       ray.tune.run(
            Emulator,
            name=config['experiment_name'],
            resources_per_trial={
                'cpu': config['ncpu_per_run'],
                'gpu': config['ngpu_per_run']},
            num_samples=1,
            local_dir=store,
            raise_on_failed_trial=True,
            verbose=1,
            with_server=False,
            ray_auto_init=False,
            loggers=[JsonLogger, CSVLogger],
            reuse_actors=False,
            stop={'patience_counter': -1 if args.test else config['patience']}
