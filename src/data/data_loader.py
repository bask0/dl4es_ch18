
from torch.utils.data import Dataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        help='Configuration file (.json).',
        required=True
    )


class DataLoader(Dataset):
    def __init__(self):


if __name__ == '__main__':

    args = parse_args()

    with open('./data/data_config.json') as f:
        config = json.load(args.file)
    
    print(config)