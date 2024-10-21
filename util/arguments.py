import argparse


def args():
    parser = argparse.ArgumentParser(description='Config files name')

    parser.add_argument('--model',
                        type=str,
                        default='mlp_2',
                        help='type of model to be used')

    parser.add_argument('--data',
                        type=str,
                        default='data1',
                        help='name of the dataset to be used')

    parsed = parser.parse_args()

    return parsed