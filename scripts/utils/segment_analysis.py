import pandas as pd
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='../../review_data/positive_segments')
    parser.add_argument('--aggregate_file', type=str, default='../../review_data/aggregate_positive.txt')
    args = parser.parse_args()
    with open(args.aggregate_file, 'w+') as agg_fd:
        for file in os.listdir(args.folder_path):
            with open(os.path.join(args.folder_path, file)) as fd:
                result = fd.read()
                agg_fd.write(result + "\n")


if __name__ == '__main__':
    main()
