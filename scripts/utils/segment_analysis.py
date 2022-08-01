import pandas as pd
import os

def main():
    with open('../review_data/aggregate_positive.txt','w+') as agg_fd:
        for file in os.listdir('../review_data/positive_segments'):
            with open(os.path.join('../review_data/positive_segments',file)) as fd:
                result = fd.read()
                agg_fd.write(result + "\n")



if __name__ == '__main__':
    main()
