import numpy as np
import argparse
parser = argparse.ArgumentParser()
if __name__ == '__main__':

    parser.add_argument('-n_init', type=int, default=5)
    args = parser.parse_args()
    n_warm_start = args.n_init
    print('ithu um varuthuuuu', n_warm_start)

# python /work/ws/nemo/fr_aa367-my_workspace-0/transferhpo/transfer_learning.py FSBO tidal anand_agg4 -fsbo_train 5000 -fsbo_tune 5000 -cc 3 -freeze 0