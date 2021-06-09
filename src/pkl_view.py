"""View the .pkl file
"""

import numpy as np
import argparse
import os
import pickle

def pkl_view(pkl_path):

    print('\n{} {} {}\n'.format('='*10, pkl_path, '='*10))

    with open(pkl_path, 'rb') as f:
        pkl = pickle.load(f)
        for key, value in pkl.items():
            print('{}:\t{}'.format(key, value))

    print('\n{} END {}\n'.format('='*10, '='*10))

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_pkl", type=str, required=True,
	                    default=None, help='PKL path.')
	args = parser.parse_args()

	pkl_view(args.input_pkl)
