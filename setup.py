from __future__ import print_function
import argparse

def setup(dataset_dir):
    print('Setting up configurations...')

    s = open('sample.py').read()
    s = s.replace('/home/yujheli/Project/ReID/reid_dataset', dataset_dir)

    f = open('config.py', 'w')
    f.write(s)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, help='The absolute path of the rppt directory of dataset')

    args = parser.parse_args()
    setup(args.dataset_dir)
