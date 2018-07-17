import os

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MEAN = [0.485, 0.456, 0.406] # ImageNet Pre-trained Mean
STDDEV = [0.229, 0.224, 0.225] # ImageNet Pre-trained STDDEV

SKIP_CONNECTION = True

# Train class number
DUKE_CLASS_NUM = 702
MARKET_CLASS_NUM = 751
MSMT_CLASS_NUM = 1041
CUHK_CLASS_NUM = 1367

# ALL class number
# DUKE_CLASS_NUM = 1812
# MARKET_CLASS_NUM = 1501
# MSMT_CLASS_NUM = 4101
# CUHK_CLASS_NUM = 1467


CURRENT_DIR = os.getcwd()

DATASET_DIR = '/home/yujheli/Project/ReID/reid_dataset'

MARKET_DATA_DIR = os.path.join(DATASET_DIR, 'Market')
DUKE_DATA_DIR = os.path.join(DATASET_DIR, 'Duke')
MSMT_DATA_DIR = os.path.join(DATASET_DIR, 'MSMT17_V1')
CUHK_DATA_DIR = os.path.join(DATASET_DIR, 'CUHK03')


CSV_DIR = os.path.join(CURRENT_DIR, 'data/csv')

MARKET_CSV_DIR = os.path.join(CSV_DIR, 'Market')
DUKE_CSV_DIR = os.path.join(CSV_DIR, 'Duke')
MSMT_CSV_DIR = os.path.join(CSV_DIR, 'MSMT17_V1')
CUHK_CSV_DIR = os.path.join(CSV_DIR, 'CUHK03')

SOURCE_DATA_CSV = 'all_list.csv'
TRAIN_DATA_CSV = 'train_list.csv'
QUERY_DATA_CSV = 'query_list.csv'
TEST_DATA_CSV = 'test_list.csv'
