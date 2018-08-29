import os

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128

#MEAN = [0.485, 0.456, 0.406] # ImageNet Pre-trained Mean
#STDDEV = [0.229, 0.224, 0.225] # ImageNet Pre-trained STDDEV
MEAN = [0.5, 0.5, 0.5]
STDDEV = [0.5, 0.5, 0.5]

SKIP_CONNECTION = False

MODE = 'train'

if MODE == 'train':
    # Train class number
    DUKE_CLASS_NUM = 702
    MARKET_CLASS_NUM = 751
    MSMT_CLASS_NUM = 1041
    CUHK_CLASS_NUM = 1367
    VIPER_CLASS_NUM = 316
    CAVIAR_CLASS_NUM = 47

elif MODE == 'all':
    # ALL class number
    DUKE_CLASS_NUM = 1812
    MARKET_CLASS_NUM = 1501
    MSMT_CLASS_NUM = 4101
    CUHK_CLASS_NUM = 1467
    VIPER_CLASS_NUM = 732
    CAVIAR_CLASS_NUM = 72


D1_INPUT_CHANNEL = 2048
D1_FC_INPUT_DIM = 1024

D2_INPUT_CHANNEL = 1024
D2_FC_INPUT_DIM = 4096

GLOBAL_MARGIN = 2
LOCAL_MARGIN = 2


CURRENT_DIR = os.getcwd()

DATASET_DIR = '/home/yujheli/Project/ReID/reid_dataset'

MARKET_DATA_DIR = os.path.join(DATASET_DIR, 'Market')
DUKE_DATA_DIR = os.path.join(DATASET_DIR, 'Duke')
MSMT_DATA_DIR = os.path.join(DATASET_DIR, 'MSMT17_V1')
CUHK_DATA_DIR = os.path.join(DATASET_DIR, 'CUHK03')
VIPER_DATA_DIR = os.path.join(DATASET_DIR, 'VIPeR')
CAVIAR_DATA_DIR = os.path.join(DATASET_DIR, 'Caviar')


CSV_DIR = os.path.join(CURRENT_DIR, 'data/csv')

MARKET_CSV_DIR = os.path.join(CSV_DIR, 'Market')
DUKE_CSV_DIR = os.path.join(CSV_DIR, 'Duke')
MSMT_CSV_DIR = os.path.join(CSV_DIR, 'MSMT17_V1')
CUHK_CSV_DIR = os.path.join(CSV_DIR, 'CUHK03')
VIPER_CSV_DIR = os.path.join(CSV_DIR, 'VIPeR')
CAVIAR_CSV_DIR = os.path.join(CSV_DIR, 'Caviar')

if MODE == 'all':
    SOURCE_DATA_CSV = 'all_list.csv'
elif MODE == 'train':
    SOURCE_DATA_CSV = 'train_list_1.csv'

TRAIN_DATA_CSV = 'train_list_2.csv'
QUERY_DATA_CSV = 'query_list.csv'
TEST_DATA_CSV = 'test_list.csv'
SEMI_DATA_CSV = 'train_list_20p.csv'
