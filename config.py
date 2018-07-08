import os

CLASS_NUM = 751

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

SKIP_CONNECTION = True

DATASET = 'Duke'

ROOT_DIR = '/home/yujheli/Project/ReID/dataset'

TRAIN_CSV_PATH = os.path.join(ROOT_DIR, DATASET, 'train_list.csv')
TEST_CSV_PATH = os.path.join(ROOT_DIR, DATASET, 'test_list.csv')

TRAIN_DATA_PATH = os.path.join(ROOT_DIR, DATASET, 'bounding_box_train')
TEST_DATA_PATH = os.path.join(ROOT_DIR, DATASET, 'bounding_box_test')
