rm -rf runs

NUM_WORKERS=2
GPU=0

BATCH_SIZE=32

SOURCE_DATASET=VIPER
TARGET_DATASET=VIPER

PRETRAINED_DIR=/home/yujheli/Project/ycchen/pretrained/$SOURCE_DATASET/down_1_8/multi
PRETRAINED_DIR=trained_models
#PRETRAINED_DIR=/home/yujheli/Project/ycchen/pretrained/$SOURCE_DATASET/down_1_2/single
#DECODER_PATH=$PRETRAINED_DIR/Decoder_${SOURCE_DATASET}.pth.tar
DECODER_PATH=$PRETRAINED_DIR/Decoder_CUHK.pth.tar
#EXTRACTOR_PATH=$PRETRAINED_DIR/Extractor_${SOURCE_DATASET}.pth.tar
EXTRACTOR_PATH=$PRETRAINED_DIR/Extractor_CUHK.pth.tar
CLASSIFIER_PATH=$PRETRAINED_DIR/Classifier_${SOURCE_DATASET}.pth.tar

python3 eval.py --num-workers $NUM_WORKERS \
                --source-dataset $SOURCE_DATASET \
                --target-dataset $TARGET_DATASET \
                --gpu $GPU \
                --batch-size $BATCH_SIZE \
                --extractor-path $EXTRACTOR_PATH \
                --decoder-path $DECODER_PATH \
                #--classifier-path $CLASSIFIER_PATH \
