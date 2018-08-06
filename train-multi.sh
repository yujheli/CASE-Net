rm -rf runs

MODEL_DIR=trained_models
NUM_WORKERS=2
GPU=1

BATCH_SIZE=12
LEARNING_RATE=0.001
MOMENTUM=0.9
WEIGHT_DECAY=0.0005

NUM_STEPS=250000
ITER_SIZE=1
EVAL_STEPS=100
IMAGE_STEPS=1000

SOURCE_DATASET=CUHK
TARGET_DATASET=CUHK

PRETRAINED_DIR=/home/yujheli/Project/ycchen/pretrained/$SOURCE_DATASET/down_1_2/multi

DECODER_PATH=$PRETRAINED_DIR/Decoder_${SOURCE_DATASET}_pretrain.pth.tar
EXTRACTOR_PATH=$PRETRAINED_DIR/Extractor_${SOURCE_DATASET}_pretrain.pth.tar
CLASSIFIER_PATH=$PRETRAINED_DIR/Classifier_${SOURCE_DATASET}_pretrain.pth.tar
DISCRIMINATOR_PATH=$PRETRAINED_DIR/Discriminator_${SOURCE_DATASET}_pretrain.pth.tar

DECODER_PATH=$PRETRAINED_DIR/Decoder_${SOURCE_DATASET}.pth.tar
EXTRACTOR_PATH=$PRETRAINED_DIR/Extractor_${SOURCE_DATASET}.pth.tar
CLASSIFIER_PATH=$PRETRAINED_DIR/Classifier_${SOURCE_DATASET}.pth.tar
D1_PATH=$PRETRAINED_DIR/D1_${SOURCE_DATASET}.pth.tar
D2_PATH=$PRETRAINED_DIR/D2_${SOURCE_DATASET}.pth.tar

RANDOM_CROP=False

REC_LOSS=True
CLS_LOSS=True
ADV_LOSS=True
DIS_LOSS=True
TRIPLET_LOSS=True

W_REC=1.0
W_CLS=10.0
W_ADV=0.01
W_DIS=0.01
W_GLOBAL=20.0
W_LOCAL=5.0

python3 train-multi.py --model-dir $MODEL_DIR \
                       --num-workers $NUM_WORKERS \
                       --source-dataset $SOURCE_DATASET \
                       --target-dataset $TARGET_DATASET \
                       --random-crop $RANDOM_CROP \
                       --batch-size $BATCH_SIZE \
                       --num-steps $NUM_STEPS \
                       --eval-steps $EVAL_STEPS \
                       --image-steps $IMAGE_STEPS \
                       --momentum $MOMENTUM \
                       --weight-decay $WEIGHT_DECAY \
                       --iter-size $ITER_SIZE \
                       --gpu $GPU \
                       --cls-loss $CLS_LOSS \
                       --rec-loss $REC_LOSS \
                       --adv-loss $ADV_LOSS \
                       --dis-loss $DIS_LOSS \
                       --triplet-loss $TRIPLET_LOSS \
                       --w-cls $W_CLS \
                       --w-rec $W_REC \
                       --w-adv $W_ADV \
                       --w-dis $W_DIS \
                       --w-global $W_GLOBAL \
                       --w-local $W_LOCAL \
                       --extractor-path $EXTRACTOR_PATH \
                       --classifier-path $CLASSIFIER_PATH \
                       --decoder-path $DECODER_PATH \
                       --D1-path $D1_PATH \
                       --D2-path $D2_PATH
