MODEL_DIR=trained_models
NUM_WORKERS=2
GPU=0

BATCH_SIZE=32
LEARNING_RATE=0.001
MOMENTUM=0.9
WEIGHT_DECAY=0.0005

NUM_STEPS=250000
NUM_STEPS_STOP=80000
ITER_SIZE=1
SAVE_STEPS=5000

SOURCE_DATASET=Market
TARGET_DATASET=Duke

#PRETRAINED_DIR=/home/yujheli/Project/ycchen/pretrained/$SOURCE_DATASET
PRETRAINED_DIR=trained_models
#DECODER_PATH=$PRETRAINED_DIR/Decoder_${SOURCE_DATASET}_pretrain.pth.tar
DECODER_PATH=$PRETRAINED_DIR/Decoder_${SOURCE_DATASET}_1000.pth.tar
EXTRACTOR_PATH=$PRETRAINED_DIR/Extractor_${SOURCE_DATASET}_1000.pth.tar
CLASSIFIER_PATH=$PRETRAINED_DIR/Classifier_${SOURCE_DATASET}_1000.pth.tar
DISCRIMINATOR_PATH=$PRETRAINED_DIR/Discriminator_${SOURCE_DATASET}_1000.pth.tar

RANDOM_CROP=False

REC_LOSS=True
CLS_LOSS=True
ADV_LOSS=True
CONTRA_LOSS=True
DIS_LOSS=True

W_REC=1.0
W_CLS=1.0
W_ADV=1.0
W_CONTRA=1.0
W_DIS=1.0

python train.py --model-dir $MODEL_DIR --num-workers $NUM_WORKERS --source-dataset $SOURCE_DATASET --target-dataset $TARGET_DATASET --random-crop $RANDOM_CROP --batch-size $BATCH_SIZE --num-steps $NUM_STEPS --num-steps-stop $NUM_STEPS_STOP --momentum $MOMENTUM --weight-decay $WEIGHT_DECAY --iter-size $ITER_SIZE --save-steps $SAVE_STEPS --gpu $GPU --rec-loss $REC_LOSS --cls-loss $CLS_LOSS --adv-loss $ADV_LOSS --contra-loss $CONTRA_LOSS --dis-loss $DIS_LOSS --w-rec $W_REC --w-cls $W_CLS --w-adv $W_ADV --w-contra $W_CONTRA --w-dis $W_DIS --decoder-path $DECODER_PATH --extractor-path $EXTRACTOR_PATH --classifier-path $CLASSIFIER_PATH --discriminator-path $DISCRIMINATOR_PATH
