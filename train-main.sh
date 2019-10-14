rm -rf runs

MODEL_DIR=trained_models
NUM_WORKERS=2
GPU=1

BATCH_SIZE=2
LEARNING_RATE=0.0001
MOMENTUM=0.9
WEIGHT_DECAY=0.0005

NUM_STEPS=250000
ITER_SIZE=1
EVAL_STEPS=10000
IMAGE_STEPS=50

SOURCE_DATASET=Market
TARGET_DATASET=Market

# PRETRAINED_DIR=/home/yujheli/Project/ycchen/pretrained/$SOURCE_DATASET/down_1_2/multi
PRETRAINED_DIR=trained_models
DECODER_PATH=$PRETRAINED_DIR/Decoder_${SOURCE_DATASET}.pth.tar
#EXTRACTOR_PATH=/home/yujheli/Project/ycchen/pretrained/Market/Extractor_Market_pretrain.pth.tar
VAR_PATH=$PRETRAINED_DIR/Var_${SOURCE_DATASET}.pth.tar
ACGAN_PATH=$PRETRAINED_DIR/ACGAN_${SOURCE_DATASET}.pth.tar
EXTRACTOR_PATH=$PRETRAINED_DIR/Extractor_${SOURCE_DATASET}.pth.tar
CLASSIFIER_PATH=$PRETRAINED_DIR/Classifier_${SOURCE_DATASET}.pth.tar
DISCRIMINATOR_PATH=$PRETRAINED_DIR/D1_${SOURCE_DATASET}.pth.tar

RANDOM_CROP=False

REC_LOSS=True
CLS_LOSS=True
ADV_LOSS=True
DIS_LOSS=True
TRIPLET_LOSS=True
ACGAN_CLS_LOSS=True
ACGAN_ADV_LOSS=True
ACGAN_DIS_LOSS=True
KL_LOSS=False
GP_LOSS=False
DIFF_LOSS=False

W_REC=1.0
W_CLS=1.0
W_ADV=0.01
W_DIS=0.01
W_GLOBAL=1.0
W_LOCAL=5.0

W_ACGAN_CLS=1
W_ACGAN_ADV=1
W_ACGAN_DIS=1
W_KL=1
W_GP=1
W_DIFF=10

CUDA_LAUNCH_BLOCKING=1 python3 train-main.py --model-dir $MODEL_DIR \
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
                     --acgan-cls-loss $ACGAN_CLS_LOSS \
                     --rec-loss $REC_LOSS \
                     --adv-loss $ADV_LOSS \
                     --KL-loss $KL_LOSS \
                     --diff-loss $DIFF_LOSS \
                     --acgan-adv-loss $ACGAN_ADV_LOSS \
                     --acgan-dis-loss $ACGAN_DIS_LOSS \
                     --dis-loss $DIS_LOSS \
                     --triplet-loss $TRIPLET_LOSS \
                     --gp-loss $GP_LOSS \
                     --w-cls $W_CLS \
                     --w-acgan-cls $W_ACGAN_CLS \
                     --w-rec $W_REC \
                     --w-KL $W_KL \
                     --w-adv $W_ADV \
                     --w-acgan-adv $W_ACGAN_ADV \
                     --w-acgan-dis $W_ACGAN_DIS \
                     --w-dis $W_DIS \
                     --w-global $W_GLOBAL \
                     --w-local $W_LOCAL \
                     --w-gp $W_GP \
                     --w-diff $W_DIFF \
#                      --extractor-path $EXTRACTOR_PATH \
#                      --classifier-path $CLASSIFIER_PATH \
#                      --var-path $VAR_PATH \
#                      --discriminator-path $DISCRIMINATOR_PATH\
#                      --decoder-path $DECODER_PATH \
#                      --acgan-path $ACGAN_PATH \

