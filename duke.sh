MODEL_DIR=trained_models
NUM_WORKERS=2
GPU=0

BATCH_SIZE=128
LEARNING_RATE=2.5e-4
MOMENTUM=0.9
WEIGHT_DECAY=0.0005

NUM_STEPS=250000
NUM_STEPS_STOP=80000
ITER_SIZE=1
SAVE_STEPS=5000

SOURCE_DATASET=Duke
TARGET_DATASET=Market

RANDOM_CROP=False

REC_LOSS=False
CLS_LOSS=False
ADV_LOSS=False
CONTRA_LOSS=False
DIS_LOSS=True

W_REC=1.0
W_CLS=1.0
W_ADV=1.0
W_CONTRA=1.0
W_DIS=1.0

python train.py --model-dir $MODEL_DIR --num-workers $NUM_WORKERS --source-dataset $SOURCE_DATASET --target-dataset $TARGET_DATASET --random-crop $RANDOM_CROP --batch-size $BATCH_SIZE --num-steps $NUM_STEPS --num-steps-stop $NUM_STEPS_STOP --momentum $MOMENTUM --weight-decay $WEIGHT_DECAY --iter-size $ITER_SIZE --save-steps $SAVE_STEPS --gpu $GPU --rec-loss $REC_LOSS --cls-loss $CLS_LOSS --adv-loss $ADV_LOSS --contra-loss $CONTRA_LOSS --dis-loss $DIS_LOSS --w-rec $W_REC --w-cls $W_CLS --w-adv $W_ADV --w-contra $W_CONTRA --w-dis $W_DIS
