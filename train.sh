MODEL_DIR=trained_models
NUM_WORKERS=2
RANDOM_CROP=False
BATCH_SIZE=4

NUM_STEPS=50
ITER_SIZE=10

SOURCE_DATASET=Duke
TARGET_DATASET=Market

REC_LOSS=True
CLS_LOSS=True
ADV_LOSS=True
CONTRA_LOSS=True

W_REC=1.0
W_CLS=1.0
W_ADV=1.0
W_CONTRA=1.0

python train.py --model-dir $MODEL_DIR --num-workers $NUM_WORKERS --source-dataset $SOURCE_DATASET --target-dataset $TARGET_DATASET --random-crop $RANDOM_CROP --batch-size $BATCH_SIZE --num-steps $NUM_STEPS --iter-size $ITER_SIZE --rec-loss $REC_LOSS --cls-loss $CLS_LOSS --adv-loss $ADV_LOSS --contra-loss $CONTRA_LOSS --w-rec $W_REC --w-cls $W_CLS --w-adv $W_ADV --w-contra $W_CONTRA
