BACKBONE=resnet-101

NUM_WORKERS=2
GPU=0

DATASET=Duke
BATCH_SIZE=32

METRIC=rank
RANK=1

python eval.py --num-workers $NUM_WORKERS --dataset $DATASET --batch-size $BATCH_SIZE --gpu $GPU --backbone $BACKBONE --eval-metric $METRIC --rank $RANK
