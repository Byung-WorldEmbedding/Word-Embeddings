#!/bin/bash

MODEL=/home/sojeong/Documents/GitHub/word/GLUE/discriminator-2400000
TOKENIZER=/home/sojeong/Documents/GitHub/word/GLUE/discriminator-2400000

for TASK in sst2 cola mrpc qqp stsb mnli qnli rte
do
    echo "===== Running $TASK ====="

    if [[ $TASK == "sst2" || $TASK == "cola" || $TASK == "mrpc" || $TASK == "rte" ]]; then
        MAX_LEN=128
        EPOCH=3
        BS=32
    elif [[ $TASK == "stsb" ]]; then
        MAX_LEN=256
        EPOCH=4
        BS=32
    elif [[ $TASK == "mnli" || $TASK == "qqp" || $TASK == "qnli" ]]; then
        MAX_LEN=256
        EPOCH=3
        BS=16
    fi

    python glue_task.py \
      --task_name $TASK \
      --model_name $MODEL \
      --tokenizer_name $TOKENIZER \
      --output_dir runs/10p_base/glue_$TASK \
      --do_train \
      --do_eval \
      --batch_size $BS \
      --learning_rate 2e-5 \
      --num_train_epochs $EPOCH \
      --warmup_steps 1000 \
      --max_length $MAX_LEN \
      --logging_steps 100
done
