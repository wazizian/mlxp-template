#!/usr/bin/env bash
LRS=(0.001 0.002 0.003 0.004 0.006 0.007 0.008 0.009)
HIDDEN_SIZES=(16 32 64 125 256)
for hidden_size in ${HIDDEN_SIZES[@]}; do
    for lr in ${LRS[@]}; do
        python launcher.py --config-name simple_classification model.hidden_sizes="[$hidden_size]" optimizer.learning_rate=$lr &
    done
    # sleep 8 hours
    sleep 28800
done
