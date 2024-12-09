#!/usr/bin/env bash
# LRS=(0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)
LRS=(0.017 0.014 0.013 0.011)
# HIDDEN_SIZES=(16 32 64 125 256)
HIDDEN_SIZES=(16)
for hidden_size in ${HIDDEN_SIZES[@]}; do
    for lr in ${LRS[@]}; do
        python launcher.py --config-name simple_classification model.hidden_sizes="[$hidden_size]" optimizer.learning_rate=$lr &
    done
    # sleep 8 hours
    # sleep 28800
done
