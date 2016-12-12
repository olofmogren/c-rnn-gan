#!/bin/bash

EXPERIMENT_LABEL=20161208-everything_and_end_classification
TRAIN_DIR=/home/mogren/experiments/2016-rnn-gan/$EXPERIMENT_LABEL
SETTINGS_LOG_DIR=$TRAIN_DIR/logs
HYPERPARAMS="--feed_previous --feature_matching --bidirectional_d --learning_rate 0.1 --pretraining_epochs 6 --num_layers_d 3 --random_input_scale 1.5 --generate_meta --tones_per_cell 3 --end_classification"
mkdir -p $SETTINGS_LOG_DIR

